"""
One pass over the grid for the XC and the non-local-correlation build.

GPU4PySCF's ``get_veff`` calls ``nr_rks`` and then ``nr_nlc_vxc``, and each of
them walks the grid twice -- once to build the density and once to contract the
potential -- so the AO values and their gradients are evaluated **four** times
per SCF iteration.  ``RKS.grids`` and ``RKS.nlcgrids`` are the same points
though: both default to level 3 and both are pruned with the same density, and
``bench/grids_same.py`` shows their coordinates and weights come out bit for
bit identical.  So the two builds can share the walk:

  * the density VV10 needs is the GGA part of the density the (meta-)GGA
    functional needs, so the first pass serves both;
  * VV10's potential enters the potential matrix as ``<ao_i| wv |ao_j>`` with
    the same four components the GGA part of the XC potential uses, so adding
    it into ``wv`` before the second pass costs nothing.

Two of the four AO passes and one of the two potential contractions go away.
Everything else -- the grid, the AO screening, libxc, the VV10 double sum, the
symmetry factors -- is GPU4PySCF's own code, and the arithmetic is in the same
order, so the result agrees to round-off.

Used by ``fastrsh.get_veff`` when ``fastnlc`` is also imported; anything it
does not cover (a different NLC grid, multiple density matrices, ``hermi != 1``,
more than one device) falls back to the two separate calls.
"""
import numpy as np
import cupy

from . import compat as g4pcompat
from gpu4pyscf.dft import numint as NI
from gpu4pyscf.dft import xc_deriv
from gpu4pyscf.dft.numint import eval_rho, _eval_rho2, _scale_ao, _tau_dot
from gpu4pyscf.lib.cupy_helper import (
    add_sparse, contract, take_last2d, transpose_sum)
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import num_devices


def same_grid(grids, nlcgrids):
    """Are the XC and NLC grids the same points?  Cached on the grids object."""
    if grids is nlcgrids:
        return True
    ans = getattr(nlcgrids, '_fastxcnlc_same', None)
    if ans is None:
        ans = (grids.coords.shape == nlcgrids.coords.shape
               and bool((grids.coords == nlcgrids.coords).all())
               and bool((grids.weights == nlcgrids.weights).all()))
        nlcgrids._fastxcnlc_same = ans
    return ans


def nr_rks_nlc(ni, mol, grids, xc_code, nlc_code, dms, hermi=1, verbose=None):
    """nr_rks and nr_nlc_vxc in one pass; returns (nelec, exc, vmat).

    Returns None if the case is not covered, so the caller can fall back.
    """
    if num_devices != 1 or hermi != 1 or getattr(dms, 'ndim', 3) != 2:
        return None
    xctype = ni._xc_type(xc_code)
    if xctype not in ('GGA', 'MGGA'):
        return None      # VV10 needs the density gradient, so LDA cannot fuse

    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    mo_coeff = getattr(dms, 'mo_coeff', None)
    mo_occ = getattr(dms, 'mo_occ', None)
    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    ngrids = grids.coords.shape[0]

    if mo_coeff is None:
        dm = opt.sort_orbitals(cupy.asarray(dms), axis=[0, 1])
    else:
        if mo_coeff.ndim != 2:
            return None
        mo = opt.sort_orbitals(cupy.asarray(mo_coeff), axis=[0])
        mo = cupy.asarray(mo[:, mo_occ > 0], order='C')
        mo = mo * mo_occ[mo_occ > 0] ** .5
        dm = None

    ao_deriv = 1
    nvar = 4 if xctype == 'GGA' else 5
    rho_tot = cupy.empty((nvar, ngrids))

    # ---- one pass for the density ----------------------------------------
    p1 = 0
    for ao_mask, idx, weight, _ in ni.block_loop(
            _sorted_mol, grids, nao, ao_deriv, max_memory=None):
        p0, p1 = p1, p1 + weight.size
        if dm is None:
            cpos = cupy.take(mo, idx, axis=0)
            rho_tot[:, p0:p1] = _eval_rho2(ao_mask, cpos, xctype, False)
        else:
            rho_tot[:, p0:p1] = eval_rho(
                _sorted_mol, ao_mask, take_last2d(dm, idx), xctype=xctype,
                hermi=1, with_lapl=False)
    if p1 != ngrids:                 # a block with no AOs was skipped
        return None
    t0 = log.timer_debug1('fastxcnlc: rho', *t0)

    weights = grids.weights
    den = rho_tot[0] * weights
    nelec = float(den.sum())
    exc, vxc = ni.eval_xc_eff(xc_code, rho_tot, deriv=1, xctype=xctype)[:2]
    # gpu4pyscf <=1.7.x returns exc with shape (N,1); 1.8.0 made it (N,)
    excsum = float(cupy.dot(den, cupy.asarray(exc, order='C').ravel()))
    wv = cupy.asarray(vxc, order='C')
    wv *= weights
    if xctype == 'GGA':
        wv[0] *= .5
    elif xctype == 'MGGA':
        wv[[0, 4]] *= .5
    exc = vxc = None

    # ---- the non-local correlation, on the density we already have -------
    # the same view object twice, so _vv10nlc sees one point set
    rho4 = rho_tot[:4]
    for nlc_pars, fac in ni.nlc_coeff(nlc_code):
        e, v = g4pcompat.vv10nlc(rho4, grids.coords, grids.weights,
                                  nlc_pars)
        excsum += fac * float(cupy.dot(den, e))
        wvn = xc_deriv.transform_vxc(rho4, v * fac, 'GGA', spin=0)
        wvn *= weights
        wvn[0] *= .5
        wv[:4] += wvn
        e = v = wvn = None
    t0 = log.timer_debug1('fastxcnlc: xc + nlc', *t0)
    rho_tot = den = None

    # ---- one pass for the potential matrix --------------------------------
    vmat = cupy.zeros((nao, nao))
    p1 = 0
    for ao_mask, idx, weight, _ in ni.block_loop(
            _sorted_mol, grids, nao, ao_deriv, max_memory=None):
        p0, p1 = p1, p1 + weight.size
        if xctype == 'GGA':
            aow = _scale_ao(ao_mask, wv[:, p0:p1])
            add_sparse(vmat, ao_mask[0].dot(aow.T), idx)
        else:
            vtmp = _tau_dot(ao_mask, ao_mask, wv[4, p0:p1])
            aow = _scale_ao(ao_mask, wv[:4, p0:p1])
            vtmp = contract('ig,jg->ij', ao_mask[0], aow, beta=1., out=vtmp)
            add_sparse(vmat, vtmp, idx)
    transpose_sum(vmat)
    vmat = opt.unsort_orbitals(vmat, axis=[0, 1])
    log.timer_debug1('fastxcnlc: integration', *t0)
    return nelec, excsum, vmat
