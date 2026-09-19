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

Both a restricted and an unrestricted ground state are covered.  The
unrestricted pass is the same two walks with a spin index on the density, the
potential and the contraction, and one thing that stays spin-free: VV10 is a
functional of the **total** density, so the double sum runs once on
``rho_a + rho_b`` and its potential is added to both spin channels.  That is
also what GPU4PySCF does -- ``dft/uks.py:get_veff`` adds an ``(nao, nao)``
``vnlc`` to a ``(2, nao, nao)`` ``vxc`` and lets it broadcast -- so the
fusion changes how often the grid is walked and nothing else.  It matters
more on the unrestricted side than on the restricted one for a reason that is
not about VV10: ``nr_uks`` contracts two potential matrices per block where
``nr_rks`` contracts one, so the pass the fusion removes is the more
expensive of the two there.

Anything a case does not cover (a different NLC grid, multiple density
matrices, ``hermi != 1``, LDA, a density-fitted ``get_veff``, more than one
device, or a whole-grid ``rho`` that does not fit the memory budget) falls
back to the two separate calls.
"""
import os

import numpy as np
import cupy

from . import compat as g4pcompat
from gpu4pyscf.dft import numint as NI
from gpu4pyscf.dft import rks as _rks
from gpu4pyscf.dft import uks as _uks
from gpu4pyscf.dft import xc_deriv
from gpu4pyscf.dft.numint import eval_rho, _eval_rho2, _scale_ao, _tau_dot
from gpu4pyscf.lib.cupy_helper import (
    add_sparse, contract, take_last2d, transpose_sum)
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import num_devices

#: ``FASTXCNLC_OFF=1`` hands every ``get_veff`` back to GPU4PySCF, which is
#: the ablation for the fusion itself.  It is a switch rather than "do not
#: import the module" because the fusion competes with ``fastxc``'s grouped
#: ``nr_rks``/``nr_uks`` -- both remove launch latency from the same two grid
#: passes, from opposite directions -- so the comparison worth running is
#: fusion on/off with everything else patched, in one process.
_OFF = bool(os.environ.get('FASTXCNLC_OFF'))

# On memory, because the fusion looks like it should cost some and does not.
# Both passes here hold ``rho`` and then ``wv`` for the *whole* grid --
# VV10's double sum needs every point's density before any potential can be
# formed -- where ``fastxc``'s grouped ``nr_uks`` holds one group at a time.
# But the arm this replaces holds the same two: GPU4PySCF's ``_nr_uks_task``
# allocates ``rho_tot`` and ``vxc`` at ``(2, nset, nvar, ngrids)`` each, and
# ``nr_nlc_vxc`` stacks a whole-grid ``rho`` of its own on top.  So the peak
# is stock's peak, and there is deliberately **no memory guard** on this
# path: an allocation that fails raises, ``get_veff`` catches it and hands the
# case back, and a threshold test here would only add a way to decline
# silently on a case that would have worked.  ``docs/ROUND2_1.8.1.md`` §5 is
# the round where a silent decline cost the whole win.



def _lex_order(grids):
    """Row order that sorts (x, y, z, w) lexicographically."""
    return cupy.lexsort(cupy.stack([grids.weights, grids.coords[:, 2],
                                    grids.coords[:, 1], grids.coords[:, 0]]))


def same_grid(grids, nlcgrids):
    """Are the XC and NLC grids the same *set* of points?

    Set, not sequence.  The VV10 energy and potential are sums over the grid,
    so a permutation of it is immaterial -- and 1.8.x builds ``RKS.grids`` and
    ``RKS.nlcgrids`` independently, so on anything bigger than a few atoms
    they come out with the same points in a **different order**: on the
    284-atom cluster, 277 700 of 3 396 608 rows differ elementwise while the
    two are bit-for-bit identical once lexicographically sorted.  Testing
    elementwise equality (which is what 1.7.x needed, because there the two
    builds agreed row for row) makes the fusion decline on every system that
    is not tiny, silently and while still giving the right answer -- so it
    costs performance rather than correctness, which is exactly the kind of
    thing an energy check cannot catch.

    It still has to be a permutation and not merely the same size, so the
    sort is done rather than assumed.  It runs once per pair of grid objects
    and is cached on the NLC grid, against an SCF that will call get_veff
    dozens of times.
    """
    if grids is nlcgrids:
        return True
    ans = getattr(nlcgrids, '_fastxcnlc_same', None)
    if ans is not None:
        return ans
    ans = False
    if (grids.coords.shape == nlcgrids.coords.shape
            and grids.weights.shape == nlcgrids.weights.shape):
        if (bool((grids.coords == nlcgrids.coords).all())
                and bool((grids.weights == nlcgrids.weights).all())):
            ans = True                    # same order; nothing to check
        else:
            i, j = _lex_order(grids), _lex_order(nlcgrids)
            ans = (bool((grids.coords[i] == nlcgrids.coords[j]).all())
                   and bool((grids.weights[i] == nlcgrids.weights[j]).all()))
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


def nr_uks_nlc(ni, mol, grids, xc_code, nlc_code, dms, hermi=1, verbose=None):
    """nr_uks and nr_nlc_vxc in one pass; returns (nelec, exc, vmat).

    ``nelec`` is ``nr_uks``'s two-element array rather than ``nr_nlc_vxc``'s
    scalar, which is the one GPU4PySCF's ``get_veff`` keeps for its own debug
    log (it overwrites the first with the second and uses neither).

    Returns None if the case is not covered, so the caller can fall back.
    """
    if num_devices != 1 or hermi != 1:
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
    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    ngrids = grids.coords.shape[0]
    nvar = 4 if xctype == 'GGA' else 5

    dm = cupy.asarray(dms)
    if dm.ndim != 3 or dm.shape[0] != 2:
        return None
    dm = opt.sort_orbitals(dm, axis=[1, 2])

    ao_deriv = 1
    rho = cupy.empty((2, nvar, ngrids))

    # ---- one pass for both spin densities --------------------------------
    p1 = 0
    for ao_mask, idx, weight, _ in ni.block_loop(
            _sorted_mol, grids, nao, ao_deriv, max_memory=None):
        p0, p1 = p1, p1 + weight.size
        for s in range(2):
            rho[s, :, p0:p1] = eval_rho(
                _sorted_mol, ao_mask, take_last2d(dm[s], idx), xctype=xctype,
                hermi=1, with_lapl=False)
    if p1 != ngrids:                 # a block with no AOs was skipped
        return None
    t0 = log.timer_debug1('fastxcnlc: rho (uks)', *t0)

    weights = grids.weights
    den = rho[:, 0] * weights
    nelec = np.array([float(den[0].sum()), float(den[1].sum())])
    exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=1)[:2]
    # gpu4pyscf <=1.7.x returns exc with shape (N,1); 1.8.0 made it (N,)
    dent = den[0] + den[1]
    excsum = float(cupy.dot(dent, cupy.asarray(exc, order='C').ravel()))
    wv = cupy.asarray(vxc, order='C')
    wv *= weights
    if xctype == 'GGA':
        wv[:, 0] *= .5
    else:
        wv[:, [0, 4]] *= .5
    exc = vxc = None

    # ---- the non-local correlation, on the total density we already have --
    # VV10 sees one density, so its potential enters both spin channels
    # unchanged -- which is what `dft/uks.py` gets by adding an (nao, nao)
    # `vnlc` to a (2, nao, nao) `vxc` and letting it broadcast.
    rho4 = rho[0, :4] + rho[1, :4]
    for nlc_pars, fac in ni.nlc_coeff(nlc_code):
        e, v = g4pcompat.vv10nlc(rho4, grids.coords, grids.weights, nlc_pars)
        excsum += fac * float(cupy.dot(rho4[0] * weights, e))
        wvn = xc_deriv.transform_vxc(rho4, v * fac, 'GGA', spin=0)
        wvn *= weights
        wvn[0] *= .5
        wv[0, :4] += wvn
        wv[1, :4] += wvn
        e = v = wvn = None
    t0 = log.timer_debug1('fastxcnlc: xc + nlc (uks)', *t0)
    rho = rho4 = den = dent = None

    # ---- one pass for the two potential matrices --------------------------
    vmat = cupy.zeros((2, nao, nao))
    p1 = 0
    for ao_mask, idx, weight, _ in ni.block_loop(
            _sorted_mol, grids, nao, ao_deriv, max_memory=None):
        p0, p1 = p1, p1 + weight.size
        for s in range(2):
            if xctype == 'GGA':
                aow = _scale_ao(ao_mask, wv[s, :, p0:p1])
                add_sparse(vmat[s], ao_mask[0].dot(aow.T), idx)
            else:
                vtmp = _tau_dot(ao_mask, ao_mask, wv[s, 4, p0:p1])
                aow = _scale_ao(ao_mask, wv[s, :4, p0:p1])
                vtmp = contract('ig,jg->ij', ao_mask[0], aow, beta=1.,
                                out=vtmp)
                add_sparse(vmat[s], vtmp, idx)
    for s in range(2):
        transpose_sum(vmat[s])
    vmat = opt.unsort_orbitals(vmat, axis=[1, 2])
    log.timer_debug1('fastxcnlc: integration (uks)', *t0)
    return nelec, excsum, vmat


# ---------------------------------------------------------------------------
# The patch point.
#
# On 1.7.x this module was driven by ``fastrsh.get_veff``, which replaced
# ``RKS.get_veff`` for the fused range-separated exchange build.  1.8.0 does
# that fusion itself, so ``fastrsh`` stands down there -- and the grid fusion
# went down with it, because ``dft.rks.get_veff`` still calls ``nr_rks`` and
# ``nr_nlc_vxc`` separately and the fusion needs a patch point above both.
#
# This puts it back without carrying a copy of ``get_veff``.  The XC and NLC
# potentials are built here, in one pass, and then GPU4PySCF's own
# ``get_veff`` runs unchanged with ``nr_rks`` temporarily answering with the
# fused result and ``do_nlc`` answering False -- so the Coulomb build, the
# exchange build, the incremental ``ecoul`` bookkeeping and the returned
# ``tag_array`` are upstream's code and stay upstream's code.
# ---------------------------------------------------------------------------
_PATCHED = False
_ORIG_GET_VEFF = None
_ORIG_GET_VEFF_U = None
_ORIG_GET_VEFF_DF = None

#: How many `get_veff` calls took the fused pass and how many handed the case
#: back, counted separately for a restricted and an unrestricted ground
#: state.  A module that is allowed to decline returns the right answer
#: whether or not it did anything, so a correctness test cannot tell the two
#: apart -- which is how an elementwise grid comparison hid behind a 1e-13
#: energy agreement while declining on every system bigger than a few atoms.
#: Assert on this, not just on the energy.
STATS = {'fused': 0, 'declined': 0, 'fused_u': 0, 'declined_u': 0}


def stats():
    """``{'fused': n, 'declined': m, ...}`` since import.  See STATS."""
    return dict(STATS)


def _fused_xc_nlc(ks, mol, dm, hermi, allow_df=False):
    """(nelec, exc, vxc) from one grid pass, or None if this case is not it."""
    if _OFF or num_devices != 1 or hermi != 1:
        return None
    if (not allow_df and getattr(ks, 'with_df', None) is not None
            and not getattr(ks, 'only_dfj', False)):
        # density fitting has its own `get_veff`; `get_veff_df` is the wrapper
        # that is on it, and it is the only caller that passes allow_df.
        #
        # `only_dfj` is the exception, and it has to be named here or a RIJONX
        # wB97M-V run loses the fused grid pass: `fastdfj` hands such an object
        # to `RKS.get_veff` on purpose (its exchange is the four-centre build,
        # so it wants the incremental Fock that `_DFHF.get_veff` refuses), and
        # that is this function's restricted caller.  `_DFHF.get_veff` will
        # not run for it, so there is nothing to double.  Without `fastdfj`
        # loaded the clause is unreachable -- `_DFHF.get_veff` precedes
        # `RKS.get_veff` in a density-fitted object's method resolution order.
        return None
    if not getattr(ks, 'do_nlc', lambda: False)():
        return None                       # nothing to fuse with
    dm = cupy.asarray(dm) if not hasattr(dm, 'ndim') else dm
    if dm.ndim != 2:
        return None                       # unrestricted, or several densities
    ni = ks._numint
    if not same_grid(ks.grids, ks.nlcgrids):
        return None
    if ni.libxc.is_nlc(ks.xc):
        nlc_code = ks.xc
    elif ni.libxc.is_nlc(ks.nlc):
        nlc_code = ks.nlc
    else:
        return None
    return nr_rks_nlc(ni, mol, ks.grids, ks.xc, nlc_code, dm, hermi=hermi)


def get_veff(ks, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
    """GPU4PySCF's get_veff with the XC and NLC builds sharing one grid pass."""
    if mol is None:
        mol = ks.mol
    if dm is None:
        dm = ks.make_rdm1()
    _rks.initialize_grids(ks, mol, dm)      # get_veff's own first step
    fused = None
    try:
        fused = _fused_xc_nlc(ks, mol, dm, hermi)
    except Exception:                        # never fail closed
        fused = None
    STATS['fused' if fused is not None else 'declined'] += 1
    if fused is None:
        return _ORIG_GET_VEFF(ks, mol, dm, dm_last, vhf_last, hermi)

    ni = ks._numint
    had_nr_rks = 'nr_rks' in ni.__dict__
    old_nr_rks = ni.__dict__.get('nr_rks')
    had_do_nlc = 'do_nlc' in ks.__dict__
    old_do_nlc = ks.__dict__.get('do_nlc')
    try:
        ni.nr_rks = lambda *args, **kwargs: fused
        ks.do_nlc = lambda: False
        return _ORIG_GET_VEFF(ks, mol, dm, dm_last, vhf_last, hermi)
    finally:
        if had_nr_rks:
            ni.__dict__['nr_rks'] = old_nr_rks
        else:
            ni.__dict__.pop('nr_rks', None)
        if had_do_nlc:
            ks.__dict__['do_nlc'] = old_do_nlc
        else:
            ks.__dict__.pop('do_nlc', None)


def _fused_xc_nlc_u(ks, mol, dm, hermi, allow_df=False):
    """The unrestricted twin of ``_fused_xc_nlc``."""
    if _OFF or num_devices != 1 or hermi != 1:
        return None
    if not allow_df and getattr(ks, 'with_df', None) is not None:
        return None                       # see `_fused_xc_nlc`
    if not getattr(ks, 'do_nlc', lambda: False)():
        return None                       # nothing to fuse with
    dm = cupy.asarray(dm)
    if dm.ndim != 3 or dm.shape[0] != 2:
        return None
    ni = ks._numint
    if not same_grid(ks.grids, ks.nlcgrids):
        return None
    if ni.libxc.is_nlc(ks.xc):
        nlc_code = ks.xc
    elif ni.libxc.is_nlc(ks.nlc):
        nlc_code = ks.nlc
    else:
        return None
    return nr_uks_nlc(ni, mol, ks.grids, ks.xc, nlc_code, dm, hermi=hermi)


def get_veff_u(ks, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
    """``dft.uks.get_veff`` with the XC and NLC builds sharing one grid pass.

    Same shape as the restricted one: the fused build runs first, then
    GPU4PySCF's own ``get_veff`` runs unchanged with ``nr_uks`` temporarily
    answering with the result and ``do_nlc`` answering False.  ``dft/uks.py``
    does its own density reshaping (a 2D ``dm`` becomes two halves) and its
    own ``initialize_grids``, so both are done here before the decision --
    otherwise the fused pass would be asked about a density and a grid that
    are not the ones the call will use.
    """
    if mol is None:
        mol = ks.mol
    if dm is None:
        dm = ks.make_rdm1()
    fused = None
    try:
        # the same two steps `dft/uks.py:get_veff` takes before it does
        # anything: a restricted density becomes two halves, and the grids are
        # built (and pruned) on the total.  Both are done on a *local* copy --
        # what is handed to the original below is the caller's own object, so
        # that a case this declines is not a case this touched.
        dm3 = cupy.asarray(dm)
        if dm3.ndim == 2:
            dm3 = cupy.repeat(dm3[None] * .5, 2, axis=0)
        if ks.grids.coords is None:
            _rks.initialize_grids(ks, mol, dm3[0] + dm3[1])
        fused = _fused_xc_nlc_u(ks, mol, dm3, hermi)
    except Exception:                        # never fail closed
        fused = None
    STATS['fused_u' if fused is not None else 'declined_u'] += 1
    if fused is None:
        return _ORIG_GET_VEFF_U(ks, mol, dm, dm_last, vhf_last, hermi)

    ni = ks._numint
    had_nr_uks = 'nr_uks' in ni.__dict__
    old_nr_uks = ni.__dict__.get('nr_uks')
    had_do_nlc = 'do_nlc' in ks.__dict__
    old_do_nlc = ks.__dict__.get('do_nlc')
    try:
        ni.nr_uks = lambda *args, **kwargs: fused
        ks.do_nlc = lambda: False
        return _ORIG_GET_VEFF_U(ks, mol, dm, dm_last, vhf_last, hermi)
    finally:
        if had_nr_uks:
            ni.__dict__['nr_uks'] = old_nr_uks
        else:
            ni.__dict__.pop('nr_uks', None)
        if had_do_nlc:
            ks.__dict__['do_nlc'] = old_do_nlc
        else:
            ks.__dict__.pop('do_nlc', None)


def get_veff_df(ks, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
    """``_DFHF.get_veff`` with the XC and NLC builds sharing one grid pass.

    ``density_fit()`` builds the object's class as
    ``lib.set_class(mf, (_DFHF, mf.__class__))``, so ``_DFHF.get_veff``
    (``df/df_jk.py:270``) comes *before* ``RKS.get_veff`` and ``UKS.get_veff``
    in the method resolution order and **neither of the two wrappers above is
    on a density-fitted run's path at all** -- `benchmarks/coverage_probe.py
    --density-fit` reads ``(-)`` on that row for every functional.  A DF
    wB97M-V or B97M-V calculation therefore walked the grid twice per
    iteration where a direct-SCF one walks it once.  Found while measuring
    `fastdf`; ``docs/README_fastdf.md`` §5 has the account.

    The body is the same as the other two wrappers: compute the fused pass,
    then let upstream's own ``get_veff`` run with ``nr_rks``/``nr_uks``
    answering from it, so the DF Coulomb and exchange builds, the ``ecoul``
    bookkeeping and the returned ``tag_array`` stay upstream's code.
    """
    if mol is None:
        mol = ks.mol
    if dm is None:
        dm = ks.make_rdm1()
    fused = None
    unrestricted = getattr(dm, 'ndim', 2) == 3
    if _usable_df(ks, dm):
        try:
            # `_DFHF.get_veff`'s own first step, so the grids exist before the
            # fused pass reads them
            _rks.initialize_grids(ks, mol, dm[0] + dm[1] if unrestricted
                                  else dm)
            fn = _fused_xc_nlc_u if unrestricted else _fused_xc_nlc
            fused = fn(ks, mol, dm, hermi, allow_df=True)
        except Exception:                    # never fail closed
            fused = None
    key = 'fused_u' if unrestricted else 'fused'
    STATS[key if fused is not None else ('declined_u' if unrestricted
                                         else 'declined')] += 1
    if fused is None:
        return _ORIG_GET_VEFF_DF(ks, mol, dm, dm_last, vhf_last, hermi)

    ni = ks._numint
    name = 'nr_uks' if unrestricted else 'nr_rks'
    had = name in ni.__dict__
    old = ni.__dict__.get(name)
    had_do_nlc = 'do_nlc' in ks.__dict__
    old_do_nlc = ks.__dict__.get('do_nlc')
    try:
        setattr(ni, name, lambda *args, **kwargs: fused)
        ks.do_nlc = lambda: False
        return _ORIG_GET_VEFF_DF(ks, mol, dm, dm_last, vhf_last, hermi)
    finally:
        if had:
            ni.__dict__[name] = old
        else:
            ni.__dict__.pop(name, None)
        if had_do_nlc:
            ks.__dict__['do_nlc'] = old_do_nlc
        else:
            ks.__dict__.pop('do_nlc', None)


def _usable_df(ks, dm):
    """Only the two branches of ``_DFHF.get_veff`` this wrapper reproduces.

    ``_DFHF.get_veff`` also serves ROHF -- where it *rewrites* the density
    into a two-row array before calling ``nr_uks``, so a fused pass computed
    on the density handed in would answer for a different matrix -- and GHF,
    which calls ``ni.get_vxc`` rather than either of the two names this
    wrapper redirects.  Both fall through.
    """
    from gpu4pyscf.scf import rohf as _rohf, ghf as _ghf
    if isinstance(ks, (_rohf.ROHF, _ghf.GHF)):
        return False
    ndim = getattr(dm, 'ndim', 2)
    return ndim in (2, 3)


def apply_patch():
    """Fuse the XC and NLC grid passes on 1.8.x, where fastrsh stands down."""
    global _PATCHED, _ORIG_GET_VEFF, _ORIG_GET_VEFF_U, _ORIG_GET_VEFF_DF
    if _PATCHED:
        return
    if not g4pcompat.NEW_JK_ABI:
        # On 1.7.x fastrsh owns RKS.get_veff and already routes through this
        # module; patching here as well would fight it.
        _PATCHED = True
        return
    _ORIG_GET_VEFF = _rks.RKS.get_veff
    # both the method and the module-level name, which are the same function:
    # `dft/rkspu.py` calls `rks.get_veff` directly (tests/test_patch_aliases.py)
    _rks.RKS.get_veff = _rks.get_veff = get_veff
    # `dft/uks.py` binds its own `get_veff`, both as a module-level function
    # and as `UKS.get_veff`; patching `dft.rks`'s reaches neither, which is
    # why an open-shell wB97M-V walked the grid four times per iteration with
    # this module loaded.  `dft/ukspu.py` calls the module-level name.
    _ORIG_GET_VEFF_U = _uks.UKS.get_veff
    _uks.UKS.get_veff = _uks.get_veff = get_veff_u
    # and the density-fitted class, which overrides both of the above.
    # Guarded: a GPU4PySCF without `df` would otherwise take the whole module
    # down over an entry point that only a `density_fit()` run reaches.
    try:
        from gpu4pyscf.df import df_jk as _dfjk
    except ImportError:
        _PATCHED = True
        return
    _ORIG_GET_VEFF_DF = _dfjk._DFHF.get_veff
    # the original stays on this module, as the other two do: putting it on
    # `_DFHF` would make it an attribute of the class the patch replaced and
    # `tests/test_patch_aliases.py` would read it as a stale alias
    _dfjk._DFHF.get_veff = get_veff_df
    _PATCHED = True


apply_patch()
