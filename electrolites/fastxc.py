"""
Faster exchange-correlation (Vxc) build for GPU4PySCF's RKS.

Importing this module monkey-patches ``gpu4pyscf.dft.numint.NumInt.nr_rks``.
Three things change; none of them touches the grid, the AO screening, the
libxc call or the symmetry factors, so the potential matrix agrees with
GPU4PySCF's to round-off.

1. **One pass over the grid instead of two.**  GPU4PySCF walks the grid once to
   build rho, calls libxc, then walks it again to contract the potential, so
   every AO and its three gradients are evaluated twice per SCF iteration.
   Here rho, libxc and the potential contraction happen inside one pass.

2. **The density is contracted with the density matrix, not the orbitals.**
   With the occupied orbitals, rho costs four GEMMs of ``nocc*nao_sub*ngrids``;
   with the density matrix it is one GEMM of ``nao_sub^2*ngrids``.  The AO
   screening keeps ``nao_sub`` near 320 on these clusters while ``nocc`` is in
   the hundreds, so the density-matrix route is several times less arithmetic.
   Which one is cheaper depends on the block's AO count and the number of
   occupied orbitals -- neither of which is the size of the molecule -- so the
   choice is made from those two numbers.

3. **The GEMMs are batched across grid blocks.**  A single block's potential
   GEMM is ``nao_sub x nao_sub`` with ``nao_sub ~ 320``, i.e. about 36 CUTLASS
   tiles for a 108-SM GPU; issued one block at a time it leaves two thirds of
   the device idle, which is why the XC build is latency- rather than
   flop-bound.  Grouping the blocks (GPU4PySCF's own ``grouped_gemm`` and
   ``grouped_dot``) fills the machine.  The group is sized by available memory,
   as in GPU4PySCF's ``grouped_block_loop``, not by the molecule.
"""
import os
import numpy as np
import cupy
from gpu4pyscf.dft import numint as NI
from gpu4pyscf.dft.numint import (
    eval_rho, _eval_rho2, _contract_rho, _contract_rho1, _scale_ao, _tau_dot)
from gpu4pyscf.lib.cupy_helper import (
    add_sparse, contract, take_last2d, transpose_sum, ndarray,
    grouped_gemm, grouped_dot)
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import num_devices

# 'grouped' batches the GEMMs across grid blocks; 'perblock' keeps GPU4PySCF's
# one-block-at-a-time issue order.  'auto' rho picks the cheaper of the
# density-matrix and orbital routes per block.
MODE = os.environ.get('FASTXC_MODE', 'grouped')
RHO = os.environ.get('FASTXC_RHO', 'auto')


def _prep(ni, mol, grids, dms):
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    mo_coeff = getattr(dms, 'mo_coeff', None)
    mo_occ = getattr(dms, 'mo_occ', None)
    if mo_coeff is not None:
        mo = opt.sort_orbitals(cupy.asarray(mo_coeff), axis=[0])
        mo = cupy.asarray(mo[:, mo_occ > 0], order='C')
        mo = mo * mo_occ[mo_occ > 0] ** .5
        # forming the density matrix once costs 2*nao^2*nocc for the whole
        # iteration, against 2*nvar*nocc*nao_sub*ngrids saved per block
        dm = mo.dot(mo.T)
    else:
        assert dms.ndim == 2
        mo = None
        dm = opt.sort_orbitals(cupy.asarray(dms), axis=[0, 1])
    return opt, dm, mo


def _rho_route(nao_sub, nocc, nvar, dm, mo):
    """Which of the two rho contractions is cheaper for this block."""
    if mo is None:
        return 'dm'
    if dm is None:
        return 'mo'
    if RHO == 'dm':
        return 'dm'
    if RHO == 'mo':
        return 'mo'
    # dm: 2*nao_sub^2*ng flops; mo: 2*nvar*nocc*nao_sub*ng
    return 'dm' if nao_sub < nvar * nocc else 'mo'


def _xc_and_weights(ni, xc_code, xctype, rho, weight, acc):
    """libxc on a whole group of blocks; returns the weighted potential."""
    den = rho[0] * weight
    acc[0] += den.sum()
    exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype)[:2]
    # gpu4pyscf <=1.7.x returns exc with shape (N,1); 1.8.0 changed it to (N,)
    acc[1] += cupy.dot(den, exc.ravel())
    wv = vxc
    wv *= weight
    if xctype == 'GGA':
        wv[0] *= .5
    elif xctype == 'MGGA':
        wv[[0, 4]] *= .5
    return wv


def _vmat_block(vmat, ao_mask, aow, idx, xctype, wv_tau=None, out=None):
    if xctype == 'LDA':
        add_sparse(vmat, ao_mask.dot(aow.T, out=out), idx)
    elif xctype == 'GGA':
        add_sparse(vmat, ao_mask[0].dot(aow.T, out=out), idx)
    else:
        vtmp = _tau_dot(ao_mask, ao_mask, wv_tau, out=out)
        vtmp = contract('ig,jg->ij', ao_mask[0], aow, beta=1., out=vtmp)
        add_sparse(vmat, vtmp, idx)


def _nr_rks_grouped(ni, mol, grids, xc_code, dms, hermi, verbose):
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    xctype = ni._xc_type(xc_code)
    opt, dm, mo = _prep(ni, mol, grids, dms)
    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    ao_deriv = 0 if xctype in ('LDA', 'HF') else 1
    nvar = 1 if xctype in ('LDA', 'HF') else 4
    nocc = 0 if mo is None else mo.shape[1]

    acc = [cupy.zeros(()), cupy.zeros(())]
    vmat = cupy.zeros((nao, nao))

    for ao_group, idx_group, w_group, _ in NI._grouped_block_loop(
            ni, _sorted_mol, grids, nao, ao_deriv):
        # ---- rho, batched over the blocks of this group -------------------
        ao0 = [a if ao_deriv == 0 else a[0] for a in ao_group]
        route = _rho_route(max(len(i) for i in idx_group), nocc, nvar, dm, mo)
        rho_group = []
        if route == 'dm':
            # dm_mask is symmetric, so grouped_gemm's A^T.B is dm_mask . ao0
            masks = [take_last2d(dm, idx) for idx in idx_group]
            c0_group = grouped_gemm(masks, ao0)
            masks = None
            for ao_mask, c0 in zip(ao_group, c0_group):
                if xctype == 'LDA':
                    rho_group.append(_contract_rho(c0, ao_mask).reshape(1, -1))
                else:
                    rho = _contract_rho1(ao_mask[:4], c0)
                    rho[1:4] *= 2
                    rho_group.append(rho)
        else:
            cpos = [cupy.take(mo, idx, axis=0) for idx in idx_group]
            if xctype == 'LDA':
                c0_group = grouped_gemm(cpos, ao0)
                for c0 in c0_group:
                    rho_group.append(_contract_rho(c0, c0).reshape(1, -1))
            else:
                cpos4, ao4 = [], []
                for ao_mask, c in zip(ao_group, cpos):
                    for i in range(4):
                        cpos4.append(c)
                        ao4.append(ao_mask[i])
                c0_group = grouped_gemm(cpos4, ao4)
                for b in range(len(ao_group)):
                    c0 = c0_group[4*b:4*b+4]
                    rho = cupy.empty((4, c0[0].shape[1]))
                    _contract_rho(c0[0], c0[0], rho=rho[0])
                    for i in range(1, 4):
                        _contract_rho(c0[0], c0[i], rho=rho[i])
                    rho[1:4] *= 2
                    rho_group.append(rho)
            cpos = None
        c0_group = None

        rho_cat = cupy.hstack(rho_group)
        rho_group = None
        w_cat = cupy.concatenate(w_group)
        wv = _xc_and_weights(ni, xc_code, xctype, rho_cat, w_cat, acc)
        rho_cat = None

        # ---- potential matrix, batched over the same blocks ---------------
        aow_group = []
        p0 = 0
        for ao_mask, w in zip(ao_group, w_group):
            p1 = p0 + w.size
            if xctype == 'LDA':
                aow_group.append(_scale_ao(ao_mask, wv[0, p0:p1]))
            else:
                aow_group.append(_scale_ao(ao_mask, wv[:, p0:p1]))
            p0 = p1
        res = grouped_dot(ao0, aow_group)
        aow_group = None
        for r, idx in zip(res, idx_group):
            add_sparse(vmat, r, idx)
        res = ao0 = None

    if xctype != 'LDA':
        transpose_sum(vmat)
    vmat = opt.unsort_orbitals(vmat, axis=[0, 1])
    log.timer_debug1('nr_rks (fastxc, grouped)', *t0)
    return float(acc[0]), float(acc[1]), vmat


def _nr_rks_perblock(ni, mol, grids, xc_code, dms, hermi, verbose):
    """Single pass, one grid block at a time (used for the ablation)."""
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    xctype = ni._xc_type(xc_code)
    opt, dm, mo = _prep(ni, mol, grids, dms)
    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    ao_deriv = 0 if xctype in ('LDA', 'HF') else 1
    nvar = 1 if xctype in ('LDA', 'HF') else 4
    nocc = 0 if mo is None else mo.shape[1]

    acc = [cupy.zeros(()), cupy.zeros(())]
    vmat = cupy.zeros((nao, nao))
    blk = NI.MIN_BLK_SIZE
    aow_buf = cupy.empty(blk * nao)
    dmm_buf = cupy.empty(nao * nao)
    vtmp_buf = cupy.empty(nao * nao)
    mo_buf = cupy.empty(nao * max(nocc, 1))
    c0_buf = cupy.empty(blk * max(nao, 2 * max(nocc, 1)))
    xc_buf = cupy.empty(blk)

    for ao_mask, idx, weight, _ in ni.block_loop(
            _sorted_mol, grids, nao, ao_deriv, max_memory=None):
        nao_sub = len(idx)
        if _rho_route(nao_sub, nocc, nvar, dm, mo) == 'dm':
            dm_mask = ndarray((nao_sub, nao_sub), buffer=dmm_buf)
            dm_mask = take_last2d(dm, idx, out=dm_mask)
            rho = eval_rho(_sorted_mol, ao_mask, dm_mask, xctype=xctype,
                           hermi=1, with_lapl=False, buf=c0_buf)
        else:
            cpos = ndarray((nao_sub, nocc), buffer=mo_buf)
            cpos = cupy.take(mo, idx, axis=0, out=cpos)
            rho = _eval_rho2(ao_mask, cpos, xctype, False, buf=c0_buf)
        if xctype in ('LDA', 'HF'):
            rho = rho.reshape(1, -1)
        wv = _xc_and_weights(ni, xc_code, xctype, rho, weight, acc)
        vtmp = ndarray((nao_sub, nao_sub), buffer=vtmp_buf)
        aow = _scale_ao(ao_mask, wv[0] if xctype == 'LDA' else wv, out=aow_buf)
        _vmat_block(vmat, ao_mask, aow, idx, xctype,
                    wv_tau=wv[4] if xctype == 'MGGA' else None, out=vtmp)

    if xctype != 'LDA':
        transpose_sum(vmat)
    vmat = opt.unsort_orbitals(vmat, axis=[0, 1])
    log.timer_debug1('nr_rks (fastxc, per block)', *t0)
    return float(acc[0]), float(acc[1]), vmat


_PATCHED = False


def _nr_rks_dispatch(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                     max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    ok = (num_devices == 1 and hermi == 1 and getattr(dms, 'ndim', 3) == 2
          and xctype in ('LDA', 'GGA'))
    if ok and MODE == 'grouped':
        return _nr_rks_grouped(ni, mol, grids, xc_code, dms, hermi, verbose)
    if ok and MODE == 'perblock':
        return _nr_rks_perblock(ni, mol, grids, xc_code, dms, hermi, verbose)
    return NI._nr_rks_orig(ni, mol, grids, xc_code, dms, relativity, hermi,
                           max_memory, verbose)


def apply_patch():
    global _PATCHED
    if not _PATCHED:
        NI._nr_rks_orig = NI.NumInt.nr_rks
        NI.NumInt.nr_rks = _nr_rks_dispatch
        _PATCHED = True


apply_patch()
