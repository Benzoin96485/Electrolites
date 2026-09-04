"""A faster XC gradient (``grad.rks.get_exc``) for GPU4PySCF.

The XC part of an RKS gradient contracts, per grid block,

    exc1[i,n] = sum_j (nabla_n phi_i | v_xc | phi_j) D_ij

and GPU4PySCF forms the three ``nao_sub x nao_sub`` matrices in the middle
before contracting them with the density.  Two changes:

1. **The density contraction comes first**, which makes those matrices
   unnecessary:

       sum_j (sum_g a_n[i,g] b[j,g]) D_ij  =  sum_g a_n[i,g] (D b)[i,g]

   so one ``nao_sub^2 x ngrids`` GEMM and a row-wise reduction replace three
   ``nao_sub^2 x ngrids`` GEMMs, twice over (the ``nabla phi_i . phi_j`` term
   and the ``phi_i . nabla phi_j`` term).  The second of the two needs
   ``D phi``, which the density evaluation has already computed.  Per grid
   block that is 2 GEMMs where GPU4PySCF issues 10 (GGA) and 4 where it
   issues 19 (meta-GGA: its ``_tau_grad_dot_`` alone is nine ``nao_sub^2 x
   ngrids`` GEMMs, and every one of them is a matrix this formulation never
   forms).

   Meta-GGA needs no GEMM at all for the potential side.  ``D . sum_m ao[m]
   wv[m] = sum_m wv[m] (D . ao[m])`` because the weights are diagonal in the
   grid point, and meta-GGA has already formed all four ``D . ao[m]`` for tau,
   so what is a GEMM for GGA is an elementwise scale here.

   The density itself goes through the density matrix rather than the occupied
   orbitals, for the reason ``fastxc`` gives for the Fock build: with the
   orbitals rho costs four GEMMs of ``nocc*nao_sub*ngrids``, with the density
   matrix one of ``nao_sub^2*ngrids``, and AO screening keeps ``nao_sub`` far
   below ``4*nocc`` on these clusters.  For meta-GGA the density matrix route
   is what makes change 1 free: tau already needs ``D . nabla_m phi`` for
   ``m = x,y,z``, and those are exactly the three products the tau part of the
   gradient contracts against.

2. **libxc is called once per group of grid blocks, not once per block.**
   GPU4PySCF calls ``eval_xc_eff`` inside the block loop, so a PfPMT/6-31G\\*
   gradient makes 830 calls on 4096 points each.  Measured on this grid:
   B3LYP on all 3 396 608 points in one call is **3.2 ms**; the same points in
   830 calls is **388 ms**, and for M06-2X 4.5 ms against 555 ms.  Essentially
   all of it is per-call overhead, and it is the largest single item in the
   XC gradient — 42 % of the GGA time and 35 % of the meta-GGA time.  Grid
   blocks are therefore accumulated into a group, one libxc call covers the
   group, and the potential contraction runs over the same blocks while their
   AO values are still resident.

   The grouping keeps its own buffer.  GPU4PySCF's ``_grouped_block_loop``
   (``dft/numint.py``) lets ``eval_ao`` allocate every block separately, and
   because ``nao_sub`` differs from block to block the memory pool cannot
   reuse them: that walk alone costs **1.60 s** at deriv=2 on PfPMT where
   ``block_loop``'s single reused buffer costs **0.13 s**.  ``_group_block_loop``
   here is ``block_loop`` with ``nslot`` fixed-size slots instead of one.

   Note what is *not* worth batching.  Change 1 leaves two GEMMs per block
   (five for meta-GGA), each ``nao_sub x nao_sub`` by 4096, and those are
   already flop-bound: on one group of 68 blocks ``grouped_gemm`` takes
   6.12 ms against 5.80 ms for the same 68 ``cupy.dot`` calls.  The
   ``FASTXCGRAD_MODE=groupedgemm`` ablation keeps the CUTLASS grouped path so
   the comparison can be re-run.

LDA, GGA and meta-GGA are covered.  The grid-response path, unrestricted
densities and multi-GPU fall through to GPU4PySCF unchanged.

Environment switches (for the ablations, not for normal use):
``FASTXCGRAD_OFF=1``          hand everything back to GPU4PySCF
``FASTXCGRAD_MODE=perblock``  one block at a time, libxc per block
``FASTXCGRAD_MODE=groupedgemm`` grouped, with the GEMMs through grouped_gemm
``FASTXCGRAD_MGGA=0``         hand meta-GGA back to GPU4PySCF
"""
import os
import functools
import numpy as np
import cupy
from gpu4pyscf.dft import numint as NI
from . import compat as g4pcompat
from gpu4pyscf.dft import xc_deriv
from gpu4pyscf.dft.numint import _contract_rho, _contract_rho1, _scale_ao
from gpu4pyscf.grad import rks as grks
from gpu4pyscf.lib.cupy_helper import (
    contract, take_last2d, grouped_gemm, get_avail_mem)
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import num_devices

from ._paths import KERNEL_DIR as HERE

_OFF = bool(os.environ.get('FASTXCGRAD_OFF'))
MODE = os.environ.get('FASTXCGRAD_MODE', 'grouped')
_MGGA = os.environ.get('FASTXCGRAD_MGGA', '1') != '0'
# fraction of free memory the AO group and its scratch may occupy
MEM_FRACTION = float(os.environ.get('FASTXCGRAD_MEM', '0.3'))
# the grid reductions of fastxcgrad.cu; 0 falls back to cuTENSOR contract
KERNEL = os.environ.get('FASTXCGRAD_KERNEL', '1') != '0'
# 1 = the single fused xcg_exc1; 0 = xcg_rowdot3 + xcg_hessdot around
# GPU4PySCF's _make_dR_dao_w, which is the same arithmetic in four kernels
FUSE = os.environ.get('FASTXCGRAD_FUSE', '1') != '0'
_NT = int(os.environ.get('FASTXCGRAD_NT', 256))

# how many density products D . ao[m] the density needs, per functional type
_NC = {'LDA': 1, 'GGA': 1, 'MGGA': 4}


@functools.lru_cache(maxsize=4)
def _module(nt):
    src = open(os.path.join(HERE, 'fastxcgrad.cu')).read()
    return cupy.RawModule(code=src, backend='nvrtc',
                          options=('-std=c++17', f'-DNT={nt}'))


@functools.lru_cache(maxsize=8)
def _kernel(name):
    return _module(_NT).get_function(name)


def _xctypes():
    return ('LDA', 'GGA', 'MGGA') if _MGGA else ('LDA', 'GGA')


def _group_block_loop(ni, _sorted_mol, grids, nao, deriv, scratch_comp):
    """``ni.block_loop``, but yielding groups of blocks that are all resident.

    ``scratch_comp`` is how many ``nao_sub x ngrids_block`` arrays the caller
    keeps alive per block for the length of a group (the density products and
    ``D . aow``); it is budgeted here together with the AO values so the group
    size follows free memory rather than anything about the molecule.
    """
    if grids.coords is None:
        grids.build(with_non0tab=False, sort_grids=True)
    opt = ni.gdftopt
    blksize = NI.MIN_BLK_SIZE
    ngrids = grids.size
    comp = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6
    non0ao_idx = grids.get_non0ao_idx(opt)
    nblk = (ngrids + blksize - 1) // blksize
    nao_max = max(len(x[1]) for x in non0ao_idx[:nblk])
    slot = comp * nao_max * blksize
    per_blk = (comp + scratch_comp) * nao_max * blksize * 8
    nslot = max(1, min(int(MEM_FRACTION * get_avail_mem() / per_blk), nblk))
    buf = cupy.empty(nslot * slot)

    ao_group, idx_group, w_group = [], [], []
    for block_id in range(nblk):
        pad, idx, non0shl_idx, ctr_offsets, ao_loc_slice = non0ao_idx[block_id]
        nao_sub = len(idx)
        if nao_sub == 0:
            continue
        ip0 = block_id * blksize
        ip1 = min(ip0 + blksize, ngrids)
        out = cupy.ndarray((comp, nao_sub, ip1 - ip0),
                           memptr=buf.data + len(ao_group) * slot * 8)
        ao_mask = NI.eval_ao(
            _sorted_mol, cupy.asarray(grids.coords[ip0:ip1]), deriv,
            nao_slice=nao_sub, shls_slice=non0shl_idx,
            ao_loc_slice=ao_loc_slice, ctr_offsets_slice=ctr_offsets,
            gdftopt=opt, transpose=False, out=out)
        if pad > 0:
            ao_mask[:, -pad:, :] = 0.0
        ao_group.append(ao_mask)
        idx_group.append(idx)
        w_group.append(cupy.asarray(grids.weights[ip0:ip1]))
        if len(ao_group) == nslot:
            yield ao_group, idx_group, w_group
            ao_group, idx_group, w_group = [], [], []
    if ao_group:
        yield ao_group, idx_group, w_group


def _prep(ni, mol, grids, dms, nao_from_sorted=False):
    """The sorted mol, and the density matrix in the sorted AO order."""
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao if nao_from_sorted else mol.nao
    dm = cupy.asarray(dms).reshape(-1, nao, nao)
    if dm.shape[0] != 1:
        return opt, None
    return opt, opt.sort_orbitals(dm, axis=[1, 2])[0]


def _rho_from_c(ao, c, xctype, buf=None):
    """rho from the density products ``c[m] = D . ao[m]``.

    Same arithmetic and the same layout as ``numint.eval_rho``: for meta-GGA
    ``rho[4]`` is tau = 1/2 sum_m |nabla_m psi|^2 and ``with_lapl`` is off.
    """
    if xctype == 'LDA':
        return _contract_rho(c[0], ao[0]).reshape(1, -1)
    if xctype == 'GGA':
        rho = _contract_rho1(ao[:4], c[0])
        rho[1:4] *= 2
        return rho
    ngrids = ao.shape[-1]
    rho = cupy.empty((5, ngrids))
    _contract_rho1(ao[:4], c[0], rho=rho)   # fills rho[:4] in place
    rho[1:4] *= 2
    tau = _contract_rho(c[1], ao[1], rho=buf)
    rho[4] = tau
    for m in (2, 3):
        rho[4] += _contract_rho(c[m], ao[m], rho=buf)
    rho[4] *= .5
    return rho


def _weights(ni, xc_code, xctype, rho, weight):
    """v_xc times the grid weights, with GPU4PySCF's symmetry factors."""
    vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
    wv = cupy.multiply(weight, vxc, out=vxc)
    if xctype != 'LDA':
        wv[0] *= .5
    if xctype == 'MGGA':
        wv[4] *= .5   # for the factor 1/2 in tau
    return wv


def _rowdot3(rT, A, b, beta):
    """rT[n,i] = beta*rT[n,i] + sum_g A[n,i,g] b[i,g]."""
    nsub, ngrids = b.shape
    if KERNEL:   # only reached with FUSE off
        _kernel('xcg_rowdot3')((nsub,), (_NT,),
                               (rT, A, b, nsub, ngrids, beta))
    else:
        contract('nig,ig->ni', A, b, beta=beta, out=rT)


def _hessdot(rT, ao2, V, beta):
    """rT[n,i] = beta*rT[n,i] + sum_b sum_g H[n,b][i,g] V[b,i,g].

    ``H`` is the AO Hessian, ``ao2 = ao[4:10]`` in the packed order
    XX XY XZ YY YZ ZZ, so no row of ``H`` is three contiguous components; the
    fall-back therefore needs five ``contract`` calls where the kernel needs
    one.
    """
    nsub, ngrids = V.shape[1:]
    if KERNEL:   # only reached with FUSE off
        _kernel('xcg_hessdot')((nsub,), (_NT,),
                               (rT, ao2, V, nsub, ngrids, beta))
        return
    assert beta == 1.
    contract('nig,ig->ni', ao2[0:3], V[0], beta=1., out=rT)
    contract('ig,ig->i', ao2[1], V[1], beta=1., out=rT[0])
    contract('nig,ig->ni', ao2[3:5], V[1], beta=1., out=rT[1:3])
    contract('ig,ig->i', ao2[2], V[2], beta=1., out=rT[0])
    contract('nig,ig->ni', ao2[4:6], V[2], beta=1., out=rT[1:3])


def _accumulate(ao, idx, wv, c, t, exc1_ao, xctype):
    """One grid block's contribution to ``exc1_ao``.

    ``c`` is the contiguous ``(nc, nao_sub, ngrids)`` array of density
    products ``c[m] = D . ao[m]``.  ``t = D . aow`` with
    ``aow = sum_{m<4} ao[m] wv[m]`` is needed for GGA; for meta-GGA it is
    ``sum_m wv[m] c[m]``, which ``xcg_exc1`` forms itself, so the driver may
    pass ``t = None``.
    """
    if xctype == 'LDA':
        exc1_ao[idx] += contract('nig,ig->in', ao[1:4], c[0] * wv[0])
        return
    nsub, ngrids = ao.shape[1:]
    rT = cupy.empty((3, nsub))
    if KERNEL and FUSE:
        _kernel('xcg_exc1')(
            (nsub,), (_NT,),
            (rT, ao, c, c if t is None else t, cupy.ascontiguousarray(wv),
             nsub, ngrids, 1 if xctype == 'MGGA' else 0))
        exc1_ao[idx] += rT.T
        return
    if t is None:
        t = _scale_ao(c[:4], wv[:4])
    # (nabla_n phi_i) v_xc phi_j D_ij, with the j side already contracted
    _rowdot3(rT, ao[1:4], t, 0.)
    # phi_i v_xc (nabla_n phi_j) D_ij, i.e. GPU4PySCF's _make_dR_dao_w term.
    # Its weights depend only on the grid point, so one fused kernel of
    # GPU4PySCF's own does the AO-Hessian contraction there.
    aow = grks._make_dR_dao_w(ao, cupy.ascontiguousarray(wv[:4]))
    _rowdot3(rT, aow, c[0], 1.)
    aow = None
    if xctype == 'MGGA':
        # tau: sum_g wv4 H[n,m][i,g] (D . nabla_m phi)[i,g].  wv4 is diagonal
        # in g, so it commutes with the density contraction and costs no GEMM;
        # unlike the GGA term above the multiplier depends on i as well as g.
        V = cupy.multiply(c[1:4], wv[4], out=c[1:4])
        _hessdot(rT, ao[4:10], V, 1.)
    exc1_ao[idx] += rT.T


# ---------------------------------------------------------------- get_exc ----

def _exc_grouped(ni, _sorted_mol, grids, xc_code, dm, xctype, nao,
                 cutlass=False):
    """The XC gradient with one libxc call per group of grid blocks.

    ``cutlass=True`` routes the two GEMMs through CUTLASS ``grouped_gemm``
    instead of one ``cupy.dot`` per block; it is the ablation for change 2's
    last paragraph and is not the default.
    """
    ao_deriv = 1 if xctype == 'LDA' else 2
    nc = _NC[xctype]
    scratch = nc + (1 if xctype == 'GGA' else 0)
    exc1_ao = cupy.zeros((nao, 3))
    for ao_group, idx_group, w_group in _group_block_loop(
            ni, _sorted_mol, grids, nao, ao_deriv, scratch):
        nblk = len(ao_group)
        masks = [take_last2d(dm, idx) for idx in idx_group]
        # ---- the density products D . ao[m], into one contiguous array per
        #      block so that the whole of it is one operand for xcg_exc1
        c_group = [cupy.empty((nc,) + ao.shape[1:]) for ao in ao_group]
        if cutlass:
            # dm_mask is symmetric, so grouped_gemm's A^T.B is dm_mask . ao[m]
            grouped_gemm([m_ for m_ in masks for _ in range(nc)],
                         [ao[m] for ao in ao_group for m in range(nc)],
                         Cs=[C[m] for C in c_group for m in range(nc)])
        else:
            for m_, ao, C in zip(masks, ao_group, c_group):
                for m in range(nc):
                    cupy.dot(m_, ao[m], out=C[m])
        # ---- rho, and one libxc call for the whole group
        buf = cupy.empty(max(w.size for w in w_group))
        rho = cupy.hstack([_rho_from_c(ao, c, xctype, buf=buf)
                           for ao, c in zip(ao_group, c_group)])
        buf = None
        wv = _weights(ni, xc_code, xctype, rho, cupy.concatenate(w_group))
        rho = None
        wv_group = []
        p0 = 0
        for w in w_group:
            p1 = p0 + w.size
            wv_group.append(cupy.ascontiguousarray(wv[:, p0:p1]))
            p0 = p1
        wv = None
        # ---- the potential product D . aow.  LDA has none; meta-GGA needs no
        #      GEMM for it, because D . sum_m ao[m] wv[m] = sum_m wv[m]
        #      (D . ao[m]) and meta-GGA already has all four D . ao[m] for tau
        #      -- so what is a GEMM for GGA is a scale here, and xcg_exc1 folds
        #      even that in.  Four GEMMs per block for meta-GGA, not five.
        if xctype in ('LDA', 'MGGA'):
            t_group = [None] * nblk
        elif cutlass:
            aow_group = [_scale_ao(ao[:4], w[:4])
                         for ao, w in zip(ao_group, wv_group)]
            t_group = grouped_gemm(masks, aow_group)
            aow_group = None
        else:
            t_group = [m_.dot(_scale_ao(ao[:4], w[:4]))
                       for m_, ao, w in zip(masks, ao_group, wv_group)]
        masks = None
        for ao, idx, w, c, t in zip(ao_group, idx_group, wv_group,
                                    c_group, t_group):
            _accumulate(ao, idx, w, c, t, exc1_ao, xctype)
        ao_group = c_group = t_group = wv_group = None
    return exc1_ao


def _exc_perblock(ni, _sorted_mol, grids, xc_code, dm, xctype, nao):
    """The same arithmetic, one grid block at a time (for the ablation)."""
    ao_deriv = 1 if xctype == 'LDA' else 2
    nc = _NC[xctype]
    exc1_ao = cupy.zeros((nao, 3))
    for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao,
                                                 ao_deriv, None):
        dm_mask = take_last2d(dm, idx)
        c = cupy.empty((nc,) + ao_mask.shape[1:])
        for m in range(nc):
            cupy.dot(dm_mask, ao_mask[m], out=c[m])
        rho = _rho_from_c(ao_mask, c, xctype)
        wv = _weights(ni, xc_code, xctype, rho, weight)
        if xctype in ('LDA', 'MGGA'):   # see _exc_grouped
            t = None
        else:
            t = dm_mask.dot(_scale_ao(ao_mask[:4], wv[:4]))
        _accumulate(ao_mask, idx, wv, c, t, exc1_ao, xctype)
        dm_mask = c = t = None
    return exc1_ao


def get_exc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    """Drop-in replacement for gpu4pyscf.grad.rks.get_exc."""
    xctype = ni._xc_type(xc_code)
    if _OFF or xctype not in _xctypes() or num_devices > 1:
        return grks._get_exc_orig(ni, mol, grids, xc_code, dms, relativity,
                                  hermi, max_memory, verbose)
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    opt, dm = _prep(ni, mol, grids, dms)
    if dm is None:
        return grks._get_exc_orig(ni, mol, grids, xc_code, dms, relativity,
                                  hermi, max_memory, verbose)
    _sorted_mol = opt._sorted_mol
    if MODE == 'perblock':
        exc1_ao = _exc_perblock(ni, _sorted_mol, grids, xc_code, dm, xctype,
                                mol.nao)
    else:
        exc1_ao = _exc_grouped(ni, _sorted_mol, grids, xc_code, dm, xctype,
                               mol.nao, cutlass=(MODE == 'groupedgemm'))
    # - sign because nabla_X = -nabla_x
    exc1 = -grks._reduce_to_atom(_sorted_mol, exc1_ao)
    log.timer_debug1(f'grad vxc (fastxcgrad, {MODE})', *t0)
    return None, exc1


# ------------------------------------------------------------ get_nlc_exc ----

def _nlc_rho_grouped(ni, _sorted_mol, grids, dm, nao):
    """Pass one: rho over the whole grid.

    Only ``ao[:4]`` is needed here, so this pass evaluates the AOs to first
    derivative where GPU4PySCF evaluates them to second.

    Neither NLC pass calls libxc -- the VV10 potential comes from
    ``_vv10nlc`` for the whole grid at once -- so grouping buys nothing here
    beyond keeping one code path with ``get_exc``; what these two functions
    change against GPU4PySCF is the density contraction (change 1), the
    first-derivative AOs in this pass, and ``xcg_exc1`` in the next.
    """
    out = []
    for ao_group, idx_group, w_group in _group_block_loop(
            ni, _sorted_mol, grids, nao, 1, 1):
        c0_group = [take_last2d(dm, idx).dot(ao[0])
                    for idx, ao in zip(idx_group, ao_group)]
        for ao, c0 in zip(ao_group, c0_group):
            rho = _contract_rho1(ao[:4], c0)
            rho[1:4] *= 2
            out.append(rho)
        ao_group = c0_group = None
    return cupy.hstack(out)


def _nlc_exc_grouped(ni, _sorted_mol, grids, dm, vv_vxc, nao):
    """Pass two: contract the VV10 potential."""
    exc1_ao = cupy.zeros((nao, 3))
    p1 = 0
    for ao_group, idx_group, w_group in _group_block_loop(
            ni, _sorted_mol, grids, nao, 2, 2):
        wv_group = []
        for w in w_group:
            p0, p1 = p1, p1 + w.size
            wv = vv_vxc[:, p0:p1] * w
            wv[0] *= .5
            wv_group.append(cupy.ascontiguousarray(wv))
        masks = [take_last2d(dm, idx) for idx in idx_group]
        c0_group = [m_.dot(ao[0]) for m_, ao in zip(masks, ao_group)]
        t_group = [m_.dot(_scale_ao(ao[:4], w[:4]))
                   for m_, ao, w in zip(masks, ao_group, wv_group)]
        masks = None
        for ao, idx, w, c0, t in zip(ao_group, idx_group, wv_group,
                                     c0_group, t_group):
            _accumulate(ao, idx, w, c0[None], t, exc1_ao, 'GGA')
        ao_group = c0_group = t_group = wv_group = None
    return exc1_ao


def _nlc_rho_perblock(ni, _sorted_mol, grids, dm, nao, max_memory):
    out = []
    for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, 1,
                                                 max_memory):
        rho = _contract_rho1(ao_mask[:4], take_last2d(dm, idx).dot(ao_mask[0]))
        rho[1:4] *= 2
        out.append(rho)
    return cupy.hstack(out)


def _nlc_exc_perblock(ni, _sorted_mol, grids, dm, vv_vxc, nao, max_memory):
    exc1_ao = cupy.zeros((nao, 3))
    p1 = 0
    for ao_mask, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, 2,
                                                 max_memory):
        p0, p1 = p1, p1 + weight.size
        wv = vv_vxc[:, p0:p1] * weight
        wv[0] *= .5
        dm_mask = take_last2d(dm, idx)
        c0 = dm_mask.dot(ao_mask[0])
        t = dm_mask.dot(_scale_ao(ao_mask[:4], wv[:4]))
        _accumulate(ao_mask, idx, wv, c0[None], t, exc1_ao, 'GGA')
        dm_mask = c0 = t = None
    return exc1_ao


def get_nlc_exc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                max_memory=2000, verbose=None):
    """Drop-in replacement for gpu4pyscf.grad.rks.get_nlc_exc.

    The VV10 double sum itself is ``numint._vv10nlc`` (``fastnlc``'s kernel if
    that module is imported); what this replaces is the two passes over the
    grid around it, with the same changes as ``get_exc`` above.
    """
    opt, dm = _prep(ni, mol, grids, dms, nao_from_sorted=True)
    if _OFF or dm is None or num_devices > 1:
        return grks._get_nlc_exc_orig(ni, mol, grids, xc_code, dms, relativity,
                                      hermi, max_memory, verbose)
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    nlc_coefs = ni.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]

    if MODE != 'perblock':
        rho = _nlc_rho_grouped(ni, _sorted_mol, grids, dm, nao)
    else:
        rho = _nlc_rho_perblock(ni, _sorted_mol, grids, dm, nao, max_memory)
    vxc = g4pcompat.vv10nlc(rho, grids.coords, grids.weights, nlc_pars)[1]
    vv_vxc = xc_deriv.transform_vxc(rho, vxc, 'GGA', spin=0)
    rho = vxc = None
    if MODE != 'perblock':
        exc1_ao = _nlc_exc_grouped(ni, _sorted_mol, grids, dm, vv_vxc, nao)
    else:
        exc1_ao = _nlc_exc_perblock(ni, _sorted_mol, grids, dm, vv_vxc, nao,
                                    max_memory)
    exc1 = -grks._reduce_to_atom(_sorted_mol, exc1_ao)
    log.timer_debug1(f'grad nlc vxc (fastxcgrad, {MODE})', *t0)
    return None, exc1


_PATCHED = False


def apply_patch():
    global _PATCHED
    if not _PATCHED:
        grks._get_exc_orig = grks.get_exc
        grks.get_exc = get_exc
        grks._get_nlc_exc_orig = grks.get_nlc_exc
        grks.get_nlc_exc = get_nlc_exc
        _PATCHED = True


apply_patch()
