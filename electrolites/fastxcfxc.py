"""
Faster XC response kernel (fxc) for GPU4PySCF -- the grid half of every
response property.

Importing this module monkey-patches ``gpu4pyscf.dft.numint.nr_rks_fxc`` and
``NumInt.nr_rks_fxc``.  Both are needed: ``nr_rks_fxc_st``, which is what
TDA/TDDFT reaches, calls the **module-level** ``nr_rks_fxc`` rather than the
bound method, so patching only the class attribute leaves the excited-state
path on GPU4PySCF's code.

``nr_rks_fxc`` is the single entry point for the XC part of the CPKS solve,
the TDA/TDDFT Davidson matrix-vector product, NMR shielding, IR/Raman and the
C6 coefficients -- ``scf/_response_functions.py`` routes all of them through
it.  Measured on GPU4PySCF 1.8.1 it is 23.2 % of a polarizability and 25.2 %
of a five-root TDA run (``benchmarks/response_anatomy.py``).

What changes, and what does not.  The grid, the AO screening, ``fxc`` itself
and every symmetry factor are GPU4PySCF's; only the arrangement of the work
changes, so the potential matrices agree to round-off.

1. **The two GEMMs are one GEMM each, not one per right-hand side.**
   GPU4PySCF issues, per grid block, a Python loop ``for i in range(nset)``
   for the density and a second one for the potential (``dft/numint.py:1301``,
   ``:1313``).  The ``nset`` masked density matrices are contiguous, so
   ``D . ao_0`` for all of them is one ``(nset*nao_sub, nao_sub) x
   (nao_sub, ngrids)`` GEMM; the ``nset`` weighted AO products likewise give
   the potential in one ``(nset*nao_sub, ngrids) x (ngrids, nao_sub)``.

2. **Three custom kernels replace ten per set.**  ``kernels/fastxcfxc.cu``
   does the density contraction, the ``fxc`` contraction and the AO weighting
   with the set dimension inside the kernel, reading ``ao`` once per tile of
   four sets instead of once per set, and folding in the ``rho1[1:4] *= 2`` of
   the Hermitian density and the ``wv[0] *= .5`` of the GGA potential.  The
   ``fxc`` contraction is the one worth the most: GPU4PySCF's
   ``_contract_rho1_fxc`` broadcasts and sums, which materialises an
   ``(nset, 4, 4, ngrids)`` temporary -- four times the traffic of its own
   result.

3. **A non-Hermitian right-hand side is symmetrised once instead of per
   block.**  For ``hermi = 0`` -- the TDDFT case -- ``eval_rho`` computes a
   second product ``dm^T . ao[0]`` and three more contractions per block.  It
   need not: with ``rho1[0] = sum_u (D.ao_0)_u ao_{0u}`` and
   ``rho1[n] = sum_u ((D + D^T).ao_0)_u ao_{nu}``, both are what the
   ``hermi = 1`` formulae give for ``D -> (D + D^T)/2``, because ``rho1[0]``
   already sees only the symmetric part.  So the density is symmetrised once,
   for ``nset*nao^2`` work per call, and the block loop runs the Hermitian
   path.  ``tests/test_fxc.py`` checks this against GPU4PySCF at
   ``hermi = 0`` rather than resting on the algebra.

**What the catalogue asks for here, and why it is not what shipped.**
Catalogue entries G1 and B4d prescribe P8 -- ``_grouped_block_loop`` plus
``grouped_gemm``, the change that carried ``fastxcgrad``.  It is implemented
here as ``MODE=grouped`` and it is **slower than keeping GPU4PySCF's
one-block-at-a-time issue order**: 0.72 s against the 0.22 s that ships, at
``nset = 20`` on penicillin V.  The reason is that this path already has ``nset`` GEMMs in
flight per block with no synchronisation between them, so there is much less
launch latency to hide than in the gradient, while ``grouped_gemm`` pays a
host-side pointer table built in Python for every ``(block, set)`` pair.  What
does pay is batching along ``nset``, which is a stride in memory rather than a
list of pointers.  ``docs/README_fastxcfxc.md`` has both columns.

LDA, GGA and meta-GGA are covered.  Meta-GGA is here even though ``fastxc``
declines it in the SCF, because on the response path it is not a tail: an
M06-2X polarizability spends **48.4 %** of its wall clock in this function and
a five-root M06-2X TDA run **47.0 %**, against 25-28 % for B3LYP, since the
``tau`` term needs a density product per gradient component and four potential
GEMMs instead of one.

**Limits, all of which fall through to GPU4PySCF rather than fail.**
The ``mo1``/``occ_coeff`` path (that is the Hessian's CPHF solve, which 1.8.1
serves from ``hessian/rks.py`` in the MO basis and which
``docs/HESSIAN_1.8.1.md`` measures at a third of the card's fp64 peak -- not
launch-bound, so this treatment is not the one it wants), UKS
(``nr_uks_fxc``), and more than one GPU.  ``MODE=grouped`` additionally
declines meta-GGA: it exists only to A/B the issue order.
"""
import functools
import os
import time as _time

import cupy
import numpy as np

from gpu4pyscf.dft import numint as NI
from gpu4pyscf.dft.numint import (
    _contract_rho, _contract_rho1, _contract_rho1_fxc, _scale_ao)
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    add_sparse, contract, grouped_dot, grouped_gemm, ndarray, take_last2d,
    transpose_sum)
from gpu4pyscf.__config__ import num_devices

from ._paths import KERNEL_DIR as HERE

#: 'blocked' is the shipped path.  'nokernel' keeps its batched GEMMs but puts
#: GPU4PySCF's own per-set contraction kernels back, which is the ablation for
#: ``kernels/fastxcfxc.cu``.  'grouped' is the catalogue's P8 prescription --
#: ``grouped_gemm`` over (block, set) -- kept because it is what entry B4d
#: asks for and it measures slower; see the module docstring.
MODE = os.environ.get('FASTXCFXC_MODE', 'blocked')

#: FASTXCFXC_TIME=1 prints the per-stage split of each call.
TIME = int(os.environ.get('FASTXCFXC_TIME', 0))

_WX = int(os.environ.get('FASTXCFXC_WX', 32))
_TSY = int(os.environ.get('FASTXCFXC_TSY', 4))
#: AO indices one block of ``fxc_scale`` writes; the AO index is tiled over
#: blockIdx.z there so that a three-right-hand-side call still fills the card.
_UPB = int(os.environ.get('FASTXCFXC_UPB', 32))

#: One chunk of the grouped ablation holds ``nset * nao_sub * blksize``
#: doubles twice over.  The budget is on bytes rather than on a block count
#: because ``nset`` is a property of the solver -- three dipole directions, or
#: a Davidson subspace -- and not of the molecule.
CHUNK_BYTES = float(os.environ.get('FASTXCFXC_CHUNK_BYTES', 512 << 20))

_T = {'prep': 0., 'gemm1': 0., 'rho1': 0., 'wv': 0., 'scale': 0.,
      'gemm2': 0., 'accum': 0., 'n': 0}


@functools.lru_cache(maxsize=8)
def _module(wx, tsy):
    src = open(os.path.join(HERE, 'fastxcfxc.cu')).read()
    return cupy.RawModule(code=src, backend='nvrtc',
                          options=('-std=c++17', f'-DWX={wx}', f'-DTSY={tsy}'))


@functools.lru_cache(maxsize=16)
def _kernel(name):
    return _module(_WX, _TSY).get_function(name)


def _usable(ni, xc_code, dms, hermi, fxc):
    """Conditions the block loop below assumes.

    None of them is the size of the molecule: they are the functional type,
    what the caller asked for, and how many devices there are.
    """
    if num_devices != 1:
        return False
    if hasattr(dms, 'mo1'):                 # the Hessian's MO-basis CPHF path
        return False
    xctype = ni._xc_type(xc_code)
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        return False
    if getattr(dms, 'ndim', 0) not in (2, 3):
        return False
    # the kernels index fxc as [y][x][g] with a stride of the whole grid
    ncomp = {'LDA': 1, 'GGA': 4, 'MGGA': 5}[xctype]
    if (fxc is None or fxc.ndim != 3 or fxc.shape[:2] != (ncomp, ncomp)
            or not fxc.flags.c_contiguous):
        return False
    return True


def _prep(ni, mol, grids, dms, hermi):
    opt = getattr(ni, 'gdftopt', None)
    if opt is None or mol not in [opt.mol, opt._sorted_mol]:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    nao = dms.shape[-1]
    dms = cupy.asarray(dms).reshape(-1, nao, nao)
    dms = opt.sort_orbitals(dms, axis=[1, 2])
    dms = cupy.asarray(dms, order='C')
    if hermi != 1:
        # item 3 of the module docstring: (D + D^T)/2 gives the same rho1
        dms = transpose_sum(dms, inplace=False)
        dms *= .5
    return opt, dms


def _nr_rks_fxc_blocked(ni, mol, grids, xc_code, dm0, dms, relativity, hermi,
                        rho0, vxc, fxc, max_memory, verbose, kernels=True):
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    if TIME:
        cupy.cuda.Stream.null.synchronize()
        _t = _time.perf_counter()
    xctype = ni._xc_type(xc_code)
    dm_shape = dms.shape
    opt, dms = _prep(ni, mol, grids, dms, hermi)
    nset = dms.shape[0]
    _sorted_mol = opt._sorted_mol

    lda = xctype == 'LDA'
    mgga = xctype == 'MGGA'
    ncomp = 1 if lda else (5 if mgga else 4)
    # how many density products D . ao[n] the density needs: meta-GGA's tau
    # term wants one per gradient component, everything else only ao[0]
    nc0 = 4 if mgga else 1
    ao_deriv = 0 if lda else 1
    scale0 = 1.0 if lda else 0.5
    # `_tau_dot` halves wv[4] once more on top of the .5 its caller applies
    scale4 = 0.25 if mgga else 1.0
    suffix = 'lda' if lda else ('mgga' if mgga else 'gga')
    nao = _sorted_mol.nao
    ngrids_tot = fxc.shape[-1]
    vmat = cupy.zeros((nset, nao, nao))

    blk = NI.MIN_BLK_SIZE
    nsub_max = max(len(x[1]) for x in grids.get_non0ao_idx(opt))
    nsub_max = max(nsub_max, 1)
    c0_buf = cupy.empty(nc0 * nset * nsub_max * blk)
    aow_buf = cupy.empty(nset * nsub_max * blk)
    rho_buf = cupy.empty(nset * ncomp * blk)
    wv_buf = cupy.empty(nset * ncomp * blk)
    v_buf = cupy.empty(nset * nsub_max * nsub_max)
    # the masked densities go in a buffer too: `docs/README_fastxchess.md`
    # 3b measured per-block allocation of an array this shape at 48 % of that
    # function, and this one is the same shape times `nset`
    dm_buf = cupy.empty(nset * nsub_max * nsub_max)
    if TIME:
        cupy.cuda.Stream.null.synchronize()
        _T['prep'] += _time.perf_counter() - _t
        _t = _time.perf_counter()

    # `strict_grid_order` keeps the empty blocks in the stream, so p0/p1 stay
    # the block's real grid offset -- which is what `fxc` is sliced by.
    p0 = p1 = 0
    for ao, idx, w, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv,
                                       max_memory=None,
                                       strict_grid_order=True):
        ng = w.size
        p0, p1 = p1, p1 + ng
        nsub = len(idx)
        if nsub == 0:
            continue
        ao0 = ao if lda else ao[0]

        # ---- rho1 = (D . ao_0) . ao, one GEMM for every set --------------
        dm_mask = take_last2d(dms, idx, out=dm_buf)
        dm2 = dm_mask.reshape(nset * nsub, nsub)
        c0 = ndarray((nc0, nset * nsub, ng), buffer=c0_buf)
        for n in range(nc0):
            cupy.dot(dm2, ao0 if lda else ao[n], out=c0[n])
        dm_mask = dm2 = None
        if TIME:
            cupy.cuda.Stream.null.synchronize()
            _T['gemm1'] += _time.perf_counter() - _t
            _t = _time.perf_counter()

        rho1 = ndarray((nset, ncomp, ng), buffer=rho_buf)
        grid = ((ng + _WX - 1) // _WX, (nset + _TSY - 1) // _TSY)
        if kernels:
            _kernel(f'fxc_rho1_{suffix}')(
                grid, (_WX, _TSY),
                (rho1, c0, ao, np.int32(nset), np.int32(nsub), np.int32(ng)))
        else:
            c0v = c0.reshape(nc0, nset, nsub, ng)
            for i in range(nset):
                if lda:
                    _contract_rho(c0v[0, i], ao0, rho=rho1[i, 0])
                else:
                    _contract_rho1(ao, c0v[0, i], rho=rho1[i][:4])
                if mgga:
                    tau = rho1[i, 4]
                    tau[:] = 0
                    for x in (1, 2, 3):
                        tau += _contract_rho(c0v[x, i], ao[x])
                    tau *= .5
            if not lda:
                rho1[:, 1:4] *= 2
        if TIME:
            cupy.cuda.Stream.null.synchronize()
            _T['rho1'] += _time.perf_counter() - _t
            _t = _time.perf_counter()

        # ---- the fxc contraction ----------------------------------------
        aow = ndarray((nset * nsub, ng), buffer=aow_buf)
        nz = (nsub + _UPB - 1) // _UPB
        if kernels:
            wv = ndarray((nset, ncomp, ng), buffer=wv_buf)
            _kernel(f'fxc_wv_{suffix}')(
                grid, (_WX, _TSY),
                (wv, rho1, fxc[:, :, p0:p1], w, np.int32(nset),
                 np.int32(ng), np.int32(ngrids_tot), np.float64(scale0),
                 np.float64(scale4)))
        else:
            wv = _contract_rho1_fxc(rho1, fxc[:, :, p0:p1] * w)
            if not lda:
                wv[:, 0] *= scale0
            if mgga:
                wv[:, 4] *= scale4
        if TIME:
            cupy.cuda.Stream.null.synchronize()
            _T['wv'] += _time.perf_counter() - _t
            _t = _time.perf_counter()

        # ---- the weighted AOs and the potential matrix -------------------
        # aow_i . ao_0^T is the transpose of GPU4PySCF's ao_0 . aow_i^T.  For
        # GGA and meta-GGA the final transpose_sum absorbs it; for LDA the
        # block matrix is symmetric, since aow = ao_0 * wv there, and the tau
        # term of meta-GGA is symmetric on its own.
        vblk = ndarray((nset * nsub, nsub), buffer=v_buf)

        def _weight(nw, w0, a0):
            # aow[i][v][g] = sum_n wv[i][w0+n][g] * ao[a0+n][v][g]
            if kernels:
                _kernel('fxc_scale_lda' if nw == 1 else 'fxc_scale_gga')(
                    (grid[0], grid[1], nz), (_WX, _TSY),
                    (aow, wv, ao, np.int32(nset), np.int32(nsub),
                     np.int32(ng), np.int32(_UPB), np.int32(ncomp),
                     np.int32(w0), np.int32(a0)))
                return
            aowv = aow.reshape(nset, nsub, ng)
            src = (ao0 if a0 == 0 else ao[a0]) if nw == 1 else ao[:nw]
            for i in range(nset):
                _scale_ao(src, wv[i, w0] if nw == 1 else wv[i, w0:w0 + nw],
                          out=aowv[i])

        _weight(1 if lda else 4, 0, 0)
        if TIME:
            cupy.cuda.Stream.null.synchronize()
            _T['scale'] += _time.perf_counter() - _t
            _t = _time.perf_counter()
        cupy.dot(aow, ao0.T, out=vblk)
        if mgga:
            # + sum_{x=1..3} (ao[x] * wv[4]) . ao[x]^T, accumulated in place
            for x in (1, 2, 3):
                _weight(1, 4, x)
                contract('ig,jg->ij', aow, ao[x], beta=1., out=vblk)
        if TIME:
            cupy.cuda.Stream.null.synchronize()
            _T['gemm2'] += _time.perf_counter() - _t
            _t = _time.perf_counter()
        add_sparse(vmat, vblk.reshape(nset, nsub, nsub), idx)
        if TIME:
            cupy.cuda.Stream.null.synchronize()
            _T['accum'] += _time.perf_counter() - _t
            _t = _time.perf_counter()

    if not lda:
        transpose_sum(vmat)
    vmat = opt.unsort_orbitals(vmat, axis=[1, 2])
    if len(dm_shape) == 2:
        vmat = vmat[0]
    if TIME:
        cupy.cuda.Stream.null.synchronize()
        _T['n'] += 1
        print('[fastxcfxc] call %2d  ' % _T['n']
              + '  '.join(f'{k} {_T[k]:6.3f}' for k in
                          ('prep', 'gemm1', 'rho1', 'wv', 'scale', 'gemm2',
                           'accum')), flush=True)
    log.timer_debug1('nr_rks_fxc (fastxcfxc, blocked)', *t0)
    return cupy.asarray(vmat)


def _chunks(nblocks, nset, nao_sub_max, blksize):
    per_block = max(1, nset * nao_sub_max * blksize * 8)
    n = int(CHUNK_BYTES // per_block)
    return max(1, min(nblocks, n))


def _nr_rks_fxc_grouped(ni, mol, grids, xc_code, dm0, dms, relativity, hermi,
                        rho0, vxc, fxc, max_memory, verbose):
    """Catalogue B4d/G1 as written: ``grouped_gemm`` over (block, set).

    Kept because it is the prescription and it is slower than
    ``_nr_rks_fxc_blocked`` -- see the module docstring.  LDA and GGA only:
    it exists to A/B the issue order, and meta-GGA was written straight into
    the blocked path.
    """
    if ni._xc_type(xc_code) == 'MGGA':
        return NI._nr_rks_fxc_orig(ni, mol, grids, xc_code, dm0, dms,
                                   relativity, hermi, rho0, vxc, fxc,
                                   max_memory, verbose)
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    xctype = ni._xc_type(xc_code)
    dm_shape = dms.shape
    opt, dms = _prep(ni, mol, grids, dms, hermi)
    nset = dms.shape[0]
    _sorted_mol = opt._sorted_mol

    lda = xctype == 'LDA'
    ncomp = 1 if lda else 4
    ao_deriv = 0 if lda else 1
    nao = _sorted_mol.nao
    vmat = cupy.zeros((nset, nao, nao))

    p1 = 0
    for ao_group, idx_group, w_group, _ in NI._grouped_block_loop(
            ni, _sorted_mol, grids, nao, ao_deriv):
        nb = len(ao_group)
        # `_grouped_block_loop` walks every block in grid order and skips
        # none, so the running offset is the block's own grid offset.
        p0s = []
        for w in w_group:
            p0s.append(p1)
            p1 += w.size
        ao0 = [a if lda else a[0] for a in ao_group]
        active = [b for b in range(nb) if len(idx_group[b]) > 0]
        if not active:
            continue
        nsub_max = max(len(idx_group[b]) for b in active)
        blksize = max(w_group[b].size for b in active)
        step = _chunks(len(active), nset, nsub_max, blksize)

        for c0b in range(0, len(active), step):
            sel = active[c0b:c0b + step]
            masks = [take_last2d(dms, idx_group[b]) for b in sel]
            As = [m[i] for m in masks for i in range(nset)]
            Bs = [ao0[b] for b in sel for _ in range(nset)]
            c0_group = grouped_gemm(As, Bs)
            masks = As = Bs = None

            rho1_group = []
            for k, b in enumerate(sel):
                ng = w_group[b].size
                rho1 = cupy.empty((nset, ncomp, ng))
                for i in range(nset):
                    c0 = c0_group[k * nset + i]
                    if lda:
                        _contract_rho(c0, ao0[b], rho=rho1[i, 0])
                    else:
                        _contract_rho1(ao_group[b], c0, rho=rho1[i])
                if not lda:
                    rho1[:, 1:4] *= 2
                rho1_group.append(rho1)
            c0_group = None

            aow_group = []
            for k, b in enumerate(sel):
                ng = w_group[b].size
                q0 = p0s[b]
                wvs = _contract_rho1_fxc(rho1_group[k],
                                         fxc[:, :, q0:q0 + ng] * w_group[b])
                if not lda:
                    wvs[:, 0] *= .5
                nsub = len(idx_group[b])
                aow = cupy.empty((nset, nsub, ng))
                for i in range(nset):
                    _scale_ao(ao_group[b], wvs[i, 0] if lda else wvs[i],
                              out=aow[i])
                aow_group.append(aow)
            rho1_group = None

            vblk = [cupy.empty((nset, len(idx_group[b]), len(idx_group[b])))
                    for b in sel]
            Cs = [v[i] for v in vblk for i in range(nset)]
            As = [ao0[b] for b in sel for _ in range(nset)]
            Bs = [a[i] for a in aow_group for i in range(nset)]
            grouped_dot(As, Bs, Cs)
            aow_group = As = Bs = Cs = None
            for k, b in enumerate(sel):
                add_sparse(vmat, vblk[k], idx_group[b])
            vblk = None

    if not lda:
        transpose_sum(vmat)
    vmat = opt.unsort_orbitals(vmat, axis=[1, 2])
    if len(dm_shape) == 2:
        vmat = vmat[0]
    log.timer_debug1('nr_rks_fxc (fastxcfxc, grouped)', *t0)
    return cupy.asarray(vmat)


def _nr_rks_fxc_nokernel(*args, **kwargs):
    return _nr_rks_fxc_blocked(*args, kernels=False, **kwargs)


_MODES = {'blocked': _nr_rks_fxc_blocked,
          'nokernel': _nr_rks_fxc_nokernel,
          'grouped': _nr_rks_fxc_grouped}

_PATCHED = False


def _nr_rks_fxc_dispatch(ni, mol, grids, xc_code, dm0=None, dms=None,
                         relativity=0, hermi=0, rho0=None, vxc=None, fxc=None,
                         max_memory=2000, verbose=None):
    if fxc is None:
        raise RuntimeError('fxc was not initialized')
    fn = _MODES.get(MODE)
    if fn is not None and _usable(ni, xc_code, dms, hermi, fxc):
        return fn(ni, mol, grids, xc_code, dm0, dms, relativity, hermi,
                  rho0, vxc, fxc, max_memory, verbose)
    return NI._nr_rks_fxc_orig(ni, mol, grids, xc_code, dm0, dms, relativity,
                               hermi, rho0, vxc, fxc, max_memory, verbose)


def apply_patch():
    global _PATCHED
    if not _PATCHED:
        NI._nr_rks_fxc_orig = NI.nr_rks_fxc
        # the module attribute is the one `nr_rks_fxc_st` (TDA/TDDFT) calls;
        # the class attribute is the one CPKS calls.  Both are needed.
        NI.nr_rks_fxc = _nr_rks_fxc_dispatch
        NI.NumInt.nr_rks_fxc = _nr_rks_fxc_dispatch
        _PATCHED = True


apply_patch()
