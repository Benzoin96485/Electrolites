"""
Drop-in replacement for GPU4PySCF's **density-fitted** Coulomb and exchange
build -- the two-electron half of every RI/DF calculation.

Everything else in this repository is on the *direct* SCF path: `fastj` and
`fastk` replace ``scf/j_engine.py`` and ``scf/jk.py``, and `fastjk` the fused
build a response solve asks for.  A ``density_fit()`` run takes none of those.
``df/df.py:117`` routes the whole build through ``df/df_jk.py:get_jk``, which
never touches a four-centre integral: it contracts the Cholesky-decomposed
three-index tensor (the *cderi*) that ``df/df.py:_cholesky_eri`` built once.
So with ``patch_all()`` loaded a DF run's J and K are stock GPU4PySCF, and
this module is what puts them on the same footing as the direct path.  It is
catalogue **A13**, whose evidence is a 1.7.0 reading; ``docs/README_fastdf.md``
§1 says which of its four claims survive on 1.8.1 and which upstream closed in
the interval.  Item 5 below is ``docs/DF_WORK_LIST.md`` W5 and its write-up is
``docs/README_fastdfblock.md``.

What changes.  The integrals, the metric, the screening and every symmetry
factor are GPU4PySCF's -- only the arrangement of the work changes, so the
matrices agree to round-off.  Three of the four items below are worth
**1.00x-1.41x** on the build with the cderi on the device and the fourth is
worth **1.64x-4.53x** with it on the host, which is the regime the catalogue
entry is about.

1. **The exchange contraction is a symmetric rank-k update, not a GEMM --
   above a measured size.**
   ``vk[n,i,j] = sum_{k,L} rhok[n,i,k,L] rhok[n,j,k,L]`` contracts a tensor
   with *itself* whenever the two density factors are the same object, which
   is every ground-state density and every ``mo_coeff``-tagged perturbed one.
   GPU4PySCF issues it as a general GEMM (``df_jk.py:541``) and therefore
   computes both triangles of a symmetric matrix.  A ``dsyrk`` is the same
   answer for half the flops, and this contraction is the larger half of the
   exchange build: **1.11x** on the whole build at 1007 basis functions,
   **1.18x** at two densities there and **1.41x** at two densities on taxol.
   It is not unconditionally faster: cuBLAS's ``dsyrk`` is less well tuned
   than its ``dgemm`` and gives the factor of two back below about a thousand
   basis functions, where ``_use_syrk`` keeps the GEMM.  That threshold is a
   measurement, not a guess -- ``docs/README_fastdf.md`` §3.2.

2. **The first contraction is left alone, and that is a result rather than an
   omission.**  GPU4PySCF unpacks the block into ``(i,j,L)`` and then takes an
   ``L``-innermost view of it (``df/df.py:185``), so step 1 reads as a batch
   over ``L`` in which ``cderi[L]`` has a unit stride in *neither* index --
   no cuBLAS call can express that, and cuTENSOR serves it with a general
   permuting contraction.  The objection is real and the obvious fix does not
   pay.  Batching the same arithmetic over **i** instead does make every
   operand a proper matrix, needs no new layout and no transpose, and is
   carried here as ``FASTDF_STEP1=bgemm``:

       out[i,L,k] = sum_j work[i,j,L] · factor[j,k]

   is one ``dgemmStridedBatched`` with ``m = nocc``, ``n = nL``, ``k = nao``
   and ``nao`` batches.  Measured, it is **1.0x** at 430 basis functions and
   **slower** at 1007 -- because ``nao`` batches of a ``(nocc x nL)`` output
   is the small-matrix regime cuBLAS's batched path is weakest in, while
   cuTENSOR sees one large contraction.  ``docs/README_fastdf.md`` §3.4 has
   the numbers.  Two further things were measured and rejected with it: an
   ``(L,i,j)`` unpack of our own, which is **2.65x slower** than GPU4PySCF's
   because the transposed half of a symmetric fill is coalesced in GPU4PySCF's
   layout and is not in that one, and a blocked triangular GEMM in place of
   item 1's ``dsyrk``.

3. **The Coulomb pass is an explicit GEMV.**  ``auxvec[n,L] = sum_p dm[n,p]
   cderi[L,p]`` and then ``vj[n,p] += sum_L auxvec[n,L] cderi[L,p]`` are the
   whole Coulomb build, and the packed block is a plain matrix in both of the
   layouts it arrives in: the auxiliary index is contiguous when the tensor
   lives on the device (``_cholesky_eri`` returns a transpose) and the pair
   index is when it has been staged from the host.  So each pass is one
   ``dgemv`` at a single density and one ``dgemm`` at several, in place of
   GPU4PySCF's two ``contract`` calls.  Worth **1.09x** on the whole build at
   one density -- where GPU4PySCF's contraction is the one with the least
   parallelism to work with -- and nothing at two.

4. **A cderi that does not fit on the device is only partly streamed.**
   Past a few hundred atoms the three-index tensor is larger than the GPU, and
   GPU4PySCF then keeps *all* of it on the host and copies *all* of it across
   PCIe on every SCF cycle -- this is the limit its own README states,
   "density fitting scheme up to ~168 atoms with def2-tzvpd basis, bounded by
   CPU memory".  The choice is all-or-nothing only because ``_cholesky_eri``
   makes it once, for the whole tensor.  This module keeps as many auxiliary
   rows resident as the free memory allows and streams only the remainder, so
   the per-cycle PCIe traffic falls by whatever fraction fits.  The copy of
   the streamed part still overlaps the contraction on a second stream, which
   is GPU4PySCF's arrangement and is kept.  What this is worth is close to
   the fraction of the tensor that fits: measured at 25, 50, 75 and 100 % of
   a 5.5 GB tensor it is 1.21x, 1.55x, 2.66x and 4.51x.  **1.64x-4.53x** on
   the build across the systems measured, and it is the whole of those
   numbers -- the other three items are irrelevant to a build that is waiting
   on PCIe (``docs/README_fastdf.md`` §3.3).  ``FASTDF_RESIDENT_FRAC`` is its
   share of free device memory, default 0.5; ``0`` restores GPU4PySCF's
   behaviour exactly, and is the setting to reach for in one case: a **DF
   Hessian** moves the cderi to the host on purpose, to leave room for the
   Krylov subspace (``df/hessian/rhf.py:1258-1273``), and this item would take
   half of what it just freed back.  Nothing here can tell that context from
   an ordinary host-resident one, so the knob is the answer rather than a
   heuristic.

5. **The first contraction on the AO tiles the screening kept, above a
   measured sparsity.**
   ``DF_WORK_LIST.md`` W5 priced this at the reciprocal of the 22.2 % AO-pair
   fill at 284 atoms.  A GEMM addresses a tile and not a pair, and the
   fraction of ``nao x nao`` covered by tiles holding at least one stored pair
   is **69.4 %** there, **66.1 %** at 545 atoms and **98.2 %** at 113 -- the
   cderi's sparsity is the product of an atom-pair factor a tile can reach and
   a shell-structure factor at one-to-five-AO granularity that no AO tile can
   (``docs/README_fastdfblock.md`` §1.1).  What is left is real but small: the
   contraction is **1.31x** at 284 atoms and **1.38x** at 545 and the exchange
   block **1.09x** and **1.20x** -- and a whole ``get_jk`` call at 284 atoms,
   the only end-to-end size that runs, reads **0.85x to 1.02x** over five
   paired measurements, i.e. no gain at all.  ``docs/README_fastdfblock.md``
   §4.1 has the three explanations that were checked and do not account for
   the gap.  So this is **off by default**: ``FASTDF_STEP1=auto`` opts in.
   The AO index is permuted freely to cut the tiles -- ``vk = rhok rhok^T``
   comes back as ``P vk P^T`` and one scatter undoes it -- and the gate is the
   **measured tile fill**, not ``nao``: taxol in def2-TZVPP has 2636 basis
   functions against PfPMT's 2697 in def2-SVP and a tile fill of 98.9 %
   against 72.3 %, so a size threshold of the kind ``SYRK_MIN_NAO`` uses would
   switch this on exactly where it loses.

Coverage is a real ``float64`` density, ``hermi`` 0 or 1, any number of
densities, restricted or unrestricted, the cderi resident on the device or on
the host, the full-range or long-range operator, and J and K together or J
alone -- a Coulomb-only call on an existing cderi has a branch of its own
rather than a fall-through, because items 3 and 4 apply to it even though
items 1 and 2 have nothing to do there.  A complex density (the
GHF/DHF spin-free build), a second device, or one of the two asymmetric
density factorisations that only the TDDFT *gradient* and NAC drivers produce
falls through to GPU4PySCF's own ``get_jk``, so a patched run computes the
same J and K as an unpatched one.  ``stats()`` says which, because nothing
numerical can.

A **pure** functional's DF run does not reach this function at all, and that
is not a decline: with no exchange to build there is no cderi, and
``df/df.py:113`` sends the Coulomb build to ``df/j_engine_3c2e.py``, which
contracts the three-centre integrals with the density directly.  `fastdf`
reports zero calls on a DF-PBE run for that reason -- see
``docs/README_fastdf.md`` §5.

Ablations, for the tables in ``docs/README_fastdf.md``:

    FASTDF_OFF=1      hand every call back to GPU4PySCF
    FASTDF_STEP2=syrk|gemm|auto
                      the exchange contraction: force the rank-k update, force
                      the general GEMM, or (default) pick by `SYRK_MIN_NAO`
    FASTDF_SYRK_MIN_NAO=<n>
                      where `auto` switches over
    FASTDF_JPASS=blas|tensor
                      the Coulomb pass over the packed block: explicit
                      GEMV/GEMM (default) or GPU4PySCF's two `contract` calls
    FASTDF_STEP1=tensor|auto|bgemm|tiled
                      the first contraction: `tensor` (default) is
                      GPU4PySCF's cuTENSOR contraction, `auto` takes item 5's
                      block-sparse one wherever the tile fill passes the gate,
                      and `bgemm` and `tiled` force one branch each.  Items 2
                      and 5 say what the alternatives are worth and where
    FASTDF_TILE=<n>   item 5's AO tile size (default 32)
    FASTDF_TILE_ORDER=native|rcm|rcb|morton
                      the AO ordering item 5's tiles are cut in; `native` is
                      the order the cderi's own addresses use
    FASTDF_TILE_MAX_FILL=<f>
                      the tile fill above which `auto` keeps the dense
                      contraction (default 0.75)
    FASTDF_BLK=<n>    override the auxiliary block size, which GPU4PySCF
                      derives from free memory and then caps at 256
    FASTDF_RESIDENT_FRAC=<f>
                      item 3's share of free device memory; 0 turns it off
                      and streams the whole tensor, as GPU4PySCF does
"""
import os

import numpy as np
import cupy as cp
from cupy.cuda import cublas
from pyscf import lib

from gpu4pyscf.df import df_jk as DFJK
from gpu4pyscf.lib.cupy_helper import (
    contract, transpose_sum, get_avail_mem, ndarray, fill_symmetric)
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu

_OFF = bool(os.environ.get('FASTDF_OFF'))
#: 'auto' (a measured size threshold), 'syrk' or 'gemm' for the exchange
#: contraction; see `_use_syrk`.
_STEP2 = os.environ.get('FASTDF_STEP2', 'auto')
#: 'tensor' (GPU4PySCF's cuTENSOR contraction, the default), 'auto' (the
#: block-sparse contraction wherever `_use_tiled`'s fill test passes),
#: 'bgemm' (the cuBLAS batch over the AO index) or 'tiled' (the block-sparse
#: one unconditionally).  The module docstring's items 2 and 5 say what the
#: alternatives are worth and why neither is the default: item 5 wins the
#: kernel it replaces and loses the call it is part of (0.85x-1.02x at 284
#: atoms), for a 10 GB buffer held on the `DF` object.
_STEP1 = os.environ.get('FASTDF_STEP1', 'tensor')
#: AO tile size for the block-sparse contraction.  32 of 16/32/64 measured:
#: 16 has the lowest fill and the most calls and ties with 32 at 284 atoms;
#: 64 loses about ten points of fill and is behind both everywhere
#: (`docs/README_fastdfblock.md` §3).
TILE = int(os.environ.get('FASTDF_TILE', 32))
#: 'native' (the AO order the cderi's pair addresses already use, which is the
#: order the geometry file lists the atoms in) or one of the reorderings
#: `_dfblock.atom_order` knows -- 'rcm', 'rcb', 'morton'.  Measured, the best
#: reordering is worth three points of tile fill at 284 atoms and *costs* two
#: at 545, which is inside the noise of the block ratio either way.
TILE_ORDER = os.environ.get('FASTDF_TILE_ORDER', 'native')
#: Above this tile fill the block-sparse contraction is slower than the dense
#: one and `auto` keeps the dense one.  The tiled contraction runs at 0.82-0.93
#: of the dense one's flop rate and its unpack is a fancy-index scatter rather
#: than a CUDA kernel, so it needs roughly a quarter of the arithmetic removed
#: before it is ahead.  Measured break-even is a fill near 0.80
#: (`docs/README_fastdfblock.md` §3); 0.75 leaves margin.
TILE_MAX_FILL = float(os.environ.get('FASTDF_TILE_MAX_FILL', 0.75))
#: 'blas' (explicit GEMV/GEMM) or 'tensor' (GPU4PySCF's two `contract` calls)
#: for the Coulomb pass over the packed cderi.
_JPASS = os.environ.get('FASTDF_JPASS', 'blas')
_BLK = int(os.environ.get('FASTDF_BLK', 0))
#: Below this many AOs a `dsyrk` is slower than the general GEMM it replaces,
#: despite doing half the arithmetic, and `_use_syrk` keeps the GEMM.  cuBLAS
#: `dsyrk` is simply less well tuned than `dgemm`: measured against
#: `dgemm(A, A^T)` at `K = 23040` on one A100-PCIE-40GB it is 0.83x at
#: `nao = 430`, 1.39x at 1123 and 1.58x at 2000
#: (`docs/README_fastdf.md` §3.2).  `FASTDF_SYRK_MIN_NAO` moves it.
SYRK_MIN_NAO = int(os.environ.get('FASTDF_SYRK_MIN_NAO', 800))
#: Hard cap on the resident prefix, in auxiliary rows.  0 means "whatever
#: `_RESIDENT_FRAC` of the free memory allows", which is the operating
#: setting; the cap exists so `benchmarks/df_jk_bench.py --resident-fracs`
#: can measure the payoff curve at a fixed tensor size instead of needing a
#: different molecule for every point on it.
RESIDENT_MAX_ROWS = int(os.environ.get('FASTDF_RESIDENT_MAX_ROWS', 0))
#: fraction of the free device memory a host-resident cderi may take for the
#: prefix item 4 keeps on the device.  0 disables it.
_RESIDENT_FRAC = float(os.environ.get('FASTDF_RESIDENT_FRAC', 0.5))

#: GPU4PySCF's own cap on the auxiliary block size (``df/df.py:37``), kept so
#: that the default configuration differs from upstream only in what the four
#: items in the module docstring change.  ``FASTDF_BLK`` moves it and
#: ``benchmarks/df_jk_bench.py --blksizes`` sweeps it: 256 is the best of
#: 64/128/256/512 on three of the four systems measured and 128 is 6 % better
#: on the fourth, so upstream's choice stands.
MAX_BLK = 256

_orig_get_jk = DFJK.get_jk
#: where the other modules stash the function they replaced, so
#: `tests/test_patch_aliases.py` can walk for a stale reference to it
DFJK._get_jk_orig = _orig_get_jk

#: counters, so an ablation or a coverage claim can be read off a run.
STATS = {'calls': 0, 'declined': 0, 'syrk': 0, 'gemm': 0,
         'bgemm': 0, 'tensor': 0, 'tiled': 0, 'blocks': 0,
         'resident_rows': 0, 'streamed_rows': 0, 'tile_fill': 0.0,
         'blksize': 0}
DECLINED = {}


def stats():
    """What this module did in this process.  Nothing numerical can say."""
    out = dict(STATS)
    out['declined_reasons'] = dict(DECLINED)
    return out


# --------------------------------------------------------------------------
# the two cuBLAS calls
# --------------------------------------------------------------------------
_ONE = np.array(1.0)
_ZERO = np.array(0.0)


def _rhok(work, factor, out, nL, nao, nocc, ld):
    """``out[i,L,k] = sum_j work[i,j,L] factor[j,k]``, one batched GEMM.

    Reached only under ``FASTDF_STEP1=bgemm``: measured, cuTENSOR does this
    contraction better (module docstring item 2), and the module keeps it
    because a rejected alternative that can still be run is worth more than
    one described in a document.

    ``work`` is GPU4PySCF's own unpacked buffer, ``(nao, nao, blksize)``
    C-contiguous, so ``work[i]`` is a row-major ``(j, L)`` matrix with row
    stride ``ld`` = the *allocated* block size, which the last block does not
    fill.  Batching over **i** rather than over ``L`` is what makes every
    operand a proper matrix: ``work[i]`` has a unit stride in ``L``, the
    factor has one in ``k``, and the answer lands in ``(i, L, k)`` order --
    which is the layout the rank-k update below contracts over.  Batching over
    ``L`` instead, which is the shape the arithmetic first suggests, cannot be
    a cuBLAS call at all: ``work[:,:,L]`` has a unit stride in neither index.

    Row-major ``C_i = A_i^T B`` with ``A_i = work[i]`` is, in cuBLAS's column
    major, ``C_i^cm = B^cm · (A_i^cm)^T`` with ``m = nocc``, ``n = nL``,
    ``k = nao``.
    """
    handle = cp.cuda.device.get_cublas_handle()
    cublas.dgemmStridedBatched(
        handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
        nocc, nL, nao,
        _ONE.ctypes.data,
        factor.data.ptr, nocc, 0,
        work.data.ptr, ld, nao * ld,
        _ZERO.ctypes.data,
        out.data.ptr, nocc, nL * nocc,
        nao)
    return out


def _syrk(rhok, vk, nao, K):
    """``vk += rhok rhok^T`` into one triangle; ``rhok`` is ``(nao, K)``."""
    handle = cp.cuda.device.get_cublas_handle()
    cublas.dsyrk(handle, cublas.CUBLAS_FILL_MODE_LOWER, cublas.CUBLAS_OP_T,
                 nao, K, _ONE.ctypes.data, rhok.data.ptr, K,
                 _ONE.ctypes.data, vk.data.ptr, nao)


def _gemm_nt(a, b, vk, nao, K):
    """``vk += a b^T``; both operands are ``(nao, K)`` row-major."""
    handle = cp.cuda.device.get_cublas_handle()
    # row-major C += A B^T  <=>  column-major C^T += B A^T
    cublas.dgemm(handle, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N,
                 nao, nao, K, _ONE.ctypes.data,
                 b.data.ptr, K, a.data.ptr, K,
                 _ONE.ctypes.data, vk.data.ptr, nao)


def _vj_pass(cderi_blk, dm_sparse, vj_sparse, auxvec, nL, npairs, n_dm):
    """The Coulomb pass over one packed block, as cuBLAS rather than cuTENSOR.

    ``auxvec[n,L] = sum_p dm[n,p] cderi[L,p]`` and then
    ``vj[n,p] += sum_L auxvec[n,L] cderi[L,p]``: two passes over the same
    block, and on a converged density the whole Coulomb build is those two and
    nothing else.  The block is a plain matrix in both of the layouts it
    arrives in -- the auxiliary index is contiguous when the tensor lives on
    the device (``_cholesky_eri`` returns a transpose) and the pair index is
    when it has been staged from the host -- so both are one ``dgemv`` per
    pass at a single density and one ``dgemm`` at several.  GPU4PySCF issues
    them as ``contract('np,Lp->nL')`` and ``contract('nL,Lp->np')``; this is
    the same arithmetic and the same operands.
    """
    handle = cp.cuda.device.get_cublas_handle()
    ptr = cderi_blk.data.ptr
    # which of the two indices is contiguous decides only the transpose flags:
    # column-major, the block is (nL x npairs) one way round and its transpose
    # the other, and the leading dimension is whichever stride is not one.
    aux_major = cderi_blk.strides[0] == 8
    lda = (cderi_blk.strides[1] if aux_major else
           cderi_blk.strides[0]) // 8
    op1 = cublas.CUBLAS_OP_N if aux_major else cublas.CUBLAS_OP_T
    op2 = cublas.CUBLAS_OP_T if aux_major else cublas.CUBLAS_OP_N
    if n_dm == 1:
        m, n = (nL, npairs) if aux_major else (npairs, nL)
        cublas.dgemv(handle, op1, m, n, _ONE.ctypes.data, ptr, lda,
                     dm_sparse.data.ptr, 1, _ZERO.ctypes.data,
                     auxvec.data.ptr, 1)
        cublas.dgemv(handle, op2, m, n, _ONE.ctypes.data, ptr, lda,
                     auxvec.data.ptr, 1, _ONE.ctypes.data,
                     vj_sparse.data.ptr, 1)
        return
    cublas.dgemm(handle, op1, cublas.CUBLAS_OP_N, nL, n_dm, npairs,
                 _ONE.ctypes.data, ptr, lda, dm_sparse.data.ptr, npairs,
                 _ZERO.ctypes.data, auxvec.data.ptr, nL)
    cublas.dgemm(handle, op2, cublas.CUBLAS_OP_N, npairs, n_dm, nL,
                 _ONE.ctypes.data, ptr, lda, auxvec.data.ptr, nL,
                 _ONE.ctypes.data, vj_sparse.data.ptr, npairs)


def _fill_from_triangle(vk):
    """One triangle of a symmetric result -> the whole matrix, in place.

    ``dsyrk`` wrote one triangle and left the other at the zero it started
    from, so ``a + a^T`` is the answer everywhere except the diagonal, which
    it doubles.  In place, because at a Hessian's ``n_dm`` this array is
    several GB and a copy of it is a second several GB.
    """
    shape = vk.shape
    nao = shape[-1]
    flat = transpose_sum(vk.reshape(-1, nao, nao), inplace=True, hermi=1)
    idx = cp.arange(nao)
    flat[:, idx, idx] *= .5
    return flat.reshape(shape)


# --------------------------------------------------------------------------
def _decline(reason):
    STATS['declined'] += 1
    DECLINED[reason] = DECLINED.get(reason, 0) + 1
    return reason


def _use_syrk(nao):
    """Whether the symmetric rank-k update is the faster of the two here.

    It always does half the arithmetic of the GEMM it replaces, and it is not
    always faster for it: cuBLAS's `dsyrk` is less well tuned than its `dgemm`
    and loses the factor of two back below about a thousand basis functions.
    A fixed threshold rather than a runtime probe, because the alternative is
    a measurement inside a hot path; `benchmarks/df_jk_bench.py` is what moved
    it and `FASTDF_STEP2` forces either branch.
    """
    if _STEP2 == 'syrk':
        return True
    if _STEP2 == 'gemm':
        return False
    return nao >= SYRK_MIN_NAO


def _blksize(nao, nocc, nbuf, on_gpu, naux_slice, fill=1.0, extra=0):
    """How many auxiliary functions one pass may unpack at a time.

    The dense block is ``blk*nao^2`` words and each ``rhok`` is
    ``blk*nao*nocc``.  The cap is GPU4PySCF's own (``df/df.py:37``), kept so
    that the default configuration differs from upstream only in what the four
    items in the module docstring change; measured, 256 is the best of
    64/128/256/512 on three of the four systems in
    ``docs/README_fastdf.md`` §3.2 and 128 is 6 % better on the fourth.
    """
    if _BLK:
        return max(1, min(_BLK, naux_slice))
    mem = get_avail_mem()
    frac = .4 if on_gpu else .2
    words = int(mem * frac / 8) - extra
    blk = max(words, 0) // (int(fill * nao * nao) + nbuf * nao * nocc)
    blk = min(max(blk, 1), MAX_BLK, naux_slice)
    return blk


def _resident_prefix(dfobj, cderi_sparse, reserve_words):
    """As many auxiliary rows of a host-resident cderi as the device can hold.

    Built once per density-fitting object and reused by every later call, so
    what it removes is a PCIe copy per SCF cycle rather than a one-off one.
    ``reserve_words`` is what this call still has to allocate on top of it.
    """
    naux_slice = cderi_sparse.shape[0]
    buf = None
    if _RESIDENT_FRAC > 0:
        # keyed on the tensor object itself rather than on its id: a Hessian's
        # CPHF moves the cderi to the host mid-run
        # (`df/hessian/rhf.py:1273`), and `reset()` replaces it, so a cache
        # that could not tell the two apart would hand back rows of the wrong
        # tensor
        cached = getattr(dfobj, '_fastdf_resident', None)
        if cached is not None and cached[0] is cderi_sparse:
            buf = cached[1]
        else:
            npairs = cderi_sparse.shape[1]
            words = int(get_avail_mem() * _RESIDENT_FRAC / 8) - reserve_words
            rows = min(naux_slice, max(0, words // npairs))
            if RESIDENT_MAX_ROWS:
                rows = min(rows, RESIDENT_MAX_ROWS)
            if rows >= 8:
                try:
                    buf = cp.empty((rows, npairs))
                    buf.set(np.ascontiguousarray(cderi_sparse[:rows]))
                except Exception:                           # noqa: BLE001
                    buf = None
            dfobj._fastdf_resident = (cderi_sparse, buf)
    STATS['resident_rows'] = 0 if buf is None else int(buf.shape[0])
    STATS['streamed_rows'] = naux_slice - STATS['resident_rows']
    return buf


def _tile_fill(dfobj, nao):
    """What fraction of ``nao x nao`` the tiled pattern would cover.

    This is the whole of `_use_tiled`'s decision and it is cheap -- a
    ``unique`` over the pair list, no allocation and no device work -- so it
    is measured rather than guessed from the size.  Guessing from the size is
    specifically wrong: taxol in def2-TZVPP has 2636 basis functions, within
    three per cent of PfPMT's 2697 in def2-SVP, and a tile fill of **98.9 %**
    against PfPMT's 72.3 % (`docs/README_fastdfblock.md` §2.1).
    """
    from electrolites import _dfblock
    key = (dfobj._cderi_idx[0], TILE, TILE_ORDER)
    cached = getattr(dfobj, '_fastdf_tile_fill', None)
    if (cached is not None and cached[0][0] is key[0]
            and cached[0][1:] == key[1:]):
        return cached[1]
    addr = dfobj._cderi_idx[0]
    addr = addr.get() if hasattr(addr, 'get') else np.asarray(addr)
    rows, cols = addr // nao, addr % nao
    if TILE_ORDER != 'native':
        perm = _dfblock.ao_perm(
            dfobj.mol, _dfblock.atom_order(dfobj.mol, rows, cols, TILE_ORDER))
        rows, cols = perm[rows], perm[cols]
    fill = _dfblock.tile_stats(rows, cols, nao, TILE)['fill']
    dfobj._fastdf_tile_fill = (key, fill)
    return fill


def _use_tiled(dfobj, nao):
    """Whether the block-sparse first contraction is the faster one here."""
    if _STEP1 == 'tiled':
        return True
    if _STEP1 != 'auto':
        return False
    fill = _tile_fill(dfobj, nao)
    STATS['tile_fill'] = fill
    return fill <= TILE_MAX_FILL


def _tiles(dfobj, nao, blksize):
    """The tile pattern of this cderi, built once and kept on the object.

    Keyed on the pair-address array itself rather than on ``id(dfobj)``:
    ``reset()`` replaces the index array and ``range_coulomb`` hands a
    *copy* of the object a different one, and a cache that could not tell
    those apart would scatter one tensor's block through another's
    destinations.

    **The block size is fixed at the first call and the caller takes it
    back**, rather than being recomputed per call the way the dense path's
    is.  The buffer this holds is 10.8 GB at 284 atoms; ``_blksize`` reads
    ``get_avail_mem()``, which moves by a few per cent between calls, and
    re-deriving the block each time would reallocate and re-zero all of it
    for a block that differs by one.  The destination tables themselves do
    not depend on the block size -- they are offsets in units of whole ``L``
    rows -- so nothing else has to be rebuilt either way.
    """
    from electrolites._dfblock import RowCompressed
    key = (dfobj._cderi_idx[0], TILE, TILE_ORDER)
    cached = getattr(dfobj, '_fastdf_tiles', None)
    if (cached is None or cached[0][0] is not key[0]
            or cached[0][1:] != key[1:]):
        rc = RowCompressed.for_dfobj(dfobj, nao, TILE, blksize,
                                     order=TILE_ORDER).to_device()
        dfobj._fastdf_tiles = (key, rc)
    else:
        rc = cached[1]
    STATS['tile_fill'] = rc.fill
    STATS['blksize'] = rc.ldL
    return rc


def _blocks(cderi_sparse, blksize, on_gpu, resident=None):
    """Yield ``(block, nL)`` with the block on the device.

    On a host-resident cderi the rows `resident` already holds are yielded
    without a copy, and the rest keep GPU4PySCF's double-buffered prefetch:
    the copy of block ``n+1`` runs on a second stream while block ``n`` is
    being contracted.
    """
    naux_slice = cderi_sparse.shape[0]
    if on_gpu:
        for p0, p1 in lib.prange(0, naux_slice, blksize):
            yield cderi_sparse[p0:p1], p1 - p0
        return

    start = 0
    if resident is not None:
        start = int(resident.shape[0])
        for p0, p1 in lib.prange(0, start, blksize):
            yield resident[p0:p1], p1 - p0
    if start >= naux_slice:
        return

    npairs = cderi_sparse.shape[1]
    nstream = min(blksize, naux_slice - start)
    buf = cp.empty((nstream, npairs))
    buf_prefetch = cp.empty_like(buf)
    comput_stream = cp.cuda.get_current_stream()
    compute_event = cp.cuda.Event()
    io_stream = cp.cuda.stream.Stream(non_blocking=True)
    io_event = cp.cuda.Event()

    buf_prefetch[:nstream].set(cderi_sparse[start:start + nstream],
                               stream=io_stream)
    io_event.record(io_stream)

    for p0, p1 in lib.prange(start, naux_slice, blksize):
        compute_event.record(comput_stream)
        buf, buf_prefetch = buf_prefetch, buf
        comput_stream.wait_event(io_event)
        p2 = min(naux_slice, p1 + blksize)
        if p1 < p2:
            io_stream.wait_event(compute_event)
            buf_prefetch[:p2 - p1].set(cderi_sparse[p1:p2], stream=io_stream)
            io_event.record(io_stream)
        yield buf[:p1 - p0], p1 - p0


def _sparse_dm(dms_3d, pair_addresses, diags, nao, hermi):
    """The density gathered onto the stored pair list, as the Coulomb pass
    wants it.  GPU4PySCF's arithmetic, including the factor two on the
    off-diagonal pairs that the packed form stands in for."""
    rows, cols = divmod(cp.asarray(pair_addresses), nao)
    dm_sparse = dms_3d[:, rows, cols]
    if hermi == 0:
        dm_sparse += dms_3d[:, cols, rows]
    else:
        dm_sparse *= 2
    dm_sparse[:, diags] *= .5
    return cp.ascontiguousarray(dm_sparse), rows, cols


def _get_j(dfobj, dms, hermi, omega):
    """A Coulomb-only call on an existing cderi.

    Items 3 and 4 apply to it and items 1 and 2 do not: there is no exchange
    tensor and nothing is unpacked.  GPU4PySCF reaches this when a hybrid has
    already built the cderi and something then asks for J alone; a *pure*
    functional never gets here at all, because with no cderi
    ``df/df.py:113`` sends it to the three-centre J engine instead.
    """
    out_cupy = isinstance(dms, cp.ndarray)
    nao = dms.shape[-1]
    dms_3d = cp.asarray(dms).reshape(-1, nao, nao)
    n_dm = dms_3d.shape[0]
    device_id = cp.cuda.Device().id
    cderi_sparse = dfobj._cderi[device_id]
    naux_slice, npairs = cderi_sparse.shape
    on_gpu = isinstance(cderi_sparse, cp.ndarray)
    pair_addresses, diags = dfobj._cderi_idx
    dm_sparse, rows, cols = _sparse_dm(dms_3d, pair_addresses, diags, nao,
                                       hermi)
    vj_sparse = cp.zeros_like(dm_sparse)

    # nothing here allocates the dense `(nao, nao, blk)` buffer the exchange
    # build needs, so the block this path can afford is set by the staging
    # buffers alone and is much larger
    if _BLK:
        blksize = max(1, min(_BLK, naux_slice))
    else:
        words = int(get_avail_mem() * (.4 if on_gpu else .2) / 8)
        per = n_dm + (0 if on_gpu else 2 * npairs)
        blksize = max(1, min(MAX_BLK, naux_slice, words // max(per, 1)))
    auxbuf = cp.empty(n_dm * blksize)
    resident = None
    if not on_gpu:
        resident = _resident_prefix(dfobj, cderi_sparse,
                                    auxbuf.size + 2 * blksize * npairs)
    for cderi_blk, nL in _blocks(cderi_sparse, blksize, on_gpu, resident):
        STATS['blocks'] += 1
        if _JPASS == 'blas':
            _vj_pass(cderi_blk, dm_sparse, vj_sparse,
                     ndarray((n_dm, nL), buffer=auxbuf), nL, npairs, n_dm)
        else:
            auxvec = contract('np,Lp->nL', dm_sparse, cderi_blk)
            contract('nL,Lp->np', auxvec, cderi_blk, beta=1, out=vj_sparse)

    vj = cp.zeros_like(dms_3d)
    vj[:, cols, rows] = vj[:, rows, cols] = vj_sparse
    vj = vj.reshape(dms.shape)
    if not out_cupy:
        vj = vj.get()
    return vj, None


def get_jk(dfobj, dms, hermi=0, with_j=True, with_k=True, omega=None):
    """``df/df_jk.py:get_jk`` with its three stages rearranged."""
    if _OFF:
        _decline('FASTDF_OFF')
        return _orig_get_jk(dfobj, dms, hermi, with_j, with_k, omega=omega)
    if multi_gpu.num_devices != 1:
        _decline('more than one device')
        return _orig_get_jk(dfobj, dms, hermi, with_j, with_k, omega=omega)
    if getattr(dms, 'dtype', None) != np.float64:
        # the GHF/DHF spin-free build hands this function a complex density
        _decline('complex density')
        return _orig_get_jk(dfobj, dms, hermi, with_j, with_k, omega=omega)

    log = logger.new_logger(dfobj.mol, dfobj.verbose)
    t0 = log.init_timer()
    if dfobj._cderi is None:
        dfobj.build(omega=omega)
    if dfobj._cderi[cp.cuda.Device().id].shape[0] == 0:
        _decline('empty cderi slice')
        return _orig_get_jk(dfobj, dms, hermi, with_j, with_k, omega=omega)
    if not with_k:
        STATS['calls'] += 1
        return _get_j(dfobj, dms, hermi, omega)

    out_cupy = isinstance(dms, cp.ndarray)
    factor_l, factor_r = DFJK.factorize_dm(dms, hermi)
    symmetrize = getattr(dms, 'symmetrize', 0)

    nspin = 1
    if factor_l.ndim == 4:
        assert dms.ndim == 4
        nspin = 2

    if factor_r is None:
        mode = 0
    elif factor_l.ndim == factor_r.ndim:
        mode = 1
    else:
        # `_aggregate_dm_factor_l` stacks one factor against many of the other;
        # only the TDDFT gradient and NAC drivers produce it
        _decline('asymmetric density factorisation')
        return _orig_get_jk(dfobj, dms, hermi, with_j, with_k, omega=omega)

    nao, nocc = factor_l.shape[-2:]
    dms_3d = cp.asarray(dms).reshape(-1, nao, nao)
    n_dm = dms_3d.shape[0] // nspin
    if nocc == 0:
        return dms, dms

    device_id = cp.cuda.Device().id
    cderi_sparse = dfobj._cderi[device_id]
    naux_slice = cderi_sparse.shape[0]
    STATS['calls'] += 1
    on_gpu = isinstance(cderi_sparse, cp.ndarray)
    pair_addresses, diags = dfobj._cderi_idx
    pair_idx = cp.asarray(pair_addresses, dtype=np.int32)

    use_syrk = (mode == 0) and _use_syrk(nao)
    use_bgemm = _STEP1 == 'bgemm'
    use_tiled = _use_tiled(dfobj, nao)
    STATS['syrk' if use_syrk else 'gemm'] += 1
    STATS['tiled' if use_tiled else
          ('bgemm' if use_bgemm else 'tensor')] += 1

    # --- J: GPU4PySCF's packed contraction, unchanged ---------------------
    rows = cols = dm_sparse = vj_sparse = None
    if with_j:
        dm_sparse, rows, cols = _sparse_dm(dms_3d, pair_addresses, diags,
                                           nao, hermi)
        vj_sparse = cp.zeros_like(dm_sparse)

    fl = cp.asarray(factor_l).reshape(nspin, n_dm, nao, nocc)
    fr = None if factor_r is None else \
        cp.asarray(factor_r).reshape(nspin, n_dm, nao, nocc)

    # one `rhok` per density factor.  The cuBLAS path holds one density at a
    # time, the cuTENSOR ablation all of them at once, which is what makes the
    # block that fits so much smaller there.
    nbuf = 1 if mode == 0 else 2
    nbuf_live = nbuf if use_bgemm else nbuf * n_dm
    fill, extra = 1.0, 0
    if use_tiled:
        # the tiled buffer is `fill` times the dense one, so the block that
        # fits is correspondingly larger.  Against that, the gathered density
        # factor is `fill * nao / TILE` times as tall as the factor itself and
        # does not shrink with the block -- at 545 atoms and tile 32 it is as
        # large as the buffer, which is why it is subtracted from the budget
        # rather than divided into it.
        fill = _tile_fill(dfobj, nao)
        extra = int(fill * nao * nao / TILE) * nocc * nbuf * nspin * n_dm
    blksize = _blksize(nao, nocc, nbuf_live, on_gpu, naux_slice, fill, extra)
    tiles = None
    if use_tiled:
        # the pattern owns its block size; see `_tiles`
        tiles = _tiles(dfobj, nao, blksize)
        blksize = tiles.ldL
    vk = cp.zeros((nspin, n_dm, nao, nao))
    log.debug1('fastdf: nao=%d nocc=%d n_dm=%d nspin=%d blksize=%d mode=%d '
               'syrk=%s bgemm=%s', nao, nocc, n_dm, nspin, blksize, mode,
               use_syrk, use_bgemm)

    work = None if use_tiled else cp.zeros((nao, nao, blksize))
    rbuf = cp.empty(nbuf_live * blksize * nao * nocc)
    if use_tiled:
        fl = tiles.factor(fl.reshape(-1, nao, nocc)).reshape(
            nspin, n_dm, -1, nocc)
        if fr is not None:
            fr = tiles.factor(fr.reshape(-1, nao, nocc)).reshape(
                nspin, n_dm, -1, nocc)
    auxbuf = cp.empty(n_dm * nspin * blksize) if with_j else None

    resident = None
    if not on_gpu:
        reserve = ((tiles.buf.size if use_tiled else work.size) + rbuf.size
                   + 2 * blksize * cderi_sparse.shape[1])
        resident = _resident_prefix(dfobj, cderi_sparse, reserve)

    for cderi_blk, nL in _blocks(cderi_sparse, blksize, on_gpu, resident):
        STATS['blocks'] += 1
        if with_j:
            if _JPASS == 'blas':
                _vj_pass(cderi_blk, dm_sparse, vj_sparse,
                         ndarray((n_dm * nspin, nL), buffer=auxbuf),
                         nL, cderi_sparse.shape[1], n_dm * nspin)
            else:
                auxvec = contract('np,Lp->nL', dm_sparse, cderi_blk)
                contract('nL,Lp->np', auxvec, cderi_blk, beta=1,
                         out=vj_sparse)

        # GPU4PySCF's own unpack kernel, into GPU4PySCF's own layout.  Writing
        # `(L,i,j)` instead was tried and is 2.65x slower here -- the
        # transposed half of a symmetric fill is coalesced in that layout and
        # is not in this one; `docs/README_fastdf.md` §3.4 has the measurement
        # and why the layout turned out not to be needed.
        if use_tiled:
            # `_dfblock` says why the AO index may be permuted and where it
            # comes back; everything between here and `unpermute` below is in
            # the reordered basis, including `vk`.
            tiles.unpack(cderi_blk, nL)
            for s in range(nspin):
                rl = ndarray((n_dm, nao, nocc, nL), buffer=rbuf)
                tiles.step1(fl[s], rl, nL)
                if mode == 0:
                    if use_syrk:
                        for i in range(n_dm):
                            _syrk(rl[i], vk[s, i], nao, nocc * nL)
                    else:
                        contract('nikL,njkL->nij', rl, rl, beta=1, out=vk[s])
                else:
                    rr = ndarray((n_dm, nao, nocc, nL),
                                 buffer=rbuf[n_dm * nao * nocc * blksize:])
                    tiles.step1(fr[s], rr, nL)
                    contract('nikL,njkL->nij', rl, rr, beta=1, out=vk[s])
            continue

        dense = fill_symmetric(cderi_blk.T, pair_idx, nao, 0, nL,
                               out=work[:, :, :nL])
        if use_bgemm:
            K = nL * nocc
            rl = ndarray((nao, K), buffer=rbuf)
            rr = ndarray((nao, K), buffer=rbuf[blksize * nao * nocc:]) \
                if mode else None
            for s in range(nspin):
                for i in range(n_dm):
                    _rhok(work, fl[s, i], rl, nL, nao, nocc, blksize)
                    if use_syrk:
                        _syrk(rl, vk[s, i], nao, K)
                    else:
                        if mode:
                            _rhok(work, fr[s, i], rr, nL, nao, nocc, blksize)
                        _gemm_nt(rl, rr if mode else rl, vk[s, i], nao, K)
        else:
            cderi = dense.transpose(2, 0, 1)
            rl = ndarray((n_dm, nao, nocc, nL), buffer=rbuf)
            for s in range(nspin):
                contract('Lij,njk->nikL', cderi, fl[s], out=rl)
                if mode == 0:
                    if use_syrk:
                        for i in range(n_dm):
                            _syrk(rl[i], vk[s, i], nao, nocc * nL)
                    else:
                        contract('nikL,njkL->nij', rl, rl, beta=1, out=vk[s])
                else:
                    rr = ndarray((n_dm, nao, nocc, nL),
                                 buffer=rbuf[n_dm * nao * nocc * blksize:])
                    contract('Lij,njk->nikL', cderi, fr[s], out=rr)
                    contract('nikL,njkL->nij', rl, rr, beta=1, out=vk[s])

    if use_syrk:
        vk = _fill_from_triangle(vk)
    if use_tiled:
        vk = tiles.unpermute(vk)

    vj = None
    if with_j:
        vj = cp.zeros_like(dms_3d)
        vj[:, cols, rows] = vj[:, rows, cols] = vj_sparse
        vj = vj.reshape(dms.shape)
        if not out_cupy:
            vj = vj.get()

    if symmetrize != 0:
        vk = transpose_sum(vk.reshape(-1, nao, nao), hermi=symmetrize)
    vk = vk.reshape(dms.shape)
    if not out_cupy:
        vk = vk.get()
    log.timer_debug1('fastdf vj and vk', *t0)
    return vj, vk


DFJK.get_jk = get_jk
