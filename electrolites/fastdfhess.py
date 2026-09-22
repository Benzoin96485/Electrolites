"""The DF Hessian's own CPHF build -- the fifth two-electron code path.

``docs/DF_WORK_LIST.md`` **W8**.  ``df/hessian/rhf.py:_get_jk`` builds J and K
for the CPHF right-hand sides directly against the cderi and **does not go
through** ``df/df_jk.py:get_jk``, so `fastdf` -- which covers the DF SCF, the
DF response solve and a DF Hessian's ``get_veff`` with one patch -- is not on
it.  ``docs/README_fastdfhess.md`` is the round, and its headline is a
negative one that this module exists in spite of rather than because of:

**The function cannot be made faster.**  It runs at **15.2 TF/s of an
A100-SXM4-80GB's 19.5** -- 78 % of fp64 peak -- its dominant contraction at
**81 %**, and every one of its four contractions is within **1.14x** of a
plain ``dgemm`` at its own shape, two of them *ahead* of one.  Three
rearrangements were implemented and measured and all three are losses:
0.92x, 0.97x and 0.99x at 42 atoms.  And it is **15 % of a density-fitted
Hessian** to begin with (24 % after ``patch_all()``), so even removing it
entirely would be 1.18x.  All three of those numbers are in
``docs/README_fastdfhess.md`` §2-§4, with the shapes and the roofline.

**What this module is for is that the function does not run at all past a
certain shape.**  GPU4PySCF sizes the density batch from free memory alone
and hands the whole batch to one cuTENSOR contraction; past about 1.3e9
elements of output ``cutensor.createPlan`` refuses
``Lpq,snqi->snpiL`` with ``CUTENSOR_STATUS_NOT_SUPPORTED``.  A **42-atom
def2-TZVPP** DF Hessian and a **113-atom def2-SVP** one both die there, on
an 80 GB card and on a 40 GB one, with tens of GB free -- it is the
descriptor that is rejected, not the workspace that is missing.  Capping the
batch makes both run.

The rest is bookkeeping that costs nothing and occasionally matters.  The
integrals, the metric, the screening and every symmetry factor are
GPU4PySCF's, and so is the flop count -- term for term, not just in total.

1. **The density batch is capped at a shape cuTENSOR will plan**, with a
   halve-and-retry behind the cap because a rejected plan raises before any
   operand is read.  This is the item above and the only one worth a factor
   of anything.

2. **The exchange arrangement.**  ``upstream`` (the default) is GPU4PySCF's
   own ordering.  ``fused`` groups the algebra by the perturbed density --
   ``vk[n] = sum_L M_L (D1_n M_L C_o)`` -- so the several-GB intermediate is
   written once and read once instead of written once and read twice, and
   the ``nao^2 nocc`` contraction writes 40 MB instead of a second
   several-GB array.  ``blasvk`` keeps upstream's ordering but gives the two
   contractions that write into ``vk`` one accumulator each in the layout
   their own GEMM produces.  Both are **slower**, measured, at every size
   tried; they are kept because a rejected alternative that can be re-run is
   worth more than one described in prose.

3. **The auxiliary vector of the Coulomb pass is a contraction, not a
   diagonal.**  Upstream reads ``cp.einsum('sniiL->nL', rhok1_oo)``, one
   element in ``nocc`` out of an array of several hundred MB; the same
   vector is ``sum_{q,i} rhok[q,i,L] X[n,q,i]``, which also makes the
   Coulomb half independent of the exchange intermediate.

4. **The Coulomb-only call is an explicit GEMV/GEMM pair**, which is
   `fastdf`'s item 3 on the one path that module cannot reach.  A pure
   functional's DF Hessian takes this branch and no other.  Worth 1.12x-1.42x
   at the small end and nothing at 42 atoms.

5. **The density batches are balanced** with ``_balance_batch_size``,
   upstream's own helper, defined forty lines below the function that needs
   it and unused there -- ``n_dm = 126`` at ``dm_batch_size = 55`` runs
   passes of 55, 55 and then 16.  On a card large enough to take the whole
   right-hand side in one pass this is worth nothing, which on an A100 is
   every size measured.

6. **The batch size stops depending on the allocator's history.**  Upstream
   derives it from ``get_avail_mem(exclude_memory_pool=True)``, which does
   not count the blocks CuPy's pool is holding but not using; the
   pool-inclusive figure is what ``DF.get_blksize`` twenty lines away
   already uses.  Measured, the batch upstream picks is stable across a
   CPHF's calls anyway.

7. **The cderi is moved to the host against what the Krylov solve will
   actually need.**  Upstream moves it whenever it is more than **0.4** of
   the free memory, with the comment "Leave space for storing Krylov
   subspace vectors" (``df/hessian/rhf.py:1258-1273``), and that number does
   not know what the subspace costs.  Measured
   (``benchmarks/df_hess_krylov_probe.py``): a CPHF converges in **five**
   Krylov cycles and the subspace is **0.37 GB** at 42 atoms, against a
   0.4-of-free-memory reserve of 31 GB and a cderi of 0.99 GB.  It also
   turns out not to matter for the exchange call -- with the tensor on the
   host that call is 1017 ms against 1024 ms on the device, because the
   prefetch hides the whole stream behind the contractions -- but a
   Coulomb-only call is **9.0x** slower on the host (81.8 ms against 9.1),
   and that is the branch a pure functional takes.

Coverage is a real ``float64`` cderi, restricted or unrestricted, J and K
together or J alone, the full-range or the long-range operator, and the
tensor resident on the device or on the host.  A second device, a density
without the ``mo1``/``occ_coeff`` tags the CPHF attaches, or ``hermi != 1``
falls through to GPU4PySCF's own function, so a patched Hessian computes the
same Hessian as an unpatched one.  ``stats()`` says which, because nothing
numerical can.

Ablations, for the tables in ``docs/README_fastdfhess.md``:

    FASTDFHESS_OFF=1        hand every call back to GPU4PySCF
    FASTDFHESS_ARRANGE=upstream|fused|blasvk
                            item 2: GPU4PySCF's ordering (default, and the
                            measured best), or either of the two that lose
    FASTDFHESS_MAX_PLAN=<n> item 1: the largest intermediate handed to one
                            cuTENSOR contraction, in elements; 0 disables
                            the cap and leaves only the retry
    FASTDFHESS_JPASS=blas|tensor
                            item 4, or GPU4PySCF's two `contract` calls
    FASTDFHESS_BALANCE=0    item 5 off: uneven batches, as upstream does
    FASTDFHESS_POOLMEM=0    item 6 off: size the batches from
                            `cudaMemGetInfo`, as upstream does
    FASTDFHESS_HOST=auto|<f>
                            item 7: the reserve-aware policy (default) or a
                            fraction of free memory, upstream's being 0.4
    FASTDFHESS_KRYLOV_CYCLES=<n>
                            how many cycles item 7 reserves for; 5 measured,
                            8 the default
    FASTDFHESS_BLK=<n>      override the auxiliary block size (256 is
                            upstream's cap and is the best of 64/128/256/512
                            measured)
    FASTDFHESS_DM_BATCH=<n> override the density batch size
"""
import ctypes
import os

import numpy as np
import cupy as cp
from cupy.cuda import cublas
from pyscf import lib

from gpu4pyscf.df.hessian import rhf as DFHESS
from gpu4pyscf.lib.cupy_helper import (
    contract, ndarray, get_avail_mem, fill_symmetric, empty_mapped)
from gpu4pyscf.lib import multi_gpu

from electrolites import _dffp32 as FP32

_OFF = bool(os.environ.get('FASTDFHESS_OFF'))
#: 'upstream' (GPU4PySCF's ordering, which is the **measured** best of the
#: three -- see `docs/README_fastdfhess.md` 3), 'fused' or 'blasvk'.  The
#: other two are kept because they are runnable: a rejected alternative that
#: can be re-measured is worth more than one described in prose.
ARRANGE = os.environ.get('FASTDFHESS_ARRANGE', 'upstream')
#: whether to even out the density batches with upstream's own helper.
BALANCE = os.environ.get('FASTDFHESS_BALANCE', '1') != '0'
#: whether to size the batches from the pool-inclusive free memory.
POOLMEM = os.environ.get('FASTDFHESS_POOLMEM', '1') != '0'
_BLK = int(os.environ.get('FASTDFHESS_BLK', 0))
_DM_BATCH = int(os.environ.get('FASTDFHESS_DM_BATCH', 0))
#: 'blas' (explicit GEMV/GEMM) or 'tensor' (GPU4PySCF's two `contract` calls).
_JPASS = os.environ.get('FASTDFHESS_JPASS', 'blas')
#: 'auto' for the reserve-aware host policy, or a float -- upstream's is 0.4.
HOST = os.environ.get('FASTDFHESS_HOST', 'auto')
#: How many Krylov cycles to reserve for under `HOST=auto`.  `solve_mo1`
#: passes `max_cycle=50`, but **measured** -- `benchmarks/df_hess_krylov_probe.py`
#: -- a converged CPHF takes *five*, at 10 atoms and at 42 alike, and the
#: subspace is then 0.37 GB where the cderi is 0.99 GB.  8 leaves margin
#: without reserving an order of magnitude more than the solve uses.
KRYLOV_CYCLES = int(os.environ.get('FASTDFHESS_KRYLOV_CYCLES', 8))

#: The largest ``nspin * dm_batch * nao * nocc * blksize`` this module will
#: hand to one cuTENSOR contraction.  **Not a memory limit** -- past about
#: this size ``cutensor.createPlan`` returns ``CUTENSOR_STATUS_NOT_SUPPORTED``
#: for ``Lpq,snqi->snpiL`` and the call fails with tens of GB free, which is
#: what stops GPU4PySCF running a 42-atom def2-TZVPP or a 113-atom def2-SVP
#: DF Hessian at all.  `benchmarks/df_hess_cutensor_limit.py` is what placed
#: it, and the boundary is **not** a clean function of the shape: bisected,
#: the largest planable ``dm_batch`` is 530 at (nao 430, nocc 92), 224 at
#: (478, 39), 22 at (1007, 92) and 8 at (1123, 226), which is 5.4e9, 1.1e9,
#: 5.2e8 and 5.2e8 output elements.  5e8 is under all four, and it costs
#: nothing: at 42 atoms the batch sweep reads 1083 ms at 48 densities against
#: 1094 at 126.  The halve-and-retry in `_jk` is the real safety net.
MAX_PLAN_ELEMENTS = float(os.environ.get('FASTDFHESS_MAX_PLAN', 5.0e8))

try:                                         # cutensor is optional upstream
    from cupy_backends.cuda.libs.cutensor import CuTensorError
    _PLAN_ERRORS = (CuTensorError, cp.cuda.memory.OutOfMemoryError)
except ImportError:                                           # pragma: no cover
    _PLAN_ERRORS = (cp.cuda.memory.OutOfMemoryError,)

_orig_get_jk = DFHESS._get_jk
#: where `tests/test_patch_aliases.py` looks for the function this replaced
DFHESS._hess_get_jk_orig = _orig_get_jk

#: counters, so a coverage claim can be read off a run rather than assumed.
STATS = {'calls': 0, 'declined': 0, 'jk': 0, 'j_only': 0, 'blocks': 0,
         'blksize': 0, 'dm_batch': 0, 'n_dm': 0, 'builds': 0,
         'moved_to_host': 0, 'kept_on_device': 0, 'arrange': '',
         'plan_retries': 0}
DECLINED = {}

_ONE = np.array(1.0)
_ZERO = np.array(0.0)


def stats():
    """What this module did in this process.  Nothing numerical can say."""
    out = dict(STATS)
    out['declined_reasons'] = dict(DECLINED)
    return out


def _decline(reason):
    STATS['declined'] += 1
    DECLINED[reason] = DECLINED.get(reason, 0) + 1
    return reason


def _check_precision(dfobj):
    """A narrow cderi needs the block iterator that knows about it.

    ``docs/DF_WORK_LIST.md`` W7: ``fill_symmetric`` and ``transpose_write``
    take a ``void*`` and are compiled for ``double``, and everything this
    function does with the tensor goes through ``DF.loop``.  With
    ``FASTDF_FP32`` set but ``electrolites/fastdffp32.py`` not imported, that
    generator hands out ``float32`` blocks and every contraction below reads
    twice as many bytes as there are -- which does not fail, it returns a
    plausible wrong answer.  There is no quiet option, so this says so.
    """
    from gpu4pyscf.df.df import DF
    store = getattr(dfobj, '_cderi', None)
    if store is None:
        return
    try:
        cderi = store[cp.cuda.Device().id]
    except (KeyError, IndexError, TypeError):
        return
    if (cderi is not None and getattr(cderi, 'dtype', None) == np.float32
            and getattr(DF, '_loop_orig', None) is None):
        raise RuntimeError(
            'FASTDF_FP32 is set and the cderi is float32, but `DF.loop` is '
            'GPU4PySCF\'s own -- it would hand this function double-width '
            'blocks of a single-precision tensor and the answer would be '
            'wrong rather than absent.  Import `electrolites.fastdffp32` '
            '(it is in `patch_all()`); see docs/README_fastdffp32.md.')


# --------------------------------------------------------------------------
# the cderi, and where it lives
# --------------------------------------------------------------------------
def _krylov_reserve(n_dm, nmo, nocc):
    """Bytes the Krylov solve will hold while this tensor is alive.

    ``lib/cupy_helper.py:krylov`` keeps two lists of trial vectors, ``xs``
    and ``ax``, each ``n_dm * nmo * nocc`` words per cycle, and then makes
    one contiguous copy of each -- hence the factor of three rather than
    two.  That is the quantity upstream's 0.4 is standing in for, and it is
    computable from the shapes this function is already handed: at 42 atoms
    in def2-SVP it is **0.37 GB measured**, against a cderi of 0.99 GB and a
    0.4-of-free-memory reserve of 31 GB on an 80 GB card.
    """
    return 3 * KRYLOV_CYCLES * n_dm * nmo * nocc * 8


def _transfer_to_host(dfobj):
    """Move the whole cderi to page-locked host memory.

    Upstream's own block (``df/hessian/rhf.py:1263-1272``) for a double
    tensor -- the device tensor is a ``(naux, npairs)`` *view* of an
    ``(npairs, naux)`` buffer and ``transpose_write`` is what makes the host
    copy C-contiguous in the layout ``DF.loop`` expects.  A **narrow** tensor
    (``docs/DF_WORK_LIST.md`` W7) cannot go through it: ``transpose_write``
    takes a ``void*`` and is compiled for ``double``, so it would read twice
    as many bytes as there are.  That case is done in slices instead, which
    is what ``electrolites/fastdffp32.py`` does for the same line.
    """
    device_id = cp.cuda.Device().id
    src = dfobj._cderi[device_id]
    naux, npairs = src.shape
    if src.dtype == np.float64:
        host = empty_mapped(src.shape)
        DFHESS.libvhf_rys.transpose_write(
            host.ctypes,
            ctypes.cast(src.data.ptr, ctypes.c_void_p),
            ctypes.c_int(naux), ctypes.c_int(npairs),
            ctypes.c_int(0), ctypes.c_int(npairs))
    else:
        host = empty_mapped(src.shape, dtype=src.dtype)
        chunk = max(1, int(64e6) // max(naux, 1))
        for c0 in range(0, npairs, chunk):
            c1 = min(npairs, c0 + chunk)
            host[:, c0:c1] = cp.asnumpy(cp.ascontiguousarray(src[:, c0:c1]))
    dfobj._cderi[device_id] = host
    del src
    cp.get_default_memory_pool().free_all_blocks()


def _ensure_cderi(dfobj, n_dm, nmo, nocc, omega):
    """Build the tensor if it is not there, and decide where it lives.

    Upstream moves a device-resident tensor to the host whenever it is more
    than 0.4 of the free memory, to leave room for the Krylov subspace.  The
    fraction is a guess at a quantity that can be computed: ``_krylov_reserve``
    is what the subspace costs at these shapes, and what this function has to
    allocate on top of it is one auxiliary block's worth of scratch.  So the
    tensor goes to the host when it does not fit *beside those two*, and
    stays on the device when it does -- which at every size whose CPHF is
    runnable is a different answer from upstream's.
    """
    if dfobj._cderi is not None:
        return
    mem_avail = get_avail_mem()
    dfobj.build(omega=omega)
    STATS['builds'] += 1
    cderi = dfobj._cderi[cp.cuda.Device().id]
    if not isinstance(cderi, cp.ndarray) or cderi.size == 0:
        return
    if HOST != 'auto':
        move = cderi.nbytes > float(HOST) * mem_avail
    else:
        reserve = _krylov_reserve(n_dm, nmo, nocc)
        # one block of scratch: the dense unpack plus this call's two
        # intermediates at a single density, which is the floor below which
        # the function cannot run at all
        scratch = 256 * (nmo * nmo + (nmo + nocc) * nocc) * 8
        # `get_avail_mem()` is read *after* the build, so the tensor is
        # already on the device and is not counted again here; what moving
        # it would buy is exactly `cderi.nbytes` of the shortfall
        move = reserve + scratch > get_avail_mem()
    if move:
        STATS['moved_to_host'] += 1
        _transfer_to_host(dfobj)
    else:
        STATS['kept_on_device'] += 1


# --------------------------------------------------------------------------
# the Coulomb-only pass
# --------------------------------------------------------------------------
def _vj_pass(blk, dm_sparse, vj_sparse, auxvec, nL, npairs, n_dm):
    """``auxvec = dm . blk^T`` then ``vj += auxvec . blk``, as cuBLAS.

    The same two passes `fastdf`'s item 3 makes over the same block, and the
    same reason they are one BLAS call each: the packed block is a plain
    matrix in both of the layouts it arrives in -- the auxiliary index is
    contiguous when the tensor is on the device (``_cholesky_eri`` returns a
    transpose) and the pair index is when it has been staged from the host --
    so which one it is decides only the transpose flags.  Carried here rather
    than imported from ``electrolites/fastdf.py`` because importing that
    module patches ``df_jk.get_jk``, and an ablation of this one has to be
    able to run without that.
    """
    handle = cp.cuda.device.get_cublas_handle()
    ptr = blk.data.ptr
    aux_major = blk.strides[0] == blk.dtype.itemsize
    lda = (blk.strides[1] if aux_major else blk.strides[0]) // blk.dtype.itemsize
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


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------
def _dm_batch(nspin, n_dm, nao, nocc, blksize):
    """How many right-hand sides one pass of the block loop may hold.

    Upstream's formula, with the module docstring's items 1, 5 and 6 on top
    of it: the batch is capped at a shape cuTENSOR will plan, the free
    memory is the pool-inclusive figure -- so the answer does not move with
    the allocator's history between one Krylov cycle and the next -- and the
    result is evened out over the passes it implies.
    """
    if _DM_BATCH:
        return max(1, min(_DM_BATCH, n_dm))
    mem = (get_avail_mem() if POOLMEM
           else get_avail_mem(exclude_memory_pool=True))
    nb = int(mem * 0.75 / (nspin * blksize * (nao + nocc) * nocc * 8))
    nb = max(1, min(nb, n_dm))
    if MAX_PLAN_ELEMENTS:
        nb = max(1, min(nb, int(MAX_PLAN_ELEMENTS
                                // (nspin * nao * nocc * blksize))))
    if BALANCE:
        nb = max(1, DFHESS._balance_batch_size(nb, n_dm))
    return nb


# --------------------------------------------------------------------------
def _get_jk(dfobj, dms, mo_coeff, mo_occ, hermi=1, with_j=True, with_k=True,
            omega=None):
    """``df/hessian/rhf.py:_get_jk``, rearranged.  Same J and K."""
    if _OFF:
        return _orig_get_jk(dfobj, dms, mo_coeff, mo_occ, hermi, with_j,
                            with_k, omega)
    if multi_gpu.num_devices > 1:
        return _fall_through(dfobj, dms, mo_coeff, mo_occ, hermi, with_j,
                             with_k, omega, 'more than one device')
    if hermi != 1 or not hasattr(dms, 'mo1') or not hasattr(dms, 'occ_coeff'):
        return _fall_through(dfobj, dms, mo_coeff, mo_occ, hermi, with_j,
                             with_k, omega, 'not a CPHF density')

    mo1 = dms.mo1
    occ_coeff = dms.occ_coeff
    _mo_coeff = mo_coeff
    if mo_coeff.ndim == 2:
        _mo_coeff = mo_coeff[None, :, :]
        mo1 = mo1[None]
        occ_coeff = occ_coeff[None, :]
    nspin, n_dm, nao, nocc = mo1.shape
    nmo = _mo_coeff.shape[-1]

    _ensure_cderi(dfobj, n_dm, nmo, nocc, omega)
    _check_precision(dfobj)
    STATS['calls'] += 1
    STATS['n_dm'] = n_dm

    if with_k:
        vj, vk = _jk(dfobj, mo1, occ_coeff, nspin, n_dm, nao, nocc, with_j)
        STATS['jk'] += 1
    else:
        assert with_j
        vj, vk = _j_only(dfobj, dms, nspin, n_dm, nao), None
        STATS['j_only'] += 1

    return _to_mo(dfobj, vj, vk, _mo_coeff, mo_occ, occ_coeff, nspin, n_dm,
                  nao, nocc, with_j, with_k)


def _fall_through(dfobj, dms, mo_coeff, mo_occ, hermi, with_j, with_k, omega,
                  reason):
    _decline(reason)
    return _orig_get_jk(dfobj, dms, mo_coeff, mo_occ, hermi, with_j, with_k,
                        omega)


def _j_only(dfobj, dms, nspin, n_dm, nao):
    """The Coulomb build of a pure functional's CPHF."""
    pair_addresses, cderi_diag = dfobj._cderi_idx
    dm_sparse = dms.reshape(nspin, n_dm, nao ** 2)[:, :, pair_addresses]
    dm_sparse = dm_sparse.sum(axis=0) if nspin == 2 else dm_sparse[0]
    dm_sparse = dm_sparse * 2
    dm_sparse[:, cderi_diag] *= .5
    dm_sparse = cp.ascontiguousarray(dm_sparse)
    vj = cp.zeros_like(dm_sparse)
    npairs = dm_sparse.shape[1]
    blksize = _BLK or dfobj.get_blksize(mem_fraction=0.2)
    # one flat buffer rather than an `(n_dm, blksize)` array sliced per
    # block: the last block is short, and a column slice of a 2-D array is
    # not contiguous, which a GEMM given a leading dimension of `nL` would
    # write straight through
    auxbuf = cp.empty(n_dm * blksize)
    for _, blk in dfobj.loop(blksize=blksize, unpack=False):
        nL = blk.shape[0]
        STATS['blocks'] += 1
        if _JPASS == 'blas':
            auxvec = auxbuf[:n_dm * nL].reshape(n_dm, nL)
            _vj_pass(blk, dm_sparse, vj, auxvec, nL, npairs, n_dm)
        else:
            av = contract('np,Lp->nL', dm_sparse, blk)
            contract('nL,Lp->np', av, blk, beta=1, out=vj)
    return vj


def _jk_once(dfobj, mo1, occ_coeff, nspin, n_dm, nao, nocc,
             with_j, nb_max=0):
    """The exchange build, and the Coulomb one that rides along with it.

    Three arrangements of the same four contractions, all of them the same
    flop count and all of them measured in ``docs/README_fastdfhess.md``:

    ``fused``     the module docstring's item 2 -- one intermediate, written
                  once and consumed once by the ``nao^2 nocc`` contraction,
                  whose output is then ``vk`` itself rather than a second
                  several-GB array;
    ``blasvk``    upstream's ordering, but with the two contractions that
                  write into ``vk`` given **one accumulator each, in the
                  layout their own GEMM produces** -- ``[(n,p), j]`` for the
                  first and ``[p, (n,j)]`` for the second.  Upstream has one
                  buffer shaped ``(n_dm, nspin, nao, nocc)`` and so one of
                  the two is a permuting write whichever way it is shaped;
                  here they are added, with a transpose of a 40 MB array, at
                  the end;
    ``upstream``  GPU4PySCF's own ordering, through this module's loop, so
                  that items 2-5 can be ablated on their own.

    The Coulomb pass differs from upstream in all three.  Upstream takes the
    auxiliary vector as the **diagonal** of the ``(nocc, nocc)`` intermediate
    (``cp.einsum('sniiL->nL', ...)``), which is a strided read of one element
    in ``nocc`` out of an array of several hundred MB.  The same vector is
    ``sum_{q,i} rhok[q,i,L] X[n,q,i]`` -- one contraction over ``(q,i)``
    against an array ``nocc`` times smaller -- and that also makes the
    Coulomb half independent of the exchange intermediate, which is what lets
    the ``fused`` arrangement keep ``U`` in the smaller buffer.
    """
    _mo1 = cp.asarray(mo1)
    _occ = cp.asarray(occ_coeff)
    arrange = ARRANGE
    STATS['arrange'] = arrange
    blasvk = arrange == 'blasvk'
    fused = arrange == 'fused'
    if blasvk:
        vkA = cp.zeros((nspin, n_dm, nao, nocc))
        vkB = cp.zeros((nspin, nao, n_dm, nocc))
        vk = None
    else:
        vk = cp.zeros((n_dm, nspin, nao, nocc))
    vj = cp.zeros((n_dm, nspin, nao, nocc)) if with_j else None

    blksize = _BLK or dfobj.get_blksize(mem_fraction=0.2)
    if not nb_max:
        nb_max = _dm_batch(nspin, n_dm, nao, nocc, blksize)
    STATS['blksize'] = blksize
    STATS['dm_batch'] = nb_max
    buf2 = cp.empty((nb_max + 1, nspin, nao, nocc, blksize))
    buf3 = cp.empty((nb_max + 1, nspin, nocc, nocc, blksize))
    buf, buf1 = buf2[-1], buf3[-1]

    for cderi, _ in dfobj.loop(blksize=blksize, unpack=True):
        nL = len(cderi)
        STATS['blocks'] += 1
        rhok = ndarray((nspin, nao, nocc, nL), buffer=buf)
        rhok_oo = ndarray((nspin, nocc, nocc, nL), buffer=buf1)
        contract('Lpq,sqj->spjL', cderi, _occ, out=rhok)
        contract('spjL,spi->sijL', rhok, _occ, out=rhok_oo)
        for i0, i1 in lib.prange(0, n_dm, nb_max):
            nb = i1 - i0
            x = _mo1[:, i0:i1]
            if with_j:
                rhoj1 = contract('sqiL,snqi->nL', rhok, x)
                contract('spiL,nL->nspi', rhok, rhoj1, beta=1,
                         out=vj[i0:i1])
            if fused:
                # U[n,i,j,L] = sum_q X[n,q,i] (M_L C_o)[q,j].  The cderi
                # block is symmetric in its two AO indices, so this is the
                # occupied projection of `rhok1` without `rhok1`.
                U = ndarray((nspin, nb, nocc, nocc, nL), buffer=buf3)
                contract('sqjL,snqi->snijL', rhok, x, out=U)
                G = ndarray((nspin, nb, nao, nocc, nL), buffer=buf2)
                contract('snpa,sajL->snpjL', x, rhok_oo, out=G)
                contract('spi,snijL->snpjL', _occ, U, beta=1, out=G)
                contract('Lpq,snqjL->nspj', cderi, G, beta=1, out=vk[i0:i1])
                continue
            rhok1 = ndarray((nspin, nb, nao, nocc, nL), buffer=buf2)
            contract('Lpq,snqi->snpiL', cderi, x, out=rhok1)
            rhok1_oo = ndarray((nspin, nb, nocc, nocc, nL), buffer=buf3)
            if blasvk:
                for s in range(nspin):
                    r1 = rhok1[s].reshape(nb * nao, nocc * nL)
                    vkA[s, i0:i1] += cp.dot(
                        r1, rhok_oo[s].reshape(nocc, nocc * nL).T
                    ).reshape(nb, nao, nocc)
                    cp.matmul(_occ[s].T,
                              rhok1[s].reshape(nb, nao, nocc * nL),
                              out=rhok1_oo[s].reshape(nb, nocc, nocc * nL))
                    vkB[s][:, i0:i1] += cp.dot(
                        rhok[s].reshape(nao, nocc * nL),
                        rhok1_oo[s].reshape(nb * nocc, nocc * nL).T
                    ).reshape(nao, nb, nocc)
            else:
                contract('snpiL,sjiL->nspj', rhok1, rhok_oo, beta=1,
                         out=vk[i0:i1])
                contract('snpiL,spj->snjiL', rhok1, _occ, out=rhok1_oo)
                contract('spiL,snjiL->nspj', rhok, rhok1_oo, beta=1,
                         out=vk[i0:i1])
    if blasvk:
        vk = cp.ascontiguousarray(
            (vkA + vkB.transpose(0, 2, 1, 3)).transpose(1, 0, 2, 3))
    return vj, vk


def _jk(dfobj, mo1, occ_coeff, nspin, n_dm, nao, nocc, with_j):
    """``_jk_once`` with the density batch halved whenever cuTENSOR balks.

    ``MAX_PLAN_ELEMENTS`` is a measured boundary and boundaries move -- a
    different cuTENSOR, a different card, a shape nobody swept.  So the cap
    is a first guess and this is the backstop: a rejected *plan* raises
    before any kernel is launched and before any operand is read, so halving
    the batch and starting the call again is safe, and the smaller batch is
    remembered on the density-fitting object so the next Krylov cycle does
    not pay for the discovery again.
    """
    blksize = _BLK or dfobj.get_blksize(mem_fraction=0.2)
    nb = getattr(dfobj, '_fastdfhess_nb', 0) or _dm_batch(
        nspin, n_dm, nao, nocc, blksize)
    nb = min(nb, n_dm)
    while True:
        try:
            return _jk_once(dfobj, mo1, occ_coeff, nspin, n_dm, nao, nocc,
                            with_j, nb_max=nb)
        except _PLAN_ERRORS:
            if nb <= 1:
                raise
            nb = max(1, nb // 2)
            dfobj._fastdfhess_nb = nb
            STATS['plan_retries'] += 1
            cp.get_default_memory_pool().free_all_blocks()


def _to_mo(dfobj, vj, vk, mo_coeff, mo_occ, occ_coeff, nspin, n_dm, nao,
           nocc, with_j, with_k):
    """The half of upstream's function after the block loop, unchanged."""
    if with_k:
        vk = contract('nspi,spq->nsqi', vk, mo_coeff)
        if nspin == 1:
            vk *= 2
        if with_j:
            vj = contract('nspi,spq->nsqi', vj, mo_coeff)
            vj *= 2
            if nspin == 1:
                vj *= 2
    elif with_j:
        pair_addresses = dfobj._cderi_idx[0]
        vj = fill_symmetric(vj.T, pair_addresses, nao)
        vj = contract('pqn,sqi->nspi', vj, occ_coeff)
        vj = contract('nspi,spq->nsqi', vj, mo_coeff)

    if nspin == 1:
        if with_j:
            vj = vj.reshape(n_dm, -1)
        if with_k:
            vk = vk.reshape(n_dm, -1)
    else:
        nocca, noccb = mo_occ.sum(axis=1).get()
        if nocca != noccb:
            def fmt(v):
                return cp.hstack([v[:, 0, :, :int(nocca)].reshape(n_dm, -1),
                                  v[:, 1, :, :int(noccb)].reshape(n_dm, -1)])
            if with_j:
                vj = fmt(vj)
            if with_k:
                vk = fmt(vk)
        else:
            if with_j:
                vj = vj.reshape(n_dm, -1)
            if with_k:
                vk = vk.reshape(n_dm, -1)
    return vj, vk


DFHESS._get_jk = _get_jk
