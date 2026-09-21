"""The density-fitting cderi build: the metric contraction, and the launch
configuration of the integrals underneath it.

``docs/DF_WORK_LIST.md`` **W4**.  ``df/df.py:_cholesky_eri`` is the one-time
build every density-fitted run pays before its first SCF cycle -- **39.4 %**
of a 15-cycle taxol DF SCF and 12-23 % of the shorter ones -- and nothing in
this repository was on it.  The item names two leads and asks for both to be
measured *on the build* rather than on a kernel in isolation, because the
build has a metric contraction and a host transpose on top of the integrals.
``benchmarks/df_cderi_probe.py`` is that measurement and this module is what
it pointed at.

**One.  The metric contraction is a dense GEMM against a matrix that is
0.3 % dense.**  The build evaluates ``(pq|v)`` over the *sorted, Cartesian,
decontracted* auxiliary basis and then contracts it with

    aux_coef = C L^{-T}        (naux_sorted, naux)

where ``C = auxmol.ctr_coeff`` is the Cartesian-to-spherical and
recontraction matrix and ``L`` is the Cholesky factor of the fitting metric.
``_decompose_j2c`` forms that product densely -- a triangular solve against
``C`` -- and the build then spends one ``npair x naux_sorted x naux`` GEMM
per block on it.  ``C`` has about 1.5 non-zeros per column, and GPU4PySCF
already owns the kernel that applies it (``SortedMole.mat_dot_C``, the one
``contract_int3c2e_dm`` uses).  So the same contraction is a sparse apply
followed by one triangular solve:

    upstream    j3c . (C L^{-T})      2 npair naux_sorted naux   flops
    here        (j3c . C) L^{-T}      npair naux^2 + sparse      flops

which is ``2 naux_sorted / naux`` less arithmetic -- 2.4x at def2-SVP and
more in a basis with more general contraction -- provided cuBLAS's ``dtrsm``
does not give the factor back, the same trap ``dsyrk`` fell into in
``docs/README_fastdf.md`` 1.

**Two.  The integral kernel is launched four shell pairs deep where one is
faster.**  ``int3c2e_evaluator`` aggregates shell pairs at
``nsp_per_block*4`` (``df/int3c2e_bdiv.py:176``) -- the same constant W3
found costing the gradient's derivative kernel 1.26x-3.15x
(``docs/README_fastdfgrad.md`` 4.2).  W4 records the sweep of it on the
*plain* kernel; this module is where it is measured on a whole build.

Ablations, for the tables in ``docs/README_fastdfcderi.md``:

    FASTDFCDERI_OFF=1        hand every build back to GPU4PySCF
    FASTDFCDERI_METRIC=gemm  keep the aggregation but form `C L^{-T}` densely
                             and contract with one GEMM, as upstream does
    FASTDFCDERI_SPMULT=<n>   shell pairs per thread block of the three-centre
                             kernel, as a multiple of the scheme's
                             `nsp_per_block`.  4 is GPU4PySCF's
    FASTDFCDERI_METRIC=trsm  the triangular *solve* W4 asks for in place of
                             the multiply -- measured, and rejected by W4's
                             own falsification criterion
    FASTDFCDERI_MIN_WORK=<x> the size of the metric contraction below which
                             the dense product is kept
    FASTDFCDERI_PIN=alloc    page-lock the host-resident cderi with
                             GPU4PySCF's `cudaHostAlloc` instead of with
                             `cudaHostRegister`
    FASTDF_FP32=1            store the tensor in single precision, which is
                             `docs/DF_WORK_LIST.md` W7 and is owned by
                             `electrolites/_dffp32.py`.  It changes the answer
                             and is off unless asked for
"""
import ctypes
import os

import numpy as np
import cupy as cp
from cupy.cuda import cublas
from cupyx.scipy.linalg import solve_triangular

from gpu4pyscf.df import df as DFMOD
from gpu4pyscf.df import int3c2e_bdiv as bdiv
from gpu4pyscf.df.int3c2e_bdiv import int3c2e_scheme
from gpu4pyscf.gto.mole import SortedMole
from gpu4pyscf.lib import logger, multi_gpu
from cupy_backends.cuda.api import runtime
from electrolites import _dffp32 as FP32
from electrolites._dfaux import (                     # noqa: F401
    _contract_aux, _sparse_ctr_coeff, _spmult, _orig_aggregate, _SPMULT)
from gpu4pyscf.lib.cupy_helper import (
    asarray, cholesky, empty_mapped, get_avail_mem, ndarray)

libvhf_rys = bdiv.libvhf_rys

_OFF = bool(os.environ.get('FASTDFCDERI_OFF'))
#: how each block's metric contraction is done.
#:
#:   'trmm'  the contraction coefficients as a sparse apply and the Cholesky
#:           factor as a triangular *multiply* against an explicit `L^{-1}`
#:   'trsm'  the same, with a triangular *solve* instead -- which is what W4
#:           asks for, and which W4's own falsification criterion rejects:
#:           cuBLAS `dtrsm` runs at 39 % of `dgemm` at these shapes
#:   'gemm'  GPU4PySCF's own dense product
_METRIC = os.environ.get('FASTDFCDERI_METRIC', 'trmm')
#: ``cderi_npairs * naux_sorted * naux`` -- half the arithmetic of the metric
#: contraction -- below which the dense product is kept.  What this module
#: does to the contraction costs about a millisecond of setup (a triangular
#: inversion and one pass over the contraction coefficients) and saves about
#: half of the contraction, so it is worth having only once the contraction
#: is worth several milliseconds.  Measured: at 1.2e9 (glycine, def2-SVP) and
#: at 2.0e10 (alanine dipeptide, def2-SVP) the whole build is **0.97x**, and
#: at 6.1e10 (a 23-atom platinum complex) it is already ahead, so the
#: threshold sits between the largest loss and the smallest gain.
#: `docs/README_fastdfcderi.md` section 4.4.
_MIN_WORK = float(os.environ.get('FASTDFCDERI_MIN_WORK', 4e10))

#: how the host-resident cderi's buffer is page-locked.  'register' allocates
#: ordinary memory, advises the kernel to back it with huge pages and pins it
#: with `cudaHostRegister`; 'alloc' is GPU4PySCF's `empty_mapped`, a plain
#: `cudaHostAlloc`.  Measured, the allocation alone is **68 %** of a
#: host-resident 113-atom build -- `docs/README_fastdfcderi.md` section 5.
_PIN = os.environ.get('FASTDFCDERI_PIN', 'register')
#: whether the buffer is aligned to a transparent huge page and the kernel
#: advised to back it with them.  Measured on a whole node it is worth
#: **0.3 %** -- the factor comes from not going through `cudaHostAlloc` at
#: all -- and on a shared login GPU it was worth 2.21 -> 2.94 GB/s, so it is
#: kept: one `madvise` against a case where it does help.
_HUGEPAGE = os.environ.get('FASTDFCDERI_HUGEPAGE', '1') != '0'

#: counters, so a coverage claim can be read off a run rather than assumed.
STATS = {'builds': 0, 'trmm': 0, 'trsm': 0, 'gemm': 0, 'blocks': 0,
         'host_registered_gb': 0.}
DECLINED = {}

_ONE = np.ones(1)

#: cuBLAS's triangular *multiply*, which cupy does not wrap.  The solve it
#: replaces is the one W4 proposes and the one its falsification criterion
#: rules out; the multiply does the same arithmetic against an explicit
#: inverse and reaches GEMM throughput.
_libcublas = ctypes.CDLL('libcublas.so.12')
_dtrmm = _libcublas.cublasDtrmm_v2
_dtrmm.restype = ctypes.c_int
_dtrmm.argtypes = ([ctypes.c_void_p] + [ctypes.c_int] * 6 +
                   [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                    ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                    ctypes.c_int])


def stats():
    """What this module did in this process.  Nothing numerical can say."""
    out = dict(STATS)
    out['declined_reasons'] = dict(DECLINED)
    return out


def _decline(reason):
    DECLINED[reason] = DECLINED.get(reason, 0) + 1


# --------------------------------------------------------------------------
# the host buffer the cderi is streamed from
# --------------------------------------------------------------------------
_PAGE = 4096
#: a transparent huge page on x86-64.  See `_HUGEPAGE` for what it is worth.
_HUGE = 2 * 1024 * 1024
#: ``MADV_HUGEPAGE`` from ``asm-generic/mman-common.h``
_MADV_HUGEPAGE = 14
#: ``cudaHostRegisterMapped``; cupy does not export the enum
_HOST_REGISTER_MAPPED = 2
_libc = ctypes.CDLL('libc.so.6')


class _Pinned:
    """Page-locked host memory, pinned rather than allocated.

    ``empty_mapped`` is one ``cudaHostAlloc``, which runs at **2.09 GB/s**
    -- 6.8 s of a 113-atom host-resident build, **65 %** of it, and 41.5 s
    of a 72.5 s 284-atom one.  Ordinary memory handed to
    ``cudaHostRegister`` is the same page-locked, device-mapped buffer at
    **8.6 GB/s**: it is the allocation that is slow, not the page-locking,
    and `docs/README_fastdfcderi.md` section 5 separates the two.  The
    registration has to be released when the buffer goes, which is what this
    object is for: the array that owns it holds a reference, and every view
    of that array holds the array.
    """

    def __init__(self, nbytes, hugepage=None):
        hugepage = _HUGEPAGE if hugepage is None else hugepage
        align = _HUGE if hugepage else _PAGE
        self.raw = np.empty(nbytes + align, dtype=np.uint8)
        base = self.raw.ctypes.data
        self.offset = int((-base) % align)
        self.ptr = base + self.offset
        self.nbytes = (nbytes + _PAGE - 1) // _PAGE * _PAGE
        if hugepage:
            _libc.madvise(ctypes.c_void_p(self.ptr),
                          ctypes.c_size_t(self.nbytes), _MADV_HUGEPAGE)
        runtime.hostRegister(self.ptr, self.nbytes, _HOST_REGISTER_MAPPED)
        self.registered = True

    def __del__(self):
        if getattr(self, 'registered', False):
            self.registered = False
            try:
                runtime.hostUnregister(self.ptr)
            except Exception:                                # noqa: BLE001
                pass


class _MappedArray(np.ndarray):
    """An ndarray that can carry the registration its buffer depends on."""


def _empty_mapped(shape, dtype=float, hugepage=None):
    """``empty_mapped``, page-locked the cheaper way when that is available."""
    if _PIN != 'register':
        return empty_mapped(shape, dtype)
    nbytes = int(np.prod(shape)) * int(np.dtype(dtype).itemsize)
    try:
        holder = _Pinned(nbytes, hugepage)
    except Exception as exc:                                 # noqa: BLE001
        _decline(f'cudaHostRegister failed ({type(exc).__name__})')
        return empty_mapped(shape, dtype)
    out = np.ndarray(shape, dtype=dtype, buffer=holder.raw.data,
                     offset=holder.offset).view(_MappedArray)
    out._pin = holder
    STATS['host_registered_gb'] += nbytes * 1e-9
    return out


# --------------------------------------------------------------------------
# the metric
# --------------------------------------------------------------------------
def _decompose_j2c(auxmol, aux_sorting=None, omega=None, keep_factor=False):
    """``_decompose_j2c``, optionally keeping the factor it throws away.

    Byte for byte upstream's function in the ``aux_coef`` it returns; the
    third return value is the Cholesky factor ``L`` of the fitting metric,
    which upstream forms, uses once and drops.  The sparse contraction needs
    it, and rebuilding it would cost a second ``int2c2e`` and a second
    ``potrf``.
    """
    j2c = bdiv.int2c2e(auxmol, omega=omega)
    cd_low = None
    try:
        cd_low = cholesky(j2c)
        if keep_factor:
            aux_coef = None
        else:
            aux_coef = solve_triangular(
                cd_low, auxmol.ctr_coeff.T, lower=True, overwrite_b=True).T
        tag = 'cd'
    except RuntimeError:
        w, v = cp.linalg.eigh(j2c)
        idx = cp.where(w > DFMOD.LINEAR_DEP_THR)[0]
        logger.debug1(auxmol, 'discard %d small eigenvectors for auxiliary '
                      'dimension', w.size - len(idx))
        v = v[:, idx] / cp.sqrt(w[idx])
        aux_coef = auxmol.apply_C_dot(v, axis=0)
        tag = 'ed'

    if aux_coef is not None and aux_sorting is not None:
        aux_coef, tmp = cp.empty_like(aux_coef), aux_coef
        aux_coef[aux_sorting] = tmp
    return aux_coef, tag, cd_low


def _trsm_right_lowerT(x, cd_low):
    """``x <- x @ L^{-T}`` in place, for a C-ordered ``(m, naux)`` ``x``.

    cuBLAS reads a C-ordered array as its transpose, so the right-side solve
    this wants is a left-side one on ``x.T``, and the factor -- C-ordered and
    lower -- is read as upper and transposed back with ``op(A) = A^T``.
    """
    m, n = x.shape
    if m == 0:
        return x
    assert x.flags.c_contiguous and cd_low.flags.c_contiguous
    cublas.dtrsm(
        cp.cuda.device.get_cublas_handle(),
        cublas.CUBLAS_SIDE_LEFT, cublas.CUBLAS_FILL_MODE_UPPER,
        cublas.CUBLAS_OP_T, cublas.CUBLAS_DIAG_NON_UNIT,
        n, m, _ONE.ctypes.data, cd_low.data.ptr, n, x.data.ptr, n)
    return x


def _trmm_right_lowerT(x, cd_inv, out=None):
    """``out <- x @ L^{-1 T}``, the same product as a triangular multiply.

    Same orientation argument as ``_trsm_right_lowerT``: the buffer holds a
    lower-triangular matrix in C order, which cuBLAS reads as upper, and
    ``op(A) = A^T`` puts it back.  cuBLAS allows ``C`` to differ from ``B``,
    so the result can be written straight into the cderi.
    """
    m, n = x.shape
    if out is None:
        out = x
    if m == 0:
        return out
    assert x.flags.c_contiguous and cd_inv.flags.c_contiguous
    err = _dtrmm(
        ctypes.c_void_p(cp.cuda.device.get_cublas_handle()),
        cublas.CUBLAS_SIDE_LEFT, cublas.CUBLAS_FILL_MODE_UPPER,
        cublas.CUBLAS_OP_T, cublas.CUBLAS_DIAG_NON_UNIT, n, m,
        ctypes.c_void_p(_ONE.ctypes.data),
        ctypes.c_void_p(cd_inv.data.ptr), n,
        ctypes.c_void_p(x.data.ptr), n,
        ctypes.c_void_p(out.data.ptr), n)
    if err != 0:
        raise RuntimeError(f'cublasDtrmm failed ({err})')
    return out


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------
_orig_cholesky_eri = DFMOD._cholesky_eri
#: the original, under the name `benchmarks/coverage_probe.py` looks for
DFMOD._cholesky_eri_orig = _orig_cholesky_eri


def _cholesky_eri(intopt, omega=None, use_gpu_memory=None):
    """``df/df.py:_cholesky_eri``, with the three changes this module is.

    Structure, batching, memory accounting, the host branch and the
    recontraction branch are upstream's; what differs is the shell-pair
    aggregation the evaluator is built with, how each block's metric
    contraction is done, and how the host buffer is page-locked.
    """
    if _OFF:
        return _orig_cholesky_eri(intopt, omega, use_gpu_memory)
    if _METRIC not in ('trmm', 'trsm', 'gemm'):
        raise ValueError(f'FASTDFCDERI_METRIC={_METRIC!r}: expected one of '
                         "'trmm', 'trsm', 'gemm'")

    assert isinstance(intopt, bdiv.Int3c2eOpt)
    if intopt._int3c2e_envs is None:
        intopt.build()
    sorted_mol = intopt.mol
    mol = sorted_mol.mol
    log = logger.new_logger(mol)
    auxmol = intopt.auxmol
    naux_sorted = auxmol.nao
    num_devices = multi_gpu.num_devices

    recontract_bas = cp.asnumpy(sorted_mol.recontract_bas)
    needs_recontraction = any(recontract_bas[:, bdiv.NCTR_OF] != 1)

    sparse = _METRIC in ('trmm', 'trsm')
    if sparse and num_devices > 1:
        # a device's slice of the cderi is a column block of `L^{-T}`, which
        # a triangular routine cannot take
        _decline('multi-device')
        sparse = False

    mem_avail = get_avail_mem(exclude_memory_pool=True)
    word_avail = mem_avail // 8
    batch_size = int(word_avail * .2) // naux_sorted

    current_device = cp.cuda.device.get_device_id()
    with _spmult(_SPMULT, short_range=(omega or 0) < 0):
        eval_j3c, aux_sorting, ao_pair_offsets, _, clone_context = \
            intopt.int3c2e_evaluator(
                ao_pair_batch_size=batch_size, reorder_aux=True,
                pair_batch_by_l=needs_recontraction,
                return_clone_context=True, omega=omega)
    cderi_batch_size = int(max(ao_pair_offsets[1:] - ao_pair_offsets[:-1]))
    batch_size = cderi_batch_size
    num_batches = len(ao_pair_offsets) - 1

    if needs_recontraction:
        recontract, ao_pair_counts, contracted_ao_pair_counts, pair_addresses = \
                bdiv._create_pair_recontraction(sorted_mol, clone_context)
        cderi_offsets = np.append(0, np.cumsum(contracted_ao_pair_counts))
        cderi_batch_size = int(max(contracted_ao_pair_counts))
        cderi_npairs = len(pair_addresses)
        pair_addresses = asarray(pair_addresses)
        rows, cols = divmod(pair_addresses, mol.nao)
        diag_addrs = cp.where(rows == cols)[0]
        cderi_idx = (pair_addresses, diag_addrs)
    else:
        cderi_offsets = ao_pair_offsets
        pair_addresses, diag_addrs = intopt.pair_and_diag_indices()
        cderi_npairs = len(pair_addresses)
        pair_addresses = cp.asarray(pair_addresses, dtype=np.int32)
        cderi_idx = (pair_addresses, diag_addrs)

    # The metric contraction, and whether it is worth replacing.  The
    # decision needs `cderi_npairs`, which needs the evaluator, so it is made
    # here rather than at the top: what this module does to the contraction
    # has a fixed cost of about a millisecond -- one triangular inversion,
    # one pass over the contraction coefficients -- and the saving is
    # proportional to the arithmetic the contraction would have done.
    naux = auxmol.mol.nao
    metric_work = float(cderi_npairs) * naux_sorted * naux
    if sparse and metric_work < _MIN_WORK:
        _decline('metric contraction too small')
        sparse = False
    aux_coef, tag, cd_low = _decompose_j2c(
        auxmol, None, omega, keep_factor=sparse)
    if sparse and tag != 'cd':
        # the filtered-eigen branch has no triangular factor to solve against
        _decline('linear-dependent auxiliary basis')
        sparse = False

    if sparse:
        assert naux == cd_low.shape[0]
        idx, coef, counts, maxnnz = _sparse_ctr_coeff(auxmol, aux_sorting)
        if _METRIC == 'trmm':
            # `L^{-1}` explicitly, which is what buys the triangular
            # *multiply*.  Upstream's own `_decompose_j2c` is no more
            # conservative -- it solves the same factor against `C`, which
            # for a matrix with about one non-zero per column is the same
            # computation column by column -- and it is a *smaller* solve:
            # `naux` right-hand sides where upstream has `naux_sorted`.
            cd_low = cp.ascontiguousarray(
                solve_triangular(cd_low, cp.eye(naux), lower=True,
                                 overwrite_b=True))
    else:
        # upstream's own product, in upstream's own layout
        if aux_coef is None:
            aux_coef = solve_triangular(
                cd_low, auxmol.ctr_coeff.T, lower=True, overwrite_b=True).T
        aux_coef, tmp = cp.empty_like(aux_coef), aux_coef
        aux_coef[aux_sorting] = tmp
        naux = aux_coef.shape[1]
        if num_devices > 1:
            aux_coef = cp.asarray(aux_coef, order='F')
        cd_low = None

    STATS['builds'] += 1
    STATS[_METRIC if sparse else 'gemm'] += 1
    STATS['blocks'] += num_batches
    naux_per_device = min(naux, (naux + num_devices - 1) // num_devices)

    # `docs/DF_WORK_LIST.md` W7.  Single precision halves the tensor and
    # every PCIe copy of it, and it changes the answer, so it is off unless
    # asked for; `electrolites/_dffp32.py` owns the switch and the layout.
    # It is declined on more than one device for the reason every module in
    # this repository declines that case -- nothing here has been measured
    # there -- and because upstream's own consumers cannot read the result.
    fp32 = FP32.enabled()
    if fp32 and num_devices > 1:
        _decline('fp32 on more than one device')
        fp32 = False
    itemsize = 4 if fp32 else 8

    cp.get_default_memory_pool().free_all_blocks()
    word_avail -= batch_size * naux_sorted
    if needs_recontraction:
        word_avail -= batch_size * naux_per_device
    if fp32:
        # the double-precision staging block the transpose reads from, which
        # a double tensor does not need: the contraction writes into the
        # tensor itself there
        word_avail -= cderi_batch_size * naux_per_device

    on_gpu = True
    if use_gpu_memory is False:
        on_gpu = False
    elif cderi_npairs * naux * itemsize > word_avail * 8 * 0.95 * num_devices:
        if use_gpu_memory:
            cderi_size = cderi_npairs * naux / num_devices * itemsize * 1e-9
            raise MemoryError(
                f'Not enough GPU memory. cderi size = {cderi_size:.2f} GB')
        on_gpu = False
    log.debug1('mem_avail=%.3f MB on_gpu=%s, nao_pairs=%d, naux_per_device=%d, '
               'batch_size=%d, num_batches=%d, metric=%s, sp_mult=%d, fp32=%s',
               mem_avail, on_gpu, cderi_npairs, naux_per_device, batch_size,
               num_batches, 'sparse' if sparse else 'gemm', _SPMULT, fp32)
    FP32.note_store((naux, cderi_npairs),
                    np.float32 if fp32 else np.float64)

    #: rows of one block the metric contraction is done at a time.  The
    #: triangular multiply is out of place -- cuBLAS allows ``C == B`` but
    #: only as a legacy courtesy -- and a whole block's scratch would be
    #: 14 GB at 113 atoms, so the contraction walks the block in slices of
    #: about 256 MB instead.
    chunk = max(1, int(32e6) // naux)

    def contract(j3c, out, scratch):
        """One block's ``(npair_blk, naux_sorted) -> (npair_blk, naux)``."""
        if not sparse:
            return j3c.dot(aux_coef, out=out)
        m = j3c.shape[0]
        out = ndarray((m, naux), buffer=out)
        for c0 in range(0, m, chunk):
            c1 = min(m, c0 + chunk)
            if _METRIC == 'trsm':
                dst = out[c0:c1]
                _contract_aux(j3c[c0:c1], idx, coef, counts,
                              np.int32(naux_sorted), np.int32(naux),
                              np.int32(maxnnz), dst)
                _trsm_right_lowerT(dst, cd_low)
            else:
                tmp = ndarray((c1 - c0, naux), buffer=scratch)
                _contract_aux(j3c[c0:c1], idx, coef, counts,
                              np.int32(naux_sorted), np.int32(naux),
                              np.int32(maxnnz), tmp)
                _trmm_right_lowerT(tmp, cd_low, out=out[c0:c1])
        return out

    if not on_gpu:
        cderi_cpu = _empty_mapped((naux, cderi_npairs),
                                  dtype=np.float32 if fp32 else np.float64)

        def proc(batch_iter):
            device_id = cp.cuda.device.get_device_id()
            c = None if sparse else cp.asarray(aux_coef)
            _eval_j3c = eval_j3c
            if device_id != current_device:
                _eval_j3c = intopt.int3c2e_evaluator(
                    reorder_aux=True, clone_context=clone_context,
                    omega=omega)[0]

            work = cp.empty(naux_sorted * batch_size)
            work2 = cp.empty(naux * batch_size)
            scratch = cp.empty(naux * min(chunk, batch_size)) if sparse else None
            if needs_recontraction:
                work1 = cp.empty(naux * cderi_batch_size)
            # W7: the transpose upstream does on the host, done on the device
            # instead, because it has to narrow the block anyway.  One buffer
            # for the whole loop -- a per-batch allocation in a loop that
            # relies on a copy overlapping is `docs/DF_WORK_LIST.md` §5.8.
            t32 = cp.empty((naux, cderi_batch_size),
                           dtype=np.float32) if fp32 else None

            for batch_id in batch_iter:
                log.debug1('processing cderi batch %d', batch_id)
                j3c = _eval_j3c(shl_pair_batch_id=batch_id, out=work)
                cderi_gpu = ndarray((j3c.shape[0], naux), buffer=work2)
                if sparse:
                    cderi_gpu = contract(j3c, cderi_gpu, scratch)
                else:
                    cderi_gpu = j3c.dot(c, out=cderi_gpu)
                if needs_recontraction:
                    cderi_gpu = recontract(batch_id, cderi_gpu, out=work1)

                p0, p1 = cderi_offsets[batch_id:batch_id+2]
                if fp32:
                    view = t32[:, :p1 - p0]
                    FP32.transpose_cast(cderi_gpu, view)
                    FP32.copy_2d_to_host(cderi_cpu[:, p0:p1], view)
                    continue
                err = libvhf_rys.transpose_write(
                    cderi_cpu.ctypes,
                    ctypes.cast(cderi_gpu.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(naux), ctypes.c_int(cderi_npairs),
                    ctypes.c_int(p0), ctypes.c_int(p1))
                if err != 0:
                    raise RuntimeError('transpose_write cderi_cpu failed')

        batch_iter = iter(range(num_batches))
        multi_gpu.run(proc, args=(batch_iter,), non_blocking=True)
        multi_gpu.synchronize()
        cderi = [cderi_cpu[i*naux_per_device:(i+1)*naux_per_device]
                 for i in range(num_devices)]

    else:
        def proc():
            device_id = cp.cuda.device.get_device_id()
            aux0 = naux_per_device * device_id
            aux1 = min(naux, aux0 + naux_per_device)
            naux_dev = aux1 - aux0
            c = None if sparse else cp.asarray(aux_coef[:, aux0:aux1])

            _eval_j3c = eval_j3c
            if device_id != current_device:
                _eval_j3c = intopt.int3c2e_evaluator(
                    reorder_aux=True, clone_context=clone_context,
                    omega=omega)[0]

            # W7: the same layout in both precisions -- `(npairs, naux)`,
            # handed back transposed -- because the widened block has to
            # reach `fill_symmetric` on the branch upstream's layout puts it
            # on.  `electrolites/_dffp32.py` has the measurement that says
            # so; the first version of this stored the narrow tensor the
            # other way round and cost 1.6x on a `get_jk` call for it.
            out = cp.empty((cderi_npairs, naux_dev),
                           dtype=np.float32 if fp32 else np.float64)
            # W7's staging block: the contraction writes double and the
            # tensor stores single, so the two cannot be the same buffer as
            # they are in the double path.  A whole batch of it would be
            # 14 GB at 284 atoms -- which is the memory the mode is trying
            # to save -- so the pairs are walked in slices of about 512 MB
            # instead.  The recontraction branch is the exception: it maps a
            # whole batch at once and cannot be cut.
            pchunk = (cderi_batch_size if needs_recontraction or not fp32
                      else max(1, min(cderi_batch_size,
                                      int(64e6) // max(naux_dev, 1))))
            stage = cp.empty(pchunk * naux_dev) if fp32 else None
            work = cp.empty(naux_sorted * batch_size)
            scratch = cp.empty(naux * min(chunk, batch_size)) if sparse else None
            if needs_recontraction:
                work1 = cp.empty(naux_dev * batch_size)

            for batch_id in range(num_batches):
                j3c = _eval_j3c(batch_id, out=work)
                p0, p1 = cderi_offsets[batch_id:batch_id+2]
                if needs_recontraction:
                    dst = (ndarray((p1 - p0, naux_dev), buffer=stage) if fp32
                           else out[p0:p1])
                    tmp = ndarray((j3c.shape[0], naux_dev), buffer=work1)
                    tmp = (contract(j3c, tmp, scratch) if sparse
                           else j3c.dot(c, out=tmp))
                    recontract(batch_id, tmp, out=dst)
                    if fp32:
                        cp.copyto(out[p0:p1], dst)
                    continue
                for c0 in range(0, p1 - p0, pchunk):
                    c1 = min(p1 - p0, c0 + pchunk)
                    dst = (ndarray((c1 - c0, naux_dev), buffer=stage) if fp32
                           else out[p0 + c0:p0 + c1])
                    src = j3c[c0:c1]
                    if sparse:
                        contract(src, dst, scratch)
                    else:
                        src.dot(c, out=dst)
                    if fp32:
                        cp.copyto(out[p0 + c0:p0 + c1], dst)
            return out.T

        cderi = multi_gpu.run(proc, non_blocking=True)
    return cderi, cderi_idx


DFMOD._cholesky_eri = _cholesky_eri
