"""Batched dense linear algebra on the device for many small/medium matrices.

``eigh_batched`` calls cuSOLVER's ``cusolverDnXsyevBatched`` (CUDA >= 12.6),
which unlike ``syevjBatched`` has no n <= 32 limit and unlike a loop of
``syevd`` calls runs one launch for the whole stack.  Falls back to
``cupy.linalg.eigh`` when the symbol is missing.
"""
import ctypes

import numpy as np
import cupy as cp

_lib = None
_params = None


def _load():
    global _lib, _params
    if _lib is not None:
        return _lib
    import cupy_backends.cuda.libs.cusolver as cs  # noqa: F401 (loads the library)
    for name in ('libcusolver.so.11', 'libcusolver.so.12', 'libcusolver.so'):
        try:
            lib = ctypes.CDLL(name)
            lib.cusolverDnXsyevBatched
            break
        except (OSError, AttributeError):
            lib = None
    if lib is None:
        _lib = False
        return _lib
    p = ctypes.c_void_p()
    if lib.cusolverDnCreateParams(ctypes.byref(p)) != 0:
        _lib = False
        return _lib
    _params = p
    _lib = lib
    return _lib


CUDA_R_64F = 1
CUSOLVER_EIG_MODE_VECTOR = 1
CUBLAS_FILL_MODE_LOWER = 0


def eigh_batched(a, overwrite=False):
    """Eigenvalues (ascending) and eigenvectors of a stack of symmetric
    matrices a[b, n, n].  Returns (w[b, n], v[b, n, n]) with v[:, :, k]
    the k-th eigenvector, like numpy."""
    a = cp.asarray(a, dtype=np.float64)
    b, n = a.shape[0], a.shape[-1]
    if b == 0 or n == 0:
        return cp.zeros((b, n)), cp.zeros((b, n, n))
    lib = _load()
    if not lib or n <= 32:
        return cp.linalg.eigh(a)
    handle = cp.cuda.device.get_cusolver_handle()
    stream = cp.cuda.get_current_stream().ptr
    lib.cusolverDnSetStream(ctypes.c_void_p(handle), ctypes.c_void_p(stream))
    # column-major: pass the transpose; symmetric so only the fill matters
    A = a.copy() if not overwrite or not a.flags.c_contiguous else a
    A = cp.ascontiguousarray(A)
    W = cp.empty((b, n))
    dws = ctypes.c_size_t()
    hws = ctypes.c_size_t()
    st = lib.cusolverDnXsyevBatched_bufferSize(
        ctypes.c_void_p(handle), _params, ctypes.c_int(CUSOLVER_EIG_MODE_VECTOR),
        ctypes.c_int(CUBLAS_FILL_MODE_LOWER), ctypes.c_int64(n),
        ctypes.c_int(CUDA_R_64F), ctypes.c_void_p(A.data.ptr), ctypes.c_int64(n),
        ctypes.c_int(CUDA_R_64F), ctypes.c_void_p(W.data.ptr), ctypes.c_int(CUDA_R_64F),
        ctypes.byref(dws), ctypes.byref(hws), ctypes.c_int64(b))
    if st != 0:
        return cp.linalg.eigh(a)
    dbuf = cp.empty(max(dws.value, 1), dtype=np.uint8)
    hbuf = np.empty(max(hws.value, 1), dtype=np.uint8)
    info = cp.zeros(b, dtype=np.int32)
    st = lib.cusolverDnXsyevBatched(
        ctypes.c_void_p(handle), _params, ctypes.c_int(CUSOLVER_EIG_MODE_VECTOR),
        ctypes.c_int(CUBLAS_FILL_MODE_LOWER), ctypes.c_int64(n),
        ctypes.c_int(CUDA_R_64F), ctypes.c_void_p(A.data.ptr), ctypes.c_int64(n),
        ctypes.c_int(CUDA_R_64F), ctypes.c_void_p(W.data.ptr), ctypes.c_int(CUDA_R_64F),
        ctypes.c_void_p(dbuf.data.ptr), ctypes.c_size_t(dws.value),
        hbuf.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(hws.value),
        ctypes.c_void_p(info.data.ptr), ctypes.c_int64(b))
    if st != 0:
        raise RuntimeError(f'cusolverDnXsyevBatched failed ({st})')
    # A now holds eigenvectors column-major: A[b] (row-major view) = V^T
    return W, A.transpose(0, 2, 1)
