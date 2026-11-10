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


_PIVCHOL = r'''
// one thread block per matrix: Psi4's Matrix::pivoted_cholesky selection.
// A: (b, n, n) row-major, L: scratch (b, n, n); out: piv (b, n) (-1 padded)
extern "C" __global__ void pivchol(const double* __restrict__ A, double* __restrict__ L,
                                   double* __restrict__ dwork, int* __restrict__ piv,
                                   const int* __restrict__ nn, const int n, const double tol)
{
    const int b = blockIdx.x;
    const int nb = nn[b];
    const double* a = A + (long long)b * n * n;
    double* l = L + (long long)b * n * n;
    double* d = dwork + (long long)b * n;
    int* pv = piv + (long long)b * n;
    __shared__ double best[256];
    __shared__ int bidx[256];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        d[i] = (i < nb) ? a[(long long)i * n + i] : -1.;
        pv[i] = -1;
    }
    __syncthreads();
    for (int m = 0; m < nb; ++m) {
        double bv = -2.; int bi = -1;
        for (int i = threadIdx.x; i < nb; i += blockDim.x)
            if (d[i] > bv) { bv = d[i]; bi = i; }
        best[threadIdx.x] = bv; bidx[threadIdx.x] = bi;
        __syncthreads();
        for (int w = blockDim.x / 2; w > 0; w >>= 1) {
            if (threadIdx.x < w) {
                double o = best[threadIdx.x + w]; int oi = bidx[threadIdx.x + w];
                if (o > best[threadIdx.x] || (o == best[threadIdx.x] && oi >= 0 && TIEBREAK(oi, bidx[threadIdx.x]))) {
                    best[threadIdx.x] = o; bidx[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }
        // Roundoff in normalized unit diagonals must not change the
        // retained PAO subspace between neighboring geometries.
        const double tie = 16. * 2.220446049250313e-16 * fmax(1., fabs(best[0]));
        bi = -1;
        for (int i = threadIdx.x; i < nb; i += blockDim.x)
            if (d[i] >= best[0] - tie && (bi < 0 || TIEBREAK(i, bi))) bi = i;
        bidx[threadIdx.x] = bi;
        __syncthreads();
        for (int w = blockDim.x / 2; w > 0; w >>= 1) {
            if (threadIdx.x < w) {
                int oi = bidx[threadIdx.x + w];
                if (oi >= 0 && (bidx[threadIdx.x] < 0 || TIEBREAK(oi, bidx[threadIdx.x])))
                    bidx[threadIdx.x] = oi;
            }
            __syncthreads();
        }
        const int p = bidx[0];
        const double dp = d[p];
        __syncthreads();
        if (dp < tol) break;
        const double s = 1. / sqrt(dp);
        for (int i = threadIdx.x; i < nb; i += blockDim.x) {
            double v = a[(long long)i * n + p];
            for (int q = 0; q < m; ++q) v -= l[(long long)i * n + q] * l[(long long)p * n + q];
            v *= s;
            l[(long long)i * n + m] = v;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < nb; i += blockDim.x) {
            double v = l[(long long)i * n + m];
            d[i] -= v * v;
        }
        __syncthreads();
        if (threadIdx.x == 0) { d[p] = -1.; pv[m] = p; }
        __syncthreads();
    }
}
'''
_pc_kern = None


def pivoted_cholesky_batched(A, nn, tol):
    """Pivot selection of a pivoted Cholesky of each A[b, :nn[b], :nn[b]]
    (stopping at diag < tol), as Psi4's.  Returns a boolean mask (b, n)."""
    global _pc_kern
    if _pc_kern is None:
        import os
        tb = '#define TIEBREAK(a, b) ((a) > (b))\n' if os.environ.get('DLPNO_TIEBREAK_HIGH') else '#define TIEBREAK(a, b) ((a) < (b))\n'
        _pc_kern = cp.RawKernel(tb + _PIVCHOL, 'pivchol')
    A = cp.ascontiguousarray(A, dtype=cp.float64)
    b, n = A.shape[0], A.shape[1]
    L = cp.zeros_like(A)
    d = cp.empty((b, n))
    piv = cp.empty((b, n), dtype=cp.int32)
    _pc_kern((b,), (256,), (A, L, d, piv, cp.asarray(nn, dtype=cp.int32), np.int32(n),
                            np.float64(tol)))
    mask = cp.zeros((b, n), dtype=bool)
    rows = cp.repeat(cp.arange(b), n).reshape(b, n)
    ok = piv >= 0
    mask[rows[ok], piv[ok]] = True
    return mask


def orthonormal_domain(S, nn, s_cut, method='pivchol'):
    """Orthonormal basis of the span of each (redundant) function set:
    S (b, n, n) zero-padded overlap, nn true sizes.  Psi4's
    orthocanonicalizer: normalise, pivoted-Cholesky selection at S_CUT,
    canonical orthogonalisation of the selected functions.  Returns X
    (b, n, n) with zero columns for dropped directions, and the keep mask."""
    b, n = S.shape[0], S.shape[1]
    m = cp.arange(n)[None, :] < cp.asarray(nn)[:, None]
    dg = cp.diagonal(S, axis1=1, axis2=2)
    dn = cp.where(m, 1. / cp.sqrt(cp.where(m, dg, 1.)), 0.)
    Sn = S * dn[:, :, None] * dn[:, None, :]
    if method == 'pivchol':
        # Psi4 orders the functions by increasing off-diagonal overlap before
        # LAPACK's dpstrf (ties go to the first in that order)
        od = cp.sum(cp.abs(Sn), axis=2) - cp.abs(cp.diagonal(Sn, axis1=1, axis2=2))
        od = cp.where(m, od, cp.inf)
        order = cp.argsort(od, axis=1)
        bi = cp.arange(b)[:, None]
        Sp = Sn[bi[:, :, None], order[:, :, None], order[:, None, :]]
        selp = pivoted_cholesky_batched(Sp, nn, s_cut)
        sel = cp.zeros_like(selp)
        sel[bi, order] = selp
        sm = sel[:, :, None] & sel[:, None, :]
        Ss = cp.where(sm, Sn, 0.)
        w, U = eigh_batched(Ss)
        keep = w > 1e-12
        # The padded zero block is a null eigenspace. Roundoff can mix
        # its rows into an eigenvector with a small positive eigenvalue;
        # division by sqrt(w) then amplifies those forbidden PAO rows.
        # Enforce the selected support exactly, as the compact CPU
        # eigensolve followed by a zero scatter does.
        U *= sel[:, :, None]
        from .setup import orthogonalization_trace
        trace = orthogonalization_trace.get()
        if trace is not None:
            for size, selected, rank in zip(np.asarray(nn), cp.asnumpy(sel), cp.asnumpy(keep.sum(axis=1))):
                trace.append([int(size), np.flatnonzero(selected[:int(size)]).tolist(), int(rank)])
    else:
        w, U = eigh_batched(Sn)
        keep = w > s_cut
    X = dn[:, :, None] * U * cp.where(keep, 1. / cp.sqrt(cp.where(keep, w, 1.)), 0.)[:, None, :]
    return X, keep


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
    if bool(cp.any(info)):
        raise np.linalg.LinAlgError(
            f'cusolverDnXsyevBatched did not converge: info={info.get().tolist()}')
    # A now holds eigenvectors column-major: A[b] (row-major view) = V^T
    return W, A.transpose(0, 2, 1)


def chol_inv_batched(J):
    """L^-1 for the Cholesky factors L of a stack of SPD matrices J[b]
    (batched potrf, then one batched triangular solve against the identity;
    cupy.linalg.inv on a stack goes through getrf/getri and is much slower)."""
    from cupy_backends.cuda.libs import cublas
    L = cp.ascontiguousarray(cp.linalg.cholesky(J))
    b, n = L.shape[0], L.shape[-1]
    if b == 0:
        return L
    X = cp.ascontiguousarray(cp.broadcast_to(cp.eye(n), (b, n, n)))
    step = n * n * 8
    Lp = cp.arange(b, dtype=cp.uint64) * step + L.data.ptr
    Xp = cp.arange(b, dtype=cp.uint64) * step + X.data.ptr
    one = np.array(1., np.float64)
    handle = cp.cuda.device.get_cublas_handle()
    # row-major L is the column-major upper factor U = L^T; X L^T... in column
    # major the buffer of X^T solves  X^T U = I  (side right, upper)
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    cublas.dtrsmBatched(handle, cublas.CUBLAS_SIDE_RIGHT, cublas.CUBLAS_FILL_MODE_UPPER,
                        cublas.CUBLAS_OP_N, cublas.CUBLAS_DIAG_NON_UNIT, n, n,
                        one.ctypes.data, Lp.data.ptr, n, Xp.data.ptr, n, b)
    return X
