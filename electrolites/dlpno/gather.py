"""Gathers of per-item sub-blocks of the per-atom (Q|uv) tensors.

``blocks(qa, rpos, cpos, rmask, cmask)`` returns
    out[b, q, u, v] = qa[q, rpos[b, u], cpos[b, v]] * rmask[b, u] * cmask[b, v]
in one kernel pass (CuPy's advanced indexing needs a gather, a transposed
copy and a masking pass, at a fraction of the memory bandwidth).
"""
import numpy as np
import cupy as cp

_SRC = r'''
extern "C" __global__ void gather_blocks(const double* __restrict__ qa, const long long N,
                                         const int* __restrict__ rpos, const int* __restrict__ cpos,
                                         const double* __restrict__ rmask, const double* __restrict__ cmask,
                                         double* __restrict__ out, const int nq, const int R, const int C,
                                         const int ru)
{
    // one block per (b, q, u) row of the output
    const long long row = blockIdx.x;
    const int u = (int)(row % R);
    const long long bq = row / R;
    const int q = (int)(bq % nq);
    const int b = (int)(bq / nq);
    // ru: output laid out (b, u, q, v) instead of (b, q, u, v)
    double* o = ru ? out + (((long long)b * R + u) * nq + q) * C : out + row * C;
    const double rm = rmask[(long long)b * R + u];
    if (rm == 0.) {
        for (int v = threadIdx.x; v < C; v += blockDim.x) o[v] = 0.;
        return;
    }
    const double* src = qa + ((long long)q * N + rpos[(long long)b * R + u]) * N;
    const int* cp_ = cpos + (long long)b * C;
    const double* cm = cmask + (long long)b * C;
    for (int v = threadIdx.x; v < C; v += blockDim.x)
        o[v] = rm * cm[v] * src[cp_[v]];
}
'''
_k = None


def blocks(qa, rpos, cpos, rmask, cmask, out=None, ru=False):
    """qa (nq, N, N) device; rpos (b, R), cpos (b, C) int positions (any
    value where the mask is 0); masks float (b, R), (b, C).  ``ru``: output
    (b, R, nq, C) (row index ahead of the aux index)."""
    global _k
    if _k is None:
        _k = cp.RawKernel(_SRC, 'gather_blocks')
    qa = cp.ascontiguousarray(qa)
    nq, N = qa.shape[0], qa.shape[1]
    b, R = rpos.shape
    C = cpos.shape[1]
    if out is None:
        out = cp.empty((b, R, nq, C) if ru else (b, nq, R, C))
    if b == 0:
        return out
    rp = cp.ascontiguousarray(cp.where(rmask > 0, rpos, 0).astype(cp.int32))
    cq = cp.ascontiguousarray(cp.where(cmask > 0, cpos, 0).astype(cp.int32))
    nt = 128 if C <= 128 else 256
    _k((b * nq * R,), (nt,), (qa, np.int64(N), rp, cq, cp.ascontiguousarray(rmask, dtype=cp.float64),
                             cp.ascontiguousarray(cmask, dtype=cp.float64), out,
                             np.int32(nq), np.int32(R), np.int32(C), np.int32(1 if ru else 0)))
    return out


_SRC2 = r'''
// out[c, i, j] = src_c[i * ld_c + j] for i < nr_c, j < nc_c, else 0
extern "C" __global__ void gather2d(const long long* __restrict__ ptr, const int* __restrict__ ld,
                                    const int* __restrict__ nr, const int* __restrict__ nc,
                                    double* __restrict__ out, const int R, const int C)
{
    const int c = blockIdx.x;
    const double* src = (const double*)ptr[c];
    const int r = nr[c], k = nc[c], l = ld[c];
    double* o = out + (long long)c * R * C;
    for (int e = threadIdx.x; e < R * C; e += blockDim.x) {
        const int i = e / C, j = e % C;
        o[e] = (i < r && j < k) ? src[(long long)i * l + j] : 0.;
    }
}
// out[c, x0, x1, x2] = src_c[x0 * s0_c + x1 * s1_c + x2 * s2_c] for x_k < n_k[c], else 0
// (permuted 3-index copy into a common padded shape)
extern "C" __global__ void gather3p(const long long* __restrict__ ptr, const long long* __restrict__ st,
                                    const int* __restrict__ nn, double* __restrict__ out,
                                    const int A0, const int A1, const int A2)
{
    const int c = blockIdx.y;
    const double* src = (const double*)ptr[c];
    const long long s0 = st[3 * c], s1 = st[3 * c + 1], s2 = st[3 * c + 2];
    const int n0 = nn[3 * c], n1 = nn[3 * c + 1], n2 = nn[3 * c + 2];
    const long long n = (long long)A0 * A1 * A2;
    double* o = out + (long long)c * n;
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < n;
         e += (long long)gridDim.x * blockDim.x) {
        const int x2 = (int)(e % A2);
        const long long t = e / A2;
        const int x1 = (int)(t % A1), x0 = (int)(t / A1);
        o[e] = (x0 < n0 && x1 < n1 && x2 < n2) ? src[x0 * s0 + x1 * s1 + x2 * s2] : 0.;
    }
}
'''
_m2 = None


def _mod2():
    global _m2
    if _m2 is None:
        _m2 = cp.RawModule(code=_SRC2)
    return _m2


def gather2d(ptr, ld, nr, nc, R, C, out=None):
    """stack of padded (R, C) blocks from device pointer tables (static)."""
    c = int(ptr.shape[0])
    if out is None:
        out = cp.empty((c, R, C))
    if c:
        _mod2().get_function('gather2d')((c,), (128,), (ptr, ld, nr, nc, out, np.int32(R), np.int32(C)))
    return out


def gather3p(ptr, strides, nn, A0, A1, A2, out=None):
    """stack of (A0, A1, A2) blocks read through per-item strides (c, 3),
    item c's block being nn[c] (c, 3) and zero beyond."""
    c = int(ptr.shape[0])
    if out is None:
        out = cp.empty((c, A0, A1, A2))
    n = A0 * A1 * A2
    gx = max(1, min(64, (n + 255) // 256))
    for c0 in range(0, c, 65535):
        c1 = min(c, c0 + 65535)
        _mod2().get_function('gather3p')((gx, c1 - c0), (256,), (ptr[c0:c1], strides[c0:c1], nn[c0:c1], out[c0:c1],
                                                                np.int32(A0), np.int32(A1), np.int32(A2)))
    return out


def xtbx(blk, X1, X2):
    """out[b, q] = X1[b]^T blk[b, :, q, :] X2[b] for a (b, R, nq, C) block
    stack (``blocks(..., ru=True)``): two large batched GEMMs.  -> (b, nq, V1, V2)."""
    b, R, nq, C = blk.shape
    V1, V2 = X1.shape[2], X2.shape[2]
    Y = cp.matmul(X1.transpose(0, 2, 1), blk.reshape(b, R, nq * C))          # (b, V1, nq*C)
    Y = cp.matmul(Y.reshape(b, V1 * nq, C), X2).reshape(b, V1, nq, V2)
    return Y.transpose(0, 2, 1, 3)


_SRC3 = r'''
// out[i, r, :] = Z[b[i], rows[i, r], :] * mask[i, r]      (Z: (nb, NR, C))
extern "C" __global__ void gather_rows(const double* __restrict__ Z, const long long NR, const int C,
                                       const long long* __restrict__ b, const long long* __restrict__ rows,
                                       const double* __restrict__ mask, double* __restrict__ out,
                                       const int R, const long long ldr)
{
    const long long ir = blockIdx.x;             // (i, r) row of the output
    const long long i = ir / R;
    const int r = (int)(ir % R);
    double* o = out + ir * C;
    const double m = mask[i * ldr + r];
    if (m == 0.) {
        for (int c = threadIdx.x; c < C; c += blockDim.x) o[c] = 0.;
        return;
    }
    const double* z = Z + ((long long)b[i] * NR + rows[i * ldr + r]) * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) o[c] = m * z[c];
}
'''
_m3 = None


def gather_rows(Z, bidx, rows, mask):
    """out[i, r] = Z[bidx[i], rows[i, r]] * mask[i, r]  (one pass; rows may be
    a row view of a larger table).  Z (nb, NR, C) contiguous."""
    global _m3
    if _m3 is None:
        _m3 = cp.RawKernel(_SRC3, 'gather_rows')
    Z = cp.ascontiguousarray(Z)
    n, R = rows.shape
    C = Z.shape[2]
    out = cp.empty((n, R, C))
    if n == 0:
        return out
    rows = cp.ascontiguousarray(rows, dtype=cp.int64)
    mask = cp.ascontiguousarray(mask, dtype=cp.float64)
    nt = 64 if C <= 64 else 128
    _m3((n * R,), (nt,), (Z, np.int64(Z.shape[1]), np.int32(C), cp.ascontiguousarray(bidx, dtype=cp.int64),
                          rows, mask, out, np.int32(R), np.int64(R)))
    return out


def xT_z_by_pointer(tab, pidx, Zg):
    """out[i] = tab[pidx[i]]^T Zg[i] with tab (np, P, VP) and Zg (n, P, V),
    as one pointer-batched GEMM (no gathered copy of the table rows)."""
    from cupy_backends.cuda.libs import cublas
    n, P, V = Zg.shape
    VP = tab.shape[2]
    out = cp.empty((n, VP, V))
    if n == 0:
        return out
    tab = cp.ascontiguousarray(tab)
    Zg = cp.ascontiguousarray(Zg)
    pidx = cp.asarray(pidx, dtype=cp.uint64)
    Ap = Zg.data.ptr + cp.arange(n, dtype=cp.uint64) * np.uint64(P * V * 8)
    Bp = tab.data.ptr + pidx * np.uint64(tab.shape[1] * VP * 8)
    Cp = out.data.ptr + cp.arange(n, dtype=cp.uint64) * np.uint64(VP * V * 8)
    one = np.array(1., np.float64)
    zero = np.array(0., np.float64)
    handle = cp.cuda.device.get_cublas_handle()
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    # column-major: C^T (V x VP) = Zg^T (V x P) . tab_i (P x VP)
    cublas.dgemmBatched(handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T, V, VP, P,
                        one.ctypes.data, Ap.data.ptr, V, Bp.data.ptr, VP,
                        zero.ctypes.data, Cp.data.ptr, V, n)
    return out
