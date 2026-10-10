"""Conventional MP2's (ia|jb) build as a blocked half transformation
(catalogue I3).

GPU4PySCF 1.8.1's ``int4c2e.get_int4c2e_ovov`` computes each unique
class-pair block of AO integrals with ``GINTfill_int2e`` and then pushes it
through eight ``einsum`` chains whose last step writes a **whole**
``ovov[o,v,o,v]`` temporary per block: per block an O(o^2 v^2) write and an
O(o^2 v^2) allocation, which is all of the time (the integrals themselves are
~1 % at alanine dipeptide) and the 40 GB wall at penicillin V.

Here the same integral blocks feed a half transformation per batch of
occupied orbitals ``I``:

    H[l, s, i, a] = sum_{mn} (mn|ls) C_occ[m, i] C_vir[n, a]     (i in I)

Each block's eight index permutations land in H's (l, s) sub-blocks as
batched GEMMs of the block's own size, so nothing O(o^2 v^2) is touched per
block.  At the end of the batch two GEMMs finish the batch's slice of
``ovov[I, :, :, :]``.  H holds ``nao^2 * |I| * nvir`` doubles; the batch is
sized to the free device memory and the integrals are recomputed per batch
(``GINTfill_int2e`` is cheap next to the transformation).  When the result
itself does not fit on the device it is kept on the host as the
``(nocc*nvir, nocc*nvir)`` matrix MP2's ``kernel`` already accepts.

``FASTMP2_OFF=1`` restores GPU4PySCF's build; ``FASTMP2_BATCH`` fixes the
occupied batch size and ``FASTMP2_HOST=1`` forces the host-resident result
(both for testing).  Measured in ``docs/KERNEL_TARGETS_1.8.1.md`` §13.26.
"""
import os

import cupy as cp
import numpy as np
from cupy.cuda import cublas

_OFF = bool(os.environ.get('FASTMP2_OFF'))
STATS = {'calls': 0, 'batches': 0, 'host_result': 0, 'retries': 0}
# doubles per slice of an integral block (and of its half-transformed slice)
CHUNK = int(os.environ.get('FASTMP2_CHUNK', 1 << 25))


_ONE = np.array(1.0)
_SYM = r'''
extern "C" __global__ void hsym(double *H, int nao, long long m)
{
    long long ls = blockIdx.x;
    int l = (int)(ls / nao), s = (int)(ls % nao);
    if (s > l) return;
    double *a = H + ((long long)l * nao + s) * m;
    if (s == l) {
        for (long long t = threadIdx.x; t < m; t += blockDim.x) a[t] *= 2.0;
        return;
    }
    double *b = H + ((long long)s * nao + l) * m;
    for (long long t = threadIdx.x; t < m; t += blockDim.x) {
        double v = a[t] + b[t];
        a[t] = v;
        b[t] = v;
    }
}
'''
_sym_kernel = None


def _symmetrize(H):
    """H[l, s] <- H[l, s] + H[s, l] in place, H of shape (nao, nao, ...)."""
    global _sym_kernel
    if _sym_kernel is None:
        _sym_kernel = cp.RawModule(code=_SYM).get_function('hsym')
    nao = H.shape[0]
    m = H.size // (nao * nao)
    _sym_kernel((nao * nao,), (256,), (H, np.int32(nao), np.int64(m)))


def _gemm_acc(A, B, C_ptr, P, R, Kd, nv, strideC):
    """C_p[R, nv] += A_p[R, Kd] @ B[Kd, nv] for p < P (row major; A is
    (P, R, Kd) contiguous, B a row slice of a C-contiguous (*, nv) array,
    C_p at C_ptr + p * strideC with row stride nv): one strided-batched GEMM
    writing straight into H."""
    handle = cp.cuda.device.get_cublas_handle()
    cublas.dgemmStridedBatched(
        handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N,
        nv, R, Kd,
        _ONE.ctypes.data,
        B.data.ptr, nv, 0,
        A.data.ptr, Kd, R * Kd,
        _ONE.ctypes.data,
        C_ptr, nv, strideC,
        P)


def _half_into(X, co, cv, H, r0, q0):
    """H[r0 + p, q0 + q, i, a] += sum_{n1 n2} X[p, q, n1, n2]
    (co1[n1, i] cv2[n2, a] + cv1[n1, a] co2[n2, i]).

    Two quarter steps as plain GEMMs over the flattened block and the
    virtual step as a strided-batched GEMM with beta = 1 into H: H's rows
    (q, i) for one p are contiguous with stride nvir, so nothing of the
    block's (p, q, i, a) size is materialised.  Only one orientation of
    (l, s) is written; ``_symmetrize`` adds the transpose at the end."""
    P, Q, N1, N2 = X.shape
    co1, co2 = co
    cv1, cv2 = cv
    nao, ni, nv = H.shape[0], H.shape[2], H.shape[3]
    C_ptr = H.data.ptr + ((int(r0) * nao + int(q0)) * ni * nv) * 8
    strideC = nao * ni * nv
    # occupied on n2, virtual on n1
    t = (X.reshape(P * Q * N1, N2) @ co2).reshape(P, Q, N1, ni)
    t = cp.ascontiguousarray(t.transpose(0, 1, 3, 2))
    _gemm_acc(t, cv1, C_ptr, P, Q * ni, N1, nv, strideC)
    # occupied on n1, virtual on n2
    Xt = cp.ascontiguousarray(X.transpose(0, 1, 3, 2)).reshape(P * Q * N2, N1)
    t = (Xt @ co1).reshape(P, Q, N2, ni)
    t = cp.ascontiguousarray(t.transpose(0, 1, 3, 2))
    _gemm_acc(t, cv2, C_ptr, P, Q * ni, N2, nv, strideC)


def _avail():
    free = cp.cuda.runtime.memGetInfo()[0]
    pool = cp.get_default_memory_pool()
    return free + pool.free_bytes()


def get_int4c2e_ovov(mol, orbo, orbv, vhfopt=None, direct_scf_tol=1e-13,
                     stream=None, omega=None):
    from gpu4pyscf.scf import int4c2e
    if omega is None:
        omega = 0.0
    if vhfopt is None:
        vhfopt = int4c2e._VHFOpt(mol, 'int2e').build(direct_scf_tol)
    STATS['calls'] += 1
    coeff = vhfopt.coeff
    orbo = coeff @ cp.asarray(orbo)
    orbv = coeff @ cp.asarray(orbv)
    nao, nocc = orbo.shape
    nvir = orbv.shape[1]

    ovov_bytes = nocc * nvir * nocc * nvir * 8
    avail = _avail()
    on_device = (ovov_bytes < 0.45 * avail
                 and not os.environ.get('FASTMP2_HOST'))
    if on_device:
        ovov = cp.empty((nocc, nvir, nocc, nvir))
        avail -= ovov_bytes
    else:
        STATS['host_result'] += 1
        ovov = np.empty((nocc * nvir, nocc * nvir))

    # the largest integral block plus its intermediates
    cart_ao_loc = vhfopt.mol.ao_loc_nr(cart=True)
    n = np.diff([int(cart_ao_loc[cp_]) for cp_ in vhfopt.l_ctr_offsets])
    pairs = sorted(int(n[i]) * int(n[j]) for i in range(len(n)) for j in range(i + 1))
    blk = pairs[-1] * pairs[-1]
    per_occ = (nao * nao * nvir                   # H
               + nocc * nao * nvir + nocc * nvir * nvir) * 8   # final stage
    # the integral block itself, ~6 slice-sized temporaries, and headroom
    # for GINTfill's own cudaMalloc scratch outside the pool
    fixed = (blk + 6 * CHUNK) * 8 + (2 << 30)
    nb = int(os.environ.get('FASTMP2_BATCH', 0)) or \
        int((0.7 * avail - fixed) // per_occ)
    nb = max(1, min(nocc, nb))

    # GPU4PySCF's allocator sends every request above 100 MB straight to
    # cudaMalloc/cudaFree; per block that is thousands of synchronising
    # calls (most of the wall time at penicillin V), so the pool is used
    # here and its free blocks handed back at the end.
    # GINTfill's launch also needs local memory from the driver, outside the
    # pool; if that (or the pool) runs out, halve the batch and start over.
    pool = cp.get_default_memory_pool()
    while True:
        try:
            with cp.cuda.using_allocator(pool.malloc):
                _batches(int4c2e, vhfopt, omega, orbo, orbv, ovov, nb, on_device)
            break
        except (RuntimeError, cp.cuda.memory.OutOfMemoryError):
            pool.free_all_blocks()
            if nb == 1:
                raise
            nb = (nb + 1) // 2
            STATS['retries'] += 1
    pool.free_all_blocks()
    return ovov


def _batches(int4c2e, vhfopt, omega, orbo, orbv, ovov, nb, on_device):
    nao, nocc = orbo.shape
    nvir = orbv.shape[1]
    H = cp.empty((nao, nao, nb, nvir))
    for i0 in range(0, nocc, nb):
        i1 = min(nocc, i0 + nb)
        STATS['batches'] += 1
        co = cp.ascontiguousarray(orbo[:, i0:i1])
        if i1 - i0 < nb:
            H = H.ravel()[:nao * nao * (i1 - i0) * nvir]
            H = H.reshape(nao, nao, i1 - i0, nvir)
        H[:] = 0
        for b0, b1, c0, c1, k0, k1, l0, l1, int4c in \
                int4c2e.loop_int4c2e_general(vhfopt, omega=omega):
            # b0:b1 = i range, c0:c1 = j range (GPU4PySCF's naming i0..j1)
            X = int4c[0]                          # [L, K, J, I]
            L, K, J, I = X.shape
            ni = i1 - i0
            # (ij|kl) and (ji|kl): occupied/virtual on (i, j), output (l, k);
            # in slices of l so the temporaries stay bounded
            step = max(1, CHUNK // (K * max(J * I, J * ni, I * ni)))
            for p0 in range(0, L, step):
                p1 = min(L, p0 + step)
                _half_into(X[p0:p1], (co[c0:c1], co[b0:b1]),
                           (orbv[c0:c1], orbv[b0:b1]), H, l0 + p0, k0)
            # (kl|ij) and (lk|ij): occupied/virtual on (k, l), output (j, i);
            # in slices of j
            step = max(1, CHUNK // (I * max(L * K, L * ni, K * ni)))
            for p0 in range(0, J, step):
                p1 = min(J, p0 + step)
                Xt = X[:, :, p0:p1].reshape(L, K, (p1 - p0), I)
                Xt = cp.ascontiguousarray(Xt.transpose(2, 3, 0, 1))
                _half_into(Xt, (co[l0:l1], co[k0:k1]),
                           (orbv[l0:l1], orbv[k0:k1]), H, c0 + p0, b0)
            del X, Xt, int4c
        _symmetrize(H)
        # ovov[i, a, j, b] = sum_{ls} H[l, s, i, a] C_occ[l, j] C_vir[s, b]
        m = (i1 - i0) * nvir
        t = (orbo.T @ H.reshape(nao, nao * m)).reshape(nocc, nao, m)
        out = cp.matmul(orbv.T, t)                        # [j, b, ia]
        del t
        out = out.reshape(nocc, nvir, i1 - i0, nvir).transpose(2, 3, 0, 1)
        if on_device:
            ovov[i0:i1] = out
        else:
            ovov[i0 * nvir:i1 * nvir] = cp.asnumpy(out).reshape(m, nocc * nvir)
        del out


def apply_patch():
    if _OFF:
        return
    from gpu4pyscf.scf import int4c2e
    if getattr(int4c2e.get_int4c2e_ovov, '_fastmp2', False):
        return
    int4c2e._get_int4c2e_ovov_orig = int4c2e.get_int4c2e_ovov
    get_int4c2e_ovov._fastmp2 = True
    int4c2e.get_int4c2e_ovov = get_int4c2e_ovov


apply_patch()
