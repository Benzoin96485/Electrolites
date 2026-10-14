"""The closed-shell (T) correction on the GPU (`docs/FEATURE_TARGETS_PYSCF_2.13.md`
P1).

GPU4PySCF 1.8.1 has ``CCSD.ccsd_t = NotImplemented``
(``cc/ccsd_incore.py:575``), and the only route was to hand the amplitudes to
PySCF on the host.  Measured on a 128-core CPU node, that (T) is 76-91 % of
today's fastest CCSD(T) route once ``fastccsd`` has put the CCSD itself on
the device (FEATURE_TARGETS §11.2).

The energy is PySCF's (``pyscf/cc/ccsd_t_slow.py``, the reference the C
kernel reproduces), rearranged from its loop over virtual triples into a
loop over **occupied** triples i >= j >= k, so that the work is GEMMs with
virtual dimensions:

    W(p,q,r)[a,b,c] = sum_f (pa|bf) t2[r,q,c,f] - sum_m (pa|qm) t2[m,r,b,c]

For each (i, j) and a batch of k <= j, the six occupied orders of W(i,j,k)
are three batched-GEMM shapes (k as the third, second or first index).  Two
CUDA passes then form, per (k, a, b, c),

    X[abc] = sum_rho W(rho ijk)[rho abc]             (6 orders rho)
    Y(rho)[abc] = W(rho ijk)[abc] + 1/2 [(pa|qb) t1[r,c] + t2[p,q,a,b] f[c,r]]
    R_s[abc] = 4 Y(s) + Y(s.kij) + Y(s.jki) - 2 Y(s.kji) - 2 Y(s.ikj) - 2 Y(s.jik)
    E += w(ijk) sum_s X[s^-1 abc] R_s[abc] / (e_i + e_j + e_k - e_a - e_b - e_c)

with w = 1, 1/2, 1/6 for three, two, one distinct of i >= j >= k, and
E(T) = 2 E.  This is PySCF's sum over all ordered (i, j, k): the summand is
*not* symmetric in the occupied indices alone (a 10 % error on water if one
assumes it is), so each canonical triple sums its six orders explicitly, all
from the same six W blocks.  The MO integrals
(ia|bc), (ia|jk), (ia|jb) come from ``fastmp2``'s half transformation
(``get_int4c2e_ovpq``); the Fock matrix in the active MO basis from one
GPU Fock build, so a non-canonical f_vo enters exactly as in PySCF.

    import electrolites
    electrolites.enable('ccsd_t')
    mycc = gpu4pyscf.cc.ccsd_incore.CCSD(mf).run()
    e_t = mycc.ccsd_t()
"""
import os
import time

import numpy as np
import cupy as cp

from gpu4pyscf.lib import logger

STATS = {'calls': 0, 'pairs': 0, 'integrals_s': 0., 'triples_s': 0.}
#: with ELECTROLITES_T_PROFILE set: device seconds in the W GEMMs and in the
#: two energy passes (synchronises once per (i, j) batch)
PROF = {'gemm': 0., 'energy': 0.}

_SRC = r'''
#define AT(W, x, y, z) W[base + ((long)(x) * nv + (y)) * nv + (z)]
#define DECODE \
        int c = idx % nv; long t = idx / nv; int b = t % nv; t /= nv; \
        int a = t % nv; int kk = t / nv; const long base = (long)kk * nv3;

// pass 1: X(ijk)[abc] = sum_rho W(rho ijk)[rho abc], the simultaneous-
// permutation sum of W
extern "C" __global__
void t_x(const double* __restrict__ W0, const double* __restrict__ W1,
         const double* __restrict__ W2, const double* __restrict__ W3,
         const double* __restrict__ W4, const double* __restrict__ W5,
         double* __restrict__ X, int nk, int nv)
{
    const long nv3 = (long)nv * nv * nv;
    const long n = (long)nk * nv3;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += (long)gridDim.x * blockDim.x) {
        DECODE
        X[idx] = AT(W0, a, b, c) + AT(W1, a, c, b) + AT(W2, b, a, c)
               + AT(W3, b, c, a) + AT(W4, c, a, b) + AT(W5, c, b, a);
    }
}

// pass 2: E += w(ijk) sum_sigma X[sigma^-1 abc] R_sigma[abc] / D[abc],
// R_sigma the r3 combination of Y(rho ijk)[abc] = W + V/2 for the occupied
// order sigma(ijk); w = 1, 1/2, 1/6 for 3, 2, 1 distinct of i >= j >= k
extern "C" __global__
void t_energy(const double* __restrict__ W0, const double* __restrict__ W1,
              const double* __restrict__ W2, const double* __restrict__ W3,
              const double* __restrict__ W4, const double* __restrict__ W5,
              const double* __restrict__ X,
              const double* __restrict__ ovov, const double* __restrict__ t1,
              const double* __restrict__ t2, const double* __restrict__ fvo,
              const double* __restrict__ eo, const double* __restrict__ ev,
              int i, int j, int k0, int nk, int no, int nv, double* out)
{
    const long nv3 = (long)nv * nv * nv;
    const long n = (long)nk * nv3;
    double acc = 0.;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += (long)gridDim.x * blockDim.x) {
        DECODE
        int k = k0 + kk;
#define OV(p, q) ovov[(((long)(p) * nv + a) * no + (q)) * nv + b]
#define T2(p, q) t2[(((long)(p) * no + (q)) * nv + a) * nv + b]
#define F(r) fvo[(long)c * no + (r)]
        double y0 = W0[idx] + .5 * (OV(i, j) * t1[k * nv + c] + T2(i, j) * F(k));
        double y1 = W1[idx] + .5 * (OV(i, k) * t1[j * nv + c] + T2(i, k) * F(j));
        double y2 = W2[idx] + .5 * (OV(j, i) * t1[k * nv + c] + T2(j, i) * F(k));
        double y3 = W3[idx] + .5 * (OV(j, k) * t1[i * nv + c] + T2(j, k) * F(i));
        double y4 = W4[idx] + .5 * (OV(k, i) * t1[j * nv + c] + T2(k, i) * F(j));
        double y5 = W5[idx] + .5 * (OV(k, j) * t1[i * nv + c] + T2(k, j) * F(i));
        double s =
              (4*y0 + y4 + y3 - 2*y5 - 2*y1 - 2*y2) * AT(X, a, b, c)
            + (4*y1 + y2 + y5 - 2*y3 - 2*y0 - 2*y4) * AT(X, a, c, b)
            + (4*y2 + y5 + y1 - 2*y4 - 2*y3 - 2*y0) * AT(X, b, a, c)
            + (4*y3 + y0 + y4 - 2*y1 - 2*y2 - 2*y5) * AT(X, c, a, b)
            + (4*y4 + y3 + y0 - 2*y2 - 2*y5 - 2*y1) * AT(X, b, c, a)
            + (4*y5 + y1 + y2 - 2*y0 - 2*y4 - 2*y3) * AT(X, c, b, a);
        double D = eo[i] + eo[j] + eo[k] - ev[a] - ev[b] - ev[c];
        double w = (i == j && j == k) ? (1. / 6) :
                   ((i == j || j == k) ? .5 : 1.);
        acc += w * s / D;
    }
    __shared__ double sh[256];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}
'''
_module = None


def _kernels():
    global _module
    if _module is None:
        _module = cp.RawModule(code=_SRC)
    return _module.get_function('t_x'), _module.get_function('t_energy')


_ONE = np.ones(1)


def _gemm(trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc,
          batch=1, sA=0, sB=0, sC=0):
    """Row-major C[m, n] = alpha A[m, k] op(B) + beta C on raw device
    pointers, ``batch`` times with element strides sA, sB, sC; op(B) = B
    (k x n, ld ldb) or, with ``trans_b``, B^T for B stored n x k."""
    from cupy_backends.cuda.libs import cublas
    handle = cp.cuda.device.get_cublas_handle()
    al = np.array([alpha])
    be = np.array([beta])
    opb = cublas.CUBLAS_OP_T if trans_b else cublas.CUBLAS_OP_N
    cublas.dgemmStridedBatched(
        handle, opb, cublas.CUBLAS_OP_N, n, m, k,
        al.ctypes.data, B, ldb, sB, A, lda, sA,
        be.ctypes.data, C, ldc, sC, batch)


def _mo_integrals(mol, orbo, orbv):
    """(ia|bc), (ia|jk), (ia|jb) on the device."""
    from electrolites import fastmp2
    nocc = orbo.shape[1]
    ovvv = fastmp2.get_int4c2e_ovpq(mol, orbo, orbv, orbv, orbv)
    ovon = fastmp2.get_int4c2e_ovpq(mol, orbo, orbv, orbo,
                                    cp.hstack([orbo, orbv]))
    ovvv = cp.asarray(ovvv)
    ovon = cp.asarray(ovon)
    ovoo = cp.ascontiguousarray(ovon[:, :, :, :nocc])
    ovov = cp.ascontiguousarray(ovon[:, :, :, nocc:])
    return ovvv, ovoo, ovov


def _fock_mo(mycc, mo):
    """The Fock matrix in the active MO basis, from one GPU Fock build on
    the reference density (what PySCF's ``_ChemistsERIs`` holds)."""
    mf = mycc._scf
    if getattr(mf, 'device', 'cpu') != 'gpu':
        mf = mf.to_gpu()
    dm = mf.make_rdm1(cp.asarray(mf.mo_coeff), cp.asarray(mf.mo_occ))
    fao = mf.get_hcore() + mf.get_veff(mf.mol, dm)
    return mo.T @ cp.asarray(fao) @ mo


def kernel(mycc, t1=None, t2=None, verbose=None, k_batch=None):
    log = logger.new_logger(mycc, verbose)
    t0 = time.perf_counter()
    STATS['calls'] += 1
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    t1 = cp.ascontiguousarray(cp.asarray(t1))
    t2 = cp.ascontiguousarray(cp.asarray(t2))
    no, nv = t1.shape
    mask = mycc.get_frozen_mask()
    mo = cp.asarray(mycc.mo_coeff)[:, cp.asarray(mask)]
    orbo, orbv = mo[:, :no], mo[:, no:]
    mol = mycc.mol

    fock = _fock_mo(mycc, mo)
    e = cp.ascontiguousarray(fock.diagonal())
    eo, ev = cp.ascontiguousarray(e[:no]), cp.ascontiguousarray(e[no:])
    fvo = cp.ascontiguousarray(fock[no:, :no])          # (v, o)

    ovvv, ovoo, ovov = _mo_integrals(mol, orbo, orbv)
    cp.cuda.Stream.null.synchronize()
    t1s = time.perf_counter()
    STATS['integrals_s'] += t1s - t0

    # the three GEMM shapes as cuBLAS (strided-batched) calls on views of
    # ovvv, ovoo and t2 -- no copies, no broadcast operands
    t2_mk = cp.ascontiguousarray(t2.transpose(1, 0, 2, 3))   # [k, m, b, c]
    v2, v3 = nv * nv, nv ** 3
    o2 = no * no
    P_ovvv, P_ovoo = ovvv.data.ptr, ovoo.data.ptr
    P_t2, P_t2mk = t2.data.ptr, t2_mk.data.ptr
    D = 8

    def form_a(p, q, k0, k1, out):
        # W(p, q, k)[k][ab][c] = sum_f ovvv[p][ab, f] t2[k, q][c, f]
        #                       - sum_m ovoo[p, a, q, m] t2[m, k][bc]
        nk = k1 - k0
        _gemm(True, v2, nv, nv, 1., P_ovvv + p * v3 * D, nv,
              P_t2 + (k0 * no + q) * v2 * D, nv, 0., out.data.ptr, nv,
              nk, 0, no * v2, v3)
        _gemm(False, nv, v2, no, -1., P_ovoo + ((p * nv) * no + q) * no * D, o2,
              P_t2mk + k0 * no * v2 * D, v2, 1., out.data.ptr, v2,
              nk, 0, no * v2, v3)

    def form_b(p, r, k0, k1, out):
        # W(p, k, r)
        nk = k1 - k0
        _gemm(True, v2, nv, nv, 1., P_ovvv + p * v3 * D, nv,
              P_t2 + (r * no + k0) * v2 * D, nv, 0., out.data.ptr, nv,
              nk, 0, v2, v3)
        _gemm(False, nv, v2, no, -1., P_ovoo + ((p * nv) * no + k0) * no * D, o2,
              P_t2 + r * v2 * D, no * v2, 1., out.data.ptr, v2,
              nk, no, 0, v3)

    def form_c(q, r, k0, k1, out):
        # W(k, q, r): one GEMM each, k folded into the rows
        nk = k1 - k0
        _gemm(True, nk * v2, nv, nv, 1., P_ovvv + k0 * v3 * D, nv,
              P_t2 + (r * no + q) * v2 * D, nv, 0., out.data.ptr, nv)
        _gemm(False, nk * nv, v2, no, -1., P_ovoo + ((k0 * nv) * no + q) * no * D,
              o2, P_t2 + r * v2 * D, no * v2, 1., out.data.ptr, v2)

    if k_batch is None:
        free = cp.cuda.runtime.memGetInfo()[0] + \
            cp.get_default_memory_pool().free_bytes()
        k_batch = int(os.environ.get('ELECTROLITES_T_KBATCH', 0)) or \
            max(1, int(0.6 * free // (7 * 8 * nv ** 3)))
    out = cp.zeros(1)
    kx, kern = _kernels()
    buf = cp.empty((7, min(k_batch, no), nv, nv, nv))
    prof = bool(os.environ.get('ELECTROLITES_T_PROFILE'))
    nthreads = 256
    for i in range(no):
        for j in range(i + 1):
            for k0 in range(0, j + 1, k_batch):
                k1 = min(j + 1, k0 + k_batch)
                nk = k1 - k0
                if prof:
                    ev0 = cp.cuda.Event(); ev0.record()
                Ws = [buf[n_][:nk] for n_ in range(6)]
                form_a(i, j, k0, k1, Ws[0])      # W(ijk)
                form_b(i, j, k0, k1, Ws[1])      # W(ikj)
                form_a(j, i, k0, k1, Ws[2])      # W(jik)
                form_b(j, i, k0, k1, Ws[3])      # W(jki)
                form_c(i, j, k0, k1, Ws[4])      # W(kij)
                form_c(j, i, k0, k1, Ws[5])      # W(kji)
                W0, W1, W2, W3, W4, W5 = Ws
                n = nk * nv ** 3
                nblk = min(65535 * 4, (n + nthreads - 1) // nthreads)
                if prof:
                    ev1 = cp.cuda.Event(); ev1.record()
                X = buf[6][:nk]
                kx((nblk,), (nthreads,),
                   (W0, W1, W2, W3, W4, W5, X, np.int32(nk), np.int32(nv)))
                kern((nblk,), (nthreads,),
                     (W0, W1, W2, W3, W4, W5, X, ovov, t1, t2, fvo, eo, ev,
                      np.int32(i), np.int32(j), np.int32(k0), np.int32(nk),
                      np.int32(no), np.int32(nv), out))
                STATS['pairs'] += 1
                if prof:
                    ev2 = cp.cuda.Event(); ev2.record(); ev2.synchronize()
                    PROF['gemm'] += cp.cuda.get_elapsed_time(ev0, ev1) / 1e3
                    PROF['energy'] += cp.cuda.get_elapsed_time(ev1, ev2) / 1e3
    et = float(out.get()[0]) * 2
    STATS['triples_s'] += time.perf_counter() - t1s
    log.info('CCSD(T) correction = %.15g  (integrals %.2f s, triples %.2f s)',
             et, t1s - t0, time.perf_counter() - t1s)
    return et


def ccsd_t(self, t1=None, t2=None, eris=None):
    """GPU (T), PySCF's ``CCSD.ccsd_t`` signature (``eris`` is not used: the
    integrals are rebuilt on the device)."""
    if not (self._scf.istype('RHF') and not self._scf.istype('ROHF')):
        raise NotImplementedError('closed-shell (RHF) (T) only')
    return kernel(self, t1, t2, self.verbose)


from gpu4pyscf.cc import ccsd_incore                        # noqa: E402
from electrolites.features import attach                    # noqa: E402

attach(ccsd_incore.CCSDBase, 'ccsd_t', ccsd_t)
