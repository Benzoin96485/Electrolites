"""
The DF-MP2 pair-energy loops, batched (catalogue I1).

Importing this module replaces ``gpu4pyscf.mp.dfmp2_addons.
get_dfmp2_energy_pair_intra`` and ``get_dfump2_energy_pair_intra`` -- the
step that is 73-81 % of a DF-MP2 and 92-94 % of a DF-UMP2
(``docs/KERNEL_TARGETS_1.8.1.md`` §10.10, §12.4).  GPU4PySCF loops over the
occupied pairs ``i >= j`` (all ``i, j`` for the opposite-spin term), and per
pair runs one ``(nvir x naux) @ (naux x nvir)`` GEMM, four elementwise
kernels and **two host synchronisations** (``float(e_bi1)``,
``float(e_bi2)``) -- ``nocc^2/2`` of each.

Here the pairs go in blocks of ``B`` occupied orbitals: one
``(B nvir x naux) @ (naux x B nvir)`` GEMM per block pair gives ``B^2``
integral blocks ``g[i,a,j,b]``, and one kernel reduces each ``(i, j)`` slice
to its pair energies,

    bi1 = sum_ab g_ab^2 / D_ab      bi2 = sum_ab g_ab g_ba / D_ab
    ss  = sum_ab (g_ab - g_ba)^2 / D_ab      (D_ab = e_i + e_j - e_a - e_b)

with no host synchronisation until the end.  These are GPU4PySCF's
expressions (``t = g / D``, ``e = sum t g``) with the division folded in.
``B`` is chosen from free memory.  The amplitudes, when the caller asks
for them (``t2``, which ``with_t2`` -- on by default -- allocates), are
written block by block from the same GEMM output.  The pair sums are
accumulated in double precision whatever the precision of the integrals
(``dtype_cderi`` is float32 by default, and GPU4PySCF sums in float32).

``FASTDFMP2_OFF=1`` keeps GPU4PySCF's loops.  ``docs/README_fastdfmp2.md``.
"""
import os

import cupy as cp
import numpy as np

from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import get_avail_mem
from gpu4pyscf.mp import dfmp2_addons as A

_OFF = bool(os.environ.get('FASTDFMP2_OFF'))
#: occupied orbitals per block; 0 chooses from free memory
_B = int(os.environ.get('FASTDFMP2_BLOCK', 0))
STATS = {'ours': 0, 'stock': 0}


def stats():
    return dict(STATS)


_SRC = r'''
template <typename T> __device__ void
pair_energy_body(double *e1, double *e2, const T *g, const double *eo_i,
                 const double *eo_j, const double *ev_a, const double *ev_b,
                 int bi, int bj, int nva, int nvb, int mode)
{
    // one thread block per (i, j) pair of the block pair; g is the GEMM
    // output (bi*nva, bj*nvb): g[(i*nva + a)*(bj*nvb) + j*nvb + b]
    // mode 0: bi1 and bi2 (restricted)  1: same spin  2: opposite spin
    const int p = blockIdx.x;
    const int i = p / bj, j = p % bj;
    const size_t ld = (size_t)bj * nvb;
    const T *gi = g + (size_t)i * nva * ld + (size_t)j * nvb;
    const double eij = eo_i[i] + eo_j[j];
    double s1 = 0., s2 = 0.;
    for (int ab = threadIdx.x; ab < nva * nvb; ab += blockDim.x) {
        const int a = ab / nvb, b = ab % nvb;
        const double gab = gi[(size_t)a * ld + b];
        const double rd = 1. / (eij - ev_a[a] - ev_b[b]);
        if (mode == 2) {
            s1 += gab * gab * rd;
        } else {
            const double gba = gi[(size_t)b * ld + a];
            if (mode == 0) {
                s1 += gab * gab * rd;
                s2 += gab * gba * rd;
            } else {
                const double d = gab - gba;
                s1 += d * d * rd;
            }
        }
    }
    __shared__ double r1[256], r2[256];
    r1[threadIdx.x] = s1; r2[threadIdx.x] = s2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            r1[threadIdx.x] += r1[threadIdx.x + s];
            r2[threadIdx.x] += r2[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        e1[p] = r1[0];
        e2[p] = r2[0];
    }
}
#define ENTRY(T, NAME)                                                         \
extern "C" __global__ void NAME(double *e1, double *e2, const T *g,          \
        const double *eo_i, const double *eo_j, const double *ev_a,          \
        const double *ev_b, int bi, int bj, int nva, int nvb, int mode)      \
{ pair_energy_body<T>(e1, e2, g, eo_i, eo_j, ev_a, ev_b, bi, bj, nva, nvb,   \
                      mode); }
ENTRY(double, pair_energy_f64)
ENTRY(float, pair_energy_f32)
'''
_M = []


def _kernel(dtype):
    if not _M:
        _M.append(cp.RawModule(code=_SRC, options=('-std=c++17',)))
    return _M[0].get_function('pair_energy_f32' if dtype == np.float32
                              else 'pair_energy_f64')


def _block(nocc, nvir_a, nvir_b, naux, with_t2=False):
    if _B:
        return max(1, min(_B, nocc))
    avail = get_avail_mem() * .4
    # B^2 nva nvb for g, and three more of that size for the amplitudes
    b = int((avail / ((32. if with_t2 else 8.) * nvir_a * nvir_b)) ** .5)
    # 16 measured best at both 81 x 349 and 226 x 897 (occupied x virtual):
    # larger blocks waste more of the diagonal block pairs, smaller ones
    # leave the GEMM too short
    return max(1, min(nocc, b, 16))


def _pairs(c0, c1, eo0, eo1, ev0, ev1, mode, tri, t2=None):
    """Pair energies of every (i in c0, j in c1); lower triangle if tri.

    ``t2``, when given, receives the amplitudes ``t[i,j,a,b] = g/D`` (the
    antisymmetrised ones for the same-spin mode), and for ``tri`` their
    ``(j, i)`` transposes, exactly as GPU4PySCF's loops store them.
    """
    n0, nva, naux = c0.shape
    n1, nvb, _ = c1.shape
    B = _block(max(n0, n1), nva, nvb, naux, t2 is not None)
    out1 = cp.zeros((n0, n1))
    out2 = cp.zeros((n0, n1))
    kern = _kernel(c0.dtype)
    on_gpu = isinstance(t2, cp.ndarray)
    for i0 in range(0, n0, B):
        i1 = min(n0, i0 + B)
        a = c0[i0:i1].reshape(-1, naux)
        for j0 in range(0, (i1 if tri else n1), B):
            j1 = min(n1, j0 + B)
            g = a @ c1[j0:j1].reshape(-1, naux).T
            bi, bj = i1 - i0, j1 - j0
            e1 = cp.empty(bi * bj)
            e2 = cp.empty(bi * bj)
            kern((bi * bj,), (256,),
                 (e1, e2, g, eo0[i0:i1], eo1[j0:j1], ev0, ev1,
                  np.int32(bi), np.int32(bj), np.int32(nva), np.int32(nvb),
                  np.int32(mode)))
            out1[i0:i1, j0:j1] = e1.reshape(bi, bj)
            out2[i0:i1, j0:j1] = e2.reshape(bi, bj)
            if t2 is not None:
                gb = g.reshape(bi, nva, bj, nvb).transpose(0, 2, 1, 3)
                if mode == 1:
                    gb = gb - gb.transpose(0, 1, 3, 2)
                d = (eo0[i0:i1, None, None, None] + eo1[None, j0:j1, None, None]
                     - ev0[None, None, :, None] - ev1[None, None, None, :])
                tb = (gb / d).astype(t2.dtype)
                for ii in range(bi):
                    i = i0 + ii
                    jmax = min(j1, i + 1) if tri else j1
                    if jmax <= j0:
                        continue
                    blk = tb[ii, :jmax - j0]
                    if mode == 1:          # the diagonal pair is zero
                        if i < jmax:
                            blk = blk.copy()
                            blk[i - j0] = 0.
                    t2[i, j0:jmax] = blk if on_gpu else blk.get()
                    if tri:
                        blkT = blk.transpose(0, 2, 1)
                        t2[j0:jmax, i] = blkT if on_gpu else blkT.get()
                del gb, d, tb
            del g
    return out1, out2


def get_dfmp2_energy_pair_intra(streamobj, cderi_ovl, occ_energy, vir_energy,
                                ss_only=False, t2=None, log=None):
    STATS['ours'] += 1
    if log is None:
        log = logger.new_logger(streamobj, verbose=streamobj.verbose)
    t0 = log.init_timer()
    c = cp.asarray(cderi_ovl)
    eo = cp.asarray(occ_energy, dtype=np.float64)
    ev = cp.asarray(vir_energy, dtype=np.float64)
    nocc = len(eo)
    e1, e2 = _pairs(c, c, eo, eo, ev, ev, 1 if ss_only else 0, True, t2)
    lower = cp.tril(cp.ones((nocc, nocc), dtype=bool), -1 if ss_only else 0)
    e1 = cp.where(lower, e1, 0.)
    e2 = cp.where(lower, e2, 0.)
    e1 = e1 + cp.tril(e1, -1).T
    e2 = e2 + cp.tril(e2, -1).T
    log.timer('get_dfmp2_energy_pair_intra (fastdfmp2)', *t0)
    if ss_only:
        return e1.get()
    return e1.get(), e2.get()


def get_dfump2_energy_pair_intra(streamobj, cderi_ovl, occ_energy, vir_energy,
                                 t2=None, log=None):
    STATS['ours'] += 1
    if log is None:
        log = logger.new_logger(streamobj, verbose=streamobj.verbose)
    t0 = log.init_timer()
    c = [cp.asarray(x) for x in cderi_ovl]
    eo = [cp.asarray(x, dtype=np.float64) for x in occ_energy]
    ev = [cp.asarray(x, dtype=np.float64) for x in vir_energy]
    e1, _ = _pairs(c[0], c[1], eo[0], eo[1], ev[0], ev[1], 2, False, t2)
    log.timer('get_dfump2_energy_pair_intra (fastdfmp2)', *t0)
    return e1.get()


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    A._get_dfmp2_energy_pair_intra_orig = A.get_dfmp2_energy_pair_intra
    A._get_dfump2_energy_pair_intra_orig = A.get_dfump2_energy_pair_intra
    # the drivers call them as `dfmp2_addons.<name>` at call time
    A.get_dfmp2_energy_pair_intra = get_dfmp2_energy_pair_intra
    A.get_dfump2_energy_pair_intra = get_dfump2_energy_pair_intra
    _PATCHED = True


apply_patch()
