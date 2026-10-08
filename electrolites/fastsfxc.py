"""The spin-flip multi-collinear XC kernel without the full tensors (D13).

``tdscf/_uhf_resp_sf.py:eval_xc_eff_sf`` -- the spin-flip TDA response and,
at third order, its gradient (``grad/tduks_sf.py``) -- evaluates the
collinear functional on 20 principal-axis samples per grid point through
``NumInt.eval_xc_eff(spin=1)``, which turns libxc's derivatives in
``(rho, sigma)`` into the full ``(2, nvar, 2, nvar[, 2, nvar])`` tensors in
``(rho, grad rho)`` (``xc_deriv.transform_kxc``: ~75 % of the evaluation,
measured, since it runs a chain of einsums over ``512 * ngrids * 20``
doubles), then into the total/spin basis (``ud2ts``), and keeps one block:
``fxc_ts[1,:,1]`` and ``kxc_ts[1,:,1,:,c]``, summed over the samples.

Here the samples' raw libxc derivatives (summed over the functional's
components, as ``eval_xc_eff`` does) go to one CUDA kernel that applies the
chain rule *in the total/spin basis* for just those blocks and sums over the
samples in registers:

    ts derivative = sum_ABC f3_ABC a~_A a~_B a~_C
                    + sum_AB f2_AB (b~_A a~_B + two permutations)

with ``A`` over ``(rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd)``, ``a~`` the
first derivatives of those with respect to the ts variables (linear in the
spin gradients) and ``b~`` their constant second derivatives.  Nothing of
size ``512 * ngrids * 20`` is formed.

LDA and GGA functionals at ``deriv`` 2 and 3; anything else (meta-GGA, a
functional libxc cannot do on the GPU) is GPU4PySCF's path.
``FASTSFXC_OFF=1`` keeps GPU4PySCF's.  ``docs/README_fastsfxc.md``.
"""
import functools
import os

import numpy as np
import cupy as cp

from gpu4pyscf.tdscf import _uhf_resp_sf as SF

_OFF = bool(os.environ.get('FASTSFXC_OFF'))
STATS = {'ours': 0, 'stock': 0}


def stats():
    return dict(STATS)


_SRC = r'''
// symmetric packed index of (i <= j) over 3 and (i <= j <= k) over 3
__device__ __forceinline__ int sp2(int i, int j)
{ if (i > j) { int t = i; i = j; j = t; }
  return i == 0 ? j : (i == 1 ? 2 + j : 5); }
__device__ __forceinline__ int sp3(int i, int j, int k)
{
    int a = i, b = j, c = k, t;
    if (a > b) { t = a; a = b; b = t; }
    if (b > c) { t = b; b = c; c = t; }
    if (a > b) { t = a; a = b; b = t; }
    // (000)0 (001)1 (002)2 (011)3 (012)4 (022)5 (111)6 (112)7 (122)8 (222)9
    if (a == 0) return b == 0 ? c : (b == 1 ? 2 + c : 5);
    if (a == 1) return b == 1 ? 5 + c : 8;
    return 9;
}

// A = 0 rho_u, 1 rho_d, 2 s_uu, 3 s_ud, 4 s_dd.  Raw libxc layouts, one row
// of ncomp per point: v2rho2 (uu,ud,dd), v2rhosigma (r*3+s), v2sigma2 sp2,
// v3rho3 (#d), v3rho2sigma (rr*3+s, rr by #d), v3rhosigma2 (r*6+sp2),
// v3sigma3 sp3.
__device__ __forceinline__ double f2(int A, int B, size_t p,
    const double* v2rho2, const double* v2rhosigma, const double* v2sigma2)
{
    if (A > B) { int t = A; A = B; B = t; }
    if (B < 2) return v2rho2[p*3 + A + B];
    if (A < 2) return v2rhosigma[p*6 + A*3 + (B-2)];
    return v2sigma2[p*6 + sp2(A-2, B-2)];
}
__device__ __forceinline__ double f3(int A, int B, int C, size_t p,
    const double* v3rho3, const double* v3rho2sigma,
    const double* v3rhosigma2, const double* v3sigma3)
{
    // sort so the rho indices come first
    int t;
    if (A > B) { t = A; A = B; B = t; }
    if (B > C) { t = B; B = C; C = t; }
    if (A > B) { t = A; A = B; B = t; }
    if (C < 2) return v3rho3[p*4 + A + B + C];
    if (B < 2) return v3rho2sigma[p*9 + (A + B)*3 + (C-2)];
    if (A < 2) return v3rhosigma2[p*12 + A*6 + sp2(B-2, C-2)];
    return v3sigma3[p*10 + sp3(A-2, B-2, C-2)];
}

// a~[A] for the ts variable (c, x): see the module docstring
__device__ __forceinline__ void atilde(double* a, int c, int x,
                                       const double* gu, const double* gd)
{
    const double m1 = c == 0 ? 1. : -1.;
    for (int A = 0; A < 5; A++) a[A] = 0.;
    if (x == 0) { a[0] = .5; a[1] = .5 * m1; }
    else {
        a[2] = gu[x-1];
        a[3] = .5 * (gd[x-1] + m1 * gu[x-1]);
        a[4] = m1 * gd[x-1];
    }
}
// b~[A] for the pair (c1, i) (c2, i), i > 0 -- constant
__device__ __forceinline__ void btilde(double* b, int c1, int c2)
{
    const double m1 = c1 == 0 ? 1. : -1., m2 = c2 == 0 ? 1. : -1.;
    b[0] = b[1] = 0.;
    b[2] = .5;
    b[3] = .25 * (m2 + m1);
    b[4] = .5 * m1 * m2;
}

template <int NVAR, int KXC>
__device__ void sf_body(double* fxc_sf, double* kxc_sf,
    const double* rho_tmz, const double* sgridz, const double* weights,
    const double* vrho, const double* vsigma,
    const double* v2rho2, const double* v2rhosigma, const double* v2sigma2,
    const double* v3rho3, const double* v3rho2sigma,
    const double* v3rhosigma2, const double* v3sigma3,
    const int ngrids, const int nsg)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int xy = blockIdx.y;              // x*NVAR + y
    if (g >= ngrids) return;
    const int x = xy / NVAR, y = xy % NVAR;
    // rho_tmz[t/m][var][g]
    double gt[3] = {0,0,0}, gm[3] = {0,0,0};
    if (NVAR == 4) for (int i = 0; i < 3; i++) {
        gt[i] = rho_tmz[(0*NVAR + 1 + i) * (size_t)ngrids + g];
        gm[i] = rho_tmz[(1*NVAR + 1 + i) * (size_t)ngrids + g];
    }
    double fx = 0., kx[2][NVAR];
    for (int c = 0; c < 2; c++) for (int z = 0; z < NVAR; z++) kx[c][z] = 0.;
    double b11[5];
    btilde(b11, 1, 1);
    for (int s = 0; s < nsg; s++) {
        const size_t p = (size_t)g * nsg + s;
        const double zs = sgridz[s], w = weights[s];
        double gu[3], gd[3];
        for (int i = 0; i < 3; i++) {
            gu[i] = .5 * (gt[i] + zs * gm[i]);
            gd[i] = .5 * (gt[i] - zs * gm[i]);
        }
        double pa[5], qa[5];
        atilde(pa, 1, x, gu, gd);
        atilde(qa, 1, y, gu, gd);
        const int A_end = NVAR == 1 ? 2 : 5;
        // fxc_ts[(1,x),(1,y)]
        double f = 0.;
        double f2pq[5];   // sum_A f2_AB p_A, for the kxc permutation terms
        for (int A = 0; A < A_end; A++) for (int B = 0; B < A_end; B++)
            f += f2(A, B, p, v2rho2, v2rhosigma, v2sigma2) * pa[A] * qa[B];
        if (NVAR == 4 && x == y && x > 0) {
            const double v1[5] = {0, 0, vsigma[p*3+0], vsigma[p*3+1], vsigma[p*3+2]};
            for (int A = 2; A < 5; A++) f += v1[A] * b11[A];
        }
        fx += w * f;
        if (KXC) {
            double T[5];
            for (int C = 0; C < A_end; C++) {
                double t = 0.;
                for (int A = 0; A < A_end; A++) {
                    if (pa[A] == 0.) continue;
                    for (int B = 0; B < A_end; B++) {
                        if (qa[B] == 0.) continue;
                        t += f3(A, B, C, p, v3rho3, v3rho2sigma, v3rhosigma2,
                                v3sigma3) * pa[A] * qa[B];
                    }
                }
                T[C] = t;
            }
            // the f2 * b~ terms
            double Uxy[5] = {0,0,0,0,0};     // sum_A b11_A f2_AC  (x == y)
            if (NVAR == 4 && x == y && x > 0)
                for (int C = 0; C < 5; C++) for (int A = 2; A < 5; A++)
                    Uxy[C] += b11[A] * f2(A, C, p, v2rho2, v2rhosigma, v2sigma2);
            for (int c = 0; c < 2; c++) {
                double b1c[5];
                btilde(b1c, 1, c);
                double Vx[5] = {0,0,0,0,0}, Vy[5] = {0,0,0,0,0};
                if (NVAR == 4) for (int B = 0; B < 5; B++) for (int A = 2; A < 5; A++) {
                    double fab = f2(A, B, p, v2rho2, v2rhosigma, v2sigma2);
                    Vy[B] += b1c[A] * fab;     // pairs with q (x == z term)
                    Vx[B] += b1c[A] * fab;     // pairs with p (y == z term)
                }
                const double wc = c == 0 ? w : w * zs;
                for (int z = 0; z < NVAR; z++) {
                    double r[5];
                    atilde(r, c, z, gu, gd);
                    double k = 0.;
                    for (int C = 0; C < A_end; C++) k += (T[C] + Uxy[C]) * r[C];
                    if (NVAR == 4 && z > 0) {
                        if (x == z) for (int B = 0; B < 5; B++) k += Vy[B] * qa[B];
                        if (y == z) for (int B = 0; B < 5; B++) k += Vx[B] * pa[B];
                    }
                    kx[c][z] += wc * k;
                }
            }
        }
    }
    fxc_sf[(size_t)xy * ngrids + g] = fx;
    if (KXC)
        for (int c = 0; c < 2; c++) for (int z = 0; z < NVAR; z++)
            // kxc_sf[x, y, c, z, g]
            kxc_sf[(((size_t)xy * 2 + c) * NVAR + z) * ngrids + g] = kx[c][z];
}

#define ENTRY(NAME, NVAR, KXC)                                                 \
extern "C" __global__ void NAME(double* fxc_sf, double* kxc_sf,                \
    const double* rho_tmz, const double* sgridz, const double* weights,        \
    const double* vrho, const double* vsigma, const double* v2rho2,            \
    const double* v2rhosigma, const double* v2sigma2, const double* v3rho3,    \
    const double* v3rho2sigma, const double* v3rhosigma2,                      \
    const double* v3sigma3, const int ngrids, const int nsg)                   \
{ sf_body<NVAR, KXC>(fxc_sf, kxc_sf, rho_tmz, sgridz, weights, vrho, vsigma,   \
    v2rho2, v2rhosigma, v2sigma2, v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3,  \
    ngrids, nsg); }
ENTRY(sf_lda_f, 1, 0)
ENTRY(sf_lda_k, 1, 1)
ENTRY(sf_gga_f, 4, 0)
ENTRY(sf_gga_k, 4, 1)
'''


@functools.lru_cache(maxsize=4)
def _module(device_id):
    return cp.RawModule(code=_SRC, options=('-std=c++17',), backend='nvrtc')


_LABELS = ('vrho', 'vsigma', 'v2rho2', 'v2rhosigma', 'v2sigma2', 'v3rho3',
           'v3rho2sigma', 'v3rhosigma2', 'v3sigma3')
_NCOMP = dict(vrho=2, vsigma=3, v2rho2=3, v2rhosigma=6, v2sigma2=6,
              v3rho3=4, v3rho2sigma=9, v3rhosigma2=12, v3sigma3=10)


def _raw(ni, xc_code, rho_ud, xctype, deriv):
    """libxc's polarized derivatives of the functional (its components summed
    with their weights, as ``NumInt.eval_xc_eff`` does), as ``(N, ncomp)``
    arrays; a label no component produces is zeros."""
    from gpu4pyscf.dft.numint import batched_vec_norm2, batched_vec_dot
    xcfuns = ni._init_xcfuns(xc_code, 1)
    n = rho_ud.shape[-1]
    inp = {}
    if xctype == 'LDA':
        inp['rho'] = cp.ascontiguousarray(rho_ud.reshape(2, n).T)
    else:
        inp['rho'] = cp.ascontiguousarray(rho_ud[:, 0].T)
        s = cp.empty((n, 3))
        s[:, 0] = batched_vec_norm2(rho_ud[0, 1:4].T)
        s[:, 1] = batched_vec_dot(rho_ud[0, 1:4].T, rho_ud[1, 1:4].T)
        s[:, 2] = batched_vec_norm2(rho_ud[1, 1:4].T)
        inp['sigma'] = s
    out = {}
    for xcfun, w in xcfuns:
        r = xcfun.compute(inp, do_exc=True, do_vxc=True, do_fxc=deriv > 1,
                          do_kxc=deriv > 2)
        for label in _LABELS:
            if label in r and r[label] is not None:
                v = cp.asarray(r[label]).reshape(n, -1) * w
                out[label] = out[label] + v if label in out else v
    for label in _LABELS:
        if label not in out:
            out[label] = cp.zeros((n, _NCOMP[label]))
        else:
            out[label] = cp.ascontiguousarray(out[label])
    return out


def _eval_sf(ni, xc_code, xctype, rho_tmz, deriv, collinear_samples):
    nvar = 1 if rho_tmz.ndim == 2 else rho_tmz.shape[1]
    ngrids = rho_tmz.shape[-1]
    sgridz, weights = SF._make_paxis_samples(collinear_samples)
    rho_ts = SF._project_spin_paxis2(rho_tmz, sgridz)
    t, s = rho_ts
    rho_ud = cp.stack([(t + s) * .5, (t - s) * .5])
    raw = _raw(ni, xc_code, rho_ud, xctype, deriv)
    rho_ud = rho_ts = t = s = None
    fxc_sf = cp.empty((nvar, nvar, ngrids))
    kxc_sf = cp.empty((nvar, nvar, 2, nvar, ngrids))
    kern = _module(cp.cuda.Device().id).get_function(
        ('sf_lda_' if nvar == 1 else 'sf_gga_') + ('k' if deriv > 2 else 'f'))
    rho = cp.ascontiguousarray(rho_tmz, dtype=np.float64)
    kern(((ngrids + 127) // 128, nvar * nvar), (128,),
         (fxc_sf, kxc_sf, rho, cp.ascontiguousarray(sgridz),
          cp.ascontiguousarray(weights),
          *[raw[label] for label in _LABELS],
          np.int32(ngrids), np.int32(len(weights))))
    if deriv < 3:
        return None, None, fxc_sf, cp.zeros_like(kxc_sf)
    return None, None, fxc_sf, kxc_sf


def mcfun_eval_xc_adapter_sf(ni, xc_code, collinear_samples):
    family = ni._xc_type(xc_code)
    stock = SF._mcfun_eval_xc_adapter_sf_orig(ni, xc_code, collinear_samples)
    if family not in ('LDA', 'GGA') or not all(
            x.on_gpu for x, _ in ni._init_xcfuns(xc_code, 1)):
        return stock

    def eval_xc_eff(xc_code_, rho, deriv=1, omega=None, xctype=None,
                    verbose=None):
        if deriv not in (2, 3):
            STATS['stock'] += 1
            return stock(xc_code_, rho, deriv, omega, xctype, verbose)
        STATS['ours'] += 1
        rho = cp.asarray(rho)
        ngrids = rho.shape[-1]
        res = []
        for p0, p1 in SF._prange(0, ngrids, SF.MAX_GRIDS_PER_TASK):
            res.append(_eval_sf(ni, xc_code, family, rho[..., p0:p1], deriv,
                                collinear_samples))
        return [None if x[0] is None else cp.concatenate(x, axis=-1)
                for x in zip(*res)]
    return eval_xc_eff


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    SF._mcfun_eval_xc_adapter_sf_orig = SF.mcfun_eval_xc_adapter_sf
    SF.mcfun_eval_xc_adapter_sf = mcfun_eval_xc_adapter_sf
    from gpu4pyscf.grad import tduks_sf as G
    G.mcfun_eval_xc_adapter_sf = mcfun_eval_xc_adapter_sf
    _PATCHED = True


apply_patch()
