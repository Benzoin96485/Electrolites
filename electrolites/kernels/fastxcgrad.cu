// Row-wise grid reductions for the density-contracted XC gradient.
//
// Both kernels reduce over grid points and keep the AO index -- the shape
// cuTENSOR handles worst here.  Measured on PfPMT/6-31G* (nao_sub 180 and
// 320, 4096 points), one contract('nig,ig->ni') costs 0.080 / 0.101 ms and one
// contract('ig,ig->i') costs 0.070 / 0.072 ms: essentially independent of how
// much data it touches, so it is cuTENSOR plan creation and cupy dispatch, not
// memory traffic.  With 830 grid blocks and up to six such call sites per
// block that overhead is most of the accumulate step (23 % of a GGA XC
// gradient, 30 % of a meta-GGA one).
//
// xcg_hessdot additionally does in one pass what the Python version needed
// eight calls for: the tau part of the meta-GGA gradient contracts the AO
// Hessian against three per-(i,g) arrays, and the packed Hessian order
// XX XY XZ YY YZ ZZ has no three contiguous components that form a row of it.

#ifndef NT
#define NT 256
#endif

#define BLOCK_REDUCE_3(s, a0, a1, a2)                          \
    s[threadIdx.x] = a0;                                       \
    s[NT + threadIdx.x] = a1;                                  \
    s[2 * NT + threadIdx.x] = a2;                              \
    __syncthreads();                                           \
    for (int k = NT / 2; k > 0; k >>= 1) {                     \
        if (threadIdx.x < k) {                                 \
            s[threadIdx.x] += s[threadIdx.x + k];              \
            s[NT + threadIdx.x] += s[NT + threadIdx.x + k];    \
            s[2 * NT + threadIdx.x] += s[2 * NT + threadIdx.x + k]; \
        }                                                      \
        __syncthreads();                                       \
    }

// out[n][i] = beta*out[n][i] + sum_g A[n][i][g] * b[i][g],  n = 0,1,2
// A is (3, nao, ngrids) and b is (nao, ngrids), both C-contiguous.
extern "C" __global__
void xcg_rowdot3(double *out, const double *A, const double *b,
                 int nao, int ngrids, double beta)
{
    const int i = blockIdx.x;
    if (i >= nao) return;
    __shared__ double s[3 * NT];
    const size_t stride = (size_t)nao * ngrids;
    const double *bi = b + (size_t)i * ngrids;
    const double *Ai = A + (size_t)i * ngrids;
    double a0 = 0., a1 = 0., a2 = 0.;
    for (int g = threadIdx.x; g < ngrids; g += NT) {
        const double bv = bi[g];
        a0 = fma(Ai[g], bv, a0);
        a1 = fma(Ai[stride + g], bv, a1);
        a2 = fma(Ai[2 * stride + g], bv, a2);
    }
    BLOCK_REDUCE_3(s, a0, a1, a2)
    if (threadIdx.x == 0) {
        out[i]           = beta == 0. ? s[0]        : fma(beta, out[i],           s[0]);
        out[nao + i]     = beta == 0. ? s[NT]       : fma(beta, out[nao + i],     s[NT]);
        out[2 * nao + i] = beta == 0. ? s[2 * NT]   : fma(beta, out[2 * nao + i], s[2 * NT]);
    }
}

// out[n][i] = beta*out[n][i] + sum_b sum_g H[n][b][i][g] * V[b][i][g]
// H is the AO Hessian in the packed order ao2 = XX XY XZ YY YZ ZZ, i.e.
// H[0] = (XX, XY, XZ), H[1] = (XY, YY, YZ), H[2] = (XZ, YZ, ZZ).
// ao2 is (6, nao, ngrids), V is (3, nao, ngrids), both C-contiguous.
extern "C" __global__
void xcg_hessdot(double *out, const double *ao2, const double *V,
                 int nao, int ngrids, double beta)
{
    const int i = blockIdx.x;
    if (i >= nao) return;
    __shared__ double s[3 * NT];
    const size_t stride = (size_t)nao * ngrids;
    const double *h = ao2 + (size_t)i * ngrids;
    const double *v = V + (size_t)i * ngrids;
    double a0 = 0., a1 = 0., a2 = 0.;
    for (int g = threadIdx.x; g < ngrids; g += NT) {
        const double v0 = v[g], v1 = v[stride + g], v2 = v[2 * stride + g];
        const double xx = h[g],              xy = h[stride + g],
                     xz = h[2 * stride + g], yy = h[3 * stride + g],
                     yz = h[4 * stride + g], zz = h[5 * stride + g];
        a0 = fma(xz, v2, fma(xy, v1, fma(xx, v0, a0)));
        a1 = fma(yz, v2, fma(yy, v1, fma(xy, v0, a1)));
        a2 = fma(zz, v2, fma(yz, v1, fma(xz, v0, a2)));
    }
    BLOCK_REDUCE_3(s, a0, a1, a2)
    if (threadIdx.x == 0) {
        out[i]           = beta == 0. ? s[0]        : fma(beta, out[i],           s[0]);
        out[nao + i]     = beta == 0. ? s[NT]       : fma(beta, out[nao + i],     s[NT]);
        out[2 * nao + i] = beta == 0. ? s[2 * NT]   : fma(beta, out[2 * nao + i], s[2 * NT]);
    }
}

// The whole per-block contribution in one pass:
//
//   rT[n][i] = sum_g [ nabla_n phi_i * u[i][g]
//                      + sum_b H[n][b][i][g] * V[b][i][g] ]
//   u[i][g]    = t[i][g] + wv[0][g] c[0][i][g]
//   V[b][i][g] = wv[1+b][g] c[0][i][g]  ( + wv[4][g] c[1+b][i][g] for mGGA )
//
// which is the density-contracted form of GPU4PySCF's _gga_grad_sum_ plus, for
// meta-GGA, _tau_grad_dot_.  Fusing them removes the (3, nao, ngrids)
// intermediate that _make_dR_dao_w writes and xcg_rowdot3 immediately reads
// back, and it removes the _scale_ao that forms aow: the AO block is read
// once and nothing of grid size is written.
//
// mgga = 0: c is (1, nao, ngrids), wv is (4, ngrids), t is used.
// mgga = 1: c is (4, nao, ngrids), wv is (5, ngrids) and t is ignored --
//   D . sum_m ao[m] wv[m] = sum_m wv[m] (D . ao[m]) is formed here from the
//   density products meta-GGA already has, so u = 2 wv[0] c[0] + sum_{m>0}
//   wv[m] c[m] (wv[0] carries GPU4PySCF's factor 1/2 already).
extern "C" __global__
void xcg_exc1(double *rT, const double *ao, const double *c, const double *t,
              const double *wv, int nao, int ngrids, int mgga)
{
    const int i = blockIdx.x;
    if (i >= nao) return;
    __shared__ double s[3 * NT];
    const size_t stride = (size_t)nao * ngrids;
    const double *aoi = ao + (size_t)i * ngrids;
    const double *ci = c + (size_t)i * ngrids;
    const double *ti = t + (size_t)i * ngrids;
    double a0 = 0., a1 = 0., a2 = 0.;
    for (int g = threadIdx.x; g < ngrids; g += NT) {
        const double w0 = wv[g], w1 = wv[ngrids + g],
                     w2 = wv[2 * ngrids + g], w3 = wv[3 * ngrids + g];
        const double c0 = ci[g];
        double u, v0, v1, v2;
        v0 = w1 * c0;
        v1 = w2 * c0;
        v2 = w3 * c0;
        if (mgga) {
            const double w4 = wv[4 * ngrids + g];
            const double cx = ci[stride + g], cy = ci[2 * stride + g],
                         cz = ci[3 * stride + g];
            u = fma(w3, cz, fma(w2, cy, fma(w1, cx, 2. * w0 * c0)));
            v0 = fma(w4, cx, v0);
            v1 = fma(w4, cy, v1);
            v2 = fma(w4, cz, v2);
        } else {
            u = fma(w0, c0, ti[g]);
        }
        const double dx = aoi[stride + g], dy = aoi[2 * stride + g],
                     dz = aoi[3 * stride + g];
        const double xx = aoi[4 * stride + g], xy = aoi[5 * stride + g],
                     xz = aoi[6 * stride + g], yy = aoi[7 * stride + g],
                     yz = aoi[8 * stride + g], zz = aoi[9 * stride + g];
        a0 = fma(xz, v2, fma(xy, v1, fma(xx, v0, fma(dx, u, a0))));
        a1 = fma(yz, v2, fma(yy, v1, fma(xy, v0, fma(dy, u, a1))));
        a2 = fma(zz, v2, fma(yz, v1, fma(xz, v0, fma(dz, u, a2))));
    }
    BLOCK_REDUCE_3(s, a0, a1, a2)
    if (threadIdx.x == 0) {
        rT[i] = s[0];
        rT[nao + i] = s[NT];
        rT[2 * nao + i] = s[2 * NT];
    }
}
