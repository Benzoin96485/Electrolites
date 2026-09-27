// The three grid contractions of the XC response kernel, batched over the
// right-hand sides.
//
// `nr_rks_fxc` is asked for `nset` right-hand sides at once -- three dipole
// directions for a polarizability, a Davidson subspace for TDDFT, 3*natm for
// a Hessian's CPHF -- and GPU4PySCF walks them in a Python loop
// (`dft/numint.py:1301`, `:1313`).  So a grid block costs `4*nset`
// GDFTcontract_rho launches for the density, `nset` GDFTscale_ao launches and
// `2*nset` GEMMs for the potential, and each of those kernels re-reads all of
// `ao`.
//
//   fxc_rho1   rho1[i][n][g] = sum_u c0[i][u][g] * ao[n][u][g],  the n > 0
//              components doubled -- which is what eval_rho's Hermitian
//              branch applies afterwards in a separate kernel.
//   fxc_wv     wv[i][y][g]   = weight[g] * sum_x rho1[i][x][g] * fxc[y][x][g],
//              with the `wv[0] *= .5` of the GGA potential folded in as
//              `scale0`.  This replaces `_contract_rho1_fxc`, whose
//              broadcast-then-sum materialises an
//              (nset, ncomp, ncomp, ngrids) temporary: four times the traffic
//              of its own result for GGA.
//   fxc_scale  aow[i][v][g]  = sum_n wv[i][n][g] * ao[n][v][g]
//
// Layouts, all C-contiguous: ao (ncomp, nao, ngrids), c0 (nset, nao, ngrids),
// rho1 and wv (nset, ncomp, ngrids), aow (nset, nao, ngrids).  `fxc` is a
// slice of the whole grid's cache and so carries its own stride.
//
// Thread mapping, the same in all three: threadIdx.x runs along the grid
// index, so every load and store is coalesced, and threadIdx.y runs along the
// right-hand sides.  Nothing is reduced across threads, so there is no shared
// memory and no __syncthreads, and the `ao` values a block reads are the same
// for all of its threadIdx.y rows -- they come from L1 rather than from
// memory, which is what makes reading `ao` once per set-tile instead of once
// per set almost free.
//
// The remaining serial dimension is the AO index.  fxc_rho1 has to sum over
// it, so it keeps it in the thread; fxc_scale writes one output per AO and can
// tile it over blockIdx.z, which is how it gets enough blocks in flight when
// `nset` is 3.

#ifndef WX
#define WX 32           // grid points per block; keep at the warp width
#endif
#ifndef TSY
#define TSY 4           // right-hand sides per block
#endif

// rho1[i][n][g] = sum_u c0[i][u][g] * ao[n][u][g]
template <int NCOMP> __device__ static void rho1_impl(
    double *rho1, const double *c0, const double *ao,
    int nset, int nao, int ngrids)
{
    const int g = blockIdx.x * WX + threadIdx.x;
    const int i = blockIdx.y * TSY + threadIdx.y;
    if (g >= ngrids || i >= nset) return;
    const size_t sg = ngrids;
    const double *ci = c0 + (size_t)i * nao * sg;

    double acc[NCOMP];
#pragma unroll
    for (int n = 0; n < NCOMP; ++n) acc[n] = 0.0;
    for (int u = 0; u < nao; ++u) {
        const double c = ci[(size_t)u * sg + g];
#pragma unroll
        for (int n = 0; n < NCOMP; ++n)
            acc[n] += c * ao[((size_t)n * nao + u) * sg + g];
    }
#pragma unroll
    for (int n = 0; n < NCOMP; ++n)
        rho1[((size_t)i * NCOMP + n) * sg + g] = (n ? 2.0 : 1.0) * acc[n];
}

// The meta-GGA density.  c0 is (4, nset, nao, ngrids) with
// c0[n] = D . ao[n]; eval_rho's MGGA branch needs all four because tau does.
//
//   rho1[i][0][g] = sum_u c0[0][i][u][g] ao[0][u][g]
//   rho1[i][n][g] = 2 sum_u c0[0][i][u][g] ao[n][u][g]        n = 1,2,3
//   rho1[i][4][g] = .5 sum_{n=1..3} sum_u c0[n][i][u][g] ao[n][u][g]
__device__ static void rho1_mgga_impl(
    double *rho1, const double *c0, const double *ao,
    int nset, int nao, int ngrids)
{
    const int g = blockIdx.x * WX + threadIdx.x;
    const int i = blockIdx.y * TSY + threadIdx.y;
    if (g >= ngrids || i >= nset) return;
    const size_t sg = ngrids;
    const size_t cstride = (size_t)nset * nao * sg;
    const double *ci = c0 + (size_t)i * nao * sg;

    double acc[5];
#pragma unroll
    for (int n = 0; n < 5; ++n) acc[n] = 0.0;
    for (int u = 0; u < nao; ++u) {
        const double c = ci[(size_t)u * sg + g];
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const double a = ao[((size_t)n * nao + u) * sg + g];
            acc[n] += c * a;
            if (n > 0) acc[4] += ci[n * cstride + (size_t)u * sg + g] * a;
        }
    }
    rho1[(size_t)i * 5 * sg + g] = acc[0];
#pragma unroll
    for (int n = 1; n < 4; ++n)
        rho1[((size_t)i * 5 + n) * sg + g] = 2.0 * acc[n];
    rho1[((size_t)i * 5 + 4) * sg + g] = .5 * acc[4];
}

// wv[i][y][g] = weight[g] * sum_x rho1[i][x][g] * fxc[y][x][g]
template <int NCOMP> __device__ static void wv_impl(
    double *wv, const double *rho1, const double *fxc, const double *weight,
    int nset, int ngrids, int fxc_stride, double scale0, double scale4)
{
    const int g = blockIdx.x * WX + threadIdx.x;
    const int i = blockIdx.y * TSY + threadIdx.y;
    if (g >= ngrids || i >= nset) return;
    const size_t sg = ngrids;
    const double w = weight[g];

    double r[NCOMP];
#pragma unroll
    for (int x = 0; x < NCOMP; ++x)
        r[x] = rho1[((size_t)i * NCOMP + x) * sg + g];
#pragma unroll
    for (int y = 0; y < NCOMP; ++y) {
        double s = 0.0;
#pragma unroll
        for (int x = 0; x < NCOMP; ++x)
            s += r[x] * fxc[((size_t)y * NCOMP + x) * fxc_stride + g];
        wv[((size_t)i * NCOMP + y) * sg + g] =
            w * s * (y == 0 ? scale0 : (y == 4 ? scale4 : 1.0));
    }
}

// The unrestricted `fxc` contraction.  On an unrestricted reference `fxc`
// couples the two spin channels, so the index contracted over is the *pair*
// (spin, component) and runs over 2*NC values where the restricted one runs
// over NC:
//
//   wv[i][t][y][g] = weight[g] * sum_{s,x} rho1[i][s][x][g] * fxc[t][y][s][x][g]
//
// This is the only kernel here that needs an unrestricted twin, and the
// reason is a layout choice on the Python side: the two spins' density
// matrices are interleaved a,b,a,b,..., so `rho1` for `nset` unrestricted
// sets occupies exactly the memory of `rho1` for `2*nset` restricted sets of
// NC components, and `fxc_rho1_*` and `fxc_scale_*` serve both cases with
// `nset` doubled and nothing else changed.  Only the contraction has to know
// that the pair index is a pair.
//
// `scale0` and `scale4` apply to the component *within* a spin, so the test
// is on `y - NC` once `y` has passed the first channel.
template <int NC> __device__ static void wv_u_impl(
    double *wv, const double *rho1, const double *fxc, const double *weight,
    int nset, int ngrids, int fxc_stride, double scale0, double scale4)
{
    const int g = blockIdx.x * WX + threadIdx.x;
    const int i = blockIdx.y * TSY + threadIdx.y;
    if (g >= ngrids || i >= nset) return;
    const int N2 = 2 * NC;
    const size_t sg = ngrids;
    const double w = weight[g];

    double r[2 * NC];
#pragma unroll
    for (int x = 0; x < N2; ++x)
        r[x] = rho1[((size_t)i * N2 + x) * sg + g];
#pragma unroll
    for (int y = 0; y < N2; ++y) {
        double s = 0.0;
#pragma unroll
        for (int x = 0; x < N2; ++x)
            s += r[x] * fxc[((size_t)y * N2 + x) * fxc_stride + g];
        const int c = y < NC ? y : y - NC;
        wv[((size_t)i * N2 + y) * sg + g] =
            w * s * (c == 0 ? scale0 : (c == 4 ? scale4 : 1.0));
    }
}

// aow[i][v][g] = sum_n wv[i][n][g] * ao[n][v][g]
// `wstride` is how many components a row of `wv` carries (5 for meta-GGA,
// where only four of them are the gradient part), `w0` the first one used and
// `a0` the first `ao` component paired with it.  The meta-GGA tau term is this
// kernel with NCOMP = 1, w0 = 4 and a0 = 1, 2, 3.
template <int NCOMP> __device__ static void scale_impl(
    double *aow, const double *wv, const double *ao,
    int nset, int nao, int ngrids, int nu_per_block,
    int wstride, int w0, int a0)
{
    const int g = blockIdx.x * WX + threadIdx.x;
    const int i = blockIdx.y * TSY + threadIdx.y;
    if (g >= ngrids || i >= nset) return;
    const size_t sg = ngrids;

    double w[NCOMP];
#pragma unroll
    for (int n = 0; n < NCOMP; ++n)
        w[n] = wv[((size_t)i * wstride + w0 + n) * sg + g];

    const int u0 = blockIdx.z * nu_per_block;
    const int u1 = min(u0 + nu_per_block, nao);
    double *ai = aow + (size_t)i * nao * sg;
    for (int u = u0; u < u1; ++u) {
        double s = 0.0;
#pragma unroll
        for (int n = 0; n < NCOMP; ++n)
            s += w[n] * ao[((size_t)(a0 + n) * nao + u) * sg + g];
        ai[(size_t)u * sg + g] = s;
    }
}

#define WRAP(name, impl, ncomp, args, call)                     \
    extern "C" __global__ void name args { impl<ncomp>call; }

WRAP(fxc_rho1_lda, rho1_impl, 1,
     (double *rho1, const double *c0, const double *ao,
      int nset, int nao, int ngrids),
     (rho1, c0, ao, nset, nao, ngrids))
WRAP(fxc_rho1_gga, rho1_impl, 4,
     (double *rho1, const double *c0, const double *ao,
      int nset, int nao, int ngrids),
     (rho1, c0, ao, nset, nao, ngrids))

WRAP(fxc_wv_lda, wv_impl, 1,
     (double *wv, const double *rho1, const double *fxc, const double *weight,
      int nset, int ngrids, int fxc_stride, double scale0, double scale4),
     (wv, rho1, fxc, weight, nset, ngrids, fxc_stride, scale0, scale4))
WRAP(fxc_wv_gga, wv_impl, 4,
     (double *wv, const double *rho1, const double *fxc, const double *weight,
      int nset, int ngrids, int fxc_stride, double scale0, double scale4),
     (wv, rho1, fxc, weight, nset, ngrids, fxc_stride, scale0, scale4))
WRAP(fxc_wv_mgga, wv_impl, 5,
     (double *wv, const double *rho1, const double *fxc, const double *weight,
      int nset, int ngrids, int fxc_stride, double scale0, double scale4),
     (wv, rho1, fxc, weight, nset, ngrids, fxc_stride, scale0, scale4))

extern "C" __global__ void fxc_rho1_mgga(
    double *rho1, const double *c0, const double *ao,
    int nset, int nao, int ngrids)
{ rho1_mgga_impl(rho1, c0, ao, nset, nao, ngrids); }

WRAP(fxc_wv_u_lda, wv_u_impl, 1,
     (double *wv, const double *rho1, const double *fxc, const double *weight,
      int nset, int ngrids, int fxc_stride, double scale0, double scale4),
     (wv, rho1, fxc, weight, nset, ngrids, fxc_stride, scale0, scale4))
WRAP(fxc_wv_u_gga, wv_u_impl, 4,
     (double *wv, const double *rho1, const double *fxc, const double *weight,
      int nset, int ngrids, int fxc_stride, double scale0, double scale4),
     (wv, rho1, fxc, weight, nset, ngrids, fxc_stride, scale0, scale4))
WRAP(fxc_wv_u_mgga, wv_u_impl, 5,
     (double *wv, const double *rho1, const double *fxc, const double *weight,
      int nset, int ngrids, int fxc_stride, double scale0, double scale4),
     (wv, rho1, fxc, weight, nset, ngrids, fxc_stride, scale0, scale4))

WRAP(fxc_scale_lda, scale_impl, 1,
     (double *aow, const double *wv, const double *ao,
      int nset, int nao, int ngrids, int nu_per_block,
      int wstride, int w0, int a0),
     (aow, wv, ao, nset, nao, ngrids, nu_per_block, wstride, w0, a0))
WRAP(fxc_scale_gga, scale_impl, 4,
     (double *aow, const double *wv, const double *ao,
      int nset, int nao, int ngrids, int nu_per_block,
      int wstride, int w0, int a0),
     (aow, wv, ao, nset, nao, ngrids, nu_per_block, wstride, w0, a0))

// ---- W33: the occupied-side route of a Hessian's CPHF --------------------
//
// A CPHF right-hand side is a density matrix with factors,
// D_i = M_i (2 C)^T + (2 C) M_i^T, and what the caller wants of the potential
// is only V_i C (`hessian/rhf.py:_ao2mo` multiplies by C next).  So on a
// block where the occupied count is well below the block's AO count both
// halves can run on the occupied side, with
//
//   X[n][k][g]    = sum_u C[u][k] ao[n][u][g]              shared by every set
//   Y[n][i][k][g] = sum_u M_i[u][k] ao[n][u][g]
//
// and sum_uv D_uv a_u b_v = 2 sum_k (Y^a X^b + X^a Y^b).  In the conventions
// of fxc_rho1_mgga (gradient components doubled, tau halved):
//
//   rho1[i][0] = 4 sum_k Y0 X0
//   rho1[i][x] = 4 sum_k (Yx X0 + Y0 Xx)                     x = 1,2,3
//   rho1[i][4] = 2 sum_x sum_k Yx Xx
//
// A GGA is the same without the tau row (NC = 4).  Layouts: Y (4, nset,
// nocc, ngrids), X (4, nocc, ngrids); set i's NC rows of rho1 start at row
// i * rss, so the unrestricted caller can write one spin of its interleaved
// (nset, 2, NC, ngrids) array (rss = 2 NC, pointer offset by the spin).  One
// thread per (grid point, set), the k loop in the thread; the X loads are the
// same for a block's TSY rows and come from L1.
template <int NC> __device__ static void rho_occ_impl(
    double *rho1, const double *Y, const double *X,
    int nset, int nocc, int ngrids, int rss)
{
    const int g = blockIdx.x * WX + threadIdx.x;
    const int i = blockIdx.y * TSY + threadIdx.y;
    if (g >= ngrids || i >= nset) return;
    const size_t sg = ngrids;
    const size_t ys = (size_t)nset * nocc * sg;     // stride of Y's n index
    const size_t xs = (size_t)nocc * sg;            // stride of X's n index
    const double *yi = Y + (size_t)i * nocc * sg;
    double r0 = 0., r1 = 0., r2 = 0., r3 = 0., t = 0.;
    for (int k = 0; k < nocc; ++k) {
        const size_t o = (size_t)k * sg + g;
        const double y0 = yi[o], y1 = yi[ys + o], y2 = yi[2 * ys + o],
                     y3 = yi[3 * ys + o];
        const double x0 = X[o], x1 = X[xs + o], x2 = X[2 * xs + o],
                     x3 = X[3 * xs + o];
        r0 += y0 * x0;
        r1 += y1 * x0 + y0 * x1;
        r2 += y2 * x0 + y0 * x2;
        r3 += y3 * x0 + y0 * x3;
        if (NC == 5) t += y1 * x1 + y2 * x2 + y3 * x3;
    }
    double *ri = rho1 + (size_t)i * rss * sg + g;
    ri[0] = 4. * r0;
    ri[sg] = 4. * r1;
    ri[2 * sg] = 4. * r2;
    ri[3 * sg] = 4. * r3;
    if (NC == 5) ri[4 * sg] = 2. * t;
}

// The potential on the occupied side.  With T = sum_g [sum_c wv_c ao_c ao_0^T
// + sum_x wv_4 ao_x ao_x^T] (what fxc_scale and the GEMMs build, the tau term
// only for meta-GGA) and V = T + T^T,
//
//   (V C)[u][k] = ao_0 [wv_0 X0 + sum_c wv_c Xc] + sum_x ao_x [wv_x X0
//                 + 2 wv_4 Xx]                                (c = 0..3)
//
// Meta-GGA takes it as four GEMMs of inner dimension ngrids and output width
// nset*nocc, (V C) = sum_n ao[n] . W[n]^T with
//
//   W[0] = 2 wv_0 X0 + sum_x wv_x Xx
//   W[x] = wv_x X0 + 2 wv_4 Xx                              x = 1,2,3
//
// A GGA has no tau term, so its W[x] is a rank-one scaling of X0 and the
// x terms fold with the wv_0 one into aow . X0^T, aow = sum_c wv_c ao_c being
// what fxc_scale_gga already writes:
//
//   (V C) = ao_0 . Wc^T + aow . X0^T,     Wc = sum_c wv_c Xc
//
// -- two GEMMs where the meta-GGA needs four.  W has Y's layout and is
// written into Y's memory; for a GGA only W[0] = Wc is written.  Set i's
// rows of wv start at row i * wss (the unrestricted caller's interleaving,
// as for rho_occ).
template <int NC> __device__ static void w_occ_impl(
    double *W, const double *wv, const double *X,
    int nset, int nocc, int ngrids, int wss)
{
    const int g = blockIdx.x * WX + threadIdx.x;
    const int i = blockIdx.y * TSY + threadIdx.y;
    if (g >= ngrids || i >= nset) return;
    const size_t sg = ngrids;
    const size_t ws = (size_t)nset * nocc * sg;
    const size_t xs = (size_t)nocc * sg;
    const double *wi = wv + (size_t)i * wss * sg + g;
    const double w0 = (NC == 5 ? 2. : 1.) * wi[0], w1 = wi[sg],
                 w2 = wi[2 * sg], w3 = wi[3 * sg];
    const double w4 = NC == 5 ? 2. * wi[4 * sg] : 0.;
    double *Wi = W + (size_t)i * nocc * sg;
    for (int k = 0; k < nocc; ++k) {
        const size_t o = (size_t)k * sg + g;
        const double x0 = X[o], x1 = X[xs + o], x2 = X[2 * xs + o],
                     x3 = X[3 * xs + o];
        Wi[o] = w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;
        if (NC == 5) {
            Wi[ws + o] = w1 * x0 + w4 * x1;
            Wi[2 * ws + o] = w2 * x0 + w4 * x2;
            Wi[3 * ws + o] = w3 * x0 + w4 * x3;
        }
    }
}

WRAP(fxc_rho_occ_gga, rho_occ_impl, 4,
     (double *rho1, const double *Y, const double *X,
      int nset, int nocc, int ngrids, int rss),
     (rho1, Y, X, nset, nocc, ngrids, rss))
WRAP(fxc_rho_occ_mgga, rho_occ_impl, 5,
     (double *rho1, const double *Y, const double *X,
      int nset, int nocc, int ngrids, int rss),
     (rho1, Y, X, nset, nocc, ngrids, rss))
WRAP(fxc_w_occ_gga, w_occ_impl, 4,
     (double *W, const double *wv, const double *X,
      int nset, int nocc, int ngrids, int wss),
     (W, wv, X, nset, nocc, ngrids, wss))
WRAP(fxc_w_occ_mgga, w_occ_impl, 5,
     (double *W, const double *wv, const double *X,
      int nset, int nocc, int ngrids, int wss),
     (W, wv, X, nset, nocc, ngrids, wss))
