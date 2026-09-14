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
