// The VV10 response pair sum, `VXC_vv10nlc_hess_eval_f_t` of GPU4PySCF 1.8.1
// (`lib/gdft/vv10.cu:286-369`), restructured as a GEMM whose left operand is
// built on the fly.
//
// For every pair of grid points (i, j) the response needs four numbers --
//
//     K_rr(i,j)  K_rg(i,j)       f_rho_t[t,i]   += sum_j  K_rr Xr[t,j] + K_rg Xg[t,j]
//     K_gr(i,j)  K_gg(i,j)       f_gamma_t[t,i] += sum_j  K_gr Xr[t,j] + K_gg Xg[t,j]
//
// with Xr = w rho_t and Xg = w gamma_t, one row per trial density t.
// GPU4PySCF gives each thread one i and six trials, so with `ntrial` trials
// every pair's ~60 operations and one fp64 division are repeated
// ceil(ntrial/6) times -- 21 times for a 42-atom Hessian's 126 -- and the
// contraction, four FMAs per (pair, trial), runs on the fp64 SIMT pipe.
//
// Here each pair is evaluated once per trial *group* (up to 8*NTW trials) and
// the contraction is an fp64 tensor-core MMA (mma.m8n8k4.f64).  The fragment
// layout makes the build free of shared-memory traffic: in an m8n8k4 MMA
// lane l holds A[l>>2][l&3], so if rows are i points and the k index is j
// points, **the lane that evaluates pair (i0 + l>>2, j0 + l&3) holds exactly
// the four A-fragment elements of that pair**, for the four MMAs
//
//     Cr += K_rr . Xr + K_rg . Xg        Cg += K_gr . Xr + K_gg . Xg
//
// Only the B operand (Xr, Xg for four j and eight trials) comes from shared
// memory, and it is reused by every m-tile of the warp.
//
// The reciprocal 1/(g_ij g_ji (g_ij+g_ji)) is a MUFU.RCP64H seed plus two
// Newton steps (`fastvv10.cu`'s RCP=1); g >= kappa > 0.28 for every point the
// callers keep (rho >= 1e-10), so the product is bounded away from zero.
//
// Point data are passed as one structure of arrays P[9][npad]:
//   x, y, z, omega, kappa, a0 = rho dkappa/drho, a1 = rho domega/drho,
//   b = rho domega/dgamma, w
// padded to npad (a multiple of the block's i extent and of JT) with
// omega = kappa = 1 and a0 = a1 = b = w = 0, so padding contributes exactly
// zero and the inner loop has no bounds tests.  The trial densities come in
// as X, already weighted, permuted and zero-padded (see ft_body).

#ifndef JT
#define JT 32          // j points per shared tile
#endif
#define XS (JT + 4)    // row stride of the X tile: 4 mod 16 doubles, so a
                       // B-fragment load is two conflict-free wavefronts

__device__ __forceinline__ double drcp(double x)
{
    double r;
    asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(r) : "d"(x));
    r = r * (2.0 - x * r);
    r = r * (2.0 - x * r);
    return r;
}

__device__ __forceinline__ void dmma(double &c0, double &c1, double a, double b)
{
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+d"(c0), "+d"(c1) : "d"(a), "d"(b));
}

// The four pair-kernel values.  Exactly GPU4PySCF's expressions, regrouped:
//   Phi = -inv,  D = d2Phi/(dg_ij dg_ji) / Phi,  Pi = 1/s + 1/g_ij, ...
//   K_rr = Phi (ri rj D - ri Pi - rj Pj + 1) = ri tj - rj Phi Pj + Phi
//   K_gr = bi Phi (rj D - Pi) = bi tj,   K_rg = bj Phi (ri D - Pj) = bj ti
//   K_gg = bi bj Phi D
// with tj = rj Phi D - Phi Pi and ti = ri Phi D - Phi Pj.
__device__ __forceinline__ void
pair_kernel(double &krr, double &krg, double &kgr, double &kgg,
            double xi, double yi, double zi, double wi, double ki,
            double a0i, double a1i, double bi,
            double xj, double yj, double zj, double wj, double kj,
            double a0j, double a1j, double bj)
{
    const double dx = xi - xj, dy = yi - yj, dz = zi - zj;
    const double r2 = dx * dx + dy * dy + dz * dz;
    const double gi = wi * r2 + ki;          // g_ij
    const double gj = wj * r2 + kj;          // g_ji
    const double s = gi + gj;
    const double gg = gi * gj;
    const double inv = drcp(gg * s);
    const double phi = -inv;
    const double ggi = s * inv;              // 1/(g_ij g_ji)
    const double si = gg * inv;              // 1/s
    const double phiD = 2. * phi * (si * si + ggi);
    const double phiPi = phi * (si + gj * ggi);
    const double phiPj = phi * (si + gi * ggi);
    const double ri = a0i + r2 * a1i;
    const double rj = a0j + r2 * a1j;
    const double bir = bi * r2, bjr = bj * r2;
    const double tj = rj * phiD - phiPi;
    const double ti = ri * phiD - phiPj;
    krr = ri * tj - rj * phiPj + phi;
    kgr = bir * tj;
    krg = bjr * ti;
    kgg = bir * bjr * phiD;
}

// The same four values in single precision, for pairs whose tiles are more
// than RCUT apart (`fastnlc`'s P10).  The reciprocal is one MUFU.RCP; the
// coordinates are relative to the molecule's centroid (the caller shifts
// them), so a far pair's dx carries ~2e-6 Bohr of noise against dx >= RCUT.
__device__ __forceinline__ void
pair_kernel_f(float &krr, float &krg, float &kgr, float &kgg,
              float xi, float yi, float zi, float wi, float ki,
              float a0i, float a1i, float bi,
              float xj, float yj, float zj, float wj, float kj,
              float a0j, float a1j, float bj)
{
    const float dx = xi - xj, dy = yi - yj, dz = zi - zj;
    const float r2 = dx * dx + dy * dy + dz * dz;
    const float gi = wi * r2 + ki;
    const float gj = wj * r2 + kj;
    const float s = gi + gj;
    const float gg = gi * gj;
    float inv;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(gg * s));
    const float phi = -inv;
    const float ggi = s * inv;
    const float si = gg * inv;
    const float phiD = 2.f * phi * (si * si + ggi);
    const float phiPi = phi * (si + gj * ggi);
    const float phiPj = phi * (si + gi * ggi);
    const float ri = a0i + r2 * a1i;
    const float rj = a0j + r2 * a1j;
    const float bir = bi * r2, bjr = bj * r2;
    const float tj = rj * phiD - phiPi;
    const float ti = ri * phiD - phiPj;
    krr = ri * tj - rj * phiPj + phi;
    kgr = bir * tj;
    krg = bjr * ti;
    kgg = bir * bjr * phiD;
}

// One block: NW warps, each MT m-tiles (8 MT i points) by NTW n-tiles
// (8 NTW trials).  blockIdx.x walks i, blockIdx.y the trial groups.
//
// The points are in the caller's order permuted by `perm` (P and X are
// already permuted and padded; the outputs are scattered through perm, and
// perm = -1 marks padding).  X holds Xr = w rho_t at X[t*npad + j] and
// Xg = w gamma_t at X[xoff + t*npad + j] for the trials tbase, tbase+1, ...
// of this launch.  With MIX, `wbox` holds each warp's i bounding
// box and `tbox` each JT-point j tile's, and a (warp, tile) pair whose boxes
// are more than sqrt(rcut2) apart is evaluated in fp32 -- a warp-uniform
// branch, so there is no divergence.  The MMA stays fp64 either way.
template <int MT, int NTW, int NW, bool MIX> __device__ static void
ft_body(double *fr, double *fg, const double *P, const float *Pf,
        const int *perm, const double *X, size_t xoff,
        const float *wbox, const float *tbox, float rcut2,
        int npad, int ng, int tbase, int ntrial)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int li = lane >> 2, lj = lane & 3;
    const int gw = blockIdx.x * NW + warp;
    const int ibase = gw * (8 * MT);
    const int t0 = blockIdx.y * (8 * NTW);

    double xi[MT], yi[MT], zi[MT], wi[MT], ki[MT], a0i[MT], a1i[MT], bi[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int i = ibase + 8 * m + li;
        xi[m] = P[i]; yi[m] = P[npad + i]; zi[m] = P[2 * npad + i];
        wi[m] = P[3 * npad + i]; ki[m] = P[4 * npad + i];
        a0i[m] = P[5 * npad + i]; a1i[m] = P[6 * npad + i];
        bi[m] = P[7 * npad + i];
    }
    float wb[6];
    if (MIX) {
#pragma unroll
        for (int d = 0; d < 6; ++d) wb[d] = wbox[6 * gw + d];
    }
    double cr[MT][NTW][2], cg[MT][NTW][2];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < NTW; ++n)
            cr[m][n][0] = cr[m][n][1] = cg[m][n][0] = cg[m][n][1] = 0.;

    __shared__ double sp[8][JT];
    __shared__ float spf[8][JT];
    __shared__ double sx[2][8 * NTW][XS];

    for (int j0 = 0; j0 < npad; j0 += JT) {
        __syncthreads();
        for (int e = threadIdx.x; e < 8 * JT; e += NW * 32) {
            const size_t g = (size_t)(e / JT) * npad + j0 + e % JT;
            sp[e / JT][e % JT] = P[g];
            if (MIX) spf[e / JT][e % JT] = Pf[g];
        }
        for (int e = threadIdx.x; e < 8 * NTW * JT; e += NW * 32) {
            const int t = e / JT, j = e % JT;
            const size_t g = (size_t)(t0 + t) * npad + j0 + j;
            sx[0][t][j] = X[g];
            sx[1][t][j] = X[xoff + g];
        }
        __syncthreads();
        bool far = false;
        if (MIX) {
            const float *tb = tbox + 6 * (j0 / JT);
            const float d0 = fmaxf(0.f, fmaxf(tb[0] - wb[3], wb[0] - tb[3]));
            const float d1 = fmaxf(0.f, fmaxf(tb[1] - wb[4], wb[1] - tb[4]));
            const float d2 = fmaxf(0.f, fmaxf(tb[2] - wb[5], wb[2] - tb[5]));
            far = d0 * d0 + d1 * d1 + d2 * d2 > rcut2;
        }
        if (MIX && far) {
            float xf[MT], yf[MT], zf[MT], wf[MT], kf[MT], a0f[MT], a1f[MT], bf[MT];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                xf[m] = (float)xi[m]; yf[m] = (float)yi[m]; zf[m] = (float)zi[m];
                wf[m] = (float)wi[m]; kf[m] = (float)ki[m];
                a0f[m] = (float)a0i[m]; a1f[m] = (float)a1i[m]; bf[m] = (float)bi[m];
            }
#pragma unroll 2
            for (int kk = 0; kk < JT; kk += 4) {
                const int j = kk + lj;
                double br[NTW], bg[NTW];
#pragma unroll
                for (int n = 0; n < NTW; ++n) {
                    br[n] = sx[0][8 * n + li][j];
                    bg[n] = sx[1][8 * n + li][j];
                }
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    float krr, krg, kgr, kgg;
                    pair_kernel_f(krr, krg, kgr, kgg,
                                  xf[m], yf[m], zf[m], wf[m], kf[m],
                                  a0f[m], a1f[m], bf[m],
                                  spf[0][j], spf[1][j], spf[2][j], spf[3][j],
                                  spf[4][j], spf[5][j], spf[6][j], spf[7][j]);
                    const double drr = krr, drg = krg, dgr = kgr, dgg = kgg;
#pragma unroll
                    for (int n = 0; n < NTW; ++n) {
                        dmma(cr[m][n][0], cr[m][n][1], drr, br[n]);
                        dmma(cr[m][n][0], cr[m][n][1], drg, bg[n]);
                        dmma(cg[m][n][0], cg[m][n][1], dgr, br[n]);
                        dmma(cg[m][n][0], cg[m][n][1], dgg, bg[n]);
                    }
                }
            }
            continue;
        }
#pragma unroll 2
        for (int kk = 0; kk < JT; kk += 4) {
            const int j = kk + lj;
            const double xj = sp[0][j], yj = sp[1][j], zj = sp[2][j];
            const double wj = sp[3][j], kj = sp[4][j];
            const double a0j = sp[5][j], a1j = sp[6][j], bj = sp[7][j];
            double br[NTW], bg[NTW];
#pragma unroll
            for (int n = 0; n < NTW; ++n) {
                br[n] = sx[0][8 * n + li][j];
                bg[n] = sx[1][8 * n + li][j];
            }
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                double krr, krg, kgr, kgg;
                pair_kernel(krr, krg, kgr, kgg,
                            xi[m], yi[m], zi[m], wi[m], ki[m],
                            a0i[m], a1i[m], bi[m],
                            xj, yj, zj, wj, kj, a0j, a1j, bj);
#pragma unroll
                for (int n = 0; n < NTW; ++n) {
                    dmma(cr[m][n][0], cr[m][n][1], krr, br[n]);
                    dmma(cr[m][n][0], cr[m][n][1], krg, bg[n]);
                    dmma(cg[m][n][0], cg[m][n][1], kgr, br[n]);
                    dmma(cg[m][n][0], cg[m][n][1], kgg, bg[n]);
                }
            }
        }
    }

    // C fragment: lane holds C[l>>2][2(l&3) + {0,1}], rows i, columns t
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int pi = perm[ibase + 8 * m + li];
        if (pi < 0) continue;
#pragma unroll
        for (int n = 0; n < NTW; ++n)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int t = tbase + t0 + 8 * n + 2 * lj + h;
                if (t < ntrial) {
                    fr[(size_t)t * ng + pi] = 1.5 * cr[m][n][h];
                    fg[(size_t)t * ng + pi] = 1.5 * cg[m][n][h];
                }
            }
    }
}

#define FT_ENTRY(MT, NTW, NW, MIX, NAME)                                       \
extern "C" __global__ void __launch_bounds__(NW * 32)                          \
NAME##_##MT##_##NTW##_##NW(double *fr, double *fg, const double *P,            \
        const float *Pf, const int *perm, const double *X, size_t xoff,        \
        const float *wbox, const float *tbox, float rcut2, int npad, int ng,   \
        int tbase, int ntrial)                                                 \
{ ft_body<MT, NTW, NW, MIX>(fr, fg, P, Pf, perm, X, xoff, wbox, tbox, rcut2,   \
                            npad, ng, tbase, ntrial); }
#define FT_BOTH(MT, NTW, NW) FT_ENTRY(MT, NTW, NW, false, ft_dmma)             \
                             FT_ENTRY(MT, NTW, NW, true, ft_mixed)

FT_BOTH(1, 1, 4)
FT_BOTH(2, 1, 4)
FT_BOTH(1, 2, 4)
FT_BOTH(2, 2, 4)
FT_BOTH(1, 4, 4)
FT_BOTH(2, 4, 4)
FT_BOTH(1, 8, 4)
FT_BOTH(2, 8, 4)
FT_BOTH(1, 4, 8)
FT_BOTH(2, 4, 8)
FT_BOTH(1, 8, 8)
FT_BOTH(1, 2, 8)

// ---------------------------------------------------------------------------
// `VXC_vv10nlc_hess_eval_EUW_with_weight1` (`vv10.cu:504`), the grid-weight
// derivative of the VV10 kernel sums, by the same construction: for each of
// the `nderivative` = 3 natm weight derivatives w1[d, j],
//
//     Ew[d,i] =  1.5 sum_j E_ij w1[d,j]      E_ij = -rho_j / (g_ij g_ji s)
//     Uw[d,i] = -1.5 sum_j U_ij w1[d,j]      U_ij = E_ij (s + g_ij) g_ji inv
//     Ww[d,i] = -1.5 sum_j U_ij r2 w1[d,j]
//
// GPU4PySCF again evaluates every pair once per six derivatives.  Here: once
// per 8 NTW, three fp64 MMAs per (pair quad, 8 derivatives).  Point data
// Q[6][npad] = x, y, z, omega, kappa, rho (padding: omega = kappa = 1,
// rho = 0); X[d][npad] = w1, permuted and zero-padded.

__device__ __forceinline__ void
euw_pair(double &e, double &u, double &w, double xi, double yi, double zi,
         double oi, double ki, double xj, double yj, double zj, double oj,
         double kj, double rj)
{
    const double dx = xi - xj, dy = yi - yj, dz = zi - zj;
    const double r2 = dx * dx + dy * dy + dz * dz;
    const double gi = oi * r2 + ki, gj = oj * r2 + kj, s = gi + gj;
    const double inv = drcp(gi * gj * s);
    e = -rj * inv;
    u = e * (s + gi) * gj * inv;
    w = u * r2;
}

__device__ __forceinline__ void
euw_pair_f(float &e, float &u, float &w, float xi, float yi, float zi,
           float oi, float ki, float xj, float yj, float zj, float oj,
           float kj, float rj)
{
    const float dx = xi - xj, dy = yi - yj, dz = zi - zj;
    const float r2 = dx * dx + dy * dy + dz * dz;
    const float gi = oi * r2 + ki, gj = oj * r2 + kj, s = gi + gj;
    float inv;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(gi * gj * s));
    e = -rj * inv;
    u = e * (s + gi) * gj * inv;
    w = u * r2;
}

template <int MT, int NTW, int NW, bool MIX> __device__ static void
euw_body(double *Ew, double *Uw, double *Ww, const double *Q, const float *Qf,
         const int *perm, const double *X, const float *wbox,
         const float *tbox, float rcut2, int npad, int ng, int tbase,
         int nder)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int li = lane >> 2, lj = lane & 3;
    const int gw = blockIdx.x * NW + warp;
    const int ibase = gw * (8 * MT);
    const int t0 = blockIdx.y * (8 * NTW);

    double xi[MT], yi[MT], zi[MT], oi[MT], ki[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int i = ibase + 8 * m + li;
        xi[m] = Q[i]; yi[m] = Q[npad + i]; zi[m] = Q[2 * npad + i];
        oi[m] = Q[3 * npad + i]; ki[m] = Q[4 * npad + i];
    }
    float wb[6];
    if (MIX) {
#pragma unroll
        for (int d = 0; d < 6; ++d) wb[d] = wbox[6 * gw + d];
    }
    double ce[MT][NTW][2], cu[MT][NTW][2], cw[MT][NTW][2];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < NTW; ++n)
#pragma unroll
            for (int h = 0; h < 2; ++h)
                ce[m][n][h] = cu[m][n][h] = cw[m][n][h] = 0.;

    __shared__ double sq[6][JT];
    __shared__ float sqf[6][JT];
    __shared__ double sx[8 * NTW][XS];

    for (int j0 = 0; j0 < npad; j0 += JT) {
        __syncthreads();
        for (int e = threadIdx.x; e < 6 * JT; e += NW * 32) {
            const size_t g = (size_t)(e / JT) * npad + j0 + e % JT;
            sq[e / JT][e % JT] = Q[g];
            if (MIX) sqf[e / JT][e % JT] = Qf[g];
        }
        for (int e = threadIdx.x; e < 8 * NTW * JT; e += NW * 32) {
            const int t = e / JT, j = e % JT;
            sx[t][j] = X[(size_t)(t0 + t) * npad + j0 + j];
        }
        __syncthreads();
        bool far = false;
        if (MIX) {
            const float *tb = tbox + 6 * (j0 / JT);
            const float d0 = fmaxf(0.f, fmaxf(tb[0] - wb[3], wb[0] - tb[3]));
            const float d1 = fmaxf(0.f, fmaxf(tb[1] - wb[4], wb[1] - tb[4]));
            const float d2 = fmaxf(0.f, fmaxf(tb[2] - wb[5], wb[2] - tb[5]));
            far = d0 * d0 + d1 * d1 + d2 * d2 > rcut2;
        }
#pragma unroll 2
        for (int kk = 0; kk < JT; kk += 4) {
            const int j = kk + lj;
            double bx[NTW];
#pragma unroll
            for (int n = 0; n < NTW; ++n) bx[n] = sx[8 * n + li][j];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                double e, u, w;
                if (MIX && far) {
                    float ef, uf, wf;
                    euw_pair_f(ef, uf, wf, (float)xi[m], (float)yi[m],
                               (float)zi[m], (float)oi[m], (float)ki[m],
                               sqf[0][j], sqf[1][j], sqf[2][j], sqf[3][j],
                               sqf[4][j], sqf[5][j]);
                    e = ef; u = uf; w = wf;
                } else {
                    euw_pair(e, u, w, xi[m], yi[m], zi[m], oi[m], ki[m],
                             sq[0][j], sq[1][j], sq[2][j], sq[3][j], sq[4][j],
                             sq[5][j]);
                }
#pragma unroll
                for (int n = 0; n < NTW; ++n) {
                    dmma(ce[m][n][0], ce[m][n][1], e, bx[n]);
                    dmma(cu[m][n][0], cu[m][n][1], u, bx[n]);
                    dmma(cw[m][n][0], cw[m][n][1], w, bx[n]);
                }
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int pi = perm[ibase + 8 * m + li];
        if (pi < 0) continue;
#pragma unroll
        for (int n = 0; n < NTW; ++n)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int t = tbase + t0 + 8 * n + 2 * lj + h;
                if (t < nder) {
                    Ew[(size_t)t * ng + pi] = 1.5 * ce[m][n][h];
                    Uw[(size_t)t * ng + pi] = -1.5 * cu[m][n][h];
                    Ww[(size_t)t * ng + pi] = -1.5 * cw[m][n][h];
                }
            }
    }
}

#define EUW_ENTRY(MT, NTW, NW, MIX, NAME)                                      \
extern "C" __global__ void __launch_bounds__(NW * 32)                          \
NAME##_##MT##_##NTW##_##NW(double *Ew, double *Uw, double *Ww,                 \
        const double *Q, const float *Qf, const int *perm, const double *X,    \
        const float *wbox, const float *tbox, float rcut2, int npad, int ng,   \
        int tbase, int nder)                                                   \
{ euw_body<MT, NTW, NW, MIX>(Ew, Uw, Ww, Q, Qf, perm, X, wbox, tbox, rcut2,    \
                             npad, ng, tbase, nder); }
#define EUW_BOTH(MT, NTW, NW) EUW_ENTRY(MT, NTW, NW, false, euw_dmma)          \
                              EUW_ENTRY(MT, NTW, NW, true, euw_mixed)
EUW_BOTH(2, 1, 4)
EUW_BOTH(1, 2, 8)
EUW_BOTH(1, 4, 8)
EUW_BOTH(1, 8, 8)

// The diagonal (i == j) term, GPU4PySCF's `vv10_hess_eval_f_t_diagonal_kernel`
// verbatim: it adds to what the off-diagonal kernel wrote.
extern "C" __global__ void
ft_diagonal(double *f_rho_t, double *f_gamma_t, const double *rho,
            const double *U, const double *W, const double *A,
            const double *B, const double *C, const double *domega_drho,
            const double *domega_dgamma, const double *dkappa_drho,
            const double *d2omega_drho2, const double *d2omega_dgamma2,
            const double *d2omega_drho_dgamma, const double *d2kappa_drho2,
            const double *rho_t, const double *gamma_t, int ngrids, int ntrial)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ngrids)
        return;
    const double rho_i = rho[i];
    const double domega_drho_i = domega_drho[i];
    const double domega_dgamma_i = domega_dgamma[i];
    const double dkappa_drho_i = dkappa_drho[i];
    const double U_i = U[i], W_i = W[i], A_i = A[i], B_i = B[i], C_i = C[i];
    const double d2omega_drho2_i = d2omega_drho2[i];
    const double d2omega_dgamma2_i = d2omega_dgamma2[i];
    const double d2omega_drho_dgamma_i = d2omega_drho_dgamma[i];
    const double d2kappa_drho2_i = d2kappa_drho2[i];

    const double f_rho_rho_ii = 2 * domega_drho_i * W_i + 2 * dkappa_drho_i * U_i
        + rho_i * (d2omega_drho2_i * W_i + d2kappa_drho2_i * U_i
                   + dkappa_drho_i * dkappa_drho_i * A_i
                   + domega_drho_i * domega_drho_i * C_i
                   + 2 * domega_drho_i * dkappa_drho_i * B_i);
    const double f_gamma_rho_ii = domega_dgamma_i * W_i
        + rho_i * (d2omega_drho_dgamma_i * W_i
                   + domega_dgamma_i * (dkappa_drho_i * B_i + domega_drho_i * C_i));
    const double f_rho_gamma_ii = f_gamma_rho_ii;
    const double f_gamma_gamma_ii = rho_i * (d2omega_dgamma2_i * W_i
                                             + domega_dgamma_i * domega_dgamma_i * C_i);
    for (int t = 0; t < ntrial; t++) {
        const double rt = rho_t[(size_t)t * ngrids + i];
        const double gt = gamma_t[(size_t)t * ngrids + i];
        f_rho_t[(size_t)t * ngrids + i] += f_rho_rho_ii * rt + f_rho_gamma_ii * gt;
        f_gamma_t[(size_t)t * ngrids + i] += f_gamma_rho_ii * rt + f_gamma_gamma_ii * gt;
    }
}
