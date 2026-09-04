// VV10 non-local-correlation double sum.
//
// GPU4PySCF's kernel evaluates, for every pair of grid points (i outer,
// j inner),
//
//     g  = R2*W0_i  + K_i        gp = R2*W0p_j + Kp_j       gt = g + gp
//     T  = RpW_j/(g*gp*gt)       F += T
//     T *= 1/g + 1/gt            U += T                     W += T*R2
//
// i.e. three IEEE double divisions per pair against about ten other
// arithmetic operations.  On an A100 a `div.rn.f64` expands to roughly fifteen
// FMA-equivalents, so the divisions are three quarters of the arithmetic and
// the whole double loop sits at the fp64 peak.  Two changes remove them:
//
//   * 1/g and 1/gt are recovered from the single reciprocal of the product,
//         inv = 1/(g*gp*gt)  ->  1/g = gp*gt*inv,  1/gt = g*gp*inv
//     so  T*(1/g + 1/gt) = RpW * gp*(g+gt) * inv*inv,
//     which is one reciprocal and two extra multiplies instead of three
//     divisions;
//   * that reciprocal is a single-precision seed plus two Newton-Raphson
//     steps in double, about five operations against fifteen.  g >= K > 0.34
//     for every point the caller keeps (K = Kvv*rho^(1/6) and rho >= 1e-10),
//     so the product cannot underflow the float seed, and where it overflows
//     it the true reciprocal is below 3e-39 and dropping it is beyond
//     round-off.
//
// Each thread keeps NGO outer points in registers so that one pass over the
// shared inner tile does NGO pairs, which is what keeps the arithmetic units
// fed once the divisions are gone.
//
// Padding: the caller pads the outer set to a multiple of NT*NGO and the
// inner set to a multiple of NT, with RpW = 0 and W0p = Kp = 1 on the padding,
// so there are no bounds tests in the inner loop.

#define NT 128

// Reciprocal.  RCP is the compile-time choice of seed:
//   0  IEEE div.rn.f64                       (the reference kernel's cost)
//   1  MUFU.RCP64H seed + 2 Newton steps     (rcp.approx.ftz.f64)
//   2  MUFU.RCP (f32) seed + 2 Newton steps
// A `1.0f/(float)x` seed is *not* one of them: without fast math that is a
// full IEEE fp32 division, about as expensive as the fp64 one it replaces.
// Two Newton steps take a 2^-23 seed to 2^-52, i.e. to fp64 round-off.
//
// g >= K = Kvv*rho^(1/6) > 0.34 for every point the caller keeps (rho >= 1e-10
// and Kvv = 16.2), so the product g*gp*gt is at least 0.08 and the seed cannot
// underflow; where it overflows, the true reciprocal is below 3e-39 and
// dropping it is far below round-off.
#ifndef RCP
#define RCP 1
#endif

__device__ __forceinline__ double drcp(double x)
{
#if RCP == 0
    return 1.0 / x;
#else
    double r;
#if RCP == 1
    asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(r) : "d"(x));
#else
    float rf;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rf) : "f"((float)x));
    r = (double)rf;
#endif
    r = r * (2.0 - x * r);
    r = r * (2.0 - x * r);
    return r;
#endif
}

template <int NGO> __device__ static void
vv10_body(double *Fvec, double *Uvec, double *Wvec,
          const double *vvcoords, const double *coords,
          const double *W0p, const double *W0, const double *K,
          const double *Kp, const double *RpW, int nin, int nout)
{
    const int tid = threadIdx.x;
    const int base = blockIdx.x * (NT * NGO) + tid;

    double xi[NGO], yi[NGO], zi[NGO], W0i[NGO], Ki[NGO];
    double Fa[NGO], Ua[NGO], Wa[NGO];
#pragma unroll
    for (int s = 0; s < NGO; ++s) {
        int i = base + s * NT;
        xi[s] = coords[i];
        yi[s] = coords[nout + i];
        zi[s] = coords[2 * nout + i];
        W0i[s] = W0[i];
        Ki[s] = K[i];
        Fa[s] = 0.; Ua[s] = 0.; Wa[s] = 0.;
    }

    __shared__ double sx[NT], sy[NT], sz[NT], sw0[NT], sk[NT], srw[NT];

    for (int j0 = 0; j0 < nin; j0 += NT) {
        int j = j0 + tid;
        sx[tid] = vvcoords[j];
        sy[tid] = vvcoords[nin + j];
        sz[tid] = vvcoords[2 * nin + j];
        sw0[tid] = W0p[j];
        sk[tid] = Kp[j];
        srw[tid] = RpW[j];
        __syncthreads();

#pragma unroll 4
        for (int l = 0; l < NT; ++l) {
            const double xj = sx[l], yj = sy[l], zj = sz[l];
            const double w0j = sw0[l], kj = sk[l], rwj = srw[l];
#pragma unroll
            for (int s = 0; s < NGO; ++s) {
                double dx = xj - xi[s];
                double dy = yj - yi[s];
                double dz = zj - zi[s];
                double r2 = dx * dx + dy * dy + dz * dz;
                double gp = r2 * w0j + kj;
                double g = r2 * W0i[s] + Ki[s];
                double gt = g + gp;
                double inv = drcp(g * gp * gt);
                double T = rwj * inv;
                Fa[s] += T;
                double Tu = T * (gp * (g + gt) * inv);
                Ua[s] += Tu;
                Wa[s] += Tu * r2;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int s = 0; s < NGO; ++s) {
        int i = base + s * NT;
        Fvec[i] = Fa[s];
        Uvec[i] = Ua[s];
        Wvec[i] = Wa[s];
    }
}

#define VV10_ENTRY(n)                                                          \
extern "C" __global__ void vv10_f64_##n(                                       \
    double *Fvec, double *Uvec, double *Wvec, const double *vvcoords,          \
    const double *coords, const double *W0p, const double *W0,                 \
    const double *K, const double *Kp, const double *RpW, int nin, int nout)   \
{ vv10_body<n>(Fvec, Uvec, Wvec, vvcoords, coords, W0p, W0, K, Kp, RpW,        \
               nin, nout); }

VV10_ENTRY(1)
VV10_ENTRY(2)
VV10_ENTRY(4)
VV10_ENTRY(8)

// ---------------------------------------------------------------------------
// Mixed-precision variant.
//
// The pair kernel decays as R^-6, but not fast enough for a cutoff: for these
// clusters dropping everything past 25 Bohr moves E_nlc by ~5 mEh.  What the
// decay does buy is *precision*: a pair at 6 Bohr contributes ~1e-4 of what a
// pair at 0.5 Bohr does, so evaluating the distant ones in fp32 costs
// 1e-7 x 1e-4 of the answer while running at twice the fp64 rate on an A100
// (19.5 against 9.7 TFLOP/s) and turning the fp64 reciprocal into one MUFU.RCP.
//
// The near/far split is made per (outer block, inner tile) pair from their
// bounding boxes, so it is uniform across the block -- no divergence, and the
// fp64 path is untouched where it is used.  For that to screen anything the
// points have to be spatially sorted, which the caller does (Morton order);
// the sum is then over the same terms in a different order.
//
// Coordinates on the fp32 path are stored relative to the outer block's box
// centre.  The far points are up to a molecular diameter away, so their fp32
// positions carry ~3e-6 Bohr of noise; at 6 Bohr and beyond that is 1e-6 of
// R^2 and 6e-6 of one pair term, again against terms that are 1e-4 of the sum.
// FASTNLC_RCUT sets the split and 0 / inf recover the all-fp32 and all-fp64
// kernels, so the cost of the choice is measurable rather than assumed.

// One Newton step after `rcp.approx.ftz.f32` takes the fp32 reciprocal from
// 2^-23 to 2^-46, i.e. below fp32 round-off; NRF=0 drops it, which is two
// operations of the twenty in the far-field pair and leaves an error the same
// size as the fp32 arithmetic around it.
#ifndef NRF
#define NRF 0
#endif

__device__ __forceinline__ float frcp(float x)
{
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
#if NRF
    r = r * (2.f - x * r);
#endif
    return r;
}

template <int NGO> __device__ static void
vv10_mixed_body(double *Fvec, double *Uvec, double *Wvec,
                const double *vvcoords, const double *coords,
                const double *W0p, const double *W0, const double *K,
                const double *Kp, const double *RpW,
                const float *tbox, const float *obox,
                int nin, int nout, float rcut2)
{
    const int tid = threadIdx.x;
    const int base = blockIdx.x * (NT * NGO) + tid;

    // this block's outer bounding box, and the origin of the fp32 frame
    const float *ob = obox + 6 * blockIdx.x;
    const float olo0 = ob[0], olo1 = ob[1], olo2 = ob[2];
    const float ohi0 = ob[3], ohi1 = ob[4], ohi2 = ob[5];
    const double c0 = 0.5 * ((double)olo0 + (double)ohi0);
    const double c1 = 0.5 * ((double)olo1 + (double)ohi1);
    const double c2 = 0.5 * ((double)olo2 + (double)ohi2);

    double xi[NGO], yi[NGO], zi[NGO], W0i[NGO], Ki[NGO];
    float xif[NGO], yif[NGO], zif[NGO], W0if[NGO], Kif[NGO];
    double Fa[NGO], Ua[NGO], Wa[NGO];
#pragma unroll
    for (int s = 0; s < NGO; ++s) {
        int i = base + s * NT;
        xi[s] = coords[i]; yi[s] = coords[nout + i]; zi[s] = coords[2*nout + i];
        W0i[s] = W0[i]; Ki[s] = K[i];
        xif[s] = (float)(xi[s] - c0);
        yif[s] = (float)(yi[s] - c1);
        zif[s] = (float)(zi[s] - c2);
        W0if[s] = (float)W0i[s]; Kif[s] = (float)Ki[s];
        Fa[s] = 0.; Ua[s] = 0.; Wa[s] = 0.;
    }

    // one buffer, read as six doubles per point on the near path and as a
    // float4 (position, RpW) plus a float2 (W0p, Kp) on the far one, so a far
    // pair costs two shared loads instead of six
    __align__(16) __shared__ double sd[6 * NT];
    float4 *sa = (float4 *)sd;
    float2 *sb = (float2 *)(sd + 2 * NT);

    const int ntile = nin / NT;
    for (int t = 0; t < ntile; ++t) {
        const float *tb = tbox + 6 * t;
        float d0 = fmaxf(0.f, fmaxf(tb[0] - ohi0, olo0 - tb[3]));
        float d1 = fmaxf(0.f, fmaxf(tb[1] - ohi1, olo1 - tb[4]));
        float d2 = fmaxf(0.f, fmaxf(tb[2] - ohi2, olo2 - tb[5]));
        bool far = (d0*d0 + d1*d1 + d2*d2) > rcut2;
        const int j = t * NT + tid;

        __syncthreads();
        if (far) {
            sa[tid] = make_float4((float)(vvcoords[j] - c0),
                                  (float)(vvcoords[nin + j] - c1),
                                  (float)(vvcoords[2*nin + j] - c2),
                                  (float)RpW[j]);
            sb[tid] = make_float2((float)W0p[j], (float)Kp[j]);
        } else {
            sd[0*NT+tid] = vvcoords[j];
            sd[1*NT+tid] = vvcoords[nin + j];
            sd[2*NT+tid] = vvcoords[2*nin + j];
            sd[3*NT+tid] = W0p[j];
            sd[4*NT+tid] = Kp[j];
            sd[5*NT+tid] = RpW[j];
        }
        __syncthreads();

        if (far) {
            float Ff[NGO], Uf[NGO], Wf[NGO];
#pragma unroll
            for (int s = 0; s < NGO; ++s) { Ff[s]=0.f; Uf[s]=0.f; Wf[s]=0.f; }
#pragma unroll 4
            for (int l = 0; l < NT; ++l) {
                const float4 aj = sa[l];
                const float2 bj = sb[l];
#pragma unroll
                for (int s = 0; s < NGO; ++s) {
                    float dx = aj.x - xif[s], dy = aj.y - yif[s], dz = aj.z - zif[s];
                    float r2 = dx*dx + dy*dy + dz*dz;
                    float gp = fmaf(r2, bj.x, bj.y);
                    float g = fmaf(r2, W0if[s], Kif[s]);
                    float gt = g + gp;
                    float inv = frcp(g * gp * gt);
                    Ff[s] = fmaf(aj.w, inv, Ff[s]);
                    float Tu = ((aj.w * gp) * (inv * inv)) * (g + gt);
                    Uf[s] += Tu;
                    Wf[s] = fmaf(Tu, r2, Wf[s]);
                }
            }
#pragma unroll
            for (int s = 0; s < NGO; ++s) {
                Fa[s] += (double)Ff[s]; Ua[s] += (double)Uf[s];
                Wa[s] += (double)Wf[s];
            }
        } else {
#pragma unroll 4
            for (int l = 0; l < NT; ++l) {
                const double xj = sd[0*NT+l], yj = sd[1*NT+l], zj = sd[2*NT+l];
                const double w0j = sd[3*NT+l], kj = sd[4*NT+l], rwj = sd[5*NT+l];
#pragma unroll
                for (int s = 0; s < NGO; ++s) {
                    double dx = xj - xi[s], dy = yj - yi[s], dz = zj - zi[s];
                    double r2 = dx*dx + dy*dy + dz*dz;
                    double gp = r2 * w0j + kj;
                    double g = r2 * W0i[s] + Ki[s];
                    double gt = g + gp;
                    double inv = drcp(g * gp * gt);
                    double T = rwj * inv;
                    Fa[s] += T;
                    double Tu = T * (gp * (g + gt) * inv);
                    Ua[s] += Tu;
                    Wa[s] += Tu * r2;
                }
            }
        }
    }

#pragma unroll
    for (int s = 0; s < NGO; ++s) {
        int i = base + s * NT;
        Fvec[i] = Fa[s]; Uvec[i] = Ua[s]; Wvec[i] = Wa[s];
    }
}

#define VV10_MIXED(n)                                                          \
extern "C" __global__ void vv10_mixed_##n(                                     \
    double *Fvec, double *Uvec, double *Wvec, const double *vvcoords,          \
    const double *coords, const double *W0p, const double *W0,                 \
    const double *K, const double *Kp, const double *RpW,                      \
    const float *tbox, const float *obox, int nin, int nout, float rcut2)      \
{ vv10_mixed_body<n>(Fvec, Uvec, Wvec, vvcoords, coords, W0p, W0, K, Kp, RpW,  \
                     tbox, obox, nin, nout, rcut2); }

VV10_MIXED(1)
VV10_MIXED(2)
VV10_MIXED(4)
VV10_MIXED(8)

// ---------------------------------------------------------------------------
// Not pursued: a symmetric kernel.
//
// D_ij = g_i*g_j*(g_i+g_j) is symmetric once the outer and inner point sets
// coincide -- which they do, nr_nlc_vxc passes the same grid and density
// twice -- so one reciprocal can serve both directions of a pair.  Both
// directions cost 27 operations against 20 for one, so visiting each unordered
// pair once is a 1.5x cut in arithmetic.  It was written and abandoned: the i
// side accumulates in registers, but the j side belongs to a tile another
// block owns, so it has to go through either
//
//   * a shared-memory read-modify-write with a rotating index.  Lanes then hit
//     distinct slots, but the ordering that makes that safe is warp lockstep,
//     which the compiler is free to break when it reorders an unrolled loop's
//     loads and stores -- and even before adding the synchronisation that
//     would fix it, the extra six shared accesses per warp step measured
//     slower than the asymmetric kernel (401 ms against 272 ms on a
//     0.18e12-pair subset of PfPMT); or
//   * a warp-shuffle rotation of the outer point and its three accumulators,
//     which is nine shuffles per step, i.e. 1.1 to 2.3 extra instructions per
//     pair against the 6.5 the halved arithmetic saves.
//
// Either way most of the 1.5x goes back, which is why the shipped kernel
// visits every ordered pair.
