/*
 * GPU4PySCF's gvhf-rys/ejk_int3c2e_ip2.cu (Apache-2.0, Copyright 2025 The
 * PySCF Developers), with its contraction and its atomics rewritten
 * (docs/DF_WORK_LIST.md W31, docs/README_fastdfhessip4.md).  The entry point,
 * its arguments and the shared-memory layout are upstream's, so
 * electrolites/_ip2launch.py launches either library the same way.
 *
 * What changes:
 *
 * 1. Of the 51 sums upstream accumulates per Cartesian component, 30 are
 *    linear combinations of the other 21 (translational invariance: the
 *    derivative on the auxiliary centre is minus the two on the pair).  A
 *    triple accumulates the pair--pair block v1 (9), the i--i block (6) and
 *    the j--j block (6); the k--k, i--k and j--k blocks are formed from them
 *    once per triple (k_term).
 * 2. The light classes -- g at most IP2_REG_GMAX doubles per direction and
 *    at most IP2_REG_NFMAX components, upstream's gout_stride 1 -- run a
 *    whole triple in one thread with g in registers (triple_reg): the
 *    recurrences and the contraction compiled per (li, lj, lk), every index
 *    a literal, no barrier inside a triple.  They are a kernel of their own
 *    (MODE 1, 256 threads), so their register budget is not the others'.
 * 3. The other classes (MODE 2) keep upstream's shared-memory g with the
 *    21-sum contraction, on 512 threads: two threads per triple where
 *    upstream has one, the same triples per block and the same shared
 *    memory, twice the warps per SM at 128 registers.
 * 4. The atomics: a block's triples park their sums in shared memory and the
 *    block adds them up -- the pair terms over its auxiliary shells, the kk
 *    terms over its pairs, the ik/jk terms over runs of pairs on one atom --
 *    before one atomicAdd per target (block_flush).
 *
 * Measured and rejected (docs/README_fastdfhessip4.md §5): the shared-memory
 * contraction unrolled with literal indices (spills at 255 registers), a
 * branch-free runtime contraction, auxiliary-fastest block order.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "vhf.cuh"
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "gvhf-rys/rys_roots_for_k.cu"
#include "build_rys_gxyz.cuh"

// upstream's block is 256 threads; IP2_THREADS = 256 * m runs m threads on
// each (pair, auxiliary shell) triple where upstream runs one, with the same
// triples per block and the same shared memory (the gout_stride table read
// from the caller is upstream's, multiplied by m here)
#ifndef IP2_SPLIT
#define IP2_SPLIT       1
#endif
#ifndef IP2_THREADS
#define IP2_THREADS     (IP2_SPLIT ? 512 : 256)
#endif
#define THREADS         256
#define BLOCK_SIZE      16

__constant__ int c_nf[] = {1, 3, 6, 10, 15, 21, 28, 36, 45};
__constant__ float c_div_nf[] = {1.f, 0.333334f, 0.166667f, 0.100001f,
    0.066667f, 0.047620f, 0.035715f, 0.027778f, 0.022223f};

#ifndef IP2_DMV_MAX
#define IP2_DMV_MAX 32
#endif
#ifndef IP2_REDUCE
#define IP2_REDUCE 1
#endif

// The 21 sums of one triple, in this order:
//   0-8   v1[d][e]  (d on i, e on j)  xx xy xz yx yy yz zx zy zz
//   9-14  vi        xx xy xz yy yz zz
//   15-20 vj        xx xy xz yy yz zz
#define NSUM 21

__device__ __forceinline__
void accumulate(double *s, double Ix, double Iy, double Iz, double dm,
                double fix, double fiy, double fiz,
                double fjx, double fjy, double fjz,
                double fijx, double fijy, double fijz,
                double fiix, double fiiy, double fiiz,
                double fjjx, double fjjy, double fjjz)
{
    double Qx = Ix * dm;
    double Qy = Iy * dm;
    double Qz = Iz * dm;
    double Px = Iy * Qz;
    double Py = Ix * Qz;
    double Pz = Ix * Qy;
    s[0] += fijx * Px;
    s[4] += fijy * Py;
    s[8] += fijz * Pz;
    s[9] += fiix * Px;
    s[12] += fiiy * Py;
    s[14] += fiiz * Pz;
    s[15] += fjjx * Px;
    s[18] += fjjy * Py;
    s[20] += fjjz * Pz;
    double a, b;
    // x, y with Qz
    a = fix * Qz; b = fjx * Qz;
    s[1] += a * fjy;   // v1xy
    s[10] += a * fiy;  // vixy
    s[3] += b * fiy;   // v1yx
    s[16] += b * fjy;  // vjxy
    // x, z with Qy
    a = fix * Qy; b = fjx * Qy;
    s[2] += a * fjz;   // v1xz
    s[11] += a * fiz;  // vixz
    s[6] += b * fiz;   // v1zx
    s[17] += b * fjz;  // vjxz
    // y, z with Qx
    a = fiy * Qx; b = fjy * Qx;
    s[5] += a * fjz;   // v1yz
    s[13] += a * fiz;  // viyz
    s[7] += b * fiz;   // v1zy
    s[19] += b * fjz;  // vjyz
}

// One Cartesian direction of one component: the value and the five
// derivative factors upstream forms inline, from g at (a, b, c) with the
// i and j neighbours at +-i_1, +-j_1.
template <int A, int B>
__device__ __forceinline__
void factors_ct(const double *g, int i_1, int j_1, double ai2, double aj2,
                double rjri, double &I, double &fi, double &fj, double &fij,
                double &fii, double &fjj)
{
    I = g[0];
    double gi = g[i_1];
    double gj = g[j_1];
    double gij = g[i_1+j_1];
    fi = ai2 * gi;
    if (A > 0) fi -= A * g[-i_1];
    fj = aj2 * gj;
    if (B > 0) fj -= B * g[-j_1];
    double f3 = ai2 * gij;
    if (A > 0) f3 -= A * g[j_1-i_1];
    f3 *= aj2;
    if (B > 0) {
        double f = ai2 * g[i_1-j_1];
        if (A > 0) f -= A * g[-i_1-j_1];
        f3 -= B * f;
    }
    fij = f3;
    double t = aj2 * (aj2 * (gij - gj * rjri) - (2*B+1) * I);
    if (B > 1) t += B*(B-1) * g[-2*j_1];
    fjj = t;
    t = ai2 * (ai2 * (gij + gi * rjri) - (2*A+1) * I);
    if (A > 1) t += A*(A-1) * g[-2*i_1];
    fii = t;
}

__device__ __forceinline__
void factors_rt(const double *g, int a, int b, int i_1, int j_1, double ai2,
                double aj2, double rjri, double &I, double &fi, double &fj,
                double &fij, double &fii, double &fjj)
{
    I = g[0];
    double gi = g[i_1];
    double gj = g[j_1];
    double gij = g[i_1+j_1];
    fi = ai2 * gi;
    if (a > 0) fi -= a * g[-i_1];
    fj = aj2 * gj;
    if (b > 0) fj -= b * g[-j_1];
    double f3 = ai2 * gij;
    if (a > 0) f3 -= a * g[j_1-i_1];
    f3 *= aj2;
    if (b > 0) {
        double f = ai2 * g[i_1-j_1];
        if (a > 0) f -= a * g[-i_1-j_1];
        f3 -= b * f;
    }
    fij = f3;
    double t = aj2 * (aj2 * (gij - gj * rjri) - (2*b+1) * I);
    if (b > 1) t += b*(b-1) * g[-2*j_1];
    fjj = t;
    t = ai2 * (ai2 * (gij + gi * rjri) - (2*a+1) * I);
    if (a > 1) t += a*(a-1) * g[-2*i_1];
    fii = t;
}

// Cartesian exponents of component n of angular momentum l, in PySCF's
// lexical order, at compile time.
__host__ __device__ constexpr int lex_x(int l, int n)
{
    // n-th of (l,0,0), (l-1,1,0), (l-1,0,1), (l-2,2,0), ...
    int lx = l;
    while (n > l - lx) { n -= l - lx + 1; --lx; }
    return lx;
}
__host__ __device__ constexpr int lex_y(int l, int n)
{
    int lx = l;
    while (n > l - lx) { n -= l - lx + 1; --lx; }
    return l - lx - n;
}
__host__ __device__ constexpr int lex_z(int l, int n)
{
    return l - lex_x(l, n) - lex_y(l, n);
}
__host__ __device__ constexpr int nf_of(int l) { return (l+1)*(l+2)/2; }

// the generic contraction: upstream's index decode, runtime branches
__device__ __forceinline__
void contract_rt(double *s, const double *gx, const double *dm_tensor, const double *auxvec, int k0, int nao,
                 int naux, int nksh, int nst, int g_size, int li, int lj,
                 int lk, int nf, int gout_id, int gout_stride, double ai2,
                 double aj2, double xjxi, double yjyi, double zjzi)
{
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);
    int idx_k = lex_xyz_offset(lk);
    int stride_j = li + 2;
    int stride_k = stride_j * (lj + 2);
    int i_1 = nst;
    int j_1 = stride_j * nst;
    int nfi = c_nf[li];
    int nfj = c_nf[lj];
    int nfij = nfi * nfj;
    float div_nfi = c_div_nf[li];
    float div_nfj = c_div_nf[lj];
    float div_nfij = div_nfi * div_nfj;
    for (int n = gout_id; n < nf; n += gout_stride) {
        uint32_t k = n * div_nfij;
        uint32_t ij = n - k * nfij;
        uint32_t j = ij * div_nfi;
        uint32_t i = ij - j * nfi;
        int ix = _c_cartesian_lexical_xyz[idx_i + i*3+0];
        int iy = _c_cartesian_lexical_xyz[idx_i + i*3+1];
        int iz = _c_cartesian_lexical_xyz[idx_i + i*3+2];
        int jx = _c_cartesian_lexical_xyz[idx_j + j*3+0];
        int jy = _c_cartesian_lexical_xyz[idx_j + j*3+1];
        int jz = _c_cartesian_lexical_xyz[idx_j + j*3+2];
        int kx = _c_cartesian_lexical_xyz[idx_k + k*3+0];
        int ky = _c_cartesian_lexical_xyz[idx_k + k*3+1];
        int kz = _c_cartesian_lexical_xyz[idx_k + k*3+2];
        double dm = auxvec == NULL ? dm_tensor[ij*naux + k*nksh]
                                   : dm_tensor[j*nao+i] * auxvec[k0+k];
        double Ix, fix, fjx, fijx, fiix, fjjx;
        double Iy, fiy, fjy, fijy, fiiy, fjjy;
        double Iz, fiz, fjz, fijz, fiiz, fjjz;
        factors_rt(gx + (ix + jx*stride_j + kx*stride_k) * nst, ix, jx,
                   i_1, j_1, ai2, aj2, xjxi, Ix, fix, fjx, fijx, fiix, fjjx);
        factors_rt(gx + (iy + jy*stride_j + ky*stride_k + g_size) * nst, iy,
                   jy, i_1, j_1, ai2, aj2, yjyi, Iy, fiy, fjy, fijy, fiiy,
                   fjjy);
        factors_rt(gx + (iz + jz*stride_j + kz*stride_k + 2*g_size) * nst, iz,
                   jz, i_1, j_1, ai2, aj2, zjzi, Iz, fiz, fjz, fijz, fiiz,
                   fjjz);
        accumulate(s, Ix, Iy, Iz, dm, fix, fiy, fiz, fjx, fjy, fjz,
                   fijx, fijy, fijz, fiix, fiiy, fiiz, fjjx, fjjy, fjjz);
    }
}

__device__ __forceinline__
void contract(double *s, const double *gx, const double *dm_tensor, const double *auxvec, int k0, int nao,
              int naux, int nksh, int nst, int g_size, int li, int lj, int lk,
              int nf, int gout_id, int gout_stride, double ai2, double aj2,
              double xjxi, double yjyi, double zjzi)
{
#ifdef IP2_NOCONTRACT
    return;
#endif
    contract_rt(s, gx, dm_tensor, auxvec, k0, nao, naux, nksh, nst, g_size, li, lj, lk, nf,
                gout_id, gout_stride, ai2, aj2, xjxi, yjyi, zjzi);
}


// ---------------------------------------------------------------------------
// The register path: a light class's whole triple in one thread, with g in
// registers (W19's design, for the classes whose g fits): no shared-memory g,
// no barriers inside the triple, the contraction's indices literals.
// ---------------------------------------------------------------------------
#ifndef IP2_REG
#define IP2_REG 1
#endif

template <int LI, int LJ, int LK, int N>
__device__ __forceinline__
void component_reg(double *s, const double (&g)[3][(LI+2)*(LJ+2)*(LK+1)],
                   double dm, double ai2, double aj2, double xjxi,
                   double yjyi, double zjzi)
{
    constexpr int NFI = nf_of(LI);
    constexpr int NFJ = nf_of(LJ);
    constexpr int NFIJ = NFI * NFJ;
    constexpr int K = N / NFIJ;
    constexpr int IJ = N - K * NFIJ;
    constexpr int J = IJ / NFI;
    constexpr int I = IJ - J * NFI;
    constexpr int IX = lex_x(LI, I), IY = lex_y(LI, I), IZ = lex_z(LI, I);
    constexpr int JX = lex_x(LJ, J), JY = lex_y(LJ, J), JZ = lex_z(LJ, J);
    constexpr int KX = lex_x(LK, K), KY = lex_y(LK, K), KZ = lex_z(LK, K);
    constexpr int SJ = LI + 2;
    constexpr int SK = SJ * (LJ + 2);
    double Ix, fix, fjx, fijx, fiix, fjjx;
    double Iy, fiy, fjy, fijy, fiiy, fjjy;
    double Iz, fiz, fjz, fijz, fiiz, fjjz;
    factors_ct<IX, JX>(&g[0][IX + JX*SJ + KX*SK], 1, SJ, ai2, aj2, xjxi,
                       Ix, fix, fjx, fijx, fiix, fjjx);
    factors_ct<IY, JY>(&g[1][IY + JY*SJ + KY*SK], 1, SJ, ai2, aj2, yjyi,
                       Iy, fiy, fjy, fijy, fiiy, fjjy);
    factors_ct<IZ, JZ>(&g[2][IZ + JZ*SJ + KZ*SK], 1, SJ, ai2, aj2, zjzi,
                       Iz, fiz, fjz, fijz, fiiz, fjjz);
    accumulate(s, Ix, Iy, Iz, dm, fix, fiy, fiz, fjx, fjy, fjz,
               fijx, fijy, fijz, fiix, fiiy, fiiz, fjjx, fjjy, fjjz);
}

template <int LI, int LJ, int LK, int N, int NF, bool PRELOAD>
__device__ __forceinline__
void unroll_reg(double *s, const double (&g)[3][(LI+2)*(LJ+2)*(LK+1)],
                const double *dmv, double ai2, double aj2, double xjxi,
                double yjyi, double zjzi, const double *dm_tensor,
                const double *auxvec, int k0, int nao, int naux, int nksh)
{
    if constexpr (N < NF) {
        double dm;
        if constexpr (PRELOAD) {
            dm = dmv[N];
        } else {
            constexpr int NFI = nf_of(LI);
            constexpr int NFIJ = NFI * nf_of(LJ);
            constexpr int K = N / NFIJ;
            constexpr int IJ = N - K * NFIJ;
            if (auxvec == NULL) {
                dm = __ldg(dm_tensor + IJ*naux + K*nksh);
            } else {
                dm = __ldg(dm_tensor + (IJ/NFI)*nao + IJ%NFI) *
                     __ldg(auxvec + k0 + K);
            }
        }
        component_reg<LI, LJ, LK, N>(s, g, dm, ai2, aj2, xjxi, yjyi, zjzi);
        unroll_reg<LI, LJ, LK, N+1, NF, PRELOAD>(
            s, g, dmv, ai2, aj2, xjxi, yjyi, zjzi, dm_tensor, auxvec, k0,
            nao, naux, nksh);
    }
}

// One triple of class (LI, LJ, LK), omega >= 0.  rw is this thread's column
// of the roots area (stride nst); the result is added to s.
template <int LI, int LJ, int LK>
__device__ __forceinline__
void triple_reg(double *s, const double *env, int iprim, int jprim,
                int kprim, int expi, int expj, int expk, int ci, int cj,
                int ck, int ri, int rj, int rk, bool same_shell,
                bool lower, double *rw, int nst, double omega,
                double lr_factor, double sr_factor,
                const double *dm_tensor, const double *auxvec, int k0,
                int nao, int naux, int nksh)
{
    constexpr int NFI = nf_of(LI);
    constexpr int NFJ = nf_of(LJ);
    constexpr int NF = NFI * NFJ * nf_of(LK);
    constexpr int SJ = LI + 2;
    constexpr int SK = SJ * (LJ + 2);
    constexpr int GS = SK * (LK + 1);
    constexpr int LIJ = LI + LJ + 2;
    constexpr int NROOTS = (LIJ + LK) / 2 + 1;
    static_assert(NROOTS <= 8, "IP2_REG_SHM assumes at most 8 roots");
    // the triple's pseudo-density, in registers when it is small; otherwise
    // read per component (NF > IP2_DMV_MAX)
    constexpr bool PRELOAD = NF <= IP2_DMV_MAX;
    double dmv[PRELOAD ? NF : 1];
#pragma unroll
    for (int n = 0; n < (PRELOAD ? NF : 0); ++n) {
        int k = n / (NFI*NFJ);
        int ij = n - k * (NFI*NFJ);
        if (auxvec == NULL) {
            dmv[n] = __ldg(dm_tensor + ij*naux + k*nksh);
        } else {
            int j = ij / NFI;
            int i = ij - j * NFI;
            dmv[n] = __ldg(dm_tensor + j*nao + i) * __ldg(auxvec + k0 + k);
        }
    }
    double xi = env[ri+0], yi = env[ri+1], zi = env[ri+2];
    double xjxi = env[rj+0] - xi;
    double yjyi = env[rj+1] - yi;
    double zjzi = env[rj+2] - zi;
    double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
    double xk = env[rk+0], yk = env[rk+1], zk = env[rk+2];
    double fac_ij = PI_FAC;
    if (same_shell) fac_ij *= .5;
    else if (lower) fac_ij = 0;
    for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
        int ip = ijp / jprim;
        int jp = ijp - jprim * ip;
        double ai = env[expi+ip];
        double aj = env[expj+jp];
        double aij = ai + aj;
        double aj_aij = aj / aij;
        double theta_ij = ai * aj_aij;
        double cicj = fac_ij * env[ci+ip] * env[cj+jp] * exp(-theta_ij * rr_ij);
        double xpa = xjxi * aj_aij;
        double ypa = yjyi * aj_aij;
        double zpa = zjzi * aj_aij;
        double xpq = xpa + xi - xk;
        double ypq = ypa + yi - yk;
        double zpq = zpa + zi - zk;
        double rr = xpq*xpq + ypq*ypq + zpq*zpq;
        double ai2 = ai * 2;
        double aj2 = aj * 2;
        for (int kp = 0; kp < kprim; ++kp) {
            double ak = env[expk+kp];
            double theta = aij * ak / (aij + ak);
            double fac = env[ck+kp] / (aij*ak*sqrt(aij+ak));
            rys_roots_for_k(NROOTS, theta, rr, rw, omega, lr_factor, sr_factor,
                            nst, 1, 0);
#pragma unroll
            for (int irys = 0; irys < NROOTS; ++irys) {
                double g[3][GS];
                double rt = rw[irys*2*nst];
                g[0][0] = fac;
                g[1][0] = cicj;
                g[2][0] = rw[(irys*2+1)*nst];
                double rt_aa = rt / (aij + ak);
                double rt_aij = rt_aa * ak;
                double b10 = .5/aij * (1 - rt_aij);
                double rt_ak = rt_aa * aij;
                double b00 = .5 * rt_aa;
                double b01 = .5/ak * (1 - rt_ak);
                double pa[3] = {xpa, ypa, zpa};
                double pq[3] = {xpq, ypq, zpq};
                double rjri[3] = {xjxi, yjyi, zjzi};
#pragma unroll
                for (int d = 0; d < 3; ++d) {
                    double *gd = g[d];
                    // i
                    double c0x = pa[d] - rt_aij * pq[d];
                    if (LIJ > 0) {
                        gd[1] = c0x * gd[0];
#pragma unroll
                        for (int i = 1; i < LIJ; ++i) {
                            gd[i+1] = c0x * gd[i] + i * b10 * gd[i-1];
                        }
                    }
                    // k
                    if (LK > 0) {
                        double cpx = rt_ak * pq[d];
#pragma unroll
                        for (int i = 0; i <= LIJ; ++i) {
                            double s0 = gd[i];
                            double s1 = cpx * s0;
                            if (i > 0) s1 += i * b00 * gd[i-1];
                            gd[i+SK] = s1;
#pragma unroll
                            for (int k = 1; k < LK; ++k) {
                                double s2 = cpx * s1 + k * b01 * s0;
                                if (i > 0) s2 += i * b00 * gd[i-1 + k*SK];
                                gd[i + (k+1)*SK] = s2;
                                s0 = s1;
                                s1 = s2;
                            }
                        }
                    }
                    // j, in place from the top as upstream
#pragma unroll
                    for (int k = 0; k <= LK; ++k) {
                        double *gk = gd + k*SK;
#pragma unroll
                        for (int j = 0; j < LJ+1; ++j) {
#pragma unroll
                            for (int ij = (LIJ-j) + j*SJ - 1; ij >= j*SJ; --ij) {
                                gk[ij+SJ] = gk[ij+1] - rjri[d] * gk[ij];
                            }
                        }
                    }
                }
                if constexpr (PRELOAD) {
                    unroll_reg<LI, LJ, LK, 0, NF, true>(
                        s, g, dmv, ai2, aj2, xjxi, yjyi, zjzi,
                        dm_tensor, auxvec, k0, nao, naux, nksh);
                } else {
                    unroll_reg<LI, LJ, LK, 0, NF, false>(
                        s, g, dmv, ai2, aj2, xjxi, yjyi, zjzi,
                        dm_tensor, auxvec, k0, nao, naux, nksh);
                }
            }
        }
    }
}


#ifndef IP2_REG_GMAX
#define IP2_REG_GMAX 24
#endif
#ifndef IP2_REG_NFMAX
#define IP2_REG_NFMAX 36
#endif
// shared memory the register path needs: the block reduction's NSUM sums
// and six header rows, and the roots (NROOTS <= 8 for every light class),
// at nst = 256
#define IP2_REG_SHM ((NSUM + 6 + 2*8) * THREADS * 8)
__host__ __device__ constexpr bool reg_class(int li, int lj, int lk)
{
    return (li+2)*(lj+2)*(lk+1) <= IP2_REG_GMAX &&
        nf_of(li)*nf_of(lj)*nf_of(lk) <= IP2_REG_NFMAX;
}

__device__ __forceinline__
bool reg_supported(int li, int lj, int lk)
{
#define IP2_CASE(a, b, c) if (li == a && lj == b && lk == c) return reg_class(a, b, c);
#include "ip2_cases.inc"
#undef IP2_CASE
    return false;
}

#define REG_ARGS s, env, iprim, jprim, kprim, expi, expj, expk, ci, cj, ck, \
    ri, rj, rk, same_shell, lower, rw, nst, omega, lr_factor, sr_factor, \
    dm_tensor, auxvec, k0, nao, naux, nksh
__device__ __forceinline__
void reg_dispatch(int li, int lj, int lk, double *s, const double *env,
                  int iprim, int jprim, int kprim, int expi, int expj,
                  int expk, int ci, int cj, int ck, int ri, int rj, int rk,
                  bool same_shell, bool lower, double *rw, int nst,
                  double omega, double lr_factor, double sr_factor,
                  const double *dm_tensor, const double *auxvec, int k0,
                  int nao, int naux, int nksh)
{
#define IP2_CASE(a, b, c) \
    if constexpr (reg_class(a, b, c)) { \
        if (li == a && lj == b && lk == c) { \
            triple_reg<a, b, c>(REG_ARGS); return; } }
#include "ip2_cases.inc"
#undef IP2_CASE
}

// upstream's layout of a symmetric 3x3 block on the diagonal of ejk: the
// lower triangle, the diagonal halved (xx xy xz yy yz zz)
__constant__ int c_sym_ejk[6] = {0, 3, 6, 4, 7, 8};
__constant__ double c_sym_fac[6] = {.5, 1., 1., .5, 1., .5};
// (d, e) -> symmetric index
__constant__ int c_sym_of[9] = {0, 1, 2, 1, 3, 4, 2, 4, 5};
__constant__ int c_sym_d[6] = {0, 0, 0, 1, 1, 2};
__constant__ int c_sym_e[6] = {0, 1, 2, 1, 2, 2};

// value of target m of one triple's sums t[] (strided by `stride`):
//   m < 6: kk (symmetric m); 6 <= m < 15: ik (d, f); m >= 15: jk (d, f)
__device__ __forceinline__
double k_term(const double *t, int stride, int m)
{
    if (m < 6) {
        int d = c_sym_d[m], e = c_sym_e[m];
        return t[(9+m)*stride] + t[(d*3+e)*stride] + t[(e*3+d)*stride]
             + t[(15+m)*stride];
    } else if (m < 15) {
        int df = m - 6;
        int d = df / 3, f = df - d * 3;
        return -(t[(d*3+f)*stride] + t[(9+c_sym_of[df])*stride]);
    } else {
        int df = m - 15;
        int d = df / 3, f = df - d * 3;
        return -(t[(f*3+d)*stride] + t[(15+c_sym_of[df])*stride]);
    }
}

// one triple's 45 atomics, as upstream (the fallback when the block
// reduction's buffer does not fit)
__device__ __forceinline__
void triple_atomics(double *ejk, const double *s, int ia, int ja, int ka,
                    int natm)
{
    for (int m = 0; m < 6; ++m) {
        atomicAdd(ejk + (ka*natm+ka)*9 + c_sym_ejk[m],
                  k_term(s, 1, m) * c_sym_fac[m]);
        atomicAdd(ejk + (ia*natm+ia)*9 + c_sym_ejk[m], s[9+m] * c_sym_fac[m]);
        atomicAdd(ejk + (ja*natm+ja)*9 + c_sym_ejk[m], s[15+m] * c_sym_fac[m]);
    }
    for (int m = 0; m < 9; ++m) {
        atomicAdd(ejk + (ia*natm+ka)*9 + m, k_term(s, 1, 6+m));
        atomicAdd(ejk + (ja*natm+ka)*9 + m, k_term(s, 1, 15+m));
        atomicAdd(ejk + (ia*natm+ja)*9 + m, s[m]);
    }
}

// The block's triples' sums, red[n*nst + sp*naux_b + a], added into ejk:
// the pair terms summed over the block's auxiliary shells, the kk terms
// over its pairs, the ik/jk terms over runs of pairs on the same atom.
template <int NT>
__device__ __forceinline__
void block_flush(double *ejk, const double *red, const int *s_ia,
                 const int *s_ja, const int *s_ka, const bool *s_pa,
                 const bool *s_kact, int natm, int nst, int nsp, int naux_b,
                 int thread_id)
{
    int ntask_p = nsp * NSUM;
    int ntask = ntask_p + naux_b * 24;
    for (int t = thread_id; t < ntask; t += NT) {
        if (t < ntask_p) {
            int sp = t / NSUM;
            int n = t - sp * NSUM;
            if (!s_pa[sp]) continue;
            const double *r = red + n * nst + sp * naux_b;
            double v = 0;
            for (int a = 0; a < naux_b; ++a) v += r[a];
            int ia = s_ia[sp];
            int ja = s_ja[sp];
            if (n < 9) {
                atomicAdd(ejk + (ia*natm+ja)*9 + n, v);
            } else if (n < 15) {
                atomicAdd(ejk + (ia*natm+ia)*9 + c_sym_ejk[n-9], v * c_sym_fac[n-9]);
            } else {
                atomicAdd(ejk + (ja*natm+ja)*9 + c_sym_ejk[n-15], v * c_sym_fac[n-15]);
            }
        } else {
            int u = t - ntask_p;
            int a = u / 24;
            int m = u - a * 24;
            if (!s_kact[a]) continue;
            int ka = s_ka[a];
            if (m < 6) {
                double v = 0;
                for (int sp = 0; sp < nsp; ++sp) {
                    v += k_term(red + sp * naux_b + a, nst, m);
                }
                atomicAdd(ejk + (ka*natm+ka)*9 + c_sym_ejk[m], v * c_sym_fac[m]);
            } else {
                const int *s_atm = m < 15 ? s_ia : s_ja;
                int e = m < 15 ? m - 6 : m - 15;
                int key = -1;
                double acc = 0;
                for (int sp = 0; sp < nsp; ++sp) {
                    if (!s_pa[sp]) continue;
                    int at = s_atm[sp];
                    if (at != key) {
                        if (key >= 0) atomicAdd(ejk + (key*natm+ka)*9 + e, acc);
                        key = at;
                        acc = 0;
                    }
                    acc += k_term(red + sp * naux_b + a, nst, m);
                }
                if (key >= 0) atomicAdd(ejk + (key*natm+ka)*9 + e, acc);
            }
        }
    }
}

// MODE 0: every class, both paths; MODE 1: the register path's classes only;
// MODE 2: the others only.  A block of the other kind returns at once.
template <int NT, int MODE>
__global__ static __launch_bounds__(NT, 1)
void ejk_int3c2e_ip2_kernel(double *ejk, double *dm, double *density_auxvec,
                            RysIntEnvVars envs, double omega, double lr_factor,
                            double sr_factor, int *shl_pair_offsets,
                            uint32_t *bas_ij_idx, int *ksh_offsets, int *gout_stride_lookup,
                            int *ao_pair_loc, int aux_offset, int naux,
                            int shm_bytes)
{
    // For better load balance, consume blocks in the reversed order
    int thread_id = threadIdx.x;
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int ksh0, ksh1, nksh;
    __shared__ int li, lj, lk, nroots, nf;
    __shared__ int iprim, jprim, kprim;
    __shared__ int g_size;
    __shared__ int nao;
    __shared__ int gout_stride, nst_per_block, aux_per_block, nsp_per_block;
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        uint32_t bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 - nbas * ish0;
        ksh0 = ksh_offsets[ksh_block_id];
        ksh1 = ksh_offsets[ksh_block_id+1];
        nksh = ksh1 - ksh0;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        lk = bas[ksh0*BAS_SLOTS+ANG_OF];
        int lij = li + lj + 2;
        nroots = (lij + lk) / 2 + 1;
        if (omega < 0) {
            nroots *= 2;
        }
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
        nao = envs.ao_loc[nbas];
        int nfi = c_nf[li];
        int nfj = c_nf[lj];
        int nfk = c_nf[lk];
        int nfij = nfi * nfj;
        nf = nfij * nfk;
        int stride_j = li + 2;
        int stride_k = stride_j * (lj + 2);
        g_size = stride_k * (lk + 1);
        gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj] *
            (NT / THREADS);
        nst_per_block = NT / gout_stride;
        aux_per_block = min(nst_per_block, BLOCK_SIZE);
        nsp_per_block = nst_per_block / aux_per_block;
    }
    __syncthreads();
    register int gout_id = thread_id / nst_per_block;
    register int st_id = thread_id - gout_id * nst_per_block;
    register int sp_id = st_id / aux_per_block;
    register int aux_id = st_id - sp_id * aux_per_block;

    int gx_len = g_size * nst_per_block;
    extern __shared__ double shared_memory[];
    double *rjri = shared_memory + sp_id;
    double *Rpq = shared_memory + nsp_per_block * 3 + st_id;
    double *gx = shared_memory + nst_per_block * 6 + st_id;
    double *rw = shared_memory + nst_per_block * (g_size*3+6) + st_id;
    // the block reduction's buffer: NSUM sums per triple, in the g/rw area
    double *red = shared_memory + nst_per_block * 6;
    __shared__ int s_ia[BLOCK_SIZE], s_ja[BLOCK_SIZE], s_ka[BLOCK_SIZE];
    __shared__ bool s_pa[BLOCK_SIZE], s_kact[BLOCK_SIZE];
    int natm = envs.natm;
    int nst = nst_per_block;
    bool light = IP2_REG && omega >= 0 && gout_stride * THREADS == NT &&
        reg_supported(li, lj, lk);
    if (MODE == 1 && !light) return;
    if (MODE == 2 && light) return;
    bool use_reg = MODE != 2 && light;
    // the register path keeps its roots after the reduction buffer
    double *rw_reg = shared_memory + nst_per_block * (6 + NSUM) + st_id;
    bool block_reduce = IP2_REDUCE &&
        (NSUM + 6) * nst * (int)sizeof(double) <= shm_bytes;


    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        int i0 = envs.ao_loc[ish];
        int j0 = envs.ao_loc[jsh];
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = env[rj+0] - env[ri+0];
        double yjyi = env[rj+1] - env[ri+1];
        double zjzi = env[rj+2] - env[ri+2];
        __syncthreads();
        if (gout_id == 0 && aux_id == 0) {
            rjri[0*nsp_per_block] = xjxi;
            rjri[1*nsp_per_block] = yjyi;
            rjri[2*nsp_per_block] = zjzi;
        }
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += aux_per_block) {
            int ksh = kidx;
            if (kidx >= ksh1) {
                ksh = ksh0;
            }
            int k0;
            double *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }

            double s[NSUM];
#pragma unroll
            for (int n = 0; n < NSUM; ++n) s[n] = 0;

            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];

            if (MODE != 2 && use_reg) {
                if (pair_ij < shl_pair1 && kidx < ksh1) {
                    reg_dispatch(li, lj, lk, s, env, iprim, jprim, kprim,
                                 expi, expj, expk, ci, cj, ck, ri, rj, rk,
                                 ish == jsh, ish < jsh, rw_reg, nst, omega,
                                 lr_factor, sr_factor, dm_tensor,
                                 density_auxvec, k0, nao, naux, nksh);
                }
            } else if (MODE != 1)
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                __syncthreads();
                if (gout_id == 0) {
                    double theta_ij = ai * aj_aij;
                    double xjxi = rjri[0*nsp_per_block];
                    double yjyi = rjri[1*nsp_per_block];
                    double zjzi = rjri[2*nsp_per_block];
                    double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                    double Kab = theta_ij * rr_ij;
                    double fac_ij = PI_FAC;
                    if (ish == jsh) {
                        fac_ij *= .5;
                    } else if (ish < jsh) {
                        fac_ij = 0;
                    }
                    double cicj = fac_ij * env[ci+ip] * env[cj+jp];
                    gx[gx_len] = cicj * exp(-Kab);
                    double xij = xjxi * aj_aij + env[ri+0];
                    double yij = yjyi * aj_aij + env[ri+1];
                    double zij = zjzi * aj_aij + env[ri+2];
                    double xk = env[rk+0];
                    double yk = env[rk+1];
                    double zk = env[rk+2];
                    double xpq = xij - xk;
                    double ypq = yij - yk;
                    double zpq = zij - zk;
                    Rpq[0*nst_per_block] = xpq;
                    Rpq[1*nst_per_block] = ypq;
                    Rpq[2*nst_per_block] = zpq;
                }
                for (int kp = 0; kp < kprim; ++kp) {
                    double ak = env[expk+kp];
                    double theta = aij * ak / (aij + ak);
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[0] = env[ck+kp] / (aij*ak*sqrt(aij+ak));
                    }
                    double xpq = Rpq[0*nst_per_block];
                    double ypq = Rpq[1*nst_per_block];
                    double zpq = Rpq[2*nst_per_block];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    rys_roots_for_k(nroots, theta, rr, rw, omega, lr_factor, sr_factor,
                                    nst_per_block, gout_stride, gout_id);
                    for (int irys = 0; irys < nroots; ++irys) {
                        int lij = li + lj + 2;
                        int stride_j = li + 2;
                        int stride_k = stride_j * (lj + 2);
                        BUILD_3C_GXYZ(lj+1, lk, nsp_per_block, pair_ij < shl_pair1 && kidx < ksh1);
                        if (pair_ij < shl_pair1 && kidx < ksh1) {
                            contract(s, gx, dm_tensor, density_auxvec, k0, nao, naux, nksh, nst,
                                     g_size, li, lj, lk, nf, gout_id,
                                     gout_stride, ai * 2, aj * 2,
                                     rjri[0], rjri[nsp_per_block],
                                     rjri[2*nsp_per_block]);
                        }
                    }
                }
            }
            bool active = pair_ij < shl_pair1 && kidx < ksh1;
            int ia = bas[ish*BAS_SLOTS+ATOM_OF];
            int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
            int ka = bas[ksh*BAS_SLOTS+ATOM_OF] - natm;
#ifdef IP2_NOATOMIC
            if (active) { double t = 0;
                for (int n = 0; n < NSUM; ++n) t += s[n];
                if (t == -1.2345e300) ejk[0] = t; }
#else
            if (block_reduce) {
                __syncthreads();
                if (gout_id == 0) {
                    if (aux_id == 0) {
                        s_ia[sp_id] = ia;
                        s_ja[sp_id] = ja;
                        s_pa[sp_id] = pair_ij < shl_pair1;
                    }
                    if (sp_id == 0) {
                        s_ka[aux_id] = ka;
                        s_kact[aux_id] = kidx < ksh1;
                    }
                }
                for (int g = 0; g < gout_stride; ++g) {
                    if (gout_id == g) {
#pragma unroll
                        for (int n = 0; n < NSUM; ++n) {
                            double v = active ? s[n] : 0.;
                            red[n*nst+st_id] = g == 0 ? v : red[n*nst+st_id] + v;
                        }
                    }
                    __syncthreads();
                }
                block_flush<NT>(ejk, red, s_ia, s_ja, s_ka, s_pa, s_kact,
                                natm, nst, nsp_per_block, aux_per_block,
                                thread_id);
            } else if (active) {
                triple_atomics(ejk, s, ia, ja, ka, natm);
            }
#endif
        }
    }
}

extern "C" {
int ejk_int3c2e_ip2(double *ejk, double *dm, double *density_auxvec,
                    RysIntEnvVars *envs, double omega, double lr_factor,
                    double sr_factor, int shm_size, int nbatches_shl_pair,
                    int nbatches_ksh, int *shl_pair_offsets, uint32_t *bas_ij_idx,
                    int *ksh_offsets, int *gout_stride_lookup,
                    int *ao_pair_loc, int aux_offset, int naux)
{
    dim3 blocks(nbatches_shl_pair, nbatches_ksh);
    // the register path needs the reduction buffer and its roots at 256
    // triples per block, whatever the caller's scheme asked for
    int shm_reg = shm_size > IP2_REG_SHM ? shm_size : IP2_REG_SHM;
#define IP2_LAUNCH(nt, mode, shm) \
    cudaFuncSetAttribute(ejk_int3c2e_ip2_kernel<nt, mode>, \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm); \
    ejk_int3c2e_ip2_kernel<nt, mode><<<blocks, nt, shm>>>( \
            ejk, dm, density_auxvec, *envs, omega, lr_factor, sr_factor, \
            shl_pair_offsets, bas_ij_idx, ksh_offsets, \
            gout_stride_lookup, ao_pair_loc, aux_offset, naux, shm);
#if IP2_SPLIT
    // the register path's blocks and the shared-memory path's blocks as two
    // kernels, so neither one's register budget is set by the other
    IP2_LAUNCH(THREADS, 1, shm_reg)
    IP2_LAUNCH(IP2_THREADS, 2, shm_size)
#else
    IP2_LAUNCH(IP2_THREADS, 0, shm_reg)
#endif
#undef IP2_LAUNCH
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ejk_int3c2e_ip2: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
