// ---------------------------------------------------------------------------
// Fast exchange-matrix (K) build kernels for GPU4PySCF, Rys quadrature.
//
// Same algorithm and the same Rys root/weight tables as GPU4PySCF's
// gvhf-rys/unrolled_rys_k.cu, but restructured so that
//
//   * the Rys roots/weights live in registers instead of shared memory, which
//     removes the __syncthreads() that GPU4PySCF executes once per *primitive
//     quartet* in the innermost loop (it is only there to protect the shared
//     rw buffer).  That barrier both costs cycles and stops warps from running
//     ahead of each other, which is what keeps the kernel at ~1.5 IPC.
//   * the parts of the primitive-pair arithmetic that depend only on the
//     block-uniform bra shell pair (aij, 1/aij, the product centre, cicj) are
//     computed once per bra pair into shared memory, instead of once per
//     thread per primitive quartet.
//   * the reciprocals are folded: one division and one rsqrt per primitive
//     quartet instead of three divisions and one sqrt.
//   * (ss|ss) skips the Rys *root* entirely -- only the weight is used there,
//     but GPU4PySCF still pays an exp() and a division for the unused root
//     because it is written through shared memory.
//
// All arithmetic is double precision.
// ---------------------------------------------------------------------------

#define BAS_SLOTS       8
#define ATOM_OF         0
#define ANG_OF          1
#define NPRIM_OF        2
#define NCTR_OF         3
#define KAPPA_OF        4
#define PTR_EXP         5
#define PTR_COEFF       6
#define PTR_BAS_COORD   7

#define PI_FAC          34.98683665524972497
#define SQRTPIE4        .8862269254527580136
#define PIE4            .7853981633974483096
#define SQRT_PIE4       .8862269254527580136

#define DEGREE          13
#define DEGREE1         (DEGREE+1)
#define INTERVALS       40

// offsets inside the packed Rys table (see fastk.py)
#define OFF_SMALLX_R0   0
#define OFF_SMALLX_R1   55
#define OFF_SMALLX_W0   110
#define OFF_SMALLX_W1   165
#define OFF_LARGEX_R    220
#define OFF_LARGEX_W    275
#define OFF_RW          330

#define MAX_PRIM_PAIR   36      // 6-prim x 6-prim is the worst case in Pople sets

// ---------------------------------------------------------------------------
// Rys roots and weights, entirely in registers.  Bit-for-bit the same
// recurrences as gvhf-rys/rys_roots.cu with block_size = stride = 1.
// rw[2*i] = root_i, rw[2*i+1] = weight_i
// ---------------------------------------------------------------------------
template <int NROOTS> __device__ __forceinline__
void rys_roots_reg(double x, double *rw, const double *tab)
{
    constexpr int off = NROOTS * (NROOTS - 1) / 2;
    if (x < 3.e-7) {
#pragma unroll
        for (int i = 0; i < NROOTS; ++i) {
            rw[i*2  ] = tab[OFF_SMALLX_R0+off+i] + tab[OFF_SMALLX_R1+off+i] * x;
            rw[i*2+1] = tab[OFF_SMALLX_W0+off+i] + tab[OFF_SMALLX_W1+off+i] * x;
        }
        return;
    }
    if (x > 35 + NROOTS*5) {
        double ix = rsqrt(x);
        double t = SQRT_PIE4 * ix;          // sqrt(PIE4/x)
        double invx = ix * ix;
#pragma unroll
        for (int i = 0; i < NROOTS; ++i) {
            rw[i*2  ] = tab[OFF_LARGEX_R+off+i] * invx;
            rw[i*2+1] = tab[OFF_LARGEX_W+off+i] * t;
        }
        return;
    }
    if (NROOTS == 1) {
        double ix = rsqrt(x);               // 1/sqrt(x); x > 3e-7 here
        double tt = x * ix;
        double fmt0 = SQRTPIE4 * ix * erf(tt);
        double e = exp(-x);
        double b = .5 * ix * ix;            // .5/x
        rw[0] = b * (fmt0 - e) / fmt0;
        rw[1] = fmt0;
        return;
    }
    const double *datax = tab + OFF_RW + DEGREE1*INTERVALS * (NROOTS*(NROOTS-1));
    int it = (int)(x * .4);
    double u = (x - it * 2.5) * 0.8 - 1.;
    double u2 = u * 2.;
#pragma unroll
    for (int i = 0; i < NROOTS*2; ++i) {
        const double *c = datax + i * DEGREE1 * INTERVALS;
        double c0 = c[it + DEGREE   *INTERVALS];
        double c1 = c[it +(DEGREE-1)*INTERVALS];
        double c2, c3;
#pragma unroll
        for (int n = DEGREE-2; n > 0; n -= 2) {
            c2 = c[it + n   *INTERVALS] - c1;
            c3 = c0 + c1*u2;
            c1 = c2 + c3*u2;
            c0 = c[it +(n-1)*INTERVALS] - c3;
        }
        rw[i] = c0 + c1*u;      // DEGREE is odd
    }
}

// (ss|ss) needs the weight only; the root is dead work.
__device__ __forceinline__
double rys_weight0(double x, const double *tab)
{
    if (x < 3.e-7) return tab[OFF_SMALLX_W0] + tab[OFF_SMALLX_W1] * x;
    double ix = rsqrt(x);
    if (x > 40.)   return tab[OFF_LARGEX_W] * SQRT_PIE4 * ix;
    return SQRTPIE4 * ix * erf(x * ix);
}

// ---------------------------------------------------------------------------
// Task generation: identical screening to gvhf-rys/create_tasks.cu
// (_fill_vk_tasks), writing a compacted list of bra-ket pairs.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void fill_vk_tasks(int *ntasks, int *bas_kl_idx, int bas_ij, int nbas,
                   const int *pair_kl_mapping, int npairs_kl,
                   const float *q_cond, const float *dm_cond, float cutoff)
{
    int t_id = threadIdx.x;
    int threads = blockDim.x;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    float q_ij = q_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;
    for (int pair_kl = t_id; pair_kl < npairs_kl; pair_kl += threads) {
        int bas_kl = pair_kl_mapping[pair_kl];
        float q_kl = q_cond[bas_kl];
        if (q_kl < kl_cutoff) continue;
        if (bas_ij < bas_kl) continue;
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        float d_cutoff = kl_cutoff - q_kl;
        if (dm_cond[ish*nbas+ksh] > d_cutoff ||
            dm_cond[jsh*nbas+ksh] > d_cutoff ||
            dm_cond[ish*nbas+lsh] > d_cutoff ||
            dm_cond[jsh*nbas+lsh] > d_cutoff) {
            int off = atomicAdd(ntasks, 1);
            bas_kl_idx[off] = bas_kl;
        }
    }
    __syncthreads();
}
// ---------------------------------------------------------------------------
// (ss|ss)
// ---------------------------------------------------------------------------
#define KARGS                                                                  \
    double *vk, const double *dm, int nao,                                     \
    const int *bas, const double *env, const int *ao_loc, int nbas,            \
    int npairs_ij, int npairs_kl,                                              \
    const int *pair_ij_mapping, const int *pair_kl_mapping,                    \
    const float *q_cond, const float *dm_cond, float cutoff,                   \
    int *pool, int *head, int queue_depth, const double *tab,               \
    double inv_om2, double coef0, double coef1
// the same names, to forward KARGS from an entry point to the templated body
#define KFWD                                                                   \
    vk, dm, nao, bas, env, ao_loc, nbas, npairs_ij, npairs_kl,                 \
    pair_ij_mapping, pair_kl_mapping, q_cond, dm_cond, cutoff,                 \
    pool, head, queue_depth, tab, inv_om2, coef0, coef1

template <int NRANGE> __device__ __forceinline__ void
kbody_0000(KARGS)
{
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int *bas_kl_idx = pool + blockIdx.x * queue_depth;

    __shared__ int ntasks, pair_ij;
    __shared__ double s_cicj[MAX_PRIM_PAIR];
    __shared__ double s_aij [MAX_PRIM_PAIR];
    __shared__ double s_iaij[MAX_PRIM_PAIR];
    __shared__ double s_xij [MAX_PRIM_PAIR];
    __shared__ double s_yij [MAX_PRIM_PAIR];
    __shared__ double s_zij [MAX_PRIM_PAIR];

    if (tid == 0) pair_ij = atomicAdd(head, 1);
    __syncthreads();

    while (pair_ij < npairs_ij) {
        int bas_ij = pair_ij_mapping[pair_ij];
        if (tid == 0) ntasks = 0;
        __syncthreads();
        fill_vk_tasks(&ntasks, bas_kl_idx, bas_ij, nbas, pair_kl_mapping,
                      npairs_kl, q_cond, dm_cond, cutoff);
        if (ntasks == 0) {
            if (tid == 0) pair_ij = atomicAdd(head, 1);
            __syncthreads();
            continue;
        }

        int ish = bas_ij / nbas;
        int jsh = bas_ij % nbas;
        int iprim = bas[ish*BAS_SLOTS+NPRIM_OF];
        int jprim = bas[jsh*BAS_SLOTS+NPRIM_OF];
        const double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        const double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        const double *ci   = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        const double *cj   = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        const double *ri   = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        const double *rj   = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double xjxi = rj[0]-ri[0], yjyi = rj[1]-ri[1], zjzi = rj[2]-ri[2];
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        int nprim_ij = iprim*jprim;
        // block-uniform bra data, computed once per bra shell pair
        for (int ij = tid; ij < nprim_ij; ij += nthreads) {
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1./aij;
            double aj_aij = aj * iaij;
            s_aij [ij] = aij;
            s_iaij[ij] = iaij;
            s_cicj[ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
            s_xij [ij] = ri[0] + xjxi*aj_aij;
            s_yij [ij] = ri[1] + yjyi*aj_aij;
            s_zij [ij] = ri[2] + zjzi*aj_aij;
        }
        __syncthreads();

        int i0 = ao_loc[ish], j0 = ao_loc[jsh];
        double sym_ij = (ish == jsh) ? .5*PI_FAC : PI_FAC;

        for (int task_id = tid; task_id < ntasks; task_id += nthreads) {
            int bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl % nbas;
            double fac_sym = sym_ij;
            if (ksh == lsh) fac_sym *= .5;
            if (bas_ij == bas_kl) fac_sym *= .5;
            int kprim = bas[ksh*BAS_SLOTS+NPRIM_OF];
            int lprim = bas[lsh*BAS_SLOTS+NPRIM_OF];
            const double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
            const double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
            const double *ck   = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
            const double *cl   = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
            const double *rk   = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            const double *rl   = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
            double xlxk = rl[0]-rk[0], ylyk = rl[1]-rk[1], zlzk = rl[2]-rk[2];
            double rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;

            double gout0 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1./akl;
                double al_akl = al * iakl;
                double ckcl = fac_sym * ck[klp/lprim] * cl[klp%lprim]
                            * exp(-ak*al_akl*rr_kl);
                double xkl = rk[0] + xlxk*al_akl;
                double ykl = rk[1] + ylyk*al_akl;
                double zkl = rk[2] + zlzk*al_akl;
                for (int ijp = 0; ijp < nprim_ij; ++ijp) {
                    double xpq = s_xij[ijp] - xkl;
                    double ypq = s_yij[ijp] - ykl;
                    double zpq = s_zij[ijp] - zkl;
                    double rr  = xpq*xpq + ypq*ypq + zpq*zpq;
                    double aij = s_aij[ijp];
                    double t = aij * akl;
                    double ss = aij + akl;
                    // inv_om2 = 0 for the full-range operator; for the
                    // long-range (erf) operator inv_om2 = 1/omega^2 and this
                    // one fma carries the whole range separation (see
                    // README_fastk.md).  NRANGE == 2 sums the two operators
                    // with weights coef0 and coef1 in one pass.
                    for (int h = 0; h < NRANGE; ++h) {
                    double inv_s = rsqrt(fma(t, NRANGE == 1 ? inv_om2
                                                : (h == 0 ? 0. : inv_om2), ss));
                    double w = rys_weight0(t * rr * inv_s * inv_s, tab);
                    if (NRANGE > 1) w *= (h == 0 ? coef0 : coef1);
                    // 1/(aij*akl*sqrt(aij+akl)) with no division
                    gout0 += s_cicj[ijp] * ckcl * w * s_iaij[ijp] * iakl * inv_s;
                    }
                }
            }
            int k0 = ao_loc[ksh], l0 = ao_loc[lsh];
            atomicAdd(vk + i0*nao + l0, gout0 * dm[j0*nao + k0]);
            atomicAdd(vk + j0*nao + l0, gout0 * dm[i0*nao + k0]);
            atomicAdd(vk + i0*nao + k0, gout0 * dm[j0*nao + l0]);
            atomicAdd(vk + j0*nao + k0, gout0 * dm[i0*nao + l0]);
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(256, 4)
k_0000(KARGS)    { kbody_0000<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(256, 4)
k_rs_0000(KARGS) { kbody_0000<2>(KFWD); }


// ===========================================================================
// Additions for the angular-momentum classes GPU4PySCF splits across
// threadIdx.y (gen_k2_kernels.py).  Those kernels keep the 2D integrals in
// shared memory, so the Rys roots have to stay there too; what changes is the
// arithmetic around them -- see README_fastk.md.
// ===========================================================================

// Same screening as fill_vk_tasks, but for a 2D thread block.
__device__ __forceinline__
void fill_vk_tasks2(int *ntasks, int *bas_kl_idx, int bas_ij, int nbas,
                    const int *pair_kl_mapping, int npairs_kl,
                    const float *q_cond, const float *dm_cond, float cutoff)
{
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    int threads = blockDim.x * blockDim.y;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    float q_ij = q_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;
    for (int pair_kl = t_id; pair_kl < npairs_kl; pair_kl += threads) {
        int bas_kl = pair_kl_mapping[pair_kl];
        float q_kl = q_cond[bas_kl];
        if (q_kl < kl_cutoff) continue;
        if (bas_ij < bas_kl) continue;
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        float d_cutoff = kl_cutoff - q_kl;
        if (dm_cond[ish*nbas+ksh] > d_cutoff ||
            dm_cond[jsh*nbas+ksh] > d_cutoff ||
            dm_cond[ish*nbas+lsh] > d_cutoff ||
            dm_cond[jsh*nbas+lsh] > d_cutoff) {
            int off = atomicAdd(ntasks, 1);
            bas_kl_idx[off] = bas_kl;
        }
    }
    __syncthreads();
    // These kernels let the last lanes of a block run one thread-block past
    // ntasks (with a zero symmetry factor), so the queue is padded with a
    // valid shell pair rather than left uninitialised -- same as GPU4PySCF's
    // _fill_vk_tasks.
    if (threadIdx.y == 0) {
        bas_kl_idx[*ntasks + threadIdx.x] = pair_kl_mapping[0];
    }
    __syncthreads();
}

// Rys roots and weights into shared memory, distributed over threadIdx.y.
// Bit-for-bit the recurrences of gvhf-rys/rys_roots.cu; the large-argument
// branch takes one rsqrt where GPU4PySCF takes a sqrt and a division per root.
template <int NROOTS> __device__ __forceinline__
void rys_roots_tab(double x, double *rw, int block_size, int rt_id, int stride,
                   const double *tab)
{
    constexpr int off = NROOTS * (NROOTS - 1) / 2;
    double *r = rw;
    double *w = rw + block_size;
    const int block_size2 = block_size * 2;
    if (x < 3.e-7) {
        for (int i = rt_id; i < NROOTS; i += stride) {
            r[i*block_size2] = tab[OFF_SMALLX_R0+off+i] + tab[OFF_SMALLX_R1+off+i]*x;
            w[i*block_size2] = tab[OFF_SMALLX_W0+off+i] + tab[OFF_SMALLX_W1+off+i]*x;
        }
        return;
    }
    if (x > 35 + NROOTS*5) {
        double ix = rsqrt(x);
        double t = SQRT_PIE4 * ix;      // sqrt(PIE4/x)
        double invx = ix * ix;          // 1/x
        for (int i = rt_id; i < NROOTS; i += stride) {
            r[i*block_size2] = tab[OFF_LARGEX_R+off+i] * invx;
            w[i*block_size2] = tab[OFF_LARGEX_W+off+i] * t;
        }
        return;
    }
    const double *datax = tab + OFF_RW + DEGREE1*INTERVALS * (NROOTS*(NROOTS-1));
    int it = (int)(x * .4);
    double u = (x - it * 2.5) * 0.8 - 1.;
    double u2 = u * 2.;
    for (int i = rt_id; i < NROOTS*2; i += stride) {
        const double *c = datax + i * DEGREE1 * INTERVALS;
        double c0 = c[it + DEGREE   *INTERVALS];
        double c1 = c[it +(DEGREE-1)*INTERVALS];
        double c2, c3;
#pragma unroll
        for (int n = DEGREE-2; n > 0; n -= 2) {
            c2 = c[it + n   *INTERVALS] - c1;
            c3 = c0 + c1*u2;
            c1 = c2 + c3*u2;
            c0 = c[it +(n-1)*INTERVALS] - c3;
        }
        rw[i*block_size] = c0 + c1*u;   // DEGREE is odd
    }
}

#define KARGS2                                                                 \
    double *vk_all, double *dm_all, int nao,                                   \
    int *bas, double *env, int *ao_loc, int nbas,                              \
    int npairs_ij, int npairs_kl,                                              \
    int *pair_ij_mapping, int *pair_kl_mapping,                                \
    float *q_cond, float *dm_cond, float cutoff,                               \
    int *pool, int *head, int queue_depth, const double *tab,                  \
    int iprim_a, int jprim_a, int kprim_a, int lprim_a,                        \
    double inv_om2, double coef0, double coef1
#define KFWD2                                                                  \
    vk_all, dm_all, nao, bas, env, ao_loc, nbas, npairs_ij, npairs_kl,         \
    pair_ij_mapping, pair_kl_mapping, q_cond, dm_cond, cutoff,                 \
    pool, head, queue_depth, tab, iprim_a, jprim_a, kprim_a, lprim_a,          \
    inv_om2, coef0, coef1
