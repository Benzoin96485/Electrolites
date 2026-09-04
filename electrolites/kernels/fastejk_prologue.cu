// ---------------------------------------------------------------------------
// Fast nuclear-gradient two-electron kernels for GPU4PySCF, Rys quadrature.
//
// Replaces gvhf-rys/rys_contract_jk_ip1.cu's RYS_per_atom_jk_ip1 -- the
// per-atom derivative of j_factor*J - k_factor*K, which is 75-96 % of an RKS
// gradient on these clusters.  The integral arithmetic is the same Rys
// recurrences and the same root/weight tables; what changes is the same
// scaffolding fastk changed for the exchange build:
//
//   * no double-precision division in the primitive loops.  GPU4PySCF
//     evaluates cicj*ckcl/(aij*akl*sqrt(aij+akl)) once per primitive quartet
//     plus rt/(aij+akl), .5/aij and .5/akl once per Rys root; here 1/aij is
//     block-uniform and cached, 1/akl is hoisted out of the bra loop, and one
//     rsqrt(aij+akl) turns the rest into multiplies.
//   * the block-uniform bra data (aij, 1/aij, aj/aij, 2ai, 2aj, the product
//     centre, cicj) is computed once per bra shell pair into shared memory
//     rather than once per thread per primitive quartet.
//   * the density products dd are kept in registers.  GPU4PySCF's *general*
//     kernel writes them to a global-memory pool (dd_pool) and re-reads nf of
//     them per Rys root per primitive quartet.
//   * every g-array address is a compile-time constant, and the gout tile is
//     exactly nf wide instead of the general kernel's padded 3x3x3x3.
//   * range separation is one fma: rsqrt(aij+akl) -> rsqrt(aij+akl +
//     aij*akl/omega^2) builds the long-range (erf) operator, so a
//     range-separated hybrid gradient never needs the short-range operator's
//     doubled Rys roots.
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
#define SQRT_PIE4       .8862269254527580136

#define DEGREE          13
#define DEGREE1         (DEGREE+1)
#define INTERVALS       40

// offsets inside the packed Rys table (see fastk.py::_tables)
#define OFF_SMALLX_R0   0
#define OFF_SMALLX_R1   55
#define OFF_SMALLX_W0   110
#define OFF_SMALLX_W1   165
#define OFF_LARGEX_R    220
#define OFF_LARGEX_W    275
#define OFF_RW          330

#define MAX_PRIM_PAIR   36      // 6-prim x 6-prim is the worst case in Pople sets

// ---------------------------------------------------------------------------
// Rys roots and weights.  Bit-for-bit the recurrences of gvhf-rys/rys_roots.cu;
// the large-argument branch takes one rsqrt where GPU4PySCF takes a sqrt and a
// division per root.
// ---------------------------------------------------------------------------
template <int NROOTS> __device__ __forceinline__
void ejk_roots_reg(double x, double *rw, const double *tab)
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
        double t = SQRT_PIE4 * ix;
        double invx = ix * ix;
#pragma unroll
        for (int i = 0; i < NROOTS; ++i) {
            rw[i*2  ] = tab[OFF_LARGEX_R+off+i] * invx;
            rw[i*2+1] = tab[OFF_LARGEX_W+off+i] * t;
        }
        return;
    }
    if (NROOTS == 1) {
        double ix = rsqrt(x);
        double fmt0 = SQRTPIE4 * ix * erf(x * ix);
        double e = exp(-x);
        double b = .5 * ix * ix;
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

// Same, into shared memory, distributed over threadIdx.y.
template <int NROOTS> __device__ __forceinline__
void ejk_roots_tab(double x, double *rw, int block_size, int rt_id, int stride,
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
        double t = SQRT_PIE4 * ix;
        double invx = ix * ix;
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
        rw[i*block_size] = c0 + c1*u;
    }
}

// ---------------------------------------------------------------------------
// Task generation: identical screening to gvhf-rys/create_tasks.cu
// (_fill_ejk_tasks).  The gradient's density factor is a *product* of two
// density-matrix elements, so the exchange test adds the two logs rather than
// testing each on its own -- that is GPU4PySCF's test, kept verbatim.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void fill_ejk_tasks(int *ntasks, int *bas_kl_idx, int bas_ij, int nbas,
                    const int *pair_kl_mapping, int npairs_kl,
                    const float *q_cond, const float *dm_cond, float cutoff,
                    int do_j, int do_k)
{
    int t_id = threadIdx.x + blockDim.x * threadIdx.y;
    int threads = blockDim.x * blockDim.y;
    int ish = bas_ij / nbas;
    int jsh = bas_ij % nbas;
    float q_ij = q_cond[bas_ij];
    float d_ij = dm_cond[bas_ij];
    float kl_cutoff = cutoff - q_ij;
    for (int pair_kl = t_id; pair_kl < npairs_kl; pair_kl += threads) {
        int bas_kl = pair_kl_mapping[pair_kl];
        float q_kl = q_cond[bas_kl];
        if (q_kl < kl_cutoff) continue;
        if (bas_ij < bas_kl) continue;
        int ksh = bas_kl / nbas;
        int lsh = bas_kl % nbas;
        float d_cutoff = kl_cutoff - q_kl;
        if ((do_k && (dm_cond[ish*nbas+lsh]+dm_cond[jsh*nbas+ksh] > d_cutoff ||
                      dm_cond[ish*nbas+ksh]+dm_cond[jsh*nbas+lsh] > d_cutoff)) ||
            (do_j && d_ij+dm_cond[bas_kl] > d_cutoff)) {
            int off = atomicAdd(ntasks, 1);
            bas_kl_idx[off] = bas_kl;
        }
    }
    __syncthreads();
    // the last lanes of a block may run one thread-block past ntasks with a
    // zero density factor, so pad the queue with a valid shell pair
    bas_kl_idx[*ntasks + t_id] = pair_kl_mapping[0];
    __syncthreads();
}

#define EARGS                                                                  \
    double *ejk, const double *dm, int nao,                                    \
    double j_factor, double k_factor,                                          \
    const int *bas, const double *env, const int *ao_loc, int nbas,            \
    int npairs_ij, int npairs_kl,                                              \
    const int *pair_ij_mapping, const int *pair_kl_mapping,                    \
    const float *q_cond, const float *dm_cond, float cutoff,                   \
    int *pool, int *head, int queue_depth, const double *tab,                  \
    int iprim_a, int jprim_a, int kprim_a, int lprim_a,                        \
    double inv_om2, double coef0
