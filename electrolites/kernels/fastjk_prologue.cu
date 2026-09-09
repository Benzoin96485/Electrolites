// ---------------------------------------------------------------------------
// Additions for the *fused* Coulomb-and-exchange build (catalogue B4b).
//
// GPU4PySCF's SCF builds J and K from two separate passes over the shell
// quartets -- the McMurchie-Davidson j_engine for J, Rys for K -- but its
// response path (`scf/_response_functions.py` -> `scf/jk.py:_VHFOpt.get_jk`)
// asks for both from **one** Rys pass, because J is nearly free from integrals
// that have already been computed for K.  That is a third code path, and
// neither `fastj` nor `fastk` is on it: a polarizability or a TDDFT run of a
// hybrid was stock GPU4PySCF in 61-82 % of its wall clock
// (docs/RESPONSE_1.8.1.md 3.2).  The kernels here are that pass, with the
// same P1-P5 treatment `fastk`'s carry and with the density loop inside the
// quartet, so n_dm densities cost one integral pass.
//
// This file is included after fastk_prologue.cu and uses its Rys roots, its
// tables and its shared-memory conventions.  What it adds is the task filter
// (J reads two shell-pair blocks of the density that K does not, so two more
// dm_cond tests decide a quartet) and the argument macros carrying `vj`.
// ---------------------------------------------------------------------------

// Screening for a fused build: gvhf-rys/create_tasks.cu's _fill_vjk_tasks.
// The two extra tests are the J ones -- dm_cond[bas_ij] for vj[k,l] and
// dm_cond[bas_kl] for vj[i,j] -- so this filter keeps a superset of what
// fill_vk_tasks keeps, and a fused build cannot drop a quartet that either
// separate build would have kept.
__device__ __forceinline__
void fill_vjk_tasks(int *ntasks, int *bas_kl_idx, int bas_ij, int nbas,
                    const int *pair_kl_mapping, int npairs_kl,
                    const float *q_cond, const float *dm_cond, float cutoff)
{
    int t_id = threadIdx.x;
    int threads = blockDim.x;
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
        if (d_ij                  > d_cutoff ||
            dm_cond[bas_kl]       > d_cutoff ||
            dm_cond[ish*nbas+ksh] > d_cutoff ||
            dm_cond[jsh*nbas+ksh] > d_cutoff ||
            dm_cond[ish*nbas+lsh] > d_cutoff ||
            dm_cond[jsh*nbas+lsh] > d_cutoff) {
            int off = atomicAdd(ntasks, 1);
            bas_kl_idx[off] = bas_kl;
        }
    }
    __syncthreads();
}

// The same filter for a 2D thread block, with the queue padding
// fill_vk_tasks2 explains.
__device__ __forceinline__
void fill_vjk_tasks2(int *ntasks, int *bas_kl_idx, int bas_ij, int nbas,
                     const int *pair_kl_mapping, int npairs_kl,
                     const float *q_cond, const float *dm_cond, float cutoff)
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
        if (d_ij                  > d_cutoff ||
            dm_cond[bas_kl]       > d_cutoff ||
            dm_cond[ish*nbas+ksh] > d_cutoff ||
            dm_cond[jsh*nbas+ksh] > d_cutoff ||
            dm_cond[ish*nbas+lsh] > d_cutoff ||
            dm_cond[jsh*nbas+lsh] > d_cutoff) {
            int off = atomicAdd(ntasks, 1);
            bas_kl_idx[off] = bas_kl;
        }
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        bas_kl_idx[*ntasks + threadIdx.x] = pair_kl_mapping[0];
    }
    __syncthreads();
}

// KARGS with a Coulomb matrix in front of the exchange one.
#define JKARGS                                                                 \
    double *vj_all, double *vk_all, const double *dm_all, int n_dm, int nao,   \
    const int *bas, const double *env, const int *ao_loc, int nbas,            \
    int npairs_ij, int npairs_kl,                                              \
    const int *pair_ij_mapping, const int *pair_kl_mapping,                    \
    const float *q_cond, const float *dm_cond, float cutoff,                   \
    int *pool, int *head, int queue_depth, const double *tab,                  \
    double inv_om2, double coef0, double coef1
#define JKFWD                                                                  \
    vj_all, vk_all, dm_all, n_dm, nao, bas, env, ao_loc, nbas,                 \
    npairs_ij, npairs_kl,                                                      \
    pair_ij_mapping, pair_kl_mapping, q_cond, dm_cond, cutoff,                 \
    pool, head, queue_depth, tab, inv_om2, coef0, coef1

// KARGS2 with the same addition, for the 2D-block families.
#define JKARGS2                                                                \
    double *vj_all, double *vk_all, double *dm_all, int n_dm, int nao,         \
    int *bas, double *env, int *ao_loc, int nbas,                              \
    int npairs_ij, int npairs_kl,                                              \
    int *pair_ij_mapping, int *pair_kl_mapping,                                \
    float *q_cond, float *dm_cond, float cutoff,                               \
    int *pool, int *head, int queue_depth, const double *tab,                  \
    int iprim_a, int jprim_a, int kprim_a, int lprim_a,                        \
    double inv_om2, double coef0, double coef1
#define JKFWD2                                                                 \
    vj_all, vk_all, dm_all, n_dm, nao, bas, env, ao_loc, nbas,                 \
    npairs_ij, npairs_kl,                                                      \
    pair_ij_mapping, pair_kl_mapping, q_cond, dm_cond, cutoff,                 \
    pool, head, queue_depth, tab, iprim_a, jprim_a, kprim_a, lprim_a,          \
    inv_om2, coef0, coef1
