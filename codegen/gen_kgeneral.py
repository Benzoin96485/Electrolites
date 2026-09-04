"""
Generate a faster copy of GPU4PySCF's *general* K kernel (gvhf-rys/rys_k_kernel).

Seven of the 25 angular-momentum classes a 6-31G* organic system generates --
(dp|dp), (dd|pp), (ds|dd), (dd|ds), (dp|dd), (dd|dp) and (dd|dd) -- are not
unrolled by GPU4PySCF, so they run on the general Rys kernel and are the
largest block fastk does not touch.  This script lifts that kernel out of
GPU4PySCF's source and applies the same three changes the unrolled classes got:

  * no double-precision division in the hot loop.  The general kernel evaluates
    aj/aij, aij*akl/(aij+akl), rt/(aij+akl), .5/aij and .5/akl -- 2 + 3*nroots
    divisions per primitive quartet per thread -- where one rsqrt(aij+akl) and
    two reciprocals cached per shell pair suffice.
  * the block-uniform bra data (aij, 1/aij, aj/aij and the product centre) is
    computed once per bra shell pair into shared memory instead of once per
    thread per primitive quartet.
  * range separation through the same rsqrt(s) -> rsqrt(s + aij*akl/omega^2)
    substitution the unrolled classes use, so the long-range (erf) operator
    costs one extra fma instead of a division, a sqrt and a scaling pass over
    the roots.

One block per bra pair with the queue indexed by SM id (GPU4PySCF's scheme)
is only correct while exactly one block is resident per SM, which the original
gets from its 255-register footprint.  Removing arithmetic frees registers, so
the copy switches to fastk's persistent grid: a fixed number of blocks, each
with its own queue slot, claiming bra pairs from an atomic counter.

Run:  python gen_kgeneral.py <path-to-rys_contract_k.cu> > cuda/myk2/rys_k_general.cu
"""
import re
import sys


def _extract(src, name):
    i = src.index('void ' + name + '(')
    i = src.rindex('__global__', 0, i)
    j = src.index('{', src.index(')', i))
    depth, k = 1, j + 1
    while depth:
        if src[k] == '{':
            depth += 1
        elif src[k] == '}':
            depth -= 1
        k += 1
    return src[i:k]


def _sub1(body, old, new, tag=''):
    assert body.count(old) == 1, (tag, old[:70], body.count(old))
    return body.replace(old, new, 1)


def transform(body):
    # ---- signature: keep GPU4PySCF's structs, add the range-separation and
    # ---- persistent-grid arguments
    body = _sub1(body, """__global__ static
void rys_k_kernel(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                  int *pool, GXYZOffset *gxyz_offsets,
                  int gout_pattern, int reserved_shm_size)
{""", """__global__ static
void myk_k_general(RysIntEnvVars envs, JKMatrix kmat, BoundsInfo bounds,
                   int *pool, GXYZOffset *gxyz_offsets,
                   int gout_pattern, int reserved_shm_size,
                   int *head, int queue_depth, double inv_om2)
{""", 'signature')

    # ---- persistent grid: one queue per block, bra pairs from an atomic
    body = _sub1(body, """    int smid = get_smid();
    int *bas_kl_idx = pool + smid * QUEUE_DEPTH;
    __shared__ int ntasks;
    if (sq_id == 0 && gout_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    int bas_ij = bounds.pair_ij_mapping[blockIdx.x];
    if (kmat.lr_factor != 0) {
        _fill_vk_tasks(&ntasks, bas_kl_idx, bas_ij, envs, bounds);
    } else {
        _fill_sr_vk_tasks(&ntasks, bas_kl_idx, bas_ij, envs, bounds);
    }
    if (ntasks == 0) {
        return;
    }
""", """    int *bas_kl_idx = pool + blockIdx.x * queue_depth;
    __shared__ int ntasks;
    __shared__ int pair_ij;
    if (sq_id == 0 && gout_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
""", 'queue')

    # the bra-independent index tables are built once, before the pair loop;
    # everything from the bra pair onwards goes inside it
    body = _sub1(body, """    __shared__ int ish, jsh, i0, j0, nao;""",
                 """    while (pair_ij < bounds.npairs_ij) {
    int bas_ij = bounds.pair_ij_mapping[pair_ij];
    __syncthreads();
    if (sq_id == 0 && gout_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    _fill_vk_tasks(&ntasks, bas_kl_idx, bas_ij, envs, bounds);
    __syncthreads();
    if (ntasks == 0) {
        if (sq_id == 0 && gout_id == 0) {
            pair_ij = atomicAdd(head, 1);
        }
        __syncthreads();
        continue;
    }

    __shared__ int ish, jsh, i0, j0, nao;""", 'pair loop head')

    # ---- shared caches for the block-uniform bra data
    body = _sub1(body, """    double *cicj_cache = shared_memory + reserved_shm_size - iprim*jprim;""",
                 """    double *cicj_cache = shared_memory + reserved_shm_size - 7*iprim*jprim;
    double *s_aij   = cicj_cache + 1*iprim*jprim;
    double *s_iaij  = cicj_cache + 2*iprim*jprim;
    double *s_ajaij = cicj_cache + 3*iprim*jprim;
    double *s_xij   = cicj_cache + 4*iprim*jprim;
    double *s_yij   = cicj_cache + 5*iprim*jprim;
    double *s_zij   = cicj_cache + 6*iprim*jprim;""", 'bra cache')

    # ---- akl_cache needs a third slot for 1/akl, so everything after it moves
    body = _sub1(body, """    double *fac_ijkl = shared_memory + nsq_per_block * 8 + sq_id;
    double *gx = shared_memory + nsq_per_block * 9 + sq_id;
    double *rw = shared_memory + nsq_per_block * (g_size*3+9) + sq_id;""",
                 """    double *fac_ijkl = shared_memory + nsq_per_block * 9 + sq_id;
    double *gx = shared_memory + nsq_per_block * 10 + sq_id;
    double *rw = shared_memory + nsq_per_block * (g_size*3+10) + sq_id;""",
                 'shm layout')

    # ---- bra primitive pairs: one division per pair instead of one per quartet
    body = _sub1(body, """    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = env[expi+ip];
        double aj = env[expj+jp];
        double aij = ai + aj;
        double theta_ij = ai * aj / aij;
        double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        double Kab = exp(-theta_ij * rr_ij);
        cicj_cache[ij] = ci[ip] * cj[jp] * Kab;
    }""", """    double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
    for (int ij = t_id; ij < iprim*jprim; ij += threads) {
        int ip = ij / jprim;
        int jp = ij % jprim;
        double ai = env[expi+ip];
        double aj = env[expj+jp];
        double aij = ai + aj;
        double iaij = 1. / aij;
        double aj_aij = aj * iaij;
        s_aij  [ij] = aij;
        s_iaij [ij] = iaij;
        s_ajaij[ij] = aj_aij;
        s_xij  [ij] = ri[0] + xjxi * aj_aij;
        s_yij  [ij] = ri[1] + yjyi * aj_aij;
        s_zij  [ij] = ri[2] + zjzi * aj_aij;
        cicj_cache[ij] = ci[ip] * cj[jp] * exp(-ai * aj_aij * rr_ij);
    }
    __syncthreads();""", 'bra fill')

    # ---- ket primitive pair: 1/akl once, next to the product centre
    body = _sub1(body, """                double akl = ak + al;
                double al_akl = al / akl;""",
                 """                double akl = ak + al;
                double iakl = 1. / akl;
                double al_akl = al * iakl;""", 'iakl')
    body = _sub1(body, """                double theta_kl = ak * al / akl;""",
                 """                double theta_kl = ak * al_akl;""", 'theta_kl')
    body = _sub1(body, """                akl_cache[0] = akl;
                akl_cache[nsq_per_block] = al_akl;""",
                 """                akl_cache[0] = akl;
                akl_cache[nsq_per_block] = al_akl;
                akl_cache[2*nsq_per_block] = iakl;""", 'akl_cache')

    # ---- the primitive-quartet prologue reads the caches
    body = _sub1(body, """                int ip = ijp / jprim;
                int jp = ijp % jprim;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double akl = akl_cache[0];
                double al_akl = akl_cache[nsq_per_block];
                double xij = ri[0] + (rjri[0]) * aj_aij;
                double yij = ri[1] + (rjri[1]) * aj_aij;
                double zij = ri[2] + (rjri[2]) * aj_aij;""",
                 """                double aij = s_aij[ijp];
                double iaij = s_iaij[ijp];
                double akl = akl_cache[0];
                double al_akl = akl_cache[nsq_per_block];
                double iakl = akl_cache[2*nsq_per_block];
                // inv_om2 = 0 is the full-range operator, 1/omega^2 the
                // long-range (erf) one; see README_fastk.md
                double inv_s = rsqrt(fma(aij * akl, inv_om2, aij + akl));
                double inv_s2 = inv_s * inv_s;
                double xij = s_xij[ijp];
                double yij = s_yij[ijp];
                double zij = s_zij[ijp];""", 'quartet prologue')

    body = _sub1(body, """                    gx[nsq_per_block*g_size] = cicj / (aij*akl*sqrt(aij+akl));
                    if (sq_id == 0) {
                        aij_cache[0] = aij;
                        aij_cache[1] = aj_aij;
                    }""",
                 """                    gx[nsq_per_block*g_size] = cicj * iaij * iakl * inv_s;""",
                 'fac')
    body = _sub1(body, """                double theta = aij * akl / (aij + akl);""",
                 """                double theta = aij * akl * inv_s2;""", 'theta')
    body = _sub1(body, """                rys_roots_for_k(nroots, theta, rr, rw, kmat.omega,
                                kmat.lr_factor, kmat.sr_factor);""",
                 """                rys_roots(nroots, theta * rr, rw, nsq_per_block,
                          gout_id, gout_stride);""", 'roots')
    body = _sub1(body, """                    double rt = rw[irys*2*nsq_per_block];
                    double aij = aij_cache[0];
                    double akl = akl_cache[0];
                    double rt_aa = rt / (aij + akl);""",
                 """                    double rt = rw[irys*2*nsq_per_block];
                    double rt_aa = rt * inv_s2;""", 'rt_aa')
    body = body.replace('aij_cache[1]', 's_ajaij[ijp]')
    body = _sub1(body, '    __shared__ double aij_cache[2];\n', '', 'aij_cache decl')
    n = body.count('.5/aij') + body.count('.5/akl')
    assert n == 2, n
    body = body.replace('.5/aij', '.5*iaij').replace('.5/akl', '.5*iakl')

    # ---- close the persistent loop
    assert body.endswith('    }\n}')
    body = body[:-len('    }\n}')] + """    }
    if (sq_id == 0 && gout_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
    }
}"""

    assert 'aij_cache' not in body
    assert 'rys_roots_for_k' not in body
    assert 'get_smid' not in body
    leftover = [l for l in body.split('\n')
                if re.search(r'/(?!/)', re.sub(r'//.*', '', l))
                and not re.search(r'(/ *nbas|/ *jprim|/ *lprim|/ *3\b|/ *2\b'
                                  r'|1\. / aij|1\. / akl|/ *gout_stride)', l)]
    assert not leftover, leftover
    return body


HOST = r'''

// ---------------------------------------------------------------------------
// Host side.  Same thread scheme as GPU4PySCF's, plus the 6 extra doubles per
// bra primitive pair the shared bra cache needs, and fastk's persistent grid.
// ---------------------------------------------------------------------------
static size_t myk_threads_scheme(dim3& threads, BoundsInfo &bounds,
                                 int shm_size, int gout_stride_max)
{
    int ijprim = bounds.iprim * bounds.jprim;
    int ntiles_i = bounds.ntiles_i;
    int ntiles_j = bounds.ntiles_j;
    int ntiles_k = bounds.ntiles_k;
    int ntiles_l = bounds.ntiles_l;
    int ldi = ntiles_i * 3;
    int ldj = ntiles_j * 3;
    int ldk = ntiles_k * 3;
    int ldl = ntiles_l * 3;
    int cart_idx_size = (ntiles_i+ntiles_j+ntiles_k+ntiles_l)*9;
    int g_size = bounds.g_size;
    int nroots = bounds.nroots;
    int dm_cache_size = max(ldi, ldj) * max(ldk, ldl);
    int root_g_cache_size = nroots*2 + g_size*3 + 10;
    int unit = max(root_g_cache_size, dm_cache_size);
    int counts = (shm_size - cart_idx_size*4 - ijprim*8*7) / (unit*8);
    int n_tiles = ntiles_i * ntiles_j * ntiles_k * ntiles_l;
    int THREADS = 256;
    int gout_stride = min(n_tiles, gout_stride_max);
    int nsq_per_block = min(counts, THREADS / gout_stride);
    if (nsq_per_block > 8) {
        nsq_per_block = nsq_per_block / 8 * 8;
    }
    threads.x = nsq_per_block;
    threads.y = gout_stride;
    int buflen = nsq_per_block * unit*8 + cart_idx_size*4 + ijprim*8*7;
    return buflen;
}

extern "C" {
int MYK_build_k_general_init(int shm_size)
{
    cudaFuncSetAttribute(myk_k_general,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA shm size %d: %s\n", shm_size,
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int MYK_build_k_general(double *vk, double *dm, int n_dm, int nao,
                        RysIntEnvVars *envs, int *shls_slice, int shm_size,
                        int npairs_ij, int npairs_kl,
                        uint32_t *pair_ij_mapping, uint32_t *pair_kl_mapping,
                        float *q_cond, float *dm_cond, float cutoff,
                        int *pool, int *head, int queue_depth, int n_blocks,
                        double inv_om2, int *atm, int natm, int *bas, int nbas,
                        double *env)
{
    int ish0 = shls_slice[0];
    int jsh0 = shls_slice[2];
    int ksh0 = shls_slice[4];
    int lsh0 = shls_slice[6];
    int li = bas[ANG_OF + ish0*BAS_SLOTS];
    int lj = bas[ANG_OF + jsh0*BAS_SLOTS];
    int lk = bas[ANG_OF + ksh0*BAS_SLOTS];
    int ll = bas[ANG_OF + lsh0*BAS_SLOTS];
    int iprim = bas[NPRIM_OF + ish0*BAS_SLOTS];
    int jprim = bas[NPRIM_OF + jsh0*BAS_SLOTS];
    int kprim = bas[NPRIM_OF + ksh0*BAS_SLOTS];
    int lprim = bas[NPRIM_OF + lsh0*BAS_SLOTS];
    int nfi = (li+1)*(li+2)/2;
    int nfj = (lj+1)*(lj+2)/2;
    int nfk = (lk+1)*(lk+2)/2;
    int nfl = (ll+1)*(ll+2)/2;
    int ntiles_i = (nfi + 2) / 3;
    int ntiles_j = (nfj + 2) / 3;
    int ntiles_k = (nfk + 2) / 3;
    int ntiles_l = (nfl + 2) / 3;
    int order = li + lj + lk + ll;
    int nroots = order / 2 + 1;
    int stride_j = li + 1;
    int stride_k = stride_j * (lj + 1);
    int stride_l = stride_k * (lk + 1);
    int g_size = stride_l * (ll + 1);
    BoundsInfo bounds = {li, lj, lk, ll, nfi, nfj, nfk, nfl,
        nroots, stride_j, stride_k, stride_l, g_size,
        iprim, jprim, kprim, lprim,
        npairs_ij, npairs_kl, pair_ij_mapping, pair_kl_mapping,
        q_cond, NULL, dm_cond, cutoff,
        ntiles_i, ntiles_j, ntiles_k, ntiles_l};
    JKMatrix kmat = {NULL, vk, dm, n_dm, 0, 0., 1., 0.};

    GXYZOffset* p_gxyz_offset = myk_make_gxyz_offset(bounds);
    int gout_pattern = (((li == 0) >> 3) |
                        ((lj == 0) >> 2) |
                        ((lk == 0) >> 1) |
                        ( ll == 0));
    int cart_idx_size = (ntiles_i+ntiles_j+ntiles_k+ntiles_l)*9;
    int n_tiles = ntiles_i * ntiles_j * ntiles_k * ntiles_l;
    for (int t0 = 0; t0 < n_tiles; t0 += 256) {
        dim3 threads;
        int buflen = myk_threads_scheme(threads, bounds, shm_size,
                                        min(256, n_tiles - t0));
        if (threads.x < 1) {
            // the shared-memory scheme does not fit; the caller falls back
            return 2;
        }
        int reserved_shm_size = (buflen - cart_idx_size*4)/8;
        cudaMemset(head, 0, sizeof(int));
        myk_k_general<<<n_blocks, threads, buflen>>>(
            *envs, kmat, bounds, pool, p_gxyz_offset + t0,
            gout_pattern, reserved_shm_size, head, queue_depth, inv_om2);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in MYK_build_k_general: %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
'''

HEAD = '''// Generated by gen_kgeneral.py from GPU4PySCF's gvhf-rys/rys_contract_k.cu.
// Do not edit; edit the generator.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/create_tasks.cu"
#include "gvhf-rys/rys_contract_k.cuh"

#define GOUT_WIDTH1     81

// GPU4PySCF's own table, copied because rys_contract_k.cu is not linked in
static GXYZOffset *myk_make_gxyz_offset(BoundsInfo &bounds)
{
    GXYZOffset goff[625];
    int nf = 0;
    for (int i = 0; i < bounds.nfi; i += 3) {
    for (int j = 0; j < bounds.nfj; j += 3) {
    for (int k = 0; k < bounds.nfk; k += 3) {
    for (int l = 0; l < bounds.nfl; l += 3) {
        goff[nf].ioff = i;
        goff[nf].joff = j;
        goff[nf].koff = k;
        goff[nf].loff = l;
        ++nf;
    } } } }
    for (int n = nf; n < 256; n += nf) {
        for (int m = 0; m < nf; ++m) {
            goff[n+m] = goff[m];
        }
    }
    cudaMemcpyToSymbol(c_gxyz_offset, goff, max(nf, 256)*sizeof(GXYZOffset),
                       0, cudaMemcpyHostToDevice);
    GXYZOffset *p_gxyz_offset;
    cudaGetSymbolAddress((void**)&p_gxyz_offset, c_gxyz_offset);
    return p_gxyz_offset;
}

'''

if __name__ == '__main__':
    src = open(sys.argv[1]).read()
    print(HEAD + transform(_extract(src, 'rys_k_kernel')) + HOST)
