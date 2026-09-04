"""
Generate fastk's per-class CUDA kernels.

The *integral* arithmetic (the Rys 2D recurrences and the density contraction)
is taken verbatim from GPU4PySCF's generated file gvhf-rys/unrolled_rys_k.cu, so
the two codes evaluate literally the same expressions in the same order.  What
this generator changes is the scaffolding around that arithmetic:

  * Rys roots/weights are held in registers, not shared memory, which lets us
    drop the __syncthreads() GPU4PySCF executes once per primitive quartet.
  * the bra-side primitive data (aij, 1/aij, the product centre, cicj) is
    block-uniform, so it is computed once per bra shell pair into shared memory
    instead of once per thread per primitive quartet.
  * the geometric prefactor uses one division and one rsqrt instead of three
    divisions and a sqrt.

Run:  python gen_kernels.py <path-to-unrolled_rys_k.cu>  > fastk_generated.cu
"""
import re, sys, os, json

# (li,lj,lk,ll) -> generated kernel name suffix; only classes GPU4PySCF unrolls
# with gout_stride == 1 are handled here.
GOUT_STRIDE_NOT_1 = {(2,0,2,1), (2,1,1,1), (2,1,2,0), (2,2,0,1), (2,2,1,0),
                     (3,0,1,1), (3,1,1,0)}


def parse_kernel(src, name):
    """Return (gout_names, irys_body, contraction_body) for rys_k_<name>."""
    m = re.search(r'\nvoid ' + name + r'\(RysIntEnvVars.*?\n\}\n', src, re.S)
    if m is None:
        return None
    body = m.group(0)
    gouts = re.findall(r'^\s*double (gout\d+);\s*$', body, re.M)
    if not gouts:
        return None
    assert body.count('for (int irys = 0; irys < nroots; ++irys) {') == 1, name
    assert 'gout_id' not in body, name
    # innermost root loop
    i = body.index('for (int irys = 0; irys < nroots; ++irys) {')
    j = body.index('{', i)
    depth, k = 1, j + 1
    while depth:
        if body[k] == '{': depth += 1
        elif body[k] == '}': depth -= 1
        k += 1
    irys_body = body[j+1:k-1]
    # density contraction: the last `if (task_id < ntasks) {` block
    i = body.rindex('if (task_id < ntasks) {')
    j = body.index('{', i)
    depth, k = 1, j + 1
    while depth:
        if body[k] == '{': depth += 1
        elif body[k] == '}': depth -= 1
        k += 1
    contract = body[j+1:k-1]
    # strip GPU4PySCF's own scaffolding out of the contraction block
    contract = re.sub(r'int \*ao_loc = envs\.ao_loc;\s*', '', contract)
    contract = re.sub(r'int nao = ao_loc\[nbas\];\s*', '', contract)
    contract = re.sub(r'int [ijkl]0 = ao_loc\[[ijkl]sh\];\s*', '', contract)
    contract = re.sub(r'for \(int i_dm = 0; i_dm < kmat\.n_dm; \+\+i_dm\) \{\s*', '', contract)
    contract = re.sub(r'double \*dm = kmat\.dm \+ i_dm \* nao \* nao;\s*', '', contract)
    contract = re.sub(r'double \*vk = kmat\.vk \+ i_dm \* nao \* nao;\s*', '', contract)
    contract = contract.rstrip()
    assert contract.endswith('}'), 'unexpected contraction tail'
    contract = contract[:-1]          # drop the i_dm loop's closing brace
    # roots/weights now come from registers
    irys_body = irys_body.replace('rw[(2*irys+1)*nsq_per_block]', 'rw[2*irys+1]')
    irys_body = irys_body.replace('rw[ 2*irys   *nsq_per_block]', 'rw[2*irys]')
    assert 'nsq_per_block' not in irys_body, 'unhandled shared-memory reference'
    # Double-precision division is ~20 instructions on sm_80 and GPU4PySCF does
    # one per root for rt_aa and one more for b10/b01.  1/(aij+akl), 1/aij and
    # 1/akl are all available as reciprocals we already had to form, so these
    # become multiplies.  (rsqrt(s)^2 differs from 1/s by ~2 ulp.)
    n_div = irys_body.count('rt / (aij + akl)') + irys_body.count('.5/aij') \
          + irys_body.count('.5/akl')
    irys_body = irys_body.replace('rt / (aij + akl)', 'rt * inv_s2')
    irys_body = irys_body.replace('.5/aij', '.5*iaij')
    irys_body = irys_body.replace('.5/akl', '.5*iakl')
    assert '/' not in re.sub(r'//[^\n]*', '', irys_body), \
        'unexpected division left in the root loop'
    return gouts, irys_body, contract


# threads per block / minimum blocks per SM, per class.  More gout registers
# means fewer resident blocks, so the wide classes get smaller launch bounds.
# Threads per block and the minimum blocks per SM we ask the compiler to fit.
# The pair (NT, MB) caps registers at 65536/(NT*MB), so wide gout classes need a
# smaller product.  Overridable per class from the command line for tuning.
# Measured on an A100-SXM4-40GB (sm_80).  The pair (NT, MB) caps registers at
# 65536/(NT*MB); the right cap is set by how many gout accumulators the class
# carries, which is a property of the angular-momentum class alone -- not of the
# molecule -- so this table transfers between systems.  See bench/sweep.sh.
LAUNCH = {
    '1000': (256, 3), '1010': (128, 4), '1100': (256, 3), '2000': (256, 3),
    '1011': (128, 4), '1110': (128, 4), '2010': (128, 4), '2100': (256, 2),
    '2110': (128, 2), '1111': ( 64, 4), '2011': ( 64, 4), '2020': (128, 2),
    '2200': (256, 2),
}

def default_launch(ngout):
    if ngout <= 10: return (256, 3)     # ~85 registers
    if ngout <= 20: return (128, 4)     # ~128 registers
    if ngout <= 40: return (256, 2)     # ~128 registers
    return (128, 2)                     # ~256 registers

TEMPLATE = r'''
#define NTHREADS_%(tag)s  %(nthreads)d
#define MINBLOCKS_%(tag)s %(minblocks)d
template <int NRANGE> __device__ __forceinline__ void
kbody_%(tag)s(KARGS)
{
    constexpr int NROOTS = %(nroots)d;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int *bas_kl_idx = pool + blockIdx.x * queue_depth;

    __shared__ int ntasks, pair_ij;
    __shared__ double s_cicj[MAX_PRIM_PAIR];
    __shared__ double s_aij [MAX_PRIM_PAIR];
    __shared__ double s_iaij[MAX_PRIM_PAIR];
    __shared__ double s_ajaij[MAX_PRIM_PAIR];
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
        int jsh = bas_ij %% nbas;
        int iprim = bas[ish*BAS_SLOTS+NPRIM_OF];
        int jprim = bas[jsh*BAS_SLOTS+NPRIM_OF];
        const double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        const double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        const double *ci   = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        const double *cj   = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        const double *ri_  = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        const double *rj_  = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double rjri[3] = {rj_[0]-ri_[0], rj_[1]-ri_[1], rj_[2]-ri_[2]};
        double rr_ij = rjri[0]*rjri[0] + rjri[1]*rjri[1] + rjri[2]*rjri[2];
        int nprim_ij = iprim*jprim;
        for (int ij = tid; ij < nprim_ij; ij += nthreads) {
            double ai = expi[ij/jprim], aj = expj[ij%%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%%jprim] * exp(-ai*aj_aij*rr_ij);
            s_xij  [ij] = ri_[0] + rjri[0]*aj_aij;
            s_yij  [ij] = ri_[1] + rjri[1]*aj_aij;
            s_zij  [ij] = ri_[2] + rjri[2]*aj_aij;
        }
        __syncthreads();

        int i0 = ao_loc[ish], j0 = ao_loc[jsh];
        double sym_ij = (ish == jsh) ? .5*PI_FAC : PI_FAC;

        for (int task_id = tid; task_id < ntasks; task_id += nthreads) {
            int bas_kl = bas_kl_idx[task_id];
            int ksh = bas_kl / nbas;
            int lsh = bas_kl %% nbas;
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
            double rw[2*NROOTS];
%(gout_decl)s
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
                double al_akl = al * iakl;
                double ckcl = fac_sym * ck[klp/lprim] * cl[klp%%lprim]
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
                    double iaij = s_iaij[ijp];
                    double aj_aij = s_ajaij[ijp];
                    double t = aij * akl;
                    double s = aij + akl;
                    // Range separation.  inv_om2 = 0 gives the full-range
                    // operator; inv_om2 = 1/omega^2 gives the long-range (erf)
                    // one, because scaling the Rys argument and the root by
                    // theta_fac = w^2/(w^2+theta), and the weight by
                    // sqrt(theta_fac), is exactly rsqrt(s) -> rsqrt(s+t/w^2)
                    // in the three places inv_s and inv_s2 are used.
                    // NRANGE == 2 accumulates coef0*(full range) +
                    // coef1*(long range) into the same gout registers, so
                    // the geometry above and the density contraction below --
                    // and its atomicAdds -- are paid once, not twice.
                    for (int h = 0; h < NRANGE; ++h) {
                    double inv_s  = rsqrt(fma(t, NRANGE == 1 ? inv_om2
                                                : (h == 0 ? 0. : inv_om2), s));
                    double inv_s2 = inv_s * inv_s;
                    // fac = cicj*ckcl/(aij*akl*sqrt(aij+akl)), no division
                    double fac = s_cicj[ijp] * ckcl * iaij * iakl * inv_s;
                    if (NRANGE > 1) fac *= (h == 0 ? coef0 : coef1);
                    rys_roots_reg<NROOTS>(t * rr * inv_s2, rw, tab);
#pragma unroll
                    for (int irys = 0; irys < NROOTS; ++irys) {
%(irys_body)s
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];
%(contract)s
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_%(tag)s, MINBLOCKS_%(tag)s)
k_%(tag)s(KARGS)    { kbody_%(tag)s<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_%(tag)s, MINBLOCKS_%(tag)s)
k_rs_%(tag)s(KARGS) { kbody_%(tag)s<2>(KFWD); }
'''


def generate(path, classes):
    src = open(path).read()
    out = []
    launch_table = {'0000': (256, 4)}   # the hand-written (ss|ss) kernel
    for (li, lj, lk, ll) in classes:
        tag = f'{li}{lj}{lk}{ll}'
        parsed = parse_kernel(src, 'rys_k_' + tag)
        if parsed is None:
            sys.stderr.write(f'skip {tag}: not unrolled in gpu4pyscf\n')
            continue
        gouts, irys_body, contract = parsed
        nroots = (li + lj + lk + ll) // 2 + 1
        decl = '            ' + ' '.join(f'double {g} = 0.;' for g in gouts)
        nth, mb = LAUNCH.get(tag, default_launch(len(gouts)))
        out.append(TEMPLATE % dict(tag=tag, nroots=nroots, gout_decl=decl,
                                   irys_body=irys_body, contract=contract,
                                   nthreads=nth, minblocks=mb))
        launch_table[tag] = (nth, mb)
        sys.stderr.write(f'generated k_{tag}: {len(gouts)} gout, nroots={nroots}, '
                         f'launch_bounds({nth},{mb})\n')
    _tab = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fastk_launch.json')
    with open(_tab + '.tmp', 'w') as f:
        json.dump(launch_table, f, indent=1, sort_keys=True)
    os.replace(_tab + '.tmp', _tab)
    return '\n'.join(out)


if __name__ == '__main__':
    path = sys.argv[1]
    classes = [tuple(int(c) for c in t) for t in sys.argv[2].split(',')]
    for spec in sys.argv[3:]:               # e.g. 1111:128:4
        tag, nt, mb = spec.split(':')
        LAUNCH[tag] = (int(nt), int(mb))
    print(generate(path, classes))
