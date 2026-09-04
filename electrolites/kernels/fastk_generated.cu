
#define NTHREADS_1000  256
#define MINBLOCKS_1000 3
template <int NRANGE> __device__ __forceinline__ void
kbody_1000(KARGS)
{
    constexpr int NROOTS = 1;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    gout0 += trr_10x * fac * wt;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += 1 * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += 1 * fac * trr_10z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_1000, MINBLOCKS_1000)
k_1000(KARGS)    { kbody_1000<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_1000, MINBLOCKS_1000)
k_rs_1000(KARGS) { kbody_1000<2>(KFWD); }


#define NTHREADS_1010  128
#define MINBLOCKS_1010 4
template <int NRANGE> __device__ __forceinline__ void
kbody_1010(KARGS)
{
    constexpr int NROOTS = 2;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.; double gout6 = 0.; double gout7 = 0.; double gout8 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double b00 = .5 * rt_aa;
                    double rt_akl = rt_aa * aij;
                    double cpx = xlxk*al_akl + xpq*rt_akl;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    gout0 += trr_11x * fac * wt;
                    double trr_01x = cpx * 1;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += trr_01x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += trr_01x * fac * trr_10z;
                    double cpy = ylyk*al_akl + ypq*rt_akl;
                    double trr_01y = cpy * fac;
                    gout3 += trr_10x * trr_01y * wt;
                    double trr_11y = cpy * trr_10y + 1*b00 * fac;
                    gout4 += 1 * trr_11y * wt;
                    gout5 += 1 * trr_01y * trr_10z;
                    double cpz = zlzk*al_akl + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    gout6 += trr_10x * fac * trr_01z;
                    gout7 += 1 * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout8 += 1 * fac * trr_11z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                val += gout3 * dm[(j0+0)*nao+(k0+1)];
                val += gout6 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                val += gout4 * dm[(j0+0)*nao+(k0+1)];
                val += gout7 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                val += gout5 * dm[(j0+0)*nao+(k0+1)];
                val += gout8 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout3 * dm[(i0+0)*nao+(k0+1)];
                val += gout6 * dm[(i0+0)*nao+(k0+2)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout4 * dm[(i0+1)*nao+(k0+1)];
                val += gout7 * dm[(i0+1)*nao+(k0+2)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                val += gout5 * dm[(i0+2)*nao+(k0+1)];
                val += gout8 * dm[(i0+2)*nao+(k0+2)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                val = 0;
                val += gout6 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                val = 0;
                val += gout7 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                val = 0;
                val += gout8 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                val = 0;
                val += gout3 * dm[(i0+0)*nao+(l0+0)];
                val += gout4 * dm[(i0+1)*nao+(l0+0)];
                val += gout5 * dm[(i0+2)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(l0+0)];
                val += gout7 * dm[(i0+1)*nao+(l0+0)];
                val += gout8 * dm[(i0+2)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+2), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_1010, MINBLOCKS_1010)
k_1010(KARGS)    { kbody_1010<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_1010, MINBLOCKS_1010)
k_rs_1010(KARGS) { kbody_1010<2>(KFWD); }


#define NTHREADS_1100  256
#define MINBLOCKS_1100 3
template <int NRANGE> __device__ __forceinline__ void
kbody_1100(KARGS)
{
    constexpr int NROOTS = 2;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.; double gout6 = 0.; double gout7 = 0.; double gout8 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double hrr_1100x = trr_20x - xjxi * trr_10x;
                    gout0 += hrr_1100x * fac * wt;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += hrr_0100x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += hrr_0100x * fac * trr_10z;
                    double hrr_0100y = trr_10y - yjyi * fac;
                    gout3 += trr_10x * hrr_0100y * wt;
                    double trr_20y = c0y * trr_10y + 1*b10 * fac;
                    double hrr_1100y = trr_20y - yjyi * trr_10y;
                    gout4 += 1 * hrr_1100y * wt;
                    gout5 += 1 * hrr_0100y * trr_10z;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    gout6 += trr_10x * fac * hrr_0100z;
                    gout7 += 1 * trr_10y * hrr_0100z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    double hrr_1100z = trr_20z - zjzi * trr_10z;
                    gout8 += 1 * fac * hrr_1100z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                val += gout3 * dm[(j0+1)*nao+(k0+0)];
                val += gout6 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                val += gout4 * dm[(j0+1)*nao+(k0+0)];
                val += gout7 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                val += gout5 * dm[(j0+1)*nao+(k0+0)];
                val += gout8 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout3 * dm[(i0+0)*nao+(k0+0)];
                val += gout4 * dm[(i0+1)*nao+(k0+0)];
                val += gout5 * dm[(i0+2)*nao+(k0+0)];
                atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(k0+0)];
                val += gout7 * dm[(i0+1)*nao+(k0+0)];
                val += gout8 * dm[(i0+2)*nao+(k0+0)];
                atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                val += gout3 * dm[(j0+1)*nao+(l0+0)];
                val += gout6 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                val += gout4 * dm[(j0+1)*nao+(l0+0)];
                val += gout7 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                val += gout5 * dm[(j0+1)*nao+(l0+0)];
                val += gout8 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                val = 0;
                val += gout3 * dm[(i0+0)*nao+(l0+0)];
                val += gout4 * dm[(i0+1)*nao+(l0+0)];
                val += gout5 * dm[(i0+2)*nao+(l0+0)];
                atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(l0+0)];
                val += gout7 * dm[(i0+1)*nao+(l0+0)];
                val += gout8 * dm[(i0+2)*nao+(l0+0)];
                atomicAdd(vk+(j0+2)*nao+(k0+0), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_1100, MINBLOCKS_1100)
k_1100(KARGS)    { kbody_1100<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_1100, MINBLOCKS_1100)
k_rs_1100(KARGS) { kbody_1100<2>(KFWD); }


#define NTHREADS_2000  256
#define MINBLOCKS_2000 3
template <int NRANGE> __device__ __forceinline__ void
kbody_2000(KARGS)
{
    constexpr int NROOTS = 2;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    gout0 += trr_20x * fac * wt;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += trr_10x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += trr_10x * fac * trr_10z;
                    double trr_20y = c0y * trr_10y + 1*b10 * fac;
                    gout3 += 1 * trr_20y * wt;
                    gout4 += 1 * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += 1 * fac * trr_20z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                val += gout3 * dm[(i0+3)*nao+(k0+0)];
                val += gout4 * dm[(i0+4)*nao+(k0+0)];
                val += gout5 * dm[(i0+5)*nao+(k0+0)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                val += gout3 * dm[(i0+3)*nao+(l0+0)];
                val += gout4 * dm[(i0+4)*nao+(l0+0)];
                val += gout5 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_2000, MINBLOCKS_2000)
k_2000(KARGS)    { kbody_2000<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_2000, MINBLOCKS_2000)
k_rs_2000(KARGS) { kbody_2000<2>(KFWD); }


#define NTHREADS_1011  128
#define MINBLOCKS_1011 4
template <int NRANGE> __device__ __forceinline__ void
kbody_1011(KARGS)
{
    constexpr int NROOTS = 2;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.; double gout6 = 0.; double gout7 = 0.; double gout8 = 0.; double gout9 = 0.; double gout10 = 0.; double gout11 = 0.; double gout12 = 0.; double gout13 = 0.; double gout14 = 0.; double gout15 = 0.; double gout16 = 0.; double gout17 = 0.; double gout18 = 0.; double gout19 = 0.; double gout20 = 0.; double gout21 = 0.; double gout22 = 0.; double gout23 = 0.; double gout24 = 0.; double gout25 = 0.; double gout26 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double b00 = .5 * rt_aa;
                    double rt_akl = rt_aa * aij;
                    double b01 = .5*iakl * (1 - rt_akl);
                    double cpx = xlxk*al_akl + xpq*rt_akl;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double trr_01x = cpx * 1;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    double hrr_1011x = trr_12x - xlxk * trr_11x;
                    gout0 += hrr_1011x * fac * wt;
                    double trr_02x = cpx * trr_01x + 1*b01 * 1;
                    double hrr_0011x = trr_02x - xlxk * trr_01x;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += hrr_0011x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += hrr_0011x * fac * trr_10z;
                    double hrr_1001x = trr_11x - xlxk * trr_10x;
                    double cpy = ylyk*al_akl + ypq*rt_akl;
                    double trr_01y = cpy * fac;
                    gout3 += hrr_1001x * trr_01y * wt;
                    double hrr_0001x = trr_01x - xlxk * 1;
                    double trr_11y = cpy * trr_10y + 1*b00 * fac;
                    gout4 += hrr_0001x * trr_11y * wt;
                    gout5 += hrr_0001x * trr_01y * trr_10z;
                    double cpz = zlzk*al_akl + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    gout6 += hrr_1001x * fac * trr_01z;
                    gout7 += hrr_0001x * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout8 += hrr_0001x * fac * trr_11z;
                    double hrr_0001y = trr_01y - ylyk * fac;
                    gout9 += trr_11x * hrr_0001y * wt;
                    double hrr_1001y = trr_11y - ylyk * trr_10y;
                    gout10 += trr_01x * hrr_1001y * wt;
                    gout11 += trr_01x * hrr_0001y * trr_10z;
                    double trr_02y = cpy * trr_01y + 1*b01 * fac;
                    double hrr_0011y = trr_02y - ylyk * trr_01y;
                    gout12 += trr_10x * hrr_0011y * wt;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    double hrr_1011y = trr_12y - ylyk * trr_11y;
                    gout13 += 1 * hrr_1011y * wt;
                    gout14 += 1 * hrr_0011y * trr_10z;
                    gout15 += trr_10x * hrr_0001y * trr_01z;
                    gout16 += 1 * hrr_1001y * trr_01z;
                    gout17 += 1 * hrr_0001y * trr_11z;
                    double hrr_0001z = trr_01z - zlzk * wt;
                    gout18 += trr_11x * fac * hrr_0001z;
                    gout19 += trr_01x * trr_10y * hrr_0001z;
                    double hrr_1001z = trr_11z - zlzk * trr_10z;
                    gout20 += trr_01x * fac * hrr_1001z;
                    gout21 += trr_10x * trr_01y * hrr_0001z;
                    gout22 += 1 * trr_11y * hrr_0001z;
                    gout23 += 1 * trr_01y * hrr_1001z;
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    double hrr_0011z = trr_02z - zlzk * trr_01z;
                    gout24 += trr_10x * fac * hrr_0011z;
                    gout25 += 1 * trr_10y * hrr_0011z;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    double hrr_1011z = trr_12z - zlzk * trr_11z;
                    gout26 += 1 * fac * hrr_1011z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                val += gout3 * dm[(j0+0)*nao+(k0+1)];
                val += gout6 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout9 * dm[(j0+0)*nao+(k0+0)];
                val += gout12 * dm[(j0+0)*nao+(k0+1)];
                val += gout15 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+0)*nao+(l0+1), val);
                val = 0;
                val += gout18 * dm[(j0+0)*nao+(k0+0)];
                val += gout21 * dm[(j0+0)*nao+(k0+1)];
                val += gout24 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+0)*nao+(l0+2), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                val += gout4 * dm[(j0+0)*nao+(k0+1)];
                val += gout7 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout10 * dm[(j0+0)*nao+(k0+0)];
                val += gout13 * dm[(j0+0)*nao+(k0+1)];
                val += gout16 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+1)*nao+(l0+1), val);
                val = 0;
                val += gout19 * dm[(j0+0)*nao+(k0+0)];
                val += gout22 * dm[(j0+0)*nao+(k0+1)];
                val += gout25 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+1)*nao+(l0+2), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                val += gout5 * dm[(j0+0)*nao+(k0+1)];
                val += gout8 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout11 * dm[(j0+0)*nao+(k0+0)];
                val += gout14 * dm[(j0+0)*nao+(k0+1)];
                val += gout17 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+2)*nao+(l0+1), val);
                val = 0;
                val += gout20 * dm[(j0+0)*nao+(k0+0)];
                val += gout23 * dm[(j0+0)*nao+(k0+1)];
                val += gout26 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+2)*nao+(l0+2), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout3 * dm[(i0+0)*nao+(k0+1)];
                val += gout6 * dm[(i0+0)*nao+(k0+2)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout4 * dm[(i0+1)*nao+(k0+1)];
                val += gout7 * dm[(i0+1)*nao+(k0+2)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                val += gout5 * dm[(i0+2)*nao+(k0+1)];
                val += gout8 * dm[(i0+2)*nao+(k0+2)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout9 * dm[(i0+0)*nao+(k0+0)];
                val += gout12 * dm[(i0+0)*nao+(k0+1)];
                val += gout15 * dm[(i0+0)*nao+(k0+2)];
                val += gout10 * dm[(i0+1)*nao+(k0+0)];
                val += gout13 * dm[(i0+1)*nao+(k0+1)];
                val += gout16 * dm[(i0+1)*nao+(k0+2)];
                val += gout11 * dm[(i0+2)*nao+(k0+0)];
                val += gout14 * dm[(i0+2)*nao+(k0+1)];
                val += gout17 * dm[(i0+2)*nao+(k0+2)];
                atomicAdd(vk+(j0+0)*nao+(l0+1), val);
                val = 0;
                val += gout18 * dm[(i0+0)*nao+(k0+0)];
                val += gout21 * dm[(i0+0)*nao+(k0+1)];
                val += gout24 * dm[(i0+0)*nao+(k0+2)];
                val += gout19 * dm[(i0+1)*nao+(k0+0)];
                val += gout22 * dm[(i0+1)*nao+(k0+1)];
                val += gout25 * dm[(i0+1)*nao+(k0+2)];
                val += gout20 * dm[(i0+2)*nao+(k0+0)];
                val += gout23 * dm[(i0+2)*nao+(k0+1)];
                val += gout26 * dm[(i0+2)*nao+(k0+2)];
                atomicAdd(vk+(j0+0)*nao+(l0+2), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                val += gout9 * dm[(j0+0)*nao+(l0+1)];
                val += gout18 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(l0+0)];
                val += gout12 * dm[(j0+0)*nao+(l0+1)];
                val += gout21 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                val = 0;
                val += gout6 * dm[(j0+0)*nao+(l0+0)];
                val += gout15 * dm[(j0+0)*nao+(l0+1)];
                val += gout24 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                val += gout10 * dm[(j0+0)*nao+(l0+1)];
                val += gout19 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(l0+0)];
                val += gout13 * dm[(j0+0)*nao+(l0+1)];
                val += gout22 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                val = 0;
                val += gout7 * dm[(j0+0)*nao+(l0+0)];
                val += gout16 * dm[(j0+0)*nao+(l0+1)];
                val += gout25 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                val += gout11 * dm[(j0+0)*nao+(l0+1)];
                val += gout20 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(l0+0)];
                val += gout14 * dm[(j0+0)*nao+(l0+1)];
                val += gout23 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                val = 0;
                val += gout8 * dm[(j0+0)*nao+(l0+0)];
                val += gout17 * dm[(j0+0)*nao+(l0+1)];
                val += gout26 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout9 * dm[(i0+0)*nao+(l0+1)];
                val += gout18 * dm[(i0+0)*nao+(l0+2)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout10 * dm[(i0+1)*nao+(l0+1)];
                val += gout19 * dm[(i0+1)*nao+(l0+2)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                val += gout11 * dm[(i0+2)*nao+(l0+1)];
                val += gout20 * dm[(i0+2)*nao+(l0+2)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                val = 0;
                val += gout3 * dm[(i0+0)*nao+(l0+0)];
                val += gout12 * dm[(i0+0)*nao+(l0+1)];
                val += gout21 * dm[(i0+0)*nao+(l0+2)];
                val += gout4 * dm[(i0+1)*nao+(l0+0)];
                val += gout13 * dm[(i0+1)*nao+(l0+1)];
                val += gout22 * dm[(i0+1)*nao+(l0+2)];
                val += gout5 * dm[(i0+2)*nao+(l0+0)];
                val += gout14 * dm[(i0+2)*nao+(l0+1)];
                val += gout23 * dm[(i0+2)*nao+(l0+2)];
                atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(l0+0)];
                val += gout15 * dm[(i0+0)*nao+(l0+1)];
                val += gout24 * dm[(i0+0)*nao+(l0+2)];
                val += gout7 * dm[(i0+1)*nao+(l0+0)];
                val += gout16 * dm[(i0+1)*nao+(l0+1)];
                val += gout25 * dm[(i0+1)*nao+(l0+2)];
                val += gout8 * dm[(i0+2)*nao+(l0+0)];
                val += gout17 * dm[(i0+2)*nao+(l0+1)];
                val += gout26 * dm[(i0+2)*nao+(l0+2)];
                atomicAdd(vk+(j0+0)*nao+(k0+2), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_1011, MINBLOCKS_1011)
k_1011(KARGS)    { kbody_1011<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_1011, MINBLOCKS_1011)
k_rs_1011(KARGS) { kbody_1011<2>(KFWD); }


#define NTHREADS_1110  128
#define MINBLOCKS_1110 4
template <int NRANGE> __device__ __forceinline__ void
kbody_1110(KARGS)
{
    constexpr int NROOTS = 2;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.; double gout6 = 0.; double gout7 = 0.; double gout8 = 0.; double gout9 = 0.; double gout10 = 0.; double gout11 = 0.; double gout12 = 0.; double gout13 = 0.; double gout14 = 0.; double gout15 = 0.; double gout16 = 0.; double gout17 = 0.; double gout18 = 0.; double gout19 = 0.; double gout20 = 0.; double gout21 = 0.; double gout22 = 0.; double gout23 = 0.; double gout24 = 0.; double gout25 = 0.; double gout26 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double b00 = .5 * rt_aa;
                    double rt_akl = rt_aa * aij;
                    double cpx = xlxk*al_akl + xpq*rt_akl;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double hrr_1110x = trr_21x - xjxi * trr_11x;
                    gout0 += hrr_1110x * fac * wt;
                    double trr_01x = cpx * 1;
                    double hrr_0110x = trr_11x - xjxi * trr_01x;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += hrr_0110x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += hrr_0110x * fac * trr_10z;
                    double hrr_0100y = trr_10y - yjyi * fac;
                    gout3 += trr_11x * hrr_0100y * wt;
                    double trr_20y = c0y * trr_10y + 1*b10 * fac;
                    double hrr_1100y = trr_20y - yjyi * trr_10y;
                    gout4 += trr_01x * hrr_1100y * wt;
                    gout5 += trr_01x * hrr_0100y * trr_10z;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    gout6 += trr_11x * fac * hrr_0100z;
                    gout7 += trr_01x * trr_10y * hrr_0100z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    double hrr_1100z = trr_20z - zjzi * trr_10z;
                    gout8 += trr_01x * fac * hrr_1100z;
                    double hrr_1100x = trr_20x - xjxi * trr_10x;
                    double cpy = ylyk*al_akl + ypq*rt_akl;
                    double trr_01y = cpy * fac;
                    gout9 += hrr_1100x * trr_01y * wt;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    double trr_11y = cpy * trr_10y + 1*b00 * fac;
                    gout10 += hrr_0100x * trr_11y * wt;
                    gout11 += hrr_0100x * trr_01y * trr_10z;
                    double hrr_0110y = trr_11y - yjyi * trr_01y;
                    gout12 += trr_10x * hrr_0110y * wt;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    double hrr_1110y = trr_21y - yjyi * trr_11y;
                    gout13 += 1 * hrr_1110y * wt;
                    gout14 += 1 * hrr_0110y * trr_10z;
                    gout15 += trr_10x * trr_01y * hrr_0100z;
                    gout16 += 1 * trr_11y * hrr_0100z;
                    gout17 += 1 * trr_01y * hrr_1100z;
                    double cpz = zlzk*al_akl + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    gout18 += hrr_1100x * fac * trr_01z;
                    gout19 += hrr_0100x * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout20 += hrr_0100x * fac * trr_11z;
                    gout21 += trr_10x * hrr_0100y * trr_01z;
                    gout22 += 1 * hrr_1100y * trr_01z;
                    gout23 += 1 * hrr_0100y * trr_11z;
                    double hrr_0110z = trr_11z - zjzi * trr_01z;
                    gout24 += trr_10x * fac * hrr_0110z;
                    gout25 += 1 * trr_10y * hrr_0110z;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    double hrr_1110z = trr_21z - zjzi * trr_11z;
                    gout26 += 1 * fac * hrr_1110z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                val += gout9 * dm[(j0+0)*nao+(k0+1)];
                val += gout18 * dm[(j0+0)*nao+(k0+2)];
                val += gout3 * dm[(j0+1)*nao+(k0+0)];
                val += gout12 * dm[(j0+1)*nao+(k0+1)];
                val += gout21 * dm[(j0+1)*nao+(k0+2)];
                val += gout6 * dm[(j0+2)*nao+(k0+0)];
                val += gout15 * dm[(j0+2)*nao+(k0+1)];
                val += gout24 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                val += gout10 * dm[(j0+0)*nao+(k0+1)];
                val += gout19 * dm[(j0+0)*nao+(k0+2)];
                val += gout4 * dm[(j0+1)*nao+(k0+0)];
                val += gout13 * dm[(j0+1)*nao+(k0+1)];
                val += gout22 * dm[(j0+1)*nao+(k0+2)];
                val += gout7 * dm[(j0+2)*nao+(k0+0)];
                val += gout16 * dm[(j0+2)*nao+(k0+1)];
                val += gout25 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                val += gout11 * dm[(j0+0)*nao+(k0+1)];
                val += gout20 * dm[(j0+0)*nao+(k0+2)];
                val += gout5 * dm[(j0+1)*nao+(k0+0)];
                val += gout14 * dm[(j0+1)*nao+(k0+1)];
                val += gout23 * dm[(j0+1)*nao+(k0+2)];
                val += gout8 * dm[(j0+2)*nao+(k0+0)];
                val += gout17 * dm[(j0+2)*nao+(k0+1)];
                val += gout26 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout9 * dm[(i0+0)*nao+(k0+1)];
                val += gout18 * dm[(i0+0)*nao+(k0+2)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout10 * dm[(i0+1)*nao+(k0+1)];
                val += gout19 * dm[(i0+1)*nao+(k0+2)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                val += gout11 * dm[(i0+2)*nao+(k0+1)];
                val += gout20 * dm[(i0+2)*nao+(k0+2)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout3 * dm[(i0+0)*nao+(k0+0)];
                val += gout12 * dm[(i0+0)*nao+(k0+1)];
                val += gout21 * dm[(i0+0)*nao+(k0+2)];
                val += gout4 * dm[(i0+1)*nao+(k0+0)];
                val += gout13 * dm[(i0+1)*nao+(k0+1)];
                val += gout22 * dm[(i0+1)*nao+(k0+2)];
                val += gout5 * dm[(i0+2)*nao+(k0+0)];
                val += gout14 * dm[(i0+2)*nao+(k0+1)];
                val += gout23 * dm[(i0+2)*nao+(k0+2)];
                atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(k0+0)];
                val += gout15 * dm[(i0+0)*nao+(k0+1)];
                val += gout24 * dm[(i0+0)*nao+(k0+2)];
                val += gout7 * dm[(i0+1)*nao+(k0+0)];
                val += gout16 * dm[(i0+1)*nao+(k0+1)];
                val += gout25 * dm[(i0+1)*nao+(k0+2)];
                val += gout8 * dm[(i0+2)*nao+(k0+0)];
                val += gout17 * dm[(i0+2)*nao+(k0+1)];
                val += gout26 * dm[(i0+2)*nao+(k0+2)];
                atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                val += gout3 * dm[(j0+1)*nao+(l0+0)];
                val += gout6 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout9 * dm[(j0+0)*nao+(l0+0)];
                val += gout12 * dm[(j0+1)*nao+(l0+0)];
                val += gout15 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                val = 0;
                val += gout18 * dm[(j0+0)*nao+(l0+0)];
                val += gout21 * dm[(j0+1)*nao+(l0+0)];
                val += gout24 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                val += gout4 * dm[(j0+1)*nao+(l0+0)];
                val += gout7 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout10 * dm[(j0+0)*nao+(l0+0)];
                val += gout13 * dm[(j0+1)*nao+(l0+0)];
                val += gout16 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                val = 0;
                val += gout19 * dm[(j0+0)*nao+(l0+0)];
                val += gout22 * dm[(j0+1)*nao+(l0+0)];
                val += gout25 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                val += gout5 * dm[(j0+1)*nao+(l0+0)];
                val += gout8 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout11 * dm[(j0+0)*nao+(l0+0)];
                val += gout14 * dm[(j0+1)*nao+(l0+0)];
                val += gout17 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                val = 0;
                val += gout20 * dm[(j0+0)*nao+(l0+0)];
                val += gout23 * dm[(j0+1)*nao+(l0+0)];
                val += gout26 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                val = 0;
                val += gout9 * dm[(i0+0)*nao+(l0+0)];
                val += gout10 * dm[(i0+1)*nao+(l0+0)];
                val += gout11 * dm[(i0+2)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                val = 0;
                val += gout18 * dm[(i0+0)*nao+(l0+0)];
                val += gout19 * dm[(i0+1)*nao+(l0+0)];
                val += gout20 * dm[(i0+2)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                val = 0;
                val += gout3 * dm[(i0+0)*nao+(l0+0)];
                val += gout4 * dm[(i0+1)*nao+(l0+0)];
                val += gout5 * dm[(i0+2)*nao+(l0+0)];
                atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                val = 0;
                val += gout12 * dm[(i0+0)*nao+(l0+0)];
                val += gout13 * dm[(i0+1)*nao+(l0+0)];
                val += gout14 * dm[(i0+2)*nao+(l0+0)];
                atomicAdd(vk+(j0+1)*nao+(k0+1), val);
                val = 0;
                val += gout21 * dm[(i0+0)*nao+(l0+0)];
                val += gout22 * dm[(i0+1)*nao+(l0+0)];
                val += gout23 * dm[(i0+2)*nao+(l0+0)];
                atomicAdd(vk+(j0+1)*nao+(k0+2), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(l0+0)];
                val += gout7 * dm[(i0+1)*nao+(l0+0)];
                val += gout8 * dm[(i0+2)*nao+(l0+0)];
                atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                val = 0;
                val += gout15 * dm[(i0+0)*nao+(l0+0)];
                val += gout16 * dm[(i0+1)*nao+(l0+0)];
                val += gout17 * dm[(i0+2)*nao+(l0+0)];
                atomicAdd(vk+(j0+2)*nao+(k0+1), val);
                val = 0;
                val += gout24 * dm[(i0+0)*nao+(l0+0)];
                val += gout25 * dm[(i0+1)*nao+(l0+0)];
                val += gout26 * dm[(i0+2)*nao+(l0+0)];
                atomicAdd(vk+(j0+2)*nao+(k0+2), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_1110, MINBLOCKS_1110)
k_1110(KARGS)    { kbody_1110<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_1110, MINBLOCKS_1110)
k_rs_1110(KARGS) { kbody_1110<2>(KFWD); }


#define NTHREADS_2010  128
#define MINBLOCKS_2010 4
template <int NRANGE> __device__ __forceinline__ void
kbody_2010(KARGS)
{
    constexpr int NROOTS = 2;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.; double gout6 = 0.; double gout7 = 0.; double gout8 = 0.; double gout9 = 0.; double gout10 = 0.; double gout11 = 0.; double gout12 = 0.; double gout13 = 0.; double gout14 = 0.; double gout15 = 0.; double gout16 = 0.; double gout17 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double b00 = .5 * rt_aa;
                    double rt_akl = rt_aa * aij;
                    double cpx = xlxk*al_akl + xpq*rt_akl;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    gout0 += trr_21x * fac * wt;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += trr_11x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += trr_11x * fac * trr_10z;
                    double trr_01x = cpx * 1;
                    double trr_20y = c0y * trr_10y + 1*b10 * fac;
                    gout3 += trr_01x * trr_20y * wt;
                    gout4 += trr_01x * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += trr_01x * fac * trr_20z;
                    double cpy = ylyk*al_akl + ypq*rt_akl;
                    double trr_01y = cpy * fac;
                    gout6 += trr_20x * trr_01y * wt;
                    double trr_11y = cpy * trr_10y + 1*b00 * fac;
                    gout7 += trr_10x * trr_11y * wt;
                    gout8 += trr_10x * trr_01y * trr_10z;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    gout9 += 1 * trr_21y * wt;
                    gout10 += 1 * trr_11y * trr_10z;
                    gout11 += 1 * trr_01y * trr_20z;
                    double cpz = zlzk*al_akl + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    gout12 += trr_20x * fac * trr_01z;
                    gout13 += trr_10x * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout14 += trr_10x * fac * trr_11z;
                    gout15 += 1 * trr_20y * trr_01z;
                    gout16 += 1 * trr_10y * trr_11z;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    gout17 += 1 * fac * trr_21z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                val += gout6 * dm[(j0+0)*nao+(k0+1)];
                val += gout12 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                val += gout7 * dm[(j0+0)*nao+(k0+1)];
                val += gout13 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                val += gout8 * dm[(j0+0)*nao+(k0+1)];
                val += gout14 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(k0+0)];
                val += gout9 * dm[(j0+0)*nao+(k0+1)];
                val += gout15 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(k0+0)];
                val += gout10 * dm[(j0+0)*nao+(k0+1)];
                val += gout16 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(k0+0)];
                val += gout11 * dm[(j0+0)*nao+(k0+1)];
                val += gout17 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout6 * dm[(i0+0)*nao+(k0+1)];
                val += gout12 * dm[(i0+0)*nao+(k0+2)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout7 * dm[(i0+1)*nao+(k0+1)];
                val += gout13 * dm[(i0+1)*nao+(k0+2)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                val += gout8 * dm[(i0+2)*nao+(k0+1)];
                val += gout14 * dm[(i0+2)*nao+(k0+2)];
                val += gout3 * dm[(i0+3)*nao+(k0+0)];
                val += gout9 * dm[(i0+3)*nao+(k0+1)];
                val += gout15 * dm[(i0+3)*nao+(k0+2)];
                val += gout4 * dm[(i0+4)*nao+(k0+0)];
                val += gout10 * dm[(i0+4)*nao+(k0+1)];
                val += gout16 * dm[(i0+4)*nao+(k0+2)];
                val += gout5 * dm[(i0+5)*nao+(k0+0)];
                val += gout11 * dm[(i0+5)*nao+(k0+1)];
                val += gout17 * dm[(i0+5)*nao+(k0+2)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout6 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                val = 0;
                val += gout12 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout7 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                val = 0;
                val += gout13 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout8 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                val = 0;
                val += gout14 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                val = 0;
                val += gout9 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                val = 0;
                val += gout15 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                val = 0;
                val += gout10 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                val = 0;
                val += gout16 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                val = 0;
                val += gout11 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                val = 0;
                val += gout17 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                val += gout3 * dm[(i0+3)*nao+(l0+0)];
                val += gout4 * dm[(i0+4)*nao+(l0+0)];
                val += gout5 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(l0+0)];
                val += gout7 * dm[(i0+1)*nao+(l0+0)];
                val += gout8 * dm[(i0+2)*nao+(l0+0)];
                val += gout9 * dm[(i0+3)*nao+(l0+0)];
                val += gout10 * dm[(i0+4)*nao+(l0+0)];
                val += gout11 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                val = 0;
                val += gout12 * dm[(i0+0)*nao+(l0+0)];
                val += gout13 * dm[(i0+1)*nao+(l0+0)];
                val += gout14 * dm[(i0+2)*nao+(l0+0)];
                val += gout15 * dm[(i0+3)*nao+(l0+0)];
                val += gout16 * dm[(i0+4)*nao+(l0+0)];
                val += gout17 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+2), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_2010, MINBLOCKS_2010)
k_2010(KARGS)    { kbody_2010<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_2010, MINBLOCKS_2010)
k_rs_2010(KARGS) { kbody_2010<2>(KFWD); }


#define NTHREADS_2100  256
#define MINBLOCKS_2100 2
template <int NRANGE> __device__ __forceinline__ void
kbody_2100(KARGS)
{
    constexpr int NROOTS = 2;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.; double gout6 = 0.; double gout7 = 0.; double gout8 = 0.; double gout9 = 0.; double gout10 = 0.; double gout11 = 0.; double gout12 = 0.; double gout13 = 0.; double gout14 = 0.; double gout15 = 0.; double gout16 = 0.; double gout17 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    double hrr_2100x = trr_30x - xjxi * trr_20x;
                    gout0 += hrr_2100x * fac * wt;
                    double hrr_1100x = trr_20x - xjxi * trr_10x;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += hrr_1100x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += hrr_1100x * fac * trr_10z;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    double trr_20y = c0y * trr_10y + 1*b10 * fac;
                    gout3 += hrr_0100x * trr_20y * wt;
                    gout4 += hrr_0100x * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += hrr_0100x * fac * trr_20z;
                    double hrr_0100y = trr_10y - yjyi * fac;
                    gout6 += trr_20x * hrr_0100y * wt;
                    double hrr_1100y = trr_20y - yjyi * trr_10y;
                    gout7 += trr_10x * hrr_1100y * wt;
                    gout8 += trr_10x * hrr_0100y * trr_10z;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    double hrr_2100y = trr_30y - yjyi * trr_20y;
                    gout9 += 1 * hrr_2100y * wt;
                    gout10 += 1 * hrr_1100y * trr_10z;
                    gout11 += 1 * hrr_0100y * trr_20z;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    gout12 += trr_20x * fac * hrr_0100z;
                    gout13 += trr_10x * trr_10y * hrr_0100z;
                    double hrr_1100z = trr_20z - zjzi * trr_10z;
                    gout14 += trr_10x * fac * hrr_1100z;
                    gout15 += 1 * trr_20y * hrr_0100z;
                    gout16 += 1 * trr_10y * hrr_1100z;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    double hrr_2100z = trr_30z - zjzi * trr_20z;
                    gout17 += 1 * fac * hrr_2100z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                val += gout6 * dm[(j0+1)*nao+(k0+0)];
                val += gout12 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                val += gout7 * dm[(j0+1)*nao+(k0+0)];
                val += gout13 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                val += gout8 * dm[(j0+1)*nao+(k0+0)];
                val += gout14 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(k0+0)];
                val += gout9 * dm[(j0+1)*nao+(k0+0)];
                val += gout15 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(k0+0)];
                val += gout10 * dm[(j0+1)*nao+(k0+0)];
                val += gout16 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(k0+0)];
                val += gout11 * dm[(j0+1)*nao+(k0+0)];
                val += gout17 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                val += gout3 * dm[(i0+3)*nao+(k0+0)];
                val += gout4 * dm[(i0+4)*nao+(k0+0)];
                val += gout5 * dm[(i0+5)*nao+(k0+0)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(k0+0)];
                val += gout7 * dm[(i0+1)*nao+(k0+0)];
                val += gout8 * dm[(i0+2)*nao+(k0+0)];
                val += gout9 * dm[(i0+3)*nao+(k0+0)];
                val += gout10 * dm[(i0+4)*nao+(k0+0)];
                val += gout11 * dm[(i0+5)*nao+(k0+0)];
                atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                val = 0;
                val += gout12 * dm[(i0+0)*nao+(k0+0)];
                val += gout13 * dm[(i0+1)*nao+(k0+0)];
                val += gout14 * dm[(i0+2)*nao+(k0+0)];
                val += gout15 * dm[(i0+3)*nao+(k0+0)];
                val += gout16 * dm[(i0+4)*nao+(k0+0)];
                val += gout17 * dm[(i0+5)*nao+(k0+0)];
                atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                val += gout6 * dm[(j0+1)*nao+(l0+0)];
                val += gout12 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                val += gout7 * dm[(j0+1)*nao+(l0+0)];
                val += gout13 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                val += gout8 * dm[(j0+1)*nao+(l0+0)];
                val += gout14 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(l0+0)];
                val += gout9 * dm[(j0+1)*nao+(l0+0)];
                val += gout15 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(l0+0)];
                val += gout10 * dm[(j0+1)*nao+(l0+0)];
                val += gout16 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(l0+0)];
                val += gout11 * dm[(j0+1)*nao+(l0+0)];
                val += gout17 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                val += gout3 * dm[(i0+3)*nao+(l0+0)];
                val += gout4 * dm[(i0+4)*nao+(l0+0)];
                val += gout5 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(l0+0)];
                val += gout7 * dm[(i0+1)*nao+(l0+0)];
                val += gout8 * dm[(i0+2)*nao+(l0+0)];
                val += gout9 * dm[(i0+3)*nao+(l0+0)];
                val += gout10 * dm[(i0+4)*nao+(l0+0)];
                val += gout11 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                val = 0;
                val += gout12 * dm[(i0+0)*nao+(l0+0)];
                val += gout13 * dm[(i0+1)*nao+(l0+0)];
                val += gout14 * dm[(i0+2)*nao+(l0+0)];
                val += gout15 * dm[(i0+3)*nao+(l0+0)];
                val += gout16 * dm[(i0+4)*nao+(l0+0)];
                val += gout17 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+2)*nao+(k0+0), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_2100, MINBLOCKS_2100)
k_2100(KARGS)    { kbody_2100<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_2100, MINBLOCKS_2100)
k_rs_2100(KARGS) { kbody_2100<2>(KFWD); }


#define NTHREADS_2110  128
#define MINBLOCKS_2110 2
template <int NRANGE> __device__ __forceinline__ void
kbody_2110(KARGS)
{
    constexpr int NROOTS = 3;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.; double gout6 = 0.; double gout7 = 0.; double gout8 = 0.; double gout9 = 0.; double gout10 = 0.; double gout11 = 0.; double gout12 = 0.; double gout13 = 0.; double gout14 = 0.; double gout15 = 0.; double gout16 = 0.; double gout17 = 0.; double gout18 = 0.; double gout19 = 0.; double gout20 = 0.; double gout21 = 0.; double gout22 = 0.; double gout23 = 0.; double gout24 = 0.; double gout25 = 0.; double gout26 = 0.; double gout27 = 0.; double gout28 = 0.; double gout29 = 0.; double gout30 = 0.; double gout31 = 0.; double gout32 = 0.; double gout33 = 0.; double gout34 = 0.; double gout35 = 0.; double gout36 = 0.; double gout37 = 0.; double gout38 = 0.; double gout39 = 0.; double gout40 = 0.; double gout41 = 0.; double gout42 = 0.; double gout43 = 0.; double gout44 = 0.; double gout45 = 0.; double gout46 = 0.; double gout47 = 0.; double gout48 = 0.; double gout49 = 0.; double gout50 = 0.; double gout51 = 0.; double gout52 = 0.; double gout53 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double b00 = .5 * rt_aa;
                    double rt_akl = rt_aa * aij;
                    double cpx = xlxk*al_akl + xpq*rt_akl;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double hrr_2110x = trr_31x - xjxi * trr_21x;
                    gout0 += hrr_2110x * fac * wt;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double hrr_1110x = trr_21x - xjxi * trr_11x;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += hrr_1110x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += hrr_1110x * fac * trr_10z;
                    double trr_01x = cpx * 1;
                    double hrr_0110x = trr_11x - xjxi * trr_01x;
                    double trr_20y = c0y * trr_10y + 1*b10 * fac;
                    gout3 += hrr_0110x * trr_20y * wt;
                    gout4 += hrr_0110x * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += hrr_0110x * fac * trr_20z;
                    double hrr_0100y = trr_10y - yjyi * fac;
                    gout6 += trr_21x * hrr_0100y * wt;
                    double hrr_1100y = trr_20y - yjyi * trr_10y;
                    gout7 += trr_11x * hrr_1100y * wt;
                    gout8 += trr_11x * hrr_0100y * trr_10z;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    double hrr_2100y = trr_30y - yjyi * trr_20y;
                    gout9 += trr_01x * hrr_2100y * wt;
                    gout10 += trr_01x * hrr_1100y * trr_10z;
                    gout11 += trr_01x * hrr_0100y * trr_20z;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    gout12 += trr_21x * fac * hrr_0100z;
                    gout13 += trr_11x * trr_10y * hrr_0100z;
                    double hrr_1100z = trr_20z - zjzi * trr_10z;
                    gout14 += trr_11x * fac * hrr_1100z;
                    gout15 += trr_01x * trr_20y * hrr_0100z;
                    gout16 += trr_01x * trr_10y * hrr_1100z;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    double hrr_2100z = trr_30z - zjzi * trr_20z;
                    gout17 += trr_01x * fac * hrr_2100z;
                    double hrr_2100x = trr_30x - xjxi * trr_20x;
                    double cpy = ylyk*al_akl + ypq*rt_akl;
                    double trr_01y = cpy * fac;
                    gout18 += hrr_2100x * trr_01y * wt;
                    double hrr_1100x = trr_20x - xjxi * trr_10x;
                    double trr_11y = cpy * trr_10y + 1*b00 * fac;
                    gout19 += hrr_1100x * trr_11y * wt;
                    gout20 += hrr_1100x * trr_01y * trr_10z;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    gout21 += hrr_0100x * trr_21y * wt;
                    gout22 += hrr_0100x * trr_11y * trr_10z;
                    gout23 += hrr_0100x * trr_01y * trr_20z;
                    double hrr_0110y = trr_11y - yjyi * trr_01y;
                    gout24 += trr_20x * hrr_0110y * wt;
                    double hrr_1110y = trr_21y - yjyi * trr_11y;
                    gout25 += trr_10x * hrr_1110y * wt;
                    gout26 += trr_10x * hrr_0110y * trr_10z;
                    double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                    double hrr_2110y = trr_31y - yjyi * trr_21y;
                    gout27 += 1 * hrr_2110y * wt;
                    gout28 += 1 * hrr_1110y * trr_10z;
                    gout29 += 1 * hrr_0110y * trr_20z;
                    gout30 += trr_20x * trr_01y * hrr_0100z;
                    gout31 += trr_10x * trr_11y * hrr_0100z;
                    gout32 += trr_10x * trr_01y * hrr_1100z;
                    gout33 += 1 * trr_21y * hrr_0100z;
                    gout34 += 1 * trr_11y * hrr_1100z;
                    gout35 += 1 * trr_01y * hrr_2100z;
                    double cpz = zlzk*al_akl + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    gout36 += hrr_2100x * fac * trr_01z;
                    gout37 += hrr_1100x * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout38 += hrr_1100x * fac * trr_11z;
                    gout39 += hrr_0100x * trr_20y * trr_01z;
                    gout40 += hrr_0100x * trr_10y * trr_11z;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    gout41 += hrr_0100x * fac * trr_21z;
                    gout42 += trr_20x * hrr_0100y * trr_01z;
                    gout43 += trr_10x * hrr_1100y * trr_01z;
                    gout44 += trr_10x * hrr_0100y * trr_11z;
                    gout45 += 1 * hrr_2100y * trr_01z;
                    gout46 += 1 * hrr_1100y * trr_11z;
                    gout47 += 1 * hrr_0100y * trr_21z;
                    double hrr_0110z = trr_11z - zjzi * trr_01z;
                    gout48 += trr_20x * fac * hrr_0110z;
                    gout49 += trr_10x * trr_10y * hrr_0110z;
                    double hrr_1110z = trr_21z - zjzi * trr_11z;
                    gout50 += trr_10x * fac * hrr_1110z;
                    gout51 += 1 * trr_20y * hrr_0110z;
                    gout52 += 1 * trr_10y * hrr_1110z;
                    double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                    double hrr_2110z = trr_31z - zjzi * trr_21z;
                    gout53 += 1 * fac * hrr_2110z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                val += gout18 * dm[(j0+0)*nao+(k0+1)];
                val += gout36 * dm[(j0+0)*nao+(k0+2)];
                val += gout6 * dm[(j0+1)*nao+(k0+0)];
                val += gout24 * dm[(j0+1)*nao+(k0+1)];
                val += gout42 * dm[(j0+1)*nao+(k0+2)];
                val += gout12 * dm[(j0+2)*nao+(k0+0)];
                val += gout30 * dm[(j0+2)*nao+(k0+1)];
                val += gout48 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                val += gout19 * dm[(j0+0)*nao+(k0+1)];
                val += gout37 * dm[(j0+0)*nao+(k0+2)];
                val += gout7 * dm[(j0+1)*nao+(k0+0)];
                val += gout25 * dm[(j0+1)*nao+(k0+1)];
                val += gout43 * dm[(j0+1)*nao+(k0+2)];
                val += gout13 * dm[(j0+2)*nao+(k0+0)];
                val += gout31 * dm[(j0+2)*nao+(k0+1)];
                val += gout49 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                val += gout20 * dm[(j0+0)*nao+(k0+1)];
                val += gout38 * dm[(j0+0)*nao+(k0+2)];
                val += gout8 * dm[(j0+1)*nao+(k0+0)];
                val += gout26 * dm[(j0+1)*nao+(k0+1)];
                val += gout44 * dm[(j0+1)*nao+(k0+2)];
                val += gout14 * dm[(j0+2)*nao+(k0+0)];
                val += gout32 * dm[(j0+2)*nao+(k0+1)];
                val += gout50 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(k0+0)];
                val += gout21 * dm[(j0+0)*nao+(k0+1)];
                val += gout39 * dm[(j0+0)*nao+(k0+2)];
                val += gout9 * dm[(j0+1)*nao+(k0+0)];
                val += gout27 * dm[(j0+1)*nao+(k0+1)];
                val += gout45 * dm[(j0+1)*nao+(k0+2)];
                val += gout15 * dm[(j0+2)*nao+(k0+0)];
                val += gout33 * dm[(j0+2)*nao+(k0+1)];
                val += gout51 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(k0+0)];
                val += gout22 * dm[(j0+0)*nao+(k0+1)];
                val += gout40 * dm[(j0+0)*nao+(k0+2)];
                val += gout10 * dm[(j0+1)*nao+(k0+0)];
                val += gout28 * dm[(j0+1)*nao+(k0+1)];
                val += gout46 * dm[(j0+1)*nao+(k0+2)];
                val += gout16 * dm[(j0+2)*nao+(k0+0)];
                val += gout34 * dm[(j0+2)*nao+(k0+1)];
                val += gout52 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(k0+0)];
                val += gout23 * dm[(j0+0)*nao+(k0+1)];
                val += gout41 * dm[(j0+0)*nao+(k0+2)];
                val += gout11 * dm[(j0+1)*nao+(k0+0)];
                val += gout29 * dm[(j0+1)*nao+(k0+1)];
                val += gout47 * dm[(j0+1)*nao+(k0+2)];
                val += gout17 * dm[(j0+2)*nao+(k0+0)];
                val += gout35 * dm[(j0+2)*nao+(k0+1)];
                val += gout53 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout18 * dm[(i0+0)*nao+(k0+1)];
                val += gout36 * dm[(i0+0)*nao+(k0+2)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout19 * dm[(i0+1)*nao+(k0+1)];
                val += gout37 * dm[(i0+1)*nao+(k0+2)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                val += gout20 * dm[(i0+2)*nao+(k0+1)];
                val += gout38 * dm[(i0+2)*nao+(k0+2)];
                val += gout3 * dm[(i0+3)*nao+(k0+0)];
                val += gout21 * dm[(i0+3)*nao+(k0+1)];
                val += gout39 * dm[(i0+3)*nao+(k0+2)];
                val += gout4 * dm[(i0+4)*nao+(k0+0)];
                val += gout22 * dm[(i0+4)*nao+(k0+1)];
                val += gout40 * dm[(i0+4)*nao+(k0+2)];
                val += gout5 * dm[(i0+5)*nao+(k0+0)];
                val += gout23 * dm[(i0+5)*nao+(k0+1)];
                val += gout41 * dm[(i0+5)*nao+(k0+2)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(k0+0)];
                val += gout24 * dm[(i0+0)*nao+(k0+1)];
                val += gout42 * dm[(i0+0)*nao+(k0+2)];
                val += gout7 * dm[(i0+1)*nao+(k0+0)];
                val += gout25 * dm[(i0+1)*nao+(k0+1)];
                val += gout43 * dm[(i0+1)*nao+(k0+2)];
                val += gout8 * dm[(i0+2)*nao+(k0+0)];
                val += gout26 * dm[(i0+2)*nao+(k0+1)];
                val += gout44 * dm[(i0+2)*nao+(k0+2)];
                val += gout9 * dm[(i0+3)*nao+(k0+0)];
                val += gout27 * dm[(i0+3)*nao+(k0+1)];
                val += gout45 * dm[(i0+3)*nao+(k0+2)];
                val += gout10 * dm[(i0+4)*nao+(k0+0)];
                val += gout28 * dm[(i0+4)*nao+(k0+1)];
                val += gout46 * dm[(i0+4)*nao+(k0+2)];
                val += gout11 * dm[(i0+5)*nao+(k0+0)];
                val += gout29 * dm[(i0+5)*nao+(k0+1)];
                val += gout47 * dm[(i0+5)*nao+(k0+2)];
                atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                val = 0;
                val += gout12 * dm[(i0+0)*nao+(k0+0)];
                val += gout30 * dm[(i0+0)*nao+(k0+1)];
                val += gout48 * dm[(i0+0)*nao+(k0+2)];
                val += gout13 * dm[(i0+1)*nao+(k0+0)];
                val += gout31 * dm[(i0+1)*nao+(k0+1)];
                val += gout49 * dm[(i0+1)*nao+(k0+2)];
                val += gout14 * dm[(i0+2)*nao+(k0+0)];
                val += gout32 * dm[(i0+2)*nao+(k0+1)];
                val += gout50 * dm[(i0+2)*nao+(k0+2)];
                val += gout15 * dm[(i0+3)*nao+(k0+0)];
                val += gout33 * dm[(i0+3)*nao+(k0+1)];
                val += gout51 * dm[(i0+3)*nao+(k0+2)];
                val += gout16 * dm[(i0+4)*nao+(k0+0)];
                val += gout34 * dm[(i0+4)*nao+(k0+1)];
                val += gout52 * dm[(i0+4)*nao+(k0+2)];
                val += gout17 * dm[(i0+5)*nao+(k0+0)];
                val += gout35 * dm[(i0+5)*nao+(k0+1)];
                val += gout53 * dm[(i0+5)*nao+(k0+2)];
                atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                val += gout6 * dm[(j0+1)*nao+(l0+0)];
                val += gout12 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout18 * dm[(j0+0)*nao+(l0+0)];
                val += gout24 * dm[(j0+1)*nao+(l0+0)];
                val += gout30 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                val = 0;
                val += gout36 * dm[(j0+0)*nao+(l0+0)];
                val += gout42 * dm[(j0+1)*nao+(l0+0)];
                val += gout48 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                val += gout7 * dm[(j0+1)*nao+(l0+0)];
                val += gout13 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout19 * dm[(j0+0)*nao+(l0+0)];
                val += gout25 * dm[(j0+1)*nao+(l0+0)];
                val += gout31 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                val = 0;
                val += gout37 * dm[(j0+0)*nao+(l0+0)];
                val += gout43 * dm[(j0+1)*nao+(l0+0)];
                val += gout49 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                val += gout8 * dm[(j0+1)*nao+(l0+0)];
                val += gout14 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout20 * dm[(j0+0)*nao+(l0+0)];
                val += gout26 * dm[(j0+1)*nao+(l0+0)];
                val += gout32 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                val = 0;
                val += gout38 * dm[(j0+0)*nao+(l0+0)];
                val += gout44 * dm[(j0+1)*nao+(l0+0)];
                val += gout50 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(l0+0)];
                val += gout9 * dm[(j0+1)*nao+(l0+0)];
                val += gout15 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                val = 0;
                val += gout21 * dm[(j0+0)*nao+(l0+0)];
                val += gout27 * dm[(j0+1)*nao+(l0+0)];
                val += gout33 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                val = 0;
                val += gout39 * dm[(j0+0)*nao+(l0+0)];
                val += gout45 * dm[(j0+1)*nao+(l0+0)];
                val += gout51 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(l0+0)];
                val += gout10 * dm[(j0+1)*nao+(l0+0)];
                val += gout16 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                val = 0;
                val += gout22 * dm[(j0+0)*nao+(l0+0)];
                val += gout28 * dm[(j0+1)*nao+(l0+0)];
                val += gout34 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                val = 0;
                val += gout40 * dm[(j0+0)*nao+(l0+0)];
                val += gout46 * dm[(j0+1)*nao+(l0+0)];
                val += gout52 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(l0+0)];
                val += gout11 * dm[(j0+1)*nao+(l0+0)];
                val += gout17 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                val = 0;
                val += gout23 * dm[(j0+0)*nao+(l0+0)];
                val += gout29 * dm[(j0+1)*nao+(l0+0)];
                val += gout35 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                val = 0;
                val += gout41 * dm[(j0+0)*nao+(l0+0)];
                val += gout47 * dm[(j0+1)*nao+(l0+0)];
                val += gout53 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                val += gout3 * dm[(i0+3)*nao+(l0+0)];
                val += gout4 * dm[(i0+4)*nao+(l0+0)];
                val += gout5 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                val = 0;
                val += gout18 * dm[(i0+0)*nao+(l0+0)];
                val += gout19 * dm[(i0+1)*nao+(l0+0)];
                val += gout20 * dm[(i0+2)*nao+(l0+0)];
                val += gout21 * dm[(i0+3)*nao+(l0+0)];
                val += gout22 * dm[(i0+4)*nao+(l0+0)];
                val += gout23 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                val = 0;
                val += gout36 * dm[(i0+0)*nao+(l0+0)];
                val += gout37 * dm[(i0+1)*nao+(l0+0)];
                val += gout38 * dm[(i0+2)*nao+(l0+0)];
                val += gout39 * dm[(i0+3)*nao+(l0+0)];
                val += gout40 * dm[(i0+4)*nao+(l0+0)];
                val += gout41 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(l0+0)];
                val += gout7 * dm[(i0+1)*nao+(l0+0)];
                val += gout8 * dm[(i0+2)*nao+(l0+0)];
                val += gout9 * dm[(i0+3)*nao+(l0+0)];
                val += gout10 * dm[(i0+4)*nao+(l0+0)];
                val += gout11 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                val = 0;
                val += gout24 * dm[(i0+0)*nao+(l0+0)];
                val += gout25 * dm[(i0+1)*nao+(l0+0)];
                val += gout26 * dm[(i0+2)*nao+(l0+0)];
                val += gout27 * dm[(i0+3)*nao+(l0+0)];
                val += gout28 * dm[(i0+4)*nao+(l0+0)];
                val += gout29 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+1)*nao+(k0+1), val);
                val = 0;
                val += gout42 * dm[(i0+0)*nao+(l0+0)];
                val += gout43 * dm[(i0+1)*nao+(l0+0)];
                val += gout44 * dm[(i0+2)*nao+(l0+0)];
                val += gout45 * dm[(i0+3)*nao+(l0+0)];
                val += gout46 * dm[(i0+4)*nao+(l0+0)];
                val += gout47 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+1)*nao+(k0+2), val);
                val = 0;
                val += gout12 * dm[(i0+0)*nao+(l0+0)];
                val += gout13 * dm[(i0+1)*nao+(l0+0)];
                val += gout14 * dm[(i0+2)*nao+(l0+0)];
                val += gout15 * dm[(i0+3)*nao+(l0+0)];
                val += gout16 * dm[(i0+4)*nao+(l0+0)];
                val += gout17 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                val = 0;
                val += gout30 * dm[(i0+0)*nao+(l0+0)];
                val += gout31 * dm[(i0+1)*nao+(l0+0)];
                val += gout32 * dm[(i0+2)*nao+(l0+0)];
                val += gout33 * dm[(i0+3)*nao+(l0+0)];
                val += gout34 * dm[(i0+4)*nao+(l0+0)];
                val += gout35 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+2)*nao+(k0+1), val);
                val = 0;
                val += gout48 * dm[(i0+0)*nao+(l0+0)];
                val += gout49 * dm[(i0+1)*nao+(l0+0)];
                val += gout50 * dm[(i0+2)*nao+(l0+0)];
                val += gout51 * dm[(i0+3)*nao+(l0+0)];
                val += gout52 * dm[(i0+4)*nao+(l0+0)];
                val += gout53 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+2)*nao+(k0+2), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_2110, MINBLOCKS_2110)
k_2110(KARGS)    { kbody_2110<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_2110, MINBLOCKS_2110)
k_rs_2110(KARGS) { kbody_2110<2>(KFWD); }


#define NTHREADS_1111  64
#define MINBLOCKS_1111 4
template <int NRANGE> __device__ __forceinline__ void
kbody_1111(KARGS)
{
    constexpr int NROOTS = 3;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.; double gout6 = 0.; double gout7 = 0.; double gout8 = 0.; double gout9 = 0.; double gout10 = 0.; double gout11 = 0.; double gout12 = 0.; double gout13 = 0.; double gout14 = 0.; double gout15 = 0.; double gout16 = 0.; double gout17 = 0.; double gout18 = 0.; double gout19 = 0.; double gout20 = 0.; double gout21 = 0.; double gout22 = 0.; double gout23 = 0.; double gout24 = 0.; double gout25 = 0.; double gout26 = 0.; double gout27 = 0.; double gout28 = 0.; double gout29 = 0.; double gout30 = 0.; double gout31 = 0.; double gout32 = 0.; double gout33 = 0.; double gout34 = 0.; double gout35 = 0.; double gout36 = 0.; double gout37 = 0.; double gout38 = 0.; double gout39 = 0.; double gout40 = 0.; double gout41 = 0.; double gout42 = 0.; double gout43 = 0.; double gout44 = 0.; double gout45 = 0.; double gout46 = 0.; double gout47 = 0.; double gout48 = 0.; double gout49 = 0.; double gout50 = 0.; double gout51 = 0.; double gout52 = 0.; double gout53 = 0.; double gout54 = 0.; double gout55 = 0.; double gout56 = 0.; double gout57 = 0.; double gout58 = 0.; double gout59 = 0.; double gout60 = 0.; double gout61 = 0.; double gout62 = 0.; double gout63 = 0.; double gout64 = 0.; double gout65 = 0.; double gout66 = 0.; double gout67 = 0.; double gout68 = 0.; double gout69 = 0.; double gout70 = 0.; double gout71 = 0.; double gout72 = 0.; double gout73 = 0.; double gout74 = 0.; double gout75 = 0.; double gout76 = 0.; double gout77 = 0.; double gout78 = 0.; double gout79 = 0.; double gout80 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double b00 = .5 * rt_aa;
                    double rt_akl = rt_aa * aij;
                    double b01 = .5*iakl * (1 - rt_akl);
                    double cpx = xlxk*al_akl + xpq*rt_akl;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                    double hrr_2011x = trr_22x - xlxk * trr_21x;
                    double trr_01x = cpx * 1;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    double hrr_1011x = trr_12x - xlxk * trr_11x;
                    double hrr_1111x = hrr_2011x - xjxi * hrr_1011x;
                    gout0 += hrr_1111x * fac * wt;
                    double trr_02x = cpx * trr_01x + 1*b01 * 1;
                    double hrr_0011x = trr_02x - xlxk * trr_01x;
                    double hrr_0111x = hrr_1011x - xjxi * hrr_0011x;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += hrr_0111x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += hrr_0111x * fac * trr_10z;
                    double hrr_0100y = trr_10y - yjyi * fac;
                    gout3 += hrr_1011x * hrr_0100y * wt;
                    double trr_20y = c0y * trr_10y + 1*b10 * fac;
                    double hrr_1100y = trr_20y - yjyi * trr_10y;
                    gout4 += hrr_0011x * hrr_1100y * wt;
                    gout5 += hrr_0011x * hrr_0100y * trr_10z;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    gout6 += hrr_1011x * fac * hrr_0100z;
                    gout7 += hrr_0011x * trr_10y * hrr_0100z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    double hrr_1100z = trr_20z - zjzi * trr_10z;
                    gout8 += hrr_0011x * fac * hrr_1100z;
                    double hrr_2001x = trr_21x - xlxk * trr_20x;
                    double hrr_1001x = trr_11x - xlxk * trr_10x;
                    double hrr_1101x = hrr_2001x - xjxi * hrr_1001x;
                    double cpy = ylyk*al_akl + ypq*rt_akl;
                    double trr_01y = cpy * fac;
                    gout9 += hrr_1101x * trr_01y * wt;
                    double hrr_0001x = trr_01x - xlxk * 1;
                    double hrr_0101x = hrr_1001x - xjxi * hrr_0001x;
                    double trr_11y = cpy * trr_10y + 1*b00 * fac;
                    gout10 += hrr_0101x * trr_11y * wt;
                    gout11 += hrr_0101x * trr_01y * trr_10z;
                    double hrr_0110y = trr_11y - yjyi * trr_01y;
                    gout12 += hrr_1001x * hrr_0110y * wt;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    double hrr_1110y = trr_21y - yjyi * trr_11y;
                    gout13 += hrr_0001x * hrr_1110y * wt;
                    gout14 += hrr_0001x * hrr_0110y * trr_10z;
                    gout15 += hrr_1001x * trr_01y * hrr_0100z;
                    gout16 += hrr_0001x * trr_11y * hrr_0100z;
                    gout17 += hrr_0001x * trr_01y * hrr_1100z;
                    double cpz = zlzk*al_akl + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    gout18 += hrr_1101x * fac * trr_01z;
                    gout19 += hrr_0101x * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout20 += hrr_0101x * fac * trr_11z;
                    gout21 += hrr_1001x * hrr_0100y * trr_01z;
                    gout22 += hrr_0001x * hrr_1100y * trr_01z;
                    gout23 += hrr_0001x * hrr_0100y * trr_11z;
                    double hrr_0110z = trr_11z - zjzi * trr_01z;
                    gout24 += hrr_1001x * fac * hrr_0110z;
                    gout25 += hrr_0001x * trr_10y * hrr_0110z;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    double hrr_1110z = trr_21z - zjzi * trr_11z;
                    gout26 += hrr_0001x * fac * hrr_1110z;
                    double hrr_1110x = trr_21x - xjxi * trr_11x;
                    double hrr_0001y = trr_01y - ylyk * fac;
                    gout27 += hrr_1110x * hrr_0001y * wt;
                    double hrr_0110x = trr_11x - xjxi * trr_01x;
                    double hrr_1001y = trr_11y - ylyk * trr_10y;
                    gout28 += hrr_0110x * hrr_1001y * wt;
                    gout29 += hrr_0110x * hrr_0001y * trr_10z;
                    double hrr_0101y = hrr_1001y - yjyi * hrr_0001y;
                    gout30 += trr_11x * hrr_0101y * wt;
                    double hrr_2001y = trr_21y - ylyk * trr_20y;
                    double hrr_1101y = hrr_2001y - yjyi * hrr_1001y;
                    gout31 += trr_01x * hrr_1101y * wt;
                    gout32 += trr_01x * hrr_0101y * trr_10z;
                    gout33 += trr_11x * hrr_0001y * hrr_0100z;
                    gout34 += trr_01x * hrr_1001y * hrr_0100z;
                    gout35 += trr_01x * hrr_0001y * hrr_1100z;
                    double hrr_1100x = trr_20x - xjxi * trr_10x;
                    double trr_02y = cpy * trr_01y + 1*b01 * fac;
                    double hrr_0011y = trr_02y - ylyk * trr_01y;
                    gout36 += hrr_1100x * hrr_0011y * wt;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    double hrr_1011y = trr_12y - ylyk * trr_11y;
                    gout37 += hrr_0100x * hrr_1011y * wt;
                    gout38 += hrr_0100x * hrr_0011y * trr_10z;
                    double hrr_0111y = hrr_1011y - yjyi * hrr_0011y;
                    gout39 += trr_10x * hrr_0111y * wt;
                    double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                    double hrr_2011y = trr_22y - ylyk * trr_21y;
                    double hrr_1111y = hrr_2011y - yjyi * hrr_1011y;
                    gout40 += 1 * hrr_1111y * wt;
                    gout41 += 1 * hrr_0111y * trr_10z;
                    gout42 += trr_10x * hrr_0011y * hrr_0100z;
                    gout43 += 1 * hrr_1011y * hrr_0100z;
                    gout44 += 1 * hrr_0011y * hrr_1100z;
                    gout45 += hrr_1100x * hrr_0001y * trr_01z;
                    gout46 += hrr_0100x * hrr_1001y * trr_01z;
                    gout47 += hrr_0100x * hrr_0001y * trr_11z;
                    gout48 += trr_10x * hrr_0101y * trr_01z;
                    gout49 += 1 * hrr_1101y * trr_01z;
                    gout50 += 1 * hrr_0101y * trr_11z;
                    gout51 += trr_10x * hrr_0001y * hrr_0110z;
                    gout52 += 1 * hrr_1001y * hrr_0110z;
                    gout53 += 1 * hrr_0001y * hrr_1110z;
                    double hrr_0001z = trr_01z - zlzk * wt;
                    gout54 += hrr_1110x * fac * hrr_0001z;
                    gout55 += hrr_0110x * trr_10y * hrr_0001z;
                    double hrr_1001z = trr_11z - zlzk * trr_10z;
                    gout56 += hrr_0110x * fac * hrr_1001z;
                    gout57 += trr_11x * hrr_0100y * hrr_0001z;
                    gout58 += trr_01x * hrr_1100y * hrr_0001z;
                    gout59 += trr_01x * hrr_0100y * hrr_1001z;
                    double hrr_0101z = hrr_1001z - zjzi * hrr_0001z;
                    gout60 += trr_11x * fac * hrr_0101z;
                    gout61 += trr_01x * trr_10y * hrr_0101z;
                    double hrr_2001z = trr_21z - zlzk * trr_20z;
                    double hrr_1101z = hrr_2001z - zjzi * hrr_1001z;
                    gout62 += trr_01x * fac * hrr_1101z;
                    gout63 += hrr_1100x * trr_01y * hrr_0001z;
                    gout64 += hrr_0100x * trr_11y * hrr_0001z;
                    gout65 += hrr_0100x * trr_01y * hrr_1001z;
                    gout66 += trr_10x * hrr_0110y * hrr_0001z;
                    gout67 += 1 * hrr_1110y * hrr_0001z;
                    gout68 += 1 * hrr_0110y * hrr_1001z;
                    gout69 += trr_10x * trr_01y * hrr_0101z;
                    gout70 += 1 * trr_11y * hrr_0101z;
                    gout71 += 1 * trr_01y * hrr_1101z;
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    double hrr_0011z = trr_02z - zlzk * trr_01z;
                    gout72 += hrr_1100x * fac * hrr_0011z;
                    gout73 += hrr_0100x * trr_10y * hrr_0011z;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    double hrr_1011z = trr_12z - zlzk * trr_11z;
                    gout74 += hrr_0100x * fac * hrr_1011z;
                    gout75 += trr_10x * hrr_0100y * hrr_0011z;
                    gout76 += 1 * hrr_1100y * hrr_0011z;
                    gout77 += 1 * hrr_0100y * hrr_1011z;
                    double hrr_0111z = hrr_1011z - zjzi * hrr_0011z;
                    gout78 += trr_10x * fac * hrr_0111z;
                    gout79 += 1 * trr_10y * hrr_0111z;
                    double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                    double hrr_2011z = trr_22z - zlzk * trr_21z;
                    double hrr_1111z = hrr_2011z - zjzi * hrr_1011z;
                    gout80 += 1 * fac * hrr_1111z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                val += gout9 * dm[(j0+0)*nao+(k0+1)];
                val += gout18 * dm[(j0+0)*nao+(k0+2)];
                val += gout3 * dm[(j0+1)*nao+(k0+0)];
                val += gout12 * dm[(j0+1)*nao+(k0+1)];
                val += gout21 * dm[(j0+1)*nao+(k0+2)];
                val += gout6 * dm[(j0+2)*nao+(k0+0)];
                val += gout15 * dm[(j0+2)*nao+(k0+1)];
                val += gout24 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout27 * dm[(j0+0)*nao+(k0+0)];
                val += gout36 * dm[(j0+0)*nao+(k0+1)];
                val += gout45 * dm[(j0+0)*nao+(k0+2)];
                val += gout30 * dm[(j0+1)*nao+(k0+0)];
                val += gout39 * dm[(j0+1)*nao+(k0+1)];
                val += gout48 * dm[(j0+1)*nao+(k0+2)];
                val += gout33 * dm[(j0+2)*nao+(k0+0)];
                val += gout42 * dm[(j0+2)*nao+(k0+1)];
                val += gout51 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+0)*nao+(l0+1), val);
                val = 0;
                val += gout54 * dm[(j0+0)*nao+(k0+0)];
                val += gout63 * dm[(j0+0)*nao+(k0+1)];
                val += gout72 * dm[(j0+0)*nao+(k0+2)];
                val += gout57 * dm[(j0+1)*nao+(k0+0)];
                val += gout66 * dm[(j0+1)*nao+(k0+1)];
                val += gout75 * dm[(j0+1)*nao+(k0+2)];
                val += gout60 * dm[(j0+2)*nao+(k0+0)];
                val += gout69 * dm[(j0+2)*nao+(k0+1)];
                val += gout78 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+0)*nao+(l0+2), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                val += gout10 * dm[(j0+0)*nao+(k0+1)];
                val += gout19 * dm[(j0+0)*nao+(k0+2)];
                val += gout4 * dm[(j0+1)*nao+(k0+0)];
                val += gout13 * dm[(j0+1)*nao+(k0+1)];
                val += gout22 * dm[(j0+1)*nao+(k0+2)];
                val += gout7 * dm[(j0+2)*nao+(k0+0)];
                val += gout16 * dm[(j0+2)*nao+(k0+1)];
                val += gout25 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout28 * dm[(j0+0)*nao+(k0+0)];
                val += gout37 * dm[(j0+0)*nao+(k0+1)];
                val += gout46 * dm[(j0+0)*nao+(k0+2)];
                val += gout31 * dm[(j0+1)*nao+(k0+0)];
                val += gout40 * dm[(j0+1)*nao+(k0+1)];
                val += gout49 * dm[(j0+1)*nao+(k0+2)];
                val += gout34 * dm[(j0+2)*nao+(k0+0)];
                val += gout43 * dm[(j0+2)*nao+(k0+1)];
                val += gout52 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+1)*nao+(l0+1), val);
                val = 0;
                val += gout55 * dm[(j0+0)*nao+(k0+0)];
                val += gout64 * dm[(j0+0)*nao+(k0+1)];
                val += gout73 * dm[(j0+0)*nao+(k0+2)];
                val += gout58 * dm[(j0+1)*nao+(k0+0)];
                val += gout67 * dm[(j0+1)*nao+(k0+1)];
                val += gout76 * dm[(j0+1)*nao+(k0+2)];
                val += gout61 * dm[(j0+2)*nao+(k0+0)];
                val += gout70 * dm[(j0+2)*nao+(k0+1)];
                val += gout79 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+1)*nao+(l0+2), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                val += gout11 * dm[(j0+0)*nao+(k0+1)];
                val += gout20 * dm[(j0+0)*nao+(k0+2)];
                val += gout5 * dm[(j0+1)*nao+(k0+0)];
                val += gout14 * dm[(j0+1)*nao+(k0+1)];
                val += gout23 * dm[(j0+1)*nao+(k0+2)];
                val += gout8 * dm[(j0+2)*nao+(k0+0)];
                val += gout17 * dm[(j0+2)*nao+(k0+1)];
                val += gout26 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout29 * dm[(j0+0)*nao+(k0+0)];
                val += gout38 * dm[(j0+0)*nao+(k0+1)];
                val += gout47 * dm[(j0+0)*nao+(k0+2)];
                val += gout32 * dm[(j0+1)*nao+(k0+0)];
                val += gout41 * dm[(j0+1)*nao+(k0+1)];
                val += gout50 * dm[(j0+1)*nao+(k0+2)];
                val += gout35 * dm[(j0+2)*nao+(k0+0)];
                val += gout44 * dm[(j0+2)*nao+(k0+1)];
                val += gout53 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+2)*nao+(l0+1), val);
                val = 0;
                val += gout56 * dm[(j0+0)*nao+(k0+0)];
                val += gout65 * dm[(j0+0)*nao+(k0+1)];
                val += gout74 * dm[(j0+0)*nao+(k0+2)];
                val += gout59 * dm[(j0+1)*nao+(k0+0)];
                val += gout68 * dm[(j0+1)*nao+(k0+1)];
                val += gout77 * dm[(j0+1)*nao+(k0+2)];
                val += gout62 * dm[(j0+2)*nao+(k0+0)];
                val += gout71 * dm[(j0+2)*nao+(k0+1)];
                val += gout80 * dm[(j0+2)*nao+(k0+2)];
                atomicAdd(vk+(i0+2)*nao+(l0+2), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout9 * dm[(i0+0)*nao+(k0+1)];
                val += gout18 * dm[(i0+0)*nao+(k0+2)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout10 * dm[(i0+1)*nao+(k0+1)];
                val += gout19 * dm[(i0+1)*nao+(k0+2)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                val += gout11 * dm[(i0+2)*nao+(k0+1)];
                val += gout20 * dm[(i0+2)*nao+(k0+2)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout27 * dm[(i0+0)*nao+(k0+0)];
                val += gout36 * dm[(i0+0)*nao+(k0+1)];
                val += gout45 * dm[(i0+0)*nao+(k0+2)];
                val += gout28 * dm[(i0+1)*nao+(k0+0)];
                val += gout37 * dm[(i0+1)*nao+(k0+1)];
                val += gout46 * dm[(i0+1)*nao+(k0+2)];
                val += gout29 * dm[(i0+2)*nao+(k0+0)];
                val += gout38 * dm[(i0+2)*nao+(k0+1)];
                val += gout47 * dm[(i0+2)*nao+(k0+2)];
                atomicAdd(vk+(j0+0)*nao+(l0+1), val);
                val = 0;
                val += gout54 * dm[(i0+0)*nao+(k0+0)];
                val += gout63 * dm[(i0+0)*nao+(k0+1)];
                val += gout72 * dm[(i0+0)*nao+(k0+2)];
                val += gout55 * dm[(i0+1)*nao+(k0+0)];
                val += gout64 * dm[(i0+1)*nao+(k0+1)];
                val += gout73 * dm[(i0+1)*nao+(k0+2)];
                val += gout56 * dm[(i0+2)*nao+(k0+0)];
                val += gout65 * dm[(i0+2)*nao+(k0+1)];
                val += gout74 * dm[(i0+2)*nao+(k0+2)];
                atomicAdd(vk+(j0+0)*nao+(l0+2), val);
                val = 0;
                val += gout3 * dm[(i0+0)*nao+(k0+0)];
                val += gout12 * dm[(i0+0)*nao+(k0+1)];
                val += gout21 * dm[(i0+0)*nao+(k0+2)];
                val += gout4 * dm[(i0+1)*nao+(k0+0)];
                val += gout13 * dm[(i0+1)*nao+(k0+1)];
                val += gout22 * dm[(i0+1)*nao+(k0+2)];
                val += gout5 * dm[(i0+2)*nao+(k0+0)];
                val += gout14 * dm[(i0+2)*nao+(k0+1)];
                val += gout23 * dm[(i0+2)*nao+(k0+2)];
                atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                val = 0;
                val += gout30 * dm[(i0+0)*nao+(k0+0)];
                val += gout39 * dm[(i0+0)*nao+(k0+1)];
                val += gout48 * dm[(i0+0)*nao+(k0+2)];
                val += gout31 * dm[(i0+1)*nao+(k0+0)];
                val += gout40 * dm[(i0+1)*nao+(k0+1)];
                val += gout49 * dm[(i0+1)*nao+(k0+2)];
                val += gout32 * dm[(i0+2)*nao+(k0+0)];
                val += gout41 * dm[(i0+2)*nao+(k0+1)];
                val += gout50 * dm[(i0+2)*nao+(k0+2)];
                atomicAdd(vk+(j0+1)*nao+(l0+1), val);
                val = 0;
                val += gout57 * dm[(i0+0)*nao+(k0+0)];
                val += gout66 * dm[(i0+0)*nao+(k0+1)];
                val += gout75 * dm[(i0+0)*nao+(k0+2)];
                val += gout58 * dm[(i0+1)*nao+(k0+0)];
                val += gout67 * dm[(i0+1)*nao+(k0+1)];
                val += gout76 * dm[(i0+1)*nao+(k0+2)];
                val += gout59 * dm[(i0+2)*nao+(k0+0)];
                val += gout68 * dm[(i0+2)*nao+(k0+1)];
                val += gout77 * dm[(i0+2)*nao+(k0+2)];
                atomicAdd(vk+(j0+1)*nao+(l0+2), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(k0+0)];
                val += gout15 * dm[(i0+0)*nao+(k0+1)];
                val += gout24 * dm[(i0+0)*nao+(k0+2)];
                val += gout7 * dm[(i0+1)*nao+(k0+0)];
                val += gout16 * dm[(i0+1)*nao+(k0+1)];
                val += gout25 * dm[(i0+1)*nao+(k0+2)];
                val += gout8 * dm[(i0+2)*nao+(k0+0)];
                val += gout17 * dm[(i0+2)*nao+(k0+1)];
                val += gout26 * dm[(i0+2)*nao+(k0+2)];
                atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                val = 0;
                val += gout33 * dm[(i0+0)*nao+(k0+0)];
                val += gout42 * dm[(i0+0)*nao+(k0+1)];
                val += gout51 * dm[(i0+0)*nao+(k0+2)];
                val += gout34 * dm[(i0+1)*nao+(k0+0)];
                val += gout43 * dm[(i0+1)*nao+(k0+1)];
                val += gout52 * dm[(i0+1)*nao+(k0+2)];
                val += gout35 * dm[(i0+2)*nao+(k0+0)];
                val += gout44 * dm[(i0+2)*nao+(k0+1)];
                val += gout53 * dm[(i0+2)*nao+(k0+2)];
                atomicAdd(vk+(j0+2)*nao+(l0+1), val);
                val = 0;
                val += gout60 * dm[(i0+0)*nao+(k0+0)];
                val += gout69 * dm[(i0+0)*nao+(k0+1)];
                val += gout78 * dm[(i0+0)*nao+(k0+2)];
                val += gout61 * dm[(i0+1)*nao+(k0+0)];
                val += gout70 * dm[(i0+1)*nao+(k0+1)];
                val += gout79 * dm[(i0+1)*nao+(k0+2)];
                val += gout62 * dm[(i0+2)*nao+(k0+0)];
                val += gout71 * dm[(i0+2)*nao+(k0+1)];
                val += gout80 * dm[(i0+2)*nao+(k0+2)];
                atomicAdd(vk+(j0+2)*nao+(l0+2), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                val += gout27 * dm[(j0+0)*nao+(l0+1)];
                val += gout54 * dm[(j0+0)*nao+(l0+2)];
                val += gout3 * dm[(j0+1)*nao+(l0+0)];
                val += gout30 * dm[(j0+1)*nao+(l0+1)];
                val += gout57 * dm[(j0+1)*nao+(l0+2)];
                val += gout6 * dm[(j0+2)*nao+(l0+0)];
                val += gout33 * dm[(j0+2)*nao+(l0+1)];
                val += gout60 * dm[(j0+2)*nao+(l0+2)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout9 * dm[(j0+0)*nao+(l0+0)];
                val += gout36 * dm[(j0+0)*nao+(l0+1)];
                val += gout63 * dm[(j0+0)*nao+(l0+2)];
                val += gout12 * dm[(j0+1)*nao+(l0+0)];
                val += gout39 * dm[(j0+1)*nao+(l0+1)];
                val += gout66 * dm[(j0+1)*nao+(l0+2)];
                val += gout15 * dm[(j0+2)*nao+(l0+0)];
                val += gout42 * dm[(j0+2)*nao+(l0+1)];
                val += gout69 * dm[(j0+2)*nao+(l0+2)];
                atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                val = 0;
                val += gout18 * dm[(j0+0)*nao+(l0+0)];
                val += gout45 * dm[(j0+0)*nao+(l0+1)];
                val += gout72 * dm[(j0+0)*nao+(l0+2)];
                val += gout21 * dm[(j0+1)*nao+(l0+0)];
                val += gout48 * dm[(j0+1)*nao+(l0+1)];
                val += gout75 * dm[(j0+1)*nao+(l0+2)];
                val += gout24 * dm[(j0+2)*nao+(l0+0)];
                val += gout51 * dm[(j0+2)*nao+(l0+1)];
                val += gout78 * dm[(j0+2)*nao+(l0+2)];
                atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                val += gout28 * dm[(j0+0)*nao+(l0+1)];
                val += gout55 * dm[(j0+0)*nao+(l0+2)];
                val += gout4 * dm[(j0+1)*nao+(l0+0)];
                val += gout31 * dm[(j0+1)*nao+(l0+1)];
                val += gout58 * dm[(j0+1)*nao+(l0+2)];
                val += gout7 * dm[(j0+2)*nao+(l0+0)];
                val += gout34 * dm[(j0+2)*nao+(l0+1)];
                val += gout61 * dm[(j0+2)*nao+(l0+2)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout10 * dm[(j0+0)*nao+(l0+0)];
                val += gout37 * dm[(j0+0)*nao+(l0+1)];
                val += gout64 * dm[(j0+0)*nao+(l0+2)];
                val += gout13 * dm[(j0+1)*nao+(l0+0)];
                val += gout40 * dm[(j0+1)*nao+(l0+1)];
                val += gout67 * dm[(j0+1)*nao+(l0+2)];
                val += gout16 * dm[(j0+2)*nao+(l0+0)];
                val += gout43 * dm[(j0+2)*nao+(l0+1)];
                val += gout70 * dm[(j0+2)*nao+(l0+2)];
                atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                val = 0;
                val += gout19 * dm[(j0+0)*nao+(l0+0)];
                val += gout46 * dm[(j0+0)*nao+(l0+1)];
                val += gout73 * dm[(j0+0)*nao+(l0+2)];
                val += gout22 * dm[(j0+1)*nao+(l0+0)];
                val += gout49 * dm[(j0+1)*nao+(l0+1)];
                val += gout76 * dm[(j0+1)*nao+(l0+2)];
                val += gout25 * dm[(j0+2)*nao+(l0+0)];
                val += gout52 * dm[(j0+2)*nao+(l0+1)];
                val += gout79 * dm[(j0+2)*nao+(l0+2)];
                atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                val += gout29 * dm[(j0+0)*nao+(l0+1)];
                val += gout56 * dm[(j0+0)*nao+(l0+2)];
                val += gout5 * dm[(j0+1)*nao+(l0+0)];
                val += gout32 * dm[(j0+1)*nao+(l0+1)];
                val += gout59 * dm[(j0+1)*nao+(l0+2)];
                val += gout8 * dm[(j0+2)*nao+(l0+0)];
                val += gout35 * dm[(j0+2)*nao+(l0+1)];
                val += gout62 * dm[(j0+2)*nao+(l0+2)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout11 * dm[(j0+0)*nao+(l0+0)];
                val += gout38 * dm[(j0+0)*nao+(l0+1)];
                val += gout65 * dm[(j0+0)*nao+(l0+2)];
                val += gout14 * dm[(j0+1)*nao+(l0+0)];
                val += gout41 * dm[(j0+1)*nao+(l0+1)];
                val += gout68 * dm[(j0+1)*nao+(l0+2)];
                val += gout17 * dm[(j0+2)*nao+(l0+0)];
                val += gout44 * dm[(j0+2)*nao+(l0+1)];
                val += gout71 * dm[(j0+2)*nao+(l0+2)];
                atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                val = 0;
                val += gout20 * dm[(j0+0)*nao+(l0+0)];
                val += gout47 * dm[(j0+0)*nao+(l0+1)];
                val += gout74 * dm[(j0+0)*nao+(l0+2)];
                val += gout23 * dm[(j0+1)*nao+(l0+0)];
                val += gout50 * dm[(j0+1)*nao+(l0+1)];
                val += gout77 * dm[(j0+1)*nao+(l0+2)];
                val += gout26 * dm[(j0+2)*nao+(l0+0)];
                val += gout53 * dm[(j0+2)*nao+(l0+1)];
                val += gout80 * dm[(j0+2)*nao+(l0+2)];
                atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout27 * dm[(i0+0)*nao+(l0+1)];
                val += gout54 * dm[(i0+0)*nao+(l0+2)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout28 * dm[(i0+1)*nao+(l0+1)];
                val += gout55 * dm[(i0+1)*nao+(l0+2)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                val += gout29 * dm[(i0+2)*nao+(l0+1)];
                val += gout56 * dm[(i0+2)*nao+(l0+2)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                val = 0;
                val += gout9 * dm[(i0+0)*nao+(l0+0)];
                val += gout36 * dm[(i0+0)*nao+(l0+1)];
                val += gout63 * dm[(i0+0)*nao+(l0+2)];
                val += gout10 * dm[(i0+1)*nao+(l0+0)];
                val += gout37 * dm[(i0+1)*nao+(l0+1)];
                val += gout64 * dm[(i0+1)*nao+(l0+2)];
                val += gout11 * dm[(i0+2)*nao+(l0+0)];
                val += gout38 * dm[(i0+2)*nao+(l0+1)];
                val += gout65 * dm[(i0+2)*nao+(l0+2)];
                atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                val = 0;
                val += gout18 * dm[(i0+0)*nao+(l0+0)];
                val += gout45 * dm[(i0+0)*nao+(l0+1)];
                val += gout72 * dm[(i0+0)*nao+(l0+2)];
                val += gout19 * dm[(i0+1)*nao+(l0+0)];
                val += gout46 * dm[(i0+1)*nao+(l0+1)];
                val += gout73 * dm[(i0+1)*nao+(l0+2)];
                val += gout20 * dm[(i0+2)*nao+(l0+0)];
                val += gout47 * dm[(i0+2)*nao+(l0+1)];
                val += gout74 * dm[(i0+2)*nao+(l0+2)];
                atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                val = 0;
                val += gout3 * dm[(i0+0)*nao+(l0+0)];
                val += gout30 * dm[(i0+0)*nao+(l0+1)];
                val += gout57 * dm[(i0+0)*nao+(l0+2)];
                val += gout4 * dm[(i0+1)*nao+(l0+0)];
                val += gout31 * dm[(i0+1)*nao+(l0+1)];
                val += gout58 * dm[(i0+1)*nao+(l0+2)];
                val += gout5 * dm[(i0+2)*nao+(l0+0)];
                val += gout32 * dm[(i0+2)*nao+(l0+1)];
                val += gout59 * dm[(i0+2)*nao+(l0+2)];
                atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                val = 0;
                val += gout12 * dm[(i0+0)*nao+(l0+0)];
                val += gout39 * dm[(i0+0)*nao+(l0+1)];
                val += gout66 * dm[(i0+0)*nao+(l0+2)];
                val += gout13 * dm[(i0+1)*nao+(l0+0)];
                val += gout40 * dm[(i0+1)*nao+(l0+1)];
                val += gout67 * dm[(i0+1)*nao+(l0+2)];
                val += gout14 * dm[(i0+2)*nao+(l0+0)];
                val += gout41 * dm[(i0+2)*nao+(l0+1)];
                val += gout68 * dm[(i0+2)*nao+(l0+2)];
                atomicAdd(vk+(j0+1)*nao+(k0+1), val);
                val = 0;
                val += gout21 * dm[(i0+0)*nao+(l0+0)];
                val += gout48 * dm[(i0+0)*nao+(l0+1)];
                val += gout75 * dm[(i0+0)*nao+(l0+2)];
                val += gout22 * dm[(i0+1)*nao+(l0+0)];
                val += gout49 * dm[(i0+1)*nao+(l0+1)];
                val += gout76 * dm[(i0+1)*nao+(l0+2)];
                val += gout23 * dm[(i0+2)*nao+(l0+0)];
                val += gout50 * dm[(i0+2)*nao+(l0+1)];
                val += gout77 * dm[(i0+2)*nao+(l0+2)];
                atomicAdd(vk+(j0+1)*nao+(k0+2), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(l0+0)];
                val += gout33 * dm[(i0+0)*nao+(l0+1)];
                val += gout60 * dm[(i0+0)*nao+(l0+2)];
                val += gout7 * dm[(i0+1)*nao+(l0+0)];
                val += gout34 * dm[(i0+1)*nao+(l0+1)];
                val += gout61 * dm[(i0+1)*nao+(l0+2)];
                val += gout8 * dm[(i0+2)*nao+(l0+0)];
                val += gout35 * dm[(i0+2)*nao+(l0+1)];
                val += gout62 * dm[(i0+2)*nao+(l0+2)];
                atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                val = 0;
                val += gout15 * dm[(i0+0)*nao+(l0+0)];
                val += gout42 * dm[(i0+0)*nao+(l0+1)];
                val += gout69 * dm[(i0+0)*nao+(l0+2)];
                val += gout16 * dm[(i0+1)*nao+(l0+0)];
                val += gout43 * dm[(i0+1)*nao+(l0+1)];
                val += gout70 * dm[(i0+1)*nao+(l0+2)];
                val += gout17 * dm[(i0+2)*nao+(l0+0)];
                val += gout44 * dm[(i0+2)*nao+(l0+1)];
                val += gout71 * dm[(i0+2)*nao+(l0+2)];
                atomicAdd(vk+(j0+2)*nao+(k0+1), val);
                val = 0;
                val += gout24 * dm[(i0+0)*nao+(l0+0)];
                val += gout51 * dm[(i0+0)*nao+(l0+1)];
                val += gout78 * dm[(i0+0)*nao+(l0+2)];
                val += gout25 * dm[(i0+1)*nao+(l0+0)];
                val += gout52 * dm[(i0+1)*nao+(l0+1)];
                val += gout79 * dm[(i0+1)*nao+(l0+2)];
                val += gout26 * dm[(i0+2)*nao+(l0+0)];
                val += gout53 * dm[(i0+2)*nao+(l0+1)];
                val += gout80 * dm[(i0+2)*nao+(l0+2)];
                atomicAdd(vk+(j0+2)*nao+(k0+2), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_1111, MINBLOCKS_1111)
k_1111(KARGS)    { kbody_1111<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_1111, MINBLOCKS_1111)
k_rs_1111(KARGS) { kbody_1111<2>(KFWD); }


#define NTHREADS_2011  64
#define MINBLOCKS_2011 4
template <int NRANGE> __device__ __forceinline__ void
kbody_2011(KARGS)
{
    constexpr int NROOTS = 3;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.; double gout6 = 0.; double gout7 = 0.; double gout8 = 0.; double gout9 = 0.; double gout10 = 0.; double gout11 = 0.; double gout12 = 0.; double gout13 = 0.; double gout14 = 0.; double gout15 = 0.; double gout16 = 0.; double gout17 = 0.; double gout18 = 0.; double gout19 = 0.; double gout20 = 0.; double gout21 = 0.; double gout22 = 0.; double gout23 = 0.; double gout24 = 0.; double gout25 = 0.; double gout26 = 0.; double gout27 = 0.; double gout28 = 0.; double gout29 = 0.; double gout30 = 0.; double gout31 = 0.; double gout32 = 0.; double gout33 = 0.; double gout34 = 0.; double gout35 = 0.; double gout36 = 0.; double gout37 = 0.; double gout38 = 0.; double gout39 = 0.; double gout40 = 0.; double gout41 = 0.; double gout42 = 0.; double gout43 = 0.; double gout44 = 0.; double gout45 = 0.; double gout46 = 0.; double gout47 = 0.; double gout48 = 0.; double gout49 = 0.; double gout50 = 0.; double gout51 = 0.; double gout52 = 0.; double gout53 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double b00 = .5 * rt_aa;
                    double rt_akl = rt_aa * aij;
                    double b01 = .5*iakl * (1 - rt_akl);
                    double cpx = xlxk*al_akl + xpq*rt_akl;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                    double hrr_2011x = trr_22x - xlxk * trr_21x;
                    gout0 += hrr_2011x * fac * wt;
                    double trr_01x = cpx * 1;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    double hrr_1011x = trr_12x - xlxk * trr_11x;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += hrr_1011x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += hrr_1011x * fac * trr_10z;
                    double trr_02x = cpx * trr_01x + 1*b01 * 1;
                    double hrr_0011x = trr_02x - xlxk * trr_01x;
                    double trr_20y = c0y * trr_10y + 1*b10 * fac;
                    gout3 += hrr_0011x * trr_20y * wt;
                    gout4 += hrr_0011x * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += hrr_0011x * fac * trr_20z;
                    double hrr_2001x = trr_21x - xlxk * trr_20x;
                    double cpy = ylyk*al_akl + ypq*rt_akl;
                    double trr_01y = cpy * fac;
                    gout6 += hrr_2001x * trr_01y * wt;
                    double hrr_1001x = trr_11x - xlxk * trr_10x;
                    double trr_11y = cpy * trr_10y + 1*b00 * fac;
                    gout7 += hrr_1001x * trr_11y * wt;
                    gout8 += hrr_1001x * trr_01y * trr_10z;
                    double hrr_0001x = trr_01x - xlxk * 1;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    gout9 += hrr_0001x * trr_21y * wt;
                    gout10 += hrr_0001x * trr_11y * trr_10z;
                    gout11 += hrr_0001x * trr_01y * trr_20z;
                    double cpz = zlzk*al_akl + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    gout12 += hrr_2001x * fac * trr_01z;
                    gout13 += hrr_1001x * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout14 += hrr_1001x * fac * trr_11z;
                    gout15 += hrr_0001x * trr_20y * trr_01z;
                    gout16 += hrr_0001x * trr_10y * trr_11z;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    gout17 += hrr_0001x * fac * trr_21z;
                    double hrr_0001y = trr_01y - ylyk * fac;
                    gout18 += trr_21x * hrr_0001y * wt;
                    double hrr_1001y = trr_11y - ylyk * trr_10y;
                    gout19 += trr_11x * hrr_1001y * wt;
                    gout20 += trr_11x * hrr_0001y * trr_10z;
                    double hrr_2001y = trr_21y - ylyk * trr_20y;
                    gout21 += trr_01x * hrr_2001y * wt;
                    gout22 += trr_01x * hrr_1001y * trr_10z;
                    gout23 += trr_01x * hrr_0001y * trr_20z;
                    double trr_02y = cpy * trr_01y + 1*b01 * fac;
                    double hrr_0011y = trr_02y - ylyk * trr_01y;
                    gout24 += trr_20x * hrr_0011y * wt;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    double hrr_1011y = trr_12y - ylyk * trr_11y;
                    gout25 += trr_10x * hrr_1011y * wt;
                    gout26 += trr_10x * hrr_0011y * trr_10z;
                    double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                    double hrr_2011y = trr_22y - ylyk * trr_21y;
                    gout27 += 1 * hrr_2011y * wt;
                    gout28 += 1 * hrr_1011y * trr_10z;
                    gout29 += 1 * hrr_0011y * trr_20z;
                    gout30 += trr_20x * hrr_0001y * trr_01z;
                    gout31 += trr_10x * hrr_1001y * trr_01z;
                    gout32 += trr_10x * hrr_0001y * trr_11z;
                    gout33 += 1 * hrr_2001y * trr_01z;
                    gout34 += 1 * hrr_1001y * trr_11z;
                    gout35 += 1 * hrr_0001y * trr_21z;
                    double hrr_0001z = trr_01z - zlzk * wt;
                    gout36 += trr_21x * fac * hrr_0001z;
                    gout37 += trr_11x * trr_10y * hrr_0001z;
                    double hrr_1001z = trr_11z - zlzk * trr_10z;
                    gout38 += trr_11x * fac * hrr_1001z;
                    gout39 += trr_01x * trr_20y * hrr_0001z;
                    gout40 += trr_01x * trr_10y * hrr_1001z;
                    double hrr_2001z = trr_21z - zlzk * trr_20z;
                    gout41 += trr_01x * fac * hrr_2001z;
                    gout42 += trr_20x * trr_01y * hrr_0001z;
                    gout43 += trr_10x * trr_11y * hrr_0001z;
                    gout44 += trr_10x * trr_01y * hrr_1001z;
                    gout45 += 1 * trr_21y * hrr_0001z;
                    gout46 += 1 * trr_11y * hrr_1001z;
                    gout47 += 1 * trr_01y * hrr_2001z;
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    double hrr_0011z = trr_02z - zlzk * trr_01z;
                    gout48 += trr_20x * fac * hrr_0011z;
                    gout49 += trr_10x * trr_10y * hrr_0011z;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    double hrr_1011z = trr_12z - zlzk * trr_11z;
                    gout50 += trr_10x * fac * hrr_1011z;
                    gout51 += 1 * trr_20y * hrr_0011z;
                    gout52 += 1 * trr_10y * hrr_1011z;
                    double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                    double hrr_2011z = trr_22z - zlzk * trr_21z;
                    gout53 += 1 * fac * hrr_2011z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                val += gout6 * dm[(j0+0)*nao+(k0+1)];
                val += gout12 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout18 * dm[(j0+0)*nao+(k0+0)];
                val += gout24 * dm[(j0+0)*nao+(k0+1)];
                val += gout30 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+0)*nao+(l0+1), val);
                val = 0;
                val += gout36 * dm[(j0+0)*nao+(k0+0)];
                val += gout42 * dm[(j0+0)*nao+(k0+1)];
                val += gout48 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+0)*nao+(l0+2), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                val += gout7 * dm[(j0+0)*nao+(k0+1)];
                val += gout13 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout19 * dm[(j0+0)*nao+(k0+0)];
                val += gout25 * dm[(j0+0)*nao+(k0+1)];
                val += gout31 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+1)*nao+(l0+1), val);
                val = 0;
                val += gout37 * dm[(j0+0)*nao+(k0+0)];
                val += gout43 * dm[(j0+0)*nao+(k0+1)];
                val += gout49 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+1)*nao+(l0+2), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                val += gout8 * dm[(j0+0)*nao+(k0+1)];
                val += gout14 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout20 * dm[(j0+0)*nao+(k0+0)];
                val += gout26 * dm[(j0+0)*nao+(k0+1)];
                val += gout32 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+2)*nao+(l0+1), val);
                val = 0;
                val += gout38 * dm[(j0+0)*nao+(k0+0)];
                val += gout44 * dm[(j0+0)*nao+(k0+1)];
                val += gout50 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+2)*nao+(l0+2), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(k0+0)];
                val += gout9 * dm[(j0+0)*nao+(k0+1)];
                val += gout15 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                val = 0;
                val += gout21 * dm[(j0+0)*nao+(k0+0)];
                val += gout27 * dm[(j0+0)*nao+(k0+1)];
                val += gout33 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+3)*nao+(l0+1), val);
                val = 0;
                val += gout39 * dm[(j0+0)*nao+(k0+0)];
                val += gout45 * dm[(j0+0)*nao+(k0+1)];
                val += gout51 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+3)*nao+(l0+2), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(k0+0)];
                val += gout10 * dm[(j0+0)*nao+(k0+1)];
                val += gout16 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                val = 0;
                val += gout22 * dm[(j0+0)*nao+(k0+0)];
                val += gout28 * dm[(j0+0)*nao+(k0+1)];
                val += gout34 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+4)*nao+(l0+1), val);
                val = 0;
                val += gout40 * dm[(j0+0)*nao+(k0+0)];
                val += gout46 * dm[(j0+0)*nao+(k0+1)];
                val += gout52 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+4)*nao+(l0+2), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(k0+0)];
                val += gout11 * dm[(j0+0)*nao+(k0+1)];
                val += gout17 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                val = 0;
                val += gout23 * dm[(j0+0)*nao+(k0+0)];
                val += gout29 * dm[(j0+0)*nao+(k0+1)];
                val += gout35 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+5)*nao+(l0+1), val);
                val = 0;
                val += gout41 * dm[(j0+0)*nao+(k0+0)];
                val += gout47 * dm[(j0+0)*nao+(k0+1)];
                val += gout53 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+5)*nao+(l0+2), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout6 * dm[(i0+0)*nao+(k0+1)];
                val += gout12 * dm[(i0+0)*nao+(k0+2)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout7 * dm[(i0+1)*nao+(k0+1)];
                val += gout13 * dm[(i0+1)*nao+(k0+2)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                val += gout8 * dm[(i0+2)*nao+(k0+1)];
                val += gout14 * dm[(i0+2)*nao+(k0+2)];
                val += gout3 * dm[(i0+3)*nao+(k0+0)];
                val += gout9 * dm[(i0+3)*nao+(k0+1)];
                val += gout15 * dm[(i0+3)*nao+(k0+2)];
                val += gout4 * dm[(i0+4)*nao+(k0+0)];
                val += gout10 * dm[(i0+4)*nao+(k0+1)];
                val += gout16 * dm[(i0+4)*nao+(k0+2)];
                val += gout5 * dm[(i0+5)*nao+(k0+0)];
                val += gout11 * dm[(i0+5)*nao+(k0+1)];
                val += gout17 * dm[(i0+5)*nao+(k0+2)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout18 * dm[(i0+0)*nao+(k0+0)];
                val += gout24 * dm[(i0+0)*nao+(k0+1)];
                val += gout30 * dm[(i0+0)*nao+(k0+2)];
                val += gout19 * dm[(i0+1)*nao+(k0+0)];
                val += gout25 * dm[(i0+1)*nao+(k0+1)];
                val += gout31 * dm[(i0+1)*nao+(k0+2)];
                val += gout20 * dm[(i0+2)*nao+(k0+0)];
                val += gout26 * dm[(i0+2)*nao+(k0+1)];
                val += gout32 * dm[(i0+2)*nao+(k0+2)];
                val += gout21 * dm[(i0+3)*nao+(k0+0)];
                val += gout27 * dm[(i0+3)*nao+(k0+1)];
                val += gout33 * dm[(i0+3)*nao+(k0+2)];
                val += gout22 * dm[(i0+4)*nao+(k0+0)];
                val += gout28 * dm[(i0+4)*nao+(k0+1)];
                val += gout34 * dm[(i0+4)*nao+(k0+2)];
                val += gout23 * dm[(i0+5)*nao+(k0+0)];
                val += gout29 * dm[(i0+5)*nao+(k0+1)];
                val += gout35 * dm[(i0+5)*nao+(k0+2)];
                atomicAdd(vk+(j0+0)*nao+(l0+1), val);
                val = 0;
                val += gout36 * dm[(i0+0)*nao+(k0+0)];
                val += gout42 * dm[(i0+0)*nao+(k0+1)];
                val += gout48 * dm[(i0+0)*nao+(k0+2)];
                val += gout37 * dm[(i0+1)*nao+(k0+0)];
                val += gout43 * dm[(i0+1)*nao+(k0+1)];
                val += gout49 * dm[(i0+1)*nao+(k0+2)];
                val += gout38 * dm[(i0+2)*nao+(k0+0)];
                val += gout44 * dm[(i0+2)*nao+(k0+1)];
                val += gout50 * dm[(i0+2)*nao+(k0+2)];
                val += gout39 * dm[(i0+3)*nao+(k0+0)];
                val += gout45 * dm[(i0+3)*nao+(k0+1)];
                val += gout51 * dm[(i0+3)*nao+(k0+2)];
                val += gout40 * dm[(i0+4)*nao+(k0+0)];
                val += gout46 * dm[(i0+4)*nao+(k0+1)];
                val += gout52 * dm[(i0+4)*nao+(k0+2)];
                val += gout41 * dm[(i0+5)*nao+(k0+0)];
                val += gout47 * dm[(i0+5)*nao+(k0+1)];
                val += gout53 * dm[(i0+5)*nao+(k0+2)];
                atomicAdd(vk+(j0+0)*nao+(l0+2), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                val += gout18 * dm[(j0+0)*nao+(l0+1)];
                val += gout36 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout6 * dm[(j0+0)*nao+(l0+0)];
                val += gout24 * dm[(j0+0)*nao+(l0+1)];
                val += gout42 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                val = 0;
                val += gout12 * dm[(j0+0)*nao+(l0+0)];
                val += gout30 * dm[(j0+0)*nao+(l0+1)];
                val += gout48 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                val += gout19 * dm[(j0+0)*nao+(l0+1)];
                val += gout37 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout7 * dm[(j0+0)*nao+(l0+0)];
                val += gout25 * dm[(j0+0)*nao+(l0+1)];
                val += gout43 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                val = 0;
                val += gout13 * dm[(j0+0)*nao+(l0+0)];
                val += gout31 * dm[(j0+0)*nao+(l0+1)];
                val += gout49 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                val += gout20 * dm[(j0+0)*nao+(l0+1)];
                val += gout38 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout8 * dm[(j0+0)*nao+(l0+0)];
                val += gout26 * dm[(j0+0)*nao+(l0+1)];
                val += gout44 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                val = 0;
                val += gout14 * dm[(j0+0)*nao+(l0+0)];
                val += gout32 * dm[(j0+0)*nao+(l0+1)];
                val += gout50 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(l0+0)];
                val += gout21 * dm[(j0+0)*nao+(l0+1)];
                val += gout39 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                val = 0;
                val += gout9 * dm[(j0+0)*nao+(l0+0)];
                val += gout27 * dm[(j0+0)*nao+(l0+1)];
                val += gout45 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                val = 0;
                val += gout15 * dm[(j0+0)*nao+(l0+0)];
                val += gout33 * dm[(j0+0)*nao+(l0+1)];
                val += gout51 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(l0+0)];
                val += gout22 * dm[(j0+0)*nao+(l0+1)];
                val += gout40 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                val = 0;
                val += gout10 * dm[(j0+0)*nao+(l0+0)];
                val += gout28 * dm[(j0+0)*nao+(l0+1)];
                val += gout46 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                val = 0;
                val += gout16 * dm[(j0+0)*nao+(l0+0)];
                val += gout34 * dm[(j0+0)*nao+(l0+1)];
                val += gout52 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(l0+0)];
                val += gout23 * dm[(j0+0)*nao+(l0+1)];
                val += gout41 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                val = 0;
                val += gout11 * dm[(j0+0)*nao+(l0+0)];
                val += gout29 * dm[(j0+0)*nao+(l0+1)];
                val += gout47 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                val = 0;
                val += gout17 * dm[(j0+0)*nao+(l0+0)];
                val += gout35 * dm[(j0+0)*nao+(l0+1)];
                val += gout53 * dm[(j0+0)*nao+(l0+2)];
                atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout18 * dm[(i0+0)*nao+(l0+1)];
                val += gout36 * dm[(i0+0)*nao+(l0+2)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout19 * dm[(i0+1)*nao+(l0+1)];
                val += gout37 * dm[(i0+1)*nao+(l0+2)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                val += gout20 * dm[(i0+2)*nao+(l0+1)];
                val += gout38 * dm[(i0+2)*nao+(l0+2)];
                val += gout3 * dm[(i0+3)*nao+(l0+0)];
                val += gout21 * dm[(i0+3)*nao+(l0+1)];
                val += gout39 * dm[(i0+3)*nao+(l0+2)];
                val += gout4 * dm[(i0+4)*nao+(l0+0)];
                val += gout22 * dm[(i0+4)*nao+(l0+1)];
                val += gout40 * dm[(i0+4)*nao+(l0+2)];
                val += gout5 * dm[(i0+5)*nao+(l0+0)];
                val += gout23 * dm[(i0+5)*nao+(l0+1)];
                val += gout41 * dm[(i0+5)*nao+(l0+2)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(l0+0)];
                val += gout24 * dm[(i0+0)*nao+(l0+1)];
                val += gout42 * dm[(i0+0)*nao+(l0+2)];
                val += gout7 * dm[(i0+1)*nao+(l0+0)];
                val += gout25 * dm[(i0+1)*nao+(l0+1)];
                val += gout43 * dm[(i0+1)*nao+(l0+2)];
                val += gout8 * dm[(i0+2)*nao+(l0+0)];
                val += gout26 * dm[(i0+2)*nao+(l0+1)];
                val += gout44 * dm[(i0+2)*nao+(l0+2)];
                val += gout9 * dm[(i0+3)*nao+(l0+0)];
                val += gout27 * dm[(i0+3)*nao+(l0+1)];
                val += gout45 * dm[(i0+3)*nao+(l0+2)];
                val += gout10 * dm[(i0+4)*nao+(l0+0)];
                val += gout28 * dm[(i0+4)*nao+(l0+1)];
                val += gout46 * dm[(i0+4)*nao+(l0+2)];
                val += gout11 * dm[(i0+5)*nao+(l0+0)];
                val += gout29 * dm[(i0+5)*nao+(l0+1)];
                val += gout47 * dm[(i0+5)*nao+(l0+2)];
                atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                val = 0;
                val += gout12 * dm[(i0+0)*nao+(l0+0)];
                val += gout30 * dm[(i0+0)*nao+(l0+1)];
                val += gout48 * dm[(i0+0)*nao+(l0+2)];
                val += gout13 * dm[(i0+1)*nao+(l0+0)];
                val += gout31 * dm[(i0+1)*nao+(l0+1)];
                val += gout49 * dm[(i0+1)*nao+(l0+2)];
                val += gout14 * dm[(i0+2)*nao+(l0+0)];
                val += gout32 * dm[(i0+2)*nao+(l0+1)];
                val += gout50 * dm[(i0+2)*nao+(l0+2)];
                val += gout15 * dm[(i0+3)*nao+(l0+0)];
                val += gout33 * dm[(i0+3)*nao+(l0+1)];
                val += gout51 * dm[(i0+3)*nao+(l0+2)];
                val += gout16 * dm[(i0+4)*nao+(l0+0)];
                val += gout34 * dm[(i0+4)*nao+(l0+1)];
                val += gout52 * dm[(i0+4)*nao+(l0+2)];
                val += gout17 * dm[(i0+5)*nao+(l0+0)];
                val += gout35 * dm[(i0+5)*nao+(l0+1)];
                val += gout53 * dm[(i0+5)*nao+(l0+2)];
                atomicAdd(vk+(j0+0)*nao+(k0+2), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_2011, MINBLOCKS_2011)
k_2011(KARGS)    { kbody_2011<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_2011, MINBLOCKS_2011)
k_rs_2011(KARGS) { kbody_2011<2>(KFWD); }


#define NTHREADS_2020  128
#define MINBLOCKS_2020 2
template <int NRANGE> __device__ __forceinline__ void
kbody_2020(KARGS)
{
    constexpr int NROOTS = 3;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.; double gout6 = 0.; double gout7 = 0.; double gout8 = 0.; double gout9 = 0.; double gout10 = 0.; double gout11 = 0.; double gout12 = 0.; double gout13 = 0.; double gout14 = 0.; double gout15 = 0.; double gout16 = 0.; double gout17 = 0.; double gout18 = 0.; double gout19 = 0.; double gout20 = 0.; double gout21 = 0.; double gout22 = 0.; double gout23 = 0.; double gout24 = 0.; double gout25 = 0.; double gout26 = 0.; double gout27 = 0.; double gout28 = 0.; double gout29 = 0.; double gout30 = 0.; double gout31 = 0.; double gout32 = 0.; double gout33 = 0.; double gout34 = 0.; double gout35 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double b00 = .5 * rt_aa;
                    double rt_akl = rt_aa * aij;
                    double b01 = .5*iakl * (1 - rt_akl);
                    double cpx = xlxk*al_akl + xpq*rt_akl;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                    gout0 += trr_22x * fac * wt;
                    double trr_01x = cpx * 1;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += trr_12x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += trr_12x * fac * trr_10z;
                    double trr_02x = cpx * trr_01x + 1*b01 * 1;
                    double trr_20y = c0y * trr_10y + 1*b10 * fac;
                    gout3 += trr_02x * trr_20y * wt;
                    gout4 += trr_02x * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += trr_02x * fac * trr_20z;
                    double cpy = ylyk*al_akl + ypq*rt_akl;
                    double trr_01y = cpy * fac;
                    gout6 += trr_21x * trr_01y * wt;
                    double trr_11y = cpy * trr_10y + 1*b00 * fac;
                    gout7 += trr_11x * trr_11y * wt;
                    gout8 += trr_11x * trr_01y * trr_10z;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    gout9 += trr_01x * trr_21y * wt;
                    gout10 += trr_01x * trr_11y * trr_10z;
                    gout11 += trr_01x * trr_01y * trr_20z;
                    double cpz = zlzk*al_akl + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    gout12 += trr_21x * fac * trr_01z;
                    gout13 += trr_11x * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout14 += trr_11x * fac * trr_11z;
                    gout15 += trr_01x * trr_20y * trr_01z;
                    gout16 += trr_01x * trr_10y * trr_11z;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    gout17 += trr_01x * fac * trr_21z;
                    double trr_02y = cpy * trr_01y + 1*b01 * fac;
                    gout18 += trr_20x * trr_02y * wt;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    gout19 += trr_10x * trr_12y * wt;
                    gout20 += trr_10x * trr_02y * trr_10z;
                    double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                    gout21 += 1 * trr_22y * wt;
                    gout22 += 1 * trr_12y * trr_10z;
                    gout23 += 1 * trr_02y * trr_20z;
                    gout24 += trr_20x * trr_01y * trr_01z;
                    gout25 += trr_10x * trr_11y * trr_01z;
                    gout26 += trr_10x * trr_01y * trr_11z;
                    gout27 += 1 * trr_21y * trr_01z;
                    gout28 += 1 * trr_11y * trr_11z;
                    gout29 += 1 * trr_01y * trr_21z;
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    gout30 += trr_20x * fac * trr_02z;
                    gout31 += trr_10x * trr_10y * trr_02z;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    gout32 += trr_10x * fac * trr_12z;
                    gout33 += 1 * trr_20y * trr_02z;
                    gout34 += 1 * trr_10y * trr_12z;
                    double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                    gout35 += 1 * fac * trr_22z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                val += gout6 * dm[(j0+0)*nao+(k0+1)];
                val += gout12 * dm[(j0+0)*nao+(k0+2)];
                val += gout18 * dm[(j0+0)*nao+(k0+3)];
                val += gout24 * dm[(j0+0)*nao+(k0+4)];
                val += gout30 * dm[(j0+0)*nao+(k0+5)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                val += gout7 * dm[(j0+0)*nao+(k0+1)];
                val += gout13 * dm[(j0+0)*nao+(k0+2)];
                val += gout19 * dm[(j0+0)*nao+(k0+3)];
                val += gout25 * dm[(j0+0)*nao+(k0+4)];
                val += gout31 * dm[(j0+0)*nao+(k0+5)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                val += gout8 * dm[(j0+0)*nao+(k0+1)];
                val += gout14 * dm[(j0+0)*nao+(k0+2)];
                val += gout20 * dm[(j0+0)*nao+(k0+3)];
                val += gout26 * dm[(j0+0)*nao+(k0+4)];
                val += gout32 * dm[(j0+0)*nao+(k0+5)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(k0+0)];
                val += gout9 * dm[(j0+0)*nao+(k0+1)];
                val += gout15 * dm[(j0+0)*nao+(k0+2)];
                val += gout21 * dm[(j0+0)*nao+(k0+3)];
                val += gout27 * dm[(j0+0)*nao+(k0+4)];
                val += gout33 * dm[(j0+0)*nao+(k0+5)];
                atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(k0+0)];
                val += gout10 * dm[(j0+0)*nao+(k0+1)];
                val += gout16 * dm[(j0+0)*nao+(k0+2)];
                val += gout22 * dm[(j0+0)*nao+(k0+3)];
                val += gout28 * dm[(j0+0)*nao+(k0+4)];
                val += gout34 * dm[(j0+0)*nao+(k0+5)];
                atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(k0+0)];
                val += gout11 * dm[(j0+0)*nao+(k0+1)];
                val += gout17 * dm[(j0+0)*nao+(k0+2)];
                val += gout23 * dm[(j0+0)*nao+(k0+3)];
                val += gout29 * dm[(j0+0)*nao+(k0+4)];
                val += gout35 * dm[(j0+0)*nao+(k0+5)];
                atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout6 * dm[(i0+0)*nao+(k0+1)];
                val += gout12 * dm[(i0+0)*nao+(k0+2)];
                val += gout18 * dm[(i0+0)*nao+(k0+3)];
                val += gout24 * dm[(i0+0)*nao+(k0+4)];
                val += gout30 * dm[(i0+0)*nao+(k0+5)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout7 * dm[(i0+1)*nao+(k0+1)];
                val += gout13 * dm[(i0+1)*nao+(k0+2)];
                val += gout19 * dm[(i0+1)*nao+(k0+3)];
                val += gout25 * dm[(i0+1)*nao+(k0+4)];
                val += gout31 * dm[(i0+1)*nao+(k0+5)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                val += gout8 * dm[(i0+2)*nao+(k0+1)];
                val += gout14 * dm[(i0+2)*nao+(k0+2)];
                val += gout20 * dm[(i0+2)*nao+(k0+3)];
                val += gout26 * dm[(i0+2)*nao+(k0+4)];
                val += gout32 * dm[(i0+2)*nao+(k0+5)];
                val += gout3 * dm[(i0+3)*nao+(k0+0)];
                val += gout9 * dm[(i0+3)*nao+(k0+1)];
                val += gout15 * dm[(i0+3)*nao+(k0+2)];
                val += gout21 * dm[(i0+3)*nao+(k0+3)];
                val += gout27 * dm[(i0+3)*nao+(k0+4)];
                val += gout33 * dm[(i0+3)*nao+(k0+5)];
                val += gout4 * dm[(i0+4)*nao+(k0+0)];
                val += gout10 * dm[(i0+4)*nao+(k0+1)];
                val += gout16 * dm[(i0+4)*nao+(k0+2)];
                val += gout22 * dm[(i0+4)*nao+(k0+3)];
                val += gout28 * dm[(i0+4)*nao+(k0+4)];
                val += gout34 * dm[(i0+4)*nao+(k0+5)];
                val += gout5 * dm[(i0+5)*nao+(k0+0)];
                val += gout11 * dm[(i0+5)*nao+(k0+1)];
                val += gout17 * dm[(i0+5)*nao+(k0+2)];
                val += gout23 * dm[(i0+5)*nao+(k0+3)];
                val += gout29 * dm[(i0+5)*nao+(k0+4)];
                val += gout35 * dm[(i0+5)*nao+(k0+5)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout6 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                val = 0;
                val += gout12 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                val = 0;
                val += gout18 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+3), val);
                val = 0;
                val += gout24 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+4), val);
                val = 0;
                val += gout30 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+5), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout7 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                val = 0;
                val += gout13 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                val = 0;
                val += gout19 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+3), val);
                val = 0;
                val += gout25 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+4), val);
                val = 0;
                val += gout31 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+5), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout8 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                val = 0;
                val += gout14 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                val = 0;
                val += gout20 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+3), val);
                val = 0;
                val += gout26 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+4), val);
                val = 0;
                val += gout32 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+5), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                val = 0;
                val += gout9 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                val = 0;
                val += gout15 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                val = 0;
                val += gout21 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+3), val);
                val = 0;
                val += gout27 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+4), val);
                val = 0;
                val += gout33 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+5), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                val = 0;
                val += gout10 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                val = 0;
                val += gout16 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                val = 0;
                val += gout22 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+3), val);
                val = 0;
                val += gout28 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+4), val);
                val = 0;
                val += gout34 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+5), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                val = 0;
                val += gout11 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                val = 0;
                val += gout17 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                val = 0;
                val += gout23 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+3), val);
                val = 0;
                val += gout29 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+4), val);
                val = 0;
                val += gout35 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+5), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                val += gout3 * dm[(i0+3)*nao+(l0+0)];
                val += gout4 * dm[(i0+4)*nao+(l0+0)];
                val += gout5 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(l0+0)];
                val += gout7 * dm[(i0+1)*nao+(l0+0)];
                val += gout8 * dm[(i0+2)*nao+(l0+0)];
                val += gout9 * dm[(i0+3)*nao+(l0+0)];
                val += gout10 * dm[(i0+4)*nao+(l0+0)];
                val += gout11 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                val = 0;
                val += gout12 * dm[(i0+0)*nao+(l0+0)];
                val += gout13 * dm[(i0+1)*nao+(l0+0)];
                val += gout14 * dm[(i0+2)*nao+(l0+0)];
                val += gout15 * dm[(i0+3)*nao+(l0+0)];
                val += gout16 * dm[(i0+4)*nao+(l0+0)];
                val += gout17 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                val = 0;
                val += gout18 * dm[(i0+0)*nao+(l0+0)];
                val += gout19 * dm[(i0+1)*nao+(l0+0)];
                val += gout20 * dm[(i0+2)*nao+(l0+0)];
                val += gout21 * dm[(i0+3)*nao+(l0+0)];
                val += gout22 * dm[(i0+4)*nao+(l0+0)];
                val += gout23 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+3), val);
                val = 0;
                val += gout24 * dm[(i0+0)*nao+(l0+0)];
                val += gout25 * dm[(i0+1)*nao+(l0+0)];
                val += gout26 * dm[(i0+2)*nao+(l0+0)];
                val += gout27 * dm[(i0+3)*nao+(l0+0)];
                val += gout28 * dm[(i0+4)*nao+(l0+0)];
                val += gout29 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+4), val);
                val = 0;
                val += gout30 * dm[(i0+0)*nao+(l0+0)];
                val += gout31 * dm[(i0+1)*nao+(l0+0)];
                val += gout32 * dm[(i0+2)*nao+(l0+0)];
                val += gout33 * dm[(i0+3)*nao+(l0+0)];
                val += gout34 * dm[(i0+4)*nao+(l0+0)];
                val += gout35 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+5), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_2020, MINBLOCKS_2020)
k_2020(KARGS)    { kbody_2020<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_2020, MINBLOCKS_2020)
k_rs_2020(KARGS) { kbody_2020<2>(KFWD); }


#define NTHREADS_2200  256
#define MINBLOCKS_2200 2
template <int NRANGE> __device__ __forceinline__ void
kbody_2200(KARGS)
{
    constexpr int NROOTS = 3;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.; double gout6 = 0.; double gout7 = 0.; double gout8 = 0.; double gout9 = 0.; double gout10 = 0.; double gout11 = 0.; double gout12 = 0.; double gout13 = 0.; double gout14 = 0.; double gout15 = 0.; double gout16 = 0.; double gout17 = 0.; double gout18 = 0.; double gout19 = 0.; double gout20 = 0.; double gout21 = 0.; double gout22 = 0.; double gout23 = 0.; double gout24 = 0.; double gout25 = 0.; double gout26 = 0.; double gout27 = 0.; double gout28 = 0.; double gout29 = 0.; double gout30 = 0.; double gout31 = 0.; double gout32 = 0.; double gout33 = 0.; double gout34 = 0.; double gout35 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                    double hrr_3100x = trr_40x - xjxi * trr_30x;
                    double hrr_2100x = trr_30x - xjxi * trr_20x;
                    double hrr_2200x = hrr_3100x - xjxi * hrr_2100x;
                    gout0 += hrr_2200x * fac * wt;
                    double hrr_1100x = trr_20x - xjxi * trr_10x;
                    double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += hrr_1200x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += hrr_1200x * fac * trr_10z;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                    double trr_20y = c0y * trr_10y + 1*b10 * fac;
                    gout3 += hrr_0200x * trr_20y * wt;
                    gout4 += hrr_0200x * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += hrr_0200x * fac * trr_20z;
                    double hrr_0100y = trr_10y - yjyi * fac;
                    gout6 += hrr_2100x * hrr_0100y * wt;
                    double hrr_1100y = trr_20y - yjyi * trr_10y;
                    gout7 += hrr_1100x * hrr_1100y * wt;
                    gout8 += hrr_1100x * hrr_0100y * trr_10z;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    double hrr_2100y = trr_30y - yjyi * trr_20y;
                    gout9 += hrr_0100x * hrr_2100y * wt;
                    gout10 += hrr_0100x * hrr_1100y * trr_10z;
                    gout11 += hrr_0100x * hrr_0100y * trr_20z;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    gout12 += hrr_2100x * fac * hrr_0100z;
                    gout13 += hrr_1100x * trr_10y * hrr_0100z;
                    double hrr_1100z = trr_20z - zjzi * trr_10z;
                    gout14 += hrr_1100x * fac * hrr_1100z;
                    gout15 += hrr_0100x * trr_20y * hrr_0100z;
                    gout16 += hrr_0100x * trr_10y * hrr_1100z;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    double hrr_2100z = trr_30z - zjzi * trr_20z;
                    gout17 += hrr_0100x * fac * hrr_2100z;
                    double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                    gout18 += trr_20x * hrr_0200y * wt;
                    double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                    gout19 += trr_10x * hrr_1200y * wt;
                    gout20 += trr_10x * hrr_0200y * trr_10z;
                    double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                    double hrr_3100y = trr_40y - yjyi * trr_30y;
                    double hrr_2200y = hrr_3100y - yjyi * hrr_2100y;
                    gout21 += 1 * hrr_2200y * wt;
                    gout22 += 1 * hrr_1200y * trr_10z;
                    gout23 += 1 * hrr_0200y * trr_20z;
                    gout24 += trr_20x * hrr_0100y * hrr_0100z;
                    gout25 += trr_10x * hrr_1100y * hrr_0100z;
                    gout26 += trr_10x * hrr_0100y * hrr_1100z;
                    gout27 += 1 * hrr_2100y * hrr_0100z;
                    gout28 += 1 * hrr_1100y * hrr_1100z;
                    gout29 += 1 * hrr_0100y * hrr_2100z;
                    double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                    gout30 += trr_20x * fac * hrr_0200z;
                    gout31 += trr_10x * trr_10y * hrr_0200z;
                    double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                    gout32 += trr_10x * fac * hrr_1200z;
                    gout33 += 1 * trr_20y * hrr_0200z;
                    gout34 += 1 * trr_10y * hrr_1200z;
                    double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                    double hrr_3100z = trr_40z - zjzi * trr_30z;
                    double hrr_2200z = hrr_3100z - zjzi * hrr_2100z;
                    gout35 += 1 * fac * hrr_2200z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                val += gout6 * dm[(j0+1)*nao+(k0+0)];
                val += gout12 * dm[(j0+2)*nao+(k0+0)];
                val += gout18 * dm[(j0+3)*nao+(k0+0)];
                val += gout24 * dm[(j0+4)*nao+(k0+0)];
                val += gout30 * dm[(j0+5)*nao+(k0+0)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                val += gout7 * dm[(j0+1)*nao+(k0+0)];
                val += gout13 * dm[(j0+2)*nao+(k0+0)];
                val += gout19 * dm[(j0+3)*nao+(k0+0)];
                val += gout25 * dm[(j0+4)*nao+(k0+0)];
                val += gout31 * dm[(j0+5)*nao+(k0+0)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                val += gout8 * dm[(j0+1)*nao+(k0+0)];
                val += gout14 * dm[(j0+2)*nao+(k0+0)];
                val += gout20 * dm[(j0+3)*nao+(k0+0)];
                val += gout26 * dm[(j0+4)*nao+(k0+0)];
                val += gout32 * dm[(j0+5)*nao+(k0+0)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(k0+0)];
                val += gout9 * dm[(j0+1)*nao+(k0+0)];
                val += gout15 * dm[(j0+2)*nao+(k0+0)];
                val += gout21 * dm[(j0+3)*nao+(k0+0)];
                val += gout27 * dm[(j0+4)*nao+(k0+0)];
                val += gout33 * dm[(j0+5)*nao+(k0+0)];
                atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(k0+0)];
                val += gout10 * dm[(j0+1)*nao+(k0+0)];
                val += gout16 * dm[(j0+2)*nao+(k0+0)];
                val += gout22 * dm[(j0+3)*nao+(k0+0)];
                val += gout28 * dm[(j0+4)*nao+(k0+0)];
                val += gout34 * dm[(j0+5)*nao+(k0+0)];
                atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(k0+0)];
                val += gout11 * dm[(j0+1)*nao+(k0+0)];
                val += gout17 * dm[(j0+2)*nao+(k0+0)];
                val += gout23 * dm[(j0+3)*nao+(k0+0)];
                val += gout29 * dm[(j0+4)*nao+(k0+0)];
                val += gout35 * dm[(j0+5)*nao+(k0+0)];
                atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                val += gout3 * dm[(i0+3)*nao+(k0+0)];
                val += gout4 * dm[(i0+4)*nao+(k0+0)];
                val += gout5 * dm[(i0+5)*nao+(k0+0)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(k0+0)];
                val += gout7 * dm[(i0+1)*nao+(k0+0)];
                val += gout8 * dm[(i0+2)*nao+(k0+0)];
                val += gout9 * dm[(i0+3)*nao+(k0+0)];
                val += gout10 * dm[(i0+4)*nao+(k0+0)];
                val += gout11 * dm[(i0+5)*nao+(k0+0)];
                atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                val = 0;
                val += gout12 * dm[(i0+0)*nao+(k0+0)];
                val += gout13 * dm[(i0+1)*nao+(k0+0)];
                val += gout14 * dm[(i0+2)*nao+(k0+0)];
                val += gout15 * dm[(i0+3)*nao+(k0+0)];
                val += gout16 * dm[(i0+4)*nao+(k0+0)];
                val += gout17 * dm[(i0+5)*nao+(k0+0)];
                atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                val = 0;
                val += gout18 * dm[(i0+0)*nao+(k0+0)];
                val += gout19 * dm[(i0+1)*nao+(k0+0)];
                val += gout20 * dm[(i0+2)*nao+(k0+0)];
                val += gout21 * dm[(i0+3)*nao+(k0+0)];
                val += gout22 * dm[(i0+4)*nao+(k0+0)];
                val += gout23 * dm[(i0+5)*nao+(k0+0)];
                atomicAdd(vk+(j0+3)*nao+(l0+0), val);
                val = 0;
                val += gout24 * dm[(i0+0)*nao+(k0+0)];
                val += gout25 * dm[(i0+1)*nao+(k0+0)];
                val += gout26 * dm[(i0+2)*nao+(k0+0)];
                val += gout27 * dm[(i0+3)*nao+(k0+0)];
                val += gout28 * dm[(i0+4)*nao+(k0+0)];
                val += gout29 * dm[(i0+5)*nao+(k0+0)];
                atomicAdd(vk+(j0+4)*nao+(l0+0), val);
                val = 0;
                val += gout30 * dm[(i0+0)*nao+(k0+0)];
                val += gout31 * dm[(i0+1)*nao+(k0+0)];
                val += gout32 * dm[(i0+2)*nao+(k0+0)];
                val += gout33 * dm[(i0+3)*nao+(k0+0)];
                val += gout34 * dm[(i0+4)*nao+(k0+0)];
                val += gout35 * dm[(i0+5)*nao+(k0+0)];
                atomicAdd(vk+(j0+5)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                val += gout6 * dm[(j0+1)*nao+(l0+0)];
                val += gout12 * dm[(j0+2)*nao+(l0+0)];
                val += gout18 * dm[(j0+3)*nao+(l0+0)];
                val += gout24 * dm[(j0+4)*nao+(l0+0)];
                val += gout30 * dm[(j0+5)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                val += gout7 * dm[(j0+1)*nao+(l0+0)];
                val += gout13 * dm[(j0+2)*nao+(l0+0)];
                val += gout19 * dm[(j0+3)*nao+(l0+0)];
                val += gout25 * dm[(j0+4)*nao+(l0+0)];
                val += gout31 * dm[(j0+5)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                val += gout8 * dm[(j0+1)*nao+(l0+0)];
                val += gout14 * dm[(j0+2)*nao+(l0+0)];
                val += gout20 * dm[(j0+3)*nao+(l0+0)];
                val += gout26 * dm[(j0+4)*nao+(l0+0)];
                val += gout32 * dm[(j0+5)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(l0+0)];
                val += gout9 * dm[(j0+1)*nao+(l0+0)];
                val += gout15 * dm[(j0+2)*nao+(l0+0)];
                val += gout21 * dm[(j0+3)*nao+(l0+0)];
                val += gout27 * dm[(j0+4)*nao+(l0+0)];
                val += gout33 * dm[(j0+5)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(l0+0)];
                val += gout10 * dm[(j0+1)*nao+(l0+0)];
                val += gout16 * dm[(j0+2)*nao+(l0+0)];
                val += gout22 * dm[(j0+3)*nao+(l0+0)];
                val += gout28 * dm[(j0+4)*nao+(l0+0)];
                val += gout34 * dm[(j0+5)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(l0+0)];
                val += gout11 * dm[(j0+1)*nao+(l0+0)];
                val += gout17 * dm[(j0+2)*nao+(l0+0)];
                val += gout23 * dm[(j0+3)*nao+(l0+0)];
                val += gout29 * dm[(j0+4)*nao+(l0+0)];
                val += gout35 * dm[(j0+5)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                val += gout3 * dm[(i0+3)*nao+(l0+0)];
                val += gout4 * dm[(i0+4)*nao+(l0+0)];
                val += gout5 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                val = 0;
                val += gout6 * dm[(i0+0)*nao+(l0+0)];
                val += gout7 * dm[(i0+1)*nao+(l0+0)];
                val += gout8 * dm[(i0+2)*nao+(l0+0)];
                val += gout9 * dm[(i0+3)*nao+(l0+0)];
                val += gout10 * dm[(i0+4)*nao+(l0+0)];
                val += gout11 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                val = 0;
                val += gout12 * dm[(i0+0)*nao+(l0+0)];
                val += gout13 * dm[(i0+1)*nao+(l0+0)];
                val += gout14 * dm[(i0+2)*nao+(l0+0)];
                val += gout15 * dm[(i0+3)*nao+(l0+0)];
                val += gout16 * dm[(i0+4)*nao+(l0+0)];
                val += gout17 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                val = 0;
                val += gout18 * dm[(i0+0)*nao+(l0+0)];
                val += gout19 * dm[(i0+1)*nao+(l0+0)];
                val += gout20 * dm[(i0+2)*nao+(l0+0)];
                val += gout21 * dm[(i0+3)*nao+(l0+0)];
                val += gout22 * dm[(i0+4)*nao+(l0+0)];
                val += gout23 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+3)*nao+(k0+0), val);
                val = 0;
                val += gout24 * dm[(i0+0)*nao+(l0+0)];
                val += gout25 * dm[(i0+1)*nao+(l0+0)];
                val += gout26 * dm[(i0+2)*nao+(l0+0)];
                val += gout27 * dm[(i0+3)*nao+(l0+0)];
                val += gout28 * dm[(i0+4)*nao+(l0+0)];
                val += gout29 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+4)*nao+(k0+0), val);
                val = 0;
                val += gout30 * dm[(i0+0)*nao+(l0+0)];
                val += gout31 * dm[(i0+1)*nao+(l0+0)];
                val += gout32 * dm[(i0+2)*nao+(l0+0)];
                val += gout33 * dm[(i0+3)*nao+(l0+0)];
                val += gout34 * dm[(i0+4)*nao+(l0+0)];
                val += gout35 * dm[(i0+5)*nao+(l0+0)];
                atomicAdd(vk+(j0+5)*nao+(k0+0), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_2200, MINBLOCKS_2200)
k_2200(KARGS)    { kbody_2200<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_2200, MINBLOCKS_2200)
k_rs_2200(KARGS) { kbody_2200<2>(KFWD); }


#define NTHREADS_3000  256
#define MINBLOCKS_3000 3
template <int NRANGE> __device__ __forceinline__ void
kbody_3000(KARGS)
{
    constexpr int NROOTS = 2;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.; double gout6 = 0.; double gout7 = 0.; double gout8 = 0.; double gout9 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    gout0 += trr_30x * fac * wt;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += trr_20x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += trr_20x * fac * trr_10z;
                    double trr_20y = c0y * trr_10y + 1*b10 * fac;
                    gout3 += trr_10x * trr_20y * wt;
                    gout4 += trr_10x * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += trr_10x * fac * trr_20z;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    gout6 += 1 * trr_30y * wt;
                    gout7 += 1 * trr_20y * trr_10z;
                    gout8 += 1 * trr_10y * trr_20z;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    gout9 += 1 * fac * trr_30z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                val = 0;
                val += gout6 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+6)*nao+(l0+0), val);
                val = 0;
                val += gout7 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+7)*nao+(l0+0), val);
                val = 0;
                val += gout8 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+8)*nao+(l0+0), val);
                val = 0;
                val += gout9 * dm[(j0+0)*nao+(k0+0)];
                atomicAdd(vk+(i0+9)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                val += gout3 * dm[(i0+3)*nao+(k0+0)];
                val += gout4 * dm[(i0+4)*nao+(k0+0)];
                val += gout5 * dm[(i0+5)*nao+(k0+0)];
                val += gout6 * dm[(i0+6)*nao+(k0+0)];
                val += gout7 * dm[(i0+7)*nao+(k0+0)];
                val += gout8 * dm[(i0+8)*nao+(k0+0)];
                val += gout9 * dm[(i0+9)*nao+(k0+0)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                val = 0;
                val += gout6 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+6)*nao+(k0+0), val);
                val = 0;
                val += gout7 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+7)*nao+(k0+0), val);
                val = 0;
                val += gout8 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+8)*nao+(k0+0), val);
                val = 0;
                val += gout9 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+9)*nao+(k0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                val += gout3 * dm[(i0+3)*nao+(l0+0)];
                val += gout4 * dm[(i0+4)*nao+(l0+0)];
                val += gout5 * dm[(i0+5)*nao+(l0+0)];
                val += gout6 * dm[(i0+6)*nao+(l0+0)];
                val += gout7 * dm[(i0+7)*nao+(l0+0)];
                val += gout8 * dm[(i0+8)*nao+(l0+0)];
                val += gout9 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_3000, MINBLOCKS_3000)
k_3000(KARGS)    { kbody_3000<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_3000, MINBLOCKS_3000)
k_rs_3000(KARGS) { kbody_3000<2>(KFWD); }


#define NTHREADS_3010  256
#define MINBLOCKS_3010 2
template <int NRANGE> __device__ __forceinline__ void
kbody_3010(KARGS)
{
    constexpr int NROOTS = 3;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.; double gout6 = 0.; double gout7 = 0.; double gout8 = 0.; double gout9 = 0.; double gout10 = 0.; double gout11 = 0.; double gout12 = 0.; double gout13 = 0.; double gout14 = 0.; double gout15 = 0.; double gout16 = 0.; double gout17 = 0.; double gout18 = 0.; double gout19 = 0.; double gout20 = 0.; double gout21 = 0.; double gout22 = 0.; double gout23 = 0.; double gout24 = 0.; double gout25 = 0.; double gout26 = 0.; double gout27 = 0.; double gout28 = 0.; double gout29 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double b00 = .5 * rt_aa;
                    double rt_akl = rt_aa * aij;
                    double cpx = xlxk*al_akl + xpq*rt_akl;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                    gout0 += trr_31x * fac * wt;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += trr_21x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += trr_21x * fac * trr_10z;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double trr_20y = c0y * trr_10y + 1*b10 * fac;
                    gout3 += trr_11x * trr_20y * wt;
                    gout4 += trr_11x * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += trr_11x * fac * trr_20z;
                    double trr_01x = cpx * 1;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    gout6 += trr_01x * trr_30y * wt;
                    gout7 += trr_01x * trr_20y * trr_10z;
                    gout8 += trr_01x * trr_10y * trr_20z;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    gout9 += trr_01x * fac * trr_30z;
                    double cpy = ylyk*al_akl + ypq*rt_akl;
                    double trr_01y = cpy * fac;
                    gout10 += trr_30x * trr_01y * wt;
                    double trr_11y = cpy * trr_10y + 1*b00 * fac;
                    gout11 += trr_20x * trr_11y * wt;
                    gout12 += trr_20x * trr_01y * trr_10z;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    gout13 += trr_10x * trr_21y * wt;
                    gout14 += trr_10x * trr_11y * trr_10z;
                    gout15 += trr_10x * trr_01y * trr_20z;
                    double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                    gout16 += 1 * trr_31y * wt;
                    gout17 += 1 * trr_21y * trr_10z;
                    gout18 += 1 * trr_11y * trr_20z;
                    gout19 += 1 * trr_01y * trr_30z;
                    double cpz = zlzk*al_akl + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    gout20 += trr_30x * fac * trr_01z;
                    gout21 += trr_20x * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout22 += trr_20x * fac * trr_11z;
                    gout23 += trr_10x * trr_20y * trr_01z;
                    gout24 += trr_10x * trr_10y * trr_11z;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    gout25 += trr_10x * fac * trr_21z;
                    gout26 += 1 * trr_30y * trr_01z;
                    gout27 += 1 * trr_20y * trr_11z;
                    gout28 += 1 * trr_10y * trr_21z;
                    double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                    gout29 += 1 * fac * trr_31z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                val += gout10 * dm[(j0+0)*nao+(k0+1)];
                val += gout20 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                val += gout11 * dm[(j0+0)*nao+(k0+1)];
                val += gout21 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                val += gout12 * dm[(j0+0)*nao+(k0+1)];
                val += gout22 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(k0+0)];
                val += gout13 * dm[(j0+0)*nao+(k0+1)];
                val += gout23 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(k0+0)];
                val += gout14 * dm[(j0+0)*nao+(k0+1)];
                val += gout24 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(k0+0)];
                val += gout15 * dm[(j0+0)*nao+(k0+1)];
                val += gout25 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                val = 0;
                val += gout6 * dm[(j0+0)*nao+(k0+0)];
                val += gout16 * dm[(j0+0)*nao+(k0+1)];
                val += gout26 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+6)*nao+(l0+0), val);
                val = 0;
                val += gout7 * dm[(j0+0)*nao+(k0+0)];
                val += gout17 * dm[(j0+0)*nao+(k0+1)];
                val += gout27 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+7)*nao+(l0+0), val);
                val = 0;
                val += gout8 * dm[(j0+0)*nao+(k0+0)];
                val += gout18 * dm[(j0+0)*nao+(k0+1)];
                val += gout28 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+8)*nao+(l0+0), val);
                val = 0;
                val += gout9 * dm[(j0+0)*nao+(k0+0)];
                val += gout19 * dm[(j0+0)*nao+(k0+1)];
                val += gout29 * dm[(j0+0)*nao+(k0+2)];
                atomicAdd(vk+(i0+9)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout10 * dm[(i0+0)*nao+(k0+1)];
                val += gout20 * dm[(i0+0)*nao+(k0+2)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout11 * dm[(i0+1)*nao+(k0+1)];
                val += gout21 * dm[(i0+1)*nao+(k0+2)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                val += gout12 * dm[(i0+2)*nao+(k0+1)];
                val += gout22 * dm[(i0+2)*nao+(k0+2)];
                val += gout3 * dm[(i0+3)*nao+(k0+0)];
                val += gout13 * dm[(i0+3)*nao+(k0+1)];
                val += gout23 * dm[(i0+3)*nao+(k0+2)];
                val += gout4 * dm[(i0+4)*nao+(k0+0)];
                val += gout14 * dm[(i0+4)*nao+(k0+1)];
                val += gout24 * dm[(i0+4)*nao+(k0+2)];
                val += gout5 * dm[(i0+5)*nao+(k0+0)];
                val += gout15 * dm[(i0+5)*nao+(k0+1)];
                val += gout25 * dm[(i0+5)*nao+(k0+2)];
                val += gout6 * dm[(i0+6)*nao+(k0+0)];
                val += gout16 * dm[(i0+6)*nao+(k0+1)];
                val += gout26 * dm[(i0+6)*nao+(k0+2)];
                val += gout7 * dm[(i0+7)*nao+(k0+0)];
                val += gout17 * dm[(i0+7)*nao+(k0+1)];
                val += gout27 * dm[(i0+7)*nao+(k0+2)];
                val += gout8 * dm[(i0+8)*nao+(k0+0)];
                val += gout18 * dm[(i0+8)*nao+(k0+1)];
                val += gout28 * dm[(i0+8)*nao+(k0+2)];
                val += gout9 * dm[(i0+9)*nao+(k0+0)];
                val += gout19 * dm[(i0+9)*nao+(k0+1)];
                val += gout29 * dm[(i0+9)*nao+(k0+2)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout10 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                val = 0;
                val += gout20 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout11 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                val = 0;
                val += gout21 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout12 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                val = 0;
                val += gout22 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                val = 0;
                val += gout13 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                val = 0;
                val += gout23 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                val = 0;
                val += gout14 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                val = 0;
                val += gout24 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                val = 0;
                val += gout15 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                val = 0;
                val += gout25 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                val = 0;
                val += gout6 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+6)*nao+(k0+0), val);
                val = 0;
                val += gout16 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+6)*nao+(k0+1), val);
                val = 0;
                val += gout26 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+6)*nao+(k0+2), val);
                val = 0;
                val += gout7 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+7)*nao+(k0+0), val);
                val = 0;
                val += gout17 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+7)*nao+(k0+1), val);
                val = 0;
                val += gout27 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+7)*nao+(k0+2), val);
                val = 0;
                val += gout8 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+8)*nao+(k0+0), val);
                val = 0;
                val += gout18 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+8)*nao+(k0+1), val);
                val = 0;
                val += gout28 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+8)*nao+(k0+2), val);
                val = 0;
                val += gout9 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+9)*nao+(k0+0), val);
                val = 0;
                val += gout19 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+9)*nao+(k0+1), val);
                val = 0;
                val += gout29 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+9)*nao+(k0+2), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                val += gout3 * dm[(i0+3)*nao+(l0+0)];
                val += gout4 * dm[(i0+4)*nao+(l0+0)];
                val += gout5 * dm[(i0+5)*nao+(l0+0)];
                val += gout6 * dm[(i0+6)*nao+(l0+0)];
                val += gout7 * dm[(i0+7)*nao+(l0+0)];
                val += gout8 * dm[(i0+8)*nao+(l0+0)];
                val += gout9 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                val = 0;
                val += gout10 * dm[(i0+0)*nao+(l0+0)];
                val += gout11 * dm[(i0+1)*nao+(l0+0)];
                val += gout12 * dm[(i0+2)*nao+(l0+0)];
                val += gout13 * dm[(i0+3)*nao+(l0+0)];
                val += gout14 * dm[(i0+4)*nao+(l0+0)];
                val += gout15 * dm[(i0+5)*nao+(l0+0)];
                val += gout16 * dm[(i0+6)*nao+(l0+0)];
                val += gout17 * dm[(i0+7)*nao+(l0+0)];
                val += gout18 * dm[(i0+8)*nao+(l0+0)];
                val += gout19 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                val = 0;
                val += gout20 * dm[(i0+0)*nao+(l0+0)];
                val += gout21 * dm[(i0+1)*nao+(l0+0)];
                val += gout22 * dm[(i0+2)*nao+(l0+0)];
                val += gout23 * dm[(i0+3)*nao+(l0+0)];
                val += gout24 * dm[(i0+4)*nao+(l0+0)];
                val += gout25 * dm[(i0+5)*nao+(l0+0)];
                val += gout26 * dm[(i0+6)*nao+(l0+0)];
                val += gout27 * dm[(i0+7)*nao+(l0+0)];
                val += gout28 * dm[(i0+8)*nao+(l0+0)];
                val += gout29 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+2), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_3010, MINBLOCKS_3010)
k_3010(KARGS)    { kbody_3010<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_3010, MINBLOCKS_3010)
k_rs_3010(KARGS) { kbody_3010<2>(KFWD); }


#define NTHREADS_3020  128
#define MINBLOCKS_3020 2
template <int NRANGE> __device__ __forceinline__ void
kbody_3020(KARGS)
{
    constexpr int NROOTS = 3;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.; double gout6 = 0.; double gout7 = 0.; double gout8 = 0.; double gout9 = 0.; double gout10 = 0.; double gout11 = 0.; double gout12 = 0.; double gout13 = 0.; double gout14 = 0.; double gout15 = 0.; double gout16 = 0.; double gout17 = 0.; double gout18 = 0.; double gout19 = 0.; double gout20 = 0.; double gout21 = 0.; double gout22 = 0.; double gout23 = 0.; double gout24 = 0.; double gout25 = 0.; double gout26 = 0.; double gout27 = 0.; double gout28 = 0.; double gout29 = 0.; double gout30 = 0.; double gout31 = 0.; double gout32 = 0.; double gout33 = 0.; double gout34 = 0.; double gout35 = 0.; double gout36 = 0.; double gout37 = 0.; double gout38 = 0.; double gout39 = 0.; double gout40 = 0.; double gout41 = 0.; double gout42 = 0.; double gout43 = 0.; double gout44 = 0.; double gout45 = 0.; double gout46 = 0.; double gout47 = 0.; double gout48 = 0.; double gout49 = 0.; double gout50 = 0.; double gout51 = 0.; double gout52 = 0.; double gout53 = 0.; double gout54 = 0.; double gout55 = 0.; double gout56 = 0.; double gout57 = 0.; double gout58 = 0.; double gout59 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double b00 = .5 * rt_aa;
                    double rt_akl = rt_aa * aij;
                    double b01 = .5*iakl * (1 - rt_akl);
                    double cpx = xlxk*al_akl + xpq*rt_akl;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    double trr_31x = cpx * trr_30x + 3*b00 * trr_20x;
                    double trr_21x = cpx * trr_20x + 2*b00 * trr_10x;
                    double trr_32x = cpx * trr_31x + 1*b01 * trr_30x + 3*b00 * trr_21x;
                    gout0 += trr_32x * fac * wt;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    double trr_22x = cpx * trr_21x + 1*b01 * trr_20x + 2*b00 * trr_11x;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += trr_22x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += trr_22x * fac * trr_10z;
                    double trr_01x = cpx * 1;
                    double trr_12x = cpx * trr_11x + 1*b01 * trr_10x + 1*b00 * trr_01x;
                    double trr_20y = c0y * trr_10y + 1*b10 * fac;
                    gout3 += trr_12x * trr_20y * wt;
                    gout4 += trr_12x * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += trr_12x * fac * trr_20z;
                    double trr_02x = cpx * trr_01x + 1*b01 * 1;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    gout6 += trr_02x * trr_30y * wt;
                    gout7 += trr_02x * trr_20y * trr_10z;
                    gout8 += trr_02x * trr_10y * trr_20z;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    gout9 += trr_02x * fac * trr_30z;
                    double cpy = ylyk*al_akl + ypq*rt_akl;
                    double trr_01y = cpy * fac;
                    gout10 += trr_31x * trr_01y * wt;
                    double trr_11y = cpy * trr_10y + 1*b00 * fac;
                    gout11 += trr_21x * trr_11y * wt;
                    gout12 += trr_21x * trr_01y * trr_10z;
                    double trr_21y = cpy * trr_20y + 2*b00 * trr_10y;
                    gout13 += trr_11x * trr_21y * wt;
                    gout14 += trr_11x * trr_11y * trr_10z;
                    gout15 += trr_11x * trr_01y * trr_20z;
                    double trr_31y = cpy * trr_30y + 3*b00 * trr_20y;
                    gout16 += trr_01x * trr_31y * wt;
                    gout17 += trr_01x * trr_21y * trr_10z;
                    gout18 += trr_01x * trr_11y * trr_20z;
                    gout19 += trr_01x * trr_01y * trr_30z;
                    double cpz = zlzk*al_akl + zpq*rt_akl;
                    double trr_01z = cpz * wt;
                    gout20 += trr_31x * fac * trr_01z;
                    gout21 += trr_21x * trr_10y * trr_01z;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    gout22 += trr_21x * fac * trr_11z;
                    gout23 += trr_11x * trr_20y * trr_01z;
                    gout24 += trr_11x * trr_10y * trr_11z;
                    double trr_21z = cpz * trr_20z + 2*b00 * trr_10z;
                    gout25 += trr_11x * fac * trr_21z;
                    gout26 += trr_01x * trr_30y * trr_01z;
                    gout27 += trr_01x * trr_20y * trr_11z;
                    gout28 += trr_01x * trr_10y * trr_21z;
                    double trr_31z = cpz * trr_30z + 3*b00 * trr_20z;
                    gout29 += trr_01x * fac * trr_31z;
                    double trr_02y = cpy * trr_01y + 1*b01 * fac;
                    gout30 += trr_30x * trr_02y * wt;
                    double trr_12y = cpy * trr_11y + 1*b01 * trr_10y + 1*b00 * trr_01y;
                    gout31 += trr_20x * trr_12y * wt;
                    gout32 += trr_20x * trr_02y * trr_10z;
                    double trr_22y = cpy * trr_21y + 1*b01 * trr_20y + 2*b00 * trr_11y;
                    gout33 += trr_10x * trr_22y * wt;
                    gout34 += trr_10x * trr_12y * trr_10z;
                    gout35 += trr_10x * trr_02y * trr_20z;
                    double trr_32y = cpy * trr_31y + 1*b01 * trr_30y + 3*b00 * trr_21y;
                    gout36 += 1 * trr_32y * wt;
                    gout37 += 1 * trr_22y * trr_10z;
                    gout38 += 1 * trr_12y * trr_20z;
                    gout39 += 1 * trr_02y * trr_30z;
                    gout40 += trr_30x * trr_01y * trr_01z;
                    gout41 += trr_20x * trr_11y * trr_01z;
                    gout42 += trr_20x * trr_01y * trr_11z;
                    gout43 += trr_10x * trr_21y * trr_01z;
                    gout44 += trr_10x * trr_11y * trr_11z;
                    gout45 += trr_10x * trr_01y * trr_21z;
                    gout46 += 1 * trr_31y * trr_01z;
                    gout47 += 1 * trr_21y * trr_11z;
                    gout48 += 1 * trr_11y * trr_21z;
                    gout49 += 1 * trr_01y * trr_31z;
                    double trr_02z = cpz * trr_01z + 1*b01 * wt;
                    gout50 += trr_30x * fac * trr_02z;
                    gout51 += trr_20x * trr_10y * trr_02z;
                    double trr_12z = cpz * trr_11z + 1*b01 * trr_10z + 1*b00 * trr_01z;
                    gout52 += trr_20x * fac * trr_12z;
                    gout53 += trr_10x * trr_20y * trr_02z;
                    gout54 += trr_10x * trr_10y * trr_12z;
                    double trr_22z = cpz * trr_21z + 1*b01 * trr_20z + 2*b00 * trr_11z;
                    gout55 += trr_10x * fac * trr_22z;
                    gout56 += 1 * trr_30y * trr_02z;
                    gout57 += 1 * trr_20y * trr_12z;
                    gout58 += 1 * trr_10y * trr_22z;
                    double trr_32z = cpz * trr_31z + 1*b01 * trr_30z + 3*b00 * trr_21z;
                    gout59 += 1 * fac * trr_32z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                val += gout10 * dm[(j0+0)*nao+(k0+1)];
                val += gout20 * dm[(j0+0)*nao+(k0+2)];
                val += gout30 * dm[(j0+0)*nao+(k0+3)];
                val += gout40 * dm[(j0+0)*nao+(k0+4)];
                val += gout50 * dm[(j0+0)*nao+(k0+5)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                val += gout11 * dm[(j0+0)*nao+(k0+1)];
                val += gout21 * dm[(j0+0)*nao+(k0+2)];
                val += gout31 * dm[(j0+0)*nao+(k0+3)];
                val += gout41 * dm[(j0+0)*nao+(k0+4)];
                val += gout51 * dm[(j0+0)*nao+(k0+5)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                val += gout12 * dm[(j0+0)*nao+(k0+1)];
                val += gout22 * dm[(j0+0)*nao+(k0+2)];
                val += gout32 * dm[(j0+0)*nao+(k0+3)];
                val += gout42 * dm[(j0+0)*nao+(k0+4)];
                val += gout52 * dm[(j0+0)*nao+(k0+5)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(k0+0)];
                val += gout13 * dm[(j0+0)*nao+(k0+1)];
                val += gout23 * dm[(j0+0)*nao+(k0+2)];
                val += gout33 * dm[(j0+0)*nao+(k0+3)];
                val += gout43 * dm[(j0+0)*nao+(k0+4)];
                val += gout53 * dm[(j0+0)*nao+(k0+5)];
                atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(k0+0)];
                val += gout14 * dm[(j0+0)*nao+(k0+1)];
                val += gout24 * dm[(j0+0)*nao+(k0+2)];
                val += gout34 * dm[(j0+0)*nao+(k0+3)];
                val += gout44 * dm[(j0+0)*nao+(k0+4)];
                val += gout54 * dm[(j0+0)*nao+(k0+5)];
                atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(k0+0)];
                val += gout15 * dm[(j0+0)*nao+(k0+1)];
                val += gout25 * dm[(j0+0)*nao+(k0+2)];
                val += gout35 * dm[(j0+0)*nao+(k0+3)];
                val += gout45 * dm[(j0+0)*nao+(k0+4)];
                val += gout55 * dm[(j0+0)*nao+(k0+5)];
                atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                val = 0;
                val += gout6 * dm[(j0+0)*nao+(k0+0)];
                val += gout16 * dm[(j0+0)*nao+(k0+1)];
                val += gout26 * dm[(j0+0)*nao+(k0+2)];
                val += gout36 * dm[(j0+0)*nao+(k0+3)];
                val += gout46 * dm[(j0+0)*nao+(k0+4)];
                val += gout56 * dm[(j0+0)*nao+(k0+5)];
                atomicAdd(vk+(i0+6)*nao+(l0+0), val);
                val = 0;
                val += gout7 * dm[(j0+0)*nao+(k0+0)];
                val += gout17 * dm[(j0+0)*nao+(k0+1)];
                val += gout27 * dm[(j0+0)*nao+(k0+2)];
                val += gout37 * dm[(j0+0)*nao+(k0+3)];
                val += gout47 * dm[(j0+0)*nao+(k0+4)];
                val += gout57 * dm[(j0+0)*nao+(k0+5)];
                atomicAdd(vk+(i0+7)*nao+(l0+0), val);
                val = 0;
                val += gout8 * dm[(j0+0)*nao+(k0+0)];
                val += gout18 * dm[(j0+0)*nao+(k0+1)];
                val += gout28 * dm[(j0+0)*nao+(k0+2)];
                val += gout38 * dm[(j0+0)*nao+(k0+3)];
                val += gout48 * dm[(j0+0)*nao+(k0+4)];
                val += gout58 * dm[(j0+0)*nao+(k0+5)];
                atomicAdd(vk+(i0+8)*nao+(l0+0), val);
                val = 0;
                val += gout9 * dm[(j0+0)*nao+(k0+0)];
                val += gout19 * dm[(j0+0)*nao+(k0+1)];
                val += gout29 * dm[(j0+0)*nao+(k0+2)];
                val += gout39 * dm[(j0+0)*nao+(k0+3)];
                val += gout49 * dm[(j0+0)*nao+(k0+4)];
                val += gout59 * dm[(j0+0)*nao+(k0+5)];
                atomicAdd(vk+(i0+9)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout10 * dm[(i0+0)*nao+(k0+1)];
                val += gout20 * dm[(i0+0)*nao+(k0+2)];
                val += gout30 * dm[(i0+0)*nao+(k0+3)];
                val += gout40 * dm[(i0+0)*nao+(k0+4)];
                val += gout50 * dm[(i0+0)*nao+(k0+5)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout11 * dm[(i0+1)*nao+(k0+1)];
                val += gout21 * dm[(i0+1)*nao+(k0+2)];
                val += gout31 * dm[(i0+1)*nao+(k0+3)];
                val += gout41 * dm[(i0+1)*nao+(k0+4)];
                val += gout51 * dm[(i0+1)*nao+(k0+5)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                val += gout12 * dm[(i0+2)*nao+(k0+1)];
                val += gout22 * dm[(i0+2)*nao+(k0+2)];
                val += gout32 * dm[(i0+2)*nao+(k0+3)];
                val += gout42 * dm[(i0+2)*nao+(k0+4)];
                val += gout52 * dm[(i0+2)*nao+(k0+5)];
                val += gout3 * dm[(i0+3)*nao+(k0+0)];
                val += gout13 * dm[(i0+3)*nao+(k0+1)];
                val += gout23 * dm[(i0+3)*nao+(k0+2)];
                val += gout33 * dm[(i0+3)*nao+(k0+3)];
                val += gout43 * dm[(i0+3)*nao+(k0+4)];
                val += gout53 * dm[(i0+3)*nao+(k0+5)];
                val += gout4 * dm[(i0+4)*nao+(k0+0)];
                val += gout14 * dm[(i0+4)*nao+(k0+1)];
                val += gout24 * dm[(i0+4)*nao+(k0+2)];
                val += gout34 * dm[(i0+4)*nao+(k0+3)];
                val += gout44 * dm[(i0+4)*nao+(k0+4)];
                val += gout54 * dm[(i0+4)*nao+(k0+5)];
                val += gout5 * dm[(i0+5)*nao+(k0+0)];
                val += gout15 * dm[(i0+5)*nao+(k0+1)];
                val += gout25 * dm[(i0+5)*nao+(k0+2)];
                val += gout35 * dm[(i0+5)*nao+(k0+3)];
                val += gout45 * dm[(i0+5)*nao+(k0+4)];
                val += gout55 * dm[(i0+5)*nao+(k0+5)];
                val += gout6 * dm[(i0+6)*nao+(k0+0)];
                val += gout16 * dm[(i0+6)*nao+(k0+1)];
                val += gout26 * dm[(i0+6)*nao+(k0+2)];
                val += gout36 * dm[(i0+6)*nao+(k0+3)];
                val += gout46 * dm[(i0+6)*nao+(k0+4)];
                val += gout56 * dm[(i0+6)*nao+(k0+5)];
                val += gout7 * dm[(i0+7)*nao+(k0+0)];
                val += gout17 * dm[(i0+7)*nao+(k0+1)];
                val += gout27 * dm[(i0+7)*nao+(k0+2)];
                val += gout37 * dm[(i0+7)*nao+(k0+3)];
                val += gout47 * dm[(i0+7)*nao+(k0+4)];
                val += gout57 * dm[(i0+7)*nao+(k0+5)];
                val += gout8 * dm[(i0+8)*nao+(k0+0)];
                val += gout18 * dm[(i0+8)*nao+(k0+1)];
                val += gout28 * dm[(i0+8)*nao+(k0+2)];
                val += gout38 * dm[(i0+8)*nao+(k0+3)];
                val += gout48 * dm[(i0+8)*nao+(k0+4)];
                val += gout58 * dm[(i0+8)*nao+(k0+5)];
                val += gout9 * dm[(i0+9)*nao+(k0+0)];
                val += gout19 * dm[(i0+9)*nao+(k0+1)];
                val += gout29 * dm[(i0+9)*nao+(k0+2)];
                val += gout39 * dm[(i0+9)*nao+(k0+3)];
                val += gout49 * dm[(i0+9)*nao+(k0+4)];
                val += gout59 * dm[(i0+9)*nao+(k0+5)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout10 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+1), val);
                val = 0;
                val += gout20 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+2), val);
                val = 0;
                val += gout30 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+3), val);
                val = 0;
                val += gout40 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+4), val);
                val = 0;
                val += gout50 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+5), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout11 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+1), val);
                val = 0;
                val += gout21 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+2), val);
                val = 0;
                val += gout31 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+3), val);
                val = 0;
                val += gout41 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+4), val);
                val = 0;
                val += gout51 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+5), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout12 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+1), val);
                val = 0;
                val += gout22 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+2), val);
                val = 0;
                val += gout32 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+3), val);
                val = 0;
                val += gout42 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+4), val);
                val = 0;
                val += gout52 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+5), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                val = 0;
                val += gout13 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+1), val);
                val = 0;
                val += gout23 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+2), val);
                val = 0;
                val += gout33 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+3), val);
                val = 0;
                val += gout43 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+4), val);
                val = 0;
                val += gout53 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+5), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                val = 0;
                val += gout14 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+1), val);
                val = 0;
                val += gout24 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+2), val);
                val = 0;
                val += gout34 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+3), val);
                val = 0;
                val += gout44 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+4), val);
                val = 0;
                val += gout54 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+5), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                val = 0;
                val += gout15 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+1), val);
                val = 0;
                val += gout25 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+2), val);
                val = 0;
                val += gout35 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+3), val);
                val = 0;
                val += gout45 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+4), val);
                val = 0;
                val += gout55 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+5), val);
                val = 0;
                val += gout6 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+6)*nao+(k0+0), val);
                val = 0;
                val += gout16 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+6)*nao+(k0+1), val);
                val = 0;
                val += gout26 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+6)*nao+(k0+2), val);
                val = 0;
                val += gout36 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+6)*nao+(k0+3), val);
                val = 0;
                val += gout46 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+6)*nao+(k0+4), val);
                val = 0;
                val += gout56 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+6)*nao+(k0+5), val);
                val = 0;
                val += gout7 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+7)*nao+(k0+0), val);
                val = 0;
                val += gout17 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+7)*nao+(k0+1), val);
                val = 0;
                val += gout27 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+7)*nao+(k0+2), val);
                val = 0;
                val += gout37 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+7)*nao+(k0+3), val);
                val = 0;
                val += gout47 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+7)*nao+(k0+4), val);
                val = 0;
                val += gout57 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+7)*nao+(k0+5), val);
                val = 0;
                val += gout8 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+8)*nao+(k0+0), val);
                val = 0;
                val += gout18 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+8)*nao+(k0+1), val);
                val = 0;
                val += gout28 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+8)*nao+(k0+2), val);
                val = 0;
                val += gout38 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+8)*nao+(k0+3), val);
                val = 0;
                val += gout48 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+8)*nao+(k0+4), val);
                val = 0;
                val += gout58 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+8)*nao+(k0+5), val);
                val = 0;
                val += gout9 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+9)*nao+(k0+0), val);
                val = 0;
                val += gout19 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+9)*nao+(k0+1), val);
                val = 0;
                val += gout29 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+9)*nao+(k0+2), val);
                val = 0;
                val += gout39 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+9)*nao+(k0+3), val);
                val = 0;
                val += gout49 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+9)*nao+(k0+4), val);
                val = 0;
                val += gout59 * dm[(j0+0)*nao+(l0+0)];
                atomicAdd(vk+(i0+9)*nao+(k0+5), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                val += gout3 * dm[(i0+3)*nao+(l0+0)];
                val += gout4 * dm[(i0+4)*nao+(l0+0)];
                val += gout5 * dm[(i0+5)*nao+(l0+0)];
                val += gout6 * dm[(i0+6)*nao+(l0+0)];
                val += gout7 * dm[(i0+7)*nao+(l0+0)];
                val += gout8 * dm[(i0+8)*nao+(l0+0)];
                val += gout9 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                val = 0;
                val += gout10 * dm[(i0+0)*nao+(l0+0)];
                val += gout11 * dm[(i0+1)*nao+(l0+0)];
                val += gout12 * dm[(i0+2)*nao+(l0+0)];
                val += gout13 * dm[(i0+3)*nao+(l0+0)];
                val += gout14 * dm[(i0+4)*nao+(l0+0)];
                val += gout15 * dm[(i0+5)*nao+(l0+0)];
                val += gout16 * dm[(i0+6)*nao+(l0+0)];
                val += gout17 * dm[(i0+7)*nao+(l0+0)];
                val += gout18 * dm[(i0+8)*nao+(l0+0)];
                val += gout19 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+1), val);
                val = 0;
                val += gout20 * dm[(i0+0)*nao+(l0+0)];
                val += gout21 * dm[(i0+1)*nao+(l0+0)];
                val += gout22 * dm[(i0+2)*nao+(l0+0)];
                val += gout23 * dm[(i0+3)*nao+(l0+0)];
                val += gout24 * dm[(i0+4)*nao+(l0+0)];
                val += gout25 * dm[(i0+5)*nao+(l0+0)];
                val += gout26 * dm[(i0+6)*nao+(l0+0)];
                val += gout27 * dm[(i0+7)*nao+(l0+0)];
                val += gout28 * dm[(i0+8)*nao+(l0+0)];
                val += gout29 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+2), val);
                val = 0;
                val += gout30 * dm[(i0+0)*nao+(l0+0)];
                val += gout31 * dm[(i0+1)*nao+(l0+0)];
                val += gout32 * dm[(i0+2)*nao+(l0+0)];
                val += gout33 * dm[(i0+3)*nao+(l0+0)];
                val += gout34 * dm[(i0+4)*nao+(l0+0)];
                val += gout35 * dm[(i0+5)*nao+(l0+0)];
                val += gout36 * dm[(i0+6)*nao+(l0+0)];
                val += gout37 * dm[(i0+7)*nao+(l0+0)];
                val += gout38 * dm[(i0+8)*nao+(l0+0)];
                val += gout39 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+3), val);
                val = 0;
                val += gout40 * dm[(i0+0)*nao+(l0+0)];
                val += gout41 * dm[(i0+1)*nao+(l0+0)];
                val += gout42 * dm[(i0+2)*nao+(l0+0)];
                val += gout43 * dm[(i0+3)*nao+(l0+0)];
                val += gout44 * dm[(i0+4)*nao+(l0+0)];
                val += gout45 * dm[(i0+5)*nao+(l0+0)];
                val += gout46 * dm[(i0+6)*nao+(l0+0)];
                val += gout47 * dm[(i0+7)*nao+(l0+0)];
                val += gout48 * dm[(i0+8)*nao+(l0+0)];
                val += gout49 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+4), val);
                val = 0;
                val += gout50 * dm[(i0+0)*nao+(l0+0)];
                val += gout51 * dm[(i0+1)*nao+(l0+0)];
                val += gout52 * dm[(i0+2)*nao+(l0+0)];
                val += gout53 * dm[(i0+3)*nao+(l0+0)];
                val += gout54 * dm[(i0+4)*nao+(l0+0)];
                val += gout55 * dm[(i0+5)*nao+(l0+0)];
                val += gout56 * dm[(i0+6)*nao+(l0+0)];
                val += gout57 * dm[(i0+7)*nao+(l0+0)];
                val += gout58 * dm[(i0+8)*nao+(l0+0)];
                val += gout59 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+5), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_3020, MINBLOCKS_3020)
k_3020(KARGS)    { kbody_3020<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_3020, MINBLOCKS_3020)
k_rs_3020(KARGS) { kbody_3020<2>(KFWD); }


#define NTHREADS_3100  256
#define MINBLOCKS_3100 2
template <int NRANGE> __device__ __forceinline__ void
kbody_3100(KARGS)
{
    constexpr int NROOTS = 3;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.; double gout6 = 0.; double gout7 = 0.; double gout8 = 0.; double gout9 = 0.; double gout10 = 0.; double gout11 = 0.; double gout12 = 0.; double gout13 = 0.; double gout14 = 0.; double gout15 = 0.; double gout16 = 0.; double gout17 = 0.; double gout18 = 0.; double gout19 = 0.; double gout20 = 0.; double gout21 = 0.; double gout22 = 0.; double gout23 = 0.; double gout24 = 0.; double gout25 = 0.; double gout26 = 0.; double gout27 = 0.; double gout28 = 0.; double gout29 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                    double hrr_3100x = trr_40x - xjxi * trr_30x;
                    gout0 += hrr_3100x * fac * wt;
                    double hrr_2100x = trr_30x - xjxi * trr_20x;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += hrr_2100x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += hrr_2100x * fac * trr_10z;
                    double hrr_1100x = trr_20x - xjxi * trr_10x;
                    double trr_20y = c0y * trr_10y + 1*b10 * fac;
                    gout3 += hrr_1100x * trr_20y * wt;
                    gout4 += hrr_1100x * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += hrr_1100x * fac * trr_20z;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    gout6 += hrr_0100x * trr_30y * wt;
                    gout7 += hrr_0100x * trr_20y * trr_10z;
                    gout8 += hrr_0100x * trr_10y * trr_20z;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    gout9 += hrr_0100x * fac * trr_30z;
                    double hrr_0100y = trr_10y - yjyi * fac;
                    gout10 += trr_30x * hrr_0100y * wt;
                    double hrr_1100y = trr_20y - yjyi * trr_10y;
                    gout11 += trr_20x * hrr_1100y * wt;
                    gout12 += trr_20x * hrr_0100y * trr_10z;
                    double hrr_2100y = trr_30y - yjyi * trr_20y;
                    gout13 += trr_10x * hrr_2100y * wt;
                    gout14 += trr_10x * hrr_1100y * trr_10z;
                    gout15 += trr_10x * hrr_0100y * trr_20z;
                    double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                    double hrr_3100y = trr_40y - yjyi * trr_30y;
                    gout16 += 1 * hrr_3100y * wt;
                    gout17 += 1 * hrr_2100y * trr_10z;
                    gout18 += 1 * hrr_1100y * trr_20z;
                    gout19 += 1 * hrr_0100y * trr_30z;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    gout20 += trr_30x * fac * hrr_0100z;
                    gout21 += trr_20x * trr_10y * hrr_0100z;
                    double hrr_1100z = trr_20z - zjzi * trr_10z;
                    gout22 += trr_20x * fac * hrr_1100z;
                    gout23 += trr_10x * trr_20y * hrr_0100z;
                    gout24 += trr_10x * trr_10y * hrr_1100z;
                    double hrr_2100z = trr_30z - zjzi * trr_20z;
                    gout25 += trr_10x * fac * hrr_2100z;
                    gout26 += 1 * trr_30y * hrr_0100z;
                    gout27 += 1 * trr_20y * hrr_1100z;
                    gout28 += 1 * trr_10y * hrr_2100z;
                    double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                    double hrr_3100z = trr_40z - zjzi * trr_30z;
                    gout29 += 1 * fac * hrr_3100z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                val += gout10 * dm[(j0+1)*nao+(k0+0)];
                val += gout20 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                val += gout11 * dm[(j0+1)*nao+(k0+0)];
                val += gout21 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                val += gout12 * dm[(j0+1)*nao+(k0+0)];
                val += gout22 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(k0+0)];
                val += gout13 * dm[(j0+1)*nao+(k0+0)];
                val += gout23 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(k0+0)];
                val += gout14 * dm[(j0+1)*nao+(k0+0)];
                val += gout24 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(k0+0)];
                val += gout15 * dm[(j0+1)*nao+(k0+0)];
                val += gout25 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                val = 0;
                val += gout6 * dm[(j0+0)*nao+(k0+0)];
                val += gout16 * dm[(j0+1)*nao+(k0+0)];
                val += gout26 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+6)*nao+(l0+0), val);
                val = 0;
                val += gout7 * dm[(j0+0)*nao+(k0+0)];
                val += gout17 * dm[(j0+1)*nao+(k0+0)];
                val += gout27 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+7)*nao+(l0+0), val);
                val = 0;
                val += gout8 * dm[(j0+0)*nao+(k0+0)];
                val += gout18 * dm[(j0+1)*nao+(k0+0)];
                val += gout28 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+8)*nao+(l0+0), val);
                val = 0;
                val += gout9 * dm[(j0+0)*nao+(k0+0)];
                val += gout19 * dm[(j0+1)*nao+(k0+0)];
                val += gout29 * dm[(j0+2)*nao+(k0+0)];
                atomicAdd(vk+(i0+9)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                val += gout3 * dm[(i0+3)*nao+(k0+0)];
                val += gout4 * dm[(i0+4)*nao+(k0+0)];
                val += gout5 * dm[(i0+5)*nao+(k0+0)];
                val += gout6 * dm[(i0+6)*nao+(k0+0)];
                val += gout7 * dm[(i0+7)*nao+(k0+0)];
                val += gout8 * dm[(i0+8)*nao+(k0+0)];
                val += gout9 * dm[(i0+9)*nao+(k0+0)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout10 * dm[(i0+0)*nao+(k0+0)];
                val += gout11 * dm[(i0+1)*nao+(k0+0)];
                val += gout12 * dm[(i0+2)*nao+(k0+0)];
                val += gout13 * dm[(i0+3)*nao+(k0+0)];
                val += gout14 * dm[(i0+4)*nao+(k0+0)];
                val += gout15 * dm[(i0+5)*nao+(k0+0)];
                val += gout16 * dm[(i0+6)*nao+(k0+0)];
                val += gout17 * dm[(i0+7)*nao+(k0+0)];
                val += gout18 * dm[(i0+8)*nao+(k0+0)];
                val += gout19 * dm[(i0+9)*nao+(k0+0)];
                atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                val = 0;
                val += gout20 * dm[(i0+0)*nao+(k0+0)];
                val += gout21 * dm[(i0+1)*nao+(k0+0)];
                val += gout22 * dm[(i0+2)*nao+(k0+0)];
                val += gout23 * dm[(i0+3)*nao+(k0+0)];
                val += gout24 * dm[(i0+4)*nao+(k0+0)];
                val += gout25 * dm[(i0+5)*nao+(k0+0)];
                val += gout26 * dm[(i0+6)*nao+(k0+0)];
                val += gout27 * dm[(i0+7)*nao+(k0+0)];
                val += gout28 * dm[(i0+8)*nao+(k0+0)];
                val += gout29 * dm[(i0+9)*nao+(k0+0)];
                atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                val += gout10 * dm[(j0+1)*nao+(l0+0)];
                val += gout20 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                val += gout11 * dm[(j0+1)*nao+(l0+0)];
                val += gout21 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                val += gout12 * dm[(j0+1)*nao+(l0+0)];
                val += gout22 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(l0+0)];
                val += gout13 * dm[(j0+1)*nao+(l0+0)];
                val += gout23 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(l0+0)];
                val += gout14 * dm[(j0+1)*nao+(l0+0)];
                val += gout24 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(l0+0)];
                val += gout15 * dm[(j0+1)*nao+(l0+0)];
                val += gout25 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                val = 0;
                val += gout6 * dm[(j0+0)*nao+(l0+0)];
                val += gout16 * dm[(j0+1)*nao+(l0+0)];
                val += gout26 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+6)*nao+(k0+0), val);
                val = 0;
                val += gout7 * dm[(j0+0)*nao+(l0+0)];
                val += gout17 * dm[(j0+1)*nao+(l0+0)];
                val += gout27 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+7)*nao+(k0+0), val);
                val = 0;
                val += gout8 * dm[(j0+0)*nao+(l0+0)];
                val += gout18 * dm[(j0+1)*nao+(l0+0)];
                val += gout28 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+8)*nao+(k0+0), val);
                val = 0;
                val += gout9 * dm[(j0+0)*nao+(l0+0)];
                val += gout19 * dm[(j0+1)*nao+(l0+0)];
                val += gout29 * dm[(j0+2)*nao+(l0+0)];
                atomicAdd(vk+(i0+9)*nao+(k0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                val += gout3 * dm[(i0+3)*nao+(l0+0)];
                val += gout4 * dm[(i0+4)*nao+(l0+0)];
                val += gout5 * dm[(i0+5)*nao+(l0+0)];
                val += gout6 * dm[(i0+6)*nao+(l0+0)];
                val += gout7 * dm[(i0+7)*nao+(l0+0)];
                val += gout8 * dm[(i0+8)*nao+(l0+0)];
                val += gout9 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                val = 0;
                val += gout10 * dm[(i0+0)*nao+(l0+0)];
                val += gout11 * dm[(i0+1)*nao+(l0+0)];
                val += gout12 * dm[(i0+2)*nao+(l0+0)];
                val += gout13 * dm[(i0+3)*nao+(l0+0)];
                val += gout14 * dm[(i0+4)*nao+(l0+0)];
                val += gout15 * dm[(i0+5)*nao+(l0+0)];
                val += gout16 * dm[(i0+6)*nao+(l0+0)];
                val += gout17 * dm[(i0+7)*nao+(l0+0)];
                val += gout18 * dm[(i0+8)*nao+(l0+0)];
                val += gout19 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                val = 0;
                val += gout20 * dm[(i0+0)*nao+(l0+0)];
                val += gout21 * dm[(i0+1)*nao+(l0+0)];
                val += gout22 * dm[(i0+2)*nao+(l0+0)];
                val += gout23 * dm[(i0+3)*nao+(l0+0)];
                val += gout24 * dm[(i0+4)*nao+(l0+0)];
                val += gout25 * dm[(i0+5)*nao+(l0+0)];
                val += gout26 * dm[(i0+6)*nao+(l0+0)];
                val += gout27 * dm[(i0+7)*nao+(l0+0)];
                val += gout28 * dm[(i0+8)*nao+(l0+0)];
                val += gout29 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+2)*nao+(k0+0), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_3100, MINBLOCKS_3100)
k_3100(KARGS)    { kbody_3100<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_3100, MINBLOCKS_3100)
k_rs_3100(KARGS) { kbody_3100<2>(KFWD); }


#define NTHREADS_3200  128
#define MINBLOCKS_3200 2
template <int NRANGE> __device__ __forceinline__ void
kbody_3200(KARGS)
{
    constexpr int NROOTS = 3;
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
        int jsh = bas_ij % nbas;
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
            double ai = expi[ij/jprim], aj = expj[ij%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij  [ij] = aij;
            s_iaij [ij] = iaij;
            s_ajaij[ij] = aj_aij;
            s_cicj [ij] = ci[ij/jprim] * cj[ij%jprim] * exp(-ai*aj_aij*rr_ij);
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
            double rw[2*NROOTS];
            double gout0 = 0.; double gout1 = 0.; double gout2 = 0.; double gout3 = 0.; double gout4 = 0.; double gout5 = 0.; double gout6 = 0.; double gout7 = 0.; double gout8 = 0.; double gout9 = 0.; double gout10 = 0.; double gout11 = 0.; double gout12 = 0.; double gout13 = 0.; double gout14 = 0.; double gout15 = 0.; double gout16 = 0.; double gout17 = 0.; double gout18 = 0.; double gout19 = 0.; double gout20 = 0.; double gout21 = 0.; double gout22 = 0.; double gout23 = 0.; double gout24 = 0.; double gout25 = 0.; double gout26 = 0.; double gout27 = 0.; double gout28 = 0.; double gout29 = 0.; double gout30 = 0.; double gout31 = 0.; double gout32 = 0.; double gout33 = 0.; double gout34 = 0.; double gout35 = 0.; double gout36 = 0.; double gout37 = 0.; double gout38 = 0.; double gout39 = 0.; double gout40 = 0.; double gout41 = 0.; double gout42 = 0.; double gout43 = 0.; double gout44 = 0.; double gout45 = 0.; double gout46 = 0.; double gout47 = 0.; double gout48 = 0.; double gout49 = 0.; double gout50 = 0.; double gout51 = 0.; double gout52 = 0.; double gout53 = 0.; double gout54 = 0.; double gout55 = 0.; double gout56 = 0.; double gout57 = 0.; double gout58 = 0.; double gout59 = 0.;
            for (int klp = 0; klp < kprim*lprim; ++klp) {
                double ak = expk[klp/lprim], al = expl[klp%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
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

                    double wt = rw[2*irys+1];
                    double rt = rw[2*irys];
                    double rt_aa = rt * inv_s2;
                    double xjxi = rjri[0];
                    double rt_aij = rt_aa * akl;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double c0x = xjxi*aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    double trr_30x = c0x * trr_20x + 2*b10 * trr_10x;
                    double trr_40x = c0x * trr_30x + 3*b10 * trr_20x;
                    double trr_50x = c0x * trr_40x + 4*b10 * trr_30x;
                    double hrr_4100x = trr_50x - xjxi * trr_40x;
                    double hrr_3100x = trr_40x - xjxi * trr_30x;
                    double hrr_3200x = hrr_4100x - xjxi * hrr_3100x;
                    gout0 += hrr_3200x * fac * wt;
                    double hrr_2100x = trr_30x - xjxi * trr_20x;
                    double hrr_2200x = hrr_3100x - xjxi * hrr_2100x;
                    double yjyi = rjri[1];
                    double c0y = yjyi*aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac;
                    gout1 += hrr_2200x * trr_10y * wt;
                    double zjzi = rjri[2];
                    double c0z = zjzi*aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    gout2 += hrr_2200x * fac * trr_10z;
                    double hrr_1100x = trr_20x - xjxi * trr_10x;
                    double hrr_1200x = hrr_2100x - xjxi * hrr_1100x;
                    double trr_20y = c0y * trr_10y + 1*b10 * fac;
                    gout3 += hrr_1200x * trr_20y * wt;
                    gout4 += hrr_1200x * trr_10y * trr_10z;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    gout5 += hrr_1200x * fac * trr_20z;
                    double hrr_0100x = trr_10x - xjxi * 1;
                    double hrr_0200x = hrr_1100x - xjxi * hrr_0100x;
                    double trr_30y = c0y * trr_20y + 2*b10 * trr_10y;
                    gout6 += hrr_0200x * trr_30y * wt;
                    gout7 += hrr_0200x * trr_20y * trr_10z;
                    gout8 += hrr_0200x * trr_10y * trr_20z;
                    double trr_30z = c0z * trr_20z + 2*b10 * trr_10z;
                    gout9 += hrr_0200x * fac * trr_30z;
                    double hrr_0100y = trr_10y - yjyi * fac;
                    gout10 += hrr_3100x * hrr_0100y * wt;
                    double hrr_1100y = trr_20y - yjyi * trr_10y;
                    gout11 += hrr_2100x * hrr_1100y * wt;
                    gout12 += hrr_2100x * hrr_0100y * trr_10z;
                    double hrr_2100y = trr_30y - yjyi * trr_20y;
                    gout13 += hrr_1100x * hrr_2100y * wt;
                    gout14 += hrr_1100x * hrr_1100y * trr_10z;
                    gout15 += hrr_1100x * hrr_0100y * trr_20z;
                    double trr_40y = c0y * trr_30y + 3*b10 * trr_20y;
                    double hrr_3100y = trr_40y - yjyi * trr_30y;
                    gout16 += hrr_0100x * hrr_3100y * wt;
                    gout17 += hrr_0100x * hrr_2100y * trr_10z;
                    gout18 += hrr_0100x * hrr_1100y * trr_20z;
                    gout19 += hrr_0100x * hrr_0100y * trr_30z;
                    double hrr_0100z = trr_10z - zjzi * wt;
                    gout20 += hrr_3100x * fac * hrr_0100z;
                    gout21 += hrr_2100x * trr_10y * hrr_0100z;
                    double hrr_1100z = trr_20z - zjzi * trr_10z;
                    gout22 += hrr_2100x * fac * hrr_1100z;
                    gout23 += hrr_1100x * trr_20y * hrr_0100z;
                    gout24 += hrr_1100x * trr_10y * hrr_1100z;
                    double hrr_2100z = trr_30z - zjzi * trr_20z;
                    gout25 += hrr_1100x * fac * hrr_2100z;
                    gout26 += hrr_0100x * trr_30y * hrr_0100z;
                    gout27 += hrr_0100x * trr_20y * hrr_1100z;
                    gout28 += hrr_0100x * trr_10y * hrr_2100z;
                    double trr_40z = c0z * trr_30z + 3*b10 * trr_20z;
                    double hrr_3100z = trr_40z - zjzi * trr_30z;
                    gout29 += hrr_0100x * fac * hrr_3100z;
                    double hrr_0200y = hrr_1100y - yjyi * hrr_0100y;
                    gout30 += trr_30x * hrr_0200y * wt;
                    double hrr_1200y = hrr_2100y - yjyi * hrr_1100y;
                    gout31 += trr_20x * hrr_1200y * wt;
                    gout32 += trr_20x * hrr_0200y * trr_10z;
                    double hrr_2200y = hrr_3100y - yjyi * hrr_2100y;
                    gout33 += trr_10x * hrr_2200y * wt;
                    gout34 += trr_10x * hrr_1200y * trr_10z;
                    gout35 += trr_10x * hrr_0200y * trr_20z;
                    double trr_50y = c0y * trr_40y + 4*b10 * trr_30y;
                    double hrr_4100y = trr_50y - yjyi * trr_40y;
                    double hrr_3200y = hrr_4100y - yjyi * hrr_3100y;
                    gout36 += 1 * hrr_3200y * wt;
                    gout37 += 1 * hrr_2200y * trr_10z;
                    gout38 += 1 * hrr_1200y * trr_20z;
                    gout39 += 1 * hrr_0200y * trr_30z;
                    gout40 += trr_30x * hrr_0100y * hrr_0100z;
                    gout41 += trr_20x * hrr_1100y * hrr_0100z;
                    gout42 += trr_20x * hrr_0100y * hrr_1100z;
                    gout43 += trr_10x * hrr_2100y * hrr_0100z;
                    gout44 += trr_10x * hrr_1100y * hrr_1100z;
                    gout45 += trr_10x * hrr_0100y * hrr_2100z;
                    gout46 += 1 * hrr_3100y * hrr_0100z;
                    gout47 += 1 * hrr_2100y * hrr_1100z;
                    gout48 += 1 * hrr_1100y * hrr_2100z;
                    gout49 += 1 * hrr_0100y * hrr_3100z;
                    double hrr_0200z = hrr_1100z - zjzi * hrr_0100z;
                    gout50 += trr_30x * fac * hrr_0200z;
                    gout51 += trr_20x * trr_10y * hrr_0200z;
                    double hrr_1200z = hrr_2100z - zjzi * hrr_1100z;
                    gout52 += trr_20x * fac * hrr_1200z;
                    gout53 += trr_10x * trr_20y * hrr_0200z;
                    gout54 += trr_10x * trr_10y * hrr_1200z;
                    double hrr_2200z = hrr_3100z - zjzi * hrr_2100z;
                    gout55 += trr_10x * fac * hrr_2200z;
                    gout56 += 1 * trr_30y * hrr_0200z;
                    gout57 += 1 * trr_20y * hrr_1200z;
                    gout58 += 1 * trr_10y * hrr_2200z;
                    double trr_50z = c0z * trr_40z + 4*b10 * trr_30z;
                    double hrr_4100z = trr_50z - zjzi * trr_40z;
                    double hrr_3200z = hrr_4100z - zjzi * hrr_3100z;
                    gout59 += 1 * fac * hrr_3200z;
                
                    }
                    }
                }
            }
            {
                int k0 = ao_loc[ksh], l0 = ao_loc[lsh];

            double val;
            val = 0;
                val += gout0 * dm[(j0+0)*nao+(k0+0)];
                val += gout10 * dm[(j0+1)*nao+(k0+0)];
                val += gout20 * dm[(j0+2)*nao+(k0+0)];
                val += gout30 * dm[(j0+3)*nao+(k0+0)];
                val += gout40 * dm[(j0+4)*nao+(k0+0)];
                val += gout50 * dm[(j0+5)*nao+(k0+0)];
                atomicAdd(vk+(i0+0)*nao+(l0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(k0+0)];
                val += gout11 * dm[(j0+1)*nao+(k0+0)];
                val += gout21 * dm[(j0+2)*nao+(k0+0)];
                val += gout31 * dm[(j0+3)*nao+(k0+0)];
                val += gout41 * dm[(j0+4)*nao+(k0+0)];
                val += gout51 * dm[(j0+5)*nao+(k0+0)];
                atomicAdd(vk+(i0+1)*nao+(l0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(k0+0)];
                val += gout12 * dm[(j0+1)*nao+(k0+0)];
                val += gout22 * dm[(j0+2)*nao+(k0+0)];
                val += gout32 * dm[(j0+3)*nao+(k0+0)];
                val += gout42 * dm[(j0+4)*nao+(k0+0)];
                val += gout52 * dm[(j0+5)*nao+(k0+0)];
                atomicAdd(vk+(i0+2)*nao+(l0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(k0+0)];
                val += gout13 * dm[(j0+1)*nao+(k0+0)];
                val += gout23 * dm[(j0+2)*nao+(k0+0)];
                val += gout33 * dm[(j0+3)*nao+(k0+0)];
                val += gout43 * dm[(j0+4)*nao+(k0+0)];
                val += gout53 * dm[(j0+5)*nao+(k0+0)];
                atomicAdd(vk+(i0+3)*nao+(l0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(k0+0)];
                val += gout14 * dm[(j0+1)*nao+(k0+0)];
                val += gout24 * dm[(j0+2)*nao+(k0+0)];
                val += gout34 * dm[(j0+3)*nao+(k0+0)];
                val += gout44 * dm[(j0+4)*nao+(k0+0)];
                val += gout54 * dm[(j0+5)*nao+(k0+0)];
                atomicAdd(vk+(i0+4)*nao+(l0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(k0+0)];
                val += gout15 * dm[(j0+1)*nao+(k0+0)];
                val += gout25 * dm[(j0+2)*nao+(k0+0)];
                val += gout35 * dm[(j0+3)*nao+(k0+0)];
                val += gout45 * dm[(j0+4)*nao+(k0+0)];
                val += gout55 * dm[(j0+5)*nao+(k0+0)];
                atomicAdd(vk+(i0+5)*nao+(l0+0), val);
                val = 0;
                val += gout6 * dm[(j0+0)*nao+(k0+0)];
                val += gout16 * dm[(j0+1)*nao+(k0+0)];
                val += gout26 * dm[(j0+2)*nao+(k0+0)];
                val += gout36 * dm[(j0+3)*nao+(k0+0)];
                val += gout46 * dm[(j0+4)*nao+(k0+0)];
                val += gout56 * dm[(j0+5)*nao+(k0+0)];
                atomicAdd(vk+(i0+6)*nao+(l0+0), val);
                val = 0;
                val += gout7 * dm[(j0+0)*nao+(k0+0)];
                val += gout17 * dm[(j0+1)*nao+(k0+0)];
                val += gout27 * dm[(j0+2)*nao+(k0+0)];
                val += gout37 * dm[(j0+3)*nao+(k0+0)];
                val += gout47 * dm[(j0+4)*nao+(k0+0)];
                val += gout57 * dm[(j0+5)*nao+(k0+0)];
                atomicAdd(vk+(i0+7)*nao+(l0+0), val);
                val = 0;
                val += gout8 * dm[(j0+0)*nao+(k0+0)];
                val += gout18 * dm[(j0+1)*nao+(k0+0)];
                val += gout28 * dm[(j0+2)*nao+(k0+0)];
                val += gout38 * dm[(j0+3)*nao+(k0+0)];
                val += gout48 * dm[(j0+4)*nao+(k0+0)];
                val += gout58 * dm[(j0+5)*nao+(k0+0)];
                atomicAdd(vk+(i0+8)*nao+(l0+0), val);
                val = 0;
                val += gout9 * dm[(j0+0)*nao+(k0+0)];
                val += gout19 * dm[(j0+1)*nao+(k0+0)];
                val += gout29 * dm[(j0+2)*nao+(k0+0)];
                val += gout39 * dm[(j0+3)*nao+(k0+0)];
                val += gout49 * dm[(j0+4)*nao+(k0+0)];
                val += gout59 * dm[(j0+5)*nao+(k0+0)];
                atomicAdd(vk+(i0+9)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(k0+0)];
                val += gout1 * dm[(i0+1)*nao+(k0+0)];
                val += gout2 * dm[(i0+2)*nao+(k0+0)];
                val += gout3 * dm[(i0+3)*nao+(k0+0)];
                val += gout4 * dm[(i0+4)*nao+(k0+0)];
                val += gout5 * dm[(i0+5)*nao+(k0+0)];
                val += gout6 * dm[(i0+6)*nao+(k0+0)];
                val += gout7 * dm[(i0+7)*nao+(k0+0)];
                val += gout8 * dm[(i0+8)*nao+(k0+0)];
                val += gout9 * dm[(i0+9)*nao+(k0+0)];
                atomicAdd(vk+(j0+0)*nao+(l0+0), val);
                val = 0;
                val += gout10 * dm[(i0+0)*nao+(k0+0)];
                val += gout11 * dm[(i0+1)*nao+(k0+0)];
                val += gout12 * dm[(i0+2)*nao+(k0+0)];
                val += gout13 * dm[(i0+3)*nao+(k0+0)];
                val += gout14 * dm[(i0+4)*nao+(k0+0)];
                val += gout15 * dm[(i0+5)*nao+(k0+0)];
                val += gout16 * dm[(i0+6)*nao+(k0+0)];
                val += gout17 * dm[(i0+7)*nao+(k0+0)];
                val += gout18 * dm[(i0+8)*nao+(k0+0)];
                val += gout19 * dm[(i0+9)*nao+(k0+0)];
                atomicAdd(vk+(j0+1)*nao+(l0+0), val);
                val = 0;
                val += gout20 * dm[(i0+0)*nao+(k0+0)];
                val += gout21 * dm[(i0+1)*nao+(k0+0)];
                val += gout22 * dm[(i0+2)*nao+(k0+0)];
                val += gout23 * dm[(i0+3)*nao+(k0+0)];
                val += gout24 * dm[(i0+4)*nao+(k0+0)];
                val += gout25 * dm[(i0+5)*nao+(k0+0)];
                val += gout26 * dm[(i0+6)*nao+(k0+0)];
                val += gout27 * dm[(i0+7)*nao+(k0+0)];
                val += gout28 * dm[(i0+8)*nao+(k0+0)];
                val += gout29 * dm[(i0+9)*nao+(k0+0)];
                atomicAdd(vk+(j0+2)*nao+(l0+0), val);
                val = 0;
                val += gout30 * dm[(i0+0)*nao+(k0+0)];
                val += gout31 * dm[(i0+1)*nao+(k0+0)];
                val += gout32 * dm[(i0+2)*nao+(k0+0)];
                val += gout33 * dm[(i0+3)*nao+(k0+0)];
                val += gout34 * dm[(i0+4)*nao+(k0+0)];
                val += gout35 * dm[(i0+5)*nao+(k0+0)];
                val += gout36 * dm[(i0+6)*nao+(k0+0)];
                val += gout37 * dm[(i0+7)*nao+(k0+0)];
                val += gout38 * dm[(i0+8)*nao+(k0+0)];
                val += gout39 * dm[(i0+9)*nao+(k0+0)];
                atomicAdd(vk+(j0+3)*nao+(l0+0), val);
                val = 0;
                val += gout40 * dm[(i0+0)*nao+(k0+0)];
                val += gout41 * dm[(i0+1)*nao+(k0+0)];
                val += gout42 * dm[(i0+2)*nao+(k0+0)];
                val += gout43 * dm[(i0+3)*nao+(k0+0)];
                val += gout44 * dm[(i0+4)*nao+(k0+0)];
                val += gout45 * dm[(i0+5)*nao+(k0+0)];
                val += gout46 * dm[(i0+6)*nao+(k0+0)];
                val += gout47 * dm[(i0+7)*nao+(k0+0)];
                val += gout48 * dm[(i0+8)*nao+(k0+0)];
                val += gout49 * dm[(i0+9)*nao+(k0+0)];
                atomicAdd(vk+(j0+4)*nao+(l0+0), val);
                val = 0;
                val += gout50 * dm[(i0+0)*nao+(k0+0)];
                val += gout51 * dm[(i0+1)*nao+(k0+0)];
                val += gout52 * dm[(i0+2)*nao+(k0+0)];
                val += gout53 * dm[(i0+3)*nao+(k0+0)];
                val += gout54 * dm[(i0+4)*nao+(k0+0)];
                val += gout55 * dm[(i0+5)*nao+(k0+0)];
                val += gout56 * dm[(i0+6)*nao+(k0+0)];
                val += gout57 * dm[(i0+7)*nao+(k0+0)];
                val += gout58 * dm[(i0+8)*nao+(k0+0)];
                val += gout59 * dm[(i0+9)*nao+(k0+0)];
                atomicAdd(vk+(j0+5)*nao+(l0+0), val);
                val = 0;
                val += gout0 * dm[(j0+0)*nao+(l0+0)];
                val += gout10 * dm[(j0+1)*nao+(l0+0)];
                val += gout20 * dm[(j0+2)*nao+(l0+0)];
                val += gout30 * dm[(j0+3)*nao+(l0+0)];
                val += gout40 * dm[(j0+4)*nao+(l0+0)];
                val += gout50 * dm[(j0+5)*nao+(l0+0)];
                atomicAdd(vk+(i0+0)*nao+(k0+0), val);
                val = 0;
                val += gout1 * dm[(j0+0)*nao+(l0+0)];
                val += gout11 * dm[(j0+1)*nao+(l0+0)];
                val += gout21 * dm[(j0+2)*nao+(l0+0)];
                val += gout31 * dm[(j0+3)*nao+(l0+0)];
                val += gout41 * dm[(j0+4)*nao+(l0+0)];
                val += gout51 * dm[(j0+5)*nao+(l0+0)];
                atomicAdd(vk+(i0+1)*nao+(k0+0), val);
                val = 0;
                val += gout2 * dm[(j0+0)*nao+(l0+0)];
                val += gout12 * dm[(j0+1)*nao+(l0+0)];
                val += gout22 * dm[(j0+2)*nao+(l0+0)];
                val += gout32 * dm[(j0+3)*nao+(l0+0)];
                val += gout42 * dm[(j0+4)*nao+(l0+0)];
                val += gout52 * dm[(j0+5)*nao+(l0+0)];
                atomicAdd(vk+(i0+2)*nao+(k0+0), val);
                val = 0;
                val += gout3 * dm[(j0+0)*nao+(l0+0)];
                val += gout13 * dm[(j0+1)*nao+(l0+0)];
                val += gout23 * dm[(j0+2)*nao+(l0+0)];
                val += gout33 * dm[(j0+3)*nao+(l0+0)];
                val += gout43 * dm[(j0+4)*nao+(l0+0)];
                val += gout53 * dm[(j0+5)*nao+(l0+0)];
                atomicAdd(vk+(i0+3)*nao+(k0+0), val);
                val = 0;
                val += gout4 * dm[(j0+0)*nao+(l0+0)];
                val += gout14 * dm[(j0+1)*nao+(l0+0)];
                val += gout24 * dm[(j0+2)*nao+(l0+0)];
                val += gout34 * dm[(j0+3)*nao+(l0+0)];
                val += gout44 * dm[(j0+4)*nao+(l0+0)];
                val += gout54 * dm[(j0+5)*nao+(l0+0)];
                atomicAdd(vk+(i0+4)*nao+(k0+0), val);
                val = 0;
                val += gout5 * dm[(j0+0)*nao+(l0+0)];
                val += gout15 * dm[(j0+1)*nao+(l0+0)];
                val += gout25 * dm[(j0+2)*nao+(l0+0)];
                val += gout35 * dm[(j0+3)*nao+(l0+0)];
                val += gout45 * dm[(j0+4)*nao+(l0+0)];
                val += gout55 * dm[(j0+5)*nao+(l0+0)];
                atomicAdd(vk+(i0+5)*nao+(k0+0), val);
                val = 0;
                val += gout6 * dm[(j0+0)*nao+(l0+0)];
                val += gout16 * dm[(j0+1)*nao+(l0+0)];
                val += gout26 * dm[(j0+2)*nao+(l0+0)];
                val += gout36 * dm[(j0+3)*nao+(l0+0)];
                val += gout46 * dm[(j0+4)*nao+(l0+0)];
                val += gout56 * dm[(j0+5)*nao+(l0+0)];
                atomicAdd(vk+(i0+6)*nao+(k0+0), val);
                val = 0;
                val += gout7 * dm[(j0+0)*nao+(l0+0)];
                val += gout17 * dm[(j0+1)*nao+(l0+0)];
                val += gout27 * dm[(j0+2)*nao+(l0+0)];
                val += gout37 * dm[(j0+3)*nao+(l0+0)];
                val += gout47 * dm[(j0+4)*nao+(l0+0)];
                val += gout57 * dm[(j0+5)*nao+(l0+0)];
                atomicAdd(vk+(i0+7)*nao+(k0+0), val);
                val = 0;
                val += gout8 * dm[(j0+0)*nao+(l0+0)];
                val += gout18 * dm[(j0+1)*nao+(l0+0)];
                val += gout28 * dm[(j0+2)*nao+(l0+0)];
                val += gout38 * dm[(j0+3)*nao+(l0+0)];
                val += gout48 * dm[(j0+4)*nao+(l0+0)];
                val += gout58 * dm[(j0+5)*nao+(l0+0)];
                atomicAdd(vk+(i0+8)*nao+(k0+0), val);
                val = 0;
                val += gout9 * dm[(j0+0)*nao+(l0+0)];
                val += gout19 * dm[(j0+1)*nao+(l0+0)];
                val += gout29 * dm[(j0+2)*nao+(l0+0)];
                val += gout39 * dm[(j0+3)*nao+(l0+0)];
                val += gout49 * dm[(j0+4)*nao+(l0+0)];
                val += gout59 * dm[(j0+5)*nao+(l0+0)];
                atomicAdd(vk+(i0+9)*nao+(k0+0), val);
                val = 0;
                val += gout0 * dm[(i0+0)*nao+(l0+0)];
                val += gout1 * dm[(i0+1)*nao+(l0+0)];
                val += gout2 * dm[(i0+2)*nao+(l0+0)];
                val += gout3 * dm[(i0+3)*nao+(l0+0)];
                val += gout4 * dm[(i0+4)*nao+(l0+0)];
                val += gout5 * dm[(i0+5)*nao+(l0+0)];
                val += gout6 * dm[(i0+6)*nao+(l0+0)];
                val += gout7 * dm[(i0+7)*nao+(l0+0)];
                val += gout8 * dm[(i0+8)*nao+(l0+0)];
                val += gout9 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+0)*nao+(k0+0), val);
                val = 0;
                val += gout10 * dm[(i0+0)*nao+(l0+0)];
                val += gout11 * dm[(i0+1)*nao+(l0+0)];
                val += gout12 * dm[(i0+2)*nao+(l0+0)];
                val += gout13 * dm[(i0+3)*nao+(l0+0)];
                val += gout14 * dm[(i0+4)*nao+(l0+0)];
                val += gout15 * dm[(i0+5)*nao+(l0+0)];
                val += gout16 * dm[(i0+6)*nao+(l0+0)];
                val += gout17 * dm[(i0+7)*nao+(l0+0)];
                val += gout18 * dm[(i0+8)*nao+(l0+0)];
                val += gout19 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+1)*nao+(k0+0), val);
                val = 0;
                val += gout20 * dm[(i0+0)*nao+(l0+0)];
                val += gout21 * dm[(i0+1)*nao+(l0+0)];
                val += gout22 * dm[(i0+2)*nao+(l0+0)];
                val += gout23 * dm[(i0+3)*nao+(l0+0)];
                val += gout24 * dm[(i0+4)*nao+(l0+0)];
                val += gout25 * dm[(i0+5)*nao+(l0+0)];
                val += gout26 * dm[(i0+6)*nao+(l0+0)];
                val += gout27 * dm[(i0+7)*nao+(l0+0)];
                val += gout28 * dm[(i0+8)*nao+(l0+0)];
                val += gout29 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+2)*nao+(k0+0), val);
                val = 0;
                val += gout30 * dm[(i0+0)*nao+(l0+0)];
                val += gout31 * dm[(i0+1)*nao+(l0+0)];
                val += gout32 * dm[(i0+2)*nao+(l0+0)];
                val += gout33 * dm[(i0+3)*nao+(l0+0)];
                val += gout34 * dm[(i0+4)*nao+(l0+0)];
                val += gout35 * dm[(i0+5)*nao+(l0+0)];
                val += gout36 * dm[(i0+6)*nao+(l0+0)];
                val += gout37 * dm[(i0+7)*nao+(l0+0)];
                val += gout38 * dm[(i0+8)*nao+(l0+0)];
                val += gout39 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+3)*nao+(k0+0), val);
                val = 0;
                val += gout40 * dm[(i0+0)*nao+(l0+0)];
                val += gout41 * dm[(i0+1)*nao+(l0+0)];
                val += gout42 * dm[(i0+2)*nao+(l0+0)];
                val += gout43 * dm[(i0+3)*nao+(l0+0)];
                val += gout44 * dm[(i0+4)*nao+(l0+0)];
                val += gout45 * dm[(i0+5)*nao+(l0+0)];
                val += gout46 * dm[(i0+6)*nao+(l0+0)];
                val += gout47 * dm[(i0+7)*nao+(l0+0)];
                val += gout48 * dm[(i0+8)*nao+(l0+0)];
                val += gout49 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+4)*nao+(k0+0), val);
                val = 0;
                val += gout50 * dm[(i0+0)*nao+(l0+0)];
                val += gout51 * dm[(i0+1)*nao+(l0+0)];
                val += gout52 * dm[(i0+2)*nao+(l0+0)];
                val += gout53 * dm[(i0+3)*nao+(l0+0)];
                val += gout54 * dm[(i0+4)*nao+(l0+0)];
                val += gout55 * dm[(i0+5)*nao+(l0+0)];
                val += gout56 * dm[(i0+6)*nao+(l0+0)];
                val += gout57 * dm[(i0+7)*nao+(l0+0)];
                val += gout58 * dm[(i0+8)*nao+(l0+0)];
                val += gout59 * dm[(i0+9)*nao+(l0+0)];
                atomicAdd(vk+(j0+5)*nao+(k0+0), val);
            
            }
        }
        if (tid == 0) pair_ij = atomicAdd(head, 1);
        __syncthreads();
    }
}
extern "C" __global__ void __launch_bounds__(NTHREADS_3200, MINBLOCKS_3200)
k_3200(KARGS)    { kbody_3200<1>(KFWD); }
extern "C" __global__ void __launch_bounds__(NTHREADS_3200, MINBLOCKS_3200)
k_rs_3200(KARGS) { kbody_3200<2>(KFWD); }

