"""Generate unrolled Rys kernels for the nuclear-gradient two-electron build.

GPU4PySCF computes the per-atom derivative of ``j*J - k*K`` with
``RYS_per_atom_jk_ip1``: 18 unrolled angular-momentum classes and, for
everything else, one general ``rys_ejk_ip1_kernel``.  That build is 75 % of a
B3LYP/6-31G* gradient and 96 % of a wB97M-V/def2-TZVPD one on these clusters,
and both paths leave the same things on the table that ``fastk`` took for the
exchange matrix:

  * ``cicj*ckcl/(aij*akl*sqrt(aij+akl))`` is a double-precision division and a
    sqrt evaluated once per *primitive quartet*, plus ``rt/(aij+akl)``,
    ``.5/aij`` and ``.5/akl`` once per Rys root.  Here 1/aij is block-uniform
    and cached in shared memory, 1/akl is hoisted out of the bra loop, and one
    ``rsqrt`` turns the rest into multiplies.
  * the bra-side primitive data is recomputed per thread per primitive quartet.
  * the *general* kernel stores the density products ``dd`` in a global-memory
    pool and re-reads nf of them per Rys root per primitive quartet; it pads
    the gout tile to 3x3x3x3 whatever the class; and its g-array addresses come
    from shared index tables, so no two loads can be reused.

Two emission styles, chosen per class by measurement (``bench/sweep_ejk.sh``):

``reg``   one thread per shell quartet.  The whole 2D-integral array is a local
          array at compile-time constant indices, so nvcc keeps the live part
          in registers and deletes the rest, and there is no ``__syncthreads``
          anywhere in the primitive loops.  This is GPU4PySCF's own arrangement
          for its 18 unrolled classes, with the scaffolding above fixed.

``shm``   the 2D integrals in shared memory, NSQ shell quartets per block over
          GSTRIDE lanes of ``threadIdx.y``.  A lane owns all nfi cartesian
          functions of the bra-i shell for PER of the (j,k,l) combinations,
          and the (j,k,l) part of every address -- the only part that differs
          between lanes -- becomes base pointers computed once per lane.  This
          is the only style that fits the wide classes.

The gout odometer is ``o = l + nfl*(k + nfk*j)`` with i fastest, *not*
GPU4PySCF's ``i,j,k,l``.  The derivative needs the cartesian powers of j and k
as multipliers (``fj = 2aj*g(j+1) - j*g(j-1)``), and with l as the fastest of
the three odometer digits a lane that owns PER <= nfl combinations has a single
(j,k) pair, so those multipliers are lane-uniform instead of per-element.  The
l power is never needed: the fourth derivative comes from translational
invariance, ``fl = -fi-fj-fk``.

Run:  python gen_ejk.py <classes> --table fastejk_launch.json > fastejk_generated.cu
"""
import sys, os, json, argparse

MAX_PRIM_PAIR = 36              # matches fastejk_prologue.cu
MAX_BLOCK = 1024


def cart(l):
    """(x,y,z) powers of a shell's cartesian functions, in PySCF's order."""
    out = []
    for ix in range(l, -1, -1):
        for iy in range(l - ix, -1, -1):
            out.append((ix, iy, l - ix - iy))
    return out


class Class:
    """Geometry of one (li,lj,lk,ll) for the *gradient*.

    The derivative raises i and k by one, so the 2D-integral array is one wider
    in both (stride_j = li+2, stride_l = stride_k*(lk+2)) and the quadrature
    needs one more Rys root than the energy build of the same class.
    """

    def __init__(self, li, lj, lk, ll):
        self.l = (li, lj, lk, ll)
        self.tag = f'{li}{lj}{lk}{ll}'
        self.lij = li + lj + 1
        self.lkl = lk + ll + 1
        self.sj = li + 2
        self.sk = self.sj * (lj + 1)
        self.sl = self.sk * (lk + 2)
        self.gsz = self.sl * (ll + 1)
        self.nroots = (li + lj + lk + ll + 1) // 2 + 1
        self.nf = [(x + 1) * (x + 2) // 2 for x in self.l]
        self.ngout = self.nf[0] * self.nf[1] * self.nf[2] * self.nf[3]
        self.cart = [cart(x) for x in self.l]

    def base(self, cj, ck, cl, d):
        """The (j,k,l) part of a 2D-integral address, plus the direction plane."""
        return (self.cart[1][cj][d] * self.sj + self.cart[2][ck][d] * self.sk +
                self.cart[3][cl][d] * self.sl) + d * self.gsz


# ---------------------------------------------------------------------------
# The Rys recurrences.  Identical to gen_khigh.py's, with this Class's wider
# geometry; `acc(n)` is how the emitted code names 2D-integral element n of the
# current direction, which is the only difference between the two styles.
# ---------------------------------------------------------------------------

def vrr_trr(c, acc):
    o = []
    if c.lij > 0:
        o += ['s0 = ' + acc(0) + ';', 's1 = c0x * s0;', acc(1) + ' = s1;']
        for i in range(1, c.lij):
            o += [f's2 = c0x * s1 + {i} * b10 * s0;', acc(i + 1) + ' = s2;',
                  's0 = s1;', 's1 = s2;']
    if c.lkl > 0:
        for i in range(c.lij + 1):
            o += ['s0 = ' + acc(i) + ';', 's1 = cpx * s0;']
            if i > 0:
                o += [f's1 += {i} * b00 * ' + acc(i - 1) + ';']
            o += [acc(i + c.sk) + ' = s1;']
            for k in range(1, c.lkl):
                o += [f's2 = cpx * s1 + {k} * b01 * s0;']
                if i > 0:
                    o += [f's2 += {i} * b00 * ' + acc(i - 1 + k * c.sk) + ';']
                o += [acc(i + (k + 1) * c.sk) + ' = s2;', 's0 = s1;', 's1 = s2;']
    return o


def hrr_ij_col(c, acc, base=0):
    o = []
    for j in range(1, c.l[1] + 1):
        o += ['s1 = ' + acc(base + (c.lij - j + 1) + (j - 1) * c.sj) + ';']
        for i in range(c.lij - j, -1, -1):
            o += ['s0 = ' + acc(base + i + (j - 1) * c.sj) + ';',
                  acc(base + i + j * c.sj) + ' = s1 - xjxi * s0;', 's1 = s0;']
    return o


def hrr_kl_col(c, acc, base=0):
    o = []
    for l in range(1, c.l[3] + 1):
        o += ['s1 = ' + acc(base + (c.lkl - l + 1) * c.sk + (l - 1) * c.sl) + ';']
        for k in range(c.lkl - l, -1, -1):
            o += ['s0 = ' + acc(base + k * c.sk + (l - 1) * c.sl) + ';',
                  acc(base + k * c.sk + l * c.sl) + ' = s1 - xlxk * s0;',
                  's1 = s0;']
    return o


def recurrence_all(c, acc):
    """Every recurrence for one cartesian direction, in one straight line."""
    o = vrr_trr(c, acc)
    if c.l[1] > 0:
        for k in range(c.lkl + 1):
            o += hrr_ij_col(c, acc, k * c.sk)
    if c.l[3] > 0:
        for m in range(c.sk):
            o += hrr_kl_col(c, acc, m)
    return o


# ---------------------------------------------------------------------------
# The derivative contraction.
#
#   fi = 2ai*g(i+1) - i*g(i-1)          fk = 2ak*g(k+1) - k*g(k-1)
#   fj = 2aj*g(i+1,j) - 2aj*(rj-ri)*g(i,j) - j*g(j-1)      [ = 2aj*g(j+1) - ... ]
#   fl = -fi - fj - fk                  (translational invariance)
#
# and each of the twelve is contracted with the product of the other two
# cartesian directions times the density factor dd.  Exactly GPU4PySCF's
# rys_ejk_ip1_kernel, with every address a compile-time constant.
# ---------------------------------------------------------------------------
AX = ('x', 'y', 'z')


def force_terms(c, g, ipow, jpow, kpow, dd, other):
    """One cartesian function's twelve accumulations.

    ``g(off)`` names 2D-integral element ``off`` of direction ``d``; ``ipow``
    is a compile-time int, ``jpow``/``kpow`` may be C expressions (they are
    lane-uniform in the shm style).  ``other`` is the product of the other two
    directions times dd.
    """
    o = []
    for d in range(3):
        a, p, q = AX[d], ipow[d], other[d]
        o.append(f'gi{a} = {g(d, 1)};')
        o.append(f'f{a}i = ai2 * gi{a}' +
                 (f' - {p} * {g(d, -1)}' if p else '') + ';')
        o.append(f'f{a}j = aj2 * (gi{a} - rjri{d} * I{a})' +
                 (f' - ({jpow[d]}) * {g(d, -c.sj)}' if jpow[d] != '0' else '') + ';')
        o.append(f'f{a}k = ak2 * {g(d, c.sk)}' +
                 (f' - ({kpow[d]}) * {g(d, -c.sk)}' if kpow[d] != '0' else '') + ';')
        o.append(f'v_i{a} += f{a}i * {q};')
        o.append(f'v_j{a} += f{a}j * {q};')
        o.append(f'v_k{a} += f{a}k * {q};')
        o.append(f'v_l{a} -= (f{a}i + f{a}j + f{a}k) * {q};')
    return o


# ---------------------------------------------------------------------------
# reg style: one thread per shell quartet, everything at compile-time indices.
# ---------------------------------------------------------------------------

def quads_reg(c):
    """(ci,cj,ck,cl) in the emitted gout order: i fastest, then l, then k, j."""
    for cj in range(c.nf[1]):
        for ck in range(c.nf[2]):
            for cl in range(c.nf[3]):
                for ci in range(c.nf[0]):
                    yield ci, cj, ck, cl


def dd_reg(c):
    o = ['int i0 = ao_loc[ish], j0 = ao_loc[jsh];',
         'int k0 = ao_loc[ksh], l0 = ao_loc[lsh];',
         'double dfac;']
    o.append('#pragma unroll')
    o.append(f'for (int n = 0; n < {c.ngout}; ++n) dd_[n] = 0.;')
    for do, fac, expr in (
            ('do_k', 'k_factor',
             'dm[(j0+{cj})*nao+(k0+{ck})] * dm[(l0+{cl})*nao+(i0+{ci})] + '
             'dm[(j0+{cj})*nao+(l0+{cl})] * dm[(k0+{ck})*nao+(i0+{ci})]'),
            ('do_j', 'j_factor',
             'dm[(l0+{cl})*nao+(k0+{ck})] * dm[(j0+{cj})*nao+(i0+{ci})]')):
        o.append(f'if ({do}) {{')
        o.append(f'    dfac = fac_sym * {fac};')
        for n, (ci, cj, ck, cl) in enumerate(quads_reg(c)):
            o.append(f'    dd_[{n}] += dfac * (' +
                     expr.format(ci=ci, cj=cj, ck=ck, cl=cl) + ');')
        o.append('}')
    return o


def contract_reg(c):
    o = []
    for n, (ci, cj, ck, cl) in enumerate(quads_reg(c)):
        b = [c.base(cj, ck, cl, d) for d in range(3)]
        ip = c.cart[0][ci]
        jp = [str(x) for x in c.cart[1][cj]]
        kp = [str(x) for x in c.cart[2][ck]]
        g = lambda d, off: f'G[{b[d] + ip[d] + off}]'
        o.append(f'dd = dd_[{n}];')
        for d in range(3):
            o.append(f'I{AX[d]} = {g(d, 0)};')
        o += ['pxy = Ix * Iy * dd;', 'pxz = Ix * Iz * dd;', 'pyz = Iy * Iz * dd;']
        o += force_terms(c, g, ip, jp, kp, 'dd', ('pyz', 'pxz', 'pxy'))
    return o


def recur_reg(c):
    o = []
    for d in range(3):
        a = AX[d]
        acc = lambda n, d=d: f'G[{d * c.gsz + n}]'
        o.append('{')
        o.append(f'    double c0x = {a}pa - rt_aij * {a}pq;')
        o.append(f'    double cpx = {a}qc + rt_akl * {a}pq;')
        o.append(f'    double xjxi = rjri{d}, xlxk = rlrk{d};')
        o.append('    (void)c0x; (void)cpx; (void)xjxi; (void)xlxk;')
        o += ['    ' + s for s in recurrence_all(c, acc)]
        o.append('}')
    return o


HEAD_REG = r'''
// ---- (%(li)d%(lj)d%(lk)d%(ll)d) reg : %(ngout)d gout, g_size %(gsz)d, %(nroots)d roots, one thread per quartet
extern "C" __global__ void __launch_bounds__(%(nthreads)d, %(minblocks)d)
ejk_%(tag)s(EARGS)
{
    constexpr int NROOTS = %(nroots)d;
    constexpr int GSZ = %(gsz)d;
    constexpr int threads = %(nthreads)d;
    __shared__ double s_aij[%(mpp)d], s_iaij[%(mpp)d], s_ajaij[%(mpp)d];
    __shared__ double s_xij[%(mpp)d], s_yij[%(mpp)d], s_zij[%(mpp)d];
    __shared__ double s_cicj[%(mpp)d], s_ai2[%(mpp)d], s_aj2[%(mpp)d];
    __shared__ int ntasks, pair_ij, ish, jsh;
    __shared__ double ri[3], rjri[3];
    int thread_id = threadIdx.x;
    int *bas_kl_idx = pool + blockIdx.x * queue_depth;
    int do_j = j_factor != 0.;
    int do_k = k_factor != 0.;
    if (thread_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
while (pair_ij < npairs_ij) {
    int bas_ij = pair_ij_mapping[pair_ij];
    if (thread_id == 0) {
        ntasks = 0;
        ish = bas_ij / nbas;
        jsh = bas_ij %% nbas;
    }
    __syncthreads();
    fill_ejk_tasks(&ntasks, bas_kl_idx, bas_ij, nbas, pair_kl_mapping,
                   npairs_kl, q_cond, dm_cond, cutoff, do_j, do_k);
    if (ntasks == 0) {
        if (thread_id == 0) { pair_ij = atomicAdd(head, 1); }
        __syncthreads();
        continue;
    }
    if (thread_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[thread_id] = env[ri_ptr+thread_id];
        rjri[thread_id] = env[rj_ptr+thread_id] - ri[thread_id];
    }
    __syncthreads();
    int iprim = iprim_a, jprim = jprim_a;
    {
        const double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        const double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        const double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        const double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double rr_ij = rjri[0]*rjri[0] + rjri[1]*rjri[1] + rjri[2]*rjri[2];
        for (int ij = thread_id; ij < iprim*jprim; ij += threads) {
            double ai = expi[ij/jprim];
            double aj = expj[ij%%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij[ij] = aij;  s_iaij[ij] = iaij;  s_ajaij[ij] = aj_aij;
            s_ai2[ij] = ai + ai;  s_aj2[ij] = aj + aj;
            s_xij[ij] = ri[0] + rjri[0] * aj_aij;
            s_yij[ij] = ri[1] + rjri[1] * aj_aij;
            s_zij[ij] = ri[2] + rjri[2] * aj_aij;
            s_cicj[ij] = ci[ij/jprim] * cj[ij%%jprim] * exp(-ai * aj_aij * rr_ij);
        }
    }
    __syncthreads();
    double rjri0 = rjri[0], rjri1 = rjri[1], rjri2 = rjri[2];
    double v_ix = 0, v_iy = 0, v_iz = 0, v_jx = 0, v_jy = 0, v_jz = 0;
    int kprim = kprim_a, lprim = lprim_a;

    for (int task_id = thread_id; task_id < ntasks; task_id += threads) {
        int bas_kl = bas_kl_idx[task_id];
        int ksh = bas_kl / nbas;
        int lsh = bas_kl %% nbas;
        double fac_sym = PI_FAC;
        if (ish == jsh) fac_sym *= .5;
        if (ksh == lsh) fac_sym *= .5;
        if (bas_ij == bas_kl) fac_sym *= .5;
        const double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        const double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        const double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        const double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        const double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        const double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        double rlrk0 = rl[0]-rk[0], rlrk1 = rl[1]-rk[1], rlrk2 = rl[2]-rk[2];
        double rr_kl = rlrk0*rlrk0 + rlrk1*rlrk1 + rlrk2*rlrk2;
        double dd_[%(ngout)d];
%(dd)s
        double v_kx = 0, v_ky = 0, v_kz = 0, v_lx = 0, v_ly = 0, v_lz = 0;
        double G[3*GSZ];
        double rw[2*NROOTS];
        double dd, Ix, Iy, Iz, pxy, pxz, pyz;
        double gix, giy, giz;
        double fxi, fyi, fzi, fxj, fyj, fzj, fxk, fyk, fzk;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            double ak = expk[klp/lprim];
            double al = expl[klp%%lprim];
            double akl = ak + al;
            double iakl = 1. / akl;
            double al_akl = al * iakl;
            double ak2 = ak + ak;
            double ckcl = ck[klp/lprim] * cl[klp%%lprim]
                        * exp(-ak * al_akl * rr_kl);
            double xqc = rlrk0 * al_akl, yqc = rlrk1 * al_akl, zqc = rlrk2 * al_akl;
            double xkl = rk[0] + xqc, ykl = rk[1] + yqc, zkl = rk[2] + zqc;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                double aij = s_aij[ijp];
                double iaij = s_iaij[ijp];
                double aj_aij = s_ajaij[ijp];
                double ai2 = s_ai2[ijp], aj2 = s_aj2[ijp];
                // inv_om2 = 0 is the full-range operator; 1/omega^2 the
                // long-range (erf) one -- scaling the Rys argument and roots
                // by w^2/(w^2+theta) is exactly this one fma.
                double inv_s = rsqrt(fma(aij * akl, inv_om2, aij + akl));
                double inv_s2 = inv_s * inv_s;
                double xpa = rjri0 * aj_aij;
                double ypa = rjri1 * aj_aij;
                double zpa = rjri2 * aj_aij;
                double xpq = s_xij[ijp] - xkl;
                double ypq = s_yij[ijp] - ykl;
                double zpq = s_zij[ijp] - zkl;
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                G[0]   = ckcl;
                G[GSZ] = s_cicj[ijp] * iaij * iakl * inv_s * coef0;
                ejk_roots_reg<NROOTS>(aij * akl * inv_s2 * rr, rw, tab);
                // deliberately not unrolled: unrolling keeps NROOTS
                // generations of the 2D-integral array live at once and spills
                for (int irys = 0; irys < NROOTS; ++irys) {
                    double rt = rw[2*irys];
                    G[2*GSZ] = rw[2*irys+1];
                    double rt_aa = rt * inv_s2;
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double b00 = .5 * rt_aa;
                    double b01 = .5*iakl * (1 - rt_akl);
                    double s0, s1, s2;
                    (void)b10; (void)b00; (void)b01; (void)s2;
%(recur)s
%(contract)s
                }
            }
        }
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
        int la = bas[lsh*BAS_SLOTS+ATOM_OF];
        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);
    }
    // every thread of the block shares ish/jsh, so the bra half reduces inside
    // the warp before it reaches the six atomics
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v_ix += __shfl_down_sync(0xffffffff, v_ix, off);
        v_iy += __shfl_down_sync(0xffffffff, v_iy, off);
        v_iz += __shfl_down_sync(0xffffffff, v_iz, off);
        v_jx += __shfl_down_sync(0xffffffff, v_jx, off);
        v_jy += __shfl_down_sync(0xffffffff, v_jy, off);
        v_jz += __shfl_down_sync(0xffffffff, v_jz, off);
    }
    if ((thread_id & 31) == 0) {
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        atomicAdd(ejk+ia*3+0, v_ix);
        atomicAdd(ejk+ia*3+1, v_iy);
        atomicAdd(ejk+ia*3+2, v_iz);
        atomicAdd(ejk+ja*3+0, v_jx);
        atomicAdd(ejk+ja*3+1, v_jy);
        atomicAdd(ejk+ja*3+2, v_jz);
    }
    __syncthreads();
    if (thread_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
}
}
'''


# ---------------------------------------------------------------------------
# shm style: NSQ shell quartets per block over GSTRIDE lanes.
# ---------------------------------------------------------------------------

def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def per_choices(c):
    """PER values whose lane blocks stay aligned to the (j,k,l) odometer.

    A lane owns PER consecutive combinations of ``o = l + nfl*(k + nfk*j)``
    starting at ``gout_id*PER``.  Keeping the block from straddling a digit is
    what makes each of j, k and l a runtime base plus a compile-time offset --
    and, for PER <= nfl, makes (j,k) a single pair for the whole lane, so the
    cartesian powers the derivative needs as multipliers are lane-uniform.
    """
    nfj, nfk, nfl = c.nf[1], c.nf[2], c.nf[3]
    out = list(_divisors(nfl))
    out += [nfl * q for q in _divisors(nfk)]
    out += [nfl * nfk * q for q in _divisors(nfj)]
    out = sorted(set(out))
    assert all(nfj * nfk * nfl % p == 0 for p in out), c.tag
    return out


def deltas(c, per):
    """(dj, dk, dl) offsets of each of a lane's PER combinations."""
    nfk, nfl = c.nf[2], c.nf[3]
    if per <= nfl:
        d = [(0, 0, m) for m in range(per)]
    elif per <= nfl * nfk:
        d = [(0, m // nfl, m % nfl) for m in range(per)]
    else:
        d = [(m // (nfl * nfk), (m // nfl) % nfk, m % nfl) for m in range(per)]
    nO = c.nf[1] * nfk * nfl
    for gid in range(nO // per):
        o0 = gid * per
        jb, kb, lb = o0 // (nfl * nfk), (o0 // nfl) % nfk, o0 % nfl
        for m, (dj, dk, dl) in enumerate(d):
            o = o0 + m
            assert (jb + dj, kb + dk, lb + dl) == \
                   (o // (nfl * nfk), (o // nfl) % nfk, o % nfl), (c.tag, per)
    return d


def _lexoff(l):
    return l * (l + 1) * (l + 2) // 2


def lane_pre(c, per, d):
    """Per-lane base addresses and cartesian powers.  Depend only on
    threadIdx.y, so they are computed once, outside every loop."""
    nfk, nfl = c.nf[2], c.nf[3]
    o = [f'    int o_base = gout_id * {per};',
         f'    int lb_ = o_base % {nfl};',
         f'    int kb_ = (o_base / {nfl}) % {nfk};',
         f'    int jb_ = o_base / {nfl * nfk};',
         f'    int gb[{3 * per}], jc[{3 * per}], kc[{3 * per}];']
    for m, (dj, dk, dl) in enumerate(d):
        o.append(f'    {{ int pj = (jb_+{dj})*3, pk = (kb_+{dk})*3, '
                 f'pl = (lb_+{dl})*3;')
        for dd in range(3):
            o.append(f'      jc[{3*m+dd}] = c_lex[{_lexoff(c.l[1])}+pj+{dd}];')
            o.append(f'      kc[{3*m+dd}] = c_lex[{_lexoff(c.l[2])}+pk+{dd}];')
            o.append(f'      gb[{3*m+dd}] = (jc[{3*m+dd}]*{c.sj} '
                     f'+ kc[{3*m+dd}]*{c.sk} '
                     f'+ c_lex[{_lexoff(c.l[3])}+pl+{dd}]*{c.sl})*NSQ '
                     f'+ {dd}*g_size*NSQ;')
        o.append('    }')
    return '\n'.join(o)


def dd_shm(c, per, d):
    nfi = c.nf[0]
    o = ['int i0 = ao_loc[ish], j0 = ao_loc[jsh];',
         'int k0 = ao_loc[ksh], l0 = ao_loc[lsh];',
         'int jj = j0 + jb_, kk = k0 + kb_, ll_ = l0 + lb_;',
         'double dfac;',
         '#pragma unroll',
         f'for (int n = 0; n < {per * nfi}; ++n) dd_[n] = 0.;']
    for do, fac, expr in (
            ('do_k', 'k_factor',
             'dm[(jj+{dj})*nao+(kk+{dk})] * dm[(ll_+{dl})*nao+(i0+{ci})] + '
             'dm[(jj+{dj})*nao+(ll_+{dl})] * dm[(kk+{dk})*nao+(i0+{ci})]'),
            ('do_j', 'j_factor',
             'dm[(ll_+{dl})*nao+(kk+{dk})] * dm[(jj+{dj})*nao+(i0+{ci})]')):
        o.append(f'if ({do}) {{')
        o.append(f'    dfac = fac_sym * {fac};')
        for m, (dj, dk, dl) in enumerate(d):
            for ci in range(nfi):
                o.append(f'    dd_[{m*nfi+ci}] += dfac * (' +
                         expr.format(ci=ci, dj=dj, dk=dk, dl=dl) + ');')
        o.append('}')
    return o


def contract_shm(c, per):
    nfi = c.nf[0]
    o = []
    for m in range(per):
        for dd in range(3):
            o.append(f'const double *gm{m}_{dd} = gx + gb[{3*m+dd}];')
        for ci in range(nfi):
            ip = c.cart[0][ci]
            g = lambda d, off, ip=ip, m=m: f'gm{m}_{d}[{ip[d] + off}*NSQ]'
            jp = [f'jc[{3*m+d}]' for d in range(3)]
            kp = [f'kc[{3*m+d}]' for d in range(3)]
            o.append(f'dd = dd_[{m*nfi+ci}];')
            for d in range(3):
                o.append(f'I{AX[d]} = {g(d, 0)};')
            o += ['pxy = Ix * Iy * dd;', 'pxz = Ix * Iz * dd;',
                  'pyz = Iy * Iz * dd;']
            o += force_terms(c, g, ip, jp, kp, 'dd', ('pyz', 'pxz', 'pxy'))
    return o


def recur_shm(c, ind, split):
    """VRR/TRR on three lanes; the two horizontal transfers either on the same
    three lanes (fused) or spread over every lane (split)."""
    p = ' ' * ind
    acc = lambda n: f'_gx[{n}*NSQ]'
    o = [p + 'for (int n = gout_id; n < 3; n += GSTRIDE) {',
         p + '    if (n == 2) { gx[2*g_size*NSQ] = rw[(2*irys+1)*NSQ]; }',
         p + '    double *_gx = gx + n * g_size * NSQ;',
         p + '    double xjxi = rjri[n], xlxk = rlrk[n*NSQ];',
         p + '    double c0x = xjxi * s_ajaij[ijp] - rt_aij * Rpq[n*NSQ];',
         p + '    double cpx = xlxk * al_akl + rt_akl * Rpq[n*NSQ];',
         p + '    (void)c0x; (void)cpx; (void)xjxi; (void)xlxk;']
    body = vrr_trr(c, acc) if split else recurrence_all(c, acc)
    o += [p + '    ' + s for s in body]
    o.append(p + '}')
    if not split:
        return '\n'.join(o)
    if c.l[1] > 0:
        o += [p + '__syncthreads();',
              p + f'for (int t = gout_id; t < {(c.lkl+1)*3}; t += GSTRIDE) {{',
              p + '    int d_ = t % 3, kc_ = t / 3;',
              p + f'    double *_gx = gx + (d_ * g_size + kc_ * {c.sk}) * NSQ;',
              p + '    double xjxi = rjri[d_];']
        o += [p + '    ' + s for s in hrr_ij_col(c, acc)]
        o.append(p + '}')
    if c.l[3] > 0:
        o += [p + '__syncthreads();',
              p + f'for (int t = gout_id; t < {c.sk*3}; t += GSTRIDE) {{',
              p + '    int d_ = t % 3, m_ = t / 3;',
              p + '    double *_gx = gx + (d_ * g_size + m_) * NSQ;',
              p + '    double xlxk = rlrk[d_*NSQ];']
        o += [p + '    ' + s for s in hrr_kl_col(c, acc)]
        o.append(p + '}')
    return '\n'.join(o)


def lex_table(lmax=4):
    rows = []
    for l in range(lmax + 1):
        for p in cart(l):
            rows.append('    %d, %d, %d,' % p)
    return ('\n// the (x,y,z) powers of each cartesian function, PySCF order\n'
            '__device__ const int c_lex[] = {\n' + '\n'.join(rows) + '\n};\n')


HEAD_SHM = r'''
// ---- (%(li)d%(lj)d%(lk)d%(ll)d) shm : %(ngout)d gout over %(gs)d lanes, %(nsq)d quartets/block, %(w)d dd/lane
extern "C" __global__ void __launch_bounds__(%(nthreads)d, %(minblocks)d)
ejk_%(tag)s(EARGS)
{
    constexpr int NSQ = %(nsq)d;
    constexpr int GSTRIDE = %(gs)d;
    constexpr int NROOTS = %(nroots)d;
    constexpr int g_size = %(gsz)d;
    constexpr int PAD = %(pad)d;
    constexpr int HEADSZ = (10 + PAD) * NSQ;
    constexpr int threads = NSQ * GSTRIDE;
    __shared__ double s_aij[%(mpp)d], s_iaij[%(mpp)d], s_ajaij[%(mpp)d];
    __shared__ double s_xij[%(mpp)d], s_yij[%(mpp)d], s_zij[%(mpp)d];
    __shared__ double s_cicj[%(mpp)d], s_ai2[%(mpp)d], s_aj2[%(mpp)d];
    __shared__ int ntasks, pair_ij, ish, jsh;
    __shared__ double ri[3], rjri[3];
    extern __shared__ double shared_memory[];
    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int thread_id = NSQ * gout_id + sq_id;
    int *bas_kl_idx = pool + blockIdx.x * queue_depth;
    int do_j = j_factor != 0.;
    int do_k = k_factor != 0.;
    double *rlrk = shared_memory + sq_id;
    double *Rpq  = shared_memory + NSQ*3 + sq_id;
    double *akl_cache = shared_memory + NSQ*6 + sq_id;
    double *gx = shared_memory + (10 + PAD)*NSQ + sq_id;
    double *rw = shared_memory + (10 + PAD + 3*g_size)*NSQ + sq_id;
    // the reduction buffer sits on top of the 2D integrals and the roots,
    // which are both rebuilt by the next task; it must stay clear of the pad
    double *reduce = shared_memory + HEADSZ + thread_id;
    // the j and k lowering terms read g(j-1) and g(k-1) at a lane-uniform
    // power that may be zero; the multiplier is then zero but the address is
    // still evaluated, so it lands in this zeroed pad instead of out of bounds
    for (int t = thread_id; t < PAD*NSQ; t += threads) {
        shared_memory[10*NSQ + t] = 0.;
    }
%(lanepre)s
    if (thread_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
while (pair_ij < npairs_ij) {
    int bas_ij = pair_ij_mapping[pair_ij];
    if (thread_id == 0) {
        ntasks = 0;
        ish = bas_ij / nbas;
        jsh = bas_ij %% nbas;
    }
    __syncthreads();
    fill_ejk_tasks(&ntasks, bas_kl_idx, bas_ij, nbas, pair_kl_mapping,
                   npairs_kl, q_cond, dm_cond, cutoff, do_j, do_k);
    if (ntasks == 0) {
        if (thread_id == 0) { pair_ij = atomicAdd(head, 1); }
        __syncthreads();
        continue;
    }
    if (thread_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[thread_id] = env[ri_ptr+thread_id];
        rjri[thread_id] = env[rj_ptr+thread_id] - ri[thread_id];
    }
    __syncthreads();
    int iprim = iprim_a, jprim = jprim_a;
    {
        const double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        const double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        const double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        const double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double rr_ij = rjri[0]*rjri[0] + rjri[1]*rjri[1] + rjri[2]*rjri[2];
        for (int ij = thread_id; ij < iprim*jprim; ij += threads) {
            double ai = expi[ij/jprim];
            double aj = expj[ij%%jprim];
            double aij = ai + aj;
            double iaij = 1. / aij;
            double aj_aij = aj * iaij;
            s_aij[ij] = aij;  s_iaij[ij] = iaij;  s_ajaij[ij] = aj_aij;
            s_ai2[ij] = ai + ai;  s_aj2[ij] = aj + aj;
            s_xij[ij] = ri[0] + rjri[0] * aj_aij;
            s_yij[ij] = ri[1] + rjri[1] * aj_aij;
            s_zij[ij] = ri[2] + rjri[2] * aj_aij;
            s_cicj[ij] = ci[ij/jprim] * cj[ij%%jprim] * exp(-ai * aj_aij * rr_ij);
        }
    }
    __syncthreads();
    double rjri0 = rjri[0], rjri1 = rjri[1], rjri2 = rjri[2];
    double v_ix = 0, v_iy = 0, v_iz = 0, v_jx = 0, v_jy = 0, v_jz = 0;
    int kprim = kprim_a, lprim = lprim_a;

    for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += NSQ) {
        __syncthreads();
        int bas_kl = bas_kl_idx[task_id];
        int ksh = bas_kl / nbas;
        int lsh = bas_kl %% nbas;
        double fac_sym = PI_FAC;
        if (task_id < ntasks) {
            if (ish == jsh) fac_sym *= .5;
            if (ksh == lsh) fac_sym *= .5;
            if (bas_ij == bas_kl) fac_sym *= .5;
        } else {
            fac_sym = 0;
        }
        const double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        const double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            rlrk[0*NSQ] = rl[0] - rk[0];
            rlrk[1*NSQ] = rl[1] - rk[1];
            rlrk[2*NSQ] = rl[2] - rk[2];
        }
        double dd_[%(w)d];
        {
%(dd)s
        }
        double v_kx = 0, v_ky = 0, v_kz = 0, v_lx = 0, v_ly = 0, v_lz = 0;
        double dd, Ix, Iy, Iz, pxy, pxz, pyz;
        double gix, giy, giz;
        double fxi, fyi, fzi, fxj, fyj, fzj, fxk, fyk, fzk;
        for (int klp = 0; klp < kprim*lprim; ++klp) {
            __syncthreads();
            if (gout_id == 0) {
                const double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
                const double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
                const double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
                const double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
                double ak = expk[klp/lprim];
                double al = expl[klp%%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
                double al_akl = al * iakl;
                double xlxk = rlrk[0*NSQ], ylyk = rlrk[1*NSQ], zlzk = rlrk[2*NSQ];
                double Kcd = exp(-ak * al_akl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                gx[0] = ck[klp/lprim] * cl[klp%%lprim] * Kcd;
                akl_cache[0*NSQ] = akl;
                akl_cache[1*NSQ] = al_akl;
                akl_cache[2*NSQ] = iakl;
                akl_cache[3*NSQ] = ak + ak;
            }
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                __syncthreads();
                double aij = s_aij[ijp];
                double iaij = s_iaij[ijp];
                double ai2 = s_ai2[ijp], aj2 = s_aj2[ijp];
                double akl = akl_cache[0*NSQ];
                double al_akl = akl_cache[1*NSQ];
                double iakl = akl_cache[2*NSQ];
                double ak2 = akl_cache[3*NSQ];
                // inv_om2 = 0 is the full-range operator, 1/omega^2 the
                // long-range (erf) one -- one fma, see fastejk_prologue.cu
                double inv_s = rsqrt(fma(aij * akl, inv_om2, aij + akl));
                double inv_s2 = inv_s * inv_s;
                double xkl = rk[0] + rlrk[0*NSQ] * al_akl;
                double ykl = rk[1] + rlrk[1*NSQ] * al_akl;
                double zkl = rk[2] + rlrk[2*NSQ] * al_akl;
                double xpq = s_xij[ijp] - xkl;
                double ypq = s_yij[ijp] - ykl;
                double zpq = s_zij[ijp] - zkl;
                if (gout_id == 0) {
                    Rpq[0*NSQ] = xpq;
                    Rpq[1*NSQ] = ypq;
                    Rpq[2*NSQ] = zpq;
                    gx[NSQ*g_size] = s_cicj[ijp] * iaij * iakl * inv_s * coef0;
                }
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                ejk_roots_tab<NROOTS>(aij*akl*inv_s2 * rr, rw, NSQ, gout_id,
                                      GSTRIDE, tab);
                for (int irys = 0; irys < NROOTS; ++irys) {
                    __syncthreads();
                    double s0, s1, s2;
                    double rt = rw[2*irys*NSQ];
                    double rt_aa = rt * inv_s2;
                    double rt_aij = rt_aa * akl;
                    double rt_akl = rt_aa * aij;
                    double b10 = .5*iaij * (1 - rt_aij);
                    double b00 = .5 * rt_aa;
                    double b01 = .5*iakl * (1 - rt_akl);
                    (void)b10; (void)b00; (void)b01; (void)s2;
%(recur)s
                    __syncthreads();
%(contract)s
                }
            }
        }
        __syncthreads();
        reduce[0*threads] = v_kx;  reduce[1*threads] = v_ky;
        reduce[2*threads] = v_kz;  reduce[3*threads] = v_lx;
        reduce[4*threads] = v_ly;  reduce[5*threads] = v_lz;
        // GSTRIDE need not be a power of two (nfj*nfk*nfl/PER is not), so
        // the tree starts at the next power of two and guards the odd half
        for (int i = %(red1)d; i > 0; i >>= 1) {
            __syncthreads();
            if (gout_id < i && gout_id + i < GSTRIDE) {
#pragma unroll
                for (int n = 0; n < 6; ++n) {
                    reduce[n*threads] += reduce[n*threads+i*NSQ];
                }
            }
        }
        if (gout_id == 0 && task_id < ntasks) {
            int ka = bas[ksh*BAS_SLOTS+ATOM_OF];
            int la = bas[lsh*BAS_SLOTS+ATOM_OF];
            atomicAdd(ejk+ka*3+0, reduce[0*threads]);
            atomicAdd(ejk+ka*3+1, reduce[1*threads]);
            atomicAdd(ejk+ka*3+2, reduce[2*threads]);
            atomicAdd(ejk+la*3+0, reduce[3*threads]);
            atomicAdd(ejk+la*3+1, reduce[4*threads]);
            atomicAdd(ejk+la*3+2, reduce[5*threads]);
        }
    }
    __syncthreads();
    reduce[0*threads] = v_ix;  reduce[1*threads] = v_iy;
    reduce[2*threads] = v_iz;  reduce[3*threads] = v_jx;
    reduce[4*threads] = v_jy;  reduce[5*threads] = v_jz;
    for (int i = %(red2)d; i > 0; i >>= 1) {
        __syncthreads();
        if (thread_id < i && thread_id + i < threads) {
#pragma unroll
            for (int n = 0; n < 6; ++n) {
                reduce[n*threads] += reduce[n*threads+i];
            }
        }
    }
    if (thread_id == 0) {
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        atomicAdd(ejk+ia*3+0, reduce[0*threads]);
        atomicAdd(ejk+ia*3+1, reduce[1*threads]);
        atomicAdd(ejk+ia*3+2, reduce[2*threads]);
        atomicAdd(ejk+ja*3+0, reduce[3*threads]);
        atomicAdd(ejk+ja*3+1, reduce[4*threads]);
        atomicAdd(ejk+ja*3+2, reduce[5*threads]);
    }
    __syncthreads();
    if (thread_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
}
}
'''


def _snap(n):
    """Round the quartets per block so a warp stays inside one lane."""
    if n >= 32:
        return n // 32 * 32
    p = 1
    while p * 2 <= n:
        p *= 2
    return p if n >= 1 else 0


# registers a lane needs besides dd: the twelve force accumulators, the lane
# base tables and the working values of one contraction step.  Measured as the
# smallest headroom at which no class spills to local memory.
REG_OVERHEAD = 72


def scheme_shm(c, threads, gout_max, shm_max):
    """Largest PER -- hence the most shell quartets per block -- that fits.

    Both limits bind: shared memory caps the 2D integrals per block, and the
    64k registers of an SM divided by the block size cap dd, which is 2*PER*nfi
    of them.  A block that is too wide spills dd to local memory, which costs
    more than the lanes buy.
    """
    nO = c.nf[1] * c.nf[2] * c.nf[3]
    best = None
    for per in per_choices(c):
        w = per * c.nf[0]
        if w > gout_max:
            continue
        gs = nO // per
        if gs > MAX_BLOCK:
            continue
        nsq = _snap(threads // gs)
        while nsq and 65536 // (nsq * gs) < 2 * w + REG_OVERHEAD:
            nsq = _snap(nsq - 1)
        while nsq:
            shm = ((10 + c.sk) * nsq +
                   max((3 * c.gsz + 2 * c.nroots) * nsq, 6 * nsq * gs))
            if shm * 8 <= shm_max and nsq * gs <= MAX_BLOCK:
                break
            nsq = _snap(nsq - 1)
        if nsq == 0:
            continue
        headsz = (10 + c.sk) * nsq
        shm = headsz + max((3 * c.gsz + 2 * c.nroots) * nsq, 6 * nsq * gs)
        if best is None or (nsq, per) > (best[0], best[2]):
            best = (nsq, gs, per, w, shm, headsz)
    return best


def _nextpow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def generate(classes, threads=256, gout_max=64, shm_max=48 * 1024 - 2688,
             minblocks=1, table=None, force_style=None, reg_max=64,
             split=False, reg_threads=256, mb_map=None, style_map=None):
    out, tab = [lex_table()], {}
    for spec in classes:
        c = Class(*(int(x) for x in spec))
        style = (style_map or {}).get(c.tag, force_style)
        minblocks = (mb_map or {}).get(c.tag, minblocks)
        if style is None:
            style = 'reg' if (c.ngout <= reg_max and 3 * c.gsz <= 128) else 'shm'
        if style == 'reg':
            body = HEAD_REG % dict(
                li=c.l[0], lj=c.l[1], lk=c.l[2], ll=c.l[3], tag=c.tag,
                ngout=c.ngout, gsz=c.gsz, nroots=c.nroots, mpp=MAX_PRIM_PAIR,
                nthreads=reg_threads, minblocks=minblocks,
                dd='\n'.join(' ' * 8 + s for s in dd_reg(c)),
                recur='\n'.join(' ' * 20 + s for s in recur_reg(c)),
                contract='\n'.join(' ' * 20 + s for s in contract_reg(c)))
            tab[c.tag] = dict(name='ejk_' + c.tag, style='reg',
                              nsq=reg_threads, gout_stride=1,
                              nroots=c.nroots, g_size=c.gsz, shm_fixed=0,
                              minblocks=minblocks)
            sys.stderr.write(f'ejk_{c.tag} reg: ngout={c.ngout} '
                             f'g_size={c.gsz} nroots={c.nroots}\n')
        else:
            sch = scheme_shm(c, threads, gout_max, shm_max)
            if sch is None:
                sys.stderr.write(f'skip {c.tag}: no thread scheme fits\n')
                continue
            nsq, gs, per, w, shm, headsz = sch
            d = deltas(c, per)
            body = HEAD_SHM % dict(
                li=c.l[0], lj=c.l[1], lk=c.l[2], ll=c.l[3], tag=c.tag,
                ngout=c.ngout, gs=gs, nsq=nsq, w=w, nroots=c.nroots,
                gsz=c.gsz, pad=c.sk, mpp=MAX_PRIM_PAIR,
                nthreads=nsq * gs, minblocks=minblocks,
                red1=_nextpow2(gs) // 2, red2=_nextpow2(nsq * gs) // 2,
                lanepre=lane_pre(c, per, d),
                dd='\n'.join(' ' * 12 + s for s in dd_shm(c, per, d)),
                recur=recur_shm(c, 20, split),
                contract='\n'.join(' ' * 20 + s for s in contract_shm(c, per)))
            tab[c.tag] = dict(name='ejk_' + c.tag, style='shm', nsq=nsq,
                              gout_stride=gs, nroots=c.nroots, g_size=c.gsz,
                              shm_fixed=shm, minblocks=minblocks)
            sys.stderr.write(
                f'ejk_{c.tag} shm: ngout={c.ngout} threads={nsq}x{gs} '
                f'dd/lane={w} g_size={c.gsz} nroots={c.nroots} '
                f'shm={shm*8/1024:.1f}KB\n')
        out.append(body)
    if table:
        with open(table + '.tmp', 'w') as f:
            json.dump(tab, f, indent=1, sort_keys=True)
        os.replace(table + '.tmp', table)
    return '\n'.join(out)


#: Every class an spdf basis reaches.  codegen/build_ejk.sh passes the same
#: list explicitly; this is the copy electrolites._gen uses on demand.
DEFAULT_CLASSES = [f'{i}{j}{k}{l}'
                   for i in range(4) for j in range(i+1)
                   for k in range(i+1) for l in range(k+1)]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('classes', nargs='?', default=','.join(DEFAULT_CLASSES))
    ap.add_argument('--threads', type=int, default=256)
    ap.add_argument('--reg-threads', type=int, default=256)
    ap.add_argument('--gout-max', type=int, default=64,
                    help='doubles of dd each lane holds in registers')
    ap.add_argument('--reg-max', type=int, default=64,
                    help='largest ngout emitted one-thread-per-quartet; 64 is '
                         'the best of 0/48/64/128 measured per class on '
                         'PfPMT/6-31G* (bench/sweep_ejk.sh)')
    ap.add_argument('--shm-max', type=int, default=48 * 1024 - 2688)
    ap.add_argument('--minblocks', type=int, default=1)
    ap.add_argument('--table', default=None)
    ap.add_argument('--style', default=None, choices=['reg', 'shm'])
    ap.add_argument('--split', action='store_true',
                    help='spread the horizontal transfers over every lane')
    ap.add_argument('--minblocks-map', default=None,
                    help='JSON {tag: minblocks}, or a path to one -- the '
                         'per-class __launch_bounds__ occupancy target')
    ap.add_argument('--style-map', default=None,
                    help='JSON {tag: "reg"|"shm"}, or a path to one')
    a = ap.parse_args()

    def _load(x):
        if not x:
            return None
        if os.path.exists(x):
            return json.load(open(x))
        return json.loads(x)

    print(generate(a.classes.split(','), a.threads, a.gout_max, a.shm_max,
                   a.minblocks, a.table, a.style, a.reg_max, a.split,
                   a.reg_threads, _load(a.minblocks_map), _load(a.style_map)))
