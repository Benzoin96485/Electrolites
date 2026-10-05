"""Generate the multi-output Rys gradient kernels for density *pairs*
(catalogue D10).

GPU4PySCF's ``_jk_energies_per_atom`` (``grad/tdrhf.py:293``) evaluates, for
a list of density-matrix pairs ``(D1_t, D2_t)`` with factors ``j_t, k_t``,
the per-atom derivative of ``j_t J[D1_t, D2_t] - k_t K[D1_t, D2_t]``, one
output per pair -- the two-electron term of every TDDFT gradient and NAC.
Its kernel (``rys_contract_jk_ip1.cu``, ``rys_ejk_ip1_multidm_kernel``)
evaluates each integral quartet once per **six** pairs and keeps their
density products in a global-memory pool that it re-reads per Rys root per
primitive quartet; measured, a pass over six pairs costs 2.5x one over a
single pair, i.e. the contraction, not the integrals, is most of it.

These kernels are ``gen_ejk.py``'s ``reg`` style -- one thread per shell
quartet, the 2D integrals at compile-time indices, no division in the
primitive loops -- with ``EJK_NT`` outputs per launch: the density products
of all of them in registers, the twelve force terms of every cartesian
function formed once and contracted with each output's product.  ``EJK_NT``
is a compile-time constant, so the output loop unrolls.

The density factor of one pair is GPU4PySCF's, term for term (both orderings
of a non-symmetric pair, both electrons):

    K:  -k/4 * sum of the eight D1 D2 products over (jk,li), (jl,ki), (ik,lj),
        (il,kj) and their D1 <-> D2 swaps
    J:   j/2 * [(D1_ji + D1_ij)(D2_lk + D2_kl) + (D2_ji + D2_ij)(D1_lk + D1_kl)]

The ``reg`` classes are ``gen_ejk.py``'s; the wider ones use its ``shm``
style (2D integrals in shared memory, a lane owning PER of the (j,k,l)
combinations) with the lane width chosen for ``GOUT_MAX // NT_GEN`` density
products per output, and the two block reductions done once per output.

Run:  python -m electrolites.codegen.gen_ejkpair > fastejkpair_generated.cu
"""
import argparse
import sys

from .gen_ejk import (Class, HEAD_REG, HEAD_SHM, AX, quads_reg, recur_reg,
                      recur_shm, lane_pre, deltas, scheme_shm, _nextpow2,
                      lex_table, DEFAULT_CLASSES)

#: registers of dd a lane may hold, all outputs together; the shm-style
#: scheme is chosen for GOUT_MAX // NT_GEN per output
GOUT_MAX = 64
NT_GEN = 2
SHM_THREADS = 256
SHM_MAX = 48 * 1024 - 2688


def shm_scheme(c):
    """(nsq, gs, per, w, shm, headsz) of a class in the pair family."""
    return scheme_shm(c, SHM_THREADS, GOUT_MAX // NT_GEN, SHM_MAX)


def dd_pair(c):
    o = ['int i0 = ao_loc[ish], j0 = ao_loc[jsh];',
         'int k0 = ao_loc[ksh], l0 = ao_loc[lsh];',
         'const size_t nao2 = (size_t)nao * nao;',
         '#pragma unroll',
         'for (int t = 0; t < EJK_NT; ++t) {',
         '    const double *a = dm + t * nao2;',
         '    const double *b = dm_tot + t * nao2;',
         '    const double kf = fac_sym * -.25 * kfs[t];',
         '    const double jf = fac_sym * .5 * jfs[t];']
    A = lambda p, q: f'a[(p{p})*nao+(p{q})]'
    for n, (ci, cj, ck, cl) in enumerate(quads_reg(c)):
        o.append('    {')
        o.append(f'    const int pi = i0+{ci}, pj = j0+{cj}, '
                 f'pk = k0+{ck}, pl = l0+{cl};')
        k8 = ('a[pj*nao+pk]*b[pl*nao+pi] + a[pj*nao+pl]*b[pk*nao+pi] + '
              'a[pi*nao+pk]*b[pl*nao+pj] + a[pi*nao+pl]*b[pk*nao+pj] + '
              'b[pj*nao+pk]*a[pl*nao+pi] + b[pj*nao+pl]*a[pk*nao+pi] + '
              'b[pi*nao+pk]*a[pl*nao+pj] + b[pi*nao+pl]*a[pk*nao+pj]')
        j2 = ('(a[pj*nao+pi]+a[pi*nao+pj])*(b[pl*nao+pk]+b[pk*nao+pl]) + '
              '(b[pj*nao+pi]+b[pi*nao+pj])*(a[pl*nao+pk]+a[pk*nao+pl])')
        o.append(f'    dd_[t*{c.ngout}+{n}] = '
                 f'(kf != 0. ? kf * ({k8}) : 0.) + (jf != 0. ? jf * ({j2}) : 0.);')
        o.append('    }')
    o.append('}')
    return o


def contract_pair(c):
    o = []
    for n, (ci, cj, ck, cl) in enumerate(quads_reg(c)):
        b = [c.base(cj, ck, cl, d) for d in range(3)]
        ip = c.cart[0][ci]
        jp = [str(x) for x in c.cart[1][cj]]
        kp = [str(x) for x in c.cart[2][ck]]
        g = lambda d, off: f'G[{b[d] + ip[d] + off}]'
        for d in range(3):
            o.append(f'I{AX[d]} = {g(d, 0)};')
        for d in range(3):
            a, p = AX[d], ip[d]
            o.append(f'gi{a} = {g(d, 1)};')
            o.append(f'f{a}i = ai2 * gi{a}' +
                     (f' - {p} * {g(d, -1)}' if p else '') + ';')
            o.append(f'f{a}j = aj2 * (gi{a} - rjri{d} * I{a})' +
                     (f' - ({jp[d]}) * {g(d, -c.sj)}' if jp[d] != '0' else '') + ';')
            o.append(f'f{a}k = ak2 * {g(d, c.sk)}' +
                     (f' - ({kp[d]}) * {g(d, -c.sk)}' if kp[d] != '0' else '') + ';')
        o.append('Iyz = Iy * Iz; Ixz = Ix * Iz; Ixy = Ix * Iy;')
        o.append('#pragma unroll')
        o.append('for (int t = 0; t < EJK_NT; ++t) {')
        o.append(f'    dd = dd_[t*{c.ngout}+{n}];')
        o.append('    pyz = Iyz * dd; pxz = Ixz * dd; pxy = Ixy * dd;')
        for a, q in zip(AX, ('pyz', 'pxz', 'pxy')):
            o.append(f'    v_i{a}[t] += f{a}i * {q};')
            o.append(f'    v_j{a}[t] += f{a}j * {q};')
            o.append(f'    v_k{a}[t] += f{a}k * {q};')
            o.append(f'    v_l{a}[t] -= (f{a}i + f{a}j + f{a}k) * {q};')
        o.append('}')
    return o


def _pairify(src, c):
    """HEAD_REG, rewritten for EJK_NT outputs."""
    rep = [
        ('extern "C" __global__ void __launch_bounds__',
         'extern "C" __global__ void __launch_bounds__'),
        (f'ejk_{c.tag}(EARGS)', f'ejkp_{c.tag}(EARGS_PAIR)'),
        ('    double v_ix = 0, v_iy = 0, v_iz = 0, v_jx = 0, v_jy = 0, v_jz = 0;',
         '    double v_ix[EJK_NT], v_iy[EJK_NT], v_iz[EJK_NT], v_jx[EJK_NT], '
         'v_jy[EJK_NT], v_jz[EJK_NT];\n#pragma unroll\n'
         '    for (int t = 0; t < EJK_NT; ++t) v_ix[t] = v_iy[t] = v_iz[t] = '
         'v_jx[t] = v_jy[t] = v_jz[t] = 0.;'),
        (f'        double dd_[{c.ngout}];',
         f'        double dd_[EJK_NT*{c.ngout}];'),
        ('        double v_kx = 0, v_ky = 0, v_kz = 0, v_lx = 0, v_ly = 0, v_lz = 0;',
         '        double v_kx[EJK_NT], v_ky[EJK_NT], v_kz[EJK_NT], v_lx[EJK_NT], '
         'v_ly[EJK_NT], v_lz[EJK_NT];\n#pragma unroll\n'
         '        for (int t = 0; t < EJK_NT; ++t) v_kx[t] = v_ky[t] = v_kz[t] = '
         'v_lx[t] = v_ly[t] = v_lz[t] = 0.;'),
        ('        double dd, Ix, Iy, Iz, pxy, pxz, pyz;',
         '        double dd, Ix, Iy, Iz, pxy, pxz, pyz, Ixy, Ixz, Iyz;'),
        ('''        atomicAdd(ejk+ka*3+0, v_kx);
        atomicAdd(ejk+ka*3+1, v_ky);
        atomicAdd(ejk+ka*3+2, v_kz);
        atomicAdd(ejk+la*3+0, v_lx);
        atomicAdd(ejk+la*3+1, v_ly);
        atomicAdd(ejk+la*3+2, v_lz);''',
         '''#pragma unroll
        for (int t = 0; t < EJK_NT; ++t) {
            double *e = ejk + t * out_stride;
            atomicAdd(e+ka*3+0, v_kx[t]);
            atomicAdd(e+ka*3+1, v_ky[t]);
            atomicAdd(e+ka*3+2, v_kz[t]);
            atomicAdd(e+la*3+0, v_lx[t]);
            atomicAdd(e+la*3+1, v_ly[t]);
            atomicAdd(e+la*3+2, v_lz[t]);
        }'''),
        ('''#pragma unroll
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
    }''',
         '''    int ia = bas[ish*BAS_SLOTS+ATOM_OF];
    int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
#pragma unroll
    for (int t = 0; t < EJK_NT; ++t) {
        double a0 = v_ix[t], a1 = v_iy[t], a2 = v_iz[t];
        double a3 = v_jx[t], a4 = v_jy[t], a5 = v_jz[t];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a0 += __shfl_down_sync(0xffffffff, a0, off);
            a1 += __shfl_down_sync(0xffffffff, a1, off);
            a2 += __shfl_down_sync(0xffffffff, a2, off);
            a3 += __shfl_down_sync(0xffffffff, a3, off);
            a4 += __shfl_down_sync(0xffffffff, a4, off);
            a5 += __shfl_down_sync(0xffffffff, a5, off);
        }
        if ((thread_id & 31) == 0) {
            double *e = ejk + t * out_stride;
            atomicAdd(e+ia*3+0, a0);
            atomicAdd(e+ia*3+1, a1);
            atomicAdd(e+ia*3+2, a2);
            atomicAdd(e+ja*3+0, a3);
            atomicAdd(e+ja*3+1, a4);
            atomicAdd(e+ja*3+2, a5);
        }
    }'''),
    ]
    for a, b in rep:
        assert a in src, (c.tag, a[:60])
        src = src.replace(a, b)
    return src


def dd_pair_shm(c, per, d):
    nfi = c.nf[0]
    w = per * nfi
    o = ['int i0 = ao_loc[ish], j0 = ao_loc[jsh];',
         'int k0 = ao_loc[ksh], l0 = ao_loc[lsh];',
         'int jj = j0 + jb_, kk = k0 + kb_, ll_ = l0 + lb_;',
         'const size_t nao2 = (size_t)nao * nao;',
         '#pragma unroll',
         'for (int t = 0; t < EJK_NT; ++t) {',
         '    const double *a = dm + t * nao2;',
         '    const double *b = dm_tot + t * nao2;',
         '    const double kf = fac_sym * -.25 * kfs[t];',
         '    const double jf = fac_sym * .5 * jfs[t];']
    k8 = ('a[pj*nao+pk]*b[pl*nao+pi] + a[pj*nao+pl]*b[pk*nao+pi] + '
          'a[pi*nao+pk]*b[pl*nao+pj] + a[pi*nao+pl]*b[pk*nao+pj] + '
          'b[pj*nao+pk]*a[pl*nao+pi] + b[pj*nao+pl]*a[pk*nao+pi] + '
          'b[pi*nao+pk]*a[pl*nao+pj] + b[pi*nao+pl]*a[pk*nao+pj]')
    j2 = ('(a[pj*nao+pi]+a[pi*nao+pj])*(b[pl*nao+pk]+b[pk*nao+pl]) + '
          '(b[pj*nao+pi]+b[pi*nao+pj])*(a[pl*nao+pk]+a[pk*nao+pl])')
    for m, (dj, dk, dl) in enumerate(d):
        for ci in range(nfi):
            o.append('    {')
            o.append(f'    const int pi = i0+{ci}, pj = jj+{dj}, '
                     f'pk = kk+{dk}, pl = ll_+{dl};')
            o.append(f'    dd_[t*{w}+{m*nfi+ci}] = '
                     f'(kf != 0. ? kf * ({k8}) : 0.) + '
                     f'(jf != 0. ? jf * ({j2}) : 0.);')
            o.append('    }')
    o.append('}')
    return o


def contract_pair_shm(c, per):
    nfi = c.nf[0]
    w = per * nfi
    o = []
    for m in range(per):
        for dd in range(3):
            o.append(f'const double *gm{m}_{dd} = gx + gb[{3*m+dd}];')
        for ci in range(nfi):
            ip = c.cart[0][ci]
            g = lambda d, off, ip=ip, m=m: f'gm{m}_{d}[{ip[d] + off}*NSQ]'
            jp = [f'jc[{3*m+d}]' for d in range(3)]
            kp = [f'kc[{3*m+d}]' for d in range(3)]
            for d in range(3):
                o.append(f'I{AX[d]} = {g(d, 0)};')
            for d in range(3):
                a, p_ = AX[d], ip[d]
                o.append(f'gi{a} = {g(d, 1)};')
                o.append(f'f{a}i = ai2 * gi{a}' +
                         (f' - {p_} * {g(d, -1)}' if p_ else '') + ';')
                o.append(f'f{a}j = aj2 * (gi{a} - rjri[{d}] * I{a})'
                         f' - ({jp[d]}) * {g(d, -c.sj)};')
                o.append(f'f{a}k = ak2 * {g(d, c.sk)}'
                         f' - ({kp[d]}) * {g(d, -c.sk)};')
            o.append('Iyz = Iy * Iz; Ixz = Ix * Iz; Ixy = Ix * Iy;')
            o.append('#pragma unroll')
            o.append('for (int t = 0; t < EJK_NT; ++t) {')
            o.append(f'    dd = dd_[t*{w}+{m*nfi+ci}];')
            o.append('    pyz = Iyz * dd; pxz = Ixz * dd; pxy = Ixy * dd;')
            for a, q in zip(AX, ('pyz', 'pxz', 'pxy')):
                o.append(f'    v_i{a}[t] += f{a}i * {q};')
                o.append(f'    v_j{a}[t] += f{a}j * {q};')
                o.append(f'    v_k{a}[t] += f{a}k * {q};')
                o.append(f'    v_l{a}[t] -= (f{a}i + f{a}j + f{a}k) * {q};')
            o.append('}')
    return o


def _pairify_shm(src, c, w):
    rep = [
        (f'ejk_{c.tag}(EARGS)', f'ejkp_{c.tag}(EARGS_PAIR)'),
        ('    double v_ix = 0, v_iy = 0, v_iz = 0, v_jx = 0, v_jy = 0, v_jz = 0;',
         '    double v_ix[EJK_NT], v_iy[EJK_NT], v_iz[EJK_NT], v_jx[EJK_NT], '
         'v_jy[EJK_NT], v_jz[EJK_NT];\n#pragma unroll\n'
         '    for (int t = 0; t < EJK_NT; ++t) v_ix[t] = v_iy[t] = v_iz[t] = '
         'v_jx[t] = v_jy[t] = v_jz[t] = 0.;'),
        (f'        double dd_[{w}];', f'        double dd_[EJK_NT*{w}];'),
        ('        double v_kx = 0, v_ky = 0, v_kz = 0, v_lx = 0, v_ly = 0, v_lz = 0;',
         '        double v_kx[EJK_NT], v_ky[EJK_NT], v_kz[EJK_NT], v_lx[EJK_NT], '
         'v_ly[EJK_NT], v_lz[EJK_NT];\n#pragma unroll\n'
         '        for (int t = 0; t < EJK_NT; ++t) v_kx[t] = v_ky[t] = v_kz[t] = '
         'v_lx[t] = v_ly[t] = v_lz[t] = 0.;'),
        ('        double dd, Ix, Iy, Iz, pxy, pxz, pyz;',
         '        double dd, Ix, Iy, Iz, pxy, pxz, pyz, Ixy, Ixz, Iyz;'),
    ]
    for a, b in rep:
        assert a in src, (c.tag, a[:60])
        src = src.replace(a, b)
    # the two reductions, once per output
    i0 = src.index('        __syncthreads();\n        reduce[0*threads] = v_kx;')
    i1 = src.index('    }\n    __syncthreads();\n    reduce[0*threads] = v_ix;')
    kl = src[i0:i1]
    kl_new = ('#pragma unroll 1\n        for (int t = 0; t < EJK_NT; ++t) {\n' +
              kl.replace('v_kx;', 'v_kx[t];').replace('v_ky;', 'v_ky[t];')
                .replace('v_kz;', 'v_kz[t];').replace('v_lx;', 'v_lx[t];')
                .replace('v_ly;', 'v_ly[t];').replace('v_lz;', 'v_lz[t];')
                .replace('atomicAdd(ejk+', 'atomicAdd(ejk+t*out_stride+') +
              '        }\n')
    src = src[:i0] + kl_new + src[i1:]
    j0 = src.index('    __syncthreads();\n    reduce[0*threads] = v_ix;')
    j1 = src.index('    __syncthreads();\n    if (thread_id == 0) {\n        pair_ij = atomicAdd(head, 1);', j0)
    br = src[j0:j1]
    br_new = ('#pragma unroll 1\n    for (int t = 0; t < EJK_NT; ++t) {\n' +
              br.replace('v_ix;', 'v_ix[t];').replace('v_iy;', 'v_iy[t];')
                .replace('v_iz;', 'v_iz[t];').replace('v_jx;', 'v_jx[t];')
                .replace('v_jy;', 'v_jy[t];').replace('v_jz;', 'v_jz[t];')
                .replace('atomicAdd(ejk+', 'atomicAdd(ejk+t*out_stride+') +
              '    }\n')
    src = src[:j0] + br_new + src[j1:]
    return src


PROLOGUE = r'''
// EJK_NT outputs per launch (compile time); see gen_ejkpair.py
#ifndef EJK_NT
#define EJK_NT 4
#endif
#define EARGS_PAIR                                                             \
    double *ejk, const double *dm, const double *dm_tot, int nao,              \
    double j_factor, double k_factor,                                          \
    const int *bas, const double *env, const int *ao_loc, int nbas,            \
    int npairs_ij, int npairs_kl,                                              \
    const int *pair_ij_mapping, const int *pair_kl_mapping,                    \
    const float *q_cond, const float *dm_cond, float cutoff,                   \
    int *pool, int *head, int queue_depth, const double *tab,                  \
    int iprim_a, int jprim_a, int kprim_a, int lprim_a,                        \
    double inv_om2, double coef0, const double *jfs, const double *kfs,        \
    int out_stride
'''


def generate(classes, reg_max=64, reg_threads=128, minblocks=1):
    out = [lex_table(), PROLOGUE]
    done = []
    for spec in classes:
        c = Class(*(int(x) for x in spec))
        if c.ngout <= reg_max and 3 * c.gsz <= 128:
            body = HEAD_REG % dict(
                li=c.l[0], lj=c.l[1], lk=c.l[2], ll=c.l[3], tag=c.tag,
                ngout=c.ngout, gsz=c.gsz, nroots=c.nroots, mpp=36,
                nthreads=reg_threads, minblocks=minblocks,
                dd='\n'.join(' ' * 8 + s for s in dd_pair(c)),
                recur='\n'.join(' ' * 20 + s for s in recur_reg(c)),
                contract='\n'.join(' ' * 20 + s for s in contract_pair(c)))
            out.append(_pairify(body, c))
            done.append(c.tag)
            continue
        sch = shm_scheme(c)
        if sch is None:
            continue
        nsq, gs, per, w, shm, headsz = sch
        d = deltas(c, per)
        body = HEAD_SHM % dict(
            li=c.l[0], lj=c.l[1], lk=c.l[2], ll=c.l[3], tag=c.tag,
            ngout=c.ngout, gs=gs, nsq=nsq, w=w, nroots=c.nroots,
            gsz=c.gsz, pad=c.sk, mpp=36, nthreads=nsq * gs,
            minblocks=minblocks, red1=_nextpow2(gs) // 2,
            red2=_nextpow2(nsq * gs) // 2, lanepre=lane_pre(c, per, d),
            dd='\n'.join(' ' * 12 + s for s in dd_pair_shm(c, per, d)),
            recur=recur_shm(c, 20, False),
            contract='\n'.join(' ' * 20 + s for s in contract_pair_shm(c, per)))
        out.append(_pairify_shm(body, c, w))
        done.append(c.tag + '(shm)')
    sys.stderr.write(f'ejkp: {len(done)} classes: {",".join(done)}\n')
    return '\n'.join(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('classes', nargs='?', default=','.join(DEFAULT_CLASSES))
    ap.add_argument('--reg-max', type=int, default=64)
    ap.add_argument('--reg-threads', type=int, default=128)
    a = ap.parse_args()
    print(generate(a.classes.split(','), a.reg_max, a.reg_threads))
