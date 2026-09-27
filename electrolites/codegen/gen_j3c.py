"""Write the three-centre McMurchie--Davidson Coulomb kernels, one per class.

`docs/DF_WORK_LIST.md` **W19**.  After W9 the Coulomb build of a pure
density-fitted functional is two integral kernels out of GPU4PySCF's
``gvhf-md/contract_int3c2e.cu``:

    contract_int3c2e_dm      rho_P = sum_ij (ij|P) D_ij
    contract_int3c2e_auxvec  J_ij  = sum_P  (ij|P) c_P

Both are one ``__global__`` for every class, templated on the auxiliary
angular momentum alone, so the pair's ``lij`` -- and with it ``nf3ij``, the
Hermite order and every index into the ``Rt`` tensor -- is a run-time value.
This generator takes nothing from that file.  It emits, for each
``(lij, lk)``, a kernel in which one thread carries one shell pair through a
batch of auxiliary shells with everything in registers:

1. **the pair data is read, not recomputed.**  Upstream evaluates the product
   centre ``(ai ri + aj rj) / aij`` -- three divisions -- once per (pair,
   auxiliary shell), then ``aij ak / (aij + ak)`` and
   ``ck / (aij ak sqrt(aij + ak))``, two more and a square root.  Here the
   centre, ``aij`` and ``1/aij`` are a ``(npair, 5)`` table built once per
   ``Int3c2eOpt``, ``1/ak`` is one per auxiliary shell, and one
   ``rsqrt(aij + ak)`` gives both remaining reciprocals: no division is left
   in either kernel.
2. **every Hermite index is a literal.**  Upstream's contraction is
   ``Rt[p1_ij[k*nf3ij + i] * nsp] * dm_ij`` with ``p1_ij`` a ``uint16`` table
   in global memory; here the recurrence, the contraction and the
   Hermite-to-Cartesian step are straight-line code, generated from the
   class alone (the recurrence is `gen_j_high.rt_recurrence`, the order
   GPU4PySCF's own unrolled kernels use).
3. **omega == 0 is compiled in**: `fastj_prologue.cu`'s register Boys
   function, which has no range-separation branch.
4. **the auxiliary Hermite components that are zero are not formed.**  A
   Cartesian Gaussian of angular momentum ``L`` has Hermite components only
   at ``e+f+g = L, L-2, ...``; upstream's density contraction accumulates
   all ``nf3(L)`` and its Cartesian step then reads the ones of the right
   parity, so for ``L = 1`` a quarter and for ``L = 2`` three tenths of that
   contraction is multiplied by zero.  The phase ``(-1)^(e+f+g)`` is then
   one sign per class.
5. **no shared memory and no barrier in the quartet loop.**  Upstream's
   thread blocks cooperate on the ``Rt`` tensor of a pair through shared
   memory with a ``__syncthreads`` per recurrence level, which is right for
   the wide classes and is pure overhead where one thread holds the whole
   tensor.  A class past `MAX_ORDER` is not generated, and the shell-pair
   groups that meet one stay GPU4PySCF's.

Run:
    python -m electrolites.codegen.gen_j3c > j3c.cu      # every class
"""
import argparse
import sys

from electrolites.codegen.gen_j_high import tuv_list, nf3, addr, rt_recurrence

#: highest ``lij + lk`` a shell-pair group may meet and still be served by
#: these kernels.  Past about 6 a quartet's ``Rt`` (84 doubles at 6, 286 at
#: 10) no longer fits one thread's registers and spills, and it is still
#: faster than upstream's shared-memory cooperation up to at least 10
#: (`benchmarks/df_j3c_class_probe.py --ours`, `docs/README_fastdfj3ck.md`);
#: the ceiling is what bounds the compile time of the heaviest classes.
MAX_ORDER = 10
LIJ_MAX = 8
LK_MAX = 6


def parity(l):
    """Hermite components of an angular-momentum-``l`` Cartesian Gaussian
    that are not identically zero: ``e+f+g`` of the parity of ``l``."""
    return [n for n, (e, f, g) in enumerate(tuv_list(l)) if (e + f + g - l) % 2 == 0]


def e_coeffs(l):
    """``x^l = sum_t E[l][t] (1/(2a))^p[l][t] Lambda_t`` for a Gaussian at its
    own centre, as ``{t: (coefficient, power)}``.

    ``x Lambda_t = (1/2a) Lambda_{t+1} + t Lambda_{t-1}``, so
    ``E^i_t = (1/2a) E^{i-1}_{t-1} + (t+1) E^{i-1}_{t+1}``, and every
    ``E^i_t`` is a single power of ``1/(2a)``: ``(i+t)/2``.
    """
    E = [{0: 1}]
    for i in range(1, l + 1):
        cur = {}
        for t, c in E[-1].items():
            cur[t + 1] = cur.get(t + 1, 0) + c           # the (1/2a) term
            if t > 0:
                cur[t - 1] = cur.get(t - 1, 0) + t * c
        E.append(cur)
    return {t: (c, (l + t) // 2) for t, c in E[l].items()}


def cart_list(l):
    """Cartesian components in PySCF's order."""
    return [(ix, iy, l - ix - iy) for ix in reversed(range(l + 1))
            for iy in reversed(range(l + 1 - ix))]


def cart_terms(l):
    """For each Cartesian component, ``[(hermite index, coefficient, power)]``
    with power the exponent of ``1/(2a)``: the Hermite-to-Cartesian step."""
    ex = [e_coeffs(i) for i in range(l + 1)]
    out = []
    for ix, iy, iz in cart_list(l):
        terms = []
        for t, (cx, px) in ex[ix].items():
            for u, (cy, py) in ex[iy].items():
                for v, (cz, pz) in ex[iz].items():
                    terms.append((addr(l, t, u, v), cx * cy * cz,
                                  px + py + pz))
        out.append(terms)
    return out


def emit_boys_rt(L, ind):
    """``g[0..L]`` scaled Boys values, then R^(0)_{tuv} for t+u+v <= L as
    scalars ``R_<n>``, n in `tuv_list(L)` order."""
    o = []
    o.append(f'{ind}double g[{L + 1}];')
    o.append(f'{ind}boys0_fn_reg<{L}>(g, theta, rr, fac);')
    if L == 0:
        o.append(f'{ind}const double R_0 = g[0];')
        return o
    # level n holds R^(L-n) for t+u+v <= n
    prev = ['g[%d]' % L]
    for n in range(1, L + 1):
        cur = [None] * nf3(n)
        cur[0] = f'g[{L - n}]'
        names = []
        for i, axis, j1, fac, j2 in rt_recurrence(n):
            name = f'r{n}_{i}' if n < L else f'R_{i}'
            expr = f'{axis} * {prev[j1]}'
            if fac:
                expr += (f' + {prev[j2]}' if fac == 1
                         else f' + {fac}. * {prev[j2]}')
            o.append(f'{ind}const double {name} = {expr};')
            cur[i] = name
            names.append(name)
        prev = cur
    o.append(f'{ind}const double R_0 = g[0];')
    return o


def kname(kind, lij, lk, scr=False):
    return f'j3c_{kind}_{lij}_{lk}' + ('_scr' if scr else '')


# --------------------------------------------------------------------------
# rho_P = sum_ij (ij|P) D_ij
#
# One block: the pairs [p0, p1) of one class against the kb auxiliary shells
# [k0, min(k0+kb, k1)).  Every thread walks pairs with the pair's Hermite
# density in registers; the block then reduces its kb*nk sums and converts
# them to Cartesian auxiliary components.
# --------------------------------------------------------------------------
def dm_smem(lk, nt, kb):
    return (nt // 32) * kb * len(parity(lk))


def emit_dm(lij, lk, nt, kb, scr=False):
    L = lij + lk
    nij = nf3(lij)
    par = parity(lk)
    nk = len(par)
    tuv_ij = tuv_list(lij)
    tuv_k = tuv_list(lk)
    sign = -1. if lk % 2 else 1.
    nw = nt // 32
    o = []
    o.append('__device__ __forceinline__ static')
    o.append(f'void {kname("dm", lij, lk, scr)}(J3C_DM_ARGS, const int p0, '
             'const int p1, const int k0, const int k_hi, double* s_red)')
    o.append('{')
    o.append(f'    // ({lij}|{lk}): {nij} pair Hermite components, {nk} '
             f'auxiliary ones of the right parity, order {L}; s_red is '
             f'[{nw}][{kb * nk}]')
    o.append(f'    double ak[{kb}], iak[{kb}], ck[{kb}], kx[{kb}], ky[{kb}], '
             f'kz[{kb}];')
    o.append('#pragma unroll')
    o.append(f'    for (int b = 0; b < {kb}; b++) {{')
    o.append('        const int k = min(k0 + b, k_hi - 1);')
    o.append('        const double4 a = aux_data[k];')
    o.append('        kx[b] = a.x; ky[b] = a.y; kz[b] = a.z; ak[b] = a.w;')
    o.append('        iak[b] = aux_inv[2*k]; '
             'ck[b] = k0 + b < k_hi ? aux_inv[2*k+1] : 0.;')
    o.append('    }')
    for b in range(kb):
        o.append(f'    double acc{b}[{nk}] = {{' + ', '.join(['0.'] * nk) + '};')
    if scr:
        # W26: the density screen.  A (warp, batch) is skipped when no pair
        # of the warp has q_ij max|dD_ij| * max_batch q_P at the threshold;
        # the loop is warp-uniform so that the vote sees every lane.
        o.append('    double qb = 0.;')
        o.append('#pragma unroll')
        o.append(f'    for (int b = 0; b < {kb}; b++)')
        o.append('        if (k0 + b < k_hi) qb = fmax(qb, scr_aux[k0 + b]);')
        o.append('    const double wmin = qb > 0. ? scr_tol / qb : 1e300;')
        o.append('    const int lane = threadIdx.x & 31;')
        o.append(f'    for (int base = p0 + (threadIdx.x & ~31); base < p1; '
                 f'base += {nt}) {{')
        o.append('        const int p = base + lane;')
        o.append('        if (!__any_sync(0xffffffff, '
                 'p < p1 && scr_pair[p] >= wmin)) continue;')
        o.append('        if (p >= p1) continue;')
    else:
        o.append(f'    for (int p = p0 + threadIdx.x; p < p1; p += {nt}) {{')
    o.append('        const double4 P = pair_data[p];')
    o.append('        const double iaij = pair_inv[p];')
    o.append('        const double* d = dm_xyz + pair_loc[p];')
    o.append(f'        double dij[{nij}];')
    o.append('#pragma unroll')
    o.append(f'        for (int i = 0; i < {nij}; i++) dij[i] = __ldg(d + i);')
    for b in range(kb):
        o.append('        {')
        ind = '            '
        o.append(f'{ind}const double xpq = P.x - kx[{b}];')
        o.append(f'{ind}const double ypq = P.y - ky[{b}];')
        o.append(f'{ind}const double zpq = P.z - kz[{b}];')
        o.append(f'{ind}const double rr = xpq*xpq + ypq*ypq + zpq*zpq;')
        o.append(f'{ind}const double rs = rsqrt(P.w + ak[{b}]);')
        o.append(f'{ind}const double theta = P.w * ak[{b}] * rs * rs;')
        o.append(f'{ind}const double fac = ck[{b}] * iaij * iak[{b}] * rs;')
        o += emit_boys_rt(L, ind)
        for kk, n in enumerate(par):
            e, f, g = tuv_k[n]
            terms = []
            for i, (t, u, v) in enumerate(tuv_ij):
                terms.append(f'R_{addr(L, t + e, u + f, v + g)} * dij[{i}]')
            o.append(f'{ind}acc{b}[{kk}] += ' + ' + '.join(terms) + ';')
        o.append('        }')
    o.append('    }')
    # block reduction: a warp shuffle, then the warps through shared memory
    if not scr:
        o.append('    const int lane = threadIdx.x & 31;')
    o.append('    const int wid = threadIdx.x >> 5;')
    for b in range(kb):
        o.append('#pragma unroll')
        o.append(f'    for (int n = 0; n < {nk}; n++) {{')
        o.append(f'        double v = acc{b}[n];')
        o.append('        for (int s = 16; s > 0; s >>= 1) '
                 'v += __shfl_down_sync(0xffffffff, v, s);')
        o.append(f'        if (lane == 0) s_red[wid * {kb * nk} + {b * nk} + n]'
                 ' = v;')
        o.append('    }')
    o.append('    __syncthreads();')
    o.append(f'    for (int n = threadIdx.x; n < {kb * nk}; n += {nt}) {{')
    o.append('        double v = 0.;')
    o.append(f'        for (int w = 0; w < {nw}; w++) v += s_red[w * {kb * nk}'
             ' + n];')
    o.append('        s_red[n] = v;')
    o.append('    }')
    o.append('    __syncthreads();')
    # Hermite -> Cartesian, (-1)^(e+f+g) folded into one sign, and the atomics
    terms = cart_terms(lk)
    hpos = {n: kk for kk, n in enumerate(par)}
    ncart = len(terms)
    o.append(f'    for (int t = threadIdx.x; t < {kb * ncart}; t += {nt}) {{')
    o.append(f'        const int b = t / {ncart};')
    o.append(f'        const int c = t % {ncart};')
    o.append('        const int k = k0 + b;')
    o.append('        if (k < k_hi) {')
    o.append(f'            const double* h = s_red + b * {nk};')
    maxp = max((p for tl in terms for _, _, p in tl), default=0)
    o.append('            const double a1 = .5 * aux_inv[2*k];')
    o.append(f'            double aa[{maxp + 1}];')
    o.append('            aa[0] = 1.;')
    for p in range(1, maxp + 1):
        o.append(f'            aa[{p}] = aa[{p - 1}] * a1;')
    o.append('            double v = 0.;')
    o.append('            switch (c) {')
    for c, tl in enumerate(terms):
        expr = ' + '.join(f'{float(cf)!r} * aa[{p}] * h[{hpos[n]}]'
                          for n, cf, p in tl)
        o.append(f'            case {c}: v = {expr}; break;')
    o.append('            }')
    o.append(f'            atomicAdd(rho + aux_cart_loc[k] + c, {sign!r} * v);')
    o.append('        }')
    o.append('    }')
    o.append('}')
    return '\n'.join(o) + '\n'


# --------------------------------------------------------------------------
# J_ij = sum_P (ij|P) c_P
#
# One block: nt pairs from p0 (not past p1) against the auxiliary shells
# [k0, k1), which are at most kb and all of one class.  One thread, one pair,
# its Hermite J in registers; one atomicAdd per component at the end.
# --------------------------------------------------------------------------
def aux_smem(lk, kb, scr=False):
    return kb * (5 + scr + len(parity(lk)))


def emit_aux(lij, lk, nt, kb, scr=False):
    L = lij + lk
    nij = nf3(lij)
    par = parity(lk)
    nk = len(par)
    tuv_ij = tuv_list(lij)
    tuv_k = tuv_list(lk)
    sign = -1. if lk % 2 else 1.
    o = []
    o.append('__device__ __forceinline__ static')
    o.append(f'void {kname("aux", lij, lk, scr)}(J3C_AUX_ARGS, const int p0, '
             'const int p1, const int k0, const int k1, double* smem)')
    o.append('{')
    o.append(f'    // ({lij}|{lk}): {nij} pair Hermite components, {nk} '
             f'auxiliary ones of the right parity, order {L}')
    o.append(f'    double* s_k = smem;                 // [{kb}][5]')
    o.append(f'    double* s_c = smem + {5 * kb};      // [{kb}][{nk}]')
    if scr:
        o.append(f'    double* s_e = smem + {(5 + nk) * kb};      // [{kb}]')
    o.append('    const int nkb = k1 - k0;')
    o.append(f'    for (int n = threadIdx.x; n < nkb; n += {nt}) {{')
    o.append('        const double4 a = aux_data[k0 + n];')
    o.append('        s_k[5*n] = a.x; s_k[5*n+1] = a.y; s_k[5*n+2] = a.z;')
    o.append('        s_k[5*n+3] = a.w; s_k[5*n+4] = aux_inv[2*(k0 + n)];')
    if scr:
        o.append('        s_e[n] = scr_aux[k0 + n];')
    o.append('    }')
    o.append(f'    for (int n = threadIdx.x; n < nkb * {nk}; n += {nt}) {{')
    o.append(f'        const int b = n / {nk};')
    o.append(f'        const int kk = n % {nk};')
    o.append('        const int par[] = {' + ', '.join(map(str, par)) + '};')
    o.append(f'        s_c[n] = {sign!r} * aux_xyz[aux_xyz_loc[k0 + b] + par[kk]];')
    o.append('    }')
    o.append('    __syncthreads();')
    o.append('    const int p = p0 + threadIdx.x;')
    if scr:
        # W26: the difference screen.  A warp leaves when none of its pairs
        # has q_ij * sum_batch q_P max|dc_P| at the threshold, and skips an
        # auxiliary shell when none has q_ij q_P max|dc_P| there.  Both votes
        # need every lane, so a lane past the group's end stays, computes
        # with the last pair and writes nothing.
        o.append('    const double qv = p < p1 ? scr_pair[p] : 0.;')
        o.append('    double S = 0.;')
        o.append('    for (int n = 0; n < nkb; n++) S += s_e[n];')
        o.append('    if (!__any_sync(0xffffffff, qv * S >= scr_tol)) return;')
        o.append('    const int pp = min(p, p1 - 1);')
    else:
        o.append('    if (p >= p1) return;')
    pp = 'pp' if scr else 'p'
    o.append(f'    const double4 P = pair_data[{pp}];')
    o.append(f'    const double iaij = pair_inv[{pp}] * PI_FAC;')
    o.append(f'    double acc[{nij}] = {{' + ', '.join(['0.'] * nij) + '};')
    o.append('    for (int b = 0; b < nkb; b++) {')
    ind = '        '
    if scr:
        o.append(f'{ind}if (!__any_sync(0xffffffff, qv * s_e[b] >= '
                 'scr_tol)) continue;')
    o.append(f'{ind}const double* K = s_k + 5*b;')
    o.append(f'{ind}const double xpq = P.x - K[0];')
    o.append(f'{ind}const double ypq = P.y - K[1];')
    o.append(f'{ind}const double zpq = P.z - K[2];')
    o.append(f'{ind}const double rr = xpq*xpq + ypq*ypq + zpq*zpq;')
    o.append(f'{ind}const double rs = rsqrt(P.w + K[3]);')
    o.append(f'{ind}const double theta = P.w * K[3] * rs * rs;')
    o.append(f'{ind}const double fac = iaij * K[4] * rs;')
    o += emit_boys_rt(L, ind)
    o.append(f'{ind}const double* c = s_c + {nk}*b;')
    for i, (t, u, v) in enumerate(tuv_ij):
        terms = []
        for kk, n in enumerate(par):
            e, f, g = tuv_k[n]
            terms.append(f'R_{addr(L, t + e, u + f, v + g)} * c[{kk}]')
        o.append(f'{ind}acc[{i}] += ' + ' + '.join(terms) + ';')
    o.append('    }')
    if scr:
        o.append('    if (p >= p1) return;')
    o.append('    double* out = vj_xyz + pair_loc[p];')
    o.append('#pragma unroll')
    o.append(f'    for (int i = 0; i < {nij}; i++) atomicAdd(out + i, acc[i]);')
    o.append('}')
    return '\n'.join(o) + '\n'


PROLOGUE = r"""
// ---- three-centre McMurchie--Davidson Coulomb kernels (gen_j3c.py) ----
// pair_data[p] = (Px, Py, Pz, aij) and pair_inv[p] = 1/aij for every stored
// shell pair; aux_data[k] = (Kx, Ky, Kz, ak) and aux_inv[2k:2k+2] =
// (1/ak, ck * PI_FAC) for every sorted auxiliary shell (k counted from 0).
//
// Classes are grouped into tiers of similar register need and each tier is
// one kernel: a block reads its (class, pair range, auxiliary range) from
// `blocks` and switches on the class, which is uniform over the block.  One
// launch per tier instead of one per class, because a class's grid can be a
// handful of blocks and W9 measured thirty serialised launches at 0.42x.
#define J3C_DM_ARGS \
    const double4* __restrict__ pair_data, const double* __restrict__ pair_inv, \
    const int* __restrict__ pair_loc, const double* __restrict__ dm_xyz, \
    const double4* __restrict__ aux_data, const double* __restrict__ aux_inv, \
    const int* __restrict__ aux_cart_loc, double* __restrict__ rho%(scr)s
#define J3C_DM_FWD pair_data, pair_inv, pair_loc, dm_xyz, aux_data, aux_inv, \
    aux_cart_loc, rho%(fwd)s
#define J3C_AUX_ARGS \
    const double4* __restrict__ pair_data, const double* __restrict__ pair_inv, \
    const int* __restrict__ pair_loc, double* __restrict__ vj_xyz, \
    const double4* __restrict__ aux_data, const double* __restrict__ aux_inv, \
    const int* __restrict__ aux_xyz_loc, const double* __restrict__ aux_xyz%(scr)s
#define J3C_AUX_FWD pair_data, pair_inv, pair_loc, vj_xyz, aux_data, aux_inv, \
    aux_xyz_loc, aux_xyz%(fwd)s
"""

#: W26's module: the same kernels with a screen compiled in, for a build from
#: a difference.  The density kernel skips a (warp, auxiliary batch) whose
#: pairs all have scr_pair[p] = q_ij max|dD_ij| below scr_tol / max_batch
#: scr_aux[k] (scr_aux = q_P); the auxiliary-vector kernel a (warp, auxiliary
#: shell) whose pairs all have scr_pair[p] = q_ij below scr_tol / scr_aux[k]
#: (scr_aux = q_P max|dc_P|).  A separate module, not a run-time flag: a flag
#: in the shipped kernels cost them 0.8-1.0 % on every build
#: (`docs/README_fastdfj3cscr.md`).
_SCR_ARGS = (', \\\n    const double* __restrict__ scr_pair, '
             'const double* __restrict__ scr_aux, \\\n    const double scr_tol')
_SCR_FWD = ', scr_pair, scr_aux, scr_tol'


def prologue(scr=False):
    return PROLOGUE % ({'scr': _SCR_ARGS, 'fwd': _SCR_FWD} if scr
                       else {'scr': '', 'fwd': ''})


#: order ranges of the tiers, and each tier's (threads, min blocks per SM)
TIERS = ((0, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 14))
TIER_CFG = {
    'dm': {0: (128, 4), 1: (128, 4), 2: (128, 2), 3: (128, 2), 4: (128, 1),
           5: (128, 1)},
    'aux': {0: (128, 4), 1: (128, 4), 2: (128, 2), 3: (128, 2), 4: (128, 1),
            5: (128, 1)},
}


def tier_of(L):
    for t, (lo, hi) in enumerate(TIERS):
        if lo <= L <= hi:
            return t
    raise ValueError(L)


def default_kb(kind, lij, lk):
    """Auxiliary shells per block.  A starting point from the register
    budget; `benchmarks/df_j3c_kernel_sweep.py` is what tunes it."""
    L = lij + lk
    if kind == 'dm':
        nk = len(parity(lk))
        return 4 if nk * 4 <= 12 and L <= 3 else (2 if nk * 2 <= 14 else 1)
    return 16


def default_cfg(kind, lij, lk):
    return default_kb(kind, lij, lk)


def classes(max_order=MAX_ORDER):
    return [(lij, lk) for lij in range(LIJ_MAX + 1)
            for lk in range(LK_MAX + 1) if lij + lk <= max_order]


def tier_name(kind, t, scr=False):
    return f'j3c_{kind}_tier{t}' + ('_scr' if scr else '')


def source(wanted=None, kb=None, tier_cfg=None, scr=False):
    """CUDA source for the classes in ``wanted`` (default: all of them).

    ``kb`` maps ``(kind, lij, lk)`` to the auxiliary shells per block and
    ``tier_cfg`` ``(kind, tier)`` to ``(threads, min blocks per SM)``; missing
    entries take the defaults.  The class id a block carries is the class's
    position in ``wanted``.
    """
    wanted = list(wanted or classes())
    kb = kb or {}
    tier_cfg = tier_cfg or {}
    out = [prologue(scr)]
    for kind, emit in (('dm', emit_dm), ('aux', emit_aux)):
        by_tier = {}
        for cid, (lij, lk) in enumerate(wanted):
            by_tier.setdefault(tier_of(lij + lk), []).append((cid, lij, lk))
        for t, members in sorted(by_tier.items()):
            nt, mb = tier_cfg.get((kind, t), TIER_CFG[kind][t])
            smem = 0
            for cid, lij, lk in members:
                b = kb.get((kind, lij, lk), default_kb(kind, lij, lk))
                out.append(emit(lij, lk, nt, b, scr))
                smem = max(smem, dm_smem(lk, nt, b) if kind == 'dm'
                           else aux_smem(lk, b, scr))
            args = 'J3C_DM_ARGS' if kind == 'dm' else 'J3C_AUX_ARGS'
            fwd = 'J3C_DM_FWD' if kind == 'dm' else 'J3C_AUX_FWD'
            o = [f'extern "C" __global__ __launch_bounds__({nt}, {mb})',
                 f'void {tier_name(kind, t, scr)}({args}, '
                 'const int* __restrict__ blocks)',
                 '{',
                 f'    __shared__ double smem[{max(smem, 1)}];',
                 '    const int* b = blocks + 5 * blockIdx.x;',
                 '    switch (b[0]) {']
            for cid, lij, lk in members:
                o.append(f'    case {cid}: {kname(kind, lij, lk, scr)}({fwd}, '
                         'b[1], b[2], b[3], b[4], smem); break;')
            o += ['    }', '}', '']
            out.append('\n'.join(o))
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--max-order', type=int, default=MAX_ORDER)
    a = ap.parse_args()
    sys.stdout.write(source(classes(a.max_order)))


if __name__ == '__main__':
    main()
