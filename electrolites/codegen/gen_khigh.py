"""Generate unrolled Rys exchange kernels for the classes GPU4PySCF does *not*
unroll.

``gen_kernels.py`` and ``gen_k2_kernels.py`` lift the integral arithmetic out
of GPU4PySCF's generated ``gvhf-rys/unrolled_rys_k.cu``.  That file only covers
25 angular-momentum classes -- everything an s/p/d basis can produce -- so with
f functions (def2-TZVP, cc-pVTZ) most of the exchange time lands on GPU4PySCF's
*general* ``rys_k_kernel``, which there is nothing left to lift from.  This
module writes that arithmetic itself: the Rys 2D recurrences (VRR, TRR, the two
horizontal transfers) and the density contraction, unrolled for one
(li,lj,lk,ll) with every g-array address a compile-time constant.

Why the general kernel is slow, measured on an A100 with ncu for (ff|ff):

  * Its ``gout`` tile is 3x3x3x3 whatever the class, so a shell with nf = 10
    (f) is padded to 12 and one with nf = 1 (s) to 3.  For (fs|ff) that is
    12*3*12*12 / (10*1*10*10) = 5.2x more contraction work than the class
    needs.  (gout_pattern is meant to drop the padded dimensions, but in
    GPU4PySCF 1.7.0 it is computed with >> where << was intended, so only
    ll == 0 is ever specialised.)  An unrolled kernel tiles exactly.
  * 81 ``gout`` accumulators are 162 registers, which pins it at 255 registers
    and one 256-thread block per SM, and the shared-memory budget then forces
    ``nsq_per_block = 1`` for the wide classes: a warp's 32 lanes each read a
    *different* g-array address, so 86% of its shared wavefronts are replays
    and it runs at 86% L1 throughput with 5% of the compute pipes busy.
    Tiling exactly needs far fewer accumulators, which buys back
    ``nsq_per_block``, and at nsq >= 16 a warp's shared reads are contiguous.
  * Its addresses come from shared ``idx_i/j/k/l`` arrays, so every one of the
    three g loads per ``gout`` element carries integer address arithmetic and
    no two loads can be CSE'd.  Unrolled, the addresses are constants and nvcc
    reuses the loads.

The emitted kernels have the same signature and shared-memory layout as the
ones ``gen_k2_kernels.py`` produces, so ``fastk`` dispatches them through the
same table and code path.

Two emission modes.  ``--mode full`` writes one straight-line body per
``threadIdx.y`` lane behind a ``switch (gout_id)``, which is GPU4PySCF's own
arrangement and is correct, but it only pays off while a warp is one lane wide
-- 32 shell quartets per block.  The wide classes cannot afford that many, and
the switch then diverges up to 32 ways inside every warp; measured over 29
classes it wins 1.6-5.6x on the narrow ones, loses up to 3.7x on the wide ones,
and comes out level overall in 444k lines of CUDA.  ``--mode compact`` (the
default) emits one body that every lane runs: a lane owns all nfi cartesian
functions of the bra-i shell for PER of the nfj*nfk*nfl (j,k,l) combinations,
and the (j,k,l) part of each address -- the only part that differs between
lanes -- becomes three base pointers computed once outside every loop.  31k
lines, no divergence, 1.64x over the general kernel on those same 29 classes.

Run:  python gen_khigh.py <classes> > fastkhigh_generated.cu
      ./build_khigh.sh                       # the class list fastk ships with
"""
import sys, os, json, argparse

MAX_PRIM_PAIR = 36              # matches fastk_prologue.cu
MAX_BLOCK = 1024                # CUDA threads per block

# _c_cartesian_lexical_xyz from gvhf-rys/rys_constant.cu: the (x,y,z) powers of
# the cartesian functions of a shell, in the order PySCF stores them.
def cart(l):
    out = []
    for ix in range(l, -1, -1):
        for iy in range(l - ix, -1, -1):
            out.append((ix, iy, l - ix - iy))
    return out


assert cart(3)[:3] == [(3, 0, 0), (2, 1, 0), (2, 0, 1)]
assert cart(2) == [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]


class Class:
    """Geometry of one (li,lj,lk,ll): strides into the 2D-integral array."""

    def __init__(self, li, lj, lk, ll):
        self.l = (li, lj, lk, ll)
        self.tag = f'{li}{lj}{lk}{ll}'
        self.lij, self.lkl = li + lj, lk + ll
        self.sj = li + 1
        self.sk = self.sj * (lj + 1)
        self.sl = self.sk * (lk + 1)
        self.gsz = self.sl * (ll + 1)
        self.nroots = (li + lj + lk + ll) // 2 + 1
        self.nf = [(x + 1) * (x + 2) // 2 for x in self.l]
        self.ngout = self.nf[0] * self.nf[1] * self.nf[2] * self.nf[3]
        self.cart = [cart(x) for x in self.l]

    # -- the address of one cartesian quadruple in the x/y/z 2D-integral array
    def addr(self, ci, cj, ck, cl, d):
        pi, pj = self.cart[0][ci][d], self.cart[1][cj][d]
        pk, pl = self.cart[2][ck][d], self.cart[3][cl][d]
        return (pi + pj * self.sj + pk * self.sk + pl * self.sl) + d * self.gsz

    def quads(self):
        """Cartesian quadruples in gout order: i fastest, then j, k, l."""
        nfi, nfj, nfk, _ = self.nf
        for cl in range(self.nf[3]):
            for ck in range(nfk):
                for cj in range(nfj):
                    for ci in range(nfi):
                        yield ci + nfi * cj + nfi * nfj * ck + \
                              nfi * nfj * nfk * cl, (ci, cj, ck, cl)


def recurrence_fused(c, ind):
    """Every recurrence for one cartesian direction on one lane, GPU4PySCF's
    arrangement: no __syncthreads between the phases, but never more than three
    lanes at work."""
    p = ' ' * ind
    o = [p + 'for (int n = gout_id; n < 3; n += GSTRIDE) {',
         p + '    if (n == 2) { gx[2*g_size*NSQ] = rw[(2*irys+1)*NSQ]; }',
         p + '    double *_gx = gx + n * g_size * NSQ;',
         p + '    double xjxi = rjri[n], xlxk = rlrk[n*NSQ];',
         p + '    double c0x = xjxi * s_ajaij[ijp] - rt_aij * Rpq[n*NSQ];',
         p + '    double cpx = xlxk * al_akl + rt_akl * Rpq[n*NSQ];',
         p + '    (void)c0x; (void)cpx; (void)xjxi; (void)xlxk;']
    o += [p + '    ' + l for l in recurrence(c)]
    o.append(p + '}')
    return '\n'.join(o)


def recurrence(c):
    """The Rys 2D recurrences for one cartesian direction, unrolled.

    Exactly the expressions GPU4PySCF's generated kernels evaluate, in the same
    order: the vertical recurrence in i, the transfer recurrence in k, then the
    two horizontal transfers that move angular momentum i->j and k->l.  All
    reads and writes are `_gx[<constant>*NSQ]`.
    """
    li, lj, lk, ll = c.l
    lij, lkl, sj, sk, sl = c.lij, c.lkl, c.sj, c.sk, c.sl
    o = []
    g = lambda n: f'_gx[{n}*NSQ]'

    # VRR:  g(i+1,0) = c0x*g(i,0) + i*b10*g(i-1,0)
    if lij > 0:
        o += ['s0 = ' + g(0) + ';', 's1 = c0x * s0;', g(1) + ' = s1;']
        for i in range(1, lij):
            o += [f's2 = c0x * s1 + {i} * b10 * s0;', g(i + 1) + ' = s2;',
                  's0 = s1;', 's1 = s2;']

    # TRR:  g(i,k+1) = cpx*g(i,k) + k*b01*g(i,k-1) + i*b00*g(i-1,k)
    if lkl > 0:
        for i in range(lij + 1):
            o += ['s0 = ' + g(i) + ';', 's1 = cpx * s0;']
            if i > 0:
                o += [f's1 += {i} * b00 * ' + g(i - 1) + ';']
            o += [g(i + sk) + ' = s1;']
            for k in range(1, lkl):
                o += [f's2 = cpx * s1 + {k} * b01 * s0;']
                if i > 0:
                    o += [f's2 += {i} * b00 * ' + g(i - 1 + k * sk) + ';']
                o += [g(i + (k + 1) * sk) + ' = s2;', 's0 = s1;', 's1 = s2;']

    # HRR i->j:  g(i,j) = g(i+1,j-1) - (rj-ri)*g(i,j-1), descending in i so the
    # source row can be overwritten in place.
    if lj > 0:
        for k in range(lkl + 1):
            b = k * sk
            for j in range(1, lj + 1):
                o += ['s1 = ' + g(b + (lij - j + 1) + (j - 1) * sj) + ';']
                for i in range(lij - j, -1, -1):
                    o += ['s0 = ' + g(b + i + (j - 1) * sj) + ';',
                          g(b + i + j * sj) + ' = s1 - xjxi * s0;', 's1 = s0;']

    # HRR k->l, over every (i,j) column the previous step left valid
    if ll > 0:
        for m in range(sk):
            for l in range(1, ll + 1):
                o += ['s1 = ' + g(m + (lkl - l + 1) * sk + (l - 1) * sl) + ';']
                for k in range(lkl - l, -1, -1):
                    o += ['s0 = ' + g(m + k * sk + (l - 1) * sl) + ';',
                          g(m + k * sk + l * sl) + ' = s1 - xlxk * s0;',
                          's1 = s0;']
    return o


def vrr_trr(c):
    """Vertical and transfer recurrences for one cartesian direction."""
    lij, lkl, sk = c.lij, c.lkl, c.sk
    o = []
    g = lambda n: f'_gx[{n}*NSQ]'
    if lij > 0:
        o += ['s0 = ' + g(0) + ';', 's1 = c0x * s0;', g(1) + ' = s1;']
        for i in range(1, lij):
            o += [f's2 = c0x * s1 + {i} * b10 * s0;', g(i + 1) + ' = s2;',
                  's0 = s1;', 's1 = s2;']
    if lkl > 0:
        for i in range(lij + 1):
            o += ['s0 = ' + g(i) + ';', 's1 = cpx * s0;']
            if i > 0:
                o += [f's1 += {i} * b00 * ' + g(i - 1) + ';']
            o += [g(i + sk) + ' = s1;']
            for k in range(1, lkl):
                o += [f's2 = cpx * s1 + {k} * b01 * s0;']
                if i > 0:
                    o += [f's2 += {i} * b00 * ' + g(i - 1 + k * sk) + ';']
                o += [g(i + (k + 1) * sk) + ' = s2;', 's0 = s1;', 's1 = s2;']
    return o


def hrr_ij_col(c, base=0):
    """i->j transfer for one (direction, k) column of the 2D array."""
    lij, lj, sj = c.lij, c.l[1], c.sj
    o = []
    g = lambda n: f'_gx[{n}*NSQ]'
    for j in range(1, lj + 1):
        o += ['s1 = ' + g(base + (lij - j + 1) + (j - 1) * sj) + ';']
        for i in range(lij - j, -1, -1):
            o += ['s0 = ' + g(base + i + (j - 1) * sj) + ';',
                  g(base + i + j * sj) + ' = s1 - xjxi * s0;', 's1 = s0;']
    return o


def hrr_kl_col(c, base=0):
    """k->l transfer for one (direction, ij-column) of the 2D array."""
    lkl, ll, sk, sl = c.lkl, c.l[3], c.sk, c.sl
    o = []
    g = lambda n: f'_gx[{n}*NSQ]'
    for l in range(1, ll + 1):
        o += ['s1 = ' + g(base + (lkl - l + 1) * sk + (l - 1) * sl) + ';']
        for k in range(lkl - l, -1, -1):
            o += ['s0 = ' + g(base + k * sk + (l - 1) * sl) + ';',
                  g(base + k * sk + l * sl) + ' = s1 - xlxk * s0;', 's1 = s0;']
    return o


def recurrence_split(c, ind):
    """VRR/TRR on three lanes, then the two horizontal transfers spread over
    every lane.

    The vertical recurrences are three independent chains -- one per cartesian
    direction -- so no more than three lanes can ever work on them, and with
    the wide classes needing 18 or 36 lanes to hold gout that leaves most of a
    block idle.  The horizontal transfers are not chains: i->j is independent
    per (direction, k) column and k->l per (direction, ij) column, tens of
    tasks either way.  Splitting them over the lanes costs one __syncthreads
    each and takes the recurrences off the critical path.
    """
    p = ' ' * ind
    o = [p + 'for (int n = gout_id; n < 3; n += GSTRIDE) {',
         p + '    if (n == 2) { gx[2*g_size*NSQ] = rw[(2*irys+1)*NSQ]; }',
         p + '    double *_gx = gx + n * g_size * NSQ;',
         p + '    double c0x = rjri[n] * s_ajaij[ijp] - rt_aij * Rpq[n*NSQ];',
         p + '    double cpx = rlrk[n*NSQ] * al_akl + rt_akl * Rpq[n*NSQ];',
         p + '    (void)c0x; (void)cpx;']
    o += [p + '    ' + l for l in vrr_trr(c)]
    o.append(p + '}')
    if c.l[1] > 0:
        o += [p + '__syncthreads();',
              p + f'for (int t = gout_id; t < {(c.lkl+1)*3}; t += GSTRIDE) {{',
              p + '    int d_ = t % 3, kc_ = t / 3;',
              p + f'    double *_gx = gx + (d_ * g_size + kc_ * {c.sk}) * NSQ;',
              p + '    double xjxi = rjri[d_];']
        o += [p + '    ' + l for l in hrr_ij_col(c)]
        o.append(p + '}')
    if c.l[3] > 0:
        o += [p + '__syncthreads();',
              p + f'for (int t = gout_id; t < {c.sk*3}; t += GSTRIDE) {{',
              p + '    int d_ = t % 3, m_ = t / 3;',
              p + '    double *_gx = gx + (d_ * g_size + m_) * NSQ;',
              p + '    double xlxk = rlrk[d_*NSQ];']
        o += [p + '    ' + l for l in hrr_kl_col(c)]
        o.append(p + '}')
    return '\n'.join(o)


def lanes(c, gout_stride, layout):
    """Which flat gout indices each threadIdx.y lane owns.

    'blocked' gives a lane a contiguous run, so its gout elements differ mostly
    in the fastest index i and the three g loads per element collapse to a
    handful of distinct addresses that nvcc keeps in registers.  'rr' is
    GPU4PySCF's round-robin, kept for comparison.
    """
    n = c.ngout
    if layout == 'rr':
        return [list(range(t, n, gout_stride)) for t in range(gout_stride)]
    per, rem = divmod(n, gout_stride)
    out, s = [], 0
    for t in range(gout_stride):
        w = per + (1 if t < rem else 0)
        out.append(list(range(s, s + w)))
        s += w
    return out


def gout_body(c, own):
    """gout accumulation for one lane."""
    q = dict(c.quads())
    o = []
    for m, n in enumerate(own):
        ci, cj, ck, cl = q[n]
        a = [c.addr(ci, cj, ck, cl, d) for d in range(3)]
        o.append(f'gout[{m}] += gx[{a[0]}*NSQ] * gx[{a[1]}*NSQ] '
                 f'* gx[{a[2]}*NSQ];')
    return o


# The four exchange contractions of one shell quartet.  ish == jsh is handled
# by the .5 in fac_sym, exactly as GPU4PySCF's unrolled kernels do, so all four
# are emitted unconditionally.
#   vk[i,l] += sum_jk gout * dm[j,k]      vk[j,l] += sum_ik gout * dm[i,k]
#   vk[i,k] += sum_jl gout * dm[j,l]      vk[j,k] += sum_il gout * dm[i,l]
TERMS = [(0, 3, 1, 2, 'i', 'l', 'j', 'k'), (1, 3, 0, 2, 'j', 'l', 'i', 'k'),
         (0, 2, 1, 3, 'i', 'k', 'j', 'l'), (1, 2, 0, 3, 'j', 'k', 'i', 'l')]


def contract_body(c, own):
    q = dict(c.quads())
    o = []
    for oa, ob, da, db, na, nb, ma, mb in TERMS:
        groups = {}
        for m, n in enumerate(own):
            cc = q[n]
            groups.setdefault((cc[oa], cc[ob]), []).append((m, cc[da], cc[db]))
        for (a, b), items in sorted(groups.items()):
            o.append('val = 0;')
            for m, x, y in items:
                o.append(f'val += gout[{m}] * dm[({ma}0+{x})*nao+({mb}0+{y})];')
            o.append(f'atomicAdd(vk+({na}0+{a})*nao+({nb}0+{b}), val);')
    return o



# ---------------------------------------------------------------------------
# Compact emission.
#
# Writing one straight-line body per threadIdx.y lane -- the way GPU4PySCF's
# own unrolled kernels do -- only works while a warp is one lane wide, that is
# while the block holds 32 shell quartets.  The wide classes cannot afford that
# many, so a `switch (gout_id)` would diverge 2- to 32-ways inside every warp;
# measured on (dd|dd) that costs more than unrolling gains.
#
# So every lane runs the *same* code instead.  A lane owns all nfi cartesian
# functions of the bra-i shell for PER of the nfj*nfk*nfl (j,k,l) combinations,
# and the (j,k,l) part of each g-array address -- the only part that differs
# between lanes -- is folded into three base pointers computed once, outside
# every loop.  What is left inside the loops is nfi addresses per (j,k,l) that
# are compile-time constants, which is what lets nvcc reuse the loads: the
# nfi = 10 functions of an f shell read only li+1 = 4 distinct x addresses.
# ---------------------------------------------------------------------------

def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def per_choices(c):
    """PER values whose lane blocks stay aligned to the (j,k,l) odometer.

    A lane owns the PER consecutive (j,k,l) combinations starting at
    gout_id*PER.  For each of j, k and l to be a runtime base plus a
    compile-time offset -- which is what lets one body serve every lane -- the
    block must not straddle a digit of that odometer.
    """
    nfj, nfk, nfl = c.nf[1], c.nf[2], c.nf[3]
    out = list(_divisors(nfj))
    out += [nfj * q for q in _divisors(nfk)]
    out += [nfj * nfk * q for q in _divisors(nfl)]
    out = sorted(set(out))
    # every one of these divides nfj*nfk*nfl, so the lanes tile it exactly and
    # no combination is left uncomputed
    assert all(nfj * nfk * nfl % p == 0 for p in out), c.tag
    return out


def deltas(c, per):
    """(dj, dk, dl) offsets of each of a lane's PER combinations."""
    nfj, nfk = c.nf[1], c.nf[2]
    if per <= nfj:
        d = [(m, 0, 0) for m in range(per)]
    elif per <= nfj * nfk:
        d = [(m % nfj, m // nfj, 0) for m in range(per)]
    else:
        d = [(m % nfj, (m // nfj) % nfk, m // (nfj * nfk)) for m in range(per)]
    # the runtime bases are the decode of gout_id*PER; check base+delta is the
    # true decode for every lane, so the emitted body cannot silently alias
    nO = c.nf[1] * c.nf[2] * c.nf[3]
    for gid in range(nO // per):
        o0 = gid * per
        jb, kb, lb = o0 % nfj, (o0 // nfj) % nfk, o0 // (nfj * nfk)
        for m, (dj, dk, dl) in enumerate(d):
            o = o0 + m
            assert (jb + dj, kb + dk, lb + dl) == \
                   (o % nfj, (o // nfj) % nfk, o // (nfj * nfk)), (c.tag, per)
    return d


def scheme_compact(c, threads, gout_max, shm_max):
    """Largest PER -- hence the most shell quartets per block -- that fits."""
    nO = c.nf[1] * c.nf[2] * c.nf[3]
    best = None
    unit = 10 + 3 * c.gsz + 2 * c.nroots
    for per in per_choices(c):
        w = per * c.nf[0]
        if w > gout_max:
            continue
        gs = nO // per
        if gs > MAX_BLOCK:              # a block cannot hold that many lanes
            continue
        nsq = _snap(threads // gs)
        while nsq and ((nsq * unit + MAX_PRIM_PAIR) * 8 > shm_max
                       or nsq * gs > MAX_BLOCK):
            nsq = _snap(nsq - 1)
        if nsq == 0:
            continue
        shm = nsq * unit + MAX_PRIM_PAIR
        # most shell quartets per block first -- that is what makes a warp's
        # g-array reads contiguous -- then the widest gout tile, which costs
        # the fewest lanes and so the fewest redundant recurrence passes
        if best is None or (nsq, per) > (best[0], best[2]):
            best = (nsq, gs, per, w, shm)
    return best


def _snap(n):
    """Round the shell quartets per block so a warp stays inside one lane.

    threadIdx is (quartet, lane), so with 32 or more quartets a warp is one
    lane and its g-array reads are 32 consecutive doubles -- one wavefront per
    128 bytes, which is the whole point of keeping nsq large.  Below 32 the
    warp straddles lanes, and a power of two at least keeps each lane's reads
    on aligned segments.
    """
    if n >= 32:
        return n // 32 * 32
    p = 1
    while p * 2 <= n:
        p *= 2
    return p if n >= 1 else 0


def _lexoff(l):
    return l * (l + 1) * (l + 2) // 2


def lane_pre(c, per, d):
    """Per-lane base pointers into the 2D-integral array.  They depend only on
    threadIdx.y, so they are computed once, outside every loop."""
    nfj, nfk = c.nf[1], c.nf[2]
    o = ['    int o_base = gout_id * %d;' % per,
         '    int jb_ = o_base %% %d;' % nfj,
         '    int kb_ = (o_base / %d) %% %d;' % (nfj, nfk),
         '    int lb_ = o_base / %d;' % (nfj * nfk),
         '    int gb[%d];' % (3 * per)]
    for m, (dj, dk, dl) in enumerate(d):
        o.append('    { int pj = (jb_+%d)*3, pk = (kb_+%d)*3, pl = (lb_+%d)*3;'
                 % (dj, dk, dl))
        for dd in range(3):
            o.append('      gb[%d] = (c_lex[%d+pj+%d]*%d + c_lex[%d+pk+%d]*%d'
                     ' + c_lex[%d+pl+%d]*%d)*NSQ + %d*g_size*NSQ;'
                     % (3 * m + dd, _lexoff(c.l[1]), dd, c.sj,
                        _lexoff(c.l[2]), dd, c.sk,
                        _lexoff(c.l[3]), dd, c.sl, dd))
        o.append('    }')
    return '\n'.join(o)


def gout_compact(c, per):
    nfi = c.nf[0]
    o = []
    for m in range(per):
        o += ['const double *gxm%d = gx + gb[%d];' % (m, 3 * m),
              'const double *gym%d = gx + gb[%d];' % (m, 3 * m + 1),
              'const double *gzm%d = gx + gb[%d];' % (m, 3 * m + 2)]
    for m in range(per):
        for ci in range(nfi):
            px, py, pz = c.cart[0][ci]
            o.append('gout[%d] += gxm%d[%d*NSQ] * gym%d[%d*NSQ] * gzm%d[%d*NSQ];'
                     % (m * nfi + ci, m, px, m, py, m, pz))
    return o


def contract_compact(c, per, d):
    """The four exchange contractions, grouped so each output element takes one
    atomicAdd.  Every group is a compile-time subset of the lane's PER
    combinations, so one body again serves every lane."""
    nfi = c.nf[0]
    o = ['int ii = i0, jj = j0 + jb_, kk = k0 + kb_, llx = l0 + lb_;']

    def g(m, ci):
        return 'gout[%d]' % (m * nfi + ci)

    def group(key):
        out = {}
        for m, dd in enumerate(d):
            out.setdefault(key(dd), []).append(m)
        return sorted(out.items())

    # vk[i,l] += sum_{j,k} gout * dm[j,k]
    for dl, ms in group(lambda t: t[2]):
        for ci in range(nfi):
            o.append('val = 0;')
            for m in ms:
                o.append('val += %s * dm[(jj+%d)*nao+(kk+%d)];'
                         % (g(m, ci), d[m][0], d[m][1]))
            o.append('atomicAdd(vk+(ii+%d)*nao+(llx+%d), val);' % (ci, dl))
    # vk[i,k] += sum_{j,l} gout * dm[j,l]
    for dk, ms in group(lambda t: t[1]):
        for ci in range(nfi):
            o.append('val = 0;')
            for m in ms:
                o.append('val += %s * dm[(jj+%d)*nao+(llx+%d)];'
                         % (g(m, ci), d[m][0], d[m][2]))
            o.append('atomicAdd(vk+(ii+%d)*nao+(kk+%d), val);' % (ci, dk))
    # vk[j,l] += sum_{i,k} gout * dm[i,k]
    for (dj, dl), ms in group(lambda t: (t[0], t[2])):
        o.append('val = 0;')
        for m in ms:
            for ci in range(nfi):
                o.append('val += %s * dm[(ii+%d)*nao+(kk+%d)];'
                         % (g(m, ci), ci, d[m][1]))
        o.append('atomicAdd(vk+(jj+%d)*nao+(llx+%d), val);' % (dj, dl))
    # vk[j,k] += sum_{i,l} gout * dm[i,l]
    for (dj, dk), ms in group(lambda t: (t[0], t[1])):
        o.append('val = 0;')
        for m in ms:
            for ci in range(nfi):
                o.append('val += %s * dm[(ii+%d)*nao+(llx+%d)];'
                         % (g(m, ci), ci, d[m][2]))
        o.append('atomicAdd(vk+(jj+%d)*nao+(kk+%d), val);' % (dj, dk))
    return o


def lex_table(lmax=3):
    rows = []
    for l in range(lmax + 1):
        for p in cart(l):
            rows.append('    %d, %d, %d,' % p)
    return ('\n// _c_cartesian_lexical_xyz (gvhf-rys/rys_constant.cu): the '
            '(x,y,z) powers of\n// each cartesian function, in PySCF order.\n'
            '__device__ const int c_lex[] = {\n' + '\n'.join(rows) + '\n};\n')


HEAD = r'''
// ---- (%(li)d%(lj)d%(lk)d%(ll)d) : %(ngout)d gout over %(gs)d lanes, %(nsq)d shell quartets per block
#define NTHREADS_%(tag)s %(nthreads)d
#define MINBLOCKS_%(tag)s %(minblocks)d
extern "C" __global__ void __launch_bounds__(NTHREADS_%(tag)s, MINBLOCKS_%(tag)s)
kh_%(tag)s(KARGS2)
{
    constexpr int NSQ = %(nsq)d;
    constexpr int GSTRIDE = %(gs)d;
    constexpr int NROOTS = %(nroots)d;
    constexpr int g_size = %(gsz)d;
    constexpr int nsq_per_block = NSQ;
    __shared__ double s_aij[%(mpp)d], s_iaij[%(mpp)d], s_ajaij[%(mpp)d];
    __shared__ double s_xij[%(mpp)d], s_yij[%(mpp)d], s_zij[%(mpp)d];
    __shared__ double s_cicj[%(mpp)d];

    int sq_id = threadIdx.x;
    int gout_id = threadIdx.y;
    int thread_id = NSQ * gout_id + sq_id;
    constexpr int threads = NSQ * GSTRIDE;
    int *bas_kl_idx = pool + blockIdx.x * queue_depth;
    __shared__ int ntasks, pair_ij;
    if (thread_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
while (pair_ij < npairs_ij) {
    int bas_ij = pair_ij_mapping[pair_ij];
    if (thread_id == 0) {
        ntasks = 0;
    }
    __syncthreads();
    fill_vk_tasks2(&ntasks, bas_kl_idx, bas_ij, nbas, pair_kl_mapping,
                   npairs_kl, q_cond, dm_cond, cutoff);
    if (ntasks == 0) {
        if (thread_id == 0) {
            pair_ij = atomicAdd(head, 1);
        }
        __syncthreads();
        continue;
    }

    extern __shared__ double shared_memory[];
    double *rlrk = shared_memory + sq_id;
    double *Rpq = shared_memory + NSQ * 3 + sq_id;
    double *akl_cache = shared_memory + NSQ * 6 + sq_id;
    double *fac_ijkl = shared_memory + NSQ * 9 + sq_id;
    double *gx = shared_memory + NSQ * 10 + sq_id;
    double *rw = shared_memory + NSQ * (g_size*3+10) + sq_id;

    __shared__ int ish, jsh;
    __shared__ double ri[3], rjri[3];
    __shared__ double *expi;
    __shared__ double *expj;
    if (thread_id == 0) {
        ish = bas_ij / nbas;
        jsh = bas_ij %% nbas;
        expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
    }
    if (thread_id < 3) {
        int ri_ptr = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj_ptr = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        ri[thread_id] = env[ri_ptr+thread_id];
        rjri[thread_id] = env[rj_ptr+thread_id] - ri[thread_id];
    }
    __syncthreads();
    int iprim = iprim_a;
    int jprim = jprim_a;
    double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
    double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
    double rr_ij = rjri[0]*rjri[0] + rjri[1]*rjri[1] + rjri[2]*rjri[2];
    for (int ij = thread_id; ij < iprim*jprim; ij += threads) {
        double ai = expi[ij/jprim];
        double aj = expj[ij%%jprim];
        double aij = ai + aj;
        double iaij = 1. / aij;
        double aj_aij = aj * iaij;
        s_aij  [ij] = aij;
        s_iaij [ij] = iaij;
        s_ajaij[ij] = aj_aij;
        s_xij  [ij] = ri[0] + rjri[0] * aj_aij;
        s_yij  [ij] = ri[1] + rjri[1] * aj_aij;
        s_zij  [ij] = ri[2] + rjri[2] * aj_aij;
        s_cicj [ij] = ci[ij/jprim] * cj[ij%%jprim] * exp(-ai * aj_aij * rr_ij);
    }
    __syncthreads();
%(lanepre)s
    for (int task_id = sq_id; task_id < ntasks+sq_id; task_id += NSQ) {
        __syncthreads();
        int kprim = kprim_a;
        int lprim = lprim_a;
        int bas_kl = bas_kl_idx[task_id];
        int ksh = bas_kl / nbas;
        int lsh = bas_kl %% nbas;
        double fac_sym = PI_FAC;
        if (task_id < ntasks) {
            if (ish == jsh) fac_sym *= .5;
            if (ksh == lsh) fac_sym *= .5;
            if (ish*nbas+jsh == bas_kl) fac_sym *= .5;
        } else {
            fac_sym = 0;
        }
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        if (gout_id == 0) {
            rlrk[0*NSQ] = rl[0] - rk[0];
            rlrk[1*NSQ] = rl[1] - rk[1];
            rlrk[2*NSQ] = rl[2] - rk[2];
            fac_ijkl[0] = fac_sym;
        }
        double gout[%(ngmax)d];
#pragma unroll
        for (int n = 0; n < %(ngmax)d; ++n) gout[n] = 0.;

        for (int klp = 0; klp < kprim*lprim; ++klp) {
            __syncthreads();
            if (gout_id == 0) {
                double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
                double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
                double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
                double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
                double ak = expk[klp/lprim];
                double al = expl[klp%%lprim];
                double akl = ak + al;
                double iakl = 1. / akl;
                double al_akl = al * iakl;
                double xlxk = rlrk[0*NSQ], ylyk = rlrk[1*NSQ], zlzk = rlrk[2*NSQ];
                double Kcd = exp(-ak * al_akl * (xlxk*xlxk+ylyk*ylyk+zlzk*zlzk));
                gx[0] = fac_ijkl[0] * ck[klp/lprim] * cl[klp%%lprim] * Kcd;
                akl_cache[0*NSQ] = akl;
                akl_cache[1*NSQ] = al_akl;
                akl_cache[2*NSQ] = iakl;
            }
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                __syncthreads();
                double aij = s_aij[ijp];
                double iaij = s_iaij[ijp];
                double akl = akl_cache[0*NSQ];
                double al_akl = akl_cache[1*NSQ];
                double iakl = akl_cache[2*NSQ];
                // Range separation: inv_om2 = 0 is the full-range operator,
                // 1/omega^2 the long-range (erf) one -- scaling the Rys
                // argument and roots by w^2/(w^2+theta) is exactly
                // rsqrt(s) -> rsqrt(s + aij*akl/w^2) here.
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
                    // fac = cicj*ckcl/(aij*akl*sqrt(aij+akl)), no division
                    gx[NSQ*g_size] = s_cicj[ijp] * iaij * iakl * inv_s * coef0;
                }
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                rys_roots_tab<NROOTS>(aij*akl*inv_s2 * rr, rw, NSQ, gout_id,
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
                    (void)b10; (void)b00; (void)b01;
%(recur)s
                    __syncthreads();
%(gout)s
                }
            }
        }
        if (task_id < ntasks) {
            int i0 = ao_loc[ish], j0 = ao_loc[jsh];
            int k0 = ao_loc[ksh], l0 = ao_loc[lsh];
            double *dm = dm_all;
            double *vk = vk_all;
            double val;
            (void)val;
%(contract)s
        }
    }
    if (thread_id == 0) {
        pair_ij = atomicAdd(head, 1);
    }
    __syncthreads();
}
}
'''


def _switch(bodies, indent):
    """switch (gout_id) over per-lane bodies, collapsing identical ones."""
    if len(bodies) == 1:
        return '\n'.join(indent + l for l in bodies[0])
    o = [indent + 'switch (gout_id) {']
    for t, b in enumerate(bodies):
        o.append(indent + f'case {t}:')
        o += [indent + l for l in b]
        o.append(indent + 'break;')
    o.append(indent + '}')
    return '\n'.join(o)


def scheme(c, threads, gout_max, shm_max):
    """Pick (nsq_per_block, gout_stride).

    Fewer lanes means more shell quartets in flight, and a warp's g-array reads
    are contiguous once nsq_per_block >= 16 -- which is the whole difference
    between this and the general kernel.  So take the smallest lane count whose
    gout fits in registers and whose 2D integrals fit in shared memory.
    """
    best = None
    gs = 1
    while gs <= threads:
        nsq = threads // gs
        per = -(-c.ngout // gs)
        shm = nsq * (10 + 3 * c.gsz + 2 * c.nroots) + MAX_PRIM_PAIR
        if per <= gout_max and shm * 8 <= shm_max:
            best = (nsq, gs, per, shm)
            break
        gs *= 2
    return best


def generate(classes, threads=256, gout_max=96, shm_max=48 * 1024,
             layout='blocked', minblocks=1, table=None, mode='compact',
             par_hrr=False):
    out, tab = [], {}
    if mode == 'compact':
        out.append(lex_table())
    for spec in classes:
        c = Class(*(int(x) for x in spec))
        if mode == 'compact':
            sch = scheme_compact(c, threads, gout_max, shm_max)
            if sch is None:
                sys.stderr.write(f'skip {c.tag}: no thread scheme fits\n')
                continue
            nsq, gs, per, w, shm = sch
            d = deltas(c, per)
            pre = lane_pre(c, per, d)
            gout = '\n'.join(' ' * 20 + l for l in gout_compact(c, per))
            contract = '\n'.join(' ' * 12 + l for l in contract_compact(c, per, d))
        else:
            sch = scheme(c, threads, gout_max, shm_max)
            if sch is None:
                sys.stderr.write(f'skip {c.tag}: no thread scheme fits\n')
                continue
            nsq, gs, w, shm = sch
            per = w
            own = lanes(c, gs, layout)
            pre = ''
            gout = _switch([gout_body(c, o) for o in own], ' ' * 20)
            contract = _switch([contract_body(c, o) for o in own], ' ' * 12)
        body = HEAD % dict(
            li=c.l[0], lj=c.l[1], lk=c.l[2], ll=c.l[3], tag=c.tag,
            ngout=c.ngout, gs=gs, nsq=nsq, nroots=c.nroots, gsz=c.gsz,
            mpp=MAX_PRIM_PAIR, ngmax=max(w, 1), lanepre=pre,
            nthreads=nsq * gs, minblocks=minblocks,
            recur=(recurrence_split(c, 20) if par_hrr
                   else recurrence_fused(c, 20)),
            gout=gout, contract=contract)
        out.append(body)
        tab[c.tag] = dict(name='kh_' + c.tag, nsq=nsq, gout_stride=gs,
                          nroots=c.nroots, g_size=c.gsz,
                          shm_fixed=shm - MAX_PRIM_PAIR, minblocks=minblocks)
        sys.stderr.write(
            f'generated kh_{c.tag}: ngout={c.ngout} threads={nsq}x{gs} '
            f'gout/lane={w} g_size={c.gsz} nroots={c.nroots} '
            f'shm={shm*8/1024:.1f}KB\n')
    if table:
        with open(table + '.tmp', 'w') as f:
            json.dump(tab, f, indent=1, sort_keys=True)
        os.replace(table + '.tmp', table)
    return '\n'.join(out)


#: The classes this generator is responsible for: everything with l <= 3
#: except the 19 that gen_kernels.py runs with one thread per shell quartet
#: (that design keeps the 2D integrals in registers and is still faster for
#: them), and 2021, where GPU4PySCF's own unrolled kernel wins -- 0.87x,
#: measured.  codegen/build_khigh.sh passes the same list explicitly; this is
#: the copy electrolites._gen uses when it generates on demand.
_LIFTED = {'0000', '1000', '1010', '1011', '1100', '1110', '1111', '2000',
           '2010', '2011', '2020', '2100', '2110', '2200', '3000', '3010',
           '3020', '3100', '3200', '2021'}
DEFAULT_CLASSES = [f'{i}{j}{k}{l}'
                   for i in range(4) for j in range(i+1)
                   for k in range(i+1) for l in range(k+1)
                   if f'{i}{j}{k}{l}' not in _LIFTED]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('classes', nargs='?', default=','.join(DEFAULT_CLASSES))
    ap.add_argument('--threads', type=int, default=256)
    # doubles of gout each lane accumulates.  Wider means fewer lanes, which
    # means more shell quartets per block and less of the block idling through
    # the recurrences; 96 was the best of 42/64/96 on def2-TZVP (see
    # bench/sweep_kh.sh).
    ap.add_argument('--gout-max', type=int, default=96)
    ap.add_argument('--shm-max', type=int, default=48 * 1024)
    ap.add_argument('--layout', default='blocked', choices=['blocked', 'rr'])
    ap.add_argument('--minblocks', type=int, default=1)
    ap.add_argument('--table', default=None)
    ap.add_argument('--mode', default='compact', choices=['compact', 'full'])
    ap.add_argument('--par-hrr', action='store_true',
                    help='spread the horizontal transfers over every lane')
    a = ap.parse_args()
    print(generate(a.classes.split(','), a.threads, a.gout_max, a.shm_max,
                   a.layout, a.minblocks, a.table, a.mode, a.par_hrr))
