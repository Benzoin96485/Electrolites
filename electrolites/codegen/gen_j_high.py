"""Write the McMurchie-Davidson Coulomb (J) kernels GPU4PySCF does not unroll.

GPU4PySCF unrolls twelve ``(lij,lkl)`` classes, up to ``lij+lkl == 5``.  A
6-31G* organic system already reaches ``(3,3)``, ``(4,2)``, ``(4,3)`` and
``(4,4)``; a triple-zeta basis reaches much further.  Every class it does not
unroll goes to one general kernel, ``md_j_1dm_kernel`` in
``gvhf-md/md_contract_j.cu``, which pays for its generality in three places:

  1. **the Hermite index table.**  Its density contraction reads
     ``Rt[Rt2_address[k*nf3ij+i] * nsq_per_block]`` -- a ``uint16`` load, an
     integer multiply-add and only then the ``double`` it wanted -- twice for
     each of the ``nf3ij*nf3kl`` pairs (1225 of them for ``(4,4)``), because
     the shape of the class is a run-time argument.  Written per class the
     address is a compile-time constant: the index load and the address
     arithmetic both disappear, and with them a two-deep dependent load chain.
  2. **the Rt recurrence.**  ``iter_Rt_n`` reads a parent index and a
     ``t/u/v`` factor out of ``__constant__`` memory for every element of
     every level, into a ``double Rt_tmp[31]`` whose trip count is not known
     at compile time.  Written per class it is straight-line code with literal
     indices and a register budget the class fixes.
  3. **the scaffolding the twelve unrolled classes already avoid** -- the
     incomplete-gamma values staged through shared memory at stride
     ``nsq_per_block``, ``fac/(aij*akl*sqrt(aij+akl))`` and
     ``aij*akl/(aij+akl)`` evaluated per quartet, and ``boys_fn``'s
     range-separation branches carried at run time although a Coulomb build
     never has ``omega != 0``.  These are the rewrites ``gen_j_kernels.py``
     applies to the twelve lifted classes.

What is *not* changed: the task and tile decomposition, the screening, the
symmetry factors and the Hermite recurrence itself are GPU4PySCF's, so the
Hermite-space J these kernels produce agrees with ``MD_build_j``'s to
round-off -- which is what ``benchmarks/perclass_j.py --check`` measures.

``gout_stride`` threads cooperate on one quartet, and the block is laid out so
that ``gout_id = threadIdx.y`` is **constant across a warp**
(``t_id = gout_id*nsq + sq_id`` with ``nsq`` a multiple of 32).  That is what
makes the per-``gout_id`` ``switch`` in the emitted code free: it is
warp-uniform, so it never diverges.  The same construction *inside* a warp is
what cost the per-lane unrolled exchange kernels up to 3.7x on the wide
classes; the distinction is the whole reason it is affordable here.

Run:
    python gen_j_high.py > ../electrolites/kernels/fastjhigh_generated.cu
    python gen_j_high.py --variants > .../fastjhigh_variants.cu
"""
import argparse
import json
import os
import sys

# --------------------------------------------------------------------------
# Hermite index bookkeeping.  ADDR(l,t,u,v) in gvhf-md/md_contract_j.cu
# enumerates t+u+v <= l with v fastest, then u, then t -- exactly the order
# `tuv_list` produces.
# --------------------------------------------------------------------------


def tuv_list(l):
    return [(t, u, v)
            for t in range(l + 1)
            for u in range(l - t + 1)
            for v in range(l - t - u + 1)]


def nf3(l):
    return (l + 1) * (l + 2) * (l + 3) // 6


_ADDR = {}


def addr(l, t, u, v):
    if l not in _ADDR:
        _ADDR[l] = {tuv: i for i, tuv in enumerate(tuv_list(l))}
    return _ADDR[l][(t, u, v)]


def rt_recurrence(n):
    """Level ``n`` of the Hermite recurrence.

    Yields ``(i, axis, j1, fac, j2)``: element ``i`` of level ``n`` is
    ``axis * Rt_prev[j1] + fac * Rt_prev[j2]``, with the second term absent
    when ``fac == 0``.  Element 0 is the incomplete-gamma value and is not
    produced here.  The decrement order is t, then u, then v, which is the
    order GPU4PySCF's own unrolled kernels emit.
    """
    out = []
    for i, (t, u, v) in enumerate(tuv_list(n)):
        if i == 0:
            continue
        if t > 0:
            axis, dec, back, fac = 'xpq', (t-1, u, v), (t-2, u, v), t-1
        elif u > 0:
            axis, dec, back, fac = 'ypq', (t, u-1, v), (t, u-2, v), u-1
        else:
            axis, dec, back, fac = 'zpq', (t, u, v-1), (t, u, v-2), v-1
        out.append((i, axis, addr(n-1, *dec), fac,
                    addr(n-1, *back) if fac else -1))
    return out


# --------------------------------------------------------------------------
# Per-class launch configuration.
#
#   threadsx/threadsy  the (bra, ket) shape of the quartet tile a block holds
#   gout_stride        threads cooperating on one quartet.  threadsx*threadsy
#                      must be a multiple of 32, so that gout_id is constant
#                      across a warp and the emitted switch is free
#   tilex/tiley        how many tiles a block walks in each direction
#   lreg               Rt levels 1..lreg are built in registers by every
#                      thread; the rest cooperatively, in shared memory
#   minblocks          the __launch_bounds__ occupancy target
#   dmreg              keep the bra Hermite density in registers instead of
#                      re-reading it from shared memory inside the ket loop
#
# Measured on an A100 with benchmarks/perclass_j.py; on another card these are
# a starting point, not an optimum.
# --------------------------------------------------------------------------
#: Every ``(lij, lkl)`` an spdf basis reaches that GPU4PySCF does not unroll.
#: A def2-TZVP or cc-pVTZ organic system reaches ``(6,6)``; the twelve classes
#: GPU4PySCF unrolls stop at ``lij+lkl == 5``, so at triple zeta most of the J
#: work was going to its general kernel.
ALL_CLASSES = ('3_3', '3_4', '4_2', '4_3', '4_4',
               '5_1', '5_2', '5_3', '5_4', '5_5',
               '6_0', '6_1', '6_2', '6_3', '6_4', '6_5', '6_6')

# Measured with benchmarks/sweep_j_high.py on an A100, on **two** systems of
# different size and basis -- a 162-atom ethanol cluster in 6-31G* and a
# 54-atom one in def2-TZVP -- and the entry kept is the one within a few per
# cent of the best on both, not the winner on either.  Every class prefers
# tiley = 8: the ket vj accumulator is what caps blocks per SM, and 8 is where
# two blocks fit.
CFG = {
    '3_4': dict(threadsx=16, threadsy=4, gout_stride=4, tilex=16, tiley=8,
                lreg=3, minblocks=1, dmreg=True),
    '3_3': dict(threadsx=8, threadsy=4, gout_stride=8, tilex=16, tiley=8,
                lreg=3, minblocks=2, dmreg=True),
    '4_2': dict(threadsx=16, threadsy=4, gout_stride=4, tilex=16, tiley=8,
                lreg=3, minblocks=1, dmreg=True),
    '4_3': dict(threadsx=16, threadsy=4, gout_stride=4, tilex=16, tiley=8,
                lreg=3, minblocks=1, dmreg=True),
    '4_4': dict(threadsx=16, threadsy=2, gout_stride=8, tilex=16, tiley=8,
                lreg=3, minblocks=1, dmreg=True),
}

#: Shared memory per SM on the card these defaults were chosen for.  A class
#: whose configuration fits twice over runs two blocks per SM.
SHM_PER_SM = 164 * 1024


def default_cfg(lij, lkl, shm_per_sm=SHM_PER_SM):
    """A launch configuration for a class with no measured entry.

    Picks the widest ket tile whose shared-memory request still leaves room
    for two blocks per SM, and falls back to the smallest request if none
    does.  This is a starting point for `benchmarks/sweep_j_high.py`, not an
    optimum -- the sweep is what turns it into one.
    """
    base = dict(threadsx=8, threadsy=4, gout_stride=8, tilex=16, tiley=4,
                lreg=3, minblocks=1, dmreg=False)
    best = None
    for tiley in (8, 16, 4):
        cfg = dict(base, tiley=tiley)
        try:
            _, meta = gen_class(lij, lkl, cfg, 'probe')
        except AssertionError:
            continue
        shm = meta['shm'] * 8
        if shm * 2 <= shm_per_sm:
            return dict(cfg, minblocks=2)
        if best is None or shm < best[1]:
            best = (cfg, shm)
    if best is None or best[1] > shm_per_sm:
        raise ValueError(
            f'({lij},{lkl}) needs {0 if best is None else best[1]/1024:.0f} KB '
            f'of shared memory for its Rt tensor, past the '
            f'{shm_per_sm/1024:.0f} KB an SM has.  Order {lij+lkl} is where '
            f'this design stops: the whole order-L Hermite triangle lives in '
            f'shared memory at one column per quartet, and nf3(L) grows as '
            f'L^3.  A class past that needs a different Rt strategy, not a '
            f'different launch configuration.')
    return best[0]



def reachable_classes(lmax):
    """The ``(lij, lkl)`` classes a basis of this ``lmax`` can reach.

    The J engine's task list is every ``(i,j,k,l)`` with ``i >= j``,
    ``i >= k``, ``k >= l`` and not ``(i == k and j < l)`` -- the same
    enumeration `fastj._setup_181` builds -- and the class of a task is
    ``(i+j, k+l)``.  Enumerating it is safer than writing the list out: the
    set is not simply ``lij >= lkl``, because e.g. ``(3,0|2,2)`` is a task and
    puts ``(3,4)`` in it.
    """
    out = set()
    for i in range(lmax + 1):
        for j in range(i + 1):
            for k in range(i + 1):
                for l in range(k + 1):
                    if i == k and j < l:
                        continue
                    out.add((i + j, k + l))
    return sorted(out)


def _layout(lij, lkl, cfg):
    """Shared-memory layout and register counts for a configuration.

    One place to compute them, because both `gen_class` and the configuration
    searches need them and a second copy of the formula would drift.
    """
    TX, TY, GS = cfg['threadsx'], cfg['threadsy'], cfg['gout_stride']
    TILEX, TILEY = cfg['tilex'], cfg['tiley']
    NDM = cfg.get('ndm', 1)
    NSQ = TX * TY
    THREADS = NSQ * GS
    BSX, BSY = TX * TILEX, TY * TILEY
    NF3IJ, NF3KL = nf3(lij), nf3(lkl)
    ORDER = lij + lkl
    RT = nf3(ORDER)
    off_rq = NF3KL * BSY * NDM
    off_iakl = off_rq + 4 * BSY
    mid = off_iakl + BSY
    if NDM == 1:
        # Rp_cache | s_iaij | dm_ij_cache share this region with vj_cache,
        # which is only live in the reduction after the ket loop.
        region = max(THREADS, 5 * TX + NF3IJ * TX)
    else:
        # no vj_cache: the reduction over ty is a warp shuffle
        region = 5 * TX + NF3IJ * TX * NDM
    off_rt = mid + region
    return dict(nsq=NSQ, threads=THREADS, bsizex=BSX, bsizey=BSY,
                nf3ij=NF3IJ, nf3kl=NF3KL, order=ORDER, rt=RT,
                off_rq=off_rq, off_iakl=off_iakl, mid=mid, off_rt=off_rt,
                shm=off_rt + RT * NSQ,
                nvj=-(-NF3IJ // GS), ndm=NDM)


#: Emitted-work budget for a multi-density kernel, as ``ndm*nf3ij*nf3kl``.
#:
#: Compile time is what sets this, not correctness or occupancy.  The density
#: dimension is a ``#pragma unroll`` loop, so the *source* is the size of a
#: single-density kernel, but the instruction count inside one basic block is
#: ``ndm`` times larger and ptxas is markedly super-linear in that:
#: measured per kernel with NVRTC on an A100, ``(2|1)`` at ndm=4 takes 1.0 s,
#: ``(3|3)`` at ndm=16 takes 19.5 s and ``(4|4)`` at ndm=16 takes **92.8 s**.
#: Emitting every class at every block size costs 413 s for the 6-31G* subset
#: alone, which is more than the whole rest of the package
#: (docs/ROUND2_1.8.1.md 4: 37 s).
#:
#: 4900 is ``4 * nf3(4) * nf3(4)``: every class a double-zeta polarised basis
#: reaches, plus ``(5,3)`` and ``(6,2)``, at the block size this package ships
#: (four densities -- see `docs/ROUND3_B4A.md` for why larger blocks lose).
#: A class past the budget falls through to GPU4PySCF at ``n_dm > 1``; raise
#: it, or name classes explicitly with ``--classes``, to pay for more.
MDM_EMIT_BUDGET = 4900

#: How many doubles of ``vj_ij`` accumulator a thread may carry.  A thread on
#: sm_80 has 255 registers; the kernel needs the Rt scratch, the Hermite
#: geometry and the gamma_inc array as well, so the accumulator gets a share.
VJ_IJ_REGISTER_BUDGET = 64

#: Launch configurations for the multi-density kernels, per ``(tag, ndm)``.
#: Empty until `benchmarks/sweep_j_high.py --ndm` has been run on the card in
#: question; `default_ndm_cfg` supplies a starting point meanwhile.
NDM_CFG = {}


def default_ndm_cfg(lij, lkl, ndm, shm_per_sm=SHM_PER_SM):
    """A starting configuration for a multi-density kernel.

    Three things differ from the single-density search.  ``threadsx*threadsy``
    is pinned to 32, because the ``vj_ij`` reduction over ``ty`` is a warp
    shuffle (see `gen_class`).  ``gout_stride`` is chosen so that the
    ``nvj*ndm`` accumulators fit in registers rather than spilling -- that is
    what decides the split, not occupancy.  And the ket tile is narrowed until
    the whole thing fits shared memory, since ``vj_kl_cache`` now scales with
    ``ndm`` on top of everything else.

    Raises `ValueError` if no configuration fits, which is how `main` decides
    that a class does not get a kernel at this ``ndm``.
    """
    order = lij + lkl
    lreg = order if order < 2 else min(3, order - 1)
    nf3ij = nf3(lij)
    work = ndm * nf3ij * nf3(lkl)
    if work > MDM_EMIT_BUDGET:
        raise ValueError(
            f'({lij},{lkl}) at ndm={ndm} is {work} of emitted work, past the '
            f'{MDM_EMIT_BUDGET} budget -- ptxas time, not correctness.  This '
            f'class falls through to GPU4PySCF at n_dm > 1; raise '
            f'MDM_EMIT_BUDGET to pay for it')
    best = None
    # `gout_stride` splits the Hermite slots of the bra over cooperating
    # threads, so it is pointless past `nf3ij` -- and the smallest value that
    # keeps the `nvj*ndm` accumulators inside the register budget is the one
    # to take, because a thread that holds more slots reads each Rt element
    # once for several of them instead of once for each.
    gs_choices = [g for g in (1, 2, 4, 8, 16) if g <= max(1, nf3ij)]
    for gs in gs_choices:
        if 32 * gs > 512:
            continue
        nvj = -(-nf3ij // gs)
        if nvj * ndm > VJ_IJ_REGISTER_BUDGET:
            continue
        for tiley in (8, 4, 2, 1):
            cfg = dict(threadsx=8, threadsy=4, gout_stride=gs, tilex=16,
                       tiley=tiley, lreg=lreg, minblocks=1, dmreg=False,
                       ndm=ndm)
            lay = _layout(lij, lkl, cfg)
            if lay['threads'] > 1024:
                continue
            shm = lay['shm'] * 8
            if shm * 2 <= shm_per_sm:
                return dict(cfg, minblocks=2)
            if shm <= shm_per_sm and (best is None or shm < best[1]):
                best = (cfg, shm)
    if best is None:
        raise ValueError(
            f'({lij},{lkl}) at ndm={ndm} does not fit: the Rt tensor, the '
            f'ndm-deep vj_kl accumulator and the ndm-deep bra density cache '
            f'together need more than the {shm_per_sm/1024:.0f} KB an SM has, '
            f'and no gout_stride keeps nvj*ndm inside '
            f'{VJ_IJ_REGISTER_BUDGET} registers.  Use a smaller ndm for this '
            f'class -- the host splits n_dm over whatever blocks exist.')
    return best[0]



HEAD = r'''// Generated by codegen/gen_j_high.py -- do not edit.
#ifndef JARGS
#define JARGS double *vj, double *dm_all, int *bas, double *env,       \
              int nbas, int npairs_ij, int npairs_kl,                  \
              int *pair_ij_mapping, int *pair_kl_mapping,              \
              int *pair_ij_loc, int *pair_kl_loc,                      \
              float *qd_ij_max, float *qd_kl_max,                      \
              float *q_cond, float cutoff
#endif
// The multi-density kernels take one extra argument: the stride, in doubles,
// between one density's Hermite vector and the next.  `vj` and `dm_all` point
// at the first density of the block this launch handles, so the host offsets
// them by dm_offset*dm_size and the kernel indexes m*dm_size from there.
#ifndef JARGSDM
#define JARGSDM double *vj, double *dm_all, int *bas, double *env,     \
              int nbas, int npairs_ij, int npairs_kl,                  \
              int *pair_ij_mapping, int *pair_kl_mapping,              \
              int *pair_ij_loc, int *pair_kl_loc,                      \
              float *qd_ij_max, float *qd_kl_max,                      \
              float *q_cond, float cutoff, int dm_size
#endif
'''


class Emit:
    """A tiny indentation-tracking line buffer."""

    def __init__(self):
        self.lines = []
        self.ind = 1

    def __call__(self, s=''):
        self.lines.append('    ' * self.ind + s if s else '')
        return self

    def op(self, s):
        self(s)
        self.ind += 1
        return self

    def cl(self, s='}'):
        self.ind -= 1
        self(s)
        return self

    def text(self):
        return '\n'.join(self.lines)


def gen_class(lij, lkl, cfg, name):
    TX, TY, GS = cfg['threadsx'], cfg['threadsy'], cfg['gout_stride']
    TILEX, TILEY, LREG = cfg['tilex'], cfg['tiley'], cfg['lreg']
    NDM = cfg.get('ndm', 1)
    NSQ = TX * TY
    THREADS = NSQ * GS
    BSX, BSY = TX * TILEX, TY * TILEY
    NF3IJ, NF3KL = nf3(lij), nf3(lkl)
    ORDER = lij + lkl
    RT = nf3(ORDER)
    assert NSQ % 32 == 0, (name, 'nsq must be a multiple of 32 so that '
                                 'gout_id is warp-uniform')
    assert THREADS <= 1024, (name, THREADS)
    # LREG == ORDER is legal and means "the whole Rt tensor in registers":
    # (0|0) and (1|0) have one and four Hermite components, so there is no
    # level left to build cooperatively.  Those two classes carry 12 % of a
    # multi-density 6-31G* J build between them (benchmarks/ndm_j.py
    # --perclass) and have the largest amortisation ratios in the table, so
    # they are worth generating even though they are trivial kernels.
    assert 0 <= LREG <= ORDER, (name, LREG)
    assert TX >= 2 and (TX & (TX-1)) == 0 and (TY & (TY-1)) == 0
    # A multi-density kernel reduces vj_ij over ty with a warp shuffle rather
    # than through shared memory: with NDM accumulators per Hermite slot the
    # shared route would cost NDM*NVJ*(log2(TY)+2) __syncthreads() per bra
    # tile, which is most of what the block cap was supposed to buy back.  The
    # shuffle needs the TY lanes that share a bra pair to sit in one warp,
    # i.e. TX*TY == 32.
    assert NDM == 1 or NSQ == 32, (
        name, 'a multi-density kernel needs threadsx*threadsy == 32 so that '
              'the vj_ij reduction over ty is a warp shuffle')

    lay = _layout(lij, lkl, cfg)
    off_rq, off_iakl, mid = lay['off_rq'], lay['off_iakl'], lay['mid']
    off_rt, SHM = lay['off_rt'], lay['shm']

    ij_tuv, kl_tuv = tuv_list(lij), tuv_list(lkl)
    phase = [(-1) ** sum(f) for f in kl_tuv]
    rt_idx = [[addr(ORDER, *[ij_tuv[i][d] + kl_tuv[k][d] for d in range(3)])
               for i in range(NF3IJ)] for k in range(NF3KL)]
    ij_of = [[i for i in range(NF3IJ) if i % GS == g] for g in range(GS)]
    kl_of = [[k for k in range(NF3KL) if k % GS == g] for g in range(GS)]
    NVJ = max(len(x) for x in ij_of)
    NRTMP = max(((nf3(n) + GS - 1) // GS
                 for n in range(LREG + 1, ORDER + 1)), default=1)

    def dmij_ref_m(i, m):
        """`dmij_ref` with both indices given as C expressions."""
        if cfg['dmreg']:
            return f'dmij[{m}*{NF3IJ}+{i}]'
        return f'dm_ij_cache[{m}*{NF3IJ*TX}+({i})*{TX}]'

    def dmij_ref(i, m=None):
        """The bra Hermite density, from registers or from shared memory.

        ``m`` is the density index.  ``None`` means "emit it as the loop
        variable ``m``": the density dimension is a ``#pragma unroll`` loop,
        not replicated source, so a class costs the same number of emitted
        lines at NDM = 16 as at NDM = 1.  Replicating it would put (6|6) --
        84x84 Hermite components -- at 113 000 lines for one kernel, which is
        the mistake the per-lane exchange emission made (README_fastk.md).
        """
        if cfg['dmreg']:
            off = f'{i}' if m is None else f'{m*NF3IJ + i}'
            return f'dmij[{off}]' if m is not None else f'dmij[m*{NF3IJ}+{i}]'
        if m is None:
            return f'dm_ij_cache[m*{NF3IJ*TX}+{i*TX}]'
        return f'dm_ij_cache[{(m*NF3IJ + i)*TX}]'

    e = Emit()
    e.ind = 0
    if NDM > 1:
        # The density dimension is a `#pragma unroll` loop, and wrapping every
        # one of the 2*nf3ij*nf3kl Hermite contractions in one costs five
        # emitted lines where the single-density kernel needs one.  For (6|6)
        # that is 56 000 lines instead of 14 000, and NVRTC charges by the
        # line -- the whole family took 17 minutes to compile at def2-TZVP
        # before these two macros, which put it back to one line per
        # contraction.  The names carry the kernel's own, so that `_ksplit`
        # can cut the file into independent chunks without an #undef landing
        # in the next one.
        acc = dmij_ref_m('(I)', 'm_')
        e(f'#define JV_{name}(RTX, I) {{ double r_ = (RTX); '
          f'_Pragma("unroll") for (int m_ = 0; m_ < {NDM}; ++m_) '
          f'v[m_] += r_ * {acc}; }}')
        e(f'#define JW_{name}(RTX, S) {{ double r_ = (RTX); '
          f'_Pragma("unroll") for (int m_ = 0; m_ < {NDM}; ++m_) '
          f'vj_ij[m_*{NVJ}+(S)] += r_ * d[m_]; }}')
    e(f'// ({lij}|{lkl})  order={ORDER}  nf3ij={NF3IJ} nf3kl={NF3KL} Rt={RT}'
      f'  block={NSQ}x{GS}  tile={TX}x{TY}  tiles={TILEX}x{TILEY}'
      f'  shm={SHM*8/1024:.1f}KB  vj_ij={NVJ} rtmp={NRTMP}'
      f'{"  dm in registers" if cfg["dmreg"] else ""}'
      f'{f"  ndm={NDM}" if NDM > 1 else ""}')
    e(f'extern "C" __global__ void __launch_bounds__({THREADS}, '
      f'{cfg["minblocks"]})')
    e(f'{name}({"JARGSDM" if NDM > 1 else "JARGS"})')
    e('{')
    e.ind = 1

    # ---- block-level screening -------------------------------------------
    e(f'int task_ij0 = blockIdx.x * {BSX};')
    e(f'int task_kl0 = blockIdx.y * {BSY};')
    e('int pair_ij0 = pair_ij_mapping[task_ij0];')
    e('int pair_kl0 = pair_kl_mapping[task_kl0];')
    e('if (q_cond[pair_ij0] + q_cond[pair_kl0] < cutoff) return;')
    e('if (pair_ij_mapping == pair_kl_mapping && '
      f'task_ij0+{BSX} <= task_kl0) return;')
    e()
    e('int sq_id = threadIdx.x;')
    e('int gout_id = threadIdx.y;')
    e(f'int t_id = gout_id * {NSQ} + sq_id;')
    e(f'int tx = sq_id % {TX};')
    e(f'int ty = sq_id / {TX};')
    e('unsigned int lane_id = t_id % 32;')
    e(f'unsigned int group_id = lane_id / {TX};')
    e(f'unsigned int mask = {hex((1 << TX) - 1)}u << (group_id * {TX});')
    e()
    e('extern __shared__ double shm[];')
    e('double *vj_kl_cache = shm;')
    e(f'double *Rq_cache = shm + {off_rq};')
    e(f'double *s_iakl = shm + {off_iakl};')
    e(f'double *Rp_cache = shm + {mid};')
    e(f'double *s_iaij = shm + {mid + 4*TX};')
    e(f'double *dm_ij_cache = shm + {mid + 5*TX} + tx;')
    if NDM == 1:
        e(f'double *vj_cache = shm + {mid} + t_id;')
    e(f'double *Rt = shm + {off_rt} + sq_id;')
    e()
    e(f'for (int n = t_id; n < {NF3KL*BSY*NDM}; n += {THREADS}) '
      'vj_kl_cache[n] = 0.;')
    e('__syncthreads();')
    e()

    # ---- ket tile prologue ------------------------------------------------
    e.op(f'for (int n = t_id; n < {BSY}; n += {THREADS}) {{')
    e(f'int task_kl = blockIdx.y * {BSY} + n;')
    e.op('if (task_kl < npairs_kl) {')
    e('int pair_kl = pair_kl_mapping[task_kl];')
    e('int ksh = pair_kl / nbas;')
    e('int lsh = pair_kl % nbas;')
    e('double ak = env[bas[ksh*BAS_SLOTS+PTR_EXP]];')
    e('double al = env[bas[lsh*BAS_SLOTS+PTR_EXP]];')
    e('double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];')
    e('double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];')
    e('double akl = ak + al;')
    e('double iakl = 1. / akl;')
    for m in range(3):
        e(f'Rq_cache[n+{m*BSY}] = (ak * rk[{m}] + al * rl[{m}]) * iakl;')
    e(f'Rq_cache[n+{3*BSY}] = akl;')
    e('s_iakl[n] = iakl;')
    e.cl('} else {')
    e.ind += 1
    for m in range(3):
        e(f'Rq_cache[n+{m*BSY}] = 1e5;')
    e(f'Rq_cache[n+{3*BSY}] = 1.;')
    e('s_iakl[n] = 1.;')
    e.cl()
    e.cl()
    e()

    # ---- bra tile loop ----------------------------------------------------
    e.op(f'for (int batch_ij = 0; batch_ij < {TILEX}; ++batch_ij) {{')
    e(f'int task_ij0 = (blockIdx.x * {TILEX} + batch_ij) * {TX};')
    e('if (task_ij0 >= npairs_ij) break;')
    # The batch-level q, not the block-level one.  GPU4PySCF screens a
    # (bra batch, ket batch) pair against `q_cond_kl[task_kl0]` and
    # `q_cond_ij[task_ij0]` with both taken at *batch* granularity; the pair
    # lists 1.8.x builds are not sorted by q, so substituting the block's
    # first pair is not conservative in either direction -- it over-screens
    # wherever the block's first pair happens to be weaker than the batch's,
    # and that shows up as a 1e-3 relative error on a dense density.
    e('float q_ij_batch = q_cond[pair_ij_mapping[task_ij0]];')
    e('__syncthreads();')
    e.op(f'if (t_id < {TX}) {{')
    e('int task_ij = task_ij0 + t_id;')
    e.op('if (task_ij < npairs_ij) {')
    e('int pair_ij = pair_ij_mapping[task_ij];')
    e('int ish = pair_ij / nbas;')
    e('int jsh = pair_ij % nbas;')
    e('double ai = env[bas[ish*BAS_SLOTS+PTR_EXP]];')
    e('double aj = env[bas[jsh*BAS_SLOTS+PTR_EXP]];')
    e('double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];')
    e('double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];')
    e('double aij = ai + aj;')
    e('double iaij = 1. / aij;')
    for m in range(3):
        e(f'Rp_cache[t_id+{m*TX}] = (ai * ri[{m}] + aj * rj[{m}]) * iaij;')
    e(f'Rp_cache[t_id+{3*TX}] = aij;')
    e('s_iaij[t_id] = iaij;')
    e.cl('} else {')
    e.ind += 1
    for m in range(3):
        e(f'Rp_cache[t_id+{m*TX}] = 2e5;')
    e(f'Rp_cache[t_id+{3*TX}] = 1.;')
    e('s_iaij[t_id] = 1.;')
    e.cl()
    e.cl()
    # task_ij stays *unclamped*: it decides both the ij_loc0 to read and,
    # inside the ket loop, whether this lane holds a real bra pair at all.
    # GPU4PySCF clamps a copy for the pair_ij_loc lookup and then shadows it
    # with a fresh unclamped one inside the ket loop; conflating the two makes
    # the padding lanes of a partly-filled bra tile contribute a small
    # spurious quartet to vj_kl (and only to vj_kl, since vj_ij is guarded
    # separately) -- a relative 1e-5 that no physical density exposes.
    e('int task_ij = task_ij0 + tx;')
    e('int ij_loc0 = pair_ij_loc[task_ij < npairs_ij ? task_ij : task_ij0];')
    if NDM == 1:
        e(f'for (int n = gout_id * {TY} + ty; n < {NF3IJ}; n += {GS*TY}) '
          f'dm_ij_cache[n*{TX}] = dm_all[ij_loc0+n];')
    else:
        e.op(f'for (int n = gout_id * {TY} + ty; n < {NF3IJ}; '
             f'n += {GS*TY}) {{')
        e('#pragma unroll')
        e(f'for (int m = 0; m < {NDM}; ++m)')
        e(f'    dm_ij_cache[n*{TX}+m*{NF3IJ*TX}] = '
          f'dm_all[m*dm_size+ij_loc0+n];')
        e.cl()
    e('__syncthreads();')
    for m, c in enumerate('xyz'):
        e(f'double {c}ij = Rp_cache[tx+{m*TX}];')
    e(f'double aij = Rp_cache[tx+{3*TX}];')
    e('double iaij_t = s_iaij[tx];')
    if cfg['dmreg']:
        e(f'double dmij[{NF3IJ*NDM}];')
        e('#pragma unroll')
        e(f'for (int n = 0; n < {NF3IJ*NDM}; ++n) dmij[n] = '
          f'dm_ij_cache[n*{TX}];')
    e(f'double vj_ij[{NVJ*NDM}];')
    e('#pragma unroll')
    e(f'for (int n = 0; n < {NVJ*NDM}; ++n) vj_ij[n] = 0.;')
    e(f'double gamma_inc[{ORDER+1}];')
    e(f'double rtmp[{NRTMP}];')
    if NDM == 1:
        e('double val;')
    e()

    # ---- ket tile loop ----------------------------------------------------
    e.op(f'for (int batch_kl = 0; batch_kl < {TILEY}; ++batch_kl) {{')
    e(f'int task_kl0 = (blockIdx.y * {TILEY} + batch_kl) * {TY};')
    e('if (task_kl0 >= npairs_kl) break;')
    e('if (pair_ij_mapping == pair_kl_mapping && '
      f'task_ij0+{TX} <= task_kl0) break;')
    e(f'if (qd_ij_max[blockIdx.x*{TILEX}+batch_ij] '
      '+ q_cond[pair_kl_mapping[task_kl0]] < cutoff &&')
    e(f'    qd_kl_max[blockIdx.y*{TILEY}+batch_kl] + q_ij_batch '
      '< cutoff) continue;')
    e(f'int sq_kl = ty + batch_kl * {TY};')
    e('int task_kl = task_kl0 + ty;')
    e('double fac = PI_FAC;')
    e('if (task_ij >= npairs_ij || task_kl >= npairs_kl) fac = 0.;')
    e.op('if (pair_ij_mapping == pair_kl_mapping) {')
    e('if (task_ij == task_kl) fac *= .5;')
    e('else if (task_ij < task_kl) fac = 0.;')
    e.cl()
    for m, c in enumerate('xyz'):
        e(f'double {c}pq = {c}ij - Rq_cache[sq_kl+{m*BSY}];')
    e(f'double akl = Rq_cache[sq_kl+{3*BSY}];')
    e('double rr = xpq*xpq + ypq*ypq + zpq*zpq;')
    e('double inv_s = rsqrt(aij + akl);')
    e('fac *= iaij_t * s_iakl[sq_kl] * inv_s;')
    e('double theta = aij * akl * (inv_s * inv_s);')
    e(f'boys0_fn_reg<{ORDER}>(gamma_inc, theta, rr, fac);')
    e()

    # ---- Rt, register levels ----------------------------------------------
    e(f'// Rt levels 1..{LREG} in registers, {LREG+1}..{ORDER} cooperatively')
    prev = {0: f'gamma_inc[{ORDER}]'}
    for n in range(1, LREG + 1):
        cur = {0: f'gamma_inc[{ORDER-n}]'}
        for (i, axis, j1, fac, j2) in rt_recurrence(n):
            expr = f'{axis} * {prev[j1]}'
            if fac == 1:
                expr += f' + {prev[j2]}'
            elif fac:
                expr += f' + {fac} * {prev[j2]}'
            e(f'double r{n}_{i} = {expr};')
            cur[i] = f'r{n}_{i}'
        prev = cur
    e('__syncthreads();')

    def switch(bodies):
        """switch (gout_id) with one warp-uniform case per gout_id."""
        e.op('switch (gout_id) {')
        for g in range(GS):
            if not bodies[g]:
                e(f'case {g}: break;')
                continue
            e.op(f'case {g}: {{')
            for line in bodies[g]:
                e(line)
            e('break;')
            e.cl('}')
        e.cl()

    switch([[f'Rt[{i*NSQ}] = {prev[i]};'
             for i in range(nf3(LREG)) if i % GS == g] for g in range(GS)])

    # ---- Rt, cooperative levels ------------------------------------------
    for n in range(LREG + 1, ORDER + 1):
        rec = {i: (axis, j1, fac, j2)
               for (i, axis, j1, fac, j2) in rt_recurrence(n)}
        reads, writes = [], []
        for g in range(GS):
            idxs = [i for i in range(nf3(n)) if i % GS == g]
            r, w = [], []
            for slot, i in enumerate(idxs):
                if i == 0:
                    r.append(f'rtmp[{slot}] = gamma_inc[{ORDER-n}];')
                else:
                    axis, j1, fac, j2 = rec[i]
                    expr = f'{axis} * Rt[{j1*NSQ}]'
                    if fac == 1:
                        expr += f' + Rt[{j2*NSQ}]'
                    elif fac:
                        expr += f' + {fac} * Rt[{j2*NSQ}]'
                    r.append(f'rtmp[{slot}] = {expr};')
                w.append(f'Rt[{i*NSQ}] = rtmp[{slot}];')
            reads.append(r)
            writes.append(w)
        e('__syncthreads();')
        switch(reads)
        e('__syncthreads();')
        switch(writes)
    e('__syncthreads();')
    e()

    # ---- pass 1: vj_kl[f] = sum_e (-1)^|f| R_{e+f} dm_ij[e] ---------------
    # Every density reads the same Rt element, so with NDM > 1 the shared
    # load is hoisted out of the density dimension and feeds NDM fused
    # multiply-adds instead of one.  That, and not only the Rt recurrence
    # paid once, is where a multi-density pass gets its arithmetic intensity.
    e('// vj_kl[f] += sum_e (-1)^|f| R_{e+f} dm_ij[e];  f split over gout_id')
    bodies = []
    for g in range(GS):
        b = []
        for k in kl_of[g]:
            if NDM == 1:
                b.append('val = 0.;')
                for i in range(NF3IJ):
                    src = dmij_ref(i, 0)
                    b.append(f'val += Rt[{rt_idx[k][i]*NSQ}] * {src};')
                if phase[k] < 0:
                    b.append('val = -val;')
                b.append('#pragma unroll')
                b.append(f'for (int off = {TX//2}; off > 0; off /= 2) '
                         'val += __shfl_down_sync(mask, val, off);')
                b.append('if (tx == 0 && task_kl < npairs_kl) '
                         f'vj_kl_cache[sq_kl+{k*BSY}] += val;')
                continue
            b.append('{')
            b.append(f'double v[{NDM}];')
            b.append('#pragma unroll')
            b.append(f'for (int m = 0; m < {NDM}; ++m) v[m] = 0.;')
            for i in range(NF3IJ):
                b.append(f'JV_{name}(Rt[{rt_idx[k][i]*NSQ}], {i})')
            b.append('#pragma unroll')
            b.append(f'for (int m = 0; m < {NDM}; ++m) {{')
            b.append(f'    double t = {"-" if phase[k] < 0 else ""}v[m];')
            b.append('    #pragma unroll')
            b.append(f'    for (int off = {TX//2}; off > 0; off /= 2) '
                     't += __shfl_down_sync(mask, t, off);')
            b.append('    if (tx == 0 && task_kl < npairs_kl)')
            b.append(f'        vj_kl_cache[sq_kl+{k*BSY}+m*{NF3KL*BSY}] '
                     '+= t;')
            b.append('}')
            b.append('}')
        bodies.append(b)
    switch(bodies)
    e()

    # ---- pass 2: vj_ij[e] += sum_f (-1)^|f| R_{e+f} dm_kl[f] --------------
    e('// vj_ij[e] += sum_f (-1)^|f| R_{e+f} dm_kl[f];  e split over gout_id')
    e.op('if (task_kl < npairs_kl) {')
    e('int kl_loc0 = pair_kl_loc[task_kl];')
    bodies = []
    for g in range(GS):
        b = []
        for k in range(NF3KL):
            sgn = '-' if phase[k] < 0 else ''
            if NDM == 1:
                b.append(f'double d{k} = {sgn}dm_all[kl_loc0+{k}];')
                for slot, i in enumerate(ij_of[g]):
                    b.append(f'vj_ij[{slot}] += Rt[{rt_idx[k][i]*NSQ}] * d{k};')
                continue
            b.append('{')
            b.append(f'double d[{NDM}];')
            b.append('#pragma unroll')
            b.append(f'for (int m = 0; m < {NDM}; ++m) '
                     f'd[m] = {sgn}dm_all[m*dm_size+kl_loc0+{k}];')
            for slot, i in enumerate(ij_of[g]):
                b.append(f'JW_{name}(Rt[{rt_idx[k][i]*NSQ}], {slot})')
            b.append('}')
        bodies.append(b)
    switch(bodies)
    e.cl()
    e.cl()                                        # end ket tile loop
    e()

    # ---- reduce vj_ij over ty, then one atomicAdd per Hermite slot --------
    if NDM == 1:
        e.op('{')
        e('#pragma unroll')
        e.op(f'for (int n = 0; n < {NVJ}; ++n) {{')
        e(f'int i = n * {GS} + gout_id;')
        e('__syncthreads();')
        e('vj_cache[0] = vj_ij[n];')
        e.op(f'for (int stride = {TY//2}; stride > 0; stride /= 2) {{')
        e('__syncthreads();')
        e(f'if (ty < stride) vj_cache[0] += vj_cache[stride*{TX}];')
        e.cl()
        e('__syncthreads();')
        e(f'if (ty == 0 && i < {NF3IJ} && task_ij < npairs_ij)')
        e('    atomicAdd(vj+ij_loc0+i, vj_cache[0]);')
        e.cl()
        e.cl()
    else:
        # NSQ == 32, so the TY lanes sharing a bra pair are one warp apart by
        # TX: the reduction needs no shared memory and no barrier.
        e.op('{')
        e('#pragma unroll')
        e.op(f'for (int n = 0; n < {NVJ}; ++n) {{')
        e(f'int i = n * {GS} + gout_id;')
        e('#pragma unroll')
        e.op(f'for (int m = 0; m < {NDM}; ++m) {{')
        e(f'double v = vj_ij[m*{NVJ}+n];')
        e('#pragma unroll')
        e(f'for (int stride = {TY//2}; stride > 0; stride /= 2)')
        e(f'    v += __shfl_down_sync(0xffffffffu, v, stride*{TX});')
        e(f'if (ty == 0 && i < {NF3IJ} && task_ij < npairs_ij)')
        e('    atomicAdd(vj+m*dm_size+ij_loc0+i, v);')
        e.cl()
        e.cl()
        e.cl()
    e.cl()                                        # end bra tile loop
    e()

    # ---- flush vj_kl ------------------------------------------------------
    e('__syncthreads();')
    e.op('{')
    e(f'int xslot_id = t_id / {TY};')
    e(f'int kty = t_id % {TY};')
    e.op(f'for (int n = xslot_id; n < {NF3KL*TILEY}; n += {TX*GS}) {{')
    e(f'int kl = n / {TILEY};')
    e(f'int batch_kl = n - kl * {TILEY};')
    e(f'int sq_kl = kty + batch_kl * {TY};')
    e(f'int task_kl = blockIdx.y * {BSY} + sq_kl;')
    e.op('if (task_kl < npairs_kl) {')
    e('int kl_loc0 = pair_kl_loc[task_kl];')
    if NDM == 1:
        e(f'atomicAdd(vj+kl_loc0+kl, vj_kl_cache[sq_kl+kl*{BSY}]);')
    else:
        e('#pragma unroll')
        e(f'for (int m = 0; m < {NDM}; ++m)')
        e(f'    atomicAdd(vj+m*dm_size+kl_loc0+kl, '
          f'vj_kl_cache[sq_kl+kl*{BSY}+m*{NF3KL*BSY}]);')
    e.cl()
    e.cl()
    e.cl()
    e.ind = 0
    e('}')
    meta = dict(order=ORDER, shm=SHM, bsizex=BSX, bsizey=BSY,
                nf3ij=NF3IJ, nf3kl=NF3KL, threadsx=TX, threadsy=TY,
                gout_stride=GS, block=[NSQ, GS], minblocks=cfg['minblocks'],
                tilex=TILEX, tiley=TILEY, dmreg=bool(cfg['dmreg']),
                lreg=LREG, nvj=NVJ, nrtmp=NRTMP, ndm=NDM)
    return e.text(), meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--variants', action='store_true',
                    help='emit the launch-bound / dm-register sweep')
    ap.add_argument('--classes', default=None,
                    help='comma separated lij_lkl tags')
    ap.add_argument('--ndm', default='',
                    help='comma separated density-block sizes.  Non-empty '
                         'switches to the multi-density family: one kernel '
                         'per (class, ndm), named j_dm<ndm>_<lij>_<lkl>, for '
                         'the J build in CPHF, TDDFT and any other caller '
                         'with more than one density (catalogue item B4a).  '
                         'A class that does not fit at an ndm is skipped, and '
                         'the host splits n_dm over the blocks that exist')
    ap.add_argument('--vj-budget', type=int, default=None,
                    help='override VJ_IJ_REGISTER_BUDGET, the number of '
                         'vj_ij accumulator doubles a thread may carry.  '
                         'Lowering it raises gout_stride, which trades '
                         'registers (and so occupancy) for more threads '
                         'cooperating on one quartet -- the axis '
                         'benchmarks/ndm_j.py --sweep-budget searches')
    ap.add_argument('--lmax', type=int, default=3,
                    help='with --ndm, generate every class a basis of this '
                         'maximum angular momentum reaches (default spdf)')
    ap.add_argument('--set', action='append', default=[],
                    help='tag:key=value override, e.g. 4_4:tiley=8')
    ap.add_argument('--json', default=None,
                    help='write the launch table here (default: next to the '
                         'kernels, unless --variants)')
    a = ap.parse_args()

    if a.vj_budget:
        global VJ_IJ_REGISTER_BUDGET
        VJ_IJ_REGISTER_BUDGET = a.vj_budget
    ndms = [int(x) for x in a.ndm.split(',') if x]
    if ndms:
        tags = (a.classes.split(',') if a.classes else
                [f'{lij}_{lkl}' for lij, lkl in reachable_classes(a.lmax)])
        cfgs = {}
        skipped = []
        for n in ndms:
            for t in tags:
                lij, lkl = (int(x) for x in t.split('_'))
                key = f'dm{n}_{t}'
                if (t, n) in NDM_CFG:
                    cfgs[key] = dict(NDM_CFG[t, n])
                    continue
                try:
                    cfgs[key] = default_ndm_cfg(lij, lkl, n)
                except ValueError as exc:
                    skipped.append((key, str(exc)))
        for key, why in skipped:
            sys.stderr.write(f'skipped j_{key}: {why}\n')
    else:
        tags = a.classes.split(',') if a.classes else list(ALL_CLASSES)
        cfgs = {}
        for t in tags:
            lij, lkl = (int(x) for x in t.split('_'))
            cfgs[t] = dict(CFG[t]) if t in CFG else default_cfg(lij, lkl)
    for spec in a.set:
        tag, kv = spec.split(':', 1)
        k, v = kv.split('=', 1)
        cfgs[tag][k] = (v.lower() in ('1', 'true', 'yes')
                        if k == 'dmreg' else int(v))

    out = [HEAD]
    table = {}
    for tag, cfg in cfgs.items():
        lij, lkl = (int(x) for x in tag.split('_')[-2:])
        if a.variants:
            for mb in (1, 2):
                for dmreg in (True, False):
                    c = dict(cfg, minblocks=mb, dmreg=dmreg)
                    nm = f'j_{tag}_mb{mb}{"dm" if dmreg else "nd"}'
                    src, meta = gen_class(lij, lkl, c, nm)
                    out.append(src)
                    table[nm[2:]] = meta
        else:
            src, meta = gen_class(lij, lkl, cfg, f'j_{tag}')
            out.append(src)
            table[tag] = meta
            sys.stderr.write(
                f'generated j_{tag}: order={meta["order"]} '
                f'shm={meta["shm"]*8/1024:.1f}KB block={meta["block"]} '
                f'tile={meta["threadsx"]}x{meta["threadsy"]} '
                f'tiles={meta["tilex"]}x{meta["tiley"]} '
                f'vj_ij={meta["nvj"]} rtmp={meta["nrtmp"]}\n')
    path = a.json
    if path is None and not a.variants:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'kernels',
                            'fastjmdm_launch.json' if ndms
                            else 'fastjhigh_launch.json')
    if path:
        with open(path, 'w') as f:
            json.dump(table, f, indent=1, sort_keys=True)
    print('\n'.join(out))


if __name__ == '__main__':
    main()
