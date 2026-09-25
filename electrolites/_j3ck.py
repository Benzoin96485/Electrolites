"""The three-centre Coulomb kernels of `electrolites/codegen/gen_j3c.py`.

`docs/DF_WORK_LIST.md` **W19**.  `fastdfj3c` drives GPU4PySCF's two
three-centre McMurchie--Davidson kernels; this module is the replacement for
the kernels themselves, class by class, and the tables they read:

* ``pair_data``/``pair_inv`` -- the product centre, ``aij`` and ``1/aij`` of
  every stored shell pair, which upstream recomputes (three divisions) for
  every (pair, auxiliary shell);
* ``aux_data``/``aux_inv`` -- centre, exponent, ``1/ak`` and the scaled
  contraction coefficient of every sorted auxiliary shell.

A class is launched with this module's kernel when its ``lij + lk`` is at most
`gen_j3c.MAX_ORDER`; the shell-pair *groups* that meet a higher class are left
to GPU4PySCF's kernels whole, launched with the block offsets of those groups
alone (`Plan.upstream_runs`).  Classes of similar register need share one
*tier* kernel that reads each block's class from a table, so a build is a
handful of launches rather than one per class -- one launch per class on one
stream serialises thirty small grids, and W9 measured exactly that at 0.42x
(`docs/README_fastdfj3c.md` §8); the tiers themselves go on a small pool of
streams.

This module patches nothing.
"""
import functools
import os

import numpy as np
import cupy as cp

from ._paths import KERNEL_DIR
from .codegen import gen_j3c

#: highest ``lij + lk`` this module's kernels serve
MAX_ORDER = int(os.environ.get('FASTDFJ3C_KMAX_ORDER', gen_j3c.MAX_ORDER))
#: streams the tier launches are spread over
NSTREAMS = int(os.environ.get('FASTDFJ3C_KSTREAMS', 4))
#: shell pairs one thread block of the density kernel walks
DM_PPB = int(os.environ.get('FASTDFJ3C_KDM_PPB', 2048))

_ATTR = '_electrolites_j3ck'

# PySCF's _bas slots
ANG_OF, NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF, PTR_BAS_COORD = 1, 2, 3, 5, 6, 7


@functools.lru_cache(maxsize=16)
def _module(classes, kb_items, tier_items):
    src = open(os.path.join(KERNEL_DIR, 'fastj_prologue.cu')).read()
    src += gen_j3c.source(list(classes), dict(kb_items), dict(tier_items))
    return cp.RawModule(code=src, options=('-std=c++17',), backend='nvrtc')


class Plan:
    """Tables and launch lists of one ``Int3c2eOpt`` on one device."""

    def __init__(self, opt, c, max_order=None, kb=(), tiers=()):
        max_order = MAX_ORDER if max_order is None else max_order
        self.envs = opt.int3c2e_envs
        mol = opt.mol
        nbas = mol.nbas
        bas = np.asarray(opt._bas)
        env = np.asarray(opt._env)
        if (bas[:, NPRIM_OF] != 1).any() or (bas[:, NCTR_OF] != 1).any():
            # `Int3c2eOpt.build` decontracts both bases, so this does not
            # happen; if it ever does the kernels would be wrong, not slow
            raise RuntimeError('fastdfj3c kernels need a decontracted basis')

        idx = np.asarray(opt.shl_pair_idx, dtype=np.int64)
        ish, jsh = idx // nbas, idx % nbas
        ai = env[bas[ish, PTR_EXP]]
        aj = env[bas[jsh, PTR_EXP]]
        ri = env[bas[ish, PTR_BAS_COORD][:, None] + np.arange(3)]
        rj = env[bas[jsh, PTR_BAS_COORD][:, None] + np.arange(3)]
        aij = ai + aj
        # the same expression, in the same order, that upstream evaluates on
        # the card for every (pair, auxiliary shell)
        pc = (ai[:, None] * ri + aj[:, None] * rj) / aij[:, None]
        self.pair_data = cp.asarray(np.hstack([pc, aij[:, None]]))
        self.pair_inv = cp.asarray(1. / aij)

        naux = opt.auxmol.nbas
        k = nbas + np.arange(naux)
        ak = env[bas[k, PTR_EXP]]
        rk = env[bas[k, PTR_BAS_COORD][:, None] + np.arange(3)]
        ck = env[bas[k, PTR_COEFF]] * 34.98683665524972497     # PI_FAC
        self.aux_data = cp.asarray(np.hstack([rk, ak[:, None]]))
        self.aux_inv = cp.asarray(np.stack([1. / ak, ck], axis=1).ravel())
        aux_l = bas[k, ANG_OF]
        cart = (aux_l + 1) * (aux_l + 2) // 2
        self.aux_cart_loc = cp.asarray(
            np.append(0, np.cumsum(cart))[:-1].astype(np.int32))
        self.aux_xyz_loc = c.aux_xyz_loc
        self.pair_loc = c.pair_loc

        # auxiliary classes: contiguous, sorted by l
        self.aux_groups = []
        k0 = 0
        while k0 < naux:
            k1 = k0
            while k1 < naux and aux_l[k1] == aux_l[k0]:
                k1 += 1
            self.aux_groups.append((int(aux_l[k0]), k0, k1))
            k0 = k1
        lk_max = max(l for l, _, _ in self.aux_groups)

        off = c._sp_offsets_host
        self.ours, self.theirs = [], []
        #: every group, empty or not, and whether this module serves it --
        #: `subset` needs the full list to map a screened pair list back
        self._groups, self._is_ours = [], []
        for lij, lo, hi in zip(c.group_lij, off[:-1], off[1:]):
            mine = lij + lk_max <= max_order
            self._groups.append((int(lij), int(lo), int(hi)))
            self._is_ours.append(mine)
            if hi <= lo:
                continue
            (self.ours if mine else self.theirs).append(
                (int(lij), int(lo), int(hi)))
        self.classes = tuple(sorted({(lij, lk) for lij, _, _ in self.ours
                                     for lk, _, _ in self.aux_groups}))
        self.npair = c.npair
        self.pairs_ours = sum(hi - lo for _, lo, hi in self.ours)
        #: ``((kind, lij, lk), kb), ...`` and ``((kind, tier), (threads,
        #: minblocks)), ...`` -- tuples, so that they can key the module
        self.kb = tuple(kb)
        self.tiers = tuple(tiers)
        self._theirs_offsets = {}
        self._tables = {}

    def module(self):
        return _module(self.classes, self.kb, self.tiers)

    def subset(self, keep, shl_pair_idx):
        """A plan over the stored shell pairs ``keep`` selects (a device
        boolean mask in ``shl_pair_idx`` order), for a screened build.

        The pair tables are *gathered*, not masked: the kernels index them by
        position, and ``pair_loc`` still points into the full Hermite density,
        so a gathered list reads and writes exactly the entries of the pairs
        it kept.  The groups keep their classes and their order; the one host
        synchronisation is the count per group.  Returns the plan and the
        gathered ``shl_pair_idx`` the groups left to GPU4PySCF are launched
        with.
        """
        idx = cp.nonzero(keep)[0].astype(np.int32)
        bounds = np.array([lo for _, lo, _ in self._groups]
                          + [self._groups[-1][2]], dtype=np.int64)
        ck = cp.concatenate([cp.zeros(1, dtype=np.int32),
                             cp.cumsum(keep, dtype=np.int32)])
        newb = cp.asnumpy(ck[cp.asarray(bounds)])
        sub = object.__new__(Plan)
        sub.__dict__.update(self.__dict__)
        sub.pair_data = self.pair_data[idx]
        sub.pair_inv = self.pair_inv[idx]
        sub.pair_loc = self.pair_loc[idx]
        sub.ours, sub.theirs = [], []
        sub._groups = []
        for n, (lij, _, _) in enumerate(self._groups):
            g = (lij, int(newb[n]), int(newb[n + 1]))
            sub._groups.append(g)
            if g[2] > g[1]:
                (sub.ours if self._is_ours[n] else sub.theirs).append(g)
        sub.npair = int(newb[-1])
        sub.pairs_ours = sum(hi - lo for _, lo, hi in sub.ours)
        sub._tables = {}
        sub._theirs_offsets = {}
        return sub, shl_pair_idx[idx]

    def kb_of(self, kind, lij, lk):
        return dict(self.kb).get((kind, lij, lk),
                                 gen_j3c.default_kb(kind, lij, lk))

    def nt_of(self, kind, t):
        return dict(self.tiers).get((kind, t), gen_j3c.TIER_CFG[kind][t])[0]

    def tables(self, kind, ppb=None, only=None):
        """``[(tier, device block table, block count), ...]`` for one
        contraction: every (class, pair range, auxiliary range) a block of
        that tier's kernel is given, the widest classes first so that the
        long blocks start before the short ones.  ``only`` restricts it to
        one ``(lij, lk)``, which is how a class is timed alone."""
        ppb = ppb or DM_PPB
        key = (kind, ppb, only, self.kb, self.tiers)
        got = self._tables.get(key)
        if got is not None:
            return got
        got = self._build_tables(kind, ppb, only, self.ours)
        self._tables[key] = got
        return got

    def upstream_runs(self, chunk):
        """``[(device offsets, block count), ...]``, one per contiguous run of
        the groups left to GPU4PySCF, each group split into ``chunk`` pairs
        (0 = whole groups).  A run, not one list: upstream's kernels treat
        every consecutive pair of offsets as a block, so a gap between two
        groups would be computed as a block of the pairs in between."""
        got = self._theirs_offsets.get(chunk)
        if got is not None:
            return got
        runs = []
        for _, lo, hi in self.theirs:
            part = np.append(np.arange(lo, hi, chunk, dtype=np.int32) if chunk
                             else np.array([lo], dtype=np.int32),
                             np.int32(hi))
            if runs and runs[-1][-1] == part[0]:
                runs[-1] = np.append(runs[-1], part[1:])
            else:
                runs.append(part)
        got = [(cp.asarray(r), len(r) - 1) for r in runs]
        self._theirs_offsets[chunk] = got
        return got

    def _build_tables(self, kind, ppb, only, groups):
        """The block tables of `tables`, for the shell-pair ``groups`` given
        (``[(lij, lo, hi), ...]``): built with NumPy, because a 113-atom
        density contraction is ~10^5 blocks."""
        per_tier = {}
        for cid, (lij, lk) in enumerate(self.classes):
            if only is not None and (lij, lk) != only:
                continue
            t = gen_j3c.tier_of(lij + lk)
            nt = self.nt_of(kind, t)
            kb = self.kb_of(kind, lij, lk)
            step = ppb if kind == 'dm' else nt
            for alk, k0, k1 in self.aux_groups:
                if alk != lk:
                    continue
                ks = np.arange(k0, k1, kb, dtype=np.int32)
                for glij, lo, hi in groups:
                    if glij != lij or hi <= lo:
                        continue
                    ps = np.arange(lo, hi, step, dtype=np.int32)
                    P0 = np.repeat(ps, len(ks))
                    K0 = np.tile(ks, len(ps))
                    r = np.empty((len(P0), 5), dtype=np.int32)
                    r[:, 0] = cid
                    r[:, 1] = P0
                    r[:, 3] = K0
                    if kind == 'dm':
                        r[:, 2] = np.minimum(P0 + step, hi)
                        r[:, 4] = k1
                    else:
                        r[:, 2] = hi
                        r[:, 4] = np.minimum(K0 + kb, k1)
                    per_tier.setdefault(t, []).append((lij + lk, r))
        got = []
        for t in sorted(per_tier, reverse=True):
            parts = [r for _, r in sorted(per_tier[t], key=lambda y: -y[0])]
            r = np.concatenate(parts)
            got.append((t, cp.asarray(r.ravel()), len(r)))
        return got

    # ------------------------------------------------------------------
    def _launches(self, kind, ppb, only):
        """``[(kernel, threads, block table, blocks)]``, cached: a Coulomb
        build of a 10-atom molecule is ~1 ms, so the host side of a launch
        is not free."""
        key = ('launch', kind, ppb, only, self.kb, self.tiers)
        got = self._tables.get(key)
        if got is None:
            mod = self.module()
            got = [(mod.get_function(gen_j3c.tier_name(kind, t)),
                    self.nt_of(kind, t), tab, nblk)
                   for t, tab, nblk in self.tables(kind, ppb, only)]
            self._tables[key] = got
        return got

    def _launch(self, kind, args, ppb=None, only=None):
        launches = self._launches(kind, ppb or DM_PPB, only)
        if len(launches) == 1 or NSTREAMS <= 1:
            for f, nt, tab, nblk in launches:
                f((nblk,), (nt,), args + (tab,))
            return
        pool = getattr(self, '_pool', None)
        if pool is None:
            pool = self._pool = (
                [cp.cuda.Stream(non_blocking=True) for _ in range(NSTREAMS)],
                cp.cuda.Event(block=False, disable_timing=True),
                [cp.cuda.Event(block=False, disable_timing=True)
                 for _ in range(NSTREAMS)])
        streams, fork, joins = pool
        cur = cp.cuda.get_current_stream()
        fork.record(cur)
        used = min(len(launches), len(streams))
        for s in streams[:used]:
            s.wait_event(fork)
        for n, (f, nt, tab, nblk) in enumerate(launches):
            with streams[n % used]:
                f((nblk,), (nt,), args + (tab,))
        for s, ev in zip(streams[:used], joins):
            ev.record(s)
            cur.wait_event(ev)

    def contract_dm(self, dm_xyz, rho, ppb=None, only=None):
        """``rho[P_cart] += sum_ij (ij|P) dm_xyz[ij]`` over this module's
        groups.  ``rho`` is the Cartesian sorted auxiliary vector."""
        self._launch('dm', (self.pair_data, self.pair_inv, self.pair_loc,
                            dm_xyz, self.aux_data, self.aux_inv,
                            self.aux_cart_loc, rho), ppb, only)

    def contract_auxvec(self, aux_xyz, vj_xyz, only=None):
        """``vj_xyz[ij] += sum_P (ij|P) aux_xyz[P]`` over this module's
        groups.  ``aux_xyz`` is the Hermite auxiliary vector ``aux_to_Rt``
        makes."""
        self._launch('aux', (self.pair_data, self.pair_inv, self.pair_loc,
                             vj_xyz, self.aux_data, self.aux_inv,
                             self.aux_xyz_loc, aux_xyz), None, only)


def plan(opt, c):
    """The cached `Plan` of one ``Int3c2eOpt`` on this device."""
    by_device = getattr(opt, _ATTR, None)
    if by_device is None:
        by_device = {}
        setattr(opt, _ATTR, by_device)
    key = cp.cuda.Device().id
    p = by_device.get(key)
    if p is None or p.envs is not opt.int3c2e_envs:
        p = Plan(opt, c)
        by_device[key] = p
    return p


__all__ = ['plan', 'Plan', 'MAX_ORDER']
