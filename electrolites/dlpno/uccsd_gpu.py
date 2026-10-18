"""Spin-blocked open-shell DLPNO-CCSD iterations on the device.

Equations and projection placement: ``ref_uhf2`` (spin-orbital T1-dressed
CCSD with Jiang et al.'s placement).  Storage is by spin block:

* an ordered spin-orbital pair (I, J) keeps the (spin I, spin J) block of
  its amplitude, T_IJ; the other block of a mixed-spin pair is -T_JI, and
  T_JI = T_IJ^T;
* PNO overlaps S(p, q) are kept per spin (alpha: PNO-PNO; beta: (PNO+SOMO)
  - (PNO+SOMO));
* every residual term is expanded into the spin-block products it actually
  contains; each list of products of the same shape and spins runs as one
  ``vbatch.VChainPlan`` (one kernel launch per chain step).

Integrals: every pair keeps its exchange blocks (x V^s|y V^t), 1-external
(x y|m V^s) and (z m|z' V^s), and 3-external (z V^s|V^t V^t); strong pairs
keep their DF tensors per spin and the non-projected ring integrals, and
their T1-dressed DF quantities are rebuilt per spin every iteration.
"""
import time

import numpy as np
import cupy as cp
import cupyx

from electrolites.dlpno.ragged import Layout, Ragged, op
from electrolites.dlpno.vbatch import VChainPlan, VDotPlan
from electrolites.dlpno.ccsd_gpu import _DeviceDIIS, _rnorm


class _Reg:
    def __init__(self):
        self.ids = {}
        self.keys = []

    def __call__(self, key):
        i = self.ids.get(key)
        if i is None:
            i = self.ids[key] = len(self.keys)
            self.keys.append(key)
        return i

    def get(self, key):
        return self.ids.get(key)

    def __len__(self):
        return len(self.keys)


class _Terms:
    """Column store of a term list.  ``add`` appends a row (all calls use the
    same keyword order); columns are materialised on first read."""

    def __init__(self):
        self._keys = None
        self._rows = []
        self._cols = None
        self._np = {}

    def add(self, **kw):
        if self._keys is None:
            self._keys = tuple(kw)
        if self._cols is not None:
            for k, v in kw.items():
                c = self._cols[k]
                if isinstance(c, np.ndarray):
                    c = self._cols[k] = c.tolist()
                c.append(v)
            self._np = {}
        else:
            self._rows.append(tuple(kw.values()))

    def _mat(self):
        if self._cols is None:
            if self._keys is None:
                self._cols = {}
            elif not self._rows:
                self._cols = {k: [] for k in self._keys}
            else:
                # one C-level conversion of the row tuples, then numpy columns
                # (integer columns exact: indices < 2**53)
                first = self._rows[0]
                M = np.array(self._rows, dtype=np.float64)
                self._cols = {}
                for j, k in enumerate(self._keys):
                    v = first[j]
                    if isinstance(v, (bool, np.bool_)):
                        self._cols[k] = M[:, j].astype(bool)
                    elif isinstance(v, (int, np.integer)):
                        self._cols[k] = M[:, j].astype(np.int64)
                    else:
                        self._cols[k] = M[:, j].copy()
            self._rows = None
        return self._cols

    def __getitem__(self, k):
        return self._mat()[k]

    def __setitem__(self, k, v):
        self._mat()[k] = list(v)
        self._np = {}

    def __contains__(self, k):
        return k in self._mat()

    def get(self, k, default=None):
        return self._mat().get(k, default)

    def a(self, k):
        key = ('a', k)
        if key not in self._np:
            self._np[key] = np.asarray(self.get(k, []), dtype=np.int64)
        return self._np[key]

    def f(self, k):
        key = ('f', k)
        if key not in self._np:
            self._np[key] = np.asarray(self.get(k, []), dtype=np.float64)
        return self._np[key]

    def fd(self, k):
        """float column on the device (cached)."""
        key = ('fd', k)
        if key not in self._np:
            self._np[key] = cp.asarray(self.f(k))
        return self._np[key]

    def __len__(self):
        if getattr(self, '_n', None) is not None:
            return self._n
        if self._cols is None:
            return len(self._rows)
        return len(next(iter(self._cols.values()))) if self._cols else 0

    def compact(self, keep=('w',)):
        """drop the Python columns once the plans are built; ``keep``
        columns stay available (as numpy) through f() / a()."""
        n = len(self)
        for k in keep:
            if k in self:
                self.f(k)
                self.fd(k)
        self._cols, self._rows, self._n = {}, None, n


def _set_items(rag, items, mats):
    lay = rag.layout
    items = np.asarray(list(items), np.int64)
    dev = cp.cuda.Device().id
    if len(items) > 8 and all(isinstance(M, cp.ndarray) and M.device.id == dev for M in mats):
        # device items: one copy kernel per layout group
        from electrolites.dlpno.ragged import _gather_items
        mats = list(mats)
        gg = lay.group[items]
        for g in np.unique(gg):
            sel = np.where(gg == g)[0]
            _gather_items(rag.data[g], [mats[k] for k in sel], lay.rows[items[sel]], lay.cols[items[sel]],
                          slots=lay.slot[items[sel]])
        return
    for it, M in zip(items, mats):
        g, s = lay.group[it], lay.slot[it]
        M = M.reshape(lay.rows[it], lay.cols[it])
        rag.data[g][s, :M.shape[0], :M.shape[1]] = M


def _get(rag, it):
    lay = rag.layout
    return rag.data[lay.group[it]][lay.slot[it], :lay.rows[it], :lay.cols[it]]


def _scal(r):
    lay = r.layout
    out = cp.zeros(lay.n)
    for g, m in enumerate(lay.members):
        if len(m):
            out[cp.asarray(m)] = r.data[g][:, 0, 0]
    return out


class _SpinPlan:
    """A term list split by the spins of its overlap operands.

    specs: list of (kind, column, spin_column_or_None, trans) where kind
    names a store resolved at run time through ``stores[kind]`` (a Ragged,
    or a pair of Raggeds indexed by spin)."""

    def __init__(self, layouts, out_layout, terms, out_col, specs, out_row=None,
                 transpose_out=False, dot_left=None):
        self.parts = []
        n = len(terms)
        self.n = n
        if n == 0:
            return
        if transpose_out:
            specs = [(k, c, s, not t) for (k, c, s, t) in reversed(specs)]
        spin_cols = [s for (k, c, s, t) in specs if s is not None]
        if dot_left is not None and dot_left[2] is not None:
            spin_cols = [dot_left[2]] + spin_cols
        if spin_cols:
            km = np.stack([terms.a(c) for c in spin_cols], 1)
            # spins are 0 / 1: one integer code per term (np.unique on rows is slow)
            code = (km << np.arange(km.shape[1], dtype=np.int64)[None, :]).sum(axis=1)
            ucode, inv = np.unique(code, return_inverse=True)
            ukeys = (ucode[:, None] >> np.arange(km.shape[1], dtype=np.int64)[None, :]) & 1
            inv = inv.reshape(-1)
            order = np.argsort(inv, kind='stable')
            bounds = np.searchsorted(inv[order], np.arange(len(ukeys) + 1))
            groups = [(tuple(int(x) for x in ukeys[g]), order[bounds[g]:bounds[g + 1]])
                      for g in range(len(ukeys))]
        else:
            groups = [((), np.arange(n, dtype=np.int64))]
        for key, sel in groups:
            it = iter(key)
            if dot_left is not None and dot_left[2] is not None:
                lspin = next(it)
            ops, kinds = [], []
            for (kind, col, scol, tr) in specs:
                spin = next(it) if scol is not None else None
                lay = layouts[kind] if spin is None else layouts[kind][spin]
                ops.append(op(lay, terms.a(col)[sel], tr))
                kinds.append((kind, spin))
            if dot_left is None:
                orow = None if out_row is None else terms.a(out_row)[sel]
                plan = VChainPlan(out_layout, terms.a(out_col)[sel], ops, out_row=orow)
            else:
                kind, col, scol, tr = dot_left
                lay = layouts[kind] if scol is None else layouts[kind][lspin]
                plan = VDotPlan([op(lay, terms.a(col)[sel])], ops)
                kinds = [(kind, None if scol is None else lspin)] + kinds
            self.parts.append((plan, cp.asarray(sel), kinds))

    @staticmethod
    def _rags(stores, kinds):
        return [stores[k] if s is None else stores[k][s] for (k, s) in kinds]

    def run(self, out, stores, w=None):
        for plan, sel, kinds in self.parts:
            plan.run(out, self._rags(stores, kinds), w=None if w is None else w[sel])
        return out

    def dots(self, stores, n):
        out = cp.zeros(n)
        for plan, sel, kinds in self.parts:
            r = self._rags(stores, kinds)
            out[sel] = plan.run(r[:1], r[1:])
        return out


class UCCSDGPU:

    #: generate the F'_vv and oooo-ladder term lists with numpy (subclasses)
    KL_NUMPY = False

    def __init__(self, u):
        """u: ref_uhf2.RefUDLPNO2 after recompute_pnos() and cc_integrals()."""
        self.u = u
        pl = u.pl
        self.pl = pl
        self.so = so = u.so
        self.nso = len(so)
        self.spin = np.array([s for (x, s) in so], dtype=np.int64)
        self.lmo = np.array([x for (x, s) in so], dtype=np.int64)
        self.npair = npair = len(pl)
        self.nvs = np.array([[u.nva(p), u.nvb(p)] for p in range(npair)], dtype=np.int64)
        self.strong_sp = np.asarray(u.strong, bool)
        self.sop = list(u.sop)
        self.sidx = {k: n for n, k in enumerate(self.sop)}
        I = np.array([k[0] for k in self.sop])
        J = np.array([k[1] for k in self.sop])
        self.sp_of = pl.ij[self.lmo[I], self.lmo[J]]
        self.LT = Layout(self.nvs[self.sp_of, self.spin[I]], self.nvs[self.sp_of, self.spin[J]])
        self.dii = pl.ij[self.lmo, self.lmo]
        self.LT1 = Layout(self.nvs[self.dii, self.spin], 1)
        self.canon = [k for k in self.sop if k[0] < k[1]]
        self.strong = [k for k in self.canon if self.strong_sp[self.sp_of[self.sidx[k]]]]
        self.canon_sp = [p for p, (x, y) in enumerate(pl.pairs) if x <= y]
        self.cmap = np.array([p if pl.pairs[p][0] <= pl.pairs[p][1] else pl.ji[p]
                              for p in range(npair)], np.int64)
        self.rows = []
        for p in range(npair):
            ds = u.dom_so[p]
            self.rows.append([[I_ for I_ in ds if self.spin[I_] == s] for s in (0, 1)])
        self.Sreg = [_Reg(), _Reg()]
        self.t_build = {}

    offload = False

    def _decide_offload(self):
        """keep the strong pairs' DF tensors and the 3-external tensors in
        pinned host memory when they would take more than a third of the
        free device memory (env DLPNO_OFFLOAD=0/1 forces it)."""
        import os
        env = os.environ.get('DLPNO_OFFLOAD')
        if env is not None:
            return env == '1'
        u = self.u
        nk3 = ndf = 0.
        for c in self.canon_sp:
            na, nb_ = self.nvs[c]
            nz = 2 if self.pl.pairs[c][0] != self.pl.pairs[c][1] else 1
            nk3 += 8. * nz * 2 * (na ** 3 + nb_ ** 3 + na * nb_ * (na + nb_)) / 2
            if self.strong_sp[c]:
                A = len(u.ribfs[c])
                R = [len(self.rows[c][s]) for s in (0, 1)]
                ndf += 8. * A * (na * na + nb_ * nb_ + R[0] * na + R[1] * nb_)
        free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        from electrolites.dlpno import mgpu
        nd = len(mgpu.devices())
        self.offload_bytes = nk3 + ndf
        self._df_bytes, self._ndev, self._k3_bytes = ndf, nd, nk3
        # the DF groups are spread over the GPUs, the 3-external tensors stay home
        return nk3 + ndf / nd > free / 3.

    def tb(self, K, L, a, b):
        sK, sL = self.spin[K], self.spin[L]
        if (a, b) == (sK, sL):
            return self.sidx[(K, L)], 1.
        if sK != sL and (a, b) == (sL, sK):
            return self.sidx[(L, K)], -1.
        return None

    def S(self, p, q, s):
        return self.Sreg[s]((p, q))

    # ------------------------------------------------------------------
    def build(self):
        for name, fn in (('terms', self._terms), ('overlaps', self._overlaps),
                         ('ints', self._load_ints), ('plans', self._plans)):
            t0 = time.perf_counter()
            fn()
            self.t_build[name] = time.perf_counter() - t0

    def _terms(self):
        u, pl = self.u, self.pl
        sp, lmo = self.spin, self.lmo
        tb, S, ij = self.tb, self.S, pl.ij
        # projected singles T_n (canonical spatial pairs)
        self.tn = [_Terms(), _Terms()]
        self.cpos = {c: r for r, c in enumerate(self.canon_sp)}
        for c in self.canon_sp:
            for s in (0, 1):
                for r, K in enumerate(self.rows[c][s]):
                    self.tn[s].add(c=self.cpos[c], row=r, s=S(c, self.dii[K], s), K=K, sa=s)
        # F'_vv projections
        self.strong_c = [c for c in self.canon_sp if self.strong_sp[c]]
        self.fvpos = {c: r for r, c in enumerate(self.strong_c)}
        self.fv = [_Terms(), _Terms()]
        for c in ([] if self.KL_NUMPY else self.strong_c):
            ds = u.dom_so[c]
            for K in ds:
                for L in ds:
                    if K == L or ij[lmo[K], lmo[L]] < 0:
                        continue
                    kl = ij[lmo[K], lmo[L]]
                    for s in (0, 1):
                        for t in (0, 1):
                            b = tb(K, L, s, t)
                            if b is not None:
                                self.fv[s].add(c=self.fvpos[c], s1=S(kl, c, s), it=b[0], sa=s, w=-.5)
        # F'_oo dot products over same-spin (K, J)
        self.Freg = _Reg()
        self.fp = _Terms()
        for p, (k, j) in enumerate(pl.pairs):
            for K in u.dom_so[p]:
                if lmo[K] != k:
                    continue
                for J_ in u.dom_so[p]:
                    if lmo[J_] != j or sp[K] != sp[J_]:
                        continue
                    f = self.Freg((K, J_))
                    for l in u.lmos[p]:
                        for L in (u.so_index.get((l, 0)), u.so_index.get((l, 1))):
                            if L is None or L == K or L == J_:
                                continue
                            kl, jl = ij[k, l], ij[j, l]
                            for a in (0, 1):
                                for b_ in (0, 1):
                                    g, t = tb(K, L, a, b_), tb(J_, L, a, b_)
                                    if g is None or t is None:
                                        continue
                                    self.fp.add(f=f, g=g[0], sa=S(kl, jl, a), t=t[0],
                                                sb=S(kl, jl, b_), a=a, b=b_, w=.5 * g[1] * t[1])
        # residual chains
        self.rg, self.rb = _Terms(), _Terms()
        self.np_reg, self.z_reg, self.breg = _Reg(), _Reg(), _Reg()
        self.ring_np = [_Terms(), _Terms()]          # [non-transposed, transposed]
        self.ring_z = [_Terms(), _Terms()]
        for (I, J) in self.strong:
            o = self.sidx[(I, J)]
            p = self.sp_of[o]
            x, y, a, b_ = lmo[I], lmo[J], sp[I], sp[J]
            for K in range(self.nso):
                k = lmo[K]
                ik, kj = ij[x, k], ij[k, y]
                f = self.Freg.get((K, J))
                if ik >= 0 and K != I and sp[K] == b_ and f is not None:
                    t = tb(I, K, a, b_)
                    if t is not None:
                        self.rg.add(o=o, s1=S(p, ik, a), t=t[0], s2=S(p, ik, b_), f=f,
                                    a=a, b=b_, w=-t[1])
                f = self.Freg.get((K, I))
                if kj >= 0 and K != J and sp[K] == a and f is not None:
                    t = tb(K, J, a, b_)
                    if t is not None:
                        self.rg.add(o=o, s1=S(p, kj, a), t=t[0], s2=S(p, kj, b_), f=f,
                                    a=a, b=b_, w=-t[1])
            ds = u.dom_so[p]
            for K in ([] if self.KL_NUMPY else ds):
                for L in ds:
                    if K == L:
                        continue
                    kl = ij[lmo[K], lmo[L]]
                    if kl < 0:
                        continue
                    t = tb(K, L, a, b_)
                    if t is None:
                        continue
                    self.rb.add(o=o, s1=S(kl, p, a), t=t[0], s2=S(kl, p, b_),
                                beta=self.breg((I, J, K, L)), a=a, b=b_, w=.5 * t[1])
            for (P, Q, sign, trans) in ((I, J, 1., 0), (J, I, -1., 0), (I, J, -1., 1), (J, I, 1., 1)):
                ap, bp = (a, b_) if not trans else (b_, a)
                xp, xq = lmo[P], lmo[Q]
                for K in ds:
                    if K == P:
                        continue
                    k = lmo[K]
                    pk, qk = ij[xp, k], ij[xq, k]
                    for c_ in (0, 1):
                        t = tb(P, K, ap, c_)
                        if t is None:
                            continue
                        if not ((bp == sp[Q] and c_ == sp[K]) or (bp == c_ and sp[K] == sp[Q])):
                            continue
                        self.ring_np[trans].add(o=o, s1=S(p, pk, ap), t=t[0],
                                                np_=self.np_reg((p, K, Q, bp, c_, pk)),
                                                a=ap, w=sign * t[1])
                        self.ring_z[trans].add(o=o, s1=S(p, pk, ap), t=t[0], s2=S(qk, pk, c_),
                                               z=self.z_reg((Q, K, bp, c_)), s3=S(p, qk, bp),
                                               a=ap, c=c_, b=bp, w=sign * t[1])
        # Z quadratic part
        self.zq = _Terms()
        for zi, (J_, K, b_, c_) in enumerate(self.z_reg.keys):
            y, k = lmo[J_], lmo[K]
            q = ij[y, k]
            for L in u.dom_so[q]:                  # T1-dressing overlaps (_Z_dress)
                kl = ij[k, lmo[L]]
                if kl >= 0 and L != K:
                    for s in {sp[K], sp[J_], b_}:
                        S(q, kl, s)
            for l in u.lmos[q]:
                for L in (u.so_index.get((l, 0)), u.so_index.get((l, 1))):
                    if L is None or L == J_ or L == K:
                        continue
                    jl, kl = ij[y, l], ij[k, l]
                    for d in (0, 1):
                        t, g = tb(J_, L, b_, d), tb(K, L, c_, d)
                        if t is None or g is None:
                            continue
                        self.zq.add(z=zi, s1=S(q, jl, b_), t=t[0], s2=S(jl, kl, d), g=g[0],
                                    s3=S(kl, q, c_), b=b_, d=d, c=c_, w=.5 * t[1] * g[1])
        # singles
        self.fkc_reg, self.u_reg, self.kx_reg = _Reg(), _Reg(), _Reg()
        self.fkc, self.r1c, self.r1a2, self.uq, self.r1b = (_Terms() for _ in range(5))
        self.r1a1 = []
        for I in range(self.nso):
            x, a = lmo[I], sp[I]
            ii = self.dii[I]
            for K in range(self.nso):
                if K == I:
                    continue
                k = lmo[K]
                ik = ij[x, k]
                if ik < 0:
                    continue
                t = tb(I, K, a, sp[K])
                if t is None:
                    continue
                self.r1c.add(i=I, s1=S(ii, ik, a), t=t[0], fk=self.fkc_reg((K, ik)), a=a, w=t[1])
                self.r1a1.append((I, K, ik, t[0], t[1]))
                for L in u.dom_so[ik]:
                    l = lmo[L]
                    kl = ij[k, l]
                    if L == K or kl < 0 or sp[L] != a:
                        continue
                    self.r1a2.add(i=I, s1=S(ii, self.dii[L], a), L=L, kx=self.kx_reg((kl, K, L)),
                                  sa=S(kl, ik, sp[K]), t=t[0], sb=S(kl, ik, a),
                                  a=a, aK=sp[K], w=-t[1])
        for fi, (K, ik) in enumerate(self.fkc_reg.keys):
            k = lmo[K]
            for L in u.dom_so[ik]:
                kl = ij[k, lmo[L]]
                if L == K or kl < 0:
                    continue
                g = tb(K, L, sp[K], sp[L])
                if g is None:
                    continue
                self.fkc.add(f=fi, s1=S(ik, kl, sp[K]), g=g[0], s2=S(kl, self.dii[L], sp[L]), L=L,
                             a=sp[K], b=sp[L], w=g[1])
        for kl, (k, l) in enumerate(pl.pairs):
            for K in u.dom_so[kl]:
                if lmo[K] != k:
                    continue
                for L in u.dom_so[kl]:
                    if lmo[L] != l or K == L:
                        continue
                    for I in u.dom_so[kl]:
                        if sp[I] != sp[K]:
                            continue
                        t = tb(K, L, sp[I], sp[L])
                        if t is None:
                            continue
                        self.r1b.add(i=I, s1=S(self.dii[I], kl, sp[I]), t=t[0],
                                     u=self.u_reg((K, L, I)), a=sp[I], w=-t[1])
        for ui, (K, L, I) in enumerate(self.u_reg.keys):
            kl = ij[lmo[K], lmo[L]]
            self.uq.add(u=ui, kx=self.kx_reg((kl, K, L)), s=S(kl, self.dii[I], sp[I]), i=I, a=sp[I])
        # singles projected into pair spaces (energy and dressing)
        self.preg = _Reg()
        for (I, J) in self.canon:
            p = self.sp_of[self.sidx[(I, J)]]
            self.preg((I, p))
            self.preg((J, p))
        self.pt = _Terms()
        for r_, (Z, p) in enumerate(self.preg.keys):
            self.pt.add(r=r_, s=S(p, self.dii[Z], self.spin[Z]), z=Z, a=self.spin[Z])

    # ------------------------------------------------------------------
    def _overlaps(self):
        from electrolites.dlpno.lmp2_gpu import overlaps_chunked
        u = self.u
        Spao = getattr(u, 'S_pao_d', None)
        if Spao is None:
            Spao = cp.asarray(u.orb.S_pao)
        npair = self.npair
        Xa = [u.Xa[p] for p in range(npair)]
        Xbp = [u.Xb[p][:len(u.pao[p])] for p in range(npair)]
        Xbs = [u.Xb[p][len(u.pao[p]):] for p in range(npair)]
        self.LS, self.Sd = [], []
        for s in (0, 1):
            keys = self.Sreg[s].keys
            pp = np.array([k[0] for k in keys], np.int64)
            qq = np.array([k[1] for k in keys], np.int64)
            LS = Layout(self.nvs[pp, s], self.nvs[qq, s])
            if s == 0:
                S = overlaps_chunked(Spao, u.pao, Xa, pp, qq, LS)
            else:
                S = overlaps_chunked(Spao, u.pao, Xbp, pp, qq, LS, Xs=Xbs if u.ns else None)
            self.LS.append(LS)
            self.Sd.append(S)

    # ------------------------------------------------------------------
    def _load_ints(self):
        u, pl = self.u, self.pl
        si = u.so_index
        self.fix = [None] * self.npair
        if hasattr(u, 'gpu_pair_ints'):
            self.offload = self._decide_offload()
            rows_of = {c: self.rows[c] for c in self.canon_sp}
            strong_of = {c: bool(self.strong_sp[c]) for c in self.canon_sp}
            if hasattr(u, 'gpu_pair_ints_many'):
                it = u.gpu_pair_ints_many(self.canon_sp, rows_of, strong_of, host=self.offload)
            else:
                it = ((c, u.gpu_pair_ints(c, rows_of[c], strong_of[c], host=self.offload))
                      for c in self.canon_sp)
            for c, ent in it:
                self.fix[c] = self.fix[pl.ji[c]] = ent
        for c in ([] if hasattr(u, 'gpu_pair_ints') else self.canon_sp):
            x, y = pl.pairs[c]
            d = u.cc[c]
            B = cp.asarray(d['B'])
            o, pos = d['o'], d['pos']
            na = u.nva(c)
            vs = [slice(o, o + na), slice(o + na, o + u.nv(c))]
            lm = list(u.lmos[c])
            lpos = {m: r for r, m in enumerate(lm)}
            arows = [pos[si[(m, 0)]] for m in lm]
            Boo = B[:, arows][:, :, arows]
            ent = dict(lm=lm, lpos=lpos, Kx={}, K1={}, K3={}, J1={})
            for (z1, z2) in {(x, y), (y, x)}:
                for s in (0, 1):
                    for t in (0, 1):
                        A_, C_ = si.get((z1, s)), si.get((z2, t))
                        if A_ is not None and C_ is not None:
                            ent['Kx'][(z1, z2, s, t)] = B[:, vs[s], pos[A_]].T @ B[:, vs[t], pos[C_]]
            for s in (0, 1):
                Bms = cp.zeros((B.shape[0], len(lm), int(self.nvs[c, s])))
                for r, m in enumerate(lm):
                    M = si.get((m, s))
                    if M is not None:
                        Bms[:, r] = B[:, pos[M], vs[s]]
                ent['J1'][s] = cp.einsum('Q,Qmb->mb', Boo[:, lpos[x], lpos[y]], Bms)
                for (z1, z2) in {(x, y), (y, x)}:
                    Z2 = si.get((z2, s))
                    if Z2 is not None:
                        ent['K1'][(z1, z2, s)] = cp.einsum('Qm,Qb->mb', Boo[:, lpos[z1], :],
                                                           B[:, vs[s], pos[Z2]])
            for z in {x, y}:
                for s in (0, 1):
                    Z = si.get((z, s))
                    if Z is None:
                        continue
                    for t in (0, 1):
                        ent['K3'][(z, s, t)] = cp.einsum('Qc,Qbd->cbd', B[:, vs[s], pos[Z]],
                                                         B[:, vs[t], vs[t]])
            f = cp.asarray(d['f'])
            ent['fov'] = [f[[pos[I_] for I_ in self.rows[c][s]]][:, vs[s]] for s in (0, 1)]
            ent['ev'] = [cp.asarray(u.ea[c]), cp.asarray(u.eb[c])]
            ent['rowsp'] = [[lpos[self.lmo[I_]] for I_ in self.rows[c][s]] for s in (0, 1)]
            if self.strong_sp[c]:
                ent['Qov'] = [B[:, [pos[I_] for I_ in self.rows[c][s]]][:, :, vs[s]] for s in (0, 1)]
                ent['Qvv'] = [B[:, vs[s], vs[s]] for s in (0, 1)]
                ent['Qmz'] = {z: Boo[:, :, lpos[z]] for z in {x, y}}
                ent['Qaz'] = {(z, s): B[:, vs[s], pos[si[(z, s)]]]
                              for z in {x, y} for s in (0, 1) if (z, s) in si}
            self.fix[c] = self.fix[pl.ji[c]] = ent
        # stores of fixed blocks used by plans
        mats = [self.fix[kl]['Kx'][(self.lmo[K], self.lmo[L], self.spin[K], self.spin[L])]
                for (kl, K, L) in self.kx_reg.keys]
        self.LKx = Layout(np.array([m.shape[0] for m in mats], np.int64),
                          np.array([m.shape[1] for m in mats], np.int64))
        self.Kx = Ragged(self.LKx)
        _set_items(self.Kx, range(len(mats)), mats)
        self.G = Ragged(self.LT)
        mats = []
        for (I, J) in self.sop:
            p = self.sp_of[self.sidx[(I, J)]]
            x, y, a, b = self.lmo[I], self.lmo[J], self.spin[I], self.spin[J]
            M = self.fix[p]['Kx'][(x, y, a, b)]
            if a == b:
                M = M - self.fix[p]['Kx'][(y, x, a, a)]
            mats.append(M)
        _set_items(self.G, range(len(self.sop)), mats)
        mats = u.gpu_ring_np(self.np_reg.keys) if hasattr(u, 'gpu_ring_np') else []
        for (p, K, Q, bp, c_, pk) in ([] if hasattr(u, 'gpu_ring_np') else self.np_reg.keys):
            Nd = cp.asarray(u._ring_np(p, K, Q, pk))
            ra = slice(0, u.nva(p)) if bp == 0 else slice(u.nva(p), u.nv(p))
            ca = slice(0, u.nva(pk)) if c_ == 0 else slice(u.nva(pk), u.nv(pk))
            mats.append(Nd[ra, ca])
        self.LNP = Layout(np.array([m.shape[0] for m in mats], np.int64),
                          np.array([m.shape[1] for m in mats], np.int64))
        self.NP = Ragged(self.LNP)
        _set_items(self.NP, range(len(mats)), mats)
        ij = self.pl.ij
        zq = [ij[self.lmo[J_], self.lmo[K]] for (J_, K, b, c) in self.z_reg.keys]
        self.LZ = Layout(np.array([self.nvs[q, k[2]] for q, k in zip(zq, self.z_reg.keys)], np.int64),
                         np.array([self.nvs[q, k[3]] for q, k in zip(zq, self.z_reg.keys)], np.int64))
        self.LFk = Layout(np.array([self.nvs[ik, self.spin[K]] for (K, ik) in self.fkc_reg.keys], np.int64), 1)
        self.LU = Layout(np.array([self.nvs[ij[self.lmo[K], self.lmo[L]], self.spin[L]]
                                   for (K, L, I) in self.u_reg.keys], np.int64), 1)
        self.LPt = Layout(np.array([self.nvs[p, self.spin[Z]] for (Z, p) in self.preg.keys], np.int64), 1)
        self.U0 = Ragged(self.LU)
        mats = []
        for (K, L, I) in self.u_reg.keys:
            fx = self.fix[ij[self.lmo[K], self.lmo[L]]]
            mats.append(fx['K1'][(self.lmo[K], self.lmo[L], self.spin[L])][fx['lpos'][self.lmo[I]]])
        _set_items(self.U0, range(len(mats)), mats)
        self.Fk0 = Ragged(self.LFk)
        mats = []
        for (K, ik) in self.fkc_reg.keys:
            c = self.cmap[ik]
            mats.append(self.fix[c]['fov'][self.spin[K]][self.rows[c][self.spin[K]].index(K)])
        _set_items(self.Fk0, range(len(mats)), mats)
        # denominators
        fd = [np.diag(u.Fa_lmo), np.diag(u.Fb_lmo)]
        self.foo = [cp.asarray(u.Fa_lmo), cp.asarray(u.Fb_lmo)]
        D2 = []
        for g, m in enumerate(self.LT.members):
            pr, pc = self.LT.shapes[g]
            Dg = np.ones((len(m), pr, pc))
            for s_, it in enumerate(m):
                I, J = self.sop[it]
                p = self.sp_of[it]
                ea = (u.ea[p], u.eb[p])[self.spin[I]]
                eb = (u.ea[p], u.eb[p])[self.spin[J]]
                Dg[s_, :len(ea), :len(eb)] = (ea[:, None] + eb[None, :]
                                              - fd[self.spin[I]][self.lmo[I]] - fd[self.spin[J]][self.lmo[J]])
            D2.append(cp.asarray(Dg))
        self.D2 = Ragged(self.LT, D2)
        D1 = []
        for g, m in enumerate(self.LT1.members):
            pr = self.LT1.shapes[g][0]
            Dg = np.ones((len(m), pr, 1))
            for s_, I in enumerate(m):
                ii = self.dii[I]
                e = (u.ea[ii], u.eb[ii])[self.spin[I]]
                Dg[s_, :len(e), 0] = e - fd[self.spin[I]][self.lmo[I]]
            D1.append(cp.asarray(Dg))
        self.D1 = Ragged(self.LT1, D1)

    # ------------------------------------------------------------------
    def _make_spin_plan(self, *args, **kw):
        return _SpinPlan(*args, **kw)

    def _layouts(self):
        cs = np.array(self.canon_sp, np.int64)
        self.LTn = [Layout(np.array([len(self.rows[c][s]) for c in cs], np.int64), self.nvs[cs, s])
                    for s in (0, 1)]
        sc = np.array(self.strong_c, np.int64)
        self.LF = [Layout(self.nvs[sc, s], self.nvs[sc, s]) for s in (0, 1)]

    def _plans(self):
        LS, LT, LT1 = self.LS, self.LT, self.LT1
        self._layouts()
        self.Lfp = Layout(np.ones(len(self.Freg), np.int64), 1)
        self.Lbeta = Layout(np.ones(len(self.breg), np.int64), 1)
        lays = dict(S=LS, T=LT, G=LT, T1=LT1, NP=self.LNP, Z=self.LZ, Kx=self.LKx,
                    Fk=self.LFk, U=self.LU, Pt=self.LPt)
        SP = self._make_spin_plan
        self.p_tn = [SP(lays, self.LTn[s], self.tn[s], 'c', [('S', 's', 'sa', False), ('T1', 'K', None, False)],
                        out_row='row') for s in (0, 1)]
        self.p_fv = [SP(lays, self.LF[s], self.fv[s], 'c', [('S', 's1', 'sa', True), ('T', 'it', None, False),
                                                           ('G', 'it', None, True), ('S', 's1', 'sa', False)])
                     for s in (0, 1)]
        self.p_fp = SP(lays, None, self.fp, None, [('S', 'sa', 'a', False), ('T', 't', None, False),
                                                   ('S', 'sb', 'b', True)], dot_left=('G', 'g', None, False))
        self.p_rg = SP(lays, LT, self.rg, 'o', [('S', 's1', 'a', False), ('T', 't', None, False),
                                               ('S', 's2', 'b', True)])
        self.p_rb = SP(lays, LT, self.rb, 'o', [('S', 's1', 'a', True), ('T', 't', None, False),
                                               ('S', 's2', 'b', False)])
        self.p_ring_np, self.p_ring_z = [], []
        for tr in (0, 1):
            self.p_ring_np.append(SP(lays, LT, self.ring_np[tr], 'o',
                                     [('S', 's1', 'a', False), ('T', 't', None, False), ('NP', 'np_', None, True)],
                                     transpose_out=bool(tr)))
            self.p_ring_z.append(SP(lays, LT, self.ring_z[tr], 'o',
                                    [('S', 's1', 'a', False), ('T', 't', None, False), ('S', 's2', 'c', True),
                                     ('Z', 'z', None, True), ('S', 's3', 'b', True)],
                                    transpose_out=bool(tr)))
        self.p_zq = SP(lays, self.LZ, self.zq, 'z', [('S', 's1', 'b', False), ('T', 't', None, False),
                                                    ('S', 's2', 'd', False), ('G', 'g', None, True),
                                                    ('S', 's3', 'c', False)])
        self.p_fkc = SP(lays, self.LFk, self.fkc, 'f', [('S', 's1', 'a', False), ('G', 'g', None, False),
                                                       ('S', 's2', 'b', False), ('T1', 'L', None, False)])
        self.p_r1c = SP(lays, LT1, self.r1c, 'i', [('S', 's1', 'a', False), ('T', 't', None, False),
                                                  ('Fk', 'fk', None, False)])
        self.p_r1a2_w = SP(lays, None, self.r1a2, None, [('S', 'sa', 'aK', False), ('T', 't', None, True),
                                                        ('S', 'sb', 'a', True)], dot_left=('Kx', 'kx', None, False))
        self.p_r1a2 = SP(lays, LT1, self.r1a2, 'i', [('S', 's1', 'a', False), ('T1', 'L', None, False)])
        self.p_uq = SP(lays, self.LU, self.uq, 'u', [('Kx', 'kx', None, True), ('S', 's', 'a', False),
                                                    ('T1', 'i', None, False)])
        self.p_r1b = SP(lays, LT1, self.r1b, 'i', [('S', 's1', 'a', False), ('T', 't', None, False),
                                                  ('U', 'u', None, False)])
        self.p_pt = SP(lays, self.LPt, self.pt, 'r', [('S', 's', 'a', False), ('T1', 'z', None, False)])
        # index helpers
        self.rg_f = cp.asarray(self.rg.a('f'))
        self.rb_beta = cp.asarray(self.rb.a('beta'))
        self.fp_f = cp.asarray(self.fp.a('f'))
        self.canon_idx = cp.asarray(np.array([self.sidx[k] for k in self.canon], np.int64))
        self.strong_mask = np.zeros(len(self.sop), bool)
        for k in self.strong:
            self.strong_mask[self.sidx[k]] = True

    # ------------------------------------------------------------------
    # per-iteration pieces
    # ------------------------------------------------------------------
    def stores(self, T, T1, **kw):
        st = dict(S=self.Sd, T=T, G=self.G, T1=T1, NP=self.NP, Kx=self.Kx, U0=self.U0)
        st.update(kw)
        return st

    def residuals(self, T, T1):
        u, pl = self.u, self.pl
        lmo, sp = self.lmo, self.spin
        st = self.stores(T, T1)
        # projected singles
        Tn = [self.p_tn[s].run(Ragged(self.LTn[s]), st) for s in (0, 1)]
        tn = lambda p, s: _get(Tn[s], self.cpos[self.cmap[p]])
        # ---- DF part of strong canonical spatial pairs --------------------
        Fv = [Ragged(self.LF[s]) for s in (0, 1)]
        Fbov, Fai = {}, {}
        R = Ragged(self.LT)
        beta = cp.zeros(len(self.breg))
        bkeys = {}
        for n, (I, J, K, L) in enumerate(self.breg.keys):
            bkeys.setdefault((I, J), []).append((n, K, L))
        for c in self.strong_c:
            fx = self.fix[c]
            x, y = pl.pairs[c]
            Tc = [tn(c, s) for s in (0, 1)]
            Qov, Qvv, Qmz, Qaz = fx['Qov'], fx['Qvv'], fx['Qmz'], fx['Qaz']
            gam = sum(cp.einsum('Qme,me->Q', Qov[s], Tc[s]) for s in (0, 1))
            Fbv, Fbo, Qvt = [], [], []
            for s in (0, 1):
                Fb_vv = cp.diag(fx['ev'][s]) + cp.einsum('Q,Qab->ab', gam, Qvv[s]) \
                    - cp.einsum('Qae,me,Qmb->ab', Qvv[s], Tc[s], Qov[s])
                Fb_ov = fx['fov'][s] + cp.einsum('Q,Qkc->kc', gam, Qov[s]) \
                    - cp.einsum('Qke,me,Qmc->kc', Qov[s], Tc[s], Qov[s])
                Fbv.append(Fb_vv)
                Fbo.append(Fb_ov)
                Qvt.append(Qvv[s] - cp.einsum('ka,Qkb->Qab', Tc[s], Qov[s]))
                _set_items(Fv[s], [self.fvpos[c]], [Fb_vv - Tc[s].T @ Fb_ov])
            Fbov[c] = Fbo
            # dressed (Q| K z) and (Q| A z)
            Bk, Ba = {}, {}
            rowsp = fx['rowsp']
            for z in {x, y}:
                for s in (0, 1):
                    Z = u.so_index.get((z, s))
                    if Z is None:
                        continue
                    tz = Tc[s][self.rows[c][s].index(Z)]
                    bk = Qmz[z][:, rowsp[s]] + cp.einsum('Qkc,c->Qk', Qov[s], tz)
                    Bk[(z, s)] = bk
                    Ba[(z, s)] = Qaz[(z, s)] - bk @ Tc[s] + cp.einsum('Qab,b->Qa', Qvv[s], tz)
            # residual DF terms and beta for the strong spin-orbital pairs of c
            for p_ in {c, pl.ji[c]}:
                pass
            for (I, J) in self._strong_of(c):
                o = self.sidx[(I, J)]
                zi, zj, a, b = lmo[I], lmo[J], sp[I], sp[J]
                t = _get(T, o)
                M = Ba[(zi, a)].T @ Ba[(zj, b)]
                if a == b:
                    M = M - Ba[(zj, a)].T @ Ba[(zi, a)]
                M = M + cp.einsum('Qac,cd,Qbd->ab', Qvt[a], t, Qvt[b])
                _set_items(R, [o], [M])
                for (n, K, L) in bkeys.get((I, J), []):
                    sK, sL = sp[K], sp[L]
                    v = 0.
                    rK = self.rows[c][sK].index(K)
                    rL = self.rows[c][sL].index(L)
                    if sK == a and sL == b:
                        v = v + Bk[(zi, a)][:, rK] @ Bk[(zj, b)][:, rL]
                    if sK == b and sL == a:
                        v = v - Bk[(zj, b)][:, rK] @ Bk[(zi, a)][:, rL]
                    tt = self.tb(I, J, sK, sL)
                    if tt is not None:
                        tblk = _get(T, tt[0]) * tt[1]
                        v = v + cp.einsum('Qc,cd,Qd->', Qov[sK][:, rK], tblk, Qov[sL][:, rL])
                    beta[n] = v
            if x == y:
                for s in (0, 1):
                    X = u.so_index.get((x, s))
                    if X is None:
                        continue
                    tx = Tc[s][self.rows[c][s].index(X)]
                    rX = self.rows[c][s].index(X)
                    fvo = fx['fov'][s][rX]
                    Fb_ai = fvo + gam @ Qaz[(x, s)] - cp.einsum('Qae,me,Qm->a', Qvv[s], Tc[s],
                                                               Qmz[x][:, rowsp[s]])
                    foo = self.foo[s][cp.asarray([lmo[K] for K in self.rows[c][s]]), x]
                    Fb_oi = foo + gam @ Qmz[x][:, rowsp[s]] - cp.einsum('Qke,me,Qm->k', Qov[s], Tc[s],
                                                                       Qmz[x][:, rowsp[s]])
                    fai = (Fb_ai + Fbv[s] @ tx - Tc[s].T @ Fb_oi - Tc[s].T @ (Fbo[s] @ tx))
                    Fai[X] = fai
        # ---- F'_vv projections ------------------------------------------
        for s in (0, 1):
            self.p_fv[s].run(Fv[s], st, w=cp.full(self.p_fv[s].n, -.5) if self.p_fv[s].n else None)
        # ---- Fbar_oo and F'_oo -------------------------------------------
        Fpp = cp.zeros(len(self.Freg))
        for n, (K, J) in enumerate(self.Freg.keys):
            k, j, s = lmo[K], lmo[J], sp[K]
            p = pl.ij[k, j]
            c = self.cmap[p]
            fx = self.fix[c]
            v = self.foo[s][k, j]
            for s2 in (0, 1):
                if len(self.rows[c][s2]):
                    v = v + cp.sum(tn(p, s2) * fx['J1'][s2][fx['rowsp'][s2]])
            if len(self.rows[c][s]):
                v = v - cp.sum(tn(p, s) * fx['K1'][(j, k, s)][fx['rowsp'][s]])
            jj = self.dii[J]
            fo = Fbov[self.cmap[jj]][s]
            rK = self.rows[self.cmap[jj]][s].index(K)
            v = v + fo[rK] @ _get(T1, J).ravel()
            Fpp[n] = v
        if self.p_fp.n:
            dots = self.p_fp.dots(st, self.p_fp.n)
            cupyx.scatter_add(Fpp, self.fp_f, dots * cp.asarray(self.fp.f('w')))
        # ---- R2 projection terms -----------------------------------------
        for s in (0, 1):
            pass
        self._E_term(R, T, Fv)
        if self.p_rg.n:
            self.p_rg.run(R, st, w=Fpp[self.rg_f] * cp.asarray(self.rg.f('w')))
        if self.p_rb.n:
            self.p_rb.run(R, st, w=beta[self.rb_beta] * cp.asarray(self.rb.f('w')))
        # ring: Z intermediates (dressing via fixed integrals, quadratic via plans)
        Z = Ragged(self.LZ)
        self._Z_dress(Z, Tn, tn)
        if self.p_zq.n:
            self.p_zq.run(Z, st, w=cp.asarray(self.zq.f('w')))
        st['Z'] = Z
        for tr in (0, 1):
            if self.p_ring_np[tr].n:
                self.p_ring_np[tr].run(R, st, w=cp.asarray(self.ring_np[tr].f('w')))
            if self.p_ring_z[tr].n:
                self.p_ring_z[tr].run(R, st, w=cp.asarray(self.ring_z[tr].f('w')))
        # ---- R1 -------------------------------------------------------------
        R1 = Ragged(self.LT1)
        _set_items(R1, list(Fai.keys()), list(Fai.values()))
        Fk = self.Fk0.copy()
        st['Fk'] = Fk
        if self.p_fkc.n:
            self.p_fkc.run(Fk, st, w=cp.asarray(self.fkc.f('w')))
        if self.p_r1c.n:
            self.p_r1c.run(R1, st, w=cp.asarray(self.r1c.f('w')))
        self._R1_A1(R1, T)
        if self.p_r1a2.n:
            wd = self.p_r1a2_w.dots(st, self.p_r1a2.n)
            self.p_r1a2.run(R1, st, w=wd * cp.asarray(self.r1a2.f('w')))
        Uv = self.U0.copy()
        st['U'] = Uv
        if self.p_uq.n:
            self.p_uq.run(Uv, st)
        if self.p_r1b.n:
            self.p_r1b.run(R1, st, w=cp.asarray(self.r1b.f('w')))
        # weak pairs keep their amplitudes; only canonical strong residuals
        R = self._mask_strong(R)
        self.last = dict(Fpp=Fpp, beta=beta, Z=Z, Tn=Tn, Fv=Fv, Fk=Fk, U=Uv)
        return R1, R

    def _strong_of(self, c):
        if not hasattr(self, '_sof'):
            self._sof = {}
            for k in self.strong:
                cc = self.cmap[self.sp_of[self.sidx[k]]]
                self._sof.setdefault(cc, []).append(k)
        return self._sof.get(c, [])

    def _E_term(self, R, T, Fv):
        for (I, J) in self.strong:
            o = self.sidx[(I, J)]
            c = self.cmap[self.sp_of[o]]
            t = _get(T, o)
            fa = _get(Fv[self.spin[I]], self.fvpos[c])
            fb = _get(Fv[self.spin[J]], self.fvpos[c])
            M = t @ fb.T + fa @ t
            g, s_ = self.LT.group[o], self.LT.slot[o]
            R.data[g][s_, :M.shape[0], :M.shape[1]] += M

    def _Z_dress(self, Z, Tn, tn):
        """T1-dressing part of Z_JK from pair (j,k)'s fixed integrals and
        pair (k,l)'s exchange integrals projected (ref_uhf2._ring_Z)."""
        u, pl, lmo, sp = self.u, self.pl, self.lmo, self.spin
        for zi, (J, K, b, c) in enumerate(self.z_reg.keys):
            y, k = lmo[J], lmo[K]
            q = pl.ij[y, k]
            cq = self.cmap[q]
            fx = self.fix[cq]
            rows = self.rows[cq]
            sJ, sK = sp[J], sp[K]
            tJ = tn(q, sJ)[rows[sJ].index(J)]
            M = cp.zeros((int(self.nvs[q, b]), int(self.nvs[q, c])))
            if (b, c) == (sJ, sK):
                tau = tn(q, sJ).T                                  # (B, l) for l of spin sJ
                K1 = fx['K1'][(y, k, sK)][fx['rowsp'][sJ]]         # (l, C): (j l | k C)
                M -= tau @ K1
                K3 = fx['K3'][(k, sK, sJ)]                         # (C, B, d)
                M += cp.einsum('cbd,d->bc', K3, tJ)
                for r, L in enumerate(rows[sJ]):
                    kl = pl.ij[k, lmo[L]]
                    if kl < 0 or L == K:
                        continue
                    sm1 = _get(self.Sd[sK], self.Sreg[sK].ids[(q, kl)]) if (q, kl) in self.Sreg[sK].ids else None
                    sm2 = _get(self.Sd[sJ], self.Sreg[sJ].ids[(q, kl)]) if (q, kl) in self.Sreg[sJ].ids else None
                    if sm1 is None or sm2 is None:
                        continue
                    kx = self._kx(kl, K, L)
                    Kp = sm1 @ kx @ sm2.T                           # (K c|L d), c, d in q
                    M -= cp.outer(tau[:, r], Kp @ tJ)
            if sK == sJ and b == c:
                s = b
                K3 = fx['K3'][(k, sK, s)]                          # (d, B, C)
                M -= cp.einsum('dbc,d->bc', K3, tJ)
                tau_s = tn(q, s).T
                J1 = fx['J1'][s][fx['rowsp'][s]]
                M += tau_s @ J1
                for r, L in enumerate(rows[s]):
                    kl = pl.ij[k, lmo[L]]
                    if kl < 0 or L == K:
                        continue
                    sm1 = self._Sget(sK, q, kl)
                    sm2 = self._Sget(s, q, kl)
                    if sm1 is None or sm2 is None:
                        continue
                    Kp = sm1 @ self._kx(kl, K, L) @ sm2.T           # (K d|L C)
                    M += cp.outer(tau_s[:, r], tJ @ Kp)
            _set_items(Z, [zi], [M])

    def _Sget(self, s, p, q):
        i = self.Sreg[s].ids.get((p, q))
        if i is None:
            return None
        return _get(self.Sd[s], i)

    def _kx(self, kl, K, L):
        return self.fix[kl]['Kx'][(self.lmo[K], self.lmo[L], self.spin[K], self.spin[L])]

    def _R1_A1(self, R1, T):
        """sum_K S(ii,ik) sum_CD (k D|A C) t_IK^{CD} (3-external, pair ik)."""
        for (I, K, ik, it, sg) in self.r1a1:
            fx = self.fix[ik]
            t = _get(T, it) * sg
            v = cp.einsum('dac,cd->a', fx['K3'][(self.lmo[K], self.spin[K], self.spin[I])], t)
            sm = self._Sget(self.spin[I], self.dii[I], ik)
            v = sm @ v
            lay = R1.layout
            g, s_ = lay.group[I], lay.slot[I]
            R1.data[g][s_, :v.shape[0], 0] += v

    def _mask_strong(self, R):
        if getattr(self, '_smask', None) is None:
            self._smask = [cp.asarray(self.strong_mask[m].astype(np.float64))[:, None, None]
                           for m in R.layout.members]
        return Ragged(R.layout, [d * mk for d, mk in zip(R.data, self._smask)])

    # ------------------------------------------------------------------
    def energy(self, T, T1):
        st = self.stores(T, T1)
        Pt = self.p_pt.run(Ragged(self.LPt), st)
        E = Ew = 0.
        ef = 0.
        for I in range(self.nso):
            c = self.cmap[self.dii[I]]
            fx = self.fix[c]
            r = self.rows[c][self.spin[I]].index(I)
            ef += float(fx['fov'][self.spin[I]][r] @ _get(T1, I).ravel())
        for (I, J) in self.canon:
            o = self.sidx[(I, J)]
            p = self.sp_of[o]
            ti = _get(Pt, self.preg.ids[(I, p)]).ravel()
            tj = _get(Pt, self.preg.ids[(J, p)]).ravel()
            tau = _get(T, o) + cp.outer(ti, tj)
            mult = 1. if self.spin[I] == self.spin[J] else 2.
            if self.spin[I] == self.spin[J]:
                tau = tau - cp.outer(tj, ti)
            e = .5 * mult * float(cp.sum(_get(self.G, o) * tau))
            E += e
            if not self.strong_sp[p]:
                Ew += e
        return E + ef, Ew

    def kernel(self, verbose=True):
        u = self.u
        opt = u.opt
        T = Ragged(self.LT)
        if hasattr(u, 'T2b'):
            init = [cp.asarray(u.T2b[k]) for k in self.sop]
        else:
            init = [cp.asarray(u.T2[k][u._vslice(self.sp_of[n], self.spin[k[0]]),
                                       u._vslice(self.sp_of[n], self.spin[k[1]])])
                    for n, k in enumerate(self.sop)]
        _set_items(T, range(len(self.sop)), init)
        T1 = Ragged(self.LT1)
        diis = _DeviceDIIS(opt['DIIS_MAX_VECS'])
        e_prev = 0.
        ci = np.array([self.sidx[k] for k in self.canon], np.int64)
        for it in range(1, opt['MAXITER'] + 1):
            t0 = time.perf_counter()
            R1, R = self.residuals(T, T1)
            r1, r2 = _rnorm(R1, opt), _rnorm(R, opt)
            import os
            if os.environ.get('DLPNO_DEBUG_R1'):
                rr = R1.rms_per_item().get()
                top = np.argsort(rr)[::-1][:4]
                print('   r1 top:', [(int(I), self.so[I], bool(self.u.is_somo[self.so[I][0]]), f'{rr[I]:.2e}',
                                     f'D1min {float(cp.min(cp.abs(_get(self.D1, I)))):.3f}') for I in top], flush=True)
            T1 = T1 - R1 / self.D1
            T = T - R / self.D2
            T = self._symmetrize(T)
            n1 = T1.flat().size
            x = diis.update(cp.concatenate([T1.flat(), T.flat()]), cp.concatenate([R1.flat(), R.flat()]))
            T1.set_flat(x[:n1])
            T.set_flat(x[n1:])
            e, ew = self.energy(T, T1)
            if getattr(self, '_prof', None) is not None and it % 10 == 0:
                self._prof.report()
            if verbose:
                print(f'  GPU ULCCSD iter {it:3d} {e:.12f} {e - e_prev:.3e} {r1:.3e} {r2:.3e} '
                      f'{time.perf_counter() - t0:.2f}s', flush=True)
            if abs(e - e_prev) < opt['E_CONVERGENCE'] and max(r1, r2) < opt['R_CONVERGENCE']:
                break
            e_prev = e
        else:
            raise RuntimeError('ULCCSD did not converge')
        self.T, self.T1 = T, T1
        return e - ew, ew

    def _symmetrize(self, T):
        """T_JI = T_IJ^T from the canonical (I < J) items."""
        if not hasattr(self, '_sym'):
            src, dst = [], []
            for (I, J) in self.canon:
                src.append(self.sidx[(I, J)])
                dst.append(self.sidx[(J, I)])
            self._sym = (np.array(src), np.array(dst))
        src, dst = self._sym
        out = T.copy()
        for s_, d_ in zip(src, dst):
            M = _get(T, s_)
            g, sl = self.LT.group[d_], self.LT.slot[d_]
            out.data[g][sl] = 0.
            out.data[g][sl, :M.shape[1], :M.shape[0]] = M.T
        return out
