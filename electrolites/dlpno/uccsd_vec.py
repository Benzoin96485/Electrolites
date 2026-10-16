"""Vectorised per-iteration work of the open-shell GPU LCCSD.

``UCCSDGPUv`` keeps ``UCCSDGPU``'s term lists and plans and replaces its
per-pair Python loops by batched device work:

* T1 dressing of the strong pairs' DF tensors (Fbar, F'_vv, the dressed
  (Q|K z) / (Q|A z), the 2- and 4-external residual terms, the oooo
  ladder coefficients beta and the singles Fock F_ai of diagonal pairs):
  per group of strong canonical spatial pairs with equal padded sizes, a
  few batched GEMMs;
* Fbar_KJ, the F'_vv and T1-dressing parts of the ring intermediate and
  the singles A term: chain plans and one batched GEMM per stored
  3-external tensor group ("K3 applies");
* energies and the T_JI = T_IJ^T symmetrisation: item-wise reductions.

Every quantity is the one ``UCCSDGPU.residuals`` computes (validated to
round-off by ``test_uccsd_vec``).
"""
import numpy as np
import cupy as cp
import cupyx

from electrolites.dlpno.ragged import Layout, Ragged, padded, op
from electrolites.dlpno.vbatch import VChainPlan, VDotPlan
from electrolites.dlpno.uccsd_gpu import UCCSDGPU, _SpinPlan, _Terms, _Reg, _set_items, _get


def gather_like(src, items, layout):
    """Ragged in ``layout`` whose item i is src item items[i] (equal padded shapes)."""
    out = Ragged(layout)
    sl = src.layout
    items = np.asarray(items, np.int64)
    for g, m in enumerate(layout.members):
        if not len(m):
            continue
        it = items[m]
        sg = sl.group[it]
        for gs in np.unique(sg):
            sel = np.where(sg == gs)[0]
            blk = src.data[gs][cp.asarray(sl.slot[it[sel]])]
            pr, pc = layout.shapes[g]
            out.data[g][cp.asarray(sel)] = blk[:, :pr, :pc]
    return out


class _Entries:
    """gather scalars rag[item][r, c] for a fixed list (item, r, c)."""

    def __init__(self, layout, items, rr, cc):
        items = np.asarray(items, np.int64)
        self.n = len(items)
        g = layout.group[items]
        self.parts = []
        for gg in np.unique(g):
            sel = np.where(g == gg)[0]
            self.parts.append((int(gg), cp.asarray(sel), cp.asarray(layout.slot[items[sel]]),
                               cp.asarray(np.asarray(rr)[sel]), cp.asarray(np.asarray(cc)[sel])))

    def __call__(self, rag):
        out = cp.zeros(self.n)
        for g, sel, s, r, c in self.parts:
            out[sel] = rag.data[g][s, r, c]
        return out


def transpose_items(T, src, dst):
    """copy of T with T[dst] = T[src]^T."""
    lay = T.layout
    out = T.copy()
    combos = {}
    for s_, d_ in zip(src, dst):
        combos.setdefault((lay.group[s_], lay.group[d_]), []).append((lay.slot[s_], lay.slot[d_]))
    for (g1, g2), lst in combos.items():
        a = cp.asarray(np.array([l[0] for l in lst], np.int64))
        b = cp.asarray(np.array([l[1] for l in lst], np.int64))
        out.data[g2][b] = T.data[g1][a].transpose(0, 2, 1)
    return out


class _K3:
    """3-external tensors Y[b, c, d] = (Z c | b d) of the pairs, grouped by
    padded shape, and batched contractions with per-term vectors."""

    def __init__(self):
        self.reg = _Reg()
        self.ops = {}

    def add_op(self, kind, k3key, src, src_item, dst, dst_item, w=1.):
        """kind 'c3' : out[b, c] += w sum_d Y[b,c,d] v[d]      (v: column item)
           kind 'c2' : out[b, c] += w sum_d Y[b,d,c] v[d]
           kind 'a'  : out[a]    += w sum_dc Y[a,d,c] M[c,d]  (M: matrix item)"""
        self.ops.setdefault((kind, src, dst), []).append((self.reg(k3key), src_item, dst_item, w))

    def build(self, fetch):
        keys = self.reg.keys
        Ys = [fetch(k) for k in keys]
        shp = np.array([Y.shape for Y in Ys], np.int64).reshape(-1, 3)
        pads = np.stack([padded(shp[:, 0]), padded(shp[:, 1]), padded(shp[:, 2])], 1)
        uk, inv = np.unique(pads, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        self.group = inv
        self.slot = np.zeros(len(keys), np.int64)
        self.data = []
        for g in range(len(uk)):
            mem = np.where(inv == g)[0]
            self.slot[mem] = np.arange(len(mem))
            buf = cp.zeros((len(mem),) + tuple(int(x) for x in uk[g]))
            for s, k in enumerate(mem):
                Y = Ys[k]
                buf[s, :Y.shape[0], :Y.shape[1], :Y.shape[2]] = Y
            self.data.append(buf)
        del Ys
        self.plans = {}
        for key, lst in self.ops.items():
            ids = np.array([l[0] for l in lst], np.int64)
            per = []
            for g in np.unique(self.group[ids]):
                sel = np.where(self.group[ids] == g)[0]
                sl = self.slot[ids[sel]]
                # rank of each term among the terms of the same slot
                order = np.argsort(sl, kind='stable')
                rank = np.zeros(len(sel), np.int64)
                ss = sl[order]
                start = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
                for a, b in zip(start, np.r_[start[1:], len(ss)]):
                    rank[order[a:b]] = np.arange(b - a)
                per.append(dict(g=int(g), sel=sel, slot=sl, rank=rank, m=int(rank.max()) + 1,
                                src=np.array([lst[i][1] for i in sel], np.int64),
                                dst=np.array([lst[i][2] for i in sel], np.int64),
                                w=cp.asarray(np.array([lst[i][3] for i in sel]))))
            self.plans[key] = per

    def run(self, kind, src_name, dst_name, src, dst):
        """src, dst: Raggeds (column items, or matrix items for 'a' sources)."""
        for P in self.plans.get((kind, src_name, dst_name), []):
            Y = self.data[P['g']]
            N, d0, d1, d2 = Y.shape
            m = P['m']
            # vectors per (slot, rank)
            slay = src.layout
            if kind in ('c3', 'c2'):
                D = d2 if kind == 'c3' else d1
                Vb = cp.zeros((N, D, m))
                for gs in np.unique(slay.group[P['src']]):
                    q = np.where(slay.group[P['src']] == gs)[0]
                    v = src.data[gs][cp.asarray(slay.slot[P['src'][q]]), :D, 0]
                    Vb[cp.asarray(P['slot'][q]), :, cp.asarray(P['rank'][q])] = v
                if kind == 'c3':
                    res = cp.matmul(Y.reshape(N, d0 * d1, d2), Vb).reshape(N, d0, d1, m)
                    res = res.transpose(0, 3, 1, 2)                         # (N, m, b, c)
                else:
                    res = cp.matmul(Vb.transpose(0, 2, 1)[:, None], Y)    # (N, b, m, c)
                    res = res.transpose(0, 2, 1, 3)
            else:
                Vb = cp.zeros((N, d1 * d2, m))
                for gs in np.unique(slay.group[P['src']]):
                    q = np.where(slay.group[P['src']] == gs)[0]
                    M = src.data[gs][cp.asarray(slay.slot[P['src'][q]])]   # (n, c, d) padded
                    M = M[:, :d2, :d1].transpose(0, 2, 1).reshape(len(q), d1 * d2)
                    Vb[cp.asarray(P['slot'][q]), :, cp.asarray(P['rank'][q])] = M
                res = cp.matmul(Y.reshape(N, d0, d1 * d2), Vb)               # (N, a, m)
                res = res.transpose(0, 2, 1)[:, :, :, None]                  # (N, m, a, 1)
            dlay = dst.layout
            vals = res[cp.asarray(P['slot']), cp.asarray(P['rank'])] * P['w'][:, None, None]
            dg = dlay.group[P['dst']]
            for gd in np.unique(dg):
                q = np.where(dg == gd)[0]
                pr, pc = dlay.shapes[gd]
                blk = vals[cp.asarray(q)][:, :pr, :pc]
                cupyx.scatter_add(dst.data[gd], cp.asarray(dlay.slot[P['dst'][q]]), blk)
        return dst


class UCCSDGPUv(UCCSDGPU):

    MAX_DF_CHUNK_BYTES = 1 << 30

    def build(self):
        import time
        super().build()
        for name, fn in (('vec_terms', self._vec_terms), ('df_groups', self._df_groups)):
            t0 = time.perf_counter()
            fn()
            self.t_build[name] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    def _vec_terms(self):
        u, pl = self.u, self.pl
        lmo, sp, ij = self.lmo, self.spin, pl.ij
        S, tb = self.S, self.tb
        cs = self.canon_sp
        # ---- fixed J1 rows in the Tn layouts --------------------------------
        self.J1R = []
        for s in (0, 1):
            R = Ragged(self.LTn[s])
            items, mats = [], []
            for c in cs:
                fx = self.fix[c]
                if len(self.rows[c][s]):
                    items.append(self.cpos[c])
                    mats.append(fx['J1'][s][cp.asarray(np.asarray(fx['rowsp'][s], np.int64))])
            _set_items(R, items, mats)
            self.J1R.append(R)
        # ---- Fbar_KJ ----------------------------------------------------------
        keys = self.Freg.keys
        nk = len(keys)
        self.F0 = cp.asarray(np.array([(u.Fa_lmo, u.Fb_lmo)[sp[K]][lmo[K], lmo[J]] for (K, J) in keys]))
        cpk = np.array([self.cpos[self.cmap[ij[lmo[K], lmo[J]]]] for (K, J) in keys], np.int64)
        self.fj1_idx = cp.asarray(cpk)
        # K1 term: dot(Tn[s] item, K1[(j,k,s)][rowsp[s]])
        self.fk1 = []
        for s in (0, 1):
            sel = [n for n, (K, J) in enumerate(keys) if sp[K] == s and len(self.rows[self.cmap[ij[lmo[K], lmo[J]]]][s])]
            if not sel:
                self.fk1.append(None)
                continue
            its = cpk[sel]
            lay = Layout(self.LTn[s].rows[its], self.LTn[s].cols[its])
            R = Ragged(lay)
            mats = []
            for n in sel:
                K, J = keys[n]
                c = self.cmap[ij[lmo[K], lmo[J]]]
                fx = self.fix[c]
                mats.append(fx['K1'][(lmo[J], lmo[K], s)][cp.asarray(np.asarray(fx['rowsp'][s], np.int64))])
            _set_items(R, range(len(sel)), mats)
            self.fk1.append((cp.asarray(np.array(sel, np.int64)), its, lay, R))
        # Fbar_KC(diag pair of J) t_J: column per J, then entry rK
        self.fbt = []
        for s in (0, 1):
            Js = [J for J in range(self.nso) if sp[J] == s]
            cj = [self.cpos[self.cmap[self.dii[J]]] for J in Js]
            lay = Layout(self.LTn[s].rows[cj], 1)
            plan = VChainPlan(lay, np.arange(len(Js)), [op(self.LTn[s], cj), op(self.LT1, Js)])
            jpos = {J: r for r, J in enumerate(Js)}
            sel = [n for n, (K, J) in enumerate(keys) if sp[K] == s]
            rr = [self.rows[self.cmap[self.dii[keys[n][1]]]][s].index(keys[n][0]) for n in sel]
            ent = _Entries(lay, [jpos[keys[n][1]] for n in sel], rr, np.zeros(len(sel), np.int64))
            self.fbt.append((lay, plan, cp.asarray(np.array(sel, np.int64)), ent))
        # ---- E term: T Fv^T + Fv T ------------------------------------------
        # ---- ring Z dressing --------------------------------------------------
        self.k3 = k3 = _K3()
        zk1 = _Terms()
        zj1 = _Terms()
        zkx1 = _Terms()
        zkx2 = _Terms()
        k1z_items, k1z_mats = [], []
        for zi, (J, K, b, c) in enumerate(self.z_reg.keys):
            y, k = lmo[J], lmo[K]
            q = ij[y, k]
            cq = self.cmap[q]
            fx = self.fix[cq]
            rows = self.rows[cq]
            sJ, sK = sp[J], sp[K]
            pJ = self.preg((J, q))
            if (b, c) == (sJ, sK):
                if len(rows[sJ]):
                    k1z_items.append(zi)
                    k1z_mats.append(fx['K1'][(y, k, sK)][cp.asarray(np.asarray(fx['rowsp'][sJ], np.int64))])
                    zk1.add(z=zi, tn=self.cpos[cq], k1=len(k1z_items) - 1, sa=sJ)
                k3.add_op('c3', (cq, k, sK, sJ), 'Pt', pJ, 'Z', zi, 1.)
                for L in rows[sJ]:
                    kl = ij[k, lmo[L]]
                    if kl < 0 or L == K:
                        continue
                    zkx1.add(z=zi, pl_=self.preg((L, q)), pj=pJ, s1=S(q, kl, sJ),
                             kx=self.kx_reg((kl, K, L)), s2=S(q, kl, sK), a=sJ, b=sK, w=-1.)
            if sK == sJ and b == c:
                s = b
                k3.add_op('c2', (cq, k, sK, s), 'Pt', pJ, 'Z', zi, -1.)
                if len(rows[s]):
                    zj1.add(z=zi, tn=self.cpos[cq], sa=s)
                for L in rows[s]:
                    kl = ij[k, lmo[L]]
                    if kl < 0 or L == K:
                        continue
                    zkx2.add(z=zi, pl_=self.preg((L, q)), pj=pJ, s1=S(q, kl, sK),
                             kx=self.kx_reg((kl, K, L)), s2=S(q, kl, s), a=sK, b=s, w=1.)
        self.zk1_terms, self.zj1_terms = zk1, zj1
        self.zkx = (zkx1, zkx2)
        # singles A term (3-external)
        self.a1 = _Terms()
        for n, (I, K, ik, it, sg) in enumerate(self.r1a1):
            k3.add_op('a', (self.cmap[ik], lmo[K], sp[K], sp[I]), 'T', it, 'V1', n, sg)
            self.a1.add(i=I, s=S(self.dii[I], ik, sp[I]), v=n, a=sp[I])
        self.LV1 = Layout(np.array([self.nvs[ik, sp[I]] for (I, K, ik, it, sg) in self.r1a1], np.int64), 1)
        self._k1z = (k1z_items, k1z_mats)
        # ---- energy ---------------------------------------------------------
        ci = [self.sidx[k] for k in self.canon]
        self.e_ci = cp.asarray(np.array(ci, np.int64))
        en_t = _Terms()
        for n, (I, J) in enumerate(self.canon):
            p = self.sp_of[self.sidx[(I, J)]]
            en_t.add(n=n, pi=self.preg((I, p)), pj=self.preg((J, p)), o=self.sidx[(I, J)])
        self.en_t = en_t
        mult = np.array([1. if sp[I] == sp[J] else 2. for (I, J) in self.canon])
        same = np.array([1. if sp[I] == sp[J] else 0. for (I, J) in self.canon])
        weak = np.array([0. if self.strong_sp[self.sp_of[self.sidx[k]]] else 1. for k in self.canon])
        self.e_mult, self.e_same, self.e_weak = (cp.asarray(x) for x in (mult, same, weak))
        F1 = Ragged(self.LT1)
        mats = []
        for I in range(self.nso):
            c = self.cmap[self.dii[I]]
            mats.append(self.fix[c]['fov'][sp[I]][self.rows[c][sp[I]].index(I)])
        _set_items(F1, range(self.nso), mats)
        self.F1 = F1
        canon = self.canon
        self.sym_src = np.array([self.sidx[k] for k in canon], np.int64)
        self.sym_dst = np.array([self.sidx[(k[1], k[0])] for k in canon], np.int64)

    # ------------------------------------------------------------------
    def _df_groups(self):
        # registries may have grown in _vec_terms: rebuild the dependent stores
        self._rebuild_after_vec()
        u, pl = self.u, self.pl
        lmo, sp = self.lmo, self.spin
        sc = self.strong_c
        fx = self.fix
        naux = np.array([fx[c]['Qov'][0].shape[0] for c in sc], np.int64)
        Ap = -(-naux // 32) * 32
        g0 = self.LTn[0].group[[self.cpos[c] for c in sc]]
        g1 = self.LTn[1].group[[self.cpos[c] for c in sc]]
        key = np.stack([Ap, g0, g1], 1)
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        self.dfg = []
        bmap = {}
        for gi in range(len(uk)):
            A = int(uk[gi][0])
            R = [self.LTn[s].shapes[int(uk[gi][1 + s])][0] for s in (0, 1)]
            V = [self.LTn[s].shapes[int(uk[gi][1 + s])][1] for s in (0, 1)]
            per = 8 * A * (sum(V[s] * V[s] + 2 * R[s] * V[s] + V[s] * R[s] for s in (0, 1)) + 4 * max(V) ** 2)
            step = max(1, self.MAX_DF_CHUNK_BYTES // per)
            allm = np.array(sc)[inv == gi]
            for c0 in range(0, len(allm), step):
                self.dfg.append(self._df_group(allm[c0:c0 + step], A, R, V, bmap))
        # beta entries -> (group, combo, which matrix, member, rK, rL)
        self.beta_src = bmap

    def _rebuild_after_vec(self):
        """overlaps, Pt layout, Kx store and plans depend on registries that
        _vec_terms extended."""
        u = self.u
        self.LPt = Layout(np.array([self.nvs[p, self.spin[Z]] for (Z, p) in self.preg.keys], np.int64), 1)
        self.pt = _Terms()
        for r_, (Z, p) in enumerate(self.preg.keys):
            self.pt.add(r=r_, s=self.S(p, self.dii[Z], self.spin[Z]), z=Z, a=self.spin[Z])
        self._overlaps()
        mats = [self.fix[kl]['Kx'][(self.lmo[K], self.lmo[L], self.spin[K], self.spin[L])]
                for (kl, K, L) in self.kx_reg.keys]
        self.LKx = Layout(np.array([m.shape[0] for m in mats], np.int64),
                          np.array([m.shape[1] for m in mats], np.int64))
        self.Kx = Ragged(self.LKx)
        _set_items(self.Kx, range(len(mats)), mats)
        self._plans()
        lays = dict(S=self.LS, T=self.LT, Pt=self.LPt, Kx=self.LKx, Tn=self.LTn, J1R=self.LTn,
                    Fv=self.LF, V1=self.LV1)
        k1z_items, k1z_mats = self._k1z
        self.LK1Z = Layout(np.array([m.shape[0] for m in k1z_mats], np.int64),
                           np.array([m.shape[1] for m in k1z_mats], np.int64))
        self.K1Z = Ragged(self.LK1Z)
        _set_items(self.K1Z, range(len(k1z_mats)), k1z_mats)
        lays['K1Z'] = self.LK1Z
        SP = _SpinPlan
        self.p_e1 = SP(lays, self.LT, self._et(), 'o', [('T', 'o', None, False), ('Fv', 'f', 'b', True)])
        self.p_e2 = SP(lays, self.LT, self._et(), 'o', [('Fv', 'f', 'a', False), ('T', 'o', None, False)])
        self.p_zk1 = SP(lays, self.LZ, self.zk1_terms, 'z', [('Tn', 'tn', 'sa', True), ('K1Z', 'k1', None, False)])
        self.p_zj1 = SP(lays, self.LZ, self.zj1_terms, 'z', [('Tn', 'tn', 'sa', True), ('J1R', 'tn', 'sa', False)])
        self.p_zkx = []
        for n_, t_ in enumerate(self.zkx):
            tr2 = n_ == 0
            self.p_zkx.append(SP(lays, self.LZ, t_, 'z',
                                 [('Pt', 'pl_', None, False), ('Pt', 'pj', None, True), ('S', 's1', 'a', False),
                                  ('Kx', 'kx', None, tr2), ('S', 's2', 'b', True)]))
        self.p_a1 = SP(lays, self.LT1, self.a1, 'i', [('S', 's', 'a', False), ('V1', 'v', None, False)])
        en = self.en_t
        self.p_en1 = VDotPlan([op(self.LPt, en.a('pi'))], [op(self.LT, en.a('o')), op(self.LPt, en.a('pj'))])
        ss = np.where(self.e_same.get() > 0)[0]
        self.en2_sel = cp.asarray(ss)
        self.p_en2 = (VDotPlan([op(self.LPt, en.a('pj')[ss])], [op(self.LT, en.a('o')[ss]), op(self.LPt, en.a('pi')[ss])])
                      if len(ss) else None)
        # K3 tensors
        lm = self.lmo

        def fetch(key):
            c, z, s, t = key
            return self.fix[c]['K3'][(z, s, t)].transpose(1, 0, 2)
        self.k3.build(fetch)

    def _et(self):
        et = _Terms()
        for (I, J) in self.strong:
            o = self.sidx[(I, J)]
            et.add(o=o, f=self.fvpos[self.cmap[self.sp_of[o]]], a=self.spin[I], b=self.spin[J])
        return et

    def _df_group(self, mem, A, R, V, bmap):
        """padded DF tensors and index maps of one group of strong canonical pairs."""
        u, pl, fx = self.u, self.pl, self.fix
        lmo, sp, si = self.lmo, self.spin, u.so_index
        g = len(mem)
        d = dict(mem=mem, A=A, R=R, V=V)
        for s in (0, 1):
            Qov = cp.zeros((g, A, R[s], V[s]))
            Qvv = cp.zeros((g, A, V[s], V[s]))
            fov = cp.zeros((g, R[s], V[s]))
            ev = cp.zeros((g, V[s]))
            Qmz = cp.zeros((g, 2, A, R[s]))
            Qaz = cp.zeros((g, 2, A, V[s]))
            zrow = np.zeros((g, 2), np.int64)
            zm = np.zeros((g, 2))
            for m, c in enumerate(mem):
                f = fx[c]
                x, y = pl.pairs[c]
                na, nr = f['Qov'][s].shape[0], f['Qov'][s].shape[1]
                nv = f['Qvv'][s].shape[1]
                Qov[m, :na, :nr, :nv] = f['Qov'][s]
                Qvv[m, :na, :nv, :nv] = f['Qvv'][s]
                fov[m, :nr, :nv] = f['fov'][s]
                ev[m, :nv] = f['ev'][s]
                rp = cp.asarray(np.asarray(f['rowsp'][s], np.int64))
                for side, z in enumerate((x, y)):
                    if nr:
                        Qmz[m, side, :na, :nr] = f['Qmz'][z][:, rp]
                    if (z, s) in si:
                        Qaz[m, side, :na, :nv] = f['Qaz'][(z, s)]
                        zrow[m, side] = self.rows[c][s].index(si[(z, s)])
                        zm[m, side] = 1.
            d[('Qov', s)], d[('Qvv', s)], d[('fov', s)], d[('ev', s)] = Qov, Qvv, fov, ev
            d[('Qmz', s)], d[('Qaz', s)] = Qmz, Qaz
            d[('zrow', s)], d[('zm', s)] = cp.asarray(zrow), cp.asarray(zm)
            d[('tn', s)] = cp.asarray(self.LTn[s].slot[[self.cpos[c] for c in mem]])
            d[('gtn', s)] = int(self.LTn[s].group[self.cpos[mem[0]]])
            d[('fv', s)] = cp.asarray(self.LF[s].slot[[self.fvpos[c] for c in mem]])
            d[('gfv', s)] = int(self.LF[s].group[self.fvpos[mem[0]]])
        # strong spin-orbital pairs by combo (side_i, a, side_j, b)
        combos = {}
        for m, c in enumerate(mem):
            x, y = pl.pairs[c]
            for (I, J) in self._strong_of(c):
                o = self.sidx[(I, J)]
                key = (0 if lmo[I] == x else 1, sp[I], 0 if lmo[J] == x and x != y else (1 if x != y else 0), sp[J])
                if x == y:
                    key = (0, sp[I], 0, sp[J])
                combos.setdefault(key, []).append((m, o, I, J))
        d['combos'] = []
        for key, lst in combos.items():
            si_, a, sj_, b = key
            mm = np.array([l[0] for l in lst], np.int64)
            oo = np.array([l[1] for l in lst], np.int64)
            gT = self.LT.group[oo]
            assert np.all(gT == gT[0])
            cid = len(d['combos'])
            d['combos'].append(dict(key=key, m=cp.asarray(mm), o=oo, gT=int(gT[0]),
                                    sT=cp.asarray(self.LT.slot[oo])))
            # beta entries of these pairs
            for (m, o, I, J), mh in zip(lst, mm):
                c = mem[m]
                for (n, K, L) in self._bkeys().get((I, J), []):
                    sK, sL = sp[K], sp[L]
                    rK = self.rows[c][sK].index(K)
                    rL = self.rows[c][sL].index(L)
                    which = 0 if (sK, sL) == (a, b) else 1
                    if a == b:
                        assert (sK, sL) == (a, a)
                        which = 0
                    bmap.setdefault((len(self.dfg), cid, which), []).append((n, m, rK, rL))
        # diagonal pairs: singles Fock
        diag = []
        for m, c in enumerate(mem):
            x, y = pl.pairs[c]
            if x != y:
                continue
            for s in (0, 1):
                X = si.get((x, s))
                if X is not None:
                    diag.append((m, s, X, self.rows[c][s].index(X),
                                 [lmo[K] for K in self.rows[c][s]]))
        d['diag'] = []
        for s in (0, 1):
            ds = [t for t in diag if t[1] == s]
            if not ds:
                continue
            mm = np.array([t[0] for t in ds], np.int64)
            Xs = [t[2] for t in ds]
            foo = np.zeros((len(ds), R[s]))
            for r, t in enumerate(ds):
                F = (u.Fa_lmo, u.Fb_lmo)[s]
                foo[r, :len(t[4])] = F[t[4], lmo[t[2]]]
            gX = self.LT1.group[Xs]
            d['diag'].append(dict(s=s, m=cp.asarray(mm), X=np.array(Xs), rX=cp.asarray(np.array([t[3] for t in ds])),
                                  foo=cp.asarray(foo), gX=gX, sX=self.LT1.slot[Xs]))
        return d

    def _bkeys(self):
        if not hasattr(self, '_bk'):
            self._bk = {}
            for n, (I, J, K, L) in enumerate(self.breg.keys):
                self._bk.setdefault((I, J), []).append((n, K, L))
        return self._bk

    # ------------------------------------------------------------------
    def residuals(self, T, T1):
        sp = self.spin
        st = self.stores(T, T1)
        Tn = [self.p_tn[s].run(Ragged(self.LTn[s]), st) for s in (0, 1)]
        st['Tn'] = Tn
        st['J1R'] = self.J1R
        Pt = self.p_pt.run(Ragged(self.LPt), st)
        st['Pt'] = Pt
        Fv = [Ragged(self.LF[s]) for s in (0, 1)]
        Fbov = [Ragged(self.LTn[s]) for s in (0, 1)]
        R = Ragged(self.LT)
        R1 = Ragged(self.LT1)
        beta = cp.zeros(len(self.breg))
        for gi, d in enumerate(self.dfg):
            self._df_run(gi, d, T, T1, Tn, Fv, Fbov, R, R1, beta)
        st['Fv'] = Fv
        # F'_vv projections
        for s in (0, 1):
            if self.p_fv[s].n:
                self.p_fv[s].run(Fv[s], st, w=cp.full(self.p_fv[s].n, -.5))
        # Fbar_KJ, F'_KJ
        Fpp = self.F0.copy()
        for s in (0, 1):
            Fpp += Tn[s].vdot_items(self.J1R[s])[self.fj1_idx]
            if self.fk1[s] is not None:
                sel, its, lay, K1 = self.fk1[s]
                Fpp[sel] -= gather_like(Tn[s], its, lay).vdot_items(K1)
            lay, plan, sel, ent = self.fbt[s]
            col = plan.run(Ragged(lay), [Fbov[s], T1])
            Fpp[sel] += ent(col)
        if self.p_fp.n:
            dots = self.p_fp.dots(st, self.p_fp.n)
            cupyx.scatter_add(Fpp, self.fp_f, dots * cp.asarray(self.fp.f('w')))
        # R2 projection terms
        if self.p_e1.n:
            self.p_e1.run(R, st)
            self.p_e2.run(R, st)
        if self.p_rg.n:
            self.p_rg.run(R, st, w=Fpp[self.rg_f] * cp.asarray(self.rg.f('w')))
        if self.p_rb.n:
            self.p_rb.run(R, st, w=beta[self.rb_beta] * cp.asarray(self.rb.f('w')))
        # ring
        Z = Ragged(self.LZ)
        st['K1Z'] = self.K1Z
        self.p_zk1.run(Z, st, w=cp.full(self.p_zk1.n, -1.) if self.p_zk1.n else None)
        self.p_zj1.run(Z, st)
        for p_, t_ in zip(self.p_zkx, self.zkx):
            if p_.n:
                p_.run(Z, st, w=cp.asarray(t_.f('w')))
        self.k3.run('c3', 'Pt', 'Z', Pt, Z)
        self.k3.run('c2', 'Pt', 'Z', Pt, Z)
        if self.p_zq.n:
            self.p_zq.run(Z, st, w=cp.asarray(self.zq.f('w')))
        st['Z'] = Z
        for tr in (0, 1):
            if self.p_ring_np[tr].n:
                self.p_ring_np[tr].run(R, st, w=cp.asarray(self.ring_np[tr].f('w')))
            if self.p_ring_z[tr].n:
                self.p_ring_z[tr].run(R, st, w=cp.asarray(self.ring_z[tr].f('w')))
        # R1
        Fk = self.Fk0.copy()
        st['Fk'] = Fk
        if self.p_fkc.n:
            self.p_fkc.run(Fk, st, w=cp.asarray(self.fkc.f('w')))
        if self.p_r1c.n:
            self.p_r1c.run(R1, st, w=cp.asarray(self.r1c.f('w')))
        V1 = self.k3.run('a', 'T', 'V1', T, Ragged(self.LV1))
        st['V1'] = V1
        if self.p_a1.n:
            self.p_a1.run(R1, st)
        if self.p_r1a2.n:
            wd = self.p_r1a2_w.dots(st, self.p_r1a2.n)
            self.p_r1a2.run(R1, st, w=wd * cp.asarray(self.r1a2.f('w')))
        Uv = self.U0.copy()
        st['U'] = Uv
        if self.p_uq.n:
            self.p_uq.run(Uv, st)
        if self.p_r1b.n:
            self.p_r1b.run(R1, st, w=cp.asarray(self.r1b.f('w')))
        R = self._mask_strong(R)
        self.last = dict(Fpp=Fpp, beta=beta, Z=Z, Tn=Tn, Fv=Fv, Fk=Fk, U=Uv)
        return R1, R

    def _df_run(self, gi, d, T, T1, Tn, Fv, Fbov, R, R1, beta):
        mm = cp.matmul
        tr = lambda X: X.swapaxes(-1, -2)
        Tc, Qov, Qvv, Fbo, Fbv, Qvt, Bk, Ba = {}, {}, {}, {}, {}, {}, {}, {}
        gam = 0.
        for s in (0, 1):
            Tc[s] = Tn[s].data[d[('gtn', s)]][d[('tn', s)]]
            Qov[s], Qvv[s] = d[('Qov', s)], d[('Qvv', s)]
            gam = gam + cp.einsum('gAme,gme->gA', Qov[s], Tc[s])
        for s in (0, 1):
            tc = Tc[s][:, None]
            X = mm(Qvv[s], tr(tc))                                       # (g, A, V, R)
            fbv = cp.einsum('gA,gAab->gab', gam, Qvv[s]) - mm(X, Qov[s]).sum(axis=1)
            ev = d[('ev', s)]
            fbv = fbv + ev[:, :, None] * cp.eye(ev.shape[1])[None]
            Y = mm(Qov[s], tr(tc))                                       # (g, A, R, R)
            fbo = d[('fov', s)] + cp.einsum('gA,gAkc->gkc', gam, Qov[s]) - mm(Y, Qov[s]).sum(axis=1)
            Fbv[s], Fbo[s] = fbv, fbo
            Fv[s].data[d[('gfv', s)]][d[('fv', s)]] = fbv - mm(tr(Tc[s]), fbo)
            Fbov[s].data[d[('gtn', s)]][d[('tn', s)]] = fbo
            Qvt[s] = Qvv[s] - mm(tr(tc), Qov[s])
            # dressed (Q|K z) and (Q|A z) for z = x, y
            g = Tc[s].shape[0]
            zrow, zm = d[('zrow', s)], d[('zm', s)]
            ar = cp.arange(g)
            for side in (0, 1):
                tz = Tc[s][ar, zrow[:, side]] * zm[:, side][:, None]       # (g, V)
                bk = d[('Qmz', s)][:, side] + cp.einsum('gAkc,gc->gAk', Qov[s], tz)
                Bk[(side, s)] = bk
                Ba[(side, s)] = (d[('Qaz', s)][:, side] - mm(bk, Tc[s])
                                 + cp.einsum('gAab,gb->gAa', Qvv[s], tz)) * zm[:, side][:, None, None]
        # residual DF terms and beta of the strong spin-orbital pairs
        for cid, cb in enumerate(d['combos']):
            si_, a, sj_, b = cb['key']
            m = cb['m']
            t = T.data[cb['gT']][cb['sT']]
            ba, bb = Ba[(si_, a)][m], Ba[(sj_, b)][m]
            M = mm(tr(ba), bb)
            if a == b:
                M = M - mm(tr(Ba[(sj_, a)][m]), Ba[(si_, a)][m])
            M = M + mm(mm(Qvt[a][m], t[:, None]), tr(Qvt[b][m])).sum(axis=1)
            R.data[cb['gT']][cb['sT']] += M
            # beta matrices
            W1 = mm(tr(Bk[(si_, a)][m]), Bk[(sj_, b)][m])                  # (n, Ra, Rb)
            W2 = mm(tr(Bk[(sj_, b)][m]), Bk[(si_, a)][m])                  # (n, Rb, Ra)
            Q1 = mm(mm(Qov[a][m], t[:, None]), tr(Qov[b][m])).sum(axis=1)
            mats = {0: W1 + Q1 - (W2 if a == b else 0.)}
            if a != b:
                Q2 = -mm(mm(Qov[b][m], tr(t)[:, None]), tr(Qov[a][m])).sum(axis=1)
                mats[1] = -W2 + Q2
            for which, Mx in mats.items():
                lst = self.beta_src.get((gi, cid, which))
                if not lst:
                    continue
                if 'bidx' not in cb:
                    cb['bidx'] = {}
                if which not in cb['bidx']:
                    arr = np.array(lst, np.int64)
                    # member index -> position within this combo
                    pos = {int(x): r for r, x in enumerate(m.get())}
                    cb['bidx'][which] = (cp.asarray(arr[:, 0]),
                                         cp.asarray(np.array([pos[x] for x in arr[:, 1]], np.int64)),
                                         cp.asarray(arr[:, 2]), cp.asarray(arr[:, 3]))
                n_, r_, k_, l_ = cb['bidx'][which]
                beta[n_] = Mx[r_, k_, l_]
        # singles Fock of the diagonal pairs
        for dg in d['diag']:
            s, m = dg['s'], dg['m']
            rX = dg['rX']
            ar = cp.arange(len(dg['X']))
            tx = Tc[s][m, rX]                                            # (n, V)
            Qmx = d[('Qmz', s)][m, 0]                                     # (n, A, R)
            Qax = d[('Qaz', s)][m, 0]                                     # (n, A, V)
            fvo = d[('fov', s)][m, rX]
            Fb_ai = (fvo + cp.einsum('gA,gAa->ga', gam[m], Qax)
                     - cp.einsum('gAae,gme,gAm->ga', Qvv[s][m], Tc[s][m], Qmx))
            Fb_oi = (dg['foo'] + cp.einsum('gA,gAk->gk', gam[m], Qmx)
                     - cp.einsum('gAke,gme,gAm->gk', Qov[s][m], Tc[s][m], Qmx))
            fai = (Fb_ai + cp.einsum('gab,gb->ga', Fbv[s][m], tx)
                   - cp.einsum('gka,gk->ga', Tc[s][m], Fb_oi)
                   - cp.einsum('gka,gk->ga', Tc[s][m], cp.einsum('gkb,gb->gk', Fbo[s][m], tx)))
            for gX in np.unique(dg['gX']):
                q = np.where(dg['gX'] == gX)[0]
                pr = self.LT1.shapes[gX][0]
                R1.data[gX][cp.asarray(dg['sX'][q]), :, 0] = fai[cp.asarray(q), :pr]

    # ------------------------------------------------------------------
    def energy(self, T, T1):
        st = self.stores(T, T1)
        Pt = self.p_pt.run(Ragged(self.LPt), st)
        ef = float(T1.vdot_items(self.F1).sum())
        gt = self.G.vdot_items(T)[self.e_ci]
        d1 = self.p_en1.run([Pt], [self.G, Pt])
        d2 = cp.zeros(len(gt))
        if self.p_en2 is not None:
            d2[self.en2_sel] = self.p_en2.run([Pt], [self.G, Pt])
        e = .5 * self.e_mult * (gt + d1 - d2)
        return float(e.sum()) + ef, float((e * self.e_weak).sum())

    def _symmetrize(self, T):
        return transpose_items(T, self.sym_src, self.sym_dst)
