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
import os

import numpy as np
import cupy as cp
import cupyx

from electrolites.dlpno.ragged import Layout, Ragged, padded, op
from electrolites.dlpno.vbatch import VChainPlan, VDotPlan
from electrolites.dlpno.ccsd_gpu import _mv, _dot2, _wsum, _Packed
from electrolites.dlpno.dfints import _host_available
from electrolites.dlpno.devmem import as_host
from electrolites.dlpno.uccsd_gpu import UCCSDGPU, _SpinPlan, _Terms, _Reg, _set_items, _get, _build_report


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


def _to_dev_retry(to_dev, x):
    """to_dev; on an out-of-memory error the pool's cached free blocks are
    released and the copy retried once (a multi-GB group needs one contiguous
    block; the pool's free blocks are fragmented after the overlap chunks)."""
    try:
        return to_dev(x)
    except cp.cuda.memory.OutOfMemoryError:
        cp.get_default_memory_pool().free_all_blocks()
        return to_dev(x)


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

    def build(self, fetch, host=False, shape=None, release=None, store=None):
        """fetch(key) -> Y; shape(key) -> Y.shape without fetching; release(key)
        drops the source once its group is assembled; store: the padded
        groups go to a file on scratch (read back at every use)."""
        from electrolites.dlpno.devmem import as_host
        keys = self.reg.keys
        shp = np.array([shape(k) if shape is not None else fetch(k).shape for k in keys],
                       np.int64).reshape(-1, 3)
        pads = np.stack([padded(shp[:, 0]), padded(shp[:, 1]), padded(shp[:, 2])], 1)
        uk0, inv0 = np.unique(pads, axis=0, return_inverse=True)
        inv0 = inv0.reshape(-1)
        # each shape group split into pieces of at most DLPNO_K3_GROUP_MB (a
        # piece is brought to the device whole at every use: a 9.5 GB group
        # found no room in the 309-atom quintet's second iteration)
        import os
        lim = float(os.environ.get('DLPNO_K3_GROUP_MB', '2048')) * 2 ** 20
        uk, pieces = [], []
        for g0 in range(len(uk0)):
            mem0 = np.where(inv0 == g0)[0]
            per = max(1, int(lim // (8. * np.prod(uk0[g0]))))
            for a in range(0, len(mem0), per):
                uk.append(uk0[g0])
                pieces.append(mem0[a:a + per])
        inv = np.zeros(len(keys), np.int64)
        for g, mem in enumerate(pieces):
            inv[mem] = g
        self.group = inv
        self.slot = np.zeros(len(keys), np.int64)
        self.data = []
        for g in range(len(uk)):
            mem = pieces[g]
            self.slot[mem] = np.arange(len(mem))
            gshape = (len(mem),) + tuple(int(x) for x in uk[g])
            if store is not None:
                buf = np.zeros(gshape)
            elif host:
                from electrolites.dlpno.devmem import pinned_empty
                buf = pinned_empty(gshape)
                buf[...] = 0.
            else:
                buf = cp.zeros(gshape)
            for s, k in enumerate(mem):
                Y = fetch(keys[k])
                Y = as_host(Y) if isinstance(buf, np.ndarray) else cp.asarray(Y)
                buf[s, :Y.shape[0], :Y.shape[1], :Y.shape[2]] = Y
                del Y
            if store is not None:
                buf = store.put(buf)
            self.data.append(buf)
            del buf
            if release is not None:
                for k in mem:
                    release(keys[k])
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

    def run_all(self, jobs):
        """jobs: (kind, src_name, dst_name, src, dst); every tensor group is
        brought to the device once (it may wait in pinned host memory)."""
        from electrolites.dlpno.devmem import to_dev, SpillRef, as_host
        need = [g for g in range(len(self.data))
                if any(P['g'] == g for (kind, sn, dn, src, dst) in jobs for P in self.plans.get((kind, sn, dn), []))]
        # groups on scratch: the next one read by a helper thread while the
        # current one is applied
        ex = fut = None
        if any(isinstance(self.data[g], SpillRef) for g in need):
            from concurrent.futures import ThreadPoolExecutor
            ex = ThreadPoolExecutor(1)
        for n_, g in enumerate(need):
            if ex is not None:
                h = fut.result() if fut is not None else as_host(self.data[g]) \
                    if isinstance(self.data[g], SpillRef) else self.data[g]
                fut = None
                if n_ + 1 < len(need) and isinstance(self.data[need[n_ + 1]], SpillRef):
                    fut = ex.submit(self.data[need[n_ + 1]].load)
                Y = _to_dev_retry(to_dev, h)
                del h
            else:
                Y = _to_dev_retry(to_dev, self.data[g])
            for (kind, sn, dn, src, dst) in jobs:
                for P in self.plans.get((kind, sn, dn), []):
                    if P['g'] == g:
                        self._apply(kind, P, Y, src, dst)
            del Y
        if ex is not None:
            ex.shutdown()

    def run(self, kind, src_name, dst_name, src, dst):
        self.run_all([(kind, src_name, dst_name, src, dst)])
        return dst

    @staticmethod
    def _index(P, src, dst):
        """device index arrays of a plan for its (fixed) source / target layouts."""
        key = (id(src.layout), id(dst.layout))
        c = P.get('_dev')
        if c is not None and c[0] == key:
            return c[1]
        slay, dlay = src.layout, dst.layout
        sg = slay.group[P['src']]
        srcs = []
        for gs in np.unique(sg):
            q = np.where(sg == gs)[0]
            srcs.append((int(gs), cp.asarray(slay.slot[P['src'][q]]), cp.asarray(P['slot'][q]),
                         cp.asarray(P['rank'][q]), len(q)))
        dg = dlay.group[P['dst']]
        dsts = []
        for gd in np.unique(dg):
            q = np.where(dg == gd)[0]
            dsts.append((int(gd), cp.asarray(q), cp.asarray(dlay.slot[P['dst'][q]])))
        ix = dict(srcs=srcs, dsts=dsts, slot=cp.asarray(P['slot']), rank=cp.asarray(P['rank']))
        P['_dev'] = (key, ix)
        return ix

    def _apply(self, kind, P, Y, src, dst):
        ix = self._index(P, src, dst)
        N, d0, d1, d2 = Y.shape
        m = P['m']
        if kind in ('c3', 'c2'):
            D = d2 if kind == 'c3' else d1
            Vb = cp.zeros((N, D, m))
            for gs, ss, sl, rk, _ in ix['srcs']:
                Vb[sl, :, rk] = src.data[gs][ss, :D, 0]
            if kind == 'c3':
                res = cp.matmul(Y.reshape(N, d0 * d1, d2), Vb).reshape(N, d0, d1, m)
                res = res.transpose(0, 3, 1, 2)                         # (N, m, b, c)
            else:
                res = cp.matmul(Vb.transpose(0, 2, 1)[:, None], Y)    # (N, b, m, c)
                res = res.transpose(0, 2, 1, 3)
        else:
            Vb = cp.zeros((N, d1 * d2, m))
            for gs, ss, sl, rk, nq in ix['srcs']:
                M = src.data[gs][ss]                                    # (n, c, d) padded
                Vb[sl, :, rk] = M[:, :d2, :d1].transpose(0, 2, 1).reshape(nq, d1 * d2)
            res = cp.matmul(Y.reshape(N, d0, d1 * d2), Vb)               # (N, a, m)
            res = res.transpose(0, 2, 1)[:, :, :, None]                  # (N, m, a, 1)
        vals = res[ix['slot'], ix['rank']] * P['w'][:, None, None]
        for gd, q, ds in ix['dsts']:
            pr, pc = dst.layout.shapes[gd]
            cupyx.scatter_add(dst.data[gd], ds, vals[q][:, :pr, :pc])


class ChunkedSpinPlan:
    """A _SpinPlan whose overlap operands ('S') are not stored: its terms
    are split into chunks (by output item) and each chunk's overlaps are
    generated on the device right before the chunk runs (ovgen)."""

    MAX_BYTES = 1 << 29

    def __init__(self, owner, layouts, out_layout, terms, out_col, specs, out_row=None,
                 transpose_out=False, dot_left=None):
        self.n = n = len(terms)
        self.chunks = []
        if n == 0:
            return
        nvs = owner.nvs
        keys = [owner.Sreg[0].keys, owner.Sreg[1].keys]
        kp = [np.array([k[0] for k in keys[s]], np.int64) for s in (0, 1)]
        kq = [np.array([k[1] for k in keys[s]], np.int64) for s in (0, 1)]
        scols = [(c, sc) for (kind, c, sc, tr) in specs if kind == 'S']
        # per term global couple codes (spin * BIG + index)
        BIG = 1 << 40
        codes = []
        for c, sc in scols:
            sp = terms.a(sc) if isinstance(sc, str) else np.full(n, sc)
            codes.append(sp * BIG + terms.a(c))
        codes = np.stack(codes, 1)
        order = np.argsort(terms.a(out_col), kind='stable') if out_col is not None else np.arange(n)
        # bytes of every couple code
        def nbytes_arr(cd):
            s_, i = cd // BIG, cd % BIG
            out = np.empty(len(cd))
            for sv in (0, 1):
                m = s_ == sv
                if m.any():
                    out[m] = 8. * nvs[kp[sv][i[m]], sv] * nvs[kq[sv][i[m]], sv]
            return out
        # greedy chunks whose distinct overlaps stay within MAX_BYTES
        # (vectorised: first uses counted within a window from the chunk start)
        bounds, start = [], 0
        k = codes.shape[1]
        W0 = 4096
        while start < n:
            W = min(n - start, W0)
            ids = codes[order[start:start + W]].ravel()
            _, first = np.unique(ids, return_index=True)
            add = np.zeros(W * k)
            add[first] = nbytes_arr(ids[first])
            cum = np.cumsum(add.reshape(W, k).sum(axis=1))
            over = np.flatnonzero(cum > self.MAX_BYTES)
            if not len(over) and W < n - start:
                W0 *= 2
                continue
            L = max(1, int(over[0])) if len(over) else W
            bounds.append((start, start + L))
            start += L
        gens = owner._ovgens
        colarr = {k: np.asarray(v) for k, v in terms._mat().items()}
        for (a, b) in bounds:
            sel = order[a:b]
            loc_terms = _Terms()
            sub = {k: v[sel] for k, v in colarr.items()}
            lays = dict(layouts)
            LSc, preps = [], []
            for s in (0, 1):
                ids = []
                for j, (c, sc) in enumerate(scols):
                    msk = (codes[sel, j] // BIG) == s
                    ids.append(codes[sel, j][msk] % BIG)
                uni = np.unique(np.concatenate(ids)) if ids else np.zeros(0, np.int64)
                LSc.append(Layout(nvs[kp[s][uni], s], nvs[kq[s][uni], s]))
                stores = getattr(owner, '_sstores', None)
                if stores is not None:
                    preps.append(stores[s].tables(LSc[s], uni) if len(uni) else [])
                else:
                    preps.append(gens[s].prepare(kp[s][uni], kq[s][uni], out_layout=LSc[s],
                                                 out_idx=np.arange(len(uni))) if len(uni) else [])
                for j, (c, sc) in enumerate(scols):
                    msk = (codes[sel, j] // BIG) == s
                    loc = np.searchsorted(uni, codes[sel, j][msk] % BIG)
                    col = np.array(sub[c], dtype=np.int64)
                    col[msk] = loc
                    sub[c] = col
            loc_terms._keys = tuple(sub)
            loc_terms._cols = dict(sub)
            loc_terms._rows = None
            lays['S'] = LSc
            plan = _SpinPlan(lays, out_layout, loc_terms, out_col, specs, out_row=out_row,
                             transpose_out=transpose_out, dot_left=dot_left)
            self.chunks.append(dict(sel=cp.asarray(sel), LS=LSc, preps=preps, plan=plan, nt=len(sel)))
        self.gens = gens
        self.budget = getattr(owner, '_s_budget', None)
        self.stores = getattr(owner, '_sstores', None)

    def _S(self, ch):
        if ch.get('S') is not None:
            return ch['S']
        if self.stores is not None:
            from electrolites.dlpno.ccsd_gpu import _SStore
            return [_SStore.gather(ch['LS'][s], ch['preps'][s]) if len(ch['preps'][s]) else Ragged(ch['LS'][s])
                    for s in (0, 1)]
        S = []
        for s in (0, 1):
            R = Ragged(ch['LS'][s])
            for P in ch['preps'][s]:
                self.gens[s].run(P, R)
            S.append(R)
        bud = getattr(self, 'budget', None)
        nb = sum(L.nbytes() for L in ch['LS'])
        # kept only while DLPNO_S_KEEP_FREE of the device stays free (as the
        # closed-shell plans): the budget was set before the plans and stores
        fr, tot = cp.cuda.runtime.memGetInfo()
        if bud is not None and fr - nb > float(os.environ.get('DLPNO_S_KEEP_FREE', '0.4')) * tot \
                and bud.take(cp.cuda.Device().id, nb):
            ch['S'] = S                   # kept: not regenerated in later iterations
        return S

    def run(self, out, stores, w=None):
        for ch in self.chunks:
            st = dict(stores)
            st['S'] = self._S(ch)
            ch['plan'].run(out, st, w=None if w is None else w[ch['sel']])
        return out

    def dots(self, stores, n):
        out = cp.zeros(n)
        for ch in self.chunks:
            st = dict(stores)
            st['S'] = self._S(ch)
            out[ch['sel']] = ch['plan'].dots(st, ch['nt'])
        return out


class UCCSDGPUv(UCCSDGPU):

    KL_NUMPY = True

    MAX_DF_CHUNK_BYTES = 1 << 30

    def build(self):
        import time
        # the base plans are built once, after the registries are final
        self._plans_base, self._plans = self._plans, self._layouts
        super().build()
        self._plans = self._plans_base
        if hasattr(self.u, 'park_ints'):
            self.u.park_ints()
        for name, fn in (('vec_terms', self._vec_terms), ('df_groups', self._df_groups)):
            t0 = time.perf_counter()
            fn()
            self.t_build[name] = time.perf_counter() - t0
            _build_report(self, name)
        # the DF tensors now live (padded) in the DF groups, the 3-external
        # tensors in the K3 store: drop the per-pair copies
        for c in self.canon_sp:
            for key in ('Qov', 'Qvv', 'Qmz', 'Qaz', 'K3'):
                self.fix[c].pop(key, None)
        if getattr(self.u, '_pair_spill', None) is not None:
            self.u._pair_spill.close()           # the per-pair blocks are assembled
            self.u._pair_spill = None
        cp.get_default_memory_pool().free_all_blocks()
        # term lists are only needed through their plans now
        lists = [self.fp, self.rg, self.zq, self.fkc, self.r1c, self.r1a2, self.uq, self.r1b, self.a1,
                 *self.ring_np, *self.ring_z, *self.zkx, *self.tn, self.zk1_terms, self.zj1_terms]
        for t_ in lists:
            t_.compact()
        for d in self.klchunks:
            for t_ in (*d['fv'], d['rb']):
                t_.compact()
        self._fvnp = self._rbnp = None
        self.mem_report = self._mem_report()

    def _mem_report(self):
        """device bytes held by the main stores (GB)."""
        def rag(r):
            return sum(d.nbytes for d in r.data) if r is not None else 0
        rep = {}
        fixb = {}
        for c in self.canon_sp:
            for k, v in self.fix[c].items():
                if isinstance(v, dict):
                    b = sum(x.nbytes for x in v.values() if hasattr(x, 'nbytes'))
                elif isinstance(v, list):
                    b = sum(x.nbytes for x in v if hasattr(x, 'nbytes'))
                else:
                    b = getattr(v, 'nbytes', 0)
                fixb[k] = fixb.get(k, 0) + b
        rep.update({'fix.' + k: v / 1e9 for k, v in fixb.items()})
        rep['S'] = sum(rag(r) for r in self.Sd) / 1e9
        rep['NP'] = rag(self.NP) / 1e9
        rep['Kx'] = rag(self.Kx) / 1e9
        rep['T'] = rag(Ragged(self.LT)) / 1e9 if False else self.LT.nbytes() / 1e9
        rep['K3'] = sum(d.nbytes for d in self.k3.data if isinstance(d, cp.ndarray)) / 1e9
        rep['dfg'] = sum(v.nbytes for d in self.dfg for v in d.values() if isinstance(v, (cp.ndarray, _Packed))) / 1e9
        rep['offload'] = float(self.offload)
        rep['pool_used'] = cp.get_default_memory_pool().used_bytes() / 1e9
        return {k: round(v, 3) for k, v in rep.items()}

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
        self._split_kl()

    # ------------------------------------------------------------------
    # overlaps S(kl, c) of the F'_vv and oooo-ladder terms: generated per
    # chunk of strong pairs every iteration instead of being stored
    # ------------------------------------------------------------------
    MAX_S_CHUNK_BYTES = 1 << 31

    def _np_kl_terms(self):
        """F'_vv rows (c, spin, kl, item) and oooo-ladder rows
        (o, K, L, kl, item, sign) as arrays."""
        u, pl = self.u, self.pl
        lmo, sp, ij = self.lmo, self.spin, pl.ij
        nso = self.nso
        SIDX = -np.ones((nso, nso), np.int64)
        for n, (I, J) in enumerate(self.sop):
            SIDX[I, J] = n
        fv = []
        for r, c in enumerate(self.strong_c):
            ds = np.array(u.dom_so[c], np.int64)
            K, L = np.meshgrid(ds, ds, indexing='ij')
            K, L = K.ravel(), L.ravel()
            kl = ij[lmo[K], lmo[L]]
            ok = (K != L) & (kl >= 0)
            K, L, kl = K[ok], L[ok], kl[ok]
            fv.append(np.stack([np.full(len(K), r), sp[K], kl, SIDX[K, L]], 1))
            mx = sp[K] != sp[L]
            fv.append(np.stack([np.full(int(mx.sum()), r), sp[L][mx], kl[mx], SIDX[L[mx], K[mx]]], 1))
        self._fvnp = np.concatenate(fv) if fv else np.zeros((0, 4), np.int64)
        rb = []
        for (I, J) in self.strong:
            o = self.sidx[(I, J)]
            p = self.sp_of[o]
            a, b = sp[I], sp[J]
            ds = np.array(u.dom_so[p], np.int64)
            K, L = np.meshgrid(ds, ds, indexing='ij')
            K, L = K.ravel(), L.ravel()
            kl = ij[lmo[K], lmo[L]]
            ok = (K != L) & (kl >= 0)
            K, L, kl = K[ok], L[ok], kl[ok]
            d = (sp[K] == a) & (sp[L] == b)
            x = (a != b) & (sp[K] == b) & (sp[L] == a)
            it = np.where(d, SIDX[K, L], SIDX[L, K])
            sg = np.where(d, 1., -1.)
            k_ = d | x
            n = int(k_.sum())
            rb.append(np.stack([np.full(n, o), K[k_], L[k_], kl[k_], it[k_], sg[k_]], 1))
        self._rbnp = np.concatenate(rb) if rb else np.zeros((0, 6))
        self.nbeta = len(self._rbnp)

    def _split_kl(self):
        self._np_kl_terms()
        u = self.u
        nvs, sp = self.nvs, self.spin
        fvn, rbn = self._fvnp, self._rbnp
        fv_c = np.array(self.strong_c, np.int64)[fvn[:, 0]] if len(fvn) else np.zeros(0, np.int64)
        rb_o = rbn[:, 0].astype(np.int64)
        rb_c = self.cmap[self.sp_of[rb_o]] if len(rb_o) else np.zeros(0, np.int64)
        rb_p = self.sp_of[rb_o] if len(rb_o) else np.zeros(0, np.int64)
        rb_kl = rbn[:, 3].astype(np.int64)
        rb_a = sp[np.array([self.sop[o][0] for o in rb_o], np.int64)] if len(rb_o) else np.zeros(0, np.int64)
        rb_b = sp[np.array([self.sop[o][1] for o in rb_o], np.int64)] if len(rb_o) else np.zeros(0, np.int64)
        PP = int(padded(max(len(u.pao[p]) for p in range(self.npair))))
        # chunk the strong pairs by the size of their S(kl, c) and of the gather
        sc = np.array(self.strong_c, np.int64)
        cost_b = np.zeros(self.npair)
        cost_g = np.zeros(self.npair)
        for cc, s_, kl in ((fv_c, fvn[:, 1] if len(fvn) else [], fvn[:, 2] if len(fvn) else []),
                           (rb_c, rb_a, rb_kl), (rb_c, rb_b, rb_kl)):
            if len(cc):
                npr = self.npair
                code = np.unique((np.asarray(cc, np.int64) * 2 + np.asarray(s_, np.int64)) * npr
                                 + np.asarray(kl, np.int64))
                k0, ks = code // npr // 2, (code // npr) % 2
                k2 = code % npr
                np.add.at(cost_b, k0, 8. * nvs[k2, ks] * nvs[k0, ks])
                np.add.at(cost_g, k0, 8. * PP * nvs[k0, ks])
        chunks, cur, cb, cg = [], [], 0., 0.
        for c in sc:
            if cur and (cb + cost_b[c] > self.MAX_S_CHUNK_BYTES or cg + cost_g[c] > self.MAX_S_CHUNK_BYTES):
                chunks.append(cur)
                cur, cb, cg = [], 0., 0.
            cur.append(c)
            cb += cost_b[c]
            cg += cost_g[c]
        if cur:
            chunks.append(cur)
        cid = -np.ones(self.npair, np.int64)
        for n, ch in enumerate(chunks):
            cid[ch] = n
        self.klchunks = []

        def by_chunk(cc):
            # term indices of every chunk (one sort instead of a scan per chunk)
            if not len(cc):
                return [np.zeros(0, np.int64)] * len(chunks)
            k = cid[cc]
            o = np.argsort(k, kind='stable')
            b = np.searchsorted(k[o], np.arange(len(chunks) + 1))
            return [o[b[i]:b[i + 1]] for i in range(len(chunks))]
        fby, rby = by_chunk(fv_c), by_chunk(rb_c)
        for n, ch in enumerate(chunks):
            cpos = -np.ones(self.npair, np.int64)
            cpos[ch] = np.arange(len(ch))
            d = dict(cs=np.array(ch, np.int64), cpl=[])
            fsel, rsel = fby[n], rby[n]
            loc_idx = {}
            for s in (0, 1):
                # couples (kl, c) of this chunk and spin
                fs = fsel[fvn[fsel, 1] == s]
                ra = rsel[rb_a[rsel] == s]
                rbs = rsel[rb_b[rsel] == s]
                kl_all = np.concatenate([fvn[fs, 2], rb_kl[ra], rb_kl[rbs]]).astype(np.int64)
                c_all = np.concatenate([fv_c[fs], rb_p[ra], rb_p[rbs]]).astype(np.int64)
                code = kl_all * self.npair + c_all
                ukey, inv = np.unique(code, return_inverse=True)
                kl_u, c_u = ukey // self.npair, ukey % self.npair
                nf, na = len(fs), len(ra)
                loc_idx[s] = (fs, inv[:nf], ra, inv[nf:nf + na], rbs, inv[nf + na:])
                d['cpl'].append((kl_u, cpos[self.cmap[c_u]], Layout(nvs[kl_u, s], nvs[c_u, s]), c_u))
            fvt = []
            for s in (0, 1):
                fs, li = loc_idx[s][0], loc_idx[s][1]
                fvt.append(_Terms.from_columns(
                    c=fvn[fs, 0].astype(np.int64), s1=np.asarray(li, np.int64),
                    it=fvn[fs, 3].astype(np.int64), sa=np.full(len(fs), s, np.int64)))
            # oooo ladder rows of the chunk: S column of spin a and of spin b
            # (rsel is ascending, ra / rbs are subsets of it)
            s1 = np.zeros(len(rsel), np.int64)
            s2 = np.zeros(len(rsel), np.int64)
            for s in (0, 1):
                _, _, ra, la, rbs, lb = loc_idx[s]
                s1[np.searchsorted(rsel, ra)] = la
                s2[np.searchsorted(rsel, rbs)] = lb
            rt = _Terms.from_columns(o=rb_o[rsel], s1=s1, t=rbn[rsel, 4].astype(np.int64), s2=s2,
                                     a=rb_a[rsel], b=rb_b[rsel])
            d['fv'], d['rb'] = fvt, rt
            d['beta_n'] = rsel
            d['rb_w'] = .5 * rbn[rsel, 5]
            self.klchunks.append(d)
        self.fv = [_Terms(), _Terms()]
        self.rb = _Terms()
        self._compact_S()

    def _s_columns(self):
        """(terms, S column, spin column or fixed spin) of every term list
        whose plans read the global overlap store."""
        L = [(self.tn[0], 's', 0), (self.tn[1], 's', 1)]
        L += [(self.fp, 'sa', 'a'), (self.fp, 'sb', 'b'), (self.rg, 's1', 'a'), (self.rg, 's2', 'b')]
        for tr in (0, 1):
            L += [(self.ring_np[tr], 's1', 'a'), (self.ring_z[tr], 's1', 'a'),
                  (self.ring_z[tr], 's2', 'c'), (self.ring_z[tr], 's3', 'b')]
        L += [(self.zq, 's1', 'b'), (self.zq, 's2', 'd'), (self.zq, 's3', 'c'),
              (self.fkc, 's1', 'a'), (self.fkc, 's2', 'b'), (self.r1c, 's1', 'a'),
              (self.r1a2, 's1', 'a'), (self.r1a2, 'sa', 'aK'), (self.r1a2, 'sb', 'a'),
              (self.uq, 's', 'a'), (self.r1b, 's1', 'a'), (self.pt, 's', 'a'), (self.a1, 's', 'a')]
        for t_ in self.zkx:
            L += [(t_, 's1', 'a'), (t_, 's2', 'b')]
        return L

    def _compact_S(self):
        cols = self._s_columns()
        import os
        if os.environ.get('DLPNO_DEBUG_S'):
            names = {}
            for t_, col, sc in cols:
                if not len(t_) or col not in t_:
                    continue
                spins = [sc] * len(t_[col]) if isinstance(sc, int) else t_[sc]
                key = id(t_)
                names.setdefault(key, set()).update(zip(spins, t_[col]))
            tot = sum(len(v) for v in names.values())
            print('S couples per term list:', sorted((len(v) for v in names.values()), reverse=True)[:12],
                  'total refs', tot, 'unique', len(set().union(*names.values())) if names else 0)
        # (numpy) the overlap couples still referenced, renumbered in order
        refs = []
        for t_, col, sc in cols:
            if not len(t_) or col not in t_:
                continue
            ii = np.asarray(t_[col], np.int64)
            ss = np.full(len(ii), sc, np.int64) if isinstance(sc, int) else np.asarray(t_[sc], np.int64)
            refs.append((t_, col, ii, ss))
        new = [_Reg(), _Reg()]
        remap = []
        for s in (0, 1):
            ids = [ii[ss == s] for _, _, ii, ss in refs]
            used = np.unique(np.concatenate(ids)) if ids else np.zeros(0, np.int64)
            keys = self.Sreg[s].keys
            new[s].keys = [keys[i] for i in used.tolist()]
            new[s].ids = {k: n for n, k in enumerate(new[s].keys)}
            m = -np.ones(len(keys), np.int64)
            m[used] = np.arange(len(used))
            remap.append(m)
        for t_, col, ii, ss in refs:
            out = np.empty_like(ii)
            for s in (0, 1):
                m = ss == s
                out[m] = remap[s][ii[m]]
            t_[col] = out
        self.Sreg = new

    def _kl_tables_dev(self, dev):
        """(LXk, Xk, MXk) on device ``dev`` (built once per device)."""
        c = self.__dict__.setdefault('_klt', {})
        if dev not in c:
            keep = (getattr(self, 'LXk', None), getattr(self, 'Xk', None), getattr(self, 'MXk', None))
            if keep[1] and keep[1][0].data and keep[1][0].data[0].device.id == dev:
                c[dev] = keep
            else:
                with cp.cuda.Device(dev):
                    self._kl_tables()
                    c[dev] = (self.LXk, self.Xk, self.MXk)
                self.LXk, self.Xk, self.MXk = keep
        return c[dev]

    def _kl_tables(self):
        """pair virtual coefficients over all PAOs (beta: + SOMO rows), and the
        same multiplied by the PAO (+SOMO) metric, as Raggeds per spin."""
        u, ns = self.u, self.u.ns
        nbf = u.orb.nbf
        self.LXk, self.Xk, self.MXk = [], [], []
        for s in (0, 1):
            nrow = nbf + (ns if s == 1 else 0)
            LX = Layout(np.full(self.npair, nrow), self.nvs[:, s])
            X = Ragged(LX)
            mats = []
            for p in range(self.npair):
                npo = len(u.pao[p])
                M = np.zeros((nrow, self.nvs[p, s]))
                if s == 0:
                    M[u.pao[p]] = u.Xa[p]
                else:
                    M[u.pao[p]] = u.Xb[p][:npo]
                    M[nbf:] = u.Xb[p][npo:]
                mats.append(cp.asarray(M))
            _set_items(X, range(self.npair), mats)
            pr = LX.shapes[0][0]
            Mt = cp.zeros((pr, pr))
            Mt[:nbf, :nbf] = u.S_pao_d
            if s == 1 and ns:
                Mt[nbf:nbf + ns, nbf:nbf + ns] = cp.eye(ns)
            self.LXk.append(LX)
            self.Xk.append(X)
            self.MXk.append(Ragged(LX, [cp.matmul(Mt, d) for d in X.data]))

    def _tmp(self, name, layout):
        """persistent zeroed work Ragged (stable addresses: the plans' pointer
        tables stay cached and no host->device copy stalls the stream)."""
        tm = self.__dict__.setdefault('_tmps', {})
        r = tm.get(name)
        if r is None or r.layout is not layout:
            r = tm[name] = Ragged(layout)
        else:
            for d in r.data:
                d.fill(0.)
        return r

    def _gens_on(self, dev):
        """overlap generators on device ``dev`` (built once per device)."""
        if dev == getattr(self, '_kl_home', dev) and getattr(self, '_ovgens', None) is not None:
            return self._ovgens
        g = self.__dict__.setdefault('_ovg_dev', {})
        if dev not in g:
            from electrolites.dlpno.ovgen import OverlapGen
            u = self.u
            with cp.cuda.Device(dev):
                Spao = cp.asarray(u.S_pao_d) if u.S_pao_d.device.id != dev else u.S_pao_d
                if not Spao.flags.c_contiguous:
                    Spao = cp.ascontiguousarray(Spao)
                g[dev] = [OverlapGen(Spao, u.pao, [u.Xa[p] for p in range(self.npair)]),
                          OverlapGen(Spao, u.pao, [u.Xb[p] for p in range(self.npair)], ne=u.ns)]
        return g[dev]

    def _S_workspace(self):
        """per device one flat buffer, viewed as every chunk's S Raggeds
        (the chunks of a device run one at a time)."""
        home = getattr(self, '_kl_home', cp.cuda.Device().id)
        for dev in sorted({d.get('dev', home) for d in self.klchunks}):
            with cp.cuda.Device(dev):
                self._S_workspace_dev(dev, [d for d in self.klchunks if d.get('dev', home) == dev], home)

    def _S_workspace_dev(self, dev, chunks, home):
        sizes = [sum(sum(len(m) * r * c for m, (r, c) in zip(c_[2].members, c_[2].shapes))
                     for c_ in d['cpl']) for d in chunks]
        need = max(sizes) if sizes else 0
        # the S(kl, c) are fixed: when all chunks of this device fit (40% of
        # its free memory) each chunk keeps its own, generated once
        free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        # (regenerating them is cheap with the overlap W tables: keep them only
        # when they take little of the device)
        keep = 8. * sum(sizes) < 0.15 * free and os.environ.get('DLPNO_KL_KEEP', '1') == '1'
        ws = None
        if keep:
            try:
                ws = cp.empty(max(sum(sizes), 1))
            except cp.cuda.memory.OutOfMemoryError:
                keep = False
        if ws is None:
            ws = cp.empty(max(need, 1))
        # S(kl, c) of the chunks from the full-PAO coefficient tables of this
        # device (one variable-size GEMM per spin; much cheaper than the
        # union-domain generator for these many small couples)
        from electrolites.dlpno import mgpu
        # full-PAO tables (X and SX per spin over all PAO rows) only when they
        # fit next to the rest: on a compact protein they are ~100 GB a device
        u = self.u
        tab = 16. * sum((u.orb.nbf + (u.ns if s_ else 0)) * float(self.nvs[:, s_].sum()) for s_ in (0, 1))
        env = os.environ.get('DLPNO_KL_TABLES', 'auto')
        fits = tab < 0.3 * free if env == 'auto' else env == '1'
        if fits and (len(mgpu.devices()) > 1 or getattr(self, '_ovgens', None) is None):
            gens, ktab = None, self._kl_tables_dev(dev)
        else:                   # one GPU, overlaps on the fly: no full-PAO tables
            gens, ktab = self._gens_on(dev), None
        base = 0
        for d, sz in zip(chunks, sizes):
            off = base if keep else 0
            if keep:
                base += sz
            d['S_keep'] = keep
            d['S_done'] = False
            rags = []
            for kl, cl, lay, cu in d['cpl']:
                data = []
                for m, (r, c) in zip(lay.members, lay.shapes):
                    n = len(m) * r * c
                    data.append(ws[off:off + n].reshape(len(m), r, c))
                    off += n
                rags.append(Ragged(lay, data))
            d['Sc'] = rags
            if gens is not None:
                d['gen'] = [gens[s].prepare(kl, cu, out_layout=lay, out_idx=np.arange(len(kl)))
                            for s, (kl, cl, lay, cu) in enumerate(d['cpl'])]
                d['gens'] = gens
            else:
                d['gen'] = [VChainPlan(lay, np.arange(len(kl)), [op(ktab[0][s], kl, True), op(ktab[0][s], cu)])
                            for s, (kl, cl, lay, cu) in enumerate(d['cpl'])]
                d['gens'] = None
                d['ktab'] = ktab
            d['ws'] = ws[off - sz if keep else 0:off]
        self.__dict__.setdefault('_sws', {})[dev] = ws

    def _gen_S(self, ch):
        """the S(kl, c) of one chunk, per spin: X_kl^T (S X_c) over full PAO
        (+SOMO) rows, one variable-size batched GEMM per spin."""
        if 'Sc' not in ch:
            self._S_workspace()
        if ch.get('S_keep') and ch.get('S_done'):
            return ch['Sc']
        ch['S_done'] = True
        ch['ws'].fill(0.)
        for s in (0, 1):
            if ch.get('gens') is not None:
                for P in ch['gen'][s]:
                    ch['gens'][s].run(P, ch['Sc'][s])
            else:
                kt = ch.get('ktab') or (self.LXk, self.Xk, self.MXk)
                ch['gen'][s].run(ch['Sc'][s], [kt[1][s], kt[2][s]])
        return ch['Sc']

    def _df_groups(self):
        o_ = self._rbnp[:, 0].astype(np.int64)
        order = np.argsort(o_, kind='stable')
        uo, st_ = np.unique(o_[order], return_index=True)
        self._rb_by_o = {int(x): g for x, g in zip(uo, np.split(order, st_[1:]))}
        u, pl = self.u, self.pl
        lmo, sp = self.lmo, self.spin
        sc = self.strong_c
        fx = self.fix
        naux = np.array([fx[c]['Qov'][0].shape[0] for c in sc], np.int64)
        # coarse buckets (few groups, few launches per iteration); every padded
        # item shape fits its bucket since padded(n) <= ceil16(n) <= bucket
        bk = lambda n, q: np.maximum(q, -(-np.asarray(n) // q) * q)
        Ap = bk(naux, 128)
        Rb = [bk(np.array([len(self.rows[c][s]) for c in sc]), 16) for s in (0, 1)]
        Vb = [bk(self.nvs[np.array(sc, np.int64), s], 32) for s in (0, 1)]
        key = np.stack([Ap, Rb[0], Rb[1], Vb[0], Vb[1]], 1)
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        self.dfg = []
        bmap = {}
        # offload: the packed DF groups and the 3-external groups in files on
        # scratch when together they would take more than 35% of the host's
        # available memory (DLPNO_CC_DF_DISK=auto|0|1)
        self._df_store = None
        env = os.environ.get('DLPNO_CC_DF_DISK', 'auto')
        nbh = getattr(self, '_df_bytes', 0.) + getattr(self, '_k3_bytes', 0.)
        if self.offload and (env == '1' or (env == 'auto' and nbh > 0.35 * _host_available())):
            from electrolites.dlpno.devmem import SpillStore
            self._df_store = SpillStore('uccdf')
        # several GPUs: the groups (and their DF tensors) are spread over them
        from electrolites.dlpno import mgpu
        devs = mgpu.devices()
        load = np.zeros(len(devs))
        home = cp.cuda.Device().id
        if len(devs) > 1:
            # the home device keeps the overlaps, 3-external tensors and plans
            load[devs.index(home)] = np.inf
            mgpu.enable_peer(devs)        # the groups copy the per-pair blocks by pointer
        for gi in range(len(uk)):
            A = int(uk[gi][0])
            R = [int(uk[gi][1]), int(uk[gi][2])]
            V = [int(uk[gi][3]), int(uk[gi][4])]
            per = 8 * A * (sum(V[s] * V[s] + 2 * R[s] * V[s] + V[s] * R[s] for s in (0, 1)) + 4 * max(V) ** 2)
            step = max(1, self.MAX_DF_CHUNK_BYTES // per)
            allm = np.array(sc)[inv == gi]
            for c0 in range(0, len(allm), step):
                r = devs.index(home) if os.environ.get('DLPNO_DBG_DF_HOME') else int(np.argmin(load))
                mem_ = allm[c0:c0 + step]
                load[r] += len(mem_) * per
                with cp.cuda.Device(devs[r]):
                    d = self._df_group(mem_, A, R, V, bmap)
                d['dev'] = devs[r]
                self.dfg.append(d)
        cp.cuda.Device(home).use()
        if self.offload and getattr(u, 'verbose', 0) >= 2:
            nb = sum(v.nbytes for d in self.dfg for v in d.values() if isinstance(v, _Packed))
            where = f'on scratch ({self._df_store.dir})' if self._df_store is not None else 'in pinned host memory'
            print(f'[mem] open CC DF groups {where}: {nb / 1e9:.1f} GB', flush=True)
        if self._df_store is not None:
            self._df_store.flush()
        # beta entries -> (group, combo, which matrix, member, rK, rL)
        self.beta_src = bmap
        # registries may have grown in _vec_terms: rebuild the dependent stores
        # (after the DF groups, whose per-pair tensors have left this device)
        cp.get_default_memory_pool().free_all_blocks()
        self._rebuild_after_vec()

    def _rebuild_after_vec(self):
        """overlaps, Pt layout, Kx store and plans depend on registries that
        _vec_terms extended."""
        u = self.u
        self.LPt = Layout(np.array([self.nvs[p, self.spin[Z]] for (Z, p) in self.preg.keys], np.int64), 1)
        self.pt = _Terms()
        for r_, (Z, p) in enumerate(self.preg.keys):
            self.pt.add(r=r_, s=self.S(p, self.dii[Z], self.spin[Z]), z=Z, a=self.spin[Z])
        self.ONFLY_S = hasattr(self.u, 'S_pao_d') and self._decide_onfly()
        if self.ONFLY_S:
            self._make_ovgens()
        else:
            self._ovgens = None
        self._overlaps()
        mats = [self.fix[kl]['Kx'][(self.lmo[K], self.lmo[L], self.spin[K], self.spin[L])]
                for (kl, K, L) in self.kx_reg.keys]
        self.LKx = Layout(np.array([m.shape[0] for m in mats], np.int64),
                          np.array([m.shape[1] for m in mats], np.int64))
        self.Kx = Ragged(self.LKx)
        _set_items(self.Kx, range(len(mats)), mats)
        # the integral caches (pair metric factors, (Q|uv) blocks: tens of GB
        # on a compact protein) give the device to the plans
        getattr(u, '_chol', {}).clear()
        if hasattr(getattr(u, 'dfi', None), 'cache_clear'):
            u.dfi.cache_clear()
        cp.get_default_memory_pool().free_all_blocks()
        self._plans()
        lays = dict(S=self.LS, T=self.LT, Pt=self.LPt, Kx=self.LKx, Tn=self.LTn, J1R=self.LTn,
                    Fv=self.LF, V1=self.LV1)
        k1z_items, k1z_mats = self._k1z
        self.LK1Z = Layout(np.array([m.shape[0] for m in k1z_mats], np.int64),
                           np.array([m.shape[1] for m in k1z_mats], np.int64))
        self.K1Z = Ragged(self.LK1Z)
        _set_items(self.K1Z, range(len(k1z_mats)), k1z_mats)
        lays['K1Z'] = self.LK1Z
        SP = self._make_spin_plan
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
        from electrolites.dlpno import mgpu
        if getattr(self, '_ovgens', None) is None and len(mgpu.devices()) == 1:
            # (several GPUs: the kl chunks run on the other devices with generators)
            self._kl_tables()
        # several GPUs: the F'_vv / oooo-ladder chunks (S(kl, c) generated per
        # chunk) are spread over the devices
        from electrolites.dlpno import mgpu
        devs = mgpu.devices()
        home = cp.cuda.Device().id
        load = np.zeros(len(devs))
        if len(devs) > 1:
            load[devs.index(home)] = np.inf
        for d in self.klchunks:
            r = devs.index(home) if os.environ.get('DLPNO_DBG_KL_HOME') else int(np.argmin(load))
            load[r] += len(d['rb']) + len(d['fv'][0]) + len(d['fv'][1]) + 1
            d['dev'] = devs[r]
            with cp.cuda.Device(devs[r]):
                cl = dict(Sc=[c[2] for c in d['cpl']], T=self.LT, G=self.LT)
                d['p_fv'] = [_SpinPlan(cl, self.LF[s], d['fv'][s], 'c',
                                       [('Sc', 's1', 'sa', True), ('T', 'it', None, False),
                                        ('G', 'it', None, True), ('Sc', 's1', 'sa', False)])
                             for s in (0, 1)]
                d['p_rb'] = _SpinPlan(cl, self.LT, d['rb'], 'o', [('Sc', 's1', 'a', True), ('T', 't', None, False),
                                                                ('Sc', 's2', 'b', False)])
                d['beta'] = cp.asarray(d['beta_n'])
                d['w'] = cp.asarray(d['rb_w'])
        cp.cuda.Device(home).use()
        self._kl_home = home
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
            Y = self.fix[c]['K3'][(z, s, t)]
            Y = as_host(Y) if not isinstance(Y, (np.ndarray, cp.ndarray)) else Y
            return Y.transpose(1, 0, 2)

        def shape(key):
            c, z, s, t = key
            a_, b_, c_ = self.fix[c]['K3'][(z, s, t)].shape
            return (b_, a_, c_)

        def release(key):
            c, z, s, t = key
            self.fix[c]['K3'].pop((z, s, t), None)
        self.k3.build(fetch, host=self.offload, shape=shape, release=release,
                      store=getattr(self, '_df_store', None))
        for c in self.canon_sp:
            self.fix[c].pop('K3', None)
        if self.offload:
            from electrolites.dlpno.devmem import trim_host
            trim_host()

    #: generate the PNO overlaps of every term on the fly (no stored S)
    import os as _os
    ONFLY_MODE = _os.environ.get('DLPNO_ONFLY_S', 'auto')
    ONFLY_S = None

    def _decide_onfly(self):
        if self.ONFLY_MODE != 'auto':
            return self.ONFLY_MODE == '1'
        need = 0.
        for s in (0, 1):
            keys = self.Sreg[s].keys
            if keys:
                pp = np.array([k[0] for k in keys], np.int64)
                qq = np.array([k[1] for k in keys], np.int64)
                need += float(np.sum(8. * self.nvs[pp, s] * self.nvs[qq, s]))
        free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        if not self.offload:
            # the grouped 3-external tensors are still to be built here
            free -= getattr(self, '_k3_bytes', 0.)
        from electrolites.dlpno import mgpu as _mg
        # several GPUs: prefer the store over the other devices early (home
        # keeps the 3-external tensors and the plans)
        lim = 0.25 if len(_mg.devices()) > 1 else 0.5
        onfly = need > lim * free or __import__('os').environ.get('DLPNO_FORCE_STORE') == '1'
        self._sstores = None
        from electrolites.dlpno import mgpu
        devs = mgpu.devices()
        home = cp.cuda.Device().id
        others = [d for d in devs if d != home]
        if onfly and others and self.ONFLY_MODE == 'auto':
            cap = 0.
            for d in others:
                with cp.cuda.Device(d):
                    cap += cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
            if need < 0.5 * cap:
                # computed once, spread over the other GPUs; chunks gather by pointer
                from electrolites.dlpno.ccsd_gpu import _SStore
                u = self.u
                npair = self.npair
                st = []
                for s_ in (0, 1):
                    keys = self.Sreg[s_].keys
                    pp = np.array([k[0] for k in keys], np.int64)
                    qq = np.array([k[1] for k in keys], np.int64)
                    X = [u.Xa[p] for p in range(npair)] if s_ == 0 else [u.Xb[p] for p in range(npair)]
                    st.append(_SStore(u.S_pao_d, u.pao, X, pp, qq, self.nvs[:, s_].astype(np.int64), others,
                                      ne=(u.ns if s_ == 1 else 0)) if len(keys) else None)
                cp.cuda.Device(home).use()
                self._sstores = st
        if onfly:
            # generated per chunk; chunks keep their overlaps up to a budget
            import os
            from electrolites.dlpno.ccsd_gpu import _SBudget
            frac = float(os.environ.get('DLPNO_S_KEEP', '0.3'))
            self._s_budget = _SBudget({cp.cuda.Device().id: frac * free})
        if getattr(self.u, 'verbose', 0) >= 2:
            print(f'[mem] open CC overlaps {need / 1e9:.1f} GB, free {free / 1e9:.1f} GB, '
                  f'on the fly: {onfly}', flush=True)
        return onfly

    def _make_spin_plan(self, layouts, out_layout, terms, out_col, specs, **kw):
        uses_s = any(k == 'S' for (k, c, sc, tr) in specs)
        if self.ONFLY_S and uses_s and getattr(self, '_ovgens', None) is not None:
            return ChunkedSpinPlan(self, layouts, out_layout, terms, out_col, specs, **kw)
        return _SpinPlan(layouts, out_layout, terms, out_col, specs, **kw)

    def _overlaps(self):
        if self.ONFLY_S is None:            # base build: the decision comes after _vec_terms
            self.LS, self.Sd = [None, None], [None, None]
            return
        if self.ONFLY_S:
            self.LS, self.Sd = [None, None], [None, None]
            return
        super()._overlaps()

    def _make_ovgens(self):
        from electrolites.dlpno.ovgen import OverlapGen
        u = self.u
        Spao = u.S_pao_d
        npair = self.npair
        self._ovgens = [OverlapGen(Spao, u.pao, [u.Xa[p] for p in range(npair)]),
                        OverlapGen(Spao, u.pao, [u.Xb[p] for p in range(npair)], ne=u.ns)]

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
        # offload: the members' blocks packed back to back (no padding) in
        # pinned host memory or, with a DF store, in a file on scratch; padded
        # on the device at every use (ccsd_gpu._Packed)
        from electrolites.dlpno.devmem import as_host
        packed = bool(self.offload)
        store = getattr(self, '_df_store', None)
        e2 = np.zeros((0, 0))
        for s in (0, 1):
            if packed:
                Qov, Qvv, Qmz, Qaz = [], [], [], []
            else:
                Qov = cp.zeros((g, A, R[s], V[s]))
                Qvv = cp.zeros((g, A, V[s], V[s]))
                Qmz = cp.zeros((g, 2, A, R[s]))
                Qaz = cp.zeros((g, 2, A, V[s]))
            fov = cp.zeros((g, R[s], V[s]))
            ev = cp.zeros((g, V[s]))
            zrow = np.zeros((g, 2), np.int64)
            zm = np.zeros((g, 2))
            on_dev = not packed and all(isinstance(fx[c]['Qov'][s], cp.ndarray) for c in mem)
            # device sources: one batched padded copy per tensor (the per-pair
            # blocks may live on a peer device)
            bat = dict(ov=[], vv=[], fov=[], ev=[], mz=[], mzc=[], az=[]) if on_dev else None
            for m, c in enumerate(mem):
                f = fx[c]
                x, y = pl.pairs[c]
                na, nr = f['Qov'][s].shape[0], f['Qov'][s].shape[1]
                nv = f['Qvv'][s].shape[1]
                if on_dev:
                    bat['ov'].append((m, f['Qov'][s]))
                    bat['vv'].append((m, f['Qvv'][s]))
                    bat['fov'].append((m, cp.asarray(f['fov'][s])))
                    bat['ev'].append((m, cp.asarray(f['ev'][s])))
                elif packed:
                    Qov.append(as_host(f['Qov'][s]))
                    Qvv.append(as_host(f['Qvv'][s]))
                    fov[m, :nr, :nv] = cp.asarray(f['fov'][s])
                    ev[m, :nv] = cp.asarray(f['ev'][s])
                else:
                    Qov[m, :na, :nr, :nv] = cp.asarray(f['Qov'][s])
                    Qvv[m, :na, :nv, :nv] = cp.asarray(f['Qvv'][s])
                    fov[m, :nr, :nv] = cp.asarray(f['fov'][s])
                    ev[m, :nv] = cp.asarray(f['ev'][s])
                rp = None
                for side, z in enumerate((x, y)):
                    if nr:
                        if on_dev:
                            if rp is None:
                                with cp.cuda.Device(f['Qmz'][z].device.id):
                                    rp = cp.asarray(np.asarray(f['rowsp'][s], np.int64))
                            bat['mz'].append((2 * m + side, f['Qmz'][z]))
                            bat['mzc'].append(rp)
                        elif packed:
                            Qmz.append(as_host(f['Qmz'][z])[:, np.asarray(f['rowsp'][s], np.int64)])
                        else:
                            rp = cp.asarray(np.asarray(f['rowsp'][s], np.int64))
                            Qmz[m, side, :na, :nr] = cp.asarray(f['Qmz'][z])[:, rp]
                    elif packed:
                        Qmz.append(e2)
                    if (z, s) in si:
                        if on_dev:
                            bat['az'].append((2 * m + side, f['Qaz'][(z, s)]))
                        elif packed:
                            Qaz.append(as_host(f['Qaz'][(z, s)]))
                        else:
                            Qaz[m, side, :na, :nv] = cp.asarray(f['Qaz'][(z, s)])
                        zrow[m, side] = self.rows[c][s].index(si[(z, s)])
                        zm[m, side] = 1.
                    elif packed:
                        Qaz.append(e2)
            if packed:
                Qov = _Packed(Qov, (A, R[s], V[s]), store=store)
                Qvv = _Packed(Qvv, (A, V[s], V[s]), store=store)
                Qmz = _Packed(Qmz, (A, R[s]), 2, store=store)
                Qaz = _Packed(Qaz, (A, V[s]), 2, store=store)
            if on_dev:
                from electrolites.dlpno.gather import pad_copy
                for key, dst in (('ov', Qov), ('vv', Qvv), ('fov', fov), ('ev', ev)):
                    pad_copy(dst, [a for _, a in bat[key]], [m for m, _ in bat[key]])
                gA = (2 * g,) + Qmz.shape[2:]
                pad_copy(Qmz.reshape(gA), [a for _, a in bat['mz']], [m for m, _ in bat['mz']], bat['mzc'])
                pad_copy(Qaz.reshape((2 * g,) + Qaz.shape[2:]), [a for _, a in bat['az']],
                         [m for m, _ in bat['az']])
                bat = None
            d[('Qov', s)], d[('Qvv', s)], d[('fov', s)], d[('ev', s)] = Qov, Qvv, fov, ev
            d[('Qmz', s)], d[('Qaz', s)] = Qmz, Qaz

            d[('zrow', s)], d[('zm', s)] = cp.asarray(zrow), cp.asarray(zm)
            if s == 1:
                for c in mem:                 # assembled: drop the per-pair copies
                    for key in ('Qov', 'Qvv', 'Qmz', 'Qaz'):
                        fx[c].pop(key, None)
                if packed:
                    from electrolites.dlpno.devmem import trim_host
                    trim_host()
            # members by source / target layout group
            its = np.array([self.cpos[c] for c in mem], np.int64)
            lay = self.LTn[s]
            d[('tn', s)] = [(int(gs), cp.asarray(np.where(lay.group[its] == gs)[0]),
                             cp.asarray(lay.slot[its[lay.group[its] == gs]])) + tuple(lay.shapes[gs])
                            for gs in np.unique(lay.group[its])]
            itf = np.array([self.fvpos[c] for c in mem], np.int64)
            lay = self.LF[s]
            d[('fv', s)] = [(int(gs), cp.asarray(np.where(lay.group[itf] == gs)[0]),
                             cp.asarray(lay.slot[itf[lay.group[itf] == gs]])) + tuple(lay.shapes[gs])
                            for gs in np.unique(lay.group[itf])]
        # strong spin-orbital pairs by combo (side_i, a, side_j, b)
        combos = {}
        for m, c in enumerate(mem):
            x, y = pl.pairs[c]
            for (I, J) in self._strong_of(c):
                o = self.sidx[(I, J)]
                key = (0 if lmo[I] == x else 1, sp[I], 0 if lmo[J] == x and x != y else (1 if x != y else 0), sp[J])
                if x == y:
                    key = (0, sp[I], 0, sp[J])
                combos.setdefault(key + (int(self.LT.group[o]),), []).append((m, o, I, J))
        d['combos'] = []
        for key, lst in combos.items():
            si_, a, sj_, b, gT = key
            mm = np.array([l[0] for l in lst], np.int64)
            oo = np.array([l[1] for l in lst], np.int64)
            cid = len(d['combos'])
            d['combos'].append(dict(key=key[:4], m=cp.asarray(mm), o=oo, gT=gT,
                                    sT=cp.asarray(self.LT.slot[oo]), shp=self.LT.shapes[gT]))
            # beta entries of these pairs: (n, position in combo, rK, rL)
            gi = len(self.dfg)
            for which in (0, 1):
                parts = []
                for r_, (m, o, I, J) in enumerate(lst):
                    ns_ = self._rb_by_o.get(o)
                    if ns_ is None:
                        continue
                    K = self._rbnp[ns_, 1].astype(np.int64)
                    L = self._rbnp[ns_, 2].astype(np.int64)
                    sK, sL = sp[K], sp[L]
                    wh = np.where((sK == a) & (sL == b), 0, 1)
                    q = wh == which
                    if not q.any():
                        continue
                    rp = self._rowpos(mem[m])
                    parts.append(np.stack([ns_[q], np.full(int(q.sum()), r_), rp[sK[q], K[q]], rp[sL[q], L[q]]], 1))
                if parts:
                    arr = np.concatenate(parts)
                    assert (arr >= 0).all()
                    bmap[(gi, cid, which)] = tuple(cp.asarray(arr[:, j]) for j in range(4))
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

    def _rowpos(self, c):
        """rp[s, K] = row of spin orbital K in pair c's spin-s rows (-1 if absent)."""
        rp = -np.ones((2, self.nso), np.int64)
        for s in (0, 1):
            rp[s, self.rows[c][s]] = np.arange(len(self.rows[c][s]))
        return rp

    # ------------------------------------------------------------------
    def residuals(self, T, T1):
        from electrolites.dlpno.devmem import Prof
        prof = self.__dict__.setdefault('_prof', Prof('ULCCSD residual'))
        prof('tn')
        sp = self.spin
        st = self.stores(T, T1)
        Tn = [self.p_tn[s].run(self._tmp(('Tn', s), self.LTn[s]), st) for s in (0, 1)]
        st['Tn'] = Tn
        st['J1R'] = self.J1R
        Pt = self.p_pt.run(self._tmp('Pt', self.LPt), st)
        st['Pt'] = Pt
        Fv = [self._tmp(('Fv', s), self.LF[s]) for s in (0, 1)]
        Fbov = [self._tmp(('Fbov', s), self.LTn[s]) for s in (0, 1)]
        R = self._tmp('R', self.LT)
        R1 = self._tmp('R1', self.LT1)
        beta = cp.zeros(self.nbeta)
        prof('df')
        self._df_all(T, T1, Tn, Fv, Fbov, R, R1, beta)
        prof('kl')
        st['Fv'] = Fv
        # F'_vv projections and the oooo ladder, with S(kl, c) generated per chunk
        self._kl_all(st, Fv, R, beta)
        st.pop('Sc', None)
        prof('fpp')
        # Fbar_KJ, F'_KJ
        Fpp = self.F0.copy()
        for s in (0, 1):
            Fpp += Tn[s].vdot_items(self.J1R[s])[self.fj1_idx]
            if self.fk1[s] is not None:
                sel, its, lay, K1 = self.fk1[s]
                Fpp[sel] -= gather_like(Tn[s], its, lay).vdot_items(K1)
            lay, plan, sel, ent = self.fbt[s]
            col = plan.run(self._tmp(('fbt', s), lay), [Fbov[s], T1])
            Fpp[sel] += ent(col)
        if self.p_fp.n:
            dots = self.p_fp.dots(st, self.p_fp.n)
            cupyx.scatter_add(Fpp, self.fp_f, dots * self.fp.fd('w'))
        # R2 projection terms
        prof('e1rg')
        if self.p_e1.n:
            self.p_e1.run(R, st)
            self.p_e2.run(R, st)
        if self.p_rg.n:
            self.p_rg.run(R, st, w=Fpp[self.rg_f] * self.rg.fd('w'))
        # ring
        prof('k3')
        Z = self._tmp('Z', self.LZ)
        V1 = self._tmp('V1', self.LV1)
        self.k3.run_all([('c3', 'Pt', 'Z', Pt, Z), ('c2', 'Pt', 'Z', Pt, Z), ('a', 'T', 'V1', T, V1)])
        st['K1Z'] = self.K1Z
        prof('ring')
        self.p_zk1.run(Z, st, w=cp.full(self.p_zk1.n, -1.) if self.p_zk1.n else None)
        self.p_zj1.run(Z, st)
        for p_, t_ in zip(self.p_zkx, self.zkx):
            if p_.n:
                p_.run(Z, st, w=t_.fd('w'))
        if self.p_zq.n:
            self.p_zq.run(Z, st, w=self.zq.fd('w'))
        st['Z'] = Z
        for tr in (0, 1):
            if self.p_ring_np[tr].n:
                self.p_ring_np[tr].run(R, st, w=self.ring_np[tr].fd('w'))
            if self.p_ring_z[tr].n:
                self.p_ring_z[tr].run(R, st, w=self.ring_z[tr].fd('w'))
        # R1
        prof('r1')
        Fk = self.Fk0.copy()
        st['Fk'] = Fk
        if self.p_fkc.n:
            self.p_fkc.run(Fk, st, w=self.fkc.fd('w'))
        if self.p_r1c.n:
            self.p_r1c.run(R1, st, w=self.r1c.fd('w'))
        st['V1'] = V1
        if self.p_a1.n:
            self.p_a1.run(R1, st)
        if self.p_r1a2.n:
            wd = self.p_r1a2_w.dots(st, self.p_r1a2.n)
            self.p_r1a2.run(R1, st, w=wd * self.r1a2.fd('w'))
        Uv = self.U0.copy()
        st['U'] = Uv
        if self.p_uq.n:
            self.p_uq.run(Uv, st)
        if self.p_r1b.n:
            self.p_r1b.run(R1, st, w=self.r1b.fd('w'))
        R = self._mask_strong(R)
        prof(None)
        self.last = dict(Fpp=Fpp, beta=beta, Z=Z, Tn=Tn, Fv=Fv, Fk=Fk, U=Uv)
        return R1, R

    @staticmethod
    def _kl_run(chunks, st, Fv, R, beta):
        for d in chunks:
            st['Sc'] = d['_self']._gen_S(d) if '_self' in d else None
            for s in (0, 1):
                if d['p_fv'][s].n:
                    d['p_fv'][s].run(Fv[s], st, w=cp.full(d['p_fv'][s].n, -.5))
            if d['p_rb'].n:
                d['p_rb'].run(R, st, w=beta[d['beta']] * d['w'])

    def _dev_rep(self, dv, name, X):
        """X (a Ragged of the home device) copied into a persistent buffer on
        device ``dv`` (the current one): stable addresses keep the plans'
        cached pointer tables valid across iterations."""
        c = self.__dict__.setdefault('_reps', {}).setdefault(dv, {})
        r = c.get(name)
        if r is None or r.layout is not X.layout:
            r = c[name] = Ragged(X.layout)
        for d_, s_ in zip(r.data, X.data):
            if s_.size:
                d_.data.copy_from_device_async(s_.data, s_.nbytes)
        return r

    def _dev_out(self, dv, name, layout):
        """persistent zeroed output Ragged on device ``dv``."""
        c = self.__dict__.setdefault('_reps', {}).setdefault(dv, {})
        r = c.get(name)
        if r is None or r.layout is not layout:
            r = c[name] = Ragged(layout)
        else:
            for d_ in r.data:
                d_.fill(0.)
        return r

    def _dev_vec(self, dv, name, x, zero=False):
        c = self.__dict__.setdefault('_reps', {}).setdefault(dv, {})
        r = c.get(name)
        if r is None or r.shape != x.shape:
            r = c[name] = cp.empty(x.shape)
        if zero:
            r.fill(0.)
        else:
            r.data.copy_from_device_async(x.data, x.nbytes)
        return r

    def _kl_all(self, st, Fv, R, beta):
        """F'_vv projections and the oooo ladder of all kl chunks; with
        several GPUs every device runs its chunks on replicas of T (and of the
        static G) and its outputs are added into the home device's."""
        home = cp.cuda.Device().id
        for d in self.klchunks:
            d['_self'] = self
        if any('Sc' not in d for d in self.klchunks):
            self._S_workspace()
        devs = sorted({d.get('dev', home) for d in self.klchunks})
        if devs in ([home], []):
            self._kl_run(self.klchunks, st, Fv, R, beta)
            return
        from electrolites.dlpno import mgpu
        res = {}
        Gd = self.__dict__.setdefault('_G_dev', {})
        # the replicas are copied on the other devices' streams
        cp.cuda.get_current_stream().synchronize()

        def work(dv):
            chs = [d for d in self.klchunks if d.get('dev', home) == dv]
            if dv == home:
                self._kl_run(chs, st, Fv, R, beta)
                return
            if dv not in Gd:
                Gd[dv] = Ragged(self.G.layout, mgpu.replicate(self.G.data))
            st_ = dict(T=self._dev_rep(dv, 'kl.T', st['T']), G=Gd[dv])
            Fv_ = [self._dev_out(dv, ('kl.Fv', s), x.layout) for s, x in enumerate(Fv)]
            R_ = self._dev_out(dv, 'kl.R', R.layout)
            self._kl_run(chs, st_, Fv_, R_, self._dev_vec(dv, 'kl.beta', beta))
            res[dv] = Fv_ + [R_]
        mgpu.run_per_device(work, devs)
        for outs in res.values():
            for D, S_ in zip(list(Fv) + [R], outs):
                for g in range(len(D.data)):
                    if D.data[g].size:
                        D.data[g] += cp.asarray(S_.data[g])

    def _df_all(self, T, T1, Tn, Fv, Fbov, R, R1, beta):
        """DF terms of all groups; with several GPUs every device runs its
        own groups on replicas of T, Tn and its outputs are added into the
        home device's."""
        home = cp.cuda.Device().id
        devs = sorted({d.get('dev', home) for d in self.dfg})
        if devs in ([home], []):
            for gi, d in enumerate(self.dfg):
                self._df_run(gi, d, T, T1, Tn, Fv, Fbov, R, R1, beta)
            return
        from electrolites.dlpno import mgpu
        res = {}
        # the replicas are copied on the other devices' streams
        cp.cuda.get_current_stream().synchronize()

        def work(dv):
            idx = [gi for gi, d in enumerate(self.dfg) if d.get('dev', home) == dv]
            if dv == home:
                for gi in idx:
                    self._df_run(gi, self.dfg[gi], T, T1, Tn, Fv, Fbov, R, R1, beta)
                return
            T_, T1_ = self._dev_rep(dv, 'df.T', T), self._dev_rep(dv, 'df.T1', T1)
            Tn_ = [self._dev_rep(dv, ('df.Tn', s), x) for s, x in enumerate(Tn)]
            Fv_ = [self._dev_out(dv, ('df.Fv', s), x.layout) for s, x in enumerate(Fv)]
            Fb_ = [self._dev_out(dv, ('df.Fb', s), x.layout) for s, x in enumerate(Fbov)]
            R_, R1_ = self._dev_out(dv, 'df.R', R.layout), self._dev_out(dv, 'df.R1', R1.layout)
            b_ = self._dev_vec(dv, 'df.beta', beta, zero=True)
            for gi in idx:
                self._df_run(gi, self.dfg[gi], T_, T1_, Tn_, Fv_, Fb_, R_, R1_, b_)
            res[dv] = (Fv_ + Fb_ + [R_, R1_], b_)
        mgpu.run_per_device(work, devs)
        dst = list(Fv) + list(Fbov) + [R, R1]
        for outs, b_ in res.values():
            for D, S_ in zip(dst, outs):
                for g in range(len(D.data)):
                    if D.data[g].size:
                        D.data[g] += cp.asarray(S_.data[g])
            beta += cp.asarray(b_)

    def _df_run(self, gi, d, T, T1, Tn, Fv, Fbov, R, R1, beta):
        if self.offload:
            from electrolites.dlpno.devmem import to_dev
            d = dict(d)
            for s in (0, 1):
                for key in ('Qov', 'Qvv', 'Qmz', 'Qaz'):
                    x = d[(key, s)]
                    d[(key, s)] = x.device() if isinstance(x, _Packed) else to_dev(x)
        mm = cp.matmul
        tr = lambda X: X.swapaxes(-1, -2)
        Tc, Qov, Qvv, Fbo, Fbv, Qvt, Bk, Ba = {}, {}, {}, {}, {}, {}, {}, {}
        gam = 0.
        g = len(d['mem'])
        for s in (0, 1):
            Tc[s] = cp.zeros((g, d['R'][s], d['V'][s]))
            for gs, mi, sl, pr, pc in d[('tn', s)]:
                Tc[s][mi, :pr, :pc] = Tn[s].data[gs][sl]
            Qov[s], Qvv[s] = d[('Qov', s)], d[('Qvv', s)]
            gam = gam + _dot2(Qov[s], Tc[s])
        for s in (0, 1):
            tc = Tc[s][:, None]
            X = mm(Qvv[s], tr(tc))                                       # (g, A, V, R)
            fbv = _wsum(gam, Qvv[s]) - mm(X, Qov[s]).sum(axis=1)
            ev = d[('ev', s)]
            fbv = fbv + ev[:, :, None] * cp.eye(ev.shape[1])[None]
            Y = mm(Qov[s], tr(tc))                                       # (g, A, R, R)
            fbo = d[('fov', s)] + _wsum(gam, Qov[s]) - mm(Y, Qov[s]).sum(axis=1)
            Fbv[s], Fbo[s] = fbv, fbo
            fvv = fbv - mm(tr(Tc[s]), fbo)
            for gs, mi, sl, pr, pc in d[('fv', s)]:
                Fv[s].data[gs][sl] = fvv[mi, :pr, :pc]
            for gs, mi, sl, pr, pc in d[('tn', s)]:
                Fbov[s].data[gs][sl] = fbo[mi, :pr, :pc]
            Qvt[s] = Qvv[s] - mm(tr(tc), Qov[s])
            # dressed (Q|K z) and (Q|A z) for z = x, y
            g = Tc[s].shape[0]
            zrow, zm = d[('zrow', s)], d[('zm', s)]
            ar = cp.arange(g)
            for side in (0, 1):
                tz = Tc[s][ar, zrow[:, side]] * zm[:, side][:, None]       # (g, V)
                bk = d[('Qmz', s)][:, side] + _mv(Qov[s], tz)
                Bk[(side, s)] = bk
                Ba[(side, s)] = (d[('Qaz', s)][:, side] - mm(bk, Tc[s])
                                 + _mv(Qvv[s], tz)) * zm[:, side][:, None, None]
        # residual DF terms and beta of the strong spin-orbital pairs
        for cid, cb in enumerate(d['combos']):
            si_, a, sj_, b = cb['key']
            m = cb['m']
            pr, pc = cb['shp']
            t = cp.zeros((len(cb['o']), d['V'][a], d['V'][b]))
            t[:, :pr, :pc] = T.data[cb['gT']][cb['sT']]
            ba, bb = Ba[(si_, a)][m], Ba[(sj_, b)][m]
            M = mm(tr(ba), bb)
            if a == b:
                M = M - mm(tr(Ba[(sj_, a)][m]), Ba[(si_, a)][m])
            M = M + mm(mm(Qvt[a][m], t[:, None]), tr(Qvt[b][m])).sum(axis=1)
            R.data[cb['gT']][cb['sT']] += M[:, :pr, :pc]
            # beta matrices
            W1 = mm(tr(Bk[(si_, a)][m]), Bk[(sj_, b)][m])                  # (n, Ra, Rb)
            W2 = mm(tr(Bk[(sj_, b)][m]), Bk[(si_, a)][m])                  # (n, Rb, Ra)
            Q1 = mm(mm(Qov[a][m], t[:, None]), tr(Qov[b][m])).sum(axis=1)
            mats = {0: W1 + Q1 - (W2 if a == b else 0.)}
            if a != b:
                Q2 = -mm(mm(Qov[b][m], tr(t)[:, None]), tr(Qov[a][m])).sum(axis=1)
                mats[1] = -W2 + Q2
            for which, Mx in mats.items():
                ent = self.beta_src.get((gi, cid, which))
                if ent is None:
                    continue
                n_, r_, k_, l_ = ent
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
