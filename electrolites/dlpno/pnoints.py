"""Closed-shell PNO-basis integrals for the LCCSD, batched over pairs.

Pairs (canonical, i <= j) are processed in chunks of equal padded PAO-domain
size.  Per chunk: one batched metric factorisation, (Q|i k), (Q|i a),
(Q|k a) gathered from the LMO/PAO tensors, and the 3-external (Q|ab) and the
strong pairs' (k i|a e) (e in the extended PAO domain) accumulated one
auxiliary atom at a time straight from the per-atom (Q|uv) blocks
(``DFInts.qab_get``: device resident or streamed from pinned host memory),
so no (Q|a e) tensor is ever formed.  Chunks are distributed over the
visible GPUs (``mgpu``).
"""
import numpy as np
import cupy as cp
from electrolites.dlpno.gather import blocks as gather_blocks
from electrolites.dlpno.linalg import chol_inv_batched

from electrolites.dlpno import setup as su
from electrolites.dlpno.ragged import padded


def _tr(Y):
    return Y.swapaxes(-1, -2)


class PNOIntsGPU:
    """Mixin for rhf.DLPNO."""

    MAX_INT_CHUNK_BYTES = 1 << 30

    def _pno_chunks(self):
        pl = self.pl
        canon = [p for p, (i, j) in enumerate(pl.pairs) if i <= j]
        self._ext = {}
        for p in canon:
            if self.strong[p]:
                self._ext[p] = su.union(self.pao[p], *[self.dom.lmo_paos[k] for k in self.lmos[p]])
        npao = np.array([len(self.pao[p]) for p in canon])
        P = padded(npao)
        cap = self.MAX_INT_CHUNK_BYTES
        chunks = []
        for Pv in np.unique(P):
            sel = [p for p, pv in zip(canon, P) if pv == Pv]
            # neighbouring pairs share aux atoms: order by the fitting domain
            # neighbouring pairs (Morton order of their fitting domains) share aux atoms
            o_ = self.dfi.spatial_order([self.riatoms[p] for p in sel])
            sel = [sel[k] for k in o_]
            ch, tot = [], 0
            for p in sel:
                A = -(-sum(len(self.dfi.atom_ribfs[a]) for a in self.riatoms[p]) // 32) * 32
                L, V = int(padded(len(self.lmos[p]))), int(padded(self.X[p].shape[1]))
                per = A * (A + 4 * L + 4 * V + 2 * L * V + 2 * V * V + 2 * L * Pv)
                if self.strong[p]:
                    E = int(padded(len(self._ext[p])))
                    per += 4 * L * V * E + 2 * A * L * E
                per *= 8
                if ch and tot + per > cap:
                    chunks.append((ch, int(Pv)))
                    ch, tot = [], 0
                ch.append(p)
                tot += per
            if ch:
                chunks.append((ch, int(Pv)))
        return chunks

    def pno_integrals_batched(self):
        from electrolites.dlpno import mgpu
        self.df_offload = self._decide_df_offload()
        npair = len(self.pl)
        for name in ('K_mibj', 'J_ijmb', 'K_ivvv', 'L_iajb', 'J_ikac', 'K_iakc',
                     'iQk', 'iQa', 'Qma', 'Qab'):
            setattr(self, name, [None] * npair)
        chunks = self._pno_chunks()
        devs = mgpu.devices()
        home = cp.cuda.Device().id

        def work(w, it):
            out = w._pno_chunk(*it)
            if cp.cuda.Device().id != home:
                # the CC driver reads the pair integrals on the home device; the
                # strong pairs' DF tensors stay here (their DF groups are spread
                # over the devices anyway)
                with cp.cuda.Device(home):
                    out = {k: (v if k in ('iQk', 'iQa', 'Qma', 'Qab') else mgpu.replicate(v))
                           for k, v in out.items()}
            for name, d in out.items():
                lst = getattr(self, name)
                for p, v in d.items():
                    lst[p] = v
        setup = lambda d: mgpu.device_view(
            self, ('S_pao_d',), sub={'dfi': ('qia', 'qij', 'J2c', 'C_pao_d', 'mask_uv_d')})
        ctxs = []
        if getattr(self.dfi, 'qab_direct', False):
            # direct (Q|uv): chunks in spatial order, grouped into regions whose
            # (Q|uv) blocks are evaluated once per atom
            order = self.dfi.spatial_order([np.unique(np.concatenate([self.riatoms[p] for p in c[0]]))
                                            for c in chunks])
            chunks = [chunks[k] for k in order]
            needs = []
            for ch, _ in chunks:
                nd = {}
                for p in ch:
                    u = self.pao[p] if not self.strong[p] else np.union1d(self.pao[p], self._ext[p])
                    for a in self.riatoms[p]:
                        nd.setdefault(int(a), []).append(u)
                needs.append({a: np.concatenate(v) for a, v in nd.items()})
            ctxs = self.dfi.run_regions(chunks, needs, work, setup, devs, getattr(self, 'verbose', 0))
        else:
            chunks.sort(key=lambda c: -len(c[0]))
            ctxs = mgpu.run_queue(chunks, work, devs, setup=setup)
        for c in ctxs:
            if getattr(c, '_prof', None) is not None:
                c._prof.report()
        self._ext = None
        if getattr(self, 'verbose', 0) >= 2:
            if (getattr(self.dfi, 'qab_host', False) or getattr(self.dfi, '_shard', None)
                    or getattr(self.dfi, 'qab_direct', False)):
                print('[mem] (Q|uv) cache', {k: (round(v, 1) if isinstance(v, float) else v)
                                            for k, v in self.dfi.cache_stats.items()}, flush=True)
        # the CC build does not read (Q|uv): the device cache (up to a quarter
        # of the device) is refilled by the triples
        self.dfi.cache_clear()
        for d in devs:
            with cp.cuda.Device(d):
                cp.get_default_memory_pool().free_all_blocks()
            sz = {}
            for name in ('K_mibj', 'J_ijmb', 'K_ivvv', 'J_ikac', 'K_iakc', 'iQk', 'iQa', 'Qma', 'Qab'):
                seen, tot = set(), 0
                for v in getattr(self, name):
                    for x in (v if isinstance(v, list) else [v]):
                        if x is not None and id(x) not in seen:
                            seen.add(id(x))
                            tot += x.nbytes
                sz[name] = round(tot / 1e9, 2)
            print('[mem] PNO integrals GB', sz, 'DF offload', self.df_offload, flush=True)
        for p, (i, j) in enumerate(self.pl.pairs):
            if i <= j:
                q = self.pl.ji[p]
                K = self.K[p]
                self.L_iajb[p] = 2 * K - K.T
                if q != p:
                    self.L_iajb[q] = self.L_iajb[p].T

    def _pno_chunk(self, ch, P):
        """PNO-basis integrals of the canonical pairs ``ch`` (padded PAO
        domain P).  Returns {list name: {pair: array}}."""
        pl, dfi, ext = self.pl, self.dfi, self.dfi.ext
        B = len(ch)
        from electrolites.dlpno.devmem import Prof
        prof = self.__dict__.setdefault('_prof', Prof('pno_ints'))
        prof('setup')
        auxs = [dfi.aux_index(self.riatoms[p]) for p in ch]
        nla = np.array([len(self.lmos[p]) for p in ch])
        nva = np.array([self.X[p].shape[1] for p in ch])
        A = int(-(-max(len(a) for a in auxs) // 32) * 32)
        L, V = int(padded(nla.max())), int(padded(nva.max()))
        ii = np.array([pl.pairs[p][0] for p in ch])
        jj = np.array([pl.pairs[p][1] for p in ch])
        pidx, pm = np.zeros((B, P), np.int64), np.zeros((B, P))
        lidx, lmk = np.zeros((B, L), np.int64), np.zeros((B, L))
        aidx, am = np.zeros((B, A), np.int64), np.zeros((B, A))
        Xh = np.zeros((B, P, V))
        offs = []
        for s, p in enumerate(ch):
            n_ = len(self.pao[p])
            pidx[s, :n_], pm[s, :n_] = self.pao[p], 1.
            lidx[s, :nla[s]], lmk[s, :nla[s]] = self.lmos[p], 1.
            aidx[s, :len(auxs[s])], am[s, :len(auxs[s])] = auxs[s], 1.
            Xh[s, :n_, :nva[s]] = self.X[p]
            o, at = 0, {}
            for a in self.riatoms[p]:
                at[int(a)] = o
                o += len(dfi.atom_ribfs[a])
            offs.append(at)
        gp, gpm = cp.asarray(pidx), cp.asarray(pm)
        gl, glm = cp.asarray(lidx), cp.asarray(lmk)
        ga, gam = cp.asarray(aidx), cp.asarray(am)
        X = cp.asarray(Xh)
        prof('metric')
        J = dfi.J2c[ga[:, :, None], ga[:, None, :]] * (gam[:, :, None] * gam[:, None, :])
        J += cp.eye(A)[None] * (1 - gam)[:, :, None]
        Linv = chol_inv_batched(J)
        del J

        def H(x, Li=Linv):           # half fit  L^-1 x
            return cp.matmul(Li, x.reshape(x.shape[0], A, -1)).reshape(x.shape)

        def C(x, Li=Linv):           # full fit  J^-1 x
            return cp.matmul(_tr(Li), H(x, Li).reshape(x.shape[0], A, -1)).reshape(x.shape)
        gi, gj = cp.asarray(ii), cp.asarray(jj)
        prof('gather_o')
        q_pair = dfi.qij[ga, gi[:, None], gj[:, None]] * gam                       # (B, A)
        mo = gam[:, :, None] * glm[:, None, :]
        q_io = dfi.qij[ga[:, :, None], gi[:, None, None], gl[:, None, :]] * mo       # (B, A, L)
        q_jo = dfi.qij[ga[:, :, None], gj[:, None, None], gl[:, None, :]] * mo
        mp = gam[:, :, None] * gpm[:, None, :]
        prof('gather_v')
        q_iv = cp.matmul(dfi.qia[ga[:, :, None], gi[:, None, None], gp[:, None, :]] * mp, X)
        q_jv = cp.matmul(dfi.qia[ga[:, :, None], gj[:, None, None], gp[:, None, :]] * mp, X)
        prof('gather_ov')
        q_ov = cp.matmul(dfi.qia[ga[:, :, None, None], gl[:, None, :, None], gp[:, None, None, :]]
                         * (mo[:, :, :, None] * gpm[:, None, None, :]), X[:, None])   # (B, A, L, V)
        st = np.array([bool(self.strong[p]) for p in ch])
        srow = np.where(st)[0]
        ns = len(srow)
        srank = -np.ones(B, np.int64)
        srank[srow] = np.arange(ns)
        prof('fit')
        if ns:
            q_io_c, q_jo_c = C(q_io), C(q_jo)
            E = int(padded(max(len(self._ext[ch[s]]) for s in srow)))
            eidx, emk = np.zeros((ns, E), np.int64), np.zeros((ns, E))
            for r, s in enumerate(srow):
                e_ = self._ext[ch[s]]
                eidx[r, :len(e_)], emk[r, :len(e_)] = e_, 1.
            gem = cp.asarray(emk)
            Kio = cp.zeros((ns, L, V, E))
            Kjo = cp.zeros((ns, L, V, E))
        q_vv = cp.zeros((B, A, V, V))
        atoms = sorted({a for o_ in offs for a in o_})
        for a in atoms:
            rows = np.array([r for r in range(B) if a in offs[r]], np.int64)
            off = np.array([offs[r][a] for r in rows], np.int64)
            need = [pidx[rows][pm[rows] > 0]]
            if ns:
                sr_ = srank[rows[st[rows]]]
                need.append(eidx[sr_][emk[sr_] > 0])
            prof('qab_get')
            qa, pmap = dfi.qab_block(a, need)
            prof('vv')
            if qa is None:
                continue
            na = qa.shape[0]
            pos = pmap[pidx[rows]]
            pos = np.where(pm[rows] > 0, pos, 0)
            assert (pos >= 0).all(), 'pair PAO outside the extended domain'
            gr, gpos = cp.asarray(rows), cp.asarray(pos)
            step = max(1, int((1 << 28) // (8 * na * P * P)))
            for r0 in range(0, len(rows), step):
                sl = slice(r0, r0 + step)
                g_ = gr[sl]
                # (b, u, q, v) layout: both contractions are single large GEMMs
                blk = gather_blocks(qa, gpos[sl], gpos[sl], gpm[g_], gpm[g_], ru=True)
                b_ = blk.shape[0]
                xr = X[g_]
                Y = cp.matmul(_tr(xr), blk.reshape(b_, P, na * P))                  # (b, V, na*P)
                del blk
                Y = cp.matmul(Y.reshape(b_, V * na, P), xr).reshape(b_, V, na, V)    # (b, V, na, V)
                oo = cp.asarray(off[sl])[:, None] + cp.arange(na)[None, :]
                q_vv[g_[:, None], oo] = Y.transpose(0, 2, 1, 3)
                del Y
            srows = rows[st[rows]]
            if not len(srows):
                continue
            prof('Kio')
            soff = np.array([offs[r][a] for r in srows], np.int64)
            sr = srank[srows]
            ppos = pmap[pidx[srows]]
            ppos = np.where(pm[srows] > 0, ppos, 0)
            epos = pmap[eidx[sr]]
            epos = np.where(emk[sr] > 0, epos, 0)
            assert (epos >= 0).all(), 'extended PAO outside the aux atom domain'
            gsr, gpp, gep, gsr2 = cp.asarray(srows), cp.asarray(ppos), cp.asarray(epos), cp.asarray(sr)
            step = max(1, int((1 << 28) // (8 * na * max(P, V) * E)))
            for r0 in range(0, len(srows), step):
                sl = slice(r0, r0 + step)
                g_, s_ = gsr[sl], gsr2[sl]
                blk = gather_blocks(qa, gpp[sl], gep[sl], gpm[g_], gem[s_], ru=True)   # (b, P, na, E)
                b_ = blk.shape[0]
                ve = cp.matmul(_tr(X[g_]), blk.reshape(b_, P, na * E))                # (b, V, na*E)
                del blk
                ve = cp.ascontiguousarray(ve.reshape(b_, V, na, E).transpose(0, 2, 1, 3)).reshape(b_, na, V * E)
                oo = cp.asarray(soff[sl])[:, None] + cp.arange(na)[None, :]
                Kio[s_] += cp.matmul(_tr(q_io_c[g_[:, None], oo]), ve).reshape(b_, L, V, E)
                Kjo[s_] += cp.matmul(_tr(q_jo_c[g_[:, None], oo]), ve).reshape(b_, L, V, E)
                del ve
        prof('rest')
        if ns:
            del q_io_c, q_jo_c
            # (k a'| i a) with a' over the extended domain
            sg = cp.asarray(srow)
            ge = cp.asarray(eidx)
            q_iv_c, q_jv_c = C(q_iv[sg], Linv[sg]), C(q_jv[sg], Linv[sg])
            mx = gam[sg][:, :, None, None] * glm[sg][:, None, :, None] * gem[:, None, None, :]
            q_ovx = (dfi.qia[ga[sg][:, :, None, None], gl[sg][:, None, :, None], ge[:, None, None, :]]
                     * mx).reshape(ns, A, L * E)
            del mx
            Koviv = cp.matmul(_tr(q_ovx), q_iv_c).reshape(ns, L, E, V)
            Kovjv = cp.matmul(_tr(q_ovx), q_jv_c).reshape(ns, L, E, V)
            del q_ovx, q_iv_c, q_jv_c
        q_pair = H(q_pair[:, :, None])[:, :, 0]
        q_io, q_jo, q_iv, q_jv, q_ov, q_vv = (H(x) for x in (q_io, q_jo, q_iv, q_jv, q_ov, q_vv))
        K_mibj = cp.matmul(_tr(q_io), q_jv)                         # (B, L, V)
        K_mjbi = cp.matmul(_tr(q_jo), q_iv)
        J_ijmb = cp.einsum('bA,bAlv->blv', q_pair, q_ov)
        K_ivvv = cp.matmul(_tr(q_iv), q_vv.reshape(B, A, V * V)).reshape(B, V, V, V)
        K_jvvv = cp.matmul(_tr(q_jv), q_vv.reshape(B, A, V * V)).reshape(B, V, V, V)
        prof('out')
        proj = self._proj_chunk(ch, st, srank, nla, nva, Kio, Koviv, Kjo, Kovjv) if ns else {}
        names = ('K_mibj', 'J_ijmb', 'K_ivvv', 'J_ikac', 'K_iakc', 'iQk', 'iQa', 'Qma', 'Qab')
        out = {k: {} for k in names}
        cc = cp.ascontiguousarray
        for s, p in enumerate(ch):
            i, j = pl.pairs[p]
            q = pl.ji[p]
            na_, nl, nv = len(auxs[s]), nla[s], nva[s]
            out['K_mibj'][p] = cc(K_mibj[s, :nl, :nv])
            out['J_ijmb'][p] = cc(J_ijmb[s, :nl, :nv])
            out['K_ivvv'][p] = cc(K_ivvv[s, :nv, :nv, :nv]).reshape(nv, nv * nv)
            if q != p:
                out['K_mibj'][q] = cc(K_mjbi[s, :nl, :nv])
                out['J_ijmb'][q] = out['J_ijmb'][p]
                out['K_ivvv'][q] = cc(K_jvvv[s, :nv, :nv, :nv]).reshape(nv, nv * nv)
            if not st[s]:
                continue
            out['J_ikac'][p], out['K_iakc'][p] = proj[(p, 0)]
            if q != p:
                out['J_ikac'][q], out['K_iakc'][q] = proj[(p, 1)]
            vals = [cc(q_io[s, :na_, :nl]), cc(q_iv[s, :na_, :nv]), cc(q_ov[s, :na_, :nl, :nv]),
                    cc(q_vv[s, :na_, :nv, :nv])]
            vq = [cc(q_jo[s, :na_, :nl]), cc(q_jv[s, :na_, :nv])]
            if self.df_offload:
                # pageable host copies: freed pinned memory is not returned to
                # the OS (CUDA keeps it for reuse), so the grouped copies the CC
                # driver makes (pinned) would add to these instead of replacing them
                vals = [cp.asnumpy(x) for x in vals]
                vq = [cp.asnumpy(x) for x in vq]
            out['iQk'][p], out['iQa'][p], out['Qma'][p], out['Qab'][p] = vals
            if q != p:
                out['iQk'][q], out['iQa'][q] = vq
                out['Qma'][q], out['Qab'][q] = vals[2], vals[3]
        prof(None)
        return out

    def _xcompact(self):
        """device store of every canonical pair's PNO coefficients in its PAO
        domain (one Ragged), and an item -> pointer table (per device)."""
        dev = cp.cuda.Device().id
        c = self.__dict__.setdefault('_xc_dev', {})
        if dev not in c:
            from electrolites.dlpno.ragged import Layout, Ragged
            pl = self.pl
            npair = len(pl)
            canon = np.array([p if pl.pairs[p][0] <= pl.pairs[p][1] else pl.ji[p] for p in range(npair)])
            lay = Layout(np.array([len(self.pao[p]) for p in range(npair)]),
                         np.array([self.X[p].shape[1] for p in range(npair)]))
            R = Ragged(lay)
            for g, m in enumerate(lay.members):
                pr, pc = lay.shapes[g]
                buf = np.zeros((len(m), pr, pc))
                for s_, p in enumerate(m):
                    x = self.X[canon[p]]
                    buf[s_, :x.shape[0], :x.shape[1]] = x
                R.data[g] = cp.asarray(buf)
            shp = np.array(lay.shapes, np.int64).reshape(-1, 2)
            base = np.array([d.data.ptr for d in R.data], np.int64)
            ptr = base[lay.group] + 8 * lay.slot * shp[lay.group, 0] * shp[lay.group, 1]
            c[dev] = (R, ptr, shp[lay.group, 1].astype(np.int32))
        return c[dev]

    def _proj_chunk(self, ch, st, srank, nla, nva, Kio, Koviv, Kjo, Kovjv):
        """(J_ikac, K_iakc) item lists of every strong pair of the chunk: the
        (k a | i e') / (k e' | i a) blocks projected onto the PNOs of the
        (k, j) / (i, k) pairs, all items of the chunk in a few batches."""
        from electrolites.dlpno.gather import gather2d
        pl = self.pl
        Xs, xptr, xld = self._xcompact()
        items = []                      # (key, pair p, side, rank r, row kk, target pair t, ext pos)
        for s_, p in enumerate(ch):
            if not st[s_]:
                continue
            i, j = pl.pairs[p]
            q = pl.ji[p]
            e_ = self._ext[p]
            pos = -np.ones(self.orb.nbf, dtype=np.int64)
            pos[e_] = np.arange(len(e_))
            lm_a = np.asarray(self.lmos[p], np.int64)
            for side, tg in ((0, pl.ij[lm_a, j]), (1, pl.ij[i, lm_a])):
                if side == 1 and q == p:
                    continue
                for kk, t in enumerate(tg):
                    items.append((p, side, srank[s_], kk, int(t), pos, nva[s_]))
        out = {}
        if not items:
            return out
        n_t = np.array([len(self.pao[it[4]]) for it in items])
        v_t = np.array([self.X[it[4]].shape[1] for it in items])
        E = Kio.shape[3]
        V = Kio.shape[2]
        Pq, Vq = int(n_t.max()), int(v_t.max())
        step = max(1, int((1 << 29) // (8 * (V * E + 2 * Pq * (V + Vq) + E * V))))
        res = [None] * len(items)
        for c0 in range(0, len(items), step):
            its = items[c0:c0 + step]
            m = len(its)
            cols = np.zeros((m, Pq), np.int64)
            msk = np.zeros((m, Pq))
            for r_, it in enumerate(its):
                c_ = it[5][self.pao[it[4]]]
                cols[r_, :len(c_)] = c_
                msk[r_, :len(c_)] = 1.
            gc, gm = cp.asarray(cols), cp.asarray(msk)
            tt = np.array([it[4] for it in its], np.int64)
            X = gather2d(cp.asarray(xptr[tt]), cp.asarray(xld[tt]), cp.asarray(n_t[c0:c0 + m].astype(np.int32)),
                         cp.asarray(v_t[c0:c0 + m].astype(np.int32)), Pq, Vq)        # (m, Pq, Vq)
            rr = cp.asarray(np.array([it[2] for it in its], np.int64))
            kk = cp.asarray(np.array([it[3] for it in its], np.int64))
            sd = np.array([it[1] for it in its])
            s1 = cp.asarray(sd == 1)
            Ka = cp.where(s1[:, None, None], Kjo[rr, kk], Kio[rr, kk])                   # (m, V, E)
            Kb = cp.where(s1[:, None, None], Kovjv[rr, kk], Koviv[rr, kk])               # (m, E, V)
            A = cp.take_along_axis(Ka, gc[:, None, :], axis=2) * gm[:, None, :]         # (m, V, Pq)
            J = cp.matmul(A, X)                                                          # (m, V, Vq)
            B = cp.take_along_axis(Kb, gc[:, :, None], axis=1) * gm[:, :, None]         # (m, Pq, V)
            K = cp.matmul(X.transpose(0, 2, 1), B).transpose(0, 2, 1)                    # (m, V, Vq)
            for r_, it in enumerate(its):
                res[c0 + r_] = (J[r_, :it[6], :v_t[c0 + r_]], K[r_, :it[6], :v_t[c0 + r_]])
        for it, (Jm, Km) in zip(items, res):
            key = (it[0], it[1])
            if key not in out:
                out[key] = ([], [])
            out[key][0].append(Jm)
            out[key][1].append(Km)
        return out
