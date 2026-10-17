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
        chunks = []
        for Pv in np.unique(P):
            sel = [p for p, pv in zip(canon, P) if pv == Pv]
            # neighbouring pairs share aux atoms: order by the fitting domain
            sel.sort(key=lambda p: (int(self.strong[p]), tuple(self.riatoms[p][:3])))
            ch, tot = [], 0
            for p in sel:
                A = -(-sum(len(self.dfi.atom_ribfs[a]) for a in self.riatoms[p]) // 32) * 32
                L, V = int(padded(len(self.lmos[p]))), int(padded(self.X[p].shape[1]))
                per = A * (A + 4 * L + 4 * V + 2 * L * V + 2 * V * V + 2 * L * Pv)
                if self.strong[p]:
                    E = int(padded(len(self._ext[p])))
                    per += 4 * L * V * E + 2 * A * L * E
                per *= 8
                if ch and tot + per > self.MAX_INT_CHUNK_BYTES:
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
                # the CC driver reads the integrals on the home device
                with cp.cuda.Device(home):
                    out = mgpu.replicate(out)
            for name, d in out.items():
                lst = getattr(self, name)
                for p, v in d.items():
                    lst[p] = v
        chunks.sort(key=lambda c: -len(c[0]))
        ctxs = mgpu.run_queue(chunks, work, devs, setup=lambda d: mgpu.device_view(
            self, ('S_pao_d',), sub={'dfi': ('qia', 'qij', 'J2c')}))
        for c in ctxs:
            if getattr(c, '_prof', None) is not None:
                c._prof.report()
        self._ext = None
        if getattr(self, 'verbose', 0) >= 2:
            if getattr(self.dfi, 'qab_host', False):
                print('[mem] (Q|uv) cache', self.dfi.cache_stats, flush=True)
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
            prof('qab_get')
            qa = dfi.qab_get(a)
            prof('vv')
            if qa is None:
                continue
            na = qa.shape[0]
            rows = np.array([r for r in range(B) if a in offs[r]], np.int64)
            off = np.array([offs[r][a] for r in rows], np.int64)
            pos = ext.pao_pos[a][pidx[rows]]
            pos = np.where(pm[rows] > 0, pos, 0)
            assert (pos >= 0).all(), 'pair PAO outside the extended domain'
            gr, gpos = cp.asarray(rows), cp.asarray(pos)
            step = max(1, int((1 << 28) // (8 * na * P * P)))
            for r0 in range(0, len(rows), step):
                sl = slice(r0, r0 + step)
                g_ = gr[sl]
                blk = gather_blocks(qa, gpos[sl], gpos[sl], gpm[g_], gpm[g_])
                xr = X[g_]
                oo = cp.asarray(off[sl])[:, None] + cp.arange(na)[None, :]
                q_vv[g_[:, None], oo] = cp.matmul(cp.matmul(_tr(xr)[:, None], blk), xr[:, None])
                del blk
            srows = rows[st[rows]]
            if not len(srows):
                continue
            prof('Kio')
            soff = np.array([offs[r][a] for r in srows], np.int64)
            sr = srank[srows]
            ppos = ext.pao_pos[a][pidx[srows]]
            ppos = np.where(pm[srows] > 0, ppos, 0)
            epos = ext.pao_pos[a][eidx[sr]]
            epos = np.where(emk[sr] > 0, epos, 0)
            assert (epos >= 0).all(), 'extended PAO outside the aux atom domain'
            gsr, gpp, gep, gsr2 = cp.asarray(srows), cp.asarray(ppos), cp.asarray(epos), cp.asarray(sr)
            step = max(1, int((1 << 28) // (8 * na * max(P, V) * E)))
            for r0 in range(0, len(srows), step):
                sl = slice(r0, r0 + step)
                g_, s_ = gsr[sl], gsr2[sl]
                blk = gather_blocks(qa, gpp[sl], gep[sl], gpm[g_], gem[s_])      # (b, na, P, E)
                b_ = blk.shape[0]
                ve = cp.matmul(_tr(X[g_])[:, None], blk).reshape(b_, na, V * E)        # (b, na, V E)
                del blk
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
            r = srank[s]
            e_ = self._ext[p]
            ne = len(e_)
            pos = -np.ones(self.orb.nbf, dtype=np.int64)
            pos[e_] = np.arange(ne)
            lm_a = np.asarray(self.lmos[p], np.int64)
            kjs, iks = pl.ij[lm_a, j], pl.ij[i, lm_a]
            Jp, Kp = self._proj_batch(Kio[r, :nl, :nv, :ne], Koviv[r, :nl, :ne, :nv], kjs, pos)
            out['J_ikac'][p], out['K_iakc'][p] = Jp, Kp
            if q != p:
                Jq, Kq = self._proj_batch(Kjo[r, :nl, :nv, :ne], Kovjv[r, :nl, :ne, :nv], iks, pos)
                out['J_ikac'][q], out['K_iakc'][q] = Jq, Kq
            vals = [cc(q_io[s, :na_, :nl]), cc(q_iv[s, :na_, :nv]), cc(q_ov[s, :na_, :nl, :nv]),
                    cc(q_vv[s, :na_, :nv, :nv])]
            vq = [cc(q_jo[s, :na_, :nl]), cc(q_jv[s, :na_, :nv])]
            if self.df_offload:
                from electrolites.dlpno.devmem import to_pinned
                vals = [to_pinned(x) for x in vals]
                vq = [to_pinned(x) for x in vq]
            out['iQk'][p], out['iQa'][p], out['Qma'][p], out['Qab'][p] = vals
            if q != p:
                out['iQk'][q], out['iQa'][q] = vq
                out['Qma'][q], out['Qab'][q] = vals[2], vals[3]
        prof(None)
        return out
