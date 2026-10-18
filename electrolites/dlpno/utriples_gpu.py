"""Open-shell DLPNO-(T0) and iterative (T) on the device, spin blocked.

Equations: ``ref_uhf`` (spin-orbital (T) in the triplet's alpha / beta TNO
spaces, alpha = TNOs, beta = TNOs + SOMOs).  Instead of the reference's
dense (alpha + beta)^3 tensors, every spin triple (I, J, K) -- one spin
orbital per position of a spatial triplet (i <= j <= k) -- keeps only its
*aligned* virtual block (a of spin(I), b of spin(J), c of spin(K)); the
other virtual spin blocks are axis permutations of it, so

    E_IJK = m * sum_abc T(a,b,c) (W + V)(a,b,c) / 6,   m = 1 (same spins) or 3.

W = P(i/jk) P(a/bc) w(i,j,k), V likewise; each of the nine (occupied,
virtual) permutations is evaluated for the spin blocks it actually has, so
the work is that of the alpha/beta blocks, not of the dense space.

Triplets are processed in chunks of equal padded PAO-domain size; within a
chunk the spin triples of one spin pattern (spins of the three positions)
form one batch.  TNOs, the metric, the three-index integrals in the TNO
basis and the projected pair amplitudes are batched over the chunk.
"""
import itertools

import numpy as np
import cupy as cp
from electrolites.dlpno.gather import blocks as gather_blocks, xtbx
from electrolites.dlpno.linalg import chol_inv_batched
import cupyx

from electrolites.dlpno.ragged import padded
from electrolites.dlpno.linalg import eigh_batched, orthonormal_domain

_OCC = [((0, 1, 2), 1.), ((1, 0, 2), -1.), ((2, 1, 0), -1.)]
_VIR = _OCC


class UTriplesGPU:
    """Mixin for uhf.UDLPNO replacing tno_transform / lccsd_t0 /
    lccsd_t_iterations of the host reference."""

    MAX_CHUNK_BYTES = None

    def _chunk_bytes(self):
        """batch budget of the triplet chunks: an eighth of the free device memory (>= 1 GB)."""
        if self.MAX_CHUNK_BYTES:
            return self.MAX_CHUNK_BYTES
        free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        return max(1 << 30, int(free // 8))

    # ------------------------------------------------------------------
    # pair tables on the device
    # ------------------------------------------------------------------
    def _so_tb(self, K, L, a, b):
        sK, sL = self.so[K][1], self.so[L][1]
        if (a, b) == (sK, sL):
            return self._tsidx.get((K, L)), 1.
        if sK != sL and (a, b) == (sL, sK):
            return self._tsidx.get((L, K)), -1.
        return None, 0.

    def _sidx_tables(self):
        """SIDX[I, J]: row of spin-orbital pair (I, J) in the amplitude table
        (-1 if absent); SOF[lmo, spin]: spin orbital (-1 if absent)."""
        cached = getattr(self, '_sidx_cache', None)
        if cached is not None and cached[0] is self._tsidx:
            return cached[1], cached[2]
        nso = len(self.so)
        SIDX = -np.ones((nso, nso), np.int64)
        for k, n in self._tsidx.items():
            SIDX[k[0], k[1]] = n
        SOF = -np.ones((self.n, 2), np.int64)
        for I, (x, s) in enumerate(self.so):
            SOF[x, s] = I
        self._sidx_cache = (self._tsidx, SIDX, SOF)
        return SIDX, SOF

    def _upair_tables(self):
        pl, ns = self.pl, self.ns
        npair = len(pl)
        nva = np.array([self.nva(p) for p in range(npair)])
        nvb = np.array([self.nvb(p) for p in range(npair)])
        npo = np.array([len(p) for p in self.pao[:npair]])
        npn = np.array([self.Xpno[p].shape[1] for p in range(npair)])
        VA, VB, PP, VN = (int(padded(x.max())) for x in (nva, nvb, npo, npn))
        XA = np.zeros((npair, PP, VA))
        XB = np.zeros((npair, PP + ns, VB))
        XN = np.zeros((npair, PP, VN))
        DN = np.zeros((npair, VN, VN))
        dom = np.zeros((npair, PP), np.int64)
        dmk = np.zeros((npair, PP))
        for p in range(npair):
            n_ = npo[p]
            XA[p, :n_, :nva[p]] = self.Xa[p]
            XB[p, :n_, :nvb[p]] = self.Xb[p][:n_]
            XB[p, PP:, :nvb[p]] = self.Xb[p][n_:]
            XN[p, :n_, :npn[p]] = self.Xpno[p]
            dom[p, :n_] = self.pao[p]
            dmk[p, :n_] = 1.
            if pl.pairs[p][0] <= pl.pairs[p][1]:
                DN[p, :npn[p], :npn[p]] = self._pair_density_blocks(p)
        for p in range(npair):
            if pl.pairs[p][0] > pl.pairs[p][1]:
                DN[p] = DN[pl.ji[p]]
        sop = list(self.T2b)
        self._tsidx = {k: n for n, k in enumerate(sop)}
        VM = max(VA, VB)
        TS = np.zeros((len(sop), VM, VM))
        for n, k in enumerate(sop):
            M = self.T2b[k]
            TS[n, :M.shape[0], :M.shape[1]] = M
        T1 = np.zeros((len(self.so), VM))
        for I, (x, s) in enumerate(self.so):
            dd = pl.ij[x, x]
            t = self.T1[I][self._vslice(dd, s)]
            T1[I, :len(t)] = t
        self._pt = dict(VA=VA, VB=VB, PP=PP, VN=VN, XA=cp.asarray(XA), XB=cp.asarray(XB),
                        XN=cp.asarray(XN), DN=cp.asarray(DN), dom=cp.asarray(dom),
                        dmk=cp.asarray(dmk), TS=cp.asarray(TS), T1=cp.asarray(T1))
        self.Fa_lp_d, self.Fb_lp_d = cp.asarray(self.Fa_lp), cp.asarray(self.Fb_lp)

    def _pair_density_blocks(self, p):
        """spin-averaged pair density of p in its PNO coordinates (block form)."""
        x, y = self.pl.pairs[p]
        paos = self.pao[p]
        npo = len(paos)
        Sd = self.orb.S_pao[np.ix_(paos, paos)]
        Xo = self.Xpno[p]
        npn = Xo.shape[1]
        W = (Xo.T @ Sd @ self.Xa[p], Xo.T @ Sd @ self.Xb[p][:npo])
        si = self.so_index

        def blk(I, J):
            T = self.T2b.get((I, J))
            if T is None:
                return np.zeros((npn, npn))
            return W[self.so[I][1]] @ T @ W[self.so[J][1]].T
        dx, dy = not self.is_somo[x], not self.is_somo[y]
        Z = np.zeros((npn, npn))
        A = blk(si[(x, 0)], si[(y, 0)]) if x != y else Z
        Bm = blk(si[(x, 1)], si[(y, 1)]) if (dx and dy and x != y) else Z
        O = blk(si[(x, 0)], si[(y, 1)]) if dy else Z
        Op = blk(si[(y, 0)], si[(x, 1)]) if dx else Z
        D = .5 * (A @ A.T + Bm @ Bm.T) + .5 * (O @ O.T + O.T @ O + Op @ Op.T + Op.T @ Op)
        if x == y:
            D *= .5
        return D

    def _proj(self, Z, s, bidx, pidx):
        """S(triplet b, pair p)^T for spin s: (n, VP_s, V_s) = X_p^T Z_b[rows of p]."""
        pt = self._pt
        bidx = cp.asarray(bidx)
        pidx = cp.asarray(pidx)
        n = len(pidx)
        V_ = Z[s].shape[2]
        step = max(1, int((1 << 28) // (8 * (pt['PP'] + self.ns) * (max(pt['VA'], pt['VB']) + V_))))
        if n > step:
            # bounded temporaries: the gathered PAO rows are (couples, PP, VP + V)
            parts = [self._proj(Z, s, bidx[c0:c0 + step], pidx[c0:c0 + step]) for c0 in range(0, n, step)]
            return cp.concatenate(parts, axis=0)
        dom, dmk = pt['dom'][pidx], pt['dmk'][pidx]
        if s == 0:
            Zg = Z[0][bidx[:, None], dom] * dmk[:, :, None]
            return cp.matmul(pt['XA'][pidx].transpose(0, 2, 1), Zg)
        nbf = self.orb.nbf
        Zg = Z[1][bidx[:, None], dom] * dmk[:, :, None]
        if self.ns:
            Zs = Z[1][bidx, nbf:]                                     # (n, ns, V)
            Zg = cp.concatenate([Zg, Zs], axis=1)
        return cp.matmul(pt['XB'][pidx].transpose(0, 2, 1), Zg)

    # ------------------------------------------------------------------
    def tno_transform(self, t_cut):
        self._t_cut_tno = t_cut

    def _tno_chunk(self, ch, Pv, t_cut):
        opt, pl = self.opt, self.pl
        pt = self._pt
        ns = self.ns
        S_d = self.S_pao_d
        trip = self._trip_arr
        b = len(ch)
        npao = self._npao_t[ch]
        idx = np.zeros((b, Pv), np.int64)
        msk = np.zeros((b, Pv))
        for s, t in enumerate(ch):
            idx[s, :npao[s]] = self.t_paos[t]
            msk[s, :npao[s]] = 1.
        gi, m = cp.asarray(idx), cp.asarray(msk)
        mm = m[:, :, None] * m[:, None, :]
        St = S_d[gi[:, :, None], gi[:, None, :]] * mm
        X, keep = orthonormal_domain(St, npao, opt['S_CUT'], getattr(self, 'PAO_ORTHO', 'pivchol'))
        Z = cp.matmul(S_d[:, gi].transpose(1, 0, 2) * m[:, None, :], X)      # (b, nbf, Pv)
        Dt = cp.zeros((b, Pv, Pv))
        tr = trip[ch]
        for (a, c) in ((0, 1), (1, 2), (0, 2)):
            pp = cp.asarray(pl.ij[tr[:, a], tr[:, c]])
            Zg = Z[cp.arange(b)[:, None], pt['dom'][pp]] * pt['dmk'][pp][:, :, None]
            sp = cp.matmul(pt['XN'][pp].transpose(0, 2, 1), Zg)              # (b, VN, Pv)
            Dt += cp.matmul(sp.transpose(0, 2, 1), cp.matmul(pt['DN'][pp], sp))
        del Z
        Dt /= 3.
        Dt -= cp.eye(Pv)[None] * (~keep)[:, :, None]
        occ, Vt = eigh_batched(Dt)
        occ, Vt = occ[:, ::-1], Vt[:, :, ::-1]
        nkeep = keep.sum(axis=1)
        cut = cp.asarray(self.tno_scale[ch] * t_cut)
        cnt = cp.sum((cp.abs(occ) >= cut[:, None]) & (cp.arange(Pv)[None, :] < nkeep[:, None]), axis=1)
        nkeep, cnt = nkeep.get(), cnt.get()
        nfin = np.maximum(1, np.maximum(cnt, np.minimum(opt['MIN_TNOS'], nkeep)))
        Nt = int(nfin.max())
        cm = (cp.arange(Nt)[None, :] < cp.asarray(nfin)[:, None]).astype(cp.float64)
        Uq = cp.matmul(X, Vt[:, :, :Nt]) * cm[:, None, :]
        tr_ = lambda Y: Y.transpose(0, 2, 1)
        Fa = self.Fa_pao_d[gi[:, :, None], gi[:, None, :]] * mm
        Fb = self.Fb_pao_d[gi[:, :, None], gi[:, None, :]] * mm
        fa = cp.matmul(tr_(Uq), cp.matmul(Fa, Uq)) + 1e6 * cp.eye(Nt)[None] * (1 - cm)[:, :, None]
        ea, Wa = eigh_batched(fa)
        Va = int(padded(Nt))
        Xa = cp.zeros((b, Pv, Va))
        Xa[:, :, :Nt] = cp.matmul(Uq, Wa) * cm[:, None, :]
        W_ = Nt + ns
        fb = cp.zeros((b, W_, W_))
        fb[:, :Nt, :Nt] = cp.matmul(tr_(Uq), cp.matmul(Fb, Uq))
        if ns:
            Fps = cp.asarray(self.Fb_ps)[gi] * m[:, :, None]
            fps = cp.matmul(tr_(Uq), Fps)
            fb[:, :Nt, Nt:] = fps
            fb[:, Nt:, :Nt] = tr_(fps)
            fb[:, Nt:, Nt:] = cp.asarray(self.Fb_ss)[None]
        mb = cp.concatenate([cm, cp.ones((b, ns))], axis=1)
        fb += 1e6 * cp.eye(W_)[None] * (1 - mb)[:, :, None]
        eb, Ub = eigh_batched(fb)
        nfb = nfin + ns
        cb = (cp.arange(W_)[None, :] < cp.asarray(nfb)[:, None]).astype(cp.float64)
        Vb = int(padded(W_))
        Xbp = cp.zeros((b, Pv, Vb))
        Xbs = cp.zeros((b, ns, Vb))
        Xbp[:, :, :W_] = cp.matmul(Uq, Ub[:, :Nt]) * cb[:, None, :]
        if ns:
            Xbs[:, :, :W_] = Ub[:, Nt:] * cb[:, None, :]
        ea_p = cp.full((b, Va), 1e3)
        ea_p[:, :Nt] = cp.where(cm > 0, ea, 1e3)
        eb_p = cp.full((b, Vb), 1e3)
        eb_p[:, :W_] = cp.where(cb > 0, eb, 1e3)
        return dict(Xa=Xa, Xbp=Xbp, Xbs=Xbs, ea=ea_p, eb=eb_p, nfa=nfin, nfb=nfb, pidx=idx, pm=msk)

    # ------------------------------------------------------------------
    def triples_sparsity(self, prescreen):
        super().triples_sparsity(prescreen)
        self.t_riatoms = [np.unique(self.ribf_atom[r]) for r in self.t_ribfs]

    def _patterns(self, t):
        """spin patterns (s0, s1, s2) of spatial triplet t (positions sorted)."""
        i, j, k = self.trip[t]
        lm = (i, j, k)
        out = []
        for pat in itertools.product((0, 1), repeat=3):
            if any(self.is_somo[lm[r]] and pat[r] == 1 for r in range(3)):
                continue
            ok = True
            for r in range(2):
                if lm[r] == lm[r + 1] and not pat[r] < pat[r + 1]:
                    ok = False
            if ok:
                out.append(pat)
        return out

    def lccsd_t0(self, save=False):
        # the (Q|uv) device caches and pooled blocks of the previous pass go first
        self.dfi.cache_clear()
        self._pt = None                   # the previous pass's pair tables
        import gc
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        if getattr(self, 'verbose', 0) >= 2:
            free, tot = cp.cuda.runtime.memGetInfo()
            print(f'[mem] (T0) pass start: device used {(tot - free) / 1e9:.1f} GB', flush=True)
        self._upair_tables()
        ntrip = len(self.trip)
        self.e_ijk = np.zeros(ntrip)
        self._trip_arr = np.array(self.trip, dtype=np.int64).reshape(-1, 3)
        self._npao_t = np.array([len(p) for p in self.t_paos], np.int64)
        self._nl_t = np.array([len(l) for l in self.t_lmos], np.int64)
        self._naux_t = np.array([len(r) for r in self.t_ribfs], np.int64)
        if save:
            self.st = {}            # (t, pat) -> saved aligned W, V, T, D
            self.tno = [None] * ntrip
        P = padded(self._npao_t)
        nbf = self.orb.nbf
        from electrolites.dlpno import mgpu
        devs = mgpu.devices()
        items = []
        for Pv in np.unique(P):
            sel = np.where(P == Pv)[0]
            sel = sel[self.dfi.spatial_order([self.t_riatoms[t] for t in sel])]
            Pv = int(Pv)
            per = 8 * (12 * Pv * Pv + 2 * nbf * Pv)
            step = max(1, int(self._chunk_bytes() // per))
            if len(devs) > 1:
                step = max(1, min(step, -(-len(sel) // (2 * len(devs)))))
            items += [(Pv, sel[c0:c0 + step]) for c0 in range(0, len(sel), step)]
        items.sort(key=lambda it: -len(it[1]) * it[0] ** 2)
        ctxs = mgpu.run_queue(items, lambda w, it: w._ut0_item(*it, save=save), devs,
                              setup=lambda d: mgpu.device_view(
                                  self, ('_pt', 'S_pao_d', 'Fa_pao_d', 'Fb_pao_d', 'Fa_lp_d', 'Fb_lp_d'),
                                  sub={'dfi': ('qia', 'qij', 'J2c')}))
        for c in ctxs:
            if getattr(c, '_prof', None) is not None:
                c._prof.report()
        return float(self.e_ijk.sum())

    def _ut0_item(self, Pv, ch, save):
        nbf = self.orb.nbf
        tn = self._tno_chunk(ch, Pv, self._t_cut_tno)
        Va, Vb = tn['Xa'].shape[2], tn['Xbp'].shape[2]
        A = int(-(-self._naux_t[ch].max() // 32) * 32)
        L = int(padded(max(1, self._nl_t[ch].max())))
        Vm = max(Va, Vb)
        per = 8 * (2 * A * A + 14 * A * Vm + 18 * Vm ** 3 + 2 * A * Vm * Vm
                   + 8 * L * Vm * Vm + 2 * nbf * Vm + 4 * A * Pv
                   + 8 * L * self._pt["PP"] * Vm)
        sub = max(1, int(self._chunk_bytes() // per))
        for s0 in range(0, len(ch), sub):
            sl = slice(s0, s0 + sub)
            part = {k: v[sl] for k, v in tn.items()}
            self._ut0_chunk(ch[sl], part, save)

    # ------------------------------------------------------------------
    def _ut0_chunk(self, ch, tn, save):
        from electrolites.dlpno.devmem import Prof
        prof = self.__dict__.setdefault('_prof', Prof('U(T0)'))
        prof('setup')
        pl, dfi, ns = self.pl, self.dfi, self.ns
        ext = dfi.ext
        pt = self._pt
        trip = self._trip_arr
        B = len(ch)
        Xa, Xbp, Xbs = tn['Xa'], tn['Xbp'], tn['Xbs']
        P, Va, Vb = Xa.shape[1], Xa.shape[2], Xbp.shape[2]
        Vs = (Va, Vb)
        pidx, pm = tn['pidx'], tn['pm']
        nlv = self._nl_t
        L = int(padded(max(1, nlv[ch].max())))
        A = int(-(-self._naux_t[ch].max() // 32) * 32)
        lidx = np.zeros((B, L), np.int64)
        lmk = np.zeros((B, L))
        aidx = np.zeros((B, A), np.int64)
        amk = np.zeros((B, A))
        offs = []
        for s, t in enumerate(ch):
            lidx[s, :nlv[t]] = self.t_lmos[t]
            lmk[s, :nlv[t]] = 1.
            ax = self.t_ribfs[t]
            aidx[s, :len(ax)] = ax
            amk[s, :len(ax)] = 1.
            o, at_off = 0, {}
            for a in self.t_riatoms[t]:
                at_off[int(a)] = o
                o += len(dfi.atom_ribfs[a])
            offs.append(at_off)
        gp, gpm = cp.asarray(pidx), cp.asarray(pm)
        gl, glm = cp.asarray(lidx), cp.asarray(lmk)
        ga, gam = cp.asarray(aidx), cp.asarray(amk)
        ijk = trip[ch]
        J = dfi.J2c[ga[:, :, None], ga[:, None, :]] * (gam[:, :, None] * gam[:, None, :])
        J += cp.eye(A)[None] * (1 - gam)[:, :, None]
        Linv = chol_inv_batched(J)
        del J
        so_idx = cp.asarray(self.somo_idx)
        tr = lambda Y: Y.swapaxes(-1, -2)
        # ---- (Q|r a^s), (Q|r l) ------------------------------------------
        qv, qo = [], []
        for r in range(3):
            lmo = cp.asarray(ijk[:, r])
            q = dfi.qia[ga[:, :, None], lmo[:, None, None], gp[:, None, :]] * (gam[:, :, None] * gpm[:, None, :])
            qa = cp.matmul(q, Xa)
            qb = cp.matmul(q, Xbp)
            if ns:
                qs = dfi.qij[ga[:, :, None], lmo[:, None, None], so_idx[None, None, :]] * gam[:, :, None]
                qb = qb + cp.matmul(qs, Xbs)
            qv.append([cp.matmul(Linv, qa), cp.matmul(Linv, qb)])
            o = dfi.qij[ga[:, :, None], lmo[:, None, None], gl[:, None, :]] * (gam[:, :, None] * glm[:, None, :])
            qo.append(cp.matmul(Linv, o))
        # ---- 3-external tensors K3[(r, s, t)][e, y, z] = (e y | r z) -----------
        # (e, y of spin s; z of spin t = the spin of position r).  The fitted
        # (Q|ab) is never formed: K3 = sum_Q (Q|ey)_raw (J^-1 (Q|rz)), accumulated
        # per auxiliary atom block, so neither A V^2 storage nor an A^2 V^2
        # fitting step is needed.
        wq = [[cp.matmul(Linv.transpose(0, 2, 1), qv[r][t]) for t in (0, 1)] for r in range(3)]
        r_spins = [(0,) if np.all(self.is_somo[ijk[:, r]]) else (0, 1) for r in range(3)]
        # raw (Q|e y) per spin over the triplet's aux functions, filled atom by
        # atom, then every K3 block is one batched GEMM with K = A
        qvv = [cp.zeros((B, A, Va, Va)), cp.zeros((B, A, Vb, Vb))]
        prof('k3')
        atoms = sorted({int(a) for t in ch for a in self.t_riatoms[t]})
        for a in atoms:
            qa_ = dfi.qab_get(a)
            if qa_ is None:
                continue
            na = qa_.shape[0]
            rows = np.array([s for s, t in enumerate(ch) if a in offs[s]], np.int64)
            off = np.array([offs[s][a] for s in rows], np.int64)
            pos = ext.pao_pos[a][pidx[rows]]
            pos = np.where(pm[rows] > 0, pos, 0)
            assert (pos >= 0).all(), 'triplet PAO outside the extended domain'
            gr, gpos = cp.asarray(rows), cp.asarray(pos)
            mrow = gpm[gr]
            step = max(1, int((1 << 28) // (8 * na * max(P, Vb) ** 2)))
            for r0 in range(0, len(rows), step):
                sl = slice(r0, r0 + step)
                g_ = gr[sl]
                blk = gather_blocks(qa_, gpos[sl], gpos[sl], mrow[sl], mrow[sl], ru=True)
                oo = cp.asarray(off[sl])[:, None] + cp.arange(na)[None, :]
                for s_, Xs in ((0, Xa), (1, Xbp)):
                    xr = Xs[g_]
                    qvv[s_][g_[:, None], oo] = xtbx(blk, xr, xr)                    # (nr, na, V, V)
                del blk
        if ns:
            # SOMO parts of the beta (Q|ab): (Q|u s), (Q|s t), dense in Q
            for q0 in range(0, A, 128):
                qs_ = slice(q0, q0 + 128)
                ga_, gam_ = ga[:, qs_], gam[:, qs_]
                qsu = dfi.qia[ga_[:, :, None, None], so_idx[None, None, :, None], gp[:, None, None, :]]
                qsu = qsu * (gam_[:, :, None, None] * gpm[:, None, None, :])
                qss = dfi.qij[ga_[:, :, None, None], so_idx[None, None, :, None], so_idx[None, None, None, :]]
                qss = qss * gam_[:, :, None, None]
                m1 = cp.matmul(cp.matmul(tr(Xbp)[:, None], tr(qsu)), Xbs[:, None])
                qvv[1][:, qs_] += m1 + tr(m1) + cp.matmul(cp.matmul(tr(Xbs)[:, None], qss), Xbs[:, None])
        K3 = {}
        for r in range(3):
            for s_ in (0, 1):
                qt = qvv[s_].reshape(B, A, -1).transpose(0, 2, 1)                          # (B, V*V, A)
                for t in r_spins[r]:
                    K3[(r, s_, t)] = cp.matmul(qt, wq[r][t]).reshape(B, Vs[s_], Vs[s_], Vs[t])
        del qvv
        prof('w')
        # ---- Fock (occupied position r -> triplet virtuals) and singles -----
        nbf = self.orb.nbf
        Zt = [cp.matmul(self.S_pao_d[:, gp].transpose(1, 0, 2) * gpm[:, None, :], Xa),
              cp.concatenate([cp.matmul(self.S_pao_d[:, gp].transpose(1, 0, 2) * gpm[:, None, :], Xbp), Xbs],
                             axis=1)]
        fdiag = [np.diag(self.Fa_lmo), np.diag(self.Fb_lmo)]
        fov, t1p = {}, {}
        for r in range(3):
            lmo = cp.asarray(ijk[:, r])
            for s in (0, 1):
                Fl = (self.Fa_lp_d, self.Fb_lp_d)[s][lmo][:, None, :]
                Fr = cp.take_along_axis(Fl[:, 0], gp, axis=1) * gpm                # (B, P)
                f = cp.einsum('bp,bpv->bv', Fr, (Xa, Xbp)[s])
                if s == 1 and ns:
                    Fbs = cp.asarray(self.Fb_lmo[:, self.somo_idx])[lmo]          # (B, ns)
                    f = f + cp.einsum('bs,bsv->bv', Fbs, Xbs)
                fov[(r, s)] = f
        # ---- pattern batches ------------------------------------------------
        pats = {}
        for s, t in enumerate(ch):
            for pat in self._patterns(t):
                pats.setdefault(pat, []).append(s)
        si = self.so_index
        # projections onto pair PNO spaces, for the whole chunk: shared by all
        # spin patterns of a triplet (a pattern selects its rows)
        ccache = {}
        Ball = np.arange(B)

        def chunk_lproj(rA, sp_):
            kp = ('P', rA, sp_)
            if kp not in ccache:
                pil_all = np.where(lmk > 0, pl.ij[ijk[:, rA][:, None], lidx], 0)
                S = self._proj(Zt, sp_, np.repeat(Ball, L), pil_all.reshape(-1))
                ccache[kp] = S.reshape((B, L) + S.shape[1:])
            return ccache[kp]

        def chunk_pproj(rB, rC, sp_):
            kp = ('Q', min(rB, rC), max(rB, rC), sp_)
            if kp not in ccache:
                ccache[kp] = self._proj(Zt, sp_, Ball, pl.ij[ijk[:, rB], ijk[:, rC]])
            return ccache[kp]
        for pat, sb_l in pats.items():
            sb = np.array(sb_l, np.int64)
            gsb = cp.asarray(sb)
            n = len(sb)
            lm3 = ijk[sb]                                               # (n, 3)
            O = [[si[(int(lm3[q, r]), pat[r])] for r in range(3)] for q in range(n)]
            cache = {}

            Oarr = np.array(O, np.int64).reshape(n, 3)
            SIDX, SOF = self._sidx_tables()

            def tproj(rB, rC, s1, s2):
                key = ('t', rB, rC, s1, s2)
                if key in cache:
                    return cache[key]
                sB, sC = pat[rB], pat[rC]
                KB, KC = Oarr[:, rB], Oarr[:, rC]
                if (s1, s2) == (sB, sC):
                    items, sg = SIDX[KB, KC], 1.
                elif sB != sC and (s1, s2) == (sC, sB):
                    items, sg = SIDX[KC, KB], -1.
                else:
                    cache[key] = None
                    return None
                ok = items >= 0
                signs = np.where(ok, sg, 0.)
                items = np.where(ok, items, 0)
                pp = pl.ij[lm3[:, rB], lm3[:, rC]]

                def pproj(sp_):
                    # (B, C) and (C, B) share their PNO space: one projection per spin
                    return chunk_pproj(rB, rC, sp_)[gsb]
                S1 = pproj(s1)                                          # (n, VP1, V1)
                S2 = pproj(s2)
                Tm = pt['TS'][cp.asarray(items)][:, :S1.shape[1], :S2.shape[1]]
                v = cp.matmul(tr(S1), cp.matmul(Tm, S2)) * cp.asarray(signs)[:, None, None]
                cache[key] = v
                return v

            def tal(rA, sm, s1, s2):
                key = ('l', rA, sm, s1, s2)
                if key in cache:
                    return cache[key]
                sA = pat[rA]
                if (s1, s2) == (sA, sm):
                    rev, sg = False, 1.
                elif sA != sm and (s1, s2) == (sm, sA):
                    rev, sg = True, -1.
                else:
                    cache[key] = None
                    return None
                lmo_l = lidx[sb]                                       # (n, L)
                M = SOF[lmo_l, sm]
                Ain = Oarr[:, rA][:, None]
                ok = (lmk[sb] > 0) & (M >= 0) & (M != Ain)
                Mc = np.where(ok, M, 0)
                items = SIDX[Mc, np.broadcast_to(Ain, Mc.shape)] if rev else SIDX[np.broadcast_to(Ain, Mc.shape), Mc]
                ok &= items >= 0
                if not ok.any():
                    cache[key] = None
                    return None
                signs = np.where(ok, sg, 0.)
                items = np.where(ok, items, 0)
                pil = np.where(lmk[sb] > 0, pl.ij[lm3[:, rA][:, None], lmo_l], 0)
                bb = np.repeat(sb, L)

                def lproj(sp_):
                    # projections onto the (A, l) pairs depend on A and the spin only
                    S = chunk_lproj(rA, sp_)[gsb]
                    return S.reshape((n * L,) + S.shape[2:])
                S1 = lproj(s1)
                S2 = lproj(s2)
                Tm = pt['TS'][cp.asarray(items.reshape(-1))][:, :S1.shape[1], :S2.shape[1]]
                v = cp.matmul(tr(S1), cp.matmul(Tm, S2)) * cp.asarray(signs.reshape(-1))[:, None, None]
                v = v.reshape(n, L, S1.shape[2], S2.shape[2])
                cache[key] = v
                return v

            def k3(rA, s, t_):
                key = ('k3', rA, s, t_)
                if key not in cache:
                    cache[key] = K3[(rA, s, t_)][gsb]
                return cache[key]

            def kov(rB, rC, s):
                key = ('ov', rB, rC, s)
                if key not in cache:
                    cache[key] = cp.matmul(tr(qo[rB][gsb]), qv[rC][s][gsb])       # (n, L, V)
                return cache[key]

            def kvv(rB, s1, rC, s2):
                key = ('vv', rB, s1, rC, s2)
                if key not in cache:
                    cache[key] = cp.matmul(tr(qv[rB][s1][gsb]), qv[rC][s2][gsb])
                return cache[key]

            def t1v(r):
                key = ('t1', r)
                if key not in cache:
                    s = pat[r]
                    pd = pl.ij[lm3[:, r], lm3[:, r]]
                    S = self._proj(Zt, s, gsb, pd)                        # (n, VP, V)
                    t1 = pt['T1'][cp.asarray([O[q][r] for q in range(n)])][:, :S.shape[1]]
                    cache[key] = cp.einsum('npv,np->nv', S, t1)
                return cache[key]

            shape = (n, Vs[pat[0]], Vs[pat[1]], Vs[pat[2]])
            Wt = cp.zeros(shape)
            Vt = cp.zeros(shape)
            for (oA, oB, oC), eo in _OCC:
                sA, sB, sC = pat[oA], pat[oB], pat[oC]
                for rho, ev_ in _VIR:
                    sx, sy, sz = pat[rho[0]], pat[rho[1]], pat[rho[2]]
                    nx, ny, nz = Vs[sx], Vs[sy], Vs[sz]
                    X = None
                    se = sB + sC - sx
                    if se in (0, 1):
                        t = tproj(oB, oC, sx, se)
                        if t is not None:
                            if se == sy and sA == sz:
                                Y = cp.matmul(t, k3(oA, sy, sA).reshape(n, Vs[se], ny * nz)).reshape(n, nx, ny, nz)
                                X = Y if X is None else X + Y
                            if se == sz and sA == sy:
                                Y = cp.matmul(t, k3(oA, sz, sA).reshape(n, Vs[se], nz * ny))
                                Y = Y.reshape(n, nx, nz, ny).transpose(0, 1, 3, 2)
                                X = -Y if X is None else X - Y
                    if sx == sC:
                        tl = tal(oA, sB, sy, sz)
                        if tl is not None:
                            Y = cp.matmul(tr(kov(oB, oC, sx) * glm[gsb][:, :, None]),
                                          tl.reshape(n, L, ny * nz)).reshape(n, nx, ny, nz)
                            X = -Y if X is None else X - Y
                    if sx == sB:
                        tl = tal(oA, sC, sy, sz)
                        if tl is not None:
                            Y = cp.matmul(tr(kov(oC, oB, sx) * glm[gsb][:, :, None]),
                                          tl.reshape(n, L, ny * nz)).reshape(n, nx, ny, nz)
                            X = Y if X is None else X + Y
                    # disconnected: t_A^x <BC||yz> + f_Ax t_BC^yz
                    Xv = None
                    if sx == sA:
                        g = None
                        if sy == sB and sz == sC:
                            g = kvv(oB, sy, oC, sz)
                        if sz == sB and sy == sC:
                            h = kvv(oC, sy, oB, sz)
                            g = -h if g is None else g - h
                        if g is not None:
                            Xv = t1v(oA)[:, :, None, None] * g[:, None]
                        tbc = tproj(oB, oC, sy, sz)
                        if tbc is not None:
                            Y = fov[(oA, sx)][gsb][:, :, None, None] * tbc[:, None]
                            Xv = Y if Xv is None else Xv + Y
                    back = (0,) + tuple(int(a) + 1 for a in np.argsort(rho))
                    if X is not None:
                        Wt += (eo * ev_) * X.transpose(back)
                    if Xv is not None:
                        Vt += (eo * ev_) * Xv.transpose(back)
            fs = cp.asarray(np.array([sum(fdiag[pat[r]][lm3[q, r]] for r in range(3)) for q in range(n)]))
            e = [tn['ea'][gsb], tn['eb'][gsb]]
            Dn = (fs[:, None, None, None] - e[pat[0]][:, :, None, None]
                  - e[pat[1]][:, None, :, None] - e[pat[2]][:, None, None, :])
            T3 = Wt / Dn
            mult = 1. if pat[0] == pat[1] == pat[2] else 3.
            eb = (mult / 6.) * cp.sum(T3 * (Wt + Vt), axis=(1, 2, 3))
            np.add.at(self.e_ijk, ch[sb], eb.get())
            if save:
                for q in range(n):
                    t = ch[sb[q]]
                    nf = (int(tn['nfa'][sb[q]]), int(tn['nfb'][sb[q]]))
                    d0, d1, d2 = (nf[x] for x in pat)
                    from electrolites.dlpno.devmem import to_pinned
                    sl3 = (q, slice(0, d0), slice(0, d1), slice(0, d2))
                    # true-size blocks; W and W+V wait in pinned host memory
                    ev = [e[pat[r]][q, :(d0, d1, d2)[r]].get() for r in range(3)]
                    self.st[(t, pat)] = dict(W=to_pinned(Wt[sl3]), U=to_pinned(Wt[sl3] + Vt[sl3]),
                                             T=cp.ascontiguousarray(T3[sl3]), O=tuple(O[q]),
                                             e=ev, fs=float(fs[q]), n=float(d0 * d1 * d2))
        if save:
            for s, t in enumerate(ch):
                na_, nb_ = int(tn['nfa'][s]), int(tn['nfb'][s])
                self.tno[t] = dict(Xa=Xa[s, :, :na_], Xb=cp.concatenate([Xbp[s, :, :nb_], Xbs[s, :, :nb_]], axis=0),
                                   pidx=pidx[s][:self._npao_t[t]])
        prof(None)

    # ------------------------------------------------------------------
    def lccsd_t_iterations(self):
        """Iterative (T): R = W - D T - sum_L f_LX T(X -> L), Jacobi sweeps
        over buckets of equal padded aligned shape; every coupling group
        (bucket of t, bucket of u, axis permutation) is one batched
        three-mode transformation with per-axis (per-spin) TNO overlaps."""
        opt = self.opt
        so, nso = self.so, len(self.so)
        F = (self.Fa_lmo, self.Fb_lmo)
        keys = list(self.st)
        ent = [self.st[k] for k in keys]
        self.st = None
        okey = lambda O: tuple(sorted(O, key=lambda I: (so[I][0], so[I][1])))
        where = {okey(e['O']): n for n, e in enumerate(ent)}
        bk0 = {}
        for n, e in enumerate(ent):
            bk0.setdefault(tuple(int(padded(x)) for x in e['W'].shape), []).append(n)
        # buckets of at most ~512 MB per array (key: shape + part number)
        bk = {}
        for shp, mem in bk0.items():
            step = max(1, int((1 << 29) // (8 * int(np.prod(shp)))))
            for i0 in range(0, len(mem), step):
                bk[shp + (i0 // step,)] = mem[i0:i0 + step]
        slot = np.zeros(len(keys), np.int64)
        bof = [None] * len(keys)
        for s, mem in bk.items():
            slot[mem] = np.arange(len(mem))
            for m_ in mem:
                bof[m_] = s
        from electrolites.dlpno.devmem import pinned_empty, to_dev

        def stack_host(name, s, mem):
            out = pinned_empty((len(mem),) + s[:3])
            out[...] = 0.
            for i, m_ in enumerate(mem):
                a = ent[m_][name]
                out[i, :a.shape[0], :a.shape[1], :a.shape[2]] = a
            return out

        def stack_dev(s, mem):
            out = cp.zeros((len(mem),) + s[:3])
            for i, m_ in enumerate(mem):
                a = ent[m_]['T']
                out[i, :a.shape[0], :a.shape[1], :a.shape[2]] = a
            return out

        def d_parts(s, mem):
            e3 = [np.full((len(mem), s[r]), 1e3) for r in range(3)]
            fsv = np.zeros(len(mem))
            for i, m_ in enumerate(mem):
                for r in range(3):
                    v = ent[m_]['e'][r]
                    e3[r][i, :len(v)] = v
                fsv[i] = ent[m_]['fs']
            return cp.asarray(fsv), [cp.asarray(x) for x in e3]

        def stack_D(s, mem):
            fsv, e3 = Dp[s]
            return (fsv[:, None, None, None] - e3[0][:, :, None, None]
                    - e3[1][:, None, :, None] - e3[2][:, None, None, :])
        Dp = {s: d_parts(s, m) for s, m in bk.items()}
        W = {s: stack_host('W', s, m) for s, m in bk.items()}
        Vv = {s: stack_host('U', s, m) for s, m in bk.items()}
        T = {s: stack_dev(s, m) for s, m in bk.items()}
        mult = {s: cp.asarray(np.array([1. if len({so[I][1] for I in ent[m_]['O']}) == 1 else 3.
                                        for m_ in m])) for s, m in bk.items()}
        trip_of = {s: np.array([keys[m_][0] for m_ in m]) for s, m in bk.items()}
        cnt = {s: cp.asarray(np.array([ent[m_]['n'] for m_ in m])) for s, m in bk.items()}
        Os = [e['O'] for e in ent]
        del ent
        S_d = self.S_pao_d
        Scache = {}

        # the TNO PAO rows of a triplet are padded to its chunk: keep only the true rows
        for t, d in enumerate(self.tno):
            if d is None:
                continue
            P = d['Xa'].shape[0]
            npao = len(d['pidx'])
            d['Xa'] = d['Xa'][:npao]
            d['Xb'] = cp.concatenate([d['Xb'][:npao], d['Xb'][P:]], axis=0)
            d['Xa_rows'] = npao

        # ---- couplings t <- u (X -> L at one position), vectorised ----------
        nT = len(Os)
        so_l = np.array([x for x, _ in so], np.int64)
        so_s = np.array([sp_ for _, sp_ in so], np.int64)
        rk = np.empty(nso, np.int64)                    # rank of (lmo, spin)
        rk[np.lexsort((so_s, so_l))] = np.arange(nso)
        Oa = np.array(Os, np.int64).reshape(-1, 3)

        def tcode(M):                                   # canonical code of spin-orbital triples
            r_ = np.sort(rk[M], axis=1)
            return (r_[:, 0] * nso + r_[:, 1]) * nso + r_[:, 2]
        codes = tcode(Oa)
        corder = np.argsort(codes)
        csorted = codes[corder]
        bucket_id = {s_: i for i, s_ in enumerate(bk)}
        bid = np.array([bucket_id[bof[n]] for n in range(nT)], np.int64)
        trip_n = np.array([k[0] for k in keys], np.int64)
        cn, cu, cf, cr = [], [], [], []
        for r in range(3):
            X = Oa[:, r]
            xs = so_s[X]
            for sg in (0, 1):
                rows = np.where(xs == sg)[0]
                if not len(rows):
                    continue
                Ls = np.where(so_s == sg)[0]
                fm = F[sg][np.ix_(so_l[Ls], so_l[X[rows]])].T          # (rows, Ls)
                ok = np.abs(fm) >= opt['F_CUT_T']
                ok &= ~(Ls[None, :] == Oa[rows][:, [0]]) & ~(Ls[None, :] == Oa[rows][:, [1]]) \
                    & ~(Ls[None, :] == Oa[rows][:, [2]])
                ri, li = np.nonzero(ok)
                if not len(ri):
                    continue
                nn_ = rows[ri]
                Mnew = Oa[nn_].copy()
                Mnew[:, r] = Ls[li]
                c_ = tcode(Mnew)
                pos = np.searchsorted(csorted, c_)
                pos = np.minimum(pos, len(csorted) - 1)
                hit = csorted[pos] == c_
                cn.append(nn_[hit])
                cu.append(corder[pos[hit]])
                cf.append(fm[ri[hit], li[hit]])
                cr.append(np.full(int(hit.sum()), r, np.int64))
        cn = np.concatenate(cn) if cn else np.zeros(0, np.int64)
        cu = np.concatenate(cu) if cu else np.zeros(0, np.int64)
        cf = np.concatenate(cf) if cf else np.zeros(0)
        cr = np.concatenate(cr) if cr else np.zeros(0, np.int64)
        # perm[q]: position in u's stored triple of the q-th orbital of the new triple
        Ou = Oa[cu]
        rem = [Oa[cn, (cr + 1) % 3], Oa[cn, (cr + 2) % 3]]
        isL = np.ones((len(cn), 3), bool)
        for q in range(3):
            isL[:, q] = (Ou[:, q] != rem[0]) & (Ou[:, q] != rem[1])
        Lcol = Ou[np.arange(len(cn)), np.argmax(isL, axis=1)]
        Mnew = Oa[cn].copy()
        Mnew[np.arange(len(cn)), cr] = Lcol
        perm = np.zeros((len(cn), 3), np.int64)
        for q in range(3):
            perm[:, q] = np.argmax(Ou == Mnew[:, [q]], axis=1)
        # the TNO overlaps S_s(t, u) once per distinct (t, u, spin), from the
        # device overlap generator over the triplets' compact TNO coefficients
        from electrolites.dlpno.ovgen import OverlapGen
        from electrolites.dlpno.ragged import Layout, Ragged
        ntr = len(self.trip)
        tpao = [np.asarray(self.tno[t]['pidx']) if self.tno[t] is not None else np.zeros(0, np.int64)
                for t in range(ntr)]
        Xs_ = [[self.tno[t]['Xa'] if self.tno[t] is not None else cp.zeros((0, 1)) for t in range(ntr)],
               [self.tno[t]['Xb'] if self.tno[t] is not None else cp.zeros((self.ns, 1)) for t in range(ntr)]]
        tt_c, uu_c = trip_n[cn], trip_n[cu]
        tu = tt_c * ntr + uu_c
        axspin = so_s[Oa[cn]]                                  # (c, 3) spin of each axis
        SU, LU, sid = [], [], np.zeros((len(cn), 3), np.int64)
        for sg in (0, 1):
            msk = axspin == sg
            codes_sg = np.unique(tu[msk.any(axis=1)])
            for q in range(3):
                sel_ = msk[:, q]
                sid[sel_, q] = np.searchsorted(codes_sg, tu[sel_])
            tt = codes_sg // ntr
            uu = codes_sg % ntr
            gen = OverlapGen(S_d, tpao, Xs_[sg], ne=(self.ns if sg == 1 else 0))
            nvt = gen.nv
            lay = Layout(nvt[tt], nvt[uu]) if len(codes_sg) else Layout(np.zeros(0, np.int64), 1)
            R_ = Ragged(lay)
            if len(codes_sg):
                for P_ in gen.prepare(tt, uu, out_layout=lay, out_idx=np.arange(len(codes_sg))):
                    gen.run(P_, R_)
            SU.append(R_)
            LU.append(lay)
            del gen
        # ---- coupling groups: destination bucket part x coarse permuted
        # source shape; every coupling carries its source pointer / strides
        # and the pointers of its three TNO-overlap blocks, so a group is a
        # handful of kernels however its permutations and spin patterns mix
        from electrolites.dlpno.gather import gather2d, gather3p
        bks = list(bk)
        shp_b = np.array([b_[:3] for b_ in bks], np.int64)          # (nb, 3)
        tbase = np.array([T[b_].data.ptr for b_ in bks], np.int64)
        su_c = shp_b[bid[cu]]                                        # (c, 3)
        up_true = np.take_along_axis(su_c, perm, axis=1)
        coarse = lambda v: -(-v // 32) * 32          # few source shapes -> few, large groups
        up_c = coarse(up_true)
        full = np.stack([su_c[:, 1] * su_c[:, 2], su_c[:, 2], np.ones(len(cn), np.int64)], 1)
        strd_all = np.take_along_axis(full, perm, axis=1)
        src_all = tbase[bid[cu]] + 8 * slot[cu] * su_c.prod(axis=1)
        # S pointer tables per axis
        sptr, sld, snr, snc = (np.zeros((len(cn), 3), np.int64) for _ in range(4))
        for sg in (0, 1):
            lay = LU[sg]
            if not lay.n:
                continue
            shp = np.array(lay.shapes, np.int64).reshape(-1, 2)
            gbase = np.array([d.data.ptr for d in SU[sg].data], np.int64)
            for q in range(3):
                sel_ = axspin[:, q] == sg
                ids = sid[sel_, q]
                g_ = lay.group[ids]
                sptr[sel_, q] = gbase[g_] + 8 * lay.slot[ids] * shp[g_, 0] * shp[g_, 1]
                sld[sel_, q] = shp[g_, 1]
                snr[sel_, q] = lay.rows[ids]
                snc[sel_, q] = lay.cols[ids]
        gkey = (bid[cn] * 4096 + up_c[:, 0] // 32 * 256 + up_c[:, 1] // 32 * 16 + up_c[:, 2] // 32)
        order = np.argsort(gkey, kind='stable')
        ukeys, starts = np.unique(gkey[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        plan = []
        for k0, k1 in zip(starts, ends):
            mem = order[k0:k1]
            sn = bks[int(bid[cn[mem[0]]])]
            up = tuple(int(x) for x in up_c[mem[0]])
            per = 8 * 6 * max(int(np.prod(sn[:3])), int(np.prod(up)))
            step = max(1, int((1 << 28) // per))
            for i0 in range(0, len(mem), step):
                m_ = mem[i0:i0 + step]
                plan.append(dict(sn=sn, up=up, c=len(m_), src=cp.asarray(src_all[m_]),
                                 strd=cp.asarray(strd_all[m_]), nn=cp.asarray(up_true[m_].astype(np.int32)),
                                 S=[(cp.asarray(sptr[m_, q]), cp.asarray(sld[m_, q].astype(np.int32)),
                                     cp.asarray(snr[m_, q].astype(np.int32)), cp.asarray(snc[m_, q].astype(np.int32)))
                                    for q in range(3)],
                                 st=cp.asarray(slot[cn[m_]]), w=cp.asarray(cf[m_])))

        self.log(f'  U(T): {len(keys)} spin triples in {len(bk)} buckets, '
                 f'{sum(g["c"] for g in plan)} couplings in {len(plan)} groups')

        e_prev = 0.
        ntrip = len(self.trip)
        by_sn = {}
        for g in plan:
            by_sn.setdefault(g['sn'], []).append(g)
        from electrolites.dlpno.devmem import Prof
        prof = Prof('U(T) iter')
        for it in range(1, opt['MAXITER'] + 1):
            # bucket by bucket (Gauss-Seidel across buckets): only one bucket's
            # W, D and residual live on the device at a time
            rms = 0.
            for s in bk:
                prof('W,D')
                Ds = stack_D(s, bk[s])
                Rs = to_dev(W[s]) - Ds * T[s]
                for g in by_sn.get(s, []):
                    prof('perm')
                    Au = gather3p(g['src'], g['strd'], g['nn'], *g['up'])
                    prof('gatherS')
                    S0, S1, S2 = (gather2d(*g['S'][q], g['sn'][q], g['up'][q]) for q in range(3))
                    prof('mm')
                    c_, a0, a1, a2 = Au.shape
                    t0, t1, t2 = S0.shape[1], S1.shape[1], S2.shape[1]
                    A1 = cp.matmul(S0, Au.reshape(c_, a0, a1 * a2)).reshape(c_, t0, a1, a2)
                    A2 = cp.matmul(A1.reshape(c_, t0 * a1, a2), S2.transpose(0, 2, 1)).reshape(c_, t0, a1, t2)
                    A3 = cp.matmul(S1[:, None], A2)
                    prof('scatter')
                    cupyx.scatter_add(Rs, g['st'], -g['w'][:, None, None, None] * A3)
                    del Au, A1, A2, A3
                prof('update')
                T[s] += Rs / Ds
                r = cp.sqrt(cp.sum(Rs ** 2, axis=(1, 2, 3)) / cnt[s])
                rms = max(rms, float(r.max()))
                del Ds, Rs
            prof('energy')
            self.e_ijk = np.zeros(ntrip)
            for s in bk:
                e = (mult[s] / 6.) * cp.sum(T[s] * to_dev(Vv[s]), axis=(1, 2, 3))
                np.add.at(self.e_ijk, trip_of[s], e.get())
            prof(None)
            prof.report()
            E = float(self.e_ijk.sum())
            self.log(f'  GPU U(T) iter {it:3d} {E:.12f} {E - e_prev:.3e} {rms:.3e}')
            if rms < opt['R_CONVERGENCE'] and abs(E - e_prev) < opt['E_CONVERGENCE']:
                break
            e_prev = E
        else:
            raise RuntimeError('(T) iterations did not converge')
        return E
