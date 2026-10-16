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
import cupyx

from electrolites.dlpno.ragged import padded
from electrolites.dlpno.linalg import eigh_batched, orthonormal_domain

_OCC = [((0, 1, 2), 1.), ((1, 0, 2), -1.), ((2, 1, 0), -1.)]
_VIR = _OCC


class UTriplesGPU:
    """Mixin for uhf.UDLPNO replacing tno_transform / lccsd_t0 /
    lccsd_t_iterations of the host reference."""

    MAX_CHUNK_BYTES = 1 << 30

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
        for Pv in np.unique(P):
            sel = np.where(P == Pv)[0]
            sel = sel[np.argsort(self._naux_t[sel], kind='stable')]
            Pv = int(Pv)
            per = 8 * (12 * Pv * Pv + 2 * nbf * Pv)
            step = max(1, int(self.MAX_CHUNK_BYTES // per))
            for c0 in range(0, len(sel), step):
                ch = sel[c0:c0 + step]
                tn = self._tno_chunk(ch, Pv, self._t_cut_tno)
                Va, Vb = tn['Xa'].shape[2], tn['Xbp'].shape[2]
                A = int(-(-self._naux_t[ch].max() // 32) * 32)
                L = int(padded(max(1, self._nl_t[ch].max())))
                Vm = max(Va, Vb)
                per = 8 * (A * A + 2 * A * (Va * Va + Vb * Vb) + 6 * A * Vm + 20 * Vm ** 3
                           + 6 * L * Vm * Vm + 2 * nbf * Vm + 3 * A * Pv)
                sub = max(1, int(self.MAX_CHUNK_BYTES // per))
                for s0 in range(0, len(ch), sub):
                    sl = slice(s0, s0 + sub)
                    part = {k: (v[sl] if k not in ('nfa', 'nfb', 'pidx', 'pm') else v[sl])
                            for k, v in tn.items()}
                    self._ut0_chunk(ch[sl], part, save)
        return float(self.e_ijk.sum())

    # ------------------------------------------------------------------
    def _ut0_chunk(self, ch, tn, save):
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
        Linv = cp.linalg.inv(cp.linalg.cholesky(J))
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
        # ---- (Q|ab) per spin -------------------------------------------------
        qvv = [cp.zeros((B, A, Va, Va)), cp.zeros((B, A, Vb, Vb))]
        atoms = sorted({int(a) for t in ch for a in self.t_riatoms[t]})
        for a in atoms:
            qa_ = dfi.qab[a]
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
            step = max(1, int((1 << 28) // (8 * na * P * P)))
            for r0 in range(0, len(rows), step):
                sl = slice(r0, r0 + step)
                blk = qa_[:, gpos[sl][:, :, None], gpos[sl][:, None, :]].transpose(1, 0, 2, 3)
                blk = blk * (mrow[sl][:, :, None] * mrow[sl][:, None, :])[:, None]
                oo = cp.asarray(off[sl])[:, None] + cp.arange(na)[None, :]
                for s, Xs in ((0, Xa), (1, Xbp)):
                    xr = Xs[gr[sl]]
                    qvv[s][gr[sl][:, None], oo] = cp.matmul(cp.matmul(tr(xr)[:, None], blk), xr[:, None])
        if ns:
            qsu = dfi.qia[ga[:, :, None, None], so_idx[None, None, :, None], gp[:, None, None, :]]
            qsu = qsu * (gam[:, :, None, None] * gpm[:, None, None, :])            # (B, A, ns, P)
            qss = dfi.qij[ga[:, :, None, None], so_idx[None, None, :, None], so_idx[None, None, None, :]]
            qss = qss * gam[:, :, None, None]
            XbpT = tr(Xbp)[:, None]
            XbsT = tr(Xbs)[:, None]
            m1 = cp.matmul(cp.matmul(XbpT, tr(qsu)), Xbs[:, None])
            qvv[1] += m1 + tr(m1) + cp.matmul(cp.matmul(XbsT, qss), Xbs[:, None])
        qvv = [cp.matmul(Linv, x.reshape(B, A, -1)).reshape(x.shape) for x in qvv]
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
                S1 = self._proj(Zt, s1, gsb, pp)                       # (n, VP1, V1)
                S2 = self._proj(Zt, s2, gsb, pp)
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
                pil = np.where(ok, pl.ij[lm3[:, rA][:, None], lmo_l], 0)
                bb = np.repeat(sb, L)
                S1 = self._proj(Zt, s1, bb, pil.reshape(-1))
                S2 = self._proj(Zt, s2, bb, pil.reshape(-1))
                Tm = pt['TS'][cp.asarray(items.reshape(-1))][:, :S1.shape[1], :S2.shape[1]]
                v = cp.matmul(tr(S1), cp.matmul(Tm, S2)) * cp.asarray(signs.reshape(-1))[:, None, None]
                v = v.reshape(n, L, S1.shape[2], S2.shape[2])
                cache[key] = v
                return v

            def k3(rA, s, t_):
                key = ('k3', rA, s, t_)
                if key not in cache:
                    q = qvv[s][gsb]
                    cache[key] = cp.matmul(tr(q.reshape(n, A, -1)), qv[rA][t_][gsb]).reshape(n, Vs[s], Vs[s], Vs[t_])
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
                    self.st[(t, pat)] = dict(W=Wt[q], V=Vt[q], T=T3[q], D=Dn[q], O=tuple(O[q]),
                                             n=float(np.prod([nf[x] for x in pat])))
        if save:
            for s, t in enumerate(ch):
                na_, nb_ = int(tn['nfa'][s]), int(tn['nfb'][s])
                self.tno[t] = dict(Xa=Xa[s, :, :na_], Xb=cp.concatenate([Xbp[s, :, :nb_], Xbs[s, :, :nb_]], axis=0),
                                   pidx=pidx[s][:self._npao_t[t]])

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
        bk = {}
        for n, e in enumerate(ent):
            bk.setdefault(tuple(e['W'].shape), []).append(n)
        slot = np.zeros(len(keys), np.int64)
        bof = [None] * len(keys)
        for s, mem in bk.items():
            slot[mem] = np.arange(len(mem))
            for m_ in mem:
                bof[m_] = s
        W = {s: cp.stack([ent[m_]['W'] for m_ in m]) for s, m in bk.items()}
        Vv = {s: cp.stack([ent[m_]['V'] for m_ in m]) for s, m in bk.items()}
        T = {s: cp.stack([ent[m_]['T'] for m_ in m]) for s, m in bk.items()}
        D = {s: cp.stack([ent[m_]['D'] for m_ in m]) for s, m in bk.items()}
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

        def ovl2(t, u, sg):
            key = (t, u, sg)
            v = Scache.get(key)
            if v is None:
                a, b = self.tno[t], self.tno[u]
                Sm = S_d[cp.asarray(a['pidx'])[:, None], cp.asarray(b['pidx'])[None, :]]
                if sg == 0:
                    v = a['Xa'].T @ Sm @ b['Xa']
                else:
                    pa, pb = a['Xa_rows'], b['Xa_rows']
                    v = a['Xb'][:pa].T @ Sm @ b['Xb'][:pb] + a['Xb'][pa:].T @ b['Xb'][pb:]
                Scache[key] = v
            return v
        groups = {}
        for n, O in enumerate(Os):
            for r in range(3):
                X = O[r]
                xs = so[X][1]
                for Lso in range(nso):
                    if so[Lso][1] != xs or Lso in O:
                        continue
                    f = F[xs][so[Lso][0], so[X][0]]
                    if abs(f) < opt['F_CUT_T']:
                        continue
                    new = list(O)
                    new[r] = Lso
                    ck = okey(new)
                    u = where.get(ck)
                    if u is None:
                        continue
                    perm = tuple(ck.index(new[q]) for q in range(3))
                    groups.setdefault((bof[n], bof[u], perm), []).append((n, u, f))
        plan = []
        for (sn, su, perm), lst in groups.items():
            c = len(lst)
            Ss = []
            for q in range(3):
                Sq = cp.zeros((c, sn[q], su[perm[q]]))
                for k_, (n, u, f) in enumerate(lst):
                    sg = so[Os[n][q]][1]
                    v = ovl2(keys[n][0], keys[u][0], sg)
                    Sq[k_, :v.shape[0], :v.shape[1]] = v
                Ss.append(Sq)
            plan.append(dict(sn=sn, su=su, perm=perm, S=Ss,
                             st=cp.asarray(slot[[l[0] for l in lst]]), suo=cp.asarray(slot[[l[1] for l in lst]]),
                             w=cp.asarray(np.array([l[2] for l in lst]))))
        Scache.clear()
        e_prev = 0.
        ntrip = len(self.trip)
        for it in range(1, opt['MAXITER'] + 1):
            R = {s: W[s] - D[s] * T[s] for s in bk}
            for g in plan:
                Au = T[g['su']][g['suo']].transpose((0,) + tuple(1 + p for p in g['perm']))
                S0, S1, S2 = g['S']
                c_, a0, a1, a2 = Au.shape
                t0, t1, t2 = S0.shape[1], S1.shape[1], S2.shape[1]
                A1 = cp.matmul(S0, Au.reshape(c_, a0, a1 * a2)).reshape(c_, t0, a1, a2)
                A2 = cp.matmul(A1.reshape(c_, t0 * a1, a2), S2.transpose(0, 2, 1)).reshape(c_, t0, a1, t2)
                A3 = cp.matmul(S1[:, None], A2)
                cupyx.scatter_add(R[g['sn']], g['st'], -g['w'][:, None, None, None] * A3)
            rms = 0.
            for s in bk:
                T[s] = T[s] + R[s] / D[s]
                r = cp.sqrt(cp.sum(R[s] ** 2, axis=(1, 2, 3)) / cnt[s])
                rms = max(rms, float(r.max()))
            self.e_ijk = np.zeros(ntrip)
            for s in bk:
                e = (mult[s] / 6.) * cp.sum(T[s] * (W[s] + Vv[s]), axis=(1, 2, 3))
                np.add.at(self.e_ijk, trip_of[s], e.get())
            E = float(self.e_ijk.sum())
            self.log(f'  GPU U(T) iter {it:3d} {E:.12f} {E - e_prev:.3e} {rms:.3e}')
            if rms < opt['R_CONVERGENCE'] and abs(E - e_prev) < opt['E_CONVERGENCE']:
                break
            e_prev = E
        else:
            raise RuntimeError('(T) iterations did not converge')
        return E
