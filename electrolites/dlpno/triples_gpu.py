"""Closed-shell DLPNO-(T0) and iterative (T) on the device.

Equations as ``ref_rhf`` (Jiang et al. 2024, Eqs. 53, 109-112).  Triplets
are processed in chunks of equal padded shape: every per-triplet loop of the
reference becomes a batched gather or a strided-batched GEMM over the chunk,
and the triplet natural orbitals of a chunk come from two batched
eigensolver calls (``linalg.eigh_batched``).

Shapes in a chunk: B triplets, A auxiliary functions (padded), P PAOs, V
TNOs, L LMOs of the triplet domain; pair quantities are padded to the
largest pair (PP PAOs, VP PNOs).
"""
import numpy as np
import cupy as cp

from electrolites.dlpno.ragged import padded
from electrolites.dlpno.linalg import eigh_batched

# the six orders of (i, j, k) in W (Jiang Eq. 109) and the transpose that
# brings each partial W back to (a, b, c)
_PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
_BACK = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 0, 1), (1, 2, 0), (2, 1, 0)]


def _perm_case(a1, a2, a3):
    """Psi4 triples_permuter: index into _CASES of the transpose used."""
    if a1 <= a2 <= a3:
        return 0
    if a1 <= a3 <= a2:
        return 1
    if a2 <= a1 <= a3:
        return 2
    if a2 <= a3 <= a1:
        return 3
    if a3 <= a1 <= a2:
        return 4
    return 5


_CASES = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 0, 1), (1, 2, 0), (2, 1, 0)]


def _energy_coeffs(i, j, k):
    """E_ijk = pref * sum_c coef[c] * <V^{case c}, T>, as a length-6 vector."""
    pref = 1.
    if i == j == k:
        pref /= 6.
    elif i == j or j == k or i == k:
        pref /= 2.
    c = np.zeros(6)
    c[0] += 8
    for args, w in (((k, j, i), -4), ((i, k, j), -4), ((j, i, k), -4),
                    ((j, k, i), 2), ((k, i, j), 2)):
        c[_perm_case(*args)] += w
    return pref * c


def _batched_energy(V, T3, coef):
    """per triplet sum_c coef[b, c] <V_b transposed by case c, T3_b>."""
    out = cp.zeros(V.shape[0])
    for c, ax in enumerate(_CASES):
        if not cp.any(coef[:, c] != 0):
            continue
        d = cp.sum(V.transpose((0,) + tuple(a + 1 for a in ax)) * T3, axis=(1, 2, 3))
        out += coef[:, c] * d
    return out


class TriplesGPU:
    """Mixin for rhf.DLPNO replacing the reference's tno_transform /
    lccsd_t0 / lccsd_t_iterations."""

    MAX_CHUNK_BYTES = 1 << 30

    # ---- pair tables --------------------------------------------------
    def _pair_tables(self):
        """Padded device copies of every pair's PNOs (in its PAO domain),
        amplitudes and pair density, and of the singles."""
        if getattr(self, '_pt_src', None) is self.T:
            return
        self._pt_src = self.T
        pl = self.pl
        npair = len(pl)
        nv = np.array([x.shape[1] for x in self.X])
        npo = np.array([len(p) for p in self.pao])
        VP, PP = int(padded(nv.max())), int(padded(npo.max()))
        Xp = np.zeros((npair, PP, VP))
        dom = np.zeros((npair, PP), np.int64)
        dmk = np.zeros((npair, PP))
        Tp = np.zeros((npair, VP, VP))
        for p in range(npair):
            Xp[p, :npo[p], :nv[p]] = self.X[p]
            dom[p, :npo[p]] = self.pao[p]
            dmk[p, :npo[p]] = 1.
            Tp[p, :nv[p], :nv[p]] = self.T[p]
        Tp_d = cp.asarray(Tp)
        Tt = 2 * Tp_d - Tp_d.transpose(0, 2, 1)
        Dp = cp.matmul(Tt, Tp_d.transpose(0, 2, 1)) + cp.matmul(Tt.transpose(0, 2, 1), Tp_d)
        diag = np.array([i == j for (i, j) in pl.pairs])
        Dp[cp.asarray(diag)] *= .5
        T1 = np.zeros((self.n, VP))
        for i in range(self.n):
            T1[i, :len(self.T1[i])] = self.T1[i]
        self._pt = dict(VP=VP, PP=PP, X=cp.asarray(Xp), dom=cp.asarray(dom),
                        dmk=cp.asarray(dmk), T=Tp_d, D=Dp, T1=cp.asarray(T1))

    def _project(self, Z, bidx, pidx):
        """S(a_t, a_p)^T = X_p^T S_pao[d_p, d_t] X_t for couples (b, p),
        from Z[b] = S_pao[:, d_t] X_t.  -> (C, VP, V)."""
        pt = self._pt
        Zg = Z[bidx[:, None], pt['dom'][pidx]] * pt['dmk'][pidx][:, :, None]
        return cp.matmul(pt['X'][pidx].transpose(0, 2, 1), Zg)

    # ---- TNOs ------------------------------------------------------------
    def tno_transform(self, t_cut):
        opt, pl = self.opt, self.pl
        self._pair_tables()
        pt = self._pt
        S_d = self.S_pao_d
        F_d = getattr(self, 'F_pao_d', None)
        if F_d is None:
            F_d = self.F_pao_d = cp.asarray(self.orb.F_pao)
        ntrip = len(self.trip)
        npao = np.array([len(p) for p in self.t_paos], dtype=np.int64)
        P = padded(npao)
        self.X_tno = [None] * ntrip
        self.e_tno = [None] * ntrip
        min_tnos = opt['MIN_TNOS']
        s_cut = opt['S_CUT']
        trip = np.array(self.trip, dtype=np.int64).reshape(-1, 3)
        nbf = self.orb.nbf
        for Pv in np.unique(P):
            sel = np.where(P == Pv)[0]
            per = 8 * (10 * Pv * Pv + 2 * nbf * Pv + 3 * pt['PP'] * Pv)
            step = max(1, int(self.MAX_CHUNK_BYTES // per))
            for c0 in range(0, len(sel), step):
                ch = sel[c0:c0 + step]
                b = len(ch)
                idx = np.zeros((b, Pv), np.int64)
                msk = np.zeros((b, Pv))
                for s, t in enumerate(ch):
                    idx[s, :npao[t]] = self.t_paos[t]
                    msk[s, :npao[t]] = 1.
                gi = cp.asarray(idx)
                m = cp.asarray(msk)
                mm = m[:, :, None] * m[:, None, :]
                St = S_d[gi[:, :, None], gi[:, None, :]] * mm
                Ft = F_d[gi[:, :, None], gi[:, None, :]] * mm
                dg = cp.diagonal(St, axis1=1, axis2=2)
                dn = cp.where(m > 0, 1. / cp.sqrt(cp.where(m > 0, dg, 1.)), 0.)
                Sn = St * dn[:, :, None] * dn[:, None, :]
                w, U = eigh_batched(Sn)
                keep = w > s_cut
                X = dn[:, :, None] * U * cp.where(keep, 1. / cp.sqrt(cp.where(keep, w, 1.)), 0.)[:, None, :]
                Z = cp.matmul(S_d[:, gi].transpose(1, 0, 2) * m[:, None, :], X)   # (b, nbf, Pv)
                Dt = cp.zeros((b, Pv, Pv))
                tr = trip[ch]
                for (a, c) in ((0, 1), (1, 2), (0, 2)):
                    pp = cp.asarray(pl.ij[tr[:, a], tr[:, c]])
                    sp = self._project(Z, cp.arange(b), pp)              # (b, VP, Pv)
                    Dt += cp.matmul(sp.transpose(0, 2, 1), cp.matmul(pt['D'][pp], sp))
                Dt /= 3.
                Dt -= cp.eye(Pv)[None] * (~keep)[:, :, None]
                occ, Vt = eigh_batched(Dt)
                occ, Vt = occ[:, ::-1], Vt[:, :, ::-1]
                nkeep = keep.sum(axis=1).get()
                cut = cp.asarray(self.tno_scale[ch] * t_cut)
                cnt = cp.sum((cp.abs(occ) >= cut[:, None])
                             & (cp.arange(Pv)[None, :] < cp.asarray(nkeep)[:, None]), axis=1).get()
                nfin = np.maximum(1, np.maximum(cnt, np.minimum(min_tnos, nkeep)))
                Nt = int(nfin.max())
                cm = (cp.arange(Nt)[None, :] < cp.asarray(nfin)[:, None]).astype(cp.float64)
                Uq = cp.matmul(X, Vt[:, :, :Nt]) * cm[:, None, :]
                Fq = cp.matmul(Uq.transpose(0, 2, 1), cp.matmul(Ft, Uq))
                Fq += 1e6 * cp.eye(Nt)[None] * (1 - cm)[:, :, None]
                et, Wt = eigh_batched(Fq)
                Xt = cp.matmul(Uq, Wt)
                for s, t in enumerate(ch):
                    self.X_tno[t] = Xt[s, :npao[t], :nfin[s]]
                    self.e_tno[t] = et[s, :nfin[s]]

    # ---- (T0) ----------------------------------------------------------
    def _trip_riatoms(self, prescreen):
        opt = self.opt
        mkn = opt['T_CUT_MKN_TRIPLES_PRE'] if prescreen else opt['T_CUT_MKN_TRIPLES']
        ria = [np.where(np.abs(self.mkn[:, i]) > mkn)[0] for i in range(self.n)]
        return [np.unique(np.concatenate([ria[i], ria[j], ria[k]])) for (i, j, k) in self.trip]

    def triples_sparsity(self, prescreen):
        super().triples_sparsity(prescreen)
        self.t_riatoms = self._trip_riatoms(prescreen)

    def _chunks(self):
        """Group triplets by padded (V, P) and split by memory."""
        nt = np.array([x.shape[1] for x in self.X_tno])
        npao = np.array([len(p) for p in self.t_paos])
        nl = np.array([len(l) for l in self.t_lmos])
        naux = np.array([len(self.dfi.aux_index(a)) for a in self.t_riatoms])
        Vp, Pp = padded(nt), padded(npao)
        key = Vp * 100000 + Pp
        order = np.lexsort((naux, key))
        nbf = self.orb.nbf
        chunks = []
        cur = []
        for t in order:
            if cur and key[t] != key[cur[0]]:
                chunks.append(cur)
                cur = []
            cur.append(t)
            V, P = int(Vp[t]), int(Pp[t])
            A = int(-(-max(naux[c] for c in cur) // 32) * 32)
            L = int(padded(max(nl[c] for c in cur)))
            per = 8 * (A * V * V + 3 * A * P + A * A + 14 * V ** 3
                       + 3 * L * V * V + nbf * V + 3 * A * V)
            if len(cur) * per > self.MAX_CHUNK_BYTES and len(cur) > 1:
                cur.pop()
                chunks.append(cur)
                cur = [t]
        if cur:
            chunks.append(cur)
        return chunks, nt, npao, nl, naux

    def lccsd_t0(self, save=False):
        pl, f = self.pl, self.orb.F_lmo
        dfi = self.dfi
        self._pair_tables()
        pt = self._pt
        ntrip = len(self.trip)
        self.e_ijk = np.zeros(ntrip)
        if save:
            self.W3, self.V3, self.T3 = [None] * ntrip, [None] * ntrip, [None] * ntrip
        fd = np.diag(f)
        trip = np.array(self.trip, dtype=np.int64).reshape(-1, 3)
        chunks, ntv, npao, nlv, naux = self._chunks()
        ext = dfi.ext
        S_d = self.S_pao_d
        for ch in chunks:
            ch = np.asarray(ch)
            B = len(ch)
            V = int(padded(ntv[ch].max()))
            P = int(padded(npao[ch].max()))
            L = int(padded(max(1, nlv[ch].max())))
            A = int(-(-naux[ch].max() // 32) * 32)
            # ---- index tables ------------------------------------------
            pidx = np.zeros((B, P), np.int64)
            pm = np.zeros((B, P))
            lidx = np.zeros((B, L), np.int64)
            lmk = np.zeros((B, L))
            aidx = np.zeros((B, A), np.int64)
            amk = np.zeros((B, A))
            ev = np.full((B, V), 1e3)
            coef = np.zeros((B, 6))
            offs = []
            X = cp.zeros((B, P, V))
            for s, t in enumerate(ch):
                pidx[s, :npao[t]] = self.t_paos[t]
                pm[s, :npao[t]] = 1.
                lidx[s, :nlv[t]] = self.t_lmos[t]
                lmk[s, :nlv[t]] = 1.
                ax = dfi.aux_index(self.t_riatoms[t])
                aidx[s, :len(ax)] = ax
                amk[s, :len(ax)] = 1.
                ev[s, :ntv[t]] = self.e_tno[t].get() if hasattr(self.e_tno[t], 'get') else self.e_tno[t]
                X[s, :npao[t], :ntv[t]] = self.X_tno[t]
                coef[s] = _energy_coeffs(*trip[t])
                o, at_off = 0, {}
                for a in self.t_riatoms[t]:
                    at_off[int(a)] = o
                    o += len(dfi.atom_ribfs[a])
                offs.append(at_off)
            gp, gpm = cp.asarray(pidx), cp.asarray(pm)
            gl, glm = cp.asarray(lidx), cp.asarray(lmk)
            ga, gam = cp.asarray(aidx), cp.asarray(amk)
            ijk = trip[ch]
            # ---- metric -------------------------------------------------
            J = dfi.J2c[ga[:, :, None], ga[:, None, :]] * (gam[:, :, None] * gam[:, None, :])
            J += cp.eye(A)[None] * (1 - gam)[:, :, None]
            Linv = cp.linalg.inv(cp.linalg.cholesky(J))
            # ---- (Q|r a), (Q|r l) for r in i, j, k --------------------------
            qv, qo = [], []
            for r in range(3):
                lmo = cp.asarray(ijk[:, r])
                q = dfi.qia[ga[:, :, None], lmo[:, None, None], gp[:, None, :]]
                q *= gam[:, :, None] * gpm[:, None, :]
                q = cp.matmul(Linv, cp.matmul(q, X))                    # (B, A, V)
                qv.append(q)
                o = dfi.qij[ga[:, :, None], lmo[:, None, None], gl[:, None, :]]
                o *= gam[:, :, None] * glm[:, None, :]
                qo.append(cp.matmul(Linv, o))                           # (B, A, L)
            # ---- (Q|ab) in the TNO basis, one aux atom at a time -----------
            qvv = cp.zeros((B, A, V, V))
            atoms = sorted({int(a) for t in ch for a in self.t_riatoms[t]})
            for a in atoms:
                qa = dfi.qab[a]
                if qa is None:
                    continue
                na = qa.shape[0]
                rows = np.array([s for s, t in enumerate(ch) if a in offs[s]], np.int64)
                off = np.array([offs[s][a] for s in rows], np.int64)
                pos = ext.pao_pos[a][pidx[rows]]
                pos = np.where(pm[rows] > 0, pos, 0)
                assert (pos >= 0).all(), 'triplet PAO outside the extended domain'
                gr = cp.asarray(rows)
                gpos = cp.asarray(pos)
                mrow = gpm[gr]
                Xr = X[gr]
                step = max(1, int((1 << 28) // (8 * na * P * P)))
                for r0 in range(0, len(rows), step):
                    sl = slice(r0, r0 + step)
                    blk = qa[:, gpos[sl][:, :, None], gpos[sl][:, None, :]]  # (na, b, P, P)
                    blk = blk.transpose(1, 0, 2, 3)
                    blk = blk * (mrow[sl][:, :, None] * mrow[sl][:, None, :])[:, None]
                    xr = Xr[sl]
                    tq = cp.matmul(cp.matmul(xr.transpose(0, 2, 1)[:, None], blk), xr[:, None])
                    oo = cp.asarray(off[sl])[:, None] + cp.arange(na)[None, :]
                    qvv[gr[sl][:, None], oo] = tq
            qvv = cp.matmul(Linv, qvv.reshape(B, A, V * V)).reshape(B, A, V, V)
            # (r a | b c) = sum_Q (L^-T L^-1 q)_Q,ra (Q|bc): qvv is half-fitted,
            # so contract with the half-fitted qv
            Kvvv = [cp.matmul(qv[r].transpose(0, 2, 1), qvv.reshape(B, A, V * V)).reshape(B, V, V, V)
                    for r in range(3)]
            Kov = {(a, c): cp.matmul(qo[a].transpose(0, 2, 1), qv[c])
                   for a in range(3) for c in range(3) if a != c}
            # ---- projected pair amplitudes --------------------------------
            Z = cp.matmul(S_d[:, gp].transpose(1, 0, 2) * gpm[:, None, :], X)   # (B, nbf, V)
            ar = cp.arange(B)
            W = cp.zeros((B, V, V, V))
            for r, (I, J_, K) in enumerate(_PERMS):
                pkj = cp.asarray(pl.ij[ijk[:, K], ijk[:, J_]])
                s = self._project(Z, ar, pkj)                           # (B, VP, V)
                Tkj = cp.matmul(s.transpose(0, 2, 1), cp.matmul(pt['T'][pkj], s))
                w = cp.matmul(Kvvv[I].reshape(B, V * V, V), Tkj.transpose(0, 2, 1))
                # t_il^{ab} (jl|kc) over the triplet's LMO domain
                pil = pl.ij[ijk[:, I][:, None], lidx]                   # (B, L)
                pil = np.where(lmk > 0, pil, 0)
                bb = cp.asarray(np.repeat(np.arange(B), L))
                gpil = cp.asarray(pil.reshape(-1))
                s = self._project(Z, bb, gpil)                          # (B*L, VP, V)
                Til = cp.matmul(s.transpose(0, 2, 1), cp.matmul(pt['T'][gpil], s))
                Til = Til.reshape(B, L, V * V) * glm[:, :, None]
                w -= cp.matmul(Til.transpose(0, 2, 1), Kov[(J_, K)])
                W += w.reshape(B, V, V, V).transpose((0,) + tuple(a + 1 for a in _BACK[r]))
            # ---- V, T3, energy -------------------------------------------
            t1 = []
            for r in range(3):
                pdiag = cp.asarray(pl.ij[ijk[:, r], ijk[:, r]])
                s = self._project(Z, ar, pdiag)
                t1.append(cp.einsum('bpv,bp->bv', s, pt['T1'][cp.asarray(ijk[:, r])]))
            Kjk = cp.matmul(qv[1].transpose(0, 2, 1), qv[2])
            Kik = cp.matmul(qv[0].transpose(0, 2, 1), qv[2])
            Kij = cp.matmul(qv[0].transpose(0, 2, 1), qv[1])
            Vt = (W + t1[0][:, :, None, None] * Kjk[:, None]
                  + t1[1][:, None, :, None] * Kik[:, :, None, :]
                  + t1[2][:, None, None, :] * Kij[:, :, :, None])
            e = cp.asarray(ev)
            fs = cp.asarray(fd[ijk].sum(axis=1))
            Dn = (e[:, :, None, None] + e[:, None, :, None] + e[:, None, None, :]
                  - fs[:, None, None, None])
            T3 = -W / Dn
            eb = _batched_energy(Vt, T3, cp.asarray(coef)).get()
            self.e_ijk[ch] = eb
            if save:
                for s, t in enumerate(ch):
                    v = ntv[t]
                    self.W3[t] = W[s, :v, :v, :v].copy()
                    self.V3[t] = Vt[s, :v, :v, :v].copy()
                    self.T3[t] = T3[s, :v, :v, :v].copy()
        return float(self.e_ijk.sum())

    # ---- iterative (T) --------------------------------------------------
    def lccsd_t_iterations(self):
        opt, f, n = self.opt, self.orb.F_lmo, self.n
        ntrip = len(self.trip)
        S_d = self.S_pao_d
        e_old = np.zeros(ntrip)
        e_prev = 0.
        coup = [[] for _ in range(ntrip)]
        for t, (i, j, k) in enumerate(self.trip):
            for l in range(n):
                for key, fl, ok in (((i, j, l), f[l, k], l != k),
                                    ((i, l, k), f[l, j], l != j),
                                    ((l, j, k), f[l, i], l != i)):
                    if ok and key in self.trip_idx and abs(fl) >= opt['F_CUT_T']:
                        coup[t].append((self.trip_idx[key], fl, _perm_case(*key)))
        Sxs = {}

        def s_tt(t, u):
            v = Sxs.get((t, u))
            if v is None:
                pt_, pu = cp.asarray(self.t_paos[t]), cp.asarray(self.t_paos[u])
                v = self.X_tno[t].T @ S_d[pt_[:, None], pu[None, :]] @ self.X_tno[u]
                Sxs[(t, u)] = v
            return v

        Dn = []
        for t, (i, j, k) in enumerate(self.trip):
            e = self.e_tno[t]
            Dn.append(e[:, None, None] + e[None, :, None] + e[None, None, :]
                      - (f[i, i] + f[j, j] + f[k, k]))
        coefs = [cp.asarray(_energy_coeffs(*t)) for t in self.trip]
        for it in range(1, opt['MAXITER'] + 1):
            rms = np.zeros(ntrip)
            for t in range(ntrip):
                if abs(self.e_ijk[t] - e_old[t]) < abs(e_old[t] * opt['T_CUT_ITER']):
                    continue
                R = self.W3[t] + self.T3[t] * Dn[t]
                for u, fl, case in coup[t]:
                    s = s_tt(t, u)
                    A = self.T3[u].transpose(_CASES[case])
                    v = s.shape[0]
                    A = (s @ A.reshape(A.shape[0], -1)).reshape(v, A.shape[1], A.shape[2])
                    A = (A.reshape(-1, A.shape[2]) @ s.T).reshape(v, -1, v)
                    A = cp.einsum('bB,aBc->abc', s, A)
                    R -= fl * A
                self.T3[t] = self.T3[t] - R / Dn[t]
                rms[t] = float(cp.sqrt(cp.mean(R * R)))
            e_old = self.e_ijk.copy()
            E = 0.
            for t in range(ntrip):
                V = self.V3[t]
                e = sum(float(coefs[t][c]) * float(cp.vdot(V.transpose(_CASES[c]), self.T3[t]))
                        for c in range(6) if float(coefs[t][c]) != 0)
                self.e_ijk[t] = e
                E += e
            self.log(f'  GPU LCCSD(T) iter {it:3d} {E:.12f} {E - e_prev:.3e} {rms.max():.3e}')
            if rms.max() < opt['R_CONVERGENCE'] and abs(E - e_prev) < opt['E_CONVERGENCE']:
                break
            e_prev = E
        else:
            raise RuntimeError('(T) iterations did not converge')
        return E
