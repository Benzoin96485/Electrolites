"""Pair prescreening, PNOs and the PNO-LMP2 on the device (closed shell).

Batched versions of ``ref_rhf``'s ``sc_mp2_pair_energies``, ``lmp2_pnos``,
``lmp2_iterations`` and ``recompute_pnos``: pairs of equal padded PAO-domain
size are processed together, the local fitting metric of a chunk is one
batched Cholesky, and the PAO orthogonalisation, PAO canonicalisation, PNO
and PNO-canonicalisation eigenproblems are each one ``eigh_batched`` call.

Only the PNO truncation rule (Psi4's sequential occupation / trace / energy
test, Jiang et al. 2024) runs on the host, over a vector per pair.
"""
import numpy as np
import cupy as cp

from electrolites.dlpno.ragged import Layout, Ragged, ChainPlan, op, padded
from electrolites.dlpno.linalg import eigh_batched, orthonormal_domain


def pno_overlaps(X, pao, S_pao_d, nbf, cp_, cq_, LS, nv):
    """S(p,q) = X_p^T S_pao[d_p, d_q] X_q for couples (cp_, cq_) -> Ragged(LS)."""
    npair = len(X)
    LX = Layout(np.full(npair, nbf), nv)
    pr = LX.shapes[0][0]
    Xs = Ragged(LX)
    for g, m in enumerate(LX.members):
        buf = cp.zeros((len(m),) + LX.shapes[g])
        for s, p in enumerate(m):
            buf[s, cp.asarray(pao[p]), :nv[p]] = cp.asarray(X[p])
        Xs.data[g] = buf
    Sp = cp.zeros((pr, pr))
    Sp[:nbf, :nbf] = S_pao_d
    SX = Ragged(LX, [cp.matmul(Sp, d) for d in Xs.data])
    S = Ragged(LS)
    ChainPlan(LS, np.arange(len(cp_)), [op(LX, cp_, True), op(LX, cq_)]).run(S, [Xs, SX])
    return S


def _truncate(occ, ecum, e_tot, occ_total, cut, trace, energy, min_n, lag):
    """Psi4's PNO selection for one pair.  occ: descending occupations,
    ecum[m]: pair energy of the first m PNOs.  lag=True is the LMP2 variant
    (energy tested lags one PNO), False the CC one."""
    n = len(occ)
    occ_pno = 0.
    nfin = 0
    prefix = True
    last = 0
    for a in range(n):
        if prefix:
            e_pno = ecum[max(a - 1, 0)] if lag else ecum[a]
        else:
            return None                     # caller falls back to exact sets
        ok = (abs(occ[a]) >= cut or occ_pno / occ_total < trace
              or abs(e_pno) < energy * abs(e_tot) or a < min_n)
        if ok:
            if a != last:
                prefix = False
            occ_pno += occ[a]
            nfin += 1
            last = a + 1
    return max(1, nfin)


def _truncate_exact(occ, Kn, Ttn, e_tot, occ_total, cut, trace, energy, min_n, lag):
    n = len(occ)
    e_pno = occ_pno = 0.
    a_curr = []
    nfin = 0
    for a in range(n):
        if (abs(occ[a]) >= cut or occ_pno / occ_total < trace
                or abs(e_pno) < energy * abs(e_tot) or a < min_n):
            if lag:
                e_pno = np.sum(Kn[np.ix_(a_curr, a_curr)] * Ttn[np.ix_(a_curr, a_curr)])
                a_curr.append(a)
            else:
                a_curr.append(a)
                e_pno = np.sum(Kn[np.ix_(a_curr, a_curr)] * Ttn[np.ix_(a_curr, a_curr)])
            occ_pno += occ[a]
            nfin += 1
    return max(1, nfin)


class LMP2GPU:
    """Mixin for rhf.DLPNO."""

    MAX_PAIR_CHUNK_BYTES = 1 << 30
    #: 'pivchol' (Psi4's partial Cholesky selection) or 'eig'
    PAO_ORTHO = 'pivchol'

    def _canon_pairs(self):
        return [p for p, (i, j) in enumerate(self.pl.pairs) if i <= j]

    def _pair_chunks(self, plist):
        npao = np.array([len(self.pao[p]) for p in plist])
        naux = np.array([len(self.dfi.aux_index(self.riatoms[p])) for p in plist])
        P = padded(npao)
        order = np.lexsort((naux, P))
        chunks, cur = [], []
        for o in order:
            if cur and P[o] != P[cur[0]]:
                chunks.append(cur)
                cur = []
            cur.append(o)
            A = int(-(-naux[cur].max() // 32) * 32)
            per = 8 * (A * A + 3 * A * int(P[o]) + 14 * int(P[o]) ** 2)
            if len(cur) > 1 and len(cur) * per > self.MAX_PAIR_CHUNK_BYTES:
                cur.pop()
                chunks.append(cur)
                cur = [o]
        if cur:
            chunks.append(cur)
        return [np.asarray(plist)[c] for c in chunks]

    def _sc_chunk(self, ch, pno=None):
        """Semicanonical pair energies of a chunk of canonical pairs; with
        ``pno`` = (cut, trace, energy) also their LMP2 PNOs."""
        dfi, orb, opt = self.dfi, self.orb, self.opt
        pl = self.pl
        B = len(ch)
        npao = np.array([len(self.pao[p]) for p in ch])
        P = int(padded(npao.max()))
        auxs = [dfi.aux_index(self.riatoms[p]) for p in ch]
        A = int(-(-max(len(a) for a in auxs) // 32) * 32)
        pidx = np.zeros((B, P), np.int64)
        pm = np.zeros((B, P))
        aidx = np.zeros((B, A), np.int64)
        am = np.zeros((B, A))
        for s, p in enumerate(ch):
            pidx[s, :npao[s]] = self.pao[p]
            pm[s, :npao[s]] = 1.
            aidx[s, :len(auxs[s])] = auxs[s]
            am[s, :len(auxs[s])] = 1.
        ii = np.array([pl.pairs[p][0] for p in ch])
        jj = np.array([pl.pairs[p][1] for p in ch])
        gp, gpm = cp.asarray(pidx), cp.asarray(pm)
        ga, gam = cp.asarray(aidx), cp.asarray(am)
        J = dfi.J2c[ga[:, :, None], ga[:, None, :]] * (gam[:, :, None] * gam[:, None, :])
        J += cp.eye(A)[None] * (1 - gam)[:, :, None]
        Linv = cp.linalg.inv(cp.linalg.cholesky(J))
        mq = gam[:, :, None] * gpm[:, None, :]
        iq = cp.matmul(Linv, dfi.qia[ga[:, :, None], cp.asarray(ii)[:, None, None], gp[:, None, :]] * mq)
        jq = cp.matmul(Linv, dfi.qia[ga[:, :, None], cp.asarray(jj)[:, None, None], gp[:, None, :]] * mq)
        K = cp.matmul(iq.transpose(0, 2, 1), jq)
        del iq, jq, Linv, J
        mm = gpm[:, :, None] * gpm[:, None, :]
        S = self.S_pao_d[gp[:, :, None], gp[:, None, :]] * mm
        F = self.F_pao_d[gp[:, :, None], gp[:, None, :]] * mm
        X0, keep = orthonormal_domain(S, npao, opt['S_CUT'], self.PAO_ORTHO)
        Fo = cp.matmul(X0.transpose(0, 2, 1), cp.matmul(F, X0))
        Fo += 1e6 * cp.eye(P)[None] * (~keep)[:, :, None]
        e, Vf = eigh_batched(Fo)
        X = cp.matmul(X0, Vf)
        valid = e < 1e5
        Kc = cp.matmul(X.transpose(0, 2, 1), cp.matmul(K, X))
        fd = np.diag(orb.F_lmo)
        fs = cp.asarray(fd[ii] + fd[jj])
        den = fs[:, None, None] - e[:, :, None] - e[:, None, :]
        vv = valid[:, :, None] & valid[:, None, :]
        T = cp.where(vv, Kc / cp.where(vv, den, 1.), 0.)
        Tt = 2 * T - T.transpose(0, 2, 1)
        e_ij = cp.sum(Kc * Tt, axis=(1, 2))
        if pno is None:
            return e_ij.get()
        cut, trace, energy = pno
        D = cp.matmul(Tt, T.transpose(0, 2, 1)) + cp.matmul(Tt.transpose(0, 2, 1), T)
        dmask = cp.asarray(ii == jj)
        D[dmask] *= .5
        D -= cp.eye(P)[None] * (~valid)[:, :, None]
        occ, Uo = eigh_batched(D)
        occ, Uo = occ[:, ::-1], Uo[:, :, ::-1]
        Kn = cp.matmul(Uo.transpose(0, 2, 1), cp.matmul(Kc, Uo))
        Ttn = cp.matmul(Uo.transpose(0, 2, 1), cp.matmul(Tt, Uo))
        C = cp.cumsum(cp.cumsum(Kn * Ttn, axis=1), axis=2)
        ecum = cp.concatenate([cp.zeros((B, 1)), cp.diagonal(C, axis1=1, axis2=2)], axis=1)
        nval = valid.sum(axis=1).get()
        occ_h, ecum_h, e_h = occ.get(), ecum.get(), e_ij.get()
        nfin = np.zeros(B, np.int64)
        for s, p in enumerate(ch):
            i, j = ii[s], jj[s]
            scale = 1.
            if i == j:
                scale *= opt['T_CUT_PNO_DIAG_SCALE']
            if i < self.ncore_scale or j < self.ncore_scale:
                scale *= opt['T_CUT_PNO_CORE_SCALE']
            o = occ_h[s, :nval[s]]
            args = (e_h[s], o.sum(), scale * cut, trace, energy, opt['MIN_PNOS'], self._lag)
            nf = _truncate(o, ecum_h[s], *args)
            if nf is None:
                nf = _truncate_exact(o, Kn[s].get(), Ttn[s].get(), *args)
            nfin[s] = nf
        Nv = int(nfin.max())
        cm = (cp.arange(Nv)[None, :] < cp.asarray(nfin)[:, None]).astype(cp.float64)
        Uq = Uo[:, :, :Nv] * cm[:, None, :]
        ee = cp.where(valid, e, 0.)
        Fq = cp.matmul(Uq.transpose(0, 2, 1) * ee[:, None, :], Uq)
        Fq += 1e6 * cp.eye(Nv)[None] * (1 - cm)[:, :, None]
        ep, Wp = eigh_batched(Fq)
        Up = cp.matmul(Uq, Wp)
        Xp = cp.matmul(X, Up)
        Kp = cp.matmul(Up.transpose(0, 2, 1), cp.matmul(Kc, Up))
        Tp = cp.matmul(Up.transpose(0, 2, 1), cp.matmul(T, Up))
        de = e_ij - cp.sum(Kp * (2 * Tp - Tp.transpose(0, 2, 1)), axis=(1, 2))
        out = []
        de = de.get()
        for s, p in enumerate(ch):
            v = nfin[s]
            out.append((p, Xp[s, :npao[s], :v].get(), ep[s, :v].get(),
                        Kp[s, :v, :v].get(), Tp[s, :v, :v].get(), de[s]))
        return e_h, out

    # ------------------------------------------------------------------
    def sc_mp2_pair_energies(self):
        self._prep_dev()
        pl = self.pl
        e = np.zeros(len(pl))
        for ch in self._pair_chunks(self._canon_pairs()):
            v = self._sc_chunk(ch)
            e[ch] = v
            e[pl.ji[ch]] = v
        return e

    def _prep_dev(self):
        if getattr(self, 'F_pao_d', None) is None:
            self.F_pao_d = cp.asarray(self.orb.F_pao)

    def lmp2_pnos(self):
        self._prep_dev()
        opt, pl = self.opt, self.pl
        npair = len(pl)
        self.X = [None] * npair
        self.e_pno = [None] * npair
        self.K = [None] * npair
        self.T = [None] * npair
        self.de_pno = np.zeros(npair)
        self._lag = True
        pno = (opt['T_CUT_PNO_MP2'], opt['T_CUT_TRACE_MP2'], opt['T_CUT_ENERGY_MP2'])
        for ch in self._pair_chunks(self._canon_pairs()):
            _, res = self._sc_chunk(ch, pno)
            for p, X, ep, K, T, de in res:
                self.de_pno[p] = de
                self._store_pair(p, X, ep, K, T)
        self.de_pno_total = self.de_pno.sum()

    # ------------------------------------------------------------------
    def lmp2_iterations(self):
        opt, pl = self.opt, self.pl
        f = self.orb.F_lmo
        n = self.n
        npair = len(pl)
        nv = np.array([x.shape[1] for x in self.X], dtype=np.int64)
        LP = Layout(nv, nv)
        cps, cqs, outs, ws = [], [], [], []
        for p, (i, j) in enumerate(pl.pairs):
            for k in range(n):
                kj, ik = pl.ij[k, j], pl.ij[i, k]
                if kj >= 0 and i != k and abs(f[i, k]) > opt['F_CUT']:
                    cps.append(p); cqs.append(kj); outs.append(kj); ws.append(-f[i, k])
                if ik >= 0 and j != k and abs(f[k, j]) > opt['F_CUT']:
                    cps.append(p); cqs.append(ik); outs.append(ik); ws.append(-f[k, j])
        cps, cqs = np.array(cps, np.int64), np.array(cqs, np.int64)
        LS = Layout(nv[cps], nv[cqs])
        S = pno_overlaps(self.X, self.pao, self.S_pao_d, self.orb.nbf, cps, cqs, LS, nv)
        cid = np.arange(len(cps))
        plan = ChainPlan(LP, cps, [op(LS, cid), op(LP, cqs), op(LS, cid, True)])
        w = cp.asarray(np.array(ws))
        K = Ragged.from_list(LP, self.K)
        T = Ragged.from_list(LP, self.T)
        fd = np.diag(f)
        D = []
        for g, m in enumerate(LP.members):
            pv = LP.shapes[g][0]
            e = np.zeros((len(m), pv))
            for s, p in enumerate(m):
                e[s, :nv[p]] = self.e_pno[p]
            pi = np.array([pl.pairs[p][0] for p in m])
            pj = np.array([pl.pairs[p][1] for p in m])
            d = e[:, :, None] + e[:, None, :] - (fd[pi] + fd[pj])[:, None, None]
            D.append(cp.asarray(np.where(LP.mask()[g].get() > 0, d, 1.)))
        D = Ragged(LP, D)
        from electrolites.dlpno.ccsd_gpu import _DeviceDIIS
        diis = _DeviceDIIS(opt['DIIS_MAX_VECS'])
        ec, rc = 0.01 * opt['E_CONVERGENCE'], 0.01 * opt['R_CONVERGENCE']
        e_prev = 0.
        for it in range(1, opt['MAXITER'] + 1):
            R = K + D * T
            plan.run(R, [S, T, S], w=w)
            rms = float(R.rms_per_item().max())
            T = T - R / D
            T.set_flat(diis.update(T.flat(), R.flat()))
            e_ij = K.vdot_items(T * 2. - T.T())
            e_cur = float(e_ij.sum())
            self.log(f'  GPU LMP2 iter {it:3d} {e_cur:.12f} {e_cur - e_prev:.3e} {rms:.3e}')
            if abs(e_cur - e_prev) < ec and rms < rc:
                break
            e_prev = e_cur
        else:
            raise RuntimeError('LMP2 did not converge')
        self.T = T.to_list()
        self.e_lmp2 = e_cur
        return e_ij.get()

    # ------------------------------------------------------------------
    def recompute_pnos(self):
        opt, pl = self.opt, self.pl
        canon = self._canon_pairs()
        nv = np.array([self.X[p].shape[1] for p in canon])
        V = padded(nv)
        for Vv in np.unique(V):
            ch = np.asarray(canon)[V == Vv]
            B = len(ch)
            Vv = int(Vv)
            Tb = np.zeros((B, Vv, Vv))
            Kb = np.zeros((B, Vv, Vv))
            eb = np.zeros((B, Vv))
            mk = np.zeros((B, Vv))
            for s, p in enumerate(ch):
                v = self.X[p].shape[1]
                Tb[s, :v, :v] = self.T[p]
                Kb[s, :v, :v] = self.K[p]
                eb[s, :v] = self.e_pno[p]
                mk[s, :v] = 1.
            T, K, e, m = (cp.asarray(x) for x in (Tb, Kb, eb, mk))
            Tt = 2 * T - T.transpose(0, 2, 1)
            e_tot = cp.sum(K * Tt, axis=(1, 2))
            D = cp.matmul(Tt, T.transpose(0, 2, 1)) + cp.matmul(Tt.transpose(0, 2, 1), T)
            ii = np.array([pl.pairs[p][0] for p in ch])
            jj = np.array([pl.pairs[p][1] for p in ch])
            D[cp.asarray(ii == jj)] *= .5
            D -= cp.eye(Vv)[None] * (1 - m)[:, :, None]
            occ, U = eigh_batched(D)
            occ, U = occ[:, ::-1], U[:, :, ::-1]
            Kn = cp.matmul(U.transpose(0, 2, 1), cp.matmul(K, U))
            Ttn = cp.matmul(U.transpose(0, 2, 1), cp.matmul(Tt, U))
            C = cp.cumsum(cp.cumsum(Kn * Ttn, axis=1), axis=2)
            ecum = cp.concatenate([cp.zeros((B, 1)), cp.diagonal(C, axis1=1, axis2=2)], axis=1)
            occ_h, ecum_h, et_h = occ.get(), ecum.get(), e_tot.get()
            nfin = np.zeros(B, np.int64)
            for s, p in enumerate(ch):
                i, j = ii[s], jj[s]
                scale = 1.
                if i == j:
                    scale *= opt['T_CUT_PNO_DIAG_SCALE']
                if i < self.ncore_scale or j < self.ncore_scale:
                    scale *= opt['T_CUT_PNO_CORE_SCALE']
                o = occ_h[s, :self.X[p].shape[1]]
                args = (et_h[s], o.sum(), scale * opt['T_CUT_PNO'], opt['T_CUT_TRACE'],
                        opt['T_CUT_ENERGY'], opt['MIN_PNOS'], False)
                # CC variant: the energy tested includes the PNO itself
                nf = _truncate(o, ecum_h[s], *args)
                if nf is None:
                    nf = _truncate_exact(o, Kn[s].get(), Ttn[s].get(), *args)
                nfin[s] = nf
            Nv = int(nfin.max())
            cm = (cp.arange(Nv)[None, :] < cp.asarray(nfin)[:, None]).astype(cp.float64)
            Uq = U[:, :, :Nv] * cm[:, None, :]
            Fq = cp.matmul(Uq.transpose(0, 2, 1) * e[:, None, :], Uq)
            Fq += 1e6 * cp.eye(Nv)[None] * (1 - cm)[:, :, None]
            ep, Wp = eigh_batched(Fq)
            Up = cp.matmul(Uq, Wp)
            Kp = cp.matmul(Up.transpose(0, 2, 1), cp.matmul(K, Up))
            Tp = cp.matmul(Up.transpose(0, 2, 1), cp.matmul(T, Up))
            de = (e_tot - cp.sum(Kp * (2 * Tp - Tp.transpose(0, 2, 1)), axis=(1, 2))).get()
            Up_h, ep_h, Kp_h, Tp_h = Up.get(), ep.get(), Kp.get(), Tp.get()
            for s, p in enumerate(ch):
                v, vo = nfin[s], self.X[p].shape[1]
                self.de_pno[p] += de[s]
                dpn = self.de_pno[p]
                self._store_pair(p, self.X[p] @ Up_h[s, :vo, :v], ep_h[s, :v],
                                 Kp_h[s, :v, :v], Tp_h[s, :v, :v])
                self.de_pno[pl.ji[p]] = dpn
        self._S = {}
        self.de_pno_total = self.de_pno.sum()
        self.de_weak = sum(np.sum(self.K[p] * (2 * self.T[p] - self.T[p].T))
                           for p in range(len(pl)) if not self.strong[p])
