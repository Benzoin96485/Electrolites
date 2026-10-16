"""Batched, device-resident closed-shell DLPNO-CCSD iterations.

The equations are those of ``ref_rhf.RefDLPNO`` (Jiang et al. 2024, the
T1-dressed formulation Psi4 implements), term for term.  What changes is the
execution: every per-pair or per-(pair, LMO) loop of the reference becomes a
static list of terms, grouped once by the padded shapes of its operands
(``ragged.ChainPlan``) and replayed as strided-batched GEMMs every
iteration.  The PNO overlap matrices every term needs are formed once, as a
batched GEMM over pair couples, and kept on the device.
"""
import time

import numpy as np
import cupy as cp
import cupyx

from electrolites.dlpno.ragged import Layout, Ragged, ChainPlan, DotPlan, op


class _Couples:
    """Registry of the PNO overlap blocks S(p, q) the equations need."""

    def __init__(self):
        self.keys = {}
        self.p = []
        self.q = []

    def __call__(self, p, q):
        k = (p, q)
        c = self.keys.get(k)
        if c is None:
            c = self.keys[k] = len(self.p)
            self.p.append(p)
            self.q.append(q)
        return c

    def many(self, p, q):
        return np.array([self(int(a), int(b)) for a, b in zip(p, q)], dtype=np.int64)


class _Terms:
    """Accumulates index columns of one term list."""

    def __init__(self, *names):
        self.cols = {n: [] for n in names}

    def add(self, **kw):
        for k, v in kw.items():
            self.cols[k].append(v)

    def __getitem__(self, k):
        return np.asarray(self.cols[k], dtype=np.int64)

    def __len__(self):
        return len(next(iter(self.cols.values())))


def _partner_slots(layout, partner):
    """per group: slots of each member's partner item (must share the group)."""
    out = []
    for g, m in enumerate(layout.members):
        pp = partner[m]
        assert np.all(layout.group[pp] == g)
        out.append(cp.asarray(layout.slot[pp]))
    return out


class CCSDGPU:
    """Device data and plans for one set of CC PNOs (built from a RefDLPNO
    object after ``recompute_pnos`` and ``pno_integrals``)."""

    def __init__(self, ref, ints):
        self.ref = ref
        self.t_build = {}
        t0 = time.perf_counter()
        pl = ref.pl
        n = ref.n
        npair = len(pl)
        self.n, self.npair = n, npair
        nv = np.array([x.shape[1] for x in ref.X], dtype=np.int64)
        nl = np.array([len(l) for l in ref.lmos], dtype=np.int64)
        self.nv, self.nl = nv, nl
        ij = pl.ij
        self.ij = ij
        ji = pl.ji
        self.ji = ji
        dii = ij[np.arange(n), np.arange(n)]
        self.dii = dii
        pi = np.array([a for a, b in pl.pairs], dtype=np.int64)
        pj = np.array([b for a, b in pl.pairs], dtype=np.int64)
        self.pi, self.pj = pi, pj
        strong = np.asarray(ref.strong, dtype=bool)
        self.strong = strong
        canon = pi <= pj
        self.canon = canon

        LP = self.LP = Layout(nv, nv)
        LV = self.LV = Layout(nv, 1)
        LN = self.LN = Layout(nl, nv)
        LL = self.LL = Layout(nl, nl)
        LT1 = self.LT1 = Layout(nv[dii], 1)
        L11 = self.L11 = Layout(np.ones(npair, np.int64), 1)
        self.pslot = {id(L): _partner_slots(L, ji) for L in (LP, LV, LN, LL)}

        C = self.C = _Couples()
        pos = ref.lmo_pos
        lmos = ref.lmos

        # ---- term lists ----------------------------------------------------
        # T_n and S(p,ii) T1[i]
        tn = _Terms('p', 'row', 's', 'n')
        for p in range(npair):
            for r, m in enumerate(lmos[p]):
                tn.add(p=p, row=r, s=C(p, dii[m]), n=m)
        ti = _Terms('p', 's', 'i')
        tj = _Terms('p', 's', 'j')
        for p in range(npair):
            ti.add(p=p, s=C(p, dii[pi[p]]), i=pi[p])
            tj.add(p=p, s=C(p, dii[pj[p]]), j=pj[p])
        # Fkc (all pairs)
        fkc = _Terms('p', 's1', 'ik', 's2', 'k')
        for p in range(npair):
            i = pi[p]
            for k in lmos[p]:
                ik, kk = ij[i, k], dii[k]
                fkc.add(p=p, s1=C(p, ik), ik=ik, s2=C(ik, kk), k=k)
        # Fkj += Fkc_bar[jj] row i . T1[j]
        # gamma (all pairs p = (k, i))
        g3 = _Terms('p', 's1', 'l', 'i', 's2', 'kl', 's3')
        g4 = _Terms('p', 's1', 'li', 's2', 'kl', 's3')
        for p in range(npair):
            k, i = pi[p], pj[p]
            ii = dii[i]
            for l in lmos[p]:
                kl, ll, li = ij[k, l], dii[l], ij[l, i]
                g3.add(p=p, s1=C(p, ll), l=l, i=i, s2=C(kl, ii), kl=kl, s3=C(p, kl))
                g4.add(p=p, s1=C(p, li), li=li, s2=C(li, kl), kl=kl, s3=C(kl, p))
        # delta (all pairs p = (i, k))
        d3 = _Terms('p', 's1', 'l', 'i', 's2', 'lk', 's3')
        d4 = _Terms('p', 's1', 'il', 's2', 'lk', 's3')
        for p in range(npair):
            i, k = pi[p], pj[p]
            ii = dii[i]
            for l in lmos[p]:
                ll, lk, il = dii[l], ij[l, k], ij[i, l]
                d3.add(p=p, s1=C(p, ll), l=l, i=i, s2=C(lk, ii), lk=lk, s3=C(lk, p))
                d4.add(p=p, s1=C(p, il), il=il, s2=C(il, lk), lk=lk, s3=C(lk, p))
        # Fkj double tilde
        fdd = _Terms('p', 'li', 's1', 'lj', 's2', 'flat')
        for p in range(npair):
            i, j = pi[p], pj[p]
            for l in lmos[p]:
                li, lj = ij[l, i], ij[l, j]
                if li < 0 or lj < 0:
                    continue
                fdd.add(p=p, li=li, s1=C(li, lj), lj=lj, s2=C(lj, li), flat=i * n + j)
        # R1
        r1a = _Terms('i', 's', 'q')
        r1b = _Terms('i', 's', 'l', 'kl', 'q', 'sa', 'sb')
        r1c = _Terms('i', 's', 'p', 'q')
        for p in range(npair):
            i, k = pi[p], pj[p]
            q = ji[p]
            ii = dii[i]
            r1a.add(i=i, s=C(q, ii), q=q)
            for l in lmos[p]:
                kl, ll = ij[k, l], dii[l]
                r1b.add(i=i, s=C(ii, ll), l=l, kl=kl, q=q, sa=C(kl, q), sb=C(q, kl))
            r1c.add(i=i, s=C(p, ii), p=p, q=q)
        r1d = _Terms('i', 's', 'p', 'row')
        for p in range(npair):
            for r, i in enumerate(lmos[p]):
                r1d.add(i=i, s=C(p, dii[i]), p=p, row=r)
        # R2: B and Fbc over (k, l) in the pair's LMO domain
        rb = _Terms('p', 's', 'kl', 'bflat')
        for p in np.where(strong & canon)[0]:
            lm = lmos[p]
            L = len(lm)
            for a, k in enumerate(lm):
                for b, l in enumerate(lm):
                    kl = ij[k, l]
                    if kl < 0:
                        continue
                    rb.add(p=p, s=C(kl, p), kl=kl, bflat=a * L + b)
        rc = _Terms('p', 'jk_cpl', 'kj', 's1', 'ki', 's2', 's3')
        rd = _Terms('p', 'jk_cpl', 'jk', 's1', 'ik', 's2', 's3')
        rg = _Terms('p', 's', 'ik', 'fflat')
        for p in np.where(strong)[0]:
            i, j = pi[p], pj[p]
            for kk, k in enumerate(lmos[p]):
                ki, kj, ik, jk = ij[k, i], ij[k, j], ij[i, k], ij[j, k]
                rc.add(p=p, jk_cpl=-1, kj=kj, s1=C(p, ki), ki=ki, s2=C(ki, kj), s3=C(kj, p))
                rd.add(p=p, jk_cpl=-1, jk=jk, s1=C(p, ik), ik=ik, s2=C(ik, jk), s3=C(jk, p))
            for k in range(n):
                ik = ij[i, k]
                if ik < 0:
                    continue
                rg.add(p=p, s=C(p, ik), ik=ik, fflat=k * n + j)
        # energy
        self.t_build['terms'] = time.perf_counter() - t0
        t0 = time.perf_counter()

        # ---- PNO overlaps --------------------------------------------------
        cp_, cq_ = np.array(C.p, np.int64), np.array(C.q, np.int64)
        LS = self.LS = Layout(nv[cp_], nv[cq_])
        self.S = self._overlaps(cp_, cq_)
        self.t_build['overlaps'] = time.perf_counter() - t0
        t0 = time.perf_counter()

        # ---- plans ---------------------------------------------------------
        P = ChainPlan
        self.p_tn = P(LN, tn['p'], [op(LS, tn['s']), op(LT1, tn['n'])], out_row=tn['row'])
        self.p_ti = P(LV, ti['p'], [op(LS, ti['s']), op(LT1, ti['i'])])
        self.p_tj = P(LV, tj['p'], [op(LS, tj['s']), op(LT1, tj['j'])])
        allp = np.arange(npair)
        self.p_e1 = P(L11, allp, [op(LV, allp, True), op(LP, allp), op(LV, allp)])
        self.p_fkc = P(LV, fkc['p'], [op(LS, fkc['s1']), op(LP, fkc['ik']),
                                      op(LS, fkc['s2']), op(LT1, fkc['k'])])
        jj = dii[pj]
        rowj = np.array([pos[jj[p]][pi[p]] for p in range(npair)], dtype=np.int64)
        self.p_fkj = P(L11, allp, [op(LN, jj, row=rowj), op(LT1, pj)])
        # gamma
        self.p_g1 = P(LP, allp, [op(LN, allp, True), op(LN, allp)])
        self.p_g3 = P(LP, g3['p'], [op(LS, g3['s1']), op(LT1, g3['l']), op(LT1, g3['i'], True),
                                    op(LS, g3['s2'], True), op(LP, g3['kl']), op(LS, g3['s3'], True)])
        self.p_g4 = P(LP, g4['p'], [op(LS, g4['s1']), op(LP, g4['li']), op(LS, g4['s2']),
                                    op(LP, g4['kl']), op(LS, g4['s3'])])
        # delta
        self.p_d1 = P(LP, allp, [op(LN, allp, True), op(LN, allp)])
        self.p_d3 = P(LP, d3['p'], [op(LS, d3['s1']), op(LT1, d3['l']), op(LT1, d3['i'], True),
                                    op(LS, d3['s2'], True), op(LP, d3['lk']), op(LS, d3['s3'])])
        self.p_d4 = P(LP, d4['p'], [op(LS, d4['s1']), op(LP, d4['il']), op(LS, d4['s2']),
                                    op(LP, d4['lk']), op(LS, d4['s3'])])
        # Fkj''
        self.p_fdd = DotPlan([op(LP, fdd['li'])],
                             [op(LS, fdd['s1']), op(LP, fdd['lj']), op(LS, fdd['s2'])])
        self.fdd_flat = cp.asarray(fdd['flat'])
        # R1
        self.p_r1a = P(LT1, r1a['i'], [op(LS, r1a['s'], True), op(LV, r1a['q'])])
        self.p_r1b_w = DotPlan([op(LP, r1b['kl'])],
                               [op(LS, r1b['sa']), op(LP, r1b['q']), op(LS, r1b['sb'])])
        self.p_r1b = P(LT1, r1b['i'], [op(LS, r1b['s']), op(LT1, r1b['l'])])
        self.p_r1c = P(LT1, r1c['i'], [op(LS, r1c['s'], True), op(LP, r1c['p']), op(LV, r1c['q'])])
        self.p_B1 = P(LN, allp, [op(LN, allp), op(LP, allp, True)])
        self.p_B2 = P(LN, allp, [op(LN, allp), op(LP, allp), op(LP, allp, True)])
        self.p_r1d = P(LT1, r1d['i'], [op(LS, r1d['s'], True), op(LN, r1d['p'], True, row=r1d['row'])])
        # R2
        self.p_rb = P(LP, rb['p'], [op(LS, rb['s'], True), op(LP, rb['kl']), op(LS, rb['s'])])
        self.p_rf = P(LP, rb['p'], [op(LS, rb['s'], True), op(LP, rb['kl']), op(LP, rb['kl'], True),
                                    op(LS, rb['s'])])
        bflat_off = np.zeros(npair, np.int64)
        self.rb_beta_idx = (rb['p'], rb['bflat'])
        self.p_rc2 = P(LP, rc['p'], [op(LS, rc['s1']), op(LP, rc['ki']), op(LS, rc['s2']),
                                     op(LP, rc['kj'], True), op(LS, rc['s3'])])
        self.p_rd3 = P(LP, rd['p'], [op(LS, rd['s1']), op(LP, rd['ik']), op(LS, rd['s2']),
                                     op(LP, rd['jk'], True), op(LS, rd['s3'])])
        self.p_rg = P(LP, rg['p'], [op(LS, rg['s']), op(LP, rg['ik']), op(LS, rg['s'], True)])
        self.rg_f = cp.asarray(rg['fflat'])
        self.t_build['plans'] = time.perf_counter() - t0
        t0 = time.perf_counter()

        # ---- integrals -----------------------------------------------------
        self._load_ints(ints, rc, rd)
        self.t_build['ints'] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    def _overlaps(self, cp_, cq_):
        """S(p,q) = X_p^T S_pao[dom_p, dom_q] X_q for all registered couples,
        as one batched GEMM through PAO-embedded PNO coefficients."""
        ref = self.ref
        nbf = ref.orb.nbf
        npair = self.npair
        Spao = cp.asarray(ref.orb.S_pao)
        LX = Layout(np.full(npair, nbf), self.nv)
        Xs = Ragged(LX)
        for g, m in enumerate(LX.members):
            buf = np.zeros((len(m),) + LX.shapes[g])
            for s, p in enumerate(m):
                buf[s, ref.pao[p], :self.nv[p]] = ref.X[p]
            Xs.data[g] = cp.asarray(buf)
        SX = Ragged(LX, [cp.matmul(Spao, d) for d in Xs.data])
        S = Ragged(self.LS)
        plan = ChainPlan(self.LS, np.arange(len(cp_)), [op(LX, cp_, True), op(LX, cq_)])
        plan.run(S, [Xs, SX])
        return S

    def _load_ints(self, ints, rc, rd):
        """Upload the PNO-basis integrals of the reference (pno_integrals)."""
        ref = self.ref
        LP, LN = self.LP, self.LN
        self.K = Ragged.from_list(LP, ref.K)
        self.Lij = Ragged.from_list(LP, ref.L_iajb)
        self.Kmibj = Ragged.from_list(LN, ref.K_mibj)
        self.Jijmb = Ragged.from_list(LN, ref.J_ijmb)
        # K_ivvv as (nv, nv, nv) per LP group
        self.K3 = []
        for g, m in enumerate(LP.members):
            pv = LP.shapes[g][0]
            buf = np.zeros((len(m), pv, pv, pv))
            for s, p in enumerate(m):
                v = self.nv[p]
                buf[s, :v, :v, :v] = ref.K_ivvv[p].reshape(v, v, v)
            self.K3.append(cp.asarray(buf))
        # non-projected integrals, per (strong p, k in lmos[p])
        pk_p, pk_k, pk_r = [], [], []
        for p in np.where(self.strong)[0]:
            for kk, k in enumerate(ref.lmos[p]):
                pk_p.append(p)
                pk_k.append(k)
                pk_r.append(kk)
        pk_p = np.array(pk_p, np.int64)
        pk_k = np.array(pk_k, np.int64)
        kj = self.ij[pk_k, self.pj[pk_p]]
        LJ = self.LJ = Layout(self.nv[pk_p], self.nv[kj])
        self.Jikac = Ragged.from_list(LJ, [ref.J_ikac[p][r] for p, r in zip(pk_p, pk_r)])
        self.Kiakc = Ragged.from_list(LJ, [ref.K_iakc[p][r] for p, r in zip(pk_p, pk_r)])
        cid = np.arange(len(pk_p))
        P = ChainPlan
        LS = self.LS
        # C: -J(p,k) T[kj]^T S(kj,p) ; D: (2K - J)(p,k) Tt[jk]^T S(jk,p)
        self.p_rc1 = P(LP, pk_p, [op(LJ, cid), op(LP, rc['kj'], True), op(LS, rc['s3'])])
        self.p_rd1 = P(LP, pk_p, [op(LJ, cid), op(LP, rd['jk'], True), op(LS, rd['s3'])])
        # DF groups over strong canonical pairs
        self._load_df()

    def _load_df(self):
        ref = self.ref
        sel = np.where(self.strong & self.canon)[0]
        naux = np.array([ref.Qma[p].shape[0] for p in sel], np.int64)
        apad = -(-naux // 32) * 32
        LPk = self.LP.group[sel]
        LNk = self.LN.group[sel]
        key = np.stack([apad, LPk, LNk], 1)
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        self.df = []
        chunks = []
        for u in range(len(uk)):
            A = int(uk[u][0])
            Lp, Vp = self.LN.shapes[int(uk[u][2])]
            per = 8 * A * max(Vp * Vp, Lp * Lp, Lp * Vp) * 4
            step = max(1, (1 << 28) // per)
            allm = sel[inv == u]
            for c0 in range(0, len(allm), step):
                chunks.append((u, allm[c0:c0 + step]))
        for u, mem in chunks:
            A = int(uk[u][0])
            Lp, Vp = self.LN.shapes[int(uk[u][2])]
            g = len(mem)
            Qma = np.zeros((g, A, Lp, Vp))
            Qab = np.zeros((g, A, Vp, Vp))
            Qk = np.zeros((g, 2, A, Lp))
            Qa = np.zeros((g, 2, A, Vp))
            for s, p in enumerate(mem):
                q = self.ji[p]
                na, nl, nv = ref.Qma[p].shape
                Qma[s, :na, :nl, :nv] = ref.Qma[p]
                Qab[s, :na, :nv, :nv] = ref.Qab[p]
                Qk[s, 0, :na, :nl] = ref.iQk[p]
                Qk[s, 1, :na, :nl] = ref.iQk[q]
                Qa[s, 0, :na, :nv] = ref.iQa[p]
                Qa[s, 1, :na, :nv] = ref.iQa[q]
            q = self.ji[mem]
            pos = ref.lmo_pos
            gL = self.LL.group[mem]
            assert np.all(gL == gL[0])
            dmask = self.pi[mem] == self.pj[mem]
            ev = np.zeros((g, Vp))
            lm_idx = np.zeros((g, Lp), np.int64)
            lm_msk = np.zeros((g, Lp))
            for s, p in enumerate(mem):
                ev[s, :self.nv[p]] = ref.e_pno[p]
                lm_idx[s, :self.nl[p]] = ref.lmos[p]
                lm_msk[s, :self.nl[p]] = 1.
            self.df.append(dict(
                p=mem, q=q, gP=int(uk[u][1]), gN=int(uk[u][2]), gL=int(gL[0]),
                sP=cp.asarray(self.LP.slot[mem]), sPq=cp.asarray(self.LP.slot[q]),
                sN=cp.asarray(self.LN.slot[mem]), sNq=cp.asarray(self.LN.slot[q]),
                sL=cp.asarray(self.LL.slot[mem]), sLq=cp.asarray(self.LL.slot[q]),
                ri=cp.asarray([pos[p][self.pi[p]] for p in mem]),
                rj=cp.asarray([pos[p][self.pj[p]] for p in mem]),
                dsel=cp.asarray(np.where(dmask)[0]), di=self.pi[mem][dmask],
                ev=cp.asarray(ev), lm_idx=cp.asarray(lm_idx), lm_msk=cp.asarray(lm_msk),
                Qma=cp.asarray(Qma), Qab=cp.asarray(Qab), Qk=cp.asarray(Qk), Qa=cp.asarray(Qa)))

    # ------------------------------------------------------------------
    # elementwise helpers
    def swapT(self, R):
        """R'[p] = R[ji(p)]^T for a square pair layout."""
        ps = self.pslot[id(R.layout)]
        return Ragged(R.layout, [d[s].transpose(0, 2, 1) for d, s in zip(R.data, ps)])

    def swap(self, R):
        ps = self.pslot[id(R.layout)]
        return Ragged(R.layout, [d[s] for d, s in zip(R.data, ps)])

    def masked(self, R, sel):
        """zero all items not in boolean mask sel (over pairs)."""
        out = R.copy()
        for g, m in enumerate(R.layout.members):
            z = cp.asarray(~sel[m])
            out.data[g][z] = 0.
        return out

    # ------------------------------------------------------------------
    # iterations
    # ------------------------------------------------------------------
    def init_state(self):
        ref = self.ref
        f = ref.orb.F_lmo
        self.F = cp.asarray(f)
        self.T = Ragged.from_list(self.LP, ref.T)
        self.T1 = Ragged(self.LT1)
        fd = np.diag(f)
        D2 = []
        for g, m in enumerate(self.LP.members):
            pv = self.LP.shapes[g][0]
            e = np.zeros((len(m), pv))
            for s, p in enumerate(m):
                e[s, :self.nv[p]] = ref.e_pno[p]
            d = (e[:, :, None] + e[:, None, :]
                 - (fd[self.pi[m]] + fd[self.pj[m]])[:, None, None])
            d = np.where(self.LP.mask()[g].get() > 0, d, 1.)
            D2.append(cp.asarray(d))
        self.D2 = Ragged(self.LP, D2)
        D1 = []
        for g, m in enumerate(self.LT1.members):
            pv = self.LT1.shapes[g][0]
            e = np.ones((len(m), pv, 1))
            for s, i in enumerate(m):
                ii = self.dii[i]
                e[s, :self.nv[ii], 0] = ref.e_pno[ii] - fd[i]
            D1.append(cp.asarray(e))
        self.D1 = Ragged(self.LT1, D1)
        self.offdiag = self.pi != self.pj

    def _beta_w(self, beta):
        p, bflat = self.rb_beta_idx
        LL = self.LL
        if not hasattr(self, '_bw_idx'):
            self._bw_idx = []
            for g in range(LL.ngroups):
                sel = np.where(LL.group[p] == g)[0]
                if len(sel) == 0:
                    continue
                Lp = LL.shapes[g][0]
                nlp = self.nl[p[sel]]
                a, b = bflat[sel] // nlp, bflat[sel] % nlp
                self._bw_idx.append((g, cp.asarray(sel), cp.asarray(LL.slot[p[sel]]),
                                     cp.asarray(a * Lp + b)))
        w = cp.zeros(len(p))
        for g, sel, sl, fl in self._bw_idx:
            w[sel] = beta.data[g].reshape(beta.data[g].shape[0], -1)[sl, fl]
        return w

    @staticmethod
    def _scal(r):
        """values of a (1x1)-block Ragged in item order."""
        lay = r.layout
        out = cp.zeros(lay.n)
        for g, m in enumerate(lay.members):
            if len(m):
                out[cp.asarray(m)] = r.data[g][:, 0, 0]
        return out

    def _v3(self, Tt):
        """v[x]_a = sum_cd K_ivvv[x]_{c,d,a} Tt[x]_{cd}, on LV."""
        V3 = Ragged(self.LV)
        LP, LV = self.LP, self.LV
        for g, m in enumerate(LP.members):
            if not len(m):
                continue
            v = cp.einsum('gcda,gcd->ga', self.K3[g], Tt.data[g])
            gv = LV.group[m]
            assert np.all(gv == gv[0])
            V3.data[gv[0]][cp.asarray(LV.slot[m]), :, 0] = v
        return V3

    def residuals(self):
        T, T1, S = self.T, self.T1, self.S
        LP, LN, LV, LL, LT1, L11 = self.LP, self.LN, self.LV, self.LL, self.LT1, self.L11
        n = self.n
        Tt = T * 2. - T.T()
        Tn = self.p_tn.run(Ragged(LN), [S, T1])
        TiP = self.p_ti.run(Ragged(LV), [S, T1])
        TjP = self.p_tj.run(Ragged(LV), [S, T1])
        # ---- T1-dressed Fock matrices and the DF terms -------------------------
        sc = 2 * Tn.vdot_items(self.Jijmb) - Tn.vdot_items(self.swap(self.Kmibj))
        pij = cp.asarray(self.pi * n + self.pj)
        Fij_bar = self.F.copy()
        cupyx.scatter_add(Fij_bar.ravel(), pij, sc)
        Fkc_bar = Ragged(LN)
        Fab = Ragged(LP)
        beta = Ragged(LL)
        Rsym = Ragged(LP)
        Fai = Ragged(LT1)
        diag_items = []
        for d in self.df:
            gP, gN, gL = d['gP'], d['gN'], d['gL']
            Tnp = Tn.data[gN][d['sN']]
            Tnq = Tn.data[gN][d['sNq']]
            Tp = T.data[gP][d['sP']]
            Tq = T.data[gP][d['sPq']]
            Qma, Qab, Qk, Qa = d['Qma'], d['Qab'], d['Qk'], d['Qa']
            ar = cp.arange(len(d['p']))
            ti = Tnp[ar, d['ri']]
            tj = Tnq[ar, d['rj']]
            qk_p = Qk[:, 0] + cp.einsum('gAka,ga->gAk', Qma, ti)
            qa_p = Qa[:, 0] - cp.matmul(qk_p, Tnp) + cp.einsum('gAab,gb->gAa', Qab, ti)
            qk_q = Qk[:, 1] + cp.einsum('gAka,ga->gAk', Qma, tj)
            qa_q = Qa[:, 1] - cp.matmul(qk_q, Tnq) + cp.einsum('gAab,gb->gAa', Qab, tj)
            gam = cp.einsum('gAme,gme->gA', Qma, Tnp)
            X = cp.matmul(Qma, Tnp.transpose(0, 2, 1)[:, None])          # gAkm
            fkc = 2 * cp.einsum('gA,gAkc->gkc', gam, Qma) - cp.matmul(X, Qma).sum(axis=1)
            Y = cp.matmul(Qab, Tnp.transpose(0, 2, 1)[:, None])          # gAam
            fab = 2 * cp.einsum('gA,gAab->gab', gam, Qab) - cp.matmul(Y, Qma).sum(axis=1)
            ev = d['ev']
            fab = fab + ev[:, :, None] * cp.eye(ev.shape[1])[None]
            Fkc_bar.data[gN][d['sN']] = fkc
            Fab.data[gP][d['sP']] = fab - cp.matmul(Tnp.transpose(0, 2, 1), fkc)
            QT = cp.matmul(Qma, Tp[:, None])
            beta.data[gL][d['sL']] = (cp.matmul(qk_p.transpose(0, 2, 1), qk_q)
                                      + cp.matmul(QT, Qma.transpose(0, 1, 3, 2)).sum(axis=1))
            QT = cp.matmul(Qma, Tq[:, None])
            beta.data[gL][d['sLq']] = (cp.matmul(qk_q.transpose(0, 2, 1), qk_p)
                                       + cp.matmul(QT, Qma.transpose(0, 1, 3, 2)).sum(axis=1))
            Kij = cp.matmul(qa_p.transpose(0, 2, 1), qa_q)
            Qt = Qab - cp.matmul(Tnp.transpose(0, 2, 1)[:, None], Qma)
            A = cp.matmul(cp.matmul(Qt, Tp[:, None]), Qt.transpose(0, 1, 3, 2)).sum(axis=1)
            Rsym.data[gP][d['sP']] += Kij + A
            if len(d['di']):
                ds = d['dsel']
                fai = 2 * cp.einsum('gA,gAa->ga', gam[ds], Qa[ds, 0])
                fai -= cp.einsum('gAam,gAm->ga', Y[ds], Qk[ds, 0])
                diag_items.append((d, ds, fai, fab[ds], fkc[ds], Tnp[ds]))
        v = self.p_fkj.run(Ragged(L11), [Fkc_bar, T1])
        Fkj = Fij_bar.copy()
        cupyx.scatter_add(Fkj.ravel(), pij, self._scal(v))
        Fkc = self.p_fkc.run(Ragged(LV), [S, self.Lij, S, T1])
        for d, ds, fai, fab, fkc, Tnd in diag_items:
            di = d['di']
            g1, sl1 = LT1.group[di], LT1.slot[di]
            t1 = cp.stack([T1.data[g][s][:, 0] for g, s in zip(g1, sl1)])
            col = Fij_bar[d['lm_idx'][ds], cp.asarray(di)[:, None]] * d['lm_msk'][ds]
            fai = (fai + cp.einsum('gab,gb->ga', fab, t1)
                   - cp.einsum('gka,gk->ga', Tnd, cp.einsum('gkb,gb->gk', fkc, t1))
                   - cp.einsum('gka,gk->ga', Tnd, col))
            for k, (g, s) in enumerate(zip(g1, sl1)):
                Fai.data[g][s, :, 0] = fai[k]
        # ---- gamma, delta, F'' ------------------------------------------------
        K3 = self.K3
        ps = self.pslot[id(LP)]
        G = Ragged(LP)
        self.p_g1.run(G, [Tn, self.Jijmb], alpha=-1.)
        for g in range(LP.ngroups):
            if G.data[g].shape[0]:
                G.data[g] += cp.einsum('gb,gbac->gac', TjP.data[g][:, :, 0], K3[g])
        self.p_g3.run(G, [S, T1, T1, S, self.K, S], alpha=-1.)
        self.p_g4.run(G, [S, T, S, self.K, S], alpha=-.5)
        Dl = Ragged(LP)
        self.p_d1.run(Dl, [Tn, self.Kmibj], alpha=-2.)
        self.p_d1.run(Dl, [Tn, self.Jijmb], alpha=1.)
        for g in range(LP.ngroups):
            if Dl.data[g].shape[0]:
                Kq = K3[g][ps[g]]
                tiv = TiP.data[g][:, :, 0]
                Dl.data[g] += 2 * cp.einsum('gcab,gb->gac', Kq, tiv)
                Dl.data[g] -= cp.einsum('gb,gbac->gac', tiv, Kq)
        self.p_d3.run(Dl, [S, T1, T1, S, self.Lij, S], alpha=-1.)
        self.p_d4.run(Dl, [S, Tt, S, self.Lij, S], alpha=.5)
        Fdd = Fkj.copy()
        cupyx.scatter_add(Fdd.ravel(), self.fdd_flat, self.p_fdd.run([self.K], [S, Tt, S]))
        # ---- R1 ---------------------------------------------------------------
        R1 = Fai
        Fai = Fai.copy()
        self.p_r1a.run(R1, [S, self._v3(Tt)])
        w = self.p_r1b_w.run([self.K], [S, Tt, S])
        self.p_r1b.run(R1, [S, T1], w=w, alpha=-1.)
        self.p_r1c.run(R1, [S, Tt, Fkc])
        B = Ragged(LN)
        self.p_B1.run(B, [self.Kmibj, Tt])
        self.p_B2.run(B, [Tn, self.K, Tt])
        self.p_r1d.run(R1, [S, B], alpha=-1.)
        # ---- R2 ---------------------------------------------------------------
        self.p_rb.run(Rsym, [S, T, S], w=self._beta_w(beta))
        Fbc = Fab
        self.p_rf.run(Fbc, [S, Tt, self.K, S], alpha=-1.)
        for g in range(LP.ngroups):
            if Rsym.data[g].shape[0]:
                Tg, Fg = T.data[g], Fbc.data[g]
                Rsym.data[g] += cp.matmul(Tg, Fg.transpose(0, 2, 1)) + cp.matmul(Fg, Tg)
        R = Rsym + self.masked(self.swapT(Rsym), self.offdiag)
        Cm = Ragged(LP)
        self.p_rc1.run(Cm, [self.Jikac, T, S], alpha=-1.)
        self.p_rc2.run(Cm, [S, G, S, T, S], alpha=-1.)
        Dm = Ragged(LP)
        self.p_rd1.run(Dm, [self.Kiakc, Tt, S], alpha=2.)
        self.p_rd1.run(Dm, [self.Jikac, Tt, S], alpha=-1.)
        self.p_rd3.run(Dm, [S, Dl, S, Tt, S])
        Rn = Cm * .5 + Cm.T() + Dm * .5
        self.p_rg.run(Rn, [S, T, S], w=Fdd.ravel()[self.rg_f], alpha=-1.)
        R = R + Rn + self.swapT(Rn)
        R = self.masked(R, self.strong)
        self.last = dict(Fkj=Fkj, Fdd=Fdd, G=G, Dl=Dl, beta=beta, Fai=Fai, Tn=Tn)
        return R1, R

    def energy(self):
        TiP = self.p_ti.run(Ragged(self.LV), [self.S, self.T1])
        TjP = self.p_tj.run(Ragged(self.LV), [self.S, self.T1])
        return self.T.vdot_items(self.Lij) + self._scal(
            self.p_e1.run(Ragged(self.L11), [TiP, self.Lij, TjP]))

    def kernel(self, verbose=True):
        opt = self.ref.opt
        self.init_state()
        diis = _DeviceDIIS(opt['DIIS_MAX_VECS'])
        e_prev = 0.
        weak = cp.asarray(~self.strong)
        for it in range(1, opt['MAXITER'] + 1):
            t0 = time.perf_counter()
            R1, R2 = self.residuals()
            r1 = float(R1.rms_per_item().max())
            r2 = float(R2.rms_per_item().max())
            self.T1 = self.T1 - R1 / self.D1
            self.T = self.T - R2 / self.D2
            n1 = self.T1.flat().size
            x = diis.update(cp.concatenate([self.T1.flat(), self.T.flat()]),
                            cp.concatenate([R1.flat(), R2.flat()]))
            self.T1.set_flat(x[:n1])
            self.T.set_flat(x[n1:])
            e = self.energy()
            e_cur = float(e.sum())
            e_weak = float(e[weak].sum())
            if verbose:
                print(f'  GPU LCCSD iter {it:3d} {e_cur:.12f} {e_cur - e_prev:.3e} '
                      f'{r1:.3e} {r2:.3e}  {time.perf_counter() - t0:.2f}s', flush=True)
            if abs(e_cur - e_prev) < opt['E_CONVERGENCE'] and max(r1, r2) < opt['R_CONVERGENCE']:
                break
            e_prev = e_cur
        else:
            raise RuntimeError('LCCSD did not converge')
        return e_cur - e_weak, e_weak


class _DeviceDIIS:
    def __init__(self, nvec=6):
        self.nvec = nvec
        self.x, self.e = [], []

    def update(self, x, e):
        self.x.append(x.copy())
        self.e.append(e.copy())
        if len(self.x) > self.nvec:
            k = int(np.argmax([float(cp.dot(v, v)) for v in self.e]))
            del self.x[k], self.e[k]
        n = len(self.x)
        if n < 2:
            return x
        B = -np.ones((n + 1, n + 1))
        B[n, n] = 0
        E = cp.stack(self.e)
        B[:n, :n] = (E @ E.T).get()
        rhs = np.zeros(n + 1)
        rhs[n] = -1
        try:
            c = np.linalg.solve(B, rhs)[:n]
        except np.linalg.LinAlgError:
            return x
        return cp.asarray(c) @ cp.stack(self.x)
