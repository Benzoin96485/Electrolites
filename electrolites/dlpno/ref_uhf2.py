"""Open-shell DLPNO-CCSD with Psi4's (Jiang et al. 2024) projection placement.

Same spin-orbital T1-dressed CCSD as ``ref_uhf``; four terms are evaluated
where Jiang et al. evaluate their closed-shell counterparts, so that every
quantity a GPU implementation needs is either a fixed-index integral of a
pair (exchange, 1- and 3-external) or a DF tensor of a strong pair:

1. F~_KJ = Fbar_KJ (pair kj) + Fbar_KC (diagonal pair jj) t_J^C   (Eq. 94)
2. the doubly-virtual 2-external pieces (K c|l d) of the T1 dressing of the
   ring intermediate are pair (k, l)'s exchange integrals projected into the
   (j, k) space (as the third terms of Eqs. 83, 84);
3. the -t_l^a (l c|k d) t_ik^cd part of the singles A term likewise uses pair
   (k, l)'s exchange integrals (Eq. 88);
4. F~_KC of the singles C term is sum_l S(ik,kl) <kl||cd> S(kl,ll) t_l
   (Eq. 95).

In the canonical limit all of these are exact, so this class reproduces
canonical UCCSD exactly too.
"""
import numpy as np

from pyscf import lib

from electrolites.dlpno.ref_uhf import RefUDLPNO


class RefUDLPNO2(RefUDLPNO):

    def _Kx(self, K, L):
        """plain exchange (K C|L D) in pair (k, l)'s dense space."""
        q = self.pl.ij[self.so[K][0], self.so[L][0]]
        d = self.cc[q]
        Bov, pos = d['Bov'], d['pos']
        return Bov[:, pos[K]].T @ Bov[:, pos[L]]

    # 1. occupied Fock: second part from the diagonal pair jj ------------
    def _Fp_oo(self, dr, Gu):
        pl = self.pl
        out = {}
        for p, (k, j) in enumerate(pl.pairs):
            d = self.cc[p]
            FD = dr[p]['FD']
            jj = pl.ij[j, j]
            dj, drj = self.cc[jj], dr[jj]
            oj = dj['o']
            for K in d['occ']:
                if self.so[K][0] != k:
                    continue
                for J in d['occ']:
                    if self.so[J][0] != j or self.so[K][1] != self.so[J][1]:
                        continue
                    val = FD[d['pos'][K], d['pos'][J]]
                    val += drj['FD'][dj['pos'][K], oj:] @ self.T1[J]
                    for l in self.lmos[p]:
                        for L in (self.so_index.get((l, 0)), self.so_index.get((l, 1))):
                            if L is None or L == K or L == J:
                                continue
                            kl, jl = pl.ij[k, l], pl.ij[j, l]
                            sm = self.S(kl, jl)
                            val += .5 * np.sum(Gu[(K, L)] * (sm @ self.T2[(J, L)] @ sm.T))
                    out[(K, J)] = val
        return out

    # 2. ring intermediate ------------------------------------------------
    def _ring_Z(self, dr, Gu, J, K):
        key = (J, K)
        if key in self._Z:
            return self._Z[key]
        pl = self.pl
        y, k = self.so[J][0], self.so[K][0]
        q = pl.ij[y, k]
        d, dq = self.cc[q], dr[q]
        o = d['o']
        B = d['B']
        tau = dq['tau'][o:, :o]                    # tau[B, l] = T_n[l, B]
        pk, pj = d['pos'][K], d['pos'][J]
        tJ = dq['Tn'][pj]                          # t_J projected into q
        BKv = B[:, pk, o:]                         # (Q| K C)
        # -(KC|lJ) tau_Bl
        KClJ = np.einsum('Qc,Ql->lc', BKv, B[:, :o, pj])
        Z = -tau @ KClJ
        # + t_J^d (KC|Bd)  - t_J^d (Kd|BC)
        Bvv = B[:, o:, o:]
        Z += np.einsum('Qc,Qbd,d->bc', BKv, Bvv, tJ)
        Z -= np.einsum('Qd,d,Qbc->bc', BKv, tJ, Bvv)
        # + tau_Bl (KJ|lC)
        KJlC = np.einsum('Q,Qlc->lc', B[:, pk, pj], B[:, :o, o:])
        Z += tau @ KJlC
        # - tau_Bl t_J^d (KC|ld) + tau_Bl t_J^d (Kd|lC), (K.|l.) of pair (k, l)
        for r, L in enumerate(d['occ']):
            l = self.so[L][0]
            kl = pl.ij[k, l]
            if kl < 0 or L == K:
                continue
            s = self.S(q, kl)
            Kx = s @ self._Kx(K, L) @ s.T            # (K c|L d), c, d in q
            Z -= np.outer(tau[:, r], Kx @ tJ)
            Z += np.outer(tau[:, r], Kx.T @ tJ)
        for l in self.lmos[q]:
            for L in (self.so_index.get((l, 0)), self.so_index.get((l, 1))):
                if L is None or L == J or L == K:
                    continue
                jl, kl = pl.ij[y, l], pl.ij[k, l]
                Z += .5 * self.S(q, jl) @ self.T2[(J, L)] @ self.S(jl, kl) @ Gu[(K, L)].T @ self.S(kl, q)
        self._Z[key] = Z
        return Z

    # 3./4. singles -------------------------------------------------------
    def _R1(self, dr):
        pl = self.pl
        R1 = []
        for I, (x, s) in enumerate(self.so):
            dd = pl.ij[x, x]
            d, dp = self.cc[dd], dr[dd]
            o = d['o']
            r = dp['Ft'][o:, d['pos'][I]].copy()
            for K in range(len(self.so)):
                if K == I:
                    continue
                k = self.so[K][0]
                ik = pl.ij[x, k]
                if ik < 0:
                    continue
                dk = self.cc[ik]
                ok = dk['o']
                pk = dk['pos'][K]
                t = self.T2[(I, K)]
                sm = self.S(dd, ik)
                # C: F~_KC t_IK^AC with F~_KC = f_KC + sum_L S(ik,kl) <KL||CD> S(kl,ll) t_L
                fkc = dk['f'][pk, ok:].copy()
                for L in dk['occ']:
                    l = self.so[L][0]
                    kl = pl.ij[k, l]
                    if L == K or kl < 0:
                        continue
                    ll = pl.ij[l, l]
                    fkc += self.S(ik, kl) @ (self._Gkl(K, L) @ (self.S(kl, ll) @ self.T1[L]))
                r += sm @ (t @ fkc)
                # A: sum_CD (K D|A C) t^CD  -  t_l^A sum_CD (l C|K D) t^CD
                B = dk['B']
                r += sm @ lib.einsum('Qd,Qac,cd->a', B[:, pk, ok:], B[:, ok:, ok:], t)
                for L in dk['occ']:
                    l = self.so[L][0]
                    kl = pl.ij[k, l]
                    if kl < 0 or L == K:
                        continue
                    ll = pl.ij[l, l]
                    tl = self.S(dd, ll) @ self.T1[L]
                    s2 = self.S(kl, ik)
                    tp = s2 @ t @ s2.T                      # t_IK in (k,l) space
                    # (l C|K D) = exchange of pair (k, l) as (K D|L C)
                    r -= tl * np.sum(self._Kx(K, L).T * tp)
            # B: -sum_KL (KI|LC)~ t_KL^AC (unchanged: pair (k,l)'s own integrals)
            for kl, (k, l) in enumerate(pl.pairs):
                dk, dpk = self.cc[kl], dr[kl]
                if I not in dk['pos']:
                    continue
                ok = dk['o']
                sm = self.S(dd, kl)
                for K in dk['occ']:
                    if self.so[K][0] != k:
                        continue
                    for L in dk['occ']:
                        if self.so[L][0] != l or K == L:
                            continue
                        u = np.einsum('Q,Qc->c', dpk['Bt'][:, dk['pos'][K], dk['pos'][I]],
                                      dk['B'][:, dk['pos'][L], ok:])
                        r -= sm @ (self.T2[(K, L)] @ u)
            m = np.zeros(self.nv(dd))
            m[self._vslice(dd, s)] = 1.
            R1.append(r * m)
        return R1

    def _Gkl(self, K, L):
        """<KL||CD> in pair (k, l)'s space."""
        q = self.pl.ij[self.so[K][0], self.so[L][0]]
        d = self.cc[q]
        return self._G(q, K, L, d['Bov'], d['pos'])
