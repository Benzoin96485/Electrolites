"""Closed-shell DLPNO-(T0) and iterative (T) on the device.

Equations as ``ref_rhf`` (Jiang et al. 2024, Eqs. 53, 109-112).  The triplet
natural orbitals of a whole chunk of triplets are built at once with batched
eigensolvers (``linalg.eigh_batched``); the W/V intermediates of one triplet
are a handful of large GEMMs, so triplets are then processed one at a time,
which already keeps the device busy for any triplet worth computing.
"""
import time

import numpy as np
import cupy as cp
import cupyx.scipy.linalg as cpla

from electrolites.dlpno import setup as su
from electrolites.dlpno.ragged import padded
from electrolites.dlpno.linalg import eigh_batched


def _cho_solve(L, x):
    s = x.shape
    y = cpla.solve_triangular(L, x.reshape(L.shape[0], -1), lower=True)
    y = cpla.solve_triangular(L, y, lower=True, trans='T')
    return y.reshape(s)


def _half(L, x):
    s = x.shape
    return cpla.solve_triangular(L, x.reshape(L.shape[0], -1), lower=True).reshape(s)


def _permute(X, a1, a2, a3):
    if a1 <= a2 <= a3:
        return X
    if a1 <= a3 <= a2:
        return X.transpose(0, 2, 1)
    if a2 <= a1 <= a3:
        return X.transpose(1, 0, 2)
    if a2 <= a3 <= a1:
        return X.transpose(2, 0, 1)
    if a3 <= a1 <= a2:
        return X.transpose(1, 2, 0)
    return X.transpose(2, 1, 0)


def _e_ijk(i, j, k, V, T3):
    pref = 1.
    if i == j == k:
        pref /= 6.
    elif i == j or j == k or i == k:
        pref /= 2.
    e = 8 * cp.vdot(V, T3)
    e -= 4 * cp.vdot(_permute(V, k, j, i), T3)
    e -= 4 * cp.vdot(_permute(V, i, k, j), T3)
    e -= 4 * cp.vdot(_permute(V, j, i, k), T3)
    e += 2 * cp.vdot(_permute(V, j, k, i), T3)
    e += 2 * cp.vdot(_permute(V, k, i, j), T3)
    return pref * e


class TriplesGPU:
    """Mixin for rhf.DLPNO: replaces tno_transform / lccsd_t0 /
    lccsd_t_iterations of the reference."""

    # ---- pair quantities on the device ----------------------------------
    def _pair_dev(self):
        if getattr(self, '_pd_src', None) is self.T:
            return
        self._pd_src = self.T
        S = self.S_pao_d
        self._XS = {}
        self._Dp = {}
        self._Td = {}
        self._Xd = {}

    def _XSp(self, p):
        """X_p^T S_pao[d_p, :] (npno, nbf)."""
        v = self._XS.get(p)
        if v is None:
            X = cp.asarray(self.X[p])
            v = X.T @ self.S_pao_d[cp.asarray(self.pao[p])]
            self._XS[p] = v
        return v

    def _Dpair(self, p):
        v = self._Dp.get(p)
        if v is None:
            i, j = self.pl.pairs[p]
            T = cp.asarray(self.T[p])
            Tt = 2 * T - T.T
            v = Tt @ T.T + Tt.T @ T
            if i == j:
                v *= .5
            self._Dp[p] = v
        return v

    def _Tpair(self, p):
        v = self._Td.get(p)
        if v is None:
            v = self._Td[p] = cp.asarray(self.T[p])
        return v

    # ---- TNOs ------------------------------------------------------------
    def tno_transform(self, t_cut):
        opt, pl = self.opt, self.pl
        self._pair_dev()
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
        for Pv in np.unique(P):
            sel = np.where(P == Pv)[0]
            step = max(1, int((1 << 29) // (8 * Pv * Pv * 8)))
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
                dn = cp.where(m > 0, 1. / cp.sqrt(cp.where(m > 0, cp.diagonal(St, axis1=1, axis2=2), 1.)), 0.)
                Sn = St * dn[:, :, None] * dn[:, None, :]
                w, U = eigh_batched(Sn)
                keep = w > s_cut
                X = dn[:, :, None] * U * cp.where(keep, 1. / cp.sqrt(cp.where(keep, w, 1.)), 0.)[:, None, :]
                Dt = cp.zeros((b, Pv, Pv))
                for role in range(3):
                    pp = []
                    for t in ch:
                        i, j, k = self.trip[t]
                        a, c = ((i, j), (j, k), (i, k))[role]
                        pp.append(pl.ij[a, c])
                    nvp = np.array([self.X[p].shape[1] for p in pp])
                    V = int(nvp.max())
                    XSg = cp.zeros((b, V, Pv))
                    Dg = cp.zeros((b, V, V))
                    for s, p in enumerate(pp):
                        xs = self._XSp(p)
                        XSg[s, :nvp[s]] = xs[:, gi[s]] * m[s][None]
                        Dg[s, :nvp[s], :nvp[s]] = self._Dpair(p)
                    sp = cp.matmul(XSg, X)                       # (b, V, Pv)
                    Dt += cp.matmul(sp.transpose(0, 2, 1), cp.matmul(Dg, sp))
                Dt /= 3.
                Dt -= cp.eye(Pv)[None] * (~keep)[:, :, None]
                occ, Vt = eigh_batched(Dt)
                occ, Vt = occ[:, ::-1], Vt[:, :, ::-1]
                nkeep = keep.sum(axis=1).get()
                occ_h = occ.get()
                nfin = np.zeros(b, np.int64)
                for s, t in enumerate(ch):
                    cnt = int(np.sum(np.abs(occ_h[s, :nkeep[s]]) >= self.tno_scale[t] * t_cut))
                    nfin[s] = max(1, max(cnt, min(min_tnos, nkeep[s])))
                Nt = int(nfin.max())
                cm = (cp.arange(Nt)[None, :] < cp.asarray(nfin)[:, None]).astype(cp.float64)
                Uq = cp.matmul(X, Vt[:, :, :Nt]) * cm[:, None, :]
                Fq = cp.matmul(Uq.transpose(0, 2, 1), cp.matmul(Ft, Uq))
                Fq += 1e6 * cp.eye(Nt)[None] * (1 - cm)[:, :, None]
                et, Wt = eigh_batched(Fq)
                Xt = cp.matmul(Uq, Wt)
                for s, t in enumerate(ch):
                    self.X_tno[t] = Xt[s, :npao[t], :nfin[s]].copy()
                    self.e_tno[t] = et[s, :nfin[s]].copy()

    # ---- (T0) ----------------------------------------------------------
    def _trip_riatoms(self, prescreen):
        opt = self.opt
        mkn = opt['T_CUT_MKN_TRIPLES_PRE'] if prescreen else opt['T_CUT_MKN_TRIPLES']
        ria = [np.where(np.abs(self.mkn[:, i]) > mkn)[0] for i in range(self.n)]
        return [np.unique(np.concatenate([ria[i], ria[j], ria[k]])) for (i, j, k) in self.trip]

    def triples_sparsity(self, prescreen):
        super().triples_sparsity(prescreen)
        self.t_riatoms = self._trip_riatoms(prescreen)

    def _S_tp(self, t, p, Xt):
        """S(a_triplet, a_pair) = X_t^T S_pao[d_t, d_p] X_p."""
        return (self._XSp(p)[:, cp.asarray(self.t_paos[t])] @ Xt).T

    def lccsd_t0(self, save=False):
        pl, f = self.pl, self.orb.F_lmo
        dfi = self.dfi
        self._pair_dev()
        ntrip = len(self.trip)
        self.e_ijk = np.zeros(ntrip)
        if save:
            self.W3, self.V3, self.T3 = [None] * ntrip, [None] * ntrip, [None] * ntrip
        T1d = [cp.asarray(t) for t in self.T1]
        E = 0.
        for t, (i, j, k) in enumerate(self.trip):
            atoms = self.t_riatoms[t]
            aux = dfi.aux_index(atoms)
            paos, lm = self.t_paos[t], self.t_lmos[t]
            X = self.X_tno[t]
            nv = X.shape[1]
            na = len(aux)
            L = dfi.metric_chol(atoms)
            qv = {a: dfi.q_lmo_pao(aux, a, paos) @ X for a in (i, j, k)}
            qo = {a: dfi.q_lmo_lmo(aux, a, lm) for a in (i, j, k)}
            q_vv = dfi.q_vv(atoms, paos, X)
            qvc = {a: _cho_solve(L, v) for a, v in qv.items()}
            qv = {a: _half(L, v) for a, v in qv.items()}
            qo = {a: _half(L, v) for a, v in qo.items()}
            Kvvv = {a: (qvc[a].T @ q_vv.reshape(na, -1)).reshape(nv, nv, nv) for a in qv}
            Kov = {(a, b): qo[a].T @ qv[b] for a in qv for b in qv}
            perms = [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]
            # projected doubles for every pair needed by this triplet
            Tproj = {}

            def tp(p):
                v = Tproj.get(p)
                if v is None:
                    s = self._S_tp(t, p, X)
                    v = Tproj[p] = s @ self._Tpair(p) @ s.T
                return v

            Wp = []
            for (I, J, K) in perms:
                Tkj = tp(pl.ij[K, J])
                w = (Kvvv[I].reshape(nv * nv, nv) @ Tkj.T).reshape(nv, nv, nv)
                Til = cp.stack([tp(pl.ij[I, l]) for l in lm])          # (nl, a, b)
                w -= (Til.reshape(len(lm), nv * nv).T @ Kov[(J, K)]).reshape(nv, nv, nv)
                Wp.append(w)
            W = (Wp[0] + Wp[1].transpose(0, 2, 1) + Wp[2].transpose(1, 0, 2)
                 + Wp[3].transpose(2, 0, 1) + Wp[4].transpose(1, 2, 0)
                 + Wp[5].transpose(2, 1, 0))
            Ti = self._S_tp(t, pl.ij[i, i], X) @ T1d[i]
            Tj = self._S_tp(t, pl.ij[j, j], X) @ T1d[j]
            Tk = self._S_tp(t, pl.ij[k, k], X) @ T1d[k]
            V = (W + Ti[:, None, None] * (qv[j].T @ qv[k])[None]
                 + Tj[None, :, None] * (qv[i].T @ qv[k])[:, None, :]
                 + Tk[None, None, :] * (qv[i].T @ qv[j])[:, :, None])
            e = self.e_tno[t]
            Dn = (e[:, None, None] + e[None, :, None] + e[None, None, :]
                  - (f[i, i] + f[j, j] + f[k, k]))
            T3 = -W / Dn
            eijk = float(_e_ijk(i, j, k, V, T3))
            self.e_ijk[t] = eijk
            E += eijk
            if save:
                self.W3[t], self.V3[t], self.T3[t] = W, V, T3
        return E

    # ---- iterative (T) --------------------------------------------------
    def lccsd_t_iterations(self):
        opt, f, n = self.opt, self.orb.F_lmo, self.n
        ntrip = len(self.trip)
        S_d = self.S_pao_d
        e_old = np.zeros(ntrip)
        e_prev = 0.
        # couplings: for each triplet the (u, f, perm) list
        coup = [[] for _ in range(ntrip)]
        for t, (i, j, k) in enumerate(self.trip):
            for l in range(n):
                for key, fl, ok in (((i, j, l), f[l, k], l != k),
                                    ((i, l, k), f[l, j], l != j),
                                    ((l, j, k), f[l, i], l != i)):
                    if ok and key in self.trip_idx and abs(fl) >= opt['F_CUT_T']:
                        coup[t].append((self.trip_idx[key], fl, key))
        Sxs = {}

        def s_tt(t, u):
            v = Sxs.get((t, u))
            if v is None:
                pt, pu = cp.asarray(self.t_paos[t]), cp.asarray(self.t_paos[u])
                v = self.X_tno[t].T @ S_d[pt[:, None], pu[None, :]] @ self.X_tno[u]
                Sxs[(t, u)] = v
            return v

        Dn = []
        for t, (i, j, k) in enumerate(self.trip):
            e = self.e_tno[t]
            Dn.append(e[:, None, None] + e[None, :, None] + e[None, None, :]
                      - (f[i, i] + f[j, j] + f[k, k]))
        for it in range(1, opt['MAXITER'] + 1):
            rms = np.zeros(ntrip)
            for t, (i, j, k) in enumerate(self.trip):
                if abs(self.e_ijk[t] - e_old[t]) < abs(e_old[t] * opt['T_CUT_ITER']):
                    continue
                R = self.W3[t] + self.T3[t] * Dn[t]
                for u, fl, perm in coup[t]:
                    s = s_tt(t, u)
                    A = _permute(self.T3[u], *perm)
                    A = cp.einsum('aA,bB,cC,ABC->abc', s, s, s, A, optimize=True)
                    R -= fl * A
                self.T3[t] = self.T3[t] - R / Dn[t]
                rms[t] = float(cp.sqrt(cp.mean(R * R)))
            e_old = self.e_ijk.copy()
            E = 0.
            for t, (i, j, k) in enumerate(self.trip):
                self.e_ijk[t] = float(_e_ijk(i, j, k, self.V3[t], self.T3[t]))
                E += self.e_ijk[t]
            self.log(f'  GPU LCCSD(T) iter {it:3d} {E:.12f} {E - e_prev:.3e} {rms.max():.3e}')
            if rms.max() < opt['R_CONVERGENCE'] and abs(E - e_prev) < opt['E_CONVERGENCE']:
                break
            e_prev = E
        else:
            raise RuntimeError('(T) iterations did not converge')
        return E
