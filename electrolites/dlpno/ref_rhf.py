"""Closed-shell DLPNO-CCSD(T) on the host: the reference implementation.

A direct transcription of Psi4 1.11's ``dlpno`` module (Jiang et al., JCP
161, 082502 (2024): T1-dressed DLPNO-CCSD, semicanonical (T0) and iterative
(T)) into NumPy, pair by pair and triplet by triplet.  It is slow and meant
to be: it is the oracle the batched GPU implementation (``rhf.py``) is
tested against term by term, and the thing that is checked against Psi4.

Differences from Psi4, none of which change a converged energy by more than
the thresholds themselves:

* the three-index integrals are exact (no coefficient screening or refit of
  the LMO coefficients, T_CUT_CLMO / T_CUT_CPAO);
* the DOI grid is PySCF's Treutler grid with the same 25 x 50 points per
  atom, unpruned (Psi4 prunes with its "robust" scheme);
* the LMOs come from PySCF's Boys localiser started from the canonical
  orbitals, as Psi4's.
"""
import time

import numpy as np
import scipy.linalg

from pyscf import lib, df
from pyscf.lib import logger

from electrolites.dlpno import setup as su
from electrolites.dlpno.options import make_options



def _phase_log(key, dt):
    """one line per phase: wall time and the device memory in use / pooled."""
    try:
        import cupy as cp
        free, tot = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        mem = f' dev used {(tot - free) / 1e9:.1f} GB (pool {pool.used_bytes() / 1e9:.1f} / {pool.total_bytes() / 1e9:.1f} GB)'
    except Exception:
        mem = ''
    print(f'[phase] {key:16s} {dt:9.2f} s{mem}', flush=True)
    import os
    if os.environ.get('DLPNO_STOP_AFTER') == key:      # profiling aid
        raise SystemExit(0)

class DIIS:
    def __init__(self, nvec=6):
        self.nvec = nvec
        self.x, self.e = [], []

    def update(self, x, e):
        self.x.append(x.copy())
        self.e.append(e.copy())
        if len(self.x) > self.nvec:
            # Psi4 removes the vector with the largest error
            k = int(np.argmax([np.dot(v, v) for v in self.e]))
            del self.x[k], self.e[k]
        n = len(self.x)
        if n < 2:
            return x
        B = -np.ones((n + 1, n + 1))
        B[n, n] = 0
        for a in range(n):
            for b in range(a + 1):
                B[a, b] = B[b, a] = np.dot(self.e[a], self.e[b])
        rhs = np.zeros(n + 1)
        rhs[n] = -1
        try:
            c = np.linalg.solve(B, rhs)[:n]
        except np.linalg.LinAlgError:
            return x
        return sum(ci * xi for ci, xi in zip(c, self.x))


def _rms(a):
    return np.sqrt(np.mean(a * a)) if a.size else 0.


def _sym_inv_sqrt_solver(A):
    """X -> A^{-1/2} X, with A^{1/2} built as Psi4's ``power(0.5, 1e-14)``."""
    e, U = np.linalg.eigh(A)
    keep = e > 1e-14 * e.max()
    ah = (U[:, keep] * np.sqrt(e[keep])) @ U[:, keep].T
    lu = scipy.linalg.lu_factor(ah)
    return lambda X: scipy.linalg.lu_solve(lu, X.reshape(len(A), -1)).reshape(X.shape)


def _permute(X, a1, a2, a3):
    """Psi4 ``triples_permuter``: X (n,n,n) in i<=j<=k order, returned in the
    order of (a1, a2, a3)."""
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


class RefDLPNO:
    """Closed-shell DLPNO-CCSD(T) reference.

    triples: None, 't0' (semicanonical) or 't' / 't1' (iterative)."""

    def __init__(self, mf, pno='normal', triples='t0', auxbasis=None,
                 frozen_core=True, localizer='boys', c_lmo=None, verbose=None,
                 **opts):
        self.mf = mf
        self.mol = mf.mol
        self.opt = make_options(pno, **opts)
        self.triples = None if triples is None else triples.lower()
        self.auxbasis = auxbasis
        self.frozen_core = frozen_core
        self.localizer = localizer
        self.c_lmo = c_lmo
        self.verbose = self.mol.verbose if verbose is None else verbose
        self.timings = {}
        self.e = {}

    # ------------------------------------------------------------------
    def _tick(self, key, t0):
        self.timings[key] = self.timings.get(key, 0.) + time.perf_counter() - t0
        if getattr(self, 'verbose', 0) >= 2:
            _phase_log(key, self.timings[key])
        return time.perf_counter()

    def log(self, *args):
        if self.verbose >= 4:
            print(*args, flush=True)

    # ------------------------------------------------------------------
    def setup(self):
        t = time.perf_counter()
        mf, mol, opt = self.mf, self.mol, self.opt
        S = su.asnumpy(mf.get_ovlp())
        dm = mf.make_rdm1()
        F = su.asnumpy(mf.get_fock(dm=dm))
        C = su.asnumpy(mf.mo_coeff)
        occ = su.asnumpy(mf.mo_occ)
        nocc = int(np.count_nonzero(occ > 0))
        nfc = su.count_frozen_core(mol)
        ncore = nfc if self.frozen_core else 0
        self.ncore_scale = 0 if self.frozen_core else nfc
        self.orb = orb = su.Orbitals(mol, S, F, C, nocc, ncore, self.localizer,
                                     c_lmo=self.c_lmo, mf=mf,
                                     conv_tol=opt['LOCAL_CONVERGENCE'])
        self.n = orb.naocc
        auxbasis = self.auxbasis
        if auxbasis is None:
            auxbasis = df.make_auxbasis(mol, mp2fit=True)
        self.auxmol = auxmol = df.addons.make_auxmol(mol, auxbasis)
        (self.bf_atom, self.ribf_atom, self.atom_bfs,
         self.atom_ribfs) = su.basis_maps(mol, auxmol)
        t = self._tick('orbitals', t)

        self.doi_ij, self.doi_iu, self.doi_uv = su.compute_doi(
            mol, orb.C_lmo, orb.C_pao, opt['DOI_GRID'])
        t = self._tick('doi', t)
        self.dip_e, self.dip_b = su.dipole_pair_energies(
            orb, self.doi_iu, opt['T_CUT_DO_PRE'], opt['S_CUT'],
            self.bf_atom, self.atom_bfs)
        t = self._tick('dipole', t)

        self.J2c = auxmol.intor('int2c2e')
        int3c = df.incore.aux_e2(mol, auxmol, 'int3c2e')
        L, P = orb.C_lmo, orb.C_pao
        tmp = lib.einsum('mnQ,mi->Qin', int3c, L)
        self.qij = lib.einsum('Qin,nj->Qij', tmp, L)
        self.qia = lib.einsum('Qin,nu->Qiu', tmp, P)
        qab = lib.einsum('mnQ,mu,nv->Quv', int3c, P, P)
        self.qab = qab * (self.doi_uv > opt['T_CUT_DO_UV'])[None]
        del int3c, tmp
        self.mkn = su.mulliken_atom_pops(mol, S, L)
        t = self._tick('df_ints', t)

    def domains(self, t_cut_mkn, t_cut_do):
        return su.Domains(self.mkn, self.doi_iu, t_cut_mkn, t_cut_do,
                          self.bf_atom, self.atom_bfs, self.atom_ribfs)

    def set_pairs(self, mask, dom):
        self.pl = pl = su.PairList(self.n, mask)
        self.dom = dom
        self.pao = [su.union(dom.lmo_paos[i], dom.lmo_paos[j]) for (i, j) in pl.pairs]
        self.ribfs = [su.union(dom.lmo_ribfs[i], dom.lmo_ribfs[j]) for (i, j) in pl.pairs]
        self.lmos = [pl.lmos(p) for p in range(len(pl))]
        self.lmo_pos = []
        for p in range(len(pl)):
            d = -np.ones(self.n, dtype=int)
            d[self.lmos[p]] = np.arange(len(self.lmos[p]))
            self.lmo_pos.append(d)
        self._S = {}

    # ------------------------------------------------------------------
    # pair energies, PNOs, LMP2
    # ------------------------------------------------------------------
    def _K_pao(self, p):
        i, j = self.pl.pairs[p]
        aux, paos = self.ribfs[p], self.pao[p]
        iqa = self.qia[np.ix_(aux, [i], paos)][:, 0]
        jqa = self.qia[np.ix_(aux, [j], paos)][:, 0]
        A = self.J2c[np.ix_(aux, aux)]
        iqa = scipy.linalg.solve(A, iqa, assume_a='sym')
        K = iqa.T @ jqa
        orb = self.orb
        X, e = su.orthocanonicalize(orb.S_pao[np.ix_(paos, paos)],
                                    orb.F_pao[np.ix_(paos, paos)], self.opt['S_CUT'])
        K = X.T @ K @ X
        f = orb.F_lmo
        T = K / (f[i, i] + f[j, j] - e[:, None] - e[None, :])
        return X, e, K, T

    def sc_mp2_pair_energies(self):
        pl = self.pl
        e_ij = np.zeros(len(pl))
        for p, (i, j) in enumerate(pl.pairs):
            if i > j:
                continue
            X, e, K, T = self._K_pao(p)
            e_ij[p] = np.sum(K * (2 * T - T.T))
            e_ij[pl.ji[p]] = e_ij[p]
        return e_ij

    def lmp2_pnos(self):
        """Refined prescreening: PNOs for the LMP2 (Psi4
        ``compute_pair_energies<false>``)."""
        opt, pl, orb = self.opt, self.pl, self.orb
        npair = len(pl)
        self.X = [None] * npair
        self.e_pno = [None] * npair
        self.K = [None] * npair
        self.T = [None] * npair
        self.de_pno = np.zeros(npair)
        for p, (i, j) in enumerate(pl.pairs):
            if i > j:
                continue
            X, e, K, T = self._K_pao(p)
            Tt = 2 * T - T.T
            e_ij = np.sum(K * Tt)
            D = Tt @ T.T + Tt.T @ T
            if i == j:
                D *= .5
            occ, U = np.linalg.eigh(D)
            occ, U = occ[::-1], U[:, ::-1]
            scale = 1.
            if i == j:
                scale *= opt['T_CUT_PNO_DIAG_SCALE']
            if i < self.ncore_scale or j < self.ncore_scale:
                scale *= opt['T_CUT_PNO_CORE_SCALE']
            occ_total = occ.sum()
            Kn = U.T @ K @ U
            Ttn = U.T @ Tt @ U
            e_pno = occ_pno = 0.
            a_curr = []
            nfin = 0
            for a in range(len(occ)):
                if (abs(occ[a]) >= scale * opt['T_CUT_PNO_MP2']
                        or occ_pno / occ_total < opt['T_CUT_TRACE_MP2']
                        or abs(e_pno) < opt['T_CUT_ENERGY_MP2'] * abs(e_ij)
                        or a < opt['MIN_PNOS']):
                    e_pno = np.sum(Kn[np.ix_(a_curr, a_curr)] * Ttn[np.ix_(a_curr, a_curr)])
                    occ_pno += occ[a]
                    a_curr.append(a)
                    nfin += 1
            nfin = max(1, nfin)
            U = U[:, :nfin]
            V, ep = su.canonicalize(U, np.diag(e))
            U = U @ V
            Kp = U.T @ K @ U
            Tp = U.T @ T @ U
            self.de_pno[p] = e_ij - np.sum(Kp * (2 * Tp - Tp.T))
            self._store_pair(p, X @ U, ep, Kp, Tp)
        self.de_pno_total = self.de_pno.sum()

    def _store_pair(self, p, X, e, K, T):
        q = self.pl.ji[p]
        self.X[p] = X
        self.e_pno[p] = e
        self.K[p] = K
        self.T[p] = T
        if q != p:
            self.X[q] = X
            self.e_pno[q] = e
            self.K[q] = K.T
            self.T[q] = T.T
            self.de_pno[q] = self.de_pno[p]

    def S(self, p, q):
        """PNO overlap S(a_p, a_q)."""
        key = (p, q)
        s = self._S.get(key)
        if s is None:
            s = self.X[p].T @ self.orb.S_pao[np.ix_(self.pao[p], self.pao[q])] @ self.X[q]
            self._S[key] = s
        return s

    def lmp2_iterations(self):
        opt, pl = self.opt, self.pl
        f = self.orb.F_lmo
        npair = len(pl)
        ec, rc = 0.01 * opt['E_CONVERGENCE'], 0.01 * opt['R_CONVERGENCE']
        diis = DIIS(opt['DIIS_MAX_VECS'])
        e_prev = 0.
        for it in range(1, opt['MAXITER'] + 1):
            R = [None] * npair
            rms = np.zeros(npair)
            for p, (i, j) in enumerate(pl.pairs):
                if i > j:
                    continue
                e = self.e_pno[p]
                r = self.K[p] + (e[:, None] + e[None, :] - f[i, i] - f[j, j]) * self.T[p]
                for k in range(self.n):
                    kj, ik = pl.ij[k, j], pl.ij[i, k]
                    if kj >= 0 and i != k and abs(f[i, k]) > opt['F_CUT']:
                        s = self.S(p, kj)
                        r -= f[i, k] * (s @ self.T[kj] @ s.T)
                    if ik >= 0 and j != k and abs(f[k, j]) > opt['F_CUT']:
                        s = self.S(p, ik)
                        r -= f[k, j] * (s @ self.T[ik] @ s.T)
                R[p] = r
                rms[p] = _rms(r)
                if i < j:
                    R[pl.ji[p]] = r.T
                    rms[pl.ji[p]] = rms[p]
            for p, (i, j) in enumerate(pl.pairs):
                e = self.e_pno[p]
                self.T[p] = self.T[p] - R[p] / (e[:, None] + e[None, :] - f[i, i] - f[j, j])
            flat = diis.update(np.concatenate([t.ravel() for t in self.T]),
                               np.concatenate([r.ravel() for r in R]))
            self._unflat(flat, self.T)
            e_ij = np.array([np.sum(self.K[p] * (2 * self.T[p] - self.T[p].T))
                             for p in range(npair)])
            e_cur = e_ij.sum()
            self.log(f'  LMP2 iter {it:3d} {e_cur:.12f} {e_cur - e_prev:.3e} {rms.max():.3e}')
            if abs(e_cur - e_prev) < ec and rms.max() < rc:
                break
            e_prev = e_cur
        else:
            raise RuntimeError('LMP2 did not converge')
        self.e_lmp2 = e_cur
        return e_ij

    @staticmethod
    def _unflat(flat, mats):
        off = 0
        for k, m in enumerate(mats):
            mats[k] = flat[off:off + m.size].reshape(m.shape)
            off += m.size

    def recompute_pnos(self):
        opt, pl = self.opt, self.pl
        for p, (i, j) in enumerate(pl.pairs):
            if i > j:
                continue
            T = self.T[p]
            K = self.K[p]
            Tt = 2 * T - T.T
            Finit = np.diag(self.e_pno[p])
            D = Tt @ T.T + Tt.T @ T
            if i == j:
                D *= .5
            occ, U = np.linalg.eigh(D)
            occ, U = occ[::-1], U[:, ::-1]
            scale = 1.
            if i == j:
                scale *= opt['T_CUT_PNO_DIAG_SCALE']
            if i < self.ncore_scale or j < self.ncore_scale:
                scale *= opt['T_CUT_PNO_CORE_SCALE']
            occ_total = occ.sum()
            e_tot = np.sum(K * Tt)
            Kn = U.T @ K @ U
            Ttn = U.T @ Tt @ U
            e_pno = occ_pno = 0.
            a_curr = []
            nfin = 0
            for a in range(len(occ)):
                if (abs(occ[a]) >= scale * opt['T_CUT_PNO']
                        or occ_pno / occ_total < opt['T_CUT_TRACE']
                        or abs(e_pno) < opt['T_CUT_ENERGY'] * abs(e_tot)
                        or a < opt['MIN_PNOS']):
                    a_curr.append(a)
                    e_pno = np.sum(Kn[np.ix_(a_curr, a_curr)] * Ttn[np.ix_(a_curr, a_curr)])
                    occ_pno += occ[a]
                    nfin += 1
            nfin = max(1, nfin)
            U = U[:, :nfin]
            V, ep = su.canonicalize(U, Finit)
            U = U @ V
            Kp = U.T @ K @ U
            Tp = U.T @ T @ U
            self.de_pno[p] += e_tot - np.sum(Kp * (2 * Tp - Tp.T))
            dpn = self.de_pno[p]
            self._store_pair(p, self.X[p] @ U, ep, Kp, Tp)
            self.de_pno[pl.ji[p]] = dpn
        self._S = {}
        self.de_pno_total = self.de_pno.sum()
        self.de_weak = sum(np.sum(self.K[p] * (2 * self.T[p] - self.T[p].T))
                           for p in range(len(pl)) if not self.strong[p])

    # ------------------------------------------------------------------
    # CCSD integrals
    # ------------------------------------------------------------------
    def pno_integrals(self):
        pl, orb = self.pl, self.orb
        npair = len(pl)
        self.K_mibj = [None] * npair
        self.J_ijmb = [None] * npair
        self.K_ivvv = [None] * npair
        self.L_iajb = [None] * npair
        self.J_ikac = [None] * npair
        self.K_iakc = [None] * npair
        self.iQk = [None] * npair
        self.iQa = [None] * npair
        self.Qma = [None] * npair
        self.Qab = [None] * npair
        for p, (i, j) in enumerate(pl.pairs):
            if i > j:
                continue
            q = pl.ji[p]
            aux, paos, lm, X = self.ribfs[p], self.pao[p], self.lmos[p], self.X[p]
            na, nl, nv = len(aux), len(lm), X.shape[1]
            q_pair = self.qij[aux, i, j]
            q_io = self.qij[np.ix_(aux, [i], lm)][:, 0]
            q_jo = self.qij[np.ix_(aux, [j], lm)][:, 0]
            q_iv = self.qia[np.ix_(aux, [i], paos)][:, 0] @ X
            q_jv = self.qia[np.ix_(aux, [j], paos)][:, 0] @ X
            q_ov = self.qia[np.ix_(aux, lm, paos)] @ X
            q_vv = lib.einsum('ua,Quv,vb->Qab', X, self.qab[np.ix_(aux, paos, paos)], X)
            A = self.J2c[np.ix_(aux, aux)]
            cho = scipy.linalg.cho_factor(A)
            q_io_c = scipy.linalg.cho_solve(cho, q_io)
            q_jo_c = scipy.linalg.cho_solve(cho, q_jo)
            q_iv_c = scipy.linalg.cho_solve(cho, q_iv)
            q_jv_c = scipy.linalg.cho_solve(cho, q_jv)
            ih = _sym_inv_sqrt_solver(A)
            q_pair, q_io, q_jo, q_iv, q_jv, q_ov, q_vv = map(
                ih, (q_pair, q_io, q_jo, q_iv, q_jv, q_ov, q_vv))
            if self.strong[p]:
                ext = su.union(paos, *[self.dom.lmo_paos[k] for k in lm])
                pos = -np.ones(orb.nbf, dtype=int)
                pos[ext] = np.arange(len(ext))
                q_vvp = lib.einsum('ua,Quv->Qav', X, self.qab[np.ix_(aux, paos, ext)])
                Kio = (q_io_c.T @ q_vvp.reshape(na, -1)).reshape(nl, nv, len(ext))
                Kjo = (q_jo_c.T @ q_vvp.reshape(na, -1)).reshape(nl, nv, len(ext))
                q_ovx = self.qia[np.ix_(aux, lm, ext)]
                Koviv = (q_ovx.reshape(na, -1).T @ q_iv_c).reshape(nl, len(ext), nv)
                Kovjv = (q_ovx.reshape(na, -1).T @ q_jv_c).reshape(nl, len(ext), nv)
                Jp, Kp, Jq, Kq = [], [], [], []
                for kk, k in enumerate(lm):
                    kj, ik = pl.ij[k, j], pl.ij[i, k]
                    Jp.append(Kio[kk][:, pos[self.pao[kj]]] @ self.X[kj])
                    Kp.append((self.X[kj].T @ Koviv[kk][pos[self.pao[kj]]]).T)
                    Jq.append(Kjo[kk][:, pos[self.pao[ik]]] @ self.X[ik])
                    Kq.append((self.X[ik].T @ Kovjv[kk][pos[self.pao[ik]]]).T)
                self.J_ikac[p], self.K_iakc[p] = Jp, Kp
                if q != p:
                    self.J_ikac[q], self.K_iakc[q] = Jq, Kq
            self.K_mibj[p] = q_io.T @ q_jv
            self.J_ijmb[p] = q_pair @ q_ov.reshape(na, -1)
            self.J_ijmb[p] = self.J_ijmb[p].reshape(nl, nv)
            self.K_ivvv[p] = q_iv.T @ q_vv.reshape(na, -1)
            if q != p:
                self.K_mibj[q] = q_jo.T @ q_iv
                self.J_ijmb[q] = self.J_ijmb[p]
                self.K_ivvv[q] = q_jv.T @ q_vv.reshape(na, -1)
            if self.strong[p]:
                self.iQk[p], self.iQa[p] = q_io, q_iv
                self.Qma[p], self.Qab[p] = q_ov, q_vv
                if q != p:
                    self.iQk[q], self.iQa[q] = q_jo, q_jv
                    self.Qma[q], self.Qab[q] = q_ov, q_vv
            K = self.K[p]
            self.L_iajb[p] = 2 * K - K.T
            if q != p:
                self.L_iajb[q] = self.L_iajb[p].T

    # ------------------------------------------------------------------
    # LCCSD
    # ------------------------------------------------------------------
    def _T_n(self):
        pl = self.pl
        self.T_n = []
        for p in range(len(pl)):
            lm = self.lmos[p]
            tn = np.zeros((len(lm), self.X[p].shape[1]))
            for nn, n in enumerate(lm):
                tn[nn] = self.S(p, pl.ij[n, n]) @ self.T1[n]
            self.T_n.append(tn)

    def _t1_ints(self):
        pl = self.pl
        self.iQk_t1 = [None] * len(pl)
        self.iQa_t1 = [None] * len(pl)
        for p, (i, j) in enumerate(pl.pairs):
            if not self.strong[p]:
                continue
            ii = self.lmo_pos[p][i]
            Qma, Qab, Tn = self.Qma[p], self.Qab[p], self.T_n[p]
            qk = self.iQk[p] + lib.einsum('qka,a->qk', Qma, Tn[ii])
            qa = self.iQa[p] - qk @ Tn + lib.einsum('qab,b->qa', Qab, Tn[ii])
            self.iQk_t1[p], self.iQa_t1[p] = qk, qa

    def _t1_fock(self):
        pl, n = self.pl, self.n
        npair = len(pl)
        Fij_bar = self.orb.F_lmo.copy()
        Fkc_bar = [None] * npair
        Fab_bar = [None] * npair
        Fai_bar = [None] * n
        for p, (i, j) in enumerate(pl.pairs):
            q = pl.ji[p]
            Tn = self.T_n[p]
            Fij_bar[i, j] += 2 * np.sum(Tn * self.J_ijmb[p]) - np.sum(Tn * self.K_mibj[q])
            if i > j or not self.strong[p]:
                continue
            Qma, Qab = self.Qma[p], self.Qab[p]
            g = lib.einsum('qme,me->q', Qma, Tn)
            fkc = 2 * lib.einsum('q,qkc->kc', g, Qma)
            fkc -= lib.einsum('qke,me,qmc->kc', Qma, Tn, Qma)
            fab = np.diag(self.e_pno[p]) + 2 * lib.einsum('q,qab->ab', g, Qab)
            fab -= lib.einsum('qae,me,qmb->ab', Qab, Tn, Qma)
            Fkc_bar[p], Fab_bar[p] = fkc, fab
            if i == j:
                Qia, Qik = self.iQa[p], self.iQk[p]
                lam = lib.einsum('qae,me->qam', Qab, Tn)
                Fai_bar[i] = 2 * g @ Qia - lib.einsum('qam,qm->a', lam, Qik)
        self.Fkj = Fij_bar.copy()
        self.Fkc = [None] * npair
        self.Fab = [None] * npair
        self.Fai = [None] * n
        for p, (i, j) in enumerate(pl.pairs):
            jj = pl.ij[j, j]
            self.Fkj[i, j] += Fkc_bar[jj][self.lmo_pos[jj][i]] @ self.T1[j]
            fkc = np.zeros(self.X[p].shape[1])
            for k in self.lmos[p]:
                ik, kk = pl.ij[i, k], pl.ij[k, k]
                tk = self.S(ik, kk) @ self.T1[k]
                fkc += self.S(p, ik) @ (self.L_iajb[ik] @ tk)
            self.Fkc[p] = fkc
            if not self.strong[p] or i > j:
                continue
            Tn = self.T_n[p]
            self.Fab[p] = Fab_bar[p] - Tn.T @ Fkc_bar[p]
            if i == j:
                fai = Fai_bar[i] + Fab_bar[p] @ self.T1[i] - Tn.T @ (Fkc_bar[p] @ self.T1[i])
                lm = self.lmos[p]
                fai -= Tn.T @ Fij_bar[lm, i]
                self.Fai[i] = fai

    def _beta(self):
        pl = self.pl
        beta = [None] * len(pl)
        for p, (i, j) in enumerate(pl.pairs):
            if not self.strong[p]:
                continue
            b = self.iQk_t1[p].T @ self.iQk_t1[pl.ji[p]]
            b += lib.einsum('qkc,cd,qld->kl', self.Qma[p], self.T[p], self.Qma[p])
            beta[p] = b
        return beta

    def _gamma(self):
        pl = self.pl
        out = [None] * len(pl)
        for p, (k, i) in enumerate(pl.pairs):
            ii = pl.ij[i, i]
            nv = self.X[p].shape[1]
            g = -self.T_n[p].T @ self.J_ijmb[p]
            Ti = self.S(p, ii) @ self.T1[i]
            g += (Ti @ self.K_ivvv[p]).reshape(nv, nv)
            for l in self.lmos[p]:
                kl, ll, li = pl.ij[k, l], pl.ij[l, l], pl.ij[l, i]
                Tl = self.S(p, ll) @ self.T1[l]
                Tikl = self.S(kl, ii) @ self.T1[i]
                Kkl = self.S(p, kl) @ (self.K[kl].T @ Tikl)
                g -= np.outer(Tl, Kkl)
                tmp = self.T[li] @ self.S(li, kl) @ self.K[kl]
                g -= .5 * (self.S(p, li) @ tmp @ self.S(kl, p))
            out[p] = g
        return out

    def _delta(self):
        pl = self.pl
        out = [None] * len(pl)
        for p, (i, k) in enumerate(pl.pairs):
            ii, q = pl.ij[i, i], pl.ji[p]
            nv = self.X[p].shape[1]
            M = 2 * self.K_mibj[p] - self.J_ijmb[p]
            d = -self.T_n[p].T @ M
            Ti = self.S(p, ii) @ self.T1[i]
            Kq = self.K_ivvv[q]
            d += (2 * (Kq.reshape(nv * nv, nv) @ Ti).reshape(nv, nv)).T
            d -= (Ti @ Kq).reshape(nv, nv)
            for l in self.lmos[p]:
                ll, lk, il = pl.ij[l, l], pl.ij[l, k], pl.ij[i, l]
                Tl = self.S(p, ll) @ self.T1[l]
                Tilk = self.S(lk, ii) @ self.T1[i]
                Llk = Tilk @ self.L_iajb[lk] @ self.S(lk, p)
                d -= np.outer(Tl, Llk)
                Til = 2 * self.T[il] - self.T[il].T
                tmp = Til @ self.S(il, lk) @ self.L_iajb[lk]
                d += .5 * (self.S(p, il) @ tmp @ self.S(lk, p))
            out[p] = d
        return out

    def _Fkj_dd(self):
        pl = self.pl
        F = self.Fkj.copy()
        for p, (i, j) in enumerate(pl.pairs):
            for l in self.lmos[p]:
                li, lj = pl.ij[l, i], pl.ij[l, j]
                if li < 0 or lj < 0:
                    continue
                U = self.S(li, lj) @ (2 * self.T[lj] - self.T[lj].T) @ self.S(lj, li)
                F[i, j] += np.sum(self.K[li] * U)
        return F

    def _R1(self):
        pl, n = self.pl, self.n
        R1 = [self.Fai[i].copy() for i in range(n)]
        for p, (i, k) in enumerate(pl.pairs):
            q = pl.ji[p]
            ii = pl.ij[i, i]
            nv = self.X[p].shape[1]
            Tt_ki = 2 * self.T[q] - self.T[q].T
            Kk = self.K_ivvv[q].reshape(nv * nv, nv)
            R1[i] += self.S(q, ii).T @ (Kk.T @ Tt_ki.ravel())
            for l in self.lmos[p]:
                kl, ll = pl.ij[k, l], pl.ij[l, l]
                Tl = self.S(ii, ll) @ self.T1[l]
                U = self.S(kl, q) @ Tt_ki @ self.S(q, kl)
                R1[i] -= Tl * np.sum(U * self.K[kl])
            Tt_ik = 2 * self.T[p] - self.T[p].T
            R1[i] += self.S(p, ii).T @ (Tt_ik @ self.Fkc[q])
        for p, (k, l) in enumerate(pl.pairs):
            Kkilc = self.K_mibj[p] + self.T_n[p] @ self.K[p]
            Tt = 2 * self.T[p] - self.T[p].T
            B = Kkilc @ Tt.T
            for ii_, i in enumerate(self.lmos[p]):
                R1[i] -= B[ii_] @ self.S(p, pl.ij[i, i])
        return R1

    def _R2(self):
        pl = self.pl
        npair = len(pl)
        beta = self._beta()
        gamma = self._gamma()
        delta = self._delta()
        Fdd = self._Fkj_dd()
        R = [np.zeros_like(t) for t in self.T]
        Rn = [np.zeros_like(t) for t in self.T]
        for p, (i, j) in enumerate(pl.pairs):
            if not self.strong[p]:
                continue
            q = pl.ji[p]
            lm = self.lmos[p]
            if i <= j:
                Kij = self.iQa_t1[p].T @ self.iQa_t1[q]
                Qab_t1 = self.Qab[p] - lib.einsum('ka,qkb->qab', self.T_n[p], self.Qma[p])
                A = lib.einsum('qac,cd,qbd->ab', Qab_t1, self.T[p], Qab_t1)
                B = np.zeros_like(A)
                Fbc = self.Fab[p].copy()
                for kk, k in enumerate(lm):
                    for ll, l in enumerate(lm):
                        kl = pl.ij[k, l]
                        if kl < 0:
                            continue
                        s = self.S(kl, p)
                        B += beta[p][kk, ll] * (s.T @ self.T[kl] @ s)
                        Ttkl = 2 * self.T[kl] - self.T[kl].T
                        Fbc -= s.T @ (Ttkl @ self.K[kl].T) @ s
                E = self.T[p] @ Fbc.T + Fbc @ self.T[p]
                tot = Kij + A + B + E
                R[p] += tot
                if i != j:
                    R[q] += tot.T
            C = np.zeros_like(self.T[p])
            D = np.zeros_like(self.T[p])
            for kk, k in enumerate(lm):
                ki, kj, ik, jk = pl.ij[k, i], pl.ij[k, j], pl.ij[i, k], pl.ij[j, k]
                gt = self.J_ikac[p][kk] + self.S(p, ki) @ gamma[ki] @ self.S(ki, kj)
                C -= gt @ self.T[kj].T @ self.S(kj, p)
                dt = 2 * self.K_iakc[p][kk] - self.J_ikac[p][kk]
                dt = dt + self.S(p, ik) @ delta[ik] @ self.S(ik, jk)
                Ttjk = 2 * self.T[jk] - self.T[jk].T
                D += dt @ Ttjk.T @ self.S(jk, p)
            Rn[p] += .5 * C + C.T + .5 * D
            G = np.zeros_like(self.T[p])
            for k in range(self.n):
                ik = pl.ij[i, k]
                if ik < 0:
                    continue
                G -= Fdd[k, j] * (self.S(p, ik) @ self.T[ik] @ self.S(ik, p))
            Rn[p] += G
        for p in range(npair):
            if self.strong[p]:
                R[p] += Rn[p] + Rn[pl.ji[p]].T
            else:
                R[p] = np.zeros_like(self.T[p])
        return R

    def lccsd_iterations(self):
        opt, pl, n = self.opt, self.pl, self.n
        f = self.orb.F_lmo
        self.T1 = [np.zeros(self.X[pl.ij[i, i]].shape[1]) for i in range(n)]
        diis = DIIS(opt['DIIS_MAX_VECS'])
        e_prev = 0.
        for it in range(1, opt['MAXITER'] + 1):
            self._T_n()
            self._t1_ints()
            self._t1_fock()
            R1 = self._R1()
            R2 = self._R2()
            r1 = max(_rms(r) for r in R1)
            r2 = max(_rms(r) for r in R2)
            for i in range(n):
                ii = pl.ij[i, i]
                self.T1[i] = self.T1[i] - R1[i] / (self.e_pno[ii] - f[i, i])
            for p, (i, j) in enumerate(pl.pairs):
                e = self.e_pno[p]
                self.T[p] = self.T[p] - R2[p] / (e[:, None] + e[None, :] - f[i, i] - f[j, j])
            flat = diis.update(
                np.concatenate([t.ravel() for t in self.T1 + self.T]),
                np.concatenate([r.ravel() for r in R1 + R2]))
            mats = self.T1 + self.T
            self._unflat(flat, mats)
            self.T1, self.T = mats[:n], mats[n:]
            e_cur = e_weak = 0.
            for p, (i, j) in enumerate(pl.pairs):
                ti = self.S(p, pl.ij[i, i]) @ self.T1[i]
                tj = self.S(p, pl.ij[j, j]) @ self.T1[j]
                eij = np.sum((self.T[p] + np.outer(ti, tj)) * self.L_iajb[p])
                e_cur += eij
                if not self.strong[p]:
                    e_weak += eij
            self.log(f'  LCCSD iter {it:3d} {e_cur:.12f} {e_cur - e_prev:.3e} {r1:.3e} {r2:.3e}')
            if abs(e_cur - e_prev) < opt['E_CONVERGENCE'] and max(r1, r2) < opt['R_CONVERGENCE']:
                break
            e_prev = e_cur
        else:
            raise RuntimeError('LCCSD did not converge')
        self.e_lccsd = e_cur - e_weak
        self.de_weak = e_weak

    # ------------------------------------------------------------------
    # (T)
    # ------------------------------------------------------------------
    def triples_sparsity(self, prescreen):
        opt, pl, n = self.opt, self.pl, self.n
        if prescreen:
            trip = []
            for p, (i, j) in enumerate(pl.pairs):
                if i > j:
                    continue
                for k in self.lmos[p]:
                    if i > k or j > k or (i == j == k):
                        continue
                    nweak = sum(not self.strong[pl.ij[a, b]] for a, b in ((i, j), (i, k), (k, j)))
                    if nweak > opt['TRIPLES_MAX_WEAK_PAIRS']:
                        continue
                    trip.append((i, j, k))
            self.trip = trip
            self.de_t_screened = 0.
        else:
            keep = [t for t, e in zip(self.trip, self.e_ijk)
                    if abs(e) >= opt['T_CUT_TRIPLES_WEAK']]
            self.de_t_screened = sum(e for e in self.e_ijk
                                     if abs(e) < opt['T_CUT_TRIPLES_WEAK'])
            self.trip = keep
        self.trip_idx = {}
        for t, (i, j, k) in enumerate(self.trip):
            for perm in ((i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)):
                self.trip_idx[perm] = t
        mkn = opt['T_CUT_MKN_TRIPLES_PRE'] if prescreen else opt['T_CUT_MKN_TRIPLES']
        do = opt['T_CUT_DO_TRIPLES_PRE'] if prescreen else opt['T_CUT_DO_TRIPLES']
        dom = self.domains(mkn, do)
        self.t_ribfs = [su.union(dom.lmo_ribfs[i], dom.lmo_ribfs[j], dom.lmo_ribfs[k])
                        for (i, j, k) in self.trip]
        self.t_paos = [su.union(dom.lmo_paos[i], dom.lmo_paos[j], dom.lmo_paos[k])
                       for (i, j, k) in self.trip]
        self.t_lmos = [np.where((pl.ij[i] >= 0) & (pl.ij[j] >= 0) & (pl.ij[k] >= 0))[0]
                       for (i, j, k) in self.trip]
        self.tno_scale = np.ones(len(self.trip))

    def tno_transform(self, t_cut):
        opt, pl, orb = self.opt, self.pl, self.orb
        self.X_tno, self.e_tno = [], []
        for t, (i, j, k) in enumerate(self.trip):
            paos = self.t_paos[t]
            Xp, ep = su.orthocanonicalize(orb.S_pao[np.ix_(paos, paos)],
                                          orb.F_pao[np.ix_(paos, paos)], opt['S_CUT'])
            D = 0.
            for a, b in ((i, j), (j, k), (i, k)):
                p = pl.ij[a, b]
                T = self.T[p]
                Tt = 2 * T - T.T
                Dp = Tt @ T.T + Tt.T @ T
                if a == b:
                    Dp *= .5
                s = self.X[p].T @ orb.S_pao[np.ix_(self.pao[p], paos)] @ Xp
                D = D + s.T @ Dp @ s
            D /= 3.
            occ, U = np.linalg.eigh(D)
            occ, U = occ[::-1], U[:, ::-1]
            nfin = sum(1 for a in range(len(occ))
                       if abs(occ[a]) >= self.tno_scale[t] * t_cut or a < opt['MIN_TNOS'])
            nfin = max(1, nfin)
            U = U[:, :nfin]
            V, e = su.canonicalize(U, np.diag(ep))
            self.X_tno.append(Xp @ U @ V)
            self.e_tno.append(e)

    def _S_tp(self, t, p):
        """S(a_triplet, a_pair)."""
        return self.X_tno[t].T @ self.orb.S_pao[np.ix_(self.t_paos[t], self.pao[p])] @ self.X[p]

    def lccsd_t0(self, save=False):
        pl, f = self.pl, self.orb.F_lmo
        E = 0.
        self.e_ijk = np.zeros(len(self.trip))
        if save:
            self.W3, self.V3, self.T3 = [], [], []
        for t, (i, j, k) in enumerate(self.trip):
            aux, paos, lm, X = self.t_ribfs[t], self.t_paos[t], self.t_lmos[t], self.X_tno[t]
            nv = X.shape[1]
            qo = {a: self.qij[np.ix_(aux, [a], lm)][:, 0] for a in (i, j, k)}
            qv = {a: self.qia[np.ix_(aux, [a], paos)][:, 0] @ X for a in (i, j, k)}
            q_vv = lib.einsum('ua,Quv,vb->Qab', X, self.qab[np.ix_(aux, paos, paos)], X)
            A = self.J2c[np.ix_(aux, aux)]
            cho = scipy.linalg.cho_factor(A)
            qvc = {a: scipy.linalg.cho_solve(cho, qv[a]) for a in qv}
            ih = _sym_inv_sqrt_solver(A)
            qv = {a: ih(v) for a, v in qv.items()}
            qo = {a: ih(v) for a, v in qo.items()}
            na = len(aux)
            Kvvv = {a: (qvc[a].T @ q_vv.reshape(na, -1)).reshape(nv, nv, nv) for a in qv}
            Kov = {(a, b): qo[a].T @ qv[b] for a in qv for b in qv}
            perms = [(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)]
            Wp = []
            for (I, J, K) in perms:
                kj = pl.ij[K, J]
                s = self._S_tp(t, kj)
                Tkj = s @ self.T[kj] @ s.T
                w = lib.einsum('abd,cd->abc', Kvvv[I], Tkj)
                Kl = Kov[(J, K)]
                for ll, l in enumerate(lm):
                    il = pl.ij[I, l]
                    s = self._S_tp(t, il)
                    Til = s @ self.T[il] @ s.T
                    w -= Til[:, :, None] * Kl[ll][None, None, :]
                Wp.append(w)
            W = (Wp[0] + Wp[1].transpose(0, 2, 1) + Wp[2].transpose(1, 0, 2)
                 + Wp[3].transpose(2, 0, 1) + Wp[4].transpose(1, 2, 0)
                 + Wp[5].transpose(2, 1, 0))
            Ti = self._S_tp(t, pl.ij[i, i]) @ self.T1[i]
            Tj = self._S_tp(t, pl.ij[j, j]) @ self.T1[j]
            Tk = self._S_tp(t, pl.ij[k, k]) @ self.T1[k]
            Kjk = qv[j].T @ qv[k]
            Kik = qv[i].T @ qv[k]
            Kij = qv[i].T @ qv[j]
            V = (W + Ti[:, None, None] * Kjk[None] + Tj[None, :, None] * Kik[:, None, :]
                 + Tk[None, None, :] * Kij[:, :, None])
            e = self.e_tno[t]
            Dn = e[:, None, None] + e[None, :, None] + e[None, None, :] - f[i, i] - f[j, j] - f[k, k]
            T3 = -W / Dn
            eijk = self._e_ijk(i, j, k, V, T3)
            self.e_ijk[t] = eijk
            E += eijk
            if save:
                self.W3.append(W)
                self.V3.append(V)
                self.T3.append(T3)
        return E

    @staticmethod
    def _e_ijk(i, j, k, V, T3):
        pref = 1.
        if i == j == k:
            pref /= 6.
        elif i == j or j == k or i == k:
            pref /= 2.
        e = 8 * np.sum(V * T3)
        e -= 4 * np.sum(_permute(V, k, j, i) * T3)
        e -= 4 * np.sum(_permute(V, i, k, j) * T3)
        e -= 4 * np.sum(_permute(V, j, i, k) * T3)
        e += 2 * np.sum(_permute(V, j, k, i) * T3)
        e += 2 * np.sum(_permute(V, k, i, j) * T3)
        return pref * e

    def sort_triplets(self, e_total):
        opt = self.opt
        order = np.argsort(-np.abs(self.e_ijk), kind='stable')
        self.tno_scale = np.full(len(self.trip), opt['T_CUT_TNO_WEAK_SCALE'])
        self.strong_trip = np.zeros(len(self.trip), bool)
        e = 0.
        for t in order:
            self.strong_trip[t] = True
            self.tno_scale[t] = opt['T_CUT_TNO_STRONG_SCALE']
            e += self.e_ijk[t]
            if e / e_total > 0.9:
                break

    def _matmul_3d(self, A, S):
        return lib.einsum('aA,bB,cC,ABC->abc', S, S, S, A)

    def lccsd_t_iterations(self):
        opt, f, n = self.opt, self.orb.F_lmo, self.n
        ntrip = len(self.trip)
        e_old = np.zeros(ntrip)
        e_prev = 0.
        orb = self.orb
        for it in range(1, opt['MAXITER'] + 1):
            rms = np.zeros(ntrip)
            for t, (i, j, k) in enumerate(self.trip):
                if abs(self.e_ijk[t] - e_old[t]) < abs(e_old[t] * opt['T_CUT_ITER']):
                    continue
                e = self.e_tno[t]
                Dn = (e[:, None, None] + e[None, :, None] + e[None, None, :]
                      - f[i, i] - f[j, j] - f[k, k])
                R = self.W3[t] + self.T3[t] * Dn
                for l in range(n):
                    for (key, fl, perm) in (((i, j, l), f[l, k], (i, j, l)) if l != k else (None, 0, None),
                                            ((i, l, k), f[l, j], (i, l, k)) if l != j else (None, 0, None),
                                            ((l, j, k), f[l, i], (l, j, k)) if l != i else (None, 0, None)):
                        if key is None or key not in self.trip_idx or abs(fl) < opt['F_CUT_T']:
                            continue
                        u = self.trip_idx[key]
                        s = (self.X_tno[t].T @ orb.S_pao[np.ix_(self.t_paos[t], self.t_paos[u])]
                             @ self.X_tno[u])
                        R -= fl * self._matmul_3d(_permute(self.T3[u], *perm), s)
                self.T3[t] = self.T3[t] - R / Dn
                rms[t] = _rms(R)
            e_old = self.e_ijk.copy()
            E = 0.
            for t, (i, j, k) in enumerate(self.trip):
                self.e_ijk[t] = self._e_ijk(i, j, k, self.V3[t], self.T3[t])
                E += self.e_ijk[t]
            self.log(f'  LCCSD(T) iter {it:3d} {E:.12f} {E - e_prev:.3e} {rms.max():.3e}')
            if rms.max() < opt['R_CONVERGENCE'] and abs(E - e_prev) < opt['E_CONVERGENCE']:
                break
            e_prev = E
        else:
            raise RuntimeError('(T) iterations did not converge')
        return E

    # ------------------------------------------------------------------
    def kernel(self):
        opt = self.opt
        t = time.perf_counter()
        self.setup()
        n = self.n
        mask = ((self.doi_ij > opt['T_CUT_DO_IJ']) | (np.abs(self.dip_b) > opt['T_CUT_PRE'])
                | np.eye(n, dtype=bool))
        self.de_dipole = float(np.sum(self.dip_e[~mask]))
        t = time.perf_counter()
        # crude prescreening
        self.set_pairs(mask, self.domains(opt['T_CUT_MKN'] * 100, opt['T_CUT_DO'] * 2))
        e_crude = self.sc_mp2_pair_energies()
        keep = np.zeros((n, n), bool)
        self.de_lmp2_eliminated = 0.
        for p, (i, j) in enumerate(self.pl.pairs):
            if abs(e_crude[p]) >= opt['T_CUT_PAIRS_MP2'] or i == j:
                keep[i, j] = True
            else:
                self.de_lmp2_eliminated += e_crude[p]
        t = self._tick('crude_prescreen', t)
        # refined prescreening
        self.set_pairs(keep, self.domains(opt['T_CUT_MKN'], opt['T_CUT_DO']))
        self.lmp2_pnos()
        e_ij = self.lmp2_iterations()
        pl = self.pl
        self.strong = np.array([abs(e_ij[p]) >= opt['T_CUT_PAIRS'] or i == j
                                for p, (i, j) in enumerate(pl.pairs)])
        self.e['lmp2'] = (self.e_lmp2 + self.de_lmp2_eliminated + self.de_dipole
                          + self.de_pno_total)
        t = self._tick('lmp2', t)
        self.recompute_pnos()
        self.pno_integrals()
        t = self._tick('pno_ints', t)
        self.lccsd_iterations()
        t = self._tick('lccsd', t)
        e_corr = (self.e_lccsd + self.de_weak + self.de_lmp2_eliminated
                  + self.de_dipole + self.de_pno_total)
        self.e.update(ccsd=e_corr, lccsd=self.e_lccsd, weak=self.de_weak,
                      sc_lmp2=self.de_lmp2_eliminated, dipole=self.de_dipole,
                      pno_trunc=self.de_pno_total)
        if self.triples:
            self.triples_sparsity(True)
            self.tno_transform(opt['T_CUT_TNO_PRE'])
            self.lccsd_t0()
            self.triples_sparsity(False)
            self.tno_transform(opt['T_CUT_TNO'])
            e_t0 = self.lccsd_t0()
            e_t = e_t0 + self.de_t_screened
            self.e.update(t0=e_t, t_screened=self.de_t_screened)
            if self.triples in ('t', 't1'):
                self.sort_triplets(e_t0)
                self.tno_transform(opt['T_CUT_TNO'])
                e_t0_crude = self.lccsd_t0(save=True)
                e_titer = self.lccsd_t_iterations()
                e_t += e_titer - e_t0_crude
                self.e.update(t0_crude=e_t0_crude, t_iter=e_titer)
            self.e['t'] = e_t
            e_corr += e_t
            t = self._tick('triples', t)
        self.e['corr'] = e_corr
        self.e['tot'] = self.mf.e_tot + e_corr
        return self.e['tot']
