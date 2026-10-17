"""High-spin open-shell DLPNO-CCSD(T) on the host: the reference.

Reference determinant: one set of spatial orbitals with doubly (DOMO) and
singly (SOMO) occupied blocks -- ROHF, or the quasi-restricted orbitals of a
UHF (``qro``).  This is what ORCA correlates (Saitow et al., JCP 146,
164105 (2017)).  The correlation treatment is spin-unrestricted:

* spatial LMO pairs (x, y) over DOMOs and SOMOs, screened and classified
  strong/weak as in the closed-shell code by a spin-summed pair energy;
* ONE set of PNOs per spatial pair (spin independent); the alpha virtual
  space of a pair is its PNOs, the beta virtual space its PNOs plus all SOMOs
  (SOMOs are virtual for beta electrons), canonicalised with F_alpha and
  F_beta respectively;
* the CC equations are the spin-orbital T1-dressed CCSD equations
  (validated against PySCF UCCSD for non-canonical ROHF orbitals), solved
  per spin-orbital pair in that pair's (alpha + beta) virtual space, with
  the PNO projections placed where the closed-shell formulation of Jiang et
  al. (2024) places them.

The PNOs come from the semicanonical spin-orbital MP2 amplitudes in the PAO
domain (the beta virtual space of this guess is the PAO domain only, so the
PNOs are PAO combinations), from the spin-averaged pair density

    D_xy = 1/2 (A_a A_a^T + A_b A_b^T) + 1/2 (O O^T + O^T O + O' O'^T + O'^T O'),

A = same-spin amplitudes, O / O' = opposite-spin amplitudes of (x a, y b) /
(y a, x b), halved for x == y -- which for a closed shell is exactly the
closed-shell PNO density.  Pair energies, PNO truncation (occupation, trace,
energy) and every threshold are the closed-shell ones applied to the
spin-summed pair energy, so a closed-shell molecule run through this code
gets the closed-shell pairs and PNOs.

Spin-orbital storage: every spin-orbital pair (I, J) keeps a dense
(nv_a + nv_b) x (nv_a + nv_b) amplitude matrix with its spin-forbidden blocks
zero.  That is wasteful but makes every term the canonical spin-orbital
formula; the GPU code stores the blocks.
"""
import time

import numpy as np
import scipy.linalg

from pyscf import lib, df, scf

from electrolites.dlpno import setup as su
from electrolites.dlpno.options import make_options
from electrolites.dlpno.ref_rhf import DIIS, _rms


def qro(mf_uhf):
    """Quasi-restricted orbitals of a UHF: natural orbitals of the total
    density, classified by occupation, each block canonicalised with the
    spin-averaged Fock operator.  Returns (C, nd, ns)."""
    mol = mf_uhf.mol
    S = su.asnumpy(mf_uhf.get_ovlp())
    dm = su.asnumpy(mf_uhf.make_rdm1())
    P = dm[0] + dm[1]
    na, nb = mol.nelec
    nd, ns = nb, na - nb
    # natural orbitals: S^1/2 P S^1/2
    e, U = np.linalg.eigh(S)
    Sh = (U * np.sqrt(e)) @ U.T
    Shi = (U / np.sqrt(e)) @ U.T
    occ, V = np.linalg.eigh(Sh @ P @ Sh)
    occ, V = occ[::-1], V[:, ::-1]
    C = Shi @ V
    F = su.asnumpy(mf_uhf.get_fock(dm=dm))
    Fav = .5 * (F[0] + F[1])
    out = []
    for blk in (slice(0, nd), slice(nd, nd + ns), slice(nd + ns, C.shape[1])):
        Cb = C[:, blk]
        w, R = np.linalg.eigh(Cb.T @ Fav @ Cb)
        out.append(Cb @ R)
    return np.hstack(out), nd, ns



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

class RefUDLPNO:
    def __init__(self, mf, pno='normal', triples='t0', auxbasis=None,
                 frozen_core=True, localizer='boys', reference='rohf',
                 verbose=None, **opts):
        self.mf = mf
        self.mol = mf.mol
        self.opt = make_options(pno, **opts)
        self.triples = None if triples is None else triples.lower()
        self.auxbasis = auxbasis
        self.frozen_core = frozen_core
        self.localizer = localizer
        self.reference = reference
        self.verbose = self.mol.verbose if verbose is None else verbose
        self.timings = {}
        self.e = {}

    def log(self, *a):
        if self.verbose >= 4:
            print(*a, flush=True)

    def _tick(self, key, t0):
        self.timings[key] = self.timings.get(key, 0.) + time.perf_counter() - t0
        if getattr(self, 'verbose', 0) >= 2:
            _phase_log(key, self.timings[key])
        return time.perf_counter()

    # ------------------------------------------------------------------
    def setup(self):
        mf, mol, opt = self.mf, self.mol, self.opt
        S = su.asnumpy(mf.get_ovlp())
        if self.reference == 'qro':
            C, nd, ns = qro(mf)
        else:
            C = su.asnumpy(mf.mo_coeff)
            occ = su.asnumpy(mf.mo_occ)
            if occ.ndim == 2:          # UHF passed with reference='rohf'
                raise ValueError('pass an ROHF object, or reference="qro" for a UHF')
            nd = int(np.count_nonzero(occ > 1.5))
            ns = int(np.count_nonzero((occ > .5) & (occ < 1.5)))
        self.C_ref = C
        self.nd_all, self.ns = nd, ns
        Da = C[:, :nd + ns] @ C[:, :nd + ns].T
        Db = C[:, :nd] @ C[:, :nd].T
        mfu = scf.UHF(mol)
        if getattr(mf, 'with_df', None) is not None:
            mfu = mfu.density_fit(auxbasis=mf.with_df.auxbasis)
        h = su.asnumpy(mf.get_hcore())
        vhf = mfu.get_veff(mol, np.array([Da, Db]))
        Fa, Fb = h + su.asnumpy(vhf[0]), h + su.asnumpy(vhf[1])
        self.e_ref = float(mfu.energy_tot(dm=np.array([Da, Db]), vhf=vhf))
        self.Fa_ao, self.Fb_ao = Fa, Fb
        nfc = su.count_frozen_core(mol) if self.frozen_core else 0
        assert nfc <= nd
        self.ncore = nfc
        c_d = C[:, nfc:nd]
        c_s = C[:, nd:nd + ns]
        L_d = su.localize(mol, c_d, self.localizer, opt['LOCAL_CONVERGENCE'], mf)
        L_s = su.localize(mol, c_s, self.localizer, opt['LOCAL_CONVERGENCE'], mf) if ns > 1 else c_s
        L = np.hstack([L_d, L_s])
        self.nda = L_d.shape[1]                 # active DOMOs
        self.n = n = L.shape[1]
        self.is_somo = np.arange(n) >= self.nda
        Fav = .5 * (Fa + Fb)
        # Orbitals() builds PAOs against all alpha-occupied orbitals
        orb = su.Orbitals(mol, S, Fav, C, nd + ns, nfc, c_lmo=L)
        self.orb = orb
        P = orb.C_pao
        self.Fa_lmo, self.Fb_lmo = L.T @ Fa @ L, L.T @ Fb @ L
        self.Fa_pao, self.Fb_pao = P.T @ Fa @ P, P.T @ Fb @ P
        self.Fa_lp, self.Fb_lp = L.T @ Fa @ P, L.T @ Fb @ P      # occ - PAO
        Ls = L[:, self.nda:]
        self.Fb_ps = P.T @ Fb @ Ls                                # PAO - SOMO (beta)
        self.Fb_ss = Ls.T @ Fb @ Ls
        self.somo_idx = np.arange(self.nda, n)
        auxbasis = self.auxbasis or df.make_auxbasis(mol, mp2fit=True)
        self.auxmol = auxmol = df.addons.make_auxmol(mol, auxbasis)
        (self.bf_atom, self.ribf_atom, self.atom_bfs,
         self.atom_ribfs) = su.basis_maps(mol, auxmol)
        self.doi_ij, self.doi_iu, self.doi_uv = su.compute_doi(mol, L, P, opt['DOI_GRID'])
        self.dip_e, self.dip_b = su.dipole_pair_energies(
            orb, self.doi_iu, opt['T_CUT_DO_PRE'], opt['S_CUT'], self.bf_atom, self.atom_bfs)
        self.J2c = auxmol.intor('int2c2e')
        int3c = df.incore.aux_e2(mol, auxmol, 'int3c2e')
        tmp = lib.einsum('mnQ,mi->Qin', int3c, L)
        self.qij = lib.einsum('Qin,nj->Qij', tmp, L)
        self.qia = lib.einsum('Qin,nu->Qiu', tmp, P)
        qab = lib.einsum('mnQ,mu,nv->Quv', int3c, P, P)
        self.qab = qab * (self.doi_uv > opt['T_CUT_DO_UV'])[None]
        del int3c, tmp
        self.mkn = su.mulliken_atom_pops(mol, S, L)
        # spin orbitals of the active occupied space: (lmo, spin)
        self.so = [(x, 0) for x in range(n)] + [(x, 1) for x in range(self.nda)]
        self.so_index = {k: I for I, k in enumerate(self.so)}

    def domains(self, mkn, do):
        return su.Domains(self.mkn, self.doi_iu, mkn, do, self.bf_atom, self.atom_bfs,
                          self.atom_ribfs)

    def set_pairs(self, mask, dom):
        self.pl = pl = su.PairList(self.n, mask)
        self.dom = dom
        self.pao = [su.union(dom.lmo_paos[i], dom.lmo_paos[j]) for (i, j) in pl.pairs]
        self.ribfs = [su.union(dom.lmo_ribfs[i], dom.lmo_ribfs[j]) for (i, j) in pl.pairs]
        self.lmos = [pl.lmos(p) for p in range(len(pl))]

    # ------------------------------------------------------------------
    # semicanonical spin-orbital MP2 in the PAO domain
    # ------------------------------------------------------------------
    def _sc_pair(self, p, pno=None):
        """SC-MP2 of the spatial pair p (x <= y): returns the spin-summed
        ordered-pair energy (e_xy + e_yx) / 2 and, with ``pno``, its PNOs."""
        x, y = self.pl.pairs[p]
        aux, paos = self.ribfs[p], self.pao[p]
        iq = self.qia[np.ix_(aux, [x], paos)][:, 0]
        jq = self.qia[np.ix_(aux, [y], paos)][:, 0]
        A = self.J2c[np.ix_(aux, aux)]
        Kp = scipy.linalg.solve(A, iq, assume_a='sym').T @ jq        # (x u|y v)
        Sd = self.orb.S_pao[np.ix_(paos, paos)]
        Xa, ea = su.orthocanonicalize(Sd, self.Fa_pao[np.ix_(paos, paos)], self.opt['S_CUT'])
        Xb, eb = su.orthocanonicalize(Sd, self.Fb_pao[np.ix_(paos, paos)], self.opt['S_CUT'])
        # common orthonormal coordinates: alpha-canonical; U maps beta -> alpha
        U = Xa.T @ Sd @ Xb
        fa, fb = np.diag(self.Fa_lmo), np.diag(self.Fb_lmo)
        dx, dy = not self.is_somo[x], not self.is_somo[y]
        Kaa = Xa.T @ Kp @ Xa
        taa = (Kaa - Kaa.T) / (fa[x] + fa[y] - ea[:, None] - ea[None, :])
        e2 = .5 * np.sum((Kaa - Kaa.T) * taa)
        Bm = np.zeros_like(taa)
        O = np.zeros_like(taa)
        Op = np.zeros_like(taa)
        amps = dict(aa=(taa, Kaa))
        if dx and dy:
            Kbb = Xb.T @ Kp @ Xb
            tbb = (Kbb - Kbb.T) / (fb[x] + fb[y] - eb[:, None] - eb[None, :])
            e2 += .5 * np.sum((Kbb - Kbb.T) * tbb)
            Bm = U @ tbb @ U.T
            amps['bb'] = (tbb, Kbb)
        if dy:
            Kab = Xa.T @ Kp @ Xb
            tab = Kab / (fa[x] + fb[y] - ea[:, None] - eb[None, :])
            e2 += np.sum(Kab * tab)
            O = tab @ U.T
            amps['ab'] = (tab, Kab)
        if dx:
            Kba = Xb.T @ Kp @ Xa
            tba = Kba / (fb[x] + fa[y] - eb[:, None] - ea[None, :])
            e2 += np.sum(Kba * tba)
            Op = (U @ tba).T          # (y alpha, x beta) in alpha coordinates
            amps['ba'] = (tba, Kba)
        e_pair = .5 * e2
        if pno is None:
            return e_pair
        cut, trace, energy, lag = pno
        D = .5 * (taa @ taa.T + Bm @ Bm.T) + .5 * (O @ O.T + O.T @ O + Op @ Op.T + Op.T @ Op)
        if x == y:
            D *= .5
        occ, V = np.linalg.eigh(D)
        occ, V = occ[::-1], V[:, ::-1]
        scale = self.opt['T_CUT_PNO_DIAG_SCALE'] if x == y else 1.
        Vb = U.T @ V                     # PNO vectors in beta-canonical coordinates
        # energy of the first m PNOs (prefix sums)
        M = np.zeros((len(occ), len(occ)))
        for key, (t, K) in amps.items():
            l, r = {'aa': (V, V), 'bb': (Vb, Vb), 'ab': (V, Vb), 'ba': (Vb, V)}[key]
            tn, Kn = l.T @ t @ r, l.T @ K @ r
            if key in ('aa', 'bb'):
                M += .25 * (Kn - Kn.T) * tn
            else:
                M += .5 * Kn * tn
        ecum = np.concatenate([[0.], np.diag(np.cumsum(np.cumsum(M, 0), 1))])
        nfin = self._truncate(occ, ecum, e_pair, scale * cut, trace, energy, lag)
        Xpno = Xa @ V[:, :nfin]           # PAO-domain coefficients (orthonormal)
        e_trunc = ecum[nfin]
        return e_pair, Xpno, e_pair - e_trunc, amps, (Xa, Xb, U, V[:, :nfin], Vb[:, :nfin])

    def _truncate(self, occ, ecum, e_tot, cut, trace, energy, lag):
        occ_total = occ.sum()
        if not occ_total > 1e-12:
            # a pair without amplitudes (the diagonal pair of a SOMO): its
            # space carries only the SOMO's singles -- keep the whole domain
            return len(occ)
        occ_pno = 0.
        nfin = 0
        for a in range(len(occ)):
            e_pno = ecum[max(a - 1, 0)] if lag else ecum[a]
            if (abs(occ[a]) >= cut or occ_pno / occ_total < trace
                    or abs(e_pno) < energy * abs(e_tot) or a < self.opt['MIN_PNOS']):
                occ_pno += occ[a]
                nfin += 1
        # directions with zero occupation (the null space of a rank-deficient
        # density, e.g. a pair of SOMOs) are arbitrary: take all or none
        nnz = int(np.count_nonzero(occ > 1e-12 * occ.max()))
        if nfin > nnz:
            nfin = len(occ)
        nfin = max(1, nfin)
        # never split a degenerate set of PNOs: the density of a pair of
        # SOMOs (same-spin amplitudes only) has pairwise degenerate
        # occupations, and a cut through a pair makes the space arbitrary
        while nfin < len(occ) and abs(occ[nfin] - occ[nfin - 1]) <= 1e-9 * abs(occ[nfin - 1]) \
                and abs(occ[nfin - 1]) > 1e-14:
            nfin += 1
        return nfin

    def sc_mp2_pair_energies(self):
        pl = self.pl
        e = np.zeros(len(pl))
        for p, (x, y) in enumerate(pl.pairs):
            if x <= y:
                e[p] = self._sc_pair(p)
                e[pl.ji[p]] = e[p]
        return e

    # ------------------------------------------------------------------
    # PNO virtual spaces of a spatial pair
    # ------------------------------------------------------------------
    def _pair_spaces(self, p, Xpno):
        """alpha: PNOs canonical in F_alpha; beta: PNOs + SOMOs canonical in
        F_beta.  Returns coefficient matrices over (PAO domain) and over
        (PAO domain + SOMOs), and orbital energies."""
        paos = self.pao[p]
        Fa = Xpno.T @ self.Fa_pao[np.ix_(paos, paos)] @ Xpno
        ea, Ua = np.linalg.eigh(Fa)
        Xa = Xpno @ Ua
        nv, ns = Xpno.shape[1], self.ns
        Fb = np.zeros((nv + ns, nv + ns))
        Fb[:nv, :nv] = Xpno.T @ self.Fb_pao[np.ix_(paos, paos)] @ Xpno
        Fb[:nv, nv:] = Xpno.T @ self.Fb_ps[paos]
        Fb[nv:, :nv] = Fb[:nv, nv:].T
        Fb[nv:, nv:] = self.Fb_ss
        eb, Ub = np.linalg.eigh(Fb)
        Xb = np.zeros((len(paos) + ns, nv + ns))
        Xb[:len(paos)] = Xpno @ Ub[:nv]
        Xb[len(paos):] = Ub[nv:]
        return Xa, ea, Xb, eb

    def _store_spaces(self, p, Xpno):
        q = self.pl.ji[p]
        Xa, ea, Xb, eb = self._pair_spaces(p, Xpno)
        for r in {p, q}:
            self.Xpno[r] = Xpno
            self.Xa[r], self.ea[r], self.Xb[r], self.eb[r] = Xa, ea, Xb, eb

    # ------------------------------------------------------------------
    # overlaps between pair spaces (dense alpha + beta)
    # ------------------------------------------------------------------
    def S(self, p, q):
        key = (p, q)
        s = self._S.get(key)
        if s is None:
            Sab = self.orb.S_pao[np.ix_(self.pao[p], self.pao[q])]
            sa = self.Xa[p].T @ Sab @ self.Xa[q]
            npp, nq = len(self.pao[p]), len(self.pao[q])
            sb = self.Xb[p][:npp].T @ Sab @ self.Xb[q][:nq] + self.Xb[p][npp:].T @ self.Xb[q][nq:]
            s = scipy.linalg.block_diag(sa, sb)
            self._S[key] = s
        return s

    def nva(self, p):
        return self.Xa[p].shape[1]

    def nvb(self, p):
        return self.Xb[p].shape[1]

    def nv(self, p):
        return self.nva(p) + self.nvb(p)

    # ------------------------------------------------------------------
    # spin orbitals and spin-orbital pairs
    # ------------------------------------------------------------------
    def _so_pairs(self):
        """ordered spin-orbital pairs (I, J), I != J, on valid spatial pairs."""
        pl = self.pl
        self.sop = [(I, J) for I, (x, s) in enumerate(self.so)
                    for J, (y, t) in enumerate(self.so) if I != J and pl.ij[x, y] >= 0]
        self.sop_of = {}                       # spatial pair -> its spin-orbital pairs
        for (I, J) in self.sop:
            p = pl.ij[self.so[I][0], self.so[J][0]]
            self.sop_of.setdefault(p, []).append((I, J))
        # spin orbitals of each pair's LMO domain
        self.dom_so = []
        for p in range(len(pl)):
            self.dom_so.append([self.so_index[(m, s)] for m in self.lmos[p] for s in (0, 1)
                                if (m, s) in self.so_index])

    def _vslice(self, p, s):
        na = self.nva(p)
        return slice(0, na) if s == 0 else slice(na, na + self.nvb(p))

    def _dense(self, p, s, t, blk):
        """dense antisymmetric amplitude of spins (s, t) from its (s, t) block."""
        v = self.nv(p)
        out = np.zeros((v, v))
        a, b = self._vslice(p, s), self._vslice(p, t)
        out[a, b] = blk
        if s != t:
            out[b, a] = -blk.T
        return out

    def _evec(self, p):
        return np.concatenate([self.ea[p], self.eb[p]])

    def _spin_mask(self, p, s, t):
        v = self.nv(p)
        m = np.zeros((v, v), bool)
        a, b = self._vslice(p, s), self._vslice(p, t)
        m[a, b] = True
        m[b, a] = True
        return m

    def _fdiag(self, I):
        x, s = self.so[I]
        return (self.Fa_lmo if s == 0 else self.Fb_lmo)[x, x]

    def _fock_oo(self, K, J):
        (k, s), (j, t) = self.so[K], self.so[J]
        if s != t:
            return 0.
        return (self.Fa_lmo if s == 0 else self.Fb_lmo)[k, j]

    # ------------------------------------------------------------------
    # half-fitted DF integrals on a pair's orbital space
    # ------------------------------------------------------------------
    def _halfL(self, p):
        key = tuple(self.ribfs[p])
        L = self._Lcache.get(key)
        if L is None:
            A = self.J2c[np.ix_(self.ribfs[p], self.ribfs[p])]
            L = self._Lcache[key] = np.linalg.cholesky(A)
        return L

    def _half(self, p, x):
        L = self._halfL(p)
        s = x.shape
        return scipy.linalg.solve_triangular(L, x.reshape(L.shape[0], -1), lower=True).reshape(s)

    def _vir_coef(self, q, s):
        """virtual functions of pair q, spin s, as (PAO idx, coef) and SOMO coef."""
        npo = len(self.pao[q])
        if s == 0:
            return self.Xa[q], None
        return self.Xb[q][:npo], self.Xb[q][npo:]

    def _B_ov(self, p, occ_so, q):
        """(Q_p | M, A_q) half-fitted, M spin orbitals, A in pair q's dense
        virtual space.  -> (naux_p, len(occ_so), nv_q)."""
        aux = self.ribfs[p]
        out = np.zeros((len(aux), len(occ_so), self.nv(q)))
        pq = self.pao[q]
        for r, M in enumerate(occ_so):
            m, s = self.so[M]
            Xp, Xs = self._vir_coef(q, s)
            v = self.qia[np.ix_(aux, [m], pq)][:, 0] @ Xp
            if Xs is not None and self.ns:
                v = v + self.qij[np.ix_(aux, [m], self.somo_idx)][:, 0] @ Xs
            out[:, r, self._vslice(q, s)] = v
        return self._half(p, out)

    def _B_vv(self, p, q1, q2):
        """(Q_p | A_q1 B_q2) half-fitted, dense spin-block-diagonal."""
        aux = self.ribfs[p]
        out = np.zeros((len(aux), self.nv(q1), self.nv(q2)))
        p1, p2 = self.pao[q1], self.pao[q2]
        so = self.somo_idx
        for s in (0, 1):
            X1, S1 = self._vir_coef(q1, s)
            X2, S2 = self._vir_coef(q2, s)
            blk = lib.einsum('ua,Quv,vb->Qab', X1, self.qab[np.ix_(aux, p1, p2)], X2)
            if s == 1 and self.ns:
                qsu1 = self.qia[np.ix_(aux, so, p1)]            # (Q| s u)
                qsu2 = self.qia[np.ix_(aux, so, p2)]
                blk += lib.einsum('ua,Qsu,sb->Qab', X1, qsu1, S2)
                blk += lib.einsum('sa,Qsv,vb->Qab', S1, qsu2, X2)
                blk += lib.einsum('sa,Qst,tb->Qab', S1, self.qij[np.ix_(aux, so, so)], S2)
            out[:, self._vslice(q1, s), self._vslice(q2, s)] = blk
        return self._half(p, out)

    def _B_oo(self, p, occ_so):
        aux = self.ribfs[p]
        o = len(occ_so)
        out = np.zeros((len(aux), o, o))
        for r, M in enumerate(occ_so):
            for c, N in enumerate(occ_so):
                (m, s), (nn, t) = self.so[M], self.so[N]
                if s == t:
                    out[:, r, c] = self.qij[aux, m, nn]
        return self._half(p, out)

    def _f_ov(self, p, occ_so):
        """spin-orbital Fock between occupied LMOs and pair p's virtuals."""
        out = np.zeros((len(occ_so), self.nv(p)))
        pp = self.pao[p]
        for r, M in enumerate(occ_so):
            m, s = self.so[M]
            Xp, Xs = self._vir_coef(p, s)
            if s == 0:
                v = self.Fa_lp[m, pp] @ Xp
            else:
                v = self.Fb_lp[m, pp] @ Xp
                if self.ns:
                    v = v + self.Fb_lmo[m, self.somo_idx] @ Xs
            out[r, self._vslice(p, s)] = v
        return out

    def _G(self, p, I, J, Bov=None, pos=None):
        """<AB||IJ> undressed in pair p's space, from its own B_ov."""
        if Bov is None:
            Bov, pos = self.ints[p]['Bov'], self.ints[p]['pos']
        bi, bj = Bov[:, pos[I]], Bov[:, pos[J]]
        return bi.T @ bj - bj.T @ bi

    # ------------------------------------------------------------------
    # LMP2 PNOs, iterations, CC PNOs
    # ------------------------------------------------------------------
    def lmp2_pnos(self):
        opt, pl = self.opt, self.pl
        npair = len(pl)
        self.Xpno = [None] * npair
        self.Xa, self.ea, self.Xb, self.eb = ([None] * npair for _ in range(4))
        self.de_pno = np.zeros(npair)
        self._S = {}
        self._Lcache = {}
        sc = {}
        pno = (opt['T_CUT_PNO_MP2'], opt['T_CUT_TRACE_MP2'], opt['T_CUT_ENERGY_MP2'], True)
        for p, (x, y) in enumerate(pl.pairs):
            if x > y:
                continue
            e, Xp, de, amps, extra = self._sc_pair(p, pno)
            self._store_spaces(p, Xp)
            self.de_pno[p] = self.de_pno[pl.ji[p]] = de
            sc[p] = (amps, extra)
        self.de_pno_total = self.de_pno.sum()
        self._so_pairs()
        letter = 'ab'
        self.T2 = {}
        for (I, J) in self.sop:
            x, s = self.so[I]
            y, t = self.so[J]
            p = pl.ij[x, y]
            flip = x > y
            c = pl.ji[p] if flip else p
            amps, (Xa_sc, Xb_sc, U, V, Vb) = sc[c]
            paos = self.pao[p]
            Sd = self.orb.S_pao[np.ix_(paos, paos)]
            npo = len(paos)
            T = {0: Xa_sc.T @ Sd @ self.Xa[p], 1: Xb_sc.T @ Sd @ self.Xb[p][:npo]}
            s1, s2 = (t, s) if flip else (s, t)
            key = letter[s1] + letter[s2]
            if key not in amps:
                continue
            blk = T[s1].T @ amps[key][0] @ T[s2]
            if flip:
                blk = blk.T
            self.T2[(I, J)] = self._dense(p, s, t, blk)

    def _pair_mp2_ints(self):
        """per spatial pair: B_ov on its own (I, J) and the exchange G."""
        pl = self.pl
        self.ints = [None] * len(pl)
        for p, (x, y) in enumerate(pl.pairs):
            if x > y:
                continue
            occ = sorted({I for (I, J) in self.sop_of.get(p, [])}
                         | {J for (I, J) in self.sop_of.get(p, [])})
            Bov = self._B_ov(p, occ, p)
            d = dict(Bov=Bov, pos={I: r for r, I in enumerate(occ)})
            self.ints[p] = self.ints[pl.ji[p]] = d

    def lmp2_iterations(self):
        opt, pl = self.opt, self.pl
        self._pair_mp2_ints()
        G = {k: self._G(pl.ij[self.so[k[0]][0], self.so[k[1]][0]], *k) for k in self.sop}
        canon = [k for k in self.sop if k[0] < k[1]]
        diis = DIIS(opt['DIIS_MAX_VECS'])
        ec, rc = 0.01 * opt['E_CONVERGENCE'], 0.01 * opt['R_CONVERGENCE']
        for k in self.sop:
            self.T2.setdefault(k, np.zeros((self.nv(self._sp(k)),) * 2))
        e_prev = 0.
        for it in range(1, opt['MAXITER'] + 1):
            R = {}
            rms = 0.
            for (I, J) in canon:
                p = self._sp((I, J))
                (x, s), (y, t) = self.so[I], self.so[J]
                e = self._evec(p)
                D = e[:, None] + e[None, :] - self._fdiag(I) - self._fdiag(J)
                r = G[(I, J)] + D * self.T2[(I, J)]
                for K in range(len(self.so)):
                    k = self.so[K][0]
                    ik, kj = pl.ij[x, k], pl.ij[k, y]
                    f = self._fock_oo(K, J)
                    if K != J and K != I and ik >= 0 and abs(f) > opt['F_CUT']:
                        sm = self.S(p, ik)
                        r -= f * (sm @ self.T2[(I, K)] @ sm.T)
                    f = self._fock_oo(K, I)
                    if K != I and K != J and kj >= 0 and abs(f) > opt['F_CUT']:
                        sm = self.S(p, kj)
                        r -= f * (sm @ self.T2[(K, J)] @ sm.T)
                r *= self._spin_mask(p, s, t)
                R[(I, J)] = r
                rms = max(rms, _rms(r))
            for (I, J) in canon:
                p = self._sp((I, J))
                e = self._evec(p)
                D = e[:, None] + e[None, :] - self._fdiag(I) - self._fdiag(J)
                self.T2[(I, J)] = self.T2[(I, J)] - R[(I, J)] / D
            flat = diis.update(np.concatenate([self.T2[k].ravel() for k in canon]),
                               np.concatenate([R[k].ravel() for k in canon]))
            off = 0
            for k in canon:
                n = self.T2[k].size
                self.T2[k] = flat[off:off + n].reshape(self.T2[k].shape)
                self.T2[(k[1], k[0])] = -self.T2[k]
                off += n
            e_sp = self._mp2_pair_energies(G)
            e_cur = e_sp.sum()
            self.log(f'  ULMP2 iter {it:3d} {e_cur:.12f} {e_cur - e_prev:.3e} {rms:.3e}')
            if abs(e_cur - e_prev) < ec and rms < rc:
                break
            e_prev = e_cur
        else:
            raise RuntimeError('LMP2 did not converge')
        self.e_lmp2 = e_cur
        # symmetrised spatial pair energies for the strong/weak test
        e_sym = .5 * (e_sp + e_sp[pl.ji])
        return e_sym

    def _sp(self, k):
        return self.pl.ij[self.so[k[0]][0], self.so[k[1]][0]]

    def _mp2_pair_energies(self, G):
        e = np.zeros(len(self.pl))
        for k in self.sop:
            e[self._sp(k)] += .25 * np.sum(G[k] * self.T2[k])
        return e

    def recompute_pnos(self):
        """CC PNOs from the LMP2 amplitudes of each spatial pair."""
        opt, pl = self.opt, self.pl
        G_old = {k: self._G(self._sp(k), *k) for k in self.sop}
        newX = {}
        dens = {}
        for p, (x, y) in enumerate(pl.pairs):
            if x > y:
                continue
            q = pl.ji[p]
            paos = self.pao[p]
            npo = len(paos)
            Sd = self.orb.S_pao[np.ix_(paos, paos)]
            Xo = self.Xpno[p]                          # old PNOs, PAO coordinates
            npn = Xo.shape[1]
            Wa = Xo.T @ Sd @ self.Xa[p]                # (npno, nva)
            Wb = np.vstack([Xo.T @ Sd @ self.Xb[p][:npo], self.Xb[p][npo:]])  # (npno+ns, nvb)
            W = scipy.linalg.block_diag(Wa, Wb)         # dense -> [alpha PNO | beta PNO, SOMO]
            # blocks of the density in PNO coordinates
            def blk(I, J, s, t):
                T = self.T2.get((I, J))
                if T is None:
                    return np.zeros((npn, npn))
                Tc = W @ T @ W.T
                a = slice(0, npn) if s == 0 else slice(npn, 2 * npn)
                b = slice(0, npn) if t == 0 else slice(npn, 2 * npn)
                return Tc[a, b]
            si = self.so_index
            dx, dy = not self.is_somo[x], not self.is_somo[y]
            A = blk(si[(x, 0)], si[(y, 0)], 0, 0) if x != y else np.zeros((npn, npn))
            Bm = blk(si[(x, 1)], si[(y, 1)], 1, 1) if (dx and dy and x != y) else np.zeros((npn, npn))
            O = blk(si[(x, 0)], si[(y, 1)], 0, 1) if dy else np.zeros((npn, npn))
            Op = blk(si[(y, 0)], si[(x, 1)], 0, 1) if dx else np.zeros((npn, npn))
            D = .5 * (A @ A.T + Bm @ Bm.T) + .5 * (O @ O.T + O.T @ O + Op @ Op.T + Op.T @ Op)
            if x == y:
                D *= .5
            dens[p] = (Xo, D)
            occ, V = np.linalg.eigh(D)
            occ, V = occ[::-1], V[:, ::-1]
            scale = opt['T_CUT_PNO_DIAG_SCALE'] if x == y else 1.
            # pair energy (spin summed, both orders) kept by the first m PNOs
            ks = self.sop_of.get(p, []) + (self.sop_of.get(q, []) if q != p else [])
            Vfull = scipy.linalg.block_diag(V, V, np.eye(self.ns))
            nfull = 2 * npn + self.ns
            terms = []
            for k in ks:
                T = Vfull.T @ (W @ self.T2[k] @ W.T) @ Vfull
                Gk = Vfull.T @ (W @ G_old[k] @ W.T) @ Vfull
                terms.append(T * Gk)
            # (e_xy + e_yx) / 2, or e_xx for a diagonal pair
            fac = .5 if q != p else 1.
            M = fac * .25 * sum(terms) if terms else np.zeros((nfull, nfull))
            keep_s = np.zeros(nfull, bool)
            keep_s[2 * npn:] = True
            e_tot = M.sum()
            ecum = []
            for m in range(npn + 1):
                kk = keep_s.copy()
                kk[:m] = True
                kk[npn:npn + m] = True
                ecum.append(M[np.ix_(kk, kk)].sum())
            ecum = np.array(ecum)
            nfin = self._truncate(occ, ecum, e_tot, scale * opt['T_CUT_PNO'],
                                  opt['T_CUT_TRACE'], opt['T_CUT_ENERGY'], False)
            Xnew = Xo @ V[:, :nfin]
            self.de_pno[p] += e_tot - ecum[nfin]
            if q != p:
                self.de_pno[q] = self.de_pno[p]
            newX[p] = Xnew
        # new spaces and amplitudes projected into them
        self._somo_singles_spaces(dens, newX)
        oldXa, oldXb = list(self.Xa), list(self.Xb)
        for p, Xnew in newX.items():
            self._store_spaces(p, Xnew)
        for k in list(self.T2):
            p = self._sp(k)
            paos = self.pao[p]
            npo = len(paos)
            Sd = self.orb.S_pao[np.ix_(paos, paos)]
            Ra = oldXa[p].T @ Sd @ self.Xa[p]
            Rb = oldXb[p][:npo].T @ Sd @ self.Xb[p][:npo] + oldXb[p][npo:].T @ self.Xb[p][npo:]
            R = scipy.linalg.block_diag(Ra, Rb)
            self.T2[k] = R.T @ self.T2[k] @ R
        self._S = {}
        self.de_pno_total = self.de_pno.sum()

    def _somo_singles_spaces(self, dens, newX):
        """Virtual space of the diagonal pair of a SOMO, which carries only
        that SOMO's singles: the natural orbitals of the averaged pair
        density of every pair the SOMO belongs to, in its PAO domain,
        truncated at T_CUT_PNO * T_CUT_PNO_DIAG_SCALE (the diagonal-pair cut)."""
        opt, pl = self.opt, self.pl
        S = self.orb.S_pao
        for p, (x, y) in enumerate(pl.pairs):
            if x != y or not self.is_somo[x]:
                continue
            paos = self.pao[p]
            X0, _ = su.orthocanonicalize(S[np.ix_(paos, paos)], self.Fa_pao[np.ix_(paos, paos)], opt['S_CUT'])
            acc, cnt = 0., 0
            for q, (a, b) in enumerate(pl.pairs):
                if a > b or q == p or x not in (a, b) or q not in dens:
                    continue
                Xo, D = dens[q]
                s = Xo.T @ S[np.ix_(self.pao[q], paos)] @ X0
                acc = acc + s.T @ D @ s
                cnt += 1
            if cnt == 0:
                newX[p] = X0
                continue
            occ, V = np.linalg.eigh(acc / cnt)
            occ, V = occ[::-1], V[:, ::-1]
            cut = opt['T_CUT_PNO'] * opt['T_CUT_PNO_DIAG_SCALE']
            nfin = max(int(np.count_nonzero(occ >= cut)), min(opt['MIN_PNOS'], len(occ)), 1)
            while nfin < len(occ) and abs(occ[nfin] - occ[nfin - 1]) <= 1e-9 * abs(occ[nfin - 1]):
                nfin += 1
            newX[p] = X0 @ V[:, :nfin]

    # ------------------------------------------------------------------
    # CC integrals on each spatial pair's (occupied domain + virtual) space
    # ------------------------------------------------------------------
    def cc_integrals(self):
        pl = self.pl
        self.cc = [None] * len(pl)
        self._mix = {}
        self._np = {}
        for p, (x, y) in enumerate(pl.pairs):
            if x > y:
                continue
            occ = self.dom_so[p]
            o, v = len(occ), self.nv(p)
            Bov = self._B_ov(p, occ, p)
            Bvv = self._B_vv(p, p, p)
            Boo = self._B_oo(p, occ)
            na = Bov.shape[0]
            B = np.zeros((na, o + v, o + v))
            B[:, :o, :o] = Boo
            B[:, :o, o:] = Bov
            B[:, o:, :o] = Bov.transpose(0, 2, 1)
            B[:, o:, o:] = Bvv
            f = np.zeros((o + v, o + v))
            for r, M in enumerate(occ):
                for c, N in enumerate(occ):
                    f[r, c] = self._fock_oo(M, N)
            fov = self._f_ov(p, occ)
            f[:o, o:] = fov
            f[o:, :o] = fov.T
            f[o:, o:] = np.diag(self._evec(p))
            d = dict(B=B, f=f, o=o, occ=occ, pos={I: r for r, I in enumerate(occ)}, Bov=Bov)
            self.cc[p] = self.cc[pl.ji[p]] = d

    def _mixed(self, p, q):
        """undressed (Q_p | M, C_q) and (Q_p | B_p, C_q) for the ring terms."""
        key = (p, q)
        v = self._mix.get(key)
        if v is None:
            occ = self.dom_so[p]
            v = self._mix[key] = (self._B_ov(p, occ, q), self._B_vv(p, p, q))
        return v

    def _dress(self, p):
        """T1-dressed B and F of pair p, and the projected singles T_n."""
        d = self.cc[p]
        o = d['o']
        v = self.nv(p)
        Tn = np.zeros((o, v))
        for r, K in enumerate(d['occ']):
            kk = self.pl.ij[self.so[K][0], self.so[K][0]]
            Tn[r] = self.S(p, kk) @ self.T1[K]
        tau = np.zeros((o + v, o + v))
        tau[o:, :o] = Tn.T
        Lm, Rm = np.eye(o + v) - tau, np.eye(o + v) + tau
        B = d['B']
        Bt = lib.einsum('pr,Qrs,sq->Qpq', Lm, B, Rm)
        gam = np.einsum('Qkc,kc->Q', B[:, :o, o:], Tn)
        Dl = np.einsum('Q,Qpq->pq', gam, B) - lib.einsum('Qpc,kc,Qkq->pq', B[:, :, o:], Tn, B[:, :o, :])
        Ft = Lm @ (d['f'] + Dl) @ Rm
        return dict(Bt=Bt, Ft=Ft, Tn=Tn, tau=tau, FD=d['f'] + Dl)

    def lccsd_iterations(self):
        opt, pl = self.opt, self.pl
        self.cc_integrals()
        nso = len(self.so)
        self.T1 = [np.zeros(self.nv(pl.ij[x, x])) for (x, s) in self.so]
        strong_sp = self.strong
        canon = [k for k in self.sop if k[0] < k[1]]
        strong = [k for k in canon if strong_sp[self._sp(k)]]
        Gu = {k: self._G_cc(k) for k in self.sop}
        diis = DIIS(opt['DIIS_MAX_VECS'])
        e_prev = 0.
        for it in range(1, opt['MAXITER'] + 1):
            dr = {}
            self._Z = {}
            for p, (x, y) in enumerate(pl.pairs):
                if x <= y:
                    dr[p] = dr[pl.ji[p]] = self._dress(p)
            R1 = self._R1(dr)
            R2 = self._R2(dr, Gu, strong)
            r1 = max(_rms(r) for r in R1)
            r2 = max((_rms(R2[k]) for k in strong), default=0.)
            for I in range(nso):
                x, s = self.so[I]
                dd = pl.ij[x, x]
                e = self._evec(dd) - self._fdiag(I)
                e = np.where(np.abs(e) < 1e-12, 1., e)
                self.T1[I] = self.T1[I] - R1[I] / e
            for k in strong:
                p = self._sp(k)
                e = self._evec(p)
                D = e[:, None] + e[None, :] - self._fdiag(k[0]) - self._fdiag(k[1])
                self.T2[k] = self.T2[k] - R2[k] / D
            flat = diis.update(np.concatenate([t.ravel() for t in self.T1] + [self.T2[k].ravel() for k in strong]),
                               np.concatenate([r.ravel() for r in R1] + [R2[k].ravel() for k in strong]))
            off = 0
            for I in range(nso):
                n = self.T1[I].size
                self.T1[I] = flat[off:off + n]
                off += n
            for k in strong:
                n = self.T2[k].size
                self.T2[k] = flat[off:off + n].reshape(self.T2[k].shape)
                self.T2[(k[1], k[0])] = -self.T2[k]
                off += n
            e_cur, e_weak = self.cc_energy(Gu)
            self.log(f'  ULCCSD iter {it:3d} {e_cur:.12f} {e_cur - e_prev:.3e} {r1:.3e} {r2:.3e}')
            if abs(e_cur - e_prev) < opt['E_CONVERGENCE'] and max(r1, r2) < opt['R_CONVERGENCE']:
                break
            e_prev = e_cur
        else:
            raise RuntimeError('LCCSD did not converge')
        self.e_lccsd = e_cur - e_weak
        self.de_weak = e_weak

    def _G_cc(self, k):
        p = self._sp(k)
        d = self.cc[p]
        return self._G(p, k[0], k[1], d['Bov'], d['pos'])

    def cc_energy(self, Gu):
        pl = self.pl
        e = ew = 0.
        for I, (x, s) in enumerate(self.so):
            dd = pl.ij[x, x]
            d = self.cc[dd]
            e += self.T1[I] @ d['f'][d['pos'][I], d['o']:]
        for k in self.sop:
            if k[0] > k[1]:
                continue
            I, J = k
            p = self._sp(k)
            ti = self.S(p, pl.ij[self.so[I][0], self.so[I][0]]) @ self.T1[I]
            tj = self.S(p, pl.ij[self.so[J][0], self.so[J][0]]) @ self.T1[J]
            tau = self.T2[k] + np.outer(ti, tj) - np.outer(tj, ti)
            eij = .5 * np.sum(Gu[k] * tau)
            e += eij
            if not self.strong[p]:
                ew += eij
        return e, ew

    def _Fp_oo(self, dr, Gu):
        """F'_KJ = F~_KJ + 1/2 sum_L <KL||CD> t_JL^CD (in the (k, l) pair spaces)."""
        pl = self.pl
        out = {}
        for p, (k, j) in enumerate(pl.pairs):
            d = self.cc[p]
            Ft = dr[p]['Ft']
            for K in d['occ']:
                if self.so[K][0] != k:
                    continue
                for J in d['occ']:
                    if self.so[J][0] != j or self.so[K][1] != self.so[J][1]:
                        continue
                    val = Ft[d['pos'][K], d['pos'][J]]
                    for l in self.lmos[p]:
                        for L in (self.so_index.get((l, 0)), self.so_index.get((l, 1))):
                            if L is None or L == K or L == J:
                                continue
                            kl, jl = pl.ij[k, l], pl.ij[j, l]
                            sm = self.S(kl, jl)
                            val += .5 * np.sum(Gu[(K, L)] * (sm @ self.T2[(J, L)] @ sm.T))
                    out[(K, J)] = val
        return out

    def _R2(self, dr, Gu, strong):
        pl = self.pl
        Fp = self._Fp_oo(dr, Gu)
        R = {}
        for (I, J) in strong:
            p = self._sp((I, J))
            x, y = self.so[I][0], self.so[J][0]
            d = self.cc[p]
            dp = dr[p]
            o = d['o']
            Bt = dp['Bt']
            pi, pj = d['pos'][I], d['pos'][J]
            t = self.T2[(I, J)]
            # <AB||IJ>~
            bi, bj = Bt[:, o:, pi], Bt[:, o:, pj]
            r = bi.T @ bj - bj.T @ bi
            # 4-external
            Bvv = Bt[:, o:, o:]
            r += lib.einsum('Qac,cd,Qbd->ab', Bvv, t, Bvv)
            # F'_vv
            Fv = dp['Ft'][o:, o:].copy()
            for K in d['occ']:
                for L in d['occ']:
                    if K == L:
                        continue
                    kl = pl.ij[self.so[K][0], self.so[L][0]]
                    if kl < 0:
                        continue
                    sm = self.S(kl, p)
                    Fv -= .5 * sm.T @ (self.T2[(K, L)] @ Gu[(K, L)].T) @ sm
            r += t @ Fv.T + Fv @ t
            # occupied Fock terms
            for K in range(len(self.so)):
                k = self.so[K][0]
                ik, kj = pl.ij[x, k], pl.ij[k, y]
                if ik >= 0 and K != I and (K, J) in Fp:
                    sm = self.S(p, ik)
                    r -= Fp[(K, J)] * (sm @ self.T2[(I, K)] @ sm.T)
                if kj >= 0 and K != J and (K, I) in Fp:
                    sm = self.S(p, kj)
                    r -= Fp[(K, I)] * (sm @ self.T2[(K, J)] @ sm.T)
            # oooo ladder
            Bo = Bt[:, :o, :o]
            Bc = d['Bov']
            for K in d['occ']:
                for L in d['occ']:
                    if K == L:
                        continue
                    kl = pl.ij[self.so[K][0], self.so[L][0]]
                    if kl < 0:
                        continue
                    pk, pl_ = d['pos'][K], d['pos'][L]
                    W = (Bo[:, pk, pi] @ Bo[:, pl_, pj] - Bo[:, pk, pj] @ Bo[:, pl_, pi])
                    Gc = Bc[:, pk].T @ Bc[:, pl_] - Bc[:, pl_].T @ Bc[:, pk]
                    W += .5 * np.sum(Gc * t)
                    sm = self.S(kl, p)
                    r += .5 * W * (sm.T @ self.T2[(K, L)] @ sm)
            # ring
            XIJ = self._ring_X(dr, Gu, p, I, J)
            XJI = self._ring_X(dr, Gu, p, J, I)
            r += XIJ - XJI - XIJ.T + XJI.T
            r *= self._spin_mask(p, self.so[I][1], self.so[J][1])
            R[(I, J)] = r
        return R

    def _ring_np(self, p, K, J, q):
        """non-projected, undressed <K B_p || C_q J> = (K C|B J) - (K J|B C),
        B in pair p's space, C in pair q's, fitted in p's domain (fixed)."""
        key = (p, K, J, q)
        v = self._np.get(key)
        if v is None:
            d = self.cc[p]
            o = d['o']
            Bov_m, Bvv_m = self._mixed(p, q)
            pk, pj = d['pos'][K], d['pos'][J]
            B = d['B']
            v = (np.einsum('Qc,Qb->bc', Bov_m[:, pk], B[:, o:, pj])
                 - np.einsum('Q,Qbc->bc', B[:, pk, pj], Bvv_m))
            self._np[key] = v
        return v

    def _ring_Z(self, dr, Gu, J, K):
        """Z_JK[B,C] in the space of the pair (j, k): the T1-dressing part of
        <KB||CJ>~ plus 1/2 sum_{L,D} <KL||CD> t_JL^{BD}."""
        key = (J, K)
        if key in self._Z:
            return self._Z[key]
        pl = self.pl
        y, k = self.so[J][0], self.so[K][0]
        q = pl.ij[y, k]
        d, dq = self.cc[q], dr[q]
        o = d['o']
        B, Bt = d['B'], dq['Bt']
        pk, pj = d['pos'][K], d['pos'][J]
        Z = (np.einsum('Qc,Qb->bc', B[:, pk, o:], Bt[:, o:, pj] - B[:, o:, pj])
             - (np.einsum('Q,Qbc->bc', Bt[:, pk, pj], Bt[:, o:, o:])
                - np.einsum('Q,Qbc->bc', B[:, pk, pj], B[:, o:, o:])))
        for l in self.lmos[q]:
            for L in (self.so_index.get((l, 0)), self.so_index.get((l, 1))):
                if L is None or L == J or L == K:
                    continue
                jl, kl = pl.ij[y, l], pl.ij[k, l]
                Z += .5 * self.S(q, jl) @ self.T2[(J, L)] @ self.S(jl, kl) @ Gu[(K, L)].T @ self.S(kl, q)
        self._Z[key] = Z
        return Z

    def _ring_X(self, dr, Gu, p, I, J):
        """X_IJ[A,B] = sum_{K,C} W_KBCJ t_IK^{AC} in pair p's space, with
        W = <KB||CJ> (non-projected, fixed) + S(p,jk) Z_JK S(jk,ik)."""
        pl = self.pl
        d = self.cc[p]
        x, y = self.so[I][0], self.so[J][0]
        X = np.zeros((self.nv(p), self.nv(p)))
        for K in d['occ']:
            if K == I:
                continue
            k = self.so[K][0]
            ik, jk = pl.ij[x, k], pl.ij[y, k]
            W = self._ring_np(p, K, J, ik) + self.S(p, jk) @ self._ring_Z(dr, Gu, J, K) @ self.S(jk, ik)
            X += self.S(p, ik) @ self.T2[(I, K)] @ W.T
        return X

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
                dk, dpk = self.cc[ik], dr[ik]
                ok = dk['o']
                pk = dk['pos'][K]
                t = self.T2[(I, K)]
                sm = self.S(dd, ik)
                # F~_KC t_IK^{AC}
                r += sm @ (t @ dpk['FD'][pk, ok:])
                # 1/2 <AK||CD>~ t_IK^CD = sum (AC|KD)~ t^CD
                Bvv = dpk['Bt'][:, ok:, ok:]
                Bk = dk['B'][:, pk, ok:]
                r += sm @ lib.einsum('Qac,cd,Qd->a', Bvv, t, Bk)
            # -1/2 <KL||IC>~ t_KL^AC = - sum (KI|LC)~ t_KL^AC
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

    # ------------------------------------------------------------------
    # (T)
    # ------------------------------------------------------------------
    def _pair_density_pno(self, p):
        """spin-averaged pair density of spatial pair p in its PNO
        coordinates (the PNO density formula with the current amplitudes)."""
        x, y = self.pl.pairs[p]
        paos = self.pao[p]
        npo = len(paos)
        Sd = self.orb.S_pao[np.ix_(paos, paos)]
        Xo = self.Xpno[p]
        npn = Xo.shape[1]
        Wa = Xo.T @ Sd @ self.Xa[p]
        Wb = np.vstack([Xo.T @ Sd @ self.Xb[p][:npo], self.Xb[p][npo:]])
        W = scipy.linalg.block_diag(Wa, Wb)
        si = self.so_index

        def blk(I, J, s, t):
            T = self.T2.get((I, J))
            if T is None:
                return np.zeros((npn, npn))
            Tc = W @ T @ W.T
            a = slice(0, npn) if s == 0 else slice(npn, 2 * npn)
            b = slice(0, npn) if t == 0 else slice(npn, 2 * npn)
            return Tc[a, b]
        dx, dy = not self.is_somo[x], not self.is_somo[y]
        A = blk(si[(x, 0)], si[(y, 0)], 0, 0) if x != y else np.zeros((npn, npn))
        Bm = blk(si[(x, 1)], si[(y, 1)], 1, 1) if (dx and dy and x != y) else np.zeros((npn, npn))
        O = blk(si[(x, 0)], si[(y, 1)], 0, 1) if dy else np.zeros((npn, npn))
        Op = blk(si[(y, 0)], si[(x, 1)], 0, 1) if dx else np.zeros((npn, npn))
        D = .5 * (A @ A.T + Bm @ Bm.T) + .5 * (O @ O.T + O.T @ O + Op @ Op.T + Op.T @ Op)
        if x == y:
            D *= .5
        return D

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
            keep = [t for t, e in zip(self.trip, self.e_ijk) if abs(e) >= opt['T_CUT_TRIPLES_WEAK']]
            self.de_t_screened = sum(e for e in self.e_ijk if abs(e) < opt['T_CUT_TRIPLES_WEAK'])
            self.trip = keep
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
        # spin-orbital triples I < J < K of every spatial triplet
        si = self.so_index
        self.sot = []
        for t, (i, j, k) in enumerate(self.trip):
            seen = set()
            for s1 in (0, 1):
                for s2 in (0, 1):
                    for s3 in (0, 1):
                        ids = [si.get((i, s1)), si.get((j, s2)), si.get((k, s3))]
                        if None in ids or len(set(ids)) < 3:
                            continue
                        key = tuple(sorted(ids))
                        if key not in seen:
                            seen.add(key)
                            self.sot.append((t, key))

    def tno_transform(self, t_cut):
        """TNOs per spatial triplet; stored as extra 'pairs' after the real
        ones so that S(), _B_ov() etc. serve triplets too."""
        opt, pl = self.opt, self.pl
        npair = len(pl)
        del self.Xa[npair:], self.Xb[npair:], self.ea[npair:], self.eb[npair:]
        del self.pao[npair:], self.ribfs[npair:], self.Xpno[npair:]
        self._S = {k: v for k, v in self._S.items() if k[0] < npair and k[1] < npair}
        self._mix = {}
        self.tp = []
        for t, (i, j, k) in enumerate(self.trip):
            paos = self.t_paos[t]
            Sd = self.orb.S_pao[np.ix_(paos, paos)]
            Xo, _ = su.orthocanonicalize(Sd, self.Fa_pao[np.ix_(paos, paos)], opt['S_CUT'])
            D = 0.
            for a, b in ((i, j), (j, k), (i, k)):
                p = pl.ij[a, b]
                s = self.Xpno[p].T @ self.orb.S_pao[np.ix_(self.pao[p], paos)] @ Xo
                D = D + s.T @ self._pair_density_pno(p) @ s
            D /= 3.
            occ, U = np.linalg.eigh(D)
            occ, U = occ[::-1], U[:, ::-1]
            nfin = sum(1 for a in range(len(occ))
                       if abs(occ[a]) >= self.tno_scale[t] * t_cut or a < opt['MIN_TNOS'])
            nfin = max(1, nfin)
            Xt = Xo @ U[:, :nfin]
            q = len(self.pao)
            self.pao.append(paos)
            self.ribfs.append(self.t_ribfs[t])
            self.Xpno.append(Xt)
            Xa, ea, Xb, eb = self._pair_spaces(q, Xt)
            self.Xa.append(Xa)
            self.ea.append(ea)
            self.Xb.append(Xb)
            self.eb.append(eb)
            self.tp.append(q)

    def _t_spaces_ints(self, t, I, J, K):
        """DF integrals in triplet t's space for the spin orbitals of the
        triplet domain (incl. I, J, K)."""
        q = self.tp[t]
        lm = self.t_lmos[t]
        occ = sorted({self.so_index[(m, s)] for m in lm for s in (0, 1) if (m, s) in self.so_index}
                     | {I, J, K})
        pos = {M: r for r, M in enumerate(occ)}
        return q, occ, pos, self._B_ov(q, occ, q), self._B_vv(q, q, q), self._B_oo(q, occ)

    def _w_ordered(self, t, q, occ, pos, Bov, Bvv, Boo, I, J, K):
        """w_ijk[a,b,c] = sum_e t_jk^ae <ei||bc> - sum_m t_im^bc <ma||jk>."""
        pl = self.pl
        si = lambda M: self.so[M][0]
        sjk = self.S(q, pl.ij[si(J), si(K)])
        tjk = sjk @ self.T2[(J, K)] @ sjk.T
        bi = Bov[:, pos[I]]
        # <ei||bc> = (eb|ic) - (ec|ib)
        g = lib.einsum('Qeb,Qc->ebc', Bvv, bi)
        g = g - g.transpose(0, 2, 1)
        w = lib.einsum('ae,ebc->abc', tjk, g)
        bj, bk = Bov[:, pos[J]], Bov[:, pos[K]]
        for M in occ:
            if M == I:
                continue
            im = pl.ij[si(I), si(M)]
            if im < 0:
                continue
            s = self.S(q, im)
            tim = s @ self.T2[(I, M)] @ s.T
            # <ma||jk> = (mj|ak) - (mk|aj)
            v = Boo[:, pos[M], pos[J]] @ bk - Boo[:, pos[M], pos[K]] @ bj      # (a,)
            w -= tim[None, :, :] * v[:, None, None]
        return w

    def _wv(self, t, I, J, K):
        q, occ, pos, Bov, Bvv, Boo = self._t_spaces_ints(t, I, J, K)
        pl = self.pl
        si = lambda M: self.so[M][0]
        ords = ((I, J, K, 1.), (J, I, K, -1.), (K, J, I, -1.))
        Y = 0.
        Yv = 0.
        fov = self._f_ov(q, [I, J, K])
        for (a, b, c, sg) in ords:
            Y = Y + sg * self._w_ordered(t, q, occ, pos, Bov, Bvv, Boo, a, b, c)
            # disconnected: t_a^A <bc||BC> + f_aA t_bc^BC
            ta = self.S(q, pl.ij[si(a), si(a)]) @ self.T1[a]
            gbc = Bov[:, pos[b]].T @ Bov[:, pos[c]]
            gbc = gbc - gbc.T
            sbc = self.S(q, pl.ij[si(b), si(c)])
            tbc = sbc @ self.T2[(b, c)] @ sbc.T
            fa = fov[[I, J, K].index(a)]
            Yv = Yv + sg * (ta[:, None, None] * gbc[None] + fa[:, None, None] * tbc[None])
        W = Y - Y.transpose(1, 0, 2) - Y.transpose(2, 1, 0)
        V = Yv - Yv.transpose(1, 0, 2) - Yv.transpose(2, 1, 0)
        e = self._evec(q)
        D = (self._fdiag(I) + self._fdiag(J) + self._fdiag(K)
             - e[:, None, None] - e[None, :, None] - e[None, None, :])
        # spin structure: zero outside the allowed virtual spin blocks
        sl = [self._vslice(q, self.so[M][1]) for M in (I, J, K)]
        msk = np.zeros(W.shape, bool)
        import itertools
        for perm in itertools.permutations(range(3)):
            msk[sl[perm[0]], sl[perm[1]], sl[perm[2]]] = True
        D = np.where(msk, D, 1.)
        return W * msk, V * msk, D

    def lccsd_t0(self, save=False):
        E = 0.
        self.e_ijk = np.zeros(len(self.trip))
        if save:
            self.W3, self.V3, self.T3, self.D3 = {}, {}, {}, {}
        for (t, key) in self.sot:
            W, V, D = self._wv(t, *key)
            T3 = W / D
            e = np.sum(T3 * (W + V)) / 6.
            self.e_ijk[t] += e
            E += e
            if save:
                self.W3[key], self.V3[key], self.T3[key], self.D3[key] = W, V, T3, D
        self._sot_trip = {key: t for t, key in self.sot}
        return E

    def sort_triplets(self, e_total):
        opt = self.opt
        order = np.argsort(-np.abs(self.e_ijk), kind='stable')
        self.tno_scale = np.full(len(self.trip), opt['T_CUT_TNO_WEAK_SCALE'])
        e = 0.
        for t in order:
            self.tno_scale[t] = opt['T_CUT_TNO_STRONG_SCALE']
            e += self.e_ijk[t]
            if e / e_total > 0.9:
                break

    def lccsd_t_iterations(self):
        """(T) with the occupied Fock couplings (Jiang Eq. 111), spin orbital."""
        import itertools
        opt = self.opt
        nso = len(self.so)
        keys = [key for (t, key) in self.sot]
        keyset = set(keys)
        e_old = {k: 0. for k in keys}
        e_cur = {k: np.sum(self.T3[k] * (self.W3[k] + self.V3[k])) / 6. for k in keys}
        e_prev = 0.

        def t3_of(L, J, K, q):
            """t_{LJK} in triplet space q (permuted and projected)."""
            ids = (L, J, K)
            key = tuple(sorted(ids))
            if key not in keyset:
                return None
            perm = [key.index(x) for x in ids]       # position of each index in the key
            sign = np.linalg.det(np.eye(3)[perm])
            # antisymmetry in the occupied indices alone (virtuals stay put)
            T = self.T3[key]
            qk = self.tp[self._sot_trip[key]]
            s = self.S(q, qk)
            T = lib.einsum('aA,bB,cC,ABC->abc', s, s, s, T)
            return sign * T
        for it in range(1, opt['MAXITER'] + 1):
            rms = 0.
            for key in keys:
                if abs(e_cur[key] - e_old[key]) < abs(e_old[key] * opt['T_CUT_ITER']):
                    continue
                I, J, K = key
                t = self._sot_trip[key]
                q = self.tp[t]
                D = self.D3[key]                     # f_ii+f_jj+f_kk - e_a-e_b-e_c
                R = self.W3[key] - D * self.T3[key]
                for L in range(nso):
                    for (pos_, other) in ((0, (J, K)), (1, (I, K)), (2, (I, J))):
                        X = (I, J, K)[pos_]
                        if L == X or L in key:
                            continue
                        f = self._fock_oo(L, X)
                        if abs(f) < opt['F_CUT_T']:
                            continue
                        ids = list(key)
                        ids[pos_] = L
                        Tl = t3_of(*ids, q)
                        if Tl is None:
                            continue
                        R -= f * Tl
                msk = D != 1.
                R = R * msk
                self.T3[key] = self.T3[key] + R / D * msk
                rms = max(rms, _rms(R))
            e_old = dict(e_cur)
            e_cur = {k: np.sum(self.T3[k] * (self.W3[k] + self.V3[k])) / 6. for k in keys}
            E = sum(e_cur.values())
            self.log(f'  U(T) iter {it:3d} {E:.12f} {E - e_prev:.3e} {rms:.3e}')
            if rms < opt['R_CONVERGENCE'] and abs(E - e_prev) < opt['E_CONVERGENCE']:
                break
            e_prev = E
        else:
            raise RuntimeError('(T) did not converge')
        self.e_ijk = np.zeros(len(self.trip))
        for k, e in e_cur.items():
            self.e_ijk[self._sot_trip[k]] += e
        return E

    # ------------------------------------------------------------------
    def kernel(self):
        opt = self.opt
        self.setup()
        n = self.n
        mask = ((self.doi_ij > opt['T_CUT_DO_IJ']) | (np.abs(self.dip_b) > opt['T_CUT_PRE'])
                | np.eye(n, dtype=bool))
        self.de_dipole = float(np.sum(self.dip_e[~mask]))
        t = time.perf_counter()
        self.set_pairs(mask, self.domains(opt['T_CUT_MKN'] * 100, opt['T_CUT_DO'] * 2))
        e_crude = self.sc_mp2_pair_energies()
        t = self._tick('crude_prescreen', t)
        keep = np.zeros((n, n), bool)
        self.de_lmp2_eliminated = 0.
        for p, (i, j) in enumerate(self.pl.pairs):
            if abs(e_crude[p]) >= opt['T_CUT_PAIRS_MP2'] or i == j:
                keep[i, j] = True
            else:
                self.de_lmp2_eliminated += e_crude[p]
        self.set_pairs(keep, self.domains(opt['T_CUT_MKN'], opt['T_CUT_DO']))
        self.lmp2_pnos()
        t = self._tick('lmp2_pnos', t)
        e_ij = self.lmp2_iterations()
        t = self._tick('lmp2_iter', t)
        pl = self.pl
        self.strong = np.array([abs(e_ij[p]) >= opt['T_CUT_PAIRS'] or i == j
                                for p, (i, j) in enumerate(pl.pairs)])
        self.e['lmp2'] = self.e_lmp2 + self.de_lmp2_eliminated + self.de_dipole + self.de_pno_total
        self.recompute_pnos()
        t = self._tick('recompute_pnos', t)
        self.lccsd_iterations()
        t = self._tick('lccsd', t)
        e_corr = (self.e_lccsd + self.de_weak + self.de_lmp2_eliminated
                  + self.de_dipole + self.de_pno_total)
        self.e.update(ccsd=e_corr, lccsd=self.e_lccsd, weak=self.de_weak,
                      sc_lmp2=self.de_lmp2_eliminated, dipole=self.de_dipole,
                      pno_trunc=self.de_pno_total, ref=self.e_ref)
        if self.triples:
            self.triples_sparsity(True)
            self.tno_transform(opt['T_CUT_TNO_PRE'])
            self.log(f'  (T0) prescreening over {len(self.trip)} triplets')
            self.lccsd_t0()
            t = self._tick('t0_prescreen', t)
            self.triples_sparsity(False)
            self.log(f'  (T0) over {len(self.trip)} triplets')
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
        self.e['tot'] = self.e_ref + e_corr
        return self.e['tot']
