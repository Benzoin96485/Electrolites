"""High-spin open-shell DLPNO-CCSD(T) on the GPU.

The algorithm is ``ref_uhf2``'s (ROHF or QRO reference, one set of PNOs per
spatial pair, alpha virtuals = PNOs, beta virtuals = PNOs + SOMOs,
spin-orbital T1-dressed CCSD with Jiang et al.'s projection placement);
this class moves the expensive steps to the device:

* three-index integrals: ``dfints.DFInts`` (as the closed-shell driver);
* semicanonical pair energies, LMP2 PNOs and the PNO spaces: batched over
  pairs of equal padded PAO-domain size (one Cholesky / ``eigh_batched``
  call per chunk for the fitting metric, PAO orthogonalisation, alpha and
  beta PAO canonicalisation, PNO density and the PNO canonicalisations);
* the PNO-LMP2 iterations: spin-blocked batched plans;
* the CC integrals of every pair, computed directly in the blocks the
  CCSD needs (no dense spin-orbital DF tensor is formed);
* the LCCSD iterations: ``uccsd_gpu.UCCSDGPU``;
* the triples: ``utriples_gpu`` when available, else the host reference
  with device integrals.

Amplitudes are kept by spin block: ``T2b[(I, J)]`` is the (spin I, spin J)
block of the spin-orbital pair amplitude (``T2b[(J, I)] = T2b[(I, J)].T``).
"""
import time

import numpy as np
import scipy.linalg
import cupy as cp
from electrolites.dlpno.gather import blocks as gather_blocks, xtbx
from electrolites.dlpno.linalg import chol_inv_batched
import cupyx.scipy.linalg as cpla

from pyscf import df, scf

from electrolites.dlpno import setup as su
from electrolites.dlpno.ref_uhf import qro
from electrolites.dlpno.ref_uhf2 import RefUDLPNO2
from electrolites.dlpno.dfints import DFInts, ExtDomains
from electrolites.dlpno.ragged import Layout, Ragged, padded, op
from electrolites.dlpno.linalg import eigh_batched, orthonormal_domain
from electrolites.dlpno.ccsd_gpu import _DeviceDIIS
from electrolites.dlpno.utriples_gpu import UTriplesGPU
from electrolites.dlpno.uccsd_gpu import _SpinPlan, _Terms, _Reg, _set_items, _get
from electrolites.dlpno.uccsd_vec import UCCSDGPUv as UCCSDGPU


def _half(L, x):
    s = x.shape
    return cpla.solve_triangular(L, x.reshape(L.shape[0], -1), lower=True).reshape(s)


class UDLPNO(UTriplesGPU, RefUDLPNO2):
    """GPU open-shell DLPNO-CCSD(T); same interface as RefUDLPNO."""

    def __init__(self, mf, *args, **opts):
        # open-shell singles converge slowly near (quasi)degeneracies of the
        # SOMOs: a longer DIIS history and more iterations by default
        opts.setdefault('MAXITER', 100)
        opts.setdefault('DIIS_MAX_VECS', 8)
        super().__init__(mf, *args, **opts)

    def kernel(self):
        from electrolites.dlpno.devmem import pooled
        from electrolites.dlpno import mgpu
        try:
            with pooled():
                return super().kernel()
        finally:
            mgpu.close_pools()

    MAX_PAIR_CHUNK_BYTES = 1 << 30
    PAO_ORTHO = 'pivchol'

    # ------------------------------------------------------------------
    def setup(self):
        t = time.perf_counter()
        mf, mol, opt = self.mf, self.mol, self.opt
        S = su.asnumpy(mf.get_ovlp())
        if self.reference == 'qro':
            C, nd, ns = qro(mf)
        else:
            C = su.asnumpy(mf.mo_coeff)
            occ = su.asnumpy(mf.mo_occ)
            if occ.ndim == 2:
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
        self.ncore = nfc
        c_d = C[:, nfc:nd]
        c_s = C[:, nd:nd + ns]
        L_d = su.localize(mol, c_d, self.localizer, opt['LOCAL_CONVERGENCE'], mf)
        L_s = su.localize(mol, c_s, self.localizer, opt['LOCAL_CONVERGENCE'], mf) if ns > 1 else c_s
        L = np.hstack([L_d, L_s])
        self.nda = L_d.shape[1]
        self.n = n = L.shape[1]
        self.is_somo = np.arange(n) >= self.nda
        orb = su.Orbitals(mol, S, .5 * (Fa + Fb), C, nd + ns, nfc, c_lmo=L)
        self.orb = orb
        P = orb.C_pao
        self.Fa_lmo, self.Fb_lmo = L.T @ Fa @ L, L.T @ Fb @ L
        self.Fa_pao, self.Fb_pao = P.T @ Fa @ P, P.T @ Fb @ P
        self.Fa_lp, self.Fb_lp = L.T @ Fa @ P, L.T @ Fb @ P
        Ls = L[:, self.nda:]
        self.Fb_ps = P.T @ Fb @ Ls
        self.Fb_ss = Ls.T @ Fb @ Ls
        self.somo_idx = np.arange(self.nda, n)
        auxbasis = self.auxbasis or df.make_auxbasis(mol, mp2fit=True)
        self.auxmol = auxmol = df.addons.make_auxmol(mol, auxbasis)
        (self.bf_atom, self.ribf_atom, self.atom_bfs,
         self.atom_ribfs) = su.basis_maps(mol, auxmol)
        t = self._tick('orbitals', t)
        self.doi_ij, self.doi_iu, self.doi_uv = su.compute_doi(mol, L, P, opt['DOI_GRID'])
        t = self._tick('doi', t)
        self.dip_e, self.dip_b = su.dipole_pair_energies(
            orb, self.doi_iu, opt['T_CUT_DO_PRE'], opt['S_CUT'], self.bf_atom, self.atom_bfs)
        self.mkn = su.mulliken_atom_pops(mol, S, L)
        t = self._tick('dipole', t)
        mask = ((self.doi_ij > opt['T_CUT_DO_IJ']) | (np.abs(self.dip_b) > opt['T_CUT_PRE'])
                | np.eye(n, dtype=bool))
        dom = self.domains(min(opt['T_CUT_MKN'], opt['T_CUT_MKN_TRIPLES']),
                           min(opt['T_CUT_DO'], opt['T_CUT_DO_TRIPLES']))
        pairs = [(i, j) for i in range(n) for j in range(n) if mask[i, j]]
        self.ext = ExtDomains(mol.natm, dom.lmo_riatoms, dom.lmo_paos, pairs, orb.nbf)
        self.dfi = DFInts(mol, auxmol, L, P, self.ext, self.doi_uv, opt['T_CUT_DO_UV'],
                          self.ribf_atom, self.atom_ribfs)
        if self.verbose >= 2:
            print('[mem] DF ints GB:', {k: round(v / 1e9, 2) for k, v in self.dfi.nbytes.items()}, flush=True)
        self.J2c = self.dfi.J2c
        self.S_pao_d = cp.asarray(orb.S_pao)
        self.Fa_pao_d, self.Fb_pao_d = cp.asarray(self.Fa_pao), cp.asarray(self.Fb_pao)
        self.so = [(x, 0) for x in range(n)] + [(x, 1) for x in range(self.nda)]
        self.so_index = {k: I for I, k in enumerate(self.so)}
        self._chol = {}
        t = self._tick('df_ints', t)

    # ------------------------------------------------------------------
    # device DF helpers
    # ------------------------------------------------------------------
    def _metric_L(self, ribfs):
        key = np.asarray(ribfs, np.int64).tobytes()
        L = self._chol.get(key)
        if L is None:
            idx = cp.asarray(ribfs)
            L = cp.linalg.cholesky(self.J2c[idx[:, None], idx[None, :]])
            if len(self._chol) > 4096:
                self._chol.clear()
            self._chol[key] = L
        return L

    def _halfL(self, p):
        return self._metric_L(self.ribfs[p])

    def _half(self, p, x):
        return _half(self._halfL(p), cp.asarray(x))

    def _vir_coef_d(self, q):
        """alpha (PAO) and beta (PAO, SOMO) virtual coefficients on the device."""
        npo = len(self.pao[q])
        return (cp.asarray(self.Xa[q]), cp.asarray(self.Xb[q][:npo]), cp.asarray(self.Xb[q][npo:]))

    def _q_lmo(self, aux, lm, q):
        """(Q| m A) for m in lm (spatial), A in pair q's alpha and beta
        virtual spaces, not fitted: (naux, nl, nva), (naux, nl, nvb)."""
        dfi = self.dfi
        ga = cp.asarray(aux)
        gl = cp.asarray(np.asarray(lm, np.int64))
        Xa, Xbp, Xbs = self._vir_coef_d(q)
        qp = dfi.qia[ga[:, None, None], gl[None, :, None], cp.asarray(self.pao[q])[None, None, :]]
        qa = qp @ Xa
        qb = qp @ Xbp
        if self.ns:
            qs = dfi.qij[ga[:, None, None], gl[None, :, None], cp.asarray(self.somo_idx)[None, None, :]]
            qb = qb + qs @ Xbs
        return qa, qb

    def _q_vv_blocks(self, p, q1, q2):
        """(Q_p | A_q1 B_q2) not fitted, per spin: [(naux, nva1, nva2), (naux, nvb1, nvb2)]."""
        dfi = self.dfi
        aux = self.ribfs[p]
        atoms = np.unique(self.ribf_atom[aux])
        Xa1, Xb1, S1 = self._vir_coef_d(q1)
        Xa2, Xb2, S2 = self._vir_coef_d(q2)
        p1, p2 = self.pao[q1], self.pao[q2]
        a = dfi.q_vv(atoms, p1, Xa1, p2, Xa2)
        b = dfi.q_vv(atoms, p1, Xb1, p2, Xb2)
        if self.ns:
            ga = cp.asarray(aux)
            so = cp.asarray(self.somo_idx)
            q1s = dfi.qia[ga[:, None, None], so[None, :, None], cp.asarray(p1)[None, None, :]]
            q2s = dfi.qia[ga[:, None, None], so[None, :, None], cp.asarray(p2)[None, None, :]]
            qss = dfi.qij[ga[:, None, None], so[None, :, None], so[None, None, :]]
            b = b + cp.matmul(cp.matmul(Xb1.T[None], q1s.transpose(0, 2, 1)), S2[None])
            b = b + cp.matmul(cp.matmul(S1.T[None], q2s), Xb2[None])
            b = b + cp.matmul(cp.matmul(S1.T[None], qss), S2[None])
        return [a, b]

    # host-facing versions of the reference's DF builders (used by the
    # host triples path)
    def _B_ov(self, p, occ_so, q):
        aux = self.ribfs[p]
        lm = [self.so[M][0] for M in occ_so]
        qa, qb = self._q_lmo(aux, lm, q)
        out = cp.zeros((len(aux), len(occ_so), self.nv(q)))
        na = self.nva(q)
        for r, M in enumerate(occ_so):
            if self.so[M][1] == 0:
                out[:, r, :na] = qa[:, r]
            else:
                out[:, r, na:] = qb[:, r]
        return self._half(p, out).get()

    def _B_vv(self, p, q1, q2):
        a, b = self._q_vv_blocks(p, q1, q2)
        na1, na2 = self.nva(q1), self.nva(q2)
        out = cp.zeros((a.shape[0], self.nv(q1), self.nv(q2)))
        out[:, :na1, :na2] = a
        out[:, na1:, na2:] = b
        return self._half(p, out).get()

    def _B_oo(self, p, occ_so):
        aux = self.ribfs[p]
        lm = np.array([self.so[M][0] for M in occ_so], np.int64)
        sp = np.array([self.so[M][1] for M in occ_so])
        ga = cp.asarray(aux)
        gl = cp.asarray(lm)
        q = self.dfi.qij[ga[:, None, None], gl[None, :, None], gl[None, None, :]]
        q = q * cp.asarray((sp[:, None] == sp[None, :]).astype(np.float64))[None]
        return self._half(p, q).get()

    # ------------------------------------------------------------------
    # pairs, chunks
    # ------------------------------------------------------------------
    def set_pairs(self, mask, dom):
        super().set_pairs(mask, dom)
        self.riatoms = [np.union1d(dom.lmo_riatoms[i], dom.lmo_riatoms[j]).astype(np.int64)
                        for (i, j) in self.pl.pairs]

    def _canon_pairs(self):
        return [p for p, (i, j) in enumerate(self.pl.pairs) if i <= j]

    def _pair_chunks(self, plist, extra=0):
        npao = np.array([len(self.pao[p]) + extra for p in plist])
        naux = np.array([len(self.ribfs[p]) for p in plist])
        P = padded(npao)
        order = np.lexsort((naux, P))
        chunks, cur = [], []
        for o in order:
            if cur and P[o] != P[cur[0]]:
                chunks.append(cur)
                cur = []
            cur.append(o)
            A = int(-(-naux[cur].max() // 32) * 32)
            per = 8 * (A * A + 4 * A * int(P[o]) + 30 * int(P[o]) ** 2)
            if len(cur) > 1 and len(cur) * per > self.MAX_PAIR_CHUNK_BYTES:
                cur.pop()
                chunks.append(cur)
                cur = [o]
        if cur:
            chunks.append(cur)
        return [np.asarray(plist)[c] for c in chunks]

    # ------------------------------------------------------------------
    # semicanonical spin-orbital MP2 in the PAO domain, batched
    # ------------------------------------------------------------------
    def _sc_chunk(self, ch, pno=None):
        opt, pl, dfi = self.opt, self.pl, self.dfi
        B = len(ch)
        npao = np.array([len(self.pao[p]) for p in ch])
        P = int(padded(npao.max()))
        auxs = [self.ribfs[p] for p in ch]
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
        xx = np.array([pl.pairs[p][0] for p in ch])
        yy = np.array([pl.pairs[p][1] for p in ch])
        gp, gpm = cp.asarray(pidx), cp.asarray(pm)
        ga, gam = cp.asarray(aidx), cp.asarray(am)
        J = dfi.J2c[ga[:, :, None], ga[:, None, :]] * (gam[:, :, None] * gam[:, None, :])
        J += cp.eye(A)[None] * (1 - gam)[:, :, None]
        Linv = chol_inv_batched(J)
        mq = gam[:, :, None] * gpm[:, None, :]
        iq = cp.matmul(Linv, dfi.qia[ga[:, :, None], cp.asarray(xx)[:, None, None], gp[:, None, :]] * mq)
        jq = cp.matmul(Linv, dfi.qia[ga[:, :, None], cp.asarray(yy)[:, None, None], gp[:, None, :]] * mq)
        K = cp.matmul(iq.transpose(0, 2, 1), jq)                      # (x u|y v)
        del iq, jq, Linv, J
        mm = gpm[:, :, None] * gpm[:, None, :]
        S = self.S_pao_d[gp[:, :, None], gp[:, None, :]] * mm
        X0, keep = orthonormal_domain(S, npao, opt['S_CUT'], self.PAO_ORTHO)
        big = 1e6 * cp.eye(P)[None] * (~keep)[:, :, None]
        out = {}
        for s, Fd in ((0, self.Fa_pao_d), (1, self.Fb_pao_d)):
            F = Fd[gp[:, :, None], gp[:, None, :]] * mm
            e, V = eigh_batched(cp.matmul(X0.transpose(0, 2, 1), cp.matmul(F, X0)) + big)
            out[s] = (cp.matmul(X0, V), e)
        (Xa, ea), (Xb, eb) = out[0], out[1]
        valid = ea < 1e5                                    # same count for both spins
        U = cp.matmul(Xa.transpose(0, 2, 1), cp.matmul(S, Xb))
        fa, fb = np.diag(self.Fa_lmo), np.diag(self.Fb_lmo)
        dx = cp.asarray((~self.is_somo[xx]).astype(np.float64))
        dy = cp.asarray((~self.is_somo[yy]).astype(np.float64))
        vv = valid[:, :, None] & valid[:, None, :]

        def amp(Xl, Xr, el, er, f1, f2, anti):
            Kc = cp.matmul(Xl.transpose(0, 2, 1), cp.matmul(K, Xr))
            if anti:
                Kc = Kc - Kc.transpose(0, 2, 1)
            den = cp.asarray(f1 + f2)[:, None, None] - el[:, :, None] - er[:, None, :]
            return Kc, cp.where(vv, Kc / cp.where(vv, den, 1.), 0.)
        Kaa, taa = amp(Xa, Xa, ea, ea, fa[xx], fa[yy], True)
        Kbb, tbb = amp(Xb, Xb, eb, eb, fb[xx], fb[yy], True)
        Kab, tab = amp(Xa, Xb, ea, eb, fa[xx], fb[yy], False)
        Kba, tba = amp(Xb, Xa, eb, ea, fb[xx], fa[yy], False)
        dxy = dx * dy
        tbb *= dxy[:, None, None]
        tab *= dy[:, None, None]
        tba *= dx[:, None, None]
        e2 = (.5 * cp.sum(Kaa * taa, axis=(1, 2)) + .5 * cp.sum(Kbb * tbb, axis=(1, 2))
              + cp.sum(Kab * tab, axis=(1, 2)) + cp.sum(Kba * tba, axis=(1, 2)))
        e_pair = .5 * e2
        if pno is None:
            return e_pair.get()
        cut, trace, energy, lag = pno
        Ut = U.transpose(0, 2, 1)
        Bm = cp.matmul(U, cp.matmul(tbb, Ut))
        O = cp.matmul(tab, Ut)
        Op = cp.matmul(U, tba).transpose(0, 2, 1)
        tr = lambda X: X.transpose(0, 2, 1)
        D = (.5 * (cp.matmul(taa, tr(taa)) + cp.matmul(Bm, tr(Bm)))
             + .5 * (cp.matmul(O, tr(O)) + cp.matmul(tr(O), O) + cp.matmul(Op, tr(Op)) + cp.matmul(tr(Op), Op)))
        D[cp.asarray(xx == yy)] *= .5
        D -= cp.eye(P)[None] * (~valid)[:, :, None]
        occ, V = eigh_batched(D)
        occ, V = occ[:, ::-1], V[:, :, ::-1]
        Vb = cp.matmul(Ut, V)

        def proj(l, r, t, Kc, f):
            return f * cp.matmul(tr(l), cp.matmul(Kc, r)) * cp.matmul(tr(l), cp.matmul(t, r))
        # Kaa above is already antisymmetrised: .25 (Kn - Kn^T) tn = .25 Kn_anti tn
        M = (proj(V, V, taa, Kaa, .25) + proj(Vb, Vb, tbb, Kbb, .25)
             + proj(V, Vb, tab, Kab, .5) + proj(Vb, V, tba, Kba, .5))
        C = cp.cumsum(cp.cumsum(M, axis=1), axis=2)
        ecum = cp.concatenate([cp.zeros((B, 1)), cp.diagonal(C, axis1=1, axis2=2)], axis=1)
        nval = valid.sum(axis=1).get()
        occ_h, ecum_h, e_h = occ.get(), ecum.get(), e_pair.get()
        nfin = np.zeros(B, np.int64)
        for s, p in enumerate(ch):
            scale = opt['T_CUT_PNO_DIAG_SCALE'] if xx[s] == yy[s] else 1.
            o = occ_h[s, :nval[s]]
            nfin[s] = self._truncate(o, ecum_h[s], e_h[s], scale * cut, trace, energy, lag)
        Xp = cp.matmul(Xa, V)
        Xa_h, Xb_h, V_h, Vb_h, Xp_h = Xa.get(), Xb.get(), V.get(), Vb.get(), Xp.get()
        amps_h = [a.get() for a in (taa, tbb, tab, tba)]
        res = []
        for s, p in enumerate(ch):
            n_, nf = npao[s], nfin[s]
            nv = nval[s]
            am_ = dict(aa=amps_h[0][s, :nv, :nv])
            if not self.is_somo[xx[s]] and not self.is_somo[yy[s]]:
                am_['bb'] = amps_h[1][s, :nv, :nv]
            if not self.is_somo[yy[s]]:
                am_['ab'] = amps_h[2][s, :nv, :nv]
            if not self.is_somo[xx[s]]:
                am_['ba'] = amps_h[3][s, :nv, :nv]
            res.append((p, Xp_h[s, :n_, :nf], e_h[s] - ecum_h[s, nf], am_,
                        Xa_h[s, :n_, :nv], Xb_h[s, :n_, :nv]))
        return e_h, res

    def sc_mp2_pair_energies(self):
        pl = self.pl
        e = np.zeros(len(pl))

        def work(w, ch):
            v = w._sc_chunk(ch)
            e[ch] = v
            e[pl.ji[ch]] = v
        self._run_chunks(self._pair_chunks(self._canon_pairs()), work)
        return e

    def _run_chunks(self, chunks, work):
        """work(view, chunk) for every pair chunk, over the visible GPUs."""
        from electrolites.dlpno import mgpu
        chunks = sorted(chunks, key=lambda c: -len(c))
        mgpu.run_queue(chunks, work, setup=lambda d: mgpu.device_view(
            self, ('S_pao_d', 'Fa_pao_d', 'Fb_pao_d'), sub={'dfi': ('qia', 'qij', 'J2c')}))

    # ------------------------------------------------------------------
    # PNO spaces (alpha: PNOs, beta: PNOs + SOMOs), batched
    # ------------------------------------------------------------------
    def _spaces_batch(self, plist, Xpnos):
        """canonical alpha / beta spaces of each pair from its PNOs."""
        ns = self.ns
        nvs = np.array([X.shape[1] for X in Xpnos])
        npaos = np.array([len(self.pao[p]) for p in plist])
        out = {}
        key = padded(nvs + ns) * 100000 + padded(npaos)
        for k in np.unique(key):
            sel = np.where(key == k)[0]
            V = int(padded((nvs[sel] + ns).max()))
            Pp = int(padded(npaos[sel].max()))
            B = len(sel)
            Xh = np.zeros((B, Pp, V))
            ph = np.zeros((B, Pp), np.int64)
            pm = np.zeros((B, Pp))
            vm = np.zeros((B, V))
            for r, s in enumerate(sel):
                n_, v = npaos[s], nvs[s]
                Xh[r, :n_, :v] = Xpnos[s]
                ph[r, :n_] = self.pao[plist[s]]
                pm[r, :n_] = 1.
                vm[r, :v] = 1.
            X, gp, gpm, gvm = cp.asarray(Xh), cp.asarray(ph), cp.asarray(pm), cp.asarray(vm)
            mm = gpm[:, :, None] * gpm[:, None, :]
            Fa = self.Fa_pao_d[gp[:, :, None], gp[:, None, :]] * mm
            Fb = self.Fb_pao_d[gp[:, :, None], gp[:, None, :]] * mm
            tr = lambda Y: Y.transpose(0, 2, 1)
            fa = cp.matmul(tr(X), cp.matmul(Fa, X))
            fa += 1e6 * cp.eye(V)[None] * (1 - gvm)[:, :, None]
            ea, Ua = eigh_batched(fa)
            Xa = cp.matmul(X, Ua)
            # beta: [PNO | SOMO] block matrix; PNO count v sits in the first V slots
            W = V + ns
            fb = cp.zeros((B, W, W))
            fb[:, :V, :V] = cp.matmul(tr(X), cp.matmul(Fb, X))
            if ns:
                Fps = cp.asarray(self.Fb_ps)[gp] * gpm[:, :, None]      # (B, Pp, ns)
                fps = cp.matmul(tr(X), Fps)
                fb[:, :V, V:] = fps
                fb[:, V:, :V] = tr(fps)
                fb[:, V:, V:] = cp.asarray(self.Fb_ss)[None]
            mb = cp.concatenate([gvm, cp.ones((B, ns))], axis=1)
            fb += 1e6 * cp.eye(W)[None] * (1 - mb)[:, :, None]
            eb, Ub = eigh_batched(fb)
            Xbp = cp.matmul(X, Ub[:, :V])
            Xa_h, ea_h, Xbp_h, Ub_h, eb_h = Xa.get(), ea.get(), Xbp.get(), Ub.get(), eb.get()
            for r, s in enumerate(sel):
                n_, v = npaos[s], nvs[s]
                Xb = np.zeros((n_ + ns, v + ns))
                Xb[:n_] = Xbp_h[r, :n_, :v + ns]
                Xb[n_:] = Ub_h[r, V:, :v + ns]
                out[plist[s]] = (Xa_h[r, :n_, :v], ea_h[r, :v], Xb, eb_h[r, :v + ns])
        return out

    def _store_all(self, plist, Xpnos):
        sp = self._spaces_batch(plist, Xpnos)
        for p, X in zip(plist, Xpnos):
            Xa, ea, Xb, eb = sp[p]
            for r in {p, self.pl.ji[p]}:
                self.Xpno[r] = X
                self.Xa[r], self.ea[r], self.Xb[r], self.eb[r] = Xa, ea, Xb, eb

    # ------------------------------------------------------------------
    def lmp2_pnos(self):
        opt, pl = self.opt, self.pl
        npair = len(pl)
        self.Xpno = [None] * npair
        self.Xa, self.ea, self.Xb, self.eb = ([None] * npair for _ in range(4))
        self.de_pno = np.zeros(npair)
        self._S = {}
        pno = (opt['T_CUT_PNO_MP2'], opt['T_CUT_TRACE_MP2'], opt['T_CUT_ENERGY_MP2'], True)
        sc = {}
        plist, Xps = [], []
        def work(w, ch):
            _, res = w._sc_chunk(ch, pno)
            for p, Xp, de, amps, Xa_sc, Xb_sc in res:
                self.de_pno[p] = self.de_pno[pl.ji[p]] = de
                sc[p] = (amps, Xa_sc, Xb_sc)
                plist.append(p)
                Xps.append(Xp)
        self._run_chunks(self._pair_chunks(self._canon_pairs()), work)
        self._store_all(plist, Xps)
        self.de_pno_total = self.de_pno.sum()
        self._so_pairs()
        letter = 'ab'
        self.T2b = {}
        tcache = {}
        for (I, J) in self.sop:
            x, s = self.so[I]
            y, t = self.so[J]
            p = pl.ij[x, y]
            flip = x > y
            c = pl.ji[p] if flip else p
            amps, Xa_sc, Xb_sc = sc[c]
            T = tcache.get(c)
            if T is None:          # semicanonical -> PNO-space maps, once per spatial pair
                paos = self.pao[c]
                Sd = self.orb.S_pao[np.ix_(paos, paos)]
                npo = len(paos)
                T = tcache[c] = {0: Xa_sc.T @ Sd @ self.Xa[c], 1: Xb_sc.T @ Sd @ self.Xb[c][:npo]}
            s1, s2 = (t, s) if flip else (s, t)
            key = letter[s1] + letter[s2]
            blk = T[s1].T @ amps[key] @ T[s2]
            self.T2b[(I, J)] = blk.T if flip else blk

    # ------------------------------------------------------------------
    # pair-space DF blocks of the pair's own LMOs
    # ------------------------------------------------------------------
    def _pair_B(self, c, lm):
        """half-fitted (Q_c | m A_c) for m in lm, per spin: [(A, nl, nva), (A, nl, nvb)]."""
        qa, qb = self._q_lmo(self.ribfs[c], lm, c)
        L = self._halfL(c)
        return [_half(L, qa), _half(L, qb)]

    def _exchange_blocks(self):
        """G_IJ (spin I, spin J) block of <A B||I J> for every spin-orbital
        pair, in the current pair spaces."""
        pl, si = self.pl, self.so_index
        G = {}
        for c in self._canon_pairs():
            x, y = pl.pairs[c]
            Bm = self._pair_B(c, [x, y] if x != y else [x])
            pos = {x: 0, y: 1 if x != y else 0}
            for (z1, z2) in {(x, y), (y, x)}:
                for s in (0, 1):
                    for t in (0, 1):
                        I, J = si.get((z1, s)), si.get((z2, t))
                        if I is None or J is None or I == J:
                            continue
                        bi, bj = Bm[s][:, pos[z1]], Bm[t][:, pos[z2]]
                        M = bi.T @ bj
                        if s == t:
                            M = M - bj.T @ bi
                        G[(I, J)] = M
        return G

    # ------------------------------------------------------------------
    # LMP2 iterations, spin blocked on the device
    # ------------------------------------------------------------------
    def _so_layout(self):
        sop = self.sop
        sp = np.array([self.pl.ij[self.so[I][0], self.so[J][0]] for (I, J) in sop], np.int64)
        nvs = np.array([[self.nva(p), self.nvb(p)] for p in range(len(self.pl))], np.int64)
        si = np.array([self.so[I][1] for (I, J) in sop])
        sj = np.array([self.so[J][1] for (I, J) in sop])
        return sp, nvs, Layout(nvs[sp, si], nvs[sp, sj])

    def lmp2_iterations(self):
        t_l0 = time.perf_counter()
        opt, pl = self.opt, self.pl
        G = self._exchange_blocks()
        sop = self.sop
        sidx = {k: n for n, k in enumerate(sop)}
        sp_of, nvs, LT = self._so_layout()
        spin = np.array([s for (x, s) in self.so])
        canon = [k for k in sop if k[0] < k[1]]
        # couplings, vectorised over (canonical spin-orbital pair, K)
        nso = len(self.so)
        so_l = np.array([x for x, _ in self.so], np.int64)
        so_s = np.array([t for _, t in self.so], np.int64)
        SIDX = -np.ones((nso, nso), np.int64)
        for n_, k_ in enumerate(sop):
            SIDX[k_[0], k_[1]] = n_
        Fso = np.zeros((nso, nso))                       # spin-orbital occupied Fock
        for sg, Fm in ((0, self.Fa_lmo), (1, self.Fb_lmo)):
            m = np.flatnonzero(so_s == sg)
            Fso[np.ix_(m, m)] = Fm[np.ix_(so_l[m], so_l[m])]
        CI = np.array([k_[0] for k_ in canon], np.int64)
        CJ = np.array([k_[1] for k_ in canon], np.int64)
        CO = SIDX[CI, CJ]
        CP = sp_of[CO]
        Kall = np.arange(nso)
        parts_ = []
        for side in (0, 1):
            # side 0: T_IK coupled through f_KJ (pair (x, k));  side 1: T_KJ through f_KI (pair (k, y))
            F = Fso[:, CJ].T if side == 0 else Fso[:, CI].T              # (ncanon, nso): f(K, J or I)
            pr = pl.ij[so_l[CI][:, None], so_l[Kall][None, :]] if side == 0 else \
                pl.ij[so_l[Kall][None, :], so_l[CJ][:, None]]
            ok = ((Kall[None, :] != CI[:, None]) & (Kall[None, :] != CJ[:, None]) & (pr >= 0)
                  & (np.abs(F) > opt['F_CUT']))
            r, K = np.nonzero(ok)
            tt = SIDX[CI[r], K] if side == 0 else SIDX[K, CJ[r]]
            parts_.append((r, pr[r, K], tt, -F[r, K]))
        rr = np.concatenate([q[0] for q in parts_])
        assert all((q[2] >= 0).all() for q in parts_), "coupled amplitude block missing"
        qp = np.concatenate([q[1] for q in parts_])
        tt = np.concatenate([q[2] for q in parts_])
        ww = np.concatenate([q[3] for q in parts_])
        npl = len(pl)
        cc_code, cid_ = np.unique(CP[rr] * npl + qp, return_inverse=True)
        Creg = _Reg()
        Creg.keys = [(int(c // npl), int(c % npl)) for c in cc_code]
        Creg.ids = {k_: n_ for n_, k_ in enumerate(Creg.keys)}
        rows_ = np.stack([CO[rr].astype(np.float64), cid_.reshape(-1).astype(np.float64), tt.astype(np.float64),
                          so_s[CI[rr]].astype(np.float64), so_s[CJ[rr]].astype(np.float64), ww], 1)
        # couple chunks of S(p, ik) per spin; with several GPUs the chunks
        # (their generators, kept overlaps and GEMM plans) are spread over them
        from electrolites.dlpno.ovgen import OverlapGen
        from electrolites.dlpno import mgpu
        npair = len(pl)
        devs = mgpu.devices()
        home = cp.cuda.Device().id
        XA = [self.Xa[p] for p in range(npair)]
        XB = [self.Xb[p] for p in range(npair)]
        gens = {}

        def gens_on(d):
            if d not in gens:
                with cp.cuda.Device(d):
                    Sd = cp.asarray(self.S_pao_d)
                    gens[d] = (OverlapGen(Sd, self.pao, XA), OverlapGen(Sd, self.pao, XB, ne=self.ns))
            return gens[d]
        gA, gB = gens_on(home)
        pp = np.array([k[0] for k in Creg.keys], np.int64)
        qq = np.array([k[1] for k in Creg.keys], np.int64)
        R_ = np.array(rows_, dtype=np.float64).reshape(-1, 6)
        parts = gB.chunks(pp, qq)
        cpl = R_[:, 1].astype(np.int64)
        chunk_of = np.zeros(len(pp), np.int64)
        local = np.zeros(len(pp), np.int64)
        for ci, (sel, _, _) in enumerate(parts):
            chunk_of[sel] = ci
            local[sel] = np.arange(len(sel))
        rc = chunk_of[cpl]
        ro = np.argsort(rc, kind='stable')
        rb_ = np.searchsorted(rc[ro], np.arange(len(parts) + 1))
        load = np.zeros(len(devs))
        lmp2_chunks = []
        for ci in range(len(parts)):
            rsel = ro[rb_[ci]:rb_[ci + 1]]
            r_ = int(np.argmin(load))
            load[r_] += len(rsel) + 1
            d = devs[r_]
            with cp.cuda.Device(d):
                ga_, gb_ = gens_on(d)
                Pa = ga_.prepare(pp, qq, parts=[parts[ci]])[0]
                Pb = gb_.prepare(pp, qq, parts=[parts[ci]])[0]
                t_ = _Terms()
                t_._keys = ('o', 's', 't', 'a', 'b', 'w')
                Rs = R_[rsel]
                t_._rows = list(zip(Rs[:, 0].astype(np.int64).tolist(), local[Rs[:, 1].astype(np.int64)].tolist(),
                                    Rs[:, 2].astype(np.int64).tolist(), Rs[:, 3].astype(np.int64).tolist(),
                                    Rs[:, 4].astype(np.int64).tolist(), Rs[:, 5].tolist()))
                lays = dict(S=[Pa['LS'], Pb['LS']], T=LT)
                pl_ = _SpinPlan(lays, LT, t_, 'o', [('S', 's', 'a', False), ('T', 't', None, False),
                                                   ('S', 's', 'b', True)])
                lmp2_chunks.append((Pa, Pb, pl_, cp.asarray(t_.f('w')) if len(t_) else None, d))
        cp.cuda.Device(home).use()
        Gr = Ragged(LT)
        _set_items(Gr, range(len(sop)), [G[k] for k in sop])
        T = Ragged(LT)
        _set_items(T, range(len(sop)), [cp.asarray(self.T2b[k]) for k in sop])
        fd = [np.diag(self.Fa_lmo), np.diag(self.Fb_lmo)]
        D = []
        for g, m in enumerate(LT.members):
            pr, pc = LT.shapes[g]
            Dg = np.ones((len(m), pr, pc))
            for s_, it in enumerate(m):
                I, J = sop[it]
                p = sp_of[it]
                ea = (self.ea[p], self.eb[p])[spin[I]]
                eb = (self.ea[p], self.eb[p])[spin[J]]
                Dg[s_, :len(ea), :len(eb)] = (ea[:, None] + eb[None, :]
                                              - fd[spin[I]][self.so[I][0]] - fd[spin[J]][self.so[J][0]])
            D.append(cp.asarray(Dg))
        D = Ragged(LT, D)
        cmask = np.zeros(len(sop), bool)
        cmask[[sidx[k] for k in canon]] = True
        mult = np.array([2. if spin[I] != spin[J] else 1. for (I, J) in sop])
        src = np.array([sidx[k] for k in canon], np.int64)
        dst = np.array([sidx[(k[1], k[0])] for k in canon], np.int64)
        diis = _DeviceDIIS(opt['DIIS_MAX_VECS'])
        ec, rc = 0.01 * opt['E_CONVERGENCE'], 0.01 * opt['R_CONVERGENCE']
        e_prev = 0.
        gm = [cp.asarray(cmask[m]) for m in LT.members]
        # keep the overlaps when they fit comfortably, else regenerate them per iteration
        import os
        mode = os.environ.get('DLPNO_ONFLY_S', 'auto')
        cdevs = sorted({c[4] for c in lmp2_chunks})
        onfly = {}
        for d in cdevs:
            with cp.cuda.Device(d):
                need = sum(Pa['LS'].nbytes() + Pb['LS'].nbytes() for Pa, Pb, _, _, dd in lmp2_chunks if dd == d)
                free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
                onfly[d] = (need > 0.5 * free) if mode == 'auto' else mode == '1'
        Skept = []
        for Pa, Pb, pl_, _, d in lmp2_chunks:
            if onfly[d] or not pl_.n:
                Skept.append(None)
                continue
            with cp.cuda.Device(d):
                ga_, gb_ = gens_on(d)
                Skept.append((ga_.run(Pa), gb_.run(Pb)))
        cp.cuda.Device(home).use()

        def coupling(R, T):
            parts_ = {}

            def work(d):
                if d == home:
                    R_, T_ = R, T
                else:
                    R_, T_ = Ragged(R.layout), Ragged(T.layout, mgpu.replicate(T.data))
                for c_, (Pa, Pb, pl_, w_, dd) in enumerate(lmp2_chunks):
                    if dd != d or not pl_.n:
                        continue
                    ga_, gb_ = gens_on(d)
                    Sab = Skept[c_] if Skept[c_] is not None else (ga_.run(Pa), gb_.run(Pb))
                    pl_.run(R_, dict(S=list(Sab), T=T_), w=w_)
                parts_[d] = R_
            mgpu.run_per_device(work, cdevs)
            for d, R_ in parts_.items():
                if d != home:
                    for g in range(len(R.data)):
                        if R.data[g].size:
                            R.data[g] += cp.asarray(R_.data[g])
        # persistent buffers (stable addresses: cached GEMM pointer tables, no
        # host synchronisation inside an iteration)
        R = Ragged(LT)
        fm = [z.astype(cp.float64)[:, None, None] for z in gm]
        tix = _transpose_index(LT, src, dst)
        self.log(f'  ULMP2 setup {time.perf_counter() - t_l0:.1f} s')
        for it in range(1, opt['MAXITER'] + 1):
            t_it = time.perf_counter()
            for g in range(len(R.data)):
                cp.multiply(D.data[g], T.data[g], out=R.data[g])
                R.data[g] += Gr.data[g]
            coupling(R, T)
            for g in range(len(R.data)):
                R.data[g] *= fm[g]
            rms = float(R.rms_per_item().max())
            for g in range(len(T.data)):
                T.data[g] -= R.data[g] / D.data[g]
            _transpose_inplace(T, tix)
            T.set_flat(diis.update(T.flat(), R.flat()))
            e_so = Gr.vdot_items(T) * cp.asarray(.25 * mult)
            e_cur = float(e_so.sum())
            self.log(f'  GPU ULMP2 iter {it:3d} {e_cur:.12f} {e_cur - e_prev:.3e} {rms:.3e} '
                     f'{time.perf_counter() - t_it:.2f}s')
            if abs(e_cur - e_prev) < ec and rms < rc:
                break
            e_prev = e_cur
        else:
            raise RuntimeError('LMP2 did not converge')
        self.e_lmp2 = e_cur
        e_sp = np.zeros(len(pl))
        np.add.at(e_sp, sp_of, e_so.get())
        Th = T.to_list()
        self.T2b = {k: Th[n] for n, k in enumerate(sop)}
        self.G_lmp2 = {k: G[k].get() for k in sop}
        return .5 * (e_sp + e_sp[pl.ji])

    def _pno_overlaps(self, Sreg):
        """per spin the PNO overlaps of the registered (p, q) couples."""
        from electrolites.dlpno.lmp2_gpu import overlaps_chunked
        npair = len(self.pl)
        nvs = np.array([[self.nva(p), self.nvb(p)] for p in range(npair)], np.int64)
        Xa = [self.Xa[p] for p in range(npair)]
        Xbp = [self.Xb[p][:len(self.pao[p])] for p in range(npair)]
        Xbs = [self.Xb[p][len(self.pao[p]):] for p in range(npair)]
        LSs, Ss = [], []
        for s in (0, 1):
            keys = Sreg[s].keys
            pp = np.array([k[0] for k in keys], np.int64)
            qq = np.array([k[1] for k in keys], np.int64)
            LS = Layout(nvs[pp, s], nvs[qq, s])
            if s == 0:
                Sr = overlaps_chunked(self.S_pao_d, self.pao, Xa, pp, qq, LS)
            else:
                Sr = overlaps_chunked(self.S_pao_d, self.pao, Xbp, pp, qq, LS, Xs=Xbs if self.ns else None)
            LSs.append(LS)
            Ss.append(Sr)
        return LSs, Ss

    # ------------------------------------------------------------------
    # CC PNOs from the LMP2 amplitudes (host, block-wise; small matrices)
    # ------------------------------------------------------------------
    def recompute_pnos(self):
        opt, pl = self.opt, self.pl
        si = self.so_index
        ns = self.ns
        newX = {}
        dens = {}
        for p, (x, y) in enumerate(pl.pairs):
            if x > y:
                continue
            q = pl.ji[p]
            paos = self.pao[p]
            npo = len(paos)
            Sd = self.orb.S_pao[np.ix_(paos, paos)]
            Xo = self.Xpno[p]
            npn = Xo.shape[1]
            Wa = Xo.T @ Sd @ self.Xa[p]
            Wbp = Xo.T @ Sd @ self.Xb[p][:npo]
            Wb = np.vstack([Wbp, self.Xb[p][npo:]])
            W = (Wa, Wbp)

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
            dens[p] = (Xo, D)
            occ, V = np.linalg.eigh(D)
            occ, V = occ[::-1], V[:, ::-1]
            scale = opt['T_CUT_PNO_DIAG_SCALE'] if x == y else 1.
            # pair energy kept by the first m PNOs; ranks: alpha / beta PNO m -> m+1, SOMO -> 0
            Y = (V.T @ Wa, scipy.linalg.block_diag(V, np.eye(ns)).T @ Wb)
            rank = (np.arange(1, npn + 1), np.concatenate([np.arange(1, npn + 1), np.zeros(ns, int)]))
            hist = np.zeros(npn + 1)
            ks = self.sop_of.get(p, []) + (self.sop_of.get(q, []) if q != p else [])
            fac = .5 if q != p else 1.
            e_tot = 0.
            for k in ks:
                a, b = self.so[k[0]][1], self.so[k[1]][1]
                Tn = Y[a] @ self.T2b[k] @ Y[b].T
                Gn = Y[a] @ self.G_lmp2[k] @ Y[b].T
                mult = 1. if a == b else 2.
                Mk = fac * .25 * mult * Tn * Gn
                e_tot += Mk.sum()
                h = np.maximum(rank[a][:, None], rank[b][None, :])
                hist += np.bincount(h.ravel(), weights=Mk.ravel(), minlength=npn + 1)
            ecum = np.cumsum(hist)
            nfin = self._truncate(occ, ecum, e_tot, scale * opt['T_CUT_PNO'],
                                  opt['T_CUT_TRACE'], opt['T_CUT_ENERGY'], False)
            self.de_pno[p] += e_tot - ecum[nfin]
            if q != p:
                self.de_pno[q] = self.de_pno[p]
            newX[p] = Xo @ V[:, :nfin]
        self._somo_singles_spaces(dens, newX)
        oldXa, oldXb = list(self.Xa), list(self.Xb)
        plist = list(newX)
        self._store_all(plist, [newX[p] for p in plist])
        rcache = {}
        for k in list(self.T2b):
            p = self.pl.ji[self._sp(k)] if self.pl.pairs[self._sp(k)][0] > self.pl.pairs[self._sp(k)][1] \
                else self._sp(k)
            R = rcache.get(p)
            if R is None:          # old -> new space maps, once per spatial pair
                paos = self.pao[p]
                npo = len(paos)
                Sd = self.orb.S_pao[np.ix_(paos, paos)]
                R = rcache[p] = (oldXa[p].T @ Sd @ self.Xa[p],
                                 oldXb[p][:npo].T @ Sd @ self.Xb[p][:npo] + oldXb[p][npo:].T @ self.Xb[p][npo:])
            self.T2b[k] = R[self.so[k[0]][1]].T @ self.T2b[k] @ R[self.so[k[1]][1]]
        self._S = {}
        self.de_pno_total = self.de_pno.sum()

    # ------------------------------------------------------------------
    # CC integrals, directly in the blocks UCCSDGPU uses
    # ------------------------------------------------------------------
    def cc_integrals(self):
        self.cc = None        # the dense host tensors of the reference are not built

    MAX_INT_CHUNK_BYTES = 1 << 30

    def _pair_core_chunk(self, ch):
        """half-fitted (Q|m A^s) for the domain LMOs m, (Q|m n) and (Q|A^s B^s)
        of a chunk of canonical pairs, batched; per pair views of true size."""
        dfi, ns, ext = self.dfi, self.ns, self.dfi.ext
        B = len(ch)
        npao = np.array([len(self.pao[c]) for c in ch])
        nla = np.array([len(self.lmos[c]) for c in ch])
        auxs = [self.ribfs[c] for c in ch]
        P, Lm = int(padded(npao.max())), int(padded(nla.max()))
        A = int(-(-max(len(a) for a in auxs) // 32) * 32)
        Va = int(padded(max(self.nva(c) for c in ch)))
        Vb = int(padded(max(self.nvb(c) for c in ch)))
        pidx, pm = np.zeros((B, P), np.int64), np.zeros((B, P))
        lidx, lmk = np.zeros((B, Lm), np.int64), np.zeros((B, Lm))
        aidx, am = np.zeros((B, A), np.int64), np.zeros((B, A))
        Xa, Xbp, Xbs = np.zeros((B, P, Va)), np.zeros((B, P, Vb)), np.zeros((B, ns, Vb))
        offs = []
        for s_, c in enumerate(ch):
            n_ = npao[s_]
            pidx[s_, :n_], pm[s_, :n_] = self.pao[c], 1.
            lidx[s_, :nla[s_]], lmk[s_, :nla[s_]] = self.lmos[c], 1.
            aidx[s_, :len(auxs[s_])], am[s_, :len(auxs[s_])] = auxs[s_], 1.
            Xa[s_, :n_, :self.nva(c)] = self.Xa[c]
            Xbp[s_, :n_, :self.nvb(c)] = self.Xb[c][:n_]
            Xbs[s_, :, :self.nvb(c)] = self.Xb[c][n_:]
            o, at = 0, {}
            for a in np.unique(self.ribf_atom[auxs[s_]]):
                at[int(a)] = o
                o += len(dfi.atom_ribfs[a])
            offs.append(at)
        gp, gpm = cp.asarray(pidx), cp.asarray(pm)
        gl, glm = cp.asarray(lidx), cp.asarray(lmk)
        ga, gam = cp.asarray(aidx), cp.asarray(am)
        Xa_d, Xbp_d, Xbs_d = cp.asarray(Xa), cp.asarray(Xbp), cp.asarray(Xbs)
        J = dfi.J2c[ga[:, :, None], ga[:, None, :]] * (gam[:, :, None] * gam[:, None, :])
        J += cp.eye(A)[None] * (1 - gam)[:, :, None]
        Linv = chol_inv_batched(J)
        del J
        mq = (gam[:, :, None, None] * glm[:, None, :, None]) * gpm[:, None, None, :]
        q = dfi.qia[ga[:, :, None, None], gl[:, None, :, None], gp[:, None, None, :]] * mq     # (B,A,L,P)
        qa = cp.matmul(q, Xa_d[:, None])
        qb = cp.matmul(q, Xbp_d[:, None])
        del q
        so = cp.asarray(self.somo_idx)
        if ns:
            qs = dfi.qij[ga[:, :, None, None], gl[:, None, :, None], so[None, None, None, :]]
            qb += cp.matmul(qs * (gam[:, :, None, None] * glm[:, None, :, None]), Xbs_d[:, None])
        Bm = [cp.matmul(Linv, x.reshape(B, A, -1)).reshape(x.shape) for x in (qa, qb)]
        del qa, qb
        qoo = dfi.qij[ga[:, :, None, None], gl[:, None, :, None], gl[:, None, None, :]]
        qoo *= gam[:, :, None, None] * glm[:, None, :, None] * glm[:, None, None, :]
        Boo = cp.matmul(Linv, qoo.reshape(B, A, -1)).reshape(B, A, Lm, Lm)
        del qoo
        qvv = [cp.zeros((B, A, Va, Va)), cp.zeros((B, A, Vb, Vb))]
        atoms = sorted({a for o_ in offs for a in o_})
        tr = lambda Y: Y.swapaxes(-1, -2)
        for a in atoms:
            qa_ = dfi.qab_get(a)
            if qa_ is None:
                continue
            na = qa_.shape[0]
            rows = np.array([r for r in range(B) if a in offs[r]], np.int64)
            off = np.array([offs[r][a] for r in rows], np.int64)
            pos = ext.pao_pos[a][pidx[rows]]
            pos = np.where(pm[rows] > 0, pos, 0)
            assert (pos >= 0).all(), 'pair PAO outside the extended domain'
            gr, gpos = cp.asarray(rows), cp.asarray(pos)
            mrow = gpm[gr]
            step = max(1, int((1 << 28) // (8 * na * P * P)))
            for r0 in range(0, len(rows), step):
                sl = slice(r0, r0 + step)
                blk = gather_blocks(qa_, gpos[sl], gpos[sl], mrow[sl], mrow[sl], ru=True)
                oo = cp.asarray(off[sl])[:, None] + cp.arange(na)[None, :]
                for s, Xs in ((0, Xa_d), (1, Xbp_d)):
                    xr = Xs[gr[sl]]
                    qvv[s][gr[sl][:, None], oo] = xtbx(blk, xr, xr)
        if ns:
            qsu = dfi.qia[ga[:, :, None, None], so[None, None, :, None], gp[:, None, None, :]]
            qsu = qsu * (gam[:, :, None, None] * gpm[:, None, None, :])
            qss = dfi.qij[ga[:, :, None, None], so[None, None, :, None], so[None, None, None, :]]
            qss = qss * gam[:, :, None, None]
            m1 = cp.matmul(cp.matmul(tr(Xbp_d)[:, None], tr(qsu)), Xbs_d[:, None])
            qvv[1] += m1 + tr(m1) + cp.matmul(cp.matmul(tr(Xbs_d)[:, None], qss), Xbs_d[:, None])
        Qvv = [cp.matmul(Linv, x.reshape(B, A, -1)).reshape(x.shape) for x in qvv]
        del qvv
        out = {}
        for s_, c in enumerate(ch):
            na_, nl_ = len(auxs[s_]), nla[s_]
            va, vb = self.nva(c), self.nvb(c)
            out[c] = ([Bm[0][s_, :na_, :nl_, :va], Bm[1][s_, :na_, :nl_, :vb]],
                      Boo[s_, :na_, :nl_, :nl_],
                      [Qvv[0][s_, :na_, :va, :va], Qvv[1][s_, :na_, :vb, :vb]])
        return out

    def gpu_pair_ints_many(self, clist, rows_of, strong_of, host=False):
        """[(c, entry)] for every canonical pair, the expensive parts batched
        over chunks of pairs with equal padded PAO-domain size; the chunks are
        distributed over the visible GPUs and the entries gathered on the
        current one."""
        clist = list(clist)
        npao = np.array([len(self.pao[c]) for c in clist])
        P = padded(npao)
        chunks = []
        for Pv in np.unique(P):
            sel = [c for c, pv in zip(clist, P) if pv == Pv]
            o_ = self.dfi.spatial_order([np.unique(self.ribf_atom[self.ribfs[c]]) for c in sel])
            sel = [sel[k] for k in o_]
            ch = []
            for c in sel + [None]:
                if c is not None:
                    A = -(-len(self.ribfs[c]) // 32) * 32
                    per = 8 * A * (int(padded(len(self.lmos[c]))) * (int(Pv) + 2 * self.nvb(c))
                                   + 3 * self.nvb(c) ** 2 + A)
                    if ch and (len(ch) + 1) * per <= self.MAX_INT_CHUNK_BYTES:
                        ch.append(c)
                        continue
                if ch:
                    chunks.append(ch)
                ch = [c] if c is not None else []
        from electrolites.dlpno import mgpu
        home = cp.cuda.Device().id
        out = []

        def work(w, ch):
            core = w._pair_core_chunk(ch)
            res = [(c2, w.gpu_pair_ints(c2, rows_of[c2], strong_of[c2], host, core=core[c2])) for c2 in ch]
            del core
            if cp.cuda.Device().id != home:
                with cp.cuda.Device(home):
                    res = [(c2, mgpu.replicate(e)) for c2, e in res]
            out.extend(res)
        chunks.sort(key=lambda c: -len(c))
        mgpu.run_queue(chunks, work, setup=lambda d: mgpu.device_view(
            self, ('S_pao_d', 'Fa_pao_d', 'Fb_pao_d'), sub={'dfi': ('qia', 'qij', 'J2c')}))
        return out

    def gpu_pair_ints(self, c, rows, strong, host=False, core=None):
        """fixed integrals of canonical spatial pair c for UCCSDGPU."""
        pl, si = self.pl, self.so_index
        x, y = pl.pairs[c]
        lm = list(self.lmos[c])
        lpos = {m: r for r, m in enumerate(lm)}
        if core is not None:
            # own copies: views would keep the whole padded chunk alive
            Bm, Boo, Qvv = core
            Bm = [cp.ascontiguousarray(b) for b in Bm]
            Boo = cp.ascontiguousarray(Boo)
            Qvv = [cp.ascontiguousarray(q) for q in Qvv]
        else:
            aux = self.ribfs[c]
            L = self._halfL(c)
            Bm = self._pair_B(c, lm)                       # [(A, nl, nva), (A, nl, nvb)]
            ga, gl = cp.asarray(aux), cp.asarray(np.asarray(lm, np.int64))
            Boo = _half(L, self.dfi.qij[ga[:, None, None], gl[None, :, None], gl[None, None, :]])
            Qvv = [_half(L, v) for v in self._q_vv_blocks(c, c, c)]
        ent = dict(lm=lm, lpos=lpos, Kx={}, K1={}, K3={}, J1={})
        for (z1, z2) in {(x, y), (y, x)}:
            for s in (0, 1):
                for t in (0, 1):
                    if (z1, s) in si and (z2, t) in si:
                        ent['Kx'][(z1, z2, s, t)] = Bm[s][:, lpos[z1]].T @ Bm[t][:, lpos[z2]]
        for s in (0, 1):
            ent['J1'][s] = cp.einsum('Q,Qmb->mb', Boo[:, lpos[x], lpos[y]], Bm[s])
            for (z1, z2) in {(x, y), (y, x)}:
                if (z2, s) in si:
                    ent['K1'][(z1, z2, s)] = Boo[:, lpos[z1], :].T @ Bm[s][:, lpos[z2]]
        for z in {x, y}:
            for s in (0, 1):
                if (z, s) not in si:
                    continue
                for t in (0, 1):
                    ent['K3'][(z, s, t)] = cp.einsum('Qc,Qbd->cbd', Bm[s][:, lpos[z]], Qvv[t])
        fov = []
        for s in (0, 1):
            fov.append(cp.asarray(self._f_ov_rows(c, [self.so[I][0] for I in rows[s]], s)))
        ent['fov'] = fov
        ent['ev'] = [cp.asarray(self.ea[c]), cp.asarray(self.eb[c])]
        ent['rowsp'] = [[lpos[self.so[I][0]] for I in rows[s]] for s in (0, 1)]
        if strong:
            ent['Qov'] = [Bm[s][:, cp.asarray(np.asarray(ent['rowsp'][s], np.int64))] for s in (0, 1)]
            ent['Qvv'] = Qvv
            ent['Qmz'] = {z: Boo[:, :, lpos[z]] for z in {x, y}}
            ent['Qaz'] = {(z, s): Bm[s][:, lpos[z]] for z in {x, y} for s in (0, 1) if (z, s) in si}
        if host:
            # the large DF / 3-external blocks wait in pinned host memory
            from electrolites.dlpno.devmem import to_pinned
            for key in ('Qov', 'Qvv'):
                if key in ent:
                    ent[key] = [to_pinned(v) for v in ent[key]]
            for key in ('Qmz', 'Qaz', 'K3'):
                if key in ent:
                    ent[key] = {k: to_pinned(v) for k, v in ent[key].items()}
        return ent

    def _f_ov_rows(self, p, lm, s):
        pp = self.pao[p]
        npo = len(pp)
        if s == 0:
            return self.Fa_lp[np.ix_(lm, pp)] @ self.Xa[p]
        v = self.Fb_lp[np.ix_(lm, pp)] @ self.Xb[p][:npo]
        if self.ns:
            v = v + self.Fb_lmo[np.ix_(lm, self.somo_idx)] @ self.Xb[p][npo:]
        return v

    def gpu_ring_np(self, keys):
        """non-projected ring integrals <K B_p || C_q Q> spin blocks for
        UCCSDGPU's registry keys (p, K, Q, bp, c, q)."""
        pl, so = self.pl, self.so
        by_p = {}
        for n, key in enumerate(keys):
            by_p.setdefault(key[0], []).append(n)
        out = [None] * len(keys)
        from electrolites.dlpno import mgpu
        home = cp.cuda.Device().id

        def setup(d):
            w = mgpu.device_view(self, ('S_pao_d', 'Fa_pao_d', 'Fb_pao_d', 'J2c'),
                                 sub={'dfi': ('qia', 'qij', 'J2c')})
            w._chol = {}
            return w

        def work(w, it):
            p, idx = it
            res = w._ring_np_pair(p, idx, keys)
            if cp.cuda.Device().id != home:
                with cp.cuda.Device(home):
                    res = mgpu.replicate(res)
            for n, v in zip(idx, res):
                out[n] = v
        items = sorted(by_p.items(), key=lambda kv: -len(kv[1]))
        mgpu.run_queue(items, work, setup=setup)
        return out

    def _ring_np_pair(self, p, idx, keys):
        pl, so = self.pl, self.so
        nbf, ns = self.orb.nbf, self.ns
        out = {}
        if True:
            aux = self.ribfs[p]
            L = self._halfL(p)
            lm = list(self.lmos[p])
            lpos = {m: r for r, m in enumerate(lm)}
            Bm = self._pair_B(p, lm)
            ga, gl = cp.asarray(aux), cp.asarray(np.asarray(lm, np.int64))
            Boo = _half(L, self.dfi.qij[ga[:, None, None], gl[None, :, None], gl[None, None, :]])
            qs = sorted({keys[n][5] for n in idx})
            ext = su.union(*[self.pao[q] for q in qs])
            epos = -np.ones(nbf, np.int64)
            epos[ext] = np.arange(len(ext))
            ne = len(ext)
            # (Q| k u) over ext PAOs + SOMOs, half fitted
            Qk = cp.zeros((len(aux), len(lm), ne + ns))
            Qk[:, :, :ne] = self.dfi.qia[ga[:, None, None], gl[None, :, None], cp.asarray(ext)[None, None, :]]
            if ns:
                Qk[:, :, ne:] = self.dfi.qij[ga[:, None, None], gl[None, :, None],
                                             cp.asarray(self.somo_idx)[None, None, :]]
            Qk = _half(L, Qk)
            # (Q| B_p u) over ext PAOs (+ SOMOs for beta), half fitted, per spin of B
            Xa, Xbp, Xbs = self._vir_coef_d(p)
            atoms = np.unique(self.ribf_atom[aux])
            E = cp.eye(ne)
            Vx = []
            for s in (0, 1):
                Xs = Xa if s == 0 else Xbp
                v = cp.zeros((len(aux), Xs.shape[1], ne + ns))
                v[:, :, :ne] = self.dfi.q_vv(atoms, self.pao[p], Xs, ext, E)
                if ns:
                    so_ = cp.asarray(self.somo_idx)
                    qsu = self.dfi.qia[ga[:, None, None], so_[None, :, None], cp.asarray(ext)[None, None, :]]
                    qss = self.dfi.qij[ga[:, None, None], so_[None, :, None], so_[None, None, :]]
                    # (Q| B u) gets the SOMO part of B (beta only); (Q| B t) for SOMO t
                    qup = self.dfi.qia[ga[:, None, None], so_[None, :, None], cp.asarray(self.pao[p])[None, None, :]]
                    v[:, :, ne:] = cp.matmul(Xs.T[None], qup.transpose(0, 2, 1))
                    if s == 1:
                        v[:, :, :ne] += cp.matmul(Xbs.T[None], qsu)
                        v[:, :, ne:] += cp.matmul(Xbs.T[None], qss)
                Vx.append(_half(L, v))
            # compact virtual coefficients of every pair on this device (one
            # overlap generator per spin); a group's target coefficients are
            # scattered into the extended domain by one kernel
            gens = self.__dict__.setdefault('_ring_gens', {})
            dv = cp.cuda.Device().id
            if dv not in gens:
                from electrolites.dlpno.ovgen import OverlapGen
                npair = len(self.pl)
                Sd = self.S_pao_d
                gens[dv] = (OverlapGen(Sd, self.pao, [self.Xa[q] for q in range(npair)]),
                            OverlapGen(Sd, self.pao, [self.Xb[q] for q in range(npair)], ne=ns))
            from electrolites.dlpno.ovgen import _scatter_kernel
            posU = cp.asarray(epos.astype(np.int32))
            grp = {}
            for n in idx:
                (_, K, Q, bp, c_, q) = keys[n]
                grp.setdefault((so[Q][0], bp), []).append(n)
            for (qo, bp), lst in grp.items():
                c1 = np.zeros(len(lst))
                c2 = np.zeros(len(lst))
                lk = np.zeros(len(lst), np.int64)
                for r_, n in enumerate(lst):
                    (_, K, Q, _, c_, q) = keys[n]
                    sK, sQ = so[K][1], so[Q][1]
                    lk[r_] = lpos[so[K][0]]
                    c1[r_] = 1. if (sK == c_ and bp == sQ) else 0.
                    c2[r_] = 1. if (sK == sQ and bp == c_) else 0.
                nvp = self.nva(p) if bp == 0 else self.nvb(p)
                qq = np.array([keys[n][5] for n in lst], np.int64)
                cc = np.array([keys[n][4] for n in lst], np.int64)
                ws = np.where(cc == 0, [self.nva(q) for q in qq], [self.nvb(q) for q in qq])
                if not c1.any() and not c2.any():
                    for r_, n in enumerate(lst):
                        out[n] = cp.zeros((nvp, int(ws[r_])))
                    continue
                glk = cp.asarray(lk)
                Z = 0.
                if c1.any():        # (k u|B q) for the domain LMOs k
                    t1 = cp.einsum('Qb,Qku->kbu', Bm[bp][:, lpos[qo]], Qk)
                    Z = Z + t1[glk] * cp.asarray(c1)[:, None, None]
                if c2.any():        # (k q|B u)
                    t2 = cp.einsum('Qk,Qbu->kbu', Boo[:, :, lpos[qo]], Vx[bp])
                    Z = Z - t2[glk] * cp.asarray(c2)[:, None, None]
                Wm = int(ws.max())
                Xs = cp.zeros((len(lst), ne + ns, Wm))
                for c_ in (0, 1):
                    rr = np.where(cc == c_)[0]
                    if not len(rr):
                        continue
                    g = gens[dv][c_]
                    dst = Xs.data.ptr + 8 * rr.astype(np.int64) * (ne + ns) * Wm
                    _scatter_kernel()((len(rr),), (128,),
                                      (cp.asarray(g._src_ptrs(qq[rr])), cp.asarray(dst),
                                       cp.asarray(g.c_ld[qq[rr]].astype(np.int32)),
                                       cp.asarray(np.full(len(rr), Wm, np.int32)),
                                       cp.asarray(g.npao[qq[rr]].astype(np.int32)),
                                       cp.asarray(g.nv[qq[rr]].astype(np.int32)),
                                       cp.asarray(g.off[qq[rr]]), g.pao_flat, posU,
                                       np.int32(ne), np.int32(ns if c_ == 1 else 0)))
                res = cp.matmul(Z, Xs)
                for r_, n in enumerate(lst):
                    out[n] = res[r_, :, :int(ws[r_])]
        return [out[n] for n in idx]

    # ------------------------------------------------------------------
    def lccsd_iterations(self):
        if self.triples:
            self.t0_pool_start()
        t0 = time.perf_counter()
        self.ucc = g = UCCSDGPU(self)
        g.build()
        # the (Q|uv) device cache and the pair metric factors are not needed
        # again before the triples (thousands of small live blocks would keep
        # the pool fragmented)
        self.dfi.cache_clear()
        self._chol = {}
        getattr(self.dfi, '_chol', {}).clear()
        cp.get_default_memory_pool().free_all_blocks()
        self.timings['cc_build'] = time.perf_counter() - t0
        if self.verbose >= 2:
            print('[phase] cc_build steps (s):', {k: round(v, 1) for k, v in g.t_build.items()}, flush=True)
        import os
        if os.environ.get('DLPNO_DEBUG_CC_SUMS'):
            _cc_checksums(g)
            raise SystemExit(0)
        self.log('  CC stores (GB):', getattr(g, 'mem_report', None))
        self.mem_cc = getattr(g, 'mem_report', None)
        self.e_lccsd, self.de_weak = g.kernel(verbose=self.verbose >= 4)
        # amplitudes back for the triples (blocks and dense host copies)
        Th = g.T.to_list()
        self.T2b = {k: Th[n] for n, k in enumerate(g.sop)}
        T1h = g.T1.to_list()
        self.T1 = []
        for I, (x, s) in enumerate(self.so):
            dd = self.pl.ij[x, x]
            t = np.zeros(self.nv(dd))
            t[self._vslice(dd, s)] = T1h[I].ravel()
            self.T1.append(t)
        self.T2 = {}
        for k, blk in self.T2b.items():
            p = self._sp(k)
            self.T2[k] = self._dense(p, self.so[k[0]][1], self.so[k[1]][1], blk)
        # the CC stores (overlaps, ring integrals, DF groups) are not needed by (T)
        self.ucc = g = None
        import gc
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        import os
        if os.environ.get('DLPNO_DEBUG_MEM'):
            alln = [o for o in gc.get_objects() if isinstance(o, cp.ndarray)]
            pool = cp.get_default_memory_pool()
            print('[mem] all live ndarrays:', len(alln), round(sum(o.nbytes for o in alln) / 1e9, 2), 'GB;',
                  'pool used', round(pool.used_bytes() / 1e9, 2), 'total', round(pool.total_bytes() / 1e9, 2),
                  flush=True)
            # group the live arrays by the attribute path of their first referrer
            grp = {}
            for o in alln:
                rs = gc.get_referrers(o)
                key = 'other'
                for r in rs:
                    if isinstance(r, dict):
                        for k, v in r.items():
                            if v is o:
                                key = 'dict:' + str(k)[:30]
                                break
                        if key != 'other':
                            break
                    elif isinstance(r, (list, tuple)):
                        key = type(r).__name__ + f'[{len(r)}]'
                        break
                grp[key] = grp.get(key, 0) + o.nbytes
            for k, v in sorted(grp.items(), key=lambda kv: -kv[1])[:15]:
                print('    ', k, round(v / 1e9, 3), 'GB', flush=True)
            big = [o for o in alln if o.nbytes > (1 << 27)]
            print('[mem] live arrays > 128 MB after LCCSD:', len(big),
                  round(sum(o.nbytes for o in big) / 1e9, 2), 'GB', flush=True)
            for o in sorted(big, key=lambda x: -x.nbytes)[:12]:
                refs = [type(r).__name__ + (':' + ','.join(list(r.keys())[:6]) if isinstance(r, dict) else '')
                        for r in gc.get_referrers(o)[:3]]
                print('   ', o.shape, o.device.id, round(o.nbytes / 1e9, 2), refs, flush=True)


def _transpose_index(lay, src, dst):
    out = []
    combos = {}
    for s_, d_ in zip(src, dst):
        combos.setdefault((lay.group[s_], lay.group[d_]), []).append((lay.slot[s_], lay.slot[d_]))
    for (g1, g2), lst in combos.items():
        out.append((g1, g2, cp.asarray(np.array([l[0] for l in lst], np.int64)),
                    cp.asarray(np.array([l[1] for l in lst], np.int64))))
    return out


def _transpose_inplace(T, tix):
    """T[dst] = T[src]^T in place (source and destination slots disjoint)."""
    lay = T.layout
    for g1, g2, a, b in tix:
        Ms = T.data[g1][a].transpose(0, 2, 1)
        pr, pc = lay.shapes[g2]
        r, c = min(pr, Ms.shape[1]), min(pc, Ms.shape[2])
        blk = cp.zeros((Ms.shape[0], pr, pc))
        blk[:, :r, :c] = Ms[:, :r, :c]
        T.data[g2][b] = blk


def _transpose_items(T, src, dst):
    """T[dst] = T[src]^T (item-wise; layouts of src and dst are transposed)."""
    lay = T.layout
    out = T.copy()
    gs, gd = lay.group, lay.group
    combos = {}
    for s_, d_ in zip(src, dst):
        combos.setdefault((gs[s_], gd[d_]), []).append((lay.slot[s_], lay.slot[d_]))
    for (g1, g2), lst in combos.items():
        a = cp.asarray(np.array([l[0] for l in lst], np.int64))
        b = cp.asarray(np.array([l[1] for l in lst], np.int64))
        Ms = T.data[g1][a].transpose(0, 2, 1)
        pr, pc = lay.shapes[g2]
        r, c = min(pr, Ms.shape[1]), min(pc, Ms.shape[2])
        blk = cp.zeros((len(lst), pr, pc))
        blk[:, :r, :c] = Ms[:, :r, :c]
        out.data[g2][b] = blk
    return out


def _cc_checksums(g):
    """debug: sums of |x| of the CC stores (compare single / multi GPU builds)."""
    def s(x):
        if x is None:
            return 0.
        if isinstance(x, cp.ndarray):
            with cp.cuda.Device(x.device.id):
                return float(cp.abs(x).sum())
        if isinstance(x, np.ndarray):
            return float(np.abs(x).sum())
        if isinstance(getattr(x, 'data', None), list):
            return sum(s(d) for d in x.data)
        if isinstance(x, dict):
            return sum(s(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return sum(s(v) for v in x)
        return 0.
    out = {}
    for name in ('G', 'Kx', 'NP', 'K1Z', 'U0', 'Fk0', 'F1', 'D2', 'D1', 'J1R', 'Sd'):
        out[name] = s(getattr(g, name, None))
    for key in ('Kx', 'K1', 'J1', 'fov', 'ev'):
        out['fix.' + key] = sum(s(g.fix[c].get(key)) for c in g.canon_sp if g.fix[c] is not None)
    out['dfg'] = sum(s({k: v for k, v in d.items()
                        if isinstance(k, tuple) and k[0] in ('Qov', 'Qvv', 'Qmz', 'Qaz', 'fov', 'ev')})
                     for d in g.dfg)
    k3 = getattr(g, 'k3', None)
    out['k3'] = s(getattr(k3, '__dict__', {})) if k3 is not None else 0.
    for sp_ in (0, 1):
        keys = g.Sreg[sp_].keys
        out[f'nS{sp_}'] = float(len(keys))
        out[f'hS{sp_}'] = float(sum((a * 7919 + b) % 1000003 for a, b in keys))
        if g.LS[sp_] is not None:
            out[f'LS{sp_}'] = float(np.sum(g.LS[sp_].rows * g.LS[sp_].cols))
            if g.Sd[sp_] is not None:
                out[f'Sd{sp_}'] = s(g.Sd[sp_])
    print('CCSUMS', {k: f'{v:.12e}' for k, v in out.items()}, flush=True)
