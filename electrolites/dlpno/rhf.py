"""Closed-shell DLPNO-CCSD(T) on the GPU.

The algorithm is the reference one (``ref_rhf``, itself Psi4 1.11's), with
the expensive parts moved to the device:

* three-index integrals: ``dfints.DFInts`` (GPU4PySCF int3c2e, LMO/PAO
  transformation on the device, per-aux-atom PAO/PAO blocks);
* PNO-basis integrals for the CC: per pair on the device;
* LCCSD iterations: ``ccsd_gpu.CCSDGPU`` (batched plans);
* (T0) and iterative (T): ``triples_gpu``.

The local density fitting uses a Cholesky factor of each fitting domain's
metric (cached per domain) where Psi4 uses its inverse square root; every
quantity contracts two fitted factors over the auxiliary index, so the two
are the same up to rounding.
"""
import time

import numpy as np
import cupy as cp
import cupyx.scipy.linalg as cpla

from pyscf import df

from electrolites.dlpno import setup as su
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.dfints import DFInts, ExtDomains
from electrolites.dlpno.ccsd_gpu import CCSDGPU
from electrolites.dlpno.triples_gpu import TriplesGPU
from electrolites.dlpno.lmp2_gpu import LMP2GPU


def _cho_solve(L, x):
    s = x.shape
    y = cpla.solve_triangular(L, x.reshape(L.shape[0], -1), lower=True)
    y = cpla.solve_triangular(L, y, lower=True, trans='T')
    return y.reshape(s)


def _half(L, x):
    s = x.shape
    return cpla.solve_triangular(L, x.reshape(L.shape[0], -1), lower=True).reshape(s)


class DLPNO(LMP2GPU, TriplesGPU, RefDLPNO):
    """GPU DLPNO-CCSD(T).  Same interface as RefDLPNO."""

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
        auxbasis = self.auxbasis or df.make_auxbasis(mol, mp2fit=True)
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
        self.mkn = su.mulliken_atom_pops(mol, S, orb.C_lmo)
        # extended domains large enough for every later step
        n = self.n
        mask = ((self.doi_ij > opt['T_CUT_DO_IJ']) | (np.abs(self.dip_b) > opt['T_CUT_PRE'])
                | np.eye(n, dtype=bool))
        dom = self.domains(min(opt['T_CUT_MKN'], opt['T_CUT_MKN_TRIPLES']),
                           min(opt['T_CUT_DO'], opt['T_CUT_DO_TRIPLES']))
        pairs = [(i, j) for i in range(n) for j in range(n) if mask[i, j]]
        self.ext = ExtDomains(mol.natm, dom.lmo_riatoms, dom.lmo_paos, pairs, orb.nbf)
        self.dfi = DFInts(mol, auxmol, orb.C_lmo, orb.C_pao, self.ext, self.doi_uv,
                          opt['T_CUT_DO_UV'], self.ribf_atom, self.atom_ribfs)
        self.J2c = self.dfi.J2c
        self.S_pao_d = cp.asarray(orb.S_pao)
        t = self._tick('df_ints', t)

    def set_pairs(self, mask, dom):
        super().set_pairs(mask, dom)
        pl = self.pl
        self.riatoms = [np.union1d(dom.lmo_riatoms[i], dom.lmo_riatoms[j]).astype(np.int64)
                        for (i, j) in pl.pairs]

    # ------------------------------------------------------------------
    def _K_pao(self, p):
        i, j = self.pl.pairs[p]
        atoms = self.riatoms[p]
        aux = self.dfi.aux_index(atoms)
        paos = self.pao[p]
        iqa = self.dfi.q_lmo_pao(aux, i, paos)
        jqa = self.dfi.q_lmo_pao(aux, j, paos)
        L = self.dfi.metric_chol(atoms)
        K = (_cho_solve(L, iqa).T @ jqa).get()
        orb = self.orb
        X, e = su.orthocanonicalize(orb.S_pao[np.ix_(paos, paos)],
                                    orb.F_pao[np.ix_(paos, paos)], self.opt['S_CUT'])
        K = X.T @ K @ X
        f = orb.F_lmo
        T = K / (f[i, i] + f[j, j] - e[:, None] - e[None, :])
        return X, e, K, T

    # ------------------------------------------------------------------
    def pno_integrals(self):
        pl = self.pl
        dfi = self.dfi
        npair = len(pl)
        for name in ('K_mibj', 'J_ijmb', 'K_ivvv', 'L_iajb', 'J_ikac', 'K_iakc',
                     'iQk', 'iQa', 'Qma', 'Qab'):
            setattr(self, name, [None] * npair)
        for p, (i, j) in enumerate(pl.pairs):
            if i > j:
                continue
            q = pl.ji[p]
            atoms = self.riatoms[p]
            aux = dfi.aux_index(atoms)
            paos, lm = self.pao[p], self.lmos[p]
            X = cp.asarray(self.X[p])
            na, nl, nv = len(aux), len(lm), X.shape[1]
            ga = cp.asarray(aux)
            q_pair = dfi.qij[ga, i, j]
            q_io = dfi.q_lmo_lmo(aux, i, lm)
            q_jo = dfi.q_lmo_lmo(aux, j, lm)
            q_iv = dfi.q_lmo_pao(aux, i, paos) @ X
            q_jv = dfi.q_lmo_pao(aux, j, paos) @ X
            q_ov = dfi.qia[ga[:, None, None], cp.asarray(lm)[None, :, None],
                           cp.asarray(paos)[None, None, :]] @ X
            q_vv = dfi.q_vv(atoms, paos, X)
            L = dfi.metric_chol(atoms)
            q_io_c, q_jo_c, q_iv_c, q_jv_c = (_cho_solve(L, x) for x in (q_io, q_jo, q_iv, q_jv))
            q_pair, q_io, q_jo, q_iv, q_jv, q_ov, q_vv = (
                _half(L, x) for x in (q_pair, q_io, q_jo, q_iv, q_jv, q_ov, q_vv))
            if self.strong[p]:
                ext = su.union(paos, *[self.dom.lmo_paos[k] for k in lm])
                pos = -np.ones(self.orb.nbf, dtype=np.int64)
                pos[ext] = np.arange(len(ext))
                q_vvp = dfi.q_vv(atoms, paos, X, ext, cp.eye(len(ext)))
                Kio = (q_io_c.T @ q_vvp.reshape(na, -1)).reshape(nl, nv, len(ext))
                Kjo = (q_jo_c.T @ q_vvp.reshape(na, -1)).reshape(nl, nv, len(ext))
                q_ovx = dfi.qia[ga[:, None, None], cp.asarray(lm)[None, :, None],
                                cp.asarray(ext)[None, None, :]]
                Koviv = (q_ovx.reshape(na, -1).T @ q_iv_c).reshape(nl, len(ext), nv)
                Kovjv = (q_ovx.reshape(na, -1).T @ q_jv_c).reshape(nl, len(ext), nv)
                Jp, Kp, Jq, Kq = [], [], [], []
                for kk, k in enumerate(lm):
                    kj, ik = pl.ij[k, j], pl.ij[i, k]
                    Xkj, Xik = cp.asarray(self.X[kj]), cp.asarray(self.X[ik])
                    ckj, cik = cp.asarray(pos[self.pao[kj]]), cp.asarray(pos[self.pao[ik]])
                    Jp.append(Kio[kk][:, ckj] @ Xkj)
                    Kp.append((Xkj.T @ Koviv[kk][ckj]).T)
                    Jq.append(Kjo[kk][:, cik] @ Xik)
                    Kq.append((Xik.T @ Kovjv[kk][cik]).T)
                self.J_ikac[p], self.K_iakc[p] = Jp, Kp
                if q != p:
                    self.J_ikac[q], self.K_iakc[q] = Jq, Kq
            self.K_mibj[p] = q_io.T @ q_jv
            self.J_ijmb[p] = (q_pair @ q_ov.reshape(na, -1)).reshape(nl, nv)
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
    def lccsd_iterations(self):
        t0 = time.perf_counter()
        self.cc = cc = CCSDGPU(self, None)
        self.timings['cc_build'] = time.perf_counter() - t0
        self.e_lccsd, self.de_weak = cc.kernel(verbose=self.verbose >= 4)
        self.T = cc.T.to_list()
        self.T1 = [t.ravel() for t in cc.T1.to_list()]
