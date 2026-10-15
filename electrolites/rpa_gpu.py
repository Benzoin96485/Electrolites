"""Direct RPA correlation energy on the GPU (`docs/FEATURE_TARGETS_PYSCF_2.13.md`
P8).

PySCF's ``gw.rpa.RPA`` (dRPA on a mean-field reference, 40-point scaled
Gauss-Legendre imaginary-frequency grid) runs on the host and re-reads the
(ov|P) tensor from HDF5 at every frequency; GPU4PySCF has no RPA.  Here the
same energy expression,

    E_c = sum_w  w_k / 2pi  [ ln det(1 - Pi(iw)) + tr Pi(iw) ],
    Pi_PQ(iw) = sum_ia B^P_ia chi0_ia(w) B^Q_ia,
    chi0 = 2 f e / (w^2 + e^2)   (e = e_i - e_a, f = f_i - f_a),

with the fitted B from GPU4PySCF's bdiv integrals (as PySCF's
DFMP2-derived class: the SCF's own auxiliary basis on a density-fitted
reference, else mp2fit) resident on the device: per frequency one SYRK
(1 - Pi = 1 + X^T X, X = |chi0|^1/2 B, positive definite) and a Cholesky
log-determinant.  E_hf is the exact-exchange (HF) energy of the reference
orbitals with the same fitting, as PySCF's ``get_e_hf``.

    rpa = electrolites.rpa_gpu.RPA(gpu_mf).run()
    rpa.e_corr, rpa.e_tot
"""
import numpy as np
import cupy as cp

from pyscf import lib
from pyscf.lib import logger


class RPA(lib.StreamObject):
    """dRPA on a GPU4PySCF RHF/RKS reference (closed shell)."""

    def __init__(self, mf, frozen=None, auxbasis=None):
        from pyscf import df
        self._scf = mf
        self.mol = mf.mol
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.frozen = frozen
        if auxbasis is None:
            # PySCF's DFMP2-derived RPA: the SCF's own fitting basis on a
            # density-fitted reference, else the mp2fit one
            gdf = getattr(mf, 'with_df', None)
            if gdf is not None:
                auxbasis = gdf.auxbasis or df.make_auxbasis(self.mol)
            else:
                auxbasis = df.make_auxbasis(self.mol, mp2fit=True)
        self.auxbasis = auxbasis
        self.e_hf = self.e_corr = None

    @property
    def e_tot(self):
        return self.e_hf + self.e_corr

    def _split(self):
        mf = self._scf
        occ = cp.asnumpy(mf.mo_occ)
        nocc = int(np.count_nonzero(occ > 0))
        nfrz = 0 if self.frozen is None else int(self.frozen)
        C = cp.asarray(mf.mo_coeff)
        e = cp.asarray(mf.mo_energy)
        f = cp.asarray(occ)
        return (C[:, nfrz:nocc], C[:, nocc:], e[nfrz:nocc], e[nocc:],
                f[nfrz:nocc], f[nocc:])

    def get_e_hf(self):
        from gpu4pyscf import scf
        mf = scf.RHF(self.mol).density_fit(auxbasis=self.auxbasis)
        dm = self._scf.make_rdm1()
        return float(mf.energy_elec(dm=cp.asarray(dm))[0]) + \
            float(mf.energy_nuc())

    def kernel(self, nw=40, x0=0.5):
        from pyscf import df
        from pyscf.gw.rpa import _get_scaled_legendre_roots
        from electrolites.dfmp2_grad import _RI
        log = logger.new_logger(self)
        t0 = (logger.process_clock(), logger.perf_counter())
        if self.e_hf is None:
            self.e_hf = self.get_e_hf()
        co, cv, eo, ev, fo, fv = self._split()
        auxmol = df.addons.make_auxmol(self.mol, self.auxbasis)
        ri = _RI(self.mol, auxmol)
        b = ri.fit(ri.raw(co, cv))[0]                       # (i, a, P)
        naux = b.shape[2]
        b = b.reshape(-1, naux)
        e_ov = (eo[:, None] - ev[None, :]).ravel()
        f_ov = (fo[:, None] - fv[None, :]).ravel()
        if float(-e_ov.max()) < 1e-3:
            log.warn('RPA is not well defined for (near-)degenerate systems')
        b2 = (b * b).sum(axis=1)                            # sum_P B^2
        e_corr = 0.
        eye = cp.eye(naux)
        for omega, wt in zip(*_get_scaled_legendre_roots(nw, x0)):
            chi0 = 2 * e_ov * f_ov / (omega ** 2 + e_ov ** 2)   # <= 0
            x = b * cp.sqrt(-chi0)[:, None]
            m = eye + x.T @ x                                # 1 - Pi
            logdet = 2 * float(cp.log(cp.diag(cp.linalg.cholesky(m))).sum())
            tr = float((chi0 * b2).sum())                    # tr Pi
            e_corr += wt / (2 * np.pi) * (logdet + tr)
        self.e_corr = e_corr
        log.timer('GPU dRPA', *t0)
        log.note('E(RPA) = %.15g  E_corr = %.15g  E_hf = %.15g',
                 self.e_tot, self.e_corr, self.e_hf)
        return self.e_corr

    run = lambda self, *a, **k: (self.kernel(*a, **k), self)[1]   # noqa: E731
