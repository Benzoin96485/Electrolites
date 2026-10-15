"""G0W0 (analytic continuation) on the GPU (`docs/FEATURE_TARGETS_PYSCF_2.13.md`
P9).

PySCF's ``gw.gw_ac.GWAC`` -- the self-energy on the imaginary axis from the
density response in the auxiliary basis (eq. 27 of Zhu & Chan, JCTC 17,
727), Pade continuation and the quasiparticle equation -- with its three
host-bound pieces moved to the device:

* the MO three-index tensor L^P_pq from GPU4PySCF's bdiv integrals and the
  Cholesky-fitted metric (the auxiliary basis PySCF would choose: the SCF's
  own when it is density fitted, else mp2fit);
* the frequency loop of ``get_sigma`` (diagonal self-energy): per point one
  SYRK for Pi, one (1 - Pi)^-1, one GEMM with the requested orbitals' L and
  two small GEMMs into Sigma;
* the mean-field V_xc and the exchange self-energy, from the GPU SCF's own
  J/K/XC instead of a host rebuild.

The continuation, the quasiparticle solve and every option are PySCF's.

    gw = electrolites.gw_gpu.GWAC(gpu_mf)
    gw.kernel()
    gw.mo_energy          # quasiparticle energies
"""
import numpy as np
import cupy as cp

from pyscf import lib
from pyscf.gw import gw_ac


def _get_sigma_gpu(gw, orbs, Lpq, quad_freqs, quad_wts, ef, mo_energy,
                   mo_coeff=None, mo_occ=None, iw_cutoff=None,
                   eval_freqs=None, mo_energy_w=None, fullsigma=False):
    """``gw_ac.get_sigma`` (closed shell, diagonal) on the device."""
    if fullsigma or hasattr(gw._scf, 'sigma'):
        return _ORIG_GET_SIGMA(gw, orbs, Lpq, quad_freqs, quad_wts, ef,
                               mo_energy, mo_coeff, mo_occ, iw_cutoff,
                               eval_freqs, mo_energy_w, fullsigma)
    if eval_freqs is None:
        eval_freqs = quad_freqs
    nocc = gw.nocc
    naux, nmo, _ = Lpq.shape
    norbs = len(orbs)
    if mo_energy_w is None:
        mo_energy_w = mo_energy
    if iw_cutoff is not None and gw.rdm is False:
        nw_sigma = int(sum(eval_freqs < iw_cutoff))
    else:
        nw_sigma = len(eval_freqs)
    nw_cutoff = nw_sigma if iw_cutoff is None else int(sum(eval_freqs < iw_cutoff))
    omega = ef + 1j * eval_freqs[:nw_sigma]
    emo = cp.asarray(omega[None] - mo_energy[:, None])            # (nmo, nw)
    L = cp.asarray(Lpq)
    lia = cp.ascontiguousarray(L[:, :nocc, nocc:]).reshape(naux, -1)
    lsl = cp.ascontiguousarray(L[:, :, list(orbs)]).reshape(naux, -1)
    ew = cp.asarray(mo_energy_w)
    eia = (ew[:nocc, None] - ew[None, nocc:]).ravel()
    sigma = cp.zeros((nw_sigma, norbs), dtype=np.complex128)
    eye = cp.eye(naux)
    for w in range(len(quad_freqs)):
        x = 4.0 * eia / (quad_freqs[w] ** 2 + eia ** 2)
        pi = (lia * x[None, :]) @ lia.T
        pinv = cp.linalg.inv(eye - pi) - eye          # (1 - Pi)^-1 - 1
        q = (pinv @ lsl) * lsl
        wmn = q.sum(axis=0).reshape(nmo, norbs)
        g0 = quad_wts[w] * emo / (emo ** 2 + quad_freqs[w] ** 2)
        sigma -= (g0.T @ wmn) / np.pi                  # (nw, norbs)
    sigma = cp.asnumpy(sigma).T.copy()
    return sigma[..., :nw_cutoff], omega[:nw_cutoff]


_ORIG_GET_SIGMA = gw_ac.get_sigma


class GWAC(gw_ac.GWAC):
    """PySCF's G0W0-AC on a GPU4PySCF RHF/RKS reference (closed shell)."""

    def __init__(self, gpu_mf, frozen=None, auxbasis=None):
        self._gpu_mf = gpu_mf
        mf = gpu_mf.to_cpu()
        gw_ac.GWAC.__init__(self, mf, frozen=frozen, auxbasis=auxbasis)

    def ao2mo(self, mo_coeff=None):
        from electrolites.dfmp2_grad import _RI
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        auxmol = self.with_df.auxmol
        if auxmol is None:
            from pyscf import df
            auxmol = df.addons.make_auxmol(self.mol, self.with_df.auxbasis)
        ri = _RI(self.mol, auxmol)
        C = cp.asarray(mo_coeff)
        b = ri.fit(ri.raw(C, C))[0]                       # (p, q, P)
        return cp.asnumpy(cp.ascontiguousarray(b.transpose(2, 0, 1)))

    def get_sigma_exchange(self, mo_coeff):
        g = self._gpu_mf
        dm = g.make_rdm1()
        if g.istype('KohnShamDFT') or not g.istype('RHF'):
            from gpu4pyscf import scf
            vk = -.5 * scf.RHF(self.mol).get_k(self.mol, dm)
        else:
            vk = g.get_veff(g.mol, dm) - g.get_j(g.mol, dm)
        C = cp.asarray(mo_coeff)
        return cp.asnumpy(C.T @ cp.asarray(vk) @ C)

    def kernel(self):
        g = self._gpu_mf
        mf = self._scf
        dm = g.make_rdm1()
        veff = cp.asnumpy(g.get_veff(g.mol, dm))
        vj = cp.asnumpy(g.get_j(g.mol, dm))
        with lib.temporary_env(gw_ac, get_sigma=_get_sigma_gpu), \
                lib.temporary_env(mf, get_veff=lambda *a, **k: veff,
                                  get_j=lambda *a, **k: vj):
            return gw_ac.GWAC.kernel(self)
