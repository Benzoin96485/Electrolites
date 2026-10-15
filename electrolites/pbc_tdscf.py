"""Gamma-point TDA excitation energies for periodic RKS/RHF on the GPU
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` X1).

GPU4PySCF 1.8.1 has no periodic TDDFT: ``pyscf.pbc.tdscf.TDA(gpu_mf)`` is an
AttributeError, the molecular GPU TDA on a Gamma-point object fails (the
PBC ``gen_response`` takes one density), and that response has a latent
argument-order bug in its singlet branch (D10: ``singlet`` lands in
``nr_rks_fxc``'s ``fxc`` slot).  This is PySCF's ``pbc.tdscf.rhf.TDA`` --
the molecular TDA operator with the exchange evaluated with ``exxdiv=None``
and orbital energies from that Fock -- built on the GPU periodic SCF's own
pieces, per trial vector:

    A z = (e_a - e_i) z + C_o^T v1[D] C_v,  D = 2 C_o z C_v^T,
    v1 = fxc_{singlet|triplet}[D/2] + (J[D] if singlet) - hyb/2 K[D],

J and K from the cell's GPU J/K (``exxdiv=None``), fxc from GPU4PySCF's
periodic ``nr_rks_fxc_st`` on the SCF grid (the kernel cached once), and
the lowest roots by the block Davidson of ``electrolites.stability``.

    td = electrolites.pbc_tdscf.TDA(gpu_cell_mf).run(nstates=5)
    td.e        # excitation energies (Hartree)
"""
import numpy as np
import cupy as cp

from pyscf import lib
from pyscf.lib import logger


class TDA(lib.StreamObject):
    """Gamma-point TDA on a GPU4PySCF periodic RKS/RHF."""

    def __init__(self, mf, singlet=True):
        self._scf = mf
        self.cell = mf.cell
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.singlet = singlet
        self.nstates = 3
        self.conv_tol = 1e-5
        self.e = self.xy = None

    def _orbital_energies(self):
        """mo_energy from the Fock matrix with exxdiv=None (PySCF's
        ``scf.addons.mo_energy_with_exxdiv_none``)."""
        mf = self._scf
        C = cp.asarray(mf.mo_coeff)
        with lib.temporary_env(mf, exxdiv=None):
            dm = mf.make_rdm1()
            f = mf.get_fock(dm=dm)
        return cp.diag(C.T @ cp.asarray(f) @ C)

    def gen_vind(self):
        mf = self._scf
        cell = self.cell
        ni = mf._numint
        C = cp.asarray(mf.mo_coeff)
        occ = cp.asnumpy(mf.mo_occ)
        co = C[:, occ == 2]
        cv = C[:, occ == 0]
        no, nv = co.shape[1], cv.shape[1]
        e = self._orbital_energies()
        e_ia = (e[occ == 0][None, :] - e[occ == 2][:, None]).ravel()
        kpts = mf.kpt.reshape(1, 3)
        is_ks = hasattr(mf, 'xc')
        if is_ks:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=0)
            if omega != 0:
                raise NotImplementedError('PBC TDA: range-separated hybrids')
            dm0 = mf.make_rdm1()
            fxc = ni.cache_xc_kernel1(cell, mf.grids, mf.xc, dm0[None], 1,
                                      kpts, is_rhf=True)[2]
        else:
            hyb = 1.
        singlet = self.singlet

        def vresp(dm):
            # the density (fxc, J) sees only the symmetric part of D
            ds = .5 * (dm + dm.T)
            v1 = cp.zeros_like(dm)
            if is_ks:
                v1 += cp.asarray(ni.nr_rks_fxc_st(
                    cell, mf.grids, mf.xc, dm0, (.5 * ds)[None, None], 1,
                    singlet, fxc, kpts)).reshape(dm.shape)
            with lib.temporary_env(mf, exxdiv=None):
                if singlet:
                    v1 += cp.asarray(mf.get_j(cell, ds, 1)).reshape(dm.shape)
                if hyb != 0:
                    v1 -= .5 * hyb * cp.asarray(
                        mf.get_k(cell, dm, 0)).reshape(dm.shape)
            return v1

        def vind(zs):
            zs = cp.asarray(zs).reshape(-1, no, nv)
            out = cp.empty_like(zs)
            for k, z in enumerate(zs):
                dm = 2 * co @ z @ cv.T
                out[k] = co.T @ vresp(dm) @ cv
            out += zs * e_ia.reshape(no, nv)[None]
            return out.reshape(len(zs), -1)
        return vind, e_ia

    def kernel(self, nstates=None):
        from electrolites.stability import _davidson
        if nstates is not None:
            self.nstates = nstates
        log = logger.new_logger(self)
        vind, hdiag = self.gen_vind()
        x0 = cp.zeros_like(hdiag)
        x0[int(cp.argmin(hdiag))] = 1.
        w, x = _davidson(vind, hdiag, x0, self.nstates, self.conv_tol ** 2,
                         log)
        self.e = np.asarray(w)
        self.xy = x
        log.note('GPU PBC TDA (%s) excitation energies (eV): %s',
                 'singlet' if self.singlet else 'triplet',
                 self.e * 27.211386245988)
        return self.e

    def run(self, nstates=None):
        self.kernel(nstates)
        return self
