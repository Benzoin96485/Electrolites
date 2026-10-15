"""DF-CASSCF analytic nuclear gradients on the GPU
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` M6).

PySCF's ``df/grad/casscf.py`` term by term, on the object
``electrolites.casscf_gpu`` builds:

* the generalised Fock matrix (for the energy-weighted density) from the
  fitted integrals the CASSCF already holds (``GPUDF.lmo``) and its J/K;
* the h1/overlap terms and the core-core / core-active J/K derivatives
  through GPU4PySCF's DF RHF gradient machinery (``_hcore_energy``,
  ``jk_energy_per_atom`` on the CASSCF's own auxiliary basis):
  E_G(D_c + D_a) - E_G(D_a) is the core-core plus core-active part;
* the active-active part 1/2 sum dm2_uvwx (uv|wx)_DF through the general
  three-index derivative kernel of ``electrolites.dfmp2_grad`` with
  Gamma^P_uv = 1/2 sum_wx dm2_uvwx c^P_wx (PySCF's ``solve_df_rdm2``).

A single-state CASSCF is variational in orbitals and CI, so no response
equations.  State-averaged gradients and non-adiabatic couplings (which need
PySCF's Lagrangian) are ``electrolites.sacasscf_grad``, reached through
``mc.nuc_grad_method()`` / ``mc.nac_method()`` of a GPU SA-CASSCF.

    mc = electrolites.casscf_gpu.CASSCF(gpu_mf, 8, 8, density_fit=True).run()
    g = mc.nuc_grad_method().kernel()
"""
import numpy as np
import cupy as cp

from pyscf import lib
from pyscf.lib import logger


def _closed(mol):
    """A closed-shell copy of ``mol`` (same atoms and basis) for GPU4PySCF's
    RHF gradient machinery, which only the geometry and basis enter."""
    if mol.spin == 0:
        return mol
    m = mol.copy()
    m.charge += mol.nelectron % 2
    m.spin = 0
    return m.build(False, False)


class _Kit:
    """The derivative machinery of one DF-CASSCF geometry: GPU4PySCF's DF RHF
    gradient object on the CASSCF's auxiliary basis (h1/S terms and J/K
    derivatives) and the general three-index kernel (active-active)."""

    def __init__(self, mc):
        from electrolites.dfmp2_grad import _RI, _AO3c
        from gpu4pyscf import scf
        mol = mc.mol
        gdf = mc.with_df
        mf = scf.RHF(_closed(mol)).density_fit(auxbasis=gdf.auxbasis)
        mf.verbose = 0
        mf.with_df.build()
        self.g = mf.nuc_grad_method()
        self.ri = _RI(mol, gdf.auxmol)
        self.k3 = _AO3c(mol, self.ri.auxmol)

    def hcore(self, dm, dme):
        """h^x . dm - S^x . dme (PySCF's convention for dme)."""
        return cp.asnumpy(self.g._hcore_energy(dm, dme))

    def eg(self, dm):
        """d/dx of E_G(dm) = 1/2 dm . (J - K/2)[dm], skeleton."""
        if float(abs(dm).max()) < 1e-9:
            # GPU4PySCF factorises dm, drops eigenvalues below 1e-8 and fails
            # on what is left of rank 0 (a transition
            # density between states of different symmetry is zero)
            return np.zeros((self.g.mol.natm, 3))
        return cp.asnumpy(self.g.jk_energy_per_atom(
            dm, j_factor=1, k_factor=1, hermi=1))

    def bilinear(self, x, y):
        """d/dx of x . (J - K/2)[y], skeleton (polarisation)."""
        return .5 * (self.eg(x + y) - self.eg(x - y))

    def fit(self, c1, c2):
        return self.ri.fit(self.ri.raw(c1, c2))[1]          # (i, j, P)

    def three(self, cl, cr, c, gam):
        """d/dx of 2 sum (ij|P) gam[i,j,P] - sum (P|Q) Gt, Gt = c^T gam."""
        naux = c.shape[2]
        gt = c.reshape(-1, naux).T @ gam.reshape(-1, naux)
        gamk = self.k3.to_kernel(cp.ascontiguousarray(gam.transpose(2, 0, 1)))
        gtk = self.k3.to_kernel(self.k3.to_kernel(gt).T).T
        return self.k3.grad(cl, cr, gamk, gtk)

    def active(self, Ca, dm2):
        """d/dx of 1/2 sum dm2_uvwx (uv|wx)_DF."""
        n = Ca.shape[1]
        c = self.fit(Ca, Ca)
        gam = .5 * (dm2.reshape(n * n, -1) @ c.reshape(n * n, -1))
        return self.three(Ca, Ca, c, gam.reshape(n, n, -1))

    def active_rotated(self, Ca, CaL, dm2):
        """d/dx of 2 sum dm2_uvwx (u'v|wx)_DF, u' = CaL (the orbital
        Lagrange term's four index substitutions, dm2 being symmetric)."""
        n = Ca.shape[1]
        c0 = self.fit(Ca, Ca)
        c1 = self.fit(CaL, Ca)
        d2 = dm2.reshape(n * n, n * n)
        gam_a = (d2 @ c0.reshape(n * n, -1)).reshape(n, n, -1)
        gam_b = (d2.T @ c1.reshape(n * n, -1)).reshape(n, n, -1)
        return (self.three(CaL, Ca, c1, gam_a)
                + self.three(Ca, Ca, c0, gam_b))


def _gfock(mc, C, dm1a, dm2, Dc, Da, nrow):
    """PySCF's generalised Fock gfock[:nrow, :nocc] on the device."""
    gdf = mc.with_df
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    Cc, Ca, Cr = C[:, :ncore], C[:, ncore:nocc], C[:, :nrow]
    vj, vk = gdf.get_jk(cp.asnumpy(cp.stack([Dc, Da])), hermi=1)
    vj, vk = cp.asarray(vj), cp.asarray(vk)
    vhf_c = vj[0] - .5 * vk[0]
    vhf_a = vj[1] - .5 * vk[1]
    h1 = cp.asarray(mc.get_hcore())
    gfock = cp.zeros((nrow, nocc))
    gfock[:, :ncore] = Cr.T @ (h1 + vhf_c + vhf_a) @ Cc * 2
    gfock[:, ncore:] = Cr.T @ (h1 + vhf_c) @ Ca @ dm1a
    gfock[:, ncore:] += _two_e_fock(gdf, Cr, Ca, dm2)
    return gfock


def _two_e_fock(gdf, Cr, Ca, dm2):
    """sum_uvw (uv|rw) dm2[v,u,w,t] (DF), (nrow, ncas)."""
    ncas = Ca.shape[1]
    laa = cp.asarray(gdf.lmo(Ca, Ca))                       # (u, v, Q)
    lra = cp.asarray(gdf.lmo(Cr, Ca))                       # (r, w, Q)
    naux = laa.shape[2]
    x = (laa.reshape(ncas * ncas, naux).T
         @ dm2.transpose(1, 0, 2, 3).reshape(ncas * ncas, -1))
    x = x.reshape(naux, ncas, ncas)                         # (Q, w, t)
    return lra.transpose(0, 2, 1).reshape(Cr.shape[1], -1) @ x.reshape(-1, ncas)


def grad_elec(mc, mo_coeff=None, ci=None, verbose=None, kit=None):
    """PySCF ``df/grad/casscf.grad_elec`` on the GPU (no nuclear term).
    With ``mc._tag_gfock_ov_nonzero`` (SA-CASSCF's state-specific part) the
    generalised Fock spans all MO rows, as PySCF's."""
    log = logger.new_logger(mc, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    if getattr(mc, 'frozen', None) is not None:
        raise NotImplementedError('CASSCF gradient: frozen orbitals')
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nocc = ncore + ncas
    C = cp.asarray(mo_coeff)
    nrow = C.shape[1] if getattr(mc, '_tag_gfock_ov_nonzero', False) else nocc
    Cc, Ca = C[:, :ncore], C[:, ncore:nocc]
    casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)
    dm1a = cp.asarray(casdm1)
    dm2 = cp.asarray(casdm2)
    Dc = 2 * Cc @ Cc.T
    Da = Ca @ dm1a @ Ca.T
    gfock = _gfock(mc, C, dm1a, dm2, Dc, Da, nrow)
    Cr = C[:, :nrow]
    dme0 = Cr @ gfock @ C[:, :nocc].T
    dme0 = .5 * (dme0 + dme0.T)
    if kit is None:
        kit = _Kit(mc)
    de = kit.hcore(Dc + Da, dme0)
    de += kit.eg(Dc + Da) - kit.eg(Da)          # E_G(Dc) + Dc.G[Da]
    de += kit.active(Ca, dm2)
    log.timer('GPU DF-CASSCF gradient', *t0)
    return de


class Gradients(lib.StreamObject):
    """``mc.nuc_grad_method()`` of a GPU DF-CASSCF (Hartree/Bohr)."""

    def __init__(self, mc):
        self.base = mc
        self.mol = mc.mol
        self.verbose = mc.verbose
        self.stdout = mc.stdout
        self.de = None

    def grad_nuc(self):
        from gpu4pyscf import scf
        return scf.RHF(_closed(self.mol)).nuc_grad_method().grad_nuc()

    def kernel(self, mo_coeff=None, ci=None):
        mc = self.base
        if mc.ci is None:
            mc.kernel()
        from pyscf.mcscf.addons import StateAverageMCSCFSolver
        from pyscf.mcscf import mc1step
        if not isinstance(mc, mc1step.CASSCF):
            raise NotImplementedError(
                'GPU CASCI gradient: CASCI is not variational in the '
                'orbitals and needs a Z-vector; not implemented')
        if isinstance(mc, StateAverageMCSCFSolver):
            raise NotImplementedError(
                'GPU CASSCF gradient: state-averaged CASSCF needs the '
                'Lagrangian; not implemented')
        self.de = grad_elec(mc, mo_coeff, ci, self.verbose) + self.grad_nuc()
        return self.de

    grad = kernel
