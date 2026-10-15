"""The QM/MM Hessian (MM-charge terms) for GPU4PySCF RHF/RKS
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` V2).

GPU4PySCF's ``QMMMSCF`` has a gradient but no Hessian: ``mf.Hessian()``
returned the vacuum Hessian (§3.2's WRONG row; Phase 0 made it raise).
PySCF has none either.  With the MM charges q_I at fixed positions R_I
(point or Gaussian), the QM-atom Hessian gains three terms on top of the
vacuum one:

* electronic second derivatives of h^MM_mn = -sum_I q_I (m|f_I|n):
  sum D_nm d^2 h^MM_mn / dR_A dR_B, from GPU4PySCF's charge-contracted
  ``int1e_grids_ipip1`` (both derivatives on one function: A = B) and
  ``int1e_grids_ipvip1`` (one on each);
* the first-order Fock matrix needs nothing: GPU4PySCF's ``make_h1``
  takes h^(A) from the SCF's own gradient object, which for a QMMMSCF
  already carries dh^MM/dR_A (adding it again was the first build's error,
  found by fitting the finite-difference Hessian term by term);
* the nucleus-MM interaction q_I Z_A f(|R_A - R_I|) (f = 1/r, or
  erf(r/a)/r for Gaussian charges), diagonal in A.

Only the QM-QM block (MM atoms are not displaced).  RHF/RKS.

    mf = gpu4pyscf.qmmm.itrf.mm_charge(gpu_mf, coords, charges).run()
    h = mf.Hessian().kernel()          # (natm, natm, 3, 3)
"""
import numpy as np
import cupy as cp

from pyscf import lib


def _mm(hessobj):
    mm = hessobj.base.mm_mol
    coords = mm.atom_coords()
    charges = mm.atom_charges()
    expnts = mm.get_zetas() if mm.charge_model == 'gaussian' else None
    return mm, coords, charges, expnts


def _dh_mm(mol, coords, charges, expnts):
    """dh^MM/dR_A for every atom, (natm, 3, nao, nao)."""
    from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip1
    ip1 = cp.asarray(int1e_grids_ip1(mol, coords, charges=charges,
                                     charge_exponents=expnts))
    natm, nao = mol.natm, mol.nao
    out = cp.zeros((natm, 3, nao, nao))
    for a, (p0, p1) in enumerate(mol.aoslice_by_atom()[:, 2:]):
        out[a, :, p0:p1] += ip1[:, p0:p1]
    return out + out.transpose(0, 1, 3, 2)


def _d2e_mm(mol, dm, coords, charges, expnts):
    """sum D d^2 h^MM / dR_A dR_B, (natm, natm, 3, 3), symmetrised."""
    from gpu4pyscf.gto.int3c1e_ipip import int1e_grids_ipip1, \
        int1e_grids_ipvip1
    natm = mol.natm
    aoslice = mol.aoslice_by_atom()[:, 2:]
    out = cp.zeros((natm, natm, 3, 3))
    aa = cp.asarray(int1e_grids_ipip1(mol, coords, charges=charges,
                                      charge_exponents=expnts))
    for a, (p0, p1) in enumerate(aoslice):
        out[a, a] += cp.einsum('ij,dDij->dD', dm[p0:p1], aa[:, :, p0:p1])
        out[a, a] += cp.einsum('ij,dDij->dD', dm[:, p0:p1],
                               aa[:, :, p0:p1].transpose(0, 1, 3, 2))
    aa = None
    ab = cp.asarray(int1e_grids_ipvip1(mol, coords, charges=charges,
                                       charge_exponents=expnts))
    for a, (pa0, pa1) in enumerate(aoslice):
        for b, (pb0, pb1) in enumerate(aoslice):
            blk = ab[:, :, pa0:pa1, pb0:pb1]
            out[a, b] += cp.einsum('ij,dDij->dD', dm[pa0:pa1, pb0:pb1], blk)
            out[a, b] += cp.einsum('ij,dDij->dD', dm[pb0:pb1, pa0:pa1],
                                   blk.transpose(0, 1, 3, 2))
    # the MM integrals enter hcore with a minus sign
    return -out


def _hess_nuc_mm(mol, mm_mol):
    """d^2/dR_A^2 of sum_I q_I Z_A f(r_AI), (natm, natm, 3, 3)."""
    from scipy.special import erf
    coords = mm_mol.atom_coords()
    charges = mm_mol.atom_charges()
    z = mol.atom_charges()
    d = mol.atom_coords()[:, None, :] - coords[None, :, :]       # (A, I, 3)
    r = np.linalg.norm(d, axis=2)
    if mm_mol.charge_model == 'gaussian':
        zeta = mm_mol.get_zetas()
        a = np.sqrt(zeta)
        f = erf(a * r) / r
        g = 2 * a / np.sqrt(np.pi) * np.exp(-zeta * r ** 2)
        f1 = (g - f) / r                                           # f'(r)
        # f'' = d/dr[(g - f)/r] = (g' - f')/r - (g - f)/r^2
        f2 = (-2 * zeta * r * g - f1) / r - (g - f) / r ** 2
    else:
        f1 = -1 / r ** 2
        f2 = 2 / r ** 3
    u = d / r[..., None]
    qz = z[:, None] * charges[None, :]
    # grad grad f = f'' u u^T + f'/r (1 - u u^T)
    uu = u[..., :, None] * u[..., None, :]
    eye = np.eye(3)
    h = (f2[..., None, None] * uu
         + (f1 / r)[..., None, None] * (eye - uu))
    blk = np.einsum('AI,AIxy->Axy', qz, h)
    out = np.zeros((mol.natm, mol.natm, 3, 3))
    out[np.arange(mol.natm), np.arange(mol.natm)] = blk
    return out


class QMMMHessMixin:
    """Mixin ahead of GPU4PySCF's RHF/RKS Hessian for a QMMMSCF."""
    __name_mixin__ = 'QMMM'

    def partial_hess_elec(self, *args, **kwargs):
        de2 = super().partial_hess_elec(*args, **kwargs)
        mm, coords, charges, expnts = _mm(self)
        dm = cp.asarray(self.base.make_rdm1())
        return de2 + _d2e_mm(self.mol, dm, coords, charges, expnts)

    def hess_nuc(self, mol=None, atmlst=None):
        if mol is None:
            mol = self.mol
        h = super().hess_nuc(mol, atmlst)
        h_mm = _hess_nuc_mm(mol, self.base.mm_mol)
        if atmlst is not None:
            h_mm = h_mm[np.ix_(atmlst, atmlst)]
        return h + h_mm


def Hessian(mf):
    """``QMMMSCF.Hessian()`` for RHF/RKS."""
    from gpu4pyscf.scf import hf, uhf
    if isinstance(mf, uhf.UHF) or not isinstance(mf, hf.RHF):
        raise NotImplementedError('QM/MM Hessian: RHF/RKS only')
    base_cls = mf.undo_qmmm().Hessian().__class__
    h = mf.undo_qmmm().Hessian()
    h.base = mf
    return lib.set_class(h, (QMMMHessMixin, base_cls))
