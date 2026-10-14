"""The SAP initial guess on the GPU (`docs/FEATURE_TARGETS_PYSCF_2.13.md` S8).

PySCF 2.13's ``init_guess='sap'`` (the Gaussian-fit superposition of atomic
potentials, doi:10.1063/5.0004046) is absent from GPU4PySCF 1.8.1: the GPU
SCF classes have no ``init_guess_by_sap`` and the option raises.  The
potential is one three-centre contraction -- every AO pair against the SAP
Gaussian charges of all atoms, summed (PySCF's ``make_sap``) -- which is
GPU4PySCF's ``contract_int3c2e_auxvec`` with a vector of -1.  The guess is
then PySCF's: diagonalise h + V_sap, occupy as RHF, and split the density
in halves for UHF/ROHF (UHF with PySCF's ``init_guess_breaksym``).

    import electrolites
    electrolites.enable('sap')
    mf = gpu4pyscf.dft.UKS(mol, xc='b3lyp').set(init_guess='sap').run()
"""
import numpy as np
import cupy as cp

from pyscf import gto
from pyscf.scf import uhf as uhf_cpu
from gpu4pyscf.scf import hf, uhf, rohf, ghf

from electrolites.features import attach, not_available

SAP_BASIS = 'sapgrasplarge'          # PySCF's default (scf/hf.py:1726)


def _sap_basis(mf, mol):
    from pyscf.gto.basis import load
    sap_basis = getattr(mf, 'sap_basis', SAP_BASIS)
    sapbas = {}
    if isinstance(sap_basis, str):
        for atom in set(a[0] for a in mol._atom):
            bs = load(sap_basis, atom)
            if isinstance(bs, dict):
                sapbas[atom] = np.asarray(bs[atom][0][1:], dtype=float)
            else:
                sapbas[atom] = np.asarray(bs[0][1:], dtype=float)
    else:
        for key in sap_basis:
            sapbas[key] = np.asarray(sap_basis[key][0][1:], dtype=float)
    return sapbas


def make_sap(mol, sapbas):
    """V_sap[p, q] = -sum_A (pq | rho_A^SAP), on the device."""
    from gpu4pyscf.df.int3c2e_bdiv import contract_int3c2e_auxvec
    fakes = []
    for sym, coord in mol._atom:
        expnt, coeff = sapbas[sym][:, 0], sapbas[sym][:, 1]
        fakes.append(gto.mole.fakemol_for_cgtf_charge(
            np.asarray([coord], dtype=float), expnt, coeff))
    auxmol = fakes[0]
    for f in fakes[1:]:
        auxmol = gto.conc_mol(auxmol, f)
    return contract_int3c2e_auxvec(mol, auxmol, -cp.ones(auxmol.nao))


def _rhf_density(mf, mol):
    v = make_sap(mol, _sap_basis(mf, mol))
    h = cp.asarray(mf.get_hcore(mol)) + v
    s = cp.asarray(mf.get_ovlp(mol))
    rhf = hf.RHF(mol)
    from electrolites.fixes import geneig
    e, c = geneig(h, s)
    occ = rhf.get_occ(e, c)
    return rhf.make_rdm1(c, occ)


def init_guess_by_sap_rhf(self, mol=None, **kwargs):
    return _rhf_density(self, mol or self.mol)


def init_guess_by_sap_uhf(self, mol=None, breaksym=None, **kwargs):
    mol = mol or self.mol
    if breaksym is None:
        breaksym = getattr(self, 'init_guess_breaksym', None)
    dm = cp.asnumpy(_rhf_density(self, mol)) * .5
    dma, dmb = uhf_cpu._break_dm_spin_symm(mol, (dm, dm.copy()), breaksym)
    return cp.asarray(np.array((dma, dmb)))


def init_guess_by_sap_rohf(self, mol=None, **kwargs):
    dm = _rhf_density(self, mol or self.mol) * .5
    return cp.stack([dm, dm])


attach(ghf.GHF, 'init_guess_by_sap', not_available(
    'GHF.init_guess_by_sap', 'not ported to GHF (PySCF has none either)'))
attach(rohf.ROHF, 'init_guess_by_sap', init_guess_by_sap_rohf)
attach(uhf.UHF, 'init_guess_by_sap', init_guess_by_sap_uhf)
attach(hf.SCF, 'init_guess_by_sap', init_guess_by_sap_rhf)
