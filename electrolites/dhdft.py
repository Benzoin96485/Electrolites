"""Double-hybrid DFT energies and gradients on the GPU
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` P4).

Neither PySCF 2.13 nor GPU4PySCF 1.8.1 has double hybrids: ``B2PLYP`` is a
``KeyError`` in libxc's name table (the wheel's libxc is built without its
double-hybrid entries), and the manual recipe -- a hybrid RKS plus a scaled
MP2 on its orbitals -- has no gradient anywhere (PySCF's ``grad/mp2.py`` on a
KS reference is not a double-hybrid gradient: its Z-vector uses the HF
response).  Here:

* the energy is a GPU RKS with the functional's DFT part, then GPU4PySCF's
  DF-MP2 on its orbitals, scaled per spin component:
  E = E_KS + c_os E_os + c_ss E_ss;
* the gradient is ``electrolites.dfmp2_grad`` on that RKS: the KS response
  (fxc and the scaled exact exchange) in the Z-vector, the XC skeleton
  derivative contracted with the relaxed density, and the scaled amplitudes
  throughout.  Global hybrids, LDA/GGA parts; DF or exact SCF.

    from electrolites.dhdft import DoubleHybrid
    dh = DoubleHybrid(mol, 'B2PLYP').run()
    g = dh.Gradients().kernel()

``PARAMETERS`` lists the built-in functionals (DFT part as a PySCF xc string,
c_os, c_ss); any other is ``DoubleHybrid(mol, (xc, c_os, c_ss))``.
"""
import numpy as np
import cupy as cp

from pyscf import lib
from pyscf.lib import logger

PARAMETERS = {
    # Grimme, J. Chem. Phys. 124, 034108 (2006)
    'B2PLYP': ('0.53*HF + 0.47*B88, 0.73*LYP', .27, .27),
    # Karton et al., J. Phys. Chem. A 112, 12868 (2008)
    'B2GPPLYP': ('0.65*HF + 0.35*B88, 0.64*LYP', .36, .36),
    # Schwabe and Grimme, Phys. Chem. Chem. Phys. 8, 4398 (2006)
    'MPW2PLYP': ('0.55*HF + 0.45*GGA_X_MPW91, 0.75*LYP', .25, .25),
    # Bremond and Adamo, J. Chem. Phys. 135, 024106 (2011)
    'PBE0-DH': ('0.5*HF + 0.5*PBE, 0.875*PBE', .125, .125),
    # Bremond et al., J. Chem. Phys. 141, 031101 (2014)
    'PBE-QIDH': ('0.693361*HF + 0.306639*PBE, 0.666667*PBE',
                 .333333, .333333),
}


class DoubleHybrid(lib.StreamObject):
    """A double-hybrid calculation: ``_scf`` (GPU RKS), ``mp`` (GPU DF-MP2 on
    its orbitals), ``e_tot``, ``e_corr`` (the scaled MP2 part)."""

    def __init__(self, mol, xc='B2PLYP', density_fit=True, auxbasis=None,
                 ri_auxbasis=None, frozen=None):
        from gpu4pyscf import dft
        if isinstance(xc, str):
            key = xc.upper().replace('-', '').replace('_', '')
            table = {k.replace('-', ''): v for k, v in PARAMETERS.items()}
            if key not in table:
                raise KeyError(f'unknown double hybrid {xc!r}; give '
                               '(xc, c_os, c_ss)')
            xc_dft, self.c_os, self.c_ss = table[key]
            self.name = xc
        else:
            xc_dft, self.c_os, self.c_ss = xc
            self.name = xc_dft
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        mf = dft.RKS(mol, xc=xc_dft)
        if density_fit:
            mf = mf.density_fit(auxbasis=auxbasis)
        self._scf = mf
        self.ri_auxbasis = ri_auxbasis
        self.frozen = frozen
        self.mp = None
        self.e_corr = None

    def kernel(self):
        from gpu4pyscf.mp import dfmp2
        mf = self._scf
        if mf.mo_coeff is None or not mf.converged:
            mf.kernel()
        mp = dfmp2.DFMP2(mf, frozen=self.frozen, auxbasis=self.ri_auxbasis)
        mp.ps, mp.pt = self.c_os, self.c_ss
        mp.kernel()
        self.mp = mp
        self.e_corr = (self.c_os * float(mp.e_corr_os)
                       + self.c_ss * float(mp.e_corr_ss))
        logger.note(self, '%s: E_KS = %.12f  E_corr(scaled) = %.12f  '
                    'E = %.12f', self.name, mf.e_tot, self.e_corr,
                    self.e_tot)
        return self.e_tot

    @property
    def e_tot(self):
        return float(self._scf.e_tot) + self.e_corr

    def Gradients(self):
        from electrolites.dfmp2_grad import Gradients
        if self.mp is None:
            self.kernel()
        return Gradients(self.mp)

    nuc_grad_method = Gradients
