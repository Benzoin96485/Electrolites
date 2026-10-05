"""`fastc6` against GPU4PySCF's C6 (catalogue C5), one process.

C6 coefficients and excitation spectra both ways on the same SCF: hybrid,
pure and range-separated functionals, HF, TDA, a molecule with itself and a
pair of different molecules.  Tolerance 1e-8 relative on C6 and 1e-9 Eh on
the excitation energies.

    python tests/test_c6.py
"""
import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fastc6')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft, scf                               # noqa: E402
from gpu4pyscf.properties import c6 as C6                    # noqa: E402
from electrolites import fastc6                              # noqa: E402

H2O = 'O 0. 0. 0.; H 0. -0.757 0.587; H 0. 0.757 0.587'
NH3 = 'N 0 0 0.1; H 0.94 0 -0.27; H -0.47 0.81 -0.27; H -0.47 -0.81 -0.27'
CH3OH = '''C 0.667 0.000 0.000; O -0.667 0.000 0.000
H 1.100 1.000 0.100; H 1.100 -0.500 -0.870; H 1.100 -0.500 0.870
H -1.000 0.900 0.000'''
fails = 0


def mk(atom, xc, basis='def2-svp', tda=False):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol) if xc == 'hf' else dft.RKS(mol, xc=xc)
    mf.conv_tol = 1e-11
    mf.kernel()
    if tda:
        return mf.TDA()
    return mf.TDHF() if xc == 'hf' else mf.TDDFT()


CASES = [(H2O, 'b3lyp', False, None), (CH3OH, 'pbe', False, None),
         (H2O, 'hf', False, None), (NH3, 'wb97x', False, None),
         (CH3OH, 'b3lyp', True, None), (H2O, 'b3lyp', False, NH3)]
for atom, xc, tda, other in CASES:
    ta = mk(atom, xc, tda=tda)
    tb = mk(other, xc, tda=tda) if other else ta
    c1 = C6.calc_c6(ta, tb)
    e1 = np.array(ta.e)
    keep = (C6._solve_full_spectrum, C6.calc_c6)
    C6._solve_full_spectrum = C6._solve_full_spectrum_orig
    C6.calc_c6 = C6._calc_c6_orig
    try:
        c0 = C6.calc_c6(ta, tb)
        e0 = np.array(ta.e)
    finally:
        C6._solve_full_spectrum, C6.calc_c6 = keep
    dc = abs(c1 - c0) / abs(c0)
    de = np.abs(np.sort(e1) - np.sort(e0)).max()
    ok = dc < 1e-8 and de < 1e-9
    fails += not ok
    print(f'{atom[:10]:10s} {xc:6s} {"TDA" if tda else "RPA"} '
          f'{"pair" if other else "self"}: C6 {c1:.6f} vs {c0:.6f} '
          f'(rel {dc:.1e}), max dE {de:.1e}  {"ok" if ok else "FAIL"}',
          flush=True)
print(fastc6.stats())
print('\nFAILED' if fails else '\nall passed', f'({fails} failures)')
raise SystemExit(1 if fails else 0)
