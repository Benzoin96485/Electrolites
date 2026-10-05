"""`fastir` against GPU4PySCF's `eval_ir_freq_intensity`, one process.

The patched driver reuses the Hessian's CPHF solution for the dipole
derivative instead of solving the same equations a second time.  Both
drivers on the same SCF, each with a fresh Hessian object: frequencies and
intensities compared, with the scatter of two *stock* runs printed beside
them (GPU4PySCF's Hessian is not bit-reproducible run to run on every path).
Pass: intensities within 1e-6 km/mol or the stock scatter x 10, frequencies
within 1e-4 cm-1, and the reuse counter advanced (so a run that silently fell
back to the second solve cannot pass).

    python tests/test_ir.py [--quick]
"""
import argparse

import cupy as cp
import numpy as np

import electrolites
electrolites.patch_all()

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft, scf                               # noqa: E402
from gpu4pyscf.properties import ir                          # noqa: E402
from electrolites import fastir                              # noqa: E402

H2O = 'O 0. 0. 0.1174; H -0.757 0. -0.4696; H 0.757 0. -0.4696'
CH3OH = '''C 0.667 0.000 0.000; O -0.667 0.000 0.000
H 1.100 1.000 0.100; H 1.100 -0.500 -0.870; H 1.100 -0.500 0.870
H -1.000 0.900 0.000'''
GLYCINE = 'benchmarks/molecules/glycine.xyz'
CASES = [  # atom, basis, xc, density fit
    (H2O, 'def2-svp', 'b3lyp', False),
    (CH3OH, '6-31g*', 'hf', False),
    (GLYCINE, 'def2-svp', 'b3lyp', False),
    (CH3OH, 'def2-svp', 'wb97m-v', False),
    (GLYCINE, 'def2-svp', 'pbe', True),
]
ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
a = ap.parse_args()

fails = 0
for atom, basis, xc, df in (CASES[:2] if a.quick else CASES):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol) if xc == 'hf' else dft.RKS(mol, xc=xc)
    if df:
        mf = mf.density_fit()
    if xc != 'hf':
        mf.grids.level = 3
    mf.conv_tol = 1e-11
    mf.kernel()
    n0 = fastir.stats()['reused']
    f1, i1 = (cp.asnumpy(x) for x in ir.eval_ir_freq_intensity(mf, mf.Hessian()))
    used = fastir.stats()['reused'] - n0
    f0, i0 = (cp.asnumpy(x) for x in ir._eval_ir_freq_intensity_orig(mf, mf.Hessian()))
    f0b, i0b = (cp.asnumpy(x) for x in ir._eval_ir_freq_intensity_orig(mf, mf.Hessian()))
    df_ = np.abs(f1 - f0).max()
    di = np.abs(i1 - i0).max()
    di0 = np.abs(i0b - i0).max()
    ok = used == 1 and df_ < 1e-4 and di < max(1e-6, 10 * di0)
    fails += not ok
    print(f'{str(atom)[:14]:14s} {basis:9s} {xc:8s} {"DF" if df else "  "} '
          f'd freq {df_:.1e} cm-1  d int {di:.1e} km/mol (stock vs stock '
          f'{di0:.1e}; max int {np.abs(i0).max():.1f})  '
          f'{"ok" if ok else "FAIL"}', flush=True)
print('\nFAILED' if fails else '\nall passed', f'({fails} failures)')
raise SystemExit(1 if fails else 0)
