"""`fastvnlch` against GPU4PySCF's `_get_vnlc_deriv1`, one process.

The VV10 Fock-matrix derivative of `make_h1`, element-wise, on the same SCF:
closed shell (two functionals, two basis sets), open shell (the two-spin
tuple), and a density-fitted Hessian object (whose module binds the name at
import).  The batched contraction reassociates the grid and atom sums, so
the tolerance is round-off: 1e-11 of the largest element.  Then the stats
counter, so that a run that never reached the batched code cannot pass.

    python tests/test_vnlch.py
"""
import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fastvnlch')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.hessian import rks as HR                      # noqa: E402
from electrolites import fastvnlch                           # noqa: E402

H2O = 'O 0. 0. 0.; H 0. -0.757 0.587; H 0. 0.757 0.587'
CH3OH = '''C 0.667 0.000 0.000; O -0.667 0.000 0.000
H 1.100 1.000 0.100; H 1.100 -0.500 -0.870; H 1.100 -0.500 0.870
H -1.000 0.900 0.000'''
CH3 = 'C 0 0 0; H 1.08 0 0; H -0.54 0.935 0; H -0.54 -0.935 0'
GLYCINE = 'benchmarks/molecules/glycine.xyz'
CASES = [  # atom, spin, basis, xc, density fit
    (H2O, 0, 'def2-svp', 'wb97m-v', False),
    (CH3OH, 0, '6-31g*', 'b97m-v', False),
    (GLYCINE, 0, 'def2-svp', 'wb97x-v', False),
    (CH3, 1, 'def2-svp', 'wb97m-v', False),
    (CH3OH, 0, 'def2-svp', 'wb97m-v', True),
]
fails = 0
for atom, spin, basis, xc, df in CASES:
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc)
    if df:
        mf = mf.density_fit()
    mf.grids.level = 2
    mf.nlcgrids.level = 2
    mf.conv_tol = 1e-11
    mf.kernel()
    h = mf.Hessian()
    mo, occ = cp.asarray(mf.mo_coeff), cp.asarray(mf.mo_occ)
    ref = HR._get_vnlc_deriv1_orig(h, mo, occ, 4000)
    n0 = fastvnlch.stats()['calls']
    got = HR._get_vnlc_deriv1(h, mo, occ, 4000)
    used = fastvnlch.stats()['calls'] - n0
    if spin:
        assert isinstance(got, tuple) and len(got) == 2
        e = max(float(cp.abs(g - r).max() / cp.abs(r).max())
                for g, r in zip(got, ref))
    else:
        e = float(cp.abs(got - ref).max() / cp.abs(ref).max())
    ok = e < 1e-11 and used == 1
    fails += not ok
    print(f'{str(atom)[:16]:16s} spin {spin} {basis:9s} {xc:8s} '
          f'{"DF " if df else "   "} rel {e:.2e}  {"ok" if ok else "FAIL"}',
          flush=True)
print('\nFAILED' if fails else '\nall passed', f'({fails} failures)')
raise SystemExit(1 if fails else 0)
