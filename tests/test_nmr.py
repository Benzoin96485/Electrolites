"""`fastnmr` against GPU4PySCF's `shielding.nr_rks` (catalogue C7), one process.

The GIAO XC term element-wise on the same SCF (and ``get_vxc`` with the
GIAO two-electron term on the GPU against upstream's host ``int2e_ig1``) (LDA, GGA, hybrid GGA;
organic, a third-row atom, def2-TZVPP which reaches f; a coarse grid), then the
whole shielding tensor both ways.  Round-off: 1e-11 relative on the matrix,
1e-8 ppm on the shielding.

    python tests/test_nmr.py [--quick]
"""
import argparse
import os

# the GIAO two-electron term on the GPU even for these small molecules
os.environ['FASTNMR_GPU2E'] = '1'

import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fastnmr')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.properties import shielding as SH             # noqa: E402
from electrolites import fastnmr                             # noqa: E402

H2O = 'O 0. 0. 0.; H 0. -0.757 0.587; H 0. 0.757 0.587'
HCL = 'H 0. 0. 0.; Cl 0. 0. 1.28'
GLY = 'benchmarks/molecules/glycine.xyz'
CASES = [
    (H2O, 'def2-svp', 'svwn', 3), (H2O, 'def2-tzvpp', 'pbe', 3),
    (HCL, 'def2-svp', 'b3lyp', 3), (GLY, '6-31g*', 'b3lyp', 3),
    (GLY, 'def2-svp', 'pbe0', 1),
]
ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
a = ap.parse_args()
fails = 0
for atom, basis, xc, level in (CASES[:3] if a.quick else CASES):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = level
    mf.conv_tol = 1e-11
    mf.kernel()
    dm = mf.make_rdm1()
    v1 = SH.nr_rks(mf._numint, mol, mf.grids, xc, dm)
    v0 = SH._nr_rks_orig(mf._numint, mol, mf.grids, xc, dm)
    ev = float(cp.abs(v1 - v0).max() / cp.abs(v0).max())
    s1 = [cp.asnumpy(cp.asarray(x)) for x in SH.eval_shielding(mf)]
    SH.nr_rks = SH._nr_rks_orig
    s0 = [cp.asnumpy(cp.asarray(x)) for x in SH.eval_shielding(mf)]
    SH.nr_rks = fastnmr.nr_rks
    es = max(np.abs(a_ - b_).max() for a_, b_ in zip(s1, s0))
    # the GIAO two-electron term on the GPU (electrolites.giao_jk)
    v0 = SH._get_vxc_orig(mf, dm)
    ej = float(cp.abs(SH.get_vxc(mf, dm) - v0).max() / cp.abs(v0).max())
    ok = ev < 1e-11 and es < 1e-8 and ej < 1e-11
    fails += not ok
    print(f'{atom.rsplit("/", 1)[-1][:12]:12s} {basis:10s} {xc:6s} L{level} '
          f'dV {ev:.1e}  dVxc {ej:.1e}  d sigma {es:.1e} ppm  {"ok" if ok else "FAIL"}',
          flush=True)
print('\nFAILED' if fails else '\nall passed', f'({fails} failures)')
raise SystemExit(1 if fails else 0)
