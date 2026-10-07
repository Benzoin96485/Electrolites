"""Do the range-separated derivatives keep GPU4PySCF's numbers?  (B16)

``electrolites/fastgrad.py`` rewrites ``grad/uks.py:energy_ee``'s exchange as
``hyb*K(1/r) + (alpha-hyb)*K(erf)`` in place of GPU4PySCF's
``alpha*K(1/r) + (hyb-alpha)*K(erfc)``, so that both passes run on
``fastejk``'s kernels.  Against GPU4PySCF's own ``energy_ee``
(``Gradients._energy_ee_orig``) on the same converged SCF, per atom, for every
case GPU4PySCF's split distinguishes: a global hybrid, SR and LR exchange
with different weights (wB97X, CAM-B3LYP), SR only (HSE06), LR only
(LC-wPBE) -- on an open-shell organic radical and an open-shell
transition-metal complex.

The Hessians' ``partial_hess_elec`` (RKS and UKS) is rewritten the same way
for its ip2 term and is checked against ``_partial_hess_elec_orig`` on the
same Hessian object, for the same functionals on ethanol (RKS) and
ethanol+ (UKS).

    python tests/test_rsh_deriv.py [--basis def2-svp]
"""
import argparse
import os
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--tol', type=float, default=1e-8,
                help='largest allowed difference, Eh/bohr')
a = ap.parse_args()

import cupy as cp                                            # noqa: E402
from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402

import electrolites                                          # noqa: E402
electrolites.patch_all()
from gpu4pyscf.grad import uks as guks                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MOLS = [('ethanol+', '''C -0.748 -0.015 0.024; C 0.558 0.420 -0.679;
         O 1.598 -0.513 -0.421; H -1.593 0.365 -0.554; H -0.834 0.355 1.048;
         H -0.780 -1.106 0.023; H 0.640 1.502 -0.580; H 0.567 0.174 -1.741;
         H 2.421 -0.190 -0.792''', 1, 1),
        ('CuCl2 (doublet)', os.path.join(HERE, '..', 'benchmarks', 'molecules',
                                         'tmc_cucl2.xyz'), 0, 1)]
XCS = ['b3lyp', 'wb97x', 'camb3lyp', 'hse06', 'lc_wpbe']

assert guks.Gradients.energy_ee is not guks.Gradients._energy_ee_orig
fails = []
for tag, atom, charge, spin in MOLS:
    mol = gto.M(atom=atom, basis=a.basis, charge=charge, spin=spin, verbose=0)
    for xc in XCS:
        mf = dft.UKS(mol, xc=xc)
        mf.conv_tol = 1e-10
        mf.kernel()
        g = mf.Gradients()
        dm = mf.make_rdm1()
        ours = cp.asnumpy(cp.asarray(g.energy_ee(mol, dm)))
        ref = cp.asnumpy(cp.asarray(guks.Gradients._energy_ee_orig(g, mol, dm)))
        err = np.abs(ours - ref).max()
        ok = err <= a.tol
        if not ok:
            fails.append(f'{tag} {xc}: {err:.3e}')
        print(f'  {"ok  " if ok else "FAIL"} {tag:16s} {xc:9s} '
              f'max|d energy_ee| {err:.3e}  (|ref| {np.abs(ref).max():.2e})',
              flush=True)
print('\n## partial_hess_elec')
from gpu4pyscf.hessian import rks as hrks, uks as huks      # noqa: E402
for tag, charge, spin in (('ethanol', 0, 0), ('ethanol+', 1, 1)):
    mol = gto.M(atom=MOLS[0][1], basis=a.basis, charge=charge, spin=spin,
                verbose=0)
    mod = huks if spin else hrks
    assert mod.Hessian.partial_hess_elec is not mod._partial_hess_elec_orig
    for xc in XCS:
        mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc)
        mf.conv_tol = 1e-10
        mf.kernel()
        h = mf.Hessian()
        ours = cp.asnumpy(cp.asarray(h.partial_hess_elec()))
        ref = cp.asnumpy(cp.asarray(mod._partial_hess_elec_orig(h)))
        err = np.abs(ours - ref).max()
        ok = err <= a.tol
        if not ok:
            fails.append(f'{tag} {xc} partial_hess_elec: {err:.3e}')
        print(f'  {"ok  " if ok else "FAIL"} {tag:16s} {xc:9s} '
              f'max|d partial_hess| {err:.3e}  (|ref| {np.abs(ref).max():.2e})',
              flush=True)
if fails:
    print('\nFAILED:\n  ' + '\n  '.join(fails))
    sys.exit(1)
print('\nall ok')
