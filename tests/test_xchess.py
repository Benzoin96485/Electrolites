"""`fastxchess` against GPU4PySCF, in one process.

Both implementations are called on the same converged SCF, so the comparison
is exact up to the reassociation of the grid sums: `_get_vxc_deriv1` and
`_get_vxc_deriv2` term by term, and then the whole analytic Hessian with the
module patched and unpatched.

    python tests/test_xchess.py [--quick]
"""
import argparse
import sys

import numpy as np
import cupy as cp

import electrolites
electrolites.patch('fastxchess')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.hessian import rks as hess_rks                 # noqa: E402
from electrolites import fastxchess                           # noqa: E402

H2O = '''O 0. 0. 0.; H 0. -0.757 0.587; H 0. 0.757 0.587'''
CH3OH = '''C 0.667 0.000 0.000; O -0.667 0.000 0.000
H 1.100 1.000 0.100; H 1.100 -0.500 -0.870; H 1.100 -0.500 0.870
H -1.000 0.900 0.000'''
GLYCINE = 'benchmarks/molecules/glycine.xyz'
ALA2 = 'benchmarks/molecules/alanine_dipeptide.xyz'

CASES = [
    (H2O, 'def2-svp', 'b3lyp', 3),
    (H2O, 'def2-tzvpp', 'pbe', 3),
    (H2O, '6-31g*', 'pbe0', 1),
    (CH3OH, 'def2-svp', 'b3lyp', 3),
    (CH3OH, 'cc-pvdz', 'blyp', 3),
    (GLYCINE, '6-31g*', 'b3lyp', 3),
    (GLYCINE, 'def2-svp', 'wb97x-d3bj', 3),
    (ALA2, 'def2-svp', 'b3lyp', 3),
    (ALA2, '6-31g**', 'pbe', 2),
    # meta-GGAs (W28): pure, hybrid, range-separated, and one with VV10
    (H2O, 'def2-svp', 'tpss', 3),
    (H2O, 'def2-tzvpp', 'r2scan', 3),
    (CH3OH, 'def2-svp', 'm06-l', 3),
    (GLYCINE, 'def2-svp', 'tpss', 3),
    (GLYCINE, '6-31g*', 'scan', 3),
    (ALA2, 'def2-svp', 'm06-2x', 3),
    (CH3OH, 'def2-svp', 'wb97m-v', 3),
    # a declined path: it must fall through and still agree exactly
    (H2O, 'def2-svp', 'svwn', 3),
    (H2O, 'def2-svp', 'm06-2x', 3),
]

#: the functionals this module hands back to GPU4PySCF
DECLINED = ('svwn',)

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
a = ap.parse_args()
cases = CASES[:4] + CASES[9:11] + CASES[-2:] if a.quick else CASES

FAST1, FAST2 = hess_rks._get_vxc_deriv1, hess_rks._get_vxc_deriv2
ORIG1, ORIG2 = hess_rks._get_vxc_deriv1_orig, hess_rks._get_vxc_deriv2_orig

fails = 0
# `self` is GPU4PySCF against itself: the same function called twice on the
# same input.  It is 0 wherever the arithmetic is deterministic, and it is the
# only sensible yardstick where it is not.
#
# A case the module declines is not a numerical comparison at all -- both
# sides are GPU4PySCF's own code -- so what is asserted there is that it
# really did decline.  Reporting the numbers anyway is how the meta-GGA path's
# run-to-run spread got noticed, but they cannot be a pass criterion: it is
# upstream against upstream, and on one run of this suite the two ORIG calls
# happened to agree exactly while the first pair differed by 5e-5.
print(f'{"basis/xc/level":34s} {"deriv1":>10s} {"deriv2":>10s} '
      f'{"hessian":>10s} {"self":>10s} {"|H|":>10s}')
print('-' * 92)
for atom, basis, xc, level in cases:
    mol = gto.M(atom=atom, basis=basis, verbose=0, max_memory=40000)
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = level
    mf.conv_tol = 1e-11
    mf.kernel()
    h = mf.Hessian()
    mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ

    d1f = cp.asarray(FAST1(h, mo_coeff, mo_occ, 4000))
    d1o = cp.asarray(ORIG1(h, mo_coeff, mo_occ, 4000))
    e1 = float(cp.abs(d1f - d1o).max())
    d1o2 = cp.asarray(ORIG1(h, mo_coeff, mo_occ, 4000))
    es = float(cp.abs(d1o2 - d1o).max())
    d2f = cp.asarray(FAST2(h, mo_coeff, mo_occ, 4000))
    d2o = cp.asarray(ORIG2(h, mo_coeff, mo_occ, 4000))
    e2 = float(cp.abs(d2f - d2o).max())
    ref1 = float(cp.abs(d1o).max())
    d1f = d1o = d2f = d2o = None

    hess_rks._get_vxc_deriv1, hess_rks._get_vxc_deriv2 = FAST1, FAST2
    hf = cp.asarray(h.kernel())
    hess_rks._get_vxc_deriv1, hess_rks._get_vxc_deriv2 = ORIG1, ORIG2
    h.de = np.zeros((0, 0, 3, 3))
    ho = cp.asarray(h.kernel())
    hess_rks._get_vxc_deriv1, hess_rks._get_vxc_deriv2 = FAST1, FAST2
    eh = float(cp.abs(hf - ho).max())
    nrm = float(cp.abs(ho).max())

    tag = f'{basis}/{xc}/L{level} {mol.natm}at'
    declined = fastxchess._setup(h, mo_occ) is None
    if declined:
        # the assertion is the fall-through itself
        ok = xc in DECLINED
        tag += ' (declined)'
    else:
        ok = (xc not in DECLINED and e1 < 1e-8 * max(ref1, 1.)
              and e2 < 1e-8 and eh < 1e-8 * max(nrm, 1.))
    fails += not ok
    print(f'{tag:34s} {e1:10.2e} {e2:10.2e} {eh:10.2e} {es:10.2e} '
          f'{nrm:10.2e}{"" if ok else "   FAIL"}')
    mol = mf = h = hf = ho = None
    cp.get_default_memory_pool().free_all_blocks()

print('-' * 92)
print(f'{len(cases) - fails}/{len(cases)} cases agree, or decline as intended')
sys.exit(1 if fails else 0)
