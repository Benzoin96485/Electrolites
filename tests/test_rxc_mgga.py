"""`fastxc`'s restricted meta-GGA XC build against GPU4PySCF's (catalogue A14).

`nr_rks` both ways on the same converged density (tagged with the orbitals,
as the SCF passes it, and untagged), for pure and hybrid meta-GGAs, organic
and closed-shell transition-metal systems, def2-TZVPP (which reaches g on
the metal-free side and f on the metal), a coarse grid; then the SCF energy
both ways.  The grid sums are reassociated, so the check is round-off:
1e-11 relative on the matrix, 1e-9 Eh on N and Exc and on the SCF energy.

    python tests/test_rxc_mgga.py [--quick]
"""
import argparse

import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fastxc')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.dft import numint as NI                       # noqa: E402

H2O = 'O 0. 0. 0.; H 0. -0.757 0.587; H 0. 0.757 0.587'
M = 'benchmarks/molecules/'
CASES = [  # atom, basis, xc, grid level, charge, spin
    (H2O, 'def2-svp', 'tpss', 3, 0, 0),
    (H2O, 'def2-tzvpp', 'r2scan', 3, 0, 0),
    (M + 'glycine.xyz', 'def2-svp', 'm06-2x', 3, 0, 0),
    (M + 'glycine.xyz', '6-31g*', 'scan', 3, 0, 0),
    (M + 'glycine.xyz', 'def2-svp', 'mn15', 2, 0, 0),
    (M + 'glycine.xyz', 'def2-svp', 'tpssh', 1, 0, 0),
    (M + 'tmc_pt_amine.xyz', 'def2-svp', 'tpss', 3, 0, 0),
    (M + 'tmc_ni_dithiolene.xyz', 'def2-svp', 'r2scan', 3, -2, 0),
]
ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
a = ap.parse_args()
cases = CASES[:3] if a.quick else CASES

fails = 0
for atom, basis, xc, level, charge, spin in cases:
    kw = dict(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
    if 'pt_' in atom:
        kw['ecp'] = basis
    mol = gto.M(**kw)
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = level
    mf.conv_tol = 1e-10
    e1 = mf.kernel()
    ni = mf._numint
    dm = mf.make_rdm1()
    errs = []
    for d in (dm, cp.asarray(dm).view(cp.ndarray)):
        n1, x1, v1 = ni.nr_rks(mol, mf.grids, xc, d)
        n0, x0, v0 = NI._nr_rks_orig(ni, mol, mf.grids, xc, d)
        errs.append((abs(n1 - n0), abs(x1 - x0),
                     float(cp.abs(v1 - v0).max() / cp.abs(v0).max())))
    NI.NumInt.nr_rks = NI._nr_rks_orig
    try:
        mf0 = dft.RKS(mol, xc=xc)
        mf0.grids.level = level
        mf0.conv_tol = 1e-10
        e0 = mf0.kernel()
    finally:
        from electrolites import fastxc
        NI.NumInt.nr_rks = fastxc._nr_rks_dispatch
    de = abs(e1 - e0)
    ok = all(dn < 1e-9 and dx < 1e-9 and dv < 1e-11 for dn, dx, dv in errs) \
        and de < 1e-8
    fails += not ok
    tag = f'{atom.rsplit("/", 1)[-1][:16]}/{basis}/{xc}/L{level}'
    print(f'{tag:42s} dV {max(e[2] for e in errs):.1e}  dExc '
          f'{max(e[1] for e in errs):.1e}  dE(SCF) {de:.1e}  '
          f'{"ok" if ok else "FAIL"}', flush=True)
print('\nFAILED' if fails else '\nall passed', f'({fails} failures)')
raise SystemExit(1 if fails else 0)
