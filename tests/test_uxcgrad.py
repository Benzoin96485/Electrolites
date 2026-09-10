"""`fastxcgrad`'s unrestricted XC gradient against GPU4PySCF's, in one process.

`grad.uks.get_exc` is called both ways on the same converged density, in both
the `grouped` mode a gradient uses and the `perblock` ablation, and then the
whole gradient is compared patched against unpatched.

The point of the second column is that `grad/uks.py` binds its own `get_exc`
at import time: patching `grad.rks.get_exc` -- which is all this module used
to do -- does not reach it, so an unrestricted gradient ran GPU4PySCF's XC
contraction with the module loaded and nothing said so.  The whole-gradient
column is what would have caught that, had it been measured rather than
assumed; `benchmarks/coverage_probe.py` is the direct check.

The cases are the transition-metal complexes of `test_uxc.py` plus small
radicals, over the three functional types.

    python tests/test_uxcgrad.py [--quick]
"""
import argparse

import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fastxcgrad')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.grad import uks as guks                       # noqa: E402
from gpu4pyscf.lib.cupy_helper import tag_array               # noqa: E402
from electrolites import fastxcgrad                          # noqa: E402

O2 = 'O 0. 0. 0.; O 0. 0. 1.208'
CH3 = '''C 0. 0. 0.; H 1.079 0. 0.
H -0.539 0.934 0.; H -0.539 -0.934 0.'''
M = 'benchmarks/molecules/'

# (atom, basis, xc, grid level, charge, spin)
CASES = [
    (O2, 'def2-svp', 'b3lyp', 3, 0, 2),
    (O2, 'def2-svp', 'svwn', 3, 0, 2),                 # LDA
    (O2, 'def2-svp', 'm06-2x', 3, 0, 2),               # meta-GGA
    (CH3, '6-31g*', 'pbe', 3, 0, 1),
    (M + 'tmc_cucl2.xyz', 'def2-svp', 'b3lyp', 3, 0, 1),
    (M + 'tmc_cucl2.xyz', 'def2-svp', 'm06-2x', 3, 0, 1),
    (M + 'tmc_fecl4.xyz', 'def2-svp', 'b3lyp', 3, -1, 5),
    (M + 'tmc_fecl4.xyz', 'def2-tzvpp', 'b3lyp', 3, -1, 5),   # reaches g
    (M + 'tmc_pt_amine.xyz', 'def2-svp', 'b3lyp', 3, 0, 2),   # 5d, ECP
    (M + 'tmc_ni_dithiolene.xyz', 'def2-svp', 'b3lyp', 3, -1, 1),
]

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
a = ap.parse_args()
cases = CASES[:3] + CASES[4:6] if a.quick else CASES


def build(atom, basis, charge, spin):
    kw = dict(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0,
              max_memory=40000)
    if basis.lower().startswith('def2') and any(
            gto.charge(sym) > 36 for sym, _ in gto.format_atom(atom)):
        kw['ecp'] = basis
    return gto.M(**kw)


print(f'{"case":38s} {"xc":9s} {"nao":>5s} {"lmax":>4s} '
      f'{"exc1(grp)":>10s} {"exc1(blk)":>10s} {"whole grad":>11s}')
print('-' * 100)
fails = skipped = 0
for atom, basis, xc, level, charge, spin in cases:
    mol = build(atom, basis, charge, spin)
    mf = dft.UKS(mol, xc=xc)
    mf.grids.level = level
    mf.conv_tol = 1e-10
    mf.direct_scf_tol = 1e-13
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        print(f'{atom.rsplit("/", 1)[-1][:24]+"/"+basis:38s} {xc:9s} '
              f'SCF did not converge -- skipped', flush=True)
        skipped += 1
        continue

    ni = mf._numint
    dm = tag_array(mf.make_rdm1(), mo_coeff=cp.asarray(mf.mo_coeff),
                   mo_occ=cp.asarray(mf.mo_occ))
    errs = {}
    for mode in ('grouped', 'perblock'):
        fastxcgrad.MODE = mode
        e1 = cp.asarray(guks.get_exc(ni, mol, mf.grids, xc, dm, verbose=0)[1])
        e0 = cp.asarray(guks._get_exc_orig(ni, mol, mf.grids, xc, dm,
                                           verbose=0)[1])
        errs[mode] = (float(cp.abs(e1 - e0).max())
                      / (float(cp.abs(e0).max()) or 1.0))
    fastxcgrad.MODE = 'grouped'

    # ... and the whole gradient on the same SCF, our XC term against
    # GPU4PySCF's.  Everything else in it is the same code both times, so this
    # column is the exc1 column diluted by the two-electron term -- it is here
    # because it is what a caller sees.
    de_fast = mf.nuc_grad_method().kernel()
    guks.get_exc = guks._get_exc_orig
    try:
        de_ref = mf.nuc_grad_method().kernel()
    finally:
        guks.get_exc = fastxcgrad.get_exc_u
    dg = float(np.abs(np.asarray(de_fast) - np.asarray(de_ref)).max())

    name = atom.rsplit('/', 1)[-1].replace('.xyz', '')[:24]
    ok = max(errs.values()) < 1e-9 and dg < 1e-9
    fails += not ok
    print(f'{name+"/"+basis:38s} {xc:9s} {mol.nao:5d} '
          f'{int(mol._bas[:, 1].max()):4d} {errs["grouped"]:10.2e} '
          f'{errs["perblock"]:10.2e} {dg:11.2e}'
          f'{"" if ok else "  <-- FAIL"}', flush=True)

print(f'\n{len(cases) - skipped} checks, {fails} failed, '
      f'{skipped} skipped (SCF did not converge)')
raise SystemExit(1 if fails else 0)
