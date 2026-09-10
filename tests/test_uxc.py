"""`fastxc`'s unrestricted XC build against GPU4PySCF's, in one process.

`nr_uks` is called both ways on the same converged density, so the comparison
is exact up to the reassociation of the grid sums -- both the `grouped` mode
the SCF uses and the `perblock` ablation, because they walk the blocks in a
different order and only one of them shares GPU4PySCF's.

The cases are transition-metal complexes from BOS-TMC-28242 (see
`benchmarks/molecules/tmc_cases.txt`) plus a few small radicals.  Two things
about that set are not corners:

* it is 83 % open-shell, so the unrestricted build is the *common* case there,
  not the exception, and
* def2-SVP already puts an f shell on the metal, which no organic def2-SVP or
  6-31G* case in `tests/` does -- so these cases reach angular-momentum
  classes that the rest of the suite only reaches at def2-TZVPP.

Meta-GGA is covered on this side even though `nr_rks` hands it back -- on the
restricted side `fastxcnlc` already reaches it through `RKS.get_veff`, and
there is no unrestricted counterpart of that module, so without it a
transition-metal complex run with M06-2X, TPSSh or wB97M-V (the usual choices
for one) would get no XC acceleration at all.

The last block checks what the module still declines and confirms it falls
through rather than fails: a stack of density *pairs* (`nset` > 1).

    python tests/test_uxc.py [--quick]
"""
import argparse

import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fastxc')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.dft import numint as NI                       # noqa: E402
from electrolites import fastxc                              # noqa: E402

O2 = 'O 0. 0. 0.; O 0. 0. 1.208'
NO = 'N 0. 0. 0.; O 0. 0. 1.154'
CH3 = '''C 0. 0. 0.; H 1.079 0. 0.
H -0.539 0.934 0.; H -0.539 -0.934 0.'''
M = 'benchmarks/molecules/'

# (atom, basis, xc, grid level, charge, spin)
CASES = [
    (O2, 'def2-svp', 'b3lyp', 3, 0, 2),               # triplet
    (O2, 'def2-tzvpp', 'pbe', 3, 0, 2),               # reaches g
    (CH3, 'def2-svp', 'b3lyp', 3, 0, 1),              # radical doublet
    (CH3, '6-31g*', 'svwn', 3, 0, 1),                 # LDA
    (NO, 'def2-svp', 'blyp', 3, 0, 1),
    (O2, 'def2-svp', 'lda,vwn', 1, 0, 2),             # LDA, coarse grid
    (O2, 'def2-svp', 'm06-2x', 3, 0, 2),              # meta-GGA hybrid
    (CH3, 'def2-svp', 'tpss', 3, 0, 1),               # meta-GGA
    (M + 'tmc_cucl2.xyz', 'def2-svp', 'b3lyp', 3, 0, 1),
    (M + 'tmc_cucl2.xyz', 'def2-tzvp', 'pbe', 3, 0, 1),
    (M + 'tmc_fecl4.xyz', 'def2-svp', 'b3lyp', 3, -1, 5),      # high spin
    (M + 'tmc_fecl4.xyz', 'def2-tzvpp', 'b3lyp', 3, -1, 5),    # reaches g
    (M + 'tmc_fecl4.xyz', 'def2-svp', 'svwn', 3, -1, 5),       # LDA
    (M + 'tmc_pt_amine.xyz', 'def2-svp', 'b3lyp', 3, 0, 2),    # 5d, ECP
    (M + 'tmc_ni_dithiolene.xyz', 'def2-svp', 'b3lyp', 3, -1, 1),
    (M + 'tmc_ni_dithiolene.xyz', 'def2-svp', 'wb97x-d3bj', 3, -1, 1),
    (M + 'tmc_cu_macrocycle.xyz', 'def2-svp', 'b3lyp', 3, 2, 3),
    (M + 'tmc_fecl4.xyz', 'def2-svp', 'm06-2x', 3, -1, 5),      # meta-GGA
    (M + 'tmc_ni_dithiolene.xyz', 'def2-svp', 'wb97m-v', 3, -1, 1),
]

#: nothing is declined by functional type any more; kept so that adding one
#: back is a one-line change and the fall-through stays checked
DECLINED = ()
DECLINED_CASES = []

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
a = ap.parse_args()
cases = CASES[:4] + CASES[8:10] + CASES[-2:] if a.quick else CASES


def build(atom, basis, charge, spin):
    """The def2 family replaces the core of everything past Kr with an ECP, and
    a 5d complex is wrong without it; nothing else here has one."""
    kw = dict(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0,
              max_memory=40000)
    if basis.lower().startswith('def2') and any(
            gto.charge(sym) > 36 for sym, _ in gto.format_atom(atom)):
        kw['ecp'] = basis
    return gto.M(**kw)


def compare(ni, mol, grids, xc, dm, mode):
    """nr_uks our way and GPU4PySCF's, on the same density."""
    fastxc.MODE = mode
    n1, e1, v1 = NI.NumInt.nr_uks(ni, mol, grids, xc, dm)
    n0, e0, v0 = NI._nr_uks_orig(ni, mol, grids, xc, dm)
    v0, v1 = cp.asarray(v0), cp.asarray(v1)
    scale = float(cp.abs(v0).max()) or 1.0
    return (float(np.max(np.abs(np.asarray(n1) - np.asarray(n0)))),
            abs(float(e1) - float(e0)),
            float(cp.abs(v1 - v0).max()) / scale)


print(f'{"case":38s} {"xc":11s} {"nao":>5s} {"lmax":>4s} '
      f'{"nelec":>9s} {"exc":>9s} {"v(grp)":>9s} {"v(blk)":>9s} {"E(SCF)":>9s}')
print('-' * 118)
fails = skipped = 0
for atom, basis, xc, level, charge, spin in cases + DECLINED_CASES:
    declined = xc in DECLINED
    mol = build(atom, basis, charge, spin)
    mf = dft.UKS(mol, xc=xc)
    mf.grids.level = level
    mf.conv_tol = 1e-10
    mf.direct_scf_tol = 1e-13
    mf.max_cycle = 200
    e_patched = mf.kernel()
    if not mf.converged:
        # convergence is not what is under test here, and a few of these
        # high-spin cases are genuinely hard; say so rather than fail
        print(f'{atom.rsplit("/", 1)[-1][:24]+"/"+basis:38s} {xc:11s} '
              f'SCF did not converge -- skipped', flush=True)
        skipped += 1
        continue

    ni = mf._numint
    dm = cp.asarray(mf.make_rdm1()).view(cp.ndarray)
    errs = {m: compare(ni, mol, mf.grids, xc, dm, m)
            for m in ('grouped', 'perblock')}
    fastxc.MODE = 'grouped'

    # ... and end to end: the same SCF driven by GPU4PySCF's own build.  The
    # two SCFs are separate convergence paths, so this column is only expected
    # at the convergence threshold, not at the round-off of the column above.
    NI.NumInt.nr_uks = NI.nr_uks = NI._nr_uks_orig
    try:
        mf2 = dft.UKS(build(atom, basis, charge, spin), xc=xc)
        mf2.grids.level = level
        mf2.conv_tol = 1e-10
        mf2.direct_scf_tol = 1e-13
        mf2.max_cycle = 200
        e_stock = mf2.kernel()
        conv2 = mf2.converged
    finally:
        NI.NumInt.nr_uks = NI.nr_uks = fastxc._nr_uks_dispatch

    # nan rather than a failure if GPU4PySCF's own SCF did not converge: the
    # column above is the comparison that matters and it is already in hand
    de = abs(float(e_patched) - float(e_stock)) if conv2 else float('nan')
    name = atom.rsplit('/', 1)[-1].replace('.xyz', '')
    if len(name) > 24:
        name = name[:24]
    tol_v = 1e-9
    ok = (max(errs['grouped']) < tol_v and max(errs['perblock']) < tol_v
          and (de < 1e-7 or de != de))
    if declined:                 # must be *bit* identical: it is the same code
        ok = ok and max(errs['grouped'][2], errs['perblock'][2]) == 0.0
    fails += not ok
    print(f'{name+"/"+basis:38s} {xc:11s} {mol.nao:5d} '
          f'{int(mol._bas[:, 1].max()):4d} '
          f'{errs["grouped"][0]:9.2e} {errs["grouped"][1]:9.2e} '
          f'{errs["grouped"][2]:9.2e} {errs["perblock"][2]:9.2e} '
          f'{de:9.2e} {"" if ok else "  <-- FAIL"}'
          f'{"  (declined)" if declined else ""}', flush=True)

# nset > 1 -- a stack of density pairs, which the module declines.  It has to
# fall through and not raise, because `nr_uks` is a public entry point.
mol = build(O2, 'def2-svp', 0, 2)
mf = dft.UKS(mol, xc='b3lyp')
mf.grids.level = 3
mf.kernel()
dm = cp.asarray(mf.make_rdm1()).view(cp.ndarray)
stack = cp.asarray([dm[0][None] * 1.0, dm[1][None] * 1.0])   # (2, 1, nao, nao)
v_fast = cp.asarray(NI.NumInt.nr_uks(mf._numint, mol, mf.grids, 'b3lyp',
                                     stack)[2])
v_ref = cp.asarray(NI._nr_uks_orig(mf._numint, mol, mf.grids, 'b3lyp',
                                   stack)[2])
nset_ok = float(cp.abs(v_fast - v_ref).max()) == 0.0
fails += not nset_ok
print(f'\nnset=2 stack falls through bit-identically: '
      f'{"yes" if nset_ok else "NO  <-- FAIL"}')

print(f'\n{len(cases) + len(DECLINED_CASES) + 1 - skipped} checks, '
      f'{fails} failed, {skipped} skipped (SCF did not converge)')
raise SystemExit(1 if fails else 0)
