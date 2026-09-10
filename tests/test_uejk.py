"""`fastejk` at one density matrix and at two, against GPU4PySCF's own kernel.

`_jk_energy_per_atom` is called both ways on the same converged density and the
same `_VHFOpt`, in one process, so the comparison is the kernels and nothing
else.

Two columns, and the closed-shell one is not a formality.  The kernels used to
contract a single density matrix and `_usable` declined `n_dm != 1` outright,
so every angular-momentum class of an open-shell gradient went to GPU4PySCF
(1649 of 1649 class blocks on a UKS B3LYP/def2-SVP run --
`benchmarks/coverage_probe.py`).  Taking `n_dm = 2` back meant putting the
unrolled exchange contraction inside a runtime loop over the densities, which
the closed-shell path now also runs, once.  The `n_dm = 1` column is the check
that it still gets the same answer; `benchmarks/ejk_ndm.py` is the check that
it still gets it as fast.

The `(j, k)` triples separate the two terms: the Coulomb contraction uses the
*total* density and the exchange one the spin densities, so a factor error in
either would show in only one row.

    python tests/test_uejk.py [--quick]
"""
import argparse

import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fastejk')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.grad import rhf as grhf                       # noqa: E402
from gpu4pyscf.scf import jk as gjk                          # noqa: E402
from electrolites import fastejk                             # noqa: E402

O2 = 'O 0. 0. 0.; O 0. 0. 1.208'
M = 'benchmarks/molecules/'

# (atom, basis, xc, charge, spin)
CASES = [
    (M + 'glycine.xyz', 'def2-svp', 'b3lyp', 0, 0),           # closed shell
    (M + 'glycine.xyz', '6-31g*', 'b3lyp', 0, 0),
    (O2, 'def2-svp', 'b3lyp', 0, 2),
    (O2, 'def2-tzvpp', 'b3lyp', 0, 2),                        # reaches g
    (M + 'tmc_cucl2.xyz', 'def2-svp', 'b3lyp', 0, 1),
    (M + 'tmc_fecl4.xyz', 'def2-svp', 'b3lyp', -1, 5),        # high spin
    (M + 'tmc_pt_amine.xyz', 'def2-svp', 'b3lyp', 0, 2),      # 5d, ECP
    (M + 'tmc_ni_dithiolene.xyz', 'def2-svp', 'b3lyp', -1, 1),
]

#: (j_factor, k_factor) -- both terms, then each alone
FACTORS = ((1.0, 0.2), (1.0, 0.0), (0.0, 0.25))

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
a = ap.parse_args()
cases = CASES[:1] + CASES[2:3] + CASES[4:6] if a.quick else CASES


def build(atom, basis, charge, spin):
    kw = dict(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0,
              max_memory=40000)
    if basis.lower().startswith('def2') and any(
            gto.charge(sym) > 36 for sym, _ in gto.format_atom(atom)):
        kw['ecp'] = basis
    return gto.M(**kw)


print(f'{"case":34s} {"n_dm":>4s} {"nao":>5s} {"lmax":>4s} '
      + ' '.join(f'{"j=%g,k=%g" % f:>13s}' for f in FACTORS))
print('-' * 108)
fails = skipped = 0
for atom, basis, xc, charge, spin in cases:
    mol = build(atom, basis, charge, spin)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc)
    mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.direct_scf_tol = 1e-13
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        print(f'{atom.rsplit("/", 1)[-1][:24]+"/"+basis:34s} '
              f'SCF did not converge -- skipped', flush=True)
        skipped += 1
        continue
    dm = cp.asarray(mf.make_rdm1())
    vhfopt = gjk._VHFOpt(mol, mf.direct_scf_tol).build()

    rels = []
    for jf, kf in FACTORS:
        ours = np.asarray(fastejk.jk_energy_per_atom(vhfopt, dm, jf, kf))
        ref = np.asarray(grhf._jk_energy_per_atom_orig(vhfopt, dm, jf, kf))
        scale = max(float(np.abs(ref).max()), 1e-30)
        rels.append(float(np.abs(ours - ref).max()) / scale)

    name = atom.rsplit('/', 1)[-1].replace('.xyz', '')[:24]
    ok = max(rels) < 1e-11
    fails += not ok
    print(f'{name+"/"+basis:34s} {1 if spin == 0 else 2:4d} {mol.nao:5d} '
          f'{int(mol._bas[:, 1].max()):4d} '
          + ' '.join(f'{r:13.2e}' for r in rels)
          + ('' if ok else '  <-- FAIL'), flush=True)

print(f'\n{len(cases) - skipped} checks, {fails} failed, '
      f'{skipped} skipped (SCF did not converge)')
raise SystemExit(1 if fails else 0)
