"""`fastfnlc` against GPU4PySCF, in one process.

The VV10 response term is called with the same ground-state orbitals on every
iteration of a CPKS or CPHF solve, so every case here calls it **twice** and
checks the second call as well: the first fills the ground-state cache, the
second uses it, and both must agree with GPU4PySCF.

The `mo` column is the `return_in_mo=True` path.  With
`FASTFNLC_RHO_BATCH=0 FASTFNLC_MO_BATCH=0` it keeps GPU4PySCF's own density
evaluation and potential contraction and therefore has to be
**bit-identical** -- that is the test of the cache on its own; by default
both are batched over the densities and it is checked to round-off.  The `ao` column is `return_in_mo=False`, whose
potential contraction is batched over the perturbed densities and reassociates
one grid sum, so it is checked to round-off instead.

Then the polarizability tensor end to end, which is where the term is 80 % of
the wall clock.

    python tests/test_fnlc.py [--quick]
"""
import argparse

import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fastfnlc')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.hessian import rks as HR                      # noqa: E402
from gpu4pyscf.properties import polarizability as gpol      # noqa: E402
from electrolites import fastfnlc                            # noqa: E402

H2O = '''O 0. 0. 0.; H 0. -0.757 0.587; H 0. 0.757 0.587'''
CH3OH = '''C 0.667 0.000 0.000; O -0.667 0.000 0.000
H 1.100 1.000 0.100; H 1.100 -0.500 -0.870; H 1.100 -0.500 0.870
H -1.000 0.900 0.000'''
HCL = 'H 0. 0. 0.; Cl 0. 0. 1.28'
GLYCINE = 'benchmarks/molecules/glycine.xyz'

# (atom, basis, xc, grid level, nset)
CASES = [
    (H2O, 'def2-svp', 'wb97m-v', 3, 3),
    (H2O, 'def2-tzvpp', 'wb97x-v', 3, 1),
    (CH3OH, 'def2-svp', 'wb97m-v', 3, 5),
    (CH3OH, '6-31g*', 'b97m-v', 3, 3),
    (HCL, 'def2-svp', 'wb97m-v', 3, 3),          # third row
    (GLYCINE, 'def2-svp', 'wb97m-v', 3, 3),
    (GLYCINE, '6-31g*', 'b97m-v', 2, 6),
    (CH3OH, 'def2-svp', 'wb97m-v', 1, 3),        # a coarse grid
]

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
a = ap.parse_args()
cases = CASES[:3] if a.quick else CASES

print(f'{"basis/xc/level":30s} {"nset":>4s} {"mo (exact?)":>12s} '
      f'{"ao":>10s} {"2nd call":>10s} {"polariz.":>10s}')
print('-' * 82)
fails = 0
for atom, basis, xc, level, nset in cases:
    mol = gto.M(atom=atom, basis=basis, verbose=0, max_memory=40000)
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = level
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    assert mf.do_nlc()

    mo, occ = cp.asarray(mf.mo_coeff), cp.asarray(mf.mo_occ)
    en = cp.asarray(mf.mo_energy)
    mocc, mvir = mo[:, occ > 0], mo[:, occ == 0]
    de = en[occ == 0][:, None] - en[occ > 0][None, :]
    with mol.with_common_orig((0, 0, 0)):
        h1 = cp.asarray(mol.intor('int1e_r'))
    xs = [-(mvir.T @ h1[i] @ mocc) / de for i in range(3)]
    order = cp.argsort(de.ravel())
    for k in range(max(0, nset - 3)):
        x = cp.zeros(de.shape)
        x.ravel()[order[k % de.size]] = 1.0
        xs.append(x)
    dm1 = 2 * cp.asarray([mvir @ x @ mocc.T for x in xs[:nset]])
    dm1 = cp.asarray(dm1 + dm1.transpose(0, 2, 1), order='C')

    errs = {}
    exact = True
    for in_mo in (True, False):
        args = (mf, mol, mo, occ, dm1, in_mo)
        ref = cp.asarray(HR._nr_rks_fnlc_mo_orig(*args))
        got = cp.asarray(fastfnlc._nr_rks_fnlc_mo(*args))
        got2 = cp.asarray(fastfnlc._nr_rks_fnlc_mo(*args))
        scale = float(cp.abs(ref).max()) or 1.0
        errs[in_mo] = float(cp.abs(got - ref).max()) / scale
        errs[(in_mo, 2)] = float(cp.abs(got2 - ref).max()) / scale
        if in_mo:
            exact = bool(cp.all(got == ref)) and bool(cp.all(got2 == ref))

    # the same SCF for both: `nr_rks_fnlc_mo` is the *response* of the VV10
    # term and is not called during an SCF, so there is nothing to reconverge
    pol_fast = gpol.eval_polarizability(mf)
    HR.nr_rks_fnlc_mo = HR._nr_rks_fnlc_mo_orig
    pol_ref = gpol.eval_polarizability(mf)
    HR.nr_rks_fnlc_mo = fastfnlc._dispatch
    d_pol = np.abs(pol_fast - pol_ref).max() / max(np.abs(pol_ref).max(), 1.0)

    tag = f'{basis}/{xc}/L{level}'
    print(f'{tag:30s} {nset:4d} {errs[True]:9.2e} {str(exact):>3s} '
          f'{errs[False]:10.2e} {max(errs[(True, 2)], errs[(False, 2)]):10.2e} '
          f'{d_pol:10.2e}', flush=True)
    # `mo` is exact, `ao` and the polarizability are round-off -- the last
    # one only because the SCF is shared; see `tests/test_fxc.py`.
    # The `mo` path is bit-identical only with the batched density and
    # MO-projection changes switched off (FASTFNLC_RHO_BATCH=0
    # FASTFNLC_MO_BATCH=0); with them on it reassociates the same sums.
    mo_ok = exact if not (fastfnlc.RHO_BATCH or fastfnlc.MO_BATCH) else \
        (errs[True] <= 1e-12 and errs[(True, 2)] <= 1e-12)
    if not (mo_ok and errs[False] <= 1e-12 and errs[(False, 2)] <= 1e-12
            and d_pol <= 1e-9):
        fails += 1
        print('  ^ FAIL', flush=True)

print('-' * 82)
print(f'{len(cases) - fails}/{len(cases)} cases pass')
raise SystemExit(1 if fails else 0)
