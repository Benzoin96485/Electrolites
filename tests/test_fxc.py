"""`fastxcfxc` against GPU4PySCF, in one process.

`nr_rks_fxc` is called both ways on the same converged SCF and the same
`fxc` cache, so the comparison is exact up to the reassociation of the grid
sums.  `hermi = 0` is the TDA/TDDFT case and is checked directly rather than
resting on the algebra in the module docstring: it is the one place where the
two implementations do genuinely different arithmetic (GPU4PySCF forms a
second `dm^T . ao` product per grid block; this module symmetrises the density
once per call instead).

Then the end-to-end quantities on the same SCF: the polarizability tensor and
a TDA excitation spectrum, with the module patched and unpatched.  Sharing the
SCF matters -- see the comment where those two columns are checked.

    python tests/test_fxc.py [--quick]
"""
import argparse

import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fastxcfxc')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.dft import numint as NI                       # noqa: E402
from gpu4pyscf.properties import polarizability as gpol      # noqa: E402
from electrolites import fastxcfxc                           # noqa: E402

H2O = '''O 0. 0. 0.; H 0. -0.757 0.587; H 0. 0.757 0.587'''
CH3OH = '''C 0.667 0.000 0.000; O -0.667 0.000 0.000
H 1.100 1.000 0.100; H 1.100 -0.500 -0.870; H 1.100 -0.500 0.870
H -1.000 0.900 0.000'''
HCL = 'H 0. 0. 0.; Cl 0. 0. 1.28'
NH4 = '''N 0. 0. 0.; H 0.55 0.55 0.55; H -0.55 -0.55 0.55
H 0.55 -0.55 -0.55; H -0.55 0.55 -0.55'''
GLYCINE = 'benchmarks/molecules/glycine.xyz'
ALA2 = 'benchmarks/molecules/alanine_dipeptide.xyz'

# (atom, basis, xc, grid level, charge, nset)
CASES = [
    (H2O, 'def2-svp', 'b3lyp', 3, 0, 3),
    (H2O, 'def2-svp', 'svwn', 3, 0, 3),          # LDA
    (H2O, 'def2-tzvpp', 'pbe', 3, 0, 1),
    (H2O, 'cc-pvqz', 'b3lyp', 3, 0, 2),
    (CH3OH, 'def2-svp', 'b3lyp', 3, 0, 5),
    (CH3OH, '6-31g*', 'blyp', 3, 0, 8),
    (CH3OH, 'def2-svp', 'pbe0', 1, 0, 4),
    (HCL, 'def2-svp', 'b3lyp', 3, 0, 3),         # third row
    (NH4, 'def2-svp', 'b3lyp', 3, 1, 3),         # cation
    (GLYCINE, '6-31g*', 'b3lyp', 3, 0, 3),
    (GLYCINE, 'def2-svp', 'wb97x-d3bj', 3, 0, 6),
    (ALA2, 'def2-svp', 'b3lyp', 3, 0, 5),
    (ALA2, 'def2-svp', 'lda,vwn', 3, 0, 3),      # LDA, larger
    # meta-GGA
    (H2O, 'def2-svp', 'm06-2x', 3, 0, 3),
    (CH3OH, 'def2-svp', 'tpss', 3, 0, 5),
    (CH3OH, 'def2-tzvpp', 'm06-2x', 3, 0, 3),
    (GLYCINE, '6-31g*', 'wb97m-v', 3, 0, 3),
    (ALA2, 'def2-svp', 'tpss', 3, 0, 5),
    # declined: an unrestricted response falls through (checked by the SCF
    # being closed shell here, so nothing in this list reaches it) -- what is
    # listed as declined is the `mo1` path, which only the Hessian takes
]

DECLINED = ()

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
a = ap.parse_args()
cases = CASES[:5] + CASES[-5:] if a.quick else CASES

FAST = fastxcfxc._nr_rks_fxc_dispatch
ORIG = NI._nr_rks_fxc_orig


def perturbed_dms(mf, nset, hermi):
    """Realistic right-hand sides: the first CPKS guess for the dipole
    perturbation, extended by the lowest single excitations."""
    mo = cp.asarray(mf.mo_coeff)
    occ = cp.asarray(mf.mo_occ)
    e = cp.asarray(mf.mo_energy)
    mocc, mvir = mo[:, occ > 0], mo[:, occ == 0]
    de = e[occ == 0][:, None] - e[occ > 0][None, :]
    with mf.mol.with_common_orig((0, 0, 0)):
        h1 = cp.asarray(mf.mol.intor('int1e_r'))
    xs = [-(mvir.T @ h1[i] @ mocc) / de for i in range(3)]
    order = cp.argsort(de.ravel())
    for k in range(max(0, nset - 3)):
        x = cp.zeros(de.shape)
        x.ravel()[order[k % de.size]] = 1.0
        xs.append(x)
    dm1 = 2 * cp.asarray([mvir @ x @ mocc.T for x in xs[:nset]])
    if hermi == 1:
        dm1 = dm1 + dm1.transpose(0, 2, 1)
    return cp.asarray(dm1, order='C')


print(f'{"basis/xc/level":30s} {"nset":>4s} {"hermi1":>10s} {"hermi0":>10s} '
      f'{"polariz.":>10s} {"tda/eV":>10s} {"used":>5s}')
print('-' * 88)
fails = 0
for atom, basis, xc, level, charge, nset in cases:
    mol = gto.M(atom=atom, basis=basis, charge=charge, verbose=0,
                max_memory=40000)
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = level
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged

    ni = mf._numint
    ni.build(mol, mf.grids.coords)
    mo, occ = cp.asarray(mf.mo_coeff), cp.asarray(mf.mo_occ)
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, xc, mo, occ, 0)

    errs = {}
    for hermi in (1, 0):
        dms = perturbed_dms(mf, nset, hermi)
        args = (ni, mol, mf.grids, xc, None, dms, 0, hermi, rho0, vxc, fxc)
        v_fast = cp.asarray(FAST(*args))
        v_ref = cp.asarray(ORIG(*args, 2000, None))
        scale = float(cp.abs(v_ref).max()) or 1.0
        errs[hermi] = float(cp.abs(v_fast - v_ref).max()) / scale

    # End to end, on the *same* SCF: `nr_rks_fxc` is not called during an SCF,
    # so there is nothing to re-converge, and running a second one would only
    # add the SCF's own run-to-run spread to the comparison.
    used = xc not in DECLINED

    def _props():
        pol = gpol.eval_polarizability(mf)
        td = mf.TDA()
        td.nstates = 3
        td.kernel()
        return pol, np.asarray(td.e)

    pol_fast, e_fast = _props()
    NI.nr_rks_fxc = NI.NumInt.nr_rks_fxc = ORIG
    pol_ref, e_ref = _props()
    NI.nr_rks_fxc = NI.NumInt.nr_rks_fxc = FAST

    d_pol = np.abs(pol_fast - pol_ref).max() / max(np.abs(pol_ref).max(), 1.0)
    d_td = np.abs(e_fast - e_ref).max() * 27.211386245988

    tag = f'{basis}/{xc}/L{level}'
    print(f'{tag:30s} {nset:4d} {errs[1]:10.2e} {errs[0]:10.2e} '
          f'{d_pol:10.2e} {d_td:10.2e} {str(used):>5s}', flush=True)
    # All four columns are round-off *because* the SCF is shared.  An earlier
    # version of this test converged a second SCF for the reference half, and
    # on one case (alanine dipeptide/LDA, the slowest SCF here) that alone
    # moved the polarizability by 2e-7 and the spectrum by 2.7e-6 eV -- the
    # two SCFs land on slightly different densities and the CPKS solve
    # amplifies the difference to its own tolerance.  Nothing about the
    # kernel: its `hermi` columns were 2e-15 in that run too.
    tol = 1e-9 if used else 0.0
    if not (errs[1] <= tol and errs[0] <= tol and d_pol <= 1e-9
            and d_td <= 1e-7):
        fails += 1
        print('  ^ FAIL', flush=True)

print('-' * 88)
print(f'{len(cases) - fails}/{len(cases)} cases pass')
raise SystemExit(1 if fails else 0)
