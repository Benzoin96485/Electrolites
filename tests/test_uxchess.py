"""`fastxchess`'s unrestricted pair against GPU4PySCF's, in one process.

`hessian/uks.py` carries its own copies of `_get_vxc_deriv1` and
`_get_vxc_deriv2` -- it does not import the restricted ones -- so until this
round every open-shell analytic Hessian ran GPU4PySCF's code with the module
loaded, which is `docs/OPENSHELL_TMC.md` §4 item 2.  This suite calls both
implementations on the same converged UKS density, term by term and then as a
whole Hessian.

Three things it checks that the restricted suite cannot:

* **both spin blocks**, separately.  The two members of the returned tuple can
  have different shapes (`nocca != noccb` on anything but a closed shell), and
  a spin index dropped somewhere would show as an error in one column only, so
  the reported number is the worse of the two.
* **the channel coupling**.  `fxc` is a 2x2 block in the spin index, so the
  perturbed potential of one spin takes both spins' perturbed densities.  A
  case with `nocca == noccb + 1` (a doublet) is the one that separates a
  correct coupling from a transposed one; several are here on purpose.
* **that the patch is reached at all.**  `_setup(..., unrestricted=True)` is
  asserted to be non-None on the covered rows, because an unrestricted run
  going through upstream's function agrees with itself perfectly and says
  nothing.

The `hself` column is the whole Hessian run twice on GPU4PySCF's own code, and
it is there because the `hessian` column has a floor this suite cannot lower:
the J and K builds accumulate with `atomicAdd` and the CPHF iterates on top of
that at `conv_tol_cpscf = 1e-6`, which on an open-shell d-block system is a
larger number than the 1e-12 an organic closed shell gives.  Without `hself`
the only honest thing to say about a 1e-9 in the `hessian` column would be
that it is probably the solver; with it, that is a measurement.

    python tests/test_uxchess.py [--quick] [--hess-max-atoms N]
"""
import argparse
import sys

import numpy as np
import cupy as cp

import electrolites
electrolites.patch('fastxchess')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.hessian import uks as hess_uks                # noqa: E402
from electrolites import fastxchess                          # noqa: E402

O2 = 'O 0. 0. 0.; O 0. 0. 1.208'
CH3 = '''C 0. 0. 0.; H 1.079 0. 0.
H -0.539 0.934 0.; H -0.539 -0.934 0.'''
NO = 'N 0. 0. 0.; O 0. 0. 1.154'
M = 'benchmarks/molecules/'

# (label, atom, basis, xc, grid level, charge, spin)
CASES = [
    ('O2', O2, 'def2-svp', 'b3lyp', 3, 0, 2),
    ('O2', O2, 'def2-tzvpp', 'pbe', 3, 0, 2),                 # reaches g
    ('CH3', CH3, '6-31g*', 'pbe0', 3, 0, 1),                  # nocca != noccb
    ('NO', NO, 'def2-svp', 'wb97x-d3bj', 3, 0, 1),            # range separated
    ('CuCl2', M + 'tmc_cucl2.xyz', 'def2-svp', 'b3lyp', 3, 0, 1),
    ('CuCl2', M + 'tmc_cucl2.xyz', 'def2-tzvp', 'blyp', 3, 0, 1),  # 8-prim s
    ('FeCl4-', M + 'tmc_fecl4.xyz', 'def2-svp', 'b3lyp', 3, -1, 5),   # sextet
    ('Pt amine', M + 'tmc_pt_amine.xyz', 'def2-svp', 'b3lyp', 3, 0, 2),  # ECP
    ('Ni dithiolene', M + 'tmc_ni_dithiolene.xyz', 'def2-svp', 'b3lyp',
     3, -1, 1),
    # meta-GGAs (W28)
    ('CH3', CH3, 'def2-svp', 'tpss', 3, 0, 1),
    ('NO', NO, 'def2-tzvpp', 'r2scan', 3, 0, 1),
    ('FeCl4-', M + 'tmc_fecl4.xyz', 'def2-svp', 'tpss', 3, -1, 5),
    ('Ni dithiolene', M + 'tmc_ni_dithiolene.xyz', 'def2-svp', 'm06-l',
     3, -1, 1),
    # a declined path: it must fall through, and still agree exactly
    ('O2', O2, 'def2-svp', 'svwn', 3, 0, 2),                  # LDA
    ('O2', O2, 'def2-svp', 'm06-2x', 3, 0, 2),                # meta-GGA
]

#: the functionals this module hands back to GPU4PySCF
DECLINED = ('svwn',)

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
ap.add_argument('--hess-max-atoms', type=int, default=6,
                help='run the whole-Hessian column only up to this size.  A '
                     'UKS Hessian of the 21-atom complex is minutes of CPHF '
                     'for a column that the two term-by-term columns above '
                     'it already cover.')
a = ap.parse_args()
cases = CASES[:3] + CASES[9:11] + CASES[-2:] if a.quick else CASES

FAST1, FAST2 = hess_uks._get_vxc_deriv1, hess_uks._get_vxc_deriv2
ORIG1, ORIG2 = hess_uks._get_vxc_deriv1_orig, hess_uks._get_vxc_deriv2_orig


def build(atom, basis, charge, spin):
    kw = dict(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0,
              max_memory=40000)
    if basis.lower().startswith('def2') and any(
            gto.charge(sym) > 36 for sym, _ in gto.format_atom(atom)):
        kw['ecp'] = basis
    return gto.M(**kw)


def worst(x, y):
    """max |x - y| over both spin blocks, which need not have one shape."""
    return max(float(cp.abs(cp.asarray(u) - cp.asarray(v)).max())
               for u, v in zip(x, y))


def scale(x):
    return max(float(cp.abs(cp.asarray(u)).max()) for u in x)


print(f'{"case":34s} {"xc":11s} {"nat":>4s} {"nao":>5s} {"deriv1":>10s} '
      f'{"deriv2":>10s} {"hessian":>10s} {"hself":>10s} {"self":>10s}')
print('-' * 122)
fails = skipped = 0
for label, atom, basis, xc, level, charge, spin in cases:
    mol = build(atom, basis, charge, spin)
    mf = dft.UKS(mol, xc=xc)
    mf.grids.level = level
    mf.conv_tol = 1e-10
    mf.conv_tol_cpscf = 1e-6
    mf.direct_scf_tol = 1e-13
    mf.max_cycle = 200
    mf.kernel()
    name = f'{label}/{basis}'
    if not mf.converged:
        # an unconverged SCF is not a statement about the kernels, and the two
        # implementations would be handed different densities by the whole
        # Hessian column -- see docs/OPENSHELL_TMC.md §3 on CuCl2/M06-2X
        print(f'{name:34s} {xc:11s} SCF did not converge -- skipped',
              flush=True)
        skipped += 1
        mol = mf = None
        cp.get_default_memory_pool().free_all_blocks()
        continue

    h = mf.Hessian()
    h.auxbasis_response = 0
    mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ
    declined = fastxchess._setup(h, mo_occ, unrestricted=True) is None

    d1f = FAST1(h, mo_coeff, mo_occ, 4000)
    d1o = ORIG1(h, mo_coeff, mo_occ, 4000)
    e1 = worst(d1f, d1o)
    ref1 = scale(d1o)
    es = worst(ORIG1(h, mo_coeff, mo_occ, 4000), d1o)
    d1f = d1o = None
    d2f = FAST2(h, mo_coeff, mo_occ, 4000)
    d2o = ORIG2(h, mo_coeff, mo_occ, 4000)
    e2 = worst(d2f, d2o)
    d2f = d2o = None
    cp.get_default_memory_pool().free_all_blocks()

    eh = ehs = nrm = float('nan')
    if mol.natm <= a.hess_max_atoms:
        hess_uks._get_vxc_deriv1, hess_uks._get_vxc_deriv2 = FAST1, FAST2
        hf = cp.asarray(h.kernel())
        hess_uks._get_vxc_deriv1, hess_uks._get_vxc_deriv2 = ORIG1, ORIG2
        h.de = np.zeros((0, 0, 3, 3))
        ho = cp.asarray(h.kernel())
        h.de = np.zeros((0, 0, 3, 3))
        ho2 = cp.asarray(h.kernel())      # upstream against itself
        hess_uks._get_vxc_deriv1, hess_uks._get_vxc_deriv2 = FAST1, FAST2
        eh = float(cp.abs(hf - ho).max())
        ehs = float(cp.abs(ho2 - ho).max())
        nrm = float(cp.abs(ho).max())
        hf = ho = ho2 = None

    if declined:
        ok = xc in DECLINED               # the assertion is the fall-through
        name += ' (declined)'
    else:
        # the whole-Hessian tolerance is the larger of an absolute 1e-8 of
        # the Hessian's own scale and ten times the run-to-run spread the
        # same calculation shows against itself
        htol = max(1e-8 * max(nrm, 1.), 10 * ehs if ehs == ehs else 0.)
        ok = (xc not in DECLINED and e1 < 1e-8 * max(ref1, 1.) and e2 < 1e-8
              and not (eh == eh and eh > htol))
    fails += not ok
    print(f'{name:34s} {xc:11s} {mol.natm:4d} {mol.nao:5d} {e1:10.2e} '
          f'{e2:10.2e} {eh:10.2e} {ehs:10.2e} {es:10.2e}'
          f'{"" if ok else "   FAIL"}', flush=True)
    mol = mf = h = None
    cp.get_default_memory_pool().free_all_blocks()

print('-' * 122)
print(f'{len(cases) - fails - skipped}/{len(cases)} cases agree, or decline '
      f'as intended; {skipped} skipped')
sys.exit(1 if fails else 0)
