"""The three-centre one-electron integrals, against PySCF's CPU intor().

    python tests/test_fast1e.py --patch ''       > a.json
    python tests/test_fast1e.py --patch fast1e   > b.json

Run twice and diff, as with test_full.py: `fast1e` patches
`gpu4pyscf.gto.int3c1e` globally, so a same-process A/B would leave it on.

The reference is `mol.intor('int1e_grids')` on the CPU, not GPU4PySCF's GPU
kernels, so a pass says the integrals are right rather than that they match the
code they replace.

The threshold is 1e-12 **relative to the largest element** of the reference.
GPU4PySCF's own `gto/tests/test_int1e_grids.py` uses absolute 1e-12 / 1e-10 /
1e-12, which are calibrated to its grid; on the grid here the unpatched arm
itself lands at 8.5e-12 absolute on the charge contraction (of a tensor whose
largest element is 24.8), so an absolute bound would fail GPU4PySCF too and
measure the grid rather than the kernels.  Relative 1e-12 clears both arms with
a factor of three in hand, and is tighter than upstream's on the contractions.

Coverage that matters for this module:

* **def2-QZVPP** reaches s, p, d, f and g shells, so every angular-momentum
  class the dispatch can select is exercised -- the (ss| and (ps| kernels the
  far-field path rewrites, and the two- to five-root kernels where it replaces
  the Rys root algebra.
* the grid **includes the nuclei**, where r2 -> 0 and the far-field branch must
  *not* be taken, alongside points several Bohr out where it must.
* **Gaussian charge exponents** on and off: with them the charge-smearing
  factor cancels analytically in the far field, which is the step most likely
  to be wrong.
* **cart and sph**, because the two take different cart2sph paths.
"""
import argparse
import json
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--patch', default='', help="'fast1e' or '' for plain GPU4PySCF")
a = ap.parse_args()

import electrolites
if a.patch:
    electrolites.patch(a.patch.split(','))
print('patched:', electrolites.patched() or '(none)',
      '| gpu4pyscf', electrolites.gpu4pyscf_version(), file=sys.stderr)

import cupy as cp
import pyscf
from pyscf import lib
from gpu4pyscf.gto.int3c1e import int1e_grids

ATOM = '''
O   0.0000   0.7375  -0.0528
O   0.0000  -0.7375  -0.0528
H   0.8190   0.8170   0.4220
H  -0.8190  -0.8170   0.4220
'''
#: relative to the largest element of the CPU reference
TOL = 1e-12

xs = np.arange(-2.01, 2.0, 0.5)
GRID = lib.cartesian_prod([xs, xs + 0.01, xs + 0.02])

results = {}
fails = []


def check(name, ref, got):
    ref = np.asarray(ref)
    got = cp.asnumpy(got) if isinstance(got, cp.ndarray) else np.asarray(got)
    err = float(np.abs(ref - got).max())
    scale = float(np.abs(ref).max())
    rel = err / max(scale, 1.0)
    results[name] = {'max_abs_err': err, 'max_abs_value': scale, 'rel': rel}
    if not (rel <= TOL):
        fails.append(f'{name}: {rel:.3e} > {TOL:.0e} relative')
    print(f'{"FAIL" if rel > TOL else "ok  "} {name:52s} '
          f'rel {rel:.3e}  (abs {err:.3e}, |ref|max {scale:.3e})',
          file=sys.stderr)


for basis in ('def2-QZVPP', '6-31G*'):
    for cart in (False, True):
        mol = pyscf.M(atom=ATOM, basis=basis, cart=cart,
                      max_memory=32000, output='/dev/null', verbose=0)
        # the nuclei themselves are in the grid: r2 = 0 there, which is the one
        # place the far-field branch must not fire
        grid = np.vstack([GRID, mol.atom_coords()])
        tag = f'{basis}/{"cart" if cart else "sph"}'

        rng = np.random.RandomState(12345)
        dm = rng.uniform(-2.0, 2.0, (mol.nao, mol.nao))
        dm_sym = 0.5 * (dm + dm.T)
        charges = rng.uniform(-2.0, 2.0, grid.shape[0])

        for exp_tag, cexp in (('bare', None), ('gauss', 3.0)):
            if cexp is None:
                ref = mol.intor('int1e_grids', grids=grid)
            else:
                # PySCF's Gaussian-charge model, the same one PCM uses
                fakemol = pyscf.gto.fakemol_for_charges(grid, expnt=cexp)
                # the charge shells are s functions, so cart and sph are the
                # same basis -- but aux_e2 refuses a mixed pair outright
                fakemol.cart = mol.cart
                ref = pyscf.df.incore.aux_e2(
                    mol, fakemol, intor='int3c2e').transpose(2, 0, 1)
            ce = None if cexp is None else np.full(grid.shape[0], cexp)

            check(f'tensor          {tag}/{exp_tag}', ref,
                  int1e_grids(mol, grid, charge_exponents=ce))
            check(f'density sym     {tag}/{exp_tag}',
                  np.einsum('pij,ij->p', ref, dm_sym),
                  int1e_grids(mol, grid, charge_exponents=ce, dm=dm_sym))
            check(f'density asym    {tag}/{exp_tag}',
                  np.einsum('pij,ij->p', ref, dm),
                  int1e_grids(mol, grid, charge_exponents=ce, dm=dm))
            check(f'charge          {tag}/{exp_tag}',
                  np.einsum('pij,p->ij', ref, charges),
                  int1e_grids(mol, grid, charge_exponents=ce, charges=charges))
        mol.stdout.close()

print(json.dumps({'patched': list(electrolites.patched()),
                  'gpu4pyscf': electrolites.gpu4pyscf_version(),
                  'results': results, 'failures': fails}, indent=1))
if fails:
    print(f'{len(fails)} FAILED', file=sys.stderr)
sys.exit(1 if fails else 0)
