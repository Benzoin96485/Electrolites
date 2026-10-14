"""Double-hybrid energies and gradients on the GPU (``electrolites.dhdft``,
FEATURE_TARGETS P4).

Neither PySCF nor GPU4PySCF has double hybrids, so: the energy against the
manual recipe on the CPU (PySCF RKS with the same DFT part, grid and
auxiliary basis, plus PySCF's DF-MP2 on its orbitals, scaled), and the
gradient against central finite differences (Richardson, steps 4e-3/2e-3
Bohr) of the GPU energy on a (99, 590) grid.  Neither side has the grid
response, so the gate is 2e-6.

    python tests/test_dhdft.py
"""
import sys

import numpy as np

from pyscf import gto
from electrolites.dhdft import DoubleHybrid

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:55s} {detail}', flush=True)


WATER = [['O', [0., 0., 0.2]], ['H', [0., 1.5, -0.9]], ['H', [0.1, -1.4, -1.0]]]
GRID = (99, 590)


def make(atom, xc, frozen=None):
    mol = gto.M(atom=atom, basis='cc-pvdz', unit='Bohr', verbose=0)
    dh = DoubleHybrid(mol, xc, auxbasis='cc-pvdz-jkfit',
                      ri_auxbasis='cc-pvdz-ri', frozen=frozen)
    dh._scf.grids.atom_grid = GRID
    dh._scf.conv_tol = 1e-12
    dh._scf.conv_tol_grad = 1e-9
    return dh


print('energy against the CPU recipe')
dh = make(WATER, 'B2PLYP')
e = dh.kernel()
from pyscf import dft as cdft                                   # noqa: E402
from pyscf.mp import dfmp2 as cdfmp2                            # noqa: E402
mf = cdft.RKS(dh.mol, xc=dh._scf.xc).density_fit(auxbasis='cc-pvdz-jkfit')
mf.grids.atom_grid = GRID
mf.conv_tol = 1e-12
mf.kernel()
pt = cdfmp2.DFMP2(mf)
pt.with_df.auxbasis = 'cc-pvdz-ri'
pt.kernel()
eref = mf.e_tot + .27 * pt.e_corr
report('B2PLYP water: GPU = PySCF RKS + 0.27 DF-MP2', abs(e - eref) < 1e-8,
       f'{abs(e - eref):.1e}')

print('gradient against finite differences')


def fd_check(label, xc, frozen, comps):
    dh = make(WATER, xc, frozen)
    dh.kernel()
    an = dh.Gradients().kernel()
    worst = 0.
    for A, x in comps:
        f = []
        for h in (4e-3, 2e-3):
            ap = [[s, list(c)] for s, c in WATER]
            am = [[s, list(c)] for s, c in WATER]
            ap[A][1][x] += h
            am[A][1][x] -= h
            f.append((make(ap, xc, frozen).kernel()
                      - make(am, xc, frozen).kernel()) / (2 * h))
        worst = max(worst, abs(an[A, x] - (4 * f[1] - f[0]) / 3))
    report(label, worst < 2e-6, f'{worst:.1e}')


allc = [(A, x) for A in range(3) for x in range(3)]
fd_check('B2PLYP (GGA), water, all components', 'B2PLYP', None, allc)
fd_check('PBE0-DH (GGA), frozen core, 4 components', 'PBE0-DH', 1,
         [(0, 2), (1, 1), (1, 2), (2, 0)])
fd_check('custom LDA double hybrid, 4 components',
         ('0.5*HF + 0.5*SLATER, 0.75*VWN5', .25, .2), None,
         [(0, 2), (1, 1), (1, 2), (2, 0)])

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
