"""sfX2C1e nuclear gradients on the GPU (``electrolites.x2c_grad``) against
PySCF's ``x2c.sfx2c1e_grad`` (and a finite difference of the GPU X2C
energy) on RHF, RKS and UKS.

    python tests/test_x2c_grad.py
"""
import sys

import electrolites

import numpy as np
from pyscf import gto, scf as pscf, dft as pdft
from gpu4pyscf import scf, dft
from gpu4pyscf.x2c.sfx2c1e import sfx2c1e

from electrolites import x2c_grad

electrolites.patch_all()
fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


hbr = 'Br 0 0 0; H 0 0.1 1.42'
cases = (
    ('HBr RHF', hbr, 0, 'cc-pvdz-dk', lambda m: scf.RHF(m),
     lambda m: pscf.RHF(m)),
    ('HBr RKS PBE', hbr, 0, 'cc-pvdz-dk', lambda m: dft.RKS(m, xc='pbe'),
     lambda m: pdft.RKS(m, xc='pbe')),
    ('BrO UKS B3LYP', 'Br 0 0 0; O 0 0.2 1.72', 1, 'cc-pvdz-dk',
     lambda m: dft.UKS(m, xc='b3lyp'), lambda m: pdft.UKS(m, xc='b3lyp')),
)
for label, atom, spin, basis, mk, mkc in cases:
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    mf = sfx2c1e(mk(mol))
    mf.conv_tol = 1e-12
    mf.kernel()
    g = x2c_grad.Gradients(mf).kernel()
    mc = mkc(mol).x2c()
    mc.conv_tol = 1e-12
    mc.kernel()
    gc = mc.Gradients().kernel()
    d = abs(g - gc).max()
    report(f'{label} gradient == PySCF', d < 2e-6 and abs(mf.e_tot - mc.e_tot)
           < 1e-7, f'{d:.1e} (E diff {abs(mf.e_tot - mc.e_tot):.1e})')
    if label == 'HBr RHF':
        dx = 1e-4
        es = []
        for sg in (1, -1):
            c = mol.atom_coords()
            c[1, 2] += sg * dx
            m2 = mol.set_geom_(c, unit='Bohr', inplace=False)
            f = sfx2c1e(mk(m2))
            f.conv_tol = 1e-12
            es.append(f.kernel())
        fd = (es[0] - es[1]) / (2 * dx)
        report('HBr RHF gradient against finite difference',
               abs(fd - g[1, 2]) < 1e-6, f'{abs(fd - g[1, 2]):.1e}')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
