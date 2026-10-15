"""The GIAO two-electron term on the GPU through shifted-basis J/K
(``electrolites.giao_jk``, FEATURE_TARGETS O1) against libcint's
``int2e_ig1`` (GPU4PySCF's ``shielding.get_jk``) on the same density.

    python tests/test_giao_jk.py
"""
import sys

import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.properties import shielding

from electrolites import giao_jk

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


rng = np.random.default_rng(7)
for label, atom, basis in (
        ('water cc-pVDZ', 'O 0 0.1 0; H 0.2 -0.757 0.587; H 0 0.757 0.587',
         'ccpvdz'),
        ('glycine def2-SVP (d -> f)',
         '/global/homes/w/weiliang/vibeqc/Electrolites/benchmarks/molecules/'
         'glycine.xyz', 'def2-svp'),
        ('HCl def2-TZVP (f -> g)', 'H 0 0 0; Cl 0 0 1.28', 'def2-tzvp')):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    d = rng.random((mol.nao, mol.nao))
    d = cp.asarray(d + d.T)
    vj0, vk0 = shielding.get_jk(mol, d)
    vj1, vk1 = giao_jk.get_jk(mol, d)
    ej = float(abs(vj1 - vj0).max() / abs(vj0).max())
    ek = float(abs(vk1 - vk0).max() / abs(vk0).max())
    report(f'{label}: GIAO J, K', max(ej, ek) < 1e-10, f'{ej:.1e}, {ek:.1e}')
    d2 = cp.stack([d, d * .5 + d.T * .1 + d @ d * 1e-3])
    d2 = .5 * (d2 + d2.transpose(0, 2, 1))
    vj0, vk0 = shielding.get_jk(mol, d2)
    vj1, vk1 = giao_jk.get_jk(mol, d2)
    e2 = float(max(abs(vj1 - vj0).max() / abs(vj0).max(),
                   abs(vk1 - vk0).max() / abs(vk0).max()))
    report(f'{label}: two spin densities', e2 < 1e-10, f'{e2:.1e}')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
