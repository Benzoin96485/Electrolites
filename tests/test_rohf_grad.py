"""GPU ROHF/ROKS gradients (`electrolites/rohf_grad.py`) against PySCF 2.13.

The same molecule and functional converged on the GPU and on the CPU; the
GPU gradient (direct and density-fitted, with and without ``patch_all()``'s
guards loaded first) against PySCF's ``ROHF/ROKS.Gradients`` to 1e-6
Eh/bohr, and a finite-difference check of the direct ROKS one.

    python tests/test_rohf_grad.py [--patch-all]
"""
import sys

import numpy as np

import electrolites
if '--patch-all' in sys.argv:
    electrolites.patch_all()
electrolites.enable('rohf_grad')

from pyscf import gto, scf as cscf, dft as cdft             # noqa: E402
from gpu4pyscf import scf, dft                              # noqa: E402

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:44s} {detail}', flush=True)


mols = {
    'H2O+ (doublet)': gto.M(atom='O 0 0 0.1; H 0 0.76 -0.45; H 0 -0.76 -0.45',
                            basis='def2-svp', charge=1, spin=1, verbose=0),
    'CH2 (triplet)': gto.M(atom='C 0 0 0; H 0 0.99 0.6; H 0 -0.99 0.6',
                           basis='def2-svp', spin=2, verbose=0),
}
for name, mol in mols.items():
    print(name)
    for label, gmk, cmk in (
            ('ROHF', lambda m: scf.ROHF(m), lambda m: cscf.ROHF(m)),
            ('ROKS PBE', lambda m: dft.ROKS(m, xc='pbe'),
             lambda m: cdft.ROKS(m, xc='pbe')),
            ('ROKS B3LYP', lambda m: dft.ROKS(m, xc='b3lyp'),
             lambda m: cdft.ROKS(m, xc='b3lyp')),
            ('DF-ROKS B3LYP', lambda m: dft.ROKS(m, xc='b3lyp').density_fit(),
             lambda m: cdft.ROKS(m, xc='b3lyp').density_fit()),
            ('DF-ROHF', lambda m: scf.ROHF(m).density_fit(),
             lambda m: cscf.ROHF(m).density_fit())):
        g = gmk(mol).run(conv_tol=1e-11)
        gg = g.Gradients()
        de = gg.kernel()
        ref = cmk(mol).run(conv_tol=1e-11).Gradients().kernel()
        err = abs(np.asarray(de) - ref).max()
        report(f'{label} vs PySCF ({type(gg).__name__})',
               err < 1e-6 and 'electrolites' in type(gg).__module__,
               f'{err:.1e}')

print('finite differences, ROKS B3LYP, H2O+ O-z')
mol = mols['H2O+ (doublet)']
de = dft.ROKS(mol, xc='b3lyp').run(conv_tol=1e-12).Gradients().kernel()
c0 = mol.atom_coords()
es = []
for s in (1, -1):
    c = c0.copy()
    c[0, 2] += s * 1e-4
    m = mol.set_geom_(c, unit='Bohr', inplace=False)
    es.append(dft.ROKS(m, xc='b3lyp').run(conv_tol=1e-12).e_tot)
fd = (es[0] - es[1]) / 2e-4
report('analytic vs FD', abs(de[0, 2] - fd) < 1e-5, f'{de[0, 2]:.8f} vs {fd:.8f}')

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
