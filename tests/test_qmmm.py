"""QM/MM's QM-nucleus--MM-charge terms against GPU4PySCF's (F6 / F13).

``electrolites/fastqmmm.py`` computes ``QMMMSCF.energy_nuc``,
``QMMMGrad.grad_nuc`` and ``QMMMGrad.grad_nuc_mm`` on the device and caches
them against the coordinates.  Checked against the replaced methods
(``_energy_nuc_orig`` etc.) on the same objects, relative to the largest
element; then that moving the MM charges (and the QM atoms) invalidates the
caches, and that a whole QM/MM SCF + gradient matches ``FASTQMMM_OFF``'s
energy and forces when run in a subprocess.

    python tests/test_qmmm.py
"""
import os
import subprocess
import sys

import numpy as np

TOL = 1e-12
fails = []


def check(label, got, ref):
    got, ref = np.asarray(got, float), np.asarray(ref, float)
    err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-300)
    ok = err <= TOL
    if not ok:
        fails.append(f'{label}: {err:.3e}')
    print(f'  {"ok  " if ok else "FAIL"} {label:44s} {err:.3e}', flush=True)


def charges(n, seed, mol):
    """Neutral +-0.4 e dipoles 1 A long, no closer than 4 A to a QM atom."""
    rng = np.random.default_rng(seed)
    qm = mol.atom_coords(unit='Angstrom')
    c0 = qm.mean(axis=0)
    out = []
    while len(out) < n:
        p = c0 + rng.uniform(-30, 30, 3)
        if np.linalg.norm(qm - p, axis=1).min() < 4:
            continue
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        out += [p, p + d]
    xyz = np.array(out[:n])
    q = np.tile([.4, -.4], n // 2 + 1)[:n]
    return xyz, q


def whole(off, n):
    code = f'''
import os, json, numpy as np
sys_path = {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r}
import sys; sys.path.insert(0, sys_path)
if {off!r}: os.environ["FASTQMMM_OFF"] = "1"
import electrolites; electrolites.patch_all()
from pyscf import gto
from gpu4pyscf import dft, qmmm
sys.path.insert(0, sys_path + "/tests")
from test_qmmm import charges
mol = gto.M(atom=ETHANOL, basis="def2-svp", verbose=0)
xyz, q = charges({n}, 7, mol)
mf = qmmm.mm_charge(dft.RKS(mol, xc="b3lyp"), xyz, q)
mf.conv_tol = 1e-11
e = mf.kernel()
g = mf.Gradients()
de = g.kernel()
dm = mf.make_rdm1()
gmm = g.grad_nuc_mm() + g.grad_hcore_mm(dm)
print(json.dumps([e, np.asarray(de).tolist(), np.asarray(gmm).tolist()]))
'''.replace('ETHANOL', repr(ETHANOL))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, check=True).stdout
    import json
    return json.loads(out.strip().splitlines()[-1])


ETHANOL = '''C -0.748 -0.015 0.024; C 0.558 0.420 -0.679;
O 1.598 -0.513 -0.421; H -1.593 0.365 -0.554; H -0.834 0.355 1.048;
H -0.780 -1.106 0.023; H 0.640 1.502 -0.580; H 0.567 0.174 -1.741;
H 2.421 -0.190 -0.792'''

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import electrolites
    electrolites.patch_all()
    from pyscf import gto
    from gpu4pyscf import dft, qmmm
    from gpu4pyscf.qmmm import itrf as I
    from electrolites import fastqmmm

    mol = gto.M(atom=ETHANOL, basis='def2-svp', verbose=0)
    for n in (1000, 150001):          # the second spans three device blocks
        xyz, q = charges(n, 3, mol)
        mf = qmmm.mm_charge(dft.RKS(mol, xc='b3lyp'), xyz, q)
        g = mf.Gradients()
        print(f'## {n} charges')
        check('energy_nuc', mf.energy_nuc(), I.QMMMSCF._energy_nuc_orig(mf))
        check('energy_nuc (cached)', mf.energy_nuc(),
              I.QMMMSCF._energy_nuc_orig(mf))
        check('grad_nuc', g.grad_nuc(), I.QMMMGrad._grad_nuc_orig(g))
        check('grad_nuc_mm', g.grad_nuc_mm(), I.QMMMGrad._grad_nuc_mm_orig(g))
        # move the charges: every cache must notice
        mf.mm_mol.set_geom_(mf.mm_mol.atom_coords() + .1, unit='Bohr')
        check('energy_nuc after moving MM', mf.energy_nuc(),
              I.QMMMSCF._energy_nuc_orig(mf))
        check('grad_nuc_mm after moving MM', g.grad_nuc_mm(),
              I.QMMMGrad._grad_nuc_mm_orig(g))
        mol2 = mol.set_geom_(mol.atom_coords() * 1.01, unit='Bohr',
                             inplace=False)
        check('grad_nuc on a moved QM molecule', g.grad_nuc(mol2),
              I.QMMMGrad._grad_nuc_orig(g, mol2))
    print('# fastqmmm', fastqmmm.stats())

    print('## whole QM/MM SCF + gradient, against FASTQMMM_OFF=1')
    a, b = whole(False, 20000), whole(True, 20000)
    for lbl, x, y in (('energy', a[0], b[0]), ('QM forces', a[1], b[1]),
                      ('MM forces', a[2], b[2])):
        err = np.abs(np.asarray(x) - np.asarray(y)).max()
        ok = err <= 1e-9
        if not ok:
            fails.append(f'whole {lbl}: {err:.3e}')
        print(f'  {"ok  " if ok else "FAIL"} {"whole " + lbl:44s} {err:.3e}')
    if fails:
        print('\nFAILED:\n  ' + '\n  '.join(fails))
        sys.exit(1)
    print('\nall ok')
