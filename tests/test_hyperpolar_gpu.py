"""Static first hyperpolarizability on the GPU
(``electrolites.polar_gpu.hyperpolarizability``), RHF/RKS and UHF/UKS.

References: pyscf-properties' ``hyper_polarizability`` for RHF, and for
every case a central finite field of the analytic polarizability
(F = 1e-3 a.u., error O(F^2)).  pyscf-properties' KS beta leaves out the
third functional derivative of the XC energy, which moves water's PBE beta
by 1.6 a.u. of 16; with it the analytic beta meets the finite field.

    python tests/test_hyperpolar_gpu.py
"""
import sys

import numpy as np
import cupy as cp
from pyscf import gto
from pyscf.prop.polarizability import rhf as prhf
from gpu4pyscf import scf, dft

from electrolites import polar_gpu as pg

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


def finite_field(mol, mk, F=1e-3):
    r = pg._dipole_ints(mol)
    out = np.zeros((3, 3, 3))
    for z in range(3):
        al = []
        for sg in (1, -1):
            mf = mk(mol)
            h0 = mf.get_hcore()
            mf.get_hcore = lambda *a, _h=h0, _s=sg, _z=z: _h + _s * F * r[_z]
            mf.run(conv_tol=1e-12)
            al.append(pg.polarizability(mf, tol=1e-11))
        out[:, :, z] = (al[0] - al[1]) / (2 * F)
    return out


water = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
              basis='aug-cc-pvdz', verbose=0)
oh = gto.M(atom='O 0 0 0; H 0 0 0.97; H 0.9 0 -0.3', charge=1, spin=1,
           basis='aug-cc-pvdz',
           verbose=0)
cases = (('water RHF', water, scf.RHF),
         ('water LDA', water, lambda m: dft.RKS(m, xc='lda,vwn')),
         ('water PBE', water, lambda m: dft.RKS(m, xc='pbe')),
         ('water B3LYP', water, lambda m: dft.RKS(m, xc='b3lyp')),
         ('H2O+ UHF', oh, scf.UHF),
         ('H2O+ UKS PBE', oh, lambda m: dft.UKS(m, xc='pbe')),
         ('H2O+ UKS B3LYP', oh, lambda m: dft.UKS(m, xc='b3lyp')))
for label, mol, mk in cases:
    mf = mk(mol).run(conv_tol=1e-12)
    b = pg.hyperpolarizability(mf)
    f = finite_field(mol, mk)
    err = abs(b - f).max()
    report(f'{label} against finite field', err < 3e-3 * max(1, abs(f).max()),
           f'{err:.1e} (max |beta| {abs(f).max():.2f})')
    if label == 'water RHF':
        p = prhf.Polarizability(mf.to_cpu())
        p.conv_tol = 1e-11
        e = abs(b - prhf.hyper_polarizability(p)).max()
        report('water RHF against pyscf-properties', e < 1e-4, f'{e:.1e}')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
