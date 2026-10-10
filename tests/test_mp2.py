"""Conventional MP2's (ia|jb) build (I3).

``electrolites/fastmp2.py`` replaces ``int4c2e.get_int4c2e_ovov`` with a
blocked half transformation.  Checked against GPU4PySCF's own build, element
by element (bar 1e-12), on glycine in def2-TZVP (d and f classes) with the
automatic batch, with 5 occupied orbitals per batch, and with the result on
the host; and the RMP2 correlation energy against PySCF's CPU MP2 on water
(bar 1e-10 Eh).

    python tests/test_mp2.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import cupy as cp
import numpy as np
import electrolites
electrolites.patch_all()
from electrolites import fastmp2
from pyscf import gto
from pyscf import scf as cpu_scf
from pyscf.mp import mp2 as cpu_mp2
from gpu4pyscf import scf
from gpu4pyscf.mp import mp2
from gpu4pyscf.scf import int4c2e

GLY = os.path.join(HERE, 'benchmarks', 'molecules', 'glycine.xyz')


def check_ovov():
    mol = gto.M(atom=GLY, basis='def2-tzvp', verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    nocc = mol.nelectron // 2
    co = cp.asarray(mf.mo_coeff[:, :nocc])
    cv = cp.asarray(mf.mo_coeff[:, nocc:])
    ref = int4c2e._get_int4c2e_ovov_orig(mol, co, cv)
    ok = True
    for label, env in (('auto batch', {}), ('5 per batch', {'FASTMP2_BATCH': '5'}),
                       ('host result', {'FASTMP2_HOST': '1', 'FASTMP2_BATCH': '7'})):
        os.environ.update(env)
        try:
            new = int4c2e.get_int4c2e_ovov(mol, co, cv)
        finally:
            for k in env:
                del os.environ[k]
        new = cp.asarray(new).reshape(ref.shape)
        err = float(abs(new - ref).max())
        good = err < 1e-12
        ok &= good
        print(f'glycine/def2-tzvp ovov, {label}: max|new - stock| = {err:.2e}'
              f'  {"ok" if good else "FAIL"}')
    return ok


def check_energy():
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    e_gpu = mp2.MP2(mf).kernel()[0]
    mfc = cpu_scf.RHF(mol)
    mfc.conv_tol = 1e-12
    mfc.kernel()
    e_cpu = cpu_mp2.MP2(mfc).kernel()[0]
    err = abs(float(e_gpu) - e_cpu)
    good = err < 1e-10 and fastmp2.STATS['calls'] > 0
    print(f'water/cc-pvdz RMP2 e_corr: GPU {float(e_gpu):.12f} CPU {e_cpu:.12f}'
          f'  |diff| {err:.1e}  {"ok" if good else "FAIL"}')
    return good


if __name__ == '__main__':
    ok = check_ovov()
    ok &= check_energy()
    print('ALL OK' if ok else 'FAILED')
    sys.exit(0 if ok else 1)
