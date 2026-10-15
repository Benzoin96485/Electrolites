"""The QM/MM Hessian (``electrolites.qmmm_hess``, FEATURE_TARGETS V2) against
central finite differences (1e-3 bohr) of GPU4PySCF's QM/MM gradient,
point and Gaussian MM charges, RHF and PBE.  PySCF has no QM/MM Hessian.

The KS Hessian has no grid response, so the PBE gate is the vacuum PBE
Hessian's own distance from its finite difference (3.8e-4 here).

    python tests/test_qmmm_hess.py
"""
import sys

import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf import dft, scf
from gpu4pyscf.qmmm import itrf

import electrolites
electrolites.patch_all()

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


B = 0.52917721092
xyz = np.array([[0, 0, 0], [0, -0.757, 0.587], [0, 0.757, 0.587]])
mm_xyz = np.array([[1.5, 0.3, 2.0], [-1.8, 0.4, -1.2], [0.2, 2.5, 0.7]])
mm_q = np.array([-0.8, 0.4, 0.4])


def build(x):
    return gto.M(atom=[('O', x[0]), ('H', x[1]), ('H', x[2])],
                 basis='def2-svp', verbose=0, unit='Angstrom')


for label, mk, gate in (('RHF', scf.RHF, 1e-5),
                        ('PBE', lambda m: dft.RKS(m, xc='pbe'), 5e-4)):
    for rad in (None, np.array([0.8, 0.5, 0.5])):
        def run(x):
            return itrf.mm_charge(mk(build(x)), mm_xyz, mm_q, radii=rad,
                                  unit='Angstrom').run(conv_tol=1e-12)
        mf = run(xyz)
        H = mf.Hessian().kernel()
        h = 1e-3
        fd = np.zeros_like(H)
        for a in range(3):
            for x in range(3):
                xp = xyz.copy()
                xp[a, x] += h * B
                xm = xyz.copy()
                xm[a, x] -= h * B
                fd[a, :, x, :] = (cp.asnumpy(run(xp).nuc_grad_method().kernel())
                                  - cp.asnumpy(run(xm).nuc_grad_method().kernel())
                                  ) / (2 * h)
        err = abs(H - fd).max()
        vac = abs(mf.undo_qmmm().Hessian().kernel() - fd).max()
        report(f'{label} {"Gaussian" if rad is not None else "point"} charges '
               'against FD', err < gate,
               f'{err:.1e} (vacuum Hessian {vac:.1e})')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
