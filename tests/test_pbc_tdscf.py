"""Gamma-point periodic TDA on the GPU (``electrolites.pbc_tdscf``,
FEATURE_TARGETS X1) against PySCF's ``pbc.tdscf.TDA`` on the same SCF.

    python tests/test_pbc_tdscf.py
"""
import sys

import numpy as np
from pyscf.pbc import gto, tdscf
from gpu4pyscf.pbc import dft, scf

from electrolites.pbc_tdscf import TDA

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


cell = gto.M(a='''0 1.7834 1.7834
1.7834 0 1.7834
1.7834 1.7834 0''', atom='C 0 0 0; C 0.8917 0.8917 0.8917',
             basis='gth-dzv', pseudo='gth-pade', verbose=0, mesh=[25] * 3)
HA2EV = 27.211386245988
for xc in ('lda', 'pbe', 'pbe0', 'HF'):
    mf = scf.RHF(cell) if xc == 'HF' else dft.RKS(cell, xc=xc)
    mf.conv_tol = 1e-10
    mf.kernel()
    for singlet in (True, False):
        g = TDA(mf, singlet=singlet)
        g.conv_tol = 1e-7
        g.kernel(nstates=4)
        c = tdscf.TDA(mf.to_cpu())
        c.singlet, c.nstates, c.conv_tol = singlet, 4, 1e-9
        c.kernel()
        err = abs(np.sort(g.e) - np.sort(c.e)).max() * HA2EV
        report(f'diamond {xc} {"singlet" if singlet else "triplet"}',
               err < 1e-5, f'max|dE| {err:.1e} eV')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
