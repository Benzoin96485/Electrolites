"""k-point periodic TDA on the GPU (``electrolites.pbc_tdscf.KTDA``,
FEATURE_TARGETS X2) against PySCF's ``pbc.tdscf.KTDA`` (kshift 0) on the
same SCF.

    python tests/test_pbc_ktda.py
"""
import sys

import numpy as np
from pyscf.pbc import gto, tdscf
from gpu4pyscf.pbc import dft, scf

from electrolites.pbc_tdscf import KTDA

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
for mesh in ([1, 1, 1], [2, 1, 1], [2, 2, 1]):
    kpts = cell.make_kpts(mesh)
    for xc in ('pbe', 'pbe0', 'HF'):
        if mesh != [2, 1, 1] and xc != ('pbe' if mesh == [1, 1, 1] else 'pbe0'):
            continue
        mf = scf.KRHF(cell, kpts) if xc == 'HF' else dft.KRKS(cell, kpts, xc=xc)
        mf.conv_tol = 1e-10
        mf.kernel()
        for singlet in (True, False):
            g = KTDA(mf, singlet=singlet)
            g.conv_tol = 1e-8
            g.kernel(nstates=4)
            c = tdscf.KTDA(mf.to_cpu())
            c.singlet, c.nstates, c.conv_tol, c.kshift_lst = singlet, 4, 1e-9, [0]
            c.kernel()
            err = abs(np.sort(g.e)[:4] - np.sort(np.asarray(c.e[0]))[:4]).max()
            report(f'diamond k{mesh} {xc} {"singlet" if singlet else "triplet"}',
                   err * HA2EV < 1e-5, f'max|dE| {err * HA2EV:.1e} eV')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
