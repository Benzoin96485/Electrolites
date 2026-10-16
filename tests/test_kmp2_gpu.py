"""k-point MP2 on the GPU (``electrolites.kmp2_gpu``) against PySCF's
``pbc.mp.KMP2`` on the same KRHF orbitals and the same GDF auxiliary basis
(PySCF rebuilds the cderi on the host; the two GDF builds differ at
~1e-8 in the correlation energy).

    python tests/test_kmp2_gpu.py
"""
import sys

import numpy as np
from pyscf.pbc import gto, mp
from gpu4pyscf.pbc import scf
from gpu4pyscf.pbc.df.df import GDF

from electrolites import kmp2_gpu

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


cell = gto.M(a='''0 1.7834 1.7834
1.7834 0 1.7834
1.7834 1.7834 0''', atom='C 0 0 0; C 0.8917 0.8917 0.8917',
             basis='gth-dzvp', pseudo='gth-pade', verbose=0)
for kmesh in ([2, 2, 2], [3, 3, 3]):
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts)
    mf.with_df = GDF(cell, kpts)
    mf.conv_tol = 1e-10
    mf.kernel()
    e1 = kmp2_gpu.kernel(mf)
    mc = mf.to_cpu()
    mc.max_memory = 100000
    pm = mp.KMP2(mc)
    pm.max_memory = 100000
    e0 = pm.kernel()[0]
    # the GDF tensors are built twice (GPU and host); their difference shows
    # in the HF energy of the same density, and bounds the MP2 difference
    dm = mc.make_rdm1()
    ehf = abs(mc.energy_tot(dm) - mf.e_tot)
    report(f'diamond {kmesh} KMP2 == PySCF', abs(e0 - e1) < max(1e-7, 10 * ehf),
           f'{abs(e0 - e1):.1e} (E_corr {e1:.10f}; HF energy of the two '
           f'GDF builds differs by {ehf:.1e})')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
