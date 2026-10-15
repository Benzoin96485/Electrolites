"""NMR spin-spin coupling on the GPU (``electrolites.ssc_gpu``,
FEATURE_TARGETS O2) against pyscf-properties' ``prop.ssc`` on the same SCF.

The reference needs pyscf-properties (``pip install electrolites[properties]``,
pinned in pyproject.toml).

    python tests/test_ssc_gpu.py
"""
import sys

import numpy as np
from pyscf import gto
from pyscf.prop import ssc as cpu_ssc
from gpu4pyscf import dft, scf

from electrolites.ssc_gpu import SSC

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


water = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
              basis='ccpvdz', verbose=0)
gly = gto.M(atom='/global/homes/w/weiliang/vibeqc/Electrolites/benchmarks/'
            'molecules/glycine.xyz', basis='def2-svp', verbose=0)
cases = (('water HF', water, scf.RHF, None),
         ('water B3LYP', water, lambda m: dft.RKS(m, xc='b3lyp'), None),
         ('water PBE', water, lambda m: dft.RKS(m, xc='pbe'), None),
         ('glycine B3LYP/DF, 3 pairs', gly,
          lambda m: dft.RKS(m, xc='b3lyp').density_fit(), [(1, 0), (5, 2),
                                                            (8, 3)]))
# (pyscf-properties' make_pso reshapes the PSO response by the first atoms
# of nuc_pair, solved for the second: pairs whose two atom sets differ in
# size fail there, so the subset here has three of each.)
for label, mol, mk, pairs in cases:
    mf = mk(mol).run(conv_tol=1e-11)
    ref = cpu_ssc.RHF if label.split()[1] == 'HF' else cpu_ssc.RKS
    for fcsd in (False, True):
        g = SSC(mf)
        g.with_fcsd = fcsd
        g.dso = 'exact'
        c = ref(mf.to_cpu())
        c.verbose = 0
        c.with_fcsd = fcsd
        if pairs:
            g.nuc_pair = c.nuc_pair = pairs
        e = g.kernel()
        ec = c.kernel()
        err = abs(e - ec).max() / abs(ec).max()
        report(f'{label} {"FC+SD" if fcsd else "FC"} (exact DSO)',
               err < 1e-5, f'rel {err:.1e}')
    g.dso = 'grid'
    e = g.kernel()
    err = abs(e - ec).max() / abs(ec).max()
    report(f'{label} FC+SD (grid DSO)', err < 1e-4, f'rel {err:.1e}')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
