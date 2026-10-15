"""Gamma-point periodic TDA timings (FEATURE_TARGETS §15, X1): diamond
n x n x n supercell, GPU (``electrolites.pbc_tdscf``) or PySCF (``--cpu``).

    python benchmarks/pbc_tda_bench.py --n 2 --xc pbe0 [--cpu]
"""
import argparse
import json
import os
import time

import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.tools import super_cell

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--n', type=int, default=2)
ap.add_argument('--xc', default='pbe')
ap.add_argument('--basis', default='gth-dzvp')
ap.add_argument('--nstates', type=int, default=5)
ap.add_argument('--cpu', action='store_true')
ap.add_argument('--mesh', type=int, default=0,
                help='uniform mesh per axis (default: from ke_cutoff)')
a = ap.parse_args()

cell = gto.M(a='''0 1.7834 1.7834
1.7834 0 1.7834
1.7834 1.7834 0''', atom='C 0 0 0; C 0.8917 0.8917 0.8917',
             basis=a.basis, pseudo='gth-pade', verbose=0)
prim = cell
cell = super_cell(cell, [a.n] * 3)
cell.verbose = 0
if a.mesh:
    cell.mesh = [a.mesh] * 3
rec = dict(n=a.n, xc=a.xc, mesh=list(map(int, cell.mesh)), basis=a.basis,
           nao=cell.nao, natm=cell.natm,
           cpu=a.cpu, threads=os.environ.get('OMP_NUM_THREADS'))


def timed(label, fn):
    sync = (lambda: None) if a.cpu else \
        __import__('cupy').cuda.Stream.null.synchronize
    sync()
    t = time.perf_counter()
    out = fn()
    sync()
    rec[label] = time.perf_counter() - t
    return out


if a.cpu:
    from pyscf.pbc import dft, scf, tdscf
else:
    import electrolites
    electrolites.patch_all()
    from gpu4pyscf.pbc import dft, scf
    from electrolites.pbc_tdscf import TDA
if not a.cpu:
    # warm-up on the primitive cell (kernel compilation, plan caches)
    p1 = super_cell(prim, [1] * 3)
    w = scf.RHF(p1) if a.xc.upper() == 'HF' else dft.RKS(p1, xc=a.xc)
    w.run()
    TDA(w).kernel(nstates=a.nstates)
mf = scf.RHF(cell) if a.xc.upper() == 'HF' else dft.RKS(cell, xc=a.xc)
mf.conv_tol = 1e-9
timed('scf_s', mf.kernel)
if a.cpu:
    td = tdscf.TDA(mf)
    td.nstates = a.nstates
else:
    td = TDA(mf)
td.conv_tol = 1e-5
timed('tda_s', lambda: td.kernel(nstates=a.nstates) if not a.cpu
      else td.kernel())
rec['e_ev'] = (np.sort(np.asarray(td.e)) * 27.211386245988).tolist()
print(f"[pbc_tda {'CPU' if a.cpu else 'GPU'} C{cell.natm} {a.basis} {a.xc}] "
      f"nao={cell.nao} scf_s {rec['scf_s']:.2f}  tda_s {rec['tda_s']:.2f}")
print('RESULT ' + json.dumps(rec))
