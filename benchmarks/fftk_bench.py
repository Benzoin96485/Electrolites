"""FFTDF k-point exchange: GPU4PySCF's get_k_kpts against fastfftk's (J3).

One converged KRKS/PBE density (tagged with its orbitals, as the SCF hands
it over), then ``fft_jk.get_k_kpts`` timed for GPU4PySCF, for ``fastfftk``
and the the largest
element-wise difference of each against GPU4PySCF.

    python benchmarks/fftk_bench.py --cell diamond --kpts 3,3,3
"""
import argparse
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--cell', default='diamond', choices=('diamond', 'diamond8'))
ap.add_argument('--kpts', default='2,2,2')
ap.add_argument('--basis', default='gth-dzvp')
ap.add_argument('--reps', type=int, default=2)
a = ap.parse_args()

import electrolites                                          # noqa: E402
electrolites.patch('fastfftk')
import cupy as cp                                            # noqa: E402
from pyscf.pbc import gto                                    # noqa: E402
from gpu4pyscf.pbc import dft                                # noqa: E402
from gpu4pyscf.pbc.df import fft_jk as F                     # noqa: E402
from electrolites import fastfftk                            # noqa: E402

if a.cell == 'diamond':
    atom = 'C 0 0 0; C 0.8917 0.8917 0.8917'
    lat = np.array([[0, 1.7834, 1.7834], [1.7834, 0, 1.7834],
                    [1.7834, 1.7834, 0]])
else:
    frac = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0), (.25, .25, .25),
            (.25, .75, .75), (.75, .25, .75), (.75, .75, .25)]
    atom = '; '.join(f'C {3.5668*f[0]} {3.5668*f[1]} {3.5668*f[2]}'
                     for f in frac)
    lat = np.eye(3) * 3.5668
cell = gto.M(atom=atom, a=lat, basis=a.basis, pseudo='gth-pade', verbose=0)
kpts = cell.make_kpts([int(x) for x in a.kpts.split(',')])
mf = dft.KRKS(cell, kpts=kpts, xc='pbe')
mf.conv_tol = 1e-9
mf.kernel()
dm = mf.make_rdm1()
print(f'# {a.cell} {a.kpts} {a.basis} nao={cell.nao} nk={len(kpts)} '
      f'mesh={cell.mesh}', flush=True)


def timed(fn):
    out = fn()
    cp.cuda.Stream.null.synchronize()
    best = 1e9
    for _ in range(a.reps):
        t = time.perf_counter()
        out = fn()
        cp.cuda.Stream.null.synchronize()
        best = min(best, time.perf_counter() - t)
    return best, out


args = (mf.with_df, dm, 1, kpts, None, 'ewald')
t_ref, ref = timed(lambda: F._get_k_kpts_orig(*args))
print(f'gpu4pyscf            {t_ref:8.3f} s')
t2, v2 = timed(lambda: F.get_k_kpts(*args))
print(f'time reversal      {t2:8.3f} s  {t_ref/t2:5.2f}x  '
      f'max|dK| {float(abs(v2 - ref).max()):.2e}  '
      f'k1 built {fastfftk.STATS["k1_built"]}/{fastfftk.STATS["k1_total"]}')
