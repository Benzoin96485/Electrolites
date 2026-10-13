"""J1 probe: the GDF k-point exchange, split into the one-time cderi build and
the per-SCF-cycle K build.

    python benchmarks/gdf_k_probe.py --cell diamond8 --kpts 2,2,2 [--patch all]

Converges a PBE0 KRKS with GDF once, then times (a) a fresh ``GDF.build``
twice (the first includes one-time module loads) and (b) ``get_k_kpts`` on the
converged density, ``--reps`` times, and checks a candidate against stock
where one is patched in.
"""
import argparse, time, json, cProfile, pstats, io
ap = argparse.ArgumentParser()
ap.add_argument('--cell', default='diamond8')
ap.add_argument('--kpts', default='2,2,2')
ap.add_argument('--patch', default='none')
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--prof', action='store_true')
a = ap.parse_args()
import electrolites
if a.patch == 'all':
    electrolites.patch_all()
import numpy as np, cupy as cp
from pyscf.pbc import gto
from gpu4pyscf.pbc import dft as pdft
from gpu4pyscf.pbc.df import df_jk, df as gdf
if a.cell == 'diamond':
    atom = 'C 0 0 0; C 0.8917 0.8917 0.8917'
    lat = [[0, 1.7834, 1.7834], [1.7834, 0, 1.7834], [1.7834, 1.7834, 0]]
else:
    x = 3.5668
    frac = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0),
            (.25, .25, .25), (.25, .75, .75), (.75, .25, .75), (.75, .75, .25)]
    atom = '; '.join(f'C {x*f[0]} {x*f[1]} {x*f[2]}' for f in frac)
    lat = np.eye(3) * x
cell = gto.M(atom=atom, a=lat, basis='gth-dzvp', pseudo='gth-pade', verbose=0, max_memory=4000)
kpts = cell.make_kpts([int(k) for k in a.kpts.split(',')])
mf = pdft.KRKS(cell, kpts=kpts, xc='pbe0').density_fit()
mf.verbose = 0
sync = cp.cuda.Device().synchronize
t = time.perf_counter(); e = mf.kernel(); sync()
print(f'# {a.cell} {a.kpts} nao={cell.nao} nk={len(kpts)} SCF {e:.10f} {time.perf_counter()-t:.2f} s', flush=True)
dm = cp.asarray(mf.make_rdm1())
mydf = mf.with_df
for r in range(2):
    d2 = gdf.GDF(cell, kpts)
    sync(); t = time.perf_counter()
    if a.prof and r == 1:
        pr = cProfile.Profile(); pr.enable()
    d2.build(); sync()
    if a.prof and r == 1:
        pr.disable(); s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(25); print(s.getvalue())
    print(f'build {r}: {time.perf_counter()-t:.3f} s', flush=True)
ts = []
for r in range(a.reps):
    sync(); t = time.perf_counter()
    vk = df_jk.get_k_kpts(mydf, dm, 1, kpts); sync()
    ts.append(time.perf_counter() - t)
print('get_k_kpts', ' '.join(f'{x:.3f}' for x in ts), flush=True)
np.save(f'/global/homes/w/weiliang/vibeqc/_el_jobs/j1/vk_{a.cell}_{a.kpts}_{a.patch}.npy', cp.asnumpy(vk))
from electrolites import fastpbcdf as _F; print('STATS', getattr(_F, 'STATS', None)) if a.patch=='all' else None
print('RESULT', json.dumps(dict(cell=a.cell, kpts=a.kpts, patch=a.patch, get_k=min(ts))))
