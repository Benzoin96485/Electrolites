"""GDF k-point cderi on the device (J1).

``electrolites/fastpbcdf.py`` keeps the k-point cderi on the device when it
fits.  Checked in one process against the stock mapped-memory build (the
patched functions swapped back for the reference): the cderi itself, J and K
for a k-point density on the 2-atom diamond cell at 2x2x2 and 3x3x3 (bar
1e-12), that the device path was taken, and that a forced host build
(``FRACTION = 0``) still works through the patched loop.

    python tests/test_pbcdf.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np
import cupy as cp
import electrolites
electrolites.patch_all()
from electrolites import fastpbcdf
from pyscf.pbc import gto
from gpu4pyscf.pbc.df import df as gdf, df_jk, rsdf_builder

cell = gto.M(atom='C 0 0 0; C 0.8917 0.8917 0.8917',
             a=[[0, 1.7834, 1.7834], [1.7834, 0, 1.7834], [1.7834, 1.7834, 0]],
             basis='gth-dzvp', pseudo='gth-pade', verbose=0)


def jk(kpts, dm):
    d = gdf.GDF(cell, kpts)
    d.build()
    vj = df_jk.get_j_kpts(d, dm, 1, kpts)
    vk = df_jk.get_k_kpts(d, dm, 1, kpts)
    cd = {k: cp.asnumpy(cp.asarray(v)) for k, v in d._cderi.items()}
    return cd, cp.asnumpy(vj), cp.asnumpy(vk), d


def stock(fn):
    new_b, new_l = rsdf_builder.compressed_cderi_kk, gdf.GDF.loop
    rsdf_builder.compressed_cderi_kk = rsdf_builder._compressed_cderi_kk_orig
    gdf.GDF.loop = gdf.GDF._loop_orig
    try:
        return fn()
    finally:
        rsdf_builder.compressed_cderi_kk, gdf.GDF.loop = new_b, new_l


ok = True
for mesh in ([2, 2, 2], [3, 3, 3]):
    kpts = cell.make_kpts(mesh)
    nao = cell.nao
    rng = np.random.RandomState(1)
    c = rng.rand(len(kpts), nao, 4) + 1j * rng.rand(len(kpts), nao, 4)
    dm = cp.asarray(np.einsum('kpi,kqi->kpq', c, c.conj()) * .1)
    n0 = fastpbcdf.STATS['device_builds']
    cd1, vj1, vk1, d1 = jk(kpts, dm)
    on_dev = isinstance(next(iter(d1._cderi.values())), cp.ndarray)
    cd0, vj0, vk0, _ = stock(lambda: jk(kpts, dm))
    dc = max(abs(cd1[k] - cd0[k]).max() for k in cd0)
    dj, dk = abs(vj1 - vj0).max(), abs(vk1 - vk0).max()
    good = (dc < 1e-12 and dj < 1e-12 and dk < 1e-12 and on_dev
            and fastpbcdf.STATS['device_builds'] == n0 + 1)
    ok &= good
    print(f'diamond {mesh}: device cderi {on_dev}, max|d cderi| {dc:.1e} '
          f'|dJ| {dj:.1e} |dK| {dk:.1e}  {"ok" if good else "FAIL"}')

fastpbcdf.FRACTION = 0.0
kpts = cell.make_kpts([2, 2, 2])
dm = cp.asarray(np.einsum('kpi,kqi->kpq', c[:8], c[:8].conj()) * .1)
cd2, vj2, vk2, d2 = jk(kpts, dm)
cd0, vj0, vk0, _ = stock(lambda: jk(kpts, dm))
host = not isinstance(next(iter(d2._cderi.values())), cp.ndarray)
dk = abs(vk2 - vk0).max()
good = host and dk < 1e-12
ok &= good
print(f'forced host build: host cderi {host}, |dK| {dk:.1e}  {"ok" if good else "FAIL"}')
print('ALL OK' if ok else 'FAILED')
sys.exit(0 if ok else 1)
