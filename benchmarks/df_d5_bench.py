"""D5's per-atom contractions at taxol's sizes, alone (W37).

Random operands of the shapes item 8's D5 sees on taxol/def2-SVP (``nsph``
5598, ``ncart`` 6598, ``nao`` 1185, ``nocc`` 226, one 14-AO atom, a group of
370 AOs), each contraction timed on its own: the per-atom loop's five
cuTENSOR contractions (``--old``) and item 26's pieces -- the two
strided-batched GEMMs for a group, the two reduction kernels, and the
``nsph x 3 nsph`` GEMM on strided and on contiguous operands.  No
chemistry; about 50 GB of device memory.

    python benchmarks/df_d5_bench.py [--old]
"""
import argparse
import time

import numpy as np
import cupy as cp

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--old', action='store_true')
a = ap.parse_args()

from gpu4pyscf.lib.cupy_helper import contract              # noqa: E402
from electrolites import fastdfhessip as H                   # noqa: E402

nsph, ncart, nao, nocc, nG, nA = 5598, 6598, 1185, 226, 370, 14
sync = cp.cuda.Stream.null.synchronize


def T(label, f, flops=0, n=3):
    f()
    sync()
    t = time.perf_counter()
    for _ in range(n):
        f()
    sync()
    dt = (time.perf_counter() - t) / n
    rate = f'{flops / dt / 1e12:6.2f} TF/s' if flops else ''
    print(f'{label:36s} {dt * 1e3:8.2f} ms {rate}', flush=True)


z = cp.random.rand(3, nsph, 1, nao, nocc)
t1s = cp.random.rand(3, nsph, 1, nocc, nocc)
dm = cp.random.rand(ncart, 1, nocc, nocc)
C = cp.random.rand(1, nao, nocc)
j10s = cp.random.rand(3, nsph, nsph)
h = cp.zeros((113, nsph, 3, 3))

if a.old:
    z4, t4, d3, c2 = z[:, :, 0], t1s[:, :, 0], dm[:, 0], C[0]
    t1 = cp.empty((3, nsph, nA, nocc))
    T("t1 'ysji,ui->ysuj'", lambda: contract(
        'ysji,ui->ysuj', t4, c2[:nA], out=t1), 2*3*nsph*nocc*nocc*nA)
    T("'xsuj,ysuj->sxy'", lambda: contract(
        'xsuj,ysuj->sxy', z4[:, :, :nA], t1, -1, 1, h[0]))
    pa = cp.empty((ncart, nA, nocc))
    T("pa 'sji,ui->suj'", lambda: contract(
        'sji,ui->suj', d3, c2[:nA], out=pa), 2*ncart*nocc*nocc*nA)
    m = cp.empty((3, nsph, nsph))
    T("m 'suj,xtuj->xst'", lambda: contract(
        'suj,xtuj->xst', pa[:nsph], z4[:, :, :nA], out=m),
      2*3*nsph*nsph*nA*nocc)
    T("'xst,yst->sxy'", lambda: contract(
        'xst,yst->sxy', m, j10s, 1, 1, h[0]))

t1 = cp.empty((3, nsph, 1, nG, nocc))


def f_t1():
    for y in range(3):
        H._batched_occ(t1s, C[0, :nG], t1, nsph, nocc*nocc,
                       y*nsph*nocc*nocc, nG*nocc, y*nsph*nG*nocc, nocc, nG)


T('item 26: t1, batched, one group', f_t1, 2*3*nsph*nocc*nocc*nG)
na = nG // nA
au0 = cp.asarray(np.arange(na) * nA, dtype=np.int32)
anu = cp.full(na, nA, dtype=np.int32)
aid = cp.arange(na, dtype=np.int32)
T(f'item 26: d5_h1, {na} atoms', lambda: H._D5_H1(
    (na, nsph), (256,), (z, t1, h, au0, anu, aid, np.int32(nsph),
                         np.int32(1), np.int32(nao), np.int32(nocc),
                         np.int32(nG), np.int32(0), np.float64(1.))))
pa = cp.empty((ncart, 1, nG, nocc))
T('item 26: P, batched, one group', lambda: H._batched_occ(
    dm, C[0, :nG], pa, ncart, nocc*nocc, 0, nG*nocc, 0, nocc, nG),
  2*ncart*nocc*nocc*nG)
m = cp.empty((nsph, 3*nsph))
k = nA * nocc
T('m GEMM, operands strided', lambda: H._gemm_rm(
    'N', 'T', nsph, 3*nsph, k, 1., pa.data.ptr, nG*nocc, z.data.ptr,
    nao*nocc, 0., m.data.ptr, 3*nsph), 2*3*nsph*nsph*k)
zc = cp.ascontiguousarray(z[:, :, 0, :nA].reshape(3*nsph, k))
pc = cp.ascontiguousarray(pa[:nsph, 0, :nA].reshape(nsph, k))
T('m GEMM, operands contiguous', lambda: H._gemm_rm(
    'N', 'T', nsph, 3*nsph, k, 1., pc.data.ptr, k, zc.data.ptr, k, 0.,
    m.data.ptr, 3*nsph), 2*3*nsph*nsph*k)
T('item 26: d5_h2, one atom', lambda: H._D5_H2(
    (nsph,), (256,), (m, j10s, h[0], np.int32(nsph),
                      np.int64(nsph*nsph), np.float64(1.))))
