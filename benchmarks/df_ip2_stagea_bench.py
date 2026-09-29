"""Stage A of the ip2 call, piece by piece: the symmetric fill and dense
``C^T J C`` against item 30's lower triangle (W39).

Times, per auxiliary block of one integral batch at the call's own sizes and
with CUDA events, the fill (``fill_symmetric`` / ``_LowTri.scatter``), the
first product (``contract('pqr,pi->iqr')`` / ``_LowTri.lt_c``), the second
(``contract('iqr,qj->rij')``) and item 30's ``G + G^T``; no SCF, a random
orthonormal ``C``.

    python benchmarks/df_ip2_stagea_bench.py --xyz benchmarks/molecules/taxol.xyz
"""
import argparse

import numpy as np
import cupy as cp

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--xyz', required=True)
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--nocc', type=int, default=226)
ap.add_argument('--blk', type=int, default=0, help='auxiliary block (0: the call\'s)')
ap.add_argument('--tiles', default='32,64,128,256')
ap.add_argument('--reps', type=int, default=5)
a = ap.parse_args()

from pyscf import gto                                        # noqa: E402
from pyscf import lib                                        # noqa: E402
from gpu4pyscf.df import df as DF                            # noqa: E402
from gpu4pyscf.lib.cupy_helper import (                      # noqa: E402
    contract, ndarray, fill_symmetric)
from electrolites import fastdfhessip as H                   # noqa: E402

mol = gto.M(atom=a.xyz, basis=a.basis, verbose=0)
wd = DF.DF(mol)
wd.build()
opt = wd.intopt
nao = mol.nao_cart()
pair = cp.asarray(opt.pair_and_diag_indices(cart=True, original_ao_order=True)[0],
                  dtype=np.int32)
npair = len(pair)
batch = 408
ev, aux_sorting, _, off = opt.int3c2e_evaluator(aux_batch_size=batch,
                                                 reorder_aux=True, cart=True)
buf = cp.empty((batch, npair))
compressed = ev(aux_batch_id=int(np.argmax(np.diff(off))), out=buf)
nb = compressed.shape[1]
blk = a.blk or nb
nocc = a.nocc
c = cp.linalg.qr(cp.random.default_rng(0).standard_normal((nao, nocc)))[0]
c = cp.ascontiguousarray(c)
print(f'# nao {nao}  pairs {npair}  block {blk}  nocc {nocc}', flush=True)


def t(fn, reps=a.reps):
    fn()
    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    e1.synchronize()
    return cp.cuda.get_elapsed_time(e0, e1) / reps


full = cp.zeros((nao, nao, blk))
tmp = cp.empty((nocc, nao, blk))
out = cp.empty((blk, nocc, nocc))
g = cp.empty((blk, nocc, nocc))
j3c = full[:, :, :blk]
r = {}
r['fill_symmetric'] = t(lambda: fill_symmetric(compressed, pair, nao, 0, blk,
                                                out=j3c))
r['pqr,pi->iqr'] = t(lambda: contract('pqr,pi->iqr', j3c, c, out=tmp))
r['iqr,qj->rij'] = t(lambda: contract('iqr,qj->rij', tmp, c, out=out))
ref = out.copy()
fl = 2. * nao * nao * nocc * blk
print(f'  dense: fill {r["fill_symmetric"]:.2f} ms  C^T J {r["pqr,pi->iqr"]:.2f} ms '
      f'({fl / r["pqr,pi->iqr"] * 1e-9:.1f} TF/s)  (.)C {r["iqr,qj->rij"]:.2f} ms',
      flush=True)
for T in [int(x) for x in a.tiles.split(',')]:
    import os
    os.environ['FASTDFHESSIP_LOW_TILE'] = str(T)
    low = H._LowTri(pair, nao)
    full[:] = 0.
    rs = {}
    rs['scatter'] = t(lambda: low.scatter(compressed, 0, blk, j3c))
    rs['lt_c'] = t(lambda: low.lt_c(c, full, blk, tmp))
    rs['second'] = t(lambda: contract('iqr,qj->rij', tmp[:, :, :blk], c, out=g))
    rs['g+gT'] = t(lambda: cp.add(g, g.transpose(0, 2, 1), out=out))
    err = float(abs(out - ref).max() / abs(ref).max())
    fll = sum(2. * nocc * (q1 - q0) * blk * (nao - q0) for q0, q1 in low.cols)
    print(f'  T={T:4d}: scatter {rs["scatter"]:.2f} ms  lt_c {rs["lt_c"]:.2f} ms '
          f'({fll / rs["lt_c"] * 1e-9:.1f} TF/s, {100 * fll / fl:.0f} % of the '
          f'flops)  (.)C {rs["second"]:.2f}  G+G^T {rs["g+gT"]:.2f}  '
          f'total {sum(rs.values()):.2f} ms against {sum(r.values()):.2f}  '
          f'rel {err:.1e}', flush=True)
