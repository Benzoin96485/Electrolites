"""D8's triangular multiply: cuBLAS ``dtrmm`` against item 27's blocked form.

``fastdfhessip._trmm`` on a random lower-triangular ``n x n`` factor and an
``(n, m)`` right-hand side, both directions, each block size; the error is
against ``dtrmm`` (W37, ``docs/README_fastdfhessip6.md`` 2.6).

    python benchmarks/df_trmm_bench.py [--n 5598] [--m 60000]
"""
import argparse
import os
import time

import cupy as cp

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--n', type=int, default=5598)
ap.add_argument('--m', type=int, default=60000)
a = ap.parse_args()

from electrolites import fastdfhessip as H                   # noqa: E402

n, m = a.n, a.m


class _Factor:
    pass


sa = _Factor()
sa.linv = cp.ascontiguousarray(cp.tril(cp.random.rand(n, n) / n + cp.eye(n)))
x0 = cp.random.rand(n, m)
for trans in (False, True):
    ref = None
    for mode, blk in (('cublas', 0), ('blocked', 512), ('blocked', 1024),
                      ('blocked', 2048)):
        os.environ['FASTDFHESSIP_TRMM'] = mode
        os.environ['FASTDFHESSIP_TRMM_BLK'] = str(blk or 1024)
        x = x0.copy()
        H._trmm(sa, x, trans)
        if ref is None:
            ref = x.copy()
        err = float(abs(x - ref).max() / abs(ref).max())
        x = x0.copy()
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        H._trmm(sa, x, trans)
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t
        print(f'{"L^-T" if trans else "L^-1"} {mode:8s} b={blk:5d} '
              f'{dt * 1e3:8.1f} ms {n * n * m / dt / 1e12:6.2f} TF/s '
              f'err {err:.1e}', flush=True)
