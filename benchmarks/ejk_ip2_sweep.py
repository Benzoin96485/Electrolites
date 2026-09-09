"""Catalogue B3: is `ejk_ip2`'s launch configuration worth sweeping?

`hessian/rhf.py`'s `_ip2_quartets_scheme` and `_ip2_type3_quartets_scheme`
choose the block shape for the second-derivative ERI kernels from the
shared-memory budget alone:

    counts = (SHM_SIZE - ij_prims*8) // (unit*8)
    n = min(THREADS, _nearest_power2(counts));  gout_stride = THREADS // n

so the block is always `THREADS = 256` threads, split as many quartets as
shared memory allows.  Nothing was measured to arrive at that, and the same
question on the exchange side -- `benchmarks/sweep_j_high.py`,
`gen_khigh.py`'s `--gout-max` -- was worth a factor on several classes.  This
is the cheapest possible attempt at B3: change nothing but the two scheme
functions and time `_partial_ejk_ip2`.

Each candidate is `(threads, min_quartets_shift)`: `threads` replaces the
fixed 256, and the shift halves the quartet count that many times, moving
threads from the quartet dimension to `gout_stride` (which is where the
contraction's `nfij*nfkl` loop is spread).  A candidate whose shared memory
does not fit is skipped by the kernel's own guard, so the run reports it as an
error rather than silently mismeasuring.

    python benchmarks/ejk_ip2_sweep.py --xyz molecules/penicillinV.xyz \
        --basis def2-svp
"""
import json
import time

import cupy as cp
import numpy as np

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--threads', default='128,256,512',
                help='comma separated block sizes to try')
ap.add_argument('--shift', default='0,1,2',
                help='comma separated halvings of the quartet count')
a = ap.parse_args()
apply_patch(a.patch)

from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.hessian import rhf as hrhf                    # noqa: E402
from gpu4pyscf.scf import jk as JK                           # noqa: E402

_sync = cp.cuda.Stream.null.synchronize
_orig12 = hrhf._ip2_quartets_scheme
_orig3 = hrhf._ip2_type3_quartets_scheme


def _patched(orig, threads, shift):
    def scheme(mol, l_ctr_pattern, shm_size=JK.SHM_SIZE):
        n, _gs = orig(mol, l_ctr_pattern, shm_size)
        # `orig` returns the largest power of two that fits at THREADS = 256;
        # the shared-memory bound scales with the quartet count alone, so a
        # different block size keeps the same n and only changes gout_stride
        n = max(1, n >> shift)
        return n, max(1, threads // n)
    return scheme


mol = build_mol(a)
tag = f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}/{a.patch or "stock"}'
print(f'# {tag}  natm={mol.natm} nao={mol.nao}', flush=True)

mf = dft.RKS(mol, xc=a.xc)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
e = mf.kernel()
print(f'# SCF {e:.9f} converged={mf.converged}', flush=True)

opt = JK._VHFOpt(mol, mf.direct_scf_tol).build()
dm0 = cp.asarray(mf.make_rdm1())


def run(reps):
    out = None
    ts = []
    # One untimed call first.  The first `_partial_ejk_ip2` of a process pays
    # cudaFuncSetAttribute, the module load and the first touch of `dd_pool`,
    # and with only two or three reps a `min` does not remove it -- an
    # unwarmed baseline reads about 12 % slow on this stage.
    hrhf._partial_ejk_ip2(mol, dm0, opt, 1.0, 1.0)
    _sync()
    for _ in range(reps):
        _sync()
        t = time.perf_counter()
        out = hrhf._partial_ejk_ip2(mol, dm0, opt, 1.0, 1.0)
        _sync()
        ts.append(time.perf_counter() - t)
    return min(ts), out


base, ejk0 = run(a.reps)
ref = float(cp.asarray(ejk0).sum())
print(f'\n# stock scheme: {base:.3f} s   sum(ejk) = {ref:.12e}')
print(f'\n{"threads":>8s} {"shift":>6s} {"time/s":>9s} {"ratio":>7s} '
      f'{"max|d ejk|":>12s}')
print('-' * 48)
rows = [dict(threads=256, shift=0, t=base, ratio=1.0, err=0.0, stock=True)]
for threads in [int(x) for x in a.threads.split(',')]:
    for shift in [int(x) for x in a.shift.split(',')]:
        if threads == 256 and shift == 0:
            continue
        hrhf._ip2_quartets_scheme = _patched(_orig12, threads, shift)
        hrhf._ip2_type3_quartets_scheme = _patched(_orig3, threads, shift)
        try:
            t, ejk = run(a.reps)
            err = float(cp.abs(cp.asarray(ejk) - cp.asarray(ejk0)).max())
        except Exception as exc:                              # noqa: BLE001
            print(f'{threads:8d} {shift:6d} {"ERR":>9s}         '
                  f'{type(exc).__name__}: {str(exc)[:40]}', flush=True)
            continue
        print(f'{threads:8d} {shift:6d} {t:9.3f} {base/t:6.2f}x {err:12.2e}',
              flush=True)
        rows.append(dict(threads=threads, shift=shift, t=t, ratio=base / t,
                         err=err, stock=False))
hrhf._ip2_quartets_scheme = _orig12
hrhf._ip2_type3_quartets_scheme = _orig3

best = max(rows, key=lambda r: r['ratio'])
print(f'\nbest: threads={best["threads"]} shift={best["shift"]}  '
      f'{best["ratio"]:.2f}x over the stock scheme')
print('\n' + json.dumps({'tag': tag, 'natm': mol.natm, 'nao': int(mol.nao),
                         'stock': base, 'rows': rows}))
