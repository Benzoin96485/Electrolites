"""The Hessian's second-derivative ERI stage, A/B'd in one process.

Catalogue **B3**.  `fastejk2` replaces the two CUDA entry points
`hessian/rhf.py:_partial_ejk_ip2` drives and leaves everything above them --
the Python driver, the task generation, the screening, the launch schemes --
GPU4PySCF's.  So both implementations can run in one process against one
density and one task list, which is the only fair way to compare them on a
shared node, and `fastejk2.unpatch()` is what makes it possible.

Three things are timed:

* **stock**: GPU4PySCF's kernels and GPU4PySCF's launch scheme.
* **P1+P3**: our kernels (no fp64 division in the primitive or root loops, a
  per-kernel register cap), same scheme.
* **+ scheme**: our kernels with the swept launch scheme, which
  `benchmarks/ejk_ip2_sweep.py` found by halving the quartet count that
  `_ip2_quartets_scheme` derives from the shared-memory budget alone.

    python benchmarks/ejk_ip2_bench.py --xyz molecules/penicillinV.xyz \
        --basis def2-svp
"""
import json
import time

import cupy as cp
import numpy as np

from _common import build_mol, parser

ap = parser(__doc__.splitlines()[0])
a = ap.parse_args()

# both implementations in one process: import the module directly rather than
# through electrolites.patch(), which is one-way by design
import electrolites.fastejk2 as fastejk2                     # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.hessian import rhf as hrhf                    # noqa: E402
from gpu4pyscf.scf import jk as JK                           # noqa: E402

_sync = cp.cuda.Stream.null.synchronize
# fastejk2 patches the two scheme functions at import, so GPU4PySCF's own are
# the ones it stashed -- taking `hrhf._ip2_quartets_scheme` here would capture
# the *swept* version and then sweep it again, which is a shift of two dressed
# up as a shift of one.
_orig12 = getattr(hrhf, '_ejk2_orig__ip2_quartets_scheme',
                  hrhf._ip2_quartets_scheme)
_orig3 = getattr(hrhf, '_ejk2_orig__ip2_type3_quartets_scheme',
                 hrhf._ip2_type3_quartets_scheme)


def _swept(orig, shift=1):
    def scheme(mol, l_ctr_pattern, shm_size=JK.SHM_SIZE):
        n, _gs = orig(mol, l_ctr_pattern, shm_size)
        n = max(1, n >> shift)
        return n, max(1, 256 // n)
    return scheme


def _use(kernels, scheme):
    """Select one of the four corners of the 2x2."""
    if kernels:
        ok = fastejk2.apply_patch(force=True)
    else:
        fastejk2.unpatch()
        ok = True
    hrhf._ip2_quartets_scheme = _swept(_orig12) if scheme else _orig12
    hrhf._ip2_type3_quartets_scheme = _swept(_orig3) if scheme else _orig3
    return ok


mol = build_mol(a)
tag = f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}'
print(f'# {tag}  natm={mol.natm} nao={mol.nao}  '
      f'fastejk2 patched={fastejk2.patched()}', flush=True)

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
    return min(ts), cp.asarray(out)


CASES = []
_use(kernels=False, scheme=False)
CASES.append(('stock', run(a.reps)))
_use(kernels=False, scheme=True)
CASES.append(('scheme only', run(a.reps)))
if _use(kernels=True, scheme=False):
    CASES.append(('kernels only', run(a.reps)))
    _use(kernels=True, scheme=True)
    CASES.append(('kernels + scheme', run(a.reps)))
else:
    print('# fastejk2 kernels unavailable; only the two scheme rows are here')
_use(kernels=False, scheme=False)

ref = CASES[0][1][1]
scale = float(cp.abs(ref).max()) or 1.
print(f'\n{"variant":18s} {"time/s":>9s} {"over stock":>11s} '
      f'{"rel|d ejk|":>11s}')
print('-' * 54)
rows = []
for name, (t, out) in CASES:
    rel = float(cp.abs(out - ref).max()) / scale
    print(f'{name:18s} {t:9.3f} {CASES[0][1][0]/t:10.2f}x {rel:11.2e}',
          flush=True)
    rows.append(dict(variant=name, t=t, ratio=CASES[0][1][0] / t, rel=rel))

print('\n' + json.dumps({'tag': tag, 'natm': mol.natm, 'nao': int(mol.nao),
                         'rows': rows}))
