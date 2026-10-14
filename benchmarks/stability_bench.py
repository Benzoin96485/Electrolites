"""GPU stability analysis (`electrolites/stability.py`) timed, after the GPU
SCF, on the systems of FEATURE_TARGETS §11.4 (whose CPU times are there).

    python benchmarks/stability_bench.py --xyz benchmarks/molecules/tmc_ni_dithiolene.xyz \
        --charge -1 --spin 1 --basis def2-svp --xc b3lyp
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parser, build_mol                          # noqa: E402

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--max-cycle', type=int, default=150)
ap.add_argument('--no-external', action='store_true')
a = ap.parse_args()

import cupy as cp                                              # noqa: E402
import electrolites                                            # noqa: E402
electrolites.patch_all()
electrolites.enable('stability')
from electrolites import stability                             # noqa: E402
from gpu4pyscf import dft                                      # noqa: E402

mol = build_mol(a)
mf = (dft.RKS if mol.spin == 0 else dft.UKS)(mol, xc=a.xc)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.max_cycle = a.max_cycle
t0 = time.perf_counter()
mf.kernel()
cp.cuda.Stream.null.synchronize()
t1 = time.perf_counter()
times = {}
ti = time.perf_counter()
_, _, si, se = mf.stability(internal=True, external=False, return_status=True)
cp.cuda.Stream.null.synchronize()
times['internal'] = time.perf_counter() - ti
if not a.no_external:
    ti = time.perf_counter()
    _, _, _, se = mf.stability(internal=False, external=True, return_status=True)
    cp.cuda.Stream.null.synchronize()
    times['external'] = time.perf_counter() - ti
rec = dict(xyz=a.xyz, basis=a.basis, xc=a.xc, natm=mol.natm, nao=mol.nao,
           scf_s=t1 - t0, scf_cycles=mf.cycles, converged=bool(mf.converged),
           e_scf=float(mf.e_tot), stability_s=times,
           stable=[bool(si), None if se is None else bool(se)],
           eigs={k: [float(x) for x in v] for k, v in stability.LAST.items()},
           gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
print(f"[{os.path.basename(a.xyz)}/{a.basis}] nao={mol.nao} SCF {rec['scf_s']:.1f} s "
      f"({mf.cycles} cyc)  stability {times}  stable {rec['stable']}")
print('RESULT ' + json.dumps(rec))
