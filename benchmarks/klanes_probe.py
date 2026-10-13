"""A18: under-filled K classes on side streams ("lanes"), A/B in one process.

    python benchmarks/klanes_probe.py --xyz benchmarks/molecules/PfPMT.xyz \
        --charge -1 --basis 6-31g* [--spin 0] [--cycles 6]

Runs ``--cycles`` SCF cycles (B3LYP) for a realistic density, then times
``mf.get_k`` alternately with ``fastk.LANES = 0`` and ``= 4`` (min of
``--reps``), checks the two K matrices agree, and reports how many class
launches went to lanes.
"""
import time
import cupy as cp
import numpy as np
from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--lanes', default='0,2,4,8')
ap.add_argument('--jk', type=int, default=0,
                help='time get_jk (fastjk) with this many hermi=0 densities instead')
a = ap.parse_args()
apply_patch('all')
from electrolites import fastk                              # noqa: E402
from gpu4pyscf import dft                                   # noqa: E402

mol = build_mol(a)
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc='b3lyp')
mf.verbose = 0
mf.max_cycle = a.cycles or 6
mf.kernel()
dm = cp.asarray(mf.make_rdm1())
if a.jk:
    rng = cp.random.RandomState(0)
    nao_ = mol.nao
    dm = rng.standard_normal((a.jk, nao_, nao_)) * 1e-2
sync = cp.cuda.Device().synchronize
res = {}
counts = {}
orig = fastk._lane_for


def counting(ws, n):
    lane = orig(ws, n)
    counts[fastk.LANES] = counts.get(fastk.LANES, 0) + (lane.stream is not None)
    return lane


fastk._lane_for = counting
lanes = [int(x) for x in a.lanes.split(',')]
for rep in range(a.reps + 1):
    for L in lanes:
        fastk.LANES = L
        sync(); t = time.perf_counter()
        vk = (mf.get_jk(mol, dm, hermi=0)[1] if a.jk
              else mf.get_k(mol, dm, 1))
        sync(); dt = time.perf_counter() - t
        if rep:
            res.setdefault(L, []).append(dt)
        if rep == 1:
            res.setdefault(('vk', L), cp.asnumpy(vk))
name = a.xyz.rsplit('/', 1)[-1]
base = min(res[lanes[0]])
ref = res[('vk', lanes[0])]
for L in lanes:
    t = min(res[L])
    d = abs(res[('vk', L)] - ref).max()
    print(f'{name}/{a.basis} {'jk' + str(a.jk) if a.jk else 'k'} spin={mol.spin} nao={mol.nao} lanes={L}: get_k '
          f'{t:.4f} s ({base/t:.3f}x)  launches on lanes/call '
          f'{counts.get(L, 0)/(a.reps+1):.0f}  max|dK| {d:.1e}', flush=True)
