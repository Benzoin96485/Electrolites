"""IEF-PCM: the two int3c1e builds per SCF cycle, and the SCF around them.

    python benchmarks/pcm_bench.py --xyz m.xyz --patch ''       > a.json
    python benchmarks/pcm_bench.py --xyz m.xyz --patch fast1e   > b.json

Separate processes, as everywhere else here: `fast1e` patches
`gpu4pyscf.gto.int3c1e` globally.

Two numbers come out.  The **per-call** timings are what the kernels do, taken
on a fixed cavity, a fixed density and fixed surface charges, so they are a
clean A/B of the integral code.  The **SCF wall clock** is what the application
gets, and on a small molecule it will not resolve: those two calls are only a
few per cent of an SCF there, so a large kernel speedup can vanish into the
run-to-run spread.  Both are reported because the difference between them is
the point (see "Results" in docs/README_fast1e.md).

`--cycles` runs the SCF first and takes the density from it: a `minao` guess is
exactly zero on the polarisation shells, but unlike the J/K builds the int3c1e
kernels do no density screening at all, so here it changes the timing only
through what the SCF's own convergence does, not through the kernel.
"""
import json
import sys
import time

sys.path.insert(0, __file__.rsplit('/', 2)[0])
from benchmarks._common import parser, apply_patch      # noqa: E402

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--lebedev', type=int, default=29,
                help="cavity point density; 29 is GPU4PySCF's default")
ap.add_argument('--method', default='IEF-PCM')
ap.add_argument('--no-scf', action='store_true',
                help='per-call timings only, no full SCF')
a = ap.parse_args()
apply_patch(a.patch)

import numpy as np                                       # noqa: E402
import cupy as cp                                        # noqa: E402
import pyscf                                             # noqa: E402
from gpu4pyscf import dft                                # noqa: E402
from gpu4pyscf.gto import int3c1e as I3                  # noqa: E402


def timeit(fn, reps):
    fn()
    cp.cuda.runtime.deviceSynchronize()
    ts = []
    for _ in range(reps):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        fn()
        cp.cuda.runtime.deviceSynchronize()
        ts.append(time.perf_counter() - t0)
    ts = np.asarray(ts)
    return {'min_ms': float(ts.min() * 1e3), 'mean_ms': float(ts.mean() * 1e3),
            'cv': float(ts.std() / ts.mean())}


def make_mf(mol):
    mf = dft.RKS(mol, xc=a.xc)
    mf.grids.level = a.grids
    mf.conv_tol = a.conv_tol
    mf.verbose = 0
    mf = mf.PCM()
    mf.with_solvent.method = a.method
    mf.with_solvent.lebedev_order = a.lebedev
    return mf


mol = pyscf.M(atom=a.xyz, basis=a.basis, charge=a.charge, spin=a.spin,
              verbose=0, max_memory=80000)
mf = make_mf(mol)
pcm = mf.with_solvent
pcm.build()
grids = pcm.surface['grid_coords']
ce2 = pcm.surface['charge_exp'] ** 2
intopt = pcm.intopt

dm = cp.asarray(mf.get_init_guess(mol, 'minao'))
if dm.ndim == 3:
    dm = dm[0]
if a.cycles:
    scratch = make_mf(pyscf.M(atom=a.xyz, basis=a.basis, charge=a.charge,
                              spin=a.spin, verbose=0, max_memory=80000))
    scratch.max_cycle = a.cycles
    scratch.kernel()
    dm = cp.asarray(scratch.make_rdm1())
q = cp.asarray(np.random.RandomState(0).rand(grids.shape[0]) * 1e-3)

out = {'patched': list(sys.modules['electrolites'].patched()),
       'system': a.xyz.rsplit('/', 1)[-1], 'natm': mol.natm, 'nao': int(mol.nao),
       'basis': a.basis, 'xc': a.xc, 'lebedev': a.lebedev,
       'ncavity': int(grids.shape[0]),
       'nshellpair': int(sum(len(x) for x in intopt.log_qs))}

out['density_contracted'] = timeit(
    lambda: I3.get_int3c1e_density_contracted(mol, grids, ce2, dm, intopt),
    a.reps)
out['charge_contracted'] = timeit(
    lambda: I3.get_int3c1e_charge_contracted(mol, grids, ce2, q, intopt),
    a.reps)
out['per_scf_cycle_ms'] = (out['density_contracted']['mean_ms']
                           + out['charge_contracted']['mean_ms'])
print(f"# {out['system']} natm={mol.natm} nao={mol.nao} "
      f"cavity={out['ncavity']} pairs={out['nshellpair']}", file=sys.stderr)
for k in ('density_contracted', 'charge_contracted'):
    print(f"#   {k:20s} {out[k]['mean_ms']:9.3f} ms  (cv {out[k]['cv']:.3f})",
          file=sys.stderr)

if not a.no_scf:
    ts, es = [], []
    make_mf(mol).kernel()                      # warm-up, not timed
    for _ in range(max(1, a.reps - 1)):
        m = make_mf(mol)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        e = m.kernel()
        cp.cuda.runtime.deviceSynchronize()
        ts.append(time.perf_counter() - t0)
        es.append(float(e))
    out['scf'] = {'wall_s_min': float(np.min(ts)), 'wall_s_mean': float(np.mean(ts)),
                  'cv': float(np.std(ts) / np.mean(ts)), 'e_tot': es,
                  'e_spread': float(np.ptp(es)), 'converged': bool(m.converged)}
    print(f"#   SCF {np.mean(ts):.3f} s  E = {es[-1]:.10f}", file=sys.stderr)

print(json.dumps(out, indent=1))
