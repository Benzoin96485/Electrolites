"""A range-separated DF run's two tensors: their size, their builds, the host.

``docs/DF_WORK_LIST.md`` **W22**.  Drives one density-fitted range-separated
calculation -- the two builds on a guess density, or a whole SCF -- and prints,
for every ``DF.build`` it makes:

* which operator it is for and the tensor it left: shape, precision, bytes,
  host or device;
* its wall time, and inside it the metric decomposition (``int2c2e``, the
  Cholesky attempt, the eigen-decomposition when the Cholesky fails);
* the process's resident set and the node's ``MemAvailable`` after it.

The build is timed by wrapping the ``DF`` *class*, never the instance --
``docs/DF_WORK_LIST.md`` §5.4 says what an instance wrapper does to
``DF.range_coulomb``'s copy.

    python benchmarks/df_rsh_bench.py --xyz benchmarks/molecules/PfPMT.xyz \
        --charge -1 --basis def2-svp --xc hyb_gga_xc_wb97x_d3 [--patch all]
    python benchmarks/df_rsh_bench.py --xyz .../taxol.xyz --basis def2-svp \
        --xc hyb_gga_xc_wb97x_d3 --scf [--patch all]

Run one variant per process (§5.10) and each pair twice (§5.3).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import _common                                              # noqa: E402

ap = _common.parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--scf', action='store_true',
                help='run the SCF to convergence instead of one get_veff on '
                     'the guess')
ap.add_argument('--max-cycle', type=int, default=50)
ap.add_argument('--cpu-cderi', action='store_true',
                help='put both tensors on the host whatever their size')
ap.add_argument('--verbose', type=int, default=0,
                help="the molecule's verbosity; 3 prints the build's NOTE")
ap.add_argument('--no-df', action='store_true',
                help='the exact four-centre SCF instead: the reference both '
                     'fits are an approximation to')
ap.add_argument('--veff-reps', type=int, default=0,
                help='after the first get_veff (which builds), time this many '
                     'more on the same density: the per-cycle cost')
ap.add_argument('--json', default='')
a = ap.parse_args()

patched = _common.apply_patch(a.patch)

import numpy as np                                          # noqa: E402
import cupy as cp                                           # noqa: E402
from gpu4pyscf.df import df as DFMOD                        # noqa: E402
from gpu4pyscf.lib import cupy_helper                       # noqa: E402


def _status(key):
    with open('/proc/self/status') as fh:
        for line in fh:
            if line.startswith(key):
                return int(line.split()[1]) * 1024
    return 0


def _meminfo(key):
    with open('/proc/meminfo') as fh:
        for line in fh:
            if line.startswith(key):
                return int(line.split()[1]) * 1024
    return 0


GB = 1e-9
TIMES = {}


def _timed(name, fn):
    def wrapper(*args, **kw):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        try:
            return fn(*args, **kw)
        finally:
            cp.cuda.Device().synchronize()
            TIMES[name] = TIMES.get(name, 0.) + time.perf_counter() - t0
    return wrapper


# the metric decomposition's parts, on whichever module is doing the build
cp.linalg.eigh = _timed('eigh', cp.linalg.eigh)
cupy_helper.cholesky = _timed('potrf', cupy_helper.cholesky)
DFMOD.cholesky = cupy_helper.cholesky
DFMOD._decompose_j2c = _timed('decompose_j2c', DFMOD._decompose_j2c)
if 'fastdfcderi' in patched:
    from electrolites import fastdfcderi as FC
    FC.cholesky = cupy_helper.cholesky
    FC._decompose_j2c = _timed('decompose_j2c', FC._decompose_j2c)

# every two-electron request of the SCF goes through this: J, the
# full-range and long-range exchange, and the fused object's (a subclass)
DFMOD.DF.get_jk = _timed('df_get_jk', DFMOD.DF.get_jk)

BUILDS = []
_LABEL = [None]
_orig_build = DFMOD.DF.build


def _build(self, *args, **kw):
    omega = kw.get('omega', args[1] if len(args) > 1 else None)
    before = dict(TIMES)
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    out = _orig_build(self, *args, **kw)
    cp.cuda.Device().synchronize()
    dt = time.perf_counter() - t0
    rec = {'omega': _LABEL[0] or omega or 0.0, 'wall_s': dt}
    for k, v in TIMES.items():
        rec[k + '_s'] = v - before.get(k, 0.)
    cderi = self._cderi
    if cderi is not None:
        t = cderi[0]
        rec.update(shape=list(t.shape), dtype=str(t.dtype),
                   gb=t.nbytes * GB,
                   where='device' if isinstance(t, cp.ndarray) else 'host')
    rec['rss_gb'] = _status('VmRSS:') * GB
    rec['avail_gb'] = _meminfo('MemAvailable:') * GB
    BUILDS.append(rec)
    shape = 'x'.join(map(str, rec.get('shape', [])))
    extra = '  '.join(f'{k[:-2]} {rec[k]:.2f} s' for k in
                      ('decompose_j2c_s', 'potrf_s', 'eigh_s') if k in rec)
    print(f'@@BUILD omega={str(rec["omega"]):<5} {dt:8.2f} s   {shape} '
          f'{rec.get("dtype", "")} {rec.get("gb", 0):7.2f} GB '
          f'{rec.get("where", "-")}   [{extra}]   RSS {rec["rss_gb"]:.1f} GB  '
          f'MemAvailable {rec["avail_gb"]:.1f} GB', flush=True)
    return out


DFMOD.DF.build = _build
if 'fastdfrsh' in patched:
    # the fused tensor's object overrides `build`, so it is wrapped too
    from electrolites import fastdfrsh as _R
    _orig_fused_build = _R._FusedDF.build

    def _fused_build(self, *args, **kw):
        global _orig_build
        saved, _orig_build = _orig_build, _orig_fused_build
        _LABEL[0] = 'fused'
        try:
            return _build(self, *args, **kw)
        finally:
            _orig_build = saved
            _LABEL[0] = None
    _R._FusedDF.build = _fused_build

mol = _common.build_mol(a)
mol.verbose = a.verbose
if mol.spin:
    from gpu4pyscf.dft import uks as KS
    mf = KS.UKS(mol, xc=a.xc)
else:
    from gpu4pyscf.dft import rks as KS
    mf = KS.RKS(mol, xc=a.xc)
if not a.no_df:
    mf = mf.density_fit(auxbasis=a.auxbasis)
if a.cpu_cderi:
    mf.with_df.use_gpu_memory = False
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.max_cycle = a.max_cycle
omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
print(f'# {os.path.basename(a.xyz)} {a.basis} {a.xc} natm={mol.natm} '
      f'nao={mol.nao} omega={omega} alpha={alpha} hyb={hyb}  '
      f'MemTotal {_meminfo("MemTotal:") * GB:.1f} GB  '
      f'MemAvailable {_meminfo("MemAvailable:") * GB:.1f} GB', flush=True)

t0 = time.perf_counter()
if a.scf:
    e = mf.kernel()
    cycles = getattr(mf, 'cycles', None)
    print(f'@@SCF E={e:.10f} converged={mf.converged} cycles={cycles} '
          f'{time.perf_counter() - t0:.2f} s', flush=True)
else:
    dm = mf.get_init_guess()
    v = mf.get_veff(mol, dm)
    e = float(v.exc)
    print(f'@@VEFF exc={e:.10f} {time.perf_counter() - t0:.2f} s', flush=True)
    ts = []
    jk0 = TIMES.get('df_get_jk', 0.)
    for _ in range(a.veff_reps):
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        try:
            v = mf.get_veff(mol, dm)
        except Exception:
            import traceback
            traceback.print_exc()
            if 'fastdf' in patched:
                from electrolites import fastdf
                print('@@FASTDF', json.dumps(fastdf.stats(), default=str))
            print(f'@@MEM device free {cp.cuda.Device().mem_info[0] * GB:.2f}'
                  f' GB, pool used {cp.get_default_memory_pool().used_bytes() * GB:.2f}'
                  f' GB, pool held {cp.get_default_memory_pool().total_bytes() * GB:.2f} GB',
                  flush=True)
            raise
        cp.cuda.Device().synchronize()
        ts.append(time.perf_counter() - t1)
    if ts:
        jk = (TIMES.get('df_get_jk', 0.) - jk0) / len(ts)
        print(f'@@CYCLE get_veff {min(ts):.3f} s, of which DF.get_jk '
              f'{jk:.3f} s (mean) (min of {len(ts)}; '
              f'{" ".join(f"{t:.3f}" for t in ts)})  exc={float(v.exc):.10f}',
              flush=True)
live = [r['gb'] for r in BUILDS if r.get('where') == 'host']
print(f'@@HOST tensors {" + ".join(f"{g:.2f}" for g in live) or "0"} = '
      f'{sum(live):.2f} GB   peak RSS {_status("VmHWM:") * GB:.1f} GB',
      flush=True)
if 'fastdfcderi' in patched:
    print('@@STATS', json.dumps(FC.stats(), default=str), flush=True)
if 'fastdf' in patched:
    from electrolites import fastdf
    st = fastdf.stats()
    print('@@FASTDF', json.dumps({k: st[k] for k in (
        'resident_dropped', 'tiles_dropped', 'resident_rows', 'blksize')}))
if 'fastdfrsh' in patched:
    from electrolites import fastdfrsh
    print('@@RSH', json.dumps(fastdfrsh.stats()), flush=True)
if a.json:
    with open(a.json, 'a') as fh:
        fh.write(json.dumps({'case': os.path.basename(a.xyz), 'basis': a.basis,
                             'xc': a.xc, 'patch': a.patch, 'scf': a.scf,
                             'e': e, 'builds': BUILDS}) + '\n')
