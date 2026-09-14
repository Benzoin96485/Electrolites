"""What a first run pays for NVRTC, and what a warmup takes off it.

Nothing is compiled at install time, so the first process on a machine to want
a kernel family pays for it inside its own wall clock -- about 36 s for a
6-31G\\* job and 98 s for a basis that reaches f, which every def2 set on a
transition metal does (`benchmarks/compile_time.py` is the per-family table).
Later processes read CuPy's disk cache and pay none of it.

For a single calculation that is a fact of life.  For a dataset run it is a
question of *who* pays it and *how many times*: one process ahead of the run,
or every rank at once on a cold cache.  `electrolites.warmup` compiles exactly
what a given basis will ask for, in worker processes, so this script measures
the three numbers that decide whether that is worth doing:

    cold        a real job with an empty CuPy cache -- compile inside the run
    warm        the same job with the cache the warmup filled
    warmup      the wall clock of filling it, at one worker and at --jobs

Every column uses its own ``CUPY_CACHE_DIR``, created empty, so a "cold" run is
really cold and nothing leaks between the columns.

    python benchmarks/warm_start.py --xyz benchmarks/molecules/tmc_fecl4.xyz \\
        --charge -1 --spin 5 --basis def2-svp --xc b3lyp --jobs 8
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--xyz', required=True)
ap.add_argument('--charge', type=int, default=0)
ap.add_argument('--spin', type=int, default=0)
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--jobs', type=int, default=8)
ap.add_argument('--modules', default='fastk,fastj,fastejk,nvcc',
                help='what to warm.  The default is what an SCF and a '
                     'gradient reach; "all" adds the fused and response '
                     'families, which this job never calls and which would '
                     'make the warmup look more expensive than it is for it')
ap.add_argument('--keep', action='store_true',
                help='keep the cache directories instead of deleting them')
a = ap.parse_args()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: One SCF cycle and one gradient, which is enough to reach every kernel
#: family this compares -- the point is the compile, not the arithmetic.
JOB = textwrap.dedent('''
    import json, sys, time
    t0 = time.perf_counter()
    import electrolites
    electrolites.patch_all()
    from pyscf import gto
    from gpu4pyscf import dft
    xyz, charge, spin, basis, xc = sys.argv[1:6]
    kw = dict(atom=xyz, basis=basis, charge=int(charge), spin=int(spin),
              verbose=0, max_memory=40000)
    if basis.lower().startswith('def2') and any(
            gto.charge(s) > 36 for s, _ in gto.format_atom(xyz)):
        kw['ecp'] = basis
    mol = gto.M(**kw)
    t1 = time.perf_counter()
    mf = (dft.UKS if int(spin) else dft.RKS)(mol, xc=xc)
    mf.grids.level = 3
    mf.direct_scf_tol = 1e-13
    mf.max_cycle = 1
    mf.verbose = 0
    mf.kernel()
    t2 = time.perf_counter()
    mf.nuc_grad_method().kernel()
    t3 = time.perf_counter()
    print('JOB ' + json.dumps(dict(imports=t1-t0, scf=t2-t1, grad=t3-t2,
                                   total=t3-t0)))
''')

tmp = tempfile.mkdtemp(prefix='warm_start_')
job_py = os.path.join(tmp, 'job.py')
with open(job_py, 'w') as f:
    f.write(JOB)


def run(argv, cache, env_extra=None):
    env = dict(os.environ)
    env['CUPY_CACHE_DIR'] = cache
    env['PYTHONPATH'] = REPO + os.pathsep + env.get('PYTHONPATH', '')
    env.update(env_extra or {})
    os.makedirs(cache, exist_ok=True)
    t = time.perf_counter()
    out = subprocess.run([sys.executable] + argv, capture_output=True,
                         text=True, env=env, cwd=REPO)
    dt = time.perf_counter() - t
    if out.returncode != 0:
        tail = (out.stderr or out.stdout).strip().splitlines()[-8:]
        raise SystemExit('failed: ' + ' '.join(argv) + '\n  '
                         + '\n  '.join(tail))
    return dt, out.stdout


job_args = [job_py, a.xyz, str(a.charge), str(a.spin), a.basis, a.xc]
warm_args = ['-m', 'electrolites.warmup', '--basis', a.basis,
             '--xyz', a.xyz, '--charge', str(a.charge), '--spin', str(a.spin),
             '--modules', a.modules]


def job_row(out):
    line = [l for l in out.splitlines() if l.startswith('JOB ')]
    return json.loads(line[-1][4:]) if line else {}


rows = {}
print(f'# {a.xyz.rsplit("/", 1)[-1]}  {a.basis}/{a.xc}  '
      f'charge {a.charge} spin {a.spin}  warming {a.modules}', flush=True)

# 1. cold: the compile happens inside the run
c = os.path.join(tmp, 'cache_cold')
dt, out = run(job_args, c)
rows['cold job'] = dict(wall=dt, **job_row(out))
print(f'  cold job (empty cache)                {dt:8.1f}s', flush=True)

# 2. warm, one worker
c1 = os.path.join(tmp, 'cache_serial')
dt1, out1 = run(warm_args + ['--jobs', '1'], c1)
rows['warmup jobs=1'] = dict(wall=dt1)
print(f'  warmup --jobs 1                       {dt1:8.1f}s', flush=True)
dt, out = run(job_args, c1)
rows['job after warmup'] = dict(wall=dt, **job_row(out))
print(f'  job on that cache                     {dt:8.1f}s', flush=True)

# 3. warm, N workers, from scratch again
cn = os.path.join(tmp, f'cache_{a.jobs}')
dtn, outn = run(warm_args + ['--jobs', str(a.jobs)], cn)
rows[f'warmup jobs={a.jobs}'] = dict(wall=dtn)
print(f'  warmup --jobs {a.jobs:<2d}                      {dtn:8.1f}s  '
      f'({dt1 / dtn:.2f}x)', flush=True)
dt, out = run(job_args, cn)
rows[f'job after warmup, jobs={a.jobs}'] = dict(wall=dt, **job_row(out))
print(f'  job on that cache                     {dt:8.1f}s', flush=True)

cold = rows['cold job']['wall']
warm = rows['job after warmup']['wall']
print(f'\n# the job itself: {cold:.1f}s cold, {warm:.1f}s warm -- '
      f'{cold - warm:.1f}s of it was NVRTC')
print(f'# filling the cache: {dt1:.1f}s at one worker, {dtn:.1f}s at '
      f'{a.jobs} ({dt1 / dtn:.2f}x)')
print('RESULT ' + json.dumps(dict(
    xyz=a.xyz, basis=a.basis, xc=a.xc, charge=a.charge, spin=a.spin,
    jobs=a.jobs, modules=a.modules, rows=rows)), flush=True)

if a.keep:
    print(f'# caches left in {tmp}')
else:
    shutil.rmtree(tmp, ignore_errors=True)
