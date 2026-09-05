"""First-run NVRTC cost, whole family versus the classes a basis reaches.

Nothing here is compiled at install time: every kernel is built at run time by
NVRTC and cached by CuPy on the source hash, so the first run of a given
kernel set pays for all of it.  Compile *options* do not move that number --
`--minimal` and `--use_fast_math` together are worth 4 % on `fastk`, because
the cost is the volume of generated code.  Handing NVRTC only the angular
momentum classes the basis can reach does move it.

Every timing here disables CuPy's disk cache, so each one is a real compile.

    python benchmarks/compile_time.py [--lmax 2]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--lmax', type=int, nargs='*', default=[2, 3],
                help='basis maximum angular momenta to report (2 = 6-31G*, '
                     '3 = def2-TZVP with f)')
ap.add_argument('--modules', default='fastk,fastejk,fastj')
a = ap.parse_args()

CHILD = textwrap.dedent('''
    import json, os, sys, time
    os.environ['CUPY_CACHE_IN_MEMORY'] = '1'
    os.environ['CUPY_CACHE_SAVE_CUDA_SOURCE'] = '0'
    import cupy as cp
    import electrolites
    from electrolites import _ksplit
    mod_name, lmax = sys.argv[1], (None if sys.argv[2] == 'none' else int(sys.argv[2]))
    mod = __import__('electrolites.' + mod_name, fromlist=['x'])
    files = mod._sources()
    names = None if lmax is None else _ksplit.names_within(files, lmax)
    kept, total = _ksplit.counts(files, names)
    src = _ksplit.source(files, names)
    t = time.perf_counter()
    m = cp.RawModule(code=src, options=('-std=c++17',), backend='nvrtc')
    m.compile()
    cp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - t
    print('CHILD ' + json.dumps(dict(module=mod_name, lmax=lmax, kept=kept,
                                     total=total, bytes=len(src), t=dt)))
''')

tmp = tempfile.TemporaryDirectory()
script = os.path.join(tmp.name, '_compile_child.py')
with open(script, 'w') as f:
    f.write(CHILD)

rows = []
print(f'{"module":10s} {"lmax":>5s} {"kernels":>9s} {"source":>9s} '
      f'{"compile":>9s} {"vs full":>8s}')
for mod in a.modules.split(','):
    full = None
    for lmax in list(a.lmax) + ['none']:
        out = subprocess.run([sys.executable, script, mod, str(lmax)],
                             capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith('CHILD ')]
        if not line:
            print(f'{mod:10s} {str(lmax):>5s}  FAILED: '
                  f'{out.stderr.strip().splitlines()[-1] if out.stderr else "?"}')
            continue
        r = json.loads(line[0][6:])
        rows.append(r)
        if lmax == 'none':
            full = r['t']
        sp = f'{full/r["t"]:7.2f}x' if full else ''
        print(f'{mod:10s} {str(lmax):>5s} {r["kept"]:4d}/{r["total"]:<4d} '
              f'{r["bytes"]/1e6:7.2f}MB {r["t"]:8.2f}s {sp}')
    # re-print the ratios now that `full` is known
print()
byname = {}
for r in rows:
    byname.setdefault(r['module'], {})[r['lmax']] = r
for mod, d in byname.items():
    full = d.get(None)
    if not full:
        continue
    for lmax, r in sorted(d.items(), key=lambda kv: (kv[0] is None, kv[0])):
        if lmax is None:
            continue
        print(f'{mod}: lmax={lmax} compiles {r["kept"]}/{r["total"]} kernels in '
              f'{r["t"]:.1f}s against {full["t"]:.1f}s for the whole family '
              f'({full["t"]/r["t"]:.2f}x)')
print('RESULT ' + json.dumps(rows))
