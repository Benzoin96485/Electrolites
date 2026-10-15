"""``fasttdpcmgrad`` (FEATURE_TARGETS V5): the TDDFT-PCM gradient with the
surface derivatives built once against GPU4PySCF's three builds, IEF-PCM,
SMD and SS(V)PE, TDA and TDDFT, to round-off (1e-12).

    python tests/test_fasttdpcmgrad.py
"""
import os
import subprocess
import sys

SNIPPET = r'''
import sys, time, json, numpy as np, cupy as cp
from pyscf import gto
import electrolites
electrolites.patch('fasttdpcmgrad')
from electrolites import fasttdpcmgrad as F
from gpu4pyscf import dft
mol = gto.M(atom="benchmarks/molecules/glycine.xyz", basis="def2-svp", verbose=0)
out = {}
for method in ("IEF-PCM", "SMD", "SS(V)PE"):
    for kind in ("TDA", "TDDFT"):
        mf = dft.RKS(mol, xc="b3lyp")
        mf = mf.SMD() if method == "SMD" else mf.PCM()
        if method != "SMD":
            mf.with_solvent.method = method
        mf.with_solvent.eps = 78.3553
        if method == "SMD":
            mf.with_solvent.solvent = "water"
        mf.run(conv_tol=1e-10)
        td = getattr(mf, kind)(equilibrium_solvation=True); td.nstates = 3; td.kernel()
        g = td.nuc_grad_method()
        g.kernel(state=1)
        t = time.perf_counter(); de = g.kernel(state=1); cp.cuda.Stream.null.synchronize()
        out[method + " " + kind] = (np.asarray(de).tolist(), time.perf_counter() - t)
print("JSON" + json.dumps({"res": out, "stats": F.STATS}))
'''


def run(off):
    env = dict(os.environ, FASTTDPCMGRAD_OFF='1' if off else '')
    if not off:
        env.pop('FASTTDPCMGRAD_OFF')
    r = subprocess.run([sys.executable, '-c', SNIPPET], env=env,
                       capture_output=True, text=True)
    line = [x for x in r.stdout.splitlines() if x.startswith('JSON')]
    if not line:
        print(r.stderr[-3000:])
        raise SystemExit(1)
    import json
    return json.loads(line[0][4:])


import numpy as np                                             # noqa: E402
new, old = run(False), run(True)
fails = []
for k, (de, t) in new['res'].items():
    de0, t0 = old['res'][k]
    err = abs(np.asarray(de) - np.asarray(de0)).max()
    ok = err < 1e-12
    if not ok:
        fails.append(k)
    print(f'  {"ok  " if ok else "FAIL"} {k:20s} |d grad| {err:.1e}  '
          f'{t0:.2f} s -> {t:.2f} s')
print('builds', new['stats'])
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
