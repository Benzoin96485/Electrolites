"""int1e_grids with many densities / charge vectors in one pass (F9).

``electrolites/fastpcmbatch.py`` builds the plain integrals once per chunk of
cavity points and contracts them with every right-hand side by GEMM.
Checked against GPU4PySCF's per-vector loop (``_int1e_grids_orig``) on a PCM
cavity, density and charge stacks, spherical and cartesian bases, element-wise
relative to the largest element; then a whole IEF-PCM Hessian and a C-PCM
TDA against ``FASTPCMBATCH_OFF=1`` in subprocesses.

    python tests/test_pcmbatch.py
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []
ETHANOL = '''C -0.748 -0.015 0.024; C 0.558 0.420 -0.679; O 1.598 -0.513 -0.421;
H -1.593 0.365 -0.554; H -0.834 0.355 1.048; H -0.780 -1.106 0.023;
H 0.640 1.502 -0.580; H 0.567 0.174 -1.741; H 2.421 -0.190 -0.792'''

RUN = '''
import sys, os, json
sys.path.insert(0, {here!r})
if {off!r}: os.environ["FASTPCMBATCH_OFF"] = "1"
import electrolites; electrolites.patch_all()
import numpy as np, cupy as cp
from pyscf import gto
from gpu4pyscf import dft
mol = gto.M(atom={atom!r}, basis="def2-svp", verbose=0)
mf = dft.RKS(mol, xc="b3lyp").PCM(); mf.with_solvent.method = {method!r}
mf.conv_tol = 1e-11; mf.kernel()
if {what!r} == "hess":
    h = mf.Hessian(); h.verbose = 0
    out = cp.asnumpy(cp.asarray(h.kernel())).ravel().tolist()
else:
    td = mf.TDA(); td.nstates = 4; td.kernel()
    out = list(map(float, td.e))
print(json.dumps(out))
'''


def whole(off, method, what):
    r = subprocess.run([sys.executable, '-c', RUN.format(
        here=HERE, off=off, atom=ETHANOL, method=method, what=what)],
        capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    import json
    return np.asarray(json.loads(r.stdout.strip().splitlines()[-1]))


if __name__ == '__main__':
    sys.path.insert(0, HERE)
    import electrolites
    electrolites.patch_all()
    import cupy as cp
    from pyscf import gto
    from gpu4pyscf import dft
    from gpu4pyscf.gto import int3c1e as I3
    from electrolites import fastpcmbatch

    for cart in (False, True):
        mol = gto.M(atom=ETHANOL, basis='def2-svp', cart=cart, verbose=0)
        mf = dft.RKS(mol, xc='b3lyp').PCM()
        mf.with_solvent.build()
        s = mf.with_solvent
        g = s.surface['grid_coords']
        e = s.surface['charge_exp'] ** 2
        rng = np.random.default_rng(1)
        nao = mol.nao
        dms = cp.asarray(rng.normal(size=(7, nao, nao)))
        q = cp.asarray(rng.normal(size=(7, g.shape[0])))
        for lbl, kw in (('densities', {'dm': dms}), ('charges', {'charges': q})):
            ours = fastpcmbatch.int1e_grids(mol, g, charge_exponents=e,
                                            intopt=s.intopt, **kw)
            ref = I3._int1e_grids_orig(mol, g, charge_exponents=e,
                                       intopt=s.intopt, **kw)
            err = float(abs(ours - ref).max() / abs(ref).max())
            ok = err <= 1e-12
            if not ok:
                fails.append(f'cart={cart} {lbl}: {err:.3e}')
            print(f'  {"ok  " if ok else "FAIL"} cart={cart!s:5s} 7 {lbl:9s} '
                  f'{err:.3e}', flush=True)
    print('# fastpcmbatch', fastpcmbatch.stats())
    for method, what, tol in (('IEF-PCM', 'hess', 1e-8), ('C-PCM', 'tda', 1e-9)):
        a, b = whole(False, method, what), whole(True, method, what)
        err = float(np.abs(a - b).max())
        ok = err <= tol
        if not ok:
            fails.append(f'{method} {what}: {err:.3e}')
        print(f'  {"ok  " if ok else "FAIL"} ethanol {method} {what} against '
              f'FASTPCMBATCH_OFF {err:.3e}', flush=True)
    if fails:
        print('\nFAILED:\n  ' + '\n  '.join(fails))
        sys.exit(1)
    print('\nall ok')
