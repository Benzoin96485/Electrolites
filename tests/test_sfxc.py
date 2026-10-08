"""The spin-flip multi-collinear XC kernel against GPU4PySCF's (D13).

``electrolites/fastsfxc.py`` computes ``fxc_sf`` and ``kxc_sf`` from libxc's
raw derivatives with the chain rule in the total/spin basis.  Checked against
GPU4PySCF's adapter (``_mcfun_eval_xc_adapter_sf_orig``) on the same
synthetic densities, ``deriv`` 2 and 3, for LDA, GGA and hybrid functionals,
relative to the largest element; and a whole SF-TDA + gradient against
``FASTSFXC_OFF=1`` in a subprocess.

    python tests/test_sfxc.py
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []

RUN = '''
import sys, json, os
sys.path.insert(0, {here!r})
if {off!r}: os.environ["FASTSFXC_OFF"] = "1"
import electrolites; electrolites.patch_all()
import numpy as np
from pyscf import gto
from gpu4pyscf import dft
mol = gto.M(atom="O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59", basis="def2-svp",
            spin=2, verbose=0)
mf = dft.UKS(mol, xc={xc!r}); mf.conv_tol = 1e-10; mf.kernel()
td = mf.SFTDA(); td.extype = 1; td.collinear = "mcol"; td.nstates = 3
td.kernel()
g = td.nuc_grad_method(); de = g.kernel(state=1)
print(json.dumps([list(map(float, td.e)), np.asarray(de).tolist()]))
'''


def whole(off, xc):
    out = subprocess.run([sys.executable, '-c', RUN.format(here=HERE, off=off,
                                                           xc=xc)],
                         capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr[-2000:])
    import json
    return json.loads(out.stdout.strip().splitlines()[-1])


if __name__ == '__main__':
    sys.path.insert(0, HERE)
    import electrolites
    electrolites.patch_all()
    import cupy as cp
    from gpu4pyscf.dft import numint
    from gpu4pyscf.tdscf import _uhf_resp_sf as SF
    from electrolites import fastsfxc

    ni = numint.NumInt()
    rng = np.random.default_rng(3)
    ng = 3000
    for xc, nvar in (('lda,vwn', 1), ('pbe', 4), ('b3lyp', 4), ('pbe0', 4)):
        rho = rng.uniform(1e-3, 1, ng)
        mz = rho * rng.uniform(-.95, .95, ng)
        if nvar == 1:
            r = cp.asarray(np.vstack([rho, mz]))
        else:
            r = np.zeros((2, 4, ng))
            r[0, 0], r[1, 0] = rho, mz
            r[:, 1:] = rng.normal(scale=.2, size=(2, 3, ng))
            r = cp.asarray(r)
        ours = fastsfxc.mcfun_eval_xc_adapter_sf(ni, xc, 20)
        ref = SF._mcfun_eval_xc_adapter_sf_orig(ni, xc, 20)
        for deriv in (2, 3):
            a = ours(xc, r, deriv=deriv)
            b = ref(xc, r, deriv=deriv)
            for k, lbl in ((2, 'fxc_sf'), (3, 'kxc_sf')):
                if deriv == 2 and k == 3:
                    continue
                x, y = cp.asnumpy(a[k]), cp.asnumpy(b[k])
                err = np.abs(x - y).max() / np.abs(y).max()
                ok = err <= 1e-11
                if not ok:
                    fails.append(f'{xc} deriv={deriv} {lbl}: {err:.3e}')
                print(f'  {"ok  " if ok else "FAIL"} {xc:8s} deriv={deriv} '
                      f'{lbl:7s} {err:.3e}', flush=True)
    print('# fastsfxc', fastsfxc.stats())
    print('## whole SF-TDA + gradient (water triplet), against FASTSFXC_OFF=1')
    for xc in ('pbe', 'b3lyp'):
        a, b = whole(False, xc), whole(True, xc)
        de = np.abs(np.asarray(a[0]) - np.asarray(b[0])).max()
        dg = np.abs(np.asarray(a[1]) - np.asarray(b[1])).max()
        ok = de <= 1e-9 and dg <= 1e-8
        if not ok:
            fails.append(f'whole {xc}: dE {de:.3e} dgrad {dg:.3e}')
        print(f'  {"ok  " if ok else "FAIL"} {xc:8s} excitation energies '
              f'{de:.3e}  gradient {dg:.3e}', flush=True)
    if fails:
        print('\nFAILED:\n  ' + '\n  '.join(fails))
        sys.exit(1)
    print('\nall ok')
