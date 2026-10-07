"""The multi-collinear XC on half the spin samples against all of them (A21).

``electrolites/fastmcfun.py`` hands ``mcfun_gpu._eval_xc_lebedev`` one
sample of each antipodal pair of the Lebedev set with twice the weight.
Checked: the halved set really is one member of every pair (770 and the
other sizes GKS uses); ``exc_eff``, ``vxc_eff`` and ``fxc_eff`` from the
halved and the full set on the same synthetic densities, for an LDA, a GGA
and a meta-GGA functional, relative to the largest element; and a whole GKS
SCF (``collinear='mcol'``) against ``FASTMCFUN_OFF=1`` in a subprocess.

    python tests/test_mcfun.py
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []


def check(label, got, ref, tol):
    got, ref = np.asarray(got), np.asarray(ref)
    err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-300)
    ok = err <= tol
    if not ok:
        fails.append(f'{label}: {err:.3e}')
    print(f'  {"ok  " if ok else "FAIL"} {label:48s} {err:.3e}', flush=True)


GKS = '''
import sys, json; sys.path.insert(0, {here!r})
import os
if {off!r}: os.environ["FASTMCFUN_OFF"] = "1"
import electrolites; electrolites.patch_all()
from pyscf import gto
from gpu4pyscf import dft
mol = gto.M(atom={atom!r}, basis="def2-svp", spin={spin}, verbose=0)
mf = dft.GKS(mol, xc={xc!r}); mf.collinear = "mcol"
mf.conv_tol = 1e-9; mf.max_cycle = 80
e = mf.kernel()
print(json.dumps([float(e), bool(mf.converged)]))
'''


def gks(off, atom, spin, xc):
    out = subprocess.run([sys.executable, '-c', GKS.format(
        here=HERE, off=off, atom=atom, spin=spin, xc=xc)],
        capture_output=True, text=True, check=True).stdout
    import json
    return json.loads(out.strip().splitlines()[-1])


if __name__ == '__main__':
    sys.path.insert(0, HERE)
    import electrolites
    electrolites.patch_all()
    import cupy as cp
    from gpu4pyscf.dft import mcfun_gpu as M, numint2c
    from electrolites import fastmcfun

    print('## the halved sample sets')
    for n in (110, 302, 770, 1202):
        d, w = M._make_sph_samples(n)
        D, W = M._make_sph_samples_orig(n)
        d, D = np.asarray(d), np.asarray(D)
        ok = (2 * len(d) == len(D) and abs(2 * W.sum() - 2 * w.sum()) < 1e-12
              and abs(np.concatenate([d, -d]) @ D.T).max(axis=1).min()
              > 1 - 1e-12)
        if not ok:
            fails.append(f'samples {n}')
        print(f'  {"ok  " if ok else "FAIL"} {f"{n} samples -> {len(d)}":48s}')

    print('## eval_xc_eff, halved against full')
    rng = np.random.default_rng(5)
    ng = 3000
    for xc, nvar in (('lda,vwn', 1), ('pbe', 4), ('b3lyp', 4), ('tpss', 5)):
        # a meta-GGA's kernel at 770 samples is (2*5)^3 doubles a point
        ng = 300 if nvar == 5 else 3000
        rho = rng.uniform(1e-3, 1., ng)
        m = rng.normal(size=(3, ng))
        m *= (rho * rng.uniform(0, .95, ng) / np.linalg.norm(m, axis=0))
        if nvar == 1:
            rho_tm = cp.asarray(np.vstack([rho, m]))
        else:
            rho_tm = np.empty((4, nvar, ng))
            rho_tm[0, 0] = rho
            rho_tm[1:, 0] = m
            rho_tm[:, 1:4] = rng.normal(scale=.1, size=(4, 3, ng))
            if nvar == 5:
                rho_tm[0, 4] = rho * rng.uniform(.5, 2, ng)
                rho_tm[1:, 4] = m * .3
            rho_tm = cp.asarray(rho_tm)
        ni = numint2c.NumInt2C()
        ni.collinear = 'mcol'
        fn = ni.mcfun_eval_xc_adapter(xc)
        for deriv in (1, 2):
            ours = fn(xc, rho_tm, deriv=deriv)
            M._make_sph_samples, keep = M._make_sph_samples_orig, \
                M._make_sph_samples
            try:
                ref = fn(xc, rho_tm, deriv=deriv)
            finally:
                M._make_sph_samples = keep
            for k, (a, b) in enumerate(zip(ours, ref)):
                check(f'{xc} deriv={deriv} {["exc", "vxc", "fxc"][k]}',
                      a.get(), b.get(), 1e-12)
    print('# fastmcfun', fastmcfun.stats())

    print('## whole GKS SCF, against FASTMCFUN_OFF=1')
    O2 = 'O 0 0 0; O 0 0 1.21'
    for tag, atom, spin, xc in (('O2 triplet PBE', O2, 2, 'pbe'),
                                ('O2 triplet B3LYP', O2, 2, 'b3lyp')):
        a, b = gks(False, atom, spin, xc), gks(True, atom, spin, xc)
        err = abs(a[0] - b[0])
        ok = err <= 1e-8 and a[1] == b[1]
        if not ok:
            fails.append(f'{tag}: {err:.3e} conv {a[1]} {b[1]}')
        print(f'  {"ok  " if ok else "FAIL"} {tag + " energy":48s} {err:.3e}'
              f'  ({a[0]:.10f})')
    if fails:
        print('\nFAILED:\n  ' + '\n  '.join(fails))
        sys.exit(1)
    print('\nall ok')
