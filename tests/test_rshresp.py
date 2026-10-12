"""Range-separated hybrids' response J + K in one exchange pass (C12).

``electrolites/fastrshresp.py`` rewrites the RSH statements of
``_gen_rhf_response``/``_gen_uhf_response``.  Checked, each case in its own
subprocess against ``FASTRSHRESP_OFF=1``: TDA singlet and triplet and TDDFT
excitation energies (restricted, wB97X-D3BJ and wB97M-V), an unrestricted
TDA, and a static polarizability, on glycine/def2-SVP, with the arrangement
forced to ``fused`` and ``split`` and left on ``auto``.  Bars: 1e-8 Eh in the
excitation energies, 1e-6 in the polarizability trace (the solvers' own
tolerances), and the fused path must have been taken where forced.

    python tests/test_rshresp.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLY = os.path.join(HERE, 'benchmarks', 'molecules', 'glycine.xyz')
RUN = '''
import sys, os, json
sys.path.insert(0, {here!r})
os.environ.update({env!r})
import numpy as np
import electrolites; electrolites.patch_all()
from pyscf import gto
from gpu4pyscf import dft
mol = gto.M(atom={gly!r}, basis='def2-svp', charge={charge}, spin={spin},
            verbose=0)
mf = (dft.UKS if {spin} else dft.RKS)(mol, xc={xc!r})
mf.conv_tol = 1e-11
mf.kernel()
kind = {kind!r}
if kind == 'pol':
    from gpu4pyscf.properties import polarizability
    out = [float(np.trace(polarizability.eval_polarizability(mf)))]
else:
    td = mf.TDA() if kind.startswith('tda') else mf.TDDFT()
    td.nstates = 4
    td.conv_tol = 1e-9
    if kind == 'tda_t':
        td.singlet = False
    td.kernel()
    out = [float(x) for x in np.asarray(td.e)]
stats = {{}}
if 'FASTRSHRESP_OFF' not in os.environ:
    from electrolites import fastrshresp
    stats = {{k: v for k, v in fastrshresp.STATS.items() if k != 'decisions'}}
print('JSON' + json.dumps([out, stats]))
'''


def run(kind, xc, env=(), charge=0, spin=0):
    code = RUN.format(here=HERE, env=dict(env), gly=GLY, xc=xc, kind=kind,
                      charge=charge, spin=spin)
    p = subprocess.run([sys.executable, '-c', code], capture_output=True,
                       text=True)
    for line in p.stdout.splitlines():
        if line.startswith('JSON'):
            return json.loads(line[4:])
    raise RuntimeError(p.stdout[-3000:] + p.stderr[-3000:])


CASES = [('tda', 'wb97x-d3bj', 0, 0), ('tda_t', 'wb97x-d3bj', 0, 0),
         ('tddft', 'wb97m-v', 0, 0), ('tda', 'wb97x-d3bj', 1, 1),
         ('pol', 'wb97x-d3bj', 0, 0)]


def main():
    ok = True
    for kind, xc, charge, spin in CASES:
        ref, _ = run(kind, xc, {'FASTRSHRESP_OFF': '1'}, charge, spin)
        bar = 1e-6 if kind == 'pol' else 1e-8
        for mode in ('fused', 'split', 'auto'):
            new, st = run(kind, xc, {'FASTRSHRESP_MODE': mode}, charge, spin)
            d = max(abs(a - b) for a, b in zip(new, ref))
            took = st.get('j_plus_fused', 0) + st.get('fused_k', 0)
            good = d < bar and (mode != 'fused' or took > 0)
            ok &= good
            print(f'{kind:6s} {xc:11s} {"UKS" if spin else "RKS"} {mode:5s}: '
                  f'max|d| {d:.1e}  {st}  {"ok" if good else "FAIL"}',
                  flush=True)
    print('ALL OK' if ok else 'FAILED')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
