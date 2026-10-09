"""Direct SCF with a loose early screen and a tight rebuild (A20).

``electrolites/fastscfscreen.py`` screens the incremental J/K builds at 1e-10
while ``max|dm - dm_last|`` is above 1e-3 and then rebuilds the Fock matrix
from scratch at ``direct_scf_tol``.  Checked: the converged energy against
``FASTSCFSCREEN_OFF=1`` (each in its own subprocess) for RKS, UKS, RHF and a
range-separated hybrid, with ``conv_tol`` 1e-10 -- the bar is 1e-9 Eh, the
SCF's own noise at that tolerance -- and that each run went through loose
builds and exactly one rebuild.

    python tests/test_scfscreen.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = '''
import sys, os, json
sys.path.insert(0, {here!r})
if {off!r}: os.environ["FASTSCFSCREEN_OFF"] = "1"
import electrolites; electrolites.patch_all()
from electrolites import fastscfscreen
from pyscf import gto
from gpu4pyscf import dft, scf
mol = gto.M(atom={atom!r}, basis="def2-svp", charge={charge}, spin={spin},
            verbose=0)
kind = {kind!r}
if kind == "hf":
    mf = (scf.UHF if {spin} else scf.RHF)(mol)
else:
    mf = (dft.UKS if {spin} else dft.RKS)(mol, xc=kind)
mf.conv_tol = 1e-10
e = mf.kernel()
print(json.dumps([float(e), bool(mf.converged), fastscfscreen.stats()]))
'''
ALANINE = os.path.join(HERE, 'benchmarks', 'molecules', 'alanine_dipeptide.xyz')
CASES = [('alanine dipeptide RKS B3LYP', ALANINE, 0, 0, 'b3lyp'),
         ('alanine dipeptide+ UKS B3LYP', ALANINE, 1, 1, 'b3lyp'),
         ('alanine dipeptide RHF', ALANINE, 0, 0, 'hf'),
         ('alanine dipeptide RKS wB97X', ALANINE, 0, 0, 'wb97x')]


def run(off, atom, charge, spin, kind):
    r = subprocess.run([sys.executable, '-c', RUN.format(
        here=HERE, off=off, atom=atom, charge=charge, spin=spin, kind=kind)],
        capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


fails = []
for tag, atom, charge, spin, kind in CASES:
    a, b = run(False, atom, charge, spin, kind), run(True, atom, charge, spin, kind)
    de = abs(a[0] - b[0])
    st = a[2]
    ok = de <= 1e-9 and a[1] and b[1] and st['rebuilds'] == 1 \
        and st['loose_builds'] > 0
    if not ok:
        fails.append(f'{tag}: dE {de:.3e} {st}')
    print(f'  {"ok  " if ok else "FAIL"} {tag:32s} dE {de:.2e}  loose builds '
          f'{st["loose_builds"]}, rebuilds {st["rebuilds"]}', flush=True)
if fails:
    print('\nFAILED:\n  ' + '\n  '.join(fails))
    sys.exit(1)
print('\nall ok')
