"""Closed-shell CCSD on the device (I4, I5's host half).

``electrolites/fastccsd.py`` runs ``ccsd_incore.CCSD``'s iterations with the
amplitudes, MO integrals and DIIS on the device.  Checked: the correlation
energy, the cycle count and the amplitudes against the stock driver
(``FASTCCSD_OFF=1`` in a subprocess) on glycine/def2-SVP (bar 1e-10 Eh,
1e-8 in t2), against PySCF's CPU CCSD on water/cc-pVDZ (bar 1e-9 Eh), with
the integral cache off, on glycine/def2-TZVP (f shells: the integral cache
goes to host memory on a 40 GB card; bar 5e-9, its convergence noise), and that a run that does not fit falls back to the
stock driver.

    python tests/test_ccsd.py
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
from electrolites import fastccsd
from pyscf import gto
from gpu4pyscf import scf
from gpu4pyscf.cc import ccsd_incore
if {nofit!r}:
    fastccsd._fits = lambda mycc: False
mol = gto.M(atom={atom!r}, basis={basis!r}, verbose=0)
mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.kernel()
cc = ccsd_incore.CCSD(mf); cc.conv_tol = 1e-10; cc.conv_tol_normt = {normt!r}; cc.max_cycle = 80
cc.kernel()
out = [float(cc.e_corr), int(cc.cycles), bool(cc.converged),
       float(np.asarray(cc.t2).ravel()[::97].sum()),
       float(abs(np.asarray(cc.t2)).max()), fastccsd.STATS]
if {cpu!r}:
    from pyscf import scf as cscf, cc as ccc
    m = cscf.RHF(mol); m.conv_tol = 1e-12; m.kernel()
    c = ccc.CCSD(m); c.conv_tol = 1e-10; c.conv_tol_normt = 1e-7; c.kernel()
    out.append(float(c.e_corr))
print('JSON' + json.dumps(out))
'''


def run(atom, basis, env=(), nofit=False, cpu=False, normt=1e-7):
    code = RUN.format(here=HERE, env=dict(env), atom=atom, basis=basis,
                      nofit=nofit, cpu=cpu, normt=normt)
    p = subprocess.run([sys.executable, '-c', code], capture_output=True,
                       text=True)
    for line in p.stdout.splitlines():
        if line.startswith('JSON'):
            return json.loads(line[4:])
    raise RuntimeError(p.stdout + p.stderr)


def main():
    ok = True
    ref = run(GLY, 'def2-svp', {'FASTCCSD_OFF': '1'})
    for label, kw in (('device', {}),
                      ('device, no integral cache',
                       {'env': {'FASTCCSD_NO_CACHE': '1'}})):
        new = run(GLY, 'def2-svp', **kw)
        de = abs(new[0] - ref[0])
        dt = abs(new[3] - ref[3])
        good = (de < 1e-10 and dt < 1e-8 and new[1] == ref[1] and new[2]
                and new[5]['runs'] == 1)
        ok &= good
        print(f'glycine/def2-svp {label}: e_corr {new[0]:.12f} vs stock '
              f'{ref[0]:.12f} (|d| {de:.1e}), cycles {new[1]}/{ref[1]}, '
              f't2 probe |d| {dt:.1e}  {"ok" if good else "FAIL"}')
    new = run(GLY, 'def2-svp', nofit=True)
    good = new[5]['fallbacks'] == 1 and abs(new[0] - ref[0]) < 1e-10
    ok &= good
    print(f'glycine/def2-svp, forced fallback: e_corr |d| '
          f'{abs(new[0] - ref[0]):.1e}  {"ok" if good else "FAIL"}')
    # this case converges slowly (40+ cycles at normt 1e-9): converged to
    # 1e-7 in the amplitudes, the two drivers' DIIS paths part at 2e-9 Eh
    # (stock -1.1112244424, device -1.1112244407, PySCF CPU -1.1112244404);
    # at 1e-9 they usually meet to 2e-12 but one run in five parted by
    # 1.4e-9, so the bar is the convergence noise, 5e-9
    ref = run(GLY, 'def2-tzvp', {'FASTCCSD_OFF': '1'}, normt=1e-9)
    new = run(GLY, 'def2-tzvp', normt=1e-9)
    de = abs(new[0] - ref[0])
    good = de < 5e-9 and new[2] and new[5]['runs'] == 1
    ok &= good
    where = ('host' if 'eri_host_cache_GB' in new[5] else
             'device' if 'eri_cache_GB' in new[5] else 'no')
    print(f'glycine/def2-tzvp (f shells; {where} integral cache; normt 1e-9): '
          f'e_corr |d| {de:.1e}, cycles {new[1]}/{ref[1]}  '
          f'{"ok" if good else "FAIL"}')
    water = 'O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467'
    new = run(water, 'cc-pvdz', cpu=True)
    de = abs(new[0] - new[-1])
    good = de < 1e-9 and new[5]['runs'] == 1
    ok &= good
    print(f'water/cc-pvdz: e_corr {new[0]:.12f} vs PySCF CPU {new[-1]:.12f} '
          f'(|d| {de:.1e})  {"ok" if good else "FAIL"}')
    print('ALL OK' if ok else 'FAILED')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
