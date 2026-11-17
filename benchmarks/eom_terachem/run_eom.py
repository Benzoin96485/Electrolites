"""EOM-EE-CCSD (singlet, 5 roots) with Electrolites on a TeraChem RR-EOM-CCSD case.

    python run_eom.py Ir/ACRHCO02_comp_0 [nroots]

Setup copied from the case's EOM-CCSD.inp: cc-pVDZ (spherical) + LANL2DZ/ECP on
the metal, RHF, frozen core = ccbox_frozen_core, charge from the input.
Coordinates converted to Bohr with TeraChem's constant (0.529177249) so E_nuc
matches TeraChem's.
"""
import json
import os
import re
import sys
import time

import numpy as np

ROOT = ('/global/homes/w/weiliang/m4787_cfs/data/EOM-RR-CCSD/'
        'xyz_15-30atoms_spin1_opt')
TC_BOHR = 0.529177249
case = sys.argv[1]
nroots = int(sys.argv[2]) if len(sys.argv) > 2 else 5
cdir = os.path.join(ROOT, case)
inp = open(os.path.join(cdir, 'EOM-CCSD.inp')).read()
xyzname = re.search(r'^coordinates\s+(\S+)', inp, re.M).group(1)
charge = int(re.search(r'^charge\s+(-?\d+)', inp, re.M).group(1))
nfrz = int(re.search(r'^ccbox_frozen_core\s+(\d+)', inp, re.M).group(1))
metal = re.search(r'^\s+(\w+)\s+lanl2dz_ecp', inp, re.M).group(1)
lines = open(os.path.join(cdir, xyzname)).read().splitlines()
nat = int(lines[0])
atoms = []
for ln in lines[2:2 + nat]:
    s, x, y, z = ln.split()[:4]
    atoms.append((s, [float(x) / TC_BOHR, float(y) / TC_BOHR,
                      float(z) / TC_BOHR]))

import cupy as cp
from pyscf import gto
import electrolites
electrolites.patch_all()
from gpu4pyscf import scf
from gpu4pyscf.cc import ccsd_incore

t = {}
T0 = time.perf_counter()
mol = gto.M(atom=atoms, unit='Bohr', charge=charge, spin=0,
            basis={metal: 'lanl2dz', 'default': 'cc-pvdz'},
            ecp={metal: 'lanl2dz'}, cart=False, verbose=4,
            max_memory=200000)
print(f'case {case}: natm {nat} nao {mol.nao} nelec {mol.nelectron} '
      f'charge {charge} frozen {nfrz}  E_nuc {mol.energy_nuc():.9f}', flush=True)

t0 = time.perf_counter()
mf = scf.RHF(mol)
mf.conv_tol = 1e-10
mf.max_cycle = 200
mf.kernel()
if not mf.converged:
    mf = mf.newton().run()
cp.cuda.Device().synchronize()
t['scf'] = time.perf_counter() - t0

t0 = time.perf_counter()
mycc = ccsd_incore.CCSD(mf, frozen=nfrz)
mycc.conv_tol = 1e-8
mycc.conv_tol_normt = 1e-6
mycc.max_cycle = 100
mycc.kernel()
cp.cuda.Device().synchronize()
t['ccsd'] = time.perf_counter() - t0

t0 = time.perf_counter()
from electrolites import eom_gpu
eom = eom_gpu.EOMEESinglet(mycc)
eom.conv_tol = 1e-7
e_ee, v_ee = eom.kernel(nroots=nroots)
cp.cuda.Device().synchronize()
t['eom'] = time.perf_counter() - t0
t['total'] = time.perf_counter() - T0

e_ee = np.atleast_1d(np.asarray(e_ee, dtype=float))
HARTREE2EV = 27.211386245988
out = dict(case=case, nao=mol.nao, natm=nat, frozen=nfrz,
           nocc_act=int(mycc.nocc), nvir=int(mycc.nmo - mycc.nocc),
           e_nuc=mol.energy_nuc(),
           e_scf=float(mf.e_tot), e_corr=float(mycc.e_corr),
           ccsd_converged=bool(mycc.converged),
           ee_au=e_ee.tolist(), ee_ev=(e_ee * HARTREE2EV).tolist(),
           times=t, gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
print('RESULT', json.dumps(out), flush=True)
os.makedirs('results_eom', exist_ok=True)
with open(os.path.join('results_eom', case.replace('/', '__') + '.json'), 'w') as f:
    json.dump(out, f, indent=1)
