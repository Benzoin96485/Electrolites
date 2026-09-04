"""Gradient correctness: patched vs unpatched GPU4PySCF, same converged density.

Each case converges one SCF, then evaluates the gradient twice in the same
process -- once through GPU4PySCF's own RYS_per_atom_jk_ip1 and the
short-range operator, once through fastejk/fastgrad -- so the two see exactly
the same orbitals and the only difference is the two-electron gradient build.
Includes the fall-through paths: deep contractions (cc-pVTZ), g functions
(cc-pVQZ), open shell (UKS), and omega < 0 (HSE06).  Meta-GGA is no longer a
fall-through -- fastxcgrad covers it -- so TPSS, M06-2X and wB97M-V are here
as covered cases, together with the omega-PBE hybrid lrc-wPBEh.
"""
import sys, time
import numpy as np
import electrolites
electrolites.patch('fastejk', 'fastgrad', 'fastxcgrad', 'fastgradh')
# the module objects too: this test flips the patches on and off inside one
# process, so it needs the replacements as well as the originals
from electrolites import fastejk, fastgrad, fastgradh, fastxcgrad
from pyscf import gto
import cupy as cp
from gpu4pyscf import dft, scf
from gpu4pyscf.grad import rhf as grhf
from gpu4pyscf.grad import rks as grks

H2O = 'O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467'
CH3 = 'C 0 0 0; H 0 1.078 0; H 0.934 -0.539 0; H -0.934 -0.539 0'
SIC = 'Si 0 0 0; Cl 1.02 1.02 1.02; Cl -1.02 -1.02 1.02; Cl 1.02 -1.02 -1.02; Cl -1.02 1.02 -1.02'
C2H4 = 'C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.238; H 0 -0.923 1.238; H 0 0.923 -1.238; H 0 -0.923 -1.238'
NH3 = 'N 0 0 0.12; H 0 0.94 -0.27; H 0.81 -0.47 -0.27; H -0.81 -0.47 -0.27'

CASES = [
    (H2O,  '6-31G*',    'b3lyp',    0, 0),
    (H2O,  '6-31G*',    'b3lyp-d3bj', 0, 0),
    (H2O,  'def2-SVP',  'pbe0',     0, 0),
    (H2O,  'def2-TZVP', 'b3lyp',    0, 0),
    (H2O,  'def2-TZVPD','wb97m-v',  0, 0),
    (H2O,  'cc-pVTZ',   'wb97x-v',  0, 0),
    (H2O,  'cc-pVQZ',   'b3lyp',    0, 0),   # g functions -> fall through
    (H2O,  '6-31G*',    'hse06',    0, 0),   # omega < 0 -> fall through
    (H2O,  '6-31G*',    'camb3lyp', 0, 0),
    (H2O,  '6-31G*',    'pbe',      0, 0),   # pure functional, no K
    (H2O,  '6-31G*',    'tpss',     0, 0),   # meta-GGA, pure
    (H2O,  'def2-TZVP', 'm06-2x',   0, 0),   # meta-GGA, global hybrid
    (H2O,  '6-31G*',    'lrc-wpbeh',0, 0),   # omega-PBE range-separated hybrid
    (C2H4, 'def2-SVP',  'm06-2x',   0, 0),
    (C2H4, '6-31G*',    'b3lyp',    0, 0),
    (C2H4, 'def2-TZVPD','wb97m-v',  0, 0),
    (NH3,  'def2-TZVP', 'wb97x',    0, 0),
    (SIC,  'def2-TZVP', 'b3lyp',    0, 0),   # third row
    (SIC,  '6-31G*',    'wb97m-v',  0, 0),
    (SIC,  '6-31G*',    'm06-2x',   0, 0),   # meta-GGA, third row
    (CH3,  '6-31G*',    'm06-2x',   0, 1),   # meta-GGA open shell -> UKS
    (CH3,  '6-31G*',    'b3lyp',    0, 1),   # open shell -> UKS fall-through
    (H2O,  '6-31G*',    'b3lyp',    1, 1),   # cation
]

print(f'{"molecule":9s} {"basis":12s} {"xc":11s} {"chg/sp":7s} '
      f'{"maxdiff":>10s} {"rel":>10s}  {"t_ref":>7s} {"t_new":>7s}')
worst = 0.
for atom, basis, xc, chg, spin in CASES:
    mol = gto.M(atom=atom, basis=basis, charge=chg, spin=spin, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc)
    mf.conv_tol = 1e-11
    mf.direct_scf_tol = 1e-14
    mf.verbose = 0
    mf.kernel()
    g = mf.nuc_grad_method()
    g.verbose = 0
    # unpatched
    grhf._jk_energy_per_atom = grhf._jk_energy_per_atom_orig
    grhf.GradientsBase.get_hcore = grhf._get_hcore_orig
    grks.Gradients.energy_ee = grks.Gradients._energy_ee_orig
    grks.get_exc = grks._get_exc_orig
    grks.get_nlc_exc = grks._get_nlc_exc_orig
    t = time.perf_counter(); de_ref = g.kernel(); t_ref = time.perf_counter()-t
    # patched
    grhf._jk_energy_per_atom = fastejk.jk_energy_per_atom
    grhf.GradientsBase.get_hcore = fastgradh.get_hcore
    grks.Gradients.energy_ee = fastgrad.energy_ee
    grks.get_exc = fastxcgrad.get_exc
    grks.get_nlc_exc = fastxcgrad.get_nlc_exc
    t = time.perf_counter(); de_new = g.kernel(); t_new = time.perf_counter()-t
    d = np.abs(de_new - de_ref).max()
    rel = np.linalg.norm(de_new-de_ref)/max(np.linalg.norm(de_ref), 1e-30)
    worst = max(worst, d)
    name = atom.split()[0] + str(mol.natm)
    print(f'{name:9s} {basis:12s} {xc:11s} {chg}/{spin:<5d} '
          f'{d:10.2e} {rel:10.2e}  {t_ref:7.2f} {t_new:7.2f}', flush=True)
print(f'worst absolute difference over {len(CASES)} cases: {worst:.2e}')
