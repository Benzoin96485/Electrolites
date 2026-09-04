"""One SCF + gradient with the whole stack, or with none of it.

Run twice and diff:
    python bench/test_full.py --patch ''    > a.json
    python bench/test_full.py --patch all   > b.json
Comparing across processes is what makes this a real test: the energy modules
patch GPU4PySCF globally, so a same-process A/B would leave half of them on.
"""
import sys, json, time, argparse
import numpy as np
import electrolites

ap = argparse.ArgumentParser()
ap.add_argument('--patch', default='',
                help="'all', or a comma separated list of module names, or "
                     "'' for unpatched GPU4PySCF")
ap.add_argument('--only', default=None, help='run one case by name')
a = ap.parse_args()
if a.patch == 'all':
    electrolites.patch_all()
elif a.patch:
    electrolites.patch(a.patch.split(','))
print('patched:', electrolites.patched() or '(none)',
      '| gpu4pyscf', electrolites.gpu4pyscf_version(), file=sys.stderr)
from pyscf import gto
from gpu4pyscf import dft

H2O = 'O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467'
C2H4 = ('C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.238; H 0 -0.923 1.238; '
        'H 0 0.923 -1.238; H 0 -0.923 -1.238')
SIC = ('Si 0 0 0; Cl 1.02 1.02 1.02; Cl -1.02 -1.02 1.02; '
       'Cl 1.02 -1.02 -1.02; Cl -1.02 1.02 -1.02')
CASES = [
    ('H2O_631_b3lypd3',  H2O,  '6-31G*',     'b3lyp-d3bj', 0, 0),
    ('H2O_tzvpd_mv',     H2O,  'def2-TZVPD', 'wb97m-v',    0, 0),
    ('H2O_tzvp_b3lyp',   H2O,  'def2-TZVP',  'b3lyp',      0, 0),
    ('H2O_pvqz_b3lyp',   H2O,  'cc-pVQZ',    'b3lyp',      0, 0),
    ('H2O_631_hse06',    H2O,  '6-31G*',     'hse06',      0, 0),
    ('H2O_631_pbe',      H2O,  '6-31G*',     'pbe',        0, 0),
    ('H2O_631_tpss',     H2O,  '6-31G*',     'tpss',       0, 0),
    ('H2O_tzvp_m062x',   H2O,  'def2-TZVP',  'm06-2x',     0, 0),
    ('H2O_631_lrcwpbeh', H2O,  '6-31G*',     'lrc-wpbeh',  0, 0),
    ('C2H4_631_b3lypd3', C2H4, '6-31G*',     'b3lyp-d3bj', 0, 0),
    ('C2H4_tzvpd_mv',    C2H4, 'def2-TZVPD', 'wb97m-v',    0, 0),
    ('SiCl4_tzvp_b3lyp', SIC,  'def2-TZVP',  'b3lyp',      0, 0),
    ('SiCl4_631_mv',     SIC,  '6-31G*',     'wb97m-v',    0, 0),
    ('SiCl4_631_m062x',  SIC,  '6-31G*',     'm06-2x',     0, 0),
    ('CH3_631_b3lyp',    'C 0 0 0; H 0 1.078 0; H 0.934 -0.539 0; H -0.934 -0.539 0',
     '6-31G*', 'b3lyp', 0, 1),
]
out = {}
for name, atom, basis, xc, chg, spin in CASES:
    if a.only and a.only != name:
        continue
    mol = gto.M(atom=atom, basis=basis, charge=chg, spin=spin, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc)
    mf.conv_tol = 1e-11
    mf.direct_scf_tol = 1e-14
    mf.verbose = 0
    t = time.perf_counter()
    e = mf.kernel()
    t_scf = time.perf_counter() - t
    g = mf.nuc_grad_method(); g.verbose = 0
    t = time.perf_counter()
    de = g.kernel()
    t_g = time.perf_counter() - t
    out[name] = dict(e=float(e), de=np.asarray(de).tolist(),
                     conv=bool(mf.converged), t_scf=t_scf, t_grad=t_g)
    print(f'{name:20s} E={e:.10f} |de|={np.linalg.norm(de):.10f} '
          f'conv={mf.converged} scf={t_scf:.2f}s grad={t_g:.2f}s',
          file=sys.stderr, flush=True)
print(json.dumps(out))
