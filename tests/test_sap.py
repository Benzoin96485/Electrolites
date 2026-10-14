"""GPU SAP initial guess (`electrolites/sap.py`) against PySCF 2.13.

V_sap element-wise against PySCF's ``make_sap``; the guess density against
PySCF's ``init_guess_by_sap`` (RHF, UHF, ROHF); and the SCF from it landing
on the minao energy (RKS, UKS, a transition-metal complex).

    python tests/test_sap.py
"""
import sys

import numpy as np
import cupy as cp

import electrolites
electrolites.enable('sap')
from electrolites import sap                                  # noqa: E402

from pyscf import gto, scf as cscf                            # noqa: E402
from pyscf.scf import hf as hf_cpu                            # noqa: E402
from gpu4pyscf import scf, dft                                # noqa: E402

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:46s} {detail}', flush=True)


water = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
              basis='def2-svp', verbose=0)
cation = gto.M(atom='O 0 0 0.1; H 0 0.76 -0.45; H 0 -0.76 -0.45',
               basis='def2-svp', charge=1, spin=1, verbose=0)

sapbas = sap._sap_basis(scf.RHF(water), water)
v = sap.make_sap(water, sapbas).get()
ref = hf_cpu.make_sap(water, sapbas)
err = abs(v - ref).max()
report('V_sap vs PySCF make_sap', err < 1e-10, f'{err:.1e}')

for label, g, c, mol in (('RHF', scf.RHF, cscf.RHF, water),
                         ('UHF', scf.UHF, cscf.UHF, cation),
                         ('ROHF', scf.ROHF, cscf.ROHF, cation)):
    dg = cp.asnumpy(g(mol).init_guess_by_sap())
    dc = np.asarray(c(mol).init_guess_by_sap())
    err = abs(dg - dc).max()
    report(f'{label} guess density vs PySCF', err < 1e-8, f'{err:.1e}')

ni = gto.M(atom='benchmarks/molecules/tmc_ni_dithiolene.xyz', basis='def2-svp',
           charge=-1, spin=1, verbose=0)
for label, mk, mol in (('RKS water', lambda m: dft.RKS(m, xc='b3lyp'), water),
                       ('UKS H2O+', lambda m: dft.UKS(m, xc='b3lyp'), cation),
                       ('UKS Ni dithiolene-', lambda m: dft.UKS(m, xc='b3lyp'),
                        ni)):
    a = mk(mol).set(init_guess='sap', conv_tol=1e-10, max_cycle=100).run()
    b = mk(mol).set(conv_tol=1e-10, max_cycle=100).run()
    report(f'{label}: SAP SCF energy = minao SCF energy',
           a.converged and abs(a.e_tot - b.e_tot) < 1e-7,
           f'{abs(a.e_tot - b.e_tot):.1e}; cycles {a.cycles} vs {b.cycles}')

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
