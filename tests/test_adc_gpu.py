"""ADC(2) on the GPU (``electrolites.adc_gpu``) against PySCF's ADC on the
host from the same SCF: EE (singlets), IP and EA; also with frozen core.

    python tests/test_adc_gpu.py
"""
import sys

import numpy as np
from pyscf import gto, adc
from gpu4pyscf import scf

from electrolites import adc_gpu

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


for name, atom, basis, frozen in (
        ('water', 'O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
         'cc-pvdz', None),
        ('glycine fc', 'benchmarks/molecules/glycine.xyz', 'def2-svp', 5)):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    mfc = mf.to_cpu()
    for kind in ('ee', 'ip', 'ea'):
        ref = adc.ADC(mfc, frozen=frozen)
        ref.method_type = kind
        ref.compute_properties = False
        ref.approx_trans_moments = True
        ref.conv_tol = 1e-10
        ref.tol_residual = 1e-7
        e0 = ref.kernel(nroots=4)[0]
        new = adc_gpu.RADC(mf, frozen=frozen, method_type=kind)
        new.conv_tol = 1e-10
        new.tol_residual = 1e-7
        e1 = new.kernel(nroots=4)[0]
        d = abs(np.asarray(e0) - np.asarray(e1)).max()
        de = abs(ref.e_corr - new.e_corr)
        report(f'{name} ADC(2)-{kind.upper()} 4 roots == PySCF',
               d < 1e-7 and de < 1e-9,
               f'{d:.1e} Eh, MP2 {de:.1e} ({np.round(np.asarray(e1) * 27.2114, 3)} eV)')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
