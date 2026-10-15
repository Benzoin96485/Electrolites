"""IR intensities for UHF/UKS on the GPU (``electrolites.ir_uks``,
FEATURE_TARGETS O3): RHF/RKS against GPU4PySCF's own driver, UHF/UKS
against pyscf-properties' ``infrared.uhf/uks`` on the same SCF.

The UHF/UKS dipole derivatives sit 6e-6 (UHF) / 5e-5 (UKS) from
pyscf-properties': GPU4PySCF's Hessian CPHF stops near a 1e-6 relative
residual (its Krylov solver's absolute stop; see ``nmr_uks.krylov_refined``),
while pyscf-properties' is 3e-7 from a finite difference of the dipole.
That is 1e-5 relative on the intensities.

    python tests/test_ir_uks.py
"""
import sys

import numpy as np
import cupy as cp
from pyscf import gto
from pyscf.prop.infrared import uhf as pir_uhf, uks as pir_uks
from gpu4pyscf import scf, dft
from gpu4pyscf.properties import ir as gir

from electrolites import ir_uks

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


w = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
          basis='def2-svp', verbose=0)
for label, mk in (('water RHF', scf.RHF),
                  ('water B3LYP', lambda m: dft.RKS(m, xc='b3lyp'))):
    mf = mk(w).run(conv_tol=1e-12)
    f0, i0 = gir.eval_ir_freq_intensity(mf, mf.Hessian())
    f1, i1 = ir_uks.eval_ir_freq_intensity(mf, mf.Hessian())
    ef = abs(cp.asnumpy(f0) - f1).max()
    ei = abs(cp.asnumpy(i0) - i1).max() / abs(cp.asnumpy(i0)).max()
    report(f'{label} == GPU4PySCF driver', ef < 1e-6 and ei < 1e-8,
           f'freq {ef:.1e} cm-1, intensity rel {ei:.1e}')
cation = gto.M(atom='O 0 0 0; H 0 0 0.97; H 0.9 0 -0.3', charge=1, spin=1,
               basis='def2-svp', verbose=0)
for label, mk, ref in (('H2O+ UHF', scf.UHF, pir_uhf),
                       ('H2O+ UKS B3LYP', lambda m: dft.UKS(m, xc='b3lyp'),
                        pir_uks)):
    mf = mk(cation).run(conv_tol=1e-12)
    f1, i1 = ir_uks.eval_ir_freq_intensity(mf, mf.Hessian())
    apt = ir_uks.dipole_derivatives(mf, mf.Hessian())
    mc = mf.to_cpu()
    if hasattr(mc, 'grids'):
        mc.grids.build()
    p = ref.Infrared(mc).run()
    ea = abs(apt.transpose(2, 1, 0) - p.de).max() / abs(p.de).max()
    nv = len(p.ir_inten)
    ef = abs(f1[-nv:] - p.vib_dict['freq_wavenumber']).max()
    ei = abs(i1[-nv:] - p.ir_inten).max() / abs(p.ir_inten).max()
    report(f'{label} against pyscf-properties',
           ea < 1e-4 and ef < 0.1 and ei < 1e-4,
           f'APT rel {ea:.1e}, freq {ef:.1e} cm-1, intensity rel {ei:.1e}')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
