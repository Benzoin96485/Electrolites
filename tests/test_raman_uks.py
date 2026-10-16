"""Raman activities for UHF/UKS on the GPU (``electrolites.raman_uks``).

RHF/RKS against GPU4PySCF's own ``properties.raman`` (a 19-point field
stencil against 13 here, same Hessian; its field SCFs stop at the SCF's own
tolerance, so it is the noisier of the two: 1e-3); a closed-shell UKS
against the RKS result (frequencies 0.1 cm^-1 apart: GPU4PySCF's UKS and
RKS Hessians); and for H2O+ (UHF, UKS B3LYP) the polarizability derivatives
against central differences of the analytic polarizability
(``polar_gpu``) over nuclear displacements.  PySCF has no Raman code.

    python tests/test_raman_uks.py
"""
import sys

import numpy as np
from pyscf import gto
from gpu4pyscf import scf, dft
from gpu4pyscf.properties import raman as graman

from electrolites import raman_uks
from electrolites.polar_gpu import polarizability

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


water = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
              basis='def2-svp', verbose=0)
mf = dft.RKS(water, xc='b3lyp').run(conv_tol=1e-12)
h = mf.Hessian().kernel()
f0, a0, d0 = graman.eval_raman_intensity(mf.copy(), h)
f1, a1, d1 = raman_uks.eval_raman_intensity(mf, h)
ea = abs(np.asarray(a0) - a1).max() / abs(np.asarray(a0)).max()
ed = abs(np.asarray(d0) - d1)[np.asarray(a0) > 1e-3 * np.max(a0)].max()
report('water RKS B3LYP ~ GPU4PySCF raman', ea < 5e-3 and ed < 5e-3,
       f'activity rel {ea:.1e}, depolarisation {ed:.1e}')
mfu = dft.UKS(water, xc='b3lyp').run(conv_tol=1e-12)
f2, a2, d2 = raman_uks.eval_raman_intensity(mfu, mfu.Hessian().kernel())
ef = abs(np.asarray(f1) - f2).max()
ea = abs(a1 - a2).max() / abs(a1).max()
report('water closed-shell UKS == RKS', ef < 0.5 and ea < 1e-4,
       f'freq {ef:.1e} cm-1, activity rel {ea:.1e}')

cation = gto.M(atom='O 0 0 0; H 0 0 0.97; H 0.9 0 -0.3', charge=1, spin=1,
               basis='def2-svp', verbose=0)
for label, mk in (('H2O+ UHF', scf.UHF),
                  ('H2O+ UKS B3LYP', lambda m: dft.UKS(m, xc='b3lyp'))):
    mf = mk(cation).run(conv_tol=1e-12)
    da = raman_uks.polarizability_derivative(mf)
    fd = np.zeros_like(da)
    dx = 1e-3
    for a in range(cation.natm):
        for x in range(3):
            al = []
            for sg in (1, -1):
                c = cation.atom_coords()
                c[a, x] += sg * dx
                m = cation.set_geom_(c, unit='Bohr', inplace=False)
                al.append(polarizability(mk(m).run(conv_tol=1e-12),
                                         tol=1e-11))
            fd[a, x] = (al[0] - al[1]) / (2 * dx)
    e = abs(da - fd).max() / abs(fd).max()
    report(f'{label} d alpha/dR against displaced alpha', e < 1e-3,
           f'rel {e:.1e} (max {abs(fd).max():.2f})')
    freq, act, dep = raman_uks.eval_raman_intensity(mf)
    report(f'{label} activities finite', np.isfinite(act).all(),
           f'{np.round(act[-3:], 3)}')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
