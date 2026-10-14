"""The correctness guards (`electrolites/guards.py`, FEATURE_TARGETS §3.2).

Each guarded entry point must either raise with a reason (ROHF/ROKS Hessian
and TDA, QM/MM Hessian) or reach a *correct* result (the ROHF/ROKS gradient
through PySCF on the CPU), and the guards must not touch the cases next to
them: the UHF Hessian, the QM/MM gradient, ``undo_qmmm().Hessian()``.

    python tests/test_guards.py
"""
import sys
import warnings

import numpy as np

import electrolites
electrolites.patch('guards')

from pyscf import gto, scf as cscf, dft as cdft          # noqa: E402
from gpu4pyscf import scf, dft                           # noqa: E402
from gpu4pyscf.qmmm import itrf                          # noqa: E402

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


def raises(label, fn, exc=NotImplementedError, needle=''):
    try:
        fn()
    except exc as e:
        report(label, needle in str(e), f'{type(e).__name__}')
        return
    except Exception as e:                               # noqa: BLE001
        report(label, False, f'wrong exception {type(e).__name__}: {e}')
        return
    report(label, False, 'did not raise')


cation = gto.M(atom='O 0 0 0.1; H 0 0.76 -0.45; H 0 -0.76 -0.45',
               basis='6-31g', charge=1, spin=1, verbose=0)

print('ROHF / ROKS')
for name, gcls, ccls in (
        ('ROHF', lambda m: scf.ROHF(m), lambda m: cscf.ROHF(m)),
        ('ROKS', lambda m: dft.ROKS(m, xc='pbe'),
         lambda m: cdft.ROKS(m, xc='pbe')),
        ('DF-ROHF', lambda m: scf.ROHF(m).density_fit(),
         lambda m: cscf.ROHF(m).density_fit())):
    mf = gcls(cation).run(conv_tol=1e-11)
    raises(f'{name}.Hessian() raises', lambda: mf.Hessian(), needle='ROHF')
    if not name.startswith('DF'):
        raises(f'{name}.TDA() raises', lambda: mf.TDA())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        g = mf.Gradients().kernel()
    warned = any('cpu_fallback' in str(x.message) for x in w)
    ref = ccls(cation).run(conv_tol=1e-11).Gradients().kernel()
    err = abs(np.asarray(g) - ref).max()
    report(f'{name}.Gradients() via CPU, warned', warned and err < 1e-6,
           f'{err:.1e}')

print('untouched neighbours')
uhf = scf.UHF(cation).run(conv_tol=1e-11)
h = uhf.Hessian().kernel()
report('UHF.Hessian() still runs', h.shape == (3, 3, 3, 3))
rhf = scf.RHF(gto.M(atom='O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
                    basis='6-31g', verbose=0)).density_fit().run()
report('DF-RHF.Gradients() still GPU',
       'gpu4pyscf' in type(rhf.Gradients()).__module__)

print('QM/MM')
water = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
              basis='6-31g', verbose=0)
qm = itrf.add_mm_charges(dft.RKS(water, xc='pbe'),
                         [[3., .5, .2], [.3, 3., -.4]], [.8, -.8],
                         unit='Bohr').run()
raises('QMMM Hessian() raises', lambda: qm.Hessian(), needle='QM/MM')
report('QMMM Gradients() still runs',
       qm.Gradients().kernel().shape == (3, 3))
report('undo_qmmm().Hessian() still runs',
       qm.undo_qmmm().Hessian().kernel().shape == (3, 3, 3, 3))

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
