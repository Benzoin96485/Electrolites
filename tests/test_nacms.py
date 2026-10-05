"""`fastnacms` against GPU4PySCF's serial `force_and_nacv`, one process.

Same SCF and TDDFT states; the force and every NACV of the patched method
(the batched multi-state solver) against the serial loop.  The two solve the
same Z-vector equations, so they agree to the accuracy those equations are
solved to -- and GPU4PySCF's Krylov solver stops at a residual of about
3e-7 whatever ``tol`` says, because it drops new directions whose squared
norm is below ``DSOLVE_LINDEP = 1e-13`` (``lib/cupy_helper.py:34, :718``).
With that floor the two differ by up to 1e-5 relative, and *which* pairs
differ depends on what else is in the batch.  So the test lowers the floor
(``lindep = 1e-22``) and sets ``tol`` to 1e-10.

Second, GPU4PySCF's ground-to-excited NACVs are not reproducible with
themselves: on glycine/HF two serial ``nac.kernel(states=(0, 1))`` calls on
the same TDHF states differ by up to 1.1e-4 (of a vector whose largest
element is 0.25), and the batched solver scatters by the same amount.  So the
serial path is run twice, and the batched result must agree with it to
three times that scatter or 1e-6 of the largest element, whichever is
larger -- per vector, with the same sign.  Then the declines: a triplet, an
open-shell reference and a sparse pair list must take the serial path.

    python tests/test_nacms.py
"""
import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fastnacms')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft, scf                               # noqa: E402
from gpu4pyscf.tdscf import rhf as TDR                       # noqa: E402
from electrolites import fastnacms                           # noqa: E402
from gpu4pyscf.lib import cupy_helper as CH                  # noqa: E402
import inspect                                               # noqa: E402

# the Krylov solver's floor, lowered for this test (see the docstring)
_names = list(inspect.signature(CH.krylov).parameters)
_d = list(CH.krylov.__defaults__)
_off = len(_names) - len(_d)
_d[_names.index('lindep') - _off] = 1e-22
_d[_names.index('max_cycle') - _off] = 300
CH.krylov.__defaults__ = tuple(_d)

H2O = 'O 0. 0. 0.1174; H -0.757 0. -0.4696; H 0.757 0. -0.4696'
HCHO = 'C 0 0 0; O 0 0 1.21; H 0 0.94 -0.54; H 0 -0.94 -0.54'
GLYCINE = 'benchmarks/molecules/glycine.xyz'
CASES = [  # atom, basis, xc, tda, density fit, grad_state
    (H2O, 'def2-svp', 'b3lyp', False, False, 1),
    (HCHO, 'def2-svp', 'pbe', True, False, 2),
    (HCHO, '6-31g*', 'hf', False, False, 1),
    (GLYCINE, 'def2-svp', 'b3lyp', False, False, 1),
    (HCHO, 'def2-svp', 'b3lyp', False, True, 0),
]
N = lambda x: np.asarray(cp.asnumpy(cp.asarray(x)), dtype=float)
fails = 0


def run(atom, basis, xc, spin=0, df=False):
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    if xc == 'hf':
        mf = (scf.UHF if spin else scf.RHF)(mol)
    else:
        mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc)
    if df:
        mf = mf.density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    return mf


for atom, basis, xc, tda, df, gs in CASES:
    mf = run(atom, basis, xc, df=df)
    td = mf.TDA() if tda else (mf.TDHF() if xc == 'hf' else mf.TDDFT())
    td.nstates = 3
    td.conv_tol = 1e-10
    td.kernel()

    def objs():
        g = td.nuc_grad_method(); g.cphf_conv_tol = 1e-10
        g.cphf_max_cycle = 300
        n = td.nac_method(); n.cphf_conv_tol = 1e-10
        n.cphf_max_cycle = 300
        return g, n
    n0 = fastnacms.stats()['batched']
    f1, v1 = td.force_and_nacv(gs, None, *objs())
    used = fastnacms.stats()['batched'] - n0
    f0, v0 = TDR.TDBase._force_and_nacv_orig(td, gs, None, *objs())
    f0b, v0b = TDR.TDBase._force_and_nacv_orig(td, gs, None, *objs())
    rel = lambda a, b, ref: np.abs(N(a) - N(b)).max() / np.abs(N(ref)).max()
    ef = rel(f1, f0, f0)
    ef0 = rel(f0b, f0, f0)
    ev = ev0 = 0.
    ok = used == 1 and set(v1) == set(v0) and ef <= max(1e-6, 3 * ef0)
    for k in v0:
        for a, b, c in zip(v1[k], v0[k], v0b[k]):
            e, e0 = rel(a, b, b), rel(c, b, b)
            ev, ev0 = max(ev, e), max(ev0, e0)
            ok &= e <= max(1e-6, 3 * e0)
    fails += not ok
    print(f'{str(atom)[:10]:10s} {basis:9s} {xc:6s} {"TDA " if tda else "RPA "}'
          f'{"DF " if df else "   "}state {gs}: force {ef:.1e} ({ef0:.1e})  '
          f'{len(v0)} NACVs {ev:.1e} (serial vs serial {ev0:.1e})  '
          f'{"ok" if ok else "FAIL"}', flush=True)

# declines
mf = run(HCHO, 'def2-svp', 'b3lyp')
td = mf.TDDFT(); td.nstates = 3; td.singlet = False; td.kernel()
s0 = fastnacms.stats()['serial']
try:
    td.force_and_nacv(1, [(1, 2)])
except Exception as exc:                                     # noqa: BLE001
    print(f'(triplet serial path raised {type(exc).__name__}, as upstream)')
td = mf.TDDFT(); td.nstates = 3; td.kernel()
td.force_and_nacv(1, [(1, 2), (0, 1)])                       # not all pairs
ok = fastnacms.stats()['serial'] - s0 == 2
fails += not ok
print(f'declines: {fastnacms.stats()["declined"]}  {"ok" if ok else "FAIL"}')
print('\nFAILED' if fails else '\nall passed', f'({fails} failures)')
raise SystemExit(1 if fails else 0)
