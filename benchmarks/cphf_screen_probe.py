"""How loose may the CPHF's integral screening be?  (catalogue B12 (b))

The B12 entry proposes inexact-Krylov relaxation: screen the two-electron
integrals of the CPHF's response builds more loosely than the SCF's
``direct_scf_tol``, since the solve only needs ``conv_tol_cpscf``.  This runs
one SCF, then the analytic Hessian once per ``(screen, conv_tol_cpscf)`` pair,
with ``screen`` the threshold the J and K builds inside ``solve_mo1`` use
(everything outside the CPHF keeps ``direct_scf_tol``), and reports the time
and the change of the Hessian and of the harmonic frequencies against the
first pair, which is the reference.

    python cphf_screen_probe.py --xyz molecules/penicillinV.xyz \
        --basis def2-svp --patch all --runs 1e-13:1e-9,1e-13:1e-6,1e-10:1e-6

``FASTCPHF_SCREEN`` is set to ``off``, so ``fastcphf``'s own rule does not
override the rows.
"""
import os
import time

import numpy as np

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--runs', default='1e-13:1e-9,1e-10:1e-9,1e-13:1e-6,'
                                  '1e-11:1e-6,1e-10:1e-6,1e-9:1e-6,1e-8:1e-6,'
                                  '1e-7:1e-6')
a = ap.parse_args()
# the probe sets the screen itself; fastcphf's own rule would override the
# 1e-13 rows
os.environ['FASTCPHF_SCREEN'] = 'off'
apply_patch(a.patch)

import cupy as cp                                            # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.scf import jk as JK, j_engine as JE           # noqa: E402
from gpu4pyscf.hessian import rhf as HR                      # noqa: E402
from pyscf.hessian import thermo                             # noqa: E402

SCREEN = [None]
_gk, _gj = JK._VHFOpt.get_k, JE._VHFOpt.get_j


def _with_screen(fn):
    def wrapped(self, *x, **k):
        if SCREEN[0] is None:
            return fn(self, *x, **k)
        keep = self.direct_scf_tol
        self.direct_scf_tol = SCREEN[0]
        try:
            return fn(self, *x, **k)
        finally:
            self.direct_scf_tol = keep
    return wrapped


JK._VHFOpt.get_k = _with_screen(_gk)
JE._VHFOpt.get_j = _with_screen(_gj)
JK._VHFOpt.get_jk = _with_screen(JK._VHFOpt.get_jk)
from gpu4pyscf.hessian import uhf as HU                      # noqa: E402
T = {}
CUR = [None]


def _timed(fn):
    # the module-level solve_mo1 both Hessian classes call at call time
    def solve_mo1(*x, **k):
        SCREEN[0] = CUR[0]
        t = time.perf_counter()
        try:
            return fn(*x, **k)
        finally:
            SCREEN[0] = None
            cp.cuda.Stream.null.synchronize()
            T['solve_mo1'] = time.perf_counter() - t
    return solve_mo1


HR.solve_mo1 = _timed(HR.solve_mo1)
HU.solve_mo1 = _timed(HU.solve_mo1)

mol = build_mol(a)
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc)
mf.grids.level = a.grids
mf.conv_tol = 1e-10
mf.kernel()
print(f'# {a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc} natm={mol.natm} '
      f'nao={mol.nao}  direct_scf_tol={mf.direct_scf_tol:g}', flush=True)
print(f'{"screen":>8s} {"cpscf":>8s} {"hess/s":>8s} {"mo1/s":>8s} '
      f'{"max|dH|":>10s} {"max|dfreq|":>11s} {"dH(scr)":>10s} '
      f'{"dfreq(scr)":>11s}')
ref = f_ref = None
base = {}          # conv_tol_cpscf -> the first run at that tolerance
for run in a.runs.split(','):
    scr, tol = (float(v) for v in run.split(':'))
    h = mf.Hessian()
    h.verbose = 0
    h.conv_tol_cpscf = tol
    mf.conv_tol_cpscf = tol
    CUR[0] = scr          # applies inside solve_mo1 only
    t = time.perf_counter()
    hess = cp.asnumpy(cp.asarray(h.kernel()))
    th = time.perf_counter() - t
    f = np.real_if_close(thermo.harmonic_analysis(mol, hess)['freq_wavenumber'])
    if ref is None:
        ref, f_ref = hess, f
    base.setdefault(tol, (hess, f))
    dh = np.abs(hess - ref).max()
    df = np.abs(f - f_ref).max()
    # the screening's own change: against the same conv_tol_cpscf
    dhs = np.abs(hess - base[tol][0]).max()
    dfs = np.abs(f - base[tol][1]).max()
    print(f'{scr:8.0e} {tol:8.0e} {th:8.2f} {T["solve_mo1"]:8.2f} '
          f'{dh:10.2e} {df:11.4f} {dhs:10.2e} {dfs:11.4f}', flush=True)
