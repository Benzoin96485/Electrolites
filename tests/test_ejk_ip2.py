"""Correctness of the lifted second-derivative ERI kernels (catalogue B3).

`fastejk2` replaces the two CUDA entry points `hessian/rhf.py`'s
`_partial_ejk_ip2` drives, and changes nothing above them.  So the oracle is
**the same driver running GPU4PySCF's kernels**: one process, one converged
density, one set of task lists, the two kernel sets swapped underneath.  That
is stronger than comparing two processes, because it holds the screening and
the task ordering fixed and leaves only the arithmetic.

Two criteria, because `_partial_ejk_ip2` returns a `natm x natm x 3 x 3` array
whose elements span many orders of magnitude:

* the largest element-wise difference, relative to the largest element of the
  reference -- the criterion the catalogue's §6 asks for on an integral kernel;
* the whole **analytic Hessian** built on top of it, patched against unpatched,
  since that is what a user sees and it folds in the CPHF solve.

The expected agreement is round-off and not bit for bit: one `rsqrt(s)` now
supplies `1/s` where upstream divides, and `rsqrt(s)*rsqrt(s)` differs from
`1/s` by about two units in the last place.

    python tests/test_ejk_ip2.py
    python tests/test_ejk_ip2.py --basis def2-tzvp --no-hessian
"""
import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='',
                help='restrict to one basis instead of the built-in sweep')
ap.add_argument('--tol', type=float, default=1e-11,
                help='largest allowed element-wise difference, relative to '
                     'the largest element of the reference')
ap.add_argument('--hess-tol', type=float, default=1e-8,
                help='largest allowed difference on the whole Hessian')
ap.add_argument('--no-hessian', action='store_true',
                help='skip the whole-Hessian rows, which dominate the runtime')
ap.add_argument('--xc', default='b3lyp')
a = ap.parse_args()

import cupy as cp                                             # noqa: E402
from pyscf import gto                                         # noqa: E402
import electrolites.fastejk2 as fastejk2                      # noqa: E402
from gpu4pyscf import dft                                     # noqa: E402
from gpu4pyscf.hessian import rhf as hrhf                      # noqa: E402
from gpu4pyscf.scf import jk as JK                            # noqa: E402

H2O = '''O 0.0 0.0 0.117; H 0.0 0.757 -0.467; H 0.0 -0.757 -0.467'''
ETHANOL = '''
C  -0.748  -0.015   0.024
C   0.558   0.420  -0.679
O   1.598  -0.513  -0.421
H  -1.293  -0.849  -0.435
H  -1.393   0.878   0.086
H  -0.581  -0.323   1.061
H   0.384   0.442  -1.766
H   0.874   1.415  -0.352
H   2.437  -0.166  -0.775
'''
SI = '''Si 0.0 0.0 0.0; H 0.0 0.0 1.48; H 1.40 0.0 -0.48; H -0.70 1.21 -0.48;
        H -0.70 -1.21 -0.48'''

CASES = [('H2O', H2O, '6-31G*'),
         ('H2O', H2O, 'def2-TZVP'),
         ('ethanol', ETHANOL, '6-31G*'),
         ('SiH4', SI, '6-31G*'),
         ('ethanol', ETHANOL, 'def2-SVP')]
if a.basis:
    seen, uniq = set(), []
    for n, g, _ in CASES:
        if n not in seen:
            seen.add(n)
            uniq.append((n, g, a.basis))
    CASES = uniq


def main():
    if not fastejk2.apply_patch(force=True):
        print('fastejk2 could not patch (no nvcc, or an untested GPU4PySCF); '
              'nothing to test')
        return 0
    import gpu4pyscf
    print(f'# gpu4pyscf {gpu4pyscf.__version__}  fastejk2 patched')
    fails = 0
    worst = worst_h = 0.
    for name, geom, basis in CASES:
        mol = gto.M(atom=geom, basis=basis, verbose=0)
        mf = dft.RKS(mol, xc=a.xc)
        mf.grids.level = 3
        mf.conv_tol = 1e-11
        mf.kernel()
        opt = JK._VHFOpt(mol, mf.direct_scf_tol).build()
        dm0 = cp.asarray(mf.make_rdm1())

        fastejk2.unpatch()
        ref = cp.asarray(hrhf._partial_ejk_ip2(mol, dm0, opt, 1.0, 1.0))
        fastejk2.apply_patch(force=True)
        got = cp.asarray(hrhf._partial_ejk_ip2(mol, dm0, opt, 1.0, 1.0))
        scale = float(cp.abs(ref).max()) or 1.
        rel = float(cp.abs(got - ref).max()) / scale
        ok = rel <= a.tol
        worst = max(worst, rel)
        fails += not ok

        hrel = float('nan')
        if not a.no_hessian:
            h = mf.Hessian()
            h.auxbasis_response = 0
            fastejk2.unpatch()
            h0 = cp.asarray(h.kernel())
            fastejk2.apply_patch(force=True)
            h1 = cp.asarray(mf.Hessian().kernel())
            hrel = float(cp.abs(h1 - h0).max())
            hok = hrel <= a.hess_tol
            worst_h = max(worst_h, hrel)
            fails += not hok
            ok = ok and hok

        print(f'{"ok " if ok else "FAIL"} {name:9s} {basis:10s} '
              f'lmax={int(mol._bas[:, 1].max())}  ejk rel={rel:.2e}  '
              f'hessian max|d|={hrel:.2e}', flush=True)
    print(f'\nworst: ejk {worst:.3e} relative (tol {a.tol:.0e}), '
          f'hessian {worst_h:.3e} absolute (tol {a.hess_tol:.0e})')
    if fails:
        print(f'{fails} check(s) FAILED')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
