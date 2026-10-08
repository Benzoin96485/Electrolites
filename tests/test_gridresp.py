"""The grid-response Hessian's fused weight second derivative (B9).

``electrolites/fastgridresp.py`` contracts the Becke weights' second
derivative with its vector without storing it.  Checked against
``contract('ABdDg,g->ABdD', get_d2weight_dAdB(...), eps)`` on the same grid
block for both partition schemes, with and without the Treutler radii
adjustment, on a block that spans several atoms; then whole
``grid_response=True`` Hessians (RKS B3LYP, UKS PBE, RKS wB97M-V for the VV10
path) against ``FASTGRIDRESP_OFF=1`` run in the same process through the
``*_orig`` functions.

    python tests/test_gridresp.py
"""
import sys

import numpy as np

import electrolites
electrolites.patch_all()
import cupy as cp                                            # noqa: E402
from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.dft import radi                               # noqa: E402
from gpu4pyscf.hessian import rks as HR, uks as HU           # noqa: E402
from gpu4pyscf.lib.cupy_helper import contract               # noqa: E402
from electrolites import fastgridresp                        # noqa: E402

fails = []
ETHANOL = '''C -0.748 -0.015 0.024; C 0.558 0.420 -0.679; O 1.598 -0.513 -0.421;
H -1.593 0.365 -0.554; H -0.834 0.355 1.048; H -0.780 -1.106 0.023;
H 0.640 1.502 -0.580; H 0.567 0.174 -1.741; H 2.421 -0.190 -0.792'''


def check(label, got, ref, tol):
    got, ref = cp.asnumpy(got), cp.asnumpy(ref)
    err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-300)
    ok = err <= tol
    if not ok:
        fails.append(f'{label}: {err:.3e}')
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {err:.3e}', flush=True)


mol = gto.M(atom=ETHANOL, basis='def2-svp', verbose=0)
print('## one block, against get_d2weight_dAdB + contract')
for scheme in ('stratmann', 'becke'):
    for adjust in (radi.treutler_atomic_radii_adjust, None):
        mf = dft.RKS(mol, xc='b3lyp')
        if scheme == 'becke':
            from gpu4pyscf.dft.gen_grid import original_becke
            mf.grids.becke_scheme = original_becke
        else:
            from gpu4pyscf.dft.gen_grid import stratmann
            mf.grids.becke_scheme = stratmann
        mf.grids.radii_adjust = adjust
        mf.grids.build()
        g = mf.grids
        smol = mol
        ng = g.coords.shape[0]
        for g0, g1 in ((0, 4096), (ng // 2 - 3000, ng // 2 + 3001)):
            eps = cp.asarray(np.random.default_rng(g0).normal(size=g1 - g0))
            ref = contract('ABdDg,g->ABdD',
                           HR.get_d2weight_dAdB(smol, g, (g0, g1)), eps)
            got = fastgridresp.d2w_contract(smol, g, (g0, g1), eps)
            check(f'{scheme} adjust={adjust is not None} [{g0}:{g1}]',
                  got, ref, 1e-12)

print('## whole grid-response Hessians, against the replaced functions')
for tag, m, xc, spin in (('ethanol RKS B3LYP', ETHANOL, 'b3lyp', 0),
                         ('ethanol+ UKS PBE', ETHANOL, 'pbe', 1),
                         ('ethanol RKS wB97M-V (VV10)', ETHANOL, 'wb97m-v', 0)):
    mol = gto.M(atom=m, basis='def2-svp', charge=spin, spin=spin, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc)
    mf.conv_tol = 1e-10
    mf.kernel()
    h = mf.Hessian()
    h.grid_response = True
    ours = cp.asnumpy(cp.asarray(h.kernel()))
    mods = (HU,) if spin else (HR,)
    saved = []
    for mod in mods:
        for name in ('_get_exc_deriv2_grid_response', '_get_enlc_deriv2'):
            if hasattr(mod, name + '_orig'):
                saved.append((mod, name, getattr(mod, name)))
                setattr(mod, name, getattr(mod, name + '_orig'))
    try:
        h = mf.Hessian()
        h.grid_response = True
        ref = cp.asnumpy(cp.asarray(h.kernel()))
    finally:
        for mod, name, f in saved:
            setattr(mod, name, f)
    err = np.abs(ours - ref).max()
    ok = err <= 1e-9
    if not ok:
        fails.append(f'{tag}: {err:.3e}')
    print(f'  {"ok  " if ok else "FAIL"} {tag + " Hessian":52s} {err:.3e} '
          f'Eh/bohr^2', flush=True)
print('# fastgridresp', fastgridresp.stats())
if fails:
    print('\nFAILED:\n  ' + '\n  '.join(fails))
    sys.exit(1)
print('\nall ok')
