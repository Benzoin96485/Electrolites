"""The direct Hessian's h1 two-electron term against GPU4PySCF's (B15).

``electrolites/fastjkip1.py`` binds ``libmyjkip1``'s ``RYS_build_jk_ip1`` --
GPU4PySCF's driver with the device query cached, and its unrolled kernels
with the fp64 divisions folded (P1) -- over GPU4PySCF's.  ``_get_jk_ip1``
with ours against ``_get_jk_ip1`` with GPU4PySCF's (``fastjkip1.unpatch``),
same molecule, same density, element-wise relative to the largest element:
J and K together, J alone, K alone, an atom slice, the long-range operator,
an open-shell density, and a basis with f functions (the classes the
unrolled kernels do not cover go to the general kernel).  Then the spin path
(B15 (b)): `get_jk_ip1_spins` against two GPU4PySCF calls, J summed, for J+K,
J only, K only and the long-range operator; and whole UKS Hessians (B3LYP,
wB97X for the long-range pass) against GPU4PySCF's ``make_h1``.

    python tests/test_jkip1.py
"""
import os
import sys

import numpy as np

import electrolites
electrolites.patch_all()
import cupy as cp                                            # noqa: E402
from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.hessian import rhf as HR                      # noqa: E402
from electrolites import fastjkip1                           # noqa: E402

assert fastjkip1.apply_patch(), 'libmyjkip1 not available'
fails = []
HERE = os.path.dirname(os.path.abspath(__file__))
ETHANOL = '''C -0.748 -0.015 0.024; C 0.558 0.420 -0.679; O 1.598 -0.513 -0.421;
H -1.593 0.365 -0.554; H -0.834 0.355 1.048; H -0.780 -1.106 0.023;
H 0.640 1.502 -0.580; H 0.567 0.174 -1.741; H 2.421 -0.190 -0.792'''
CUCL2 = os.path.join(HERE, '..', 'benchmarks', 'molecules', 'tmc_cucl2.xyz')


def both(mol, dm, **kw):
    ours = HR._get_jk_ip1(mol, dm, **kw)
    fastjkip1.unpatch()
    try:
        ref = HR._get_jk_ip1(mol, dm, **kw)
    finally:
        fastjkip1.apply_patch()
    return ours, ref


def check(tag, ours, ref, tol=1e-12):
    err = 0.
    for a, b in zip(ours, ref):
        if a is None and b is None:
            continue
        a, b = cp.asnumpy(a), cp.asnumpy(b)
        err = max(err, np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
    ok = err <= tol
    if not ok:
        fails.append(f'{tag}: {err:.3e}')
    print(f'  {"ok  " if ok else "FAIL"} {tag:44s} {err:.3e}', flush=True)


#: The spin path screens one call on all three densities (the absmax over
#: total, alpha and beta), where two GPU4PySCF calls screen each spin on its
#: own; the kept quartets are a superset, so the two differ by what the
#: per-spin screens drop -- up to 1.5e-10 relative on these molecules.
TOL_SPINS = 1e-9


for tag, atom, basis, charge, spin in (
        ('ethanol def2-svp', ETHANOL, 'def2-svp', 0, 0),
        ('ethanol+ def2-svp', ETHANOL, 'def2-svp', 1, 1),
        ('CuCl2 def2-tzvp (f)', CUCL2, 'def2-tzvp', 0, 1)):
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc='b3lyp')
    mf.max_cycle = 5
    mf.kernel()
    dm = cp.asarray(mf.make_rdm1())
    dm = dm[0] if dm.ndim == 3 else dm
    check(f'{tag} J+K', *both(mol, dm))
    check(f'{tag} J only', *both(mol, dm, with_k=False))
    check(f'{tag} K only', *both(mol, dm, with_j=False))
    check(f'{tag} atoms [1:3]', *both(mol, dm, atoms_slice=(1, 3)))
    with mol.with_range_coulomb(0.3):
        check(f'{tag} long range (omega 0.3)', *both(mol, dm, with_j=False))
print('## the spin path')
from gpu4pyscf.hessian import uks as HUKS                     # noqa: E402
for tag, atom, basis, charge, spin in (
        ('ethanol+ def2-svp', ETHANOL, 'def2-svp', 1, 1),
        ('CuCl2 def2-tzvp (f)', CUCL2, 'def2-tzvp', 0, 1)):
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = dft.UKS(mol, xc='b3lyp')
    mf.max_cycle = 5
    mf.kernel()
    dma, dmb = cp.asarray(mf.make_rdm1())
    for kw, lbl in (({}, 'J+K'), ({'with_k': False}, 'J only'),
                    ({'with_j': False}, 'K only')):
        vj, vka, vkb = fastjkip1.get_jk_ip1_spins(mol, dma, dmb, **kw)
        fastjkip1.unpatch()
        try:
            ja, ka = HR._get_jk_ip1(mol, dma, **kw)
            jb, kb = HR._get_jk_ip1(mol, dmb, **kw)
        finally:
            fastjkip1.apply_patch()
        check(f'{tag} spins {lbl}',
              [vj, vka, vkb],
              [None if ja is None else ja + jb, ka, kb], TOL_SPINS)
    with mol.with_range_coulomb(0.3):
        _, vka, vkb = fastjkip1.get_jk_ip1_spins(mol, dma, dmb, with_j=False)
        fastjkip1.unpatch()
        try:
            ka = HR._get_jk_ip1(mol, dma, with_j=False)[1]
            kb = HR._get_jk_ip1(mol, dmb, with_j=False)[1]
        finally:
            fastjkip1.apply_patch()
        check(f'{tag} spins long range', [vka, vkb], [ka, kb], TOL_SPINS)

for tag, xc in (('ethanol+ UKS B3LYP Hessian', 'b3lyp'),
                ('ethanol+ UKS wB97X Hessian', 'wb97x')):
    mol = gto.M(atom=ETHANOL, basis='def2-svp', charge=1, spin=1, verbose=0)
    mf = dft.UKS(mol, xc=xc)
    mf.conv_tol = 1e-10
    mf.kernel()
    s0 = fastjkip1.stats()['spin_calls']
    ours = cp.asnumpy(cp.asarray(mf.Hessian().kernel()))
    used = fastjkip1.stats()['spin_calls'] > s0
    fastjkip1.unpatch()
    try:
        ref = cp.asnumpy(cp.asarray(mf.Hessian().kernel()))
    finally:
        fastjkip1.apply_patch()
    err = np.abs(ours - ref).max()
    ok = err <= 1e-8 and used
    if not ok:
        fails.append(f'{tag}: {err:.3e} spin path used {used}')
    print(f'  {"ok  " if ok else "FAIL"} {tag:44s} {err:.3e} Eh/bohr^2',
          flush=True)
if fails:
    print('\nFAILED:\n  ' + '\n  '.join(fails))
    sys.exit(1)
print('\nall ok')
