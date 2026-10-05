"""`fastvv10ft` against GPU4PySCF's `VXC_vv10nlc_hess_eval_f_t`, one process.

Part 1, the kernel: on the ground-state arrays every caller hands it
(`fastfnlc._ground_state`), both implementations on the same trial
densities, element-wise, at trial counts that are and are not multiples of
the tile (1, 5, 8, 13, 70) and with all three NLC functionals, a third-row
element and a coarse grid.  Twice: with every pair in fp64 (tolerance 1e-13
of the largest element) and with the default single-precision far field
(1e-9).  Then `EUW_with_weight1`, the grid-weight-derivative sums, the same two
ways (1e-13 and 1e-8: its terms are not damped by a density derivative, so
the far field carries more of it).

Part 2, end to end: the symbol is swapped back to GPU4PySCF's inside the same
process and a wB97M-V analytic Hessian -- whose three VV10 terms are the
kernel's three callers in `hessian/rks.py` -- is computed both ways, closed
shell and open shell; then a polarizability.  Tolerance 1e-6 relative on the
Hessian, because GPU4PySCF's own Hessian is not reproducible below that: two
stock runs in one process differ by up to 5.6e-7 relative on water (the
test prints that scatter beside the difference); 1e-8 on the polarizability.

    python tests/test_vv10ft.py [--quick]
"""
import argparse
import ctypes

import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fastfnlc', 'fastvv10ft')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.dft import numint                             # noqa: E402
from gpu4pyscf.properties import polarizability as gpol      # noqa: E402
from electrolites import fastfnlc, fastvv10ft                # noqa: E402

H2O = 'O 0. 0. 0.; H 0. -0.757 0.587; H 0. 0.757 0.587'
CH3OH = '''C 0.667 0.000 0.000; O -0.667 0.000 0.000
H 1.100 1.000 0.100; H 1.100 -0.500 -0.870; H 1.100 -0.500 0.870
H -1.000 0.900 0.000'''
HCL = 'H 0. 0. 0.; Cl 0. 0. 1.28'
CH3 = 'C 0 0 0; H 1.08 0 0; H -0.54 0.935 0; H -0.54 -0.935 0'
GLYCINE = 'benchmarks/molecules/glycine.xyz'

KCASES = [
    (H2O, 'def2-svp', 'wb97m-v', 3),
    (CH3OH, '6-31g*', 'b97m-v', 3),
    (HCL, 'def2-svp', 'wb97x-v', 3),
    (GLYCINE, 'def2-svp', 'wb97m-v', 3),
    (CH3OH, 'def2-svp', 'wb97m-v', 1),
]
NTRIAL = (1, 5, 8, 13, 70)

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
a = ap.parse_args()

lib = numint.libgdft
ours = lib.VXC_vv10nlc_hess_eval_f_t
stock = fastvv10ft.stock()
assert ours is not stock
P = lambda x: ctypes.cast(x.data.ptr, ctypes.c_void_p)
NAMES = ['grids_coords', 'grids_weights', 'rho_i', 'omega_i', 'kappa_i',
         'U_i', 'W_i', 'A_i', 'B_i', 'C_i', 'domega_drho_i',
         'domega_dgamma_i', 'dkappa_drho_i', 'd2omega_drho2_i',
         'd2omega_dgamma2_i', 'd2omega_drho_dgamma_i', 'd2kappa_drho2_i']


def scf(atom, basis, xc, level, spin=0):
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0,
                max_memory=40000)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc)
    mf.grids.level = level
    mf.nlcgrids.level = level
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    return mf


fails = 0
print('part 1: the kernel, element-wise')
RC = fastvv10ft.RCUT
print(f'fp64 everywhere / fp32 beyond {RC} Bohr (the default), at ntrial =')
print(f'{"case":36s} {"ng":>7s} ' + ' '.join(f'{n:>19d}' for n in NTRIAL))
rng = np.random.default_rng(3)
for atom, basis, xc, level in (KCASES[:2] if a.quick else KCASES):
    mf = scf(atom, basis, xc, level)
    ni = mf._numint
    grids = mf.nlcgrids
    if getattr(ni, 'gdftopt', None) is None:
        ni.build(mf.mol, grids.coords)
    gs = fastfnlc._ground_state(mf, cp.asarray(mf.mo_coeff),
                                cp.asarray(mf.mo_occ), ni, ni.gdftopt, grids)
    ng = gs['ngrids']
    ins_arr = [cp.ascontiguousarray(gs[k]) for k in NAMES]
    ins = [P(x) for x in ins_arr]
    errs = []
    for nt in NTRIAL:
        rho_t = cp.ascontiguousarray(
            gs['rho_i'] * cp.asarray(rng.uniform(-1, 1, (nt, ng))))
        gamma_t = cp.ascontiguousarray(
            gs['gamma_i'] * cp.asarray(rng.uniform(-1, 1, (nt, ng))))
        out = []
        for fn in (stock, ours, ours):
            fastvv10ft.RCUT = float('inf') if len(out) == 1 else RC
            fr = cp.full((nt, ng), np.nan)
            fg = cp.full((nt, ng), np.nan)
            fn(ctypes.cast(cp.cuda.get_current_stream().ptr, ctypes.c_void_p),
               P(fr), P(fg), *ins, P(rho_t), P(gamma_t), ctypes.c_int(ng),
               ctypes.c_int(nt))
            out.append((fr, fg))
        fastvv10ft.RCUT = RC
        (fr0, fg0) = out[0]
        for (fr1, fg1), tol in zip(out[1:], (1e-13, 1e-9)):
            e = max(float(cp.abs(fr1 - fr0).max() / cp.abs(fr0).max()),
                    float(cp.abs(fg1 - fg0).max() / cp.abs(fg0).max()))
            errs.append(e)
            if not e < tol:
                fails += 1
    # EUW_with_weight1, on random weight derivatives of the weights' size
    eus = []
    for nd in (3, 13, 70):
        w1 = cp.ascontiguousarray(gs['grids_weights'] * cp.asarray(
            rng.uniform(-1, 1, (nd, ng))))
        c_ = P(ins_arr[0]); r_ = P(gs['rho_i']); o_ = P(ins_arr[3])
        k_ = P(ins_arr[4])
        outs = []
        for fn, rc in ((fastvv10ft.stock_euw(), RC),
                       (lib.VXC_vv10nlc_hess_eval_EUW_with_weight1, np.inf),
                       (lib.VXC_vv10nlc_hess_eval_EUW_with_weight1, RC)):
            fastvv10ft.RCUT = rc
            o3 = [cp.full((nd, ng), np.nan) for _ in range(3)]
            fn(ctypes.cast(cp.cuda.get_current_stream().ptr, ctypes.c_void_p),
               *[P(x) for x in o3], c_, P(w1), r_, o_, k_, ctypes.c_int(ng),
               ctypes.c_int(nd))
            outs.append(o3)
        fastvv10ft.RCUT = RC
        for got, tol in zip(outs[1:], (1e-13, 1e-8)):
            e = max(float(cp.abs(b - a).max() / cp.abs(a).max())
                    for a, b in zip(outs[0], got))
            eus.append(e)
            if not e < tol:
                fails += 1
    tag = f'{str(atom)[-12:].strip()}/{basis}/{xc}/L{level}'
    print(f'{tag:36s} {ng:7d} ' + ' '.join(
        f'{errs[k]:9.1e}/{errs[k+1]:9.1e}' for k in range(0, len(errs), 2)),
        flush=True)
    print(f'{"   EUW_with_weight1, nder 3/13/70":44s} ' + ' '.join(
        f'{eus[k]:9.1e}/{eus[k+1]:9.1e}' for k in range(0, len(eus), 2)),
        flush=True)


def with_stock(fn):
    lib.VXC_vv10nlc_hess_eval_f_t = stock
    try:
        return fn()
    finally:
        lib.VXC_vv10nlc_hess_eval_f_t = ours


print('\npart 2: end to end, the symbol swapped in the same process')
E2E = [(H2O, 0, 'wb97m-v'), (CH3, 1, 'wb97m-v')]
if not a.quick:
    E2E.append((CH3OH, 0, 'wb97x-v'))
for atom, spin, xc in E2E:
    mf = scf(atom, 'def2-svp', xc, 2, spin)

    def hess():
        h = mf.Hessian()
        h.conv_tol = 1e-10
        return cp.asnumpy(cp.asarray(h.kernel()))
    fastfnlc._CACHE.clear()
    n0 = fastvv10ft.stats()['ours']
    h1 = hess()
    used = fastvv10ft.stats()['ours'] - n0
    fastfnlc._CACHE.clear()
    h0 = with_stock(hess)
    fastfnlc._CACHE.clear()
    h0b = with_stock(hess)
    e = np.abs(h1 - h0).max() / np.abs(h0).max()
    e0 = np.abs(h0b - h0).max() / np.abs(h0).max()
    ok = e < 1e-6 and used >= 3
    fails += not ok
    print(f'Hessian {"UKS" if spin else "RKS"} {xc:8s} {str(atom)[:14]:14s} '
          f'rel {e:.2e}  (stock against itself {e0:.2e}; {used} calls of '
          f'ours)  {"ok" if ok else "FAIL"}', flush=True)
    if not spin:
        fastfnlc._CACHE.clear()
        p1 = cp.asnumpy(gpol.eval_polarizability(mf))
        fastfnlc._CACHE.clear()
        p0 = with_stock(lambda: cp.asnumpy(gpol.eval_polarizability(mf)))
        e = np.abs(p1 - p0).max() / np.abs(p0).max()
        fails += not e < 1e-8
        print(f'polarizability {xc:8s} rel {e:.2e}  '
              f'{"ok" if e < 1e-8 else "FAIL"}', flush=True)

print('\nFAILED' if fails else '\nall passed', f'({fails} failures)')
raise SystemExit(1 if fails else 0)
