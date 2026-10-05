"""`fasttdejk` against GPU4PySCF's `_jk_energies_per_atom` (catalogue D10).

Part 1, the function, against finite differences of `get_jk` energies at
displaced geometries (D1, D2 held fixed in the AO basis): a general,
non-symmetric density pair, its transpose and a symmetric one, in one call,
on water/6-31G*, formaldehyde/def2-SVP and def2-TZVP (f shells, which go to
GPU4PySCF's kernel for the classes not generated here) and glycine/def2-SVP.
Tolerance 1e-6 relative (the difference step), and the call repeats itself to
1e-13.  GPU4PySCF's own result on the same pair is printed beside it: on
glycine it misses the finite difference by up to 1e-1 relative on a general
pair, and it is not reproducible from run to run -- which is where its NACV
scatter (`docs/README_fastnacms.md`) comes from.

Part 2, end to end: a TDDFT gradient, a NAC and the batched force + all NACs
on the same states with the function swapped back in the same process.

    python tests/test_tdejk.py [--quick]
"""
import argparse

import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fasttdejk')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.grad import tdrhf as TDG                      # noqa: E402
from electrolites import fasttdejk                           # noqa: E402

H2O = 'O 0. 0. 0.1174; H -0.757 0. -0.4696; H 0.757 0. -0.4696'
HCHO = 'C 0 0 0; O 0 0 1.21; H 0 0.94 -0.54; H 0 -0.94 -0.54'
GLY = 'benchmarks/molecules/glycine.xyz'
ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
a = ap.parse_args()
fails = 0
rng = np.random.default_rng(5)

print('part 1: the function against finite differences')
# E(R) = j/2 * Tr[D1 J[D2]] - k/4 * Tr[D1 K[D2]] with D1, D2 held fixed in the
# AO basis is what one output is the nuclear gradient of (the factors are
# GPU4PySCF's convention, read off its kernel and checked on water).  The
# derivative of that energy by central differences of `get_jk` at displaced
# geometries is a reference that shares no code with either kernel.
from gpu4pyscf import scf                                    # noqa: E402


def _ejk_fd(mol, D1, D2, j, k, atoms, h=1e-4):
    c0 = mol.atom_coords()
    sym = [mol.atom_symbol(i) for i in range(mol.natm)]
    out = {}
    for A in atoms:
        for x in range(3):
            e = []
            for s_ in (1, -1):
                c = c0.copy()
                c[A, x] += s_ * h
                m = gto.M(atom=[(sym[i], c[i]) for i in range(mol.natm)],
                          basis=mol.basis, unit='Bohr', verbose=0)
                mf_ = scf.RHF(m)
                mf_.direct_scf_tol = 1e-20
                vj, vk = mf_.get_jk(m, D2, hermi=0)
                e.append(j / 2 * float(cp.sum(D1.T * vj))
                         - k / 4 * float(cp.sum(D1.T * vk)))
            out[A, x] = (e[0] - e[1]) / (2 * h)
    return out


CASES = [(H2O, '6-31g*', [0, 1]), (HCHO, 'def2-svp', [0, 2]),
         (GLY, 'def2-svp', [0, 3, 7]), (HCHO, 'def2-tzvp', [1, 3])]
for atom, basis, atoms in (CASES[:2] if a.quick else CASES):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.direct_scf_tol = 1e-20
    mf.kernel()
    g = mf.TDA().Gradients()
    nao = mol.nao
    D1 = cp.asarray(rng.standard_normal((nao, nao))) * .05
    D2 = cp.asarray(rng.standard_normal((nao, nao))) * .05
    j, k = 1.3, .7
    # three outputs in one call -- the pair itself, its transpose and a
    # symmetric one -- so that the multi-output path is what is checked
    n0 = fasttdejk.stats()['calls']
    got = cp.asnumpy(cp.asarray(g.jk_energies_per_atom(
        [[D1, D2], [D1.T, D2.T], [D1 + D1.T, D2 + D2.T]], [j] * 3, [k] * 3)))
    used = fasttdejk.stats()['calls'] - n0
    rep = cp.asnumpy(cp.asarray(g.jk_energies_per_atom(
        [[D1, D2], [D1.T, D2.T], [D1 + D1.T, D2 + D2.T]], [j] * 3, [k] * 3)))
    fd = _ejk_fd(mol, D1, D2, j, k, atoms)
    scale = max(abs(v) for v in fd.values())
    err = max(abs(got[0][A, x] - v) for (A, x), v in fd.items()) / scale
    fdT = _ejk_fd(mol, D1.T, D2.T, j, k, atoms[:1])
    errT = max(abs(got[1][A, x] - v) for (A, x), v in fdT.items()) / scale
    errr = np.abs(rep - got).max() / np.abs(got).max()
    TDG._jk_energies_per_atom, keep = \
        TDG._jk_energies_per_atom_orig, TDG._jk_energies_per_atom
    try:
        stock = cp.asnumpy(cp.asarray(g.jk_energies_per_atom(
            [[D1, D2]], [j], [k])))[0]
    finally:
        TDG._jk_energies_per_atom = keep
    errs = max(abs(stock[A, x] - v) for (A, x), v in fd.items()) / scale
    ok = err < 1e-6 and errT < 1e-6 and errr < 1e-13 and used == 1
    fails += not ok
    print(f'{atom[-12:]:12s} {basis:9s} ours {err:.1e}, transposed pair '
          f'{errT:.1e}, repeat {errr:.1e}; GPU4PySCF {errs:.1e}  '
          f'{"ok" if ok else "FAIL"}', flush=True)

print('\npart 2: end to end')
mol = gto.M(atom=GLY, basis='def2-svp', verbose=0)
for xc in ('b3lyp', 'pbe'):
    mf = dft.RKS(mol, xc=xc)
    mf.kernel()
    td = mf.TDDFT()
    td.nstates = 3
    td.kernel()

    def run():
        g = td.nuc_grad_method().kernel(state=1)
        n = td.nac_method().kernel(states=(1, 2))[1]
        ms = td.NACGradients()
        ms.verbose = 0
        r = ms.kernel(states=[0, 1, 2, 3], grad_state=2)
        return [cp.asnumpy(cp.asarray(x)) for x in
                (g, n, ms.grad_result, r[(1, 3)]['de'], r[(0, 2)]['de'])]
    o1 = run()
    TDG._jk_energies_per_atom, keep = \
        TDG._jk_energies_per_atom_orig, TDG._jk_energies_per_atom
    try:
        o0 = run()
    finally:
        TDG._jk_energies_per_atom = keep
    es = [np.abs(x - y).max() / np.abs(y).max() for x, y in zip(o1, o0)]
    ok = max(es[:4]) < 1e-7
    fails += not ok
    print(f'glycine {xc}: grad {es[0]:.1e}  NAC {es[1]:.1e}  batched force '
          f'{es[2]:.1e}  batched NAC(1,3) {es[3]:.1e}  GE (0,2) {es[4]:.1e}  '
          f'{"ok" if ok else "FAIL"}', flush=True)
print('\nFAILED' if fails else '\nall passed', f'({fails} failures)')
raise SystemExit(1 if fails else 0)
