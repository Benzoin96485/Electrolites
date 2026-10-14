"""DF-MP2 relaxed density and gradient on the GPU (``electrolites.dfmp2_grad``,
FEATURE_TARGETS P3) against PySCF and finite differences.

PySCF has no DF-MP2 gradient, so the gradient's oracle is a central finite
difference (Richardson-extrapolated over steps 4e-3 and 2e-3 Bohr) of the
GPU DF-MP2 total energy itself, on a DF-RHF reference converged to 1e-10 in
the orbital gradient.  The relaxed density is compared with PySCF's
``dfmp2_native`` on the same exact-integral RHF orbitals.

    python tests/test_dfmp2_grad.py
"""
import sys

import numpy as np
import cupy as cp

from pyscf import gto, df
from pyscf.mp import dfmp2_native
from gpu4pyscf import scf
from gpu4pyscf.mp import dfmp2

from electrolites import dfmp2_grad

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:55s} {detail}', flush=True)


WATER = [['O', [0., 0., 0.2]], ['H', [0., 1.5, -0.9]], ['H', [0.1, -1.4, -1.0]]]
H2CO = [['C', [0., 0., -1.05]], ['O', [0., 0.1, 1.24]],
        ['H', [0., 1.77, -2.1]], ['H', [0.2, -1.75, -2.15]]]

print('three-index derivative contraction')
mol = gto.M(atom=WATER, unit='Bohr', basis='cc-pvdz', verbose=0)
aux = df.make_auxmol(mol, 'cc-pvdz-ri')
rng = np.random.default_rng(3)
cl = rng.normal(size=(mol.nao, 4))
cr = rng.normal(size=(mol.nao, 6))
gam = rng.normal(size=(aux.nao, 4, 6))
gt = rng.normal(size=(aux.nao, aux.nao))
got = dfmp2_grad.nonsep_grad(mol, aux, cl, cr, gam, gt)
ip3 = df.incore.aux_e2(mol, aux, 'int3c2e_ip1', comp=3)
ip3p = df.incore.aux_e2(mol, aux, 'int3c2e_ip2', comp=3)
ip2 = aux.intor('int2c2e_ip1')
G = np.einsum('mi,Pij,nj->mnP', cl, gam, cr)
ref = np.zeros((mol.natm, 3))
for A, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
    q0, q1 = aux.aoslice_by_atom()[A][2:]
    ref[A] = -2 * (np.einsum('xmnP,mnP->x', ip3[:, p0:p1],
                             (G + G.transpose(1, 0, 2))[p0:p1])
                   + np.einsum('xmnP,mnP->x', ip3p[..., q0:q1], G[..., q0:q1]))
    ref[A] += np.einsum('xPQ,PQ->x', ip2[:, q0:q1], (gt + gt.T)[q0:q1])
err = abs(got - ref).max() / abs(ref).max()
report('(ij|P)^x Gamma and (P|Q)^x Gt vs PySCF ip integrals', err < 1e-12,
       f'{err:.1e}')


def run(atom, basis, auxbasis, jkfit, frozen, grad=False):
    mol = gto.M(atom=atom, basis=basis, unit='Bohr', verbose=0)
    mf = scf.RHF(mol)
    if jkfit:
        mf = mf.density_fit(auxbasis=jkfit)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    mp = dfmp2.DFMP2(mf, auxbasis=auxbasis, frozen=frozen or None)
    if grad:
        return mp, dfmp2_grad.Gradients(mp).kernel()
    mp.kernel()
    return mp.e_hf + mp.e_corr


def fd_check(label, atom, basis, auxbasis, jkfit, frozen, comps):
    _, an = run(atom, basis, auxbasis, jkfit, frozen, grad=True)
    worst = 0.
    for A, x in comps:
        f = []
        for h in (4e-3, 2e-3):
            ap = [[s, list(c)] for s, c in atom]
            am = [[s, list(c)] for s, c in atom]
            ap[A][1][x] += h
            am[A][1][x] -= h
            f.append((run(ap, basis, auxbasis, jkfit, frozen)
                      - run(am, basis, auxbasis, jkfit, frozen)) / (2 * h))
        worst = max(worst, abs(an[A, x] - (4 * f[1] - f[0]) / 3))
    report(label, worst < 2e-9, f'{worst:.1e}')


print('gradient against finite differences (DF-RHF reference)')
allc = [(A, x) for A in range(3) for x in range(3)]
fd_check('water cc-pVDZ, all 9 components', WATER, 'cc-pvdz', 'cc-pvdz-ri',
         'cc-pvdz-jkfit', 0, allc)
fd_check('water cc-pVDZ, frozen core, all 9 components', WATER, 'cc-pvdz',
         'cc-pvdz-ri', 'cc-pvdz-jkfit', 1, allc)
fd_check('formaldehyde def2-SVP, frozen core, 4 components', H2CO,
         'def2-svp', 'def2-svp-ri', 'def2-universal-jkfit', 2,
         [(0, 2), (1, 1), (2, 1), (3, 2)])

print('relaxed density against PySCF dfmp2_native (exact RHF)')
for frozen in (0, 1):
    mol = gto.M(atom=WATER, basis='cc-pvdz', unit='Bohr', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    mp = dfmp2.DFMP2(mf, auxbasis='cc-pvdz-ri', frozen=frozen or None)
    got = cp.asnumpy(dfmp2_grad.make_rdm1_relaxed(mp, ao_repr=True))
    pt = dfmp2_native.DFRMP2(mf.to_cpu(), frozen=frozen or None,
                             auxbasis='cc-pvdz-ri')
    pt.cphf_tol = 1e-12
    pt.cphf_max_cycle = 200
    ref = pt.make_rdm1(relaxed=True, ao_repr=True)
    err = abs(got - ref).max()
    # PySCF's cphf.solve leaves ~1e-7 in the Z-vector at this tolerance
    report(f'relaxed rdm1, frozen={frozen}', err < 1e-6, f'{err:.1e}')

print('the feature binding, and what it refuses')
import electrolites                                             # noqa: E402
electrolites.enable('dfmp2_grad')
mp, g = run(WATER, 'cc-pvdz', 'cc-pvdz-ri', 'cc-pvdz-jkfit', 0, grad=True)
g2 = mp.Gradients().kernel()
report('DFMP2.Gradients().kernel() == dfmp2_grad.Gradients',
       abs(g - g2).max() < 1e-9, f'{abs(g - g2).max():.1e}')
mol = gto.M(atom=WATER, basis='cc-pvdz', unit='Bohr', verbose=0)
try:
    from gpu4pyscf import dft
    dfmp2.DFMP2(dft.RKS(mol, xc='wb97x').run()).Gradients().kernel()
    report('range-separated RKS reference raises', False)
except NotImplementedError:
    report('range-separated RKS reference raises', True)

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
