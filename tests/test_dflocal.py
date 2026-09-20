"""Correctness of the robust pair-atomic build.

`electrolites/dflocal.py` is the one module in this repository that does
**not** compute what GPU4PySCF computes: it is a different approximation, by
about 2.5e-5 Eh per atom (`docs/README_dflocal.md` §2.2).  So "agrees with
upstream at 1e-14", which is every other test here, is not the bar and cannot
be made the bar.  What is checked instead is that the module computes **its
own definition** -- the robust (Dunlap) pair-atomic expansion

    (mn|ls) ~ (mn|Q) c[ls,Q] + c[mn,P] (P|ls) - c[mn,P] J[P,Q] c[ls,Q]

-- assembled independently, in dense ``numpy``, from the same integrals: an
explicit four-index tensor, contracted with an explicit density.  That
reference is the one `benchmarks/df_local_accuracy.py` used to decide the
approximation in the first place, and it shares no line of code with the
module.

Five things, because one tolerance would hide four of them:

1. **The fit.**  ``T = c J`` must equal the exact three-centre integrals
   *on the domain* -- that is the fitting equation itself, and it is the one
   identity in the module that is exact rather than approximate.
2. **J and K** against the dense reference, at a random symmetric density and
   at a converged one.
3. **The folded tensor.**  ``M = B - T/2`` is what makes the exchange one
   pass instead of two, and it is checked element-wise against ``B`` and ``T``
   built separately.
4. **The energy**, against the same reference driven through a four-index
   SCF, and against global DF -- the second is a *distance*, reported rather
   than asserted, because it is the approximation and not an error.
5. **The lambda screening** (`DFLOCAL_LAMBDA_TOL`), which is an approximation
   on top of an approximation: at 1e-10 it must leave the exchange matrix
   alone to far better than the pair-atomic error it sits inside.

    python tests/test_dflocal.py
    python tests/test_dflocal.py --basis def2-tzvpp

Needs one GPU.  The reference is ``nao^4``, so the molecules are small on
purpose; `benchmarks/df_local_k_bench.py` is where the same question is asked
at 42 and 113 atoms without one.
"""
import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--tol', type=float, default=1e-9,
                help='maximum difference relative to the largest element of '
                     'the reference matrix.  Looser than the 1e-14 the rest '
                     'of tests/ uses, and §1 says why: the module applies an '
                     'explicit inverse of each domain metric where the '
                     'reference solves, and a metric of condition number 2e7 '
                     'costs about 1e-11 of the fit either way')
ap.add_argument('--domains', default='pair,pair+3.0')
a = ap.parse_args()

import cupy as cp                                            # noqa: E402
from pyscf import gto, df, scf                               # noqa: E402
from pyscf.df import addons                                  # noqa: E402
import electrolites.dflocal as dl                            # noqa: E402

MOLS = {
    'H2O': 'O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
    # a dative bond, where `docs/README_dflocal.md` §2.1 found the non-robust
    # form collapsing by 11 Eh and the robust one holding at 1.8e-4
    'NH3BH3': ('N 0 0 0.7; B 0 0 -0.9; H 0.94 0 1.06; H -0.47 0.81 1.06; '
               'H -0.47 -0.81 1.06; H 0.98 0 -1.25; H -0.49 0.85 -1.25; '
               'H -0.49 -0.85 -1.25'),
}


def _atom_of(m):
    out = np.empty(m.nao, dtype=int)
    for at, (_, _, p0, p1) in enumerate(m.aoslice_by_atom()):
        out[p0:p1] = at
    return out


def reference(mol, spec):
    """The dense pair-atomic fit, solved per atom pair, in plain numpy."""
    aux = addons.make_auxmol(mol, a.auxbasis)
    nao, naux = mol.nao, aux.nao
    B = df.incore.aux_e2(mol, aux, intor='int3c2e', aosym='s1')
    j2c = aux.intor('int2c2e')
    x2a, sl = _atom_of(aux), mol.aoslice_by_atom()
    if spec == 'pair':
        near = np.eye(mol.natm, dtype=bool)
    else:
        from pyscf import lib
        r = float(spec.split('+')[1])
        d = np.linalg.norm(mol.atom_coords()[:, None] - mol.atom_coords()[None],
                           axis=-1) * lib.param.BOHR
        near = d <= r
    c = np.zeros((nao, nao, naux))
    for A in range(mol.natm):
        for Bm in range(mol.natm):
            idx = np.where(np.isin(x2a, np.where(near[A] | near[Bm])[0]))[0]
            i0, i1, j0, j1 = sl[A, 2], sl[A, 3], sl[Bm, 2], sl[Bm, 3]
            rhs = B[i0:i1, j0:j1][:, :, idx].reshape(-1, len(idx))
            c[i0:i1, j0:j1][:, :, idx] = np.linalg.solve(
                j2c[np.ix_(idx, idx)], rhs.T).T.reshape(i1 - i0, j1 - j0,
                                                        len(idx))
    T = np.einsum('mnP,PQ->mnQ', c, j2c)
    eri = (np.einsum('mnQ,lsQ->mnls', B, c)
           + np.einsum('mnP,lsP->mnls', c, B)
           - np.einsum('mnQ,lsQ->mnls', T, c))
    return B, j2c, c, T, eri


def check(label, got, ref, tol=None, fails=None):
    tol = a.tol if tol is None else tol
    scale = max(np.abs(ref).max(), 1e-30)
    err = float(np.abs(np.asarray(got) - np.asarray(ref)).max()) / scale
    ok = err <= tol
    print(f'{"ok  " if ok else "FAIL"} {label:44s} {err:9.2e} (tol {tol:.0e})')
    if fails is not None:
        fails.append(not ok)
    return ok


def matrices():
    fails = []
    for name, geom in MOLS.items():
        mol = gto.M(atom=geom, basis=a.basis, verbose=0)
        for spec in a.domains.split(','):
            B, j2c, c, T, eri = reference(mol, spec)
            o = dl.PARIDF(mol, a.auxbasis, domain=spec)
            o.build()
            rows, cols = o.rows.get(), o.cols.get()

            # 1 -- the fitting equation, which is exact on the domain
            dom = np.abs(c) > 0
            check(f'{name}/{spec} T == B on the domain',
                  T[dom], B[dom], 1e-11, fails)

            # 3 -- the folded tensor
            M = cp.asnumpy(o.M).T if isinstance(o.M, cp.ndarray) else o.M.T
            check(f'{name}/{spec} M == B - T/2', M, (B - .5 * T)[rows, cols],
                  fails=fails)

            # 2 -- J and K
            rng = np.random.default_rng(7)
            D = rng.standard_normal((mol.nao, mol.nao))
            D = D + D.T
            vj, vk = o.get_jk(cp.asarray(D), hermi=1)
            check(f'{name}/{spec} vj', cp.asnumpy(vj),
                  np.einsum('mnls,ls->mn', eri, D), fails=fails)
            check(f'{name}/{spec} vk', cp.asnumpy(vk),
                  np.einsum('mnls,ns->ml', eri, D), fails=fails)

            # J alone and K alone must be the same matrices
            vj2 = o.get_jk(cp.asarray(D), with_k=False)[0]
            vk2 = o.get_jk(cp.asarray(D), with_j=False)[1]
            check(f'{name}/{spec} vj (J-only call)', cp.asnumpy(vj2),
                  cp.asnumpy(vj), 1e-12, fails)
            check(f'{name}/{spec} vk (K-only call)', cp.asnumpy(vk2),
                  cp.asnumpy(vk), 1e-12, fails)

            # 5 -- the lambda screening, at a threshold that must be inert
            o2 = dl.PARIDF(mol, a.auxbasis, domain=spec, lambda_tol=1e-10)
            o2.build()
            check(f'{name}/{spec} vk with DFLOCAL_LAMBDA_TOL=1e-10',
                  cp.asnumpy(o2.get_jk(cp.asarray(D), with_j=False)[1]),
                  cp.asnumpy(vk), 1e-7, fails)
            del o, o2
            cp.get_default_memory_pool().free_all_blocks()
    return sum(fails)


def energies():
    """4 -- the SCF energy, against the reference expansion and against DF."""
    from gpu4pyscf import scf as gscf
    fails = []
    for name, geom in MOLS.items():
        mol = gto.M(atom=geom, basis=a.basis, verbose=0)
        _, _, _, _, eri = reference(mol, 'pair')

        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.conv_tol = 1e-11
        mf.get_jk = lambda mol_=None, dm=None, hermi=1, with_j=True, \
            with_k=True, omega=None: (
                np.einsum('ijkl,kl->ij', eri, np.asarray(dm)),
                np.einsum('ikjl,kl->ij', eri, np.asarray(dm)))
        e_ref = mf.kernel()

        e_pari = dl.density_fit(gscf.RHF(mol)).run(conv_tol=1e-11).e_tot
        e_df = gscf.RHF(mol).density_fit().run(conv_tol=1e-11).e_tot
        e_ex = gscf.RHF(mol).run(conv_tol=1e-11).e_tot
        ok = abs(e_pari - e_ref) < 1e-8
        fails.append(not ok)
        print(f'{"ok  " if ok else "FAIL"} {name} SCF vs the dense reference '
              f'{e_pari - e_ref:+.2e} Eh')
        print(f'#    {name}: exact {e_ex:.9f}  global DF {e_df - e_ex:+.2e}  '
              f'PARI-r {e_pari - e_ex:+.2e}  PARI-r - DF '
              f'{(e_pari - e_df) / mol.natm:+.2e} Eh per atom')
        cp.get_default_memory_pool().free_all_blocks()
    return sum(fails)


def main():
    print(f'# basis {a.basis}  domains {a.domains}')
    fails = matrices()
    fails += energies()
    print(f'\ndflocal {dl.stats()}')
    print('FAILURES' if fails else 'all ok')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
