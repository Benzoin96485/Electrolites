"""Does a pair-atomic fit give the right energy?  W6's first question.

`DF_WORK_LIST.md` **W6** says it in as many words: pair-atomic fitting "is a
different approximation, not a faster route to the same number", and anything
built for it "must be benchmarked on *energies* against global DF, not only on
wall clocks".  This is that benchmark, and it is deliberately a **reference**
implementation on small molecules rather than anything that could be fast:
dense `(nao^2, naux)` fitting coefficients, an explicit four-index tensor, and
an SCF driven through it.  Nothing here belongs on a GPU or in
`electrolites/`; its only job is to say whether the approximation is worth
implementing properly.

Four expansions are compared at the same geometry and basis:

    exact       the four-centre integrals
    global      ordinary density fitting -- every pair fitted in every
                auxiliary function, which is what `fastdf` contracts
    pari        pair-atomic, **non-robust**: each AO pair fitted only in the
                auxiliary functions of its own two atoms, and the integral
                approximated as `c J c`
    pari-r      pair-atomic, **robust** (Dunlap): `(mn|Q)c + c(P|ls) - c J c`,
                whose error is second order in the fitting residual where the
                non-robust one's is first order

and `--domains` widens the domain from the two atoms of the pair to every
atom within a cutoff, which is the knob between the two.

What matters is not that `pari` is worse than `global` -- it must be -- but
by how much, and whether the error is a constant per atom or grows.  The
exchange matrix is reported separately from the energy because it is where
pair-atomic fitting is known to be weakest.

`c1`..`c6` are an idealised alkane homologous series, which is what says
whether the error is a constant per atom or grows with the molecule -- the
question that decides whether the approximation is usable at the sizes
`DF_WORK_LIST.md` §3.2 is written around.

    python benchmarks/df_local_accuracy.py
    python benchmarks/df_local_accuracy.py --cases c1,c2,c3,c4 --domains pair
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--domains', default='pair,pair+3.0')
ap.add_argument('--cases', default='H2O,CH4,NH3BH3,ethanol',
                help='comma separated names from the table below')
ap.add_argument('--xc', default='', help='empty for HF, else a DFT functional')
ap.add_argument('--max-nao', type=int, default=110,
                help='refuse a case above this; the four-index tensor is '
                     'nao^4 and this script is a reference, not a method')
a = ap.parse_args()

from pyscf import gto, scf, dft, df, lib                     # noqa: E402
from pyscf.df import addons                                  # noqa: E402

GEOM = {
    'H2O': 'O 0.0 0.0 0.117; H 0.0 0.757 -0.467; H 0.0 -0.757 -0.467',
    'CH4': ('C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; '
            'H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629'),
    # a dative bond and a long H...H contact: the pair-atomic expansion has to
    # describe a product density whose two atoms are not bonded in the usual
    # sense, which is where PARI is reported to struggle
    'NH3BH3': ('N 0 0 0.7; B 0 0 -0.9; H 0.94 0 1.06; H -0.47 0.81 1.06; '
               'H -0.47 -0.81 1.06; H 0.98 0 -1.25; H -0.49 0.85 -1.25; '
               'H -0.49 -0.85 -1.25'),
    'ethanol': ('C -0.748 -0.015 0.024; C 0.558 0.420 -0.679; '
                'O 1.598 -0.513 -0.421; H -1.293 -0.849 -0.435; '
                'H -1.393 0.878 0.086; H -0.581 -0.323 1.061; '
                'H 0.384 0.442 -1.766; H 0.874 1.415 -0.352; '
                'H 2.437 -0.166 -0.775'),
}


from _common import alkane                                # noqa: E402

for _n in range(1, 7):
    GEOM[f'c{_n}'] = alkane(_n)


def _atom_of(m):
    out = np.empty(m.nao, dtype=np.int32)
    for at, (_, _, p0, p1) in enumerate(m.aoslice_by_atom()):
        out[p0:p1] = at
    return out


def fit_coefficients(mol, aux, int3c, j2c, spec):
    """``c[m,n,P]``: each AO pair fitted only inside its own domain.

    Solved per *atom* pair rather than per AO pair, because every AO pair of
    one atom pair shares a domain and therefore a metric -- one Cholesky per
    atom pair instead of one per AO pair, which is the same saving a real
    implementation would take.
    """
    nao, naux = mol.nao, aux.nao
    a2a, x2a = _atom_of(mol), _atom_of(aux)
    sl = mol.aoslice_by_atom()
    if spec == 'pair':
        near = np.eye(mol.natm, dtype=bool)
    else:
        r = float(spec.split('+')[1])
        d = np.linalg.norm(mol.atom_coords()[:, None] -
                           mol.atom_coords()[None], axis=-1) * lib.param.BOHR
        near = d <= r
    c = np.zeros((nao, nao, naux))
    for A in range(mol.natm):
        for B in range(mol.natm):
            dom = np.where(near[A] | near[B])[0]
            idx = np.where(np.isin(x2a, dom))[0]
            i0, i1, j0, j1 = sl[A, 2], sl[A, 3], sl[B, 2], sl[B, 3]
            rhs = int3c[i0:i1, j0:j1][:, :, idx].reshape(-1, len(idx))
            sol = np.linalg.solve(j2c[np.ix_(idx, idx)], rhs.T).T
            c[i0:i1, j0:j1][:, :, idx] = sol.reshape(i1 - i0, j1 - j0,
                                                     len(idx))
    return c


def eri_from_fit(int3c, j2c, c, robust):
    nao = c.shape[0]
    C = c.reshape(nao * nao, -1)
    I = int3c.reshape(nao * nao, -1)
    T = C @ j2c
    out = -(T @ C.T) if robust else (T @ C.T)
    if robust:
        out += I @ C.T
        out += C @ I.T
    return out.reshape(nao, nao, nao, nao)


def scf_energy(mol, eri, xc):
    """An SCF driven through an explicit four-index tensor."""
    mf = (dft.RKS(mol, xc=xc) if xc else scf.RHF(mol))
    mf.verbose = 0
    mf.conv_tol = 1e-11
    if xc:
        mf.grids.level = 3

    def get_jk(mol_=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        dm = np.asarray(dm)
        vj = np.einsum('ijkl,kl->ij', eri, dm) if with_j else None
        vk = np.einsum('ikjl,kl->ij', eri, dm) if with_k else None
        return vj, vk
    mf.get_jk = get_jk
    return mf.kernel(), mf.make_rdm1()


print(f'{"molecule":10s} {"nao":>4s} {"naux":>5s} {"domain":>9s} '
      f'{"variant":>8s} {"E":>18s} {"dE vs exact":>13s} {"dE vs DF":>12s} '
      f'{"per atom":>11s} {"relK":>9s}')
rows = []
for name in a.cases.split(','):
    mol = gto.M(atom=GEOM[name], basis=a.basis, verbose=0)
    if mol.nao > a.max_nao:
        print(f'{name:10s} skipped: nao {mol.nao} > --max-nao {a.max_nao}')
        continue
    aux = addons.make_auxmol(mol, a.auxbasis)
    int3c = df.incore.aux_e2(mol, aux, intor='int3c2e', aosym='s1')
    j2c = aux.intor('int2c2e')
    nao = mol.nao

    eri_exact = mol.intor('int2e', aosym='s1').reshape([nao] * 4)
    e_exact, dm_exact = scf_energy(mol, eri_exact, a.xc)
    # global DF, in the same four-index form so that every row differs only
    # in the expansion and not in the SCF driver
    c_glob = np.linalg.solve(j2c, int3c.reshape(-1, aux.nao).T).T.reshape(
        nao, nao, aux.nao)
    eri_df = eri_from_fit(int3c, j2c, c_glob, robust=False)
    e_df, _ = scf_energy(mol, eri_df, a.xc)

    def _k(eri):
        return np.einsum('ikjl,kl->ij', eri, dm_exact)
    k_exact = _k(eri_exact)
    kscale = np.abs(k_exact).max()

    for label, e, eri in (('exact', e_exact, eri_exact),
                          ('global', e_df, eri_df)):
        print(f'{name:10s} {nao:4d} {aux.nao:5d} {"-":>9s} {label:>8s} '
              f'{e:18.10f} {e-e_exact:13.2e} {e-e_df:12.2e} '
              f'{(e-e_df)/mol.natm:11.2e} '
              f'{np.abs(_k(eri)-k_exact).max()/kscale:9.2e}')
    for spec in a.domains.split(','):
        c = fit_coefficients(mol, aux, int3c, j2c, spec)
        for label, robust in (('pari', False), ('pari-r', True)):
            eri = eri_from_fit(int3c, j2c, c, robust)
            try:
                e, _ = scf_energy(mol, eri, a.xc)
            except Exception as exc:                          # noqa: BLE001
                print(f'{name:10s} {nao:4d} {aux.nao:5d} {spec:>9s} '
                      f'{label:>8s}  SCF failed: {type(exc).__name__}')
                continue
            relk = np.abs(_k(eri) - k_exact).max() / kscale
            print(f'{name:10s} {nao:4d} {aux.nao:5d} {spec:>9s} {label:>8s} '
                  f'{e:18.10f} {e-e_exact:13.2e} {e-e_df:12.2e} '
                  f'{(e-e_df)/mol.natm:11.2e} {relk:9.2e}')
            rows.append(dict(name=name, nao=int(nao), naux=int(aux.nao),
                             natm=int(mol.natm), domain=spec, variant=label,
                             e=float(e), de_exact=float(e - e_exact),
                             de_df=float(e - e_df), relk=float(relk)))
print('RESULT ' + json.dumps(dict(basis=a.basis, xc=a.xc, rows=rows)))
