"""What a pair-atomic fit would store, and how badly conditioned it would be.

`DF_WORK_LIST.md` **W6** proposes fitting each AO pair in the auxiliary
functions on its own two atoms (PARI) rather than on all of them, so the
stored tensor goes from `npair x naux` to `npair x naux_local` with
`naux_local` independent of system size.  The item's own instruction is to
decide the approximation before writing any of it, and two things have to be
measured before that decision can be made.

**The storage curve.**  `naux` is linear in the system and `npair` grows like
N^1.5 under the screening, so the global tensor grows like N^2.5 -- 86.4 GB at
284 atoms and, measured here, 414 GB at 545, which is past the 251 GB of a
Perlmutter node.  A pair-atomic domain makes it `sum_p naux_local(p)`, which
should track `npair` alone.  This prints both and fits the exponent.

**The conditioning.**  A pair-atomic fit factorises a
`naux_local x naux_local` Coulomb metric per *atom* pair instead of one
`naux x naux` metric for the molecule, and how badly conditioned those small
ones are is the cheapest available statement about whether the approximation
is usable.  `--cond` samples pair domains and reports each local condition
number next to the global one.  Measured, the answer is the opposite of what
"smaller domains are worse conditioned" would suggest: the global metric's
condition number grows with the molecule (2.2e8 at 10 atoms, 2.3e9 at 42) and
the local ones do not (median ~2e7 throughout).

Neither needs an SCF.  `--domain` chooses how big a domain is:

    pair            the two atoms of the pair (PARI proper)
    pair+<r>        plus every atom within <r> Angstrom of either

    python benchmarks/df_local_probe.py --basis def2-svp
    python benchmarks/df_local_probe.py --cases penicillinV:0:0 --cond
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--domains', default='pair,pair+3.0,pair+5.0')
ap.add_argument('--cases', default='glycine:0:0,alanine_dipeptide:0:0,'
                                   'penicillinV:0:0,taxol:0:0,PfPMT:0:-1,'
                                   'HcgC:0:0',
                help='comma separated name:spin:charge from '
                     'benchmarks/molecules')
ap.add_argument('--cond', action='store_true',
                help='also report the condition number of the local Coulomb '
                     'metric for sampled pair domains, against the global one')
ap.add_argument('--cond-samples', type=int, default=12)
a = ap.parse_args()

from pyscf import gto                                        # noqa: E402
from pyscf.df import addons                                  # noqa: E402
from electrolites._dfblock import pair_indices, ao2atom      # noqa: E402


def aux2atom(aux):
    out = np.empty(aux.nao, dtype=np.int32)
    for at, (_, _, p0, p1) in enumerate(aux.aoslice_by_atom()):
        out[p0:p1] = at
    return out


def domain_sizes(mol, aux, pair_atoms, spec):
    """How many auxiliary functions each *atom pair* fits in."""
    naux_per_atom = np.bincount(aux2atom(aux), minlength=mol.natm)
    if spec == 'pair':
        near = None
    else:
        r = float(spec.split('+')[1])
        d = np.linalg.norm(mol.atom_coords()[:, None] -
                           mol.atom_coords()[None], axis=-1) * 0.529177210903
        near = d <= r
    # one size per *distinct* atom pair, then looked up per AO pair
    uniq, inv = np.unique(pair_atoms[0] * mol.natm + pair_atoms[1],
                          return_inverse=True)
    A, B = uniq // mol.natm, uniq % mol.natm
    if near is None:
        size = np.where(A == B, naux_per_atom[A],
                        naux_per_atom[A] + naux_per_atom[B])
    else:
        size = np.array([naux_per_atom[near[i] | near[j]].sum()
                         for i, j in zip(A, B)])
    return size[inv]


print(f'{"molecule":20s} {"natm":>5s} {"npair":>9s} {"naux":>6s} '
      f'{"global/GB":>10s} {"domain":>9s} {"nlocal":>7s} {"local/GB":>9s} '
      f'{"ratio":>7s}')
rows = []
for case in a.cases.split(','):
    name, spin, charge = (case.split(':') + ['0', '0'])[:3]
    mol = gto.M(atom=f'benchmarks/molecules/{name}.xyz', basis=a.basis,
                spin=int(spin), charge=int(charge), verbose=0)
    r, c, nao, naux = pair_indices(mol, a.auxbasis)
    aux = addons.make_auxmol(mol, a.auxbasis)
    a2a = ao2atom(mol)
    pair_atoms = (a2a[r], a2a[c])
    npair = len(r)
    g_gb = npair * naux * 8 / 1e9
    for spec in a.domains.split(','):
        sz = domain_sizes(mol, aux, pair_atoms, spec)
        l_gb = float(sz.sum()) * 8 / 1e9
        print(f'{name:20s} {mol.natm:5d} {npair:9d} {naux:6d} '
              f'{g_gb:10.2f} {spec:>9s} {sz.mean():7.0f} {l_gb:9.3f} '
              f'{g_gb/max(l_gb, 1e-12):6.0f}x')
        rows.append(dict(name=name, natm=int(mol.natm), npair=int(npair),
                         naux=int(naux), nao=int(nao), domain=spec,
                         global_gb=float(g_gb), local_gb=l_gb,
                         nlocal_mean=float(sz.mean())))

    if a.cond:
        # the global metric, then a sample of local ones.  `int2c2e` on the
        # auxiliary basis is the same matrix `_decompose_j2c` factorises.
        j2c = aux.intor('int2c2e')
        gl = np.linalg.cond(j2c)
        a2aux = aux2atom(aux)
        uniq = np.unique(pair_atoms[0] * mol.natm + pair_atoms[1])
        rng = np.random.default_rng(20260919)
        pick = rng.choice(len(uniq), min(a.cond_samples, len(uniq)),
                          replace=False)
        conds = []
        for k in uniq[pick]:
            A, B = k // mol.natm, k % mol.natm
            idx = np.where((a2aux == A) | (a2aux == B))[0]
            conds.append(np.linalg.cond(j2c[np.ix_(idx, idx)]))
        conds = np.array(conds)
        print(f'#   {name}: metric condition number -- global {gl:.2e}, '
              f'local pair domains min {conds.min():.2e} '
              f'median {np.median(conds):.2e} max {conds.max():.2e} '
              f'({len(conds)} of {len(uniq)} distinct atom pairs)')
        rows[-1]['cond_global'] = float(gl)
        rows[-1]['cond_local_median'] = float(np.median(conds))
        rows[-1]['cond_local_max'] = float(conds.max())

# the exponent, per domain, over the molecules that have one
print()
for spec in a.domains.split(','):
    sel = [x for x in rows if x['domain'] == spec]
    if len(sel) < 3:
        continue
    n = np.log([x['natm'] for x in sel])
    loc = np.polyfit(n, np.log([x['local_gb'] for x in sel]), 1)[0]
    glo = np.polyfit(n, np.log([x['global_gb'] for x in sel]), 1)[0]
    print(f'{spec:>9s}: storage ~ N^{loc:.2f}   (global ~ N^{glo:.2f})')
print('RESULT ' + json.dumps(dict(basis=a.basis, rows=rows)))
