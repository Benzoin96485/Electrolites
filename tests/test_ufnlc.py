"""`fastfnlc` on an unrestricted reference, against GPU4PySCF, in one process.

The VV10 response term reaches an open-shell caller through
`hessian.uks.nr_uks_fnlc_mo`, which sums the two perturbed densities and
forwards to the restricted function -- so every case here goes in through that
function rather than through `fastfnlc` directly, because the forwarding is
half of what is being tested: `hessian/uks.py` imports `nr_rks_fnlc_mo` **by
name at import time**, and until this round patching `hessian.rks` reached
none of it.

Each case calls the term twice, as a CPKS or CPHF solve does, so the second
call exercises the ground-state cache.  `mo` is `return_in_mo=True`, which
keeps GPU4PySCF's own contraction and therefore has to be **bit-identical**
(and, for an unrestricted reference, has to come back as a two-element tuple);
`ao` is `return_in_mo=False`, whose contraction is batched over the perturbed
densities and is checked to round-off.

Then a whole UKS TDA run, which is where this matters most: `tdscf/rhf.py`
sets `exclude_nlc = True`, but `tdscf/uhf.py`'s `TDBase.gen_response` -- what
`tdscf/uks.py` inherits -- forwards to `gen_response` with no `with_nlc`
argument at all, and its default is True.  So an open-shell TDA with wB97M-V
pays this term on every Davidson iteration where a closed-shell one pays none
of it.

    python tests/test_ufnlc.py [--quick]
"""
import argparse

import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fastfnlc')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.hessian import rks as HR                      # noqa: E402
from gpu4pyscf.hessian import uks as HU                      # noqa: E402
from electrolites import fastfnlc                            # noqa: E402

# the alias patch itself, which no numerical check can see
assert HU.nr_rks_fnlc_mo is fastfnlc._dispatch, \
    'hessian/uks.py still holds the unpatched nr_rks_fnlc_mo'

O2 = 'O 0. 0. 0.; O 0. 0. 1.208'
CH3 = '''C 0. 0. 0.; H 1.079 0. 0.
H -0.539 0.934 0.; H -0.539 -0.934 0.'''
NO = 'N 0. 0. 0.; O 0. 0. 1.154'
M = 'benchmarks/molecules/'

# (name, atom, basis, xc, grid level, charge, spin, nset)
CASES = [
    ('O2', O2, 'def2-svp', 'wb97m-v', 3, 0, 2, 3),
    ('O2', O2, 'def2-svp', 'wb97x-v', 3, 0, 2, 1),
    ('CH3', CH3, '6-31g*', 'b97m-v', 3, 0, 1, 3),
    ('CH3', CH3, 'def2-svp', 'wb97m-v', 3, 0, 1, 6),
    ('NO', NO, 'def2-tzvpp', 'wb97m-v', 2, 0, 1, 3),      # reaches g
    ('CuCl2', M + 'tmc_cucl2.xyz', 'def2-svp', 'wb97m-v', 2, 0, 1, 3),
    ('Pt amine', M + 'tmc_pt_amine.xyz', 'def2-svp', 'wb97x-v', 1, 0, 2, 2),
]

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
a = ap.parse_args()
cases = CASES[:3] if a.quick else CASES


def build(atom, basis, charge, spin):
    kw = dict(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0,
              max_memory=40000)
    if basis.lower().startswith('def2') and any(
            gto.charge(sym) > 36 for sym, _ in gto.format_atom(atom)):
        kw['ecp'] = basis
    return gto.M(**kw)


def _perturbed(mo, occ, en, mol, nset):
    """`nset` perturbed density pairs, from the dipole right-hand side."""
    out = []
    for s in range(2):
        mocc, mvir = mo[s][:, occ[s] > 0], mo[s][:, occ[s] == 0]
        de = en[s][occ[s] == 0][:, None] - en[s][occ[s] > 0][None, :]
        with mol.with_common_orig((0, 0, 0)):
            h1 = cp.asarray(mol.intor('int1e_r'))
        xs = [-(mvir.T @ h1[i] @ mocc) / de for i in range(3)]
        order = cp.argsort(de.ravel())
        for k in range(max(0, nset - 3)):
            x = cp.zeros(de.shape)
            x.ravel()[order[k % de.size]] = 1.0
            xs.append(x)
        d = cp.asarray([mvir @ x @ mocc.T for x in xs[:nset]])
        out.append(cp.asarray(d + d.transpose(0, 2, 1), order='C'))
    return cp.asarray(out, order='C')


print(f'{"case":34s} {"xc":9s} {"nset":>4s} {"mo (exact?)":>12s} '
      f'{"ao":>10s} {"2nd call":>10s}')
print('-' * 86)
fails = skipped = 0
for name, atom, basis, xc, level, charge, spin, nset in cases:
    mol = build(atom, basis, charge, spin)
    mf = dft.UKS(mol, xc=xc)
    mf.grids.level = level
    mf.nlcgrids.level = level
    mf.conv_tol = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        print(f'{name+"/"+basis:34s} {xc:9s} SCF did not converge -- skipped',
              flush=True)
        skipped += 1
        continue
    assert mf.do_nlc()

    mo = cp.asarray(mf.mo_coeff)
    occ = cp.asarray(mf.mo_occ)
    en = cp.asarray(mf.mo_energy)
    dm1 = _perturbed(mo, occ, en, mol, nset)

    def call(in_mo, patched):
        HU.nr_rks_fnlc_mo = (fastfnlc._dispatch if patched
                             else HR._nr_rks_fnlc_mo_orig)
        try:
            return HU.nr_uks_fnlc_mo(mf, mol, mo, occ, dm1, in_mo)
        finally:
            HU.nr_rks_fnlc_mo = fastfnlc._dispatch

    errs = {}
    exact = True
    for in_mo in (True, False):
        ref = call(in_mo, False)
        got = call(in_mo, True)
        got2 = call(in_mo, True)          # the cached ground-state half
        if in_mo:
            assert isinstance(got, tuple) and len(got) == 2, \
                'an unrestricted MO-basis answer must be a two-element tuple'
            ref = cp.concatenate([cp.asarray(r).ravel() for r in ref])
            got = cp.concatenate([cp.asarray(g).ravel() for g in got])
            got2 = cp.concatenate([cp.asarray(g).ravel() for g in got2])
        else:
            ref, got, got2 = (cp.asarray(x) for x in (ref, got, got2))
        scale = float(cp.abs(ref).max()) or 1.0
        errs[in_mo] = float(cp.abs(got - ref).max()) / scale
        errs[(in_mo, 2)] = float(cp.abs(got2 - ref).max()) / scale
        if in_mo:
            exact = bool(cp.all(got == ref)) and bool(cp.all(got2 == ref))

    ok = (exact and errs[False] <= 1e-12 and errs[(False, 2)] <= 1e-12)
    fails += not ok
    print(f'{name+"/"+basis:34s} {xc:9s} {nset:4d} {errs[True]:9.2e} '
          f'{str(exact):>3s} {errs[False]:10.2e} '
          f'{max(errs[(True, 2)], errs[(False, 2)]):10.2e}'
          f'{"" if ok else "  <-- FAIL"}', flush=True)

# --- a whole UKS TDA run, which does include the VV10 response ------------
print()
mol = build(O2, 'def2-svp', 0, 2)
mf = dft.UKS(mol, xc='wb97m-v')
mf.grids.level = 3
mf.nlcgrids.level = 3
mf.conv_tol = 1e-11
mf.kernel()
es = {}
for tag, fn in (('fast', fastfnlc._dispatch),
                ('stock', HR._nr_rks_fnlc_mo_orig)):
    HU.nr_rks_fnlc_mo = fn
    td = mf.TDA()
    td.nstates = 3
    td.conv_tol = 1e-9
    td.kernel()
    es[tag] = np.asarray(td.e)
HU.nr_rks_fnlc_mo = fastfnlc._dispatch
d_td = float(np.abs(es['fast'] - es['stock']).max())
print('O2/def2-svp/wb97m-v UKS TDA, 3 roots (Eh):')
print('  fast ', '  '.join(f'{x:.10f}' for x in es['fast']))
print('  stock', '  '.join(f'{x:.10f}' for x in es['stock']))
print(f'  max |d| = {d_td:.2e}')
td_ok = d_td < 1e-9
fails += not td_ok

print(f'\n{len(cases) - skipped} cases + 1 TDA check, {fails} failed, '
      f'{skipped} skipped (SCF did not converge)')
raise SystemExit(1 if fails else 0)
