"""The unrestricted VV10 terms of `fastxcnlc` and `fastxcgrad`, in one process.

Three things are checked on the same converged UKS density, over GGA+VV10 and
meta-GGA+VV10 functionals:

* **`fastxcnlc.nr_uks_nlc`** -- the fused XC and non-local-correlation grid
  pass -- against GPU4PySCF's `nr_uks` *plus* `nr_nlc_vxc`, which is the pair
  of calls `dft/uks.py:get_veff` makes.  Both the energy and both spin blocks
  of the potential matrix.
* **`fastxcgrad.get_nlc_exc` at two density matrices**, in the `grouped` mode
  a gradient uses and in the `perblock` ablation.  GPU4PySCF's own function
  covers this case, so there is a reference to compare against; what changes
  here is that the module used to hand it back.
* the **whole UKS gradient**, patched against unpatched, which is what a
  caller sees.

And, separately, that the fusion *fired*: `fastxcnlc` is allowed to decline
and returns the right answer either way, so `stats()` is asserted on rather
than inferred from an energy agreement.  `docs/ROUND2_1.8.1.md` §5 is the
round where that distinction was learned the hard way.

`fastnlc` is deliberately **not** patched here: the VV10 double sum has to be
the same code on both sides of the comparison, or this test would be measuring
that kernel instead of the grid passes around it.

    python tests/test_unlc.py [--quick]
"""
import argparse

import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fastxcnlc', 'fastxcgrad')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.dft import numint as NI                       # noqa: E402
from gpu4pyscf.dft import uks as _uks                        # noqa: E402
from gpu4pyscf.grad import rks as grks                       # noqa: E402
from gpu4pyscf.lib.cupy_helper import tag_array              # noqa: E402
from electrolites import fastxcgrad, fastxcnlc               # noqa: E402

O2 = 'O 0. 0. 0.; O 0. 0. 1.208'
CH3 = '''C 0. 0. 0.; H 1.079 0. 0.
H -0.539 0.934 0.; H -0.539 -0.934 0.'''
NO = 'N 0. 0. 0.; O 0. 0. 1.154'
M = 'benchmarks/molecules/'

# (name, atom, basis, xc, grid level, charge, spin)
CASES = [
    ('O2', O2, 'def2-svp', 'wb97x-v', 3, 0, 2),           # GGA + VV10
    ('O2', O2, 'def2-svp', 'wb97m-v', 3, 0, 2),           # meta-GGA + VV10
    ('CH3', CH3, '6-31g*', 'b97m-v', 3, 0, 1),            # non-hybrid + VV10
    ('NO', NO, 'def2-tzvpp', 'wb97m-v', 2, 0, 1),         # reaches g
    ('CuCl2', M + 'tmc_cucl2.xyz', 'def2-svp', 'wb97m-v', 3, 0, 1),
    ('CuCl2', M + 'tmc_cucl2.xyz', 'def2-tzvp', 'wb97x-v', 2, 0, 1),
    ('FeCl4-', M + 'tmc_fecl4.xyz', 'def2-svp', 'wb97m-v', 2, -1, 5),
    ('Pt amine', M + 'tmc_pt_amine.xyz', 'def2-svp', 'wb97x-v', 1, 0, 2),
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


def nlc_code(mf):
    ni = mf._numint
    return mf.xc if ni.libxc.is_nlc(mf.xc) else mf.nlc


print(f'{"case":34s} {"xc":9s} {"nao":>5s} '
      f'{"fused exc":>10s} {"fused vmat":>10s} '
      f'{"nlc1(grp)":>10s} {"nlc1(blk)":>10s} {"whole grad":>11s}')
print('-' * 112)
fails = skipped = 0
for name, atom, basis, xc, level, charge, spin in cases:
    mol = build(atom, basis, charge, spin)
    mf = dft.UKS(mol, xc=xc)
    mf.grids.level = level
    mf.nlcgrids.level = level      # else the two grids differ and the fusion
    mf.conv_tol = 1e-10            # declines, which is correct but untested
    mf.direct_scf_tol = 1e-13
    mf.max_cycle = 200
    n0 = fastxcnlc.stats()
    mf.kernel()
    if not mf.converged:
        print(f'{name+"/"+basis:34s} {xc:9s} SCF did not converge -- skipped',
              flush=True)
        skipped += 1
        continue
    n1 = fastxcnlc.stats()
    fired = n1['fused_u'] - n0['fused_u']
    declined = n1['declined_u'] - n0['declined_u']

    ni = mf._numint
    dm = cp.asarray(mf.make_rdm1()).view(cp.ndarray)

    # --- the fused grid pass against the two calls it replaces ------------
    got = fastxcnlc.nr_uks_nlc(ni, mol, mf.grids, mf.xc, nlc_code(mf), dm)
    assert got is not None, 'nr_uks_nlc declined a case the SCF fused'
    ne_x, exc_x, v_x = NI._nr_uks_orig(ni, mol, mf.grids, mf.xc, dm) \
        if hasattr(NI, '_nr_uks_orig') else ni.nr_uks(mol, mf.grids, mf.xc, dm)
    _ne_n, exc_n, v_n = ni.nr_nlc_vxc(mol, mf.nlcgrids, nlc_code(mf), dm)
    exc_ref = float(exc_x) + float(exc_n)
    v_ref = cp.asarray(v_x) + cp.asarray(v_n)      # (nao,nao) broadcasts
    d_exc = abs(got[1] - exc_ref) / max(abs(exc_ref), 1.0)
    d_v = (float(cp.abs(cp.asarray(got[2]) - v_ref).max())
           / (float(cp.abs(v_ref).max()) or 1.0))
    d_ne = float(np.abs(np.asarray(got[0]) - np.asarray(ne_x)).max())
    ne_x = v_x = v_n = v_ref = None

    # --- the VV10 gradient term at two density matrices -------------------
    dmt = tag_array(mf.make_rdm1(), mo_coeff=cp.asarray(mf.mo_coeff),
                    mo_occ=cp.asarray(mf.mo_occ))
    errs = {}
    for mode in ('grouped', 'perblock'):
        fastxcgrad.MODE = mode
        e1 = cp.asarray(grks.get_nlc_exc(ni, mol, mf.nlcgrids, nlc_code(mf),
                                         dmt, verbose=0)[1])
        e0 = cp.asarray(grks._get_nlc_exc_orig(ni, mol, mf.nlcgrids,
                                               nlc_code(mf), dmt,
                                               verbose=0)[1])
        errs[mode] = (float(cp.abs(e1 - e0).max())
                      / (float(cp.abs(e0).max()) or 1.0))
    fastxcgrad.MODE = 'grouped'

    # --- and the whole gradient, ours against GPU4PySCF's -----------------
    de_fast = mf.nuc_grad_method().kernel()
    grks.get_nlc_exc = grks._get_nlc_exc_orig
    try:
        de_ref = mf.nuc_grad_method().kernel()
    finally:
        grks.get_nlc_exc = fastxcgrad.get_nlc_exc
    dg = float(np.abs(np.asarray(de_fast) - np.asarray(de_ref)).max())

    ok = (d_exc < 1e-11 and d_v < 1e-11 and d_ne < 1e-8
          and max(errs.values()) < 1e-9 and dg < 1e-9
          and fired > 0 and declined == 0)
    fails += not ok
    print(f'{name+"/"+basis:34s} {xc:9s} {mol.nao:5d} '
          f'{d_exc:10.2e} {d_v:10.2e} {errs["grouped"]:10.2e} '
          f'{errs["perblock"]:10.2e} {dg:11.2e}'
          f'{"" if ok else "  <-- FAIL"}', flush=True)
    if not ok:
        print(f'    fused {fired} / declined {declined}, d(nelec) {d_ne:.2e}',
              flush=True)

# --- the whole SCF energy, fused against the two separate calls -----------
print()
mol = build(O2, 'def2-svp', 0, 2)
e_fused = e_stock = None
for tag in ('fused', 'stock'):
    if tag == 'stock':
        _uks.UKS.get_veff = _uks.get_veff = fastxcnlc._ORIG_GET_VEFF_U
    mf = dft.UKS(mol, xc='wb97m-v')
    mf.grids.level = 3
    mf.nlcgrids.level = 3
    mf.conv_tol = 1e-11
    e = mf.kernel()
    if tag == 'fused':
        e_fused = e
    else:
        e_stock = e
_uks.UKS.get_veff = _uks.get_veff = fastxcnlc.get_veff_u
d_scf = abs(e_fused - e_stock)
print(f'O2/def2-svp/wb97m-v UKS SCF: fused {e_fused:.11f}  '
      f'two passes {e_stock:.11f}  d = {d_scf:.2e} Eh')
scf_ok = d_scf < 1e-9
fails += not scf_ok

print(f'\n{len(cases) - skipped} cases + 1 whole-SCF check, {fails} failed, '
      f'{skipped} skipped (SCF did not converge)')
print(f'fastxcnlc.stats() = {fastxcnlc.stats()}')
raise SystemExit(1 if fails else 0)
