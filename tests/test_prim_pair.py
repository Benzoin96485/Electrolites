"""The wide bra primitive-pair module, against the default one and against
GPU4PySCF.

The kernels cache the bra shell pair's primitive data in a shared array of
``MAX_PRIM_PAIR`` doubles, and 36 -- 6x6, the worst case in a Pople set -- used
to be the only width compiled, so a class block whose bra pair was longer fell
through.  A def2 set forms longer ones: 49 on an S or Cl pair, 64 on a 3d
metal's innermost s shell, 100 at def2-QZVP.  `_ksplit.mpp_plan` now builds a
second module for exactly those classes.

Three things are checked, in one process each time:

* **the plan** is a function of the (l, nprim) group table alone -- no GPU
  needed for that half, and it runs first so that a wrong plan fails before any
  compile;
* **the answer does not depend on the bound**: the exchange matrix, the fused
  build and the gradient's per-atom derivative come out the same whether the
  wide module is allowed or not, and both agree with GPU4PySCF's own kernels;
* **a basis with nothing to widen is untouched**: 6-31G\\* and def2-SVP produce
  no second module at all, so those rows are the regression check on the
  default path rather than a comparison.

The density is fixed after a few SCF cycles rather than converged: both bounds
are handed the same one, so what is compared is the kernels either way, and
some of these cases -- CuCl2 in particular -- take a long time to converge for
reasons that have nothing to do with the kernels (`docs/OPENSHELL_TMC.md` §3).

    python tests/test_prim_pair.py [--quick] [--cycles N] [--cap N]
"""
import argparse

import numpy as np

import electrolites
electrolites.patch('fastk', 'fastjk', 'fastejk')

import cupy as cp                                            # noqa: E402
from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.grad import rhf as grhf                       # noqa: E402
from gpu4pyscf.scf import jk as gjk                          # noqa: E402
from electrolites import _ksplit, fastk, fastejk             # noqa: E402
from electrolites.compat import (                            # noqa: E402
    sorted_meta as _sorted_meta, resolve_rsh as _resolve_rsh)

M = 'benchmarks/molecules/'

#: (atom, basis, charge, spin, the longest bra pair this basis forms)
CASES = [
    (M + 'glycine.xyz', '6-31g*', 0, 0, 36),        # nothing to widen
    (M + 'glycine.xyz', 'def2-svp', 0, 0, 36),      # nor here
    (M + 'tmc_cucl2.xyz', 'def2-svp', 0, 1, 36),    # nor on a metal at DZ
    (M + 'glycine.xyz', 'def2-tzvp', 0, 0, 36),     # C, N, O, H stop at 6x6
    (M + 'penicillinV.xyz', 'def2-tzvp', 0, 0, 49),  # S: 7x7, and organic
    (M + 'tmc_cucl2.xyz', 'def2-tzvp', 0, 1, 64),   # Cu: 8x8
    (M + 'tmc_fecl4.xyz', 'def2-tzvp', -1, 5, 64),  # high spin, n_dm = 2
    # A 5d metal with the def2 ECP has *no* deep core s contraction -- the ECP
    # replaced it -- so this complex's longest pair is its chlorines' 7 x 7,
    # exactly as an organic molecule's would be.  The entry that motivated this
    # work reads the bound as a transition-metal problem; this row is the case
    # where the metal is the one atom that does not cause it.
    (M + 'tmc_pt_amine.xyz', 'def2-tzvp', 0, 2, 49),
    (M + 'tmc_cucl2.xyz', 'def2-tzvpp', 0, 1, 64),  # g shells present too
    (M + 'tmc_cucl2.xyz', 'def2-qzvp', 0, 1, 100),  # the top of the ladder
]

#: (group table, expected (mpp, lmax)) -- the plan on its own, no GPU
PLANS = [
    ([(0, 6), (0, 3), (1, 3), (2, 1)], (36, -1)),          # 6-31G*
    ([(0, 5), (0, 3), (1, 3), (2, 1)], (36, -1)),          # def2-SVP, C/H/O
    ([(0, 7), (0, 6), (1, 5), (2, 1), (3, 1)], (49, 0)),   # def2-TZVP with S
    ([(0, 8), (0, 7), (1, 6), (2, 4), (3, 1)], (64, 1)),   # def2-TZVP, 3d
    # def2-QZVP: 10 x 10 = 100 on the s pair, and the *d* group matters here
    # too -- 4 x 10 = 40 is past the bound, so the plan has to reach l = 2
    ([(0, 10), (0, 7), (1, 6), (2, 4), (4, 1)], (100, 2)),
]

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
ap.add_argument('--cap', type=int, default=None)
ap.add_argument('--cycles', type=int, default=6,
                help='SCF cycles before the density is fixed; see below')
a = ap.parse_args()
CYCLES = a.cycles
CAP = _ksplit.MAX_PRIM_PAIR_CAP if a.cap is None else a.cap
cases = (CASES[:1] + CASES[4:7]) if a.quick else CASES

fails = 0
print('the plan, from the (l, nprim) group table alone')
for tbl, want in PLANS:
    got = _ksplit.mpp_plan(np.array(tbl), 36, 100)
    ok = got == want
    fails += not ok
    print(f'  {str(tbl):46s} -> {str(got):10s} '
          + ('' if ok else f'  <-- FAIL, expected {want}'))
# the cap is what makes a pair too long to serve rather than a hard limit
got = _ksplit.mpp_plan(np.array([(0, 12), (1, 6)]), 36, 100)
if got != (72, 1):        # 12x6 = 72 is inside the cap, 12x12 = 144 is not
    fails += 1
    print(f'  a pair past the cap must be left out: got {got}, want (72, 1)')
if _ksplit.mpp_plan(np.array([(0, 8), (1, 6)]), 36, 0) != (36, -1):
    fails += 1
    print('  cap 0 must plan no second module at all')


def build(atom, basis, charge, spin):
    kw = dict(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0,
              max_memory=40000)
    if basis.lower().startswith('def2') and any(
            gto.charge(sym) > 36 for sym, _ in gto.format_atom(atom)):
        kw['ecp'] = basis
    return gto.M(**kw)


def at_cap(cap, fn):
    """Run ``fn`` with the second module allowed up to ``cap``."""
    old = _ksplit.MAX_PRIM_PAIR_CAP
    _ksplit.MAX_PRIM_PAIR_CAP = cap
    try:
        return fn()
    finally:
        _ksplit.MAX_PRIM_PAIR_CAP = old


print(f'\n{"case":40s} {"nao":>5s} {"pair":>5s} '
      f'{"K on/off":>10s} {"K vs ref":>10s} {"JK vs ref":>10s} '
      f'{"ejk on/off":>11s} {"ejk vs ref":>11s}')
print('-' * 112)
for atom, basis, charge, spin, want_pair in cases:
    mol = build(atom, basis, charge, spin)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc='b3lyp')
    mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.direct_scf_tol = 1e-13
    # A converged density is not wanted here and is not cheap on some of these
    # cases: both bounds are handed the *same* density, so what is compared is
    # the kernels either way, and a few cycles is enough to put a realistic
    # amount of charge on the polarisation shells (a `minao` guess is exactly
    # zero there and screens far harder -- docs/ROUND2_1.8.1.md 2).
    mf.max_cycle = CYCLES
    mf.kernel()
    name = atom.rsplit('/', 1)[-1].replace('.xyz', '')[:26] + '/' + basis
    dm = cp.asarray(mf.make_rdm1())
    vhfopt = gjk._VHFOpt(mol, mf.direct_scf_tol).build()
    uniq_l_ctr = _sorted_meta(vhfopt)[0]
    mpp, mpp_lmax = _ksplit.mpp_plan(uniq_l_ctr, fastk.MAX_PRIM_PAIR, CAP)
    if mpp != want_pair:
        fails += 1
        print(f'{name:40s} longest bra pair {mpp}, expected {want_pair}'
              '  <-- FAIL', flush=True)
        continue

    sdm = cp.asarray(vhfopt.apply_coeff_C_mat_CT(
        dm if dm.ndim == 3 else dm[None]), order='C')

    k_on = at_cap(CAP, lambda: fastk.get_k(vhfopt, sdm, 1, 0, 0., 1., 1.))
    k_off = at_cap(0, lambda: fastk.get_k(vhfopt, sdm, 1, 0, 0., 1., 1.))
    k_ref = gjk._VHFOpt._get_k_orig(vhfopt, sdm, 1, 0, 0., 1., 1.)
    kscale = max(float(cp.abs(k_ref).max()), 1e-30)
    d_k = float(cp.abs(k_on - k_off).max()) / kscale
    d_kref = float(cp.abs(k_on - k_ref).max()) / kscale

    jk_on = at_cap(CAP, lambda: gjk._VHFOpt.get_jk(vhfopt, sdm, 1, 0))
    jk_ref = gjk._VHFOpt._get_jk_orig(vhfopt, sdm, 1, 0)
    d_jk = max(float(cp.abs(x - y).max()) / max(float(cp.abs(y).max()), 1e-30)
               for x, y in zip(jk_on, jk_ref))

    e_on = np.asarray(at_cap(
        CAP, lambda: fastejk.jk_energy_per_atom(vhfopt, dm, 1.0, 0.2)))
    e_off = np.asarray(at_cap(
        0, lambda: fastejk.jk_energy_per_atom(vhfopt, dm, 1.0, 0.2)))
    e_ref = np.asarray(grhf._jk_energy_per_atom_orig(vhfopt, dm, 1.0, 0.2))
    escale = max(float(np.abs(e_ref).max()), 1e-30)
    d_e = float(np.abs(e_on - e_off).max()) / escale
    d_eref = float(np.abs(e_on - e_ref).max()) / escale

    worst = max(d_k, d_kref, d_jk, d_e, d_eref)
    ok = worst < 1e-11
    fails += not ok
    print(f'{name:40s} {mol.nao:5d} {mpp:5d} {d_k:10.2e} {d_kref:10.2e} '
          f'{d_jk:10.2e} {d_e:11.2e} {d_eref:11.2e}'
          + ('' if ok else '  <-- FAIL'), flush=True)


# The range-separated path is a different driver -- `get_k_rsh` builds
# lr*erf + sr*erfc in one pass over the quartets -- with its own class loop and
# its own copy of the bra-pair test, so it gets its own row.  The arguments are
# not guessed: one `get_veff` call with a range-separated functional is run
# with `get_k` wrapped, and whatever GPU4PySCF asked for is replayed through
# our dispatch at both bounds and through its own kernel.
print('\nthe range-separated driver (get_k_rsh), same three ways')
for atom, basis, charge, spin in [(M + 'tmc_cucl2.xyz', 'def2-tzvp', 0, 1)]:
    mol = build(atom, basis, charge, spin)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc='wb97x')
    mf.grids.level = 3
    mf.direct_scf_tol = 1e-13
    mf.max_cycle = CYCLES
    mf.kernel()
    dm = cp.asarray(mf.make_rdm1())

    seen = []
    patched = gjk._VHFOpt.get_k

    def record(self, dms, hermi, verbose, *rsh):
        # `self` and `dms` have to be captured, not rebuilt: for a
        # range-separated request GPU4PySCF calls this on an option object
        # built inside `mol.with_range_coulomb(|omega|)`, and `get_k_rsh`
        # asserts exactly that.  A freshly built full-range `_VHFOpt` is a
        # different object with a different operator on its mol.
        seen.append((self, dms, hermi, rsh))
        return patched(self, dms, hermi, verbose, *rsh)
    gjk._VHFOpt.get_k = record
    try:
        mf.get_veff(mol, dm)
    finally:
        gjk._VHFOpt.get_k = patched
    opt, rdms, hermi, rsh = seen[-1]
    name = atom.rsplit('/', 1)[-1].replace('.xyz', '')[:26] + '/' + basis
    # The operator GPU4PySCF asks for is not the one it passes: 1.8.x hands
    # over (omega, lr_factor, sr_factor) and `_check_rsh_factors` resolves it,
    # and a *negative* resolved omega is the fused lr*erf + sr*erfc request
    # that `get_k_rsh` serves.  Resolve before deciding which path this is.
    om = _resolve_rsh(opt, *rsh)[0] if rsh else _resolve_rsh(opt)[0]
    route = 'get_k_rsh (fused)' if om < 0 else 'get_k'
    rdms = rdms() if callable(rdms) else rdms
    on = at_cap(CAP, lambda: patched(opt, rdms, hermi, 0, *rsh))
    off = at_cap(0, lambda: patched(opt, rdms, hermi, 0, *rsh))
    ref = gjk._VHFOpt._get_k_orig(opt, rdms, hermi, 0, *rsh)
    sc = max(float(cp.abs(ref).max()), 1e-30)
    d_on = float(cp.abs(on - off).max()) / sc
    d_ref = float(cp.abs(on - ref).max()) / sc
    ok = max(d_on, d_ref) < 1e-11
    fails += not ok
    print(f'{name:34s} {mol.nao:5d} {route:18s} omega={om:+.3f} '
          f'on/off {d_on:9.2e}  vs GPU4PySCF {d_ref:9.2e}'
          + ('' if ok else '  <-- FAIL'), flush=True)

print(f'\n{fails} failed')
raise SystemExit(1 if fails else 0)
