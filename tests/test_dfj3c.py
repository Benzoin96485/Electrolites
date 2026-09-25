"""Correctness of `fastdfj3c`, the three-centre density-fitted Coulomb build.

`docs/DF_WORK_LIST.md` **W9**.  This module is unusual among the ones in this
repository in that it changes **no arithmetic at all**: it launches GPU4PySCF's
own CUDA kernels with GPU4PySCF's own arguments, and what it removes is the
host-side work of re-deriving those arguments on every call.  So the bar is not
a tolerance argued from the physics -- it is *bit-for-bit or very nearly*, and
anything above round-off is a bug rather than a trade.

Every comparison here is made **in one process**, by swapping the bound methods
between GPU4PySCF's and this module's, so the two sides see the same molecule,
the same option objects and the same density.  That removes the cross-process
noise a patched/unpatched pair of runs would carry and makes an element-wise
comparison meaningful.

What is checked:

* the Coulomb matrix out of ``df_jk.get_j`` at a fixed density -- one and two
  densities, ``hermi`` 0 and 1, restricted and unrestricted.  Two densities is
  not a duplicate of one: ``df_jk.get_j`` sends ``n_dm > 1`` to the *Rys*
  ``int3c2e_bdiv`` engine instead of the McMurchie--Davidson one, so that row
  exercises the recontraction half of the module on a code path the other does
  not reach;
* a whole **pure**-functional DF SCF, which is what W9 is about, and a whole
  **hybrid** one, which builds a cderi and so runs the cached recontraction
  tables through ``_cholesky_eri`` as well;
* a **gradient**, because ``get_hcore`` and the cderi build reach the same four
  ``SortedGTO`` methods by a different route;
* a **RIJONX** (``only_dfj``) build, the other consumer of this Coulomb path;
* the caches are keyed on the **device** as well as on the object, because
  ``multi_gpu.run`` hands one ``SortedMole`` to a thread per device and a CuPy
  array carries the device it was allocated on.  That check is skipped, and
  says so, unless more than one GPU is visible;
* a **rebuild**: the cache is keyed on the option object's ``int3c2e_envs``,
  and a geometry optimisation calls ``build()`` again on the same object.  The
  test moves an atom, rebuilds, and requires the answer to follow -- a cache
  that did not invalidate would silently return the previous geometry's
  Coulomb matrix, which no energy check would catch.

    python tests/test_dfj3c.py
    python tests/test_dfj3c.py --basis def2-tzvpp
    python tests/test_dfj3c.py --no-scf        # matrices only

Needs one GPU.
"""
import argparse
import os
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--tol', type=float, default=1e-12,
                help='maximum difference relative to the largest element of '
                     "GPU4PySCF's own answer")
ap.add_argument('--etol', type=float, default=1e-9,
                help='maximum SCF energy difference, in Eh')
ap.add_argument('--no-scf', action='store_true')
a = ap.parse_args()

import electrolites                                           # noqa: E402
# `fastdfj` is patched underneath because one of the cases below is RIJONX,
# and `only_dfj=True` does not run at all on stock 1.8.1 -- it passes `mol`
# where `DF.get_jk` expects the density (`docs/README_fastdfj.md`).  It is a
# repair of a *different* entry point and both sides of every comparison here
# carry it, so it does not enter the comparison.
electrolites.patch(['fastdfj', 'fastdfj3c'])

import cupy as cp                                             # noqa: E402
from pyscf import gto                                         # noqa: E402
from gpu4pyscf import dft, scf                                # noqa: E402
from gpu4pyscf.df import df_jk, j_engine_3c2e                 # noqa: E402
from gpu4pyscf.gto.mole import SortedGTO                      # noqa: E402
from electrolites import fastdfj3c, _j3c                      # noqa: E402

ETHANOL = '''
C  -0.748  -0.015   0.024
C   0.558   0.420  -0.679
O   1.598  -0.513  -0.421
H  -1.293  -0.849  -0.435
H  -1.393   0.878   0.086
H  -0.581  -0.323   1.061
H   0.384   0.442  -1.766
H   0.874   1.415  -0.352
H   2.437  -0.166  -0.775
'''
CH3 = 'C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0'

_CTR = ('CT_dot_mat', 'C_dot_mat', 'mat_dot_C', 'mat_dot_CT')
_STOCK = {n: getattr(fastdfj3c, '_orig_' + n) for n in _CTR}
_FAST = {n: getattr(fastdfj3c, n) for n in _CTR}
_OPT = j_engine_3c2e.Int3c2eOpt


def install(fast):
    """Bind either GPU4PySCF's functions or this module's.  Nothing else moves.

    ``get_j`` is part of the swap, so a multi-density row below compares this
    module's McMurchie--Davidson build against GPU4PySCF's **Rys** one.  Those
    are two different integral algorithms, not two spellings of one, so that
    row is the only comparison here that is allowed to differ by more than
    round-off -- and it is held to the same bar anyway.
    """
    for n in _CTR:
        setattr(SortedGTO, n, (_FAST if fast else _STOCK)[n])
    _OPT.contract_dm = (fastdfj3c.contract_dm if fast
                        else fastdfj3c._orig_contract_dm)
    _OPT.contract_auxvec = (fastdfj3c.contract_auxvec if fast
                            else fastdfj3c._orig_contract_auxvec)
    df_jk.get_j = fastdfj3c.get_j if fast else fastdfj3c._orig_get_j


def _err(got, ref):
    got = np.asarray(cp.asnumpy(got), dtype=float)
    ref = np.asarray(cp.asnumpy(ref), dtype=float)
    scale = max(float(np.abs(ref).max()), 1e-300)
    return float(np.abs(got - ref).max()) / scale


def _dm(mol, spin, ndm, seed=0):
    """A physical density, and ``ndm`` of them when asked."""
    rng = np.random.default_rng(seed)
    mf = (scf.UHF if spin else scf.RHF)(mol)
    mf.max_cycle = 1
    mf.conv_tol = 1e-4
    mf.verbose = 0
    mf.kernel()
    dm0 = cp.asarray(mf.make_rdm1())
    if ndm == 1:
        return dm0
    out = [dm0]
    for _ in range(ndm - 1):
        d = cp.asarray(rng.normal(scale=1e-3, size=dm0.shape))
        out.append(dm0 + (d + d.swapaxes(-1, -2)) * .5)
    return cp.stack(out)


def matrices():
    """``df_jk.get_j`` element-wise, both bindings, one process."""
    fails = 0
    for name, geom, spin in (('ethanol', ETHANOL, 0), ('CH3', CH3, 1)):
        mol = gto.M(atom=geom, basis=a.basis, spin=spin, verbose=0)
        mf = (scf.UHF if spin else scf.RHF)(mol).density_fit(
            auxbasis=a.auxbasis)
        mf.verbose = 0
        dfobj = mf.with_df
        dfobj.build(build_cderi=False)

        for ndm in (1, 2):
            for hermi in (1, 0):
                dm = _dm(mol, spin, ndm)
                d = cp.asarray(dm)
                d = d.reshape(-1, d.shape[-1], d.shape[-1])
                install(False)
                ref = df_jk.get_j(dfobj, d, hermi)
                install(True)
                got = df_jk.get_j(dfobj, d, hermi)
                e = _err(got, ref)
                ok = e < a.tol
                fails += not ok
                engine = ('MD vs MD' if len(d) == 1 else 'MD vs Rys/bdiv')
                print(f'{"ok  " if ok else "FAIL"} {name}/{a.basis} '
                      f'spin={spin} ndm={len(d)} hermi={hermi} '
                      f'({engine}): {e:.1e}')

        # `get_hcore` reaches the same four recontraction methods by another
        # route entirely (`scf/hf.py:159`), and it is the one caller of them
        # that is not density fitting at all
        install(False)
        h_ref = mf.get_hcore(mol)
        install(True)
        h = mf.get_hcore(mol)
        e = _err(h, h_ref)
        ok = e < a.tol
        fails += not ok
        print(f'{"ok  " if ok else "FAIL"} {name} get_hcore: {e:.1e}')
    return fails


def rebuild():
    """A second ``build()`` on the same option object must invalidate the cache.

    This is the only way this module can be wrong without being wrong
    everywhere: the constants are keyed on ``int3c2e_envs``, and a geometry
    optimisation, a finite-difference gradient and ``DF.reset()`` all call
    ``build()`` again on an object that already has them.
    """
    mol = gto.M(atom=ETHANOL, basis=a.basis, verbose=0)
    moved = gto.M(atom=ETHANOL.replace('2.437', '2.537'), basis=a.basis,
                  verbose=0)
    install(True)
    mf = scf.RHF(mol).density_fit(auxbasis=a.auxbasis)
    mf.verbose = 0
    dfobj = mf.with_df
    dfobj.build(build_cderi=False)
    df_jk.get_j(dfobj, _dm(mol, 0, 1), 1)          # fills the cache

    # the same DF object, rebuilt around the moved geometry
    dm = _dm(moved, 0, 1)
    dfobj.reset(moved)
    dfobj.build(build_cderi=False)
    got = df_jk.get_j(dfobj, dm, 1)

    install(False)
    ref_mf = scf.RHF(moved).density_fit(auxbasis=a.auxbasis)
    ref_mf.verbose = 0
    ref_mf.with_df.build(build_cderi=False)
    ref = df_jk.get_j(ref_mf.with_df, dm, 1)
    e = _err(got, ref)
    ok = e < a.tol
    print(f'{"ok  " if ok else "FAIL"} rebuild after reset(): {e:.1e}')
    fails = not ok

    # `DF.reset(mol)` drops `j_engine` entirely, so the paragraph above only
    # proves that a *fresh* option object gets a fresh cache.  The invalidation
    # itself is the `int3c2e_envs` identity test, and it is asserted directly:
    # a second `build()` on the same object must not be served the first
    # build's constants.
    install(True)
    opt = j_engine_3c2e.Int3c2eOpt(mol, dfobj.auxmol).build()
    first = _j3c.constants(opt)
    opt.build()
    second = _j3c.constants(opt)
    ok = second is not first and second.envs is opt.int3c2e_envs
    print(f'{"ok  " if ok else "FAIL"} Int3c2eOpt.build() twice: '
          f'{"cache invalidated" if ok else "STALE CONSTANTS SERVED"}')
    return fails + (not ok)


def devices():
    """A cache keyed on the object alone would hand device 1 device 0's rows.

    GPU4PySCF distributes the cderi build with ``multi_gpu.run``, which hands
    the **same** ``SortedMole`` to a thread per device, and a CuPy array
    carries the device it was allocated on.  `docs/DF_WORK_LIST.md` W2 records
    the same trap on `fastdf`'s ``_resident_prefix``.  Nothing else in this
    suite can see it, because ``env.sh`` hides every GPU but one.

    Skipped, loudly, when only one device is visible.
    """
    n = cp.cuda.runtime.getDeviceCount()
    if n < 2:
        print(f'skip  per-device caches: {n} device visible '
              '(run with CUDA_VISIBLE_DEVICES=0,1 to exercise this)')
        return 0
    mol = gto.M(atom=ETHANOL, basis=a.basis, verbose=0)
    smol = SortedGTO.from_mol(mol, decontract=True, diffuse_cutoff=1e200)
    bad = []
    for i in range(min(n, 2)):
        with cp.cuda.Device(i):
            t = _j3c.sorted_constants(smol)
            for name in ('c_ao_loc', 'p_ao_loc', 'recontract_coef',
                         'recontract_bas', 'recontraction_idx'):
                arr = getattr(t, name)
                if arr.device.id != i:
                    bad.append(f'{name} on device {arr.device.id}, '
                               f'asked on {i}')
    ok = not bad
    print(f'{"ok  " if ok else "FAIL"} per-device recontraction tables '
          f'({min(n, 2)} devices): {"; ".join(bad) or "each on its own device"}')
    return not ok


def energies():
    """Whole runs: a pure functional, a hybrid, RIJONX, and a gradient."""
    fails = 0
    # `tpss` is a pure meta-GGA, i.e. a second functional class that reaches
    # this Coulomb build for its whole two-electron half; `wb97x` is
    # range-separated, which builds a **second** `DF` object through
    # `DF.range_coulomb` and so a second set of every cache this module keeps
    # (`docs/DF_WORK_LIST.md` §5.4 is the trap that has).
    cases = (('ethanol', ETHANOL, 0, 'pbe', False),
             ('ethanol', ETHANOL, 0, 'tpss', False),
             ('ethanol', ETHANOL, 0, 'b3lyp', False),
             ('ethanol', ETHANOL, 0, 'wb97x', False),
             ('ethanol', ETHANOL, 0, 'b3lyp', True),
             ('CH3', CH3, 1, 'pbe', False),
             ('CH3', CH3, 1, 'b3lyp', False))
    for name, geom, spin, xc, only_dfj in cases:
        mol = gto.M(atom=geom, basis=a.basis, spin=spin, verbose=0)
        out = {}
        for fast in (False, True):
            install(fast)
            mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc).density_fit(
                auxbasis=a.auxbasis, only_dfj=only_dfj)
            mf.grids.level = 3
            mf.verbose = 0
            mf.conv_tol = 1e-11
            out[fast] = (mf.kernel(), mf.cycles)
        de = abs(out[True][0] - out[False][0])
        ok = de < a.etol and out[True][1] == out[False][1]
        fails += not ok
        tag = 'RIJONX' if only_dfj else 'DF'
        print(f'{"ok  " if ok else "FAIL"} {name}/{xc}/{tag} spin={spin}: '
              f'dE = {de:.1e} Eh, cycles {out[False][1]} vs {out[True][1]}')

    # a gradient: `get_hcore` and the cderi build reach the cached tables by a
    # different route from `get_j`
    mol = gto.M(atom=ETHANOL, basis=a.basis, verbose=0)
    g = {}
    for fast in (False, True):
        install(fast)
        mf = dft.RKS(mol, xc='b3lyp').density_fit(auxbasis=a.auxbasis)
        mf.grids.level = 3
        mf.verbose = 0
        mf.conv_tol = 1e-11
        mf.kernel()
        g[fast] = np.asarray(mf.nuc_grad_method().kernel())
    dg = float(np.abs(g[True] - g[False]).max())
    ok = dg < 1e-9
    fails += not ok
    print(f'{"ok  " if ok else "FAIL"} ethanol/b3lyp/DF gradient: '
          f'max component difference {dg:.1e} Eh/Bohr')
    return fails


NI_DITHIOLENE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'benchmarks', 'molecules',
                             'tmc_ni_dithiolene.xyz')


def kernels():
    """W19's kernels class by class, and the split with GPU4PySCF's.

    `electrolites/_j3ck.py` serves every class up to ``KMAX_ORDER`` and
    leaves the shell-pair groups that meet a higher one to GPU4PySCF's
    kernels.  Three cases: an open-shell 3d complex, whose auxiliary set
    reaches ``i`` (lk 6) on the metal, with every class this module's; the
    same with the ceiling at 8, so that the groups past ``lij = 2`` go back to
    GPU4PySCF and the two sets of kernels write into one output; and ethanol
    with the ceiling at 6, the same split in a basis whose auxiliary set
    stops at ``g``.
    """
    from electrolites import _j3ck
    fails = 0
    cases = [('Ni dithiolene', NI_DITHIOLENE, -1, 1, 99),
             ('Ni dithiolene', NI_DITHIOLENE, -1, 1, 8),
             ('ethanol', ETHANOL, 0, 0, 6)]
    for name, geom, charge, spin, kmax in cases:
        mol = gto.M(atom=geom, basis=a.basis, charge=charge, spin=spin,
                    verbose=0)
        mf = (scf.UHF if spin else scf.RHF)(mol).density_fit(
            auxbasis=a.auxbasis)
        mf.verbose = 0
        dfobj = mf.with_df
        dfobj.build(build_cderi=False)
        dm = cp.asarray(_dm(mol, spin, 1))
        dm = dm.reshape(-1, mol.nao, mol.nao).sum(0)
        install(False)
        ref = df_jk.get_j(dfobj, dm, 1)
        install(True)
        old = _j3ck.MAX_ORDER
        _j3ck.MAX_ORDER = kmax
        if dfobj.j_engine is not None:
            setattr(dfobj.j_engine, _j3ck._ATTR, {})
        try:
            got = df_jk.get_j(dfobj, dm, 1)
            plan = _j3ck.plan(dfobj.j_engine, _j3c.constants(dfobj.j_engine))
        finally:
            _j3ck.MAX_ORDER = old
            setattr(dfobj.j_engine, _j3ck._ATTR, {})
        e = _err(got, ref)
        ok = e < a.tol
        fails += not ok
        print(f'{"ok  " if ok else "FAIL"} W19 kernels, {name}/{a.basis}, '
              f'ceiling {kmax}: {len(plan.classes)} classes ours, '
              f'{len(plan.theirs)} groups GPU4PySCF\'s: {e:.1e}')
    return fails


def incremental():
    """W20's build from the difference: the same SCF, the same answer.

    The screen drops what a Schwarz bound says is below ``INC_TOL``, so the
    Fock matrix differs by that much and a DIIS trajectory can take one cycle
    more or fewer; the energy has to agree to ``--etol``.  Then the guard: a
    call whose density is nothing like the previous one (a response density)
    must be built whole, and must equal a build that never had a state.
    """
    fails = 0
    for name, geom, spin, xc in (('ethanol', ETHANOL, 0, 'pbe'),
                                 ('CH3', CH3, 1, 'pbe'),
                                 ('ethanol', ETHANOL, 0, 'b3lyp')):
        mol = gto.M(atom=geom, basis=a.basis, spin=spin, verbose=0)
        out = {}
        for inc in (False, True):
            install(True)
            fastdfj3c.INCREMENTAL = inc
            fastdfj3c.STATS['inc_step'] = 0
            mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc).density_fit(
                auxbasis=a.auxbasis)
            mf.grids.level = 3
            mf.verbose = 0
            mf.conv_tol = 1e-11
            e = mf.kernel()
            out[inc] = (e, mf.cycles, fastdfj3c.STATS['inc_step'], mf)
        fastdfj3c.INCREMENTAL = False
        de = abs(out[True][0] - out[False][0])
        dc = abs(out[True][1] - out[False][1])
        ok = de < a.etol and dc <= 1 and (out[True][2] > 0 or xc == 'b3lyp')
        fails += not ok
        print(f'{"ok  " if ok else "FAIL"} W20 {name}/{xc} spin={spin}: dE = '
              f'{de:.1e} Eh, cycles {out[False][1]} vs {out[True][1]}, '
              f'{out[True][2]} builds from the difference')

    # the guard, on the state the last SCF left behind
    mf = out[True][3]
    dfobj = mf.with_df
    rng = np.random.default_rng(3)
    d = cp.asarray(rng.normal(scale=1e-2, size=(mol.nao, mol.nao)))
    d = d + d.T
    fastdfj3c.INCREMENTAL = True
    full0 = fastdfj3c.STATS['inc_full']
    got = df_jk.get_j(dfobj, d, 1)
    whole = fastdfj3c.STATS['inc_full'] - full0
    fastdfj3c.INCREMENTAL = False
    ref = df_jk.get_j(dfobj, d, 1)
    e = _err(got, ref)
    ok = e < a.tol and whole == 1
    fails += not ok
    print(f'{"ok  " if ok else "FAIL"} W20 a response-like density after an '
          f'SCF is built whole ({whole}): {e:.1e}')
    return fails


def main():
    print(f'# gpu4pyscf {electrolites.gpu4pyscf_version()}  '
          f'patched: {electrolites.patched()}  basis {a.basis}')
    fails = matrices()
    fails += kernels()
    fails += rebuild()
    fails += devices()
    if not a.no_scf:
        fails += energies()
        fails += incremental()
    install(True)
    print(f'\nfastdfj3c {fastdfj3c.stats()}')
    print('FAILURES' if fails else 'all ok')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
