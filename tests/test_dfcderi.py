"""Correctness of the cderi build.

``docs/DF_WORK_LIST.md`` **W4**.  `fastdfcderi` replaces
``df/df.py:_cholesky_eri``, the one-time three-centre build every
density-fitted run makes before its first SCF cycle.  Two things in it
change: the shell-pair aggregation the integral kernel is launched with,
which cannot change an answer, and the *association* of the metric
contraction, which can -- ``(j3c . C) L^{-T}`` instead of
``j3c . (C L^{-T})`` is the same tensor in exact arithmetic and not the same
rounding -- and the default form of it multiplies by an explicit ``L^{-1}``
rather than solving, which is one more step away from upstream's rounding.

So the bar here is the tensor itself, element-wise against GPU4PySCF's own
build, both implementations in one process:

* every angular-momentum class the auxiliary basis reaches -- def2-SVP's
  jkfit already carries f functions, def2-TZVPP's g -- because the
  Cartesian-to-spherical part of ``C`` is what the sparse apply replaces and
  it is the identity on s and p;
* a **Cartesian** orbital basis (6-31G\\*) as well as spherical ones, which
  take different branches of ``mat_dot_C``;
* an **ECP** case, where the auxiliary basis is the metal's own;
* both residencies, because the host branch contracts into a work buffer and
  the device branch straight into the output;
* both ``sp_mult`` values, since a launch configuration may only change how
  the same integrals are distributed over threads;
* and, on the host branch, **all three stores** -- upstream's synchronous
  host transpose, the device transpose with a ``cudaMemcpy2DAsync`` on a
  second stream, and the transpose that writes straight into the mapped
  buffer (``docs/DF_WORK_LIST.md`` W11).  Those cannot change an element
  either, and that is exactly why they need checking: an ordering mistake in
  a two-buffer, two-stream store shows up as a *block* of the tensor being
  the wrong one, which nothing but an element-wise comparison would catch.

The page-locked buffer is checked twice over: that one is page-locked at all,
and that a second build of the same size **reuses** it, which is W12.

Then, end to end: a density-fitted SCF energy against the unpatched one,
which is the only check that the whole path is still the same calculation.

    python tests/test_dfcderi.py
    python tests/test_dfcderi.py --basis def2-tzvpp      # up to g functions
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--tol', type=float, default=1e-9,
                help='maximum difference relative to the largest element of '
                     'the reference tensor')
ap.add_argument('--etol', type=float, default=1e-9, help='SCF energy, Eh')
ap.add_argument('--no-scf', action='store_true', help='skip the end-to-end pair')
ap.add_argument('--verbose', action='store_true')
a = ap.parse_args()

import cupy as cp                                             # noqa: E402
from pyscf import gto                                         # noqa: E402
from gpu4pyscf import dft                                     # noqa: E402
from gpu4pyscf.df import df as DFMOD                          # noqa: E402
from gpu4pyscf.df.df import DF                                # noqa: E402
from gpu4pyscf.df import int3c2e_bdiv as bdiv                 # noqa: E402
from gpu4pyscf.gto.mole import SortedMole                     # noqa: E402

_REF = DFMOD._cholesky_eri                                    # before patching

H2O = 'O 0.0 0.0 0.117; H 0.0 0.757 -0.467; H 0.0 -0.757 -0.467'
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
SI = ('Si 0.0 0.0 0.0; H 0.0 0.0 1.48; H 1.40 0.0 -0.48; '
      'H -0.70 1.21 -0.48; H -0.70 -1.21 -0.48')
#: an ECP case: def2 replaces the core of everything past Kr, and the
#: auxiliary basis is the metal's own
PTH = 'Pt 0.0 0.0 0.0; Cl 0.0 0.0 2.30; Cl 0.0 0.0 -2.30'

CASES = [('H2O', H2O, 0, 0, '6-31G*'),          # Cartesian orbital basis
         ('H2O', H2O, 0, 0, 'def2-SVP'),
         ('H2O', H2O, 0, 0, 'def2-TZVPP'),      # g functions in the jkfit set
         ('ethanol', ETHANOL, 0, 0, 'def2-SVP'),
         ('SiH4', SI, 0, 0, 'cc-pVDZ'),
         ('PtCl2', PTH, 0, 2, 'def2-SVP')]
if a.basis:
    seen, uniq = set(), []
    for n, g, c, s, _ in CASES:
        if n not in seen:
            seen.add(n)
            uniq.append((n, g, c, s, a.basis))
    CASES = uniq


def _intopt(mol, auxbasis):
    from pyscf.df import addons
    auxmol = addons.make_auxmol(mol, auxbasis)
    intopt = bdiv.Int3c2eOpt(mol, auxmol)
    intopt.mol = SortedMole.from_mol(mol, decontract=True)
    return intopt.build()


def _cmp(got, ref):
    got = cp.asarray(got)
    ref = cp.asarray(ref)
    scale = float(cp.abs(ref).max()) or 1.
    return float(cp.abs(got - ref).max()) / scale


def main():
    import electrolites
    electrolites.patch('fastdfcderi')
    from electrolites import fastdfcderi as F
    from electrolites import _dfj2c as J2C
    from electrolites import _dfpin as PIN
    from electrolites import _dfstore as STORE

    def ref(intopt, **kw):
        """GPU4PySCF's own build, with GPU4PySCF's own two-centre integrals.

        W13 replaces ``int3c2e_bdiv.int2c2e`` at the *module* level, and
        ``_REF`` looks it up there at call time -- so without this the
        reference and the patched build would share the very thing being
        checked, which is `docs/DF_WORK_LIST.md` 5.13 exactly.
        """
        mode, J2C._J2C = J2C._J2C, 'stock'
        try:
            return _REF(intopt, **kw)
        finally:
            J2C._J2C = mode
    # Every molecule here is below `_MIN_WORK`, so the default
    # configuration would take the dense product on all of them and the
    # sparse path would never be checked at all.  Forcing the threshold to
    # zero is the only way this suite covers it -- the same reason
    # `tests/test_df.py` forces `fastdf`'s rank-k branch.
    _DEFAULT_MIN_WORK = F._MIN_WORK
    F._MIN_WORK = 0
    print(f'# gpu4pyscf {electrolites.gpu4pyscf_version()}  '
          f'patched: {electrolites.patched()}  (_MIN_WORK forced to 0)',
          flush=True)

    worst = 0.
    fails = 0
    for name, geom, charge, spin, basis in CASES:
        mol = gto.M(atom=geom, basis=basis, charge=charge, spin=spin,
                    ecp=basis if basis.lower().startswith('def2')
                    and any(gto.charge(s) > 36 for s, _ in gto.format_atom(geom))
                    else None, verbose=0)
        for on_gpu in (True, False):
            intopt = _intopt(mol, a.auxbasis)
            reference = ref(
                intopt, use_gpu_memory=None if on_gpu else False)[0][0]
            reference = cp.asarray(reference)
            stores = ('host',) if on_gpu else ('host', 'stage', 'direct')
            for metric in ('trmm', 'trsm', 'gemm'):
                for sp_mult in (1, 4):
                    for store in stores:
                        F._METRIC, F._SPMULT = metric, sp_mult
                        STORE._STORE = store
                        got, idx = F._cholesky_eri(
                            intopt, use_gpu_memory=None if on_gpu else False)
                        rel = _cmp(got[0], reference)
                        ok = rel <= a.tol
                        worst = max(worst, rel)
                        fails += not ok
                        if a.verbose or not ok:
                            print(f'{"ok  " if ok else "FAIL"} {name:8s} '
                                  f'{basis:11s} gpu={on_gpu!s:5s} '
                                  f'metric={metric:5s} sp_mult={sp_mult} '
                                  f'store={store:6s} '
                                  f'naux={reference.shape[0]:5d}  {rel:.2e}',
                                  flush=True)
            STORE._STORE = 'stage'
            del reference
            cp.get_default_memory_pool().free_all_blocks()
        F._METRIC, F._SPMULT = 'trmm', 1
        print(f'{"ok  " if not fails else "    "} {name:8s} {basis:11s} '
              f'both residencies, both metrics, both aggregations, '
              f'all three stores', flush=True)

    print(f'\nworst relative difference on the cderi: {worst:.2e} '
          f'(tolerance {a.tol:.0e})', flush=True)
    for k in ('trmm', 'trsm'):
        if F.stats()[k] == 0:
            print(f'FAIL the {k} contraction never ran', flush=True)
            fails += 1
    if STORE.stats()['async_stores'] == 0:
        print('FAIL the asynchronous store never ran', flush=True)
        fails += 1
    if STORE.stats()['declined_reasons']:
        # a decline here is not wrong, but every host case above is small
        # enough that the second block fits ten times over, so one means the
        # arrangement is being refused for a reason worth reading
        print(f'#    store declined: {STORE.stats()["declined_reasons"]}',
              flush=True)
    # and the threshold itself: at the module's own default every case here
    # declines, which is the point of it
    F._MIN_WORK = _DEFAULT_MIN_WORK
    before = F.stats()['gemm']
    mol = gto.M(atom=H2O, basis='def2-SVP', verbose=0)
    F._cholesky_eri(_intopt(mol, a.auxbasis))
    if F.stats()['gemm'] != before + 1:
        print('FAIL the metric-work threshold did not send a small case to the '
              'dense product', flush=True)
        fails += 1
    F._MIN_WORK = 0
    print(f'# {F.stats()}', flush=True)

    if not a.no_scf:
        print('\n--- end to end: a density-fitted SCF ---', flush=True)
        # the last case runs with the cderi on the **host**, which is the
        # only residency a large molecule gets and the one this module pins
        # itself: `DF.loop` streams it back with an asynchronous copy, which
        # is a copy from page-locked memory or a silently synchronous one
        for i, (name, geom, charge, spin, basis) in enumerate(CASES[:4]):
            host = i == 3
            mol = gto.M(atom=geom, basis=basis, charge=charge, spin=spin,
                        verbose=0)
            mf = (dft.UKS if spin else dft.RKS)(mol, xc='b3lyp')
            mf = mf.density_fit(auxbasis=a.auxbasis)
            mf.conv_tol = 1e-11
            mf.grids.level = 3
            if host:
                mf.with_df.use_gpu_memory = False
            e = mf.kernel()
            DFMOD._cholesky_eri = _REF
            J2C._J2C = 'stock'          # the same trap as `ref` above
            mf2 = (dft.UKS if spin else dft.RKS)(mol, xc='b3lyp')
            mf2 = mf2.density_fit(auxbasis=a.auxbasis)
            mf2.conv_tol = 1e-11
            mf2.grids.level = 3
            if host:
                mf2.with_df.use_gpu_memory = False
            eref = mf2.kernel()
            J2C._J2C = 'cache'
            DFMOD._cholesky_eri = F._cholesky_eri
            d = abs(e - eref)
            ok = d <= a.etol
            fails += not ok
            print(f'{"ok  " if ok else "FAIL"} {name:8s} {basis:11s} '
                  f'{"cderi on the host" if host else "":18s} '
                  f'E = {e:.10f}  unpatched {eref:.10f}  d = {d:.2e}',
                  flush=True)

    # W12: that the buffer is page-locked at all, and that a second build of
    # the same size gets the first one's buffer back rather than locking a
    # new one.  The tensor is compared again after the reuse, because a pool
    # that hands out the wrong buffer is a pool that computes the wrong
    # answer and nothing in the timings would say so.
    if PIN._PIN == 'register':
        if PIN.stats()['registered_gb'] <= 0:
            print('FAIL no host buffer was page-locked by this module',
                  flush=True)
            fails += 1
        print('\n--- W12: the buffer, built twice ---', flush=True)
        PIN.free_pool()
        mol = gto.M(atom=ETHANOL, basis='def2-SVP', verbose=0)
        intopt = _intopt(mol, a.auxbasis)
        reference = cp.asarray(ref(intopt, use_gpu_memory=False)[0][0])
        before = PIN.stats()['reused']
        first = F._cholesky_eri(intopt, use_gpu_memory=False)[0][0]
        rel1 = _cmp(first, reference)
        ptr1 = first.ctypes.data
        del first
        second = F._cholesky_eri(intopt, use_gpu_memory=False)[0][0]
        rel2 = _cmp(second, reference)
        ptr2 = second.ctypes.data
        reused = PIN.stats()['reused'] - before
        ok = reused == 1 and ptr1 == ptr2 and max(rel1, rel2) <= a.tol
        fails += not ok
        print(f'{"ok  " if ok else "FAIL"} ethanol  def2-SVP    second build '
              f'reused the buffer: {reused == 1} (same address: '
              f'{ptr1 == ptr2}), cderi {rel1:.2e} then {rel2:.2e}', flush=True)
        del second, reference
        PIN.free_pool()
        # and that turning the pool off leaves the tensor right anyway
        PIN._POOL = False
        third = F._cholesky_eri(intopt, use_gpu_memory=False)[0][0]
        rel3 = _cmp(third, cp.asarray(ref(intopt, use_gpu_memory=False)[0][0]))
        fails += rel3 > a.tol
        print(f'{"ok  " if rel3 <= a.tol else "FAIL"} ethanol  def2-SVP    '
              f'FASTDFCDERI_POOL=0                    {rel3:.2e}', flush=True)
        PIN._POOL = True
        del third
        cp.get_default_memory_pool().free_all_blocks()
    # W13: the two-centre metric integrals, whose optimiser is now built
    # once and kept.  The bar is the matrix itself against GPU4PySCF's, at
    # omega = 0 and at omega != 0 -- the range-separated case is the one a
    # kept optimiser can get wrong, because `_check_rsh_factors` falls back
    # to `cell.omega` and a cached optimiser holds the cell of whichever
    # build made it.
    print('\n--- W13: the two-centre metric integrals ---', flush=True)
    from gpu4pyscf.df import int3c2e_bdiv as _bdiv
    J2C.free_cache()
    # `def2-SVP` and `def2-TZVPP` share `def2-universal-jkfit`, so the second
    # basis here is a different family: two auxiliary sets, not two orbital
    # ones, is what this section needs
    for basis in ('def2-SVP', 'cc-pVTZ'):
        mol = gto.M(atom=ETHANOL, basis=basis, verbose=0)
        saux = _intopt(mol, a.auxbasis).auxmol
        for omega in (None, 0.3, -0.3):
            saux.omega = 0. if omega is None else omega
            J2C._J2C = 'stock'
            want = _bdiv.int2c2e_orig(saux, omega=omega)
            worst_j2c = 0.
            for mode in ('build', 'cache', 'cache'):
                J2C._J2C = mode
                got = J2C.int2c2e(saux, omega=omega)
                worst_j2c = max(worst_j2c, _cmp(got, want))
            ok = worst_j2c <= a.tol
            fails += not ok
            print(f'{"ok  " if ok else "FAIL"} ethanol  {basis:11s} '
                  f'omega={str(omega):5s} naux={want.shape[0]:5d}  '
                  f'built, cached and cached again: {worst_j2c:.2e}',
                  flush=True)
        saux.omega = 0.
        # W27: a cached optimiser keeps the `cell` it was built from, which
        # can be another caller's auxmol; a long-range call through a second
        # object must not leave its omega on the first one.  It did, and a
        # range-separated DF gradient reusing the SCF's `Int3c2eOpt` then
        # built the long-range metric for its full-range call
        J2C._J2C = 'cache'
        first = _intopt(mol, a.auxbasis).auxmol
        second = _intopt(mol, a.auxbasis).auxmol
        want = _bdiv.int2c2e_orig(first)
        J2C._CACHE.clear()
        J2C.int2c2e(first)
        second.omega = 0.3
        J2C.int2c2e(second, omega=0.3)
        left = first.omega
        err = _cmp(J2C.int2c2e(first), want)
        ok = left == 0. and err <= a.tol
        fails += not ok
        print(f'{"ok  " if ok else "FAIL"} ethanol  {basis:11s} a long-range '
              f'call through a second auxmol leaves the first at omega='
              f'{left}: {err:.2e}', flush=True)
    J2C._J2C = 'cache'
    if J2C.stats()['reused'] == 0:
        print('FAIL the optimiser cache never hit', flush=True)
        fails += 1
    print(f'# {PIN.stats()}', flush=True)
    print(f'# {STORE.stats()}', flush=True)
    print(f'# {J2C.stats()}', flush=True)
    print(f'\n{"PASS" if not fails else f"FAIL ({fails})"}', flush=True)
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
