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
  the same integrals are distributed over threads.

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
            ref = _REF(intopt, use_gpu_memory=None if on_gpu else False)[0][0]
            ref = cp.asarray(ref)
            for metric in ('trmm', 'trsm', 'gemm'):
                for sp_mult in (1, 4):
                    F._METRIC, F._SPMULT = metric, sp_mult
                    got, idx = F._cholesky_eri(
                        intopt, use_gpu_memory=None if on_gpu else False)
                    rel = _cmp(got[0], ref)
                    ok = rel <= a.tol
                    worst = max(worst, rel)
                    fails += not ok
                    if a.verbose or not ok:
                        print(f'{"ok  " if ok else "FAIL"} {name:8s} '
                              f'{basis:11s} gpu={on_gpu!s:5s} '
                              f'metric={metric:5s} sp_mult={sp_mult} '
                              f'naux={ref.shape[0]:5d}  {rel:.2e}', flush=True)
            del ref
            cp.get_default_memory_pool().free_all_blocks()
        F._METRIC, F._SPMULT = 'trmm', 1
        print(f'{"ok  " if not fails else "    "} {name:8s} {basis:11s} '
              f'both residencies, both metrics, both aggregations', flush=True)

    print(f'\nworst relative difference on the cderi: {worst:.2e} '
          f'(tolerance {a.tol:.0e})', flush=True)
    for k in ('trmm', 'trsm'):
        if F.stats()[k] == 0:
            print(f'FAIL the {k} contraction never ran', flush=True)
            fails += 1
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
            mf2 = (dft.UKS if spin else dft.RKS)(mol, xc='b3lyp')
            mf2 = mf2.density_fit(auxbasis=a.auxbasis)
            mf2.conv_tol = 1e-11
            mf2.grids.level = 3
            if host:
                mf2.with_df.use_gpu_memory = False
            eref = mf2.kernel()
            DFMOD._cholesky_eri = F._cholesky_eri
            d = abs(e - eref)
            ok = d <= a.etol
            fails += not ok
            print(f'{"ok  " if ok else "FAIL"} {name:8s} {basis:11s} '
                  f'{"cderi on the host" if host else "":18s} '
                  f'E = {e:.10f}  unpatched {eref:.10f}  d = {d:.2e}',
                  flush=True)

    if F._PIN == 'register' and F.stats()['host_registered_gb'] <= 0:
        print('FAIL no host buffer was page-locked by this module', flush=True)
        fails += 1
    print(f'\n{"PASS" if not fails else f"FAIL ({fails})"}', flush=True)
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
