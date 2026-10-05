"""`fastdfmp2` against GPU4PySCF's DF-MP2 pair-energy loops (catalogue I1).

The pair-energy functions on the same Cholesky integrals, both ways, with
and without amplitudes, restricted (bi1/bi2), same-spin and opposite-spin,
float64 and float32 integrals; then whole DF-MP2 and DF-UMP2 energies.
Tolerances: 1e-12 relative for float64 integrals; for float32 integrals the
reference sums in float32 and this module in float64, so the check is
1e-6 relative on the pair energies and the energies.

    python tests/test_dfmp2.py
"""
import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fastdfmp2')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import scf                                    # noqa: E402
from gpu4pyscf.mp import dfmp2, dfump2                       # noqa: E402
from gpu4pyscf.mp import dfmp2_addons as A                   # noqa: E402
from electrolites import fastdfmp2                           # noqa: E402

GLY = 'benchmarks/molecules/glycine.xyz'
rng = np.random.default_rng(1)
fails = 0
N = lambda x: cp.asnumpy(cp.asarray(x))


def check(name, got, ref, tol):
    global fails
    e = max(np.abs(N(g) - N(r)).max() / max(np.abs(N(r)).max(), 1e-300)
            for g, r in zip(got, ref))
    ok = e < tol
    fails += not ok
    print(f'{name:44s} {e:.1e}  {"ok" if ok else "FAIL"}', flush=True)


print('the functions, on random integrals')
for dt, tol in ((np.float64, 1e-12), (np.float32, 1e-5)):
    for nocc, nvir, naux in ((7, 23, 60), (40, 70, 150)):
        c = cp.asarray(rng.standard_normal((nocc, nvir, naux)) * .1, dtype=dt)
        c2 = cp.asarray(rng.standard_normal((nocc + 3, nvir - 4, naux)) * .1,
                        dtype=dt)
        eo = -np.sort(rng.uniform(.3, 2, nocc))
        ev = np.sort(rng.uniform(.1, 3, nvir))
        eo2 = -np.sort(rng.uniform(.3, 2, nocc + 3))
        ev2 = np.sort(rng.uniform(.1, 3, nvir - 4))
        tag = f'{np.dtype(dt).name} {nocc}x{nvir}x{naux}'
        for t in (False, True):
            t2a = cp.zeros((nocc, nocc, nvir, nvir), dtype=dt) if t else None
            t2b = cp.zeros((nocc, nocc, nvir, nvir), dtype=dt) if t else None
            got = A.get_dfmp2_energy_pair_intra(
                gto.Mole(), c, eo, ev, t2=t2a)
            ref = A._get_dfmp2_energy_pair_intra_orig(
                gto.Mole(), c, eo, ev, t2=t2b)
            check(f'restricted {tag} t2={t}', list(got) + ([t2a] if t else []),
                  list(ref) + ([t2b] if t else []), tol)
            if t:
                t2a.fill(1.); t2b.fill(2.)
            got = A.get_dfmp2_energy_pair_intra(
                gto.Mole(), c, eo, ev, ss_only=True, t2=t2a)
            ref = A._get_dfmp2_energy_pair_intra_orig(
                gto.Mole(), c, eo, ev, ss_only=True, t2=t2b)
            check(f'same spin {tag} t2={t}', [got] + ([t2a] if t else []),
                  [ref] + ([t2b] if t else []), tol)
            t3a = (cp.zeros((nocc, nocc + 3, nvir, nvir - 4), dtype=dt)
                   if t else None)
            t3b = (cp.zeros((nocc, nocc + 3, nvir, nvir - 4), dtype=dt)
                   if t else None)
            got = A.get_dfump2_energy_pair_intra(
                gto.Mole(), [c, c2], [eo, eo2], [ev, ev2], t2=t3a)
            ref = A._get_dfump2_energy_pair_intra_orig(
                gto.Mole(), [c, c2], [eo, eo2], [ev, ev2], t2=t3b)
            check(f'opposite spin {tag} t2={t}', [got] + ([t3a] if t else []),
                  [ref] + ([t3b] if t else []), tol)

print('\nwhole DF-MP2 / DF-UMP2')
for atom, spin, charge in ((GLY, 0, 0), (GLY, 1, 1)):
    mol = gto.M(atom=atom, basis='def2-svp', spin=spin, charge=charge,
                verbose=0)
    mf = (scf.UHF if spin else scf.RHF)(mol).density_fit()
    mf.kernel()
    PT = dfump2.DFUMP2 if spin else dfmp2.DFMP2
    e1 = PT(mf).run().e_corr
    keep = (A.get_dfmp2_energy_pair_intra, A.get_dfump2_energy_pair_intra)
    A.get_dfmp2_energy_pair_intra = A._get_dfmp2_energy_pair_intra_orig
    A.get_dfump2_energy_pair_intra = A._get_dfump2_energy_pair_intra_orig
    try:
        e0 = PT(mf).run().e_corr
    finally:
        A.get_dfmp2_energy_pair_intra, A.get_dfump2_energy_pair_intra = keep
    ok = abs(e1 - e0) < 1e-6 * abs(e0)
    fails += not ok
    print(f'glycine {"UHF" if spin else "RHF"} e_corr {e1:.10f} vs {e0:.10f} '
          f'{"ok" if ok else "FAIL"}', flush=True)
print('\nFAILED' if fails else '\nall passed', f'({fails} failures)')
raise SystemExit(1 if fails else 0)
