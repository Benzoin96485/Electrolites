"""Does `fastdfhess` compute the same CPHF J and K as GPU4PySCF?

``docs/DF_WORK_LIST.md`` **W8**.  ``df/hessian/rhf.py:_get_jk`` is the fifth
two-electron code path and the only one `fastdf` does not reach, and
`electrolites/fastdfhess.py` rearranges it: the exchange becomes one pass over
one intermediate instead of two passes over a bigger one, the density batches
are evened out, and the Coulomb-only branch becomes two BLAS calls.  Every one
of those is claimed to be the *same arithmetic*, so the bar here is the bar
the rest of the repository uses -- element-wise against GPU4PySCF's own
function, in one process, on one cderi tensor.

Six things are checked.

* **J and K, all three arrangements.**  ``fused`` (the default),
  ``blasvk`` and ``upstream`` (the same ordering GPU4PySCF uses, through
  this module's own loop) against the real function, at several
  right-hand-side counts.
* **The Coulomb-only call**, which is the whole two-electron half of a *pure*
  functional's CPHF and takes a different branch here.
* **Unrestricted**, because ``df/hessian/uhf.py`` binds the same ``_get_jk``
  and the spin index runs through every contraction.
* **The long-range operator**, which arrives through
  ``DF.range_coulomb(omega)`` on a *copy* of the density-fitting object --
  ``docs/DF_WORK_LIST.md`` §5.4 is what happens when a patch does not survive
  that copy.
* **A whole Hessian**, patched against unpatched, because a matrix that is
  right and a Hessian that is right are different claims: the CPHF is
  iterative and this module changes what its first call leaves in the
  allocator.
* **The batch arithmetic**, which has no numerical symptom: the evened-out
  passes have to cover ``n_dm`` exactly once.

    python tests/test_dfhess.py
    python tests/test_dfhess.py --basis def2-tzvpp
    python tests/test_dfhess.py --no-hessian     # the matrices only
"""
import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--tol', type=float, default=1e-11,
                help='bar on |dJ| and |dK| relative to the largest element '
                     'of the matrix GPU4PySCF returns')
ap.add_argument('--tol-hessian', type=float, default=1e-8,
                help='bar on the largest Hessian element, in Eh/bohr^2')
ap.add_argument('--natm-block', type=int, default=0)
ap.add_argument('--no-hessian', action='store_true')
ap.add_argument('--verbose', action='store_true')
a = ap.parse_args()

import cupy as cp                                            # noqa: E402
from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.lib.cupy_helper import (                      # noqa: E402
    contract, tag_array, transpose_sum)
from gpu4pyscf.df.hessian import rhf as DFHESS               # noqa: E402

import electrolites                                          # noqa: E402
electrolites.patch('fastdfhess')
from electrolites import fastdfhess                          # noqa: E402

#: GPU4PySCF's own function, saved before anything patched it
UPSTREAM = getattr(DFHESS, '_get_jk_orig', None) or fastdfhess._orig_get_jk

H2O = 'O 0.0 0.0 0.117; H 0.0 0.757 -0.467; H 0.0 -0.757 -0.467'
ETHANOL = '''
C  -0.748  -0.015   0.024
C   0.558   0.420  -0.679
O   1.598  -0.513  -0.421
H  -1.593   0.365  -0.554
H  -0.834   0.355   1.048
H  -0.780  -1.106   0.023
H   0.640   1.502  -0.580
H   0.567   0.174  -1.741
H   2.421  -0.190  -0.792
'''

fails = []


def check(label, got, ref, tol=None):
    tol = a.tol if tol is None else tol
    got = cp.asarray(got)
    ref = cp.asarray(ref)
    if got.shape != ref.shape:
        fails.append(f'{label}: shape {got.shape} != {ref.shape}')
        print(f'  FAIL {label:52s} shape {got.shape} != {ref.shape}')
        return
    scale = float(cp.abs(ref).max()) or 1.0
    err = float(cp.abs(got - ref).max()) / scale
    ok = err <= tol
    if not ok:
        fails.append(f'{label}: {err:.3e} > {tol:.1e}')
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {err:.3e}'
          f'{"" if ok else f"   > {tol:.1e}"}')


def cphf_density(mf, n_dm, seed=0):
    """The right-hand sides a CPHF iteration sees.

    ``hessian/rhf.py:gen_vind`` builds them: an MO-basis trial vector, taken
    to the AO basis, symmetrised, and tagged with the two factors the
    density-fitted build contracts against rather than with the matrix.
    """
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    if mo_coeff.ndim == 3:                                   # UHF
        orbo = [mo_coeff[s][:, mo_occ[s] > 0] for s in range(2)]
        nocc = min(o.shape[1] for o in orbo)
        orbo = cp.stack([o[:, :nocc] for o in orbo])
        nmo = mo_coeff.shape[-1]
        rng = np.random.default_rng(seed)
        mo1 = cp.asarray(rng.standard_normal((2, n_dm, nmo, nocc)) * 1e-2)
        mo1_ao = contract('snpo,sip->snio', mo1, mo_coeff)
        dm1 = cp.asarray(contract('snpi,sqi->snpq', mo1_ao, orbo),
                         order='C')
        for s in range(2):
            transpose_sum(dm1[s], inplace=True, hermi=1)
        return tag_array(dm1, mo1=mo1_ao, occ_coeff=orbo, symmetrize=1)
    orbo = mo_coeff[:, mo_occ > 0]
    nmo = mo_coeff.shape[1]
    nocc = orbo.shape[1]
    rng = np.random.default_rng(seed)
    mo1 = cp.asarray(rng.standard_normal((n_dm, nmo, nocc)) * 1e-2)
    mo1_ao = contract('npo,ip->nio', mo1, mo_coeff)
    dm1 = cp.asarray(contract('npi,qi->npq', mo1_ao, orbo * 2), order='C')
    transpose_sum(dm1, inplace=True, hermi=1)
    return tag_array(dm1, mo1=mo1_ao, occ_coeff=orbo, symmetrize=1)


def run(mf, tag, n_dms=(1, 3, 6), omegas=(None,)):
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    dfobj = mf.with_df
    if dfobj._cderi is None:
        dfobj.build()
    for n_dm in n_dms:
        dms = cphf_density(mf, n_dm)
        for omega in omegas:
            for with_j, with_k in ((True, True), (True, False)):
                lbl = (f'{tag} n_dm={n_dm} '
                       f'{"J+K" if with_k else "J  "}'
                       f'{"" if omega is None else f" omega={omega}"}')
                with dfobj.range_coulomb(omega) as df_ref:
                    if df_ref._cderi is None:
                        df_ref.build(omega=omega)
                    vj_r, vk_r = UPSTREAM(df_ref, dms, mo_coeff, mo_occ, 1,
                                          with_j, with_k, omega)
                for arrange in ('fused', 'blasvk', 'upstream'):
                    fastdfhess.ARRANGE = arrange
                    with dfobj.range_coulomb(omega) as df_p:
                        vj_p, vk_p = fastdfhess._get_jk(
                            df_p, dms, mo_coeff, mo_occ, 1, with_j, with_k,
                            omega)
                    check(f'{lbl} [{arrange}] J', vj_p, vj_r)
                    if with_k:
                        check(f'{lbl} [{arrange}] K', vk_p, vk_r)
                fastdfhess.ARRANGE = 'fused'


print(f'# gpu4pyscf {electrolites.gpu4pyscf_version()}  '
      f'patched {electrolites.patched()}  basis {a.basis}')

print('\n## restricted, B3LYP')
mol = gto.M(atom=ETHANOL, basis=a.basis, verbose=0)
mf = dft.RKS(mol, xc='b3lyp').density_fit()
mf.grids.level = 3
mf.conv_tol = 1e-10
mf.kernel()
run(mf, 'ethanol')

print('\n## the long-range operator')
molw = gto.M(atom=H2O, basis=a.basis, verbose=0)
mfw = dft.RKS(molw, xc='b3lyp').density_fit()
mfw.grids.level = 3
mfw.conv_tol = 1e-10
mfw.kernel()
run(mfw, 'water', n_dms=(3,), omegas=(None, 0.3))

print('\n## unrestricted')
molu = gto.M(atom=H2O, basis=a.basis, charge=1, spin=1, verbose=0)
mfu = dft.UKS(molu, xc='b3lyp').density_fit()
mfu.grids.level = 3
mfu.conv_tol = 1e-10
mfu.kernel()
run(mfu, 'water+', n_dms=(1, 3))

print('\n## the batch arithmetic (no numerical symptom)')
from pyscf import lib as pyscf_lib                           # noqa: E402
bad = []
for n_dm in range(1, 200):
    for raw in range(1, n_dm + 1):
        nb = max(1, DFHESS._balance_batch_size(raw, n_dm))
        passes = [b - c for c, b in pyscf_lib.prange(0, n_dm, nb)]
        if sum(passes) != n_dm or nb > n_dm:
            bad.append((n_dm, raw, nb, passes))
print(f'  {"ok  " if not bad else "FAIL"} '
      f'balanced passes cover n_dm exactly once for every (n_dm, raw) '
      f'below 200   ({len(bad)} bad)')
if bad:
    fails.append(f'batch arithmetic: {bad[:3]}')

print('\n## a call this module does not cover falls through')
before = fastdfhess.STATS['declined']
plain = cp.asarray(mf.make_rdm1())
try:
    fastdfhess._get_jk(mf.with_df, plain, cp.asarray(mf.mo_coeff),
                       cp.asarray(mf.mo_occ), 1, True, True, None)
    raised = None
except Exception as exc:                                      # noqa: BLE001
    raised = exc
declined = fastdfhess.STATS['declined'] - before
ok = declined == 1
if not ok:
    fails.append(f'untagged density: declined {declined} times, expected 1')
print(f'  {"ok  " if ok else "FAIL"} '
      f'{"an untagged density is handed back to GPU4PySCF":52s} '
      f'declined={declined}, upstream then raised '
      f'{type(raised).__name__ if raised else "nothing"}')

if not a.no_hessian:
    print('\n## a whole Hessian, patched against unpatched')
    molh = gto.M(atom=H2O, basis=a.basis, verbose=0)
    ref = None
    for off in (True, False):
        fastdfhess._OFF = off
        mfh = dft.RKS(molh, xc='b3lyp').density_fit()
        mfh.grids.level = 3
        mfh.conv_tol = 1e-11
        mfh.conv_tol_cpscf = 1e-8
        mfh.kernel()
        h = np.asarray(mfh.Hessian().kernel())
        if ref is None:
            ref = h
        else:
            err = np.abs(h - ref).max()
            ok = err <= a.tol_hessian
            if not ok:
                fails.append(f'water Hessian: {err:.3e}')
            print(f'  {"ok  " if ok else "FAIL"} '
                  f'{"water B3LYP/def2-SVP Hessian":52s} {err:.3e} Eh/bohr^2')
    fastdfhess._OFF = False
    print(f'  stats: {fastdfhess.stats()}')

print()
if fails:
    print(f'{len(fails)} FAILURES')
    for f in fails:
        print(f'  {f}')
    sys.exit(1)
print('all ok')
