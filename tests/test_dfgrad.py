"""Correctness of the density-fitted gradient's two-electron derivative.

`fastdfgrad` does not recompute the three-centre integrals
``df/grad/rhf.py:_jk_energy_per_atom`` evaluates in its first loop.  It takes
the occupied-occupied fitting coefficients from the **cderi** the SCF already
built, using the identity

    aux_coef @ aux_coef.T == metric        (``docs/README_fastdfgrad.md`` §3)

where ``aux_coef`` is the matrix ``df/df.py:_decompose_j2c`` defines the cderi
by and ``metric`` is the one the gradient rebuilds for itself.  Everything
that can go wrong with that is an *ordering* question -- which permutation the
auxiliary index is in, which AO basis the density factor is in, which of the
two decompositions the metric got -- and none of it shows up as anything but a
wrong number.  So the bar here is element-wise against GPU4PySCF's own
function, in the same process, on the same converged density:

  * **§1** the identity itself, reported against the condition number of the
    fitting metric that bounds it;
  * **§2** the per-atom derivative, patched against stock, over two basis
    sets, four functionals (including a range-separated one, whose second
    call asks for a long-range cderi the SCF built separately), both spin
    cases and a charged system;
  * **§3** the whole analytic gradient, and a **finite-difference** check of
    it -- which is the only test here that can say the patched answer is not
    merely *close to* upstream's but closer to the derivative of the energy;
  * **§4** the declines, because a module that silently computed the wrong
    thing on an uncovered case would pass §2;
  * **§5** the store **W10** put that array in -- the raw-pointer round trip
    on shapes the molecules here do not produce, and then the gradient
    through every combination of the two knobs, including the batched clone
    of GPU4PySCF's own first loop that the route with no cderi to reuse
    takes.  A host-resident store is a second CUDA stream and a `cudaMemcpy`
    per spin, and the failure mode is a *race*: the first version of it read
    the array back before the metric pass had finished writing it and was
    wrong by 1e-2, which nothing but an element-wise check finds.

    python tests/test_dfgrad.py
    python tests/test_dfgrad.py --basis def2-tzvpp
    python tests/test_dfgrad.py --no-fd        # skip the 12 extra SCFs

Needs one GPU.
"""
import argparse
import os
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--tol', type=float, default=1e-8,
                help='maximum absolute difference in Eh/Bohr against the '
                     'same quantity from GPU4PySCF, in the same process')
ap.add_argument('--lr-tol', type=float, default=1e-7,
                help='the same, for a range-separated functional\'s '
                     'long-range term.  Looser for a measured reason: the '
                     'long-range fitting metric is rank deficient -- 127 of '
                     '335 auxiliary directions are below LINEAR_DEP_THR on '
                     'ethanol/def2-svp at omega=0.3 -- so both routes go '
                     'through a filtered pseudo-inverse whose elements reach '
                     '4e7, and merely *associating* the metric contraction '
                     'the other way round on upstream\'s own j3c_oo already '
                     'moves dm_oo by 3.6e-8.  See docs/README_fastdfgrad.md '
                     'section 5')
ap.add_argument('--fd-tol', type=float, default=2e-5,
                help='finite-difference agreement, in Eh/Bohr.  Loose '
                     'because the reference is a 3-point difference of two '
                     'SCF energies, not because the gradient is')
ap.add_argument('--no-fd', action='store_true')
ap.add_argument('--verbose', action='store_true')
a = ap.parse_args()

import electrolites                                           # noqa: E402
electrolites.patch('fastdfgrad')

import cupy as cp                                             # noqa: E402
from pyscf import gto                                         # noqa: E402
from gpu4pyscf import dft                                     # noqa: E402
from gpu4pyscf.df import df as DFMOD                          # noqa: E402
from gpu4pyscf.df.grad import rhf as df_rhf_grad              # noqa: E402
from gpu4pyscf.df.grad import uhf as df_uhf_grad              # noqa: E402
from gpu4pyscf.df.int3c2e_bdiv import (                       # noqa: E402
    Int3c2eOpt, int2c2e, argsort_aux)
from gpu4pyscf.gto.mole import SortedMole                     # noqa: E402
from electrolites import fastdfgrad                           # noqa: E402

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
WATER = 'O 0 0 0.117; H 0 0.757 -0.470; H 0 -0.757 -0.470'

FAIL = []


def check(label, err, tol):
    ok = err <= tol
    if not ok:
        FAIL.append(label)
    print(f'{label:58s} {err:10.3e}   {"ok" if ok else "FAIL"} '
          f'(tol {tol:.0e})', flush=True)
    return ok


def build(atom, basis, xc, spin=0, charge=0, conv_tol=1e-11):
    mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc).density_fit(
        auxbasis=a.auxbasis)
    mf.conv_tol = conv_tol
    mf.grids.level = 3
    mf.kernel()
    assert mf.converged, f'{atom[:12]} {basis} {xc} did not converge'
    return mol, mf


# ---------------------------------------------------------------------------
print('\n=== 1. the identity the reuse rests on ===', flush=True)
# `aux_coef @ aux_coef.T` is the metric the gradient builds with a Cholesky
# solve of its own.  The fitting metric is badly conditioned by construction
# -- 1e8 is ordinary for a jkfit set -- so the relative agreement here is the
# bound on everything below, and it is reported next to that condition number
# rather than against an absolute tolerance plucked from the air.
for basis in (a.basis, 'def2-tzvpp'):
    mol, mf = build(WATER, basis, 'b3lyp')
    auxmol_raw = mf.with_df.auxmol
    opt = Int3c2eOpt(SortedMole.from_mol(mol, decontract=True),
                     auxmol_raw).build()
    aux = opt.auxmol
    offsets = np.append(0, np.cumsum(aux.l_ctr_counts))
    sorting = argsort_aux(offsets, aux.uniq_l_ctr)
    j2c = int2c2e(aux)
    w = cp.linalg.eigvalsh(j2c)
    cond = float(w[-1] / w[0])
    ctr = cp.asarray(aux.ctr_coeff)
    ctr_s = cp.empty_like(ctr)
    ctr_s[sorting] = ctr
    metric = ctr_s.dot(df_rhf_grad._gen_metric_solver(j2c, 'CD')(ctr_s.T))
    aux_coef, tag = DFMOD._decompose_j2c(aux, sorting)
    rel = (float(cp.abs(metric - aux_coef.dot(aux_coef.T)).max())
           / float(cp.abs(metric).max()))
    print(f'  {basis:12s} naux={aux.nao:5d} tag={tag} cond(j2c)={cond:.2e}',
          flush=True)
    check(f'  |metric - aux_coef @ aux_coef.T| / |metric|', rel, 1e-11)

# ---------------------------------------------------------------------------
print('\n=== 2. the per-atom derivative, patched against stock ===', flush=True)
CASES = [
    (ETHANOL, a.basis, 'b3lyp', 0, 0),
    (ETHANOL, 'def2-tzvpp', 'b3lyp', 0, 0),
    (ETHANOL, a.basis, 'pbe0', 0, 0),
    (ETHANOL, a.basis, 'm06-2x', 0, 0),
    (ETHANOL, a.basis, 'wb97x-v', 0, 0),     # range separated: two calls
    (CH3, a.basis, 'b3lyp', 1, 0),           # unrestricted
    (CH3, a.basis, 'wb97x-v', 1, 0),         # unrestricted, range separated
    (ETHANOL, a.basis, 'b3lyp', 1, 1),       # charged doublet
]
for atom, basis, xc, spin, charge in CASES:
    mol, mf = build(atom, basis, xc, spin, charge)
    dm = mf.make_rdm1()
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    g = mf.nuc_grad_method()
    # the module's path first: it needs the cderi, which the stock path
    # deletes on its first line
    fastdfgrad._CDERI = True
    before = fastdfgrad.stats()['cderi_reuse']
    got = g.jk_energy_per_atom(dm, j_factor=1, k_factor=hyb or 1., hermi=1)
    got_lr = None
    expect = 1
    if omega != 0 and alpha != hyb:
        # the long-range call.  It reaches for the long-range cderi the SCF
        # built under `with_df._rsh_df`, which upstream's `reset()` on the
        # first call would have thrown away -- `_release` is what keeps it
        got_lr = g.jk_energy_per_atom(dm, j_factor=0, k_factor=alpha - hyb,
                                      hermi=1, omega=omega)
        expect = 2
    reuse = fastdfgrad.stats()['cderi_reuse'] - before

    # GPU4PySCF's own function, built from scratch, on the same density
    def stock(om, jf, kf):
        with mol.with_range_coulomb(om), mf.with_df.auxmol.with_range_coulomb(om):
            opt = Int3c2eOpt(SortedMole.from_mol(mol, decontract=True),
                             mf.with_df.auxmol).build()
            mod = df_uhf_grad if spin else df_rhf_grad
            return mod._jk_energy_per_atom(opt, dm, jf, kf, hermi=1,
                                           auxbasis_response=True)

    ref = stock(0, 1, hyb or 1.)
    tag = f'{atom[:6].strip()}/{basis}/{xc}/spin{spin}/q{charge}'
    check(f'  ejk           {tag}', float(np.abs(got - ref).max()), a.tol)
    if got_lr is not None:
        ref_lr = stock(omega, 0, alpha - hyb)
        check(f'  ejk long-range{tag}',
              float(np.abs(got_lr - ref_lr).max()), a.lr_tol)
    if reuse != expect:
        FAIL.append(f'{tag}: reused a cderi on {reuse} of {expect} calls')
        print(f'  !! {tag}: {reuse}/{expect} calls reused a cderi -- '
              f'{fastdfgrad.stats()["declined_reasons"]}', flush=True)

# ---------------------------------------------------------------------------
print('\n=== 3. the whole gradient, and finite differences ===', flush=True)
mol, mf = build(WATER, a.basis, 'b3lyp')
g = mf.nuc_grad_method()
g.verbose = 0
fastdfgrad._CDERI = True
de_fast = g.kernel()
mf.with_df.build()
fastdfgrad._CDERI = False                    # GPU4PySCF's own first loop
de_stock = g.kernel()
fastdfgrad._CDERI = True
check('  whole gradient, patched against stock',
      float(np.abs(de_fast - de_stock).max()), a.tol)

if not a.no_fd:
    # A 3-point difference of the *density-fitted* energy, which is the
    # quantity this gradient is the derivative of.  It is the only check here
    # that does not take GPU4PySCF's answer as the definition of the right one.
    step = 2e-3
    coords = mol.atom_coords()
    de_num = np.zeros_like(coords)
    for iatm in range(mol.natm):
        for x in range(3):
            e2 = []
            for sign in (1, -1):
                c = coords.copy()
                c[iatm, x] += sign * step
                m = gto.M(atom=[(mol.atom_symbol(i), c[i]) for i in
                                range(mol.natm)],
                          unit='Bohr', basis=a.basis, verbose=0)
                mm = dft.RKS(m, xc='b3lyp').density_fit(auxbasis=a.auxbasis)
                mm.conv_tol = 1e-12
                mm.grids.level = 5
                e2.append(mm.kernel())
            de_num[iatm, x] = (e2[0] - e2[1]) / (2 * step)
    check('  patched against finite differences',
          float(np.abs(de_fast - de_num).max()), a.fd_tol)
    check('  stock   against finite differences',
          float(np.abs(de_stock - de_num).max()), a.fd_tol)
    print(f'  [the two analytic gradients differ from each other by '
          f'{float(np.abs(de_fast - de_stock).max()):.2e}]', flush=True)

# ---------------------------------------------------------------------------
print('\n=== 4. what the module declines ===', flush=True)
# A pure functional never reaches this function at all: `df/grad/rhf.py:68`
# sends a `k_factor == 0` call to `_j_energy_per_atom`, which builds no cderi.
# The check is that a DF-PBE gradient is still right and the module reports
# no reuse rather than quietly claiming one.
before = fastdfgrad.stats()['cderi_reuse']
mol, mf = build(WATER, a.basis, 'pbe')
de_pbe = mf.nuc_grad_method().kernel()
after = fastdfgrad.stats()['cderi_reuse']
check('  DF-PBE gradient reaches no cderi reuse', float(after - before), 0.)
print(f'  [|de| max {np.abs(de_pbe).max():.6f}]', flush=True)

# `only_dfj` has no cderi at all (docs/README_fastdfj.md §6), so the module
# must hand the call back rather than produce something from nothing.
electrolites.patch('fastdfj')
mol, mf = build(WATER, a.basis, 'b3lyp')
mf2 = dft.RKS(mol, xc='b3lyp').density_fit(only_dfj=True)
mf2.conv_tol = 1e-11
mf2.grids.level = 3
mf2.kernel()
before = fastdfgrad.stats()['declined']
de_dfj = mf2.nuc_grad_method().kernel()
declined = fastdfgrad.stats()['declined'] - before
check('  only_dfj gradient is declined, not answered',
      float(declined == 0), 0.)
print(f'  [|de| max {np.abs(de_dfj).max():.6f}]', flush=True)

# ---------------------------------------------------------------------------
print('\n=== 5. the store W10 put the (naux, nocc, nocc) array in ===',
      flush=True)
# `_OOStore` is the only thing in the module that knows which side of the PCIe
# link that array is on, and every one of its transfers is raw pointer
# arithmetic -- one `cudaMemcpy` per spin for an auxiliary row slab and one
# `cudaMemcpy2D` per spin for a slice of the occupied index.  Nothing about a
# wrong offset there shows up as anything but a wrong gradient, and the shapes
# that break it are the ones the molecules below do not produce: two spin
# channels, and a store whose allocated row count is larger than the number of
# rows actually written (which is what the fitting dimension of the cderi is,
# whenever the auxiliary basis is Cartesian on one side and spherical on the
# other).  So the round trip is checked directly first.
rng = np.random.default_rng(0)
for nset, nrows, used, nocc in ((1, 7, 7, 5), (2, 7, 7, 5), (2, 9, 6, 4),
                                (1, 4, 3, 11)):
    ref = cp.asarray(rng.standard_normal((nset, used, nocc, nocc)))
    for host in (False, True):
        st = fastdfgrad._OOStore(nset, nrows, nocc, host)
        for a0 in range(0, used, 2):                   # written in row slabs
            a1 = min(used, a0 + 2)
            slab = st.out_rows(a0, a1, cp.empty(nset * 2 * nocc * nocc))
            slab[:] = ref[:, a0:a1]
            st.commit_rows(a0, a1, slab)
        back = cp.empty((nset, used, nocc, nocc))
        for a0 in range(0, used, 3):                   # read back in others
            a1 = min(used, a0 + 3)
            back[:, a0:a1] = st.read_rows(a0, a1, cp.empty(nset * 3 * nocc * nocc))
        e_rows = float(cp.abs(back - ref).max())
        cols = cp.empty((nset, used, nocc, nocc))
        for c0 in range(0, nocc, 2):                   # and in column slices
            c1 = min(nocc, c0 + 2)
            cols[:, :, c0:c1] = st.read_cols(
                c0, c1, used, cp.empty(nset * used * 2 * nocc))
        e_cols = float(cp.abs(cols - ref).max())
        st.close()
        check(f'  round trip nset={nset} rows={used}/{nrows} nocc={nocc} '
              f'{"host" if host else "device"}', max(e_rows, e_cols), 0.)

# and then the gradient itself, through every combination of the two knobs.
# The `j3c` rows are our batched clone of GPU4PySCF's own first loop: it is the
# same arithmetic in the same order, so it is checked against a much tighter
# tolerance than the reuse is -- anything above round-off there is a batching
# bug, not a conditioning one.
STORE_CASES = [
    ('device, one slice',    dict(_OOSTORE='device', _OOSLICE=0)),
    ('device, 3 columns',    dict(_OOSTORE='device', _OOSLICE=3)),
    ('host, one slice',      dict(_OOSTORE='host', _OOSLICE=0)),
    ('host, 2 columns',      dict(_OOSTORE='host', _OOSLICE=2)),
    ('host, no prefetch',    dict(_OOSTORE='host', _OOSLICE=2,
                                  _OOPREFETCH=False)),
    ('host, unpinned',       dict(_OOSTORE='host', _OOSLICE=2,
                                  _OOPIN='none')),
    ('host, cudaHostAlloc',  dict(_OOSTORE='host', _OOSLICE=0,
                                  _OOPIN='alloc')),
]
DEFAULTS = dict(_OOSTORE='auto', _OOSLICE=0, _OOPREFETCH=True,
                _OOPIN='register', _J3CBATCH='auto', _CDERI=True)


def with_flags(**kw):
    for k, v in {**DEFAULTS, **kw}.items():
        setattr(fastdfgrad, k, v)


for atom, spin, tag in ((ETHANOL, 0, 'ethanol'), (CH3, 1, 'CH3 doublet')):
    mol, mf = build(atom, a.basis, 'b3lyp', spin)
    dm = mf.make_rdm1()
    g = mf.nuc_grad_method()
    with mol.with_range_coulomb(0), mf.with_df.auxmol.with_range_coulomb(0):
        opt = Int3c2eOpt(SortedMole.from_mol(mol, decontract=True),
                         mf.with_df.auxmol).build()
        mod = df_uhf_grad if spin else df_rhf_grad
        ref = mod._jk_energy_per_atom(opt, dm, 1, 1., hermi=1,
                                      auxbasis_response=True)
    for name, flags in STORE_CASES:
        for route, extra, tol in (('reuse', dict(), a.tol),
                                  ('j3c  ', dict(_CDERI=False,
                                                 _J3CBATCH='always'), 1e-11)):
            with_flags(**flags, **extra)
            if mf.with_df._cderi is None:
                mf.with_df.build()
            got = g.jk_energy_per_atom(dm, j_factor=1, k_factor=1., hermi=1)
            check(f'  {route} {name:20s} {tag}',
                  float(np.abs(got - ref).max()), tol)
    with_flags()

# the halve-and-retry.  The shape that actually makes `cutensor.createPlan`
# answer CUTENSOR_STATUS_NOT_SUPPORTED is a 284-atom one -- W10 made it
# reachable by freeing the array GPU4PySCF used to reserve for, which lets the
# ip1 loop pick a bigger block than upstream would have -- so the failure is
# injected here instead.  What is checked is that the retry restarts the loop
# cleanly: `ejk` must not keep the half-accumulated contributions of the
# attempt that failed, which is the only way this can go wrong quietly.
with_flags()
mol, mf = build(ETHANOL, a.basis, 'b3lyp')
dm = mf.make_rdm1()
g = mf.nuc_grad_method()
ref_retry = g.jk_energy_per_atom(dm, j_factor=1, k_factor=1., hermi=1)
_real_contract = fastdfgrad.contract
_fired = []


def _flaky(pattern, *args, **kw):
    if pattern == 'rji,qj->iqr' and len(_fired) < 2:
        _fired.append(pattern)
        raise cp.cuda.memory.OutOfMemoryError(1 << 40, 1 << 30)
    return _real_contract(pattern, *args, **kw)


mf.with_df.build()
before = fastdfgrad.stats()['plan_retries']
fastdfgrad.contract = _flaky
try:
    got = g.jk_energy_per_atom(dm, j_factor=1, k_factor=1., hermi=1)
finally:
    fastdfgrad.contract = _real_contract
retries = fastdfgrad.stats()['plan_retries'] - before
check('  the ip1 loop restarts cleanly when a plan is refused',
      float(np.abs(got - ref_retry).max()), a.tol)
check(f'  ... and it took {retries} retries to get there',
      float(retries != 2), 0.)

# the threshold itself: `auto` has to put a small system on the card and a
# large one on the host, and `stats()` has to say which, because nothing
# numerical can.  Forcing `_OOFRAC` to zero is the same question the 284-atom
# case asks by being 32 GB.
with_flags()
mol, mf = build(ETHANOL, a.basis, 'b3lyp')
dm = mf.make_rdm1()
g = mf.nuc_grad_method()
before = fastdfgrad.stats()
g.jk_energy_per_atom(dm, j_factor=1, k_factor=1., hermi=1)
mid = fastdfgrad.stats()
check('  auto keeps a 9-atom array on the card',
      float(mid['oo_device'] - before['oo_device'] != 1), 0.)
fastdfgrad._OOFRAC = 0.
mf.with_df.build()
g.jk_energy_per_atom(dm, j_factor=1, k_factor=1., hermi=1)
after = fastdfgrad.stats()
check('  auto sends an array it cannot fit to the host',
      float(after['oo_host'] - mid['oo_host'] != 1), 0.)
fastdfgrad._OOFRAC = float(os.environ.get('FASTDFGRAD_OOFRAC', 0.8))

print(f'\nstats: {fastdfgrad.stats()}', flush=True)
if FAIL:
    print(f'\n{len(FAIL)} FAILED: {FAIL}')
    sys.exit(1)
print('\nall checks passed')
