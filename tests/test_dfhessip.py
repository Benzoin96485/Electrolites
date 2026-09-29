"""Do W16, W17 and W18 compute the same Hessian terms as GPU4PySCF?

``docs/DF_WORK_LIST.md`` **W16** (``fastxcfxc``'s binding of
``hessian/rks.py:nr_rks_fxc_mo``) and **W17**/**W18**
(``electrolites/fastdfhessip.py``: ``df/hessian/rhf.py:_jk_energy_per_atom``
and ``_get_veff``).  All three are claimed to be the same arithmetic in a
different arrangement, so the bar is the repository's usual one: element-wise
against GPU4PySCF's own function, in one process, on one converged SCF.

Checked:

* **ip2 and h1** for a hybrid (B3LYP's ``j = 1, k = 0.2``), for Hartree-Fock
  factors (``k = 1``), for a pure functional's Coulomb-only terms
  (``k = 0``, which ip2 serves from ``_j_energy_per_atom``), and for the
  long-range operator (``omega = 0.3``) that a range-separated functional
  adds as a second call;
* **every switch**, each alone, so an ablation column in the write-up is an
  ablation of a correct code path;
* **the paths upstream only takes when memory is short** -- a second pass
  split over the metric dimension, which also turns item 3's fusion off, by
  capping the memory the module sees (``FASTDFHESSIP_MEM``) and, with the
  first pass fused into it, by capping the chunk (``FASTDFHESSIP_VCHUNK``);
  and several
  occupied batches with ``dm_oo`` on the host, by forcing the batch
  (``FASTDFHESSIP_OCC_BATCH``);
* **the CPHF grid half** for LDA, GGA and meta-GGA at one, three and six
  right-hand sides, with W33's occupied-side route on every block, on the
  default share of them and on none, and that a density without the CPHF's tags is handed
  back;
* **a whole Hessian**, patched against unpatched.

    python tests/test_dfhessip.py [--basis def2-tzvpp] [--no-hessian]
"""
import argparse
import os
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--tol', type=float, default=1e-10,
                help='bar on the largest difference relative to the largest '
                     'element GPU4PySCF returns')
ap.add_argument('--tol-hessian', type=float, default=1e-8,
                help='bar on the largest Hessian element, in Eh/bohr^2')
ap.add_argument('--no-hessian', action='store_true')
a = ap.parse_args()

import cupy as cp                                            # noqa: E402
from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.lib.cupy_helper import (                      # noqa: E402
    contract, tag_array, transpose_sum)
from gpu4pyscf.df.hessian import rhf as DFHESS               # noqa: E402
from gpu4pyscf.hessian import rks as RKSHESS                 # noqa: E402
from gpu4pyscf.hessian.rhf import _ao2mo                     # noqa: E402

import electrolites                                          # noqa: E402
electrolites.patch('fastxcfxc', 'fastdfhessip')
from electrolites import fastdfhessip, fastxcfxc             # noqa: E402

UP_IP2 = DFHESS._jk_energy_per_atom_orig
UP_H1 = DFHESS._get_veff_orig
UP_FXC = RKSHESS._nr_rks_fxc_mo_orig

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
        print(f'  FAIL {label:60s} shape {got.shape} != {ref.shape}')
        return
    scale = float(cp.abs(ref).max()) or 1.0
    err = float(cp.abs(got - ref).max()) / scale
    ok = err <= tol
    if not ok:
        fails.append(f'{label}: {err:.3e} > {tol:.1e}')
    print(f'  {"ok  " if ok else "FAIL"} {label:60s} {err:.3e}'
          f'{"" if ok else f"   > {tol:.1e}"}', flush=True)


class env:
    def __init__(self, **kw):
        self.kw = {f'FASTDFHESSIP_{k}': str(v) for k, v in kw.items()}

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.kw}
        os.environ.update(self.kw)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def scf(atom, xc, basis=None):
    mol = gto.M(atom=atom, basis=basis or a.basis, verbose=0)
    mf = dft.RKS(mol, xc=xc).density_fit()
    mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.kernel()
    if mf.with_df.intopt is None:
        mf.with_df.build(build_cderi=False)
    return mf


#: (label, switches) -- the default first, then each item off on its own
SWITCHES = [('default', {}),
            ('SPMULT=4', {'SPMULT': 4}),
            ('OOSUM=transpose', {'OOSUM': 'transpose'}),
            ('OOSUM=gemm', {'OOSUM': 'gemm'}),
            ('FUSE=0', {'FUSE': 0}),
            ('H1J=upstream', {'H1J': 'upstream'}),
            ('H1ATOM=upstream', {'H1ATOM': 'upstream'}),
            ('H1ATOM=batched', {'H1ATOM': 'batched'}),
            ('H1ATOM=occ', {'H1ATOM': 'occ'}),
            # W35's batched D2 (the default under H1ATOM=occ), W30's per-atom
            # loop, and item 5 off, which takes the per-atom loop too
            ('H1ATOM=occ H1D2=atom', {'H1ATOM': 'occ', 'H1D2': 'atom'}),
            ('H1D2=atom', {'H1D2': 'atom'}),
            ('H1ATOM=occ H1J=upstream', {'H1ATOM': 'occ', 'H1J': 'upstream'}),
            ('K0=compute', {'K0': 'compute'}),
            # W32's grouped stage C: W30's per-atom loop, one atom a group,
            # and groups of a few atoms
            ('H1C=atom', {'H1C': 'atom'}),
            ('H1C_ROWS=3', {'H1C_ROWS': 3}),
            ('H1C_ROWS=240', {'H1C_ROWS': 240}),
            ('JPAIR=0', {'JPAIR': 0}),
            ('METRIC=eigen', {'METRIC': 'eigen'}),
            ('TRI=solve', {'TRI': 'solve'}),
            ('SYRK=0', {'SYRK': 0}),
            ('D4=gemm', {'D4': 'gemm'}),
            ('T2SPH=0', {'T2SPH': 0}),
            ('CARVE=0', {'CARVE': 0}),
            # W29's launch: item 1's instead, a forced split of every
            # auxiliary batch (3 shells a range), one range, fixed pairs
            ('IP2_TILE=0', {'IP2_TILE': 0}),
            ('IP2_KCHUNK=3', {'IP2_KCHUNK': 3}),
            ('IP2_KCHUNK=0', {'IP2_KCHUNK': 0}),
            ('IP2_PCHUNK=5', {'IP2_PCHUNK': 5}),
            ('FILL_TILE=0', {'FILL_TILE': 0}),
            ('FILL_TILE=0.05', {'FILL_TILE': 0.05}),
            # W31's kernel: GPU4PySCF's instead, and W31's under the other
            # launches (every triple in a few-shell range; the long-range
            # operator's classes all take its shared-memory path)
            ('IP2_KERNEL=upstream', {'IP2_KERNEL': 'upstream'}),
            ('IP2_KCHUNK=3 IP2_PCHUNK=5', {'IP2_KCHUNK': 3, 'IP2_PCHUNK': 5}),
            # W34's buffer: upstream's size, the Gram form in chunks of a
            # few rows (both D4 forms), and the Gram form in one chunk
            ('WORK=upstream', {'WORK': 'upstream'}),
            ('D4_ROWS=7', {'D4_ROWS': 7}),
            ('D4_ROWS=7 D4=gemm', {'D4_ROWS': 7, 'D4': 'gemm'}),
            ('D4_ROWS=100000', {'D4_ROWS': 100000}),
            # W37's packed pairs in the h_aux block: off, in chunks of a
            # few pairs, and the same with the block's temporaries allocated
            ('PACK=0', {'PACK': 0}),
            ('D1D3=0', {'D1D3': 0}),
            ('D3SCATTER=0', {'D3SCATTER': 0}),
            ('TRMM=blocked TRMM_BLK=16', {'TRMM': 'blocked', 'TRMM_BLK': 16}),
            ('D5=atom', {'D5': 'atom'}),
            ('D5_GROUP=1', {'D5_GROUP': 1}),
            ('PACK_CHUNK=7', {'PACK_CHUNK': 7}),
            ('PACK_CHUNK=7 CARVE=0', {'PACK_CHUNK': 7, 'CARVE': 0})]


def run_terms(mf, tag, factors, switches=SWITCHES, mem=None):
    intopt = mf.with_df.intopt
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    for jf, kf, omega in factors:
        f = f'j={jf} k={kf}' + ('' if omega is None else f' omega={omega}')
        ref_h1 = UP_H1(intopt, mo_coeff, mo_occ, jf, kf, omega)
        # k = 0 is a pure functional's ip2 term, which both sides hand to
        # their own `_j_energy_per_atom`
        ref_ip2 = UP_IP2(intopt, dm0, jf, kf, omega)
        for lbl, sw in switches:
            extra = dict(sw)
            if mem is not None:
                extra['MEM'] = mem
            with env(**extra):
                got = fastdfhessip._get_veff(intopt, mo_coeff, mo_occ, jf, kf,
                                             omega)
                check(f'{tag} h1  {f} [{lbl}]', got, ref_h1)
                if ref_ip2 is not None:
                    got = fastdfhessip._jk_energy_per_atom(intopt, dm0, jf,
                                                           kf, omega)
                    check(f'{tag} ip2 {f} [{lbl}]', got, ref_ip2)


print(f'# gpu4pyscf {electrolites.gpu4pyscf_version()}  '
      f'patched {electrolites.patched()}  basis {a.basis}')

print('\n## ip2 and h1, ethanol')
mf = scf(ETHANOL, 'b3lyp')
run_terms(mf, 'ethanol', [(1., 0.2, None), (1., 1., None), (1., 0., None)])

print('\n## the long-range operator, water')
mfw = scf(H2O, 'b3lyp')
run_terms(mfw, 'water', [(0., 0.6, 0.3), (1., 0.2, None)])

print('\n## the paths taken when memory is short, ethanol')
# Upstream's sizing asserts when the cap is too small to hold even one
# block, so the cap is walked down a ladder and every rung that runs is
# checked; the section fails if, together with the two forced sweeps below,
# the rungs did not reach every batched path.
intopt = mf.with_df.intopt
mo_coeff = cp.asarray(mf.mo_coeff)
mo_occ = cp.asarray(mf.mo_occ)
dm0 = mf.make_rdm1(mo_coeff, mo_occ)
ref = {'ip2': UP_IP2(intopt, dm0, 1., 0.2), 'h1': UP_H1(intopt, mo_coeff,
                                                          mo_occ, 1., 0.2)}
reached = {'ip2': set(), 'h1': set()}
for mem, term, fuse in [(m, t, f) for m in 4e7 * 0.85 ** np.arange(16)
                        for t, f in (('ip2', 1), ('ip2', 0), ('h1', 1))]:
    fused0 = fastdfhessip.stats()['fused']
    try:
        with env(MEM=mem, FUSE=fuse):
            if term == 'ip2':
                got = fastdfhessip._jk_energy_per_atom(intopt, dm0, 1., 0.2)
            else:
                got = fastdfhessip._get_veff(intopt, mo_coeff, mo_occ, 1.,
                                             0.2)
    except (AssertionError, MemoryError, ZeroDivisionError):
        # upstream's own sizing gives up below one block (and divides by
        # zero once its word count goes negative); so does the copy
        continue
    st = fastdfhessip.stats()
    fused = st['fused'] > fused0
    shape = (f'occ_batches={st["occ_batches"]} vchunks={st["vchunks"]}'
             + (' fused' if fused else '') if term == 'ip2' else '')
    if term == 'h1' and st.get('h1_rebalanced'):
        # W35's item 21: the block's minimum taken from the batch
        reached['h1'].add('rebalanced')
        shape = f'blksize={st["h1_blksize"]}, rebalanced'
    check(f'ethanol {term} at {mem / 1e6:.1f} MB ({shape})', got, ref[term])
    if st['occ_batches'] > 1:
        reached[term].add('occ')
    if term == 'ip2' and st['vchunks'] > 1:
        reached[term].add('v fused' if fused else 'v')
# a multi-chunk second pass with the first pass fused into it, forced: the
# fusion's own memory margin keeps it off at every cap above
# W40's item 33 is the Cholesky factor in chunks of rows (the default
# wherever item 8's whole index does not fit); CROWS=0 is the chunked
# eigen-style arithmetic it replaces, D5_GROUP=8 and D5_ROWS=7 its D5 in
# many groups and row blocks; unfused, item 34 packs D5's partners
# (PACKROWS=0 does not)
CROWS = [('', {}), (' [CROWS=0]', {'CROWS': 0}),
         (' [D5_GROUP=8]', {'D5_GROUP': 8}),
         (' [D5_ROWS=7]', {'D5_ROWS': 7}),
         (' [PACKROWS=0]', {'PACKROWS': 0})]
for vchunk in (160, 64, 40):
    for fuse in (1, 0):
      for lbl, sw in (CROWS if fuse else CROWS[:2] + CROWS[3:]):
        fused0 = fastdfhessip.stats()['fused']
        rows0 = fastdfhessip.stats().get('chol_rows', 0)
        with env(VCHUNK=vchunk, FUSE=fuse, **sw):
            got = fastdfhessip._jk_energy_per_atom(intopt, dm0, 1., 0.2)
        st = fastdfhessip.stats()
        fused = st['fused'] > fused0
        rows = st.get('chol_rows', 0) > rows0
        if rows:
            reached['ip2'].add('chol rows')
        check(f'ethanol ip2 chunk {vchunk} (vchunks={st["vchunks"]}'
              f'{" fused" if fused else ""}{" rows" if rows else ""}){lbl}',
              got, ref['ip2'])
        if st['vchunks'] > 1:
            reached['ip2'].add('v fused' if fused else 'v')
# and the occupied batching, forced: ethanol's 13 occupied orbitals never
# need it at any cap its sizing survives
for nb in (8, 5):
    for term in ('ip2', 'h1', 'h1 occ'):
        with env(OCC_BATCH=nb, H1ATOM='occ' if term == 'h1 occ' else 'batched'):
            if term == 'ip2':
                got = fastdfhessip._jk_energy_per_atom(intopt, dm0, 1., 0.2)
            else:
                got = fastdfhessip._get_veff(intopt, mo_coeff, mo_occ, 1., 0.2)
                assert fastdfhessip.stats()['h1_occ'] == (term == 'h1 occ')
        st = fastdfhessip.stats()
        check(f'ethanol {term} {nb} occupied per batch '
              f'(occ_batches={st["occ_batches"]})', got, ref[term[:3].strip()])
        if st['occ_batches'] > 1:
            reached[term[:3].strip()].add('occ')
# W35: the occupied form's D loop at small auxiliary blocks, whose last
# block in a batch is shorter (item 20 relays it contiguously), batched D2
# and W30's per-atom loop, with and without the occupied index batched
for blk in (40, 24):
    for lbl, sw in (('', {}), (' H1D2=atom', {'H1D2': 'atom'}),
                    (' 5 occupied per batch', {'OCC_BATCH': 5})):
        with env(H1ATOM='occ', H1BLK=blk, **sw):
            got = fastdfhessip._get_veff(intopt, mo_coeff, mo_occ, 1., 0.2)
            st = fastdfhessip.stats()
            assert st['h1_blksize'] == blk and st['h1_occ']
            assert st['h1_d2_batched'] == ('H1D2' not in sw)
        check(f'ethanol h1 occ, blocks of {blk}{lbl}', got, ref['h1'])
# the memory-short exchange branch (the per-atom AO matrices do not fit),
# forced: no cap the sizing survives reaches it on ethanol.  W32's grouped
# stage C has its own form of it
for lbl, sw in (('', {}), (' [H1C=atom]', {'H1C': 'atom'}),
                (' [H1C_ROWS=240]', {'H1C_ROWS': 240}),
                (' 5 occupied per batch', {'OCC_BATCH': 5}),
                # W35's item 20 is this branch's D2 too: W30's per-atom
                # loop, and blocks with a short last one
                (' [H1D2=atom]', {'H1D2': 'atom'}),
                (' blocks of 24', {'H1BLK': 24}),
                (' blocks of 24, 5 occupied per batch',
                 {'H1BLK': 24, 'OCC_BATCH': 5})):
    with env(H1KAO=0, **sw):
        got = fastdfhessip._get_veff(intopt, mo_coeff, mo_occ, 1., 0.2)
        assert fastdfhessip.stats()['h1_d2_batched'] == ('H1D2' not in sw)
    if fastdfhessip.stats()['h1_kocc']:
        reached['h1'].add('k occ')
    check(f'ethanol h1, exchange on the occupied side{lbl}', got, ref['h1'])
ok = (reached['ip2'] == {'occ', 'v', 'v fused', 'chol rows'} and
      reached['h1'] == {'occ', 'k occ', 'rebalanced'})
if not ok:
    fails.append(f'memory cap did not reach the batched paths: {reached}')
print(f'  {"ok  " if ok else "FAIL"} {"the ladder reached the batched paths":60s}'
      f' {reached}')

print('\n## the CPHF grid half (W16)')


def cphf_density(mf, n_dm, seed=0):
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    orbo = mo_coeff[:, mo_occ > 0]
    nmo = mo_coeff.shape[1]
    nocc = orbo.shape[1]
    rng = np.random.default_rng(seed)
    mo1 = cp.asarray(rng.standard_normal((n_dm, nmo, nocc)) * 1e-2)
    mo1_ao = contract('npo,ip->nio', mo1, mo_coeff)
    dm1 = cp.asarray(contract('npi,qi->npq', mo1_ao, orbo * 2), order='C')
    transpose_sum(dm1, inplace=True, hermi=1)
    return tag_array(dm1, mo1=mo1_ao, occ_coeff=orbo, symmetrize=1)


for xc in ('svwn', 'b3lyp', 'tpss', 'r2scan'):
    mfx = scf(ETHANOL, xc)
    ni = mfx._numint
    mo_coeff = cp.asarray(mfx.mo_coeff)
    mo_occ = cp.asarray(mfx.mo_occ)
    grids = mfx.grids
    rho0, vxc, fxc = ni.cache_xc_kernel(mfx.mol, grids, xc, mo_coeff, mo_occ,
                                        0)
    orbo = mo_coeff[:, mo_occ > 0]
    for n_dm in (1, 3, 6):
        dm1 = cphf_density(mfx, n_dm)
        up = UP_FXC(ni, mfx.mol, grids, xc, None, dm1, mo_coeff, 0, 1,
                    rho0, vxc, fxc, max_memory=None)
        got = fastxcfxc._nr_rks_fxc_mo(ni, mfx.mol, grids, xc, None, dm1,
                                       mo_coeff, 0, 1, rho0, vxc, fxc,
                                       max_memory=None)
        # The reference is GPU4PySCF's *AO-basis* `nr_rks_fxc` on the same
        # density matrix, taken to the MO basis with upstream's own `_ao2mo`.
        # Upstream's `nr_rks_fxc_mo` is not used as the reference for
        # meta-GGA because it is wrong there: its `fxc` contraction is
        # `contract('axg,xyg->ayg', rho1, fxc_w, out=rho1)`, an in-place
        # cuTENSOR call that is exact at ncomp = 4 and not at ncomp = 5
        # (docs/README_fastdfhessip.md 3.2).
        plain = cp.asarray(dm1).view(cp.ndarray).copy()
        ref = _ao2mo(fastxcfxc.NI._nr_rks_fxc_orig(
            ni, mfx.mol, grids, xc, None, plain, 0, 1, rho0, vxc, fxc,
            max_memory=None), orbo, mo_coeff)
        check(f'nr_rks_fxc_mo {xc:6s} n_dm={n_dm} vs AO path', got, ref)
        os.environ['FASTXCFXC_FACTORS'] = '0'
        got0 = fastxcfxc._nr_rks_fxc_mo(ni, mfx.mol, grids, xc, None, dm1,
                                        mo_coeff, 0, 1, rho0, vxc, fxc,
                                        max_memory=None)
        os.environ.pop('FASTXCFXC_FACTORS')
        check(f'nr_rks_fxc_mo {xc:6s} n_dm={n_dm} [FACTORS=0] vs AO path',
              got0, ref)
        # W33: every block on the occupied side, and none of them
        for label, env in (('OCC ratio 0', {'FASTXCFXC_OCC_RATIO': '0',
                                            'FASTXCFXC_OCC_RATIO_GGA': '0'}),
                           ('OCC=0', {'FASTXCFXC_OCC': '0'})):
            os.environ.update(env)
            got1 = fastxcfxc._nr_rks_fxc_mo(ni, mfx.mol, grids, xc, None,
                                            dm1, mo_coeff, 0, 1, rho0, vxc,
                                            fxc, max_memory=None)
            for k in env:
                os.environ.pop(k)
            check(f'nr_rks_fxc_mo {xc:6s} n_dm={n_dm} [{label}] vs AO path',
                  got1, ref)
        if xc not in ('tpss', 'r2scan'):
            check(f'nr_rks_fxc_mo {xc:6s} n_dm={n_dm} vs upstream', got, up)
        else:
            sc = float(cp.abs(ref).max())
            print(f'  info upstream nr_rks_fxc_mo {xc} n_dm={n_dm} vs AO path '
                  f'{float(cp.abs(up - ref).max()) / sc:.3e}  (the in-place '
                  f'contraction)')

before = fastxcfxc.stats()['stock_mo']
plain = cp.asarray(cphf_density(mfx, 3)).view(cp.ndarray)
try:
    fastxcfxc._nr_rks_fxc_mo(ni, mfx.mol, grids, 'tpss', None, plain,
                             mo_coeff, 0, 1, rho0, vxc, fxc, max_memory=None)
except Exception:                                             # noqa: BLE001
    pass
declined = fastxcfxc.stats()['stock_mo'] - before
ok = declined == 1
if not ok:
    fails.append(f'untagged density: declined {declined} times, expected 1')
print(f'  {"ok  " if ok else "FAIL"} '
      f'{"an untagged density is handed back to GPU4PySCF":60s} '
      f'declined={declined}')
nfb = fastxcfxc.stats().get('factor_blocks', 0)
if nfb == 0:
    fails.append('the factor route was never taken')
print(f'  {"ok  " if nfb else "FAIL"} '
      f'{"the factor route was taken on some blocks":60s} {nfb} blocks')
nob = fastxcfxc.stats().get('occ_blocks', 0)
if nob == 0:
    fails.append('the occupied-side route was never taken')
print(f'  {"ok  " if nob else "FAIL"} '
      f'{"the occupied-side route (W33) was taken on some blocks":60s} '
      f'{nob} blocks')

if not a.no_hessian:
    print('\n## a whole Hessian, patched against unpatched')
    molh = gto.M(atom=ETHANOL, basis=a.basis, verbose=0)
    hs = {}
    for label, off in (('stock', True), ('patched', False)):
        saved_fxc = RKSHESS.nr_rks_fxc_mo
        if off:
            os.environ['FASTDFHESSIP_OFF'] = '1'
            RKSHESS.nr_rks_fxc_mo = UP_FXC
        mfh = dft.RKS(molh, xc='b3lyp').density_fit()
        mfh.grids.level = 3
        mfh.conv_tol = 1e-11
        mfh.conv_tol_cpscf = 1e-8
        mfh.kernel()
        hs[label] = np.asarray(mfh.Hessian().kernel())
        os.environ.pop('FASTDFHESSIP_OFF', None)
        RKSHESS.nr_rks_fxc_mo = saved_fxc
    err = np.abs(hs['patched'] - hs['stock']).max()
    ok = err <= a.tol_hessian
    if not ok:
        fails.append(f'ethanol Hessian: {err:.3e}')
    print(f'  {"ok  " if ok else "FAIL"} '
          f'{"ethanol B3LYP Hessian":60s} {err:.3e} Eh/bohr^2')
    print(f'  stats: {fastdfhessip.stats()}  {fastxcfxc.stats()}')

print()
if fails:
    print(f'{len(fails)} FAILURES')
    for f in fails:
        print(f'  {f}')
    sys.exit(1)
print('all ok')
