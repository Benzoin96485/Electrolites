"""Do W25's unrestricted twins compute the same Hessian terms as GPU4PySCF?

``docs/DF_WORK_LIST.md`` **W25** (``electrolites/_dfhessip_u.py``:
``df/hessian/uhf.py:_jk_energy_per_atom`` and ``_get_veff``).  The same bar
as ``tests/test_dfhessip.py``: element-wise against GPU4PySCF's own function,
in one process, on one converged open-shell SCF.

Checked:

* **ip2 and h1** for a hybrid (B3LYP's ``j = 1, k = 0.2``), Hartree-Fock
  factors (``k = 1``), a pure functional's Coulomb-only terms (``k = 0``,
  which ip2 hands to the restricted ``_j_energy_per_atom`` on the total
  density) and the long-range operator (``omega = 0.3``), on a doublet and
  on a triplet -- where ``nocca - noccb = 2`` and the padded beta columns
  are two;
* **every switch**, each alone;
* **the memory-short paths**: a second pass split over the metric
  dimension, with and without the first pass fused into it, by capping the
  chunk (``FASTDFHESSIP_VCHUNK``); the ``mem_sufficient = False`` branches of
  h1 and the unfused plan of ip2, by capping the memory the module sees
  (``FASTDFHESSIP_MEM``);
* **the cuBLAS route** item 13 takes for a contraction cuTENSOR refuses,
  with the refusal simulated;
* **a whole UKS Hessian**, patched against unpatched.

    python tests/test_dfuhessip.py [--basis def2-tzvpp] [--no-hessian]
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
from gpu4pyscf.df.hessian import uhf as DFUHESS              # noqa: E402

import electrolites                                          # noqa: E402
electrolites.patch('fastxcfxc', 'fastdfhessip')
from electrolites import fastdfhessip, _dfhessip_u as U      # noqa: E402

UP_IP2 = DFUHESS._jk_energy_per_atom_orig
UP_H1 = DFUHESS._get_veff_orig

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
    if isinstance(ref, (list, tuple)):
        for s, (g, r) in enumerate(zip(got, ref)):
            check(f'{label} {"ab"[s]}', g, r, tol)
        return
    got = cp.asarray(got)
    ref = cp.asarray(ref)
    if got.shape != ref.shape:
        fails.append(f'{label}: shape {got.shape} != {ref.shape}')
        print(f'  FAIL {label:62s} shape {got.shape} != {ref.shape}')
        return
    scale = float(cp.abs(ref).max()) or 1.0
    err = float(cp.abs(got - ref).max()) / scale
    ok = err <= tol
    if not ok:
        fails.append(f'{label}: {err:.3e} > {tol:.1e}')
    print(f'  {"ok  " if ok else "FAIL"} {label:62s} {err:.3e}'
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


def scf(atom, xc, charge, spin, basis=None):
    mol = gto.M(atom=atom, basis=basis or a.basis, charge=charge, spin=spin,
                verbose=0)
    mf = dft.UKS(mol, xc=xc).density_fit()
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
            ('D4=gemm', {'D4': 'gemm'}),
            ('FUSE=0', {'FUSE': 0}),
            ('CARVE=0', {'CARVE': 0}),
            ('SYRK=0', {'SYRK': 0}),
            ('H1J=upstream', {'H1J': 'upstream'}),
            ('H1ATOM=upstream', {'H1ATOM': 'upstream'}),
            ('H1ATOM=batched', {'H1ATOM': 'batched'}),
            ('H1ATOM=occ', {'H1ATOM': 'occ'}),
            ('K0=compute', {'K0': 'compute'}),
            # W32's grouped stage C: W30's per-atom loop, one atom a group,
            # and groups of a few atoms
            ('H1C=atom', {'H1C': 'atom'}),
            ('H1C_ROWS=3', {'H1C_ROWS': 3}),
            ('H1C_ROWS=240', {'H1C_ROWS': 240}),
            ('JPAIR=0', {'JPAIR': 0}),
            ('IP2_TILE=0', {'IP2_TILE': 0}),
            ('IP2_KCHUNK=3', {'IP2_KCHUNK': 3}),
            ('FILL_TILE=0', {'FILL_TILE': 0}),
            ('FILL_TILE=0.05', {'FILL_TILE': 0.05}),
            ('IP2_KERNEL=upstream', {'IP2_KERNEL': 'upstream'})]


def terms(mf):
    intopt = mf.with_df.intopt
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    return intopt, mo_coeff, mo_occ, dm0


def run_terms(mf, tag, factors, switches=SWITCHES):
    intopt, mo_coeff, mo_occ, dm0 = terms(mf)
    for jf, kf, omega in factors:
        f = f'j={jf} k={kf}' + ('' if omega is None else f' omega={omega}')
        ref_h1 = UP_H1(intopt, mo_coeff, mo_occ, jf, kf, omega)
        ref_ip2 = UP_IP2(intopt, dm0, jf, kf, omega)
        for lbl, sw in switches:
            with env(**sw):
                got = U._get_veff(intopt, mo_coeff, mo_occ, jf, kf, omega)
                check(f'{tag} h1  {f} [{lbl}]', got, ref_h1)
                got = U._jk_energy_per_atom(intopt, dm0, jf, kf, omega)
                check(f'{tag} ip2 {f} [{lbl}]', got, ref_ip2)


print(f'# gpu4pyscf {electrolites.gpu4pyscf_version()}  '
      f'patched {electrolites.patched()}  basis {a.basis}')

print('\n## ip2 and h1, the ethanol radical cation (doublet)')
mf = scf(ETHANOL, 'b3lyp', 1, 1)
run_terms(mf, 'ethanol+', [(1., 0.2, None), (1., 1., None), (1., 0., None)])

print('\n## a triplet, ethanol dication, and the long-range operator')
mft = scf(ETHANOL, 'b3lyp', 2, 2)
run_terms(mft, 'ethanol2+ T', [(1., 0.2, None), (0., 0.6, 0.3)],
          switches=SWITCHES[:1] + [('OOSUM=transpose',
                                    {'OOSUM': 'transpose'})])

print('\n## the water cation, the long-range operator')
mfw = scf(H2O, 'b3lyp', 1, 1)
run_terms(mfw, 'water+', [(0., 0.6, 0.3)], switches=SWITCHES[:5])

print('\n## the paths taken when memory is short, ethanol+')
intopt, mo_coeff, mo_occ, dm0 = terms(mf)
ref = {'ip2': UP_IP2(intopt, dm0, 1., 0.2),
       'h1': UP_H1(intopt, mo_coeff, mo_occ, 1., 0.2)}
reached = set()
for vchunk in (160, 64):
    for fuse in (1, 0):
        fused0 = fastdfhessip.stats()['fused']
        with env(VCHUNK=vchunk, FUSE=fuse):
            got = U._jk_energy_per_atom(intopt, dm0, 1., 0.2)
        st = fastdfhessip.stats()
        fused = st['fused'] > fused0
        check(f'ethanol+ ip2 chunk {vchunk} (vchunks={st["vchunks"]}'
              f'{" fused" if fused else ""})', got, ref['ip2'])
        if st['vchunks'] > 1:
            reached.add('v fused' if fused else 'v')
# Upstream's sizing asserts when the cap is too small to hold one block, so
# the cap is walked down a ladder and every rung that runs is checked.  The
# h1 rungs reach mem_sufficient = False once the per-atom AO matrices stop
# fitting; the ip2 rungs reach the unfused plan
for mem in 6e7 * 0.8 ** np.arange(14):
    for term in ('ip2', 'h1'):
        fused0 = fastdfhessip.stats()['fused']
        try:
            with env(MEM=mem):
                if term == 'ip2':
                    got = U._jk_energy_per_atom(intopt, dm0, 1., 0.2)
                else:
                    got = U._get_veff(intopt, mo_coeff, mo_occ, 1., 0.2)
        except (AssertionError, MemoryError, ZeroDivisionError):
            continue
        st = fastdfhessip.stats()
        extra = ''
        if term == 'ip2':
            fused = st['fused'] > fused0
            extra = f'vchunks={st["vchunks"]}{" fused" if fused else ""}'
            if not fused:
                reached.add('unfused by memory')
        check(f'ethanol+ {term} at {mem / 1e6:.1f} MB ({extra})', got,
              ref[term])
nao = mf.mol.nao
natm = mf.mol.natm
# h1's mem_sufficient test is natm*3*2*nao^2 < 0.6 words: a cap below that
# (plus the rest of its budget) reaches the branch without the AO matrices
cap = natm * 3 * 2 * nao**2 * 8 / .6 * 0.9 + 2e7
try:
    with env(MEM=cap):
        got = U._get_veff(intopt, mo_coeff, mo_occ, 1., 0.2)
    check(f'ethanol+ h1 without the per-atom AO matrices ({cap / 1e6:.1f} MB)',
          got, ref['h1'])
    reached.add('h1 mem_insufficient')
    for lbl, sw in (('H1C=atom', {'H1C': 'atom'}),
                    ('H1C_ROWS=240', {'H1C_ROWS': 240})):
        with env(MEM=cap, **sw):
            got = U._get_veff(intopt, mo_coeff, mo_occ, 1., 0.2)
        check(f'ethanol+ h1 without the per-atom AO matrices [{lbl}]', got,
              ref['h1'])
except (AssertionError, MemoryError):
    pass
# and forced, the grouped form with the rest of the memory it would have
with env(H1KAO=0):
    got = U._get_veff(intopt, mo_coeff, mo_occ, 1., 0.2)
check('ethanol+ h1, exchange on the occupied side (forced)', got, ref['h1'])
ok = {'v', 'v fused', 'h1 mem_insufficient'} <= reached
if not ok:
    fails.append(f'memory caps did not reach the batched paths: {reached}')
print(f'  {"ok  " if ok else "FAIL"} {"the caps reached the batched paths":62s}'
      f' {sorted(reached)}')

print('\n## the cuBLAS route for outputs cuTENSOR refuses, ethanol+')
# At 113 atoms cuTENSOR refuses four of these contractions (their outputs
# pass 2**29 elements); here the refusal is simulated so the route is run
from electrolites import _cuplan                             # noqa: E402
_planable = _cuplan.planable
REFUSE = ('uv,unij->vnij', 'uv,vnij->unij', 'rs,npir->snpi',
          'xuv,vij->xuij')
_cuplan.planable = lambda pattern, *ops, **kw: (
    False if pattern in REFUSE else _planable(pattern, *ops, **kw))
n0 = fastdfhessip.stats().get('plan_gemm', 0)
for jf, kf in ((1., 0.2), (1., 1.)):
    got = U._jk_energy_per_atom(intopt, dm0, jf, kf)
    check(f'ethanol+ ip2 j={jf} k={kf} through cuBLAS', got,
          UP_IP2(intopt, dm0, jf, kf))
_cuplan.planable = _planable
n1 = fastdfhessip.stats().get('plan_gemm', 0)
ok = n1 - n0 >= 2 * 4
if not ok:
    fails.append(f'the cuBLAS route was taken {n1 - n0} times')
print(f'  {"ok  " if ok else "FAIL"} {"the cuBLAS route was taken":62s} '
      f'{n1 - n0} times')

if not a.no_hessian:
    print('\n## a whole UKS Hessian, patched against unpatched')
    molh = gto.M(atom=ETHANOL, basis=a.basis, charge=1, spin=1, verbose=0)
    hs = {}
    st0 = fastdfhessip.stats()
    for label, off in (('stock', True), ('patched', False)):
        if off:
            os.environ['FASTDFHESSIP_OFF'] = '1'
        mfh = dft.UKS(molh, xc='b3lyp').density_fit()
        mfh.grids.level = 3
        mfh.conv_tol = 1e-11
        mfh.conv_tol_cpscf = 1e-8
        mfh.kernel()
        hs[label] = np.asarray(mfh.Hessian().kernel())
        os.environ.pop('FASTDFHESSIP_OFF', None)
    err = np.abs(hs['patched'] - hs['stock']).max()
    ok = err <= a.tol_hessian
    if not ok:
        fails.append(f'ethanol+ Hessian: {err:.3e}')
    print(f'  {"ok  " if ok else "FAIL"} '
          f'{"ethanol+ UKS B3LYP Hessian":62s} {err:.3e} Eh/bohr^2')
    print(f'  stats: {fastdfhessip.stats()}')
    st = fastdfhessip.stats()
    ok = (st.get('ip2_u', 0) > st0.get('ip2_u', 0) and
          st.get('h1_u', 0) > st0.get('h1_u', 0))
    if not ok:
        fails.append(f'the UKS Hessian did not reach the twins: {st}')
    print(f'  {"ok  " if ok else "FAIL"} {"the UKS Hessian reached both":62s}')

print()
if fails:
    print(f'{len(fails)} FAILURES')
    for f in fails:
        print(f'  {f}')
    sys.exit(1)
print('all ok')
