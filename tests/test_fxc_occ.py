"""W33: the occupied-side route of a Hessian's CPHF grid half.

``docs/DF_WORK_LIST.md`` **W33**, ``docs/README_fastxcfxcmgga.md``.  The
route computes ``V C`` rather than ``V`` and never forms the AO potential on
the blocks it takes, so the bar is the MO-basis response a CPHF cycle
actually consumes, element-wise, against GPU4PySCF's own code on the same
converged SCF and the same random ``mo1``:

* **restricted** -- ``hessian/rks.py:nr_rks_fxc_mo`` against GPU4PySCF's
  AO-basis ``nr_rks_fxc`` taken to the MO basis with upstream's ``_ao2mo``
  (upstream's own ``nr_rks_fxc_mo`` is wrong for meta-GGAs, W23), with the
  route on every block (``FASTXCFXC_OCC_RATIO=0``), on the default share and
  on none (``FASTXCFXC_OCC=0``), and with a restricted GGA's density on the
  occupied side too (``FASTXCFXC_OCC_RHO_GGA=1``);
* **unrestricted** -- the whole ``hessian/uks.py:get_veff_resp_mo`` (the
  method a CPHF cycle calls) against upstream's function with GPU4PySCF's own
  ``nr_uks_fxc``, on doublets with ``nocca != noccb`` so that the padded
  beta block is exercised, organic and transition-metal, GGA and meta-GGA,
  hybrid and not;
* that the route was taken (``stats()['occ_blocks']``), since a
  fall-through would agree perfectly and say nothing.

    python tests/test_fxc_occ.py [--quick]
"""
import argparse
import os

import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fastxcfxc')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.dft import numint as NI                       # noqa: E402
from gpu4pyscf.hessian import rhf as rhf_hess                # noqa: E402
from gpu4pyscf.hessian import uhf as uhf_hess                # noqa: E402
from gpu4pyscf.hessian import uks as uks_hess                # noqa: E402
from electrolites import fastxcfxc                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
ap.add_argument('--tol', type=float, default=1e-11)
a = ap.parse_args()

M = 'benchmarks/molecules/'
ETHANOL = '''C -0.748 -0.015 0.024; C 0.558 0.420 -0.278; O 0.629 1.735 0.213
H -1.588 0.528 -0.407; H -0.873 -0.009 1.110; H -0.889 -1.047 -0.290
H 1.402 -0.126 0.144; H 0.656 0.408 -1.371; H 1.513 2.038 -0.026'''
CH3 = '''C 0. 0. 0.; H 1.079 0. 0.
H -0.539 0.934 0.; H -0.539 -0.934 0.'''
O2 = 'O 0. 0. 0.; O 0. 0. 1.208'

# (label, atom, basis, xc, charge, spin)
RCASES = [
    ('ethanol', ETHANOL, 'def2-tzvpp', 'tpss', 0, 0),
    ('ethanol', ETHANOL, 'def2-svp', 'm06-2x', 0, 0),
    ('ethanol', ETHANOL, 'def2-svp', 'pbe', 0, 0),
    ('glycine', M + 'glycine.xyz', 'def2-svp', 'm06-l', 0, 0),
    ('Pt amine', M + 'tmc_pt_amine.xyz', 'def2-svp', 'r2scan', 0, 0),
    ('Ni dithiolene2-', M + 'tmc_ni_dithiolene.xyz', 'def2-svp', 'b3lyp',
     -2, 0),
]
UCASES = [
    ('CH3', CH3, '6-31g*', 'tpss', 0, 1),
    ('CH3', CH3, 'def2-tzvpp', 'pbe0', 0, 1),
    ('O2', O2, 'def2-svp', 'm06-2x', 0, 2),
    ('ethanol+', ETHANOL, 'def2-svp', 'r2scan', 1, 1),
    ('FeCl4-', M + 'tmc_fecl4.xyz', 'def2-svp', 'tpss', -1, 5),
    ('Ni dithiolene-', M + 'tmc_ni_dithiolene.xyz', 'def2-svp', 'b3lyp',
     -1, 1),
    ('Pt amine', M + 'tmc_pt_amine.xyz', 'def2-svp', 'm06-l', 0, 2),
]
if a.quick:
    RCASES, UCASES = RCASES[:3], UCASES[:3]

fails = []


def build(atom, basis, charge, spin):
    kw = dict(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0,
              max_memory=40000)
    if basis.lower().startswith('def2') and any(
            gto.charge(sym) > 36 for sym, _ in gto.format_atom(atom)):
        kw['ecp'] = basis
    return gto.M(**kw)


def scf(atom, basis, xc, charge, spin):
    mol = build(atom, basis, charge, spin)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc).density_fit()
    mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    mf.verbose = 0
    mf.kernel()
    return mf


def check(label, got, ref):
    err = float(cp.abs(got - ref).max()) / max(float(cp.abs(ref).max()),
                                                1e-300)
    ok = err <= a.tol
    if not ok:
        fails.append(f'{label}: {err:.3e}')
    print(f'  {"ok  " if ok else "FAIL"} {label:66s} {err:.3e}', flush=True)


def with_env(env, fn):
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return fn()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


VARIANTS = (('default', {}),
            ('every block', {'FASTXCFXC_OCC_RATIO': '0',
                             'FASTXCFXC_OCC_RATIO_GGA': '0'}),
            ('OCC_RHO_GGA=1', {'FASTXCFXC_OCC_RHO_GGA': '1',
                               'FASTXCFXC_OCC_RATIO_GGA': '0'}),
            ('OCC=0', {'FASTXCFXC_OCC': '0'}))

print('## restricted: nr_rks_fxc_mo against the AO path')
rng = np.random.default_rng(7)
for label, atom, basis, xc, charge, spin in RCASES:
    mf = scf(atom, basis, xc, charge, spin)
    ni = mf._numint
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    orbo = mo_coeff[:, mo_occ > 0]
    nmo, nocc = mo_coeff.shape[1], orbo.shape[1]
    rho0, vxc, fxc = ni.cache_xc_kernel(mf.mol, mf.grids, xc, mo_coeff,
                                        mo_occ, 0)
    nset = 6
    mo1 = cp.asarray(rng.standard_normal((nset, nmo, nocc)) * 1e-2)
    mo1_ao = fastxcfxc.contract('npo,ip->nio', mo1, mo_coeff)
    dm1 = cp.asarray(fastxcfxc.contract('npi,qi->npq', mo1_ao, orbo * 2),
                     order='C')
    fastxcfxc.transpose_sum(dm1, inplace=True, hermi=1)
    plain = dm1.copy()
    dm1 = rhf_hess.tag_array(dm1, mo1=mo1_ao, occ_coeff=orbo, symmetrize=1)
    ref = rhf_hess._ao2mo(NI._nr_rks_fxc_orig(
        ni, mf.mol, mf.grids, xc, None, plain, 0, 1, rho0, vxc, fxc,
        max_memory=None), orbo, mo_coeff)
    tag = f'{label}/{basis} {xc} nao={mf.mol.nao} nocc={nocc}'
    print(f'# {tag}')
    for vl, env in VARIANTS:
        got = with_env(env, lambda: fastxcfxc._nr_rks_fxc_mo(
            ni, mf.mol, mf.grids, xc, None, dm1, mo_coeff, 0, 1, rho0, vxc,
            fxc, max_memory=None))
        check(f'{label} {xc} [{vl}]', got, ref)
    mf = None
    cp.get_default_memory_pool().free_all_blocks()

print('\n## unrestricted: hessian/uks.py:get_veff_resp_mo against upstream')
OURS = fastxcfxc._get_veff_resp_mo_u
UP = fastxcfxc._get_veff_resp_mo_u_orig
FAST = fastxcfxc._nr_uks_fxc_dispatch
ORIG = NI._nr_uks_fxc_orig
n_u0 = fastxcfxc.stats().get('occ_u_calls', 0)
for label, atom, basis, xc, charge, spin in UCASES:
    mf = scf(atom, basis, xc, charge, spin)
    if not mf.converged:
        print(f'# {label} {xc}: SCF did not converge -- skipped')
        continue
    h = mf.Hessian()
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    nmo = mo_coeff.shape[-1]
    nocca = int((mo_occ[0] > 0).sum())
    noccb = int((mo_occ[1] > 0).sum())
    nset = 5
    mo1 = cp.asarray(rng.standard_normal((nset, nmo * (nocca + noccb)))
                     * 1e-2)
    fx = uhf_hess.gen_vind(h, mo_coeff, mo_occ)

    def run(fn, nuks):
        # `gen_vind`'s closure calls `hessobj.get_veff_resp_mo`, which
        # resolves on the class
        saved = uks_hess.Hessian.get_veff_resp_mo, NI.NumInt.nr_uks_fxc
        uks_hess.Hessian.get_veff_resp_mo = fn
        NI.nr_uks_fxc = NI.NumInt.nr_uks_fxc = nuks
        try:
            return fx(mo1)
        finally:
            uks_hess.Hessian.get_veff_resp_mo = saved[0]
            NI.nr_uks_fxc = NI.NumInt.nr_uks_fxc = saved[1]
    ref = run(UP, ORIG)
    tag = (f'{label}/{basis} {xc} nao={mf.mol.nao} '
           f'nocc={nocca}/{noccb}')
    print(f'# {tag}')
    for vl, env in VARIANTS[:2] + VARIANTS[3:]:
        got = with_env(env, lambda: run(OURS, FAST))
        check(f'{label} {xc} [{vl}]', got, ref)
    mf = h = None
    cp.get_default_memory_pool().free_all_blocks()

st = fastxcfxc.stats()
for key, n in (('occ_blocks', st.get('occ_blocks', 0)),
               ('occ_u_calls', st.get('occ_u_calls', 0) - n_u0)):
    ok = n > 0
    if not ok:
        fails.append(f'{key} = 0: the route was never taken')
    print(f'  {"ok  " if ok else "FAIL"} {"route taken: " + key:66s} {n}')
print()
if fails:
    print('FAILED:')
    for f in fails:
        print('  ' + f)
    raise SystemExit(1)
print('all ok')
