"""Is a restricted meta-GGA Hessian reproducible, and right? (W23)

``docs/DF_WORK_LIST.md`` **W23**, ``docs/README_mgga_repro.md``.
``hessian/rks.py`` contracts ``fxc`` in place in ``_get_vxc_deriv1_task``
(both branches) and in ``_nr_rks_fxc_mo_task``; cuTENSOR gets that wrong at
five density components, so every meta-GGA Hessian GPU4PySCF 1.8.1 computes
is off by 1e-5-1e-4 in ``make_h1`` and the CPHF, by a different amount each
run.  ``fastxchess`` swaps the ``contract`` that module looked up for
``_alias_safe_contract``.  Checked:

* the contraction itself, in place and out of place, at 1, 4 and 5
  components, against ``cupy.einsum``;
* ``_get_vxc_deriv1`` for TPSS against PySCF's CPU implementation on the same
  orbitals and grid, and three calls against each other bit for bit;
* upstream's ``nr_rks_fxc_mo`` (which ``fastxcfxc`` otherwise replaces) for
  TPSS against the AO-basis ``nr_rks_fxc`` taken to the MO basis -- the
  comparison ``tests/test_dfhessip.py`` could only print as ``info``;
* a whole DF-TPSS Hessian, three times in one process: the traces agree to
  the 1e-9 a GGA one does.

    python tests/test_mgga_inplace.py [--no-hessian]
"""
import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--tol', type=float, default=1e-10)
ap.add_argument('--no-hessian', action='store_true')
a = ap.parse_args()

import cupy as cp                                            # noqa: E402
from pyscf import gto                                        # noqa: E402

import electrolites                                          # noqa: E402
electrolites.patch('fastxchess')
from electrolites import fastxchess                          # noqa: E402

from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.hessian import rks as RKSHESS                 # noqa: E402
from gpu4pyscf.hessian.rhf import _ao2mo                     # noqa: E402
from gpu4pyscf.lib.cupy_helper import (                      # noqa: E402
    contract, tag_array, transpose_sum)

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


def report(label, err, tol):
    ok = err <= tol
    if not ok:
        fails.append(f'{label}: {err:.3e} > {tol:.1e}')
    print(f'  {"ok  " if ok else "FAIL"} {label:62s} {err:.3e}', flush=True)


def rel(got, ref):
    got, ref = cp.asarray(got), cp.asarray(ref)
    return float(abs(got - ref).max()) / (float(abs(ref).max()) or 1.)


print('## the contraction')
rng = cp.random.default_rng(3)
for nc in (1, 4, 5):
    wf = rng.random((nc, nc, 4096))
    x0 = rng.random((3, nc, 4096))
    ref = cp.einsum('xyg,sxg->syg', wf, x0)
    upstream = 0.
    for _ in range(20):         # the corruption is a race: not every call
        x = x0.copy()
        upstream = max(upstream,
                       rel(contract('xyg,sxg->syg', wf, x, out=x), ref))
    x = x0.copy()
    got = fastxchess._alias_safe_contract('xyg,sxg->syg', wf, x, out=x)
    report(f'in place, ncomp={nc} (upstream in place: {upstream:.1e})',
           rel(got, ref), 1e-14)
    y = rng.random((3, nc, 4096))
    got = fastxchess._alias_safe_contract('xyg,sxg->syg', wf, x0, beta=.5,
                                          out=y.copy())
    report(f'out of place with beta, ncomp={nc}', rel(got, ref + .5 * y),
           1e-14)
x = x0.copy()
got = fastxchess._alias_safe_contract('xyg,sxg->syg', wf, x, beta=2., out=x)
report('in place with beta', rel(got, ref + 2 * x0), 1e-14)
ok = RKSHESS.contract is fastxchess._alias_safe_contract
if not ok:
    fails.append('hessian.rks.contract is not the alias-safe one')
print(f'  {"ok  " if ok else "FAIL"} {"hessian/rks.py looks up the alias-safe contract":62s}')

print('\n## _get_vxc_deriv1, TPSS, against PySCF on the CPU')
mol = gto.M(atom=ETHANOL, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='tpss').density_fit()
mf.grids.level = 3
mf.conv_tol = 1e-10
mf.kernel()
mo_c, mo_o = mf.mo_coeff, mf.mo_occ
hs = mf.Hessian()
g = [RKSHESS._get_vxc_deriv1(hs, mo_c, mo_o, 4000) for _ in range(3)]
report('three calls, bit for bit', max(rel(x, g[0]) for x in g[1:]), 0.)
from pyscf import dft as cdft                                # noqa: E402
from pyscf.hessian import rks as crks                        # noqa: E402
cmf = cdft.RKS(mol, xc='tpss')
cmf.grids.level = 3
cmf.grids.build()
cmf.mo_coeff, cmf.mo_occ = cp.asnumpy(mo_c), cp.asnumpy(mo_o)
cmf.mo_energy = cp.asnumpy(mf.mo_energy)
c = crks._get_vxc_deriv1(cmf.Hessian(), cmf.mo_coeff, cmf.mo_occ, 4000)
mocc = cmf.mo_coeff[:, cmf.mo_occ > 0]
c = np.einsum('axpq,pi,qj->axij', c, cmf.mo_coeff, mocc)
report('against pyscf.hessian.rks._get_vxc_deriv1', rel(g[0], c), 1e-9)

print("\n## upstream's nr_rks_fxc_mo, TPSS, against the AO-basis path")
ni = mf._numint
rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, 'tpss', mo_c, mo_o, 0)
orbo = mo_c[:, mo_o > 0]
nmo, nocc = mo_c.shape[1], orbo.shape[1]
for n_dm in (1, 3, 6):
    mo1 = cp.asarray(np.random.default_rng(n_dm).standard_normal(
        (n_dm, nmo, nocc)) * 1e-2)
    mo1_ao = contract('npo,ip->nio', mo1, mo_c)
    dm1 = cp.asarray(contract('npi,qi->npq', mo1_ao, orbo * 2), order='C')
    transpose_sum(dm1, inplace=True, hermi=1)
    ref = _ao2mo(ni.nr_rks_fxc(mol, mf.grids, 'tpss', None, dm1.copy(), 0, 1,
                               rho0, vxc, fxc), orbo, mo_c)
    dm1 = tag_array(dm1, mo1=mo1_ao, occ_coeff=orbo, symmetrize=1)
    up = RKSHESS.nr_rks_fxc_mo(ni, mol, mf.grids, 'tpss', None, dm1, mo_c, 0,
                               1, rho0, vxc, fxc)
    report(f'n_dm={n_dm}', rel(up, ref), a.tol)

if not a.no_hessian:
    print('\n## a whole DF-TPSS Hessian, three times')
    tr = []
    for _ in range(3):
        h = mf.Hessian()
        h.auxbasis_response = 2
        e = np.asarray(h.kernel())
        tr.append(float(np.einsum('aaxx->', e)))
    print('  traces ' + '  '.join(f'{t:.10f}' for t in tr))
    report('spread of the traces, Eh/bohr^2', max(tr) - min(tr), 1e-9)

print(f'\n# fastxchess.stats() = {fastxchess.stats()}')
print('\nRESULT', 'FAIL' if fails else 'PASS')
for f in fails:
    print('  ' + f)
sys.exit(1 if fails else 0)
