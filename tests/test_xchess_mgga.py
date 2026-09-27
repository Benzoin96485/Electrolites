"""`fastxchess`'s meta-GGA branches against GPU4PySCF and against PySCF (W28).

``docs/DF_WORK_LIST.md`` **W28**, ``docs/README_fastxchessmgga.md``.  Since
W28 the module covers meta-GGAs as well as GGAs, on both references.  Checked,
on one converged SCF per case:

* ``_get_vxc_deriv1`` and ``_get_vxc_deriv2`` against GPU4PySCF's own
  functions (with W23's in-place contraction repaired, which this module
  does on import) and against PySCF's CPU functions on the same orbitals and
  grid -- the second is the check that matters, because before W23 the
  GPU4PySCF reference was itself wrong for meta-GGAs;
* the same for the unrestricted pair in ``hessian/uks.py``;
* a whole DF-TPSS Hessian, patched against unpatched, and patched three
  times: the traces agree to the CPHF's own tolerance, and the patched
  function is deterministic to the atomics' round-off.

    python tests/test_xchess_mgga.py [--no-hessian] [--no-cpu]
"""
import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--no-hessian', action='store_true')
ap.add_argument('--no-cpu', action='store_true')
a = ap.parse_args()

import cupy as cp                                            # noqa: E402
from pyscf import gto                                        # noqa: E402

import electrolites                                          # noqa: E402
electrolites.patch('fastxchess')
from electrolites import fastxchess                          # noqa: E402

from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.hessian import rks as RKSHESS                 # noqa: E402
from gpu4pyscf.hessian import uks as UKSHESS                 # noqa: E402

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
CH2O = '''C 0. 0. 0.; O 0. 0. 1.205; H 0. 0.943 -0.587; H 0. -0.943 -0.587'''
M = 'benchmarks/molecules/'
fails = []


def report(label, err, tol):
    ok = err <= tol
    if not ok:
        fails.append(f'{label}: {err:.3e} > {tol:.1e}')
    print(f'  {"ok  " if ok else "FAIL"} {label:66s} {err:.3e}', flush=True)


def rel(got, ref):
    got, ref = cp.asarray(got), cp.asarray(ref)
    return float(abs(got - ref).max()) / (float(abs(ref).max()) or 1.)


# (label, atom, basis, xc, charge, spin)
CASES = [
    ('ethanol', ETHANOL, 'def2-svp', 'tpss', 0, 0),
    ('ethanol', ETHANOL, 'def2-svp', 'r2scan', 0, 0),
    ('ethanol', ETHANOL, 'def2-svp', 'm06-l', 0, 0),
    ('CH2O', CH2O, 'def2-tzvpp', 'm06-2x', 0, 0),     # hybrid, reaches f
    ('CH2O', CH2O, 'def2-svp', 'wb97m-v', 0, 0),      # range separated + VV10
    ('ethanol+', ETHANOL, 'def2-svp', 'tpss', 1, 1),
    ('CH2O+', CH2O, 'def2-tzvpp', 'r2scan', 1, 1),
    ('FeCl4-', M + 'tmc_fecl4.xyz', 'def2-svp', 'tpss', -1, 5),
]

for label, atom, basis, xc, charge, spin in CASES:
    print(f'\n## {label} {basis} {xc} (2S={spin})')
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc)
    mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.kernel()
    H = UKSHESS if spin else RKSHESS
    h = mf.Hessian()
    mo_c, mo_o = mf.mo_coeff, mf.mo_occ
    assert fastxchess._setup(h, mo_o, unrestricted=bool(spin)) is not None
    out = {}
    for name in ('_get_vxc_deriv1', '_get_vxc_deriv2'):
        f = getattr(H, name)(h, mo_c, mo_o, 4000)
        o = getattr(H, name + '_orig')(h, mo_c, mo_o, 4000)
        f = f if spin else (f,)
        o = o if spin else (o,)
        report(f'{name} against GPU4PySCF', max(rel(x, y)
                                                 for x, y in zip(f, o)), 1e-11)
        out[name] = f
    if a.no_cpu:
        continue
    # PySCF on the CPU, same orbitals, same grid level
    from pyscf import dft as cdft
    from pyscf.hessian import rks as crks, uks as cuks
    cmf = (cdft.UKS if spin else cdft.RKS)(mol, xc=xc)
    cmf.grids.level = 3
    cmf.grids.build(with_non0tab=True)
    cmf.mo_coeff, cmf.mo_occ = cp.asnumpy(mo_c), cp.asnumpy(mo_o)
    cmf.mo_energy = cp.asnumpy(mf.mo_energy)
    ch = cmf.Hessian()
    C = cuks if spin else crks
    d1 = C._get_vxc_deriv1(ch, cmf.mo_coeff, cmf.mo_occ, 4000)
    d2 = C._get_vxc_deriv2(ch, cmf.mo_coeff, cmf.mo_occ, 4000)
    dm0 = cmf.make_rdm1()
    if not spin:
        d1, d2, dm0 = (d1,), (d2,), (dm0,)
        mo_cs, mo_os = (cmf.mo_coeff,), (cmf.mo_occ,)
    else:
        mo_cs, mo_os = cmf.mo_coeff, cmf.mo_occ
    e1 = e2 = 0.
    for s in range(len(d1)):
        mocc = mo_cs[s][:, mo_os[s] > 0]
        # PySCF's deriv1 is in the AO basis (natm, 3, nao, nao)
        ref1 = np.einsum('axpq,pi,qj->axij', d1[s], mo_cs[s], mocc)
        e1 = max(e1, rel(out['_get_vxc_deriv1'][s], ref1))
        # and its deriv2 is not yet contracted with the density
        ref2 = np.einsum('axypq,pq->axyp', d2[s], dm0[s])
        e2 = max(e2, rel(out['_get_vxc_deriv2'][s], ref2))
    report('_get_vxc_deriv1 against PySCF on the CPU', e1, 1e-9)
    report('_get_vxc_deriv2 against PySCF on the CPU', e2, 1e-9)
    mol = mf = h = cmf = None
    cp.get_default_memory_pool().free_all_blocks()

if not a.no_hessian:
    print('\n## a whole DF-TPSS Hessian of ethanol: patched x3, unpatched x1')
    mol = gto.M(atom=ETHANOL, basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='tpss').density_fit()
    mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.kernel()
    tr = []
    for _ in range(3):
        h = mf.Hessian()
        h.auxbasis_response = 2
        e = np.asarray(h.kernel())
        tr.append(float(np.einsum('aaxx->', e)))
    F1, F2 = RKSHESS._get_vxc_deriv1, RKSHESS._get_vxc_deriv2
    RKSHESS._get_vxc_deriv1 = RKSHESS._get_vxc_deriv1_orig
    RKSHESS._get_vxc_deriv2 = RKSHESS._get_vxc_deriv2_orig
    h = mf.Hessian()
    h.auxbasis_response = 2
    eo = np.asarray(h.kernel())
    RKSHESS._get_vxc_deriv1, RKSHESS._get_vxc_deriv2 = F1, F2
    print('  traces, patched ' + '  '.join(f'{t:.10f}' for t in tr)
          + f'   unpatched {np.einsum("aaxx->", eo):.10f}')
    report('spread of the patched traces, Eh/bohr^2', max(tr) - min(tr),
           1e-9)
    report('patched against unpatched, max element, Eh/bohr^2',
           float(abs(e - eo).max()), 1e-7)

print(f'\n# fastxchess.stats() = {fastxchess.stats()}')
print('\nRESULT', 'FAIL' if fails else 'PASS')
for f in fails:
    print('  ' + f)
sys.exit(1 if fails else 0)
