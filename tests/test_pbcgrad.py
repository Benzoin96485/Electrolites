"""The periodic gradients' one-electron term against GPU4PySCF's (J7).

``electrolites/fastpbcgrad.py`` replaces the per-atom ``hcore_generator`` loop
in ``pbc/grad/{krhf,kuhf}.py:grad_elec`` and the Gamma-point
``{rhf,uhf}.Gradients.grad_elec`` by one density on the grid.  Each patched
``grad_elec`` against the replaced one on the same converged SCF: diamond
(GTH pseudopotential) with k-points, restricted and unrestricted, at the
Gamma point, and an all-electron cell (the Coulomb branch of the local term).

And the stress tensor's GTH non-local term (J5):
``_get_pp_nonloc_strain_derivatives`` against the replaced one on the same
density, k-points and Gamma, one species and two (SiC), bar ``--tol-pp``
(a central difference over 2*disp ~ 6e-5 of energies equal to round-off).

    python tests/test_pbcgrad.py
"""
import sys

import numpy as np

import electrolites
electrolites.patch_all()
from pyscf.pbc import gto                                    # noqa: E402
from gpu4pyscf.pbc import dft                                # noqa: E402
from gpu4pyscf.pbc.grad import krhf, kuhf, rhf, uhf          # noqa: E402

TOL = 1e-10
fails = []
DIAMOND = 'C 0 0 0; C 0.8917 0.8917 1.0'      # off-equilibrium: forces != 0
LAT = '0 1.7834 1.7834; 1.7834 0 1.7834; 1.7834 1.7834 0'


def cell_(pseudo=True, spin=0):
    if pseudo:
        return gto.M(atom=DIAMOND, a=LAT, basis='gth-szv', pseudo='gth-pade',
                     spin=spin, verbose=0)
    return gto.M(atom='Li 0 0 0; H 1.9 1.7 1.8', a=np.eye(3) * 4.0,
                 basis='sto-3g', spin=spin, verbose=0)


def run(tag, mf, orig):
    mf.conv_tol = 1e-10
    mf.kernel()
    g = mf.nuc_grad_method()
    ours = np.asarray(g.grad_elec())
    ref = np.asarray(orig(g))
    err = np.abs(ours - ref).max()
    ok = err <= TOL
    if not ok:
        fails.append(f'{tag}: {err:.3e}')
    print(f'  {"ok  " if ok else "FAIL"} {tag:40s} {err:.3e}  '
          f'(|grad_elec| {np.abs(ref).max():.2e})', flush=True)


kpts = cell_().make_kpts([2, 2, 2])
run('KRKS diamond 2x2x2 (GTH)', dft.KRKS(cell_(), kpts=kpts, xc='pbe'),
    krhf._grad_elec_orig)
run('KUKS diamond 2x2x2 (GTH)', dft.KUKS(cell_(), kpts=kpts, xc='pbe'),
    kuhf._grad_elec_orig)
run('RKS diamond Gamma (GTH)', dft.RKS(cell_(), xc='pbe'),
    rhf.Gradients._grad_elec_orig)
run('UKS diamond Gamma (GTH)', dft.UKS(cell_(), xc='pbe'),
    uhf.Gradients._grad_elec_orig)
ae = cell_(pseudo=False)
run('KRKS LiH 2x2x2 (all-electron)',
    dft.KRKS(ae, kpts=ae.make_kpts([2, 2, 2]), xc='pbe'), krhf._grad_elec_orig)
run('RKS LiH Gamma (all-electron)', dft.RKS(ae, xc='pbe'),
    rhf.Gradients._grad_elec_orig)
print('## the GTH non-local strain derivative (J5)')
from gpu4pyscf.pbc.grad import rks_stress as RKSS               # noqa: E402
from electrolites import fastpbcgrad                            # noqa: E402
SIC = gto.M(atom='Si 0 0 0; C 1.09 1.09 1.2', a=LAT.replace('1.7834', '2.18'),
            basis='gth-dzvp', pseudo='gth-pade', verbose=0)
for tag, c, kmesh in (('diamond 2x2x2', cell_(), [2, 2, 2]),
                      ('SiC 2x2x2', SIC, [2, 2, 2]),
                      ('SiC Gamma', SIC, None)):
    kp = c.make_kpts(kmesh) if kmesh else None
    mf = (dft.KRKS(c, kpts=kp, xc='pbe') if kmesh else dft.RKS(c, xc='pbe'))
    mf.max_cycle = 3
    mf.kernel()
    dm = mf.make_rdm1()
    args = (c, c.mesh, dm, kp) if kmesh else (c, c.mesh, dm)
    ours = fastpbcgrad.pp_nonloc_strain_derivatives(*args)
    ref = RKSS._get_pp_nonloc_strain_derivatives_orig(*args)
    err = np.abs(ours - ref).max()
    ok = err <= 1e-8
    if not ok:
        fails.append(f'pp strain {tag}: {err:.3e}')
    print(f'  {"ok  " if ok else "FAIL"} {"pp strain " + tag:40s} {err:.3e}  '
          f'(|ref| {np.abs(ref).max():.2e})', flush=True)
if fails:
    print('\nFAILED:\n  ' + '\n  '.join(fails))
    sys.exit(1)
print('\nall ok')
