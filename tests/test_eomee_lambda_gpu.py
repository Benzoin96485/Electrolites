"""Singlet EOM-EE-CCSD (``electrolites.eom_gpu.EOMEESinglet``) and the CCSD
Λ equations (``electrolites.ccsd_lambda_gpu``) on the GPU against PySCF's
``eom_rccsd.EOMEESinglet`` and ``ccsd_lambda`` on the host, from the same
CCSD amplitudes (GPU4PySCF's CCSD, copied with ``to_cpu``).

    python tests/test_eomee_lambda_gpu.py
"""
import sys

import numpy as np
from pyscf import gto
from pyscf.cc import eom_rccsd, ccsd_lambda
from gpu4pyscf import scf
from gpu4pyscf.cc import ccsd_incore

import electrolites
from electrolites import eom_gpu, ccsd_lambda_gpu

electrolites.patch_all()
fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


for name, atom, basis, frozen in (
        ('water', 'O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
         'cc-pvdz', None),
        ('glycine fc', 'benchmarks/molecules/glycine.xyz', 'def2-svp', 5)):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    mycc = ccsd_incore.CCSD(mf, frozen=frozen)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-8
    mycc.kernel()
    ccpu = mycc.to_cpu()
    eris = ccpu.ao2mo()

    # Lambda
    conv0, l1c, l2c = ccsd_lambda.kernel(ccpu, eris, tol=1e-9, max_cycle=100,
                                         verbose=0)
    conv1, l1g, l2g = ccsd_lambda_gpu.kernel(mycc, tol=1e-9, max_cycle=100)
    d = max(abs(l1c - l1g).max(), abs(l2c - l2g).max())
    report(f'{name} Lambda == PySCF', conv1 and d < 1e-7, f'{d:.1e}')
    # one update on the same (l1, l2)
    imds_c = ccsd_lambda.make_intermediates(ccpu, ccpu.t1, ccpu.t2, eris)
    u0 = ccsd_lambda.update_lambda(ccpu, ccpu.t1, ccpu.t2, l1c, l2c, eris,
                                   imds_c)
    imds_g = ccsd_lambda_gpu.make_intermediates(eom_gpu._Imds(mycc))
    import cupy as cp
    u1 = ccsd_lambda_gpu.update_lambda(imds_g, mycc, cp.asarray(l1c),
                                       cp.asarray(l2c))
    du = max(abs(u0[0] - u1[0].get()).max(), abs(u0[1] - u1[1].get()).max())
    report(f'{name} one Lambda update == PySCF', du < 1e-9, f'{du:.1e}')

    # EOM-EE singlet
    nroots = 4
    ref = eom_rccsd.EOMEESinglet(ccpu)
    ref.conv_tol = 1e-11
    e0 = ref.kernel(nroots=nroots, eris=eris)[0]
    new = eom_gpu.EOMEESinglet(mycc)
    new.conv_tol = 1e-11
    e1 = new.kernel(nroots=nroots)[0]
    d = abs(np.asarray(e0) - np.asarray(e1)).max()
    report(f'{name} EOM-EE singlet {nroots} roots == PySCF', d < 5e-8,
           f'{d:.1e} Eh ({np.round(np.asarray(e1) * 27.2114, 3)} eV)')
    imds_c = ref.make_imds(eris)
    imds_g = new.make_imds()
    x = np.random.default_rng(2).standard_normal(ref.vector_size())
    s0 = ref.matvec(x, imds_c)
    s1 = new.matvec(x, imds_g)
    ds = abs(s0 - s1).max() / abs(s0).max()
    dd = abs(eom_rccsd.eeccsd_diag(ref, imds_c)[0] - new.get_diag(imds_g)).max()
    report(f'{name} EE sigma and diagonal == PySCF', ds < 1e-9 and dd < 1e-8,
           f'sigma rel {ds:.1e}, diag {dd:.1e}')
    if frozen is None:
        # the methods bound on GPU4PySCF's CCSD
        e2 = mycc.eomee_ccsd_singlet(nroots=2)[0]
        mycc.solve_lambda()
        report(f'{name} CCSD.eomee_ccsd_singlet / solve_lambda',
               abs(e2[0] - e1[0]) < 1e-6
               and abs(mycc.l1 - l1c).max() < 1e-6,
               f'{abs(e2[0] - e1[0]):.1e}')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
