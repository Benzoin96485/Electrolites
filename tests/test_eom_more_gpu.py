"""Triplet and spin-flip EOM-EE-CCSD, left EOM-IP/EA vectors and the CCSD
1-/2-RDMs on the GPU against PySCF on the host, from the same CCSD (and Λ)
amplitudes (GPU4PySCF's CCSD, copied with ``to_cpu``).

    python tests/test_eom_more_gpu.py
"""
import sys

import numpy as np
import cupy as cp
from pyscf import gto, ao2mo
from pyscf.cc import eom_rccsd, ccsd_rdm, ccsd_lambda
from gpu4pyscf import scf
from gpu4pyscf.cc import ccsd_incore

import electrolites
from electrolites import eom_gpu, ccsd_lambda_gpu, ccsd_rdm_gpu

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
    rng = np.random.default_rng(3)

    for kind, Ref, New in (('triplet', eom_rccsd.EOMEETriplet,
                            eom_gpu.EOMEETriplet),
                           ('spin-flip', eom_rccsd.EOMEESpinFlip,
                            eom_gpu.EOMEESpinFlip)):
        ref = Ref(ccpu)
        ref.conv_tol = 1e-11
        e0 = np.sort(np.atleast_1d(ref.kernel(nroots=4, eris=eris)[0]))
        new = New(mycc)
        new.conv_tol = 1e-11
        e1 = np.sort(np.atleast_1d(new.kernel(nroots=4)[0]))
        d = abs(e0 - e1).max()
        report(f'{name} EOM-EE {kind} 4 roots == PySCF', d < 5e-8,
               f'{d:.1e} Eh ({np.round(e1 * 27.2114, 3)} eV)')
        ic = ref.make_imds(eris)
        ig = new.make_imds()
        x = rng.standard_normal(ref.vector_size())
        s0 = ref.matvec(x, ic)
        s1 = new.matvec(x, ig)
        ds = abs(s0 - s1).max() / abs(s0).max()
        report(f'{name} {kind} sigma == PySCF', ds < 1e-9, f'rel {ds:.1e}')

    for kind, Ref, New in (('IP', eom_rccsd.EOMIP, eom_gpu.EOMIP),
                           ('EA', eom_rccsd.EOMEA, eom_gpu.EOMEA)):
        ref = Ref(ccpu)
        ref.conv_tol = 1e-10
        e0 = ref.kernel(nroots=3, left=True, eris=eris)[0]
        new = New(mycc)
        new.conv_tol = 1e-10
        e1 = new.kernel(nroots=3, left=True)[0]
        er = new.kernel(nroots=3)[0]
        d = abs(np.asarray(e0) - np.asarray(e1)).max()
        dr = abs(np.asarray(er) - np.asarray(e1)).max()
        report(f'{name} left EOM-{kind} == PySCF, == right', d < 1e-8
               and dr < 1e-7, f'{d:.1e}, left-right {dr:.1e}')
        ic = ref.make_imds(eris)
        ig = new.make_imds()
        x = rng.standard_normal(ref.vector_size())
        s0 = ref.l_matvec(x, ic)
        s1 = new.l_matvec(x, ig)
        ds = abs(s0 - s1).max() / abs(s0).max()
        report(f'{name} left {kind} sigma == PySCF', ds < 1e-9,
               f'rel {ds:.1e}')

    # RDMs
    conv, l1, l2 = ccsd_lambda.kernel(ccpu, eris, tol=1e-9, max_cycle=100,
                                      verbose=0)
    ccsd_lambda_gpu.kernel(mycc, tol=1e-9, max_cycle=100)
    dm1c = ccsd_rdm.make_rdm1(ccpu, ccpu.t1, ccpu.t2, l1, l2)
    dm2c = ccsd_rdm.make_rdm2(ccpu, ccpu.t1, ccpu.t2, l1, l2)
    dm1g = mycc.make_rdm1().get()
    dm2g = mycc.make_rdm2().get()
    d1 = abs(dm1c - dm1g).max()
    d2 = abs(dm2c - dm2g).max()
    report(f'{name} CCSD 1-RDM and 2-RDM == PySCF', d1 < 1e-7 and d2 < 1e-7,
           f'{d1:.1e}, {d2:.1e}')
    mfc = mf.to_cpu()
    mo = mfc.mo_coeff
    h1 = mo.T @ mfc.get_hcore() @ mo
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mo), mo.shape[1])
    e = (np.einsum('pq,qp', h1, dm1g) + .5 * np.einsum('pqrs,pqrs', eri, dm2g)
         + mol.energy_nuc())
    de = abs(e - (mf.e_tot + mycc.e_corr))
    report(f'{name} CCSD energy from the RDMs', de < 1e-7,
           f'{de:.1e} Eh (Lagrangian = energy)')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
