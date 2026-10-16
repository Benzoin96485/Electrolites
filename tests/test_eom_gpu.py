"""EOM-CCSD on the GPU (``electrolites.eom_gpu``) against PySCF's EOM on
the host, from the same CCSD amplitudes (GPU4PySCF's CCSD, copied to
PySCF with ``to_cpu``).

    python tests/test_eom_gpu.py
"""
import sys

import numpy as np
from pyscf import gto
from pyscf.cc import eom_rccsd
from gpu4pyscf import scf
from gpu4pyscf.cc import ccsd_incore

import electrolites
from electrolites import eom_gpu

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
    nroots = 4
    ref = eom_rccsd.EOMIP(ccpu)
    ref.conv_tol = 1e-9
    e0, v0 = ref.kernel(nroots=nroots)
    new = eom_gpu.EOMIP(mycc)
    new.conv_tol = 1e-9
    e1, v1 = new.kernel(nroots=nroots)
    d = abs(np.asarray(e0) - np.asarray(e1)).max()
    report(f'{name} EOM-IP {nroots} roots == PySCF', d < 1e-7,
           f'{d:.1e} Eh (IPs {np.round(np.asarray(e1) * 27.2114, 3)} eV)')
    # sigma vectors and the diagonal on one random vector
    imds_c = ref.make_imds()
    imds_g = new.make_imds()
    x = np.random.default_rng(0).standard_normal(ref.vector_size())
    s0 = ref.matvec(x, imds_c)
    s1 = new.matvec(x, imds_g)
    ds = abs(s0 - s1).max() / abs(s0).max()
    dd = abs(ref.get_diag(imds_c) - new.get_diag(imds_g)).max()
    report(f'{name} IP sigma and diagonal == PySCF', ds < 1e-10 and dd < 1e-9,
           f'sigma rel {ds:.1e}, diag {dd:.1e}')
    # EOM-EA
    ref = eom_rccsd.EOMEA(ccpu)
    ref.conv_tol = 1e-9
    e0, v0 = ref.kernel(nroots=nroots)
    new = eom_gpu.EOMEA(mycc)
    new.conv_tol = 1e-9
    e1, v1 = new.kernel(nroots=nroots)
    d = abs(np.asarray(e0) - np.asarray(e1)).max()
    report(f'{name} EOM-EA {nroots} roots == PySCF', d < 1e-7,
           f'{d:.1e} Eh (EAs {np.round(np.asarray(e1) * 27.2114, 3)} eV)')
    imds_c = ref.make_imds()
    imds_g = new.make_imds()
    x = np.random.default_rng(1).standard_normal(ref.vector_size())
    s0 = ref.matvec(x, imds_c)
    s1 = new.matvec(x, imds_g)
    ds = abs(s0 - s1).max() / abs(s0).max()
    dd = abs(ref.get_diag(imds_c) - new.get_diag(imds_g)).max()
    report(f'{name} EA sigma and diagonal == PySCF', ds < 1e-9 and dd < 1e-9,
           f'sigma rel {ds:.1e}, diag {dd:.1e}')
    if frozen:
        from pyscf import cc as pcc
        ec = pcc.CCSD(mf.to_cpu(), frozen=frozen)
        ec.conv_tol = 1e-10
        ec.kernel()
        de = abs(ec.e_corr - mycc.e_corr)
        report(f'{name} frozen-core CCSD energy == PySCF', de < 1e-8,
               f'{de:.1e} Eh')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
