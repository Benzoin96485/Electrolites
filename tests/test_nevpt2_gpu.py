"""SC-NEVPT2 with GPU intermediates (``electrolites.nevpt2_gpu``,
FEATURE_TARGETS M7) against PySCF's ``mrpt.NEVPT``.

    python tests/test_nevpt2_gpu.py
"""
import sys

import numpy as np
import cupy as cp

from pyscf import gto, fci, mcscf, mrpt
from pyscf import scf as cscf
from pyscf.fci import cistring
from pyscf.mrpt import nevpt2
from gpu4pyscf import scf

from electrolites.casscf_gpu import CASCI, CASSCF
from electrolites.nevpt2_gpu import NEVPT, make_dm123_f3

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


print('dm1..3 and the 4-RDM/ERI contractions against PySCF (random c, eri)')
rng = np.random.default_rng(2)
for norb, nelec in ((4, (2, 2)), (5, (3, 2)), (7, (4, 3)), (8, (4, 4))):
    na = cistring.num_strings(norb, nelec[0])
    nb = cistring.num_strings(norb, nelec[1])
    c = rng.normal(size=(na, nb))
    c /= np.linalg.norm(c)
    eri = rng.normal(size=(norb,) * 4)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(2, 3, 0, 1)
    ref = list(fci.rdm.make_dm123('FCI3pdm_kern_sf', c, c, norb, nelec))
    ref += [nevpt2._contract4pdm('NEVPTkern_cedf_aedf', eri, c, norb, nelec),
            nevpt2._contract4pdm('NEVPTkern_aedf_ecdf', eri, c, norb, nelec)]
    got = make_dm123_f3(c, eri, norb, nelec)
    err = max(abs(g - r).max() / abs(r).max() for g, r in zip(got, ref))
    report(f'CAS({norb}, {nelec}) dm1, dm2, dm3, f3ca, f3ac', err < 1e-13,
           f'{err:.1e}')

pb = '/global/homes/w/weiliang/vibeqc/Electrolites/benchmarks/molecules/pbenzyne.xyz'
mol = gto.M(atom=pb, basis='6-31g', verbose=0)
print('NEVPT2 energies')
mfc = cscf.RHF(mol).run()
mc = mcscf.CASCI(mfc, 8, 8).run()
e0 = mrpt.NEVPT(mc).kernel()
e1 = NEVPT(mc).kernel()
report('CPU CASCI(8,8), exact integrals', abs(e0 - e1) < 1e-10,
       f'{abs(e0 - e1):.1e}')
mf = scf.RHF(mol).run()
for kind, n in (('CASCI', 6), ('CASCI', 10), ('CASSCF', 6)):
    mc = (CASCI if kind == 'CASCI' else CASSCF)(mf, n, n, density_fit=True)
    if kind == 'CASSCF':
        mc.conv_tol = 1e-10
    mc.kernel()
    eg = NEVPT(mc).kernel()
    mfd = mf.to_cpu().density_fit(auxbasis=mc.with_df.auxbasis)
    cm = (mcscf.CASCI if kind == 'CASCI' else mcscf.CASSCF)(mfd, n, n)
    if kind == 'CASCI':
        cm.kernel(cp.asnumpy(mc.mo_coeff))
    else:
        cm.mo_coeff, cm.ci = cp.asnumpy(mc.mo_coeff), mc.ci
        cm.converged = True
    ec = mrpt.NEVPT(cm).kernel()
    report(f'GPU DF {kind}({n},{n}) == PySCF DF NEVPT2', abs(eg - ec) < 1e-9,
           f'{abs(eg - ec):.1e}')

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
