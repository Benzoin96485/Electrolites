"""G0W0-AC on the GPU (``electrolites.gw_gpu``, FEATURE_TARGETS P9) against
PySCF's ``gw.gw_ac.GWAC``.

    python tests/test_gw_gpu.py
"""
import sys

import numpy as np

from pyscf import gto
from pyscf.gw import gw_ac
from gpu4pyscf import dft, scf

from electrolites import gw_gpu
from electrolites.gw_gpu import GWAC

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


water = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
              basis='def2-svp', verbose=0)
gly = gto.M(atom='/global/homes/w/weiliang/vibeqc/Electrolites/benchmarks/'
            'molecules/glycine.xyz', basis='def2-svp', verbose=0)

print('the imaginary-axis self-energy, same inputs')
mf = dft.RKS(water, xc='pbe').run(conv_tol=1e-11)
c = gw_ac.GWAC(mf.to_cpu())
c.initialize_df()
lpq = c.ao2mo()
nocc = water.nelectron // 2
orbs = list(range(nocc - 3, nocc + 3))
c._nocc = nocc
fr, wt = gw_ac._get_scaled_legendre_roots(c.nw)
ef = .5 * (mf.mo_energy[nocc - 1] + mf.mo_energy[nocc]).get()
args = (c, orbs, lpq, fr, wt, ef, mf.mo_energy.get())
s_ref, w_ref = gw_ac.get_sigma(*args, iw_cutoff=5.0)
s_gpu, w_gpu = gw_gpu._get_sigma_gpu(*args, iw_cutoff=5.0)
err = abs(s_gpu - s_ref).max()
report('Sigma_c(i w), water PBE, 6 orbitals', err < 1e-12, f'{err:.1e}')

print('quasiparticle energies (qpe_tol 1e-10).  The Pade continuation is')
print('ill-conditioned: rotating PySCF\'s own auxiliary basis by a random')
print('orthogonal matrix (exactly invariant) moves its water QP energies by')
print('1.7e-7, so the gate here is 1e-6; the self-energy above is the tight one')
for label, mol, mk in (('water PBE', water, lambda m: dft.RKS(m, xc='pbe')),
                       ('water HF, DF', water, lambda m: scf.RHF(m).density_fit()),
                       ('glycine PBE, DF', gly,
                        lambda m: dft.RKS(m, xc='pbe').density_fit())):
    mf = mk(mol).run(conv_tol=1e-11)
    nocc = mol.nelectron // 2
    orbs = list(range(nocc - 3, nocc + 3))
    g = GWAC(mf)
    g.orbs, g.qpe_tol = orbs, 1e-10
    g.kernel()
    c = gw_ac.GWAC(mf.to_cpu())
    c.orbs, c.qpe_tol = orbs, 1e-10
    c.kernel()
    err = abs(np.asarray(g.mo_energy)[orbs] - np.asarray(c.mo_energy)[orbs]).max()
    report(f'{label}: HOMO-2..LUMO+2 QP energies', err < 1e-6, f'{err:.1e}')

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
