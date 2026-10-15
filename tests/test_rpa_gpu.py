"""dRPA on the GPU (``electrolites.rpa_gpu``, FEATURE_TARGETS P8) against
PySCF's ``gw.rpa.RPA`` on the same orbitals.

    python tests/test_rpa_gpu.py
"""
import sys

from pyscf import gto
from pyscf.gw import rpa as cpu_rpa
from gpu4pyscf import dft, scf

from electrolites.rpa_gpu import RPA

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


water = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
              basis='def2-svp', verbose=0)
gly = gto.M(atom='/global/homes/w/weiliang/vibeqc/Electrolites/benchmarks/'
            'molecules/glycine.xyz', basis='def2-svp', verbose=0)
for label, mol, mk in (('water PBE', water, lambda m: dft.RKS(m, xc='pbe')),
                       ('water HF', water, scf.RHF),
                       ('glycine PBE', gly, lambda m: dft.RKS(m, xc='pbe')),
                       ('water DF-PBE (the SCF\'s auxbasis)', water,
                        lambda m: dft.RKS(m, xc='pbe').density_fit())):
    mf = mk(mol).run(conv_tol=1e-11)
    for frz in (None, 1):
        g = RPA(mf, frozen=frz)
        g.kernel()
        c = cpu_rpa.RPA(mf.to_cpu(), frozen=frz)
        c.kernel()
        err = max(abs(g.e_corr - c.e_corr), abs(g.e_hf - c.e_hf))
        report(f'{label}, frozen={frz}: E_corr, E_hf', err < 1e-9, f'{err:.1e}')

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
