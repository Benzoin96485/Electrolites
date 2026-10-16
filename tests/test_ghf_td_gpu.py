"""GHF TDA/TDHF on the GPU (``electrolites.ghf_td_gpu``) against PySCF's
``tdscf.ghf`` on the same GHF solution (copied with ``to_cpu``).

    python tests/test_ghf_td_gpu.py
"""
import sys

import numpy as np
from pyscf import gto
from pyscf.tdscf import ghf as pghf
from gpu4pyscf import scf

from electrolites import ghf_td_gpu

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


for name, atom, charge, spin in (
        ('water', 'O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467', 0, 0),
        ('H2O+', 'O 0 0 0; H 0 0 0.97; H 0.9 0 -0.3', 1, 1)):
    mol = gto.M(atom=atom, basis='cc-pvdz', charge=charge, spin=spin,
                verbose=0)
    mf = scf.GHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    mc = mf.to_cpu()
    kinds = [('TDA', pghf.TDA, ghf_td_gpu.TDA)]
    if spin:
        # closed-shell GHF has three spin-rotation zero modes in TDHF, whose
        # near-zero roots depend on the solver's noise: TDHF on H2O+ only
        kinds.append(('TDHF', pghf.TDHF, ghf_td_gpu.TDHF))
    for kind, ref, new in kinds:
        r = ref(mc)
        r.conv_tol = 1e-9
        r.nstates = 6
        e0 = r.kernel()[0]
        n = new(mf)
        n.conv_tol = 1e-9
        n.nstates = 6
        e1 = n.kernel()[0]
        d = abs(np.asarray(e0) - np.asarray(e1)).max()
        report(f'{name} GHF {kind} 6 states == PySCF', d < 1e-7,
               f'{d:.1e} ({np.round(np.asarray(e1) * 27.2114, 3)} eV)')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
