"""GPU (T) (`electrolites/ccsd_t.py`) against PySCF 2.13 on the same
amplitudes and orbitals.

PySCF's CCSD on the GPU mean field's orbitals (``mf.to_cpu()``), then both
(T)s from those amplitudes: closed shells, frozen core, frozen virtuals,
and a forced small k batch (the batching path).  Gate 1e-10 Eh; measured
~1e-14.

    python tests/test_ccsd_t.py
"""
import sys

import electrolites
electrolites.enable('ccsd_t')
from electrolites import ccsd_t as ct                          # noqa: E402

from pyscf import gto, cc as ccc                               # noqa: E402
from gpu4pyscf import scf                                      # noqa: E402
from gpu4pyscf.cc import ccsd_incore                           # noqa: E402

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:44s} {detail}', flush=True)


CASES = [
    ('water', 'O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467', None, None),
    ('water, frozen core', 'O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467', 1, None),
    ('N2, k batch 2', 'N 0 0 0; N 0 0 1.1', None, 2),
    ('CO, frozen core + virtuals', 'C 0 0 0; O 0 0 1.13', [0, 1, 16, 17], None),
    ('formaldehyde', 'C 0 0 0; O 0 0 1.21; H 0 0.94 -0.59; H 0 -0.94 -0.59',
     None, None),
]
for label, atom, frozen, kb in CASES:
    mol = gto.M(atom=atom, basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    c = ccc.CCSD(mf.to_cpu(), frozen=frozen).run(conv_tol=1e-10)
    g = ccsd_incore.CCSD(mf, frozen=frozen)
    et = ct.kernel(g, c.t1, c.t2, k_batch=kb)
    ref = c.ccsd_t()
    report(label, abs(et - ref) < 1e-10, f'{et:.12f} vs {ref:.12f}')

print('the method on a GPU CCSD object')
mol = gto.M(atom=CASES[0][1], basis='6-31g', verbose=0)
mf = scf.RHF(mol).run(conv_tol=1e-11)
g = ccsd_incore.CCSD(mf)
g.conv_tol = 1e-10
g.kernel()
c = ccc.CCSD(mf.to_cpu())
c.t1, c.t2 = g.t1, g.t2
report('CCSD.ccsd_t() == PySCF on its amplitudes',
       abs(g.ccsd_t() - c.ccsd_t()) < 1e-10)

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
