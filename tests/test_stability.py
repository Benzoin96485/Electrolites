"""GPU stability analysis (`electrolites/stability.py`) against PySCF 2.13.

For each case the same mean field is converged on the GPU and on the CPU,
then both analyses run; PySCF's lowest eigenvalues are captured from its
``lib.davidson`` calls, ours from ``stability.LAST``.  Checked: the lowest
eigenvalue of every analysis (to 1e-5, the Davidson's own resolution), the
stable/unstable verdicts, and -- for an unstable case -- that the rotated
orbitals lead an SCF to a lower energy, as PySCF's do.

    python tests/test_stability.py
"""
import sys

import numpy as np

import electrolites
electrolites.enable('stability')
from electrolites import stability                          # noqa: E402

from pyscf import gto, lib                                  # noqa: E402
from pyscf import scf as cscf, dft as cdft                   # noqa: E402
from pyscf.scf import stability as stab_cpu                 # noqa: E402
from gpu4pyscf import scf, dft                              # noqa: E402

fails = []
TOL = 1e-5


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:58s} {detail}', flush=True)


CPU_EIGS = []
_dav = lib.davidson


def _recording_davidson(*args, **kwargs):
    e, v = _dav(*args, **kwargs)
    CPU_EIGS.append(np.atleast_1d(e)[0])
    return e, v


stab_cpu.lib.davidson = _recording_davidson

def _dense_uhf2ghf_lowest(mf):
    """Lowest real eigenvalue of the UHF -> GHF operator, from its dense
    matrix (one batched product on the identity)."""
    import cupy as cp
    ops = []
    orig = stability._davidson

    def capture(hop, hdiag, *args, **kwargs):
        ops.append((hop, hdiag))
        return orig(hop, hdiag, *args, **kwargs)
    stability._davidson = capture
    try:
        stability.uhf_external(mf)
    finally:
        stability._davidson = orig
    hop, hd = ops[1]
    H = hop(cp.eye(hd.size)).get().T
    return np.sort(np.linalg.eigvals(H).real)[0]


water = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
              basis='def2-svp', verbose=0)
n2 = gto.M(atom='N 0 0 0; N 0 0 2.0', basis='6-31g', verbose=0)
cation = gto.M(atom='O 0 0 0.1; H 0 0.76 -0.45; H 0 -0.76 -0.45',
               basis='def2-svp', charge=1, spin=1, verbose=0)

CASES = [
    # label, gpu builder, cpu builder, kinds (in PySCF's call order)
    ('water RHF', lambda: scf.RHF(water), lambda: cscf.RHF(water),
     ['rhf_internal', 'rhf_real2complex', 'rhf_external']),
    ('water B3LYP', lambda: dft.RKS(water, xc='b3lyp'),
     lambda: cdft.RKS(water, xc='b3lyp'),
     ['rhf_internal', 'rhf_real2complex', 'rhf_external']),
    ('water DF-PBE0', lambda: dft.RKS(water, xc='pbe0').density_fit(),
     lambda: cdft.RKS(water, xc='pbe0').density_fit(),
     ['rhf_internal', 'rhf_real2complex', 'rhf_external']),
    ('N2 (2.0 A) RHF', lambda: scf.RHF(n2), lambda: cscf.RHF(n2),
     ['rhf_internal', 'rhf_real2complex', 'rhf_external']),
    ('H2O+ UHF', lambda: scf.UHF(cation), lambda: cscf.UHF(cation),
     ['uhf_internal', 'uhf_real2complex', 'uhf_external']),
    ('H2O+ UKS B3LYP', lambda: dft.UKS(cation, xc='b3lyp'),
     lambda: cdft.UKS(cation, xc='b3lyp'),
     ['uhf_internal', 'uhf_real2complex', 'uhf_external']),
    ('H2O+ ROHF', lambda: scf.ROHF(cation), lambda: cscf.ROHF(cation),
     ['rohf_internal']),
]

for label, gmk, cmk, kinds in CASES:
    print(label)
    external = len(kinds) > 1
    g = gmk().run(conv_tol=1e-10)
    c = cmk().run(conv_tol=1e-10)
    stability.LAST.clear()
    CPU_EIGS.clear()
    # tol 1e-8 on both: PySCF's default (1e-4) stops its Davidson at a
    # residual of 1e-2, which is looser than the comparison
    _, _, gi, ge = g.stability(external=external, return_status=True, tol=1e-8)
    _, _, ci, ce = c.stability(external=external, return_status=True, tol=1e-8)
    for kind, ref in zip(kinds, CPU_EIGS):
        got = stability.LAST[kind][0]
        if kind == 'uhf_external' and 'UKS' in label:
            # not symmetric: PySCF's symmetric Davidson is only good to
            # ~1e-4 here; the exact answer is the lowest real eigenvalue of
            # the operator, taken from the dense matrix
            exact = _dense_uhf2ghf_lowest(g)
            report(f'{kind} lowest eig vs dense (non-symmetric)',
                   abs(got - exact) < TOL, f'{got:+.6f} vs {exact:+.6f}')
            report(f'{kind} lowest eig vs PySCF, loosely', abs(got - ref) < 1e-4,
                   f'{got:+.6f} vs {ref:+.6f}')
            continue
        report(f'{kind} lowest eig', abs(got - ref) < TOL,
               f'{got:+.6f} vs {ref:+.6f}')
    report('verdicts', (gi, ge) == (ci, ce), f'{(gi, ge)} vs {(ci, ce)}')

print('unstable N2: the rotated orbitals lower the UHF energy')
g = scf.RHF(n2).run(conv_tol=1e-10)
(mo_i, mo_e, si, se) = g.stability(external=True, return_status=True)
u = scf.UHF(n2)
dm = u.make_rdm1(mo_e, (g.mo_occ / 2, g.mo_occ / 2)) if not se else None
if dm is not None:
    e_u = u.run(dm, conv_tol=1e-10).e_tot
    report('UHF from the rotated orbitals is lower', e_u < g.e_tot - 1e-4,
           f'{e_u:.8f} vs RHF {g.e_tot:.8f}')
else:
    report('N2 at 2.0 A is RHF->UHF unstable', False)

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
