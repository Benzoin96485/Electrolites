"""Static and frequency-dependent polarizability on the GPU
(``electrolites.polar_gpu``, FEATURE_TARGETS O3), RHF/RKS and UHF/UKS.

References: pyscf-properties' ``polarizability_with_freq`` on the same SCF
(closed shell), and, open shell, a dense solve of the response problem on
PySCF's own TDDFT A and B matrices: pyscf-properties' UHF routes
(``ucphf`` static and the ``newton_krylov`` frequency one) are 3e-5 off it
at w -> 0 on UKS OH/aug-cc-pVDZ and 3e-7 at w = 0.1 on UHF, while this code
agrees with it to 1e-10.

    python tests/test_polar_gpu.py
"""
import sys

import numpy as np
from pyscf import gto, tdscf
from pyscf.prop.polarizability import rhf as prhf, uhf as puhf
from gpu4pyscf import scf, dft

from electrolites.polar_gpu import polarizability

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


w = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
          basis='aug-cc-pvdz', verbose=0)
oh = gto.M(atom='O 0 0 0; H 0 0 0.97', spin=1, basis='aug-cc-pvdz',
           verbose=0)
for label, mol, mk, ref in (
        ('water RHF', w, scf.RHF, prhf),
        ('water B3LYP', w, lambda m: dft.RKS(m, xc='b3lyp'), prhf),
        ('OH UHF', oh, scf.UHF, puhf),
        ('OH UKS PBE0', oh, lambda m: dft.UKS(m, xc='pbe0'), puhf)):
    mf = mk(mol).run(conv_tol=1e-11)
    mc = mf.to_cpu()
    if hasattr(mc, 'grids'):
        mc.grids.build()
    p = ref.Polarizability(mc)
    p.conv_tol = 1e-11
    for f in (0., .1):
        a = polarizability(mf, freq=f)
        if ref is puhf:
            td = tdscf.TDHF(mc) if label.endswith('UHF') else tdscf.TDDFT(mc)
            (aaa, aab, abb), (baa, bab, bbb) = td.get_ab()
            na = aaa.shape[0] * aaa.shape[1]
            nb = abb.shape[0] * abb.shape[1]
            A = np.block([[aaa.reshape(na, na), aab.reshape(na, nb)],
                          [aab.reshape(na, nb).T, abb.reshape(nb, nb)]])
            B = np.block([[baa.reshape(na, na), bab.reshape(na, nb)],
                          [bab.reshape(na, nb).T, bbb.reshape(nb, nb)]])
            charges = mol.atom_charges()
            ctr = np.einsum('i,ix->x', charges, mol.atom_coords()) / charges.sum()
            with mol.with_common_orig(ctr):
                r = mol.intor('int1e_r')
            mo, occ = mc.mo_coeff, mc.mo_occ
            h = np.hstack([np.einsum('xpq,pi,qa->xia', r, mo[s][:, occ[s] > 0],
                                     mo[s][:, occ[s] == 0]).reshape(3, -1)
                           for s in (0, 1)])
            n = A.shape[0]
            M = np.block([[A - f * np.eye(n), B], [B, A + f * np.eye(n)]])
            xy = np.linalg.solve(M, -np.vstack([h.T, h.T]))
            c = -h @ (xy[:n] + xy[n:])
            what = 'dense [A-w B; B A+w]'
        else:
            c = ref.polarizability_with_freq(p, freq=f)
            what = 'pyscf-properties'
        err = abs(a - c).max() / abs(c).max()
        report(f'{label} w={f} against {what}', err < 1e-7, f'rel {err:.1e}')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
