"""Periodic RHF/RKS stability on the GPU (``electrolites.pbc_stability``,
FEATURE_TARGETS X3): the internal and RHF->UHF Hessians element-wise
against PySCF's ``pbc.scf.newton_ah.gen_g_hop_rhf`` /
``pbc.scf.stability._gen_hop_rhf_external`` on the same orbitals, and the
drivers' lowest eigenvalue against the dense one.

PySCF's k-point KS external path requests a hermi=1 response and then
asserts hermi == 0 in ``nr_rks_fxc_st`` (a PySCF bug); the reference here
asks for hermi=0.  A Gamma-point SCF is compared with PySCF's k-point code
on the one-k-point version of the same cell.

    python tests/test_pbc_stability.py
"""
import sys

import numpy as np
import cupy as cp
from pyscf.pbc import gto
from pyscf.pbc import scf as cscf, dft as cdft
from pyscf.pbc.scf import newton_ah, stability as cstab
from gpu4pyscf.pbc import dft, scf

from electrolites import pbc_stability as ps

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


cell = gto.M(a='''0 1.7834 1.7834
1.7834 0 1.7834
1.7834 1.7834 0''', atom='C 0 0 0; C 0.8917 0.8917 0.8917',
             basis='gth-szv', pseudo='gth-pade', verbose=0, mesh=[21] * 3)
k2 = cell.make_kpts([2, 1, 1])
cases = (('Gamma PBE', lambda: dft.RKS(cell, xc='pbe'),
          lambda: cdft.KRKS(cell, xc='pbe')),
         ('Gamma HF', lambda: scf.RHF(cell), lambda: cscf.KRHF(cell)),
         ('2k PBE0', lambda: dft.KRKS(cell, k2, xc='pbe0'), None),
         ('2k HF', lambda: scf.KRHF(cell, k2), None))
for label, mk, ref_mk in cases:
    mf = mk()
    mf.conv_tol = 1e-10
    mf.kernel()
    mc = mf.to_cpu()
    if ref_mk is not None:
        ref = ref_mk()
        ref.mo_coeff, ref.mo_occ = [mc.mo_coeff], [mc.mo_occ]
        ref.mo_energy = [mc.mo_energy]
        if hasattr(ref, 'grids'):
            ref.grids = mc.grids
    else:
        ref = mc
    for external in (False, True):
        hop, hd, _ = ps._ops(mf, external)
        n = hd.size
        H = cp.asnumpy(hop(cp.eye(n)))
        if external:
            orig = ref.gen_response
            ref.gen_response = lambda *a, **k: orig(*a, **{**k, 'hermi': 0})
            h2 = cstab._gen_hop_rhf_external(ref)[0]
            Hc = np.array([np.asarray(h2(e)).real for e in np.eye(n)])
            ref.gen_response = orig
        else:
            h = newton_ah.gen_g_hop_rhf(ref, ref.mo_coeff, ref.mo_occ)[1]
            Hc = np.array([np.asarray(h(e)).real * 2 for e in np.eye(n)])
        err = abs(H - Hc).max() / abs(Hc).max()
        e_dense = np.linalg.eigvalsh(.5 * (Hc + Hc.T))[0]
        e_gpu = (ps.external if external else ps.internal)(mf, verbose=0)[2]
        report(f'{label} {"RHF->UHF" if external else "internal"}',
               err < 1e-8 and abs(e_gpu - e_dense) < 1e-6,
               f'H rel {err:.1e}; lowest {e_gpu:.6f} (dense {e_dense:.6f})')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
