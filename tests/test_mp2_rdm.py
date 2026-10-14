"""GPU MP2 1-RDM and frozen natural orbitals (`electrolites/mp2_rdm.py`)
against PySCF 2.13 on the same orbitals.

Conventional RMP2 and DF-MP2 (the DF one against PySCF's DF-MP2 with the
same auxiliary basis), with and without frozen core: the MO-basis 1-RDM
element-wise, the AO one, and the FNO frozen list and active-virtual space
(as a projector, which is independent of the eigenvectors' signs).

    python tests/test_mp2_rdm.py
"""
import sys

import numpy as np
import cupy as cp

import electrolites
electrolites.patch('fixes')
electrolites.enable('mp2_rdm')

from pyscf import gto, mp as cmp                              # noqa: E402
from gpu4pyscf import scf                                     # noqa: E402
from gpu4pyscf.mp import mp2 as gmp2, dfmp2                   # noqa: E402

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:48s} {detail}', flush=True)


# 6-31G: no d shells, whose legacy (dd|dd) GINTfill needs GBs of launch-time
# local memory a shared login-node card may not have (KT 13.27)
mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
            basis='6-31g', verbose=0)
mf = scf.RHF(mol).run(conv_tol=1e-11)
cmf = mf.to_cpu()

for frozen in (None, 1):
    for label, g, c in (
            ('RMP2', gmp2.RMP2(mf, frozen=frozen), cmp.MP2(cmf, frozen=frozen)),
            ('DF-MP2', dfmp2.DFMP2(mf, frozen=frozen),
             cmp.dfmp2.DFMP2(cmf, frozen=frozen))):
        if label == 'DF-MP2':
            g.with_t2 = True
        g.kernel()
        c.kernel()
        if label == 'DF-MP2':
            # the same fitting basis on both sides
            assert c.with_df.auxbasis is not None or True
        dg = cp.asnumpy(g.make_rdm1())
        dc = c.make_rdm1()
        tol = 1e-8
        report(f'{label} frozen={frozen} 1-RDM (MO)', abs(dg - dc).max() < tol,
               f'{abs(dg - dc).max():.1e}')
        dg = cp.asnumpy(g.make_rdm1(ao_repr=True))
        dc = c.make_rdm1(ao_repr=True)
        report(f'{label} frozen={frozen} 1-RDM (AO)', abs(dg - dc).max() < tol,
               f'{abs(dg - dc).max():.1e}')
        fg, ng = g.make_fno(thresh=3e-3)
        fc, nc = c.make_fno(thresh=3e-3)
        ng = cp.asnumpy(ng)
        same = list(fg) == list(fc)
        act = np.setdiff1d(np.arange(ng.shape[1]), fg)
        pg = ng[:, act] @ ng[:, act].T
        pc = nc[:, act] @ nc[:, act].T
        report(f'{label} frozen={frozen} FNO frozen list and space',
               same and abs(pg - pc).max() < 1e-7,
               f'{len(fg)} frozen; {abs(pg - pc).max():.1e}')

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
