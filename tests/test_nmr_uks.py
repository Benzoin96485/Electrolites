"""GIAO NMR shielding for UHF/UKS on the GPU (``electrolites.nmr_uks``,
FEATURE_TARGETS O1) against pyscf-properties' ``prop.nmr.uhf``/``uks`` on
the same SCF, and a closed-shell UKS against GPU4PySCF's RKS shielding.

pyscf-properties is not installed in barw-dev; the reference is a checkout
reached through PYSCF_EXT_PATH (default ~/vibeqc/_el_jobs/ext/properties).
Its UHF ``gen_vind`` stacks the alpha and beta responses with ``hstack``
over (nset, nmo, nocc) arrays, which fails whenever nocc_a != nocc_b (every
open-shell coupled case); the reference here uses the same function with
the two blocks flattened before stacking.  pyscf's CPHF stops at a residual
near 2e-7 (ours is refined to ~1e-12), so coupled cases agree to ~1e-6
relative; uncoupled (pure functional) cases agree to round-off.

    python tests/test_nmr_uks.py
"""
import os
import sys
from functools import reduce

os.environ.setdefault('PYSCF_EXT_PATH', os.path.expanduser(
    '~/vibeqc/_el_jobs/ext/properties'))
import numpy as np                                             # noqa: E402
import cupy as cp                                              # noqa: E402
from pyscf import gto                                          # noqa: E402
from pyscf.prop import nmr as cpu_nmr                          # noqa: E402
from pyscf.prop.nmr import uhf as cpu_uhf_nmr                  # noqa: E402
from gpu4pyscf import dft, scf                                 # noqa: E402

from electrolites.nmr_uks import NMR                           # noqa: E402

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


def gen_vind(mf, mo_coeff, mo_occ):
    """pyscf-properties' uhf.gen_vind with the spin blocks flattened."""
    vresp = mf.gen_response(hermi=2)
    oa, ob = mo_occ[0] > 0, mo_occ[1] > 0
    orboa, orbob = mo_coeff[0][:, oa], mo_coeff[1][:, ob]
    na, nb = orboa.shape[1], orbob.shape[1]
    nmo = mo_coeff[0].shape[1]

    def vind(mo1):
        mo1a = mo1[:, :na * nmo].reshape(-1, nmo, na)
        mo1b = mo1[:, na * nmo:].reshape(-1, nmo, nb)
        dm1 = np.asarray((
            [d - d.T for d in (reduce(np.dot, (mo_coeff[0], x, orboa.T))
                               for x in mo1a)],
            [d - d.T for d in (reduce(np.dot, (mo_coeff[1], x, orbob.T))
                               for x in mo1b)]))
        v1 = vresp(dm1)
        v1a = [reduce(np.dot, (mo_coeff[0].T, x, orboa)) for x in v1[0]]
        v1b = [reduce(np.dot, (mo_coeff[1].T, x, orbob)) for x in v1[1]]
        n = len(v1a)
        return np.hstack((np.asarray(v1a).reshape(n, -1),
                          np.asarray(v1b).reshape(n, -1))).ravel()
    return vind


cpu_uhf_nmr.gen_vind = gen_vind

oh = gto.M(atom='O 0 0 0; H 0 0 0.97', spin=1, basis='ccpvdz', verbose=0)
ch3 = gto.M(atom='C 0 0 0; H 1.08 0 0; H -0.54 0.935 0; H -0.54 -0.935 0.1',
            spin=1, basis='def2-svp', verbose=0)
water = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
              basis='ccpvdz', verbose=0)
for label, mol, xc in (('OH UHF', oh, 'HF'), ('OH UKS LDA', oh, 'lda'),
                       ('OH UKS PBE', oh, 'pbe'),
                       ('OH UKS B3LYP', oh, 'b3lyp'),
                       ('CH3 UKS PBE', ch3, 'pbe'),
                       ('CH3 UKS B3LYP', ch3, 'b3lyp')):
    mf = (scf.UHF(mol) if xc == 'HF' else dft.UKS(mol, xc=xc))
    mf.run(conv_tol=1e-11)
    g = NMR(mf).kernel()
    mfc = mf.to_cpu()
    if xc != 'HF':
        mfc.grids.build()
    c = (cpu_nmr.uhf.NMR if xc == 'HF' else cpu_nmr.uks.NMR)(mfc)
    c.verbose = 0
    ref = c.kernel()
    err = abs(g - ref).max() / abs(ref).max()
    coupled = xc in ('HF', 'b3lyp')
    report(f'{label} vs pyscf-properties', err < (2e-6 if coupled else 1e-10),
           f'rel {err:.1e}')

# closed-shell UKS == GPU4PySCF RKS shielding (its CPHF tol 1e-10)
from gpu4pyscf.properties import shielding                     # noqa: E402
for xc in ('pbe', 'b3lyp'):
    u = dft.UKS(water, xc=xc).run(conv_tol=1e-11)
    r = dft.RKS(water, xc=xc).run(conv_tol=1e-11)
    g = NMR(u).kernel()
    d, p = shielding.eval_shielding(r)
    ref = cp.asnumpy(d + p)
    err = abs(g - ref).max() / abs(ref).max()
    report(f'water closed-shell UKS {xc} == GPU4PySCF RKS', err < 1e-6,
           f'rel {err:.1e}')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
