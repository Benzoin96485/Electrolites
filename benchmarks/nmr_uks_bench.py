"""UKS GIAO NMR shielding timings (FEATURE_TARGETS §15, O1): GPU
(``electrolites.nmr_uks``) or pyscf-properties on the CPU (``--cpu``;
PYSCF_EXT_PATH must reach it).

    python benchmarks/nmr_uks_bench.py --xyz benchmarks/molecules/glycine.xyz \\
        --charge 1 --spin 1 --basis def2-svp --xc b3lyp [--cpu]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parser, build_mol                        # noqa: E402

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--cpu', action='store_true')
a = ap.parse_args()
os.environ.setdefault('PYSCF_EXT_PATH', os.path.expanduser(
    '~/vibeqc/_el_jobs/ext/properties'))

import numpy as np                                             # noqa: E402

mol = build_mol(a)
rec = dict(xyz=a.xyz, basis=a.basis, xc=a.xc, nao=mol.nao, natm=mol.natm,
           spin=mol.spin, cpu=a.cpu, threads=os.environ.get('OMP_NUM_THREADS'))


def timed(label, fn):
    sync = (lambda: None) if a.cpu else \
        __import__('cupy').cuda.Stream.null.synchronize
    sync()
    t = time.perf_counter()
    out = fn()
    sync()
    rec[label] = time.perf_counter() - t
    return out


if a.cpu:
    from pyscf import dft
    from pyscf.prop import nmr
else:
    import electrolites
    electrolites.patch_all()
    from gpu4pyscf import dft
    from electrolites.nmr_uks import NMR
if not a.cpu:
    from pyscf import gto
    w = gto.M(atom='O 0 0 0; H 0 0 0.97', spin=1, basis=a.basis, verbose=0)
    NMR(dft.UKS(w, xc=a.xc).run()).kernel()
mf = dft.UKS(mol, xc=a.xc)
mf.conv_tol = 1e-10
timed('scf_s', mf.kernel)
s = nmr.uks.NMR(mf) if a.cpu else NMR(mf)
s.verbose = 0
if a.cpu:
    # pyscf-properties' UHF gen_vind fails for nocc_a != nocc_b; the same
    # function with the spin blocks flattened (tests/test_nmr_uks.py)
    from functools import reduce
    from pyscf.prop.nmr import uhf as U

    def gen_vind(mf, mo_coeff, mo_occ):
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
    U.gen_vind = gen_vind
e = timed('nmr_s', s.kernel)
rec['stages'] = getattr(s, 'timings', None)
rec['iso'] = (np.trace(np.asarray(e), axis1=1, axis2=2) / 3).tolist()
print(f"[nmr_uks {'CPU' if a.cpu else 'GPU'} {os.path.basename(a.xyz)}/"
      f"{a.basis} {a.xc}] natm={mol.natm} nao={mol.nao} "
      f"scf_s {rec['scf_s']:.2f}  nmr_s {rec['nmr_s']:.2f}")
print('RESULT ' + json.dumps(rec))
