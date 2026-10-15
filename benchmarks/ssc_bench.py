"""NMR spin-spin coupling timings (FEATURE_TARGETS §15, O2): all nucleus
pairs, FC (or FC+SD) + PSO + DSO, GPU (``electrolites.ssc_gpu``) or
pyscf-properties on the CPU (``--cpu``; PYSCF_EXT_PATH must reach it).

    python benchmarks/ssc_bench.py --xyz benchmarks/molecules/glycine.xyz \\
        --basis def2-svp --xc b3lyp [--fcsd] [--cpu]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parser, build_mol                        # noqa: E402

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--cpu', action='store_true')
ap.add_argument('--fcsd', action='store_true')
ap.add_argument('--df', action='store_true')
a = ap.parse_args()
os.environ.setdefault('PYSCF_EXT_PATH', os.path.expanduser(
    '~/vibeqc/_el_jobs/ext/properties'))

import numpy as np                                             # noqa: E402

mol = build_mol(a)
rec = dict(xyz=a.xyz, basis=a.basis, xc=a.xc, nao=mol.nao, natm=mol.natm,
           fcsd=a.fcsd, df=a.df, cpu=a.cpu,
           threads=os.environ.get('OMP_NUM_THREADS'))


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
    from pyscf.prop import ssc
else:
    import electrolites
    electrolites.patch_all()
    from gpu4pyscf import dft
    from electrolites.ssc_gpu import SSC
mf = dft.RKS(mol, xc=a.xc)
if a.df:
    mf = mf.density_fit()
mf.conv_tol = 1e-10
timed('scf_s', mf.kernel)
s = ssc.RKS(mf) if a.cpu else SSC(mf)
s.verbose = 0
s.with_fcsd = a.fcsd
e = timed('ssc_s', s.kernel)
rec['npair'] = len(s.nuc_pair)
rec['stages'] = getattr(s, 'timings', None)
rec['norm'] = float(np.linalg.norm(e))
print(f"[ssc {'CPU' if a.cpu else 'GPU'} {os.path.basename(a.xyz)}/{a.basis} "
      f"{a.xc}{' FC+SD' if a.fcsd else ''}] natm={mol.natm} nao={mol.nao} "
      f"scf_s {rec['scf_s']:.2f}  ssc_s {rec['ssc_s']:.2f}")
print('RESULT ' + json.dumps(rec))
