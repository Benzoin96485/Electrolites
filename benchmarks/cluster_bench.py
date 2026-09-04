"""End-to-end SCF and gradient wall clock.

The gradient is timed twice on the converged density and the second timing is
reported, so the number is the gradient and not the SCF's tail.  Check the
`conv=` field: if a patched and an unpatched run disagree on convergence they
did different amounts of work and their wall clocks are not comparable --
use fixed_density.py instead.

    python benchmarks/cluster_bench.py --xyz benchmarks/molecules/PfPMT.xyz \
        --charge -1 [--patch all]
"""
import json
import time

import cupy as cp
import numpy as np

from _common import apply_patch, build_mol, parser

a = parser(__doc__.splitlines()[0]).parse_args()
apply_patch(a.patch)

import gpu4pyscf                                            # noqa: E402
from gpu4pyscf import dft                                   # noqa: E402

mol = build_mol(a)
tag = f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}/{a.patch or "stock"}'
print(f'[{tag}] natm={mol.natm} nbas={mol.nbas} nao={mol.nao} '
      f'gpu4pyscf={gpu4pyscf.__version__}', flush=True)

mf = dft.RKS(mol, xc=a.xc)
mf.conv_tol = a.conv_tol
mf.direct_scf_tol = 1e-13
mf.grids.level = a.grids
mf.verbose = 0

t = time.perf_counter()
e = mf.kernel()
cp.cuda.Stream.null.synchronize()
t_scf = time.perf_counter() - t
print(f'[{tag}] SCF   {t_scf:9.2f} s   E={e:.10f} conv={mf.converged}',
      flush=True)

g = mf.nuc_grad_method()
g.verbose = 0
ts = []
for _ in range(2):
    cp.cuda.Stream.null.synchronize()
    t = time.perf_counter()
    de = g.kernel()
    cp.cuda.Stream.null.synchronize()
    ts.append(time.perf_counter() - t)
print(f'[{tag}] GRAD  {ts[-1]:9.2f} s   (first {ts[0]:.2f})  '
      f'|de|={np.linalg.norm(de):.10f}', flush=True)

print('RESULT ' + json.dumps(dict(
    tag=tag, gpu4pyscf=gpu4pyscf.__version__, xyz=a.xyz, basis=a.basis,
    xc=a.xc, patch=a.patch, natm=mol.natm, nao=int(mol.nao), e=float(e),
    conv=bool(mf.converged), t_scf=t_scf, t_grad=ts[-1],
    de=np.asarray(de).tolist())), flush=True)
