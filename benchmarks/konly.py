"""Just the exchange build, on a fixed density: the leanest A/B for a K change.

    python benchmarks/konly.py --xyz benchmarks/molecules/PfPMT.xyz \
        --charge -1 --basis def2-TZVP --patch fastk
"""
import inspect
import time

import cupy as cp

from _common import apply_patch, build_mol, parser

a = parser(__doc__.splitlines()[0]).parse_args()
apply_patch(a.patch)

from gpu4pyscf import dft                                   # noqa: E402
from gpu4pyscf.scf import jk as JK                          # noqa: E402

mol = build_mol(a)
mf = dft.RKS(mol, xc=a.xc)
mf.direct_scf_tol = 1e-13
mf.verbose = 0
dm = cp.asarray(mf.get_init_guess(mol, 'minao'))
omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(a.xc, spin=mol.spin)
new_abi = len(inspect.signature(JK._VHFOpt.get_k).parameters) >= 7
f = ((lambda: mf.get_k(mol, dm, 1, omega, alpha, hyb)) if new_abi
     else (lambda: mf.get_k(mol, dm, 1)))

f()
cp.cuda.Stream.null.synchronize()
ts = []
for _ in range(a.reps):
    cp.cuda.Stream.null.synchronize()
    t = time.perf_counter()
    f()
    cp.cuda.Stream.null.synchronize()
    ts.append(time.perf_counter() - t)
name = a.xyz.rsplit('/', 1)[-1]
print(f'{name}/{a.basis} nao={mol.nao} patch={a.patch or "stock"}  '
      f'get_k min={min(ts):.4f} s  median={sorted(ts)[len(ts) // 2]:.4f}  '
      f'all={[round(x, 4) for x in ts]}', flush=True)
