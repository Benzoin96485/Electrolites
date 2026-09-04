"""Time each build separately, on one fixed density.

Use this rather than an end-to-end wall clock whenever the SCF may not
converge, or may converge in a different number of cycles once patched -- on a
large ill-conditioned system a converged wall clock measures the convergence
path, not the kernels.  One `minao` guess is built once and every build is
timed on it.

    python benchmarks/fixed_density.py --xyz benchmarks/molecules/HcgC.xyz \
        --charge -4 [--patch all]
"""
import inspect
import json
import sys

import cupy as cp
import numpy as np

from _common import apply_patch, build_mol, make_timer, parser

a = parser(__doc__.splitlines()[0]).parse_args()
apply_patch(a.patch)

import gpu4pyscf                                            # noqa: E402
from gpu4pyscf import dft                                   # noqa: E402
from gpu4pyscf.lib.cupy_helper import tag_array             # noqa: E402
from gpu4pyscf.scf import jk as JK                          # noqa: E402

mol = build_mol(a)
tag = f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}/{a.patch or "stock"}'
print(f'[{tag}] natm={mol.natm} nbas={mol.nbas} nao={mol.nao} '
      f'gpu4pyscf={gpu4pyscf.__version__}', flush=True)

mf = dft.RKS(mol, xc=a.xc)
mf.direct_scf_tol = 1e-13
mf.grids.level = a.grids
mf.verbose = 0
dm = cp.asarray(mf.get_init_guess(mol, 'minao'))

res = {'tag': tag, 'gpu4pyscf': gpu4pyscf.__version__, 'xyz': a.xyz,
       'basis': a.basis, 'xc': a.xc, 'patch': a.patch,
       'natm': mol.natm, 'nao': int(mol.nao)}
timeit = make_timer(tag, res)

ni = mf._numint
omega, alpha, hyb = ni.rsh_and_hybrid_coeff(a.xc, spin=mol.spin)
new_abi = len(inspect.signature(JK._VHFOpt.get_k).parameters) >= 7

timeit('get_veff', lambda: mf.get_veff(mol, dm), a.reps)
timeit('get_j', lambda: mf.get_j(mol, dm, 1), a.reps)
timeit('get_k', (lambda: mf.get_k(mol, dm, 1, omega, alpha, hyb)) if new_abi
       else (lambda: mf.get_k(mol, dm, 1)), a.reps)
timeit('nr_rks', lambda: ni.nr_rks(mol, mf.grids, a.xc, dm), a.reps)

# the gradient, on the same density
mo_occ = cp.zeros(mol.nao)
mo_occ[:mol.nelectron // 2] = 2
mf.mo_coeff = cp.eye(mol.nao)
mf.mo_occ = mo_occ
mf.converged = True
gobj = mf.nuc_grad_method()
gobj.verbose = 0
timeit('grad_total', lambda: gobj.kernel(), max(2, a.reps // 2))

print('RESULT ' + json.dumps(res), flush=True)
