"""The gradient's two-electron per-atom derivative, ours against GPU4PySCF's,
at one density matrix and at two.

Both implementations run in one process on the same converged density and the
same `_VHFOpt`, so the ratio is the kernels and nothing else.

Two questions, and the second is the one that needs asking:

* **n_dm = 2** is the unrestricted case, which `fastejk` used to decline
  outright -- every angular-momentum class of an open-shell gradient went to
  GPU4PySCF.  This says what taking it back is worth.
* **n_dm = 1** is the closed-shell case, which the density loop that made
  n_dm = 2 possible now also runs, once.  The loop bound is a kernel argument,
  not a compile-time constant, so the unrolled contraction sits inside a
  runtime loop that was not there before.  This column is the check that the
  closed-shell path did not pay for the open-shell one.

    python benchmarks/ejk_ndm.py --xyz benchmarks/molecules/tmc_cucl2.xyz \
        --spin 1 --basis def2-svp
    python benchmarks/ejk_ndm.py --xyz benchmarks/molecules/glycine.xyz \
        --basis def2-svp          # closed shell: the n_dm = 1 row only
"""
import json
import time

import numpy as np

from _common import build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--k-factor', type=float, default=0.2)
ap.add_argument('--j-factor', type=float, default=1.0)
a = ap.parse_args()

import electrolites                                          # noqa: E402
electrolites.patch('fastejk')

import cupy as cp                                            # noqa: E402
import gpu4pyscf                                             # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.grad import rhf as grhf                       # noqa: E402
from gpu4pyscf.scf import jk as gjk                          # noqa: E402
from electrolites import fastejk                             # noqa: E402

mol = build_mol(a)
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.direct_scf_tol = 1e-13
mf.verbose = 0
mf.kernel()
assert mf.converged, 'SCF did not converge'
dm_spin = cp.asarray(mf.make_rdm1())
dm_tot = dm_spin if dm_spin.ndim == 2 else dm_spin[0] + dm_spin[1]

vhfopt = gjk._VHFOpt(mol, mf.direct_scf_tol).build()
tag = (f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/'
       f'{"UKS" if mol.spin else "RKS"}')
print(f'[{tag}] natm={mol.natm} nao={mol.nao} spin={mol.spin} '
      f'lmax={int(mol._bas[:, 1].max())} gpu4pyscf={gpu4pyscf.__version__}',
      flush=True)


def timeit(fn, reps):
    fn()
    cp.cuda.Stream.null.synchronize()
    ts = []
    for _ in range(reps):
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        out = fn()
        cp.cuda.Stream.null.synchronize()
        ts.append(time.perf_counter() - t)
    return min(ts), out


rows = {}
print(f'{"n_dm":>5s} {"GPU4PySCF":>10s} {"ours":>10s} {"ratio":>7s} '
      f'{"max|d|":>10s}')
cases = [('1', dm_tot)]
if dm_spin.ndim == 3:
    cases.append(('2', dm_spin))
for label, dm in cases:
    t_ref, ref = timeit(
        lambda dm=dm: grhf._jk_energy_per_atom_orig(
            vhfopt, dm, a.j_factor, a.k_factor), a.reps)
    t_our, our = timeit(
        lambda dm=dm: fastejk.jk_energy_per_atom(
            vhfopt, dm, a.j_factor, a.k_factor), a.reps)
    d = float(np.abs(np.asarray(our) - np.asarray(ref)).max())
    rel = d / max(float(np.abs(np.asarray(ref)).max()), 1e-30)
    rows[label] = dict(ref=t_ref, ours=t_our, ratio=t_ref / t_our,
                       absmax=d, rel=rel)
    print(f'{label:>5s} {t_ref:10.4f} {t_our:10.4f} {t_ref/t_our:6.2f}x '
          f'{d:10.2e}', flush=True)

print('RESULT ' + json.dumps(dict(
    tag=tag, xyz=a.xyz, basis=a.basis, natm=mol.natm, nao=int(mol.nao),
    spin=int(mol.spin), gpu4pyscf=gpu4pyscf.__version__, rows=rows)),
    flush=True)
