"""Compute an analytic Hessian and save it, so that two patch sets can be
compared as harmonic frequencies (`--compare a.npy b.npy`).

    python hess_save.py --xyz molecules/alanine_dipeptide.xyz --basis def2-svp \
        --xc wb97m-v --patch all --out ala_all.npy
    python hess_save.py --xyz molecules/alanine_dipeptide.xyz --compare \
        ala_none.npy ala_all.npy
"""
import time

import numpy as np

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--out', default='')
ap.add_argument('--compare', nargs=2, default=None)
a = ap.parse_args()

if a.compare:
    from pyscf.hessian import thermo
    mol = build_mol(a)
    h0, h1 = (np.load(f) for f in a.compare)
    f0 = thermo.harmonic_analysis(mol, h0)['freq_wavenumber']
    f1 = thermo.harmonic_analysis(mol, h1)['freq_wavenumber']
    f0 = np.real_if_close(f0); f1 = np.real_if_close(f1)
    print(f'max |dH| / max |H| = {np.abs(h1 - h0).max() / np.abs(h0).max():.3e}')
    print(f'max |d freq| = {np.abs(f1 - f0).max():.4f} cm-1 over {f0.size} modes'
          f' (lowest {np.sort(np.abs(f0))[:3]})')
    raise SystemExit(0)

apply_patch(a.patch)
import cupy as cp                                            # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
mol = build_mol(a)
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc)
mf.grids.level = a.grids
mf.nlcgrids.level = a.grids
mf.conv_tol = 1e-10
mf.kernel()
t = time.perf_counter()
h = mf.Hessian()
h.verbose = 0
hess = cp.asnumpy(cp.asarray(h.kernel()))
print(f'hessian {time.perf_counter() - t:.1f} s', flush=True)
np.save(a.out, hess)
