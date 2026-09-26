"""How far the ip2 term moves when the fitting metric moves by one ulp (W24).

Two implementations of the same term that differ by 1e-9 do not say which of
them is off; this does.  The metric ``int2c2e`` returns is multiplied
element-wise by ``1 + 2.2e-16 r`` (``r`` symmetric, standard normal) before
either path factorises it, and each path is compared with itself.  A path
whose answer moves by as much as it differs from the other is carrying the
metric's round-off in its answer; one that does not move is not.

    python benchmarks/df_hess_metric_sens.py --xyz benchmarks/molecules/tmc_ni_dithiolene.xyz \\
        --charge -2 [--basis def2-svp] [--xc b3lyp] [--seeds 3]
"""
import os

import numpy as np
import cupy as cp

from _common import build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--seeds', type=int, default=3)
ap.add_argument('--eps', type=float, default=2.2e-16)
ap.set_defaults(basis='def2-svp', xc='b3lyp')
a = ap.parse_args()

import electrolites                                          # noqa: E402
electrolites.patch('fastdfhessip')
from electrolites import fastdfhessip as F                   # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.df import int3c2e_bdiv as B                   # noqa: E402
from gpu4pyscf.df.hessian import rhf as DH                   # noqa: E402

mol = build_mol(a)
mf = dft.RKS(mol, xc=a.xc).density_fit()
mf.conv_tol = 1e-10
mf.verbose = 0
mf.kernel()
mf.with_df.build(build_cderi=False)
opt = mf.with_df.intopt
dm0 = mf.make_rdm1()
omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=0)

orig = B.int2c2e
pert = {'eps': 0., 'seed': 0}


def int2c2e(*args, **kw):
    j = orig(*args, **kw)
    if pert['eps']:
        r = cp.random.default_rng(pert['seed']).standard_normal(j.shape)
        j = j * (1 + pert['eps'] * (r + r.T) * .5)
    return j


# every name the three paths reach it by
B.int2c2e = DH.int2c2e = F.int2c2e = int2c2e
up = DH._jk_energy_per_atom_orig(opt, dm0, 1., hyb)
res = {}
for label, env in (('eigen', {'FASTDFHESSIP_METRIC': 'eigen'}),
                   ('chol', {'FASTDFHESSIP_METRIC': 'chol'}),
                   ('mixed', {'FASTDFHESSIP_METRIC': 'mixed'}),
                   ('chol TRI=solve', {'FASTDFHESSIP_METRIC': 'chol',
                                       'FASTDFHESSIP_TRI': 'solve'})):
    for k in ('FASTDFHESSIP_METRIC', 'FASTDFHESSIP_TRI'):
        os.environ.pop(k, None)
    os.environ.update(env)
    pert['eps'] = 0.
    base = F._jk_energy_per_atom(opt, dm0, 1., hyb)
    moved = []
    for seed in range(1, a.seeds + 1):
        pert.update(eps=a.eps, seed=seed)
        out = F._jk_energy_per_atom(opt, dm0, 1., hyb)
        moved.append(float(abs(out - base).max() / abs(base).max()))
    res[label] = base
    print(f'{label:15s} vs upstream '
          f'{float(abs(base - up).max() / abs(up).max()):.2e}   moved by a '
          f'1-ulp metric: ' + ' '.join(f'{x:.2e}' for x in moved), flush=True)
pert['eps'] = 0.
w = cp.linalg.eigvalsh(orig(opt.auxmol))
print(f'metric eigenvalues {float(w.min()):.3e} .. {float(w.max()):.3e}, '
      f'condition {float(w.max() / w.min()):.2e}')
print('stats', F.stats())
