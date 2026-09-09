"""How the exchange build's cost grows with the number of density matrices.

Catalogue item **B4b**.  GPU4PySCF's K kernels amortise the integrals over the
densities properly -- the ``for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm)`` loop
sits around the *contraction*, inside the shell quartet -- so unlike the
Coulomb build (B4a, ``docs/ROUND3_B4A.md``) there is no block cap to remove.
What B4b is really about is therefore the split of the cost:

    T(n) = R  +  n * C  +  L

with ``R`` the density-independent work per quartet (the Rys roots, the 2D
recurrences, the geometry, the screening), ``C`` the per-density contraction
and its atomicAdds, and ``L`` a launch constant.  ``R`` is what the P1-P5
treatment in ``fastk`` made cheaper; ``C`` it left alone.  So the ceiling for a
rewritten kernel at large ``n`` is set by ``C``, and this measures both halves
on both codes by fitting a line through the measured ``n``.

Every density is a copy of the same matrix, so ``dm_cond`` -- and therefore the
surviving quartet list -- is identical at every ``n``.  That isolates the
per-density cost from the union-screening question, which is B4c and
``benchmarks/ndm_screen_probe.py``.

That holds exactly for ``hermi = 1`` only.  The ``hermi = 0`` rows have to
break the symmetry the caller would have broken, and they do it with a
perturbation that grows with the density index, so ``dm_cond`` -- an absmax
over the densities -- rises slightly with ``n`` and the screening loosens a
little.  Read those rows for the *ratio* between the two codes and for the
factor of two in ``C`` that the doubled density stack explains, not as a clean
fit.

    python benchmarks/ndm_k.py --xyz benchmarks/molecules/penicillinV.xyz \
        --basis def2-svp --cycles 1
"""
import json
import os
import time

import cupy as cp
import numpy as np

from _common import build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--ndm', default='1,2,4,8,16,32',
                help='comma separated n_dm values to time')
ap.add_argument('--hermi', default='1', help='comma separated hermi values')
ap.add_argument('--no-check', action='store_true',
                help='skip the element-wise check against GPU4PySCF, which is '
                     'otherwise done at every n_dm')
a = ap.parse_args()

# Both implementations have to live in one process, so the patch is not applied
# through electrolites.patch(): the original bound method is kept by hand.
from gpu4pyscf import scf as gscf                            # noqa: E402
from gpu4pyscf.scf import jk as JK                           # noqa: E402
_ref_get_k = JK._VHFOpt.get_k
import electrolites.fastk as fastk                           # noqa: E402
from electrolites.compat import NEW_JK_ABI                   # noqa: E402
import gpu4pyscf                                             # noqa: E402

# 1.8.x takes the operator as (omega, lr_factor, sr_factor); 1.7.x rides it on
# the option object's mol, so the extra arguments are absent there.
_RSH = (0, 0.0, 1.0, 1.0) if NEW_JK_ABI else (0,)

_sync = cp.cuda.Stream.null.synchronize
mol = build_mol(a)
tag = f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}'
print(f'# gpu4pyscf {gpu4pyscf.__version__}  '
      f'{cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}')
print(f'# {tag} natm={mol.natm} nbas={mol.nbas} nao={mol.nao}  '
      f'FASTK_NO_NDM={os.environ.get("FASTK_NO_NDM", "")}', flush=True)

mf = gscf.RHF(mol)
mf.direct_scf_tol = 1e-13
dm = mf.get_init_guess(mol, 'minao')
if a.cycles:
    mf.max_cycle = a.cycles
    mf.conv_tol = 1e-12
    mf.kernel(dm0=dm)
    dm = cp.asarray(mf.make_rdm1())
    print(f'# density after {a.cycles} SCF cycle(s)', flush=True)
dm = cp.asarray(dm).reshape(mol.nao, mol.nao)

opt = JK._VHFOpt(mol, mf.direct_scf_tol).build()
sorted_dm = opt.apply_coeff_C_mat_CT(dm.reshape(1, mol.nao, mol.nao))[0]
nao = sorted_dm.shape[-1]


def timeit(fn, reps):
    out = fn()
    _sync()
    ts = []
    for _ in range(reps):
        _sync()
        t = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t)
    return min(ts), out


def fit(ns, ts):
    """Least squares (R+L, C) of T = (R+L) + n*C, and the ratio R/C."""
    A = np.stack([np.ones(len(ns)), np.asarray(ns, float)], axis=1)
    (base, per), *_ = np.linalg.lstsq(A, np.asarray(ts, float), rcond=None)
    return base, per


ROWS = []
for hermi in [int(x) for x in a.hermi.split(',')]:
    print(f'\nhermi = {hermi}')
    print(f'{"n_dm":>5s} {"gpu4pyscf/s":>12s} {"ours/s":>10s} {"ratio":>7s} '
          f'{"max|dK|":>10s}')
    print('-' * 50)
    ns, t_ref, t_our = [], [], []
    for n_dm in [int(x) for x in a.ndm.split(',')]:
        dms = cp.repeat(sorted_dm[None], n_dm, axis=0)
        if hermi == 0:                    # break the symmetry the caller broke
            dms = dms + 1e-3 * cp.arange(n_dm)[:, None, None] * cp.tril(
                cp.ones((nao, nao)))
        tr, vref = timeit(
            lambda: _ref_get_k(opt, dms.copy(), hermi, *_RSH),
            a.reps)
        to, vour = timeit(
            lambda: fastk.get_k(opt, dms.copy(), hermi, *_RSH),
            a.reps)
        err = (float('nan') if a.no_check
               else float(cp.abs(vour - vref).max()))
        print(f'{n_dm:5d} {tr:12.4f} {to:10.4f} {tr/to:6.2f}x {err:10.2e}',
              flush=True)
        ns.append(n_dm)
        t_ref.append(tr)
        t_our.append(to)
        ROWS.append(dict(hermi=hermi, n_dm=n_dm, ref=tr, ours=to, err=err))
    if len(ns) >= 3:
        b_r, c_r = fit(ns, t_ref)
        b_o, c_o = fit(ns, t_our)
        print(f'  fit T = base + n_dm*C')
        print(f'    gpu4pyscf  base {b_r:8.4f} s   C {c_r:8.5f} s/density')
        print(f'    ours       base {b_o:8.4f} s   C {c_o:8.5f} s/density'
              f'   -> base {b_r/b_o:.2f}x, C {c_r/c_o:.2f}x')
        print(f'    our asymptote as n_dm -> inf: {c_r/c_o:.2f}x')
        ROWS.append(dict(hermi=hermi, fit_ref=[b_r, c_r], fit_ours=[b_o, c_o]))

print('\n' + json.dumps({'tag': tag, 'natm': mol.natm, 'nao': int(mol.nao),
                         'rows': ROWS}))
