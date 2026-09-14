"""What the bra primitive-pair bound costs, and what widening it buys.

The kernels of `fastk`, `fastjk` and `fastejk` cache the bra shell pair's
primitive data in a fixed-size shared array, ``MAX_PRIM_PAIR`` doubles wide.
36 is 6x6, which is the worst case in a Pople set and was the whole story while
this repository measured on 6-31G\\*.  It is not the worst case in a def2 set:
def2-TZVP contracts a 3d metal's innermost s shell over **8** primitives and S
or Cl over **7**, and def2-QZVP reaches **10**, so bra pairs of 49, 56, 64 and
100 exist there and every class block built on one used to fall through.

`_ksplit.mpp_plan` now compiles a *second*, small module for those classes --
the one-thread-per-quartet family alone, up to the angular momentum a long
contraction can reach -- rather than widening the array for every kernel in the
package.  This script measures both halves of that trade in **one process**:

* which class blocks the bound moves, and how much of the build's time they
  are (``--report``);
* the build with the second module and without it, alternating, on the same
  converged density and the same ``_VHFOpt``.

Both bounds are live at once -- they are two entries of the same module cache,
keyed on the bound -- so the comparison is the kernels and not two processes on
a shared node.

    python benchmarks/prim_pair.py --xyz benchmarks/molecules/tmc_cucl2.xyz \\
        --spin 1 --basis def2-tzvp --xc b3lyp
    python benchmarks/prim_pair.py --xyz benchmarks/molecules/glycine.xyz \\
        --basis 6-31G*              # the regression check: nothing to widen

``--cap`` is the largest pair a second module will be built for; ``--cap 0`` is
the ablation and reproduces the behaviour before this was measured.
"""
import json
import time

import numpy as np

from _common import build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--cap', type=int, default=None,
                help='largest bra pair to compile a second module for '
                     '(default: ELECTROLITES_MAX_PRIM_PAIR, itself 100)')
ap.add_argument('--k-factor', type=float, default=0.2)
ap.add_argument('--j-factor', type=float, default=1.0)
ap.add_argument('--report', action='store_true',
                help='also time each angular-momentum class block of the '
                     'exchange build separately, and say which of them the '
                     'bound moves.  Adds a synchronise per class block, so '
                     'the totals it prints are not the totals above.')
a = ap.parse_args()

import electrolites                                          # noqa: E402
electrolites.patch('fastk', 'fastejk')

import cupy as cp                                            # noqa: E402
import gpu4pyscf                                             # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.grad import rhf as grhf                       # noqa: E402
from gpu4pyscf.scf import jk as gjk                          # noqa: E402
from electrolites import _ksplit, fastk, fastejk             # noqa: E402
from electrolites.compat import sorted_meta as _sorted_meta  # noqa: E402

CAP = _ksplit.MAX_PRIM_PAIR_CAP if a.cap is None else a.cap

mol = build_mol(a)
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.direct_scf_tol = 1e-13
mf.verbose = 0
if a.cycles:            # a fixed density, for a case whose SCF is awkward
    mf.max_cycle = a.cycles
    mf.kernel()
else:
    mf.kernel()
    assert mf.converged, 'SCF did not converge; --cycles N fixes a density'
dm_spin = cp.asarray(mf.make_rdm1())
dm_tot = dm_spin if dm_spin.ndim == 2 else dm_spin[0] + dm_spin[1]

vhfopt = gjk._VHFOpt(mol, mf.direct_scf_tol).build()
uniq_l_ctr = _sorted_meta(vhfopt)[0]
groups = [(int(l), int(p)) for l, p in uniq_l_ctr[:, :2]]
mpp, mpp_lmax = _ksplit.mpp_plan(uniq_l_ctr, fastk.MAX_PRIM_PAIR, CAP)

tag = (f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/'
       f'{"UKS" if mol.spin else "RKS"}')
print(f'[{tag}] natm={mol.natm} nao={mol.nao} spin={mol.spin} '
      f'lmax={int(mol._bas[:, 1].max())} gpu4pyscf={gpu4pyscf.__version__}')
print(f'# (l, nprim) groups: {groups}')
if mpp == fastk.MAX_PRIM_PAIR:
    print(f'# no bra pair in this basis exceeds {fastk.MAX_PRIM_PAIR}: '
          f'nothing to widen, and the two columns below should agree')
else:
    print(f'# longest bra pair {mpp} > {fastk.MAX_PRIM_PAIR}; the second '
          f'module carries the classes up to l={mpp_lmax}')

sdm = cp.asarray(vhfopt.apply_coeff_C_mat_CT(
    dm_spin if dm_spin.ndim == 3 else dm_spin[None]), order='C')


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


def at_cap(cap, fn, reps):
    """Time ``fn`` with the second module allowed up to ``cap``.

    The bound is read out of `_ksplit` by every build, and the compiled
    modules are cached per bound, so flipping it here swaps which kernel a
    wide bra pair is launched from and nothing else.
    """
    old = _ksplit.MAX_PRIM_PAIR_CAP
    _ksplit.MAX_PRIM_PAIR_CAP = cap
    try:
        return timeit(fn, reps)
    finally:
        _ksplit.MAX_PRIM_PAIR_CAP = old


def get_k():
    return fastk.get_k(vhfopt, sdm, 1, 0, 0.0, 1.0, 1.0)


def ejk():
    return fastejk.jk_energy_per_atom(
        vhfopt, dm_spin, a.j_factor, a.k_factor)


rows = {}
print(f'\n{"term":>26s} {"bound 36":>10s} {f"bound {mpp}":>10s} '
      f'{"ratio":>7s} {"max|d|":>10s}')
for label, fn in (('exchange build (get_k)', get_k),
                  ('gradient ejk (per atom)', ejk)):
    # alternate, so a drift in the node's load shows up in both columns
    t_off, off = at_cap(0, fn, a.reps)
    t_on, on = at_cap(CAP, fn, a.reps)
    t_off = min(t_off, at_cap(0, fn, a.reps)[0])
    t_on = min(t_on, at_cap(CAP, fn, a.reps)[0])
    d = float(np.abs(cp.asnumpy(cp.asarray(on) - cp.asarray(off))).max())
    rel = d / max(float(np.abs(cp.asnumpy(cp.asarray(off))).max()), 1e-30)
    rows[label] = dict(off=t_off, on=t_on, ratio=t_off / t_on, absmax=d,
                       rel=rel)
    print(f'{label:>26s} {t_off:10.4f} {t_on:10.4f} {t_off/t_on:6.3f}x '
          f'{d:10.2e}')

if a.report:
    # Per class block, with the bound at its default, so that "fell through"
    # means "the bound sent it to GPU4PySCF".
    import collections
    rec = collections.defaultdict(lambda: [0, 0.0])
    state = {}
    orig = {n: getattr(fastk, n) for n in
            ('_run_k1', '_run_k2', '_run_general', '_run_ref_new', '_run_ref')
            if hasattr(fastk, n)}

    def wrap(name, kind):
        fn = orig[name]

        def f(*args, **kw):
            cp.cuda.Stream.null.synchronize()
            t = time.perf_counter()
            r = fn(*args, **kw)
            cp.cuda.Stream.null.synchronize()
            e = rec[(state.get('lll'), state.get('prim'), kind)]
            e[0] += 1
            e[1] += time.perf_counter() - t
            return r
        setattr(fastk, name, f)

    orig_classes = fastk._classes

    def classes(self):
        nprim = _sorted_meta(self)[0][:, 1]
        for idx, lll in orig_classes(self):
            state['lll'] = lll
            state['prim'] = tuple(int(nprim[x]) for x in idx)
            yield idx, lll
    fastk._classes = classes
    for n, kind in (('_run_k1', 'ours'), ('_run_k2', 'ours'),
                    ('_run_general', 'ours'), ('_run_ref_new', 'stock'),
                    ('_run_ref', 'stock')):
        if n in orig:
            wrap(n, kind)
    at_cap(0, get_k, 1)
    rec.clear()
    at_cap(0, get_k, 1)
    wide = {k: v for k, v in rec.items()
            if k[1] and k[1][0] * k[1][1] > fastk.MAX_PRIM_PAIR}
    tot = sum(v[1] for v in rec.values())
    twide = sum(v[1] for v in wide.values())
    print(f'\n# at the default bound: {len(wide)} of {len(rec)} class blocks '
          f'have a bra pair past it, and they are {twide:.4f}s of '
          f'{tot:.4f}s ({100 * twide / max(tot, 1e-30):.1f}%)')
    for k, v in sorted(wide.items(), key=lambda x: -x[1][1])[:8]:
        print(f'#   class {k[0]} nprim {k[1]} -> {k[2]:5s} {v[1]:.4f}s')
    for n, f in orig.items():
        setattr(fastk, n, f)
    fastk._classes = orig_classes

print('RESULT ' + json.dumps(dict(
    tag=tag, xyz=a.xyz, basis=a.basis, natm=mol.natm, nao=int(mol.nao),
    spin=int(mol.spin), groups=groups, mpp=mpp, mpp_lmax=mpp_lmax, cap=CAP,
    gpu4pyscf=gpu4pyscf.__version__, rows=rows)), flush=True)
