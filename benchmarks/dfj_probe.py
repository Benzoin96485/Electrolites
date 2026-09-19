"""RIJONX against full DF and against the direct path, per call and end to end.

`docs/DF_WORK_LIST.md` W1.  Three ways to build the two-electron half of a
hybrid, on one density, in one process:

* **full DF** -- ``df_jk.get_jk`` contracting the cderi, which is what
  ``density_fit()`` does and what `fastdf` rearranges.  It needs the tensor:
  86.4 GB at 284 atoms, host-resident, and a one-time ``_cholesky_eri``.
* **RIJONX** -- ``density_fit(only_dfj=True)``: J from the three-centre
  ``j_engine_3c2e`` contraction, which builds **no cderi at all**, and K from
  the exact four-centre build, which is `fastk`.  Broken in GPU4PySCF 1.8.1
  and repaired by `fastdfj`; without that module this script has nothing to
  measure.
* **direct** -- neither fitted: ``scf/j_engine.py`` and ``scf/jk.py``, i.e.
  `fastj` and `fastk`.

The per-call rows are what W1's acceptance criterion is stated in: the bar for
the whole RIJONX build is full DF's ``get_jk(J+K)``, and the bar for its
Coulomb half alone is full DF's J-only call.  The one-time rows are the other
half of the comparison -- a cderi that takes a minute to build and 86 GB to
hold does not appear in a per-call number at all.

Variants are timed **round robin** rather than one after the other, because
the node is shared and a competing job's load drifts over the minutes a row
takes (`docs/DF_WORK_LIST.md` §5.2).

    python benchmarks/dfj_probe.py --xyz benchmarks/molecules/taxol.xyz \
        --basis def2-svp --xc b3lyp --patch all --cycles 1
    python benchmarks/dfj_probe.py --xyz ... --no-fulldf   # skip the cderi
    python benchmarks/dfj_probe.py --xyz ... --scf         # whole SCF runs too
"""
import json
import time

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--no-fulldf', action='store_true',
                help='skip the full-DF variant, which is the only one that '
                     'builds a cderi -- 86 GB and a minute at 284 atoms')
ap.add_argument('--no-direct', action='store_true')
ap.add_argument('--scf', action='store_true',
                help='also run each variant to convergence.  Every pair is '
                     'run twice and the second is quoted, because the first '
                     'process of a Slurm job pays a cold start '
                     '(docs/DF_WORK_LIST.md 5.3)')
ap.add_argument('--scf-max-cycle', type=int, default=50)
ap.add_argument('--scf-reps', type=int, default=2,
                help='how many times to run each end-to-end variant; the '
                     'last is the one to quote (the first process of a Slurm '
                     'job pays a cold start)')
ap.add_argument('--no-percall', action='store_true',
                help='skip the fixed-density table and measure only whole '
                     'runs -- at 284 atoms the two together do not fit in a '
                     'debug allocation')
ap.add_argument('--scf-ablate', action='store_true',
                help='add a fourth end-to-end variant: RIJONX with '
                     '`FASTDFJ_NO_INCREMENTAL`, i.e. GPU4PySCF\'s own '
                     '`_DFHF.get_veff`, which rebuilds the exchange from the '
                     'whole density every cycle instead of from `D - D_last`. '
                     'The gap between the two is what the driver half of '
                     '`fastdfj` is worth')
a = ap.parse_args()
apply_patch(a.patch)

import gpu4pyscf                                             # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.df import df as gdf                           # noqa: E402

_sync = cp.cuda.Stream.null.synchronize
R = {}
MEM = {}


def _mk(kind):
    """One mean-field object of each of the three kinds."""
    mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc)
    if kind == 'rijonx':
        mf = mf.density_fit(auxbasis=a.auxbasis, only_dfj=True)
    elif kind == 'fulldf':
        mf = mf.density_fit(auxbasis=a.auxbasis)
    mf.conv_tol = a.conv_tol
    mf.grids.level = a.grids
    mf.verbose = 0
    return mf


def timeit(label, fn, reps=None):
    """One variant on its own; `roundrobin` is preferred where it applies."""
    return roundrobin([(label, fn)], reps)


def roundrobin(items, reps=None):
    """Time every variant once, then again, `reps` times over.

    A sequential schedule puts the whole of a competing job's load drift on
    whichever variant happened to run while it was there.  Interleaving does
    not remove the noise but spreads it over every row equally, which is what
    makes the *ratio* between two rows meaningful.
    """
    reps = a.reps if reps is None else reps
    ts = {label: [] for label, _ in items}
    dead = set()
    for label, fn in items:                       # warm up / catch failures
        try:
            fn()
            _sync()
        except Exception as exc:                                 # noqa: BLE001
            R[label] = f'ERR {type(exc).__name__}: {exc}'
            print(f'{label:38s} ERR {type(exc).__name__}: {str(exc)[:90]}',
                  flush=True)
            dead.add(label)
    for _ in range(reps):
        for label, fn in items:
            if label in dead:
                continue
            _sync()
            t = time.perf_counter()
            fn()
            _sync()
            ts[label].append(time.perf_counter() - t)
    out = {}
    for label, _ in items:
        if label in dead:
            continue
        R[label] = out[label] = min(ts[label])
        print(f'{label:38s} {min(ts[label])*1e3:10.2f} ms   (min of {reps})',
              flush=True)
    return out


mol = build_mol(a)
tag = (f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.auxbasis or "auto"}/{a.xc}/'
       f'{a.patch or "stock"}')

# The density every per-call row is measured on.  It comes from the **RIJONX**
# object because that is the one variant that needs no cderi to produce one --
# a `minao` guess screens far harder than a converged density and would not be
# the build the SCF runs (`benchmarks/_common.py --cycles`).
ref = _mk('rijonx')
ref.max_cycle = max(1, a.cycles)
try:
    ref.kernel()
except Exception as exc:                                         # noqa: BLE001
    raise SystemExit(f'the RIJONX reference density failed: {exc!r}.  '
                     'Without `fastdfj` this script has nothing to measure; '
                     'pass --patch all or --patch fastdfj.')
dm = ref.make_rdm1()

# Perlmutter has both 40 GB and 80 GB A100 nodes behind `-C gpu`, and which
# one a job lands on decides whether a cderi is resident or streamed -- i.e.
# most of what this script measures.  Record it with the numbers.
_props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
GPU = (_props['name'].decode() if isinstance(_props['name'], bytes)
       else str(_props['name']))
print(f'[{tag}] natm={mol.natm} nao={mol.nao} '
      f'naux={ref.with_df.auxmol.nao} spin={mol.spin} charge={mol.charge} '
      f'cycles={a.cycles} gpu={GPU} gpu4pyscf={gpu4pyscf.__version__}',
      flush=True)

# ---- the one-time costs ----------------------------------------------------
# RIJONX's is the three-centre option object `df_jk.get_j` builds on its first
# call; full DF's is that plus `_cholesky_eri` and the tensor it returns.
print('\n--- one-time ---', flush=True)
mfs = {'rijonx': ref}

# RIJONX's whole one-time cost, which is the honest thing to set against
# `_cholesky_eri`: a *fresh* object's first Coulomb call builds the sorted
# three-centre option object, the auxiliary metric and its Cholesky factor,
# and `j_engine_3c2e`'s own option object -- and no tensor.
_fresh = _mk('rijonx')
_sync()
_t = time.perf_counter()
_fresh.get_jk(mol, dm, 1, with_j=True, with_k=False)
_sync()
R['rijonx setup + first J (one-time)'] = time.perf_counter() - _t
print(f'{"rijonx setup + first J":38s} '
      f'{R["rijonx setup + first J (one-time)"]*1e3:10.2f} ms   '
      f'no tensor', flush=True)
del _fresh
if a.no_fulldf:
    pass
elif a.no_percall:
    # the tensor is not needed for anything but the fixed-density table, and
    # building it twice (here and inside the SCF) does not fit in a debug
    # allocation at 284 atoms.  The end-to-end rows pay it where it belongs.
    mfs['fulldf'] = _mk('fulldf')
    print('cderi build                            deferred to the SCF runs '
          '(--no-percall)', flush=True)
else:
    full = _mk('fulldf')
    _sync()
    t = time.perf_counter()
    full.with_df.build()
    _sync()
    R['cderi build (one-time)'] = time.perf_counter() - t
    cderi = full.with_df._cderi[cp.cuda.Device().id]
    MEM['cderi GB'] = cderi.nbytes / 1e9
    MEM['cderi on gpu'] = bool(isinstance(cderi, cp.ndarray))
    MEM['blksize'] = int(full.with_df.get_blksize(mem_fraction=0.4))
    print(f'{"cderi build (_cholesky_eri)":38s} '
          f'{R["cderi build (one-time)"]*1e3:10.2f} ms   '
          f'{MEM["cderi GB"]:.2f} GB on '
          f'{"gpu" if MEM["cderi on gpu"] else "host"}, '
          f'blksize {MEM["blksize"]}', flush=True)
    mfs['fulldf'] = full
if a.no_fulldf:
    print('cderi build                            skipped (--no-fulldf)',
          flush=True)
if not a.no_direct:
    mfs['direct'] = _mk('direct')

# ---- the per-call rows -----------------------------------------------------
print('\n--- per call, on the same density ---', flush=True)


def _jk(mf, wj, wk):
    return lambda: mf.get_jk(mol, dm, 1, with_j=wj, with_k=wk)


def _direct(mf, wj, wk):
    if wj and wk:
        return lambda: (mf.get_j(mol, dm, 1), mf.get_k(mol, dm, 1))
    if wj:
        return lambda: mf.get_j(mol, dm, 1)
    return lambda: mf.get_k(mol, dm, 1)


items = []
for kind, mf in ({} if a.no_percall else mfs).items():
    mk = _direct if kind == 'direct' else _jk
    for what, (wj, wk) in (('J+K', (True, True)), ('J', (True, False)),
                           ('K', (False, True))):
        items.append((f'{kind:7s} {what}', mk(mf, wj, wk)))
if items:
    roundrobin(items)
else:
    print('skipped (--no-percall)', flush=True)

# ---- what the rows say -----------------------------------------------------
def _get(k):
    v = R.get(k)
    return v if isinstance(v, float) else None


print('\n--- W1 ---', flush=True)
bar_jk, bar_j = _get('fulldf  J+K'), _get('fulldf  J')
got_jk, got_j = _get('rijonx  J+K'), _get('rijonx  J')
if bar_jk and got_jk:
    print(f'whole build   RIJONX {got_jk:7.3f} s  against full DF '
          f'{bar_jk:7.3f} s   -> {bar_jk/got_jk:5.2f}x', flush=True)
    R['ratio J+K'] = bar_jk / got_jk
if bar_j and got_j:
    print(f'Coulomb half  RIJONX {got_j:7.3f} s  against full DF '
          f'{bar_j:7.3f} s   -> {bar_j/got_j:5.2f}x', flush=True)
    R['ratio J'] = bar_j / got_j
d_j, d_k = _get('direct  J'), _get('direct  K')
if d_j and got_j:
    print(f'and the fitted Coulomb against the direct one: '
          f'{got_j:7.3f} s against {d_j:7.3f} s   -> {d_j/got_j:5.2f}x',
          flush=True)
    R['ratio J vs direct'] = d_j / got_j

# ---- end to end ------------------------------------------------------------
if a.scf:
    print('\n--- whole SCF (the second of two runs of each) ---', flush=True)
    from electrolites import fastdfj                          # noqa: E402
    E = {}
    kinds = list(mfs)
    if a.scf_ablate:
        kinds.insert(1, 'rijonx/whole-dm')
    for rep in range(a.scf_reps):
        for kind in kinds:
            # the ablation differs from `rijonx` only in the driver: the same
            # object, the same J and the same K, rebuilt from the whole
            # density each cycle rather than from the difference
            fastdfj._NO_INCREMENTAL = (kind == 'rijonx/whole-dm')
            mf = _mk('rijonx' if kind.startswith('rijonx') else kind)
            mf.max_cycle = a.scf_max_cycle
            _sync()
            t = time.perf_counter()
            try:
                e = mf.kernel()
            except Exception as exc:                             # noqa: BLE001
                R[f'scf {kind}'] = f'ERR {type(exc).__name__}: {exc}'
                print(f'scf {kind:34s} ERR {type(exc).__name__}: '
                      f'{str(exc)[:80]}', flush=True)
                fastdfj._NO_INCREMENTAL = False
                continue
            _sync()
            dt = time.perf_counter() - t
            # every run is printed and the *second* is the one to quote; the
            # first process of a Slurm job pays a cold start large enough to
            # read as a 2.66x on its own (docs/DF_WORK_LIST.md 5.3).  Printing
            # both also means a job that runs out of wall clock still leaves a
            # usable row behind.
            R[f'scf {kind} run{rep}'] = dt
            R[f'scf {kind}'] = dt
            E[kind] = (float(e), int(mf.cycles), bool(mf.converged))
            fastdfj._NO_INCREMENTAL = False
            print(f'scf {kind:16s} run{rep} {dt:10.3f} s  '
                  f'cycles={mf.cycles:3d}  E={e:.10f}  conv={mf.converged}',
                  flush=True)
    if 'rijonx' in E and 'rijonx/whole-dm' in E:
        _last = a.scf_reps - 1
        r = (R.get(f'scf rijonx/whole-dm run{_last}'),
             R.get(f'scf rijonx run{_last}'))
        if all(isinstance(x, float) for x in r):
            print(f'\nthe driver half of fastdfj: {r[0]:.3f} s -> {r[1]:.3f} s '
                  f'= {r[0]/r[1]:.2f}x, same energy to '
                  f'{abs(E["rijonx"][0]-E["rijonx/whole-dm"][0]):.1e} Eh',
                  flush=True)
            R['ratio incremental'] = r[0] / r[1]
    if 'direct' in E:
        print('\nand what the fitting costs in the energy:', flush=True)
        for kind in ('rijonx', 'fulldf'):
            if kind in E:
                print(f'  E({kind}) - E(direct) = '
                      f'{E[kind][0]-E["direct"][0]:+.3e} Eh', flush=True)
    R['E'] = E

print('\nRESULT ' + json.dumps(dict(
    tag=tag, gpu4pyscf=gpu4pyscf.__version__, gpu=GPU, xyz=a.xyz,
    basis=a.basis,
    auxbasis=a.auxbasis, xc=a.xc, patch=a.patch, natm=mol.natm,
    nao=int(mol.nao), naux=int(ref.with_df.auxmol.nao), spin=int(mol.spin),
    charge=int(mol.charge), cycles=a.cycles, mem=MEM, t=R)), flush=True)
