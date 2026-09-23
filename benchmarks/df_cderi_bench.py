"""The cderi build, patched against unpatched, item by item.

``docs/DF_WORK_LIST.md`` **W4**.  ``benchmarks/df_cderi_probe.py`` says where
the time in ``df/df.py:_cholesky_eri`` is; this prices what
``electrolites/fastdfcderi.py`` does about it, on the real function rather
than on a clone.

Four columns, timed **round robin** rather than one after the other, because
a competing job's load drifts over the minutes a row takes and a sequential
schedule puts all of that drift on one variant
(``docs/DF_WORK_LIST.md`` 5.2):

    GPU4PySCF   the unpatched function, saved before the module is imported
    sp_mult     the aggregation alone, upstream's dense metric contraction
    trmm        the sparse metric contraction alone, upstream's aggregation
    trsm        the same with the triangular *solve* W4 asks for instead of
                the multiply -- the rejected variant, kept so it can be re-run
    pin         the host cderi's buffer page-locked with `cudaHostRegister`
                rather than allocated with `cudaHostAlloc` (host runs only)
    pool        the same buffer *kept* for the next build, which is W12 --
                so this column is the second and later builds of a process,
                which is what a geometry optimisation and a range-separated
                functional pay (host runs only)
    store       the block written to the host on a second stream, which is
                W11 (host runs only)
    all         the module's default

    python benchmarks/df_cderi_bench.py --xyz molecules/taxol.xyz \\
        --basis def2-svp [--cpu-cderi] [--scf]
"""
import json
import time

import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--cpu-cderi', action='store_true',
                help='force the host-resident branch of the build')
ap.add_argument('--scf', action='store_true',
                help='also run a whole density-fitted SCF, so the build can '
                     'be read as a share of the run it sits in')
ap.add_argument('--variants', default='',
                help='comma separated subset of the column names below, for '
                     'a system where the round robin over all of them does '
                     'not fit in a debug allocation.  `GPU4PySCF` is always '
                     'kept')
ap.add_argument('--steps', type=int, default=0,
                help='W12: build the whole `DF` object this many times over, '
                     'on geometries displaced by a thousandth of an Angstrom '
                     '-- which is the shape of a geometry optimisation, and '
                     'the only caller that pays the page-locking more than '
                     'once.  A range-separated functional does *not*: '
                     '`DF.range_coulomb` keeps its second object in '
                     '`_rsh_df`, so the two tensors are alive at the same '
                     'time and neither can have the other\'s buffer.')
a = ap.parse_args()
apply_patch(a.patch)

import gpu4pyscf                                             # noqa: E402
from pyscf.df import addons                                  # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.df import df as DFMOD                         # noqa: E402
from gpu4pyscf.df import int3c2e_bdiv as bdiv                # noqa: E402
from gpu4pyscf.gto.mole import SortedMole                    # noqa: E402

_REF = DFMOD._cholesky_eri                    # before anything is patched
import electrolites                                          # noqa: E402
electrolites.patch('fastdfcderi')
from electrolites import fastdfcderi as F                    # noqa: E402
from electrolites import _dfj2c as J2C                       # noqa: E402
from electrolites import _dfpin as PIN                       # noqa: E402
from electrolites import _dfstore as STORE                   # noqa: E402

#: the switches do not all live on one module any more -- the page-locking is
#: `_dfpin`'s and the store is `_dfstore`'s, because a probe has to be able to
#: import either without patching the build it is measuring
_MODULES = {'F': F, 'J2C': J2C, 'PIN': PIN, 'STORE': STORE}

_sync = cp.cuda.Stream.null.synchronize
_mem = 'gpu' if not a.cpu_cderi else 'host'
_kw = dict(use_gpu_memory=False if a.cpu_cderi else None)

mol = build_mol(a)
auxmol = addons.make_auxmol(mol, a.auxbasis)
intopt = bdiv.Int3c2eOpt(mol, auxmol)
intopt.mol = SortedMole.from_mol(mol, decontract=True)
intopt.build()
print(f'[{a.xyz.rsplit("/", 1)[-1]}/{a.basis}] natm={mol.natm} '
      f'nao={mol.nao} naux={auxmol.nao} cderi={_mem} '
      f'gpu4pyscf={gpu4pyscf.__version__}', flush=True)

#: what every column but the one being priced is pinned to: GPU4PySCF's own
#: choice on each of the four axes this module touches.
_STOCK = {'F._METRIC': 'gemm', 'F._SPMULT': 4, 'F._MIN_WORK': F._MIN_WORK,
          'J2C._J2C': 'stock',
          'PIN._PIN': 'alloc', 'PIN._POOL': False, 'STORE._STORE': 'host'}


def _cfg(**over):
    out = dict(_STOCK)
    out.update({k.replace('__', '.'): v for k, v in over.items()})
    return out


VARIANTS = {
    'GPU4PySCF': None,
    'sp_mult':   _cfg(F___SPMULT=1),
    # W13: the two-centre metric integrals' own optimiser -- built the cheap
    # way (`build`) and then kept (`cache`).  Only the first of the two is a
    # gain on a *first* build; the second is what a repeat at the same
    # geometry gets, which is what `min` over the repeats reports
    'j2c/build': _cfg(J2C___J2C='build'),
    'j2c':       _cfg(J2C___J2C='cache'),
    # `_MIN_WORK` is forced to zero in these two so that the contraction is
    # priced on every system rather than only on the ones the module would
    # take it on; the `all` row keeps the module's own threshold, so the last
    # column is what a run actually gets
    'trmm':      _cfg(F___METRIC='trmm', F___MIN_WORK=0.),
    'trsm':      _cfg(F___METRIC='trsm', F___MIN_WORK=0.),
    # the host buffer's page-locking, which only a host-resident build pays
    'pin':       _cfg(PIN___PIN='register'),
    # W12: the same page-locking, kept.  Every repeat but the first reuses
    # the buffer, and `min` over the repeats is therefore what the *second*
    # and later builds of a process pay -- which is the thing W12 is about
    'pool':      _cfg(PIN___PIN='register', PIN___POOL=True),
    # W11: the store on its own stream, against upstream's synchronous one.
    # Two of them, because the two differ in who does the PCIe write -- a
    # copy engine for `stage`, the SMs for `direct`, which is upstream's own
    # kernel moved to another stream
    'store':     _cfg(STORE___STORE='stage'),
    'store/dir': _cfg(STORE___STORE='direct'),
    'all':       _cfg(F___METRIC='trmm', F___SPMULT=1, J2C___J2C='cache',
                      PIN___PIN='register', PIN___POOL=True,
                      STORE___STORE='stage'),
    # the two new items priced at the *top* of the stack rather than the
    # bottom.  `store` and `j2c` above are each measured against GPU4PySCF
    # on every other axis, and on a host-resident build that means measured
    # against a `cudaHostAlloc` that is half the function: the store can be
    # worth 16 % of the block loop and still read 1.00x there.  These two
    # columns are `all` with one item taken back out, so `all` divided by
    # them is what the item is worth to a run that has the rest
    'all -store': _cfg(F___METRIC='trmm', F___SPMULT=1, J2C___J2C='cache',
                       PIN___PIN='register', PIN___POOL=True),
    'all -j2c':  _cfg(F___METRIC='trmm', F___SPMULT=1,
                      PIN___PIN='register', PIN___POOL=True,
                      STORE___STORE='stage'),
    # `all` a second time, in the last position.  The two ablation columns
    # above sit at the end of the round-robin, and a difference that is
    # really about *where* a column runs rather than what it switches would
    # show up here and nowhere else -- which is the only way to tell the two
    # apart without a profiler
    'all again': _cfg(F___METRIC='trmm', F___SPMULT=1, J2C___J2C='cache',
                      PIN___PIN='register', PIN___POOL=True,
                      STORE___STORE='stage'),
    # the same configuration with the pool emptied before every timed call,
    # which is what a *first* build pays.  The pooled columns above are
    # repeats -- `min` over the repeats is a cache hit by construction -- and
    # a single SCF builds the tensor once, so this is the column that
    # describes it.  `--steps` is the same distinction drawn the other way,
    # as a sequence rather than as two columns
    'all/cold':  _cfg(F___METRIC='trmm', F___SPMULT=1, J2C___J2C='cache',
                      PIN___PIN='register', PIN___POOL=True,
                      STORE___STORE='stage'),
}
if not a.cpu_cderi:
    for k in ('pin', 'pool', 'store', 'store/dir', 'all -store', 'all/cold'):
        del VARIANTS[k]
if a.variants:
    want = {'GPU4PySCF'} | set(a.variants.split(','))
    unknown = want - set(VARIANTS)
    if unknown:
        raise SystemExit(f'unknown variant(s) {sorted(unknown)}; '
                         f'known: {list(VARIANTS)}')
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in want}


def _apply(cfg):
    """Put the switches where this column wants them.

    The reference column gets ``_STOCK`` rather than nothing.  It has to:
    W13 replaces ``int3c2e_bdiv.int2c2e`` at the *module* level, so the
    unpatched ``_cholesky_eri`` saved above would otherwise call this
    repository's two-centre integrals and the two columns would agree to
    three digits for the reason `docs/DF_WORK_LIST.md` 5.13 gives.
    """
    for k, v in cfg.items():
        mod, attr = k.split('.')
        setattr(_MODULES[mod], attr, v)


def call(cfg, name=''):
    """One column's `_cholesky_eri`.

    Nothing is freed between columns, and that is deliberate.  An earlier
    draft called `PIN.free_pool()` before every column that was not pooled,
    which left the pool empty whenever a pooled column ran *first* after
    one that was not -- so `all` paid the 1.6 s page-locking every round and
    the two ablation columns after it did not, and the store read 0.55x when
    it is a gain.  `PIN._POOL` now gates the pool's *lookup* as well as its
    retention, so a column that is not pooled cannot take the buffer, and
    the pooled columns share it the way a repeated build would.
    """
    _apply(cfg if cfg is not None else _STOCK)
    if name.endswith('/cold'):
        PIN.free_pool()
    out = _REF(intopt, **_kw) if cfg is None else F._cholesky_eri(intopt, **_kw)
    _sync()
    del out
    cp.get_default_memory_pool().free_all_blocks()


for name, cfg in VARIANTS.items():               # warm every path once
    call(cfg, name)

R = {}
best = {k: float('inf') for k in VARIANTS}
# the block loop on its own, which the module times for itself.  The round
# robin drifts across the columns of one round on the host branch -- an
# `all` column repeated in the last position read 2238 ms against 1499 --
# so the store, which is a few hundred milliseconds of a 1.5 s function,
# cannot be read off a difference between two adjacent columns.  This one is
# a loop measured against itself.
loop = {k: float('inf') for k in VARIANTS}
for _ in range(max(2, a.reps)):
    for name, cfg in VARIANTS.items():
        _sync()
        before = F.STATS['loop_s']
        t = time.perf_counter()
        call(cfg, name)
        best[name] = min(best[name], time.perf_counter() - t)
        d = F.STATS['loop_s'] - before
        if d > 0:
            loop[name] = min(loop[name], d)

base = best['GPU4PySCF']
print(f'\n{"variant":12s} {"_cholesky_eri":>14s}   speedup', flush=True)
for name in VARIANTS:
    print(f'{name:12s} {best[name]*1e3:11.1f} ms   '
          f'{base/best[name]:6.3f}x', flush=True)
finite = {k: v for k, v in loop.items() if v < float('inf')}
if finite:
    # `GPU4PySCF` runs the unpatched function, which does not keep this
    # counter, so the baseline for the loop is the column that keeps
    # upstream's *store* while everything else is this module's
    base_l = finite.get('all -store', next(iter(finite.values())))
    print(f'\n{"variant":12s} {"block loop":>14s}   vs `all -store`   '
          f'(the store is in this; the page-locking is not)', flush=True)
    for name, v in finite.items():
        print(f'{name:12s} {v*1e3:11.1f} ms   {base_l/v:6.3f}x', flush=True)
    R['loop_ms'] = {k: v * 1e3 for k, v in finite.items()}
for item, without in (('the store (W11)', 'all -store'),
                      ('int2c2e   (W13)', 'all -j2c')):
    if without in best:
        line = (f'  {item} is worth {best[without]/best["all"]:6.3f}x of the '
                f'`all` column')
        if without in finite and 'all' in finite:
            line += f', {finite[without]/finite["all"]:6.3f}x of the block loop'
        print(line, flush=True)

# W13 priced on the function rather than inferred from the difference of two
# columns.  The `all -j2c` column is the whole build with the two-centre
# integrals put back the way GPU4PySCF does them, and a column difference
# carries whatever else moved with it -- several hundred small device
# allocations per call, among other things -- so the stage itself is timed
# here as well and the two are printed side by side.  If they disagree, the
# difference is not `_decompose_j2c` and the write-up has to say so.
print('\n--- W13: `_decompose_j2c` on its own ---', flush=True)
R_j2c = {}
for mode in ('stock', 'build', 'cache'):
    J2C._J2C = mode
    if mode != 'cache':
        J2C.free_cache()
    DFMOD._decompose_j2c(intopt.auxmol, None)
    _sync()
    ts = []
    for _ in range(max(5, a.reps)):
        _sync()
        t = time.perf_counter()
        DFMOD._decompose_j2c(intopt.auxmol, None)
        _sync()
        ts.append(time.perf_counter() - t)
    R_j2c[mode] = min(ts)
    print(f'  {mode:8s} {min(ts)*1e3:9.2f} ms', flush=True)
print(f'  stock/cache = {R_j2c["stock"]/R_j2c["cache"]:.2f}x, '
      f'worth {(R_j2c["stock"] - R_j2c["cache"])*1e3:.1f} ms of the build',
      flush=True)
J2C._J2C = 'cache'

R.update(xyz=a.xyz, basis=a.basis, natm=mol.natm, nao=int(mol.nao),
         naux=int(auxmol.nao), cderi=_mem, cholesky_eri=best,
         pin=PIN.stats(), store=STORE.stats(), j2c=J2C.stats(),
         decompose_j2c=R_j2c)

if a.scf:
    # Twice, and the second round is the one quoted: the first SCF of a
    # process pays every one-time cost there is -- NVRTC, the grid, the
    # cuBLAS workspaces -- and a stock-then-patched ordering would read all
    # of that as a speedup.  `docs/DF_WORK_LIST.md` 5.3.
    print('\n--- the whole SCF the build sits in ---', flush=True)
    R['scf'] = {}
    for rnd in (1, 2):
        for name, cfg in (('GPU4PySCF', None), ('all', VARIANTS['all'])):
            _apply(cfg if cfg is not None else _STOCK)
            if cfg is None:
                DFMOD._cholesky_eri = _REF
                PIN.free_pool()
                J2C.free_cache()
            else:
                DFMOD._cholesky_eri = F._cholesky_eri
            mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc)
            mf = mf.density_fit(auxbasis=a.auxbasis)
            mf.conv_tol = a.conv_tol
            mf.grids.level = a.grids
            mf.verbose = 0
            if a.cpu_cderi:
                mf.with_df.use_gpu_memory = False
            _sync()
            t = time.perf_counter()
            e = mf.kernel()
            _sync()
            dt = time.perf_counter() - t
            if rnd == 2:
                R['scf'][name] = dict(t=dt, e=float(e), cycles=int(mf.cycles))
            print(f'  round {rnd}  {name:12s} {dt:8.2f} s   E = {e:.9f}  '
                  f'({mf.cycles} cycles)', flush=True)
            del mf
            cp.get_default_memory_pool().free_all_blocks()
    DFMOD._cholesky_eri = F._cholesky_eri
    s_ = R['scf']
    print(f'  whole SCF {s_["GPU4PySCF"]["t"]/s_["all"]["t"]:.3f}x, '
          f'dE = {abs(s_["GPU4PySCF"]["e"]-s_["all"]["e"]):.2e} Eh',
          flush=True)

if a.steps:
    # W12 end to end.  `DF.build()` is what a geometry optimisation calls per
    # step, and the buffer it page-locks is the same size every time.
    print(f'\n--- W12: {a.steps} whole `DF.build()` calls, as an optimisation '
          f'step loop ---', flush=True)
    from gpu4pyscf.df.df import DF                            # noqa: E402
    R['steps'] = {}
    base_coords = mol.atom_coords().copy()
    for name in ('GPU4PySCF', 'pin', 'pool', 'j2c', 'all'):
        if name not in VARIANTS:
            continue
        cfg = VARIANTS[name]
        _apply(cfg if cfg is not None else _STOCK)
        DFMOD._cholesky_eri = _REF if cfg is None else F._cholesky_eri
        # every column starts this loop cold, because what the loop is for
        # is the *shape* of the step sequence -- what the first step pays
        # and what the later ones do -- and a pool carried over from the
        # previous column would make step 1 look like step 2
        PIN.free_pool()
        J2C.free_cache()
        dfobj = DF(mol)
        dfobj.use_gpu_memory = False if a.cpu_cderi else None
        ts = []
        for step in range(a.steps):
            mol.set_geom_(base_coords * (1. + 1e-3 * step), unit='Bohr')
            dfobj.reset()
            _sync()
            t = time.perf_counter()
            dfobj.build()
            _sync()
            ts.append(time.perf_counter() - t)
        dfobj.reset()
        del dfobj
        cp.get_default_memory_pool().free_all_blocks()
        R['steps'][name] = ts
        print(f'  {name:10s} ' + '  '.join(f'{t:6.2f} s' for t in ts) +
              f'   total {sum(ts):7.2f} s', flush=True)
    mol.set_geom_(base_coords, unit='Bohr')
    DFMOD._cholesky_eri = F._cholesky_eri
    print(f'  {PIN.stats()}', flush=True)

print('RESULT ' + json.dumps(R), flush=True)
