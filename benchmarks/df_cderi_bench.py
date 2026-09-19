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

VARIANTS = {
    'GPU4PySCF': None,
    'sp_mult':   dict(_METRIC='gemm', _SPMULT=1, _PIN='alloc',
                      _MIN_WORK=F._MIN_WORK),
    # `_MIN_WORK` is forced to zero in these two so that the contraction is
    # priced on every system rather than only on the ones the module would
    # take it on; the `all` row keeps the module's own threshold, so the last
    # column is what a run actually gets
    'trmm':      dict(_METRIC='trmm', _SPMULT=4, _PIN='alloc', _MIN_WORK=0.),
    'trsm':      dict(_METRIC='trsm', _SPMULT=4, _PIN='alloc', _MIN_WORK=0.),
    # the host buffer's page-locking, which only a host-resident build pays
    'pin':       dict(_METRIC='gemm', _SPMULT=4, _PIN='register',
                      _MIN_WORK=F._MIN_WORK),
    'all':       dict(_METRIC='trmm', _SPMULT=1, _PIN='register',
                      _MIN_WORK=F._MIN_WORK),
}
if not a.cpu_cderi:
    del VARIANTS['pin']


def call(cfg):
    if cfg is None:
        out = _REF(intopt, **_kw)
    else:
        for k, v in cfg.items():
            setattr(F, k, v)
        out = F._cholesky_eri(intopt, **_kw)
    _sync()
    del out
    cp.get_default_memory_pool().free_all_blocks()


for cfg in VARIANTS.values():                    # warm every path once
    call(cfg)

best = {k: float('inf') for k in VARIANTS}
for _ in range(max(2, a.reps)):
    for name, cfg in VARIANTS.items():
        _sync()
        t = time.perf_counter()
        call(cfg)
        best[name] = min(best[name], time.perf_counter() - t)

base = best['GPU4PySCF']
print(f'\n{"variant":12s} {"_cholesky_eri":>14s}   speedup', flush=True)
for name in VARIANTS:
    print(f'{name:12s} {best[name]*1e3:11.1f} ms   '
          f'{base/best[name]:6.3f}x', flush=True)

R = dict(xyz=a.xyz, basis=a.basis, natm=mol.natm, nao=int(mol.nao),
         naux=int(auxmol.nao), cderi=_mem, cholesky_eri=best)

if a.scf:
    # Twice, and the second round is the one quoted: the first SCF of a
    # process pays every one-time cost there is -- NVRTC, the grid, the
    # cuBLAS workspaces -- and a stock-then-patched ordering would read all
    # of that as a speedup.  `docs/DF_WORK_LIST.md` 5.3.
    print('\n--- the whole SCF the build sits in ---', flush=True)
    R['scf'] = {}
    for rnd in (1, 2):
        for name, cfg in (('GPU4PySCF', None), ('all', VARIANTS['all'])):
            if cfg is None:
                DFMOD._cholesky_eri = _REF
            else:
                DFMOD._cholesky_eri = F._cholesky_eri
                for k, v in cfg.items():
                    setattr(F, k, v)
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

print('RESULT ' + json.dumps(R), flush=True)
