"""Where the ``(naux, nocc, nocc)`` array belongs, and what slicing it costs.

``docs/DF_WORK_LIST.md`` **W10** asks for two numbers rather than one.  The
item is a capacity fix -- at 284 atoms the density-fitted gradient does not run
at all because that array is 32.0 GB and upstream holds two of them -- but it
comes with a *falsification*: "the extra passes cost more than the slicing
saves at 113 atoms, i.e. the batched version is slower where the unbatched one
fits.  Guard it with a size threshold ... and measure the threshold rather than
guessing it."

So this sweeps the two knobs that decide it, at fixed density, round robin in
one process (``docs/DF_WORK_LIST.md`` §5 item 2), on a system small enough that
every cell runs:

  * where the array lives -- a device array, or page-locked host memory
    streamed a row slab at a time under the ip1 kernel;
  * how many slices of the occupied index the metric pass takes, which is what
    decides whether the array has to exist twice.

and, on the host rows, the two things that are only there because the array is
on the host: the double-buffered prefetch, and how the buffer was page-locked.
Every cell's answer is checked against the first, because a store that reads a
slab before the pass that wrote it has finished is *wrong*, not slow, and it is
wrong by an amount that looks like a tolerance.

    python benchmarks/df_grad_oo_sweep.py \
        --xyz benchmarks/molecules/taxol.xyz --basis def2-svp --xc b3lyp
"""
import json
import time

import numpy as np
import cupy as cp

from _common import build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--slices', default='1,2,4,8',
                help='how many slices of the occupied index the metric pass '
                     'takes, as a comma separated list')
ap.add_argument('--stores', default='device,host')
ap.add_argument('--ballast', default='',
                help='comma separated GB of device memory to hold resident '
                     'during each timed call, allocated outside CuPy\'s pool '
                     'and outside the timed region.  This is how the '
                     'threshold is measured rather than guessed: the device '
                     'store is faster whenever it fits, so what has to be '
                     'found is the point where holding it stops leaving the '
                     'ip1 loop room, and a 113-atom system on a 40 GB card '
                     'does not reach that point on its own')
ap.add_argument('--extras', action='store_true',
                help='also time the host store without its prefetch, without '
                     'page-locking, and page-locked with cudaHostAlloc')
a = ap.parse_args()

import electrolites                                           # noqa: E402
electrolites.patch('fastdfgrad')

import gpu4pyscf                                              # noqa: E402
from gpu4pyscf import dft                                     # noqa: E402
from gpu4pyscf.lib.cupy_helper import get_avail_mem           # noqa: E402
from electrolites import fastdfgrad                           # noqa: E402

_sync = cp.cuda.Stream.null.synchronize

mol = build_mol(a)
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc).density_fit(
    auxbasis=a.auxbasis)
mf.conv_tol = a.conv_tol
mf.grids.level = a.grids
mf.verbose = 0
if a.cycles:
    mf.max_cycle = a.cycles
mf.kernel()
dm = mf.make_rdm1()
omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
g = mf.nuc_grad_method()
g.verbose = 0

nocc = int((cp.asarray(mf.mo_occ) > 0).sum()) if mf.mo_occ.ndim == 1 else \
    int((cp.asarray(mf.mo_occ) > 0).any(axis=0).sum())
naux_cart = mf.with_df.intopt.auxmol.nao
nset = 2 if mol.spin else 1
oo_gb = nset * naux_cart * nocc**2 * 8e-9
mem_free = get_avail_mem(exclude_memory_pool=True)
tag = f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.auxbasis or "auto"}/{a.xc}'
print(f'[{tag}] natm={mol.natm} nao={mol.nao} nocc={nocc} '
      f'naux(cart)={naux_cart} nset={nset}\n'
      f'  the array W10 is about: {oo_gb:.3f} GB, and upstream holds two of '
      f'them ({2 * oo_gb:.3f} GB)\n'
      f'  free device memory now: {mem_free * 1e-9:.3f} GB  '
      f'-> oo/free = {oo_gb * 1e9 / mem_free:.3f}  '
      f'(FASTDFGRAD_OOFRAC is {fastdfgrad._OOFRAC})\n'
      f'  gpu4pyscf={gpu4pyscf.__version__}', flush=True)

SL = [x.strip() for x in a.slices.split(',')]
CASES = []
for store in a.stores.split(','):
    for s in SL:
        if s == 'auto':
            CASES.append((f'{store}, auto slices',
                          dict(_OOSTORE=store, _OOSLICE=0)))
            continue
        n = int(s)
        CASES.append((f'{store}, {n} slice' + ('s' if n > 1 else ''),
                      dict(_OOSTORE=store, _OOSLICE=-(-nocc // n))))
if a.extras and 'host' in a.stores:
    CASES += [
        ('host, no prefetch', dict(_OOSTORE='host', _OOSLICE=0,
                                   _OOPREFETCH=False)),
        ('host, unpinned', dict(_OOSTORE='host', _OOSLICE=0, _OOPIN='none')),
        ('host, cudaHostAlloc', dict(_OOSTORE='host', _OOSLICE=0,
                                     _OOPIN='alloc')),
    ]

DEFAULTS = dict(_OOSTORE='auto', _OOSLICE=0, _OOPREFETCH=True,
                _OOPIN='register')
R = {name: [] for name, _ in CASES}
ANS = {}


def apply(flags):
    for k, v in {**DEFAULTS, **flags}.items():
        setattr(fastdfgrad, k, v)


def timed(ballast_gb=0.):
    if mf.with_df._cderi is None:
        # untimed: the SCF pays for this tensor, and it has to be rebuilt
        # *before* the ballast goes on or `_cholesky_eri` would put it on the
        # host and this would be measuring something else entirely
        mf.with_df.build()
    ptr = 0
    if ballast_gb:
        # outside CuPy's pool on purpose: a pool allocation that is freed is
        # still not free as far as `get_avail_mem(exclude_memory_pool=True)`
        # is concerned, so the next row would inherit this one's ballast
        ptr = cp.cuda.runtime.malloc(int(ballast_gb * 1e9))
    try:
        _sync()
        t = time.perf_counter()
        out = g.jk_energy_per_atom(dm, j_factor=1, k_factor=hyb or 1., hermi=1)
        _sync()
        dt = time.perf_counter() - t
    finally:
        if ptr:
            cp.cuda.runtime.free(ptr)
    return dt, out


BALLAST = [float(x) for x in a.ballast.split(',')] if a.ballast else [0.]
ref = None
OUT = []
for gb in BALLAST:
    for name, flags in CASES:              # one untimed pass: JIT and plans
        apply(flags)
        R[name] = []
        try:
            timed(gb)
        except Exception as exc:                               # noqa: BLE001
            pass
    for rep in range(a.reps):
        for name, flags in CASES:
            apply(flags)
            try:
                dt, out = timed(gb)
            except Exception as exc:                           # noqa: BLE001
                # running out of memory *is* the answer at the top of the
                # ballast ladder, and it is the answer the threshold is for
                R[name] = f'{type(exc).__name__}'
                continue
            R[name].append(dt)
            ANS[name] = out
            if ref is None:
                ref = out
    apply({})
    free_here = mem_free - gb * 1e9
    print(f'\n--- ballast {gb:.1f} GB: {free_here * 1e-9:6.2f} GB free, '
          f'oo/free = {oo_gb * 1e9 / max(free_here, 1.):.3f} ---\n'
          f'{"case":24s} {"jk_energy_per_atom":>20s} {"":>9s} {"vs first":>12s}',
          flush=True)
    base = min(R[CASES[0][0]]) if isinstance(R[CASES[0][0]], list) \
        and R[CASES[0][0]] else None
    row = {}
    for name, _ in CASES:
        v = R[name]
        if not isinstance(v, list) or not v:
            print(f'{name:24s} {str(v):>20s}', flush=True)
            row[name] = str(v)
            continue
        t = min(v)
        row[name] = t
        print(f'{name:24s} {t * 1e3:17.1f} ms '
              f'{(base / t if base else float("nan")):8.3f}x '
              f'{float(np.abs(ANS[name] - ref).max()):11.3e} Eh/Bohr',
              flush=True)
    OUT.append(dict(ballast_gb=gb, free_gb=free_here * 1e-9, t=row))

print(f'\nstats: {fastdfgrad.stats()}', flush=True)
print('RESULT ' + json.dumps(dict(
    tag=tag, xyz=a.xyz, basis=a.basis, xc=a.xc, natm=mol.natm,
    nao=int(mol.nao), nocc=nocc, naux_cart=int(naux_cart), nset=nset,
    oo_gb=oo_gb, mem_free_gb=mem_free * 1e-9, ladder=OUT,
    err={k: float(np.abs(ANS[k] - ref).max()) for k in ANS},
    stats=fastdfgrad.stats())), flush=True)
