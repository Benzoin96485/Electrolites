"""A/B the density-fitted J and K build: GPU4PySCF's against `fastdf`'s.

Both implementations run in **one process** on **one cderi tensor**, at a
fixed density, so the ratio is the kernel's and not the SCF's.  `fastdf` keeps
a reference to the function it replaced, which is what makes that possible --
see `docs/README_fastdf.md` §3 for why the end-to-end numbers in
`df_anatomy.py` are smaller than these and which of the two answers a given
question.

Each row is also broken into the module's items, by toggling its own ablation
switches -- and, with ``--cpu-cderi``, into how much of the tensor is held on
the device:

  * ``step2=gemm`` / ``step2=syrk`` -- the exchange contraction forced to the
    general GEMM (GPU4PySCF's) or to the symmetric rank-k update, rather than
    chosen by size
  * ``step1=bgemm`` -- the first contraction as a cuBLAS batch over the AO
    index instead of GPU4PySCF's cuTENSOR contraction
  * ``j=tensor`` -- the Coulomb pass as GPU4PySCF's two `contract` calls
    instead of an explicit GEMV/GEMM pair
  * ``no-resid`` -- the whole tensor streamed from the host, which is what
    GPU4PySCF does once the cderi no longer fits

so a row's gain can be attributed rather than inferred.  ``--resident-fracs``
sweeps the resident share of the *tensor* at a fixed size, which is the payoff
curve for the last of those.

    python benchmarks/df_jk_bench.py --xyz benchmarks/molecules/taxol.xyz \
        --basis def2-svp --cycles 6 --ndm 1,2,4,8
"""
import json
import time

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--ndm', default='1,2,4',
                help='density counts to time; 1 is the ground state, the rest '
                     'are what a CPKS or Davidson solve asks for')
ap.add_argument('--blksizes', default='',
                help='also sweep these auxiliary block sizes')
ap.add_argument('--resident-fracs', default='',
                help='with --cpu-cderi, sweep the share of the *tensor* that '
                     'is held on the device (0 = stream all of it, which is '
                     'what GPU4PySCF does; 1 = hold all of it).  The point '
                     'of fixing the fraction rather than the memory budget '
                     'is that the payoff curve is then measured on one '
                     'molecule instead of needing a different one per point.')
ap.add_argument('--tiled', default='',
                help='also time the block-sparse first contraction '
                     '(`DF_WORK_LIST.md` W5) at these AO tile sizes, e.g. '
                     '32,64.  Each is timed in the same round robin as the '
                     'rest, and `--tile-orders` chooses the AO orderings.')
ap.add_argument('--tile-orders', default='native,rcm')
ap.add_argument('--tile-unpacks', default='kernel',
                help='which scatter into the row-compressed buffer to time '
                     'for each --tiled row: `kernel` (W14), `cupy` (what it '
                     'replaced), or both.  Both run on the *same* pattern -- '
                     'rebuilding a several-gigabyte pattern between two '
                     'variants of a round robin would time the rebuild')
ap.add_argument('--cpu-cderi', action='store_true',
                help='force the cderi onto host memory, which is what '
                     'GPU4PySCF does once it no longer fits -- the regime '
                     'its README calls "bounded by CPU memory".  Adds a '
                     'no-resident column: the whole tensor streamed, as '
                     'GPU4PySCF streams it')
a = ap.parse_args()
apply_patch(a.patch)

import gpu4pyscf                                              # noqa: E402
from gpu4pyscf import dft                                     # noqa: E402
from gpu4pyscf.lib.cupy_helper import tag_array                # noqa: E402
from electrolites import fastdf                               # noqa: E402
from electrolites import _dfblock                             # noqa: E402

_sync = cp.cuda.Stream.null.synchronize
R = {}


def _once(fn):
    _sync()
    t = time.perf_counter()
    fn()
    _sync()
    return time.perf_counter() - t


def interleave(variants, reps=None):
    """Time several variants **round robin**, and keep each one's minimum.

    The node is shared, so a competing job's load drifts over the minutes a
    row takes.  Timing all of A and then all of B lets that drift land
    entirely on one of them -- on this node the same call measured 42 ms and
    26 ms an hour apart.  Round robin puts both variants in the same slice of
    whatever else is running, which is the same argument
    `docs/RESPONSE_1.8.1.md` §4b makes for measuring a stock/patched pair
    back to back rather than in two jobs.
    """
    reps = a.reps if reps is None else reps
    out = {}
    for label, fn in variants:
        try:
            fn()
            _sync()
            out[label] = []
        except Exception as exc:                              # noqa: BLE001
            R[label] = f'ERR {type(exc).__name__}: {exc}'
            print(f'{label:38s} ERR {type(exc).__name__}: {str(exc)[:90]}',
                  flush=True)
    for _ in range(reps):
        for label, fn in variants:
            if label in out:
                out[label].append(_once(fn))
    for label, ts in out.items():
        R[label] = min(ts)
    return out


mol = build_mol(a)
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc).density_fit(
    auxbasis=a.auxbasis)
if a.cpu_cderi:
    mf.with_df.use_gpu_memory = False
mf.conv_tol = a.conv_tol
mf.grids.level = a.grids
mf.verbose = 0
mf.max_cycle = max(1, a.cycles)
mf.kernel()

dfobj = mf.with_df
if dfobj._cderi is None:
    dfobj.build()
dm = mf.make_rdm1()
nao = dfobj.nao
cderi_sparse = dfobj._cderi[cp.cuda.Device().id]
on_gpu = isinstance(cderi_sparse, cp.ndarray)

tag = (f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.auxbasis or "auto"}/{a.xc}')
print(f'[{tag}] natm={mol.natm} nao={nao} naux={dfobj.naux} '
      f'npair={cderi_sparse.shape[1]} spin={mol.spin} '
      f'cderi={cderi_sparse.nbytes/1e9:.2f} GB on_gpu={on_gpu} '
      f'gpu4pyscf={gpu4pyscf.__version__}', flush=True)


def _densities(n_dm):
    """``n_dm`` ``mo_coeff``-tagged densities, the shape every solve produces."""
    mo = cp.asarray(mf.mo_coeff)
    occ = cp.asarray(mf.mo_occ)
    if mo.ndim == 3:            # unrestricted: one pair, not a stack
        return tag_array(cp.asarray(dm), mo_coeff=mo, mo_occ=occ)
    if n_dm == 1:
        return tag_array(cp.asarray(dm), mo_coeff=mo, mo_occ=occ)
    mos = cp.repeat(mo[None], n_dm, axis=0)
    occs = cp.repeat(occ[None], n_dm, axis=0)
    dms = cp.repeat(cp.asarray(dm)[None], n_dm, axis=0)
    return tag_array(dms, mo_coeff=mos, mo_occ=occs)


NDMS = [int(x) for x in a.ndm.split(',')]
print(f'{"":24s} {"stock":>9s} {"fastdf":>9s} {"ratio":>7s} '
      f'{"s2=gemm":>9s} {"s2=syrk":>9s} {"s1=bgemm":>9s} {"s1=dense":>9s} '
      f'{"j=tensor":>9s} {"no-resid":>9s}')


def _variant(dms, **flags):
    def run():
        # `_RESIDENT_FRAC = 0` makes `_resident_prefix` return before it
        # looks at its cache, so the resident block stays allocated and the
        # *next* variant does not pay to rebuild it.  Dropping the cache here
        # instead made every `fastdf` repetition carry a host-to-device copy
        # of the whole tensor and read 0.88x where it is 3.5x.
        old = {k: getattr(fastdf, k) for k in flags}
        for k, v in flags.items():
            setattr(fastdf, k, v)
        try:
            return fastdf.get_jk(dfobj, dms, 1, True, True)
        finally:
            for k, v in old.items():
                setattr(fastdf, k, v)
    return run


for n_dm in NDMS:
    dms = _densities(n_dm)
    variants = [
        (f'n_dm={n_dm} stock',
         lambda d=dms: fastdf._orig_get_jk(dfobj, d, 1, True, True)),
        (f'n_dm={n_dm} fastdf', _variant(dms)),
        (f'n_dm={n_dm} gemm', _variant(dms, _STEP2='gemm')),
        (f'n_dm={n_dm} syrk', _variant(dms, _STEP2='syrk')),
        (f'n_dm={n_dm} bgemm', _variant(dms, _STEP1='bgemm')),
        (f'n_dm={n_dm} s1tensor', _variant(dms, _STEP1='tensor')),
        (f'n_dm={n_dm} jtensor', _variant(dms, _JPASS='tensor')),
    ]
    if not on_gpu:
        variants.append((f'n_dm={n_dm} no-resident',
                         _variant(dms, _RESIDENT_FRAC=0.)))
    tiled = [(t, o) for t in (a.tiled.split(',') if a.tiled else [])
             for o in a.tile_orders.split(',')]
    interleave(variants)
    rj, rk = fastdf._orig_get_jk(dfobj, dms, 1, True, True)
    gj, gk = fastdf.get_jk(dfobj, dms, 1, True, True)
    scale = float(cp.abs(rk).max()) or 1.
    rel = float(cp.abs(gk - rk).max()) / scale

    def _ms(key):
        v = R.get(f'n_dm={n_dm} {key}')
        return f'{v*1e3:9.2f}' if isinstance(v, float) else f'{"-":>9s}'
    t0, t1 = R[f'n_dm={n_dm} stock'], R[f'n_dm={n_dm} fastdf']
    ratio = t0 / t1
    extra = ''
    if not on_gpu:
        extra = (f'  resident={fastdf.STATS["resident_rows"]}/'
                 f'{fastdf.STATS["resident_rows"] + fastdf.STATS["streamed_rows"]}')
    print(f'{f"get_jk  n_dm={n_dm}":24s} {t0*1e3:9.2f} {t1*1e3:9.2f} '
          f'{ratio:6.2f}x {_ms("gemm")} {_ms("syrk")} '
          f'{_ms("bgemm")} {_ms("s1tensor")} {_ms("jtensor")} '
          f'{_ms("no-resident")}  '
          f'relK={rel:.1e}{extra}', flush=True)
    # Each (tile, order) is timed in a round robin of its **own**, paired
    # against the default first contraction.  Only one tile pattern fits on
    # the object at a time -- it is several GB at 284 atoms -- so putting them
    # all in one round robin would make every timed call rebuild the pattern
    # the previous variant evicted, and time the rebuild.
    for t, o in tiled:
        # the reference is the *dense* first contraction, forced: `fastdf`'s
        # own default is `auto`, which takes the tiled one wherever the fill
        # test passes, and would otherwise be compared against itself
        def _tiled(u, t=t, o=o):
            run = _variant(dms, _STEP1='tiled', TILE=int(t), TILE_ORDER=o)

            def go():
                old_u, _dfblock.UNPACK = _dfblock.UNPACK, u
                try:
                    return run()
                finally:
                    _dfblock.UNPACK = old_u
            return go
        pair = ([(f'n_dm={n_dm} ref{t}/{o}', _variant(dms, _STEP1='tensor'))]
                + [(f'n_dm={n_dm} tiled{t}/{o}/{u}', _tiled(u))
                   for u in a.tile_unpacks.split(',')])
        interleave(pair)
        ref = R[f'n_dm={n_dm} ref{t}/{o}']
        for u in a.tile_unpacks.split(','):
            gj, gk = _tiled(u)()
            rel_t = float(cp.abs(gk - rk).max()) / scale
            v = R[f'n_dm={n_dm} tiled{t}/{o}/{u}']
            print(f'{"  tiled " + t + "/" + o + "/" + u:24s} {ref*1e3:9.2f} '
                  f'{v*1e3:9.2f} '
                  f'{ref/v:6.2f}x   fill={fastdf.STATS["tile_fill"]:.3f}  '
                  f'blk={fastdf.STATS["blocks"]}  relK={rel_t:.1e}',
                  flush=True)
        if hasattr(dfobj, '_fastdf_tiles'):
            del dfobj._fastdf_tiles
        cp.get_default_memory_pool().free_all_blocks()

if a.resident_fracs and not on_gpu:
    print()
    fracs = [float(x) for x in a.resident_fracs.split(',')]
    dms = _densities(NDMS[0])
    rows_for = {}
    naux = int(cderi_sparse.shape[0])
    for f in fracs:
        # each point needs its own prefix, so the cache is dropped once here
        # and rebuilt inside the first (untimed) call of `interleave`
        if hasattr(dfobj, '_fastdf_resident'):
            del dfobj._fastdf_resident
        fastdf._RESIDENT_FRAC = 0. if f <= 0 else 0.9
        fastdf.RESIDENT_MAX_ROWS = int(round(f * naux))
        fastdf.get_jk(dfobj, dms, 1, True, True)
        rows_for[f] = (fastdf.STATS['resident_rows'],
                       fastdf.STATS['resident_rows'] +
                       fastdf.STATS['streamed_rows'])
        interleave([(f'frac={f}', lambda d=dms: fastdf.get_jk(
            dfobj, d, 1, True, True))])
    fastdf._RESIDENT_FRAC = 0.5
    fastdf.RESIDENT_MAX_ROWS = 0
    if hasattr(dfobj, '_fastdf_resident'):
        del dfobj._fastdf_resident
    for f in fracs:
        res, tot = rows_for[f]
        print(f'{f"fastdf resident_frac={f}":30s} {R[f"frac={f}"]*1e3:9.2f} ms'
              f'   {res}/{tot} rows resident '
              f'({100.0*res/max(tot,1):5.1f} %)', flush=True)

if a.blksizes:
    print()
    dms = _densities(NDMS[0])
    blks = [int(x) for x in a.blksizes.split(',')]
    interleave([(f'blk={b}', _variant(dms, _BLK=b)) for b in blks])
    for b in blks:
        v = R[f'blk={b}']
        print(f'{f"fastdf blk={b}":26s} '
              f'{v*1e3:9.2f} ms' if isinstance(v, float) else v, flush=True)

print('RESULT ' + json.dumps(dict(
    tag=tag, gpu4pyscf=gpu4pyscf.__version__, xyz=a.xyz, basis=a.basis,
    auxbasis=a.auxbasis, xc=a.xc, natm=mol.natm, nao=int(nao),
    naux=int(dfobj.naux), npair=int(cderi_sparse.shape[1]),
    on_gpu=bool(on_gpu), spin=int(mol.spin), t=R)), flush=True)
