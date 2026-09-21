"""What does a single-precision cderi buy?  W7's other half.

``docs/DF_WORK_LIST.md`` **W7**: "exactly 2x on storage and 2x on PCIe".  The
storage half is arithmetic and ``benchmarks/df_fp32_probe.py`` prints it; this
is the half that has to be measured, because halving a transfer only pays
where the transfer is exposed.  §3.3 of the work list says it is not always:
at 284 atoms the double-buffered prefetch already hides the whole 86 GB
behind the exchange contractions, so a J+K call there should gain **nothing**
and a J-only call should gain a factor of two.  That is a prediction, and
these are the four stages it is checked on.

    build       `df/df.py:_cholesky_eri`, which now writes half as many bytes
                and pays a transpose it did not pay before
    loop        `DF.loop(unpack=False)` -- the stream and nothing else, which
                is where the 2x has to appear if it appears anywhere
    jk          one whole `get_jk` J+K call at a converged density
    j           the same call with `with_k=False`, which is PCIe-bound
    scf         the whole SCF, twice, quoting the second (§5.3)

Two precisions in one process is fine while both tensors fit; past about a
hundred atoms it is not, and the second one lands on the host with nothing
said (``docs/DF_WORK_LIST.md`` §5.10).  ``--only`` runs one per process and
``--dm-cache`` makes the two processes drive the same density.  Each row
prints where its tensor actually lives, because that is the single largest
term in what follows and it is not visible in the timing.

    python benchmarks/df_fp32_bench.py --xyz molecules/taxol.xyz \\
        --basis def2-svp --dm-cache $SCRATCH/fp32dm
    python benchmarks/df_fp32_bench.py --xyz molecules/PfPMT.xyz --charge -1 \\
        --basis def2-svp --only fp32 --dm-cache $SCRATCH/fp32dm --reps 2
"""
import json
import os
import time

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--cpu-cderi', action='store_true',
                help='force the host-resident branch in both precisions, '
                     'which is the only way to compare the streamed path at '
                     'a size where the narrow tensor would otherwise fit')
ap.add_argument('--gpu-cderi', action='store_true',
                help='force the device-resident branch, so a build that '
                     'would have fallen to the host raises instead of '
                     'quietly measuring a different thing.  `_cholesky_eri` '
                     'decides from the free memory *at the moment of the '
                     'build*, and a rebuild inside a timing loop can land on '
                     'the other side of that from the one before it')
ap.add_argument('--only', default='both', choices=('both', 'fp64', 'fp32'))
ap.add_argument('--scf', action='store_true',
                help='also run whole SCFs, so the stages can be read as a '
                     'share of the run they sit in')
ap.add_argument('--resident-frac', type=float, default=-1.,
                help="`fastdf`'s item 4, which keeps as much of a "
                     'host-resident tensor on the device as free memory '
                     'allows and is what actually decides whether there is '
                     'any PCIe traffic left to halve.  0 restores '
                     "GPU4PySCF's streaming; -1 (default) leaves the "
                     "module's own 0.5")
ap.add_argument('--loop-ablate', action='store_true',
                help="`DF.loop` reads more than the factor of two the "
                     'transfer itself gives (`--pcie` says the link runs at '
                     'the same GB/s in both precisions), so the rest is in '
                     'the generator.  This times a local clone of it with '
                     'the staging dtype, the widening and the buffer '
                     'allocation each switched on and off, against '
                     "upstream's own generator on the same tensor")
ap.add_argument('--pcie', action='store_true',
                help='also time the host-to-device copy on its own, at the '
                     'block size `DF.loop` uses and at the whole tensor, in '
                     'both precisions and at matched *byte* counts.  This is '
                     'what separates "half the bytes" from "a different '
                     'transfer rate" -- see `docs/README_fastdffp32.md` §3')
ap.add_argument('--stages', default='build,loop,jk,j',
                help='which of the four stages to time.  Dropping `build` '
                     'is what makes the largest cases runnable at all: the '
                     'stage rebuilds the tensor, and at 545 atoms a second '
                     'one does not fit next to the first')
ap.add_argument('--guess-dm', action='store_true',
                help='drive the fixed-density calls with the initial guess '
                     'instead of a converged density.  A `minao` guess '
                     'screens far harder than a converged one, so a ratio '
                     'measured on it is not the one the SCF sees '
                     '(`docs/ROUND2_1.8.1.md` §2) -- this is for the sizes '
                     'where converging at all is the thing being '
                     'demonstrated')
ap.add_argument('--dm-cache', default='',
                help='converge once, save the density here, and load it in '
                     'every later process')
a = ap.parse_args()
apply_patch(a.patch or 'all')

from pyscf import lib                                        # noqa: E402
from gpu4pyscf import dft, scf as gscf                       # noqa: E402
from electrolites import fastdf                              # noqa: E402
from electrolites import _dffp32 as FP32                     # noqa: E402

if a.resident_frac >= 0:
    fastdf._RESIDENT_FRAC = a.resident_frac

_sync = cp.cuda.Stream.null.synchronize
mol = build_mol(a)
name = a.xyz.rsplit('/', 1)[-1].rsplit('.', 1)[0]


def make_mf(density_fit=True):
    mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc) if a.xc else \
        (gscf.UHF if mol.spin else gscf.RHF)(mol)
    mf.verbose = 0
    mf.conv_tol = a.conv_tol
    if a.xc:
        mf.grids.level = a.grids
    if not density_fit:
        return mf
    mf = mf.density_fit(auxbasis=a.auxbasis)
    if a.cpu_cderi:
        mf.with_df.use_gpu_memory = False
    elif a.gpu_cderi:
        mf.with_df.use_gpu_memory = True
    return mf


def converged_dm():
    if a.guess_dm:
        mf = make_mf()
        dm = cp.asarray(mf.get_init_guess())
        del mf
        cp.get_default_memory_pool().free_all_blocks()
        return dm
    cache = f'{a.dm_cache}.{name}.{a.basis}.npy' if a.dm_cache else ''
    if cache and os.path.exists(cache):
        return cp.asarray(np.load(cache))
    # a converged density, because a `minao` guess screens far harder than a
    # converged one and a fixed-density ratio measured on it is not the one
    # the SCF sees -- docs/ROUND2_1.8.1.md §2
    mf = make_mf()
    mf.kernel()
    dm = cp.asarray(mf.make_rdm1())
    if cache:
        np.save(cache, cp.asnumpy(dm))
    del mf
    cp.get_default_memory_pool().free_all_blocks()
    return dm


def timeit(fn, reps):
    fn()
    _sync()
    ts = []
    for _ in range(reps):
        _sync()
        t = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t)
    return min(ts)


dm = converged_dm()
print(f'[{name}/{a.basis}] natm={mol.natm} nao={mol.nao} '
      f'cderi={"host" if a.cpu_cderi else "auto"} patched={a.patch or "all"} '
      f'resident_frac={fastdf._RESIDENT_FRAC}', flush=True)

SIDES = ['fp64', 'fp32'] if a.only == 'both' else [a.only]
obj, info = {}, {}
for side in SIDES:
    FP32._FP32 = (side == 'fp32')
    t = time.perf_counter()
    dfobj = make_mf().with_df
    dfobj.build()
    _sync()
    FP32._FP32 = False
    obj[side] = dfobj
    c = dfobj._cderi[0]
    info[side] = dict(build=time.perf_counter() - t, gb=c.nbytes * 1e-9,
                      on_gpu=bool(isinstance(c, cp.ndarray)),
                      dtype=str(np.dtype(c.dtype)))
    if (a.pcie or a.loop_ablate) and not isinstance(c, cp.ndarray):
        info[side]['host_array'] = c
    # and let go of it here: a name bound to the tensor in this scope keeps
    # the *old* one alive while `rebuild` allocates the new one, which at
    # 284 atoms is 173 GB of pinned host memory and an OOM kill
    del c
    print(f'  {side}: cderi {info[side]["gb"]:.2f} GB on '
          f'{"the device" if info[side]["on_gpu"] else "the host"}, '
          f'first build {info[side]["build"]:.2f} s', flush=True)


def rebuild(side):
    FP32._FP32 = (side == 'fp32')
    try:
        obj[side].reset()
        obj[side].build()
    finally:
        FP32._FP32 = False


def loop_only(side):
    n = 0
    for _, blk in obj[side].loop(unpack=False):
        n += blk.shape[0]
    return n


STAGES = {
    'build': lambda s: rebuild(s),
    'loop': lambda s: loop_only(s),
    'jk': lambda s: obj[s].get_jk(dm, hermi=1, with_j=True, with_k=True),
    'j': lambda s: obj[s].get_jk(dm, hermi=1, with_j=True, with_k=False),
}
STAGES = {k: v for k, v in STAGES.items() if k in a.stages.split(',')}

# round robin, not one stage after the other: a competing job's load drifts
# over the minutes a row takes and a sequential schedule puts all of it on
# one variant (`docs/DF_WORK_LIST.md` §5.2)
best = {s: {k: float('inf') for k in STAGES} for s in SIDES}
for stage, fn in STAGES.items():
    for side in SIDES:
        fn(side)
_sync()
seen = {s: {} for s in SIDES}
for _ in range(max(1, a.reps)):
    for stage, fn in STAGES.items():
        for side in SIDES:
            best[side][stage] = min(best[side][stage],
                                    timeit(lambda: fn(side), 1))
            if stage == 'jk':
                # where the tensor is *at the moment the call runs*, which
                # a `reset()`/`build()` in the loop can move and which is
                # the single largest term in the number above.  Recorded
                # per repetition, because if it moved between them the
                # minimum below is a minimum over two different regimes
                c = obj[side]._cderi[0]
                where = seen[side].get('where_seen', set())
                where.add('device' if isinstance(c, cp.ndarray) else 'host')
                seen[side] = dict(
                    where_seen=where,
                    on_gpu=bool(isinstance(c, cp.ndarray)),
                    gb=float(c.nbytes) * 1e-9,
                    resident_rows=int(fastdf.STATS['resident_rows']),
                    streamed_rows=int(fastdf.STATS['streamed_rows']),
                    blksize=int(fastdf.STATS['blksize']),
                    tiled=int(fastdf.STATS['tiled']))
            fastdf.STATS['blocks'] = 0

print(f'\n{"stage":8s} ' + ' '.join(f'{s:>12s}' for s in SIDES) +
      ('   speedup' if len(SIDES) == 2 else ''), flush=True)
for stage in STAGES:
    line = f'{stage:8s} ' + ' '.join(f'{best[s][stage]*1e3:11.1f}ms'
                                     for s in SIDES)
    if len(SIDES) == 2:
        line += f'   {best["fp64"][stage]/best["fp32"][stage]:6.3f}x'
    print(line, flush=True)
for side in SIDES:
    d = seen[side]
    if not d:
        continue
    moved = '' if len(d['where_seen']) == 1 else \
        f'  ** MOVED between repetitions: {sorted(d["where_seen"])} **'
    print(f'  {side}: {d["gb"]:.2f} GB on '
          f'{"the device" if d["on_gpu"] else "the host"} during `jk`, '
          f'resident_rows={d["resident_rows"]} '
          f'streamed_rows={d["streamed_rows"]} tiled_calls={d["tiled"]} '
          f'tile_blksize={d["blksize"]}{moved}', flush=True)
    d['where_seen'] = sorted(d['where_seen'])

R = dict(xyz=a.xyz, basis=a.basis, natm=int(mol.natm), nao=int(mol.nao),
         cpu_cderi=a.cpu_cderi, info=info, stages=best, during_jk=seen)

if a.pcie:
    # The copy on its own.  `DF.loop` reported more than the factor of two
    # W7 predicts, and there are two candidate explanations -- half the
    # bytes, or a different effective rate -- which only a matched-byte
    # comparison separates: the third row below moves exactly as many bytes
    # as the single-precision tensor but in double precision.
    print('\n--- the host-to-device copy on its own ---', flush=True)
    print(f'{"tensor":8s} {"rows":>6s} {"blk":>5s} {"bytes":>9s} '
          f'{"time":>9s} {"GB/s":>7s}', flush=True)
    R['pcie'] = {}
    for side in SIDES:
        host = info[side].get('host_array')
        if host is None:
            continue
        naux = host.shape[0]
        for rows in (naux, naux // 2):
            src = host[:rows]
            for blk in (256, rows):
                buf = cp.empty((min(blk, rows), src.shape[1]),
                               dtype=src.dtype)

                def copy(src=src, buf=buf, blk=blk, rows=rows):
                    for p0 in range(0, rows, blk):
                        p1 = min(rows, p0 + blk)
                        buf[:p1 - p0].set(src[p0:p1])

                t = timeit(copy, max(2, a.reps))
                nbytes = src.nbytes
                print(f'{side:8s} {rows:6d} {blk:5d} {nbytes*1e-9:8.3f}G '
                      f'{t*1e3:8.1f}ms {nbytes*1e-9/t:7.2f}', flush=True)
                R['pcie'][f'{side}:{rows}:{blk}'] = dict(t=t, nbytes=nbytes)
                del buf
                cp.get_default_memory_pool().free_all_blocks()

if a.loop_ablate:
    # A clone of upstream's host-streaming generator, with the three things
    # that differ between the two precisions made switchable.  Checked
    # against the real one by the `upstream` row: it moves the same bytes in
    # the same order, so a difference between that row and `clone fp64` is
    # the clone's, not the precision's.
    print('\n--- the block stream, item by item ---', flush=True)
    print(f'{"variant":22s} {"bytes":>9s} {"time":>9s} {"GB/s":>7s}',
          flush=True)
    R['loop_ablate'] = {}

    def clone(host, widen, reuse=None):
        naux_slice, npairs = host.shape
        blksize = min(256, naux_slice)
        if reuse is None:
            buf = cp.empty((blksize, npairs), dtype=host.dtype)
            buf_prefetch = cp.empty_like(buf)
            wide = cp.empty((blksize, npairs)) if widen else None
        else:
            buf, buf_prefetch, wide = reuse
        comput_stream = cp.cuda.get_current_stream()
        compute_event = cp.cuda.Event()
        io_stream = cp.cuda.stream.Stream(non_blocking=True)
        io_event = cp.cuda.Event()
        buf_prefetch[:blksize].set(host[:blksize], stream=io_stream)
        io_event.record(io_stream)
        for p0, p1 in lib.prange(0, naux_slice, blksize):
            compute_event.record(comput_stream)
            buf, buf_prefetch = buf_prefetch, buf
            comput_stream.wait_event(io_event)
            p2 = min(naux_slice, p1 + blksize)
            if p1 < p2:
                io_stream.wait_event(compute_event)
                buf_prefetch[:p2 - p1].set(host[p1:p2], stream=io_stream)
                io_event.record(io_stream)
            if widen:
                cp.copyto(wide[:p1 - p0], buf[:p1 - p0])

    def row(label, fn, nbytes):
        t = timeit(fn, max(2, a.reps))
        print(f'{label:22s} {nbytes*1e-9:8.3f}G {t*1e3:8.1f}ms '
              f'{nbytes*1e-9/t:7.2f}', flush=True)
        R['loop_ablate'][label] = dict(t=t, nbytes=int(nbytes))

    for side in SIDES:
        host = info[side].get('host_array')
        if host is None:
            continue
        nb = host.nbytes
        row(f'{side} upstream loop',
            lambda s=side: [None for _ in obj[s].loop(unpack=False)], nb)
        row(f'{side} clone', lambda h=host: clone(h, False), nb)
        row(f'{side} clone+widen', lambda h=host: clone(h, True), nb)
        blk = min(256, host.shape[0])
        keep = (cp.empty((blk, host.shape[1]), dtype=host.dtype),
                cp.empty((blk, host.shape[1]), dtype=host.dtype),
                cp.empty((blk, host.shape[1])))
        row(f'{side} clone, buf kept', lambda h=host: clone(h, False, keep),
            nb)
        row(f'{side} clone+widen, kept',
            lambda h=host: clone(h, True, keep), nb)
        del keep
        cp.get_default_memory_pool().free_all_blocks()

for side in SIDES:
    obj[side].reset()
del obj
cp.get_default_memory_pool().free_all_blocks()

if a.scf:
    print('\n--- the whole SCF the stages sit in ---', flush=True)
    R['scf'] = {}
    for rnd in (1, 2):
        for side in SIDES:
            FP32._FP32 = (side == 'fp32')
            try:
                mf = make_mf()
                _sync()
                t = time.perf_counter()
                e = mf.kernel()
                _sync()
                dt = time.perf_counter() - t
            finally:
                FP32._FP32 = False
            if rnd == 2:
                R['scf'][side] = dict(t=dt, e=float(e), cycles=int(mf.cycles))
            print(f'  round {rnd}  {side}  {dt:8.2f} s   E = {e:.9f}  '
                  f'({mf.cycles} cycles)', flush=True)
            del mf
            cp.get_default_memory_pool().free_all_blocks()
    if len(SIDES) == 2:
        s_ = R['scf']
        print(f'  whole SCF {s_["fp64"]["t"]/s_["fp32"]["t"]:.3f}x, '
              f'dE = {abs(s_["fp64"]["e"]-s_["fp32"]["e"]):.2e} Eh',
              flush=True)

for side in info:
    info[side].pop('host_array', None)
print('RESULT ' + json.dumps(R, default=float), flush=True)
