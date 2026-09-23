"""Where the block-sparse exchange build's extra 1.5 s a call actually goes.

`docs/README_fastdfblock.md` §4.1 is an open question: the three kernels the
tiled first contraction replaces are **1.09x** faster in
`benchmarks/df_block_bench.py`, and the `get_jk` call they are part of is
**0.85x-1.02x**.  Wall clocks cannot say why, so this annotates the block loop
with NVTX and hands the answer to Nsight Systems.

It puts a range around every stage of every block, for both variants, in one
process on one cderi tensor:

    nextblock   `fastdf._blocks` resuming -- the wait on the prefetch's event,
                which is where an exposed PCIe stream would show up
    jpass       the Coulomb pass over the packed block
    unpack      `fill_symmetric` (dense) or the row-compressed scatter (tiled)
    step1       one cuTENSOR contraction (dense) or one per tile row (tiled)
    step2       the `dsyrk`

The patching is done here rather than in `electrolites/fastdf.py` because
nothing in the shipped module should carry a profiler hook.

Three numbers decide it, and `--stats` prints the commands that produce them:

* **GPU busy time per range** (`nvtx_gpu_proj_sum`).  If the tiled path's
  kernels really are faster and the call is not, the sum of its GPU time is
  smaller than the dense path's and the wall clock is larger.
* **Launch-to-execution gap** (`cuda_kern_exec_sum`).  85 to 169 contractions
  per block against one is 5400 launches a call against 130; if that is the
  cost, it is here.
* **Time in `nextblock`**.  If it grows in the tiled variant, the contraction
  got short enough to expose the host-to-device stream and the answer is
  bandwidth rather than scheduling.

    nsys profile -t cuda,nvtx -c cudaProfilerApi --cuda-memory-usage=false \\
        -o /path/prof python benchmarks/df_block_profile.py \\
        --xyz benchmarks/molecules/PfPMT.xyz --charge -1 --basis def2-svp \\
        --cycles 1 --cpu-cderi --tile 16
    python benchmarks/df_block_profile.py --stats /path/prof.nsys-rep
"""
import sys
import time

import numpy as np

if '--stats' in sys.argv:              # print the post-processing and go
    rep = sys.argv[sys.argv.index('--stats') + 1]
    for r in ('nvtx_pushpop_sum', 'nvtx_gpu_proj_sum', 'cuda_kern_exec_sum',
              'cuda_gpu_kern_sum'):
        print(f'nsys stats --report {r} {rep}')
    for v in ('dense', 'tiled', 'tiled-kernel', 'tiled-cupy'):
        print(f'nsys stats --report cuda_gpu_kern_sum --filter-nvtx={v} {rep}')
        print(f'nsys stats --report cuda_kern_exec_sum '
              f'--filter-nvtx={v} {rep}')
    sys.exit(0)

from _common import apply_patch, build_mol, parser        # noqa: E402

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--tile', type=int, default=16)
ap.add_argument('--tile-order', default='native')
ap.add_argument('--unpacks', default='kernel',
                help='which scatters into the row-compressed buffer to '
                     'profile, as their own NVTX variants: `kernel` (W14), '
                     '`cupy` (what it replaced), or both')
ap.add_argument('--cpu-cderi', action='store_true',
                help='force the cderi onto host memory, which is the regime '
                     'the open question is about')
ap.add_argument('--no-nvtx-stages', action='store_true',
                help='only the two outer ranges; the per-stage ones cost a '
                     'push/pop per block per stage and this says what that '
                     'costs')
a = ap.parse_args()
apply_patch(a.patch or 'fastdf')

import cupy as cp                                          # noqa: E402
from cupy.cuda import nvtx, profiler                       # noqa: E402
from gpu4pyscf import dft                                  # noqa: E402
from electrolites import fastdf                            # noqa: E402
from electrolites import _dfblock                          # noqa: E402


def _range(name, fn):
    """Wrap `fn` in an NVTX push/pop."""
    def run(*args, **kw):
        nvtx.RangePush(name)
        try:
            return fn(*args, **kw)
        finally:
            nvtx.RangePop()
    return run


def _patch_stages():
    fastdf.fill_symmetric = _range('unpack', fastdf.fill_symmetric)
    fastdf._syrk = _range('step2', fastdf._syrk)
    fastdf._vj_pass = _range('jpass', fastdf._vj_pass)
    _dfblock.RowCompressed.unpack = _range(
        'unpack', _dfblock.RowCompressed.unpack)
    _dfblock.RowCompressed.step1 = _range(
        'step1', _dfblock.RowCompressed.step1)

    _contract = fastdf.contract

    def contract(pattern, *args, **kw):
        # the dense first contraction is the only `contract` in the exchange
        # half; the Coulomb ablation's two are named for what they are
        name = 'step1' if pattern == 'Lij,njk->nikL' else f'contract:{pattern}'
        nvtx.RangePush(name)
        try:
            return _contract(pattern, *args, **kw)
        finally:
            nvtx.RangePop()
    fastdf.contract = contract

    _blocks = fastdf._blocks

    def blocks(*args, **kw):
        # resuming the generator is where the wait on the prefetch's event
        # lives, so the range has to be around `next()` and not around the
        # body of the loop
        it = _blocks(*args, **kw)
        while True:
            nvtx.RangePush('nextblock')
            try:
                item = next(it)
            except StopIteration:
                return
            finally:
                nvtx.RangePop()
            yield item
    fastdf._blocks = blocks


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
cderi = dfobj._cderi[cp.cuda.Device().id]
print(f'[{a.xyz.rsplit("/", 1)[-1]}/{a.basis}] natm={mol.natm} '
      f'nao={dfobj.nao} naux={dfobj.naux} npair={cderi.shape[1]} '
      f'cderi={cderi.nbytes/1e9:.2f} GB '
      f'on_gpu={isinstance(cderi, cp.ndarray)} '
      f'tile={a.tile}/{a.tile_order}', flush=True)

fastdf.TILE, fastdf.TILE_ORDER = a.tile, a.tile_order
VARIANTS = ([('dense', 'tensor', 'kernel')]
            + [(f'tiled-{u}' if len(a.unpacks.split(',')) > 1 else 'tiled',
                'tiled', u) for u in a.unpacks.split(',')])


def call(step1, unpack='kernel'):
    fastdf._STEP1 = step1
    _dfblock.UNPACK = unpack
    return fastdf.get_jk(dfobj, dm, 1, True, True)


# warm both: the tile pattern, the resident prefix and every cuTENSOR plan are
# built on first use and none of them is what this is measuring
ref = None
for name, step1, unpack in VARIANTS:
    vj, vk = call(step1, unpack)
    if ref is None:
        ref = vk
    else:
        scale = float(cp.abs(ref).max()) or 1.
        print(f'# {name} vs dense: relK = '
              f'{float(cp.abs(vk - ref).max()) / scale:.2e}', flush=True)
    del vj, vk
cp.get_default_memory_pool().free_all_blocks()

if not a.no_nvtx_stages:
    _patch_stages()

profiler.start()
for name, step1, unpack in VARIANTS:
    cp.cuda.Device().synchronize()
    nvtx.RangePush(name)
    t0 = time.perf_counter()
    call(step1, unpack)
    cp.cuda.Device().synchronize()
    dt = time.perf_counter() - t0
    nvtx.RangePop()
    print(f'{name:6s} {dt*1e3:9.2f} ms   blocks={fastdf.STATS["blocks"]} '
          f'blksize={fastdf.STATS["blksize"]} '
          f'fill={fastdf.STATS["tile_fill"]:.3f}', flush=True)
profiler.stop()
