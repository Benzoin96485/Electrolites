"""The cderi's host store and its page-locked buffer: the two rates.

``docs/DF_WORK_LIST.md`` **W11** and **W12**, measured on the shapes the build
uses but without a molecule, because neither is a property of the chemistry:
the store moves a ``(blk, naux)`` device block into a ``(naux, npairs)``
page-locked host tensor, and the buffer is page-locked once per build.

**W11** asks first for a falsification: *"measure the write's GB/s first;
14.31 GB in 821 ms is 17 GB/s, which is well under a PCIe gen4 x16 link and
suggests it is the transpose, not the transfer"*.  So the table prices

  * ``libvhf_rys.transpose_write``, upstream's host transpose, and what it
    does when it is given more OpenMP threads -- which says whether it is one
    thread or many;
  * a plain ``cudaMemcpy`` of the same bytes with no transpose at all, which
    is the link's own floor;
  * a device transpose into a staging slice and a ``cudaMemcpy2DAsync``;
  * the transpose kernel writing straight into the mapped host buffer;

and then the thing the item is actually about -- whether the store overlaps
the next block's integrals -- against a synthetic block loop with a kernel of
a known duration standing in for ``fill_int3c2e``.

**W12** is the buffer: ``--pool`` builds the same tensor several times over
and reports what the second and later builds pay.

    python benchmarks/df_store_probe.py --naux 5598 --blk 45683 --nblk 7
    python benchmarks/df_store_probe.py --naux 13365 --blk 111000 --nblk 8
    python benchmarks/df_store_probe.py --pool --naux 5598 --blk 45683 --nblk 7
"""
import argparse
import ctypes
import json
import os
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--naux', type=int, default=5598,
                help='auxiliary functions; 5598 is taxol/def2-SVP, 13365 is '
                     'PfPMT/def2-SVP')
ap.add_argument('--blk', type=int, default=45683,
                help='pairs in one block of the build')
ap.add_argument('--nblk', type=int, default=7, help='blocks in the tensor')
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--threads', default='',
                help='comma separated OpenMP thread counts to re-time '
                     '`transpose_write` at, e.g. 1,8,32')
ap.add_argument('--compute-ms', type=float, default=0.,
                help='the synthetic block loop stands a kernel of this many '
                     'milliseconds in for `fill_int3c2e`; 0 skips the loop')
ap.add_argument('--stage-mbs', default='',
                help='comma separated staging-slice sizes in MB to re-run '
                     'the block loop at, e.g. 64,256,1024')
ap.add_argument('--pool', action='store_true',
                help="W12: build the tensor's buffer several times over")
ap.add_argument('--fp32', action='store_true',
                help='the narrow tensor of W7, whose store is already a '
                     'device transpose')
a = ap.parse_args()

import cupy as cp                                            # noqa: E402
from cupy_backends.cuda.api import runtime                   # noqa: E402
from gpu4pyscf.df import int3c2e_bdiv as bdiv                # noqa: E402
from gpu4pyscf.lib.cupy_helper import empty_mapped           # noqa: E402

# `_dfpin` and `_dfstore` rather than `fastdfcderi`, because importing that
# module patches the build (`docs/DF_WORK_LIST.md` §5.12); neither of these
# patches anything.
from electrolites import _dfpin as PIN                       # noqa: E402
from electrolites import _dfstore as STORE                   # noqa: E402
from electrolites import _dffp32 as FP32                     # noqa: E402

libvhf_rys = bdiv.libvhf_rys
_sync = cp.cuda.Stream.null.synchronize
_D2H = 2

naux, blk, nblk = a.naux, a.blk, a.nblk
npairs = blk * nblk
dtype = np.float32 if a.fp32 else np.float64
itemsize = np.dtype(dtype).itemsize
gb = naux * npairs * itemsize * 1e-9
blk_gb = naux * blk * itemsize * 1e-9
print(f'# naux={naux} blk={blk} nblk={nblk} npairs={npairs} '
      f'dtype={np.dtype(dtype).name}  tensor {gb:.2f} GB, block {blk_gb:.2f} GB'
      f'  OMP_NUM_THREADS={os.environ.get("OMP_NUM_THREADS")}', flush=True)

R = {'naux': naux, 'blk': blk, 'nblk': nblk, 'fp32': a.fp32, 'gb': gb}


def timeit(fn, reps=None):
    reps = a.reps if reps is None else reps
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


# --------------------------------------------------------------------------
# W12: the buffer
# --------------------------------------------------------------------------
if a.pool:
    print('\n--- W12: the page-locked buffer, built again and again ---',
          flush=True)
    R['pool'] = {}
    for label, kw in (('cudaHostAlloc (GPU4PySCF)', None),
                      ('cudaHostRegister, no pool', dict(pool=False)),
                      ('cudaHostRegister + pool', dict(pool=True))):
        ts = []
        for _ in range(4):
            t = time.perf_counter()
            buf = (empty_mapped((naux, npairs), dtype) if kw is None
                   else PIN.empty_mapped_pooled((naux, npairs), dtype, **kw))
            ts.append(time.perf_counter() - t)
            buf[0, 0] = 1.                      # touch it, so it is resident
            del buf
        R['pool'][label] = ts
        rates = '  '.join(f'{gb/t:5.2f}' for t in ts)
        print(f'  {label:30s} ' +
              '  '.join(f'{t*1e3:8.1f} ms' for t in ts) +
              f'   GB/s {rates}', flush=True)
    PIN.free_pool()
    print(f'  {PIN.stats()}', flush=True)

# --------------------------------------------------------------------------
# W11: the store, one block at a time
# --------------------------------------------------------------------------
host = PIN.empty_mapped_pooled((naux, npairs), dtype, pool=False)
block = cp.asarray(np.random.default_rng(0).random((blk, naux)))
stage = cp.empty((naux, blk), dtype=dtype)
host_dev = STORE.device_view_of(host)
print(f'\n--- W11: one {blk_gb:.2f} GB block into the host tensor ---',
      flush=True)
R['store'] = {}


def _sync_write():
    err = libvhf_rys.transpose_write(
        host.ctypes, ctypes.cast(block.data.ptr, ctypes.c_void_p),
        ctypes.c_int(naux), ctypes.c_int(npairs),
        ctypes.c_int(0), ctypes.c_int(blk))
    assert err == 0


def _memcpy_flat():
    """The same bytes with no transpose: the link's own floor."""
    runtime.memcpyAsync(host.ctypes.data, block.data.ptr,
                        blk * naux * itemsize, _D2H,
                        cp.cuda.get_current_stream().ptr)


def _transpose_only():
    if a.fp32:
        FP32.transpose_cast(block, stage)
    else:
        STORE.transpose(block, stage)


def _stage_then_copy():
    """One whole-block staging buffer, then one wide 2D copy."""
    _transpose_only()
    FP32.copy_2d_to_host(host[:, :blk], stage)


_STAGE_ROWS = max(1, min(naux, STORE._STAGE_BYTES // (blk * itemsize)))
_slice = cp.empty((_STAGE_ROWS, blk), dtype=dtype)


def _stage_rows():
    """The store's own path: a slice of *auxiliary functions* at a time.

    Every row of the copy is then the whole block wide, which is what the
    2D copy needs; slicing pairs instead makes the rows as narrow as the
    slice and the copy does not reach the link's rate.
    """
    for L0 in range(0, naux, _STAGE_ROWS):
        L1 = min(naux, L0 + _STAGE_ROWS)
        view = _slice[:L1 - L0]
        STORE.transpose(block[:, L0:L1], view)
        FP32.copy_2d_to_host(host[L0:L1, :blk], view)


def _direct():
    """Upstream's access pattern into mapped host memory, on this stream."""
    STORE.scatter_write(block, host_dev[:, :blk])


def _direct_tiled():
    """The same destination, reached by the tiled kernel instead."""
    if a.fp32:
        FP32.transpose_cast(block, host_dev[:, :blk])
    else:
        STORE.transpose(block, host_dev[:, :blk])


cases = [('transpose_write (GPU4PySCF)', _sync_write, not a.fp32),
         ('cudaMemcpy, no transpose', _memcpy_flat, True),
         ('device transpose only (no copy)', _transpose_only, True),
         ('device transpose + memcpy2DAsync', _stage_then_copy, True),
         (f'the same in {_STAGE_ROWS}-row slices', _stage_rows, True),
         ('scatter into mapped host memory', _direct, host_dev is not None),
         ('tiled transpose into mapped host', _direct_tiled,
          host_dev is not None)]
for label, fn, ok in cases:
    if not ok:
        print(f'  {label:38s}        n/a', flush=True)
        continue
    t = timeit(fn)
    R['store'][label] = t
    print(f'  {label:38s} {t*1e3:9.1f} ms   {blk_gb/t:6.2f} GB/s', flush=True)

# how many threads upstream's transpose uses, by giving it more
if a.threads and not a.fp32:
    print('\n--- `transpose_write` against OMP_NUM_THREADS ---', flush=True)
    try:
        gomp = ctypes.CDLL('libgomp.so.1')
    except OSError:
        gomp = None
    if gomp is None:
        print('  libgomp not loadable; skipped', flush=True)
    else:
        R['threads'] = {}
        for n in [int(x) for x in a.threads.split(',')]:
            gomp.omp_set_num_threads(ctypes.c_int(n))
            t = timeit(_sync_write)
            R['threads'][n] = t
            print(f'  {n:3d} threads {t*1e3:9.1f} ms   {blk_gb/t:6.2f} GB/s',
                  flush=True)
        gomp.omp_set_num_threads(
            ctypes.c_int(int(os.environ.get('OMP_NUM_THREADS', 32))))

# --------------------------------------------------------------------------
# W11: what the overlap is worth, against a synthetic block loop
# --------------------------------------------------------------------------
if a.compute_ms > 0:
    print(f'\n--- the block loop, with {a.compute_ms:.0f} ms of "integrals" '
          f'per block ---', flush=True)
    # a kernel that occupies the device for a fixed time without touching
    # anything the store touches, standing in for `fill_int3c2e`
    spin = cp.RawKernel(r'''
extern "C" __global__ void spin(long long cycles, float *sink) {
    long long t0 = clock64();
    float acc = 0.f;
    while (clock64() - t0 < cycles) acc += 1.f;
    if (threadIdx.x == 1024) sink[0] = acc;
}
''', 'spin')
    sink = cp.zeros(1, dtype=np.float32)
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    khz = props['clockRate']                       # kHz
    cycles = int(a.compute_ms * khz)
    # enough blocks to fill the device, so that the `direct` mode's transpose
    # -- which is done by the SMs rather than by a copy engine -- has to
    # compete for them the way it would with `fill_int3c2e`
    spin_grid = (props['multiProcessorCount'] * 4,)

    R['loop'] = {}
    # the staging slice is swept as well as the mode, because the store
    # stream runs one transpose and one copy per slice *in series* and the
    # number of slices is `naux / stage_rows`: if the slice is what decides
    # whether the overlap is reached, this is where it shows
    stage_mbs = [float(x) for x in a.stage_mbs.split(',')] if a.stage_mbs \
        else [STORE._STAGE_BYTES * 1e-6]
    cases = ([('host', None), ('direct', None)] +
             [('stage', mb) for mb in stage_mbs])
    saved = STORE._STAGE_BYTES
    for mode, mb in cases:
        STORE._STAGE_BYTES = saved if mb is None else int(mb * 1e6)
        store = STORE.HostStore(host, naux, npairs, blk, fp32=a.fp32,
                                mode=mode)
        if store.mode != mode:
            print(f'  {mode:8s} declined -> {store.mode}', flush=True)

        def one_loop(store=store):
            for i in range(nblk):
                spin(spin_grid, (256,), (np.int64(cycles), sink))
                dst = store.block(blk)
                cp.copyto(dst, block)
                store.commit(dst, i * blk, (i + 1) * blk)
            store.finish()
        t = timeit(one_loop, reps=2)
        label = mode if mb is None else f'{mode}/{mb:.0f}MB'
        rows = getattr(store, 'stage_rows', 0)
        R['loop'][label] = t
        print(f'  {label:14s} {t*1e3:9.1f} ms   '
              f'({nblk} x {a.compute_ms:.0f} ms = '
              f'{nblk*a.compute_ms:.0f} ms of "integrals"'
              f'{f", {rows}-row slices, {-(-naux // rows)} of them" if rows else ""})',
              flush=True)
        del store
        cp.get_default_memory_pool().free_all_blocks()
    STORE._STAGE_BYTES = saved

print('@@@JSON ' + json.dumps(R), flush=True)
