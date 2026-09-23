"""The cderi build's host store: the transpose, and the stream it runs on.

``docs/DF_WORK_LIST.md`` **W11**.  When the cderi does not fit on the device
every block of it is moved to host memory by ``libvhf_rys.transpose_write``:
**24 %** of a 113-atom block loop and **16 %** of a 284-atom one, with nothing
overlapping it -- the next block's ``fill_int3c2e`` cannot start until it
returns.  Upstream's own source carries the TODO (``df/df.py:551``,
"async-write to host memory in another stream").

**It is not what W11 says it is.**  W11 calls it "a synchronous host-side
transpose-and-copy on one thread" and asks for the write's GB/s first, on the
theory that the transpose rather than the transfer is the cost.  Disassembled,
``transpose_write`` calls ``cudaHostGetDevicePointer`` and launches a CUDA
kernel -- one thread block per auxiliary function, 512 threads -- **on the
default stream**; measured, it moves a 2.05 GB block in 117.8 ms at
**17.37 GB/s**, and that number does not move by a microsecond when OpenMP is
given 1, 4, 16 or 32 threads.  There is no host transpose.  What sets its rate
is how long a contiguous run it writes into host memory, which is the block's
*pair* count: at 284-atom shapes the run is 888 kB and it reaches
**24.61 GB/s** against a plain ``cudaMemcpy``'s 26.28.  It is not slow there.
It is **synchronous**, on the stream that orders itself against everything
else the build does.

So the fix is the one the item names: the write on a stream of its own, with
**two** destination blocks so that the block being written is not the block
being computed.  Batch ``n`` contracts into one while batch ``n-1`` is still
being written out of the other, and batch ``n+2`` waits on an event that says
the write of ``n`` is done.  That is one extra ``(batch_pairs, naux)`` buffer
-- 2.05 GB at 113 atoms and 11.9 GB at 284 on an 80 GB card, where the sizing
that produced those numbers already leaves 30 GB free -- and this object
declines the whole arrangement and falls back to upstream's store if the
buffer will not fit.

**And the write is handed to a copy engine rather than to the SMs, which is
the half the item does not name and the half that pays.**  The transpose is
done on the device into a staging slice -- 2.7 ms of 117, 744 GB/s, free --
and what is left is a plain ``cudaMemcpy2DAsync`` at **24.36 GB/s** against
the kernel's 17.37.  Measured on the real block loop, that is **1.11x** at
both 42 and 113 atoms where moving *upstream's own kernel* to the same stream
is **1.00x**: a store done by the SMs has nothing to overlap with, because
``fill_int3c2e`` and the metric contraction are what the SMs are already
doing.  ``FASTDFCDERI_STORE=direct`` is that ablation and it is kept, because
it is the thing a synthetic block loop says should win -- a spin kernel
standing in for the integrals leaves the SMs free in a way the integrals do
not, and it reads `direct` 1886 ms against `stage` 1937 where the real call
reads the opposite.  ``docs/DF_WORK_LIST.md`` §5.7 in a third costume.

Three modes, so that the arrangement can be priced against what it replaces:

    FASTDFCDERI_STORE=host     upstream's `transpose_write`, on stream 0
    FASTDFCDERI_STORE=stage    the default: a tiled device transpose into a
                               staging slice and a `cudaMemcpy2DAsync` on
                               the store stream, so a copy engine does the
                               PCIe write
    FASTDFCDERI_STORE=direct   upstream's own kernel moved to the store
                               stream and nothing else -- the ablation that
                               separates the stream from the copy engine

``docs/README_fastdfstore.md`` is which of the three wins and by how much.

This module patches nothing (``docs/DF_WORK_LIST.md`` §5.12): the probe that
measures it must be able to import it without changing the build it is
measuring.
"""
import ctypes
import os

import numpy as np
import cupy as cp


from gpu4pyscf.df import int3c2e_bdiv as bdiv
from gpu4pyscf.lib.cupy_helper import get_avail_mem, ndarray
from electrolites import _dffp32 as FP32

libvhf_rys = bdiv.libvhf_rys

#: how a block reaches the host-resident tensor.  See the module docstring.
_STORE = os.environ.get('FASTDFCDERI_STORE', 'stage')
#: bytes of the staging slice the device transpose writes into.  It is walked
#: rather than sized to a whole block because a whole block is the tensor's
#: own batch -- 11.9 GB at 284 atoms -- and the staging buffer buys nothing
#: by being large: it is consumed by one ``cudaMemcpy2DAsync`` at a time on
#: one stream.
_STAGE_BYTES = int(float(os.environ.get('FASTDFCDERI_STAGE_MB', 256)) * 1e6)

#: ``cudaMemcpyDeviceToHost``
_D2H = 2
#: ``cudaHostGetDevicePointer``, which cupy's runtime wrapper does not export
#: -- it has ``hostRegister`` and ``hostAlloc`` but not the lookup that makes
#: the mapping usable from a kernel.  ``libvhf_rys.transpose_write`` calls it
#: itself, which is how upstream's own store reaches the host buffer.
try:
    _cudart = ctypes.CDLL('libcudart.so.12')
    _host_get_devptr = _cudart.cudaHostGetDevicePointer
    _host_get_devptr.restype = ctypes.c_int
    _host_get_devptr.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                 ctypes.c_void_p, ctypes.c_uint]
except OSError:                                              # noqa: BLE001
    _host_get_devptr = None

STATS = {'stores': 0, 'blocks': 0, 'bytes': 0, 'async_stores': 0}
DECLINED = {}


def _decline(reason):
    DECLINED[reason] = DECLINED.get(reason, 0) + 1


def stats():
    """What this module did in this process.  Nothing numerical can say."""
    out = dict(STATS)
    out['bytes'] = int(out['bytes'])         # numpy scalars are not JSON
    out['mode'] = _STORE
    out['declined_reasons'] = dict(DECLINED)
    return out


# --------------------------------------------------------------------------
# the transpose, on the device
# --------------------------------------------------------------------------
_TRANSPOSE_SRC = r'''
// `in` is (m, n) with row stride `ldin` and unit column stride; `out` is
// (n, m) with row stride `ldout`.  32x32 tiles through shared memory so that
// both the read and the write are coalesced; the +1 pad kills the bank
// conflict the diagonal read would otherwise have.  The strides are
// arguments rather than shapes because both sides are slices of something
// wider: the source is a column block of one batch's `(npairs, naux)` and
// the destination a row block of the `(naux, npairs)` tensor.
#define TILED_TRANSPOSE(OUT_T)                                              \
    __shared__ double tile[32][33];                                         \
    int x = blockIdx.x * 32 + threadIdx.x;          /* along n */           \
    int y = blockIdx.y * 32 + threadIdx.y;          /* along m */           \
    for (int k = 0; k < 32; k += 8) {                                       \
        if (x < n && y + k < m)                                             \
            tile[threadIdx.y + k][threadIdx.x] =                            \
                in[(long long)(y + k) * ldin + x];                          \
    }                                                                       \
    __syncthreads();                                                        \
    x = blockIdx.y * 32 + threadIdx.x;              /* along m */           \
    y = blockIdx.x * 32 + threadIdx.y;              /* along n */           \
    for (int k = 0; k < 32; k += 8) {                                       \
        if (x < m && y + k < n)                                             \
            out[(long long)(y + k) * ldout + x] =                           \
                (OUT_T)tile[threadIdx.x][threadIdx.y + k];                  \
    }

extern "C" __global__
void transpose_f64(double *out, const double *in, int m, int n,
                   long long ldin, long long ldout)
{ TILED_TRANSPOSE(double) }

extern "C" __global__
void transpose_f32(float *out, const double *in, int m, int n,
                   long long ldin, long long ldout)
{ TILED_TRANSPOSE(float) }

// The `direct` mode's kernel, which is `libvhf_rys.transpose_write`'s own
// access pattern: one thread block per auxiliary function, the threads of a
// block striding along the pair index.  The read is then strided by `naux`
// and the *write* is a long contiguous run -- which is the right way round
// when the destination is host memory across PCIe and the wrong way round
// for a device-to-device transpose, and is why the staging path uses the
// tiled kernel above and this one does not.  Keeping upstream's pattern is
// also what makes `direct` an ablation of the *stream* alone.
extern "C" __global__
void scatter_write_f64(double *out, const double *in,
                       int naux, int m, long long ldout)
{
    long long L = blockIdx.x;
    const double *row = in + L;
    double *dst = out + L * ldout;
    for (int p = threadIdx.x; p < m; p += blockDim.x)
        dst[p] = row[(long long)p * naux];
}

extern "C" __global__
void scatter_write_f32(float *out, const double *in,
                       int naux, int m, long long ldout)
{
    long long L = blockIdx.x;
    const double *row = in + L;
    float *dst = out + L * ldout;
    for (int p = threadIdx.x; p < m; p += blockDim.x)
        dst[p] = (float)row[(long long)p * naux];
}
'''
_mod = None


def _kernel(name):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_TRANSPOSE_SRC)
    return _mod.get_function(name)


def transpose(src, out):
    """``out[j, i] = src[i, j]``, for slices of two wider arrays.

    ``src`` is ``(m, n)`` and ``out`` is ``(n, m)``; both may be strided in
    their first index -- the store hands in a *column* block of one batch's
    ``(npairs, naux)`` and a *row* block of the staging buffer -- so the row
    strides are read off the arrays rather than assumed from the shapes.
    ``out`` may be ``float32``, which narrows in the same pass.
    """
    m, n = src.shape
    assert src.dtype == np.float64 and src.strides[1] == 8
    assert out.shape == (n, m) and out.strides[1] == out.dtype.itemsize
    if m == 0 or n == 0:
        return out
    name = 'transpose_f32' if out.dtype == np.float32 else 'transpose_f64'
    grid = ((n + 31) // 32, (m + 31) // 32)
    _kernel(name)(grid, (32, 8),
                  (out, src, np.int32(m), np.int32(n),
                   np.int64(src.strides[0] // 8),
                   np.int64(out.strides[0] // out.dtype.itemsize)))
    return out


def scatter_write(src, out):
    """``out[j, i] = src[i, j]`` with upstream's own access pattern.

    One thread block per row of ``out``; see the kernel's own comment for
    why that is the right shape when ``out`` is host memory and the wrong
    one when it is not.
    """
    m, n = src.shape
    assert src.flags.c_contiguous and src.dtype == np.float64
    assert out.shape == (n, m)
    if m == 0 or n == 0:
        return out
    name = ('scatter_write_f32' if out.dtype == np.float32
            else 'scatter_write_f64')
    ldout = out.strides[0] // out.dtype.itemsize
    _kernel(name)((n,), (512,), (out, src, np.int32(n), np.int32(m),
                                 np.int64(ldout)))
    return out


def device_view_of(host_array):
    """A device-side ``cp.ndarray`` over mapped host memory, or ``None``.

    ``cudaHostRegisterMapped``/``cudaHostAllocMapped`` memory has a device
    address, which is what lets a kernel write into it; cupy does not wrap
    the lookup in anything that returns an array, so the pointer is fetched
    and re-wrapped here.
    """
    if _host_get_devptr is None:
        return None
    out = ctypes.c_void_p()
    err = _host_get_devptr(ctypes.byref(out),
                           ctypes.c_void_p(host_array.ctypes.data), 0)
    if err != 0 or not out.value:
        return None
    mem = cp.cuda.UnownedMemory(out.value, host_array.nbytes, host_array)
    return cp.ndarray(host_array.shape, dtype=host_array.dtype,
                      memptr=cp.cuda.MemoryPointer(mem, 0))


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------
class HostStore:
    """Two destination blocks, two more streams, and the write that overlaps.

    The build's loop becomes

        dest = store.block(m)      # a (m, naux) device buffer to contract into
        ...                        # contract, and recontract, into `dest`
        store.commit(dest, p0, p1)
        ...
        store.finish()

    and what happens behind ``commit`` depends on the mode: upstream's own
    kernel on the default stream, the same kernel moved to a stream of its
    own (``direct``), or a device transpose and a ``cudaMemcpy2DAsync``
    (``stage``).  ``block`` is where the ordering lives -- it waits on the
    event that says the store two batches ago has finished with the buffer
    it is about to hand out.

    """

    def __init__(self, cderi_cpu, naux, npairs, batch_pairs, fp32=False,
                 mode=None):
        self.host = cderi_cpu
        self.naux = int(naux)
        self.npairs = int(npairs)
        self.batch_pairs = int(batch_pairs)
        self.fp32 = bool(fp32)
        self.mode = mode or _STORE
        if self.mode not in ('host', 'stage', 'direct'):
            raise ValueError(f'FASTDFCDERI_STORE={self.mode!r}: expected one '
                             "of 'host', 'stage', 'direct'")
        self.itemsize = 4 if fp32 else 8
        self.i = 0
        self.stream = None
        self.copy_stream = None
        self.host_dev = None
        self.stage = None

        block_words = self.batch_pairs * self.naux
        nbuf = 1
        if self.mode != 'host':
            # the second destination block is what buys the overlap, and it
            # is the only thing this costs.  `get_avail_mem` is asked *after*
            # the caller's own buffers are allocated -- and with the memory
            # pool counted in, because that is where these allocations come
            # from -- and the margin is the staging slice plus a fifth of the
            # block, because a cderi build that is this close to the edge
            # should not be made to fail for a 16 % saving.
            want = block_words * 8 + _STAGE_BYTES + block_words * 8 // 5
            if get_avail_mem() < want:
                _decline('not enough device memory for the second block')
                self.mode = 'host'
            else:
                nbuf = 2
        if self.mode == 'direct':
            self.host_dev = device_view_of(cderi_cpu)
            if self.host_dev is None:
                _decline('host buffer has no device pointer')
                self.mode = 'stage'

        self.bufs = [cp.empty(block_words) for _ in range(nbuf)]
        self.ev_store = [cp.cuda.Event(block=False, disable_timing=True)
                         for _ in range(nbuf)]
        self.ev_block = [cp.cuda.Event(block=False, disable_timing=True)
                         for _ in range(nbuf)]
        if self.mode != 'host':
            self.stream = cp.cuda.Stream(non_blocking=True)
        # `stage` was also tried with the transposes and the copies on two
        # streams and two staging slices, so that the copy engine never waits
        # for a transpose.  It is **slower**, at both sizes measured -- 2177
        # against 1937 ms on a synthetic 113-atom block loop and 12880
        # against 12520 at 284 atoms -- so the second stream is not here.
        # `docs/README_fastdfstore.md` 2.4.
        if self.mode == 'stage':
            # the slice is taken along the **auxiliary** index, not the pair
            # index, and that is a measurement rather than a preference.  A
            # slice of pairs makes each row of the `cudaMemcpy2DAsync` as
            # narrow as the slice -- 45 kB at a 256 MB budget and taxol's
            # shapes -- and a 2D copy that narrow does not reach the link's
            # rate; a slice of auxiliary functions keeps every row the whole
            # block wide, which is 365 kB, and the copy runs at 24.4 GB/s.
            self.stage_rows = max(
                1, min(self.naux,
                       _STAGE_BYTES // (self.batch_pairs * self.itemsize)))
            self.stage = cp.empty(
                (self.stage_rows, self.batch_pairs),
                dtype=np.float32 if fp32 else np.float64)
        elif self.mode == 'host' and fp32:
            # the narrow path has to transpose somewhere even without the
            # second stream, and upstream's host transpose is compiled for
            # `double`; this is `docs/README_fastdffp32.md`'s own staging
            # block, kept for the fallback
            self.stage_rows = self.naux
            self.stage = cp.empty((self.naux, self.batch_pairs),
                                  dtype=np.float32)

    # -- what the loop calls ------------------------------------------------
    def block(self, nrows):
        """A ``(nrows, naux)`` device buffer for this batch to contract into."""
        i = self.i % len(self.bufs)
        if self.stream is not None:
            # the store of batch `i - len(bufs)` read this buffer; an event
            # that was never recorded counts as complete, so the first two
            # batches do not wait
            cp.cuda.get_current_stream().wait_event(self.ev_store[i])
        return ndarray((nrows, self.naux), buffer=self.bufs[i])

    def commit(self, block, p0, p1):
        """Move ``block`` into ``host[:, p0:p1]``, asynchronously if it can."""
        STATS['blocks'] += 1
        STATS['bytes'] += int(p1 - p0) * self.naux * self.itemsize
        i = self.i % len(self.bufs)
        self.i += 1
        if self.mode == 'host':
            return self._store_sync(block, p0, p1)
        STATS['async_stores'] += 1
        ev = self.ev_block[i]
        ev.record(cp.cuda.get_current_stream())
        self.stream.wait_event(ev)
        with self.stream:
            if self.mode == 'direct':
                scatter_write(block, self.host_dev[:, p0:p1])
            else:
                m = p1 - p0
                for L0 in range(0, self.naux, self.stage_rows):
                    L1 = min(self.naux, L0 + self.stage_rows)
                    view = self.stage[:L1 - L0, :m]
                    transpose(block[:, L0:L1], view)
                    FP32.copy_2d_to_host(self.host[L0:L1, p0:p1], view)
            self.ev_store[i].record(self.stream)

    def finish(self):
        """Everything written and visible to the host."""
        STATS['stores'] += 1
        if self.stream is not None:
            self.stream.synchronize()

    # -- upstream's own -----------------------------------------------------
    def _store_sync(self, block, p0, p1):
        if self.fp32:
            view = self.stage[:, :p1 - p0]
            FP32.transpose_cast(block, view)
            FP32.copy_2d_to_host(self.host[:, p0:p1], view)
            return
        err = libvhf_rys.transpose_write(
            self.host.ctypes,
            ctypes.cast(block.data.ptr, ctypes.c_void_p),
            ctypes.c_int(self.naux), ctypes.c_int(self.npairs),
            ctypes.c_int(p0), ctypes.c_int(p1))
        if err != 0:
            raise RuntimeError('transpose_write cderi_cpu failed')
