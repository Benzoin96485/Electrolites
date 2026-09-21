"""The cderi in single precision: the switch, the layout, and the two casts.

``docs/DF_WORK_LIST.md`` **W7**.  The three-index tensor is the largest object
a density-fitted run holds -- 14.3 GB at 113 atoms and 86.4 GB at 284
(``docs/DF_WORK_LIST.md`` §3.2) -- and it is *storage*: every element is read,
multiplied and summed into a double accumulator, and none is read twice
within a block.  Stored as ``float32`` it is exactly half the bytes and
exactly half the PCIe traffic per SCF cycle, and the arithmetic does not
change: each block is widened back to ``float64`` on the device, in a buffer
that is allocated once, and every contraction downstream of that is the same
``dgemm``/``dsyrk``/cuTENSOR call on the same operand shapes.

**It changes the answer.**  That is not a bug to be tolerated, it is the whole
of what the mode costs, and this repository's correctness tables are at 1e-14.
So it is off unless asked for, it is asked for by name, and the error it
introduces is measured rather than argued -- ``benchmarks/df_fp32_bench.py``
on the §3.2 ladder, ``docs/README_fastdffp32.md`` for the numbers.

This module patches nothing, for the reason ``docs/DF_WORK_LIST.md`` §5.12
gives: ``electrolites/fastdf.py`` and ``electrolites/fastdfcderi.py`` both need
the policy and the casts, and importing either of *those* patches a build.
``electrolites/fastdffp32.py`` is the module that patches, and what it patches
is the two consumers of the tensor that are not ``df_jk.get_jk``.

    FASTDF_FP32=1   store the cderi as float32.  Default off

**The layout is upstream's, in both places, and that is a measurement.**
A device-resident tensor is ``(npairs, naux)`` handed back as a view of its
transpose and a host-resident one is ``(naux, npairs)``; single precision
keeps both.  The first version of this module did not -- it wrote
``(naux, npairs)`` everywhere, on the argument that a block is then
contiguous and the widening pass perfectly coalesced.  That is true and it
is the wrong trade: the widened block is what ``fill_symmetric`` unpacks,
and the two layouts take *different kernels* there.  Upstream's device
layout takes ``decompress_and_fill`` and the other takes
``decompress_and_transpose``, which is the slower of the two for the same
reason an ``(L,i,j)`` unpack was 2.65x slower in
``docs/README_fastdf.md`` §3.4 -- and measured, that cost **1.6x on a whole
`get_jk` call** at 42 atoms where the widening pass itself costs a fifth of
that.  Keeping upstream's layout makes the widening a strided *copy* rather
than a transpose (the auxiliary index is the contiguous one on both sides of
it, so each row of the copy is a contiguous run) and leaves every kernel
downstream on the branch it was tuned for.

The one place a transpose is still needed is the **host** build, where
upstream pays one too: ``libvhf_rys.transpose_write`` moves the
``(npairs, naux)`` device block into the ``(naux, npairs)`` host tensor on
the CPU, and it is compiled for ``double``.  `transpose_cast` below does
that on the device instead, narrowing in the same pass.

Nothing outside this repository can read the result: ``fill_symmetric`` and
``transpose_write`` are compiled for ``double`` and take ``void*``, so they
would not fail on a ``float32`` tensor, they would read past the end of it.
That is why the mode reaches every consumer or none -- see
``electrolites/fastdffp32.py``.
"""
import os

import numpy as np
import cupy as cp
from cupy_backends.cuda.api import runtime

#: whether the cderi is stored in single precision.  Read once at import, as
#: every other switch in this repository is, and flipped on the module object
#: by `tests/test_dffp32.py` so that both precisions can be built in one
#: process.
_FP32 = os.environ.get('FASTDF_FP32', '') not in ('', '0')

#: what the mode did, so a claim about it can be read off a run.  `stores` is
#: how many tensors were written in single precision and `widened` how many
#: blocks were widened back; `saved_gb` is the difference between what was
#: allocated and what a double tensor would have taken.
STATS = {'stores': 0, 'widened': 0, 'saved_gb': 0.0}


def enabled():
    """Whether new cderi tensors are stored in single precision."""
    return _FP32


# --------------------------------------------------------------------------
# the transpose the build pays once
# --------------------------------------------------------------------------
_TRANSPOSE_SRC = r'''
extern "C" __global__
void transpose_cast_f64_f32(float *out, const double *in,
                            int m, int n, long long ldout)
{
    // in is (m, n) row major and contiguous; out is (n, m) row major with
    // row stride `ldout`.  32x32 tiles through shared memory so that both
    // the read and the write are coalesced; the +1 pad kills the bank
    // conflict the diagonal read would otherwise have.
    __shared__ double tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;          // along n
    int y = blockIdx.y * 32 + threadIdx.y;          // along m
    for (int k = 0; k < 32; k += 8) {
        if (x < n && y + k < m)
            tile[threadIdx.y + k][threadIdx.x] = in[(long long)(y + k) * n + x];
    }
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;              // along m
    y = blockIdx.x * 32 + threadIdx.y;              // along n
    for (int k = 0; k < 32; k += 8) {
        if (x < m && y + k < n)
            out[(long long)(y + k) * ldout + x] =
                (float)tile[threadIdx.x][threadIdx.y + k];
    }
}
'''
_transpose_mod = None


def _transpose_kernel():
    global _transpose_mod
    if _transpose_mod is None:
        _transpose_mod = cp.RawModule(code=_TRANSPOSE_SRC)
    return _transpose_mod.get_function('transpose_cast_f64_f32')


def transpose_cast(src, out):
    """``out[j, i] = (float32) src[i, j]`` for a C-contiguous ``(m, n)`` src.

    ``out`` is ``(n, m)``-shaped and may be a column slice of a wider array,
    which is how the build writes one block of pairs into the whole tensor;
    its row stride is taken from the array rather than from its shape.
    """
    m, n = src.shape
    assert src.flags.c_contiguous and src.dtype == np.float64
    assert out.shape == (n, m) and out.dtype == np.float32
    if m == 0 or n == 0:
        return out
    ldout = out.strides[0] // 4
    grid = ((n + 31) // 32, (m + 31) // 32)
    _transpose_kernel()(grid, (32, 8), (out, src, np.int32(m), np.int32(n),
                                        np.int64(ldout)))
    return out


#: ``cudaMemcpyDeviceToHost``; cupy does not export the enum by name here.
_D2H = 2


def copy_2d_to_host(dst, src):
    """``dst[:] = src`` for a device ``(rows, cols)`` into a host slice.

    The host destination is a column block of the pinned ``(naux, npairs)``
    buffer, so it is strided and a flat copy cannot express it.  Issued on
    the current stream, and ordered with respect to the kernel that filled
    ``src`` for that reason -- the caller reuses ``src`` on the next batch.
    """
    rows, cols = src.shape
    if rows == 0 or cols == 0:
        return dst
    itemsize = src.dtype.itemsize
    assert dst.shape == src.shape and dst.dtype == src.dtype
    assert dst.strides[1] == itemsize and src.strides[1] == itemsize
    runtime.memcpy2DAsync(
        dst.ctypes.data, dst.strides[0], src.data.ptr, src.strides[0],
        cols * itemsize, rows, _D2H, cp.cuda.get_current_stream().ptr)
    return dst


# --------------------------------------------------------------------------
# the cast every block pays
# --------------------------------------------------------------------------
class BlockWidener:
    """One ``float64`` block buffer, reused, and the copy that fills it.

    Allocated before the block loop rather than inside it: an allocation in
    a loop that relies on a prefetch costs the overlap rather than the
    allocation, which is ``docs/DF_WORK_LIST.md`` §5.8 and §5.11.

    The buffer's shape follows the tensor's layout so that the **auxiliary**
    index is the contiguous one on both sides of the copy for a
    device-resident tensor and the **pair** index is for a host-resident
    one, which is what upstream's two layouts are.  The copy is then issued
    on the orientation that has a unit stride last, so cupy's elementwise
    kernel walks both operands contiguously; issued on the block's own
    ``(nL, npairs)`` orientation instead it would be strided on both sides.
    """

    def __init__(self, cderi_sparse, blksize, on_gpu):
        npairs = cderi_sparse.shape[1]
        self.on_gpu = on_gpu
        self.buf = (cp.empty((npairs, blksize)) if on_gpu
                    else cp.empty((blksize, npairs)))

    def __call__(self, src, nL):
        """``src`` is one ``(nL, npairs)`` block of the narrow tensor."""
        STATS['widened'] += 1
        if self.on_gpu:
            cp.copyto(self.buf[:, :nL], src.T)
            return self.buf.T[:nL]
        cp.copyto(self.buf[:nL], src)
        return self.buf[:nL]


def note_store(shape, dtype):
    """Record a tensor this mode sized, for `stats`."""
    if dtype == np.float32:
        STATS['stores'] += 1
        STATS['saved_gb'] += float(np.prod(shape)) * 4e-9


def stats():
    """What the mode did in this process.  Nothing numerical can say."""
    out = dict(STATS)
    out['enabled'] = _FP32
    return out
