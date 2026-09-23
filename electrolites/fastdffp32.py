"""The two consumers of a single-precision cderi that are not ``get_jk``.

``docs/DF_WORK_LIST.md`` **W7**.  ``electrolites/_dffp32.py`` is the mode
itself -- the switch, the layout and the two casts -- and it patches nothing,
because ``electrolites/fastdf.py`` (which consumes the tensor in the SCF) and
``electrolites/fastdfcderi.py`` (which writes it) both need it and importing
either of those patches a build (``docs/DF_WORK_LIST.md`` §5.12).  This module
is what patches, and it exists because **a narrow tensor has no fall-through**:

``fill_symmetric`` and ``transpose_write`` are compiled for ``double`` and
take a ``void*``.  Handed a ``float32`` tensor they do not raise, they read
twice as many bytes as there are and return a plausible wrong answer.  So
every path that touches the tensor either knows about the mode or the mode
cannot be switched on at all.  Three do:

1. ``df/df_jk.py:get_jk`` -- the SCF, the response solve and a DF Hessian's
   ``get_veff``.  ``electrolites/fastdf.py`` covers it, and the calls that
   module *declines* now raise rather than fall through to a build that would
   misread the tensor.
2. ``df/df.py:DF.loop`` -- the block iterator **everything else** goes
   through: the DF gradient's two-electron derivative
   (``electrolites/fastdfgrad.py``'s ``_fill_from_cderi``) and the DF
   Hessian's own MO-basis CPHF build (``df/hessian/rhf.py:_get_jk``, the
   fifth two-electron path, which is ``docs/DF_WORK_LIST.md`` W8).  Patched
   here to widen each block, which is the same buffer-allocated-once
   arrangement ``fastdf._blocks`` uses.
3. ``df/hessian/rhf.py:_get_jk`` -- **only** for the cderi transfer at its
   top (``:1258-1273``), which moves a device-resident tensor to the host to
   leave room for the Krylov vectors and does it with ``empty_mapped`` at
   ``float64`` and a ``transpose_write``.  The wrapper here does that
   transfer itself when the tensor is narrow, in slices, and then hands the
   call on.  Upstream's own block is
   inside ``if dfobj._cderi is None`` and so does not run a second time.
   Nothing else in that function is touched: it reaches the tensor through
   ``DF.loop``.

Both patches are **upstream's own function, unchanged, whenever the tensor is
``float64``** -- ``loop`` delegates to the original generator and the Hessian
wrapper does nothing -- so importing this module changes no double-precision
result and no double-precision timing.  ``stats()`` says whether the narrow
path was taken, because nothing numerical can.
"""
import numpy as np
import cupy as cp
from pyscf import lib

from gpu4pyscf.df import df as DFMOD
from gpu4pyscf.df.df import DF
from gpu4pyscf.df.hessian import rhf as DFHESS
from gpu4pyscf.lib.cupy_helper import (
    empty_mapped, fill_symmetric, get_avail_mem)

from electrolites import _dffp32 as FP32

#: counters, so a coverage claim can be read off a run rather than assumed.
STATS = {'loops': 0, 'blocks': 0, 'hessian_transfers': 0}

_orig_loop = DF.loop
#: where `tests/test_patch_aliases.py` looks for the function this replaced
DF._loop_orig = _orig_loop
_orig_hess_get_jk = DFHESS._get_jk
DFHESS._get_jk_orig = _orig_hess_get_jk


def stats():
    """What this module did in this process."""
    out = dict(STATS)
    out.update(FP32.stats())
    return out


def loop(self, blksize=None, unpack=True):
    """``df/df.py:DF.loop`` with the block widened to ``float64``.

    Upstream's generator for a double tensor, byte for byte -- this is a
    delegation, not a copy, so nothing about the default configuration
    changes.  For a narrow tensor it is upstream's structure with three
    differences: the staging buffers are ``float32`` (which is the half of
    the PCIe traffic the mode is for), one ``float64`` block buffer is
    allocated before the loop and reused (§5.8: an allocation inside a loop
    that relies on a prefetch costs the overlap, not the allocation), and the
    unpack is fed the widened block.
    """
    if self._cderi is None:
        self.build()
    device_id = cp.cuda.Device().id
    cderi_sparse = self._cderi[device_id]
    if cderi_sparse.dtype != np.float32:
        yield from _orig_loop(self, blksize, unpack)
        return

    STATS['loops'] += 1
    nao = self.nao
    naux_slice, npairs = cderi_sparse.shape
    if blksize is None:
        blksize = self.get_blksize()
    blksize = min(blksize, naux_slice)
    if blksize == 0:
        return
    on_gpu = isinstance(cderi_sparse, cp.ndarray)
    pair_idx = cp.asarray(self._cderi_idx[0], dtype=np.int32)
    work = cp.zeros((nao, nao, blksize)) if unpack else None
    widen = FP32.BlockWidener(cderi_sparse, blksize, on_gpu)

    def emit(blk, nL):
        STATS['blocks'] += 1
        # the widened block carries the layout the tensor has, so
        # `fill_symmetric` takes the same branch it would have taken on a
        # double tensor in the same place -- which is not a detail, see
        # `electrolites/_dffp32.py`
        blk = widen(blk, nL)
        if not unpack:
            return None, blk
        out = fill_symmetric(blk.T, pair_idx, nao, 0, nL, out=work[:, :, :nL])
        return out.transpose(2, 0, 1), blk

    if on_gpu:
        for p0, p1 in lib.prange(0, naux_slice, blksize):
            yield emit(cderi_sparse[p0:p1], p1 - p0)
        return

    buf = cp.empty((blksize, npairs), dtype=np.float32)
    buf_prefetch = cp.empty_like(buf)
    comput_stream = cp.cuda.get_current_stream()
    compute_event = cp.cuda.Event()
    io_stream = cp.cuda.stream.Stream(non_blocking=True)
    io_event = cp.cuda.Event()

    buf_prefetch[:blksize].set(cderi_sparse[:blksize], stream=io_stream)
    io_event.record(io_stream)

    for p0, p1 in lib.prange(0, naux_slice, blksize):
        compute_event.record(comput_stream)
        buf, buf_prefetch = buf_prefetch, buf
        comput_stream.wait_event(io_event)
        p2 = min(naux_slice, p1 + blksize)
        if p1 < p2:
            io_stream.wait_event(compute_event)
            buf_prefetch[:p2 - p1].set(cderi_sparse[p1:p2], stream=io_stream)
            io_event.record(io_stream)
        yield emit(buf[:p1 - p0], p1 - p0)


def _transfer_to_host(dfobj):
    """The Hessian's cderi transfer, for a tensor upstream cannot transpose.

    ``df/hessian/rhf.py:1264`` allocates ``empty_mapped(shape)`` -- double,
    unconditionally -- and calls ``transpose_write``, which is compiled for
    ``double``: a device tensor is ``(npairs, naux)`` and a host one is
    ``(naux, npairs)``, in either precision.  Done here in slices of about
    256 MB, because the point of the transfer is to *free* device memory and
    transposing the whole tensor on the device first would need a second
    copy of it.
    """
    STATS['hessian_transfers'] += 1
    device_id = cp.cuda.Device().id
    src = dfobj._cderi[device_id]                       # (naux, npairs) view
    naux, npairs = src.shape
    host = empty_mapped(src.shape, dtype=src.dtype)
    chunk = max(1, int(64e6) // max(naux, 1))
    for c0 in range(0, npairs, chunk):
        c1 = min(npairs, c0 + chunk)
        host[:, c0:c1] = cp.asnumpy(cp.ascontiguousarray(src[:, c0:c1]))
    dfobj._cderi[device_id] = host
    del src
    cp.get_default_memory_pool().free_all_blocks()


def _get_jk(dfobj, dms, mo_coeff, mo_occ, hermi=1, with_j=True, with_k=True,
            omega=None):
    """``df/hessian/rhf.py:_get_jk``, with its one fp64-only line handled.

    Everything this function does with the tensor goes through ``DF.loop``,
    which is patched above.  The exception is the build-and-move block at its
    top, which only runs when the CPHF is the first thing to want a cderi;
    done here first, it does not run there.
    """
    if dfobj._cderi is None and FP32.enabled():
        mem_avail = get_avail_mem()
        dfobj.build(omega=omega)
        c = dfobj._cderi[cp.cuda.Device().id]
        # upstream's own policy, `df/hessian/rhf.py:1259-1262`: leave room
        # for the Krylov subspace vectors
        if isinstance(c, cp.ndarray) and c.nbytes > 0.4 * mem_avail:
            _transfer_to_host(dfobj)
    return _orig_hess_get_jk(dfobj, dms, mo_coeff, mo_occ, hermi,
                             with_j, with_k, omega)


DF.loop = loop
DFMOD.DF.loop = loop
DFHESS._get_jk = _get_jk
