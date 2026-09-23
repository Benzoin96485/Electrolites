"""Page-locked host memory for the cderi, and the pool that keeps it.

``docs/DF_WORK_LIST.md`` **W12**, and the second half of **W4**.

A cderi that does not fit on the device is written into page-locked host
memory, because ``DF.loop``'s prefetch copies from it asynchronously.
GPU4PySCF allocates it with ``empty_mapped`` -- one ``cudaHostAlloc`` -- which
runs at **2.09 GB/s**: 6.8 s for taxol's 14.31 GB and 41.3 s for PfPMT's
86.41 GB, which is 57-65 % of the whole build.  ``docs/README_fastdfcderi.md``
§5 replaced that with ordinary memory handed to ``cudaHostRegister``, which is
the same mapped, page-locked buffer at **8.6 GB/s** -- it is the allocation
that is slow, not the pinning -- and left W12: 8.6 GB/s is still 10.1 s at
284 atoms, and it is paid **again on every build**.  A geometry optimisation
pays it per step and a range-separated functional pays it twice per SCF,
because ``DF.range_coulomb`` builds a second ``DF`` object.

So the buffer is kept.  A released buffer goes into a module-level pool keyed
by its capacity instead of being unregistered, and the next build of the same
size gets it for nothing.  What decides whether it is kept is host memory:
the pool is capped, and a buffer is only retained while ``MemAvailable``
stays above a reserve -- W12's own falsification is "the pool holds 86 GB of
the node's 251 across a calculation that needs the DRAM for something else",
and that is the case this has to refuse.

This module patches nothing, for the reason ``docs/DF_WORK_LIST.md`` §5.12
gives: ``electrolites/fastdfcderi.py`` needs it and so does anything that
wants to measure it, and importing *that* module patches a build.

    FASTDFCDERI_PIN=alloc      GPU4PySCF's `cudaHostAlloc`, no pool
    FASTDFCDERI_POOL=0         page-lock the cheap way but keep nothing
    FASTDFCDERI_POOL_GB=<x>    cap the pool at x GB of host memory
    FASTDFCDERI_HUGEPAGE=0     do not align to, or ask for, huge pages
"""
import ctypes
import os

import numpy as np
from cupy_backends.cuda.api import runtime
from gpu4pyscf.lib.cupy_helper import empty_mapped

#: how the host-resident cderi's buffer is page-locked.  'register' allocates
#: ordinary memory, advises the kernel to back it with huge pages and pins it
#: with `cudaHostRegister`; 'alloc' is GPU4PySCF's `empty_mapped`, a plain
#: `cudaHostAlloc`.  `docs/README_fastdfcderi.md` §5.
_PIN = os.environ.get('FASTDFCDERI_PIN', 'register')
#: whether the buffer is aligned to a transparent huge page and the kernel
#: advised to back it with them.  Measured on a whole node it is worth
#: **0.3 %** -- the factor comes from not going through `cudaHostAlloc` at
#: all -- and on a shared login GPU it was worth 2.21 -> 2.94 GB/s, so it is
#: kept: one `madvise` against a case where it does help.
_HUGEPAGE = os.environ.get('FASTDFCDERI_HUGEPAGE', '1') != '0'
#: whether a released buffer is kept for the next build (W12).
_POOL = os.environ.get('FASTDFCDERI_POOL', '1') != '0'
#: the pool's cap in GB.  The default is a fraction of the node's memory
#: rather than a constant, because the whole point of the pool is a buffer
#: whose size is the calculation's, not a number chosen in advance.
_POOL_GB = float(os.environ.get('FASTDFCDERI_POOL_GB', 0) or 0)
#: how much of the node's memory the pool will not take, and how much it
#: leaves free.  A retained buffer is already resident and already locked, so
#: `MemAvailable` at the moment of release already counts it: keeping it is
#: refused when that number is under the reserve.
_POOL_FRACTION = 0.6
_RESERVE_FRACTION = 0.1
_RESERVE_MIN = 8 << 30

_PAGE = 4096
#: a transparent huge page on x86-64.  See `_HUGEPAGE` for what it is worth.
_HUGE = 2 * 1024 * 1024
#: ``MADV_HUGEPAGE`` from ``asm-generic/mman-common.h``
_MADV_HUGEPAGE = 14
#: ``cudaHostRegisterMapped``; cupy does not export the enum
_HOST_REGISTER_MAPPED = 2
_libc = ctypes.CDLL('libc.so.6')

#: what this module did, so a claim about it can be read off a run rather
#: than assumed.  `registered_gb` is what was page-locked and `reused_gb`
#: what the pool handed back without page-locking anything.
STATS = {'allocated': 0, 'reused': 0, 'registered_gb': 0., 'reused_gb': 0.,
         'retained': 0, 'released': 0, 'evicted': 0, 'refused_by_memory': 0}
DECLINED = {}


def _decline(reason):
    DECLINED[reason] = DECLINED.get(reason, 0) + 1


def _meminfo(key):
    """One field of ``/proc/meminfo``, in bytes, or ``None``."""
    try:
        with open('/proc/meminfo') as fh:
            for line in fh:
                if line.startswith(key):
                    return int(line.split()[1]) * 1024
    except OSError:                                          # noqa: BLE001
        pass
    return None


def host_avail():
    """``MemAvailable`` in bytes -- what the kernel thinks can still be had."""
    return _meminfo('MemAvailable:')


def _limits():
    total = _meminfo('MemTotal:') or 0
    cap = int(_POOL_GB * 1e9) if _POOL_GB else int(total * _POOL_FRACTION)
    reserve = max(_RESERVE_MIN, int(total * _RESERVE_FRACTION))
    return cap, reserve


# --------------------------------------------------------------------------
# one page-locked buffer
# --------------------------------------------------------------------------
class _Pinned:
    """Ordinary memory, aligned, advised and handed to ``cudaHostRegister``.

    The registration has to be released when the buffer goes, which is what
    this object is for.  It is *not* what the returned array holds: a
    ``_Lease`` is, so that the buffer can outlive the array that used it and
    be handed to the next build (W12).
    """

    def __init__(self, nbytes, hugepage=None):
        hugepage = _HUGEPAGE if hugepage is None else hugepage
        align = _HUGE if hugepage else _PAGE
        self.capacity = int(nbytes)
        self.raw = np.empty(self.capacity + align, dtype=np.uint8)
        base = self.raw.ctypes.data
        self.offset = int((-base) % align)
        self.ptr = base + self.offset
        self.nbytes = (self.capacity + _PAGE - 1) // _PAGE * _PAGE
        if hugepage:
            _libc.madvise(ctypes.c_void_p(self.ptr),
                          ctypes.c_size_t(self.nbytes), _MADV_HUGEPAGE)
        runtime.hostRegister(self.ptr, self.nbytes, _HOST_REGISTER_MAPPED)
        self.registered = True

    def close(self):
        if getattr(self, 'registered', False):
            self.registered = False
            try:
                runtime.hostUnregister(self.ptr)
            except Exception:                                # noqa: BLE001
                pass
            self.raw = None

    def __del__(self):
        self.close()


class _Lease:
    """One use of a ``_Pinned``.  Dropping it offers the buffer to the pool.

    The array the build writes into holds this, and every view of that array
    holds the array, so the offer happens when the last view is dropped --
    including the per-device slices ``_cholesky_eri`` returns.  A lease is
    made fresh on every hand-out, which is why resurrecting a ``_Pinned``
    from ``__del__`` -- where CPython would never call ``__del__`` a second
    time -- is not needed.
    """

    __slots__ = ('pinned',)

    def __init__(self, pinned):
        self.pinned = pinned

    def __del__(self):
        pinned, self.pinned = self.pinned, None
        if pinned is None:
            return
        try:
            _release(pinned)
        except Exception:                                    # noqa: BLE001
            # a lease can be collected during interpreter shutdown, where
            # this module's globals may already be gone; the registration
            # goes with the process in that case
            pass


class _MappedArray(np.ndarray):
    """An ndarray that can carry the lease its buffer depends on."""


# --------------------------------------------------------------------------
# the pool
# --------------------------------------------------------------------------
#: retained buffers, least recently released first
_pool = []


def _pool_bytes():
    return sum(p.capacity for p in _pool)


def _acquire(nbytes):
    """A retained buffer big enough and not wastefully bigger, or ``None``.

    The window is deliberately narrow.  A build that asks for 14 GB must not
    be handed the 86 GB buffer of the last one and then hold it for the rest
    of the process; the point of the pool is the *same* calculation built
    again, which is what a geometry optimisation step and a range-separated
    functional's second ``DF`` object both are, and those ask for exactly the
    same number of bytes.
    """
    best = None
    for p in _pool:
        if nbytes <= p.capacity <= nbytes * 1.25:
            if best is None or p.capacity < best.capacity:
                best = p
    if best is not None:
        _pool.remove(best)
    return best


def _release(pinned):
    """Offer a buffer back to the pool, or unregister it."""
    STATS['released'] += 1
    if not _POOL:
        pinned.close()
        return
    cap, reserve = _limits()
    avail = host_avail()
    if avail is not None and avail < reserve:
        # the node needs the memory more than the next build needs the
        # 10 s -- W12's own falsification criterion
        STATS['refused_by_memory'] += 1
        _decline('host memory under the reserve')
        pinned.close()
        return
    _pool.append(pinned)
    STATS['retained'] += 1
    while _pool and _pool_bytes() > cap:
        STATS['evicted'] += 1
        _pool.pop(0).close()


def free_pool():
    """Unregister and drop everything the pool holds."""
    n = len(_pool)
    while _pool:
        _pool.pop().close()
    return n


# --------------------------------------------------------------------------
# what the build calls
# --------------------------------------------------------------------------
def empty_mapped_pooled(shape, dtype=float, hugepage=None, pool=None):
    """``empty_mapped``, page-locked the cheaper way and kept for the next one.

    Falls back to GPU4PySCF's ``empty_mapped`` if ``cudaHostRegister`` fails
    or if ``FASTDFCDERI_PIN=alloc``; the caller does not have to care which
    it got, because both are mapped and page-locked.
    """
    if _PIN != 'register':
        return empty_mapped(shape, dtype)
    nbytes = int(np.prod(shape)) * int(np.dtype(dtype).itemsize)
    # `pool=False` is what a probe that is *timing the page-locking* passes:
    # with the pool on, the second and later repeats measure a dict lookup.
    # `_POOL` gates the *lookup* as well as the retention, so that an
    # ablation column measuring the unpooled page-locking cannot be handed a
    # buffer some other column left behind -- and then unregister it.
    pinned = (_acquire(nbytes)
              if _POOL and (pool is None or pool) else None)
    if pinned is not None:
        STATS['reused'] += 1
        STATS['reused_gb'] += nbytes * 1e-9
    else:
        try:
            pinned = _Pinned(nbytes, hugepage)
        except Exception as exc:                             # noqa: BLE001
            _decline(f'cudaHostRegister failed ({type(exc).__name__})')
            return empty_mapped(shape, dtype)
        STATS['allocated'] += 1
        STATS['registered_gb'] += nbytes * 1e-9
    out = np.ndarray(shape, dtype=dtype, buffer=pinned.raw.data,
                     offset=pinned.offset).view(_MappedArray)
    out._pin = _Lease(pinned) if pool is not False else _NoPool(pinned)
    return out


class _NoPool:
    """A lease that unregisters rather than retaining.  See ``pool=False``."""

    __slots__ = ('pinned',)

    def __init__(self, pinned):
        self.pinned = pinned

    def __del__(self):
        pinned, self.pinned = self.pinned, None
        if pinned is not None:
            try:
                pinned.close()
            except Exception:                                # noqa: BLE001
                pass


def stats():
    """What this module did in this process.  Nothing numerical can say."""
    out = dict(STATS)
    out['pooled'] = len(_pool)
    out['pooled_gb'] = _pool_bytes() * 1e-9
    out['declined_reasons'] = dict(DECLINED)
    return out
