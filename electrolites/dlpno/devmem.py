"""Device memory policy for the DLPNO drivers.

GPU4PySCF installs an allocator that sends every request above 100 MB
straight to cudaMalloc / cudaFree (synchronising, ~10 ms each).  The DLPNO
iterations allocate and release large batched temporaries many times per
iteration, so inside them the plain CuPy memory pool is used instead.
"""
import contextlib

import numpy as np

import cupy as cp


@contextlib.contextmanager
def pooled():
    pool = cp.get_default_memory_pool()
    prev = cp.cuda.get_allocator()
    cp.cuda.set_allocator(pool.malloc)
    try:
        yield pool
    finally:
        cp.cuda.set_allocator(prev)
        pool.free_all_blocks()


def free_pinned():
    """return the cached free pinned blocks to the OS.  GPU4PySCF installs its
    own pinned pool (``gpu4pyscf.lib.cupy_helper.pinned_memory_pool``) as the
    allocator, so freeing CuPy's default pool alone releases nothing."""
    cp.get_default_pinned_memory_pool().free_all_blocks()
    try:
        from gpu4pyscf.lib import cupy_helper
        cupy_helper.pinned_memory_pool.free_all_blocks()
    except (ImportError, AttributeError):
        pass


def trim_host():
    """cached host memory back to the OS: the free pinned blocks and the C
    heap's freed blocks (numpy temporaries)."""
    free_pinned()
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except (OSError, AttributeError):
        pass


def pinned_empty(shape, dtype=np.float64):
    """numpy array in page-locked host memory (fast, asynchronous H2D)."""
    n = int(np.prod(shape))
    mem = cp.cuda.alloc_pinned_memory(max(n, 1) * np.dtype(dtype).itemsize)
    return np.frombuffer(mem, dtype, n).reshape(shape)


def to_pinned(x):
    out = pinned_empty(x.shape, x.dtype)
    if isinstance(x, cp.ndarray):
        cp.ascontiguousarray(x).get(out=out)
    else:
        out[...] = x
    return out


def to_dev(x):
    return x if isinstance(x, cp.ndarray) else cp.asarray(x)


class Prof:
    """section timer, active with DLPNO_PROF=1 (synchronises the device at
    every section boundary)."""

    def __init__(self, name):
        import os
        self.on = os.environ.get('DLPNO_PROF') == '1'
        self.name = name
        self.t = {}
        self._t0 = None

    def __call__(self, key):
        if not self.on:
            return
        import time
        cp.cuda.Device().synchronize()
        now = time.perf_counter()
        if self._t0 is not None:
            k, t0 = self._t0
            self.t[k] = self.t.get(k, 0.) + now - t0
        self._t0 = (key, now) if key else None

    def report(self):
        if self.on and self.t:
            print(f'[prof {self.name}]', ' '.join(f'{k}={v:.2f}s' for k, v in self.t.items()), flush=True)
            self.t = {}
