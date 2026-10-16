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


def pinned_empty(shape, dtype=np.float64):
    """numpy array in page-locked host memory (fast, asynchronous H2D)."""
    n = int(np.prod(shape))
    mem = cp.cuda.alloc_pinned_memory(max(n, 1) * np.dtype(dtype).itemsize)
    return np.frombuffer(mem, dtype, n).reshape(shape)


def to_pinned(x):
    out = pinned_empty(x.shape, x.dtype)
    if isinstance(x, cp.ndarray):
        x.get(out=out)
    else:
        out[...] = x
    return out


def to_dev(x):
    return x if isinstance(x, cp.ndarray) else cp.asarray(x)
