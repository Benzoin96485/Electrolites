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


class SpillRef:
    """a host array kept in a file on the scratch file system (``SpillStore``):
    ``shape``, ``nbytes`` and ``load()``."""
    __slots__ = ('path', 'offset', 'shape', 'dtype')

    def __init__(self, path, offset, shape, dtype):
        self.path, self.offset, self.shape, self.dtype = path, offset, tuple(shape), np.dtype(dtype)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nbytes(self):
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def load(self, threads=8):
        """the array (large ones read by several threads: one stream reads
        the scratch file system at ~0.8 GB/s, eight at ~3.5 GB/s)"""
        n = int(np.prod(self.shape))
        nb = n * self.dtype.itemsize
        if nb < (64 << 20) or threads <= 1:
            return np.fromfile(self.path, dtype=self.dtype, count=n, offset=self.offset).reshape(self.shape)
        import os
        from concurrent.futures import ThreadPoolExecutor
        out = np.empty(n, self.dtype)
        mv = memoryview(out.view(np.uint8))
        seg = -(-nb // threads)
        fd = os.open(self.path, os.O_RDONLY)
        try:
            def rd(k):
                a, b = k * seg, min(nb, (k + 1) * seg)
                while a < b:
                    r = os.preadv(fd, [mv[a:b]], self.offset + a)
                    if r <= 0:
                        raise IOError(f'short read {self.path}')
                    a += r
            with ThreadPoolExecutor(threads) as ex:
                list(ex.map(rd, range(threads)))
        finally:
            os.close(fd)
        return out.reshape(self.shape)


class Prefetcher:
    """Reads a fixed list of SpillRefs over and over, in that order, in
    background threads, ahead of their use and up to ``budget`` bytes held
    (pageable host memory): a consumer that takes the refs in about the
    same order, once per pass (the CC's DF groups every iteration), finds
    them read while the iteration did other work.  ``get(ref)`` returns
    the array (read directly when the ref is not in the list)."""

    def __init__(self, refs, budget, nio=2):
        import threading
        self.refs = list(refs)
        self.pos = {id(r): k for k, r in enumerate(self.refs)}
        self.uses = [0] * len(self.refs)
        # at most one pass ahead
        self.total = float(sum(r.nbytes for r in self.refs))
        self.budget = min(float(budget), self.total)
        self.ready = {}            # absolute index -> array
        self.held = 0.
        self.next = 0              # next absolute index to read
        self.stop = False
        self.wait_s = 0.
        self.cv = threading.Condition()
        self.th = [threading.Thread(target=self._run, daemon=True) for _ in range(nio)]
        for t in self.th:
            t.start()

    def _run(self):
        n = len(self.refs)
        while True:
            with self.cv:
                while not self.stop and self.held > 0 \
                        and self.held + self.refs[self.next % n].nbytes > self.budget:
                    self.cv.wait()
                if self.stop or not n:
                    return
                a = self.next
                self.next += 1
                r = self.refs[a % n]
                self.held += r.nbytes
            try:
                x = r.load()
            except BaseException as e:      # noqa: B902 - handed to the consumer
                x = e
            with self.cv:
                self.ready[a] = x
                self.cv.notify_all()

    def get(self, ref):
        import time
        k = self.pos.get(id(ref))
        if k is None or self.stop:
            return ref.load()
        n = len(self.refs)
        with self.cv:
            a = self.uses[k] * n + k
            self.uses[k] += 1
            t0 = time.perf_counter()
            while a not in self.ready:
                if a < self.next - n * len(self.th) - 1 or not self.cv.wait(timeout=600.):
                    # out of step (never read, or the readers stalled): read it here
                    print(f'[mem] DF prefetch out of step at {a} (next {self.next}): direct read', flush=True)
                    return ref.load()
            self.wait_s += time.perf_counter() - t0
            x = self.ready.pop(a)
            self.held -= ref.nbytes
            self.cv.notify_all()
        if isinstance(x, BaseException):
            raise x
        return x

    def close(self):
        with self.cv:
            self.stop = True
            self.ready.clear()
            self.cv.notify_all()


def as_host(x):
    """numpy array of a (device / host / spilled) array."""
    if isinstance(x, SpillRef):
        return x.load()
    return x if isinstance(x, np.ndarray) else cp.asnumpy(x)


class SpillStore:
    """append-only files (one per writing thread) under a scratch directory
    holding host arrays until they are read back; ``close()`` deletes them."""

    def __init__(self, tag):
        import threading
        from electrolites.dlpno.dfints import scratch_dir
        self.dir = scratch_dir(tag)
        self._f = {}
        self._lock = threading.Lock()
        self.nbytes = 0

    def put(self, x):
        import os
        import threading
        x = np.ascontiguousarray(as_host(x))
        tid = threading.get_ident()
        with self._lock:
            fh = self._f.get(tid)
            if fh is None:
                fh = self._f[tid] = open(os.path.join(self.dir, f'spill_{len(self._f)}.bin'), 'ab')
            self.nbytes += x.nbytes
        off = fh.tell()
        x.tofile(fh)
        return SpillRef(fh.name, off, x.shape, x.dtype)

    def flush(self):
        for fh in self._f.values():
            fh.flush()

    def __getstate__(self):
        self.flush()
        return dict(dir=self.dir, nbytes=self.nbytes)

    def __setstate__(self, st):
        import threading
        self.__dict__.update(st)
        self._f = {}
        self._lock = threading.Lock()
        self.keep = True               # restored from a checkpoint: its files stay

    def close(self):
        import shutil
        for fh in self._f.values():
            fh.close()
        self._f = {}
        if self.dir and not getattr(self, 'keep', False):
            shutil.rmtree(self.dir, ignore_errors=True)
        self.dir = None


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


def mapped_zeros(shape, dtype=np.float64):
    """a device array backed by pinned host memory (zero copy: kernels read
    and write it over the bus) for stores too large for the device; the
    host buffer is kept alive by the array).  Returns the device view and
    the host array."""
    h = pinned_empty(shape, dtype)
    h[...] = 0
    ptr = int(cp.cuda.runtime.pointerGetAttributes(h.ctypes.data).devicePointer)
    mem = cp.cuda.UnownedMemory(ptr, h.nbytes, h)
    a = cp.ndarray(shape, dtype=dtype, memptr=cp.cuda.MemoryPointer(mem, 0))
    return a, h


def to_dev(x):
    if isinstance(x, SpillRef):
        return cp.asarray(x.load())
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
            # device memory in use when the section ended (pool, GB)
            self.mem[k] = cp.get_default_memory_pool().used_bytes() / 1e9
        self._t0 = (key, now) if key else None

    @property
    def mem(self):
        return self.__dict__.setdefault('_mem', {})

    def report(self):
        if self.on and self.t:
            print(f'[prof {self.name}]', ' '.join(f'{k}={v:.2f}s' for k, v in self.t.items()),
                  '| pool used after (GB):', ' '.join(f'{k}={v:.1f}' for k, v in self.mem.items()), flush=True)
            self.t = {}
