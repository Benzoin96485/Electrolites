"""Several GPUs of one node, one Python thread per device.

The expensive DLPNO stages are lists of independent work items (pair or
triplet chunks).  ``run_queue`` hands them to one worker thread per device
from a shared queue (dynamic load balance); the GPU work of a chunk is a few
large batched kernels, so the threads hardly contend for the GIL.  A worker
operates on a *device view* of the driver object: a shallow copy whose
device arrays are replicated on that device once (peer copies over NVLink);
everything else (host arrays, result lists) is shared.

DLPNO_NGPU limits the number of devices used (default: all visible).
"""
import copy
import os
import queue
import threading

import numpy as np
import cupy as cp

_lock = threading.RLock()
_peer = set()


def devices():
    n = cp.cuda.runtime.getDeviceCount()
    env = os.environ.get('DLPNO_NGPU')
    if env:
        n = max(1, min(n, int(env)))
    return list(range(n))


def enable_peer(devs):
    """peer access between all pairs of ``devs`` (idempotent)."""
    for i in devs:
        for j in devs:
            if i == j or (i, j) in _peer:
                continue
            if cp.cuda.runtime.deviceCanAccessPeer(i, j):
                with cp.cuda.Device(i):
                    try:
                        cp.cuda.runtime.deviceEnablePeerAccess(j)
                    except cp.cuda.runtime.CUDARuntimeError:
                        pass            # already enabled
            _peer.add((i, j))


def replicate(x):
    """x (cupy array, or dict / list / tuple of them) on the current device."""
    if isinstance(x, cp.ndarray):
        cur = cp.cuda.Device().id
        if x.device.id == cur:
            return x
        if not x.flags.c_contiguous:
            with cp.cuda.Device(x.device.id):
                x = cp.ascontiguousarray(x)
        return cp.asarray(x)
    if isinstance(x, dict):
        return {k: replicate(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(replicate(v) for v in x)
    return x


def device_view(obj, attrs, sub=None):
    """shallow copy of ``obj`` for the current device: ``attrs`` replicated,
    ``sub`` = {attr: (attrs,...)} nested device views."""
    v = copy.copy(obj)
    for a in attrs:
        if getattr(obj, a, None) is not None:
            setattr(v, a, replicate(getattr(obj, a)))
    for a, at in (sub or {}).items():
        if getattr(obj, a, None) is not None:
            setattr(v, a, device_view(getattr(obj, a), at))
    return v


def run_queue(items, work, devs=None, setup=None):
    """work(ctx, item) for every item, one thread per device pulling items
    from a shared queue (in order).  ``setup(dev)`` builds the per-device
    ctx (e.g. a device view) inside that device's thread.  Exceptions are
    re-raised in the caller.  Returns the per-device ctx list."""
    devs = devices() if devs is None else devs
    if len(devs) == 1:
        with cp.cuda.Device(devs[0]):
            ctx = setup(devs[0]) if setup else None
            for it in items:
                work(ctx, it)
        return [ctx]
    enable_peer(devs)
    q = queue.Queue()
    for it in items:
        q.put(it)
    errs, ctxs = [], [None] * len(devs)
    alloc = cp.cuda.get_allocator()

    def run(r, d):
        try:
            cp.cuda.set_allocator(alloc)
            with cp.cuda.Device(d):
                ctxs[r] = ctx = setup(d) if setup else None
                while not errs:
                    try:
                        it = q.get_nowait()
                    except queue.Empty:
                        break
                    work(ctx, it)
                cp.cuda.Device(d).synchronize()
        except BaseException as e:     # noqa: B902 - re-raised below
            errs.append(e)
    th = [threading.Thread(target=run, args=(r, d)) for r, d in enumerate(devs)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errs:
        raise errs[0]
    return ctxs


def run_per_device(fn, devs):
    """fn(dev) in one thread per device (current device set), concurrently."""
    if len(devs) == 1:
        with cp.cuda.Device(devs[0]):
            fn(devs[0])
        return
    enable_peer(devs)
    errs = []
    alloc = cp.cuda.get_allocator()

    def run(d):
        try:
            cp.cuda.set_allocator(alloc)
            with cp.cuda.Device(d):
                fn(d)
                cp.cuda.Device(d).synchronize()
        except BaseException as e:     # noqa: B902 - re-raised below
            errs.append(e)
    th = [threading.Thread(target=run, args=(d,)) for d in devs]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errs:
        raise errs[0]


def lock():
    return _lock


# ----------------------------------------------------------------------
# one process per GPU (no shared GIL): for stages whose chunks only need
# read-only tables and return small results
# ----------------------------------------------------------------------
def to_host(x):
    """x with every cupy array (also inside dict / list / tuple) on the host."""
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    if isinstance(x, dict):
        return {k: to_host(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_host(v) for v in x)
    return x


def to_device(x):
    """inverse of to_host for numeric numpy arrays (current device)."""
    if isinstance(x, np.ndarray) and x.dtype != object:
        return cp.asarray(x)
    if isinstance(x, dict):
        return {k: to_device(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v) for v in x)
    return x


import pickle as _pickle


class _ArrPickler(_pickle.Pickler):
    """large numpy arrays go to node-local files (shared, memory mapped by
    every worker) instead of the pickle stream."""

    def __init__(self, f, prefix):
        super().__init__(f, protocol=_pickle.HIGHEST_PROTOCOL)
        self.prefix = prefix
        self.files = []

    def persistent_id(self, obj):
        if isinstance(obj, np.ndarray) and obj.dtype != object and obj.nbytes > (1 << 20):
            path = f'{self.prefix}_{len(self.files)}.npy'
            np.save(path, np.ascontiguousarray(obj))
            self.files.append(path)
            return ('npy', path)
        return None


class _ArrUnpickler(_pickle.Unpickler):
    def persistent_load(self, pid):
        kind, path = pid
        return np.load(path, mmap_mode='r')


def _mp_worker(path, dev, tasks, results, setup_name, work_name):
    import os
    import traceback
    try:
        import cupy as cp_
        cp_.cuda.Device(dev).use()
        pool = cp_.get_default_memory_pool()
        cp_.cuda.set_allocator(pool.malloc)
        with open(path, 'rb') as f:
            ctx = _ArrUnpickler(f).load()
        getattr(ctx, setup_name)()
        work = getattr(ctx, work_name)
        while True:
            it = tasks.get()
            if it is None:
                break
            results.put(('ok', work(it)))
    except BaseException:                  # noqa: B902 - reported to the parent
        results.put(('err', f'device {dev}:\n' + traceback.format_exc()))
    finally:
        results.put(('done', dev))


def run_queue_mp(items, ctx, setup_name, work_name, devs, collect):
    """work_name(item) of the (picklable, host-only) ``ctx`` for every item in
    one spawned process per device; ``setup_name()`` runs once per process
    (moving its tables to the device); ``collect(result)`` in the parent."""
    import multiprocessing as mp
    import os
    import pickle
    import tempfile
    sp = mp.get_context('spawn')
    tmpd = '/dev/shm' if os.path.isdir('/dev/shm') else None
    fd, path = tempfile.mkstemp(prefix='dlpno_ctx_', suffix='.pkl', dir=tmpd)
    pk = None
    try:
        with os.fdopen(fd, 'wb') as f:
            pk = _ArrPickler(f, path[:-4])
            pk.dump(ctx)
        if getattr(ctx, 'verbose', 0) >= 2:
            nb = os.path.getsize(path) + sum(os.path.getsize(p) for p in pk.files)
            print(f'[mem] worker context {nb / 1e9:.2f} GB (shared by {len(devs)} processes)', flush=True)
        tasks, results = sp.Queue(), sp.Queue()
        for it in items:
            tasks.put(it)
        for _ in devs:
            tasks.put(None)
        procs = [sp.Process(target=_mp_worker, args=(path, d, tasks, results, setup_name, work_name),
                            daemon=True) for d in devs]
        # spawned children must not re-run the user's main script
        import sys
        main = sys.modules.get('__main__')
        saved = {k: getattr(main, k) for k in ('__file__', '__spec__') if hasattr(main, k)}
        if '__file__' in saved:
            del main.__file__
        main.__spec__ = None
        try:
            for p in procs:
                p.start()
        finally:
            for k, v in saved.items():
                setattr(main, k, v)
        done, err = 0, None
        import queue as _q
        while done < len(devs):
            try:
                kind, val = results.get(timeout=30)
            except _q.Empty:
                if not any(p.is_alive() for p in procs):
                    raise RuntimeError('worker processes died without reporting')
                continue
            if kind == 'ok':
                collect(val)
            elif kind == 'err':
                err = val
            else:
                done += 1
        for p in procs:
            p.join()
        if err:
            raise RuntimeError('worker failed: ' + err)
    finally:
        for p in [path] + (pk.files if pk is not None else []):
            try:
                os.unlink(p)
            except OSError:
                pass
