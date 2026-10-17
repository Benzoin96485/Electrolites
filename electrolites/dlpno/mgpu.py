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
        if x.device.id == cp.cuda.Device().id:
            return x
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


def lock():
    return _lock
