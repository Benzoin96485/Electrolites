"""Checkpoint of a DLPNO run after the PNO integrals (debugging / restarts of
the CC and the triples without redoing the integrals).

``save(obj, path)`` pickles the object with device arrays as host arrays
(restored on the current device), pinned host blocks re-pinned on load, and
the spilled DF tensors' files copied next to the pickle.  Everything goes to
the scratch file system (``dfints.scratch_dir`` refuses the home directory).
"""
import copyreg
import os
import pickle
import shutil

import numpy as np
import cupy as cp


def _cp_restore(x, dev=None):
    """a device array back on the device it was saved from (when present):
    a (Q|iu) spread over the GPUs would not fit one of them."""
    if dev is not None and dev < cp.cuda.runtime.getDeviceCount():
        with cp.cuda.Device(dev):
            return cp.asarray(x)
    return cp.asarray(x)


_SAVE = {}      # save in progress: dir, counter
_LOAD = {}      # load in progress: dir
_FILE_MIN = 64 << 20
_SEG = 1 << 27   # elements per host <-> device slice


def _cp_load_file(name, dev, shape, dtype):
    """a large device array saved to its own file (memory-mapped and copied
    slice by slice: the unpickler's memo would otherwise keep a host copy of
    every device array alive until the end of the load)."""
    fn = os.path.join(_LOAD['dir'], 'arrays', name)
    m = np.load(fn, mmap_mode='r').reshape(-1)
    devc = dev if dev is not None and dev < cp.cuda.runtime.getDeviceCount() else cp.cuda.Device().id
    with cp.cuda.Device(devc):
        out = cp.empty(shape, dtype=np.dtype(dtype))
        flat = out.reshape(-1)
        for i in range(0, m.size, _SEG):
            flat[i:i + _SEG] = cp.asarray(np.ascontiguousarray(m[i:i + _SEG]))
        cp.cuda.Device(devc).synchronize()
    del m
    try:
        fd = os.open(fn, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except (AttributeError, OSError):
        pass
    return out


def _cp_reduce(a):
    if _SAVE and a.nbytes >= _FILE_MIN:
        name = f'dev{_SAVE["n"]}.npy'
        _SAVE['n'] += 1
        m = np.lib.format.open_memmap(os.path.join(_SAVE['dir'], 'arrays', name), mode='w+',
                                      dtype=a.dtype, shape=(a.size,))
        with cp.cuda.Device(a.device.id):
            flat = cp.ascontiguousarray(a).reshape(-1)
            for i in range(0, a.size, _SEG):
                m[i:i + _SEG] = cp.asnumpy(flat[i:i + _SEG])
        m.flush()
        del m
        return _cp_load_file, (name, int(a.device.id), tuple(a.shape), a.dtype.str)
    return _cp_restore, (cp.asnumpy(a), int(a.device.id))


def save(obj, path):
    from electrolites.dlpno.dfints import BlockQ
    os.makedirs(path, exist_ok=True)
    home = os.path.realpath(os.path.expanduser('~'))
    if os.path.realpath(path).startswith(home):
        raise RuntimeError(f'checkpoint {path} is inside the home directory')
    sp = getattr(obj, '_df_spill', None)
    if sp is not None:
        sp.flush()
        shutil.copytree(sp.dir, os.path.join(path, 'spill'), dirs_exist_ok=True)
    disp = copyreg.dispatch_table.copy()
    disp[cp.ndarray] = _cp_reduce
    os.makedirs(os.path.join(path, 'arrays'), exist_ok=True)
    _SAVE.update(dir=path, n=0)
    try:
        with open(os.path.join(path, 'state.pkl'), 'wb') as f:
            p = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
            p.dispatch_table = disp
            p.dump(obj)
    finally:
        _SAVE.clear()
    return path


def _legacy_asarray(x):
    """device arrays of checkpoints written before _cp_restore (their device
    was not recorded): very large ones (the spread (Q|iu) stores) go to the
    device with the most free memory and are moved by their owner (the host
    could not hold them all at once), the rest to the current device."""
    if x.nbytes <= (16 << 30):
        return cp.asarray(x)
    best, bf = None, -1
    for d in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(d):
            f_ = cp.cuda.runtime.memGetInfo()[0]
        if f_ > bf:
            best, bf = d, f_
    with cp.cuda.Device(best):
        return cp.asarray(x)


class _DropCache:
    """file wrapper for the unpickler that drops the pages already read from
    the page cache every few GB (a 160 GB checkpoint read through the cache
    pushed a 256 GB node into the OOM killer)."""

    STEP = 4 << 30

    def __init__(self, f):
        self.f, self.n, self.last = f, 0, 0

    def _tick(self, k):
        self.n += k
        if self.n - self.last > self.STEP:
            try:
                os.posix_fadvise(self.f.fileno(), 0, self.n, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
            self.last = self.n

    def read(self, k=-1):
        b = self.f.read(k)
        self._tick(len(b))
        return b

    def readinto(self, buf):
        k = self.f.readinto(buf)
        self._tick(k or 0)
        return k

    def readline(self, k=-1):
        b = self.f.readline(k)
        self._tick(len(b))
        return b

    def peek(self, k=0):
        return self.f.peek(k)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('cupy') and name == 'asarray':
            return _legacy_asarray
        return super().find_class(module, name)


def load(path):
    from electrolites.dlpno.dfints import BlockQ
    from electrolites.dlpno.devmem import SpillRef, to_pinned
    _LOAD['dir'] = path
    try:
        with open(os.path.join(path, 'state.pkl'), 'rb') as f:
            obj = _Unpickler(_DropCache(f)).load()
    finally:
        _LOAD.clear()
    spill = os.path.join(path, 'spill')

    def fix(x, seen):
        if id(x) in seen:
            return
        seen.add(id(x))
        if isinstance(x, SpillRef):
            x.path = os.path.join(spill, os.path.basename(x.path))
        elif isinstance(x, BlockQ):
            x._tabs = None
            if isinstance(x.buf, np.ndarray):
                x.buf = to_pinned(x.buf)           # zero-copy gathers need pinned memory
            if getattr(x, 'bufs', None):
                # spread storage: each store back where it was placed
                devs = [d for d, _ in getattr(x, 'placed', [])] or [None] * len(x.bufs)
                out = []
                for b, d in zip(x.bufs, devs):
                    if isinstance(b, np.ndarray):
                        if d is None:
                            b = to_pinned(b)
                        else:
                            with cp.cuda.Device(d):
                                b = cp.asarray(b)
                    elif d is not None and b.device.id != d:
                        with cp.cuda.Device(d):
                            b2 = cp.empty_like(b)
                            b2.data.copy_from_device(b.data, b.nbytes)
                        cp.cuda.Device(d).synchronize()
                        b = b2
                    elif d is None:
                        b = to_pinned(b)
                    out.append(b)
                x.bufs = out
                x.buf = x.bufs[0]
        elif isinstance(x, (list, tuple)):
            for y in x:
                fix(y, seen)
        elif isinstance(x, dict):
            for y in x.values():
                fix(y, seen)
        elif hasattr(x, '__dict__') and not isinstance(x, type):
            mod = type(x).__module__ or ''
            if mod.startswith('electrolites'):
                for y in vars(x).values():
                    fix(y, seen)
    fix(obj, set())
    if getattr(obj, '_df_spill', None) is not None:
        obj._df_spill.dir = spill
    return obj
