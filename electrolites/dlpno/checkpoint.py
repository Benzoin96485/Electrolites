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


def _cp_reduce(a):
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
    with open(os.path.join(path, 'state.pkl'), 'wb') as f:
        p = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
        p.dispatch_table = disp
        p.dump(obj)
    return path


def _legacy_asarray(x):
    """device arrays of checkpoints written before _cp_restore: large ones
    stay on the host until their owner places them (the spread (Q|iu))."""
    return x if x.nbytes > (16 << 30) else cp.asarray(x)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('cupy') and name == 'asarray':
            return _legacy_asarray
        return super().find_class(module, name)


def load(path):
    from electrolites.dlpno.dfints import BlockQ
    from electrolites.dlpno.devmem import SpillRef, to_pinned
    with open(os.path.join(path, 'state.pkl'), 'rb') as f:
        obj = _Unpickler(f).load()
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
