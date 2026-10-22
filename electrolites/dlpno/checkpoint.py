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


def _cp_reduce(a):
    return cp.asarray, (cp.asnumpy(a),)


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


def load(path):
    from electrolites.dlpno.dfints import BlockQ
    from electrolites.dlpno.devmem import SpillRef, to_pinned
    with open(os.path.join(path, 'state.pkl'), 'rb') as f:
        obj = pickle.load(f)
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
                x.bufs = [to_pinned(b) if isinstance(b, np.ndarray) else b for b in x.bufs]
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
