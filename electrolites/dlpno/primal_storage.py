"""Compact converged GPU primals without retaining energy workspace blocks.

CuPy cannot return a split pool allocation while any of its pieces remains
live. TNO views and small energy tables can therefore pin much larger,
otherwise unused, allocations. Keep the converged values on their original
devices in tightly packed direct allocations, independent of that pool.
"""
import gc

import cupy as cp
import numpy as np


_SEQUENCES = ('X', 'T', 'T1', 'X_tno', 'e_tno', 'W3', 'V3', 'T3')
_MATRICES = ('S_pao_d', 'F_pao_d')


def _compact_primals(base):
    arrays, locations, groups = {}, [], {}

    def keep(value, location):
        if not isinstance(value, cp.ndarray):
            return
        identity = id(value)
        locations.append((location, identity))
        if identity not in arrays:
            arrays[identity] = value
            groups.setdefault((value.device.id, value.dtype.str), []).append(identity)

    for name in _SEQUENCES:
        sequence = getattr(base, name, None)
        if isinstance(sequence, (list, tuple)):
            for position, value in enumerate(sequence):
                keep(value, (name, position))
    for name in _MATRICES:
        keep(getattr(base, name, None), (name, None))

    replacements, buffers = {}, []
    for (device, dtype), identities in groups.items():
        elements = sum(arrays[key].size for key in identities)
        # This is also the direct allocator used by GPU4PySCF for large
        # arrays. It must bypass the default pool even for a small primal.
        with cp.cuda.Device(device), cp.cuda.using_allocator(cp.cuda.memory._malloc):
            buffer = cp.empty(elements, dtype=np.dtype(dtype))
            offset = 0
            for key in identities:
                source = arrays[key]
                target = buffer[offset:offset+source.size].reshape(source.shape)
                target[...] = source
                replacements[key] = target
                offset += source.size
            buffers.append(dict(device=device, dtype=dtype, bytes=buffer.nbytes))
    # Complete copies while every original allocation still has an owner.
    for device in sorted({device for device, dtype in groups}):
        with cp.cuda.Device(device):
            cp.cuda.Device().synchronize()

    tuples = {}
    for (name, position), identity in locations:
        if position is None:
            setattr(base, name, replacements[identity])
        elif isinstance(getattr(base, name), list):
            getattr(base, name)[position] = replacements[identity]
        else:
            if name not in tuples:
                tuples[name] = list(getattr(base, name))
            tuples[name][position] = replacements[identity]
    for name, values in tuples.items():
        setattr(base, name, tuple(values))
    return dict(copied_unique_arrays=len(arrays), copied_array_references=len(locations),
                logical_bytes=sum(value.nbytes for value in arrays.values()), buffers=buffers)


def prepare_gradient_storage(base):
    """Discard reconstructible tables and preserve all converged primals.

    Host amplitudes and supports are untouched. Device-valued primals keep
    their values, shapes, dtypes, devices and exact-object aliases. The
    pair-table and PNO-table caches rebuild through their existing lazy
    accessors if an energy operation is requested again.
    """
    devices = {cp.cuda.Device().id}
    for name in _SEQUENCES:
        values = getattr(base, name, None)
        if isinstance(values, (list, tuple)):
            devices.update(x.device.id for x in values if isinstance(x, cp.ndarray))
    for name in _MATRICES:
        value = getattr(base, name, None)
        if isinstance(value, cp.ndarray):
            devices.add(value.device.id)

    def snapshot(device):
        with cp.cuda.Device(device):
            cp.cuda.Device().synchronize()
            pool = cp.get_default_memory_pool()
            return dict(device=device, pool_used_bytes=pool.used_bytes(),
                        pool_reserved_bytes=pool.total_bytes(),
                        device_free_bytes=int(cp.cuda.runtime.memGetInfo()[0]))

    before = [snapshot(device) for device in sorted(devices)]
    removed = []
    for name in ('_pt', '_pt_src', '_xt'):
        if name in vars(base):
            delattr(base, name)
            removed.append(name)
    # Keep old array references inside the helper, so none survives when
    # the unused split blocks are returned below.
    moved = _compact_primals(base)
    # The discovery loops above can retain their last array or list.
    del value, values
    gc.collect()
    for device in sorted(devices):
        with cp.cuda.Device(device):
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
    after = [snapshot(device) for device in sorted(devices)]
    return dict(storage='compact_direct_cuda_primals', removed_caches=removed,
                before=before, after=after,
                released_pool_bytes=sum(x['pool_reserved_bytes'] for x in before)
                                    -sum(x['pool_reserved_bytes'] for x in after), **moved)
