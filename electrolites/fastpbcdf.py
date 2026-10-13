"""GDF k-point cderi on the device when it fits (catalogue J1).

GPU4PySCF 1.8.1's ``rsdf_builder.compressed_cderi_kk`` writes the k-point
three-index tensor into *pinned, mapped* host memory (``empty_mapped``, one
``cudaHostAlloc`` per unique k-transfer), and every J/K build reads it back
through ``GDF.loop``: the unpacking kernel (``z_decompress_and_transpose``)
reads the mapped buffer across PCIe, block by block, in every SCF cycle.
Pinning the buffer is itself a large part of the build (1.9 s of 6.2 s for
the 8-atom diamond cell at 2x2x2).

Here, when the whole cderi fits in a third of the free device memory, the
builder allocates it on the device instead -- before it sizes its own work
buffers, so those see the right free memory -- stores each batch with a
device-side slice assignment instead of ``store_col_segment``, and
``GDF.loop`` copies device-to-device (the unpack kernel already takes device
input).  Nothing else changes: the same tensor, the same batches, the same
consumers.  When it does not fit, the stock mapped-memory path runs.
``FASTPBCDF_OFF=1`` keeps the stock path.  Measured in
``docs/KERNEL_TARGETS_1.8.1.md`` §13.29.
"""
import inspect
import os
import textwrap
import types

import cupy as cp

_OFF = bool(os.environ.get('FASTPBCDF_OFF'))
FRACTION = float(os.environ.get('FASTPBCDF_FRACTION', 1 / 3))
STATS = {'device_builds': 0, 'host_builds': 0, 'device_GB': 0.0}


def _prealloc(cd_j2c_cache, nao_pairs, kpt_iters):
    """The cderi dict on the device, or None (then the stock path runs)."""
    from gpu4pyscf.__config__ import num_devices
    total = sum(int(x.shape[1]) for x in cd_j2c_cache) * int(nao_pairs) * 16
    free = cp.cuda.runtime.memGetInfo()[0]
    if _OFF or num_devices != 1 or total > FRACTION * free:
        STATS['host_builds'] += 1
        return None
    out = {}
    for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        naux = cd_j2c_cache[j2c_idx].shape[1]
        out[kp] = cp.empty((naux, nao_pairs), dtype=cp.complex128)
    STATS['device_builds'] += 1
    STATS['device_GB'] = total / 1e9
    return out


def _store(dst, src, p0, p1):
    """Device cderi: dst[:, p0:p1] = src, True.  Host: False (stock call)."""
    if isinstance(dst, cp.ndarray):
        dst[:, p0:p1] = src
        return True
    return False


def _rebind(fn, src):
    mod = inspect.getmodule(fn)
    ns = {}
    exec(compile(src, inspect.getsourcefile(fn), 'exec'), mod.__dict__, ns)
    new = ns[fn.__name__]
    out = types.FunctionType(new.__code__, mod.__dict__, fn.__name__,
                             fn.__defaults__, fn.__closure__)
    out.__kwdefaults__ = fn.__kwdefaults__
    out.__doc__ = fn.__doc__
    return out


def _sub(src, old, new, count=1):
    n = src.count(old)
    if n != count:
        raise RuntimeError(f'fastpbcdf: expected {count} x {old!r}, found {n}')
    return src.replace(old, new)


def _patch_builder(rsdf):
    fn = rsdf.compressed_cderi_kk
    src = textwrap.dedent(inspect.getsource(fn))
    src = _sub(src, '''    Gv = cell.get_Gv(mesh)
    ngrids = len(Gv)

    mem_free = get_avail_mem(exclude_memory_pool=True)''', '''    Gv = cell.get_Gv(mesh)
    ngrids = len(Gv)

    _fastpbcdf_dev = _fastpbcdf_prealloc(cd_j2c_cache, nao_pairs, kpt_iters)
    mem_free = get_avail_mem(exclude_memory_pool=True)''')
    src = _sub(src, '''        cderi[kp] = empty_mapped((naux,nao_pairs), dtype=np.complex128)''',
               '''        cderi[kp] = (_fastpbcdf_dev[kp] if _fastpbcdf_dev is not None
                     else empty_mapped((naux,nao_pairs), dtype=np.complex128))''')
    src = _sub(src, '''err = libvhf_rys.store_col_segment(''',
               '''err = 0 if _fastpbcdf_store(cderi[kp], cderi_k, p0, p1) else libvhf_rys.store_col_segment(''')
    rsdf.__dict__['_fastpbcdf_prealloc'] = _prealloc
    rsdf.__dict__['_fastpbcdf_store'] = _store
    new = _rebind(fn, src)
    rsdf._compressed_cderi_kk_orig = fn
    rsdf.compressed_cderi_kk = new


def _patch_loop(dfmod):
    fn = dfmod.GDF.loop
    src = textwrap.dedent(inspect.getsource(fn))
    src = _sub(src, '''            out.set(tmp)''',
               '''            if isinstance(tmp, cp.ndarray):
                out[...] = tmp
            else:
                out.set(tmp)''')
    new = _rebind(fn, src)
    dfmod.GDF._loop_orig = fn
    dfmod.GDF.loop = new


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    from gpu4pyscf.pbc.df import rsdf_builder, df
    _patch_builder(rsdf_builder)
    _patch_loop(df)
    _PATCHED = True


apply_patch()
