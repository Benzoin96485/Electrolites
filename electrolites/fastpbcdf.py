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

GPU4PySCF 1.9 (#924) builds the tensor from primitive-pair blocks that a
host thread recontracts into the mapped buffer (``PBCzrecontract_cderi``).
There the device cderi is filled by the same recontraction run on the
device (`_DeviceRecontract`, deterministic: one thread per output element),
for the diffuse-pair part too; the device allocation is decided once the
builder's work-buffer sizes are known and is taken out of its budget.  See
the section below and ``docs/UPGRADE_1.9.md``.
"""
import inspect
import os
import textwrap
import types

import cupy as cp
import numpy as np

_OFF = bool(os.environ.get('FASTPBCDF_OFF'))
FRACTION = float(os.environ.get('FASTPBCDF_FRACTION', 1 / 3))
STATS = {'device_builds': 0, 'host_builds': 0, 'device_GB': 0.0,
         'device_scatter_calls': 0, 'device_dd_calls': 0}


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


# ---------------------------------------------------------------- 1.9.x
# GPU4PySCF 1.9 (#924) rebuilt the k-point store.  The int3c2e kernel now
# works on *primitive* shell pairs; each batch's (pair, aux) block is copied
# to pinned host memory and a host thread scatters it into the contracted
# cderi (``int3c2e._create_pair_recontractor`` -> ``PBCzrecontract_cderi``,
# OpenMP):
#     cderi[a, out_idx[k]] += coef[k] * j3c[i, a],  k in out_offsets[i:i+2]
# with the tensor itself still in mapped host memory that every J/K build
# reads back across PCIe.  Here, when the whole cderi fits, it is allocated
# on the device and the same scatter runs there, on the builder's stream,
# grouped by output column (CSR of the transposed map) so that each output
# element is written once by one thread and the result is deterministic.

_SCATTER_SRC = r'''
extern "C" __global__
void recontract_z(double2 *out, const double2 *j3c, const int *cols,
                  const int *indptr, const int *rows, const double *coef,
                  int naux, long out_ncol)
{
    int u = blockIdx.x;
    int k0 = indptr[u], k1 = indptr[u + 1];
    long col = cols[u];
    for (int a = threadIdx.x + blockIdx.y * blockDim.x; a < naux;
         a += blockDim.x * gridDim.y) {
        double re = 0., im = 0.;
        for (int k = k0; k < k1; k++) {
            double2 v = j3c[(long)rows[k] * naux + a];
            double c = coef[k];
            re += c * v.x;
            im += c * v.y;
        }
        double2 *o = out + (long)a * out_ncol + col;
        o->x += re;
        o->y += im;
    }
}
extern "C" __global__
void recontract_d(double *out, const double *j3c, const int *cols,
                  const int *indptr, const int *rows, const double *coef,
                  int naux, long out_ncol)
{
    int u = blockIdx.x;
    int k0 = indptr[u], k1 = indptr[u + 1];
    long col = cols[u];
    for (int a = threadIdx.x + blockIdx.y * blockDim.x; a < naux;
         a += blockDim.x * gridDim.y) {
        double s = 0.;
        for (int k = k0; k < k1; k++)
            s += coef[k] * j3c[(long)rows[k] * naux + a];
        out[(long)a * out_ncol + col] += s;
    }
}
'''
_scatter_mod = None


def _scatter_kernels():
    global _scatter_mod
    if _scatter_mod is None:
        _scatter_mod = cp.RawModule(code=_SCATTER_SRC)
    return _scatter_mod


class _DeviceRecontract:
    """Upstream's ``recontract(batch_id, cderi, j3c, pair_start=0)`` -- the
    same map -- for a device ``cderi`` (``j3c`` on either side).  A host
    ``cderi`` goes to upstream's own."""

    def __init__(self, recontract):
        self.host = recontract
        self.params = inspect.getclosurevars(
            recontract).nonlocals['recontraction_params']
        self._maps = {}

    def _map(self, batch_id, pair_start, npair):
        key = (batch_id, pair_start, npair)
        m = self._maps.get(key)
        if m is None:
            out_idx, out_offsets, coef = self.params[batch_id]
            offs = np.asarray(out_offsets[pair_start:pair_start + npair + 1])
            k0, k1 = int(offs[0]), int(offs[-1])
            idx = np.asarray(out_idx[k0:k1])
            rows = np.repeat(np.arange(npair, dtype=np.int32), np.diff(offs))
            order = np.lexsort((rows, idx))
            idx, rows = idx[order], rows[order]
            c = np.asarray(coef[k0:k1])[order]
            cols, start = np.unique(idx, return_index=True)
            indptr = np.append(start, len(idx)).astype(np.int32)
            m = (cp.asarray(cols, dtype=np.int32), cp.asarray(indptr),
                 cp.asarray(rows, dtype=np.int32),
                 cp.asarray(c, dtype=np.float64))
            self._maps[key] = m
        return m

    def __call__(self, batch_id, cderi, j3c, pair_start=0):
        if not isinstance(cderi, cp.ndarray):
            return self.host(batch_id, cderi, j3c, pair_start)
        j3c = cp.ascontiguousarray(cp.asarray(j3c))
        assert j3c.dtype == cderi.dtype and cderi.flags.c_contiguous
        npair, naux = j3c.shape
        if npair == 0:
            return cderi
        cols, indptr, rows, coef = self._map(batch_id, pair_start, npair)
        ncols = int(cols.size)
        if ncols == 0:
            return cderi
        kern = _scatter_kernels().get_function(
            'recontract_z' if cderi.dtype == np.complex128 else 'recontract_d')
        threads = 128
        gy = min((naux + threads - 1) // threads, 64)
        kern((ncols, gy), (threads,),
             (cderi, j3c, cols, indptr, rows, coef, np.int32(naux),
              np.int64(cderi.shape[1])))
        STATS['device_scatter_calls'] += 1
        if batch_id == -1:
            STATS['device_dd_calls'] += 1
        return cderi


def _prealloc19(cd_j2c_cache, nao_pairs, kpt_iters, mem_free, work_bytes):
    """The device cderi dict and its size in bytes, or ``(None, 0)``.  It
    must fit in ``FRACTION`` of the free memory and leave the builder's work
    buffers (``work_bytes``, whose sizes are fixed by this point) their room
    in the builder's own budget ``mem_free``."""
    from gpu4pyscf.__config__ import num_devices
    total = sum(int(x.shape[1]) for x in cd_j2c_cache) * int(nao_pairs) * 16
    free = cp.cuda.runtime.memGetInfo()[0]
    if (_OFF or num_devices != 1 or total > FRACTION * free
            or total + work_bytes > 0.8 * mem_free):
        STATS['host_builds'] += 1
        return None, 0
    out = {}
    for j2c_idx, (kp, kp_conj, ki_idx, kj_idx) in enumerate(kpt_iters):
        naux = cd_j2c_cache[j2c_idx].shape[1]
        out[kp] = cp.empty((naux, nao_pairs), dtype=cp.complex128)
    STATS['device_builds'] += 1
    STATS['device_GB'] = total / 1e9
    return out, total


_A19_SIZE = '''    max_pair_size = int(ao_pair_counts.max(initial=0))
'''
_B19_SIZE = '''    max_pair_size = int(ao_pair_counts.max(initial=0))
    _fastpbcdf_dev, _fastpbcdf_bytes = _fastpbcdf_prealloc(
        cd_j2c_cache, nao_pairs, kpt_iters, mem_free,
        max_pair_size*nkpts*naux_cart*16
        + max_pair_size*max(2, bvk_ncells)*naux_cart*8)
    mem_free -= _fastpbcdf_bytes
    _fastpbcdf_recon = _fastpbcdf_recontractor(recontract)
'''
_A19_ALLOC = '''        cderi[kp] = empty_mapped((naux,nao_pairs), dtype=np.complex128)
'''
_B19_ALLOC = '''        cderi[kp] = (_fastpbcdf_dev[kp] if _fastpbcdf_dev is not None
                     else empty_mapped((naux,nao_pairs), dtype=np.complex128))
'''
_A19_STORE = '''                host_j3c = np.ndarray((pair_size, naux), dtype=np.complex128, buffer=write_buf)
                cderi_k.get(out=host_j3c)
'''
_B19_STORE = '''                if _fastpbcdf_dev is not None:
                    _fastpbcdf_recon(batch_id, cderi[kp], cderi_k)
                    continue
                host_j3c = np.ndarray((pair_size, naux), dtype=np.complex128, buffer=write_buf)
                cderi_k.get(out=host_j3c)
'''


def _patch_builder19(rsdf):
    fn = rsdf.compressed_cderi_kk
    src = textwrap.dedent(inspect.getsource(fn))
    src = _sub(src, _A19_SIZE, _B19_SIZE)
    src = _sub(src, _A19_ALLOC, _B19_ALLOC)
    src = _sub(src, _A19_STORE, _B19_STORE)
    # the diffuse (dd) part goes through the same recontractor
    src = _sub(src, 'recontract, uniq_kpts)', '_fastpbcdf_recon, uniq_kpts)')
    rsdf.__dict__['_fastpbcdf_prealloc'] = _prealloc19
    rsdf.__dict__['_fastpbcdf_recontractor'] = _DeviceRecontract
    new = _rebind(fn, src)
    rsdf._compressed_cderi_kk_orig = fn
    rsdf.compressed_cderi_kk = new


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    from gpu4pyscf.pbc.df import rsdf_builder, df
    if 'store_col_segment' in inspect.getsource(
            rsdf_builder.compressed_cderi_kk):        # 1.7 / 1.8
        _patch_builder(rsdf_builder)
    else:                                             # 1.9 (#924)
        _patch_builder19(rsdf_builder)
    _patch_loop(df)
    _PATCHED = True


apply_patch()
