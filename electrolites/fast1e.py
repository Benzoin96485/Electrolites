"""
Faster three-centre one-electron integrals for GPU4PySCF.

Importing this module replaces ``gpu4pyscf.gto.int3c1e``'s two contracted
drivers -- ``get_int3c1e_density_contracted`` (the electrostatic potential of
the density on a set of points) and ``get_int3c1e_charge_contracted`` (the
Fock-matrix contribution of a set of point charges) -- plus the uncontracted
``get_int3c1e``.  One kernel family serves **PCM, QM/MM, ESP, CHELPG and the
Hessian's hess_nuc_elec**; an IEF-PCM SCF calls two of them every cycle.

What changes is the far field.  For a primitive pair with product exponent
``p`` at ``P`` and a point ``C`` carrying a Gaussian charge of exponent ``q``,
the Boys argument is ``x = a0*r2`` with ``a0 = p*q/(p+q)*theta`` and
``r2 = |P-C|^2``.  Above ``x = 40`` both ``erf(sqrt(x))`` and ``exp(-x)`` are
constant to double precision, and then the whole charge-smearing and
range-separation algebra cancels:

    prefactor * F_0 = 2*pi*sqrt(pi/4) * eij * rsqrt(p^3 * r2)

with no reference to ``q`` or ``theta`` -- far from a smeared charge only its
total charge is visible.  One ``rsqrt`` replaces ``erf``, ``exp``, two square
roots and four fp64 divisions.  On a PCM cavity at GPU4PySCF's default
``lebedev_order = 29`` that branch takes **92 %** of all (primitive pair,
cavity point) tasks, because the cavity wraps the whole molecule while a shell
pair is local to one bond.

``GINTrys_root<NROOTS>`` for two or more roots already had a large-``x``
branch; ``GINTrys_root<1>`` and the six hand-written (ss| and (ps| kernels had
none, and those are half the time.  See ``docs/README_fast1e.md`` for the
derivation, the measured distribution and what the alternative -- screening --
turns out to be worth.

The second change is the launch shape: both upstream drivers hard-code how much
work one thread does, and ``n_pair_sum_per_thread = nao_cart`` leaves the
device most of the way empty.  See ``_pair_work`` below.

Anything this module does not cover -- a short-range operator, an angular
momentum past g, a ``BasisProdCache`` whose layout does not match the one the
kernels were compiled against -- falls through to GPU4PySCF's own driver.
"""
import ctypes
import functools
import os

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.gto import ATOM_OF

from gpu4pyscf.gto import int3c1e as I3
from gpu4pyscf.lib.cupy_helper import cart2sph, get_avail_mem

from ._paths import CUDA_DIR, DATA_DIR

THREADSX = 8          # GPU4PySCF's lib/gint/config.h, and the kernels assume it
THREADSY = 8

_DISABLE = bool(os.environ.get('FAST1E_OFF'))
#: FAST1E_NO_LAUNCH_TUNE=1 keeps GPU4PySCF's own work-per-thread constants,
#: which is the ablation under "What each change buys" in
#: docs/README_fast1e.md.
_TUNE_LAUNCH = not bool(os.environ.get('FAST1E_NO_LAUNCH_TUNE'))


# --------------------------------------------------------------------------
# the module

# Everything below is keyed on the current device as well as on the basis.  A
# module's __device__ table pointers and its __constant__ BasisProdCache are
# per device, so a process that ever touches int3c1e from a second GPU -- one
# inside gpu4pyscf's multi_gpu.run, say -- needs its own compiled module and its
# own copy of the tables there.  GPU4PySCF does not distribute the int3c1e path
# today, so in practice this is device 0; keying on it costs nothing and means
# the module cannot quietly read another device's memory if that changes.

@functools.lru_cache(maxsize=32)
def _module_for(lmax, device_id):
    src = open(os.path.join(CUDA_DIR, 'my1e', 'int3c1e.cu')).read()
    mod = cp.RawModule(code=src, backend='nvrtc',
                       options=('-std=c++17', f'-I{CUDA_DIR}',
                                f'-DE1E_LMAX={lmax}'))
    mod.compile()
    _bind_tables(mod, device_id)
    return mod


def _module(lmax):
    return _module_for(lmax, cp.cuda.Device().id)


@functools.lru_cache(maxsize=8)
def _tables(device_id):
    """The Rys root/weight tables, in the order rys_tables.cuh expects."""
    d = np.load(os.path.join(DATA_DIR, 'rys_roots_dat.npz'))
    names = ('ROOT_SMALLX_R0', 'ROOT_SMALLX_R1', 'ROOT_SMALLX_W0',
             'ROOT_SMALLX_W1', 'ROOT_LARGEX_R_DATA', 'ROOT_LARGEX_W_DATA',
             'ROOT_RW_DATA')
    parts = [np.ascontiguousarray(d[n], dtype=np.float64) for n in names]
    return names, cp.asarray(np.concatenate(parts)), [p.size for p in parts]


def _bind_tables(mod, device_id):
    """Point the module's seven table pointers at one device buffer."""
    names, buf, sizes = _tables(device_id)
    off = 0
    for name, n in zip(names, sizes):
        slot = cp.ndarray((1,), dtype=cp.uint64, memptr=mod.get_global(name))
        slot[0] = buf.data.ptr + off * 8
        off += n


def _function(lmax, name):
    dev = cp.cuda.Device().id
    try:
        return _function_for(lmax, name, dev)
    except Exception:            # a class the lmax subset left out
        return _function_for(4, name, dev)


@functools.lru_cache(maxsize=256)
def _function_for(lmax, name, device_id):
    return _module_for(lmax, device_id).get_function(name)


# --------------------------------------------------------------------------
# GPU4PySCF's BasisProdCache, read from the host and pushed to our __constant__

class _BasisProdCache(ctypes.Structure):
    """Mirror of lib/gint/gint.h's BasisProdCache; see int3c1e_prologue.cuh."""
    _fields_ = [('nbas', ctypes.c_int), ('ncptype', ctypes.c_int),
                ('cptype', ctypes.c_void_p),
                ('bas_pairs_locs', ctypes.c_void_p),
                ('primitive_pairs_locs', ctypes.c_void_p),
                ('bas_pair2shls', ctypes.c_void_p),
                ('aexyz', ctypes.c_void_p),
                ('bas_coords', ctypes.c_void_p), ('bas_atm', ctypes.c_void_p),
                ('bas_pair2bra', ctypes.c_void_p),
                ('bas_pair2ket', ctypes.c_void_p),
                ('ao_loc', ctypes.c_void_p),
                ('a12', ctypes.c_void_p), ('e12', ctypes.c_void_p),
                ('x12', ctypes.c_void_p), ('y12', ctypes.c_void_p),
                ('z12', ctypes.c_void_p),
                ('a1', ctypes.c_void_p), ('a2', ctypes.c_void_p)]


def _bpcache_ok(intopt):
    """Check the struct we are about to copy really is the one we think it is.

    The kernels read c_bpcache field by field, so a GPU4PySCF whose struct grew
    a member would make them read garbage rather than fail.  Two cheap tests
    catch that: nbas must match the sorted molecule, and the first primitive
    pair's product exponent and centre must equal what the molecule says they
    are.  Neither can pass by accident on a shifted layout.
    """
    try:
        c = ctypes.cast(intopt.bpcache, ctypes.POINTER(_BasisProdCache)).contents
        mol = intopt._sorted_mol
        if c.nbas != mol.nbas:
            return False, f'bpcache.nbas {c.nbas} != sorted mol.nbas {mol.nbas}'
        ish = int(intopt.bas_pair2shls[0, 0])
        jsh = int(intopt.bas_pair2shls[1, 0])
        ai = mol.bas_exp(ish)[0]
        aj = mol.bas_exp(jsh)[0]
        A = mol.bas_coord(ish)
        B = mol.bas_coord(jsh)
        aij = ai + aj
        P = (ai * A + aj * B) / aij
        got = _read_device(c.a12, 1)[0], _read_device(c.x12, 1)[0]
        if abs(got[0] - aij) > 1e-9 * max(1.0, aij):
            return False, f'bpcache.a12[0] {got[0]} != ai+aj {aij}'
        if abs(got[1] - P[0]) > 1e-9 * max(1.0, abs(P[0])):
            return False, f'bpcache.x12[0] {got[1]} != Px {P[0]}'
        return True, None
    except Exception as exc:                       # pragma: no cover
        return False, f'{type(exc).__name__}: {exc}'


def _read_device(ptr, n):
    mem = cp.cuda.UnownedMemory(ptr, n * 8, None)
    arr = cp.ndarray((n,), dtype=cp.float64,
                     memptr=cp.cuda.MemoryPointer(mem, 0))
    return cp.asnumpy(arr)


_BPCACHE_SIZE = ctypes.sizeof(_BasisProdCache)


def _push_bpcache(mod, intopt):
    """cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof) -- what the upstream
    launcher does at the top of every GINTfill_int3c1e* call."""
    src = ctypes.cast(intopt.bpcache, ctypes.c_void_p).value
    cp.cuda.runtime.memcpy(mod.get_global('c_bpcache').ptr, src,
                           _BPCACHE_SIZE, cp.cuda.runtime.memcpyHostToDevice)


# --------------------------------------------------------------------------
# launch geometry

def _pair_work(ngrids):
    """``n_pair_sum_per_thread`` for the density-contracted kernel.

    The kernel makes about ``(ntasks_ij / n) * ngrids`` threads and finishes
    with one atomicAdd per thread onto ``out[igrid]``, so ``n`` trades
    parallelism against atomic traffic.  Upstream uses ``nao_cart`` -- one
    thread walks *every* shell pair of its angular type -- which pins the launch
    at ~8*ngrids threads however many pairs there are.  Measured optima on an
    A100 (docs/README_fast1e.md has the sweep):

        ngrids   2428   4721  10963  17142
        n_opt     2-4      8  8..16  16..32

    so the optimum tracks ngrids/700 and saturates.  The curves are flat to
    within ~5 % over a factor of two in n on both A100 variants tested, so this
    is not a knife-edge fit -- but it is a hardware tuning and has not been
    checked on another GPU generation.  FAST1E_NO_LAUNCH_TUNE=1 restores
    upstream's values.
    """
    return min(16, max(2, ngrids // 700))


def _charge_work(ngrids):
    """``n_charge_sum_per_thread``.  Upstream's 10 is already right below about
    10 000 cavity points; above that the optimum drifts to 20-40, worth 3-5 %."""
    return min(40, max(10, ngrids // 500))


def _blocks(nx, ny):
    return ((nx + THREADSX - 1) // THREADSX, (ny + THREADSY - 1) // THREADSY)


# --------------------------------------------------------------------------
# what we decline

def _usable(mol, intopt):
    """Conditions the kernels assume.  Anything else falls through upstream.

    None of these depend on the size of the molecule; they are properties of
    the basis and of the operator asked for.
    """
    if _DISABLE:
        return False, 'FAST1E_OFF'
    if mol.omega < 0:
        return False, 'short-range (erfc) operator'
    if intopt is None or not hasattr(intopt, 'density_offset'):
        return False, 'intopt not built'
    lmax = max(intopt.angular) if len(intopt.angular) else 0
    if lmax > 4:
        return False, f'angular momentum {lmax} > g'
    ok, why = _bpcache_ok(intopt)
    if not ok:
        return False, f'BasisProdCache layout: {why}'
    return True, None


def _lmax(intopt):
    """The angular momenta this basis reaches, which is what gets compiled."""
    if os.environ.get('ELECTROLITES_FULL_MODULE'):
        return 4
    return max(intopt.angular) if len(intopt.angular) else 0


# --------------------------------------------------------------------------
# dispatch, mirroring lib/gint/nr_fill_ao_int3c1e.cu

_CHARGE_FIXED = {0: 'e1e_c00', 10: 'e1e_c10', 11: 'e1e_c11', 20: 'e1e_c20',
                 21: 'e1e_c21', 22: 'e1e_c22', 30: 'e1e_c30', 31: 'e1e_c31',
                 32: 'e1e_c32', 40: 'e1e_c40', 41: 'e1e_c41'}


def _charge_kernel(i_l, j_l):
    name = _CHARGE_FIXED.get(i_l * 10 + j_l)
    if name is not None:
        return name, False
    nroots = (i_l + j_l) // 2 + 1
    if nroots in (4, 5):
        return f'e1e_cg{nroots}', True
    raise NotImplementedError(f'int3c1e charge class ({i_l},{j_l})')


def _plain_kernel(i_l, j_l):
    nroots = (i_l + j_l) // 2 + 1
    if nroots == 1:
        return ('e1e_p00' if (i_l, j_l) == (0, 0) else 'e1e_p10'), False
    if nroots <= 5:
        return f'e1e_pg{nroots}', True
    raise NotImplementedError(f'int3c1e plain class ({i_l},{j_l})')


# --------------------------------------------------------------------------
# the three drivers.  Each mirrors the upstream one in gpu4pyscf/gto/int3c1e.py
# and differs only in the kernel launched and the work-per-thread constant.

def get_int3c1e_charge_contracted(mol, grids, charge_exponents, charges, intopt):
    ok, _why = _usable(mol, intopt)
    if not ok:
        return _ORIG['charge'](mol, grids, charge_exponents, charges, intopt)

    omega = mol.omega
    assert omega >= 0.0
    nao = mol.nao
    ngrids = grids.shape[0]
    assert ngrids > 0
    assert charges.ndim == 1 and charges.shape[0] == grids.shape[0]

    grids = cp.asarray(grids, order='C')
    charges = cp.asarray(charges).astype(np.float64).reshape([-1, 1], order='C')
    grids = cp.concatenate([grids, charges], axis=1)
    if charge_exponents is not None:
        charge_exponents = cp.asarray(charge_exponents, order='C')
        if charge_exponents.size == 1:
            charge_exponents = cp.zeros(ngrids) + charge_exponents

    lmax = _lmax(intopt)
    mod = _module(lmax)
    _push_bpcache(mod, intopt)
    n_charge = _charge_work(ngrids) if _TUNE_LAUNCH else 10

    out = cp.zeros([nao, nao], order='C')
    for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
        cpi = intopt.cp_idx[cp_ij_id]
        cpj = intopt.cp_jdx[cp_ij_id]
        i_l = intopt.angular[cpi]
        j_l = intopt.angular[cpj]
        name, pass_l = _charge_kernel(i_l, j_l)

        i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi + 1]
        j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj + 1]
        ni, nj = i1 - i0, j1 - j0
        sl = cp.zeros([j1 - j0, i1 - i0], order='C')

        ntasks_ij = len(log_q_ij)
        nprim_ij, bas_ij, prim_ij = _pair_block(intopt, cp_ij_id)
        args = [sl, np.int32(ntasks_ij), np.int32(ngrids), np.int32(bas_ij),
                np.int32(prim_ij), np.int32(nprim_ij), np.int32(ni),
                np.int32(ni * nj), np.int32(i0), np.int32(j0),
                np.float64(omega), grids, _ptr_or_null(charge_exponents)]
        if pass_l:
            args += [np.int32(i_l), np.int32(j_l)]
        if ntasks_ij > 0:
            grid_dim = _blocks(ntasks_ij,
                               (ngrids + n_charge - 1) // n_charge)
            _function(lmax, name)(grid_dim, (THREADSX, THREADSY), tuple(args))

        i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi + 1]
        j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj + 1]
        if not mol.cart:
            sl = cart2sph(sl, axis=0, ang=j_l)
            sl = cart2sph(sl, axis=1, ang=i_l)
        out[j0:j1, i0:i1] = sl

    row, col = np.tril_indices(nao)
    out[row, col] = out[col, row]
    return intopt.unsort_orbitals(out, axis=[0, 1])


def get_int3c1e_density_contracted(mol, grids, charge_exponents, dm, intopt):
    ok, _why = _usable(mol, intopt)
    if not ok:
        return _ORIG['density'](mol, grids, charge_exponents, dm, intopt)

    omega = mol.omega
    assert omega >= 0.0
    dm = cp.asarray(dm)
    assert dm.ndim == 2 and dm.shape[0] == dm.shape[1] == mol.nao

    nao_cart = intopt._sorted_mol.nao
    ngrids = grids.shape[0]
    assert ngrids > 0

    dm_pair_ordered = _hermite_density(mol, dm, intopt, nao_cart)

    n_threads_per_block_1d = 8
    n_max_threads_1d = n_threads_per_block_1d * 65535
    n_grid_split = int(np.ceil(ngrids / n_max_threads_1d))
    ngrids_per_split = (ngrids + n_grid_split - 1) // n_grid_split

    grids = cp.asarray(grids, order='C')
    if charge_exponents is not None:
        charge_exponents = cp.asarray(charge_exponents, order='C')
        if charge_exponents.size == 1:
            charge_exponents = cp.zeros(grids.shape[0]) + charge_exponents

    lmax = _lmax(intopt)
    mod = _module(lmax)
    _push_bpcache(mod, intopt)

    out = cp.zeros(ngrids)
    for p0, p1 in lib.prange(0, ngrids, ngrids_per_split):
        n_here = p1 - p0
        n_pair = _pair_work(n_here) if _TUNE_LAUNCH else nao_cart
        gslice = grids[p0:p1, :]
        eslice = _ptr_or_null(charge_exponents, p0, p1)
        for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
            ntasks_ij = len(log_q_ij)
            if ntasks_ij == 0:
                continue
            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            l_sum = intopt.angular[cpi] + intopt.angular[cpj]
            nprim_ij, bas_ij, prim_ij = _pair_block(intopt, cp_ij_id)
            grid_dim = _blocks((ntasks_ij + n_pair - 1) // n_pair, n_here)
            _function(lmax, f'e1e_d{l_sum}')(
                grid_dim, (THREADSX, THREADSY),
                (out[p0:p1], dm_pair_ordered,
                 np.int32(intopt.density_offset[cp_ij_id]),
                 np.int32(intopt.bas_pairs_locs[cp_ij_id]),
                 np.int32(intopt.bas_pairs_locs[cp_ij_id + 1]
                          - intopt.bas_pairs_locs[cp_ij_id]),
                 np.int32(ntasks_ij), np.int32(n_here), np.int32(bas_ij),
                 np.int32(prim_ij), np.int32(nprim_ij),
                 np.float64(omega), gslice, eslice))
    return out


def get_int3c1e(mol, grids, charge_exponents, intopt):
    ok, _why = _usable(mol, intopt)
    if not ok:
        return _ORIG['plain'](mol, grids, charge_exponents, intopt)

    omega = mol.omega
    assert omega >= 0.0
    nao = mol.nao
    ngrids = grids.shape[0]
    assert ngrids > 0

    total_double_number = ngrids * nao * nao
    cp.get_default_memory_pool().free_all_blocks()
    avail_mem = get_avail_mem()
    allowed_double_number = (avail_mem // 4) // 8
    n_grid_split = int(np.ceil(total_double_number / allowed_double_number))
    if n_grid_split > 100:
        raise Exception(
            f'Available GPU memory ({avail_mem / 1e9 : .1f} GB) is too small '
            'for the 3 center integral')
    ngrids_per_split = (ngrids + n_grid_split - 1) // n_grid_split

    buf_size = ngrids * nao * nao
    pinned = cp.cuda.alloc_pinned_memory(buf_size * 8)
    int3c = np.frombuffer(pinned, np.float64, buf_size).reshape(
        [ngrids, nao, nao], order='C')

    grids = cp.asarray(grids, order='C')
    if charge_exponents is not None:
        charge_exponents = cp.asarray(charge_exponents, order='C')
        if charge_exponents.size == 1:
            charge_exponents = cp.zeros(grids.shape[0]) + charge_exponents

    lmax = _lmax(intopt)
    mod = _module(lmax)
    _push_bpcache(mod, intopt)

    for p0, p1 in lib.prange(0, ngrids, ngrids_per_split):
        n_here = p1 - p0
        sl_all = cp.zeros([n_here, nao, nao], order='C')
        gslice = grids[p0:p1]
        eslice = _ptr_or_null(charge_exponents, p0, p1)
        for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
            ntasks_ij = len(log_q_ij)
            if ntasks_ij == 0:
                continue
            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            i_l = intopt.angular[cpi]
            j_l = intopt.angular[cpj]
            name, pass_l = _plain_kernel(i_l, j_l)
            i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi + 1]
            j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj + 1]
            ni, nj = i1 - i0, j1 - j0
            sl = cp.zeros([n_here, j1 - j0, i1 - i0], order='C')
            nprim_ij, bas_ij, prim_ij = _pair_block(intopt, cp_ij_id)
            args = [sl, np.int32(ntasks_ij), np.int32(n_here),
                    np.int32(bas_ij), np.int32(prim_ij), np.int32(nprim_ij),
                    np.int32(ni), np.int32(ni * nj), np.int32(i0),
                    np.int32(j0), np.float64(omega), gslice, eslice]
            if pass_l:
                args += [np.int32(i_l), np.int32(j_l)]
            _function(lmax, name)(_blocks(ntasks_ij, n_here),
                                  (THREADSX, THREADSY), tuple(args))
            i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi + 1]
            j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj + 1]
            if not mol.cart:
                sl = cart2sph(sl, axis=1, ang=j_l)
                sl = cart2sph(sl, axis=2, ang=i_l)
            sl_all[:, j0:j1, i0:i1] = sl
        row, col = np.tril_indices(nao)
        sl_all[:, row, col] = sl_all[:, col, row]
        sl_all = intopt.unsort_orbitals(sl_all, axis=[1, 2])
        sl_all.get(out=int3c[p0:p1, :, :])
    return int3c


# --------------------------------------------------------------------------
# helpers shared with the upstream drivers

def _ptr_or_null(arr, p0=None, p1=None):
    """A device pointer for the kernel, or a null one when there is no array.

    The kernels test ``charge_exponents != NULL`` to decide whether the points
    carry a Gaussian charge, so "no exponents" has to arrive as an actual null
    pointer.  A zero-length CuPy array does not reliably provide one.
    """
    if arr is None:
        return np.uint64(0)
    return arr if p0 is None else arr[p0:p1]


def _pair_blocks(intopt):
    """(nprim_ij, bas_ij, primitive_ij) per angular-pair type.

    ``bas_pairs_locs`` is on the Python side already; ``nprim_12`` and
    ``primitive_pairs_locs`` live only in the C BasisProdCache, so they are read
    once per intopt and cached on it rather than per launch.
    """
    cached = getattr(intopt, '_fast1e_pair_blocks', None)
    if cached is not None:
        return cached
    c = ctypes.cast(intopt.bpcache,
                    ctypes.POINTER(_BasisProdCache)).contents
    cptype = ctypes.cast(
        c.cptype, ctypes.POINTER(ctypes.c_int * (4 * c.ncptype))).contents
    prim_pairs_locs = np.ctypeslib.as_array(
        ctypes.cast(c.primitive_pairs_locs, ctypes.POINTER(ctypes.c_int)),
        (c.ncptype + 1,)).copy()
    bas_pairs_locs = np.asarray(intopt.bas_pairs_locs)
    out = [(int(cptype[4 * i + 2]), int(bas_pairs_locs[i]),
            int(prim_pairs_locs[i])) for i in range(c.ncptype)]
    intopt._fast1e_pair_blocks = out
    return out


def _pair_block(intopt, cp_ij_id):
    return _pair_blocks(intopt)[cp_ij_id]


def _hermite_density(mol, dm, intopt, nao_cart):
    """The density in the Hermite pair order the kernels read.

    This is upstream's own host-side preprocessing, called through GPU4PySCF's
    libgint, not reimplemented: it is not on the hot path (once per call) and
    the pair ordering is the kernels' contract.
    """
    dm = intopt.sort_orbitals(dm, [0, 1])
    if not mol.cart:
        c2s = intopt.cart2sph
        dm = c2s @ dm @ c2s.T
    dm = cp.asnumpy(dm.flatten(order='F'))

    ao_loc_sorted_order = intopt._sorted_mol.ao_loc_nr(cart=True)
    l_ij = intopt.l_ij.T.flatten()
    bas_coords = intopt._sorted_mol.atom_coords()[
        intopt._sorted_mol._bas[:, ATOM_OF]].flatten()
    out = np.empty(intopt.density_offset[-1])
    I3.libgint.GINTinit_J_density_rys_preprocess(
        dm.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(1), ctypes.c_int(nao_cart),
        ctypes.c_int(len(intopt.bas_pairs_locs) - 1),
        intopt.bas_pair2shls.ctypes.data_as(ctypes.c_void_p),
        intopt.bas_pairs_locs.ctypes.data_as(ctypes.c_void_p),
        l_ij.ctypes.data_as(ctypes.c_void_p),
        intopt.density_offset.ctypes.data_as(ctypes.c_void_p),
        ao_loc_sorted_order.ctypes.data_as(ctypes.c_void_p),
        bas_coords.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_bool(True))
    return cp.asarray(out)


# --------------------------------------------------------------------------

_ORIG = {'density': I3.get_int3c1e_density_contracted,
         'charge': I3.get_int3c1e_charge_contracted,
         'plain': I3.get_int3c1e}

I3.get_int3c1e_density_contracted = get_int3c1e_density_contracted
I3.get_int3c1e_charge_contracted = get_int3c1e_charge_contracted
I3.get_int3c1e = get_int3c1e
