"""
Drop-in replacement for GPU4PySCF's **fused** Coulomb-and-exchange build.

``scf/jk.py:_VHFOpt.get_jk`` builds J and K from one pass over the Rys
integrals, because J is nearly free once the integrals have been computed for
K.  That is not the path an SCF takes -- ``scf/hf.py:get_veff`` asks for
``mf.get_j`` (the McMurchie-Davidson ``j_engine``) and ``mf.get_k`` (Rys)
separately, which is what `fastj` and `fastk` replace -- but it *is* the path
every response property takes: ``scf/_response_functions.py`` calls
``mf.get_jk``, so a polarizability, an NMR shielding, an IR intensity or a
TDDFT/TDA run of a hybrid spends 61-82 % of its wall clock in that one kernel
and neither of those two modules is on it (``docs/RESPONSE_1.8.1.md`` 3.2).

This module is that kernel, and it is catalogue **B4b**.  It carries the same
P1-P5 treatment as `fastk` -- the Rys roots in registers, the block-uniform
bra data computed once per shell pair, one division and one ``rsqrt`` per
primitive quartet instead of five divisions and a ``sqrt``, per-class
``__launch_bounds__`` -- and the density loop sits inside the shell quartet, so
a call with ``n_dm`` densities pays for the integrals once.  Both ``hermi = 1``
(CPKS, NMR, the Hessian's response) and ``hermi = 0`` (every TDDFT call, where
``jk.py`` stacks the transposed densities and runs the kernel at ``2*nset``)
are served.

Coverage is every angular-momentum class an spdf basis reaches: nineteen with
one thread per shell quartet and six spread over ``threadIdx.y``, both lifted
from ``gvhf-rys/unrolled_rys_jk.cu`` by ``codegen/gen_kernels.py --jk`` and
``codegen/gen_k2_kernels.py --jk``, and forty-five written from the class
itself by ``electrolites/codegen/gen_khigh.py --jk``.  Anything else -- a
short-range (erfc) operator, a shell with ``l > 3`` on the wrong side, the
``h_shls`` CPU path -- falls through to GPU4PySCF's own ``RYS_build_jk``, one
class at a time, so a patched run computes the same matrices as an unpatched
one.

See README_fastjk.md for what the kernels change and what it is worth.
"""
import os, json, math, ctypes, functools
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.gto import ANG_OF

from gpu4pyscf.scf import jk as JK
from gpu4pyscf.lib.cupy_helper import condense, transpose_sum, asarray
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import props as gpu_specs, num_devices

from ._paths import KERNEL_DIR as HERE
from . import _ksplit
from . import _gen
from .fastk import QUEUE_DEPTH, BLOCKS_PER_SM, MAX_PRIM_PAIR, POOL_STRIDE
from .fastk import _tables, _classes, _mpp_plan  # noqa: F401

from .compat import (
    NEW_JK_ABI as _NEW_ABI, sorted_meta as _sorted_meta,
    dense_q_cond as _dense_q_cond, tril_pair_mappings as _tril_pair_mappings,
    resolve_rsh as _resolve_rsh, diffuse_exps as _diffuse_exps)

# Ablations, for the timing tables: hand every class back to GPU4PySCF's fused
# kernel, or only the written / only the lifted half of the family.
_OFF = bool(os.environ.get('FASTJK_OFF'))
_NO_HIGH = bool(os.environ.get('FASTJK_NO_HIGH'))
_NO_LIFTED = bool(os.environ.get('FASTJK_NO_LIFTED'))

_HIGH_SRC = 'fastjkhigh_generated.cu'
_HIGH = (not _NO_HIGH) and _gen.available(_HIGH_SRC)

with open(os.path.join(HERE, 'fastjk_launch.json')) as _f:
    _LAUNCH = json.load(_f)
#: (li,lj,lk,ll) -> (kernel name, threads per block); one thread per quartet
IMPLEMENTED = {} if _NO_LIFTED else {
    tuple(int(c) for c in t): ('jk_' + t, nt) for t, (nt, _mb) in _LAUNCH.items()}

with open(os.path.join(HERE, 'fastjk2_launch.json')) as _f:
    _LAUNCH2 = json.load(_f)
#: (li,lj,lk,ll) -> (tag, launch entry); gout spread over threadIdx.y
IMPLEMENTED2 = {} if _NO_LIFTED else {
    tuple(int(c) for c in t): ('jk2_' + t, e) for t, e in _LAUNCH2.items()}
if _HIGH:
    with open(os.path.join(HERE, 'fastjkhigh_launch.json')) as _f:
        _LAUNCH_H = json.load(_f)
    # added last, so a written kernel wins wherever both have an entry -- the
    # same convention fastk uses, and for the same measured reason
    IMPLEMENTED2.update({tuple(int(c) for c in t): (e['name'], e)
                         for t, e in _LAUNCH_H.items()})


def _sources(wide=False):
    """The CUDA files a module is compiled from.

    ``wide`` is the subset the second, wide-bra-pair module may be built from;
    `fastk._sources` says why that subset is a correctness restriction and not
    a tuning one -- ``fastjk2_generated.cu`` bakes ``double[36]`` in where the
    prologues and the one-thread-per-quartet family use the macro.
    """
    out = [os.path.join(HERE, 'fastk_prologue.cu'),
           os.path.join(HERE, 'fastjk_prologue.cu'),
           _gen.source_path('fastjk_generated.cu')]
    if wide:
        return out
    out.append(_gen.source_path('fastjk2_generated.cu'))
    if _HIGH:
        out.append(_gen.source_path(_HIGH_SRC))
    return out


#: Maximum angular momentum of the basis being built; see fastk._note_lmax.
_LMAX = None


@functools.lru_cache(maxsize=16)
def _module(lmax=None, mpp=MAX_PRIM_PAIR):
    """The kernels for this basis' angular momenta, at this bra-pair bound.

    ``mpp`` above the default asks for `fastk._mpp_plan`'s second module: the
    one-thread-per-quartet family alone, up to the angular momentum a long
    contraction can reach.  See `fastk._module` for why the bound is a
    preprocessor definition and why the second module is a small one.
    """
    files = _sources(wide=mpp != MAX_PRIM_PAIR)
    names = None if lmax is None else _ksplit.names_within(files, lmax)
    opts = ('-std=c++17',) if mpp == MAX_PRIM_PAIR else (
        '-std=c++17', f'-DMAX_PRIM_PAIR={int(mpp)}')
    return cp.RawModule(code=_ksplit.source(files, names),
                        options=opts, backend='nvrtc')


@functools.lru_cache(maxsize=256)
def _function_for(name, lmax, mpp):
    try:
        return _module(lmax, mpp).get_function(name)
    except Exception:                     # a class the lmax subset missed
        return _module(None, mpp).get_function(name)


def _function(name, mpp=MAX_PRIM_PAIR, lmax=None):
    return _function_for(name, _LMAX if lmax is None else lmax, mpp)


class _Workspace:
    """Scratch buffers reused across iterations."""
    def __init__(self):
        self.n_blocks = gpu_specs['multiProcessorCount'] * BLOCKS_PER_SM
        self.pool = cp.empty(self.n_blocks * POOL_STRIDE, dtype=np.int32)
        self.head = cp.zeros(1, dtype=np.int32)
        self.ref_pool = cp.empty(
            gpu_specs['multiProcessorCount'] * JK.QUEUE_DEPTH + 3,
            dtype=np.int32)


@functools.lru_cache(maxsize=1)
def _workspace():
    return _Workspace()


def _usable(omega, nprim, i, j, k, l, mpp=None):
    """Conditions these kernels assume; anything else falls through.

    None of them depends on the size of the molecule -- they are properties of
    the basis and of the operator asked for.  ``n_dm`` is deliberately absent:
    the density loop is inside the quartet, which is the point of the module.
    """
    if omega < 0:                    # the short-range (erfc) operator needs
        return False                 # twice the roots and s_estimator
    # the bra primitive-pair cache is a fixed-size shared array; ``mpp`` is
    # the bound of the module this class block would be launched from
    if int(nprim[i]) * int(nprim[j]) > (MAX_PRIM_PAIR if mpp is None else mpp):
        return False
    return True


_ONE = np.float64(1.0)
_ZERO = np.float64(0.0)


def _prep(self, dms, hermi):
    """Screening data for one _VHFOpt: dm_cond, cutoff, q_cond, pair mappings.

    Identical to what ``jk.py``'s ``get_jk`` computes, so the surviving quartet
    list is the same one GPU4PySCF would have built.
    """
    dm_cond = condense('absmax', dms, self.sorted_mol.ao_loc)
    if hermi == 0:
        dm_cond = dm_cond + dm_cond.T
    dm_cond = cp.log(dm_cond + 1e-300).astype(np.float32)
    log_cutoff = math.log(self.direct_scf_tol)
    q_cond = _dense_q_cond(self)
    _, l_ctr_offsets = _sorted_meta(self)
    pair_mappings = _tril_pair_mappings(
        l_ctr_offsets, q_cond, log_cutoff - float(dm_cond.max()), tile=6)
    return dm_cond, log_cutoff, q_cond, pair_mappings


def _run_jk1(name, nthreads, ws, tab, dev, vj, vk, dms, n_dm, nao, nbas,
             pm_ij, pm_kl, q_cond, dm_cond, log_cutoff, inv_om2,
             mpp=MAX_PRIM_PAIR, mpp_lmax=None):
    """One of the nineteen classes with one thread per shell quartet."""
    d_bas, d_env, d_ao_loc = dev
    kern = _function(name, mpp, mpp_lmax)
    n_blocks = min(ws.n_blocks, int(pm_ij.size))
    for kl0, kl1 in lib.prange(0, int(pm_kl.size), QUEUE_DEPTH):
        ws.head.fill(0)
        kern((n_blocks,), (nthreads,),
             (vj, vk, dms, np.int32(n_dm), np.int32(nao),
              d_bas, d_env, d_ao_loc, np.int32(nbas),
              np.int32(pm_ij.size), np.int32(kl1 - kl0), pm_ij, pm_kl[kl0:],
              q_cond, dm_cond, np.float32(log_cutoff), ws.pool, ws.head,
              np.int32(POOL_STRIDE), tab, inv_om2, _ONE, _ZERO))


def _run_jk2(name, e, prim, ws, tab, dev, vj, vk, dms, n_dm, nao, nbas,
             pm_ij, pm_kl, q_cond, dm_cond, log_cutoff, inv_om2):
    """One of the classes whose gout is spread over threadIdx.y."""
    iprim, jprim, kprim, lprim = prim
    d_bas, d_env, d_ao_loc = dev
    shm = (e['shm_fixed'] + iprim * jprim) * 8
    nthreads = (e['nsq'], e['gout_stride'])
    n_blocks = min(ws.n_blocks, int(pm_ij.size))
    kern = _function(name)
    for kl0, kl1 in lib.prange(0, int(pm_kl.size), QUEUE_DEPTH):
        ws.head.fill(0)
        kern((n_blocks,), nthreads,
             (vj, vk, dms, np.int32(n_dm), np.int32(nao),
              d_bas, d_env, d_ao_loc, np.int32(nbas),
              np.int32(pm_ij.size), np.int32(kl1 - kl0), pm_ij, pm_kl[kl0:],
              q_cond, dm_cond, np.float32(log_cutoff), ws.pool, ws.head,
              np.int32(POOL_STRIDE), tab, np.int32(iprim), np.int32(jprim),
              np.int32(kprim), np.int32(lprim), inv_om2, _ONE, _ZERO),
             shared_mem=shm)


def _run_ref(self, ws, vj, vk, dms, n_dm, nao, shls_slice, ij, kl, dm_cond,
             log_cutoff, dm_penalty, lll):
    """Hand one class back to GPU4PySCF's own RYS_build_jk.

    It screens inside the kernel from the per-group q_cond/s_cond of
    ``_VHFOpt.bas_pair_cache`` plus dm_cond and a density penalty, so this path
    uses GPU4PySCF's own pair lists rather than the tile=6 lists our kernels
    consume.
    """
    mol = self.sorted_mol
    pair_ij, q_cond_ij, s_cond_ij = self.bas_pair_cache[ij]
    pair_kl, q_cond_kl, s_cond_kl = self.bas_pair_cache[kl]
    if pair_ij.size == 0 or pair_kl.size == 0:
        return
    err = JK.libvhf_rys.RYS_build_jk(
        ctypes.cast(vj.data.ptr, ctypes.c_void_p),
        ctypes.cast(vk.data.ptr, ctypes.c_void_p),
        ctypes.cast(dms.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n_dm), ctypes.c_int(nao),
        ctypes.byref(self.rys_envs), (ctypes.c_int*8)(*shls_slice),
        ctypes.c_int(JK.SHM_SIZE),
        ctypes.c_int(pair_ij.size), ctypes.c_int(pair_kl.size),
        ctypes.cast(pair_ij.data.ptr, ctypes.c_void_p),
        ctypes.cast(pair_kl.data.ptr, ctypes.c_void_p),
        ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
        ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
        ctypes.cast(s_cond_ij.data.ptr, ctypes.c_void_p),
        ctypes.cast(s_cond_kl.data.ptr, ctypes.c_void_p),
        ctypes.cast(_diffuse_exps(self).data.ptr, ctypes.c_void_p),
        ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
        ctypes.c_float(log_cutoff),
        ctypes.c_float(dm_penalty),
        ctypes.cast(ws.ref_pool.data.ptr, ctypes.c_void_p),
        mol._bas.ctypes, mol._env.ctypes)
    if err != 0:
        raise RuntimeError(f'RYS_build_jk kernel for {lll} failed')


def get_jk(self, dms, hermi, verbose=None):
    """Build (J, K) for the sorted molecule; see jk._VHFOpt.get_jk."""
    global _LMAX
    if self.sorted_mol is None:
        self.build()
    if callable(dms):
        dms = dms()
    assert num_devices == 1, 'fastjk currently supports a single device'
    mol = self.sorted_mol
    log = logger.new_logger(self.mol, verbose)
    assert dms.ndim == 3 and dms.shape[-1] == mol.ao_loc[-1]
    rsh_omega, lr, sr = _resolve_rsh(self)
    assert rsh_omega >= 0, 'the short-range operator is handled by _dispatch'

    dm_cond, log_cutoff, q_cond, pair_mappings = _prep(self, dms, hermi)
    if hermi == 0:
        # the tril-only quartet enumeration reaches both triangles by being
        # handed the transposed densities as extra ones -- jk.py:443
        dms = cp.vstack([dms, dms.transpose(0, 2, 1)])
    dms = cp.asarray(dms, order='C')
    n_dm, nao = dms.shape[:2]
    vj = cp.zeros(dms.shape)
    vk = cp.zeros(dms.shape)
    dev = self.rys_envs._env_ref_holder[1:]

    uniq_l_ctr, l_ctr_bas_loc = _sorted_meta(self)
    _LMAX = int(mol._bas[:, ANG_OF].max())
    nprim = uniq_l_ctr[:, 1]
    mpp_big, mpp_lmax = _mpp_plan(uniq_l_ctr)
    inv_om2 = np.float64(0.0 if rsh_omega == 0 else 1.0 / rsh_omega**2)
    dm_penalty = float(dm_cond.max())
    ws = _workspace()
    tab = _tables()
    n_ours = n_ref = 0

    for (i, j, k, l), lll in _classes(self):
        pm_ij = pair_mappings[i, j]
        pm_kl = pair_mappings[k, l]
        if pm_ij.size == 0 or pm_kl.size == 0:
            continue
        prim = tuple(int(nprim[x]) for x in (i, j, k, l))
        # a bra pair past the default bound is served from the second module;
        # see fastk.get_k for why only the one-thread-per-quartet classes are
        # in it
        mpp = MAX_PRIM_PAIR if prim[0] * prim[1] <= MAX_PRIM_PAIR else mpp_big
        ok = _usable(rsh_omega, nprim, i, j, k, l, mpp)
        entry2 = IMPLEMENTED2.get(lll) if (ok and mpp == MAX_PRIM_PAIR) else None
        if entry2 is not None:
            name, e = entry2
            _run_jk2(name, e, prim, ws, tab, dev, vj, vk, dms, n_dm, nao,
                     mol.nbas, pm_ij, pm_kl, q_cond, dm_cond, log_cutoff,
                     inv_om2)
            n_ours += 1
            continue
        entry = IMPLEMENTED.get(lll) if ok else None
        if entry is not None:
            name, nthreads = entry
            _run_jk1(name, nthreads, ws, tab, dev, vj, vk, dms, n_dm, nao,
                     mol.nbas, pm_ij, pm_kl, q_cond, dm_cond, log_cutoff,
                     inv_om2, mpp,
                     None if mpp == MAX_PRIM_PAIR else mpp_lmax)
            n_ours += 1
            continue
        shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
        _run_ref(self, ws, vj, vk, dms, n_dm, nao, shls_slice, (i, j), (k, l),
                 dm_cond, log_cutoff, dm_penalty, lll)
        n_ref += 1
    log.debug1('fastjk: %d classes here, %d handed back', n_ours, n_ref)

    if hermi == 1:
        vj *= 2.
        vk = transpose_sum(vk)
    elif hermi == 2:
        vj[:] = 0
        vk = transpose_sum(vk, hermi=2)
    else:
        vj, vjT = vj[:n_dm//2], vj[n_dm//2:]
        vj += vjT.transpose(0, 2, 1)
        vk, vkT = vk[:n_dm//2], vk[n_dm//2:]
        vk += vkT.transpose(0, 2, 1)
    return transpose_sum(vj), vk


_PATCHED = False


def _get_jk_dispatch(self, dms, hermi, verbose=None):
    """Use our fused kernels where they apply, otherwise GPU4PySCF's get_jk."""
    if _OFF or self.h_shls or num_devices > 1:
        return JK._VHFOpt._get_jk_orig(self, dms, hermi, verbose)
    omega = _resolve_rsh(self)[0]
    if omega < 0:                        # the short-range operator
        return JK._VHFOpt._get_jk_orig(self, dms, hermi, verbose)
    return get_jk(self, dms, hermi, verbose)


def apply_patch():
    global _PATCHED
    if not _PATCHED:
        JK._VHFOpt._get_jk_orig = JK._VHFOpt.get_jk
        JK._VHFOpt.get_jk = _get_jk_dispatch
        _PATCHED = True


apply_patch()
