"""Faster nuclear-gradient two-electron build for GPU4PySCF.

Importing this module replaces ``gpu4pyscf.grad.rhf._jk_energy_per_atom`` --
the per-atom derivative of ``j*J - k*K``, which is 75 % of a B3LYP/6-31G*
gradient and 96 % of a wB97M-V/def2-TZVPD one on the PfPMT and HcgC clusters.
Anything the generated kernels do not cover falls through to GPU4PySCF's own
``RYS_per_atom_jk_ip1`` for that angular-momentum group, so the result is the
same gradient either way.

``gen_ejk.py`` writes the kernels; ``fastejk_prologue.cu`` holds the shared
scaffolding and says what changed against GPU4PySCF.
"""
import os, math, json, ctypes, functools
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.gto import ANG_OF
from gpu4pyscf.scf import jk as JK
from gpu4pyscf.grad import rhf as grhf
from gpu4pyscf.lib.cupy_helper import condense
from gpu4pyscf.__config__ import props as gpu_specs
from .compat import (
    NEW_JK_ABI as _NEW_ABI, sorted_meta as _sorted_meta,
    dense_q_cond as _dense_q_cond, tril_pair_mappings as _tril_pair_mappings,
    diffuse_exps as _diffuse_exps)

from ._paths import KERNEL_DIR as HERE
from . import _ksplit
from . import _gen
QUEUE_DEPTH = int(os.environ.get('FASTEJK_QUEUE_DEPTH', 1 << 16))
# a block may run one thread-block of tasks past ntasks with a zero density
# factor, and pads its queue there, so each slot needs a block's worth of slack
POOL_STRIDE = QUEUE_DEPTH + 1024
BLOCKS_PER_SM = int(os.environ.get('FASTEJK_BLOCKS_PER_SM', 4))
MAX_PRIM_PAIR = 36              # must match fastejk_prologue.cu

_SRC = os.environ.get('FASTEJK_SRC', 'fastejk_generated.cu')
_TAB = os.path.join(HERE, os.path.splitext(_SRC)[0].replace('_generated', '')
                    + '_launch.json')
_OFF = set(x for x in os.environ.get('FASTEJK_SKIP', '').split(',') if x)
IMPLEMENTED = {}
if os.path.exists(_TAB) and _gen.available(_SRC):
    IMPLEMENTED = {t: e for t, e in json.load(open(_TAB)).items()
                   if t not in _OFF}

_ZERO = np.float64(0.0)
_ONE = np.float64(1.0)


#: Maximum angular momentum of the basis currently being built; see
#: fastk._LMAX.  This module is the largest generated file here -- 140 000
#: lines for 65 classes -- so it is where compiling only what the basis
#: reaches is worth the most.
_LMAX = None


def _sources():
    return [os.path.join(HERE, 'fastejk_prologue.cu'), _gen.source_path(_SRC)]


@functools.lru_cache(maxsize=8)
def _module(lmax=None):
    files = _sources()
    names = None if lmax is None else _ksplit.names_within(files, lmax)
    return cp.RawModule(code=_ksplit.source(files, names),
                        options=('-std=c++17',), backend='nvrtc')


@functools.lru_cache(maxsize=256)
def _function_for(name, shm, lmax):
    try:
        f = _module(lmax).get_function(name)
    except Exception:                     # a class the lmax subset missed
        f = _module(None).get_function(name)
    if shm > 48 * 1024 - 2688:      # opt in to the A100's large shared window
        f.max_dynamic_shared_size_bytes = shm
    return f


def _function(name, shm=0):
    return _function_for(name, shm, _LMAX)


@functools.lru_cache(maxsize=1)
def _tables():
    """Rys root/weight tables, packed as fastejk_prologue.cu expects."""
    d = np.load(os.path.join(HERE, 'data', 'rys_roots_dat.npz'))
    return cp.asarray(np.concatenate([
        d['ROOT_SMALLX_R0'], d['ROOT_SMALLX_R1'], d['ROOT_SMALLX_W0'],
        d['ROOT_SMALLX_W1'], d['ROOT_LARGEX_R_DATA'], d['ROOT_LARGEX_W_DATA'],
        d['ROOT_RW_DATA']]), dtype=np.float64)


class _Workspace:
    def __init__(self):
        self.n_blocks = gpu_specs['multiProcessorCount'] * BLOCKS_PER_SM
        self.pool = cp.empty(self.n_blocks * POOL_STRIDE, dtype=np.int32)
        self.head = cp.zeros(1, dtype=np.int32)
        self.ref_pool = cp.empty(
            gpu_specs['multiProcessorCount'] * JK.QUEUE_DEPTH + 1,
            dtype=np.int32)
        self.dd_pool = cp.empty(
            (gpu_specs['multiProcessorCount'], grhf.DD_CACHE_MAX),
            dtype=np.float64)


@functools.lru_cache(maxsize=1)
def _workspace():
    return _Workspace()


def _usable(mol, n_dm, nprim, i, j, k, l, tag):
    """Conditions the kernels assume.  None depends on the size of the
    molecule; they are properties of the basis and of what the caller asked."""
    if tag not in IMPLEMENTED:
        return False
    if mol.omega < 0:               # short-range (erfc) needs twice the roots
        return False                # and the s_estimator screening
    if n_dm != 1:                   # closed-shell only
        return False
    if int(nprim[i]) * int(nprim[j]) > MAX_PRIM_PAIR:
        return False
    return True


def _run(tag, ws, tab, dev, ejk, dms, nao, nbas, jf, kf, pm_ij, pm_kl,
         q_cond, dm_cond, log_cutoff, prim, inv_om2, coef0):
    e = IMPLEMENTED[tag]
    d_bas, d_env, d_ao_loc = dev
    iprim, jprim, kprim, lprim = prim
    nthreads = (e['nsq'], e['gout_stride'])
    shm = e['shm_fixed'] * 8
    n_blocks = min(ws.n_blocks, int(pm_ij.size))
    kern = _function(e['name'], shm)
    for kl0, kl1 in lib.prange(0, int(pm_kl.size), QUEUE_DEPTH):
        ws.head.fill(0)
        kern((n_blocks,), nthreads,
             (ejk, dms, np.int32(nao), np.float64(jf), np.float64(kf),
              d_bas, d_env, d_ao_loc, np.int32(nbas),
              np.int32(pm_ij.size), np.int32(kl1 - kl0), pm_ij, pm_kl[kl0:],
              q_cond, dm_cond, np.float32(log_cutoff), ws.pool, ws.head,
              np.int32(POOL_STRIDE), tab, np.int32(iprim), np.int32(jprim),
              np.int32(kprim), np.int32(lprim), inv_om2, coef0),
             shared_mem=shm)


def _run_ref_new(vhfopt, ws, ejk, dms, n_dm, nao, j_factor, k_factor,
                 shls_slice, scheme, ij, kl, dm_cond, log_cutoff):
    """GPU4PySCF 1.8.x's own kernel, for the groups we hand back.

    1.8.x screens inside the kernel from the per-group q_cond/s_cond of
    _VHFOpt.bas_pair_cache plus dm_cond and a density penalty (which its own
    driver leaves at 0 here), so this path uses GPU4PySCF's pair lists rather
    than the tile=6 lists our kernels consume.
    """
    smol = vhfopt.sorted_mol
    pair_ij, q_cond_ij, s_cond_ij = vhfopt.bas_pair_cache[ij]
    pair_kl, q_cond_kl, s_cond_kl = vhfopt.bas_pair_cache[kl]
    if pair_ij.size == 0 or pair_kl.size == 0:
        return
    err = JK.libvhf_rys.RYS_per_atom_jk_ip1(
        ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
        ctypes.c_double(j_factor), ctypes.c_double(k_factor),
        ctypes.cast(dms.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n_dm), ctypes.c_int(nao),
        vhfopt.rys_envs, (ctypes.c_int*2)(*scheme),
        (ctypes.c_int*8)(*shls_slice),
        ctypes.c_int(pair_ij.size), ctypes.c_int(pair_kl.size),
        ctypes.cast(pair_ij.data.ptr, ctypes.c_void_p),
        ctypes.cast(pair_kl.data.ptr, ctypes.c_void_p),
        ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
        ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
        ctypes.cast(s_cond_ij.data.ptr, ctypes.c_void_p),
        ctypes.cast(s_cond_kl.data.ptr, ctypes.c_void_p),
        ctypes.cast(_diffuse_exps(vhfopt).data.ptr, ctypes.c_void_p),
        ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
        ctypes.c_float(log_cutoff),
        ctypes.c_float(0.),
        ctypes.cast(ws.ref_pool.data.ptr, ctypes.c_void_p),
        ctypes.cast(ws.dd_pool.data.ptr, ctypes.c_void_p),
        smol._atm.ctypes, ctypes.c_int(smol.natm),
        smol._bas.ctypes, ctypes.c_int(smol.nbas), smol._env.ctypes)
    if err != 0:
        raise RuntimeError('RYS_per_atom_jk_ip1 failed')


def _run_ref(vhfopt, ws, ejk, dms, n_dm, nao, j_factor, k_factor, shls_slice,
             scheme, pm_ij, pm_kl, q_cond, dm_cond, log_cutoff):
    """GPU4PySCF's own kernel, for the groups we hand back."""
    smol = vhfopt.sorted_mol
    s_ptr = lib.c_null_ptr()
    if smol.omega < 0:
        s_ptr = ctypes.cast(vhfopt.s_estimator.data.ptr, ctypes.c_void_p)
    for kl0, kl1 in lib.prange(0, int(pm_kl.size), JK.QUEUE_DEPTH):
        err = JK.libvhf_rys.RYS_per_atom_jk_ip1(
            ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
            ctypes.c_double(j_factor), ctypes.c_double(k_factor),
            ctypes.cast(dms.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_dm), ctypes.c_int(nao),
            vhfopt.rys_envs, (ctypes.c_int*2)(*scheme),
            (ctypes.c_int*8)(*shls_slice),
            ctypes.c_int(pm_ij.size), ctypes.c_int(kl1 - kl0),
            ctypes.cast(pm_ij.data.ptr, ctypes.c_void_p),
            ctypes.cast(pm_kl[kl0:].data.ptr, ctypes.c_void_p),
            ctypes.cast(q_cond.data.ptr, ctypes.c_void_p), s_ptr,
            ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff),
            ctypes.cast(ws.ref_pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(ws.dd_pool.data.ptr, ctypes.c_void_p),
            smol._atm.ctypes, ctypes.c_int(smol.natm),
            smol._bas.ctypes, ctypes.c_int(smol.nbas), smol._env.ctypes)
        if err != 0:
            raise RuntimeError('RYS_per_atom_jk_ip1 failed')


def jk_energy_per_atom(vhfopt, dm, j_factor=1., k_factor=1., verbose=None):
    """Drop-in replacement for gpu4pyscf.grad.rhf._jk_energy_per_atom."""
    smol = vhfopt.sorted_mol
    dm = cp.asarray(dm, order='C')
    dms = vhfopt.apply_coeff_C_mat_CT(
        dm.reshape(-1, vhfopt.mol.nao, vhfopt.mol.nao))
    dms = cp.asarray(dms, order='C')
    n_dm, nao = dms.shape[:2]
    ao_loc = smol.ao_loc
    uniq_l_ctr, l_ctr_bas_loc = _sorted_meta(vhfopt)
    global _LMAX
    _LMAX = int(vhfopt.sorted_mol._bas[:, ANG_OF].max())
    uniq_l = uniq_l_ctr[:, 0]
    nprim = uniq_l_ctr[:, 1]
    n_groups = len(uniq_l_ctr)

    dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
    log_cutoff = math.log(vhfopt.direct_scf_tol)
    q_cond = _dense_q_cond(vhfopt)
    pair_mappings = _tril_pair_mappings(
        l_ctr_bas_loc, q_cond, log_cutoff - float(dm_cond.max()), tile=6)

    ws = _workspace()
    tab = _tables()
    dev = vhfopt.rys_envs._env_ref_holder[1:]
    ejk = cp.zeros((smol.natm, 3))
    # the kernel receives GPU4PySCF's scaled factors: the 8-fold permutation
    # symmetry and the 1/2 of the Coulomb operator give 4 in the J contraction,
    # and the K contraction builds (dm_jk*dm_il + dm_jl*dm_ik), a further 1/2
    jf = 4. * j_factor
    kf = -k_factor if n_dm == 1 else -2. * k_factor
    omega = smol.omega
    inv_om2 = np.float64(0.0 if omega <= 0 else 1.0 / omega**2)

    for i in range(n_groups):
        for j in range(i + 1):
            for k in range(i + 1):
                for l in range(k + 1):
                    pm_ij = pair_mappings[i, j]
                    pm_kl = pair_mappings[k, l]
                    if pm_ij.size == 0 or pm_kl.size == 0:
                        continue
                    lll = (int(uniq_l[i]), int(uniq_l[j]),
                           int(uniq_l[k]), int(uniq_l[l]))
                    tag = '%d%d%d%d' % lll
                    prim = (int(nprim[i]), int(nprim[j]),
                            int(nprim[k]), int(nprim[l]))
                    if _usable(smol, n_dm, nprim, i, j, k, l, tag):
                        _run(tag, ws, tab, dev, ejk, dms, nao, smol.nbas,
                             jf, kf, pm_ij, pm_kl, q_cond, dm_cond,
                             log_cutoff, prim, inv_om2, _ONE)
                    else:
                        shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1,
                                                    k, k+1, l, l+1]]
                        scheme = grhf._ejk_quartets_scheme(
                            smol, uniq_l_ctr[[i, j, k, l]])
                        if _NEW_ABI:
                            _run_ref_new(vhfopt, ws, ejk, dms, n_dm, nao,
                                         j_factor, k_factor, shls_slice,
                                         scheme, (i, j), (k, l), dm_cond,
                                         log_cutoff)
                        else:
                            _run_ref(vhfopt, ws, ejk, dms, n_dm, nao,
                                     j_factor, k_factor, shls_slice, scheme,
                                     pm_ij, pm_kl, q_cond, dm_cond, log_cutoff)
    return ejk.get()


_PATCHED = False


def apply_patch():
    global _PATCHED
    if not _PATCHED and IMPLEMENTED:
        grhf._jk_energy_per_atom_orig = grhf._jk_energy_per_atom
        grhf._jk_energy_per_atom = jk_energy_per_atom
        _PATCHED = True


apply_patch()
