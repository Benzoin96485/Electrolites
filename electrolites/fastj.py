"""
Drop-in replacement for the CUDA kernels of GPU4PySCF's McMurchie-Davidson
Coulomb (J) engine.

Importing this module monkey-patches
``gpu4pyscf.scf.j_engine._VHFOpt.get_j`` so that the angular-momentum classes
built by ``gen_j_kernels.py`` are handled by our kernels; every other class or
option falls through to GPU4PySCF's own ``MD_build_j``.  The screening, the
task/tile decomposition, the Hermite density transformation and the symmetry
factors are GPU4PySCF's own and are untouched, and the integral arithmetic is
lifted verbatim out of its generated kernels, so the J matrix agrees with
GPU4PySCF's to round-off.

See README_fastj.md for what the kernels change and why.
"""
import os, json, math, ctypes, functools
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.gto import ANG_OF, PTR_EXP

from gpu4pyscf.scf import j_engine as JE
from gpu4pyscf.lib.cupy_helper import condense, transpose_sum, hermi_triu, asarray
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.__config__ import num_devices
from .compat import NEW_JK_ABI as _NEW_ABI, dense_q_cond as _dense_q_cond

from ._paths import KERNEL_DIR as HERE
PTR_RANGE_OMEGA = 8
# FASTJ_TIME=1 prints the host/transfer vs kernel split of each J build
TIME = int(os.environ.get('FASTJ_TIME', 0))
_T = {'prep': 0., 'kern': 0., 'post': 0., 'n': 0}

with open(os.path.join(HERE, 'fastj_launch.json')) as _f:
    LAUNCH = json.load(_f)


@functools.lru_cache(maxsize=1)
def _module():
    gen = os.environ.get('FASTJ_SRC', 'fastj_generated.cu')
    src = (open(os.path.join(HERE, 'fastj_prologue.cu')).read() +
           open(os.path.join(HERE, gen)).read())
    opts = ['-std=c++17']
    if os.environ.get('FASTJ_LINEINFO'):
        opts.append('-lineinfo')
    return cp.RawModule(code=src, options=tuple(opts), backend='nvrtc')


@functools.lru_cache(maxsize=32)
def _function(name):
    return _module().get_function(name)


def _qd_offset_for_threads(npairs, threads):
    """Offset of the `threads`-wide tile-max block inside qd_*_max.

    Mirrors qd_offset_for_threads() in gvhf-md/md_j_driver.cu.
    """
    npairs_aligned = (npairs + 31) & 0xffffffe0
    address = 0
    i = 1
    while i < threads:
        address += npairs_aligned
        npairs_aligned //= 2
        i *= 2
    return address


def get_j_181(self, dms, verbose=None):
    """Build the J matrix on GPU4PySCF 1.8.x (see j_engine._VHFOpt.get_j).

    1.8.0 folded the primitive molecule into the option object's own
    ``sorted_mol`` (``SortedGTO.from_mol(..., decontract=True)``, and
    ``j_engine._cache_q_cond_and_non0pairs`` asserts one primitive per shell),
    so ``prim_mol`` and ``prim_to_ctr_mapping`` are gone.  It also moved the
    Hermite transform to the GPU -- ``_dm_to_Rt`` / ``_Rt_to_dm`` with
    ``rys_envs`` instead of the host-side ``Et_dot_dm`` / ``jengine_dot_Et``
    -- and made ``MD_build_j`` take the per-group compact ``q_cond_ij`` /
    ``q_cond_kl`` rather than one global matrix.

    Our kernels are unchanged: they index a dense ``nbas x nbas`` q_cond,
    which ``g4pcompat.dense_q_cond`` rebuilds from the per-group cache, and
    the tile-max hierarchy ``_make_tile_max_hierarchy`` writes has the same
    layout ``_qd_offset_for_threads`` assumes.
    """
    assert num_devices == 1, 'fastj currently supports a single device'
    log = logger.new_logger(self.mol, verbose)
    mol = self.sorted_mol
    assert mol.nbas < 65536
    if callable(dms):
        dms = dms()
    ao_loc = mol.ao_loc
    n_dm, nao = dms.shape[:2]
    assert dms.ndim == 3 and nao == ao_loc[-1]
    dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
    log_cutoff = math.log(self.direct_scf_tol)

    l_counts = np.bincount(mol._bas[:, ANG_OF])[:JE.LMAX+1]
    n_groups = len(l_counts)
    l_ctr_bas_loc = np.cumsum(np.append(0, l_counts))
    l_symb = lib.param.ANGULAR
    pair_mappings = JE._make_pair_qd_cond(mol, self.bas_pair_cache, dm_cond)
    dm_cond = None

    pair_lst = []
    task_offsets = {}
    p0 = p1 = 0
    for i in range(n_groups):
        for j in range(i+1):
            pair_ij_mapping = pair_mappings[i, j][0]
            pair_lst.append(pair_ij_mapping)
            p0, p1 = p1, p1 + pair_ij_mapping.size
            task_offsets[i, j] = p0
    pair_lst = cp.asarray(cp.hstack(pair_lst), dtype=np.int32)

    ls = cp.asarray(mol._bas[:, ANG_OF], dtype=np.int32)
    ll = (ls[:, None] + ls).ravel()[pair_lst]
    xyz_size = (ll+1)*(ll+2)*(ll+3)//6
    pair_loc = cp.cumsum(cp.append(cp.zeros(1, dtype=np.int32), xyz_size.ravel()),
                         dtype=np.int32)
    xyz_size = ls = ll = None
    dm_xyz_size = int(pair_loc[-1])

    _env = JE._scale_sp_ctr_coeff(mol)
    dm_xyz = JE._dm_to_Rt(mol, dms, pair_lst, pair_loc, self.rys_envs)
    vj_xyz = cp.zeros_like(dm_xyz)

    tasks = [(i, j, k, l)
             for i in range(n_groups)
             for j in range(i+1)
             for k in range(i+1)
             for l in range(k+1)
             if not (i == k and j < l)]
    schemes = {t: JE._md_j_engine_quartets_scheme(t, n_dm=n_dm) for t in tasks}

    q_cond = _dense_q_cond(self)
    rys_envs = self.rys_envs
    d_bas, d_env = rys_envs._env_ref_holder[1], rys_envs._env_ref_holder[2]
    kern_ref = JE.libvhf_md.MD_build_j
    use_ours = (n_dm == 1 and float(_env[PTR_RANGE_OMEGA]) == 0.)

    for i, j, k, l in tasks:
        shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
        pair_ij_mapping, q_cond_ij, qd_ij = pair_mappings[i, j]
        pair_kl_mapping, q_cond_kl, qd_kl = pair_mappings[k, l]
        npairs_ij = pair_ij_mapping.size
        npairs_kl = pair_kl_mapping.size
        if npairs_ij == 0 or npairs_kl == 0:
            continue
        pair_ij_loc = pair_loc[task_offsets[i, j]:]
        pair_kl_loc = pair_loc[task_offsets[k, l]:]
        tag = f'{i+j}_{k+l}'
        entry = LAUNCH.get(tag) if use_ours else None
        if entry is not None:
            bx, by = entry['bsizex'], entry['bsizey']
            off_ij = _qd_offset_for_threads(npairs_ij, 16)
            off_kl = _qd_offset_for_threads(npairs_kl, 16)
            grid = ((npairs_ij + bx - 1) // bx, (npairs_kl + by - 1) // by)
            kern = _function('j_' + tag)
            kern(grid, (16, 16),
                 (vj_xyz, dm_xyz, d_bas, d_env, np.int32(mol.nbas),
                  np.int32(npairs_ij), np.int32(npairs_kl),
                  pair_ij_mapping, pair_kl_mapping, pair_ij_loc, pair_kl_loc,
                  qd_ij[off_ij:], qd_kl[off_kl:], q_cond,
                  np.float32(log_cutoff)),
                 shared_mem=entry['shm'] * 8)
            continue
        err = kern_ref(
            ctypes.cast(vj_xyz.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
            ctypes.byref(rys_envs), (ctypes.c_int*6)(*schemes[i, j, k, l]),
            (ctypes.c_int*8)(*shls_slice),
            ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
            ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
            ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
            ctypes.cast(pair_ij_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(pair_kl_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(qd_ij.data.ptr, ctypes.c_void_p),
            ctypes.cast(qd_kl.data.ptr, ctypes.c_void_p),
            ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
            ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff),
            mol._atm.ctypes, ctypes.c_int(mol.natm),
            mol._bas.ctypes, ctypes.c_int(mol.nbas), _env.ctypes)
        if err != 0:
            llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
            raise RuntimeError(f'MD_build_j kernel for {llll} failed')

    if self.h_shls:
        raise NotImplementedError('fastj does not handle the CPU high-l path')
    vj = JE._Rt_to_dm(mol, vj_xyz.get(), pair_lst, pair_loc, self.rys_envs)
    vj *= 2.
    log.timer_debug1('get_j (fastj)')
    return vj


def get_j(self, dms, verbose=None):
    """Build the J matrix (see j_engine._VHFOpt.get_j)."""
    assert num_devices == 1, 'fastj currently supports a single device'
    if TIME:
        import time as _time
        cp.cuda.Stream.null.synchronize(); _t0 = _time.perf_counter()
    log = logger.new_logger(self.mol, verbose)
    sorted_mol = self.sorted_mol
    prim_mol = self.prim_mol
    assert prim_mol.nbas < 65536
    if callable(dms):
        dms = dms()
    p2c_mapping = cp.asarray(self.prim_to_ctr_mapping)
    ao_loc = sorted_mol.ao_loc
    n_dm, nao = dms.shape[:2]
    assert dms.ndim == 3 and nao == ao_loc[-1]
    dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
    log_max_dm = float(dm_cond.max())
    log_cutoff = math.log(self.direct_scf_tol)
    q_cutoff = log_cutoff - log_max_dm
    dm_cond = dm_cond[p2c_mapping[:, None], p2c_mapping]

    l_counts = np.bincount(prim_mol._bas[:, ANG_OF])[:JE.LMAX+1]
    n_groups = len(l_counts)
    l_ctr_bas_loc = np.cumsum(np.append(0, l_counts))
    l_symb = lib.param.ANGULAR
    q_cond = self.q_cond
    pair_mappings = JE._make_pair_qd_cond(
        prim_mol, l_ctr_bas_loc, q_cond, dm_cond, q_cutoff)
    dm_cond = None

    pair_lst = []
    task_offsets = {}
    p0 = p1 = 0
    for i in range(n_groups):
        for j in range(i+1):
            pair_ij_mapping = pair_mappings[i, j][0]
            pair_lst.append(pair_ij_mapping)
            p0, p1 = p1, p1 + pair_ij_mapping.size
            task_offsets[i, j] = p0
    pair_lst = cp.asarray(cp.hstack(pair_lst), dtype=np.int32)

    ls = cp.asarray(prim_mol._bas[:, ANG_OF], dtype=np.int32)
    ll = (ls[:, None] + ls).ravel()[pair_lst]
    xyz_size = (ll+1)*(ll+2)*(ll+3)//6
    pair_loc_gpu = cp.cumsum(cp.append(np.int32(0), xyz_size.ravel()), dtype=np.int32)
    xyz_size = ls = ll = None

    pair_lst = np.asarray(pair_lst.get(), dtype=np.int32)
    pair_loc = pair_loc_gpu.get()
    dm_xyz_size = pair_loc[-1]
    dms = dms.get()
    dm_xyz = np.zeros((n_dm, dm_xyz_size))
    _env = JE._scale_sp_ctr_coeff(prim_mol)
    JE.libvhf_md.Et_dot_dm(
        dm_xyz.ctypes, dms.ctypes,
        ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
        ao_loc.ctypes, pair_loc.ctypes,
        pair_lst.ctypes, ctypes.c_int(len(pair_lst)),
        self.prim_to_ctr_mapping.ctypes,
        ctypes.c_int(prim_mol.nbas), ctypes.c_int(sorted_mol.nbas),
        prim_mol._bas.ctypes, _env.ctypes)

    tasks = [(i, j, k, l)
             for i in range(n_groups)
             for j in range(i+1)
             for k in range(i+1)
             for l in range(k+1)
             if not (i == k and j < l)]
    schemes = {t: JE._md_j_engine_quartets_scheme(t, n_dm=n_dm) for t in tasks}

    if TIME:
        cp.cuda.Stream.null.synchronize()
        _T['prep'] += _time.perf_counter() - _t0; _t0 = _time.perf_counter()
    dm_xyz = asarray(dm_xyz)
    vj_xyz = cp.zeros_like(dm_xyz)
    pair_loc_gpu = cp.asarray(pair_loc_gpu)
    q_cond = cp.asarray(self.q_cond)
    rys_envs = self.rys_envs
    d_atm, d_bas, d_env = (rys_envs._env_ref_holder[0], rys_envs._env_ref_holder[1],
                           rys_envs._env_ref_holder[2])
    kern_ref = JE.libvhf_md.MD_build_j
    use_ours = (n_dm == 1 and float(_env[PTR_RANGE_OMEGA]) == 0.)

    for i, j, k, l in tasks:
        shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
        pair_ij_mapping, qd_ij = pair_mappings[i, j]
        pair_kl_mapping, qd_kl = pair_mappings[k, l]
        npairs_ij = pair_ij_mapping.size
        npairs_kl = pair_kl_mapping.size
        if npairs_ij == 0 or npairs_kl == 0:
            continue
        pair_ij_loc = pair_loc_gpu[task_offsets[i, j]:]
        pair_kl_loc = pair_loc_gpu[task_offsets[k, l]:]
        tag = f'{i+j}_{k+l}'
        entry = LAUNCH.get(tag) if use_ours else None
        if entry is not None:
            bx, by = entry['bsizex'], entry['bsizey']
            # GPU4PySCF's kernels all use a 16x16 thread block, and the
            # tile-max hierarchy is indexed at that granularity.
            off_ij = _qd_offset_for_threads(npairs_ij, 16)
            off_kl = _qd_offset_for_threads(npairs_kl, 16)
            grid = ((npairs_ij + bx - 1) // bx, (npairs_kl + by - 1) // by)
            kern = _function('j_' + tag)
            kern(grid, (16, 16),
                 (vj_xyz, dm_xyz, d_bas, d_env, np.int32(prim_mol.nbas),
                  np.int32(npairs_ij), np.int32(npairs_kl),
                  pair_ij_mapping, pair_kl_mapping, pair_ij_loc, pair_kl_loc,
                  qd_ij[off_ij:], qd_kl[off_kl:], q_cond,
                  np.float32(log_cutoff)),
                 shared_mem=entry['shm'] * 8)
            continue
        err = kern_ref(
            ctypes.cast(vj_xyz.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
            ctypes.byref(rys_envs), (ctypes.c_int*6)(*schemes[i, j, k, l]),
            (ctypes.c_int*8)(*shls_slice),
            ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
            ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
            ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
            ctypes.cast(pair_ij_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(pair_kl_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(qd_ij.data.ptr, ctypes.c_void_p),
            ctypes.cast(qd_kl.data.ptr, ctypes.c_void_p),
            ctypes.cast(q_cond.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff),
            prim_mol._atm.ctypes, ctypes.c_int(prim_mol.natm),
            prim_mol._bas.ctypes, ctypes.c_int(prim_mol.nbas), _env.ctypes)
        if err != 0:
            llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
            raise RuntimeError(f'MD_build_j kernel for {llll} failed')

    if TIME:
        cp.cuda.Stream.null.synchronize()
        _T['kern'] += _time.perf_counter() - _t0; _t0 = _time.perf_counter()
    vj_xyz = vj_xyz.get()
    h_shls = self.h_shls
    if h_shls:
        raise NotImplementedError('fastj does not handle the CPU high-l path')
    vj, dms = dms, None
    vj[:] = 0.
    JE.libvhf_md.jengine_dot_Et(
        vj.ctypes, vj_xyz.ctypes,
        ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
        ao_loc.ctypes, pair_loc.ctypes,
        pair_lst.ctypes, ctypes.c_int(len(pair_lst)),
        self.prim_to_ctr_mapping.ctypes,
        ctypes.c_int(prim_mol.nbas), ctypes.c_int(sorted_mol.nbas),
        prim_mol._bas.ctypes, _env.ctypes)
    vj = transpose_sum(asarray(vj))
    vj *= 2.
    if TIME:
        cp.cuda.Stream.null.synchronize()
        _T['post'] += _time.perf_counter() - _t0; _T['n'] += 1
        print(f'[fastj] call {_T["n"]:2d}  prep(host+H2D) {_T["prep"]:6.2f} s'
              f'  kernels {_T["kern"]:6.2f} s  post(D2H+host) {_T["post"]:6.2f} s',
              flush=True)
    log.timer_debug1('get_j (fastj)')
    return vj


_PATCHED = False


def _get_j_dispatch(self, dms, verbose=None):
    if self.h_shls or num_devices > 1:
        return JE._VHFOpt._get_j_orig(self, dms, verbose)
    if _NEW_ABI:
        return get_j_181(self, dms, verbose)
    return get_j(self, dms, verbose)


def apply_patch():
    global _PATCHED
    if not _PATCHED:
        JE._VHFOpt._get_j_orig = JE._VHFOpt.get_j
        JE._VHFOpt.get_j = _get_j_dispatch
        _PATCHED = True


apply_patch()
