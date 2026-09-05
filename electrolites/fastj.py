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
from . import _ksplit
from . import _gen
PTR_RANGE_OMEGA = 8
# FASTJ_TIME=1 prints the host/transfer vs kernel split of each J build
TIME = int(os.environ.get('FASTJ_TIME', 0))
_T = {'prep': 0., 'kern': 0., 'post': 0., 'n': 0}

with open(os.path.join(HERE, 'fastj_launch.json')) as _f:
    LAUNCH = json.load(_f)

# The four classes GPU4PySCF does not unroll -- (3,3), (4,2), (4,3), (4,4) --
# are written from the angular-momentum class alone by codegen/gen_j_high.py
# rather than lifted, because there is no upstream kernel for them to lift.
# FASTJ_NO_HIGH=1 hands them back, which is the ablation for that generator.
_HIGH = {}
if not int(os.environ.get('FASTJ_NO_HIGH', 0)):
    with open(os.path.join(HERE, 'fastjhigh_launch.json')) as _f:
        _HIGH = json.load(_f)
    LAUNCH.update(_HIGH)

# The multi-density family (catalogue item B4a).  GPU4PySCF's MD_build_j takes
# at most DM_BLOCK = 4 densities per launch -- `scf/j_engine.py` pins n_dm to 4
# inside `_md_j_engine_quartets_scheme` and the driver steps `dm_offset += 4`
# -- so a CPHF, TDDFT or polarizability build with 126 densities runs the whole
# kernel 32 times over, Hermite Rt recurrence and all, although the Rt tensor
# does not depend on the density at all.  Measured on PfPMT/6-31G* with
# `benchmarks/ndm_j.py`: a four-density pass costs 5.08 s of which only 0.40 s
# per density is the contraction, so roughly 70 % of a large-n_dm J build is
# work the cap makes GPU4PySCF repeat.
#
# These kernels carry the density dimension *inside* the quartet loop instead:
# the Rt tensor is built once and contracted with every density in the block,
# which also turns each shared-memory Rt load into NDM fused multiply-adds
# rather than one.  `gen_j_high.py --ndm` writes one per (class, block size).
#
# **What ships is a block of four, which is GPU4PySCF's own**, and the reason
# is the most useful thing this round measured.  That 70 % is GPU4PySCF's
# per-quartet overhead (the catalogue's 3.6 P1-P5: fp64 divisions, the
# incomplete-gamma values staged through shared memory, block-uniform data
# recomputed per thread), which the cap makes it pay once per pass -- not the
# Rt recurrence.  These kernels do not pay it at all, so the same fit on them
# gives R/C = 0.37 rather than 9.65, and removing the cap altogether is worth
# 1.09x, not 3.4x.  Blocks of 8 and 16 were written and measured at 0.90x and
# 0.64x against blocks of 4: the vj_ij accumulators they need cost more
# occupancy than the recurrence they save.  docs/ROUND3_B4A.md 4 has the
# tables; `gen_j_high.py --ndm 4,8,16` with FASTJ_MDM_BLOCKS reproduces them.
#
# What the shipped kernels are worth is 1.23x on the whole J build at
# n_dm = 8, 16 and 32, and 1.57x at n_dm = 4 -- the kernel's own margin,
# carried into a build GPU4PySCF used to have to itself.
#
# FASTJ_NO_MDM=1 hands every multi-density build back, which is the ablation.
_MDM = {}
#: class tag -> the density-block sizes that class has a kernel for, largest
#: first, so the greedy split below takes the biggest amortisation available.
_MDM_BLOCKS = {}
if not int(os.environ.get('FASTJ_NO_MDM', 0)):
    with open(os.environ.get('FASTJ_MDM_TABLE',
                             os.path.join(HERE, 'fastjmdm_launch.json'))) as _f:
        _MDM = json.load(_f)
    # FASTJ_MDM_BLOCKS restricts which density-block sizes are used, without
    # regenerating anything.  It is the ablation that separates "our kernel is
    # faster per density" (which n_dm = 4 already shows, since that is
    # GPU4PySCF's own block) from "amortising the Rt recurrence over more than
    # four densities pays", which is the B4a claim proper.
    _only = os.environ.get('FASTJ_MDM_BLOCKS')
    _only = {int(x) for x in _only.split(',')} if _only else None
    for _k in _MDM:
        _n, _tag = _k.split('_', 1)
        if _only is None or int(_n[2:]) in _only:
            _MDM_BLOCKS.setdefault(_tag, []).append(int(_n[2:]))
    for _v in _MDM_BLOCKS.values():
        _v.sort(reverse=True)


def _sources(family='single'):
    """The CUDA files one kernel family is compiled from.

    The multi-density family is a **separate** module, compiled only when a
    build with more than one density asks for it.  Folding it into the
    single-density module would make every ordinary SCF pay its NVRTC cost --
    which is the whole point of `_ksplit` and of docs/ROUND2_1.8.1.md 4,
    where
    a 6-31G* job's first run went from 162 s to 37 s by compiling only what
    the basis reaches.
    """
    if family == 'mdm':
        return [os.path.join(HERE, 'fastj_prologue.cu'),
                _gen.source_path(os.environ.get(
                    'FASTJ_MDM_SRC', 'fastjmdm_generated.cu'))]
    gen = os.environ.get('FASTJ_SRC', 'fastj_generated.cu')
    out = [os.path.join(HERE, 'fastj_prologue.cu'), _gen.source_path(gen)]
    if _HIGH:
        out.append(_gen.source_path(os.environ.get(
            'FASTJ_HIGH_SRC', 'fastjhigh_generated.cu')))
    return out


@functools.lru_cache(maxsize=8)
def _module(lmax=None, family='single', names=None):
    """The kernels a basis of this ``lmax`` can reach, and only those.

    ``lmax`` of ``None`` compiles the whole family, which is what a caller
    that cannot name its basis gets.  ``names`` overrides the ``lmax`` rule
    with an explicit set, which is how the multi-density family is kept cheap:
    a build knows its task list *and* how it will split ``n_dm``, so it can
    name the (class, block size) pairs it will launch instead of every pair
    the basis could in principle reach -- three block sizes over twenty-seven
    classes is 32 MB of CUDA, and NVRTC charges by the line.
    """
    files = _sources(family)
    if names is None and lmax is not None:
        names = _ksplit.names_within(files, lmax)
    opts = ['-std=c++17']
    if os.environ.get('FASTJ_LINEINFO'):
        opts.append('-lineinfo')
    return cp.RawModule(code=_ksplit.source(files, names),
                        options=tuple(opts), backend='nvrtc')


@functools.lru_cache(maxsize=256)
def _function(name, lmax=None, family='single', names=None):
    try:
        f = _module(lmax, family, names).get_function(name)
    except Exception:                     # a class the subset missed
        f = _module(None, family).get_function(name)
    # The written high-l classes carry the whole Rt tensor plus the ket vj
    # accumulators in shared memory, which is past the 48 KB a kernel gets
    # without asking.
    table = _MDM if family == 'mdm' else LAUNCH
    shm = table.get(name[2:], {}).get('shm', 0) * 8
    if shm > 48 * 1024:
        f.max_dynamic_shared_size_bytes = shm
    return f


def _dm_split(tag, n_dm):
    """How to cover ``n_dm`` densities for one class.

    Returns ``([(block size, offset), ...], remainder)``: our kernels take as
    many densities as their largest block sizes cover, and whatever is left --
    always fewer than the smallest block, so at most three densities -- goes
    to GPU4PySCF, which handles small ``n_dm`` with its own specialisations.
    """
    blocks = _MDM_BLOCKS.get(tag)
    if not blocks:
        return [], n_dm
    plan = []
    off = 0
    for b in blocks:
        while n_dm - off >= b:
            plan.append((b, off))
            off += b
    return plan, n_dm - off


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


class _JSetup:
    """Everything ``MD_build_j`` (or our kernels) needs, built once.

    Split out of :func:`get_j_181` so that the per-class benchmark can drive
    exactly the same task list, screening data and Hermite density as a real
    build does, instead of an approximation of it.
    """
    __slots__ = ('opt', 'mol', 'n_dm', 'dm_xyz', 'dm_xyz_size', 'pair_lst',
                 'pair_loc', 'task_offsets', 'pair_mappings', 'tasks',
                 'schemes', 'q_cond', 'l_ctr_bas_loc', 'log_cutoff', '_env',
                 'd_bas', 'd_env', 'use_ours', 'lmax', '_scheme_cache',
                 'mdm_names')

    def scheme(self, task, n_dm):
        """GPU4PySCF's shared-memory scheme for this task at this ``n_dm``.

        A fall-through remainder is built with fewer densities than the whole
        request, and the scheme depends on that count, so it cannot be the one
        cached for the full ``n_dm``.
        """
        hit = self._scheme_cache.get((task, n_dm))
        if hit is None:
            hit = JE._md_j_engine_quartets_scheme(task, n_dm=n_dm)
            self._scheme_cache[task, n_dm] = hit
        return hit


def _setup_181(self, dms):
    """Build the task list, screening data and Hermite density for a J build."""
    mol = self.sorted_mol
    assert mol.nbas < 65536
    ao_loc = mol.ao_loc
    n_dm, nao = dms.shape[:2]
    assert dms.ndim == 3 and nao == ao_loc[-1]
    dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
    log_cutoff = math.log(self.direct_scf_tol)

    l_counts = np.bincount(mol._bas[:, ANG_OF])[:JE.LMAX+1]
    n_groups = len(l_counts)
    l_ctr_bas_loc = np.cumsum(np.append(0, l_counts))
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

    tasks = [(i, j, k, l)
             for i in range(n_groups)
             for j in range(i+1)
             for k in range(i+1)
             for l in range(k+1)
             if not (i == k and j < l)]

    s = _JSetup()
    s.opt = self
    s.mol = mol
    s.n_dm = n_dm
    s.dm_xyz = dm_xyz
    s.dm_xyz_size = dm_xyz_size
    s.pair_lst = pair_lst
    s.pair_loc = pair_loc
    s.task_offsets = task_offsets
    s.pair_mappings = pair_mappings
    s.tasks = tasks
    s.schemes = {t: JE._md_j_engine_quartets_scheme(t, n_dm=n_dm) for t in tasks}
    s._scheme_cache = {(t, n_dm): s.schemes[t] for t in tasks}
    s.q_cond = _dense_q_cond(self)
    s.l_ctr_bas_loc = l_ctr_bas_loc
    s.log_cutoff = log_cutoff
    s._env = _env
    rys_envs = self.rys_envs
    s.d_bas = rys_envs._env_ref_holder[1]
    s.d_env = rys_envs._env_ref_holder[2]
    # The long-range operator is still GPU4PySCF's; the density count is not
    # a reason to decline any more (catalogue B4a).
    s.use_ours = float(_env[PTR_RANGE_OMEGA]) == 0.
    s.lmax = int(mol._bas[:, ANG_OF].max())
    s.mdm_names = _mdm_names(s) if (s.use_ours and n_dm > 1) else None
    return s


def _mdm_names(s):
    """The multi-density kernels this molecule can launch.

    Every block size of every class in the task list -- deliberately **not**
    the ones this particular ``n_dm`` happens to need.  A Davidson solve calls
    ``get_j`` with a different number of trial vectors on every iteration, and
    a name set that moved with ``n_dm`` would hand NVRTC a different source
    text each time and compile the family over and over.
    """
    names = set()
    for (i, j, k, l) in s.tasks:
        tag = f'{i+j}_{k+l}'
        for nblk in _MDM_BLOCKS.get(tag, ()):
            names.add(f'j_dm{nblk}_{tag}')
    return frozenset(names)


def _launch_task(s, task, vj_xyz, force=None, override=None):
    """Run one (i,j,k,l) task into ``vj_xyz``.

    ``force`` is ``'ours'`` or ``'ref'`` to override the dispatch, which is
    what the per-class benchmark uses to A/B one class at a time.
    ``override`` maps a class tag to ``(entry, kernel)`` -- or, when
    ``s.n_dm > 1``, to ``(block size, entry, kernel)`` -- which is how the
    launch-configuration sweep drives a candidate kernel through exactly the
    same task list and screening data as the shipped one.  Returns the tag
    that ran, or ``None`` if the task has no pairs.
    """
    i, j, k, l = task
    self = s.opt
    mol = s.mol
    shls_slice = s.l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
    pair_ij_mapping, q_cond_ij, qd_ij = s.pair_mappings[i, j]
    pair_kl_mapping, q_cond_kl, qd_kl = s.pair_mappings[k, l]
    npairs_ij = pair_ij_mapping.size
    npairs_kl = pair_kl_mapping.size
    if npairs_ij == 0 or npairs_kl == 0:
        return None
    pair_ij_loc = s.pair_loc[s.task_offsets[i, j]:]
    pair_kl_loc = s.pair_loc[s.task_offsets[k, l]:]
    tag = f'{i+j}_{k+l}'

    if s.n_dm > 1:
        # Cover as many densities as our block sizes reach, hand the tail --
        # fewer than the smallest block, so at most three -- to GPU4PySCF.
        if force == 'ref' or not s.use_ours:
            plan, rem = [], s.n_dm
        elif override and tag in override:
            nblk, entry, kern = override[tag]
            plan, rem = [], s.n_dm
            while rem >= nblk:
                plan.append((nblk, s.n_dm - rem))
                rem -= nblk
        else:
            plan, rem = _dm_split(tag, s.n_dm)
            entry = kern = None
        if force == 'ours' and rem:
            raise KeyError(f'no fastj multi-density kernel covers the last '
                           f'{rem} of {s.n_dm} densities for class {tag}')
        for nblk, off in plan:
            _run_mdm(s, tag, nblk, off, vj_xyz, npairs_ij, npairs_kl,
                     pair_ij_mapping, pair_kl_mapping, pair_ij_loc,
                     pair_kl_loc, qd_ij, qd_kl, kern, entry)
        if rem:
            _run_ref(s, task, vj_xyz, s.n_dm - rem, rem, shls_slice,
                     npairs_ij, npairs_kl, pair_ij_mapping, pair_kl_mapping,
                     pair_ij_loc, pair_kl_loc, qd_ij, qd_kl,
                     q_cond_ij, q_cond_kl)
        return tag

    entry = LAUNCH.get(tag) if s.use_ours else None
    kern = None
    if override and tag in override and force != 'ref':
        entry, kern = override[tag]
    elif force == 'ref':
        entry = None
    elif force == 'ours' and entry is None:
        raise KeyError(f'no fastj kernel for class {tag}')
    if entry is not None:
        bx, by = entry['bsizex'], entry['bsizey']
        tx, ty = entry.get('threadsx', 16), entry.get('threadsy', 16)
        off_ij = _qd_offset_for_threads(npairs_ij, tx)
        off_kl = _qd_offset_for_threads(npairs_kl, ty)
        grid = ((npairs_ij + bx - 1) // bx, (npairs_kl + by - 1) // by)
        if kern is None:
            kern = _function('j_' + tag, s.lmax)
        kern(grid, tuple(entry.get('block', (16, 16))),
             (vj_xyz, s.dm_xyz, s.d_bas, s.d_env, np.int32(mol.nbas),
              np.int32(npairs_ij), np.int32(npairs_kl),
              pair_ij_mapping, pair_kl_mapping, pair_ij_loc, pair_kl_loc,
              qd_ij[off_ij:], qd_kl[off_kl:], s.q_cond,
              np.float32(s.log_cutoff)),
             shared_mem=entry['shm'] * 8)
        return tag
    _run_ref(s, task, vj_xyz, 0, s.n_dm, shls_slice, npairs_ij, npairs_kl,
             pair_ij_mapping, pair_kl_mapping, pair_ij_loc, pair_kl_loc,
             qd_ij, qd_kl, q_cond_ij, q_cond_kl)
    return tag


def _run_ref(s, task, vj_xyz, dm_off, n_dm, shls_slice, npairs_ij, npairs_kl,
             pair_ij_mapping, pair_kl_mapping, pair_ij_loc, pair_kl_loc,
             qd_ij, qd_kl, q_cond_ij, q_cond_kl):
    """GPU4PySCF's own MD_build_j over ``n_dm`` densities from ``dm_off``.

    The offset is what lets a multi-density build hand back only the tail:
    ``vj_xyz`` and ``dm_xyz`` are ``(n_dm, dm_xyz_size)`` and C-contiguous, so
    a row slice is a view with the right device pointer, and the scheme has to
    be rebuilt for the smaller count because it sizes shared memory from it.
    """
    i, j, k, l = task
    mol = s.mol
    rys_envs = s.opt.rys_envs
    err = JE.libvhf_md.MD_build_j(
        ctypes.cast(vj_xyz[dm_off:].data.ptr, ctypes.c_void_p),
        ctypes.cast(s.dm_xyz[dm_off:].data.ptr, ctypes.c_void_p),
        ctypes.c_int(n_dm), ctypes.c_int(s.dm_xyz_size),
        ctypes.byref(rys_envs), (ctypes.c_int*6)(*s.scheme(task, n_dm)),
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
        ctypes.c_float(s.log_cutoff),
        mol._atm.ctypes, ctypes.c_int(mol.natm),
        mol._bas.ctypes, ctypes.c_int(mol.nbas), s._env.ctypes)
    if err != 0:
        l_symb = lib.param.ANGULAR
        llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
        raise RuntimeError(f'MD_build_j kernel for {llll} failed')


def _run_mdm(s, tag, nblk, dm_off, vj_xyz, npairs_ij, npairs_kl,
             pair_ij_mapping, pair_kl_mapping, pair_ij_loc, pair_kl_loc,
             qd_ij, qd_kl, kern=None, entry=None):
    """One multi-density kernel over ``nblk`` densities from ``dm_off``."""
    name = f'dm{nblk}_{tag}'
    if entry is None:
        entry = _MDM[name]
    bx, by = entry['bsizex'], entry['bsizey']
    tx, ty = entry['threadsx'], entry['threadsy']
    off_ij = _qd_offset_for_threads(npairs_ij, tx)
    off_kl = _qd_offset_for_threads(npairs_kl, ty)
    grid = ((npairs_ij + bx - 1) // bx, (npairs_kl + by - 1) // by)
    if kern is None:
        kern = _function('j_' + name, s.lmax, 'mdm', s.mdm_names)
    kern(grid, tuple(entry['block']),
         (vj_xyz[dm_off:], s.dm_xyz[dm_off:], s.d_bas, s.d_env,
          np.int32(s.mol.nbas), np.int32(npairs_ij), np.int32(npairs_kl),
          pair_ij_mapping, pair_kl_mapping, pair_ij_loc, pair_kl_loc,
          qd_ij[off_ij:], qd_kl[off_kl:], s.q_cond,
          np.float32(s.log_cutoff), np.int32(s.dm_xyz_size)),
         shared_mem=entry['shm'] * 8)


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
    if callable(dms):
        dms = dms()
    s = _setup_181(self, dms)
    vj_xyz = cp.zeros_like(s.dm_xyz)
    for task in s.tasks:
        _launch_task(s, task, vj_xyz)

    if self.h_shls:
        raise NotImplementedError('fastj does not handle the CPU high-l path')
    vj = JE._Rt_to_dm(s.mol, vj_xyz.get(), s.pair_lst, s.pair_loc, self.rys_envs)
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
    lmax = int(prim_mol._bas[:, ANG_OF].max())
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
            kern = _function('j_' + tag, lmax)
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
