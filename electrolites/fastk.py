"""
Drop-in replacement for GPU4PySCF's exchange-matrix (K) build.

Importing this module monkey-patches ``gpu4pyscf.scf.jk._VHFOpt.get_k`` so that
the angular-momentum classes built by ``gen_kernels.py`` (one thread per shell
quartet), ``gen_k2_kernels.py`` (lifted from the classes GPU4PySCF splits
across ``threadIdx.y``), ``gen_khigh.py`` (written from the class itself, for
everything an spdf basis reaches that GPU4PySCF has no unrolled kernel for) and
``gen_kgeneral.py`` (a faster copy of its general kernel, for whatever the
others turn away) are handled by our kernels; every other class, basis or
option falls through to GPU4PySCF's own ``RYS_build_k``.  Screening, task
ordering, symmetry factors and the Rys root/weight tables are taken unchanged
from GPU4PySCF, so the K matrix agrees with GPU4PySCF's to round-off (~1e-14
relative, measured).

Both the full-range operator and the long-range (erf) one of a range-separated
hybrid are built here; ``get_k_rsh`` builds a linear combination of the two in
a single pass, which is what ``fastrsh`` uses for omega-B97X.  The short-range
(erfc) operator still falls through.

See README_fastk.md and README_fastrsh.md for what the kernels change and why.
"""
import os, re, json, math, ctypes, functools, inspect
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.gto import ANG_OF

from gpu4pyscf.scf import jk as JK
from gpu4pyscf.lib.cupy_helper import condense, transpose_sum, hermi_triu, asarray
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import props as gpu_specs, num_devices

from ._paths import KERNEL_DIR as HERE
from . import _ksplit
from . import _gen
from . import _nvcc

from .compat import (
    NEW_JK_ABI as _NEW_ABI, sorted_meta as _sorted_meta,
    dense_q_cond as _dense_q_cond, tril_pair_mappings as _tril_pair_mappings,
    resolve_rsh as _resolve_rsh, diffuse_exps as _diffuse_exps)

# Tasks buffered per block.  pair_kl is chunked to this, and each chunk repeats
# the bra-pair sweep, so a queue that is too small costs extra passes on large
# systems while a queue that is too large only costs pool memory.
QUEUE_DEPTH = int(os.environ.get('FASTK_QUEUE_DEPTH', 1 << 16))
# Persistent grid width, in blocks per SM.
BLOCKS_PER_SM = int(os.environ.get('FASTK_BLOCKS_PER_SM', 4))
#: Bra primitive pairs the kernels cache in shared memory.  36 is 6x6, the
#: worst case in a Pople set, and it is the bound every kernel here is
#: compiled with by default -- see fastk_prologue.cu.
MAX_PRIM_PAIR = 36             # must match MAX_PRIM_PAIR in fastk_prologue.cu
#: The largest bra pair a *second* module will be compiled for; see
#: `_ksplit.mpp_plan`, and ELECTROLITES_MAX_PRIM_PAIR=0 for the ablation.
MAX_PRIM_PAIR_CAP = _ksplit.MAX_PRIM_PAIR_CAP
# Ablations, for the timing tables:  hand the classes GPU4PySCF does not unroll
# back to it, and/or hand back every range-separated (omega != 0) class, which
# is what this module did before the omega-B97X round.
_USE_GENERAL = not os.environ.get('FASTK_NO_GENERAL')
_NO_OMEGA = bool(os.environ.get('FASTK_NO_OMEGA'))
# ablation: hand 1.8.x's fused lr*erf + sr*erfc request straight back
_FUSED_OFF = bool(os.environ.get('FASTK_NO_FUSED'))
# ablation: hand every multi-density request back, which is what this module
# did before the B4b round -- the measurement behind docs/ROUND4_B4B.md
_NO_NDM = bool(os.environ.get('FASTK_NO_NDM'))

# ablation: one atomicAdd per lane for the writes into a ket column, which is
# what every kernel did before catalogue B12 (docs/KERNEL_TARGETS_1.8.1.md
# §13.11)
_NO_KAGG = bool(os.environ.get('FASTK_NO_KAGG'))

# angular-momentum classes handled by our kernels: (li,lj,lk,ll) -> entry point
with open(os.path.join(HERE, 'fastk_launch.json')) as _f:
    _LAUNCH = json.load(_f)
# (li,lj,lk,ll) -> (kernel name, threads per block)
IMPLEMENTED = {tuple(int(c) for c in t): ('k_' + t, nt)
               for t, (nt, _mb) in _LAUNCH.items()}
# the classes GPU4PySCF splits across threadIdx.y (gen_k2_kernels.py)
with open(os.path.join(HERE, 'fastk2_launch.json')) as _f:
    _LAUNCH2 = json.load(_f)
IMPLEMENTED2 = {tuple(int(c) for c in t): (t, e) for t, e in _LAUNCH2.items()}
if os.environ.get('FASTK_NO_K2'):       # ablation: hand the lifted ones back
    IMPLEMENTED2 = {}
# The classes gen_khigh.py writes rather than lifts: everything an spdf basis
# reaches that GPU4PySCF has no unrolled kernel for, plus five it does unroll
# where the written kernel measured faster than the lifted one.  Same kernel
# signature and dispatch path, so they go in the same table -- and because they
# are added after it, they win where both have an entry.  FASTK_NO_HIGH=1 is
# the ablation that puts those five back on gen_k2_kernels.py's kernels and the
# rest on GPU4PySCF's general kernel.
_HIGH_SRC = os.environ.get('FASTKH_SRC', 'fastkhigh_generated.cu')
_HIGH_TAB = os.path.join(HERE, os.path.splitext(_HIGH_SRC)[0]
                         .replace('_generated', '_launch') + '.json')
_HIGH = ((not os.environ.get('FASTK_NO_HIGH')) and os.path.exists(_HIGH_TAB)
         and _gen.available(_HIGH_SRC))
if _HIGH:
    with open(_HIGH_TAB) as _f:
        _LAUNCH_H = json.load(_f)
    IMPLEMENTED2.update({tuple(int(c) for c in t): (t, e)
                         for t, e in _LAUNCH_H.items()})


def _sources(wide=False):
    """The CUDA files a module is compiled from.

    ``wide`` is the subset the second, wide-bra-pair module may be built from
    (see `_module`), and it is a *correctness* restriction rather than a
    tuning one.  The shared arrays of the prologue and of the
    one-thread-per-quartet family are declared ``double[MAX_PRIM_PAIR]``, so
    they follow whatever the module is compiled with; `gen_k2_kernels.py`
    bakes the default in as ``double[36]`` -- its output is tracked and
    regenerating it needs a GPU4PySCF checkout -- and a module built from it
    at a wider bound would write past the array instead of falling through.
    Those classes never see a long bra pair in any case: their bra is the
    highest-l group of the quartet, and what a def2 set contracts over more
    than six primitives is an s or a p shell.
    """
    gen1 = os.environ.get('FASTK_SRC', 'fastk_generated.cu')
    out = [os.path.join(HERE, 'fastk_prologue.cu'), _gen.source_path(gen1)]
    if wide:
        return out
    gen2 = os.environ.get('FASTK2_SRC', 'fastk2_generated.cu')
    out.append(_gen.source_path(gen2))
    if _HIGH:                      # the classes GPU4PySCF does not unroll
        out.append(_gen.source_path(_HIGH_SRC))
    return out


#: Maximum angular momentum of the basis currently being built.  The driver
#: sets it before any launch; the kernels of higher classes are then not
#: compiled at all, which is what a 6-31G* job would otherwise pay ~30 s of
#: NVRTC for on its first run.  ``None`` compiles the whole family.
_LMAX = None


#: Catalogue B12.  In the one-thread-per-quartet kernels every lane of a
#: block shares the bra pair (i, j), so the two exchange blocks a quartet
#: writes into a *ket* column, K[i,k] and K[j,k], have the same addresses on
#: every lane whose task has the same ``ksh`` -- and the task queue is in
#: ``pair_kl`` order, ``ksh`` major, so neighbouring lanes usually do.  Those
#: writes are 2/3 of the build's atomicAdds, ~4 lanes per address per warp
#: instruction, and the atomics are 72 % of the per-density cost ``C``.
#: Only that family: in the ones that spread a quartet over
#: ``threadIdx.y`` (``gen_k2``, ``gen_khigh``) the same treatment, keyed on
#: ``(ksh, threadIdx.y)``, measured slower -- 1.6x slower ``C`` on a
#: def2-TZVP build (§13.11).
#: `krun_init` finds the runs of equal ``ksh`` among adjacent active lanes
#: once per quartet; `krun_add` sums each value over its run with a
#: segmented shuffle tree and lets the run's first lane write it.
_KAGG_SRC = r'''
struct KRun { unsigned mask, lane, nxt, maxlen; bool head; };
__device__ __forceinline__ KRun krun_init(int key)
{
    KRun r;
    r.mask = __activemask();
    asm("mov.u32 %0, %%laneid;" : "=r"(r.lane));
    int prev = __shfl_up_sync(r.mask, key, 1);
    bool prev_in = r.lane > 0 && ((r.mask >> (r.lane - 1)) & 1u);
    r.head = !prev_in || prev != key;
    unsigned bound = __ballot_sync(r.mask, r.head) | ~r.mask;
    unsigned after = r.lane == 31 ? 0u : bound & ~((2u << r.lane) - 1u);
    r.nxt = after ? __ffs(after) - 1 : 32;
    r.maxlen = __reduce_max_sync(r.mask, r.head ? r.nxt - r.lane : 1u);
    return r;
}
__device__ __forceinline__ void krun_add(const KRun &r, double *p, double v)
{
    if (r.maxlen > 1) {
        for (unsigned off = 1; off < r.maxlen; off <<= 1) {
            double o = __shfl_down_sync(r.mask, v, off);
            if (r.lane + off < r.nxt) v += o;
        }
        if (!r.head) return;
    }
    atomicAdd(p, v);
}
'''
# the density loop of the one-thread-per-quartet family (and of the (ss|ss)
# kernel in the prologue), which opens right after nao2 is declared
_KAGG_MARK = re.compile(
    r'size_t nao2 = \(size_t\)nao \* nao;\n\s*'
    r'for \(int i_dm = 0; i_dm < n_dm; \+\+i_dm\) \{')
_KAGG_END = 'pair_ij = atomicAdd(head, 1);'
# vk+(i0+a)*nao+(k0+c), vk + j0*nao + k0, vk+(jj+b)*nao+(kk+c), ...
_KAGG_KCOL = re.compile(
    r'atomicAdd\((vk ?\+ ?\(?(?:[ij]0|ii|jj)(?:\+\d+\))?\*nao ?\+ ?'
    r'\(?(?:k0|kk)\b)')


# fastjk's Coulomb block J[i,j]: the bra pair is the block's, so every lane
# of a warp writes the same addresses
_KAGG_JIJ = re.compile(
    r'atomicAdd\((vj ?\+ ?\(?i0(?:\+\d+\))?\*nao ?\+ ?\(?j0\b)')


def _kagg(src):
    """Route the ket-column writes of the one-thread-per-quartet kernels
    through `krun_add` (see `_KAGG_SRC`); the other families are untouched.
    In `fastjk`'s fused kernels the J[i,j] writes go through it too, with
    every active lane of the warp in one run.
    """
    out, pos, n = [], 0, 0
    for m in _KAGG_MARK.finditer(src):
        if m.start() < pos:
            continue
        end = src.find(_KAGG_END, m.end())
        end = len(src) if end < 0 else end
        body, k = _KAGG_KCOL.subn(r'krun_add(kr_, \1', src[m.end():end])
        body, kj = _KAGG_JIJ.subn(r'krun_add(krj_, \1', body)
        out.append(src[pos:m.start()])
        out.append('const KRun kr_ = krun_init(ksh);\n'
                   + ('const KRun krj_ = krun_init(0);\n' if kj else '')
                   + m.group(0))
        out.append(body)
        n += k + kj
        pos = end
    out.append(src[pos:])
    if not n:
        return src
    return _KAGG_SRC + ''.join(out)


@functools.lru_cache(maxsize=16)
def _module(lmax=None, mpp=MAX_PRIM_PAIR):
    """The kernels for this basis' angular momenta, at this bra-pair bound.

    ``mpp`` above the default asks for the second module of `_mpp_plan`: the
    one-thread-per-quartet family alone, and only up to the angular momentum a
    long contraction can reach -- 14 kernels and 127 KB against the default
    module's 388 KB on a def2-TZVP metal basis, because the cost of a larger
    shared array is paid by every kernel compiled with it.  The bound is a
    preprocessor definition rather than a kernel argument for the same reason
    ``EJK_NDM`` is one in `fastejk`: as an argument the array would have to be
    sized for the worst case always, and the common path would pay for the
    rare one in shared memory it cannot be shown not to pay.
    """
    files = _sources(wide=mpp != MAX_PRIM_PAIR)
    names = None if lmax is None else _ksplit.names_within(files, lmax)
    opts = ('-std=c++17',) if mpp == MAX_PRIM_PAIR else (
        '-std=c++17', f'-DMAX_PRIM_PAIR={int(mpp)}')
    code = _ksplit.source(files, names)
    if not _NO_KAGG:
        code = _kagg(code)
    return cp.RawModule(code=code, options=opts, backend='nvrtc')


@functools.lru_cache(maxsize=1)
def _tables():
    """Rys root/weight tables, packed exactly as fastk_prologue.cu expects."""
    d = np.load(os.path.join(HERE, 'data', 'rys_roots_dat.npz'))
    packed = np.concatenate([d['ROOT_SMALLX_R0'], d['ROOT_SMALLX_R1'],
                             d['ROOT_SMALLX_W0'], d['ROOT_SMALLX_W1'],
                             d['ROOT_LARGEX_R_DATA'], d['ROOT_LARGEX_W_DATA'],
                             d['ROOT_RW_DATA']])
    return cp.asarray(packed, dtype=np.float64)


# Stride between the per-block task queues.  The gen_k2_kernels.py classes let
# the last lanes of a block run one thread-block past ntasks (as GPU4PySCF's own
# do) and pad the queue there, so each block's slot needs 256 elements of slack
# past QUEUE_DEPTH -- otherwise a block whose queue fills up writes its padding
# into the next block's task list.
POOL_STRIDE = QUEUE_DEPTH + 256


class _Workspace:
    """Scratch buffers reused across SCF iterations."""
    def __init__(self):
        self.n_blocks = gpu_specs['multiProcessorCount'] * BLOCKS_PER_SM
        self.pool = cp.empty(self.n_blocks * POOL_STRIDE, dtype=np.int32)
        self.head = cp.zeros(1, dtype=np.int32)
        # task queue for the GPU4PySCF kernels we fall through to
        self.ref_pool = cp.empty(
            gpu_specs['multiProcessorCount'] * JK.QUEUE_DEPTH + 1, dtype=np.int32)


@functools.lru_cache(maxsize=1)
def _workspace():
    return _Workspace()


def _mpp_plan(uniq_l_ctr, cap=None):
    """`_ksplit.mpp_plan` for this module's default bound."""
    return _ksplit.mpp_plan(uniq_l_ctr, MAX_PRIM_PAIR, cap)


@functools.lru_cache(maxsize=256)
def _function_for(name, lmax, mpp):
    try:
        return _module(lmax, mpp).get_function(name)
    except Exception:                     # a class the lmax subset missed
        return _module(None, mpp).get_function(name)


def _function(name, mpp=MAX_PRIM_PAIR, lmax=None):
    """``lmax`` overrides the basis' own for the wide-bra-pair module, whose
    subset is the angular momenta a long contraction can reach and not the
    whole basis -- see `_mpp_plan`."""
    return _function_for(name, _LMAX if lmax is None else lmax, mpp)


def _usable(omega, n_dm, nprim, i, j, k, l, mpp=None):
    """Conditions our kernels assume.  Anything else falls through to GPU4PySCF.

    None of these depend on the size of the molecule; they are properties of the
    basis set and of what the caller asked for.  ``omega`` is the resolved
    operator (see _resolve_rsh), not mol.omega: on 1.8.x the operator is an
    argument of get_k rather than a property of the option object.

    ``n_dm`` used to be one of these conditions -- the kernels accumulated a
    single density, so every CPHF, TDDFT and polarizability call fell through
    (catalogue B4b).  They now carry the density loop inside the quartet, so
    the only thing left of that restriction is the ablation switch.
    """
    if omega < 0:                           # short-range (erfc) ERIs need
        return False                        # twice the roots and s_estimator
    if omega != 0 and _NO_OMEGA:            # ablation: the pre-range-separation
        return False                        # behaviour, for the timing tables
    if n_dm != 1 and _NO_NDM:               # ablation: the pre-B4b behaviour
        return False
    # the bra primitive-pair cache is a fixed-size shared array.  ``mpp`` is
    # the bound of the module this class block would be launched from, which
    # is the default for every basis `_mpp_plan` finds nothing to widen.
    if int(nprim[i]) * int(nprim[j]) > (MAX_PRIM_PAIR if mpp is None else mpp):
        return False
    return True


_ONE = np.float64(1.0)
_ZERO = np.float64(0.0)


def _prep(self, dms, hermi):
    """Screening data for one _VHFOpt: dm_cond, cutoff, q_cond, pair mappings.

    q_cond depends on omega, so the full-range and long-range options give
    different pair mappings for the same density.
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


# Catalogue A18: a class with fewer bra pairs than the grid's blocks
# (``n_blocks = min(ws.n_blocks, npairs_ij)``) leaves most SMs idle, and the
# classes go one after another on one stream.  Such launches are put on a few
# side streams ("lanes"), each with its own task pool and head counter, so
# they overlap with each other and with the large classes on the caller's
# stream; ``_join_lanes`` makes the caller's stream wait for them before vk
# is used.  The sums into vk are atomic either way, so only the order of the
# floating-point additions changes.  FASTK_LANES=0 keeps one stream.
LANES = int(os.environ.get('FASTK_LANES', 4))
_LANE_STATE = {'next': 0, 'used': set(), 'event': None}


class _Lane:
    def __init__(self, n_blocks):
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.pool = cp.empty(n_blocks * POOL_STRIDE, dtype=np.int32)
        self.head = cp.zeros(1, dtype=np.int32)


@functools.lru_cache(maxsize=1)
def _lanes():
    return [_Lane(_workspace().n_blocks) for _ in range(LANES)]


class _Main:
    """The caller's stream with the shared workspace's pool and head."""
    def __init__(self, ws):
        self.stream = None
        self.pool = ws.pool
        self.head = ws.head


def _lane_for(ws, npairs_ij):
    if LANES <= 0 or npairs_ij >= ws.n_blocks:
        return _Main(ws)
    lanes = _lanes()
    i = _LANE_STATE['next'] % len(lanes)
    _LANE_STATE['next'] += 1
    lane = lanes[i]
    if i not in _LANE_STATE['used']:
        # the lane must see the caller's dms and vk before it starts
        if _LANE_STATE['event'] is None:
            ev = cp.cuda.Event(disable_timing=True)
            ev.record(cp.cuda.get_current_stream())
            _LANE_STATE['event'] = ev
        lane.stream.wait_event(_LANE_STATE['event'])
        _LANE_STATE['used'].add(i)
    return lane


def _join_lanes():
    if not _LANE_STATE['used']:
        _LANE_STATE['event'] = None
        return
    cur = cp.cuda.get_current_stream()
    lanes = _lanes()
    for i in sorted(_LANE_STATE['used']):
        ev = cp.cuda.Event(disable_timing=True)
        ev.record(lanes[i].stream)
        cur.wait_event(ev)
    _LANE_STATE['used'].clear()
    _LANE_STATE['event'] = None
    _LANE_STATE['next'] = 0


class _on:
    """``with _on(lane):`` -- the lane's stream, or nothing for _Main."""
    def __init__(self, lane):
        self.s = lane.stream

    def __enter__(self):
        if self.s is not None:
            self.s.__enter__()

    def __exit__(self, *exc):
        if self.s is not None:
            self.s.__exit__(*exc)


def _run_k1(name, nthreads, ws, tab, dev, vk, dms, nao, nbas, pm_ij, pm_kl,
            q_cond, dm_cond, log_cutoff, inv_om2, coef0, coef1, n_dm=1,
            mpp=MAX_PRIM_PAIR, mpp_lmax=None):
    """One of the 14 classes with one thread per shell quartet."""
    d_bas, d_env, d_ao_loc = dev
    kern = _function(name, mpp, mpp_lmax)
    # One block can only ever work on one bra pair at a time, so there is
    # nothing to gain from a grid wider than the number of bra pairs.
    n_blocks = min(ws.n_blocks, int(pm_ij.size))
    lane = _lane_for(ws, int(pm_ij.size))
    with _on(lane):
        for kl0, kl1 in lib.prange(0, int(pm_kl.size), QUEUE_DEPTH):
            lane.head.fill(0)
            kern((n_blocks,), (nthreads,),
                 (vk, dms, np.int32(n_dm), np.int32(nao),
                  d_bas, d_env, d_ao_loc, np.int32(nbas),
                  np.int32(pm_ij.size), np.int32(kl1 - kl0), pm_ij, pm_kl[kl0:],
                  q_cond, dm_cond, np.float32(log_cutoff), lane.pool, lane.head,
                  np.int32(POOL_STRIDE), tab, inv_om2, coef0, coef1))


def _run_k2(tag, e, prim, ws, tab, dev, vk, dms, nao, nbas, pm_ij, pm_kl,
            q_cond, dm_cond, log_cutoff, inv_om2, coef0, n_dm=1):
    """One of the 4 classes GPU4PySCF splits across threadIdx.y."""
    iprim, jprim, kprim, lprim = prim
    d_bas, d_env, d_ao_loc = dev
    shm = (e['shm_fixed'] + iprim * jprim) * 8
    nthreads = (e['nsq'], e['gout_stride'])
    n_blocks = min(ws.n_blocks, int(pm_ij.size))
    kern = _function(e.get('name', 'k2_' + tag))
    lane = _lane_for(ws, int(pm_ij.size))
    with _on(lane):
        for kl0, kl1 in lib.prange(0, int(pm_kl.size), QUEUE_DEPTH):
            lane.head.fill(0)
            kern((n_blocks,), nthreads,
                 (vk, dms, np.int32(n_dm), np.int32(nao),
                  d_bas, d_env, d_ao_loc, np.int32(nbas),
                  np.int32(pm_ij.size), np.int32(kl1 - kl0), pm_ij, pm_kl[kl0:],
                  q_cond, dm_cond, np.float32(log_cutoff), lane.pool, lane.head,
                  np.int32(POOL_STRIDE), tab, np.int32(iprim), np.int32(jprim),
                  np.int32(kprim), np.int32(lprim), inv_om2, coef0, _ZERO),
                 shared_mem=shm)


# Angular-momentum classes GPU4PySCF unrolls into a dedicated kernel (the tags
# with a rys_k_<tag> function in gvhf-rys/unrolled_rys_k.cu).  Everything else
# goes through its general rys_k_kernel, which is what fastkg replaces.
UNROLLED = {'0000', '1000', '1010', '1011', '1100', '1110', '1111', '2000',
            '2010', '2011', '2020', '2021', '2100', '2110', '2111', '2120',
            '2200', '2210', '3000', '3010', '3011', '3020', '3100', '3110',
            '3200'}


@functools.lru_cache(maxsize=1)
def _libmykg():
    """Our copy of GPU4PySCF's general K kernel (cuda/build_kgeneral.sh).

    Returns None if the library has not been built, in which case those classes
    keep going to GPU4PySCF."""
    so = _nvcc.general_k_library()
    if so is None:
        return None
    try:
        lib_ = ctypes.CDLL(so)
    except OSError:
        return None
    lib_.MYK_build_k_general_init.restype = ctypes.c_int
    if lib_.MYK_build_k_general_init(ctypes.c_int(JK.SHM_SIZE)) != 0:
        raise RuntimeError('MYK_build_k_general_init failed')
    return lib_


def _run_general(self, ws, vk, dms, n_dm, nao, shls_slice, pm_ij, pm_kl,
                 q_cond, dm_cond, log_cutoff, inv_om2):
    """One of the classes GPU4PySCF does not unroll.  Returns False (having
    done nothing) if the thread scheme does not fit, so the caller can fall
    back to GPU4PySCF."""
    mol = self.sorted_mol
    libg = _libmykg()
    if libg is None:
        return False
    n_blocks = min(ws.n_blocks, int(pm_ij.size))
    for kl0, kl1 in lib.prange(0, int(pm_kl.size), QUEUE_DEPTH):
        err = libg.MYK_build_k_general(
            ctypes.cast(vk.data.ptr, ctypes.c_void_p),
            ctypes.cast(dms.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_dm), ctypes.c_int(nao),
            ctypes.byref(self.rys_envs), (ctypes.c_int*8)(*shls_slice),
            ctypes.c_int(JK.SHM_SIZE),
            ctypes.c_int(pm_ij.size), ctypes.c_int(kl1 - kl0),
            ctypes.cast(pm_ij.data.ptr, ctypes.c_void_p),
            ctypes.cast(pm_kl[kl0:].data.ptr, ctypes.c_void_p),
            ctypes.cast(q_cond.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff),
            ctypes.cast(ws.pool.data.ptr, ctypes.c_void_p),
            ctypes.cast(ws.head.data.ptr, ctypes.c_void_p),
            ctypes.c_int(POOL_STRIDE), ctypes.c_int(n_blocks),
            ctypes.c_double(float(inv_om2)),
            mol._atm.ctypes, ctypes.c_int(mol.natm),
            mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
        if err == 2:                 # shared memory does not fit the scheme
            assert kl0 == 0, 'the thread scheme cannot depend on the chunk'
            return False
        if err != 0:
            raise RuntimeError('MYK_build_k_general failed')
    return True


def _general_ok(omega, n_dm, lll):
    tag = ''.join(str(c) for c in lll)
    return (_USE_GENERAL and tag not in UNROLLED
            and not (n_dm != 1 and _NO_NDM)
            and omega >= 0 and not (_NO_OMEGA and omega != 0))


def _run_ref_new(self, ws, vk, dms, n_dm, nao, shls_slice, ij, kl, dm_cond,
                 log_cutoff, dm_penalty, rsh, lll):
    """Hand the class back to GPU4PySCF 1.8.x's own RYS_build_k.

    1.8.x screens inside the kernel from the per-group q_cond/s_cond of
    _VHFOpt.bas_pair_cache plus dm_cond and a density penalty, so this path
    uses GPU4PySCF's own pair lists rather than the tile=6 lists our kernels
    consume, and passes the operator through as (omega, lr_factor, sr_factor).
    """
    mol = self.sorted_mol
    omega, lr_factor, sr_factor = rsh
    pair_ij, q_cond_ij, s_cond_ij = self.bas_pair_cache[ij]
    pair_kl, q_cond_kl, s_cond_kl = self.bas_pair_cache[kl]
    if pair_ij.size == 0 or pair_kl.size == 0:
        return
    err = JK.libvhf_rys.RYS_build_k(
        ctypes.cast(vk.data.ptr, ctypes.c_void_p),
        ctypes.cast(dms.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n_dm), ctypes.c_int(nao),
        ctypes.c_double(omega),
        ctypes.c_double(lr_factor), ctypes.c_double(sr_factor),
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
        mol._bas.ctypes)
    if err != 0:
        raise RuntimeError(f'RYS_build_k kernel for {lll} failed')


def _run_ref(self, ws, vk, dms, n_dm, nao, shls_slice, pm_ij, pm_kl, q_cond,
             dm_cond, log_cutoff, lll):
    """Hand the class back to GPU4PySCF 1.7.x's own RYS_build_k."""
    mol = self.sorted_mol
    s_ptr = lib.c_null_ptr()
    if mol.omega < 0:
        s_ptr = ctypes.cast(self.s_estimator.data.ptr, ctypes.c_void_p)
    for kl0, kl1 in lib.prange(0, int(pm_kl.size), JK.QUEUE_DEPTH):
        err = JK.libvhf_rys.RYS_build_k(
            ctypes.cast(vk.data.ptr, ctypes.c_void_p),
            ctypes.cast(dms.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_dm), ctypes.c_int(nao),
            ctypes.byref(self.rys_envs), (ctypes.c_int*8)(*shls_slice),
            ctypes.c_int(JK.SHM_SIZE),
            ctypes.c_int(pm_ij.size), ctypes.c_int(kl1 - kl0),
            ctypes.cast(pm_ij.data.ptr, ctypes.c_void_p),
            ctypes.cast(pm_kl[kl0:].data.ptr, ctypes.c_void_p),
            ctypes.cast(q_cond.data.ptr, ctypes.c_void_p), s_ptr,
            ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
            ctypes.c_float(log_cutoff),
            ctypes.cast(ws.ref_pool.data.ptr, ctypes.c_void_p),
            mol._atm.ctypes, ctypes.c_int(mol.natm),
            mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
        if err != 0:
            raise RuntimeError(f'RYS_build_k kernel for {lll} failed')


def _note_lmax(mol):
    """Record the basis' maximum angular momentum for the module split."""
    global _LMAX
    _LMAX = int(mol._bas[:, ANG_OF].max())


def _classes(self):
    """(i, j, k, l) group indices in GPU4PySCF's order, with their l values."""
    uniq_l = _sorted_meta(self)[0][:, 0]
    n_groups = np.count_nonzero(uniq_l <= JK.LMAX)
    for i in range(n_groups):
        for j in range(i + 1):
            for k in range(i + 1):
                for l in range(k + 1):
                    yield (i, j, k, l), (int(uniq_l[i]), int(uniq_l[j]),
                                         int(uniq_l[k]), int(uniq_l[l]))


def _inv_om2(mol):
    """Range separation enters the kernels only through rsqrt(aij+akl) ->
    rsqrt(aij+akl + aij*akl/omega^2).  omega == 0 (the full-range operator) is
    inv_om2 == 0, which leaves that fma exact."""
    return np.float64(0.0 if mol.omega == 0 else 1.0 / mol.omega**2)


def get_k(self, dms, hermi, verbose, omega=None, lr_factor=None,
          sr_factor=None):
    """Build the K matrix for the sorted molecule (see jk._VHFOpt.get_k).

    ``omega``/``lr_factor``/``sr_factor`` are GPU4PySCF 1.8.x's way of asking
    for the operator; on 1.7.x they are absent and the operator rides on the
    option object's mol.  Here only a single-range request is served -- the
    full-range operator, or the long-range (erf) one -- and it is built
    unscaled and multiplied by the factor on the way out, so that the classes
    that fall through to GPU4PySCF (which does apply the factor itself) can be
    asked for the unscaled operator too.  A request that mixes the ranges
    (lr_factor != sr_factor with omega != 0) is handed back whole.
    """
    if callable(dms):
        dms = dms()
    assert num_devices == 1, 'fastk currently supports a single device'
    mol = self.sorted_mol
    log = logger.new_logger(mol, verbose)
    assert dms.ndim == 3 and dms.shape[-1] == mol.ao_loc[-1]
    rsh_omega, lr, sr = _resolve_rsh(self, omega, lr_factor, sr_factor)
    # single-range only: omega == 0 forces lr == sr, and omega > 0 forces sr == 0
    assert rsh_omega >= 0, 'the short-range operator is handled by _dispatch'
    coef = np.float64(lr)
    dm_cond, log_cutoff, q_cond, pair_mappings = _prep(self, dms, hermi)

    if hermi == 0:
        dms = cp.vstack([dms, dms.transpose(0, 2, 1)])
    n_dm, nao = dms.shape[:2]
    vk = cp.zeros(dms.shape)
    dev = self.rys_envs._env_ref_holder[1:]

    uniq_l_ctr, l_ctr_bas_loc = _sorted_meta(self)
    _note_lmax(self.sorted_mol)
    nprim = uniq_l_ctr[:, 1]             # used by the shared-memory guard
    mpp_big, mpp_lmax = _mpp_plan(uniq_l_ctr)
    inv_om2 = np.float64(0.0 if rsh_omega == 0 else 1.0 / rsh_omega**2)
    dm_penalty = float(dm_cond.max())
    ref_rsh = (rsh_omega, 1.0, 0.0) if rsh_omega > 0 else (0.0, 1.0, 1.0)
    ws = _workspace()
    tab = _tables()

    for (i, j, k, l), lll in _classes(self):
        pm_ij = pair_mappings[i, j]
        pm_kl = pair_mappings[k, l]
        if pm_ij.size == 0 or pm_kl.size == 0:
            continue
        prim = tuple(int(nprim[x]) for x in (i, j, k, l))
        # A bra pair past the default bound is served from the second
        # module, which carries only the one-thread-per-quartet classes: the
        # shared memory a wider cache costs is paid by whatever is compiled
        # with it, and `_sources(wide=True)` says which files may be.  A class
        # the second module does not have -- one GPU4PySCF splits across
        # threadIdx.y, which happens from def2-QZVP up, where a d group can
        # pair with a 10-primitive s one -- falls through as before, so
        # `entry2` is suppressed rather than served from the wrong module.
        mpp = MAX_PRIM_PAIR if prim[0] * prim[1] <= MAX_PRIM_PAIR else mpp_big
        ok = _usable(rsh_omega, n_dm, nprim, i, j, k, l, mpp)
        entry2 = IMPLEMENTED2.get(lll) if (ok and mpp == MAX_PRIM_PAIR) else None
        if entry2 is not None:
            tag, e = entry2
            _run_k2(tag, e, prim, ws, tab, dev, vk, dms, nao, mol.nbas,
                    pm_ij, pm_kl, q_cond, dm_cond, log_cutoff, inv_om2, _ONE,
                    n_dm)
            continue
        entry = IMPLEMENTED.get(lll) if ok else None
        if entry is not None:
            name, nthreads = entry
            _run_k1(name, nthreads, ws, tab, dev, vk, dms, nao, mol.nbas,
                    pm_ij, pm_kl, q_cond, dm_cond, log_cutoff,
                    inv_om2, _ONE, _ZERO, n_dm, mpp,
                    None if mpp == MAX_PRIM_PAIR else mpp_lmax)
            continue
        shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
        if _general_ok(rsh_omega, n_dm, lll) and _run_general(
                self, ws, vk, dms, n_dm, nao, shls_slice, pm_ij, pm_kl,
                q_cond, dm_cond, log_cutoff, inv_om2):
            continue
        if _NEW_ABI:
            _run_ref_new(self, ws, vk, dms, n_dm, nao, shls_slice, (i, j),
                         (k, l), dm_cond, log_cutoff, dm_penalty, ref_rsh, lll)
        else:
            _run_ref(self, ws, vk, dms, n_dm, nao, shls_slice, pm_ij, pm_kl,
                     q_cond, dm_cond, log_cutoff, lll)
    _join_lanes()

    if hermi == 1:
        vk = transpose_sum(vk)
    elif hermi == 2:
        vk = transpose_sum(vk, hermi=2)
    else:
        vk, vkT = vk[:n_dm//2], vk[n_dm//2:]
        vk += vkT.transpose(0, 2, 1)

    if self.h_shls:
        # l > LMAX shells are handled on the CPU; delegate that piece to
        # GPU4PySCF rather than duplicating it.
        raise NotImplementedError('fastk does not handle the CPU high-l path')
    if coef != 1.0:
        vk *= coef
    return vk



def _full_range_opt(opt):
    """A sibling _VHFOpt on the same molecule carrying the full-range operator.

    GPU4PySCF 1.8.x asks for ``lr*erf(w r)/r + sr*erfc(w r)/r`` in one call,
    and the option object it asks on was built inside
    ``mol.with_range_coulomb(|omega|)``, so its q_cond estimates the long-range
    operator only.  The fused kernels here decompose that request as
    ``sr*(full range) + (lr-sr)*(long range)`` -- the identity
    ``erf + erfc == 1`` -- and the full-range half has to be screened on a
    full-range q_cond, which is the larger of the two.  Building one sibling
    option object per SCF gives it; the shell ordering is a function of the
    molecule alone, so the two share pair indices, and the assertion below
    keeps that assumption honest.
    """
    hit = getattr(opt, '_fastk_opt_full', None)
    if hit is not None:
        return hit
    mol = opt.mol
    kwargs = {}
    if hasattr(opt, 'tile'):
        kwargs['tile'] = opt.tile
    with mol.with_range_coulomb(0.):
        sib = JK._VHFOpt(mol, opt.direct_scf_tol, **kwargs).build()
    assert np.array_equal(sib.sorted_mol._bas, opt.sorted_mol._bas), \
        'the full-range sibling sorted the shells differently'
    opt._fastk_opt_full = sib
    return sib


def get_k_rsh(opt_f, opt_l, dms, hermi, coef_f, coef_l, verbose=None):
    """coef_f * K(full-range) + coef_l * K(long-range, erf) in a single pass.

    This is the exchange a range-separated hybrid such as omega-B97X needs.
    GPU4PySCF builds the two matrices with two independent passes over the
    shell quartets; the kernels here carry a range loop inside the primitive
    loop, so the screening, the bra and ket primitive data, the geometry and
    the density contraction with its atomicAdds are paid once instead of
    twice.  Only the Rys roots and the 2D recurrences -- the part that
    genuinely differs between the two operators -- are evaluated twice.

    Screening for the fused classes uses the full-range q_cond, which is the
    larger of the two, so nothing that the separate long-range build would
    have kept is dropped.

    ``hermi = 0`` (TDDFT and TDA trial densities) is served the way ``get_k``
    serves it: the transposed densities ride along as extra ones and the two
    halves are added on the way out.
    """
    assert num_devices == 1, 'fastk currently supports a single device'
    assert hermi in (0, 1, 2)
    if callable(dms):
        dms = dms()
    mol = opt_f.sorted_mol
    assert mol.omega == 0 and opt_l.sorted_mol.omega > 0
    assert dms.ndim == 3 and dms.shape[-1] == mol.ao_loc[-1]
    assert opt_f.direct_scf_tol == opt_l.direct_scf_tol
    if opt_f.h_shls or opt_l.h_shls:
        raise NotImplementedError('fastk does not handle the CPU high-l path')
    dm_cond, log_cutoff, q_cond_f, pm_f = _prep(opt_f, dms, hermi)
    _, _, q_cond_l, pm_l = _prep(opt_l, dms, hermi)
    if hermi == 0:
        dms = cp.asarray(cp.vstack([dms, dms.transpose(0, 2, 1)]), order='C')

    n_dm, nao = dms.shape[:2]
    vk = cp.zeros(dms.shape)
    dev_f = opt_f.rys_envs._env_ref_holder[1:]
    dev_l = opt_l.rys_envs._env_ref_holder[1:]
    _note_lmax(opt_f.sorted_mol)
    nprim = _sorted_meta(opt_f)[0][:, 1]
    mpp_big, mpp_lmax = _mpp_plan(_sorted_meta(opt_f)[0])
    inv_om2 = _inv_om2(opt_l.sorted_mol)
    c_f, c_l = np.float64(coef_f), np.float64(coef_l)
    ws = _workspace()
    tab = _tables()
    l_ctr_bas_loc = _sorted_meta(opt_f)[1]
    fallthrough = []

    for (i, j, k, l), lll in _classes(opt_f):
        prim = tuple(int(nprim[x]) for x in (i, j, k, l))
        # the wide bra-pair module, per class block, as in get_k
        mpp = MAX_PRIM_PAIR if prim[0] * prim[1] <= MAX_PRIM_PAIR else mpp_big
        ok = _usable(opt_l.sorted_mol.omega, n_dm, nprim, i, j, k, l, mpp)
        entry = IMPLEMENTED.get(lll) if ok else None
        if entry is not None:
            pm_ij, pm_kl = pm_f[i, j], pm_f[k, l]
            if pm_ij.size == 0 or pm_kl.size == 0:
                continue
            name, nthreads = entry
            _run_k1('k_rs_' + name[2:], nthreads, ws, tab, dev_f, vk, dms, nao,
                    mol.nbas, pm_ij, pm_kl, q_cond_f, dm_cond, log_cutoff,
                    inv_om2, c_f, c_l, n_dm, mpp,
                    None if mpp == MAX_PRIM_PAIR else mpp_lmax)
            continue
        entry2 = IMPLEMENTED2.get(lll) if (ok and mpp == MAX_PRIM_PAIR) else None
        if entry2 is not None:
            # These still take one pass per operator; the coefficient rides on
            # the geometric prefactor inside the kernel.
            tag, e = entry2
            for dev, pm, qc, io2, cf in ((dev_f, pm_f, q_cond_f, _ZERO, c_f),
                                         (dev_l, pm_l, q_cond_l, inv_om2, c_l)):
                pm_ij, pm_kl = pm[i, j], pm[k, l]
                if pm_ij.size == 0 or pm_kl.size == 0:
                    continue
                _run_k2(tag, e, prim, ws, tab, dev, vk, dms, nao, mol.nbas,
                        pm_ij, pm_kl, qc, dm_cond, log_cutoff, io2, cf, n_dm)
            continue
        fallthrough.append(((i, j, k, l), lll))
    _join_lanes()

    if fallthrough:
        # GPU4PySCF's kernel has no coefficient, so these accumulate into a
        # scratch matrix that is scaled in on the way out.
        omega_l = opt_l.sorted_mol.omega
        dm_penalty = float(dm_cond.max())
        tmp = cp.empty(dms.shape)
        for opt, pm, qc, c, io2, rsh in (
                (opt_f, pm_f, q_cond_f, c_f, _ZERO, (0.0, 1.0, 1.0)),
                (opt_l, pm_l, q_cond_l, c_l, inv_om2, (omega_l, 1.0, 0.0))):
            tmp.fill(0.)
            for (i, j, k, l), lll in fallthrough:
                pm_ij, pm_kl = pm[i, j], pm[k, l]
                if pm_ij.size == 0 or pm_kl.size == 0:
                    continue
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                if _general_ok(opt.sorted_mol.omega, n_dm, lll) and _run_general(
                        opt, ws, tmp, dms, n_dm, nao, shls_slice, pm_ij, pm_kl,
                        qc, dm_cond, log_cutoff, io2):
                    continue
                if _NEW_ABI:
                    _run_ref_new(opt, ws, tmp, dms, n_dm, nao, shls_slice,
                                 (i, j), (k, l), dm_cond, log_cutoff,
                                 dm_penalty, rsh, lll)
                else:
                    _run_ref(opt, ws, tmp, dms, n_dm, nao, shls_slice, pm_ij,
                             pm_kl, qc, dm_cond, log_cutoff, lll)
            vk += c * tmp

    if hermi == 1:
        return transpose_sum(vk)
    if hermi == 2:
        return transpose_sum(vk, hermi=2)
    vk, vkT = vk[:n_dm//2], vk[n_dm//2:]
    vk += vkT.transpose(0, 2, 1)
    return vk


_PATCHED = False

def _get_k_dispatch(self, dms, hermi, verbose, *rsh):
    """Use our kernels where they apply, otherwise GPU4PySCF's own get_k.

    ``*rsh`` is ``(omega, lr_factor, sr_factor)`` on GPU4PySCF 1.8.x and empty
    on 1.7.x.
    """
    if self.h_shls:
        return JK._VHFOpt._get_k_orig(self, dms, hermi, verbose, *rsh)
    if _NEW_ABI:
        omega, lr, sr = _resolve_rsh(self, *rsh) if rsh else _resolve_rsh(self)
        if omega < 0:
            # lr*erf + sr*erfc, which 1.8.x asks for in one call.  Our kernels
            # never evaluate erfc; erf + erfc == 1 turns the request into
            # sr*(full range) + (lr-sr)*(long range), which the fused kernels
            # build in one pass over the quartets (README_fastrsh.md).
            if _FUSED_OFF:
                return JK._VHFOpt._get_k_orig(self, dms, hermi, verbose, *rsh)
            if callable(dms):
                dms = dms()
            return get_k_rsh(_full_range_opt(self), self, dms, hermi,
                             sr, lr - sr, verbose)
    return get_k(self, dms, hermi, verbose, *rsh)


def apply_patch():
    global _PATCHED
    if not _PATCHED:
        JK._VHFOpt._get_k_orig = JK._VHFOpt.get_k
        JK._VHFOpt.get_k = _get_k_dispatch
        _PATCHED = True

apply_patch()
