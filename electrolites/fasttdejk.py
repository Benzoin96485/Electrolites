"""
The TDDFT gradient / NAC two-electron derivative for density *pairs*
(catalogue D10).

Importing this module replaces ``gpu4pyscf.grad.tdrhf._jk_energies_per_atom``
-- the per-atom derivative of ``j_t J[D1_t, D2_t] - k_t K[D1_t, D2_t]`` for a
list of density-matrix pairs, one output per pair (or their sum).  Every
restricted TDDFT/TDA gradient (``grad/tdrks.py``), every NAC
(``nac/tdrks.py``) and the batched multi-state NAC solver
(``nac/tdrks_grad_nacv.py``, which ``fastnacms`` routes ``force_and_nacv``
through) reach it through ``Gradients.jk_energies_per_atom``, which looks the
module-level name up at call time.

GPU4PySCF's kernel evaluates each integral quartet once per six pairs and
reads the density products from a global-memory pool per Rys root per
primitive quartet.  Here (``codegen/gen_ejkpair.py``): ``fastejk``'s
division-free one-thread-per-quartet kernels, ``EJK_NT`` outputs per launch
with their density products in registers.  Classes those kernels do not
cover (the wide ones, a bra pair of more than 36 primitives, a short-range
operator) go to GPU4PySCF's own kernel for that class, with every pair.

``FASTTDEJK_OFF=1`` keeps GPU4PySCF's function; ``FASTTDEJK_NT`` the outputs
per launch.  ``docs/README_fasttdejk.md``.
"""
import ctypes
import functools
import math
import os

import cupy as cp
import numpy as np

from gpu4pyscf.grad import tdrhf as TDG
from gpu4pyscf.lib.cupy_helper import condense
from gpu4pyscf.__config__ import props as gpu_specs

from . import fastejk as E
from . import _gen, _ksplit
from ._paths import KERNEL_DIR as HERE

_OFF = bool(os.environ.get('FASTTDEJK_OFF'))
NT = int(os.environ.get('FASTTDEJK_NT', 2))
_SRC = 'fastejkpair_generated.cu'

STATS = {'calls': 0, 'ours_groups': 0, 'stock_groups': 0}


def stats():
    return dict(STATS)


def _files():
    return [os.path.join(HERE, 'fastejk_prologue.cu'), _gen.source_path(_SRC)]


@functools.lru_cache(maxsize=8)
def _module(lmax, nt):
    files = _files()
    names = _ksplit.names_within(files, lmax)
    return cp.RawModule(code=_ksplit.source(files, names),
                        options=('-std=c++17', f'-DEJK_NT={int(nt)}'),
                        backend='nvrtc')


@functools.lru_cache(maxsize=1)
def _implemented():
    _, ks = _ksplit.chunks(_files()[1])
    return frozenset(k[len('ejkp_'):] for k in ks)


@functools.lru_cache(maxsize=512)
def _kernel(tag, lmax, nt):
    return _module(lmax, nt).get_function('ejkp_' + tag)


_REG_THREADS = 128


@functools.lru_cache(maxsize=256)
def _launch(tag):
    """(block, shared bytes) of a class: gen_ejkpair's own choice."""
    from .codegen import gen_ejkpair as G
    c = G.Class(*(int(x) for x in tag))
    if c.ngout <= 64 and 3 * c.gsz <= 128:
        return (_REG_THREADS,), 0
    nsq, gs, per, w, shm, headsz = G.shm_scheme(c)
    return (nsq, gs), shm * 8


def _stock_group(vhfopt, ejk, dm1, dm2, n_dm, nao, jf, kf, i, j, k, l,
                 dm_cond, log_cutoff, uniq_l_ctr, l_ctr_bas_loc, ws_pool,
                 dd_pool, dd_size):
    """GPU4PySCF's multi-density kernel for one class group, every pair."""
    mol = vhfopt.sorted_mol
    shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
    pair_ij, q_ij, s_ij = vhfopt.bas_pair_cache[i, j]
    pair_kl, q_kl, s_kl = vhfopt.bas_pair_cache[k, l]
    if pair_ij.size == 0 or pair_kl.size == 0:
        return
    scheme = TDG._ejk_quartets_scheme(mol, uniq_l_ctr[[i, j, k, l]])
    jfd, kfd = cp.asarray(jf), cp.asarray(kf)
    err = TDG.libvhf_rys.RYS_per_atom_jk_ip1_multidm(
        ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
        ctypes.cast(jfd.data.ptr, ctypes.c_void_p), jf.ctypes,
        ctypes.cast(kfd.data.ptr, ctypes.c_void_p), kf.ctypes,
        ctypes.cast(dm1.data.ptr, ctypes.c_void_p),
        ctypes.cast(dm2.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n_dm), ctypes.c_int(nao),
        vhfopt.rys_envs, (ctypes.c_int*2)(*scheme),
        (ctypes.c_int*8)(*shls_slice),
        ctypes.c_int(pair_ij.size), ctypes.c_int(pair_kl.size),
        ctypes.cast(pair_ij.data.ptr, ctypes.c_void_p),
        ctypes.cast(pair_kl.data.ptr, ctypes.c_void_p),
        ctypes.cast(q_ij.data.ptr, ctypes.c_void_p),
        ctypes.cast(q_kl.data.ptr, ctypes.c_void_p),
        ctypes.cast(s_ij.data.ptr, ctypes.c_void_p),
        ctypes.cast(s_kl.data.ptr, ctypes.c_void_p),
        ctypes.cast(E._diffuse_exps(vhfopt).data.ptr, ctypes.c_void_p),
        ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
        ctypes.c_float(log_cutoff), ctypes.c_float(0.),
        ctypes.cast(ws_pool.data.ptr, ctypes.c_void_p),
        ctypes.cast(dd_pool.data.ptr, ctypes.c_void_p),
        ctypes.c_int(dd_size),
        mol._atm.ctypes, ctypes.c_int(mol.natm),
        mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
    if err != 0:
        raise RuntimeError('RYS_per_atom_jk_ip1_multidm failed')


def _jk_energies_per_atom(vhfopt, dm_pairs, j_factor=None, k_factor=None,
                          sum_results=False, verbose=None):
    smol = vhfopt.sorted_mol
    if smol.omega < 0:            # short range: twice the roots, s_cond
        return TDG._jk_energies_per_atom_orig(
            vhfopt, dm_pairs, j_factor, k_factor, sum_results, verbose)
    STATS['calls'] += 1
    nao0 = vhfopt.mol.nao
    n_dm = len(dm_pairs)
    ntp = -(-n_dm // NT) * NT
    dm1 = cp.zeros((ntp, nao0, nao0))
    dm2 = cp.zeros((ntp, nao0, nao0))
    for t, p in enumerate(dm_pairs):
        if isinstance(p, cp.ndarray) and p.ndim == 2:
            dm1[t] = dm2[t] = p
        else:
            dm1[t] = p[0]
            dm2[t] = p[1]
    dm1 = cp.asarray(vhfopt.apply_coeff_C_mat_CT(dm1), order='C')
    dm2 = cp.asarray(vhfopt.apply_coeff_C_mat_CT(dm2), order='C')
    nao = dm1.shape[-1]
    jf = np.zeros(ntp)
    kf = np.zeros(ntp)
    if j_factor is not None:
        jf[:n_dm] = np.asarray(j_factor, dtype=np.float64)
    if k_factor is not None:
        kf[:n_dm] = np.asarray(k_factor, dtype=np.float64)
    do_j = float(np.any(jf != 0))
    do_k = float(np.any(kf != 0))

    ao_loc = smol.ao_loc
    uniq_l_ctr, l_ctr_bas_loc = E._sorted_meta(vhfopt)
    uniq_l = uniq_l_ctr[:, 0]
    nprim = uniq_l_ctr[:, 1]
    lmax = int(smol._bas[:, 1].max())
    n_groups = len(uniq_l_ctr)
    # GPU4PySCF's estimate: the max over every density of the call
    dm_cond = cp.log(condense('absmax', cp.vstack([dm1[:n_dm], dm2[:n_dm]]),
                              ao_loc) + 1e-300).astype(np.float32)
    log_cutoff = math.log(vhfopt.direct_scf_tol)
    q_cond = E._dense_q_cond(vhfopt)
    pair_mappings = E._tril_pair_mappings(
        l_ctr_bas_loc, q_cond, log_cutoff - float(dm_cond.max()), tile=6)
    ws = E._workspace()
    tab = E._tables()
    d_bas, d_env, d_ao_loc = vhfopt.rys_envs._env_ref_holder[1:]
    natm = smol.natm
    ejk = cp.zeros((ntp, natm, 3))
    jfd, kfd = cp.asarray(jf), cp.asarray(kf)
    omega = smol.omega
    inv_om2 = np.float64(0.0 if omega <= 0 else 1.0 / omega**2)
    impl = _implemented()

    stock_pool = dd_pool = None
    for i in range(n_groups):
        for j in range(i + 1):
            for k in range(i + 1):
                for l in range(k + 1):
                    tag = '%d%d%d%d' % (uniq_l[i], uniq_l[j], uniq_l[k],
                                        uniq_l[l])
                    prim = (int(nprim[i]), int(nprim[j]), int(nprim[k]),
                            int(nprim[l]))
                    if tag in impl and prim[0] * prim[1] <= E.MAX_PRIM_PAIR:
                        pm_ij = pair_mappings[i, j]
                        pm_kl = pair_mappings[k, l]
                        if pm_ij.size == 0 or pm_kl.size == 0:
                            continue
                        STATS['ours_groups'] += 1
                        kern = _kernel(tag, lmax, NT)
                        block, shm = _launch(tag)
                        nb = min(ws.n_blocks, int(pm_ij.size))
                        for t0 in range(0, ntp, NT):
                            if not (jf[t0:t0+NT].any() or kf[t0:t0+NT].any()):
                                continue
                            for kl0 in range(0, int(pm_kl.size),
                                             E.QUEUE_DEPTH):
                                kl1 = min(int(pm_kl.size), kl0 + E.QUEUE_DEPTH)
                                ws.head.fill(0)
                                kern((nb,), block,
                                     (ejk[t0], dm1[t0], dm2[t0], np.int32(nao),
                                      np.float64(do_j), np.float64(do_k),
                                      d_bas, d_env, d_ao_loc,
                                      np.int32(smol.nbas),
                                      np.int32(pm_ij.size),
                                      np.int32(kl1 - kl0), pm_ij,
                                      pm_kl[kl0:], q_cond, dm_cond,
                                      np.float32(log_cutoff), ws.pool,
                                      ws.head, np.int32(E.POOL_STRIDE), tab,
                                      *[np.int32(x) for x in prim], inv_om2,
                                      E._ONE, jfd[t0:], kfd[t0:],
                                      np.int32(natm * 3)),
                                     shared_mem=shm)
                    else:
                        STATS['stock_groups'] += 1
                        if stock_pool is None:
                            workers = gpu_specs['multiProcessorCount']
                            stock_pool = cp.empty(
                                workers * TDG.QUEUE_DEPTH + ntp + 1,
                                dtype=np.int32)
                            dd_size = TDG.DD_CACHE_MAX[lmax] * (ntp + 1)
                            dd_pool = cp.empty((workers, dd_size))
                        # GPU4PySCF's kernel faults at exactly six
                        # densities (its DM_BLOCK); a zero pair avoids it
                        nd = n_dm if n_dm != 6 else 7
                        _stock_group(vhfopt, ejk, dm1, dm2, nd, nao,
                                     jf[:nd].copy(), kf[:nd].copy(),
                                     i, j, k, l, dm_cond, log_cutoff,
                                     uniq_l_ctr, l_ctr_bas_loc, stock_pool,
                                     dd_pool, dd_size)
    ejk = ejk[:n_dm]
    if sum_results:
        ejk = ejk.sum(axis=0)
    return ejk.get()


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    TDG._jk_energies_per_atom_orig = TDG._jk_energies_per_atom
    TDG._jk_energies_per_atom = _jk_energies_per_atom
    _PATCHED = True


apply_patch()
