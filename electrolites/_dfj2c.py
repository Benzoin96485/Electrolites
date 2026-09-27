"""The two-centre metric integrals, and the optimiser they rebuild per call.

``docs/DF_WORK_LIST.md`` **W13**.  The item asks which of ``_decompose_j2c``'s
three operations -- ``int2c2e``, ``potrf``, the triangular solve -- is the
20-62 ms it costs on every system from 10 to 42 atoms.  Measured
(``benchmarks/df_j2c_probe.py``), the answer is **none of them**:

    glycine, def2-SVP (naux 471)       18.3 ms total,  13.9 in `Int2c2eOpt()`
    Pt amine, def2-SVP (1358)          41.5 ms total,  31.3 in `Int2c2eOpt()`
    penicillin V, def2-TZVPP (2191)    31.1 ms total,  19.2 in `Int2c2eOpt()`

The integral *kernel* is 2.9-5.8 ms, ``potrf`` 0.6-3.9 and the solve 0.4-3.2.
What the stage actually is, at the small end, is the ``Int2c2eOpt`` object
``int2c2e`` constructs on every call and throws away when it returns -- and
inside that, the shell-pair index table it builds with ``O(groups^2)``
individual cupy calls, one ``arange``/broadcast/``tril_indices``/``asarray``
per pair of angular-momentum classes.  None of that depends on the geometry
or on anything a kernel computed, and none of it is arithmetic; it is launch
latency, which is why it is nearly the same 14-31 ms at ``naux`` 471 and at
5598.

Two things follow, and this module is both:

* **the table is built on the host** and moved in one transfer, which makes
  the construction cheaper for the *first* call as well as later ones;
* **the optimiser is kept**, keyed on the contents of the auxiliary basis, so
  a second call at the same geometry -- a range-separated functional's second
  build, the gradient's own ``_decompose_j2c``, a Hessian's -- pays nothing
  at all.

W13 also asks whether a CPU factorisation is faster at those orders.  It is
not, by a wide margin: ``benchmarks/df_j2c_probe.py`` measures LAPACK
``dpotrf`` plus the two transfers at **4.3 ms against cuSOLVER's 0.7** at
naux 471 and **43.2 against 3.7** at 1992, and the whole tail on the host is
worse again.  That lead is rejected.

This module patches nothing (``docs/DF_WORK_LIST.md`` §5.12); it is
``electrolites/fastdfcderi.py`` that puts ``int2c2e`` in place, and this file
is what a probe imports to measure the replacement without patching a build.

    FASTDFCDERI_J2C=stock    GPU4PySCF's own `int2c2e`, optimiser per call
    FASTDFCDERI_J2C=build    build the optimiser the cheap way but keep none
    FASTDFCDERI_J2C=cache    the default: build it cheaply and keep it
"""
import hashlib
import os

import numpy as np
import cupy as cp

from pyscf.gto import Mole, PTR_RANGE_OMEGA
from gpu4pyscf.df import int3c2e_bdiv as bdiv
from gpu4pyscf.gto.mole import (
    SortedGTO, PBCIntEnvVars, _scale_sp_ctr_coeff)
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from gpu4pyscf.df.int3c2e_bdiv import L_AUX_MAX

#: 'stock', 'build' or 'cache'; see the module docstring.
_J2C = os.environ.get('FASTDFCDERI_J2C', 'cache')
#: how many optimisers are kept.  Each one holds the sorted auxiliary basis
#: it was built from and its shell-pair table, and the first of those can be
#: a few hundred megabytes at 113 atoms, so the cache is small on purpose:
#: what it is for is the *same* auxiliary basis asked for again, which is one
#: entry, and the second slot is there for a calculation that alternates
#: between two of them.
_MAX = int(os.environ.get('FASTDFCDERI_J2C_CACHE', 2))

STATS = {'calls': 0, 'built': 0, 'reused': 0, 'stock': 0}
DECLINED = {}

#: the original, kept for the fallback and for `benchmarks/coverage_probe.py`
_orig_int2c2e = bdiv.int2c2e

#: fingerprint -> Int2c2eOpt, least recently used first
_CACHE = {}


def _decline(reason):
    DECLINED[reason] = DECLINED.get(reason, 0) + 1


def stats():
    """What this module did in this process.  Nothing numerical can say."""
    out = dict(STATS)
    out['mode'] = _J2C
    out['cached'] = len(_CACHE)
    out['declined_reasons'] = dict(DECLINED)
    return out


def free_cache():
    """Drop every kept optimiser."""
    n = len(_CACHE)
    _CACHE.clear()
    return n


# --------------------------------------------------------------------------
# the shell-pair table
# --------------------------------------------------------------------------
def pair_cache(cell):
    """``Int2c2eOpt``'s ``bas_ij_cache``, built on the host.

    Upstream builds it with a handful of cupy calls per pair of
    angular-momentum classes -- ``groups*(groups+1)/2`` pairs, and a def2-SVP
    jkfit basis on a few elements already has ten or more classes -- so the
    table costs several hundred kernel launches and no arithmetic.  Here the
    indices are formed with numpy and the whole table crosses the bus once;
    the per-class arrays are then views of that single transfer, which is
    what ``aggregate_shl_pairs`` reads.

    ``bvk_ncells`` is 1 for a molecule, which is what makes upstream's
    ``img[:, None] * nbas + ijsh`` the identity and lets this be a plain
    index computation.
    """
    nbas = cell.nbas
    l_ctr_offsets = np.append(0, np.cumsum(cell.l_ctr_counts))
    groups = len(cell.uniq_l_ctr)
    keys, parts = [], []
    for i in range(groups):
        for j in range(i + 1):
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i + 1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j + 1]
            ish = np.arange(ish0, ish1, dtype=np.int64)
            jsh = np.arange(jsh0, jsh1, dtype=np.int64)
            ijsh = ish[:, None] * nbas + jsh
            if i == j:
                ijsh = ijsh[np.tril_indices(ish1 - ish0)]
            else:
                ijsh = ijsh.ravel()
            keys.append((i, j))
            parts.append(ijsh.astype(np.uint32))
    flat = cp.asarray(np.concatenate(parts)) if parts else cp.empty(
        0, dtype=np.uint32)
    out, off = {}, 0
    for key, part in zip(keys, parts):
        out[key] = flat[off:off + part.size]
        off += part.size
    return out


# --------------------------------------------------------------------------
# the optimiser
# --------------------------------------------------------------------------
def _fingerprint(cell):
    """What has to be equal for two calls to want the same optimiser.

    The basis structure and the coordinates, which is ``_atm``, ``_bas`` and
    ``_env``, plus whether the shells are Cartesian -- **except** the one
    word of ``_env`` that holds ``omega``.

    That exception is the whole reason a range-separated functional gets
    anything out of this.  ``mol.omega`` is not an attribute, it is
    ``_env[PTR_RANGE_OMEGA]``, so ``DF.range_coulomb`` setting it changes the
    array this hashes and a naive fingerprint would rebuild the optimiser for
    the second of the two builds every such SCF makes.  It is safe to leave
    out because the kernel does not read it: ``fill_int2c2e`` takes ``omega``
    as an explicit argument, and building the optimiser with ``omega = 0`` in
    its environment and then asking it for ``omega = +0.3`` and ``-0.3``
    reproduces GPU4PySCF's own matrices to **0.0** on a def2-TZVPP jkfit
    basis.  What *does* read ``cell.omega`` is ``_check_rsh_factors``, when
    the call does not name an omega -- which is why ``int2c2e`` below puts
    the kept optimiser's cell back in step before every call.
    """
    env = np.array(cell._env, copy=True)
    env[PTR_RANGE_OMEGA] = 0.
    h = hashlib.blake2b(digest_size=16)
    for arr in (cell._atm, cell._bas, env):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(bytes([bool(cell.cart)]))
    return h.digest()


def _build_opt(cell):
    """``Int2c2eOpt.__init__``'s molecular path, with a host-built table.

    Returns ``None`` -- and the caller then uses GPU4PySCF's own function --
    for anything this has not been measured on: a periodic cell, an angular
    momentum past the kernel's limit, or a ``SortedGTO`` conversion that
    changes the object, which would mean the fingerprint does not describe
    what the optimiser was built from.
    """
    from pyscf.pbc.gto import Cell
    if isinstance(cell, Cell):
        _decline('periodic cell')
        return None
    if not isinstance(cell, Mole):
        _decline('not a Mole')
        return None
    sorted_cell = SortedGTO.from_cell(cell)
    if int(sorted_cell.uniq_l_ctr[:, 0].max()) > L_AUX_MAX:
        _decline('angular momentum past L_AUX_MAX')
        return None
    # the same argument `int2c2e` passes, worked out the same way; for a
    # molecule it is `[1, 1, 1]`, and anything else means this is not the
    # case the rest of this function assumes
    bvk_kmesh = kpts_to_kmesh(cell, None, bound_by_supmol=True)
    if int(np.prod(bvk_kmesh)) != 1:
        _decline('more than one BvK cell')
        return None
    opt = Int2c2eOpt.__new__(Int2c2eOpt)
    opt.cell = sorted_cell
    opt.bvk_kmesh = bvk_kmesh
    opt.bvkcell = sorted_cell
    opt.bvkmesh_Ls = np.zeros((1, 3))
    _env = _scale_sp_ctr_coeff(sorted_cell)
    opt._rys_envs = PBCIntEnvVars.new(
        sorted_cell.natm, sorted_cell.nbas, 1, 1, sorted_cell._atm,
        sorted_cell._bas, _env, sorted_cell.ao_loc, cp.zeros((1, 3)))
    opt.bas_ij_cache = pair_cache(sorted_cell)
    return opt


def _get_opt(cell):
    """A built optimiser for ``cell``, from the cache when there is one."""
    if _J2C == 'build':
        return _build_opt(cell)
    key = _fingerprint(cell)
    opt = _CACHE.pop(key, None)
    if opt is None:
        opt = _build_opt(cell)
        if opt is None:
            return None
        STATS['built'] += 1
    else:
        STATS['reused'] += 1
    _CACHE[key] = opt                       # reinsert: last is most recent
    while len(_CACHE) > max(1, _MAX):
        _CACHE.pop(next(iter(_CACHE)))
    return opt


def int2c2e(mol, sort_output=True, omega=None):
    """``df/int3c2e_bdiv.py:int2c2e``, with the optimiser built once.

    Byte for byte upstream's integrals: the optimiser is the same object with
    the same fields, and the call that follows is upstream's own method.
    """
    STATS['calls'] += 1
    if _J2C == 'stock':
        STATS['stock'] += 1
        return _orig_int2c2e(mol, sort_output, omega)
    opt = _get_opt(mol)
    if opt is None:
        STATS['stock'] += 1
        return _orig_int2c2e(mol, sort_output, omega)
    # `_check_rsh_factors` falls back to `cell.omega` when the call does not
    # name one, and a kept optimiser holds the `cell` of whichever build made
    # it -- `DF.range_coulomb` sets `omega` on the *caller's* auxmol, so the
    # two have to be put back in step before the kernel reads it -- and put
    # back afterwards: `from_cell` returns a `SortedGTO` it is given, so the
    # kept `cell` can be the very object a full-range `Int3c2eOpt` holds as
    # its `auxmol`, and leaving the long-range omega on it made the next
    # user of that optimiser -- the DF gradient's reuse of the SCF's -- build
    # the long-range metric for the full-range call (`DF_WORK_LIST.md` W27)
    saved = opt.cell.omega
    opt.cell.omega = mol.omega
    try:
        return opt.int2c2e(None, sort_output=sort_output, omega=omega)
    finally:
        opt.cell.omega = saved
