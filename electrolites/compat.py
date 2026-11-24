"""What differs between GPU4PySCF 1.7.x and 1.8.x, in one place.

1.8.0 rebuilt the molecular J/K driver ("Refactor the molecular J/K matrix
construction, improving K-matrix performance for range-separated hybrid
functionals") and rewrote the VV10 non-local-correlation kernel.  The CUDA
side these modules bind to did not move -- ``RysIntEnvVars`` and the
``vhf.cuh`` structs are byte-identical between the two releases -- so the
kernels are unaffected and only the Python driver has to adapt.

What actually changed, and what this module hides:

* ``jk._VHFOpt.get_k`` gained ``(omega, lr_factor, sr_factor)``.  The operator
  is now an argument rather than a property of the option object, and one call
  builds ``lr_factor*erf(w r)/r + sr_factor*erfc(w r)/r``.
* ``_VHFOpt.q_cond`` and ``.s_estimator`` are properties that raise
  ``RuntimeError('deprecated')``.  The screening data lives in
  ``_VHFOpt.bas_pair_cache[(i,j)] = (pair_ij, q_cond_ij, s_cond_ij)``, per
  angular-momentum group, holding only the pairs that survived a build-time
  overlap mask.
* ``_VHFOpt.l_ctr_offsets`` / ``.uniq_l_ctr`` are gone; ``uniq_l_ctr`` and
  ``l_ctr_counts`` live on ``sorted_mol``.
* ``jk._make_tril_pair_mappings`` was removed.
* ``numint._vv10nlc`` takes ``(rho_drho, coords, weights, nlc_pars)`` instead
  of ``(rho, coords, vvrho, vvweight, vvcoords, nlc_pars)``.
* ``numint.eval_xc_eff`` returns ``exc`` with shape ``(N,)`` rather than
  ``(N,1)`` (call ``.ravel()`` and both work).

And between 1.8.x and 1.9.x (docs/UPGRADE_1.9_PLAN.md):

* **C ABI.**  ``RYS_build_jk``, ``RYS_build_jk_ip1``, ``RYS_per_atom_jk_ip1``,
  ``RYS_per_atom_jk_ip1_multidm``, ``RYS_per_atom_jk_ip2_type12/3`` and
  ``sum_ejk_int3c2e_ip1`` gained ``double omega, double lr_factor, double
  sr_factor`` in the *middle* of their argument lists (#859, #873).  A ctypes
  call with the 1.8.1 list is not rejected: on x86-64 the integer and pointer
  arguments still line up and the kernel reads the three doubles (and the
  ``cutoff``/``dm_penalty`` doubles after them) from whatever is left in the
  XMM registers.  :data:`RSH_CABI` and :func:`rsh_cargs` are the one place
  that inserts them.
* ``numint.eval_ao`` takes the screened-shell slice as one ``screening_info``
  tuple instead of ``shls_slice``/``nao_slice``/``ao_loc_slice``/
  ``ctr_offsets_slice`` (:func:`eval_ao_screened`).
* ``numint._block_loop`` lost ``strict_grid_order`` and always yields every
  block, empty ones included -- the old ``strict_grid_order=True`` behaviour
  (:func:`block_loop_kwargs`).
* ``scf.int4c2e.get_int4c2e`` returns the full ``(ij|kl)`` tensor through
  ``_ao2mo_general``; 1.8.x filled one triangle and left the symmetrisation
  to the caller (:data:`INT4C2E_FULL`).
"""
import ctypes
import functools
import inspect

import os

import numpy as np
import cupy as cp

from gpu4pyscf.scf import jk as JK
from gpu4pyscf.dft import numint as NI

# ---------------------------------------------------------------- capabilities
NEW_JK_ABI = len(inspect.signature(JK._VHFOpt.get_k).parameters) >= 7
# read through any patch these modules may already have applied
NEW_VV10 = len(inspect.signature(
    getattr(NI, '_vv10nlc_orig', NI._vv10nlc)).parameters) == 4


def _has_param(fn, name):
    fn = inspect.unwrap(fn)
    return name in inspect.signature(fn).parameters


def _rsh_cabi():
    # no module here replaces `hessian.rhf._get_jk_ip1` (fastjkip1 swaps the
    # ctypes symbol under it), so its signature is upstream's on every release
    from gpu4pyscf.hessian import rhf as _hrhf
    return _has_param(_hrhf._get_jk_ip1, 'lr_factor')


#: The direct-SCF and DF derivative kernels take ``(omega, lr_factor,
#: sr_factor)`` as C doubles (GPU4PySCF >= 1.9.0)
RSH_CABI = _rsh_cabi()
#: ``eval_ao(..., screening_info=...)`` (>= 1.9.0)
EVAL_AO_SCREENING_INFO = _has_param(NI.eval_ao, 'screening_info')
#: ``_block_loop(..., strict_grid_order=...)`` (<= 1.8.x)
BLOCK_LOOP_STRICT = _has_param(NI._block_loop, 'strict_grid_order')


def _int4c2e_full():
    from gpu4pyscf.scf import int4c2e as _i4
    return not _has_param(_i4.get_int4c2e, 'stream')


#: ``int4c2e.get_int4c2e`` returns the full symmetric tensor (>= 1.9.0)
INT4C2E_FULL = _int4c2e_full()


def rsh_cargs(mol, omega=None, lr_factor=None, sr_factor=None):
    """The ``(omega, lr_factor, sr_factor)`` ctypes doubles a >= 1.9 kernel
    expects (resolved by upstream's own ``_check_rsh_factors``, so the
    libcint sign convention for ``omega`` holds), or ``()`` on a release whose
    kernels do not take them.  Splice it into the argument list where the
    kernel declares them -- right after ``n_dm`` or ``nao``."""
    if not RSH_CABI:
        return ()
    return rsh_cdoubles(*JK._check_rsh_factors(mol, omega, lr_factor,
                                               sr_factor))


def rsh_cdoubles(omega, lr_factor, sr_factor):
    """:func:`rsh_cargs` for an operator already resolved to its three
    numbers: the ctypes doubles on >= 1.9, ``()`` before."""
    if not RSH_CABI:
        return ()
    return (ctypes.c_double(omega), ctypes.c_double(lr_factor),
            ctypes.c_double(sr_factor))


def bind_tail(fname, args, kwargs, positional, keywords=()):
    """Bind the optional trailing arguments of a replaced function whose
    signature differs between releases.

    ``positional`` names the arguments a caller may pass by position after
    the fixed ones, in order; ``keywords`` are further names accepted only
    by keyword.  Returns ``{name: value}`` for what was passed; raises the
    ``TypeError`` Python would for anything else.  Used where 1.9 inserted
    ``lr_factor``/``sr_factor`` ahead of ``verbose`` or ``sum_results`` in a
    positional list, so the same replacement can serve both releases'
    callers.
    """
    if len(args) > len(positional):
        raise TypeError(f'{fname}() got {len(args)} trailing positional '
                        f'arguments, at most {len(positional)} accepted')
    out = dict(zip(positional, args))
    allowed = set(positional) | set(keywords)
    for k, v in kwargs.items():
        if k not in allowed:
            raise TypeError(f'{fname}() got an unexpected keyword argument '
                            f'{k!r}')
        if k in out:
            raise TypeError(f'{fname}() got multiple values for argument '
                            f'{k!r}')
        out[k] = v
    return out


def rsh_kwargs(bound):
    """The ``lr_factor``/``sr_factor`` entries of ``bound`` that are set --
    what a hand-back to upstream should forward (never a key a 1.8 function
    does not take, since 1.8 callers never set them)."""
    return {k: bound[k] for k in ('lr_factor', 'sr_factor')
            if bound.get(k) is not None}


def _release():
    import gpu4pyscf
    return tuple(int(x) for x in gpu4pyscf.__version__.split('.')[:3]
                 if x.isdigit())


#: ``libvhf_rys.transpose_write`` takes the stream as its first argument
#: (1.9.0, ``lib/gvhf-rys/decompress.cu``).  No Python signature to probe --
#: it is a ctypes symbol -- so this one is read off the release number.
TRANSPOSE_WRITE_STREAM = _release() >= (1, 9, 0)


def transpose_write(host, src, naux, npairs, p0, p1):
    """``libvhf_rys.transpose_write`` of the device block ``src`` (``naux`` x
    ``p1 - p0`` of an ``npairs``-wide tensor) into the mapped host array
    ``host``, through whichever argument list this release has; on 1.9 the
    copy is queued on the current stream and waited for, as upstream does.
    Returns the kernel's error code."""
    from gpu4pyscf.df.int3c2e_bdiv import libvhf_rys
    args = (host.ctypes, ctypes.cast(src.data.ptr, ctypes.c_void_p),
            ctypes.c_int(naux), ctypes.c_int(npairs),
            ctypes.c_int(p0), ctypes.c_int(p1))
    if not TRANSPOSE_WRITE_STREAM:
        return libvhf_rys.transpose_write(*args)
    stream = cp.cuda.get_current_stream()
    err = libvhf_rys.transpose_write(ctypes.cast(stream.ptr, ctypes.c_void_p),
                                     *args)
    stream.synchronize()
    return err


def rsh_df_key(omega, lr_factor=None, sr_factor=None):
    """The key ``DF.range_coulomb`` files a range-separated DF object under
    in ``DF._rsh_df`` on this release (1.9 added the two factors to it)."""
    from gpu4pyscf.df import df as _df
    if not _has_param(_df.DF.range_coulomb, 'lr_factor'):
        return '%.6f' % omega
    lr = '%.6f' % lr_factor if lr_factor is not None else 'None'
    sr = '%.6f' % sr_factor if sr_factor is not None else 'None'
    return 'omega_%.6f_lr_factor_%s_sr_factor_%s' % (omega, lr, sr)


def ip2_envs_and_operator(envs_ref, omega, lr_factor, sr_factor):
    """The ``envs`` pointer and the operator's three doubles in the order
    upstream's ``libvhf_rys.ejk_int3c2e_ip2`` declares them: ``envs`` first
    up to 1.8, the doubles first from 1.9 (#859).  (On x86-64 the two orders
    are the same call -- pointer and double arguments use separate
    registers -- but the list should match the callee.)"""
    op = (ctypes.c_double(omega), ctypes.c_double(lr_factor),
          ctypes.c_double(sr_factor))
    return op + (envs_ref,) if RSH_CABI else (envs_ref,) + op


def block_loop_kwargs():
    """``{'strict_grid_order': True}`` where ``_block_loop`` still takes it,
    ``{}`` where it always behaves that way."""
    return {'strict_grid_order': True} if BLOCK_LOOP_STRICT else {}


def eval_ao_screened(mol, coords, deriv, nao_slice, shls_slice, ao_loc_slice,
                     ctr_offsets_slice, **kwargs):
    """``numint.eval_ao`` on a screened shell subset, through whichever
    signature this release has.  1.9.x packs the four slices into
    ``screening_info=(nao, shls, ao_loc, ctr_offsets)`` -- note the order."""
    if EVAL_AO_SCREENING_INFO:
        return NI.eval_ao(mol, coords, deriv, screening_info=(
            nao_slice, shls_slice, ao_loc_slice, ctr_offsets_slice), **kwargs)
    return NI.eval_ao(mol, coords, deriv, nao_slice=nao_slice,
                      shls_slice=shls_slice, ao_loc_slice=ao_loc_slice,
                      ctr_offsets_slice=ctr_offsets_slice, **kwargs)


def sorted_meta(opt):
    """``(uniq_l_ctr, l_ctr_offsets)`` for either GPU4PySCF generation."""
    if 'l_ctr_offsets' in opt.__dict__:
        return opt.uniq_l_ctr, opt.l_ctr_offsets
    sm = opt.sorted_mol
    return sm.uniq_l_ctr, np.append(0, np.cumsum(sm.l_ctr_counts))


def dense_q_cond(opt):
    """The ``nbas x nbas`` ``log(sqrt(absmax((ij|ij))))`` matrix.

    1.7.0 keeps it as ``_VHFOpt.q_cond``.  1.8.1 keeps only the surviving
    pairs, per group, in ``bas_pair_cache``, keyed by ``ish*nbas + jsh`` -- the
    same encoding the kernels index with -- so scattering those values back
    into a dense matrix reproduces it.  Pairs 1.8.1 dropped on its build-time
    overlap mask get ``-inf``, which screens them out of the pair mappings
    exactly as a negligible q_cond would have.
    """
    if not NEW_JK_ABI:
        return cp.asarray(opt.q_cond)
    cache = opt.bas_pair_cache
    hit = getattr(opt, '_g4pc_qcond', None)
    if hit is not None and hit[0] is cache:
        return hit[1]
    nbas = opt.sorted_mol.nbas
    q = cp.full(nbas * nbas, -np.inf, dtype=np.float32)
    # jk's cache holds (pair, q_cond, s_cond); j_engine's holds (pair, q_cond)
    for entry in cache.values():
        pair_ij, q_ij = entry[0], entry[1]
        q[pair_ij.astype(np.int64)] = q_ij
    q = q.reshape(nbas, nbas)
    opt._g4pc_qcond = (cache, q)
    return q


_TRIL_GEOM = {}


def _tril_geometry(l_ctr_bas_loc, nbas, tile):
    """The candidate ``ish*nbas + jsh`` of every ``(i,j)`` group before the
    screen, concatenated in the order the groups are listed, with each
    candidate's group index.  They depend on the shell layout only, so they
    are built once per layout: building them per call was ~40 boolean-mask
    indexings per group, each a device-host synchronisation, and 97 % of
    ``fastk._prep``'s time with the device idle (docs/DIRECT_SURVEY.md)."""
    key = (tuple(int(x) for x in l_ctr_bas_loc), int(nbas), int(tile))
    hit = _TRIL_GEOM.get(key)
    if hit is not None:
        return hit
    if len(_TRIL_GEOM) > 8:
        _TRIL_GEOM.clear()
    n_groups = len(l_ctr_bas_loc) - 1
    keys, lists = [], []
    for i in range(n_groups):
        for j in range(i + 1):
            ish0, ish1 = int(l_ctr_bas_loc[i]), int(l_ctr_bas_loc[i + 1])
            jsh0, jsh1 = int(l_ctr_bas_loc[j]), int(l_ctr_bas_loc[j + 1])
            ntiles_i = (ish1 - ish0 + tile - 1) // tile
            ntiles_j = (jsh1 - jsh0 + tile - 1) // tile
            ish = cp.arange(ish0, ish0 + ntiles_i * tile,
                            dtype=np.int32).reshape(ntiles_i, tile)
            jsh = cp.arange(jsh0, jsh0 + ntiles_j * tile,
                            dtype=np.int32).reshape(ntiles_j, tile)
            ish = ish[:, None, :, None]
            jsh = jsh[None, :, None, :]
            pair_ij = ish * nbas + jsh
            if i == j:
                pair_ij = pair_ij[(ish >= jsh) & (ish < ish1) & (jsh < jsh1)]
            else:
                pair_ij = pair_ij[(ish < ish1) & (jsh < jsh1)]
            keys.append((i, j))
            lists.append(pair_ij.ravel())
    allp = cp.concatenate(lists)
    gid = cp.concatenate([cp.full(len(x), g, dtype=np.int32)
                          for g, x in enumerate(lists)])
    hit = _TRIL_GEOM[key] = (keys, allp, gid)
    return hit


def tril_pair_mappings(l_ctr_bas_loc, q_cond, cutoff, tile=6):
    """Density-screened ``ish*nbas + jsh`` lists, one per ``(i,j)`` group.

    Vendored from GPU4PySCF 1.7.0's ``jk._make_tril_pair_mappings``
    (Apache-2.0), which 1.8.1 removed.  Our kernels consume its output, so it
    has to keep producing identical lists on both releases -- and it does:
    the unscreened lists are cached per shell layout (``_tril_geometry``) and
    screened here in one pass, with two synchronisations per call instead of
    one per group.  ``ELECTROLITES_TRIL=pergroup`` is the vendored loop.
    """
    if not NEW_JK_ABI:
        return JK._make_tril_pair_mappings(l_ctr_bas_loc, q_cond, cutoff, tile)
    nbas = q_cond.shape[0]
    q_flat = q_cond.ravel()
    if os.environ.get('ELECTROLITES_TRIL', '') == 'pergroup':
        return _tril_pair_mappings_pergroup(l_ctr_bas_loc, q_flat, nbas,
                                            cutoff, tile)
    keys, allp, gid = _tril_geometry(l_ctr_bas_loc, nbas, tile)
    idx = cp.nonzero(q_flat[allp] > cutoff)[0]
    sel = allp[idx]
    if idx.size == 0:            # an all-zero density: nothing survives
        return {k: sel for k in keys}
    counts = cp.bincount(gid[idx], minlength=len(keys)).get()
    offs = np.append(0, np.cumsum(counts))
    return {k: sel[offs[g]:offs[g + 1]] for g, k in enumerate(keys)}


def _tril_pair_mappings_pergroup(l_ctr_bas_loc, q_flat, nbas, cutoff, tile):
    """The vendored loop, one screened list per group as it is built."""
    n_groups = len(l_ctr_bas_loc) - 1
    pair_mappings = {}
    for i in range(n_groups):
        for j in range(i + 1):
            ish0, ish1 = int(l_ctr_bas_loc[i]), int(l_ctr_bas_loc[i + 1])
            jsh0, jsh1 = int(l_ctr_bas_loc[j]), int(l_ctr_bas_loc[j + 1])
            ntiles_i = (ish1 - ish0 + tile - 1) // tile
            ntiles_j = (jsh1 - jsh0 + tile - 1) // tile
            ish = cp.arange(ish0, ish0 + ntiles_i * tile,
                            dtype=np.int32).reshape(ntiles_i, tile)
            jsh = cp.arange(jsh0, jsh0 + ntiles_j * tile,
                            dtype=np.int32).reshape(ntiles_j, tile)
            ish = ish[:, None, :, None]
            jsh = jsh[None, :, None, :]
            pair_ij = ish * nbas + jsh
            if i == j:
                pair_ij = pair_ij[(ish >= jsh) & (ish < ish1) & (jsh < jsh1)]
            else:
                pair_ij = pair_ij[(ish < ish1) & (jsh < jsh1)]
            pair_ij = pair_ij[q_flat[pair_ij] > cutoff]
            pair_mappings[i, j] = cp.asarray(pair_ij, dtype=np.int32)
    return pair_mappings


def resolve_rsh(opt, omega=None, lr_factor=None, sr_factor=None):
    """``(omega, lr_factor, sr_factor)`` of the operator the caller asked for.

    On 1.7.x the operator rides on the option object's own mol -- one option
    per omega -- and ``get_k`` carries no factors, so the request is always
    that mol's whole, unscaled operator.
    """
    if NEW_JK_ABI:
        return JK._check_rsh_factors(opt.mol, omega, lr_factor, sr_factor)
    return opt.sorted_mol.omega, 1.0, 1.0


def diffuse_exps(opt):
    """The float32 diffuse-exponent array 1.8.x's K kernel screens with."""
    hit = getattr(opt, '_g4pc_diffuse', None)
    if hit is not None:
        return hit
    from gpu4pyscf.gto.mole import extract_pgto_params
    exps, _coef = extract_pgto_params(opt.sorted_mol, 'diffuse')
    out = cp.asarray(exps, dtype=np.float32)
    opt._g4pc_diffuse = out
    return out


def vv10nlc(rho4, coords, weights, nlc_pars):
    """``numint._vv10nlc`` through whichever signature this release has.

    Both releases sum the same O(n_grid^2) pair term over one grid; 1.7.x
    spelled the single grid out as three separate arguments.
    """
    if NEW_VV10:
        return NI._vv10nlc(rho4, coords, weights, nlc_pars)
    return NI._vv10nlc(rho4, coords, rho4, weights, coords, nlc_pars)
