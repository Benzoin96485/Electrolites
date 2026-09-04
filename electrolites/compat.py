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
"""
import functools
import inspect

import numpy as np
import cupy as cp

from gpu4pyscf.scf import jk as JK
from gpu4pyscf.dft import numint as NI

# ---------------------------------------------------------------- capabilities
NEW_JK_ABI = len(inspect.signature(JK._VHFOpt.get_k).parameters) >= 7
# read through any patch these modules may already have applied
NEW_VV10 = len(inspect.signature(
    getattr(NI, '_vv10nlc_orig', NI._vv10nlc)).parameters) == 4


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


def tril_pair_mappings(l_ctr_bas_loc, q_cond, cutoff, tile=6):
    """Density-screened ``ish*nbas + jsh`` lists, one per ``(i,j)`` group.

    Vendored from GPU4PySCF 1.7.0's ``jk._make_tril_pair_mappings``
    (Apache-2.0), which 1.8.1 removed.  Our kernels consume its output, so it
    has to keep producing identical lists on both releases.
    """
    if not NEW_JK_ABI:
        return JK._make_tril_pair_mappings(l_ctr_bas_loc, q_cond, cutoff, tile)
    nbas = q_cond.shape[0]
    q_flat = q_cond.ravel()
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
