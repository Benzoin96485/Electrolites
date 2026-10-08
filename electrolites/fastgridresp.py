"""The grid-response Hessian without storing the weights' second derivative (B9).

Every use GPU4PySCF 1.8.1 makes of ``hessian/rks.py:get_d2weight_dAdB`` --
the RKS and UKS ``_get_exc_deriv2_grid_response`` (three times each) and the
VV10 ``_get_enlc_deriv2`` -- is

    d2w = get_d2weight_dAdB(mol, grids, (g0, g1))     # natm^2 * 9 per point
    d2e += contract('ABdDg,g->ABdD', d2w, eps)

which writes ``natm^2 * 9 * ngrids`` doubles per block (520 MB per 4096 points
at 42 atoms), fills the blocks of each point's own atom with 18*natm small
launches, and stops at ~241 atoms on its int32 size assert
(``docs/KERNEL_TARGETS_1.8.1.md`` §11.5).

`d2w_contract` computes the same ``(natm, natm, 3, 3)`` directly: GPU4PySCF's
two kernels, lifted unchanged into ``__device__`` functions by
``codegen/gen_d2w_fused.py``, hand each point's 3x3 block to a wrapper that
multiplies it by ``eps`` and accumulates it -- warp-reduced -- into the plain
term and the two translation-invariance terms (the derivation is at the top
of ``kernels/cuda/mygrid/d2w_fused_wrappers.cu``).  Memory is ``3 natm^2 * 9``
doubles whatever the block, so the int32 wall is gone.

The callers are GPU4PySCF's source with those two lines replaced, bound to
their modules.  ``FASTGRIDRESP_OFF=1`` keeps GPU4PySCF's.
``docs/README_fastgridresp.md``.
"""
import ctypes
import functools
import inspect
import os
import re
import textwrap
import types

import numpy as np
import cupy as cp

from ._paths import CUDA_DIR

_OFF = bool(os.environ.get('FASTGRIDRESP_OFF'))
STATS = {'fused': 0, 'stock': 0}


def stats():
    return dict(STATS)


@functools.lru_cache(maxsize=4)
def _module(device_id):
    d = os.path.join(CUDA_DIR, 'mygrid')
    code = (open(os.path.join(d, 'd2w_fused.cu')).read() +
            open(os.path.join(d, 'd2w_fused_wrappers.cu')).read())
    return cp.RawModule(code=code, options=('-std=c++17',), backend='nvrtc')


def d2w_contract(mol, grids, grid_range, eps):
    """``contract('ABdDg,g->ABdD', get_d2weight_dAdB(mol, grids, grid_range),
    eps)`` without the intermediate."""
    from gpu4pyscf.hessian.rks import libgdft
    from gpu4pyscf.dft import radi
    from gpu4pyscf.dft.gen_grid import get_C_interface_scheme_id
    from pyscf import lib
    STATS['fused'] += 1
    natm = mol.natm
    atm_coords = cp.asarray(mol.atom_coords(), order='F')
    inv_atom_distance = 1 / cp.linalg.norm(
        atm_coords[:, None, :] - atm_coords[None, :, :], axis=2)
    cp.fill_diagonal(inv_atom_distance, 0)
    if grids.radii_adjust is None:
        a_factor = None
    else:
        assert grids.radii_adjust == radi.treutler_atomic_radii_adjust
        a_factor = radi.get_treutler_fac(mol, grids.atomic_radii)
    scheme_id = get_C_interface_scheme_id(grids.becke_scheme)
    g0, g1 = grid_range
    ngrids = g1 - g0
    coords = cp.asfortranarray(cp.asarray(grids.coords)[g0:g1])
    qw = cp.ascontiguousarray(cp.asarray(grids.quadrature_weights)[g0:g1])
    atm_idx = cp.ascontiguousarray(
        cp.asarray(grids.atm_idx)[g0:g1]).astype(np.int32)
    Ar = cp.linalg.norm(atm_coords[:, None, :] - coords[None, :, :], axis=2)
    Ar = cp.ascontiguousarray(Ar)
    PB = cp.zeros((natm, ngrids))
    a_ptr = (lib.c_null_ptr() if a_factor is None
             else ctypes.cast(a_factor.data.ptr, ctypes.c_void_p))
    libgdft.GDFTbecke_eval_PB(
        ctypes.cast(PB.data.ptr, ctypes.c_void_p), a_ptr,
        ctypes.cast(inv_atom_distance.data.ptr, ctypes.c_void_p),
        ctypes.cast(Ar.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids), ctypes.c_int(natm), ctypes.c_int(scheme_id))
    sPB = PB.sum(axis=0)
    inv_sPB = cp.zeros(ngrids)
    nz = sPB > 1e-14
    inv_sPB[nz] = 1. / sPB[nz]

    scheme = {100: 'becke', 101: 'stratmann'}[scheme_id]
    ra = '_ra' if a_factor is not None else ''
    mod = _module(cp.cuda.Device().id)
    P = cp.zeros((natm, natm, 9))
    U = cp.zeros((natm, natm, 9))
    V = cp.zeros((natm, natm, 9))
    eps = cp.ascontiguousarray(cp.asarray(eps, dtype=np.float64))
    a_arg = a_factor if a_factor is not None else cp.zeros(1)
    args = (P, U, V, eps, coords, qw, atm_coords, a_arg, inv_atom_distance,
            atm_idx, Ar, PB, inv_sPB, np.int32(ngrids), np.int32(natm))
    mod.get_function('d2w_offdiag_' + scheme + ra)(
        ((ngrids + 15) // 16, (natm + 3) // 4, (natm + 3) // 4), (16, 4, 4),
        args)
    mod.get_function('d2w_diag_' + scheme + ra)(
        ((ngrids + 63) // 64, (natm + 3) // 4), (64, 4), args)
    out = P - U - V
    idx = cp.arange(natm)
    out[idx, idx] += U.sum(axis=0)
    return out.reshape(natm, natm, 3, 3)


def _fused(mol, grids, grid_range, eps):
    g0, g1 = grid_range
    if len(eps) != g1 - g0:          # a caller masked eps: keep its algebra
        from gpu4pyscf.hessian.rks import get_d2weight_dAdB
        from gpu4pyscf.lib.cupy_helper import contract
        STATS['stock'] += 1
        return contract('ABdDg,g->ABdD',
                        get_d2weight_dAdB(mol, grids, grid_range), eps)
    return d2w_contract(mol, grids, grid_range, eps)


_PAIR = re.compile(
    r"( +)d2w_dAdB = get_d2weight_dAdB\(_sorted_mol, grids, \(g0,g1\)\)\n"
    r" +d2e \+= contract\(\"ABdDg,g->ABdD\", d2w_dAdB, epsilon\)\n"
    r" +del d2w_dAdB, epsilon\n")
_VV10 = re.compile(
    r"( +)d2w_dAdB = get_d2weight_dAdB\(_sorted_mol, grids, \(g0_full, g1_full\)\)\n"
    r" +d2w_dAdB = d2w_dAdB\[:, :, :, :, rho_nonzero_mask\[g0_full : g1_full\]\]\n"
    r" +g1_nonzero = g0_nonzero \+ d2w_dAdB\.shape\[4\]\n"
    r" +d2e \+= contract\(\"ABdDg,g->ABdD\", d2w_dAdB, (rho_i\[g0_nonzero : g1_nonzero\] \* \(beta \+ E_i\[g0_nonzero : g1_nonzero\]\))\)\n")


def _rebuild(fn, mod):
    src = textwrap.dedent(inspect.getsource(fn))
    src, n1 = _PAIR.subn(
        r"\1d2e += _fastgr_d2w(_sorted_mol, grids, (g0,g1), epsilon)\n"
        r"\1del epsilon\n", src)
    src, n2 = _VV10.subn(
        r"\1_fastgr_mask = rho_nonzero_mask[g0_full : g1_full]\n"
        r"\1g1_nonzero = g0_nonzero + int(_fastgr_mask.sum())\n"
        r"\1_fastgr_eps = cupy.zeros(g1_full - g0_full)\n"
        r"\1_fastgr_eps[_fastgr_mask] = \2\n"
        r"\1d2e += _fastgr_d2w(_sorted_mol, grids, (g0_full, g1_full), "
        r"_fastgr_eps)\n", src)
    if n1 + n2 == 0 or 'get_d2weight_dAdB(' in src:
        return None
    ns = {}
    exec(compile(src, mod.__file__ + ' (fastgridresp)', 'exec'), ns)
    f = ns[fn.__name__]
    g = types.FunctionType(f.__code__, mod.__dict__, f.__name__,
                           f.__defaults__, f.__closure__)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    from gpu4pyscf.hessian import rks as HR, uks as HU
    for mod, names in ((HR, ('_get_exc_deriv2_grid_response',
                             '_get_enlc_deriv2')),
                       (HU, ('_get_exc_deriv2_grid_response',))):
        mod._fastgr_d2w = _fused
        for name in names:
            new = _rebuild(getattr(mod, name), mod)
            if new is not None:
                setattr(mod, name + '_orig', getattr(mod, name))
                setattr(mod, name, new)
    _PATCHED = True


apply_patch()
