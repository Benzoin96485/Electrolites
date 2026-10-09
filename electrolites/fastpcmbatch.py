"""Many densities / charge vectors through ``int1e_grids`` in one integral
pass (catalogue F9, response half).

``gto/int3c1e.py:int1e_grids`` with a stack of densities (or of charge
vectors) loops over them, one full ``(pair, point)`` integral pass each.
``PCM._B_dot_x`` -- the solvent response inside every PCM CPHF, TDDFT-PCM
and polarizability -- calls it with all right-hand sides at once twice per
response (the potential of the densities on the cavity, then the Fock
contribution of the induced charges): 12.6-14.8 % of a patched PCM Hessian
(``docs/KERNEL_TARGETS_1.8.1.md`` §10.6), 126 passes per call at 42 atoms.

Here, for ``n >= MIN_BATCH`` right-hand sides, the plain integrals
``I[g, mu, nu]`` are built once per chunk of cavity points (``fast1e``'s
kernels, ``fast1e._plain_chunk``) and contracted with all of them by GEMM:
``v[i, g] = I[g] . dm_i`` and ``vmat_i = q[i] . I``.  The chunk is sized from
free memory.  Fewer right-hand sides, or a basis ``fast1e`` declines, keep
the per-vector path.  The numbers are the same integrals summed in a
different order.

And F11's cheapest part: ``solvent/hessian/pcm.py:get_dqsym_dx`` -- the
derivative of the symmetrised surface charges -- is computed three times in
one PCM Hessian (``analytical_hess_nuc``, ``analytical_hess_qv`` and
``make_h1``'s ``analytical_grad_vmat``) with the same object, density and
atoms; it is kept for one reuse key (object, atoms, molecule geometry and an
element-wise check of the density), and a copy is handed back.  Bitwise.

Patched where the name is imported: ``gto.int3c1e``, ``solvent.pcm``,
``solvent.hessian.pcm``, ``tdscf.{rhf,uhf}``, ``pop.esp``,
``properties.eda``.  ``FASTPCMBATCH_OFF=1`` keeps GPU4PySCF's loop.
"""
import importlib
import os

import numpy as np
import cupy as cp
from pyscf import lib

from gpu4pyscf.gto import int3c1e as I3
from gpu4pyscf.lib.cupy_helper import get_avail_mem

_OFF = bool(os.environ.get('FASTPCMBATCH_OFF'))
#: right-hand sides from which one integral pass + GEMM is used
MIN_BATCH = int(os.environ.get('FASTPCMBATCH_MIN', 3))
STATS = {'batched': 0, 'rhs': 0, 'per_vector': 0, 'dqdx_built': 0,
         'dqdx_reused': 0}


def stats():
    return dict(STATS)


def _batched(mol, grids, charge_exponents, intopt, dm=None, charges=None):
    from . import fast1e
    ok, _ = fast1e._usable(mol, intopt)
    if not ok:
        return None
    nao = mol.nao
    grids = cp.asarray(grids, order='C')
    ngrids = grids.shape[0]
    if charge_exponents is not None:
        charge_exponents = cp.asarray(charge_exponents, order='C')
        if charge_exponents.size == 1:
            charge_exponents = cp.zeros(ngrids) + charge_exponents
    lmax = fast1e._lmax(intopt)
    mod = fast1e._module(lmax)
    fast1e._push_bpcache(mod, intopt)
    # the chunk, its symmetrisation copy and the cart2sph temporaries
    chunk = int(max(1, min(ngrids, get_avail_mem() * .3 // (3 * nao * nao * 8))))
    if dm is not None:
        rhs = cp.asarray(dm).reshape(-1, nao * nao)
        out = cp.empty((rhs.shape[0], ngrids))
    else:
        q = cp.asarray(charges)
        out = cp.zeros((q.shape[0], nao * nao))
    for p0, p1 in lib.prange(0, ngrids, chunk):
        ints = fast1e._plain_chunk(
            mol, intopt, mod, lmax, grids[p0:p1],
            fast1e._ptr_or_null(charge_exponents, p0, p1), mol.omega)
        ints = ints.reshape(p1 - p0, nao * nao)
        if dm is not None:
            out[:, p0:p1] = rhs @ ints.T
        else:
            out += q[:, p0:p1] @ ints
        ints = None
    STATS['batched'] += 1
    STATS['rhs'] += out.shape[0]
    if dm is not None:
        return out
    return out.reshape(-1, nao, nao)


def int1e_grids(mol, grids, charge_exponents=None, dm=None, charges=None,
                direct_scf_tol=1e-13, intopt=None):
    n = 0
    if dm is not None and dm is not charges and getattr(dm, 'ndim', 0) == 3:
        n = dm.shape[0]
    elif charges is not None and getattr(charges, 'ndim', 0) == 2 and dm is None:
        n = charges.shape[0]
    if n >= MIN_BATCH and mol.omega >= 0:
        if intopt is None:
            intopt = I3.VHFOpt(mol)
            intopt.build(direct_scf_tol, aosym=True)
        r = _batched(mol, grids, charge_exponents, intopt, dm=dm,
                     charges=charges if dm is None else None)
        if r is not None:
            return r
    STATS['per_vector'] += 1
    return I3._int1e_grids_orig(mol, grids, charge_exponents, dm, charges,
                                direct_scf_tol, intopt)


_DQ = {}


def get_dqsym_dx(pcmobj, dm, atmlst, intopt_derivative):
    from gpu4pyscf.solvent.hessian import pcm as HP
    atoms = tuple(int(a) for a in atmlst)
    coords = pcmobj.mol.atom_coords()
    dm_ = cp.asarray(dm)
    hit = _DQ.get('entry')
    if (hit is not None and hit[0] is pcmobj and hit[1] == atoms
            and np.array_equal(hit[2], coords) and hit[3].shape == dm_.shape
            and bool(cp.array_equal(hit[3], dm_))):
        STATS['dqdx_reused'] = STATS.get('dqdx_reused', 0) + 1
        return hit[4].copy()
    out = HP._get_dqsym_dx_orig(pcmobj, dm, atmlst, intopt_derivative)
    _DQ['entry'] = (pcmobj, atoms, coords.copy(), dm_.copy(), out.copy())
    STATS['dqdx_built'] = STATS.get('dqdx_built', 0) + 1
    return out


_TARGETS = ('gpu4pyscf.solvent.pcm', 'gpu4pyscf.solvent.hessian.pcm',
            'gpu4pyscf.tdscf.rhf', 'gpu4pyscf.tdscf.uhf', 'gpu4pyscf.pop.esp',
            'gpu4pyscf.properties.eda')
_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    I3._int1e_grids_orig = I3.int1e_grids
    I3.int1e_grids = int1e_grids
    for name in _TARGETS:
        try:
            m = importlib.import_module(name)
        except ImportError:
            continue
        if getattr(m, 'int1e_grids', None) is I3._int1e_grids_orig:
            m.int1e_grids = int1e_grids
    from gpu4pyscf.solvent.hessian import pcm as HP
    HP._get_dqsym_dx_orig = HP.get_dqsym_dx
    HP.get_dqsym_dx = get_dqsym_dx
    _PATCHED = True


apply_patch()
