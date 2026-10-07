"""The periodic gradient's one-electron term with one density pass (J7).

GPU4PySCF's FFTDF gradients (``pbc/grad/krhf.py``, ``kuhf.py`` and the
Gamma-point ``rhf.py`` / ``uhf.py``) take the electronic part of the local
nuclear (or local pseudopotential) derivative atom by atom:
``hcore_generator(ia)`` evaluates every AO on the whole grid for every k-point
and builds an ``(nk, 3, nao, nao)`` matrix, which ``grad_elec`` then contracts
with the density -- ``natm`` passes over the grid for one number per atom and
direction.  It was 56 % of an 8-atom 2x2x2 k-point gradient
(``docs/KERNEL_TARGETS_1.8.1.md`` §11.6).

The contraction is linear in the density, so

    sum_k tr(h1(ia)_k dm_k) = sum_r v_ia,x(r) rho(r)  -  (the h1 terms)

with ``rho(r) = sum_k sum_ij conj(ao_ki) dm_kji ao_kj`` evaluated **once**,
``v_ia,x`` the same ``ifft`` of ``i G SI_ia vloc_ia(G)`` GPU4PySCF forms per
atom, and the two kinetic/``h1`` corrections ``hcore_generator`` subtracts on
atom ``ia``'s rows and columns taken as row sums of ``h1 * dm^T``.  One grid
pass instead of ``natm``; the arithmetic is otherwise GPU4PySCF's.

**J5** -- the stress tensor's GTH non-local pseudopotential term,
``pbc/grad/rks_stress.py:_get_pp_nonloc_strain_derivatives``, a central
difference over 9 strains x 2 cells of the non-local energy.  GPU4PySCF
evaluates that energy per k-point *per atom*: the projectors' Gaussian on
``G + k`` with host ``eval_gto`` and host ``_qli`` (the same for every atom
of a species), copied to the device atom by atom, then two GEMMs per atom.
`pp_nonloc_energy` builds the projector table once per species and k-point,
applies every atom's structure factor on the device, and does the atoms of a
species as one GEMM and one batched contraction.  Same finite difference,
same displacement, same expressions.

The four ``grad_elec`` are GPU4PySCF's source with the per-atom loop replaced
by `dh1e_local`, bound to their modules' namespaces.  ``FASTPBCGRAD_OFF=1``
keeps GPU4PySCF's.  ``docs/README_fastpbcgrad.md``.
"""
import inspect
import os
import re
import textwrap
import types

import numpy as np
import cupy as cp

from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc.dft import numint as pbc_numint
from gpu4pyscf.pbc.dft import UniformGrids
from gpu4pyscf.pbc.df.aft import get_SI
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.tools.pbc import get_coulG
from gpu4pyscf.pbc.grad import krhf as KRHF

_OFF = bool(os.environ.get('FASTPBCGRAD_OFF'))
STATS = {'ours': 0, 'pp_strain': 0}


def stats():
    return dict(STATS)


def dh1e_local(mf_grad, cell, kpts, dm):
    """``[sum_k einsum('kxij,kji->x', hcore_generator(ia), dm) for ia]`` as a
    device ``(natm, 3)`` array, from one density on the grid.  ``dm`` is
    ``(nk, nao, nao)``."""
    from pyscf.pbc.gto.pseudo.pp import get_vlocG
    STATS['ours'] += 1
    kpts = np.asarray(kpts).reshape(-1, 3)
    dm = cp.asarray(dm).reshape(len(kpts), cell.nao, cell.nao)
    natm = cell.natm
    mesh = cell.mesh
    Gv = cp.asarray(cell.Gv)
    SI = get_SI(cell)

    # rho on the sorted grid, then in mesh order
    ni = pbc_numint.KNumInt()
    grids = UniformGrids(cell)
    rho = []
    for ao_ks, weight, coords in ni.block_loop(cell, grids, 0, kpts,
                                               sort_grids=True):
        ao_ks = cp.asarray(ao_ks)                      # [nk, ng, nao]
        aod = contract('kgj,kji->kgi', ao_ks, dm)
        rho.append((ao_ks.conj() * aod).sum(axis=(0, 2)))
    rho_sorted = cp.concatenate(rho)
    rho_r = cp.empty_like(rho_sorted)
    rho_r[cp.asarray(grids.argsort())] = rho_sorted

    if cell._pseudo:
        vlocG = cp.asarray(get_vlocG(cell))
    else:
        Z = cell.atom_charges()
        coulG = get_coulG(cell, mesh=mesh, Gv=Gv)
    de = cp.empty((natm, 3), dtype=np.complex128)
    for ia in range(natm):
        if cell._pseudo:
            vloc_g = cp.einsum('ga,g,g->ag', Gv, 1j * SI[ia], vlocG[ia])
        else:
            vloc_g = cp.einsum('ga,g,g->ag', Gv, Z[ia] * 1j * SI[ia], coulG)
        vloc_r = tools.ifft(vloc_g, mesh).real          # [3, ngrids]
        de[ia] = vloc_r @ rho_r

    # hcore_generator's  hcore[:,:,A,:] -= h1[:,:,A,:]  and
    #                    hcore[:,:,:,A] -= h1[:,:,A,:]^H
    h1 = cp.asarray(KRHF.get_hcore(cell, kpts))         # [nk, 3, nao, nao]
    t_row = contract('kxij,kji->xi', h1, dm)
    t_col = contract('kxji,kji->xj', h1.conj(), dm)
    t = t_row + t_col
    aoslices = cell.aoslice_by_atom()
    for ia in range(natm):
        p0, p1 = aoslices[ia, 2:]
        de[ia] -= t[:, p0:p1].sum(axis=1)
    return de.real


# --------------------------------------------------------------------------
# J5: the non-local PP energy of one (strained) cell
# --------------------------------------------------------------------------
#: atoms of a species per batched GEMM
ATOM_BATCH = 64


def _projectors(cell, symb, Gk, G_rad):
    """``(P, H)``: the species' projectors on ``Gk`` as rows ``(l, i, m)``,
    real, exactly as ``rks_stress.eval_pp_nonloc`` fills its buffer, and the
    block-diagonal ``h`` with ``H[(i,m),(j,m)] = hl[i,j]``."""
    from pyscf import gto
    from pyscf.gto.mole import ANG_OF, NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF
    from pyscf.pbc.gto import pseudo
    fakemol = gto.Mole()
    fakemol._atm = np.zeros((1, gto.ATM_SLOTS), dtype=np.int32)
    fakemol._bas = np.zeros((1, gto.BAS_SLOTS), dtype=np.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = np.zeros(ptr + 10)
    fakemol._bas[0, NPRIM_OF] = 1
    fakemol._bas[0, NCTR_OF] = 1
    fakemol._bas[0, PTR_EXP] = ptr + 3
    fakemol._bas[0, PTR_COEFF] = ptr + 4
    rows, hs = [], []
    for l, proj in enumerate(cell._pseudo[symb][5:]):
        rl, nl, hl = proj
        if nl <= 0:
            continue
        fakemol._bas[0, ANG_OF] = l
        fakemol._env[ptr + 3] = .5 * rl**2
        fakemol._env[ptr + 4] = rl**(l + 1.5) * np.pi**1.25
        part = fakemol.eval_gto('GTOval', Gk).T          # [nf, ngrids]
        for k in range(nl):
            rows.append(part * pseudo.pp._qli(G_rad * rl, l, k))
        nf = 2 * l + 1
        hs.append(np.kron(np.asarray(hl), np.eye(nf)))
    if not rows:
        return None, None
    n = sum(h.shape[0] for h in hs)
    H = np.zeros((n, n))
    p = 0
    for h in hs:
        H[p:p + len(h), p:p + len(h)] = h
        p += len(h)
    return np.concatenate(rows).astype(np.float64), H


def pp_nonloc_energy(cell, mesh, dm_kpts, kpts):
    """``rks_stress.eval_pp_nonloc(cell)``: the GTH non-local energy of the
    densities ``dm_kpts`` (device, ``(nk, nao, nao)``) at ``kpts``
    (fractional, i.e. ``kpts . a^T``)."""
    from gpu4pyscf.pbc.df import ft_ao
    from gpu4pyscf.pbc.df.ft_ao import SortedGTO
    from pyscf import lib
    vol = cell.vol
    # sorted once per cell rather than once per k-point, and one output
    # buffer for every k-point: ft_ao otherwise re-sorts the basis and
    # allocates (nao x ngrids) complex anew per call
    sorted_cell = SortedGTO.from_cell(cell)
    buf = cp.empty((int(sorted_cell.ao_loc[-1]), len(cell.get_Gv(mesh))),
                   dtype=np.complex128)
    b = cell.reciprocal_vectors(norm_to=1)
    Gv = cell.get_Gv(mesh)
    SI = cp.asarray(get_SI(cell, mesh=mesh))
    species = {}
    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if symb in cell._pseudo:
            species.setdefault(symb, []).append(ia)
    e = cp.zeros(())
    for k, dm in enumerate(dm_kpts):
        kpt = kpts[k].dot(b)
        Gk = Gv + kpt
        G_rad = np.sqrt(np.einsum('gx,gx->g', Gk, Gk))
        # the transform in the sorted Cartesian basis, into the reused
        # buffer; the basis transformation is applied to the small product
        # below instead of to this (ngrids x nao) array
        aokG_s = ft_ao.ft_ao(sorted_cell, Gv, kpt=kpt, sort_output=False,
                             out=buf)
        for symb, atoms in species.items():
            P, H = _projectors(cell, symb, Gk, G_rad)
            if P is None:
                continue
            P = cp.asarray(P)
            HT = cp.asarray(H.T)
            np_ = len(P)
            for a0 in range(0, len(atoms), ATOM_BATCH):
                ats = atoms[a0:a0 + ATOM_BATCH]
                A = P[None] * SI[ats].conj()[:, None, :]    # [na, np, ng]
                B = A.reshape(-1, A.shape[-1]) @ aokG_s    # [na*np, nao_s]
                B = sorted_cell.apply_CT_dot(cp.ascontiguousarray(B.T), axis=0)
                B = B.T * (1 / vol)**.5
                X = (B @ dm).reshape(len(ats), np_, -1)
                B = B.reshape(len(ats), np_, -1)
                rho = cp.einsum('apn,aqn->apq', X, B.conj()).real
                e += (rho * HT[None]).sum()
    return float(e) / (len(dm_kpts) * vol)


def pp_nonloc_strain_derivatives(cell, mesh, dm_kpts, kpts=None):
    """``rks_stress._get_pp_nonloc_strain_derivatives`` with
    `pp_nonloc_energy` for its energy."""
    from gpu4pyscf.pbc.grad.rks_stress import _finite_diff_cells
    STATS['pp_strain'] = STATS.get('pp_strain', 0) + 1
    if kpts is None:
        dm_kpts = cp.asarray(dm_kpts)
        assert dm_kpts.ndim == 2
        dm_kpts = dm_kpts[None, :, :]
        kpts = np.zeros((1, 3))
    dm_kpts = cp.asarray(dm_kpts)
    scaled_kpts = np.asarray(kpts).dot(cell.lattice_vectors().T)
    disp = max(1e-5, (cell.precision * .1)**.5)
    out = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp)
            e1 = pp_nonloc_energy(cell1, mesh, dm_kpts, scaled_kpts)
            e2 = pp_nonloc_energy(cell2, mesh, dm_kpts, scaled_kpts)
            out[i, j] = (e1 - e2) / (2 * disp)
    return out


# --------------------------------------------------------------------------
# the four grad_elec, with their per-atom loop replaced
# --------------------------------------------------------------------------
_LOOP_K = re.compile(
    r"( +)hcore_deriv = mf_grad\.hcore_generator\(cell, kpts\)\n"
    r" +dh1e = cp\.empty\(\[natm, 3\]\)\n"
    r" +for ia in range\(natm\):\n"
    r" +h1ao = hcore_deriv\(ia\)\n"
    r" +dh1e\[ia\] = cp\.einsum\('kxij,kji->x', h1ao, (\w+)\)\.real\n")
_LOOP_G = re.compile(
    r"( +)from gpu4pyscf\.pbc\.grad\.krhf import hcore_generator\n"
    r" +hcore_deriv = hcore_generator\(self, cell, np\.zeros\(\(1, 3\)\)\)\n"
    r" +dh1e = cp\.empty\(\[cell\.natm, 3\]\)\n"
    r" +for ia in range\(cell\.natm\):\n"
    r" +h1ao = hcore_deriv\(ia\)\n"
    r" +dh1e\[ia\] = cp\.einsum\('xij,ji->x', h1ao\[0\], (\w+)\)\.real\n")


def _rebuild(fn, mod):
    src = textwrap.dedent(inspect.getsource(fn))
    src, n1 = _LOOP_K.subn(
        r"\1dh1e = _fastpbc_dh1e(mf_grad, cell, kpts, \2)\n", src)
    src, n2 = _LOOP_G.subn(
        r"\1dh1e = _fastpbc_dh1e(self, cell, np.zeros((1, 3)), \2[None])\n",
        src)
    if n1 + n2 != 1:
        return None
    ns = {}
    exec(compile(src, mod.__file__ + ' (fastpbcgrad)', 'exec'), ns)
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
    import importlib
    from gpu4pyscf.pbc.grad import rks_stress as RKSS
    RKSS._get_pp_nonloc_strain_derivatives_orig = \
        RKSS._get_pp_nonloc_strain_derivatives
    for name in ('rks_stress', 'krks_stress', 'kuks_stress', 'uks_stress',
                 'krhf_stress'):
        try:
            m = importlib.import_module('gpu4pyscf.pbc.grad.' + name)
        except ImportError:
            continue
        if hasattr(m, '_get_pp_nonloc_strain_derivatives'):
            m._get_pp_nonloc_strain_derivatives = pp_nonloc_strain_derivatives
    from gpu4pyscf.pbc.grad import kuhf as KUHF, rhf as RHF, uhf as UHF
    for mod, owner in ((KRHF, KRHF), (KUHF, KUHF),
                       (RHF, RHF.Gradients), (UHF, UHF.Gradients)):
        mod._fastpbc_dh1e = dh1e_local
        fn = owner.grad_elec if owner is not mod else mod.grad_elec
        new = _rebuild(fn, mod)
        if new is None:
            continue
        if owner is mod:
            mod._grad_elec_orig = mod.grad_elec
            mod.grad_elec = new
            mod.Gradients.grad_elec = new
        else:
            owner._grad_elec_orig = fn
            owner.grad_elec = new
    _PATCHED = True


apply_patch()
