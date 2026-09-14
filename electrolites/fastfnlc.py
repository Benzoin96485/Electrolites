"""
The VV10 non-local-correlation **response** term, which is 80 % of a
wB97M-V polarizability.

Importing this module monkey-patches
``gpu4pyscf.hessian.rks.nr_rks_fnlc_mo``.  That function is the NLC part of
the coupled-perturbed solve: ``scf/_response_functions.py`` adds it to every
CPKS matrix-vector product for an NLC functional, and
``hessian/rks.py:get_veff_resp_mo`` adds it to every CPHF one.  So it is on
the path of a **polarizability, NMR shielding, C6 coefficient and analytic
Hessian** with wB97M-V, wB97X-V, B97M-V or VV10.  (TDDFT excludes it:
``tdscf/rhf.py:408 gen_response`` passes ``with_nlc=not self.exclude_nlc``
and ``exclude_nlc`` is on by default.)

It is not an entry in ``instructions/06``.  Measured
(``benchmarks/response_anatomy.py``, penicillin V/def2-SVP, level-3 grids,
one A100): a wB97M-V polarizability spends **124.74 s of 154.67 s -- 80.6 %**
in it, against 12.7 % in the J and K build and 6.4 % in the XC response
kernel.  It is the largest single cost in any response property this
repository has profiled.

**What this module changes, and what it does not.**  The two
``O(ngrids^2)`` double sums, the VV10 parameters, the grid, the zero-density
threshold and every factor are GPU4PySCF's; what changes is how often the
ground-state half runs and in what order the potential contraction is summed.
Change 1 is exact -- ``tests/test_fnlc.py`` asserts **bit-identical** results
on the ``return_in_mo=True`` path, which uses the cache and keeps GPU4PySCF's
contraction -- and change 2 reassociates one grid sum, which agrees to
**3.3e-15** relative over that suite.

1. **The ground-state half of the function is computed once per solve
   instead of once per iteration.**  Everything from the density on the NLC
   grid down to ``f_gamma_i`` -- including
   ``VXC_vv10nlc_hess_eval_UWABCE``, an ``O(ngrids^2)`` double sum -- depends
   only on ``mo_coeff`` and ``mo_occ``, which a CPKS or CPHF solve holds fixed
   for its whole run.  GPU4PySCF recomputes all of it on every call.
   Measured at **27.5 %** of a call (`benchmarks/fnlc_probe.py`): 1.965 s of
   7.134 s, of which the double sum alone is 1.869 s.  This is the same
   phenomenon as catalogue entry **D8**, and unlike D8 it is worth taking:
   D8's repeated work is 2.2 % of a polarizability, this one's is 22 %.

2. **The potential contraction is batched over the perturbed densities.**
   GPU4PySCF issues five kernels per (grid block, density) to build
   ``V_munu``; here the four AO components go through one weighted-AO kernel
   and one GEMM for all of them at once, and the ``V + V^T`` that the
   ``nabla`` term needs is folded into a single ``transpose_sum``.  The
   identity used is
   ``V = S + G + G^T = (M + M^T)/2`` with ``M = (ao_0 f_rho + 2 nabla_dot) . ao_0^T``,
   which is exact because ``S`` is symmetric.

**What is left, and it is the larger half.**
``VXC_vv10nlc_hess_eval_f_t`` is **64.7 %** of a call and depends on the
perturbed densities, so it cannot be cached.  It is an ``O(ngrids^2)`` fp64
double sum over 519 680 points -- 2.7e11 pairs in 4.61 s -- and the three
techniques ``fastnlc`` measured on the *ground-state* VV10 double sum apply to
it verbatim: a MUFU-seeded reciprocal (1.22x), single precision beyond 10 Bohr
(2.82x) and an error-budgeted grid trim (3.92x together).  Writing it needs
GPU4PySCF's CUDA source for that kernel, which a wheel does not ship, so it is
the next round rather than this one.  ``docs/README_fastfnlc.md`` §4.

**The unrestricted case, and why it is not a corner.**  ``V_munu`` here is
spin-free: VV10 is a functional of the total density, so GPU4PySCF's own
``hessian.uks.nr_uks_fnlc_mo`` sums the two perturbed densities and forwards
to the restricted function, and the only thing the spin index changes is that
an MO-basis answer is projected onto two MO bases instead of one.  So this
module covers UKS with the ground-state density summed over spins in
``_ground_state`` and one projection per spin at the end.

Reaching it needs a second patch.  ``hessian/uks.py`` imports
``nr_rks_fnlc_mo`` **by name at import time**, so its ``nr_uks_fnlc_mo`` calls
that binding rather than ``hessian.rks``'s attribute -- and patching only
``hessian.rks`` leaves every open-shell caller on GPU4PySCF's own code while
still returning the right answer.  ``apply_patch`` sets both.

Who those callers are is worth stating, because the restricted and
unrestricted response paths do **not** agree about this term:
``tdscf/rhf.py`` sets ``exclude_nlc = True`` and passes
``with_nlc=not self.exclude_nlc`` into ``gen_response``, so a closed-shell
TDA/TDDFT run skips the VV10 response entirely.  ``tdscf/uhf.py``'s
``TDBase.gen_response`` -- which ``tdscf/uks.py``'s ``TDA`` and ``TDDFT``
inherit -- forwards to ``_scf.gen_response(...)`` with **no** ``with_nlc``
argument, and its default is ``True``.  An open-shell TDA or TDDFT run with
wB97M-V therefore pays this term on every Davidson iteration where the
closed-shell one pays none of it, which makes UKS TDDFT the largest consumer
of this function in the package and not, as the restricted side suggests, a
polarizability-only path.

**Limits, all of which fall through to GPU4PySCF rather than fail.**
The ``return_in_mo=True`` path (the Hessian's CPHF solve asks for the response
in the MO basis) keeps GPU4PySCF's per-density loop for the potential
contraction but still gets the cache; a coefficient *tuple* rather than an
array; and more than one GPU, as everywhere here.
"""
import ctypes
import functools
import os

import cupy
import numpy

from gpu4pyscf.dft import numint
from gpu4pyscf.dft.numint import NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD
from gpu4pyscf.hessian import rks as HR
from gpu4pyscf.hessian import uks as HU
from gpu4pyscf.lib.cupy_helper import (
    add_sparse, batched_vec_norm2, contract, get_avail_mem, ndarray,
    take_last2d, transpose_sum)
from gpu4pyscf.__config__ import num_devices

from ._paths import KERNEL_DIR as HERE

libgdft = numint.libgdft

#: FASTFNLC_CACHE=0 recomputes the ground-state half on every call, which is
#: the ablation for change 1.  FASTFNLC_BATCH=0 puts GPU4PySCF's per-density
#: potential loop back, which is the ablation for change 2.
CACHE = os.environ.get('FASTFNLC_CACHE', '1') != '0'
BATCH = os.environ.get('FASTFNLC_BATCH', '1') != '0'

#: Budget, in the same units as ``rho*weight``, for the grid points left out
#: of the two ``O(ngrids^2)`` pair sums -- ``fastnlc``'s ``FASTNLC_BUDGET``
#: criterion applied to the *response* term.  **Off by default**, unlike
#: ``fastnlc``'s, and §4d of ``docs/README_fastfnlc.md`` says why: the pair
#: sums here shrink quadratically in the point count so the win is real, but
#: what it perturbs is a response operator rather than an energy, this
#: module's correctness guarantee is currently *bit-identity* against
#: GPU4PySCF on the MO path, and one measurement is not the suite that
#: guarantee is worth trading for.  Set it to 1e-12 to get ``fastnlc``'s
#: default.
BUDGET = float(os.environ.get('FASTFNLC_BUDGET', 0.0))

_WX = int(os.environ.get('FASTFNLC_WX', 32))
_TSY = int(os.environ.get('FASTFNLC_TSY', 4))
_UPB = int(os.environ.get('FASTFNLC_UPB', 32))


@functools.lru_cache(maxsize=4)
def _kernel(name):
    # the weighted-AO kernel of `fastxcfxc`, compiled here rather than
    # imported from that module, because importing it would apply its patch
    # as a side effect.  CuPy caches NVRTC output on the source hash, so the
    # two modules share one compile.
    src = open(os.path.join(HERE, 'fastxcfxc.cu')).read()
    mod = cupy.RawModule(code=src, backend='nvrtc',
                         options=('-std=c++17', f'-DWX={_WX}',
                                  f'-DTSY={_TSY}'))
    return mod.get_function(name)

#: One entry, holding a reference to the `mo_coeff` and `mo_occ` it was built
#: from so that identity is a sound cache key: while the entry is live those
#: arrays cannot be freed and their addresses cannot be reused.
_CACHE = {}


def _trim(keep, rho0, weights):
    """Drop the grid points that carry the least density-weight.

    ``fastnlc._trim``'s criterion, restated for this function rather than
    imported from that module (importing it would apply its patch as a side
    effect).  Every term of both pair sums here carries a factor
    ``rho_j * w_j``, so leaving a set of points out of them perturbs the
    result by at most a bounded multiple of ``sum |rho_j w_j|`` over that set
    -- which is what the budget names.  The same set leaves the outer sum and
    the potential contraction, where a point's share is bounded by the same
    quantity.

    The saving is quadratic in the point count, because both sums are.  How
    many points a budget drops is a property of the system, not of the
    budget: ``docs/README_fastnlc.md`` §3 measures 14-15 % on a level-3 grid
    of an organic cluster, where the diffuse tails of many atomic grids carry
    almost no density-weight, and §4d here measures far less on a small
    metal complex, whose grid is compact and most of whose points matter.
    """
    rw = cupy.abs(rho0 * weights)
    srt = cupy.sort(rw[keep])
    k = min(int(cupy.searchsorted(cupy.cumsum(srt), BUDGET)), srt.size - 1)
    if k <= 0:
        return keep
    cut = float(srt[k - 1])
    out = keep & (rw > cut)
    # the threshold form of the cut can take extra points along if |rho*w|
    # ties at it, so the total actually left out is checked, not assumed
    drop = float(cupy.where(out, 0., rw * keep).sum())
    if drop > 4 * BUDGET:
        return keep
    return out


def _ground_state(mf, mo_coeff, mo_occ, ni, opt, grids):
    """The half of the function that only the ground state enters.

    Returned verbatim from the cache when `mo_coeff` and `mo_occ` are the same
    objects as last time, which is what a CPKS or CPHF solve passes on every
    one of its iterations.
    """
    # `opt` is in the key because the cached `mo_sorted` is in its AO order
    key = (id(mf), id(mo_coeff), id(mo_occ), id(opt), grids.coords.shape[0])
    hit = _CACHE.get(key)
    if CACHE and hit is not None:
        mo_ref, occ_ref, payload = hit
        if mo_ref is mo_coeff and occ_ref is mo_occ:
            return payload

    _sorted_mol = opt._sorted_mol
    if numint.libxc.is_nlc(mf.xc):
        xc_code = mf.xc
    else:
        xc_code = mf.nlc
    nlc_coefs = ni.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, _fac = nlc_coefs[0]
    kappa_prefactor = nlc_pars[0] * 1.5 * numpy.pi * (9 * numpy.pi)**(-1.0/6.0)
    C_in_omega = nlc_pars[1]

    mo_sorted = opt.sort_orbitals(mo_coeff, axis=[mo_coeff.ndim-2])
    ngrids_full = grids.coords.shape[0]
    rho_drho = cupy.empty([4, ngrids_full])
    g1 = 0
    for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv=1,
                                            strict_grid_order=True):
        g0, g1 = g1, g1 + weight.size
        if mo_sorted.ndim == 2:
            rho_drho[:, g0:g1] = numint.eval_rho2(
                _sorted_mol, ao, mo_sorted[idx, :], mo_occ, None, 'GGA')
        else:
            # unrestricted: everything below the density is a functional of
            # the *total* density, so the two spins are summed here and the
            # rest of the function never sees a spin index
            rho_drho[:, g0:g1] = numint.eval_rho2(
                _sorted_mol, ao, mo_sorted[0, idx, :], mo_occ[0], None, 'GGA')
            rho_drho[:, g0:g1] += numint.eval_rho2(
                _sorted_mol, ao, mo_sorted[1, idx, :], mo_occ[1], None, 'GGA')
    assert g1 == ngrids_full

    rho_i = rho_drho[0, :]
    rho_nonzero_mask = cupy.logical_and(
        rho_i >= NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD,
        cupy.abs(grids.weights) > 1e-14,
    )
    if BUDGET > 0:
        rho_nonzero_mask = _trim(rho_nonzero_mask, rho_i, grids.weights)
    rho_i = rho_i[rho_nonzero_mask]
    nabla_rho_i = rho_drho[1:4, rho_nonzero_mask]
    grids_coords = cupy.ascontiguousarray(grids.coords[rho_nonzero_mask, :])
    grids_weights = grids.weights[rho_nonzero_mask]
    ngrids = grids_coords.shape[0]
    gamma_i = batched_vec_norm2(nabla_rho_i.T)

    stream = cupy.cuda.get_current_stream()
    omega_i = cupy.empty(ngrids)
    domega_drho_i = cupy.empty(ngrids)
    domega_dgamma_i = cupy.empty(ngrids)
    d2omega_drho2_i = cupy.empty(ngrids)
    d2omega_dgamma2_i = cupy.empty(ngrids)
    d2omega_drho_dgamma_i = cupy.empty(ngrids)
    libgdft.VXC_vv10nlc_hess_eval_omega_derivative(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(domega_drho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(domega_dgamma_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_drho2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_dgamma2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_drho_dgamma_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(gamma_i.data.ptr, ctypes.c_void_p),
        ctypes.c_double(C_in_omega),
        ctypes.c_int(ngrids))
    kappa_i = kappa_prefactor * rho_i**(1.0/6.0)
    dkappa_drho_i = (kappa_prefactor * (1.0/6.0)) * rho_i**(-5.0/6.0)
    d2kappa_drho2_i = (kappa_prefactor * (-5.0/36.0)) * rho_i**(-11.0/6.0)

    rho_weight_i = rho_i * grids_weights
    U_i = cupy.empty(ngrids)
    W_i = cupy.empty(ngrids)
    A_i = cupy.empty(ngrids)
    B_i = cupy.empty(ngrids)
    C_i = cupy.empty(ngrids)
    E_i = cupy.empty(ngrids)            # not used, as upstream
    libgdft.VXC_vv10nlc_hess_eval_UWABCE(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(U_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(W_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(A_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(B_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(C_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(E_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_weight_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids))
    del rho_weight_i, E_i, rho_drho

    payload = dict(
        rho_i=rho_i, nabla_rho_i=nabla_rho_i, gamma_i=gamma_i,
        grids_coords=grids_coords, grids_weights=grids_weights,
        rho_nonzero_mask=rho_nonzero_mask, ngrids=ngrids,
        ngrids_full=ngrids_full, mo_sorted=mo_sorted,
        omega_i=omega_i, kappa_i=kappa_i, dkappa_drho_i=dkappa_drho_i,
        d2kappa_drho2_i=d2kappa_drho2_i,
        domega_drho_i=domega_drho_i, domega_dgamma_i=domega_dgamma_i,
        d2omega_drho2_i=d2omega_drho2_i,
        d2omega_dgamma2_i=d2omega_dgamma2_i,
        d2omega_drho_dgamma_i=d2omega_drho_dgamma_i,
        U_i=U_i, W_i=W_i, A_i=A_i, B_i=B_i, C_i=C_i,
        f_gamma_i=rho_i * domega_dgamma_i * W_i,
    )
    if CACHE:
        _CACHE.clear()
        _CACHE[key] = (mo_coeff, mo_occ, payload)
    return payload


def _nr_rks_fnlc_mo(mf, mol, mo_coeff, mo_occ, dm1s, return_in_mo=True):
    nao = mol.nao
    output_in_2d = False
    if dm1s.ndim == 2:
        assert dm1s.shape == (nao, nao)
        dm1s = dm1s.reshape((1, nao, nao))
        output_in_2d = True
    else:
        assert dm1s.ndim == 3

    grids = mf.nlcgrids
    if grids.coords is None:
        grids.build()
    n_dm1 = dm1s.shape[0]

    ni = mf._numint
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    mol = None

    gs = _ground_state(mf, mo_coeff, mo_occ, ni, opt, grids)
    rho_i = gs['rho_i']
    nabla_rho_i = gs['nabla_rho_i']
    grids_coords = gs['grids_coords']
    grids_weights = gs['grids_weights']
    rho_nonzero_mask = gs['rho_nonzero_mask']
    ngrids = gs['ngrids']
    ngrids_full = gs['ngrids_full']
    mo_sorted = gs['mo_sorted']
    f_gamma_i = gs['f_gamma_i']
    stream = cupy.cuda.get_current_stream()

    dm1s_sorted = opt.sort_orbitals(dm1s, axis=[1, 2])
    dm_mask_buf = cupy.empty(nao * nao)
    dm1s = None

    # ``V_munu`` below is spin-free, so the unrestricted case differs only in
    # what it is projected onto: one MO basis per spin when the caller asks
    # for the response in the MO basis, and nothing at all when it does not.
    uks = mo_sorted.ndim == 3
    if return_in_mo:
        mos = list(mo_sorted) if uks else [mo_sorted]
        occs = list(mo_occ) if uks else [mo_occ]
        moccs = [m[:, o > 0] for m, o in zip(mos, occs)]
        vmats = [cupy.zeros([n_dm1, m.shape[1], mo.shape[1]])
                 for m, mo in zip(mos, moccs)]
    else:
        vmats = [cupy.zeros([n_dm1, nao, nao])]
    vmat = vmats[0]

    available_gpu_memory = int(get_avail_mem() * 0.25)
    fxc_nbytes_per_dm1 = ((1*6 + 3*2) * ngrids + (1*2 + 3*2) * ngrids_full) * 8
    ndm1_per_batch = int(available_gpu_memory / fxc_nbytes_per_dm1)
    if ndm1_per_batch < 6:
        raise MemoryError(
            'Out of GPU memory for NLC response (orbital hessian), available '
            f'gpu memory = {get_avail_mem()} bytes, nao = {nao}, '
            f'natm = {_sorted_mol.natm}, ngrids (nonzero rho) = {ngrids}')
    ndm1_per_batch = (ndm1_per_batch + 6 - 1) // 6 * 6

    # fixed scratch for the batched potential contraction: allocating it per
    # grid block is what `docs/README_fastxchess.md` 3b measured at 48 % of
    # that function
    nsub_max = max(len(x[1]) for x in grids.get_non0ao_idx(opt))
    blk = numint.MIN_BLK_SIZE
    aow_buf = wv_buf = v_buf = None

    for i_dm1_batch in range(0, n_dm1, ndm1_per_batch):
        n_dm1_batch = min(ndm1_per_batch, n_dm1 - i_dm1_batch)
        if BATCH and not return_in_mo and aow_buf is None:
            aow_buf = cupy.empty(n_dm1_batch * nsub_max * blk)
            wv_buf = cupy.empty(n_dm1_batch * 4 * blk)
            v_buf = cupy.empty(n_dm1_batch * nsub_max * nsub_max)

        rho_drho_t = cupy.empty([n_dm1_batch, 4, ngrids_full])
        g1 = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv=1,
                                                strict_grid_order=True):
            g0, g1 = g1, g1 + weight.shape[0]
            for i_dm in range(n_dm1_batch):
                dm1_masked = take_last2d(dm1s_sorted[i_dm + i_dm1_batch],
                                         idx, out=dm_mask_buf)
                rho_drho_t[i_dm, :, g0:g1] = numint.eval_rho(
                    _sorted_mol, ao, dm1_masked, xctype='NLC', hermi=0)
        assert g1 == ngrids_full

        rho_drho_t = rho_drho_t[:, :, rho_nonzero_mask]
        rho_t_i = rho_drho_t[:, 0, :]
        nabla_rho_t_i = rho_drho_t[:, 1:4, :]
        gamma_t_i = nabla_rho_i[0, :] * nabla_rho_t_i[:, 0, :] \
            + nabla_rho_i[1, :] * nabla_rho_t_i[:, 1, :] \
            + nabla_rho_i[2, :] * nabla_rho_t_i[:, 2, :]
        gamma_t_i *= 2
        del rho_drho_t

        rho_t_i = cupy.ascontiguousarray(rho_t_i)
        gamma_t_i = cupy.ascontiguousarray(gamma_t_i)
        f_rho_t_i = cupy.empty([n_dm1_batch, ngrids], order='C')
        f_gamma_t_i = cupy.empty([n_dm1_batch, ngrids], order='C')
        libgdft.VXC_vv10nlc_hess_eval_f_t(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(f_rho_t_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(f_gamma_t_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_weights.data.ptr, ctypes.c_void_p),
            ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(gs['omega_i'].data.ptr, ctypes.c_void_p),
            ctypes.cast(gs['kappa_i'].data.ptr, ctypes.c_void_p),
            ctypes.cast(gs['U_i'].data.ptr, ctypes.c_void_p),
            ctypes.cast(gs['W_i'].data.ptr, ctypes.c_void_p),
            ctypes.cast(gs['A_i'].data.ptr, ctypes.c_void_p),
            ctypes.cast(gs['B_i'].data.ptr, ctypes.c_void_p),
            ctypes.cast(gs['C_i'].data.ptr, ctypes.c_void_p),
            ctypes.cast(gs['domega_drho_i'].data.ptr, ctypes.c_void_p),
            ctypes.cast(gs['domega_dgamma_i'].data.ptr, ctypes.c_void_p),
            ctypes.cast(gs['dkappa_drho_i'].data.ptr, ctypes.c_void_p),
            ctypes.cast(gs['d2omega_drho2_i'].data.ptr, ctypes.c_void_p),
            ctypes.cast(gs['d2omega_dgamma2_i'].data.ptr, ctypes.c_void_p),
            ctypes.cast(gs['d2omega_drho_dgamma_i'].data.ptr,
                        ctypes.c_void_p),
            ctypes.cast(gs['d2kappa_drho2_i'].data.ptr, ctypes.c_void_p),
            ctypes.cast(rho_t_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(gamma_t_i.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(n_dm1_batch))
        del rho_t_i, gamma_t_i

        fxc_rho = f_rho_t_i * grids_weights
        del f_rho_t_i
        fxc_gamma = contract('dg,tg->tdg', nabla_rho_i, f_gamma_t_i)
        del f_gamma_t_i
        fxc_gamma += nabla_rho_t_i * f_gamma_i
        del nabla_rho_t_i
        fxc_gamma = 2 * fxc_gamma * grids_weights

        g0_full = g0_nonzero = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv=1,
                                                strict_grid_order=True):
            g1_full = g0_full + weight.shape[0]
            ao = ao[:, :, rho_nonzero_mask[g0_full:g1_full]]
            if ao.size == 0:
                g0_full = g1_full
                continue
            g1_nonzero = g0_nonzero + ao.shape[-1]
            nsub, ngb = ao.shape[1], ao.shape[2]
            f_rho = fxc_rho[:, g0_nonzero:g1_nonzero]
            f_gam = fxc_gamma[:, :, g0_nonzero:g1_nonzero]

            if BATCH and not return_in_mo:
                # V = S + G + G^T with S = ao0 . (ao0 f_rho)^T symmetric and
                # G = ao0 . nabla_dot^T, so V = (M + M^T)/2 for
                # M = (ao0 f_rho + 2 nabla_dot) . ao0^T -- one weighted-AO
                # pass and one GEMM for the whole batch.
                wv = ndarray((n_dm1_batch, 4, ngb), buffer=wv_buf)
                wv[:, 0] = f_rho
                cupy.multiply(f_gam, 2.0, out=wv[:, 1:4])
                aow = ndarray((n_dm1_batch * nsub, ngb), buffer=aow_buf)
                _kernel('fxc_scale_gga')(
                    ((ngb + _WX - 1) // _WX,
                     (n_dm1_batch + _TSY - 1) // _TSY,
                     (nsub + _UPB - 1) // _UPB), (_WX, _TSY),
                    (aow, wv, ao, numpy.int32(n_dm1_batch),
                     numpy.int32(nsub), numpy.int32(ngb), numpy.int32(_UPB),
                     numpy.int32(4), numpy.int32(0), numpy.int32(0)))
                vblk = ndarray((n_dm1_batch * nsub, nsub), buffer=v_buf)
                cupy.dot(aow, ao[0].T, out=vblk)
                vblk = vblk.reshape(n_dm1_batch, nsub, nsub)
                transpose_sum(vblk)
                vblk *= .5
                add_sparse(vmat[i_dm1_batch:i_dm1_batch + n_dm1_batch],
                           vblk, idx)
            else:
                for i_dm in range(n_dm1_batch):
                    V_munu = contract('ig,jg->ij', ao[0],
                                      ao[0] * f_rho[i_dm, :])
                    nabla_fxc_dot_nabla_ao = contract(
                        'dg,dig->ig', f_gam[i_dm, :, :], ao[1:4])
                    V_munu_gamma = contract('ig,jg->ij', ao[0],
                                            nabla_fxc_dot_nabla_ao)
                    del nabla_fxc_dot_nabla_ao
                    V_munu += V_munu_gamma
                    V_munu += V_munu_gamma.T
                    del V_munu_gamma
                    if return_in_mo:
                        for m, mo, vm in zip(mos, moccs, vmats):
                            vm[i_dm + i_dm1_batch, :, :] += \
                                m[idx, :].T @ V_munu @ mo[idx, :]
                    else:
                        add_sparse(vmat[i_dm + i_dm1_batch], V_munu, idx)

            g0_nonzero = g1_nonzero
            g0_full = g1_full
        assert g1_full == ngrids_full
        assert g1_nonzero == ngrids

    if return_in_mo and uks:
        # GPU4PySCF's own shape for this case: a two-element tuple
        if output_in_2d:
            return tuple(v[0] for v in vmats)
        return tuple(vmats)
    if not return_in_mo:
        vmat = opt.unsort_orbitals(vmat, axis=[1, 2])
    if output_in_2d:
        vmat = vmat[0]
    return vmat


_PATCHED = False

#: How many calls this module answered and how many it handed back, since
#: import.  A module that is allowed to decline returns the right answer
#: either way, so nothing numerical can tell the two apart -- which is how the
#: `hessian/uks.py` binding above kept every open-shell caller on GPU4PySCF's
#: code through a round of benchmarking.  Assert on this.
STATS = {'ours': 0, 'stock': 0}


def stats():
    """``{'ours': n, 'stock': m}`` since import.  See STATS."""
    return dict(STATS)


def _dispatch(mf, mol, mo_coeff, mo_occ, dm1s, return_in_mo=True):
    # `ndim` rather than a type test because `_gen_uhf_response` can hand a
    # *tuple* of two coefficient matrices along (a SymAdaptedUHF reference
    # converted with `to_gpu`), which has no `ndim` and falls through.
    ok = (num_devices == 1 and getattr(mo_coeff, 'ndim', 0) in (2, 3)
          and getattr(dm1s, 'ndim', 0) in (2, 3))
    STATS['ours' if ok else 'stock'] += 1
    if ok:
        return _nr_rks_fnlc_mo(mf, mol, mo_coeff, mo_occ, dm1s, return_in_mo)
    return HR._nr_rks_fnlc_mo_orig(mf, mol, mo_coeff, mo_occ, dm1s,
                                   return_in_mo)


def apply_patch():
    global _PATCHED
    if not _PATCHED:
        HR._nr_rks_fnlc_mo_orig = HR.nr_rks_fnlc_mo
        # `_gen_rhf_response` imports the name inside the function body, so
        # the module attribute is what the call site reaches.
        HR.nr_rks_fnlc_mo = _dispatch
        # `hessian/uks.py` does `from ...hessian.rks import nr_rks_fnlc_mo` at
        # **import** time, and its `nr_uks_fnlc_mo` -- which is what both the
        # unrestricted response function and the UKS Hessian's CPHF
        # right-hand side call -- reaches that module-level name, not
        # `hessian.rks`'s.  Patching only `HR` leaves every open-shell caller
        # on GPU4PySCF's own code with this module loaded and nothing saying
        # so.  `tests/test_patch_aliases.py` is the check that keeps it
        # visible.
        HU.nr_rks_fnlc_mo = _dispatch
        _PATCHED = True


apply_patch()
