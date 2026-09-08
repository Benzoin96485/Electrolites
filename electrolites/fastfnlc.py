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

**Limits, all of which fall through to GPU4PySCF rather than fail.**
The ``return_in_mo=True`` path (the Hessian's CPHF solve asks for the response
in the MO basis) keeps GPU4PySCF's per-density loop for the potential
contraction but still gets the cache; UKS (``mo_coeff.ndim == 3``) is handed
back whole; and more than one GPU, as everywhere here.
"""
import ctypes
import functools
import os

import cupy
import numpy

from gpu4pyscf.dft import numint
from gpu4pyscf.dft.numint import NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD
from gpu4pyscf.hessian import rks as HR
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
        rho_drho[:, g0:g1] = numint.eval_rho2(
            _sorted_mol, ao, mo_sorted[idx, :], mo_occ, None, 'GGA')
    assert g1 == ngrids_full

    rho_i = rho_drho[0, :]
    rho_nonzero_mask = cupy.logical_and(
        rho_i >= NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD,
        cupy.abs(grids.weights) > 1e-14,
    )
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

    if return_in_mo:
        mocc = mo_sorted[:, mo_occ > 0]
        vmat = cupy.zeros([n_dm1, mo_sorted.shape[1], mocc.shape[1]])
    else:
        vmat = cupy.zeros([n_dm1, nao, nao])

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
                        vmat[i_dm + i_dm1_batch, :, :] += \
                            mo_sorted[idx, :].T @ V_munu @ mocc[idx, :]
                    else:
                        add_sparse(vmat[i_dm + i_dm1_batch], V_munu, idx)

            g0_nonzero = g1_nonzero
            g0_full = g1_full
        assert g1_full == ngrids_full
        assert g1_nonzero == ngrids

    if not return_in_mo:
        vmat = opt.unsort_orbitals(vmat, axis=[1, 2])
    if output_in_2d:
        vmat = vmat[0]
    return vmat


_PATCHED = False


def _dispatch(mf, mol, mo_coeff, mo_occ, dm1s, return_in_mo=True):
    ok = (num_devices == 1 and getattr(mo_coeff, 'ndim', 3) == 2
          and getattr(dm1s, 'ndim', 0) in (2, 3))
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
        _PATCHED = True


apply_patch()
