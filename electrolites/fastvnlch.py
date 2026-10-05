"""
The VV10 Fock-matrix derivative of an analytic Hessian, with its atom loops
batched (catalogue B14).

Importing this module replaces ``gpu4pyscf.hessian.rks._get_vnlc_deriv1`` --
the ``make_h1`` term of every VV10 Hessian (wB97M-V, wB97X-V, B97M-V, VV10),
restricted and unrestricted, direct and density-fitted
(``df/hessian/{rks,uks}.py`` import it by name and are patched too).

GPU4PySCF's function has two grid-block loops, and inside each block a loop
over the atoms issuing about ten contractions per atom (five for the
grid-response loop) to build a dense ``(natm, 3, nsub, nocc)`` block, which
is then projected onto the MO basis per block.  That is ``natm`` x ``nblocks``
x ~15 launches of small contractions: B5's launch-bound shape on the VV10
grid.  Here the per-atom sums are rewritten as **one GEMM per block** over a
stacked basis,

    sum_g B_c[p, g] L_c[g, (A d), j]       B = [mu, dmu.drho, dmu_x, dmu_y, dmu_z]

with ``L`` built by broadcasting from the per-atom grid functions (chunked
over the atoms to bound its memory), the result is accumulated in the AO
basis across blocks, and the MO projection is done once at the end.  The
arithmetic is the same sums, reassociated.

Everything else in the function -- the density and its derivatives on the
grid, the three ``libgdft`` pair kernels (two of which ``fastvv10ft``
replaces), the per-atom ``dmudA_nu`` correction, the grid-response kernels --
is GPU4PySCF's code, copied verbatim from 1.8.1 ``hessian/rks.py:2040-2600``.

``FASTVNLCH_OFF=1`` leaves GPU4PySCF's function in place;
``FASTVNLCH_LBYTES`` bounds the batched operand (default 256 MB).
``docs/README_fastvv10ft.md`` section on B14.
"""
import ctypes
import os

import cupy
import numpy

from gpu4pyscf.dft import numint
from gpu4pyscf.dft.numint import NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD
from gpu4pyscf.hessian import rks as HR
from gpu4pyscf.hessian.rks import (get_d2mu_dr2, get_dweight_dA,
                                   get_drho_dA_sparse)
from gpu4pyscf.lib.cupy_helper import contract, take_last2d, batched_vec_norm2

libgdft = numint.libgdft

_OFF = bool(os.environ.get('FASTVNLCH_OFF'))
_LBYTES = int(float(os.environ.get('FASTVNLCH_LBYTES', 256e6)))

STATS = {'calls': 0, 'blocks': 0, 'chunks': 0}


def stats():
    return dict(STATS)


def _chunk(ng, nocc, ncomp, K):
    """Atom-derivative rows per GEMM so that L stays within _LBYTES."""
    per = ncomp * ng * nocc * 8
    return max(1, min(K, _LBYTES // max(per, 1)))


def _orbital_response_block(acc, idx, mu, dmu_dr, dd, mm, weight, fg, drho,
                            FA, GA, DA):
    """The first loop's per-atom contractions for one block and one spin.

    ``acc`` is the (nao, natm, 3, nocc) AO accumulator; FA, GA are the
    block's (natm, 3, ng) f_rho^A and f_gamma^A, DA its (natm, 3, 3, ng)
    d(nabla rho)/dA.  Upstream's five terms per atom, regrouped by the AO
    factor they end on:

        mu   : fA w_mu + 2 gA drw + 2 sum_x DA_x fwdmu_x
        dd   : 2 gA w_mu
        dmu_x: 2 DA_x fwmu
    """
    natm = FA.shape[0]
    K = 3 * natm
    nsub, ng = mu.shape
    nocc = mm.shape[1]
    w_mu = weight[:, None] * (mu.T @ mm)                       # (g, j)
    w_dmu = contract("dpg,pj->djg", dmu_dr, mm) * weight         # (3, j, g)
    drw = contract("xg,xjg->gj", drho, w_dmu)                    # (g, j)
    fwmu = fg[:, None] * w_mu
    fwdmu = cupy.ascontiguousarray((w_dmu * fg).transpose(0, 2, 1))  # (3,g,j)
    del w_dmu
    F = FA.reshape(K, ng)
    G = GA.reshape(K, ng)
    D = DA.reshape(K, 3, ng)
    B = cupy.empty((nsub, 5, ng))
    B[:, 0] = mu
    B[:, 1] = dd
    B[:, 2:5] = dmu_dr.transpose(1, 0, 2)
    B = B.reshape(nsub, 5 * ng)
    acc2 = acc.reshape(acc.shape[0], K, nocc)
    kc = _chunk(ng, nocc, 5, K)
    for k0 in range(0, K, kc):
        k1 = min(K, k0 + kc)
        n = k1 - k0
        Fc = F[k0:k1].T[:, :, None]                              # (g, n, 1)
        Gc = 2 * G[k0:k1].T[:, :, None]
        L = cupy.empty((5, ng, n, nocc))
        cupy.multiply(Fc, w_mu[:, None, :], out=L[0])
        L[0] += Gc * drw[:, None, :]
        cupy.multiply(Gc, w_mu[:, None, :], out=L[1])
        for x in range(3):
            Dx = 2 * D[k0:k1, x].T[:, :, None]
            L[0] += Dx * fwdmu[x][:, None, :]
            cupy.multiply(Dx, fwmu[:, None, :], out=L[2 + x])
        out = B @ L.reshape(5 * ng, n * nocc)
        acc2[idx, k0:k1] += out.reshape(nsub, n, nocc)
        STATS['chunks'] += 1
    STATS['blocks'] += 1


def _grid_response_block(acc, idx, mu, dmu_dr, dd, mm, drho, RP, GP):
    """The grid-response loop's per-atom contractions, one block and spin:

        mu: RP mu_m + 2 GP ddm        dd: 2 GP mu_m
    """
    natm = RP.shape[0]
    K = 3 * natm
    nsub, ng = mu.shape
    nocc = mm.shape[1]
    mu_m = mu.T @ mm                                             # (g, j)
    ddm = contract("xjg,xg->gj", contract("dpg,pj->djg", dmu_dr, mm), drho)
    R = RP.reshape(K, ng)
    G = GP.reshape(K, ng)
    B = cupy.empty((nsub, 2, ng))
    B[:, 0] = mu
    B[:, 1] = dd
    B = B.reshape(nsub, 2 * ng)
    acc2 = acc.reshape(acc.shape[0], K, nocc)
    kc = _chunk(ng, nocc, 2, K)
    for k0 in range(0, K, kc):
        k1 = min(K, k0 + kc)
        n = k1 - k0
        Rc = R[k0:k1].T[:, :, None]
        Gc = 2 * G[k0:k1].T[:, :, None]
        L = cupy.empty((2, ng, n, nocc))
        cupy.multiply(Rc, mu_m[:, None, :], out=L[0])
        L[0] += Gc * ddm[:, None, :]
        cupy.multiply(Gc, mu_m[:, None, :], out=L[1])
        out = B @ L.reshape(2 * ng, n * nocc)
        acc2[idx, k0:k1] += out.reshape(nsub, n, nocc)
        STATS['chunks'] += 1
    STATS['blocks'] += 1


def _get_vnlc_deriv1(hessobj, mo_coeff, mo_occ, max_memory):
    """
        Equation notation follows:
        Liang J, Feng X, Liu X, Head-Gordon M. Analytical harmonic vibrational frequencies with
        VV10-containing density functionals: Theory, efficient implementation, and
        benchmark assessments. J Chem Phys. 2023 May 28;158(20):204109. doi: 10.1063/5.0152838.
    """

    # Note (Henry Wang 20250428):
    # We observed that in several very simple systems, for example H2O2, H2CO, C2H4,
    # if we do not include the grid response term, the analytical and numerical Fock matrix
    # derivative, although only diff by else than 1e-7 (norm 1), can cause a 1e-3 error in hessian,
    # likely because the CPHF converged to a different solution.
    grid_response = True

    mol = hessobj.mol
    mf = hessobj.base

    grids = mf.nlcgrids
    if not grid_response:
        if grids.coords is None:
            grids.build()

        ni = mf._numint
        opt = getattr(ni, 'gdftopt', None)
        if opt is None:
            ni.build(mol, grids.coords)
            opt = ni.gdftopt
    else:
        grids = grids.copy()
        grids.build(sort_grids_of_each_atom = True)

        ni = numint.NumInt()
        ni.gdftopt = None
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    natm = _sorted_mol.natm
    mol = None

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    if dm0.ndim == 3:
        assert dm0.shape[0] == 2
        dm0 = dm0[0] + dm0[1]
    dm0_sorted = opt.sort_orbitals(dm0, axis=[0,1])
    dm_mask_buf = cupy.empty(nao * nao)
    dm0 = None

    if numint.libxc.is_nlc(mf.xc):
        xc_code = mf.xc
    else:
        xc_code = mf.nlc
    nlc_coefs = ni.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars, fac = nlc_coefs[0]

    kappa_prefactor = nlc_pars[0] * 1.5 * numpy.pi * (9 * numpy.pi)**(-1.0/6.0)
    C_in_omega = nlc_pars[1]
    beta = 0.03125 * (3.0 / nlc_pars[0]**2)**0.75

    ngrids_full = grids.coords.shape[0]
    rho_drho = cupy.empty([4, ngrids_full])
    g1 = 0
    for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, nao, deriv = 1, strict_grid_order = True):
        g0, g1 = g1, g1 + weight.size
        dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)
        rho_drho[:, g0:g1] = numint.eval_rho(_sorted_mol, ao, dm0_masked, xctype = "NLC", hermi = 1)
    assert g1 == ngrids_full

    rho_i = rho_drho[0,:]

    rho_nonzero_mask = cupy.logical_and(
        rho_i >= NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD,
        cupy.abs(grids.weights) > 1e-14,
    )

    rho_i = rho_i[rho_nonzero_mask]
    grids_coords = cupy.ascontiguousarray(grids.coords[rho_nonzero_mask, :])
    grids_weights = grids.weights[rho_nonzero_mask]
    ngrids = grids_coords.shape[0]

    nabla_rho_i = rho_drho[1:4, rho_nonzero_mask]
    gamma_i = batched_vec_norm2(nabla_rho_i.T)

    stream = cupy.cuda.get_current_stream()

    omega_i               = cupy.empty(ngrids)
    domega_drho_i         = cupy.empty(ngrids)
    domega_dgamma_i       = cupy.empty(ngrids)
    d2omega_drho2_i       = cupy.empty(ngrids)
    d2omega_dgamma2_i     = cupy.empty(ngrids)
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
        ctypes.c_int(ngrids)
    )
    kappa_i         = kappa_prefactor * rho_i**(1.0/6.0)
    dkappa_drho_i   = (kappa_prefactor * (1.0/6.0)) * rho_i**(-5.0/6.0)
    d2kappa_drho2_i = (kappa_prefactor * (-5.0/36.0)) * rho_i**(-11.0/6.0)

    rho_weight_i = rho_i * grids_weights
    U_i = cupy.empty(ngrids)
    W_i = cupy.empty(ngrids)
    A_i = cupy.empty(ngrids)
    B_i = cupy.empty(ngrids)
    C_i = cupy.empty(ngrids)
    E_i = cupy.empty(ngrids)

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
        ctypes.c_int(ngrids)
    )
    del rho_weight_i

    f_rho_i = beta + E_i + rho_i * (dkappa_drho_i * U_i + domega_drho_i * W_i)
    f_gamma_i = rho_i * domega_dgamma_i * W_i
    fw_rho_i   =   f_rho_i * grids_weights
    fw_gamma_i = f_gamma_i * grids_weights

    ao_loc_sorted = _sorted_mol.ao_loc
    ao_expand = ao_loc_sorted[1:] - ao_loc_sorted[:-1]
    from pyscf.gto.mole import ATOM_OF
    i_atom_of_aos = numpy.repeat(_sorted_mol._bas[:,ATOM_OF], ao_expand)
    i_atom_of_aos = cupy.asarray(i_atom_of_aos, dtype = cupy.int32)

    drho_dA   = cupy.empty([natm, 3, ngrids], order = "C")
    dgamma_dA = cupy.empty([natm, 3, ngrids], order = "C")
    dnablarho_dA = cupy.empty([natm, 3, 3, ngrids], order = "C")

    g0_full = 0
    g0_nonzero = 0
    for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv = 2, strict_grid_order = True):
        g1_full = g0_full + weight.shape[0]

        ao = ao[:, :, rho_nonzero_mask[g0_full : g1_full]]

        if ao.size == 0:
            g0_full = g1_full
            continue

        g1_nonzero = g0_nonzero + ao.shape[-1]

        mu = ao[0, :, :]
        dmu_dr = ao[1:4, :, :]
        d2mu_dr2 = get_d2mu_dr2(ao)

        split_drho_dr = nabla_rho_i[:, g0_nonzero : g1_nonzero]

        dm0_masked = take_last2d(dm0_sorted, idx, out = dm_mask_buf)
        if grid_response:
            i_atom_of_grids = int(grids.atm_idx[g0_full])
            assert cupy.max(cupy.abs(grids.atm_idx[g0_full : g1_full] - i_atom_of_grids)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)
        else:
            i_atom_of_grids = None
        masked_i_atom_of_aos = i_atom_of_aos[idx]

        drho_dA_orbital_response, drho_dA_grid_response = \
            get_drho_dA_sparse(dm0_masked, "GGA", natm, masked_i_atom_of_aos, i_atom_of_grids, mu, dmu_dr, d2mu_dr2, with_grid_response = grid_response)
        if not grid_response:
            drho_dA_full = drho_dA_orbital_response
        else:
            drho_dA_full = drho_dA_orbital_response + drho_dA_grid_response
        del drho_dA_orbital_response, drho_dA_grid_response
        drho_dA[:, :, g0_nonzero : g1_nonzero] = drho_dA_full[:, :, 0, :]
        dnablarho_dA[:, :, :, g0_nonzero : g1_nonzero] = drho_dA_full[:, :, 1:4, :]
        dgamma_dA[:, :, g0_nonzero : g1_nonzero] = 2 * contract("Adxg,xg->Adg", drho_dA_full[:, :, 1:4, :], split_drho_dr)

        del drho_dA_full
        del split_drho_dr
        del mu, dmu_dr, d2mu_dr2

        g0_nonzero = g1_nonzero
        g0_full = g1_full
    assert g1_full == ngrids_full
    assert g1_nonzero == ngrids

    drho_dA   = cupy.ascontiguousarray(drho_dA)
    dgamma_dA = cupy.ascontiguousarray(dgamma_dA)
    f_rho_A_i   = cupy.empty([natm, 3, ngrids], order = "C")
    f_gamma_A_i = cupy.empty([natm, 3, ngrids], order = "C")

    libgdft.VXC_vv10nlc_hess_eval_f_t(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(f_rho_A_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(f_gamma_A_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(grids_weights.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(U_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(W_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(A_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(B_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(C_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(domega_drho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(domega_dgamma_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(dkappa_drho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_drho2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_dgamma2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2omega_drho_dgamma_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2kappa_drho2_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(drho_dA.data.ptr, ctypes.c_void_p),
        ctypes.cast(dgamma_dA.data.ptr, ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ctypes.c_int(3 * natm),
    )
    del drho_dA, dgamma_dA

    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[mo_coeff.ndim-2])
    nmo = mo_coeff.shape[-1]
    if mo_coeff.ndim == 3:
        mocca = mo_coeff[0][:, mo_occ[0]>0]
        moccb = mo_coeff[1][:, mo_occ[1]>0]
        nocca = mocca.shape[1]
        noccb = moccb.shape[1]
        dFock_mo_occa = cupy.zeros([natm, 3, nmo, nocca])
        dFock_mo_occb = cupy.zeros([natm, 3, nmo, noccb])
    else:
        mocc = mo_coeff[:, mo_occ>0]
        nocc = mocc.shape[1]
        dFock_mo_occ = cupy.zeros([natm, 3, nmo, nocc])

    dFock_orbital_response_dmudA_nu_term = cupy.zeros((3, nao, nao))


    # ---- Electrolites: the per-block contractions batched over the atoms
    # and accumulated in the AO basis; one projection at the end.
    if mo_coeff.ndim == 3:
        moccs = [mocca, moccb]
        mos = [mo_coeff[0], mo_coeff[1]]
        dFock_mo_occs = [dFock_mo_occa, dFock_mo_occb]
    else:
        moccs = [mocc]
        mos = [mo_coeff]
        dFock_mo_occs = [dFock_mo_occ]
    dFock_ao = [cupy.zeros((nao, natm, 3, m.shape[1])) for m in moccs]

    g0_full = 0
    g0_nonzero = 0
    for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv = 2, strict_grid_order = True):
        g1_full = g0_full + weight.shape[0]

        ao = ao[:, :, rho_nonzero_mask[g0_full : g1_full]]

        if ao.size == 0:
            g0_full = g1_full
            continue

        g1_nonzero = g0_nonzero + ao.shape[-1]

        mu = ao[0, :, :]
        dmu_dr = ao[1:4, :, :]
        d2mu_dr2 = get_d2mu_dr2(ao)

        split_drho_dr = nabla_rho_i[:, g0_nonzero : g1_nonzero]
        split_dnablarho_dA = dnablarho_dA[:, :, :, g0_nonzero : g1_nonzero]
        split_f_rho_A_i = f_rho_A_i[:, :, g0_nonzero : g1_nonzero]
        split_f_gamma_A_i = f_gamma_A_i[:, :, g0_nonzero : g1_nonzero]

        weight = weight[rho_nonzero_mask[g0_full : g1_full]]
        dmudr_drhodr = contract("xpg,xg->pg", dmu_dr, split_drho_dr)
        fg = f_gamma_i[g0_nonzero : g1_nonzero]
        for mocc_s, acc in zip(moccs, dFock_ao):
            mm = mocc_s[idx]
            _orbital_response_block(acc, idx, mu, dmu_dr, dmudr_drhodr, mm, weight, fg,
                                    split_drho_dr, split_f_rho_A_i, split_f_gamma_A_i,
                                    split_dnablarho_dA)

        # w_i f_i^\rho \nabla_A (\phi_\mu \phi_nu)_i
        # dmudA_nu_ao = cupy.einsum('dpg,qg->dpq', dmu_dr, mu * fw_rho_i[g0_nonzero : g1_nonzero])

        # w_i 2 f_i^\gamma \nabla\rho \cdot \nabla_A \nabla(\phi_\mu \phi_nu)_i
        # dmudA_nu_ao += 2 * cupy.einsum('dxpg,qg,xg->dpq', d2mu_dr2, mu, split_drho_dr * fw_gamma_i[g0_nonzero : g1_nonzero])
        # dmudA_nu_ao += 2 * cupy.einsum('dpg,xqg,xg->dpq', dmu_dr, dmu_dr, split_drho_dr * fw_gamma_i[g0_nonzero : g1_nonzero])
        mu_prefactor = dmu_dr * fw_rho_i[g0_nonzero : g1_nonzero]
        mu_prefactor += 2 * contract("dxpg,xg->dpg", d2mu_dr2, split_drho_dr * fw_gamma_i[g0_nonzero : g1_nonzero])
        dmudA_nu_ao = mu_prefactor @ mu.T
        dmudr_prefactor = contract("xpg,xg->pg", dmu_dr, split_drho_dr * fw_gamma_i[g0_nonzero : g1_nonzero])
        dmudA_nu_ao += 2 * dmu_dr @ dmudr_prefactor.T

        del mu, dmu_dr, d2mu_dr2

        dFock_orbital_response_dmudA_nu_term[numpy.ix_(range(3), idx, idx)] += dmudA_nu_ao

        if grid_response:
            i_atom_of_grids = int(grids.atm_idx[g0_full])
            assert cupy.max(cupy.abs(grids.atm_idx[g0_full : g1_full] - i_atom_of_grids)) == 0 # Guaranteed by grids.build(sort_grids_of_each_atom = True)

            dmudA_nu_ao_ao = dmudA_nu_ao + dmudA_nu_ao.transpose(0,2,1)
            for mocc_s, acc in zip(moccs, dFock_ao):
                acc[idx, i_atom_of_grids] += contract("dpq,qj->pdj", dmudA_nu_ao_ao, mocc_s[idx])
            del dmudA_nu_ao_ao
        del dmudA_nu_ao

        g0_nonzero = g1_nonzero
        g0_full = g1_full
    assert g1_full == ngrids_full
    assert g1_nonzero == ngrids

    for i_atom in range(0, natm):
        ao_of_atom_i = cupy.where(i_atom_of_aos == i_atom)[0]
        if ao_of_atom_i.size > 0:
            dFock_ao_of_atom_i = dFock_orbital_response_dmudA_nu_term[:, ao_of_atom_i, :]

            if mo_coeff.ndim == 3:
                dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i @ mocca
                dFock_mo_occa[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff[0, ao_of_atom_i, :], dFock_ao_of_atom_i_mocc)
                dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i @ moccb
                dFock_mo_occb[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff[1, ao_of_atom_i, :], dFock_ao_of_atom_i_mocc)
                dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i.transpose(0,2,1) @ mocca[ao_of_atom_i]
                dFock_mo_occa[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff[0], dFock_ao_of_atom_i_mocc)
                dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i.transpose(0,2,1) @ moccb[ao_of_atom_i]
                dFock_mo_occb[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff[1], dFock_ao_of_atom_i_mocc)
            else:
                dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i @ mocc
                dFock_mo_occ[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff[ao_of_atom_i], dFock_ao_of_atom_i_mocc)
                dFock_ao_of_atom_i_mocc = dFock_ao_of_atom_i.transpose(0,2,1) @ mocc[ao_of_atom_i]
                dFock_mo_occ[i_atom, :, :, :] -= cupy.einsum("pi,dpj->dij", mo_coeff, dFock_ao_of_atom_i_mocc)
            del dFock_ao_of_atom_i_mocc, dFock_ao_of_atom_i
    del dFock_orbital_response_dmudA_nu_term

    if grid_response:
        assert grids.atm_idx.shape[0] == grids.coords.shape[0]
        grid_to_atom_index_map = grids.atm_idx[rho_nonzero_mask]
        assert cupy.all(grid_to_atom_index_map >= 0) # There shouldn't be padded grids not from any atom with sort_grids_of_each_atom = True

        grid_offsets_of_atom = cupy.bincount(grid_to_atom_index_map, minlength = natm)
        grid_offsets_of_atom = cupy.concatenate((cupy.array([0]), grid_offsets_of_atom))
        grid_offsets_of_atom = cupy.cumsum(grid_offsets_of_atom)
        grid_offsets_of_atom = cupy.asarray(grid_offsets_of_atom, dtype = cupy.int32)

        assert grid_offsets_of_atom.shape == (natm + 1,)
        for i_atom in range(natm):
            assert cupy.all(grid_to_atom_index_map[grid_offsets_of_atom[i_atom] : grid_offsets_of_atom[i_atom + 1]] == i_atom)
        assert cupy.all(grid_to_atom_index_map[grid_offsets_of_atom[natm] : ] < 0)

        rho_weight_i = rho_i * grids_weights
        E_Bgr_i = cupy.empty([natm, 3, ngrids], order = "C")
        U_Bgr_i = cupy.empty([natm, 3, ngrids], order = "C")
        W_Bgr_i = cupy.empty([natm, 3, ngrids], order = "C")
        libgdft.VXC_vv10nlc_hess_eval_EUW_grid_response_offdiagonal(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(E_Bgr_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(U_Bgr_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(W_Bgr_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
            ctypes.cast(rho_weight_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grid_to_atom_index_map.data.ptr, ctypes.c_void_p),
            ctypes.cast(grid_offsets_of_atom.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(natm),
        )
        del rho_weight_i
        for i_atom in range(natm):
            g0, g1 = grid_offsets_of_atom[i_atom], grid_offsets_of_atom[i_atom + 1]

            E_Bgr_i[i_atom, :, g0:g1] = -cupy.sum(E_Bgr_i[:, :, g0:g1], axis = 0)
            U_Bgr_i[i_atom, :, g0:g1] = -cupy.sum(U_Bgr_i[:, :, g0:g1], axis = 0)
            W_Bgr_i[i_atom, :, g0:g1] = -cupy.sum(W_Bgr_i[:, :, g0:g1], axis = 0)

        grids_weights_1 = get_dweight_dA(_sorted_mol, grids)
        grids_weights_1 = grids_weights_1[:, :, rho_nonzero_mask]
        grids_weights_1 = cupy.ascontiguousarray(grids_weights_1)

        E_Bw_i = cupy.empty([natm, 3, ngrids], order = "C")
        U_Bw_i = cupy.empty([natm, 3, ngrids], order = "C")
        W_Bw_i = cupy.empty([natm, 3, ngrids], order = "C")
        libgdft.VXC_vv10nlc_hess_eval_EUW_with_weight1(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(E_Bw_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(U_Bw_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(W_Bw_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_coords.data.ptr, ctypes.c_void_p),
            ctypes.cast(grids_weights_1.data.ptr, ctypes.c_void_p),
            ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
            ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(natm * 3),
        )

        f_rho_grid_response_i = (E_Bw_i + E_Bgr_i) + ((U_Bw_i + U_Bgr_i) * dkappa_drho_i + (W_Bw_i + W_Bgr_i) * domega_drho_i) * rho_i
        f_gamma_grid_response_i = (W_Bw_i + W_Bgr_i) * domega_dgamma_i * rho_i
        del E_Bw_i, U_Bw_i, W_Bw_i
        del E_Bgr_i, U_Bgr_i, W_Bgr_i

        g0_full = 0
        g0_nonzero = 0
        for ao, idx, weight, _ in ni.block_loop(_sorted_mol, grids, deriv = 1, strict_grid_order = True):
            g1_full = g0_full + weight.shape[0]

            ao = ao[:, :, rho_nonzero_mask[g0_full : g1_full]]

            if ao.size == 0:
                g0_full = g1_full
                continue

            g1_nonzero = g0_nonzero + ao.shape[-1]

            mu = ao[0, :, :]
            dmu_dr = ao[1:4, :, :]

            split_drho_dr = nabla_rho_i[:, g0_nonzero : g1_nonzero]
            split_dwdA = grids_weights_1[:, :, g0_nonzero : g1_nonzero]
            split_f_rho_grid_response_i = f_rho_grid_response_i[:, :, g0_nonzero : g1_nonzero]
            split_f_gamma_grid_response_i = f_gamma_grid_response_i[:, :, g0_nonzero : g1_nonzero]

            weight = weight[rho_nonzero_mask[g0_full : g1_full]]
            rho_prefactor = split_dwdA * f_rho_i[g0_nonzero : g1_nonzero] + split_f_rho_grid_response_i * weight
            gamma_prefactor = split_dwdA * f_gamma_i[g0_nonzero : g1_nonzero] + split_f_gamma_grid_response_i * weight
            dmudr_drhodr = contract("xpg,xg->pg", dmu_dr, split_drho_dr)
            for mocc_s, acc in zip(moccs, dFock_ao):
                _grid_response_block(acc, idx, mu, dmu_dr, dmudr_drhodr, mocc_s[idx],
                                     split_drho_dr, rho_prefactor, gamma_prefactor)

            g0_nonzero = g1_nonzero
            g0_full = g1_full
        assert g1_full == ngrids_full
        assert g1_nonzero == ngrids

    for mo_s, acc, dF in zip(mos, dFock_ao, dFock_mo_occs):
        dF += contract("pi,pAdj->Adij", mo_s, acc)
    del dFock_ao
    if mo_coeff.ndim == 3:
        return (dFock_mo_occa, dFock_mo_occb)
    else:
        return dFock_mo_occ

_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    from gpu4pyscf.df.hessian import rks as DHR
    from gpu4pyscf.df.hessian import uks as DHU
    HR._get_vnlc_deriv1_orig = HR._get_vnlc_deriv1

    def _counted(hessobj, mo_coeff, mo_occ, max_memory):
        STATS['calls'] += 1
        return _get_vnlc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    # `hessian/rks.py` (`make_h1`) and `hessian/uks.py` reach it through
    # `hessian.rks`'s module attribute; the density-fitted Hessians import
    # the name at import time and carry their own binding
    HR._get_vnlc_deriv1 = _counted
    from gpu4pyscf.hessian import uks as HU
    for mod in (HU, DHR, DHU):
        if getattr(mod, '_get_vnlc_deriv1', None) is HR._get_vnlc_deriv1_orig:
            mod._get_vnlc_deriv1 = _counted
    _PATCHED = True


apply_patch()
