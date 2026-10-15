"""The CASSCF orbital step on the device (`docs/FEATURE_TARGETS_PYSCF_2.13.md`
M4 proper, Phase 5).

PySCF's one-step CASSCF (``mcscf/mc1step.py``) spends a GPU CASSCF's
macro-iteration, once the integrals, J/K and FCI are on the device
(Phase 2, §13), in host code:

* ``gen_g_hop``: Python loops over the nmo orbitals building the
  2-RDM-contracted Hessian blocks (``hdm2``, nmo^2 ncas^2), and an AH
  Hessian-vector product ``h_op`` that contracts ``hdm2`` and calls a
  two-density J/K (``update_jk_in_ah``) per AH iteration;
* ``gorb_update`` and ``update_casdm``: the same loops over ppaa/papa;
* ``solve_approx_ci``: the CI response subspace built from host copies of
  the N_det-long CI vectors (heff/seff by numpy dots).

Here the same expressions, term for term, as device GEMMs over the whole
ppaa/papa (kept on the device next to the host copies PySCF's other code
reads), the AH product with every operand resident and the J/K taken and
returned on the device, and the CI response with the vectors and the sigma
builds on the device.  The AH driver (``ciah.davidson_cc``), the
macro/micro logic and every convergence decision stay PySCF's; vectors
cross to the host only as the packed orbital-rotation vectors PySCF's
driver holds (nmo^2).
"""
import numpy as np
import cupy as cp

from pyscf import lib


def _dev_eris(eris):
    """Device ppaa/papa/vhf_c of a PySCF-layout ``_ERIS`` (cached on it)."""
    d = getattr(eris, '_dev', None)
    if d is None:
        d = eris._dev = dict(ppaa=cp.asarray(eris.ppaa),
                             papa=cp.asarray(eris.papa),
                             vhf_c=cp.asarray(eris.vhf_c),
                             j_pc=cp.asarray(eris.j_pc),
                             k_pc=cp.asarray(eris.k_pc))
    return d


def _jk(casscf, dms):
    """Two-density J/K on the device (DF cached tensor, or the CASSCF's own
    J/K with a host round trip)."""
    from electrolites.casscf_gpu import _GPUCAS
    if isinstance(casscf, _GPUCAS):          # DF route (the direct route
        return casscf.with_df.get_jk(      # also carries a with_df)
            cp.stack(dms), hermi=1, device=True)
    vj, vk = casscf.get_jk(casscf.mol, cp.asnumpy(cp.stack(dms)))
    return cp.asarray(vj), cp.asarray(vk)


class _Packer:
    """pack_uniq_var / unpack_uniq_var on the device with PySCF's mask."""

    def __init__(self, casscf, nmo):
        idx = casscf.uniq_var_indices(nmo, casscf.ncore, casscf.ncas,
                                      casscf.frozen)
        self.idx = cp.asarray(np.flatnonzero(idx.ravel()))
        self.nmo = nmo

    def pack(self, mat):
        return mat.ravel()[self.idx]

    def unpack(self, v):
        m = cp.zeros(self.nmo * self.nmo)
        m[self.idx] = cp.asarray(v)
        m = m.reshape(self.nmo, self.nmo)
        return m - m.T


def gen_g_hop(casscf, mo, u, casdm1, casdm2, eris):
    """PySCF ``mc1step.gen_g_hop`` on the device (same terms, same return:
    host g_orb / h_diag, host-vector ``gorb_update`` and ``h_op``)."""
    ncas, nelecas, ncore = casscf.ncas, casscf.nelecas, casscf.ncore
    nocc = ncas + ncore
    nmo = mo.shape[1]
    d = _dev_eris(eris)
    ppaa, papa, vhf_c = d['ppaa'], d['papa'], d['vhf_c']
    mo_d = cp.asarray(mo)
    dm1a = cp.asarray(casdm1)
    dm2 = cp.asarray(casdm2)
    n2 = ncas * ncas

    dm1 = cp.zeros((nmo, nmo))
    dm1[cp.arange(ncore), cp.arange(ncore)] = 2
    dm1[ncore:nocc, ncore:nocc] = dm1a

    ar = cp.arange(nocc)
    # part5: jkcaa[i,u] = sum_v (6 papa[i,u,i,v] - 2 ppaa[i,i,u,v]) dm1[u,v]
    kii = papa[ar, :, ar, :]                                  # (nocc, u, v)
    jii = ppaa[ar, ar]                                        # (nocc, u, v)
    jkcaa = ((6 * kii - 2 * jii) * dm1a[None]).sum(axis=2)
    # part2, part3
    vhf_a = (ppaa.reshape(nmo * nmo, n2) @ dm1a.ravel()).reshape(nmo, nmo)
    vhf_a -= .5 * cp.einsum('iuqv,uv->iq', papa, dm1a)
    # part1 ~ (J + 2K)
    dm2tmp = (dm2.transpose(1, 2, 0, 3) + dm2.transpose(0, 2, 1, 3)
              ).reshape(n2, n2)
    jtmp = (ppaa.reshape(nmo * nmo, n2) @ dm2.reshape(n2, n2)
            ).reshape(nmo, nmo, ncas, ncas)
    ktmp = (cp.ascontiguousarray(papa.transpose(0, 2, 1, 3)
                                 ).reshape(nmo * nmo, n2) @ dm2tmp
            ).reshape(nmo, nmo, ncas, ncas)
    hdm2 = cp.ascontiguousarray((ktmp + jtmp).transpose(0, 2, 1, 3))
    au = cp.arange(ncas)
    g_dm2 = jtmp[:, ncore + au, au, :].sum(axis=1)            # (nmo, ncas)
    ktmp = None
    vhf_ca = vhf_c + vhf_a
    h1e_mo = mo_d.T @ cp.asarray(casscf.get_hcore()) @ mo_d

    g = cp.zeros_like(h1e_mo)
    g[:, :ncore] = (h1e_mo[:, :ncore] + vhf_ca[:, :ncore]) * 2
    g[:, ncore:nocc] = (h1e_mo[:, ncore:nocc] + vhf_c[:, ncore:nocc]) @ dm1a
    g[:, ncore:nocc] += g_dm2
    jtmp = None
    pk = _Packer(casscf, nmo)

    def gorb_update(u, fcivec):
        u = cp.asarray(u)
        uc = u[:, :ncore]
        ua = cp.ascontiguousarray(u[:, ncore:nocc])
        ra = cp.ascontiguousarray((u - cp.eye(nmo))[:, ncore:nocc])
        mo1 = mo_d @ u
        mo_c = mo_d @ uc
        mo_a = mo_d @ ua
        dm_c = mo_c @ mo_c.T * 2
        cd1, cd2 = casscf.fcisolver.make_rdm12(fcivec, ncas, nelecas)
        cd1, cd2 = cp.asarray(cd1), cp.asarray(cd2)
        dm_a = mo_a @ cd1 @ mo_a.T
        vj, vk = _jk(casscf, (dm_c, dm_a))
        vhf_c1 = mo1.T @ (vj[0] - vk[0] * .5) @ mo1[:, :nocc]
        vhf_a1 = mo1.T @ (vj[1] - vk[1] * .5) @ mo1[:, :nocc]
        h1e_mo1 = u.T @ h1e_mo @ u[:, :nocc]
        # p1aa[i] = ua^T ppaa[i]; paa1[i] = papa[i]^(u,v,q) ra
        p1aa = cp.matmul(ua.T[None], ppaa.reshape(nmo, nmo, n2))
        paa1 = (cp.ascontiguousarray(papa.transpose(0, 1, 3, 2)
                                     ).reshape(nmo * n2, nmo) @ ra)
        gg = cp.zeros_like(h1e_mo)
        gg[:, :ncore] = (h1e_mo1[:, :ncore] + vhf_c1[:, :ncore]
                         + vhf_a1[:, :ncore]) * 2
        gg[:, ncore:nocc] = (h1e_mo1[:, ncore:nocc]
                             + vhf_c1[:, ncore:nocc]) @ cd1
        p1aa = (u.T @ p1aa.reshape(nmo, -1)).reshape(nmo, ncas, ncas, ncas)
        paa1 = (u.T @ paa1.reshape(nmo, -1)).reshape(nmo, ncas, ncas, ncas)
        p1aa += paa1
        p1aa += paa1.transpose(0, 1, 3, 2)
        # einsum('puwx,wxuv->pv')
        # einsum('puwx,wxuv->pv') as one GEMM over (u, w, x)
        gg[:, ncore:nocc] += (
            cp.ascontiguousarray(p1aa).reshape(nmo, -1)
            @ cp.ascontiguousarray(cd2.transpose(2, 0, 1, 3)
                                   ).reshape(-1, ncas))
        return cp.asnumpy(pk.pack(gg - gg.T))

    # hessian diagonal
    h1d = cp.diag(h1e_mo)
    dmd = cp.diag(dm1)
    h_diag = h1d[:, None] * dmd[None, :] - h1e_mo * dm1
    h_diag = h_diag + h_diag.T
    g_diag = cp.diag(g)
    h_diag -= g_diag + g_diag.reshape(-1, 1)
    idx = cp.arange(nmo)
    h_diag[idx, idx] += g_diag * 2
    v_diag = cp.diag(vhf_ca)
    h_diag[:, :ncore] += v_diag.reshape(-1, 1) * 2
    h_diag[:ncore] += v_diag * 2
    ic = cp.arange(ncore)
    h_diag[ic, ic] -= v_diag[:ncore] * 4
    tmp = cp.diag(vhf_c)[:, None] * cp.diag(dm1a)[None, :]
    h_diag[:, ncore:nocc] += tmp
    h_diag[ncore:nocc, :] += tmp.T
    tmp = -vhf_c[ncore:nocc, ncore:nocc] * dm1a
    h_diag[ncore:nocc, ncore:nocc] += tmp + tmp.T
    tmp = 6 * d['k_pc'] - 2 * d['j_pc']
    h_diag[ncore:, :ncore] += tmp[ncore:]
    h_diag[:ncore, ncore:] += tmp[ncore:].T
    h_diag[:nocc, ncore:nocc] -= jkcaa
    h_diag[ncore:nocc, :nocc] -= jkcaa.T
    # einsum('ijij->ij', hdm2), hdm2 (nmo, ncas, nmo, ncas)
    ii = cp.arange(nmo)
    v_diag = hdm2[ii[:, None], au[None, :], ii[:, None], au[None, :]]
    h_diag[ncore:nocc, :] += v_diag.T
    h_diag[:, ncore:nocc] += v_diag

    g_orb = cp.asnumpy(pk.pack(g - g.T))
    h_diag = cp.asnumpy(pk.pack(h_diag))
    gpg = (g + g.T) * .5
    hdm2_2 = hdm2.reshape(nmo * ncas, nmo * ncas)
    mo_core = mo_d[:, :ncore]
    mo_cas = mo_d[:, ncore:nocc]

    def h_op(x):
        x1 = pk.unpack(x)
        x2 = h1e_mo @ x1 @ dm1
        x2 -= gpg @ x1
        x2[:ncore] += x1[:ncore, ncore:] @ vhf_ca[ncore:] * 2
        x2[ncore:nocc] += dm1a @ x1[ncore:nocc] @ vhf_c
        # einsum('purv,rv->pu', hdm2, x1[:, ncore:nocc])
        x2[:, ncore:nocc] += (hdm2_2 @ x1[:, ncore:nocc].ravel()
                              ).reshape(nmo, ncas)
        if ncore > 0:
            va, vc = _update_jk_in_ah(casscf, mo_d, mo_core, mo_cas, x1,
                                      dm1a, ncore, nocc)
            x2[ncore:nocc] += va
            x2[:ncore, ncore:] += vc
        x2 = x2 - x2.T
        return cp.asnumpy(pk.pack(x2))

    return g_orb, gorb_update, h_op, h_diag


def _update_jk_in_ah(casscf, mo, mo_core, mo_cas, r, casdm1, ncore, nocc):
    """PySCF ``CASSCF.update_jk_in_ah`` on the device."""
    dm3 = mo_core @ r[:ncore, ncore:] @ mo[:, ncore:].T
    dm3 = dm3 + dm3.T
    dm4 = mo_cas @ casdm1 @ r[ncore:nocc] @ mo.T
    dm4 = dm4 + dm4.T
    vj, vk = _jk(casscf, (dm3, dm3 * 2 + dm4))
    va = casdm1 @ mo_cas.T @ (vj[0] * 2 - vk[0]) @ mo
    vc = mo_core.T @ (vj[1] * 2 - vk[1]) @ mo[:, ncore:]
    return va, vc


def update_casdm(casscf, mo, u, fcivec, e_cas, eris, envs={}):
    """PySCF ``CASSCF.update_casdm`` (``with_dep4`` False) with the
    hessian_co part on the device."""
    if casscf.with_dep4:
        from pyscf.mcscf import mc1step
        return mc1step.CASSCF.update_casdm(casscf, mo, u, fcivec, e_cas,
                                           eris, envs)
    nmo = mo.shape[1]
    ncas, nelecas, ncore = casscf.ncas, casscf.nelecas, casscf.ncore
    nocc = ncore + ncas
    n2 = ncas * ncas
    d = _dev_eris(eris)
    ppaa, papa, vhf_c = d['ppaa'], d['papa'], d['vhf_c']
    mo_d = cp.asarray(mo)
    u = cp.asarray(u)
    rmat = u - cp.eye(nmo)
    uc = u[:, :ncore]
    ua = cp.ascontiguousarray(u[:, ncore:nocc])
    ra = cp.ascontiguousarray(rmat[:, ncore:nocc])
    h1e_mo = mo_d.T @ cp.asarray(casscf.get_hcore()) @ mo_d
    ddm = uc @ uc.T * 2
    ic = cp.arange(ncore)
    ddm[ic, ic] -= 2
    jk = ua.T @ vhf_c @ ua
    jk += (ddm.ravel() @ ppaa.reshape(nmo * nmo, n2)).reshape(ncas, ncas)
    jk -= .5 * cp.einsum('iuqv,iq->uv', papa, ddm)
    p1aa = cp.matmul(ua.T[None], ppaa.reshape(nmo, nmo, n2))  # (i, a, uv)
    paa1 = (cp.ascontiguousarray(papa.transpose(0, 1, 3, 2)
                                 ).reshape(nmo * n2, nmo) @ ra)
    h1 = ua.T @ h1e_mo @ ua + jk
    aa11 = (ua.T @ p1aa.reshape(nmo, -1)).reshape((ncas,) * 4)
    aaaa = ppaa[ncore:nocc, ncore:nocc]
    aa11 = aa11 + aa11.transpose(2, 3, 0, 1) - aaaa
    a11a = (ra.T @ paa1.reshape(nmo, -1)).reshape((ncas,) * 4)
    a11a = a11a + a11a.transpose(1, 0, 2, 3)
    a11a = a11a + a11a.transpose(0, 1, 3, 2)
    h2 = aa11 + a11a
    ecore = float((h1e_mo * ddm).sum() + (vhf_c * ddm).sum())
    h1 = cp.asnumpy(h1)
    h2 = cp.asnumpy(h2)

    ci1, g = casscf.solve_approx_ci(h1, h2, fcivec, ecore, e_cas, envs)
    if g is not None:
        ovlp = np.dot(np.asarray(fcivec).ravel(), ci1.ravel())
        norm_g = np.linalg.norm(g)
        if 1 - abs(ovlp) > norm_g * casscf.ci_grad_trust_region:
            from pyscf.lib import logger
            logger.debug(casscf, '<ci1|ci0>=%5.3g |g|=%5.3g, ci1 out of '
                         'trust region', ovlp, norm_g)
            ci1 = np.asarray(fcivec).ravel() + g
            ci1 *= 1 / np.linalg.norm(ci1)
    casdm1, casdm2 = casscf.fcisolver.make_rdm12(ci1, ncas, nelecas)
    return casdm1, casdm2, g, ci1


def solve_approx_ci(casscf, h1, h2, ci0, ecore, e_cas, envs):
    """PySCF ``CASSCF.solve_approx_ci`` with the response subspace on the
    device (the GPU FCI solver's own sigma); other solvers are PySCF's."""
    from pyscf.mcscf import mc1step
    from electrolites import fci_gpu
    fs = casscf.fcisolver
    if (type(fs).contract_2e is not fci_gpu.FCISolver.contract_2e
            or not isinstance(fs, fci_gpu.FCISolver)
            or isinstance(ci0, (list, tuple)) or np.ndim(ci0) > 2
            or getattr(fs, 'approx_kernel', None)):
        return mc1step.CASSCF.solve_approx_ci(casscf, h1, h2, ci0, ecore,
                                              e_cas, envs)
    ncas, nelecas = casscf.ncas, casscf.nelecas
    from pyscf.fci import direct_spin1
    nelec_u = direct_spin1._unpack_nelec(nelecas, fs.spin)
    tb = fci_gpu._tables(ncas, nelec_u)
    if not tb.sigma_ok or casscf.ci_response_space > 7 \
            or np.size(ci0) <= fs.pspace_size:
        return mc1step.CASSCF.solve_approx_ci(casscf, h1, h2, ci0, ecore,
                                              e_cas, envs)
    h2eff = cp.asarray(fs.absorb_h1e(h1, h2, ncas, nelec_u, .5))

    def contract_2e(c):
        return fci_gpu._contract_2e_dev(h2eff, c, ncas, nelec_u).ravel()

    c0 = cp.asarray(ci0).ravel()
    hc = contract_2e(c0)
    g = hc - (e_cas - ecore) * c0
    nd = min(casscf.ci_response_space, c0.size)
    xs = [c0]
    ax = [hc]
    heff = np.empty((nd, nd))
    seff = np.empty((nd, nd))
    heff[0, 0] = float(xs[0] @ ax[0])
    seff[0, 0] = 1
    tol_residual = fs.conv_tol ** .5
    for i in range(1, nd):
        dx = ax[i - 1] - xs[i - 1] * e_cas
        if float(cp.linalg.norm(dx)) < tol_residual:
            break
        xs.append(dx)
        ax.append(contract_2e(dx))
        hcol = cp.asnumpy(cp.stack(ax) @ xs[i])       # x_i . A x_j
        srow = cp.asnumpy(cp.stack(xs) @ xs[i])
        for j in range(i + 1):
            heff[i, j] = heff[j, i] = hcol[j]
            seff[i, j] = seff[j, i] = srow[j]
    nd = len(xs)
    e, v, seig = lib.safe_eigh(heff[:nd, :nd], seff[:nd, :nd])
    ci1 = cp.asarray(v[:, 0]) @ cp.stack(xs)
    shape = np.shape(ci0)
    return cp.asnumpy(ci1).reshape(shape), cp.asnumpy(g)


class OrbStepMixin:
    """The device orbital step on a GPU CASSCF (``ELECTROLITES_HOST_ORB=1``:
    PySCF's host code)."""

    def gen_g_hop(self, mo, u, casdm1, casdm2, eris):
        if _host():
            from pyscf.mcscf import mc1step
            return mc1step.gen_g_hop(self, mo, u, casdm1, casdm2, eris)
        return gen_g_hop(self, mo, u, casdm1, casdm2, eris)

    def update_casdm(self, mo, u, fcivec, e_cas, eris, envs={}):
        if _host():
            from pyscf.mcscf import mc1step
            return mc1step.CASSCF.update_casdm(self, mo, u, fcivec, e_cas,
                                               eris, envs)
        return update_casdm(self, mo, u, fcivec, e_cas, eris, envs)

    def solve_approx_ci(self, h1, h2, ci0, ecore, e_cas, envs):
        if _host():
            from pyscf.mcscf import mc1step
            return mc1step.CASSCF.solve_approx_ci(self, h1, h2, ci0, ecore,
                                                  e_cas, envs)
        return solve_approx_ci(self, h1, h2, ci0, ecore, e_cas, envs)


def _host():
    import os
    return bool(os.environ.get('ELECTROLITES_HOST_ORB'))


# --------------------------------------- newton_casscf.gen_g_hop (SA gradients)
def newton_gen_g_hop(casscf, mo, ci0, eris, verbose=None):
    """PySCF ``newton_casscf.gen_g_hop`` (the coupled orbital-CI gradient,
    Hessian product and diagonal, also SA-CASSCF's Lagrange equations) with
    every orbital-space term on the device; the CI-space terms use the
    solver's sigma (GPU) on PySCF's host CI vectors as PySCF does.  Not for
    a state-average-mix solver (PySCF's is used there)."""
    from pyscf.mcscf import newton_casscf, addons
    if isinstance(casscf.fcisolver, addons.StateAverageMixFCISolver):
        return _ORIG_NEWTON_GEN_G_HOP(casscf, mo, ci0, eris, verbose)
    ncas, ncore = casscf.ncas, casscf.ncore
    nocc = ncas + ncore
    nelecas = casscf.nelecas
    nmo = mo.shape[1]
    n2 = ncas * ncas
    ci0, _Hci, _Hdiag, linkstrl, linkstr, _pack_ci, _unpack_ci = \
        newton_casscf._pack_ci_get_H(casscf, mo, ci0)
    weights = getattr(casscf, 'weights', [1.0])
    nroots = len(weights)
    w_d = cp.asarray(weights, dtype=np.float64)
    d = _dev_eris(eris)
    ppaa, papa, vhf_c = d['ppaa'], d['papa'], d['vhf_c']
    mo_d = cp.asarray(mo)
    try:
        casdm1, casdm2 = casscf.fcisolver.states_make_rdm12(
            ci0, ncas, nelecas, link_index=linkstr)
        casdm1 = cp.asarray(np.asarray(casdm1))
        casdm2 = cp.asarray(np.asarray(casdm2))
    except AttributeError as e:
        assert not isinstance(casscf.fcisolver, addons.StateAverageFCISolver), e
        casdm1, casdm2 = casscf.fcisolver.make_rdm12(ci0, ncas, nelecas,
                                                     link_index=linkstr)
        casdm1 = cp.asarray(casdm1)[None]
        casdm2 = cp.asarray(casdm2)[None]
    ar = cp.arange(nocc)
    kii = papa[ar, :, ar, :]
    jii = ppaa[ar, ar]
    jkcaa = cp.einsum('iuv,ruv->riu', 6 * kii - 2 * jii, casdm1)
    vhf_a = (cp.einsum('iquv,ruv->riq', ppaa, casdm1)
             - .5 * cp.einsum('iuqv,ruv->riq', papa, casdm1))
    dm2tmp = (casdm2.transpose(0, 2, 3, 1, 4)
              + casdm2.transpose(0, 1, 3, 2, 4)).reshape(nroots, n2, n2)
    pp2 = ppaa.reshape(nmo * nmo, n2)
    kk2 = cp.ascontiguousarray(papa.transpose(0, 2, 1, 3)).reshape(nmo * nmo, n2)
    au = cp.arange(ncas)
    hdm2 = cp.zeros((nmo, ncas, nmo, ncas))
    g_dm2 = cp.empty((nroots, nmo, ncas))
    for r in range(nroots):
        jtmp = (pp2 @ casdm2[r].reshape(n2, n2)).reshape(nmo, nmo, ncas, ncas)
        ktmp = (kk2 @ dm2tmp[r]).reshape(nmo, nmo, ncas, ncas)
        hdm2 += weights[r] * (ktmp + jtmp).transpose(0, 2, 1, 3)
        g_dm2[r] = jtmp[:, ncore + au, au, :].sum(axis=1)
        jtmp = ktmp = None
    eri_cas = cp.asnumpy(ppaa[ncore:nocc, ncore:nocc])
    vhf_ca_r = vhf_c[None] + vhf_a
    h1e_mo = mo_d.T @ cp.asarray(casscf.get_hcore()) @ mo_d

    gpq = cp.zeros((nroots, nmo, nmo))
    gpq[:, :, :ncore] = (h1e_mo[None, :, :ncore] + vhf_ca_r[:, :, :ncore]) * 2
    gpq[:, :, ncore:nocc] = cp.einsum(
        'pu,ruv->rpv', h1e_mo[:, ncore:nocc] + vhf_c[:, ncore:nocc], casdm1)
    gpq[:, :, ncore:nocc] += g_dm2

    vhf_ca = cp.einsum('r,rpq->pq', w_d, vhf_ca_r)
    casdm1 = cp.einsum('r,rpq->pq', w_d, casdm1)
    jkcaa = cp.einsum('r,rpq->pq', w_d, jkcaa)
    casdm1_h = cp.asnumpy(casdm1)

    h1cas_0 = cp.asnumpy(h1e_mo[ncore:nocc, ncore:nocc]
                         + vhf_c[ncore:nocc, ncore:nocc])
    hci0 = _Hci(h1cas_0, eri_cas, ci0)
    eci0 = [c.dot(hc) for c, hc in zip(ci0, hci0)]
    gci = [hc - c * e for hc, c, e in zip(hci0, ci0, eci0)]
    pk = _Packer(casscf, nmo)

    def g_update(u, fcivec):
        if not isinstance(casscf, addons.StateAverageMCSCFSolver):
            fcivec = [fcivec]
        u = cp.asarray(u)
        uc = u[:, :ncore]
        ua = cp.ascontiguousarray(u[:, ncore:nocc])
        ra = cp.ascontiguousarray((u - cp.eye(nmo))[:, ncore:nocc])
        mo1 = mo_d @ u
        mo_c = mo_d @ uc
        mo_a = mo_d @ ua
        dm_c = mo_c @ mo_c.T * 2
        fcivec = [c / np.linalg.norm(c) for c in fcivec]
        fciarg = fcivec if casscf.fcisolver.nroots > 1 else fcivec[0]
        cd1, cd2 = casscf.fcisolver.make_rdm12(fciarg, ncas, nelecas,
                                               link_index=linkstr)
        cd1, cd2 = cp.asarray(cd1), cp.asarray(cd2)
        dm_a = mo_a @ cd1 @ mo_a.T
        vj, vk = _jk(casscf, (dm_c, dm_a))
        vhf_c1 = mo1.T @ (vj[0] - vk[0] * .5) @ mo1[:, :nocc]
        vhf_a1 = mo1.T @ (vj[1] - vk[1] * .5) @ mo1[:, :nocc]
        h1e_mo1 = u.T @ h1e_mo @ u[:, :nocc]
        p1aa = cp.matmul(ua.T[None], ppaa.reshape(nmo, nmo, n2))
        paa1 = (cp.ascontiguousarray(papa.transpose(0, 1, 3, 2)
                                     ).reshape(nmo * n2, nmo) @ ra)
        aaaa = ppaa[ncore:nocc, ncore:nocc]
        aa11 = (ua.T @ p1aa.reshape(nmo, -1)).reshape((ncas,) * 4)
        aa11 = aa11 + aa11.transpose(2, 3, 0, 1) - aaaa
        a11a = (ra.T @ paa1.reshape(nmo, -1)).reshape((ncas,) * 4)
        a11a = a11a + a11a.transpose(1, 0, 2, 3)
        a11a = a11a + a11a.transpose(0, 1, 3, 2)
        eri_cas_2 = cp.asnumpy(aa11 + a11a)
        h1cas_2 = cp.asnumpy(h1e_mo1[ncore:nocc, ncore:nocc]
                             + vhf_c1[ncore:nocc, ncore:nocc])
        hci = _Hci(h1cas_2, eri_cas_2, fcivec)
        gc = [hc - c * c.dot(hc) for c, hc in zip(fcivec, hci)]
        gc = [g_ * w for g_, w in zip(gc, weights)]
        gg = cp.zeros_like(h1e_mo)
        gg[:, :ncore] = (h1e_mo1[:, :ncore] + vhf_c1[:, :ncore]
                         + vhf_a1[:, :ncore]) * 2
        gg[:, ncore:nocc] = (h1e_mo1[:, ncore:nocc]
                             + vhf_c1[:, ncore:nocc]) @ cd1
        p1aa = (u.T @ p1aa.reshape(nmo, -1)).reshape(nmo, ncas, ncas, ncas)
        paa1 = (u.T @ paa1.reshape(nmo, -1)).reshape(nmo, ncas, ncas, ncas)
        p1aa += paa1
        p1aa += paa1.transpose(0, 1, 3, 2)
        gg[:, ncore:nocc] += (
            cp.ascontiguousarray(p1aa).reshape(nmo, -1)
            @ cp.ascontiguousarray(cd2.transpose(2, 0, 1, 3)).reshape(-1, ncas))
        g_orb = cp.asnumpy(pk.pack(gg - gg.T))
        return np.hstack((g_orb * 2, _pack_ci(gc) * 2))

    # hessian diagonal
    dm1 = cp.zeros((nmo, nmo))
    ic = cp.arange(ncore)
    dm1[ic, ic] = 2
    dm1[ncore:nocc, ncore:nocc] = casdm1
    h1d = cp.diag(h1e_mo)
    h_diag = h1d[:, None] * cp.diag(dm1)[None, :] - h1e_mo * dm1
    h_diag = h_diag + h_diag.T
    g_diag = cp.einsum('r,rpp->p', w_d, gpq)
    h_diag -= g_diag + g_diag.reshape(-1, 1)
    idx = cp.arange(nmo)
    h_diag[idx, idx] += g_diag * 2
    v_diag = cp.diag(vhf_ca)
    h_diag[:, :ncore] += v_diag.reshape(-1, 1) * 2
    h_diag[:ncore] += v_diag * 2
    h_diag[ic, ic] -= v_diag[:ncore] * 4
    tmp = cp.diag(vhf_c)[:, None] * cp.diag(casdm1)[None, :]
    h_diag[:, ncore:nocc] += tmp
    h_diag[ncore:nocc, :] += tmp.T
    tmp = -vhf_c[ncore:nocc, ncore:nocc] * casdm1
    h_diag[ncore:nocc, ncore:nocc] += tmp + tmp.T
    tmp = 6 * d['k_pc'] - 2 * d['j_pc']
    h_diag[ncore:, :ncore] += tmp[ncore:]
    h_diag[:ncore, ncore:] += tmp[ncore:].T
    h_diag[:nocc, ncore:nocc] -= jkcaa
    h_diag[ncore:nocc, :nocc] -= jkcaa.T
    v_diag = hdm2[idx[:, None], au[None, :], idx[:, None], au[None, :]]
    h_diag[ncore:nocc, :] += v_diag.T
    h_diag[:, ncore:nocc] += v_diag
    h_diag = cp.asnumpy(pk.pack(h_diag))

    hci_diag = [hd - ec - gc * c * 2 for hd, ec, gc, c in
                zip(_Hdiag(h1cas_0, eri_cas), eci0, gci, ci0)]
    hci_diag = [h * w for h, w in zip(hci_diag, weights)]
    hdiag_all = np.hstack((h_diag * 2, _pack_ci(hci_diag) * 2))

    g_orb_m = cp.einsum('r,rpq->pq', w_d, gpq)
    g_orb = cp.asnumpy(pk.pack(g_orb_m - g_orb_m.T))
    gci = [g_ * w for g_, w in zip(gci, weights)]
    g_all = np.hstack((g_orb * 2, _pack_ci(gci) * 2))
    ngorb = g_orb.size

    gpg = (g_orb_m + g_orb_m.T) * .5
    hdm2_2 = hdm2.reshape(nmo * ncas, nmo * ncas)
    hv_c = h1e_mo + vhf_c
    paaa = ppaa[:, ncore:nocc]                           # (p, u, w, x)
    pc = ppaa[:, :ncore]                                 # (i, q<ncore, u, v)
    kc = papa[:, :, :ncore]                              # (i, u, q<ncore, v)
    mo_core = mo_d[:, :ncore]
    mo_cas = mo_d[:, ncore:nocc]

    def h_op(x):
        x1 = pk.unpack(x[:ngorb])
        ci1 = _unpack_ci(x[ngorb:])
        # H_cc
        hci1 = [hc1 - c1 * ec0 for hc1, c1, ec0 in
                zip(_Hci(h1cas_0, eri_cas, ci1), ci1, eci0)]
        hci1 = [hc1 - (hc0 - c0 * ec0) * c0.dot(c1) for hc1, hc0, c0, ec0, c1
                in zip(hci1, hci0, ci0, eci0, ci1)]
        hci1 = [hc1 - c0 * (hc0 - c0 * ec0).dot(c1) for hc1, hc0, c0, ec0, c1
                in zip(hci1, hci0, ci0, eci0, ci1)]
        # H_co
        rc = x1[:, :ncore]
        ra = cp.ascontiguousarray(x1[:, ncore:nocc])
        ddm_c = cp.zeros((nmo, nmo))
        ddm_c[:, :ncore] = rc[:, :ncore] * 2
        ddm_c[:ncore, :] += rc[:, :ncore].T * 2
        tdm1, tdm2 = casscf.fcisolver.trans_rdm12(ci1, ci0, ncas, nelecas,
                                                  link_index=linkstr)
        tdm1 = cp.asarray(tdm1)
        tdm2 = cp.asarray(tdm2)
        tdm1 = tdm1 + tdm1.T
        tdm2 = tdm2 + tdm2.transpose(1, 0, 3, 2)
        tdm2 = (tdm2 + tdm2.transpose(2, 3, 0, 1)) * .5
        vhf_at = (pc.reshape(nmo, ncore, n2) @ tdm1.ravel()
                  - .5 * cp.einsum('iuqv,uv->iq', kc, tdm1))
        jk = ((ddm_c.ravel() @ ppaa.reshape(nmo * nmo, n2)).reshape(ncas, ncas)
              - .5 * cp.einsum('iuqv,iq->uv', papa, ddm_c))
        g_dm2t = (cp.ascontiguousarray(paaa).reshape(nmo, -1)
                  @ cp.ascontiguousarray(tdm2.transpose(2, 0, 1, 3)
                                         ).reshape(-1, ncas))
        aaaa = (ra.T @ paaa.reshape(nmo, -1)).reshape((ncas,) * 4)
        aaaa = aaaa + aaaa.transpose(1, 0, 2, 3)
        aaaa = aaaa + aaaa.transpose(2, 3, 0, 1)
        h1aa = hv_c[ncore:nocc] @ ra
        h1aa = h1aa + h1aa.T + jk
        kci0 = _Hci(cp.asnumpy(h1aa), cp.asnumpy(aaaa), ci0)
        kci0 = [kc0 - kc0.dot(c0) * c0 for kc0, c0 in zip(kci0, ci0)]
        hci1 = [hc1 + kc0 for hc1, kc0 in zip(hci1, kci0)]
        hci1 = [h * w for h, w in zip(hci1, weights)]
        # H_oo
        x2 = h1e_mo @ x1 @ dm1
        x2 -= gpg @ x1
        x2[:ncore] += x1[:ncore, ncore:] @ vhf_ca[ncore:] * 2
        x2[ncore:nocc] += casdm1 @ x1[ncore:nocc] @ vhf_c
        x2[:, ncore:nocc] += (hdm2_2 @ x1[:, ncore:nocc].ravel()
                              ).reshape(nmo, ncas)
        if ncore > 0:
            va, vc = _update_jk_in_ah(casscf, mo_d, mo_core, mo_cas, x1,
                                      casdm1, ncore, nocc)
            x2[ncore:nocc] += va
            x2[:ncore, ncore:] += vc
        # H_oc
        s10 = [c1.dot(c0) * 2 * w for c1, c0, w in zip(ci1, ci0, weights)]
        x2[:, :ncore] += (hv_c[:, :ncore] * sum(s10) + vhf_at) * 2
        x2[:, ncore:nocc] += hv_c[:, ncore:nocc] @ tdm1
        x2[:, ncore:nocc] += g_dm2t
        x2 -= cp.einsum('r,rpq->pq', cp.asarray(s10), gpq)
        x2 = x2 - x2.T
        return np.hstack((cp.asnumpy(pk.pack(x2)) * 2, _pack_ci(hci1) * 2))

    return g_all, g_update, h_op, hdiag_all


def _orig_newton():
    from pyscf.mcscf import newton_casscf
    return newton_casscf.gen_g_hop


_ORIG_NEWTON_GEN_G_HOP = _orig_newton()
