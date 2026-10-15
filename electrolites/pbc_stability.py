"""Periodic RHF/RKS stability analysis on the GPU
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` X3).

``pyscf.pbc.scf.stability`` cannot run on a GPU4PySCF periodic SCF (a
TypeError in its CPU response).  This is its ``rhf_internal`` and
``rhf_external`` -- the same Hessian-vector products (PySCF's
``pbc.scf.newton_ah.gen_g_hop_rhf`` and ``_gen_hop_rhf_external``) -- with
the transition densities, the response and the MO transforms on the device,
for Gamma-point (``RHF``/``RKS``) and k-point (``KRHF``/``KRKS``) objects:

* internal: the RHF -> RHF Hessian, GPU4PySCF's own ``gen_response``
  (``singlet=None``, the SOSCF response), real rotations, ``.real * 2`` as
  PySCF;
* external (RHF -> UHF): the triplet response (``pbc_tdscf``'s, which also
  fixes D10).

The lowest eigenvalues come from ``electrolites.stability._davidson``.

    mo_i, mo_e, stable_i, stable_e = electrolites.pbc_stability.stability(
        gpu_kmf, internal=True, external=True)
"""
import numpy as np
import cupy as cp

from pyscf import lib
from pyscf.lib import logger


def _k_arrays(mf):
    C = cp.asarray(mf.mo_coeff)
    occ = cp.asnumpy(mf.mo_occ)
    if C.ndim == 2:                               # Gamma point
        return C[None], occ[None], False
    return C, occ, True


def _fock_mo(mf, C, gamma_first):
    dm0 = mf.make_rdm1()
    f = cp.asarray(mf.get_fock(dm=dm0))
    if f.ndim == 2:
        f = f[None]
    return cp.einsum('kpi,kpq,kqj->kij', C.conj(), f, C)


def _ops(mf, triplet):
    C, occ, is_k = _k_arrays(mf)
    nk = C.shape[0]
    no = int((occ[0] == 2).sum())
    if any(int((o == 2).sum()) != no for o in occ):
        raise NotImplementedError('PBC stability: metallic occupations')
    co, cv = C[:, :, :no], C[:, :, no:]
    nv = cv.shape[2]
    fock = _fock_mo(mf, C, is_k)
    foo = fock[:, :no, :no]
    fvv = fock[:, no:, no:]
    hdiag = (cp.diagonal(fvv, axis1=1, axis2=2).real[:, :, None]
             - cp.diagonal(foo, axis1=1, axis2=2).real[:, None, :]) * 2
    if triplet:
        if is_k:
            from electrolites.pbc_tdscf import gen_response_k
            vk = gen_response_k(mf, singlet=False)
            vresp = lambda d: vk(d[None])[0]                  # noqa: E731
        else:
            from electrolites.pbc_tdscf import gen_response
            vg = gen_response(mf, singlet=False, hermi=1)
            vresp = lambda d: vg(d[0])[None]                  # noqa: E731
    else:
        vr = mf.gen_response(singlet=None, hermi=1)
        vresp = (lambda d: cp.asarray(vr(d[None]))[0]) if is_k else \
            (lambda d: cp.asarray(vr(d[0]))[None])

    def hop(xs):
        """rows of real rotations x (nk*nv*no) -> rows of H x."""
        xs = cp.asarray(xs).reshape(-1, nk, nv, no)
        out = cp.empty(xs.shape)
        for z, x in enumerate(xs):
            d1 = 2 * cp.einsum('kpa,kai,kqi->kpq', cv, x, co.conj())
            d1 = d1 + d1.conj().transpose(0, 2, 1)
            if not is_k:
                d1 = d1.real
            v1 = vresp(d1)
            x2 = (cp.einsum('kab,kbi->kai', fvv, x)
                  - cp.einsum('kji,kaj->kai', foo, x)
                  + cp.einsum('kpa,kpq,kqi->kai', cv.conj(), v1, co))
            if not triplet:
                x2 = x2 * 2
            out[z] = x2.real * (1 if triplet else 2)
        return out.reshape(len(xs), -1)
    hd = (hdiag * (2 if not triplet else 1)).ravel()
    return hop, hd, (C, occ, no, nv, is_k)


def _rotate(C, no, nv, v, is_k):
    import scipy.linalg
    nk = C.shape[0]
    v = cp.asnumpy(v).reshape(nk, nv, no)
    out = []
    for k in range(nk):
        nmo = C.shape[2]
        dr = np.zeros((nmo, nmo))
        dr[no:, :no] = v[k]
        dr[:no, no:] = -v[k].T
        out.append(cp.asarray(C[k]) @ cp.asarray(scipy.linalg.expm(dr)))
    out = cp.stack(out)
    return out if is_k else out[0]


def _nguess(hdiag):
    """Start from the 8 lowest diagonal elements and every partner
    degenerate with them (one start vector converged onto a higher root on
    diamond: Gamma RHF->UHF -0.008 for the true -0.129)."""
    hs = cp.sort(hdiag)
    k = min(8, hs.size)
    return min(hs.size, max(k, int((hdiag <= hs[k - 1] + 1e-4).sum())))


def internal(mf, verbose=None, return_status=True):
    from electrolites.stability import _davidson
    log = logger.new_logger(mf, verbose)
    hop, hdiag, (C, occ, no, nv, is_k) = _ops(mf, triplet=False)
    # PySCF: hessian_x = hop(x).real * 2 with hop already x2, so the
    # operator above carries the 4 and hdiag*2 its preconditioner
    # a random start besides the unit vectors: the lowest root can sit in a
    # symmetry block no low-e_ia unit vector touches (2k PBE0 diamond)
    x0 = cp.asarray(np.random.default_rng(7).standard_normal(hdiag.size))
    e, v = _davidson(hop, hdiag, x0, 1, 1e-8, log, nguess=_nguess(hdiag))
    e = float(e[0])
    stable = not (e < -1e-5)
    log.note('GPU %s internal stability: lowest Hessian eigenvalue %.6g (%s)',
             'KRHF/KRKS' if is_k else 'RHF/RKS', e,
             'stable' if stable else 'unstable')
    mo = mf.mo_coeff if stable else _rotate(C, no, nv, v[0], is_k)
    return (mo, stable, e) if return_status else mo


def external(mf, verbose=None, return_status=True):
    from electrolites.stability import _davidson
    log = logger.new_logger(mf, verbose)
    hop, hdiag, (C, occ, no, nv, is_k) = _ops(mf, triplet=True)
    # a random start besides the unit vectors: the lowest root can sit in a
    # symmetry block no low-e_ia unit vector touches (2k PBE0 diamond)
    x0 = cp.asarray(np.random.default_rng(7).standard_normal(hdiag.size))
    e, v = _davidson(hop, hdiag, x0, 1, 1e-8, log, nguess=_nguess(hdiag))
    e = float(e[0])
    stable = not (e < -1e-5)
    log.note('GPU %s RHF->UHF stability: lowest Hessian eigenvalue %.6g (%s)',
             'KRHF/KRKS' if is_k else 'RHF/RKS', e,
             'stable' if stable else 'unstable')
    return (None, stable, e) if return_status else None


def stability(mf, internal_=True, external_=False, verbose=None):
    """(mo_internal, mo_external, stable_internal, stable_external), as
    PySCF's ``rhf_stability(..., return_status=True)``."""
    mo_i = mo_e = s_i = s_e = None
    if internal_:
        mo_i, s_i, _ = internal(mf, verbose)
    if external_:
        mo_e, s_e, _ = external(mf, verbose)
    return mo_i, mo_e, s_i, s_e
