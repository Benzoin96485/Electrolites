"""Analytic AO, RHF and occupied-localization responses.

The integral derivatives use libcint's spatial derivatives and nuclear
center masks.  Nuclear coordinates and returned derivatives are in bohr.
The CPHF uses the reference's own J/K operator, including density fitting.
This reference implementation retains full AO three-center tensors.
"""
from types import SimpleNamespace

import numpy as np
from pyscf import df, lo
from scipy.sparse.linalg import LinearOperator, gmres


def three_center_derivatives(mol, auxmol):
    """Yield (atom, Cartesian, d(mu nu|P), d(P|Q))."""
    nao, naux = mol.nao_nr(), auxmol.nao_nr()
    ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1', aosym='s1', comp=3)
    ip2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2', aosym='s1', comp=3)
    metric_ip = auxmol.intor('int2c2e_ip1', comp=3)
    ao_slices, aux_slices = mol.aoslice_by_atom(), auxmol.aoslice_by_atom()
    for atom in range(mol.natm):
        p0, p1 = ao_slices[atom, 2:]
        q0, q1 = aux_slices[atom, 2:]
        for xyz in range(3):
            b = np.zeros((nao, nao, naux))
            b[p0:p1] -= ip1[xyz, p0:p1]
            b[:, p0:p1] -= ip1[xyz, p0:p1].transpose(1, 0, 2)
            b[:, :, q0:q1] -= ip2[xyz, :, :, q0:q1]
            m = np.zeros((naux, naux))
            m[q0:q1] -= metric_ip[xyz, q0:q1]
            m[:, q0:q1] -= metric_ip[xyz, q0:q1].T
            yield atom, xyz, b, m


def one_center_derivatives(mol):
    """AO overlap and dipole derivatives, with a fixed origin at zero."""
    ip_s = mol.intor('int1e_ipovlp', comp=3)
    with mol.with_common_orig((0., 0., 0.)):
        rp = mol.intor('int1e_irp', comp=9).reshape(3, 3, mol.nao, mol.nao)
    for atom, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        for xyz in range(3):
            s = np.zeros((mol.nao, mol.nao))
            s[p0:p1] -= ip_s[xyz, p0:p1]
            s[:, p0:p1] -= ip_s[xyz, p0:p1].T
            r = np.zeros((3, mol.nao, mol.nao))
            r[:, :, p0:p1] -= rp[:, xyz, :, p0:p1]
            r[:, p0:p1, :] -= rp[:, xyz, :, p0:p1].transpose(0, 2, 1)
            yield atom, xyz, s, r


def rhf_response(mf, s1, f1, tol=1e-11):
    """dC_occ and dF from one explicit AO Fock/overlap perturbation.

    Restore the canonical occupied gauge after CPHF's symmetric gauge.
    This is essential when a frozen core is separated from active LMOs.
    """
    c = np.asarray(mf.mo_coeff)
    occ, eps = np.asarray(mf.mo_occ), np.asarray(mf.mo_energy)
    co = c[:, occ > 0]
    nocc = co.shape[1]
    s_mo = c.T @ s1 @ co
    h_mo = c.T @ f1 @ co

    def vind(u):
        u = u.reshape(-1, c.shape[1], nocc)
        dm = np.einsum('mp,xpi,ni->xmn', c, u, co) * 2
        dm += dm.transpose(0, 2, 1)
        v = np.asarray(mf.get_veff(mf.mol, dm))
        return np.einsum('mp,xmn,ni->xpi', c, v, co).reshape(u.shape)

    u = np.zeros_like(h_mo)
    u[:nocc] = -.5 * s_mo[:nocc]
    den = eps[nocc:, None] - eps[:nocc]
    rhs = -(h_mo - s_mo * eps[:nocc] + vind(u)[0])[nocc:]

    def action(flat):
        trial = np.zeros_like(u)
        trial[nocc:] = flat.reshape(den.shape)
        return (den * trial[nocc:] + vind(trial)[0, nocc:]).ravel()

    operator = LinearOperator((rhs.size, rhs.size), matvec=action, dtype=float)
    pre = LinearOperator(operator.shape, matvec=lambda x: x / den.ravel(), dtype=float)
    response, status = gmres(operator, rhs.ravel(), M=pre, rtol=tol, atol=tol * .01,
                            maxiter=100, restart=min(rhs.size, 50))
    if status:
        raise RuntimeError(f'CPHF GMRES did not converge: {status}')
    u[nocc:] = response.reshape(den.shape)
    dc = c @ u
    dm1 = 2 * (dc @ co.T + co @ dc.T)
    f_total = f1 + np.asarray(mf.get_veff(mf.mol, dm1))
    f_mo = c.T @ f_total @ c
    s_full = c.T @ s1 @ c
    eo = eps[:nocc]
    gap = eo[None, :] - eo[:, None]
    mask = np.abs(gap) > 1e-10
    uoo = (f_mo[:nocc, :nocc] - s_full[:nocc, :nocc] * eo) / np.where(mask, gap, 1.)
    u[:nocc] = np.where(mask, uoo, -.5 * s_full[:nocc, :nocc])
    residual = (f_mo[nocc:, :nocc] - s_full[nocc:, :nocc] * eo
                + (eps[nocc:, None] - eo) * u[nocc:])
    error = float(np.max(np.abs(residual), initial=0.))
    if error > max(1e-9, 100 * tol):
        raise RuntimeError(f'CPHF response residual {error:.3e}')
    return c @ u, f_total, error


def boys_response(mol, c_lmo, dc_transport, r, dr, tol=1e-9):
    """Solve the coupled-perturbed Boys equations in active occupied space.

    Zero Hessian modes are parallel transported, as in Pinski--Neese
    (2019), Eqs. 80--85.  The discarded RHS norm is returned for diagnosis.
    """
    n = c_lmo.shape[1]
    if n < 2:
        return dc_transport, dict(residual=0., null_rhs=0., nullity=0., stationary_residual=0.)
    loc = lo.Boys(mol, c_lmo)
    g, hop, _ = loc.gen_g_hop(np.eye(n))
    if np.max(np.abs(g)) > 1e-8:
        raise RuntimeError('Boys orbitals are not stationary enough for analytic response '
                           f'(residual={np.max(np.abs(g)):.3e}); use stationary_boys '
                           'and recompute the energy with the polished c_lmo')
    dip = np.einsum('mi,xmn,nj->xij', c_lmo, r, c_lmo)
    ddip = (np.einsum('mi,xmn,nj->xij', dc_transport, r, c_lmo)
            + np.einsum('mi,xmn,nj->xij', c_lmo, dr, c_lmo)
            + np.einsum('mi,xmn,nj->xij', c_lmo, r, dc_transport))
    dg0 = (np.einsum('xii,xip->pi', ddip, dip)
           + np.einsum('xii,xip->pi', dip, ddip))
    rhs = -2 * loc.pack_uniq_var(dg0 - dg0.T)
    eye = np.eye(len(g))
    h = np.column_stack([hop(v) for v in eye])
    h = (h + h.T) * .5
    w, v = np.linalg.eigh(h)
    keep = np.abs(w) > tol
    projected = v.T @ rhs
    k = -v[:, keep] @ (projected[keep] / w[keep])
    err = np.max(np.abs(v[:, keep].T @ (h @ k + rhs)), initial=0.)
    info = dict(residual=float(err), null_rhs=float(np.linalg.norm(projected[~keep])),
                nullity=int((~keep).sum()), stationary_residual=float(np.max(np.abs(g))))
    return dc_transport + c_lmo @ loc.unpack_uniq_var(k), info


def validate_reference(reference):
    """Check derivative scope without constructing derivative integrals."""
    mf = reference.mf.to_cpu() if hasattr(reference.mf, 'to_cpu') else reference.mf
    mol, occ = mf.mol, np.asarray(mf.mo_occ)
    if mol.spin or occ.ndim != 1 or not np.all((occ == 0) | (occ == 2)):
        raise NotImplementedError('The current response driver requires closed-shell RHF')
    if hasattr(mf, 'xc') or mol.ecp or getattr(mf, 'with_x2c', None):
        raise NotImplementedError('Response currently supports nonrelativistic all-electron RHF')
    if getattr(mf, 'with_solvent', None) or getattr(mol, 'pbc_intor', None):
        raise NotImplementedError('Solvent and periodic responses are not implemented')
    if np.iscomplexobj(mf.mo_coeff):
        raise NotImplementedError('Complex orbital response is not implemented')
    if not mf.converged:
        raise RuntimeError('A converged RHF reference is required')
    if reference.localizer.lower() not in ('boys', 'fb'):
        raise NotImplementedError('Only Boys localization response is implemented')
    return mf


class RHFResponse:
    """Geometry independent preparation shared by all nuclear directions."""

    def __init__(self, reference):
        self.mf = validate_reference(reference)
        self.ref = reference
        mol = self.mf.mol
        self.s = np.asarray(self.mf.get_ovlp())
        self.f = np.asarray(self.mf.get_fock())
        self.c = np.asarray(self.mf.mo_coeff)
        self.co = self.c[:, np.asarray(self.mf.mo_occ) > 0]
        self.l = reference.orb.C_lmo
        nc = reference.orb.ncore
        self.rotation = self.c[:, nc:self.co.shape[1]].T @ self.s @ self.l
        with mol.with_common_orig((0., 0., 0.)):
            self.r = mol.intor('int1e_r', comp=3)
        hess = self.mf.Hessian()
        if hasattr(hess, 'auxbasis_response'):
            hess.auxbasis_response = 2
        self.f1 = hess.make_h1(self.mf.mo_coeff, self.mf.mo_occ)
        self.int3c = df.incore.aux_e2(mol, reference.auxmol, 'int3c2e')
        self.metric = reference.auxmol.intor('int2c2e')
        self.hf_gradient = self.mf.nuc_grad_method().kernel()
        self.diagnostics = []

    def directions(self):
        mol, ref = self.mf.mol, self.ref
        for one, three in zip(one_center_derivatives(mol), three_center_derivatives(mol, ref.auxmol)):
            atom, xyz, ds, dr = one
            a2, x2, db, dm = three
            assert (atom, xyz) == (a2, x2)
            dc, df_total, error = rhf_response(self.mf, ds, self.f1[atom][xyz])
            dc_active = dc[:, ref.orb.ncore:]
            dl, info = boys_response(mol, self.l, dc_active @ self.rotation, self.r, dr)
            info.update(atom=atom, xyz=xyz, cphf_residual=error)
            self.diagnostics.append(info)
            yield SimpleNamespace(atom=atom, xyz=xyz, ds=ds, dr=dr, db=db, dm=dm,
                                  dc=dc, df=df_total, dl=dl)
