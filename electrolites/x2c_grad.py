"""Spin-free X2C (sfX2C1e) nuclear gradients on the GPU.

GPU4PySCF's ``SFX2C1E_SCF.Gradients`` raises; PySCF's
``x2c.sfx2c1e_grad`` is on the host.  The only X2C-specific piece of the
gradient is dh/dR_A of the one-electron Foldy-Wouthuysen Hamiltonian.  PySCF
builds it per atom from the response of the (2n x 2n) one-electron Dirac
problem on the uncontracted basis: first-order MO coefficients and X
matrix, the renormalisation R^(1) (JCP 135, 084114, Eq. 34), and
h_FW^(1) -- a dozen (2n)^3 GEMMs, a linear solve and the half-transforms,
per atom and Cartesian direction.  Here the same algebra runs on the device
(the derivative integrals on the uncontracted basis stay host libcint
calls, once for the common ones and two per atom for the nucleus-centred
1/r operators), batched over the three directions.  Everything else of the
gradient is GPU4PySCF's RHF/RKS/UHF/UKS gradient: ``_hcore_energy`` is the
only method replaced.

    mf = gpu4pyscf.scf.RHF(mol).x2c().run()        # or RKS/UKS
    g = electrolites.x2c_grad.Gradients(mf).kernel()
"""
import numpy as np
import cupy as cp

from pyscf import lib


def _eigh_gen(h, s):
    L = cp.linalg.cholesky(s)
    Li = cp.linalg.inv(L)
    e, y = cp.linalg.eigh(Li @ h @ Li.T)
    return e, Li.T @ y


def _get_r(s, snesc):
    w, v = cp.linalg.eigh(s)
    idx = w > 1e-14
    v = v[:, idx]
    w_sqrt = cp.sqrt(w[idx])
    w_invsqrt = 1 / w_sqrt
    snesc = v.T @ snesc @ v
    r_mid = w_invsqrt[:, None] * snesc * w_invsqrt
    w1, v1 = cp.linalg.eigh(r_mid)
    idx1 = w1 > 1e-14
    v1 = v1[:, idx1]
    r_mid = (v1 / cp.sqrt(w1[idx1])) @ v1.T
    r = w_invsqrt[:, None] * r_mid * w_sqrt
    return v @ r @ v.T


def _get_r1(w_sqrt, v_s, s_nesc0, s1, s_nesc1, wr0_sqrt, vr0):
    """R^(1) for a batch of directions (PySCF ``_get_r1``)."""
    w_invsqrt = 1. / w_sqrt
    wr0_invsqrt = 1. / wr0_sqrt
    s1 = v_s.T @ s1 @ v_s
    s_nesc1 = v_s.T @ s_nesc1 @ v_s
    s1_sqrt = s1 / (w_sqrt[:, None] + w_sqrt)
    s1_invsqrt = ((w_invsqrt ** 2)[:, None] * s1 * w_invsqrt ** 2
                  / -(w_invsqrt[:, None] + w_invsqrt))
    R1_mid = (s1_invsqrt @ s_nesc0) * w_invsqrt
    R1_mid = R1_mid + R1_mid.transpose(0, 2, 1)
    R1_mid += w_invsqrt[:, None] * s_nesc1 * w_invsqrt
    R1_mid = vr0.T @ R1_mid @ vr0
    R1_mid /= -(wr0_invsqrt[:, None] + wr0_invsqrt)
    R1_mid = (wr0_invsqrt ** 2)[:, None] * R1_mid * wr0_invsqrt ** 2
    vr0_wr0_sqrt = vr0 * wr0_invsqrt
    vr0_s0_sqrt = vr0.T * w_sqrt
    vr0_s0_invsqrt = vr0.T * w_invsqrt
    R1 = vr0_s0_invsqrt.T @ R1_mid @ vr0_s0_sqrt
    R1 += s1_invsqrt @ (vr0_wr0_sqrt @ vr0_s0_sqrt)
    R1 += (vr0_s0_invsqrt.T @ vr0_wr0_sqrt.T) @ s1_sqrt
    return v_s @ R1 @ v_s.T


def _gen_sf_hfw(mol, approx='1E'):
    """dh_FW/dR_A generator on the device (PySCF ``gen_sf_hfw``)."""
    from pyscf.x2c import x2c as _x2c
    approx = approx.upper()
    c = lib.param.LIGHT_SPEED
    s = mol.intor_symmetric('int1e_ovlp')
    t = mol.intor_symmetric('int1e_kin')
    v = mol.intor_symmetric('int1e_nuc')
    w = mol.intor_symmetric('int1e_pnucp')
    n = s.shape[0]
    n2 = 2 * n
    h0 = np.zeros((n2, n2))
    s0 = np.zeros((n2, n2))
    h0[:n, :n] = v
    h0[:n, n:] = t
    h0[n:, :n] = t
    h0[n:, n:] = w * (.25 / c ** 2) - t
    s0[:n, :n] = s
    s0[n:, n:] = t * (.5 / c ** 2)
    h0 = cp.asarray(h0)
    s0 = cp.asarray(s0)
    e0, c0 = _eigh_gen(h0, s0)
    aoslices = mol.aoslice_by_atom()
    if 'ATOM' in approx:
        x0 = np.zeros((n, n))
        for ia in range(mol.natm):
            ish0, ish1, p0, p1 = aoslices[ia]
            sl = (ish0, ish1, ish0, ish1)
            t1 = mol.intor('int1e_kin', shls_slice=sl)
            s1 = mol.intor('int1e_ovlp', shls_slice=sl)
            with mol.with_rinv_at_nucleus(ia):
                z = -mol.atom_charge(ia)
                v1 = z * mol.intor('int1e_rinv', shls_slice=sl)
                w1 = z * mol.intor('int1e_prinvp', shls_slice=sl)
            x0[p0:p1, p0:p1] = _x2c._x2c1e_xmatrix(t1, v1, w1, s1, c)
        x0 = cp.asarray(x0)
    else:
        cl0 = c0[:n, n:]
        cs0 = c0[n:, n:]
        x0 = cp.linalg.solve(cl0.T, cs0.T).T
    s_d = cp.asarray(s)
    t_d = cp.asarray(t)
    s_nesc0 = s_d + x0.T @ s0[n:, n:] @ x0
    R0 = _get_r(s_d, s_nesc0)
    c_fw0 = cp.vstack((R0, x0 @ R0))
    h0_fw_half = h0 @ c_fw0

    # first-order machinery (PySCF ``_gen_first_order_quantities``)
    epq = e0[:, None] - e0
    degen = abs(epq) < 1e-7
    epq[degen] = 1e200
    cl0 = c0[:n, n:]
    t0x0 = (t_d @ x0) * (.5 / c ** 2)
    s_nesc0 = s_d + x0.T @ t0x0
    w_s, v_s = cp.linalg.eigh(s_d)
    w_sqrt = cp.sqrt(w_s)
    s_nesc0_vbas = v_s.T @ s_nesc0 @ v_s
    R0_mid = (1. / w_sqrt)[:, None] * s_nesc0_vbas * (1. / w_sqrt)
    wr0, vr0 = cp.linalg.eigh(R0_mid)
    wr0_sqrt = cp.sqrt(wr0)
    R0b = (vr0 / wr0_sqrt) @ vr0.T
    R0b *= w_sqrt
    R0b /= w_sqrt[:, None]
    R0b = v_s @ R0b @ v_s.T

    s1g = cp.asarray(mol.intor('int1e_ipovlp', comp=3))
    t1g = cp.asarray(mol.intor('int1e_ipkin', comp=3))
    v1g = cp.asarray(mol.intor('int1e_ipnuc', comp=3))
    w1g = cp.asarray(mol.intor('int1e_ippnucp', comp=3))

    def get_h1_s1(ia):
        ish0, ish1, i0, i1 = aoslices[ia]
        with mol.with_rinv_origin(mol.atom_coord(ia)):
            z = mol.atom_charge(ia)
            rinv1 = cp.asarray(-z * mol.intor('int1e_iprinv', comp=3))
            prinvp1 = cp.asarray(-z * mol.intor('int1e_ipprinvp', comp=3))
        rinv1[:, i0:i1, :] -= v1g[:, i0:i1]
        prinvp1[:, i0:i1, :] -= w1g[:, i0:i1]
        s1cc = cp.zeros((3, n, n))
        t1cc = cp.zeros((3, n, n))
        s1cc[:, i0:i1, :] = -s1g[:, i0:i1]
        s1cc[:, :, i0:i1] -= s1g[:, i0:i1].transpose(0, 2, 1)
        t1cc[:, i0:i1, :] = -t1g[:, i0:i1]
        t1cc[:, :, i0:i1] -= t1g[:, i0:i1].transpose(0, 2, 1)
        v1cc = rinv1 + rinv1.transpose(0, 2, 1)
        w1cc = prinvp1 + prinvp1.transpose(0, 2, 1)
        h1 = cp.zeros((3, n2, n2))
        m1 = cp.zeros((3, n2, n2))
        h1[:, :n, :n] = v1cc
        h1[:, :n, n:] = t1cc
        h1[:, n:, :n] = t1cc
        h1[:, n:, n:] = w1cc * (.25 / c ** 2) - t1cc
        m1[:, :n, :n] = s1cc
        m1[:, n:, n:] = t1cc * (.5 / c ** 2)
        return h1, m1

    def hcore_deriv(ia):
        h1ao, s1ao = get_h1_s1(ia)
        if 'ATOM' in approx:
            s_nesc1 = x0.T @ s1ao[:, n:, n:] @ x0 + s1ao[:, :n, :n]
            x1 = None
        else:
            h1mo = c0.T @ h1ao @ c0
            s1mo = c0.T @ s1ao @ c0
            f1 = h1mo[:, :, n:] - s1mo[:, :, n:] * e0[n:]
            c1 = f1 / -epq[:, n:]
            c1_ao = c0 @ c1
            cl1 = c1_ao[:, :n]
            cs1 = c1_ao[:, n:]
            tmp = cs1 - x0 @ cl1
            x1 = cp.linalg.solve(cl0.T, tmp.reshape(-1, n).T)
            x1 = x1.T.reshape(3, n, n)
            s_nesc1 = x1.transpose(0, 2, 1) @ t0x0
            s_nesc1 = s_nesc1 + s_nesc1.transpose(0, 2, 1)
            s_nesc1 += x0.T @ s1ao[:, n:, n:] @ x0
            s_nesc1 += s1ao[:, :n, :n]
        R1 = _get_r1(w_sqrt, v_s, s_nesc0_vbas, s1ao[:, :n, :n], s_nesc1,
                     wr0_sqrt, vr0)
        c_fw1 = cp.empty((3, n2, n))
        c_fw1[:, :n] = R1
        c_fw1[:, n:] = x0 @ R1
        if x1 is not None:
            c_fw1[:, n:] += x1 @ R0b
        hfw1 = c_fw1.transpose(0, 2, 1) @ h0_fw_half
        hfw1 = hfw1 + hfw1.transpose(0, 2, 1)
        hfw1 += c_fw0.T @ h1ao @ c_fw0
        return hfw1

    return hcore_deriv


def hcore_grad_generator(x2cobj, mol=None):
    """dh/dR_A in the molecular basis, device arrays (3, nao, nao).  The
    uncontracted molecule and contraction come from PySCF's helper with
    the same settings (GPU4PySCF's returns a shell-sorted molecule)."""
    from pyscf.x2c import sfx2c1e as _cpu
    if mol is None:
        mol = x2cobj.mol
    helper = _cpu.SpinFreeX2CHelper(mol)
    for key in ('approx', 'xuncontract', 'basis'):
        setattr(helper, key, getattr(x2cobj, key))
    x2cobj = helper
    xmol, contr = x2cobj.get_xmol(mol)
    if x2cobj.basis is not None:
        from pyscf import gto
        s22 = xmol.intor_symmetric('int1e_ovlp')
        s21 = gto.intor_cross('int1e_ovlp', xmol, mol)
        contr = lib.cho_solve(s22, s21)
    get_h1 = _gen_sf_hfw(xmol, x2cobj.approx)
    if contr is not None:
        contr = cp.asarray(contr)

    def hcore_deriv(ia):
        h1 = get_h1(ia)
        if contr is not None:
            h1 = contr.T @ h1 @ contr
        return h1

    return hcore_deriv


class _X2CGradMixin:
    """``_hcore_energy`` with the sfX2C1e one-electron derivative."""

    def _hcore_energy(self, dm0, dme0):
        from gpu4pyscf.grad.rhf import contract_h1e_dm
        mol = self.mol
        dm0 = cp.asarray(dm0)
        if dm0.ndim == 3:
            dm0 = dm0[0] + dm0[1]
        hd = hcore_grad_generator(self.base.with_x2c, mol)
        dh = np.zeros((mol.natm, 3))
        for ia in range(mol.natm):
            dh[ia] = cp.asnumpy(cp.einsum('xij,ij->x', hd(ia), dm0))
        s1 = cp.asarray(self.get_ovlp(mol))
        dh -= contract_h1e_dm(mol, s1, dme0, hermi=1)
        return dh


def Gradients(mf):
    """Nuclear gradient of a GPU4PySCF ``sfx2c1e()`` RHF/RKS/UHF/UKS."""
    if getattr(mf, 'with_x2c', None) is None:
        raise ValueError('not an X2C SCF object')
    base = mf.undo_x2c().Gradients()
    base.base = mf
    return lib.set_class(base, (_X2CGradMixin, base.__class__))
