"""Raman activities for UHF/UKS (and RHF/RKS) on the GPU.

GPU4PySCF's ``properties.raman.eval_raman_intensity`` asserts an RHF/RKS
reference; PySCF and pyscf-properties have no Raman at all.  The
polarizability derivatives come, as in GPU4PySCF, from second differences
of analytic nuclear gradients in a uniform field F,

    d alpha_xy / dR_A = - d^2 g_A / dF_x dF_y,

with g_A the gradient plus the field term F . (d mu / dR_A) . D (the
gradient object does not see the field in hcore).  Here for either spin
treatment, on 13 field points instead of 19: the diagonal from (+F_x, -F_x)
and the off-diagonal from the (+F_x+F_y, -F_x-F_y) pair alone,

    g(++) + g(--) - 2 g(0) = F^2 (g_xx + g_yy + 2 g_xy) + O(F^4),

and each field SCF starts from dm0 + F . P1 (P1 the static first-order
densities, three CPHF solves), an O(F^2) guess.  The second
difference divides SCF noise by F^2 = 6e-6, so the field SCFs are converged
to an orbital gradient of 1e-8 (GPU4PySCF's stop at the SCF's own
tolerance: 1e-2 relative error on H2O+ UHF at conv_tol 1e-10).  Harmonic analysis,
activities (Angstrom^4/amu) and depolarisation ratios as GPU4PySCF's.

    freq, act, depol = electrolites.raman_uks.eval_raman_intensity(mf)
"""
import numpy as np
import cupy as cp

from pyscf.data import nist
from pyscf.hessian import thermo


def _check(mf):
    from gpu4pyscf.scf import hf, uhf, rohf
    if isinstance(mf, rohf.ROHF) or not isinstance(mf, (hf.RHF, uhf.UHF)):
        raise NotImplementedError('Raman: RHF/RKS or UHF/UKS references')
    if getattr(mf, 'with_solvent', None) is not None:
        raise NotImplementedError('Raman with a solvent model: use '
                                  'gpu4pyscf.properties.raman (RHF/RKS)')


def polarizability_derivative(mf, dE=2.5e-3, conv_tol=1e-10,
                              conv_tol_grad=1e-8, max_cycle=100,
                              first_order_guess=True):
    """d alpha / dR, (natm, 3, 3, 3): (atom, displacement, alpha_xy)."""
    _check(mf)
    mol = mf.mol
    nao = mol.nao
    with mol.with_common_orig((0, 0, 0)):
        mu = cp.asarray(mol.intor('int1e_r'))
        dmu = -cp.asarray(mol.intor('int1e_irp')).reshape(3, 3, nao, nao)
    hcore = cp.asarray(mf.get_hcore())
    dm_ref = cp.asarray(mf.make_rdm1())
    # first-order densities: each field SCF starts from dm0 + F . P1, an
    # O(F^2) guess, instead of dm0
    from electrolites.polar_gpu import _response_U
    r = _response_U(mf, 1e-8, 50)
    pdm = r['dms'][0] if r['restricted'] else cp.stack(r['dms'], axis=1)
    if not first_order_guess:
        pdm = pdm * 0
    mf = mf.copy()
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.max_cycle = max(mf.max_cycle, max_cycle)
    aoslices = mol.aoslice_by_atom()[:, 2:]
    cycles = []

    def grad_at(E):
        E = cp.asarray(E, dtype=float)
        mf.get_hcore = lambda *a: hcore + cp.einsum('d,dij->ij', E, mu)
        guess = dm_ref + cp.einsum('d,d...->...', E, pdm)
        mf.kernel(dm0=guess)
        cycles.append(getattr(mf, 'cycles', -1))
        if not mf.converged:
            # DIIS creeps near a 1e-8 orbital gradient on some open shells
            # (H2O+ UHF: 4% per cycle); the result is still usable
            import warnings
            warnings.warn(f'Raman: field SCF not converged to the requested '
                          f'tolerance, F={E}')
        dm = cp.asarray(mf.make_rdm1())
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        g = cp.asarray(mf.nuc_grad_method().kernel())
        # field term E . d<mu>/dR_A; int1e_irp differentiates the ket, and
        # the bra derivative is its transpose (dm symmetric)
        t = 2 * cp.einsum('dxij,ij->dxj', dmu, dm)
        for a, (p0, p1) in enumerate(aoslices):
            g[a] += cp.einsum('dx,d->x', t[:, :, p0:p1].sum(axis=2), E)
        return g

    g0 = grad_at(np.zeros(3))
    diag = {}
    for i in range(3):
        e = np.zeros(3)
        e[i] = dE
        gp, gm = grad_at(e), grad_at(-e)
        diag[i] = (gp + gm - 2 * g0) / dE ** 2
    d2 = cp.empty((mol.natm, 3, 3, 3))
    for i in range(3):
        d2[:, :, i, i] = diag[i]
        for j in range(i + 1, 3):
            e = np.zeros(3)
            e[i] = e[j] = dE
            gpp, gmm = grad_at(e), grad_at(-e)
            dij = (gpp + gmm - 2 * g0) / (2 * dE ** 2) \
                - (diag[i] + diag[j]) / 2
            d2[:, :, i, j] = d2[:, :, j, i] = dij
    polarizability_derivative.cycles = cycles
    return cp.asnumpy(-d2)


def eval_raman_intensity(mf, hessian=None):
    """(frequencies cm^-1, Raman activities Angstrom^4/amu, depolarisation
    ratios), RHF/RKS or UHF/UKS."""
    _check(mf)
    mol = mf.mol
    if hessian is None:
        hessian = mf.Hessian().kernel()
    hessian = cp.asnumpy(cp.asarray(hessian))
    info = thermo.harmonic_analysis(mol, hessian)
    mode = info['norm_mode']
    da = np.einsum('AdEe,iAd->iEe', polarizability_derivative(mf), mode)
    a2 = (np.einsum('iee->i', da) / 3) ** 2
    b2 = .5 * ((da[:, 0, 0] - da[:, 1, 1]) ** 2
               + (da[:, 0, 0] - da[:, 2, 2]) ** 2
               + (da[:, 1, 1] - da[:, 2, 2]) ** 2
               + 6 * (da[:, 0, 1] ** 2 + da[:, 0, 2] ** 2 + da[:, 1, 2] ** 2))
    act = (45 * a2 + 7 * b2) * nist.BOHR ** 4
    depol = 3 * b2 / (45 * a2 + 4 * b2)
    return info['freq_wavenumber'], act, depol
