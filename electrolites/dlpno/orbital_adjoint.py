"""Coordinate-independent Boys and RHF adjoints and force contraction.

The orbital response is transposed once, without a dense orbital-pair
Hessian or separate solves for each nuclear direction. The initial
three-center density adapter is dense; its contraction is streamed by
auxiliary atom so the derivative tensor never carries a nuclear axis.
"""
import numpy as np
from pyscf import df, gto, lo
from scipy.sparse.linalg import LinearOperator, gmres, minres

from . import reverse as ad
from .reference_cpu import rebind


def _exact_density_factorization(mf_grad, mol, density):
    """D = U V.T without discarding small, signed response components.

    PySCF's occupied-orbital DF gradient factorization drops singular or
    occupation values below 1e-8.  A correlation response density need not
    be idempotent; dropping its small eigenvalues breaks the bilinear
    Fock derivative and rigid-rotation invariance.
    """
    matrices = np.asarray(density).reshape(-1, mol.nao, mol.nao)
    left = [np.asfortranarray(np.eye(mol.nao)) for _ in matrices]
    right = [np.asfortranarray(matrix.T) for matrix in matrices]
    return left, right


def _df_response_potential(mf_grad, density):
    from pyscf import lib
    from pyscf.df.grad import rhf as df_gradient
    get_jk = rebind(df_gradient.get_jk, _decompose_rdm1=_exact_density_factorization,
                    _decompose_rdm1_svd=_exact_density_factorization)
    j, k = get_jk(mf_grad, mf_grad.mol, density, hermi=1)
    potential = j-.5*k
    if mf_grad.auxbasis_response:
        potential = lib.tag_array(potential, aux=(j.aux-.5*k.aux).sum((0, 1)))
    return potential


def _linear_fock(mf, density, fock=None):
    potential = (lambda dm: np.asarray(mf.get_veff(mf.mol, dm))) if fock is None else fock
    value = potential(density.v)

    def reverse(bar):
        # The RHF Coulomb-exchange map is self-adjoint on symmetric AO
        # densities, for both exact and density-fitted reference J/K.
        return (potential((bar+bar.T)*.5),)

    return ad.operation(value, [density], reverse)


def _boys_pullback(mol, cl, r, rotation, ncore, nocc, bar_cl, pair_radius=None):
    nao, n = cl.shape
    transported = ad.Variable(np.zeros((nao, nocc)))
    dr = ad.Variable(np.zeros_like(r))
    dc = transported[:, ncore:]@rotation
    objective = np.sum(dc*bar_cl)
    info = dict(solves=0, residual=0., stationary_residual=0.)
    if n > 1:
        loc = lo.Boys(mol, cl)
        if pair_radius is None:
            g, hop, _ = loc.gen_g_hop(np.eye(n))
        else:
            dipole_reference = lo.boys.dipole_integral(mol, cl)
            generate = rebind(type(loc).gen_g_hop, dipole_integral=lambda mol, c: dipole_reference)
            g, hop, _ = generate(loc, np.eye(n))
        info['stationary_residual'] = float(np.max(abs(g)))
        if info['stationary_residual'] > 1e-8:
            raise RuntimeError('The Boys orbitals must be stationary for their adjoint')
        indices = np.tril_indices(n, -1)
        orbital_source = cl.T@bar_cl
        source = (orbital_source-orbital_source.T)[indices]
        if pair_radius is None:
            operator = LinearOperator((len(g), len(g)), matvec=hop, rmatvec=hop, dtype=float)
            # Starting MINRES at zero preserves parallel transport in the
            # null space of a symmetric localization Hessian. No dense
            # Hessian or inverse is built. An incompatible source fails.
            z, status = minres(operator, -source, rtol=1e-12, maxiter=1000)
            error = float(np.max(abs(hop(z)+source), initial=0.))
            if status or error > 1e-8:
                raise RuntimeError(f'Boys adjoint failed: status={status}, residual={error:.3e}')
            info.update(solves=1, residual=error, variables=len(g))
        else:
            from .local_boys import solve_projected_boys, constraint_dipole_density
            z, local_info = solve_projected_boys(dipole_reference, source, pair_radius, hop)
            info.update(local_info)
            # Exact analytic pullback for these multipliers. Sparse
            # occupied dipole densities avoid the dense localization
            # differential graph. The AO backtransforms below remain
            # dense and are not asserted to have linear scaling.
            active = np.flatnonzero(z)
            pairs = np.column_stack([indices[0][active], indices[1][active]])
            centers = np.diagonal(dipole_reference, axis1=1, axis2=2).T
            densities = constraint_dipole_density(centers, pairs, z[active],
                                                  lambda i, j: dipole_reference[:, i, j])
            left = [(density@cl.T).T for density in densities]
            coefficient_bar = bar_cl+2*sum(moment@block for moment, block in zip(r, left))
            bar_co = np.zeros((nao, nocc))
            bar_co[:, ncore:] = coefficient_bar@rotation.T
            bar_r = np.array([block@cl.T for block in left])
            info.update(constraint_density_nonzeros=sum(d.nnz for d in densities),
                        constraint_pullback='sparse_occupied_density_dense_ao_backtransform')
            return bar_co, bar_r, info
        dip = np.einsum('mi,xmn,nj->xij', cl, r, cl, optimize=True)
        ddip = (ad.einsum('mi,xmn,nj->xij', dc, r, cl, optimize=True)
                 + ad.einsum('mi,xmn,nj->xij', cl, dr, cl, optimize=True)
                 + ad.einsum('mi,xmn,nj->xij', cl, r, dc, optimize=True))
        d_diag = ad.concatenate([np.diag(ddip[x])[None] for x in range(3)])
        diag = np.diagonal(dip, axis1=1, axis2=2)
        dg0 = (ad.einsum('xi,xip->pi', d_diag, dip)
               + ad.einsum('xi,xip->pi', diag, ddip))
        rhs = -2*(dg0-dg0.T)[indices]
        objective += np.sum(z*rhs)
    bar_co, bar_r = ad.gradients(objective, [transported, dr])
    return bar_co, bar_r, info


def orbital_densities(base, bars, *, boys_pair_radius=None, fock=None):
    """Transpose all reference-orbital response into explicit S/F/r bars."""
    mf = base.mf.to_cpu() if hasattr(base.mf, 'to_cpu') else base.mf
    mol, c = mf.mol, np.asarray(mf.mo_coeff)
    occupied = np.asarray(mf.mo_occ)>0
    co = c[:, occupied]
    nocc, nao = co.shape[1], c.shape[0]
    nc, cl = base.orb.ncore, base.orb.C_lmo
    overlap = np.asarray(mf.get_ovlp())
    rotation = c[:, nc:nocc].T@overlap@cl
    with mol.with_common_orig((0., 0., 0.)):
        r = mol.intor('int1e_r', comp=3)
    local_co, local_r, local_info = _boys_pullback(mol, cl, r, rotation, nc, nocc, bars['cl'], boys_pair_radius)
    bar_co = bars['co']+local_co

    s1, f1 = ad.Variable(np.zeros((nao, nao))), ad.Variable(np.zeros((nao, nao)))
    uv = ad.Variable(np.zeros((c.shape[1]-nocc, nocc)))
    eps = np.asarray(mf.mo_energy)
    den = eps[nocc:, None]-eps[:nocc]
    s_mo = c.T@s1@co
    u = ad.concatenate([-.5*s_mo[:nocc], uv])
    dc_parallel = c@u
    dm = 2*(dc_parallel@co.T+co@dc_parallel.T)
    total_f = f1+_linear_fock(mf, dm, fock)
    f_mo = c.T@total_f@c
    s_full = c.T@s1@c
    eo = eps[:nocc]
    gap = eo[None, :]-eo[:, None]
    mask = abs(gap)>1e-10
    uoo = ((f_mo[:nocc, :nocc]-s_full[:nocc, :nocc]*eo)/np.where(mask, gap, 1.))*mask
    uoo += -.5*s_full[:nocc, :nocc]*(~mask)
    dc = c@ad.concatenate([uoo, uv])
    residual = den*uv+f_mo[nocc:, :nocc]-s_mo[nocc:]*eo
    objective = np.sum(bar_co*dc)+np.sum(bars['f']*total_f)
    (source,) = ad.gradients(objective, [uv])
    linearization = ad.Linearization([residual], [uv])

    def action(vector):
        return linearization([vector.reshape(den.shape)])[0].ravel()

    operator = LinearOperator((uv.size, uv.size), matvec=action, dtype=float)
    pre = LinearOperator(operator.shape, matvec=lambda x: x/den.ravel(), dtype=float)
    iterations = []
    z, status = gmres(operator, -source.ravel(), M=pre, rtol=1e-11, atol=1e-13,
                      restart=min(uv.size, 50), maxiter=100, callback=iterations.append,
                      callback_type='pr_norm')
    error = float(np.max(abs(action(z)+source.ravel()), initial=0.))
    if status or error > 1e-9:
        raise RuntimeError(f'CPHF adjoint failed: status={status}, residual={error:.3e}')
    lagrangian = objective+np.sum(z.reshape(den.shape)*residual)
    bar_s, bar_f = ad.gradients(lagrangian, [s1, f1])
    result = dict(s=bars['s']+bar_s, f=bar_f, r=bars['r']+local_r)
    return result, dict(boys=local_info, cphf=dict(solves=1, residual=error,
                                                 iterations=len(iterations), variables=uv.size))


def _jk_quadratic_force(mf_grad, density):
    """Derivative of 1/2 Tr[D G(D)], including a DF auxiliary response."""
    mol = mf_grad.mol
    potential = (_df_response_potential(mf_grad, density)
                 if getattr(mf_grad.base, 'with_df', None) else mf_grad.get_veff(mol, density))
    result = np.zeros((mol.natm, 3))
    for atom, (_, _, first, last) in enumerate(mol.aoslice_by_atom()):
        result[atom] = 2*np.einsum('xij,ij->x', potential[:, first:last], density[first:last])
    if getattr(mf_grad, 'auxbasis_response', False):
        result += potential.aux
    return result


def contract_orbital_densities(base, densities, *, jk_force=None):
    mf = base.mf.to_cpu() if hasattr(base.mf, 'to_cpu') else base.mf
    mol, grad = mf.mol, mf.nuc_grad_method()
    result = np.zeros((mol.natm, 3))
    fbar = (densities['f']+densities['f'].T)*.5
    core = grad.hcore_generator(mol)
    ip_s = mol.intor('int1e_ipovlp', comp=3)
    with mol.with_common_orig((0., 0., 0.)):
        rp = mol.intor('int1e_irp', comp=9).reshape(3, 3, mol.nao, mol.nao)
    sbar = densities['s']+densities['s'].T
    rbar = densities['r']+densities['r'].transpose(0, 2, 1)
    for atom, (_, _, first, last) in enumerate(mol.aoslice_by_atom()):
        result[atom] += np.einsum('xij,ij->x', core(atom), fbar)
        result[atom] -= np.einsum('xij,ij->x', ip_s[:, first:last], sbar[first:last])
        result[atom] -= np.einsum('rxij,rij->x', rp[:, :, :, first:last], rbar[:, :, first:last])
    dm = np.asarray(mf.make_rdm1())
    # Polarization of a quadratic JK functional is valid also for the
    # nonlinear-in-D auxiliary response stored by PySCF's DF gradient.
    quadratic = (lambda density: _jk_quadratic_force(grad, density)) if jk_force is None else jk_force
    result += quadratic(dm+fbar)-quadratic(dm)-quadratic(fbar)
    return result


def contract_dense_df_densities(base, bars):
    """Dense oracle densities, streamed derivative integrals by aux atom.

    This eliminates the nuclear-coordinate tensor/loop, but the dense
    density and unrestricted AO supports still prevent linear scaling.
    """
    mol, auxmol = base.mol, base.auxmol
    result = np.zeros((mol.natm, 3))
    bf_atom = base.bf_atom
    peak = 0
    for atom, (_, _, first, last) in enumerate(auxmol.aoslice_by_atom()):
        sub = auxmol.copy(deep=False)
        sub._bas = auxmol._bas[auxmol._bas[:, gto.ATOM_OF] == atom]
        gamma = bars['b'][:, :, first:last]
        ip1 = df.incore.aux_e2(mol, sub, intor='int3c2e_ip1', aosym='s1', comp=3)
        centers = -np.einsum('mnQ,xmnQ->xm', gamma+gamma.transpose(1, 0, 2), ip1)
        for xyz in range(3):
            result[:, xyz] += np.bincount(bf_atom, centers[xyz], minlength=mol.natm)
        peak = max(peak, ip1.nbytes)
        del ip1
        ip2 = df.incore.aux_e2(mol, sub, intor='int3c2e_ip2', aosym='s1', comp=3)
        result[atom] -= np.einsum('mnQ,xmnQ->x', gamma, ip2)
        peak = max(peak, ip2.nbytes)
    metric_ip = auxmol.intor('int2c2e_ip1', comp=3)
    z = bars['m']+bars['m'].T
    centers = -np.einsum('PQ,xPQ->xP', z, metric_ip)
    for xyz in range(3):
        result[:, xyz] += np.bincount(base.ribf_atom, centers[xyz], minlength=mol.natm)
    return result, dict(derivative_peak_block_bytes=peak, derivative_aux_blocks=mol.natm,
                        density_bytes=bars['b'].nbytes+bars['m'].nbytes)
