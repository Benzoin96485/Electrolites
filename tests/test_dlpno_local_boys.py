"""Sparse localization response checked against the independent PySCF operator."""
import numpy as np
import pytest
from pyscf import gto, lo, scf

from electrolites.dlpno.local_boys import (
    LocalBoysResidualError, SparseBoysHessian, solve_projected_boys, constraint_dipole_density,
)
from electrolites.dlpno.reference_cpu import rebind


def reference_action(dipoles):
    # The molecular integral is replaced, not the reference Hessian code.
    n = dipoles.shape[1]
    mol = gto.M(atom='He 0 0 0', basis='sto-3g', verbose=0)
    loc = lo.Boys(mol, np.eye(n))
    generate = rebind(type(loc).gen_g_hop, dipole_integral=lambda mol, c: dipoles)
    return generate(loc, np.eye(n))[1]


@pytest.mark.parametrize('stride', [1, 3])
def test_sparse_boys_projection_matches_pyscf_at_nonstationary_orbitals(stride):
    rng = np.random.default_rng(728)
    n = 9
    dipoles = rng.normal(size=(3, n, n))
    dipoles = (dipoles+dipoles.transpose(0, 2, 1))*.5
    pairs = list(zip(*np.tril_indices(n, -1)))[::stride]
    operator = SparseBoysHessian(np.diagonal(dipoles, axis1=1, axis2=2).T,
                                  pairs, lambda i, j: dipoles[:, i, j])
    # Test every column: absence of nonincident edges is a derived property.
    expected = np.column_stack([reference_action(dipoles)(operator.expand(v))[operator.full_indices]
                                for v in np.eye(len(pairs))])
    assert np.max(abs(operator.matrix.toarray()-expected)) < 2e-13


def test_sparse_boys_is_translation_and_rotation_covariant():
    rng = np.random.default_rng(44)
    n = 13
    centers = np.column_stack([np.arange(n)*2., rng.normal(size=(n, 2))])
    dipoles = rng.normal(size=(3, n, n))*.03
    dipoles = (dipoles+dipoles.transpose(0, 2, 1))*.5
    dipoles[:, np.arange(n), np.arange(n)] = centers.T
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    transformed = np.einsum('xy,yij->xij', rotation, dipoles)
    transformed[:, np.arange(n), np.arange(n)] += np.array([17., -5., 23.])[:, None]
    a, b = [SparseBoysHessian.from_dense(d, 6.1) for d in (dipoles, transformed)]
    assert np.array_equal(a.pairs, b.pairs)
    assert np.max(abs((a.matrix-b.matrix).data), initial=0.) < 1e-12


def test_projected_solve_checks_omitted_equations():
    rng = np.random.default_rng(85)
    n = 11
    dipoles = rng.normal(size=(3, n, n))*.01
    dipoles = (dipoles+dipoles.transpose(0, 2, 1))*.5
    dipoles[0, np.arange(n), np.arange(n)] = np.arange(n)*3.
    hop = reference_action(dipoles)
    hessian = SparseBoysHessian.from_dense(dipoles, 4.)
    exact = hessian.expand(rng.normal(size=len(hessian.pairs)))
    source = -hop(exact)
    solved, info = solve_projected_boys(dipoles, source, 4., hop)
    assert np.max(abs(exact-solved)) < 1e-11
    assert info['residual'] < 1e-10
    assert info['variables'] < info['full_variables']
    omitted = next(i for i in range(len(source)) if i not in hessian.full_indices)
    source[omitted] += 1.
    with pytest.raises(LocalBoysResidualError) as error:
        solve_projected_boys(dipoles, source, 4., hop)
    assert error.value.diagnostics['projected_residual'] < 1e-10
    assert error.value.diagnostics['residual'] > .99
    with pytest.raises(ValueError, match='Finite source'):
        solve_projected_boys(dipoles, source*np.nan, 4., hop)


def test_fixed_degree_hessian_has_linear_work_and_storage():
    counts = []
    for n in (32, 64, 128):
        centers = np.column_stack([np.arange(n)*2., np.zeros((n, 2))])
        pairs = [(i, j) for i in range(n) for j in range(max(0, i-2), i)]
        def dipole(i, j):
            return np.array([.01, .02, -.03])*np.exp(-abs(i-j))
        operator = SparseBoysHessian(centers, pairs, dipole)
        info = operator.diagnostics
        assert info['maximum_pair_neighbors'] == 4
        counts.append([info[k] for k in ('variables', 'hessian_nonzeros', 'dipole_queries',
                                         'hessian_storage_bytes')])
    assert np.max(np.array(counts[1:])/np.array(counts[:-1])) < 2.2


def test_constraint_density_transposes_the_localization_differential():
    rng = np.random.default_rng(53)
    n = 8
    d, tangent = rng.normal(size=(2, 3, n, n))
    d, tangent = [(v+v.transpose(0, 2, 1))*.5 for v in (d, tangent)]
    pairs = np.column_stack(np.tril_indices(n, -1))[::3]
    z = rng.normal(size=len(pairs))
    centers = np.diagonal(d, axis1=1, axis2=2).T
    density = constraint_dipole_density(centers, pairs, z, lambda i, j: d[:, i, j])
    def lagrangian(d):
        return sum(2*v*np.dot(d[:, i, i]-d[:, j, j], d[:, i, j]) for (i, j), v in zip(pairs, z))
    h = 1e-5
    fd = (lagrangian(d+h*tangent)-lagrangian(d-h*tangent))/(2*h)
    analytic = sum(b.multiply(t).sum() for b, t in zip(density, tangent))
    assert abs(analytic-fd) < 1e-9
    # Changing the dipole origin contributes a diagonal identity, whose
    # contraction with each localization constraint density is zero.
    assert max(abs(b.diagonal().sum()) for b in density) < 1e-14


@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
def test_full_radius_recovers_t1_gradient_and_small_radius_is_rejected(backend):
    from electrolites.dlpno.local import LocalDLPNO
    from electrolites.dlpno.orbital_adjoint import orbital_densities
    from benchmarks.dlpno_gradient_check import STRESS
    if backend == 'cupy':
        cp = pytest.importorskip('cupy')
        try:
            count = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError:
            count = 0
        if not count:
            pytest.skip('A CUDA device is required')
    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-10)
    base = LocalDLPNO(mf, triples='t1', verbose=0, **STRESS,
                      E_CONVERGENCE=1e-11, R_CONVERGENCE=1e-10, MAXITER=150, T_CUT_ITER=0.)
    base.kernel()
    complete = base.nuc_grad_method()
    expected = complete.kernel()
    projected = base.nuc_grad_method(boys_pair_radius=100., backend=backend)
    result = projected.kernel()
    assert np.max(abs(result-expected)) < 2e-10
    info = projected.diagnostics['orbitals']['boys']
    assert info['variables'] == info['full_variables']
    assert info['residual'] < 1e-10
    assert info['full_residual_oracle_used']
    with pytest.raises(LocalBoysResidualError):
        orbital_densities(base, complete.orbital_sources, boys_pair_radius=1e-6)
