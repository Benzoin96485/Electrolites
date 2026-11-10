"""Independent real nonsymmetric solves for device-resident adjoint GMRES."""
import json

import numpy as np
import pytest

from electrolites.dlpno.krylov_gpu import gmres


@pytest.fixture
def cp():
    cupy = pytest.importorskip('cupy')
    if not cupy.cuda.runtime.getDeviceCount():
        pytest.skip('CUDA device required')
    return cupy


@pytest.mark.parametrize('restart', [2, 7])
@pytest.mark.parametrize('case', ['identity', 'diagonal', 'nonsymmetric', 'nonnormal', 'mixed_scale'])
def test_gpu_gmres_matches_independent_dense_solve(cp, case, restart):
    n = 19
    rng = np.random.default_rng(9641)
    if case == 'identity':
        a = np.eye(n)
    elif case == 'diagonal':
        a = np.diag(np.linspace(-3., -1., n))
    elif case == 'nonsymmetric':
        a = np.diag(np.linspace(2., 4., n)) + .03*rng.normal(size=(n, n))
    elif case == 'nonnormal':
        a = np.eye(n) + .08*np.triu(np.ones((n, n)), 1)
    else:
        a = np.logspace(-2, 2, n)[:, None]*(np.eye(n)+.01*rng.normal(size=(n, n)))
    b = rng.normal(size=n)
    expected = np.linalg.solve(a, b)
    matrix, rhs, diagonal = cp.asarray(a), cp.asarray(b), cp.asarray(np.diag(a).copy())
    input_ids = []

    def action(vector):
        assert isinstance(vector, cp.ndarray) and vector.dtype == cp.float64
        input_ids.append(vector.device.id)
        return matrix@vector

    actual, info = gmres(action, rhs, diagonal, restart=restart, maxiter=100,
                        rtol=1e-12, atol=1e-13)
    result = cp.asnumpy(actual)
    error = float(np.max(abs(result-expected)))
    residual = float(np.linalg.norm(a@result-b))
    assert info['status'] == 0 and info['termination'] == 'true_residual_converged'
    assert np.allclose(result, expected, rtol=2e-10, atol=2e-10)
    assert residual <= 2*max(1e-13, 1e-12*np.linalg.norm(b))
    assert all(device == actual.device.id for device in input_ids)
    assert len(input_ids) == info['matvec_calls']
    assert np.array_equal(cp.asnumpy(rhs), b)
    assert np.array_equal(cp.asnumpy(diagonal), np.diag(a))
    assert not info['rhs_copied_to_host']
    print('GPU_KRYLOV_REPORT '+json.dumps(dict(case=case, restart=restart,
          maximum_solution_error=error, independent_residual_norm=residual, solver=info)))


@pytest.mark.parametrize('size,scale', [(0, 0.), (13, 0.), (13, 1e-14)])
def test_zero_and_tiny_rhs_need_no_matrix_actions(cp, size, scale):
    def forbidden(vector):
        raise AssertionError('The zero solution already meets the absolute tolerance')
    rhs = cp.full(size, scale, dtype=cp.float64)
    result, info = gmres(forbidden, rhs, cp.ones_like(rhs), atol=1e-11)
    assert info['status'] == 0 and info['iterations'] == info['matvec_calls'] == 0
    assert np.array_equal(cp.asnumpy(result), np.zeros(size))


def test_gpu_gmres_reports_exhausted_iterations(cp):
    rng = np.random.default_rng(188)
    a = np.eye(17) + .8*rng.normal(size=(17, 17))
    b = rng.normal(size=17)
    matrix, rhs = cp.asarray(a), cp.asarray(b)
    result, info = gmres(lambda x: matrix@x, rhs, cp.ones_like(rhs),
                        restart=2, maxiter=1, rtol=1e-13, atol=1e-14)
    assert info['status'] > 0 and info['termination'] == 'maximum_restart_cycles'
    assert info['iterations'] == 2 and info['cycles'] == 1
    assert np.linalg.norm(a@cp.asnumpy(result)-b) > 1e-8


def test_singular_breakdown_is_not_false_convergence(cp):
    rhs = cp.ones(9, dtype=cp.float64)
    result, info = gmres(lambda x: cp.zeros_like(x), rhs, cp.ones_like(rhs), restart=7)
    assert info['status'] > 0
    assert info['termination'] == 'arnoldi_breakdown_without_convergence'
    assert info['iterations'] == 1 and info['residual_maximum'] == 1.
    assert np.array_equal(cp.asnumpy(result), np.zeros(9))


def test_single_unknown_happy_breakdown(cp):
    rhs = cp.asarray([-8.])
    result, info = gmres(lambda x: 4*x, rhs, cp.asarray([4.]), restart=40)
    assert info['status'] == 0 and info['iterations'] == 1
    assert np.array_equal(cp.asnumpy(result), [-2.])


@pytest.mark.parametrize('scale', [1., 1e3])
def test_maximum_residual_bound_refines_norm_converged_solution(cp, scale):
    rng = np.random.default_rng(1882)
    a = 3*np.eye(19) + .09*rng.normal(size=(19, 19))
    b = scale*rng.normal(size=19)
    expected = np.linalg.solve(a, b)
    matrix, rhs, diagonal = cp.asarray(a), cp.asarray(b), cp.asarray(np.diag(a).copy())
    relaxed, first = gmres(lambda x: matrix@x, rhs, diagonal,
                          restart=2, rtol=1e-2, atol=1e-13)
    assert first['status'] == 0 and first['residual_maximum'] > 1e-8
    actual, info = gmres(lambda x: matrix@x, rhs, diagonal,
                        restart=2, rtol=1e-2, atol=1e-13, max_residual=1e-8)
    result = cp.asnumpy(actual)
    residual = float(np.max(abs(a@result-b)))
    error = float(np.max(abs(result-expected)))
    assert info['status'] == 0 and info['residual_maximum'] <= 1e-8
    assert residual < 1.01e-8 and error < 1e-8
    assert info['iterations'] > first['iterations']
    assert np.array_equal(cp.asnumpy(rhs), b)
    print('GPU_KRYLOV_MAXIMUM_REPORT '+json.dumps(dict(scale=scale,
          norm_only_residual_maximum=first['residual_maximum'],
          independent_residual_maximum=residual, maximum_solution_error=error, solver=info)))


def test_maximum_residual_bound_prevents_early_zero_solution(cp):
    rhs = cp.full(7, 1e-3, dtype=cp.float64)
    actual, info = gmres(lambda x: x.copy(), rhs, cp.ones_like(rhs),
                        rtol=0., atol=1e-2, max_residual=1e-8)
    assert info['status'] == 0 and info['iterations'] == 1
    assert np.max(abs(cp.asnumpy(actual)-1e-3)) < 1e-12


def test_maximum_residual_bound_is_reported_when_cycles_exhausted(cp):
    rng = np.random.default_rng(541)
    a = 3*np.eye(19) + .09*rng.normal(size=(19, 19))
    matrix, rhs = cp.asarray(a), cp.asarray(rng.normal(size=19))
    actual, info = gmres(lambda x: matrix@x, rhs, cp.asarray(np.diag(a).copy()),
                        restart=1, maxiter=1, rtol=1., atol=1e-13, max_residual=1e-14)
    assert info['residual_norm'] <= info['tolerance']
    assert info['residual_maximum'] > 1e-14 and info['status'] > 0
    assert info['termination'] == 'maximum_restart_cycles'


@pytest.mark.parametrize('bound', [-1., np.nan, np.inf])
def test_invalid_maximum_residual_tolerance(cp, bound):
    rhs = cp.ones(3, dtype=cp.float64)
    with pytest.raises(ValueError, match='Maximum residual tolerance'):
        gmres(lambda x: x.copy(), rhs, cp.ones_like(rhs), max_residual=bound)
