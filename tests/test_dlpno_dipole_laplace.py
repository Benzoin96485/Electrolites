"""Far-pair separation, response accuracy and fixed numerical definition."""
import numpy as np
import pytest

from electrolites.dlpno.dipole_laplace import (
    LaplaceQuadrature, polarizability, polarizability_pullback,
    polarizability_ad, quadrature_pair, exact_pair, dipole_kernel,
)


@pytest.mark.parametrize('interval', [(0.02, 70.), (1e-3, 1e3)])
def test_positive_quadrature_bounds_energy_and_gap_response(interval):
    rule = LaplaceQuadrature.build(interval, tolerance=1e-12)
    x = np.geomspace(2*interval[0], 2*interval[1], 2001)
    exponential = np.exp(-x[:, None]*rule.nodes)
    value = exponential@rule.weights
    response = exponential@(rule.weights*rule.nodes)
    assert np.max(abs(x*value-1)) < 1e-12
    assert np.max(abs(x*x*response-1)) < 1e-12
    assert rule.energy_relative_bound <= rule.tolerance
    assert rule.gap_response_relative_bound <= rule.tolerance
    assert np.all(rule.weights > 0)
    with pytest.raises(ValueError, match='outside the fixed'):
        rule.check_gaps([interval[0]/2])
    assert not rule.nodes.flags.writeable


def data():
    rng = np.random.default_rng(7108)
    d, e = rng.normal(size=(3, 7)), rng.uniform(.2, 4., 7)
    q, f = rng.normal(size=(3, 5)), rng.uniform(.1, 6., 5)
    r = np.array([8.1, -2.7, 1.3])
    return d, e, q, f, r


def test_separated_energy_and_all_local_sources_match_virtual_sum():
    rule = LaplaceQuadrature.build((.05, 10.))
    d, e, q, f, r = data()
    left, right = polarizability(d, e, rule), polarizability(q, f, rule)
    energy, left_bar, right_bar, gradient = quadrature_pair(left, right, r)
    db, eb = polarizability_pullback(d, e, rule, left_bar)
    qb, fb = polarizability_pullback(q, f, rule, right_bar)
    expected = exact_pair(d, e, q, f, r)
    for actual, exact in zip((energy, db, eb, qb, fb, gradient), expected):
        assert np.max(abs(actual-exact)) < 1e-13
    # Independent finite difference of the virtual-index parent formula.
    step = 2e-5
    for axis in range(3):
        direction = np.eye(3)[axis]*step
        numeric = (exact_pair(d, e, q, f, r+direction)[0]
                   -exact_pair(d, e, q, f, r-direction)[0])/(2*step)
        assert abs(numeric-gradient[axis]) < 1e-11


def test_local_polarizability_vjp_and_rotational_covariance():
    from electrolites.dlpno import reverse as ad
    rule = LaplaceQuadrature.build((.05, 10.))
    d, e, q, f, r = data()
    rng = np.random.default_rng(14)
    source = rng.normal(size=(len(rule.nodes), 6))
    vd, ve = ad.Variable(d), ad.Variable(e)
    objective = np.sum(source*polarizability_ad(vd, ve, rule))
    db, eb = ad.gradients(objective, [vd, ve])
    dd, de = rng.normal(size=d.shape), rng.normal(size=e.shape)
    step = 1e-6
    numeric = np.sum(source*(polarizability(d+step*dd, e+step*de, rule)
                             -polarizability(d-step*dd, e-step*de, rule)))/(2*step)
    assert abs(numeric-np.sum(db*dd)-np.sum(eb*de)) < 2e-8
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    original = exact_pair(d, e, q, f, r)
    moved = exact_pair(rotation@d, e, rotation@q, f, rotation@r)
    assert abs(original[0]-moved[0]) < 1e-14
    assert np.max(abs(rotation@original[-1]-moved[-1])) < 1e-13
    kernel, _ = dipole_kernel(r)
    assert np.max(abs(kernel-kernel.T)) < 1e-16
