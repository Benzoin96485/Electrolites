"""No false negatives in the original pair screen, including edge cases."""
import numpy as np
import pytest

from electrolites.dlpno.dipole_prescreen import (
    excitation_strength, bound_pairs, select_dipole_pairs,
)


@pytest.mark.parametrize('threshold', [1e-3, 1e-6, 1e-9])
def test_sparse_selector_matches_dense_ragged_virtual_oracle(threshold):
    rng = np.random.default_rng(61)
    centers = rng.uniform(-40., 40., (70, 3))
    centers[1] = centers[0]  # Preserve the parent's mandatory close pair.
    dipoles = [rng.normal(size=(3, int(rng.integers(2, 7)))) for _ in centers]
    gaps = [np.exp(rng.uniform(-3, 3, d.shape[1])) for d in dipoles]
    retained, info = select_dipole_pairs(centers, dipoles, gaps, threshold)
    strengths = [excitation_strength(d, g) for d, g in zip(dipoles, gaps)]
    reference = {}
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            radius = np.linalg.norm(centers[i]-centers[j])
            if radius < 1e-12:
                reference[i, j] = -np.inf
                continue
            dot = dipoles[i].T@dipoles[j]
            value = -16/radius**6*np.sum(dot**2/(gaps[i][:, None]+gaps[j][None, :]))
            upper = 8*strengths[i]*strengths[j]/radius**6
            assert abs(value) <= upper*(1+1e-14)
            if abs(value) > threshold:
                reference[i, j] = value
    assert retained.keys() == reference.keys()
    for key in retained:
        assert retained[key] == reference[key]
    assert info['virtual_pair_candidates'] <= len(centers)*(len(centers)-1)//2


def test_zero_strength_and_threshold_boundary_are_conservative():
    centers = np.array([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
    zeros = [np.zeros((3, 1)) for _ in centers]
    retained, _ = select_dipole_pairs(centers, zeros, [np.ones(1)]*3, 1e-6)
    assert retained == {(0, 1): -np.inf}
    # Parallel one-dimensional dipoles with equal gaps saturate the bound.
    d, g, threshold = np.array([[1.], [0.], [0.]]), np.array([2.]), 1e-6
    strength = excitation_strength(d, g)
    radius = (8*strength**2/threshold)**(1/6)
    for scale in [1-1e-12, 1., 1+1e-12]:
        candidates, _ = bound_pairs([[0., 0., 0.], [scale*radius, 0., 0.]], [strength]*2, threshold)
        exact = 4/(scale*radius)**6
        if exact > threshold:
            assert (0, 1) in candidates
    with pytest.raises(ValueError, match='positive finite threshold'):
        bound_pairs(centers, np.ones(3), 0.)
    with pytest.raises(ValueError, match='Positive finite excitation gaps'):
        excitation_strength(d, [-1.])


def test_fixed_density_bound_discovery_has_linear_operation_counts():
    records = []
    for n in (256, 512, 1024):
        centers = np.zeros((n, 3)); centers[:, 0] = 2*np.arange(n)+.1
        _, info = bound_pairs(centers, np.full(n, .1), 1e-6)
        records.append(info)
    for before, after in zip(records, records[1:]):
        assert after['distance_tests'] < 2.1*before['distance_tests']
        assert after['virtual_pair_candidates'] < 2.1*before['virtual_pair_candidates']
        assert after['largest_cell'] == before['largest_cell']
