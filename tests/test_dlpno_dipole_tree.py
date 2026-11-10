"""Independent pair oracle, fixed-branch FD, covariance and pair accounting."""
import numpy as np
import pytest

from electrolites.dlpno.dipole_laplace import SYMMETRIC_BASIS, dipole_kernel, quadrature_pair
from electrolites.dlpno.dipole_tree import (
    MutualTree, _layout, kernel_taylor, kernel_remainder_bound, tree_energy_ad,
)


def direct(centers, sources, exclusions=()):
    skip = {tuple(sorted(pair)) for pair in exclusions}
    energy, pb, xb = 0., np.zeros_like(sources), np.zeros_like(centers)
    for i in range(len(centers)):
        for j in range(i):
            if (j, i) in skip:
                continue
            e, bi, bj, g = quadrature_pair(sources[i], sources[j], centers[i]-centers[j])
            energy += e; pb[i] += bi; pb[j] += bj
            xb[i] += g; xb[j] -= g
    return energy, pb, xb


def sample():
    rng = np.random.default_rng(818)
    sites = np.array([[0, 0, 0], [0, 0, 12.], [12., 0, 0], [12., 0, 12.]])
    offsets = np.array([[.4, .6, .8], [1.8, .7, .6], [.6, 1.9, .7], [.5, .6, 1.9]])
    centers = np.repeat(sites, 4, axis=0)+np.tile(offsets, (4, 1))+rng.uniform(-.02, .02, (16, 3))
    tensors = .2*rng.normal(size=(16, 3, 3, 2))
    sources = np.einsum('nqia,nqja,bij->nqb', tensors, tensors, SYMMETRIC_BASIS)
    return centers, sources


@pytest.mark.parametrize('order', [0, 3, 6, 9])
def test_kernel_total_degree_matches_tensor_and_spatial_derivative(order):
    r = np.array([9.1, -3.3, 2.4])
    coefficients = kernel_taylor(r, order+1)
    kernel, derivative = dipole_kernel(r)
    assert np.max(abs(coefficients[0]-kernel)) < 1e-19
    layout = _layout(order+1)
    for axis in range(3):
        alpha = tuple(int(k == axis) for k in range(3))
        assert np.max(abs(coefficients[layout['lookup'][alpha]]-derivative[axis])) < 1e-19
    displacement = np.array([.4, -.2, .1])
    powers = _layout(order)['powers']
    approximation = np.einsum('m,mab->ab', np.prod(displacement**powers, axis=1),
                              coefficients[:len(powers)])
    error = np.linalg.norm(approximation-dipole_kernel(r+displacement)[0], ord=2)
    bound = kernel_remainder_bound(order, np.linalg.norm(r), np.linalg.norm(displacement))
    assert error < bound+1e-20


def test_complete_pair_accounting_sources_and_centroid_response():
    x, p = sample()
    excluded = [(0, 1), (7, 6)]
    tree = MutualTree(x, order=6, theta=.22, leaf_width=3., exclusions=excluded)
    energy, pb, xb, info = tree.evaluate(p, channel_block=2)
    exact = direct(x, p, excluded)
    assert info['far_cell_pairs'] > 0
    assert info['represented_pairs'] == len(x)*(len(x)-1)//2-len(excluded)
    assert abs(energy-exact[0]) <= info['analytic_energy_absolute_bound']+1e-8
    assert abs(energy-exact[0]) < 2e-7
    assert np.max(abs(pb-exact[1])) < 1e-7
    assert np.max(abs(xb-exact[2])) < 1e-7
    assert np.max(abs(xb.sum(axis=0))) < 1e-12
    rng = np.random.default_rng(12)
    dx, dp = rng.normal(size=x.shape), rng.normal(size=p.shape)
    # Subtract the direct near contribution to resolve the small far derivative
    # accurately, independently of the much larger near-pair energies.
    far_tree = MutualTree(x, order=6, theta=.22, leaf_width=3.,
                          exclusions=[(i, j) for i in range(16) for j in range(i)
                                      if i//4 == j//4])
    energy, pb, xb, info = far_tree.evaluate(p)
    assert info['far_cell_pairs'] > 0
    step = 2e-5
    plus = far_tree.evaluate(p+step*dp, centers=x+step*dx)
    minus = far_tree.evaluate(p-step*dp, centers=x-step*dx)
    numeric = (plus[0]-minus[0])/(2*step)
    analytic = np.sum(pb*dp)+np.sum(xb*dx)
    assert abs(numeric-analytic) < 1e-10
    assert np.max(abs(xb.sum(axis=0))) < 1e-16


def test_fixed_tree_rigid_covariance_and_scalar_custom_vjp():
    from electrolites.dlpno import reverse as ad
    x, p = sample()
    excluded = [(i, j) for i in range(16) for j in range(i) if i//4 == j//4]
    tree = MutualTree(x, order=5, theta=.3, leaf_width=3., exclusions=excluded)
    result = tree.evaluate(p)
    rng = np.random.default_rng(5)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    transform = np.einsum('aij,ik,bkl,jl->ab', SYMMETRIC_BASIS, rotation,
                          SYMMETRIC_BASIS, rotation)
    moved = tree.evaluate(p@transform.T, centers=x@rotation.T+2.)
    assert abs(result[0]-moved[0]) < 1e-16
    assert np.max(abs(result[1]@transform.T-moved[1])) < 1e-16
    assert np.max(abs(result[2]@rotation.T-moved[2])) < 1e-16
    vx, vp = ad.Variable(x), ad.Variable(p)
    objective = tree_energy_ad(vx, vp, order=5, theta=.3, leaf_width=3., exclusions=excluded)
    xb, pb = ad.gradients(objective, [vx, vp])
    assert np.max(abs(xb-result[2])) < 1e-16
    assert np.max(abs(pb-result[1])) < 1e-16


def test_long_exclusion_cannot_hide_in_far_cells():
    x, p = sample()
    excluded = [(0, 15)]
    tree = MutualTree(x, order=3, theta=.5, leaf_width=3., exclusions=excluded)
    energy, pb, xb, info = tree.evaluate(p)
    exact = direct(x, p, excluded)
    assert np.max(abs(pb-exact[1])) < 5e-7
    assert abs(energy-exact[0]) < info['analytic_energy_absolute_bound']+2e-7
    assert info['represented_pairs'] == 119
    with pytest.raises(ValueError, match='indices out of range'):
        MutualTree(x, exclusions=[(0, 16)])


def test_hierarchy_operation_counts_for_bounded_density_chain():
    rows = []
    for n in (256, 512, 1024):
        x = np.zeros((n, 3)); x[:, 0] = 2.*np.arange(n)+.3
        tree = MutualTree(x, order=6, theta=.3, leaf_width=4.,
                          exclusions=[(i, i+1) for i in range(n-1)])
        rows.append((len(tree.cells), tree.visits, len(tree.far), len(tree.near)))
        assert tree.represented_pairs == n*(n-1)//2-(n-1)
    # Structural count, not a runtime or full DLPNO scaling certification.
    assert np.all(np.array(rows[2]) < 2.2*np.array(rows[1]))
    assert np.all(np.array(rows[1]) < 2.2*np.array(rows[0]))


def test_channel_geometry_cache_is_bounded_exact_and_invalidated_on_displacement():
    centers, sources = sample()
    sources = np.tile(sources, (1, 5, 1))
    tree = MutualTree(centers, order=5, theta=.3, leaf_width=3.)
    cached = tree.evaluate(sources, channel_block=3)
    uncached = tree.evaluate(sources, channel_block=3, geometry_cache_bytes=0)
    assert cached[0] == uncached[0]
    for first, second in zip(cached[1:3], uncached[1:3]):
        assert np.array_equal(first, second)
    assert cached[-1]['analytic_energy_absolute_bound'] == uncached[-1]['analytic_energy_absolute_bound']
    nfar = len(tree.far)
    assert nfar > 0
    info = cached[-1]['geometry_cache']
    assert info['kernel_builds'] == nfar
    assert info['kernel_cache_hits'] == nfar*4
    assert info['numeric_payload_bytes'] <= info['budget_bytes']
    assert uncached[-1]['geometry_cache']['kernel_builds'] == nfar*5
    assert uncached[-1]['geometry_cache']['numeric_payload_bytes'] == 0
    moved = centers.copy(); moved[0, 0] += .001
    a = tree.evaluate(sources, centers=moved)
    b = tree.evaluate(sources, centers=moved, geometry_cache_bytes=0)
    assert a[0] == b[0]
    assert a[0] != cached[0]
    for first, second in zip(a[1:3], b[1:3]):
        assert np.array_equal(first, second)
    # A changed batch size may change only floating-point summation order.
    one_batch = tree.evaluate(sources, channel_block=sources.shape[1])
    assert abs(one_batch[0]-cached[0]) < 1e-12
    assert max(np.max(abs(a-b)) for a, b in zip(one_batch[1:3], cached[1:3])) < 1e-12
    forward = tree.evaluate(sources, derivatives=False, channel_block=3)
    assert forward[0] == cached[0]
    assert forward[1] is forward[2] is None
