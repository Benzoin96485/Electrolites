"""Independent equations and differences for common-left triples overlaps."""
import json
import numpy as np
import pytest


def gpu():
    cp = pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('CUDA required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA required')
    return cp


@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
@pytest.mark.parametrize('mask', range(1, 16))
def test_common_left_all_source_subsets(backend, mask):
    cp = gpu()
    xp = np if backend == 'numpy' else cp
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.adjoint import host
    from electrolites.dlpno.triples_adjoint import transfer
    from electrolites.dlpno.triples_overlap_adjoint_gpu import common_left_batch
    rng = np.random.default_rng(193)
    positions = (np.array([3, 0, 3]), np.array([4, 1]))
    raw = [rng.normal(size=s) for s in ((4, 5), (4, 3), (3, 2), (2, 4))]
    leaves = [ad.Variable(xp.asarray(x)) for x in raw]
    weights = [rng.normal(size=(3, 2)), rng.normal(size=(3, 4))]

    def equations(values):
        m, a, *bs = values
        return [a.T @ m[:, pos] @ b for pos, b in zip(positions, bs)]

    outputs, _ = ad.checkpoint(lambda: tuple(transfer(y, xp) for y in
        common_left_batch(leaves[0], leaves[1], leaves[2:], positions)))
    exact = equations(leaves)
    forward_error = max(float(np.max(abs(host(a)-host(b)))) for a, b in zip(outputs, exact))
    selected = [i for i in range(4) if mask & (1 << i)]
    inputs = [leaves[i] for i in selected]
    seeds = [xp.asarray(w) for w in weights]
    expected = ad.Linearization(exact, inputs, boundary=leaves)(seeds)
    graph = ad.Linearization(outputs, inputs, boundary=leaves)
    actual = graph(seeds)
    error = max(float(np.max(abs(host(a)-host(b)))) for a, b in zip(actual, expected))
    assert forward_error < 2e-12 and error < 2e-11
    repeated = graph([.37*w for w in seeds])
    assert max(float(np.max(abs(host(a)-.37*host(b)))) for a, b in zip(repeated, actual)) < 2e-11
    direction = [rng.normal(size=x.shape) if i in selected else np.zeros_like(x)
                 for i, x in enumerate(raw)]
    norm = np.sqrt(sum(np.sum(x*x) for x in direction))
    direction = [x/norm for x in direction]

    def oracle(values):
        return sum(np.sum(y*w) for y, w in zip(equations(values), weights))

    h = 1e-5
    finite = (oracle([x+h*d for x, d in zip(raw, direction)])
              - oracle([x-h*d for x, d in zip(raw, direction)]))/(2*h)
    analytic = sum(np.sum(host(bar)*direction[i]) for i, bar in zip(selected, actual))
    fd_error = float(abs(finite-analytic))
    assert fd_error < 2e-7
    consumed = graph(seeds, release_graph=True)
    assert max(float(np.max(abs(host(a)-host(b)))) for a, b in zip(consumed, actual)) < 2e-11
    print('GPU_TRIPLES_OVERLAP_REPORT '+json.dumps(dict(backend=backend, mask=mask,
        forward_error=forward_error, vjp_error=error, directional_fd_error=fd_error)))


@pytest.mark.parametrize('backend', ['numpy', 'cupy', 'mixed'])
def test_common_left_aliases_empty_rank_and_late_checkpoint(backend):
    cp = gpu()
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.adjoint import host
    from electrolites.dlpno.triples_adjoint import transfer
    from electrolites.dlpno.triples_overlap_adjoint_gpu import common_left_batch
    rng = np.random.default_rng(194)
    raw = [rng.normal(size=s) for s in ((3, 5), (3, 2), (2, 3), (2, 0))]
    xps = [np]*4 if backend == 'numpy' else [cp]*4 if backend == 'cupy' else [np, cp, np, cp]
    leaves = [ad.Variable(xp.asarray(x)) for xp, x in zip(xps, raw)]
    m, a, b, empty = leaves
    positions = (np.array([3, 0, 3]), np.array([1, 4]), np.array([4, 1, 0]), np.array([2, 2]))
    bs = (a, b, a, empty)
    weights = [rng.normal(size=(2, x.shape[1])) for x in bs]
    outputs, _ = ad.checkpoint(lambda: tuple(transfer(y, np) for y in
        common_left_batch(m, a, bs, positions)))
    (early,), _ = ad.checkpoint(lambda: (np.sum(outputs[0]*outputs[0]),))
    (late,), _ = ad.checkpoint(lambda: (np.sum(outputs[0]@outputs[2].T),))
    loss = early+.13*late+sum(np.sum(y*w) for y, w in zip(outputs, weights))
    oracle_leaves = [ad.Variable(x.copy()) for x in raw]
    om, oa, ob, oe = oracle_leaves
    exact = [oa.T@om[:, pos]@x for pos, x in zip(positions, (oa, ob, oa, oe))]
    expected_loss = (np.sum(exact[0]*exact[0])+.13*np.sum(exact[0]@exact[2].T)
                     + sum(np.sum(y*w) for y, w in zip(exact, weights)))
    expected = ad.gradients(expected_loss, oracle_leaves)
    graph = ad.Linearization([loss], leaves)
    actual = graph([np.asarray(1.)])
    consumed = graph([np.asarray(1.)], release_graph=True, release_inputs=True)
    error = max(float(np.max(abs(host(a)-host(b)), initial=0.)) for a, b in zip(consumed, expected))
    assert error < 2e-10
    assert max(float(np.max(abs(host(a)-host(b)), initial=0.)) for a, b in zip(consumed, actual)) < 2e-10
    assert all(isinstance(bar, xp.ndarray) for bar, xp in zip(consumed, xps))
    assert all(leaf.v is None for leaf in leaves)
    print('GPU_TRIPLES_OVERLAP_ALIAS_REPORT '+json.dumps(dict(backend=backend,
        max_source_error=error, inputs_released=True, empty_shape=list(consumed[-1].shape))))


def test_common_left_batches_bound_exact_requests_and_rows():
    gpu()
    from electrolites.dlpno.triples_overlap_adjoint_gpu import overlap_batches, workspace_bytes
    pao = [np.arange(n)[::-1] for n in (7, 12, 65, 97)]
    xs = [np.empty((len(p), n)) for p, n in zip(pao, (2, 4, 13, 20))]
    left = np.empty((80, 25))
    groups = overlap_batches((2, 1, 2, 3, 0), left, pao, xs, max_bytes=2 << 20, max_count=2)
    assert tuple(p for group, _, _ in groups for p in group) == (2, 1, 3, 0)
    for group, union, bound in groups:
        assert len(group) <= 2 and bound <= 2 << 20
        assert list(union) == sorted({int(row) for p in group for row in pao[p]})
        assert bound == workspace_bytes(80, len(union), 25,
            sum(xs[p].shape[1] for p in group), sum(xs[p].size for p in group))
    with pytest.raises(ValueError, match='One common-left overlap'):
        overlap_batches((3,), left, pao, xs, max_bytes=64)
