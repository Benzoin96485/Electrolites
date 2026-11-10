"""Shared metric and repeated-row GPU overlap VJPs against independent equations."""
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
@pytest.mark.parametrize('direct',[False,True])
def test_embedded_overlap_all_source_subsets(backend, mask, direct):
    cp = gpu()
    xp = np if backend == 'numpy' else cp
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.adjoint import host
    from electrolites.dlpno.overlap_union_adjoint_gpu import embedded_batch
    rng = np.random.default_rng(120)
    positions = [np.array([3, 0, 3]), np.array([1, 2]), np.array([4, 0, 2])]
    raw = [rng.normal(size=(5, 5))] + [rng.normal(size=(len(p), n)) for p, n in zip(positions, (2, 3, 1))]
    couples = ((0, 1), (1, 0), (0, 0), (1, 2), (2, 1))
    weights = [rng.normal(size=(raw[p+1].shape[1], raw[q+1].shape[1])) for p, q in couples]
    leaves = [ad.Variable(xp.asarray(x)) for x in raw]
    if direct:
        padded=embedded_batch(leaves[0],leaves[1:],positions,couples,padded_output=(8,8))
        outputs=[padded[i,:raw[p+1].shape[1],:raw[q+1].shape[1]] for i,(p,q) in enumerate(couples)]
    else:outputs = embedded_batch(leaves[0], leaves[1:], positions, couples)
    def equations(values):
        return [values[p+1].T @ values[0][np.ix_(positions[p], positions[q])] @ values[q+1] for p, q in couples]
    exact = equations(leaves)
    forward_error = max(float(np.max(abs(host(a)-host(b)), initial=0.)) for a, b in zip(outputs, exact))
    assert forward_error < 2e-12
    loss = sum(np.sum(y*xp.asarray(w)) for y, w in zip(outputs, weights))
    exact_loss = sum(np.sum(y*xp.asarray(w)) for y, w in zip(exact, weights))
    selected = [i for i in range(4) if mask & (1 << i)]
    inputs = [leaves[i] for i in selected]
    expected = ad.gradients(exact_loss, inputs)
    graph = ad.Linearization([loss], inputs)
    actual = graph([xp.asarray(1.)])
    vjp_error = max(float(np.max(abs(host(a)-host(b)), initial=0.)) for a, b in zip(actual, expected))
    assert vjp_error < 2e-11
    repeated = graph([xp.asarray(.31)])
    assert max(float(np.max(abs(host(a)-.31*host(b)), initial=0.)) for a, b in zip(repeated, actual)) < 2e-11
    direction = [rng.normal(size=x.shape) if i in selected else np.zeros_like(x) for i, x in enumerate(raw)]
    norm = np.sqrt(sum(np.sum(x*x) for x in direction))
    direction = [x/norm for x in direction]
    def oracle(values):
        return sum(np.sum(y*w) for y, w in zip(equations(values), weights))
    h = 1e-5
    finite = (oracle([x+h*d for x, d in zip(raw, direction)])-oracle([x-h*d for x, d in zip(raw, direction)]))/(2*h)
    analytic = sum(np.sum(host(bar)*direction[i]) for i, bar in zip(selected, actual))
    fd_error = float(abs(finite-analytic))
    assert fd_error < 2e-7
    consumed = graph([xp.asarray(1.)], release_graph=True)
    assert max(float(np.max(abs(host(a)-host(b)), initial=0.)) for a, b in zip(consumed, actual)) < 2e-11
    print('GPU_UNION_REPORT '+json.dumps(dict(backend=backend, mask=mask, direct=direct, forward_error=forward_error, vjp_error=vjp_error, directional_fd_error=fd_error)))


@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
def test_embedded_overlap_aliased_coefficients_empty_rank_and_late_checkpoint(backend):
    cp = gpu()
    xp = np if backend == 'numpy' else cp
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.adjoint import host
    from electrolites.dlpno.overlap_union_adjoint_gpu import embedded_batch
    rng = np.random.default_rng(121)
    raw = [rng.normal(size=(5, 5)), rng.normal(size=(3, 2)), rng.normal(size=(2, 3)), np.empty((2, 0))]
    leaves = [ad.Variable(xp.asarray(x)) for x in raw]
    m, x, y, empty = leaves
    coefficients = (x, y, x, empty)
    positions = (np.array([3, 0, 3]), np.array([1, 2]), np.array([4, 1, 0]), np.array([2, 2]))
    couples = ((0, 1), (1, 0), (0, 2), (2, 0), (2, 2), (0, 1), (3, 0), (1, 3), (3, 3))
    weights = [rng.normal(size=(coefficients[p].shape[1], coefficients[q].shape[1])) for p, q in couples]
    outputs, _ = ad.checkpoint(lambda: embedded_batch(m, coefficients, positions, couples))
    (early,), _ = ad.checkpoint(lambda: (np.sum(outputs[0]*outputs[0]),))
    (late,), _ = ad.checkpoint(lambda: (np.sum(outputs[0]@outputs[1]),))
    loss = early+.13*late+sum(np.sum(z*xp.asarray(w)) for z, w in zip(outputs, weights))
    exact = [coefficients[p].T@m[np.ix_(positions[p], positions[q])]@coefficients[q] for p, q in couples]
    expected_loss = np.sum(exact[0]*exact[0])+.13*np.sum(exact[0]@exact[1])+sum(np.sum(z*xp.asarray(w)) for z, w in zip(exact, weights))
    expected = ad.gradients(expected_loss, leaves)
    graph = ad.Linearization([loss], leaves)
    reusable = graph([xp.asarray(1.)])
    consumed = graph([xp.asarray(1.)], release_graph=True, release_inputs=True)
    error = max(float(np.max(abs(host(a)-host(b)), initial=0.)) for a, b in zip(consumed, expected))
    assert error < 2e-10
    assert max(float(np.max(abs(host(a)-host(b)), initial=0.)) for a, b in zip(consumed, reusable)) < 2e-10
    assert all(leaf.v is None for leaf in leaves)
    assert host(consumed[-1]).shape == (2, 0)
    print('GPU_UNION_CHECKPOINT_REPORT '+json.dumps(dict(backend=backend, max_source_error=error, aliased_coefficients=True, repeated_pao_rows=True, empty_rank=True, consuming_inputs_released=True)))


def test_embedded_overlap_workspace_and_exact_union():
    gpu()
    from electrolites.dlpno.overlap_union_adjoint_gpu import union_batches, union_workspace_bytes
    pao = [np.arange(n)[::-1] for n in (7, 12, 65, 97)]
    coefficients = [np.empty((len(p), n)) for p, n in zip(pao, (2, 4, 13, 20))]
    keys = tuple((p, q) for p in range(4) for q in range(4))
    groups = union_batches(keys, pao, coefficients, max_bytes=2 << 20, max_count=3)
    assert sorted(key for group, _, _, _ in groups for key in group) == sorted(keys)
    for group, union, involved, bound in groups:
        assert len(group) <= 3 and bound <= 2 << 20
        assert list(involved) == sorted({p for key in group for p in key})
        assert list(union) == sorted({int(row) for p in involved for row in pao[p]})
        assert bound == union_workspace_bytes(len(involved), len(group), len(union), max(coefficients[p].shape[1] for p in involved))
    with pytest.raises(ValueError, match='One embedded overlap'):
        union_batches(((3, 3),), pao, coefficients, max_bytes=64)


@pytest.mark.parametrize('backend',['numpy','cupy'])
def test_rectangular_direct_overlap_empty_rank_alias_and_full_sources(backend):
    cp=gpu();xp=np if backend=='numpy' else cp
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.adjoint import host
    from electrolites.dlpno.overlap_union_adjoint_gpu import embedded_batch
    rng=np.random.default_rng(764)
    raw=[rng.normal(size=(5,5)),rng.normal(size=(3,11)),rng.normal(size=(2,7)),rng.normal(size=(3,2)),np.empty((2,0))]
    leaves=[ad.Variable(xp.asarray(x)) for x in raw]
    m,x,y,z,empty=leaves;coefficients=(x,y,z,empty,x)
    positions=(np.array([3,0,3]),np.array([1,2]),np.array([4,0,2]),np.array([2,2]),np.array([4,1,0]))
    couples=((0,1),(0,2),(2,1),(2,2),(0,1),(3,1),(0,3),(3,3),(4,1))
    result=embedded_batch(m,coefficients,positions,couples,padded_output=(16,8))
    exact=[coefficients[p].T@m[np.ix_(positions[p],positions[q])]@coefficients[q] for p,q in couples]
    seed=rng.normal(size=result.shape) # Nonzero padding seeds must have no physical effect.
    loss=(result*xp.asarray(seed)).sum()
    ref=sum((value*xp.asarray(seed[i,:value.shape[0],:value.shape[1]])).sum() for i,value in enumerate(exact))
    gold=ad.gradients(ref,leaves);graph=ad.Linearization([loss],leaves);bars=graph([xp.asarray(1.)])
    error=lambda aa,bb:max(float(np.max(abs(host(a)-host(b)),initial=0.)) for a,b in zip(aa,bb))
    vjp_error=error(bars,gold);assert vjp_error<2e-10,vjp_error
    value_error=max(float(np.max(abs(host(result)[i,:e.shape[0],:e.shape[1]]-host(e)),initial=0.)) for i,e in enumerate(exact))
    assert value_error<2e-12,value_error
    for i,e in enumerate(exact):
        assert np.max(abs(host(result)[i,e.shape[0]:,:]),initial=0.)==0.
        assert np.max(abs(host(result)[i,:,e.shape[1]:]),initial=0.)==0.
    for i in range(len(leaves)):
        assert error(ad.Linearization([loss],[leaves[i]],boundary=leaves)([xp.asarray(1.)]),[gold[i]])<2e-10
    assert error(graph([xp.asarray(.31)]),[.31*x for x in gold])<2e-10
    assert error(graph([xp.asarray(1.)],release_graph=True,release_inputs=True),gold)<2e-10
    print('GPU_DIRECT_RECTANGLE_REPORT '+json.dumps(dict(backend=backend,value_error=value_error,vjp_error=vjp_error,shape=[len(couples),16,8])))
