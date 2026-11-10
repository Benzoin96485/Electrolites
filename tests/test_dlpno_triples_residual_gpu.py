"""Complete coupled-residual VJPs against unchanged equations and differences."""
import gc
import json
from itertools import permutations
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import _checkpointed_triplet_residual, host
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.reference_cpu import rebind


def gpu():
    cp = pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('CUDA required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA required')
    return cp


def raw_sources():
    rng = np.random.default_rng(195)
    raw = dict(metric=np.eye(8)+.1*rng.normal(size=(8, 8)),
               fock=np.array([[-1., .03, -.04], [.03, -.8, .02], [-.04, .02, -.6]]))
    for t, (n, p) in enumerate(zip((2, 3, 4, 2), (5, 6, 7, 5))):
        raw['x'+str(t)] = .2*rng.normal(size=(p, n))
        raw['e'+str(t)] = np.linspace(.2, 1., n)
        raw['w'+str(t)] = rng.normal(size=(n, n, n))
        raw['t'+str(t)] = rng.normal(size=(n, n, n))
    return raw, rng


def state_from(values, cut, *, backend='cupy'):
    trip = [(0, 0, 1), (0, 1, 1), (0, 1, 2), (0, 2, 2)]
    state = SimpleNamespace(trip=trip, numeric=ad.Numpy(np), diagnostics={},
        triples_backend=backend, opt={'F_CUT_T': cut},
        trip_idx={p: t for t, key in enumerate(trip) for p in permutations(key)},
        orb=SimpleNamespace(S_pao=values['metric'], F_lmo=values['fock']),
        t_paos=[np.array(p) for p in ((5, 0, 3, 0, 1), (7, 4, 0, 2, 5, 1),
                                     (6, 1, 0, 7, 2, 4, 6), (5, 0, 3, 0, 1))],
        X_tno=[values['x'+str(t)] for t in range(4)],
        e_tno=[values['e'+str(t)] for t in range(4)],
        W3=[values['w'+str(t)] for t in range(4)],
        T3=[values['t'+str(t)] for t in range(4)])
    # A shared coefficient is used in distinct triplet domains.
    state.X_tno[3] = state.X_tno[0]
    state._matmul_3d = MethodType(rebind(RefDLPNO._matmul_3d,
        lib=SimpleNamespace(einsum=ad.einsum)), state)
    return state


@pytest.mark.parametrize('backend', ['numpy', 'cupy', 'mixed'])
@pytest.mark.parametrize('cut', [0., .035, 2.])
def test_coupled_residual_all_sources_and_independent_difference(backend, cut):
    cp = gpu()
    raw, rng = raw_sources()
    xps = [cp if backend == 'cupy' or (backend == 'mixed' and i % 2) else np
           for i in range(len(raw))]
    leaves = {name: ad.Variable(xp.asarray(x)) for (name, x), xp in zip(raw.items(), xps)}
    oracle_leaves = {name: ad.Variable(x.copy()) for name, x in raw.items()}
    state, reference = state_from(leaves, cut), state_from(oracle_leaves, cut)
    connections = RefDLPNO._triples_couplings(state)
    # Exercise all permutations, repeated labels, repeated sources and a self edge.
    connections[0] = list(connections[0])+[(p, 2, 0, 2) for p in permutations((0, 1, 2))]
    connections[0] += [((1, 0, 0), 0, 0, 1)]
    evaluate = rebind(RefDLPNO._triplet_residual, np=ad.Numpy(np))
    expected = [evaluate(reference, t, c) for t, c in enumerate(connections)]
    actual = [_checkpointed_triplet_residual(state, t, c) for t, c in enumerate(connections)]
    seeds = [rng.normal(size=x.shape) for x in expected]
    gpu_seeds = [ad._xp([x]).asarray(z) for x, z in zip(actual, seeds)]
    exact_bars = ad.Linearization(expected, list(oracle_leaves.values()))(seeds)
    graph = ad.Linearization(actual, list(leaves.values()))
    actual_bars = graph(gpu_seeds)
    error = lambda aa, bb: max(float(np.max(abs(host(a)-host(b)), initial=0.)) for a, b in zip(aa, bb))
    value_error, vjp_error = error(actual, expected), error(actual_bars, exact_bars)
    assert value_error < 2e-12 and vjp_error < 2e-10
    assert all(isinstance(bar, xp.ndarray) for bar, xp in zip(actual_bars, xps))
    assert error(graph([.37*z for z in gpu_seeds]), [.37*b for b in actual_bars]) < 2e-10
    names, inputs = list(leaves), list(leaves.values())
    partial_errors = []
    for group in ('t', 'w', 'e', 'x', 'fock', 'metric'):
        indices = [i for i, name in enumerate(names) if name == group or name.startswith(group)]
        got = ad.Linearization(actual, [inputs[i] for i in indices], boundary=inputs)(gpu_seeds)
        partial_errors.append(error(got, [exact_bars[i] for i in indices]))
    assert max(partial_errors) < 2e-10
    direction = {name: .1*rng.normal(size=x.shape) for name, x in raw.items()}
    h, targets = 2e-5, []
    for sign in (1, -1):
        shifted = {name: x+sign*h*direction[name] for name, x in raw.items()}
        shifted_state = state_from(shifted, cut)
        # The finite difference evaluates the numerical CPU equations only.
        shifted_state._matmul_3d = MethodType(RefDLPNO._matmul_3d, shifted_state)
        targets.append(sum(float(np.sum(host(evaluate(shifted_state, t, c))*z))
                           for t, (c, z) in enumerate(zip(connections, seeds))))
    finite = (targets[0]-targets[1])/(2*h)
    analytic = sum(float(np.sum(host(b)*direction[name])) for name, b in zip(names, actual_bars))
    fd_error = abs(finite-analytic)
    assert fd_error < 2e-8
    # Reinstallation must leave the captured checkpoint inputs intact.
    state.T3 = []
    consumed = graph(gpu_seeds, release_graph=True, release_inputs=True)
    assert error(consumed, actual_bars) < 2e-10
    assert all(x.v is None for x in inputs)
    assert state.diagnostics['triples_residual_checkpoints']['gpu_count'] == 4
    print('GPU_TRIPLES_RESIDUAL_REPORT '+json.dumps(dict(backend=backend, cut=cut,
        maximum_value_error=value_error, maximum_vjp_error=vjp_error,
        maximum_selected_source_error=max(partial_errors), directional_fd_error=fd_error,
        all_source_backends_preserved=True, consuming_inputs_released=True)))


@pytest.mark.parametrize('shape', [(2, 3), (4, 2), (1, 1), (0, 2), (2, 0)])
@pytest.mark.parametrize('needed', [(True, False), (False, True), (True, True)])
def test_three_index_projection_vjp(shape, needed):
    cp = gpu()
    from electrolites.dlpno.triples_residual_adjoint_gpu import _project, _project_vjp
    n, m = shape
    rng = np.random.default_rng(195)
    a, s, z = rng.normal(size=(m, m, m)), rng.normal(size=(n, m)), rng.normal(size=(n, n, n))
    av, sv = ad.Variable(a), ad.Variable(s)
    original = ad.einsum('aA,bB,cC,ABC->abc', sv, sv, sv, av)
    expected = ad.Linearization([original], [av, sv])([z])
    actual = _project(cp.asarray(a), cp.asarray(s))
    bars = _project_vjp(cp.asarray(a), cp.asarray(s), cp.asarray(z), *needed)
    assert np.max(abs(host(actual)-host(original)), initial=0.) < 2e-11
    for bar, exact, need in zip(bars, expected, needed):
        if need:
            assert np.max(abs(host(bar)-exact), initial=0.) < 2e-10
        else:
            assert bar is None


def test_threshold_equality_and_oversized_workspace_fallback():
    gpu()
    from electrolites.dlpno.triples_residual_adjoint_gpu import triplet_residual, workspace_bytes
    raw, _ = raw_sources()
    leaves = {name: ad.Variable(x) for name, x in raw.items()}
    state = state_from(leaves, .03)
    rows = [((0, 1, 1), 1, 0, 1), ((0, 0, 1), 0, 1, 2)]
    actual = triplet_residual(state, 0, rows)
    expected = rebind(RefDLPNO._triplet_residual, np=ad.Numpy(np))(state, 0, rows)
    assert np.max(abs(host(actual)-host(expected))) < 2e-12
    bars = ad.Linearization([actual], list(leaves.values()))([np.ones(actual.shape)])
    exact = ad.Linearization([expected], list(leaves.values()))([np.ones(actual.shape)])
    assert max(np.max(abs(a-b), initial=0.) for a, b in zip(bars, exact)) < 2e-10
    assert triplet_residual(state, 0, rows, max_workspace_bytes=1) is None
    assert workspace_bytes(288, 288) < 8 << 30
    state.triples_backend = 'numpy'
    fallback = _checkpointed_triplet_residual(state, 0, rows)
    assert np.max(abs(host(fallback)-host(expected))) < 2e-12
    assert state.diagnostics['triples_residual_checkpoints']['gpu_count'] == 0


def test_host_checkpoints_retain_no_residual_gpu_workspaces():
    cp = gpu()
    raw, _ = raw_sources()
    leaves = {name: ad.Variable(x) for name, x in raw.items()}
    state = state_from(leaves, 0.)
    rows = RefDLPNO._triples_couplings(state)[2]
    def checkpoint():
        return [_checkpointed_triplet_residual(state, 2, rows)]
    for _ in range(2):
        output = checkpoint()
        ad.Linearization(output, list(leaves.values()))([np.ones(x.shape) for x in output])
    del output
    cp.cuda.Stream.null.synchronize()
    gc.collect()
    before = cp.get_default_memory_pool().used_bytes()
    held = [checkpoint() for _ in range(24)]
    cp.cuda.Stream.null.synchronize()
    gc.collect()
    after_forward = cp.get_default_memory_pool().used_bytes()
    assert after_forward <= before
    for outputs in held:
        assert all(isinstance(x.v, np.ndarray) for x in outputs)
        bars = ad.Linearization(outputs, list(leaves.values()))([np.ones(x.shape) for x in outputs])
        assert all(isinstance(x, np.ndarray) for x in bars)
    cp.cuda.Stream.null.synchronize()
    gc.collect()
    after_reverse = cp.get_default_memory_pool().used_bytes()
    assert after_reverse <= before
    print('GPU_TRIPLES_RESIDUAL_LIFETIME_REPORT '+json.dumps(dict(checkpoints=24,
        before_used_bytes=before, after_forward_used_bytes=after_forward,
        after_reverse_used_bytes=after_reverse, all_retained_outputs_on_host=True)))
