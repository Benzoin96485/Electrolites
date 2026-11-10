"""Independent reference equations and directional FD for ragged batch sources."""
import gc
import json
import weakref
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import _names, host
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.reference_cpu import rebind
from electrolites.dlpno.triples_source_batch_gpu import source_batch, source_batches, _layout, workspace_bytes
from test_dlpno_triples_gpu_sources import gpu, raw_sources, state_from


def multi_state(values):
    state = state_from(values, (0, 1, 2), False)
    state.trip = [(0, 1, 2), (0, 0, 2), (1, 1, 1), (0, 1, 2)]
    state.t_ribfs = [np.array(x) for x in ((6, 1, 4, 0, 3), (4, 0, 1), (0, 2), (6, 1, 4, 0))]
    state.t_paos = [np.array(x) for x in ((5, 1, 3, 0), (3, 0, 1), (1, 1), (5, 1, 3, 0))]
    state.t_lmos = [np.array(x, dtype=int) for x in ((2, 0, 1), (), (1, 0), (2, 2, 0))]
    state.X_tno = [values['xt'], values['xt'][:3, :2], values['xt'][:2, :1], values['xt']]
    state.e_tno = [values['eps'], values['eps'][:2], values['eps'][:1], values['eps']]
    state.X[4] = state.X[0]
    state.T[4] = state.T[0]
    return state


@pytest.mark.parametrize('reuse_reads', [False, True])
@pytest.mark.parametrize('backend', ['numpy', 'cupy', 'mixed'])
@pytest.mark.parametrize('energy_only', [False, True])
def test_batch_full_sources_ragged_aliases_repeats_and_fd(backend, energy_only, reuse_reads):
    cp = gpu()
    raw, rng = raw_sources()
    cpu = {k: ad.Variable(v.copy()) for k, v in raw.items()}
    variables = {k: ad.Variable((cp if backend == 'cupy' or backend == 'mixed' and i % 2 else np).asarray(v))
                 for i, (k, v) in enumerate(raw.items())}
    oracle = rebind(RefDLPNO._triplet_t0, **_names)
    reference_state, state = multi_state(cpu), multi_state(variables)
    count = 1 if energy_only else 4
    expected = tuple(value for t in range(4) for value in oracle(reference_state, t)[:count])
    actual, info = ad.checkpoint(lambda: source_batch(state, (0, 1, 2, 3), energy_only=energy_only, reuse_reads=reuse_reads))
    seeds = [rng.normal(size=x.shape) for x in expected]
    gold = ad.Linearization(expected, list(cpu.values()))(seeds)
    device_seeds = [ad._xp([out]).asarray(seed) for out, seed in zip(actual, seeds)]
    inputs = list(variables.values())
    graph = ad.Linearization(actual, inputs)
    bars = graph(device_seeds)
    value_error = max(np.max(abs(host(a)-host(b)), initial=0.) for a, b in zip(actual, expected))
    vjp_error = max(np.max(abs(host(a)-host(b)), initial=0.) for a, b in zip(bars, gold))
    assert value_error < 2e-11 and vjp_error < 3e-10
    repeated = graph([.37*z for z in device_seeds])
    assert max(np.max(abs(host(a)-.37*host(b)), initial=0.) for a, b in zip(repeated, gold)) < 3e-10
    selected_error = 0.
    for pos, leaf in enumerate(inputs):
        selected, = ad.Linearization(actual, [leaf], boundary=inputs)(device_seeds)
        assert ad._xp([selected]) is ad._xp([leaf])
        selected_error = max(selected_error, np.max(abs(host(selected)-host(gold[pos])), initial=0.))
    assert selected_error < 3e-10
    directions = {k: .1*rng.normal(size=v.shape) for k, v in raw.items()}
    step = 2e-5
    targets = []
    for sign in (1, -1):
        displaced = multi_state({k: v+sign*step*directions[k] for k, v in raw.items()})
        outputs = tuple(value for t in range(4) for value in oracle(displaced, t)[:count])
        targets.append(sum(float(np.sum(host(v)*z)) for v, z in zip(outputs, seeds)))
    finite = (targets[0]-targets[1])/(2*step)
    analytic = sum(float(np.sum(host(bar)*directions[k])) for k, bar in zip(variables, bars))
    fd_error = abs(finite-analytic)
    assert fd_error < 2e-8
    consumed = graph(device_seeds, release_graph=True, release_inputs=True)
    assert max(np.max(abs(host(a)-host(b)), initial=0.) for a, b in zip(consumed, gold)) < 3e-10
    assert all(x.v is None for x in inputs)
    print('BATCH_SOURCE_REPORT '+json.dumps(dict(backend=backend, energy_only=energy_only, reuse_reads=reuse_reads,
        triplets=4, value_error=float(value_error), vjp_error=float(vjp_error),
        selected_error=float(selected_error), directional_fd_error=fd_error,
        consuming_inputs_released=True, checkpoint=info)))


def test_batch_bound_and_planner_cover_each_triplet_once(monkeypatch):
    gpu()
    raw, _ = raw_sources()
    state = multi_state(raw)
    groups = source_batches(state, max_count=2, max_bytes=2 << 30)
    assert sorted(t for group in groups for t in group) == list(range(4))
    assert max(map(len, groups)) <= 2
    import electrolites.dlpno.triples_source_batch_gpu as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Reject oversized batch before any overlap allocation')
    monkeypatch.setattr(module, 'triplet_overlaps', forbidden)
    assert source_batch(state, (0, 1), max_workspace_bytes=0) is None
    assert source_batch(state, ()) == ()
    assert workspace_bytes([_layout(state, t) for t in range(4)]) > 0


@pytest.mark.parametrize('energy_only', [False, True])
def test_batch_replay_workspace_is_reused_and_released(energy_only, monkeypatch):
    cp = gpu()
    raw, _ = raw_sources()
    variables = {k: ad.Variable(v.copy()) for k, v in raw.items()}
    state = multi_state(variables)
    fits = []
    original = cp.linalg.solve
    def solve(*args, **kwargs):
        result = original(*args, **kwargs)
        fits.append(weakref.ref(result))
        return result
    monkeypatch.setattr(cp.linalg, 'solve', solve)
    actual, _ = ad.checkpoint(lambda: source_batch(state, (0, 1, 2, 3), energy_only=energy_only))
    assert len(fits) == 1
    gc.collect()
    assert all(x() is None for x in fits)
    graph = ad.Linearization(actual, list(variables.values()))
    for scale in (1., .37):
        before = len(fits)
        graph([scale*np.ones(x.shape) for x in actual])
        assert len(fits)-before == 2
        cp.cuda.Stream.null.synchronize()
        gc.collect()
        assert all(x() is None for x in fits)
        assert ad.checkpoint_replay_workspace() is None
