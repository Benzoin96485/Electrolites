"""Whole-source fusion: aliases, mixed backends, sparse selections and bounds."""
import json
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import _names, host
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.reference_cpu import rebind
from electrolites.dlpno.triples_adjoint import triplet_source, operator_triplet_source
from electrolites.dlpno.triples_source_adjoint_gpu import triplet_source as fused
from test_dlpno_triples_gpu_sources import gpu, raw_sources, state_from


@pytest.mark.parametrize('energy_only', [False, True])
def test_fused_source_mixed_backends_aliases_and_single_input_selections(energy_only):
    cp = gpu()
    raw, rng = raw_sources()
    cpu = {k: ad.Variable(v.copy()) for k, v in raw.items()}
    mixed = {k: ad.Variable((cp if pos % 2 else np).asarray(v))
             for pos, (k, v) in enumerate(raw.items())}
    for variables in (cpu, mixed):
        variables['x4'] = variables['x0']
        variables['t4'] = variables['t0']
    keys = [k for k in raw if k not in ('x4', 't4')]
    oracle = rebind(RefDLPNO._triplet_t0, **_names)
    expected = oracle(state_from(cpu, (0, 1, 2), False), 0)
    if energy_only:
        expected = expected[:1]
    state = state_from(mixed, (0, 1, 2), False)
    actual, info = ad.checkpoint(lambda: triplet_source(state, 0, gpu=True, energy_only=energy_only))
    _, previous = ad.checkpoint(lambda: operator_triplet_source(state, 0, gpu=True, energy_only=energy_only))
    assert info['records'] < previous['records']
    seeds = [rng.normal(size=x.shape) for x in expected]
    gold = ad.Linearization(expected, [cpu[k] for k in keys])(seeds)
    inputs = [mixed[k] for k in keys]
    device_seeds = [ad._xp([out]).asarray(seed) for out, seed in zip(actual, seeds)]
    bars = ad.Linearization(actual, inputs)(device_seeds)
    error = max(np.max(abs(host(a)-host(b)), initial=0.) for a, b in zip(bars, gold))
    assert error < 2e-10
    for pos, leaf in enumerate(inputs):
        selected, = ad.Linearization(actual, [leaf], boundary=inputs)(device_seeds)
        assert ad._xp([selected]) is ad._xp([leaf])
        assert np.max(abs(host(selected)-host(gold[pos])), initial=0.) < 2e-10
    value_error = max(np.max(abs(host(a)-host(b)), initial=0.) for a, b in zip(actual, expected))
    assert value_error < 1e-11
    print('FUSED_SOURCE_REPORT '+json.dumps(dict(energy_only=energy_only,
          previous_records=previous['records'], fused_records=info['records'],
          previous_bytes=previous['inner_value_bytes'], fused_bytes=info['inner_value_bytes'],
          value_error=float(value_error), vjp_error=float(error), mixed_backends=True,
          shared_coefficient_and_amplitude_aliases=True, individual_selections=len(inputs))))


def test_fused_source_workspace_bound_declines_before_building_overlap(monkeypatch):
    gpu()
    raw, _ = raw_sources()
    state = state_from(raw, (0, 1, 2), False)
    import electrolites.dlpno.triples_source_adjoint_gpu as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Workspace rejection must precede GPU overlap allocation')
    monkeypatch.setattr(module, 'triplet_overlaps', forbidden)
    assert fused(state, 0, max_workspace_bytes=0) is None


@pytest.mark.parametrize('energy_only', [False, True])
@pytest.mark.parametrize('selection', ['all', 'amplitudes'])
def test_replay_reuses_fit_and_releases_workspace(energy_only, selection, monkeypatch):
    import gc
    import weakref
    cp = gpu()
    raw, rng = raw_sources()
    variables = {k: ad.Variable(v.copy()) for k, v in raw.items()}
    state = state_from(variables, (0, 0, 2), False)
    oracle = rebind(RefDLPNO._triplet_t0, **_names)(state, 0)
    expected = oracle[:1] if energy_only else oracle
    seeds = [rng.normal(size=value.shape) for value in expected]
    inputs = [v for k, v in variables.items()
              if selection == 'all' or k.startswith(('t', 's'))]
    gold = ad.Linearization(expected, inputs, boundary=list(variables.values()))(seeds)
    solves = []
    original = cp.linalg.solve
    def solve(*args, **kwargs):
        value = original(*args, **kwargs)
        solves.append(weakref.ref(value))
        return value
    monkeypatch.setattr(cp.linalg, 'solve', solve)
    actual, _ = ad.checkpoint(lambda: fused(state, 0, energy_only=energy_only))
    assert len(solves) == 1
    gc.collect()
    assert all(ref() is None for ref in solves)
    graph = ad.Linearization(actual, inputs, boundary=list(variables.values()))
    errors, reverse_solves = [], []
    for scale in (1., .37):
        before = len(solves)
        bars = graph([scale*seed for seed in seeds])
        reverse_solves.append(len(solves)-before)
        # One forward fit per replay, plus its exact transpose fit only
        # when integral/metric sources are requested. No second forward fit.
        assert reverse_solves[-1] == (2 if selection == 'all' else 1)
        errors.append(max(np.max(abs(got-scale*want), initial=0.)
                          for got, want in zip(bars, gold)))
        assert errors[-1] < 2e-10
        cp.cuda.Stream.null.synchronize()
        gc.collect()
        assert all(ref() is None for ref in solves)
        assert ad.checkpoint_replay_workspace() is None
    print('REPLAY_REUSE_REPORT '+json.dumps(dict(energy_only=energy_only,
          selection=selection, repeated_reverse_solve_counts=reverse_solves,
          maximum_vjp_error=float(max(errors)), all_fit_arrays_released=True)))


def test_replay_clears_unused_and_failed_gpu_workspaces(monkeypatch):
    import gc
    import weakref
    cp = gpu()
    raw, _ = raw_sources()
    variables = {k: ad.Variable(v.copy()) for k, v in raw.items()}
    state = state_from(variables, (0, 1, 2), False)
    independent = ad.Variable(np.asarray(2.))
    fail, scopes, fits = [False], [], []
    original = cp.linalg.solve
    def solve(*args, **kwargs):
        value = original(*args, **kwargs)
        fits.append(weakref.ref(value))
        return value
    monkeypatch.setattr(cp.linalg, 'solve', solve)
    def forward():
        energy, = fused(state, 0, energy_only=True)
        workspace = ad.checkpoint_replay_workspace()
        if workspace is not None:
            assert workspace
            scopes.append(workspace)
            if fail[0]:
                raise RuntimeError('intentional replay failure')
        return energy, independent*independent
    outputs, _ = ad.checkpoint(forward)
    selected = ad.Linearization(outputs, [independent],
                                boundary=list(variables.values())+[independent])
    bar, = selected([np.asarray(1.), np.asarray(1.)])
    assert bar == 4.
    assert len(scopes) == 1 and not scopes[0]
    fail[0] = True
    with pytest.raises(RuntimeError, match='intentional replay failure') as error:
        selected([np.asarray(1.), np.asarray(1.)])
    # Retain the traceback to ensure clearing is explicit even on failure.
    assert error.value is not None
    cp.cuda.Stream.null.synchronize()
    gc.collect()
    assert all(not scope for scope in scopes)
    assert all(ref() is None for ref in fits)
    assert ad.checkpoint_replay_workspace() is None
    fail[0] = False
    bar, = selected([np.asarray(.37), np.asarray(.37)])
    assert abs(bar-1.48) < 1e-12
    assert all(not scope for scope in scopes)


def test_nested_checkpoint_replay_restores_outer_scope():
    x = ad.Variable(np.array([.3, -.2]))
    seen = []
    def inner():
        scope = ad.checkpoint_replay_workspace()
        if scope is not None:
            seen.append(scope)
            assert not scope
            scope['inner-only'] = object()
        return (np.sum(x*x),)
    def outer():
        scope = ad.checkpoint_replay_workspace()
        if scope is not None:
            seen.append(scope)
            scope['outer-only'] = object()
        value, _ = ad.checkpoint(inner)
        assert ad.checkpoint_replay_workspace() is scope
        if scope is not None:
            assert set(scope) == {'outer-only'}
        return (value[0]*np.sum(x),)
    outputs, _ = ad.checkpoint(outer)
    graph = ad.Linearization(outputs, [x])
    for scale in (1., .37):
        actual, = graph([np.asarray(scale)])
        expected = scale*(2*x.v*np.sum(x.v)+np.sum(x.v*x.v))
        assert np.max(abs(actual-expected)) < 1e-12
        assert all(not scope for scope in seen)
        assert ad.checkpoint_replay_workspace() is None
    assert len(seen) == 4
