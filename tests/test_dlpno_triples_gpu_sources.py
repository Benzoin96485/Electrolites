"""GPU local triples sources against original equations and full source finite differences."""
import json
from types import MethodType, SimpleNamespace
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import _names, host
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.reference_cpu import rebind
from electrolites.dlpno.triples_adjoint import triplet_source, transfer


def gpu():
    cp = pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('CUDA required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA required')
    return cp


def state_from(values, triplet, empty):
    ranks = [2, 3, 4, 3, 2, 4, 4, 3, 2]
    state = SimpleNamespace(pl=SimpleNamespace(ij=np.arange(9).reshape(3, 3)),
        trip=[triplet], t_ribfs=[np.array([6, 1, 4, 0, 3])],
        t_paos=[np.array([5, 1, 3, 0])],
        t_lmos=[np.array([], dtype=int) if empty else np.array([2, 0, 1])],
        pao=[np.roll(np.arange(6), p)[:5] for p in range(9)],
        X=[values['x'+str(p)] for p in range(9)],
        T=[values['t'+str(p)] for p in range(9)],
        T1=[values['s'+str(i)] for i in range(3)],
        X_tno=[values['xt']], e_tno=[values['eps']],
        qij=values['qo'], qia=values['qv'], qab=values['qvv'],
        J2c=(values['metric']+values['metric'].T)*.5,
        orb=SimpleNamespace(S_pao=(values['overlap']+values['overlap'].T)*.5,
                            F_lmo=(values['fock']+values['fock'].T)*.5))
    state._S_tp = MethodType(rebind(RefDLPNO._S_tp, **_names), state)
    state._e_ijk = rebind(RefDLPNO._e_ijk, **_names)
    return state


def raw_sources():
    rng = np.random.default_rng(124)
    ranks = [2, 3, 4, 3, 2, 4, 4, 3, 2]
    raw = dict(qo=.2*rng.normal(size=(7, 3, 3)), qv=.2*rng.normal(size=(7, 3, 6)),
               qvv=.2*rng.normal(size=(7, 6, 6)), xt=.2*rng.normal(size=(4, 3)),
               eps=np.array([.2, .5, .9]), fock=-np.diag([1., .8, .6]))
    frame, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    raw['metric'] = (frame*np.array([.4, .4, .4, 1., 1., 2., 3.]))@frame.T
    raw['overlap'] = np.eye(6)+.02*rng.normal(size=(6, 6))
    for p, n in enumerate(ranks):
        raw['x'+str(p)] = .3*rng.normal(size=(5, n))
        raw['t'+str(p)] = .2*rng.normal(size=(n, n))
    for i in range(3):
        raw['s'+str(i)] = .1*rng.normal(size=ranks[3*i+i])
    return raw, rng


@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
@pytest.mark.parametrize('triplet,empty', [((0, 1, 2), False), ((0, 0, 2), False),
                                         ((1, 1, 1), False), ((0, 1, 2), True)])
@pytest.mark.parametrize('output_count', [1, 4])
def test_gpu_local_triples_sources_and_consuming_replay(backend, triplet, empty, output_count):
    cp = gpu()
    xp = np if backend == 'numpy' else cp
    raw, rng = raw_sources()
    variables = {name: ad.Variable(xp.asarray(value)) for name, value in raw.items()}
    state = state_from(variables, triplet, empty)
    oracle = rebind(RefDLPNO._triplet_t0, **_names)
    expected = oracle(state, 0)[:output_count]
    actual, _ = ad.checkpoint(lambda: triplet_source(state, 0, gpu=True)[:output_count])
    assert all(isinstance(value.v, xp.ndarray) for value in actual)
    seeds = [xp.asarray(rng.normal(size=value.shape)) for value in expected]
    inputs = list(variables.values())
    exact_bars = ad.Linearization(expected, inputs)(seeds)
    graph = ad.Linearization(actual, inputs)
    actual_bars = graph(seeds)
    value_error = max(float(np.max(abs(host(a)-host(b)), initial=0.)) for a, b in zip(actual, expected))
    vjp_error = max(float(np.max(abs(host(a)-host(b)), initial=0.)) for a, b in zip(actual_bars, exact_bars))
    assert value_error < 1e-11 and vjp_error < 2e-10
    repeated = graph([.37*seed for seed in seeds])
    assert max(float(np.max(abs(host(a)-.37*host(b)), initial=0.)) for a, b in zip(repeated, actual_bars)) < 2e-10
    selected_errors = []
    names = list(variables)
    for amplitudes in (False, True):
        indices = [i for i, name in enumerate(names) if name.startswith(('t', 's')) == amplitudes]
        selected = ad.Linearization(actual, [inputs[i] for i in indices], boundary=inputs)(seeds)
        selected_errors.append(max(float(np.max(abs(host(bar)-host(exact_bars[i])), initial=0.)) for i, bar in zip(indices, selected)))
    assert max(selected_errors) < 2e-10
    # Independent FD always uses the original CPU equations and varies all
    # integrals, the metric, subspaces, orbital energies and amplitudes.
    directions = {name: .1*rng.normal(size=value.shape) for name, value in raw.items()}
    h = 2e-5
    targets = []
    for sign in (1, -1):
        displaced = {name: value+sign*h*directions[name] for name, value in raw.items()}
        outputs = oracle(state_from(displaced, triplet, empty), 0)[:output_count]
        targets.append(sum(float(np.sum(host(a)*host(b))) for a, b in zip(outputs, seeds)))
    finite = (targets[0]-targets[1])/(2*h)
    analytic = sum(float(np.sum(host(bar)*directions[name])) for name, bar in zip(variables, actual_bars))
    fd_error = abs(finite-analytic)
    assert fd_error < 2e-8
    consumed = graph(seeds, release_graph=True, release_inputs=True)
    assert max(float(np.max(abs(host(a)-host(b)), initial=0.)) for a, b in zip(consumed, actual_bars)) < 2e-10
    assert all(leaf.v is None for leaf in inputs)
    print('GPU_TRIPLES_SOURCE_REPORT '+json.dumps(dict(backend=backend, triplet=triplet, empty=empty,
          outputs=output_count, max_value_error=value_error, max_vjp_error=vjp_error,
          max_selected_source_error=max(selected_errors), directional_fd_error=fd_error,
          consuming_inputs_released=True), sort_keys=True))


@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
def test_transfer_roundtrip_retains_shared_and_empty_inputs(backend):
    cp = gpu()
    xp = np if backend == 'numpy' else cp
    other = cp if xp is np else np
    errors = []
    for shape in ((3, 4), (2, 0)):
        values = xp.arange(np.prod(shape), dtype=float).reshape(shape)*.1
        x = ad.Variable(values)
        moved = transfer(transfer(x, other), xp)
        assert isinstance(moved.v, xp.ndarray)
        loss = np.sum(moved*moved)+.2*np.sum(moved*x)
        graph = ad.Linearization([loss], [x])
        bar, = graph([xp.asarray(1.)])
        error = float(np.max(abs(host(bar)-2.4*host(values)), initial=0.))
        assert error < 2e-12
        consumed, = graph([xp.asarray(1.)], release_graph=True, release_inputs=True)
        assert np.array_equal(host(consumed), host(bar)) and x.v is None
        errors.append(error)
    print('GPU_TRANSFER_REPORT '+json.dumps(dict(backend=backend, maximum_error=max(errors), empty_shape_tested=True)))


def test_cpu_checkpoint_does_not_retain_per_triplet_gpu_workspaces():
    import gc
    cp = gpu()
    raw, _ = raw_sources()
    variables = {name: ad.Variable(value.copy()) for name, value in raw.items()}
    state = state_from(variables, (0, 1, 2), False)
    def checkpoint():
        return ad.checkpoint(lambda: triplet_source(state, 0, gpu=True))[0]
    for _ in range(2):
        warm = checkpoint()
        ad.Linearization(warm, list(variables.values()))([np.ones(value.shape) for value in warm])
    del warm
    cp.cuda.Stream.null.synchronize()
    gc.collect()
    before = cp.get_default_memory_pool().used_bytes()
    held = [checkpoint() for _ in range(24)]
    cp.cuda.Stream.null.synchronize()
    gc.collect()
    after_forward = cp.get_default_memory_pool().used_bytes()
    assert after_forward <= before
    for outputs in held:
        assert all(isinstance(value.v, np.ndarray) for value in outputs)
        bars = ad.Linearization(outputs, list(variables.values()))([np.ones(value.shape) for value in outputs])
        assert all(isinstance(value, np.ndarray) for value in bars)
    cp.cuda.Stream.null.synchronize()
    gc.collect()
    after_reverse = cp.get_default_memory_pool().used_bytes()
    assert after_reverse <= before
    print('GPU_TRIPLES_LIFETIME_REPORT '+json.dumps(dict(checkpoints=24, before_used_bytes=before,
          after_forward_used_bytes=after_forward, after_reverse_used_bytes=after_reverse,
          all_retained_outputs_on_host=True, allocator_reserve_not_counted=True)))
