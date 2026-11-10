"""Batched MP2/PNO values and every source VJP against scalar equations."""
import json
from types import MethodType
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import _names, host
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.reference_cpu import rebind
from test_dlpno_parallel_local_gpu import raw_data, make_state, aligned


def state_for(raw, truncation):
    state, _ = make_state(raw)
    state.e_pno = [raw['ep'+str(p)] for p in range(9)]
    state.numeric = ad.Numpy(np)
    state.de_pno = state.numeric.zeros(9)
    state._store_pair = MethodType(RefDLPNO._store_pair, state)
    if truncation:
        state.opt.update(T_CUT_PNO_MP2=.06, T_CUT_TRACE_MP2=0., T_CUT_ENERGY_MP2=0., MIN_PNOS=1)
    for name in ('T_CUT_PNO', 'T_CUT_TRACE', 'T_CUT_ENERGY'):
        state.opt[name] = state.opt[name+'_MP2']
    return state


def oracle(raw, kind, items, truncation):
    state = state_for(raw, truncation)
    if kind == 'recompute':
        rebind(RefDLPNO.recompute_pnos, **_names)(state)
        return tuple(x for p in items for x in (state.X[p], state.e_pno[p], state.K[p], state.T[p], state.de_pno[p]))
    function = rebind(getattr(RefDLPNO, '_sc_mp2_pair_energy' if kind == 'sc_mp2' else '_lmp2_pno_pair'), **_names)
    return tuple(x for p in items for x in ((function(state, p),) if kind == 'sc_mp2' else function(state, p)))


@pytest.mark.parametrize('kind', ['sc_mp2', 'pno', 'recompute'])
@pytest.mark.parametrize('devices', [(0,), (0, 1)])
@pytest.mark.parametrize('batch_size', [1, 16])
def test_pno_batch_all_sources(kind, devices, batch_size):
    _check(kind, devices, batch_size, False)


@pytest.mark.parametrize('kind', ['pno', 'recompute'])
def test_pno_batch_truncated_space(kind):
    _check(kind, (0, 1), 16, True)


def _check(kind, devices, batch_size, truncation):
    from electrolites.dlpno.pno_batch_gpu import Executor, PNOPlan
    raw, rng = raw_data()
    raw.update({'ep'+str(p): np.linspace(.2, 1.6, raw['T'+str(p)].shape[0]) for p in range(9)})
    variables = {k: ad.Variable(v.copy()) for k, v in raw.items()}
    state = state_for(variables, truncation)
    items = (0, 1, 2, 4, 5, 8)
    inputs = list(variables.values())
    expected = oracle(variables, kind, items, truncation)
    seeds = [rng.normal(size=x.shape) for x in expected]
    gold = ad.Linearization(expected, inputs)(seeds)
    executor, plan, second = Executor(devices), None, None
    error = lambda a, b: max(float(np.max(abs(host(x)-host(y)), initial=0.)) for x, y in zip(a, b))
    try:
        cache = {}
        plan = PNOPlan(state, kind, executor, cache=cache, items=items, batch_size=batch_size)
        result = plan.bridge()
        outputs = tuple(x for p in items for x in result[p])
        alignment = 'pno' if kind != 'sc_mp2' else kind
        actual = aligned(outputs, expected, alignment)
        graph = ad.Linearization(actual, inputs)
        bars = graph(seeds)
        value_error, vjp_error = error(actual, expected), error(bars, gold)
        assert value_error < 1e-9, value_error
        assert vjp_error < 2e-8, vjp_error
        assert error(graph([.37*z for z in seeds]), [.37*x for x in gold]) < 2e-8
        selected = [variables[k] for k in ('S', 'F', 'Fo', 'M', 'q', 'X0', 'T0', 'ep0')]
        wanted = [gold[next(j for j, x in enumerate(inputs) if x is v)] for v in selected]
        assert error(ad.Linearization(actual, selected, boundary=inputs)(seeds), wanted) < 2e-8
        directions = {k: .02*rng.normal(size=v.shape) for k, v in raw.items()}
        h, targets = 2e-5, []
        for sign in (1, -1):
            shifted = {k: v+sign*h*directions[k] for k, v in raw.items()}
            values = aligned(oracle(shifted, kind, items, truncation), expected, alignment)
            targets.append(sum(float(np.sum(host(x)*z)) for x, z in zip(values, seeds)))
        numerical = (targets[0]-targets[1])/(2*h)
        analytic = sum(float(np.sum(host(bar)*directions[k])) for k, bar in zip(variables, bars))
        fd_error = abs(numerical-analytic)
        assert fd_error < 2e-6, fd_error
        second = PNOPlan(state, kind, executor, cache=cache, items=items, batch_size=batch_size)
        second_result = second.bridge()
        again = aligned(tuple(x for p in items for x in second_result[p]), expected, alignment)
        assert error(ad.Linearization(again, inputs)(seeds), gold) < 2e-8
        assert error(graph(seeds, release_graph=True, release_inputs=True), gold) < 2e-8
        assert all(x.v is None for x in inputs)
        evidence = dict(kind=kind, devices=devices, batch_size=batch_size, truncation=truncation,
                        value_error=value_error, vjp_error=vjp_error, directional_fd_error=fd_error,
                        unique_domains=len(cache), batches=plan.statistics['batches'])
    finally:
        if second is not None:
            second.release()
        if plan is not None:
            plan.release()
        executor.close()
    assert all(x['live_bytes'] == 0 for x in plan.statistics['released_batches'])
    print('BATCHED_PNO_REPORT '+json.dumps(evidence))
