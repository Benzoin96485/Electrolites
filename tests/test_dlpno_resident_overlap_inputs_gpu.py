"""All coefficient sources, permutations and device shards against the CPU equations."""
import json
from itertools import permutations
from types import MethodType

import numpy as np
import pytest

from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import host
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.reference_cpu import rebind
from test_dlpno_triples_residual_gpu import gpu, raw_sources, state_from



class DeferredMetric:
    def __init__(self,value):
        self.value=value
    def __getitem__(self,key):
        from electrolites.dlpno.deferred_reads import active,DeferredRead
        if not active():return self.value[key]
        shape=np.broadcast_shapes(*(x.shape for x in key))
        return DeferredRead((self.value,),((key,(slice(None),slice(None))),),shape,shape)

@pytest.mark.parametrize('mixed', [False, True])
@pytest.mark.parametrize('devices', [(0,), (0, 1)])
@pytest.mark.parametrize('cut', [0., .035, 2.])
def test_resident_overlap_sources(devices, cut, mixed, monkeypatch):
    monkeypatch.setenv("DLPNO_TRIPLES_RESIDENT_OVERLAPS", "1")
    cp = gpu()
    if len(devices) > cp.cuda.runtime.getDeviceCount():
        pytest.skip('Two allocated GPUs required')
    from electrolites.dlpno.triples_resident_gpu import ResidentTriples, ResidentTranspose
    raw, rng = raw_sources()
    leaves = {name: ad.Variable(cp.asarray(x) if mixed and (name.startswith('x') or name=='metric') else x.copy()) for name, x in raw.items()}
    exact_leaves = {name: ad.Variable(x.copy()) for name, x in raw.items()}
    state, exact = state_from(leaves, cut), state_from(exact_leaves, cut)
    state.orb.S_pao = DeferredMetric(leaves["metric"])
    connections = RefDLPNO._triples_couplings(state)
    connections[0] = list(connections[0])+[(p, 2, 0, 2) for p in permutations((0, 1, 2))]
    connections[0] += [((1, 0, 0), 0, 0, 1)]
    evaluate = rebind(RefDLPNO._triplet_residual, np=ad.Numpy(np))
    expected = [evaluate(exact, t, c) for t, c in enumerate(connections)]
    # Small workspace forces many batches, with ragged shapes and aliases.
    plan = ResidentTriples(state, devices, connections=connections, workspace_bytes=1 << 20)
    try:
        for live, initial in zip(plan.statistics['resident_bytes_by_device'], plan.statistics['initialized']):
            assert initial['resident_bytes'] > 0
            assert initial['resident_bytes'] <= live <= plan.statistics['memory_bound_bytes']
        assert plan.statistics["resident_overlap_sources"]
        assert not plan.statistics["overlap_boundary"]["parent_overlap_values_constructed"]
        outputs = plan.bridge()
        seeds = [rng.normal(size=x.shape) for x in outputs]
        graph = ad.Linearization(outputs, list(leaves.values()))
        bars = graph(seeds)
        exact_bars = ad.Linearization(expected, list(exact_leaves.values()))(seeds)
        error = lambda a, b: max(float(np.max(abs(host(x)-host(y)), initial=0.)) for x, y in zip(a, b))
        value_error, vjp_error = error(outputs, expected), error(bars, exact_bars)
        assert value_error < 3e-12
        assert vjp_error < 3e-10
        repeated_error = error(graph([.37*z for z in seeds]), [.37*b for b in bars])
        assert repeated_error < 3e-10
        partial_errors = []
        names, inputs = list(leaves), list(leaves.values())
        for group in ('t', 'w', 'e', 'x', 'fock', 'metric'):
            indices = [i for i, name in enumerate(names) if name == group or name.startswith(group)]
            actual = ad.Linearization(outputs, [inputs[i] for i in indices], boundary=inputs)(seeds)
            partial_errors.append(error(actual, [exact_bars[i] for i in indices]))
        assert max(partial_errors) < 3e-10
        operator = ResidentTranspose(plan)
        seed = cp.asarray(np.concatenate([x.ravel() for x in seeds]))
        transpose = operator([seed])[0]
        expected_t = np.concatenate([exact_bars[names.index('t'+str(t))].ravel() for t in range(4)])
        transpose_error = float(np.max(abs(cp.asnumpy(transpose)-expected_t)))
        assert transpose_error < 3e-11
        operator.release()
        del transpose
        assert not plan.released
        direction = {name: .1*rng.normal(size=x.shape) for name, x in raw.items()}
        h, targets = 2e-5, []
        for sign in (1, -1):
            shifted = {name: x+sign*h*direction[name] for name, x in raw.items()}
            numeric = state_from(shifted, cut)
            numeric._matmul_3d = MethodType(RefDLPNO._matmul_3d, numeric)
            targets.append(sum(float(np.sum(evaluate(numeric, t, c)*z))
                               for t, (c, z) in enumerate(zip(connections, seeds))))
        finite = (targets[0]-targets[1])/(2*h)
        analytic = sum(float(np.sum(host(b)*direction[name])) for name, b in zip(names, bars))
        fd_error = abs(finite-analytic)
        assert fd_error < 2e-8
        state.T3 = []
        consuming_error = error(graph(seeds, release_graph=True, release_inputs=True), bars)
        assert consuming_error < 3e-10
        assert all(x.v is None for x in inputs)
        evidence = dict(devices=list(devices), cut=cut, mixed=mixed, maximum_value_error=value_error,
            maximum_vjp_error=vjp_error, maximum_selected_error=max(partial_errors),
            repeated_vjp_error=repeated_error, transpose_error=transpose_error,
            directional_fd_error=fd_error, consuming_error=consuming_error)
    finally:
        plan.release()
    assert all(x['live_bytes'] == 0 for x in plan.statistics['released_devices'])
    evidence['released_devices'] = plan.statistics['released_devices']
    print('RESIDENT_OVERLAP_REPORT '+json.dumps(evidence))


def test_resident_overlap_bounds(monkeypatch):
    monkeypatch.setenv("DLPNO_TRIPLES_RESIDENT_OVERLAPS", "1")
    gpu()
    from electrolites.dlpno.triples_resident_gpu import ResidentTriples
    raw, _ = raw_sources()
    state = state_from({name: ad.Variable(x) for name, x in raw.items()}, 0.)
    rows = RefDLPNO._triples_couplings(state)
    with pytest.raises(ValueError, match='workspace'):
        ResidentTriples(state, (0,), connections=rows, workspace_bytes=1)
    with pytest.raises(ValueError, match='memory'):
        ResidentTriples(state, (0,), connections=rows, memory_bytes=1)
