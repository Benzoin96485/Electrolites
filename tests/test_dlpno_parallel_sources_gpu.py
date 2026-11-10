"""Exact two-device source derivatives, shared operands and failure recovery."""
import copy
import json
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import _names, host
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.reference_cpu import rebind
from test_dlpno_triples_source_batch_gpu import multi_state
from test_dlpno_triples_gpu_sources import gpu, raw_sources


@pytest.mark.parametrize('executor,devices,energy_only', [
    ('thread', (0,), False), ('thread', (0,1), False),
    ('thread', (0,1), True), ('process', (0,1), False)])
def test_parallel_sources_match_independent_equations_and_fd(executor, devices, energy_only):
    cp = gpu()
    assert cp.cuda.runtime.getDeviceCount() >= 2, 'This test requires two allocated GPUs'
    from electrolites.dlpno.parallel_sources_gpu import SourcePool, source_window
    raw, rng = raw_sources()
    variables = {k: ad.Variable(v.copy()) for k,v in raw.items()}
    oracle = rebind(RefDLPNO._triplet_t0, **_names)
    reference = multi_state(variables)
    count = 1 if energy_only else 4
    expected = tuple(value for t in range(4) for value in oracle(reference,t)[:count])
    seeds = [rng.normal(size=x.shape) for x in expected]
    inputs = list(variables.values())
    gold = ad.Linearization(expected, inputs)(seeds)
    pool = SourcePool(devices, executor)
    try:
        actual, info = source_window(multi_state(variables), ((0,1),(2,3)), devices=devices,
                                     energy_only=energy_only, pool=pool)
        assert all(isinstance(x.v,np.ndarray) for x in actual)
        graph = ad.Linearization(actual, inputs)
        bars = graph(seeds)
        error = max(float(np.max(abs(a-b),initial=0.)) for a,b in zip(bars,gold))
        value_error = max(float(np.max(abs(host(a)-host(b)),initial=0.)) for a,b in zip(actual,expected))
        assert error < 3e-10 and value_error < 2e-11
        repeated = graph([.37*x for x in seeds])
        assert max(np.max(abs(a-.37*b),initial=0.) for a,b in zip(repeated,gold)) < 3e-10
        for names in [('metric',), ('overlap','xt','eps','fock'), ('t0','s0'), ('qo','qv','qvv')]:
            chosen = [variables[name] for name in names]
            selected = ad.Linearization(actual, chosen, boundary=inputs)(seeds)
            assert max(np.max(abs(bar-gold[list(variables).index(name)]),initial=0.) for name,bar in zip(names,selected)) < 3e-10
        directions = {k:.1*rng.normal(size=v.shape) for k,v in raw.items()}
        energies=[];step=2e-5
        for sign in (1,-1):
            state=multi_state({k:v+sign*step*directions[k] for k,v in raw.items()})
            outputs=tuple(value for t in range(4) for value in oracle(state,t)[:count])
            energies.append(sum(float(np.sum(host(x)*z)) for x,z in zip(outputs,seeds)))
        numerical=(energies[0]-energies[1])/(2*step)
        analytic=sum(float(np.sum(bar*directions[k])) for k,bar in zip(variables,bars))
        assert abs(numerical-analytic)<2e-8
        consumed=graph(seeds,release_graph=True,release_inputs=True)
        assert max(np.max(abs(a-b),initial=0.) for a,b in zip(consumed,gold))<3e-10
        assert all(x.v is None for x in inputs)
        assert set(map(int,info['device_task_counts']))==set(devices)
        assert info['largest_live_worker_device_bytes']==0
        print('PARALLEL_SOURCE_REPORT '+json.dumps(dict(executor=executor,devices=devices,
            energy_only=energy_only,value_error=value_error,vjp_error=error,
            directional_fd_error=abs(numerical-analytic),evidence=info)))
    finally:
        cleanup=pool.close()
    assert len(cleanup)==len(devices) and all(x['live_device_bytes']==0 for x in cleanup)


@pytest.mark.parametrize('executor',['thread','process'])
def test_worker_failure_does_not_poison_next_replay(executor):
    assert gpu().cuda.runtime.getDeviceCount()>=2
    from electrolites.dlpno.parallel_sources_gpu import SourcePool, prepare_task
    raw,_=raw_sources();state=multi_state({k:ad.Variable(v) for k,v in raw.items()})
    spec,inputs=prepare_task(state,(0,1),False,512<<20,True)
    values=tuple(ad.primal(x) for x in inputs);needed=[False]*len(values)
    bad=copy.deepcopy(spec);bad['tensors']['S_pao']['reads'].clear()
    pool=SourcePool((0,1),executor)
    try:
        with pytest.raises(KeyError):pool.run([(bad,values,None,needed),(spec,values,None,needed)])
        rows=pool.run([(spec,values,None,needed)]*2)
        assert all(info['live_worker_device_bytes']==0 for _,info in rows)
        np.testing.assert_allclose(rows[0][0][0],rows[1][0][0],atol=1e-13,rtol=1e-13)
    finally:
        cleanup=pool.close()
    assert len(cleanup)==2 and all(x['live_device_bytes']==0 for x in cleanup)
