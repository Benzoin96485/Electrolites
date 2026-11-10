"""Batched TNO coefficients, shared PAO bases and degenerate-boundary tests."""
import json
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import host
from test_dlpno_parallel_local_gpu import raw_data,make_state,oracle,aligned


@pytest.mark.parametrize('devices',[(0,),(0,1)])
@pytest.mark.parametrize('gpu_output',[False,True])
@pytest.mark.parametrize('batch_size',[1,16])
def test_batched_tno_all_sources(devices,gpu_output,batch_size):
    import cupy as cp
    from electrolites.dlpno.tno_batch_gpu import Executor,TNOPlan
    assert cp.cuda.runtime.getDeviceCount()>=len(devices)
    raw,rng=raw_data();variables={k:ad.Variable(v.copy()) for k,v in raw.items()}
    expected=oracle(variables,'tno',(0,1,2,3));inputs=list(variables.values())
    seeds=[rng.normal(size=x.shape) for x in expected]
    gold=ad.Linearization(expected,inputs)(seeds)
    state,_=make_state(variables);executor=Executor(devices);cache={};plan=None;second=None
    try:
        plan=TNOPlan(state,.008,executor,cache=cache,batch_size=batch_size)
        x,e=plan.bridge(gpu_output=gpu_output);outputs=tuple(y for pair in zip(x,e) for y in pair)
        actual=aligned(outputs,expected,'tno')
        gpu_seeds=[cp.asarray(z) if gpu_output else z for z in seeds]
        graph=ad.Linearization(actual,inputs);bars=graph(gpu_seeds)
        error=lambda a,b:max(float(np.max(abs(host(x)-host(y)),initial=0.)) for x,y in zip(a,b))
        value_error,vjp_error=error(actual,expected),error(bars,gold)
        assert value_error<2e-9,value_error
        assert vjp_error<2e-8,vjp_error
        assert error(graph([.37*z for z in gpu_seeds]),[.37*b for b in gold])<2e-8
        selected=[variables[k] for k in ('S','F','X0','T0')]
        wanted=[gold[next(i for i,x in enumerate(inputs) if x is item)] for item in selected]
        assert error(ad.Linearization(actual,selected,boundary=inputs)(gpu_seeds),wanted)<2e-8
        directions={k:.02*rng.normal(size=v.shape) for k,v in raw.items()};h=2e-5;targets=[]
        for sign in (1,-1):
            shifted={k:v+sign*h*directions[k] for k,v in raw.items()}
            values=aligned(oracle(shifted,'tno',(0,1,2,3)),expected,'tno')
            targets.append(sum(float(np.sum(host(v)*z)) for v,z in zip(values,seeds)))
        numerical=(targets[0]-targets[1])/(2*h)
        analytic=sum(float(np.sum(host(bar)*directions[k])) for k,bar in zip(variables,bars))
        fd_error=abs(analytic-numerical);assert fd_error<2e-6
        # Reusing the domain bases must preserve every source path.
        second=TNOPlan(state,.008,executor,cache=cache,batch_size=batch_size)
        sx,se=second.bridge();second_outputs=aligned(tuple(y for p in zip(sx,se) for y in p),expected,'tno')
        assert error(ad.Linearization(second_outputs,inputs)(seeds),gold)<2e-8
        assert len(cache)==3
        assert error(graph(gpu_seeds,release_graph=True,release_inputs=True),gold)<2e-8
        assert all(x.v is None for x in inputs)
        evidence=dict(devices=list(devices),gpu_output=gpu_output,batch_size=batch_size,
            value_error=value_error,vjp_error=vjp_error,directional_fd_error=fd_error,unique_domains=len(cache))
    finally:
        if second is not None:second.release()
        if plan is not None:plan.release()
        executor.close()
    assert all(x['live_bytes']==0 for x in plan.statistics['released_batches'])
    print('BATCHED_TNO_REPORT '+json.dumps(evidence))


def test_batched_spectrum_value_and_vjp():
    import cupy as cp
    from electrolites.dlpno.tno_batch_gpu import batched_eigh
    rng=np.random.default_rng(413);raw=rng.normal(size=(3,6,6));raw=raw+raw.transpose(0,2,1)
    a=ad.Variable(cp.asarray(raw));w,u,_=batched_eigh(a)
    z=rng.normal(size=(3,6));gold=[]
    for value,seed in zip(raw,z):
        leaf=ad.Variable(value);ew,ev=ad.eigh(leaf)
        gold.append(ad.Linearization([ew],[leaf])([seed])[0])
    got=ad.Linearization([w],[a])([cp.asarray(z)])[0]
    assert np.max(abs(cp.asnumpy(got)-gold))<1e-12
