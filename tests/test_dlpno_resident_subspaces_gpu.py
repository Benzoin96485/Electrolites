"""Shared-worker subspaces against independent scalar values and all sources."""
import json
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import host
from test_dlpno_parallel_local_gpu import raw_data,make_state,oracle as tno_oracle,aligned
from test_dlpno_pno_batch_gpu import state_for,oracle as pno_oracle


class Owner:
    pass


@pytest.mark.parametrize('kind,devices,executor,batch,truncated,gpu',[
    ('sc_mp2','0','thread',1,False,False),
    ('pno','0','thread',16,False,False),
    ('recompute','0','thread',16,False,False),
    ('sc_mp2','0,1','process',16,False,False),
    ('pno','0,1','process',16,True,False),
    ('recompute','0,1','process',1,True,False),
    ('tno','0','thread',1,False,False),
    ('tno','0,1','process',16,False,True),
])
def test_shared_subspace_complete_sources(monkeypatch,kind,devices,executor,batch,truncated,gpu):
    import cupy as cp
    from electrolites.dlpno.resident_subspaces_gpu import pno_stage,tno_stage
    from electrolites.dlpno.resident_regions_gpu import finish
    for name,value in dict(DLPNO_RESIDENT_SHARED_STORE='1',DLPNO_RESIDENT_REGION_DEVICES=devices,
        DLPNO_RESIDENT_REGION_EXECUTOR=executor,DLPNO_RESIDENT_REGION_MEMORY_MIB='512',
        DLPNO_RESIDENT_PARALLEL_SHUTDOWN='1',DLPNO_PNO_BATCH_SIZE=str(batch),DLPNO_TNO_BATCH_SIZE=str(batch),
        DLPNO_TNO_OUTPUT_GPU=str(int(gpu))).items():monkeypatch.setenv(name,value)
    raw,rng=raw_data()
    raw.update({'ep'+str(p):np.linspace(.2,1.6,raw['T'+str(p)].shape[0]) for p in range(9)})
    variables={k:ad.Variable(v.copy()) for k,v in raw.items()};inputs=list(variables.values())
    items=(0,1,2,4,5,8);alignment='tno' if kind=='tno' else ('sc_mp2' if kind=='sc_mp2' else 'pno')
    def oracle(values):return tno_oracle(values,'tno',(0,1,2,3)) if kind=='tno' else pno_oracle(values,kind,items,truncated)
    original=make_state(variables)[0] if kind=='tno' else state_for(variables,truncated)
    state=Owner();state.__dict__.update(original.__dict__);state.diagnostics={};state._release_transposes=False
    expected=oracle(variables);seeds=[rng.normal(size=x.shape) for x in expected]
    gold=ad.Linearization(expected,inputs)(seeds)
    error=lambda a,b:max(float(np.max(abs(host(x)-host(y)),initial=0.)) for x,y in zip(a,b))
    def stage():
        if kind=='tno':
            x,e=tno_stage(state,.008);return tuple(y for pair in zip(x,e) for y in pair)
        result=pno_stage(state,kind,items=items);return tuple(x for p in items for x in result[p])
    try:
        actual=aligned(stage(),expected,alignment)
        if gpu:assert all(isinstance(ad.primal(x),cp.ndarray) for x in actual)
        z=[cp.asarray(x) if gpu else x for x in seeds]
        graph=ad.Linearization(actual,inputs);bars=graph(z)
        value_error,vjp_error=error(actual,expected),error(bars,gold)
        assert value_error<2e-9,value_error
        assert vjp_error<2e-8,vjp_error
        assert error(graph([.37*x for x in z]),[.37*x for x in gold])<2e-8
        selected=[variables[k] for k in ('S','F','Fo','M','q','X0','T0','ep0')]
        wanted=[gold[next(i for i,x in enumerate(inputs) if x is v)] for v in selected]
        assert error(ad.Linearization(actual,selected,boundary=inputs)(z),wanted)<2e-8
        directions={k:.02*rng.normal(size=v.shape) for k,v in raw.items()};h=2e-5;targets=[]
        for sign in (1,-1):
            values=aligned(oracle({k:v+sign*h*directions[k] for k,v in raw.items()}),expected,alignment)
            targets.append(sum(float(np.sum(host(x)*seed)) for x,seed in zip(values,seeds)))
        finite_difference=(targets[0]-targets[1])/(2*h)
        analytic=sum(float(np.sum(host(b)*directions[k])) for k,b in zip(variables,bars))
        fd_error=abs(finite_difference-analytic);assert fd_error<2e-6,fd_error
        # A second stage shares canonical inputs but owns separate spectra.
        second=aligned(stage(),expected,alignment)
        assert error(ad.Linearization(second,inputs)(z),gold)<2e-8
        expected_domains=(3 if kind=='tno' else 0 if kind=='recompute' else len({tuple(map(int,state.pao[p])) for p in items}))
        assert len(state._tno_basis_cache)==expected_domains
        state._release_transposes=True
        assert error(graph(z,release_graph=True,release_inputs=True),gold)<2e-8
        assert all(x.v is None for x in inputs)
        assert state._resident_region_plans[0].closed
        evidence=dict(kind=kind,devices=devices,executor=executor,batch=batch,truncated=truncated,gpu_output=gpu,
            value_error=value_error,vjp_error=vjp_error,directional_fd_error=fd_error,
            all_source_responses=True,consuming_reverse=True,shared_basis_domains=len(state._tno_basis_cache))
    finally:finish(state)
    pool=state._resident_region_pool
    assert pool.closed and all(x['live_bytes']==x['reserved_bytes']==x['regions']==0 for x in pool.statistics['released_workers'])
    assert all(p.statistics['released'] for p in state._resident_region_plans)
    print('SHARED_SUBSPACE_REPORT '+json.dumps(evidence))
