"""Oversized-singleton fallback: exact sources, deferred reads and consuming replay."""
import json
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import _names,host
from electrolites.dlpno.reference_cpu import rebind
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.parallel_sources_gpu import prepare_task
from electrolites.dlpno.triples_source_batch_gpu import _layout,workspace_bytes
from electrolites.dlpno.deferred_reads import capture_reads
from electrolites.dlpno.resident_shared_store_gpu import SharedResidentPool,SharedRegionPlan
from test_dlpno_triples_gpu_sources import raw_sources
from test_dlpno_triples_source_batch_gpu import multi_state
from test_dlpno_resident_shared_store_gpu import defer_sources,Owner

@pytest.mark.parametrize('checkpoint',[False,True])
@pytest.mark.parametrize('energy_only',[False,True])
@pytest.mark.parametrize('devices,executor',[((0,),'thread'),((0,1),'process')])
def test_unpadded_single_source_full_vjp_fd_and_replay(monkeypatch,checkpoint,energy_only,devices,executor):
    monkeypatch.setenv('DLPNO_RESIDENT_TRIPLES_SINGLE_FALLBACK','1')
    monkeypatch.setenv('DLPNO_RESIDENT_CHECKPOINT_GRAPHS',str(int(checkpoint)))
    monkeypatch.setenv('DLPNO_RESIDENT_INPUT_CACHE_MIB','0')
    monkeypatch.setenv('DLPNO_ASYNC_SOURCE_REVERSE','0')
    raw,rng=raw_sources();variables={k:ad.Variable(v.copy()) for k,v in raw.items()};inputs=list(variables.values())
    equation=rebind(RefDLPNO._triplet_t0,**_names);stride=1 if energy_only else 4
    def oracle(values):
        view=multi_state(values)
        return tuple(x for t in range(4) for x in equation(view,t)[:stride])
    expected=oracle(variables);seeds=[rng.normal(size=x.shape) for x in expected]
    gold=ad.Linearization(expected,inputs)(seeds)
    state=multi_state(variables);defer_sources(state)
    with capture_reads():
        prepared=[prepare_task(state,(t,),energy_only,workspace_bytes([_layout(state,t)])-1,True) for t in range(4)]
    owner=Owner();owner._release_transposes=False;pool=SharedResidentPool(devices,budget_bytes=128<<20,executor=executor)
    try:
        plan=SharedRegionPlan(prepared,pool,owner=owner);actual=plan.bridge()
        assert all(x['triples_single_fallback']['coefficients_complete'] for x in plan.statistics['worker_graphs'])
        value_error=max(float(np.max(abs(host(x)-host(y)),initial=0.)) for x,y in zip(actual,expected));assert value_error<2e-10
        graph=ad.Linearization(actual,inputs);bars=graph(seeds)
        error=max(float(np.max(abs(x-y),initial=0.)) for x,y in zip(bars,gold));assert error<2e-9
        again=graph([.37*x for x in seeds]);assert max(np.max(abs(x-.37*y),initial=0.) for x,y in zip(again,gold))<2e-9
        names=['metric','overlap','xt','eps','fock','t0'];chosen=[variables[x] for x in names]
        selected=ad.Linearization(actual,chosen,boundary=inputs)(seeds)
        assert max(np.max(abs(x-gold[list(variables).index(name)]),initial=0.) for name,x in zip(names,selected))<2e-9
        directions={k:.05*rng.normal(size=x.shape) for k,x in raw.items()};h=2e-5;values=[]
        for sign in (1,-1):
            displaced={k:x+sign*h*directions[k] for k,x in raw.items()}
            values.append(sum(float(np.sum(host(x)*z)) for x,z in zip(oracle(displaced),seeds)))
        fd=(values[0]-values[1])/(2*h);analytic=sum(float(np.sum(x*directions[k])) for k,x in zip(variables,bars));assert abs(fd-analytic)<3e-8
        owner._release_transposes=True;last=graph(seeds,release_graph=True,release_inputs=True)
        assert max(np.max(abs(x-y),initial=0.) for x,y in zip(last,gold))<2e-9
        print('SINGLE_FALLBACK_REPORT '+json.dumps(dict(checkpoint=checkpoint,energy_only=energy_only,devices=devices,executor=executor,value_error=value_error,vjp_error=error,fd_error=abs(fd-analytic),fallback_regions=len(plan.statistics['worker_graphs']))))
    finally:cleanup=pool.close()
    assert all(x['live_bytes']==x['reserved_bytes']==0 for x in cleanup)


def test_default_and_true_single_bound_rejection_cleanup(monkeypatch):
    raw,_=raw_sources();variables={k:ad.Variable(v.copy()) for k,v in raw.items()};state=multi_state(variables)
    for flag,limit,match in [('0',workspace_bytes([_layout(state,0)])-1,'Shared resident triples workspace'),('1',1,'Unpadded shared triples source exceeds')]:
        monkeypatch.setenv('DLPNO_RESIDENT_TRIPLES_SINGLE_FALLBACK',flag)
        pool=SharedResidentPool((0,),budget_bytes=64<<20)
        try:
            with pytest.raises(MemoryError,match=match):
                SharedRegionPlan([prepare_task(state,(0,),False,limit,True)],pool)
        finally:cleanup=pool.close()
        assert all(x['live_bytes']==x['reserved_bytes']==0 for x in cleanup)
