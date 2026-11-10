"""Shared CCSD overlap packing: independent sources, padding and full residual."""
import json
from itertools import permutations
from types import SimpleNamespace
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import host
from test_dlpno_triples_residual_gpu import gpu,raw_sources,state_from
from test_dlpno_ccsd_complete_sources_gpu import reference


def configure(monkeypatch,devices):
    for key,value in dict(DLPNO_CCSD_SHARED_OVERLAPS='1',DLPNO_RESIDENT_REGION_DEVICES=devices,
        DLPNO_RESIDENT_SHARED_STORE='1',DLPNO_RESIDENT_ISOLATE_DEVICES='1',DLPNO_RESIDENT_INLINE_PRIMARY='1',
        DLPNO_RESIDENT_PARALLEL_SHUTDOWN='1',DLPNO_RESIDENT_REGION_EXECUTOR='process',
        DLPNO_RESIDENT_REGION_MEMORY_MIB='1024').items():monkeypatch.setenv(key,value)


@pytest.mark.parametrize('devices',['0','0,1'])
@pytest.mark.parametrize('mixed',[False,True])
def test_packed_complete_overlap_sources(monkeypatch,devices,mixed):
    from electrolites.dlpno.ccsd_shared_overlap_gpu import shared_plan,RaggedPacking
    from electrolites.dlpno.resident_regions_gpu import finish
    from electrolites.dlpno.ragged import Layout
    configure(monkeypatch,devices);cp=gpu();raw,rng=raw_sources()
    leaves={k:ad.Variable(cp.asarray(v) if mixed and k.startswith('x') else v.copy()) for k,v in raw.items()}
    s=state_from(leaves,0.);s.pao=s.t_paos;s.X=s.X_tno
    keys=list(permutations(range(4),2))+[(0,0),(2,2)];keys=[keys[i] for i in rng.permutation(len(keys))]
    layout=Layout([s.X[p].shape[1] for p,q in keys],[s.X[q].shape[1] for p,q in keys])
    gold_leaves={k:ad.Variable(v.copy()) for k,v in raw.items()};exact=state_from(gold_leaves,0.)
    def equations(state):
        outputs=[]
        for members,(pr,pc) in zip(layout.members,layout.shapes):
            bucket=ad.Variable(np.zeros((len(members),pr,pc)))
            for slot,k in enumerate(members):
                p,q=keys[k];value=state.X_tno[p].T@state.orb.S_pao[np.ix_(state.t_paos[p],state.t_paos[q])]@state.X_tno[q]
                bucket[slot,:value.shape[0],:value.shape[1]]=value
            outputs.append(bucket)
        return outputs
    expected=equations(exact);seeds=[rng.normal(size=x.shape) for x in expected]
    gold=ad.Linearization(expected,list(gold_leaves.values()))(seeds)
    error=lambda aa,bb:max(float(np.max(abs(host(a)-host(b)),initial=0.)) for a,b in zip(aa,bb))
    try:
        plan=shared_plan(s,keys,max_count=3);packing=RaggedPacking(layout,keys,plan.keys,plan.overlap_shapes,plan.overlap_offsets)
        actual=packing.buckets(packing.pack(plan.packed_bridge()));graph=ad.Linearization(actual,list(leaves.values()))
        bars=graph(seeds);value_error=error(actual,expected);vjp_error=error(bars,gold)
        assert value_error<3e-12 and vjp_error<3e-11,(value_error,vjp_error)
        assert error(graph([.37*z for z in seeds]),[.37*b for b in gold])<3e-11
        for name in ['metric','x0','x1','x2','x3']:
            selected=ad.Linearization(actual,[leaves[name]],boundary=list(leaves.values()))(seeds)
            assert error(selected,[gold[list(leaves).index(name)]])<3e-11
        pads=[np.ones(x.shape) for x in actual]
        for bucket,members in zip(pads,layout.members):
            for slot,k in enumerate(members):bucket[slot,:layout.rows[k],:layout.cols[k]]=0.
        assert max(np.max(abs(host(x)),initial=0.) for x in graph(pads))==0.
        directions={k:.02*rng.normal(size=v.shape) for k,v in raw.items()};h=1e-5;targets=[]
        for sign in (1,-1):
            values={k:ad.Variable(v+sign*h*directions[k],tracked=False) for k,v in raw.items()}
            values=equations(state_from(values,0.))
            targets.append(sum(float(np.sum(host(v)*z)) for v,z in zip(values,seeds)))
        fd_error=abs((targets[0]-targets[1])/(2*h)-sum(float(np.sum(host(v)*directions[k])) for k,v in zip(leaves,bars)))
        assert fd_error<2e-7,fd_error
        assert error(graph(seeds,release_graph=True,release_inputs=True),gold)<3e-11
        assert all(x.v is None for x in leaves.values())
        evidence=dict(devices=devices,mixed=mixed,value_error=value_error,vjp_error=vjp_error,source_fd_error=fd_error,packing=packing.statistics)
    finally:finish(s)
    assert all(x['live_bytes']==0 for x in s._resident_region_pool.statistics['released_workers'])
    print('SHARED_CCSD_PACK_REPORT '+json.dumps(evidence))


@pytest.mark.parametrize('keep_df',[False,True])
@pytest.mark.parametrize('mixed_pairs',[False,True])
def test_complete_ccsd_shared(reference,monkeypatch,keep_df,mixed_pairs):
    from electrolites.dlpno.ccsd_sources_gpu import SourcePlan
    from electrolites.dlpno.resident_regions_gpu import finish
    from test_dlpno_ccsd_complete_sources_gpu import test_complete_batched_residual_vjp_and_coefficient_fd
    states=[];original=SourcePlan.__init__
    # Oracle state preparation remains the exact scalar reference path.
    monkeypatch.setenv('DLPNO_RESIDENT_REGION_DEVICES','')
    def initialize(self,state,*args,**kw):
        states.append(state)
        with monkeypatch.context() as local:
            configure(local,'0,1');original(self,state,*args,**kw)
    monkeypatch.setattr(SourcePlan,'__init__',initialize)
    try:test_complete_batched_residual_vjp_and_coefficient_fd(reference,monkeypatch,mixed_pairs,keep_df)
    finally:
        for state in states:finish(state)
    assert states
    for state in states:
        d=state.diagnostics['shared_ccsd_overlaps']
        assert d['parent_per_overlap_records']==0 and d['complete_nonsymmetric_metric_vjp'] and d['released']
        assert all(x['live_bytes']==0 for x in state._resident_region_pool.statistics['released_workers'])


@pytest.mark.parametrize('keep_df',[False,True])
@pytest.mark.parametrize('mixed_pairs',[False,True])
def test_unmodified_ccsd_default(reference,monkeypatch,keep_df,mixed_pairs):
    from test_dlpno_ccsd_complete_sources_gpu import test_complete_batched_residual_vjp_and_coefficient_fd
    monkeypatch.setenv('DLPNO_CCSD_SHARED_OVERLAPS','0');monkeypatch.setenv('DLPNO_RESIDENT_REGION_DEVICES','')
    test_complete_batched_residual_vjp_and_coefficient_fd(reference,monkeypatch,mixed_pairs,keep_df)


def test_failed_capture_discards_only_its_stage(monkeypatch):
    from electrolites.dlpno.ccsd_shared_overlap_gpu import build,shared_plan
    from electrolites.dlpno.resident_regions_gpu import finish
    from electrolites.dlpno.ragged import Layout
    configure(monkeypatch,'0,1');raw,rng=raw_sources();s=state_from({k:ad.Variable(v.copy()) for k,v in raw.items()},0.)
    s.X=s.X_tno;s.pao=s.t_paos;keys=[(0,1),(1,0)];first=shared_plan(s,keys,max_count=1)
    first_value=first.packed_bridge();gold=first_value.v.copy()
    layout=Layout([s.X[p].shape[1] for p,q in keys],[s.X[q].shape[1] for p,q in keys])
    def fail(_):raise MemoryError('injected capture failure')
    plan=SimpleNamespace(source=s,LS=layout,_capture=fail,statistics={})
    try:
        with pytest.raises(MemoryError,match='injected'):build(plan,keys)
        assert s._resident_region_plans[-1].closed and not first.closed
        assert np.array_equal(first_value.v,gold)
        bars=ad.Linearization([first_value],list(first.inputs))([np.ones_like(gold)])
        assert all(np.isfinite(host(x)).all() for x in bars)
        info=s._resident_region_pool.run('ready',[(i,()) for i in range(2)])
        assert sum(x['regions'] for _,x in info)==len(first.assignments)
    finally:finish(s)
    assert all(x['live_bytes']==0 for x in s._resident_region_pool.statistics['released_workers'])
