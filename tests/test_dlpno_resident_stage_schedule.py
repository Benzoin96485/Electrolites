"""Stage scheduling preserves output order, full sources and process ownership."""
import json
import os
import numpy as np
import pytest


def test_stage_assignment_bounds_and_order():
    from electrolites.dlpno.resident_shared_store_gpu import stage_assignment
    ranks,loads=stage_assignment([1,1,1,1,8],(0,1),2)
    assert ranks==[1,1,1,1,0] and loads==[8,4]
    assert stage_assignment([],(0,1),2)==([],[0,0])
    assert stage_assignment([1,8,2],(2,),4)==([2,2,2],[0,0,11,0])
    for costs,allowed,count in [([-1],(0,),1),([np.nan],(0,),1),([1],(),1),([1],(1,),1)]:
        with pytest.raises(ValueError):stage_assignment(costs,allowed,count)


@pytest.mark.parametrize('kind',['pno','triples','triples_energy'])
def test_complete_sources_with_stage_schedule(kind,monkeypatch):
    from electrolites.dlpno import resident_shared_store_gpu as shared
    from test_dlpno_resident_shared_store_gpu import test_complete_resident_regions
    monkeypatch.setenv('DLPNO_RESIDENT_STAGE_BALANCE','1')
    monkeypatch.setenv('DLPNO_RESIDENT_INLINE_PRIMARY','1')
    monkeypatch.setenv('DLPNO_RESIDENT_ISOLATE_DEVICES','1')
    monkeypatch.setenv('DLPNO_RESIDENT_PARALLEL_SHUTDOWN','1')
    original=shared.SharedResidentPool.__init__;pools=[]
    def initialize(pool,*args,**kwargs):
        original(pool,*args,**kwargs)
        # Previous completed stages must not bias a new barrier.
        pool.loads[:]=[10**15]+[0]*(len(pool.devices)-1)
        pools.append(pool)
    monkeypatch.setattr(shared.SharedResidentPool,'__init__',initialize)
    test_complete_resident_regions(kind,(0,1),'process',True)
    assert pools and all(p.statistics['released'] for p in pools)
    for pool in pools:
        assert all(x['live_bytes']==x['reserved_bytes']==0 for x in pool.statistics['released_workers'])
        for worker in pool.statistics['released_workers']:
            if worker['pid']!=os.getpid():
                with pytest.raises(ProcessLookupError):os.kill(worker['pid'],0)


def test_large_cost_first_does_not_reorder_output_or_cotangents(monkeypatch):
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.parallel_local_gpu import Operand
    from electrolites.dlpno.resident_shared_store_gpu import SharedResidentPool,SharedRegionPlan
    monkeypatch.setenv('DLPNO_RESIDENT_STAGE_BALANCE','1')
    rng=np.random.default_rng(651)
    leaves=[ad.Variable(rng.normal(size=(4,4))) for _ in range(7)]
    prepared=[];expected=[];costs=[1,1,1,1,8]
    for j,cost in enumerate(costs):
        spec=dict(kind='t1_overlap_sources',resident_cost=cost,groups=[dict(metric=Operand(0),left=Operand(1),right=(Operand(2),),positions=(np.arange(4),))])
        prepared.append((spec,(leaves[j],leaves[5],leaves[6])))
        expected.append(leaves[5].T@leaves[j]@leaves[6])
    seed=[rng.normal(size=x.shape) for x in expected]
    gold=ad.Linearization(expected,leaves)(seed)
    pool=SharedResidentPool((0,1),executor='process',budget_bytes=128<<20)
    try:
        plan=SharedRegionPlan(prepared,pool);actual=plan.bridge()
        assert plan.statistics['task_ranks']==[1,1,1,1,0]
        assert plan.statistics['stage_loads']==[8,4]
        assert max(np.max(abs(x.v-y.v)) for x,y in zip(actual,expected))<2e-12
        bars=ad.Linearization(actual,leaves)(seed)
        error=max(np.max(abs(x-y)) for x,y in zip(bars,gold));assert error<3e-12
        print('STAGE_ORDER_REPORT '+json.dumps(dict(value_error=float(max(np.max(abs(x.v-y.v)) for x,y in zip(actual,expected))),vjp_error=float(error),statistics=plan.statistics)))
    finally:cleanup=pool.close()
    assert all(x['live_bytes']==x['reserved_bytes']==0 for x in cleanup)


def test_recompute_can_use_already_running_devices(monkeypatch):
    from electrolites.dlpno import resident_shared_store_gpu as shared
    from test_dlpno_resident_subspaces_gpu import test_shared_subspace_complete_sources
    monkeypatch.setenv('DLPNO_RESIDENT_STAGE_BALANCE','1')
    monkeypatch.setenv('DLPNO_RESIDENT_PRIMARY_PNO','1')
    monkeypatch.setenv('DLPNO_RESIDENT_PARALLEL_RECOMPUTE','1')
    original=shared.SharedRegionPlan.__init__;plans=[]
    def initialize(plan,*args,**kwargs):
        original(plan,*args,**kwargs);plans.append(plan)
    monkeypatch.setattr(shared.SharedRegionPlan,'__init__',initialize)
    test_shared_subspace_complete_sources(monkeypatch,'recompute','0,1','process',1,True,False)
    assert plans and all(p.statistics['scheduling_ranks']==[0,1] for p in plans)
    assert all(set(p.statistics['task_ranks'])=={0,1} for p in plans)
