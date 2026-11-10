"""One visible parent GPU, explicit remote selectors and complete source VJPs."""
import os,json
import numpy as np
import pytest


@pytest.mark.parametrize('kind',['sc_mp2','pno','recompute','tno'])
def test_inline_subspace_all_sources(kind,monkeypatch):
    from test_dlpno_resident_subspaces_gpu import test_shared_subspace_complete_sources
    monkeypatch.setenv('DLPNO_RESIDENT_INLINE_PRIMARY','1')
    monkeypatch.setenv('DLPNO_RESIDENT_ISOLATE_DEVICES','1')
    test_shared_subspace_complete_sources(monkeypatch,kind,'0,1','process',16,kind in ('pno','recompute'),kind=='tno')


@pytest.mark.parametrize('cut',[0.,.035])
def test_inline_t1_all_sources(cut,monkeypatch):
    from test_dlpno_shared_t1_overlaps_gpu import test_shared_complete_sources
    monkeypatch.setenv('DLPNO_RESIDENT_INLINE_PRIMARY','1')
    monkeypatch.setenv('DLPNO_RESIDENT_ISOLATE_DEVICES','1')
    test_shared_complete_sources((0,),(0,1),'process',True,cut,monkeypatch)


def test_inline_multiple_stores_and_restore(monkeypatch):
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.parallel_local_gpu import Operand
    from electrolites.dlpno.resident_shared_store_gpu import SharedResidentPool,SharedRegionPlan,_local,close_pools
    monkeypatch.setenv('DLPNO_RESIDENT_INLINE_PRIMARY','1');monkeypatch.setenv('DLPNO_RESIDENT_ISOLATE_DEVICES','1')
    before=dict(_local.__dict__);pools=[];graphs=[]
    try:
        for scale in [1.,2.3]:
            pool=SharedResidentPool((0,),executor='process',budget_bytes=64<<20);pools.append(pool)
            inputs=[ad.Variable(scale*np.array([[1.,.2],[.3,2.]])),ad.Variable(np.array([[.1,.7],[.8,.2]])),ad.Variable(np.array([[.3,.5],[.4,.6]]))]
            spec=dict(kind='t1_overlap_sources',groups=[dict(metric=Operand(0),left=Operand(1),right=(Operand(2),),positions=(np.arange(2),))])
            plan=SharedRegionPlan([(spec,tuple(inputs))],pool);output=plan.packed_bridge()
            reference=(inputs[1].T@inputs[0]@inputs[2]).ravel();seed=np.array([.2,.3,.7,.8])
            gold=ad.Linearization([reference],inputs)([seed]);graph=ad.Linearization([output],inputs)
            assert np.max(abs(output.v-reference.v))<1e-12
            assert max(np.max(abs(a-b)) for a,b in zip(graph([seed]),gold))<1e-12
            graphs.append((graph,seed,gold));assert _local.__dict__==before
        pools[1].close()
        graph,seed,gold=graphs[0]
        assert max(np.max(abs(a-b)) for a,b in zip(graph([seed]),gold))<1e-12
        assert _local.__dict__==before
    finally:
        for pool in pools:pool.close()
        close_pools()
    assert _local.__dict__==before
    assert all(x['live_bytes']==0 for p in pools for x in p.statistics['released_workers'])


def test_inline_lifecycle_mapping(monkeypatch):
    import cupy as cp
    from electrolites.dlpno.resident_shared_store_gpu import SharedResidentPool,close_pools
    monkeypatch.setenv('DLPNO_RESIDENT_INLINE_PRIMARY','1');monkeypatch.setenv('DLPNO_RESIDENT_ISOLATE_DEVICES','1')
    monkeypatch.setenv('DLPNO_RESIDENT_PARALLEL_SHUTDOWN','1')
    assert cp.cuda.runtime.getDeviceCount()==1
    selectors=os.environ['DLPNO_RESIDENT_DEVICE_SELECTORS'].split(',');assert len(selectors)>=2
    pool=SharedResidentPool((0,1),executor='process',budget_bytes=64<<20)
    try:
        pool.prestart();result=pool.run('ready',[(0,()),(1,())]);first,second=[x[1] for x in result]
        assert first['pid']==os.getpid() and second['pid']!=os.getpid()
        assert first['device']==0 and second['device']==1
        assert first['local_device']==second['local_device']==0
        assert first['cuda_visible_devices']==selectors[0] and second['cuda_visible_devices']==selectors[1]
        pool.close();assert pool.statistics['inline_primary'] and pool.statistics['isolated_process_devices']
        with pytest.raises(ProcessLookupError):os.kill(second['pid'],0)
        os.kill(os.getpid(),0)
        assert all(x['live_bytes']==x['reserved_bytes']==0 for x in pool.statistics['released_workers'])
        print('INLINE_LIFECYCLE_REPORT '+json.dumps(pool.statistics))
    finally:close_pools()
