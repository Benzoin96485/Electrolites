"""Complete-source routing and bounded gradient-scoped process cleanup."""
import json,os,threading,time
import numpy as np
import pytest


@pytest.mark.parametrize('kind',['sc_mp2','pno','recompute'])
def test_primary_pno_complete_sources(kind,monkeypatch):
    from electrolites.dlpno import resident_shared_store_gpu as shared
    from test_dlpno_resident_subspaces_gpu import test_shared_subspace_complete_sources
    monkeypatch.setenv('DLPNO_RESIDENT_INLINE_PRIMARY','1')
    monkeypatch.setenv('DLPNO_RESIDENT_ISOLATE_DEVICES','1')
    monkeypatch.setenv('DLPNO_RESIDENT_PRIMARY_PNO','1')
    monkeypatch.setenv('DLPNO_RESIDENT_WARMUP','1')
    original=shared.SharedResidentPool.__init__;started=[]
    def initialize(pool,*args,**kwargs):
        original(pool,*args,**kwargs);started.append(pool);pool.prestart()
    monkeypatch.setattr(shared.SharedResidentPool,'__init__',initialize)
    try:
        test_shared_subspace_complete_sources(monkeypatch,kind,'0,1','process',16,kind!='sc_mp2',False)
        assert started and all(p.statistics['remote_warmup'] and p.statistics['released'] for p in started)
        print('PRIMARY_PNO_REPORT '+json.dumps(dict(kind=kind,lifecycle=[p.statistics for p in started])))
    finally:shared.close_pools()


def build_overlap(pool,ranks=None):
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.parallel_local_gpu import Operand
    from electrolites.dlpno.resident_shared_store_gpu import SharedRegionPlan
    rng=np.random.default_rng(626)
    inputs=[ad.Variable(rng.normal(size=shape)) for shape in [(5,5),(5,3),(5,2)]]
    spec=dict(kind='t1_overlap_sources',groups=[dict(metric=Operand(0),left=Operand(1),right=(Operand(2),),positions=(np.arange(5),))])
    plan=SharedRegionPlan([(spec,tuple(inputs)),(spec,tuple(inputs))],pool,ranks=ranks,balance_stage=True)
    output=plan.packed_bridge();oracle=ad.Numpy().concatenate([(inputs[1].T@inputs[0]@inputs[2]).ravel()]*2)
    seed=rng.normal(size=output.shape)
    gold=ad.Linearization([oracle],inputs)([seed]);bars=ad.Linearization([output],inputs)([seed])
    assert np.max(abs(output.v-oracle.v))<2e-12
    assert max(np.max(abs(a-b)) for a,b in zip(bars,gold))<2e-12
    return plan


@pytest.mark.parametrize('failure',[False,True])
def test_gradient_joins_background_exit_on_every_path(monkeypatch,failure):
    from electrolites.dlpno.local import LocalGradients
    from electrolites.dlpno.resident_lifecycle import register,owns
    from electrolites.dlpno.resident_shared_store_gpu import SharedResidentPool,close_pools
    for key in ['DLPNO_RESIDENT_INLINE_PRIMARY','DLPNO_RESIDENT_ISOLATE_DEVICES','DLPNO_RESIDENT_PARALLEL_SHUTDOWN','DLPNO_RESIDENT_OVERLAP_LIFECYCLE','DLPNO_RESIDENT_WARMUP']:
        monkeypatch.setenv(key,'1')
    entered=threading.Event();release=threading.Event();state={}
    gradient=LocalGradients(None)
    def body(atmlst=None):
        pool=SharedResidentPool((0,1),executor='process',budget_bytes=256<<20)
        state['pool']=pool;register(pool);assert owns(pool);pool.prestart()
        plan=build_overlap(pool);assert plan.statistics['scheduling_ranks']==[0,1]
        mapping=pool.run('ready',[(0,()),(1,())]);state['pid']=mapping[1][1]['pid']
        shutdown=pool.workers[1].shutdown
        def controlled_shutdown(*args,**kwargs):
            entered.set()
            if not release.wait(8):raise RuntimeError('Test shutdown release was not signaled')
            return shutdown(*args,**kwargs)
        monkeypatch.setattr(pool.workers[1],'shutdown',controlled_shutdown)
        try:
            evidence=pool.begin_close();assert entered.wait(5)
            assert pool.closed and not pool.statistics['released']
            assert all(x['live_bytes']==x['reserved_bytes']==x['regions']==0 for x in evidence)
            assert not pool._closing_future.done()
            with pytest.raises(RuntimeError,match='Closed shared pool'):pool.run('ready',[(0,())])
            # Computation following the density sweep can proceed while exit
            # is pending. The public kernel must still join before returning.
            result=np.dot(np.arange(17,dtype=float),np.arange(17,dtype=float))
            state['work_while_exit_pending']=result
            if failure:raise ValueError('Injected post-adjoint response failure')
            return np.array([[result,0.,0.]])
        finally:release.set()
    monkeypatch.setattr(gradient,'_kernel',body)
    try:
        if failure:
            with pytest.raises(ValueError,match='Injected post-adjoint'):gradient.kernel()
        else:assert gradient.kernel()[0,0]==1496.
        pool=state['pool'];assert pool.statistics['released']
        assert pool.statistics['background_shutdown'] and pool._closing_future is None
        assert 'resident_shutdown_wait' in gradient.timings
        with pytest.raises(ProcessLookupError):os.kill(state['pid'],0)
        assert state['work_while_exit_pending']==1496.
        print('GRADIENT_LIFECYCLE_REPORT '+json.dumps(dict(failure=failure,timings=gradient.timings,pool=pool.statistics)))
    finally:release.set();close_pools()


def test_gradient_closes_worker_after_energy_graph_failure(monkeypatch):
    from electrolites.dlpno.local import LocalGradients
    from electrolites.dlpno.resident_lifecycle import register
    from electrolites.dlpno.resident_shared_store_gpu import SharedResidentPool,close_pools
    for key in ['DLPNO_RESIDENT_INLINE_PRIMARY','DLPNO_RESIDENT_ISOLATE_DEVICES','DLPNO_RESIDENT_PARALLEL_SHUTDOWN','DLPNO_RESIDENT_OVERLAP_LIFECYCLE']:
        monkeypatch.setenv(key,'1')
    gradient=LocalGradients(None);state={}
    def body(atmlst=None):
        pool=SharedResidentPool((0,1),executor='process',budget_bytes=256<<20)
        state['pool']=pool;register(pool);pool.prestart();build_overlap(pool,ranks=(0,))
        rows=pool.run('ready',[(0,()),(1,())]);state['pid']=rows[1][1]['pid']
        raise ValueError('Injected energy graph failure')
    monkeypatch.setattr(gradient,'_kernel',body)
    try:
        with pytest.raises(ValueError,match='Injected energy graph'):gradient.kernel()
        assert state['pool'].statistics['released']
        assert all(x['live_bytes']==0 for x in state['pool'].statistics['released_workers'])
        with pytest.raises(ProcessLookupError):os.kill(state['pid'],0)
    finally:close_pools()


def test_gradient_propagates_shutdown_failure(monkeypatch):
    from electrolites.dlpno.local import LocalGradients
    from electrolites.dlpno.resident_lifecycle import register
    from electrolites.dlpno.resident_shared_store_gpu import SharedResidentPool,close_pools
    for key in ['DLPNO_RESIDENT_INLINE_PRIMARY','DLPNO_RESIDENT_ISOLATE_DEVICES','DLPNO_RESIDENT_PARALLEL_SHUTDOWN','DLPNO_RESIDENT_OVERLAP_LIFECYCLE']:
        monkeypatch.setenv(key,'1')
    gradient=LocalGradients(None);state={}
    def body(atmlst=None):
        pool=SharedResidentPool((0,1),executor='process',budget_bytes=256<<20)
        state['pool']=pool;register(pool);pool.prestart();build_overlap(pool)
        state['pid']=pool.run('ready',[(1,())])[0][1]['pid']
        shutdown=pool.workers[1].shutdown
        def fail_after_shutdown(*args,**kwargs):
            shutdown(*args,**kwargs);raise RuntimeError('Injected shutdown failure')
        monkeypatch.setattr(pool.workers[1],'shutdown',fail_after_shutdown)
        pool.begin_close()
        return np.zeros((1,3))
    monkeypatch.setattr(gradient,'_kernel',body)
    try:
        with pytest.raises(RuntimeError,match='Injected shutdown failure'):gradient.kernel()
        assert not state['pool'].statistics['released']
        with pytest.raises(ProcessLookupError):os.kill(state['pid'],0)
    finally:close_pools()
