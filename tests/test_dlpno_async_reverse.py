"""Async outer VJP: dependencies, aliases, source FD and resource ownership."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import host
from electrolites.dlpno.reverse_async import Pending


@pytest.mark.parametrize('mixed',[False,True])
def test_ready_dependencies_aliases_selected_repeated_and_consuming(monkeypatch,mixed):
    import cupy as cp
    monkeypatch.setenv('DLPNO_ASYNC_SOURCE_REVERSE','1')
    rng=np.random.default_rng(668);a=rng.normal(size=(4,3));b=rng.normal(size=(1,3))
    x=ad.Variable(a);y=ad.Variable(cp.asarray(b) if mixed else b);unused=ad.Variable(np.ones(2))
    arrived=threading.Event();lock=threading.Lock();submitted=[]
    executor=ThreadPoolExecutor(max_workers=2)
    def source(u,v):
        uu,vv=host(u).copy(),host(v).copy()
        def pullback(bar,needed):
            return tuple(z if need else None for z,need in zip((bar*vv,bar*(uu+uu),bar*vv),needed))
        def submit(bar,needed):
            def task():
                with lock:
                    submitted.append(1)
                    if len(submitted)%2==0:arrived.set()
                assert arrived.wait(timeout=5),'Independent source never submitted'
                return pullback(bar,needed)
            future=executor.submit(task)
            return Pending((future,),future.result)
        return ad.operation((uu+uu)*vv,[u,v,u],lambda z:pullback(z,[True]*3),pullback,async_pullback=submit)
    try:
        outputs=[source(x,y),source(.3*x[:,::-1],y+.2)]
        gx,gy,gu=ad.Variable(a.copy()),ad.Variable(b.copy()),ad.Variable(np.ones(2))
        gold_outputs=[(gx+gx)*gy,(.3*gx[:,::-1]+.3*gx[:,::-1])*(gy+.2)]
        seeds=[rng.normal(size=z.shape) for z in outputs]
        gold=ad.Linearization(gold_outputs,[gx,gy,gu])(seeds)
        graph=ad.Linearization(outputs,[x,y,unused]);actual=graph(seeds)
        error=max(float(np.max(abs(host(u)-host(v)),initial=0.)) for u,v in zip(actual,gold));assert error<1e-12
        assert graph.last_statistics['maximum_pending']==2
        assert graph.last_statistics['completed_records']==len(graph.records)
        assert max(float(np.max(abs(host(u)-.37*host(v)),initial=0.)) for u,v in zip(graph([.37*z for z in seeds]),gold))<1e-12
        subset=ad.Linearization(outputs,[y],boundary=[x,y,unused])(seeds)
        assert np.max(abs(host(subset[0])-host(gold[1])))<1e-12
        # Finite difference perturbs both independent arrays, including a repeated operand.
        da=rng.normal(size=a.shape);db=rng.normal(size=b.shape);h=1e-6
        def value(amount):
            aa=a+amount*da;bb=b+amount*db
            return np.sum(2*aa*bb*seeds[0])+np.sum(.6*aa[:,::-1]*(bb+.2)*seeds[1])
        fd=(value(h)-value(-h))/(2*h);anal=np.sum(host(actual[0])*da)+np.sum(host(actual[1])*db)
        assert abs(fd-anal)<1e-7
        consumed=graph(seeds,release_graph=True,release_inputs=True)
        assert max(float(np.max(abs(host(u)-host(v)),initial=0.)) for u,v in zip(consumed,gold))<1e-12
        assert x.v is y.v is unused.v is None
        with pytest.raises(RuntimeError,match='consumed'):graph(seeds)
        print('ASYNC_DAG_REPORT '+json.dumps(dict(mixed=mixed,vjp_error=error,source_fd_error=float(abs(fd-anal)),statistics=graph.last_statistics)))
    finally:executor.shutdown(wait=True,cancel_futures=True)


@pytest.mark.parametrize('failure',['worker','caller'])
def test_failure_joins_inflight_sibling(monkeypatch,failure):
    monkeypatch.setenv('DLPNO_ASYNC_SOURCE_REVERSE','1')
    x=ad.Variable(np.ones(3));started=threading.Event();release=threading.Event();finished=threading.Event()
    executor=ThreadPoolExecutor(max_workers=2);futures=[]
    def submit(bar,needed):
        def task():
            started.set()
            assert release.wait(timeout=5)
            finished.set()
            return (bar.copy(),)
        future=executor.submit(task);futures.append(future);return Pending((future,),future.result)
    slow=ad.operation(x.v.copy(),[x],lambda b:(b,),lambda b,n:(b,),async_pullback=submit)
    def fail(bar,needed):
        assert started.wait(timeout=5);release.set()
        raise ValueError('injected async sibling failure')
    if failure=='worker':
        def submit_fail(bar,needed):
            future=executor.submit(fail,bar,needed);futures.append(future);return Pending((future,),future.result)
        bad=ad.operation(x.v.copy(),[x],lambda b:(b,),lambda b,n:(b,),async_pullback=submit_fail)
    else:bad=ad.operation(x.v.copy(),[x],lambda b:fail(b,[True]),fail)
    try:
        with pytest.raises(ValueError,match='injected async sibling failure'):
            ad.Linearization([slow+bad],[x])([np.ones(3)],release_graph=True)
        assert finished.is_set() and all(f.done() for f in futures)
    finally:release.set();executor.shutdown(wait=True,cancel_futures=True)


def configure(monkeypatch):
    for k,v in dict(DLPNO_NGPU='1',DLPNO_ASYNC_SOURCE_REVERSE='1',DLPNO_RESIDENT_THREADED_PRIMARY='1',
        DLPNO_RESIDENT_INLINE_PRIMARY='1',DLPNO_RESIDENT_ISOLATE_DEVICES='1',DLPNO_RESIDENT_PARALLEL_SHUTDOWN='1').items():monkeypatch.setenv(k,v)


@pytest.mark.parametrize('kind',['pno','triples','triples_energy'])
def test_complete_process_sources(monkeypatch,kind):
    from test_dlpno_resident_shared_store_gpu import test_complete_resident_regions
    configure(monkeypatch)
    test_complete_resident_regions(kind,(0,1),'process',True)


def test_shared_parent_two_stages_and_primary_thread(monkeypatch):
    from electrolites.dlpno.resident_shared_store_gpu import SharedResidentPool,SharedRegionPlan
    from test_dlpno_resident_shared_store_gpu import Owner,pno_data,pno_state,pno_oracle,defer_sources
    from electrolites.dlpno.resident_regions_gpu import prepare_pno
    from electrolites.dlpno.deferred_reads import capture_reads
    configure(monkeypatch);raw,rng=pno_data();leaves={k:ad.Variable(v.copy()) for k,v in raw.items()}
    inputs=list(leaves.values());state=pno_state(leaves);defer_sources(state)
    with capture_reads():prepared=[prepare_pno(state,g) for g in [(0,1),(2,4),(5,8)]]
    expected=pno_oracle(leaves);seeds=[rng.normal(size=x.shape) for x in expected]
    gold=ad.Linearization(expected,inputs)([1.37*z for z in seeds])
    pool=SharedResidentPool((0,1),executor='process',budget_bytes=256<<20);owner=Owner();owner._release_transposes=True
    try:
        first=SharedRegionPlan(prepared,pool,owner=owner);out1=first.bridge()
        # Reuse checks require identical rank placement, not inherited load bias.
        pool.loads[:]=[0]*len(pool.devices)
        second=SharedRegionPlan(prepared,pool,owner=owner);out2=second.bridge()
        assert [r for r,_ in first.assignments]==[r for r,_ in second.assignments]
        assert second.statistics['uploaded_input_bytes']==0 and pool.statistics['threaded_primary']
        graph=ad.Linearization(tuple(out1)+tuple(out2),inputs)
        bars=graph(seeds+[.37*z for z in seeds],release_graph=True,release_inputs=True)
        error=max(float(np.max(abs(a-b),initial=0.)) for a,b in zip(bars,gold));assert error<3e-9
        assert first.closed and second.closed
        assert graph.last_statistics['async_submissions']==2
        assert all(x.v is None for x in inputs)
        print('ASYNC_SHARED_REPORT '+json.dumps(dict(vjp_error=error,statistics=graph.last_statistics)))
    finally:cleanup=pool.close()
    assert all(x['live_bytes']==x['reserved_bytes']==0 for x in cleanup)
    for info in cleanup:
        if info['pid']!=os.getpid():
            with pytest.raises(ProcessLookupError):os.kill(info['pid'],0)


def test_default_never_submits_async_tasks(monkeypatch):
    monkeypatch.setenv('DLPNO_ASYNC_SOURCE_REVERSE','0')
    x=ad.Variable(np.ones(3))
    def forbidden(*args):raise AssertionError('Default path submitted a task')
    y=ad.operation(2*x.v,[x],lambda b:(2*b,),lambda b,n:(2*b,),async_pullback=forbidden)
    graph=ad.Linearization([y],[x]);bars=graph([np.ones(3)],release_graph=True,release_inputs=True)
    assert np.array_equal(bars[0],np.full(3,2.)) and x.v is None
    assert 'async_submissions' not in graph.last_statistics
