"""Full T1 source derivatives through the persistent region process store."""
import json
import os
import pytest
from test_dlpno_resident_overlap_inputs_gpu import test_resident_overlap_sources as run_reference_case


@pytest.mark.parametrize('devices,store,executor,mixed,cut',[
    ((0,),(0,),'thread',False,0.),
    ((0,),(0,),'thread',True,.035),
    ((0,1),(0,1),'thread',True,0.),
    ((0,),(0,1),'thread',True,.035),
    ((0,),(0,),'process',False,0.),
    ((0,),(0,1),'process',True,0.),
    ((0,1),(0,1),'process',False,.035),
    ((0,),(0,1),'process',True,2.),
])
def test_shared_complete_sources(devices,store,executor,mixed,cut,monkeypatch):
    monkeypatch.setenv('DLPNO_TRIPLES_SHARED_OVERLAPS','1')
    monkeypatch.setenv('DLPNO_RESIDENT_REGION_DEVICES',','.join(map(str,store)))
    monkeypatch.setenv('DLPNO_RESIDENT_REGION_EXECUTOR',executor)
    monkeypatch.setenv('DLPNO_RESIDENT_SHARED_STORE','1')
    monkeypatch.setenv('DLPNO_RESIDENT_PARALLEL_SHUTDOWN','1')
    monkeypatch.setenv('DLPNO_TRIPLES_OVERLAP_BATCH','2')
    from electrolites.dlpno.resident_shared_store_gpu import close_pools
    try:
        run_reference_case(devices,cut,mixed,monkeypatch)
    finally:
        released=close_pools()
    assert all(x['live_bytes']==0 for x in released)
    print('SHARED_T1_RELEASE_REPORT '+json.dumps(dict(devices=devices,store=store,
        executor=executor,mixed=mixed,cut=cut,released=released)))


def test_early_start_and_parallel_shutdown(monkeypatch):
    from electrolites.dlpno.resident_shared_store_gpu import SharedResidentPool,close_pools
    monkeypatch.setenv('DLPNO_RESIDENT_PARALLEL_SHUTDOWN','1')
    pool=SharedResidentPool((0,1),executor='process',budget_bytes=64<<20)
    try:
        pool.prestart();pool.prestart()
        assert len(pool.prestart_futures)==2
        result=pool.run('ready',[(0,()),(1,())])
        pids=[info['pid'] for _,info in result]
        assert len(set(pids))==2 and os.getpid() not in pids
        assert all(t is not None and t>=0 for t in pool.statistics['startup_ready_seconds'])
        released=pool.close();assert pool.close()==released
        assert all(x['live_bytes']==0 for x in released)
        for pid in pids:
            with pytest.raises(ProcessLookupError):os.kill(pid,0)
        with pytest.raises(RuntimeError,match='Closed'):
            pool.run('ready',[(0,())])
        assert pool.statistics['shutdown_seconds']>=0
        print('SHARED_LIFECYCLE_REPORT '+json.dumps(pool.statistics))
    finally:
        close_pools()
