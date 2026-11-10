"""Exact source replay, bounded storage, immutable primals and failure cleanup."""
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno import resident_shared_store_gpu as shared


@pytest.fixture(autouse=True)
def replay(monkeypatch):
    monkeypatch.setenv('DLPNO_RESIDENT_CHECKPOINT_GRAPHS','1')
    monkeypatch.setenv('DLPNO_RESIDENT_INPUT_CACHE_MIB','0')
    monkeypatch.setenv('DLPNO_ASYNC_SOURCE_REVERSE','0')
    yield
    shared.close_pools()


@pytest.mark.parametrize('kind',['pno','triples','triples_energy'])
@pytest.mark.parametrize('devices,executor',[((0,),'thread'),((0,1),'process')])
def test_complete_sources_with_eviction(kind,devices,executor):
    from test_dlpno_resident_shared_store_gpu import test_complete_resident_regions
    test_complete_resident_regions(kind,devices,executor,True)


@pytest.mark.parametrize('kind',['sc_mp2','pno','recompute','tno'])
def test_subspaces_recompute_all_responses(monkeypatch,kind):
    from test_dlpno_resident_subspaces_gpu import test_shared_subspace_complete_sources
    test_shared_subspace_complete_sources(monkeypatch,kind,'0,1','process',16,kind in ('pno','recompute'),kind=='tno')


@pytest.mark.parametrize('cache_mib',[0,1])
def test_immutable_inputs_and_consumed_sibling(monkeypatch,cache_mib):
    monkeypatch.setenv('DLPNO_RESIDENT_INPUT_CACHE_MIB',str(cache_mib))
    from test_dlpno_resident_shared_store_gpu import pno_data,pno_state,pno_oracle,defer_sources,Owner
    from electrolites.dlpno.resident_regions_gpu import prepare_pno
    from electrolites.dlpno.deferred_reads import capture_reads
    raw,rng=pno_data();variables={k:ad.Variable(v.copy()) for k,v in raw.items()};inputs=list(variables.values())
    state=pno_state(variables);defer_sources(state)
    with capture_reads():prepared=[prepare_pno(state,g) for g in [(0,1),(2,4),(5,8)]]
    expected=pno_oracle(variables);seeds=[rng.normal(size=x.shape) for x in expected]
    gold=ad.Linearization(expected,inputs)(seeds)
    first_owner=Owner();second_owner=Owner();first_owner._release_transposes=second_owner._release_transposes=True
    pool=shared.SharedResidentPool((0,),budget_bytes=64<<20)
    try:
        first=shared.SharedRegionPlan(prepared,pool,owner=first_owner);out1=first.bridge()
        second=shared.SharedRegionPlan(prepared,pool,owner=second_owner);out2=second.bridge()
        assert second.statistics['uploaded_input_bytes']==0
        assert all(x['live_bytes']<=cache_mib<<20 for x in second.statistics['workers'])
        # Each worker holds the original immutable primal snapshot.
        variables['M'].v.flat[0]+=.25
        bars1=ad.Linearization(out1,inputs)(seeds)
        assert first.closed and not second.closed
        bars2=ad.Linearization(out2,inputs)([.37*x for x in seeds])
        assert max(np.max(abs(a-b),initial=0.) for a,b in zip(bars1,gold))<2e-9
        assert max(np.max(abs(a-.37*b),initial=0.) for a,b in zip(bars2,gold))<2e-9
        info=second.statistics['last_reverse_workers'][0]
        assert info['live_bytes']<=cache_mib<<20 and info['store']['verified_replays']==6
        assert info['regions']==0 and info['store']['device_input_cache_bytes']<=cache_mib<<20
        if not cache_mib:assert info['store']['input_evictions']>0
    finally:cleanup=pool.close()
    assert all(x['live_bytes']==x['reserved_bytes']==0 and x['store']['host_input_bytes']==0 for x in cleanup)


def test_replay_drift_fails_and_releases():
    from test_dlpno_triples_gpu_sources import raw_sources
    from test_dlpno_triples_source_batch_gpu import multi_state
    from electrolites.dlpno.parallel_sources_gpu import prepare_task
    raw,_=raw_sources();variables={k:ad.Variable(v.copy()) for k,v in raw.items()}
    prepared=[prepare_task(multi_state(variables),(0,1),False,64<<20,True)]
    pool=shared.SharedResidentPool((0,),budget_bytes=64<<20)
    try:
        plan=shared.SharedRegionPlan(prepared,pool);outputs=plan.bridge()
        def corrupt():
            next(iter(shared._local.store.regions.values())).digest='deliberately-invalid'
        pool.workers[0].submit(corrupt).result()
        with pytest.raises(RuntimeError,match='changed its forward outputs'):
            ad.Linearization(outputs,list(variables.values()))([np.ones(x.shape) for x in outputs])
    finally:cleanup=pool.close()
    assert all(x['live_bytes']==x['reserved_bytes']==0 for x in cleanup)


def test_private_memory_failure_and_recovery():
    from test_dlpno_resident_shared_store_gpu import test_memory_cap_and_recovery
    test_memory_cap_and_recovery()


def test_async_dependency_schedule(monkeypatch):
    from test_dlpno_async_reverse import test_shared_parent_two_stages_and_primary_thread
    test_shared_parent_two_stages_and_primary_thread(monkeypatch)


def test_ccsd_metric_overlap_sources(monkeypatch):
    from test_dlpno_ccsd_shared_overlap_gpu import test_packed_complete_overlap_sources
    test_packed_complete_overlap_sources(monkeypatch,'0,1',True)


def test_t1_metric_overlap_sources(monkeypatch):
    from test_dlpno_shared_t1_overlaps_gpu import test_shared_complete_sources
    test_shared_complete_sources((0,),(0,1),'process',True,0.,monkeypatch)
