"""Shared immutable inputs with every PNO/TNO and singleton response."""
import os
import numpy as np
import pytest
from electrolites.dlpno import resident_shared_store_gpu as shared
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.host_input_snapshots import HostSnapshot

@pytest.fixture(autouse=True)
def mapped(monkeypatch):
    monkeypatch.setenv('DLPNO_RESIDENT_MMAP_INPUTS','1')
    monkeypatch.setenv('DLPNO_RESIDENT_MMAP_MIN_KIB','0')
    monkeypatch.setenv('DLPNO_RESIDENT_CHECKPOINT_GRAPHS','1')
    monkeypatch.setenv('DLPNO_RESIDENT_INPUT_CACHE_MIB','0')
    monkeypatch.setenv('DLPNO_ASYNC_SOURCE_REVERSE','0')
    yield
    shared.close_pools()

@pytest.mark.parametrize('energy_only',[False,True])
@pytest.mark.parametrize('devices,executor',[((0,),'thread'),((0,1),'process')])
def test_mapped_exact_singleton_full_sources(monkeypatch,energy_only,devices,executor):
    from test_dlpno_resident_single_fallback_gpu import test_unpadded_single_source_full_vjp_fd_and_replay
    test_unpadded_single_source_full_vjp_fd_and_replay(monkeypatch,True,energy_only,devices,executor)

@pytest.mark.parametrize('kind',['sc_mp2','pno','recompute','tno'])
def test_mapped_complete_subspace_responses(monkeypatch,kind):
    from test_dlpno_resident_checkpoint_gpu import test_subspaces_recompute_all_responses
    test_subspaces_recompute_all_responses(monkeypatch,kind)

@pytest.mark.parametrize('threshold_kib',[0,8192,65536])
def test_large_only_threshold_and_lifetime(monkeypatch,threshold_kib):
    monkeypatch.setenv('DLPNO_RESIDENT_MMAP_MIN_KIB',str(threshold_kib))
    pool=shared.SharedResidentPool((0,),budget_bytes=64<<20)
    path=pool.host_arena.path
    try:
        expected_bytes=expected_count=0
        for mib in [1,9]:
            raw=np.full((mib<<17,),1.25)
            key=pool.register(ad.Variable(raw))
            payload=pool.input_payload(key)
            if mib*1024>=threshold_kib:
                assert isinstance(payload,HostSnapshot)
                raw.fill(9.5)
                assert np.all(pool.values[key]==1.25)
                expected_bytes+=mib<<20;expected_count+=1
            else:assert not isinstance(payload,HostSnapshot)
        assert pool.statistics['mapped_host_input_count']==expected_count
        assert pool.statistics['mapped_host_input_bytes']==expected_bytes
    finally:cleanup=pool.close()
    assert not os.path.exists(path)
    assert all(x['live_bytes']==x['reserved_bytes']==0 for x in cleanup)
