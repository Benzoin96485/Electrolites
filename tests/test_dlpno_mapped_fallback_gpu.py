"""Combined mapped snapshots and exact oversized-singleton coefficient replay."""
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.resident_shared_store_gpu import SharedResidentPool
from electrolites.dlpno.host_input_snapshots import HostSnapshot

@pytest.mark.parametrize('energy_only',[False,True])
@pytest.mark.parametrize('devices,executor',[((0,),'thread'),((0,1),'process')])
def test_mapped_exact_singleton_complete_vjp(monkeypatch,energy_only,devices,executor):
    from test_dlpno_resident_single_fallback_gpu import test_unpadded_single_source_full_vjp_fd_and_replay
    monkeypatch.setenv('DLPNO_RESIDENT_MMAP_INPUTS','1')
    monkeypatch.setenv('DLPNO_RESIDENT_MMAP_MIN_KIB','0')
    test_unpadded_single_source_full_vjp_fd_and_replay(monkeypatch,True,energy_only,devices,executor)


def test_eight_mib_threshold_mixed_snapshots(monkeypatch):
    monkeypatch.setenv('DLPNO_RESIDENT_CHECKPOINT_GRAPHS','1')
    monkeypatch.setenv('DLPNO_RESIDENT_MMAP_INPUTS','1')
    monkeypatch.setenv('DLPNO_RESIDENT_MMAP_MIN_KIB','8192')
    raw=[np.full(n,.37) for n in ((1<<20)-1,1<<20,(1<<20)+1)]
    leaves=[ad.Variable(v) for v in raw]
    pool=SharedResidentPool((0,),budget_bytes=64<<20,executor='thread')
    try:
        ids=[pool.register(v) for v in leaves]
        assert not isinstance(pool.input_payload(ids[0]),HostSnapshot)
        assert all(isinstance(pool.input_payload(k),HostSnapshot) for k in ids[1:])
        assert pool.statistics['mapped_host_input_count']==2
        assert pool.statistics['mapped_host_input_bytes']==raw[1].nbytes+raw[2].nbytes
        for v in raw:v.fill(19.)
        for k in ids[1:]:
            assert not pool.values[k].flags.writeable
            assert np.all(pool.values[k]==.37)
    finally:rows=pool.close()
    assert pool.statistics['mapped_host_arena_released']
    assert all(x['live_bytes']==x['reserved_bytes']==0 for x in rows)
