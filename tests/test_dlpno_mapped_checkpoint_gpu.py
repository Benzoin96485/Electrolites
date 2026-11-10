import os
import numpy as np
from electrolites.dlpno import reverse as ad
from electrolites.dlpno import resident_shared_store_gpu as shared


def test_mapped_pno_sources_and_arena_cleanup(monkeypatch):
    from test_dlpno_resident_shared_store_gpu import pno_data,pno_state,pno_oracle,defer_sources,Owner
    from electrolites.dlpno.resident_regions_gpu import prepare_pno
    from electrolites.dlpno.deferred_reads import capture_reads
    monkeypatch.setenv('DLPNO_RESIDENT_CHECKPOINT_GRAPHS','1')
    monkeypatch.setenv('DLPNO_RESIDENT_MMAP_INPUTS','1')
    monkeypatch.setenv('DLPNO_RESIDENT_MMAP_MIN_KIB','0')
    monkeypatch.setenv('DLPNO_RESIDENT_INPUT_CACHE_MIB','0')
    monkeypatch.setenv('DLPNO_ASYNC_SOURCE_REVERSE','0')
    raw,rng=pno_data();variables={k:ad.Variable(v.copy()) for k,v in raw.items()};inputs=list(variables.values())
    state=pno_state(variables);defer_sources(state)
    with capture_reads():prepared=[prepare_pno(state,g) for g in [(0,1),(2,4),(5,8)]]
    expected=pno_oracle(variables);seeds=[rng.normal(size=x.shape) for x in expected]
    gold=ad.Linearization(expected,inputs)(seeds)
    owner=Owner();owner._release_transposes=True
    pool=shared.SharedResidentPool((0,1),budget_bytes=64<<20,executor='process')
    path=pool.host_arena.path
    try:
        plan=shared.SharedRegionPlan(prepared,pool,owner=owner);outputs=plan.bridge()
        variables['M'].v.flat[0]+=.25
        actual=ad.Linearization(outputs,inputs)(seeds)
        assert max(np.max(abs(a-b),initial=0.) for a,b in zip(actual,gold))<2e-9
        workers=plan.statistics['workers']
        assert all(x['store']['mapped_host_input_bytes']>0 for x in workers)
        assert sum(x['store']['mapped_host_input_bytes'] for x in workers)>pool.statistics['mapped_host_input_bytes']
        assert all(not x.flags.writeable for x in pool.values)
    finally:cleanup=pool.close()
    assert not os.path.exists(path) and pool.statistics['mapped_host_arena_released']
    assert all(x['live_bytes']==x['reserved_bytes']==x['store']['mapped_host_input_bytes']==0 for x in cleanup)
