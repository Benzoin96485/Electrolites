"""Eliding structural zero bars preserves full source VJPs and bounded memory."""
import json
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno import resident_shared_store_gpu as shared


@pytest.mark.parametrize('kind',['pno','triples','triples_energy'])
@pytest.mark.parametrize('defer',[False,True])
def test_complete_connected_sources(monkeypatch,kind,defer):
    from test_dlpno_resident_shared_store_gpu import test_complete_resident_regions
    monkeypatch.setenv('DLPNO_RESIDENT_CONNECTED_SOURCES','1')
    monkeypatch.setenv('DLPNO_RESIDENT_CHECKPOINT_GRAPHS','1')
    monkeypatch.setenv('DLPNO_RESIDENT_INPUT_CACHE_MIB','0')
    monkeypatch.setenv('DLPNO_RESIDENT_PRIMAL_FORWARD','1')
    monkeypatch.setenv('DLPNO_ASYNC_SOURCE_REVERSE','0')
    stores=[];original=shared.SharedResidentPool.close
    def close(self):
        rows=original(self);stores.extend(row['store'] for row in rows);return rows
    monkeypatch.setattr(shared.SharedResidentPool,'close',close)
    # Independent CPU equations, directional finite differences, selected,
    # repeated and consuming VJPs, plus zero-live-byte worker cleanup.
    test_complete_resident_regions(kind,(0,1),'process',defer)
    if defer:
        assert sum(x.get('disconnected_boundary_bars_skipped',0) for x in stores)>0
        assert sum(x.get('disconnected_boundary_bytes_avoided',0) for x in stores)>0


@pytest.mark.parametrize('enabled',[False,True])
def test_real_graph_with_32mib_disconnected_source_and_48mib_pool(monkeypatch,enabled):
    import cupy as cp
    from test_dlpno_resident_shared_store_gpu import pno_data,pno_state,defer_sources
    from electrolites.dlpno.resident_regions_gpu import prepare_pno
    from electrolites.dlpno.deferred_reads import capture_reads
    monkeypatch.setenv('DLPNO_RESIDENT_CONNECTED_SOURCES',str(int(enabled)))
    monkeypatch.setenv('DLPNO_RESIDENT_CHECKPOINT_GRAPHS','1')
    monkeypatch.setenv('DLPNO_RESIDENT_INPUT_CACHE_MIB','0')
    monkeypatch.setenv('DLPNO_RESIDENT_PRIMAL_FORWARD','1')
    monkeypatch.setenv('DLPNO_ASYNC_SOURCE_REVERSE','0')
    raw,_=pno_data();variables={k:ad.Variable(v.copy()) for k,v in raw.items()}
    state=pno_state(variables);defer_sources(state)
    with capture_reads():spec,operands=prepare_pno(state,(0,1))
    unused=ad.Variable(np.ones((2048,2048),dtype=np.float64))
    pool=shared.SharedResidentPool((0,),budget_bytes=48<<20,executor='thread')
    try:
        plan=shared.SharedRegionPlan([(spec,operands+(unused,))],pool)
        outputs=plan.bridge();seeds=[np.ones(x.shape) for x in outputs]
        graph=ad.Linearization(outputs,list(variables.values())+[unused])
        if enabled:
            bars=graph(seeds)
            np.testing.assert_array_equal(bars[-1],np.zeros_like(unused.v))
            small=ad.Linearization(outputs,[variables['M']],boundary=list(variables.values())+[unused])(seeds)
            np.testing.assert_allclose(small[0],bars[list(variables).index('M')],rtol=0,atol=2e-9)
            stats=plan.statistics['last_reverse_workers'][0]['store']
            assert stats['disconnected_boundary_bytes_avoided']>=unused.v.nbytes
            print('CONNECTED_CAPACITY '+json.dumps(dict(pool_bytes=48<<20,unused_source_bytes=unused.v.nbytes,enabled=True,statistics=stats)))
        else:
            with pytest.raises(cp.cuda.memory.OutOfMemoryError):graph(seeds)
            print('CONNECTED_CAPACITY '+json.dumps(dict(pool_bytes=48<<20,unused_source_bytes=unused.v.nbytes,enabled=False,expected_oom=True)))
    finally:cleanup=pool.close()
    if enabled:assert all(x['live_bytes']==x['reserved_bytes']==0 for x in cleanup)
