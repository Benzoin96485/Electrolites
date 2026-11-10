"""Bound reverse copies under an actual private pool cap without losing VJPs."""
from types import SimpleNamespace
import numpy as np
import pytest
from electrolites.dlpno import resident_shared_store_gpu as shared
from electrolites.dlpno.resident_checkpoint_store_gpu import CheckpointStore


def test_reproduce_pack_oom_then_bound_copies(monkeypatch):
    import cupy as cp
    monkeypatch.setenv('DLPNO_RESIDENT_INPUT_CACHE_MIB','0')
    pool=cp.cuda.MemoryPool();pool.set_limit(size=32<<20)
    saved=dict(shared._local.__dict__);shared._local.pool=pool
    try:
        with cp.cuda.using_allocator(pool.malloc):
            store=CheckpointStore();n=(5<<20)//8
            store.host_inputs={k:(np.empty(n),True) for k in range(4)}
            store.host_bytes=sum(x[0].nbytes for x in store.host_inputs.values())
            monkeypatch.setenv('DLPNO_RESIDENT_COTANGENT_FAST_GUARD','1')
            arrays=[cp.full(n,k+0.125,dtype=cp.float64) for k in range(4)]
            bars=[(('input',k),v) for k,v in enumerate(arrays)]
            monkeypatch.setenv('DLPNO_RESIDENT_COTANGENT_CHUNKED','0')
            with pytest.raises(cp.cuda.memory.OutOfMemoryError):
                store._accumulate_bars(bars,set(range(4)),{})
            assert pool.used_bytes()==20<<20
            monkeypatch.setenv('DLPNO_RESIDENT_COTANGENT_CHUNKED','1')
            first={};store._accumulate_bars(bars,set(range(4)),first)
            second={k:v.copy() for k,v in first.items()}
            store._accumulate_bars(bars,set(range(4)),second)
            for k,a in enumerate(arrays):
                assert np.array_equal(first[k],cp.asnumpy(a))
                assert np.array_equal(second[k],2*first[k])
                assert bool(cp.all(a==k+0.125))
            assert store.statistics['chunked_reverse_regions']==2
            assert store.statistics['largest_legacy_pack_estimated_bytes']>32<<20
            assert store.statistics['largest_chunked_pool_bytes']<=25<<20
            arrays.clear();bars.clear();a=None;store.close()
        pool.free_all_blocks();assert pool.used_bytes()==0
    finally:
        shared._local.__dict__.clear();shared._local.__dict__.update(saved)


@pytest.mark.parametrize('selected',[False,True])
def test_aliased_direct_and_indexed_contributions(monkeypatch,selected):
    import cupy as cp
    monkeypatch.setenv('DLPNO_RESIDENT_INPUT_CACHE_MIB','0')
    pool=cp.cuda.MemoryPool();pool.set_limit(size=32<<20)
    saved=dict(shared._local.__dict__);shared._local.pool=pool
    try:
        with cp.cuda.using_allocator(pool.malloc):
            store=CheckpointStore();store.host_inputs={k:(np.empty(8),True) for k in range(3)}
            common=cp.asarray([1.,2.,3.,4.,5.,6.,7.,8.]);indexed=cp.asarray([.5,.25,-.125,2.])
            local=(np.array([0,2,2,5]),)
            descriptor=SimpleNamespace(cartesian_shape=(4,),sources=(0,2),source_shapes=((8,),(8,)),parts=((local,Ellipsis),(local,Ellipsis)))
            store.reads['read']=(descriptor,None)
            bars=[(('input',0),common),(('input',1),common),(('read','read'),indexed),(('input',0),common[::-1])]
            needed={0,1} if selected else {0,1,2}
            monkeypatch.setenv('DLPNO_RESIDENT_COTANGENT_CHUNKED','0')
            gold={};store._accumulate_bars(bars,needed,gold)
            monkeypatch.setenv('DLPNO_RESIDENT_COTANGENT_CHUNKED','1')
            monkeypatch.setattr(store,'_requires_chunking',lambda groups:True)
            result={};store._accumulate_bars(bars,needed,result)
            assert gold.keys()==result.keys()==needed
            for k in needed:assert np.array_equal(result[k],gold[k])
            assert np.array_equal(cp.asnumpy(common),np.arange(1.,9.))
            assert np.array_equal(cp.asnumpy(indexed),[.5,.25,-.125,2.])
            assert not np.shares_memory(result[0],result[1])
            bars.clear();common=indexed=None;store.close()
        pool.free_all_blocks();assert pool.used_bytes()==0
    finally:
        shared._local.__dict__.clear();shared._local.__dict__.update(saved)


def force_chunking(monkeypatch):
    monkeypatch.setenv('DLPNO_RESIDENT_CHECKPOINT_GRAPHS','1')
    monkeypatch.setenv('DLPNO_RESIDENT_INPUT_CACHE_MIB','0')
    monkeypatch.setenv('DLPNO_RESIDENT_COTANGENT_CHUNKED','1')
    monkeypatch.setenv('DLPNO_RESIDENT_COTANGENT_FAST_GUARD','0')
    monkeypatch.setenv('DLPNO_ASYNC_SOURCE_REVERSE','0')
    monkeypatch.setattr(CheckpointStore,'_requires_chunking',lambda self,groups:True)


@pytest.mark.parametrize('kind',['pno','triples','triples_energy'])
def test_full_formula_directional_fd(monkeypatch,kind):
    from test_dlpno_resident_shared_store_gpu import test_complete_resident_regions
    force_chunking(monkeypatch)
    test_complete_resident_regions(kind,(0,1),'thread',True)


@pytest.mark.parametrize('kind',['sc_mp2','pno','recompute','tno'])
def test_complete_subspace_responses(monkeypatch,kind):
    from test_dlpno_resident_subspaces_gpu import test_shared_subspace_complete_sources
    force_chunking(monkeypatch)
    test_shared_subspace_complete_sources(monkeypatch,kind,'0,1','thread',16,kind in ('pno','recompute'),kind=='tno')


@pytest.mark.parametrize('selected',[False,True])
@pytest.mark.parametrize('fortran',[False,True])
def test_headroom_skips_plan_with_aliased_repeated_and_indexed_bars(monkeypatch,selected,fortran):
    import cupy as cp
    monkeypatch.setenv('DLPNO_RESIDENT_INPUT_CACHE_MIB','0')
    pool=cp.cuda.MemoryPool();pool.set_limit(size=32<<20)
    saved=dict(shared._local.__dict__);shared._local.pool=pool
    try:
        with cp.cuda.using_allocator(pool.malloc):
            store=CheckpointStore()
            store.host_inputs={k:(np.empty((4,4)),True) for k in range(3)}
            store.host_bytes=sum(x[0].nbytes for x in store.host_inputs.values())
            host=np.arange(16,dtype=np.float64).reshape(4,4)*.125
            common=cp.asarray(host,order='F' if fortran else 'C')
            indexed=cp.asarray([.5,.25,-.125,2.])
            local=(np.array([0,2,2,3]),np.array([1,2,2,0]))
            descriptor=SimpleNamespace(cartesian_shape=(4,),sources=(0,2),
                source_shapes=((4,4),(4,4)),parts=((local,Ellipsis),(local,Ellipsis)))
            store.reads['read']=(descriptor,None)
            bars=[(('input',0),common),(('input',1),common),
                (('read','read'),indexed),(('input',0),common[::-1])]
            needed={0,1} if selected else {0,1,2}
            monkeypatch.setenv('DLPNO_RESIDENT_COTANGENT_CHUNKED','0')
            gold={};store._accumulate_bars(bars,needed,gold)
            monkeypatch.setenv('DLPNO_RESIDENT_COTANGENT_CHUNKED','1')
            monkeypatch.setenv('DLPNO_RESIDENT_COTANGENT_FAST_GUARD','1')
            def forbidden(*args):raise AssertionError('Small source built a cotangent group plan')
            monkeypatch.setattr(store,'_cotangent_groups',forbidden)
            actual={};store._accumulate_bars(bars,needed,actual)
            first={k:v.copy() for k,v in actual.items()}
            store._accumulate_bars(bars,needed,actual)
            assert actual.keys()==gold.keys()==needed
            for k in needed:
                assert np.array_equal(first[k],gold[k])
                assert np.array_equal(actual[k],2*gold[k])
            assert np.array_equal(cp.asnumpy(common),host)
            assert np.array_equal(cp.asnumpy(indexed),[.5,.25,-.125,2.])
            assert store.statistics['cotangent_group_plans_skipped']==2
            assert store.statistics.get('cotangent_group_plans_built',0)==0
            assert not np.shares_memory(actual[0],actual[1])
            bars.clear();common=indexed=None;store.close()
        pool.free_all_blocks();assert pool.used_bytes()==0
    finally:
        shared._local.__dict__.clear();shared._local.__dict__.update(saved)


@pytest.mark.parametrize('host_bytes,bar_bytes,used,expected',[
    (1024,1024,0,True),
    (1024,9<<20,0,False),
    (1024,1024,17<<20,False),
    (5<<20,1024,0,False),
    (0,1024,0,False),
])
def test_headroom_counts_gather_volume_and_existing_pool(monkeypatch,host_bytes,bar_bytes,used,expected):
    # No allocation is necessary to check the conservative decision boundary.
    saved=dict(shared._local.__dict__)
    shared._local.pool=SimpleNamespace(get_limit=lambda:32<<20,used_bytes=lambda:used)
    try:
        store=object.__new__(CheckpointStore);store.host_bytes=host_bytes
        store.host_inputs={0:None};store.statistics={}
        assert store._legacy_pack_has_headroom([(('read',0),SimpleNamespace(nbytes=bar_bytes))]) is expected
    finally:
        shared._local.__dict__.clear();shared._local.__dict__.update(saved)
