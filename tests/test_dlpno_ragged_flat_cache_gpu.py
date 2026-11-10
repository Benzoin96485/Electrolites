"""Immutable-record cache must preserve aliases and indexed mutations."""
import numpy as np
from electrolites.dlpno import reverse as ad


def test_cached_flat_versions_and_vjp(monkeypatch):
    import cupy as cp
    from electrolites.dlpno.ragged import Layout
    from electrolites.dlpno.ragged_adjoint import AdjointRagged
    monkeypatch.setenv('DLPNO_ADJOINT_REUSE_RAGGED_FLAT','1')
    layout=Layout([2,9],[3,2]);rng=np.random.default_rng(790)
    data=[ad.Variable(cp.asarray(rng.normal(size=(len(m),r,c)))) for m,(r,c) in zip(layout.members,layout.shapes)]
    original=[ad.Variable._from_record(x._record) for x in data]
    rag=AdjointRagged(layout,data);first=rag.flat();again=rag.flat()
    assert again is not first and again._record is first._record
    assert rag.flat_cache_statistics['hits']==1
    # A caller's indexed assignment cannot corrupt the cached record.
    again[0]=23.;assert float(rag.flat().v[0])==float(first.v[0])
    replacement=ad.Variable(cp.asarray(7.))
    rag.data[0][0,0,0]=replacement
    later=rag.flat();assert later._record is not first._record and float(later.v[0])==7.
    seeds=[cp.asarray(rng.normal(size=x.size)) for x in (first,later)]
    bars=ad.Linearization([first,later],[*original,replacement])(seeds)
    offsets=np.cumsum([0]+[x.size for x in original]);gold=[]
    for i,x in enumerate(original):
        val=(seeds[0][offsets[i]:offsets[i+1]]+seeds[1][offsets[i]:offsets[i+1]]).reshape(x.shape)
        if i==0:val[0,0,0]-=seeds[1][0]
        gold.append(val)
    gold.append(seeds[1][0])
    assert max(float(cp.max(abs(x-y))) for x,y in zip(bars,gold))<1e-14
    # Whole-bucket replacement also invalidates the cached snapshot.
    rag.data[0]=ad.Variable(cp.zeros_like(data[0].v));latest=rag.flat()
    assert float(latest.v[0])==0. and latest._record is not later._record


def test_raw_mutable_arrays_are_never_cached(monkeypatch):
    import cupy as cp
    from electrolites.dlpno.ragged import Layout
    from electrolites.dlpno.ragged_adjoint import AdjointRagged
    monkeypatch.setenv('DLPNO_ADJOINT_REUSE_RAGGED_FLAT','1')
    layout=Layout([2],[3]);raw=cp.ones((1,8,8));rag=AdjointRagged(layout,[raw])
    before=rag.flat();raw[0,0,0]=9.;after=rag.flat()
    assert float(before.v[0])==1. and float(after.v[0])==9.
    assert not hasattr(rag,'_flat_snapshot')


def test_cache_disabled_keeps_original_allocation_behavior(monkeypatch):
    import cupy as cp
    from electrolites.dlpno.ragged import Layout
    from electrolites.dlpno.ragged_adjoint import AdjointRagged
    monkeypatch.delenv('DLPNO_ADJOINT_REUSE_RAGGED_FLAT',raising=False)
    rag=AdjointRagged(Layout([2],[3]),[ad.Variable(cp.ones((1,8,8)))])
    a,b=rag.flat(),rag.flat();assert a._record is not b._record
    assert not hasattr(rag,'_flat_snapshot')
