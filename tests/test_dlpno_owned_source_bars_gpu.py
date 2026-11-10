"""Full source equations and mixed CPU/GPU boundaries with owned bars."""
import numpy as np
import pytest
from concurrent.futures import Future
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.reverse_async import Pending


@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('kind', ['pno', 'triples', 'triples_energy'])
def test_complete_owned_process_sources(monkeypatch, checkpoint, kind):
    from test_dlpno_async_reverse import configure
    from test_dlpno_resident_shared_store_gpu import test_complete_resident_regions
    configure(monkeypatch)
    monkeypatch.setenv('DLPNO_RESIDENT_TRANSFER_BARS', '1')
    monkeypatch.setenv('DLPNO_RESIDENT_CHECKPOINT_GRAPHS', str(int(checkpoint)))
    monkeypatch.setenv('DLPNO_RESIDENT_INPUT_CACHE_MIB', '0')
    test_complete_resident_regions(kind, (0, 1), 'process', True)


@pytest.mark.parametrize('asynchronous', [False, True])
def test_owned_mixed_host_device_conversion(monkeypatch, asynchronous):
    import cupy as cp
    monkeypatch.setenv('DLPNO_ASYNC_SOURCE_REVERSE', str(int(asynchronous)))
    a=np.arange(12.,dtype=np.float64).reshape(4,3)/13
    b=np.array([[.5,.7,1.2]])
    x=ad.Variable(cp.asarray(a));y=ad.Variable(b.copy())
    def pullback(z,n):return (z*b if n[0] else None,z*a if n[1] else None)
    def submit(z,n):
        future=Future();future.set_result(pullback(z,n));return Pending((future,),future.result)
    output=ad.operation(a*b,[x,y],lambda z:pullback(z,[True,True]),pullback,async_pullback=submit,owns_cotangents=True)
    graph=ad.Linearization([output],[x,y]);seed=np.ones_like(a)
    bars=graph([seed]);assert isinstance(bars[0],cp.ndarray) and isinstance(bars[1],np.ndarray)
    np.testing.assert_allclose(cp.asnumpy(bars[0]),np.broadcast_to(b,a.shape),rtol=0,atol=1e-14)
    np.testing.assert_allclose(bars[1],a.sum(axis=0,keepdims=True),rtol=0,atol=1e-14)
    assert graph.last_statistics['adopted_elements']>0
    again=graph([.5*seed],release_graph=True,release_inputs=True)
    for c,d in zip(bars,again):np.testing.assert_allclose(cp.asnumpy(d),.5*cp.asnumpy(c),rtol=0,atol=1e-14)
    assert x.v is y.v is None
