"""Cotangents crossing a CPU/GPU edge retain their parent's backend."""
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad


@pytest.mark.parametrize('gpu_left',[False,True])
def test_mixed_backend_chain_selected_consuming(gpu_left):
    import cupy as cp
    xp,yp=(cp,np) if gpu_left else (np,cp)
    a=ad.Variable(xp.arange(6,dtype=float).reshape(2,3)+1)
    b=ad.Variable(yp.asarray([[.2,.3,.4]]))
    value=((a*a)*b).sum()
    seed=cp.array(1.)
    graph=ad.Linearization([value],[a,b])
    ga,gb=graph([seed])
    assert isinstance(ga,xp.ndarray) and isinstance(gb,yp.ndarray)
    ah=a.v.get() if xp is cp else a.v;bh=b.v.get() if yp is cp else b.v
    assert np.max(abs((ga.get() if xp is cp else ga)-2*ah*bh))<1e-12
    assert np.max(abs((gb.get() if yp is cp else gb)-(ah*ah).sum(0,keepdims=True)))<1e-12
    partial=ad.Linearization([value],[a],boundary=[a,b])([seed])[0]
    assert isinstance(partial,xp.ndarray)
    consumed=graph([seed],release_graph=True,release_inputs=True)
    assert isinstance(consumed[0],xp.ndarray) and isinstance(consumed[1],yp.ndarray)
    assert a.v is None and b.v is None


def test_mixed_indexed_parent_and_host_result():
    import cupy as cp
    x=ad.Variable(np.arange(6,dtype=float))
    ids=cp.array([1,1,4]);value=cp.asarray(x.v)[ids]
    y=ad.operation(value,[x],lambda b:(ad.IndexedCotangent(x.shape,ids,b),))
    bar=ad.Linearization([y],[x])([cp.array([2.,3.,7.])])[0]
    assert isinstance(bar,np.ndarray)
    np.testing.assert_array_equal(bar,[0.,5.,0.,0.,7.,0.])
    a=ad.Variable(cp.arange(3,dtype=float));host=ad.operation(cp.asnumpy(a.v),[a],lambda b:(b,))
    bar=ad.Linearization([host],[a])([np.ones(3)])[0]
    assert isinstance(bar,cp.ndarray)
