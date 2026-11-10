"""Persistent pointer tables must not pin temporary operand allocations."""
import gc
import json
import numpy as np
import pytest

@pytest.fixture
def address_pool(monkeypatch):
    cp = pytest.importorskip('cupy')
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip('CUDA device required')
    from electrolites.dlpno import vbatch
    cache = {}
    owner = cp.cuda.MemoryPool()
    monkeypatch.setattr(vbatch, '_BASE', cache)
    monkeypatch.setattr(vbatch, '_BASE_POOL', owner)
    yield cp, vbatch, owner
    cp.cuda.get_current_stream().synchronize()
    cache.clear()
    gc.collect()
    owner.free_all_blocks()
    assert owner.used_bytes() == owner.total_bytes() == 0


def test_pointer_cache_does_not_pin_caller_pool(address_pool):
    cp, vbatch, owner = address_pool
    from electrolites.dlpno.ragged import Layout, Ragged
    layout = Layout([3, 9], [5, 2])
    blocks = [np.arange(15.).reshape(3, 5), np.arange(18.).reshape(9, 2)]
    rag = Ragged.from_list(layout, blocks)
    caller = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(caller.malloc):
        temporary = cp.empty(8<<20, dtype=cp.uint8)
        del temporary
        reserved = caller.total_bytes()
        addresses = vbatch._base(rag)
        assert vbatch._base(Ragged(layout, list(rag.data))) is addresses
        assert caller.used_bytes() == 0
        caller.free_all_blocks()
        assert caller.total_bytes() == 0
        # Leaving the cache allocator must restore the caller's allocator.
        anchor = cp.asarray([17.])
        assert caller.used_bytes() > 0
    np.testing.assert_array_equal(cp.asnumpy(addresses), [x.data.ptr for x in rag.data])
    for actual, expected in zip(rag.to_list(), blocks):
        np.testing.assert_array_equal(actual, expected)
    assert owner.used_bytes() > 0
    assert float(cp.asnumpy(anchor)[0]) == 17.
    del anchor
    caller.free_all_blocks()
    assert caller.total_bytes() == 0
    print('ADDRESS_POOL_REPORT '+json.dumps(dict(caller_reserved_before_bytes=reserved,
        caller_reserved_after_bytes=caller.total_bytes(), pointer_pool_bytes=owner.used_bytes(),
        cache_hit_preserved=True, caller_allocator_restored=True, operand_values_preserved=True)))


@pytest.mark.parametrize('order', ['l2r', 'r2l'])
def test_chain_cache_reuse_replacement_and_eviction(address_pool, order):
    cp, vbatch, owner = address_pool
    from electrolites.dlpno.ragged import Layout, Ragged, op
    rng = np.random.default_rng(7165)
    m, k, n, p = [3,9,2], [4,2,11], [2,10,5], [7,3,9]
    layouts = [Layout(m,k), Layout(n,k), Layout(n,p)]
    out_layout = Layout(m,p)
    blocks = [[rng.normal(size=(int(a),int(b))) for a,b in zip(l.rows,l.cols)] for l in layouts]
    rags = [Ragged.from_list(l,b) for l,b in zip(layouts,blocks)]
    indices = np.array([0,1,0,2,1])
    operands = [op(layouts[0],indices),op(layouts[1],indices,True),op(layouts[2],indices)]
    weights = np.array([1.,-.3,.2,0.,-2.]);alpha=.73
    plan = vbatch.VChainPlan(out_layout,indices,operands,order=order)
    maximum = 0.
    for repeat in range(3):
        if repeat == 1:
            blocks[0] = [rng.normal(size=a.shape) for a in blocks[0]]
            replacement = Ragged.from_list(layouts[0],blocks[0])
            rags[0].data = replacement.data
        if repeat == 2:
            vbatch._BASE.clear()
        initial = [rng.normal(size=(a,b)) for a,b in zip(m,p)]
        expected = [a.copy() for a in initial]
        for t, i in enumerate(indices):
            expected[i] += alpha*weights[t]*(blocks[0][i] @ blocks[1][i].T @ blocks[2][i])
        out = Ragged.from_list(out_layout,initial)
        plan.run(out,rags,w=cp.asarray(weights),alpha=alpha)
        for actual, wanted in zip(out.to_list(),expected):
            maximum = max(maximum,float(np.max(abs(actual-wanted))))
    assert maximum < 1e-10
    print('ADDRESS_CHAIN_REPORT '+json.dumps(dict(order=order,maximum_action_error=maximum,
        data_replacement_tested=True, cache_eviction_tested=True, independent_numpy_reference=True)))
