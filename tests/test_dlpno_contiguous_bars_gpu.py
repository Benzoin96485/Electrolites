"""Contiguous cotangents preserve independent storage and captured VJPs."""
import gc,json
import numpy as np
import pytest

@pytest.mark.parametrize('rows,cols',[([],[]),([2,9,17],[3,7,19]),([128,256,512],[128,256,512])])
def test_flat_bar_storage_and_lifetime(monkeypatch,rows,cols):
    import cupy as cp
    from electrolites.dlpno.ragged import Layout
    from electrolites.dlpno.ragged_adjoint import _empty_bar
    layout=Layout(rows,cols);reports=[]
    for enabled in (False,True):
        monkeypatch.setenv('DLPNO_ADJOINT_CONTIGUOUS_BARS',str(int(enabled)))
        pool=cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(pool.malloc):
            result=_empty_bar(layout)
            for i,b in enumerate(result.data):b.fill(i+1.)
            cp.cuda.get_current_stream().synchronize();before=pool.used_bytes()
            flat=result.flat();cp.cuda.get_current_stream().synchronize();after=pool.used_bytes()
            expected=np.concatenate([np.full((len(m)*r*c,),i+1.) for i,(m,(r,c)) in enumerate(zip(layout.members,layout.shapes))]) if layout.shapes else np.empty(0)
            np.testing.assert_array_equal(cp.asnumpy(flat),expected)
            if enabled:assert result.flat() is flat and after==before
            elif layout.nbytes():assert after-before==layout.nbytes()
            b=None
            del result
            gc.collect();np.testing.assert_array_equal(cp.asnumpy(flat),expected)
            other=_empty_bar(layout)
            if flat.size:
                other.flat().fill(-1.);assert float(flat[0])==expected[0]
            del other,flat
        gc.collect();pool.free_all_blocks();assert pool.used_bytes()==0
        reports.append(dict(contiguous=enabled,bytes=layout.nbytes(),before=before,after=after))
    print('CONTIGUOUS_BAR_MEMORY '+json.dumps(reports))


def test_complete_chain_vjp_cuda_replay(monkeypatch):
    import cupy as cp
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.ragged import Layout,op
    from electrolites.dlpno.ragged_adjoint import AdjointRagged,AdjointChainPlan
    from electrolites.dlpno.cuda_graph_replay import Replay
    monkeypatch.setenv('DLPNO_ADJOINT_CONTIGUOUS_BARS','1')
    monkeypatch.setenv('DLPNO_ADJOINT_REUSE_RAGGED_FLAT','1')
    layout=Layout([3,11,3],[3,11,3]);rng=np.random.default_rng(802)
    x,y=[ad.Variable(cp.asarray(rng.normal(size=layout.nbytes()//8))) for _ in range(2)]
    a=AdjointRagged(layout).set_flat(x);b=AdjointRagged(layout).set_flat(y)
    result=AdjointRagged(layout);ids=np.asarray([0,1,0,2,1]);p=AdjointChainPlan(layout,ids,[op(layout,ids),op(layout,ids,True)])
    p.run(result,[a,b]);z=result.flat();graph=ad.Linearization([z],[x,y]);seed=cp.asarray(rng.normal(size=z.size));gold=graph([seed])
    stream=cp.cuda.Stream(non_blocking=True);replay=None
    try:
        cp.cuda.get_current_stream().synchronize()
        with stream:
            replay=Replay(lambda seeds:graph(seeds),[seed],memory_bytes=128<<20)
            for scale in (1.,-.31):
                bars=replay([scale*seed]);stream.synchronize()
                for actual,expected in zip(bars,gold):assert float(cp.max(abs(actual-scale*expected)))<1e-11
        graph([seed],release_graph=True,release_inputs=True)
    finally:
        if replay is not None:
            replay.release();assert replay.statistics['live_bytes']==replay.statistics['reserved_bytes']==0
