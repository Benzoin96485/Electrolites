"""Independent source and consuming-replay tests for GPU overlap pullbacks."""
import json
from types import SimpleNamespace
import numpy as np
import pytest


def gpu():
    cp=pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount():pytest.skip('CUDA required')
    except cp.cuda.runtime.CUDARuntimeError:pytest.skip('CUDA required')
    return cp


@pytest.mark.parametrize('backend',['numpy','cupy'])
@pytest.mark.parametrize('mask',[1,2,3,4,5,6,7])
def test_gpu_bilinear_all_source_subsets(backend,mask):
    cp=gpu();xp=np if backend=='numpy' else cp
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.adjoint import host
    from electrolites.dlpno.overlap_adjoint_gpu import bilinear_batch
    rng=np.random.default_rng(117)
    dims=[(4,5,2,3),(3,2,1,2),(2,4,0,2)]
    raw=[rng.normal(size=(m,a)) for m,n,a,b in dims]+[rng.normal(size=(m,n)) for m,n,a,b in dims]+[rng.normal(size=(n,b)) for m,n,a,b in dims]
    weights=[rng.normal(size=(a,b)) for m,n,a,b in dims]
    leaves=[ad.Variable(xp.asarray(x)) for x in raw];count=len(dims)
    outputs=bilinear_batch(leaves[:count],leaves[count:2*count],leaves[2*count:])
    exact=[leaves[i].T@leaves[count+i]@leaves[2*count+i] for i in range(count)]
    error=max(float(np.max(abs(host(a)-host(b)),initial=0.)) for a,b in zip(outputs,exact));assert error<2e-12
    loss=sum(np.sum(y*xp.asarray(w)) for y,w in zip(outputs,weights))
    oracle_loss=sum(np.sum(y*xp.asarray(w)) for y,w in zip(exact,weights))
    selected=[i for i in range(3*count) if mask&(1<<(i//count))]
    inputs=[leaves[i] for i in selected]
    expected=ad.gradients(oracle_loss,inputs)
    graph=ad.Linearization([loss],inputs)
    actual=graph([xp.asarray(1.)])
    vjp_error=max(float(np.max(abs(host(a)-host(b)),initial=0.)) for a,b in zip(actual,expected));assert vjp_error<2e-11
    repeat=graph([xp.asarray(.31)])
    assert max(float(np.max(abs(host(a)-.31*host(b)),initial=0.)) for a,b in zip(repeat,actual))<2e-11
    direction=[rng.normal(size=x.shape) if i in selected else np.zeros_like(x) for i,x in enumerate(raw)]
    norm=np.sqrt(sum(np.sum(x*x) for x in direction));direction=[x/norm for x in direction]
    def oracle(values):
        return sum(np.sum((values[i].T@values[count+i]@values[2*count+i])*weights[i]) for i in range(count))
    h=1e-5;finite=(oracle([x+h*d for x,d in zip(raw,direction)])-oracle([x-h*d for x,d in zip(raw,direction)]))/(2*h)
    analytic=sum(np.sum(host(bar)*direction[i]) for i,bar in zip(selected,actual));fd_error=float(abs(finite-analytic));assert fd_error<2e-7
    consumed=graph([xp.asarray(1.)],release_graph=True)
    assert max(float(np.max(abs(host(a)-host(b)),initial=0.)) for a,b in zip(consumed,actual))<2e-11
    print('GPU_BILINEAR_REPORT '+json.dumps(dict(backend=backend,mask=mask,forward_error=error,vjp_error=vjp_error,directional_fd_error=fd_error)))


@pytest.mark.parametrize('backend',['numpy','cupy'])
def test_gpu_overlap_checkpoint_late_shared_sources_and_repeated_rows(backend):
    cp=gpu();xp=np if backend=='numpy' else cp
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.adjoint import _checkpointed_ccsd_overlaps_gpu,host
    rng=np.random.default_rng(118)
    pao=[np.array([3,0,3]),np.array([1,2]),np.array([4,0,2])]
    raw=[rng.normal(size=(5,5))]+[rng.normal(size=(len(p),n)) for p,n in zip(pao,(2,3,2))]
    leaves=[ad.Variable(xp.asarray(x)) for x in raw]
    state=SimpleNamespace(orb=SimpleNamespace(S_pao=leaves[0]),X=leaves[1:],pao=pao,_S={},diagnostics={})
    first=_checkpointed_ccsd_overlaps_gpu(state,((0,1),(1,0),(0,0)))
    a,b,c=(first[key] for key in ((0,1),(1,0),(0,0)))
    (y,),_=ad.checkpoint(lambda:(a@b+.2*c,))
    later=_checkpointed_ccsd_overlaps_gpu(state,((0,1),(1,2)))
    assert later[0,1] is a
    (z,),_=ad.checkpoint(lambda:(a@later[1,2],))
    target=np.sum(y*y)+.17*np.sum(z*z)
    m,x0,x1,x2=leaves
    def overlap(p,q):return leaves[1+p].T@m[np.ix_(pao[p],pao[q])]@leaves[1+q]
    ey=overlap(0,1)@overlap(1,0)+.2*overlap(0,0);ez=overlap(0,1)@overlap(1,2)
    expected=ad.gradients(np.sum(ey*ey)+.17*np.sum(ez*ez),leaves)
    graph=ad.Linearization([target],leaves);reusable=graph([xp.asarray(1.)]);actual=graph([xp.asarray(1.)],release_graph=True,release_inputs=True)
    error=max(float(np.max(abs(host(a)-host(b)),initial=0.)) for a,b in zip(actual,expected));assert error<2e-10
    assert max(float(np.max(abs(host(a)-host(b)),initial=0.)) for a,b in zip(actual,reusable))<2e-10
    assert all(x.v is None for x in leaves)
    info=state.diagnostics['ccsd_shared_overlaps'];assert info['gpu_overlaps']==4 and info['cache_hits']==1
    print('GPU_OVERLAP_CHECKPOINT_REPORT '+json.dumps(dict(backend=backend,max_source_error=error,statistics=info)))


def test_gpu_overlap_workspace_planner_bounds_and_ordered_keys():
    gpu()
    from electrolites.dlpno.overlap_adjoint_gpu import batches,workspace_bytes
    pao=[np.arange(x) for x in (7,12,65,97)]
    coeff=[np.empty((len(p),n)) for p,n in zip(pao,(2,4,13,20))]
    keys=tuple((p,q) for p in range(4) for q in range(4))
    groups=batches(keys,pao,coeff,max_bytes=2<<20,max_count=3)
    assert sorted(key for group,bound in groups for key in group)==sorted(keys)
    assert all(len(group)<=3 and bound<=2<<20 for group,bound in groups)
    for group,bound in groups:
        maximum=tuple(max((len(pao[p]),len(pao[q]),coeff[p].shape[1],coeff[q].shape[1])[k] for p,q in group) for k in range(4))
        assert bound==workspace_bytes(len(group),maximum)
    with pytest.raises(ValueError,match='One overlap'):
        batches(((3,3),),pao,coeff,max_bytes=64)
