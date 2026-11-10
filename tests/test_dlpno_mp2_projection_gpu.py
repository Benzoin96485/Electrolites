"""All nonsymmetric projection sources, zero weights and selected transpose."""
import json
import numpy as np
from electrolites.dlpno import reverse as ad


def test_fused_projection_all_sources_and_zero_weights():
    import cupy as cp
    from electrolites.dlpno.mp2_sources_gpu import _project_couplings,host
    rng=np.random.default_rng(754)
    raw=[.1*rng.normal(size=(5,11,7)),.1*rng.normal(size=(5,7,7)),np.array([0.,-.2,.3,-.4,.1])]
    ref=[ad.Variable(x.copy()) for x in raw]
    equation=lambda s,t,w:-w[:,None,None]*((s@t)@s.transpose(0,2,1))
    expected=equation(*ref);seed=rng.normal(size=expected.shape)
    gold=ad.Linearization([expected],ref)([seed])
    args=[ad.Variable(cp.asarray(x)) for x in raw];actual=_project_couplings(*args)
    graph=ad.Linearization([actual],args);bars=graph([cp.asarray(seed)])
    error=lambda aa,bb:max(float(np.max(abs(host(a)-host(b)),initial=0.)) for a,b in zip(aa,bb))
    value_error=error([actual],[expected]);vjp_error=error(bars,gold)
    assert value_error<2e-14 and vjp_error<2e-13,(value_error,vjp_error)
    assert error(graph([cp.asarray(.37*seed)]),[.37*x for x in gold])<2e-13
    for i in range(3):assert error(ad.Linearization([actual],[args[i]],boundary=args)([cp.asarray(seed)]),[gold[i]])<2e-13
    assert np.max(abs(host(bars[0])[0]))==np.max(abs(host(bars[1])[0]))==0.
    assert abs(host(bars[2])[0])>1e-8
    direction=[.02*rng.normal(size=x.shape) for x in raw];h=2e-5
    vals=[float(np.sum(equation(*(x+sign*h*d for x,d in zip(raw,direction)))*seed)) for sign in [1,-1]]
    fd_error=abs((vals[0]-vals[1])/(2*h)-sum(float(np.sum(host(bar)*d)) for bar,d in zip(bars,direction)))
    assert fd_error<2e-9,fd_error
    old=ad.Linearization([equation(*args)],args)
    assert graph.statistics['value_bytes']<old.statistics['value_bytes']
    assert error(graph([cp.asarray(seed)],release_graph=True,release_inputs=True),gold)<2e-13
    print('MP2_PROJECTION_REPORT '+json.dumps(dict(value_error=value_error,vjp_error=vjp_error,fd_error=fd_error,
        fused_value_bytes=graph.statistics['value_bytes'],ordinary_value_bytes=old.statistics['value_bytes'])))



def test_indexed_projection_repeated_sources_and_zero_weights():
    import cupy as cp
    from electrolites.dlpno.mp2_sources_gpu import _project_indexed_couplings,_project_couplings,host
    rng=np.random.default_rng(760);sources=np.array([2,0,1,2,0,2]);width=7
    raw=[.1*rng.normal(size=(6,11,width)),.1*rng.normal(size=(3,10,10)),np.array([0.,-.2,.3,-.4,.1,.2])]
    def equation(s,t,w):return -w[:,None,None]*((s@t[sources,:width,:width])@s.transpose(0,2,1))
    ref=[ad.Variable(x.copy()) for x in raw];expected=equation(*ref);seed=rng.normal(size=expected.shape)
    gold=ad.Linearization([expected],ref)([seed])
    args=[ad.Variable(cp.asarray(x)) for x in raw];actual=_project_indexed_couplings(*args,cp.asarray(sources))
    graph=ad.Linearization([actual],args);bars=graph([cp.asarray(seed)])
    error=lambda aa,bb:max(float(np.max(abs(host(a)-host(b)),initial=0.)) for a,b in zip(aa,bb))
    value_error=error([actual],[expected]);vjp_error=error(bars,gold)
    assert value_error<2e-14 and vjp_error<2e-13,(value_error,vjp_error)
    assert error(graph([cp.asarray(.37*seed)]),[.37*x for x in gold])<2e-13
    for i in range(3):assert error(ad.Linearization([actual],[args[i]],boundary=args)([cp.asarray(seed)]),[gold[i]])<2e-13
    tb=host(bars[1]);assert np.max(abs(tb[:,width:,:]))==np.max(abs(tb[:,:,width:]))==0.
    direction=[.02*rng.normal(size=x.shape) for x in raw];h=2e-5
    vals=[float(np.sum(equation(*(x+sign*h*d for x,d in zip(raw,direction)))*seed)) for sign in [1,-1]]
    fd_error=abs((vals[0]-vals[1])/(2*h)-sum(float(np.sum(host(bar)*d)) for bar,d in zip(bars,direction)))
    assert fd_error<2e-9,fd_error
    ordinary=_project_couplings(args[0],args[1][cp.asarray(sources),:width,:width],args[2])
    old=ad.Linearization([ordinary],args)
    assert graph.statistics['value_bytes']<old.statistics['value_bytes']
    assert error(graph([cp.asarray(seed)],release_graph=True,release_inputs=True),gold)<2e-13
    print('MP2_INDEXED_PROJECTION_REPORT '+json.dumps(dict(value_error=value_error,vjp_error=vjp_error,fd_error=fd_error,
        indexed_value_bytes=graph.statistics['value_bytes'],ordinary_value_bytes=old.statistics['value_bytes'],source_indices=sources.tolist())))
