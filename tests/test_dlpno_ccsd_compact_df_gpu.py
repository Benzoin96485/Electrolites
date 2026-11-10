"""Independent complete-source and FD checks of compact DF replay."""
import numpy as np
import pytest
from test_dlpno_ccsd_complete_sources_gpu import reference
from test_dlpno_ccsd_complete_sources_gpu import test_complete_batched_residual_vjp_and_coefficient_fd as complete_check
from test_dlpno_ccsd_df_parallel_gpu import test_df_multigpu_sources_and_iterative_transpose as parallel_check


@pytest.mark.parametrize('mixed_pairs',[False,True])
def test_compact_single_sources(reference,monkeypatch,mixed_pairs):
    monkeypatch.setenv('DLPNO_CCSD_COMPACT_DF','1')
    monkeypatch.delenv('DLPNO_CCSD_DF_DEVICES',raising=False)
    complete_check(reference,monkeypatch,mixed_pairs,False)


@pytest.mark.parametrize('devices',['0','0,1'])
@pytest.mark.parametrize('batch_size',[1,64])
@pytest.mark.parametrize('mixed_pairs',[False,True])
def test_compact_multigpu_sources(reference,monkeypatch,devices,batch_size,mixed_pairs):
    monkeypatch.setenv('DLPNO_CCSD_DF_CHECKPOINT_GRAPHS','1')
    parallel_check(reference,monkeypatch,devices,batch_size,mixed_pairs)


@pytest.mark.parametrize('tracked_base',[False,True])
def test_group_scatter_repeated_destinations_and_selected_vjp(tracked_base):
    import cupy as cp
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.ccsd_df_compact_gpu import scatter_groups
    rng=np.random.default_rng(787)
    raw=[rng.normal(size=(7,3,5)),rng.normal(size=(3,3,5)),rng.normal(size=(4,3,5))]
    slots=[np.array([1,2,5]),np.array([1,3,5,6])]
    variables=[ad.Variable(cp.asarray(x),tracked=(i>0 or tracked_base)) for i,x in enumerate(raw)]
    out=scatter_groups(variables[0],[(cp.asarray(k),x) for k,x in zip(slots,variables[1:])])
    expected=raw[0].copy()
    for k,x in zip(slots,raw[1:]):expected[k]+=x
    assert np.max(abs(out.v.get()-expected))<1e-14
    seed=rng.normal(size=raw[0].shape)
    selected=variables if tracked_base else variables[1:]
    bars=ad.Linearization([out],selected)([cp.asarray(seed)])
    gold=[seed,*[seed[k] for k in slots]] if tracked_base else [seed[k] for k in slots]
    assert max(np.max(abs(x.get()-y)) for x,y in zip(bars,gold))<1e-14
    limited=ad.Linearization([out],[variables[-1]],boundary=variables)([cp.asarray(seed)])
    assert np.max(abs(limited[0].get()-seed[slots[-1]]))<1e-14
    directions=[rng.normal(size=x.shape) for x in raw]
    if not tracked_base:directions[0]*=0
    def numerical(e):
        value=raw[0]+e*directions[0]
        for k,x,dx in zip(slots,raw[1:],directions[1:]):value[k]+=x+e*dx
        return np.sum(seed*value)
    fd=(numerical(1e-5)-numerical(-1e-5))/(2e-5)
    analytic=sum(np.sum(x*d) for x,d in zip(gold,directions if tracked_base else directions[1:]))
    assert abs(fd-analytic)<2e-8
    assert scatter_groups(variables[0],[]) is variables[0]
