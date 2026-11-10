"""Deferred amplitudes, complete source pullbacks and existing-worker failures."""
import json
import os
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import AdjointDLPNO, host
from test_dlpno_ccsd_complete_sources_gpu import reference, unique_inputs


@pytest.mark.parametrize('mixed_pairs',[False,True])
@pytest.mark.parametrize('keep_df',[False,True])
def test_deferred_amplitudes_keep_complete_coefficients(reference,monkeypatch,mixed_pairs,keep_df):
    from electrolites.dlpno.ccsd_sources_gpu import SourcePlan
    from test_dlpno_ccsd_complete_sources_gpu import test_complete_batched_residual_vjp_and_coefficient_fd
    original=SourcePlan.__init__
    def deferred(plan,state,**kwargs):
        final_T,final_T1=state.T,state.T1
        state.T=[ad.Variable(np.zeros(x.shape)) for x in final_T]
        del state.T1
        try:original(plan,state,prepare_only=True,**kwargs)
        finally:state.T,state.T1=final_T,final_T1
        assert not plan.finished and not hasattr(plan,'output')
        plan.finish()
        assert plan.finished and not hasattr(plan,'source') and not hasattr(plan,'ref')
        with pytest.raises(RuntimeError,match='exactly once'):plan.finish()
    monkeypatch.setattr(SourcePlan,'__init__',deferred)
    test_complete_batched_residual_vjp_and_coefficient_fd(reference,monkeypatch,mixed_pairs,keep_df)


def configure(monkeypatch):
    for k,v in dict(DLPNO_CCSD_PRIMAL_PIPELINE='1',DLPNO_CCSD_BATCHED_SOURCES='1',
        DLPNO_CCSD_SOURCE_KEEP_DF='1',DLPNO_CCSD_SOURCE_REUSE_TRANSPOSE='1',DLPNO_CCSD_SHARED_OVERLAPS='1',
        DLPNO_RESIDENT_REGION_DEVICES='0,1',DLPNO_RESIDENT_SHARED_STORE='1',DLPNO_RESIDENT_ISOLATE_DEVICES='1',
        DLPNO_RESIDENT_INLINE_PRIMARY='1',DLPNO_RESIDENT_REGION_EXECUTOR='process',
        DLPNO_RESIDENT_REGION_MEMORY_MIB='2048',DLPNO_RESIDENT_PARALLEL_SHUTDOWN='1').items():monkeypatch.setenv(k,v)


@pytest.mark.parametrize('case',['strong','weak','source_failure','worker_failure'])
def test_existing_worker_pipeline_complete_sources_and_failure(reference,monkeypatch,case):
    import cupy as cp
    from benchmarks.dlpno_gradient_check import STRESS
    from electrolites.dlpno.reference_cpu import CPUDLPNO
    from electrolites.dlpno.ccsd_adjoint_gpu import _fixed_state
    from electrolites.dlpno.ccsd_primal_pipeline_gpu import solve_primal
    from electrolites.dlpno.ccsd_sources_gpu import SourcePlan,SourceTranspose
    from electrolites.dlpno.resident_regions_gpu import finish,_pool
    monkeypatch.setenv('DLPNO_NGPU','1');monkeypatch.setenv('DLPNO_RESIDENT_REGION_DEVICES','')
    monkeypatch.setenv('DLPNO_CCSD_BATCHED_SOURCES','0');monkeypatch.setenv('DLPNO_CCSD_PRIMAL_PIPELINE','0')
    monkeypatch.setenv('DLPNO_GRADIENT_DEVICES','')
    options=dict(triples=None,verbose=0,**STRESS,E_CONVERGENCE=1e-11,R_CONVERGENCE=1e-10,MAXITER=150,T_CUT_ITER=0.)
    if case=='weak':options['T_CUT_PAIRS']=1e6
    base=CPUDLPNO(reference,**options);base.kernel()
    original_ccsd=AdjointDLPNO.lccsd_iterations;original_implicit=AdjointDLPNO._implicit
    states=[];plans=[];numeric=[]
    class Checked(Exception):pass
    def ccsd(self):
        states.append(self)
        configure(monkeypatch);self.ccsd_primal_backend='gpu'
        # Match the gradient: workers already belong to an earlier source stage.
        _pool(self).prestart()
        if case in ('strong','weak'):
            # Identical initial MP2 doubles; numerical CCSD always starts T1=0.
            numeric.append(solve_primal(_fixed_state(self,before_ccsd=True)))
            return original_ccsd(self)
        if case=='source_failure':
            original_load=SourcePlan._load_ints
            def fail(plan,*args):
                plans.append(plan)
                original_load(plan,*args)
                raise ValueError('injected preparation failure')
            monkeypatch.setattr(SourcePlan,'_load_ints',fail)
            with pytest.raises(ValueError,match='injected preparation failure'):original_ccsd(self)
            assert all(p.released and p.statistics['released_pool_live_bytes']==0 for p in plans)
        else:
            self.opt=dict(self.opt,MAXITER=0)
            with pytest.raises(RuntimeError,match='did not converge'):original_ccsd(self)
        assert not hasattr(self,'_ccsd_primal_reserved_rank')
        assert self._resident_region_pool.run('ready',[(1,())])[0][1]['pid']!=os.getpid()
        raise Checked
    def inspect(self,name,independent,residual,variables):
        if name!='ccsd':return original_implicit(self,name,independent,residual,variables)
        old=numeric[0]
        primal_error=max(float(np.max(abs(host(a)-b),initial=0.)) for a,b in zip(self.T1+self.T,old.T1+old.T))
        assert primal_error<3e-11,primal_error
        inputs,keys=unique_inputs(self)
        r2,r1=self._cc_residual()  # Unmodified scalar reference bound before enabling pipeline.
        expected=variables.pack(r2,r1)
        rng=np.random.default_rng(659);seed=rng.normal(size=independent.size)
        gold=ad.Linearization([expected],inputs)([seed])
        graph=ad.Linearization([residual],inputs);actual=graph([seed])
        error=max(float(np.max(abs(a-b),initial=0.)) for a,b in zip(actual,gold));assert error<3e-8,error
        plan=self._batched_ccsd_source_plan
        transpose=SourceTranspose(plan,variables,independent,residual)
        try:
            expected_t=ad.Linearization([expected],[independent])([seed])[0]
            transpose_error=float(np.max(abs(host(transpose([seed])[0])-expected_t)));assert transpose_error<3e-8
        finally:transpose.release()
        assert not plan.released
        assert self.diagnostics['shared_ccsd_overlaps']['scheduling_ranks']==[0]
        evidence=self.diagnostics['ccsd_primal_pipeline'];assert evidence['new_workers_created']==0
        assert evidence['worker']['device']==1 and evidence['worker']['pid']!=os.getpid()
        self._release_transposes=True
        consumed=graph([seed],release_graph=True)
        assert max(float(np.max(abs(a-b),initial=0.)) for a,b in zip(consumed,gold))<3e-8
        assert plan.released and plan.statistics['released_pool_live_bytes']==0
        print('CCSD_PIPELINE_REPORT '+json.dumps(dict(case=case,primal_error=primal_error,source_vjp_error=error,
            transpose_error=transpose_error,pipeline=evidence)))
        raise Checked
    monkeypatch.setattr(AdjointDLPNO,'lccsd_iterations',ccsd)
    monkeypatch.setattr(AdjointDLPNO,'_implicit',inspect)
    try:
        with pytest.raises(Checked):AdjointDLPNO(base).kernel()
    finally:
        for state in states:finish(state)
    assert states
    for state in states:
        pool=state._resident_region_pool
        assert pool.statistics['released']
        for info in pool.statistics['released_workers']:
            assert info['live_bytes']==info['reserved_bytes']==0,info
            if info['pid']!=os.getpid():
                with pytest.raises(ProcessLookupError):os.kill(info['pid'],0)


def test_pipeline_never_creates_workers(monkeypatch):
    from types import SimpleNamespace
    from electrolites.dlpno.ccsd_primal_pipeline_gpu import prepare
    monkeypatch.setenv('DLPNO_CCSD_BATCHED_SOURCES','1')
    state=SimpleNamespace(xp=np,ccsd_primal_backend='gpu')
    with pytest.raises(ValueError,match='cannot create workers'):prepare(state)
    assert not hasattr(state,'_resident_region_pool')
