"""All CCSD source derivatives, not only the fixed-coefficient transpose."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from pyscf import gto,scf
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import AdjointDLPNO,host
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.reference_cpu import CPUDLPNO
from electrolites.dlpno.response import ResponseDLPNO
from benchmarks.dlpno_gradient_check import STRESS


@pytest.fixture(scope='module')
def reference():
    import cupy as cp
    assert cp.cuda.runtime.getDeviceCount()>=1
    mol=gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57',basis='6-31g',verbose=0)
    return scf.RHF(mol).density_fit().run(conv_tol=1e-13,conv_tol_grad=1e-10)


def unique_inputs(state):
    names=('X','e_pno','T','T1','K','L_iajb','K_mibj','J_ijmb','K_ivvv','J_ikac','K_iakc','Qma','Qab','iQk','iQa')
    result,keys,seen=[],[],set()
    def add(x,name):
        if isinstance(x,(list,tuple)):
            for i,y in enumerate(x):add(y,name+':'+str(i))
        elif isinstance(x,ad.Variable) and x._record not in seen:
            seen.add(x._record);result.append(ad.Variable._from_record(x._record));keys.append(name)
    for name in names:add(getattr(state,name),name)
    for name in ('S_pao','F_lmo'):add(getattr(state.orb,name),name)
    return result,keys


def displaced_state(state,inputs,directions,amount):
    mapping={x._record:host(x)+amount*d for x,d in zip(inputs,directions)}
    def clone(x):
        if isinstance(x,ad.Variable):return mapping.get(x._record,host(x)).copy()
        if isinstance(x,list):return [clone(y) for y in x]
        if isinstance(x,tuple):return tuple(clone(y) for y in x)
        if isinstance(x,dict):return {k:clone(v) for k,v in x.items()}
        if isinstance(x,SimpleNamespace):return SimpleNamespace(**{k:clone(v) for k,v in vars(x).items()})
        return x
    result=object.__new__(RefDLPNO)
    names=('pl','n','strong','X','lmos','lmo_pos','pao','K','K_mibj','J_ijmb','K_ivvv','L_iajb','J_ikac','K_iakc',
           'iQk','iQa','Qma','Qab','e_pno','T','T1','opt','orb')
    result.__dict__={name:clone(getattr(state,name)) for name in names}
    result._S={}
    return result


@pytest.mark.parametrize('keep_df',[False,True])
@pytest.mark.parametrize('mixed_pairs',[False,True])
def test_complete_batched_residual_vjp_and_coefficient_fd(reference,monkeypatch,mixed_pairs,keep_df):
    from electrolites.dlpno.ccsd_sources_gpu import SourcePlan, SourceTranspose
    monkeypatch.setenv('DLPNO_NGPU','1');monkeypatch.setenv('DLPNO_GRADIENT_DEVICES','')
    monkeypatch.setenv('DLPNO_CCSD_BATCHED_SOURCES','0')
    monkeypatch.setenv('DLPNO_CCSD_SOURCE_KEEP_DF',str(int(keep_df)))
    options=dict(triples=None,verbose=0,**STRESS,E_CONVERGENCE=1e-11,R_CONVERGENCE=1e-10,MAXITER=150,T_CUT_ITER=0.)
    if mixed_pairs:options['T_CUT_PAIRS']=1e6
    base=CPUDLPNO(reference,**options);base.kernel()
    original=AdjointDLPNO._implicit
    class Checked(Exception):pass
    def inspect(self,name,independent,residual,variables):
        if name!='ccsd':return original(self,name,independent,residual,variables)
        inputs,keys=unique_inputs(self)
        expected2,expected1=self._cc_residual()
        expected=tuple(expected2)+tuple(expected1)
        rng=np.random.default_rng(288)
        seeds=[rng.normal(size=x.shape) for x in expected]
        gold=ad.Linearization(expected,inputs)(seeds)
        plan=SourcePlan(self)
        try:
            transpose=SourceTranspose(plan,variables,independent,residual)
            transpose_errors=[]
            exact=ad.Linearization([residual],[independent])
            for _ in range(2):
                seed=rng.normal(size=independent.size)
                transpose_errors.append(float(np.max(abs(host(transpose([seed])[0])-exact([seed])[0]))))
            assert max(transpose_errors)<3e-8,transpose_errors
            transpose.release()
            assert not plan.released
            actual2,actual1=plan.bridge();actual=tuple(actual2)+tuple(actual1)
            value_error=max(float(np.max(abs(host(a)-host(b)),initial=0.)) for a,b in zip(actual,expected))
            assert value_error<2e-8,value_error
            graph=ad.Linearization(actual,inputs)
            bars=graph(seeds)
            errors=[float(np.max(abs(a-b),initial=0.)) for a,b in zip(bars,gold)]
            worst=keys[int(np.argmax(errors))]
            assert max(errors)<3e-8,(worst,max(errors))
            repeat=graph([.37*s for s in seeds])
            assert max(np.max(abs(a-.37*b),initial=0.) for a,b in zip(repeat,gold))<3e-8
            selected=[i for i,k in enumerate(keys) if k.split(':')[0] in ('X','S_pao','F_lmo','e_pno','Qma','Qab')]
            limited=ad.Linearization(actual,[inputs[i] for i in selected],boundary=inputs)(seeds)
            assert max(np.max(abs(a-gold[i]),initial=0.) for a,i in zip(limited,selected))<3e-8
            directions=[rng.normal(size=x.shape) for x in inputs]
            scale=np.sqrt(sum(np.sum(d*d) for d in directions));directions=[d/scale for d in directions]
            targets=[];step=1e-5
            for sign in (1,-1):
                numerical=displaced_state(self,inputs,directions,sign*step)
                r2,r1=ResponseDLPNO._cc_residual(numerical)
                targets.append(sum(float(np.sum(v*s)) for v,s in zip(tuple(r2)+tuple(r1),seeds)))
            finite=(targets[0]-targets[1])/(2*step)
            analytic=sum(float(np.sum(v*d)) for v,d in zip(bars,directions))
            assert abs(finite-analytic)<2e-6,(finite,analytic)
            # The final outer sweep consumes the resident graph exactly once.
            self._release_transposes=True
            consumed=graph(seeds,release_graph=True)
            self._release_transposes=False
            assert max(np.max(abs(a-b),initial=0.) for a,b in zip(consumed,gold))<3e-8
            assert plan.released and plan.statistics['released_pool_live_bytes']==0,plan.statistics
            print('COMPLETE_CCSD_SOURCE_REPORT '+json.dumps(dict(mixed_pairs=mixed_pairs,keep_df=keep_df,transpose_errors=transpose_errors,inputs=len(inputs),
                value_error=value_error,maximum_vjp_error=max(errors),worst_input=worst,
                coefficient_directional_fd_error=abs(finite-analytic),statistics=plan.statistics)))
        finally:plan.release()
        raise Checked
    monkeypatch.setattr(AdjointDLPNO,'_implicit',inspect)
    with pytest.raises(Checked):AdjointDLPNO(base).kernel()
