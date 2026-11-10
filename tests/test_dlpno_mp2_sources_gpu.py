"""MP2 GPU sources against scalar residuals, independent FD and implicit solves."""
import json
from types import MethodType
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import AdjointDLPNO,Amplitudes,_names,host,values
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.reference_cpu import rebind
from test_dlpno_parallel_local_gpu import raw_data,make_state


def data():
    raw,rng=raw_data()
    raw['Fo']+=.025*rng.normal(size=(3,3))
    for p in range(9):raw['ep'+str(p)]=np.linspace(.3,1.8,raw['T'+str(p)].shape[0])
    return raw,rng


def state_for(raw,cut=0.,physical=False):
    original,_=make_state(raw);s=object.__new__(AdjointDLPNO);s.__dict__.update(original.__dict__)
    s.n=3;s.numeric=ad.Numpy(np);s.xp=np;s.progress=None;s.krylov_memory_bytes=32<<20;s.verbose=0;s.level='mp2';s.e={}
    s.opt.update(F_CUT=cut,E_CONVERGENCE=1e-10,R_CONVERGENCE=1e-9,DIIS_MAX_VECS=8,MAXITER=100)
    s.e_pno=[raw['ep'+str(p)] for p in range(9)]
    s.orb.S_pao=(raw['S']+raw['S'].T)*.5 if physical else raw['S']
    s.orb.F_lmo=(raw['Fo']+raw['Fo'].T)*.5 if physical else raw['Fo']
    if physical:
        for p,(i,j) in enumerate(s.pl.pairs):
            if i>j:continue
            if i==j:s.K[p]=(s.K[p]+s.K[p].T)*.5;s.T[p]=(s.T[p]+s.T[p].T)*.5
            q=int(s.pl.ji[p]);s.pao[q]=s.pao[p];s.X[q]=s.X[p];s.e_pno[q]=s.e_pno[p]
            s.K[q]=s.K[p].T;s.T[q]=s.T[p].T
    s.S=MethodType(rebind(RefDLPNO.S,**_names),s)
    return s


def scalar_residual(raw,amp,cut):
    s=state_for(raw,cut)
    if not isinstance(amp,ad.Variable):s.numeric=np
    v=Amplitudes(s);v.install(amp);r,_=s._mp2_residual();return v.pack(r)


@pytest.mark.parametrize('cut,batch,mixed',[(0.,1,False),(.021,16,False),(0.,16,True),(100.,16,False)])
def test_complete_sources_selected_repeat_and_fd(cut,batch,mixed,monkeypatch):
    import cupy as cp
    from electrolites.dlpno.mp2_sources_gpu import SourcePlan,SourceTranspose
    monkeypatch.setenv('DLPNO_NGPU','1');monkeypatch.setenv('DLPNO_MP2_SOURCE_BATCH',str(batch))
    raw,rng=data();template=state_for(raw,cut);av=host(Amplitudes(template).pack(template.T))
    reference={k:ad.Variable(x.copy()) for k,x in raw.items()};ref_amp=ad.Variable(av.copy())
    expected=scalar_residual(reference,ref_amp,cut);seed=rng.normal(size=expected.shape)
    gold=ad.Linearization([expected],list(reference.values())+[ref_amp])([seed])
    actual={k:ad.Variable(cp.asarray(x) if mixed and k in ('S','Fo','X0','K0','ep0') else x.copy()) for k,x in raw.items()}
    independent=ad.Variable(cp.asarray(av) if mixed else av.copy());state=state_for(actual,cut)
    if mixed:state.numeric=ad.Numpy(cp);state.xp=cp
    variables=Amplitudes(state);variables.install(independent)
    state._release_transposes=False;plan=None;transpose=None
    error=lambda aa,bb:max(float(np.max(abs(host(a)-host(b)),initial=0.)) for a,b in zip(aa,bb))
    try:
        plan=SourcePlan(state,variables,independent);output=plan.bridge()
        source_inputs=list(actual.values())+[independent];graph=ad.Linearization([output],source_inputs)
        bars=graph([seed]);value_error=error([output],[expected]);vjp_error=error(bars,gold)
        assert value_error<2e-12,value_error
        assert vjp_error<2e-11,vjp_error
        assert error(graph([.37*seed]),[.37*x for x in gold])<2e-11
        selected=[actual[k] for k in ('S','Fo','X0','K0','ep0')]+[independent]
        wanted=[gold[list(actual).index(k)] for k in ('S','Fo','X0','K0','ep0')]+[gold[-1]]
        assert error(ad.Linearization([output],selected,boundary=source_inputs)([seed]),wanted)<2e-11
        transpose=SourceTranspose(plan,variables.denominator())
        assert error(transpose([cp.asarray(seed)]),[gold[-1]])<2e-11
        assert error(transpose([cp.asarray(.19*seed)]),[.19*gold[-1]])<2e-11
        transpose.release();assert not plan.closed
        direction={k:.02*rng.normal(size=x.shape) for k,x in raw.items()};da=.02*rng.normal(size=av.shape);h=2e-5
        vals=[]
        for sign in (1,-1):
            shifted={k:x+sign*h*direction[k] for k,x in raw.items()}
            vals.append(float(np.dot(host(scalar_residual(shifted,av+sign*h*da,cut)),seed)))
        numeric=(vals[0]-vals[1])/(2*h)
        analytic=sum(float(np.sum(host(b)*direction[k])) for k,b in zip(raw,bars[:-1]))+float(np.dot(host(bars[-1]),da))
        fd_error=abs(numeric-analytic);assert fd_error<2e-8,fd_error
        state._release_transposes=True
        assert error(graph([seed],release_graph=True,release_inputs=True),gold)<2e-11
        assert plan.closed and plan.statistics['live_bytes']==0 and plan.statistics['reserved_bytes']==0,plan.statistics
        print('MP2_SOURCE_REPORT '+json.dumps(dict(cut=cut,batch=batch,mixed=mixed,value_error=value_error,vjp_error=vjp_error,directional_fd_error=fd_error,plan=plan.statistics)))
    finally:
        if transpose is not None:transpose.release()
        if plan is not None:plan.release()


def numerical_energy(raw,cut):
    s=state_for(raw,cut,True)._primal_state();RefDLPNO.lmp2_iterations(s)
    return s.e_lmp2


@pytest.mark.parametrize('cut',[0.,.021])
def test_upstream_primal_and_complete_implicit_energy(cut,monkeypatch):
    from electrolites.dlpno.mp2_sources_gpu import primal
    monkeypatch.setenv('DLPNO_NGPU','1');raw,rng=data()
    cpu=state_for(raw,cut,True)._primal_state();gpu=state_for(raw,cut,True)._primal_state()
    RefDLPNO.lmp2_iterations(cpu);primal(gpu)
    primal_error=max(np.max(abs(x-y)) for x,y in zip(cpu.T,gpu.T));assert primal_error<2e-10,primal_error
    assert abs(cpu.e_lmp2-gpu.e_lmp2)<2e-12
    results=[];states=[]
    try:
        for accelerated in [False,True]:
            monkeypatch.setenv('DLPNO_MP2_BATCHED_SOURCES',str(int(accelerated)))
            variables={k:ad.Variable(x.copy()) for k,x in raw.items()};state=state_for(variables,cut,True)
            state.lmp2_iterations();state.e['lmp2']=state.e_lmp2;state.leaves=variables
            energy=float(host(state.e_lmp2));bars=state.densities(release_graph=True,release_inputs=True)
            results.append((energy,bars));states.append(state)
        energy_error=abs(results[0][0]-results[1][0]);vjp_error=max(np.max(abs(results[0][1][k]-results[1][1][k])) for k in raw)
        assert energy_error<2e-12,energy_error
        assert vjp_error<2e-9,vjp_error
        direction={k:.02*rng.normal(size=x.shape) for k,x in raw.items()};h=2e-5
        plus=numerical_energy({k:x+h*direction[k] for k,x in raw.items()},cut)
        minus=numerical_energy({k:x-h*direction[k] for k,x in raw.items()},cut)
        numeric=(plus-minus)/(2*h);analytic=sum(float(np.sum(results[1][1][k]*direction[k])) for k in raw)
        fd_error=abs(numeric-analytic);assert fd_error<2e-8,fd_error
        info=states[1].diagnostics
        assert info['mp2']['krylov_solver_backend']=='cupy'
        assert info['batched_mp2_sources']['released'] and info['batched_mp2_sources']['live_bytes']==0
        print('MP2_IMPLICIT_REPORT '+json.dumps(dict(cut=cut,primal_amplitude_error=primal_error,energy_error=energy_error,implicit_vjp_error=vjp_error,directional_fd_error=fd_error,diagnostics=info)))
    finally:
        for state in states:
            if hasattr(state,'_mp2_source_plan'):state._mp2_source_plan.release()


def test_fixed_local_domains_keep_graph_storage_linear(monkeypatch):
    from types import SimpleNamespace
    from electrolites.dlpno.mp2_sources_gpu import SourcePlan
    monkeypatch.setenv('DLPNO_MP2_SOURCE_BATCH','1')
    records=[]
    for n in [4,8,16]:
        rng=np.random.default_rng(640+n)
        pairs=[(i,j) for i in range(n) for j in range(n) if abs(i-j)<=1]
        class ChainPairs:
            def __init__(self):
                self.pairs=pairs;self.ij=np.full((n,n),-1,dtype=int)
                for p,(i,j) in enumerate(pairs):self.ij[i,j]=p
                self.ji=np.array([self.ij[j,i] for i,j in pairs])
            def __len__(self):return len(self.pairs)
            def neighbors(self,i):return np.flatnonzero(self.ij[i]>=0)
        s=object.__new__(AdjointDLPNO);s.pl=ChainPairs();s.n=n;s.numeric=ad.Numpy(np);s.xp=np
        s.opt=dict(F_CUT=0.);s._S={};s._release_transposes=False
        f=-np.eye(n)+.01*np.eye(n,k=1)+.01*np.eye(n,k=-1)
        s.orb=SimpleNamespace(S_pao=ad.Variable(np.eye(2*n)),F_lmo=ad.Variable(f))
        s.pao=[np.array(sorted({2*i,2*i+1,2*j,2*j+1})) for i,j in pairs]
        s.X=[ad.Variable(.1*rng.normal(size=(len(p),2))) for p in s.pao]
        s.K=[ad.Variable(.1*rng.normal(size=(2,2))) for _ in pairs]
        s.e_pno=[ad.Variable(np.array([.3,.7])) for _ in pairs]
        s.T=[.01*rng.normal(size=(2,2)) for _ in pairs]
        s.S=MethodType(rebind(RefDLPNO.S,**_names),s)
        variables=Amplitudes(s);independent=ad.Variable(host(variables.pack(s.T)))
        variables.install(independent);expected=variables.pack(s._mp2_residual()[0]);plan=None
        try:
            plan=SourcePlan(s,variables,independent);output=plan.bridge()
            assert np.max(abs(host(output)-host(expected)))<2e-12
            records.append(dict(occupied=n,pairs=len(pairs),value_elements=plan.statistics['value_elements'],retained_device_bytes=plan.statistics['retained_device_bytes'],couplings=plan.statistics['couplings']))
        finally:
            if plan is not None:plan.release()
        assert plan.statistics['live_bytes']==0
    ratios=[b['value_elements']/a['value_elements'] for a,b in zip(records,records[1:])]
    assert max(ratios)<2.6,(records,ratios)
    print('MP2_LOCAL_STORAGE_REPORT '+json.dumps(dict(records=records,doubling_ratios=ratios,condition='Fixed two-orbital ranks and bounded nearest-neighbor domains; this is not an asymptotic claim for full Boys/CPHF gradients.')))
