"""Distributed local graphs against CPU equations and all-source differences."""
import json
from types import MethodType, SimpleNamespace
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import _names, host
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.reference_cpu import rebind


class Pairs:
    def __init__(self):
        self.pairs = [(i,j) for i in range(3) for j in range(3)]
        self.ij = np.arange(9).reshape(3,3)
        self.ji = np.arange(9).reshape(3,3).T.ravel()
    def __len__(self):
        return 9
    def neighbors(self,i):
        return np.arange(3)


def raw_data():
    rng = np.random.default_rng(281)
    raw = dict(S=np.eye(6)+.02*rng.normal(size=(6,6)), F=np.diag(np.linspace(.2,1.5,6))+.01*rng.normal(size=(6,6)),
               Fo=-np.diag([.8,.7,.6]), M=np.eye(8)+.01*rng.normal(size=(8,8)), q=.2*rng.normal(size=(8,3,6)),
               fdd=.1*rng.normal(size=(3,3)))
    ranks = [2,3,2,3,2,3,2,3,2]
    for p,n in enumerate(ranks):
        for name,shape in [('X',(5,n)),('T',(n,n)),('K',(n,n)),('iq',(4,n)),('Qab',(4,n,n)),
                           ('Tn',(3,n)),('Qma',(4,3,n)),('Fab',(n,n)),('beta',(3,3)),
                           ('gamma',(n,n)),('delta',(n,n))]:
            raw[name+str(p)] = .15*rng.normal(size=shape)
        i,j=divmod(p,3)
        for k in range(3):
            raw[f'J{p}_{k}'] = .1*rng.normal(size=(n,ranks[3*k+j]))
            raw[f'C{p}_{k}'] = .1*rng.normal(size=(n,ranks[3*j+k]))
    return raw,rng


def make_state(raw):
    s=SimpleNamespace(pl=Pairs(), opt=dict(S_CUT=1e-10,T_CUT_PNO_DIAG_SCALE=1.,T_CUT_PNO_CORE_SCALE=1.,
        T_CUT_PNO_MP2=1e-4,T_CUT_TRACE_MP2=.9,T_CUT_ENERGY_MP2=.9,MIN_PNOS=2,MIN_TNOS=2),ncore_scale=0,
        orb=SimpleNamespace(S_pao=(raw['S']+raw['S'].T)*.5,F_pao=(raw['F']+raw['F'].T)*.5,F_lmo=raw['Fo']),
        J2c=(raw['M']+raw['M'].T)*.5,qia=raw['q'],
        pao=[np.roll(np.arange(6),p)[:5] for p in range(9)],ribfs=[np.arange(8) for _ in range(9)],
        trip=[(0,1,2),(0,0,2),(1,1,1),(0,1,2)],t_paos=[np.arange(n) for n in [6,5,4,6]],tno_scale=np.ones(4),
        lmos=[np.arange(3) for _ in range(9)],strong=np.ones(9,bool),diagnostics={},_S={},ccsd_adjoint_backend=None)
    for name,key in [('X','X'),('T','T'),('K','K'),('iQa_t1','iq'),('Qab','Qab'),('T_n','Tn'),('Qma','Qma'),('Fab','Fab')]:
        setattr(s,name,[raw[key+str(p)] for p in range(9)])
    # Shared operands are intentional: bars must be added once per use.
    s.X[4],s.T[4]=s.X[0],s.T[0]
    s.J_ikac=[[raw[f'J{p}_{k}'] for k in range(3)] for p in range(9)]
    s.K_iakc=[[raw[f'C{p}_{k}'] for k in range(3)] for p in range(9)]
    s._K_pao=MethodType(rebind(RefDLPNO._K_pao,**_names),s)
    s.S=MethodType(rebind(RefDLPNO.S,**_names),s)
    return s,([raw['beta'+str(p)] for p in range(9)],[raw['gamma'+str(p)] for p in range(9)],
              [raw['delta'+str(p)] for p in range(9)],raw['fdd'])


def oracle(raw,kind,items):
    state,args=make_state(raw)
    method=dict(sc_mp2='_sc_mp2_pair_energy',pno='_lmp2_pno_pair',tno='_tno_triplet',ccsd_pair='_R2_pair')[kind]
    evaluate=rebind(getattr(RefDLPNO,method),**_names)
    result=[]
    for item in items:
        extra=args if kind=='ccsd_pair' else ((.008,) if kind=='tno' else ())
        value=evaluate(state,item,*extra)
        result.extend((value,) if kind=='sc_mp2' else value)
    return tuple(result)


def aligned(outputs,reference,kind):
    if kind not in ('pno','tno'):
        return outputs
    stride=5 if kind=='pno' else 2
    result=list(outputs)
    for p in range(0,len(outputs),stride):
        signs=np.sign(np.sum(host(outputs[p])*host(reference[p]),axis=0))
        assert np.all(signs!=0)
        result[p]=outputs[p]*signs
        if kind=='pno':
            result[p+2]=outputs[p+2]*signs[:,None]*signs[None,:]
            result[p+3]=outputs[p+3]*signs[:,None]*signs[None,:]
    return tuple(result)


@pytest.mark.parametrize('kind',['sc_mp2','pno','tno','ccsd_pair'])
@pytest.mark.parametrize('backend,executor',[('numpy','thread'),('numpy','process'),('cupy','thread'),('cupy','process')])
def test_local_parallel_equations_all_sources_and_fd(kind,backend,executor):
    import cupy as cp
    assert cp.cuda.runtime.getDeviceCount()>=2
    from electrolites.dlpno.parallel_local_gpu import local_window
    from electrolites.dlpno.parallel_sources_gpu import SourcePool
    raw,rng=raw_data();variables={k:ad.Variable(v.copy()) for k,v in raw.items()}
    items=(0,1,2,3) if kind=='tno' else (0,1,3,4)
    expected=oracle(variables,kind,items)
    seeds=[rng.normal(size=x.shape) for x in expected]
    inputs=list(variables.values());gold=ad.Linearization(expected,inputs)(seeds)
    state,args=make_state(variables)
    pool=SourcePool((0,1),executor)
    try:
        output,info=local_window(state,kind,(items[:2],items[2:]),devices=(0,1),cut=.008,arguments=args,backend=backend,pool=pool)
        actual=aligned(output,expected,kind)
        graph=ad.Linearization(actual,inputs)
        bars=graph(seeds)
        value_error=max(float(np.max(abs(host(a)-host(b)),initial=0.)) for a,b in zip(actual,expected))
        vjp_error=max(float(np.max(abs(a-b),initial=0.)) for a,b in zip(bars,gold))
        assert value_error<1e-9,(kind,value_error)
        assert vjp_error<2e-8,(kind,vjp_error)
        repeated=graph([.37*z for z in seeds])
        assert max(np.max(abs(a-.37*b),initial=0.) for a,b in zip(repeated,gold))<2e-8
        selected=[variables[k] for k in ('S','F','M','q','X0','T0','fdd')]
        sparse=ad.Linearization(actual,selected,boundary=inputs)(seeds)
        assert max(np.max(abs(bar-gold[next(i for i,x in enumerate(inputs) if x is value)]),initial=0.) for bar,value in zip(sparse,selected))<2e-8
        directions={k:.02*rng.normal(size=v.shape) for k,v in raw.items()}
        targets=[];h=2e-5
        for sign in (1,-1):
            displaced={k:v+sign*h*directions[k] for k,v in raw.items()}
            vals=aligned(oracle(displaced,kind,items),expected,kind)
            targets.append(sum(float(np.sum(host(v)*z)) for v,z in zip(vals,seeds)))
        numerical=(targets[0]-targets[1])/(2*h)
        analytic=sum(float(np.sum(bar*directions[k])) for k,bar in zip(variables,bars))
        assert abs(numerical-analytic)<2e-6,(kind,numerical,analytic)
        consumed=graph(seeds,release_graph=True,release_inputs=True)
        assert max(np.max(abs(a-b),initial=0.) for a,b in zip(consumed,gold))<2e-8
        assert all(x.v is None for x in inputs)
        assert set(info['device_task_counts'])=={'0','1'} and info['largest_live_worker_device_bytes']==0
        print('PARALLEL_LOCAL_REPORT '+json.dumps(dict(kind=kind,backend=backend,executor=executor,
            value_error=value_error,vjp_error=vjp_error,directional_fd_error=abs(numerical-analytic),evidence=info)))
    finally:
        cleanup=pool.close()
    assert len(cleanup)==2 and all(x['live_device_bytes']==0 for x in cleanup)
