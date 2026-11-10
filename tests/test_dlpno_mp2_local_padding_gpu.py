"""Heterogeneous local ranks exercise padding, all sources and amplitude J.T."""
import json
from types import SimpleNamespace,MethodType
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import AdjointDLPNO,Amplitudes,host,_names
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.reference_cpu import rebind


class Pairs:
    def __init__(self,n):
        self.pairs=[(i,j) for i in range(n) for j in range(n) if abs(i-j)<=1]
        self.ij=np.full((n,n),-1,dtype=int)
        for p,(i,j) in enumerate(self.pairs):self.ij[i,j]=p
        self.ji=np.array([self.ij[j,i] for i,j in self.pairs])
    def __len__(self):return len(self.pairs)
    def neighbors(self,i):return np.flatnonzero(self.ij[i]>=0)


def data():
    rng=np.random.default_rng(746);n=6;pl=Pairs(n)
    raw=dict(S=np.eye(20*n)+.01*rng.normal(size=(20*n,20*n)),F=-np.eye(n)+.02*rng.normal(size=(n,n)))
    for p,(i,j) in enumerate(pl.pairs):
        rank=17 if max(i,j)==n-1 else 2+(i+j)%3
        width=20 if i==j else 40
        raw['X'+str(p)]=.08*rng.normal(size=(width,rank))
        raw['K'+str(p)]=.1*rng.normal(size=(rank,rank))
        raw['T'+str(p)]=.01*rng.normal(size=(rank,rank))
        raw['e'+str(p)]=np.linspace(.3,1.6,rank)
    return raw,rng


def state(raw):
    s=object.__new__(AdjointDLPNO);s.n=6;s.pl=Pairs(s.n);s.xp=np;s.numeric=ad.Numpy(np)
    s.opt=dict(F_CUT=0.);s._S={};s._release_transposes=False
    s.orb=SimpleNamespace(S_pao=raw['S'],F_lmo=raw['F'])
    s.pao=[np.array(sorted(set(range(20*i,20*i+20))|set(range(20*j,20*j+20)))) for i,j in s.pl.pairs]
    s.X=[raw['X'+str(p)] for p in range(len(s.pl))];s.K=[raw['K'+str(p)] for p in range(len(s.pl))]
    s.T=[raw['T'+str(p)] for p in range(len(s.pl))];s.e_pno=[raw['e'+str(p)] for p in range(len(s.pl))]
    s.S=MethodType(rebind(RefDLPNO.S,**_names),s)
    return s


def oracle(raw,amplitude):
    s=state(raw)
    if not isinstance(amplitude,ad.Variable):s.numeric=np
    v=Amplitudes(s);v.install(amplitude);return v.pack(s._mp2_residual()[0])


@pytest.mark.parametrize('batch',[1,4])
def test_local_rank_complete_sources_and_transpose(batch,monkeypatch):
    import cupy as cp
    from electrolites.dlpno.mp2_sources_gpu import SourcePlan,SourceTranspose
    monkeypatch.setenv('DLPNO_NGPU','1');monkeypatch.setenv('DLPNO_MP2_SOURCE_BATCH',str(batch))
    raw,rng=data();template=state(raw);av=host(Amplitudes(template).pack(template.T));seed=rng.normal(size=av.shape)
    ref={k:ad.Variable(x.copy()) for k,x in raw.items()};ra=ad.Variable(av.copy())
    expected=oracle(ref,ra);gold=ad.Linearization([expected],list(ref.values())+[ra])([seed])
    error=lambda a,b:max(float(np.max(abs(host(x)-host(y)),initial=0.)) for x,y in zip(a,b))
    reports=[]
    for local in [False,True]:
        monkeypatch.setenv('DLPNO_MP2_LOCAL_PADDING',str(int(local)))
        monkeypatch.setenv('DLPNO_MP2_RANK_BUCKETS',str(int(local)))
        monkeypatch.setenv('DLPNO_MP2_INDEXED_PROJECTION',str(int(local)))
        variables={k:ad.Variable(x.copy()) for k,x in raw.items()};amp=ad.Variable(av.copy());s=state(variables)
        avars=Amplitudes(s);avars.install(amp);plan=None;transpose=None
        try:
            plan=SourcePlan(s,avars,amp);out=plan.bridge();inputs=list(variables.values())+[amp]
            graph=ad.Linearization([out],inputs);bars=graph([seed])
            value_error=error([out],[expected]);vjp_error=error(bars,gold)
            assert value_error<2e-12 and vjp_error<3e-11,(value_error,vjp_error)
            assert error(graph([.37*seed]),[.37*x for x in gold])<3e-11
            selected=[variables['S'],variables['F'],variables['X0'],amp]
            target=[gold[list(variables).index(k)] for k in ['S','F','X0']]+[gold[-1]]
            assert error(ad.Linearization([out],selected,boundary=inputs)([seed]),target)<3e-11
            transpose=SourceTranspose(plan,avars.denominator());assert error(transpose([cp.asarray(seed)]),[gold[-1]])<3e-11
            transpose.release()
            direction={k:.02*rng.normal(size=x.shape) for k,x in raw.items()};da=.02*rng.normal(size=av.shape);h=2e-5
            vals=[float(np.dot(host(oracle({k:x+sign*h*direction[k] for k,x in raw.items()},av+sign*h*da)),seed)) for sign in [1,-1]]
            numeric=(vals[0]-vals[1])/(2*h)
            analytic=sum(float(np.sum(host(bar)*direction[k])) for k,bar in zip(raw,bars[:-1]))+float(np.dot(host(bars[-1]),da))
            fd_error=abs(numeric-analytic);assert fd_error<2e-8,fd_error
            evidence=dict(local=local,batch=batch,value_error=value_error,vjp_error=vjp_error,fd_error=fd_error,
                retained_device_bytes=plan.statistics['retained_device_bytes'],local_partial_elements=plan.statistics['local_partial_elements'],
                global_partial_elements=plan.statistics['global_padded_partial_elements'],rank_histogram=plan.statistics['coupling_batch_rank_histogram'],shape_histogram=plan.statistics['coupling_batch_shape_histogram'],
                indexed_projection=plan.statistics['indexed_projection'],identity_overlap_gathers=plan.statistics['identity_overlap_gathers'])
            s._release_transposes=True
            assert error(graph([seed],release_graph=True,release_inputs=True),gold)<3e-11
            assert plan.statistics['live_bytes']==plan.statistics['reserved_bytes']==0
            reports.append(evidence)
        finally:
            if transpose is not None:transpose.release()
            if plan is not None:plan.release()
    assert reports[1]['local_partial_elements']<reports[0]['local_partial_elements']
    assert reports[1]['retained_device_bytes']<reports[0]['retained_device_bytes']
    print('MP2_LOCAL_PADDING_REPORT '+json.dumps(reports))
