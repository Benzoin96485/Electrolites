"""Exact final T1 energy pullbacks, including repeated occupied labels."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

from electrolites.dlpno import reverse as ad
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.reference_cpu import rebind
from electrolites.dlpno.triples_energy_gpu import energy_vector
from test_dlpno_triples_residual_gpu import gpu


def scalar(triplets, v, a):
    evaluate=rebind(RefDLPNO._e_ijk,np=ad.Numpy(np))
    return ad.concatenate([evaluate(*labels,x,y).reshape(1) for labels,x,y in zip(triplets,v,a)])


@pytest.mark.parametrize('backend',['numpy','cupy'])
@pytest.mark.parametrize('inputs',['numpy','cupy','mixed'])
def test_energy_sources(backend,inputs):
    cp=gpu();rng=np.random.default_rng(649)
    triplets=[(0,1,2),(0,0,1),(0,1,1),(1,1,1),(0,2,3),(0,1,2)]
    ranks=[3,3,3,3,0,1];raw={}
    for name in ('v','t'):
        for j,n in enumerate(ranks):raw[name+str(j)]=rng.normal(size=(n,n,n))
    def sources(values):
        v=[values['v'+str(j)] for j in range(6)];t=[values['t'+str(j)] for j in range(6)]
        v[2]=v[0];t[2]=t[1];t[3]=v[3]
        return v,t
    backends=[cp if inputs=='cupy' or inputs=='mixed' and i%2 else np for i in range(len(raw))]
    leaves={k:ad.Variable(xp.asarray(v)) for (k,v),xp in zip(raw.items(),backends)}
    refs={k:ad.Variable(v.copy()) for k,v in raw.items()}
    actual=energy_vector(triplets,*sources(leaves),backend=backend,max_batch=2,workspace_bytes=32768)
    expected=scalar(triplets,*sources(refs))
    seed=rng.normal(size=6);graph=ad.Linearization([actual],list(leaves.values()))
    bars=graph([seed]);gold=ad.Linearization([expected],list(refs.values()))([seed])
    def host(v):return cp.asnumpy(ad.primal(v)) if isinstance(ad.primal(v),cp.ndarray) else ad.primal(v)
    error=lambda a,b:max(float(np.max(abs(host(x)-host(y)),initial=0.)) for x,y in zip(a,b))
    value_error=error([actual],[expected]);vjp_error=error(bars,gold)
    assert value_error<2e-12 and vjp_error<2e-12
    assert all(isinstance(bar,xp.ndarray) for bar,xp in zip(bars,backends))
    assert error(graph([.37*seed]),[.37*x for x in bars])<2e-12
    for group in ('v','t'):
        indices=[i for i,k in enumerate(leaves) if k.startswith(group)]
        got=ad.Linearization([actual],[list(leaves.values())[i] for i in indices],boundary=list(leaves.values()))([seed])
        assert error(got,[gold[i] for i in indices])<2e-12
    direction={k:.1*rng.normal(size=x.shape) for k,x in raw.items()}
    h=2e-5;values=[]
    for sign in (1,-1):
        shifted={k:x+sign*h*direction[k] for k,x in raw.items()};vs,ts=sources(shifted)
        values.append(sum(z*RefDLPNO._e_ijk(*labels,v,t) for z,labels,v,t in zip(seed,triplets,vs,ts)))
    finite=(values[0]-values[1])/(2*h)
    analytic=sum(float(np.sum(host(bar)*direction[k])) for k,bar in zip(raw,bars))
    fd_error=abs(finite-analytic);assert fd_error<2e-8
    assert error(graph([seed],release_graph=True,release_inputs=True),bars)<2e-12
    assert all(x.v is None for x in leaves.values())
    print('TRIPLES_ENERGY_REPORT '+json.dumps(dict(backend=backend,inputs=inputs,value_error=value_error,
        vjp_error=vjp_error,directional_fd_error=fd_error,records=graph.statistics['records'])))


def test_bounds_and_empty():
    x=ad.Variable(np.ones((4,4,4)))
    with pytest.raises(ValueError,match='workspace'):
        energy_vector([(0,1,2)],[x],[x],backend='numpy',workspace_bytes=1024)
    assert energy_vector([],[],[],backend='numpy').shape==(0,)


def test_fixed_shape_energy_micro():
    """Synthetic FP64 sources with saved COF2 TNO dimensions; no new solve."""
    cp=gpu();import cloudpickle
    p=Path('/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/cof384-primal.pkl')
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='fce75b5efd87b69440c622a13a3e08cbe9fcbbf817fd2b6f135b72cf51d11741'
    with p.open('rb') as f:base=cloudpickle.load(f)
    assert base.mol.natm==4 and base.mol.nao==56
    triplets=base.trip;ranks=[len(x) for x in base.e_tno]
    rng=np.random.default_rng(650)
    raw=[.01*rng.normal(size=(n,n,n)) for n in ranks]*2
    seed=np.cos(np.arange(len(triplets)));runs=[];reference=None
    count=len(triplets)
    for mode in ['legacy','numpy','cupy','numpy','cupy','legacy']:
        leaves=[ad.Variable(x) for x in raw]
        cp.cuda.get_current_stream().synchronize();begin=time.perf_counter()
        energy=(scalar(triplets,leaves[:count],leaves[count:]) if mode=='legacy' else
                energy_vector(triplets,leaves[:count],leaves[count:],backend=mode))
        graph=ad.Linearization([energy],leaves)
        cp.cuda.get_current_stream().synchronize();forward=time.perf_counter()-begin
        numeric=energy.v.copy();begin=time.perf_counter()
        bars=graph([seed],release_graph=True,release_inputs=True)
        cp.cuda.get_current_stream().synchronize();reverse=time.perf_counter()-begin
        if reference is None:reference=(numeric,[x.copy() for x in bars])
        ve=float(np.max(abs(numeric-reference[0]),initial=0.))
        ge=max(float(np.max(abs(a-b),initial=0.)) for a,b in zip(bars,reference[1]))
        assert ve<1e-11 and ge<1e-11
        runs.append(dict(mode=mode,forward_seconds=forward,reverse_seconds=reverse,complete_seconds=forward+reverse,
            value_error=ve,vjp_error=ge,records=graph.statistics['records'],value_bytes=graph.statistics['value_bytes']))
        del bars,graph,energy,leaves
    median={mode:float(np.median([x['complete_seconds'] for x in runs if x['mode']==mode])) for mode in ('legacy','numpy','cupy')}
    print('TRIPLES_ENERGY_MICRO '+json.dumps(dict(complete=True,atoms=4,ao=56,triplets=count,
        tensor_elements=sum(n**3 for n in ranks),synthetic_values=True,shape_source=str(p),runs=runs,median_seconds=median,
        median_saving_seconds={mode:median['legacy']-median[mode] for mode in ('numpy','cupy')},
        molecular_gradients=0,new_scf_calculations=0,new_energy_calculations=0,
        limitation='Only final energy tensor contraction and source bars with synthetic values at saved small-system shapes.')))
