"""Distributed sources accept resident TNO parents and restore their bars."""
import json
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import _names,host
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.reference_cpu import rebind
from test_dlpno_triples_source_batch_gpu import multi_state
from test_dlpno_triples_gpu_sources import raw_sources


@pytest.mark.parametrize('energy_only',[False,True])
@pytest.mark.parametrize('devices',[(0,),(0,1)])
def test_resident_tno_parallel_source_boundary(energy_only,devices):
    import cupy as cp
    from electrolites.dlpno.parallel_sources_gpu import SourcePool,source_window
    raw,rng=raw_sources();cpu={k:ad.Variable(v.copy()) for k,v in raw.items()}
    variables={k:ad.Variable((cp if k in ('xt','eps') else np).asarray(v)) for k,v in raw.items()}
    oracle=rebind(RefDLPNO._triplet_t0,**_names);count=1 if energy_only else 4
    expected=tuple(v for t in range(4) for v in oracle(multi_state(cpu),t)[:count])
    seeds=[rng.normal(size=x.shape) for x in expected];gold=ad.Linearization(expected,list(cpu.values()))(seeds)
    pool=SourcePool(devices,'thread')
    try:
        out,info=source_window(multi_state(variables),((0,1),(2,3)),devices=devices,energy_only=energy_only,pool=pool)
        graph=ad.Linearization(out,list(variables.values()));bars=graph(seeds)
        error=lambda a,b:max(float(np.max(abs(host(x)-host(y)),initial=0.)) for x,y in zip(a,b))
        value_error,vjp_error=error(out,expected),error(bars,gold)
        assert value_error<2e-11 and vjp_error<3e-10
        assert all(ad._xp([bar]) is ad._xp([leaf]) for leaf,bar in zip(variables.values(),bars))
        assert error(graph([.37*z for z in seeds]),[.37*b for b in gold])<3e-10
        direction={k:.1*rng.normal(size=v.shape) for k,v in raw.items()};h=2e-5;targets=[]
        for sign in (1,-1):
            state=multi_state({k:v+sign*h*direction[k] for k,v in raw.items()})
            values=tuple(v for t in range(4) for v in oracle(state,t)[:count])
            targets.append(sum(float(np.sum(host(v)*z)) for v,z in zip(values,seeds)))
        finite=(targets[0]-targets[1])/(2*h)
        analytic=sum(float(np.sum(host(bar)*direction[k])) for k,bar in zip(variables,bars))
        assert abs(finite-analytic)<2e-8
        assert error(graph(seeds,release_graph=True,release_inputs=True),gold)<3e-10
        print('RESIDENT_SOURCE_REPORT '+json.dumps(dict(devices=devices,energy_only=energy_only,
            value_error=value_error,vjp_error=vjp_error,fd_error=abs(finite-analytic))))
    finally:cleanup=pool.close()
    assert all(x['live_device_bytes']==0 for x in cleanup)
