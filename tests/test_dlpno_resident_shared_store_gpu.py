"""Resident complete regions: CPU equations, selected VJPs, FD and ownership."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import _names,host
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.reference_cpu import rebind
from electrolites.dlpno.resident_regions_gpu import prepare_pno,_flatten_pair
from electrolites.dlpno.resident_shared_store_gpu import SharedResidentPool as ResidentPool,SharedRegionPlan as RegionPlan,close_pools
from electrolites.dlpno.parallel_sources_gpu import prepare_task
from test_dlpno_triples_source_batch_gpu import multi_state
from test_dlpno_triples_gpu_sources import raw_sources
from test_dlpno_parallel_local_gpu import raw_data,make_state


from electrolites.dlpno.deferred_reads import DeferredRead,capture_reads,active
from electrolites.dlpno.local_blocks import SelectionTensor
from electrolites.dlpno.local_integrals import index_arrays


def defer_sources(state):
 def wrap(source):
  def get(key):
   if not active():return source[key]
   shape=index_arrays(key,source.shape)[0].shape
   return DeferredRead((source,),((key,Ellipsis),),shape,shape)
  return SelectionTensor(source.shape,get)
 for name in ('qij','qia','qab'):setattr(state,name,wrap(getattr(state,name)))
 for name in ('S_pao','F_pao'):
  if hasattr(state.orb,name):setattr(state.orb,name,wrap(getattr(state.orb,name)))


class Owner:pass


def pno_data():
 raw,rng=raw_data()
 raw['qo']=.1*rng.normal(size=(8,3,3));raw['qab']=.1*rng.normal(size=(8,6,6))
 return raw,rng


def pno_state(values):
 s,_=make_state(values)
 s.qij=(values['qo']+values['qo'].transpose(0,2,1))*.5
 s.qab=(values['qab']+values['qab'].transpose(0,2,1))*.5
 s.dom=SimpleNamespace(lmo_paos=[np.arange(6) for _ in range(3)])
 s.strong[[1,3]]=False
 return s


def pno_oracle(values):
 s=pno_state(values);evaluate=rebind(RefDLPNO._pno_pair_integrals,**_names)
 return tuple(x for p in [0,1,2,4,5,8] for x in _flatten_pair(evaluate(s,p))[0])


@pytest.mark.parametrize('defer',[False,True])
@pytest.mark.parametrize('kind',['pno','triples','triples_energy'])
@pytest.mark.parametrize('devices,executor',[((0,),'thread'),((0,1),'thread'),((0,1),'process')])
def test_complete_resident_regions(kind,devices,executor,defer):
 import cupy as cp
 assert cp.cuda.runtime.getDeviceCount()>=2
 raw,rng=pno_data() if kind=='pno' else raw_sources()
 variables={k:ad.Variable(v.copy()) for k,v in raw.items()};inputs=list(variables.values())
 if kind=='pno':
  oracle=pno_oracle;state=pno_state(variables)
  if defer:defer_sources(state)
  with capture_reads(defer) as capture_statistics:
   prepared=[prepare_pno(state,pairs) for pairs in [(0,1),(2,4),(5,8)]]
 else:
  energy_only=kind=='triples_energy';count=1 if energy_only else 4
  equation=rebind(RefDLPNO._triplet_t0,**_names)
  def oracle(values):
   state=multi_state(values)
   return tuple(v for t in range(4) for v in equation(state,t)[:count])
  state=multi_state(variables)
  if defer:defer_sources(state)
  with capture_reads(defer) as capture_statistics:
   prepared=[prepare_task(state,group,energy_only,64<<20,True) for group in [(0,1),(2,3)]]
 if defer:assert capture_statistics['reads']>0
 expected=oracle(variables);seeds=[rng.normal(size=x.shape) for x in expected]
 gold=ad.Linearization(expected,inputs)(seeds)
 owner=Owner();owner._release_transposes=False
 pool=ResidentPool(devices,budget_bytes=128<<20,executor=executor)
 try:
  plan=RegionPlan(prepared,pool,owner=owner);outputs=plan.bridge()
  assert len(outputs)==len(expected)
  assert plan.statistics['uploaded_input_bytes']<=plan.statistics['logical_task_input_bytes']
  value_error=max(float(np.max(np.abs(host(a)-host(b)),initial=0.)) for a,b in zip(outputs,expected))
  assert value_error<2e-10,(kind,value_error)
  graph=ad.Linearization(outputs,inputs);bars=graph(seeds)
  error=max(float(np.max(np.abs(a-b),initial=0.)) for a,b in zip(bars,gold));assert error<2e-9,(kind,error)
  replay=graph([.37*z for z in seeds]);assert max(np.max(abs(a-.37*b),initial=0.) for a,b in zip(replay,gold))<2e-9
  names=('M','q','qo','qab','X0','K0') if kind=='pno' else ('metric','overlap','xt','eps','fock','t0')
  subset=[variables[n] for n in names]
  selected=ad.Linearization(outputs,subset,boundary=inputs)(seeds)
  assert max(np.max(abs(a-gold[list(variables).index(n)]),initial=0.) for a,n in zip(selected,names))<2e-9
  directions={k:.05*rng.normal(size=v.shape) for k,v in raw.items()};h=2e-5;vals=[]
  for sign in (1,-1):
   displaced={k:v+sign*h*directions[k] for k,v in raw.items()}
   vals.append(sum(float(np.sum(host(x)*z)) for x,z in zip(oracle(displaced),seeds)))
  fd=(vals[0]-vals[1])/(2*h);analytic=sum(float(np.sum(b*directions[k])) for k,b in zip(variables,bars))
  assert abs(fd-analytic)<3e-8,(kind,fd,analytic)
  owner._release_transposes=True
  final=graph(seeds,release_graph=True,release_inputs=True)
  assert max(np.max(abs(a-b),initial=0.) for a,b in zip(final,gold))<2e-9
  assert all(x.v is None for x in inputs)
  assert {x['device'] for x in plan.statistics['workers']}==set(devices)
  print('SHARED_RESIDENT_REPORT '+json.dumps(dict(kind=kind,devices=devices,executor=executor,defer=defer,capture_statistics=capture_statistics,
   value_error=value_error,vjp_error=error,directional_fd_error=abs(fd-analytic),statistics=plan.statistics)))
 finally:cleanup=pool.close()
 assert len(cleanup)==len(devices) and all(x['live_bytes']==0 for x in cleanup),cleanup


def test_memory_cap_and_recovery():
 raw,_=raw_sources();variables={k:ad.Variable(v) for k,v in raw.items()}
 prepared=[prepare_task(multi_state(variables),(0,1),False,64<<20,True)]
 pool=ResidentPool((0,),budget_bytes=512)
 try:
  with pytest.raises(MemoryError):RegionPlan(prepared,pool)
 finally:cleanup=pool.close()
 assert all(x['live_bytes']==0 for x in cleanup),cleanup
 # A new pool must remain usable after the bounded failure.
 pool=ResidentPool((0,),budget_bytes=64<<20)
 try:
  plan=RegionPlan(prepared,pool);assert plan.bridge()
 finally:cleanup=pool.close()
 assert all(x['live_bytes']==0 for x in cleanup),cleanup
 close_pools()


def test_shared_sources_survive_first_consumed_region_and_reuse_across_stages():
 raw,rng=pno_data();variables={k:ad.Variable(v.copy()) for k,v in raw.items()}
 inputs=list(variables.values());state=pno_state(variables);defer_sources(state)
 with capture_reads():prepared=[prepare_pno(state,group) for group in [(0,1),(2,4),(5,8)]]
 expected=pno_oracle(variables);seeds=[rng.normal(size=x.shape) for x in expected]
 gold=ad.Linearization(expected,inputs)(seeds)
 first_owner=Owner();second_owner=Owner();first_owner._release_transposes=True;second_owner._release_transposes=True
 pool=ResidentPool((0,),budget_bytes=128<<20)
 try:
  first=RegionPlan(prepared,pool,owner=first_owner);out1=first.bridge()
  second=RegionPlan(prepared,pool,owner=second_owner);out2=second.bridge()
  assert second.statistics['uploaded_input_bytes']==0
  assert sum(g['deferred_reads']['materialized'] for g in second.statistics['worker_graphs'])==0
  assert sum(g['deferred_reads']['cache_hits'] for g in second.statistics['worker_graphs'])>0
  bars1=ad.Linearization(out1,inputs)(seeds)
  assert first.closed and not second.closed
  bars2=ad.Linearization(out2,inputs)([.37*z for z in seeds])
  assert second.closed
  error=max(np.max(abs(a-b),initial=0.) for a,b in zip(bars1,gold))
  error=max(error,max(np.max(abs(a-.37*b),initial=0.) for a,b in zip(bars2,gold)))
  assert error<2e-9,error
  info=second.statistics['last_reverse_workers'][0]
  assert info['store']['forward_transfers']==2 and info['store']['reverse_transfers']==2
  print('SHARED_LIFETIME_REPORT '+json.dumps(dict(maximum_error=float(error),second_stage_upload_bytes=0,second_stage_materializations=0,worker=info)))
 finally:cleanup=pool.close()
 assert all(x['live_bytes']==0 for x in cleanup),cleanup
