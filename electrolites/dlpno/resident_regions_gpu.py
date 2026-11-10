"""Complete resident PNO-integral and triples-source regions on selected GPUs.

Each task receives immutable numerical boundary values once. It retains
its complete local AD graph and the bounded triples workspace until the
final density pullback. RI metric, orbital, subspace and amplitude sources
all return through shared parent records. No response is frozen.
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import count
import atexit
import multiprocessing
import os
import threading
import time
from types import MethodType,SimpleNamespace
import weakref
import numpy as np
import cupy as cp
from . import reverse as ad
from .deferred_reads import OperandBank,capture_reads
from .parallel_local_gpu import Capture,_decode
from .parallel_sources_gpu import prepare_task,_restore

_worker=threading.local()
_ids=count()
_pools=[]


def _initialize(device,budget):
    import pyscf.__config__ as config
    config.num_devices=1
    _worker.device=device;_worker.regions={}
    with cp.cuda.Device(device):
        _worker.pool=cp.cuda.MemoryPool();_worker.pool.set_limit(size=budget)
        _worker.stream=cp.cuda.Stream(non_blocking=True)


def _flatten_pair(outputs):
    flat,seen,layout=[],{},[]
    for key,value in outputs.items():
        sequence=isinstance(value,list)
        indices=[]
        for item in value if sequence else [value]:
            token=id(item)
            if token not in seen:seen[token]=len(flat);flat.append(item)
            indices.append(seen[token])
        layout.append((key,sequence,indices))
    return tuple(flat),layout


def prepare_pno(state,pairs):
    from . import setup as su
    cap=Capture();views=[]
    for p in pairs:
        p=int(p);i,j=state.pl.pairs[p];q=int(state.pl.ji[p])
        aux=np.asarray(state.ribfs[p]).copy();paos=np.asarray(state.pao[p]).copy();lm=np.asarray(state.lmos[p]).copy()
        pair_ids={p};ij={}
        if state.strong[p]:
            for k in lm:
                for key in [(int(k),int(j)),(int(i),int(k))]:ij[key]=int(state.pl.ij[key]);pair_ids.add(ij[key])
            ext=su.union(paos,*[state.dom.lmo_paos[k] for k in lm])
        else:ext=paos
        ppaos={k:np.asarray(state.pao[k]).copy() for k in pair_ids}
        qijkeys=[(aux,int(i),int(j)),np.ix_(aux,[i],lm),np.ix_(aux,[j],lm)]
        qiakeys=[np.ix_(aux,[i],paos),np.ix_(aux,[j],paos),np.ix_(aux,lm,paos)]
        qabkeys=[np.ix_(aux,paos,paos)]
        if state.strong[p]:qiakeys.append(np.ix_(aux,lm,ext));qabkeys.append(np.ix_(aux,paos,ext))
        view=SimpleNamespace(pl=SimpleNamespace(pairs={p:(int(i),int(j))},ji={p:q},ij=ij),
          orb=SimpleNamespace(),ribfs={p:aux},pao=ppaos,lmos={p:lm},strong={p:bool(state.strong[p])},
          dom=SimpleNamespace(lmo_paos={int(k):np.asarray(state.dom.lmo_paos[k]).copy() for k in lm}),
          X={k:cap.value(state.X[k]) for k in pair_ids},K={p:cap.value(state.K[p])},
          qij=cap.selection(state.qij,qijkeys),qia=cap.selection(state.qia,qiakeys),
          qab=cap.selection(state.qab,qabkeys),J2c=cap.selection(state.J2c,[np.ix_(aux,aux)]))
        views.append((p,view))
    return dict(kind='pno_integrals',views=views),tuple(cap.inputs)


class _Region:
    def __init__(self,spec,values,flags):
        self.inputs=();self.workspaces={};self.outputs=();self.graph=None;self.closed=False
        token=ad._replay_workspaces.set(self.workspaces)
        try:
            self.inputs=tuple(ad.Variable(cp.asarray(v),tracked=flag) for v,flag in zip(values,flags))
            bank=OperandBank(self.inputs)
            if spec.get('kind')=='pno_integrals':
                from .ref_rhf import RefDLPNO
                from .adjoint import _names
                from .reference_cpu import rebind
                evaluate=rebind(RefDLPNO._pno_pair_integrals,**dict(_names,np=ad.Numpy(cp)))
                outputs=[];self.layout=[]
                for p,encoded in spec['views']:
                    flat,layout=_flatten_pair(evaluate(_decode(encoded,bank),p))
                    offset=len(outputs);outputs.extend(flat)
                    self.layout.append((p,[(key,seq,[offset+i for i in indices]) for key,seq,indices in layout]))
                self.outputs=tuple(outputs)
            else:
                from .triples_source_batch_gpu import source_batch
                self.outputs=source_batch(_restore(spec,bank),spec['triplets'],energy_only=spec['energy_only'],
                    max_workspace_bytes=spec['workspace_bytes'],reuse_reads=spec['reuse_reads'])
                if self.outputs is None:raise MemoryError('Resident triples task exceeds its workspace bound')
                self.layout=None
            self.shapes=tuple(x.shape for x in self.outputs)
            self.offsets=np.cumsum([0]+[x.size for x in self.outputs])
            self.graph=ad.Linearization(self.outputs,self.inputs)
            self.statistics=dict(self.graph.statistics,workspace_entries=len(self.workspaces),deferred_reads=dict(bank.statistics))
        except BaseException:
            self.release();raise
        finally:ad._replay_workspaces.reset(token)

    def forward(self):
        return cp.asnumpy(cp.concatenate([ad.primal(x).ravel() for x in self.outputs])),self.shapes,self.layout,self.statistics

    def reverse(self,seed,needed,consume):
        if self.closed:raise RuntimeError('A released resident source region cannot be reused')
        z=cp.asarray(seed)
        seeds=[z[a:b].reshape(shape) for a,b,shape in zip(self.offsets[:-1],self.offsets[1:],self.shapes)]
        graph=ad.Linearization(self.outputs,[x for x,flag in zip(self.inputs,needed) if flag],boundary=self.inputs)
        token=ad._replay_workspaces.set(self.workspaces)
        try:
            bars=graph(seeds,release_graph=consume,release_inputs=consume)
            counts=[b.size for b in bars];shapes=[b.shape for b in bars]
            packed=cp.asnumpy(cp.concatenate([b.ravel() for b in bars])) if bars else np.empty(0)
            offsets=np.cumsum([0]+counts)
            pieces=iter(packed[a:b].reshape(sh) for a,b,sh in zip(offsets[:-1],offsets[1:],shapes))
            return tuple(next(pieces) if flag else None for flag in needed)
        finally:
            ad._replay_workspaces.reset(token)
            if consume:self.release()

    def release(self):
        self.workspaces.clear();self.inputs=self.outputs=();self.graph=None;self.closed=True


def _task(method,key,*args):
    begin=time.perf_counter()
    with cp.cuda.Device(_worker.device),_worker.stream,cp.cuda.using_allocator(_worker.pool.malloc):
        before=_worker.pool.used_bytes()
        if method=='build':
            region=_Region(*args)
            try:result=region.forward()
            except BaseException:region.release();raise
            _worker.regions[key]=region
        elif method=='reverse':
            region=_worker.regions[key]
            try:result=region.reverse(*args)
            finally:
                if region.closed:del _worker.regions[key]
        elif method=='close':
            for region in _worker.regions.values():region.release()
            _worker.regions.clear()
            import gc
            gc.collect();_worker.stream.synchronize();_worker.pool.free_all_blocks();result=None
        else:raise ValueError(method)
        _worker.stream.synchronize()
        info=dict(device=_worker.device,pid=os.getpid(),seconds=time.perf_counter()-begin,
          live_bytes=_worker.pool.used_bytes(),retained_delta_bytes=_worker.pool.used_bytes()-before,
          regions=len(_worker.regions),reserved_bytes=_worker.pool.total_bytes())
        return result,info


class ResidentPool:
    def __init__(self,devices,*,budget_bytes=8<<30,executor='thread'):
        self.devices=tuple(devices);self.closed=False;self.workers=[];self.loads=[0]*len(devices)
        if not self.devices or len(set(devices))!=len(devices) or min(devices)<0 or max(devices)>=cp.cuda.runtime.getDeviceCount():
            raise ValueError('Resident source devices are invalid')
        if executor not in ('thread','process'):raise ValueError(executor)
        self.statistics=dict(devices=list(devices),executor=executor,budget_per_device_bytes=budget_bytes,
          build_seconds=0.,reverse_seconds=0.,build_calls=0,reverse_calls=0,released=False)
        for d in devices:
            options=dict(max_workers=1,initializer=_initialize,initargs=(d,budget_bytes))
            if executor=='process':options['mp_context']=multiprocessing.get_context('spawn')
            cls=ProcessPoolExecutor if executor=='process' else ThreadPoolExecutor
            self.workers.append(cls(**options))
        _pools.append(self)

    def run(self,tasks):
        if self.closed:raise RuntimeError('Resident pool already closed')
        start=time.perf_counter()
        if self.workers:
            futures=[self.workers[rank].submit(_task,method,key,*args) for rank,method,key,args in tasks]
            results=[];errors=[]
            for future in futures:
                try:results.append(future.result())
                except BaseException as error:errors.append(error)
            if errors:raise errors[0]
        else:results=[_task(method,key,*args) for rank,method,key,args in tasks]
        phase=tasks[0][1] if tasks else 'build'
        if phase in ('build','reverse'):
            self.statistics[phase+'_seconds']+=time.perf_counter()-start
            self.statistics[phase+'_calls']+=1
        return results

    def close(self):
        if self.closed:return self.statistics.get('released_workers',[])
        evidence=[]
        try:
            evidence=[info for _,info in self.run([(i,'close',None,()) for i in range(len(self.devices))])]
        finally:
            for w in self.workers:w.shutdown(wait=True,cancel_futures=True)
            self.closed=True;self.statistics.update(released=True,released_workers=evidence)
        return evidence


class RegionPlan:
    def __init__(self,prepared,pool,*,owner=None):
        self.pool=pool;self.owner=None if owner is None else weakref.ref(owner);self.closed=False
        start=time.perf_counter();self.inputs=[];identity={};self.mappings=[];self.tasks=[]
        for spec,operands in prepared:
            mapping=[]
            for operand in operands:
                token=('record',operand._record) if isinstance(operand,ad.Variable) else ('array',id(operand))
                if token not in identity:identity[token]=len(self.inputs);self.inputs.append(operand)
                mapping.append(identity[token])
            self.mappings.append(tuple(mapping))
        self.inputs=tuple(self.inputs)
        self.values=tuple(cp.asnumpy(ad.primal(x)) if ad._xp([x]) is cp else np.asarray(ad.primal(x)) for x in self.inputs)
        if sum(x.nbytes for x in self.values)>2<<30:raise MemoryError('Resident region host boundary exceeds 2 GiB')
        all_flags=tuple(isinstance(x,ad.Variable) and x._record.tracked for x in self.inputs)
        self.assignments=[]
        for (spec,_),mapping in zip(prepared,self.mappings):
            rank=min(range(len(pool.loads)),key=lambda k:pool.loads[k]);key=next(_ids)
            weight=spec.get('resident_cost',sum(self.values[i].nbytes for i in mapping));pool.loads[rank]+=weight
            self.assignments.append((rank,key))
            self.tasks.append((rank,'build',key,(spec,tuple(self.values[i] for i in mapping),tuple(all_flags[i] for i in mapping))))
        self.statistics=dict(prepare_seconds=time.perf_counter()-start,reverse_seconds=0.,reverse_calls=0,
          tasks=len(self.tasks),input_bytes=sum(x.nbytes for x in self.values),coefficients_complete=True)
        begin=time.perf_counter();built=pool.run(self.tasks);self.tasks=None
        self.statistics.update(forward_seconds=time.perf_counter()-begin,workers=[i for _,i in built],worker_graphs=[x[3] for x,_ in built])
        self.parts=[x[0] for x,_ in built];self.shapes=tuple(shape for x,_ in built for shape in x[1])
        self.layouts=[x[2] for x,_ in built];self.task_sizes=[len(x[1]) for x,_ in built]
        self.task_offsets=np.cumsum([0]+[x.size for x in self.parts])
        self.output_offsets=np.cumsum([0]+[int(np.prod(shape)) for shape in self.shapes])

    def bridge(self):
        packed=np.concatenate(self.parts);self.parts=None
        def reverse(seed,needed):
            if self.closed:raise RuntimeError('A consumed resident region cannot be reused')
            owner=None if self.owner is None else self.owner()
            consume=owner is not None and getattr(owner,'_release_transposes',False)
            start=time.perf_counter()
            tasks=[]
            for (rank,key),mapping,a,b in zip(self.assignments,self.mappings,self.task_offsets[:-1],self.task_offsets[1:]):
                tasks.append((rank,'reverse',key,(np.asarray(seed[a:b]),tuple(needed[i] for i in mapping),consume)))
            parts=self.pool.run(tasks);result=[None]*len(self.inputs)
            for mapping,(bars,_) in zip(self.mappings,parts):
                for index,bar in zip(mapping,bars):
                    if bar is not None:
                        if result[index] is None:result[index]=bar.copy()
                        else:result[index]+=bar
            self.statistics['reverse_calls']+=1;self.statistics['reverse_seconds']+=time.perf_counter()-start
            self.statistics['last_reverse_workers']=[info for _,info in parts]
            if consume:self.closed=True
            return tuple(np.zeros_like(self.values[i]) if flag and result[i] is None else result[i] for i,flag in enumerate(needed))
        output=ad.operation(packed,self.inputs,lambda seed:reverse(seed,[True]*len(self.inputs)),reverse)
        return tuple(output[a:b].reshape(shape) for a,b,shape in zip(self.output_offsets[:-1],self.output_offsets[1:],self.shapes))


def _pool(state):
    if not hasattr(state,'_resident_region_pool'):
        devices=tuple(map(int,os.environ['DLPNO_RESIDENT_REGION_DEVICES'].split(',')))
        memory=int(os.environ.get('DLPNO_RESIDENT_REGION_MEMORY_MIB','8192'))<<20
        if not 32<<20<=memory<=16<<30:raise ValueError('Resident region budget must be 32..16384 MiB per device')
        pool_class=ResidentPool
        if os.environ.get('DLPNO_RESIDENT_SHARED_STORE','0')=='1':
            from .resident_shared_store_gpu import SharedResidentPool
            pool_class=SharedResidentPool
        state._resident_region_pool=pool_class(devices,budget_bytes=memory,executor=os.environ.get('DLPNO_RESIDENT_REGION_EXECUTOR','thread'))
        state.diagnostics['resident_region_pool']=state._resident_region_pool.statistics
        if os.environ.get('DLPNO_RESIDENT_OVERLAP_LIFECYCLE','0')=='1':
            from .resident_lifecycle import register
            register(state._resident_region_pool)
    return state._resident_region_pool


def _plan(prepared,pool,owner):
    if pool.statistics.get('shared_store'):
        from .resident_shared_store_gpu import SharedRegionPlan
        return SharedRegionPlan(prepared,pool,owner=owner)
    return RegionPlan(prepared,pool,owner=owner)


def pno_integrals(state):
    names=('K_mibj','J_ijmb','K_ivvv','L_iajb','J_ikac','K_iakc','iQk','iQa','Qma','Qab')
    for name in names:setattr(state,name,[None]*len(state.pl))
    pairs=[p for p,(i,j) in enumerate(state.pl.pairs) if i<=j]
    batch=int(os.environ.get('DLPNO_RESIDENT_REGION_PNO_BATCH','4'))
    if not 1<=batch<=32:raise ValueError('Resident PNO batch must be 1..32')
    with capture_reads(os.environ.get('DLPNO_RESIDENT_DEFER_READS','0')=='1') as deferred:
        prepared=[prepare_pno(state,pairs[a:a+batch]) for a in range(0,len(pairs),batch)]
    plan=_plan(prepared,_pool(state),state)
    plan.statistics['deferred_reads']=deferred
    outputs=plan.bridge();offset=0
    for layout,size in zip(plan.layouts,plan.task_sizes):
        for p,entries in layout:
            for (name,index),sequence,indices in entries:
                values=[outputs[offset+i] for i in indices]
                getattr(state,name)[index]=values if sequence else values[0]
        offset+=size
    plan.statistics['kind']='pno_integrals'
    state.diagnostics.setdefault('resident_regions',[]).append(plan.statistics)
    state._resident_region_plans.append(plan)


def triples(state,save=False):
    from .triples_source_batch_gpu import source_batches,workspace_bytes,_layout
    count=int(os.environ.get('DLPNO_TRIPLES_SOURCE_BATCH','8'))
    workspace=int(os.environ.get('DLPNO_TRIPLES_SOURCE_WORKSPACE_MIB','2048'))<<20
    if not 1<=count<=64 or not 1<<20<=workspace<=2<<30:raise ValueError('Resident triples batch/workspace outside bounds')
    groups=source_batches(state,max_count=count,max_bytes=workspace)
    with capture_reads(os.environ.get('DLPNO_RESIDENT_DEFER_READS','0')=='1') as deferred:
        prepared=[prepare_task(state,group,not save,workspace,True) for group in groups]
    for group,(spec,_) in zip(groups,prepared):spec['resident_cost']=workspace_bytes([_layout(state,t) for t in group])
    plan=_plan(prepared,_pool(state),state);plan.statistics['deferred_reads']=deferred
    output=plan.bridge();stride=4 if save else 1
    energies=[None]*len(state.trip)
    if save:state.W3,state.V3,state.T3=([None]*len(state.trip) for _ in range(3))
    for pos,t in enumerate(t for group in groups for t in group):
        energies[t]=output[pos*stride]
        if save:state.W3[t],state.V3[t],state.T3[t]=output[pos*stride+1:pos*stride+4]
    state.e_ijk=state.numeric.array(energies)
    plan.statistics['kind']='triples'
    state.diagnostics.setdefault('resident_regions',[]).append(plan.statistics)
    state._resident_region_plans.append(plan)
    return np.sum(state.e_ijk)


def configure(state):
    state._resident_region_plans=[]
    state.pno_integrals=MethodType(pno_integrals,state)
    state.lccsd_t0=MethodType(triples,state)
    if os.environ.get('DLPNO_RESIDENT_EARLY_START','0')=='1':
        pool=_pool(state)
        if not hasattr(pool,'prestart'):raise ValueError('Early start requires the shared resident store')
        pool.prestart()


def finish(state):
    pool=getattr(state,'_resident_region_pool',None)
    if pool is not None:
        from .resident_lifecycle import owns
        if owns(pool) and hasattr(pool,'begin_close') and pool.statistics.get('executor')=='process' and len(pool.devices)>1:
            pool.begin_close()
        else:pool.close()
        for plan in state._resident_region_plans:
            plan.closed=True;plan.values=();plan.inputs=()
            plan.statistics['released']=True


def close_pools():
    results=[]
    for pool in list(_pools):
        results.extend(pool.close())
        _pools.remove(pool)
    return results


atexit.register(close_pools)
