"""Persistent canonical inputs and deferred reads across complete local regions.

A worker owns each immutable source leaf and each canonical gather once.
Region graphs stop at those shared leaves. Their cotangents are summed on
that worker before the exact gather pullback and one packed host transfer.
The parent still owns the full dependency graph and deterministic reduction
across devices. No amplitude, metric, orbital or subspace response is frozen.
"""
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,Future
from itertools import count
from dataclasses import replace
import atexit,multiprocessing,os,threading,time,weakref
import numpy as np
import cupy as cp
from . import reverse as ad
from .deferred_reads import EncodedRead,_index
from .parallel_local_gpu import _decode
from .parallel_sources_gpu import _restore

_local=threading.local();_ids=count();_pools=[]

def host_budget_bytes():
    budget=int(os.environ.get('DLPNO_RESIDENT_HOST_MEMORY_MIB','2048'))<<20
    if not 32<<20<=budget<=64<<30:
        raise ValueError('Shared resident host boundary budget must be 32..65536 MiB')
    return budget



class _Bank:
    def __init__(self,store,keys,*,recording=True):
        self.store=store;self.keys=tuple(keys);self.reads={};self.recording=bool(recording)
        self.inputs=tuple(store.inputs[k] if recording else
                          ad.Variable(ad.primal(store.inputs[k]),tracked=False) for k in keys)
        self.statistics=dict(materialized=0,cache_hits=0,output_bytes=0)

    def __getitem__(self,index):
        return self.read(index) if isinstance(index,EncodedRead) else self.inputs[_index(index)]

    def read(self,encoded):
        keys=tuple(self.keys[_index(i)] for i in encoded.sources)
        signature=(encoded.cartesian_shape,encoded.output_shape,
            tuple((key,)+part[1:] for key,part in zip(keys,encoded.signature[2])))
        if signature not in self.store.reads:
            descriptor=replace(encoded,sources=keys,signature=signature)
            sources=tuple(self.store.inputs[k] for k in keys)
            value=descriptor.evaluate(tuple(ad.primal(x) for x in sources))
            leaf=ad.Variable(ad.primal(value),tracked=any(x._record.tracked for x in sources))
            self.store.reads[signature]=(descriptor,leaf)
            self.statistics['materialized']+=1
            self.store.statistics['read_materializations']+=1
            self.store.statistics['read_value_bytes']+=leaf.size*8
        else:
            self.statistics['cache_hits']+=1;self.store.statistics['read_cache_hits']+=1
        descriptor,leaf=self.store.reads[signature]
        if not self.recording:leaf=ad.Variable(ad.primal(leaf),tracked=False)
        self.reads[signature]=leaf;self.statistics['output_bytes']+=leaf.size*8
        return leaf


class _Region:
    def __init__(self,spec,bank):
        self.bank=bank;self.inputs=bank.inputs;self.workspaces={};self.closed=False;self.outputs=()
        single_fallback=None
        token=ad._replay_workspaces.set(self.workspaces if bank.recording else None)
        try:
            if spec.get('kind')=='pno_integrals':
                from .ref_rhf import RefDLPNO
                from .adjoint import _names
                from .reference_cpu import rebind
                from .resident_regions_gpu import _flatten_pair
                evaluate=rebind(RefDLPNO._pno_pair_integrals,**dict(_names,np=ad.Numpy(cp)))
                outputs=[];self.layout=[]
                for p,encoded in spec['views']:
                    flat,layout=_flatten_pair(evaluate(_decode(encoded,bank),p))
                    offset=len(outputs);outputs.extend(flat)
                    self.layout.append((p,[(key,seq,[offset+i for i in indices]) for key,seq,indices in layout]))
                self.outputs=tuple(outputs)
            elif spec.get('kind') in ('pno_subspace','tno_subspace'):
                from .resident_subspaces_gpu import subspace_outputs
                self.outputs=subspace_outputs(spec,bank);self.layout=None
            elif spec.get('kind')=='ccsd_overlap_sources':
                from .overlap_union_adjoint_gpu import embedded_batch
                packed,_=embedded_batch(bank[spec['metric']],tuple(bank[x] for x in spec['coefficients']),
                    spec['positions'],spec['couples'],packed_output=True)
                self.outputs=(packed,);self.layout=None
            elif spec.get('kind')=='t1_overlap_sources':
                from .triples_overlap_adjoint_gpu import common_left_batch
                outputs=[]
                for group in spec['groups']:
                    outputs.extend(common_left_batch(bank[group['metric']],bank[group['left']],
                        tuple(bank[key] for key in group['right']),group['positions'],reuse_workspace=True))
                self.outputs=tuple(outputs);self.layout=None
            else:
                from .triples_source_batch_gpu import source_batch
                self.outputs=source_batch(_restore(spec,bank),spec['triplets'],energy_only=spec['energy_only'],
                    max_workspace_bytes=spec['workspace_bytes'],reuse_reads=spec['reuse_reads'])
                if self.outputs is None:
                    # The batch planner may emit one triplet whose padded
                    # batch envelope exceeds the limit. The unpadded exact
                    # single-triplet operation has a smaller envelope; retain
                    # the same explicit workspace limit and every AD edge.
                    if os.environ.get('DLPNO_RESIDENT_TRIPLES_SINGLE_FALLBACK') != '1':
                        raise MemoryError('Shared resident triples workspace exceeded')
                    if len(spec['triplets']) != 1:
                        raise MemoryError('Oversized shared triples batch is not a singleton')
                    from .triples_source_adjoint_gpu import triplet_source
                    from .triples_source_batch_gpu import _layout,workspace_bytes
                    view=_restore(spec,bank);t=spec['triplets'][0]
                    layout=_layout(view,t)
                    self.outputs=triplet_source(view,t,energy_only=spec['energy_only'],
                        max_workspace_bytes=spec['workspace_bytes'])
                    if self.outputs is None:
                        raise MemoryError('Unpadded shared triples source exceeds its workspace limit: '
                            f'triplet={t}, limit={spec["workspace_bytes"]}, layout={layout}')
                    single_fallback=dict(triplet=int(t),batch_workspace_bound_bytes=workspace_bytes([layout]),
                        workspace_limit_bytes=int(spec['workspace_bytes']),coefficients_complete=True,
                        method='exact_unpadded_single_source')
                self.layout=None
            self.read_keys=tuple(bank.reads)
            self.boundary=self.inputs+tuple(bank.reads[k] for k in self.read_keys)
            self.shapes=tuple(x.shape for x in self.outputs)
            self.offsets=np.cumsum([0]+[x.size for x in self.outputs])
            graph=ad.Linearization(self.outputs,self.boundary)
            self.connected_boundary=None
            if os.environ.get('DLPNO_RESIDENT_CONNECTED_SOURCES','0')=='1':
                # The canonical source and a deferred gather are separate
                # boundary leaves. An unconnected canonical leaf contributes
                # exactly zero; requesting it would allocate a full source bar.
                connected=set(graph.records)
                self.connected_boundary=tuple(x._record in connected for x in self.boundary)
            self.statistics=dict(graph.statistics,workspace_entries=len(self.workspaces),deferred_reads=dict(bank.statistics))
            if single_fallback is not None:self.statistics["triples_single_fallback"]=single_fallback
        except BaseException:
            self.release();raise
        finally:ad._replay_workspaces.reset(token)

    def reverse(self,seed,needed_keys,consume):
        if self.closed:raise RuntimeError('Shared region already consumed')
        z=cp.asarray(seed);seeds=[z[a:b].reshape(s) for a,b,s in zip(self.offsets[:-1],self.offsets[1:],self.shapes)]
        flags=[k in needed_keys for k in self.bank.keys]
        flags += [any(k in needed_keys for k in self.bank.store.reads[s][0].sources) for s in self.read_keys]
        if self.connected_boundary is not None:
            skipped=[x for x,requested,connected in zip(self.boundary,flags,self.connected_boundary)
                     if requested and not connected]
            statistics=self.bank.store.statistics
            statistics['disconnected_boundary_bars_skipped']=statistics.get('disconnected_boundary_bars_skipped',0)+len(skipped)
            statistics['disconnected_boundary_bytes_avoided']=statistics.get('disconnected_boundary_bytes_avoided',0)+sum(x.size*8 for x in skipped)
            flags=[requested and connected for requested,connected in zip(flags,self.connected_boundary)]
            del skipped
        graph=ad.Linearization(self.outputs,[x for x,flag in zip(self.boundary,flags) if flag],boundary=self.boundary)
        token=ad._replay_workspaces.set(self.workspaces)
        try:
            # Shared sources/read values are stops and outlive any one region.
            bars=iter(graph(seeds,release_graph=consume,release_inputs=False))
            keys=[('input',k) for k in self.bank.keys]+[('read',s) for s in self.read_keys]
            return [(key,next(bars)) for key,flag in zip(keys,flags) if flag]
        finally:
            ad._replay_workspaces.reset(token)
            if consume:self.release()

    def release(self):
        self.workspaces.clear();self.inputs=self.outputs=self.boundary=();self.closed=True;self.bank=None


class _Store:
    def __init__(self):
        self.inputs={};self.reads={};self.regions={}
        self.statistics=dict(input_uploads=0,input_upload_bytes=0,read_materializations=0,
            read_cache_hits=0,read_value_bytes=0,forward_host_bytes=0,reverse_host_bytes=0,
            forward_transfers=0,reverse_transfers=0)

    def build(self,new_inputs,tasks):
        for key,value,flag in new_inputs:
            if key in self.inputs:raise RuntimeError('Immutable input uploaded twice to one worker')
            self.inputs[key]=ad.Variable(cp.asarray(value),tracked=flag)
            self.statistics['input_uploads']+=1;self.statistics['input_upload_bytes']+=value.nbytes
        rows=[];pieces=[];offset=0
        for key,spec,input_keys in tasks:
            if key in self.regions:raise RuntimeError('Duplicate shared region identifier')
            before=_local.pool.used_bytes()
            region=_Region(spec,_Bank(self,input_keys));self.regions[key]=region
            region.statistics['retained_device_bytes']=_local.pool.used_bytes()-before
            count=int(region.offsets[-1]);pieces.extend(ad.primal(x).ravel() for x in region.outputs)
            rows.append((key,offset,offset+count,region.shapes,region.layout,region.statistics));offset+=count
        packed=cp.asnumpy(cp.concatenate(pieces)) if pieces else np.empty(0)
        self.statistics['forward_host_bytes']+=packed.nbytes;self.statistics['forward_transfers']+=bool(pieces)
        return packed,rows

    def reverse(self,tasks,needed_keys,consume):
        needed_keys=set(needed_keys);direct={};reads={}
        def add(table,key,value):
            if key in table:table[key]+=value
            else:table[key]=value.copy()
        for key,seed in tasks:
            region=self.regions[key]
            try:
                for (kind,index),bar in region.reverse(seed,needed_keys,consume):
                    add(direct if kind=='input' else reads,index,bar)
            finally:
                if region.closed:del self.regions[key]
        # Each cached gather is differentiated once after its region bars sum.
        for signature,bar in reads.items():
            descriptor,_=self.reads[signature];bar=bar.reshape(descriptor.cartesian_shape)
            for key,shape,(local,output) in zip(descriptor.sources,descriptor.source_shapes,descriptor.parts):
                if key not in needed_keys:continue
                value=bar[output]
                if local is None:add(direct,key,value)
                else:
                    if key not in direct:direct[key]=cp.zeros(shape,dtype=cp.float64)
                    cp.add.at(direct[key],local,value)
        keys=tuple(sorted(needed_keys));offsets=np.cumsum([0]+[self.inputs[k].size for k in keys])
        pieces=[direct[k].ravel() if k in direct else cp.zeros(self.inputs[k].size,dtype=cp.float64) for k in keys]
        packed=cp.asnumpy(cp.concatenate(pieces)) if pieces else np.empty(0)
        self.statistics['reverse_host_bytes']+=packed.nbytes;self.statistics['reverse_transfers']+=bool(pieces)
        return [(key,packed[a:b].reshape(self.inputs[key].shape)) for key,a,b in zip(keys,offsets[:-1],offsets[1:])]

    def discard(self,keys):
        for key in keys:
            region=self.regions.pop(key,None)
            if region is not None:region.release()
        import gc
        gc.collect()

    def ccsd_primal(self, snapshot):
        from .ccsd_primal_pipeline_gpu import solve_primal
        return solve_primal(snapshot)

    def close(self):
        for region in self.regions.values():region.release()
        self.regions.clear();self.reads.clear();self.inputs.clear()


def _initialize(device,budget,logical_device=None):
    import pyscf.__config__ as config
    config.num_devices=1
    _local.device=device
    _local.logical_device=device if logical_device is None else logical_device
    with cp.cuda.Device(device):
        _local.pool=cp.cuda.MemoryPool();_local.pool.set_limit(size=budget)
        _local.stream=cp.cuda.Stream(non_blocking=True)
        if os.environ.get('DLPNO_RESIDENT_CHECKPOINT_GRAPHS','0')=='1':
            from .resident_checkpoint_store_gpu import CheckpointStore
            _local.store=CheckpointStore()
        else:_local.store=_Store()


def _task(method,*args):
    begin=time.perf_counter()
    with cp.cuda.Device(_local.device),_local.stream,cp.cuda.using_allocator(_local.pool.malloc):
        if method=='close':
            _local.store.close()
            import gc
            gc.collect();_local.stream.synchronize();_local.pool.free_all_blocks();result=None
        elif method=='prepare':
            # Import and initialize libraries while primary PNO/MP2 work runs.
            # The worker was already isolated to its assigned physical device.
            from . import adjoint,triples_source_batch_gpu,resident_subspaces_gpu
            a=cp.eye(2,dtype=cp.float64)
            cp.linalg.eigh(a);cp.linalg.cholesky(a);a@a
            result=None
        elif method=='ready':result=None
        else:result=getattr(_local.store,method)(*args)
        _local.stream.synchronize()
        return result,dict(device=_local.logical_device,local_device=_local.device,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),pid=os.getpid(),seconds=time.perf_counter()-begin,
            live_bytes=_local.pool.used_bytes(),reserved_bytes=_local.pool.total_bytes(),
            regions=len(_local.store.regions),store=dict(_local.store.statistics),completed_at=time.perf_counter())


class _InlineWorker:
    """GPU0 uses the parent CUDA context, with an independent private store.

    Every call installs this worker's state and restores the caller's thread
    local state even on failure. Multiple live inline pools cannot alias.
    """
    def __init__(self,device,budget):
        self.device=device;self.budget=budget;self.state=None;self.closed=False

    def submit(self,function,*args):
        if self.closed:raise RuntimeError('Inline worker is closed')
        future=Future();saved=dict(_local.__dict__)
        try:
            _local.__dict__.clear()
            if self.state is None:
                _initialize(self.device,self.budget)
                self.state=dict(_local.__dict__)
            else:_local.__dict__.update(self.state)
            future.set_result(function(*args))
        except BaseException as error:future.set_exception(error)
        finally:
            _local.__dict__.clear();_local.__dict__.update(saved)
        return future

    def shutdown(self,wait=True,cancel_futures=True):
        self.closed=True;self.state=None


class _PrimaryThreadWorker:
    """Own GPU0's source store in one thread of the existing parent process."""
    def __init__(self,device,budget):
        self.worker=ThreadPoolExecutor(max_workers=1,initializer=_initialize,initargs=(device,budget))
    def submit(self,function,*args):
        return self.worker.submit(function,*args)
    def shutdown(self,wait=True,cancel_futures=True):
        return self.worker.shutdown(wait=wait,cancel_futures=cancel_futures)


class SharedResidentPool:
    def __init__(self,devices,*,budget_bytes=8<<30,executor='thread'):
        self.devices=tuple(devices);self.closed=False;self.workers=[];self.loads=[0]*len(devices)
        if executor not in ('thread','process'):raise ValueError(executor)
        isolate=executor=='process' and os.environ.get('DLPNO_RESIDENT_ISOLATE_DEVICES','0')=='1'
        inline=executor=='process' and os.environ.get('DLPNO_RESIDENT_INLINE_PRIMARY','0')=='1'
        threaded=inline and os.environ.get('DLPNO_RESIDENT_THREADED_PRIMARY','0')=='1'
        explicit=os.environ.get('DLPNO_RESIDENT_DEVICE_SELECTORS') if isolate else None
        selectors=explicit or os.environ.get('CUDA_VISIBLE_DEVICES')
        count=cp.cuda.runtime.getDeviceCount()
        selectors=selectors.split(',') if selectors is not None else [str(i) for i in range(count)]
        available=len(selectors) if explicit else count
        if not self.devices or len(set(devices))!=len(devices) or min(devices)<0 or max(devices)>=available:
            raise ValueError('Invalid shared resident devices')
        if isolate and any(d>=len(selectors) or not selectors[d].strip() for d in devices):
            raise ValueError('Cannot resolve each worker to an allocated CUDA device')
        if inline:
            primary=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0].strip()
            if devices[0]!=0 or selectors[0].strip()!=primary:
                raise ValueError('Inline primary must name the parent first visible device')
        self.identities={};self.references=[];self.values=[];self.flags=[];self.host_bytes=0;self.locations=[set() for _ in devices]
        self.host_arena=None;self.host_payloads=[]
        self.mapped_inputs=os.environ.get('DLPNO_RESIDENT_MMAP_INPUTS','0')=='1'
        self.mapped_threshold=int(os.environ.get('DLPNO_RESIDENT_MMAP_MIN_KIB','128'))<<10
        if self.mapped_inputs:
            if os.environ.get('DLPNO_RESIDENT_CHECKPOINT_GRAPHS','0')!='1':
                raise ValueError('Mapped inputs require checkpointed resident graphs')
            if not 0<=self.mapped_threshold<=64<<20:
                raise ValueError('Mapped host input threshold must be 0..65536 KiB')
            from .host_input_snapshots import HostSnapshotArena
            self.host_arena=HostSnapshotArena()
        self.statistics=dict(devices=list(devices),executor=executor,budget_per_device_bytes=budget_bytes,
            build_seconds=0.,reverse_seconds=0.,build_calls=0,reverse_calls=0,released=False,shared_store=True,
            isolated_process_devices=isolate,inline_primary=inline,threaded_primary=threaded,explicit_device_selectors=bool(explicit))
        self.statistics.update(mapped_host_inputs=self.mapped_inputs,
            mapped_host_input_bytes=0,mapped_host_input_count=0,
            mapped_host_minimum_bytes=self.mapped_threshold)
        for device in devices:
            if inline and device==0:
                self.workers.append((_PrimaryThreadWorker if threaded else _InlineWorker)(0,budget_bytes));continue
            options=dict(max_workers=1,initializer=_initialize,initargs=(device,budget_bytes))
            if isolate:
                from .resident_worker_bootstrap import initialize
                options.update(initializer=initialize,initargs=(selectors[device].strip(),device,budget_bytes))
            if executor=='process':options['mp_context']=multiprocessing.get_context('spawn')
            self.workers.append((ThreadPoolExecutor if executor=='thread' else ProcessPoolExecutor)(**options))
        self.prestart_futures=[];self.prestart_by_rank={}
        self._closing_executor=None;self._closing_future=None
        self._close_lock=threading.RLock()
        _pools.append(self)

    def prestart(self):
        if self.closed:raise RuntimeError('Cannot prestart a closed resident pool')
        if self.prestart_futures:return
        begin=time.perf_counter()
        self.statistics['startup_ready_seconds']=[None]*len(self.workers)
        ranks=sorted(range(len(self.workers)),key=lambda i:isinstance(self.workers[i],_InlineWorker))
        for rank in ranks:
            worker=self.workers[rank]
            method='prepare' if os.environ.get('DLPNO_RESIDENT_WARMUP','0')=='1' and not isinstance(worker,_InlineWorker) else 'ready'
            future=worker.submit(_task,method);self.prestart_by_rank[rank]=future
            self.statistics['remote_warmup']=os.environ.get('DLPNO_RESIDENT_WARMUP','0')=='1'
            def ready(future,rank=rank):
                try:self.statistics['startup_ready_seconds'][rank]=future.result()[1]['completed_at']-begin
                except BaseException as error:self.statistics.setdefault('startup_errors',[]).append(type(error).__name__+': '+str(error))
            future.add_done_callback(ready);self.prestart_futures.append(future)
        self.statistics['prestart_submit_seconds']=time.perf_counter()-begin
        self.statistics['early_start']=True

    def register(self,value):
        token=('record',value._record) if isinstance(value,ad.Variable) else ('array',id(value))
        if token not in self.identities:
            raw=ad.primal(value);raw=cp.asnumpy(raw) if isinstance(raw,cp.ndarray) else np.asarray(raw)
            if raw.dtype!=np.float64:raise TypeError('Shared resident boundary requires float64')
            if self.host_bytes+raw.nbytes>host_budget_bytes():raise MemoryError('Shared resident persistent host boundary exceeds its explicit budget')
            self.identities[token]=len(self.references);self.references.append(value)
            payload=None
            if self.host_arena is not None and raw.nbytes>=self.mapped_threshold and raw.nbytes:
                payload=self.host_arena.snapshot(raw);raw=payload.open()
                self.statistics.update(mapped_host_input_bytes=self.host_arena.bytes,
                    mapped_host_input_count=self.host_arena.count)
            self.values.append(raw);self.host_payloads.append(payload)
            self.flags.append(isinstance(value,ad.Variable) and value._record.tracked);self.host_bytes+=raw.nbytes
        return self.identities[token]

    def input_payload(self,key):
        payload=self.host_payloads[key]
        return self.values[key] if payload is None else payload

    def run(self,method,tasks):
        if self.closed:raise RuntimeError('Closed shared pool cannot be reused')
        begin=time.perf_counter();futures={}
        # Queue every remote task before executing the synchronous primary.
        order=sorted(range(len(tasks)),key=lambda i:isinstance(self.workers[tasks[i][0]],_InlineWorker))
        for i in order:
            rank,args=tasks[i];futures[i]=self.workers[rank].submit(_task,method,*args)
        result=[];errors=[]
        for i in range(len(tasks)):
            future=futures[i]
            try:result.append(future.result())
            except BaseException as exc:errors.append(exc)
        for rank in dict.fromkeys(rank for rank,_ in tasks):
            future=self.prestart_by_rank.get(rank)
            if future is not None:
                try:future.result()
                except BaseException as error:errors.append(error)
        if method in ('build','reverse'):
            self.statistics[method+'_seconds']+=time.perf_counter()-begin;self.statistics[method+'_calls']+=1
        if errors:raise errors[0]
        return result

    def submit_run(self,method,tasks):
        """Queue complete worker operations; reduce results on the caller later."""
        from .reverse_async import Pending
        if self.closed:raise RuntimeError('Closed shared pool cannot be reused')
        if any(isinstance(self.workers[rank],_InlineWorker) for rank,_ in tasks):
            raise ValueError('Async primary reverse requires DLPNO_RESIDENT_THREADED_PRIMARY=1')
        begin=time.perf_counter()
        futures=[]
        try:
            for rank,args in tasks:futures.append(self.workers[rank].submit(_task,method,*args))
        except BaseException as error:
            for future in futures:
                try:future.result()
                except BaseException as cleanup:error.add_note('Queued source completion: '+repr(cleanup))
            raise
        def resolve():
            result=[];errors=[]
            for future in futures:
                try:result.append(future.result())
                except BaseException as error:errors.append(error)
            if method in ('build','reverse'):
                self.statistics[method+'_seconds']+=time.perf_counter()-begin
                self.statistics[method+'_calls']+=1
            self.statistics['asynchronous_calls']=self.statistics.get('asynchronous_calls',0)+1
            if errors:raise errors[0]
            return result
        return Pending(futures,resolve)

    def _shutdown(self,begin,released,evidence):
        success=False
        try:
            if len(self.workers)>1 and os.environ.get('DLPNO_RESIDENT_PARALLEL_SHUTDOWN','0')=='1':
                with ThreadPoolExecutor(max_workers=len(self.workers)) as closing:
                    futures=[closing.submit(w.shutdown,wait=True,cancel_futures=True) for w in self.workers]
                    failures=[]
                    for future in futures:
                        try:future.result()
                        except BaseException as error:failures.append(error)
                    if failures:raise failures[0]
            else:
                for w in self.workers:w.shutdown(wait=True,cancel_futures=True)
            success=True
        finally:
            self.statistics.update(released=success,released_workers=evidence,
                release_work_seconds=released-begin,shutdown_seconds=time.perf_counter()-released,
                close_seconds=time.perf_counter()-begin,shutdown_completed_at=time.perf_counter())
        return evidence

    def _release(self):
        begin=time.perf_counter();evidence=[];error=None
        try:evidence=[info for _,info in self.run('close',[(i,()) for i in range(len(self.devices))])]
        except BaseException as exception:error=exception
        self.closed=True
        self.prestart_futures.clear();self.prestart_by_rank.clear()
        self.identities.clear();self.references.clear();self.values.clear();self.flags.clear();self.locations.clear();self.host_bytes=0
        self.host_payloads.clear()
        if self.host_arena is not None:
            self.host_arena.close();self.host_arena=None
            self.statistics['mapped_host_arena_released']=True
        self.statistics.update(device_store_released=error is None and len(evidence)==len(self.workers) and all(x['live_bytes']==x['reserved_bytes']==0 for x in evidence),released_workers=evidence)
        return begin,time.perf_counter(),evidence,error

    def begin_close(self):
        """Free all device graphs now; join process exit at the kernel boundary."""
        with self._close_lock:
            if self.closed:return self.statistics.get('released_workers',[])
            begin,released,evidence,error=self._release()
            self.statistics['background_shutdown']=True
            self._closing_executor=ThreadPoolExecutor(max_workers=1)
            self._closing_future=self._closing_executor.submit(self._shutdown,begin,released,evidence)
            if error is not None:
                try:self.close()
                except BaseException as cleanup:error.add_note('Shutdown: '+repr(cleanup))
                raise error
            return evidence

    def close(self):
        with self._close_lock:
            if self._closing_future is not None:
                begin=time.perf_counter()
                try:return self._closing_future.result()
                finally:
                    self._closing_executor.shutdown(wait=True,cancel_futures=True)
                    self._closing_executor=None;self._closing_future=None
                    self.statistics['shutdown_join_wait_seconds']=time.perf_counter()-begin
            if self.closed:return self.statistics.get('released_workers',[])
            begin,released,evidence,error=self._release()
            try:result=self._shutdown(begin,released,evidence)
            except BaseException as cleanup:
                if error is None:raise
                error.add_note('Shutdown: '+repr(cleanup))
            if error is not None:raise error
            return result


def stage_assignment(costs, ranks, count):
    """Longest bounded-shape costs first, with stable original task indices.

    Only distinct local shape costs are sorted; their count is bounded when
    local dimensions are bounded. Earlier completed stages cannot bias the
    next barrier's device loads. Scientific output order stays unchanged.
    """
    if not ranks or len(set(ranks)) != len(ranks) or any(r < 0 or r >= count for r in ranks):
        raise ValueError('Invalid stage scheduling ranks')
    buckets = {}
    for index, cost in enumerate(costs):
        if cost < 0 or not np.isfinite(cost):
            raise ValueError('Stage costs must be finite and nonnegative')
        buckets.setdefault(cost, []).append(index)
    assignment = [None] * len(costs)
    loads = [0] * count
    for cost in sorted(buckets, reverse=True):
        for index in buckets[cost]:
            rank = min(ranks, key=loads.__getitem__)
            assignment[index] = rank
            loads[rank] += cost
    return assignment, loads


class SharedRegionPlan:
    def __init__(self,prepared,pool,*,owner=None,balance_stage=False,ranks=None):
        ranks=tuple(range(len(pool.devices))) if ranks is None else tuple(ranks)
        if not ranks or len(set(ranks))!=len(ranks) or any(i<0 or i>=len(pool.devices) for i in ranks):
            raise ValueError('Invalid shared region scheduling ranks')
        self.pool=pool;self.owner=None if owner is None else weakref.ref(owner);self.closed=False
        begin=time.perf_counter();self.inputs=[];self.global_keys=[];identity={};self.assignments=[];self.mappings=[]
        grouped=[[] for _ in pool.devices];new=[{} for _ in pool.devices];self.rank_keys=[set() for _ in pool.devices]
        logical_input_bytes=0;stage_loads=[0]*len(pool.devices)
        largest_first = os.environ.get('DLPNO_RESIDENT_STAGE_BALANCE','0') == '1'
        costs = [spec.get('resident_cost',sum(ad.primal(x).nbytes for x in operands))
                 for spec,operands in prepared]
        if largest_first:
            balance_stage = True
            scheduling,stage_loads = stage_assignment(costs,ranks,len(pool.devices))
        for task_index,(spec,operands) in enumerate(prepared):
            keys=[];mapping=[]
            for operand in operands:
                key=pool.register(operand);keys.append(key)
                if key not in identity:
                    identity[key]=len(self.inputs);self.inputs.append(operand);self.global_keys.append(key)
                mapping.append(identity[key])
            loads=stage_loads if balance_stage else pool.loads
            rank=scheduling[task_index] if largest_first else min(ranks,key=loads.__getitem__)
            region=next(_ids)
            sizes=sum(ad.primal(x).nbytes for x in operands);logical_input_bytes+=sizes
            if not largest_first:loads[rank]+=costs[task_index]
            self.assignments.append((rank,region));self.mappings.append(tuple(mapping));self.rank_keys[rank].update(keys)
            grouped[rank].append((region,spec,tuple(keys)))
            for key,value in zip(keys,operands):
                if key not in pool.locations[rank] and key not in new[rank]:
                    new[rank][key]=(key,pool.input_payload(key),pool.flags[key])
        self.inputs=tuple(self.inputs);self.global_keys=tuple(self.global_keys)
        self.values=tuple(pool.values[key] for key in self.global_keys)
        input_bytes=sum(x.nbytes for x in self.values)
        if input_bytes>host_budget_bytes():raise MemoryError('Shared resident host boundary exceeds its explicit budget')
        if largest_first:
            region_cost={region:cost for (_,region),cost in zip(self.assignments,costs)}
            for rank,group in enumerate(grouped):
                buckets={}
                for task in group:buckets.setdefault(region_cost[task[0]],[]).append(task)
                grouped[rank]=[task for cost in sorted(buckets,reverse=True) for task in buckets[cost]]
        tasks=[(rank,(tuple(new[rank].values()),group)) for rank,group in enumerate(grouped) if group]
        self.statistics=dict(prepare_seconds=time.perf_counter()-begin,reverse_seconds=0.,reverse_calls=0,
            tasks=len(prepared),input_bytes=input_bytes,logical_task_input_bytes=logical_input_bytes,
            uploaded_input_bytes=sum(v[1].nbytes for rank in new for v in rank.values()),
            coefficients_complete=True,shared_store=True,balance_current_stage=balance_stage,stage_loads=list(stage_loads),scheduling_ranks=list(ranks),
            largest_cost_first=largest_first,task_costs=list(costs),task_ranks=[rank for rank,_ in self.assignments])
        started=time.perf_counter();built=pool.run('build',tasks)
        rows={};workers=[]
        for (rank,_),((packed,metadata),info) in zip(tasks,built):
            pool.locations[rank].update(new[rank]);workers.append(info)
            for key,a,b,shapes,layout,statistics in metadata:rows[key]=(packed[a:b],shapes,layout,statistics)
        ordered=[rows[key] for _,key in self.assignments]
        self.statistics.update(forward_seconds=time.perf_counter()-started,workers=workers,worker_graphs=[x[3] for x in ordered])
        self.parts=[x[0] for x in ordered];self.shapes=tuple(shape for x in ordered for shape in x[1])
        self.layouts=[x[2] for x in ordered];self.task_sizes=[len(x[1]) for x in ordered]
        self.task_offsets=np.cumsum([0]+[x.size for x in self.parts])
        self.output_offsets=np.cumsum([0]+[int(np.prod(shape)) for shape in self.shapes])

    def release(self):
        if self.closed:return
        if not self.pool.closed:
            groups=[[] for _ in self.pool.devices]
            for rank,key in self.assignments:groups[rank].append(key)
            self.pool.run('discard',[(rank,(keys,)) for rank,keys in enumerate(groups) if keys])
        self.closed=True;self.parts=None;self.inputs=();self.values=()
        self.statistics['released']=True

    def packed_bridge(self, *, gpu_output=False):
        packed=np.concatenate(self.parts);self.parts=None
        if gpu_output:packed=cp.asarray(packed)
        def submit(seed,needed):
            if self.closed:raise RuntimeError('Consumed shared region plan cannot be reused')
            owner=None if self.owner is None else self.owner();consume=owner is not None and getattr(owner,'_release_transposes',False)
            begin=time.perf_counter();groups=[[] for _ in self.pool.devices]
            if isinstance(seed,cp.ndarray):seed=cp.asnumpy(seed)
            selected={k for k,flag in zip(self.global_keys,needed) if flag}
            for (rank,key),a,b in zip(self.assignments,self.task_offsets[:-1],self.task_offsets[1:]):groups[rank].append((key,np.asarray(seed[a:b])))
            tasks=[(rank,(group,tuple(sorted(selected&self.rank_keys[rank])),consume)) for rank,group in enumerate(groups) if group]
            pending=self.pool.submit_run('reverse',tasks)
            def reduce(parts):
                result={}
                for bars,_ in parts:
                    for key,bar in bars:
                        if key in result:result[key]+=bar
                        else:result[key]=bar.copy()
                self.statistics['reverse_calls']+=1;self.statistics['reverse_seconds']+=time.perf_counter()-begin
                self.statistics['last_reverse_workers']=[info for _,info in parts]
                if consume:self.closed=True
                return tuple((result[key] if key in result else np.zeros_like(self.values[i])) if flag else None for i,(key,flag) in enumerate(zip(self.global_keys,needed)))
            return pending.map(reduce)
        def reverse(seed,needed):
            # Serial/default calls retain their existing inline worker policy.
            if os.environ.get('DLPNO_ASYNC_SOURCE_REVERSE','0')=='1':
                return submit(seed,needed).result()
            if self.closed:raise RuntimeError('Consumed shared region plan cannot be reused')
            owner=None if self.owner is None else self.owner();consume=owner is not None and getattr(owner,'_release_transposes',False)
            begin=time.perf_counter();groups=[[] for _ in self.pool.devices]
            if isinstance(seed,cp.ndarray):seed=cp.asnumpy(seed)
            selected={k for k,flag in zip(self.global_keys,needed) if flag}
            for (rank,key),a,b in zip(self.assignments,self.task_offsets[:-1],self.task_offsets[1:]):groups[rank].append((key,np.asarray(seed[a:b])))
            tasks=[(rank,(group,tuple(sorted(selected&self.rank_keys[rank])),consume)) for rank,group in enumerate(groups) if group]
            parts=self.pool.run('reverse',tasks);result={}
            for bars,_ in parts:
                for key,bar in bars:
                    if key in result:result[key]+=bar
                    else:result[key]=bar.copy()
            self.statistics['reverse_calls']+=1;self.statistics['reverse_seconds']+=time.perf_counter()-begin
            self.statistics['last_reverse_workers']=[info for _,info in parts]
            if consume:self.closed=True
            return tuple((result[key] if key in result else np.zeros_like(self.values[i])) if flag else None for i,(key,flag) in enumerate(zip(self.global_keys,needed)))
        return ad.operation(packed,self.inputs,lambda seed:reverse(seed,[True]*len(self.inputs)),reverse,
            async_pullback=submit if os.environ.get('DLPNO_ASYNC_SOURCE_REVERSE','0')=='1' else None,
            # Each reduction starts with fresh, disjoint host arrays. The
            # outer graph may own them without another full boundary copy.
            owns_cotangents=os.environ.get('DLPNO_RESIDENT_TRANSFER_BARS','0')=='1')

    def bridge(self, *, gpu_output=False):
        output=self.packed_bridge(gpu_output=gpu_output)
        return tuple(output[a:b].reshape(shape) for a,b,shape in zip(self.output_offsets[:-1],self.output_offsets[1:],self.shapes))


def close_pools():
    output=[]
    for pool in _pools:output.extend(pool.close())
    _pools.clear();return output


atexit.register(close_pools)
