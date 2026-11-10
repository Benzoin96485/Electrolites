"""Complete shared CCSD overlap sources directly into ragged buckets.

The packed union contraction and the RaggedPacking map are reused from
validated source524. SharedRegionPlan owns the source graphs until the
parent density sweep has returned all metric and coefficient cotangents.
"""
import os,time
import numpy as np
from . import reverse as ad
from .ragged_adjoint import AdjointRagged


class RaggedPacking:
    def __init__(self,layout,keys,source_keys,source_shapes,source_offsets,*,map_bytes=512<<20,workspace_bytes=8<<20):
        start=time.perf_counter()
        keys,source_keys=tuple(keys),tuple(source_keys)
        if len(keys)!=layout.n or len(source_keys)!=len(keys) or len(set(keys))!=len(keys) or set(keys)!=set(source_keys):
            raise ValueError('Ragged overlap couples must be distinct and cover the union')
        offsets=np.asarray(source_offsets,dtype=np.int64)
        if len(offsets)!=len(source_keys)+1 or offsets[0]!=0:
            raise ValueError('Invalid union overlap offsets')
        self.logical_size=int(offsets[-1]);self.shapes=[];self.offsets=[0]
        self.compact=os.environ.get('DLPNO_CCSD_COMPACT_PACKING','0')=='1'
        self.workspace_bytes=int(workspace_bytes);self.compact_groups=();groups={}
        if self.workspace_bytes<=0:raise ValueError('Positive packing workspace required')
        if not self.compact and self.logical_size*8>map_bytes:
            raise MemoryError('CCSD overlap packing map exceeds its explicit budget')
        self.destinations=None if self.compact else np.empty(self.logical_size,dtype=np.int64)
        source={key:i for i,key in enumerate(source_keys)}
        for members,(pr,pc) in zip(layout.members,layout.shapes):
            shape=(len(members),int(pr),int(pc));first=self.offsets[-1]
            self.shapes.append(shape);self.offsets.append(first+int(np.prod(shape)))
            for slot,p in enumerate(members):
                k=source[keys[int(p)]];nr,nc=int(layout.rows[p]),int(layout.cols[p])
                a,b=map(int,offsets[k:k+2])
                if tuple(source_shapes[k])!=(nr,nc) or b-a!=nr*nc:
                    raise ValueError('Union and CCSD overlap shapes differ')
                if self.compact:
                    if nr*nc:
                        if 56*nr*nc>self.workspace_bytes:raise MemoryError('One overlap exceeds packing workspace')
                        groups.setdefault((nr,nc,int(pc)),[]).append((a,int(first+slot*pr*pc)))
                else:
                    self.destinations[a:b]=(first+slot*pr*pc+
                        np.arange(nr,dtype=np.int64)[:,None]*pc+np.arange(nc,dtype=np.int64)).ravel()
        self.size=self.offsets[-1]
        if self.compact:
            self.compact_groups=tuple((shape,np.asarray(starts,dtype=np.int64).reshape(-1,2)) for shape,starts in groups.items())
            metadata_bytes=sum(starts.nbytes+24 for shape,starts in self.compact_groups)
            if metadata_bytes>map_bytes:raise MemoryError('Compact CCSD packing metadata exceeds its explicit budget')
        else:metadata_bytes=self.destinations.nbytes
        self.statistics=dict(logical_elements=self.logical_size,padded_elements=self.size,
            map_bytes=metadata_bytes,element_map_bytes_avoided=self.logical_size*8 if self.compact else 0,
            compact_blocks=self.compact,packing_workspace_limit_bytes=self.workspace_bytes,
            maximum_packing_workspace_bytes=0,packing_chunks=0,buckets=len(self.shapes),parent_overlap_records=0,
            prepare_seconds=time.perf_counter()-start,pack_seconds=0.,reverse_seconds=0.,reverse_calls=0)

    def _copy_blocks(self,source,destination,*,reverse=False):
        # Shape types are bounded at fixed local rank. Keep only two offsets
        # per physical matrix; form element maps in bounded temporary chunks.
        for (nr,nc,stride),starts in self.compact_groups:
            count=nr*nc;flat=np.arange(count,dtype=np.int64)
            local=(flat//nc)*stride+flat%nc
            step=max(1,(self.workspace_bytes-32*count)//(24*count))
            for begin in range(0,len(starts),step):
                part=starts[begin:begin+step]
                a=part[:,0,None]+flat; b=part[:,1,None]+local
                if reverse:destination[a]=source[b]
                else:destination[b]=source[a]
                self.statistics['packing_chunks']+=1
                self.statistics['maximum_packing_workspace_bytes']=max(
                    self.statistics['maximum_packing_workspace_bytes'],32*count+24*count*len(part))
                del a,b
            del flat,local

    def pack(self,value):
        if value.shape!=(self.logical_size,):raise ValueError('Invalid packed overlap input shape')
        start=time.perf_counter();raw=ad.primal(value)
        if not isinstance(raw,np.ndarray) or raw.dtype!=np.float64:
            raise TypeError('Ragged overlap boundary requires host float64 values')
        data=np.zeros(self.size)
        if self.compact:self._copy_blocks(raw,data)
        else:data[self.destinations]=raw
        self.statistics['pack_seconds']+=time.perf_counter()-start
        def pullback(seed):
            start=time.perf_counter()
            if self.compact:
                result=np.empty(self.logical_size);self._copy_blocks(seed,result,reverse=True)
            else:result=seed[self.destinations].copy()
            self.statistics['reverse_seconds']+=time.perf_counter()-start
            self.statistics['reverse_calls']+=1
            return (result,)
        return ad.operation(data,[value],pullback)

    def buckets(self,value):
        return [value[a:b].reshape(shape) for a,b,shape in zip(self.offsets[:-1],self.offsets[1:],self.shapes)]



def prepare(state,keys,*,max_count=256,workspace_bytes=256<<20):
    from .deferred_reads import capture_reads
    from .overlap_union_adjoint_gpu import union_batches
    from .resident_overlap_inputs_gpu import OverlapInputs
    groups=union_batches(tuple(dict.fromkeys(keys)),state.pao,state.X,
        max_bytes=workspace_bytes,max_count=max_count,stable_groups=True)
    prepared=[];ordered=[];shapes=[]
    with capture_reads(os.environ.get('DLPNO_RESIDENT_DEFER_READS','0')=='1') as deferred:
        for group,union,involved,bound in groups:
            capture=OverlapInputs();index={int(p):j for j,p in enumerate(involved)}
            spec=dict(kind='ccsd_overlap_sources',metric=capture.capture(state.orb.S_pao[np.ix_(union,union)]),
                coefficients=tuple(capture.capture(state.X[p]) for p in involved),
                positions=tuple(np.searchsorted(union,state.pao[p]) for p in involved),
                couples=np.asarray([(index[p],index[q]) for p,q in group],dtype=np.int64),resident_cost=bound)
            prepared.append((spec,tuple(capture.inputs)));ordered.extend(group)
            shapes.extend((state.X[p].shape[1],state.X[q].shape[1]) for p,q in group)
    return prepared,tuple(ordered),tuple(shapes),deferred


def shared_plan(state,keys,*,max_count=256,workspace_bytes=256<<20,memory_bytes=2048<<20):
    from .resident_regions_gpu import _pool
    from .resident_shared_store_gpu import SharedRegionPlan
    if not os.environ.get('DLPNO_RESIDENT_REGION_DEVICES') or os.environ.get('DLPNO_RESIDENT_SHARED_STORE','0')!='1':
        raise ValueError('Shared CCSD overlaps require the resident canonical source store')
    if not keys:raise ValueError('Shared CCSD source overlap list is empty')
    if not hasattr(state,'diagnostics'):state.diagnostics={}
    if not hasattr(state,'_resident_region_plans'):state._resident_region_plans=[]
    begin=time.perf_counter();prepared,ordered,shapes,deferred=prepare(state,keys,max_count=max_count,workspace_bytes=workspace_bytes)
    pool=_pool(state)
    reserved=getattr(state,'_ccsd_primal_reserved_rank',None)
    ranks=tuple(i for i in range(len(pool.devices)) if i!=reserved)
    plan=SharedRegionPlan(prepared,pool,owner=state if hasattr(state,'__weakref__') else None,balance_stage=True,ranks=ranks)
    totals={device:0 for device in pool.devices}
    for (rank,_),graph in zip(plan.assignments,plan.statistics['worker_graphs']):totals[pool.devices[rank]]+=graph['retained_device_bytes']
    if max(totals.values(),default=0)>memory_bytes:
        plan.release();raise MemoryError('Shared CCSD overlap graphs exceed per-stage memory bound')
    plan.keys=ordered;plan.overlap_shapes=shapes;plan.overlap_offsets=np.cumsum([0]+[int(np.prod(s)) for s in shapes])
    plan.statistics.update(kind='ccsd_overlap_sources',overlaps=len(ordered),parent_per_overlap_records=0,
        complete_nonsymmetric_metric_vjp=True,deferred_reads=deferred,prepare_forward_seconds=time.perf_counter()-begin,
        retained_graph_bytes_by_device=totals,workspace_limit_bytes=workspace_bytes,memory_limit_bytes=memory_bytes)
    state._resident_region_plans.append(plan);state.diagnostics.setdefault('resident_regions',[]).append(plan.statistics)
    state.diagnostics['shared_ccsd_overlaps']=plan.statistics
    return plan


def build(source_plan,keys):
    state=source_plan.source
    workspace=int(os.environ.get('DLPNO_CCSD_SHARED_OVERLAP_WORKSPACE_MIB','256'))<<20
    memory=int(os.environ.get('DLPNO_CCSD_SHARED_OVERLAP_MEMORY_MIB','2048'))<<20
    map_bytes=int(os.environ.get('DLPNO_CCSD_SHARED_OVERLAP_MAP_MIB','512'))<<20
    count=int(os.environ.get('DLPNO_CCSD_SHARED_OVERLAP_BATCH','256'))
    if not 1<=count<=4096 or not 1<<20<=workspace<=2<<30 or not 32<<20<=memory<=8<<30 or not 1<<20<=map_bytes<=2<<30:
        raise ValueError('Shared CCSD overlap limits outside bounds')
    plan=shared_plan(state,keys,max_count=count,workspace_bytes=workspace,memory_bytes=memory)
    try:
        packing=RaggedPacking(source_plan.LS,keys,plan.keys,plan.overlap_shapes,plan.overlap_offsets,map_bytes=map_bytes)
        packed=packing.pack(plan.packed_bridge());device=source_plan._capture(packed)
        plan.statistics['ragged_packing']=packing.statistics
        source_plan.statistics['shared_overlap_sources']=plan.statistics
        return AdjointRagged(source_plan.LS,packing.buckets(device))
    except BaseException:
        plan.release();raise
