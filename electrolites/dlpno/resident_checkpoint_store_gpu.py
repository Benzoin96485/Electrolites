"""Bounded GPU replay of complete local source graphs.

Recipes and immutable FP64 input snapshots live on the host. Each worker
rebuilds one local region, checks its outputs against the original forward
bytes, evaluates every requested source cotangent, then releases the graph.
A bounded LRU cache can retain immutable device inputs between regions.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
import numpy as np
import cupy as cp
from . import reverse as ad
from .resident_shared_store_gpu import _Bank, _Region, _Store, _local, host_budget_bytes


@dataclass
class _Recipe:
    spec: dict
    keys: tuple
    shapes: tuple
    digest: str


def _packed(region):
    pieces=[ad.primal(x).ravel() for x in region.outputs]
    return cp.asnumpy(cp.concatenate(pieces)) if pieces else np.empty(0,dtype=np.float64)


def _digest(value):
    return hashlib.sha256(memoryview(value).cast('B')).hexdigest()


class CheckpointStore(_Store):
    def __init__(self):
        super().__init__()
        self.primal_only_forward=os.environ.get('DLPNO_RESIDENT_PRIMAL_FORWARD','0')=='1'
        self.inputs=OrderedDict()
        self.host_inputs={}
        self.host_bytes=0
        self.host_limit=host_budget_bytes()
        self.cached_bytes=0
        limit=_local.pool.get_limit()
        requested=int(os.environ.get('DLPNO_RESIDENT_INPUT_CACHE_MIB','1024'))<<20
        if not 0<=requested<=limit//2:
            raise ValueError('Resident input cache must fit within half the private device budget')
        self.cache_limit=requested
        self.statistics.update(checkpointed_graphs=True,host_input_bytes=0,
            device_input_cache_limit_bytes=requested,device_input_cache_bytes=0,
            device_input_uploads=0,device_input_upload_bytes=0,input_evictions=0,
            region_replays=0,verified_replays=0,retained_graphs=0,
            largest_region_pool_bytes=0,forward_checksums=True,
            primal_only_forward=self.primal_only_forward,primal_only_forward_regions=0,
            primal_only_forward_records=0,reverse_traced_records=0)

    def _trim(self,keep=()):
        keep=set(keep)
        for key in list(self.inputs):
            if self.cached_bytes<=self.cache_limit:break
            if key not in keep:
                leaf=self.inputs.pop(key);self.cached_bytes-=leaf.size*8
                self.statistics['input_evictions']+=1
        self.statistics['device_input_cache_bytes']=self.cached_bytes

    def _activate(self,keys):
        keep=set(keys)
        # Reserve space for all current operands before any upload. A single
        # region may exceed the cache target, but never the hard pool limit.
        incoming=sum(self.host_inputs[k][0].nbytes for k in keep if k not in self.inputs)
        target=max(0,self.cache_limit-incoming)
        for key in list(self.inputs):
            if self.cached_bytes<=target:break
            if key not in keep:
                leaf=self.inputs.pop(key);self.cached_bytes-=leaf.size*8
                self.statistics['input_evictions']+=1
        for key in dict.fromkeys(keys):
            if key in self.inputs:self.inputs.move_to_end(key);continue
            value,flag=self.host_inputs[key]
            self.inputs[key]=ad.Variable(cp.asarray(value),tracked=flag)
            self.cached_bytes+=value.nbytes
            self.statistics['device_input_uploads']+=1
            self.statistics['device_input_upload_bytes']+=value.nbytes
        self.statistics['device_input_cache_bytes']=self.cached_bytes

    def _clear_region(self,region):
        if region is not None:region.release()
        self.reads.clear()
        self._trim()

    def build(self,new_inputs,tasks):
        for key,value,flag in new_inputs:
            if key in self.host_inputs:raise RuntimeError('Immutable input uploaded twice to one worker')
            if self.host_bytes+value.nbytes>self.host_limit:
                raise MemoryError('Checkpoint source host inputs exceed their explicit budget')
            # Inline workers also need a snapshot: later parent mutations or
            # consuming graph sweeps cannot alter a replay's primal values.
            from .host_input_snapshots import HostSnapshot
            if isinstance(value,HostSnapshot):
                value=value.open()
                self.statistics['mapped_host_input_bytes']=self.statistics.get('mapped_host_input_bytes',0)+value.nbytes
                self.statistics['mapped_host_input_count']=self.statistics.get('mapped_host_input_count',0)+1
            else:
                value=np.array(value,copy=True);value.flags.writeable=False
            self.host_inputs[key]=(value,flag);self.host_bytes+=value.nbytes
            self.statistics['input_uploads']+=1;self.statistics['input_upload_bytes']+=value.nbytes
        self.statistics['host_input_bytes']=self.host_bytes
        rows=[];pieces=[];offset=0
        for key,spec,input_keys in tasks:
            if key in self.regions:raise RuntimeError('Duplicate checkpoint region identifier')
            region=None
            try:
                self._activate(input_keys)
                region=_Region(spec,_Bank(self,input_keys,recording=not self.primal_only_forward))
                if self.primal_only_forward:
                    self.statistics['primal_only_forward_regions']+=1
                    self.statistics['primal_only_forward_records']+=region.statistics['records']
                value=_packed(region)
                statistics=dict(region.statistics,retained_device_bytes=0,checkpointed=True,
                    forward_region_pool_bytes=_local.pool.used_bytes())
                self.statistics['largest_region_pool_bytes']=max(self.statistics['largest_region_pool_bytes'],_local.pool.used_bytes())
                self.regions[key]=_Recipe(spec,tuple(input_keys),region.shapes,_digest(value))
                rows.append((key,offset,offset+value.size,region.shapes,region.layout,statistics))
                pieces.append(value);offset+=value.size
            finally:self._clear_region(region)
        packed=np.concatenate(pieces) if pieces else np.empty(0,dtype=np.float64)
        self.statistics['forward_host_bytes']+=packed.nbytes;self.statistics['forward_transfers']+=bool(pieces)
        return packed,rows

    def reverse(self,tasks,needed_keys,consume):
        needed_keys=set(needed_keys);accumulated={}
        for key,seed in tasks:
            recipe=self.regions[key];region=None;bars=None;direct={}
            try:
                self._activate(recipe.keys)
                region=_Region(recipe.spec,_Bank(self,recipe.keys));self.statistics['region_replays']+=1
                value=_packed(region)
                if region.shapes!=recipe.shapes or _digest(value)!=recipe.digest:
                    raise RuntimeError('Checkpoint replay changed its forward outputs or discrete subspace')
                del value
                self.statistics['verified_replays']+=1
                self.statistics['reverse_traced_records']+=region.statistics['records']
                self.statistics['largest_region_pool_bytes']=max(self.statistics['largest_region_pool_bytes'],_local.pool.used_bytes())
                # Every inner replay is consumed. The host recipe survives a
                # reusable outer VJP, including selected-input sweeps.
                bars=region.reverse(seed,needed_keys,True)
                self._accumulate_bars(bars,needed_keys,accumulated)
                if consume:del self.regions[key]
            except BaseException as error:
                context=dict(region=int(key),kind=recipe.spec.get('kind','triples_sources'),
                    pool_used_bytes=_local.pool.used_bytes(),pool_reserved_bytes=_local.pool.total_bytes(),
                    pool_limit_bytes=_local.pool.get_limit(),replay_count=self.statistics['region_replays'])
                self.statistics['last_failed_region']=context
                error.add_note('Checkpoint source reverse context: '+str(context))
                raise
            finally:
                bars=None;direct.clear();self._clear_region(region)
        return [(key,accumulated[key] if key in accumulated else np.zeros_like(self.host_inputs[key][0])) for key in sorted(needed_keys)]

    def _cotangent_groups(self,bars,needed_keys):
        # Record views symbolically. Reshaping/indexing is delayed until the
        # corresponding input is reduced, so the plan owns no extra arrays.
        groups={}
        for (kind,index),bar in bars:
            if kind=='input':
                groups.setdefault(index,[]).append((bar,None,None,None))
                continue
            descriptor,_=self.reads[index]
            for source,shape,(local,output) in zip(descriptor.sources,descriptor.source_shapes,descriptor.parts):
                if source in needed_keys:
                    groups.setdefault(source,[]).append((bar,descriptor.cartesian_shape,local,output))
        return groups

    def _requires_chunking(self,groups):
        sizes=[self.host_inputs[key][0].nbytes for key in groups]
        # The original path owns each reduced input and a concatenation of
        # all of them. Allow one extra input for a temporary read/reshape.
        projected=_local.pool.used_bytes()+2*sum(sizes)+max(sizes,default=0)
        self.statistics['largest_legacy_pack_estimated_bytes']=max(
            self.statistics.get('largest_legacy_pack_estimated_bytes',0),projected)
        limit=_local.pool.get_limit()
        return bool(limit and projected>limit)

    def _accumulate_chunked(self,groups,accumulated):
        count=0;transferred=0;value=None;reduced=None
        try:
            for index in sorted(groups):
                # Contributions retain their original order for each input;
                # no bar, seed, canonical primal or sibling view is mutated.
                reduced=None
                for bar,cartesian,local,output in groups[index]:
                    value=bar if cartesian is None else bar.reshape(cartesian)[output]
                    if local is None:
                        if reduced is None:reduced=value.copy()
                        else:reduced+=value
                    else:
                        if reduced is None:
                            reduced=cp.zeros(self.host_inputs[index][0].shape,dtype=cp.float64)
                        cp.add.at(reduced,local,value)
                    value=None
                self.statistics['largest_chunked_group_bytes']=max(
                    self.statistics.get('largest_chunked_group_bytes',0),reduced.nbytes)
                self.statistics['largest_chunked_pool_bytes']=max(
                    self.statistics.get('largest_chunked_pool_bytes',0),_local.pool.used_bytes())
                host=cp.asnumpy(reduced);reduced=None
                if index in accumulated:accumulated[index]+=host
                else:accumulated[index]=host
                count+=1;transferred+=host.nbytes
            self.statistics['chunked_reverse_regions']=self.statistics.get('chunked_reverse_regions',0)+1
            self.statistics['chunked_reverse_transfers']=self.statistics.get('chunked_reverse_transfers',0)+count
            self.statistics['reverse_host_bytes']+=transferred
            self.statistics['reverse_transfers']+=count
        finally:
            value=reduced=None

    def _legacy_pack_has_headroom(self,bars):
        # All canonical source arrays are immutable host registrations. Their
        # total bounds the reduced arrays, optional ravel copies and packed
        # result without walking every deferred-gather descriptor. Keep two
        # complete bar volumes for simultaneous reshape/index temporaries,
        # allocation rounding and fixed headroom. This is intentionally loose;
        # an inconclusive bound takes the precise grouping path below.
        total=getattr(self,'host_bytes',0)
        limit=_local.pool.get_limit()
        if not total or not limit:return False
        projected=(_local.pool.used_bytes()+4*total
            +2*sum(bar.nbytes for _,bar in bars)
            +512*(4*len(self.host_inputs)+2*len(bars))+(16<<20))
        self.statistics['largest_fast_guard_estimated_bytes']=max(
            self.statistics.get('largest_fast_guard_estimated_bytes',0),projected)
        return projected<=limit

    def _accumulate_bars(self,bars,needed_keys,accumulated):
        if os.environ.get('DLPNO_RESIDENT_COTANGENT_CHUNKED','0')=='1':
            fast=(os.environ.get('DLPNO_RESIDENT_COTANGENT_FAST_GUARD','0')=='1'
                and self._legacy_pack_has_headroom(bars))
            if fast:
                self.statistics['cotangent_group_plans_skipped']=self.statistics.get('cotangent_group_plans_skipped',0)+1
            else:
                self.statistics['cotangent_group_plans_built']=self.statistics.get('cotangent_group_plans_built',0)+1
                groups=self._cotangent_groups(bars,needed_keys)
                if self._requires_chunking(groups):
                    self._accumulate_chunked(groups,accumulated)
                    return
        direct={};value=bar=None
        try:
            def add(index,value):
                if index in direct:direct[index]+=value
                else:direct[index]=value.copy()
            for (kind,index),bar in bars:
                if kind=='input':add(index,bar);continue
                descriptor,_=self.reads[index];bar=bar.reshape(descriptor.cartesian_shape)
                for source,shape,(local,output) in zip(descriptor.sources,descriptor.source_shapes,descriptor.parts):
                    if source not in needed_keys:continue
                    value=bar[output]
                    if local is None:add(source,value)
                    else:
                        if source not in direct:direct[source]=cp.zeros(shape,dtype=cp.float64)
                        cp.add.at(direct[source],local,value)
            indices=sorted(direct);offsets=np.cumsum([0]+[direct[i].size for i in indices])
            packed=cp.asnumpy(cp.concatenate([direct[i].ravel() for i in indices])) if indices else np.empty(0,dtype=np.float64)
            self.statistics['reverse_host_bytes']+=packed.nbytes
            self.statistics['reverse_transfers']+=bool(indices)
            for index,a,b in zip(indices,offsets[:-1],offsets[1:]):
                value=packed[a:b].reshape(self.host_inputs[index][0].shape)
                if index in accumulated:accumulated[index]+=value
                else:accumulated[index]=value.copy()
        finally:
            direct.clear();value=bar=None

    def discard(self,keys):
        for key in keys:self.regions.pop(key,None)

    def close(self):
        self.regions.clear();self.reads.clear();self.inputs.clear();self.host_inputs.clear()
        self.cached_bytes=self.host_bytes=0
        self.statistics.update(device_input_cache_bytes=0,host_input_bytes=0,mapped_host_input_bytes=0)
