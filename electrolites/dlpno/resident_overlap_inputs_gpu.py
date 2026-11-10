"""Canonical overlap inputs owned by complete resident triples shards.

Build only overlap groups used by a shard, within its private pool/stream.
The same resident values feed its residual and transpose. The final source
pullback propagates the accumulated overlap bars to all metric and orbital
coefficient inputs; no such coefficient is held fixed in a density sweep.
"""
import time
import numpy as np
import cupy as cp
from . import reverse as ad
from .deferred_reads import DeferredRead, EncodedRead, OperandBank, _index
from .parallel_local_gpu import Operand
from .triples_overlap_adjoint_gpu import common_left_batch


class OverlapInputs:
    """Parent-side immutable metadata and explicit canonical AD boundary."""
    def __init__(self):
        self.inputs = []
        self.identity = {}
        self.groups = []
        self.shapes = []

    def capture(self, value):
        if isinstance(value, DeferredRead):
            return value.encode(self.capture)
        raw = ad.primal(value)
        if raw.dtype != np.float64:
            raise TypeError('Resident overlap inputs require real float64 arrays')
        key = ('record', value._record) if isinstance(value, ad.Variable) else ('array', id(value))
        if key not in self.identity:
            self.identity[key] = len(self.inputs)
            self.inputs.append(ad.Variable._from_record(value._record) if isinstance(value, ad.Variable) else value)
        return Operand(self.identity[key])

    def add(self, metric, left, coefficients, positions):
        first = len(self.shapes)
        shapes = [(left.shape[1], x.shape[1]) for x in coefficients]
        self.shapes.extend(shapes)
        self.groups.append(dict(metric=self.capture(metric), left=self.capture(left),
            right=tuple(self.capture(x) for x in coefficients),
            positions=tuple(np.asarray(x, dtype=np.int64) for x in positions),
            output_ids=tuple(range(first, first+len(shapes)))))
        return shapes

    def freeze(self, *, materialize=True):
        self.inputs = tuple(self.inputs)
        self.identity.clear()
        self.values = (tuple(cp.asnumpy(ad.primal(x)) if isinstance(ad.primal(x), cp.ndarray)
                             else np.asarray(ad.primal(x)) for x in self.inputs) if materialize else None)
        self.tracked = tuple(isinstance(x, ad.Variable) and x._record.tracked for x in self.inputs)
        self.offsets = np.cumsum([0]+[int(np.prod(s)) for s in self.shapes])
        self.input_bytes = sum(ad.primal(x).nbytes for x in self.inputs)
        if self.input_bytes > 2 << 30:
            raise MemoryError('Resident overlap canonical host inputs exceed 2 GiB')
        self.statistics = dict(groups=len(self.groups), overlaps=len(self.shapes),
            canonical_inputs=len(self.inputs), input_bytes=self.input_bytes,
            parent_overlap_values_constructed=False, complete_metric_and_coefficient_vjp=True)
        return self


def _references(value):
    return tuple(_index(x) for x in value.sources) if isinstance(value, EncodedRead) else (_index(value),)


class OverlapWorker:
    """Resident source graph; all numeric operations run on the shard device."""
    def __init__(self, boundary, wanted):
        begin = time.perf_counter()
        wanted = set(wanted)
        self.inputs, self.outputs, self.output_ids = {}, [], []
        self.workspaces = {}
        self.input_shapes = tuple(x.shape for x in boundary.values)
        self.offsets = boundary.offsets
        self.output_shapes = boundary.shapes
        groups, used = [], set()
        for group in boundary.groups:
            keep = [i for i,k in enumerate(group['output_ids']) if k in wanted]
            if not keep:
                continue
            right = tuple(group['right'][i] for i in keep)
            positions = tuple(group['positions'][i] for i in keep)
            ids = tuple(group['output_ids'][i] for i in keep)
            groups.append((group['metric'], group['left'], right, positions, ids))
            for operand in (group['metric'], group['left'])+right:
                used.update(_references(operand))
        values = [None]*len(boundary.values)
        for key in sorted(used):
            value = ad.Variable(cp.asarray(boundary.values[key]), tracked=boundary.tracked[key])
            self.inputs[key] = values[key] = value
        self.bank = OperandBank(values)
        self.flat = cp.zeros(int(self.offsets[-1])+1, dtype=cp.float64)
        token = ad._replay_workspaces.set(self.workspaces)
        try:
            for metric, left, right, positions, ids in groups:
                outputs = common_left_batch(self.bank[metric], self.bank[left],
                    tuple(self.bank[k] for k in right), positions)
                self.outputs.extend(outputs)
                self.output_ids.extend(ids)
                for key, value in zip(ids, outputs):
                    a,b = self.offsets[key:key+2]
                    self.flat[a:b] = ad.primal(value).ravel()
        finally:
            ad._replay_workspaces.reset(token)
        self.keys = tuple(sorted(self.inputs))
        self.leaves = tuple(self.inputs[k] for k in self.keys)
        graph = ad.Linearization(self.outputs, self.leaves)
        self.statistics = dict(graph.statistics, groups=len(groups), overlaps=len(self.output_ids),
            canonical_inputs=len(used), deferred_reads=dict(self.bank.statistics),
            forward_seconds=time.perf_counter()-begin, reverse_seconds=0., reverse_calls=0,
            source_values_retained=True)

    def reverse(self, seed, needed):
        begin = time.perf_counter()
        selected = tuple(k for k in self.keys if needed[k])
        result = [cp.zeros(shape, dtype=cp.float64) if flag else None
                  for shape,flag in zip(self.input_shapes, needed)]
        if selected:
            graph = ad.Linearization(self.outputs, [self.inputs[k] for k in selected], boundary=self.leaves)
            seeds = [seed[self.offsets[k]:self.offsets[k+1]].reshape(self.output_shapes[k])
                     for k in self.output_ids]
            token = ad._replay_workspaces.set(self.workspaces)
            try:
                bars = graph(seeds)
            finally:
                ad._replay_workspaces.reset(token)
            for key,bar in zip(selected,bars):
                result[key] = bar
        self.statistics['reverse_seconds'] += time.perf_counter()-begin
        self.statistics['reverse_calls'] += 1
        return result

    def release(self):
        self.inputs.clear()
        self.outputs = self.leaves = ()
        self.workspaces.clear()
        self.bank = self.flat = None


def shared_overlaps(state, boundary):
    """One packed parent edge; complete target groups stay within one worker."""
    import os
    from dataclasses import replace
    from .resident_regions_gpu import _pool
    from .resident_shared_store_gpu import SharedRegionPlan
    if not os.environ.get('DLPNO_RESIDENT_REGION_DEVICES') or os.environ.get('DLPNO_RESIDENT_SHARED_STORE','0')!='1':
        raise ValueError('Shared T1 overlaps require configured shared resident region devices')
    if not hasattr(state,'diagnostics'):state.diagnostics={}
    if not hasattr(state,'_resident_region_plans'):state._resident_region_plans=[]
    if not boundary.groups:
        state.diagnostics['shared_t1_overlaps']=dict(groups=0,overlaps=0,packed_parent_edge=True)
        return ad.Variable(np.empty(0),tracked=False)
    count=int(os.environ.get('DLPNO_TRIPLES_OVERLAP_BATCH','8'))
    if not 1<=count<=64:raise ValueError('Shared overlap batch must be1..64 complete target groups')
    prepared=[]
    for first in range(0,len(boundary.groups),count):
        inputs=[];mapping={}
        def remap(operand):
            if isinstance(operand,EncodedRead):
                sources=tuple(remap(x) for x in operand.sources)
                signature=(operand.signature[0],operand.signature[1],
                    tuple((_index(ref),)+part[1:] for ref,part in zip(sources,operand.signature[2])))
                return replace(operand,sources=sources,signature=signature)
            key=_index(operand)
            if key not in mapping:
                mapping[key]=len(inputs);inputs.append(boundary.inputs[key])
            return Operand(mapping[key])
        groups=[];cost=0
        for group in boundary.groups[first:first+count]:
            n=ad.primal(boundary.inputs[_index(group['left'])]).shape[1]
            metric=group['metric']
            rows,columns=(metric.cartesian_shape if isinstance(metric,EncodedRead)
                          else ad.primal(boundary.inputs[_index(metric)]).shape)
            ranks=sum(ad.primal(boundary.inputs[_index(x)]).shape[1] for x in group['right'])
            cost+=n*columns*(rows+ranks)
            groups.append(dict(metric=remap(metric),left=remap(group['left']),
                right=tuple(remap(x) for x in group['right']),positions=group['positions']))
        prepared.append((dict(kind='t1_overlap_sources',groups=groups,resident_cost=max(1,cost)),tuple(inputs)))
    pool=_pool(state)
    if not pool.statistics.get('shared_store'):raise ValueError('Shared T1 overlaps require the canonical source store')
    plan=SharedRegionPlan(prepared,pool,owner=state if hasattr(state,'__weakref__') else None,balance_stage=True)
    plan.statistics.update(kind='t1_overlap_sources',groups=len(boundary.groups),overlaps=len(boundary.shapes),
                           packed_parent_edge=True,complete_target_groups=True)
    state._resident_region_plans.append(plan)
    state.diagnostics.setdefault('resident_regions',[]).append(plan.statistics)
    state.diagnostics['shared_t1_overlaps']=plan.statistics
    output=plan.packed_bridge()
    if output.size!=int(boundary.offsets[-1]):raise RuntimeError('Packed overlap boundary size differs')
    return output
