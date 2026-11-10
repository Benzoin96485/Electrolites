"""Shared PAO canonical bases and resident batched TNO source graphs.

Each distinct PAO domain has one differentiable canonical basis in a gradient
graph. Pair-density projections and the two TNO eigensystems are batched on
the selected GPUs. All shared input cotangents return through explicit AD
edges. The screening and degenerate-boundary rules match the scalar oracle.
"""
from concurrent.futures import ThreadPoolExecutor
import os
import time

import numpy as np
import cupy as cp
from . import reverse as ad
from .parallel_local_gpu import Capture


def _transpose(x):
    return x.transpose(tuple(range(x.ndim-2))+(x.ndim-1, x.ndim-2))


def batched_eigh(a):
    w, u = cp.linalg.eigh(ad.primal(a))
    tolerance = 32*max(1, w.shape[-1])*np.finfo(np.float64).eps*cp.max(cp.abs(w), axis=-1)
    gap = w[:, None, :]-w[:, :, None]
    mask = cp.abs(gap) > tolerance[:, None, None]
    inverse = cp.where(mask, 1/cp.where(mask, gap, 1.), 0.)
    values = ad.operation(w, [a], lambda b: ((u*b[:, None, :])@_transpose(u),))
    def reverse(bar):
        value = (_transpose(u)@bar)*inverse
        return (u@((value+_transpose(value))*.5)@_transpose(u),)
    vectors = ad.operation(u, [a], reverse)
    return values, vectors, tolerance


class Executor:
    def __init__(self, devices):
        self.devices = tuple(devices)
        if (not devices or len(set(devices)) != len(devices) or devices[0] != 0
                or min(devices) < 0 or max(devices) >= cp.cuda.runtime.getDeviceCount()):
            raise ValueError('TNO devices must be distinct visible IDs starting with 0')
        self.workers = ([ThreadPoolExecutor(max_workers=1) for _ in devices]
                        if len(devices) > 1 else [])
        self.closed = False

    def run(self, tasks):
        if self.closed:
            raise RuntimeError('TNO executor already closed')
        if not self.workers:
            return [batch.call(method, *args) for batch, method, args in tasks]
        futures = [self.workers[batch.rank].submit(batch.call, method, *args)
                   for batch, method, args in tasks]
        outputs, failures = [], []
        for future in futures:
            try:
                outputs.append(future.result())
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise failures[0]
        return outputs

    def close(self):
        for worker in self.workers:
            worker.shutdown(wait=True, cancel_futures=True)
        self.closed = True


def evaluate_subspace(spec, a):
    """Shared batched equations with every field an explicit AD operand."""
    b = len(spec['rows']); r = spec['r']
    projected = (_transpose(a['x'])@a['s'])@a['xp'][:,None]
    anti = 2*a['t']-_transpose(a['t'])
    density = anti@_transpose(a['t'])+_transpose(anti)@a['t']
    weights = cp.asarray([row['weights'] for row in spec['rows']])
    total = ((_transpose(projected)@density)@projected*weights[:,:,None,None]).sum(axis=1)/3.
    occ, u, tolerance = batched_eigh(total)
    occ, u = occ[:,::-1],u[:,:,::-1]
    occupations = cp.asnumpy(occ.v)
    ranks = np.maximum(1, np.sum((np.abs(occupations)>=np.array([row['cut'] for row in spec['rows']])[:,None])
                                 |(np.arange(r)[None,:]<spec['minimum']),axis=1))
    tol = cp.asnumpy(tolerance)
    for row, k in enumerate(ranks):
        if k < r and np.min(abs(occupations[row,:k,None]-occupations[row,None,k:])) <= tol[row]:
            raise ValueError('PNO/TNO selection splits a degenerate eigenvalue block')
    outputs = [None]*(2*b)
    for k in sorted(set(ranks)):
        rows = np.where(ranks==k)[0]
        selected = u[rows,:,:k]
        e, rotation, _ = batched_eigh(_transpose(selected)@(a['ep'][rows,:,None]*selected))
        x = (a['xp'][rows]@selected)@rotation
        for j, row in enumerate(rows):
            outputs[2*row],outputs[2*row+1]=x[j],e[j]
    return tuple(outputs)


class Batch:
    def __init__(self, rank, device, items, values, spec):
        self.rank, self.device, self.items = rank, device, items
        self.values, self.spec = values, spec
        with cp.cuda.Device(device):
            self.pool = cp.cuda.MemoryPool()
            self.stream = cp.cuda.Stream(non_blocking=True)

    def call(self, method, *args):
        with cp.cuda.Device(self.device), self.stream, cp.cuda.using_allocator(self.pool.malloc):
            result = getattr(self, method)(*args)
            self.stream.synchronize()
            return result

    def build(self):
        spec, values = self.spec, self.values
        b = len(spec['rows']);p, r = spec['p'], spec['r']
        q = max(row['q'] for row in spec['rows'])
        v = max(row['v'] for row in spec['rows'])
        shapes = dict(xp=(b,p,r), ep=(b,r), x=(b,3,q,v), t=(b,3,v,v), s=(b,3,q,p))
        field_offsets, length = {}, 0
        for field, shape in shapes.items():
            count = int(np.prod(shape));field_offsets[field]=(length,length+count);length+=count
        host = np.zeros(length)
        views = {field:host[a:z].reshape(shapes[field]) for field,(a,z) in field_offsets.items()}
        self.slots = []
        for j, row in enumerate(spec['rows']):
            for field in ('xp','ep'):
                index=row[field];views[field][j]=values[index];self.slots.append((field,j,index))
            for k in range(3):
                for field in ('x','t','s'):
                    index = row[field][k];value=values[index]
                    key=(j,k)+tuple(slice(n) for n in value.shape)
                    views[field][key]=value;self.slots.append((field,key,index))
        self.input = ad.Variable(cp.asarray(host))
        a = {field:self.input[first:last].reshape(shapes[field]) for field,(first,last) in field_offsets.items()}
        outputs = evaluate_subspace(spec, a)
        self.output = ad.concatenate([x.ravel() for x in outputs])
        self.shapes = tuple(x.shape for x in outputs)
        self.offsets = np.cumsum([0]+[x.size for x in outputs])
        self.graph = ad.Linearization([self.output], [self.input])
        self.field_offsets,self.shapes_in=field_offsets,shapes
        self.value_shapes = tuple(x.shape for x in values)
        self.values = self.spec = None
        return (cp.asnumpy(self.output.v), self.shapes,
                dict(device=self.device,items=len(self.items),records=self.graph.statistics['records'],
                     graph_value_bytes=self.graph.statistics['value_bytes'],resident_bytes=self.pool.used_bytes()))

    def reverse(self, seed, needed):
        bar = self.graph([cp.asarray(seed)])[0]
        host = cp.asnumpy(bar)
        fields = {field:host[a:z].reshape(self.shapes_in[field]) for field,(a,z) in self.field_offsets.items()}
        result = {}
        for field, key, index in self.slots:
            if needed[index]:
                contribution = fields[field][key]
                if index not in result:result[index]=contribution.copy()
                else:result[index]+=contribution
        return result

    def release(self):
        self.input=self.output=self.graph=self.values=self.spec=None
        self.stream.synchronize();self.pool.free_all_blocks()
        return dict(device=self.device,live_bytes=self.pool.used_bytes())


def _prepare(state, cut, cache):
    from .adjoint import orthocanonicalize
    capture = Capture();rows=[];reads={}
    def read(paos, other):
        key=tuple(map(int,paos)),tuple(map(int,other))
        if key not in reads:reads[key]=state.orb.S_pao[np.ix_(paos,other)]
        return reads[key]
    for t,(i,j,k) in enumerate(state.trip):
        paos=state.t_paos[t];key=(float(state.opt['S_CUT']),tuple(map(int,paos)))
        if key not in cache:
            cache[key]=orthocanonicalize(read(paos,paos),state.orb.F_pao[np.ix_(paos,paos)],state.opt['S_CUT'])
        xp,ep=cache[key]
        row=dict(t=t,p=xp.shape[0],r=xp.shape[1],xp=capture.value(xp).index,ep=capture.value(ep).index,
                 x=[],tensors=[],s=[],weights=[],q=0,v=0,cut=float(state.tno_scale[t])*cut)
        for a,b in ((i,j),(j,k),(i,k)):
            pair=state.pl.ij[a,b]
            row['x'].append(capture.value(state.X[pair]).index)
            row['tensors'].append(capture.value(state.T[pair]).index)
            row['s'].append(capture.value(read(state.pao[pair],paos)).index)
            row['weights'].append(.5 if a==b else 1.)
            row['q']=max(row['q'],state.X[pair].shape[0]);row['v']=max(row['v'],state.T[pair].shape[0])
        row['item']=row.pop('t');row['t']=row.pop('tensors');rows.append(row)
    return capture.inputs,rows


class TNOPlan:
    def __init__(self,state,cut,executor,*,cache=None,batch_size=16,memory_bytes=2048<<20):
        start=time.perf_counter();self.executor=executor;self.closed=False;self.batches=[]
        self.inputs,rows=_prepare(state,cut,{} if cache is None else cache)
        self.inputs=tuple(self.inputs);self.shapes=tuple(x.shape for x in self.inputs)
        values=tuple(np.asarray(ad.primal(x)) for x in self.inputs)
        if not 1<=batch_size<=64:raise ValueError('TNO batch size must be 1..64')
        buckets={}
        for row in rows:buckets.setdefault((row['p'],row['r']),[]).append(row)
        loads=[0]*len(executor.devices)
        for (p,r),group in buckets.items():
            for first in range(0,len(group),batch_size):
                chunk=group[first:first+batch_size]
                rank=min(range(len(loads)),key=lambda i:loads[i]);loads[rank]+=len(chunk)*r**3
                spec=dict(p=p,r=r,rows=chunk,minimum=state.opt['MIN_TNOS'])
                self.batches.append(Batch(rank,executor.devices[rank],tuple(row['item'] for row in chunk),values,spec))
        try:
            built=executor.run([(b,'build',()) for b in self.batches])
            totals={d:sum(info['resident_bytes'] for _,_,info in built if info['device']==d) for d in executor.devices}
            if max(totals.values(),default=0)>memory_bytes:raise ValueError('Resident TNO graphs exceed per-stage memory bound')
        except BaseException:
            self.release();raise
        self.pieces,self.output_shapes,self.output_map=[],[],[]
        for batch,(value,shapes,_) in zip(self.batches,built):
            self.pieces.append(value);self.output_shapes.extend(shapes)
            self.output_map.extend((t,kind) for t in batch.items for kind in (0,1))
        self.offsets=np.cumsum([0]+[x.size for x in self.pieces])
        self.statistics=dict(devices=list(executor.devices),triplets=len(rows),batches=len(self.batches),
            prepare_forward_seconds=time.perf_counter()-start,resident_bytes_by_device=totals,
            batches_info=[x[2] for x in built],reverse_seconds=0.,reverse_calls=0,released=False,
            unique_canonical_domains=len(cache) if cache is not None else None,memory_limit_per_stage_bytes=memory_bytes)

    def bridge(self, *, gpu_output=False):
        packed=np.concatenate(self.pieces);self.pieces=None
        if gpu_output:packed=cp.asarray(packed)
        def reverse(seed,needed):
            if self.closed:raise RuntimeError('Released TNO graph cannot be reused')
            start=time.perf_counter()
            host=cp.asnumpy(seed) if isinstance(seed,cp.ndarray) else seed
            results=self.executor.run([(b,'reverse',(host[a:z],needed)) for b,a,z in zip(self.batches,self.offsets[:-1],self.offsets[1:])])
            bars=[np.zeros(shape) if flag else None for shape,flag in zip(self.shapes,needed)]
            for result in results:
                for index,value in result.items():bars[index]+=value
            self.statistics['reverse_seconds']+=time.perf_counter()-start;self.statistics['reverse_calls']+=1
            return tuple(bars)
        value=ad.operation(packed,self.inputs,lambda z:reverse(z,[True]*len(self.inputs)),reverse)
        result={};start=0
        for key,shape in zip(self.output_map,self.output_shapes):
            end=start+int(np.prod(shape));result[key]=value[start:end].reshape(shape);start=end
        return [result[t,0] for t in range(len(result)//2)],[result[t,1] for t in range(len(result)//2)]

    def release(self):
        if self.closed:return
        stats=self.executor.run([(b,'release',()) for b in self.batches]);self.closed=True;self.inputs=()
        if hasattr(self,'statistics'):self.statistics.update(released=True,released_batches=stats)


def transform(state,cut):
    if os.environ.get('DLPNO_RESIDENT_SUBSPACES','0')=='1':
        from .resident_subspaces_gpu import tno_stage
        state.X_tno,state.e_tno=tno_stage(state,cut)
        return
    text=os.environ.get('DLPNO_TNO_BATCH_DEVICES','')
    devices=tuple(int(s) for s in text.split(','))
    if not hasattr(state,'_tno_batch_executor'):
        state._tno_batch_executor=Executor(devices);state._tno_batch_plans=[]
        if not hasattr(state,'_tno_basis_cache'):state._tno_basis_cache={}
    memory=int(os.environ.get('DLPNO_TNO_BATCH_MEMORY_MIB','2048'))<<20
    if not (32<<20)<=memory<=(8<<30):raise ValueError('TNO memory limit must be 32..8192 MiB')
    plan=TNOPlan(state,cut,state._tno_batch_executor,cache=state._tno_basis_cache,
                 batch_size=int(os.environ.get('DLPNO_TNO_BATCH_SIZE','16')),memory_bytes=memory)
    state._tno_batch_plans.append(plan)
    state.diagnostics.setdefault('batched_tno',[]).append(plan.statistics)
    state.X_tno,state.e_tno=plan.bridge(gpu_output=os.environ.get('DLPNO_TNO_OUTPUT_GPU','0')=='1')


def finish(state):
    for plan in getattr(state,'_tno_batch_plans',()):plan.release()
    executor=getattr(state,'_tno_batch_executor',None)
    if executor is not None:executor.close()
    for key in ('_tno_batch_plans','_tno_batch_executor','_tno_basis_cache'):state.__dict__.pop(key,None)
