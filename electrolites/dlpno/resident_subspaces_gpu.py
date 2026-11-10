"""Complete PNO/TNO subspace graphs owned by the existing source workers.

Canonical inputs and exact source cotangents share the persistent store.
PAO canonicalization stays in the parent with its original differentiable
cache. No eigensystem, rank rule, screening correction or response is frozen.
"""
import os,time
import numpy as np
import cupy as cp
from . import reverse as ad


def _field_layout(spec, bank):
    rows=spec['rows'];b=len(rows);p,r=spec['p'],spec['r'];slots=[]
    if spec['kind']=='pno_subspace':
        shapes=dict(x=(b,p,r),e=(b,r))
        if spec['subspace']=='recompute':shapes.update(k=(b,r,r),t=(b,r,r))
        else:
            na=len(rows[0]['aux'])
            shapes.update(m=(na,na),qi=(b,na,p),qj=(b,na,p),f=(b,))
        for field in shapes:
            slots.extend((field,key,index) for key,index in
                ([(slice(None),rows[0][field])] if field=='m' else [(j,row[field]) for j,row in enumerate(rows)]))
    else:
        q=max(row['q'] for row in rows);v=max(row['v'] for row in rows)
        shapes=dict(xp=(b,p,r),ep=(b,r),x=(b,3,q,v),t=(b,3,v,v),s=(b,3,q,p))
        for j,row in enumerate(rows):
            slots.extend((field,j,row[field]) for field in ('xp','ep'))
            for k in range(3):
                for field in ('x','t','s'):
                    index=row[field][k]
                    key=(j,k)+tuple(slice(n) for n in bank[index].shape)
                    slots.append((field,key,index))
    return shapes,slots


def subspace_outputs(spec, bank):
    """Pack on the worker, summing aliases exactly in the inverse packing."""
    shapes,slots=_field_layout(spec,bank);offsets={};length=0
    for field,shape in shapes.items():
        count=int(np.prod(shape));offsets[field]=(length,length+count);length+=count
    leaves=tuple(bank.inputs)
    raw=cp.zeros(length,dtype=cp.float64)
    def views(value):return {k:value[a:b].reshape(shapes[k]) for k,(a,b) in offsets.items()}
    fields=views(raw)
    for field,key,index in slots:fields[field][key]=ad.primal(leaves[index])
    def reverse(seed,needed):
        z=views(seed);bars=[cp.zeros(x.shape,dtype=cp.float64) if flag else None for x,flag in zip(leaves,needed)]
        for field,key,index in slots:
            if needed[index]:bars[index]+=z[field][key]
        return tuple(bars)
    packed=ad.operation(raw,leaves,lambda seed:reverse(seed,[True]*len(leaves)),reverse)
    fields=views(packed)
    if spec['kind']=='pno_subspace':
        from .pno_batch_gpu import evaluate_subspace
        return evaluate_subspace(dict(spec,kind=spec['subspace']),fields)
    from .tno_batch_gpu import evaluate_subspace
    return evaluate_subspace(spec,fields)


def _localize(inputs,rows,fields,sequences=()):
    """Each task names only its inputs; shared store identities join stages."""
    local=[];indices={};result=[]
    def remap(index):
        if index not in indices:indices[index]=len(local);local.append(inputs[index])
        return indices[index]
    for row in rows:
        item=dict(row)
        for field in fields:item[field]=remap(row[field])
        for field in sequences:item[field]=[remap(i) for i in row[field]]
        result.append(item)
    return tuple(local),result


def _plan(state,prepared,memory,label,start):
    from .resident_regions_gpu import _pool
    from .resident_shared_store_gpu import SharedRegionPlan
    if os.environ.get('DLPNO_RESIDENT_SHARED_STORE','0')!='1' or not os.environ.get('DLPNO_RESIDENT_REGION_DEVICES'):
        raise ValueError('Resident subspaces require configured shared source workers')
    if not hasattr(state,'diagnostics'):state.diagnostics={}
    if not hasattr(state,'_resident_region_plans'):state._resident_region_plans=[]
    pool=_pool(state)
    if not pool.statistics.get('shared_store'):raise ValueError('Subspaces need the shared source store')
    ranks=None
    parallel_recompute=label=='pno_recompute' and os.environ.get('DLPNO_RESIDENT_PARALLEL_RECOMPUTE','0')=='1'
    if label.startswith('pno_') and not parallel_recompute and os.environ.get('DLPNO_RESIDENT_PRIMARY_PNO','0')=='1' and len(pool.devices)>1:
        if not pool.statistics.get('inline_primary'):raise ValueError('Primary PNO routing requires an inline primary')
        ranks=(0,)
    plan=SharedRegionPlan(prepared,pool,owner=state if hasattr(state,'__weakref__') else None,balance_stage=True,ranks=ranks)
    totals={device:0 for device in pool.devices}
    for (rank,_),graph in zip(plan.assignments,plan.statistics['worker_graphs']):
        totals[pool.devices[rank]]+=graph['retained_device_bytes']
    if max(totals.values(),default=0)>memory:raise MemoryError('Resident subspace graphs exceed per-stage memory bound')
    plan.statistics.update(kind=label,prepare_forward_seconds=time.perf_counter()-start,
        retained_graph_bytes_by_device=totals,memory_limit_per_stage_bytes=memory,
        shared_canonical_inputs_counted_in_pool_budget=True,released=False)
    state._resident_region_plans.append(plan)
    state.diagnostics.setdefault('resident_regions',[]).append(plan.statistics)
    state.diagnostics.setdefault('shared_subspaces',[]).append(plan.statistics)
    return plan


def _options(name):
    count=int(os.environ.get('DLPNO_'+name+'_BATCH_SIZE','16'))
    memory=int(os.environ.get('DLPNO_'+name+'_BATCH_MEMORY_MIB','2048'))<<20
    if not 1<=count<=64 or not 32<<20<=memory<=8<<30:raise ValueError('Subspace batch or memory outside bounds')
    return count,memory


def pno_stage(state,kind,*,items=None):
    from .pno_batch_gpu import _prepare
    if kind not in ('sc_mp2','pno','recompute'):raise ValueError('Unknown PNO subspace kind')
    begin=time.perf_counter();count,memory=_options('PNO')
    if not hasattr(state,'_tno_basis_cache'):state._tno_basis_cache={}
    items=tuple(p for p,(i,j) in enumerate(state.pl.pairs) if i<=j) if items is None else tuple(items)
    inputs,rows=_prepare(state,kind,state._tno_basis_cache,items);buckets={}
    for row in rows:buckets.setdefault((row['aux'],row['p'],row['r']),[]).append(row)
    prepared=[];order=[]
    fields=('x','e','k','t') if kind=='recompute' else ('x','e','m','qi','qj','f')
    for (aux,p,r),group in buckets.items():
        for first in range(0,len(group),count):
            local,chunk=_localize(inputs,group[first:first+count],fields)
            spec=dict(kind='pno_subspace',subspace=kind,p=p,r=r,rows=chunk,options=dict(state.opt),
                resident_cost=len(aux)**3+len(chunk)*(r**3+len(aux)*p*r))
            prepared.append((spec,local));order.extend(row['item'] for row in chunk)
    if not prepared:return {}
    plan=_plan(state,prepared,memory,'pno_'+kind,begin);outputs=plan.bridge();stride=1 if kind=='sc_mp2' else 5
    return {p:tuple(outputs[stride*j:stride*(j+1)]) for j,p in enumerate(order)}


def tno_stage(state,cut):
    from .tno_batch_gpu import _prepare
    begin=time.perf_counter();count,memory=_options('TNO')
    if not hasattr(state,'_tno_basis_cache'):state._tno_basis_cache={}
    inputs,rows=_prepare(state,cut,state._tno_basis_cache);buckets={}
    for row in rows:buckets.setdefault((row['p'],row['r']),[]).append(row)
    prepared=[];order=[]
    for (p,r),group in buckets.items():
        for first in range(0,len(group),count):
            local,chunk=_localize(inputs,group[first:first+count],('xp','ep'),('x','t','s'))
            spec=dict(kind='tno_subspace',p=p,r=r,rows=chunk,minimum=state.opt['MIN_TNOS'],resident_cost=len(chunk)*r**3)
            prepared.append((spec,local));order.extend(row['item'] for row in chunk)
    if not prepared:return [],[]
    plan=_plan(state,prepared,memory,'tno',begin)
    outputs=plan.bridge(gpu_output=os.environ.get('DLPNO_TNO_OUTPUT_GPU','0')=='1')
    x=[None]*len(rows);e=[None]*len(rows)
    for j,t in enumerate(order):x[t],e[t]=outputs[2*j:2*j+2]
    return x,e
