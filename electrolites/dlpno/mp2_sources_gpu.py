"""Batched, complete coupled-MP2 residual and source pullback on GPU0.

The scalar ResponseDLPNO residual is retained as an independent oracle.
No coefficient is frozen: pair K, PNO energies, both PNO coefficient
occurrences, the full local overlap metric and occupied Fock sources all
remain parents. Only the amplitude transpose stops at those coefficients.
"""
from contextlib import contextmanager
import gc,os,time,weakref
import numpy as np
import cupy as cp
from . import reverse as ad


def host(x):
    x=ad.primal(x)
    return cp.asnumpy(x) if isinstance(x,cp.ndarray) else np.asarray(x)


def primal(state):
    """Use the existing energy solver in the newly rebuilt MP2 PNO basis."""
    from .lmp2_gpu import LMP2GPU
    from .mgpu import devices
    if devices()!=[0]:raise ValueError('The MP2 source prototype requires primary GPU0')
    for name in ('S_pao','F_lmo'):
        matrix=getattr(state.orb,name)
        indices=np.ix_(*(np.arange(n) for n in matrix.shape))
        setattr(state.orb,name,host(matrix[indices]))
    state.S_pao_d=cp.asarray(state.orb.S_pao)
    try:return LMP2GPU.lmp2_iterations(state)
    finally:del state.S_pao_d


def _stack(values,shape):
    values=tuple(values);result=cp.zeros((len(values),)+tuple(shape),dtype=cp.float64)
    keys=[(i,)+tuple(slice(n) for n in x.shape) for i,x in enumerate(values)]
    for key,x in zip(keys,values):result[key]=ad.primal(x)
    def reverse(seed,needed):return tuple(seed[key].copy() if flag else None for key,flag in zip(keys,needed))
    return ad.operation(result,values,lambda seed:reverse(seed,[True]*len(values)),reverse)


def _mapped(vector,shape,positions,indices):
    positions=cp.asarray(positions,dtype=cp.int64);indices=cp.asarray(indices,dtype=cp.int64)
    result=cp.zeros(shape,dtype=cp.float64);result.ravel()[positions]=ad.primal(vector)[indices]
    def reverse(seed):
        bar=cp.zeros(vector.shape,dtype=cp.float64)
        cp.add.at(bar,indices,seed.ravel()[positions])
        return (bar,)
    return ad.operation(result,[vector],reverse)


def _add_rows(original,parts):
    """One global residual buffer; all partial arrays have bounded local rows.

    Copying the full pair array once per coupling chunk would make retained
    values and the transpose sweep quadratic at fixed local domain sizes.
    This single boundary instead gathers the global seed once per local row.
    """
    parts=tuple((cp.asarray(rows,dtype=cp.int64),value) for rows,value in parts)
    result=ad.primal(original).copy()
    for rows,value in parts:cp.add.at(result[:,:value.shape[1],:value.shape[2]],rows,ad.primal(value))
    operands=(original,)+tuple(value for _,value in parts)
    def reverse(seed,needed):
        return (seed if needed[0] else None,)+tuple(seed[rows,:value.shape[1],:value.shape[2]] if flag else None for (rows,value),flag in zip(parts,needed[1:]))
    return ad.operation(result,operands,lambda seed:reverse(seed,[True]*len(operands)),reverse)


def _projection_value(s,t,w):
    value=(s@t)@s.transpose(0,2,1)
    value*=-w[:,None,None]
    return value


def _projection_pullback(s,t,w,seed,needed):
    # t may be None for the amplitude-only transpose: its value is unused.
    transpose=lambda x:x.transpose(0,2,1)
    gs=gt=gw=None
    if needed[0]:
        gs=(seed@s)@transpose(t)
        gs+=(transpose(seed)@s)@t
        gs*=-w[:,None,None]
    if needed[1]:
        gt=(transpose(s)@seed)@s
        gt*=-w[:,None,None]
    if needed[2]:
        gw=-(seed*((s@t)@transpose(s))).sum(axis=(1,2))
    return gs,gt,gw


def _project_couplings(overlap,amplitudes,weights):
    """Exact nonsymmetric projection sources without retained GEMM products."""
    s,t,w=(ad.primal(x) for x in (overlap,amplitudes,weights))
    reverse=lambda seed,needed:_projection_pullback(s,t,w,seed,needed)
    return ad.operation(_projection_value(s,t,w),(overlap,amplitudes,weights),
        lambda seed:reverse(seed,(True,True,True)),reverse)


def _project_indexed_couplings(overlap,amplitudes,weights,sources):
    """Read current fixed global amplitudes transiently; scatter all T bars.

    Repeated source indices accumulate exactly through IndexedCotangent.
    A selected amplitude J.T does not need an amplitude-value gather at all.
    """
    s,t,w=(ad.primal(x) for x in (overlap,amplitudes,weights))
    sources=cp.asarray(sources,dtype=cp.int64)
    width=s.shape[2];key=(sources,slice(0,width),slice(0,width))
    value=_projection_value(s,t[key],w)
    def reverse(seed,needed):
        local=t[key] if needed[0] or needed[2] else None
        gs,gt,gw=_projection_pullback(s,local,w,seed,needed)
        if gt is not None:gt=ad.IndexedCotangent(t.shape,key,gt)
        return gs,gt,gw
    return ad.operation(value,(overlap,amplitudes,weights),
        lambda seed:reverse(seed,(True,True,True)),reverse)


class SourcePlan:
    """One bounded private GPU graph and one packed parent residual edge."""
    def __init__(self,state,variables,independent):
        self.device=0;self.parent_allocator=cp.cuda.get_allocator();self.pool=cp.cuda.MemoryPool();self.closed=False
        budget=int(os.environ.get('DLPNO_MP2_SOURCE_MEMORY_MIB','2048'))<<20
        workspace=int(os.environ.get('DLPNO_MP2_SOURCE_WORKSPACE_MIB','256'))<<20
        count=int(os.environ.get('DLPNO_MP2_SOURCE_BATCH','128'))
        if not 32<<20<=budget<=32<<30 or not 1<<20<=workspace<=2<<30 or not 1<=count<=1024:
            raise ValueError('MP2 source memory/batch outside bounds')
        self.pool.set_limit(size=budget);self.owner=weakref.ref(state) if hasattr(state,'__weakref__') else None
        self.parents=[];self.inputs=[];self.identities={};self.output=None;self.graph=None;self.amplitude=None
        self.statistics=dict(operator_kind='complete_batched_mp2_sources',coefficients_complete=True,released=False,
            memory_limit_bytes=budget,workspace_limit_bytes=workspace,reverse_calls=0,reverse_seconds=0.)
        begin=time.perf_counter()
        try:
            with self.context():self._build(state,variables,independent,count,workspace)
            self.statistics.update(build_seconds=time.perf_counter()-begin,retained_device_bytes=self.pool.used_bytes(),
                **self.graph.statistics)
            if self.pool.used_bytes()>budget:raise MemoryError('MP2 source graph exceeds private memory budget')
        except BaseException:self.release();raise

    @contextmanager
    def context(self):
        with cp.cuda.Device(self.device),cp.cuda.using_allocator(self.pool.malloc):yield

    def capture(self,value):
        key=('record',value._record) if isinstance(value,ad.Variable) else ('array',id(value))
        if key not in self.identities:
            raw=ad.primal(value)
            if raw.dtype!=np.float64:raise TypeError('MP2 source inputs must be float64')
            self.identities[key]=len(self.inputs);self.parents.append(value)
            self.inputs.append(ad.Variable(cp.array(raw,copy=True),tracked=isinstance(value,ad.Variable) and value._record.tracked))
        return self.inputs[self.identities[key]]

    def _build(self,state,variables,independent,count,workspace):
        from .overlap_union_adjoint_gpu import union_batches,embedded_batch,_pad
        pairs=tuple(state.pl.pairs);npair=len(pairs);ranks=[x.shape[0] for x in state.T];v=_pad(max(ranks,default=0))
        if 8*npair*v*v*12>self.statistics['memory_limit_bytes']:
            raise MemoryError('Padded MP2 pair matrices exceed bounded prototype memory')
        # Exact canonical-pair and diagonal-symmetry expansion.
        positions=[];indices=[];selected=[];offset=0
        for p in variables.pairs:
            n=ranks[p];row,col=variables.diag[p] if p in variables.diag else np.indices((n,n)).reshape(2,-1)
            source=np.arange(offset,offset+len(row));dest=p*v*v+row*v+col
            positions.extend(dest);indices.extend(source)
            # ResponseDLPNO assigns out[p], out[ji[p]] = r, r.T. The
            # second assignment wins for diagonal pairs, including under
            # independent nonsymmetric coefficient perturbations.
            selected.extend(p*v*v+col*v+row if p in variables.diag else dest)
            q=int(state.pl.ji[p])
            if q!=p:
                positions.extend(q*v*v+col*v+row);indices.extend(source)
            elif p in variables.diag:
                off=row!=col;positions.extend(p*v*v+col[off]*v+row[off]);indices.extend(source[off])
            offset+=len(row)
        if offset!=independent.size:raise ValueError('MP2 independent amplitude layout changed')
        self.amplitude=self.capture(independent)
        T=_mapped(self.amplitude,(npair,v,v),positions,indices)
        K=_stack([self.capture(x) for x in state.K],(v,v))
        e=_stack([self.capture(x) for x in state.e_pno],(v,))
        # Capture only occupied-Fock entries actually used by the local graph.
        fock={};fkeys={};fvalues=[]
        def fentry(i,j):
            key=(int(i),int(j))
            if key not in fkeys:
                fkeys[key]=len(fvalues)
                with cp.cuda.using_allocator(self.parent_allocator):x=state.orb.F_lmo[key]
                fock[key]=float(host(x));fvalues.append(x)
            return fkeys[key],fock[key]
        diagonal=[fentry(i,i)[0] for i in range(state.n)]
        edges=[];unique={}
        for p in variables.pairs:
            i,j=pairs[p]
            for k in np.union1d(state.pl.neighbors(i),state.pl.neighbors(j)):
                for q,a,b,active in [(state.pl.ij[k,j],i,k,i!=k),(state.pl.ij[i,k],k,j,j!=k)]:
                    if q<0 or not active:continue
                    fi,weight=fentry(a,b)
                    if abs(weight)<=state.opt['F_CUT']:continue
                    key=(p,int(q))
                    if key not in unique:unique[key]=len(unique)
                    edges.append((p,int(q),fi,key))
        # Parent boundary records outlive this private graph during VJP.
        # Their GPU values must use the caller's allocator.
        with cp.cuda.using_allocator(self.parent_allocator):
            fparent=ad.concatenate([x.reshape(1) if hasattr(x,'reshape') else np.asarray(x).reshape(1) for x in fvalues])
        f=self.capture(fparent)
        ii=cp.asarray([diagonal[i] for i,j in pairs]);jj=cp.asarray([diagonal[j] for i,j in pairs])
        D=e[:,:,None]+e[:,None,:]-f[ii,None,None]-f[jj,None,None]
        result=K+D*T;partials=[]
        # Complete local PAO metric/PNO coefficient overlap sources.
        direct_overlaps=os.environ.get('DLPNO_MP2_DIRECT_OVERLAPS','0')=='1'
        local_padding=os.environ.get('DLPNO_MP2_LOCAL_PADDING','0')=='1'
        fused_projection=os.environ.get('DLPNO_MP2_FUSED_PROJECTION','0')=='1'
        rank_buckets=os.environ.get('DLPNO_MP2_RANK_BUCKETS','0')=='1'
        indexed_projection=os.environ.get('DLPNO_MP2_INDEXED_PROJECTION','0')=='1'
        if rank_buckets and not local_padding:raise ValueError('MP2 rank buckets require local padding')
        if indexed_projection and not fused_projection:raise ValueError('Indexed MP2 projection requires fused projection')
        buckets={}
        for key in unique:
            shape=(_pad(ranks[key[0]]),_pad(ranks[key[1]])) if rank_buckets else None
            buckets.setdefault(shape,[]).append(key)
        groups=[group for keys in buckets.values() for group in
            union_batches(tuple(keys),state.pao,state.X,max_count=count,max_bytes=workspace,stable_groups=True)]
        group_ranks=[];group_shapes=[];global_partial_elements=0;identity_overlap_gathers=0
        grouped_edges={key:[] for key in unique}
        for row in edges:grouped_edges[row[3]].append(row)
        for group,union,involved,bound in groups:
            local={int(p):i for i,p in enumerate(involved)}
            group_rank=_pad(max(ranks[p] for p in involved)) if local_padding else v
            left_rank=_pad(max(ranks[p] for p,q in group)) if rank_buckets else group_rank
            right_rank=_pad(max(ranks[q] for p,q in group)) if rank_buckets else group_rank
            group_ranks.append(group_rank);group_shapes.append((left_rank,right_rank))
            with cp.cuda.using_allocator(self.parent_allocator):metric_parent=state.orb.S_pao[np.ix_(union,union)]
            metric=self.capture(metric_parent)
            coefficients=tuple(self.capture(state.X[p]) for p in involved)
            couples=np.array([(local[p],local[q]) for p,q in group],dtype=np.int64)
            if direct_overlaps:
                overlaps=embedded_batch(metric,coefficients,tuple(np.searchsorted(union,state.pao[p]) for p in involved),
                    couples,padded_output=(left_rank,right_rank))
            else:
                packed,shapes=embedded_batch(metric,coefficients,tuple(np.searchsorted(union,state.pao[p]) for p in involved),couples,packed_output=True)
                dest=[];source=[];off=0
                for a,(nr,nc) in enumerate(shapes):
                    row,col=np.indices((nr,nc)).reshape(2,-1)
                    dest.extend(a*left_rank*right_rank+row*right_rank+col);source.extend(np.arange(off,off+nr*nc));off+=nr*nc
                overlaps=_mapped(packed,(len(group),left_rank,right_rank),dest,source)
            rows=[row for key in group for row in grouped_edges[key]]
            lookup={key:i for i,key in enumerate(group)}
            overlap_rows=[lookup[row[3]] for row in rows]
            q=cp.asarray([row[1] for row in rows]);fi=cp.asarray([row[2] for row in rows])
            if indexed_projection and overlap_rows==list(range(len(group))):
                s=overlaps;identity_overlap_gathers+=1
            else:s=overlaps[cp.asarray(overlap_rows)]
            if indexed_projection:
                contribution=_project_indexed_couplings(s,T,f[fi],q)
            else:
                source_amplitudes=T[q,:right_rank,:right_rank] if right_rank<v else T[q]
                contribution=(_project_couplings(s,source_amplitudes,f[fi]) if fused_projection else
                              -f[fi,None,None]*(s@source_amplitudes@s.transpose(0,2,1)))
            global_partial_elements+=len(rows)*v*v
            partials.append(([row[0] for row in rows],contribution))
        result=_add_rows(result,partials)
        self.statistics.update(global_residual_accumulations=1,local_partial_elements=sum(x.size for _,x in partials),
            global_padded_partial_elements=global_partial_elements,local_batch_padding=local_padding,fused_projection=fused_projection,
            rank_buckets=rank_buckets,indexed_projection=indexed_projection,direct_overlaps=direct_overlaps,identity_overlap_gathers=identity_overlap_gathers,
            coupling_batch_shape_histogram={str(shape):group_shapes.count(shape) for shape in sorted(set(group_shapes))},
            coupling_batch_rank_histogram={str(rank):group_ranks.count(rank) for rank in sorted(set(group_ranks))})
        self.output=result.ravel()[cp.asarray(selected,dtype=cp.int64)]
        self.parents=tuple(self.parents);self.inputs=tuple(self.inputs)
        self.graph=ad.Linearization([self.output],self.inputs,boundary=self.inputs)
        self.statistics.update(pairs=npair,independent_pairs=len(variables.pairs),variables=independent.size,
            couplings=len(edges),overlaps=len(unique),batches=len(groups),occupied_fock_entries=len(fvalues),padded_rank=v)

    def bridge(self):
        with self.context():value=cp.asnumpy(ad.primal(self.output))
        owner=self.owner
        def reverse(seed,needed):
            if self.closed:raise RuntimeError('Consumed MP2 source plan cannot be reused')
            state=None if owner is None else owner();consume=state is not None and getattr(state,'_release_transposes',False)
            begin=time.perf_counter()
            try:
                with self.context():
                    selected=[x for x,flag in zip(self.inputs,needed) if flag]
                    graph=ad.Linearization([self.output],selected,boundary=self.inputs)
                    bars=iter(graph([cp.asarray(seed)],release_graph=consume,release_inputs=False))
                    result=[]
                    for parent,flag in zip(self.parents,needed):
                        if not flag:result.append(None);continue
                        bar=next(bars)
                        if isinstance(ad.primal(parent),cp.ndarray):
                            with cp.cuda.using_allocator(self.parent_allocator):result.append(bar.copy())
                        else:result.append(cp.asnumpy(bar))
                    self.statistics['last_reverse_statistics']=graph.last_statistics
                    bars=bar=graph=selected=None
                    self.statistics['reverse_calls']+=1
                    self.statistics['reverse_seconds']+=time.perf_counter()-begin
                return tuple(result)
            finally:
                if consume:self.release()
        return ad.operation(value,self.parents,lambda seed:reverse(seed,[True]*len(self.parents)),reverse)

    def release(self):
        if self.closed:return
        self.output=self.graph=self.amplitude=None;self.inputs=self.parents=();self.identities.clear();self.closed=True
        gc.collect()
        with self.context():cp.cuda.get_current_stream().synchronize();self.pool.free_all_blocks()
        self.statistics.update(released=True,live_bytes=self.pool.used_bytes(),reserved_bytes=self.pool.total_bytes())


class SourceTranspose:
    """Amplitude-only J.T reuses the complete source graph's current values."""
    xp=cp
    def __init__(self,plan,diagonal):
        self.plan=plan;self.released=False;self.calls=0
        with plan.context():
            self.graph=ad.Linearization([plan.output],[plan.amplitude],boundary=plan.inputs)
            self.diagonal=cp.asarray(diagonal)
        self.statistics=dict(operator_kind='resident_mp2_source_transpose',backend='cupy',shared_source_graph=True,
            amplitude_only=True,coefficients_approximated=False,**self.graph.statistics)
        self.last_statistics={}
    def __call__(self,seeds):
        if self.released:raise RuntimeError('Released MP2 transpose cannot be reused')
        start=time.perf_counter()
        with self.plan.context():result=self.graph([cp.asarray(seeds[0])])
        self.calls+=1;self.last_statistics=dict(self.graph.last_statistics,calls=self.calls,seconds=time.perf_counter()-start)
        return result
    def release(self):
        self.graph=self.diagonal=None;self.released=True
