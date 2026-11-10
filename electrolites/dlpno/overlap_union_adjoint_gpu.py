"""Exact GPU overlap pullbacks on a shared, bounded PAO union.

This follows the energy OverlapGen embedding: coefficients of each involved
pair are embedded once into the union and share one metric block. The VJP
aggregates coefficient sources before applying the metric, and returns one
metric cotangent per batch. Neither metric symmetry nor unique PAO indices
are assumed. No quantum-chemical approximation is introduced.
"""
import numpy as np
import cupy as cp

from . import reverse as ad


def _pad(n):
    return max(1,((int(n)+7)//8)*8)


def union_workspace_bytes(pair_count,couple_count,union_size,maximum_rank):
    """Device-array envelope; excludes external arrays, BLAS and allocator."""
    p,b,u,v=int(pair_count),int(couple_count),_pad(union_size),_pad(maximum_rank)
    return int(8*(6*p*u*v+3*b*u*v+3*u*u+3*b*v*v))


def union_batches(keys,pao,coefficients,max_bytes=256 << 20,max_count=256):
    """Sort couples by their right pair and form bounded common-PAO batches."""
    keys=tuple(sorted(keys,key=lambda key:(key[1],key[0])))
    if not keys:return []
    used={p for key in keys for p in key}
    pao_sets={p:set(map(int,pao[p])) for p in used}
    width={p:coefficients[p].shape[1] for p in used}
    result=[];current=[];involved=set();union=set();maximum=0
    def finish():
        result.append((tuple(current),np.asarray(sorted(union),dtype=np.int64),np.asarray(sorted(involved),dtype=np.int64),union_workspace_bytes(len(involved),len(current),len(union),maximum)))
    for key in keys:
        additions=set(key)-involved
        trial_pairs=involved|additions
        trial_union=union.union(*(pao_sets[p] for p in additions)) if additions else union
        trial_rank=max([maximum]+[width[p] for p in additions])
        bound=union_workspace_bytes(len(trial_pairs),len(current)+1,len(trial_union),trial_rank)
        if current and (len(current)>=max_count or bound>max_bytes):
            finish();current=[];involved=set();union=set();maximum=0
            trial_pairs=set(key);trial_union=set().union(*(pao_sets[p] for p in trial_pairs));trial_rank=max(width[p] for p in trial_pairs)
            bound=union_workspace_bytes(len(trial_pairs),1,len(trial_union),trial_rank)
        if bound>max_bytes:raise ValueError('One embedded overlap exceeds the GPU array-workspace bound')
        current.append(key);involved=trial_pairs;union=trial_union;maximum=trial_rank
    if current:finish()
    return result


def embedded_batch(metric,coefficients,positions,couples):
    """Return C[p].T M C[q] and full shared coefficient/metric cotangents.

    C is formed by additive embedding of local coefficient rows. For seeds
    B[p,q], D[p]=sum_q C[q] B[p,q].T and A[q]=sum_p C[p] B[p,q].
    Then bar_C=M D + M.T A and bar_M=sum_q A[q] C[q].T.
    These formulas retain both sides even when p=q or M is nonsymmetric.
    """
    coefficients=tuple(coefficients);positions=tuple(np.asarray(x,dtype=np.int64) for x in positions)
    pairs=np.asarray(couples,dtype=np.int64).reshape(-1,2)
    if not len(pairs):return ()
    if len(coefficients)!=len(positions):raise ValueError('Coefficient/PAO position counts differ')
    operands=(metric,)+coefficients;xp=ad._xp(operands);values=tuple(ad.primal(x) for x in operands)
    matrix=values[0];xs=values[1:];count=len(xs)
    if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or matrix.dtype!=np.float64:raise ValueError('The union metric must be square real float64')
    n=matrix.shape[0]
    if min(pairs.ravel())<0 or max(pairs.ravel())>=count:raise ValueError('Invalid embedded overlap couple')
    for x,pos in zip(xs,positions):
        if x.ndim!=2 or x.dtype!=np.float64 or x.shape[0]!=len(pos):raise ValueError('Invalid embedded coefficient matrix')
        if np.any(pos<0) or np.any(pos>=n):raise ValueError('Embedded coefficient row is outside the PAO union')
    ranks=tuple(x.shape[1] for x in xs)
    u,v=_pad(n),_pad(max(ranks,default=0));pidx=cp.asarray(pairs[:,0]);qidx=cp.asarray(pairs[:,1])
    shapes=tuple((ranks[p],ranks[q]) for p,q in pairs);offsets=np.cumsum([0]+[a*b for a,b in shapes])
    unique=tuple(len(np.unique(pos))==len(pos) for pos in positions)

    def restore(value):return cp.asnumpy(value) if xp is np else value

    def inputs():
        c=xp.zeros((count,u,v));m=xp.zeros((u,u));m[:n,:n]=matrix
        for i,(x,pos,rank,is_unique) in enumerate(zip(xs,positions,ranks,unique)):
            if not rank:continue
            if is_unique:c[i,pos,:rank]=x
            else:xp.add.at(c[i,:,:rank],xp.asarray(pos),x)
        return cp.asarray(c),cp.asarray(m)

    def forward():
        c,m=inputs();mc=cp.matmul(m,c)
        y=cp.matmul(c[pidx].transpose(0,2,1),mc[qidx])
        return restore(cp.concatenate([y[i,:a,:b].ravel() for i,(a,b) in enumerate(shapes)]))

    def pullback(seed,needed):
        want_m=needed[0];want_c=any(needed[1:])
        if not (want_m or want_c):return (None,)*len(operands)
        c,m=inputs();seeds=xp.zeros((len(pairs),v,v))
        for i,(a,b) in enumerate(shapes):seeds[i,:a,:b]=seed[offsets[i]:offsets[i+1]].reshape(a,b)
        bar=cp.asarray(seeds)
        a=cp.zeros_like(c)
        term=cp.matmul(c[pidx],bar);cp.add.at(a,qidx,term);del term
        gm=None;gc=None
        if want_m:
            gm=restore(a.transpose(1,0,2).reshape(u,count*v)@c.transpose(1,0,2).reshape(u,count*v).T)
        if want_c:
            d=cp.zeros_like(c)
            term=cp.matmul(c[qidx],bar.transpose(0,2,1));cp.add.at(d,pidx,term);del term
            gc=cp.matmul(m,d);gc+=cp.matmul(m.T,a);gc=restore(gc)
        result=[gm[:n,:n].copy() if want_m else None]
        result.extend(gc[i,pos,:rank].copy() if need else None for i,(pos,rank,need) in enumerate(zip(positions,ranks,needed[1:])))
        return tuple(result)

    packed=ad.operation(forward(),operands,lambda seed:pullback(seed,[True]*len(operands)),pullback)
    outputs=[]
    for first,last,shape in zip(offsets[:-1],offsets[1:],shapes):
        key=slice(int(first),int(last))
        def output_bar(bar,key=key):return (ad.IndexedCotangent(packed.shape,key,bar.ravel()),)
        outputs.append(ad.operation(packed.v[key].reshape(shape),[packed],output_bar))
    return tuple(outputs)
