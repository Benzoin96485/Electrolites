"""Bounded GPU bilinear contractions with exact coefficient and metric VJPs.

The outer correlation graph may live on the host. Each batch then transfers
only its local matrices and returns host outputs/bars; device workspaces do
not survive a forward or reverse action. No metric or coefficient response
is frozen, and no symmetry of the metric is assumed.
"""
import numpy as np
import cupy as cp

from . import reverse as ad


def _pad(n):
    return max(1, ((int(n)+7)//8)*8)


def workspace_bytes(count, shape):
    """Conservative device-array envelope, excluding BLAS/allocator overhead."""
    m,n,a,b=map(_pad,shape)
    return int(8*count*(3*m*n+4*m*a+4*n*b+3*a*b+2*m*b+2*n*a))


def batches(keys, pao, coefficients, max_bytes=256 << 20, max_count=128):
    """Keep small and large PAO blocks in separate bounded GPU batches."""
    groups={}
    for p,q in keys:
        shape=(len(pao[p]),len(pao[q]),coefficients[p].shape[1],coefficients[q].shape[1])
        bucket=tuple((int(x)+31)//32 for x in shape)
        groups.setdefault(bucket,[]).append(((p,q),shape))
    result=[]
    for items in groups.values():
        current=[];maximum=(0,0,0,0)
        for key,shape in items:
            enlarged=tuple(max(x,y) for x,y in zip(maximum,shape))
            if current and (len(current)>=max_count or workspace_bytes(len(current)+1,enlarged)>max_bytes):
                result.append((tuple(current),workspace_bytes(len(current),maximum)))
                current=[];maximum=(0,0,0,0);enlarged=shape
            if workspace_bytes(1,enlarged)>max_bytes:
                raise ValueError('One overlap exceeds the configured GPU array-workspace bound')
            current.append(key);maximum=enlarged
        if current:result.append((tuple(current),workspace_bytes(len(current),maximum)))
    return result


def bilinear_batch(left, metric, right):
    """Return L[p].T M[p] R[p] with all three exact analytic pullbacks.

    For output seed B, dL=M R B.T, dM=L B R.T and dR=M.T L B.
    Repeated input records accumulate normally in the outer linearization.
    The primitive supports both host and device graphs; computation is GPU.
    """
    left,metric,right=tuple(left),tuple(metric),tuple(right)
    if not (len(left)==len(metric)==len(right)):
        raise ValueError('Bilinear batch operand counts differ')
    if not left:return ()
    operands=left+metric+right
    xp=ad._xp(operands)
    values=tuple(ad.primal(x) for x in operands)
    count=len(left)
    shapes=[]
    for i in range(count):
        l,m,r=(values[i+count*j] for j in range(3))
        if any(x.ndim!=2 or x.dtype!=np.float64 for x in (l,m,r)):
            raise ValueError('Bilinear GPU batches require real float64 matrices')
        if (l.shape[0],r.shape[0])!=m.shape:
            raise ValueError('Bilinear GPU matrix dimensions differ')
        shapes.append((l.shape[0],r.shape[0],l.shape[1],r.shape[1]))
    pm,pn,pa,pb=tuple(_pad(max(s[j] for s in shapes)) for j in range(4))
    offsets=np.cumsum([0]+[s[2]*s[3] for s in shapes])

    def pack_values():
        packed=[xp.zeros((count,pm,pa)),xp.zeros((count,pm,pn)),xp.zeros((count,pn,pb))]
        for i,(m,n,a,b) in enumerate(shapes):
            packed[0][i,:m,:a]=values[i]
            packed[1][i,:m,:n]=values[count+i]
            packed[2][i,:n,:b]=values[2*count+i]
        return tuple(cp.asarray(x) for x in packed)

    def restore(value):
        return cp.asnumpy(value) if xp is np else value

    def forward():
        l,m,r=pack_values()
        y=cp.matmul(cp.matmul(l.transpose(0,2,1),m),r)
        return restore(cp.concatenate([y[i,:s[2],:s[3]].ravel() for i,s in enumerate(shapes)]))

    def pullback(seed,needed):
        l,m,r=pack_values()
        seed_buffer=xp.zeros((count,pa,pb))
        for i,shape in enumerate(shapes):
            seed_buffer[i,:shape[2],:shape[3]]=seed[offsets[i]:offsets[i+1]].reshape(shape[2:])
        bar=cp.asarray(seed_buffer)
        need_l=any(needed[:count]);need_m=any(needed[count:2*count]);need_r=any(needed[2*count:])
        dl=restore(cp.matmul(m,cp.matmul(r,bar.transpose(0,2,1)))) if need_l else None
        tmp=cp.matmul(l,bar) if need_m or need_r else None
        dm=restore(cp.matmul(tmp,r.transpose(0,2,1))) if need_m else None
        dr=restore(cp.matmul(m.transpose(0,2,1),tmp)) if need_r else None
        bars=[]
        for group,array in enumerate((dl,dm,dr)):
            for i,(nr,nc,na,nb) in enumerate(shapes):
                dims=((nr,na),(nr,nc),(nc,nb))[group]
                bars.append(array[i,:dims[0],:dims[1]].copy() if needed[group*count+i] else None)
        return tuple(bars)

    packed=ad.operation(forward(),operands,lambda seed:pullback(seed,[True]*len(operands)),pullback)
    outputs=[]
    for first,last,shape in zip(offsets[:-1],offsets[1:],shapes):
        key=slice(int(first),int(last));output_shape=shape[2:]
        def unpack_bar(bar,key=key):
            return (ad.IndexedCotangent(packed.shape,key,bar.ravel()),)
        outputs.append(ad.operation(packed.v[key].reshape(output_shape),[packed],unpack_bar))
    return tuple(outputs)
