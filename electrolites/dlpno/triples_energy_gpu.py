"""Bounded final T1 energy contraction with complete V3 and T3 sources.

For each occupied triplet, E = pref * sum_p c[p] <P[p] V, T>.
The two exact source bars are pref * sum_p c[p] P[p]^{-1} T and
pref * sum_p c[p] P[p] V. Stable occupied-label ties retain the reference
permutations. Batching changes neither amplitudes nor subspace responses.
"""
import numpy as np
from . import reverse as ad


_COEFFICIENTS = (8., -4., -4., -4., 2., 2.)


def _permutations(i, j, k):
    labels = ((i,j,k), (k,j,i), (i,k,j), (j,i,k), (j,k,i), (k,i,j))
    # The first term of RefDLPNO._e_ijk always uses V itself.
    return ((0,1,2),) + tuple(tuple(np.argsort(np.argsort(key,kind='stable'),kind='stable'))
                              for key in labels[1:])


def energy_vector(triplets, integrals, amplitudes, *, backend='cupy',
                  max_batch=64, workspace_bytes=256 << 20):
    """One host output edge; every original source backend is preserved.

    CUDA intermediates live only during each bounded forward/reverse batch.
    Inputs are captured as immutable record values. Selected/repeated VJPs,
    duplicate parents, empty tensors and consuming outer sweeps are supported.
    """
    if backend not in ('numpy','cupy'):
        raise ValueError('Fused triples energy backend must be numpy or cupy')
    if not 1 <= max_batch <= 256 or not 1024 <= workspace_bytes <= 2 << 30:
        raise ValueError('Fused triples energy batch/workspace outside bounds')
    if backend == 'cupy':
        import cupy as xp
    else:
        xp = np
    triplets, integrals, amplitudes = tuple(triplets), tuple(integrals), tuple(amplitudes)
    count = len(triplets)
    if len(integrals) != count or len(amplitudes) != count:
        raise ValueError('Triplet and source counts differ')
    operands = integrals + amplitudes
    values = tuple(ad.primal(x) for x in operands)
    backends = tuple(ad._xp([x]) for x in operands)
    groups = {}
    for t, (labels, v, a) in enumerate(zip(triplets, values[:count], values[count:])):
        if (v.ndim != 3 or v.shape != a.shape or len(set(v.shape)) != 1
                or v.dtype != np.float64 or a.dtype != np.float64):
            raise ValueError('Fused triples energy requires cubic real float64 sources')
        i,j,k = labels
        pref = 1./6. if i == j == k else .5 if i == j or j == k or i == k else 1.
        key = (v.shape[0], _permutations(i,j,k), pref)
        groups.setdefault(key, []).append(t)
    batches = []
    for (n, permutations, pref), rows in groups.items():
        per_row = 12 * max(1,n**3) * 8
        if per_row > workspace_bytes:
            raise ValueError('One fused triples energy tensor exceeds workspace bound')
        step = min(max_batch, max(1,workspace_bytes//per_row))
        inverses = tuple(tuple(np.argsort(p)) for p in permutations)
        for start in range(0,len(rows),step):
            batches.append((tuple(rows[start:start+step]),permutations,inverses,pref))

    def host(value):
        return value if isinstance(value,np.ndarray) else value.get()

    def pack(rows, offset):
        items = [values[offset+t] for t in rows]
        if all(backends[offset+t] is np for t in rows):
            # One H2D transfer per group instead of one transfer per triplet.
            return xp.asarray(np.stack(items))
        if xp is np:
            return np.stack([host(x) for x in items])
        return xp.stack([xp.asarray(x) for x in items])

    def combine(value, permutations):
        result = _COEFFICIENTS[0] * value.transpose((0,)+tuple(x+1 for x in permutations[0]))
        for weight, permutation in zip(_COEFFICIENTS[1:],permutations[1:]):
            result += weight * value.transpose((0,)+tuple(x+1 for x in permutation))
        return result

    result = np.empty(count,dtype=np.float64)
    for rows, permutations, _, pref in batches:
        v, a = pack(rows,0), pack(rows,count)
        value = (combine(v,permutations)*a).sum(axis=(1,2,3))*pref
        result[list(rows)] = host(value)
        del v,a,value

    def reverse(seed, needed):
        bars = [None] * (2*count)
        for rows, permutations, inverses, pref in batches:
            scale = xp.asarray(np.asarray(seed)[list(rows)]*pref)[:,None,None,None]
            for offset, other, maps in ((0,count,inverses),(count,0,permutations)):
                if not any(needed[offset+t] for t in rows):
                    continue
                value = combine(pack(rows,other),maps)
                value *= scale
                if xp is not np and any(backends[offset+t] is np for t in rows):
                    numeric_host = host(value)
                else:
                    numeric_host = value if xp is np else None
                for j,t in enumerate(rows):
                    if needed[offset+t]:
                        target = backends[offset+t]
                        bars[offset+t] = (numeric_host[j].copy() if target is np
                            else target.array(value[j] if xp is not np else numeric_host[j],copy=True))
                del value,numeric_host
        return tuple(bars)

    return ad.operation(result, operands, lambda seed: reverse(seed,[True]*len(operands)),reverse)
