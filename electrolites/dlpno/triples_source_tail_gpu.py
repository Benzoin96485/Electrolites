"""Exact packed triples source tail with a shared, selected GPU pullback.

This first fusion keeps RI fits and projected pairs in the existing graph.
No device intermediates are captured by a host-output checkpoint.
"""
import numpy as np
import cupy as cp
from . import reverse as ad

_AXES = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 0, 1), (1, 2, 0), (2, 1, 0))


def _axes(key):
    # Match ref_rhf._permute, including stable ordering of equal labels.
    a, b, c = key
    if a <= b <= c:
        return _AXES[0]
    if a <= c <= b:
        return _AXES[1]
    if b <= a <= c:
        return _AXES[2]
    if b <= c <= a:
        return _AXES[3]
    if c <= a <= b:
        return _AXES[4]
    return _AXES[5]


def source_tail(parts, singles, pairs, energies, occupied, labels, *,
                output_backend, energy_only=False):
    """Return (E,) or (E,W,V,T0), with all repeated operand edges retained."""
    operands = tuple(parts)+tuple(singles)+tuple(pairs)+(energies,)+tuple(occupied)
    values = tuple(ad.primal(x) for x in operands)
    backends = tuple(ad._xp([x]) for x in operands)
    if len(operands) != 16:
        raise ValueError('Expected six permutations, three singles and three pairs')
    n = values[12].size
    expected = [(n,)*3]*6+[(n,)]*3+[(n, n)]*3+[(n,)]+[()]*3
    if any(x.shape != shape or np.dtype(x.dtype) != np.float64
           for x, shape in zip(values, expected)):
        raise ValueError('Source tail requires consistent real float64 operands')
    i, j, k = labels
    pref = 1/6 if i == j == k else .5 if i == j or j == k or i == k else 1.
    spin = ((8., (0, 1, 2)), (-4., _axes((k, j, i))),
            (-4., _axes((i, k, j))), (-4., _axes((j, i, k))),
            (2., _axes((j, k, i))), (2., _axes((k, i, j))))

    def rebuild():
        w = cp.array(values[0], copy=True)
        for value, axes in zip(values[1:6], _AXES[1:]):
            w += cp.asarray(value).transpose(axes)
        s = [cp.asarray(x) for x in values[6:9]]
        p = [cp.asarray(x) for x in values[9:12]]
        v = w+s[0][:, None, None]*p[0][None]
        v += s[1][None, :, None]*p[1][:, None, :]
        v += s[2][None, None, :]*p[2][:, :, None]
        e = cp.asarray(values[12])
        d = e[:, None, None]+e[None, :, None]+e[None, None, :]
        for f in values[13:]:
            d -= cp.asarray(f)
        return w, v, -w/d, d, s, p

    def restore(x, xp):
        return cp.asnumpy(x) if xp is np else x

    def forward():
        w, v, t, _, _, _ = rebuild()
        energy = sum(c*cp.sum(v.transpose(axes)*t) for c, axes in spin)*pref
        if energy_only:
            return restore(energy.reshape(1), output_backend)
        return restore(cp.concatenate((energy.reshape(1), w.ravel(), v.ravel(), t.ravel())), output_backend)

    def pullback(seed, needed):
        z = cp.asarray(seed)
        w, v, t, d, s, p = rebuild()
        bv, bt = cp.zeros_like(v), cp.zeros_like(t)
        for c, axes in spin:
            bv += (pref*c*z[0])*t.transpose(tuple(np.argsort(axes)))
            bt += (pref*c*z[0])*v.transpose(axes)
        if not energy_only:
            bv += z[1+n**3:1+2*n**3].reshape((n,)*3)
            bt += z[1+2*n**3:].reshape((n,)*3)
        bars = [None]*16
        if any(needed[:6]):
            bw = bv-bt/d
            if not energy_only:
                bw += z[1:1+n**3].reshape((n,)*3)
            for a, axes in enumerate(_AXES):
                if needed[a]:
                    bars[a] = restore(bw.transpose(tuple(np.argsort(axes))), backends[a])
        singles_bars = (lambda: cp.sum(bv*p[0][None], axis=(1, 2)),
                        lambda: cp.sum(bv*p[1][:, None, :], axis=(0, 2)),
                        lambda: cp.sum(bv*p[2][:, :, None], axis=(0, 1)))
        pair_bars = (lambda: cp.sum(bv*s[0][:, None, None], axis=0),
                     lambda: cp.sum(bv*s[1][None, :, None], axis=1),
                     lambda: cp.sum(bv*s[2][None, None, :], axis=2))
        for a, fn in enumerate(singles_bars+pair_bars, 6):
            if needed[a]:
                bars[a] = restore(fn(), backends[a])
        if any(needed[12:]):
            bd = bt*w/(d*d)
            if needed[12]:
                bars[12] = restore(bd.sum((1, 2))+bd.sum((0, 2))+bd.sum((0, 1)), backends[12])
            for a in range(13, 16):
                if needed[a]:
                    bars[a] = restore(-bd.sum(), backends[a])
        return tuple(bars)

    result = ad.operation(forward(), operands,
                          lambda z: pullback(z, [True]*16), pullback)
    if energy_only:
        return (result[0],)
    return (result[0],)+(tuple(result[1+a*n**3:1+(a+1)*n**3].reshape((n,)*3) for a in range(3)))
