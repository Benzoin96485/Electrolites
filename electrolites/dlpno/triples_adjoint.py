"""Compact, fully differentiated sources for a single triples checkpoint.

Only evaluation order changes. Each call owns its projection cache, so a
checkpoint replay rebuilds all geometry, PNO and amplitude dependencies.
The unmodified RefDLPNO._triplet_t0 remains an independent equation oracle.
"""
import numpy as np

from . import reverse as ad


def transfer(value, target):
    """An exact CPU/GPU graph edge, including the reverse data transfer."""
    source = ad._xp([value])
    if source is target:
        return value
    primal = ad.primal(value)
    moved = source.asnumpy(primal) if target is np else target.asarray(primal)
    def pullback(bar):
        return (target.asnumpy(bar) if source is np else source.asarray(bar),)
    return ad.operation(moved, [value], pullback)


def triplet_source(state, t, *, gpu=False):
    """Return E, W, V and T0 with shared projections and local GEMMs.

    For the positive definite RI metric A, (A**-.5 Q).T(A**-.5 R)
    equals Q.T solve(A, R). Keeping A in the reverse graph includes the
    complete metric derivative; no fit coefficients or subspaces are frozen.
    """
    pl, f = state.pl, state.orb.F_lmo
    i, j, k = state.trip[t]
    aux, paos, lm, X = state.t_ribfs[t], state.t_paos[t], state.t_lmos[t], state.X_tno[t]
    output_backend = ad._xp([X])
    if gpu:
        import cupy
        convert = lambda value: transfer(value, cupy)
    else:
        convert = lambda value: value
    X = convert(X)
    nv, na = X.shape[1], len(aux)
    occupied = tuple(dict.fromkeys((i, j, k)))
    qo = {a: convert(state.qij[np.ix_(aux, [a], lm)][:, 0]) for a in occupied}
    qv = {a: convert(state.qia[np.ix_(aux, [a], paos)][:, 0]) @ X for a in occupied}
    qvv = ad.einsum('ua,Quv,vb->Qab', X, convert(state.qab[np.ix_(aux, paos, paos)]), X)
    metric = convert(state.J2c[np.ix_(aux, aux)])
    # Solve all right-hand sides once, including repeated occupied labels
    # only once. Reverse mode accumulates every use into the shared solve.
    fitted = ad.solve(metric, ad.concatenate([qv[a] for a in occupied], axis=1))
    qvc = {a: fitted[:, pos*nv:(pos+1)*nv] for pos, a in enumerate(occupied)}
    Kvvv = {a: (qvc[a].T @ qvv.reshape(na, -1)).reshape(nv, nv, nv) for a in occupied}
    Kov = {(a, b): qo[a].T @ qvc[b] for a in occupied for b in occupied}

    overlaps, projected = {}, {}

    def overlap(p):
        if p not in overlaps:
            overlaps[p] = convert(state._S_tp(t, p))
        return overlaps[p]

    def pair(p):
        if p not in projected:
            s = overlap(p)
            projected[p] = s @ convert(state.T[p]) @ s.T
        return projected[p]

    # Sum over local occupied partners with one GEMM per permutation.
    # The same projected amplitudes enter both permutations with a fixed I.
    partners = {}
    if len(lm):
        for a in occupied:
            partners[a] = ad.concatenate([
                pair(pl.ij[a, l]).reshape(1, nv*nv) for l in lm
            ], axis=0).T
    perms = ((i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i))
    Wp = []
    for I, J, K in perms:
        w = (Kvvv[I].reshape(nv*nv, nv) @ pair(pl.ij[K, J]).T).reshape(nv, nv, nv)
        if len(lm):
            w = w - (partners[I] @ Kov[J, K]).reshape(nv, nv, nv)
        Wp.append(w)
    W = (Wp[0] + Wp[1].transpose(0, 2, 1) + Wp[2].transpose(1, 0, 2)
         + Wp[3].transpose(2, 0, 1) + Wp[4].transpose(1, 2, 0)
         + Wp[5].transpose(2, 1, 0))
    singles = {a: overlap(pl.ij[a, a]) @ convert(state.T1[a]) for a in occupied}
    Kjk, Kik, Kij = qv[j].T @ qvc[k], qv[i].T @ qvc[k], qv[i].T @ qvc[j]
    V = (W + singles[i][:, None, None] * Kjk[None]
         + singles[j][None, :, None] * Kik[:, None, :]
         + singles[k][None, None, :] * Kij[:, :, None])
    e = convert(state.e_tno[t])
    denominator = (e[:, None, None] + e[None, :, None] + e[None, None, :]
                   - convert(f[i, i]) - convert(f[j, j]) - convert(f[k, k]))
    T3 = -W / denominator
    values = (state._e_ijk(i, j, k, V, T3), W, V, T3)
    return tuple(transfer(value, output_backend) for value in values) if gpu else values
