"""One exact GPU operation for a local triples source and its selected VJP.

Primitive integral reads and common-left overlap pullbacks remain outside
this operation. All RI fits, pair projections, permutations and energy
terms inside it share one reverse sweep. A checkpoint replay reuses its
forward intermediates for that sweep only. No workspace survives a replay
or is captured by the retained graph.
"""
import numpy as np
import cupy as cp
from . import reverse as ad
from .triples_source_tail_gpu import _AXES, _axes
from .triples_overlap_adjoint_gpu import triplet_overlaps


def triplet_source(state, t, *, energy_only=False, max_workspace_bytes=2 << 30):
    """Return E/W/V/T0, or None to request the existing bounded fallback."""
    i, j, k = labels = tuple(state.trip[t])
    occ = tuple(dict.fromkeys(labels))
    perms = ((i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i))
    aux, paos, lm = state.t_ribfs[t], state.t_paos[t], tuple(state.t_lmos[t])
    ij = state.pl.ij
    projected_keys = tuple(dict.fromkeys(
        [ij[a, l] for a in occ for l in lm]+[ij[K, J] for I, J, K in perms]))
    overlap_keys = tuple(dict.fromkeys(list(projected_keys)+[ij[a, a] for a in occ]))
    x = state.X_tno[t]
    n, nq, npao = x.shape[1], len(aux), len(paos)
    ranks = [state.X[p].shape[1] for p in overlap_keys]
    envelope = 8*(48*n**3 + 6*nq*npao*npao + 12*nq*n*n + 6*nq*nq
                  + 16*nq*npao + 16*nq*n + 8*sum(r*r+n*r for r in ranks)
                  + 8*len(overlap_keys)*n*n + 8*len(occ)*len(lm)*n*n)
    if envelope > max_workspace_bytes:
        return None
    overlaps = triplet_overlaps(state, t, overlap_keys)
    operands = (x, state.J2c[np.ix_(aux, aux)], state.qab[np.ix_(aux, paos, paos)],
                state.e_tno[t])+tuple(state.orb.F_lmo[a, a] for a in labels)
    operands += tuple(state.qij[np.ix_(aux, [a], lm)][:, 0] for a in occ)
    operands += tuple(state.qia[np.ix_(aux, [a], paos)][:, 0] for a in occ)
    s0 = len(operands)
    operands += tuple(overlaps[p] for p in overlap_keys)
    t0 = len(operands)
    operands += tuple(state.T[p] for p in projected_keys)
    u0 = len(operands)
    operands += tuple(state.T1[a] for a in occ)
    values = tuple(ad.primal(x) for x in operands)
    backends = tuple(ad._xp([x]) for x in operands)
    count, no = len(operands), len(occ)
    if any(np.dtype(v.dtype) != np.float64 for v in values):
        raise ValueError('Fused triples source requires real float64 operands')
    spos = {p: s0+a for a, p in enumerate(overlap_keys)}
    tpos = {p: t0+a for a, p in enumerate(projected_keys)}
    opos = {a: pos for pos, a in enumerate(occ)}
    singles_keys = ((j, k), (i, k), (i, j))
    spin = ((8., _AXES[0]), (-4., _axes((k, j, i))),
            (-4., _axes((i, k, j))), (-4., _axes((j, i, k))),
            (2., _axes((j, k, i))), (2., _axes((k, i, j))))
    pref = 1/6 if i == j == k else .5 if len(occ) == 2 else 1.
    workspace_key = object()

    def rebuild():
        x, metric, raw, e = (cp.asarray(v) for v in values[:4])
        qo = {a: cp.asarray(values[7+pos]) for pos, a in enumerate(occ)}
        qi = {a: cp.asarray(values[7+no+pos]) for pos, a in enumerate(occ)}
        b = {a: qi[a] @ x for a in occ}
        fitted = cp.linalg.solve(metric, cp.concatenate([b[a] for a in occ], axis=1))
        c = {a: fitted[:, pos*n:(pos+1)*n] for pos, a in enumerate(occ)}
        h = (x.T[None] @ raw) @ x
        kv = {a: (c[a].T @ h.reshape(nq, n*n)).reshape(n*n, n) for a in occ}
        ko = {(a, b): qo[a].T @ c[b] for a in occ for b in occ}
        s = {p: cp.asarray(values[pos]) for p, pos in spos.items()}
        ts = {p: cp.asarray(values[pos]) for p, pos in tpos.items()}
        u = {p: s[p] @ ts[p] @ s[p].T for p in projected_keys}
        partner = {a: cp.stack([u[ij[a, l]].ravel() for l in lm], axis=1)
                   for a in occ} if lm else {}
        w = None
        for (I, J, K), axes in zip(perms, _AXES):
            part = kv[I] @ u[ij[K, J]].T
            if lm:
                part -= partner[I] @ ko[J, K]
            part = part.reshape((n,)*3).transpose(axes)
            if w is None:
                w = part.copy()
            else:
                w += part
        singles = {a: s[ij[a, a]] @ cp.asarray(values[u0+opos[a]]) for a in occ}
        pair = [b[a].T @ c[b_] for a, b_ in singles_keys]
        v = w+singles[i][:, None, None]*pair[0][None]
        v += singles[j][None, :, None]*pair[1][:, None, :]
        v += singles[k][None, None, :]*pair[2][:, :, None]
        denom = e[:, None, None]+e[None, :, None]+e[None, None, :]
        for f in values[4:7]:
            denom -= cp.asarray(f)
        t3 = -w/denom
        return x, metric, raw, qo, qi, b, c, h, kv, ko, s, ts, u, partner, singles, pair, w, v, denom, t3

    def restore(value, pos):
        return cp.asnumpy(value) if backends[pos] is np else value

    def forward():
        primals = rebuild()
        workspace = ad.checkpoint_replay_workspace()
        if workspace is not None:
            workspace[workspace_key] = primals
        *_, w, v, denom, t3 = primals
        energy = pref*sum(weight*cp.sum(v.transpose(axes)*t3) for weight, axes in spin)
        packed = energy.reshape(1) if energy_only else cp.concatenate(
            (energy.reshape(1), w.ravel(), v.ravel(), t3.ravel()))
        return restore(packed, 0)

    def pullback(seed, needed):
        z = cp.asarray(seed)
        workspace = ad.checkpoint_replay_workspace()
        primals = None if workspace is None else workspace.pop(workspace_key, None)
        if primals is None:
            primals = rebuild()
        x, metric, raw, qo, qi, b, c, h, kv, ko, s, ts, u, partner, singles, pair, w, v, denom, t3 = primals
        bars = [None]*count
        bv, bt = cp.zeros_like(v), cp.zeros_like(t3)
        for weight, axes in spin:
            bv += pref*weight*z[0]*t3.transpose(tuple(np.argsort(axes)))
            bt += pref*weight*z[0]*v.transpose(axes)
        if not energy_only:
            bv += z[1+n**3:1+2*n**3].reshape((n,)*3)
            bt += z[1+2*n**3:].reshape((n,)*3)
        bw = bv-bt/denom
        if not energy_only:
            bw += z[1:1+n**3].reshape((n,)*3)
        if any(needed[3:7]):
            bd = bt*w/(denom*denom)
            if needed[3]:
                bars[3] = restore(bd.sum((1, 2))+bd.sum((0, 2))+bd.sum((0, 1)), 3)
            for a in range(4, 7):
                if needed[a]:
                    bars[a] = restore(-bd.sum(), a)
        need_ri = any(needed[:3]) or any(needed[7:s0])
        need_proj = any(needed[s0:u0])
        gu = {p: cp.zeros_like(u[p]) for p in projected_keys} if need_proj else {}
        gkv = {a: cp.zeros_like(kv[a]) for a in occ} if need_ri else {}
        gko = {key: cp.zeros_like(value) for key, value in ko.items()} if need_ri else {}
        for (I, J, K), axes in zip(perms, _AXES):
            zp = bw.transpose(tuple(np.argsort(axes))).reshape(n*n, n)
            p = ij[K, J]
            if need_ri:
                gkv[I] += zp @ u[p]
                if lm:
                    gko[J, K] -= partner[I].T @ zp
            if need_proj:
                gu[p] += zp.T @ kv[I]
                if lm:
                    gp = -zp @ ko[J, K].T
                    for col, l in enumerate(lm):
                        gu[ij[I, l]] += gp[:, col].reshape(n, n)
        gs = {p: cp.zeros_like(s[p]) for p in overlap_keys if needed[spos[p]]}
        if need_proj:
            for p in projected_keys:
                g = gu[p]
                if needed[tpos[p]]:
                    bars[tpos[p]] = restore(s[p].T @ g @ s[p], tpos[p])
                if p in gs:
                    gs[p] += g @ s[p] @ ts[p].T + g.T @ s[p] @ ts[p]
        # Repeated occupied labels are additional uses of the same single.
        for a, g in zip(labels, (cp.sum(bv*pair[0][None], (1, 2)),
                                cp.sum(bv*pair[1][:, None, :], (0, 2)),
                                cp.sum(bv*pair[2][:, :, None], (0, 1)))):
            p, pos = ij[a, a], u0+opos[a]
            if needed[pos]:
                value = s[p].T @ g
                bars[pos] = value if bars[pos] is None else bars[pos]+value
            if p in gs:
                gs[p] += cp.outer(g, cp.asarray(values[pos]))
        for a in occ:
            pos = u0+opos[a]
            if needed[pos]:
                bars[pos] = restore(bars[pos], pos)
        for p, g in gs.items():
            bars[spos[p]] = restore(g, spos[p])
        if need_ri:
            gc = {a: cp.zeros_like(c[a]) for a in occ}
            gb = {a: cp.zeros_like(b[a]) for a in occ}
            go = {a: cp.zeros_like(qo[a]) for a in occ}
            gh = cp.zeros_like(h)
            for a in occ:
                g = gkv[a].reshape(n, n*n)
                gc[a] += h.reshape(nq, n*n) @ g.T
                gh += (c[a] @ g).reshape(nq, n, n)
            for (a, b_), g in gko.items():
                go[a] += c[b_] @ g.T
                gc[b_] += qo[a] @ g
            pair_bars = (cp.sum(bv*singles[i][:, None, None], axis=0),
                         cp.sum(bv*singles[j][None, :, None], axis=1),
                         cp.sum(bv*singles[k][None, None, :], axis=2))
            for (a, b_), g in zip(singles_keys, pair_bars):
                gb[a] += c[b_] @ g.T
                gc[b_] += b[a] @ g
            fitted_bar = cp.concatenate([gc[a] for a in occ], axis=1)
            fit_rhs_bar = cp.linalg.solve(metric.T, fitted_bar)
            if needed[1]:
                bars[1] = restore(-fit_rhs_bar @ cp.concatenate([c[a] for a in occ], axis=1).T, 1)
            gx = cp.zeros_like(x) if needed[0] else None
            for a, pos in opos.items():
                gb[a] += fit_rhs_bar[:, pos*n:(pos+1)*n]
                if needed[0]:
                    gx += qi[a].T @ gb[a]
                if needed[7+pos]:
                    bars[7+pos] = restore(go[a], 7+pos)
                if needed[7+no+pos]:
                    bars[7+no+pos] = restore(gb[a] @ x.T, 7+no+pos)
            if needed[2]:
                bars[2] = restore((x[None] @ gh) @ x.T, 2)
            if needed[0]:
                gx += cp.sum((raw @ x) @ gh.transpose(0, 2, 1), axis=0)
                gx += cp.sum((raw.transpose(0, 2, 1) @ x) @ gh, axis=0)
                bars[0] = restore(gx, 0)
        return tuple(bars)

    packed = ad.operation(forward(), operands,
                          lambda z: pullback(z, [True]*count), pullback)
    shapes = [()] if energy_only else [()]+[(n,)*3]*3
    offsets = [0, 1] if energy_only else [0, 1, 1+n**3, 1+2*n**3, 1+3*n**3]
    outputs = []
    for a, b_, shape in zip(offsets[:-1], offsets[1:], shapes):
        key = slice(a, b_)
        def output_bar(bar, key=key):
            return (ad.IndexedCotangent(packed.shape, key, bar.reshape(-1)),)
        outputs.append(ad.operation(packed.v[key].reshape(shape), [packed], output_bar))
    return tuple(outputs)
