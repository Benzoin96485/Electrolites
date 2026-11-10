"""Bounded cross-triplet E/W/V/T0 operation with complete source pullbacks.

Padding is algebraic only: auxiliary metrics have identity padding, all
coefficients have zero padding, and padding energies are positive constants.
Every physical operand remains an AD edge, including repeated occupied
labels, shared amplitudes, RI metrics, and common-left overlap sources.
Only a checkpoint replay may retain a batch's forward workspace.
"""
import numpy as np
import cupy as cp
from . import reverse as ad
from .triples_source_tail_gpu import _AXES, _axes
from .triples_overlap_adjoint_gpu import triplet_overlaps

_PERMS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def _layout(state, t):
    labels = tuple(state.trip[t])
    lm = tuple(state.t_lmos[t])
    ij = state.pl.ij
    projected = tuple(dict.fromkeys(
        [ij[a, l] for a in labels for l in lm] +
        [ij[labels[k], labels[j]] for i, j, k in _PERMS]))
    overlaps = tuple(dict.fromkeys(projected + tuple(ij[a, a] for a in labels)))
    n = state.X_tno[t].shape[1]
    return dict(t=t, labels=labels, lm=lm, projected=projected, overlaps=overlaps,
                n=n, q=len(state.t_ribfs[t]), p=len(state.t_paos[t]),
                r=max((state.X[p].shape[1] for p in overlaps), default=0),
                l=len(lm), k=len(overlaps))


def _dimensions(layouts):
    return (len(layouts),) + tuple(max(d[key] for d in layouts) for key in ('n', 'q', 'p', 'r', 'l', 'k'))


def workspace_bytes(layouts):
    if not layouts:
        return 0
    b, n, q, p, r, l, k = _dimensions(layouts)
    # Includes packed inputs/outputs, forward/reverse arrays and GEMM temporaries.
    return 8*b*(128*n**3 + 12*q*p*p + 24*q*q + 48*q*n*n +
                32*q*(p+l+n) + 20*k*(r*r+n*r+n*n) + 32*l*n*n + 24*p*n + 128)


def source_batches(state, *, max_count=8, max_bytes=512 << 20):
    if max_count < 1 or max_bytes < 1:
        raise ValueError('Positive batch count and workspace limit required')
    layouts = [_layout(state, t) for t in range(len(state.trip))]
    # Sort distinct bounded local shapes, preserving the previous order.
    # With bounded local dimensions the number of keys is independent of N;
    # this avoids sorting all N triplets in the conditional-local regime.
    buckets = {}
    for d in layouts:
        buckets.setdefault((d['n'], d['q'], d['p'], d['l']), []).append(d)
    layouts = [d for key in sorted(buckets) for d in buckets[key]]
    groups, current = [], []
    for d in layouts:
        trial = current + [d]
        # Keep cubic padding bounded, as well as the absolute live workspace.
        useful = sum(max(x['n'], 1)**3 for x in trial)
        padded = len(trial)*max(max(x['n'], 1) for x in trial)**3
        if current and (len(trial) > max_count or workspace_bytes(trial) > max_bytes or padded > 2*useful):
            groups.append(tuple(x['t'] for x in current))
            current = []
        current.append(d)
    if current:
        groups.append(tuple(x['t'] for x in current))
    return groups


def source_batch(state, triplets, *, energy_only=False, max_workspace_bytes=512 << 20, reuse_reads=True):
    """Return flat per-triplet outputs, or None before allocating an oversized batch."""
    layouts = [_layout(state, t) for t in triplets]
    if not layouts:
        return ()
    if workspace_bytes(layouts) > max_workspace_bytes:
        return None
    B, N, Q, P, R, L, K = _dimensions(layouts)
    shapes = dict(x=(B, P, N), metric=(B, Q, Q), raw=(B, Q, P, P),
                  e=(B, N), f=(B, 3), qo=(B, 3, Q, L), qi=(B, 3, Q, P),
                  s=(B, K, N, R), ts=(B, K, R, R), t1=(B, 3, R))
    operands, slots = [], []
    pair_index = np.zeros((B, 3, 3), dtype=np.int64)
    partner_index = np.zeros((B, 3, L), dtype=np.int64)
    diagonal_index = np.zeros((B, 3), dtype=np.int64)
    spin = np.zeros((B, 6))
    offsets = [0]

    def add(value, field, key):
        operands.append(value)
        slots.append((field, key))

    read_cache = {}
    def read(field, *indices):
        key = (field,)+tuple(tuple(map(int, index)) for index in indices)
        if reuse_reads and key in read_cache:
            return read_cache[key]
        value = getattr(state, field)[np.ix_(*indices)]
        if field in ('qij', 'qia'):
            value = value[:, 0]
        if reuse_reads:
            read_cache[key] = value
        return value

    for row, d in enumerate(layouts):
        t, labels = d['t'], d['labels']
        aux, paos, lm = state.t_ribfs[t], state.t_paos[t], d['lm']
        n, q, p, l = d['n'], d['q'], d['p'], d['l']
        ix = {pair: k for k, pair in enumerate(d['overlaps'])}
        overlaps = triplet_overlaps(state, t, d['overlaps'])
        add(state.X_tno[t], 'x', (row, slice(p), slice(n)))
        add(read('J2c', aux, aux), 'metric', (row, slice(q), slice(q)))
        add(read('qab', aux, paos, paos), 'raw', (row, slice(q), slice(p), slice(p)))
        add(state.e_tno[t], 'e', (row, slice(n)))
        for a, label in enumerate(labels):
            add(state.orb.F_lmo[label, label], 'f', (row, a))
            add(read('qij', aux, [label], lm), 'qo', (row, a, slice(q), slice(l)))
            add(read('qia', aux, [label], paos), 'qi', (row, a, slice(q), slice(p)))
            rank = state.T1[label].shape[0]
            add(state.T1[label], 't1', (row, a, slice(rank)))
            diagonal_index[row, a] = ix[state.pl.ij[label, label]]
            for j, other in enumerate(labels):
                pair_index[row, a, j] = ix[state.pl.ij[label, other]]
            for j, other in enumerate(lm):
                partner_index[row, a, j] = ix[state.pl.ij[label, other]]
        for pair in d['overlaps']:
            rank = state.X[pair].shape[1]
            add(overlaps[pair], 's', (row, ix[pair], slice(n), slice(rank)))
        for pair in d['projected']:
            rank = state.T[pair].shape[0]
            add(state.T[pair], 'ts', (row, ix[pair], slice(rank), slice(rank)))
        i, j, k = labels
        pref = 1/6 if i == j == k else .5 if len(set(labels)) == 2 else 1.
        for weight, axes in ((8., _AXES[0]), (-4., _axes((k, j, i))),
                             (-4., _axes((i, k, j))), (-4., _axes((j, i, k))),
                             (2., _axes((j, k, i))), (2., _axes((k, i, j)))):
            spin[row, _AXES.index(axes)] += pref*weight
        offsets.append(offsets[-1]+(1 if energy_only else 1+3*n**3))
    values = tuple(ad.primal(v) for v in operands)
    backends = tuple(ad._xp([v]) for v in operands)
    if any(np.dtype(v.dtype) != np.float64 for v in values):
        raise ValueError('Triples batch requires real float64 operands')
    output_xps = tuple(ad._xp([state.X_tno[d['t']]]) for d in layouts)
    packed_xp = np if all(xp is np for xp in output_xps) else cp
    workspace_key = object()
    field_offsets, size = {}, 0
    for field, shape in shapes.items():
        end = size + int(np.prod(shape))
        field_offsets[field] = (size, end)
        size = end

    def transpose(x, axes):
        return x.transpose((0,)+tuple(a+1 for a in axes))

    def rebuild():
        host = np.zeros(size)
        parts = {field: host[a:b].reshape(shapes[field]) for field, (a, b) in field_offsets.items()}
        parts['metric'][:] = np.eye(Q)
        parts['e'][:] = 1e5
        for value, xp, (field, key) in zip(values, backends, slots):
            if xp is np:
                parts[field][key] = value
        device = cp.asarray(host)
        a = {field: device[first:last].reshape(shapes[field]) for field, (first, last) in field_offsets.items()}
        for value, xp, (field, key) in zip(values, backends, slots):
            if xp is not np:
                a[field][key] = value
        rows = cp.arange(B)
        pi, li, di, weights = map(cp.asarray, (pair_index, partner_index, diagonal_index, spin))
        x, metric, raw = a['x'], a['metric'], a['raw']
        b = a['qi'] @ x[:, None]
        fitted = cp.linalg.solve(metric, b.transpose(0, 2, 1, 3).reshape(B, Q, 3*N))
        c = fitted.reshape(B, Q, 3, N).transpose(0, 2, 1, 3)
        h = (x.swapaxes(-1, -2)[:, None] @ raw) @ x[:, None]
        kv = (c.swapaxes(-1, -2) @ h.reshape(B, 1, Q, N*N)).reshape(B, 3, N*N, N)
        ko = a['qo'].swapaxes(-1, -2)[:, :, None] @ c[:, None]
        u = (a['s'] @ a['ts']) @ a['s'].swapaxes(-1, -2)
        partner = u[rows[:, None, None], li].reshape(B, 3, L, N*N).swapaxes(-1, -2)
        w = cp.zeros((B, N, N, N))
        for (i, j, k), axes in zip(_PERMS, _AXES):
            part = kv[:, i] @ u[rows, pi[:, k, j]].swapaxes(-1, -2)
            if L:
                part -= partner[:, i] @ ko[:, j, k]
            w += transpose(part.reshape(B, N, N, N), axes)
        sd = a['s'][rows[:, None], di]
        singles = (sd @ a['t1'][..., None])[..., 0]
        pairs = b[:, [1, 0, 0]].swapaxes(-1, -2) @ c[:, [2, 2, 1]]
        v = w + singles[:, 0, :, None, None]*pairs[:, 0, None]
        v += singles[:, 1, None, :, None]*pairs[:, 1, :, None, :]
        v += singles[:, 2, None, None, :]*pairs[:, 2, :, :, None]
        e = a['e']
        denom = e[:, :, None, None]+e[:, None, :, None]+e[:, None, None, :]-a['f'].sum(1)[:, None, None, None]
        t3 = -w/denom
        return a, rows, pi, li, di, weights, b, c, h, kv, ko, u, partner, sd, singles, pairs, w, v, denom, t3

    def forward():
        primals = rebuild()
        workspace = ad.checkpoint_replay_workspace()
        if workspace is not None:
            workspace[workspace_key] = primals
        weights = primals[5]
        w, v, denom, t3 = primals[-4:]
        energy = sum(weights[:, p]*cp.sum(transpose(v, axes)*t3, axis=(1, 2, 3)) for p, axes in enumerate(_AXES))
        blocks = []
        for row, d in enumerate(layouts):
            blocks.append(energy[row:row+1])
            if not energy_only:
                n = d['n']
                blocks.extend(value[row, :n, :n, :n].ravel() for value in (w, v, t3))
        packed = cp.concatenate(blocks)
        return cp.asnumpy(packed) if packed_xp is np else packed

    def pullback(seed, needed):
        workspace = ad.checkpoint_replay_workspace()
        primals = None if workspace is None else workspace.pop(workspace_key, None)
        if primals is None:
            primals = rebuild()
        a, rows, pi, li, di, weights, b, c, h, kv, ko, u, partner, sd, singles, pairs, w, v, denom, t3 = primals
        z = cp.asarray(seed)
        ze = z[cp.asarray(offsets[:-1])]
        bv, bt = cp.zeros_like(v), cp.zeros_like(t3)
        for p, axes in enumerate(_AXES):
            factor = (weights[:, p]*ze)[:, None, None, None]
            bv += factor*transpose(t3, tuple(np.argsort(axes)))
            bt += factor*transpose(v, axes)
        if not energy_only:
            for row, d in enumerate(layouts):
                n, start = d['n'], offsets[row]+1
                bv[row, :n, :n, :n] += z[start+n**3:start+2*n**3].reshape((n,)*3)
                bt[row, :n, :n, :n] += z[start+2*n**3:start+3*n**3].reshape((n,)*3)
        bw = bv-bt/denom
        if not energy_only:
            for row, d in enumerate(layouts):
                n, start = d['n'], offsets[row]+1
                bw[row, :n, :n, :n] += z[start:start+n**3].reshape((n,)*3)
        fields = {field for need, (field, key) in zip(needed, slots) if need}
        need_ri = bool(fields & {'x', 'metric', 'raw', 'qo', 'qi'})
        need_projection = bool(fields & {'s', 'ts'})
        grads = {}
        if fields & {'e', 'f'}:
            bd = bt*w/(denom*denom)
            grads['e'] = bd.sum((2, 3))+bd.sum((1, 3))+bd.sum((1, 2))
            grads['f'] = cp.broadcast_to(-bd.sum((1, 2, 3))[:, None], (B, 3))
        gu = cp.zeros_like(u) if need_projection else None
        gkv = cp.zeros_like(kv) if need_ri else None
        gko = cp.zeros_like(ko) if need_ri else None
        for (i, j, k), axes in zip(_PERMS, _AXES):
            zp = transpose(bw, tuple(np.argsort(axes))).reshape(B, N*N, N)
            if need_ri:
                gkv[:, i] += zp @ u[rows, pi[:, k, j]]
                if L:
                    gko[:, j, k] -= partner[:, i].swapaxes(-1, -2) @ zp
            if need_projection:
                gu[rows, pi[:, k, j]] += zp.swapaxes(-1, -2) @ kv[:, i]
                if L:
                    gp = (-zp @ ko[:, j, k].swapaxes(-1, -2)).swapaxes(-1, -2).reshape(B, L, N, N)
                    cp.add.at(gu, (rows[:, None], li[:, i]), gp)
        if need_projection:
            if 'ts' in fields:
                grads['ts'] = (a['s'].swapaxes(-1, -2) @ gu) @ a['s']
            if 's' in fields:
                grads['s'] = (gu @ a['s']) @ a['ts'].swapaxes(-1, -2)
                grads['s'] += (gu.swapaxes(-1, -2) @ a['s']) @ a['ts']
        single_bars = cp.stack((cp.sum(bv*pairs[:, 0, None], (2, 3)),
                               cp.sum(bv*pairs[:, 1, :, None, :], (1, 3)),
                               cp.sum(bv*pairs[:, 2, :, :, None], (1, 2))), axis=1)
        if 't1' in fields:
            grads['t1'] = (sd.swapaxes(-1, -2) @ single_bars[..., None])[..., 0]
        if 's' in fields:
            cp.add.at(grads['s'], (rows[:, None], di), single_bars[..., None]*a['t1'][:, :, None, :])
        if need_ri:
            g = gkv.reshape(B, 3, N, N*N)
            gc = h.reshape(B, 1, Q, N*N) @ g.swapaxes(-1, -2)
            gh = (c @ g).sum(axis=1).reshape(B, Q, N, N)
            go = (c[:, None] @ gko.swapaxes(-1, -2)).sum(axis=2)
            gc += (a['qo'][:, :, None] @ gko).sum(axis=1)
            gb = cp.zeros_like(b)
            pair_bars = (cp.sum(bv*singles[:, 0, :, None, None], axis=1),
                         cp.sum(bv*singles[:, 1, None, :, None], axis=2),
                         cp.sum(bv*singles[:, 2, None, None, :], axis=3))
            for (i, j), g in zip(((1, 2), (0, 2), (0, 1)), pair_bars):
                gb[:, i] += c[:, j] @ g.swapaxes(-1, -2)
                gc[:, j] += b[:, i] @ g
            rhs_bar = cp.linalg.solve(a['metric'].swapaxes(-1, -2), gc.transpose(0, 2, 1, 3).reshape(B, Q, 3*N))
            if 'metric' in fields:
                grads['metric'] = -rhs_bar @ c.transpose(0, 2, 1, 3).reshape(B, Q, 3*N).swapaxes(-1, -2)
            gb += rhs_bar.reshape(B, Q, 3, N).transpose(0, 2, 1, 3)
            if 'qo' in fields:
                grads['qo'] = go
            if 'qi' in fields:
                grads['qi'] = gb @ a['x'].swapaxes(-1, -2)[:, None]
            if 'raw' in fields:
                grads['raw'] = (a['x'][:, None] @ gh) @ a['x'].swapaxes(-1, -2)[:, None]
            if 'x' in fields:
                gx = (a['qi'].swapaxes(-1, -2) @ gb).sum(axis=1)
                gx += ((a['raw'] @ a['x'][:, None]) @ gh.swapaxes(-1, -2)).sum(axis=1)
                gx += ((a['raw'].swapaxes(-1, -2) @ a['x'][:, None]) @ gh).sum(axis=1)
                grads['x'] = gx
        bars = [grads[field][key] if need else None for need, (field, key) in zip(needed, slots)]
        host_indices = [i for i, (need, xp) in enumerate(zip(needed, backends)) if need and xp is np]
        if host_indices:
            lengths = [bars[i].size for i in host_indices]
            transferred = cp.asnumpy(cp.concatenate([bars[i].ravel() for i in host_indices]))
            start = 0
            for pos, length in zip(host_indices, lengths):
                bars[pos] = transferred[start:start+length].reshape(values[pos].shape)
                start += length
        return tuple(bars)

    packed = ad.operation(forward(), operands,
                          lambda z: pullback(z, [True]*len(operands)), pullback)
    outputs = []
    for row, d in enumerate(layouts):
        n, start, xp = d['n'], offsets[row], output_xps[row]
        pieces = [(start, start+1, ())]
        if not energy_only:
            pieces += [(start+1+i*n**3, start+1+(i+1)*n**3, (n,)*3) for i in range(3)]
        for first, last, shape in pieces:
            key = slice(first, last)
            value = packed.v[key].reshape(shape)
            if xp is not packed_xp:
                value = cp.asnumpy(value) if xp is np else cp.asarray(value)
            def output_bar(bar, key=key, xp=xp):
                if xp is not packed_xp:
                    bar = cp.asnumpy(bar) if packed_xp is np else cp.asarray(bar)
                return (ad.IndexedCotangent(packed.shape, key, bar.reshape(-1)),)
            outputs.append(ad.operation(value, [packed], output_bar))
    return tuple(outputs)
