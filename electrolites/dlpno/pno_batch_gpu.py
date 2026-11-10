"""Resident batched semicanonical MP2 and PNO coefficient graphs.

Equal auxiliary domains share one RI solve with concatenated right-hand
sides per batch. PAO canonical bases are differentiable and shared across
all MP2, PNO and TNO stages. Every coefficient, including the RI metric and
the selected orbital subspaces, participates in the reverse sweep.
"""
import os
import time
import numpy as np
import cupy as cp

from . import reverse as ad
from .parallel_local_gpu import Capture
from .tno_batch_gpu import Executor, Batch as TNOBuffer, batched_eigh, _transpose


def _prepare(state, kind, cache, items):
    from .adjoint import orthocanonicalize
    capture = Capture()
    rows, metrics, integrals = [], {}, {}
    for pair in items:
        i, j = state.pl.pairs[pair]
        row = dict(item=pair, diagonal=i == j)
        if kind == 'recompute':
            fields = dict(x=state.X[pair], e=state.e_pno[pair],
                          k=state.K[pair], t=state.T[pair])
            row['aux'] = ()
        else:
            paos, aux = state.pao[pair], state.ribfs[pair]
            domain, auxiliary = tuple(map(int, paos)), tuple(map(int, aux))
            key = (float(state.opt['S_CUT']), domain)
            if key not in cache:
                cache[key] = orthocanonicalize(state.orb.S_pao[np.ix_(paos, paos)],
                    state.orb.F_pao[np.ix_(paos, paos)], state.opt['S_CUT'])
            x, e = cache[key]
            if auxiliary not in metrics:
                metrics[auxiliary] = state.J2c[np.ix_(aux, aux)]
            for occupied in (i, j):
                token = (auxiliary, occupied, domain)
                if token not in integrals:
                    integrals[token] = state.qia[np.ix_(aux, [occupied], paos)][:, 0]
            fields = dict(x=x, e=e, m=metrics[auxiliary],
                qi=integrals[auxiliary, i, domain], qj=integrals[auxiliary, j, domain],
                f=state.orb.F_lmo[i, i] + state.orb.F_lmo[j, j])
            row['aux'] = auxiliary
        row.update({name: capture.value(value).index for name, value in fields.items()})
        row.update(p=fields['x'].shape[0], r=fields['x'].shape[1])
        scale = state.opt['T_CUT_PNO_DIAG_SCALE'] if i == j else 1.
        if i < state.ncore_scale or j < state.ncore_scale:
            scale *= state.opt['T_CUT_PNO_CORE_SCALE']
        row['scale'] = scale
        rows.append(row)
    return tuple(capture.inputs), rows


def _ranks(occupations, energy_products, total_energy, rows, options, initial):
    """Keep the reference's precise trace/energy selection update order."""
    suffix = '_MP2' if initial else ''
    ranks = []
    for occ, products, total, row in zip(occupations, energy_products, total_energy, rows):
        occ_total, occ_pno, e_pno = occ.sum(), 0., 0.
        selected = []
        for a in range(len(occ)):
            trace = occ_pno / occ_total if occ_total else np.nan
            if (abs(occ[a]) >= row['scale'] * options['T_CUT_PNO' + suffix]
                    or trace < options['T_CUT_TRACE' + suffix]
                    or abs(e_pno) < options['T_CUT_ENERGY' + suffix] * abs(total)
                    or a < options['MIN_PNOS']):
                if not initial:
                    selected.append(a)
                e_pno = products[np.ix_(selected, selected)].sum()
                occ_pno += occ[a]
                if initial:
                    selected.append(a)
        ranks.append(max(1, len(selected)))
    return np.asarray(ranks)


def evaluate_subspace(spec, a):
    """Shared batched equations with every field an explicit AD operand."""
    rows, kind = spec['rows'], spec['kind']
    b, p, r = len(rows), spec['p'], spec['r']
    if kind != 'recompute':na = len(rows[0]['aux'])
    if kind == 'recompute':
        k, t = a['k'], a['t']
    else:
        qi, qj = a['qi'] @ a['x'], a['qj'] @ a['x']
        # One factorization per shared auxiliary domain/batch. This is a
        # matrix solve, so the existing complete metric VJP is applicable.
        rhs = qi.transpose((1, 0, 2)).reshape(na, b*r)
        solved = ad.solve(a['m'], rhs).reshape(na, b, r).transpose((1, 0, 2))
        k = _transpose(solved) @ qj
        t = k / (a['f'][:, None, None] - a['e'][:, :, None] - a['e'][:, None, :])
    anti = 2*t - _transpose(t)
    energy = (k*anti).sum(axis=(1, 2))
    if kind == 'sc_mp2':
        outputs = [energy[j] for j in range(b)]
    else:
        density = anti @ _transpose(t) + _transpose(anti) @ t
        density *= cp.asarray([.5 if row['diagonal'] else 1. for row in rows])[:, None, None]
        occ, u, tolerance = batched_eigh(density)
        occ, u = occ[:, ::-1], u[:, :, ::-1]
        kn, tn = (_transpose(u) @ k) @ u, (_transpose(u) @ anti) @ u
        ranks = _ranks(cp.asnumpy(occ.v), cp.asnumpy((kn*tn).v), cp.asnumpy(energy.v),
                       rows, spec['options'], kind == 'pno')
        occupations, tol = cp.asnumpy(occ.v), cp.asnumpy(tolerance)
        for j, n in enumerate(ranks):
            if n < r and np.min(abs(occupations[j, :n, None] - occupations[j, None, n:])) <= tol[j]:
                raise ValueError('PNO/TNO selection splits a degenerate eigenvalue block')
        outputs = [None] * (5*b)
        for n in sorted(set(ranks)):
            selected_rows = np.where(ranks == n)[0]
            selected = u[selected_rows, :, :n]
            ep, rotation, _ = batched_eigh(_transpose(selected) @ (a['e'][selected_rows, :, None]*selected))
            transform = selected @ rotation
            x = a['x'][selected_rows] @ transform
            kp = (_transpose(transform) @ k[selected_rows]) @ transform
            tp = (_transpose(transform) @ t[selected_rows]) @ transform
            correction = energy[selected_rows] - (kp*(2*tp - _transpose(tp))).sum(axis=(1, 2))
            for j, row in enumerate(selected_rows):
                outputs[5*row:5*row+5] = [x[j], ep[j], kp[j], tp[j], correction[j]]
    return tuple(outputs)


class Batch(TNOBuffer):
    def build(self):
        spec, values = self.spec, self.values
        rows, kind = spec['rows'], spec['kind']
        b, p, r = len(rows), spec['p'], spec['r']
        shapes = dict(x=(b, p, r), e=(b, r))
        if kind == 'recompute':
            shapes.update(k=(b, r, r), t=(b, r, r))
        else:
            na = len(rows[0]['aux'])
            shapes.update(m=(na, na), qi=(b, na, p), qj=(b, na, p), f=(b,))
        field_offsets, length = {}, 0
        for field, shape in shapes.items():
            count = int(np.prod(shape))
            field_offsets[field] = length, length + count
            length += count
        host = np.zeros(length)
        views = {field: host[a:z].reshape(shapes[field])
                 for field, (a, z) in field_offsets.items()}
        self.slots = []
        for field in shapes:
            indices = [(slice(None), rows[0][field])] if field == 'm' else [
                (j, row[field]) for j, row in enumerate(rows)]
            for key, index in indices:
                views[field][key] = values[index]
                self.slots.append((field, key, index))
        self.input = ad.Variable(cp.asarray(host))
        a = {field: self.input[first:last].reshape(shapes[field])
             for field, (first, last) in field_offsets.items()}
        outputs = evaluate_subspace(spec, a)
        self.output = ad.concatenate([x.ravel() for x in outputs])
        self.shapes = tuple(x.shape for x in outputs)
        self.offsets = np.cumsum([0]+[x.size for x in outputs])
        self.graph = ad.Linearization([self.output], [self.input])
        self.field_offsets, self.shapes_in = field_offsets, shapes
        self.values = self.spec = None
        return (cp.asnumpy(self.output.v), self.shapes,
                dict(device=self.device, pairs=b, records=self.graph.statistics['records'],
                     graph_value_bytes=self.graph.statistics['value_bytes'], resident_bytes=self.pool.used_bytes()))


class PNOPlan:
    def __init__(self, state, kind, executor, *, cache=None, batch_size=16,
                 memory_bytes=2048 << 20, items=None):
        if kind not in ('sc_mp2', 'pno', 'recompute'):
            raise ValueError('Unknown PNO batch stage')
        if not 1 <= batch_size <= 64:
            raise ValueError('PNO batch size must be 1..64')
        start = time.perf_counter()
        self.executor, self.closed, self.batches = executor, False, []
        self.items = tuple(p for p, (i, j) in enumerate(state.pl.pairs) if i <= j) if items is None else tuple(items)
        self.inputs, rows = _prepare(state, kind, {} if cache is None else cache, self.items)
        self.shapes = tuple(x.shape for x in self.inputs)
        values = tuple(np.asarray(ad.primal(x)) for x in self.inputs)
        buckets, loads = {}, [0]*len(executor.devices)
        for row in rows:
            buckets.setdefault((row['aux'], row['p'], row['r']), []).append(row)
        for (aux, p, r), group in buckets.items():
            for first in range(0, len(group), batch_size):
                chunk = group[first:first+batch_size]
                rank = min(range(len(loads)), key=loads.__getitem__)
                loads[rank] += len(aux)**3 + len(chunk)*(r**3 + len(aux)*p*r)
                spec = dict(p=p, r=r, rows=chunk, options=dict(state.opt), kind=kind)
                self.batches.append(Batch(rank, executor.devices[rank], tuple(row['item'] for row in chunk), values, spec))
        try:
            built = executor.run([(batch, 'build', ()) for batch in self.batches])
            totals = {d: sum(info['resident_bytes'] for _, _, info in built if info['device'] == d)
                      for d in executor.devices}
            if max(totals.values(), default=0) > memory_bytes:
                raise ValueError('Resident PNO graphs exceed per-stage memory bound')
        except BaseException:
            self.release()
            raise
        self.pieces, self.output_shapes, self.output_map = [], [], []
        stride = 1 if kind == 'sc_mp2' else 5
        for batch, (value, shapes, _) in zip(self.batches, built):
            self.pieces.append(value)
            self.output_shapes.extend(shapes)
            self.output_map.extend((p, k) for p in batch.items for k in range(stride))
        self.offsets = np.cumsum([0]+[x.size for x in self.pieces])
        self.statistics = dict(kind=kind, devices=list(executor.devices), pairs=len(rows), batches=len(self.batches),
            prepare_forward_seconds=time.perf_counter()-start, resident_bytes_by_device=totals,
            batches_info=[x[2] for x in built], reverse_seconds=0., reverse_calls=0, released=False,
            unique_canonical_domains=len(cache) if cache is not None else None,
            memory_limit_per_stage_bytes=memory_bytes)

    def bridge(self):
        packed = np.concatenate(self.pieces)
        self.pieces = None
        def reverse(seed, needed):
            if self.closed:
                raise RuntimeError('Released PNO graph cannot be reused')
            start = time.perf_counter()
            results = self.executor.run([(b, 'reverse', (seed[a:z], needed))
                for b, a, z in zip(self.batches, self.offsets[:-1], self.offsets[1:])])
            bars = [np.zeros(shape) if flag else None for shape, flag in zip(self.shapes, needed)]
            for result in results:
                for index, value in result.items():
                    bars[index] += value
            self.statistics['reverse_seconds'] += time.perf_counter()-start
            self.statistics['reverse_calls'] += 1
            return tuple(bars)
        value = ad.operation(packed, self.inputs, lambda z: reverse(z, [True]*len(self.inputs)), reverse)
        result, start = {}, 0
        for key, shape in zip(self.output_map, self.output_shapes):
            end = start + int(np.prod(shape))
            result[key] = value[start:end].reshape(shape)
            start = end
        stride = 1 if self.statistics['kind'] == 'sc_mp2' else 5
        return {p: tuple(result[p, k] for k in range(stride)) for p in self.items}

    def release(self):
        if self.closed:
            return
        stats = self.executor.run([(b, 'release', ()) for b in self.batches])
        self.closed, self.inputs = True, ()
        if hasattr(self, 'statistics'):
            self.statistics.update(released=True, released_batches=stats)


def _stage(state, kind):
    if os.environ.get('DLPNO_RESIDENT_SUBSPACES','0')=='1':
        from .resident_subspaces_gpu import pno_stage
        return pno_stage(state,kind)
    if not hasattr(state, '_pno_batch_executor'):
        state._pno_batch_executor = Executor(tuple(int(s) for s in os.environ['DLPNO_PNO_BATCH_DEVICES'].split(',')))
        state._pno_batch_plans = []
    if not hasattr(state, '_tno_basis_cache'):
        state._tno_basis_cache = {}
    memory = int(os.environ.get('DLPNO_PNO_BATCH_MEMORY_MIB', '2048')) << 20
    if not (32 << 20) <= memory <= (8 << 30):
        raise ValueError('PNO memory limit must be 32..8192 MiB')
    plan = PNOPlan(state, kind, state._pno_batch_executor, cache=state._tno_basis_cache,
        batch_size=int(os.environ.get('DLPNO_PNO_BATCH_SIZE', '16')), memory_bytes=memory)
    state._pno_batch_plans.append(plan)
    state.diagnostics.setdefault('batched_pno', []).append(plan.statistics)
    return plan.bridge()


def sc_mp2_pair_energies(state):
    result = state.numeric.zeros(len(state.pl))
    for p, (value,) in _stage(state, 'sc_mp2').items():
        result[p] = value
        result[state.pl.ji[p]] = value
    return result


def lmp2_pnos(state):
    size = len(state.pl)
    for name in ('X', 'e_pno', 'K', 'T'):
        setattr(state, name, [None]*size)
    state.de_pno = state.numeric.zeros(size)
    for p, (x, e, k, t, correction) in _stage(state, 'pno').items():
        state.de_pno[p] = correction
        state._store_pair(p, x, e, k, t)
    state.de_pno_total = state.de_pno.sum()


def recompute_pnos(state):
    for p, (x, e, k, t, correction) in _stage(state, 'recompute').items():
        state.de_pno[p] += correction
        state._store_pair(p, x, e, k, t)
    state._S = {}
    state.de_pno_total = state.de_pno.sum()
    state.de_weak = sum((state.K[p]*(2*state.T[p]-state.T[p].T)).sum()
                        for p in range(len(state.pl)) if not state.strong[p])


def finish(state):
    for plan in getattr(state, '_pno_batch_plans', ()):
        plan.release()
    executor = getattr(state, '_pno_batch_executor', None)
    if executor is not None:
        executor.close()
    for key in ('_pno_batch_plans', '_pno_batch_executor', '_tno_basis_cache'):
        state.__dict__.pop(key, None)
