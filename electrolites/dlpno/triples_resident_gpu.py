"""Resident batched (T1) residual, complete coefficient VJP and transpose.

The same bounded edge batches serve the residual and both kinds of reverse
actions. Device shards own independent pools/streams and retain coefficients
between Krylov actions. All orbital/metric dependencies remain explicit host
AD inputs; only the small-system opt-in path retains the complete plan.
"""
from concurrent.futures import ThreadPoolExecutor
import os
import time
import weakref

import numpy as np
import cupy as cp

from . import reverse as ad
from .gather import gather3p, scatter3p
from .ragged import padded
from .triples_overlap_adjoint_gpu import common_left_batch, overlap_batches


def configured_devices():
    value = os.environ.get('DLPNO_TRIPLES_RESIDENT_DEVICES', '')
    if not value:
        return ()
    devices = tuple(int(x) for x in value.split(','))
    if (not devices or len(set(devices)) != len(devices) or devices[0] != 0
            or min(devices) < 0 or max(devices) >= cp.cuda.runtime.getDeviceCount()):
        raise ValueError('Resident triples devices must be unique visible IDs starting with 0')
    return devices


def _flat(items):
    return (ad.concatenate([x.ravel() for x in items]) if items
            else ad.Variable(np.empty(0), tracked=False))


def _project(a, s):
    b, n, m = s.shape
    left = ad._matmul(s, a.reshape(b, m, m*m)).reshape(b, n, m, m)
    middle = ad._matmul(s[:, None], left)
    return ad._matmul(middle.reshape(b, n*n, m), s.transpose(0, 2, 1)).reshape(b, n, n, n)


def _project_vjp(a, s, z, need_a, need_s):
    b, n, m = s.shape
    gz = z.reshape(b, n*n, n)
    middle_bar = (gz @ s).reshape(b, n, n, m)
    left_bar = s.transpose(0, 2, 1)[:, None] @ middle_bar
    tensor_bar = ((s.transpose(0, 2, 1) @ left_bar.reshape(b, n, m*m))
                  .reshape(b, m, m, m) if need_a else None)
    overlap_bar = None
    if need_s:
        left = (s @ a.reshape(b, m, m*m)).reshape(b, n, m, m)
        middle = s[:, None] @ left
        overlap_bar = gz.transpose(0, 2, 1) @ middle.reshape(b, n*n, m)
        overlap_bar += (middle_bar.transpose(0, 2, 1, 3).reshape(b, n, n*m)
                        @ left.transpose(0, 1, 3, 2).reshape(b, n*m, m))
        overlap_bar += (left_bar.reshape(b, n, m*m)
                        @ a.reshape(b, m, m*m).transpose(0, 2, 1))
    return tensor_bar, overlap_bar


def _workspace(n, m):
    # Envelope for one edge, including simultaneous VJP GEMM temporaries.
    return 8*(4*n**3 + 4*m**3 + 6*n*m*m + 6*n*n*m + 16*n*m + 128)


def _gather(vector, batch, side):
    v = batch['n' if side == 'target' else 'm']
    return gather3p(vector.data.ptr+batch[side+'_offsets']*8,
                   batch[side+'_strides'], batch[side+'_sizes3'], v, v, v)


class _Shard:
    def __init__(self, device, specs, arrays, diagonal, sizes, overlap_inputs=None):
        self.device = device
        self.pool = cp.cuda.MemoryPool()
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.specs = specs
        self.arrays = arrays
        self.host_diagonal = diagonal
        self.sizes = sizes
        self.transpose_replay = None
        self.transpose_replay_statistics = None
        self.overlap_inputs = overlap_inputs
        self.overlap_worker = None

    def call(self, name, *args):
        with cp.cuda.Device(self.device), self.stream, cp.cuda.using_allocator(self.pool.malloc):
            result = getattr(self, name)(*args)
            self.stream.synchronize()
            return result

    def initialize(self):
        # W, T, e, occupied sums, edge Fock weights, overlap coefficients.
        self.t = cp.asarray(self.arrays[1])
        self.batches = []
        if self.overlap_inputs is None:
            flat = np.append(self.arrays[5], 0.)
        else:
            from .resident_overlap_inputs_gpu import OverlapWorker
            wanted = {int(k) for spec in self.specs for k in spec['overlap_ids']}
            self.overlap_worker = OverlapWorker(self.overlap_inputs, wanted)
            flat = self.overlap_worker.flat
        for spec in self.specs:
            batch = {key: cp.asarray(value, order='C') if isinstance(value, np.ndarray) else value
                     for key, value in spec.items()}
            index = spec['s_indices']
            batch['s'] = cp.asarray(flat[cp.asarray(index)] if isinstance(flat, cp.ndarray) else flat[index])
            batch['f'] = cp.asarray(self.arrays[4][spec['edge_ids']])
            self.batches.append(batch)
        if self.device == 0:
            self.w = cp.asarray(self.arrays[0])
            self.diagonal = cp.asarray(self.host_diagonal)
        self.arrays = self.specs = self.host_diagonal = self.overlap_inputs = None
        if self.overlap_worker is not None:
            self.overlap_worker.flat = None
        del flat  # Exclude the temporary packed staging buffer from resident statistics.
        return dict(device=self.device, batches=len(self.batches),
                    resident_bytes=self.pool.used_bytes(),
                    overlap_sources=None if self.overlap_worker is None else self.overlap_worker.statistics)

    def forward(self):
        result = (self.w+self.diagonal*self.t if self.device == 0
                  else cp.zeros_like(self.t))
        for batch in self.batches:
            a = _gather(self.t, batch, 'source')
            term = _project(a, batch['s'])
            scatter3p(term, batch['target_offsets'], batch['target_sizes'], -batch['f'], result)
        return result

    def transpose(self, seed):
        from .cuda_graph_replay import Replay, enabled
        seed = cp.asarray(seed)
        if enabled():
            if self.transpose_replay is None:
                self.transpose_replay = Replay(lambda xs: [self._transpose(xs[0])], [seed])
            return self.transpose_replay([seed])[0]
        return self._transpose(seed)

    def _transpose(self, seed):
        seed = cp.asarray(seed)
        result = self.diagonal*seed if self.device == 0 else cp.zeros_like(seed)
        for batch in self.batches:
            z = _gather(seed, batch, 'target')
            term = _project(z, batch['s'].transpose(0, 2, 1))
            term = term.transpose((0,)+tuple(i+1 for i in batch['inverse']))
            scatter3p(term, batch['source_offsets'], batch['source_sizes'], -batch['f'], result)
        return result

    def reverse(self, seed, needed):
        if self.overlap_worker is None:
            return self._reverse(seed, needed)
        selected = list(needed[:5])+[any(needed[5:])]
        result = self._reverse(seed, selected)
        sources = (self.overlap_worker.reverse(result[5], needed[5:]) if selected[5]
                   else [None]*len(needed[5:]))
        return result[:5]+sources

    def _reverse(self, seed, needed):
        seed = cp.asarray(seed)
        result = [None]*6
        if needed[0] and self.device == 0:
            result[0] = seed.copy()
        if needed[1]:
            result[1] = self.diagonal*seed if self.device == 0 else cp.zeros_like(seed)
        if self.device == 0 and (needed[2] or needed[3]):
            result[2] = cp.zeros(self.sizes[2]) if needed[2] else None
            result[3] = cp.zeros(self.sizes[3]) if needed[3] else None
            for batch in self.diagonal_batches:
                weighted = _gather(seed, batch, 'target')*_gather(self.t, batch, 'target')
                if needed[2]:
                    bar = weighted.sum((2, 3))+weighted.sum((1, 3))+weighted.sum((1, 2))
                    cp.add.at(result[2], batch['e_indices'], bar[batch['e_mask']])
                if needed[3]:
                    result[3][batch['trip_ids']] = -weighted.sum((1, 2, 3))
        if needed[4]:
            result[4] = cp.zeros(self.sizes[4])
        if needed[5]:
            # One sentinel collects zero-padding bars and is discarded below.
            result[5] = cp.zeros(self.sizes[5]+1)
        for batch in self.batches:
            if not (needed[1] or needed[4] or needed[5]):
                break
            z = _gather(seed, batch, 'target')
            a = _gather(self.t, batch, 'source') if needed[4] or needed[5] else None
            if needed[4]:
                result[4][batch['edge_ids']] = -(z*_project(a, batch['s'])).sum((1, 2, 3))
            if needed[1] or needed[5]:
                ga, gs = _project_vjp(a, batch['s'], -batch['f'][:, None, None, None]*z,
                                      needed[1], needed[5])
                if needed[1]:
                    ga = ga.transpose((0,)+tuple(i+1 for i in batch['inverse']))
                    scatter3p(ga, batch['source_offsets'], batch['source_sizes'], batch['ones'], result[1])
                if needed[5]:
                    cp.add.at(result[5], batch['s_indices'].ravel(), gs.ravel())
        if needed[5]:
            result[5] = result[5][:-1].copy()
        return result

    def release_transpose(self):
        """Drop only capture buffers; coefficients remain for the final VJP."""
        if self.transpose_replay is not None:
            self.transpose_replay.release()
            self.transpose_replay_statistics = self.transpose_replay.statistics
            self.transpose_replay = None
        return self.transpose_replay_statistics

    def release(self):
        replay_stats = self.release_transpose()
        if self.overlap_worker is not None:
            self.overlap_worker.release()
            self.overlap_worker = None
        self.overlap_inputs = self.arrays = self.specs = self.host_diagonal = None
        for name in ('batches', 'diagonal_batches', 't', 'w', 'diagonal'):
            self.__dict__.pop(name, None)
        self.stream.synchronize()
        self.pool.free_all_blocks()
        return dict(device=self.device, live_bytes=self.pool.used_bytes(), cuda_graph=replay_stats)


def batched_fock_sources(fock, triplets, connections, cut):
    """Exact active-edge weights with one indexed AD edge per packed field.

    Screening observes only the requested Fock entries. Equal-to-cut entries
    are retained, and repeated or asymmetric indices are not deduplicated.
    Their cotangents are accumulated by the existing indexed AD pullback.
    Metadata and staging storage are linear in the local coupling count.
    """
    indices = np.asarray([(occ, old) for rows in connections
                          for _, _, occ, old in rows], dtype=np.int64).reshape(-1, 2)
    occupied = np.asarray(triplets, dtype=np.int64).reshape(-1, 3)
    # The real block gradient exposes SparseMatrix, not a dense AD tensor.
    # Read each requested primitive once, then gather repeated occurrences
    # from this short explicit boundary. Never materialize the full Fock.
    identity = {}
    def index(pair):
        pair = tuple(map(int,pair))
        if pair not in identity:identity[pair] = len(identity)
        return identity[pair]
    edge_ids = np.asarray([index(pair) for pair in indices], dtype=np.int64)
    diagonal_ids = np.asarray([index((i,i)) for i in occupied.ravel()],dtype=np.int64).reshape(occupied.shape)
    requested = np.asarray(list(identity),dtype=np.int64).reshape(-1,2)
    sources = fock[requested[:,0],requested[:,1]]
    xp = ad._xp([sources])
    values = ad.primal(sources)[xp.asarray(edge_ids)]
    values = cp.asnumpy(values) if xp is cp else values
    keep = np.abs(values) >= cut
    active = []
    first = 0
    for rows in connections:
        active.append([(key, u, None) for row, (key, u, _, _) in enumerate(rows)
                       if keep[first + row]])
        first += len(rows)
    edge_values = sources[xp.asarray(edge_ids[keep])]
    entries = sources[xp.asarray(diagonal_ids)]
    # Preserve the scalar oracle's left-to-right addition order.
    occupied_values = (entries[:, 0] + entries[:, 1]) + entries[:, 2]
    return active, occupied_values, edge_values


def batched_overlap_indices(ranks, offsets, edge_rows, selected, n, m):
    """Same padded overlap gather map, using bounded per-batch arrays."""
    rows = np.asarray([(edge_rows[e][0], edge_rows[e][1], edge_rows[e][2])
                       for e in selected], dtype=np.int64).reshape(-1, 3)
    nr, mr = ranks[rows[:, 0]], ranks[rows[:, 1]]
    i, j = np.arange(n)[None, :, None], np.arange(m)[None, None, :]
    value = offsets[rows[:, 2], None, None] + i * mr[:, None, None] + j
    valid = (i < nr[:, None, None]) & (j < mr[:, None, None])
    return np.where(valid, value, int(offsets[-1]))


class ResidentTriples:
    def __init__(self, state, devices, *, workspace_bytes=1024 << 20, memory_bytes=2048 << 20,
                 connections=None):
        start = time.perf_counter()
        self.devices = tuple(devices)
        if (not self.devices or self.devices[0] != 0 or len(set(self.devices)) != len(self.devices)
                or min(self.devices) < 0 or max(self.devices) >= cp.cuda.runtime.getDeviceCount()):
            raise ValueError('Invalid resident triples devices')
        if getattr(state, 'xp', np) is not np:
            raise ValueError('Resident triples require the host outer AD graph')
        if workspace_bytes <= 0 or memory_bytes <= 0:
            raise ValueError('Positive resident memory and workspace limits required')
        from .mgpu import enable_peer
        enable_peer(self.devices)
        self.released = False
        self.executor = ThreadPoolExecutor(max_workers=len(devices)) if len(devices) > 1 else None
        self.owner = weakref.ref(state) if hasattr(state, '__weakref__') else lambda: None
        self.ranks = np.asarray([x.shape[0] for x in state.T3], dtype=np.int64)
        self.offsets = np.cumsum([0]+[x.size for x in state.T3])
        self.e_offsets = np.cumsum(np.r_[0, self.ranks])
        overlaps, edge_rows, overlap_values, f_values = {}, [], [], []
        overlap_inputs = None
        shared_overlap_inputs = os.environ.get('DLPNO_TRIPLES_SHARED_OVERLAPS', '0') == '1'
        if shared_overlap_inputs or os.environ.get('DLPNO_TRIPLES_RESIDENT_OVERLAPS', '0') == '1':
            from .resident_overlap_inputs_gpu import OverlapInputs
            overlap_inputs = OverlapInputs()
        f = state.orb.F_lmo
        if connections is None:
            connections = state._triples_couplings()
        batched_metadata = os.environ.get('DLPNO_TRIPLES_BATCHED_METADATA', '0') == '1'
        if batched_metadata:
            active_rows, occupied_fock, edge_fock = batched_fock_sources(
                f, state.trip, connections, state.opt['F_CUT_T'])
        permutations = {}
        for t, rows in enumerate(connections):
            if batched_metadata:
                active = active_rows[t]
            else:
                active = []
                for key, u, occ, old in rows:
                    weight = f[occ, old]
                    if abs(float(ad.primal(weight))) >= state.opt['F_CUT_T']:
                        active.append((key, u, weight))
            for group, union, _ in overlap_batches((u for _, u, _ in active), state.X_tno[t],
                                                  state.t_paos, state.X_tno):
                coefficients = tuple(state.X_tno[u] for u in group)
                positions = tuple(np.searchsorted(union, state.t_paos[u]) for u in group)
                if overlap_inputs is None:
                    metric = state.orb.S_pao[np.ix_(state.t_paos[t], union)]
                    values = common_left_batch(metric, state.X_tno[t], coefficients, positions)
                else:
                    from .deferred_reads import capture_reads
                    with capture_reads(True):
                        metric = state.orb.S_pao[np.ix_(state.t_paos[t], union)]
                        values = overlap_inputs.add(metric, state.X_tno[t], coefficients, positions)
                for u, value in zip(group, values):
                    overlaps[t, u] = len(overlap_values)
                    overlap_values.append(value)
            for key, u, weight in active:
                if batched_metadata:
                    if key not in permutations:
                        inverse = tuple(np.argsort(key, kind='stable'))
                        permutations[key] = (inverse, tuple(np.argsort(inverse, kind='stable')))
                    inverse, permutation = permutations[key]
                else:
                    inverse = tuple(np.argsort(key, kind='stable'))
                    permutation = tuple(np.argsort(inverse, kind='stable'))
                edge_rows.append((t, u, overlaps[t, u], inverse, permutation))
                if not batched_metadata:
                    f_values.append(weight)
        if not batched_metadata:
            occ = [f[i, i]+f[j, j]+f[k, k] for i, j, k in state.trip]
            occupied_fock = _flat([x.reshape(1) for x in occ])
            edge_fock = _flat([x.reshape(1) for x in f_values])
        inputs = [_flat(state.W3), _flat(state.T3), _flat(state.e_tno),
                  occupied_fock, edge_fock]
        if overlap_inputs is None:
            inputs.append(_flat(overlap_values))
        else:
            overlap_inputs.freeze(materialize=not shared_overlap_inputs)
            if shared_overlap_inputs:
                from .resident_overlap_inputs_gpu import shared_overlaps
                inputs.append(shared_overlaps(state,overlap_inputs))
            else:
                inputs.extend(overlap_inputs.inputs)
        self.inputs = tuple(ad.Variable._from_record(x._record) if isinstance(x, ad.Variable) else x for x in inputs)
        self.backends = tuple(cp if isinstance(ad.primal(x), cp.ndarray) else np for x in inputs)
        arrays = [cp.asnumpy(ad.primal(x)) if xp is cp else np.asarray(ad.primal(x))
                  for x, xp in zip(inputs if overlap_inputs is None or shared_overlap_inputs else inputs[:5], self.backends)]
        if any(x.dtype != np.float64 for x in arrays):
            raise ValueError('Resident triples require real float64 inputs')
        if overlap_inputs is None:
            s_offsets = np.cumsum([0]+[x.size for x in overlap_values])
        else:
            s_offsets = overlap_inputs.offsets
            if not shared_overlap_inputs:
                arrays = arrays[:5]+[None]
        diagonal = np.concatenate([(arrays[2][a:b, None, None]+arrays[2][None, a:b, None]
                                   +arrays[2][None, None, a:b]-arrays[3][t]).ravel()
                                  for t, (a, b) in enumerate(zip(self.e_offsets[:-1], self.e_offsets[1:]))])
        padding = padded(self.ranks)
        groups = {}
        for edge, (t, u, s, inverse, permutation) in enumerate(edge_rows):
            groups.setdefault((int(padding[t]), int(padding[u]), inverse), []).append(edge)
        batches = []
        for (n, m, inverse), edges in groups.items():
            bound = _workspace(n, m)
            if bound > workspace_bytes:
                raise ValueError('A resident triples edge exceeds the workspace bound')
            step = min(65535, max(1, workspace_bytes//bound))
            for begin in range(0, len(edges), step):
                selected = edges[begin:begin+step]
                target, source = np.array([(edge_rows[e][0], edge_rows[e][1]) for e in selected]).T
                nr, mr = self.ranks[target], self.ranks[source]
                if batched_metadata:
                    s_indices = batched_overlap_indices(self.ranks, s_offsets, edge_rows, selected, n, m)
                else:
                    s_indices = np.full((len(selected), n, m), int(s_offsets[-1]), dtype=np.int64)
                    for row, e in enumerate(selected):
                        t, u, s, _, _ = edge_rows[e]
                        s_indices[row, :self.ranks[t], :self.ranks[u]] = (
                            s_offsets[s]+np.arange(self.ranks[t])[:, None]*self.ranks[u]+np.arange(self.ranks[u]))
                perm = edge_rows[selected[0]][4]
                batches.append(dict(n=n, m=m, inverse=inverse, edge_ids=np.array(selected, dtype=np.int64),
                    target_offsets=self.offsets[target], source_offsets=self.offsets[source],
                    target_strides=np.stack([nr**2, nr, np.ones_like(nr)], axis=1).astype(np.int64),
                    source_strides=np.stack([mr**2, mr, np.ones_like(mr)], axis=1)[:, perm].astype(np.int64),
                    target_sizes3=np.repeat(nr[:, None], 3, axis=1).astype(np.int32),
                    source_sizes3=np.repeat(mr[:, None], 3, axis=1).astype(np.int32),
                    target_sizes=nr.astype(np.int32), source_sizes=mr.astype(np.int32),
                    s_indices=s_indices, overlap_ids=np.asarray([edge_rows[e][2] for e in selected], dtype=np.int64),
                    ones=np.ones(len(selected)), workspace_bound=bound*len(selected)))
        # Balance estimated arithmetic without creating per-edge Python tasks.
        bins, loads = [[] for _ in devices], [0]*len(devices)
        costs = {}
        for batch in batches:
            costs.setdefault(batch['workspace_bound'], []).append(batch)
        # With bounded local dimensions/counts only the distinct keys are
        # sorted; avoid an N log N sort over all local edge batches.
        for cost in sorted(costs, reverse=True):
            for batch in costs[cost]:
                rank = min(range(len(devices)), key=lambda i: loads[i])
                bins[rank].append(batch)
                loads[rank] += cost
        estimated = []
        for rank, specs in enumerate(bins):
            count = arrays[1].nbytes + (arrays[0].nbytes+diagonal.nbytes if rank == 0 else 0)
            count += sum(sum(v.nbytes for v in spec.values() if isinstance(v, np.ndarray))
                         +spec['s_indices'].size*8+len(spec['edge_ids'])*8 for spec in specs)
            if overlap_inputs is not None and not shared_overlap_inputs:
                count += overlap_inputs.input_bytes+8*int(s_offsets[-1])
            estimated.append(count)
        if max(estimated, default=0) > memory_bytes:
            self.close_executor()
            raise ValueError('Resident triples coefficients exceed the per-device memory bound')
        self.shards = []
        try:
            for device, specs in zip(devices, bins):
                with cp.cuda.Device(device):
                    sizes = [x.size for x in arrays[:5]]+[int(s_offsets[-1])]
                    self.shards.append(_Shard(device, specs, arrays, diagonal, sizes, None if shared_overlap_inputs else overlap_inputs))
            cp.cuda.get_current_stream().synchronize()
            initialized = self._run('initialize')
            self._diagonal_batches(padding)
            live = []
            for shard in self.shards:
                # CuPy pools report the current device's sub-pool.
                with cp.cuda.Device(shard.device):
                    live.append(shard.pool.used_bytes())
            if max(live, default=0) > memory_bytes:
                raise ValueError('Resident triples allocation exceeds its memory bound')
        except BaseException:
            self.release()
            raise
        self.statistics = dict(operator_kind='resident_batched_triples', devices=list(devices),
            couplings=len(edge_rows), batches=len(batches), batched_metadata=batched_metadata,
            input_bytes=sum(x.nbytes for x in arrays if x is not None)+(0 if overlap_inputs is None or shared_overlap_inputs else overlap_inputs.input_bytes),
            resident_overlap_sources=overlap_inputs is not None, shared_overlap_sources=shared_overlap_inputs,
            overlap_boundary=None if overlap_inputs is None else overlap_inputs.statistics,
            resident_bytes_by_device=live, memory_bound_bytes=memory_bytes,
            workspace_bound_bytes=max((b['workspace_bound'] for b in batches), default=0),
            prepare_seconds=time.perf_counter()-start, initialized=initialized,
            forward_seconds=0., reverse_seconds=0., transpose_seconds=0.,
            transpose_calls=0, reverse_calls=0, released=False, complete_coefficient_vjp=True)

    def _diagonal_batches(self, padding):
        shard = self.shards[0]
        with cp.cuda.Device(0), shard.stream, cp.cuda.using_allocator(shard.pool.malloc):
            shard.diagonal_batches = []
            for n in sorted(set(padding)):
                ids = np.where(padding == n)[0]
                nr = self.ranks[ids]
                mask = np.arange(n)[None, :] < nr[:, None]
                indices = self.e_offsets[ids, None]+np.arange(n)[None, :]
                shard.diagonal_batches.append(dict(n=int(n), target_offsets=cp.asarray(self.offsets[ids]),
                    target_strides=cp.asarray(np.stack([nr**2, nr, np.ones_like(nr)], axis=1)),
                    target_sizes3=cp.asarray(np.repeat(nr[:, None], 3, axis=1).astype(np.int32)),
                    e_mask=cp.asarray(mask), e_indices=cp.asarray(indices[mask]), trip_ids=cp.asarray(ids)))
            shard.stream.synchronize()

    def _run(self, name, *args):
        if self.executor is None:
            return [self.shards[0].call(name, *args)]
        futures = [self.executor.submit(shard.call, name, *args) for shard in self.shards]
        results, failures = [], []
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise failures[0]
        return results

    def _sum(self, parts, *, device=0):
        if os.environ.get('DLPNO_TRIPLES_STREAM_REDUCTION','0')!='1':
            return self._sum_retained(parts, device=device)
        # Every shard has already synchronized its producer stream in _run.
        # A peer buffer is reused in the primary stream's addition order.
        # Synchronizing each peer copy also protects temporary contiguous
        # source arrays until their transfer has finished.
        with cp.cuda.Device(device):
            result = staging = None
            stream = cp.cuda.get_current_stream()
            for part in parts:
                if part is None:
                    continue
                if part.device.id != device:
                    source_device = part.device.id
                    if not part.flags.c_contiguous:
                        with cp.cuda.Device(source_device):
                            part = cp.ascontiguousarray(part)
                            cp.cuda.get_current_stream().synchronize()
                    if staging is None:
                        staging = cp.empty(part.shape, dtype=part.dtype)
                    elif staging.shape != part.shape or staging.dtype != part.dtype:
                        raise ValueError('Peer reduction inputs must have identical shape and dtype')
                    cp.cuda.runtime.memcpyPeerAsync(staging.data.ptr, device,
                        part.data.ptr, source_device, part.nbytes, stream.ptr)
                    stream.synchronize()
                    part = staging
                if result is None:
                    result = cp.array(part, copy=True)
                else:
                    result += part
            stream.synchronize()
            return result

    def _sum_retained(self, parts, *, device=0):
        with cp.cuda.Device(device):
            result = None
            transfers = []
            for part in parts:
                if part is not None:
                    if part.device.id != device:
                        source_device = part.device.id
                        if not part.flags.c_contiguous:
                            with cp.cuda.Device(source_device):
                                part = cp.ascontiguousarray(part)
                                cp.cuda.get_current_stream().synchronize()
                        copied = cp.empty(part.shape, dtype=part.dtype)
                        stream = cp.cuda.get_current_stream()
                        cp.cuda.runtime.memcpyPeerAsync(copied.data.ptr, device,
                            part.data.ptr, source_device, part.nbytes, stream.ptr)
                        # Make the cross-device transfer and the lifetime of
                        # both buffers explicit before launching the reduction.
                        stream.synchronize()
                        transfers.append((part, copied))
                        part = copied
                    if result is None:
                        result = cp.array(part, copy=True)
                    else:
                        result += part
            cp.cuda.get_current_stream().synchronize()
            return result

    @property
    def diagonal(self):
        return self.shards[0].diagonal

    def transpose(self, seed, *, result_device=0):
        if self.released:
            raise RuntimeError('A released triples plan cannot be reused')
        if result_device not in self.devices:
            raise ValueError('T1 transpose result device must be a resident shard')
        start = time.perf_counter()
        cp.cuda.get_current_stream().synchronize()
        result = self._sum(self._run('transpose', seed), device=result_device)
        self.statistics['transpose_seconds'] += time.perf_counter()-start
        self.statistics['transpose_calls'] += 1
        return result

    def bridge(self):
        start = time.perf_counter()
        packed = cp.asnumpy(self._sum(self._run('forward')))
        self.statistics['forward_seconds'] += time.perf_counter()-start

        def pullback(seed, needed):
            if self.released:
                raise RuntimeError('A released triples plan cannot be reused')
            start = time.perf_counter()
            with cp.cuda.Device(0):
                device_seed = cp.asarray(seed)
                cp.cuda.get_current_stream().synchronize()
            results = self._run('reverse', device_seed, needed)
            bars = []
            for i, flag in enumerate(needed):
                value = self._sum([r[i] for r in results]) if flag else None
                bars.append(cp.asnumpy(value) if flag and self.backends[i] is np else value)
            bars = tuple(bars)
            del results, device_seed
            self.statistics['reverse_seconds'] += time.perf_counter()-start
            self.statistics['reverse_calls'] += 1
            owner = self.owner()
            if owner is not None and getattr(owner, '_release_transposes', False):
                self.release()
            return bars

        value = ad.operation(packed, self.inputs, lambda z: pullback(z, [True]*len(self.inputs)), pullback)
        return [value[a:b].reshape((int(n),)*3)
                for a, b, n in zip(self.offsets[:-1], self.offsets[1:], self.ranks)]

    def close_executor(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def release(self):
        if self.released:
            return
        released = self._run('release')
        self.close_executor()
        self.released = True
        self.inputs = ()
        if hasattr(self, 'statistics'):
            self.statistics.update(released=True, released_devices=released)


class ResidentTranspose:
    xp = cp

    def __init__(self, plan):
        self.plan = plan
        self.solve_device = int(os.environ.get('DLPNO_TRIPLES_KRYLOV_DEVICE', '0'))
        if self.solve_device not in plan.devices:
            raise ValueError('T1 Krylov device must be a resident shard')
        self.diagonal = plan.diagonal
        replica_bytes = 0
        if self.solve_device != 0:
            with cp.cuda.Device(self.solve_device):
                self.diagonal = cp.array(plan.diagonal, copy=True)
                cp.cuda.get_current_stream().synchronize()
            replica_bytes = self.diagonal.nbytes
        self.statistics = dict(plan.statistics, records=0, backend='cupy',
                               operator_kind='resident_batched_triples_transpose',
                               krylov_device=self.solve_device,
                               krylov_diagonal_replica_bytes=replica_bytes)
        self.last_statistics = {}

    def __call__(self, seeds):
        result = self.plan.transpose(seeds[0], result_device=self.solve_device)
        self.last_statistics = dict(calls=self.plan.statistics['transpose_calls'],
                                    seconds=self.plan.statistics['transpose_seconds'])
        return [result]

    def release(self):
        # The coefficient VJP still needs the plan after the lambda solve,
        # but does not use its CUDA graph's seed/output/temporary buffers.
        if (self.plan is not None
                and os.environ.get('DLPNO_TRIPLES_EARLY_RELEASE', '0') == '1'):
            records = self.plan._run('release_transpose')
            self.plan.statistics['transpose_capture_released_early'] = True
            self.plan.statistics['released_transpose_captures'] = records
        self.plan = self.diagonal = None
        if self.solve_device != 0:
            # Accepted lambda has already returned to the host density graph.
            # Do not strand large freed Krylov vectors in the peer default
            # pool while the coefficient VJP allocates from shard pools.
            with cp.cuda.Device(self.solve_device):
                cp.cuda.get_current_stream().synchronize()
                cp.get_default_memory_pool().free_all_blocks()


def residual(state):
    memory = int(os.environ.get('DLPNO_TRIPLES_RESIDENT_MEMORY_MIB', '2048')) << 20
    workspace = int(os.environ.get('DLPNO_TRIPLES_RESIDENT_WORKSPACE_MIB', '1024')) << 20
    if not (32 << 20) <= memory <= (32 << 30) or not (16 << 20) <= workspace <= (4 << 30):
        raise ValueError('Resident triples limits: memory 32..32768 MiB, workspace 16..4096 MiB')
    plan = ResidentTriples(state, configured_devices(), memory_bytes=memory, workspace_bytes=workspace)
    state._resident_triples_plan = plan
    state.diagnostics['resident_triples'] = plan.statistics
    try:
        return plan.bridge()
    except BaseException:
        plan.release()
        raise
