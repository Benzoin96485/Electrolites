"""Persistent multi-device DF graphs for complete CCSD source pullbacks.

DF coefficients and primal amplitude intermediates are uploaded once.
Each lambda action transfers only output cotangents and returns selected
amplitude cotangents. The final source sweep also differentiates every DF
coefficient. All communication stays between device arrays; no host staging
of numerical source tensors is needed at this boundary.
"""
import os
import time
from types import SimpleNamespace, MethodType
import numpy as np
import cupy as cp

from . import reverse as ad
from .ccsd_gpu import CCSDGPU, _mv, _dot2, _wsum
from .ccsd_adjoint_gpu import _ARRAY
from .ragged_adjoint import AdjointRagged
from .reference_cpu import rebind
from .tno_batch_gpu import Executor


_FIELDS = ('Qma', 'Qab', 'Qk', 'Qa', 'ev')


def _chunks(source, devices, batch_size):
    chunks = []
    for original in source.df:
        count = len(original['p'])
        # Keep the upstream coarse GEMM batches whenever there are already
        # enough independent groups. Splitting every group doubles Python
        # graph work without helping a launch-bound lambda action.
        capacity = count if len(source.df) >= len(devices) else max(1, (count+len(devices)-1)//len(devices))
        step = min(batch_size, capacity)
        for first in range(0, count, step):
            last = min(count, first+step)
            group = dict(original)
            for name in ('p', 'q', 'ri', 'rj', 'lm_idx', 'lm_msk', *_FIELDS):
                group[name] = original[name][first:last]
            p, q = group['p'], group['q']
            for name, layout, members in (('N', source.LN, p), ('Nq', source.LN, q),
                    ('P', source.LP, p), ('Pq', source.LP, q),
                    ('L', source.LL, p), ('Lq', source.LL, q)):
                group[name] = CCSDGPU._items(layout, members)
            diagonal = source.pi[p] == source.pj[p]
            group['dsel'] = cp.asarray(np.where(diagonal)[0])
            group['di'] = source.pi[p][diagonal]
            chunks.append(group)
    return chunks


def _copy(value):
    if isinstance(value, cp.ndarray):
        return cp.asarray(value)
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_copy(v) for v in value)
    return value


class Shard:
    def __init__(self, rank, device, groups, values, layouts, shared):
        self.rank, self.device = rank, device
        self.groups, self.values = groups, values
        self.layouts, self.shared = layouts, shared
        with cp.cuda.Device(device):
            self.pool = cp.cuda.MemoryPool()
            self.stream = cp.cuda.Stream(non_blocking=True)
        self.statistics = dict(device=device, groups=len(groups),
            pairs=sum(len(group['p']) for group in groups), reverse_calls=0,
            selected_reverse_calls=0, reverse_seconds=0.,
            group_shapes=[dict(pairs=len(g['p']), auxiliary=g['A'], lmos=g['Lb'], pnos=g['Vb']) for g in groups])

    def call(self, method, *args):
        with cp.cuda.Device(self.device), self.stream, cp.cuda.using_allocator(self.pool.malloc):
            result = getattr(self, method)(*args)
            self.stream.synchronize()
            return result

    def build(self):
        indices = list(self.shared[0])+list(self.shared[1])
        for group in self.groups:
            indices.extend(group[name] for name in _FIELDS)
        self.indices = tuple(dict.fromkeys(indices))
        self.leaves = [ad.Variable(cp.asarray(self.values[i])) for i in self.indices]
        mapping = dict(zip(self.indices, self.leaves))
        t, tn = [AdjointRagged(layout, [mapping[i] for i in keys])
                 for layout, keys in zip(self.layouts[:2], self.shared)]
        groups = []
        for original in self.groups:
            group = _copy({k: v for k, v in original.items() if k not in _FIELDS})
            group.update({name: mapping[original[name]] for name in _FIELDS})
            group['dev'] = self.device
            groups.append(group)
        namespace = dict(cp=_ARRAY, _mv=rebind(_mv, cp=_ARRAY),
                         _dot2=rebind(_dot2, cp=_ARRAY), _wsum=rebind(_wsum, cp=_ARRAY))
        view = SimpleNamespace(_gat=rebind(CCSDGPU._gat, cp=_ARRAY),
                               _put=rebind(CCSDGPU._put, cp=_ARRAY))
        outputs = [AdjointRagged(layout) for layout in self.layouts[2:]]
        if os.environ.get('DLPNO_CCSD_DF_CHECKPOINT_GRAPHS') == '1':
            from .ccsd_df_compact_gpu import compact_df_checkpoints
            view._df_groups = MethodType(rebind(CCSDGPU._df_groups, **namespace), view)
            view.df = groups
            view.adjoint_df_statistics = dict(groups=0, discarded_records=0,
                largest_inner_value_bytes=0, stored_output_bytes=0)
            diagonal = compact_df_checkpoints(view, t, tn, *outputs)
            self.statistics['df_checkpoints'] = view.adjoint_df_statistics
        else:
            diagonal = rebind(CCSDGPU._df_groups, **namespace)(view, groups, t, tn, *outputs)
        flattened = [array for output in outputs for array in output.data]
        for _, _, fai, fab, fkc, tnd in diagonal:
            flattened.extend((fai, fab, fkc, tnd))
        self.shapes = tuple(x.shape for x in flattened)
        self.output = ad.concatenate([x.ravel() for x in flattened])
        self.graphs = {}
        self.replays = {}
        self.statistics['cuda_graphs'] = []
        self.values = None
        self.statistics.update(resident_bytes=self.pool.used_bytes(),
            graph_value_bytes=ad.Linearization([self.output], self.leaves).statistics['value_bytes'])
        return self.output.v, self.shapes

    def reverse(self, seed, needed):
        start = time.perf_counter()
        selected = tuple(j for j, index in enumerate(self.indices) if needed[index])
        if selected not in self.graphs:
            self.graphs[selected] = ad.Linearization([self.output],
                [self.leaves[j] for j in selected], boundary=self.leaves)
        from .cuda_graph_replay import Replay, enabled
        amplitude_indices = set(self.shared[0]+self.shared[1])
        device_seed = cp.asarray(seed)
        if enabled() and selected and all(self.indices[j] in amplitude_indices for j in selected):
            if selected not in self.replays:
                replay = Replay(self.graphs[selected], [device_seed])
                self.replays[selected] = replay
                self.statistics['cuda_graphs'].append(replay.statistics)
            bars = self.replays[selected]([device_seed])
        else:
            bars = self.graphs[selected]([device_seed])
        self.statistics['reverse_calls'] += 1
        self.statistics['selected_reverse_calls'] += int(len(selected) != len(self.indices))
        self.statistics['reverse_seconds'] += time.perf_counter()-start
        return {self.indices[j]: bar for j, bar in zip(selected, bars)}

    def release(self):
        for replay in getattr(self, 'replays', {}).values():
            replay.release()
        self.replays = {}
        self.leaves = self.output = self.values = self.groups = None
        self.layouts = self.shared = None
        self.graphs = {}
        self.stream.synchronize()
        self.pool.free_all_blocks()
        self.statistics.update(released=True, live_bytes=self.pool.used_bytes())
        return self.statistics


class DFPlan:
    def __init__(self, source, t, tn, destinations, devices, *, batch_size=64, memory_bytes=2048 << 20):
        if not 1 <= batch_size <= 64:
            raise ValueError('CCSD DF batch size must be 1..64')
        self.home, self.closed = cp.cuda.Device().id, False
        self.executor = Executor(devices)
        self.shards, self.inputs = [], []
        start = time.perf_counter()
        identity = {}
        def capture(value):
            if not isinstance(ad.primal(value), cp.ndarray) or ad.primal(value).dtype != np.float64:
                raise ValueError('CCSD DF bridge requires FP64 device inputs')
            token = value._record if isinstance(value, ad.Variable) else id(value)
            if token not in identity:
                identity[token] = len(self.inputs)
                self.inputs.append(ad.Variable._from_record(value._record) if isinstance(value, ad.Variable) else value)
            return identity[token]
        shared = tuple(tuple(capture(x) for x in rag.data) for rag in (t, tn))
        self.layouts = tuple(rag.layout for rag in (t, tn, *destinations))
        groups, loads = [[] for _ in devices], [0]*len(devices)
        for group in _chunks(source, devices, batch_size):
            rank = min(range(len(devices)), key=loads.__getitem__)
            loads[rank] += len(group['p'])*group['A']*group['Vb']**2*(group['Vb']+group['Lb'])
            group.update({name: capture(group[name]) for name in _FIELDS})
            groups[rank].append(group)
        self.inputs = tuple(self.inputs)
        values = tuple(ad.primal(x) for x in self.inputs)
        # The worker streams consume slices and packing kernels produced on
        # the main stream. Make their readiness explicit before dispatch.
        cp.cuda.get_current_stream().synchronize()
        from .mgpu import enable_peer
        enable_peer(devices)
        for rank, chunk in enumerate(groups):
            if chunk:
                self.shards.append(Shard(rank, devices[rank], chunk, values, self.layouts, shared))
        try:
            self.built = self.executor.run([(s, 'build', ()) for s in self.shards])
            if any(s.statistics['resident_bytes'] > memory_bytes for s in self.shards):
                raise MemoryError('CCSD DF graph exceeds per-device memory bound')
        except BaseException:
            self.release()
            raise
        self.offsets = np.cumsum([0]+[x.size for x, _ in self.built])
        self.statistics = dict(devices=list(devices), shards=len(self.shards),
            groups=sum(len(g) for g in groups), batch_size=batch_size,
            forward_seconds=time.perf_counter()-start, reverse_seconds=0., reverse_calls=0,
            coefficient_derivatives_complete=True, host_staging=False,
            memory_limit_per_device_bytes=memory_bytes, workers=[s.statistics for s in self.shards], released=False)

    def bridge(self, destinations):
        packed = cp.concatenate([cp.asarray(value) for value, _ in self.built])
        shapes = [shape for _, group in self.built for shape in group]
        self.built = None
        def reverse(seed, needed):
            if self.closed:
                raise RuntimeError('Released CCSD DF graph cannot be reused')
            start = time.perf_counter()
            cp.cuda.get_current_stream().synchronize()
            results = self.executor.run([(s, 'reverse', (seed[a:z], needed))
                for s, a, z in zip(self.shards, self.offsets[:-1], self.offsets[1:])])
            bars = [cp.zeros_like(ad.primal(x)) if flag else None for x, flag in zip(self.inputs, needed)]
            for result in results:
                for i, contribution in result.items():
                    bars[i] += cp.asarray(contribution)
            # Keep peer-copy sources alive until the home reduction completes.
            cp.cuda.get_current_stream().synchronize()
            self.statistics['reverse_seconds'] += time.perf_counter()-start
            self.statistics['reverse_calls'] += 1
            return tuple(bars)
        flat = ad.operation(packed, self.inputs, lambda seed: reverse(seed, [True]*len(self.inputs)), reverse)
        outputs, first = [], 0
        for shape in shapes:
            last = first+int(np.prod(shape))
            outputs.append(flat[first:last].reshape(shape))
            first = last
        cursor, diagonal = 0, []
        for shard in self.shards:
            for destination in destinations:
                for bucket in range(destination.layout.ngroups):
                    destination.data[bucket] = destination.data[bucket]+outputs[cursor]
                    cursor += 1
            for group in shard.groups:
                if len(group['di']):
                    diagonal.append((group, group['dsel'], *outputs[cursor:cursor+4]))
                    cursor += 4
        if cursor != len(outputs):
            raise RuntimeError('CCSD DF output layout differs')
        return diagonal

    def release(self):
        if self.closed:
            return
        self.inputs = ()
        self.built = None
        self.layouts = None
        stats = self.executor.run([(s, 'release', ()) for s in self.shards])
        self.executor.close()
        self.closed = True
        if hasattr(self, 'statistics'):
            self.statistics.update(released=True, workers=stats)


def distributed_df(source, t, tn, fkc, fab, beta, rsym):
    devices = tuple(int(s) for s in os.environ['DLPNO_CCSD_DF_DEVICES'].split(','))
    memory = int(os.environ.get('DLPNO_CCSD_DF_MEMORY_MIB', '2048')) << 20
    if not (32 << 20) <= memory <= (32 << 30):
        raise ValueError('CCSD DF memory limit must be 32..32768 MiB')
    destinations = (fkc, fab, beta, rsym)
    plan = DFPlan(source, t, tn, destinations, devices,
        batch_size=int(os.environ.get('DLPNO_CCSD_DF_BATCH_SIZE', '64')), memory_bytes=memory)
    source._distributed_df_plan = plan
    source.statistics['distributed_df'] = plan.statistics
    return plan.bridge(destinations)
