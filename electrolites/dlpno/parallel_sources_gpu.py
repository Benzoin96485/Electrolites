"""Bounded multi-GPU local source tasks with exact shared-input reduction.

Only host-boundary triples sources are distributed. Every worker rebuilds
its own local reverse graph. The parent owns the global graph and adds
cotangents of aliased/shared operands in deterministic task order. CCSD
and triples amplitude solves remain on the primary device in this stage.
"""
import atexit
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import cupy as cp
from . import reverse as ad
from .deferred_reads import DeferredRead
from .local_blocks import _key_token

_worker_state = threading.local()
_pools = {}


def configured_devices():
    text = os.environ.get('DLPNO_GRADIENT_DEVICES', '')
    if not text:
        return ()
    devices = tuple(int(x.strip()) for x in text.split(','))
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('Gradient device IDs must be nonempty and unique')
    if min(devices) < 0 or max(devices) >= cp.cuda.runtime.getDeviceCount():
        raise ValueError('Gradient device ID is outside the visible allocation')
    return devices


class _Selections:
    def __init__(self, table, operands):
        self.shape, self.ndim = table['shape'], len(table['shape'])
        self.table, self.operands = table['reads'], operands

    def __getitem__(self, key):
        return self.operands[self.table[_key_token(key)]]


def prepare_task(state, triplets, energy_only, workspace_bytes, reuse_reads):
    """Capture local operands sequentially; workers never mutate providers."""
    from .triples_source_batch_gpu import _layout
    from .triples_overlap_adjoint_gpu import overlap_batches
    inputs, identity = [], {}

    def operand(value):
        if isinstance(value,DeferredRead):
            return value.encode(operand)
        array = ad.primal(value)
        if np.dtype(array.dtype) != np.float64:
            raise ValueError('Parallel source boundary requires float64 arrays')
        token = ('record', value._record) if isinstance(value, ad.Variable) else ('array', id(value))
        if token not in identity:
            identity[token] = len(inputs)
            inputs.append(ad.Variable._from_record(value._record) if isinstance(value, ad.Variable) else value)
        return identity[token]

    tensors = {name: dict(shape=getattr(state, name).shape, reads={}) for name in ('J2c', 'qij', 'qia', 'qab')}
    tensors.update({name: dict(shape=getattr(state.orb, name).shape, reads={}) for name in ('S_pao', 'F_lmo')})

    def capture(name, key):
        table = tensors[name]['reads']
        token = _key_token(key)
        if token not in table:
            source = getattr(state.orb, name) if name in ('S_pao', 'F_lmo') else getattr(state, name)
            table[token] = operand(source[key])

    spec = dict(triplets=tuple(triplets), energy_only=energy_only, workspace_bytes=workspace_bytes,
                reuse_reads=reuse_reads, tensors=tensors, ij={}, trip={}, t_ribfs={}, t_paos={}, t_lmos={},
                pao={}, X={}, T={}, T1={}, X_tno={}, e_tno={})
    for t in triplets:
        layout = _layout(state, t)
        labels, lm = layout['labels'], layout['lm']
        aux, paos = state.t_ribfs[t], state.t_paos[t]
        spec['trip'][t] = labels
        for name in ('t_ribfs', 't_paos', 't_lmos'):
            spec[name][t] = np.asarray(getattr(state, name)[t]).copy()
        spec['X_tno'][t] = operand(state.X_tno[t])
        spec['e_tno'][t] = operand(state.e_tno[t])
        for a in labels:
            for b in tuple(lm)+tuple(labels):
                spec['ij'][a, b] = int(state.pl.ij[a, b])
            spec['T1'][a] = operand(state.T1[a])
            capture('F_lmo', (a, a))
            capture('qij', np.ix_(aux, [a], lm))
            capture('qia', np.ix_(aux, [a], paos))
        capture('J2c', np.ix_(aux, aux))
        capture('qab', np.ix_(aux, paos, paos))
        for p in layout['overlaps']:
            spec['pao'][p] = np.asarray(state.pao[p]).copy()
            spec['X'][p] = operand(state.X[p])
        for p in layout['projected']:
            spec['T'][p] = operand(state.T[p])
        for _, union, _ in overlap_batches(layout['overlaps'], state.X_tno[t], state.pao, state.X):
            capture('S_pao', np.ix_(paos, union))
    return spec, tuple(inputs)


def _restore(spec, inputs):
    view = SimpleNamespace(pl=SimpleNamespace(ij=spec['ij']))
    for name in ('trip', 't_ribfs', 't_paos', 't_lmos', 'pao'):
        setattr(view, name, spec[name])
    for name in ('X', 'T', 'T1', 'X_tno', 'e_tno'):
        setattr(view, name, {key: inputs[index] for key, index in spec[name].items()})
    view.orb = SimpleNamespace()
    for name, table in spec['tensors'].items():
        setattr(view.orb if name in ('S_pao', 'F_lmo') else view, name, _Selections(table, inputs))
    return view


def _initialize(device):
    _worker_state.device = device
    with cp.cuda.Device(device):
        _worker_state.pool = cp.cuda.MemoryPool()
        _worker_state.stream = cp.cuda.Stream(non_blocking=True)


def _evaluate(spec, values, seed, needed):
    if spec.get('task_kind') == 'local_graph':
        from .parallel_local_gpu import evaluate_local
        return evaluate_local(spec, values, seed, needed)
    from .triples_source_batch_gpu import source_batch
    variables = [ad.Variable(np.asarray(value), tracked=seed is not None and flag)
                 for value, flag in zip(values, needed)]
    workspaces = {} if seed is not None else None
    token = ad._replay_workspaces.set(workspaces)
    try:
        outputs = source_batch(_restore(spec, variables), spec['triplets'],
                               energy_only=spec['energy_only'], max_workspace_bytes=spec['workspace_bytes'],
                               reuse_reads=spec['reuse_reads'])
        if outputs is None:
            raise ValueError('Parallel task exceeds the configured source workspace')
        if seed is None:
            return np.concatenate([x.v.ravel() for x in outputs]), tuple(x.shape for x in outputs)
        lengths = [x.size for x in outputs]
        offsets = np.cumsum([0]+lengths)
        seeds = [seed[a:b].reshape(x.shape) for x, a, b in zip(outputs, offsets[:-1], offsets[1:])]
        selected = [x for x, flag in zip(variables, needed) if flag]
        graph = ad.Linearization(outputs, selected, boundary=variables)
        bars = iter(graph(seeds, release_graph=True, release_inputs=True))
        return tuple(next(bars) if flag else None for flag in needed)
    finally:
        if workspaces is not None:
            workspaces.clear()
        ad._replay_workspaces.reset(token)


def _execute(spec, values, seed, needed):
    start = time.perf_counter()
    device = _worker_state.device
    with cp.cuda.Device(device), _worker_state.stream, cp.cuda.using_allocator(_worker_state.pool.malloc):
        result = _evaluate(spec, values, seed, needed)
        _worker_state.stream.synchronize()
        used = _worker_state.pool.used_bytes()
    info = dict(device=device, pid=os.getpid(), thread=threading.get_ident(),
                phase='forward' if seed is None else 'reverse', seconds=time.perf_counter()-start,
                input_bytes=sum(np.asarray(x).nbytes for x in values), live_worker_device_bytes=used)
    return result, info


def _clear_worker():
    with cp.cuda.Device(_worker_state.device):
        _worker_state.stream.synchronize()
        before = _worker_state.pool.used_bytes()
        _worker_state.pool.free_all_blocks()
        return dict(device=_worker_state.device, live_device_bytes=before)


class SourcePool:
    def __init__(self, devices, executor='thread'):
        if executor not in ('thread', 'process'):
            raise ValueError('Gradient executor must be thread or process')
        self.devices, self.executor = tuple(devices), executor
        self.workers = []
        for device in self.devices:
            options = dict(max_workers=1, initializer=_initialize, initargs=(device,))
            if executor == 'process':
                options['mp_context'] = multiprocessing.get_context('spawn')
            self.workers.append((ThreadPoolExecutor if executor == 'thread' else ProcessPoolExecutor)(**options))
        self.closed = False

    def run(self, tasks):
        if self.closed:
            raise RuntimeError('A closed source pool cannot be reused')
        futures = [self.workers[i % len(self.workers)].submit(_execute, *task) for i, task in enumerate(tasks)]
        results, failures = [], []
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise failures[0]
        return results

    def close(self):
        if self.closed:
            return []
        self.closed = True
        stats = []
        try:
            for worker in self.workers:
                try:
                    stats.append(worker.submit(_clear_worker).result())
                except BaseException:
                    pass
        finally:
            for worker in self.workers:
                worker.shutdown(wait=True, cancel_futures=True)
        return stats


def pool_for(devices, executor=None):
    executor = executor or os.environ.get('DLPNO_GRADIENT_EXECUTOR', 'thread')
    key = tuple(devices), executor
    if key not in _pools:
        _pools[key] = SourcePool(*key)
    return _pools[key]


def close_pools():
    result = []
    for key in list(_pools):
        result.extend(_pools.pop(key).close())
    return result


atexit.register(close_pools)


def source_window(state, groups, *, devices, energy_only=False,
                  max_workspace_bytes=512 << 20, reuse_reads=True, pool=None):
    """One coarse operation over a bounded window of independent tasks."""
    pool = pool or pool_for(devices)
    start = time.perf_counter()
    prepared = [prepare_task(state, group, energy_only, max_workspace_bytes, reuse_reads) for group in groups]
    return task_window(prepared, devices=devices, kind='multi_gpu_triples_sources', pool=pool, prepare_start=start)


def task_window(prepared, *, devices, kind, pool=None, prepare_start=None, extra_info=None):
    """Host boundary shared by independent source and local graph workers."""
    pool = pool or pool_for(devices)
    start = time.perf_counter() if prepare_start is None else prepare_start
    inputs, identity, mappings = [], {}, []
    for spec, operands in prepared:
        mapping = []
        for value in operands:
            key = ('record', value._record) if isinstance(value, ad.Variable) else ('array', id(value))
            if key not in identity:
                identity[key] = len(inputs)
                inputs.append(value)
            mapping.append(identity[key])
        mappings.append(tuple(mapping))
    specs = [spec for spec, _ in prepared]
    del prepared
    values = tuple(cp.asnumpy(ad.primal(x)) if ad._xp([x]) is cp else np.asarray(ad.primal(x))
                   for x in inputs)
    if sum(x.nbytes for x in values) > (2 << 30):
        raise ValueError('Parallel window exceeds the 2 GiB host-input bound')
    info = dict(kind=kind, devices=list(devices), executor=pool.executor,
                tasks=len(specs), shared_inputs=len(inputs), input_bytes=sum(x.nbytes for x in values),
                prepare_seconds=time.perf_counter()-start, forward_seconds=0., reverse_seconds=0.,
                reverse_calls=0, device_task_counts={}, largest_live_worker_device_bytes=0)
    info.update(extra_info or {})

    def run(seed=None, needed=None):
        begin = time.perf_counter()
        tasks = []
        for i, (spec, mapping) in enumerate(zip(specs, mappings)):
            flags = [False]*len(mapping) if needed is None else [needed[j] for j in mapping]
            tasks.append((spec, tuple(values[j] for j in mapping),
                          None if seed is None else seed[task_offsets[i]:task_offsets[i+1]], flags))
        outputs = pool.run(tasks)
        phase = 'forward' if seed is None else 'reverse'
        info[phase+'_seconds'] += time.perf_counter()-begin
        for _, evidence in outputs:
            key = str(evidence['device'])
            info['device_task_counts'][key] = info['device_task_counts'].get(key, 0)+1
            info['largest_live_worker_device_bytes'] = max(info['largest_live_worker_device_bytes'], evidence['live_worker_device_bytes'])
        return [value for value, _ in outputs]

    forward = run()
    task_offsets = np.cumsum([0]+[packed.size for packed, _ in forward])
    packed = np.concatenate([x[0] for x in forward])
    shapes = tuple(shape for _, group_shapes in forward for shape in group_shapes)
    output_offsets = np.cumsum([0]+[int(np.prod(shape)) for shape in shapes])

    def pullback(seed, needed):
        parts = run(np.asarray(seed), needed)
        info['reverse_calls'] += 1
        result = [None]*len(inputs)
        for mapping, bars in zip(mappings, parts):
            for index, bar in zip(mapping, bars):
                if bar is not None:
                    if result[index] is None:
                        result[index] = bar.copy()
                    else:
                        result[index] += bar
        return tuple(np.zeros_like(values[i]) if flag and result[i] is None else result[i]
                     for i, flag in enumerate(needed))

    output = ad.operation(packed, inputs, lambda seed: pullback(seed, [True]*len(inputs)), pullback)
    return tuple(output[a:b].reshape(shape) for a, b, shape in zip(output_offsets[:-1], output_offsets[1:], shapes)), info
