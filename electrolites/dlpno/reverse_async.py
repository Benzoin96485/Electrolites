"""Dependency-ready outer VJP schedule with explicitly submitted source tasks.

Scalar/GPU local graphs retain the serial evaluator. A source result is
merged only after all worker futures join; a parent becomes ready only
after every child has contributed. No amplitude or coefficient is frozen.
"""
from concurrent.futures import wait, FIRST_COMPLETED
import heapq
import os
import time


class Pending:
    def __init__(self, futures, resolve):
        self.waitables = tuple(futures)
        self.resolve = resolve
        self.finished = False
        self.value = None
        self.error = None

    def done(self):
        return all(f.done() for f in self.waitables)

    def result(self):
        if not self.finished:
            try:self.value = self.resolve()
            except BaseException as error:self.error = error
            finally:
                self.finished = True
                self.resolve = None
        if self.error is not None:raise self.error
        return self.value

    def map(self, function):
        return Pending(self.waitables, lambda: function(self.result()))


def evaluate(graph,seeds,*,release_graph=False,release_inputs=False):
    from .reverse import IndexedCotangent, _xp
    if graph._consumed:
        raise RuntimeError('A consumed reverse graph cannot be reused')
    if release_graph and graph.inputs is None:
        raise ValueError('Graph release requires explicit retained inputs')
    if release_inputs and not release_graph:
        raise ValueError('Input release requires a consuming graph sweep')
    limit=int(os.environ.get('DLPNO_ASYNC_SOURCE_LIMIT','4'))
    if not 1<=limit<=8:
        raise ValueError('Async source limit must be between one and eight')
    graph._consumed=release_graph
    input_records=set() if graph.inputs is None else set(graph.inputs)
    records=list(reversed(graph.records));position={r:i for i,r in enumerate(records)}
    children={r:0 for r in records}
    for r in records:
        if r not in graph.stops:
            for p in r.parents:
                if p in children:children[p]+=1
    ready=[i for i,r in enumerate(records) if children[r]==0];heapq.heapify(ready)
    bars={};pending={};completed=0
    stats=dict(allocated_elements=0,adopted_elements=0,contribution_elements=0,indexed_contributions=0,indexed_elements=0,
        skipped_parent_pullbacks=0,released_records=0,released_value_bytes=0,
        released_input_records=0,released_input_bytes=0,async_submissions=0,async_wait_seconds=0.,
        maximum_pending=0,scheduler='dependency_ready',completed_records=0)

    def release(record,is_input=False):
        if record.value is None:return
        if is_input and record not in bars:
            bars[record]=_xp([record.value]).zeros_like(record.value)
            stats['allocated_elements']+=bars[record].size
        stats['released_records']+=1;stats['released_value_bytes']+=record.value.nbytes
        if is_input:
            stats['released_input_records']+=1;stats['released_input_bytes']+=record.value.nbytes
        record.value=None;record.parents=()
        record.pullback=record.selected_pullback=record.async_pullback=None

    def accumulate(record,contribution,owned=False):
        indexed=isinstance(contribution,IndexedCotangent)
        stats['contribution_elements']+=contribution.values.size if indexed else contribution.size
        if record not in bars:
            bars[record]=_xp([record.value]).zeros_like(record.value) if indexed else (contribution if owned else contribution.copy())
            if owned and not indexed:stats['adopted_elements']+=contribution.size
            stats['allocated_elements']+=bars[record].size
            if not indexed:return
        if indexed:
            stats['indexed_contributions']+=1;stats['indexed_elements']+=contribution.values.size
            contribution.add_to(bars[record])
        else:bars[record]+=contribution

    def finish(record,contributions):
        nonlocal completed
        if contributions is not None:
            for parent,contribution in zip(record.parents,contributions):
                if parent in graph.active:accumulate(parent,contribution,record.owns_cotangents)
            if graph.inputs is not None:bars.pop(record,None)
        if record not in graph.stops:
            for parent in record.parents:
                if parent in children:
                    children[parent]-=1
                    if children[parent]==0:heapq.heappush(ready,position[parent])
                    if children[parent]<0:raise RuntimeError('Async VJP dependency counted twice')
        if release_graph and record not in graph.stops:release(record)
        elif release_inputs and record in input_records:release(record,True)
        completed+=1

    def collect(block=False):
        started=time.perf_counter()
        while True:
            finished=[r for r,p in pending.items() if p.done()]
            if finished or not block or not pending:break
            futures=[f for p in pending.values() for f in p.waitables if not f.done()]
            if not futures:raise RuntimeError('Pending source has no live waitable')
            wait(futures,return_when=FIRST_COMPLETED)
        if block:stats['async_wait_seconds']+=time.perf_counter()-started
        # Deterministic order among results available at this checkpoint.
        for record in sorted(finished,key=position.get):
            result=pending[record].result()
            del pending[record]
            finish(record,result)
            del result

    for record,seed in zip(graph.outputs,seeds):
        if record in graph.active:accumulate(record,seed)
    try:
        while ready or pending:
            collect()
            if not ready:
                collect(True);continue
            record=records[heapq.heappop(ready)];bar=bars.get(record)
            if bar is None or record.pullback is None or record in graph.stops:
                finish(record,None);del bar;continue
            needed=[parent in graph.active for parent in record.parents]
            if record.async_pullback is not None:
                while len(pending)>=limit:collect(True)
                pending[record]=record.async_pullback(bar,needed)
                stats['async_submissions']+=1;stats['maximum_pending']=max(stats['maximum_pending'],len(pending))
                stats['skipped_parent_pullbacks']+=needed.count(False)
            else:
                if record.selected_pullback is None:contributions=record.pullback(bar)
                else:
                    stats['skipped_parent_pullbacks']+=needed.count(False)
                    contributions=record.selected_pullback(bar,needed)
                finish(record,contributions)
                del contributions
            del bar
        if completed!=len(records):raise RuntimeError('Async VJP did not visit the complete graph')
        if release_inputs:
            for record in input_records:release(record,True)
        stats['completed_records']=completed;graph.last_statistics=stats
        if graph.inputs is None:return bars
        return [bars[r] if r in bars else _xp([r.value]).zeros_like(r.value) for r in graph.inputs]
    except BaseException as error:
        # A failed sibling must never leave a task using an abandoned graph.
        for record,result in list(pending.items()):
            try:result.result()
            except BaseException as cleanup:
                if cleanup is not error:error.add_note('Pending source completion: '+repr(cleanup))
        raise
