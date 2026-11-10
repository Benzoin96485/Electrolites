"""Overlap the unchanged numerical CCSD solve with complete source preparation.

Only the numerical primal is remote. The full coefficient graph and the
amplitude CUDA-graph transpose retain their established differentiation.
Requires an existing isolated process pool; disabled by default.
"""
import gc
import os
import time
from types import SimpleNamespace
import numpy as np
import cupy as cp


def solve_primal(snapshot):
    from .ccsd_gpu import CCSDGPU
    start = time.perf_counter()
    solver = None
    try:
        solver = CCSDGPU(snapshot, None)
        built = time.perf_counter()
        energy, weak = solver.kernel(verbose=False)
        solved = time.perf_counter()
        result = SimpleNamespace(T=solver.T.to_list(),
            T1=[x.ravel() for x in solver.T1.to_list()], e_lccsd=energy, de_weak=weak)
        cp.cuda.get_current_stream().synchronize()
        result.statistics = dict(build_seconds=built-start, solve_seconds=solved-built,
            output_seconds=time.perf_counter()-solved, numerical_solver='unchanged_CCSDGPU',
            coefficients_approximated=False, started_at=start)
        return result
    finally:
        solver = None
        gc.collect()
        cp.cuda.get_current_stream().synchronize()
        if 'result' in locals():
            result.statistics.update(seconds=time.perf_counter()-start, completed_at=time.perf_counter())


def prepare(state):
    from .ccsd_adjoint_gpu import _fixed_state
    from .ccsd_sources_gpu import SourcePlan
    from .resident_shared_store_gpu import _task
    if state.xp is not np or state.ccsd_primal_backend != 'gpu':
        raise ValueError('CCSD primal pipeline requires the host graph and GPU primal')
    if os.environ.get('DLPNO_CCSD_BATCHED_SOURCES','0') != '1':
        raise ValueError('CCSD primal pipeline requires complete batched source derivatives')
    pool = getattr(state, '_resident_region_pool', None)
    if pool is None:
        raise ValueError('CCSD primal pipeline requires an existing resident pool; it cannot create workers')
    if (pool.closed or len(pool.devices)<2 or pool.statistics.get('executor')!='process'
        or not pool.statistics.get('isolated_process_devices') or not pool.statistics.get('inline_primary')):
        raise ValueError('CCSD primal pipeline requires an existing isolated process pool with >=2 devices')
    budget = int(os.environ.get('DLPNO_CCSD_SOURCE_MEMORY_MIB','2048')) << 20
    if not (32 << 20) <= budget <= (32 << 30):
        raise ValueError('CCSD source memory budget must be between 32 and 32768 MiB')
    if hasattr(state, '_ccsd_primal_reserved_rank'):
        raise RuntimeError('A CCSD numerical solve is already in flight')
    start = time.perf_counter()
    snapshot = _fixed_state(state, before_ccsd=True)
    snapshot_seconds = time.perf_counter()-start
    rank = 1
    future = pool.workers[rank].submit(_task, 'ccsd_primal', snapshot)
    submitted = time.perf_counter()
    state._ccsd_primal_reserved_rank = rank
    plan = None
    try:
        plan = SourcePlan(state, budget_bytes=budget, prepare_only=True)
        prepared = time.perf_counter()
        result, worker = future.result()
        joined = time.perf_counter()
        overlap = max(0., min(prepared,result.statistics['completed_at'])-
                      max(submitted,result.statistics['started_at']))
        state.diagnostics['ccsd_primal_pipeline'] = dict(
            rank=rank, worker=worker, primal=result.statistics, snapshot_seconds=snapshot_seconds,
            source_prepare_seconds=prepared-submitted, join_wait_seconds=joined-prepared,
            seconds=joined-start, measured_overlap_seconds=overlap,
            new_workers_created=0, coefficients_complete=True, amplitude_transpose_on_primary=True)
        return result, plan
    except BaseException as error:
        if plan is not None:
            plan.release()
        # Join the scheduled solve even if source preparation failed. Pool
        # shutdown remains owned by the surrounding gradient's scope.
        try:
            future.result()
        except BaseException as cleanup:
            if cleanup is not error:
                error.add_note('CCSD worker completion: '+repr(cleanup))
        raise
    finally:
        del state._ccsd_primal_reserved_rank
