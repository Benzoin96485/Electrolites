"""CUDA Graph lifetime, changed seeds and independent complete source oracles."""
import json
import numpy as np
import pytest
from test_dlpno_ccsd_complete_sources_gpu import reference
from test_dlpno_ccsd_complete_sources_gpu import test_complete_batched_residual_vjp_and_coefficient_fd as check_ccsd
from test_dlpno_triples_resident_gpu import test_resident_complete_sources as check_triples


def test_replay_changes_seeds_and_owns_memory():
    import cupy as cp
    from electrolites.dlpno.cuda_graph_replay import Replay
    from electrolites.dlpno import reverse as ad
    rng = np.random.default_rng(326)
    a = rng.normal(size=(32,16))
    indices = np.arange(256) % 73
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        device_a, device_indices = cp.asarray(a), cp.asarray(indices)
        def action(seeds):
            value = ad._matmul(device_a, seeds[0].reshape(16,8))
            result = cp.zeros(73)
            cp.add.at(result, device_indices, value.ravel())
            contracted = ad._einsum(cp,'ik,kj->ij',device_a,seeds[0].reshape(16,8),optimize=True)
            return [result, cp.sin(seeds[0])+seeds[0]**2, contracted]
        replay = Replay(action, [cp.zeros(128)], memory_bytes=32 << 20)
        saved, errors = [], []
        try:
            for _ in range(5):
                x = rng.normal(size=128)
                outputs = replay([cp.asarray(x)])
                expected = np.zeros(73)
                np.add.at(expected, indices, (a @ x.reshape(16,8)).ravel())
                gold = [expected, np.sin(x)+x*x, a @ x.reshape(16,8)]
                errors.extend(float(np.max(abs(cp.asnumpy(y)-g))) for y,g in zip(outputs,gold))
                saved.append((outputs,gold))
            # Previous caller-owned outputs must survive later replays and
            # destruction of the graph, including its temporary memory pool.
            with pytest.raises(ValueError, match='shape'):
                replay([cp.zeros(127)])
        finally:
            replay.release()
        assert replay.statistics['live_bytes'] == replay.statistics['reserved_bytes'] == 0
        for outputs, gold in saved:
            errors.extend(float(np.max(abs(cp.asnumpy(y)-g))) for y,g in zip(outputs,gold))
        assert max(errors) < 1e-12
        with pytest.raises(RuntimeError, match='released'):
            replay([cp.zeros(128)])
    print('CUDA_GRAPH_REPORT '+json.dumps(dict(maximum_error=max(errors),statistics=replay.statistics)))


def test_replay_memory_bound():
    import cupy as cp
    from electrolites.dlpno.cuda_graph_replay import Replay
    stream = cp.cuda.Stream(non_blocking=True)
    with stream, pytest.raises(cp.cuda.memory.OutOfMemoryError):
        Replay(lambda x: [x[0]*2], [cp.ones(1024)], memory_bytes=512)


@pytest.mark.parametrize('shape_a,shape_b,transpose_b', [
    ((17,),(17,),False), ((4,17),(17,),False),
    ((17,),(2,17,3),False), ((2,1,5,7),(1,3,4,7),True),
    ((3,0),(0,5),False)])
def test_captured_gemm_layouts(shape_a, shape_b, transpose_b):
    import cupy as cp
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.cuda_graph_replay import Replay
    rng=np.random.default_rng(881)
    stream=cp.cuda.Stream(non_blocking=True)
    with stream:
        def action(seeds):
            b=seeds[1].swapaxes(-1,-2) if transpose_b else seeds[1]
            return [ad._matmul(seeds[0],b)]
        replay=Replay(action,[cp.zeros(shape_a),cp.zeros(shape_b)],memory_bytes=32<<20)
        errors=[]
        try:
            for _ in range(3):
                a,b=rng.normal(size=shape_a),rng.normal(size=shape_b)
                expected=a @ (b.swapaxes(-1,-2) if transpose_b else b)
                result=replay([cp.asarray(a),cp.asarray(b)])[0]
                errors.append(float(np.max(abs(cp.asnumpy(result)-expected),initial=0.)))
            assert max(errors)<1e-12
        finally:replay.release()
        assert replay.statistics['live_bytes']==replay.statistics['reserved_bytes']==0
    print('CAPTURE_GEMM_REPORT '+json.dumps(dict(shape_a=shape_a,shape_b=shape_b,
        transpose_b=transpose_b,maximum_error=max(errors),statistics=replay.statistics)))


@pytest.mark.parametrize('devices', ['', '0', '0,1'])
@pytest.mark.parametrize('mixed_pairs', [False, True])
def test_captured_complete_ccsd(reference, monkeypatch, devices, mixed_pairs):
    from electrolites.dlpno.ccsd_sources_gpu import SourcePlan, SourceTranspose
    monkeypatch.setenv('DLPNO_ADJOINT_CUDA_GRAPH', '1')
    monkeypatch.setenv('DLPNO_CCSD_DF_DEVICES', devices)
    monkeypatch.setenv('DLPNO_CCSD_DF_BATCH_SIZE', '64')
    original_plan, original_transpose = SourcePlan.release, SourceTranspose.release
    records = []
    def release_plan(plan):
        was_released = plan.released
        original_plan(plan)
        if devices and not was_released:
            for worker in plan.statistics['distributed_df']['workers']:
                records.extend(worker['cuda_graphs'])
    def release_transpose(op):
        replay = op.replay
        original_transpose(op)
        if replay is not None:
            records.append(replay.statistics)
    monkeypatch.setattr(SourcePlan, 'release', release_plan)
    monkeypatch.setattr(SourceTranspose, 'release', release_transpose)
    check_ccsd(reference, monkeypatch, mixed_pairs, True)
    print('CCSD_CAPTURE_REPORT '+json.dumps(dict(devices=devices,mixed_pairs=mixed_pairs,graphs=records)))
    assert records and all(x['calls'] >= 2 for x in records), records
    assert all(x['released'] and x['live_bytes'] == x['reserved_bytes'] == 0 for x in records), records


@pytest.mark.parametrize('devices', [(0,), (0,1)])
@pytest.mark.parametrize('cut', [0., .035, 2.])
def test_captured_complete_triples(monkeypatch, devices, cut):
    import cupy as cp
    from electrolites.dlpno.triples_resident_gpu import ResidentTranspose, ResidentTriples
    monkeypatch.setenv('DLPNO_ADJOINT_CUDA_GRAPH', '1')
    original_call, original_release = ResidentTranspose.__call__, ResidentTriples.release
    errors, records = [], []
    def call(op, seeds):
        result = original_call(op, seeds)
        for scale in (0., -.5, 1.37):
            changed = original_call(op, [scale*seeds[0]])
            errors.append(float(cp.max(abs(changed[0]-scale*result[0]))))
        return result
    def release(plan):
        was_released = plan.released
        original_release(plan)
        if not was_released:
            records.extend(x['cuda_graph'] for x in plan.statistics['released_devices'])
    monkeypatch.setattr(ResidentTranspose, '__call__', call)
    monkeypatch.setattr(ResidentTriples, 'release', release)
    check_triples(devices, cut)
    assert max(errors) < 3e-11
    assert records and all(x and x['calls'] == 4 for x in records), records
    assert all(x['released'] and x['live_bytes'] == x['reserved_bytes'] == 0 for x in records), records
    print('TRIPLES_CAPTURE_REPORT '+json.dumps(dict(devices=list(devices),cut=cut,maximum_repeated_error=max(errors),graphs=records)))
