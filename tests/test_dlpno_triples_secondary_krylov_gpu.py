"""Secondary-device T1 solve and same-order reduction against CPU equations."""
import json
from types import MethodType

import numpy as np
import pytest
from test_dlpno_triples_resident_gpu import test_resident_complete_sources as check_triples


@pytest.mark.parametrize('cut', [0., .035, 2.])
@pytest.mark.parametrize('capture', [False, True])
@pytest.mark.parametrize('stream_reduction', [False, True])
def test_secondary_transpose_and_complete_sources(monkeypatch, cut, capture, stream_reduction):
    import cupy as cp
    from electrolites.dlpno.triples_resident_gpu import ResidentTranspose
    assert cp.cuda.runtime.getDeviceCount() >= 2
    monkeypatch.setenv('DLPNO_TRIPLES_KRYLOV_DEVICE', '1')
    monkeypatch.setenv('DLPNO_ADJOINT_CUDA_GRAPH', str(int(capture)))
    monkeypatch.setenv('DLPNO_TRIPLES_STREAM_REDUCTION', str(int(stream_reduction)))
    monkeypatch.setenv('DLPNO_TRIPLES_EARLY_RELEASE', '1')
    original = ResidentTranspose.__call__
    errors = []
    def call(operator, seeds):
        assert operator.diagonal.device.id == operator.solve_device == 1
        assert operator.statistics['krylov_diagonal_replica_bytes'] == operator.diagonal.nbytes
        result = original(operator, seeds)
        assert result[0].device.id == 1
        for scale in (0., -.5, 1.37):
            value = original(operator, [scale*seeds[0]])[0]
            assert value.device.id == 1
            errors.append(float(cp.max(abs(value-scale*result[0]))))
        return result
    monkeypatch.setattr(ResidentTranspose, '__call__', call)
    before = cp.cuda.Device().id
    check_triples((0, 1), cut)
    assert cp.cuda.Device().id == before
    assert max(errors) < 3e-11
    print('SECONDARY_TRANSPOSE_REPORT '+json.dumps(dict(cut=cut,capture=capture,stream_reduction=stream_reduction,maximum_error=max(errors))))


@pytest.mark.parametrize('cut', [0., .035, 2.])
@pytest.mark.parametrize('capture', [False, True])
def test_secondary_gmres_matches_independent_cpu_matrix(monkeypatch, cut, capture):
    import cupy as cp
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.ref_rhf import RefDLPNO
    from electrolites.dlpno.triples_resident_gpu import ResidentTriples, ResidentTranspose
    from electrolites.dlpno.krylov_gpu import gmres
    from test_dlpno_triples_residual_gpu import raw_sources, state_from
    monkeypatch.setenv('DLPNO_TRIPLES_KRYLOV_DEVICE', '1')
    monkeypatch.setenv('DLPNO_ADJOINT_CUDA_GRAPH', str(int(capture)))
    monkeypatch.setenv('DLPNO_TRIPLES_STREAM_REDUCTION', '1')
    monkeypatch.setenv('DLPNO_TRIPLES_EARLY_RELEASE', '1')
    raw, rng = raw_sources()
    state = state_from({name: ad.Variable(value.copy()) for name,value in raw.items()}, cut)
    connections = RefDLPNO._triples_couplings(state)
    offsets = np.cumsum([0]+[raw['t'+str(t)].size for t in range(4)])
    size = offsets[-1]
    def cpu_residual(vector):
        data = dict(raw)
        for t,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
            data['t'+str(t)] = vector[a:b].reshape(raw['t'+str(t)].shape)
        numeric = state_from(data, cut)
        numeric._matmul_3d = MethodType(RefDLPNO._matmul_3d, numeric)
        return np.concatenate([RefDLPNO._triplet_residual(numeric,t,c).ravel()
                               for t,c in enumerate(connections)])
    zero = cpu_residual(np.zeros(size))
    matrix = np.column_stack([cpu_residual(v)-zero for v in np.eye(size)])
    rhs = rng.normal(size=size)
    expected = np.linalg.solve(matrix.T, rhs)
    plan = ResidentTriples(state, (0,1), connections=connections, workspace_bytes=1 << 20)
    operator = ResidentTranspose(plan)
    before = cp.cuda.Device().id
    try:
        with cp.cuda.Device(operator.solve_device):
            device_rhs = cp.asarray(rhs)
            assert operator.diagonal.device.id == device_rhs.device.id == 1
            seen = []
            def action(vector):
                value = operator([vector])[0]
                assert vector.device.id == value.device.id == 1
                seen.append(value.device.id)
                return value
            solution, info = gmres(action, device_rhs, operator.diagonal,
                                  restart=8,maxiter=20,rtol=1e-12,atol=1e-13,max_residual=1e-8)
            result = cp.asnumpy(solution)
            assert solution.device.id == 1 and info['status'] == 0
            assert np.array_equal(cp.asnumpy(device_rhs), rhs)
        error = float(np.max(abs(result-expected)))
        residual = float(np.max(abs(matrix.T@result-rhs)))
        assert error < 3e-11 and residual < 1e-10
        assert len(seen) == info['matvec_calls']
        operator.release()
        assert not plan.released
        assert np.array_equal(cp.asnumpy(solution), result)
        assert cp.cuda.Device().id == before
    finally:
        operator.release()
        plan.release()
    assert all(x['live_bytes']==0 for x in plan.statistics['released_devices'])
    print('SECONDARY_KRYLOV_REPORT '+json.dumps(dict(cut=cut,capture=capture,size=int(size),solution_error=error,independent_residual=residual,solver=info)))


def test_krylov_device_must_be_a_resident_shard(monkeypatch):
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.ref_rhf import RefDLPNO
    from electrolites.dlpno.triples_resident_gpu import ResidentTriples, ResidentTranspose
    from test_dlpno_triples_residual_gpu import raw_sources, state_from
    raw, _ = raw_sources()
    state = state_from({name: ad.Variable(value.copy()) for name,value in raw.items()}, .035)
    plan = ResidentTriples(state, (0,), connections=RefDLPNO._triples_couplings(state))
    monkeypatch.setenv('DLPNO_TRIPLES_KRYLOV_DEVICE', '1')
    try:
        with pytest.raises(ValueError,match='resident shard'):
            ResidentTranspose(plan)
    finally:
        plan.release()
