"""Persistent multi-GPU DF graphs including every CCSD coefficient source."""
import pytest
from test_dlpno_ccsd_complete_sources_gpu import reference
from test_dlpno_ccsd_complete_sources_gpu import test_complete_batched_residual_vjp_and_coefficient_fd as check


@pytest.mark.parametrize('devices', ['0', '0,1'])
@pytest.mark.parametrize('batch_size', [1, 64])
@pytest.mark.parametrize('mixed_pairs', [False, True])
def test_df_multigpu_sources_and_iterative_transpose(reference, monkeypatch, devices, batch_size, mixed_pairs):
    from electrolites.dlpno.ccsd_sources_gpu import SourcePlan
    monkeypatch.setenv('DLPNO_CCSD_DF_DEVICES', devices)
    monkeypatch.setenv('DLPNO_CCSD_DF_BATCH_SIZE', str(batch_size))
    original, reports = SourcePlan.release, []
    def release(plan):
        was_released = plan.released
        original(plan)
        if not was_released:
            reports.append(plan.statistics)
    monkeypatch.setattr(SourcePlan, 'release', release)
    check(reference, monkeypatch, mixed_pairs, True)
    assert len(reports) == 1
    info = reports[0]['distributed_df']
    assert info['released'] and info['coefficient_derivatives_complete']
    assert {x['device'] for x in info['workers']} == {int(s) for s in devices.split(',')}
    assert all(x['released'] and x['live_bytes'] == 0 for x in info['workers'])
    assert all(x['selected_reverse_calls'] >= 2 for x in info['workers'])
