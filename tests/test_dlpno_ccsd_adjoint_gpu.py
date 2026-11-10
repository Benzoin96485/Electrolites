"""Batched CCSD transpose, physical gradients and unfrozen nuclear FD."""
import json
import time

import numpy as np
import pytest
from pyscf import gto, scf


@pytest.fixture(scope='module', autouse=True)
def cuda():
    cp = pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('A CUDA device is required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('A CUDA device is required')


@pytest.fixture(scope='module')
def reference():
    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57', basis='6-31g', verbose=0)
    return scf.RHF(mol).density_fit().run(conv_tol=1e-13, conv_tol_grad=1e-10)


def options(triples, mixed_pairs=False):
    from benchmarks.dlpno_gradient_check import STRESS
    result = dict(triples=triples, verbose=0, **STRESS, E_CONVERGENCE=1e-11,
                  R_CONVERGENCE=1e-10, MAXITER=150, T_CUT_ITER=0.)
    if mixed_pairs:
        # Test-only limit: diagonal CC pairs and fixed weak off-diagonal
        # MP2 pairs exercise the independent-variable and source maps.
        result['T_CUT_PAIRS'] = 1e6
    return result


@pytest.mark.parametrize('onfly', ['0', '1'])
@pytest.mark.parametrize('mixed_pairs', [False, True])
def test_ccsd_transpose_vs_reference_and_amplitude_direction(reference, monkeypatch, onfly, mixed_pairs):
    import cupy as cp
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.adjoint import AdjointDLPNO, Amplitudes, host
    from electrolites.dlpno.ccsd_adjoint_gpu import CCSDTranspose
    from electrolites.dlpno.local import LocalDLPNO, _ri_provider
    from electrolites.dlpno.local_blocks import BlockAdjointDLPNO
    from electrolites.dlpno.response import ResponseDLPNO
    monkeypatch.setenv('DLPNO_NGPU', '1')
    monkeypatch.setenv('DLPNO_ONFLY_S', onfly)
    base = LocalDLPNO(reference, **options(None, mixed_pairs));base.kernel()
    if mixed_pairs:assert np.any(~base.strong)
    original = AdjointDLPNO._implicit
    reports = []
    def inspect_operator(self, name, independent, residual, variables):
        if name == 'ccsd':
            reference_vjp = ad.Linearization([residual], [independent])
            operator = CCSDTranspose(self, variables, independent, residual)
            rng = np.random.default_rng(911)
            errors = []
            for _ in range(3):
                seed = rng.normal(size=independent.size)
                expected = host(reference_vjp([seed])[0])
                actual = cp.asnumpy(operator([cp.asarray(seed)])[0])
                errors.append(float(np.max(abs(actual-expected))))
            assert max(errors) < 2e-9
            # Independent central difference in amplitude space. Both
            # matrix operands and the full T1-dressed residual are varied.
            direction = rng.normal(size=independent.size)
            direction /= np.linalg.norm(direction)
            primal = self._primal_state();primal.numeric = np
            mapping = Amplitudes(primal, cc=True)
            values = []
            h = 1e-5
            for sign in (1, -1):
                mapping.install(host(independent)+sign*h*direction)
                r2, r1 = ResponseDLPNO._cc_residual(primal)
                values.append(host(mapping.pack(r2, r1)))
            lhs = float(seed@((values[0]-values[1])/(2*h)))
            rhs = float(direction@actual)
            assert abs(lhs-rhs) < 2e-8
            reports.append(dict(onfly=onfly,mixed_pairs=mixed_pairs,variables=independent.size,
                random_vjp_max_errors=errors,directional_identity_error=abs(lhs-rhs),
                statistics=operator.statistics,last_action=operator.last_statistics))
            operator.release()
        return original(self, name, independent, residual, variables)
    monkeypatch.setattr(AdjointDLPNO, '_implicit', inspect_operator)
    provider = _ri_provider(base, **base.local_ri_options)
    provider.validate_energy_support()
    state = BlockAdjointDLPNO(base, inputs=provider, checkpoint_ccsd=True)
    state.kernel()
    assert len(reports) == 1
    print('CCSD_OPERATOR_REPORT '+json.dumps(reports[0], sort_keys=True))


@pytest.mark.parametrize('triples,mixed_pairs', [(None, False), ('t0', False), ('t1', False), (None, True)])
def test_batched_ccsd_lambda_complete_gradient_and_independent_fd(reference, triples, mixed_pairs, tmp_path, monkeypatch):
    from benchmarks.dlpno_gradient_check import topology, tracked_driver
    from electrolites.dlpno.local import LocalDLPNO, LocalGradients
    monkeypatch.setenv('DLPNO_NGPU', '1')
    monkeypatch.setenv('DLPNO_ONFLY_S', '1')
    driver = tracked_driver(LocalDLPNO)
    base = driver(reference, **options(triples, mixed_pairs));base.kernel()
    start = time.perf_counter()
    previous = base.nuc_grad_method(checkpoint_ccsd=True)
    expected = previous.kernel();cpu_seconds = time.perf_counter()-start
    path = tmp_path/'response.npz'
    gradient = base.nuc_grad_method(ccsd_adjoint_backend='cupy', checkpoint_ccsd=True,
                                   response_checkpoint=path)
    start = time.perf_counter();actual = gradient.kernel();gpu_seconds = time.perf_counter()-start
    error = float(np.max(abs(actual-expected)))
    assert error < 2e-8
    ccsd = gradient.diagnostics['amplitudes']['ccsd']
    assert ccsd['operator_kind'] == 'batched_ccsd_transpose'
    assert ccsd['adjoint_residual'] < 1e-8
    assert ccsd['backend'] == 'cupy'
    assert ccsd['df_checkpoints']['groups'] > 0
    recovered = LocalGradients.from_response_checkpoint(path)
    assert np.max(abs(recovered.kernel()-actual)) < 2e-11
    original_topology = topology(base, effective=True)
    differences = []
    for step in (1e-4, 5e-5):
        energies = []
        for sign in (1, -1):
            coordinates = reference.mol.atom_coords()
            coordinates[1, 2] += sign*step
            mol = reference.mol.copy().set_geom_(coordinates, unit='Bohr')
            mf = scf.RHF(mol).density_fit(auxbasis=reference.with_df.auxbasis)
            mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
            mf.kernel(dm0=reference.make_rdm1());assert mf.converged
            displaced = driver(mf, **options(triples, mixed_pairs));displaced.kernel()
            assert topology(displaced, effective=True) == original_topology
            energies.append(mf.e_tot+displaced.e['corr'])
        numerical = (energies[0]-energies[1])/(2*step)
        differences.append(dict(step_bohr=step,gradient=numerical,error=actual[1, 2]-numerical,
                       topology_stable=True,scf_boys_domains_recomputed=True))
    assert max(abs(row['error']) for row in differences) < 2e-7
    print('CCSD_GRADIENT_REPORT '+json.dumps(dict(triples=triples,mixed_pairs=mixed_pairs,max_gradient_error=error,
        cpu_gradient_seconds=cpu_seconds,gpu_gradient_seconds=gpu_seconds,
        lambda_info=ccsd,independent_nuclear_fd=differences), sort_keys=True))
