"""GPU transpose lifetimes for consuming and reusable density VJPs."""
import gc
import json
import weakref

import numpy as np
import pytest


@pytest.mark.parametrize('consume', [False, True])
def test_gpu_transposes_release_only_after_consuming_solve(consume, monkeypatch):
    cp = pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('A CUDA device is required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('A CUDA device is required')
    from pyscf import gto, scf
    from benchmarks.dlpno_gradient_check import STRESS
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.adjoint import TriplesTranspose
    from electrolites.dlpno.ccsd_adjoint_gpu import CCSDTranspose
    from electrolites.dlpno.local import LocalDLPNO, _ri_provider
    from electrolites.dlpno.local_blocks import BlockAdjointDLPNO
    monkeypatch.setenv('DLPNO_NGPU', '1')
    operators = []
    for cls in (CCSDTranspose, TriplesTranspose):
        original = cls.__init__
        def capture(self, *args, _original=original, **kwargs):
            _original(self, *args, **kwargs)
            operators.append(self)
        monkeypatch.setattr(cls, '__init__', capture)
    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-13, conv_tol_grad=1e-10)
    assert mf.converged
    base = LocalDLPNO(mf, triples='t1', verbose=0, **STRESS,
                     E_CONVERGENCE=1e-11, R_CONVERGENCE=1e-10,
                     MAXITER=150, T_CUT_ITER=0.)
    base.kernel()
    provider = _ri_provider(base, **base.local_ri_options)
    provider.validate_energy_support()
    state = BlockAdjointDLPNO(base, inputs=provider, checkpoint_ccsd=True,
                             ccsd_adjoint_backend='cupy', triples_backend='cupy')
    energy = state.kernel()
    leaves = list(state.leaves.values())
    graph = ad.Linearization([energy], leaves)
    expected = graph([np.asarray(1.)])
    assert len(operators) == 2
    assert all(not op.released for op in operators)
    owned_arrays = []
    for op in operators:
        value = op.linearization.inputs[0].value if isinstance(op, CCSDTranspose) else op.diagonal
        assert isinstance(value, cp.ndarray)
        owned_arrays.append(weakref.ref(value))
    del value
    actual = state.densities(release_graph=consume, release_inputs=consume)
    error = max(float(np.max(abs(actual[name]-want), initial=0.))
                for name, want in zip(state.leaves, expected))
    assert error < 2e-10
    assert not state._release_transposes
    if consume:
        assert all(op.released for op in operators)
        assert state.diagnostics['ccsd']['operator_released']
        assert state.diagnostics['triples']['operator_released']
        gc.collect()
        assert all(ref() is None for ref in owned_arrays)
        # Densities are cached, without invoking a released transpose.
        assert state.densities() is actual
        for op in operators:
            with pytest.raises(RuntimeError, match='released'):
                op([np.zeros(1)])
    else:
        repeat = graph([np.asarray(.37)])
        assert all(not op.released for op in operators)
        assert all(ref() is not None for ref in owned_arrays)
        for got, want in zip(repeat, expected):
            assert np.max(abs(got-.37*want), initial=0.) < 2e-10
        for op in operators:
            op.release()
    print('ADJOINT_LIFETIME_REPORT '+json.dumps(dict(consuming=consume,
          max_density_error=error,operators=len(operators),
          automatic_release_observed=consume,
          owned_gpu_arrays_released_by_end=all(ref() is None for ref in owned_arrays))))
