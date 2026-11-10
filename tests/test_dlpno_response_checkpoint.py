"""Response restart preserves gradients and rejects mismatched artifacts."""
import json
from pathlib import Path

import numpy as np
import pytest
from pyscf import gto, scf

from electrolites.dlpno.local import LocalDLPNO, LocalGradients
from electrolites.dlpno import response_checkpoint as checkpoint


@pytest.fixture(scope='module')
def reference():
    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57', basis='6-31g', verbose=0)
    return scf.RHF(mol).density_fit().run(conv_tol=1e-13, conv_tol_grad=1e-10)


def calculation(reference, triples):
    from benchmarks.dlpno_gradient_check import STRESS
    base = LocalDLPNO(reference, triples=triples, verbose=0, **STRESS,
        E_CONVERGENCE=1e-11, R_CONVERGENCE=1e-10, MAXITER=150, T_CUT_ITER=0.)
    base.kernel()
    return base


@pytest.mark.parametrize('triples', ['t0', 't1'])
def test_restart_matches_all_uninterrupted_components(reference, triples, tmp_path, monkeypatch):
    base = calculation(reference, triples)
    path = tmp_path/'response.npz'
    original = base.nuc_grad_method(response_checkpoint=path)
    expected = original.kernel()
    manifest = json.loads(Path(str(path)+'.json').read_text())
    restored = LocalGradients.from_response_checkpoint(path, expected_sha256=manifest['sha256'])
    assert not hasattr(restored.base, 'orb')  # No correlation solver is deserialized.
    assert restored.base.e == base.e
    assert np.array_equal(restored.base.mf.mo_coeff, reference.mo_coeff)
    def forbidden(*args, **kwargs):
        raise AssertionError('No energy, amplitude or orbital solve is needed during response restart')
    from electrolites.dlpno import local
    monkeypatch.setattr(local, '_ri_provider', forbidden)
    monkeypatch.setattr(local, 'orbital_densities', forbidden)
    monkeypatch.setattr(local, 'BlockAdjointDLPNO', forbidden)
    actual = restored.kernel()
    assert np.max(abs(actual-expected)) < 2e-11
    for key in original.components:
        assert np.max(abs(restored.components[key]-original.components[key])) < 2e-11
    assert set(restored.timings) == {'orbital_contraction', 'hf_gradient'}
    assert np.max(abs(restored.kernel(atmlst=[2, 0])-expected[[2, 0]])) < 2e-11
    with pytest.raises(ValueError, match='atmlst'):
        restored.kernel(atmlst=[0, 0])
    with pytest.raises(ValueError, match='data hash'):
        LocalGradients.from_response_checkpoint(path, expected_sha256='0'*64)
    with pytest.raises(FileExistsError):
        checkpoint.save_response_checkpoint(path, original, base, original.components['local_ri'])
    # Corrupted data cannot be mistaken for a valid completed adjoint.
    damaged = tmp_path/'damaged.npz'
    data = bytearray(path.read_bytes());data[len(data)//2] ^= 1
    damaged.write_bytes(data)
    Path(str(damaged)+'.json').write_text(Path(str(path)+'.json').read_text())
    with pytest.raises(ValueError, match='data hash'):
        LocalGradients.from_response_checkpoint(damaged)
    # Reusing a numerically valid checkpoint with different code is forbidden.
    monkeypatch.setattr(checkpoint, 'source_sha256', lambda: {'changed.py': 'new'})
    with pytest.raises(ValueError, match='source hash'):
        LocalGradients.from_response_checkpoint(path)


def test_force_failure_leaves_recoverable_checkpoint(reference, tmp_path, monkeypatch):
    from electrolites.dlpno import local
    base = calculation(reference, None)
    path = tmp_path/'failed-force.npz'
    gradient = base.nuc_grad_method(level='mp2', response_checkpoint=path)
    def fail(*args, **kwargs):
        raise MemoryError('Injected final-contraction allocation failure')
    with monkeypatch.context() as patch:
        patch.setattr(local, 'contract_orbital_densities', fail)
        with pytest.raises(MemoryError, match='Injected'):
            gradient.kernel()
    assert path.exists() and Path(str(path)+'.json').exists()
    restarted = LocalGradients.from_response_checkpoint(path)
    actual = restarted.kernel()
    expected = base.nuc_grad_method(level='mp2').kernel()
    assert np.max(abs(actual-expected)) < 2e-11
    assert 'amplitude_adjoint' not in restarted.timings
    # An interrupted publication has no committed manifest and cannot load.
    Path(str(path)+'.json').unlink()
    with pytest.raises(FileNotFoundError):
        LocalGradients.from_response_checkpoint(path)
