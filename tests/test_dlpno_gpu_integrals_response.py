"""GPU integral blocks and signed response densities against CPU oracles."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope='module', autouse=True)
def cuda():
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('A CUDA device is required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('A CUDA device is required')


@pytest.fixture(scope='module')
def reference():
    from pyscf import gto, scf
    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57', basis='def2-svp', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
    mf.kernel()
    assert mf.converged
    return mf


def test_streamed_independent_hf_energy_and_nuclear_difference(reference):
    from pyscf import scf
    from benchmarks.dlpno_reference_energy import reference_energy
    original = reference.mo_coeff.copy()
    cpu = reference_energy(reference, 'cpu')
    gpu = reference_energy(reference, 'legacy-gpu')
    assert abs(cpu['energy']-reference.e_tot) < 1e-10
    assert abs(gpu['energy']-cpu['energy']) < 1e-10
    assert not gpu['full_ao_three_center_allocated']
    assert not gpu['scf_recomputed']
    assert np.array_equal(original, reference.mo_coeff)
    atom, axis, step = 1, 2, 1e-4
    energies = []
    for sign in (1, -1):
        coordinates = reference.mol.atom_coords()
        coordinates[atom, axis] += sign*step
        mol = reference.mol.copy().set_geom_(coordinates, unit='Bohr')
        mf = scf.RHF(mol).density_fit(auxbasis=reference.with_df.auxbasis)
        mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
        mf.kernel(dm0=reference.make_rdm1())
        assert mf.converged
        energies.append(reference_energy(mf, 'legacy-gpu')['energy'])
    numerical = (energies[0]-energies[1])/(2*step)
    assert abs(numerical-reference.nuc_grad_method().kernel()[atom, axis]) < 1e-7


def test_reference_metric_application_reuses_owned_tensor():
    import cupy as cp
    from electrolites.dlpno.gpu_response import _MetricArrayModule
    rng = np.random.default_rng(814)
    matrix = cp.asarray(rng.normal(size=(37, 37)))
    tensor = cp.asarray(rng.normal(size=(37, 13, 13)))
    expected = cp.einsum('uv,vij->uij', matrix, tensor)
    info = dict(metric_streaming_calls=0, metric_streaming_blocks=0,
                largest_metric_workspace_bytes=0)
    budget = 2*37*8*7
    pointer = tensor.data.ptr
    actual = _MetricArrayModule(budget, info).einsum('uv,vij->uij', matrix, tensor)
    assert actual.data.ptr == pointer
    assert float(cp.max(cp.abs(actual-expected))) < 1e-12
    assert info['metric_streaming_calls'] == 1
    assert info['metric_streaming_blocks'] > 1
    assert info['largest_metric_workspace_bytes'] <= budget


def test_gpu_pao_projector_matches_cpu_overlap_ordering():
    import cupy as cp
    from electrolites.dlpno.linalg import orthonormal_domain
    from electrolites.dlpno.setup import orthocanonicalize, orthogonalization_trace
    rng = np.random.default_rng(701)
    vectors = rng.normal(size=(16, 9))
    overlap = vectors@vectors.T+1e-7*np.eye(16)
    fock = rng.normal(size=(16, 16))
    fock = (fock+fock.T)*.5
    cpu, _ = orthocanonicalize(overlap, fock, 1e-5)
    selections = []
    token = orthogonalization_trace.set(selections)
    try:
        gpu, keep = orthonormal_domain(cp.asarray(overlap[None]), np.array([16]), 1e-5)
    finally:
        orthogonalization_trace.reset(token)
    assert selections[0][0] == 16
    assert selections[0][2] == int(keep.sum()) == cpu.shape[1]
    gpu = cp.asnumpy(gpu[0][:, keep[0]])
    assert gpu.shape == cpu.shape
    # The selected approximate subspace, not just its rank, must agree.
    assert float(np.max(abs(gpu@gpu.T-cpu@cpu.T))) < 1e-10


def test_gpu_ill_conditioned_pao_basis_has_no_discarded_rows():
    import cupy as cp
    from electrolites.dlpno.linalg import orthonormal_domain
    from electrolites.dlpno.setup import orthogonalization_trace
    rng = np.random.default_rng(914)
    left, _ = np.linalg.qr(rng.normal(size=(176, 163)))
    spectrum = np.geomspace(1e-9, 1., 163)
    overlap = (left*spectrum)@left.T
    selections = []
    token = orthogonalization_trace.set(selections)
    try:
        basis, keep = orthonormal_domain(cp.asarray(overlap[None]), np.array([176]), 1e-10)
    finally:
        orthogonalization_trace.reset(token)
    selected = np.asarray(selections[0][1])
    assert 0 < len(selected) < len(overlap)
    actual = cp.asnumpy(basis[0][:, keep[0]])
    outside = np.setdiff1d(np.arange(len(overlap)), selected)
    # These rows are excluded by the pivot selection, independently of
    # eigensolver roundoff or the metric condition number.
    assert np.count_nonzero(actual[outside]) == 0
    assert np.max(abs(actual.T@overlap@actual-np.eye(actual.shape[1]))) < 2e-5


def test_gpu_scf_converted_reference_checkpoint(reference, tmp_path):
    from benchmarks.dlpno_adjoint_scaling import load_primal, save_primal
    gpu = reference.to_gpu()
    gpu.kernel()
    cpu = gpu.to_cpu()
    cpu.get_fock()
    optimizer = getattr(cpu.with_df, 'intopt', None)
    base = SimpleNamespace(mf=cpu, e={'corr': -.1})
    report = dict(code_sha256={}, settings={})
    path = tmp_path/'energy.pkl'
    save_primal(path, base, report)
    restored, _ = load_primal(path, report)
    assert getattr(cpu.with_df, 'intopt', None) is optimizer
    restored.mf.with_df.build()
    assert np.array_equal(restored.mf.mo_coeff, cpu.mo_coeff)
    assert abs(restored.mf.energy_tot()-cpu.energy_tot()) < 1e-11


@pytest.mark.parametrize('intor,comp', [('int3c2e', 1), ('int3c2e_ip1', 3), ('int3c2e_ip2', 3)])
@pytest.mark.parametrize('atoms', [[1], [0], [2, 0]])
def test_gpu_local_integral_values_and_derivatives(reference, intor, comp, atoms):
    from pyscf import df
    from electrolites.dlpno.gpu_integrals import GPULocalRIBlocks
    from electrolites.dlpno.local_integrals import compact_molecule
    aux = df.addons.make_auxmol(reference.mol, df.make_auxbasis(reference.mol, mp2fit=True))
    compact = compact_molecule(aux, atoms)
    # Exercise the integral kernel directly, including spherical AO/aux ordering.
    provider = object.__new__(GPULocalRIBlocks)
    provider.counts = {}
    actual = provider._evaluate_raw(reference.mol, compact, intor, comp)
    expected = df.incore.aux_e2(reference.mol, compact, intor=intor, aosym='s1', comp=comp)
    assert actual.shape == expected.shape
    assert float(np.max(abs(actual-expected))) < 2e-11


@pytest.mark.parametrize('fitted', [False, True])
def test_gpu_reference_preserves_signed_small_density(reference, fitted):
    from pyscf import scf
    from electrolites.dlpno.gpu_response import GPUReferenceBackend
    from electrolites.dlpno.orbital_adjoint import _jk_quadratic_force
    mf = reference if fitted else scf.RHF(reference.mol).run(conv_tol=1e-13)
    backend = GPUReferenceBackend(SimpleNamespace(mf=mf), metric_block_bytes=64 << 10)
    rng = np.random.default_rng(592)
    vectors, _ = np.linalg.qr(rng.normal(size=(mf.mol.nao, mf.mol.nao)))
    spectrum = np.linspace(-.7, .8, mf.mol.nao)
    spectrum[:3] = [1e-9, -1e-9, 0.]
    density = (vectors*spectrum)@vectors.T
    expected_potential = mf.get_veff(mf.mol, density)
    expected_force = _jk_quadratic_force(mf.nuc_grad_method(), density)
    for scale in (1., 1e-6):
        assert np.max(abs(backend.potential(density*scale)/scale-expected_potential)) < 2e-9
        assert np.max(abs(backend.quadratic_force(density*scale)/scale**2-expected_force)) < 2e-8
    if fitted:
        assert backend.info['metric_streaming_calls'] == 2
        assert backend.info['metric_streaming_blocks'] > 2
        assert backend.info['largest_metric_workspace_bytes'] <= 64 << 10
        assert backend.info['auxiliary_streaming_calls'] == 2
        assert backend.info['auxiliary_streaming_blocks'] > 2
        assert backend.info['largest_auxiliary_workspace_bytes'] <= 64 << 10


@pytest.mark.parametrize('triples', ['t0', 't1'])
@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
@pytest.mark.parametrize('ccsd_adjoint_backend', [None, 'cupy'])
def test_gpu_integrals_and_response_in_complete_local_gradient(triples, backend, ccsd_adjoint_backend, tmp_path):
    from pyscf import gto, scf
    from electrolites.dlpno.local import LocalDLPNO
    from electrolites.dlpno.local_gpu import LocalGPUDLPNO
    from benchmarks.dlpno_gradient_check import STRESS
    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-13, conv_tol_grad=1e-10)
    options = dict(triples=triples, verbose=0, **STRESS, E_CONVERGENCE=1e-11,
                   R_CONVERGENCE=1e-10, MAXITER=150, T_CUT_ITER=0.)
    cpu = LocalDLPNO(mf, **options)
    cpu.kernel()
    expected = cpu.nuc_grad_method(checkpoint_triples=False, checkpoint_integrals=False).kernel()
    gpu = LocalGPUDLPNO(mf, c_lmo=cpu.orb.C_lmo, ri_backend='gpu', **options)
    gpu.kernel()
    assert abs(gpu.e['corr']-cpu.e['corr']) < 1e-9
    grad = gpu.nuc_grad_method(backend=backend, response_backend='gpu',
                              ccsd_adjoint_backend=ccsd_adjoint_backend,
                              triples_backend='cupy' if triples == 't1' else None,
                              ccsd_primal_backend='gpu', checkpoint_ccsd=True,
                              triples_primal_backend='gpu', checkpoint_pnos=True,
                              response_checkpoint=tmp_path/'response.npz')
    actual = grad.kernel()
    gradient_error = float(np.max(abs(actual-expected)))
    assert gradient_error < 2e-8
    if ccsd_adjoint_backend == 'cupy':
        assert grad.diagnostics['amplitudes']['ccsd']['operator_kind'] == 'batched_ccsd_transpose'
    from electrolites.dlpno.local import LocalGradients
    for restarted_backend in ('cpu', 'gpu'):
        restarted = LocalGradients.from_response_checkpoint(
            tmp_path/'response.npz', response_backend=restarted_backend)
        assert np.max(abs(restarted.kernel()-actual)) < 2e-8
        assert set(restarted.timings) == {'orbital_contraction', 'hf_gradient'}
    assert grad.diagnostics['amplitudes']['ccsd_primal']['backend'] == 'gpu'
    assert grad.diagnostics['amplitudes']['ccsd_pair_checkpoints']['count'] > 0
    assert grad.diagnostics['amplitudes']['mp2_pno_checkpoints']['count'] > 0
    assert grad.diagnostics['amplitudes']['sc_mp2_checkpoints']['count'] > 0
    assert grad.diagnostics['amplitudes']['tno_subspace_checkpoints']['count'] > 0
    assert grad.diagnostics['amplitudes']['tno_subspace_checkpoints']['discarded_value_bytes'] > 0
    if triples == 't1':
        assert grad.diagnostics['amplitudes']['triples_primal']['backend'] == 'gpu'
    assert grad.diagnostics['gpu_reference']['quadratic_force_calls'] == 3
    assert grad.diagnostics['gpu_reference']['fock_actions'] > 0
    assert grad.diagnostics['integrals']['integral_blocks']['gpu_backtransform_atoms'] == mol.natm


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('layout', ['contiguous', 'strided'])
@pytest.mark.parametrize('with_output', [False, True])
def test_auxiliary_density_streamed_pair_columns(transpose, layout, with_output):
    import cupy as cp
    from electrolites.dlpno.gpu_response import _AuxiliaryDensityContraction
    rng = np.random.default_rng(882)
    a = cp.asarray(rng.normal(size=(11, 7, 5)))
    b = cp.asarray(rng.normal(size=(13, 5, 7) if transpose else (13, 7, 5)))
    if layout == 'strided':
        a = a[:, ::-1, :]
        b = b[:, :, ::-1]
    original_a, original_b = cp.asnumpy(a), cp.asnumpy(b)
    spec = 'rij,sji->rs' if transpose else 'rij,sij->rs'
    alpha, beta = -.37, .63
    initial = rng.normal(size=(11, 13))
    out = cp.array(initial, order='F') if with_output else None
    expected = alpha*np.einsum(spec, original_a, original_b)
    if with_output:
        expected += beta*initial
    info = dict(auxiliary_streaming_calls=0, auxiliary_streaming_blocks=0,
                largest_auxiliary_workspace_bytes=0)
    budget = 8*(11+13)*3
    def forbidden(*args, **kwargs):
        raise AssertionError('The bounded pattern must not call cuTENSOR')
    actual = _AuxiliaryDensityContraction(forbidden, budget, info)(
        spec, a, b, alpha=alpha, beta=beta, out=out)
    if with_output:
        assert actual.data.ptr == out.data.ptr
    assert np.max(abs(cp.asnumpy(actual)-expected)) < 2e-13
    assert np.array_equal(cp.asnumpy(a), original_a)
    assert np.array_equal(cp.asnumpy(b), original_b)
    assert info['auxiliary_streaming_calls'] == 1
    assert info['auxiliary_streaming_blocks'] == 12
    assert info['largest_auxiliary_workspace_bytes'] <= budget


def test_auxiliary_density_empty_pair_and_zero_alpha():
    import cupy as cp
    from electrolites.dlpno.gpu_response import _AuxiliaryDensityContraction
    for shape, alpha in [((4, 0, 3), 1.), ((4, 2, 3), 0.)]:
        a, b = cp.zeros(shape), cp.zeros((5, *shape[1:]))
        out = cp.full((4, 5), 7.)
        info = dict(auxiliary_streaming_calls=0, auxiliary_streaming_blocks=0,
                    largest_auxiliary_workspace_bytes=0)
        result = _AuxiliaryDensityContraction(None, 1024, info)(
            'rij,sij->rs', a, b, alpha=alpha, beta=.5, out=out)
        assert np.array_equal(cp.asnumpy(result), np.full((4, 5), 3.5))
        assert info['auxiliary_streaming_blocks'] == 0
