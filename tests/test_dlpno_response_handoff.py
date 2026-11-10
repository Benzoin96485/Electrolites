"""Unused pooled storage must not obstruct direct force allocations."""
import json
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope='module', autouse=True)
def cuda():
    cp = pytest.importorskip('cupy')
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip('CUDA device required')


@pytest.fixture(scope='module')
def reference():
    from pyscf import gto, scf
    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57', basis='6-31g', verbose=0)
    return scf.RHF(mol).density_fit().run(conv_tol=1e-13, conv_tol_grad=1e-10)


def test_handoff_unblocks_direct_allocation_and_preserves_live_arrays(reference):
    import cupy as cp
    from electrolites.dlpno.gpu_response import GPUReferenceBackend
    backend = GPUReferenceBackend(SimpleNamespace(mf=reference))
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    # Keep both a view and an asynchronously computed value alive across the
    # boundary. No cached block is counted as released while it is in use.
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        anchor = cp.arange(65536, dtype=cp.float64)
        view = anchor[::2]
        computed = view*3.+1.
    stream.synchronize()
    expected = np.arange(0, 65536, 2, dtype=float)*3.+1.
    free, _ = cp.cuda.runtime.memGetInfo()
    assert free > 1 << 30, 'Pressure test needs a dedicated allocated GPU'
    reservation = min(2 << 30, int(free)//4)
    # GPU4PySCF bypasses the pool above 100 MiB. Explicitly create the
    # unused pool reservation whose interaction with direct malloc is tested.
    cache = pool.malloc(reservation)
    del cache
    free_with_cache, _ = cp.cuda.runtime.memGetInfo()
    requested = int(free_with_cache)+(16 << 20)
    pointer = None
    try:
        # runtime.malloc bypasses CuPy's pool, just as the failing library
        # allocation does. It must fail with the cache held, then succeed
        # after the handoff, without any deletion of the live operands.
        with pytest.raises(cp.cuda.runtime.CUDARuntimeError) as failure:
            pointer = cp.cuda.runtime.malloc(requested)
        assert failure.value.status == 2  # cudaErrorMemoryAllocation
        info = backend.prepare_force_contractions()
        assert info['released_pool_bytes'] >= reservation
        assert info['before']['pool_used_bytes'] == info['after']['pool_used_bytes']
        pointer = cp.cuda.runtime.malloc(requested)
        assert np.array_equal(cp.asnumpy(computed), expected)
        assert np.array_equal(cp.asnumpy(view), np.arange(0, 65536, 2, dtype=float))
        assert view.data.ptr == anchor.data.ptr
        print('RESPONSE_HANDOFF_PRESSURE_REPORT '+json.dumps(dict(
            handoff=info, direct_allocation_bytes=requested,
            allocation_failed_before=True, allocation_succeeded_after=True,
            live_arrays_preserved=True)), flush=True)
    finally:
        if pointer is not None:
            cp.cuda.runtime.free(pointer)
        del anchor, view, computed
        pool.free_all_blocks()


@pytest.mark.parametrize('scale', [1., 1e-6])
def test_handoff_preserves_reference_actions_and_signed_forces(reference, scale):
    import cupy as cp
    from electrolites.dlpno.gpu_response import GPUReferenceBackend
    from electrolites.dlpno.orbital_adjoint import _jk_quadratic_force
    backend = GPUReferenceBackend(SimpleNamespace(mf=reference))
    rng = np.random.default_rng(201)
    q, _ = np.linalg.qr(rng.normal(size=(reference.mol.nao, reference.mol.nao)))
    spectrum = np.linspace(-.7, .8, reference.mol.nao)
    spectrum[:3] = [1e-9, -1e-9, 0.]
    density = (q*spectrum)@q.T
    before_potential = backend.potential(density*scale)
    before_force = backend.quadratic_force(density*scale)
    before_hf = backend.hf_gradient()
    unused = cp.empty(32 << 20, dtype=cp.uint8)
    del unused
    info = backend.prepare_force_contractions()
    after_potential = backend.potential(density*scale)
    after_force = backend.quadratic_force(density*scale)
    after_hf = backend.hf_gradient()
    potential_error = float(np.max(abs(after_potential-before_potential)))/scale
    force_error = float(np.max(abs(after_force-before_force)))/scale**2
    hf_error = float(np.max(abs(after_hf-before_hf)))
    cpu_force = _jk_quadratic_force(reference.nuc_grad_method(), density)
    cpu_error = float(np.max(abs(after_force/scale**2-cpu_force)))
    assert potential_error < 2e-11 and force_error < 2e-11 and hf_error < 2e-11
    assert cpu_error < 2e-8
    assert info['released_pool_bytes'] >= 0
    assert info['before']['pool_used_bytes'] == info['after']['pool_used_bytes']
    print('RESPONSE_HANDOFF_FORCE_REPORT '+json.dumps(dict(
        scale=scale, potential_error=potential_error, force_error=force_error,
        hf_error=hf_error, cpu_force_error=cpu_error, handoff=info)), flush=True)
