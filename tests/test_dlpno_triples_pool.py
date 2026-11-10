"""Triples coefficients must not pin the caller's cached temporary allocations."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from electrolites.dlpno.adjoint import TriplesTranspose


@pytest.mark.parametrize('coefficient_backend', ['numpy', 'cupy'])
def test_plan_isolates_caller_pool_and_releases_owned_storage(coefficient_backend):
    cp = pytest.importorskip('cupy')
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip('CUDA device required')
    rng = np.random.default_rng(7713)
    xp = np if coefficient_backend == 'numpy' else cp
    fock = np.array([[-.5, .04], [.04, -.6]])
    x = [rng.normal(size=(5, 3)), rng.normal(size=(5, 4))]
    couplings = [[((0, 1, 1), 1, 0, 1)], [((0, 0, 1), 0, 1, 0)]]
    state = SimpleNamespace(xp=xp, T3=[SimpleNamespace(shape=(n,n,n),size=n**3) for n in (3,4)],
        t_paos=[np.arange(5),np.arange(5)], X_tno=[xp.asarray(a) for a in x],
        orb=SimpleNamespace(F_lmo=xp.asarray(fock),S_pao=xp.eye(5)),
        opt={'F_CUT_T':0.}, _triples_couplings=lambda:couplings)
    diagonal = np.linspace(-4., -3., 91)
    cpu_state = SimpleNamespace(**vars(state))
    cpu_state.xp = np
    cpu_state.X_tno = x
    cpu_state.orb = SimpleNamespace(F_lmo=fock,S_pao=np.eye(5))
    reference = TriplesTranspose(cpu_state, diagonal, backend='numpy')
    caller = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(caller.malloc):
        cached = cp.empty(8<<20, dtype=cp.uint8)
        del cached
        anchor = cp.asarray([7.])
        before = caller.used_bytes()
        operator = TriplesTranspose(state, diagonal, backend='cupy')
        after = caller.used_bytes()
    # Only the denominator belongs to the caller; all batched coefficient
    # allocations must have independent ownership, including GPU inputs.
    assert after-before <= diagonal.nbytes+512
    owned = operator._coefficient_pool
    owned_bytes = owned.used_bytes()
    assert owned_bytes > 0 and operator.statistics['coefficient_allocator']=='dedicated_cupy_pool'
    error = 0.
    for _ in range(2):
        vector = rng.normal(size=91)
        expected = reference([vector])[0]
        actual = cp.asnumpy(operator([cp.asarray(vector)])[0])
        error = max(error,float(np.max(abs(actual-expected))))
    assert error < 2e-12
    operator.release()
    cp.cuda.get_current_stream().synchronize()
    assert operator.released and owned.used_bytes()==owned.total_bytes()==0
    assert caller.used_bytes()==before and float(cp.asnumpy(anchor)[0])==7.
    del anchor
    caller.free_all_blocks()
    assert caller.used_bytes()==caller.total_bytes()==0
    print('TRIPLES_POOL_REPORT '+json.dumps(dict(coefficient_backend=coefficient_backend,
          caller_live_delta_bytes=after-before,denominator_bytes=diagonal.nbytes,
          isolated_plan_bytes=owned_bytes,maximum_action_error=error,
          owned_storage_released=True,caller_anchor_preserved=True)))
