"""Primal compaction releases split blocks and preserves reusable energies."""
import json
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope='module', autouse=True)
def cuda():
    cp = pytest.importorskip('cupy')
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip('CUDA device required')


def test_split_pool_compaction_preserves_values_aliases_and_allocator():
    import cupy as cp
    from electrolites.dlpno.primal_storage import prepare_gradient_storage
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    baseline_reserved = pool.total_bytes()
    caller_allocator = cp.cuda.get_allocator()
    reservation = pool.malloc(2 << 30)
    del reservation

    def build():
        with cp.cuda.using_allocator(pool.malloc):
            parent = cp.arange(8192, dtype=cp.float64).reshape(128, 64)
            view = parent[::3, ::-2]
            empty = parent[:0, :3]
            host = np.array([.2, .4])
            return SimpleNamespace(X_tno=[view, view, empty],
                e_tno=(cp.arange(view.shape[1], dtype=cp.float64),),
                S_pao_d=cp.eye(8), F_pao_d=cp.eye(8)*.5,
                T=[host], _pt={'scratch': cp.ones(4096)},
                _pt_src=[host], _xt=([], cp.ones(8192), 8))

    base = build()
    before = [cp.asnumpy(x) for x in base.X_tno]
    energies = cp.asnumpy(base.e_tno[0])
    host = base.T[0]
    pool.free_all_blocks()
    pinned = pool.total_bytes()
    assert pinned >= baseline_reserved+(2 << 30)
    result = prepare_gradient_storage(base)
    assert cp.cuda.get_allocator() is caller_allocator
    assert all(np.array_equal(a, cp.asnumpy(b)) for a, b in zip(before, base.X_tno))
    assert np.array_equal(energies, cp.asnumpy(base.e_tno[0]))
    assert np.array_equal(cp.asnumpy(base.S_pao_d), np.eye(8))
    assert np.array_equal(cp.asnumpy(base.F_pao_d), np.eye(8)*.5)
    assert base.X_tno[0] is base.X_tno[1]
    assert base.X_tno[2].shape == (0, 3) and isinstance(base.e_tno, tuple)
    assert base.T[0] is host
    assert result['released_pool_bytes'] >= 2 << 30
    assert pool.total_bytes() <= baseline_reserved+(1 << 20)
    assert not any(hasattr(base, name) for name in ('_pt', '_pt_src', '_xt'))
    repeated = prepare_gradient_storage(base)
    assert all(np.array_equal(a, cp.asnumpy(b)) for a, b in zip(before, base.X_tno))
    assert base.X_tno[0] is base.X_tno[1]
    print('PRIMAL_STORAGE_PRESSURE_REPORT '+json.dumps(dict(
        initial_reserved_bytes=pinned, first=result, repeated=repeated,
        exact_values=True, exact_object_alias_preserved=True,
        strided_and_empty_values_preserved=True, host_input_preserved=True,
        caller_allocator_preserved=True)), flush=True)


@pytest.mark.parametrize('triples', ['t0', 't1'])
def test_primal_compaction_keeps_energy_reusable(triples):
    import cupy as cp
    from pyscf import gto, scf
    from benchmarks.dlpno_gradient_check import STRESS
    from electrolites.dlpno.local_gpu import LocalGPUDLPNO
    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-13, conv_tol_grad=1e-10)
    base = LocalGPUDLPNO(mf, triples=triples, ri_backend='gpu', verbose=0,
        **STRESS, E_CONVERGENCE=1e-11, R_CONVERGENCE=1e-10, MAXITER=150, T_CUT_ITER=0.)
    base.kernel()
    # The same T0 operation before/after exercises lazy table reconstruction,
    # stored TNO views and all three saved cubic tensors on actual chemistry.
    before_energy = base.lccsd_t0(save=True)
    fields = ('X_tno', 'e_tno', 'W3', 'V3', 'T3')
    original = {name: [cp.asnumpy(x) for x in getattr(base, name)] for name in fields}
    base._xtab()
    info = base.prepare_gradient_storage()
    assert all(np.array_equal(a, cp.asnumpy(b)) for name in fields
               for a, b in zip(original[name], getattr(base, name)))
    after_energy = base.lccsd_t0(save=True)
    table, padded_rows = base._xtab()
    assert table.ndim == 3 and padded_rows > 0
    error = abs(after_energy-before_energy)
    assert error < 1e-10
    print('PRIMAL_STORAGE_ENERGY_REPORT '+json.dumps(dict(
        triples=triples, energy_error=error, primals_bitwise_preserved=True,
        energy_tables_rebuilt=True, storage=info)), flush=True)
