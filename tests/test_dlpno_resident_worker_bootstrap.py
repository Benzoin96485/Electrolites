import os
import pytest


def test_lightweight_worker_bootstrap_import():
    # Its top-level code must not import CuPy or the heavy source module.
    import ast,inspect
    from electrolites.dlpno import resident_worker_bootstrap as module
    tree=ast.parse(inspect.getsource(module))
    imports=[a.name for n in tree.body if isinstance(n,ast.Import) for a in n.names]
    assert imports==['os','sys']


def test_single_visible_device_mapping(monkeypatch):
    import cupy as cp
    from electrolites.dlpno.resident_shared_store_gpu import SharedResidentPool,close_pools
    monkeypatch.setenv('DLPNO_RESIDENT_ISOLATE_DEVICES','1')
    monkeypatch.setenv('DLPNO_RESIDENT_PARALLEL_SHUTDOWN','1')
    selectors=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    pool=SharedResidentPool((1,0),executor='process',budget_bytes=64<<20)
    try:
        pool.prestart();result=pool.run('ready',[(0,()),(1,())])
        for logical,(_,info) in zip((1,0),result):
            assert info['device']==logical and info['local_device']==0
            assert info['cuda_visible_devices']==selectors[logical]
        assert pool.statistics['isolated_process_devices']
        release=pool.close();assert all(x['live_bytes']==x['reserved_bytes']==0 for x in release)
    finally:close_pools()
