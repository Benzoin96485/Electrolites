"""The common-left replay workspace is reused without retaining device buffers."""
import gc
import weakref
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.triples_adjoint import transfer
from electrolites.dlpno import local_read_plans as plans


def test_overlap_replay_releases_all_device_buffers_and_failed_scopes(monkeypatch):
    cp = pytest.importorskip('cupy')
    from electrolites.dlpno.triples_overlap_adjoint_gpu import common_left_batch
    monkeypatch.setitem(plans.OPTIONS, 'overlap_replay', True)
    rng = np.random.default_rng(267)
    leaves = [ad.Variable(rng.normal(size=s)) for s in ((5,6),(5,3),(4,2),(3,4))]
    positions = (np.array([5,0,5,2]), np.array([1,4,0]))
    scopes, arrays = [], []
    fail = [False]
    def forward():
        values = common_left_batch(leaves[0], leaves[1], leaves[2:], positions)
        scope = ad.checkpoint_replay_workspace()
        if scope is not None:
            scopes.append(scope)
            assert len(scope) == 1
            arrays.extend(weakref.ref(x) for group in scope.values() for x in group)
            if fail[0]:raise RuntimeError('intentional overlap replay failure')
        return tuple(transfer(x,np) for x in values)
    outputs, _ = ad.checkpoint(forward)
    graph = ad.Linearization(outputs, leaves)
    for scale in (1., .37):
        graph([scale*np.ones(x.shape) for x in outputs])
        cp.cuda.Stream.null.synchronize();gc.collect()
        assert all(not scope for scope in scopes)
        assert all(ref() is None for ref in arrays)
        assert ad.checkpoint_replay_workspace() is None
    fail[0] = True
    with pytest.raises(RuntimeError,match='intentional overlap'):
        graph([np.ones(x.shape) for x in outputs])
    cp.cuda.Stream.null.synchronize();gc.collect()
    assert all(not scope for scope in scopes)
    assert ad.checkpoint_replay_workspace() is None
