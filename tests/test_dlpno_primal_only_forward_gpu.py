"""Forward record elision preserves canonical leaves and complete replay VJPs."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno import resident_shared_store_gpu as shared


def test_untracked_bank_preserves_original_inputs():
    import cupy as cp
    source = ad.Variable(cp.asarray([1., 2., 3.]))
    store = SimpleNamespace(inputs={9:source})
    bank = shared._Bank(store, (9,), recording=False)
    temporary = bank.inputs[0]
    assert not temporary._record.tracked and source._record.tracked
    assert temporary.v.data.ptr == source.v.data.ptr
    untracked = (temporary*temporary).sum()
    assert not untracked._record.tracked
    original = (source*source).sum()
    result = ad.Linearization([original],[source])([cp.asarray(1.)])[0]
    assert np.array_equal(cp.asnumpy(result), [2.,4.,6.])
    assert np.array_equal(cp.asnumpy(untracked.v), cp.asnumpy(original.v))
    assert shared._Bank(store, (9,)).inputs[0] is source


@pytest.mark.parametrize('kind', ['pno','triples','triples_energy'])
@pytest.mark.parametrize('primal_only', [False,True])
def test_forward_record_count_and_exact_complete_replay(monkeypatch,kind,primal_only):
    from test_dlpno_resident_shared_store_gpu import test_complete_resident_regions
    monkeypatch.setenv('DLPNO_RESIDENT_PRIMAL_FORWARD',str(int(primal_only)))
    monkeypatch.setenv('DLPNO_RESIDENT_CHECKPOINT_GRAPHS','1')
    monkeypatch.setenv('DLPNO_RESIDENT_INPUT_CACHE_MIB','0')
    monkeypatch.setenv('DLPNO_ASYNC_SOURCE_REVERSE','0')
    original_init, original_close = shared.SharedRegionPlan.__init__, shared.SharedResidentPool.close
    forwards, stores = [], []
    def initialize(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        forwards.extend(self.statistics['worker_graphs'])
    def close(self):
        rows=original_close(self)
        stores.extend(row['store'] for row in rows if 'store' in row)
        return rows
    monkeypatch.setattr(shared.SharedRegionPlan,'__init__',initialize)
    monkeypatch.setattr(shared.SharedResidentPool,'close',close)
    # The oracle includes coefficients, selected/repeated/consuming VJPs and
    # finite differences. Replays additionally enforce identical forward bytes.
    test_complete_resident_regions(kind,(0,1),'process',True)
    assert forwards and stores
    if primal_only:
        assert all(row['records']==0 for row in forwards)
        assert all(row['workspace_entries']==0 for row in forwards)
        assert sum(row['primal_only_forward_regions'] for row in stores)>0
        assert all(row['primal_only_forward_records']==0 for row in stores)
    else:
        assert all(row['records']>0 for row in forwards)
        assert all(row['primal_only_forward_regions']==0 for row in stores)
    assert sum(row['reverse_traced_records'] for row in stores)>0
    assert sum(row['verified_replays'] for row in stores)>0
    print('PRIMAL_FORWARD_REPORT '+json.dumps(dict(kind=kind,primal_only=primal_only,forward_records=sum(x['records'] for x in forwards),reverse_records=sum(x['reverse_traced_records'] for x in stores),verified_replays=sum(x['verified_replays'] for x in stores))))
