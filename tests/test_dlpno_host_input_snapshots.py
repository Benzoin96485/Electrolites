import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pytest
from electrolites.dlpno.host_input_snapshots import HostSnapshot, HostSnapshotArena


def inspect_snapshot(snapshot):
    value=snapshot.open()
    try:
        value.flat[0]=123.
    except ValueError:
        readonly=True
    else:
        readonly=False
    stat=os.stat(snapshot.path)
    return float(value.sum()),readonly,stat.st_dev,stat.st_ino


@pytest.mark.parametrize('kind',['strided','scalar','empty'])
def test_snapshot_mutation_and_lifetime(kind):
    raw={'strided':np.arange(126.,dtype=np.float64).reshape(7,18)[:,::2],
         'scalar':np.array(3.25),'empty':np.empty((0,7))}[kind]
    expected=raw.copy();arena=HostSnapshotArena();path=arena.path
    try:
        snapshot=arena.snapshot(raw);view=snapshot.open();raw[...] += 17.
        np.testing.assert_array_equal(view,expected)
        assert not view.flags.writeable
        with pytest.raises(ValueError):view[...] = 0.
        arena.close();assert not os.path.exists(path)
        np.testing.assert_array_equal(view,expected)
        with pytest.raises(FileNotFoundError):snapshot.open()
    finally:arena.close()


def test_spawned_readers_share_one_immutable_file():
    raw=np.arange(4096.,dtype=np.float64).reshape(64,64)
    arena=HostSnapshotArena()
    try:
        first=arena.snapshot(raw);second=arena.snapshot(raw[::-1,::2])
        expected=[float(raw.sum()),float(raw[::-1,::2].sum())]
        raw.fill(-123.)
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            rows=list(pool.map(inspect_snapshot,[first,second]))
        assert [x[0] for x in rows]==expected and all(x[1] for x in rows)
        assert rows[0][2:]==rows[1][2:]
        assert arena.count==2 and arena.bytes==first.nbytes+second.nbytes
    finally:arena.close()


def test_bad_snapshot_metadata_and_closed_arena():
    arena=HostSnapshotArena()
    try:
        with pytest.raises(TypeError):arena.snapshot(np.ones(4,dtype=np.float32))
        good=arena.snapshot(np.ones(4))
        with pytest.raises(ValueError,match='byte count'):
            HostSnapshot(good.path,good.offset,(8,),good.nbytes).open()
        with pytest.raises(ValueError,match='Truncated'):
            HostSnapshot(good.path,good.offset+64,good.shape,good.nbytes).open()
    finally:arena.close()
    with pytest.raises(RuntimeError):arena.snapshot(np.ones(4))
