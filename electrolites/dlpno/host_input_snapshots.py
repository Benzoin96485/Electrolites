"""Immutable host inputs shared by same-node spawned source workers.

Only small descriptors cross the process pipe. Read-only file mappings share
one page-cache copy and cannot be changed by later mutations of parent arrays.
The parent owns the arena until all workers have discarded their inputs.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import numpy as np


@dataclass(frozen=True)
class HostSnapshot:
    path: str
    offset: int
    shape: tuple
    nbytes: int

    def open(self):
        if self.offset < 0 or any(x < 0 for x in self.shape):
            raise ValueError('Invalid host snapshot bounds')
        if self.nbytes != 8 * int(np.prod(self.shape, dtype=object)):
            raise ValueError('Host snapshot byte count differs from its shape')
        if os.stat(self.path).st_size < self.offset + self.nbytes:
            raise ValueError('Truncated host snapshot file')
        if not self.nbytes:
            value = np.empty(self.shape, dtype=np.float64)
            value.flags.writeable = False
            return value
        return np.memmap(self.path, mode='r', dtype=np.float64,
                         offset=self.offset, shape=self.shape, order='C')


class HostSnapshotArena:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix='dlpno-host-inputs-')
        self.path = str(Path(self.directory.name) / 'inputs.bin')
        self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        self.end = self.bytes = self.count = 0
        self.closed = False

    def snapshot(self, value):
        if self.closed:
            raise RuntimeError('Closed host snapshot arena')
        value = np.asarray(value)
        if value.dtype != np.float64:
            raise TypeError('Host snapshots require float64')
        offset = (self.end + 63) // 64 * 64
        end = offset + value.nbytes
        os.ftruncate(self.fd, end)
        if value.nbytes:
            writer = np.memmap(self.path, mode='r+', dtype=np.float64,
                               offset=offset, shape=value.shape, order='C')
            try:
                np.copyto(writer, value)
            finally:
                writer._mmap.close()
            # MAP_SHARED writes are coherent with readers on this node.
            # There is no disk-persistence requirement, so do not msync.
        self.end = end
        self.bytes += value.nbytes
        self.count += 1
        return HostSnapshot(self.path, offset, value.shape, value.nbytes)

    def close(self):
        if self.closed:
            return
        self.closed = True
        os.close(self.fd)
        self.directory.cleanup()
