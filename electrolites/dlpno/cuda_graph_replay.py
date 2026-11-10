"""Bounded CUDA Graph replay of a fixed-shape, pure-device adjoint action.

The private pool is reserved exclusively for the captured program. Its cached
temporary blocks must stay allocated until the graph is destroyed; another
calculation must never allocate from this pool after capture. Inputs and outputs
have stable addresses. Returned values are independent copies in the caller's
allocator, so a later replay cannot overwrite a caller's Krylov vector.
"""
import os
import time
import cupy as cp


def enabled():
    return os.environ.get('DLPNO_ADJOINT_CUDA_GRAPH', '0') == '1'


def budget():
    value = int(os.environ.get('DLPNO_ADJOINT_CUDA_GRAPH_MIB', '2048')) << 20
    if not (32 << 20) <= value <= (8 << 30):
        raise ValueError('CUDA Graph memory budget must be 32..8192 MiB')
    return value


class Replay:
    """One device/stream, immutable primal coefficients, mutable seed buffers."""
    def __init__(self, function, seeds, *, memory_bytes=None):
        start = time.perf_counter()
        self.device = cp.cuda.Device().id
        self.stream = cp.cuda.get_current_stream()
        if self.stream.ptr == 0:
            raise ValueError('CUDA Graph capture requires an explicit stream')
        self.pool = cp.cuda.MemoryPool()
        self.memory_bytes = budget() if memory_bytes is None else memory_bytes
        if self.memory_bytes <= 0:
            raise ValueError('CUDA Graph memory budget must be positive')
        self.pool.set_limit(size=self.memory_bytes)
        self.inputs = self.outputs = self.graph = None
        self.blas = None
        self.released = False
        self.statistics = dict(device=self.device, calls=0, released=False,
                               memory_limit_bytes=self.memory_bytes)
        try:
            from .cuda_graph_blas import CaptureBLAS
            from .reverse import _capture_backend
            self.blas = CaptureBLAS(self.stream)
            caller_allocator = cp.cuda.get_allocator()
            with cp.cuda.using_allocator(self.pool.malloc):
                self.inputs = [cp.array(x, copy=True, order='C') for x in seeds]
                self.blas.set_workspace(cp.empty(4 << 20, dtype=cp.uint8))
                # Compile lazy kernels, initialize BLAS handles, and reserve
                # the reusable temporary blocks before stream capture.
                token = _capture_backend.set(self.blas)
                try:
                    # Lazy backward plans can keep masks/coefficients until
                    # the outer source graph is released. Prime those in the
                    # caller's pool so replay destruction releases its own pool.
                    with cp.cuda.using_allocator(caller_allocator):
                        warm = function(self.inputs)
                        self.stream.synchronize()
                        del warm
                    for _ in range(2):
                        warm = function(self.inputs)
                        self.stream.synchronize()
                        del warm
                    self.stream.begin_capture()
                    try:
                        self.outputs = function(self.inputs)
                    except BaseException:
                        # End even an invalidated capture before releasing arrays.
                        try:
                            self.stream.end_capture()
                        except Exception:
                            pass
                        raise
                    self.graph = self.stream.end_capture()
                finally:
                    _capture_backend.reset(token)
                if not all(isinstance(x, cp.ndarray) and x.device.id == self.device
                           for x in self.outputs):
                    raise ValueError('CUDA Graph outputs must be local device arrays')
                self.graph.upload(stream=self.stream)
                self.stream.synchronize()
            self.statistics.update(build_seconds=time.perf_counter()-start,
                live_bytes=self.pool.used_bytes(), reserved_bytes=self.pool.total_bytes(),
                capture_live_bytes=self.pool.used_bytes(), capture_reserved_bytes=self.pool.total_bytes(),
                preparation_gemm_calls=self.blas.gemm_calls)
        except BaseException:
            self.release()
            raise

    def __call__(self, seeds):
        if self.released:
            raise RuntimeError('A released CUDA Graph cannot be reused')
        if cp.cuda.Device().id != self.device or cp.cuda.get_current_stream().ptr != self.stream.ptr:
            raise RuntimeError('CUDA Graph replay must use its owning device and stream')
        if len(seeds) != len(self.inputs):
            raise ValueError('CUDA Graph seed count differs')
        for source, target in zip(seeds, self.inputs):
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError('CUDA Graph seed shape or dtype differs')
            cp.copyto(target, source)
        self.graph.launch(stream=self.stream)
        # The surrounding shard/transpose synchronizes before giving results
        # to another stream/device. Never allocate from the captured pool here.
        result = [x.copy() for x in self.outputs]
        self.statistics['calls'] += 1
        return result

    def release(self):
        if self.released:
            return
        with cp.cuda.Device(self.device):
            self.stream.synchronize()
            self.graph = None
            self.outputs = self.inputs = None
            if self.blas is not None:
                self.blas.close()
                self.blas = None
            self.statistics['release_live_before_free'] = self.pool.used_bytes()
            self.pool.free_all_blocks()
            self.statistics.update(released=True, live_bytes=self.pool.used_bytes(),
                                   reserved_bytes=self.pool.total_bytes())
        self.released = True
