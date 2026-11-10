"""Private FP64 cuBLAS handle used only while preparing a captured adjoint.

CuPy's installed BLAS wrapper rejects stream capture. The native cuBLAS API
supports capturing GEMM. This adapter retains CuPy arrays and its einsum
planner, replacing only GEMM submission, without patching installed modules.
"""
import ctypes as ct
import math
from types import SimpleNamespace
import numpy as np
import cupy as cp


class CaptureBLAS:
    def __init__(self, stream):
        self.lib = None
        for name in ('libcublas.so', 'libcublas.so.12', 'libcublas.so.13', 'libcublas.so.11'):
            try:
                self.lib = ct.CDLL(name)
                break
            except OSError:
                pass
        if self.lib is None:
            raise RuntimeError('CUDA Graph requires an available native cuBLAS library')
        self.handle = ct.c_void_p()
        self.workspace = None
        pointer, integer, stride = ct.c_void_p, ct.c_int, ct.c_longlong
        signatures = dict(cublasCreate_v2=[ct.POINTER(pointer)], cublasDestroy_v2=[pointer],
            cublasSetStream_v2=[pointer,pointer], cublasSetWorkspace_v2=[pointer,pointer,ct.c_size_t],
            cublasDgemmStridedBatched=[pointer,integer,integer,integer,integer,integer,
                pointer,pointer,integer,stride,pointer,integer,stride,pointer,pointer,integer,stride,integer])
        # CUDA exports cublasSetWorkspace without a version suffix.
        self.workspace_name = 'cublasSetWorkspace_v2' if hasattr(self.lib, 'cublasSetWorkspace_v2') else 'cublasSetWorkspace'
        signatures[self.workspace_name] = signatures.pop('cublasSetWorkspace_v2')
        for name, types in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = types, integer
        self.check(self.lib.cublasCreate_v2(ct.byref(self.handle)))
        try:
            self.check(self.lib.cublasSetStream_v2(self.handle, pointer(stream.ptr)))
        except BaseException:
            self.close()
            raise
        self.one, self.zero = ct.c_double(1.), ct.c_double(0.)
        self.gemm_calls = 0
        from cupy.linalg import _einsum
        from .reference_cpu import rebind
        class ArrayProxy:
            def __getattr__(_, name):
                return self.matmul if name == 'matmul' else getattr(cp,name)
        binary = rebind(_einsum.reduced_binary_einsum, cupy=ArrayProxy(),
                        _accelerator=SimpleNamespace(get_routine_accelerators=lambda: []))
        self.einsum = rebind(_einsum.einsum, reduced_binary_einsum=binary,
                            _try_use_cutensornet=lambda *args, **kwargs: None)

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f'Native capture cuBLAS returned status {status}')

    def set_workspace(self, memory):
        self.workspace = memory
        self.check(getattr(self.lib,self.workspace_name)(self.handle, ct.c_void_p(memory.data.ptr), memory.nbytes))

    def matmul(self, a, b):
        if not (isinstance(a,cp.ndarray) and isinstance(b,cp.ndarray)
                and a.dtype == b.dtype == np.float64):
            raise ValueError('Captured GEMM requires FP64 CuPy arrays')
        if a.ndim < 1 or b.ndim < 1:
            raise ValueError('Captured GEMM requires at least one-dimensional operands')
        av, bv = a.ndim == 1, b.ndim == 1
        if av:a=a[None,:]
        if bv:b=b[:,None]
        m,k=a.shape[-2:];kb,n=b.shape[-2:]
        if k != kb:raise ValueError('Captured GEMM contraction dimensions differ')
        shape=np.broadcast_shapes(a.shape[:-2],b.shape[:-2]);batch=math.prod(shape)
        if max(m,k,n,batch)>=2**31:raise ValueError('Captured GEMM exceeds 32-bit BLAS dimensions')
        a=cp.ascontiguousarray(cp.broadcast_to(a,shape+(m,k)))
        b=cp.ascontiguousarray(cp.broadcast_to(b,shape+(k,n)))
        result=cp.empty(shape+(m,n),dtype=cp.float64)
        if not batch or not m or not n or not k:
            result.fill(0.)
        else:
            # Row-major AB is column-major B^T A^T. Broadcasting and strides
            # have been normalized before the capture-safe BLAS submission.
            self.check(self.lib.cublasDgemmStridedBatched(self.handle,0,0,n,m,k,
                ct.byref(self.one),ct.c_void_p(b.data.ptr),n,k*n,
                ct.c_void_p(a.data.ptr),k,m*k,ct.byref(self.zero),
                ct.c_void_p(result.data.ptr),n,m*n,batch))
            self.gemm_calls += 1
        if av:result=result.squeeze(axis=-2)
        if bv:result=result.squeeze(axis=-1)
        return result

    def close(self):
        if self.handle and self.handle.value:
            self.check(self.lib.cublasDestroy_v2(self.handle))
            self.handle = ct.c_void_p()
        self.workspace = None
        # The rebound einsum planner holds its ArrayProxy closure.
        self.einsum = None
