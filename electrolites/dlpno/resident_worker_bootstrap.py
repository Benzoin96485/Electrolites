"""Lightweight initializer: bind a spawned source worker before CUDA starts.

Importing this module performs no CUDA or scientific-library operations.
The parent-visible logical rank is retained separately from the child's
single visible CUDA device. Task arguments and results stay on the host.
"""
import os,sys


def initialize(selector,logical_device,budget):
    if 'gpu4pyscf.__config__' in sys.modules:
        raise RuntimeError('Worker CUDA isolation must precede GPU4PySCF imports')
    os.environ['CUDA_VISIBLE_DEVICES']=selector
    import pyscf.__config__ as config
    config.num_devices=1
    from .resident_shared_store_gpu import _initialize
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount()!=1:
        raise RuntimeError('Worker CUDA runtime started before device isolation')
    _initialize(0,budget,logical_device=logical_device)
