"""Where the kernel sources, launch tables and Rys tables live.

The CUDA sources are package data next to this file, so a wheel carries them
and nothing depends on where the repository was checked out.
"""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_DIR = os.path.join(PACKAGE_DIR, 'kernels')
CUDA_DIR = os.path.join(KERNEL_DIR, 'cuda')
DATA_DIR = os.path.join(KERNEL_DIR, 'data')

__all__ = ['PACKAGE_DIR', 'KERNEL_DIR', 'CUDA_DIR', 'DATA_DIR']
