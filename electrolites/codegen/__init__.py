"""The generators that write a kernel from the angular-momentum class alone.

These three ship with the package, and the CUDA they emit does not: it is
13 MB of straight-line device code that ``electrolites._gen`` produces on
first use, in about 0.7 s, and caches.  Keeping the generator rather than its
output is only possible because these three take **nothing** from GPU4PySCF's
source -- they emit the recurrences and the density contraction from
``(li,lj,lk,ll)`` (or ``(lij,lkl)``) themselves.

The generators in the repository's top-level ``codegen/`` directory are the
other kind: ``gen_kernels.py``, ``gen_k2_kernels.py`` and ``gen_j_kernels.py``
*lift* the integral arithmetic out of a GPU4PySCF source file and rewrite the
scaffolding around it, so they need a GPU4PySCF checkout that an installed
user does not have.  Their output stays in the repository.
"""
