"""
Drop-in replacement for GPU4PySCF's second-derivative ERI kernels.

Catalogue entry **B3**, rank 9 in `instructions/06`'s §5 list.
`hessian/rhf.py:_partial_ejk_ip2` drives two CUDA entry points --
``RYS_per_atom_jk_ip2_type12`` and ``RYS_per_atom_jk_ip2_type3`` -- and between
them they are **98.9 %** of that stage, 8.6 s of a 42-atom B3LYP/def2-SVP
Hessian (`benchmarks/ejk_ip2_probe.py`).

The module does two independent things, each with its own switch, because one
of them needs a CUDA compiler and the other does not.

**(a) The launch configuration** (`FASTEJK2_NO_SCHEME=1` to disable).
`hessian/rhf.py:_ip2_quartets_scheme` and `_ip2_type3_quartets_scheme` derive
the block shape from the shared-memory budget alone -- 256 threads, as many
shell quartets as shared memory allows, the rest to ``gout_stride`` -- and
nothing was measured to arrive at that.  Halving the quartet count, which moves
those threads to ``gout_stride`` where the contraction's ``nfij*nfkl`` loop is
spread, is **1.13x** on the whole stage
(`benchmarks/ejk_ip2_sweep.py`, penicillin V/def2-SVP: 7.43 s -> 6.56 s).  It
is pure Python and always available.

**(b) The kernels themselves.**  Their two symbols are replaced on the
``libvhf_rys`` handle by the same functions compiled out of
`electrolites/kernels/cuda/myejk2/`, which `codegen/gen_ejk_ip2.py` produces
from GPU4PySCF's own sources with two changes and no others:

* **P1.**  No double-precision division anywhere in the primitive or root
  loops.  Upstream evaluates ``aj/aij``, ``al/akl``, ``aij*akl/(aij+akl)`` and
  ``cicj*ckcl/(aij*akl*sqrt(aij+akl))`` per primitive quartet and then
  ``rt/(aij+akl)``, ``.5/aij`` and ``.5/akl`` **once per Rys root** -- up to
  eighteen fp64 divisions and a ``sqrt`` per primitive quartet, each about
  twenty instructions on sm_80.  ``1/aij`` and ``1/akl`` are formed once where
  the exponents are read and one ``rsqrt(aij+akl)`` supplies both
  ``1/(aij+akl)`` and ``1/sqrt(aij+akl)``: two divisions and one ``rsqrt`` per
  primitive quartet, none per root.
* **P3, which is measured not to pay and is therefore not in the shipped
  sources** (``codegen/gen_ejk_ip2.py --p3`` writes it).  A
  ``__launch_bounds__(256, 2)`` -- a 128-register cap and two blocks per SM,
  where upstream compiles at the 255-register ceiling and one block -- is
  **0.58x**.  The catalogue's B3 entry expects P3 to be worth more here than on
  the exchange build because the register pressure is worse; the pressure is
  worse and that is exactly why capping it loses, since the spills cost more
  than the second block buys.

  The cap has to be written at the block size the driver launches, which is
  the round's one real surprise.  ``__launch_bounds__(512, 1)`` is the same
  128-register cap with a 512-thread maximum, and it computes numbers **9x to
  37x wrong** -- with ``vhf.cuh``'s ``adjust_threads`` live *or* disabled, so
  with the launch geometry identical at 256 threads either way.  ``(256, 2)``
  agrees with upstream to 1e-13.  The cause is not identified.
  `codegen/gen_ejk_ip2.py` records it, caps at 256, and also switches off
  ``adjust_threads`` -- upstream's own register-driven thread doubling, dead
  code without a cap -- so that nothing can change the block shape out from
  under the bound.

Everything above the kernels is GPU4PySCF's, unaltered: the Python driver, the
task generation, the screening, the launch schemes, the per-atom contraction
and its symmetry factors.  That is what makes the oracle exact -- the same
driver, the same task lists, one arithmetic substitution.

**Why this one is not NVRTC.**  These kernels are split over three translation
units and take kernel pointers (``adjust_threads``), so they need relocatable
device code, which NVRTC cannot produce.  They are built with ``nvcc`` on first
use and cached under ``~/.cache/electrolites/sm_<arch>/``, exactly as `fastk`'s
general exchange kernel is.  If ``nvcc`` is not installed the module says so
once and stands down.

**Version gate.**  Unlike every other module here, this one binds to
GPU4PySCF's C *struct layouts* (``RysIntEnvVars``, ``BoundsInfo``,
``JKEnergy``), not to a Python signature, so `check_targets` cannot protect it.
``BoundsInfo`` gained per-group ``q_cond``/``s_cond`` between 1.7.0 and 1.8.0.
The module therefore refuses to patch unless the installed release is one whose
layout the vendored headers match, and says which.

See README_fastejk2.md for what it is worth and for what B3 turns out to be
about, which is not what the entry says.
"""
import ctypes
import functools
import os
import sys

from gpu4pyscf.scf import jk as JK
from gpu4pyscf.hessian import rhf as HRHF          # noqa: F401  (binds first)

from . import _nvcc

#: releases whose gvhf-rys struct layouts the vendored headers match.  The
#: headers are copied from 1.8.1; 1.8.0 shipped the same vhf.cuh.
SUPPORTED = ('1.8.0', '1.8.1')

_OFF = bool(os.environ.get('FASTEJK2_OFF'))
_NO_SCHEME = bool(os.environ.get('FASTEJK2_NO_SCHEME'))
#: The kernel half was opt-in while it was P1 alone (1.02x, README_fastejk2.md
#: 2).  Since catalogue B3's round (docs/KERNEL_TARGETS_1.8.1.md §13.20) the
#: library also carries per-class instances of the general kernels and a
#: cached device query, worth 1.10x more on top of the launch scheme (1.26x
#: and 1.58x together on penicillin V and a Ni complex), so it is **on by
#: default**; ``FASTEJK2_NO_KERNELS=1`` keeps GPU4PySCF's kernels.  Without
#: nvcc the module still stands down to the scheme half.
_KERNELS = not bool(os.environ.get('FASTEJK2_NO_KERNELS'))
#: quartets-per-block halvings; 1 is what benchmarks/ejk_ip2_sweep.py measured
_SHIFT = int(os.environ.get('FASTEJK2_SHIFT', 1))
_PATCHED = False
_SCHEME_PATCHED = False


@functools.lru_cache(maxsize=1)
def _library():
    """Our copy of the ip2 kernels, or None if it cannot be built."""
    so = _nvcc.ejk_ip2_library()
    if so is None:
        return None
    try:
        lib = ctypes.CDLL(so)
    except OSError:
        return None
    for name in ('MYEJK2_per_atom_jk_ip2_type12',
                 'MYEJK2_per_atom_jk_ip2_type3'):
        getattr(lib, name).restype = ctypes.c_int
    return lib


def _version_ok():
    import gpu4pyscf
    return gpu4pyscf.__version__ in SUPPORTED


def _swept(orig):
    """``orig``'s scheme with the quartet count halved ``_SHIFT`` times.

    ``orig`` returns the largest power of two whose shared memory fits at 256
    threads; the shared-memory bound scales with the quartet count alone, so
    taking fewer quartets and more ``gout_stride`` lanes always fits.
    """
    def scheme(mol, l_ctr_pattern, shm_size=JK.SHM_SIZE):
        n, _gs = orig(mol, l_ctr_pattern, shm_size)
        n = max(1, n >> _SHIFT)
        return n, max(1, 256 // n)
    return scheme


def apply_scheme_patch():
    """The launch-configuration half; needs no compiler."""
    global _SCHEME_PATCHED
    if _SCHEME_PATCHED or _OFF or _NO_SCHEME or _SHIFT == 0:
        return _SCHEME_PATCHED
    for name in ('_ip2_quartets_scheme', '_ip2_type3_quartets_scheme'):
        orig = getattr(HRHF, name)
        setattr(HRHF, '_ejk2_orig_' + name, orig)
        setattr(HRHF, name, _swept(orig))
    _SCHEME_PATCHED = True
    return True


def unpatch_scheme():
    global _SCHEME_PATCHED
    for name in ('_ip2_quartets_scheme', '_ip2_type3_quartets_scheme'):
        orig = getattr(HRHF, '_ejk2_orig_' + name, None)
        if orig is not None:
            setattr(HRHF, name, orig)
    _SCHEME_PATCHED = False


def apply_patch(force=False):
    """Bind our two kernels over GPU4PySCF's on the ctypes handle.

    On unless ``FASTEJK2_NO_KERNELS`` is set (``force`` overrides it); ``force`` is
    what `tests/test_ejk_ip2.py` and `benchmarks/ejk_ip2_bench.py` use, since
    a switch that hides the thing under test would hide the measurement too.

    ``hessian/rhf.py`` looks the two functions up on ``libvhf_rys`` at call
    time (``kern1 = libvhf_rys.RYS_per_atom_jk_ip2_type12`` inside ``proc``),
    so replacing the attributes is enough and no Python driver is duplicated.
    The originals are kept for the ablation and for the tests' oracle.
    """
    global _PATCHED
    if _PATCHED or _OFF or not (_KERNELS or force):
        return _PATCHED
    import gpu4pyscf
    if not _version_ok():
        print(f'[electrolites] fastejk2 stands down: it binds to GPU4PySCF\'s '
              f'CUDA struct layouts and the vendored headers are from '
              f'{"/".join(SUPPORTED)}, not {gpu4pyscf.__version__}',
              file=sys.stderr, flush=True)
        return False
    lib = _library()
    if lib is None:
        return False
    for name in ('RYS_per_atom_jk_ip2_type12', 'RYS_per_atom_jk_ip2_type3'):
        if not hasattr(JK.libvhf_rys, '_ejk2_orig_' + name):
            setattr(JK.libvhf_rys, '_ejk2_orig_' + name,
                    getattr(JK.libvhf_rys, name))
        setattr(JK.libvhf_rys, name, getattr(lib, 'MYEJK2' + name[3:]))
    _PATCHED = True
    return True


def unpatch():
    """Put GPU4PySCF's kernels back.  For the tests' oracle only.

    This is the one place in the repository with an unpatch, and it exists
    because the oracle for these kernels is the *same driver* running the other
    kernels: there is nothing else to compare against, and running the driver
    twice in one process is the only way to hold the task lists fixed.
    """
    global _PATCHED
    for name in ('RYS_per_atom_jk_ip2_type12', 'RYS_per_atom_jk_ip2_type3'):
        orig = getattr(JK.libvhf_rys, '_ejk2_orig_' + name, None)
        if orig is not None:
            setattr(JK.libvhf_rys, name, orig)
    _PATCHED = False


def patched():
    return _PATCHED, _SCHEME_PATCHED


apply_scheme_patch()
apply_patch()
