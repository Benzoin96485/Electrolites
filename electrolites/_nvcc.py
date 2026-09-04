"""Build the one library that is not JIT-compiled, on first use.

Almost every kernel here is compiled at run time by NVRTC through
``cupy.RawModule``, so nothing has to be built at install time and no wheel
carries device code.  The exception is ``libmykg``: a faster copy of
GPU4PySCF's *general* Rys exchange kernel, which is split over several
translation units and needs relocatable device code, so NVRTC cannot build it.

It is therefore compiled with ``nvcc`` the first time ``fastk`` wants it and
cached per compute capability.  If ``nvcc`` is not on the machine, this returns
None and ``fastk`` hands the angular-momentum classes that would have used it
back to GPU4PySCF -- slower, still correct.

Overrides:
  ELECTROLITES_LIBMYKG   path to a prebuilt library; used as-is
  ELECTROLITES_CACHE     where to put the build (default ~/.cache/electrolites)
  ELECTROLITES_NVCC      the compiler to use (default: nvcc on PATH)
  ELECTROLITES_ARCH      compute capability digits, e.g. 80 (default: this GPU)
"""
import os
import shutil
import subprocess
import sys
import tempfile
import functools

from ._paths import CUDA_DIR

_SOURCES = (
    'gvhf-rys/rys_roots_dat.cu',
    'gvhf-rys/rys_constant.cu',
    'myk2/rys_k_general.cu',
)
_warned = False


def _log(msg):
    global _warned
    if not _warned:
        print(f'[electrolites] {msg}', file=sys.stderr, flush=True)
        _warned = True


def cache_dir():
    base = os.environ.get('ELECTROLITES_CACHE')
    if not base:
        base = os.path.join(
            os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')),
            'electrolites')
    return base


def compute_capability():
    cc = os.environ.get('ELECTROLITES_ARCH')
    if cc:
        return str(cc)
    import cupy as cp
    return str(cp.cuda.Device().compute_capability)   # e.g. '80'


def nvcc_path():
    exe = os.environ.get('ELECTROLITES_NVCC')
    if exe:
        return exe
    exe = shutil.which('nvcc')
    if exe:
        return exe
    try:                                  # cupy knows where its own nvcc is
        import cupy
        return cupy.cuda.get_nvcc_path().split()[0]
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def general_k_library():
    """Path to ``libmykg`` for this GPU, or None if it cannot be provided."""
    override = os.environ.get('ELECTROLITES_LIBMYKG')
    if override:
        return override if os.path.exists(override) else None

    arch = compute_capability()
    out_dir = os.path.join(cache_dir(), f'sm_{arch}')
    out = os.path.join(out_dir, 'libmykg.so')
    if os.path.exists(out):
        return out

    nvcc = nvcc_path()
    if nvcc is None:
        _log('nvcc not found: the general exchange kernel falls back to '
             'GPU4PySCF (set ELECTROLITES_NVCC or ELECTROLITES_LIBMYKG)')
        return None

    srcs = [os.path.join(CUDA_DIR, s) for s in _SOURCES]
    if not all(os.path.exists(s) for s in srcs):
        _log('the general exchange kernel sources are not installed; '
             'those classes fall back to GPU4PySCF')
        return None

    os.makedirs(out_dir, exist_ok=True)
    cmd = [nvcc, '-O3', '-std=c++17', f'-arch=sm_{arch}',
           '--compiler-options', '-fPIC', '-rdc=true',
           '-I', CUDA_DIR, '-I', os.path.join(CUDA_DIR, 'gvhf-rys'),
           '-shared', '-o', 'libmykg.so'] + srcs
    # the staging directory has to sit inside the cache directory: the final
    # step is a rename, and a rename across filesystems is not possible
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(
                os.path.join(tmp, 'libmykg.so')):
            tail = (proc.stderr or proc.stdout or '').strip().splitlines()[-6:]
            _log('could not build the general exchange kernel; those classes '
                 'fall back to GPU4PySCF:\n  ' + '\n  '.join(tail))
            return None
        # atomic, so two processes racing on a cold cache cannot half-write it
        os.replace(os.path.join(tmp, 'libmykg.so'), out)
    return out


__all__ = ['general_k_library', 'cache_dir', 'compute_capability', 'nvcc_path']
