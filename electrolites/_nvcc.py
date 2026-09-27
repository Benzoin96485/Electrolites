"""Build the two libraries that are not JIT-compiled, on first use.

Almost every kernel here is compiled at run time by NVRTC through
``cupy.RawModule``, so nothing has to be built at install time and no wheel
carries device code.  Three are exceptions, all because they are split over
several translation units and need relocatable device code, which NVRTC cannot
do:

* ``libmykg``, a faster copy of GPU4PySCF's *general* Rys exchange kernel,
  which `fastk` wants for the angular-momentum classes it does not unroll.
* ``libmyejk2``, the second-derivative ERI kernels of catalogue B3, lifted from
  GPU4PySCF's ``rys_contract_jk_ip2.cu`` and ``unrolled_ejk_ip2_type{12,3}.cu``
  with the divisions folded and a register cap added (codegen/gen_ejk_ip2.py),
  which `fastejk2` binds over GPU4PySCF's own entry points.
* ``libmyip2``, the density-fitted Hessian's ``ejk_int3c2e_ip2`` with its
  contraction and atomics rewritten (``docs/DF_WORK_LIST.md`` W31), which
  ``_ip2launch`` launches in place of GPU4PySCF's.

Each is compiled with ``nvcc`` the first time it is wanted and cached per
compute capability.  If ``nvcc`` is not on the machine, the accessor returns
None and the module that wanted it stands down -- slower, still correct.

Overrides:
  ELECTROLITES_LIBMYKG   path to a prebuilt libmykg; used as-is
  ELECTROLITES_LIBMYEJK2 path to a prebuilt libmyejk2; used as-is
  ELECTROLITES_LIBMYIP2  path to a prebuilt libmyip2; used as-is
  ELECTROLITES_CACHE     where to put the build (default ~/.cache/electrolites)
  ELECTROLITES_NVCC      the compiler to use (default: nvcc on PATH)
  ELECTROLITES_ARCH      compute capability digits, e.g. 80 (default: this GPU)
"""
import functools
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

from ._paths import CUDA_DIR

_SOURCES = (
    'gvhf-rys/rys_roots_dat.cu',
    'gvhf-rys/rys_constant.cu',
    'myk2/rys_k_general.cu',
)
#: the second-derivative ERI library (catalogue B3).  Its support headers are
#: a 1.8.x copy and live in their own directory, because BoundsInfo gained
#: per-group q_cond/s_cond after 1.7.0 and the two sets are not interchangeable.
_EJK2_SOURCES = (
    'gvhf-rys181/gvhf-rys/rys_roots_dat.cu',
    'gvhf-rys181/gvhf-rys/rys_constant.cu',
    'myejk2/rys_contract_jk_ip2.cu',
    'myejk2/unrolled_ejk_ip2_type12.cu',
    'myejk2/unrolled_ejk_ip2_type3.cu',
)
#: the density-fitted Hessian's ip2 kernel (W31): GPU4PySCF's
#: ejk_int3c2e_ip2 with its contraction and atomics rewritten
_IP2_SOURCES = (
    'gvhf-rys181/gvhf-rys/rys_roots_dat.cu',
    'gvhf-rys181/gvhf-rys/rys_constant.cu',
    'myip2/ejk_int3c2e_ip2.cu',
)
_IP2_DEPS = (
    'myip2/build_rys_gxyz.cuh',
    'myip2/ip2_cases.inc',
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


def _build(name, sources, includes, what, override_env, deps=()):
    """Path to one nvcc-built library for this GPU, or None.

    ``what`` names the thing that stands down if the build cannot happen, so
    the one warning a process prints says which module lost what.
    """
    override = os.environ.get(override_env)
    if override:
        return override if os.path.exists(override) else None

    arch = compute_capability()
    srcs = [os.path.join(CUDA_DIR, s) for s in sources]
    # The cache key carries a hash of the sources and of the compile line, so
    # editing a kernel or changing a flag invalidates it.  Keying on the
    # compute capability alone silently serves a stale library, which is a
    # whole afternoon of measuring the previous build.
    out_dir = os.path.join(cache_dir(), f'sm_{arch}')
    out = os.path.join(out_dir, name)
    if all(os.path.exists(s) for s in srcs):
        h = hashlib.sha256()
        for s_ in srcs + [os.path.join(CUDA_DIR, d) for d in deps]:
            with open(s_, 'rb') as f:
                h.update(f.read())
        h.update(repr((arch, includes)).encode())
        out = os.path.join(out_dir, h.hexdigest()[:16], name)
        out_dir = os.path.dirname(out)
    if os.path.exists(out):
        return out

    nvcc = nvcc_path()
    if nvcc is None:
        _log(f'nvcc not found: {what} falls back to GPU4PySCF '
             f'(set ELECTROLITES_NVCC or {override_env})')
        return None

    if not all(os.path.exists(s) for s in srcs):
        _log(f'the sources for {what} are not installed; it falls back to '
             f'GPU4PySCF')
        return None

    os.makedirs(out_dir, exist_ok=True)
    cmd = [nvcc, '-O3', '-std=c++17', f'-arch=sm_{arch}',
           '--compiler-options', '-fPIC', '-rdc=true']
    for inc in includes:
        cmd += ['-I', os.path.join(CUDA_DIR, inc) if inc else CUDA_DIR]
    cmd += ['-shared', '-o', name] + srcs
    # the staging directory has to sit inside the cache directory: the final
    # step is a rename, and a rename across filesystems is not possible
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(os.path.join(tmp, name)):
            tail = (proc.stderr or proc.stdout or '').strip().splitlines()[-6:]
            _log(f'could not build {what}; it falls back to GPU4PySCF:\n  '
                 + '\n  '.join(tail))
            return None
        # atomic, so two processes racing on a cold cache cannot half-write it
        os.replace(os.path.join(tmp, name), out)
    return out


@functools.lru_cache(maxsize=1)
def general_k_library():
    """Path to ``libmykg`` for this GPU, or None if it cannot be provided."""
    return _build('libmykg.so', _SOURCES, ('', 'gvhf-rys'),
                  'the general exchange kernel', 'ELECTROLITES_LIBMYKG')


@functools.lru_cache(maxsize=1)
def ejk_ip2_library():
    """Path to ``libmyejk2`` for this GPU, or None if it cannot be provided."""
    return _build('libmyejk2.so', _EJK2_SOURCES,
                  ('gvhf-rys181', 'gvhf-rys181/gvhf-rys'),
                  "the Hessian's second-derivative ERI kernels",
                  'ELECTROLITES_LIBMYEJK2')


@functools.lru_cache(maxsize=1)
def ip2_library():
    """Path to ``libmyip2`` (W31's ``ejk_int3c2e_ip2``) for this GPU, or None."""
    return _build('libmyip2.so', _IP2_SOURCES,
                  ('gvhf-rys181', 'gvhf-rys181/gvhf-rys'),
                  "the DF Hessian's ip2 kernel", 'ELECTROLITES_LIBMYIP2',
                  deps=_IP2_DEPS)


__all__ = ['general_k_library', 'ejk_ip2_library', 'ip2_library', 'cache_dir',
           'compute_capability', 'nvcc_path']
