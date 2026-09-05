"""Produce the written kernel sources on first use instead of shipping them.

Three of the generators here take **nothing** from GPU4PySCF's source: they
emit the Rys or Hermite recurrences and the density contraction from the
angular-momentum class alone.  Their output is 13 MB of straight-line CUDA --
`fastejk` 6.6 MB, `fastjhigh` 4.4 MB, `fastkhigh` 1.9 MB -- and regenerating
all three takes **0.7 s**, byte for byte identical every time.  Carrying the
generator and not its output is the better trade at that ratio, so the
repository carries `electrolites/codegen/` and this module produces the CUDA
on demand and caches it.

The other three generators are not like this.  `gen_kernels.py`,
`gen_k2_kernels.py` and `gen_j_kernels.py` *lift* the integral arithmetic out
of a GPU4PySCF source file and rewrite only the scaffolding around it, so they
need a GPU4PySCF checkout, which someone who installed a wheel does not have.
Their output stays in the package, and `source_path` returns it untouched.

The launch tables stay in the package too, all three of them.  They are small,
and they are not really products of the generators: they are the measured
tuning, one entry per angular-momentum class, which
`benchmarks/sweep_j_high.py` and its friends produce and which no amount of
regeneration would recover.

Overrides:
  ELECTROLITES_GENERATED_DIR   a directory of pre-generated sources, used as
                               is; for an offline or read-only installation,
                               or to share one copy across a cluster
  ELECTROLITES_CACHE           where the cache lives (default
                               ~/.cache/electrolites), shared with `_nvcc`
"""
import functools
import hashlib
import importlib
import os
import subprocess
import sys
import tempfile

from ._paths import KERNEL_DIR

#: generated file -> (generator module, extra arguments).  A generator that
#: needs its launch table as *input* is given the shipped one.
_SPECS = {
    'fastjhigh_generated.cu': ('electrolites.codegen.gen_j_high', ()),
    'fastkhigh_generated.cu': ('electrolites.codegen.gen_khigh',
                               ('--table', 'fastkhigh_launch.json')),
    'fastejk_generated.cu': ('electrolites.codegen.gen_ejk',
                             ('--table', 'fastejk_launch.json')),
}


def cache_dir():
    base = os.environ.get('ELECTROLITES_CACHE')
    if not base:
        base = os.path.join(
            os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')),
            'electrolites')
    return os.path.join(base, 'generated')


def _fingerprint(module, args):
    """Hash of everything that decides the output, so an edit invalidates."""
    h = hashlib.sha256()
    src = importlib.import_module(module).__file__
    with open(src, 'rb') as f:
        h.update(f.read())
    for a in args:
        h.update(a.encode())
        p = os.path.join(KERNEL_DIR, a)
        if os.path.exists(p):
            with open(p, 'rb') as f:
                h.update(f.read())
    return h.hexdigest()[:16]


@functools.lru_cache(maxsize=16)
def source_path(name):
    """Path to a kernel source, generating and caching it if it is not shipped.

    Returns the packaged file when there is one, so the lifted kernels, a
    pre-generated tree pointed at by ``ELECTROLITES_GENERATED_DIR``, and any
    file a user drops in are all used as they are.
    """
    override = os.environ.get('ELECTROLITES_GENERATED_DIR')
    if override:
        p = os.path.join(override, name)
        if os.path.exists(p):
            return p
    p = os.path.join(KERNEL_DIR, name)
    if os.path.exists(p):
        return p

    spec = _SPECS.get(name)
    if spec is None:
        raise FileNotFoundError(
            f'{name} is not in {KERNEL_DIR} and no generator writes it')
    module, args = spec
    args = [os.path.join(KERNEL_DIR, a) if a.endswith('.json') else a
            for a in args]

    out_dir = os.path.join(cache_dir(), _fingerprint(module, spec[1]))
    out = os.path.join(out_dir, name)
    if os.path.exists(out):
        return out

    os.makedirs(out_dir, exist_ok=True)
    # staged inside the cache directory: the last step is a rename, and a
    # rename across filesystems is not possible
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        staged = os.path.join(tmp, name)
        cmd = [sys.executable, '-m', module, *args]
        if module.endswith('gen_j_high'):
            # it writes its launch table as a side effect; keep that out of
            # the package directory, which may be read-only
            cmd += ['--json', os.path.join(tmp, 'launch.json')]
        with open(staged, 'w') as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE,
                                  text=True)
        if proc.returncode != 0 or os.path.getsize(staged) == 0:
            tail = (proc.stderr or '').strip().splitlines()[-6:]
            raise RuntimeError(
                f'could not generate {name} with {module}:\n  '
                + '\n  '.join(tail))
        os.replace(staged, out)          # atomic: two processes cannot race
    return out


def available(name):
    """Can `source_path` produce this source -- shipped, overridden or written?"""
    override = os.environ.get('ELECTROLITES_GENERATED_DIR')
    return bool((override and os.path.exists(os.path.join(override, name)))
                or os.path.exists(os.path.join(KERNEL_DIR, name))
                or name in _SPECS)


def ensure_all():
    """Generate everything now rather than on first use.  Returns the paths."""
    return {n: source_path(n) for n in _SPECS}


__all__ = ['source_path', 'available', 'ensure_all', 'cache_dir']
