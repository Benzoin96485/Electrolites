"""The direct Hessian's ``h1`` two-electron term (catalogue B15).

``hessian/rhf.py:_get_jk_ip1`` launches ``RYS_build_jk_ip1`` once per
angular-momentum class -- 1372 times for penicillin V, 5292 for the Ni
dithiolene anion -- and GPU4PySCF's driver calls ``cudaGetDeviceProperties``
on every one of them (on device 0, whatever the current device).  That is a
millisecond-scale host call, and it is what made the term host-bound: 26.8 %
of a patched UKS transition-metal Hessian (``docs/KERNEL_TARGETS_1.8.1.md``
§10.7).

``libmyjkip1`` (built by ``_nvcc`` from ``kernels/cuda/myjkip1``, which
``codegen/gen_jk_ip1.py`` writes from GPU4PySCF's sources) is the same
driver and the same kernels with

* **(a)** the multiprocessor count of the current device queried once and
  cached -- ``_get_jk_ip1`` reads the function off ``libvhf_rys`` at call
  time, so binding ours over that attribute is the whole patch;
* **(b)** K over several densities from one integral pass: the open-shell
  ``make_h1`` (``hessian/uhf.py``, ``uks.py``) called ``_get_jk_ip1`` once per
  spin, computing every integral twice and J twice.  `get_jk_ip1_spins`
  builds J of the total density and K of each spin in one call (GPU4PySCF's
  ``_get_jk_ip1`` source with the single-density assert and the ``vk``
  shape changed), and the two ``make_h1`` are GPU4PySCF's source calling it;
* P1 (no fp64 division in the hot loops) on the kernels, as in B3.

``FASTJKIP1_OFF=1`` keeps GPU4PySCF's; ``FASTJKIP1_NO_SPINS=1`` keeps (a)
and P1 but the per-spin calls.  Without ``nvcc`` the module stands down.
``docs/README_fastjkip1.md``.
"""
import ctypes
import functools
import inspect
import os
import re
import textwrap
import types

from gpu4pyscf.hessian import rhf as HR

from . import _nvcc

_OFF = bool(os.environ.get('FASTJKIP1_OFF'))
_NO_SPINS = bool(os.environ.get('FASTJKIP1_NO_SPINS'))
_PATCHED = False
STATS = {'spin_calls': 0}


def stats():
    return dict(STATS)


_MULTI_EDITS = [
    ('def _get_jk_ip1(', 'def _get_jk_ip1_multi('),
    ('    assert n_dm == 1\n', '    nkdm = max(n_dm - 1, 1)\n'),
    ('            vk = cp.zeros(((atom1-atom0)*3, nao, nao))\n',
     '            vk = cp.zeros(((atom1-atom0)*3*nkdm, nao, nao))\n'),
    ('        vk = vk.reshape(atom1-atom0, 3, nao_orig, nao_orig)\n',
     '        vk = vk.reshape(atom1-atom0, 3, nkdm, nao_orig, nao_orig)\n'),
]


@functools.lru_cache(maxsize=1)
def _multi():
    """GPU4PySCF's ``_get_jk_ip1`` taking ``[total, spin a, spin b]``: J of
    the first density, K of the others (``libmyjkip1``'s contract)."""
    src = textwrap.dedent(inspect.getsource(HR._get_jk_ip1))
    for old, new in _MULTI_EDITS:
        if src.count(old) != 1:
            return None
        src = src.replace(old, new)
    ns = {}
    exec(compile(src, HR.__file__ + ' (fastjkip1)', 'exec'), ns)
    f = ns['_get_jk_ip1_multi']
    return types.FunctionType(f.__code__, HR.__dict__, f.__name__,
                              f.__defaults__, f.__closure__)


def get_jk_ip1_spins(mol, dma, dmb, with_j=True, with_k=True,
                     atoms_slice=None, verbose=None):
    """``(vj, vka, vkb)``: J of ``dma + dmb`` and each spin's K, as two
    ``_get_jk_ip1`` calls would give them summed, from one integral pass."""
    import cupy as cp
    STATS['spin_calls'] += 1
    dma = cp.asarray(dma)
    dmb = cp.asarray(dmb)
    dms = cp.stack([dma + dmb, dma, dmb])
    vj, vk = _multi()(mol, dms, with_j=with_j, with_k=with_k,
                      atoms_slice=atoms_slice, verbose=verbose)
    if vk is None:
        return vj, None, None
    return (vj, cp.ascontiguousarray(vk[:, :, 0]),
            cp.ascontiguousarray(vk[:, :, 1]))


_PAIR = re.compile(
    r"( +)vja, vka = (\w+)\._get_jk_ip1\(mol, dm0a(, [^\n]*)\)\n"
    r" +vjb, vkb = \2\._get_jk_ip1\(mol, dm0b\3\)\n")
_LR = re.compile(
    r"( +)vka_lr = (\w+)\._get_jk_ip1\(mol, dm0a(, [^\n]*)\)\[1\]\n"
    r" +vkb_lr = \2\._get_jk_ip1\(mol, dm0b\3\)\[1\]\n")


def _rebuild_make_h1(mod):
    src = textwrap.dedent(inspect.getsource(mod.make_h1))
    src, n1 = _PAIR.subn(
        r"\1vja, vka, vkb = _fastjkip1_spins(mol, dm0a, dm0b\3)\n"
        r"\1vjb = 0\n", src)
    src, n2 = _LR.subn(
        r"\1vka_lr, vkb_lr = _fastjkip1_spins(mol, dm0a, dm0b\3)[1:]\n", src)
    if n1 != 1:
        return None
    ns = {}
    exec(compile(src, mod.__file__ + ' (fastjkip1)', 'exec'), ns)
    f = ns['make_h1']
    return types.FunctionType(f.__code__, mod.__dict__, f.__name__,
                              f.__defaults__, f.__closure__)


@functools.lru_cache(maxsize=1)
def _library():
    so = _nvcc.jk_ip1_library()
    if so is None:
        return None
    try:
        lib = ctypes.CDLL(so)
    except OSError:
        return None
    lib.MYJKIP1_build_jk_ip1.restype = ctypes.c_int
    return lib


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return _PATCHED
    import gpu4pyscf
    if not gpu4pyscf.__version__.startswith('1.8.'):
        return False
    lib = _library()
    if lib is None:
        return False
    vhf = HR.libvhf_rys
    if not hasattr(vhf, '_jkip1_orig_RYS_build_jk_ip1'):
        vhf._jkip1_orig_RYS_build_jk_ip1 = vhf.RYS_build_jk_ip1
    vhf.RYS_build_jk_ip1 = lib.MYJKIP1_build_jk_ip1
    if not _NO_SPINS and _multi() is not None:
        from gpu4pyscf.hessian import uhf as HU, uks as HUKS
        for mod in (HU, HUKS):
            new = _rebuild_make_h1(mod)
            if new is None:
                continue
            mod._fastjkip1_spins = get_jk_ip1_spins
            mod._make_h1_orig = mod.make_h1
            mod.make_h1 = new
            mod.Hessian.make_h1 = new
    _PATCHED = True
    return True


def unpatch():
    """GPU4PySCF's function back: for the tests' oracle only."""
    global _PATCHED
    vhf = HR.libvhf_rys
    orig = getattr(vhf, '_jkip1_orig_RYS_build_jk_ip1', None)
    if orig is not None:
        vhf.RYS_build_jk_ip1 = orig
    from gpu4pyscf.hessian import uhf as HU, uks as HUKS
    for mod in (HU, HUKS):
        o = getattr(mod, '_make_h1_orig', None)
        if o is not None:
            mod.make_h1 = o
            mod.Hessian.make_h1 = o
    _PATCHED = False


apply_patch()
