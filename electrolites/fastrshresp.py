"""Range-separated hybrids in the response: both exchange ranges in one pass
(catalogue C12).

``scf/_response_functions.py`` builds a range-separated hybrid's response
potential as

    vj, vk = mf.get_jk(mol, dm1, hermi=hermi)      # J + full-range K
    vk *= hyb
    if omega > 1e-10:
        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha - hyb)   # long range

-- two passes over the shell quartets per call, each with its own screening,
density loads and atomicAdds, in every CPKS, TDA and TDDFT iteration.  The
triplet and the no-J UHF branches do the same with ``get_k`` for the first
pass.  ``fastk.get_k_rsh`` already builds ``hyb*K + (alpha-hyb)*K_LR`` in one
pass (the SCF uses it through ``fastrsh``); it now also takes ``hermi = 0``.

This module rewrites those statements in ``_gen_rhf_response`` and
``_gen_uhf_response`` (source transform, bound to the live module
namespace) to call ``rsh_jk``:

* without J (triplet; UHF ``with_j=False``): the fused exchange, always;
* with J there are two arrangements -- GPU4PySCF's (``fastjk`` J + full-range
  K, then ``fastk`` long-range K) and ``fastj`` J + the fused exchange.  Which
  is faster depends on ``hermi`` and the size: on penicillin V the fused one
  wins at ``hermi = 0`` by 1.06-1.30x and loses at ``hermi = 1`` by up to
  10 %; on taxol it wins both (1.11-1.42x; ``benchmarks/rsh_response_probe.py``).
  So each (SCF object, hermi, density-count bucket) times both on its 2nd-5th
  calls and keeps the faster; the answers agree to 1e-11.

Falls back to GPU4PySCF's statements whenever ``omega`` is zero, the SCF is
density-fitted, the density is on the host, or a shell is beyond the GPU
kernels.  ``FASTRSHRESP_OFF=1`` leaves the response functions untouched;
``FASTRSHRESP_MODE=fused|split`` fixes the arrangement.  Measured in
``docs/KERNEL_TARGETS_1.8.1.md`` §13.28.
"""
import inspect
import os
import re
import textwrap
import time
import types

import cupy as cp

_OFF = bool(os.environ.get('FASTRSHRESP_OFF'))
_MODE = os.environ.get('FASTRSHRESP_MODE', 'auto')
STATS = {'fused_k': 0, 'split': 0, 'j_plus_fused': 0, 'stock': 0,
         'decisions': {}}


def _stock(mf, mol, dm1, hermi, hyb, alpha, omega, with_j):
    if with_j:
        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
    else:
        vj, vk = None, mf.get_k(mol, dm1, hermi=hermi)
    vk *= hyb
    if abs(omega) > 1e-10:
        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha - hyb)
    return vj, vk


def _fused_k(mf, mol, dm1, hermi, hyb, alpha, omega):
    """hyb*K + (alpha-hyb)*K_LR in one pass, or None where not served."""
    from gpu4pyscf.scf import jk as _jk
    from . import fastk
    opt_l = mf._opt_gpu.get(omega)
    if opt_l is None:
        with mol.with_range_coulomb(omega):
            opt_l = mf._opt_gpu[omega] = _jk._VHFOpt(
                mol, mf.direct_scf_tol).build()
    opt_f = fastk._full_range_opt(opt_l)
    if opt_f.h_shls or opt_l.h_shls:
        return None
    nao = opt_f.mol.nao
    dms = dm1.reshape(-1, nao, nao)
    dms = cp.asarray(opt_f.apply_coeff_C_mat_CT(dms), order='C')
    vk = fastk.get_k_rsh(opt_f, opt_l, dms, hermi, hyb, alpha - hyb)
    return opt_f.apply_coeff_CT_mat_C(vk).reshape(dm1.shape)


def _eligible(mf, dm1, hermi, omega):
    from gpu4pyscf.__config__ import num_devices
    return (omega > 1e-10 and num_devices == 1 and hermi in (0, 1, 2)
            and isinstance(dm1, cp.ndarray)
            and getattr(mf, 'with_df', None) is None
            and hasattr(mf, '_opt_gpu') and not _OFF)


def _bucket(n):
    return max(0, int(n).bit_length() - 1)


def rsh_jk(mf, mol, dm1, hermi, hyb, alpha, omega, with_j):
    """``(vj, hyb*K + (alpha-hyb)*K_LR)`` for one response call; ``vj`` is
    None when ``with_j`` is False."""
    if not _eligible(mf, dm1, hermi, omega):
        STATS['stock'] += 1
        return _stock(mf, mol, dm1, hermi, hyb, alpha, omega, with_j)
    if not with_j:
        vk = _fused_k(mf, mol, dm1, hermi, hyb, alpha, omega)
        if vk is None:
            STATS['stock'] += 1
            return _stock(mf, mol, dm1, hermi, hyb, alpha, omega, with_j)
        STATS['fused_k'] += 1
        return None, vk

    def split():
        STATS['split'] += 1
        return _stock(mf, mol, dm1, hermi, hyb, alpha, omega, True)

    def fused():
        vk = _fused_k(mf, mol, dm1, hermi, hyb, alpha, omega)
        if vk is None:
            return None
        STATS['j_plus_fused'] += 1
        return mf.get_j(mol, dm1, hermi=hermi), vk

    if _MODE == 'split':
        return split()
    if _MODE == 'fused':
        return fused() or split()

    # auto: per (hermi, density-count bucket), call 1 warms both paths'
    # kernels up through whichever runs, calls 2-5 alternate and are timed,
    # and the faster one is kept
    ndm = dm1.size // (dm1.shape[-1] ** 2)
    key = (hermi, _bucket(ndm))
    tune = mf.__dict__.setdefault('_fastrshresp_tune', {})
    rec = tune.setdefault(key, {'n': 0, 'split': [], 'fused': [], 'pick': None})
    if rec['pick'] == 'split':
        return split()
    if rec['pick'] == 'fused':
        out = fused()
        return out if out is not None else split()
    rec['n'] += 1
    which = 'fused' if rec['n'] % 2 == 0 else 'split'
    sync = cp.cuda.get_current_stream().synchronize
    sync()
    t0 = time.perf_counter()
    out = fused() if which == 'fused' else split()
    if out is None:                           # fused not served here
        rec['pick'] = 'split'
        return split()
    sync()
    if rec['n'] > 1:
        rec[which].append(time.perf_counter() - t0)
    if len(rec['split']) >= 2 and len(rec['fused']) >= 2:
        rec['pick'] = ('fused' if min(rec['fused']) < min(rec['split'])
                       else 'split')
        STATS['decisions'][(id(mf),) + key] = (
            rec['pick'], min(rec['split']), min(rec['fused']))
    return out


# the statements replaced, in both response generators
_P_JK = re.compile(
    r'(?P<ind>[ ]*)vj, vk = mf\.get_jk\(mol, dm1, hermi=hermi\)\n'
    r'(?P=ind)vk \*= hyb\n'
    r'(?P=ind)if (?:abs\(omega\)|omega) > 1e-10:[^\n]*\n'
    r'(?P=ind)    vk \+= mf\.get_k\(mol, dm1, hermi, omega\) \* \(alpha-hyb\)\n')
_P_K = re.compile(
    r'(?P<ind>[ ]*)vk = mf\.get_k\(mol, dm1, hermi=hermi\)\n'
    r'(?P=ind)vk \*= hyb\n'
    r'(?P=ind)if (?:abs\(omega\)|omega) > 1e-10:[^\n]*\n'
    r'(?P=ind)    vk \+= mf\.get_k\(mol, dm1, hermi, omega\) \* \(alpha-hyb\)\n')


def _transform(fn, n_jk, n_k):
    src = textwrap.dedent(inspect.getsource(fn))
    src, a = _P_JK.subn(lambda m: m['ind'] + 'vj, vk = _fastrshresp_jk('
                        'mf, mol, dm1, hermi, hyb, alpha, omega, True)\n', src)
    src, b = _P_K.subn(lambda m: m['ind'] + 'vk = _fastrshresp_jk('
                       'mf, mol, dm1, hermi, hyb, alpha, omega, False)[1]\n', src)
    if (a, b) != (n_jk, n_k):
        raise RuntimeError(f'fastrshresp: {fn.__name__} has {a} J+K and {b} '
                           f'K-only RSH sites, expected {n_jk} and {n_k}')
    mod = inspect.getmodule(fn)
    ns = {}
    exec(compile(src, inspect.getsourcefile(fn), 'exec'), mod.__dict__, ns)
    new = ns[fn.__name__]
    out = types.FunctionType(new.__code__, mod.__dict__, fn.__name__,
                             fn.__defaults__, fn.__closure__)
    out.__kwdefaults__ = fn.__kwdefaults__
    out.__doc__ = fn.__doc__
    out._fastrshresp = True
    return out


_PATCHED = False


def _report():
    import sys
    sys.stderr.write(f'fastrshresp: {STATS}\n')


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    from gpu4pyscf.scf import _response_functions as rf, hf, uhf, rohf
    rf.__dict__['_fastrshresp_jk'] = rsh_jk
    r = _transform(rf._gen_rhf_response, 2, 1)
    u = _transform(rf._gen_uhf_response, 1, 1)
    rf._gen_rhf_response_orig = rf._gen_rhf_response
    rf._gen_uhf_response_orig = rf._gen_uhf_response
    rf._gen_rhf_response = r
    rf._gen_uhf_response = u
    hf.RHF.gen_response = r
    uhf.UHF.gen_response = u
    rohf.ROHF.gen_response = u
    if os.environ.get('FASTRSHRESP_VERBOSE'):
        import atexit
        atexit.register(_report)
    _PATCHED = True


apply_patch()
