"""Range-separated nuclear gradients through the long-range operator.

A range-separated hybrid's exchange is ``hyb*K(1/r) + (alpha-hyb)*K(erf(wr)/r)``.
Because ``K(erfc) = K(1/r) - K(erf)``, that is the same matrix as GPU4PySCF's
``alpha*K(1/r) + (hyb-alpha)*K(erfc(wr)/r)`` -- and GPU4PySCF's gradient takes
the second form, with the comment "prefer computing the SR part".  The
short-range operator needs **twice** the Rys roots (the quadrature is the
difference of two Boys functions), which is why ``fastejk`` declines it; the
long-range one is one ``fma`` on top of the full-range kernel
(``rsqrt(aij+akl) -> rsqrt(aij+akl + aij*akl/w^2)``, see
``fastejk_prologue.cu``).

So this module swaps the two builds round: the full-range pass carries ``hyb``
instead of ``alpha`` and the second pass builds the long-range operator with
``alpha-hyb``.  The exchange is identical; what changes is that both passes now
run on the generated kernels at the same Rys order.  When ``hyb == 0`` -- a
functional with no short-range HF exchange -- the first exchange build
disappears entirely.

The unrestricted gradient (``grad/uks.py:energy_ee``) splits the exchange
the same way, and is rewritten the same way (catalogue B16): its source,
with the exchange block replaced by the one below, executed in its module's
namespace -- the XC, NLC and grid-response lines stay GPU4PySCF's.

Both analytic Hessians' ``partial_hess_elec`` (``hessian/rks.py``,
``hessian/uks.py``) make the same choice for their ip2 term, and get the same
swap: the full-range pass carries ``hyb``, and the second
``_partial_ejk_ip2`` builds the long-range operator with ``alpha-hyb``
instead of the short-range one (whose ``_ip2_quartets_scheme`` doubles the
Rys roots and the shared memory per quartet).  ``FASTGRAD_HESS_OFF=1`` keeps
GPU4PySCF's.

Importing this module also imports ``fastejk``, which is what makes the two
builds fast; on its own this file only changes which operator they build.
"""
import inspect
import textwrap
import types

import os

import numpy as np
from gpu4pyscf.grad import rks as grks
from gpu4pyscf.grad import uks as guks
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array, ensure_numpy

from . import fastejk                    # noqa: F401  (patches the 2e gradient build)

_PATCHED = False


def energy_ee(ks_grad, mol=None, dm=None, verbose=None):
    """Drop-in replacement for gpu4pyscf.grad.rks.energy_ee."""
    if mol is None:
        mol = ks_grad.mol
    if dm is None:
        dm = ks_grad.base.make_rdm1()
    if not hasattr(dm, "mo_coeff"):
        dm = tag_array(dm, mo_coeff=ks_grad.base.mo_coeff)
    if not hasattr(dm, "mo_occ"):
        dm = tag_array(dm, mo_occ=ks_grad.base.mo_occ)
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    mf = ks_grad.base
    ni = mf._numint
    grids = ks_grad.grids if ks_grad.grids is not None else mf.grids
    if grids.coords is None:
        grids.build(sort_grids=True)

    if ks_grad.grid_response:
        exc, exc1 = grks.get_exc_full_response(ni, mol, grids, mf.xc, dm,
                                               verbose=log)
        exc1 *= 2
        exc1 += exc
    else:
        exc, exc1 = grks.get_exc(ni, mol, grids, mf.xc, dm, verbose=log)
        exc1 *= 2
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if mf.do_nlc():
        enlc1_per_atom, enlc1_grid = grks._get_denlc(ks_grad, mol, dm)
        exc1 += enlc1_per_atom * 2
        if ks_grad.grid_response:
            exc1 += enlc1_grid

    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    # hyb is the short-range (and full-range) HF coefficient, alpha the
    # long-range one; K = hyb*K(1/r) + (alpha-hyb)*K(erf(wr)/r) covers every
    # case GPU4PySCF splits into four, including alpha == 0 and hyb == 0.
    k_factor = hyb if with_k else 0.
    exc1 += ks_grad.jk_energy_per_atom(dm, 1., k_factor, verbose=log)
    if with_k and omega != 0 and alpha != hyb:
        exc1 += ks_grad.jk_energy_per_atom(
            dm, 0., alpha - hyb, omega=omega, verbose=log)
    return exc1


_UKS_K_OLD = "    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)\n"
_UKS_K_NEW = """    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    # fastgrad: hyb*K(1/r) + (alpha-hyb)*K(erf), both on the generated
    # kernels, instead of alpha*K(1/r) + (hyb-alpha)*K(erfc)
    k_factor = hyb if with_k else 0.
    exc1 += ks_grad.jk_energy_per_atom(dm, 1., k_factor, verbose=log)
    if with_k and omega != 0 and alpha != hyb:
        exc1 += ks_grad.jk_energy_per_atom(
            dm, 0., alpha - hyb, omega=omega, verbose=log)
    return exc1
"""


def _build_uks_energy_ee():
    """``grad/uks.py:energy_ee`` with its exchange block -- everything from
    ``rsh_and_hybrid_coeff`` to the return -- replaced; None if the source
    is not the one this was written against."""
    src = textwrap.dedent(inspect.getsource(guks.energy_ee))
    i = src.find(_UKS_K_OLD)
    if i < 0 or src.count('return exc1') != 1 or 'omega = -omega' not in src:
        return None
    src = src[:i] + _UKS_K_NEW
    ns = {}
    exec(compile(src, guks.__file__ + ' (fastgrad)', 'exec'), ns)
    # bound to the live module namespace rather than a copy of it, so that
    # the names it calls (`get_exc`, `_get_denlc`, ...) are whatever later
    # patches -- fastxcgrad's -- put there
    f = ns['energy_ee']
    return types.FunctionType(f.__code__, guks.__dict__, f.__name__,
                              f.__defaults__, f.__closure__)


_HESS_EDITS = [
    # the full-range pass carries hyb in every RSH case
    ("    de2, ejk = ", "    if with_k and omega != 0:\n"
                      "        k_factor = hyb  # fastgrad (B16)\n"
                      "    de2, ejk = "),
    # and the second pass is the long-range operator with alpha - hyb
    ("        omega = -omega # Prefer computing the SR part\n",
     "        # fastgrad (B16): K(erf) with alpha-hyb, not K(erfc) with hyb-alpha\n"),
    ("        vhfopt = mf._opt_gpu.get(omega",
     "        k_factor = alpha - hyb\n        vhfopt = mf._opt_gpu.get(omega"),
]


def _build_partial_hess_elec(mod):
    """``mod.partial_hess_elec`` with `_HESS_EDITS` applied, bound to the
    live module namespace; None if any edit does not match exactly once."""
    src = textwrap.dedent(inspect.getsource(mod.partial_hess_elec))
    for old, new in _HESS_EDITS:
        if src.count(old) != 1:
            return None
        src = src.replace(old, new)
    ns = {}
    exec(compile(src, mod.__file__ + ' (fastgrad)', 'exec'), ns)
    f = ns['partial_hess_elec']
    return types.FunctionType(f.__code__, mod.__dict__, f.__name__,
                              f.__defaults__, f.__closure__)


def apply_patch():
    global _PATCHED
    if not _PATCHED:
        # FASTGRAD_UKS_OFF=1: GPU4PySCF's unrestricted energy_ee (ablation)
        if not os.environ.get('FASTGRAD_HESS_OFF'):
            from gpu4pyscf.hessian import rks as hrks, uks as huks
            for mod in (hrks, huks):
                f = _build_partial_hess_elec(mod)
                if f is not None:
                    mod._partial_hess_elec_orig = mod.partial_hess_elec
                    mod.partial_hess_elec = f
                    mod.Hessian.partial_hess_elec = f
        uks_ee = (None if os.environ.get('FASTGRAD_UKS_OFF')
                  else _build_uks_energy_ee())
        if uks_ee is not None:
            guks.Gradients._energy_ee_orig = guks.Gradients.energy_ee
            # `grad/ukspu.py` calls `uks_grad.energy_ee` directly
            guks.Gradients.energy_ee = guks.energy_ee = uks_ee
        grks.Gradients._energy_ee_orig = grks.Gradients.energy_ee
        # both the method and the module-level name, which are the same
        # function: `grad/rkspu.py` calls `rks_grad.energy_ee` directly
        grks.Gradients.energy_ee = grks.energy_ee = energy_ee
        _PATCHED = True


apply_patch()
