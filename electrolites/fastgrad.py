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

Importing this module also imports ``fastejk``, which is what makes the two
builds fast; on its own this file only changes which operator they build.
"""
import numpy as np
from gpu4pyscf.grad import rks as grks
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


def apply_patch():
    global _PATCHED
    if not _PATCHED:
        grks.Gradients._energy_ee_orig = grks.Gradients.energy_ee
        grks.Gradients.energy_ee = energy_ee
        _PATCHED = True


apply_patch()
