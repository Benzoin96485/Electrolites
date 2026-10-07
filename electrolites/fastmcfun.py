"""The multi-collinear XC functional on half the spin samples (A21).

GPU4PySCF's GKS ``collinear='mcol'`` path (``dft/mcfun_gpu.py``) projects the
magnetisation of every grid point onto each of 770 Lebedev directions ``n``
and evaluates the collinear functional, at one derivative order above the one
asked for, on ``(rho, s = m.n)`` for every one -- 90-94 % of a patched GKS
SCF (``docs/KERNEL_TARGETS_1.8.1.md`` §11.8).

A Lebedev grid is symmetric under inversion: ``-n`` is a sample whenever
``n`` is, with the same weight.  And a collinear functional is symmetric
under exchanging the two spins, i.e. ``f(rho, -s) = f(rho, s)``, so the
per-sample term ``g_n(rho, m) = f(rho, m.n)`` that the multi-collinear
functional integrates over the sphere satisfies ``g_{-n} = g_n`` as a function
of ``(rho, m)`` -- and so do all of its derivatives with respect to ``rho``
and ``m``, which are exactly the per-sample contributions
``_eval_xc_lebedev`` accumulates into ``exc_eff``, ``vxc_eff``, ``fxc_eff``
and ``kxc_eff``.  Every antipodal pair contributes twice what one member
does, at every derivative order.

So ``_make_sph_samples`` hands ``_eval_xc_lebedev`` one member of each pair
with twice the weight: half the functional evaluations and half the sample
contractions, the same result up to rounding (libxc evaluated on
spin-swapped inputs is symmetric to rounding).  If a sample set is not
inversion-symmetric the full set is returned.  The collinear-threshold path
(principal-axis samples) is untouched.

``FASTMCFUN_OFF=1`` keeps GPU4PySCF's samples.  ``docs/README_fastmcfun.md``.
"""
import functools
import os

import numpy as np

from gpu4pyscf.dft import mcfun_gpu as M

_OFF = bool(os.environ.get('FASTMCFUN_OFF'))
STATS = {'halved': 0, 'full': 0}


def stats():
    return dict(STATS)


@functools.lru_cache(maxsize=8)
def _half(spin_samples):
    directions, weights = M._make_sph_samples_orig(spin_samples)
    d = np.asarray(directions)
    w = np.asarray(weights)
    # one representative per antipodal pair: z > 0, or z == 0 and y > 0, ...
    tol = 1e-12
    keep = (d[:, 2] > tol) | ((abs(d[:, 2]) <= tol) & (d[:, 1] > tol)) | (
        (abs(d[:, 2]) <= tol) & (abs(d[:, 1]) <= tol) & (d[:, 0] > tol))
    if keep.sum() * 2 != len(d):
        return None
    # every kept sample's antipode must be in the set with the same weight
    other = ~keep
    for i in np.where(keep)[0]:
        j = np.where(other)[0][np.argmin(
            np.abs(d[other] + d[i]).sum(axis=1))]
        if abs(d[j] + d[i]).max() > 1e-10 or abs(w[j] - w[i]) > 1e-12 * w[i]:
            return None
    return (np.asfortranarray(d[keep]), w[keep] * 2)


def _make_sph_samples(spin_samples):
    half = _half(spin_samples)
    if half is None:
        STATS['full'] += 1
        return M._make_sph_samples_orig(spin_samples)
    STATS['halved'] += 1
    return half[0].copy(order='F'), half[1].copy()


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    M._make_sph_samples_orig = M._make_sph_samples
    M._make_sph_samples = _make_sph_samples
    _PATCHED = True


apply_patch()
