"""
C6 dispersion coefficients without the host (catalogue C5).

Importing this module replaces ``gpu4pyscf.properties.c6._solve_full_spectrum``
and ``calc_c6``.  GPU4PySCF 1.8.1's C6 is a full-spectrum Casida solve:
``td.get_ab`` builds the ``(nov x nov)`` A and B matrices -- with the
two-electron part from ``mol.intor('int2e_sph')`` on the **CPU**, restored to
the full ``nao^4`` tensor on the host and copied to the device -- and then
the Cholesky factor of A - B, ``L^T (A+B) L`` and its ``eigh``, a triangular
solve and the back-transformation all run in **NumPy** on the host.
§11.9 measured that at 95 % of a 22-atom run: ``get_ab`` 41 %, the host
linear algebra 54 %.  And ``calc_c6(td, td)`` -- the C6 of a molecule with
itself, the usual call -- solves the same spectrum twice.

Here, the arithmetic unchanged:

* the AO two-electron tensor is filled on the GPU (``int4c2e.get_int4c2e``,
  GPU4PySCF's own kernel; the range-separated one for an RSH functional) and
  handed to ``get_ab``'s unchanged transformation, when twice ``nao^4``
  doubles fit on the device -- otherwise the CPU path, as before.  The
  long-range (erf) tensor of an RSH functional stays on the CPU path: the
  GPU fill's range-separated mode does not reproduce libcint's integrals;
* the Cholesky factor, the eigenproblem, the triangular solve, the
  back-transformation and the normalisation run in CuPy on the device, and
  only the final ``(X, Y)`` come back;
* the second spectrum is not solved when ``td_b is td_a``.

The non-Hermitian fallback (A - B not positive definite) stays GPU4PySCF's.
``FASTC6_OFF=1`` keeps GPU4PySCF's functions.  ``docs/README_fastc6.md``.
"""
import inspect
import os
import textwrap

import cupy as cp
import numpy as np

from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import get_avail_mem
from gpu4pyscf.properties import c6 as C6
from gpu4pyscf.tdscf import rhf as TDR
from gpu4pyscf.tdscf import rks as TDK

_OFF = bool(os.environ.get('FASTC6_OFF'))
STATS = {'gpu_eri': 0, 'cpu_eri': 0, 'gpu_eigh': 0, 'stock_eigh': 0,
         'reused_spectrum': 0}


def stats():
    return dict(STATS)


def _eri_ao(mol, omega=0.):
    """The full AO ERI tensor on the device, or None if it does not fit."""
    nao = mol.nao
    # the range-separated fill (GINTfill_int2e with omega) does not reproduce
    # libcint's erf integrals (C6 of NH3/wB97X off by 36 %), so an RSH
    # functional keeps the host path for its long-range part
    if omega != 0 or 2.2 * nao**4 * 8 > get_avail_mem():
        STATS['cpu_eri'] += 1
        return None
    from gpu4pyscf.scf import int4c2e
    STATS['gpu_eri'] += 1
    with mol.with_range_coulomb(omega):
        eri = cp.ascontiguousarray(int4c2e.get_int4c2e(mol, omega=omega))
    # the fill writes one half of each pair index (scf/tests/test_int4c2e.py
    # symmetrises it the same way)
    eri += eri.transpose(0, 1, 3, 2)
    eri += eri.transpose(1, 0, 2, 3)
    return eri


def _build_get_ab():
    """``tdscf.rhf.get_ab`` with its two host ERI builds replaced by
    `_eri_ao` (each falls back to the original lines when it returns None);
    everything else -- the transformation, the XC kernel, the solvent term --
    is GPU4PySCF's source, executed in its module's namespace."""
    src = textwrap.dedent(inspect.getsource(TDR.get_ab))
    old1 = ("            eri = mol.intor('int2e_sph', aosym='s8')\n"
            "            eri= ao2mo.restore(1, eri, nao)\n"
            "            eri = cp.asarray(eri)\n")
    new1 = ("            eri = _fastc6_eri_ao(mol)\n"
            "            if eri is None:\n"
            "                eri = mol.intor('int2e_sph', aosym='s8')\n"
            "                eri= ao2mo.restore(1, eri, nao)\n"
            "                eri = cp.asarray(eri)\n")
    old2 = ("                with mol.with_range_coulomb(omega):\n"
            "                    eri = mol.intor('int2e_sph', aosym='s8')\n"
            "                    eri= ao2mo.restore(1, eri, nao)\n"
            "                    eri = cp.asarray(eri)\n")
    new2 = ("                eri = _fastc6_eri_ao(mol, omega)\n"
            "                if eri is None:\n"
            "                  with mol.with_range_coulomb(omega):\n"
            "                    eri = mol.intor('int2e_sph', aosym='s8')\n"
            "                    eri= ao2mo.restore(1, eri, nao)\n"
            "                    eri = cp.asarray(eri)\n")
    if old1 not in src or old2 not in src:
        return None
    src = src.replace(old1, new1).replace(old2, new2)
    ns = dict(TDR.__dict__)
    ns['_fastc6_eri_ao'] = _eri_ao
    exec(compile(src, TDR.__file__ + ' (fastc6)', 'exec'), ns)
    return ns['get_ab']


_GET_AB = None


def _get_ab(td, mf):
    global _GET_AB
    if _GET_AB is None:
        _GET_AB = _build_get_ab() or TDR.get_ab
    return _GET_AB(td, mf, singlet=td.singlet)


def _solve_full_spectrum(td):
    log = logger.new_logger(td)
    mf = td._scf
    a_mat, b_mat = _get_ab(td, mf)
    a_mat = cp.asarray(np.asarray(a_mat).real)
    b_mat = cp.asarray(np.asarray(b_mat).real)
    nocc, nvir = a_mat.shape[:2]
    nov = nocc * nvir
    a_mat = a_mat.reshape(nov, nov)
    b_mat = b_mat.reshape(nov, nov)
    is_tda = isinstance(td, (TDR.TDA, TDK.TDA))

    if is_tda:
        w, x_mat = cp.linalg.eigh(a_mat)
        x_mat = (x_mat * np.sqrt(.5)).T.reshape(-1, nocc, nvir).get()
        td.e = w.get()
        td.xy = [(x, 0) for x in x_mat]
        td.converged = [True] * len(td.e)
        STATS['gpu_eigh'] += 1
        return td

    from cupyx.scipy.linalg import solve_triangular
    amb = a_mat - b_mat
    apb = a_mat + b_mat
    del a_mat, b_mat
    try:
        l_mat = cp.linalg.cholesky(amb)
        bad = not bool(cp.isfinite(l_mat).all())
    except np.linalg.LinAlgError:
        bad = True
    if bad:
        # A - B not positive definite: GPU4PySCF's non-Hermitian fallback
        STATS['stock_eigh'] += 1
        log.warn('fastc6: A-B not positive definite, using GPU4PySCF\'s path')
        return C6._solve_full_spectrum_orig(td)
    del amb
    h_eff = l_mat.T @ (apb @ l_mat)
    del apb
    w2, v = cp.linalg.eigh(h_eff)
    del h_eff
    mask = w2 > 1e-6
    w_pos = cp.sqrt(w2[mask])
    v_pos = v[:, mask]
    del v
    d_vecs = solve_triangular(l_mat.T, v_pos, lower=False)
    z_vecs = (l_mat @ v_pos) / w_pos[None, :]
    x_all = .5 * (z_vecs + d_vecs)
    y_all = .5 * (z_vecs - d_vecs)
    norm_diff = (x_all * x_all).sum(axis=0) - (y_all * y_all).sum(axis=0)
    scale = cp.where(cp.abs(norm_diff) < 1e-9, 1.,
                     cp.sqrt(.5 / cp.abs(norm_diff)))
    x_all = (x_all * scale).T.reshape(-1, nocc, nvir).get()
    y_all = (y_all * scale).T.reshape(-1, nocc, nvir).get()
    td.e = w_pos.get()
    td.xy = list(zip(x_all, y_all))
    td.converged = [True] * len(td.e)
    STATS['gpu_eigh'] += 1
    return td


def calc_c6(td_a, td_b, n_grid=20):
    if td_b is td_a:
        # the molecule with itself: one spectrum, solved once
        log = logger.new_logger(td_a)
        x, w_leg = np.polynomial.legendre.leggauss(n_grid)
        w0 = 0.5
        freqs_im = w0 * (1 + x) / (1 - x)
        weights = w_leg * w0 * 2 / ((1 - x)**2)
        C6._solve_full_spectrum(td_a)
        f_osc = td_a.oscillator_strength()
        e_exc = td_a.e
        alpha = np.sum(f_osc[:, None] /
                       (e_exc[:, None]**2 + freqs_im[None, :]**2), axis=0)
        c6_val = (3.0 / np.pi) * np.sum(alpha * alpha * weights)
        STATS['reused_spectrum'] += 1
        log.info(f'Calculated C6 coefficient: {c6_val:.6f} a.u.')
        return float(c6_val.real)
    return C6._calc_c6_orig(td_a, td_b, n_grid)


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    C6._solve_full_spectrum_orig = C6._solve_full_spectrum
    C6._calc_c6_orig = C6.calc_c6
    C6._solve_full_spectrum = _solve_full_spectrum
    C6.calc_c6 = calc_c6
    _PATCHED = True


apply_patch()
