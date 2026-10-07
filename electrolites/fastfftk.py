"""FFTDF k-point exchange with time reversal (J3).

``pbc/df/fft_jk.py:get_k_kpts`` -- the exchange of every k-point hybrid on
the default (FFTDF) back end, 94-97 % of such an SCF
(``docs/KERNEL_TARGETS_1.8.1.md`` §10.11) -- loops over every pair of
k-points ``(k1, k2)`` and, inside, over the AOs of ``k1`` in blocks of 32:
pair densities, an FFT, the Coulomb kernel, an inverse FFT, a contraction.

One change, the arithmetic of each pair otherwise GPU4PySCF's -- **time
reversal**.  With real basis functions the Bloch AOs satisfy
  ``phi_{-k} = conj(phi_k)``; when the k-point set is closed under ``k -> -k``
  (modulo a reciprocal lattice vector, which leaves a Bloch sum unchanged) and
  every density satisfies ``dm(-k) = conj(dm(k))``, ``K(-k) = conj(K(k))``.
  Only one ``k1`` of each pair ``{k, -k}`` is built -- half the pair loop on
  any inversion-symmetric Monkhorst-Pack set.  Both conditions are checked
  (the densities to 1e-10 of their largest element); otherwise every ``k1``
  is built.  Not used with ``kpts_band``.  A set made only of
  time-reversal-invariant momenta (every Gamma-centred even mesh up to
  2x2x2, where ``k = -k`` for all ``k``) gains nothing.

Sizing the AO block from free memory instead of GPU4PySCF's 32 was measured
and dropped: the per-pair FFTs are already large enough (1.00x).

``FASTFFTK_OFF=1`` keeps GPU4PySCF's function; ``FASTFFTK_NO_TR=1`` is the
ablation, ``FASTFFTK_BLOCK=<n>`` sets the AO block.  ``docs/README_fastfftk.md``.
"""
import os

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.df.df_jk import _format_kpts_band

from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.df import fft_jk as F

_OFF = bool(os.environ.get('FASTFFTK_OFF'))
_NO_TR = bool(os.environ.get('FASTFFTK_NO_TR'))
_BLOCK = int(os.environ.get('FASTFFTK_BLOCK', 0))
STATS = {'calls': 0, 'time_reversal': 0, 'k1_built': 0, 'k1_total': 0,
         'blksize': 0}


def stats():
    return dict(STATS)


def _partners(cell, kpts):
    """``j`` with ``kpts[j] = -kpts[i]`` modulo the reciprocal lattice, for
    every ``i``; None if some ``-k`` is missing."""
    frac = np.asarray(kpts).dot(cell.lattice_vectors().T) / (2 * np.pi)
    out = []
    for f in frac:
        d = frac + f                     # kpts[j] + kpts[i] = G ?
        d -= np.round(d)
        j = np.where(abs(d).max(axis=1) < 1e-8)[0]
        if len(j) == 0:
            return None
        out.append(int(j[0]))
    return out


def _time_reversal_ok(dms, partner):
    scale = float(abs(dms).max()) or 1.
    for k, j in enumerate(partner):
        if float(abs(dms[:, j] - dms[:, k].conj()).max()) > 1e-10 * scale:
            return False
    return True


def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=None, kpts_band=None,
               exxdiv=None):
    STATS['calls'] += 1
    cell = mydf.cell
    mesh = mydf.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1
    coords = mydf.grids.coords
    ngrids = coords.shape[0]

    if getattr(dm_kpts, 'mo_coeff', None) is not None:
        mo_coeff = dm_kpts.mo_coeff
        mo_occ = dm_kpts.mo_occ
    else:
        mo_coeff = None

    ni = mydf._numint
    if kpts is None:
        kpts = np.zeros((1, 3))
    else:
        kpts = np.asarray(kpts).reshape(-1, 3)

    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = F._format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    weight = 1. / nkpts * (cell.vol / ngrids)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if is_zero(kpts_band) and is_zero(kpts):
        vk_kpts = cp.zeros((nset, nband, nao, nao), dtype=dms.dtype)
    else:
        vk_kpts = cp.zeros((nset, nband, nao, nao), dtype=np.complex128)

    # time reversal: which k1 to build, and from which partner the rest
    build = list(range(nband))
    partner = None
    if (not _NO_TR and input_band is None and nkpts > 1
            and vk_kpts.dtype == np.complex128):
        partner = _partners(cell, kpts)
        if partner is not None and _time_reversal_ok(dms, partner):
            build = [k for k in range(nkpts) if partner[k] >= k]
            STATS['time_reversal'] += 1
        else:
            partner = None
    STATS['k1_built'] += len(build)
    STATS['k1_total'] += nband

    ao2_kpts = ni.eval_ao(cell, coords, kpts=kpts)
    if input_band is None:
        ao1_kpts = ao2_kpts
    else:
        ao1_kpts = ni.eval_ao(cell, coords, kpts=kpts_band)

    if mo_coeff is not None and nset == 1:
        mo2_kpts = [
            ao.dot(mo[:, occ > 0] * occ[occ > 0]**.5)
            for occ, mo, ao in zip(mo_occ, mo_coeff, ao2_kpts)]
        ao2_kpts = mo2_kpts
    else:
        mo2_kpts = None

    vR_dm = cp.empty((nset, nao, ngrids), dtype=vk_kpts.dtype)
    blksize = _BLOCK or 32
    STATS['blksize'] = blksize

    for k2, ao2 in enumerate(ao2_kpts):
        ao2T = ao2.T
        kpt2 = kpts[k2]
        naoj = ao2.shape[1]
        if mo2_kpts is None:
            ao_dms = [dms[i, k2].dot(ao2T.conj()) for i in range(nset)]
        else:
            ao_dms = [ao2T.conj()]

        for k1 in build:
            ao1 = ao1_kpts[k1]
            ao1T = ao1.T
            kpt1 = kpts_band[k1]
            coulG = tools.get_coulG(cell, kpt2 - kpt1, exxdiv, mesh=mesh,
                                    kpts=kpts)
            if is_zero(kpt1 - kpt2):
                expmikr = cp.array(1.)
            else:
                expmikr = cp.exp(-1j * coords.dot(cp.asarray(kpt2 - kpt1)))

            for p0, p1 in lib.prange(0, nao, blksize):
                rho1 = contract('ig,jg->ijg', ao1T[p0:p1].conj() * expmikr,
                                ao2T)
                vG = tools.fft(rho1.reshape(-1, ngrids), mesh)
                rho1 = None
                vG *= coulG
                vR = tools.ifft(vG, mesh).reshape(p1 - p0, naoj, ngrids)
                vG = None
                if vk_kpts.dtype == np.double:
                    vR = vR.real
                for i in range(nset):
                    vR_dm[i, p0:p1] = contract('ijg,jg->ig', vR, ao_dms[i])
                vR = None
            vR_dm *= expmikr.conj()

            for i in range(nset):
                vk_kpts[i, k1] += weight * vR_dm[i].dot(ao1)

    if partner is not None:
        for k in range(nkpts):
            if partner[k] < k:
                vk_kpts[:, k] = vk_kpts[:, partner[k]].conj()

    return F._format_jks(vk_kpts, dm_kpts, input_band, kpts)


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    F._get_k_kpts_orig = F.get_k_kpts
    F.get_k_kpts = get_k_kpts
    _PATCHED = True


apply_patch()
