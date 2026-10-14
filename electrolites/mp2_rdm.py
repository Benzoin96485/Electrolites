"""MP2 one-particle density and frozen natural orbitals on the GPU
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` P5).

GPU4PySCF 1.8.1's MP2 classes (conventional ``mp.mp2.RMP2`` and
density-fitted ``mp.dfmp2.DFMP2``) compute the energy and, on request, the
amplitudes, but have no ``make_rdm1`` or ``make_fno``.  These are PySCF's
(``pyscf/mp/mp2.py``): the unrelaxed spin-traced 1-RDM from

    D_vir[b,a] =  sum_{ijc} t2[i,j,c,a] (2 t2[i,j,c,b] - t2[i,j,b,c])
    D_occ[x,y] = -sum_{iab} t2[i,x,a,b] (2 t2[i,y,a,b] - t2[i,y,b,a])

-- two GEMMs over the device-resident amplitudes instead of PySCF's loop
over i -- and the frozen natural orbitals (FNO) from the virtual block, which
feed a cheaper CCSD(T) (``frozen`` + ``mo_coeff`` of the returned pair).

If the object holds no amplitudes, the MP2 is re-run with ``with_t2=True``.
Amplitudes stored in single precision (DF-MP2's ``fp_type='FP32'``) are
widened before the contraction.

    import electrolites
    electrolites.enable('mp2_rdm')
    pt = gpu4pyscf.mp.dfmp2.DFMP2(mf).run(with_t2=True)
    frozen, no_coeff = pt.make_fno(thresh=1e-5)
"""
import numpy as np
import cupy as cp

from pyscf.cc import ccsd_rdm
from pyscf.lib import logger

from gpu4pyscf.mp import mp2

from electrolites.features import attach


def _t2(mp):
    t2 = mp.t2
    if t2 is None:
        logger.info(mp, 'make_rdm1: no stored amplitudes; re-running MP2 '
                    'with with_t2=True')
        mp.kernel(with_t2=True)
        t2 = mp.t2
    t2 = cp.asarray(t2)
    if t2.dtype != np.float64:
        t2 = t2.astype(np.float64)
    return t2


def _gamma1_intermediates(mp, t2=None):
    t2 = _t2(mp) if t2 is None else cp.asarray(t2, dtype=np.float64)
    no, _, nv, _ = t2.shape
    tt = 2 * t2 - t2.transpose(0, 1, 3, 2)
    dvir = tt.reshape(-1, nv).T @ t2.reshape(-1, nv)             # [b, a]
    a = cp.ascontiguousarray(t2.transpose(1, 0, 2, 3)).reshape(no, -1)
    b = cp.ascontiguousarray(tt.transpose(1, 0, 2, 3)).reshape(no, -1)
    docc = a @ b.T
    return -docc, dvir


class _HostView:
    """What ``ccsd_rdm._make_rdm1`` reads, on the host."""

    def __init__(self, mp):
        self.frozen = mp.frozen
        self.mo_occ = cp.asnumpy(mp.mo_occ)
        self.get_frozen_mask = lambda: cp.asnumpy(mp.get_frozen_mask())


def make_rdm1(mp, t2=None, eris=None, ao_repr=False, with_frozen=True):
    """PySCF's ``mp2.make_rdm1`` (unrelaxed, spin-traced); ``eris`` unused."""
    doo, dvv = _gamma1_intermediates(mp, t2)
    doo, dvv = cp.asnumpy(doo), cp.asnumpy(dvv)
    nocc, nvir = doo.shape[0], dvv.shape[0]
    dov = np.zeros((nocc, nvir))
    dm1 = ccsd_rdm._make_rdm1(_HostView(mp), (doo, dov, dov.T, dvv),
                              with_frozen=with_frozen, ao_repr=False)
    dm1 = cp.asarray(dm1)
    if ao_repr:
        mo = cp.asarray(mp.mo_coeff)
        dm1 = mo @ dm1 @ mo.T
    return dm1


def make_fno(mp, thresh=1e-6, pct_occ=None, nvir_act=None, t2=None):
    """PySCF's ``mp2.make_fno``: (frozen indices, semicanonical NO
    coefficients in the AO basis, on the device)."""
    mf = mp._scf
    dm = cp.asnumpy(make_rdm1(mp, t2=t2, with_frozen=False))
    nocc = mp.nocc
    nvir = mp.nmo - nocc
    n, v = np.linalg.eigh(dm[nocc:, nocc:])
    idx = np.argsort(n)[::-1]
    n, v = n[idx], v[:, idx]
    if nvir_act is None:
        if pct_occ is None:
            nkeep = np.count_nonzero(n > thresh)
        else:
            cs = np.cumsum(n / np.sum(n))
            nkeep = np.count_nonzero([c <= pct_occ or np.isclose(c, pct_occ)
                                      for c in cs])
    else:
        nkeep = min(nvir, nvir_act)
    maskact = cp.asnumpy(mp.get_frozen_mask())
    maskocc = cp.asnumpy(mp.mo_occ) > 1e-6
    masks = [maskocc & ~maskact, maskocc & maskact,
             ~maskocc & maskact, ~maskocc & ~maskact]
    mo_e = cp.asnumpy(mf.mo_energy)
    mo_c = cp.asarray(mf.mo_coeff)
    _, _, moevir, _ = [mo_e[m] for m in masks]
    oocc0, oocc, ovir, ovir0 = [mo_c[:, cp.asarray(m)] for m in masks]
    fvv_no = v.T @ np.diag(moevir) @ v
    _, vc = np.linalg.eigh(fvv_no[:nkeep, :nkeep])
    ovir_act = ovir @ cp.asarray(v[:, :nkeep] @ vc)
    ovir_frz = ovir @ cp.asarray(v[:, nkeep:])
    comp = (oocc0, oocc, ovir_act, ovir_frz, ovir0)
    no_coeff = cp.hstack(comp)
    loc = np.cumsum([0] + [x.shape[1] for x in comp]).astype(int)
    frozen = np.hstack((np.arange(loc[0], loc[1]),
                        np.arange(loc[3], loc[5]))).astype(int)
    return frozen, no_coeff


attach(mp2.MP2Base, 'make_rdm1', make_rdm1)
attach(mp2.MP2Base, 'make_fno', make_fno)
