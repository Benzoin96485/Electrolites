"""k-point RMP2 on the GPU from GPU4PySCF's GDF cderi.

PySCF's ``pbc.mp.kmp2`` builds L_ov for every (k_i, k_a) on the host from
the stored cderi, then loops over (k_i, k_j, k_a) in Python with one
``einsum('Lia,Ljb->iajb')`` per triple -- nk^3 host contractions of
naux.o^2.v^2.  GPU4PySCF has no periodic correlation method.

Here the GDF tensor is read the way GPU4PySCF's own J/K reads it
(``GDF.loop``: per momentum transfer q, the dense L(k, k+q) for every k,
already on the device), transformed to L_ov(k_i, k_a) on the device, and
for each (k_i, k_j) the integrals of every k_a are one batched complex
GEMM over k_a; the direct and exchange energies are reduced on the
device.  Same expression as PySCF's (insulators with the same number of
occupied orbitals at every k; no frozen orbitals).

    from electrolites import kmp2_gpu
    e_corr = kmp2_gpu.kernel(gpu_krhf)      # GPU KRHF with a GPU GDF
"""
import numpy as np
import cupy as cp

from pyscf.lib import logger


def _lov(mf, nocc):
    """L_ov[ki, ka] = C_o(ki)^H L(ki, ka) C_v(ka), (nk, nk, naux, o, v), on
    the device.  GDF stores one of each (q, -q) pair of momentum transfers;
    L(ka, ki) = L(ki, ka)^H gives the other (as GPU4PySCF's K build)."""
    from pyscf import lib
    from gpu4pyscf.pbc.lib.kpts_helper import kk_adapted_iter
    mydf = mf.with_df
    if mydf._cderi is None:
        mydf.build()
    nk = len(mf.kpts)
    C = cp.asarray(np.asarray([cp.asnumpy(c) for c in mf.mo_coeff]))
    Co = C[:, :, :nocc]
    Cv = C[:, :, nocc:]
    nv = Cv.shape[2]
    nao = C.shape[1]
    naux = mydf.get_naoaux()
    conj = {}
    for kp, kp_conj, ki_idx, kj_idx in kk_adapted_iter(mydf.kmesh):
        conj[kp] = (kp_conj, np.asarray(kj_idx))
    out = cp.zeros((nk, nk, naux, nocc, nv), dtype=np.complex128)
    blk = max(1, min(naux, int(2e9 // (nk * nao ** 2 * 16))))
    blocks = [(kp, p0, p1) for kp in mydf._cderi
              for p0, p1 in lib.prange(0, mydf._cderi[kp].shape[0], blk)]
    for (kp, p0, p1), (kq, L, sign) in zip(
            blocks, mydf.loop(blk, aux_iter=iter(blocks))):
        assert kq == kp
        if sign != 1:
            raise NotImplementedError('kMP2 on the GPU: 3D cells only')
        kp_conj, kj_idx = conj[kp]
        ki_idx = np.arange(nk)
        # L[n] = L(ki=n, kj=kj_idx[n]), (naux_blk, nao, nao)
        x = cp.einsum('npi,nLpq->nLiq', Co.conj(), L)
        out[ki_idx, kj_idx, p0:p1] = cp.einsum('nLiq,nqa->nLia', x,
                                               Cv[kj_idx])
        if kp != kp_conj:
            # L(kj, ki)_{mu nu} = conj(L(ki, kj)_{nu mu})
            x = cp.einsum('npi,nLqp->nLiq', Co[kj_idx].conj(), L.conj())
            out[kj_idx, ki_idx, p0:p1] = cp.einsum('nLiq,nqa->nLia', x, Cv)
    return out


def kernel(mf, verbose=None):
    """KRHF MP2 correlation energy per cell (PySCF's kmp2 expression)."""
    log = logger.new_logger(mf, verbose)
    kpts = mf.kpts
    nk = len(kpts)
    mo_occ = [cp.asnumpy(o) for o in mf.mo_occ]
    nocc = int((mo_occ[0] > 0).sum())
    if any(int((o > 0).sum()) != nocc for o in mo_occ):
        raise NotImplementedError('kMP2 on the GPU: same nocc at every k')
    Lov = _lov(mf, nocc)
    e = cp.asarray(np.asarray([cp.asnumpy(x) for x in mf.mo_energy]))
    eo, ev = e[:, :nocc], e[:, nocc:]
    from pyscf.pbc.lib import kpts_helper
    kconserv = kpts_helper.get_kconserv(mf.cell, kpts)
    ka = np.arange(nk)
    e_d = e_x = 0.
    for ki in range(nk):
        for kj in range(nk):
            kb = kconserv[ki, ka, kj]
            # (ia|jb) for every ka: [ka, i, j, a, b]
            A = Lov[ki, ka]                                  # (ka, L, o, v)
            B = Lov[kj, kb]                                  # (ka, L, o, v)
            g = cp.einsum('kLia,kLjb->kijab', A, B) / nk
            eia = eo[ki][:, None] - ev[ka][:, None, :]       # (ka, o, v)
            ejb = eo[kj][:, None] - ev[kb][:, None, :]
            D = eia[:, :, None, :, None] + ejb[:, None, :, None, :]
            t2 = (g / D).conj()
            e_d += float(cp.einsum('kijab,kijab', t2, g).real) * 2
            # exchange: oovv[kb] for the same (ki, kj), indices a<->b
            e_x -= float(cp.einsum('kijab,kijba', t2, g[kb]).real)
    emp2 = (e_d + e_x) / nk
    log.note('KMP2 energy (GPU) = %.15g', emp2)
    return emp2
