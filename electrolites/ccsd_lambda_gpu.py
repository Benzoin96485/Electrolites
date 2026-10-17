"""CCSD Λ (lambda) equations on the GPU for a closed-shell GPU4PySCF CCSD.

GPU4PySCF has no Λ solver (``CCSD.solve_lambda`` is PySCF's, on the host,
and reads an ``eris`` with ``ovvv`` GPU4PySCF's integral object does not
carry).  This is PySCF's ``ccsd_lambda`` -- the same intermediates
(``make_intermediates``) and the same update (``update_lambda``), term for
term -- with every array on the device and no blocking: the integrals come
from ``fastmp2`` (as ``eom_gpu``'s), the (ac|bd) ladder on λ2 runs
AO-direct through ``fastccsd``'s integral blocks and cache, and DIIS is
GPU4PySCF's device DIIS on the (λ1, λ2) vector.

    from electrolites import ccsd_lambda_gpu
    mycc = gpu4pyscf.cc.ccsd_incore.CCSD(mf).run()
    conv, l1, l2 = ccsd_lambda_gpu.kernel(mycc)
"""
import numpy as np
import cupy as cp

from pyscf.lib import logger

_ein = cp.einsum


def make_intermediates(imds):
    """PySCF ``ccsd_lambda.make_intermediates``, unblocked, on the device."""
    t1, t2 = imds.t1, imds.t2
    ovvv, ovoo, ovvo = imds.ovvv, imds.ovoo, imds.ovvo
    oovv, oooo = imds.oovv, imds.oooo
    foo, fov, fvv = imds._foo_fov_fvv()
    w1 = fvv - _ein('ja,jb->ba', fov, t1)
    w2 = foo + _ein('ib,jb->ij', fov, t1)
    w3 = _ein('kc,jkbc->bj', fov, t2) * 2 + fov.T
    w3 -= _ein('kc,kjbc->bj', fov, t2)
    w3 += t1.T @ fov @ t1.T                       # 'kc,kb,jc->bj'
    w4 = fov.copy()

    vvov = ovvv.transpose(2, 3, 0, 1)
    w1 += _ein('jcba,jc->ba', ovvv, t1 * 2)
    w1 -= _ein('jabc,jc->ba', ovvv, t1)
    theta = t2 * 2 - t2.transpose(1, 0, 2, 3)
    w3 += _ein('jkcd,kdcb->bj', theta, ovvv)
    wVOov = _ein('jbcd,kd->bjkc', ovvv, t1)
    wvOOv = _ein('cbjd,kd->cjkb', vvov, -t1)
    g2vovv = vvov.transpose(0, 2, 1, 3) * 2 - vvov.transpose(0, 2, 3, 1)
    tau = t2 + _ein('ia,jb->ijab', t1, t1)
    wvooo = _ein('cibd,jkbd->ckij', g2vovv, tau)
    g2vovv = None
    wvvov = _ein('jabd,jkcd->abkc', ovvv, t2) * -1.5
    wvvov += vvov.transpose(0, 3, 2, 1) * 2
    wvvov -= vvov
    g2vvov = vvov * 2 - ovvv.transpose(1, 2, 0, 3)
    theta2 = t2 * 2 - t2.transpose(0, 1, 3, 2)
    vackb = _ein('acjd,kjbd->ackb', g2vvov, theta2)
    wvvov += vackb.transpose(0, 3, 2, 1)
    wvvov -= vackb * .5
    vackb = g2vvov = None

    w2 += _ein('kbij,kb->ij', ovoo, t1) * 2
    w2 -= _ein('ibkj,kb->ij', ovoo, t1)
    theta3 = t2.transpose(1, 0, 2, 3) * 2 - t2
    w3 -= _ein('lckj,klcb->bj', ovoo, theta3)
    tmp = _ein('lc,jcik->ijkl', t1, ovoo)
    woooo = tmp + tmp.transpose(1, 0, 3, 2)
    wvOOv += _ein('lbjk,lc->bjkc', ovoo, t1)
    wVOov -= _ein('jbkl,lc->bjkc', ovoo, t1)
    wvooo += ovoo.transpose(1, 3, 2, 0) * 2
    wvooo -= ovoo.transpose(1, 0, 2, 3)
    wvooo -= _ein('klbc,iblj->ckij', t2, ovoo * 1.5)
    g2ovoo = ovoo * 2 - ovoo.transpose(2, 1, 0, 3)
    vcjik = _ein('jlcb,lbki->cjki', theta, g2ovoo)
    wvooo += vcjik.transpose(0, 3, 2, 1)
    wvooo -= vcjik * .5
    vcjik = None

    voov = ovvo.transpose(1, 0, 3, 2)
    woooo += _ein('cijd,klcd->ijkl', voov, tau)
    g2voov = voov * 2 - voov.transpose(0, 2, 1, 3)
    tmpw4 = _ein('ckld,ld->kc', g2voov, t1)
    w1 -= _ein('ckja,kjcb->ba', g2voov, t2)
    w1 -= _ein('ja,jb->ba', tmpw4, t1)
    w2 += _ein('jkbc,bikc->ij', t2, g2voov)
    w2 += _ein('ib,jb->ij', tmpw4, t1)
    w3 += t1.T @ tmpw4 @ t1.T
    w4 += tmpw4
    x = _ein('bljd,kd->bljk', voov, t1)
    wvOOv += _ein('bljk,lc->bjkc', x, t1)
    x = _ein('bjld,kd->bjlk', voov, t1)
    wVOov -= _ein('bjlk,lc->bjkc', x, t1)
    x = None
    VOov = _ein('bjld,klcd->bjkc', g2voov, t2)
    VOov -= _ein('bjld,kldc->bjkc', voov, t2)
    VOov += voov
    vOOv = _ein('bljd,kldc->bjkc', voov, t2)
    vOOv -= oovv.transpose(2, 1, 0, 3)
    wVOov += VOov
    wvOOv += vOOv
    ov1 = vOOv * 2 + VOov
    ov2 = VOov * 2 + vOOv
    vOOv = VOov = None
    wvooo -= _ein('jb,bikc->ckij', t1, ov1)
    wvooo += _ein('kb,bijc->ckij', t1, ov2)
    w3 += _ein('ckjb,kc->bj', ov2, t1)
    wvvov += _ein('ajkc,jb->abkc', ov1, t1)
    wvvov -= _ein('ajkb,jc->abkc', ov2, t1)
    wvvov += _ein('laki,klbc->abic', g2ovoo, tau)
    ov1 = ov2 = g2ovoo = None
    woooo += oooo.transpose(0, 2, 1, 3)
    w3 += _ein('bc,jc->bj', w1, t1)
    w3 -= _ein('kj,kb->bj', w2, t1)
    imds.w1, imds.w2, imds.w3, imds.w4 = w1, w2, w3, w4
    imds.woooo, imds.wvooo, imds.wVOov = woooo, wvooo, wVOov
    imds.wvOOv, imds.wvvov = wvOOv, wvvov
    return imds


def _ladder_oovv(imds, mycc, x):
    """sum_cd (ac|bd) x[i,j,c,d] AO-direct (PySCF ``_add_vvvv``)."""
    from electrolites.eom_gpu import _ladder_setup
    no, _, nv, _ = x.shape
    if not hasattr(imds, 'orbv_s'):
        mask = mycc.get_frozen_mask()
        mo = cp.asarray(mycc.mo_coeff)[:, cp.asarray(mask)]
        imds.orbv_s = _ladder_setup(mycc).coeff @ mo[:, no:]
    from electrolites.eom_gpu import _ladder_pairsym
    return _ladder_pairsym(mycc, imds.orbv_s, x)


def update_lambda(imds, mycc, l1, l2):
    """PySCF ``ccsd_lambda.update_lambda``, unblocked, on the device."""
    t1, t2 = imds.t1, imds.t2
    ovvv, ovoo, ovvo, oovv = imds.ovvv, imds.ovoo, imds.ovvo, imds.oovv
    no = imds.nocc
    fov = imds.fock[:no, no:]
    e = cp.diag(imds.fock)
    mo_e_o = e[:no]
    mo_e_v = e[no:] + mycc.level_shift

    theta = t2 * 2 - t2.transpose(0, 1, 3, 2)
    mba = _ein('klca,klcb->ba', l2, theta)
    mij = _ein('ikcd,jkcd->ij', l2, theta)
    theta = None
    mba1 = _ein('jc,jb->bc', l1, t1) + mba
    mij1 = _ein('kb,jb->kj', l1, t1) + mij
    mia1 = t1 + _ein('kc,jkbc->jb', l1, t2) * 2
    mia1 -= _ein('kc,jkcb->jb', l1, t2)
    mia1 -= t1 @ l1.T @ t1
    mia1 -= _ein('bd,jd->jb', mba, t1)
    mia1 -= _ein('lj,lb->jb', mij, t1)

    l2new = _ladder_oovv(imds, mycc, l2)
    l1new = _ein('ijab,jb->ia', l2new, t1) * 2
    l1new -= _ein('jiab,jb->ia', l2new, t1)
    l2new *= .5

    w1 = imds.w1 - cp.diag(mo_e_v)
    w2 = imds.w2 - cp.diag(mo_e_o)
    l1new += fov
    l1new += _ein('ib,ba->ia', l1, w1)
    l1new -= _ein('ja,ij->ia', l1, w2)
    l1new -= _ein('ik,ka->ia', mij, imds.w4)
    l1new -= _ein('ca,ic->ia', mba, imds.w4)
    l1new += _ein('ijab,bj->ia', l2, imds.w3) * 2
    l1new -= _ein('ijba,bj->ia', l2, imds.w3)
    l2new += _ein('ia,jb->ijab', l1, imds.w4)
    l2new += _ein('jibc,ca->jiba', l2, w1)
    l2new -= _ein('jk,kiba->jiba', w2, l2)

    l1new -= _ein('iajk,kj->ia', ovoo, mij1) * 2
    l1new += _ein('jaik,kj->ia', ovoo, mij1)
    l2new -= _ein('jbki,ka->jiba', ovoo, l1)

    tau = t2 + _ein('ia,jb->ijab', t1, t1)
    l2tau = _ein('ijcd,klcd->ijkl', l2, tau)
    tau = None
    l2t1 = _ein('jidc,kc->ijkd', l2, t1)

    l1new -= _ein('jb,jiab->ia', l1, oovv)
    l1new += _ein('iabc,bc->ia', ovvv, mba1) * 2
    l1new -= _ein('ibca,bc->ia', ovvv, mba1)
    l2new += _ein('jbac,ic->jiba', ovvv, l1)
    m4 = _ein('ijkd,kadb->ijab', l2t1, ovvv)
    l2new -= m4
    l1new -= _ein('ijab,jb->ia', m4, t1) * 2
    l1new -= _ein('ijab,ia->jb', m4, t1) * 2
    l1new += _ein('jiab,jb->ia', m4, t1)
    l1new += _ein('jiab,ia->jb', m4, t1)
    m4 = None

    voov = ovvo.transpose(1, 0, 3, 2)
    l1new += _ein('jb,aijb->ia', l1, voov) * 2
    l2new += voov.transpose(1, 2, 0, 3) * .5
    l2new -= _ein('bjic,ca->jiba', voov, mba1)
    l2new -= _ein('bjka,ik->jiba', voov, mij1)
    l1new += _ein('aijb,jb->ia', voov, mia1) * 2
    l1new -= _ein('bija,jb->ia', voov, mia1)
    m4 = _ein('ijkl,aklb->ijab', l2tau, voov)
    l2new += m4 * .5
    l1new += _ein('ijab,jb->ia', m4, t1) * 2
    l1new -= _ein('ijba,jb->ia', m4, t1)
    m4 = None

    l1new -= _ein('ckij,jkca->ia', imds.wvooo, l2)
    l1new += _ein('abkc,kibc->ia', imds.wvvov, l2)
    tmp_voov = imds.wVOov * 2 + imds.wvOOv
    tmp = l2.transpose(0, 2, 1, 3) - l2.transpose(0, 3, 1, 2) * .5
    l2new += _ein('iakc,bjkc->jiba', tmp, tmp_voov)
    tmp = tmp_voov = None
    tmp = _ein('jkca,bikc->jiba', l2, imds.wvOOv)
    l2new += tmp
    l2new += tmp.transpose(1, 0, 2, 3) * .5
    tmp = None

    m3 = _ein('ijkl,klab->ijab', imds.woooo, l2)
    l2new += m3 * .5
    l1new += _ein('ijab,jb->ia', m3, t1) * 2
    l1new -= _ein('ijba,jb->ia', m3, t1)
    m3 = None

    eia = mo_e_o[:, None] - mo_e_v[None, :]
    l1new /= eia
    l2new = l2new + l2new.transpose(1, 0, 3, 2)
    l2new /= eia[:, None, :, None] + eia[None, :, None, :]
    return l1new, l2new


def _kernel_on_pool(*args, **kwargs):
    from electrolites.eom_gpu import _on_pool
    return _on_pool(_kernel)(*args, **kwargs)


def kernel(*args, **kwargs):
    """Solve the Λ equations; returns (converged, l1, l2) as host arrays
    (and sets ``mycc.l1``, ``mycc.l2``)."""
    return _kernel_on_pool(*args, **kwargs)


def _kernel(mycc, t1=None, t2=None, l1=None, l2=None, max_cycle=50,
           tol=1e-8, verbose=None):
    """Solve the Λ equations; returns (converged, l1, l2) as host arrays
    (and sets ``mycc.l1``, ``mycc.l2``)."""
    from electrolites.eom_gpu import _Imds
    from gpu4pyscf.lib.diis import DIIS
    log = logger.new_logger(mycc, verbose)
    imds = _Imds(mycc)
    if t1 is not None:
        imds.t1 = cp.asarray(t1)
    if t2 is not None:
        imds.t2 = cp.asarray(t2)
    make_intermediates(imds)
    l1 = imds.t1.copy() if l1 is None else cp.asarray(l1)
    l2 = imds.t2.copy() if l2 is None else cp.asarray(l2)
    no, nv = l1.shape
    adiis = DIIS()
    adiis.space = getattr(mycc, 'diis_space', 6)
    conv = False
    for it in range(max_cycle):
        l1new, l2new = update_lambda(imds, mycc, l1, l2)
        normt = float(cp.sqrt(cp.linalg.norm(l1new - l1) ** 2
                              + cp.linalg.norm(l2new - l2) ** 2))
        vec = adiis.update(cp.hstack([l1new.ravel(), l2new.ravel()]))
        l1 = vec[:no * nv].reshape(no, nv)
        l2 = vec[no * nv:].reshape(no, no, nv, nv)
        log.info('lambda cycle = %d  norm(lambda1,lambda2) = %.6g', it + 1,
                 normt)
        if normt < tol:
            conv = True
            break
    mycc.l1, mycc.l2 = cp.asnumpy(l1), cp.asnumpy(l2)
    return conv, mycc.l1, mycc.l2


def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None):
    """``CCSD.solve_lambda`` on the device (``eris`` is not used)."""
    self.converged_lambda, self.l1, self.l2 = kernel(
        self, t1, t2, l1, l2, max_cycle=self.max_cycle,
        tol=self.conv_tol_normt, verbose=self.verbose)
    return self.l1, self.l2


from gpu4pyscf.cc import ccsd_incore as _ci                   # noqa: E402
from electrolites.features import attach as _attach           # noqa: E402

if _ci.CCSDBase.solve_lambda is NotImplemented:
    _attach(_ci.CCSDBase, 'solve_lambda', solve_lambda)
