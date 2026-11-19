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

    # every o v^3 term in slices (``eom_gpu._ein_o`` or explicit loops):
    # the unsliced build held five o v^3 arrays at once (ovvv, its
    # transposed copies, g2vovv, g2vvov, vackb) and ran an 80 GB device out
    # of memory at o 66, v 288
    from electrolites.eom_gpu import _ein_o, _vslice
    no, nv = t1.shape
    w1 += _ein_o('jcba,jc->ba', ovvv, t1 * 2, 0)
    w1 -= _ein_o('jabc,jc->ba', ovvv, t1, 0)
    theta = t2 * 2 - t2.transpose(1, 0, 2, 3)
    w3 += _ein_o('jkcd,kdcb->bj', theta, ovvv, 1, big=1)
    wVOov = _ein_o('jbcd,kd->bjkc', ovvv, t1, None, out_axis=1)
    # vvov[c,b,j,d] = ovvv[j,d,c,b]
    wvOOv = _ein_o('jdcb,kd->cjkb', ovvv, -t1, None, out_axis=1)
    tau = t2 + _ein('ia,jb->ijab', t1, t1)
    # wvooo[c,k,i,j] = sum_bd (2 ovvv[i,d,c,b] - ovvv[i,b,c,d]) tau[j,k,b,d]
    wvooo = cp.empty((nv, no, no, no))
    for i0, i1 in _vslice(no, nv ** 3 * 8 * 4):
        ov = ovvv[i0:i1]
        g = ov.transpose(0, 2, 3, 1) * 2 - ov.transpose(0, 2, 1, 3)  # icbd
        wvooo[:, :, i0:i1] = _ein('icbd,jkbd->ckij', g, tau)
        g = ov = None
    # wvvov[a,b,k,c] = -1.5 sum_jd ovvv[j,a,b,d] t2[j,k,c,d]
    #   + 2 vvov[a,c,k,b] - vvov[a,b,k,c] + vackb[a,c,k,b] - vackb[a,b,k,c]/2
    # vvov[a,b,k,c] = ovvv[k,c,a,b]; vackb[a,c,k,b] = sum_jd g2vvov[a,c,j,d]
    # theta2[k,j,b,d], g2vvov[a,c,j,d] = 2 ovvv[j,d,a,c] - ovvv[j,a,c,d]
    theta2 = t2 * 2 - t2.transpose(0, 1, 3, 2)
    wvvov = cp.empty((nv, nv, no, nv))
    for a0, a1 in _vslice(nv, no * nv * nv * 8 * 5):
        out = wvvov[a0:a1]
        out[...] = _ein('jabd,jkcd->abkc', ovvv[:, a0:a1], t2) * -1.5
        out += ovvv[:, :, a0:a1].transpose(2, 1, 0, 3) * 2     # vvov[a,c,k,b]
        out -= ovvv[:, :, a0:a1].transpose(2, 3, 0, 1)         # vvov[a,b,k,c]
        g = (ovvv[:, :, a0:a1].transpose(2, 3, 0, 1) * 2
             - ovvv[:, a0:a1].transpose(1, 2, 0, 3))           # g2vvov[a,c,j,d]
        v = _ein('acjd,kjbd->ackb', g, theta2)
        g = None
        out += v.transpose(0, 3, 2, 1)
        out -= v * .5
        v = out = None

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
    for a0, a1 in _vslice(nv, no * nv * nv * 8 * 3):
        wvvov[a0:a1] += _ein('ajkc,jb->abkc', ov1[a0:a1], t1)
        wvvov[a0:a1] -= _ein('ajkb,jc->abkc', ov2[a0:a1], t1)
        wvvov[a0:a1] += _ein('laki,klbc->abic', g2ovoo[:, a0:a1], tau)
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
    no = x.shape[-4]                 # x may carry leading batch indices
    if not hasattr(imds, 'orbv_s'):
        mask = mycc.get_frozen_mask()
        mo = cp.asarray(mycc.mo_coeff)[:, cp.asarray(mask)]
        imds.orbv_s = _ladder_setup(mycc).coeff @ mo[:, no:]
    from electrolites.eom_gpu import _ladder_pairsym
    return _ladder_pairsym(mycc, imds.orbv_s, x)


def update_lambda(imds, mycc, l1, l2):
    """PySCF ``ccsd_lambda.update_lambda``, unblocked, on the device."""
    l1new, l2new = lambda_rhs(imds, mycc, l1, l2)
    e = cp.diag(imds.fock)
    no = imds.nocc
    eia = e[:no, None] - (e[None, no:] + mycc.level_shift)
    l1new /= eia
    l2new /= eia[:, None, :, None] + eia[None, :, None, :]
    return l1new, l2new


def lambda_rhs(imds, mycc, l1, l2):
    """The Λ update before its division by the orbital-energy differences
    (λ2 part symmetrised): the Λ equations are lambda_rhs(λ) = D λ with
    D = e_i - e_a (e_i + e_j - e_a - e_b).  Assembled from its λ-linear
    part (``lambda_linear``, which is λ (H̄ - E_CC) on the singles and
    doubles -- the left EOM-EE operator), the diagonal D λ and the
    λ-independent part (η, ``_lambda_const``)."""
    c1, c2 = _lambda_const(imds, mycc)
    s1, s2 = lambda_linear(imds, mycc, l1[None], l2[None])
    d1 = _lambda_diag(imds, mycc)
    s1, s2 = s1[0], s2[0]
    s1 += c1
    s1 += d1 * l1
    s2 += c2
    s2 += (d1[:, None, :, None] + d1[None, :, None, :]) * l2
    return s1, s2


def _lambda_diag(imds, mycc):
    e = cp.diag(imds.fock)
    no = imds.nocc
    return e[:no, None] - (e[None, no:] + mycc.level_shift)


def _lambda_const(imds, mycc):
    """The λ-independent part of the Λ update (η): f_ov, the t1 piece of
    mia1 through voov, and voov itself (symmetrised as the λ2 update)."""
    c = getattr(imds, '_lambda_const', None)
    if c is not None:
        return c
    t1 = imds.t1
    no = imds.nocc
    voov = imds.ovvo.transpose(1, 0, 3, 2)
    c1 = imds.fock[:no, no:].copy()
    c1 += _ein('aijb,jb->ia', voov, t1) * 2
    c1 -= _ein('bija,jb->ia', voov, t1)
    c2 = voov.transpose(1, 2, 0, 3) * .5
    c2 = c2 + c2.transpose(1, 0, 3, 2)
    imds._lambda_const = (c1, c2)
    return c1, c2


def lambda_linear(imds, mycc, L1, L2):
    """λ (H̄ - E_CC) on the singles and doubles for a batch of λ:
    L1 (n, o, v), L2 (n, o, o, v, v) with L2[n,j,i,b,a] = L2[n,i,j,a,b].
    PySCF's ``update_lambda`` without its constant terms and without the
    orbital-energy diagonal (w1, w2 are used as they are, the D λ of the
    Λ update cancelling the diag(e) taken off them there), with a leading
    batch index on every λ-dependent array: the left EOM-EE Davidson runs
    all its trial vectors of a step through here at once."""
    from electrolites.eom_gpu import _ein_o
    t1, t2 = imds.t1, imds.t2
    ovvv, ovoo, ovvo, oovv = imds.ovvv, imds.ovoo, imds.ovvo, imds.oovv
    n, no, nv = L1.shape
    w1, w2 = imds.w1, imds.w2

    theta = t2 * 2 - t2.transpose(0, 1, 3, 2)
    mba = _ein('nklca,klcb->nba', L2, theta)
    mij = _ein('nikcd,jkcd->nij', L2, theta)
    theta = None
    mba1 = _ein('njc,jb->nbc', L1, t1) + mba
    mij1 = _ein('nkb,jb->nkj', L1, t1) + mij
    mia1 = _ein('nkc,jkbc->njb', L1, t2) * 2
    mia1 -= _ein('nkc,jkcb->njb', L1, t2)
    mia1 -= _ein('ja,nka,kb->njb', t1, L1, t1)
    mia1 -= _ein('nbd,jd->njb', mba, t1)
    mia1 -= _ein('nlj,lb->njb', mij, t1)

    L2new = _ladder_oovv(imds, mycc, L2)
    L1new = _ein('nijab,jb->nia', L2new, t1) * 2
    L1new -= _ein('njiab,jb->nia', L2new, t1)
    L2new *= .5

    L1new += L1 @ w1                                      # nib,ba->nia
    L1new -= _ein('nja,ij->nia', L1, w2)
    L1new -= _ein('nik,ka->nia', mij, imds.w4)
    L1new -= _ein('nca,ic->nia', mba, imds.w4)
    L1new += _ein('nijab,bj->nia', L2, imds.w3) * 2
    L1new -= _ein('nijba,bj->nia', L2, imds.w3)
    L2new += _ein('nia,jb->nijab', L1, imds.w4)
    L2new += L2 @ w1                                      # njibc,ca->njiba
    L2new -= cp.matmul(w2, L2.reshape(n, no, -1)).reshape(L2.shape)

    L1new -= _ein('iajk,nkj->nia', ovoo, mij1) * 2
    L1new += _ein('jaik,nkj->nia', ovoo, mij1)
    ov_t = ovoo.transpose(0, 3, 1, 2).reshape(-1, no)      # (jib, k)
    L2new -= cp.matmul(ov_t, L1).reshape(L2.shape)       # jbki,nka->njiba
    ov_t = None

    tau = t2 + _ein('ia,jb->ijab', t1, t1)
    l2tau = _ein('nijcd,klcd->nijkl', L2, tau)
    tau = None
    l2t1 = _ein('njidc,kc->nijkd', L2, t1)

    L1new -= _ein('njb,jiab->nia', L1, oovv)
    L1new += _ein_o('iabc,nbc->nia', ovvv, mba1, None, out_axis=1) * 2
    L1new -= _ein_o('ibca,nbc->nia', ovvv, mba1, None, out_axis=1)
    L2new += _ein_o('jbac,nic->njiba', ovvv, L1, None, out_axis=1)
    m4 = _ein_o('nijkd,kadb->nijab', l2t1, ovvv, 3, big=1)
    l2t1 = None
    L2new -= m4
    L1new -= _ein('nijab,jb->nia', m4, t1) * 2
    L1new -= _ein('nijab,ia->njb', m4, t1) * 2
    L1new += _ein('njiab,jb->nia', m4, t1)
    L1new += _ein('njiab,ia->njb', m4, t1)
    m4 = None

    voov = ovvo.transpose(1, 0, 3, 2)
    L1new += _ein('njb,aijb->nia', L1, voov) * 2
    L2new -= _ein('bjic,nca->njiba', voov, mba1)
    L2new -= _ein('bjka,nik->njiba', voov, mij1)
    L1new += _ein('aijb,njb->nia', voov, mia1) * 2
    L1new -= _ein('bija,njb->nia', voov, mia1)
    m4 = _ein('nijkl,aklb->nijab', l2tau, voov)
    l2tau = None
    L2new += m4 * .5
    L1new += _ein('nijab,jb->nia', m4, t1) * 2
    L1new -= _ein('nijba,jb->nia', m4, t1)
    m4 = None

    L1new -= _ein('ckij,njkca->nia', imds.wvooo, L2)
    L1new += cp.matmul(L2.transpose(0, 2, 3, 1, 4).reshape(n, no, -1),
                       imds.wvvov.reshape(nv, -1).T)      # abkc,nkibc->nia
    tmp_voov = getattr(imds, '_tmp_voov', None)
    if tmp_voov is None:
        tmp_voov = imds._tmp_voov = imds.wVOov * 2 + imds.wvOOv
    tmp = L2.transpose(0, 1, 3, 2, 4) - L2.transpose(0, 1, 4, 2, 3) * .5
    L2new += _ein('niakc,bjkc->njiba', tmp, tmp_voov)
    tmp = _ein('njkca,bikc->njiba', L2, imds.wvOOv)
    L2new += tmp
    L2new += tmp.transpose(0, 2, 1, 3, 4) * .5
    tmp = None

    m3 = _ein('ijkl,nklab->nijab', imds.woooo, L2)
    L2new += m3 * .5
    L1new += _ein('nijab,jb->nia', m3, t1) * 2
    L1new -= _ein('nijba,jb->nia', m3, t1)
    m3 = None

    L2new = L2new + L2new.transpose(0, 2, 1, 4, 3)
    return L1new, L2new


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
    from electrolites.fastccsd import _avail
    space = getattr(mycc, 'diis_space', 6)
    from electrolites.eom_gpu import _pack_singlet, _unpack_singlet
    nov = no * nv
    vec = (nov + nov * (nov + 1) // 2) * 8
    # DIIS vectors in host memory when they would be a large share of the
    # device (o 66, v 288: 12 x 2.9 GB)
    adiis = DIIS(incore=2 * space * vec < 0.2 * _avail())
    adiis.space = space
    conv = False
    for it in range(max_cycle):
        l1new, l2new = update_lambda(imds, mycc, l1, l2)
        normt = float(cp.sqrt(cp.linalg.norm(l1new - l1) ** 2
                              + cp.linalg.norm(l2new - l2) ** 2))
        l1 = l2 = None
        # λ2 is pair-symmetric: DIIS on the packed (λ1, λ2[ia>=jb]) vector
        vec = adiis.update(_pack_singlet(l1new[None], l2new[None])[0])
        l1new = l2new = None
        l1, l2 = _unpack_singlet(vec[None], no, nv)
        l1, l2 = l1[0].copy(), l2[0]
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
