"""EOM-CCSD on the GPU for a closed-shell (GPU4PySCF) CCSD.

GPU4PySCF's ``CCSD`` has no EOM (``ipccsd`` & co. are PySCF's methods on
the host, reading an ``eris`` with ``ovvv`` that GPU4PySCF's integral
object does not carry).  Here PySCF's EOM classes keep their Davidson
(``lib.davidson_nosym1``), guesses and vector layout, and everything that
touches an integral runs on the device:

* the MO integrals from ``fastmp2``'s half transformation (as the (T)
  correction's): ``ovvv``, ``ovoo``, ``ovov``, ``oooo``, ``oovv``, and the
  Fock matrix from one GPU Fock build;
* the intermediates (``rintermediates`` -- Loo, Lvv, Fov, Woooo, Wooov,
  Wovoo, Wovov, Wovvo for IP; Wvovv, Wvvvo for EA), term for term in
  CuPy;
* **no v^4 array**: PySCF's EA stores Wvvvv (110 GB at penicillin
  V/def2-SVP).  Its pieces are applied instead: the (ac|bd) ladder on the
  trial vectors AO-direct, through the same integral blocks and cache as
  ``fastccsd``'s CCSD ladder (sum_cd (ac|bd) X_cd is an exchange-type
  contraction of C_v X C_v^T), the t1/t2 parts through ``ovov``/``ovvv``;
  Wvvvo's sum_d Wvvvv[abcd] t1[jd] uses (ac|b j~) with j~ = C_v t1[j],
  one more ``ovvv``-shaped transformation; the diagonal's (aa|bb) from
  one batch of GPU J builds;
* the sigma vectors for all of a Davidson step's trial vectors at once
  (batched contractions).

    from electrolites import eom_gpu
    mycc = gpu4pyscf.cc.ccsd_incore.CCSD(mf).run()
    e, v = eom_gpu.EOMIP(mycc).kernel(nroots=4)
"""
import numpy as np
import cupy as cp

from pyscf.cc import eom_rccsd
from pyscf.lib import logger

_ein = cp.einsum


class _Imds:
    """Device integrals, amplitudes and intermediates."""

    def __init__(self, mycc):
        from electrolites import fastmp2
        from electrolites.ccsd_t import _fock_mo
        t1 = cp.asarray(mycc.t1)
        t2 = cp.asarray(mycc.t2)
        no, nv = t1.shape
        mask = mycc.get_frozen_mask()
        mo = cp.asarray(mycc.mo_coeff)[:, cp.asarray(mask)]
        orbo, orbv = mo[:, :no], mo[:, no:]
        mol = mycc.mol
        self.t1, self.t2 = t1, t2
        self.fock = _fock_mo(mycc, mo)
        self.ovvv = cp.asarray(fastmp2.get_int4c2e_ovpq(mol, orbo, orbv,
                                                        orbv, orbv))
        ovon = cp.asarray(fastmp2.get_int4c2e_ovpq(mol, orbo, orbv, orbo,
                                                   mo))
        self.ovoo = cp.ascontiguousarray(ovon[:, :, :, :no])
        self.ovov = cp.ascontiguousarray(ovon[:, :, :, no:])
        ovon = None
        oonn = cp.asarray(fastmp2.get_int4c2e_ovpq(mol, orbo, orbo, mo, mo))
        self.oooo = cp.ascontiguousarray(oonn[:, :, :no, :no])
        self.oovv = cp.ascontiguousarray(oonn[:, :, no:, no:])
        oonn = None
        self.ovvo = self.ovov.transpose(0, 1, 3, 2)
        self.nocc, self.nvir = no, nv

    # --- rintermediates, on the device --------------------------------
    def _foo_fov_fvv(self):
        f = self.fock
        no = self.nocc
        return f[:no, :no], f[:no, no:], f[no:, no:]

    def cc_Foo(self):
        t1, t2, ovov = self.t1, self.t2, self.ovov
        foo = self._foo_fov_fvv()[0]
        tau = t2 + _ein('ic,ld->ilcd', t1, t1)
        F = 2 * _ein('kcld,ilcd->ki', ovov, tau)
        F -= _ein('kdlc,ilcd->ki', ovov, tau)
        return F + foo

    def cc_Fvv(self):
        t1, t2, ovov = self.t1, self.t2, self.ovov
        fvv = self._foo_fov_fvv()[2]
        tau = t2 + _ein('ka,ld->klad', t1, t1)
        F = -2 * _ein('kcld,klad->ac', ovov, tau)
        F += _ein('kdlc,klad->ac', ovov, tau)
        return F + fvv

    def cc_Fov(self):
        t1, ovov = self.t1, self.ovov
        fov = self._foo_fov_fvv()[1]
        F = 2 * _ein('kcld,ld->kc', ovov, t1)
        F -= _ein('kdlc,ld->kc', ovov, t1)
        return F + fov

    def make_Loo(self):
        t1 = self.t1
        fov = self._foo_fov_fvv()[1]
        L = self.cc_Foo() + _ein('kc,ic->ki', fov, t1)
        L += 2 * _ein('lcki,lc->ki', self.ovoo, t1)
        L -= _ein('kcli,lc->ki', self.ovoo, t1)
        return L

    def make_Lvv(self):
        t1 = self.t1
        fov = self._foo_fov_fvv()[1]
        L = self.cc_Fvv() - _ein('kc,ka->ac', fov, t1)
        L += 2 * _ein('kdac,kd->ac', self.ovvv, t1)
        L -= _ein('kcad,kd->ac', self.ovvv, t1)
        return L

    def make_Wooov(self):
        W = _ein('ic,kcld->klid', self.t1, self.ovov)
        return W + self.ovoo.transpose(2, 0, 3, 1)

    def make_W1ovov(self):
        W = -_ein('kcld,ilcb->kbid', self.ovov, self.t2)
        return W + self.oovv.transpose(0, 2, 1, 3)

    def make_W1ovvo(self):
        ovov, t2 = self.ovov, self.t2
        W = 2 * _ein('kcld,ilad->kaci', ovov, t2)
        W -= _ein('kcld,liad->kaci', ovov, t2)
        W -= _ein('kdlc,ilad->kaci', ovov, t2)
        return W + self.ovvo.transpose(0, 2, 1, 3)

    def make_Woooo(self):
        t1, t2 = self.t1, self.t2
        tau = t2 + _ein('ic,jd->ijcd', t1, t1)
        W = _ein('kcld,ijcd->klij', self.ovov, tau)
        W += _ein('ldki,jd->klij', self.ovoo, t1)
        W += _ein('kclj,ic->klij', self.ovoo, t1)
        return W + self.oooo.transpose(0, 2, 1, 3)

    def make_ip(self):
        t1, t2 = self.t1, self.t2
        ovoo, ovvv = self.ovoo, self.ovvv
        self.Loo = self.make_Loo()
        self.Lvv = self.make_Lvv()
        self.Fov = self.cc_Fov()
        Wooov = self.make_Wooov()
        W1ovov = self.make_W1ovov()
        W1ovvo = self.make_W1ovvo()
        self.Wovov = W1ovov - _ein('klid,lb->kbid', Wooov, t1) \
            + _ein('kcbd,ic->kbid', ovvv, t1)
        self.Wovvo = W1ovvo - _ein('la,lkic->kaci', t1, Wooov) \
            + _ein('kcad,id->kaci', ovvv, t1)
        self.Woovv = self.ovov.transpose(0, 2, 1, 3)
        self.Woooo = self.make_Woooo()
        self.Wooov = Wooov
        W = _ein('kbid,jd->kbij', W1ovov, t1)
        W -= _ein('klij,lb->kbij', self.Woooo, t1)
        W += _ein('kbcj,ic->kbij', W1ovvo, t1)
        W += 2 * _ein('ldki,ljdb->kbij', ovoo, t2)
        W -= _ein('ldki,jldb->kbij', ovoo, t2)
        W -= _ein('kdli,ljdb->kbij', ovoo, t2)
        tau = t2 + _ein('jd,ic->jidc', t1, t1)
        W += _ein('kcbd,jidc->kbij', ovvv, tau)
        W -= _ein('kclj,libc->kbij', ovoo, t2)
        W += _ein('kc,ijcb->kbij', self.Fov, t2)
        W += ovoo.transpose(3, 1, 2, 0)
        self.Wovoo = W
        return self


def _make_ea(imds, mycc):
    """Wvovv and Wvvvo (and the pieces of Wvvvv) for EA."""
    from electrolites import fastmp2
    t1, t2 = imds.t1, imds.t2
    ovov, ovvv, ovoo = imds.ovov, imds.ovvv, imds.ovoo
    no, nv = imds.nocc, imds.nvir
    if not hasattr(imds, 'Wovvo'):
        imds.make_ip()
    imds.Wvovv = -_ein('ka,kcld->alcd', t1, ovov) + ovvv.transpose(2, 0, 3, 1)
    tau = t2 + _ein('ka,lb->klab', t1, t1)
    imds.tau = tau
    W1ovov = imds.make_W1ovov()
    W1ovvo = imds.make_W1ovvo()
    W = -_ein('alcj,lb->abcj', W1ovov.transpose(1, 0, 3, 2), t1)
    W -= _ein('kbcj,ka->abcj', W1ovvo, t1)
    W1ovov = W1ovvo = None
    W += 2 * _ein('ldac,ljdb->abcj', ovvv, t2)
    W -= _ein('ldac,ljbd->abcj', ovvv, t2)
    W -= _ein('lcad,ljdb->abcj', ovvv, t2)
    W -= _ein('kcbd,jkda->abcj', ovvv, t2)
    W += _ein('kclj,lkba->abcj', ovoo, t2)
    W += _ein('kcjb,ka->abcj', _ein('kclj,lb->kcjb', ovoo, t1), t1)
    W -= _ein('kc,kjab->abcj', imds.Fov, t2)
    W += ovvv.transpose(3, 1, 2, 0)
    # sum_d Wvvvv[abcd] t1[jd], Wvvvv = (ac|bd) + ovov.tau - ovvv.t1 terms
    mask = mycc.get_frozen_mask()
    mo = cp.asarray(mycc.mo_coeff)[:, cp.asarray(mask)]
    orbv = mo[:, no:]
    jt = orbv @ t1.T                                          # (nao, o)
    bjac = cp.asarray(fastmp2.get_int4c2e_ovpq(mycc.mol, jt, orbv, orbv,
                                               orbv))         # (j~ b|a c)
    W += bjac.transpose(2, 1, 3, 0)
    bjac = None
    g = _ein('kcld,jd->kclj', ovov, t1)
    W += _ein('kclj,klab->abcj', g, tau)
    g = _ein('ldac,jd->lacj', ovvv, t1)
    W -= _ein('lacj,lb->abcj', g, t1)
    g = _ein('kcbd,jd->kcbj', ovvv, t1)
    W -= _ein('kcbj,ka->abcj', g, t1)
    g = None
    imds.Wvvvo = W
    # Wvvvv[abab] for the diagonal: (aa|bb) from J builds on the virtual
    # densities, plus the t-dependent parts
    mf = mycc._scf
    if getattr(mf, 'device', 'cpu') != 'gpu':
        mf = mf.to_gpu()
    dms = _ein('pa,qa->apq', orbv, orbv)
    vj = cp.asarray(mf.get_j(mycc.mol, dms, hermi=1))
    aabb = _ein('apq,pb,qb->ab', vj, orbv, orbv)
    vj = dms = None
    imds.Wabab = aabb + _ein('kalb,klab->ab', ovov, tau) \
        - _ein('lbaa,lb->ab', ovvv, t1) - _ein('kabb,ka->ab', ovvv, t1)
    vhf = _ladder_setup(mycc)
    imds.orbv_s = vhf.coeff @ orbv
    return imds


def _ladder_setup(mycc):
    """vhfopt (as ``fastccsd``'s) and the AO-block cache on mycc."""
    from gpu4pyscf.scf import int4c2e
    vhfopt = getattr(mycc, '_fastccsd_vhfopt', None)
    if vhfopt is None:
        nao = mycc.mol.nao_nr(cart=True)
        vhfopt = int4c2e._VHFOpt(mycc.mol, 'int2e')
        vhfopt.build(group_size=max(4, (nao + 3) // 4),
                     diag_block_with_triu=True)
        vhfopt._fastccsd_blk = None
        mycc._fastccsd_vhfopt = vhfopt
        mycc._fastccsd_eri = None
    return vhfopt


def _ladder(mycc, x2):
    """Y[x] = sum_rs (pr|qs) X[x]_rs for a batch of AO matrices in
    vhfopt's basis (nx, nao, nao) -- the CCSD ladder's AO step for any X
    (``fastccsd._direct_ovvv_vvvv``'s block loop and cache)."""
    import os
    from electrolites import fastccsd as F
    vhfopt = _ladder_setup(mycc)
    mol = vhfopt.mol
    nao = mol.nao
    nx = len(x2)
    x2 = cp.ascontiguousarray(x2)
    Ht2ao = cp.zeros_like(x2)
    ao_loc = mol.ao_loc
    nao2 = nao * nao
    _dgemm = cp.cuda.cublas.dgemm
    handle = cp.cuda.device.get_cublas_handle()
    N = cp.cuda.cublas.CUBLAS_OP_N
    T = cp.cuda.cublas.CUBLAS_OP_T
    one = np.ones(1)
    one_ptr = one.ctypes.data
    x2_ptr = int(x2.data.ptr)
    H_ptr = int(Ht2ao.data.ptr)
    l_ctr_offsets = vhfopt.l_ctr_offsets
    log_qs = vhfopt.log_qs
    cp_idx, cp_jdx = np.tril_indices(len(vhfopt.uniq_l_ctr))
    idx, idy = cp.tril_indices(nao)
    cache = getattr(mycc, '_fastccsd_eri', None)
    if (cache is None and not os.environ.get('FASTCCSD_NO_CACHE')
            and not getattr(mycc, '_fastccsd_nocache', False)):
        total = 0
        for cid in range(len(log_qs)):
            ci_, cj_ = cp_idx[cid], cp_jdx[cid]
            ni = ao_loc[l_ctr_offsets[ci_ + 1]] - ao_loc[l_ctr_offsets[ci_]]
            nj = ao_loc[l_ctr_offsets[cj_ + 1]] - ao_loc[l_ctr_offsets[cj_]]
            total += int(ni) * int(nj) * nao2 * 8
        if total < (F._avail() - getattr(mycc, '_fastccsd_lmem', 0)) / 3:
            cache = mycc._fastccsd_eri = {}
        elif total < F._host_avail() / 4:
            cache = mycc._fastccsd_eri = F._HostCache()
    for cid, log_q_ij in enumerate(log_qs):
        if log_q_ij.size == 0:
            continue
        cpi, cpj = cp_idx[cid], cp_jdx[cid]
        ish0, ish1 = l_ctr_offsets[cpi], l_ctr_offsets[cpi + 1]
        jsh0, jsh1 = l_ctr_offsets[cpj], l_ctr_offsets[cpj + 1]
        i0, i1 = int(ao_loc[ish0]), int(ao_loc[ish1])
        j0, j1 = int(ao_loc[jsh0]), int(ao_loc[jsh1])
        eri = cache.get(cid) if cache is not None else None
        if isinstance(eri, np.ndarray):
            host, eri = eri, cp.empty(eri.shape)
            eri.set(host)
        if eri is None:
            eri = cp.zeros([i1 - i0, nao, j1 - j0, nao])
            strides = [1, (j1 - j0) * nao, (j1 - j0) * nao**2, nao]
            F._fill(mycc, eri, strides, [0, 0, i0, j0], vhfopt, cid)
            cache = getattr(mycc, '_fastccsd_eri', None)
            eri[:, idx, :, idy] = eri[:, idy, :, idx]
            if isinstance(cache, F._HostCache):
                cache[cid] = eri.get()
            elif cache is not None:
                cache[cid] = eri
        ic, jc = i1 - i0, j1 - j0
        e2 = eri.reshape(-1, jc * nao)
        _dgemm(handle, N, N, jc * nao, nx, ic * nao, one_ptr, e2.data.ptr,
               jc * nao, x2_ptr + i0 * nao * 8, nao2, one_ptr,
               H_ptr + j0 * nao * 8, nao2)
        if i0 > j0:
            _dgemm(handle, T, N, ic * nao, nx, jc * nao, one_ptr,
                   e2.data.ptr, jc * nao, x2_ptr + j0 * nao * 8, nao2,
                   one_ptr, H_ptr + i0 * nao * 8, nao2)
        eri = e2 = None
    lmem = F._release_lmem()
    mycc._fastccsd_lmem = max(lmem, getattr(mycc, '_fastccsd_lmem', 0))
    return Ht2ao


def _ip_sigma(imds, r1, r2):
    """ipccsd_matvec (partition None) for a batch: r1 (n, o), r2 (n, o, o, v)."""
    Hr1 = -_ein('ki,nk->ni', imds.Loo, r1)
    Hr1 += 2 * _ein('ld,nild->ni', imds.Fov, r2)
    Hr1 -= _ein('kd,nkid->ni', imds.Fov, r2)
    Hr1 -= 2 * _ein('klid,nkld->ni', imds.Wooov, r2)
    Hr1 += _ein('lkid,nkld->ni', imds.Wooov, r2)
    Hr2 = -_ein('kbij,nk->nijb', imds.Wovoo, r1)
    Hr2 += _ein('bd,nijd->nijb', imds.Lvv, r2)
    Hr2 -= _ein('ki,nkjb->nijb', imds.Loo, r2)
    Hr2 -= _ein('lj,nilb->nijb', imds.Loo, r2)
    Hr2 += _ein('klij,nklb->nijb', imds.Woooo, r2)
    Hr2 += 2 * _ein('lbdj,nild->nijb', imds.Wovvo, r2)
    Hr2 -= _ein('kbdj,nkid->nijb', imds.Wovvo, r2)
    Hr2 -= _ein('lbjd,nild->nijb', imds.Wovov, r2)
    Hr2 -= _ein('kbid,nkjd->nijb', imds.Wovov, r2)
    tmp = 2 * _ein('lkdc,nkld->nc', imds.Woovv, r2)
    tmp -= _ein('kldc,nkld->nc', imds.Woovv, r2)
    Hr2 -= _ein('nc,ijcb->nijb', tmp, imds.t2)
    return Hr1, Hr2


def _ip_diag(imds):
    no, nv = imds.nocc, imds.nvir
    Loo, Lvv = imds.Loo, imds.Lvv
    t2 = imds.t2
    Hr1 = -cp.diag(Loo)
    dLoo = cp.diag(Loo)
    Hr2 = Lvv.diagonal()[None, None, :] - dLoo[:, None, None] \
        - dLoo[None, :, None]
    Hr2 = Hr2 + _ein('ijij->ij', imds.Woooo)[:, :, None]
    wvo = _ein('jbbj->jb', imds.Wovvo)
    Hr2 = Hr2 + 2 * wvo[None, :, :]
    Hr2 = Hr2 - cp.eye(no)[:, :, None] * wvo[:, None, :]
    wov = _ein('jbjb->jb', imds.Wovov)
    Hr2 = Hr2 - wov[None, :, :] - wov[:, None, :]
    Hr2 = Hr2 - 2 * _ein('jibc,ijcb->ijb', imds.Woovv, t2)
    Hr2 = Hr2 + _ein('ijbc,ijcb->ijb', imds.Woovv, t2)
    return Hr1, Hr2


def _ea_sigma(imds, mycc, r1, r2):
    """eaccsd_matvec (partition None) for a batch: r1 (n, v), r2 (n, o, v, v)."""
    t1, t2 = imds.t1, imds.t2
    n, no, nv = r2.shape[0], imds.nocc, imds.nvir
    Hr1 = _ein('ac,nc->na', imds.Lvv, r1)
    Hr1 += 2 * _ein('ld,nlad->na', imds.Fov, r2)
    Hr1 -= _ein('ld,nlda->na', imds.Fov, r2)
    Hr1 += 2 * _ein('alcd,nlcd->na', imds.Wvovv, r2)
    Hr1 -= _ein('aldc,nlcd->na', imds.Wvovv, r2)
    Hr2 = _ein('abcj,nc->njab', imds.Wvvvo, r1)
    Hr2 += _ein('ac,njcb->njab', imds.Lvv, r2)
    Hr2 += _ein('bd,njad->njab', imds.Lvv, r2)
    Hr2 -= _ein('lj,nlab->njab', imds.Loo, r2)
    Hr2 += 2 * _ein('lbdj,nlad->njab', imds.Wovvo, r2)
    Hr2 -= _ein('lbjd,nlad->njab', imds.Wovov, r2)
    Hr2 -= _ein('lajc,nlcb->njab', imds.Wovov, r2)
    Hr2 -= _ein('lbcj,nlca->njab', imds.Wovvo, r2)
    # sum_cd Wvvvv[abcd] r2[jcd]
    ov = imds.orbv_s
    x = _ein('pc,mcd,qd->mpq', ov, r2.reshape(n * no, nv, nv), ov)
    y = _ladder(mycc, x)
    x = None
    Hr2 += _ein('pa,mpq,qb->mab', ov, y, ov).reshape(n, no, nv, nv)
    y = None
    g = _ein('kcld,njcd->njkl', imds.ovov, r2)
    Hr2 += _ein('njkl,klab->njab', g, imds.tau)
    g = _ein('ldac,njcd->njal', imds.ovvv, r2)
    Hr2 -= _ein('njal,lb->njab', g, t1)
    g = _ein('kcbd,njcd->njkb', imds.ovvv, r2)
    Hr2 -= _ein('njkb,ka->njab', g, t1)
    g = None
    tmp = 2 * _ein('klcd,nlcd->nk', imds.Woovv, r2)
    tmp -= _ein('kldc,nlcd->nk', imds.Woovv, r2)
    Hr2 -= _ein('nk,kjab->njab', tmp, t2)
    return Hr1, Hr2


def _ea_diag(imds):
    no, nv = imds.nocc, imds.nvir
    Lvv, Loo, t2 = imds.Lvv, imds.Loo, imds.t2
    Hr1 = cp.diag(Lvv)
    dv = cp.diag(Lvv)
    Hr2 = dv[None, :, None] + dv[None, None, :] - cp.diag(Loo)[:, None, None]
    wvo = _ein('jbbj->jb', imds.Wovvo)
    wov = _ein('jbjb->jb', imds.Wovov)
    Hr2 = Hr2 + 2 * wvo[:, None, :] - wov[:, None, :] - wov[:, :, None]
    Hr2 = Hr2 - cp.eye(nv)[None] * wvo[:, :, None]
    Hr2 = Hr2 + imds.Wabab[None]
    Hr2 = Hr2 - 2 * _ein('kjab,kjab->jab', imds.Woovv, t2)
    Hr2 = Hr2 + _ein('kjba,kjab->jab', imds.Woovv, t2)
    return Hr1, Hr2


class EOMEA(eom_rccsd.EOMEA):
    """PySCF's EOM-EA-CCSD with the integrals, intermediates and sigma
    vectors on the device and no v^4 array."""

    def make_imds(self, eris=None):
        if self.partition is not None:
            raise NotImplementedError('EOM-EA on the GPU: partition=None')
        return _make_ea(_Imds(self._cc), self._cc)

    def get_diag(self, imds=None):
        if imds is None:
            imds = self.make_imds()
        Hr1, Hr2 = _ea_diag(imds)
        return np.hstack([cp.asnumpy(Hr1), cp.asnumpy(Hr2).ravel()])

    def gen_matvec(self, imds=None, left=False, **kwargs):
        if left:
            raise NotImplementedError('left EOM-EA on the GPU')
        if imds is None:
            imds = self.make_imds()
        diag = self.get_diag(imds)
        no, nv = imds.nocc, imds.nvir
        mycc = self._cc

        def matvec(xs):
            x = cp.asarray(np.asarray(xs).reshape(len(xs), -1))
            r1 = x[:, :nv]
            r2 = x[:, nv:].reshape(-1, no, nv, nv)
            h1, h2 = _ea_sigma(imds, mycc, r1, r2)
            out = cp.hstack([h1, h2.reshape(len(x), -1)])
            return list(cp.asnumpy(out))

        return matvec, diag

    def matvec(self, vector, imds=None, diag=None):
        if imds is None:
            imds = self.make_imds()
        return self.gen_matvec(imds)[0]([vector])[0]


class EOMIP(eom_rccsd.EOMIP):
    """PySCF's EOM-IP-CCSD with the integrals, intermediates and sigma
    vectors on the device."""

    def __init__(self, cc):
        eom_rccsd.EOMIP.__init__(self, cc)

    def make_imds(self, eris=None):
        if self.partition is not None:
            raise NotImplementedError('EOM-IP on the GPU: partition=None')
        return _Imds(self._cc).make_ip()

    def get_diag(self, imds=None):
        if imds is None:
            imds = self.make_imds()
        Hr1, Hr2 = _ip_diag(imds)
        return np.hstack([cp.asnumpy(Hr1), cp.asnumpy(Hr2).ravel()])

    def gen_matvec(self, imds=None, left=False, **kwargs):
        if left:
            raise NotImplementedError('left EOM-IP on the GPU')
        if imds is None:
            imds = self.make_imds()
        diag = self.get_diag(imds)
        no, nv = imds.nocc, imds.nvir

        def matvec(xs):
            x = cp.asarray(np.asarray(xs).reshape(len(xs), -1))
            r1 = x[:, :no]
            r2 = x[:, no:].reshape(-1, no, no, nv)
            h1, h2 = _ip_sigma(imds, r1, r2)
            out = cp.hstack([h1, h2.reshape(len(x), -1)])
            return list(cp.asnumpy(out))

        return matvec, diag

    def matvec(self, vector, imds=None, diag=None):
        if imds is None:
            imds = self.make_imds()
        return self.gen_matvec(imds)[0]([vector])[0]
