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
    if isinstance(cache, F._HostCache) and cache:
        # the CCSD left its integral blocks in host memory (no room beside
        # its amplitudes); the EOM/Lambda step usually has the room, and
        # copying the whole cache host->device on every sigma batch was
        # 93 of 203 s of a ferrocene EOM-EE run
        total = sum(v.nbytes for v in cache.values())
        if total < .4 * F._avail():
            cache = mycc._fastccsd_eri = {k: cp.asarray(v)
                                          for k, v in cache.items()}
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
        if imds is None:
            imds = self.make_imds()
        diag = self.get_diag(imds)
        no, nv = imds.nocc, imds.nvir
        mycc = self._cc
        sigma = _lea_sigma if left else _ea_sigma

        def matvec(xs):
            x = cp.asarray(np.asarray(xs).reshape(len(xs), -1))
            r1 = x[:, :nv]
            r2 = x[:, nv:].reshape(-1, no, nv, nv)
            h1, h2 = sigma(imds, mycc, r1, r2)
            out = cp.hstack([h1, h2.reshape(len(x), -1)])
            return list(cp.asnumpy(out))

        return matvec, diag

    def matvec(self, vector, imds=None, diag=None):
        if imds is None:
            imds = self.make_imds()
        return self.gen_matvec(imds)[0]([vector])[0]

    def l_matvec(self, vector, imds=None, diag=None):
        if imds is None:
            imds = self.make_imds()
        return self.gen_matvec(imds, left=True)[0]([vector])[0]


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
        if imds is None:
            imds = self.make_imds()
        diag = self.get_diag(imds)
        no, nv = imds.nocc, imds.nvir
        sigma = _lip_sigma if left else _ip_sigma

        def matvec(xs):
            x = cp.asarray(np.asarray(xs).reshape(len(xs), -1))
            r1 = x[:, :no]
            r2 = x[:, no:].reshape(-1, no, no, nv)
            h1, h2 = sigma(imds, r1, r2)
            out = cp.hstack([h1, h2.reshape(len(x), -1)])
            return list(cp.asnumpy(out))

        return matvec, diag

    def matvec(self, vector, imds=None, diag=None):
        if imds is None:
            imds = self.make_imds()
        return self.gen_matvec(imds)[0]([vector])[0]

    def l_matvec(self, vector, imds=None, diag=None):
        if imds is None:
            imds = self.make_imds()
        return self.gen_matvec(imds, left=True)[0]([vector])[0]


# ------------------------------------------------------------------ EOM-EE


def _make_tau(t2, t1, r1, fac=1.):
    tau = _ein('ia,jb->ijab', t1, r1)
    return t2 + (tau + tau.transpose(1, 0, 3, 2)) * (fac * .5)


def _virtual_coulomb_diag(mycc, orbv):
    """(aa|bb) from one batch of GPU J builds on the virtual densities."""
    mf = mycc._scf
    if getattr(mf, 'device', 'cpu') != 'gpu':
        mf = mf.to_gpu()
    dms = _ein('pa,qa->apq', orbv, orbv)
    vj = cp.asarray(mf.get_j(mycc.mol, dms, hermi=1))
    return _ein('apq,pb,qb->ab', vj, orbv, orbv)


def _make_ee(imds, mycc):
    """PySCF's ``_IMDS.make_ee`` (singlet EOM-EE intermediates), unblocked,
    on the device."""
    t1, t2 = imds.t1, imds.t2
    ovvv, ovoo, ovov = imds.ovvv, imds.ovoo, imds.ovov
    oovv, ovvo, oooo = imds.oovv, imds.ovvo, imds.oooo
    no = imds.nocc
    foo, fov, fvv = imds._foo_fov_fvv()
    tau = _make_tau(t2, t1, t1)
    theta = t2 * 2 - t2.transpose(0, 1, 3, 2)

    Fvv = _ein('mf,mfae->ae', t1, ovvv) * 2
    Fvv -= _ein('mf,meaf->ae', t1, ovvv)
    woVoO = _ein('mebf,ijef->mbij', ovvv, tau)
    woVvO = _ein('jf,mebf->mbej', t1, ovvv)
    woVVo = _ein('jf,mfbe->mbej', -t1, ovvv)
    woOoV = _ein('if,mfne->mnie', t1, ovov) + ovoo.transpose(2, 0, 3, 1)
    tmp = _ein('njbf,mfne->mbej', t2, ovov)
    woVvO -= tmp * .5
    woVVo += tmp
    x = _ein('menf,jf->menj', ovov, t1)
    woVvO -= _ein('nb,menj->mbej', t1, x)
    x = _ein('mfne,jf->menj', ovov, t1)
    woVVo += _ein('nb,menj->mbej', t1, x)
    ovov2 = ovov * 2 - ovov.transpose(0, 3, 2, 1)
    woVvO += _ein('njfb,menf->mbej', theta, ovov2) * .5
    Fov = _ein('nf,menf->me', t1, ovov2)
    tilab = _ein('ia,jb->ijab', t1, t1) * .5 + t2
    Foo = _ein('mief,menf->ni', tilab, ovov2)
    Fvv -= _ein('mnaf,menf->ae', tilab, ovov2)
    tilab = None
    woVvO -= _ein('nb,menj->mbej', t1, ovoo)
    woVVo += _ein('nb,nemj->mbej', t1, ovoo)
    woVvO += ovvo.transpose(0, 2, 1, 3)
    woVVo -= oovv.transpose(0, 2, 3, 1)
    Fov += fov
    Foo += foo + .5 * _ein('me,ie->mi', Fov + fov, t1)
    Fvv += fvv - .5 * _ein('me,ma->ae', Fov + fov, t1)

    woOoO = _ein('je,nemi->mnij', t1, ovoo)
    woOoO = woOoO + woOoO.transpose(1, 0, 3, 2)
    woOoO += oooo.transpose(0, 2, 1, 3)
    tmp = _ein('meni,jneb->mbji', ovoo, t2)
    woVoO -= tmp.transpose(0, 1, 3, 2) * .5
    woVoO -= tmp
    ovooX = ovoo * 2 - ovoo.transpose(2, 1, 0, 3)
    woVoO += _ein('nemi,njeb->mbij', ovooX, theta) * .5
    Foo += _ein('ne,nemi->mi', t1, ovooX)
    ovooX = None
    woOoO += _ein('ijef,menf->mnij', tau, ovov)
    woVoO -= _ein('nb,mnij->mbij', t1, woOoO)
    tmpoovv = _ein('njbf,nemf->ejmb', t2, ovov)
    tmpovvo = _ein('nifb,menf->eimb', theta, ovov2)
    ovov2 = None
    tmpovvo *= -.5
    tmpovvo += tmpoovv * .5
    woVoO -= _ein('ie,ejmb->mbij', t1, tmpovvo)
    woVoO -= _ein('ie,ejmb->mbji', t1, tmpoovv)
    woVoO += ovoo.transpose(3, 1, 2, 0)
    tmpovvo -= ovvo.transpose(1, 3, 0, 2)
    tmpoovv -= oovv.transpose(3, 1, 0, 2)
    woVoO += _ein('mebj,ie->mbij', ovvo, t1)
    woVoO += _ein('mjbe,ie->mbji', oovv, t1)
    woVoO += _ein('me,ijeb->mbij', Fov, t2)

    ebmf = ovvv.transpose(1, 2, 0, 3)                      # (me|bf) -> ebmf
    w = _ein('ebmf,miaf->eiab', ebmf, t2)
    wvOvV = -.5 * w.transpose(0, 1, 3, 2) - w
    w = None
    wvOvV += ebmf.transpose(1, 2, 3, 0)
    tmp = -.5 * ebmf + ebmf.transpose(1, 0, 2, 3)
    wvOvV += _ein('efmb,mifa->eiba', tmp, theta)
    tmp = None
    wvOvV += _ein('meni,mnab->eiab', ovoo, tau)
    wvOvV -= _ein('me,miab->eiab', Fov, t2)
    wvOvV += _ein('ma,eimb->eiab', t1, tmpovvo)
    wvOvV += _ein('ma,eimb->eiba', t1, tmpoovv)

    imds.Foo, imds.Fov, imds.Fvv = Foo, Fov, Fvv
    imds.woOoO, imds.woVoO, imds.woOoV = woOoO, woVoO, woOoV
    imds.woVvO, imds.woVVo, imds.wvOvV = woVvO, woVVo, wvOvV
    imds.tau_half = tau * .5
    mask = mycc.get_frozen_mask()
    mo = cp.asarray(mycc.mo_coeff)[:, cp.asarray(mask)]
    imds.orbv = mo[:, no:]
    imds.orbv_s = _ladder_setup(mycc).coeff @ imds.orbv
    return imds


def _ladder_pairsym(mycc, ov, x):
    """sum_ef (ae|bf) x[..., i, j, e, f] for x with x[j,i] = x[i,j]^T (the
    singlet tau2, lambda2): the ladder of X^T is the transpose of the
    ladder of X, so only the i >= j pairs go through the AO integrals."""
    lead = x.shape[:-4]
    no, nv = x.shape[-4], x.shape[-1]
    x = x.reshape((-1, no, no, nv, nv))
    i, j = cp.tril_indices(no)
    xt = x[:, i, j]                                     # (n, npair, v, v)
    X = _ein('pe,mef,qf->mpq', ov, xt.reshape(-1, nv, nv), ov)
    Y = _ladder(mycc, X)
    X = None
    y = _ein('pa,mpq,qb->mab', ov, Y, ov).reshape(xt.shape)
    Y = None
    out = cp.empty_like(x)
    out[:, i, j] = y
    out[:, j, i] = y.transpose(0, 1, 3, 2)
    return out.reshape(lead + (no, no, nv, nv))


def _ee_ladder(imds, mycc, x):
    """sum_ef (ae|bf) x[n,i,j,e,f] for a batch, AO-direct (x is
    pair-symmetric: tau2 = r2 + r1 t1 + t1 r1)."""
    return _ladder_pairsym(mycc, imds.orbv_s, x)


def _ee_sigma(imds, mycc, r1, r2):
    """eeccsd_matvec_singlet for a batch: r1 (n, o, v), r2 (n, o, o, v, v)."""
    t1, t2 = imds.t1, imds.t2
    ovvv, ovov = imds.ovvv, imds.ovov
    Hr1 = _ein('ae,nie->nia', imds.Fvv, r1)
    Hr1 -= _ein('mi,nma->nia', imds.Foo, r1)
    Hr1 += _ein('me,nimae->nia', imds.Fov, r2) * 2
    Hr1 -= _ein('me,nimea->nia', imds.Fov, r2)

    rt = _ein('nia,jb->nijab', r1, t1)
    tau2 = r2 + rt + rt.transpose(0, 2, 1, 4, 3)
    rt = None
    Hr2 = _ee_ladder(imds, mycc, tau2)
    Hr2 += _ein('mnij,kmnab->kijab', imds.woOoO, r2)
    Hr2 *= .5
    Hr2 += _ein('be,nijae->nijab', imds.Fvv, r2)
    Hr2 -= _ein('mj,nimab->nijab', imds.Foo, r2)

    theta = r2 * 2 - r2.transpose(0, 1, 2, 4, 3)
    Hr1 += _ein('mfae,nmife->nia', ovvv, theta)
    tmp = _ein('meaf,nijef->nmaij', ovvv, tau2)
    Hr2 -= _ein('ma,nmbij->nijab', t1, tmp)
    tmp = _ein('meaf,nme->naf', ovvv, r1) * 2
    tmp -= _ein('mfae,nme->naf', ovvv, r1)
    Hr2 += _ein('naf,ijfb->nijab', tmp, t2)
    Hr2 -= _ein('mbij,nma->nijab', imds.woVoO, r1)
    Hr2 += _ein('ejab,nie->nijab', imds.wvOvV, r1)

    tmp = _ein('mbej,nimea->njiab', imds.woVVo, r2)
    Hr2 += tmp
    Hr2 += tmp.transpose(0, 1, 2, 4, 3) * .5
    tmp = None
    woVvO = imds.woVVo * .5 + imds.woVvO
    Hr1 += _ein('maei,nme->nia', woVvO, r1) * 2
    Hr2 += _ein('mbej,nimae->nijab', woVvO, theta)
    woVvO = None

    Hr1 -= _ein('mnie,kmnae->kia', imds.woOoV, theta)
    tmp = _ein('nmie,kme->kni', imds.woOoV, r1) * 2
    tmp -= _ein('mnie,kme->kni', imds.woOoV, r1)
    Hr2 -= _ein('kni,njab->kijab', tmp, t2)

    tmp = _ein('mfne,kmf->ken', ovov, r1) * 2
    tmp -= _ein('menf,kmf->ken', ovov, r1)
    tmp = _ein('ken,nb->keb', tmp, t1)
    tmp += _ein('menf,kmnbf->keb', ovov, theta)
    Hr2 -= _ein('keb,ijea->kjiab', tmp, t2)
    tmp = _ein('nemf,kimef->kni', ovov, theta)
    Hr1 -= _ein('na,kni->kia', t1, tmp)
    Hr2 -= _ein('kmj,miab->kijba', tmp, t2)
    theta = None

    tmp = _ein('menf,kijef->kmnij', ovov, tau2)
    tau2 = None
    Hr2 += _ein('kmnij,mnab->kijab', tmp, imds.tau_half)
    Hr2 = Hr2 + Hr2.transpose(0, 2, 1, 4, 3)
    return Hr1, Hr2


def _ee_diag(imds, mycc):
    """The singlet part of ``eeccsd_diag``: (Hr1aa, Hr2ab)."""
    t1, t2, ovov, ovvv = imds.t1, imds.t2, imds.ovov, imds.ovvv
    no, nv = t1.shape
    tau = _make_tau(t2, t1, t1)
    Fo, Fv = cp.diag(imds.Foo), cp.diag(imds.Fvv)
    Wovab = _ein('iaai->ia', imds.woVVo)
    Wovaa = Wovab + _ein('iaai->ia', imds.woVvO)
    Hr1 = Fv[None, :] - Fo[:, None] + Wovaa
    Wvvab = _ein('mnab,manb->ab', tau, ovov)
    ijb = _ein('iejb,ijeb->ijb', ovov, t2)
    Hr2 = -ijb[:, :, None, :] + Fv[None, None, :, None]
    jab = _ein('kajb,kjab->jab', ovov, t2)
    Hr2 = Hr2 - Fo[:, None, None, None] - jab[None]
    Hr2 = Hr2 + Wovaa[None, :, None, :] + Wovab[:, None, None, :]
    Hr2 = Hr2 + Hr2.transpose(1, 0, 3, 2)
    Hr2 = Hr2 + _ein('ijij->ij', imds.woOoO)[:, :, None, None]
    tmp = _ein('mb,mbaa->ab', t1, ovvv)
    Wvvab = Wvvab - tmp - tmp.T + _virtual_coulomb_diag(mycc, imds.orbv)
    Hr2 = Hr2 + Wvvab[None, None]
    return Hr1, Hr2


def _pack_singlet(r1, r2):
    """ccsd.amplitudes_to_vector for a batch: t1 + lower triangle of
    t2[ia,jb]."""
    n, no, nv = r1.shape
    nov = no * nv
    i, j = cp.tril_indices(nov)
    m = r2.transpose(0, 1, 3, 2, 4).reshape(n, nov, nov)
    return cp.hstack([r1.reshape(n, -1), m[:, i, j]])


def _unpack_singlet(v, no, nv):
    n = len(v)
    nov = no * nv
    i, j = cp.tril_indices(nov)
    m = cp.zeros((n, nov, nov), dtype=v.dtype)
    m[:, i, j] = v[:, nov:]
    m[:, j, i] = v[:, nov:]
    r2 = m.reshape(n, no, nv, no, nv).transpose(0, 1, 3, 2, 4)
    return v[:, :nov].reshape(n, no, nv), cp.ascontiguousarray(r2)


def _davidson_nosym(matvec, x0, diag, nroots, tol, tol_residual, max_cycle,
                    max_space, log):
    """Block Davidson for the lowest roots of a non-symmetric operator on
    the device, with a batched ``matvec`` ((n, dim) -> (n, dim)); PySCF's
    convergence test (energy change < tol, residual < tol_residual), real
    roots only, spare guess vectors kept but not expanded."""
    def orth(vs, basis):
        out = []
        for v in vs:
            for _ in range(2):
                if basis is not None and len(basis):
                    v = v - basis.T @ (basis @ v)
                for u in out:
                    v = v - u * (u @ v)
            nrm = cp.linalg.norm(v)
            if nrm > 1e-7:
                out.append(v / nrm)
        return out

    nwant = len(x0)
    V = cp.stack(orth(list(x0), None))
    AV = matvec(V)
    e_last = None
    conv = np.zeros(nwant, dtype=bool)
    for it in range(max_cycle):
        H = cp.asnumpy(V @ AV.T)
        w, c = np.linalg.eig(H)
        real = np.where(abs(w.imag) < 1e-3)[0]
        w, c = w[real].real, c[:, real].real
        order = np.argsort(w)[:nwant]
        w, c = w[order], cp.asarray(c[:, order])
        X = c.T @ V
        AX = c.T @ AV
        nrm = cp.linalg.norm(X, axis=1)
        X /= nrm[:, None]
        AX /= nrm[:, None]
        wd = cp.asarray(w)
        R = AX - wd[:, None] * X
        rnorm = cp.asnumpy(cp.linalg.norm(R, axis=1))
        if e_last is not None and len(e_last) == len(w):
            de = abs(w - e_last)
        else:
            de = np.ones(len(w))
        conv = (de < tol) & (rnorm < tol_residual)
        log.debug('EOM Davidson %d  space %d  |r| %s  dE %s', it, len(V),
                  rnorm[:nroots], de[:nroots])
        e_last = w
        if conv[:nroots].all():
            break
        new = []
        for k in np.where(~conv[:nroots])[0]:
            d = wd[k] - diag
            d = cp.where(abs(d) < 1e-8, 1e-8, d)
            new.append(R[k] / d)
        if len(V) + len(new) > max_space:
            q, r_ = cp.linalg.qr(X.T)
            V = q.T
            AV = cp.linalg.solve(r_.T, AX)
        new = orth(new, V)
        if not new:
            break
        Vn = cp.stack(new)
        V = cp.vstack([V, Vn])
        AV = cp.vstack([AV, matvec(Vn)])
    return conv[:nroots], e_last[:nroots], X[:nroots]


class EOMEESinglet(eom_rccsd.EOMEESinglet):
    """PySCF's singlet EOM-EE-CCSD (same vector layout and equations) with
    the intermediates, the sigma vectors (the (ae|bf) ladder AO-direct,
    batched over each Davidson step's trial vectors) and a non-symmetric
    block Davidson on the device."""

    def make_imds(self, eris=None):
        return _make_ee(_Imds(self._cc), self._cc)

    def get_diag(self, imds=None):
        if imds is None:
            imds = self.make_imds()
        h1, h2 = _ee_diag(imds, self._cc)
        return cp.asnumpy(_pack_singlet(h1[None], h2[None])[0])

    def gen_matvec(self, imds=None, diag=None, **kwargs):
        if imds is None:
            imds = self.make_imds()
        if diag is None:
            diag = self.get_diag(imds)
        no, nv = imds.nocc, imds.nvir
        mycc = self._cc

        def matvec(xs):
            x = cp.asarray(np.asarray(xs).reshape(len(xs), -1))
            out = []
            for p0 in range(0, len(x), _EE_BLK):
                r1, r2 = _unpack_singlet(x[p0:p0 + _EE_BLK], no, nv)
                h1, h2 = _ee_sigma(imds, mycc, r1, r2)
                out.append(_pack_singlet(h1, h2))
            return list(cp.asnumpy(cp.vstack(out)))

        return matvec, diag

    def matvec(self, vector, imds=None, diag=None):
        if imds is None:
            imds = self.make_imds()
        return self.gen_matvec(imds, diag=np.zeros(1))[0]([vector])[0]

    def kernel(self, nroots=1, koopmans=False, guess=None, eris=None,
               imds=None, diag=None):
        log = logger.new_logger(self)
        if imds is None:
            imds = self.make_imds()
        no, nv = imds.nocc, imds.nvir
        mycc = self._cc
        h1, h2 = _ee_diag(imds, mycc)
        d = _pack_singlet(h1[None], h2[None])[0]
        h1 = h2 = None
        nwant = min(len(d), nroots + max(3, nroots // 2))
        if guess is not None:
            x0 = cp.asarray(np.asarray(guess)).reshape(len(guess), -1)
            nwant = len(x0)
        else:
            idx = cp.argsort(d[:no * nv] if koopmans else d)[:nwant]
            x0 = cp.zeros((nwant, len(d)))
            x0[cp.arange(nwant), idx] = 1.

        def matvec(V):
            out = []
            for p0 in range(0, len(V), _EE_BLK):
                r1, r2 = _unpack_singlet(V[p0:p0 + _EE_BLK], no, nv)
                h1, h2 = _ee_sigma(imds, mycc, r1, r2)
                out.append(_pack_singlet(h1, h2))
            return cp.vstack(out)

        conv, e, X = _davidson_nosym(
            matvec, x0, d, nroots, self.conv_tol,
            self.conv_tol ** .5, self.max_cycle,
            self.max_space + nwant, log)
        self.converged, self.e = conv, e
        self.v = list(cp.asnumpy(X))
        for n in range(nroots):
            log.info('EOM-EE-CCSD singlet root %d E = %.16g  conv = %s', n,
                     e[n], conv[n])
        if nroots == 1:
            return e[0], self.v[0]
        return e, self.v


#: trial vectors per EE sigma batch
_EE_BLK = 8


def _ipccsd(self, nroots=1, left=False, koopmans=False, guess=None,
            partition=None, eris=None):
    return EOMIP(self).kernel(nroots, left, koopmans, guess, partition)


def _eaccsd(self, nroots=1, left=False, koopmans=False, guess=None,
            partition=None, eris=None):
    return EOMEA(self).kernel(nroots, left, koopmans, guess, partition)


def _eomee_ccsd_singlet(self, nroots=1, koopmans=False, guess=None,
                        eris=None, imds=None):
    return EOMEESinglet(self).kernel(nroots, koopmans, guess)


from gpu4pyscf.cc import ccsd_incore as _ci                   # noqa: E402
from electrolites.features import attach as _attach           # noqa: E402

for _name, _fn in (('ipccsd', _ipccsd), ('eaccsd', _eaccsd),
                   ('eomee_ccsd_singlet', _eomee_ccsd_singlet),
                   ('eomip_method', lambda self: EOMIP(self)),
                   ('eomea_method', lambda self: EOMEA(self)),
                   ('eomee_method', lambda self: EOMEESinglet(self))):
    if getattr(_ci.CCSDBase, _name, None) is NotImplemented:
        _attach(_ci.CCSDBase, _name, _fn)


# ------------------------------------------- triplet, spin-flip, left IP/EA
#
# The triplet and spin-flip sigma vectors and the EE diagonals are PySCF's
# own functions (``eom_rccsd.eeccsd_matvec_triplet``, ``eeccsd_matvec_sf``,
# ``eeccsd_diag``) compiled again with CuPy as the array module (as
# ``adc_gpu`` does for ADC); the vvvv ladder they call through
# ``_cc._add_vvvv`` is the AO-direct one, with PySCF's t2sym symmetries.


def _ladder_sym(mycc, ov, x, t2sym=None):
    """sum_ef (ae|bf) x[i,j,e,f] (PySCF ``_add_vvvv`` with ``t1=None``):
    't2sym' 'jiba' (x[j,i] = x[i,j]^T), '-jiba' (= -x[i,j]^T) and '-jiab'
    (x[j,i] = -x[i,j]) go through the i >= j pairs only."""
    no, nv = x.shape[0], x.shape[-1]
    if t2sym not in ('jiba', '-jiba', '-jiab'):
        X = _ein('pe,mef,qf->mpq', ov, x.reshape(-1, nv, nv), ov)
        Y = _ladder(mycc, X)
        return _ein('pa,mpq,qb->mab', ov, Y, ov).reshape(x.shape)
    i, j = cp.tril_indices(no)
    xt = x[i, j]
    X = _ein('pe,mef,qf->mpq', ov, xt, ov)
    Y = _ladder(mycc, X)
    X = None
    y = _ein('pa,mpq,qb->mab', ov, Y, ov)
    out = cp.empty_like(x)
    out[i, j] = y
    if t2sym == 'jiba':
        out[j, i] = y.transpose(0, 2, 1)
    elif t2sym == '-jiba':
        out[j, i] = -y.transpose(0, 2, 1)
    else:
        out[j, i] = -y
    return out


class _CCShim:
    """What the transplanted EOM functions read from ``eom._cc``."""

    def __init__(self, mycc, imds):
        self.mycc, self.imds = mycc, imds

    def _add_vvvv(self, t1, t2, eris, out=None, with_ovvv=False,
                  t2sym=None):
        assert t1 is None and not with_ovvv
        return _ladder_sym(self.mycc, self.imds.orbv_s, t2, t2sym)


class _ERIShim:
    def __init__(self, imds):
        self.fock = imds.fock
        self.ovov, self.ovoo, self.ovvo = imds.ovov, imds.ovoo, imds.ovvo
        self.oovv, self.oooo = imds.oovv, imds.oooo
        self._ovvv = imds.ovvv
        self.vvvv = None              # AO-direct: no (aa|bb) in the diagonal

    def get_ovvv(self, *slices):
        return self._ovvv[slices]


def _otril(n, k=-1):
    return cp.tril_indices(n, k=k)


def _pack_triplet(r1, r2):
    r2aa, r2ab = r2
    no, nv = r1.shape
    nov = no * nv
    oi, oj = _otril(no)
    vi, vj = _otril(nv)
    aa = r2aa[oi, oj][:, vi, vj]
    i, j = cp.tril_indices(nov)
    ab = r2ab.transpose(0, 2, 1, 3).reshape(nov, nov)[i, j]
    return cp.hstack([r1.ravel(), aa.ravel(), ab])


def _unpack_triplet(vec, nmo, no):
    nv = nmo - no
    nov = no * nv
    oi, oj = _otril(no)
    vi, vj = _otril(nv)
    npo, npv = len(oi), len(vi)
    r1 = vec[:nov].reshape(no, nv)
    aa = vec[nov:nov + npo * npv].reshape(npo, npv)
    r2aa = cp.zeros((no, no, nv, nv), dtype=vec.dtype)
    r2aa[oi[:, None], oj[:, None], vi, vj] = aa
    r2aa[oj[:, None], oi[:, None], vj, vi] = aa
    r2aa[oi[:, None], oj[:, None], vj, vi] = -aa
    r2aa[oj[:, None], oi[:, None], vi, vj] = -aa
    i, j = cp.tril_indices(nov)
    m = cp.zeros((nov, nov), dtype=vec.dtype)
    # PySCF's unpack_tril(filltriu=ANTIHERMI): the upper triangle *and the
    # diagonal* are minus the packed lower triangle
    m[i, j] = vec[nov + npo * npv:]
    m[j, i] = -vec[nov + npo * npv:]
    r2ab = m.reshape(no, nv, no, nv).transpose(0, 2, 1, 3)
    return r1, (r2aa, cp.ascontiguousarray(r2ab))


def _pack_sf(r1, r2):
    baaa, aaba = r2
    no, nv = r1.shape
    vi, vj = _otril(nv)
    oi, oj = _otril(no)
    b = baaa.reshape(no * no, nv, nv)[:, vi, vj]
    a = aaba[oi, oj]
    return cp.hstack([r1.ravel(), b.ravel(), a.ravel()])


def _unpack_sf(vec, nmo, no):
    nv = nmo - no
    nov = no * nv
    vi, vj = _otril(nv)
    oi, oj = _otril(no)
    nb = no * no * len(vi)
    na = len(oi) * nv * nv
    r1 = vec[:nov].reshape(no, nv)
    v = vec[nov:nov + nb].reshape(no * no, len(vi))
    baaa = cp.zeros((no * no, nv, nv), dtype=vec.dtype)
    baaa[:, vi, vj] = v
    baaa[:, vj, vi] = -v
    v = vec[nov + nb:nov + nb + na].reshape(len(oi), nv, nv)
    aaba = cp.zeros((no, no, nv, nv), dtype=vec.dtype)
    aaba[oi, oj] = v
    aaba[oj, oi] = -v
    return r1, (baaa.reshape(no, no, nv, nv), aaba)


def _eom_module():
    from pyscf.cc import eom_rccsd as _m
    from electrolites.adc_gpu import _transplant
    mod = _transplant(_m)
    g = mod.eeccsd_diag.__globals__
    g['amplitudes_to_vector_singlet'] = lambda r1, r2: _pack_singlet(
        r1[None], r2[None])[0]
    g['amplitudes_to_vector_triplet'] = _pack_triplet
    g['amplitudes_to_vector_eomsf'] = _pack_sf
    return mod


_EOMT = None


def _eomt():
    global _EOMT
    if _EOMT is None:
        _EOMT = _eom_module()
    return _EOMT


class _EOMShim:
    def __init__(self, eom, imds, mycc, kind):
        self.nocc, self.nmo = imds.nocc, imds.nocc + imds.nvir
        self.max_memory = eom.max_memory
        self.partition = getattr(eom, 'partition', None)
        self._cc = _CCShim(mycc, imds)
        pack, unpack = {'triplet': (_pack_triplet, _unpack_triplet),
                        'sf': (_pack_sf, _unpack_sf)}[kind]
        self.amplitudes_to_vector = pack
        self.vector_to_amplitudes = unpack


class _EEOtherBase:
    """Triplet / spin-flip EOM-EE: PySCF's equations, device Davidson."""
    _kind = None

    def make_imds(self, eris=None):
        imds = _make_ee(_Imds(self._cc), self._cc)
        imds.eris = _ERIShim(imds)
        return imds

    def get_diag(self, imds=None):
        if imds is None:
            imds = self.make_imds()
        d = _eomt().eeccsd_diag(self, imds)
        return cp.asnumpy(d[1] if self._kind == 'triplet' else d[2])

    def _sigma_fn(self, imds):
        shim = _EOMShim(self, imds, self._cc, self._kind)
        f = (_eomt().eeccsd_matvec_triplet if self._kind == 'triplet'
             else _eomt().eeccsd_matvec_sf)

        def matvec(V):
            return cp.stack([f(shim, v.copy(), imds) for v in V])
        return matvec

    def gen_matvec(self, imds=None, diag=None, **kwargs):
        if imds is None:
            imds = self.make_imds()
        if diag is None:
            diag = self.get_diag(imds)
        mv = self._sigma_fn(imds)

        def matvec(xs):
            return list(cp.asnumpy(mv(cp.asarray(np.asarray(xs)))))
        return matvec, diag

    def matvec(self, vector, imds=None, diag=None):
        if imds is None:
            imds = self.make_imds()
        return cp.asnumpy(self._sigma_fn(imds)(
            cp.asarray(np.asarray(vector))[None])[0])

    def kernel(self, nroots=1, koopmans=False, guess=None, eris=None,
               imds=None, diag=None):
        log = logger.new_logger(self)
        if imds is None:
            imds = self.make_imds()
        no, nv = imds.nocc, imds.nvir
        d = _eomt().eeccsd_diag(self, imds)
        d = d[1] if self._kind == 'triplet' else d[2]
        mask = None
        if self._kind == 'triplet':
            # the packed (ia, ia) diagonal of the triplet ab block is not a
            # degree of freedom (the antisymmetrised sigma is zero there):
            # keep the trial space out of it
            nov = no * nv
            npair = no * (no - 1) // 2 * (nv * (nv - 1) // 2)
            i, j = cp.tril_indices(nov)
            mask = cp.ones(len(d))
            mask[nov + npair + cp.where(i == j)[0]] = 0
            d = cp.where(mask > 0, d, 1e9)
        nwant = min(len(d), nroots + max(3, nroots // 2))
        if guess is not None:
            x0 = cp.asarray(np.asarray(guess)).reshape(len(guess), -1)
        else:
            idx = cp.argsort(d[:no * nv] if koopmans else d)[:nwant]
            x0 = cp.zeros((nwant, len(d)))
            x0[cp.arange(nwant), idx] = 1.
        sigma = self._sigma_fn(imds)
        if mask is not None:
            raw = sigma

            def sigma(V):
                return raw(V * mask) * mask
        conv, e, X = _davidson_nosym(
            sigma, x0, d, nroots, self.conv_tol, self.conv_tol ** .5,
            self.max_cycle, self.max_space + len(x0), log)
        self.converged, self.e = conv, e
        self.v = list(cp.asnumpy(X))
        for n in range(nroots):
            log.info('EOM-EE-CCSD %s root %d E = %.16g  conv = %s',
                     self._kind, n, e[n], conv[n])
        if nroots == 1:
            return e[0], self.v[0]
        return e, self.v


class EOMEETriplet(_EEOtherBase, eom_rccsd.EOMEETriplet):
    """Triplet EOM-EE-CCSD on the device."""
    _kind = 'triplet'


class EOMEESpinFlip(_EEOtherBase, eom_rccsd.EOMEESpinFlip):
    """Spin-flip EOM-EE-CCSD on the device."""
    _kind = 'sf'


# left IP / EA (PySCF ``lipccsd_matvec`` / ``leaccsd_matvec``), batched

def _lip_sigma(imds, r1, r2):
    Hr1 = -_ein('ki,ni->nk', imds.Loo, r1)
    Hr1 -= _ein('kbij,nijb->nk', imds.Wovoo, r2)
    Hr2 = -_ein('kd,nl->nkld', imds.Fov, r1)
    Hr2 += 2 * _ein('ld,nk->nkld', imds.Fov, r1)
    W = 2 * imds.Wooov - imds.Wooov.transpose(1, 0, 2, 3)
    Hr2 -= _ein('klid,ni->nkld', W, r1)
    Hr2 += _ein('bd,nklb->nkld', imds.Lvv, r2)
    Hr2 -= _ein('ki,nild->nkld', imds.Loo, r2)
    Hr2 -= _ein('lj,nkjd->nkld', imds.Loo, r2)
    W = 2 * imds.Wovvo - imds.Wovov.transpose(0, 1, 3, 2)
    Hr2 += _ein('lbdj,nkjb->nkld', W, r2)
    Hr2 -= _ein('kbdj,nljb->nkld', imds.Wovvo, r2)
    Hr2 += _ein('klij,nijd->nkld', imds.Woooo, r2)
    Hr2 -= _ein('kbid,nilb->nkld', imds.Wovov, r2)
    tmp = _ein('ijcb,nijb->nc', imds.t2, r2)
    W = 2 * imds.Woovv - imds.Woovv.transpose(1, 0, 2, 3)
    Hr2 -= _ein('lkdc,nc->nkld', W, tmp)
    return Hr1, Hr2


def _lea_sigma(imds, mycc, r1, r2):
    t1 = imds.t1
    n, no, nv = r2.shape[0], imds.nocc, imds.nvir
    Hr1 = _ein('ac,na->nc', imds.Lvv, r1)
    Hr1 += _ein('abcj,njab->nc', imds.Wvvvo, r2)
    Hr2 = 2 * _ein('nc,ld->nlcd', r1, imds.Fov)
    Hr2 -= _ein('nd,lc->nlcd', r1, imds.Fov)
    W = 2 * imds.Wvovv - imds.Wvovv.transpose(0, 1, 3, 2)
    Hr2 += _ein('na,alcd->nlcd', r1, W)
    Hr2 += _ein('nlad,ac->nlcd', r2, imds.Lvv)
    Hr2 += _ein('nlcb,bd->nlcd', r2, imds.Lvv)
    Hr2 -= _ein('njcd,lj->nlcd', r2, imds.Loo)
    W = 2 * imds.Wovvo - imds.Wovov.transpose(0, 1, 3, 2)
    Hr2 += _ein('njcb,lbdj->nlcd', r2, W)
    Hr2 -= _ein('lajc,njab->nlcb', imds.Wovov, r2)
    Hr2 -= _ein('lbcj,njab->nlca', imds.Wovvo, r2)
    # sum_ab r[l,a,b] Wvvvv[a,b,c,d]: the ladder ((ac|bd) = (ca|db)) and
    # the tau / t1 pieces of Wvvvv
    ov = imds.orbv_s
    x = _ein('pc,mcd,qd->mpq', ov, r2.reshape(n * no, nv, nv), ov)
    y = _ladder(mycc, x)
    x = None
    Hr2 += _ein('pa,mpq,qb->mab', ov, y, ov).reshape(n, no, nv, nv)
    y = None
    g = _ein('nlab,kmab->nlkm', r2, imds.tau)
    Hr2 += _ein('nlkm,kcmd->nlcd', g, imds.ovov)
    u = _ein('nlab,mb->nlam', r2, t1)
    Hr2 -= _ein('nlam,mdac->nlcd', u, imds.ovvv)
    u = _ein('nlab,ka->nlkb', r2, t1)
    Hr2 -= _ein('nlkb,kcbd->nlcd', u, imds.ovvv)
    u = None
    tmp = _ein('ijcb,nibc->nj', imds.t2, r2)
    W = 2 * imds.Woovv - imds.Woovv.transpose(0, 1, 3, 2)
    Hr2 -= _ein('kjfe,nj->nkef', W, tmp)
    return Hr1, Hr2


def _eomee_ccsd_triplet(self, nroots=1, koopmans=False, guess=None,
                        eris=None, imds=None):
    return EOMEETriplet(self).kernel(nroots, koopmans, guess)


def _eomsf_ccsd(self, nroots=1, koopmans=False, guess=None, eris=None,
                imds=None):
    return EOMEESpinFlip(self).kernel(nroots, koopmans, guess)


for _name, _fn in (('eomee_ccsd_triplet', _eomee_ccsd_triplet),
                   ('eomsf_ccsd', _eomsf_ccsd)):
    if getattr(_ci.CCSDBase, _name, None) is NotImplemented:
        _attach(_ci.CCSDBase, _name, _fn)
