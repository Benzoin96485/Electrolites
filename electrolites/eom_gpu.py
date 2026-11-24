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


def _on_pool(fn):
    """Run ``fn`` on CuPy's memory pool.  GPU4PySCF's allocator sends every
    request above 100 MB to cudaMalloc/cudaFree, which synchronise: o^2 v^2
    arrays at alanine dipeptide are ~80-160 MB, and in the per-vector EOM-EE
    sigma functions those calls were a third of the run."""
    import functools

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # the whole device for the pool, as fastccsd's CCSD (GPU4PySCF caps
        # it at 90 %: 8.5 GB of an A100-80GB the largest EE cases need)
        pool = cp.get_default_memory_pool()
        lim = pool.get_limit()
        pool.set_limit(fraction=1.0)
        try:
            with cp.cuda.using_allocator(pool.malloc):
                return fn(*args, **kwargs)
        finally:
            pool.set_limit(size=lim)
    return wrapped


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
    _vvvv_mo(mycc, nv)
    return imds


def _ladder_setup(mycc):
    """vhfopt (as ``fastccsd``'s) and the AO-block cache on mycc.  The AO
    shell groups are sized so that one integral block [i, nao, j, nao]
    takes at most ~8 % of the free device memory: the fixed nao/4 grouping
    made 15.6 GB blocks at nao 420 (cartesian) and ran the AO-direct ladder
    out of memory beside the EE intermediates."""
    from gpu4pyscf.scf import int4c2e
    from electrolites.fastccsd import _avail
    nao = mycc.mol.nao_nr(cart=True)
    gmax = max(4, (nao + 3) // 4)
    g = int(max(4, min(gmax, (.08 * _avail() / (8 * nao * nao)) ** .5)))
    vhfopt = getattr(mycc, '_fastccsd_vhfopt', None)
    if vhfopt is not None and vhfopt._fastccsd_blk is not None and \
            vhfopt._fastccsd_blk > g and getattr(mycc, '_fastccsd_eri', None) is None:
        vhfopt = None                   # CCSD's grouping, too big here
    if vhfopt is None:
        from electrolites.fastccsd import _vhfopt
        vhfopt = _vhfopt(mycc.mol, g)
        vhfopt._fastccsd_blk = g
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


def _vvvv_mo(mycc, nv):
    """(ac|bd) as a (v^2, v^2) matrix [(a,b), (c,d)] on the device, or None
    when it would take more than 45 % of the free device memory."""
    from electrolites import fastmp2
    from electrolites.fastccsd import _avail
    key = getattr(mycc, '_eom_vvvv', None)
    if key is not None and key[0] == nv:
        return key[1]
    if getattr(mycc, '_eom_ao_ladder', False):
        return None
    # the v^4 array is built in slices of its first index straight into
    # its final [(a,b), (c,d)] layout, so the peak is v^4 plus one slice
    # (a full (ac|bd) build and its transposed copy, 3 v^4 with the
    # workspace, sent v >= 198 -- 4-6 of 9 TM complexes at cc-pVDZ -- to
    # the AO-direct ladder, 75-78 % of their EOM time on an A100-80GB)
    need = nv ** 4 * 8
    avail = _avail()
    if need > .45 * avail:
        mycc._eom_ao_ladder = True
        return None
    mask = mycc.get_frozen_mask()
    mo = cp.asarray(mycc.mo_coeff)[:, cp.asarray(mask)]
    orbv = mo[:, mo.shape[1] - nv:]
    na = max(1, min(nv, int(.25 * (avail - need) // (nv ** 3 * 8))))
    try:
        W = cp.empty((nv * nv, nv * nv))
        Wv = W.reshape(nv, nv, nv, nv)
        for a0 in range(0, nv, na):
            a1 = min(nv, a0 + na)
            v = cp.asarray(fastmp2.get_int4c2e_ovpq(
                mycc.mol, orbv[:, a0:a1], orbv, orbv, orbv))  # [a,c,b,d]
            Wv[a0:a1] = v.transpose(0, 2, 1, 3)
            v = None
        Wv = None
    except (cp.cuda.memory.OutOfMemoryError, RuntimeError):
        v = W = Wv = None
        cp.get_default_memory_pool().free_all_blocks()
        mycc._eom_ao_ladder = True
        return None
    v = None
    mycc._eom_vvvv = (nv, W)
    return W


def _ladder_mo(mycc, ov, x):
    """y[m,a,b] = sum_cd (ac|bd) x[m,c,d] for MO matrices x (m, v, v).

    With the MO (ac|bd) on the device (v^4 doubles, kept on ``mycc`` when
    they fit in a third of the free memory) this is one GEMM over v^4
    instead of the AO-direct contraction over nao^4 (alanine dipeptide:
    v = 176 against nao = 200, 1.7x fewer flops, no integral-block loop);
    otherwise the AO-direct ladder through ``fastccsd``'s blocks."""
    nv = x.shape[-1]
    if not len(x):
        return cp.zeros_like(x)
    W = None if getattr(mycc, '_eom_ao_ladder', False) else \
        _vvvv_mo(mycc, nv)
    if W is not None:
        return (x.reshape(len(x), -1) @ W).reshape(x.shape)
    X = _ein('pe,mef,qf->mpq', ov, x, ov)
    Y = _ladder(mycc, X)
    X = None
    return _ein('pa,mpq,qb->mab', ov, Y, ov)


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
    Hr2 += _ladder_mo(mycc, ov, r2.reshape(n * no, nv, nv)).reshape(
        n, no, nv, nv)
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

    @_on_pool
    def kernel(self, *args, **kwargs):
        return eom_rccsd.EOMEA.kernel(self, *args, **kwargs)

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

    @_on_pool
    def kernel(self, *args, **kwargs):
        return eom_rccsd.EOMIP.kernel(self, *args, **kwargs)

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


def _make_ee(imds, mycc, keep_eris=False):
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

    # wvOvV (v^3 o) built in slices of its first index e: the unsliced build
    # held three more v^3 o arrays (w, its transposed copy, ebmf terms) and
    # ran an 80 GB device out of memory at o 66, v 288
    from electrolites.fastccsd import _avail
    nv = ovvv.shape[1]
    wvOvV = cp.empty((nv, no, nv, nv))
    ce = max(1, min(nv, int(.03 * _avail() // (8 * no * nv * nv))))
    import os
    ce = int(os.environ.get('EOM_WVOVV_CHUNK', 0)) or ce     # tests: force slices
    for e0 in range(0, nv, ce):
        e1 = min(nv, e0 + ce)
        # ebmf[e,b,m,f] = ovvv[m,e,b,f]
        eb = ovvv[:, e0:e1].transpose(1, 2, 0, 3)          # [e,b,m,f]
        w = _ein('ebmf,miaf->eiab', eb, t2)
        out = wvOvV[e0:e1]
        out[...] = -.5 * w.transpose(0, 1, 3, 2)
        out -= w
        w = None
        # + ebmf.transpose(1,2,3,0): [e,i,a,b] = ebmf[b,e,i,a] = ovvv[i,b,e,a]
        out += ovvv[:, :, e0:e1].transpose(2, 0, 3, 1)
        # tmp[e,f,m,b] = -.5 ebmf[e,f,m,b] + ebmf[f,e,m,b]
        tmp = -.5 * eb + ovvv[:, :, e0:e1].transpose(2, 1, 0, 3)
        eb = None
        out += _ein('efmb,mifa->eiba', tmp, theta)
        tmp = None
        out += _ein('meni,mnab->eiab', ovoo[:, e0:e1], tau)
        out -= _ein('me,miab->eiab', Fov[:, e0:e1], t2)
        out += _ein('ma,eimb->eiab', t1, tmpovvo[e0:e1])
        out += _ein('ma,eimb->eiba', t1, tmpoovv[e0:e1])
        out = None
    tmpovvo = tmpoovv = None

    imds.Foo, imds.Fov, imds.Fvv = Foo, Fov, Fvv
    imds.woOoO, imds.woVoO, imds.woOoV = woOoO, woVoO, woOoV
    imds.woVvO, imds.woVVo, imds.wvOvV = woVvO, woVVo, wvOvV
    imds.tau_half = tau * .5
    mask = mycc.get_frozen_mask()
    mo = cp.asarray(mycc.mo_coeff)[:, cp.asarray(mask)]
    imds.orbv = mo[:, no:]
    imds.orbv_s = _ladder_setup(mycc).coeff @ imds.orbv
    _vvvv_mo(mycc, imds.nvir)
    if not keep_eris:   # the singlet sigma and diagonal do not read these
        imds.oovv = imds.oooo = imds.ovoo = None
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
    y = _ladder_mo(mycc, ov, xt.reshape(-1, nv, nv)).reshape(xt.shape)
    out = cp.empty_like(x)
    out[:, i, j] = y
    out[:, j, i] = y.transpose(0, 1, 3, 2)
    return out.reshape(lead + (no, no, nv, nv))


def _ee_ladder(imds, mycc, x):
    """sum_ef (ae|bf) x[n,i,j,e,f] for a batch, AO-direct (x is
    pair-symmetric: tau2 = r2 + r1 t1 + t1 r1)."""
    return _ladder_pairsym(mycc, imds.orbv_s, x)


def _vslice(n, per_bytes, frac=.04):
    """(i0, i1) slices of range(n) with about frac of the free memory per
    slice, per_bytes being the temporary bytes per index"""
    import os
    from electrolites.fastccsd import _avail
    step = max(1, min(n, int(frac * _avail() // max(per_bytes, 1))))
    step = int(os.environ.get('EOM_EIN_SLICE', 0)) or step      # tests
    return [(i0, min(n, i0 + step)) for i0 in range(0, n, step)]


def _ein_o(sub, A, B, b_axis, out_axis=None, big=0):
    """einsum(sub, A, B) (or (B, A) when ``big=1``) in slices of the first
    index of the v^3 o operand (``A`` / ``B``): cupy.einsum copies an
    operand it has to transpose, and a whole ovvv / wvOvV copy (12.6 GB at
    o 66, v 288) did not fit beside the EE intermediates.  The sliced index
    is summed over (the slices accumulate) unless ``out_axis`` gives its
    position in the output (the slices are written there).  ``b_axis`` is
    the position of the same index in the other operand (None: absent)."""
    from electrolites.fastccsd import _avail
    W = B if big else A
    n0 = W.shape[0]
    per = W[0].size * 8 * 3
    step = max(1, min(n0, int(.04 * _avail() // per)))
    import os
    step = int(os.environ.get('EOM_EIN_SLICE', 0)) or step      # tests
    if step >= n0:
        return _ein(sub, A, B)
    lhs, rhs = sub.split('->')
    sa, sb = lhs.split(',')
    out = None
    for i0 in range(0, n0, step):
        i1 = min(n0, i0 + step)
        Wi = W[i0:i1]
        if b_axis is None:
            Oi = B if not big else A
        else:
            sl = [slice(None)] * (B.ndim if not big else A.ndim)
            sl[b_axis] = slice(i0, i1)
            Oi = (B if not big else A)[tuple(sl)]
        r = _ein(sub, Wi, Oi) if not big else _ein(sub, Oi, Wi)
        if out_axis is None:
            if out is None:
                out = r
            else:
                out += r
        else:
            if out is None:
                shape = list(r.shape)
                shape[out_axis] = n0
                out = cp.empty(shape)
            sl = [slice(None)] * out.ndim
            sl[out_axis] = slice(i0, i1)
            out[tuple(sl)] = r
    return out


def _ee_sigma(imds, mycc, r1, r2, ladder=True, skip=()):
    """eeccsd_matvec_singlet for a batch: r1 (n, o, v), r2 (n, o, o, v, v).
    ``ladder=False`` leaves out sum_ef (ae|bf) tau2[ijef] (tau2 = r2 + r1 t1
    + t1 r1), whose whole contribution to Hr2 is that term itself (the
    rank-reduced EOM adds it factorised); a callable is used as the ladder
    (tau2 -> its ladder).  ``skip`` leaves out more of the doubles, for
    ``rreom_gpu`` to add in factorised form (Hr2 = X + X^T, X[ia,jb] =
    X[i,j,a,b] as an ov x ov matrix, M the same for r2):

    * 'ring': the uncrossed rings, X = 2 M Wc + P Wx with Wc[me,jb] =
      (woVvO + woVVo/2)[m,b,e,j], Wx[me,jb] = -woVvO[m,b,e,j],
      P[ia,me] = r2[i,m,e,a] (the crossed woVVo ring stays);
    * 'dress': the T1 dressing of the ladder, -t1[m,a] (me|bf) tau2[ijef]
      and its t1 t1 (me|nf) part (in tau_half), which with the ladder make
      sum_ef (ae^|bf^) tau2[ijef] in the T1-transformed integrals."""
    t1, t2 = imds.t1, imds.t2
    ovvv, ovov = imds.ovvv, imds.ovov
    ring, dress = 'ring' in skip, 'dress' in skip
    Hr1 = _ein('ae,nie->nia', imds.Fvv, r1)
    Hr1 -= _ein('mi,nma->nia', imds.Foo, r1)
    Hr1 += _ein('me,nimae->nia', imds.Fov, r2) * 2
    Hr1 -= _ein('me,nimea->nia', imds.Fov, r2)

    rt = _ein('nia,jb->nijab', r1, t1)
    tau2 = r2 + rt + rt.transpose(0, 2, 1, 4, 3)
    rt = None
    assert not dress or ladder is not True
    if callable(ladder):          # e.g. the T1-dressed one ('dress')
        Hr2 = ladder(tau2)
    else:
        Hr2 = _ee_ladder(imds, mycc, tau2) if ladder else cp.zeros_like(r2)
    # the two other tau2 terms now, so that tau2 (o^2 v^2) can go
    tmp_t = None if dress else \
        _ein_o('meaf,nijef->nmaij', ovvv, tau2, None, out_axis=1)
    tmp_end = _ein('menf,kijef->kmnij', ovov, tau2)
    tau2 = None
    Hr2 += _ein('mnij,kmnab->kijab', imds.woOoO, r2)
    Hr2 *= .5
    Hr2 += _ein('be,nijae->nijab', imds.Fvv, r2)
    Hr2 -= _ein('mj,nimab->nijab', imds.Foo, r2)

    theta = r2 * 2 - r2.transpose(0, 1, 2, 4, 3)
    Hr1 += _ein_o('mfae,nmife->nia', ovvv, theta, 1)
    if tmp_t is not None:
        Hr2 -= _ein('ma,nmbij->nijab', t1, tmp_t)
    tmp_t = None
    tmp = _ein_o('meaf,nme->naf', ovvv, r1, 1) * 2
    tmp -= _ein_o('mfae,nme->naf', ovvv, r1, 1)
    Hr2 += _ein('naf,ijfb->nijab', tmp, t2)
    Hr2 -= _ein('mbij,nma->nijab', imds.woVoO, r1)
    Hr2 += _ein_o('ejab,nie->nijab', imds.wvOvV, r1, 2)

    tmp = _ein('mbej,nimea->njiab', imds.woVVo, r2)
    Hr2 += tmp
    if not ring:
        Hr2 += tmp.transpose(0, 1, 2, 4, 3) * .5
    tmp = None
    woVvO = imds.woVVo * .5 + imds.woVvO
    Hr1 += _ein('maei,nme->nia', woVvO, r1) * 2
    if not ring:
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

    Hr2 += _ein('kmnij,mnab->kijab', tmp_end,
                t2 * .5 if dress else imds.tau_half)
    tmp_end = None
    Hr2 += Hr2.transpose(0, 2, 1, 4, 3).copy()
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
    tmp = _ein_o('mb,mbaa->ab', t1, ovvv, 0, big=1)
    Wvvab = Wvvab - tmp - tmp.T + _virtual_coulomb_diag(mycc, imds.orbv)
    Hr2 = Hr2 + Wvvab[None, None]
    return Hr1, Hr2


# packed singlet layout without index arrays or transposed copies: element
# k of the triangle is (r, c), r >= c, row-major (cupy.tril_indices order),
# r = (i, a), c = (j, b), value t2[i, j, a, b].  The index arrays of
# tril_indices(nov) alone were 2.9 GB at o 66, v 288.
_TRI_RC = r"""
    long long r = (long long)((sqrt(8.0 * (double)k + 1.0) - 1.0) * 0.5);
    while (r * (r + 1) / 2 > k) r--;
    while ((r + 1) * (r + 2) / 2 <= k) r++;
    long long c = k - r * (r + 1) / 2;
    long long oi = r / nv, va = r % nv, oj = c / nv, vb = c % nv;
    long long t = (((m * no + oi) * no + oj) * nv + va) * nv + vb;
    long long u = (((m * no + oj) * no + oi) * nv + vb) * nv + va;
"""
_PACK = cp.ElementwiseKernel(
    'raw float64 t2, int64 no, int64 nv, int64 npk, int64 dim, int64 nov',
    'raw float64 out',
    """long long m = i / npk, k = i % npk;""" + _TRI_RC +
    """out[m * dim + nov + k] = t2[t];""",
    'eom_pack_t2_tril')
_UNPACK = cp.ElementwiseKernel(
    'raw float64 v, int64 no, int64 nv, int64 npk, int64 dim, int64 nov',
    'raw float64 t2',
    """long long m = i / npk, k = i % npk;""" + _TRI_RC +
    """double x = v[m * dim + nov + k]; t2[t] = x; t2[u] = x;""",
    'eom_unpack_t2_tril')


def _pack_singlet(r1, r2, out=None):
    """ccsd.amplitudes_to_vector for a batch: t1 + lower triangle of
    t2[ia,jb]."""
    n, no, nv = r1.shape
    nov = no * nv
    npk = nov * (nov + 1) // 2
    dim = nov + npk
    if out is None:
        out = cp.empty((n, dim))
    out[:, :nov] = r1.reshape(n, nov)
    _PACK(cp.ascontiguousarray(r2), no, nv, npk, dim, nov, out, size=n * npk)
    return out


def _unpack_singlet(v, no, nv):
    v = cp.ascontiguousarray(v)
    n = len(v)
    nov = no * nv
    npk = nov * (nov + 1) // 2
    r2 = cp.empty((n, no, no, nv, nv))
    _UNPACK(v, no, nv, npk, v.shape[1], nov, r2, size=n * npk)
    return v[:, :nov].reshape(n, no, nv), r2


def _davidson_nosym(matvec, x0, diag, nroots, tol, tol_residual, max_cycle,
                    max_space, log, ckpt=None):
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

    import os
    nwant, dim = x0.shape
    from electrolites.fastccsd import _avail
    vbytes = 8 * dim
    room = (.5 * _avail() - 4 * nwant * vbytes) // (2 * vbytes)
    if room < 2 * nwant + 4 or os.environ.get('EOM_DAVIDSON_HOST'):
        log.info('EOM Davidson: subspace in pinned host memory (device room '
                 'for %d vectors of %.2f GB)', max(room, 0), vbytes / 1e9)
        return _davidson_nosym_host(matvec, x0, diag, nroots, tol,
                                    tol_residual, max_cycle, max_space, log,
                                    ckpt)
    x0 = cp.asarray(x0)
    V0 = cp.stack(orth(list(x0), None))
    # the first sigma batch builds the persistent pieces (MO vvvv or the AO
    # integral cache) before the subspace is sized against the free memory
    AV0 = matvec(V0)
    # V and AV buffers, X and AX (= R) of nwant rows; half the free memory,
    # the rest for the sigma intermediates
    room = int((.5 * _avail() - 2 * nwant * vbytes) // (2 * vbytes))
    if room < max_space:
        log.warn('EOM Davidson: subspace capped at %d vectors (memory)', room)
        max_space = max(room, 2 * nwant)
    Vb = cp.empty((max_space, dim))
    AVb = cp.empty((max_space, dim))
    k = len(V0)
    Vb[:k] = V0
    AVb[:k] = AV0
    V0 = AV0 = None
    e_last = None
    conv = np.zeros(nwant, dtype=bool)
    for it in range(max_cycle):
        V, AV = Vb[:k], AVb[:k]
        H = cp.asnumpy(V @ AV.T)
        w, c = np.linalg.eig(H)
        # real roots only; of a nearly real complex pair keep one member --
        # both have the same real part of the eigenvector, and keeping the
        # two put identical rows in X
        real = np.where((abs(w.imag) < 1e-3) & (w.imag >= 0))[0]
        w, c = w[real].real, c[:, real].real
        order = np.argsort(w)[:nwant]
        w, c = w[order], c[:, order]
        cd = cp.asarray(c)
        X = cd.T @ V
        R = cd.T @ AV
        nrm = cp.linalg.norm(X, axis=1)
        X /= nrm[:, None]
        R /= nrm[:, None]
        wd = cp.asarray(w)
        R -= wd[:, None] * X
        rnorm = cp.asnumpy(cp.linalg.norm(R, axis=1))
        if e_last is not None and len(e_last) == len(w):
            de = abs(w - e_last)
        else:
            de = np.ones(len(w))
        conv = (de < tol) & (rnorm < tol_residual)
        log.debug('EOM Davidson %d  space %d  |r| %s  dE %s', it, k,
                  rnorm[:nroots], de[:nroots])
        e_last = w
        if conv[:nroots].all():
            break
        new = []
        for j in np.where(~conv[:nroots])[0]:
            d = wd[j] - diag
            d = cp.where(abs(d) < 1e-8, 1e-8, d)
            new.append(R[j] / d)
        R = None
        if k + len(new) > max_space:
            # collapse onto span(X) through an orthonormal basis of the
            # Ritz coefficients: rank-deficient X (near-equal Ritz vectors)
            # made a QR + solve(r^T, AX) collapse give rows with V != 0 and
            # AV == 0, i.e. a spurious exact zero root
            u, sv, _ = np.linalg.svd(c, full_matrices=False)
            U = cp.asarray(u[:, sv > 1e-8 * sv[0]])
            m = U.shape[1]
            Vb[:m] = U.T @ V
            AVb[:m] = U.T @ AV
            k = m
        new = orth(new, Vb[:k])
        if not new:
            break
        n = min(len(new), max_space - k)
        Vb[k:k + n] = cp.stack(new[:n])
        new = None
        AVb[k:k + n] = matvec(Vb[k:k + n])
        k += n
    return conv[:nroots], e_last[:nroots], X[:nroots]


class _HostRows:
    """(nrow, dim) float64 in pinned host memory, stored as column blocks
    (each block (nrow, cs) contiguous), so that a column chunk of the
    leading rows is one contiguous H2D copy."""

    def __init__(self, nrow, dim, cs):
        import cupyx
        self.dim, self.cs = dim, cs
        self.cols = [(c0, min(dim, c0 + cs)) for c0 in range(0, dim, cs)]
        self.blk = [cupyx.empty_pinned((nrow, c1 - c0)) for c0, c1 in self.cols]

    def chunk(self, b, r0, r1):
        """rows r0:r1 of column block b, on the device"""
        return cp.asarray(self.blk[b][r0:r1])

    def put_chunk(self, b, r0, r1, x):
        x.get(out=self.blk[b][r0:r1]) if x.flags.c_contiguous else \
            cp.ascontiguousarray(x).get(out=self.blk[b][r0:r1])

    def set_rows(self, r0, x):
        for b, (c0, c1) in enumerate(self.cols):
            self.put_chunk(b, r0, r0 + len(x), x[:, c0:c1])

    def rows(self, r0, r1):
        out = cp.empty((r1 - r0, self.dim))
        for b, (c0, c1) in enumerate(self.cols):
            out[:, c0:c1] = self.chunk(b, r0, r1)
        return out

    def host_row(self, r):
        return np.concatenate([blk[r] for blk in self.blk])


def _davidson_nosym_host(matvec, x0, diag, nroots, tol, tol_residual,
                         max_cycle, max_space, log, ckpt=None):
    """``_davidson_nosym`` with the subspace (V, AV) in pinned host memory,
    for vectors too long to keep a subspace on the device (Pt EWANIZ,
    o 66, v 288: 1.45 GB per singlet vector).  The device holds the
    integrals, the diagonal, the new block (or one sigma sub-block) and one
    column chunk of V/AV at a time:

    * H = V AV^T is kept and extended by the new rows/columns;
    * Ritz vectors and the residuals of the first nroots roots come from one
      streamed pass over V and AV (both kept on the host);
    * new vectors: two streamed classical Gram-Schmidt passes against V,
      then among themselves on the device;
    * a collapse is U^T V and U^T AV chunk by chunk, H -> U^T H U.
    Same convergence test and root selection as the device version."""
    import os
    from electrolites.fastccsd import _avail
    nwant, dim = x0.shape
    max_space = max(max_space, 2 * nwant)
    cs = int(max(1 << 20, min(dim, 1e9 // (8 * max_space))))
    V = _HostRows(max_space, dim, cs)
    AV = _HostRows(max_space, dim, cs)
    X = _HostRows(nwant, dim, cs)
    R = _HostRows(nroots, dim, cs)
    nb = len(V.cols)
    H = np.zeros((max_space, max_space))
    diag = cp.asarray(diag)
    k = 0

    def project_out(B):
        """B (n, dim) on the device: two classical Gram-Schmidt passes
        against V[:k], then among the rows of B; drops dependent rows."""
        for _ in range(2):
            if k:
                P = cp.zeros((k, len(B)))
                for b, (c0, c1) in enumerate(V.cols):
                    P += V.chunk(b, 0, k) @ B[:, c0:c1].T
                for b, (c0, c1) in enumerate(V.cols):
                    B[:, c0:c1] -= P.T @ V.chunk(b, 0, k)
        keep = []
        for i in range(len(B)):
            for _ in range(2):
                for j in keep:
                    B[i] -= B[j] * (B[j] @ B[i])
            nrm = float(cp.linalg.norm(B[i]))
            if nrm > 1e-7:
                B[i] /= nrm
                keep.append(i)
        return B[keep] if len(keep) < len(B) else B

    def add(n):
        """append the n orthonormal rows already written to V[k:k+n] and
        their sigma to V, AV, H; the rows come back from the host one sigma
        block at a time, so the device holds one block and its sigma
        intermediates beside the integrals (the caller drops its device
        copy of the rows before calling)"""
        nonlocal k
        sb = max(1, min(n, int(.05 * _avail() // (8 * dim))))
        for j0 in range(0, n, sb):
            j1 = min(n, j0 + sb)
            AV.set_rows(k + j0, matvec(V.rows(k + j0, k + j1)))
        kn = k + n
        Hc = np.zeros((kn, n))          # V[:kn] . AV[k:kn]^T
        Hr = np.zeros((n, k))           # V[k:kn] . AV[:k]^T
        for b in range(nb):
            Vc = V.chunk(b, 0, kn)
            Ac = AV.chunk(b, 0, kn)
            Hc += cp.asnumpy(Vc @ Ac[k:].T)
            if k:
                Hr += cp.asnumpy(Vc[k:] @ Ac[:k].T)
            Vc = Ac = None
        H[:kn, k:kn] = Hc
        H[k:kn, :k] = Hr
        k = kn

    x0 = np.asarray(x0.get() if isinstance(x0, cp.ndarray) else x0)
    step = max(1, nroots)
    for p0 in range(0, nwant, step):
        B = project_out(cp.asarray(x0[p0:p0 + step]))
        n = len(B)
        if n:
            V.set_rows(k, B)
            B = None
            add(n)
        B = None
    e_last = None
    conv = np.zeros(nwant, dtype=bool)
    xnrm = np.ones(nwant)
    for it in range(max_cycle):
        w, c = np.linalg.eig(H[:k, :k])
        real = np.where((abs(w.imag) < 1e-3) & (w.imag >= 0))[0]
        w, c = w[real].real, c[:, real].real
        order = np.argsort(w)[:nwant]
        w, c = w[order], c[:, order]
        nw = len(w)
        nr = min(nroots, nw)
        cd = cp.asarray(c)
        wd = cp.asarray(w)
        xn2 = np.zeros(nw)
        rn2 = np.zeros(nw)
        for b in range(nb):
            Xc = cd.T @ V.chunk(b, 0, k)
            Rc = cd.T @ AV.chunk(b, 0, k)
            Rc -= wd[:, None] * Xc
            xn2 += cp.asnumpy((Xc * Xc).sum(1))
            rn2 += cp.asnumpy((Rc * Rc).sum(1))
            X.put_chunk(b, 0, nw, Xc)
            R.put_chunk(b, 0, nr, Rc[:nr])
            Xc = Rc = None
        xnrm = np.sqrt(xn2)
        rnorm = np.sqrt(rn2) / xnrm
        if e_last is not None and len(e_last) == len(w):
            de = abs(w - e_last)
        else:
            de = np.ones(len(w))
        conv = (de < tol) & (rnorm < tol_residual)
        log.debug('EOM Davidson (host) %d  space %d  |r| %s  dE %s', it, k,
                  rnorm[:nroots], de[:nroots])
        e_last = w
        if ckpt and it % 3 == 2:
            # the current Ritz vectors: a restart passes them as guess
            np.save(ckpt + '.tmp.npy', np.asarray(
                [X.host_row(i) / xnrm[i] for i in range(nw)]))
            os.replace(ckpt + '.tmp.npy', ckpt)
        if conv[:nroots].all():
            break
        todo = np.where(~conv[:nr])[0]
        B = cp.empty((len(todo), dim))
        for i, j in enumerate(todo):
            B[i] = R.rows(j, j + 1)[0]
            dd = wd[j] - diag
            dd = cp.where(abs(dd) < 1e-8, 1e-8, dd)
            B[i] /= dd * xnrm[j]
            dd = None
        if k + len(B) > max_space:
            u, sv, _ = np.linalg.svd(c, full_matrices=False)
            U = u[:, sv > 1e-8 * sv[0]]
            m = U.shape[1]
            Ud = cp.asarray(U)
            for b in range(nb):
                V.put_chunk(b, 0, m, Ud.T @ V.chunk(b, 0, k))
                AV.put_chunk(b, 0, m, Ud.T @ AV.chunk(b, 0, k))
            H[:m, :m] = U.T @ H[:k, :k] @ U
            k = m
        B = project_out(B)
        n = min(len(B), max_space - k)
        if not n:
            break
        V.set_rows(k, B[:n])
        B = None
        add(n)
    Xr = np.asarray([X.host_row(i) / xnrm[i] for i in range(nroots)])
    return conv[:nroots], e_last[:nroots], Xr


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

    @_on_pool
    def kernel(self, nroots=1, koopmans=False, guess=None, eris=None,
               imds=None, diag=None):
        if imds is not None:
            return self._kernel(nroots, koopmans, guess, imds)
        return _drop_vvvv_on_oom(self._cc, logger.new_logger(self),
                                 self._kernel, nroots, koopmans, guess, None)

    def _kernel(self, nroots, koopmans, guess, imds):
        log = logger.new_logger(self)
        if imds is None:
            imds = self.make_imds()
        no, nv = imds.nocc, imds.nvir
        mycc = self._cc
        h1, h2 = _ee_diag(imds, mycc)
        d = _pack_singlet(h1[None], h2[None])[0]
        h1 = h2 = None
        self._diag = cp.asnumpy(d)       # for the left solve (same diagonal)
        nwant = min(len(d), nroots + max(3, nroots // 2))
        if guess is not None:
            x0 = np.asarray(guess).reshape(len(guess), -1)
            nwant = len(x0)
        else:
            idx = cp.argsort(d[:no * nv] if koopmans else d)[:nwant]
            x0 = np.zeros((nwant, len(d)))
            x0[np.arange(nwant), cp.asnumpy(idx)] = 1.

        def matvec(V):
            out = cp.empty_like(V)
            blk = _ee_blk(no, nv)
            for p0 in range(0, len(V), blk):
                r1, r2 = _unpack_singlet(V[p0:p0 + blk], no, nv)
                h1, h2 = _ee_sigma(imds, mycc, r1, r2)
                r1 = r2 = None
                out[p0:p0 + blk] = _pack_singlet(h1, h2)
                h1 = h2 = None
                blk = _ee_blk(no, nv)
            return out

        conv, e, X = _davidson_nosym(
            matvec, x0, d, nroots, self.conv_tol,
            self.conv_tol ** .5, self.max_cycle,
            self.max_space + nwant, log,
            getattr(self, 'davidson_checkpoint', None))
        self.converged, self.e = conv, e
        self.v = list(cp.asnumpy(X))
        for n in range(nroots):
            log.info('EOM-EE-CCSD singlet root %d E = %.16g  conv = %s', n,
                     e[n], conv[n])
        if nroots == 1:
            return e[0], self.v[0]
        return e, self.v


# ------------------------------- left singlet EOM-EE, transition properties
#
# PySCF has neither for RCCSD.  The left operator is the λ-linear part of
# the Λ equations (``ccsd_lambda_gpu.lambda_rhs``): lambda_rhs(λ) - D λ =
# η + λ (H̄ - E_CC), so L (H̄ - E_CC) = lambda_rhs(L) - lambda_rhs(0) - D L,
# with L in the same (spin-adapted) representation as Λ.  The pairing of a
# Λ-like bra with a T-like ket is
#     <L|R> = 2 sum l1 r1 + sum (2 l2[ijab] - l2[ijba]) r2[ijab].
# Transition densities come from the CCSD 1-RDM functional
# γ(λ; T) = <0|(1+λ) e^-T E_pq e^T|0> (``ccsd_rdm_gpu``):
#   left   <0|L_k Ē|0>          = γ(L_k; T) - γ(0; T)
#   right  <0|(1+Λ) Ē (r0+R)|0> = r0 [γ(Λ;T) - γ(0;T)]
#                                 + d/de γ(Λ; T + e R)|e=0
#                                 + γ(ζ, 0; T) - γ(0; T)
# where the derivative is the commutator <0|(1+Λ)[Ē, R]|0>, the last term
# is <0|Λ R Ē|0> with <0|Λ2 R1 = sum ζ_jb <S_jb|,
# ζ_jb = sum_ia (2 l2[ijab] - l2[ijba]) r1[ia], and r0 = -<Λ|R> makes the
# state biorthogonal to the ground state.  γ is a polynomial of degree 3 in
# T, so a 5-point central difference gives the derivative exactly.  PySCF's
# dm1 is the symmetrised density, which is all a symmetric operator (the
# dipole) sees.


def _pair(l1, l2, r1, r2):
    """<L|R> for a Λ-like (l1, l2) and a T-like (r1, r2)."""
    return float(2 * cp.vdot(l1, r1) + 2 * cp.vdot(l2, r2)
                 - cp.vdot(l2.transpose(0, 1, 3, 2), r2))


def _gamma(mycc, t1, t2, l1, l2):
    """Active-space spin-traced 1-RDM functional γ(λ; T) without the
    reference occupations (PySCF ``ccsd_rdm``, device)."""
    from electrolites import ccsd_rdm_gpu
    m = ccsd_rdm_gpu._mod()
    d1 = m._gamma1_intermediates(mycc, t1, t2, l1, l2)
    return m._make_rdm1(mycc, d1, with_frozen=False, with_mf=False)


def _drop_vvvv_on_oom(mycc, log, fn, *args):
    """Run ``fn``; if the device runs out of memory while the MO (ac|bd)
    (v^4 doubles) is held, drop it and run ``fn`` again with the AO-direct
    ladder (Rh EDOYAZ, v = 226 on an 80 GB A100: (ac|bd) 21 GB + the EE
    intermediates left room for 6 Davidson vectors and the sigma failed)."""
    try:
        return fn(*args)
    except cp.cuda.memory.OutOfMemoryError:
        if getattr(mycc, '_eom_vvvv', None) is None:
            raise
    log.warn('EOM-EE: out of device memory with the MO vvvv held; '
             'repeating with the AO-direct ladder')
    del mycc._eom_vvvv
    mycc._eom_ao_ladder = True
    cp.get_default_memory_pool().free_all_blocks()
    return fn(*args)


@_on_pool
def _left_kernel(eom, nroots, guess=None):
    """Left singlet EOM-EE-CCSD eigenvectors (Λ-like representation),
    packed as the right ones; ``guess`` (e.g. the right vectors) seeds the
    Davidson.  If the device runs out of memory with the MO (ac|bd) held,
    it is dropped and the solve repeated with the AO-direct ladder."""
    log = logger.new_logger(eom)
    return _drop_vvvv_on_oom(eom._cc, log, _left_solve, eom, nroots, guess,
                             log)


def _left_solve(eom, nroots, guess, log):
    from electrolites import ccsd_lambda_gpu as LG
    mycc = eom._cc
    d = getattr(eom, '_diag', None)
    if d is None:
        imds = _make_ee(_Imds(mycc), mycc)
        h1, h2 = _ee_diag(imds, mycc)
        d = _pack_singlet(h1[None], h2[None])[0]
        h1 = h2 = imds = None
    d = cp.asarray(d)
    imds = LG.make_intermediates(_Imds(mycc))
    no, nv = imds.nocc, imds.nvir

    def matvec(V):
        # all trial vectors of the step through one batched λ-linear
        # update (the per-vector loop was 2x the right sigma's time per
        # vector: o^3 v^3 contractions at one vector each)
        out = cp.empty_like(V)
        blk = _ee_blk(no, nv)
        for p0 in range(0, len(V), blk):
            l1, l2 = _unpack_singlet(V[p0:p0 + blk], no, nv)
            s1, s2 = LG.lambda_linear(imds, mycc, l1, l2)
            l1 = l2 = None
            out[p0:p0 + blk] = _pack_singlet(s1, s2)
            s1 = s2 = None
            blk = _ee_blk(no, nv)
        return out

    nspare = max(3, nroots // 2)
    ng = len(guess) if guess is not None else 0
    idx = cp.asnumpy(cp.argsort(d)[:nroots + nspare])
    x0 = np.zeros((ng + (nspare if ng else nroots + nspare), len(d)))
    if ng:
        x0[:ng] = np.asarray(guess).reshape(ng, -1)
        idx = idx[:nspare]
    x0[np.arange(ng, len(x0)), idx] = 1.
    conv, ev, X = _davidson_nosym(matvec, x0, d, nroots, eom.conv_tol,
                                  eom.conv_tol ** .5, eom.max_cycle,
                                  eom.max_space + len(x0), log,
                                  getattr(eom, 'davidson_checkpoint_left',
                                          None))
    for n in range(nroots):
        log.info('left EOM-EE-CCSD singlet root %d E = %.16g  conv = %s', n,
                 ev[n], conv[n])
    return conv, ev, list(cp.asnumpy(X))


def _dipole_mo(mycc):
    mol = mycc.mol
    mask = mycc.get_frozen_mask()
    mo = np.asarray(mycc.mo_coeff.get() if hasattr(mycc.mo_coeff, 'get')
                    else mycc.mo_coeff)[:, mask]
    charges = mol.atom_charges()
    origin = charges @ mol.atom_coords() / charges.sum()
    with mol.with_common_orig(origin):
        dip = mol.intor_symmetric('int1e_r', comp=3)
    return cp.asarray(np.einsum('xpq,pi,qj->xij', dip, mo, mo))


@_on_pool
def transition_properties(mycc, e, rvecs, lvecs, l1=None, l2=None):
    """Transition dipoles <0|mu|k> (right), <k|mu|0> (left) and oscillator
    strengths f = 2/3 w sum_x <0|mu_x|k><k|mu_x|0> for singlet EOM-EE-CCSD
    states (packed right/left vectors as from ``EOMEESinglet``)."""
    from electrolites import ccsd_lambda_gpu
    t1, t2 = cp.asarray(mycc.t1), cp.asarray(mycc.t2)
    if l1 is None or l2 is None:
        if getattr(mycc, 'l1', None) is None:
            ccsd_lambda_gpu.kernel(mycc, tol=mycc.conv_tol_normt)
        l1, l2 = mycc.l1, mycc.l2
    l1, l2 = cp.asarray(l1), cp.asarray(l2)
    no, nv = t1.shape
    dip = _dipole_mo(mycc)
    z1, z2 = cp.zeros_like(t1), cp.zeros_like(t2)
    g0 = _gamma(mycc, t1, t2, z1, z2)
    gl = _gamma(mycc, t1, t2, l1, l2) - g0
    out = dict(e=np.asarray(e, dtype=float), mu_0k=[], mu_k0=[], f=[], r0=[])
    for w, rv, lv in zip(e, rvecs, lvecs):
        r1, r2 = _unpack_singlet(cp.asarray(rv)[None], no, nv)
        x1, x2 = _unpack_singlet(cp.asarray(lv)[None], no, nv)
        r1, r2, x1, x2 = r1[0], r2[0], x1[0], x2[0]
        s = _pair(x1, x2, r1, r2)
        x1 /= s
        x2 /= s
        r0 = -_pair(l1, l2, r1, r2)
        zeta = (2 * _ein('ijab,ia->jb', l2, r1)
                - _ein('ijba,ia->jb', l2, r1))
        h = 1e-2
        gs = [_gamma(mycc, t1 + c * h * r1, t2 + c * h * r2, l1, l2)
              for c in (-2, -1, 1, 2)]
        dg = (gs[0] - 8 * gs[1] + 8 * gs[2] - gs[3]) / (12 * h)
        gs = None
        g_r = r0 * gl + dg + (_gamma(mycc, t1, t2, zeta, z2) - g0)
        g_l = _gamma(mycc, t1, t2, x1, x2) - g0
        m0k = cp.asnumpy(_ein('xpq,pq->x', dip, g_r))
        mk0 = cp.asnumpy(_ein('xpq,pq->x', dip, g_l))
        out['mu_0k'].append(m0k)
        out['mu_k0'].append(mk0)
        out['f'].append(2. / 3 * float(w) * float(m0k @ mk0))
        out['r0'].append(r0)
    for k in ('mu_0k', 'mu_k0', 'f', 'r0'):
        out[k] = np.asarray(out[k])
    return out


@_on_pool
def eomee_singlet_properties(mycc, nroots=1, conv_tol=1e-7, eom=None):
    """Right and left singlet EOM-EE-CCSD roots, the CCSD Λ and the
    transition properties: returns (eom, props) with props as from
    ``transition_properties`` plus 'e_left' and 'converged'."""
    from electrolites import ccsd_lambda_gpu
    if eom is None:
        eom = EOMEESinglet(mycc)
        eom.conv_tol = conv_tol
    e, rv = eom.kernel(nroots=nroots)
    e = np.atleast_1d(e)
    rv = rv if nroots > 1 else [rv]
    if getattr(mycc, 'l1', None) is None:
        ccsd_lambda_gpu.kernel(mycc, tol=mycc.conv_tol_normt)
    convl, el, lv = _left_kernel(eom, nroots, guess=rv)
    # pair each right root with the left root of the same energy
    order = [int(np.argmin(abs(el - x))) for x in e]
    el = el[order]
    lv = [lv[i] for i in order]
    log = logger.new_logger(eom)
    dmax = float(abs(el - e).max())
    if dmax > 10 * eom.conv_tol ** .5:
        log.warn('left/right EOM-EE roots differ by up to %.2e Eh', dmax)
    props = transition_properties(mycc, e, rv, lv)
    props['e_left'] = el
    props['converged'] = np.asarray(eom.converged) & np.asarray(convl)[order]
    for n in range(nroots):
        log.info('EOM-EE-CCSD singlet %d  %.6f eV  f = %.6f  r0 = %.4f', n,
                 e[n] * 27.211386245988, props['f'][n], props['r0'][n])
    return eom, props


#: trial vectors per EE sigma batch (at most)
_EE_BLK = 8


def _ee_blk(no, nv):
    """Trial vectors per singlet sigma batch that fit the free memory: the
    sigma holds ~16 o^2 v^2 arrays per vector at its peak (r2, tau2, Hr2,
    theta, the ladder's pair halves and output, einsum and transpose copies,
    the packing indices).  At o = 47, v = 182 one vector is ~9 GB, and a
    fixed batch of 8 ran an 80 GB device out of memory."""
    from electrolites.fastccsd import _avail
    per = 16 * 8 * (no * nv) ** 2
    return max(1, min(_EE_BLK, int(.8 * _avail() // per)))


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
        return _ladder_mo(mycc, ov, x.reshape(-1, nv, nv)).reshape(x.shape)
    i, j = cp.tril_indices(no)
    xt = x[i, j]
    y = _ladder_mo(mycc, ov, xt)
    out = cp.empty_like(x)
    out[i, j] = y
    if t2sym == 'jiba':
        out[j, i] = y.transpose(0, 2, 1)
    elif t2sym == '-jiba':
        out[j, i] = -y.transpose(0, 2, 1)
    else:
        out[j, i] = -y
    return out


def _ladder_many(mycc, ov, items):
    """``_ladder_sym`` for a list of (x, t2sym) in one AO-direct pass: the
    matrices of every item (i >= j pairs for the symmetric ones) are
    contracted together, so the integral-block GEMMs see thousands of
    columns instead of one vector's."""
    no, nv = items[0][0].shape[0], items[0][0].shape[-1]
    i, j = cp.tril_indices(no)
    mats, plan = [], []
    for x, sym in items:
        xs = x[i, j] if sym in ('jiba', '-jiba', '-jiab') else \
            x.reshape(-1, nv, nv)
        plan.append((len(xs), sym))
        mats.append(xs)
    y = _ladder_mo(mycc, ov, cp.concatenate(mats))
    mats = None
    out, p0 = [], 0
    for (x, sym), (n, _) in zip(items, plan):
        yy = y[p0:p0 + n]
        p0 += n
        if sym not in ('jiba', '-jiba', '-jiab'):
            out.append(yy.reshape(x.shape))
            continue
        o = cp.empty_like(x)
        o[i, j] = yy
        if sym == 'jiba':
            o[j, i] = yy.transpose(0, 2, 1)
        elif sym == '-jiba':
            o[j, i] = -yy.transpose(0, 2, 1)
        else:
            o[j, i] = -yy
        out.append(o)
    return out


class _CCShim:
    """What the transplanted EOM functions read from ``eom._cc``.  With
    ``pending`` set (a list of ladders computed for the whole Davidson
    batch beforehand, in the order the sigma function asks for them) the
    ladder is popped instead of computed."""

    def __init__(self, mycc, imds):
        self.mycc, self.imds = mycc, imds
        self.pending = None

    def _add_vvvv(self, t1, t2, eris, out=None, with_ovvv=False,
                  t2sym=None):
        assert t1 is None and not with_ovvv
        if self.pending:
            return self.pending.pop(0)
        return _ladder_sym(self.mycc, self.imds.orbv_s, t2, t2sym)


def _ee_taus(kind, r1, r2, t1):
    """The ladder inputs of eeccsd_matvec_triplet / _sf, as those functions
    build them, with their t2sym, in their call order."""
    rt = _ein('ia,jb->ijab', r1, t1)
    if kind == 'triplet':
        r2aa, r2ab = r2
        aa = rt - _ein('ia,jb->jiab', r1, t1)
        aa = aa - aa.transpose(0, 1, 3, 2) + r2aa
        ab = rt - _ein('ia,jb->ijab', t1, r1) + r2ab
        return [(aa, 'jiba'), (ab, '-jiba')]
    baaa, aaba = r2
    b = rt + baaa * .5
    b = b - b.transpose(0, 1, 3, 2)
    a = rt + aaba * .5
    a = a - a.transpose(1, 0, 2, 3)
    return [(a, '-jiab'), (b, None)]


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
        imds = _make_ee(_Imds(self._cc), self._cc, keep_eris=True)
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

        cc = shim._cc

        def matvec(V):
            out = []
            for p0 in range(0, len(V), _EE_BLK):
                vs = [v.copy() for v in V[p0:p0 + _EE_BLK]]
                items = []
                for v in vs:
                    r1, r2 = shim.vector_to_amplitudes(v, shim.nmo,
                                                       shim.nocc)
                    items += _ee_taus(self._kind, r1, r2, imds.t1)
                lad = _ladder_many(self._cc, imds.orbv_s, items)
                items = None
                for k, v in enumerate(vs):
                    cc.pending = lad[2 * k:2 * k + 2]
                    out.append(f(shim, v, imds))
                cc.pending = None
            return cp.stack(out)
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

    @_on_pool
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
            x0 = np.asarray(guess).reshape(len(guess), -1)
        else:
            idx = cp.argsort(d[:no * nv] if koopmans else d)[:nwant]
            x0 = np.zeros((nwant, len(d)))
            x0[np.arange(nwant), cp.asnumpy(idx)] = 1.
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
    Hr2 += _ladder_mo(mycc, ov, r2.reshape(n * no, nv, nv)).reshape(
        n, no, nv, nv)
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
