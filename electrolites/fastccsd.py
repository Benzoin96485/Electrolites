"""Closed-shell CCSD (``gpu4pyscf.cc.ccsd_incore``) with the amplitudes, the
MO integrals and the DIIS on the device (catalogue I4, and the host half of
I5).

GPU4PySCF 1.8.1's ``CCSD`` runs PySCF's CPU driver: every iteration uploads
``t1``/``t2``, evaluates about half of ``update_amps`` with host ``numpy`` /
``lib.einsum`` on O(o²v²)–O(o³v²) arrays, pulls the device terms back with
``.get()``, and extrapolates with PySCF's (h5py-backed) DIIS.  The AO-direct
``ovvv``/``vvvv`` step keeps its two ``nao² nocc²`` intermediates on the host
and copies every block's contribution down.  At alanine dipeptide the device
is busy ~11 s of a ~165 s CCSD (nsys); the rest is host arithmetic, host
reshapes, PCIe and ``cudaMalloc`` (GPU4PySCF's allocator sends every request
above 100 MB straight to the driver).

Here ``CCSD.ccsd`` runs its own copy of PySCF's ``kernel`` loop on device
arrays:

* the MO integrals (``oooo``, ``ovoo``, ``oovv``, ``ovvo``; ``ovov`` is a
  view of ``ovvo``) are built by GPU4PySCF's ``_make_eris_incore`` algorithm
  with its two ``nao² nocc²`` accumulators on the device, and stay there (an
  ``eris`` passed in is uploaded once);
* ``update_amps`` is the same algebra as ``ccsd_incore.update_amps``, term by
  term, with every array on the device;
* ``_direct_ovvv_vvvv`` is GPU4PySCF's AO-direct loop (same ``GINTfill_int2e``
  blocks, same ``dgemm`` into ``Ht2ao``) with ``wVVoo``/``wVvoO`` on the
  device and no per-block ``.get()``;
* DIIS is ``gpu4pyscf.lib.diis.DIIS`` (device-resident) on PySCF's packed
  vector (``t1`` and the lower triangle of ``t2`` as an (ia, jb) matrix), so
  the extrapolation sees the same inner products as PySCF's;
* the AO integral blocks are computed in the first iteration and kept on
  the device when they fit in a third of the free memory left beside the
  integral kernels' local memory, else in host memory (a quarter of
  the free host memory at most), copied back block by block;
* that local memory -- the driver keeps what the largest ``GINTfill``
  launch reserved, ~37 GB for (ff|ff) on a 40 GB A100 -- is handed back
  after each integral loop (``_release_lmem``);
* everything runs on the CuPy pool rather than the >100 MB ``cudaMalloc``
  path.

``t1``/``t2`` are returned to the object as host arrays, as before.  When the
estimated device footprint does not fit, or a shell is beyond the GPU
integral kernels' angular momentum, the stock driver runs.
``FASTCCSD_OFF=1`` restores it unconditionally.  Measured in
``docs/KERNEL_TARGETS_1.8.1.md`` §13.27.
"""
import os

import cupy as cp
import numpy as np

_OFF = bool(os.environ.get('FASTCCSD_OFF'))
STATS = {'runs': 0, 'fallbacks': 0, 'iterations': 0}

_einsum = cp.einsum


class _DevERIs:
    """Device copies of the MO integrals ``update_amps`` reads."""

    def __init__(self, eris, nocc):
        self.nocc = nocc
        self.fock = cp.asarray(eris.fock)
        self.mo_energy = cp.asarray(eris.mo_energy)
        self.mo_coeff = cp.asarray(eris.mo_coeff)
        self.oooo = cp.asarray(eris.oooo)
        self.ovoo = cp.asarray(eris.ovoo)
        self.oovv = cp.asarray(eris.oovv)
        self.ovvo = cp.asarray(eris.ovvo)
        self.ovov = self.ovvo.transpose(0, 1, 3, 2)


def _tril_index(nov):
    i, j = cp.tril_indices(nov)
    return i * nov + j


def _to_vector(t1, t2, idx):
    nocc, nvir = t1.shape
    nov = nocc * nvir
    m = t2.transpose(0, 2, 1, 3).reshape(nov * nov)
    return cp.concatenate([t1.ravel(), m[idx]])


def _from_vector(vec, nocc, nvir, idx):
    nov = nocc * nvir
    t1 = vec[:nov].reshape(nocc, nvir).copy()
    m = cp.zeros(nov * nov)
    m[idx] = vec[nov:]
    m = m.reshape(nov, nov)
    m = m + m.T - cp.diag(cp.diag(m))
    t2 = cp.ascontiguousarray(
        m.reshape(nocc, nvir, nocc, nvir).transpose(0, 2, 1, 3))
    return t1, t2


def energy(t1, t2, E):
    nocc = t1.shape[0]
    fov = E.fock[:nocc, nocc:]
    e = 2 * cp.sum(fov * t1)
    tau = t2 + _einsum('ia,jb->ijab', t1, t1)
    e += 2 * _einsum('ijab,iabj', tau, E.ovvo)
    e -= _einsum('jiab,iabj', tau, E.ovvo)
    return float(e.real)


def _direct_ovvv_vvvv(mycc, t1, t2):
    """``ccsd_incore._direct_ovvv_vvvv`` with its accumulators on the
    device; returns device arrays."""
    from pyscf import lib
    from gpu4pyscf.scf import int4c2e
    from gpu4pyscf.cc import ccsd_incore as ci
    nocc, nvir = t1.shape
    nocc2 = nocc * (nocc + 1) // 2
    nao_cart = mycc.mol.nao_nr(cart=True)
    max_memory = max(ci.MEMORYMIN, mycc.max_memory - lib.current_memory()[0])
    blksize = ((max_memory * .9e6 - t2.size * 4 * 8) / 8 / nao_cart**2 / 3.5)**.5
    Ht2_mem = nocc2 * nao_cart**2 * 8 * 2
    mem_avail = int(_avail() * .75)
    blksize = max(ci.BLKMIN, int(min((nao_cart + 3) / 4, blksize,
                                     max(0, (mem_avail - Ht2_mem) * .5
                                         / 8 / nao_cart**2)**.5)))

    vhfopt = getattr(mycc, '_fastccsd_vhfopt', None)
    if vhfopt is None or vhfopt._fastccsd_blk != blksize:
        vhfopt = int4c2e._VHFOpt(mycc.mol, 'int2e')
        vhfopt.build(group_size=blksize, diag_block_with_triu=True)
        vhfopt._fastccsd_blk = blksize
        mycc._fastccsd_vhfopt = vhfopt
        mycc._fastccsd_eri = None
    mol = vhfopt.mol

    # the active orbitals: GPU4PySCF's own loop takes mycc.mo_coeff whole,
    # so any frozen orbital crashes it (orbv's width is off by nfrozen)
    mask = cp.asarray(mycc.get_frozen_mask())
    mo = vhfopt.coeff.dot(cp.asarray(mycc.mo_coeff)[:, mask])
    orbo = cp.asarray(mo[:, :nocc])
    orbv = cp.asarray(mo[:, nocc:])
    t1po = orbv.dot(t1.T)
    tau = _einsum('ia,jb->ijab', t1, t1)
    tau += t2
    tau = tau[cp.tril_indices(nocc)]
    x2 = _einsum('xab,pa->xpb', tau, orbv)
    x2 = cp.ascontiguousarray(_einsum('xpb,qb->xpq', x2, orbv))
    tau = None

    nao = mo.shape[0]
    ao_loc = mol.ao_loc
    nao2 = nao * nao
    Ht2ao = cp.zeros_like(x2)
    _dgemm = cp.cuda.cublas.dgemm
    handle = cp.cuda.device.get_cublas_handle()
    N = cp.cuda.cublas.CUBLAS_OP_N
    T = cp.cuda.cublas.CUBLAS_OP_T
    one = np.ones(1)
    one_ptr = one.ctypes.data
    x2_ptr = int(x2.data.ptr)
    Ht2ao_ptr = int(Ht2ao.data.ptr)

    def contract_vvvv_(eri, i0, i1, j0, j1):
        ic = i1 - i0
        jc = j1 - j0
        eri = eri.reshape(-1, jc * nao)
        _dgemm(handle, N, N, jc * nao, nocc2, ic * nao,
               one_ptr, eri.data.ptr, jc * nao, x2_ptr + i0 * nao * 8, nao2,
               one_ptr, Ht2ao_ptr + j0 * nao * 8, nao2)
        if i0 > j0:
            _dgemm(handle, T, N, ic * nao, nocc2, jc * nao,
                   one_ptr, eri.data.ptr, jc * nao, x2_ptr + j0 * nao * 8, nao2,
                   one_ptr, Ht2ao_ptr + i0 * nao * 8, nao2)

    l_ctr_offsets = vhfopt.l_ctr_offsets
    log_qs = vhfopt.log_qs
    cp_idx, cp_jdx = np.tril_indices(len(vhfopt.uniq_l_ctr))
    idx, idy = cp.tril_indices(nao)

    # The AO integral blocks do not change between iterations: keep them on
    # the device when all of them fit in a third of the free memory
    # (``FASTCCSD_NO_CACHE=1`` recomputes them every iteration, as stock).
    cache = getattr(mycc, '_fastccsd_eri', None)
    if (cache is None and not os.environ.get('FASTCCSD_NO_CACHE')
            and not getattr(mycc, '_fastccsd_nocache', False)):
        total = 0
        for cp_ij_id in range(len(log_qs)):
            ci_, cj_ = cp_idx[cp_ij_id], cp_jdx[cp_ij_id]
            ni = ao_loc[l_ctr_offsets[ci_ + 1]] - ao_loc[l_ctr_offsets[ci_]]
            nj = ao_loc[l_ctr_offsets[cj_ + 1]] - ao_loc[l_ctr_offsets[cj_]]
            total += int(ni) * int(nj) * nao2 * 8
        lmem = getattr(mycc, '_fastccsd_lmem', 0)
        if total < (_avail() - lmem) / 3:
            cache = mycc._fastccsd_eri = {}
            STATS['eri_cache_GB'] = total / 1e9
        elif total < _host_avail() / 4:
            # e.g. f shells on a 40 GB card, where the (ff|ff) launch
            # alone reserves most of the device: host memory, one
            # H2D copy per block per iteration instead of GINTfill
            cache = mycc._fastccsd_eri = _HostCache()
            STATS['eri_host_cache_GB'] = total / 1e9

    wVVoo = cp.zeros((nao, nao, nocc, nocc))
    wVvoO = cp.zeros((nao, nao, nocc, nocc))
    for cp_ij_id, log_q_ij in enumerate(log_qs):
        cpi = cp_idx[cp_ij_id]
        cpj = cp_jdx[cp_ij_id]
        if log_q_ij.size == 0:
            continue
        ish0, ish1 = l_ctr_offsets[cpi], l_ctr_offsets[cpi + 1]
        jsh0, jsh1 = l_ctr_offsets[cpj], l_ctr_offsets[cpj + 1]
        i0, i1 = int(ao_loc[ish0]), int(ao_loc[ish1])
        j0, j1 = int(ao_loc[jsh0]), int(ao_loc[jsh1])
        eri = cache.get(cp_ij_id) if cache is not None else None
        if isinstance(eri, np.ndarray):
            host, eri = eri, cp.empty(eri.shape)
            eri.set(host)
            host = None
        if eri is None:
            eri = cp.zeros([i1 - i0, nao, j1 - j0, nao])
            strides = [1, (j1 - j0) * nao, (j1 - j0) * nao**2, nao]
            _fill(mycc, eri, strides, [0, 0, i0, j0], vhfopt, cp_ij_id)
            cache = getattr(mycc, '_fastccsd_eri', None)
            eri[:, idx, :, idy] = eri[:, idy, :, idx]
            if isinstance(cache, _HostCache):
                host = np.empty(eri.shape)
                eri.get(out=host)
                cache[cp_ij_id] = host
                host = None
            elif cache is not None:
                cache[cp_ij_id] = eri
        contract_vvvv_(eri, i0, i1, j0, j1)

        pppo = _einsum('prqs,si->prqi', eri, orbo)
        wVvoO[j0:j1] += _einsum('prqi,pj->qrij', pppo, t1po[i0:i1])
        wVVoo[i0:i1, j0:j1] = _einsum('prqi,rj->pqij', pppo, t1po)
        pppo = None
        if ish0 != jsh0:
            wVVoo[j0:j1, i0:i1] = wVVoo[i0:i1, j0:j1].transpose(1, 0, 2, 3)
            tmp = _einsum('prqs,ri->piqs', eri, orbo)
            wVvoO[i0:i1] += _einsum('piqs,qj->psij', tmp, t1po[j0:j1])
            tmp = None
        eri = None
    x2 = None
    lmem = _release_lmem()
    mycc._fastccsd_lmem = max(lmem, getattr(mycc, '_fastccsd_lmem', 0))

    Ht2full = ci._unpack_t2_tril(Ht2ao, nocc, nao)
    t1tmp = _einsum('ijpq,qj->ip', Ht2full, orbo) * 2
    t1tmp -= _einsum('ijqp,qj->ip', Ht2full, orbo)
    t1new = t1tmp.dot(orbv)
    Ht2full = t1tmp = None

    Ht2tril = _einsum('xpq,pa->xaq', Ht2ao, orbv)
    Ht2tril = _einsum('xaq,qb->xab', Ht2tril, orbv)
    t1pv = orbo.dot(t1)
    tmp = _einsum('xpq,pa->xaq', Ht2ao, orbv)
    Ht2tril -= _einsum('xaq,qb->xab', tmp, t1pv)
    tmp = _einsum('xpq,pa->xaq', Ht2ao, t1pv)
    Ht2tril -= _einsum('xaq,qb->xab', tmp, orbv)
    tmp = Ht2ao = None
    t2new = ci._unpack_t2_tril(Ht2tril, nocc, nvir)
    Ht2tril = None

    c = vhfopt.coeff
    dVV = wVVoo.reshape(nao, nao, nocc * nocc)[:, :, ::nocc + 1].sum(axis=2)
    dVv = wVvoO.reshape(nao, nao, nocc * nocc)[:, :, ::nocc + 1].sum(axis=2)
    wpq = 2 * c.T.dot(dVV).dot(c)
    wpq -= c.T.dot(dVv).dot(c).T

    tmp = _einsum('pqji,qb->pbji', wVvoO, orbv)
    wVOov = _einsum('pbji,pa->bjia', tmp, orbv)
    tmp = _einsum('pqji,pa->aqji', wVVoo, -orbv)
    wVooV = _einsum('aqji,qb->bjia', tmp, orbv)
    return wpq, t1new, t2new, wVOov, wVooV


def _fill(mycc, eri, strides, offsets, vhfopt, group_id):
    """``_fill_eri_block``, retried after handing the pool's free blocks
    back and then after dropping the integral cache: the high-l
    ``GINTfill_int2e`` launches need driver local memory outside the pool,
    which fails as a CUDA error rather than as a pool allocation."""
    from gpu4pyscf.cc import ccsd_incore as ci
    pool = cp.get_default_memory_pool()
    lmem = getattr(mycc, '_fastccsd_lmem', 0)
    if lmem and cp.cuda.runtime.memGetInfo()[0] < lmem * 1.05:
        pool.free_all_blocks()
    for attempt in range(3):
        try:
            ci._fill_eri_block(eri, strides, offsets, vhfopt, group_id)
            # a high-l launch can leave the rest of the loop a few GB
            rt = cp.cuda.runtime
            free, total = rt.memGetInfo()
            if free < total // 4:
                lmem = _release_lmem()
                mycc._fastccsd_lmem = max(lmem, getattr(mycc, '_fastccsd_lmem', 0))
                STATS['lmem_releases'] = STATS.get('lmem_releases', 0) + 1
            return eri
        except RuntimeError:
            if attempt == 2:
                raise
            pool.free_all_blocks()
            if attempt == 1 and getattr(mycc, '_fastccsd_eri', None) is not None:
                mycc._fastccsd_eri.clear()
                mycc._fastccsd_eri = None
                mycc._fastccsd_nocache = True
                STATS['cache_dropped'] = STATS.get('cache_dropped', 0) + 1
                pool.free_all_blocks()


class _HostCache(dict):
    """Integral blocks kept in host memory.  Pageable on purpose: pinning
    8.7 GB (glycine/def2-TZVP) costs 5.8 s of ``cudaHostAlloc``, more than
    the faster copies (27 against 14 GB/s) save over a CCSD's iterations."""


def _host_avail():
    try:
        return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (ValueError, OSError):
        return 0


def _release_lmem():
    """Give back the local memory the driver reserved for the largest
    ``GINTfill_int2e`` launch so far.  The driver keeps it after the launch
    (CUDA 12: always), and for (ff|ff) it is ~37 GB of a 40 GB A100, which
    would leave the rest of the iteration ~2 GB.  Re-setting the stack-size
    limit, even to its current value, makes the driver drop it; the next
    launch that needs it takes it again.  Returns the bytes released."""
    rt = cp.cuda.runtime
    cp.cuda.Device().synchronize()
    free0 = rt.memGetInfo()[0]
    # the driver raises the reported limit to what the launch needed, so
    # the value to restore is the one from before any such launch
    rt.deviceSetLimit(rt.cudaLimitStackSize, _stack0())
    return max(0, rt.memGetInfo()[0] - free0)


_STACK0 = {}


def _stack0():
    dev = cp.cuda.Device().id
    if dev not in _STACK0:
        rt = cp.cuda.runtime
        _STACK0[dev] = min(rt.deviceGetLimit(rt.cudaLimitStackSize), 8192)
    return _STACK0[dev]


def _make_eris(mycc):
    """``ccsd_incore._make_eris_incore`` with ``ppOO``/``pPoO`` and the MO
    integrals on the device (no per-block ``.get()``)."""
    from pyscf.cc import ccsd as pyscf_ccsd
    from gpu4pyscf.scf import int4c2e
    from gpu4pyscf.cc import ccsd_incore as ci
    eris = pyscf_ccsd._ChemistsERIs()
    eris._common_init_(mycc, mycc.mo_coeff)
    mo_coeff = cp.asarray(eris.mo_coeff, order='F')
    nocc = eris.nocc
    nmo = mo_coeff.shape[1]
    nvir = nmo - nocc
    nao_cart = mycc.mol.nao_nr(cart=True)
    mem_avail = int(_avail() * .3)
    blksize = max(ci.BLKMIN, int(min((nao_cart + 3) / 4,
                                     (mem_avail * .5 / 8 / nao_cart**2)**.5)))
    vhfopt = int4c2e._VHFOpt(mycc.mol, 'int2e')
    vhfopt.build(group_size=blksize, diag_block_with_triu=True)
    mol = vhfopt.mol
    mo = vhfopt.coeff.dot(mo_coeff)
    orbo = cp.asarray(mo[:, :nocc])
    orbv = cp.asarray(mo[:, nocc:])
    ao_loc = mol.ao_loc
    nao = mo.shape[0]
    l_ctr_offsets = vhfopt.l_ctr_offsets
    cp_idx, cp_jdx = np.tril_indices(len(vhfopt.uniq_l_ctr))

    ppOO = cp.zeros((nao, nao, nocc, nocc))
    pPoO = cp.zeros((nao, nao, nocc, nocc))
    idx, idy = cp.tril_indices(nao)
    for cp_ij_id, log_q_ij in enumerate(vhfopt.log_qs):
        if log_q_ij.size == 0:
            continue
        cpi, cpj = cp_idx[cp_ij_id], cp_jdx[cp_ij_id]
        ish0, ish1 = l_ctr_offsets[cpi], l_ctr_offsets[cpi + 1]
        jsh0, jsh1 = l_ctr_offsets[cpj], l_ctr_offsets[cpj + 1]
        i0, i1 = int(ao_loc[ish0]), int(ao_loc[ish1])
        j0, j1 = int(ao_loc[jsh0]), int(ao_loc[jsh1])
        eri = cp.zeros((nao, i1 - i0, j1 - j0, nao))
        strides = [1, (i1 - i0) * (j1 - j0) * nao, (j1 - j0) * nao, nao]
        _fill(mycc, eri, strides, [0, 0, i0, j0], vhfopt, cp_ij_id)
        eri[idx, :, :, idy] = eri[idy, :, :, idx]

        pijo = cp.dot(eri.reshape(-1, nao), orbo)
        eri = None
        ijoo = cp.dot(pijo.reshape(nao, -1).T, orbo)
        ppOO[i0:i1, j0:j1] = ijoo.reshape(i1 - i0, j1 - j0, nocc, nocc)
        ijoo = None
        jopi = cp.ascontiguousarray(
            pijo.reshape(nao * (i1 - i0), (j1 - j0) * nocc).T)
        pijo = None
        jopo = cp.dot(jopi.reshape(-1, i1 - i0), orbo[i0:i1])
        pPoO[j0:j1] += jopo.reshape(j1 - j0, nocc, nao, nocc).transpose(0, 2, 1, 3)
        jopo = None
        if ish0 != jsh0:
            ppOO[j0:j1, i0:i1] = ppOO[i0:i1, j0:j1].transpose(1, 0, 2, 3)
            opio = cp.dot(jopi.reshape(j1 - j0, -1).T, orbo[j0:j1])
            pPoO[i0:i1] += opio.reshape(nocc, nao, i1 - i0, nocc).transpose(2, 1, 0, 3)
            opio = None
        jopi = None
    mycc._fastccsd_lmem = max(_release_lmem(),
                              getattr(mycc, '_fastccsd_lmem', 0))

    pooo = cp.dot(ppOO.reshape(nao, -1).T, orbo)
    eris.oooo = cp.dot(pooo.reshape(nao, -1).T, orbo).reshape(nocc, nocc, nocc, nocc)
    ooov = cp.dot(pooo.reshape(nao, -1).T, orbv).reshape(nocc, nocc, nocc, nvir)
    eris.ovoo = cp.ascontiguousarray(
        ooov.reshape(nocc * nocc, nocc * nvir).T).reshape(nocc, nvir, nocc, nocc)
    pooo = ooov = None
    poov = cp.dot(ppOO.reshape(nao, -1).T, orbv)
    ppOO = None
    eris.oovv = cp.dot(poov.reshape(nao, -1).T, orbv).reshape(nocc, nocc, nvir, nvir)
    poov = cp.dot(pPoO.reshape(nao, -1).T, orbv)
    pPoO = None
    voov = cp.dot(orbv.T, poov.reshape(nao, -1))
    poov = None
    eris.ovvo = cp.ascontiguousarray(
        voov.reshape(nvir * nocc, nocc * nvir).T).reshape(nocc, nvir, nvir, nocc)
    eris.ovov = eris.ovvo.transpose(0, 1, 3, 2)
    return eris


def update_amps(mycc, t1, t2, E):
    """``ccsd_incore.update_amps`` on device arrays."""
    nocc, nvir = t1.shape
    fock = E.fock
    mo_e_o = E.mo_energy[:nocc]
    mo_e_v = E.mo_energy[nocc:] + mycc.level_shift
    orbo = E.mo_coeff[:, :nocc]
    orbv = E.mo_coeff[:, nocc:]

    wpq, t1new, t2new, wVOov, wVooV = _direct_ovvv_vvvv(mycc, t1, t2)
    t2new *= .5

    fov = fock[:nocc, nocc:].copy()
    t1new += fock[:nocc, nocc:]

    foo = fock[:nocc, :nocc] - cp.diag(mo_e_o)
    foo += .5 * fock[:nocc, nocc:].dot(t1.T)

    wpqT = wpq.T
    fvv = orbv.T.dot(wpqT).dot(orbv)
    t1new -= t1.dot(fvv.T)

    fvv += fock[nocc:, nocc:] - cp.diag(mo_e_v)
    fvv -= .5 * t1.T.dot(fock[:nocc, nocc:])

    foo += orbo.T.dot(wpqT).dot(orbo)
    fov += orbo.T.dot(wpqT).dot(orbv)

    tau = _einsum('ia,jb->ijab', t1, t1)
    tau += t2
    woooo = _einsum('ijab,kabl->ijkl', tau, E.ovvo)
    woooo += E.oooo.transpose(0, 2, 1, 3)
    tmp = _einsum('la,jaik->lkji', t1, E.ovoo)
    woooo += tmp
    woooo += tmp.transpose(1, 0, 3, 2)
    t2new += .5 * _einsum('ijkl,klab->ijab', woooo, tau)
    woooo = tau = tmp = None

    wVOov -= _einsum('jbik,ka->bjia', E.ovoo, t1)
    t2new += wVOov.transpose(1, 2, 0, 3)

    wVooV += _einsum('kbij,ka->bija', E.ovoo, t1)
    wVooV -= E.oovv.transpose(2, 0, 1, 3)
    wVOov += wVooV * .5
    wVOov += E.ovvo.transpose(2, 3, 0, 1)

    t2new += (E.ovvo * 0.5).transpose(0, 3, 1, 2)
    t1new += orbo.T.dot(wpq).dot(orbv)

    tmp = _einsum('ic,kjbc->ikjb', t1, E.oovv)
    tmp += _einsum('jbck,ic->jkib', E.ovvo, t1)
    t2new -= _einsum('ka,jkib->jiba', t1, tmp)
    tmp = None

    tau = t2 * .5
    tau += _einsum('ia,jb->ijab', t1, t1)
    wVooV += _einsum('kbci,jkca->bija', E.ovvo, tau)
    tau = None

    tmp = _einsum('jkca,ckib->jaib', t2, wVooV)
    t2new += tmp.transpose(2, 0, 1, 3)
    tmp *= .5
    t2new += tmp.transpose(0, 2, 1, 3)
    tmp = None

    tau = _einsum('ia,jb->iajb', t1 * .5, t1)
    tau += t2.transpose(0, 2, 1, 3)
    eris_ovOV = E.ovvo.transpose(0, 1, 3, 2) * 2
    eris_ovOV -= E.ovvo.transpose(3, 1, 0, 2)
    fvv -= _einsum('jcia,jcib->ab', tau, eris_ovOV)
    foo += _einsum('iakb,jakb->ij', eris_ovOV, tau)

    theta = t2.transpose(0, 2, 1, 3) * 2
    theta -= t2.transpose(1, 2, 0, 3)
    tau = theta * .25
    tau -= _einsum('ia,jb->jaib', t1 * .5, t1)
    wVOov += _einsum('kcia,kcjb->aijb', eris_ovOV, tau)
    eris_ovOV = tau = None

    t2new += _einsum('kcia,ckjb->ijab', theta, wVOov)
    theta = wVOov = wVooV = None

    t1new += _einsum('jb,ijab->ia', fov, t2) * 2
    t1new -= _einsum('jb,ijba->ia', fov, t2)
    ovoo = E.ovoo * 2
    ovoo -= E.ovoo.transpose(2, 1, 0, 3)
    t1new -= _einsum('jbki,jkba->ia', ovoo, t2)
    ovoo = None

    ft_ij = foo + .5 * fov.dot(t1.T)
    ft_ab = fvv - .5 * t1.T.dot(fov)
    t2new += _einsum('ijac,bc->ijab', t2, ft_ab)
    t2new -= _einsum('ki,kjab->ijab', ft_ij, t2)

    eia = mo_e_o[:, None] - mo_e_v
    t1new += t1.dot(fvv.T)
    t1new -= foo.T.dot(t1)
    t1new /= eia

    t2new = t2new + t2new.transpose(1, 0, 3, 2)
    t2new /= eia[:, None, :, None] + eia[:, None, :]
    return t1new, t2new


def _avail():
    pool = cp.get_default_memory_pool()
    return cp.cuda.runtime.memGetInfo()[0] + pool.free_bytes()


def _fits(mycc):
    """Device footprint estimate: ~14 o²v² arrays (amplitudes, integrals,
    intermediates), 2·space packed DIIS vectors, the AO-direct step's
    Ht2ao/x2 and wVVoo/wVvoO, and one AO integral block."""
    from gpu4pyscf.scf import int4c2e
    nocc = mycc.nocc
    nvir = mycc.nmo - nocc
    nao = mycc.mol.nao_nr(cart=True)
    o2v2 = nocc**2 * nvir**2
    space = mycc.diis_space if mycc.diis else 0
    need = (14 * o2v2 + 2 * space * (o2v2 // 2 + nocc * nvir)
            + 3 * nocc * (nocc + 1) // 2 * nao**2 + 4 * nao**2 * nocc**2
            + 4 * nao**3) * 8
    lmax = max(mycc.mol._bas[:, 1]) if mycc.mol.nbas else 0
    return need < 0.8 * _avail() and lmax <= int4c2e.LMAX_ON_GPU


def kernel(mycc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8,
           tolnormt=1e-6, verbose=None, callback=None):
    """PySCF's ``ccsd.kernel`` loop on device arrays."""
    from pyscf import lib
    from gpu4pyscf.lib import logger
    from gpu4pyscf.lib.diis import DIIS
    log = logger.new_logger(mycc, verbose)
    nocc = mycc.nocc
    E = _DevERIs(eris, nocc)
    nvir = E.mo_energy.size - nocc
    eia = E.mo_energy[:nocc, None] - E.mo_energy[None, nocc:]
    if t1 is None and t2 is None:
        t1 = E.fock[:nocc, nocc:] / eia
        t2 = (E.ovov.transpose(0, 2, 1, 3).conj()
              / (eia[:, None, :, None] + eia[None, :, None, :]))
        emp2 = 2 * _einsum('ijab,iajb', t2, E.ovov)
        emp2 -= _einsum('jiab,iajb', t2, E.ovov)
        mycc.emp2 = float(emp2.real)
        e_hf = mycc.e_hf if mycc.e_hf is not None else mycc.get_e_hf()
        log.info('Init t2, MP2 energy = %.15g  E_corr(MP2) %.15g',
                 e_hf + mycc.emp2, mycc.emp2)
    else:
        if t2 is None:
            t2 = (E.ovov.transpose(0, 2, 1, 3).conj()
                  / (eia[:, None, :, None] + eia[None, :, None, :]))
        t1 = cp.asarray(t1)
        t2 = cp.asarray(t2)

    name = mycc.__class__.__name__
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    eccsd = energy(t1, t2, E)
    log.info('Init E_corr(%s) = %.15g', name, eccsd)

    idx = _tril_index(nocc * nvir)
    adiis = None
    if mycc.diis:
        adiis = DIIS(mycc, incore=True)
        adiis.space = mycc.diis_space

    converged = False
    mycc.cycles = 0
    for istep in range(max_cycle):
        t1new, t2new = update_amps(mycc, t1, t2, E)
        if callback is not None:
            callback(locals())
        normt = float(cp.sqrt(cp.sum((t1new - t1)**2) +
                              _packed_norm2(t2new - t2)))
        if mycc.iterative_damping < 1.0:
            alpha = mycc.iterative_damping
            t1new = (1 - alpha) * t1 + alpha * t1new
            t2new *= alpha
            t2new += (1 - alpha) * t2
        t1, t2 = t1new, t2new
        t1new = t2new = None
        if (adiis and istep >= mycc.diis_start_cycle and
                abs(eccsd - eold) < mycc.diis_start_energy_diff):
            vec = adiis.update(_to_vector(t1, t2, idx))
            t1, t2 = _from_vector(vec, nocc, nvir, idx)
            vec = None
            log.debug1('DIIS for step %d', istep)
        eold, eccsd = eccsd, energy(t1, t2, E)
        mycc.cycles = istep + 1
        STATS['iterations'] += 1
        log.info('cycle = %d  E_corr(%s) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep + 1, name, eccsd, eccsd - eold, normt)
        cput1 = log.timer(f'{name} iter', *cput1)
        if abs(eccsd - eold) < tol and normt < tolnormt:
            converged = True
            break
    log.timer(name, *cput0)
    return converged, eccsd, t1.get(), t2.get()


def _packed_norm2(d2):
    """Squared norm of the packed lower triangle of t2 as an (ia, jb)
    matrix, which is what PySCF's ``amplitudes_to_vector`` holds."""
    nocc, _, nvir, _ = d2.shape
    nov = nocc * nvir
    m = d2.transpose(0, 2, 1, 3).reshape(nov, nov)
    return (cp.sum(m * m) + cp.sum(cp.diag(m)**2)) * .5


def _ccsd(self, t1=None, t2=None, eris=None):
    from pyscf.cc import ccsd as pyscf_ccsd
    from gpu4pyscf.lib import logger
    if _OFF or not _fits(self):
        STATS['fallbacks'] += 1
        return pyscf_ccsd.CCSDBase.ccsd(self, t1, t2, eris)
    assert self.mo_coeff is not None
    assert self.mo_occ is not None
    if self.verbose >= logger.WARN:
        self.check_sanity()
    self.dump_flags()
    self.e_hf = self.get_e_hf()
    STATS['runs'] += 1
    _stack0()
    pool = cp.get_default_memory_pool()
    pool.set_limit(fraction=1.0)
    with cp.cuda.using_allocator(pool.malloc):
        if eris is None:
            eris = _make_eris(self)
        self.converged, self.e_corr, self.t1, self.t2 = kernel(
            self, eris, t1, t2, max_cycle=self.max_cycle, tol=self.conv_tol,
            tolnormt=self.conv_tol_normt, verbose=self.verbose,
            callback=self.callback)
    self._fastccsd_vhfopt = self._fastccsd_eri = None
    self._fastccsd_nocache = False
    pool.free_all_blocks()
    self._finalize()
    return self.e_corr, self.t1, self.t2


def apply_patch():
    if _OFF:
        return
    from gpu4pyscf.cc import ccsd_incore
    cls = ccsd_incore.CCSDBase
    if getattr(cls.ccsd, '_fastccsd', False):
        return
    cls._ccsd_orig = cls.ccsd
    _ccsd._fastccsd = True
    cls.ccsd = _ccsd


apply_patch()
