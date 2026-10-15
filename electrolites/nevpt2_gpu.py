"""SC-NEVPT2's active-space intermediates on the GPU
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` M7).

PySCF's ``mrpt.NEVPT`` spends its large-CAS time in two host C kernels:

* ``fci.rdm.make_dm123('FCI3pdm_kern_sf')`` -- dm1, dm2, dm3 as the plain
  operator products <E_pq>, <E_pq E_rs>, <E_pq E_rs E_tu>;
* ``_contract4pdm('NEVPTkern_cedf_aedf' | 'NEVPTkern_aedf_ecdf')`` -- the
  4-RDM contracted with the active ERIs, never formed:
    f3ca[0..5] = sum_jki (ki|5 j) <E_01 E_23 E_4j E_ki>,
    f3ac[0..5] = sum_ijk (ik|j 4) <E_01 E_23 E_j5 E_ik>
  (identified numerically against PySCF, ``tests/test_nevpt2_gpu.py``).

Here every one of them is <T_10| E_23 |V_45> = <E_32 T_10 | V_45>:
T_pq = E_pq|c> on the device, V = T (dm3), V_4a = sum_j E_4j W_aj (f3ca)
or V_a5 = sum_j E_j5 W_aj (f3ac), W_aj = sum_ki (ki|aj) T_ki (one GEMM).
E_rs is applied to a block of vectors by its alpha and beta excitation
lists (two raw kernels, no atomics: each map is injective), once per (r, s)
to the T vectors, and the result is one GEMM against the stacked V.

``NEVPT(mc)`` is PySCF's class with those swapped in, and, on a GPU DF
CASSCF/CASCI (``casscf_gpu``), also the integrals (ppaa, papa, pacv, the
core Fock) from its fitted tensors, the S_r/S_i spaces as cupy einsums and
S_ijrs from the fitted (cv|Q) -- the remaining spaces are PySCF's host
code on the GPU-built integrals.
"""
import numpy as np
import cupy as cp

from pyscf import lib
from pyscf.fci import cistring
from pyscf.lib import logger


_APPLY_A = cp.ElementwiseKernel(
    'raw float64 v, raw int64 src, raw int64 dst, raw float64 sg, '
    'int64 nk, int64 na, int64 nb', 'raw float64 out',
    """
    long long b = i % nb;
    long long k = (i / nb) % nk;
    long long w = i / (nb * nk);
    out[(w * na + dst[k]) * nb + b] += sg[k] * v[(w * na + src[k]) * nb + b];
    """, 'electrolites_nevpt_apply_a')

_APPLY_B = cp.ElementwiseKernel(
    'raw float64 v, raw int64 src, raw int64 dst, raw float64 sg, '
    'int64 nk, int64 na, int64 nb', 'raw float64 out',
    """
    long long k = i % nk;
    long long a = (i / nk) % na;
    long long w = i / (nk * na);
    long long row = (w * na + a) * nb;
    out[row + dst[k]] += sg[k] * v[row + src[k]];
    """, 'electrolites_nevpt_apply_b')


class _Excitations:
    """Per (r, s), the strings I with a+_r a_s |I> = sign |J>, per spin."""

    def __init__(self, norb, nelec):
        self.norb = norb
        self.na = cistring.num_strings(norb, nelec[0])
        self.nb = cistring.num_strings(norb, nelec[1])
        self.a = self._lists(norb, nelec[0])
        self.b = self.a if nelec[0] == nelec[1] else self._lists(norb, nelec[1])

    @staticmethod
    def _lists(norb, ne):
        link = cistring.gen_linkstr_index(range(norb), ne)
        out = {}
        a, i, j, s = (link[..., k] for k in range(4))
        src = np.broadcast_to(np.arange(link.shape[0])[:, None], a.shape)
        for r in range(norb):
            for q in range(norb):
                m = (a == r) & (i == q)
                out[r, q] = (cp.asarray(src[m].astype(np.int64)),
                             cp.asarray(j[m].astype(np.int64)),
                             cp.asarray(s[m].astype(np.float64)))
        return out

    def apply(self, r, s, v, out=None):
        """E_rs on vectors v (nvec, na, nb) (added into ``out`` if given);
        each spin's string map is injective, so no two threads write one
        element."""
        v = cp.ascontiguousarray(v)
        nvec = v.shape[0]
        if out is None:
            out = cp.zeros_like(v)
        ia, ja, sa = self.a[r, s]
        if ia.size:
            _APPLY_A(v, ia, ja, sa, ia.size, self.na, self.nb, out,
                     size=nvec * ia.size * self.nb)
        ib, jb, sb = self.b[r, s]
        if ib.size:
            _APPLY_B(v, ib, jb, sb, ib.size, self.na, self.nb, out,
                     size=nvec * self.na * ib.size)
        return out


def make_dm123_f3(civec, eri, norb, nelec, with_dm3=True, with_f3=True,
                  max_bytes=None):
    """(dm1, dm2, dm3, f3ca, f3ac) as PySCF's make_dm123('FCI3pdm_kern_sf')
    and _contract4pdm with ``eri`` = (pq|rs) over the active orbitals (None
    for the parts not asked for).

    All three 6-index tensors are <T_10| E_23 |V_45> = <E_32 T_10 | V_45>
    with V in {T, Y, Z} (module docstring).  For each (r, s), E_sr is
    applied once to the T vectors (in blocks) and the results are one GEMM
    with the stacked V -- n^4 vector applications and 2 n^4 nV N_det flops
    in all."""
    if isinstance(nelec, (int, np.integer)):
        nelec = (nelec // 2, nelec - nelec // 2)
    n = norb
    ex = _Excitations(norb, nelec)
    nd = ex.na * ex.nb
    c = cp.asarray(civec, dtype=np.float64).reshape(1, ex.na, ex.nb)
    t = cp.zeros((n * n, ex.na, ex.nb))
    for p in range(n):
        for q in range(n):
            ex.apply(p, q, c, out=t[p * n + q:p * n + q + 1])
    t2 = t.reshape(n * n, nd)
    dm1 = (t2 @ c.ravel()).reshape(n, n)
    dm2 = (t2 @ t2.T).reshape(n, n, n, n).transpose(1, 0, 2, 3)  # <T_qp|T_rs>
    sets = []
    if with_dm3:
        sets.append(t2)
    if with_f3:
        # W[a, j] = sum_ki (ki|aj) T_ki ; Y[x, a] = sum_j E_xj W[a, j];
        # Z[a, x] = sum_j E_jx W[a, j]
        # W one j column at a time (never the full n^2 N_det W)
        g = cp.asarray(eri, dtype=np.float64).reshape(n * n, n, n)
        y = cp.zeros((n, n, ex.na, ex.nb))                       # (x4, a)
        z = cp.zeros((n, n, ex.na, ex.nb))                       # (a, x5)
        for j in range(n):
            wj = (cp.ascontiguousarray(g[:, :, j]).T @ t2).reshape(
                n, ex.na, ex.nb)                                 # all a
            for x in range(n):
                y[x] += ex.apply(x, j, wj)
                z[:, x] += ex.apply(j, x, wj)
            wj = None
        sets += [y.reshape(n * n, nd), z.reshape(n * n, nd)]
        y = z = None
    if not sets:
        return cp.asnumpy(dm1), cp.asnumpy(dm2), None, None, None
    # the V sets are used in place (no stacked copy: at CAS(14,14) each
    # is 18 GB)
    nv = len(sets) * n * n
    res = cp.empty((n * n, n, n, nv))                 # [(x1 x0), r, s, k]
    if max_bytes is None:
        max_bytes = int(0.25 * (cp.cuda.runtime.memGetInfo()[0]
                                + cp.get_default_memory_pool().free_bytes()))
    blk = max(1, min(n * n, max_bytes // (nd * 8 * 2)))
    for r in range(n):
        for s_ in range(n):
            for b0 in range(0, n * n, blk):
                b1 = min(n * n, b0 + blk)
                u = ex.apply(s_, r, t[b0:b1]).reshape(b1 - b0, nd)  # E_sr T
                for k, v in enumerate(sets):
                    res[b0:b1, r, s_, k * n * n:(k + 1) * n * n] = u @ v.T
                u = None
    # res[(p q), r, s, k] = <T_pq| E_rs |V_k>, bra index (x0, x1) = (q, p)
    res = res.reshape(n, n, n, n, nv).transpose(1, 0, 2, 3, 4)
    k = 0
    out = []
    for flag in (with_dm3, with_f3, with_f3):
        if flag:
            out.append(res[..., k:k + n * n].reshape((n,) * 6))
            k += n * n
        else:
            out.append(None)
    dm3, f3ca, f3ac = out
    return tuple(None if x is None else cp.asnumpy(cp.ascontiguousarray(x))
                 for x in (dm1, dm2, dm3, f3ca, f3ac))


# ----------------------------------------------- the S_r and S_i spaces
# PySCF's make_a16/a17/a19/a22/a23/a25 and Sr/Si (mrpt/nevpt2.py), the
# einsums on the device (the 4-RDM terms through f3ca/f3ac as in PySCF).


def _es(spec, *ops):
    return cp.einsum(spec, *ops, optimize=True)


def _make_a16(h1e, h2e, dms, civec, norb, nelec, link_index=None):
    dm3 = dms['3']
    f3ca = dms['f3ca']
    f3ac = dms['f3ac']

    a16 = -_es('ib,rpqiac->pqrabc', h1e, dm3)
    a16 += _es('ia,rpqbic->pqrabc', h1e, dm3)
    a16 -= _es('ci,rpqbai->pqrabc', h1e, dm3)

# qjkiac = acqjki + delta(ja)qcki + delta(ia)qjkc - delta(qc)ajki - delta(kc)qjai
    #:a16 -= _es('kbij,rpqjkiac->pqrabc', h2e, dm4)
    a16 -= f3ca.transpose(1,4,0,2,5,3) # c'a'acb'b -> a'b'c'abc
    a16 -= _es('kbia,rpqcki->pqrabc', h2e, dm3)
    a16 -= _es('kbaj,rpqjkc->pqrabc', h2e, dm3)
    a16 += _es('cbij,rpqjai->pqrabc', h2e, dm3)
    fdm2 = _es('kbij,rpajki->prab'  , h2e, dm3)
    for i in range(norb):
        a16[:,i,:,:,:,i] += fdm2

    #:a16 += _es('ijka,rpqbjcik->pqrabc', h2e, dm4)
    a16 += f3ac.transpose(1,2,0,4,3,5) # c'a'b'bac -> a'b'c'abc

    #:a16 -= _es('kcij,rpqbajki->pqrabc', h2e, dm4)
    a16 -= f3ca.transpose(1,2,0,4,3,5) # c'a'b'bac -> a'b'c'abc

    a16 += _es('jbij,rpqiac->pqrabc', h2e, dm3)
    a16 -= _es('cjka,rpqbjk->pqrabc', h2e, dm3)
    a16 += _es('jcij,rpqbai->pqrabc', h2e, dm3)
    return a16

def _make_a22(h1e, h2e, dms, civec, norb, nelec, link_index=None):
    dm3 = dms['3']
    dm2 = dms['2']
    f3ca = dms['f3ca']
    f3ac = dms['f3ac']

    a22 = -_es('pb,kipjac->ijkabc', h1e, dm3)
    a22 -= _es('pa,kibjpc->ijkabc', h1e, dm3)
    a22 += _es('cp,kibjap->ijkabc', h1e, dm3)
    a22 += _es('cqra,kibjqr->ijkabc', h2e, dm3)
    a22 -= _es('qcpq,kibjap->ijkabc', h2e, dm3)

# qjprac = acqjpr + delta(ja)qcpr + delta(ra)qjpc - delta(qc)ajpr - delta(pc)qjar
    #a22 -= _es('pqrb,kiqjprac->ijkabc', h2e, dm4)
    a22 -= f3ac.transpose(1,5,0,2,4,3) # c'a'acbb'
    fdm2 = _es('pqrb,kiqcpr->ikbc', h2e, dm3)
    for i in range(norb):
        a22[:,i,:,i,:,:] -= fdm2
    a22 -= _es('pqab,kiqjpc->ijkabc', h2e, dm3)
    a22 += _es('pcrb,kiajpr->ijkabc', h2e, dm3)
    a22 += _es('cqrb,kiqjar->ijkabc', h2e, dm3)

    #a22 -= _es('pqra,kibjqcpr->ijkabc', h2e, dm4)
    a22 -= f3ac.transpose(1,3,0,4,2,5) # c'a'bb'ac -> a'b'c'abc

    #a22 += _es('rcpq,kibjaqrp->ijkabc', h2e, dm4)
    a22 += f3ca.transpose(1,3,0,4,2,5) # c'a'bb'ac -> a'b'c'abc

    a22 += 2.0*_es('jb,kiac->ijkabc', h1e, dm2)
    a22 += 2.0*_es('pjrb,kiprac->ijkabc', h2e, dm3)
    fdm2  = _es('pa,kipc->ikac', h1e, dm2)
    fdm2 -= _es('cp,kiap->ikac', h1e, dm2)
    fdm2 -= _es('cqra,kiqr->ikac', h2e, dm2)
    fdm2 += _es('qcpq,kiap->ikac', h2e, dm2)
    fdm2 += _es('pqra,kiqcpr->ikac', h2e, dm3)
    fdm2 -= _es('rcpq,kiaqrp->ikac', h2e, dm3)
    for i in range(norb):
        a22[:,i,:,:,i,:] += fdm2 * 2

    return a22

def _make_a17(h1e,h2e,dm2,dm3):
    h1e = h1e - _es('mjjn->mn',h2e)

    a17 = -_es('pi,cabi->abcp',h1e,dm2)\
          -_es('kpij,cabjki->abcp',h2e,dm3)
    return a17

def _make_a19(h1e,h2e,dm1,dm2):
    h1e = h1e - _es('mjjn->mn',h2e)

    a19 = -_es('pi,ai->ap',h1e,dm1)\
          -_es('kpij,ajki->ap',h2e,dm2)
    return a19

def _make_a23(h1e,h2e,dm1,dm2,dm3):
    a23 = -_es('ip,caib->abcp',h1e,dm2)\
          -_es('pijk,cajbik->abcp',h2e,dm3)\
          +2.0*_es('bp,ca->abcp',h1e,dm1)\
          +2.0*_es('pibk,caik->abcp',h2e,dm2)

    return a23

def _make_a25(h1e,h2e,dm1,dm2):

    a25 = -_es('pi,ai->ap',h1e,dm1)\
          -_es('pijk,jaik->ap',h2e,dm2)\
          +2.0*_es('ap->pa',h1e)\
          +2.0*_es('piaj,ij->ap',h2e,dm1)

    return a25


def _norm_to_energy(norm, h, diff):
    from pyscf.mrpt.nevpt2 import NUMERICAL_ZERO
    norm, h = cp.asnumpy(norm), cp.asnumpy(h)
    idx = abs(norm) > NUMERICAL_ZERO
    ener_t = -(norm[idx] / (diff[idx] + h[idx] / norm[idx])).sum()
    return norm.sum(), ener_t


def _gpu_dms(dms):
    return {k: (cp.asarray(v) if isinstance(v, np.ndarray) else v)
            for k, v in dms.items()}


def Sr(mc, ci, dms, eris=None, verbose=None):
    """PySCF ``nevpt2.Sr`` (S_r^(-1)) on the device."""
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    d = _gpu_dms(dms)
    dm1, dm2, dm3 = d['1'], d['2'], d['3']
    h1e = cp.asarray(eris['h1eff'][ncore:nocc, ncore:nocc])
    h2e = cp.asarray(eris['ppaa'][ncore:nocc, ncore:nocc]).transpose(0, 2, 1, 3)
    h2e_v = cp.asarray(eris['ppaa'][nocc:, ncore:nocc]).transpose(0, 2, 1, 3)
    h1e_v = (cp.asarray(eris['h1eff'][nocc:, ncore:nocc])
             - _es('mbbn->mn', h2e_v))
    a16 = _make_a16(h1e, h2e, d, ci, ncas, mc.nelecas)
    a17 = _make_a17(h1e, h2e, dm2, dm3)
    a19 = _make_a19(h1e, h2e, dm1, dm2)
    ener = (_es('ipqr,pqrabc,iabc->i', h2e_v, a16, h2e_v)
            + _es('ipqr,pqra,ia->i', h2e_v, a17, h1e_v) * 2.0
            + _es('ip,pa,ia->i', h1e_v, a19, h1e_v))
    norm = (_es('ipqr,rpqbac,iabc->i', h2e_v, dm3, h2e_v)
            + _es('ipqr,rpqa,ia->i', h2e_v, dm2, h1e_v) * 2.0
            + _es('ip,pa,ia->i', h1e_v, dm1, h1e_v))
    return _norm_to_energy(norm, ener, mc.mo_energy[nocc:])


def Si(mc, ci, dms, eris=None, verbose=None):
    """PySCF ``nevpt2.Si`` (S_i^(+1)) on the device."""
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    d = _gpu_dms(dms)
    dm1, dm2, dm3 = d['1'], d['2'], d['3']
    h1e = cp.asarray(eris['h1eff'][ncore:nocc, ncore:nocc])
    h2e = cp.asarray(eris['ppaa'][ncore:nocc, ncore:nocc]).transpose(0, 2, 1, 3)
    h2e_v = cp.asarray(eris['ppaa'][ncore:nocc, :ncore]).transpose(0, 2, 1, 3)
    h1e_v = cp.asarray(eris['h1eff'][ncore:nocc, :ncore])
    a22 = _make_a22(h1e, h2e, d, ci, ncas, mc.nelecas)
    a23 = _make_a23(h1e, h2e, dm1, dm2, dm3)
    a25 = _make_a25(h1e, h2e, dm1, dm2)
    delta = cp.eye(ncas)
    dm3_h = (_es('abef,cd->abcdef', dm2, delta) * 2
             - dm3.transpose(0, 1, 3, 2, 4, 5))
    dm2_h = _es('ab,cd->abcd', dm1, delta) * 2 - dm2.transpose(0, 1, 3, 2)
    dm1_h = 2 * delta - dm1.transpose(1, 0)
    ener = (_es('qpir,pqrabc,baic->i', h2e_v, a22, h2e_v)
            + _es('qpir,pqra,ai->i', h2e_v, a23, h1e_v) * 2.0
            + _es('pi,pa,ai->i', h1e_v, a25, h1e_v))
    norm = (_es('qpir,rpqbac,baic->i', h2e_v, dm3_h, h2e_v)
            + _es('qpir,rpqa,ai->i', h2e_v, dm2_h, h1e_v) * 2.0
            + _es('pi,pa,ai->i', h1e_v, dm1_h, h1e_v))
    return _norm_to_energy(norm, ener, -mc.mo_energy[:ncore])


# ------------------------------------------- integrals and S_ijrs on a DF mc
def _gpu_eris(mc, mo):
    """PySCF NEVPT2's ``_ERIS`` dict from the GPU CASSCF's fitted integrals
    (``casscf_gpu.GPUDF``): ppaa, papa, pacv = (pu|cv), vhf_c, h1eff; cvcv
    is not formed (``Sijrs`` below works from the fitted (cv|Q))."""
    gdf = mc.with_df
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    C = cp.asarray(mo)
    nmo = C.shape[1]
    lpa = cp.asarray(gdf.lmo(C, C[:, ncore:nocc]))          # (p, u, Q)
    naux = lpa.shape[2]
    laa = lpa[ncore:nocc].reshape(ncas * ncas, naux)
    lp2 = lpa.reshape(nmo * ncas, naux)
    papa = (lp2 @ lp2.T).reshape(nmo, ncas, nmo, ncas)
    ppaa = cp.empty((nmo, nmo, ncas, ncas))
    blk = max(1, int(2e8 // (nmo * naux)))
    for p0 in range(0, nmo, blk):
        p1 = min(nmo, p0 + blk)
        lpp = cp.asarray(gdf.lmo(C[:, p0:p1], C))               # (p, q, Q)
        ppaa[p0:p1] = (lpp.reshape(-1, naux) @ laa.T).reshape(
            p1 - p0, nmo, ncas, ncas)
    lcv = cp.asarray(gdf.lmo(C[:, :ncore], C[:, nocc:]))      # (c, v, Q)
    pacv = (lp2 @ lcv.reshape(-1, naux).T).reshape(nmo, ncas, ncore, -1)
    dmcore = cp.asnumpy(2 * C[:, :ncore] @ C[:, :ncore].T)
    vj, vk = mc.get_jk(mc.mol, dmcore, hermi=1)
    vhf = cp.asarray(vj) - .5 * cp.asarray(vk)
    vhfcore = C.T @ vhf @ C
    h1 = C.T @ cp.asarray(mc.get_hcore()) @ C + vhfcore
    eris = {k: cp.asnumpy(v) for k, v in (('vhf_c', vhfcore), ('ppaa', ppaa),
                                         ('papa', papa), ('pacv', pacv),
                                         ('h1eff', h1))}
    eris['cvcv'] = None
    eris['_lcv'] = lcv
    return eris


def Sijrs(mc, eris, verbose=None):
    """PySCF ``nevpt2.Sijrs`` (S_ijrs^(0), an MP2-like sum over core pairs and
    virtual pairs) from the fitted (cv|Q) on the device."""
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    lcv = eris['_lcv']
    nv = lcv.shape[1]
    naux = lcv.shape[2]
    e = cp.asarray(mc.mo_energy)
    eia = e[:ncore, None] - e[None, nocc:]
    l2 = lcv.reshape(ncore * nv, naux)
    norm = 0.
    ener = 0.
    blk = max(1, int(2e9 // (ncore * nv * nv * 8 * 3)))
    for i0 in range(0, ncore, blk):
        i1 = min(ncore, i0 + blk)
        g = (lcv[i0:i1].reshape(-1, naux) @ l2.T).reshape(i1 - i0, nv, ncore, nv)
        theta = 2 * g - g.transpose(0, 3, 2, 1)
        den = eia[i0:i1, :, None, None] + eia[None, None, :, :]
        norm += float(cp.vdot(g, theta))
        ener += float(cp.vdot(g / den, theta))
    return norm, ener


class _no_df_attr:
    """``mf.with_df = None`` for the duration (mf may have no such
    attribute at all; it is removed again afterwards)."""

    def __init__(self, mf):
        self.mf = mf

    def __enter__(self):
        self.had = 'with_df' in self.mf.__dict__
        self.old = self.mf.__dict__.get('with_df')
        self.mf.with_df = None

    def __exit__(self, *exc):
        if self.had:
            self.mf.with_df = self.old
        else:
            del self.mf.with_df


def NEVPT(mc, root=0, **kwargs):
    """PySCF's ``mrpt.NEVPT(mc)``, whose ``kernel`` takes dm1-3 and the
    4-RDM/ERI contractions from ``make_dm123_f3`` (the integrals, the
    canonicalisation and the eight perturber spaces are PySCF's)."""
    from pyscf.mrpt import nevpt2
    obj = nevpt2.NEVPT(mc, root=root, **kwargs)
    return lib.set_class(obj, (_GPUNEVPT, obj.__class__))


class _GPUNEVPT:
    __name_mixin__ = 'GPU'

    def kernel(self):
        from pyscf import fci
        from pyscf.mrpt import nevpt2
        orig = fci.rdm.make_dm123
        cache = {}

        def dm123(fname, cibra, ciket, norb, nelec, *args, **kwargs):
            if fname != 'FCI3pdm_kern_sf' or cibra is not ciket or args \
                    or kwargs:
                return orig(fname, cibra, ciket, norb, nelec, *args, **kwargs)
            return make_dm123_f3(ciket, None, norb, nelec, with_f3=False)[:3]

        def c4(kern, eri, civec, norb, nelec, link_index=None):
            if 'f3' not in cache:
                cache['f3'] = make_dm123_f3(civec, eri, norb, nelec,
                                            with_dm3=False)[3:]
            return cache['f3'][0 if kern == 'NEVPTkern_cedf_aedf' else 1]

        from electrolites.casscf_gpu import GPUDF
        patches = dict(_contract4pdm=c4, Sr=Sr, Si=Si)
        if isinstance(getattr(self._mc, 'with_df', None), GPUDF):
            # the GPU CASSCF's fitted integrals; keep PySCF off its own DF
            # branch (that needs a PySCF DF object on mc._scf)
            patches.update(_ERIS=lambda nev, mo, *a, **k: _gpu_eris(nev._mc,
                                                                      mo),
                           Sijrs=Sijrs)
            scf_env = _no_df_attr(self._mc._scf)
        else:
            scf_env = lib.temporary_env(self, density_fit=self.density_fit)
        with lib.temporary_env(fci.rdm, make_dm123=dm123), \
                lib.temporary_env(nevpt2, **patches), scf_env:
            return super().kernel()
