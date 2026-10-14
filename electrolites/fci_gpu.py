"""A PySCF-compatible FCI solver whose sigma build and RDMs run on the GPU
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` M1).

GPU4PySCF 1.8.1 ships only ``fci.direct_spin1.contract_2e``: correct, ~35x
PySCF's at CAS(16,16) (KT §12.4), but unusable inside a CASCI/CASSCF -- it
takes no default ``link_index`` (every CASSCF fails at
``mcscf/mc1step.py:1121``), unpacks a 2-D vector (any Davidson above 400
determinants fails), rebuilds and uploads its address tables on every call,
and leaves the RDMs on the CPU (FEATURE_TARGETS §3.4).  At CAS(14,13) the FCI
step was 29 % of a host DF-CASSCF, the sigma build most of it (§11.3).

``FCISolver`` here is PySCF's ``direct_spin1.FCISolver`` with

* ``contract_2e`` -- GPU4PySCF's two kernels (``_build_t1``/``_gather`` over
  32-wide beta tiles, one GEMM with the packed integrals per tile), behind
  PySCF's signature: optional ``link_index``, 1-D or 2-D vectors in, the same
  shape out, address tables built once per (norb, nelec) and kept on the
  device;
* ``make_rdm1`` / ``make_rdm12`` -- one gather kernel builds
  T[pq, I] = (E_pq c)[I] for a block of beta strings from the output
  strings' own link tables (no atomics), then rdm1 = T c and
  <E_pq E_rs> = T T^T as GEMMs; PySCF's ``reorder_rdm`` convention
  (dm2[p,q,r,s] = <p+ r+ s q>) follows;
* ``trans_rdm1`` / ``trans_rdm12`` the same with a bra vector.

* ``kernel`` / ``eig`` -- the Davidson on device vectors (PySCF's
  ``lib.davidson1`` criteria: |de| < tol and |r| < sqrt(tol), restart from
  the Ritz vectors past ``max_space``, PySCF's level-shifted diagonal
  preconditioner, multi-root).  PySCF's driver (``kernel_ms1``) still makes
  the initial guess, ``hdiag`` and the ``pspace`` step, and gets a host
  vector back.  The first build moved vectors to the host on every sigma
  call and ran ``lib.davidson1`` there; at CAS(14,14) that host side was
  most of the CASSCF (``_outprod_to_subspace``, ``_normalize_xt_`` and the
  transfers, §13).

``make_hdiag``, ``pspace`` and the CASSCF's own ``solve_approx_ci`` stay
PySCF's.  A subclass that overrides ``contract_2e`` (``fix_spin_``) gets
PySCF's Davidson on that operator.  GPU4PySCF's address tables are 16-bit,
so above 65535 strings per spin the sigma build falls back to PySCF's
(logged).

    from electrolites.fci_gpu import FCISolver
    mc.fcisolver = FCISolver(mol)
"""
import numpy as np
import cupy as cp

from pyscf import lib
from pyscf.fci import direct_spin1, cistring
from pyscf.lib import logger

STATS = {'sigma': 0, 'sigma_cpu': 0, 'rdm': 0}

_RDM_SRC = r'''
extern "C" __global__
void t_build(const double* __restrict__ bra, double* __restrict__ T,
             int na, int nb, int ib0, int nbb, int norb,
             const int* __restrict__ linka, int nla,
             const int* __restrict__ linkb, int nlb)
{
    // T[(p*norb + q) * M + col] = (E_pq c)[ia, ib], col = ia*nbb + ib - ib0,
    // gathered from the output string's links (a, i, str1, sign):
    // a+ i |out> = sign |str1>  =>  <out| E_ia |str1> = sign
    const long M = (long)na * nbb;
    long col = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= M) return;
    int ia = col / nbb;
    int ib = ib0 + (int)(col % nbb);
    for (int k = 0; k < nla; k++) {
        const int* l = linka + ((long)ia * nla + k) * 4;
        T[((long)l[1] * norb + l[0]) * M + col] += l[3] * bra[(long)l[2] * nb + ib];
    }
    for (int k = 0; k < nlb; k++) {
        const int* l = linkb + ((long)ib * nlb + k) * 4;
        T[((long)l[1] * norb + l[0]) * M + col] += l[3] * bra[(long)ia * nb + l[2]];
    }
}
'''
_rdm_kernel = None


def _t_kernel():
    global _rdm_kernel
    if _rdm_kernel is None:
        _rdm_kernel = cp.RawModule(code=_RDM_SRC).get_function('t_build')
    return _rdm_kernel


class _Tables:
    """Device address tables for one (norb, neleca, nelecb)."""

    def __init__(self, norb, nelec):
        from gpu4pyscf.fci.direct_spin1 import _link_index_to_addrs
        neleca, nelecb = nelec
        self.na = cistring.num_strings(norb, neleca)
        self.nb = cistring.num_strings(norb, nelecb)
        nnorb = norb * (norb + 1) // 2
        self.sigma_ok = max(self.na, self.nb) < 65536
        if self.sigma_ok:
            la = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
            lb = la if neleca == nelecb else \
                cistring.gen_linkstr_index_trilidx(range(norb), nelecb)
            self.addra, self.signa = _link_index_to_addrs(la, nnorb)
            if neleca == nelecb:
                self.addrb, self.signb = self.addra, self.signa
            else:
                self.addrb, self.signb = _link_index_to_addrs(lb, nnorb)
        fa = cistring.gen_linkstr_index(range(norb), neleca)
        fb = fa if neleca == nelecb else \
            cistring.gen_linkstr_index(range(norb), nelecb)
        self.linka = cp.asarray(np.ascontiguousarray(fa, dtype=np.int32))
        self.linkb = cp.asarray(np.ascontiguousarray(fb, dtype=np.int32))


_TABLES = {}


def _tables(norb, nelec):
    nelec = direct_spin1._unpack_nelec(nelec)
    key = (norb, nelec)
    t = _TABLES.get(key)
    if t is None:
        t = _TABLES[key] = _Tables(norb, nelec)
    return t


def contract_2e(eri, fcivec, norb, nelec, link_index=None):
    """PySCF's ``direct_spin1.contract_2e`` on the GPU (``eri`` absorbed and
    packed (npair, npair), as PySCF's)."""
    from gpu4pyscf.fci.direct_spin1 import _build_t1, _gather, TILE
    tb = _tables(norb, nelec)
    shape = np.shape(fcivec)
    if not tb.sigma_ok:
        STATS['sigma_cpu'] += 1
        return direct_spin1.contract_2e(eri, fcivec, norb, nelec, link_index)
    return _contract_2e_dev(eri, fcivec, norb, nelec).get().reshape(shape)


def _contract_2e_dev(eri, fcivec, norb, nelec):
    """The sigma build with input and output on the device, (na, nb)."""
    from gpu4pyscf.fci.direct_spin1 import _build_t1, _gather, TILE
    tb = _tables(norb, nelec)
    STATS['sigma'] += 1
    na, nb = tb.na, tb.nb
    nnorb = norb * (norb + 1) // 2
    ci0 = cp.asarray(fcivec, dtype=np.float64).reshape(na, nb)
    eri = cp.asarray(eri, dtype=np.float64)
    if eri.shape != (nnorb, nnorb):
        from pyscf import ao2mo
        eri = cp.asarray(ao2mo.restore(4, cp.asnumpy(eri), norb))
    out = cp.zeros_like(ci0)
    threads = (TILE, TILE)
    blocks = (1, (na + TILE - 1) // TILE)
    rest = (np.int64(na), np.int64(nb), np.int64(nnorb),
            tb.addra, tb.addrb, tb.signa, tb.signb)
    t1 = cp.empty((nnorb, na * TILE))
    gt1 = cp.empty((nnorb, na * TILE))
    for strb0 in range(0, nb, TILE):
        _build_t1(blocks, threads, (ci0, t1, np.int64(strb0)) + rest)
        eri.dot(t1, out=gt1)
        _gather(blocks, threads, (out, gt1, np.int64(strb0)) + rest)
    return out


def _t_blocks(bra, norb, nelec, max_bytes=2 << 30):
    """Yield (T block, ket-shaped slice) with T[pq, col] = (E_pq bra)[I]."""
    tb = _tables(norb, nelec)
    na, nb = tb.na, tb.nb
    bra = cp.asarray(bra, dtype=np.float64).reshape(na, nb)
    n2 = norb * norb
    nbb = max(1, min(nb, int(max_bytes // (8 * n2 * na))))
    kern = _t_kernel()
    for ib0 in range(0, nb, nbb):
        nbk = min(nbb, nb - ib0)
        T = cp.zeros((n2, na * nbk))
        m = na * nbk
        kern(((m + 255) // 256,), (256,),
             (bra, T, np.int32(na), np.int32(nb), np.int32(ib0),
              np.int32(nbk), np.int32(norb),
              tb.linka, np.int32(tb.linka.shape[1]),
              tb.linkb, np.int32(tb.linkb.shape[1])))
        yield T, (ib0, ib0 + nbk)


def _rdm12(bra, ket, norb, nelec, with_dm2=True):
    """Spin-traced (transition) RDMs, PySCF's convention:
    dm1[p,q] = <q+ p>, dm2[p,q,r,s] = <p+ r+ s q>."""
    STATS['rdm'] += 1
    tb = _tables(norb, nelec)
    na, nb = tb.na, tb.nb
    n2 = norb * norb
    ketd = cp.asarray(ket, dtype=np.float64).reshape(na, nb)
    same = bra is ket
    g1 = cp.zeros(n2)
    g2 = cp.zeros((n2, n2)) if with_dm2 else None
    tbra_iter = _t_blocks(bra, norb, nelec)
    tket_iter = None if same else _t_blocks(ket, norb, nelec)
    for Tb, (b0, b1) in tbra_iter:
        Tk = Tb if same else next(tket_iter)[0]
        # g1[x] = sum_I (E_x bra)[I] ket[I] = <bra|E_x^+|ket>
        g1 += Tb @ ketd[:, b0:b1].reshape(-1)
        if with_dm2:
            # g2[x, y] = <bra|E_x^+ E_y|ket>
            g2 += Tb @ Tk.T
    # g1[p*n+q] = <bra|E_qp|ket> = <bra|q+ p|ket> = dm1[p,q] (PySCF)
    dm1 = g1.reshape(norb, norb)
    if not with_dm2:
        return cp.asnumpy(dm1), None
    # g2[q*n+p, r*n+s] = <bra|E_pq E_rs|ket>
    dm2 = g2.reshape(norb, norb, norb, norb).transpose(1, 0, 2, 3).copy()
    # <p+ r+ s q> = <E_pq E_rs> - delta_qr <p+ s>,  <p+ s> = dm1[s,p]
    for q in range(norb):
        dm2[:, q, q, :] -= dm1.T
    return cp.asnumpy(dm1), cp.asnumpy(dm2)


def _davidson(hop, x0, hdiag, level_shift, nroots, tol, tol_residual,
              lindep, max_cycle, max_space, log):
    """PySCF's ``lib.davidson1`` on device vectors: the lowest ``nroots``
    of a symmetric operator, converged when |de| < tol and |r| < tol_residual
    (sqrt(tol) by default, as PySCF), restarting from the Ritz vectors when
    the space exceeds ``max_space + 4 (nroots - 1)`` (PySCF's size); PySCF's
    diagonal preconditioner, level-shifted at the lowest root for all roots."""
    if tol_residual is None:
        tol_residual = tol ** .5
    max_space = max_space + (nroots - 1) * 4          # as lib.davidson1

    x0 = [cp.asarray(x, dtype=np.float64).ravel() for x in x0]
    ndet = x0[0].size
    cap = max_space + nroots
    # Basis and its images in preallocated rows: no per-cycle stacking.
    V = cp.empty((cap + nroots, ndet))
    AV = cp.empty_like(V)
    m = 0

    def add(vs):
        nonlocal m
        k0 = m
        for v in vs:
            for _ in range(2):
                if m:
                    v = v - V[:m].T @ (V[:m] @ v)
            nv = float(cp.linalg.norm(v))
            if nv ** 2 > lindep:
                V[m] = v / nv
                m += 1
        for k in range(k0, m):
            AV[k] = hop(V[k])
        return m - k0

    add(x0)
    e_last = None
    conv = [False] * nroots
    for cycle in range(max_cycle):
        h = cp.asnumpy(V[:m] @ AV[:m].T)
        w, u = np.linalg.eigh((h + h.T) * .5)
        nr = min(nroots, len(w))
        e = w[:nr]
        U = cp.asarray(u[:, :nr])
        X = U.T @ V[:m]
        AX = U.T @ AV[:m]
        R = AX - cp.asarray(e)[:, None] * X
        rnorm = cp.linalg.norm(R, axis=1).get()
        de = np.abs(e - e_last) if (e_last is not None and
                                     len(e_last) == nr) else np.abs(e)
        conv = list((de < tol) & (rnorm < tol_residual))
        log.debug('GPU davidson %d  dim %d  e %s  |de| %s  |r| %s',
                  cycle, m, e, de, rnorm)
        if all(conv):
            break
        e_last = e
        if m + nr > max_space:
            V[:nr] = X
            AV[:nr] = AX
            m = nr
        del AX
        new = []
        for k in range(nr):
            if conv[k]:
                continue
            hd = hdiag - (e[0] - level_shift)          # e[0], as PySCF
            hd[abs(hd) < 1e-8] = 1e-8
            new.append(R[k] / hd)
        del R
        if not add(new):
            log.warn('GPU davidson: no new direction, |r| %s', rnorm)
            break
    return conv, e, [cp.asnumpy(x) for x in X]


class FCISolver(direct_spin1.FCISolver):
    """``direct_spin1.FCISolver`` with the sigma build, the Davidson
    vectors and the RDMs on the GPU."""

    def kernel(self, h1e, eri, norb, nelec, ci0=None, **kwargs):
        nelec_u = direct_spin1._unpack_nelec(nelec, self.spin)
        tb = _tables(norb, nelec_u)
        if (not tb.sigma_ok or 'hop' in kwargs or
                type(self).contract_2e is not FCISolver.contract_2e):
            return super().kernel(h1e, eri, norb, nelec, ci0, **kwargs)
        h2e = cp.asarray(self.absorb_h1e(h1e, eri, norb, nelec_u, .5))
        self._hdiag_dev = cp.asarray(
            self.make_hdiag(h1e, eri, norb, nelec_u, compress=False).ravel())

        def hop(c):
            return _contract_2e_dev(h2e, c, norb, nelec_u).ravel()
        hop._device = True
        try:
            return super().kernel(h1e, eri, norb, nelec, ci0, hop=hop,
                                  **kwargs)
        finally:
            self._hdiag_dev = None

    def eig(self, op, x0=None, precond=None, **kwargs):
        if not getattr(op, '_device', False):
            return super().eig(op, x0, precond, **kwargs)
        log = kwargs.get('verbose')
        if not isinstance(log, logger.Logger):
            log = logger.new_logger(self, log)
        nroots = kwargs['nroots']
        if callable(x0):
            x0 = x0()
        conv, e, x = _davidson(
            op, x0, self._hdiag_dev, self.level_shift, nroots,
            kwargs['tol'], kwargs.get('tol_residual'), kwargs['lindep'],
            kwargs['max_cycle'], kwargs['max_space'], log)
        if nroots == 1:
            self.converged = conv[0]
            return e[0], x[0]
        self.converged = conv
        return e, x

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_2e(eri, fcivec, norb, nelec, link_index)

    def make_rdm1(self, fcivec, norb, nelec, link_index=None):
        return _rdm12(fcivec, fcivec, norb, nelec, with_dm2=False)[0]

    def make_rdm12(self, fcivec, norb, nelec, link_index=None, reorder=True):
        dm1, dm2 = _rdm12(fcivec, fcivec, norb, nelec)
        if not reorder:
            # PySCF's un-reordered dm2 is <E_pq E_rs>
            dm2 = dm2.copy()
            for q in range(norb):
                dm2[:, q, q, :] += dm1.T
        return dm1, dm2

    def trans_rdm1(self, cibra, ciket, norb, nelec, link_index=None):
        return _rdm12(cibra, ciket, norb, nelec, with_dm2=False)[0]

    def trans_rdm12(self, cibra, ciket, norb, nelec, link_index=None,
                    reorder=True):
        dm1, dm2 = _rdm12(cibra, ciket, norb, nelec)
        if not reorder:
            dm2 = dm2.copy()
            for q in range(norb):
                dm2[:, q, q, :] += dm1.T
        return dm1, dm2


FCI = FCISolver
