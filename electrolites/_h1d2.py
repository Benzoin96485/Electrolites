"""Stage D2 of both spin cases' h1 term as a few batched GEMMs (W35).

``docs/DF_WORK_LIST.md`` **W35**, ``docs/README_fastdfh1d2.md``.  W30's
occupied form of the per-atom exchange term (``fastdfhessip`` item 12) is,
for an auxiliary block of ``dk`` functions and one direction,

    vhf[A][n,j,q] += a sum_{u in A, r} j3c[u,q,r] s[n,u,r,j]
                   + a sum_{u in A, r} pri[n,u,r,j] t[n,u,r,q]

for every atom ``A`` -- ``2 natm`` contractions per block and direction,
each one ``nocc x nao x (nao_A dk)``.  Their arithmetic does not depend on
the block, but their number does, and so does their shape: at the block a
40 GB card leaves, ``nao_A dk`` is a few hundred and the loop is launches.

Here ``j3c`` is held transposed, ``jT[q,u,r]``, and ``pri`` as ``priT[n,j,u,r]``,
so that ``(u, r)`` is one contiguous index of every operand and each atom's
two products are plain row-major GEMMs whose ``K`` is a run of that index.
Atoms with the same number of AOs give GEMMs of one shape, so each product
is one ``cublasDgemmBatched`` per distinct atom size, both spins in the
same batch: two to a dozen launches per block and direction instead of
``2 natm``.  Nothing here
patches anything (``docs/DF_WORK_LIST.md`` §5.12).
"""
import numpy as np
import cupy as cp


def _gemm_rm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc):
    """Row-major ``C[m,n] = alpha op(A) op(B) + beta C`` on device pointers
    (``fastdfhessip._gemm_rm``)."""
    from cupy.cuda import cublas
    op = {'N': cublas.CUBLAS_OP_N, 'T': cublas.CUBLAS_OP_T}
    al = np.array([alpha], dtype=np.float64)
    be = np.array([beta], dtype=np.float64)
    cublas.dgemm(cp.cuda.device.get_cublas_handle(), op[tb], op[ta],
                 n, m, k, al.ctypes.data, b, ldb, a, lda, be.ctypes.data,
                 c, ldc)


def _shapes(jT, s, priT, t, vhf):
    nao, _, dk = jT.shape
    ns, _, _, nocc = s.shape
    it = 8
    assert jT.flags.c_contiguous and s.flags.c_contiguous
    assert priT.flags.c_contiguous and t.flags.c_contiguous
    assert priT.shape == (ns, nocc, nao, dk) and t.shape == (ns, nao, dk, nao)
    # vhf is (natm, ns, nocc, nao) with rows of nao contiguous
    assert vhf.strides[3] == it and vhf.strides[2] == nao * it
    return nao, dk, ns, nocc, it


def d2_per_atom(jT, s, priT, t, vhf, alpha, aoslices):
    """Both products, one cuBLAS GEMM each per atom and spin (no batching):
    the probe's middle column, and the reference for the batched one."""
    nao, dk, ns, nocc, it = _shapes(jT, s, priT, t, vhf)
    for i, (p0, p1) in enumerate(zip(aoslices[:-1], aoslices[1:])):
        K = int(p1 - p0) * dk
        if K == 0:
            continue
        for n in range(ns):
            out = vhf[i, n].data.ptr
            _gemm_rm('T', 'T', nocc, nao, K, alpha,
                     s[n].data.ptr + int(p0) * dk * nocc * it, nocc,
                     jT.data.ptr + int(p0) * dk * it, nao * dk, 1., out, nao)
            _gemm_rm('N', 'N', nocc, nao, K, alpha,
                     priT[n].data.ptr + int(p0) * dk * it, nao * dk,
                     t[n].data.ptr + int(p0) * dk * nao * it, nao, 1., out,
                     nao)


class AtomGroups:
    """The atoms grouped by their AO count, and the pointer arrays of one
    call's batched GEMMs, cached per (buffers, block size, output)."""

    def __init__(self, aoslices):
        aoslices = np.asarray(aoslices)
        sizes = aoslices[1:] - aoslices[:-1]
        self.groups = [(int(n), np.nonzero(sizes == n)[0])
                       for n in np.unique(sizes) if n > 0]
        self.p0 = aoslices[:-1]
        self.ngroups = len(self.groups)
        self._cache = {}

    def _ptrs(self, jT, s, priT, t, vhf):
        nao, dk, ns, nocc, it = _shapes(jT, s, priT, t, vhf)
        key = (jT.data.ptr, s.data.ptr, priT.data.ptr, t.data.ptr,
               vhf.data.ptr, vhf.strides, dk, ns, nocc, nao)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = []
        host = []
        for n_a, atoms in self.groups:
            # both spins' products have the same shape: one batch
            p0 = np.tile(self.p0[atoms].astype(np.int64), ns)
            n = np.repeat(np.arange(ns, dtype=np.int64), len(atoms))
            c = (vhf.data.ptr + np.tile(atoms.astype(np.int64), ns)
                 * vhf.strides[0] + n * vhf.strides[1])
            a1 = s.data.ptr + (n * nao * dk + p0 * dk) * nocc * it
            b1 = jT.data.ptr + p0 * dk * it
            a2 = priT.data.ptr + (n * nocc * nao * dk + p0 * dk) * it
            b2 = t.data.ptr + (n * nao * dk + p0 * dk) * nao * it
            off = sum(len(h) for h in host)
            host.extend([a1, b1, a2, b2, c])
            out.append((n_a * dk, len(p0), off))
        flat = np.concatenate(host).astype(np.uint64)
        dev = cp.asarray(flat)
        hit = (dev, out, nao, dk, nocc)
        self._cache[key] = hit
        return hit

    def d2(self, jT, s, priT, t, vhf, alpha):
        """``vhf[A, n] += alpha (both products)`` for every atom and spin."""
        from cupy.cuda import cublas
        dev, launches, nao, dk, nocc = self._ptrs(jT, s, priT, t, vhf)
        h = cp.cuda.device.get_cublas_handle()
        al = np.array([alpha], dtype=np.float64)
        be = np.array([1.], dtype=np.float64)
        N, T = cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T
        base = dev.data.ptr
        for K, cnt, off in launches:
            a1, b1, a2, b2, c = (base + (off + q * cnt) * 8 for q in range(5))
            # row-major C = s_A^T jT_A^T  is column-major C^T = jT_A s_A
            cublas.dgemmBatched(h, T, T, nao, nocc, K, al.ctypes.data,
                                b1, nao * dk, a1, nocc, be.ctypes.data,
                                c, nao, cnt)
            # row-major C = priT_A t_A  is column-major C^T = t_A^T priT_A^T
            cublas.dgemmBatched(h, N, N, nao, nocc, K, al.ctypes.data,
                                b2, nao, a2, nao * dk, be.ctypes.data,
                                c, nao, cnt)


def contiguous_block(buf, dk, first):
    """``buf[:, :, :dk]`` as a C-contiguous ``(nao, nao, dk)`` array.

    The D loop's block buffer is ``(nao, nao, blksize)`` and a batch's last
    block is shorter, so its view has ``blksize`` as the row stride and
    ``(p, r)`` is not one index of it.  That block is laid out afresh in the
    buffer's leading ``nao^2 dk`` elements instead, zeroed before its first
    direction is written (the scatter writes the stored pairs only and the
    rest must be zero).  The buffer is then no longer zero outside the full
    layout's pairs, and the caller zeroes it before anything reads it with
    that layout again (``restore``).
    """
    nao, _, blk = buf.shape
    if dk == blk:
        return buf
    v = cp.ndarray((nao, nao, dk), dtype=buf.dtype, memptr=buf.data)
    if first:
        v[:] = 0.
    return v


def restore(buf, naux_in_batch):
    """Zero ``buf`` after a batch whose last block ``contiguous_block``
    relaid, so that the full layout is zero outside the stored pairs."""
    if naux_in_batch % buf.shape[2]:
        buf[:] = 0.
