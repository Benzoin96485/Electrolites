"""The cderi's sparsity pattern, tiled.

`fastdf`'s first contraction runs on this whenever `fastdf._use_tiled` says
the pattern is sparse enough for it; `benchmarks/df_block_probe.py` and
`benchmarks/df_block_bench.py` are what measured where that is, and
``docs/README_fastdfblock.md`` is the write-up.

`DF_WORK_LIST.md` W5 proposed replacing the exchange build's dense first
contraction with a block-sparse one.  Everything that decides whether that can
work is in this file: what the stored pair list looks like when it is quantised
to AO tiles, which AO orderings make it look better, and the row-compressed
buffer the tiled contraction runs on.  The headline is that a tile reaches only
half the sparsity the pair list has -- **69.4 %** of ``nao x nao`` is covered
by tiles at 284 atoms where 22.2 % of the pairs are stored -- because the pair
fill is the product of an atom-pair factor a tile can address and a shell
-structure factor at one-to-five-AO granularity that it cannot.

Two facts set the shape of all of it.

* The cderi's pair addresses are in the **original** Mole's AO order
  (``int3c2e_bdiv.py:506`` passes ``original_ao_order=True``), which is
  atom-by-atom in the order the geometry file lists the atoms.  So the tile
  locality of the pattern is inherited from the xyz file.
* Both AO indices of the exchange contraction may be permuted freely.
  ``rhok[i,k,L] = sum_j A[i,j,L] f[j,k]`` and ``vk = rhok rhok^T`` are
  covariant under a permutation ``P`` applied to the first index of ``A``, the
  second index of ``A`` and the first index of ``f``: ``vk`` comes out as
  ``P vk P^T`` and one scatter puts it back.  A reordering therefore costs a
  gather of the density factor and a scatter of the result, both
  ``O(nao*nocc)`` and ``O(nao^2)`` against the contraction's
  ``O(nao^2 * nocc * naux)``, and is free in practice.
"""
import os

import numpy as np


#: ``kernel`` (default) is W14's one-pass CUDA scatter; ``cupy`` is the
#: transpose-into-a-staging-buffer plus two fancy-index scatters that
#: `docs/README_fastdfblock.md` §4.1 shipped, kept as the ablation column
#: `docs/README_fastdfunpack.md` measures against.
UNPACK = os.environ.get('FASTDF_TILE_UNPACK', 'kernel')


# --------------------------------------------------------------------------
# the pattern
# --------------------------------------------------------------------------
def pair_indices(mol, auxbasis=None):
    """The stored ``(i, j)`` of the cderi, in the original Mole's AO order.

    No integrals and no GPU arithmetic: the screening is a property of the
    geometry and the two basis sets, which is what lets `df_block_probe.py`
    run anywhere.
    """
    from pyscf.df import addons
    from gpu4pyscf.df import int3c2e_bdiv
    from gpu4pyscf.gto.mole import SortedMole
    aux = addons.make_auxmol(mol, auxbasis)
    opt = int3c2e_bdiv.Int3c2eOpt(mol, aux)
    opt.mol = SortedMole.from_mol(mol, decontract=True)
    opt.build()
    addr = opt.pair_and_diag_indices()[0].get()
    nao = mol.nao
    return addr // nao, addr % nao, int(nao), int(aux.nao)


def ao2atom(mol):
    out = np.empty(mol.nao, dtype=np.int32)
    for at, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        out[p0:p1] = at
    return out


def atom_order(mol, rows, cols, order):
    """A permutation of the atoms.  AOs follow their atom and keep their
    within-atom order, so no permutation here can split a shell."""
    natm = mol.natm
    if order == 'native':
        return np.arange(natm)
    if order == 'morton':
        xyz = mol.atom_coords()
        q = np.floor((xyz - xyz.min(0)) / max(np.ptp(xyz, axis=0).max(), 1e-9)
                     * (2**10 - 1)).astype(np.int64)
        key = np.zeros(natm, dtype=np.int64)
        for b in range(10):               # interleave the three coordinates
            for d in range(3):
                key |= ((q[:, d] >> b) & 1) << (3 * b + d)
        return np.argsort(key, kind='stable')
    if order == 'rcb':
        # recursive coordinate bisection: split at the median of the widest
        # axis and recurse.  Unlike a Morton curve every contiguous chunk at
        # every level is a compact spatial cluster, which is what a tile has
        # to be.
        xyz = mol.atom_coords()

        def split(idx):
            if len(idx) <= 2:
                return [idx]
            d = np.ptp(xyz[idx], axis=0).argmax()
            o = idx[np.argsort(xyz[idx, d], kind='stable')]
            h = len(o) // 2
            return split(o[:h]) + split(o[h:])
        return np.concatenate(split(np.arange(natm)))
    if order == 'rcm':
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        a2a = ao2atom(mol)
        ai, aj = a2a[rows], a2a[cols]
        adj = coo_matrix((np.ones(len(ai), dtype=np.int8), (ai, aj)),
                         shape=(natm, natm)).tocsr()
        adj = ((adj + adj.T) > 0).astype(np.int8).tocsr()
        return np.asarray(reverse_cuthill_mckee(adj, symmetric_mode=True))
    raise ValueError(f'unknown AO order {order!r}')


def ao_perm(mol, atoms):
    """old AO index -> new AO index, for an atom permutation."""
    sl = mol.aoslice_by_atom()
    new = np.concatenate([np.arange(sl[at, 2], sl[at, 3]) for at in atoms])
    inv = np.empty(mol.nao, dtype=np.int64)
    inv[new] = np.arange(mol.nao)
    return inv


def tile_mask(rows, cols, nao, T):
    """The symmetric tile-pair pattern, and the AO count of each tile."""
    nt = (nao + T - 1) // T
    key = np.unique((rows // T) * nt + (cols // T))
    I, J = key // nt, key % nt
    mask = np.zeros((nt, nt), dtype=bool)
    mask[I, J] = mask[J, I] = True
    size = np.full(nt, T, dtype=np.int64)
    size[-1] = nao - T * (nt - 1)
    return mask, size


def tile_stats(rows, cols, nao, T):
    """What fraction of ``nao x nao`` the tiled pattern covers, and how wide
    each tile row is.  ``fill`` is the ceiling on what a tiled contraction can
    remove; ``pad_fill`` is what one *padded* batched call would reach, which
    is the same number with ``width_max`` in place of ``width_mean``."""
    mask, size = tile_mask(rows, cols, nao, T)
    covered = float((size[:, None] * size[None, :])[mask].sum())
    width = (mask * size[None, :]).sum(1)
    return dict(tile=T, ntile=int(len(size)), fill=covered / nao**2,
                width_mean=float(width.mean()), width_max=int(width.max()),
                pad_fill=float(width.max()) / nao, width=width)


# --------------------------------------------------------------------------
# the unpack, as one kernel  (`docs/DF_WORK_LIST.md` W14)
# --------------------------------------------------------------------------
#: The scatter GPU4PySCF's `decompress_and_transpose` is, with a destination
#: table in place of its `divmod`.  The packed block arrives as ``(nL, npair)``
#: and every stored pair owns two whole rows of the row-compressed buffer, so
#: the write is `nL` contiguous doubles per pair per destination and the read
#: is one column of the block.  Which of the two is the strided one depends on
#: the layout the block arrives in, and there are two of those:
#:
#: * **pair major** -- ``in[L*ldin + p]``, which is every block of a
#:   host-staged tensor and of a device-resident ``float64`` one.  The read is
#:   coalesced along `p` and the write along `L`, so the two disagree and the
#:   transpose goes through a 32x32 shared tile.  The ``+1`` pad is the usual
#:   one: unpadded, every thread of a store phase reads the same two shared
#:   banks and the read serialises sixteen ways.
#: * **auxiliary major** -- ``in[L + p*ldp]``, which is what W7's widener hands
#:   back for a device-resident ``float32`` tensor.  Here the read is already
#:   coalesced along `L`, which is the direction the write wants, so there is
#:   no transpose and no shared memory at all.
#:
#: Both write `d1[p]` and `d2[p]`; for a diagonal pair those are the same row
#: and the second store is a harmless repeat of the first, which is what the
#: two fancy-index scatters this replaces also did.
_UNPACK_SRC = r"""
extern "C" __global__
void unpack_pair_major(double *buf, const double *in,
                       const long long *d1, const long long *d2,
                       int npair, int nL, long long ldin, long long ldL)
{
    __shared__ double tile[32][33];
    const int p0 = blockIdx.x * 32;
    const int l0 = blockIdx.y * 32;
    int p = p0 + threadIdx.x;
    for (int k = 0; k < 32; k += 8) {
        int L = l0 + threadIdx.y + k;
        if (p < npair && L < nL)
            tile[threadIdx.y + k][threadIdx.x] = in[(long long)L * ldin + p];
    }
    __syncthreads();
    const int L = l0 + threadIdx.x;
    if (L >= nL) return;
    for (int k = 0; k < 32; k += 8) {
        int q = p0 + threadIdx.y + k;
        if (q < npair) {
            double v = tile[threadIdx.x][threadIdx.y + k];
            buf[d1[q] * ldL + L] = v;
            buf[d2[q] * ldL + L] = v;
        }
    }
}

extern "C" __global__
void unpack_aux_major(double *buf, const double *in,
                      const long long *d1, const long long *d2,
                      int npair, int nL, long long ldp, long long ldL)
{
    const int L = blockIdx.y * 32 + threadIdx.x;
    if (L >= nL) return;
    for (int k = 0; k < 32; k += 8) {
        int q = blockIdx.x * 32 + threadIdx.y + k;
        if (q < npair) {
            double v = in[(long long)q * ldp + L];
            buf[d1[q] * ldL + L] = v;
            buf[d2[q] * ldL + L] = v;
        }
    }
}
"""
_unpack_mod = None


def _kernel(name):
    global _unpack_mod
    if _unpack_mod is None:
        import cupy as cp
        _unpack_mod = cp.RawModule(code=_UNPACK_SRC)
    return _unpack_mod.get_function(name)


# --------------------------------------------------------------------------
# the row-compressed buffer a tiled contraction runs on
# --------------------------------------------------------------------------
class RowCompressed:
    """Tile rows of the unpacked cderi block, with the absent tile columns
    squeezed out.

    Tile row ``I`` holds ``(size_I, w_I, nL)`` where ``w_I`` is the number of
    AO columns in the tile columns that are present for it, so the first
    contraction over that row is
    ``contract('ijL,jk->ikL', A_I, factor[cols_I])`` -- the same contraction
    the dense path issues once, issued ``ntile`` times on a narrower ``j``.
    The arithmetic saved is exactly `tile_stats`'s ``fill``.

    The widths differ per tile row, which is why this is a loop rather than
    one padded batched call: `tile_stats`'s ``pad_fill`` says what padding
    would cost, and on every system measured it is within a few per cent of
    dense.

    Built once per ``(pattern, tile size, block size)``; the destination
    tables are what the unpack scatters through and are the only part that
    touches the pair list at run time.
    """

    @classmethod
    def for_dfobj(cls, dfobj, nao, T, ldL, order='native'):
        """The pattern of an existing ``DF`` object's stored pair list.

        ``dfobj._cderi_idx[0]`` is ``i*nao + j`` over the pairs the screening
        kept, in the original Mole's AO order.  The reordering is chosen here
        and folded into the destination tables, so nothing downstream of this
        has to know that the AO index moved -- except `unpermute`, which is
        the one place it has to come back.
        """
        addr = dfobj._cderi_idx[0]
        addr = addr.get() if hasattr(addr, 'get') else np.asarray(addr)
        rows, cols = addr // nao, addr % nao
        perm = None
        if order != 'native':
            perm = ao_perm(dfobj.mol, atom_order(dfobj.mol, rows, cols, order))
        return cls(rows, cols, nao, T, ldL, perm=perm)

    def __init__(self, rows, cols, nao, T, ldL, perm=None):
        if perm is not None:
            # the pattern is built in the permuted order and the density
            # factor is gathered through the same permutation, so the only
            # thing that ever sees the original order again is `unpermute`
            rows, cols = perm[rows], perm[cols]
        self.perm = perm
        mask, size = tile_mask(rows, cols, nao, T)
        nt = len(size)
        offs = np.append(0, np.cumsum(size))          # AO offset of each tile
        # the AO columns kept for each tile row, and where a global AO sits in
        # that list
        colpos = np.full((nt, nao), -1, dtype=np.int64)
        cols_of = []
        width = np.zeros(nt, dtype=np.int64)
        for I in range(nt):
            keep = np.concatenate([np.arange(offs[J], offs[J + 1])
                                   for J in np.flatnonzero(mask[I])])
            cols_of.append(keep)
            colpos[I, keep] = np.arange(len(keep))
            width[I] = len(keep)
        rowoff = np.append(0, np.cumsum(size * width))
        # where tile row I's gathered density-factor rows start
        self.fofs = np.append(0, np.cumsum(width))

        tI, tJ = rows // T, cols // T
        d1 = rowoff[tI] + (rows - offs[tI]) * width[tI] + colpos[tI, cols]
        d2 = rowoff[tJ] + (cols - offs[tJ]) * width[tJ] + colpos[tJ, rows]
        assert colpos[tI, cols].min() >= 0 and colpos[tJ, rows].min() >= 0

        self.nao, self.T, self.ldL = nao, T, ldL
        self.npair = len(rows)
        self.ntile, self.size, self.offs = nt, size, offs
        self.width, self.rowoff = width, rowoff
        self.cols_of = cols_of
        self.nrow = int(rowoff[-1])                   # == fill * nao^2
        self.fill = self.nrow / nao**2
        self._d1, self._d2 = d1, d2

    # -- device side -------------------------------------------------------
    def to_device(self):
        import cupy as cp
        self.d1 = cp.asarray(self._d1)
        self.d2 = cp.asarray(self._d2)
        gather = np.concatenate(self.cols_of)
        if self.perm is not None:
            # one gather does the reordering and the column compression at
            # once: `gather` indexes the permuted AOs, and `inv` sends those
            # back to the rows the caller's density factor actually has
            inv = np.empty(self.nao, dtype=np.int64)
            inv[self.perm] = np.arange(self.nao)
            gather = inv[gather]
            self.permd = cp.asarray(self.perm)
        self.gather = cp.asarray(gather)
        self.tbuf = None
        # zeroed, not empty: the AO pairs that sit inside a present tile
        # pair but were dropped by the screening are never written, and the
        # dense path leaves exactly the same positions at the zero its own
        # `cp.zeros` buffer started from.  Paid once per `get_jk` call, as
        # `fastdf`'s is -- every block writes the same positions.
        self.buf = cp.zeros(self.nrow * self.ldL)
        # The staging buffer the packed block is transposed into, which only
        # the `cupy` unpack needs.  When it exists it is allocated **once**
        # and not per block, and that is not a tidiness point:
        # `cp.ascontiguousarray` on a 763 MB temporary went to the driver
        # every block, and `cudaFree` synchronises the device, so the
        # double-buffered prefetch was drained 114 times a call and 2.33 s of
        # the 11.73 s was GPU idle (`docs/README_fastdfblock.md` §4.1).
        # W14's kernel writes the buffer straight from the packed block and
        # the staging array goes away entirely -- 0.67 GB at 284 atoms, which
        # `fastdf._blksize` gives back to the auxiliary block.
        self.tbuf = None if UNPACK == 'kernel' else self._staging()
        return self

    def _staging(self):
        """The `cupy` unpack's transpose target, allocated once.

        Eagerly in `to_device` when that is the shipped mode, and on first use
        if the mode is flipped afterwards, which only
        `benchmarks/df_block_bench.py` and `benchmarks/df_jk_bench.py` do --
        they time both unpacks against one pattern, and rebuilding a
        several-gigabyte pattern between two variants of a round robin would
        put the rebuild in the measurement.  Never per block, which is
        `docs/DF_WORK_LIST.md` §5.8.
        """
        import cupy as cp
        if self.tbuf is None:
            self.tbuf = cp.empty(self.npair * self.ldL)
        return self.tbuf

    def nbytes(self, unpack=None):
        """Device footprint of the pattern's per-block buffers.

        `unpack` names the scatter the footprint is for, because the staging
        buffer is the whole difference between the two: `kernel` does not
        have one.  It defaults to the shipped mode rather than to whether
        `tbuf` happens to be allocated, so that a benchmark which has just
        timed both does not report the other one's footprint.
        """
        mode = UNPACK if unpack is None else unpack
        return (self.nrow + (0 if mode == 'kernel' else self.npair)) \
            * self.ldL * 8

    def unpack(self, cderi_blk, nL):
        """``cderi_blk`` is ``(nL, npairs)``; scatter it and its mirror."""
        import cupy as cp
        if UNPACK == 'kernel':
            return self._unpack_kernel(cderi_blk, nL)
        flat = self.buf.reshape(self.nrow, self.ldL)
        blk = self._staging()[:self.npair * nL].reshape(self.npair, nL)
        cp.copyto(blk, cderi_blk.T)                   # into `tbuf`, no malloc
        flat[self.d1, :nL] = blk
        flat[self.d2, :nL] = blk
        return self.buf

    def _unpack_kernel(self, cderi_blk, nL):
        """W14: the transpose and the two scatters as one pass.

        The block is ``(nL, npair)``; whichever of its two strides is unit
        picks the kernel, and nothing else about the caller changes.  Both
        layouts occur -- `fastdf._blocks` yields a row slice of a
        ``(naux, npair)`` tensor for a ``float64`` cderi and W7's widener
        yields the transpose of a ``(npair, blk)`` buffer for a ``float32``
        device-resident one -- and a layout this does not recognise is a
        programming error rather than a case to fall through, because the
        fall-through would be a silent 763 MB temporary per block.
        """
        assert cderi_blk.dtype == np.float64 and cderi_blk.shape[0] >= nL
        assert cderi_blk.shape[1] == self.npair
        s0, s1 = (x // 8 for x in cderi_blk.strides)
        if self.npair == 0 or nL == 0:
            return self.buf
        grid = ((self.npair + 31) // 32, (nL + 31) // 32)
        if s1 == 1:
            name, ld = 'unpack_pair_major', s0
        elif s0 == 1:
            name, ld = 'unpack_aux_major', s1
        else:
            raise ValueError(f'cderi block strides {cderi_blk.strides} are '
                             'unit in neither index')
        _kernel(name)(
            grid, (32, 8),
            (self.buf, cderi_blk, self.d1, self.d2,
             np.int32(self.npair), np.int32(nL), np.int64(ld),
             np.int64(self.ldL)))
        return self.buf

    def factor(self, f):
        """Gather the density-factor rows each tile row needs.

        ``f`` is ``(nao, nocc)`` or ``(n_dm, nao, nocc)`` in the *caller's* AO
        order; the result is ``(..., sum_I w_I, nocc)``, which is
        ``fill * nao / tile`` times as tall as ``f`` and is built once per
        ``get_jk`` call rather than once per block.
        """
        return f[..., self.gather, :]

    def unpermute(self, vk):
        """``vk`` in the reordered AO basis -> the caller's.

        The exchange matrix comes out as ``P vk P^T``; two gathers put it
        back.  ``O(nao^2)`` against the contraction's ``O(nao^2 nocc naux)``,
        so the reordering is free in everything but code.
        """
        if self.perm is None:
            return vk
        return vk[..., self.permd, :][..., :, self.permd]

    def step1(self, fgathered, out, nL):
        """``out[...,i,k,L] = sum_j A[i,j,L] f[...,j,k]``, one call per tile
        row.

        ``fgathered`` is what `factor` returned, so it may carry a leading
        density axis; ``out`` is ``(nao, nocc, nL)`` or
        ``(n_dm, nao, nocc, nL)`` to match.  The dense path issues this
        contraction once over the whole ``j``; this issues it ``ntile`` times
        over the ``w_I`` columns the screening kept for each tile row, which
        is where the arithmetic goes.
        """
        import cupy as cp
        from gpu4pyscf.lib.cupy_helper import contract
        many = fgathered.ndim == 3
        pattern = 'ijL,njk->nikL' if many else 'ijL,jk->ikL'
        base = self.buf.data
        itemsize = 8
        for I in range(self.ntile):
            s, w = int(self.size[I]), int(self.width[I])
            ptr = base + int(self.rowoff[I]) * self.ldL * itemsize
            A = cp.ndarray((s, w, nL), dtype=np.float64,
                           memptr=ptr,
                           strides=(w * self.ldL * itemsize,
                                    self.ldL * itemsize, itemsize))
            f0, f1 = int(self.fofs[I]), int(self.fofs[I + 1])
            p0 = int(self.offs[I])
            contract(pattern, A, fgathered[..., f0:f1, :],
                     out=out[..., p0:p0 + s, :, :nL])
        return out
