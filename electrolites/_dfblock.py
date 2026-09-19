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
import numpy as np


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
        # zeroed, not empty: the AO pairs that sit inside a present tile
        # pair but were dropped by the screening are never written, and the
        # dense path leaves exactly the same positions at the zero its own
        # `cp.zeros` buffer started from.  Paid once per `get_jk` call, as
        # `fastdf`'s is -- every block writes the same positions.
        self.buf = cp.zeros(self.nrow * self.ldL)
        return self

    def nbytes(self):
        return self.nrow * self.ldL * 8

    def unpack(self, cderi_blk, nL):
        """``cderi_blk`` is ``(nL, npairs)``; scatter it and its mirror."""
        import cupy as cp
        flat = self.buf.reshape(self.nrow, self.ldL)
        blk = cp.ascontiguousarray(cderi_blk.T)       # (npairs, nL)
        flat[self.d1, :nL] = blk
        flat[self.d2, :nL] = blk
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
