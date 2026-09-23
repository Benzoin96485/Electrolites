"""W15's precondition: how local is a Cholesky-localised occupied set, really?

`docs/DF_WORK_LIST.md` **W15** observes that
``vk[i,j] = sum_{k,L} rhok[i,k,L] rhok[j,k,L]`` is *exactly* invariant under an
orthogonal rotation of the density factor's second index, so the factor may be
replaced by a localised one for free and to round-off, and that ``rhok`` is
then sparse in ``(i, k)`` as well as in ``(i, j)``.  It also states the order
the work has to be done in: **measure the supports before writing anything**,
the way `df_block_probe.py` measured the tile fill before `_dfblock.py`
existed.  This is that measurement.

It is deliberately the same shape of argument as
`docs/README_fastdfblock.md` §1.1, because that is the shape W15 has to
survive.  A *pair* count is not what a GEMM does arithmetic on; a *tile* count
is.  So every fill here is reported twice -- once at element granularity,
which is the sparsity that exists, and once at (AO tile x orbital group)
granularity, which is the part of it a blocked contraction can address.

What it computes, from a converged density matrix and the cderi's own pair
list (no integrals, no SCF, no cderi -- `--dm` loads a density another run
cached):

1. **The localised factor.**  A pivoted Cholesky of ``D`` to rank ``nocc``
   gives ``D = f f^T`` exactly, with ``f`` localised; the canonical
   ``mo_coeff sqrt(mo_occ)`` is the same ``D`` with dense columns, and is the
   baseline every ratio here is against.
2. **Support.**  Per orbital, the AOs and the *atoms* a column needs to
   reproduce all but ``--drop`` of its squared norm, and the density matrix
   error that truncation actually costs.
3. **The ``(i, k)`` pattern of ``rhok``.**  ``rhok[i,k,:]`` can be nonzero
   only where some stored pair ``(i, j)`` has ``j`` in orbital ``k``'s
   support, so the pattern is the boolean product of the cderi's pair mask
   with the support mask.  Its fill is the ceiling on what W15 removes from
   the *first* contraction.
4. **The cost of both contractions under an orbital grouping.**  Orbitals are
   sorted by centroid along a recursive coordinate bisection and cut into
   groups; a group's column support is the union of its members', which is
   what makes its GEMM large and its fill worse at the same time.  The two
   numbers printed, `step1` and `step2`, are the arithmetic each contraction
   would do relative to the dense one -- directly comparable to
   `docs/README_fastdfblock.md` §3's 0.619 tile fill, which is what W5
   already removes.

    python benchmarks/df_local_occ_probe.py --cases penicillinV:0:0
    python benchmarks/df_local_occ_probe.py --cases PfPMT:0:-1 \\
        --dm /pscratch/.../dfloc_dm --groups 8,16,32,64 --hist
"""
import argparse
import json
import os
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--cases', default='penicillinV:0:0',
                help='comma separated name:spin:charge from benchmarks/molecules')
ap.add_argument('--dm', default='',
                help='prefix of cached converged densities: <prefix>.<name>.npy.  '
                     'Without it the probe converges an SCF itself, which at '
                     '284 atoms is the expensive part of the run and at 545 '
                     'is not possible at all')
ap.add_argument('--xc', default='b3lyp-d3bj')
ap.add_argument('--drop', default='1e-4,1e-5,1e-6',
                help='the share of a column\'s squared norm a support may '
                     'discard; the support is the smallest set of AOs that '
                     'keeps the rest')
ap.add_argument('--tiles', default='16',
                help='AO tile sizes, matching `df_block_probe.py`')
ap.add_argument('--groups', default='1,8,16,32,64',
                help='orbital group sizes; 1 is the element-granular ceiling '
                     'and is not a GEMM')
ap.add_argument('--orb-order', default='rcb,native',
                help='how orbitals are sorted before they are cut into '
                     'groups: `rcb` bisects their centroids, `native` is the '
                     'order the pivoted Cholesky produced them in')
ap.add_argument('--hist', action='store_true',
                help='print the distribution of the per-orbital supports')
ap.add_argument('--rhok', action='store_true',
                help='also build a real `rhok` and screen *it*, which is the '
                     'measurement the structural fill above is only an upper '
                     'bound on: it needs a cderi, and it is the only thing '
                     'that prices the item in the currency its success '
                     'criterion is written in -- the error in `vk`')
ap.add_argument('--rhok-blocks', type=int, default=2,
                help='how many auxiliary blocks to build `rhok` from')
ap.add_argument('--rhok-nl', type=int, default=64,
                help='auxiliary functions per block')
ap.add_argument('--taus', default='1e-4,1e-6,1e-8,1e-10,1e-12',
                help='|rhok| thresholds, relative to the block maximum')
a = ap.parse_args()

import cupy as cp                                             # noqa: E402
from pyscf import gto                                         # noqa: E402
from scipy.sparse import csr_matrix                           # noqa: E402
from electrolites._dfblock import pair_indices, ao2atom       # noqa: E402


# --------------------------------------------------------------------------
# the localised factor
# --------------------------------------------------------------------------
def pivoted_cholesky(D, rank, tol=1e-12):
    """``D ~= f f^T`` with ``f`` ``(nao, rank)``, pivoting on the largest
    remaining diagonal.

    This is the localisation W15 names, and the reason it is the one to try
    first is that it is not an optimisation: each column is a column of ``D``
    itself, orthogonalised against the ones already taken, so it inherits
    ``D``'s decay with distance and costs one pass instead of a Jacobi sweep.
    ``f`` is a density factor in exactly the sense `fastdf` means -- it is
    ``mo_coeff sqrt(mo_occ)`` times an orthogonal matrix, which is the
    rotation ``vk = rhok rhok^T`` does not see.
    """
    D = cp.asarray(D, dtype=cp.float64)
    n = D.shape[0]
    d = cp.diag(D).copy()
    f = cp.zeros((n, rank))
    piv = cp.zeros(rank, dtype=cp.int64)
    for k in range(rank):
        p = int(cp.argmax(d))
        dp = float(d[p])
        if dp <= tol:
            f = f[:, :k]
            piv = piv[:k]
            break
        piv[k] = p
        col = D[:, p] - f[:, :k] @ f[p, :k]
        f[:, k] = col / np.sqrt(dp)
        d -= f[:, k] ** 2
        d = cp.maximum(d, 0.)
    return f, piv


def supports(f, drop):
    """The smallest per-column AO set that keeps all but `drop` of the
    squared norm, as a boolean ``(nao, ncol)`` mask.

    A norm criterion rather than a magnitude one because the quantity that
    has to be controlled is the *density matrix* the truncated factor
    reproduces, and that is a sum of squares.
    """
    w = f ** 2
    order = cp.argsort(w, axis=0)                     # smallest first
    ws = cp.take_along_axis(w, order, axis=0)
    keep_from = cp.cumsum(ws, axis=0) > drop * w.sum(0)[None, :]
    mask = cp.zeros(f.shape, dtype=bool)
    cp.put_along_axis(mask, order, keep_from, axis=0)
    return mask


def centroids(f, coords, a2a):
    """Each column's Mulliken-like centre, for the grouping below."""
    w = cp.asarray(f) ** 2
    at = cp.zeros((int(a2a.max()) + 1, w.shape[1]))
    cp.add.at(at, cp.asarray(a2a), w)
    at /= at.sum(0)[None, :]
    return (cp.asarray(coords).T @ at).T.get()        # (ncol, 3)


def rcb_order(xyz):
    """Recursive coordinate bisection of a point set -- `_dfblock.atom_order`'s
    `rcb`, applied to orbital centroids instead of to nuclei."""
    def split(idx):
        if len(idx) <= 2:
            return [idx]
        d = np.ptp(xyz[idx], axis=0).argmax()
        o = idx[np.argsort(xyz[idx, d], kind='stable')]
        h = len(o) // 2
        return split(o[:h]) + split(o[h:])
    return np.concatenate(split(np.arange(len(xyz))))


# --------------------------------------------------------------------------
# the two contractions, costed
# --------------------------------------------------------------------------
def costs(P, S, nao, group, tile):
    """What W15 would remove from step 1 and step 2, at this grouping.

    `P` is the cderi's symmetric AO-pair mask as a CSR matrix and `S` the
    dense boolean support mask with its columns already in group order.  Both
    accumulate in a width that cannot overflow: an `int8` mask summed over a
    group of orbitals wraps at 128 and silently reports a *smaller* support
    than the one that exists, which is the one failure mode of this probe that
    would flatter the item it is testing.  For one group of orbitals ``g``:

    * its **column support** ``U`` is the union of its members' supports --
      the AO rows of the density factor the group's GEMM has to read;
    * its **row support** ``R`` is every AO ``i`` that shares a stored pair
      with some ``j`` in ``U``, which is where ``rhok`` can be nonzero.

    Step 1 for the group is then ``(|R| x |U|) . (|U| x |g|)`` per auxiliary
    function and step 2 is ``(|R| x |g|) . (|g| x |R|)``, against ``nao`` in
    place of every one of ``|R|`` and ``|U|`` in the dense path.  Quantising
    ``U`` and ``R`` to AO tiles of `tile` is what makes the answer a GEMM's
    rather than an element's, and the difference between the two is the whole
    of §1.1's lesson applied to this item.
    """
    nocc = S.shape[1]
    nt = (nao + tile - 1) // tile
    sizes = np.full(nt, tile)
    sizes[-1] = nao - tile * (nt - 1)
    s1 = s2 = s1t = s2t = 0.0
    umax = rmax = 0
    for g0 in range(0, nocc, group):
        g1 = min(g0 + group, nocc)
        ng = g1 - g0
        U = S[:, g0:g1].any(1)
        R = np.asarray(P @ U.astype(np.float64)).ravel() > 0
        s1 += R.sum() * U.sum() * ng
        s2 += R.sum() ** 2 * ng
        # the same two sets, rounded out to whole AO tiles
        ut = sizes[np.flatnonzero(
            np.add.reduceat(U.astype(np.int64), np.r_[0, np.cumsum(sizes)[:-1]])
            > 0)].sum()
        rt = sizes[np.flatnonzero(
            np.add.reduceat(R.astype(np.int64), np.r_[0, np.cumsum(sizes)[:-1]])
            > 0)].sum()
        s1t += rt * ut * ng
        s2t += rt ** 2 * ng
        umax, rmax = max(umax, int(U.sum())), max(rmax, int(R.sum()))
    dense = float(nao) ** 2 * nocc
    return dict(step1=s1 / dense, step2=s2 / dense,
                step1_tiled=s1t / dense, step2_tiled=s2t / dense,
                u_max=umax, r_max=rmax)


# --------------------------------------------------------------------------
# what screening `rhok` itself would leave  (`--rhok`)
# --------------------------------------------------------------------------
def rhok_screen(mol, f, tile, groups, taus, nblk, nL, auxbasis):
    """Build a real ``rhok`` and ask the only question that prices W15.

    Everything above is *structural*: it asks where ``rhok`` **can** be
    nonzero, given which AO pairs the cderi stores.  That is an upper bound
    and a loose one, because the cderi's entries decay long before the
    screening drops them.  This builds
    ``rhok[i,k,L] = sum_j A[i,j,L] f[j,k]`` for a few auxiliary blocks and
    measures, at a series of magnitude thresholds:

    * what fraction of the ``(AO tile, orbital group)`` blocks survive -- the
      arithmetic a blocked step 1 would still have to do, and the arithmetic
      a blocked step 2 would do on the surviving row supports;
    * **the relative error in ``vk``** that discarding the rest costs, on the
      same block.  That is the number W15's success criterion is written in,
      and it is the one that cannot be inferred from a pattern.

    The same measurement runs on the canonical factor as a control: if a
    magnitude threshold prunes the canonical ``rhok`` just as well, the
    localisation is not what bought the sparsity.
    """
    import cupy as cp
    from gpu4pyscf.df.df import DF
    from gpu4pyscf.lib.cupy_helper import contract, fill_symmetric
    dfobj = DF(mol, auxbasis=auxbasis)
    dfobj.build()
    cderi = dfobj._cderi[cp.cuda.Device().id]
    naux, npair = cderi.shape
    nao, nocc = f.shape
    pair_idx = cp.asarray(dfobj._cderi_idx[0], dtype=np.int32)
    nt = (nao + tile - 1) // tile
    starts = np.linspace(0, max(naux - nL, 0), nblk, dtype=int)
    work = cp.zeros((nao, nao, nL))
    out = []
    for label, fac in (('cholesky', f), ('canonical', None)):
        if fac is None:                       # the control: canonical MOs
            w, v = cp.linalg.eigh(cp.asarray(f) @ cp.asarray(f).T)
            fac = v[:, -nocc:] * cp.sqrt(cp.maximum(w[-nocc:], 0.))[None, :]
        fac = cp.ascontiguousarray(cp.asarray(fac))
        acc = {}
        for p0 in starts:
            blk = cp.asarray(cderi[p0:p0 + nL])
            dense = fill_symmetric(blk.T, pair_idx, nao, 0, nL, out=work)
            rhok = contract('Lij,jk->ikL', dense.transpose(2, 0, 1), fac)
            vk = contract('ikL,jkL->ij', rhok, rhok)
            scale = float(cp.abs(vk).max()) or 1.
            amp = cp.abs(rhok).max(axis=2)            # (nao, nocc)
            top = float(amp.max())
            # The same fill for **one** auxiliary function rather than for the
            # whole block.  The gap between the two says whether localising
            # the auxiliary index as well would buy anything: if one `L` is
            # no sparser in `(i, k)` than the max over 64 of them, then no
            # auxiliary domain, however compact, concentrates the pattern --
            # and if it is much sparser, that is the direction this probe did
            # not test.  Averaged over four `L` spread across the block.
            for tau in taus:
                f1 = fb1 = 0.
                for l in np.linspace(0, nL - 1, 4, dtype=int):
                    k1 = (cp.abs(rhok[:, :, int(l)]) > tau * top).get()
                    f1 += k1.mean() / 4
                    gg = groups[min(1, len(groups) - 1)]
                    ng = (nocc + gg - 1) // gg
                    b1 = np.zeros((nt, ng), dtype=bool)
                    for gi in range(ng):
                        b1[:, gi] = np.add.reduceat(
                            k1[:, gi * gg:(gi + 1) * gg].any(1).astype(
                                np.int64), np.arange(0, nao, tile)) > 0
                    fb1 += b1.mean() / 4
                acc.setdefault('per_L', {}).setdefault(tau, [0., 0., 0])
                pl = acc['per_L'][tau]
                pl[0] += f1
                pl[1] += fb1
                pl[2] += 1
            for tau in taus:
                keep = amp > tau * top
                err = float(cp.abs(contract(
                    'ikL,jkL->ij', cp.where(keep[:, :, None], rhok, 0.),
                    cp.where(keep[:, :, None], rhok, 0.)) - vk).max()) / scale
                k = acc.setdefault(tau, dict(elem=0., err=0., n=0,
                                             blocked={g: 0. for g in groups},
                                             s2={g: 0. for g in groups}))
                k['elem'] += float(keep.mean())
                k['err'] = max(k['err'], err)
                k['n'] += 1
                kh = keep.get()
                for g in groups:
                    ng = (nocc + g - 1) // g
                    # (AO tile x orbital group) occupancy, and the row support
                    # each group would still need for step 2
                    b = np.zeros((nt, ng), dtype=bool)
                    for gi in range(ng):
                        col = kh[:, gi * g:(gi + 1) * g].any(1)
                        rows_t = np.add.reduceat(
                            col.astype(np.int64),
                            np.arange(0, nao, tile)) > 0
                        b[:, gi] = rows_t
                        rt = min(rows_t.sum() * tile, nao)
                        k['s2'][g] += rt ** 2 * min(g, nocc - gi * g)
                    k['blocked'][g] += b.mean()
            del rhok, vk, amp
            cp.get_default_memory_pool().free_all_blocks()
        per_L = acc.pop('per_L', {})
        for tau, k in sorted(acc.items()):
            pl = per_L.get(tau, [0., 0., 1])
            row = dict(kind=label, tau=tau, elem=k['elem'] / k['n'],
                       elem_1L=pl[0] / pl[2], blocked_1L=pl[1] / pl[2],
                       vk_err=k['err'],
                       blocked={g: k['blocked'][g] / k['n'] for g in groups},
                       step2={g: k['s2'][g] / k['n'] / (float(nao) ** 2 * nocc)
                              for g in groups})
            out.append(row)
            print(f'  rhok {label:9s} tau={tau:.0e}  elem={row["elem"]:6.3f}  '
                  + '  '.join(f'g{g}={row["blocked"][g]:.3f}/'
                              f'{row["step2"][g]:.3f}' for g in groups)
                  + f'   1L={row["elem_1L"]:.3f}/{row["blocked_1L"]:.3f}'
                  + f'   relK={row["vk_err"]:.1e}', flush=True)
    del work
    cp.get_default_memory_pool().free_all_blocks()
    return out


# --------------------------------------------------------------------------
results = []
DROPS = [float(x) for x in a.drop.split(',')]
TILES = [int(x) for x in a.tiles.split(',')]
GROUPS = [int(x) for x in a.groups.split(',')]
TAUS = [float(x) for x in a.taus.split(',')]

for case in a.cases.split(','):
    name, spin, charge = (case.split(':') + ['0', '0'])[:3]
    mol = gto.M(atom=f'benchmarks/molecules/{name}.xyz', basis=a.basis,
                spin=int(spin), charge=int(charge), verbose=0)
    nao = mol.nao
    nocc = (mol.nelectron - mol.spin) // 2
    cache = f'{a.dm}.{name}.npy' if a.dm else ''
    if cache and os.path.exists(cache):
        dm = np.load(cache)
    else:
        from gpu4pyscf import dft
        mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc).density_fit(
            auxbasis=a.auxbasis)
        mf.verbose = 0
        mf.kernel()
        dm = cp.asnumpy(mf.make_rdm1())
        if cache:
            np.save(cache, dm)
        del mf
    if dm.ndim == 3:                                   # unrestricted: alpha
        dm = dm[0] * 2
    rows, cols, nao_p, naux = pair_indices(mol, a.auxbasis)
    assert nao_p == nao
    P = csr_matrix((np.ones(2 * len(rows)),
                    (np.r_[rows, cols], np.r_[cols, rows])), shape=(nao, nao))
    P.data[:] = 1.
    pairfill = P.nnz / nao ** 2

    t0 = time.perf_counter()
    f, piv = pivoted_cholesky(dm, nocc)
    tchol = time.perf_counter() - t0
    err = float(cp.abs(cp.asarray(dm) - f @ f.T).max()
                / cp.abs(cp.asarray(dm)).max())
    print(f'# {name}: natm {mol.natm} nao {nao} nocc {nocc} naux {naux} '
          f'npair {len(rows)} pairfill {100*pairfill:.1f} %  '
          f'| pivoted Cholesky rank {f.shape[1]} in {tchol:.2f} s, '
          f'|D - f f^T|/|D| = {err:.1e}', flush=True)

    a2a = np.asarray(ao2atom(mol), dtype=np.int64)
    xyz = centroids(f, mol.atom_coords(), a2a)
    print(f'{"drop":>6s} {"AO supp":>8s} {"max":>6s} {"atoms":>7s} {"max":>5s} '
          f'{"dD":>8s} {"order":>7s} {"group":>6s} {"tile":>5s} '
          f'{"step1":>7s} {"step2":>7s} {"step1/T":>8s} {"step2/T":>8s} '
          f'{"|U|max":>7s} {"|R|max":>7s}')
    for drop in DROPS:
        S = supports(f, drop)
        ao_supp = S.sum(0).get()
        # the truncated factor's own density-matrix error, which is the
        # error budget W15's exchange matrix would inherit
        ft = cp.where(S, f, 0.)
        dtrunc = float(cp.abs(cp.asarray(dm) - ft @ ft.T).max()
                       / cp.abs(cp.asarray(dm)).max())
        atoms = np.zeros(S.shape[1], dtype=np.int64)
        Sh = S.get()
        for at in range(mol.natm):
            atoms += Sh[a2a == at].any(0)
        if a.hist:
            q = np.percentile(ao_supp, [50, 75, 90, 99])
            print(f'#   drop {drop:.0e}: AO support percentiles '
                  f'50/75/90/99 = {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}/{q[3]:.0f} '
                  f'of {nao}', flush=True)
        for oo in a.orb_order.split(','):
            perm = (rcb_order(xyz) if oo == 'rcb'
                    else np.arange(S.shape[1]))
            Sp = Sh[:, perm]
            for group in GROUPS:
                for tile in TILES:
                    c = costs(P, Sp, nao, group, tile)
                    row = dict(name=name, nao=nao, nocc=nocc, natm=mol.natm,
                               drop=drop, order=oo, group=group, tile=tile,
                               pairfill=pairfill, dtrunc=dtrunc,
                               ao_supp_mean=float(ao_supp.mean()),
                               ao_supp_max=int(ao_supp.max()),
                               atom_supp_mean=float(atoms.mean()),
                               atom_supp_max=int(atoms.max()), **c)
                    print(f'{drop:6.0e} {ao_supp.mean():8.1f} '
                          f'{ao_supp.max():6d} {atoms.mean():7.1f} '
                          f'{atoms.max():5d} {dtrunc:8.1e} {oo:>7s} '
                          f'{group:6d} {tile:5d} '
                          f'{c["step1"]:6.3f}x {c["step2"]:6.3f}x '
                          f'{c["step1_tiled"]:7.3f}x {c["step2_tiled"]:7.3f}x '
                          f'{c["u_max"]:7d} {c["r_max"]:7d}', flush=True)
                    results.append(row)
    if a.rhok:
        results.extend(
            dict(name=name, nao=nao, nocc=nocc, natm=mol.natm, tile=TILES[0],
                 **r)
            for r in rhok_screen(mol, f, TILES[0], GROUPS, TAUS,
                                 a.rhok_blocks, a.rhok_nl, a.auxbasis))
    del f, ft, S
    cp.get_default_memory_pool().free_all_blocks()


print('RESULT ' + json.dumps(dict(basis=a.basis, rows=results)))
