"""
Robust **pair-atomic** density fitting (PARI), as an explicitly opt-in
approximation rather than a drop-in replacement.

``docs/DF_WORK_LIST.md`` **W6** proposes fitting every AO pair in the
auxiliary functions of its own two atoms instead of in all of them.
``docs/README_dflocal.md`` settled the two questions that item attached before
any code -- the fit has to be **robust** (Dunlap), because the non-robust form
collapses variationally by 16.5 Eh on butane, and the storage claim is real:
a pair domain holds 118-130 auxiliary functions on every system from 10 to 545
atoms.  What it left open is the implementation and, inside that, *"PARI-K is
a different contraction order and estimating its cost is the next piece of
work"*.  This module is that contraction order, written so that it can be
measured; ``docs/README_dflocal_k.md`` is what measuring it said.

**It does not compute the same number as GPU4PySCF, and that is why it is not
a patch.**  Every other module in ``electrolites/`` replaces an entry point and
agrees with upstream to 1e-14; this one is a different approximation --
**2e-5 to 8e-5 Eh per atom** away from global density fitting by
construction, flat in system size and varying by a factor of three with the
chemistry -- so it is reached only by asking for it:

    import electrolites.dflocal as dflocal
    mf = dflocal.density_fit(mol.RKS(xc='b3lyp').to_gpu())
    e = mf.kernel()                      # PARI-r, with a warning on stdout

``patch_all()`` never loads it and ``tests/test_full.py`` never sees it.

The algebra
-----------

Write ``rho_mn`` for an AO pair density and ``rho~_mn`` for its fit inside the
pair's own domain.  The robust (Dunlap) integral is

    (mn|ls) ~ (mn|rho~_ls) + (rho~_mn|ls) - (rho~_mn|rho~_ls)

whose error is second order in the fit residual where the non-robust
``(rho~_mn|rho~_ls)`` alone is first order.  With ``B[mn,Q] = (mn|Q)`` the
exact three-centre integrals, ``c[mn,P]`` the local fitting coefficients and
``T = c J`` (``J`` the auxiliary Coulomb metric), that is

    eri = B c^T + c B^T - T c^T

and the two working forms this module uses follow from it in one line each.
With ``D`` the density, ``d = c^T D``, ``a = M^T D`` and ``X[n,Q,l] = sum_s
c[ls,Q] D[n,s]``:

    M   = B - T/2                      folded once, at build time
    vj  = M d + c a
    vk  = M X + (M X)^T

**One tensor and one pass.**  ``T`` is never formed at run time and the
exchange needs no second contraction: ``M X`` is the whole of it, because
``T c^T`` is symmetric and so may be split evenly between the two halves of
the robust expression.  Global DF needs two contractions of the same shape --
``rhok = cderi . mo`` and ``vk = rhok rhok^T`` -- so the flop ratio is

    global / PARI = 2 nocc / nlambda

where ``nlambda`` is the number of AOs that overlap the auxiliary function's
own atom.  That ratio, not the storage, is what decides the exchange build,
and
``benchmarks/df_local_k_probe.py`` measures it: 1.09 at 284 atoms unscreened,
and **2.93** once the fitting coefficients are screened at a threshold whose
cost on the exchange matrix is 4400 times inside the approximation's own
error.  §1 of the write-up is that measurement and what it means.

What is fitted where
--------------------

The fit is solved per *atom* pair, not per AO pair: every AO pair of one atom
pair shares a domain and therefore a metric, so there is one factorisation per
atom pair (of a 118-130 row matrix, condition number ~2e7) in place of global
DF's one factorisation of the whole ``naux x naux`` metric (condition number
2.2e8 to 2.3e9, and growing with the molecule -- ``docs/README_dflocal.md``
§1.1).

Switches
--------

    DFLOCAL_DOMAIN=pair|pair+<r>   the fitting domain: the pair's own two
                                   atoms, or every atom within <r> Angstrom
                                   of either (default ``pair``)
    DFLOCAL_STORE=auto|gpu|cpu     where the folded tensor lives; `auto`
                                   keeps it on the device while it is under
                                   half the free memory
    DFLOCAL_BLK=<n>                auxiliary functions per exchange pass;
                                   0 (default) takes one whole atom at a
                                   time, split if the unpacked block would
                                   not fit
    DFLOCAL_KMODE=auto|ao|mo       which of the two exchange contraction
                                   orders to run.  ``ao`` contracts the
                                   coefficients with the density first, which
                                   is the one whose cost is ``2 nao^2 naux
                                   nlambda``; ``mo`` contracts them with the
                                   occupied orbitals instead, which leaves
                                   global DF's first contraction untouched and
                                   replaces only its second.  ``auto``
                                   compares the two flop counts, which cross
                                   at ``nlambda = nocc/(1 - nocc/nao)``
    DFLOCAL_LAMBDA_TOL=<t>         drop from the exchange intermediate's
                                   second index every AO whose largest fitting
                                   coefficient on the pass's atom is below
                                   ``t`` times the atom's largest.  0 (the
                                   default) keeps all of them; the exchange
                                   cost is linear in what is left, and
                                   ``benchmarks/df_local_k_probe.py --lam``
                                   is where the error it costs is measured.
                                   Measured, **1e-6** leaves 48 % of
                                   ``nlambda`` at 113 atoms and 37 % at 284
                                   for 1.2e-7 relative on the exchange matrix
    DFLOCAL_FOLD=atom|csr          how ``B -= cJ/2`` is done at build time:
                                   one dense GEMM per auxiliary atom, or one
                                   cuSPARSE product per batch.  5.8x apart at
                                   113 atoms, and `_fold` says why
    DFLOCAL_STORE=gpu|cpu          where the folded tensor lives
"""
import os
import warnings

import numpy as np
import cupy as cp

from pyscf import lib
from pyscf.df import addons

from gpu4pyscf.df import df as DFMOD
from gpu4pyscf.df import df_jk as DFJK
from gpu4pyscf.df import int3c2e_bdiv
from gpu4pyscf.gto.mole import SortedMole
from gpu4pyscf.lib.cupy_helper import (
    contract, get_avail_mem, fill_symmetric, ndarray)
from gpu4pyscf.lib import logger
# `_dfaux` and not `fastdfcderi`: importing that module patches
# `_cholesky_eri`, and a module that only wants its kernels must not change
# the build it is being measured against
from electrolites._dfaux import (
    _contract_aux, _sparse_ctr_coeff, _spmult, _SPMULT)
from cupyx import scatter_add
from cupyx.scipy.sparse import csr_matrix

DOMAIN = os.environ.get('DFLOCAL_DOMAIN', 'pair')
STORE = os.environ.get('DFLOCAL_STORE', 'auto')
BLK = int(os.environ.get('DFLOCAL_BLK', 0))
#: 'atom' (one dense GEMM per auxiliary atom) or 'csr' (one
#: cuSPARSE product per batch) for the fold; see `_fold`.
FOLD = os.environ.get('DFLOCAL_FOLD', 'atom')
LAMBDA_TOL = float(os.environ.get('DFLOCAL_LAMBDA_TOL', 0))
KMODE = os.environ.get('DFLOCAL_KMODE', 'auto')

#: counters, so a run can say what it did without being timed
STATS = {'builds': 0, 'calls': 0, 'passes': 0, 'nlocal': 0.0, 'nlambda': 0.0,
         'store': '', 'npair': 0, 'naux': 0, 'kmode': ''}

#: filled only when a `PARIDF` has ``profile`` set, and then with a
#: synchronising timer between stages -- which perturbs the overlap and is
#: the point: it is what says where a pass's time goes, not what the pass
#: costs.  `benchmarks/df_local_k_bench.py --anatomy` prints it.
STAGE = {}


def stats():
    return dict(STATS)


# --------------------------------------------------------------------------
# the domains
# --------------------------------------------------------------------------
def _ao2atom(m):
    out = np.empty(m.nao, dtype=np.int32)
    for at, (_, _, p0, p1) in enumerate(m.aoslice_by_atom()):
        out[p0:p1] = at
    return out


def _domain_atoms(mol, spec):
    """``near[A]`` -- the atoms whose auxiliary functions a pair on ``A`` is
    fitted in.  ``pair`` is PARI proper; ``pair+r`` is the wider domain
    ``docs/README_dflocal.md`` §2.2 measures at 1e-7 Eh per atom."""
    if spec == 'pair':
        return np.eye(mol.natm, dtype=bool)
    r = float(spec.split('+')[1])
    d = np.linalg.norm(mol.atom_coords()[:, None] - mol.atom_coords()[None],
                       axis=-1) * lib.param.BOHR
    return d <= r


class _Domains:
    """The distinct atom-pair domains, their padded auxiliary index lists and
    the inverse of each domain's Coulomb metric.

    Padding is to the widest domain, with a zero coefficient and a dummy
    auxiliary index, so that every per-pair operation is one dense array
    rather than a ragged one.  The padded metric is the identity on the pad,
    which keeps the batched factorisation well posed and leaves the padded
    coefficients exactly zero.
    """

    def __init__(self, mol, auxmol, rows, cols, spec, j2c):
        natm = mol.natm
        a2a = cp.asarray(_ao2atom(mol))
        ar, ac = a2a[rows], a2a[cols]
        lo = cp.minimum(ar, ac).astype(cp.int64)
        hi = cp.maximum(ar, ac).astype(cp.int64)
        uniq, self.blk = cp.unique(lo * natm + hi, return_inverse=True)
        uniq = uniq.get()
        self.lo, self.hi = uniq // natm, uniq % natm
        self.nblock = len(uniq)

        xsl = auxmol.aoslice_by_atom()
        near = _domain_atoms(mol, spec)
        x2a = _ao2atom(auxmol)
        aux_of = [np.where(np.isin(x2a, np.where(near[A])[0]))[0]
                  for A in range(natm)]
        # the domain of an atom pair, sorted, so that each of the two atoms'
        # own auxiliary blocks is a contiguous slice inside it whose offset
        # `_build_domain_blocks` can find with one search
        doms = [aux_of[A] if A == B else np.union1d(aux_of[A], aux_of[B])
                for A, B in zip(self.lo, self.hi)]
        self.nloc = np.array([len(d) for d in doms])
        self.nlm = int(self.nloc.max())
        # the pad points at the *last* auxiliary function rather than the
        # first, so that every row of the coefficient matrix has ascending
        # column indices; the fold reads it as a CSR and cuSPARSE wants that.
        # Its coefficient is zero either way -- `jinv`'s pad is zeroed below.
        dom = np.full((self.nblock, self.nlm), auxmol.nao - 1, dtype=np.int32)
        for b, d in enumerate(doms):
            dom[b, :len(d)] = d
        self.dom = cp.asarray(dom)
        #: where atom ``lo[b]``/``hi[b]``'s own auxiliary block starts inside
        #: the domain of block ``b``
        self.off_lo = np.array([np.searchsorted(d, xsl[A, 2])
                                for d, A in zip(doms, self.lo)], dtype=np.int32)
        self.off_hi = np.array([np.searchsorted(d, xsl[A, 2])
                                for d, A in zip(doms, self.hi)], dtype=np.int32)

        # one padded inverse metric per atom-pair domain.  `cp.linalg.inv` is
        # a batched cuSOLVER factorisation; the domains are the well
        # conditioned ones of `docs/README_dflocal.md` §1.1 (median 2e7), so
        # an explicit inverse loses about 1e-9 of a matrix whose approximation
        # error is 2.5e-5 Eh per atom.
        # one padded inverse metric per atom-pair domain, in chunks of
        # blocks so that the gather does not need the whole `nblock x nlm^2`
        # at once.  The pad is the identity, which keeps the batched
        # factorisation well posed; zeroing its rows and columns afterwards is
        # what makes the padded coefficients exactly zero, and every later
        # operation -- the fold, the `d` scatter, the `c a` term -- rests on
        # that rather than on the auxiliary index the pad points at.
        keep = cp.asarray(np.arange(self.nlm)[None, :] < self.nloc[:, None])
        eye = cp.eye(self.nlm)
        self.jinv = cp.empty((self.nblock, self.nlm, self.nlm))
        step = max(1, int(3e7) // (self.nlm * self.nlm))
        for b0 in range(0, self.nblock, step):
            b1 = min(self.nblock, b0 + step)
            d = self.dom[b0:b1]
            m2 = keep[b0:b1, :, None] & keep[b0:b1, None, :]
            jp = cp.where(m2, j2c[d[:, :, None], d[:, None, :]], eye)
            self.jinv[b0:b1] = cp.linalg.inv(jp) * m2
        del jp, eye, m2


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------
def _aux_coeff(intopt, aux_sorting):
    """The contraction-and-spherical matrix upstream applies to the integral
    kernel's output, *without* ``L^{-T}``.

    That factor is exactly the difference between a cderi and the raw
    three-centre integrals this module needs: a robust fit is written in
    ``(mn|P)``, not in ``L^{-1}(P|mn)``.  And dropping it is the difference
    between a dense ``2 npair naux_sorted naux`` product -- which is
    `docs/README_fastdfcderi.md` §2's metric contraction, **59.5 %** of a
    284-atom block loop -- and a gather of about 1.5 entries per column,
    because ``ctr_coeff`` alone is the Cartesian-to-spherical matrix of one
    shell at a time.  `fastdfcderi` already has that kernel and this returns
    what it takes.
    """
    return _sparse_ctr_coeff(intopt.auxmol, aux_sorting)


class PARIDF(DFMOD.DF):
    """A ``DF``-shaped object whose tensor is a robust pair-atomic fit.

    It keeps ``DF``'s interface -- ``build()``, ``get_jk()``, ``auxmol`` --
    so that ``df_jk._DFHF`` drives it unchanged, and nothing else: there is no
    ``_cderi``, so any code that reaches for one (the DF gradient, the DF
    Hessian) fails loudly instead of silently using the wrong tensor.  W6's
    entry says so in as many words: a non-variational fit carries gradient
    terms global DF does not.
    """

    def __init__(self, mol, auxbasis=None, domain=None, store=None,
                 lambda_tol=None, kmode=None):
        super().__init__(mol, auxbasis)
        self.domain = DOMAIN if domain is None else domain
        self.store = STORE if store is None else store
        self.lambda_tol = LAMBDA_TOL if lambda_tol is None else lambda_tol
        self.kmode = KMODE if kmode is None else kmode
        self.profile = False
        self.M = None
        self.c = None
        self.fit_only = False

    # -- the folded tensor ------------------------------------------------
    def build(self, direct_scf_tol=None, omega=None, build_cderi=True,
              fold=.5, fit_only=False):
        """``fold`` is how much of ``T = c J`` is subtracted from the raw
        three-centre integrals: ``0.5`` is the tensor the build wants
        (§0), ``0`` leaves the raw integrals ``B`` and ``1`` leaves the fit
        residual's potential ``R = B - T``, which is what
        ``benchmarks/df_local_k_probe.py --residual`` measures the decay
        of.

        ``fit_only`` keeps the coefficients and throws the integrals away as
        they are produced, so the fit -- and with it ``nlambda``, which is
        what the exchange costs -- can be measured at a size where the tensor
        itself does not fit anywhere: 86 GB at 284 atoms and 414 at 545.
        Nothing can be contracted afterwards."""
        assert omega is None or omega == 0, 'range separation is not fitted here'
        mol = self.mol
        log = logger.new_logger(mol, mol.verbose)
        t0 = log.init_timer()
        if self.auxmol is None:
            self.auxmol = addons.make_auxmol(mol, self.auxbasis)
        auxmol = self.auxmol
        self.nao = nao = mol.nao
        self.naux = naux = auxmol.nao

        intopt = int3c2e_bdiv.Int3c2eOpt(mol, auxmol)
        intopt.mol = SortedMole.from_mol(mol, decontract=True)
        intopt.build()
        self.intopt = intopt

        pair_addresses, diags = intopt.pair_and_diag_indices()
        self.pair_idx = cp.asarray(pair_addresses, dtype=np.int32)
        self.diags = diags
        rows, cols = cp.divmod(cp.asarray(pair_addresses), nao)
        self.rows, self.cols = rows.astype(cp.int32), cols.astype(cp.int32)
        npair = len(rows)

        j2c = cp.asarray(auxmol.intor('int2c2e'))
        self.j2c = j2c
        dom = _Domains(mol, auxmol, rows, cols, self.domain, j2c)
        self.dom = dom
        log.debug1('dflocal: %d atom-pair domains, nlocal %d-%d',
                   dom.nblock, dom.nloc.min(), dom.nloc.max())

        self.fit_only = fit_only
        on_gpu = fit_only or self._on_gpu(npair, naux)
        self.M = None if fit_only else (cp.empty((naux, npair)) if on_gpu else
                                        np.empty((naux, npair)))
        self.c = cp.empty((npair, dom.nlm))

        mem = get_avail_mem(exclude_memory_pool=True)
        batch = max(1, int(mem * .15) // (8 * max(naux, 1)))
        if not on_gpu:
            # a host-resident tensor is written one column stripe at a time,
            # and the stripe is transposed on the device first -- so the batch
            # has to leave room for two copies of it, not one
            batch = min(batch, max(1, int(1.5e9) // (8 * max(naux, 1))))
        # `fastdfcderi`'s launch fix, which is not about this module at all:
        # `int3c2e_evaluator` aggregates shell pairs four deep where one is
        # 1.19x-1.50x faster on the same kernel (`docs/README_fastdfcderi.md`
        # §3).  The integrals here are upstream's integrals and there is no
        # reason for them to be launched worse.
        with _spmult(_SPMULT):
            eval_j3c, aux_sorting, ao_pair_offsets, _ = \
                intopt.int3c2e_evaluator(ao_pair_batch_size=batch,
                                         reorder_aux=True)
        idx, coef, counts, maxnnz = _aux_coeff(intopt, aux_sorting)
        naux_sorted = int(intopt.auxmol.ctr_coeff.shape[0])
        mbatch = int(max(ao_pair_offsets[1:] - ao_pair_offsets[:-1]))
        work = cp.empty(naux_sorted * mbatch)
        bbuf = cp.empty(naux * mbatch)

        blk_h = dom.blk
        tick = _ticker(self, 'build.')
        tick('setup')
        for bid in range(len(ao_pair_offsets) - 1):
            p0, p1 = ao_pair_offsets[bid:bid + 2]
            j3c = eval_j3c(shl_pair_batch_id=bid, out=work)
            B = ndarray((p1 - p0, naux), buffer=bbuf)  # raw (mn|Q)
            _contract_aux(j3c, idx, coef, counts, np.int32(naux_sorted),
                          np.int32(naux), np.int32(maxnnz), B)
            tick('int3c2e')
            self._fit_and_fold(B, blk_h[p0:p1], p0, p1,
                               0.0 if fit_only else fold)
            tick('fit+fold')
            if fit_only:
                continue
            if on_gpu:
                self.M[:, p0:p1] = B.T
            else:
                self.M[:, p0:p1] = cp.asnumpy(B.T)
            tick('store')
        del work

        if self.domain == 'pair':
            self._build_domain_blocks()
        else:
            self._build_domain_blocks_wide()
        tick('blocks')
        STATS['builds'] += 1
        STATS['nlocal'] = float(dom.nloc.mean())
        STATS['store'] = 'gpu' if on_gpu else 'cpu'
        STATS['npair'], STATS['naux'] = int(npair), int(naux)
        log.timer_debug1('dflocal build', *t0)
        return self

    def _on_gpu(self, npair, naux):
        if self.store == 'gpu':
            return True
        if self.store == 'cpu':
            return False
        need = npair * naux * 8
        return need < get_avail_mem(exclude_memory_pool=True) * .5

    def _fit_and_fold(self, B, blk, p0, p1, fold=.5):
        """``c = J_dom^-1 B_dom`` per atom-pair domain, then ``B -= fold T``.

        Every AO pair of one atom pair shares a metric, so the solve is one
        multiply per *atom* pair -- and the pairs of an atom pair are as many
        as the product of the two atoms' AO counts, which takes only a handful
        of distinct values across a molecule.  So the blocks are grouped by
        that count and each group is one gather and one batched GEMM, rather
        than a Python loop over ten thousand domains.

        The fold is the other half and is `_fold`.
        """
        dom = self.dom
        order = cp.argsort(blk)
        present, counts = cp.unique(blk[order], return_counts=True)
        starts = cp.concatenate([cp.zeros(1, dtype=counts.dtype),
                                 cp.cumsum(counts)])[:-1]
        nlm = dom.nlm
        for n in [int(x) for x in cp.unique(counts).get()]:
            sel = cp.where(counts == n)[0]
            idx = order[starts[sel][:, None] + cp.arange(n)[None, :]]
            d = dom.dom[present[sel]]                       # (nb, nlm)
            bd = B[idx[:, :, None], d[:, None, :]]          # (nb, n, nlm)
            cb = cp.matmul(bd, dom.jinv[present[sel]])
            self.c[p0 + idx.ravel()] = cb.reshape(-1, nlm)
        if fold:
            self._fold(B, blk, p0, p1, fold)

    def _fold(self, B, blk, p0, p1, fold):
        """``B -= fold * c J``.

        This is what a pair-atomic build has in place of global DF's metric
        contraction, and the whole of the build's case rests on the flop
        count: ``npair nlocal naux`` here against upstream's ``2 npair
        naux_sorted naux``, which is **49x** less at 113 atoms and 59.5 % of
        a 284-atom block loop (`docs/README_fastdfcderi.md` §2).

        Getting that ratio out needs the right shape.  ``c`` is a sparse
        matrix -- ``nlocal`` of ``naux`` entries in a row -- and handing it to
        cuSPARSE as one is the obvious reading (``DFLOCAL_FOLD=csr``, kept so
        the alternative can be re-run): measured, that is **1.54 s** of a
        3.34 s build at 113 atoms, a tenth of the flop rate the GEMM below
        reaches.  The structure the sparse form throws away is that the
        non-zeros are not scattered: a pair's domain is *two* contiguous
        auxiliary blocks, one per atom, and every pair that touches atom
        ``A`` carries ``A``'s block at the same offset.  So the fold is one
        dense ``(npair_A, naux_A) x (naux_A, naux)`` GEMM per auxiliary atom
        -- a couple of hundred large GEMMs instead of ten million scattered
        rows.
        """
        dom = self.dom
        # a domain wider than the pair's own two atoms covers more than two
        # auxiliary blocks per pair, and the grouping below would miss all
        # but the first two; the sparse form is general and is what the
        # accuracy reference runs on
        if FOLD == 'csr' or self.domain != 'pair':
            m = p1 - p0
            csr = csr_matrix(
                (self.c[p0:p1].ravel(), dom.dom[blk].ravel(),
                 cp.arange(0, (m + 1) * dom.nlm, dom.nlm, dtype=np.int32)),
                shape=(m, self.naux))
            B -= fold * csr.dot(self.j2c)
            return
        b = blk.get()
        xsl = self.auxmol.aoslice_by_atom()
        # (which atom's block, at which offset inside the domain), for each
        # of the two atoms of every pair.  A diagonal atom pair has one.
        sides = [(dom.lo[b], dom.off_lo[b], np.ones(len(b), dtype=bool)),
                 (dom.hi[b], dom.off_hi[b], dom.lo[b] != dom.hi[b])]
        groups = []
        for atoms, offs, live in sides:
            key = np.where(live, atoms.astype(np.int64) * (dom.nlm + 1) + offs,
                           -1)
            order = np.argsort(key, kind='stable')
            ks, starts = np.unique(key[order], return_index=True)
            starts = np.append(starts, len(key))
            for j, k in enumerate(ks):
                if k >= 0:
                    groups.append((int(k), order[starts[j]:starts[j + 1]]))
        # one buffer for the product, not one per group: `DF_WORK_LIST.md`
        # §5.8 again -- the largest group's output is 269 MB at 113 atoms and
        # allocating it two hundred times is two hundred synchronisations
        buf = cp.empty(max(len(g) for _, g in groups) * self.naux)
        for k, rows in groups:
            A, off = int(k // (dom.nlm + 1)), int(k % (dom.nlm + 1))
            q0, q1 = int(xsl[A, 2]), int(xsl[A, 3])
            idx = cp.asarray(rows)
            # the scale goes onto the gathered coefficients, which are three
            # orders of magnitude smaller than the product
            cg = cp.take(self.c[:, off:off + q1 - q0], p0 + idx, axis=0)
            cg *= -fold
            out = ndarray((len(rows), self.naux), buffer=buf)
            cg.dot(self.j2c[q0:q1], out=out)
            scatter_add(B, idx, out)

    def _build_domain_blocks(self):
        """``cA[A][Q,i,s]`` -- the fit coefficients of every pair with an AO
        on atom ``A``, laid out so that the exchange pass's two intermediates
        are single GEMMs rather than permuting contractions.

        ``i`` runs over ``A``'s own AOs and ``s`` over every AO that pairs
        with one of them; ``part[A]`` is that list with ``A``'s own AOs first,
        which is what lets the ``lambda on A`` half of ``X`` overwrite the
        other half instead of being masked out of it.  Two index orders of the
        same numbers are kept, ``(Q,i,s)`` and ``(i,Q,s)``, because the two
        halves contract different indices of it and a three-index permutation
        is not something a GEMM can absorb -- a cuBLAS transpose is.
        """
        mol, auxmol = self.mol, self.auxmol
        a2a = _ao2atom(mol)
        sl, xsl = mol.aoslice_by_atom(), auxmol.aoslice_by_atom()
        rows, cols = self.rows.get(), self.cols.get()
        ar, ac = a2a[rows], a2a[cols]
        blk = self.dom.blk.get()
        nlm = self.dom.nlm
        c_flat = self.c.ravel()

        self.part, self.cA, self.cAt, self.aoA = [], [], [], []
        nlam = 0.0
        for A in range(mol.natm):
            i0, i1 = int(sl[A, 2]), int(sl[A, 3])
            q0, q1 = int(xsl[A, 2]), int(xsl[A, 3])
            naA, nqA = i1 - i0, q1 - q0
            m1, m2 = ar == A, ac == A
            pl = np.concatenate([np.where(m1)[0], np.where(m2)[0]])
            ii = np.concatenate([rows[m1], cols[m2]])
            ss = np.concatenate([cols[m1], rows[m2]])
            off = np.where(self.dom.lo[blk[pl]] == A, self.dom.off_lo[blk[pl]],
                           self.dom.off_hi[blk[pl]])
            gather = (pl[:, None] * nlm + off[:, None] +
                      np.arange(nqA)[None, :])
            vals = c_flat[cp.asarray(gather.ravel())].reshape(len(pl), nqA).T
            own = np.arange(i0, i1)
            part = np.concatenate([own, np.setdiff1d(np.unique(ss), own)])
            if self.lambda_tol:
                part = self._screen_lambda(part, ss, vals, i0, i1)
            pos = np.full(mol.nao, -1, dtype=np.int32)
            pos[part] = np.arange(len(part))
            keep = pos[ss] >= 0
            cA = cp.zeros((nqA, naA, len(part)))
            cA[:, (ii - i0)[keep], pos[ss][keep]] = vals[:, cp.asarray(keep)]
            self.part.append(cp.asarray(part, dtype=np.int32))
            self.aoA.append((i0, i1))
            self.cA.append(cA)
            self.cAt.append(cp.ascontiguousarray(cA.transpose(1, 0, 2)))
            nlam += nqA * len(part)
        STATS['nlambda'] = nlam / self.naux

    def _build_domain_blocks_wide(self):
        """The same intermediate for a domain wider than the pair's own two
        atoms, where the compact layout above does not hold.

        With a ``pair+r`` domain an auxiliary function on ``A`` is in the
        domain of pairs that have *neither* index on ``A``, so the first index
        of ``cA`` can no longer be ``A``'s own AOs -- both indices run over
        every AO the relevant pairs touch, and the array is ``nlambda`` square
        instead of ``nao_A`` by ``nlambda``.  That is a hundred times the
        memory at 284 atoms, which is why it is not the production path: wide
        domains are ``docs/README_dflocal.md`` §2.2's accuracy reference
        (1e-7 Eh per atom), measured on molecules where this fits, and PARI
        proper is what `_build_domain_blocks` is written for.
        """
        mol, auxmol = self.mol, self.auxmol
        sl, xsl = mol.aoslice_by_atom(), auxmol.aoslice_by_atom()
        rows, cols = self.rows.get(), self.cols.get()
        blk = self.dom.blk.get()
        dom = self.dom.dom.get()
        nlm = self.dom.nlm
        near = _domain_atoms(mol, self.domain)
        a2a = _ao2atom(mol)
        c_flat = self.c.ravel()
        # which auxiliary atoms each pair's domain covers
        cover = near[self.dom.lo] | near[self.dom.hi]      # (nblock, natm)

        self.part, self.cA, self.cAt, self.aoA = [], [], [], []
        nlam = 0.0
        for A in range(mol.natm):
            q0, q1 = int(xsl[A, 2]), int(xsl[A, 3])
            nqA = q1 - q0
            pl = np.where(cover[blk, A])[0]
            lam = np.concatenate([rows[pl], cols[pl]])
            sig = np.concatenate([cols[pl], rows[pl]])
            part = np.unique(np.concatenate([lam, sig]))
            pos = np.full(mol.nao, -1, dtype=np.int32)
            pos[part] = np.arange(len(part))
            off = np.array([np.searchsorted(dom[b], q0) for b in blk[pl]])
            gather = (pl[:, None] * nlm + off[:, None] +
                      np.arange(nqA)[None, :])
            vals = c_flat[cp.asarray(gather.ravel())].reshape(len(pl), nqA).T
            cA = cp.zeros((nqA, len(part), len(part)))
            cA[:, pos[lam], pos[sig]] = cp.concatenate([vals, vals], axis=1)
            self.part.append(cp.asarray(part, dtype=np.int32))
            self.aoA.append((int(sl[A, 2]), int(sl[A, 3])))
            self.cA.append(cA)
            self.cAt.append(None)
            nlam += nqA * len(part)
        STATS['nlambda'] = nlam / self.naux

    def _screen_lambda(self, part, ss, vals, i0, i1):
        """Keep only the AOs whose fitting coefficients on this atom are worth
        carrying.  The exchange pass's cost is linear in how many are left,
        and what dropping one costs is an error in the exchange matrix rather
        than in the fit -- so it is measured, in
        ``benchmarks/df_local_k_probe.py --lam``, and off by default."""
        mag = cp.abs(vals).max(axis=0).get()
        big = np.zeros(self.nao)
        np.maximum.at(big, ss, mag)
        keep = big > self.lambda_tol * big.max()
        keep[i0:i1] = True
        return part[keep[part]]

    # -- the build's own arithmetic, for a benchmark to time separately ---
    def reset(self, mol=None):
        self.M = self.c = None
        return super().reset(mol)

    # -- the two-electron build -------------------------------------------
    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=None, omega=None):
        if omega:
            raise NotImplementedError('dflocal: range-separated exchange')
        if getattr(self, 'fit_only', False):
            raise RuntimeError('dflocal: this object was built with '
                               'fit_only=True and holds no tensor')
        if self.M is None:
            self.build()
        log = logger.new_logger(self.mol, self.verbose)
        t0 = log.init_timer()
        STATS['calls'] += 1

        out_cupy = isinstance(dm, cp.ndarray)
        dms = cp.asarray(dm).reshape(-1, self.nao, self.nao)
        # the occupied factorisation, if the caller's density has one.  It is
        # what the `mo` contraction order needs and the only thing this module
        # takes from `df_jk`; `factorize_dm` returns `factor_r is not None`
        # only for the asymmetric densities a TDDFT gradient produces, and
        # those fall back to the AO order.
        occ_l = occ_r = None
        if with_k and self.kmode != 'ao':
            try:
                fl, fr = DFJK.factorize_dm(dm, hermi)
                occ_l = cp.asarray(fl).reshape(-1, self.nao, fl.shape[-1])
                occ_r = occ_l if fr is None else cp.asarray(fr).reshape(
                    -1, self.nao, fr.shape[-1])
            except Exception:                              # noqa: BLE001
                occ_l = occ_r = None
        vj = cp.zeros_like(dms) if with_j else None
        vk = cp.zeros_like(dms) if with_k else None
        for i in range(dms.shape[0]):
            j, k = self._jk_one(dms[i], hermi, with_j, with_k,
                                None if occ_l is None else occ_l[i],
                                None if occ_r is None else occ_r[i])
            if with_j:
                vj[i] = j
            if with_k:
                vk[i] = k
        vj = vj.reshape(dm.shape) if with_j else None
        vk = vk.reshape(dm.shape) if with_k else None
        if not out_cupy:
            vj = None if vj is None else vj.get()
            vk = None if vk is None else vk.get()
        log.timer_debug1('dflocal get_jk', *t0)
        return vj, vk

    def _use_mo(self, occ):
        """Which exchange contraction order, by flop count.

        ``ao``: one pass of ``2 nao^2 naux nlambda``.
        ``mo``: global DF's own first contraction, ``2 nao^2 naux nocc``, and
        then ``2 nao nocc naux nlambda`` in place of its second -- so it
        leaves the larger half alone and makes the smaller one ``nao/nlambda``
        times cheaper.  The two cross at ``nlambda = nocc/(1 - nocc/nao)``,
        which is 686 of 1004 at 284 atoms: the AO order is the asymptotic one
        (its ratio to global DF grows like N^0.3) and the MO order is the
        faster one at every size measured so far.
        """
        if occ is None or self.kmode == 'ao':
            return False
        if self.kmode == 'mo':
            return True
        nocc, nao, nlam = occ.shape[-1], self.nao, STATS['nlambda']
        return nocc * (nao + nlam) < nao * nlam

    def _jk_one(self, D, hermi, with_j, with_k, occ=None,
                occ_r=None):
        nao, naux = self.nao, self.naux
        rows, cols = self.rows, self.cols
        D = cp.asarray(D)
        if hermi == 0:
            D = (D + D.T) * .5

        # the density on the stored pair list, with GPU4PySCF's factor two on
        # the off-diagonal entries the packed form stands in for
        dpair = D[rows, cols] * 2
        dpair[self.diags] *= .5

        # d[Q] = sum_ls c[ls,Q] D[ls]  -- local, so a scatter over domains
        blk = self.dom.blk
        d = cp.zeros(naux)
        seg = _segment_sum(self.c * dpair[:, None], blk, self.dom.nblock)
        scatter_add(d, self.dom.dom.ravel(), seg.ravel())

        a = cp.zeros(naux) if with_j else None
        vj_pair = cp.zeros(len(rows)) if with_j else None
        KA = cp.zeros((nao, nao)) if with_k else None
        use_mo = with_k and self._use_mo(occ)
        STATS['kmode'] = 'mo' if use_mo else 'ao'

        # Every buffer a pass needs, allocated once.  `docs/DF_WORK_LIST.md`
        # §5.8: a temporary allocated inside a block loop does not cost its
        # own allocation time, it costs the overlap -- `cudaFree` is a
        # synchronisation point.  Measured here before and after: the unpack
        # was **57 %** of a 113-atom call when `fill_symmetric` allocated its
        # own output, against a total written volume that is 56.5 GB and
        # 40 ms of an A100's bandwidth.
        passes = list(self._passes())
        mq = max(q1 - q0 for q0, q1, _ in passes)
        mp = max(len(self.part[A]) for _, _, A in passes)
        work = cp.zeros((nao, nao, mq)) if with_k else None
        nrow = (occ.shape[-1] if use_mo else nao)
        rbuf = cp.empty(nao * nrow * mq) if use_mo else None
        xbuf = cp.empty(nrow * mq * mp) if with_k else None

        tick = _ticker(self)
        for q0, q1, A in passes:
            Mblk = self._block(q0, q1)              # (nQ, npair) on device
            tick('stream')
            if with_j:
                vj_pair += Mblk.T.dot(d[q0:q1])
                a[q0:q1] = Mblk.dot(dpair)
                tick('J')
            if with_k:
                nQ = q1 - q0
                dense = fill_symmetric(Mblk.T, self.pair_idx, nao, 0, nQ,
                                       out=work[:, :, :nQ])
                part = self.part[A]
                tick('unpack')
                if use_mo:
                    # sum_i rhok[m,i,Q] G[i,Q,l], with `rhok` exactly the
                    # intermediate `fastdf` builds and `G` the same local
                    # object as `X` with the occupied factor in the density's
                    # place
                    G = self._x_block(occ_r.T, A, q0, q1, xbuf)
                    tick('local')
                    rhok = ndarray((nao, nrow, nQ), buffer=rbuf)
                    contract('mnQ,ni->miQ', dense, occ, out=rhok)
                    tick('step1')
                    KA[:, part] += rhok.reshape(nao, -1).dot(
                        G.reshape(-1, G.shape[-1]))
                    tick('step2')
                else:
                    X = self._x_block(D, A, q0, q1, xbuf)
                    tick('local')
                    KA[:, part] += dense.reshape(nao, -1).dot(
                        X.reshape(-1, X.shape[-1]))
                    tick('step2')
            STATS['passes'] += 1

        vj = vk = None
        if with_j:
            ca = self.c * a[self.dom.dom[blk]]
            vj_pair += ca.sum(axis=1)
            vj = cp.zeros((nao, nao))
            vj[rows, cols] = vj_pair
            vj[cols, rows] = vj_pair
        if with_k:
            vk = KA + KA.T
        return vj, vk

    def _passes(self):
        """One pass per auxiliary atom, split if the unpacked block would be
        too large.

        Per *atom* and not per fixed block, because the exchange
        intermediate's second index is the set of AOs that overlap *that
        atom*: a pass spanning two atoms would carry the union of two such
        sets for no gain.  Splitting one atom's auxiliary functions across
        several passes costs nothing but a second unpack, and it is what
        keeps the dense ``(nao, nao, nQ)`` buffer -- 2.7 GB at 284 atoms and
        9.1 GB at 545 if a whole atom is taken at once -- inside the card.
        """
        xsl = self.auxmol.aoslice_by_atom()
        cap = BLK
        if not cap:
            budget = get_avail_mem(exclude_memory_pool=True) * .25
            cap = max(1, int(budget) // (8 * self.nao * self.nao))
        for A in range(self.mol.natm):
            q0, q1 = int(xsl[A, 2]), int(xsl[A, 3])
            for s0 in range(q0, q1, cap):
                yield s0, min(q1, s0 + cap), A

    def _block(self, q0, q1):
        m = self.M[q0:q1]
        return m if isinstance(m, cp.ndarray) else cp.asarray(m)

    def _x_block(self, D, A, q0, q1, buf=None):
        """``X[n,Q,l] = sum_s c[ls,Q] D[n,s]`` for ``Q`` on atom ``A``.

        ``c[ls,Q]`` is non-zero only when ``l`` or ``s`` sits on ``A``, which
        is the whole of PARI's locality and the reason this intermediate is
        ``nlambda`` wide instead of ``nao`` wide.  Two GEMMs, in the order
        that lets the second overwrite what the first wrote where they
        overlap:

            l not on A -- then s is, and ``c[ls,Q] = cA[Q,s,l]``
            l on A     -- s runs over every AO that pairs with A
        """
        i0, i1 = self.aoA[A]
        npart = len(self.part[A])
        naA, nqA = i1 - i0, q1 - q0
        k0 = q0 - int(self.auxmol.aoslice_by_atom()[A, 2])
        cA = self.cA[A][k0:k0 + nqA]
        if self.cAt[A] is None:                 # the wide-domain reference
            return D[:, self.part[A]].dot(
                cA.reshape(nqA * npart, npart).T).reshape(-1, nqA, npart)
        cAt = self.cAt[A][:, k0:k0 + nqA]
        if not cAt.flags.c_contiguous:
            cAt = cp.ascontiguousarray(cAt)
        X = ndarray((D.shape[0], nqA, npart), buffer=buf)
        D[:, i0:i1].dot(cAt.reshape(naA, nqA * npart),
                        out=X.reshape(-1, nqA * npart))
        X[:, :, :naA] = D[:, self.part[A]].dot(
            cA.reshape(nqA * naA, npart).T).reshape(-1, nqA, naA)
        return X


def _ticker(obj, prefix=''):
    """A stage timer that costs nothing unless it is asked for."""
    if not getattr(obj, 'profile', False):
        return lambda _name: None
    import time as _time
    last = [_time.perf_counter()]

    def tick(name):
        cp.cuda.Stream.null.synchronize()
        t = _time.perf_counter()
        STAGE[prefix + name] = STAGE.get(prefix + name, 0.0) + t - last[0]
        last[0] = t
    return tick


def _segment_sum(x, seg, nseg):
    """``out[s] = sum of the rows of x whose segment is s``."""
    out = cp.zeros((nseg, x.shape[1]))
    scatter_add(out, seg, x)
    return out


# --------------------------------------------------------------------------
# the opt-in entry point
# --------------------------------------------------------------------------
_WARNED = [False]


def _no_derivative(what):
    def raiser(*args, **kwargs):
        raise NotImplementedError(
            f'dflocal: no {what} for a pair-atomic fit.  The fit is not '
            'variational, so the energy carries terms the density-fitted '
            f'{what} drivers do not have (docs/DF_WORK_LIST.md W6).  They '
            'would silently differentiate the *global* fit of the same '
            'auxiliary basis instead, which is a different energy.')
    return raiser


def density_fit(mf, auxbasis=None, domain=None, store=None):
    """Put ``mf`` on a robust pair-atomic fit.

    This is **not** ``mf.density_fit()``: the object it returns computes a
    different number, by about 2.5e-5 Eh per atom against global density
    fitting (``docs/README_dflocal.md`` §2.2).  It says so once, on stdout,
    because a repository whose every other module agrees with upstream to
    1e-14 should not be able to produce a PARI energy by accident.
    """
    mf = mf.density_fit(auxbasis=auxbasis)
    mf.with_df = PARIDF(mf.mol, auxbasis, domain=domain, store=store)
    # W6: "a non-variational fit carries terms global DF does not, and
    # `fastdfgrad` assumes the global ones".  Nothing here supplies them, and
    # a density-fitted gradient driver would happily build its own global
    # cderi from the same auxiliary basis and hand back a derivative of a
    # different energy than the one this object converged.  So the two
    # drivers that would do that are closed rather than left to be found.
    mf.nuc_grad_method = _no_derivative('gradient')
    mf.Gradients = _no_derivative('gradient')
    mf.Hessian = _no_derivative('Hessian')
    if not _WARNED[0]:
        _WARNED[0] = True
        warnings.warn(
            'electrolites.dflocal: robust pair-atomic fitting is an '
            'approximation, not a faster route to the density-fitted energy. '
            'Measured error against global DF: 2.5e-5 Eh per atom with the '
            "'pair' domain, 1e-7 with 'pair+3.0' (docs/README_dflocal.md).")
    return mf
