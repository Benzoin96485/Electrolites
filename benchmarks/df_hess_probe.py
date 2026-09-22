"""Anatomy of ``df/hessian/rhf.py:_get_jk`` -- the DF Hessian's own CPHF build.

`docs/DF_WORK_LIST.md` **W8**.  That function is a **fifth** two-electron code
path: it builds J and K directly in the MO basis out of the same cderi, does
not go through ``df_jk.get_jk``, and so nothing in this repository is on it.
The entry names two things and measures neither -- the four cuTENSOR
contractions, and the cderi transfer at its top (``:1258-1273``) that moves a
device-resident tensor to the host "to leave space for storing Krylov subspace
vectors".

This is the measurement.  It reports three things.

**The anatomy** (``--whole``): what share of a density-fitted Hessian this
function is, measured at the method, with ``partial_hess_elec``, ``make_h1``
and the CPHF solve separated and the XC half of the CPHF separated from the
two-electron half.  Nothing can be ranked without it.

**The stage table**: a traced clone of ``_get_jk`` -- the same arithmetic in
the same order, checked element-wise against the real function on every run --
with a synchronising timer around each of the six stages inside its block
loop:

  * ``loop``        the block iterator, i.e. the unpack and (host-resident)
                    the PCIe stream;
  * ``rhok``        ``contract('Lpq,sqj->spjL', cderi, occ)``;
  * ``rhok_oo``     ``contract('spjL,spi->sijL', rhok, occ)``;
  * ``rhok1``       ``contract('Lpq,snqi->snpiL', cderi, mo1)`` -- the one
                    with an ``nao^2`` in it;
  * ``vk_a``        ``contract('snpiL,sjiL->nspj', rhok1, rhok_oo)``;
  * ``rhok1_oo``    ``contract('snpiL,spj->snjiL', rhok1, occ)``;
  * ``vk_b``        ``contract('spiL,snjiL->nspj', rhok, rhok1_oo)``;
  * ``vj``          the two Coulomb lines inside the exchange branch.

Each is reported against its own flop count and against the fp64 peak, so a
stage that is already at the machine is visible as such -- which is the whole
lesson of ``docs/DF_WORK_LIST.md`` §2.

**The alternatives**, priced at the shapes the function actually uses:
``rhok1_oo`` formed from ``rhok`` instead of from ``rhok1`` (the same flops
through a different operand, which is available because the cderi block is
symmetric in its two AO indices), the ``rhok1`` contraction as one cuBLAS
``dgemm`` over a ``(L,p)`` row index, and a sweep of the auxiliary block size
and of the density batch size that the function derives from free memory.

    python benchmarks/df_hess_probe.py \
        --xyz benchmarks/molecules/penicillinV.xyz --basis def2-svp --whole

``--natm-block`` caps the number of atoms a CPHF right-hand-side block holds,
which is what sets ``n_dm``; upstream derives it from free memory and it is
the single largest shape in the function.
"""
import json
import time

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--whole', action='store_true',
                help='also time a whole Hessian around it')
ap.add_argument('--cphf-only', action='store_true',
                help='with --whole, skip partial_hess_elec and make_h1')
ap.add_argument('--conv-tol-cpscf', type=float, default=1e-6)
ap.add_argument('--natm-block', type=int, default=0,
                help='atoms per CPHF right-hand-side block; 0 = all of them, '
                     'which is what a small molecule gets')
ap.add_argument('--blk', type=int, default=0,
                help='override the auxiliary block size the clone uses')
ap.add_argument('--dm-batches', default='',
                help='comma separated density batch sizes to sweep')
ap.add_argument('--blksizes', default='',
                help='comma separated auxiliary block sizes to sweep')
ap.add_argument('--cpu-cderi', action='store_true',
                help='force the host-resident branch, which is what upstream '
                     'does to itself whenever the tensor is more than 0.4 of '
                     'the free memory')
ap.add_argument('--no-check', action='store_true')
ap.add_argument('--no-alt', action='store_true')
ap.add_argument('--no-fused', action='store_true')
ap.add_argument('--repeat-calls', type=int, default=6,
                help='how many consecutive upstream calls to time; '
                     'a CPHF makes one per Krylov cycle')
ap.add_argument('--no-balance', action='store_true',
                help='leave the density batches uneven, as upstream does')
ap.add_argument('--roofline', action='store_true',
                help="time a plain dgemm at each stage's shape, which is the "
                     'only thing that says whether a stage is slow')
a = ap.parse_args()
apply_patch(a.patch)

import gpu4pyscf                                             # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from pyscf import lib                                        # noqa: E402
from gpu4pyscf.lib.cupy_helper import (                      # noqa: E402
    contract, ndarray, get_avail_mem, tag_array, transpose_sum, empty_mapped)
from gpu4pyscf.df.hessian import rhf as dfhess_rhf           # noqa: E402

_sync = cp.cuda.Stream.null.synchronize
#: A100 fp64 tensor-core peak, the number every other table in `docs/` uses.
PEAK = 19.5e12
GB = 1024.0 ** 3


def timeit(label, fn, reps=None):
    reps = a.reps if reps is None else reps
    out = fn()
    _sync()
    ts = []
    for _ in range(reps):
        _sync()
        t = time.perf_counter()
        out = fn()
        _sync()
        ts.append(time.perf_counter() - t)
    print(f'  {label:28s} {min(ts) * 1e3:10.2f} ms  (min of {reps})',
          flush=True)
    return min(ts), out


class Stages(dict):
    """Accumulating synchronised timers, one per stage of the block loop."""

    def __init__(self):
        super().__init__()
        self.order = []

    def __call__(self, label):
        if label not in self:
            self[label] = 0.0
            self.order.append(label)
        return _Stage(self, label)

    def report(self, flops, total):
        print(f'  {"stage":14s} {"time":>10s} {"share":>8s} '
              f'{"Gflop":>10s} {"TF/s":>8s} {"of peak":>8s}')
        for k in self.order:
            t = self[k]
            f = flops.get(k, 0.0)
            rate = f / t if t > 0 else 0.0
            print(f'  {k:14s} {t * 1e3:9.2f}ms {100 * t / total:7.1f}% '
                  f'{f / 1e9:10.1f} '
                  f'{rate / 1e12:8.2f} {100 * rate / PEAK:7.1f}%')
        acc = sum(self.values())
        print(f'  {"(sum)":14s} {acc * 1e3:9.2f}ms {100 * acc / total:7.1f}%')


class _Stage:
    def __init__(self, owner, label):
        self.owner, self.label = owner, label

    def __enter__(self):
        _sync()
        self.t = time.perf_counter()

    def __exit__(self, *exc):
        _sync()
        self.owner[self.label] += time.perf_counter() - self.t


# --------------------------------------------------------------------------
# the traced clone
# --------------------------------------------------------------------------
def clone_get_jk(dfobj, dms, mo_coeff, mo_occ, with_j=True, with_k=True,
                 blksize=None, dm_batch_size=None, st=None, flops=None,
                 alt_rhok1_oo=False):
    """``df/hessian/rhf.py:_get_jk``, single device, with timers.

    The multi-GPU wrapper and the build-and-move block at the top are dropped
    -- the first because everything in ``electrolites/`` asserts one device
    and the second because it runs at most once per process, is timed
    separately below, and holding it here would make the cderi's residency a
    property of call order rather than of the flag.
    """
    st = st if st is not None else Stages()
    flops = flops if flops is not None else {}
    mo1 = dms.mo1
    occ_coeff = dms.occ_coeff
    if mo_coeff.ndim == 2:
        mo_coeff = mo_coeff[None, :, :]
        mo1 = mo1[None]
        occ_coeff = occ_coeff[None, :]
    nspin, n_dm, nao, nocc = mo1.shape

    if not with_k:
        pair_addresses, cderi_diag = dfobj._cderi_idx
        dm_sparse = dms.reshape(nspin, n_dm, nao ** 2)[:, :, pair_addresses]
        dm_sparse = dm_sparse.sum(axis=0) if nspin == 2 else dm_sparse[0]
        dm_sparse = dm_sparse * 2
        dm_sparse[:, cderi_diag] *= .5

    vj = vk = None
    if with_k:
        _mo1 = cp.asarray(mo1)
        _occ_coeff = cp.asarray(occ_coeff)
        vk = cp.zeros((n_dm, nspin, nao, nocc))
        if with_j:
            vj = cp.zeros_like(vk)
    else:
        _dm_sparse = cp.asarray(dm_sparse)
        vj = cp.zeros_like(_dm_sparse)

    if blksize is None:
        blksize = dfobj.get_blksize(mem_fraction=0.2)
    if with_k:
        if dm_batch_size is None:
            mem_avail = get_avail_mem(exclude_memory_pool=True)
            dm_batch_size = int(mem_avail * 0.75 /
                                (nspin * blksize * (nao + nocc) * nocc * 8))
        dm_batch_size = max(1, min(dm_batch_size, n_dm))
        buf2 = cp.empty((dm_batch_size + 1, nspin, nao, nocc, blksize))
        buf3 = cp.empty((dm_batch_size + 1, nspin, nocc, nocc, blksize))
        buf, buf1 = buf2[-1], buf3[-1]

    nblk = 0
    for label in ('loop', 'rhok', 'rhok_oo', 'rhok1', 'vk_a', 'rhok1_oo',
                  'vk_b', 'vj'):
        st(label)
        flops.setdefault(label, 0.0)

    it = dfobj.loop(blksize=blksize, unpack=with_k)
    while True:
        with st('loop'):
            try:
                cderi, cderi_tril = next(it)
            except StopIteration:
                cderi = cderi_tril = None
        if cderi is None and cderi_tril is None:
            break
        nblk += 1
        if with_k:
            nL = len(cderi)
            rhok = ndarray((nspin, nao, nocc, nL), buffer=buf)
            rhok_oo = ndarray((nspin, nocc, nocc, nL), buffer=buf1)
            with st('rhok'):
                contract('Lpq,sqj->spjL', cderi, _occ_coeff, out=rhok)
            flops['rhok'] += 2 * nspin * nL * nao * nao * nocc
            with st('rhok_oo'):
                contract('spjL,spi->sijL', rhok, _occ_coeff, out=rhok_oo)
            flops['rhok_oo'] += 2 * nspin * nL * nao * nocc * nocc
            for i0, i1 in lib.prange(0, n_dm, dm_batch_size):
                nb = i1 - i0
                rhok1 = ndarray((nspin, nb, nao, nocc, nL), buffer=buf2)
                with st('rhok1'):
                    contract('Lpq,snqi->snpiL', cderi, _mo1[:, i0:i1],
                             out=rhok1)
                flops['rhok1'] += 2 * nspin * nb * nL * nao * nao * nocc
                with st('vk_a'):
                    contract('snpiL,sjiL->nspj', rhok1, rhok_oo, beta=1,
                             out=vk[i0:i1])
                flops['vk_a'] += 2 * nspin * nb * nL * nao * nocc * nocc
                rhok1_oo = ndarray((nspin, nb, nocc, nocc, nL), buffer=buf3)
                with st('rhok1_oo'):
                    if alt_rhok1_oo:
                        # the same tensor out of `rhok` rather than `rhok1`,
                        # which the symmetry of the cderi block in (p,q)
                        # allows: C_o^T M_L X = (M_L C_o)^T X
                        contract('sqjL,snqi->snjiL', rhok, _mo1[:, i0:i1],
                                 out=rhok1_oo)
                    else:
                        contract('snpiL,spj->snjiL', rhok1, _occ_coeff,
                                 out=rhok1_oo)
                flops['rhok1_oo'] += 2 * nspin * nb * nL * nao * nocc * nocc
                with st('vk_b'):
                    contract('spiL,snjiL->nspj', rhok, rhok1_oo, beta=1,
                             out=vk[i0:i1])
                flops['vk_b'] += 2 * nspin * nb * nL * nao * nocc * nocc
                if with_j:
                    with st('vj'):
                        rhoj1 = cp.einsum('sniiL->nL', rhok1_oo)
                        contract('spiL,nL->nspi', rhok, rhoj1, beta=1,
                                 out=vj[i0:i1])
                    flops['vj'] += 2 * nspin * nb * nL * nao * nocc
        else:
            with st('vj'):
                auxvec = contract('np,Lp->nL', _dm_sparse, cderi_tril)
                contract('nL,Lp->np', auxvec, cderi_tril, beta=1, out=vj)
            nL, npair = cderi_tril.shape
            flops['vj'] += 4 * n_dm * nL * npair

    return vj, vk, nblk, blksize, (dm_batch_size if with_k else 0)


def clone_get_jk_blas(dfobj, dms, mo_coeff, mo_occ, with_j=True,
                      with_k=True, blksize=None, dm_batch_size=None,
                      st=None, flops=None, balance=True):
    """The same arithmetic with the three ``nocc^2`` contractions as dgemms.

    What changes, and nothing else does: upstream's ``vk`` accumulator is
    ``(n_dm, nspin, nao, nocc)`` and **neither** of the two contractions that
    write into it produces that layout -- ``vk_a`` naturally comes out as
    ``[(n,p), j]`` and ``vk_b`` as ``[p, (n,j)]``, so one of the two is a
    permuting write whichever way the buffer is shaped.  Here they get one
    accumulator each, in the layout their own GEMM produces, and the two are
    added with a transpose of a 40 MB array at the end.  ``rhok1`` is left on
    cuTENSOR: its output has the auxiliary index innermost and no cuBLAS call
    can express that (the same finding as ``docs/README_fastdf.md`` §3.4).

    ``balance`` evens the density batches -- upstream's ``n_dm=126`` at
    ``dm_batch_size=55`` runs 55, 55 and then **16**, and the short pass is
    a third of the launches for an eighth of the work.  ``_balance_batch_size``
    is upstream's own helper, in the same file, and this function does not
    call it.
    """
    st = st if st is not None else Stages()
    flops = flops if flops is not None else {}
    mo1 = dms.mo1
    occ_coeff = dms.occ_coeff
    if mo_coeff.ndim == 2:
        mo_coeff = mo_coeff[None, :, :]
        mo1 = mo1[None]
        occ_coeff = occ_coeff[None, :]
    nspin, n_dm, nao, nocc = mo1.shape
    assert with_k

    _mo1 = cp.asarray(mo1)
    _occ_coeff = cp.asarray(occ_coeff)
    _occT = cp.ascontiguousarray(_occ_coeff.transpose(0, 2, 1))
    vkA = cp.zeros((nspin, n_dm, nao, nocc))
    vkB = cp.zeros((nspin, nao, n_dm, nocc))
    vj = cp.zeros((n_dm, nspin, nao, nocc)) if with_j else None

    if blksize is None:
        blksize = dfobj.get_blksize(mem_fraction=0.2)
    if dm_batch_size is None:
        mem_avail = get_avail_mem(exclude_memory_pool=True)
        dm_batch_size = int(mem_avail * 0.75 /
                            (nspin * blksize * (nao + nocc) * nocc * 8))
    dm_batch_size = max(1, min(dm_batch_size, n_dm))
    if balance:
        dm_batch_size = dfhess_rhf._balance_batch_size(dm_batch_size, n_dm)
    buf2 = cp.empty((dm_batch_size + 1, nspin, nao, nocc, blksize))
    buf3 = cp.empty((dm_batch_size + 1, nspin, nocc, nocc, blksize))
    buf, buf1 = buf2[-1], buf3[-1]

    for label in ('loop', 'rhok', 'rhok_oo', 'rhok1', 'vk_a', 'rhok1_oo',
                  'vk_b', 'vj', 'fold'):
        st(label)
        flops.setdefault(label, 0.0)

    nblk = 0
    it = dfobj.loop(blksize=blksize, unpack=True)
    while True:
        with st('loop'):
            try:
                cderi, _ = next(it)
            except StopIteration:
                cderi = None
        if cderi is None:
            break
        nblk += 1
        nL = len(cderi)
        rhok = ndarray((nspin, nao, nocc, nL), buffer=buf)
        rhok_oo = ndarray((nspin, nocc, nocc, nL), buffer=buf1)
        with st('rhok'):
            contract('Lpq,sqj->spjL', cderi, _occ_coeff, out=rhok)
        flops['rhok'] += 2 * nspin * nL * nao * nao * nocc
        with st('rhok_oo'):
            contract('spjL,spi->sijL', rhok, _occ_coeff, out=rhok_oo)
        flops['rhok_oo'] += 2 * nspin * nL * nao * nocc * nocc
        for i0, i1 in lib.prange(0, n_dm, dm_batch_size):
            nb = i1 - i0
            rhok1 = ndarray((nspin, nb, nao, nocc, nL), buffer=buf2)
            with st('rhok1'):
                contract('Lpq,snqi->snpiL', cderi, _mo1[:, i0:i1], out=rhok1)
            flops['rhok1'] += 2 * nspin * nb * nL * nao * nao * nocc
            rhok1_oo = ndarray((nspin, nb, nocc, nocc, nL), buffer=buf3)
            for s in range(nspin):
                r1 = rhok1[s].reshape(nb * nao, nocc * nL)
                roo = rhok_oo[s].reshape(nocc, nocc * nL)
                with st('vk_a'):
                    # out[(n,p), j] -- the layout the GEMM produces
                    vkA[s, i0:i1] += cp.dot(r1, roo.T).reshape(nb, nao, nocc)
                flops['vk_a'] += 2 * nb * nL * nao * nocc * nocc
                with st('rhok1_oo'):
                    cp.matmul(_occT[s], rhok1[s].reshape(nb, nao, nocc * nL),
                              out=rhok1_oo[s].reshape(nb, nocc, nocc * nL))
                flops['rhok1_oo'] += 2 * nb * nL * nao * nocc * nocc
                with st('vk_b'):
                    # out[p, (n,j)] -- likewise
                    vkB[s][:, i0:i1] += cp.dot(
                        rhok[s].reshape(nao, nocc * nL),
                        rhok1_oo[s].reshape(nb * nocc, nocc * nL).T
                    ).reshape(nao, nb, nocc)
                flops['vk_b'] += 2 * nb * nL * nao * nocc * nocc
            if with_j:
                with st('vj'):
                    rhoj1 = cp.einsum('sniiL->nL', rhok1_oo)
                    contract('spiL,nL->nspi', rhok, rhoj1, beta=1,
                             out=vj[i0:i1])
                flops['vj'] += 2 * nspin * nb * nL * nao * nocc

    with st('fold'):
        vk = (vkA + vkB.transpose(0, 2, 1, 3)).transpose(1, 0, 2, 3)
        vk = cp.ascontiguousarray(vk)
    return vj, vk, nblk, blksize, dm_batch_size


def clone_get_jk_fused(dfobj, dms, mo_coeff, mo_occ, with_j=True,
                       with_k=True, blksize=None, dm_batch_size=None,
                       st=None, flops=None, balance=True):
    """The same exchange, written as one pass over one big intermediate.

    Upstream forms ``rhok1[n,p,i,L] = (M_L X_n)[p,i]``, the only ``nao^2``
    object in the function, and then reads it **twice** -- once against
    ``rhok_oo`` and once to project it onto the occupied space.  Grouping the
    algebra by the perturbed density instead,

        vk[n] = sum_L M_L (D1_n M_L C_o),   D1_n = X_n C_o^T + C_o X_n^T

    so the bracket is ``G[n,q,j,L] = X_n B_L + C_o U`` -- two ``nao*nocc^2``
    contractions -- and the one ``nao^2 nocc`` contraction *consumes* it and
    writes straight into ``vk``.  The flop count is identical, term for term:
    ``nao^2 nocc + 3 nao nocc^2`` per auxiliary function and density either
    way.  What changes is that the 4.5 GB intermediate is written once and
    read once instead of written once and read twice, and that the expensive
    contraction's output is ``(n,p,j)`` -- 40 MB -- rather than a second
    4.5 GB array.
    """
    st = st if st is not None else Stages()
    flops = flops if flops is not None else {}
    mo1 = dms.mo1
    occ_coeff = dms.occ_coeff
    if mo_coeff.ndim == 2:
        mo_coeff = mo_coeff[None, :, :]
        mo1 = mo1[None]
        occ_coeff = occ_coeff[None, :]
    nspin, n_dm, nao, nocc = mo1.shape
    assert with_k

    _mo1 = cp.asarray(mo1)
    _occ_coeff = cp.asarray(occ_coeff)
    vk = cp.zeros((n_dm, nspin, nao, nocc))
    vj = cp.zeros_like(vk) if with_j else None

    if blksize is None:
        blksize = dfobj.get_blksize(mem_fraction=0.2)
    if dm_batch_size is None:
        mem_avail = get_avail_mem(exclude_memory_pool=True)
        dm_batch_size = int(mem_avail * 0.75 /
                            (nspin * blksize * (nao + nocc) * nocc * 8))
    dm_batch_size = max(1, min(dm_batch_size, n_dm))
    if balance:
        dm_batch_size = dfhess_rhf._balance_batch_size(dm_batch_size, n_dm)
    buf2 = cp.empty((dm_batch_size + 1, nspin, nao, nocc, blksize))
    buf3 = cp.empty((dm_batch_size + 1, nspin, nocc, nocc, blksize))
    buf, buf1 = buf2[-1], buf3[-1]

    for label in ('loop', 'rhok', 'rhok_oo', 'U', 'G', 'vk_fused', 'vj'):
        st(label)
        flops.setdefault(label, 0.0)

    nblk = 0
    it = dfobj.loop(blksize=blksize, unpack=True)
    while True:
        with st('loop'):
            try:
                cderi, _ = next(it)
            except StopIteration:
                cderi = None
        if cderi is None:
            break
        nblk += 1
        nL = len(cderi)
        rhok = ndarray((nspin, nao, nocc, nL), buffer=buf)
        rhok_oo = ndarray((nspin, nocc, nocc, nL), buffer=buf1)
        with st('rhok'):
            contract('Lpq,sqj->spjL', cderi, _occ_coeff, out=rhok)
        flops['rhok'] += 2 * nspin * nL * nao * nao * nocc
        with st('rhok_oo'):
            contract('spjL,spi->sijL', rhok, _occ_coeff, out=rhok_oo)
        flops['rhok_oo'] += 2 * nspin * nL * nao * nocc * nocc
        for i0, i1 in lib.prange(0, n_dm, dm_batch_size):
            nb = i1 - i0
            U = ndarray((nspin, nb, nocc, nocc, nL), buffer=buf3)
            with st('U'):
                # U[n,i,j,L] = sum_q mo1[n,q,i] rhok[q,j,L]
                contract('sqjL,snqi->snijL', rhok, _mo1[:, i0:i1], out=U)
            flops['U'] += 2 * nspin * nb * nL * nao * nocc * nocc
            G = ndarray((nspin, nb, nao, nocc, nL), buffer=buf2)
            with st('G'):
                contract('snpa,sajL->snpjL', _mo1[:, i0:i1], rhok_oo, out=G)
                contract('spi,snijL->snpjL', _occ_coeff, U, beta=1, out=G)
            flops['G'] += 4 * nspin * nb * nL * nao * nocc * nocc
            with st('vk_fused'):
                contract('Lpq,snqjL->nspj', cderi, G, beta=1, out=vk[i0:i1])
            flops['vk_fused'] += 2 * nspin * nb * nL * nao * nao * nocc
            if with_j:
                with st('vj'):
                    rhoj1 = cp.einsum('sniiL->nL', U)
                    contract('spiL,nL->nspi', rhok, rhoj1, beta=1,
                             out=vj[i0:i1])
                flops['vj'] += 2 * nspin * nb * nL * nao * nocc

    return vj, vk, nblk, blksize, dm_batch_size


def finish(dfobj, dms, mo_coeff, mo_occ, vj, vk, with_j, with_k):
    """The post-loop half of upstream's function, verbatim, so the clone's
    output is comparable with the real one's."""
    from gpu4pyscf.lib.cupy_helper import fill_symmetric
    mo1 = dms.mo1
    occ_coeff = dms.occ_coeff
    if mo_coeff.ndim == 2:
        mo_coeff = mo_coeff[None, :, :]
        mo1 = mo1[None]
        occ_coeff = occ_coeff[None, :]
    nspin, n_dm, nao, nocc = mo1.shape
    if with_k:
        vk = contract('nspi,spq->nsqi', vk, mo_coeff)
        if nspin == 1:
            vk *= 2
        if with_j:
            vj = contract('nspi,spq->nsqi', vj, mo_coeff)
            vj *= 2
            if nspin == 1:
                vj *= 2
    elif with_j:
        pair_addresses = dfobj._cderi_idx[0]
        vj = fill_symmetric(vj.T, pair_addresses, nao)
        vj = contract('pqn,sqi->nspi', vj, occ_coeff)
        vj = contract('nspi,spq->nsqi', vj, mo_coeff)
    if with_j:
        vj = vj.reshape(n_dm, -1)
    if with_k:
        vk = vk.reshape(n_dm, -1)
    return vj, vk


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------
mol = build_mol(a)
natm = mol.natm
mf = dft.RKS(mol, xc=a.xc).density_fit(auxbasis=a.auxbasis)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.conv_tol_cpscf = a.conv_tol_cpscf
mf.verbose = 0
t0 = time.perf_counter()
e = mf.kernel()
print(f'# {mol.natm} atoms  nao {mol.nao}  basis {a.basis}  xc {a.xc}')
print(f'# DF-SCF {time.perf_counter() - t0:.2f} s   E = {e:.10f}', flush=True)

mo_coeff = cp.asarray(mf.mo_coeff)
mo_occ = cp.asarray(mf.mo_occ)
nao, nmo = mo_coeff.shape
orbo = mo_coeff[:, mo_occ > 0]
nocc = orbo.shape[1]

hess = mf.Hessian()
hess.auxbasis_response = 2
dfobj = mf.with_df
if dfobj._cderi is None:
    dfobj.build()
cderi0 = dfobj._cderi[cp.cuda.Device().id]
resident = isinstance(cderi0, cp.ndarray)
naux, npair = cderi0.shape
print(f'# naux {naux}  stored pairs {npair}  cderi '
      f'{cderi0.nbytes / GB:.2f} GB  '
      f'{"device" if resident else "host"}-resident'
      f'   free {get_avail_mem() / GB:.1f} GB')

if a.cpu_cderi and resident:
    host = empty_mapped(cderi0.shape)
    host[:] = cp.asnumpy(cderi0)
    dfobj._cderi[cp.cuda.Device().id] = host
    del cderi0
    cp.get_default_memory_pool().free_all_blocks()
    resident = False
    print(f'# moved to host; free {get_avail_mem() / GB:.1f} GB')

# The right-hand sides a CPHF iteration actually sees: `natm_blk` atoms of
# three Cartesian directions each, shaped exactly as `hessian/rhf.py:gen_vind`
# shapes them.
natm_blk = a.natm_block or natm
n_dm = 3 * natm_blk
rng = np.random.default_rng(0)
mo1 = cp.asarray(rng.standard_normal((n_dm, nmo, nocc)) * 1e-2)
mo1_ao = contract('npo,ip->nio', mo1, mo_coeff)
dm1 = cp.asarray(contract('npi,qi->npq', mo1_ao, orbo * 2), order='C')
transpose_sum(dm1, inplace=True, hermi=1)
dm1 = tag_array(dm1, mo1=mo1_ao, occ_coeff=orbo, symmetrize=1)
print(f'# n_dm {n_dm} ({natm_blk} atoms)  nocc {nocc}  nmo {nmo}')

results = {'atoms': natm, 'nao': int(nao), 'nocc': int(nocc),
           'naux': int(naux), 'npair': int(npair), 'n_dm': int(n_dm),
           'cderi_gb': cderi0.nbytes / GB if resident else
                       dfobj._cderi[cp.cuda.Device().id].nbytes / GB,
           'resident': bool(resident)}

# --------------------------------------------------------------------------
# 1. the real function, and the clone against it
# --------------------------------------------------------------------------
print('\n## the call')
orig = getattr(dfhess_rhf, '_get_jk_orig', dfhess_rhf._get_jk)
t_jk, (vj_ref, vk_ref) = timeit(
    'upstream _get_jk (J+K)',
    lambda: orig(dfobj, dm1, mo_coeff, mo_occ, 1, True, True, None))
results['t_jk'] = t_jk

t_j, (vj_ref_j, _) = timeit(
    'upstream _get_jk (J only)',
    lambda: orig(dfobj, dm1, mo_coeff, mo_occ, 1, True, False, None))
results['t_j_only'] = t_j

st, flops = Stages(), {}
_sync()
t = time.perf_counter()
vj_c, vk_c, nblk, blksize, dmb = clone_get_jk(
    dfobj, dm1, mo_coeff, mo_occ, True, True,
    blksize=a.blk or None, st=st, flops=flops)
_sync()
t_clone = time.perf_counter() - t
vj_c, vk_c = finish(dfobj, dm1, mo_coeff, mo_occ, vj_c, vk_c, True, True)
print(f'  {"traced clone":28s} {t_clone * 1e3:10.2f} ms   '
      f'blksize {blksize}  blocks {nblk}  dm_batch {dmb}')
results.update(blksize=int(blksize), nblk=int(nblk), dm_batch=int(dmb),
               t_clone=t_clone)

if not a.no_check:
    dj = float(cp.abs(vj_c - vj_ref).max() / cp.abs(vj_ref).max())
    dk = float(cp.abs(vk_c - vk_ref).max() / cp.abs(vk_ref).max())
    print(f'  clone vs upstream: J {dj:.2e}   K {dk:.2e}')
    results.update(clone_err_j=dj, clone_err_k=dk)

print('\n## stages of the block loop')
st.report(flops, t_clone)
results['stages'] = {k: st[k] for k in st.order}
results['flops'] = dict(flops)
tot_flops = sum(flops.values())
print(f'  total {tot_flops / 1e12:.2f} Tflop  ->  '
      f'{tot_flops / t_clone / 1e12:.2f} TF/s over the whole call')
results['total_flops'] = tot_flops

# --------------------------------------------------------------------------
# 2. the alternatives
# --------------------------------------------------------------------------
if not a.no_alt:
    print('\n## alternatives')

    st2, fl2 = Stages(), {}
    _sync()
    t = time.perf_counter()
    vj_a, vk_a_, _, _, _ = clone_get_jk(
        dfobj, dm1, mo_coeff, mo_occ, True, True, blksize=a.blk or None,
        st=st2, flops=fl2, alt_rhok1_oo=True)
    _sync()
    t_alt = time.perf_counter() - t
    vj_a, vk_a_ = finish(dfobj, dm1, mo_coeff, mo_occ, vj_a, vk_a_, True, True)
    d = float(cp.abs(vk_a_ - vk_ref).max() / cp.abs(vk_ref).max())
    print(f'  rhok1_oo from rhok:  whole clone {t_alt * 1e3:9.2f} ms '
          f'({t_clone / t_alt:.2f}x)   K err {d:.2e}')
    print(f'    stage rhok1_oo  {st["rhok1_oo"] * 1e3:8.2f} ms -> '
          f'{st2["rhok1_oo"] * 1e3:8.2f} ms')
    results['alt_rhok1_oo'] = {'t': t_alt, 'err': d,
                               'stage': st2['rhok1_oo']}

    # the `rhok1` contraction as one dgemm over a (L,p) row index, priced on
    # one block at the shapes the loop uses
    it = dfobj.loop(blksize=blksize, unpack=True)
    cderi, _ = next(it)
    nL = len(cderi)
    del it
    cderi = cp.ascontiguousarray(cderi)
    nb = min(dmb, n_dm)
    x = cp.ascontiguousarray(
        cp.asarray(dm1.mo1[:nb]).transpose(1, 0, 2).reshape(nao, nb * nocc))
    out = cp.empty((nL * nao, nb * nocc))
    f_blk = 2 * nb * nL * nao * nao * nocc

    def _bigemm():
        # row-major out[(L,p), m] = sum_q cderi[(L,p), q] x[q, m]
        cp.dot(cderi.reshape(nL * nao, nao), x, out=out)
        return out

    t_big, _ = timeit('rhok1 as one dgemm', _bigemm)
    print(f'    {f_blk / t_big / 1e12:.2f} TF/s   against the loop stage at '
          f'{flops["rhok1"] / st["rhok1"] / 1e12:.2f} TF/s')
    results['alt_rhok1_dgemm'] = {'t': t_big, 'tflops': f_blk / t_big}

    del out, x, cderi
    cp.get_default_memory_pool().free_all_blocks()

# --------------------------------------------------------------------------
# 1b.  the same call, several times over -- which is what a CPHF does
# --------------------------------------------------------------------------
# `dm_batch_size` is derived from `get_avail_mem(exclude_memory_pool=True)`,
# i.e. from `cudaMemGetInfo`, which does **not** count the blocks CuPy's pool
# is holding but not using.  The first call of a process leaves 5-6 GB in the
# pool, so every later call sees less free memory than the first and picks a
# smaller batch -- and a CPHF makes twenty of these calls, not one.
print(f'\n## {a.repeat_calls} consecutive calls, as a CPHF makes them')
seq = []
for c in range(a.repeat_calls):
    free = cp.cuda.runtime.memGetInfo()[0]
    would = int(free * 0.75 / (blksize * (nao + nocc) * nocc * 8))
    _sync()
    t = time.perf_counter()
    orig(dfobj, dm1, mo_coeff, mo_occ, 1, True, True, None)
    _sync()
    dt = time.perf_counter() - t
    seq.append({'t': dt, 'free_gb': free / GB, 'dm_batch': min(would, n_dm)})
    print(f'  call {c + 1:2d}   {dt * 1e3:9.2f} ms   free '
          f'{free / GB:6.2f} GB   dm_batch it would pick '
          f'{min(would, n_dm):4d}')
results['repeat_calls'] = seq

print('\n## the same arithmetic with the vk accumulators laid out for cuBLAS')
st3, fl3 = Stages(), {}
cp.get_default_memory_pool().free_all_blocks()
_sync()
t = time.perf_counter()
vj_b, vk_b_, nblk_b, _, dmb_b = clone_get_jk_blas(
    dfobj, dm1, mo_coeff, mo_occ, True, True, blksize=a.blk or None,
    st=st3, flops=fl3, balance=not a.no_balance)
_sync()
t_blas = time.perf_counter() - t
vj_b, vk_b_ = finish(dfobj, dm1, mo_coeff, mo_occ, vj_b, vk_b_, True, True)
dkb = float(cp.abs(vk_b_ - vk_ref).max() / cp.abs(vk_ref).max())
djb = float(cp.abs(vj_b - vj_ref).max() / cp.abs(vj_ref).max())
print(f'  blas clone {t_blas * 1e3:9.2f} ms   {t_clone / t_blas:.2f}x the '
      f'traced clone, {t_jk / t_blas:.2f}x upstream   dm_batch {dmb_b}')
print(f'  vs upstream: J {djb:.2e}   K {dkb:.2e}')
st3.report(fl3, t_blas)
results['blas'] = {'t': t_blas, 'err_j': djb, 'err_k': dkb,
                   'dm_batch': int(dmb_b),
                   'stages': {k: st3[k] for k in st3.order}}

if not a.no_fused:
    print('\n## the same exchange as one pass over one intermediate')
    st4, fl4 = Stages(), {}
    cp.get_default_memory_pool().free_all_blocks()
    _sync()
    t = time.perf_counter()
    vj_f, vk_f, _, _, dmb_f = clone_get_jk_fused(
        dfobj, dm1, mo_coeff, mo_occ, True, True, blksize=a.blk or None,
        st=st4, flops=fl4, balance=not a.no_balance)
    _sync()
    t_fused = time.perf_counter() - t
    vj_f, vk_f = finish(dfobj, dm1, mo_coeff, mo_occ, vj_f, vk_f, True, True)
    dkf = float(cp.abs(vk_f - vk_ref).max() / cp.abs(vk_ref).max())
    djf = float(cp.abs(vj_f - vj_ref).max() / cp.abs(vj_ref).max())
    print(f'  fused clone {t_fused * 1e3:9.2f} ms   '
          f'{t_clone / t_fused:.2f}x the traced clone, '
          f'{t_jk / t_fused:.2f}x upstream   dm_batch {dmb_f}')
    print(f'  vs upstream: J {djf:.2e}   K {dkf:.2e}')
    st4.report(fl4, t_fused)
    results['fused'] = {'t': t_fused, 'err_j': djf, 'err_k': dkf,
                        'dm_batch': int(dmb_f),
                        'stages': {k: st4[k] for k in st4.order}}

if a.roofline:
    print("\n## the roofline: a plain dgemm at each stage's shape")
    nL = min(blksize, naux)
    nb = min(dmb, n_dm)
    shapes = {
        # out[(L,p),(n,i)] = cderi[(L,p),q] . x[q,(n,i)]
        'rhok1': (nL * nao, nb * nocc, nao),
        # out[(n,p),j] = rhok1[(n,p),(i,L)] . rhok_oo[j,(i,L)]^T
        'vk_a': (nb * nao, nocc, nocc * nL),
        # out[j,(i,L)] = occ[p,j]^T . rhok1[n][p,(i,L)], batched over n
        'rhok1_oo': (nocc, nocc * nL, nao),
        # out[p,(n,j)] = rhok[p,(i,L)] . rhok1_oo[(n,j),(i,L)]^T
        'vk_b': (nao, nb * nocc, nocc * nL),
    }
    results['roofline'] = {}
    for k, (m, n, kk) in shapes.items():
        need = (m * kk + kk * n + m * n) * 8 / GB
        if need > 0.5 * get_avail_mem() / GB:
            print(f'  {k:10s} {m}x{n}x{kk}  skipped ({need:.1f} GB)')
            continue
        A = cp.random.random((m, kk))
        B = cp.random.random((kk, n))
        C = cp.empty((m, n))
        f = 2.0 * m * n * kk
        rep = max(2, min(a.reps, int(2e12 // max(f, 1)) + 1))
        cp.dot(A, B, out=C)
        _sync()
        ts = []
        for _ in range(rep):
            _sync()
            t = time.perf_counter()
            cp.dot(A, B, out=C)
            _sync()
            ts.append(time.perf_counter() - t)
        dt = min(ts)
        loop_rate = (flops[k] / st[k]) if st.get(k) else 0.0
        print(f'  {k:10s} {m:7d}x{n:6d}x{kk:6d}  {dt * 1e3:8.2f} ms  '
              f'{f / dt / 1e12:6.2f} TF/s   loop stage '
              f'{loop_rate / 1e12:6.2f} TF/s   headroom '
              f'{(f / dt) / loop_rate if loop_rate else 0:.2f}x')
        results['roofline'][k] = {'m': m, 'n': n, 'k': kk,
                                  'tflops': f / dt,
                                  'loop_tflops': loop_rate}
        del A, B, C
        cp.get_default_memory_pool().free_all_blocks()

    # what a C-contiguous (L,p,q) block costs, which is what a cuBLAS
    # arrangement of `rhok1` needs and `DF.loop` does not hand out
    it = dfobj.loop(blksize=blksize, unpack=True)
    cderi, _ = next(it)
    del it
    t_tr, _ = timeit('ascontiguousarray(cderi)',
                     lambda: cp.ascontiguousarray(cderi))
    print(f'    {cderi.nbytes * 2 / t_tr / 1e9:.0f} GB/s, '
          f'{t_tr * naux / len(cderi) * 1e3:.1f} ms over the whole tensor')
    results['t_block_transpose'] = t_tr
    del cderi
    cp.get_default_memory_pool().free_all_blocks()

if a.dm_batches:
    print('\n## density batch size (upstream derives it from free memory)')
    for b in [int(x) for x in a.dm_batches.split(',')]:
        b = min(b, n_dm)
        try:
            cp.get_default_memory_pool().free_all_blocks()
            _sync()
            t = time.perf_counter()
            clone_get_jk(dfobj, dm1, mo_coeff, mo_occ, True, True,
                         blksize=a.blk or None, dm_batch_size=b,
                         st=Stages(), flops={})
            _sync()
            dt = time.perf_counter() - t
            passes = [min(b, n_dm - i) for i in range(0, n_dm, b)]
            print(f'  dm_batch {b:4d}  {dt * 1e3:9.2f} ms   passes {passes}')
            results.setdefault('dmsweep', {})[b] = dt
        except Exception as exc:                              # noqa: BLE001
            print(f'  dm_batch {b:4d}  ERR {type(exc).__name__}: {exc}')
        cp.get_default_memory_pool().free_all_blocks()

if a.blksizes:
    print('\n## auxiliary block size')
    for b in [int(x) for x in a.blksizes.split(',')]:
        try:
            _sync()
            t = time.perf_counter()
            clone_get_jk(dfobj, dm1, mo_coeff, mo_occ, True, True, blksize=b,
                         st=Stages(), flops={})
            _sync()
            dt = time.perf_counter() - t
            print(f'  blksize {b:5d}   {dt * 1e3:9.2f} ms')
            results.setdefault('blksweep', {})[b] = dt
        except Exception as exc:                              # noqa: BLE001
            print(f'  blksize {b:5d}   ERR {type(exc).__name__}: {exc}')
        cp.get_default_memory_pool().free_all_blocks()

# --------------------------------------------------------------------------
# 3. the whole Hessian around it
# --------------------------------------------------------------------------
if a.whole:
    print('\n## the whole Hessian, stage by stage')
    # Nothing in this repository has the *density-fitted* Hessian's anatomy:
    # `docs/HESSIAN_1.8.1.md` is the direct path, whose `partial_hess_elec`
    # and `make_h1` are different functions.  Nested rows (prefixed `..`)
    # are sub-intervals of their parent and overlap it.
    import importlib
    from collections import OrderedDict
    T, N = OrderedDict(), OrderedDict()
    _undo = []

    def wrap(mod_path, attr, label):
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            return
        owner, name = mod, attr
        if '.' in attr:
            cls, name = attr.split('.')
            owner = getattr(mod, cls, None)
            if owner is None:
                return
        orig = getattr(owner, name, None)
        if orig is None:
            return
        T.setdefault(label, 0.0)
        N.setdefault(label, 0)

        def timed(*args, __orig=orig, __label=label, **kw):
            _sync()
            t = time.perf_counter()
            out = __orig(*args, **kw)
            _sync()
            T[__label] += time.perf_counter() - t
            N[__label] += 1
            return out
        setattr(owner, name, timed)
        _undo.append((owner, name, orig))

    # the two big ones are bound as *class attributes* at class-creation
    # time (`partial_hess_elec = partial_hess_elec` in the class body), so a
    # patch of the module global never reaches them -- the same stale-alias
    # trap `tests/test_patch_aliases.py` exists for
    wrap('gpu4pyscf.df.hessian.rks', 'Hessian.partial_hess_elec',
         'partial_hess_elec')
    wrap('gpu4pyscf.df.hessian.rks', 'Hessian.make_h1', 'make_h1')
    wrap('gpu4pyscf.hessian.rhf', 'solve_mo1', 'solve_mo1 (CPHF)')
    wrap('gpu4pyscf.df.hessian.rhf', '_get_jk', '.. _get_jk (W8)')
    wrap('gpu4pyscf.hessian.rks', 'nr_rks_fxc_mo', '.. nr_rks_fxc_mo (XC)')
    wrap('gpu4pyscf.hessian.rks', 'nr_rks_fnlc_mo', '.. nr_rks_fnlc_mo')
    wrap('gpu4pyscf.df.df', '_cholesky_eri', 'cderi build')
    wrap('gpu4pyscf.hessian.rks', '_get_vxc_deriv1', '.. _get_vxc_deriv1')
    wrap('gpu4pyscf.hessian.rks', '_get_vxc_deriv2', '.. _get_vxc_deriv2')
    wrap('gpu4pyscf.df.hessian.rhf', '_int2c2e_ip2_per_atom', '.. int2c2e_ip2')
    wrap('gpu4pyscf.df.hessian.rhf', '_hcore_energy', '.. _hcore_energy')
    wrap('gpu4pyscf.df.hessian.rhf', '_jk_energy_per_atom', '.. ejk_per_atom')
    wrap('gpu4pyscf.df.hessian.rhf', '_get_veff', '.. _get_veff (h1)')
    wrap('gpu4pyscf.grad.rhf', 'get_grad_hcore', '.. get_grad_hcore')

    dfobj.reset()
    cp.get_default_memory_pool().free_all_blocks()
    _sync()
    t = time.perf_counter()
    h = hess.kernel()
    _sync()
    t_h = time.perf_counter() - t
    for owner, name, orig in _undo:
        setattr(owner, name, orig)
    print(f'  {"whole Hessian":26s} {t_h:8.2f} s')
    for k in T:
        print(f'  {k:26s} {T[k]:8.2f} s {100 * T[k] / t_h:6.1f} %'
              f'   {N[k]} calls')
    results.update(t_hessian=t_h,
                   hess_stages={k: T[k] for k in T},
                   hess_counts={k: N[k] for k in N},
                   hess_trace=float(np.trace(
                       np.asarray(h).reshape(3 * natm, 3 * natm))))

print('\n@@JSON ' + json.dumps(results))
