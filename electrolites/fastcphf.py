"""The Hessian's CPHF solve: one atom block, and a Krylov subspace that is a matrix.

``docs/DF_WORK_LIST.md`` **W44**.  ``hessian/{rhf,uhf}.py:solve_mo1`` is
94 % of a 113-atom UKS Hessian (`docs/README_fastdfhessip8.md` §3.1), and on
a 40 GB card GPU4PySCF solves it in two atom blocks where the full card
takes one: the same 339 right-hand sides in two Krylov iterations, each run
to its own tolerance, fifteen response calls instead of six and 1.25x the
time.  ``docs/README_fastcphf.md`` is the round.

Upstream sizes the atom block from what the *response* holds per
right-hand side -- AO-basis densities, ``vj`` and ``vk``, ``6 nao^2`` words
a set unrestricted, ``3 nao^2`` restricted -- as a fraction of the free
memory (0.3 and 0.5).  But the response is linear and every right-hand side
is independent, so that budget bounds how many sets one *response call*
may take, not how many one Krylov iteration may carry.  This module keeps
upstream's budget and moves it to where the AO arrays live:

1. **The response is called on groups of right-hand sides** that upstream's
   per-set budget allows at the memory free at that call, and the Krylov
   iteration carries every atom in **one block**, whose own arrays are
   MO-sized (``nmo nocc`` words a set).  The block is split only when those
   do not fit.  With one block the tolerance is the full card's
   (``conv_tol_cpscf`` times the block's atoms), so a 40 GB card solves the
   same equations to the same tolerance as an 80 GB one.
   ``FASTCPHF_BLOCK=upstream`` is upstream's blocks (and no groups);
   ``FASTCPHF_FXSETS=<n>`` caps a group (test-only).

2. **The Krylov subspace is two lists of matrices, not of vectors**
   (``krylov``).  Upstream keeps ``xs`` and ``ax`` as lists of rows and
   projects each new block against them one row at a time (a GEMV and an
   update of the whole block per row), stacks both lists into contiguous
   copies at the end, and forms the solution in a loop over rows again.
   Here each cycle's block is stored as it comes, the projection is two
   GEMMs per stored block (block modified Gram-Schmidt, in upstream's order),
   the subspace matrix ``h = xs ax^T`` is accumulated a block row and column
   per cycle, and the solution is one GEMM per block.  The same algorithm and
   the same arithmetic to round-off.  ``FASTCPHF_KRYLOV=upstream``.

3. **The subspace can live in page-locked host memory**: a block is kept on
   the device only when what is left still takes the next response call in
   one group; otherwise it goes to the host, which the projections read back
   once per cycle.  ``FASTCPHF_STORE=auto|device|host``.

4. **The response builds inside the solve screen at the solve's precision**
   (catalogue B12, ``docs/KERNEL_TARGETS_1.8.1.md`` §13.11).  The direct J
   and K builds screen a quartet against ``direct_scf_tol`` (1e-13), the
   SCF's precision; the CPHF is converged to ``conv_tol_cpscf``.  While
   ``solve_mo1`` runs, the builds screen against
   ``max(direct_scf_tol, min(1e-3 conv_tol_cpscf, 1e-9))`` instead -- 1e-9
   at a 1e-6 solve, 1e-11 at PySCF's default 1e-8.  The Krylov vectors are
   scaled to their residuals, so an absolute threshold is already a relative
   one that loosens as the solve converges; the factor keeps it three
   decades under the tolerance, and the cap at the loosest threshold
   measured safe (1e-7 at a 1e-6 solve breaks the iteration).
   ``FASTCPHF_SCREEN=off`` (or ``FASTCPHF_OFF=1``) keeps
   ``direct_scf_tol``.

Anything else -- a callback, a custom inner product, a matrix operator -- is
upstream's ``krylov``; ``FASTCPHF_OFF=1`` is upstream's ``solve_mo1``.
"""
import os

import numpy as np
import cupy as cp
import cupyx
from cupy import cublas

from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import cupy_helper
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem
from gpu4pyscf.hessian import rhf as HRHF
from gpu4pyscf.hessian import uhf as HUHF

_solve_mo1_r_orig = HRHF.solve_mo1
_solve_mo1_u_orig = HUHF.solve_mo1
#: where `tests/test_patch_aliases.py` looks for the functions this replaced
HRHF._solve_mo1_orig = _solve_mo1_r_orig
HUHF._solve_mo1_orig = _solve_mo1_u_orig
_krylov_orig = cupy_helper.krylov
_stable_qr = cupy_helper._stable_qr
DSOLVE_LINDEP = cupy_helper.DSOLVE_LINDEP
ALIGNED = HUHF.ALIGNED
GB = 1024 * 1024 * 1024

_S = {'solve_u': 0, 'solve_r': 0, 'declined': 0, 'blocks': 0,
      'fx_calls': 0, 'fx_groups': 0, 'krylov': 0, 'krylov_upstream': 0,
      'store_device': 0, 'store_host': 0, 'cycles': 0, 'screen': 0.}
#: item 4: the screening threshold of the response builds is this times
#: conv_tol_cpscf, but never looser than SCREEN_CAP
SCREEN_FACTOR = 1e-3
SCREEN_CAP = 1e-9


def _env(name, default):
    return os.environ.get(name, default)


def stats():
    """What this module did in this process."""
    return dict(_S)


# --------------------------------------------------------------------------
# the Krylov subspace
# --------------------------------------------------------------------------
class _Subspace:
    """The trial vectors ``xs`` and their images ``ax``, a block per cycle.

    ``h[i, j] = xs[i] . ax[j]`` is kept up to date as blocks arrive, so
    nothing stacks the lists at the end.  A block is stored on the device or,
    when ``keep(nbytes)`` says no, in page-locked host memory; ``_get`` hands
    back a device array either way (a host block through a staging buffer).
    """

    def __init__(self, ndim, keep):
        self.ndim = ndim
        self.keep = keep
        self.xs = []           # (array, on_device)
        self.ax = []
        self.p = []            # host float64, the rows' squared norms
        self.rows = []         # (r0, r1)
        self.nd = 0
        self.h = cp.zeros((0, 0))
        self._stage = [None, None]

    def _get(self, entry, slot):
        arr, dev = entry
        if dev:
            return arr
        n = arr.size
        st = self._stage[slot]
        if st is None or st.size < n:
            self._stage[slot] = None
            st = self._stage[slot] = cp.empty(max(n, self._maxsize()))
        out = st[:n].reshape(arr.shape)
        out.set(arr)
        return out

    def _maxsize(self):
        return max([a.size for a, _ in self.xs] + [0])

    def _store(self, a):
        if self.keep(a.nbytes):
            _S['store_device'] += 1
            return (a, True)
        _S['store_host'] += 1
        h = cupyx.empty_pinned(a.shape, dtype=a.dtype)
        a.get(out=h)
        return (h, False)

    def append_project(self, x, axt, p, x1):
        """Store ``(x, axt)`` and project ``x1`` against every stored block,
        this one last -- upstream's loop over rows, a block at a time."""
        nk = x.shape[0]
        nd0, nd1 = self.nd, self.nd + nk
        h = cp.zeros((nd1, nd1))
        h[:nd0, :nd0] = self.h
        for j, (r0, r1) in enumerate(self.rows):
            xj = self._get(self.xs[j], 0)
            h[r0:r1, nd0:nd1] = xj.dot(axt.T)
            _project(x1, xj, self.p[j])
            axj = self._get(self.ax[j], 1)
            h[nd0:nd1, r0:r1] = x.dot(axj.T)
        h[nd0:nd1, nd0:nd1] = x.dot(axt.T)
        _project(x1, x, p)
        self.h = h
        self.xs.append(self._store(x))
        self.ax.append(self._store(axt))
        self.p.append(np.asarray(p, dtype=np.float64))
        self.rows.append((nd0, nd1))
        self.nd = nd1
        return x1

    def rows_dot(self, b, m):
        """``xs[:m] . b^T``."""
        g = cp.zeros((self.nd, b.shape[0]), dtype=b.dtype)
        for j, (r0, r1) in enumerate(self.rows):
            if r0 >= m:
                break
            e = min(r1, m)
            xj = self._get(self.xs[j], 0)
            g[r0:e] = xj[:e - r0].dot(b.T)
        return g

    def combine(self, c):
        """``sum_i c[i] xs[i]``, a GEMM per block."""
        x = cp.zeros((c.shape[1], self.ndim), dtype=c.dtype)
        for j, (r0, r1) in enumerate(self.rows):
            xj = self._get(self.xs[j], 0)
            cublas.gemm('T', 'N', c[r0:r1], xj, out=x, beta=1.)
        return x

    def innerprod(self):
        return np.concatenate(self.p) if self.p else np.zeros(0)

    def free(self):
        self.xs = self.ax = self.p = self.rows = None
        self._stage = [None, None]
        self.h = None


def _project(x1, xj, pj):
    """``x1 -= (x1 xj^T / pj) xj`` -- the rows of ``xj`` are orthogonal, so
    this is upstream's row-by-row loop over them as two GEMMs;
    it runs twice (classical Gram-Schmidt twice, which takes the difference from upstream's order from 3.5e-11 to 9e-12 on glycine+); ``FASTCPHF_REORTH=0`` once."""
    pj = cp.asarray(pj)
    for _ in range(2 if _env('FASTCPHF_REORTH', '1') != '0' else 1):
        w = x1.dot(xj.T)
        w /= pj
        # in place: `x1 -= w.dot(xj)` would allocate an x1-sized temporary
        # (over GPU4PySCF's 100 MB pool threshold, so a cudaMalloc and a
        # cudaFree per block, docs/DF_WORK_LIST.md 5.8)
        cublas.gemm('N', 'N', w, xj, out=x1, alpha=-1., beta=1.)


def _keep_all(nbytes):
    return True


def krylov(aop, b, x0=None, tol=1e-10, max_cycle=30, dot=cp.dot,
           lindep=DSOLVE_LINDEP, callback=None, hermi=False,
           verbose=logger.WARN, keep=None):
    r"""``lib/cupy_helper.py:krylov`` -- Pople's solver for ``(1+a) x = b`` --
    with the subspace as blocks (item 2) that may live on the host (item 3).

    ``keep(nbytes)`` says whether a block of that size may stay on the
    device; ``None`` keeps everything there.  A matrix operator, a callback
    or another inner product is upstream's function.
    """
    if (isinstance(aop, cp.ndarray) or callback is not None
            or dot is not cp.dot
            or _env('FASTCPHF_KRYLOV', 'blocks') == 'upstream'):
        _S['krylov_upstream'] += 1
        return _krylov_orig(aop, b, x0, tol, max_cycle, dot, lindep, callback,
                            hermi, verbose)
    _S['krylov'] += 1
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        import sys
        log = logger.Logger(sys.stdout, verbose)

    if not (isinstance(b, cp.ndarray) and b.ndim == 1):
        b = cp.asarray(b)
    b_ndim = b.ndim
    if x0 is None:
        x1 = b
    else:
        b = b - (x0 + aop(x0))
        x1 = b
    if x1.ndim == 1:
        x1 = x1.reshape(1, x1.size)
    b2 = b.reshape(-1, b.shape[-1])
    nroots, ndim = x1.shape
    x1, rmat = _stable_qr(x1, cp.dot, lindep=lindep)
    if len(x1) == 0:
        return cp.zeros_like(b)
    d = rmat.diagonal()
    x1 *= d[:, None]
    p = cp.asnumpy(d.real) ** 2
    max_innerprod = p.max()
    if max_innerprod < lindep or max_innerprod < tol**2:
        if x0 is None:
            return cp.zeros_like(b)
        return x0

    store = _env('FASTCPHF_STORE', 'auto')
    if store == 'device' or keep is None:
        keep = _keep_all
    elif store == 'host':
        keep = lambda nbytes: False                        # noqa: E731
    sub = _Subspace(ndim, keep)
    max_cycle = min(max_cycle, ndim)
    for cycle in range(max_cycle):
        axt = aop(x1)
        if axt.ndim == 1:
            axt = axt.reshape(1, ndim)
        xn = axt.copy()
        xn = sub.append_project(x1, axt, p, xn)
        x1 = axt = None
        x1, rmat = _stable_qr(xn, cp.dot, lindep=lindep)
        xn = None
        d = rmat.diagonal()
        x1 *= d[:, None]
        p1 = cp.asnumpy(d.real) ** 2
        max_innerprod = p1.max() if p1.size else 0.
        _S['cycles'] += 1

        log.info(f'krylov cycle {cycle}, r = {max_innerprod**.5:.3e}, '
                 f'{x1.shape[0]} equations')
        if max_innerprod < lindep or max_innerprod < tol**2:
            break
        mask = (p1 > lindep) & (p1 > tol**2)
        x1 = x1[cp.asarray(mask)]
        p = p1[mask]
        if max_innerprod > 1e10:
            raise RuntimeError('Krylov subspace iterations diverge')
    else:
        raise RuntimeError('Krylov solver failed to converge')
    x1 = None

    nd = sub.nd
    log.info(f'krylov space size {nd}')
    h = sub.h
    # Add the contribution of I in (1+a)
    h += cp.diag(cp.asarray(sub.innerprod()[:nd]))
    if b_ndim == 1:
        g = cp.zeros((nd, 1), dtype=h.dtype)
        g[0] = float(sub.p[0][0])
    else:
        # Restore the first nroots vectors, which are array b or b-(1+a)x0
        g = sub.rows_dot(b2, min(nd, nroots))
    c = cp.linalg.solve(h, g)
    x = sub.combine(c)
    sub.free()
    if b_ndim == 1:
        x = x[0]
    if x0 is not None:
        x += x0
    return x


# --------------------------------------------------------------------------
# the response in groups of right-hand sides
# --------------------------------------------------------------------------
class _Budget:
    """Upstream's per-set budget for the response, applied per call.

    ``frac`` of the free memory over ``per_set`` bytes is how many sets one
    response call may take -- the rule upstream applies to the atom block,
    with the same numbers (0.3 and ``6 nao^2`` words unrestricted, 0.5 and
    ``3 nao^2`` restricted).
    """

    def __init__(self, per_set, frac):
        self.per_set = per_set
        self.frac = frac
        self.cap = int(_env('FASTCPHF_FXSETS', '0'))

    def sets(self, avail=None):
        if avail is None:
            avail = get_avail_mem()
        n = int(self.frac * max(avail, 0) / self.per_set)
        if self.cap > 0:
            n = min(n, self.cap)
        return max(1, n)


def _grouped(fx, budget, ndim):
    """``fx`` on groups of rows that ``budget`` allows (item 1).  The rows
    are independent right-hand sides and the response is linear, so the
    result is the same matrix."""
    def fxg(mo1):
        mo1 = cp.asarray(mo1)
        m = mo1.reshape(-1, ndim)
        nset = m.shape[0]
        _S['fx_calls'] += 1
        g = budget.sets()
        if g >= nset:
            _S['fx_groups'] += 1
            return fx(mo1)
        ngroup = -(-nset // g)
        g = -(-nset // ngroup)
        out = cp.empty((nset, ndim))
        for i0, i1 in lib.prange(0, nset, g):
            _S['fx_groups'] += 1
            out[i0:i1] = cp.asarray(fx(m[i0:i1])).reshape(i1 - i0, ndim)
        return out
    return fxg


def _block_atoms(natm, mo_words_per_atom, avail, frac=.5):
    """Atoms per Krylov block: all of them when the block's own MO-sized
    arrays take less than ``frac`` of the free memory, otherwise upstream's
    granularity of what does."""
    blk = int(frac * avail / (mo_words_per_atom * 8))
    if blk >= natm:
        return natm
    if blk < ALIGNED**2:
        raise RuntimeError('GPU memory insufficient for solving CPHF equations')
    return (blk // ALIGNED**2) * ALIGNED**2


def _keeper(budget, nsets_next):
    """Item 3: a subspace block stays on the device while what is left
    still takes the next response call's ``nsets_next[0]`` sets in one
    group."""
    def keep(nbytes):
        return budget.sets(get_avail_mem() - nbytes) >= nsets_next[0]
    return keep


# --------------------------------------------------------------------------
# solve_mo1
# --------------------------------------------------------------------------
def _screen_tol(mf):
    """Item 4's threshold for the response builds of this solve."""
    base = getattr(mf, 'direct_scf_tol', 1e-13)
    if (_env('FASTCPHF_SCREEN', 'on') == 'off'
            or _env('FASTCPHF_OFF', '') == '1'):
        return base
    tol = getattr(mf, 'conv_tol_cpscf', 1e-8)
    return max(base, min(SCREEN_FACTOR * tol, SCREEN_CAP))


def _screened(solve):
    """Run ``solve`` with every direct J/K build screening at
    `_screen_tol`: the options objects the response reaches (one per
    ``omega``, built lazily by ``_get_jk_mo``) are GPU4PySCF's ``_VHFOpt``
    of ``scf/jk.py`` and ``scf/j_engine.py``, and each reads its
    ``direct_scf_tol`` at call time, so the methods are wrapped for the
    duration of the solve and restored after it."""
    import functools
    from gpu4pyscf.scf import jk as JK, j_engine as JE

    def call_with(fn, tol):
        @functools.wraps(fn)
        def wrapped(self, *x, **k):
            keep = self.direct_scf_tol
            self.direct_scf_tol = max(keep, tol)
            try:
                return fn(self, *x, **k)
            finally:
                self.direct_scf_tol = keep
        return wrapped

    @functools.wraps(solve)
    def run(mf, *x, **k):
        tol = _screen_tol(mf)
        if tol <= getattr(mf, 'direct_scf_tol', 1e-13):
            return solve(mf, *x, **k)
        _S['screen'] = tol
        saved = [(JK._VHFOpt, 'get_k'), (JK._VHFOpt, 'get_jk'),
                 (JE._VHFOpt, 'get_j')]
        saved = [(c, n, c.__dict__[n]) for c, n in saved if n in c.__dict__]
        for c, n, fn in saved:
            setattr(c, n, call_with(fn, tol))
        try:
            return solve(mf, *x, **k)
        finally:
            for c, n, fn in saved:
                setattr(c, n, fn)
    return run


def _use_upstream(mf):
    return (_env('FASTCPHF_OFF', '') == '1'
            or _env('FASTCPHF_BLOCK', 'auto') == 'upstream')


def solve_mo1_u(mf, mo_energy, mo_coeff, mo_occ, h1mo,
                fx=None, atmlst=None, max_memory=4000, verbose=None,
                max_cycle=50, level_shift=0):
    """``hessian/uhf.py:solve_mo1`` with items 1-3.  Upstream's arithmetic
    line for line otherwise."""
    if _use_upstream(mf):
        _S['declined'] += 1
        return _solve_mo1_u_orig(mf, mo_energy, mo_coeff, mo_occ, h1mo, fx,
                                 atmlst, max_memory, verbose, max_cycle,
                                 level_shift)
    _S['solve_u'] += 1
    from gpu4pyscf.pbc.gto import int1e
    mol = mf.mol
    log = logger.new_logger(mf, verbose)
    t0 = log.init_timer()

    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = mo_occ[0] == 0
    viridxb = mo_occ[1] == 0
    mo_ea, mo_eb = mo_energy
    ei_a = mo_ea[occidxa]
    ei_b = mo_eb[occidxb]
    ea_a = mo_ea[viridxa]
    ea_b = mo_eb[viridxb]
    eai_a = 1 / (ea_a[:, None] + level_shift - ei_a)
    eai_b = 1 / (ea_b[:, None] + level_shift - ei_b)
    nvira, nocca = eai_a.shape
    nvirb, noccb = eai_b.shape
    nocc = nocca + noccb

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:, occidxa]
    moccb = mo_coeff[1][:, occidxb]
    natm = mol.natm

    if fx is None:
        fx = HUHF.gen_vind(mf, mo_coeff, mo_occ)
    ndim = nmo * nocc
    # item 1: upstream's per-set budget ("*8 for spin-up/down input dm, vj,
    # vk, and vxc": 0.3 of the free memory over 6 nao^2 words a set) per
    # response call
    budget = _Budget(8 * nao * nao * 6, .3)
    fxg = _grouped(fx, budget, ndim)
    nsets_next = [0]

    def fvind_vo(mo1):
        mo1 = mo1.reshape(-1, nmo*nocc)
        v = fxg(mo1).reshape(-1, nmo*nocc)
        if level_shift != 0:
            v -= mo1 * level_shift
        v1a = v[:, :nmo*nocca].reshape(-1, nmo, nocca)
        v1b = v[:, nmo*nocca:].reshape(-1, nmo, noccb)
        v1a[:, viridxa] *= eai_a
        v1b[:, viridxb] *= eai_b
        v1a[:, occidxa] = 0
        v1b[:, occidxb] = 0
        # the next call's sets are at most these (converged rows drop out)
        nsets_next[0] = v.shape[0]
        return v.reshape(-1, nmo*nocc)

    ipovlp = -int1e.int1e_ipovlp(mol)
    cp.get_default_memory_pool().free_all_blocks()

    avail_mem = get_avail_mem()
    # the block's own arrays, MO-sized: h1 and s1 (both spins), mo1 and
    # its stacked copy, the solver's x1, its image and the QR's two copies
    # of it, and the result -- about ten nmo nocc words a set, three sets
    # an atom
    blksize = _block_atoms(natm, 3 * 10 * ndim, avail_mem)
    log.debug(f'GPU memory {avail_mem/GB:.1f} GB available')
    log.debug(f'{blksize} atoms in each block CPHF equation')

    h1moa, h1mob = h1mo
    mo1sa = np.zeros(h1moa.shape)
    mo1sb = np.zeros(h1mob.shape)
    e1sa = np.zeros((natm, 3, nocca, nocca))
    e1sb = np.zeros((natm, 3, noccb, noccb))
    aoslices = mol.aoslice_by_atom()
    for i0, i1 in lib.prange(0, natm, blksize):
        log.info('Solving CPHF equation for atoms [%d:%d]', i0, i1)
        _S['blocks'] += 1

        h1a_blk = h1moa[i0:i1]
        h1b_blk = h1mob[i0:i1]
        if not isinstance(h1moa, cp.ndarray):
            h1a_blk = cp.asarray(h1a_blk)
            h1b_blk = cp.asarray(h1b_blk)
        s1a_blk = cp.empty_like(h1a_blk)
        s1b_blk = cp.empty_like(h1b_blk)
        for k, (p0, p1) in enumerate(aoslices[i0:i1, 2:]):
            s1ao = cp.zeros((3, nao, nao))
            s1ao[:, p0:p1] += ipovlp[:, p0:p1]
            s1ao[:, :, p0:p1] += ipovlp[:, p0:p1].transpose(0, 2, 1)
            tmp = contract('xij,jo->xio', s1ao, mocca)
            s1a_blk[k] = contract('xio,ip->xpo', tmp, mo_coeff[0])
            tmp = contract('xij,jo->xio', s1ao, moccb)
            s1b_blk[k] = contract('xio,ip->xpo', tmp, mo_coeff[1])

        mo1a = hs_a = h1a_blk - s1a_blk * ei_a
        mo1b = hs_b = h1b_blk - s1b_blk * ei_b
        mo_e1a = hs_a[:, :, occidxa]
        mo_e1b = hs_b[:, :, occidxb]
        mo1a[:, :, viridxa] *= -eai_a
        mo1b[:, :, viridxb] *= -eai_b
        mo1a[:, :, occidxa] = -s1a_blk[:, :, occidxa] * .5
        mo1b[:, :, occidxb] = -s1b_blk[:, :, occidxb] * .5
        nset = (i1 - i0) * 3
        mo1 = cp.hstack((mo1a.reshape(nset, -1), mo1b.reshape(nset, -1)))
        hs_a = hs_b = h1a_blk = h1b_blk = s1a_blk = s1b_blk = None
        s1ao = tmp = None

        tol = mf.conv_tol_cpscf * (i1 - i0)
        nsets_next[0] = nset
        raw_mo1 = krylov(fvind_vo, mo1.reshape(-1, nmo*nocc),
                         tol=tol, max_cycle=max_cycle, verbose=log,
                         keep=_keeper(budget, nsets_next))
        raw_mo1a = mo1[:, :nmo*nocca].reshape(i1-i0, 3, nmo, nocca)
        raw_mo1b = mo1[:, nmo*nocca:].reshape(i1-i0, 3, nmo, noccb)

        # The occ-occ block of mo1 is non-canonical
        raw_mo1a[:, :, occidxa] = mo1a[:, :, occidxa]
        raw_mo1b[:, :, occidxb] = mo1b[:, :, occidxb]

        v1 = fxg(raw_mo1)
        v1a = v1[:, :nmo*nocca].reshape(i1-i0, 3, nmo, nocca)
        v1b = v1[:, nmo*nocca:].reshape(i1-i0, 3, nmo, noccb)
        mo1a[:, :, viridxa] -= v1a[:, :, viridxa] * eai_a
        mo1b[:, :, viridxb] -= v1b[:, :, viridxb] * eai_b
        mo_e1a += v1a[:, :, occidxa]
        mo_e1b += v1b[:, :, occidxb]
        mo_e1a += mo1a[:, :, occidxa] * (ei_a[:, None] - ei_a)
        mo_e1b += mo1b[:, :, occidxb] * (ei_b[:, None] - ei_b)

        mo1sa[i0:i1] = mo1a.get()
        mo1sb[i0:i1] = mo1b.get()
        e1sa[i0:i1] = mo_e1a.get()
        e1sb[i0:i1] = mo_e1b.get()
        mo1a = mo1b = mo1 = mo_e1a = mo_e1b = None
        raw_mo1a = raw_mo1b = raw_mo1 = None
        v1a = v1b = v1 = None
    log.timer('CPHF solver', *t0)
    return (mo1sa, mo1sb), (e1sa, e1sb)


def solve_mo1_r(mf, mo_energy, mo_coeff, mo_occ, h1mo,
                fx=None, atmlst=None, max_memory=4000, verbose=None,
                max_cycle=50, level_shift=0):
    """``hessian/rhf.py:solve_mo1`` with items 1-3."""
    if _use_upstream(mf):
        _S['declined'] += 1
        return _solve_mo1_r_orig(mf, mo_energy, mo_coeff, mo_occ, h1mo, fx,
                                 atmlst, max_memory, verbose, max_cycle,
                                 level_shift)
    _S['solve_r'] += 1
    from gpu4pyscf.pbc.gto import int1e
    mol = mf.mol
    log = logger.new_logger(mf, verbose)
    t0 = log.init_timer()

    occidx = mo_occ > 0
    viridx = mo_occ == 0
    e_a = mo_energy[viridx]
    e_i = mo_energy[occidx]
    e_ai = 1 / (e_a[:, None] + level_shift - e_i)
    nvir, nocc = e_ai.shape

    if cp.any(cp.isinf(e_ai)) or cp.any(cp.isnan(e_ai)):
        raise ValueError(f"e_ai = {e_ai} contains inf or nan, likely because "
                         "HOMO-LUMO gap is zero.")

    mocc = mo_coeff[:, occidx]
    nao, nmo = mo_coeff.shape
    natm = mol.natm

    if fx is None:
        fx = HRHF.gen_vind(mf, mo_coeff, mo_occ)
    ndim = nmo * nocc
    # item 1: upstream's "vj, vk, dm in AO" budget, 0.5 of the free memory
    # over 3 nao^2 words a set, per response call
    budget = _Budget(8 * nao * nao * 3, .5)
    fxg = _grouped(fx, budget, ndim)
    nsets_next = [0]

    def fvind_vo(mo1):
        mo1 = mo1.reshape(-1, nmo, nocc)
        v = fxg(mo1.reshape(-1, ndim)).reshape(-1, nmo, nocc)
        if level_shift != 0:
            v -= mo1 * level_shift
        v[:, viridx, :] *= e_ai
        v[:, occidx, :] = 0
        nsets_next[0] = v.shape[0]
        return v.reshape(-1, nmo*nocc)

    ipovlp = -int1e.int1e_ipovlp(mol)
    cp.get_default_memory_pool().free_all_blocks()

    avail_mem = get_avail_mem()
    blksize = _block_atoms(natm, 3 * 10 * ndim, avail_mem)
    log.debug(f'GPU memory {avail_mem/GB:.1f} GB available')
    log.debug(f'{blksize} atoms in each block CPHF equation')

    mo1s = np.zeros(h1mo.shape)
    e1s = np.zeros((natm, 3, nocc, nocc))
    aoslices = mol.aoslice_by_atom()
    for i0, i1 in lib.prange(0, natm, blksize):
        log.info('Solving CPHF equation for atoms [%d:%d]', i0, i1)
        _S['blocks'] += 1

        h1mo_blk = cp.asarray(h1mo[i0:i1])
        s1mo_blk = cp.empty_like(h1mo_blk)
        for k, (p0, p1) in enumerate(aoslices[i0:i1, 2:]):
            s1ao = cp.zeros((3, nao, nao))
            s1ao[:, p0:p1] += ipovlp[:, p0:p1]
            s1ao[:, :, p0:p1] += ipovlp[:, p0:p1].transpose(0, 2, 1)
            tmp = contract('xij,jo->xio', s1ao, mocc)
            s1mo_blk[k] = contract('xio,ip->xpo', tmp, mo_coeff)

        mo1 = hs = h1mo_blk - s1mo_blk * e_i
        mo_e1 = hs[:, :, occidx]
        mo1[:, :, viridx] *= -e_ai
        mo1[:, :, occidx] = -s1mo_blk[:, :, occidx] * .5
        hs = s1mo_blk = h1mo_blk = s1ao = tmp = None

        tol = mf.conv_tol_cpscf * (i1 - i0)
        nsets_next[0] = (i1 - i0) * 3
        raw_mo1 = krylov(fvind_vo, mo1.reshape(-1, nmo*nocc),
                         tol=tol, max_cycle=max_cycle, verbose=log,
                         keep=_keeper(budget, nsets_next))
        raw_mo1 = raw_mo1.reshape(i1-i0, 3, nmo, nocc)
        raw_mo1[:, :, occidx] = mo1[:, :, occidx]

        v1 = fxg(raw_mo1.reshape(-1, ndim)).reshape(i1-i0, 3, nmo, nocc)
        mo1[:, :, viridx] -= v1[:, :, viridx] * e_ai
        mo_e1 += v1[:, :, occidx]
        mo_e1 += mo1[:, :, occidx] * (e_i[:, None] - e_i)

        mo1s[i0:i1] = mo1.get()
        e1s[i0:i1] = mo_e1.get()
        mo1 = raw_mo1 = mo_e1 = v1 = None
    log.timer('CPHF solver', *t0)
    return mo1s, e1s


HRHF.solve_mo1 = _screened(solve_mo1_r)
HUHF.solve_mo1 = _screened(solve_mo1_u)
