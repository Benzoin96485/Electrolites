"""Anatomy of one density-fitted J+K build, and what its two contractions cost.

`df_anatomy.py` says how much of a DF run is `with_df.get_jk`.  This says
what is inside it, on a converged density, one piece at a time:

  * the *unpack* -- `DF.loop` expanding the packed `(naux, npair)` cderi into
    a dense `(nao, nao, blk)` block per iteration, which is what the exchange
    contraction reads;
  * **step 1**, `contract('Lij,njk->nikL')`, whose `L`-innermost output layout
    is what makes step 2 a clean GEMM and step 1 a batched one whose matrices
    have a unit stride in *neither* dimension;
  * **step 2**, `contract('nikL,njkL->nij')`, which for a ground-state density
    contracts a tensor with itself and therefore computes both triangles of a
    symmetric matrix -- a `dsyrk` is the same answer for half the flops.

It also reports the sparsity the packed form carries and the dense unpack
throws away (`npair` against `nao*(nao+1)/2`), and times step 1 again from a
`(L,i,j)` C-contiguous copy, where the whole contraction collapses to one
`dgemm` of shape `(blk*nao, nao) x (nao, nocc)`.

    python benchmarks/df_probe.py --xyz benchmarks/molecules/taxol.xyz \
        --basis def2-svp --cycles 6
"""
import json
import time

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--blksizes', default='',
                help='comma separated block sizes to sweep; default is the '
                     'one GPU4PySCF picks and that doubled and halved')
a = ap.parse_args()
apply_patch(a.patch)

import gpu4pyscf                                             # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.df import df_jk as gdf_jk                     # noqa: E402
from gpu4pyscf.lib.cupy_helper import contract               # noqa: E402
from pyscf import lib                                        # noqa: E402

_sync = cp.cuda.Stream.null.synchronize
R = {}


def timeit(label, fn, reps=None):
    reps = a.reps if reps is None else reps
    try:
        fn()
        _sync()
    except Exception as exc:                                 # noqa: BLE001
        R[label] = f'ERR {type(exc).__name__}: {exc}'
        print(f'{label:34s} ERR {type(exc).__name__}: {str(exc)[:100]}',
              flush=True)
        return None
    ts = []
    for _ in range(reps):
        _sync()
        t = time.perf_counter()
        out = fn()
        _sync()
        ts.append(time.perf_counter() - t)
    R[label] = min(ts)
    print(f'{label:34s} {min(ts)*1e3:9.2f} ms   (min of {reps})', flush=True)
    return out


mol = build_mol(a)
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc).density_fit(
    auxbasis=a.auxbasis)
mf.conv_tol = a.conv_tol
mf.grids.level = a.grids
mf.verbose = 0
mf.max_cycle = max(1, a.cycles)
mf.kernel()

dfobj = mf.with_df
if dfobj._cderi is None:
    dfobj.build()
dm = mf.make_rdm1()
nao = dfobj.nao
naux = dfobj.naux
cderi_sparse = dfobj._cderi[cp.cuda.Device().id]
npair = cderi_sparse.shape[1]
on_gpu = isinstance(cderi_sparse, cp.ndarray)
blk_default = dfobj.get_blksize(mem_fraction=0.4)

factor_l, factor_r = gdf_jk.factorize_dm(dm, hermi=1)
factor_l = cp.asarray(factor_l).reshape(-1, nao, factor_l.shape[-1])
nocc = factor_l.shape[-1]
nspin_dm = factor_l.shape[0]

tag = (f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.auxbasis or "auto"}/{a.xc}/'
       f'{a.patch or "stock"}')
print(f'[{tag}] natm={mol.natm} nao={nao} naux={naux} nocc={nocc} '
      f'ndm={nspin_dm} npair={npair} nao(nao+1)/2={nao*(nao+1)//2} '
      f'fill={100.0*npair/(nao*(nao+1)//2):.1f}% cderi_on_gpu={on_gpu} '
      f'blksize={blk_default} cderi={cderi_sparse.nbytes/1e9:.2f} GB '
      f'gpu4pyscf={gpu4pyscf.__version__}', flush=True)

# ---- the whole build, three ways ------------------------------------------
timeit('get_jk (J+K)',
       lambda: gdf_jk.get_jk(dfobj, dm, 1, True, True))
timeit('get_jk (J only)',
       lambda: gdf_jk.get_jk(dfobj, dm, 1, True, False))
timeit('get_jk (K only)',
       lambda: gdf_jk.get_jk(dfobj, dm, 1, False, True))

# ---- the unpack, on its own, at several block sizes ------------------------
blks = ([int(x) for x in a.blksizes.split(',')] if a.blksizes else
        sorted({max(8, blk_default // 2), blk_default, blk_default * 2}))


def _drain(blksize, unpack):
    def run():
        n = 0
        for _cderi, _tril in dfobj.loop(blksize=blksize, unpack=unpack):
            n += 1
        return n
    return run


for b in blks:
    timeit(f'DF.loop unpack=True blk={b}', _drain(b, True))
timeit(f'DF.loop unpack=False blk={blk_default}', _drain(blk_default, False))

# ---- the two contractions, separately --------------------------------------
# One block is held live so the contractions can be timed on it repeatedly.
blk = blk_default
cderi_blk = tril_blk = None
for cderi_blk, tril_blk in dfobj.loop(blksize=blk, unpack=True):
    break
nL = cderi_blk.shape[0]
rhok = cp.empty((nspin_dm, nao, nocc, nL))
vk = cp.zeros((nspin_dm, nao, nao))
nblocks = (naux + blk - 1) // blk
print(f'  [one block: nL={nL}, {nblocks} of them; rhok '
      f'{rhok.nbytes/1e9:.2f} GB]', flush=True)

timeit('  step1 Lij,njk->nikL',
       lambda: contract('Lij,njk->nikL', cderi_blk, factor_l, out=rhok))
timeit('  step2 nikL,njkL->nij (gemm)',
       lambda: contract('nikL,njkL->nij', rhok, rhok, beta=1, out=vk))

# the same step 2 as a symmetric rank-k update: vk is symmetric whenever the
# two factors are the same tensor, which every ground-state SCF density is
handle = cp.cuda.device.get_cublas_handle()
from cupy.cuda import cublas                                 # noqa: E402
one = np.array(1.0)
_K = nocc * nL


def _syrk():
    for s in range(nspin_dm):
        # cublas is column major: C = A^T A with A stored (K, nao) in F order
        # is C = A A^T on our C-order (nao, K) matrix.
        cublas.dsyrk(handle, cublas.CUBLAS_FILL_MODE_LOWER,
                     cublas.CUBLAS_OP_T, nao, _K,
                     one.ctypes.data, rhok[s].data.ptr, _K,
                     one.ctypes.data, vk[s].data.ptr, nao)


timeit('  step2 as dsyrk (half the flops)', _syrk)

# ---- `fastdf`'s two items, on the same block ------------------------------
# Item 2 is step 1 batched over the AO index instead of over the auxiliary
# one.  It reads GPU4PySCF's own unpacked buffer -- the layout does not change
# -- and writes `(i,L,k)`, which is what item 1's `dsyrk` contracts over.
from electrolites import fastdf                               # noqa: E402

try:
    work = cp.zeros((nao, nao, nL))
    from gpu4pyscf.lib.cupy_helper import fill_symmetric       # noqa: E402
    pair_idx = cp.asarray(dfobj._cderi_idx[0], dtype=np.int32)
    # a host-resident cderi has to be staged before any of this can touch it,
    # and the staged buffer is pair-major where the device-resident tensor is
    # auxiliary-major -- `fill_symmetric` branches on that, so timing the
    # unpack on the wrong one is timing a different kernel
    cderi_raw = (cderi_sparse[:nL] if on_gpu
                 else cp.asarray(np.ascontiguousarray(cderi_sparse[:nL])))
    timeit('  unpack (i,j,L) GPU4PySCF',
           lambda: fill_symmetric(cderi_raw.T, pair_idx, nao, 0, nL, out=work))
    rhok_f = cp.empty((nao, nL * nocc))
    timeit('  step1 batched over i: fastdf',
           lambda: fastdf._rhok(work, factor_l[0], rhok_f, nL, nao, nocc, nL))
    vk1 = cp.zeros((nao, nao))
    timeit('  step2 dsyrk on (i,L,k)',
           lambda: fastdf._syrk(rhok_f, vk1, nao, nL * nocc))
    # and the same block both ways from freshly zeroed accumulators, so the
    # rows above are known to be the same arithmetic and not two different
    # ones.  Both have to be recomputed: every timed call above accumulates
    # into its output with `beta = 1`.
    if nspin_dm == 1:
        vk[:] = 0
        contract('Lij,njk->nikL', cderi_blk, factor_l, out=rhok)
        ref = contract('nikL,njkL->nij', rhok, rhok, beta=1, out=vk)[0]
        vk1[:] = 0
        fastdf._rhok(work, factor_l[0], rhok_f, nL, nao, nocc, nL)
        fastdf._syrk(rhok_f, vk1, nao, nL * nocc)
        got = fastdf._fill_from_triangle(vk1[None].copy())[0]
        rel = (float(cp.abs(got - ref).max()) /
               (float(cp.abs(ref).max()) or 1.))
        print(f'{"  the two step 2s agree to":34s} {rel:9.2e}   relative',
              flush=True)
        R['step2 agreement'] = rel
except cp.cuda.memory.OutOfMemoryError as exc:
    print(f'  fastdf variants skipped: {exc}')

print('RESULT ' + json.dumps(dict(
    tag=tag, gpu4pyscf=gpu4pyscf.__version__, xyz=a.xyz, basis=a.basis,
    auxbasis=a.auxbasis, xc=a.xc, patch=a.patch, natm=mol.natm, nao=int(nao),
    naux=int(naux), nocc=int(nocc), npair=int(npair), on_gpu=bool(on_gpu),
    blksize=int(blk_default), nL=int(nL), nblocks=int(nblocks), t=R)),
    flush=True)
