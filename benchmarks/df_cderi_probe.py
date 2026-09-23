"""Where the time in ``df/df.py:_cholesky_eri`` actually is.

``docs/DF_WORK_LIST.md`` **W4**.  The cderi build is 39.4 % of a 15-cycle
taxol DF SCF and 12-23 % of the shorter ones, and nothing in this repository
is on it.  W4 names two leads -- the metric contraction ``j3c.dot(aux_coef)``,
which is a dense ``npair x naux_sorted x naux`` GEMM against a matrix that is
0.06 % dense folded with a triangular factor, and the ``nsp_per_block*4``
shell-pair aggregation W3 found costing the gradient's kernels 1.26x-3.15x --
and its first instruction is to *measure them on the build* rather than on a
kernel in isolation, because the build has a metric contraction and a host
transpose on top of the integrals.

This is that measurement.  It runs a traced clone of ``_cholesky_eri``,
checked against the real function on every run, and reports

  * ``_decompose_j2c``          the metric: int2c2e, Cholesky, the solve
  * ``eval_j3c``                the Rys kernel, per batch
  * ``j3c.dot(aux_coef)``       the metric contraction, per batch
  * the store                   ``out[p0:p1]`` on the device, or
                                ``transpose_write`` to host memory
  * the evaluator's construction, which is where the aggregation happens

then sweeps ``sp_mult`` over the whole build and prices the metric
contraction's alternatives at the shapes the build actually uses.

    python benchmarks/df_cderi_probe.py \
        --xyz benchmarks/molecules/taxol.xyz --basis def2-svp [--cpu-cderi]
"""
import ctypes
import json
import time

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--cpu-cderi', action='store_true',
                help='force the host-resident branch of the build')
ap.add_argument('--sp-mults', default='1,2,4,8',
                help="shell pairs per thread block, as a multiple of the "
                     "scheme's nsp_per_block.  GPU4PySCF's is 4")
ap.add_argument('--no-check', action='store_true',
                help='skip the comparison against `_cholesky_eri`, which '
                     'holds a second copy of the cderi')
ap.add_argument('--trmm-ladder', default='',
                help='comma separated auxiliary dimensions; times a '
                     'triangular multiply, a triangular solve and a GEMM at '
                     'each of them on synthetic matrices, which is what '
                     'places the size threshold the module needs')
ap.add_argument('--skip-metric', action='store_true',
                help='skip the metric-contraction alternatives')
a = ap.parse_args()
apply_patch(a.patch)

import gpu4pyscf                                             # noqa: E402
from pyscf.df import addons                                  # noqa: E402
from cupy.cuda import cublas                                 # noqa: E402
from cupyx.scipy.linalg import solve_triangular              # noqa: E402
from gpu4pyscf.df import df as DFMOD                         # noqa: E402
from gpu4pyscf.df import int3c2e_bdiv as bdiv                # noqa: E402
from gpu4pyscf.df.int3c2e_bdiv import int3c2e_scheme         # noqa: E402
from gpu4pyscf.gto.mole import SortedMole                    # noqa: E402
from gpu4pyscf.lib.cupy_helper import (                      # noqa: E402
    cholesky, empty_mapped, get_avail_mem, ndarray)

libvhf_rys = bdiv.libvhf_rys
_sync = cp.cuda.Stream.null.synchronize


def timeit(label, fn, reps=None):
    reps = a.reps if reps is None else reps
    fn()
    _sync()
    ts = []
    for _ in range(reps):
        _sync()
        t = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t)
    print(f'{label:52s} {min(ts)*1e3:10.2f} ms', flush=True)
    return min(ts)


# --------------------------------------------------------------------------
# the object `DF.build` hands `_cholesky_eri`
# --------------------------------------------------------------------------
mol = build_mol(a)
auxmol = addons.make_auxmol(mol, a.auxbasis)
intopt = bdiv.Int3c2eOpt(mol, auxmol)
intopt.mol = SortedMole.from_mol(mol, decontract=True)
t = time.perf_counter()
intopt.build()
_sync()
t_build = time.perf_counter() - t

smol, saux = intopt.mol, intopt.auxmol
naux = auxmol.nao
naux_sorted = saux.nao
nao = mol.nao
C = saux.ctr_coeff
nnz = int((cp.abs(C) > 1e-14).sum())
recontract_bas = cp.asnumpy(smol.recontract_bas)
needs_recontraction = any(recontract_bas[:, bdiv.NCTR_OF] != 1)
print(f'[{a.xyz.rsplit("/", 1)[-1]}/{a.basis}] natm={mol.natm} nao={nao} '
      f'naux={naux} naux_sorted={naux_sorted} '
      f'recontract={needs_recontraction} '
      f'ctr_coeff {naux_sorted}x{naux} nnz={nnz} '
      f'({100.*nnz/C.size:.3f} % dense)  gpu4pyscf={gpu4pyscf.__version__}',
      flush=True)
print(f'{"Int3c2eOpt.build (setup, once)":52s} {t_build*1e3:10.2f} ms',
      flush=True)


# --------------------------------------------------------------------------
# the aggregation constant, scoped to one evaluator construction
# --------------------------------------------------------------------------
_orig_agg = SortedMole.aggregate_shl_pairs
_REF_NSP = int3c2e_scheme(short_range=False, gout_width=54,
                          cache_cart_idx=True)[0][0]


class spmult:
    """``nsp_per_block*sp_mult`` in ``int3c2e_evaluator``, for one call.

    The constant is written into upstream's body
    (``df/int3c2e_bdiv.py:176``), so the only way to reach it without copying
    the whole evaluator is to intercept what it passes to
    ``aggregate_shl_pairs``.  The interception asserts that what it sees is
    the scheme's own array times 4 -- if upstream ever changes the constant
    the assertion fires rather than the sweep quietly measuring nothing.
    """

    def __init__(self, mult):
        self.mult = mult

    def __enter__(self):
        mult = self.mult

        def agg(self, cache=None, nsp_per_block=512):
            if not isinstance(nsp_per_block, (int, np.integer)):
                assert cp.array_equal(nsp_per_block, _REF_NSP * 4), \
                    'the evaluator no longer aggregates at nsp_per_block*4'
                nsp_per_block = _REF_NSP * mult
            return _orig_agg(self, cache, nsp_per_block)

        SortedMole.aggregate_shl_pairs = agg
        return self

    def __exit__(self, *exc):
        SortedMole.aggregate_shl_pairs = _orig_agg


# --------------------------------------------------------------------------
# a traced clone of `_cholesky_eri`, single device
# --------------------------------------------------------------------------
def make_clone(sp_mult=4, on_gpu=None, metric='gemm'):
    """Return ``(run, info)``; ``run()`` performs one whole build."""
    mem_avail = get_avail_mem(exclude_memory_pool=True)
    word_avail = mem_avail // 8
    batch_size = int(word_avail * .2) // naux_sorted
    reorder = metric in ('gemm', 'gather')

    t = time.perf_counter()
    with spmult(sp_mult):
        eval_j3c, aux_sorting, ao_pair_offsets, _, clone_context = \
            intopt.int3c2e_evaluator(
                ao_pair_batch_size=batch_size, reorder_aux=reorder,
                pair_batch_by_l=needs_recontraction,
                return_clone_context=True)
    _sync()
    t_evaluator = time.perf_counter() - t
    cderi_batch_size = int(max(ao_pair_offsets[1:] - ao_pair_offsets[:-1]))
    batch_size = cderi_batch_size
    num_batches = len(ao_pair_offsets) - 1

    if needs_recontraction:
        recontract, _, contracted_ao_pair_counts, pair_addresses = \
            bdiv._create_pair_recontraction(smol, clone_context)
        cderi_offsets = np.append(0, np.cumsum(contracted_ao_pair_counts))
        cderi_batch_size = int(max(contracted_ao_pair_counts))
        cderi_npairs = len(pair_addresses)
    else:
        recontract = None
        cderi_offsets = ao_pair_offsets
        pair_addresses = intopt.pair_and_diag_indices()[0]
        cderi_npairs = len(pair_addresses)

    t = time.perf_counter()
    aux_coef, tag = DFMOD._decompose_j2c(
        saux, aux_sorting if reorder else None)
    _sync()
    t_j2c = time.perf_counter() - t

    # the triangular factor the alternatives need.  `_decompose_j2c` throws it
    # away, so it is rebuilt here; on the module's side it is kept.
    j2c = bdiv.int2c2e(saux)
    cd_low = cholesky(j2c) if tag == 'cd' else None

    host = (not on_gpu) if on_gpu is not None else a.cpu_cderi
    t_alloc = 0.
    if host:
        # the page-locked host buffer, which is the one stage of the build
        # that is neither integrals nor arithmetic
        t = time.perf_counter()
        cderi_cpu = empty_mapped((naux, cderi_npairs))
        t_alloc = time.perf_counter() - t
        out = None
    else:
        out = cp.empty((cderi_npairs, naux))
        cderi_cpu = None

    work = cp.empty(naux_sorted * batch_size)
    work2 = cp.empty(naux * batch_size)
    work1 = cp.empty(naux * cderi_batch_size) if needs_recontraction else None
    times = {}

    def contract(j3c, dest):
        """One block's metric contraction, into ``dest`` (npair_blk, naux)."""
        if metric == 'gemm':                      # GPU4PySCF's
            return j3c.dot(aux_coef, out=dest)
        if metric == 'gather':                    # undo reorder_aux, then C
            j3c = j3c[:, aux_sorting]
        x = saux.apply_CT_dot(j3c, axis=-1, out=dest)
        return _apply_linv(x, cd_low)

    def run(trace=False):
        for batch_id in range(num_batches):
            if trace:
                _sync()
                t0 = time.perf_counter()
            j3c = eval_j3c(shl_pair_batch_id=batch_id, out=work)
            if trace:
                _sync()
                t1 = time.perf_counter()
                times['eval_j3c'] = times.get('eval_j3c', 0) + t1 - t0
            p0, p1 = cderi_offsets[batch_id:batch_id + 2]
            if needs_recontraction or host:
                dest = ndarray((j3c.shape[0], naux), buffer=work2)
            else:
                dest = out[p0:p1]
            blk = contract(j3c, dest)
            if trace:
                _sync()
                t2 = time.perf_counter()
                times['metric'] = times.get('metric', 0) + t2 - t1
            if needs_recontraction:
                blk = recontract(batch_id, blk,
                                 out=work1 if host else out[p0:p1])
            if host:
                err = libvhf_rys.transpose_write(
                    cderi_cpu.ctypes,
                    ctypes.cast(blk.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(naux), ctypes.c_int(cderi_npairs),
                    ctypes.c_int(p0), ctypes.c_int(p1))
                if err != 0:
                    raise RuntimeError('transpose_write failed')
            if trace:
                _sync()
                times['store'] = times.get('store', 0) + time.perf_counter() - t2
        return cderi_cpu if host else out.T

    info = dict(num_batches=num_batches, batch_size=int(batch_size),
                cderi_npairs=int(cderi_npairs), tag=tag, host=host,
                t_evaluator=t_evaluator, t_j2c=t_j2c, t_alloc=t_alloc,
                times=times)
    return run, info


def _apply_linv(x, cd_low):
    """``x @ cd_low^{-T}`` in place, as a triangular solve.

    ``x`` is C-ordered ``(m, naux)``, which cuBLAS reads as an F-ordered
    ``(naux, m)``; in that picture the right-side solve is a left-side one.
    """
    handle = cp.cuda.device.get_cublas_handle()
    m, n = x.shape
    one = np.ones(1)
    # the buffer holds `cd_low` in C order, which cuBLAS reads as its
    # transpose -- an upper-triangular matrix -- so the factor itself is
    # `op(A) = A^T`.
    cublas.dtrsm(handle, cublas.CUBLAS_SIDE_LEFT, cublas.CUBLAS_FILL_MODE_UPPER,
                 cublas.CUBLAS_OP_T, cublas.CUBLAS_DIAG_NON_UNIT,
                 n, m, one.ctypes.data,
                 cd_low.data.ptr, n, x.data.ptr, n)
    return x


# --------------------------------------------------------------------------
# 1. the clone is the function
# --------------------------------------------------------------------------
if not a.no_check:
    print('\n--- the clone against `_cholesky_eri` ---', flush=True)
    ref_cderi, ref_idx = DFMOD._cholesky_eri(
        intopt, use_gpu_memory=False if a.cpu_cderi else None)
    ref = cp.asarray(ref_cderi[0])
    run_ref, info_ref = make_clone(sp_mult=4)
    got = cp.asarray(run_ref())
    err = float(cp.abs(got - ref).max())
    scale = float(cp.abs(ref).max())
    print(f'batches={info_ref["num_batches"]} '
          f'npairs={info_ref["cderi_npairs"]} tag={info_ref["tag"]} '
          f'host={info_ref["host"]}  '
          f'max|clone - _cholesky_eri| = {err:.3e}  '
          f'(max|cderi| = {scale:.3e})', flush=True)
    del ref_cderi, ref, got, run_ref
    cp.get_default_memory_pool().free_all_blocks()

# --------------------------------------------------------------------------
# 2. the stage table, at GPU4PySCF's configuration
# --------------------------------------------------------------------------
print('\n--- stages of one build, GPU4PySCF\'s sp_mult=4 ---', flush=True)
run, info = make_clone(sp_mult=4)
run(trace=True)                      # warm
info['times'].clear()
t_total = timeit('  whole build (loop only)', run)
run(trace=True)
tr = info['times']
print(f'{"  evaluator construction (once per build)":52s} '
      f'{info["t_evaluator"]*1e3:10.2f} ms', flush=True)
print(f'{"  _decompose_j2c (once per build)":52s} '
      f'{info["t_j2c"]*1e3:10.2f} ms', flush=True)
if info['t_alloc']:
    gb = naux * info['cderi_npairs'] * 8e-9
    print(f'{"  empty_mapped, the host buffer (once per build)":52s} '
          f'{info["t_alloc"]*1e3:10.2f} ms   ({gb:.2f} GB, '
          f'{gb/info["t_alloc"]:.2f} GB/s)', flush=True)
for k in ('eval_j3c', 'metric', 'store'):
    if k in tr:
        print(f'  {k:50s} {tr[k]*1e3:10.2f} ms '
              f'({100.*tr[k]/sum(tr.values()):5.1f} % of the loop)', flush=True)
whole = t_total + info['t_evaluator'] + info['t_j2c'] + info['t_alloc']
print(f'{"  build, with setup":52s} {whole*1e3:10.2f} ms', flush=True)
R = {'stages': {k: v for k, v in tr.items()},
     't_loop': t_total, 't_evaluator': info['t_evaluator'],
     't_j2c': info['t_j2c'], 't_alloc': info['t_alloc'],
     't_intopt': t_build, 'info': {
         k: v for k, v in info.items() if k != 'times'}}
del run
cp.get_default_memory_pool().free_all_blocks()

# --------------------------------------------------------------------------
# 3. sp_mult, on the whole build
# --------------------------------------------------------------------------
print('\n--- sp_mult, on the whole build ---', flush=True)
R['sp_mult'] = {}
ref_c = None
for sm in [int(x) for x in a.sp_mults.split(',')]:
    run = None
    try:
        run, info = make_clone(sp_mult=sm)
        t = timeit(f'  sp_mult={sm:<3d}', run)
        R['sp_mult'][str(sm)] = dict(
            loop=t, evaluator=info['t_evaluator'],
            whole=t + info['t_evaluator'] + info['t_j2c'])
        # a fingerprint rather than a copy, and one taken by reductions on
        # the tensor in place: at 113 atoms a second cderi is 14 GB and the
        # sweep would spend the device on it
        v = run()
        fp = (float(v[::7].max()), float(v[::7].min()))
        del v
        if ref_c is None:
            ref_c = fp
        elif max(abs(x - y) for x, y in zip(fp, ref_c)) > 1e-9:
            print(f'      !! the tensor differs from the first aggregation: '
                  f'{fp} against {ref_c}', flush=True)
    except Exception as exc:                                 # noqa: BLE001
        print(f'  sp_mult={sm:<3d}  ERR {type(exc).__name__}: '
              f'{str(exc)[:100]}', flush=True)
    finally:
        del run
        cp.get_default_memory_pool().free_all_blocks()
_BEST_SPMULT = min(R['sp_mult'], key=lambda k: R['sp_mult'][k]['loop']) \
    if R['sp_mult'] else '4'
_BEST_SPMULT = int(_BEST_SPMULT)
print(f'  best sp_mult = {_BEST_SPMULT}', flush=True)
del ref_c
cp.get_default_memory_pool().free_all_blocks()

# --------------------------------------------------------------------------
# 4. the metric contraction, component by component
# --------------------------------------------------------------------------
# Upstream does the whole thing as one dense `npair x naux_sorted x naux`
# GEMM against `C L^{-T}`.  W4 proposes splitting it -- `C` as a sparse
# gather, then a triangular solve -- and attaches its own falsification:
# "`dtrsm` at these shapes runs at less than half of `dgemm`".  This prices
# every piece separately, in achieved TF/s, so the answer does not depend on
# the rest of the build.
if not a.skip_metric:
    print('\n--- the metric contraction, per component ---', flush=True)
    import electrolites                                       # noqa: E402
    electrolites.patch('fastdfcderi')
    from electrolites import fastdfcderi as F                 # noqa: E402

    # no clone here: its output tensor is 14 GB at 113 atoms and the
    # components are measured on one slice of one block instead
    m = max(1, int(32e6) // naux * 8)
    aux_coef, tag, cd_low = F._decompose_j2c(saux, None, None)
    cd_low_c = cp.ascontiguousarray(cd_low)
    cd_inv = cp.ascontiguousarray(solve_triangular(
        cd_low, cp.eye(naux), lower=True))
    with spmult(_BEST_SPMULT):
        ev_sorted, aux_sorting, offs, _ = intopt.int3c2e_evaluator(
            ao_pair_batch_size=m, reorder_aux=True)[:4]
        ev_natural = intopt.int3c2e_evaluator(
            ao_pair_batch_size=m, reorder_aux=False)[0]
    j3c = ev_sorted(0)
    j3c_nat = ev_natural(0)
    m = j3c.shape[0]
    print(f'  one block of {m} pairs x {naux_sorted} sorted auxiliary '
          f'functions -> {naux}', flush=True)

    aux_coef_p = cp.empty_like(aux_coef)
    aux_coef_p[aux_sorting] = aux_coef
    idx, coef, counts, maxnnz = F._sparse_ctr_coeff(saux, aux_sorting)
    out = cp.empty((m, naux))
    out2 = cp.empty((m, naux))

    gemm_flops = 2. * m * naux_sorted * naux
    tri_flops = 1. * m * naux * naux
    comp = {}

    def report(label, fn, flops):
        t = timeit(f'  {label}', fn)
        comp[label.strip()] = dict(t=t, tflops=flops / t / 1e12)
        print(f'{"":52s} {flops/t/1e12:10.2f} TF/s', flush=True)
        return t

    t_gemm = report('j3c . (C L^-T)            GPU4PySCF, dense',
                    lambda: j3c.dot(aux_coef_p, out=out), gemm_flops)
    t_sp = report('j3c . C                   this module, sparse',
                  lambda: F._contract_aux(
                      j3c, idx, coef, counts, np.int32(naux_sorted),
                      np.int32(naux), np.int32(maxnnz), out), 0.)
    t_mdc = report('j3c . C                   mat_dot_C, natural order',
                   lambda: saux.apply_CT_dot(j3c_nat, axis=-1, out=out2), 0.)
    t_trsm = report('. L^-T                    dtrsm',
                    lambda: F._trsm_right_lowerT(out, cd_low_c), tri_flops)
    t_trmm = report('. L^-T                    dtrmm, explicit inverse',
                    lambda: F._trmm_right_lowerT(out, cd_inv, out=out2),
                    tri_flops)
    t_dense = report('. L^-T                    dgemm, explicit inverse',
                     lambda: cp.dot(out, cd_inv.T, out=out2), 2 * tri_flops)

    print(f'\n  GPU4PySCF                     {t_gemm*1e3:9.2f} ms', flush=True)
    for name, t in (('sparse + dtrmm', t_sp + t_trmm),
                    ('sparse + dtrsm', t_sp + t_trsm),
                    ('sparse + dgemm', t_sp + t_dense),
                    ('mat_dot_C + dtrsm', t_mdc + t_trsm)):
        print(f'  {name:28s}  {t*1e3:9.2f} ms   {t_gemm/t:6.3f}x', flush=True)
    R['metric'] = comp
    del j3c, j3c_nat, out, out2, aux_coef, aux_coef_p, cd_inv, cd_low_c
    cp.get_default_memory_pool().free_all_blocks()

# --------------------------------------------------------------------------
# 4b. the page-locked host buffer
# --------------------------------------------------------------------------
if a.cpu_cderi:
    print('\n--- page-locking the host buffer ---', flush=True)
    # `_dfpin` rather than `fastdfcderi`, because importing the latter
    # patches the build this probe is measuring (docs/DF_WORK_LIST.md 5.12),
    # and `pool=False` because the pool would make every repeat but the first
    # measure a dict lookup instead of a page-locking
    from electrolites import _dfpin as _FP                    # noqa: E402
    gb = naux * R['info']['cderi_npairs'] * 8e-9
    R['pin'] = {}
    for label, fn in (('cudaHostAlloc (GPU4PySCF)',
                       lambda: empty_mapped((naux, R['info']['cderi_npairs']))),
                      ('cudaHostRegister, 4 kB pages',
                       lambda: _FP.empty_mapped_pooled(
                           (naux, R['info']['cderi_npairs']),
                           hugepage=False, pool=False)),
                      ('cudaHostRegister + MADV_HUGEPAGE',
                       lambda: _FP.empty_mapped_pooled(
                           (naux, R['info']['cderi_npairs']), pool=False))):
        ts = []
        for _ in range(max(2, a.reps)):
            t = time.perf_counter()
            buf = fn()
            ts.append(time.perf_counter() - t)
            del buf
        R['pin'][label] = min(ts)
        print(f'  {label:44s} {min(ts)*1e3:9.1f} ms   '
              f'{gb/min(ts):5.2f} GB/s', flush=True)

# --------------------------------------------------------------------------
# 5. where the triangular routines stop paying
# --------------------------------------------------------------------------
# `dtrmm` does half the arithmetic of the `dgemm` that can replace it and is
# worth having only while cuBLAS runs it at GEMM throughput.  That is a
# property of the auxiliary dimension alone, so it can be measured on
# synthetic matrices and does not need a molecule per point.
if a.trmm_ladder:
    print('\n--- dtrmm, dtrsm and dgemm against the auxiliary dimension ---',
          flush=True)
    from electrolites import fastdfcderi as _F                # noqa: E402
    R['ladder'] = {}
    rng = np.random.default_rng(0)
    print(f'{"naux":>6s} {"npair":>8s} {"dtrmm":>9s} {"dtrsm":>9s} '
          f'{"dgemm":>9s}   TF/s', flush=True)
    for n in [int(x) for x in a.trmm_ladder.split(',')]:
        # a fixed footprint per point (400 MB), so the rows are comparable
        npair = max(n, int(5e7) // n)
        low = cp.asarray(np.tril(rng.standard_normal((n, n)))
                         + np.eye(n) * n)
        inv = cp.ascontiguousarray(
            solve_triangular(low, cp.eye(n), lower=True))
        x = cp.asarray(rng.standard_normal((npair, n)))
        y = cp.empty_like(x)
        f = 1. * npair * n * n
        t1 = timeit(f'  naux={n:<6d} npair={npair:<8d} dtrmm',
                    lambda: _F._trmm_right_lowerT(x, inv, out=y))
        t2 = timeit(f'  naux={n:<6d} npair={npair:<8d} dtrsm',
                    lambda: _F._trsm_right_lowerT(y, low))
        t3 = timeit(f'  naux={n:<6d} npair={npair:<8d} dgemm',
                    lambda: cp.dot(x, inv.T, out=y))
        R['ladder'][str(n)] = dict(npair=npair, trmm=t1, trsm=t2, dgemm=t3)
        print(f'{n:6d} {npair:8d} {f/t1/1e12:9.2f} {f/t2/1e12:9.2f} '
              f'{2*f/t3/1e12:9.2f}   TF/s   '
              f'trmm/gemm(equal work) {t3/t1/2:.3f}', flush=True)
        del low, inv, x, y
        cp.get_default_memory_pool().free_all_blocks()

print('RESULT ' + json.dumps(dict(
    xyz=a.xyz, basis=a.basis, natm=mol.natm, nao=int(nao), naux=int(naux),
    naux_sorted=int(naux_sorted), nnz=nnz, cpu_cderi=bool(a.cpu_cderi),
    **R), default=float), flush=True)
