"""``_decompose_j2c``, taken apart: which of its parts is the 20-62 ms.

``docs/DF_WORK_LIST.md`` **W13**.  The metric stage of the cderi build is
20-62 ms on every system from 10 to 42 atoms -- 40 ms at 23 atoms against a
27 ms block loop -- and ``docs/README_fastdfcderi.md`` §2 measured it as a
stage without looking inside it.  W13's instruction is to *find out which of
the three it is* (``int2c2e``, ``potrf``, the triangular solve) and whether a
batched or a CPU factorisation is faster at those orders.

This is that measurement.  ``_decompose_j2c`` is

    j2c      = int3c2e_bdiv.int2c2e(auxmol)       two-centre integrals
    cd_low   = cusolver.cholesky(j2c)             potrf, plus a copy and a tril
    aux_coef = solve_triangular(cd_low, C.T).T    cuBLAS dtrsm
    aux_coef = aux_coef[aux_sorting]              one scatter

and every one of those lines hides something a stage timer charges to it:
``int2c2e`` constructs a whole ``Int2c2eOpt`` per call -- which rebuilds the
sorted auxiliary basis on the **host** -- ``cholesky`` copies its argument,
queries a workspace, allocates it and reads ``dev_info[0]`` back, which is a
synchronisation, and the solve is a ``dtrsm`` with ``naux_sorted``
right-hand sides.

The table prices each part, then the alternatives: the factorisation on the
host, the whole tail (potrf and trsm) on the host in one round trip, the
integrals with the optimiser built once instead of per call, and ``potrf``
alone with the copy and the ``tril`` taken off it.

    python benchmarks/df_j2c_probe.py --cases glycine,tmc_pt_amine,penicillinV
    python benchmarks/df_j2c_probe.py --cases taxol:0:0 --basis def2-svp
"""
import argparse
import json
import os
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--cases', default='glycine,tmc_pt_amine,tmc_ni_dithiolene,'
                                   'alanine_dipeptide,penicillinV',
                help='comma separated `name[:charge[:spin]]`, resolved as '
                     'benchmarks/molecules/<name>.xyz')
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--reps', type=int, default=20,
                help='this stage is latency, so it needs many more repeats '
                     'than an arithmetic one')
a = ap.parse_args()

import cupy as cp                                            # noqa: E402
import scipy.linalg                                          # noqa: E402
from pyscf import gto                                        # noqa: E402
from pyscf.df import addons                                  # noqa: E402
from cupyx.scipy.linalg import solve_triangular              # noqa: E402
from cupy_backends.cuda.libs import cublas, cusolver         # noqa: E402
from cupy.cuda import device                                 # noqa: E402

from gpu4pyscf.df import df as DFMOD                         # noqa: E402
from gpu4pyscf.df import int3c2e_bdiv as bdiv                # noqa: E402
from gpu4pyscf.gto.mole import SortedMole                    # noqa: E402
from gpu4pyscf.lib.cusolver import cholesky                  # noqa: E402
from gpu4pyscf.pbc.df.int2c2e import Int2c2eOpt              # noqa: E402
from gpu4pyscf.pbc.tools.k2gamma import kpts_to_kmesh        # noqa: E402
from gpu4pyscf.gto.mole import (                             # noqa: E402
    SortedGTO, PBCIntEnvVars, _scale_sp_ctr_coeff)

from _common import needs_ecp                                # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_sync = cp.cuda.Stream.null.synchronize


def timeit(fn, reps=None):
    """Minimum wall time of ``reps`` synchronised calls, in ms."""
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
    return min(ts) * 1e3


def build_case(spec):
    name, _, rest = spec.partition(':')
    charge, _, spin = rest.partition(':')
    xyz = os.path.join(_HERE, 'molecules', f'{name}.xyz')
    kw = dict(atom=xyz, basis=a.basis, charge=int(charge or 0),
              spin=int(spin or 0), verbose=0, max_memory=80000)
    if needs_ecp(xyz, a.basis):
        kw['ecp'] = a.basis
    mol = gto.M(**kw)
    auxmol = addons.make_auxmol(mol, a.auxbasis)
    intopt = bdiv.Int3c2eOpt(mol, auxmol)
    intopt.mol = SortedMole.from_mol(mol, decontract=True)
    intopt.build()
    return mol, intopt


def _potrf_only(x):
    """``cusolver.cholesky`` without the copy, the ``tril`` or the sync.

    What is left out is exactly what is not the factorisation: the
    ``A.copy()`` that makes the buffer, ``cupy.linalg._util._tril`` that
    zeroes the other triangle, and the ``dev_info[0]`` read that drags the
    host to a halt to find out whether it worked.
    """
    n = len(x)
    handle = device.get_cusolver_handle()
    size = cusolver.dpotrf_bufferSize(
        handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n)
    work = cp.empty(size)
    info = cp.empty(1, dtype=np.int32)
    cusolver.dpotrf(handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n,
                    work.data.ptr, size, info.data.ptr)
    return x


def _host_cholesky(j2c):
    """The factorisation on the host: D2H, LAPACK ``dpotrf``, H2D."""
    h = cp.asnumpy(j2c)
    low = scipy.linalg.cholesky(h, lower=True, overwrite_a=True,
                                check_finite=False)
    return cp.asarray(np.ascontiguousarray(np.tril(low)))


def _host_tail(j2c, ctr_coeff_h):
    """potrf *and* the solve on the host, one round trip for both."""
    h = cp.asnumpy(j2c)
    low = scipy.linalg.cholesky(h, lower=True, overwrite_a=True,
                                check_finite=False)
    x = scipy.linalg.solve_triangular(low, ctr_coeff_h.T, lower=True,
                                      overwrite_b=False, check_finite=False)
    return cp.asarray(np.ascontiguousarray(x.T))


def _pair_cache_gpu(cell):
    """`Int2c2eOpt.__init__`'s own loop, verbatim, with bvk_ncells = 1."""
    nbas = cell.nbas
    out = {}
    l_ctr_offsets = np.append(0, np.cumsum(cell.l_ctr_counts))
    uniq_l = cell.uniq_l_ctr[:, 0]
    img = cp.arange(1, dtype=np.int32)
    for i in range(len(uniq_l)):
        for j in range(i + 1):
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i + 1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j + 1]
            ish = cp.arange(ish0, ish1, dtype=np.int32)
            jsh = cp.arange(jsh0, jsh1, dtype=np.int32)
            ijsh = ish[:, None] * nbas + jsh
            if i == j:
                ijsh = ijsh[cp.tril_indices(ish1 - ish0)]
            else:
                ijsh = ijsh.ravel()
            idx = (img[:, None] * nbas + ijsh).ravel()
            out[i, j] = cp.asarray(idx, dtype=np.uint32)
    return out


def _pair_cache_cpu(cell):
    """The same table, built on the host and moved in one transfer.

    The loop above is `O(groups^2)` *cupy* calls -- an `arange`, a
    broadcast, a `tril_indices`, a ravel and an `asarray` per pair of
    angular-momentum classes -- and a def2-SVP auxiliary basis has enough
    classes that the launch latency of those is the whole of it.  None of it
    depends on the geometry or on anything a kernel computed.
    """
    nbas = cell.nbas
    l_ctr_offsets = np.append(0, np.cumsum(cell.l_ctr_counts))
    uniq_l = cell.uniq_l_ctr[:, 0]
    parts, keys = [], []
    for i in range(len(uniq_l)):
        for j in range(i + 1):
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i + 1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j + 1]
            ish = np.arange(ish0, ish1, dtype=np.int64)
            jsh = np.arange(jsh0, jsh1, dtype=np.int64)
            ijsh = ish[:, None] * nbas + jsh
            if i == j:
                ijsh = ijsh[np.tril_indices(ish1 - ish0)]
            else:
                ijsh = ijsh.ravel()
            keys.append((i, j))
            parts.append(ijsh.astype(np.uint32))
    flat = cp.asarray(np.concatenate(parts))
    out, off = {}, 0
    for key, part in zip(keys, parts):
        out[key] = flat[off:off + part.size]
        off += part.size
    return out


print(f'# reps={a.reps}  basis={a.basis}  '
      f'OMP_NUM_THREADS={os.environ.get("OMP_NUM_THREADS")}', flush=True)
print(f'{"":30s} {"naux":>6s} {"nauxS":>6s} | {"_decompose_j2c":>14s} | '
      f'{"int2c2e":>8s} {"kmesh":>7s} {"Opt()":>7s} {"kernel":>7s} | '
      f'{"cholesky":>8s} {"copy":>6s} {"potrf":>7s} {"tril+info":>9s} | '
      f'{"trsm":>7s} {"scatter":>7s}', flush=True)

rows = []
for spec in a.cases.split(','):
    if not spec:
        continue
    mol, intopt = build_case(spec)
    saux = intopt.auxmol
    naux = saux.mol.nao
    naux_sorted = saux.nao
    C = saux.ctr_coeff
    C_h = cp.asnumpy(C)

    t_total = timeit(lambda: DFMOD._decompose_j2c(saux, None))

    # --- the integrals ---------------------------------------------------
    t_int2c2e = timeit(lambda: bdiv.int2c2e(saux))
    t_kmesh = timeit(lambda: kpts_to_kmesh(saux, None, bound_by_supmol=True))
    kmesh = kpts_to_kmesh(saux, None, bound_by_supmol=True)
    t_opt = timeit(lambda: Int2c2eOpt(saux, kmesh))
    opt = Int2c2eOpt(saux, kmesh)
    t_kernel = timeit(lambda: opt.int2c2e(None, omega=None))
    # and what inside the optimiser's construction that is
    sorted_cell = SortedGTO.from_cell(saux)
    t_sort = timeit(lambda: SortedGTO.from_cell(saux))
    t_scale = timeit(lambda: _scale_sp_ctr_coeff(sorted_cell))
    _env_scaled = _scale_sp_ctr_coeff(sorted_cell)
    _Ls = cp.zeros((1, 3))
    t_envs = timeit(lambda: PBCIntEnvVars.new(
        sorted_cell.natm, sorted_cell.nbas, 1, 1, sorted_cell._atm,
        sorted_cell._bas, _env_scaled, sorted_cell.ao_loc, _Ls))
    t_pairs = timeit(lambda: _pair_cache_gpu(sorted_cell))
    t_pairs_cpu = timeit(lambda: _pair_cache_cpu(sorted_cell))
    ref_pairs = _pair_cache_gpu(sorted_cell)
    got_pairs = _pair_cache_cpu(sorted_cell)
    pairs_ok = (set(ref_pairs) == set(got_pairs) and
                all(bool(cp.array_equal(ref_pairs[k], got_pairs[k]))
                    for k in ref_pairs))
    ngroups = len(sorted_cell.uniq_l_ctr)

    # --- the factorisation ------------------------------------------------
    j2c = bdiv.int2c2e(saux)
    t_chol = timeit(lambda: cholesky(j2c))
    buf = cp.empty_like(j2c)
    t_copy = timeit(lambda: cp.copyto(buf, j2c))

    def _copy_potrf():
        cp.copyto(buf, j2c)
        return _potrf_only(buf)
    t_potrf = max(timeit(_copy_potrf) - t_copy, 0.)

    # --- the solve and the scatter ---------------------------------------
    cd_low = cholesky(j2c)
    t_trsm = timeit(lambda: solve_triangular(cd_low, C.T, lower=True,
                                             overwrite_b=False).T)
    aux_coef = solve_triangular(cd_low, C.T, lower=True,
                                overwrite_b=False).T
    perm = cp.asarray(np.random.permutation(aux_coef.shape[0]))

    def _scatter():
        out = cp.empty_like(aux_coef)
        out[perm] = aux_coef
        return out
    t_scatter = timeit(_scatter)

    print(f'{spec + "/" + a.basis:30s} {naux:6d} {naux_sorted:6d} | '
          f'{t_total:13.2f}m | {t_int2c2e:7.2f}m {t_kmesh:6.2f}m '
          f'{t_opt:6.2f}m {t_kernel:6.2f}m | {t_chol:7.2f}m {t_copy:5.2f}m '
          f'{t_potrf:6.2f}m {t_chol - t_copy - t_potrf:8.2f}m | '
          f'{t_trsm:6.2f}m {t_scatter:6.2f}m', flush=True)
    print(f'{"":30s} {"inside Int2c2eOpt():":>19s} '
          f'{ngroups} l-classes -> {ngroups*(ngroups+1)//2} pairs | '
          f'from_cell {t_sort:6.2f}m | _scale_sp_ctr_coeff {t_scale:6.2f}m | '
          f'PBCIntEnvVars {t_envs:6.2f}m | pair cache {t_pairs:6.2f}m '
          f'-> on the host {t_pairs_cpu:6.2f}m (same table: {pairs_ok})',
          flush=True)

    # --- the alternatives -------------------------------------------------
    t_host_chol = timeit(lambda: _host_cholesky(j2c))
    t_host_tail = timeit(lambda: _host_tail(j2c, C_h))
    err_low = float(cp.abs(_host_cholesky(j2c) - cd_low).max() /
                    float(cp.abs(cd_low).max()))
    err_tail = float(cp.abs(_host_tail(j2c, C_h) - aux_coef).max() /
                     float(cp.abs(aux_coef).max()))
    print(f'{"":30s} {"alternatives:":>13s} '
          f'int2c2e, optimiser cached {t_kernel:7.2f}m  | '
          f'host potrf {t_host_chol:7.2f}m (rel {err_low:.1e})  | '
          f'host potrf+trsm {t_host_tail:7.2f}m (rel {err_tail:.1e})  | '
          f'GPU potrf+trsm {t_chol + t_trsm:7.2f}m', flush=True)

    rows.append(dict(case=spec, naux=naux, naux_sorted=naux_sorted,
                     total=t_total, int2c2e=t_int2c2e, kmesh=t_kmesh,
                     opt=t_opt, kernel=t_kernel, cholesky=t_chol,
                     copy=t_copy, potrf=t_potrf, trsm=t_trsm,
                     scatter=t_scatter, host_potrf=t_host_chol,
                     host_tail=t_host_tail, groups=int(ngroups),
                     sort=t_sort, scale=t_scale, envs=t_envs,
                     pairs=t_pairs, pairs_cpu=t_pairs_cpu,
                     pairs_ok=bool(pairs_ok)))
    del j2c, cd_low, aux_coef, buf, opt
    cp.get_default_memory_pool().free_all_blocks()

print('@@@JSON ' + json.dumps(rows), flush=True)
