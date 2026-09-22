"""`fastdfhess` against GPU4PySCF, on the call and on the whole Hessian.

``docs/DF_WORK_LIST.md`` **W8**.  ``benchmarks/df_hess_probe.py`` is the
anatomy -- where the time inside ``df/hessian/rhf.py:_get_jk`` goes, stage by
stage, and what each alternative would cost at the shapes the function uses.
This is the A/B: the function GPU4PySCF ships against the one this repository
does, and the whole density-fitted Hessian around it.

Every variant is timed **round robin** rather than one after the other
(``docs/DF_WORK_LIST.md`` §5.2): a competing job's load drifts over the
minutes a row takes, and a sequential schedule puts all of that drift on one
variant.  Only the ratio inside a round is meaningful.

    python benchmarks/df_hess_bench.py \
        --xyz benchmarks/molecules/penicillinV.xyz --basis def2-svp
    python benchmarks/df_hess_bench.py ... --whole      # the Hessian too

``--arrangements`` selects which of ``fused``/``blasvk``/``upstream`` to
include, ``--knobs`` adds the item-by-item ablation (balanced batches, the
pool-inclusive batch size, the Coulomb pass), and ``--natm-block`` caps the
number of right-hand sides, which is what ``solve_mo1`` does from free
memory and is the largest shape in the function.
"""
import json
import time

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--natm-block', type=int, default=0)
ap.add_argument('--arrangements', default='upstream,fused,blasvk')
ap.add_argument('--knobs', action='store_true',
                help='ablate the four knobs one at a time as well')
ap.add_argument('--whole', action='store_true',
                help='also time a whole Hessian, patched and unpatched, in '
                     'this process and in that order twice (§5.3)')
ap.add_argument('--conv-tol-cpscf', type=float, default=1e-6)
ap.add_argument('--cpu-cderi', action='store_true',
                help='move the cderi to page-locked host memory before '
                     'timing, which is what upstream does to itself for any '
                     'tensor over 0.4 of the free memory -- this is how '
                     'item 5 is priced without needing a molecule big '
                     'enough to trigger it')
ap.add_argument('--j-only', action='store_true',
                help="time the Coulomb-only call, which is a pure "
                     "functional's whole two-electron CPHF")
a = ap.parse_args()
# this bench patches by hand, so that `_OFF` can be flipped between rounds
apply_patch(a.patch)

from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.lib.cupy_helper import (                      # noqa: E402
    contract, tag_array, transpose_sum, get_avail_mem)

import electrolites                                          # noqa: E402
electrolites.patch('fastdfhess')
from electrolites import fastdfhess                          # noqa: E402

GB = 1024.0 ** 3
_sync = cp.cuda.Stream.null.synchronize


def round_robin(variants, reps):
    """``{name: fn}`` timed one round at a time; the minimum of the rounds.

    A variant that *raises* is recorded rather than allowed to end the run:
    at 42 atoms in def2-TZVPP and at 113 atoms in def2-SVP GPU4PySCF's own
    function raises ``CUTENSOR_STATUS_NOT_SUPPORTED`` with tens of GB free,
    and the whole point of the comparison is what happens then.
    """
    best = {k: float('inf') for k in variants}
    err = {}
    for _ in range(reps):
        for name, fn in variants.items():
            if name in err:
                continue
            try:
                _sync()
                t = time.perf_counter()
                fn()
                _sync()
            except Exception as exc:                          # noqa: BLE001
                err[name] = f'{type(exc).__name__}: {str(exc)[:80]}'
                cp.get_default_memory_pool().free_all_blocks()
                continue
            best[name] = min(best[name], time.perf_counter() - t)
    for k, v in err.items():
        best[k] = v
    return best


mol = build_mol(a)
mf = dft.RKS(mol, xc=a.xc).density_fit(auxbasis=a.auxbasis)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.conv_tol_cpscf = a.conv_tol_cpscf
mf.verbose = 0
t0 = time.perf_counter()
e = mf.kernel()
print(f'# {mol.natm} atoms  nao {mol.nao}  {a.basis}  {a.xc}')
print(f'# DF-SCF {time.perf_counter() - t0:.1f} s   E = {e:.10f}', flush=True)

mo_coeff = cp.asarray(mf.mo_coeff)
mo_occ = cp.asarray(mf.mo_occ)
nao, nmo = mo_coeff.shape
orbo = mo_coeff[:, mo_occ > 0]
nocc = orbo.shape[1]
dfobj = mf.with_df
if dfobj._cderi is None:
    dfobj.build()
cderi = dfobj._cderi[cp.cuda.Device().id]
print(f'# naux {cderi.shape[0]}  pairs {cderi.shape[1]}  '
      f'cderi {cderi.nbytes / GB:.2f} GB '
      f'{"device" if isinstance(cderi, cp.ndarray) else "host"}-resident   '
      f'free {get_avail_mem() / GB:.1f} GB')

if a.cpu_cderi and isinstance(cderi, cp.ndarray):
    from gpu4pyscf.lib.cupy_helper import empty_mapped
    host = empty_mapped(cderi.shape)
    host[:] = cp.asnumpy(cderi)
    dfobj._cderi[cp.cuda.Device().id] = host
    del cderi
    cp.get_default_memory_pool().free_all_blocks()
    cderi = dfobj._cderi[cp.cuda.Device().id]
    print(f'# moved to host; free {get_avail_mem() / GB:.1f} GB', flush=True)

natm_blk = a.natm_block or mol.natm
n_dm = 3 * natm_blk
rng = np.random.default_rng(0)
mo1 = cp.asarray(rng.standard_normal((n_dm, nmo, nocc)) * 1e-2)
mo1_ao = contract('npo,ip->nio', mo1, mo_coeff)
dm1 = cp.asarray(contract('npi,qi->npq', mo1_ao, orbo * 2), order='C')
transpose_sum(dm1, inplace=True, hermi=1)
dm1 = tag_array(dm1, mo1=mo1_ao, occ_coeff=orbo, symmetrize=1)
print(f'# n_dm {n_dm}  nocc {nocc}', flush=True)

UP = fastdfhess._orig_get_jk
res = {'atoms': mol.natm, 'nao': int(nao), 'nocc': int(nocc),
       'naux': int(cderi.shape[0]), 'n_dm': int(n_dm),
       'cderi_gb': cderi.nbytes / GB}


def call(with_k=True):
    return lambda: fastdfhess._get_jk(dfobj, dm1, mo_coeff, mo_occ, 1, True,
                                      with_k, None)


def upcall(with_k=True):
    return lambda: UP(dfobj, dm1, mo_coeff, mo_occ, 1, True, with_k, None)


def with_arrange(name, with_k=True):
    def fn():
        fastdfhess.ARRANGE = name
        return fastdfhess._get_jk(dfobj, dm1, mo_coeff, mo_occ, 1, True,
                                  with_k, None)
    return fn


variants = {'GPU4PySCF': upcall()}
for name in a.arrangements.split(','):
    if name:
        variants[name] = with_arrange(name)

print('\n## the J+K call, round robin')
best = round_robin(variants, a.reps)
base = best['GPU4PySCF']
for k, v in best.items():
    if isinstance(v, str):
        print(f'  {k:14s} {"RAISED":>10s}   {v}')
    elif isinstance(base, str):
        print(f'  {k:14s} {v * 1e3:10.2f} ms   (GPU4PySCF raised)')
    else:
        print(f'  {k:14s} {v * 1e3:10.2f} ms   {base / v:5.2f}x')
res['jk'] = {k: v for k, v in best.items()}
fastdfhess.ARRANGE = 'fused'

if a.j_only:
    print('\n## the J-only call (a pure functional\'s whole two-electron CPHF)')
    jv = {'GPU4PySCF': upcall(False), 'fastdfhess': call(False)}

    def tensor_pass():
        fastdfhess._JPASS = 'tensor'
        out = fastdfhess._get_jk(dfobj, dm1, mo_coeff, mo_occ, 1, True,
                                 False, None)
        fastdfhess._JPASS = 'blas'
        return out
    jv['fastdfhess (tensor)'] = tensor_pass
    bestj = round_robin(jv, a.reps)
    basej = bestj['GPU4PySCF']
    for k, v in bestj.items():
        if isinstance(v, str) or isinstance(basej, str):
            print(f'  {k:22s} {v if isinstance(v, str) else f"{v * 1e3:.2f} ms"}')
        else:
            print(f'  {k:22s} {v * 1e3:10.2f} ms   {basej / v:5.2f}x')
    res['j_only'] = bestj

if a.knobs:
    print('\n## one knob at a time, against the default')
    knobs = {}

    def _knob(attr, value):
        def fn():
            old = getattr(fastdfhess, attr)
            setattr(fastdfhess, attr, value)
            try:
                return fastdfhess._get_jk(dfobj, dm1, mo_coeff, mo_occ, 1,
                                          True, True, None)
            finally:
                setattr(fastdfhess, attr, old)
        return fn

    knobs['default'] = call()
    knobs['BALANCE=0'] = _knob('BALANCE', False)
    knobs['POOLMEM=0'] = _knob('POOLMEM', False)
    bestk = round_robin(knobs, a.reps)
    for k, v in bestk.items():
        if isinstance(v, str) or isinstance(bestk['default'], str):
            print(f'  {k:14s} {v if isinstance(v, str) else f"{v * 1e3:.2f} ms"}')
        else:
            print(f'  {k:14s} {v * 1e3:10.2f} ms   '
                  f'{bestk["default"] / v:5.2f}x the default')
    print(f'  shapes the default picks: blksize {fastdfhess.STATS["blksize"]}'
          f'  dm_batch {fastdfhess.STATS["dm_batch"]}  of n_dm {n_dm}')
    res['knobs'] = bestk
    res['blksize'] = fastdfhess.STATS['blksize']
    res['dm_batch'] = fastdfhess.STATS['dm_batch']

if a.whole:
    print('\n## the whole Hessian, patched and unpatched, twice (§5.3)')
    rows = []
    for rnd in range(2):
        for off in (True, False):
            fastdfhess._OFF = off
            dfobj.reset()
            cp.get_default_memory_pool().free_all_blocks()
            mfh = dft.RKS(mol, xc=a.xc).density_fit(auxbasis=a.auxbasis)
            mfh.grids.level = a.grids
            mfh.conv_tol = a.conv_tol
            mfh.conv_tol_cpscf = a.conv_tol_cpscf
            mfh.verbose = 0
            mfh.kernel()
            _sync()
            t = time.perf_counter()
            try:
                h = mfh.Hessian().kernel()
                _sync()
            except Exception as exc:                          # noqa: BLE001
                print(f'  round {rnd}  '
                      f'{"GPU4PySCF " if off else "fastdfhess"}  RAISED  '
                      f'{type(exc).__name__}: {str(exc)[:80]}', flush=True)
                rows.append({'round': rnd, 'off': off,
                             'error': type(exc).__name__})
                cp.get_default_memory_pool().free_all_blocks()
                del mfh
                continue
            dt = time.perf_counter() - t
            tr = float(np.trace(cp.asnumpy(cp.asarray(h)).reshape(
                3 * mol.natm, -1)))
            rows.append({'round': rnd, 'off': off, 't': dt, 'trace': tr})
            print(f'  round {rnd}  {"GPU4PySCF " if off else "fastdfhess"}'
                  f'  {dt:9.2f} s   trace {tr:.10f}', flush=True)
            del mfh, h
            cp.get_default_memory_pool().free_all_blocks()
    fastdfhess._OFF = False
    second = [r for r in rows if r['round'] == 1 and 't' in r]
    if len(second) == 2:
        up = [r for r in second if r['off']][0]
        pt = [r for r in second if not r['off']][0]
        print(f'  -> {up["t"] / pt["t"]:.2f}x on the whole Hessian '
              f'(second round), trace differs by '
              f'{abs(up["trace"] - pt["trace"]):.2e}')
    res['whole'] = rows
    res['stats'] = fastdfhess.stats()

print('\n@@JSON ' + json.dumps(res))
