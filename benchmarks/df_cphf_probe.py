"""The CPHF of a density-fitted Hessian, piece by piece (W44).

``solve_mo1`` is 94 % of a 113-atom UKS Hessian (`docs/README_fastdfhessip8.md`
§3.1), and `benchmarks/df_hess_anatomy.py` sees three rows of it: the
exchange build (`_get_jk`), the XC response and `cache_xc_kernel`.  This one
runs the SCF and `make_h1` once (or loads both from `--cache`), then only
the CPHF, with every piece of it on its own row:

* ``krylov``         the Krylov solve, of which ``.. fx`` is its operator
                     calls and ``.. own`` everything else (the projections,
                     the QR, the final subspace solve);
* ``fx``             `gen_vind`'s closure, of which ``.. veff_resp_mo`` is
                     the response and ``.. own`` the MO/AO transforms and the
                     AO densities it builds first;
* ``_get_jk``, ``fxc``, ``cache_xc_kernel`` inside the response;
* the memory: the allocator's high-water mark over the whole call and inside
  each ``fx`` call, per right-hand side.

    python benchmarks/df_cphf_probe.py --xyz benchmarks/molecules/taxol.xyz \\
        --charge 1 --spin 1 --basis def2-svp --xc b3lyp --patch all \\
        --cache $SCRATCH/cphf_taxp --variants stock,ours

``stock`` is `hessian/{rhf,uhf}.py:solve_mo1` as GPU4PySCF has it, ``ours``
whatever is installed after `--patch`; every variant after the first is
compared with the first (``mo1`` and ``mo_e1``).
"""
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--conv-tol-cpscf', type=float, default=1e-6)
ap.add_argument('--variants', default='stock,ours')
ap.add_argument('--env', default='',
                help="extra variants as 'label:VAR=val;VAR2=val2,...' -- each "
                     "runs 'ours' with those switches set")
ap.add_argument('--cache', default='',
                help='directory: the SCF orbitals and h1mo are read from it '
                     'if present, written to it otherwise')
ap.add_argument('--ballast', type=float, default=0,
                help='GB held on the device for the whole run, before the SCF')
ap.add_argument('--free-after', type=float, default=0,
                help='GB left free after the SCF and h1mo, by a ballast '
                     'allocated then')
ap.add_argument('--h1-host', default='auto', choices=('auto', 'host', 'device'),
                help="where h1mo lives when the CPHF starts; 'auto' is "
                     "`hess_elec`'s rule (host if 10x its size does not fit)")
ap.add_argument('--verbose-cphf', type=int, default=4)
a = ap.parse_args()

if a.ballast:
    _ballast = cp.empty(int(a.ballast * 1e9) // 8)
    _ballast[:] = 0
    print(f'# ballast {a.ballast} GB, '
          f'{cp.cuda.runtime.memGetInfo()[0] / 1e9:.1f} GB free', flush=True)

from gpu4pyscf.hessian import rhf as hess_rhf                # noqa: E402
from gpu4pyscf.hessian import uhf as hess_uhf                # noqa: E402

STOCK = {'r': hess_rhf.solve_mo1, 'u': hess_uhf.solve_mo1}
patched = apply_patch(a.patch)

from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.lib import logger                             # noqa: E402
from gpu4pyscf.lib.cupy_helper import get_avail_mem          # noqa: E402

_sync = cp.cuda.Stream.null.synchronize
pool = cp.get_default_memory_pool()

# the device's high-water mark.  GPU4PySCF's allocator sends anything over
# 100 MB straight to cudaMalloc, past CuPy's pool (`set_conditional_mempool_
# malloc`), so the pool's counters do not see the arrays that matter; this
# wraps whatever allocator is installed -- without changing which one runs,
# which would change the timings -- and reads cudaMemGetInfo after every
# large allocation
_PEAK = [0]
_alloc0 = cp.cuda.get_allocator()
_TOTAL = cp.cuda.runtime.memGetInfo()[1]


def _in_use():
    return _TOTAL - cp.cuda.runtime.memGetInfo()[0]


def _malloc(nbytes):
    p = _alloc0(nbytes)
    if nbytes > (1 << 26):
        u = _in_use()
        if u > _PEAK[0]:
            _PEAK[0] = u
    return p


cp.cuda.set_allocator(_malloc)

mol = build_mol(a)
natm = mol.natm
uks = mol.spin != 0
mf = (dft.UKS if uks else dft.RKS)(mol, xc=a.xc).density_fit(
    auxbasis=a.auxbasis)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.conv_tol_cpscf = a.conv_tol_cpscf
mf.verbose = 0

cache = a.cache
tag = f'{os.path.basename(a.xyz)}_{a.basis}_{a.xc}_{a.charge}_{a.spin}'
fn_scf = os.path.join(cache, f'{tag}.scf.npz') if cache else ''
fn_h1 = os.path.join(cache, f'{tag}.h1mo.npz') if cache else ''
t0 = time.perf_counter()
if fn_scf and os.path.exists(fn_scf):
    z = np.load(fn_scf)
    mf.mo_coeff = cp.asarray(z['mo_coeff'])
    mf.mo_occ = cp.asarray(z['mo_occ'])
    mf.mo_energy = cp.asarray(z['mo_energy'])
    mf.e_tot = float(z['e_tot'])
    mf.converged = True
    mf.grids.build(with_non0tab=False, sort_grids=True)
    print(f'# SCF from {fn_scf}  E = {mf.e_tot:.10f}', flush=True)
else:
    mf.kernel()
    if fn_scf:
        os.makedirs(cache, exist_ok=True)
        _tmp = f'{fn_scf}.{os.getpid()}.npz'
        np.savez(_tmp, mo_coeff=cp.asnumpy(mf.mo_coeff),
                 mo_occ=cp.asnumpy(mf.mo_occ),
                 mo_energy=cp.asnumpy(mf.mo_energy), e_tot=mf.e_tot)
        os.replace(_tmp, fn_scf)    # atomic: jobs may share the cache
    print(f'# SCF {time.perf_counter() - t0:.1f} s  E = {mf.e_tot:.10f}',
          flush=True)
print(f'# {natm} atoms  nao {mol.nao}  {"UKS" if uks else "RKS"}  '
      f'{a.basis}  {a.xc}', flush=True)

mo_energy = cp.asarray(mf.mo_energy)
mo_coeff = cp.asarray(mf.mo_coeff)
mo_occ = cp.asarray(mf.mo_occ)

_occ = mo_occ if uks else mo_occ[None]
_PER_SET = [int(sum(mo_coeff.shape[-1] * int((o > 0).sum()) for o in _occ))]
hess = mf.Hessian()
hess.auxbasis_response = 2
log = logger.new_logger(hess, a.verbose_cphf)
t0 = time.perf_counter()
if fn_h1 and os.path.exists(fn_h1):
    z = np.load(fn_h1)
    h1mo = (z['a'], z['b']) if uks else z['a']
    print(f'# h1mo from {fn_h1}', flush=True)
else:
    mf.with_df.reset()   # as `partial_hess_elec`, which runs first, does
    pool.free_all_blocks()
    h1mo = hess.make_h1(mo_coeff, mo_occ, None, None, log)
    h1mo = tuple(cp.asnumpy(x) for x in h1mo) if uks else cp.asnumpy(h1mo)
    if fn_h1:
        os.makedirs(cache, exist_ok=True)
        _tmp = f'{fn_h1}.{os.getpid()}.npz'
        if uks:
            np.savez(_tmp, a=h1mo[0], b=h1mo[1])
        else:
            np.savez(_tmp, a=h1mo)
        os.replace(_tmp, fn_h1)
    print(f'# make_h1 {time.perf_counter() - t0:.1f} s', flush=True)
mf.with_df.reset()
pool.free_all_blocks()


def _h1_for_run():
    arrs = h1mo if uks else (h1mo,)
    where = a.h1_host
    if where == 'auto':
        where = ('host' if arrs[0].size * 8 * 10 > get_avail_mem()
                 else 'device')
    out = tuple(cp.asarray(x) if where == 'device' else x for x in arrs)
    return (out if uks else out[0]), where


if a.free_after:
    _free = cp.cuda.runtime.memGetInfo()[0]
    _ballast2 = cp.empty(max(0, int(_free - a.free_after * 1e9)) // 8)
    _ballast2[:] = 0
    print(f'# ballast after the SCF {_ballast2.nbytes / 1e9:.1f} GB, '
          f'{cp.cuda.runtime.memGetInfo()[0] / 1e9:.1f} GB free', flush=True)

T, N, M = OrderedDict(), OrderedDict(), OrderedDict()
FX = []          # per fx call: nset, seconds, peak GB above the entry


def _reset():
    for d in (T, N):
        for k in d:
            d[k] = 0
    M.clear()
    FX.clear()


def _timed(label, fn, fxrow=False):
    T.setdefault(label, 0.0)
    N.setdefault(label, 0)

    def timed(*args, **kw):
        _sync()
        used0 = _in_use()
        peak0 = _PEAK[0]
        _PEAK[0] = used0
        t = time.perf_counter()
        out = fn(*args, **kw)
        _sync()
        dt = time.perf_counter() - t
        T[label] += dt
        N[label] += 1
        pk = _PEAK[0]
        M[label] = max(M.get(label, 0), pk - used0)
        _PEAK[0] = max(peak0, pk)
        if fxrow:
            n = args[0].size // _PER_SET[0]
            FX.append((n, dt, (pk - used0) / 1e9, used0 / 1e9))
        return out
    return timed


# module globals and class attributes, looked up at call time
from gpu4pyscf.df.hessian import rhf as dfhess_rhf           # noqa: E402
from gpu4pyscf.dft import numint                             # noqa: E402
from gpu4pyscf.lib import cupy_helper                        # noqa: E402

dfhess_rhf._get_jk = _timed('.. .. _get_jk', dfhess_rhf._get_jk)
numint.NumInt.cache_xc_kernel = _timed('.. .. cache_xc_kernel',
                                       numint.NumInt.cache_xc_kernel)
if 'electrolites.fastxcfxc' in sys.modules:
    from electrolites import fastxcfxc
    for nm in ('_fxc_u_groups', '_fxc_blocked'):
        if hasattr(fastxcfxc, nm):
            setattr(fastxcfxc, nm, _timed(f'.. .. fxc ({nm})',
                                          getattr(fastxcfxc, nm)))
            break
numint.NumInt.nr_uks_fxc = _timed('.. .. nr_uks_fxc', numint.NumInt.nr_uks_fxc)
numint.NumInt.nr_rks_fxc = _timed('.. .. nr_rks_fxc', numint.NumInt.nr_rks_fxc)
if not uks:
    from gpu4pyscf.hessian import rks as hess_rks
    hess_rks.nr_rks_fxc_mo = _timed('.. .. nr_rks_fxc_mo',
                                    hess_rks.nr_rks_fxc_mo)
_krylov_mods = [hess_rhf, hess_uhf]
if 'electrolites.fastcphf' in sys.modules:
    _krylov_mods.append(sys.modules['electrolites.fastcphf'])
_krylov_orig = {m: m.krylov for m in _krylov_mods}


def _variants():
    out = []
    for v in [x for x in a.variants.split(',') if x]:
        out.append((v, v, {}))
    for spec in [x for x in a.env.split(',') if x]:
        label, env = spec.split(':', 1)
        out.append((label, 'ours',
                    dict(kv.split('=', 1) for kv in env.split(';') if kv)))
    return out


results, ref = {}, None
for label, kind, env in _variants():
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    _reset()
    key = 'u' if uks else 'r'
    solver = STOCK[key] if kind == 'stock' else (
        hess_uhf.solve_mo1 if uks else hess_rhf.solve_mo1)
    fx0 = hess.gen_vind(mo_coeff, mo_occ)
    fx = _timed('.. fx', fx0, fxrow=True)
    vr0 = hess.get_veff_resp_mo
    hess.get_veff_resp_mo = _timed('.. .. veff_resp_mo', vr0)
    for m in _krylov_mods:
        m.krylov = _timed('krylov', _krylov_orig[m])
    h1, where = _h1_for_run()
    pool.free_all_blocks()
    _sync()
    used0 = _in_use()
    _PEAK[0] = used0
    free0 = cp.cuda.runtime.memGetInfo()[0]
    t = time.perf_counter()
    try:
        mo1, e1 = solver(mf, mo_energy, mo_coeff, mo_occ, h1, fx, None, 4000,
                         log)
        err = None
    except Exception as exc:                             # noqa: BLE001
        mo1 = e1 = None
        err = f'{type(exc).__name__}: {str(exc)[:200]}'
    _sync()
    dt = time.perf_counter() - t
    peak = (_PEAK[0] - used0) / 1e9
    del hess.get_veff_resp_mo
    for m in _krylov_mods:
        m.krylov = _krylov_orig[m]
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    print(f'\n## {label}   solve_mo1 {dt:8.2f} s   h1mo on {where}   '
          f'{free0 / 1e9:.1f} GB free at the start   peak +{peak:.1f} GB'
          + (f'   ERROR {err}' if err else ''), flush=True)
    for k in T:
        if N[k]:
            print(f'  {k:28s} {T[k]:9.2f} s  {N[k]:4d} calls   '
                  f'peak +{M.get(k, 0) / 1e9:6.2f} GB')
    fxs = sum(x[1] for x in FX)
    kry = T.get('krylov', 0)
    fx_in_kry = sum(x[1] for x in FX[:-1]) if FX else 0
    print(f'  derived: krylov own {kry - fx_in_kry:8.2f} s   '
          f'fx own {fxs - T.get(".. .. veff_resp_mo", 0):8.2f} s   '
          f'solve_mo1 outside krylov and fx '
          f'{dt - kry - (FX[-1][1] if FX else 0):8.2f} s')
    for i, (n, s, pk, u0) in enumerate(FX):
        print(f'    fx {i:2d}  nset {n:4d}  {s:8.2f} s  {s / max(n, 1):7.3f} '
              f's/set  peak +{pk:6.2f} GB ({pk / max(n, 1) * 1e3:7.1f} MB/set)'
              f'  in use at entry {u0:6.2f} GB')
    row = {'t': dt, 'peak_gb': peak, 'stages': dict(T), 'calls': dict(N),
           'fx': FX[:], 'err': err}
    if mo1 is not None:
        m1 = [np.asarray(cp.asnumpy(x)) for x in (mo1 if uks else (mo1,))]
        ee = [np.asarray(cp.asnumpy(x)) for x in (e1 if uks else (e1,))]
        if ref is None:
            ref = (label, m1, ee)
        else:
            d1 = max(abs(x - y).max() for x, y in zip(m1, ref[1]))
            d2 = max(abs(x - y).max() for x, y in zip(ee, ref[2]))
            s1 = max(abs(y).max() for y in ref[1])
            print(f'  against {ref[0]}: mo1 max|d| {d1:.3e} (rel {d1 / s1:.2e})'
                  f'   mo_e1 max|d| {d2:.3e}')
            row['d_mo1'] = d1
            row['d_e1'] = d2
    results[label] = row
    mo1 = e1 = h1 = None
    pool.free_all_blocks()

try:
    import electrolites
    stats = {}
    for m in ('fastcphf', 'fastdfhess', 'fastxcfxc'):
        mod = sys.modules.get(f'electrolites.{m}')
        if mod is not None and hasattr(mod, 'stats'):
            stats[m] = mod.stats()
            print(f'# {m}', stats[m])
except ImportError:
    stats = {}
print('\n@@JSON ' + json.dumps({'natm': natm, 'nao': int(mol.nao),
                                'variants': results, 'stats': stats},
                               default=str))
