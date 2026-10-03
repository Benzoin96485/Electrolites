"""The three density-fitted Hessian terms W8's anatomy found and nothing covered.

`docs/DF_WORK_LIST.md` **W16**, **W17** and **W18**:

  * ``fxc``  ``hessian/rks.py:nr_rks_fxc_mo`` -- the CPHF's grid half, one call
             per Krylov cycle, on the right-hand sides a CPHF cycle actually
             sees (``hessian/rhf.py:gen_vind``'s tagged density);
  * ``ip2``  ``df/hessian/rhf.py:_jk_energy_per_atom`` -- the two-electron
             half of ``partial_hess_elec``;
  * ``h1``   ``df/hessian/rhf.py:_get_veff`` -- the two-electron half of
             ``make_h1``.

Each term is called on one converged DF-SCF, the variants **round robin**
(`docs/DF_WORK_LIST.md` §5.2), and every variant is compared element-wise
against GPU4PySCF's own function in the same process.  ``stock`` is always
upstream's function object, saved before anything patched it -- this script
imports the modules it times inside the branch that times them (§5.13).

    python benchmarks/df_hess_terms_probe.py \\
        --xyz benchmarks/molecules/penicillinV.xyz --basis def2-svp \\
        --xc b3lyp --terms fxc,ip2,h1 --variants stock,ours

``--nvtx`` wraps every timed call in an NVTX range and brackets the timed
region with ``cudaProfilerStart/Stop``, so

    nsys profile -t cuda,nvtx -c cudaProfilerApi -o $SCRATCH/p python ...

captures only the calls and not the SCF in front of them.

An open-shell molecule (``--spin`` not 0, or ``--uks``) runs a UKS SCF and
times the unrestricted twins in ``df/hessian/uhf.py`` instead (W25); the
``fxc`` term is restricted-only here.
"""
import json
import time

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--terms', default='fxc,ip2,h1')
ap.add_argument('--variants', default='stock,ours',
                help="'stock' is upstream's function; 'ours' the module's")
ap.add_argument('--natm-block', type=int, default=0,
                help='atoms per CPHF right-hand-side block for the fxc term; '
                     '0 = all of them')
ap.add_argument('--nvtx', action='store_true')
ap.add_argument('--no-check', action='store_true')
ap.add_argument('--uks', action='store_true',
                help='UKS even for a closed shell (implied by --spin)')
ap.add_argument('--env', default='',
                help="extra variants as 'label:VAR=val;VAR2=val2,...' -- each "
                     "runs 'ours' with those module switches set")
ap.add_argument('--ballast', type=float, default=0,
                help='GB held on the device for the whole run, before the '
                     'SCF: 40 on an 80 GB card is the memory of a 40 GB one '
                     '(W35; FASTDFHESSIP_MEM only caps the sizing)')
ap.add_argument('--free-after', type=float, default=0,
                help='GB left free after the SCF, by a ballast allocated '
                     'then: the terms at a 40 GB card\'s memory (44.6) for '
                     'a molecule whose SCF does not fit one (W38)')
ap.add_argument('--keep-cderi', action='store_true',
                help="keep the SCF's cderi on the device while the ip2 and h1 "
                     "terms run.  Upstream's `partial_hess_elec` and "
                     "`make_h1` call `with_df.reset()` before them, so a "
                     "Hessian does not; W40's 40 GB rows were taken with "
                     "it kept (14.3 GB at 113 atoms, W42)")
ap.add_argument('--save-out', default='',
                help='save each term\'s first-variant result to <path>.<term>.npy')
ap.add_argument('--ref-out', default='',
                help='check every variant against <path>.<term>.npy, saved by '
                     '--save-out in another process (e.g. on the full card)')
ap.add_argument('--prefree-env', default=None,
                help="with --free-after: each term's reference is 'ours' "
                     "with these switches ('VAR=val;...'), computed in this "
                     "process on this SCF *before* the ballast -- e.g. "
                     "'FASTDFHESSIP_H1UBATCH=0', upstream's one batch, which "
                     "does not fit after it (W38)")
a = ap.parse_args()
if a.ballast:
    _ballast = cp.empty(int(a.ballast * 1e9) // 8)
    print(f'# ballast {a.ballast} GB, '
          f'{cp.cuda.runtime.memGetInfo()[0] / 1e9:.1f} GB free', flush=True)
apply_patch(a.patch)

from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.lib.cupy_helper import (                      # noqa: E402
    contract, tag_array, transpose_sum)
from gpu4pyscf.df.hessian import rhf as dfhess_rhf           # noqa: E402
from gpu4pyscf.df.hessian import uhf as dfhess_uhf           # noqa: E402
from gpu4pyscf.hessian import rks as rks_hess                # noqa: E402

_sync = cp.cuda.Stream.null.synchronize
terms = [t for t in a.terms.split(',') if t]
variants = [v for v in a.variants.split(',') if v]

uks = a.uks or a.spin != 0
if uks and 'fxc' in terms:
    raise SystemExit('the fxc term is restricted-only; drop it from --terms')

# upstream's function objects, before anything below can patch them
STOCK = {
    'fxc': getattr(rks_hess, '_nr_rks_fxc_mo_orig', rks_hess.nr_rks_fxc_mo),
    'ip2': getattr(dfhess_rhf, '_jk_energy_per_atom_orig',
                   dfhess_rhf._jk_energy_per_atom),
    'h1': getattr(dfhess_rhf, '_get_veff_orig', dfhess_rhf._get_veff),
}
if uks:
    STOCK = {
        'ip2': getattr(dfhess_uhf, '_jk_energy_per_atom_orig',
                       dfhess_uhf._jk_energy_per_atom),
        'h1': getattr(dfhess_uhf, '_get_veff_orig', dfhess_uhf._get_veff),
    }

mol = build_mol(a)
natm = mol.natm
mf = (dft.UKS if uks else dft.RKS)(mol, xc=a.xc).density_fit(
    auxbasis=a.auxbasis)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.verbose = 0
t0 = time.perf_counter()
e = mf.kernel()
print(f'# {natm} atoms  nao {mol.nao}  basis {a.basis}  xc {a.xc}  '
      f'{"UKS spin " + str(mol.spin) if uks else "RKS"}')
print(f'# DF-SCF {time.perf_counter() - t0:.2f} s   E = {e:.10f}', flush=True)
PREFREE = {}


def _prefree_refs():
    # called once the term closures exist; see --prefree-env
    env = dict(x.split('=', 1) for x in a.prefree_env.split(';') if x)
    saved = _set_env(env)
    try:
        for term in terms:
            PREFREE[term] = cp.asnumpy(SETUP[term]()(_ours(term)))
    finally:
        _restore(saved)
    print(f'# references before the ballast, {env}', flush=True)


def _free_after():
    cp.get_default_memory_pool().free_all_blocks()
    global _ballast2
    _free = cp.cuda.runtime.memGetInfo()[0]
    _ballast2 = cp.empty(max(0, int(_free - a.free_after * 1e9)) // 8)
    _ballast2[:] = 0
    print(f'# ballast after the SCF {_ballast2.nbytes / 1e9:.1f} GB, '
          f'{cp.cuda.runtime.memGetInfo()[0] / 1e9:.1f} GB free', flush=True)

if not a.keep_cderi:
    # as `df/hessian/{rhf,uhf}.py:partial_hess_elec` and `make_h1` do
    # before the terms (`# Release GPU memory`); the terms build their own
    # integral optimiser without a cderi
    mf.with_df.reset()
    cp.get_default_memory_pool().free_all_blocks()

mo_coeff = cp.asarray(mf.mo_coeff)
mo_occ = cp.asarray(mf.mo_occ)
mo_energy = cp.asarray(mf.mo_energy)
nao, nmo = mo_coeff.shape[-2:]
if uks:
    nocc = int((mo_occ[0] > 0).sum())
else:
    orbo = mo_coeff[:, mo_occ > 0]
    nocc = orbo.shape[1]
hess = mf.Hessian()
ni = mf._numint
omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
print(f'# nocc {nocc}  nmo {nmo}  hyb {hyb}  omega {omega}', flush=True)

results = {'atoms': natm, 'nao': int(nao), 'nocc': int(nocc),
           'basis': a.basis, 'xc': a.xc, 'terms': {}}


def _ours(term):
    """The module's entry point for one term, imported only when asked."""
    if term == 'fxc':
        from electrolites import fastxcfxc
        return fastxcfxc._nr_rks_fxc_mo
    from electrolites import fastdfhessip
    if uks:
        from electrolites import _dfhessip_u
        return {'ip2': _dfhessip_u._jk_energy_per_atom,
                'h1': _dfhessip_u._get_veff}[term]
    return {'ip2': fastdfhessip._jk_energy_per_atom,
            'h1': fastdfhessip._get_veff}[term]


def _spmult_hack(fn, mult):
    """Upstream's function with every ``nsp_per_block*4`` aggregation it
    makes rescaled to ``nsp_per_block*mult`` -- a sweep, not a variant: it
    also rescales the one aggregation in there that is not ``*4``
    (``_int2c2e_ip2_per_atom``'s), which changes its launch and not its
    answer."""
    from gpu4pyscf.gto.mole import SortedMole
    orig = SortedMole.aggregate_shl_pairs

    def agg(this, cache=None, nsp_per_block=512):
        if not isinstance(nsp_per_block, (int, np.integer)):
            nsp_per_block = np.maximum(1, nsp_per_block * mult // 4)
        return orig(this, cache, nsp_per_block)

    def wrapped(*args, **kw):
        SortedMole.aggregate_shl_pairs = agg
        try:
            return fn(*args, **kw)
        finally:
            SortedMole.aggregate_shl_pairs = orig
    return wrapped


def _variants(term):
    out = []
    for v in variants:
        if v == 'stock':
            out.append(('stock', STOCK[term], {}))
        elif v.startswith('stock_sp'):
            out.append((v, _spmult_hack(STOCK[term], int(v[8:])), {}))
        elif v == 'ours':
            out.append(('ours', _ours(term), {}))
    for spec in [s for s in a.env.split(',') if s]:
        label, kv = spec.split(':', 1)
        env = dict(x.split('=', 1) for x in kv.split(';') if x)
        out.append((label, _ours(term), env))
    return out


def _set_env(env):
    """Module switches are read at call time through ``_env``."""
    import os
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    return saved


def _restore(saved):
    import os
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# --------------------------------------------------------------------------
# the calls, one closure per term
# --------------------------------------------------------------------------
def setup_fxc():
    grids = getattr(mf, 'cphf_grids', None) or mf.grids
    if grids.coords is None:
        grids.build(mol=mol, with_non0tab=False, sort_grids=True)
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, mf.xc, mo_coeff, mo_occ, 0)
    nblk = a.natm_block or natm
    n_dm = 3 * nblk
    rng = np.random.default_rng(0)
    mo1 = cp.asarray(rng.standard_normal((n_dm, nmo, nocc)) * 1e-2)
    mo1_ao = contract('npo,ip->nio', mo1, mo_coeff)
    dm1 = cp.asarray(contract('npi,qi->npq', mo1_ao, orbo * 2), order='C')
    transpose_sum(dm1, inplace=True, hermi=1)
    dm1 = tag_array(dm1, mo1=mo1_ao, occ_coeff=orbo, symmetrize=1)
    print(f'# fxc: n_dm {n_dm}  ngrids {grids.coords.shape[0]}  '
          f'xctype {ni._xc_type(mf.xc)}', flush=True)

    def call(fn):
        return fn(ni, mol, grids, mf.xc, None, dm1, mo_coeff, 0, 1,
                  rho0, vxc, fxc, max_memory=None)
    return call


def setup_ip2():
    wd = mf.with_df
    if wd.intopt is None:
        wd.build(build_cderi=False)
    intopt = wd.intopt
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    def call(fn):
        out = fn(intopt, dm0, 1., hyb)
        if abs(omega) > 1e-10 and abs(alpha - hyb) > 1e-10:
            out = out + fn(intopt, dm0, 0., alpha - hyb, omega)
        return out
    return call


def setup_h1():
    wd = mf.with_df
    if wd.intopt is None:
        wd.build(build_cderi=False)
    intopt = wd.intopt

    def call(fn):
        out = fn(intopt, mo_coeff, mo_occ, 1., hyb)
        if abs(omega) > 1e-10 and abs(alpha - hyb) > 1e-10:
            lr = fn(intopt, mo_coeff, mo_occ, 0., alpha - hyb, omega)
            out = ([x + y for x, y in zip(out, lr)] if uks else out + lr)
        if uks:
            # [vhfa, vhfb] have different shapes: compare them as one vector
            out = cp.concatenate([x.ravel() for x in out])
        return out
    return call


SETUP = {'fxc': setup_fxc, 'ip2': setup_ip2, 'h1': setup_h1}

if a.free_after:
    if a.prefree_env is not None:
        _prefree_refs()
    _free_after()

if a.nvtx:
    # one capture range for the whole of the timed part: nsys's default
    # ends the process at the first cudaProfilerStop
    from cupy.cuda import nvtx, profiler
    profiler.start()

for term in terms:
    print(f'\n## {term}', flush=True)
    call = SETUP[term]()
    vs = _variants(term)
    ref = None
    outs = {}
    # one untimed call each: NVRTC compiles, allocator warm-up, and the
    # reference every other variant is checked against
    for label, fn, env in vs:
        saved = _set_env(env)
        try:
            out = call(fn)
        finally:
            _restore(saved)
        _sync()
        outs[label] = cp.asnumpy(out)
        if label == 'stock' or (ref is None and 'stock' not in variants):
            # without GPU4PySCF's column the first variant is the reference
            ref = outs[label]
    if term in PREFREE:
        ref = PREFREE[term]
    cp.get_default_memory_pool().free_all_blocks()
    ts = {label: [] for label, _, _ in vs}
    for r in range(a.reps):
        for label, fn, env in vs:
            saved = _set_env(env)
            if a.nvtx:
                nvtx.RangePush(f'{term}:{label}:{r}')
            _sync()
            t = time.perf_counter()
            try:
                call(fn)
                _sync()
            finally:
                _restore(saved)
            ts[label].append(time.perf_counter() - t)
            if a.nvtx:
                nvtx.RangePop()
            cp.get_default_memory_pool().free_all_blocks()
    base = min(ts['stock']) if 'stock' in ts else None
    res = {}
    for label, _, _ in vs:
        tmin = min(ts[label])
        line = f'  {label:14s} {tmin * 1e3:10.1f} ms'
        if base:
            line += f'   {base / tmin:5.2f}x'
        err = None
        if (ref is not None and not a.no_check and label != 'stock'
                and outs[label] is not ref):
            d = np.abs(outs[label] - ref).max()
            err = float(d / max(np.abs(ref).max(), 1e-300))
            line += f'   max|d| {d:.2e}  rel {err:.2e}'
        print(line, flush=True)
        res[label] = {'ms': tmin * 1e3, 'all_ms': [x * 1e3 for x in ts[label]],
                      'rel_err': err}
    if a.save_out:
        np.save(f'{a.save_out}.{term}.npy', outs[vs[0][0]])
    if a.ref_out:
        import os
        fn = f'{a.ref_out}.{term}.npy'
        if os.path.exists(fn):
            r0 = np.load(fn)
            for label, _, _ in vs:
                d = np.abs(outs[label] - r0).max()
                e0 = float(d / max(np.abs(r0).max(), 1e-300))
                res[label]['rel_err_saved'] = e0
                print(f'  {label:14s} against {fn}: max|d| {d:.2e}  '
                      f'rel {e0:.2e}', flush=True)
    results['terms'][term] = res
    call = None
    cp.get_default_memory_pool().free_all_blocks()

if a.nvtx:
    profiler.stop()
import sys                                                   # noqa: E402
for name in ('fastdfhessip', 'fastxcfxc'):
    mod = sys.modules.get(f'electrolites.{name}')
    if mod is not None:
        results.setdefault('stats', {})[name] = mod.stats()
        print(f'# {name}.stats() = {mod.stats()}')
print('\n@@JSON ' + json.dumps(results, default=float))
