"""The VV10 response term A/B, both implementations in one process.

The term is called once per CPKS/CPHF (or Davidson) iteration with the same
ground-state orbitals, so the number that matters is the cost of the *second*
and later calls, not the first.  Both are reported.

``--spin`` picks the unrestricted reference, which goes in through
``hessian.uks.nr_uks_fnlc_mo`` -- the function that sums the two perturbed
densities and forwards -- rather than through ``nr_rks_fnlc_mo`` directly, so
that the module-level binding this round had to patch is on the timed path.

    python benchmarks/fnlc_bench.py --xyz benchmarks/molecules/penicillinV.xyz \
        --basis def2-svp --xc wb97m-v --nset 1,3,6
    python benchmarks/fnlc_bench.py --xyz benchmarks/molecules/tmc_cucl2.xyz \
        --spin 1 --basis def2-svp --xc wb97m-v --nset 1,3
"""
import json
import time

import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--nset', default='1,3,6')
ap.add_argument('--in-mo', action='store_true',
                help="ask for the response in the MO basis, which is what a "
                     "Hessian's CPHF solve does")
ap.add_argument('--budget', default='',
                help='comma separated FASTFNLC_BUDGET values to sweep after '
                     'the SCF, e.g. 0,1e-14,1e-12.  The grid trim shrinks '
                     'both O(ngrids^2) sums quadratically and perturbs the '
                     'answer, so the sweep reports the point count, the time '
                     'and the error together.  One SCF serves every budget.')
a = ap.parse_args()
apply_patch('fastfnlc')

from gpu4pyscf import dft                                   # noqa: E402
from gpu4pyscf.hessian import rks as HR                     # noqa: E402
from gpu4pyscf.hessian import uks as HU                     # noqa: E402
from electrolites import fastfnlc                           # noqa: E402

_sync = cp.cuda.Stream.null.synchronize


def timeit(fn, reps):
    fn()                             # warm up the allocator and any JIT
    fastfnlc._CACHE.clear()          # ... but not the ground-state cache
    _sync()
    t = time.perf_counter()
    fn()
    _sync()
    t_first = time.perf_counter() - t
    ts = []
    for _ in range(reps):
        _sync()
        t = time.perf_counter()
        out = fn()
        _sync()
        ts.append(time.perf_counter() - t)
    return t_first, min(ts), out


mol = build_mol(a)
tag = f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}/L{a.grids}'
print(f'# {tag}  natm={mol.natm} nao={mol.nao}', flush=True)
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc)
mf.grids.level = a.grids
mf.nlcgrids.level = a.grids
mf.conv_tol = a.conv_tol
e = mf.kernel()
print(f'# SCF {e:.9f}  converged={mf.converged}', flush=True)
print(f'# nlcgrids {mf.nlcgrids.coords.shape[0]} points', flush=True)

mo, occ = cp.asarray(mf.mo_coeff), cp.asarray(mf.mo_occ)
en = cp.asarray(mf.mo_energy)
uks = mo.ndim == 3
with mol.with_common_orig((0, 0, 0)):
    h1 = cp.asarray(mol.intor('int1e_r'))


def _basis(mo_s, occ_s, en_s):
    """Up to 3 + 16 perturbed-orbital vectors for one spin channel."""
    mocc, mvir = mo_s[:, occ_s > 0], mo_s[:, occ_s == 0]
    de = en_s[occ_s == 0][:, None] - en_s[occ_s > 0][None, :]
    xs = [-(mvir.T @ h1[i] @ mocc) / de for i in range(3)]
    order = cp.argsort(de.ravel())
    for k in range(16):
        x = cp.zeros(de.shape)
        x.ravel()[order[k % de.size]] = 1.0
        xs.append(x)
    return mocc, mvir, xs


channels = ([_basis(mo[s], occ[s], en[s]) for s in range(2)] if uks
            else [_basis(mo, occ, en)])


def _dm1(nset):
    """The perturbed densities in the shape the entry point expects.

    Restricted: ``(nset, nao, nao)`` with the factor 2 for double occupancy.
    Unrestricted: ``(2, nset, nao, nao)``, which ``nr_uks_fnlc_mo`` sums.
    """
    out = []
    for mocc, mvir, xs in channels:
        d = cp.asarray([mvir @ x @ mocc.T for x in xs[:nset]])
        if not uks:
            d = 2 * d
        out.append(cp.asarray(d + d.transpose(0, 2, 1), order='C'))
    return cp.asarray(out, order='C') if uks else out[0]


def _flat(v):
    """One flat array, whether the answer is one matrix or a spin pair."""
    if isinstance(v, tuple):
        return cp.concatenate([cp.asarray(x).ravel() for x in v])
    return cp.asarray(v).ravel()


def _call(patched):
    if not uks:
        return (fastfnlc._nr_rks_fnlc_mo if patched
                else HR._nr_rks_fnlc_mo_orig)
    def go(*args):
        HU.nr_rks_fnlc_mo = (fastfnlc._dispatch if patched
                             else HR._nr_rks_fnlc_mo_orig)
        try:
            return HU.nr_uks_fnlc_mo(*args)
        finally:
            HU.nr_rks_fnlc_mo = fastfnlc._dispatch
    return go


rows = []
print(f'\n{"nset":>5s} {"stock/s":>9s} {"cold/s":>9s} {"cached/s":>9s} '
      f'{"cold":>7s} {"cached":>8s} {"rel.err":>9s}')
print('-' * 64)
for nset in [int(x) for x in a.nset.split(',')]:
    args = (mf, mol, mo, occ, _dm1(nset), a.in_mo)
    ref_fn, new_fn = _call(False), _call(True)
    _, t_ref, v_ref = timeit(lambda: ref_fn(*args), a.reps)
    t1, t_new, v_new = timeit(lambda: new_fn(*args), a.reps)
    v_ref, v_new = _flat(v_ref), _flat(v_new)
    scale = float(cp.abs(v_ref).max()) or 1.0
    err = float(cp.abs(v_new - v_ref).max()) / scale
    print(f'{nset:5d} {t_ref:9.4f} {t1:9.4f} {t_new:9.4f} '
          f'{t_ref/t1:6.2f}x {t_ref/t_new:7.2f}x {err:9.2e}', flush=True)
    rows.append(dict(nset=nset, stock=t_ref, first=t1, cached=t_new, err=err))

# --- the grid trim, one SCF for every budget -------------------------------
trim = []
if a.budget:
    nset = max(int(x) for x in a.nset.split(','))
    args = (mf, mol, mo, occ, _dm1(nset), a.in_mo)
    ref_fn, new_fn = _call(False), _call(True)
    _, t_ref, v_ref = timeit(lambda: ref_fn(*args), a.reps)
    v_ref = _flat(v_ref)
    scale = float(cp.abs(v_ref).max()) or 1.0
    print(f'\n{"budget":>9s} {"points":>9s} {"dropped":>8s} {"cached/s":>9s} '
          f'{"vs stock":>9s} {"rel.err":>9s}   (nset = {nset})')
    print('-' * 62)
    for b in [float(x) for x in a.budget.split(',')]:
        fastfnlc.BUDGET = b
        fastfnlc._CACHE.clear()
        _, t_new, v_new = timeit(lambda: new_fn(*args), a.reps)
        ng = fastfnlc._CACHE[next(iter(fastfnlc._CACHE))][2]['ngrids'] \
            if fastfnlc._CACHE else 0
        ngf = mf.nlcgrids.coords.shape[0]
        err = float(cp.abs(_flat(v_new) - v_ref).max()) / scale
        print(f'{b:9.0e} {ng:9d} {100*(1-ng/ngf):7.1f}% {t_new:9.4f} '
              f'{t_ref/t_new:8.2f}x {err:9.2e}', flush=True)
        trim.append(dict(budget=b, ngrids=ng, ngrids_full=ngf, t=t_new,
                         stock=t_ref, err=err))
    fastfnlc.BUDGET = 0.0
    fastfnlc._CACHE.clear()

print('\n' + json.dumps({'tag': tag, 'in_mo': a.in_mo, 'natm': mol.natm,
                         'nao': int(mol.nao), 'spin': int(mol.spin),
                         'rows': rows, 'trim': trim}))
