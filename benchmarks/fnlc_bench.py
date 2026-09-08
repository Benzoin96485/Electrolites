"""`nr_rks_fnlc_mo` A/B, both implementations in one process.

The VV10 response term is called once per CPKS/CPHF iteration with the same
ground-state orbitals, so the number that matters is the cost of the *second*
and later calls, not the first.  Both are reported.

    python benchmarks/fnlc_bench.py --xyz benchmarks/molecules/penicillinV.xyz \
        --basis def2-svp --xc wb97m-v --nset 1,3,6
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
a = ap.parse_args()
apply_patch('fastfnlc')

from gpu4pyscf import dft                                   # noqa: E402
from gpu4pyscf.hessian import rks as HR                     # noqa: E402
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
mf = dft.RKS(mol, xc=a.xc)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
e = mf.kernel()
print(f'# SCF {e:.9f}  converged={mf.converged}', flush=True)
print(f'# nlcgrids {mf.nlcgrids.coords.shape[0]} points', flush=True)

mo, occ = cp.asarray(mf.mo_coeff), cp.asarray(mf.mo_occ)
en = cp.asarray(mf.mo_energy)
mocc, mvir = mo[:, occ > 0], mo[:, occ == 0]
de = en[occ == 0][:, None] - en[occ > 0][None, :]
with mol.with_common_orig((0, 0, 0)):
    h1 = cp.asarray(mol.intor('int1e_r'))
xs = [-(mvir.T @ h1[i] @ mocc) / de for i in range(3)]
order = cp.argsort(de.ravel())
for k in range(16):
    x = cp.zeros(de.shape)
    x.ravel()[order[k]] = 1.0
    xs.append(x)

rows = []
print(f'\n{"nset":>5s} {"stock/s":>9s} {"cold/s":>9s} {"cached/s":>9s} '
      f'{"cold":>7s} {"cached":>8s} {"rel.err":>9s}')
print('-' * 64)
for nset in [int(x) for x in a.nset.split(',')]:
    dm1 = 2 * cp.asarray([mvir @ x @ mocc.T for x in xs[:nset]])
    dm1 = cp.asarray(dm1 + dm1.transpose(0, 2, 1), order='C')
    args = (mf, mol, mo, occ, dm1, a.in_mo)
    _, t_ref, v_ref = timeit(lambda: HR._nr_rks_fnlc_mo_orig(*args), a.reps)
    t1, t_new, v_new = timeit(
        lambda: fastfnlc._nr_rks_fnlc_mo(*args), a.reps)
    scale = float(cp.abs(cp.asarray(v_ref)).max()) or 1.0
    err = float(cp.abs(cp.asarray(v_new) - cp.asarray(v_ref)).max()) / scale
    print(f'{nset:5d} {t_ref:9.4f} {t1:9.4f} {t_new:9.4f} '
          f'{t_ref/t1:6.2f}x {t_ref/t_new:7.2f}x {err:9.2e}', flush=True)
    rows.append(dict(nset=nset, stock=t_ref, first=t1, cached=t_new, err=err))

print('\n' + json.dumps({'tag': tag, 'in_mo': a.in_mo, 'natm': mol.natm,
                         'nao': int(mol.nao), 'rows': rows}))
