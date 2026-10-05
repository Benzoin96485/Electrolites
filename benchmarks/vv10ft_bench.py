"""The VV10 response pair sum, GPU4PySCF's against fastvv10ft's, one process.

One SCF; the ground-state half of the response (``fastfnlc._ground_state``,
exactly the arrays every caller hands the kernel) is built once, and both
implementations of ``VXC_vv10nlc_hess_eval_f_t`` are timed on the same
trial densities at each ``ntrial`` and compared element-wise.

The trial densities are ``rho_t = rho * u`` and ``gamma_t = gamma * v`` with
``u, v`` uniform in [-1, 1]: the right magnitude point by point, which is
what an element-wise comparison of a linear map needs.

    python vv10ft_bench.py --xyz molecules/penicillinV.xyz --basis def2-svp \
        --xc wb97m-v --ntrial 3,30,126
"""
import ctypes
import json
import time

import cupy as cp
import numpy as np

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--ntrial', default='3,30,126')
ap.add_argument('--tiles', default='',
                help='semicolon separated MT,NTW,NW triples to sweep')
ap.add_argument('--no-stock', action='store_true')
ap.add_argument('--rcut', default='inf',
                help='comma separated fp32 cut-offs (Bohr) to sweep')
a = ap.parse_args()
apply_patch('fastfnlc,fastvv10ft')

from gpu4pyscf import dft                                   # noqa: E402
from electrolites import fastfnlc, fastvv10ft               # noqa: E402

_sync = cp.cuda.Stream.null.synchronize


def timeit(fn, reps):
    fn()
    _sync()
    ts = []
    for _ in range(reps):
        _sync()
        t = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t)
    return min(ts)


mol = build_mol(a)
tag = f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}/L{a.grids}'
print(f'# {tag}  natm={mol.natm} nao={mol.nao}', flush=True)
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc)
mf.grids.level = a.grids
mf.nlcgrids.level = a.grids
mf.conv_tol = a.conv_tol
e = mf.kernel()
print(f'# SCF {e:.9f}  converged={mf.converged}', flush=True)
mo, occ = cp.asarray(mf.mo_coeff), cp.asarray(mf.mo_occ)
ni = mf._numint
grids = mf.nlcgrids
opt = getattr(ni, 'gdftopt', None)
if opt is None:
    ni.build(mol, grids.coords)
    opt = ni.gdftopt
gs = fastfnlc._ground_state(mf, mo, occ, ni, opt, grids)
ng = gs['ngrids']
print(f'# nlc points kept {ng}', flush=True)

names = ['grids_coords', 'grids_weights', 'rho_i', 'omega_i', 'kappa_i',
         'U_i', 'W_i', 'A_i', 'B_i', 'C_i', 'domega_drho_i',
         'domega_dgamma_i', 'dkappa_drho_i', 'd2omega_drho2_i',
         'd2omega_dgamma2_i', 'd2omega_drho_dgamma_i', 'd2kappa_drho2_i']
ins = [cp.ascontiguousarray(gs[k]) for k in names]
P = lambda x: ctypes.cast(x.data.ptr, ctypes.c_void_p)
stock = fastvv10ft.stock()

rng = np.random.default_rng(7)
rows = []
for nt in [int(x) for x in a.ntrial.split(',')]:
    u = cp.asarray(rng.uniform(-1, 1, (nt, ng)))
    v = cp.asarray(rng.uniform(-1, 1, (nt, ng)))
    rho_t = cp.ascontiguousarray(gs['rho_i'] * u)
    gamma_t = cp.ascontiguousarray(gs['gamma_i'] * v)
    fr0 = cp.empty((nt, ng)); fg0 = cp.empty((nt, ng))
    fr1 = cp.empty((nt, ng)); fg1 = cp.empty((nt, ng))
    stream = cp.cuda.get_current_stream()

    def run_stock():
        stock(ctypes.cast(stream.ptr, ctypes.c_void_p), P(fr0), P(fg0),
              *[P(x) for x in ins], P(rho_t), P(gamma_t),
              ctypes.c_int(ng), ctypes.c_int(nt))

    def run_ours(tile=None, rcut=float('inf')):
        kw = dict(rcut=rcut)
        if tile:
            kw.update(mt=tile[0], ntw=tile[1], nw=tile[2])
        fastvv10ft.f_t(fr1, fg1, *ins, rho_t, gamma_t, nt, **kw)

    row = {'ntrial': nt, 'ng': ng}
    if not a.no_stock:
        row['stock'] = timeit(run_stock, a.reps)
    tiles = [None] + [tuple(int(y) for y in t.split(','))
                      for t in a.tiles.split(';') if t]
    for tile, rc in [(t, r) for t in tiles
                     for r in map(float, a.rcut.split(','))]:
        t = timeit(lambda: run_ours(tile, rc), a.reps)
        key = 'ours' if tile is None else 'ours_%d_%d_%d' % tile
        key += '' if rc == float('inf') else f'_rc{rc:g}'
        row[key] = t
        if not a.no_stock:
            er = float(cp.abs(fr1 - fr0).max() / cp.abs(fr0).max())
            eg = float(cp.abs(fg1 - fg0).max() / cp.abs(fg0).max())
            row[key + '_err'] = max(er, eg)
            print(f'ntrial {nt:4d}  {key:14s} {t:8.3f} s   stock '
                  f'{row["stock"]:8.3f} s   {row["stock"]/t:5.2f}x   '
                  f'err r {er:.2e} g {eg:.2e}', flush=True)
        else:
            print(f'ntrial {nt:4d}  {key:14s} {t:8.3f} s', flush=True)
    rows.append(row)
    del u, v, rho_t, gamma_t, fr0, fg0, fr1, fg1
print(json.dumps({'case': tag, 'rows': rows}))
