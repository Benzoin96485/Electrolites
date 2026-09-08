"""`fastxchess` against GPU4PySCF on the two XC Hessian terms.

Both implementations are pure functions of a converged SCF, so they are timed
in one process on the same density -- there is no convergence path to confound
the comparison the way there is for an SCF wall clock.  The ablation column
keeps mechanism 2 (the masked AO space and the batched atom dimension) and
drops mechanism 1 (the atom screen), so the two can be separated.

    python benchmarks/xchess_bench.py --xyz benchmarks/molecules/penicillinV.xyz \
        --basis def2-svp --xc b3lyp --grids 3
"""
import json
import time

import cupy as cp

from _common import build_mol, parser

ap = parser(__doc__.splitlines()[0])
a = ap.parse_args()

import electrolites                                          # noqa: E402
electrolites.patch('fastxchess')
from electrolites import fastxchess                          # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.hessian import rks as hess_rks                 # noqa: E402

_sync = cp.cuda.Stream.null.synchronize


def timeit(fn, reps):
    fn()
    _sync()
    ts = []
    for _ in range(reps):
        _sync()
        t = time.perf_counter()
        out = fn()
        _sync()
        ts.append(time.perf_counter() - t)
    return min(ts), out


mol = build_mol(a)
mf = dft.RKS(mol, xc=a.xc)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.kernel()
h = mf.Hessian()
args = (h, mf.mo_coeff, mf.mo_occ, 4000)
tag = (f'{a.xyz.rsplit("/",1)[-1]}/{a.basis}/{a.xc}/L{a.grids}')
gpu = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
print(f'# {tag}  natm={mol.natm} nao={mol.nao}  {gpu}', flush=True)

rows = []
for name, fast, orig in (
        ('_get_vxc_deriv1', hess_rks._get_vxc_deriv1,
         hess_rks._get_vxc_deriv1_orig),
        ('_get_vxc_deriv2', hess_rks._get_vxc_deriv2,
         hess_rks._get_vxc_deriv2_orig)):
    t_o, ref = timeit(lambda: orig(*args), a.reps)
    t_f, out = timeit(lambda: fast(*args), a.reps)
    err = float(cp.abs(cp.asarray(out) - cp.asarray(ref)).max())
    scale = float(cp.abs(cp.asarray(ref)).max())
    fastxchess._ATOMSCREEN = False
    t_n, _ = timeit(lambda: fast(*args), a.reps)
    fastxchess._ATOMSCREEN = True
    ref = out = None
    cp.get_default_memory_pool().free_all_blocks()
    rows.append((name, t_o, t_f, t_n, err, scale))
    print(f'# {name}: {t_o:.3f} -> {t_f:.3f} s  ({t_o/t_f:.2f}x), '
          f'no atom screen {t_n:.3f} s ({t_o/t_n:.2f}x), '
          f'max err {err:.2e} of {scale:.2e}', flush=True)

print(f'\n{"term":18s} {"GPU4PySCF":>10s} {"fastxchess":>11s} {"speedup":>8s} '
      f'{"no screen":>10s} {"":>8s} {"max err":>9s}')
print('-' * 82)
tot_o = tot_f = tot_n = 0.
for name, t_o, t_f, t_n, err, scale in rows:
    tot_o += t_o
    tot_f += t_f
    tot_n += t_n
    print(f'{name:18s} {t_o:10.3f} {t_f:11.3f} {t_o/t_f:7.2f}x '
          f'{t_n:10.3f} {t_o/t_n:7.2f}x {err:9.2e}')
print('-' * 82)
print(f'{"both":18s} {tot_o:10.3f} {tot_f:11.3f} {tot_o/tot_f:7.2f}x '
      f'{tot_n:10.3f} {tot_o/tot_n:7.2f}x')
print('\n' + json.dumps({'tag': tag, 'gpu': gpu, 'natm': mol.natm,
                         'nao': int(mol.nao),
                         'rows': [{'term': r[0], 'orig': r[1], 'fast': r[2],
                                   'no_atom_screen': r[3], 'err': r[4],
                                   'scale': r[5]} for r in rows]}))
