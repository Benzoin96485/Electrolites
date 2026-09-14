"""The unrestricted VV10 terms A/B, every arm in one process on one density.

Three arms for the SCF's XC-and-VV10 half, all on the same converged UKS
density, with the same ``_vv10nlc`` on every one of them (so this measures the
grid passes and not the double-sum kernel):

    stock    GPU4PySCF's ``nr_uks`` then its ``nr_nlc_vxc`` -- four walks of
             the grid, which is what ``dft/uks.py:get_veff`` does
    fastxc   ``fastxc``'s grouped ``nr_uks`` then ``nr_nlc_vxc`` -- still four
    fused    ``fastxcnlc.nr_uks_nlc`` -- two

and the gradient's VV10 term, ours against GPU4PySCF's, at two density
matrices.  Every row carries the agreement with the arm it replaces, because
a timing on its own cannot tell a faster path from a different answer.

Run it twice, with and without ``FASTNLC_OFF=1``, to see the same grid passes
against a slow and a fast double sum: the fusion's share of the whole term
depends entirely on which one is underneath it.

    python benchmarks/unlc_bench.py --xyz benchmarks/molecules/tmc_cucl2.xyz \
        --spin 1 --basis def2-svp --xc wb97m-v
"""
import json
import os
import time

import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
a = ap.parse_args()
# every arm needs the unpatched entry points saved, so patch the modules this
# script switches between rather than taking `--patch` at face value
active = apply_patch(a.patch or 'fastk,fastj,fastxc,fastnlc,fastxcnlc,'
                                'fastxcgrad')

from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.dft import numint as NI                       # noqa: E402
from gpu4pyscf.grad import rks as grks                       # noqa: E402
from gpu4pyscf.lib.cupy_helper import (                      # noqa: E402
    tag_array, take_last2d)
from electrolites import compat as g4pcompat                 # noqa: E402
from electrolites import fastxcnlc                           # noqa: E402

_sync = cp.cuda.Stream.null.synchronize
assert a.spin, 'this benchmark is the unrestricted path; pass --spin'


def timeit(fn, reps):
    fn()
    _sync()
    ts = []
    out = None
    for _ in range(reps):
        _sync()
        t = time.perf_counter()
        out = fn()
        _sync()
        ts.append(time.perf_counter() - t)
    return min(ts), out


mol = build_mol(a)
tag = f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}/spin{a.spin}'
print(f'# {tag}  natm={mol.natm} nao={mol.nao} '
      f'FASTNLC_OFF={os.environ.get("FASTNLC_OFF", "0")}', flush=True)

mf = dft.UKS(mol, xc=a.xc)
mf.grids.level = a.grids
mf.nlcgrids.level = a.grids       # else the two grids differ and the fusion
mf.conv_tol = a.conv_tol          # declines -- correctly, and silently
mf.direct_scf_tol = 1e-13
mf.max_cycle = 100
mf.verbose = 0
e = mf.kernel()
print(f'# SCF {e:.9f}  converged={mf.converged}  cycles={mf.cycles}',
      flush=True)
print(f'# grids {mf.grids.coords.shape[0]} points, nlcgrids '
      f'{mf.nlcgrids.coords.shape[0]}, same set='
      f'{fastxcnlc.same_grid(mf.grids, mf.nlcgrids)}', flush=True)

ni = mf._numint
dm = cp.asarray(mf.make_rdm1()).view(cp.ndarray)
nlc = mf.xc if ni.libxc.is_nlc(mf.xc) else mf.nlc


def two_calls(nr_uks):
    def go():
        n, exc, v = nr_uks(ni, mol, mf.grids, mf.xc, dm)
        _n2, enlc, vnlc = ni.nr_nlc_vxc(mol, mf.nlcgrids, nlc, dm)
        return n, float(exc) + float(enlc), cp.asarray(v) + cp.asarray(vnlc)
    return go


def fused():
    return fastxcnlc.nr_uks_nlc(ni, mol, mf.grids, mf.xc, nlc, dm)


arms = [('stock', two_calls(NI._nr_uks_orig)),
        ('fastxc', two_calls(lambda *args: ni.nr_uks(*args[1:]))),
        ('fused', fused)]

rows = {}
ref = None
print(f'\n{"arm":10s} {"time/s":>9s} {"vs stock":>9s} {"d(exc)":>10s} '
      f'{"d(vmat)":>10s}')
print('-' * 54)
for name, fn in arms:
    t, out = timeit(fn, a.reps)
    assert out is not None, f'{name} declined'
    if ref is None:
        ref = out
        t0 = t
        d_e = d_v = 0.0
    else:
        d_e = abs(out[1] - ref[1]) / max(abs(ref[1]), 1.0)
        d_v = (float(cp.abs(cp.asarray(out[2]) - cp.asarray(ref[2])).max())
               / (float(cp.abs(cp.asarray(ref[2])).max()) or 1.0))
    rows[name] = dict(t=t, d_exc=d_e, d_vmat=d_v)
    print(f'{name:10s} {t:9.4f} {t0/t:8.2f}x {d_e:10.2e} {d_v:10.2e}',
          flush=True)

# --- how much of the term is the double sum at all? -----------------------
# The 284-atom cluster `fastnlc` was written on spends 61.4 s of a 65.1 s
# `nr_nlc_vxc` in the double sum, which makes the grid passes look like noise.
# That is an O(ngrids^2) against an O(ngrids) statement, so it does not carry
# down to a 21-atom complex, and the arms above are only readable next to it.
rho4 = cp.empty((4, mf.nlcgrids.coords.shape[0]))
p1 = 0
dmt_sum = ni.gdftopt.sort_orbitals(cp.asarray(dm[0] + dm[1])[None],
                                  axis=[1, 2])[0]
for ao_mask, idx, weight, _ in ni.block_loop(
        ni.gdftopt._sorted_mol, mf.nlcgrids, ni.gdftopt._sorted_mol.nao, 1,
        max_memory=None):
    p0, p1 = p1, p1 + weight.size
    rho4[:, p0:p1] = NI.eval_rho(
        ni.gdftopt._sorted_mol, ao_mask,
        take_last2d(dmt_sum, idx), xctype='GGA', hermi=1, with_lapl=False)
nlc_pars = ni.nlc_coeff(nlc)[0][0]
t_ds_new, _ = timeit(lambda: g4pcompat.vv10nlc(
    rho4, mf.nlcgrids.coords, mf.nlcgrids.weights, nlc_pars), a.reps)
_orig = getattr(NI, '_vv10nlc_orig', None)
t_ds_ref = float('nan')
if _orig is not None:
    _saved, NI._vv10nlc = NI._vv10nlc, _orig
    try:
        t_ds_ref, _ = timeit(lambda: g4pcompat.vv10nlc(
            rho4, mf.nlcgrids.coords, mf.nlcgrids.weights, nlc_pars), a.reps)
    finally:
        NI._vv10nlc = _saved
# Every arm above ran with `fastnlc`'s double sum underneath, so its time is
# the part of each of them that none of the three changes touch: subtract it
# and what is left is the grid passes, which is what they do change.  Print
# both, because the ratio of the *stock* double sum to the arms is not a
# share of anything -- it can exceed 100 %.
_share = t_ds_new / rows['stock']['t']
print(f'\ndouble sum alone: {t_ds_new:.4f} s (fastnlc, {100*_share:.0f} % of '
      f'the stock arm, and the same term in all three)  {t_ds_ref:.4f} s '
      f'(GPU4PySCF, i.e. fastnlc is {t_ds_ref/t_ds_new:.2f}x on it)',
      flush=True)
print(f'{"grid passes":10s} {"time/s":>9s} {"vs stock":>9s}   '
      f'(each arm minus the double sum)')
print('-' * 42)
for _n in ('stock', 'fastxc', 'fused'):
    _t = rows[_n]['t'] - t_ds_new
    rows[_n]['passes'] = _t
    print(f'{_n:10s} {_t:9.4f} '
          f'{(rows["stock"]["t"] - t_ds_new)/_t:8.2f}x', flush=True)
rows['double_sum'] = dict(fastnlc=t_ds_new, stock=t_ds_ref)
rho4 = dmt_sum = None

# --- the gradient's VV10 term at two density matrices ---------------------
dmt = tag_array(mf.make_rdm1(), mo_coeff=cp.asarray(mf.mo_coeff),
                mo_occ=cp.asarray(mf.mo_occ))
print(f'\n{"grad nlc":10s} {"time/s":>9s} {"vs stock":>9s} {"d(exc1)":>10s}')
print('-' * 42)
t_ref, e_ref = timeit(
    lambda: grks._get_nlc_exc_orig(ni, mol, mf.nlcgrids, nlc, dmt,
                                   verbose=0)[1], a.reps)
t_new, e_new = timeit(
    lambda: grks.get_nlc_exc(ni, mol, mf.nlcgrids, nlc, dmt, verbose=0)[1],
    a.reps)
d = (float(cp.abs(cp.asarray(e_new) - cp.asarray(e_ref)).max())
     / (float(cp.abs(cp.asarray(e_ref)).max()) or 1.0))
print(f'{"stock":10s} {t_ref:9.4f} {1.0:8.2f}x')
print(f'{"ours":10s} {t_new:9.4f} {t_ref/t_new:8.2f}x {d:10.2e}', flush=True)
rows['grad_nlc'] = dict(stock=t_ref, ours=t_new, d=d)

print('\n' + json.dumps({'tag': tag, 'natm': mol.natm, 'nao': int(mol.nao),
                         'spin': int(mol.spin), 'grids': a.grids,
                         'fastnlc_off': os.environ.get('FASTNLC_OFF', '0'),
                         'patched': list(active), 'rows': rows}))
