"""W33: a Hessian's CPHF grid half, stage by stage, on both routes.

`docs/DF_WORK_LIST.md` **W33**, `docs/README_fastxcfxcmgga.md`.
``fastxcfxc._nr_rks_fxc_mo`` on the right-hand sides one CPHF cycle hands it
(3 natm tagged densities, ``hessian/rhf.py:gen_vind``), timed per stage with
``FASTXCFXC_TIME=1`` on the AO-matrix route (``FASTXCFXC_OCC=0``, what W16
shipped) and on W33's occupied-side route, with the flops each stage does
worked out block by block from the same grid and the same decisions the
module makes -- so every stage has a rate, and "is it at the GEMM rate" has
an answer rather than a guess.

    python benchmarks/fxc_occ_probe.py --xyz benchmarks/molecules/penicillinV.xyz \\
        --basis def2-tzvpp --xc tpss [--ratio 1] [--ratio-gga 3]

An open shell (``--spin`` not 0) times the grid half of an unrestricted
Hessian's CPHF instead -- ``nr_uks_fxc`` on ``gen_vind``'s tagged density,
both spins taken to the MO basis -- whose matrix route evaluates the density
with GPU4PySCF's ``eval_rho4``, so its ``gemm1`` row is empty and its
``rho1`` row is that evaluation.

The stage timers synchronise, so their sum is a little above an untimed
call; `benchmarks/df_hess_terms_probe.py --env` is the untimed A/B.
"""
import os
import time

import numpy as np
import cupy as cp

os.environ['FASTXCFXC_TIME'] = '1'

from _common import apply_patch, build_mol, parser            # noqa: E402

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--ratio', type=float, default=None,
                help='FASTXCFXC_OCC_RATIO (meta-GGA) for the occupied route')
ap.add_argument('--ratio-gga', type=float, default=None,
                help='FASTXCFXC_OCC_RATIO_GGA for the occupied route')
ap.add_argument('--routes', default='mat,occ',
                help='which routes to time; `occ` alone for a profile of '
                     'the occupied route that no matrix-route kernel mixes into')
a = ap.parse_args()
apply_patch(a.patch or 'fastxcfxc')

from gpu4pyscf import dft                                     # noqa: E402
from gpu4pyscf.dft import numint as NI                        # noqa: E402
from gpu4pyscf.lib.cupy_helper import (                       # noqa: E402
    contract, tag_array, transpose_sum)
from electrolites import fastxcfxc                            # noqa: E402

if a.ratio is not None:
    os.environ['FASTXCFXC_OCC_RATIO'] = str(a.ratio)
if a.ratio_gga is not None:
    os.environ['FASTXCFXC_OCC_RATIO_GGA'] = str(a.ratio_gga)
r_m = float(os.environ.get('FASTXCFXC_OCC_RATIO', '1'))
r_g = float(os.environ.get('FASTXCFXC_OCC_RATIO_GGA', '3'))
r_f = float(os.environ.get('FASTXCFXC_FACTOR_RATIO', '4'))

mol = build_mol(a)
uks = mol.spin != 0
mf = (dft.UKS if uks else dft.RKS)(mol, xc=a.xc).density_fit()
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.verbose = 0
mf.kernel()
ni = mf._numint
xctype = ni._xc_type(mf.xc)
mo_coeff = cp.asarray(mf.mo_coeff)
mo_occ = cp.asarray(mf.mo_occ)
nset = 3 * mol.natm
rng = np.random.default_rng(0)
grids = getattr(mf, 'cphf_grids', None) or mf.grids
if not uks:
    orbo = mo_coeff[:, mo_occ > 0]
    nmo, nocc = mo_coeff.shape[1], orbo.shape[1]
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, mf.xc, mo_coeff, mo_occ,
                                        0)
    mo1 = cp.asarray(rng.standard_normal((nset, nmo, nocc)) * 1e-2)
    mo1_ao = contract('npo,ip->nio', mo1, mo_coeff)
    dm1 = cp.asarray(contract('npi,qi->npq', mo1_ao, orbo * 2), order='C')
    transpose_sum(dm1, inplace=True, hermi=1)
    dm1 = tag_array(dm1, mo1=mo1_ao, occ_coeff=orbo, symmetrize=1)
else:
    # what `hessian/uhf.py:gen_vind` hands `get_veff_resp_mo`
    from gpu4pyscf.hessian.uhf import _stack_uhf_occ_oribtals
    from gpu4pyscf.hessian.rhf import _ao2mo
    nmo = mo_coeff.shape[-1]
    occs = [mo_coeff[s][:, mo_occ[s] > 0] for s in (0, 1)]
    nocca, noccb = occs[0].shape[1], occs[1].shape[1]
    nocc = nocca
    orbo = _stack_uhf_occ_oribtals(occs[0], occs[1])
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, mf.xc, mo_coeff, mo_occ,
                                        1)
    mo1 = _stack_uhf_occ_oribtals(
        cp.asarray(rng.standard_normal((nset, nmo, nocca)) * 1e-2),
        cp.asarray(rng.standard_normal((nset, nmo, noccb)) * 1e-2))
    mo1_mo = contract('snai,spa->snpi', mo1, mo_coeff)
    dm1 = cp.asarray(contract('snpi,sqi->snpq', mo1_mo, orbo), order='C')
    transpose_sum(dm1.reshape(-1, mol.nao, mol.nao), inplace=True, hermi=1)
    dm1 = tag_array(dm1, mo1=mo1_mo, occ_coeff=orbo, symmetrize=1)


def call_u():
    """The grid half of `hessian/uks.py:get_veff_resp_mo`, both spins taken
    to the MO basis the way the method does (the occupied route's own
    contraction when it returns V C, upstream's `_ao2mo` otherwise)."""
    v = fastxcfxc._nr_uks_fxc_blocked(ni, mol, grids, mf.xc, None, dm1, 0, 1,
                                      rho0, vxc, fxc, None, None, occ=True)
    if isinstance(v, list):
        return [contract('nio,ip->npo', v[s][:, :, :(nocca, noccb)[s]],
                         mo_coeff[s]) for s in (0, 1)]
    return [_ao2mo(v[s], occs[s], mo_coeff[s]) for s in (0, 1)]


# ---- the blocks, and the flops each route spends on them ------------------
opt = ni.gdftopt
mgga = xctype == 'MGGA'
subs = []
for ao, idx, w, _ in ni.block_loop(opt._sorted_mol, grids, mol.nao, 1,
                                   max_memory=None, strict_grid_order=True):
    if len(idx):
        subs.append((len(idx), w.size))
nsub = np.array([s[0] for s in subs], dtype=float)
ng = np.array([s[1] for s in subs], dtype=float)
yrho = mgga or uks or os.environ.get('FASTXCFXC_OCC_RHO_GGA', '0') == '1'
r_p = r_m if mgga else r_g                  # the potential half's ratio
r_d = r_m if (mgga or uks) else r_g         # the density half's
occ_any = min(r_d if yrho else r_p, r_p) * nocc < nsub.max()
F = {'mat': {'gemm1': 0., 'gemm2': 0., 'rho1': 0.},
     'occ': {'gemm1': 0., 'gemm2': 0., 'rho1': 0.}}
for n, g in zip(nsub, ng):
    ns = 2 if uks else 1
    m2 = 2 * nset * n * n * g * ns              # one nsub x nsub GEMM, all sets
    o1 = 2 * nset * nocc * n * g * ns           # one nocc-wide GEMM, all sets
    x = 4 * 2 * nocc * n * g * ns               # X = C^T ao[n], shared
    fac = (not mgga) and (not uks) and r_f * nocc < n
    # the unrestricted matrix route evaluates its density with upstream's
    # `eval_rho4` from the orbitals (4 nocc-wide products), not by a GEMM
    g1 = (4 * o1 if uks else 4 * m2 if mgga
          else (3 * o1 + 2 * nocc * n * g if fac else m2))
    gk = 'rho1' if uks else 'gemm1'
    p_mat = (4 if mgga else 1) * m2
    F['mat'][gk] += g1
    F['mat']['gemm2'] += p_mat
    ob_d = occ_any and yrho and r_d * nocc < n
    ob_p = occ_any and r_p * nocc < n
    F['occ']['gemm1'] += (x if (ob_d or ob_p) else 0.) + (4 * o1 if ob_d
                                                          else 0.)
    if not ob_d:
        F['occ'][gk] += g1
    F['occ']['gemm2'] += (4 if mgga else 2) * o1 if ob_p else p_mat
print(f'# {mol.natm} atoms  nao {mol.nao}  nocc {nocc}  {a.basis}  {a.xc} '
      f'({xctype})  nset {nset}  ngrids {int(ng.sum())}  blocks {len(nsub)}')
print(f'# nsub: mean {nsub.mean():.0f}  median {np.median(nsub):.0f}  '
      f'max {nsub.max():.0f}   nsub/nocc (grid-weighted) '
      f'{(nsub * ng).sum() / ng.sum() / nocc:.2f}   ratio used '
      f'{r_d}/{r_p} (density/potential)'
      + (f'   UKS nocc {nocca}/{noccb}' if uks else ''))


def run(env):
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        best = None
        for _ in range(max(2, a.reps)):
            for k in fastxcfxc._T:
                fastxcfxc._T[k] = 0 if k == 'n' else 0.
            cp.cuda.Stream.null.synchronize()
            t = time.perf_counter()
            if uks:
                call_u()
            else:
                fastxcfxc._nr_rks_fxc_mo(ni, mol, grids, mf.xc, None, dm1,
                                         mo_coeff, 0, 1, rho0, vxc, fxc,
                                         max_memory=None)
            cp.cuda.Stream.null.synchronize()
            t = time.perf_counter() - t
            st = dict(fastxcfxc._T)
            if best is None or t < best[0]:
                best = (t, st)
        return best
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


routes = a.routes.split(',')
cp.cuda.profiler.start()       # `nsys -c cudaProfilerApi` sees the calls only
res = {r: run({'FASTXCFXC_OCC': '0'} if r == 'mat' else {}) for r in routes}
cp.cuda.profiler.stop()
if len(routes) == 1:
    res[{'mat': 'occ', 'occ': 'mat'}[routes[0]]] = res[routes[0]]
print(f'\n| stage | AO-matrix route | | occupied route | |')
print('|---|---|---|---|---|')
for k in ('prep', 'gemm1', 'rho1', 'wv', 'scale', 'gemm2', 'accum'):
    row = f'| {k} |'
    for r in ('mat', 'occ'):
        t = res[r][1][k]
        fl = F[r].get(k)
        rate = f'{fl / t / 1e12:5.1f} TF/s' if fl and t > 0 else ''
        row += f' {t * 1e3:8.1f} ms | {rate} |'
    print(row)
print(f'| **call** | {res["mat"][0] * 1e3:8.1f} ms | | '
      f'{res["occ"][0] * 1e3:8.1f} ms | **{res["mat"][0] / res["occ"][0]:.2f}x** |')
for r in ('mat', 'occ'):
    print(f'# {r}: gemm1 {F[r]["gemm1"] / 1e12:.2f} TF, gemm2 '
          f'{F[r]["gemm2"] / 1e12:.2f} TF')
print('# (the occupied route\'s gemm2 includes its W kernel and the scatter; '
      'gemm1 its X and Y GEMMs and gathers)')
print('# stats', fastxcfxc.stats())
