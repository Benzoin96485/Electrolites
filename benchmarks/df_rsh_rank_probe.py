"""How much of a range-separated DF run's second tensor carries anything.

``docs/DF_WORK_LIST.md`` **W22**.  ``DF.range_coulomb`` builds a second cderi
for the long-range operator ``erf(omega r)/r`` and keeps it next to the
full-range one.  The item asks whether the second tensor has to be a second
*tensor*.  This probe answers the question the item does not ask first: how
many of its ``naux`` rows are worth storing at all.

The long-range operator is smooth, so its fitting metric ``(P|erf|Q)`` is
numerically rank-deficient in any auxiliary basis built for ``1/r``.  In the
metric's eigenbasis ``J = V w V^T`` the fitted exchange is a sum over modes,

    tr(D K[D]) = sum_q  tr(D B_q D B_q) / w_q,      B_q = sum_P V_Pq (P|ij)

so the exchange energy that a mode carries can be read off the tensor built in
that basis, one row at a time, and the error of *not storing* every mode below
a threshold is the sum of what those modes carry.  One build of the whole
eigenbasis gives the curve for every threshold at once.

For each operator (``omega = 0`` and the functional's own) the probe prints:

* which route GPU4PySCF's own ``_decompose_j2c`` takes -- ``cd`` (Cholesky
  succeeded, all ``naux`` rows stored) or ``ed`` (Cholesky failed, rows with
  ``w <= LINEAR_DEP_THR`` dropped) -- and so how many rows it stores;
* the spectrum: ``w_max``, ``w_min`` and the number of modes above each
  relative threshold ``w > tau w_max``;
* the exchange energy, in hartree, carried by the modes each threshold drops,
  with the functional's own factor on it (``lr_factor - sr_factor`` for the
  long-range operator, ``sr_factor`` for the full-range one).

    python benchmarks/df_rsh_rank_probe.py --xyz benchmarks/molecules/glycine.xyz \
        --basis def2-svp --xc hyb_gga_xc_wb97x_d3 --dm-cache $SCRATCH/rshdm
    python benchmarks/df_rsh_rank_probe.py --xyz .../tmc_fecl4.xyz --charge -1 \
        --spin 5 --basis def2-tzvp --xc hyb_gga_xc_wb97x_d3

Nothing here is patched: the density comes from GPU4PySCF's own DF SCF and the
eigenbasis tensor from GPU4PySCF's own ``_cholesky_eri`` with its metric
routine swapped for the full eigen-decomposition.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import _common                                              # noqa: E402

ap = _common.parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--omega', type=float, default=None,
                help="the long-range operator's omega; default the "
                     "functional's")
ap.add_argument('--dm-cache', default='',
                help='prefix of a file the converged density is kept in, so a '
                     'second run skips the SCF')
ap.add_argument('--taus', default='1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,'
                                  '1e-7,1e-6')
ap.add_argument('--blk', type=int, default=64,
                help='modes unpacked at once when the per-mode energies are '
                     'summed')
ap.add_argument('--spectrum-only', action='store_true',
                help='no SCF and no tensor: the metric, its route and its '
                     'spectrum only, which is what a 284-atom case can afford '
                     'on a login GPU')
ap.add_argument('--json', default='')
a = ap.parse_args()

import numpy as np                                          # noqa: E402
import cupy as cp                                           # noqa: E402
from gpu4pyscf.df import df as DFMOD                        # noqa: E402
from gpu4pyscf.df import int3c2e_bdiv as bdiv               # noqa: E402
from gpu4pyscf.lib.cupy_helper import cholesky              # noqa: E402

mol = _common.build_mol(a)
uks = mol.spin != 0
if uks:
    from gpu4pyscf.dft import uks as KS
    mf = KS.UKS(mol, xc=a.xc)
else:
    from gpu4pyscf.dft import rks as KS
    mf = KS.RKS(mol, xc=a.xc)
mf = mf.density_fit(auxbasis=a.auxbasis)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
omega, lr_factor, sr_factor = mf._numint.rsh_and_hybrid_coeff(mf.xc,
                                                              spin=mol.spin)
if a.omega is not None:
    omega = a.omega
assert omega > 0, f'{a.xc} is not a long-range corrected functional'
print(f'# {os.path.basename(a.xyz)} {a.basis} {a.xc}  natm={mol.natm} '
      f'nao={mol.nao} omega={omega} lr_factor={lr_factor} '
      f'sr_factor={sr_factor}', flush=True)

name = os.path.splitext(os.path.basename(a.xyz))[0]

if a.spectrum_only:
    from gpu4pyscf.df import df as DFMOD
    from gpu4pyscf.df import int3c2e_bdiv as bdiv
    from gpu4pyscf.lib.cupy_helper import cholesky
    import cupy as cp
    dfobj = mf.with_df
    dfobj.build(build_cderi=False)
    # the sorted auxiliary basis is Cartesian and decontracted; the metric,
    # and so the tensor, is in the contracted spherical one
    auxmol = dfobj.intopt.auxmol
    naux = dfobj.auxmol.nao
    npair = len(dfobj.intopt.pair_and_diag_indices()[0])
    print(f'# naux={naux} stored pairs={npair}  full tensor '
          f'{naux * npair * 8e-9:.2f} GB', flush=True)
    for om in (0.0, omega):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        j2c = bdiv.int2c2e(auxmol, omega=om or None)
        assert j2c.shape[0] == naux
        cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        try:
            cholesky(j2c)
            route = 'cd'
        except RuntimeError:
            route = 'ed'
        cp.cuda.Device().synchronize()
        t2 = time.perf_counter()
        w = cp.asnumpy(cp.linalg.eigvalsh(j2c))[::-1]
        t3 = time.perf_counter()
        n_stock = naux if route == 'cd' else int((w > DFMOD.LINEAR_DEP_THR).sum())
        print(f'omega={om}: route {route}  stored rows {n_stock}/{naux} '
              f'({100 * n_stock / naux:.1f} %)  -> {n_stock * npair * 8e-9:.2f} GB   '
              f'w_max {w[0]:.3e} w_min {w[-1]:.3e}   int2c2e {t1 - t0:.2f} s  '
              f'potrf {t2 - t1:.2f} s  eigvalsh {t3 - t2:.2f} s', flush=True)
        for tau in [float(t) for t in a.taus.split(',')]:
            k = int((w > tau * w[0]).sum())
            print(f'   tau {tau:7.0e}: {k:6d} ({100 * k / naux:5.1f} %)')
        del j2c
    sys.exit(0)

cache = f'{a.dm_cache}.{name}.{a.basis}.{a.xc}.npz' if a.dm_cache else ''
if cache and os.path.exists(cache):
    z = np.load(cache)
    mo_coeff, mo_occ = z['mo_coeff'], z['mo_occ']
    print(f'# density from {cache}', flush=True)
else:
    t0 = time.perf_counter()
    e = mf.kernel()
    print(f'# SCF {e:.10f}  converged={mf.converged}  '
          f'{time.perf_counter() - t0:.1f} s', flush=True)
    mo_coeff, mo_occ = cp.asnumpy(mf.mo_coeff), cp.asnumpy(mf.mo_occ)
    if cache:
        np.savez(cache, mo_coeff=mo_coeff, mo_occ=mo_occ)
mf.with_df.reset()
cp.get_default_memory_pool().free_all_blocks()

# occupied orbitals per spin, and the weight a spin's ||C^T B C||^2 carries in
# the exchange energy: -1/4 tr(D K[D]) closed shell, -1/2 sum_s tr(D_s K[D_s])
# open shell
if uks:
    occ = [cp.asarray(mo_coeff[s][:, mo_occ[s] > 0]) for s in (0, 1)]
    spin_w = 0.5
else:
    occ = [cp.asarray(mo_coeff[:, mo_occ > 0])]
    spin_w = 1.0

dfobj = mf.with_df
dfobj.build(build_cderi=False)
intopt, auxmol = dfobj.intopt, dfobj.auxmol
naux = auxmol.nao

state = {}


def _eigen_decompose(auxmol_s, aux_sorting=None, omega=None):
    """``_decompose_j2c`` with every mode kept, strongest first."""
    j2c = bdiv.int2c2e(auxmol_s, omega=omega)
    try:
        cholesky(j2c)
        state['route'] = 'cd'
    except RuntimeError:
        state['route'] = 'ed'
    w, v = cp.linalg.eigh(j2c)
    order = cp.argsort(-w)
    w, v = w[order], v[:, order]
    wmax = float(w[0])
    # a mode at or under zero is rounding, not a function; it cannot be
    # normalised, and it is counted as dropped by every threshold
    keep = cp.where(w > 1e-16 * wmax)[0]
    state['w'] = cp.asnumpy(w)
    state['nkept'] = int(keep.size)
    v = v[:, keep] / cp.sqrt(w[keep])
    coef = auxmol_s.apply_C_dot(v, axis=0)
    if aux_sorting is not None:
        coef, tmp = cp.empty_like(coef), coef
        coef[aux_sorting] = tmp
    return coef, 'ed'


def per_mode(cderi, cderi_idx):
    """Exchange energy each stored row carries, in the probe's units."""
    nao = mol.nao
    rows, cols = divmod(cp.asarray(cderi_idx[0]), nao)
    L = cderi[0]
    nq = L.shape[0]
    out = np.zeros(nq)
    buf = cp.zeros((a.blk, nao, nao))
    for q0 in range(0, nq, a.blk):
        q1 = min(nq, q0 + a.blk)
        blk = cp.asarray(L[q0:q1])
        A = buf[:q1 - q0]
        A[:] = 0
        A[:, rows, cols] = blk
        A[:, cols, rows] = blk
        acc = cp.zeros(q1 - q0)
        for c in occ:
            x = cp.einsum('qij,jo->qio', A, c)
            x = cp.einsum('ip,qio->qpo', c, x)
            acc += cp.einsum('qpo,qpo->q', x, x)
        out[q0:q1] = cp.asnumpy(acc) * spin_w
    return out


taus = [float(t) for t in a.taus.split(',')]
result = {'case': name, 'basis': a.basis, 'xc': a.xc, 'natm': mol.natm,
          'nao': mol.nao, 'naux': naux, 'omega': omega}
orig = DFMOD._decompose_j2c
for om, factor, label in ((0.0, sr_factor, 'full'),
                          (omega, lr_factor - sr_factor, 'long-range')):
    state.clear()
    DFMOD._decompose_j2c = _eigen_decompose
    try:
        t0 = time.perf_counter()
        with mol.with_range_coulomb(om), auxmol.with_range_coulomb(om):
            cderi, idx = DFMOD._cholesky_eri(intopt, omega=om or None)
        t_build = time.perf_counter() - t0
    finally:
        DFMOD._decompose_j2c = orig
    w = state['w']
    wmax = w[0]
    e = per_mode(cderi, idx)
    del cderi
    cp.get_default_memory_pool().free_all_blocks()
    # the stock route's own count, for the rows GPU4PySCF actually stores
    n_stock = naux if state['route'] == 'cd' else int(
        (w > DFMOD.LINEAR_DEP_THR).sum())
    tot = e.sum()
    print(f'\n## {label} operator  omega={om}  factor={factor:.6f}  '
          f'build {t_build:.1f} s', flush=True)
    print(f'   stock route: {state["route"]}  rows stored by GPU4PySCF: '
          f'{n_stock}/{naux}   w_max {wmax:.4e}  w_min {w[-1]:.4e}  '
          f'(normalisable {state["nkept"]})')
    print(f'   exchange energy carried, all modes: '
          f'{-factor * tot:.10f} Eh')
    print(f'   {"tau":>7s} {"kept":>7s} {"kept %":>7s} '
          f'{"dropped E_x / Eh":>17s} {"per atom":>10s}')
    rows = []
    for tau in taus:
        k = int((w > tau * wmax).sum())
        k_e = min(k, e.size)
        dE = factor * (tot - e[:k_e].sum())
        rows.append({'tau': tau, 'kept': k, 'dE': dE})
        print(f'   {tau:7.0e} {k:7d} {100 * k / naux:6.1f}% '
              f'{dE:17.3e} {dE / mol.natm:10.2e}')
    result[label] = {'route': state['route'], 'n_stock': n_stock,
                     'wmax': float(wmax), 'wmin': float(w[-1]),
                     'E_x': float(-factor * tot), 'rows': rows}

if a.json:
    with open(a.json, 'a') as fh:
        fh.write(json.dumps(result) + '\n')
