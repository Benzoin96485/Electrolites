"""How much of the three-centre Coulomb build a density screen could remove.

``docs/DF_WORK_LIST.md`` **W20**: ``J`` is linear in ``D``, so ``J(D) =
J(D_last) + J(D - D_last)`` exactly, and the three-centre McMurchie--Davidson
build walks every stored shell pair against every auxiliary shell on every
cycle whatever the difference is.  The entry says to measure the surviving
pair count through an SCF before writing anything, and this is that
measurement.  It needs no new kernel: it runs one converged DF SCF, keeps the
density every cycle hands ``get_veff``, and prices a screen on each difference
against the Schwarz bounds the build already has --

* ``q_ij = sqrt(max (ij|ij))``, which ``Int3c2eOpt.build`` computes and keeps
  (as a logarithm) for every stored shell pair, and
* ``q_P = sqrt(max (P|P))`` for every sorted auxiliary shell, in closed form
  (``electrolites/_j3cinc.py:self_coulomb``, which is ``SortedMole``'s own
  ``int2c2e`` diagonal to the last digit), in the sorted Cartesian basis the
  kernels run in -- ``apply_CT_dot`` of that metric is the one GPU4PySCF
  solves with, to 3e-11 on glycine.

For the **density contraction** ``rho_P = sum_ij (ij|P) dD_ij`` a pair can go
if ``q_ij max|dD_ij| max_P q_P < tau``; for the **auxiliary-vector** one
``dJ_ij = sum_P (ij|P) dc_P`` an auxiliary shell can go if ``q_P max|dc_P|
max_ij q_ij < tau``, and a pair can go if ``q_ij sum_P q_P |dc_P| < tau``.
Each is reported as a fraction of the kernel's work, weighting a pair by its
Hermite count ``nf3(li+lj)`` and an auxiliary shell by ``nf3(lk)`` -- the two
loop lengths the kernels' inner products run over.

    python benchmarks/df_j3c_screen_probe.py \\
        --xyz benchmarks/molecules/taxol.xyz --basis def2-svp --xc pbe
"""
import json
import time

import numpy as np
import cupy as cp

from _common import build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--taus', default='1e-9,1e-10,1e-11,1e-12,1e-13')
ap.add_argument('--save', default='', help='write the per-cycle table as JSON')
a = ap.parse_args()

from gpu4pyscf import dft                                      # noqa: E402
from gpu4pyscf.df import df_jk as DFJK                         # noqa: E402

taus = [float(x) for x in a.taus.split(',')]
mol = build_mol(a)
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc).density_fit(
    auxbasis=a.auxbasis)
mf.conv_tol = a.conv_tol
mf.grids.level = a.grids
mf.verbose = 0

dms = []
_orig_get_j = DFJK._DFHF.get_j


def get_j(self, mol=None, dm=None, hermi=1, *args, **kw):
    d = cp.asarray(dm)
    dms.append((d[0] + d[1]) if d.ndim == 3 else d.copy())
    return _orig_get_j(self, mol, dm, hermi, *args, **kw)


DFJK._DFHF.get_j = get_j
t0 = time.perf_counter()
mf.kernel()
DFJK._DFHF.get_j = _orig_get_j
t_scf = time.perf_counter() - t0
dfobj = mf.with_df
opt = dfobj.j_engine
mols, auxs = opt.mol, opt.auxmol
print(f'# {a.xyz.split("/")[-1]} {a.basis} {a.xc}  natm {mol.natm}  nao '
      f'{mol.nao}  naux {dfobj.auxmol.nao}  sorted: {mols.nbas} shells / '
      f'{mols.nao} functions, {auxs.nbas} auxiliary shells / {auxs.nao}')
print(f'# SCF {t_scf:.1f} s, {mf.cycles} cycles, E = {mf.e_tot:.10f}, '
      f'{len(dms)} Coulomb builds')

# ---- the Schwarz factors, the same ones `fastdfj3c`'s screen uses ----
from electrolites import _j3c, _j3cinc                          # noqa: E402
c = _j3c.constants(opt)
t0 = time.perf_counter()
scr = _j3cinc.screen(opt, c)
q_pair, qP = scr.q_pair, scr.q_aux
ls = np.asarray(mols._bas[:, 1])
lij = cp.asarray(ls)[scr.ish] + cp.asarray(ls)[scr.jsh]
w_pair = ((lij + 1) * (lij + 2) * (lij + 3) // 6).astype(np.float64)
aux_ls = np.asarray(auxs._bas[:, 1])
w_aux = cp.asarray(((aux_ls + 1) * (aux_ls + 2) * (aux_ls + 3) // 6)
                   .astype(np.float64))
print(f'# stored shell pairs {c.npair}, max q_ij {scr.q_pair_max:.3g}; q_P for '
      f'{auxs.nbas} auxiliary shells, max {scr.q_aux_max:.3g} '
      f'({time.perf_counter() - t0:.2f} s)')
shell_max = scr.pair_absmax
aux_shell_max = scr.aux_absmax


# the fitted coefficients each build solves for, in the sorted space the
# auxiliary-vector kernel reads
j2c, tag = dfobj._cd_j2c


def coeffs(d):
    ds = mols.apply_C_mat_CT(d)
    rho = auxs.apply_CT_dot(opt.contract_dm(ds, 1), axis=-1)
    if tag == 'cd':
        c = DFJK.solve_triangular(j2c, rho[:, None], lower=True)
        c = DFJK.solve_triangular(j2c.T, c, lower=False)[:, 0]
    else:
        c = cp.linalg.solve(j2c, rho)
    return auxs.apply_C_dot(c, axis=-1), ds


W_pair = float(w_pair.sum())
W_aux = float(w_aux.sum())
qmax_pair = float(q_pair.max())
qmax_aux = float(qP.max())
rows = []
prev = None
print(f'\n{"cycle":>5s} {"max|dD|":>9s} {"max|dc|":>9s}   '
      + '  '.join(f'{"tau=%.0e" % t:>23s}' for t in taus))
print(f'{"":>5s} {"":>9s} {"":>9s}   '
      + '  '.join(f'{"dm   aux  auxpair":>23s}' for t in taus))
for n, d in enumerate(dms):
    cs, ds = coeffs(d)
    if prev is not None:
        dds = ds - prev[1]
        dcs = cs - prev[0]
        dmax = shell_max(dds)
        cmax = aux_shell_max(dcs)
        s_aux = float((qP * cmax).sum())
        row = {'cycle': n, 'max_dD': float(cp.abs(dds).max()),
               'max_dc': float(cp.abs(dcs).max()), 'tau': {}}
        cells = []
        for t in taus:
            keep_p = q_pair * dmax * qmax_aux >= t
            keep_k = qP * cmax * qmax_pair >= t
            keep_q = q_pair * s_aux >= t
            f_dm = float((w_pair * keep_p).sum()) / W_pair
            f_aux = float((w_aux * keep_k).sum()) / W_aux
            f_ap = float((w_pair * keep_q).sum()) / W_pair
            row['tau'][t] = {'dm': f_dm, 'aux': f_aux, 'auxvec_pairs': f_ap,
                             'auxvec_both': f_aux * f_ap}
            cells.append(f'{f_dm:6.3f} {f_aux:6.3f} {f_ap:6.3f}')
        rows.append(row)
        print(f'{n:5d} {row["max_dD"]:9.2e} {row["max_dc"]:9.2e}   '
              + '  '.join(f'{c:>23s}' for c in cells))
    prev = (cs, ds)

print('\n(columns: the fraction of the density-contraction kernel\'s pair work,'
      '\n of the auxiliary-vector kernel\'s auxiliary-shell work, and of its '
      'pair work,\n that a screen at tau keeps)')
out = {'xyz': a.xyz, 'basis': a.basis, 'xc': a.xc, 'natm': mol.natm,
       'cycles': mf.cycles, 'rows': rows}
if a.save:
    with open(a.save, 'w') as f:
        json.dump(out, f)
print('@@JSON ' + json.dumps(out))
