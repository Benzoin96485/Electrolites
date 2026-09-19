"""How much of the DF gradient's last digits the fitting metric owns.

`fastdfgrad` builds the pseudo-density as ``aux_coef @ (aux_coefᵀ @ j3c_oo)``
-- through the cderi -- where ``df/grad/rhf.py:_jk_energy_per_atom`` builds it
as ``metric @ j3c_oo``.  The two are the same matrix, because
``aux_coef @ aux_coefᵀ == metric`` by construction (``_decompose_j2c``), and
the patched and unpatched gradients nevertheless differ in the last digits.

This says whose digits they are, and it does so **without the cderi**: it
takes GPU4PySCF's *own* ``j3c_oo``, contracts it with the metric both ways,
and reports the disagreement next to the condition number and the element
size of the metric that bounds it.  Anything the cderi contributed would show
up as the difference between this number and the one
``benchmarks/df_grad_bench.py`` reports for the whole term.

    python benchmarks/df_grad_metric_probe.py \
        --xyz benchmarks/molecules/taxol.xyz --basis def2-svp

`--omega` adds the long-range operator, where the 2c2e matrix is rank
deficient and both routes go through a filtered pseudo-inverse instead.
"""
import json

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--omega', type=float, default=0.0,
                help='also do the long-range operator at this omega')
ap.add_argument('--no-eig', action='store_true',
                help='skip the eigenvalues of j2c, which are an O(naux^3) '
                     'decomposition of their own')
a = ap.parse_args()
apply_patch(a.patch)

import gpu4pyscf                                             # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.df import df as DFMOD                         # noqa: E402
from gpu4pyscf.df.grad import rhf as df_rhf_grad             # noqa: E402
from gpu4pyscf.df.int3c2e_bdiv import (                      # noqa: E402
    Int3c2eOpt, int2c2e, argsort_aux, _split_l_ctr_pattern)
from gpu4pyscf.gto.mole import SortedMole                    # noqa: E402

mol = build_mol(a)
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc).density_fit(
    auxbasis=a.auxbasis)
mf.conv_tol = a.conv_tol
mf.grids.level = a.grids
mf.verbose = 0
if a.cycles:
    mf.max_cycle = a.cycles
mf.kernel()
dm = mf.make_rdm1()
omega_xc, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
auxmol_raw = mf.with_df.auxmol
mf.with_df.reset()

# `_jk_energy_per_atom` forms `dm_oo` with exactly one `cp.einsum` call, and
# it is the one this probe is about.  Intercepting it is how the *upstream*
# `j3c_oo` is obtained rather than a reimplementation of it.
_orig_einsum = cp.einsum
captured = {}


def _spy(subs, *ops, **kw):
    if subs == 'uv,vij->uij':
        captured['metric'], captured['j3c_oo'] = ops
    return _orig_einsum(subs, *ops, **kw)


R = []
omegas = [0.0] + ([a.omega] if a.omega else [])
for om in omegas:
    with mol.with_range_coulomb(om), auxmol_raw.with_range_coulomb(om):
        opt = Int3c2eOpt(SortedMole.from_mol(mol, decontract=True),
                         auxmol_raw).build()
        captured.clear()
        cp.einsum = _spy
        try:
            df_rhf_grad._jk_energy_per_atom(
                opt, dm, 1 if om == 0 else 0, hyb or 1., hermi=1,
                auxbasis_response=True)
        finally:
            cp.einsum = _orig_einsum
        metric, j3c_oo = captured['metric'], captured['j3c_oo']

        aux = opt.auxmol
        offsets = np.append(0, np.cumsum(aux.l_ctr_counts))
        sorting = argsort_aux(offsets, aux.uniq_l_ctr)
        aux_coef, tag = DFMOD._decompose_j2c(aux, sorting)

        naux_cart, nocc = j3c_oo.shape[0], j3c_oo.shape[-1]
        a1 = _orig_einsum('uv,vij->uij', metric, j3c_oo)
        a2 = aux_coef.dot(aux_coef.T.dot(
            j3c_oo.reshape(naux_cart, -1))).reshape(-1, nocc, nocc)
        scale = float(cp.abs(a1).max())
        assoc = float(cp.abs(a1 - a2).max())
        m_scale = float(cp.abs(metric).max())
        m_err = float(cp.abs(metric - aux_coef.dot(aux_coef.T)).max())

        row = dict(omega=om, tag=tag, naux_cart=int(naux_cart),
                   nkept=int(aux_coef.shape[1]), nocc=int(nocc),
                   metric_max=m_scale, metric_err=m_err / m_scale,
                   dm_oo_max=scale, assoc=assoc, assoc_rel=assoc / scale)
        if not a.no_eig:
            w = cp.linalg.eigvalsh(int2c2e(aux))
            row['w_min'] = float(w[0])
            row['w_max'] = float(w[-1])
            row['cond'] = float(w[-1] / w[0])
            row['n_below_thr'] = int((w < DFMOD.LINEAR_DEP_THR).sum())
        R.append(row)

        print(f'omega={om:.4f}  tag={tag}  naux_cart={naux_cart} '
              f'nkept={aux_coef.shape[1]} nocc={nocc}', flush=True)
        if not a.no_eig:
            print(f'  j2c   eigenvalues {row["w_min"]:.3e} .. '
                  f'{row["w_max"]:.3e}   cond {row["cond"]:.3e}   '
                  f'{row["n_below_thr"]} below LINEAR_DEP_THR '
                  f'({DFMOD.LINEAR_DEP_THR:g})', flush=True)
        print(f'  |metric|max                      {m_scale:.3e}', flush=True)
        print(f'  |metric - aux_coef@aux_coefT|    {m_err:.3e}   '
              f'rel {m_err/m_scale:.3e}', flush=True)
        print(f'  |dm_oo|max                       {scale:.3e}', flush=True)
        print(f'  the two associations differ by   {assoc:.3e}   '
              f'rel {assoc/scale:.3e}', flush=True)

print('RESULT ' + json.dumps(dict(
    xyz=a.xyz, basis=a.basis, auxbasis=a.auxbasis, xc=a.xc, natm=mol.natm,
    nao=int(mol.nao), gpu4pyscf=gpu4pyscf.__version__, rows=R)), flush=True)
