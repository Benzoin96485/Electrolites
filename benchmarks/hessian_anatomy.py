"""Where does an analytic Hessian's wall clock go, on GPU4PySCF 1.8.1?

The catalogue's Hessian profile (`instructions/06` 2.3) is a 1.7.0 profile, and
1.8.1 moved the CPHF solve into the MO basis: `hessian.rhf.solve_mo1` now calls
`get_veff_resp_mo`, which calls `nr_rks_fxc_mo` and a multi-density `get_j` /
`get_k`, and the per-charge `hess_nuc_elec` loop the catalogue names is gone.
So the shares have to be re-measured before anything is written.

Every stage is wrapped with a `cudaDeviceSynchronize()` on both sides, so the
rows are attributable.  Nested rows (prefixed `..`) are sub-intervals of their
parent and overlap it, so a column does not sum to its parent.

    python benchmarks/hessian_anatomy.py \
        --xyz benchmarks/molecules/penicillinV.xyz --basis def2-svp [--patch all]

`--cphf-only` skips `partial_hess_elec` and `make_h1` and times the CPHF solve
alone, which is what to use when iterating on a CPHF target.
"""
import json
import time
from collections import OrderedDict

import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--cphf-only', action='store_true')
# 1e-6 is GPU4PySCF's own default (scf/hf.py:702)
ap.add_argument('--conv-tol-cpscf', type=float, default=1e-6)
a = ap.parse_args()
apply_patch(a.patch)

import numpy as np                                          # noqa: E402
from gpu4pyscf import dft                                   # noqa: E402
from gpu4pyscf.hessian import rhf as hrhf, rks as hrks       # noqa: E402
from gpu4pyscf.grad import rhf as grhf                       # noqa: E402
from gpu4pyscf.scf import jk as sjk, j_engine as sje         # noqa: E402
from gpu4pyscf.pbc.gto import int1e as pint1e                # noqa: E402

T = OrderedDict()
N = OrderedDict()
NDM = OrderedDict()
_sync = cp.cuda.Stream.null.synchronize


def _wrap(obj, name, label, ndm_from=None):
    orig = getattr(obj, name)
    T.setdefault(label, 0.0)
    N.setdefault(label, 0)

    def timed(*args, **kwargs):
        _sync()
        t = time.perf_counter()
        out = orig(*args, **kwargs)
        _sync()
        T[label] += time.perf_counter() - t
        N[label] += 1
        if ndm_from is not None:
            dms = args[ndm_from]
            n = 1 if getattr(dms, 'ndim', 2) == 2 else len(dms)
            NDM.setdefault(label, set()).add(n)
        return out
    timed.__name__ = getattr(orig, '__name__', name)
    setattr(obj, name, timed)


# (object, attribute, label, indent) -- class attributes where the method is
# bound at class creation (`make_h1 = make_h1`), module attributes where the
# call site looks the name up as a global at call time.
Hess = hrks.Hessian
PLAN = [
    (Hess, 'partial_hess_elec', 'partial_hess_elec', 0),
    (hrhf, '_partial_ejk_ip2', '..ejk_ip2', 1),
    (hrhf, '_hcore_energy', '..hcore_energy', 1),
    (hrks, '_get_exc_deriv2', '..exc_deriv2', 1),
    (hrks, '_get_vxc_diag', '....vxc_diag', 2),
    (hrks, '_get_vxc_deriv2', '....vxc_deriv2', 2),
    (hrks, '_get_enlc_deriv2', '..enlc_deriv2', 1),
    (Hess, 'make_h1', 'make_h1', 0),
    (hrks, '_get_vxc_deriv1', '..vxc_deriv1', 1),
    (hrks, '_get_vnlc_deriv1', '..vnlc_deriv1', 1),
    (grhf, 'get_grad_hcore', '..grad_hcore', 1),
    (hrhf, '_get_jk_ip1', '..jk_ip1', 1),
    (Hess, 'solve_mo1', 'solve_mo1', 0),
    (Hess, 'get_veff_resp_mo', '..veff_resp_mo', 1),
    (hrks, 'nr_rks_fxc_mo', '....nr_rks_fxc_mo', 2),
    (hrks, 'nr_rks_fnlc_mo', '....nr_rks_fnlc_mo', 2),
    (Hess, 'get_jk_mo', '....jk_mo', 2),
    (sje._VHFOpt, 'get_j', '......get_j', 3),
    (sjk._VHFOpt, 'get_k', '......get_k', 3),
    (pint1e, 'int1e_ipovlp', 'int1e_ipovlp', 0),
]
for obj, name, label, _ in PLAN:
    # get_j/get_k are wrapped on the class, so args[0] is the _VHFOpt.
    _wrap(obj, name, label, ndm_from=1 if name in ('get_j', 'get_k') else None)


mol = build_mol(a)
tag = (f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}/L{a.grids}/'
       f'{a.patch or "stock"}')
print(f'# {tag}  natm={mol.natm} nao={mol.nao}', flush=True)

mf = dft.RKS(mol, xc=a.xc)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.conv_tol_cpscf = a.conv_tol_cpscf
_sync()
t = time.perf_counter()
e = mf.kernel()
_sync()
t_scf = time.perf_counter() - t
print(f'# SCF {e:.9f} in {t_scf:.2f} s, converged={mf.converged}', flush=True)

# The SCF's own get_j/get_k calls are not part of the Hessian.
for k in list(T):
    T[k] = 0.0
    N[k] = 0
NDM.clear()

h = mf.Hessian()
h.auxbasis_response = 0
_sync()
t = time.perf_counter()
if a.cphf_only:
    mo_coeff, mo_occ, mo_energy = mf.mo_coeff, mf.mo_occ, mf.mo_energy
    h1mo = h.make_h1(mo_coeff, mo_occ, None, None, mol.verbose)
    fx = h.gen_vind(mo_coeff, mo_occ)
    _sync()
    t = time.perf_counter()
    mo1, mo_e1 = h.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx, None,
                             4000, mol.verbose)
    _sync()
    total = time.perf_counter() - t
    ref = float(cp.asarray(mo1).sum()) if not isinstance(mo1, np.ndarray) \
        else float(mo1.sum())
    print(f'# CPHF only: sum(mo1) = {ref:.12e}', flush=True)
else:
    de2 = h.kernel()
    _sync()
    total = time.perf_counter() - t
    de2 = cp.asarray(de2)
    h2 = de2.transpose(0, 2, 1, 3).reshape(mol.natm * 3, mol.natm * 3)
    print(f'# hessian trace = {float(cp.trace(h2)):.12e}  '
          f'frobenius = {float(cp.linalg.norm(h2)):.12e}', flush=True)

print(f'\n{"stage":26s} {"time/s":>9s} {"share":>7s} {"calls":>6s}  n_dm')
print('-' * 62)
print(f'{"TOTAL":26s} {total:9.3f} {100.0:6.1f}%')
for _, name, label, ind in PLAN:
    if N.get(label, 0) == 0:
        continue
    ndm = NDM.get(label)
    ndm = '' if not ndm else ','.join(str(x) for x in sorted(ndm))
    print(f'{label:26s} {T[label]:9.3f} {100*T[label]/total:6.1f}% '
          f'{N[label]:6d}  {ndm}')
acc = sum(T[l] for _, _, l, ind in PLAN if ind == 0)
print('-' * 62)
print(f'{"accounted (indent 0)":26s} {acc:9.3f} {100*acc/total:6.1f}%')

out = {'tag': tag, 'natm': mol.natm, 'nao': int(mol.nao), 'total': total,
       'scf': t_scf, 'cphf_only': a.cphf_only,
       'times': dict(T), 'calls': dict(N),
       'n_dm': {k: sorted(v) for k, v in NDM.items()}}
print('\n' + json.dumps(out))
