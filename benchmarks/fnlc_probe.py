"""Where does the VV10 response term's time go, and how much of it repeats?

`benchmarks/response_anatomy.py` puts `hessian.rks.nr_rks_fnlc_mo` at **80 %**
of a wB97M-V polarizability -- larger than the J and K build and the XC
response kernel put together, and not an entry in `instructions/06` at all.
This script says which part of it that is, by timing the two O(ngrids^2)
double sums and the grid work around them.

The distinction that matters: everything up to `f_gamma_i` depends only on the
**ground-state** density, which is the same on every call of a CPKS or
Davidson solve, while `VXC_vv10nlc_hess_eval_f_t` depends on the perturbed
densities and cannot be reused.

    python benchmarks/fnlc_probe.py --xyz benchmarks/molecules/penicillinV.xyz \
        --basis def2-svp --xc wb97m-v
"""
import json
import time
from collections import OrderedDict

import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--nset', type=int, default=3)
a = ap.parse_args()
apply_patch(a.patch)

from gpu4pyscf import dft                                   # noqa: E402
from gpu4pyscf.dft import numint as gnumint                 # noqa: E402
from gpu4pyscf.hessian import rks as hrks                    # noqa: E402

T = OrderedDict()
N = OrderedDict()
_sync = cp.cuda.Stream.null.synchronize

# The three libgdft entry points and the two grid contractions inside
# `nr_rks_fnlc_mo`.  ctypes caches function pointers in the CDLL instance's
# __dict__, so assigning over them wraps them.
LIB = hrks.libgdft
_CALLS = ['VXC_vv10nlc_hess_eval_omega_derivative',
          'VXC_vv10nlc_hess_eval_UWABCE',
          'VXC_vv10nlc_hess_eval_f_t',
          'VXC_vv10nlc_hess_eval_vmat']


def _wrap_c(name):
    try:
        orig = getattr(LIB, name)
    except AttributeError:
        return
    T.setdefault(name, 0.0)
    N.setdefault(name, 0)

    def timed(*args):
        _sync()
        t = time.perf_counter()
        out = orig(*args)
        _sync()
        T[name] += time.perf_counter() - t
        N[name] += 1
        return out
    setattr(LIB, name, timed)


for nm in _CALLS:
    _wrap_c(nm)


def _wrap_py(obj, name, label):
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
        return out
    setattr(obj, name, timed)


_wrap_py(gnumint, 'eval_rho2', 'eval_rho2 (ground state rho on the NLC grid)')
_wrap_py(gnumint, 'eval_rho', 'eval_rho (perturbed rho on the NLC grid)')

mol = build_mol(a)
print(f'# {a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}/L{a.grids}  '
      f'natm={mol.natm} nao={mol.nao}', flush=True)
mf = dft.RKS(mol, xc=a.xc)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
e = mf.kernel()
print(f'# SCF {e:.9f}  converged={mf.converged}', flush=True)
print(f'# nlcgrids {mf.nlcgrids.coords.shape[0]} points, '
      f'xc grids {mf.grids.coords.shape[0]}', flush=True)

mo = cp.asarray(mf.mo_coeff)
occ = cp.asarray(mf.mo_occ)
en = cp.asarray(mf.mo_energy)
mocc, mvir = mo[:, occ > 0], mo[:, occ == 0]
de = en[occ == 0][:, None] - en[occ > 0][None, :]
with mol.with_common_orig((0, 0, 0)):
    h1 = cp.asarray(mol.intor('int1e_r'))
xs = [-(mvir.T @ h1[i] @ mocc) / de for i in range(3)]
dm1 = 2 * cp.asarray([mvir @ x @ mocc.T for x in xs[:a.nset]])
dm1 = cp.asarray(dm1 + dm1.transpose(0, 2, 1), order='C')

for k in list(T):
    T[k] = 0.0
    N[k] = 0
_sync()
t = time.perf_counter()
for _ in range(a.reps):
    v = hrks.nr_rks_fnlc_mo(mf, mol, mo, occ, dm1, return_in_mo=False)
_sync()
total = (time.perf_counter() - t) / a.reps
print(f'# checksum {float(cp.asarray(v).sum()):.12e}', flush=True)

print(f'\n{"stage":52s} {"s/call":>9s} {"share":>7s} {"calls":>6s}')
print('-' * 78)
print(f'{"TOTAL (one nr_rks_fnlc_mo call)":52s} {total:9.4f} {100.0:6.1f}%')
gs = 0.0
for k in T:
    if N[k] == 0:
        continue
    per = T[k] / a.reps
    print(f'{k:52s} {per:9.4f} {100*per/total:6.1f}% {N[k]//a.reps:6d}')
    if 'f_t' not in k and 'vmat' not in k and 'perturbed' not in k:
        gs += per
print('-' * 78)
print(f'{"ground-state part (the same on every call)":52s} {gs:9.4f} '
      f'{100*gs/total:6.1f}%')

print('\n' + json.dumps({'tag': f'{a.xyz}/{a.basis}/{a.xc}', 'total': total,
                         'ground_state': gs,
                         'times': {k: T[k]/a.reps for k in T},
                         'calls': {k: N[k]//a.reps for k in N}}))
