"""Catalogue B3 (rank 9): where does `ejk_ip2` actually spend its time?

`instructions/06`'s B3 entry says the second-derivative ERI kernels are "the
same shape as B1" -- the gradient's `ejk_ip1`, which `fastejk` rewrote -- and
that the P1-P5 treatment therefore transfers, with P3 (the register cap) worth
more here because "register pressure [is] worse than high-L ERIs at the same
nominal basis L".

That description fits the *unrolled* ip2 kernels.  But GPU4PySCF unrolls only
**seven** angular-momentum classes for each of the two ip2 types --
``0000 1000 1010 1011 1100 1110 1111``, i.e. nothing above ``(pp|pp)`` -- while
its exchange build unrolls twenty-five.  Every class with a d function on any
centre falls through to ``rys_contract_jk_ip2.cu``'s *general* kernel, which
carries its g-array addresses in shared ``idx_i/j/k/l`` arrays and its loop
bounds at run time.  So for any basis with polarisation functions -- 6-31G*,
def2-SVP, anything larger -- B3 is not "as B1"; it is the situation
``gen_khigh.py`` was written for, one shell of angular momentum earlier.

This measures that split.  Each ``RYS_per_atom_jk_ip2_type{12,3}`` call is
wrapped with a ``cudaDeviceSynchronize()`` on both sides and attributed to its
angular-momentum class, which the driver derives from ``shls_slice`` exactly as
the kernel does, so "unrolled" and "general" are read off the class rather than
guessed.

    python benchmarks/ejk_ip2_probe.py --xyz molecules/penicillinV.xyz \
        --basis def2-svp
"""
import ctypes
import json
import time
from collections import OrderedDict

import cupy as cp
import numpy as np

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
a = ap.parse_args()
apply_patch(a.patch)

from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.hessian import rhf as hrhf                    # noqa: E402
from gpu4pyscf.scf import jk as JK                           # noqa: E402

#: the classes GPU4PySCF has a dedicated ip2 kernel for, both types
UNROLLED = {'0000', '1000', '1010', '1011', '1100', '1110', '1111'}

_sync = cp.cuda.Stream.null.synchronize
T = OrderedDict()          # (type, class) -> seconds
N = OrderedDict()


def _wrap(name):
    orig = getattr(JK.libvhf_rys, name)
    kind = 'type12' if name.endswith('type12') else 'type3'

    def timed(*args):
        # args[8] is int shls_slice[8]; the driver reads the angular momenta of
        # the four leading shells out of it, and so do we
        sl = args[8]
        bas = MOL_BAS
        lll = ''.join(str(int(bas[sl[2*i]][1])) for i in range(4))
        _sync()
        t = time.perf_counter()
        out = orig(*args)
        _sync()
        key = (kind, lll)
        T[key] = T.get(key, 0.0) + time.perf_counter() - t
        N[key] = N.get(key, 0) + 1
        return out

    timed.restype = getattr(orig, 'restype', ctypes.c_int)
    setattr(JK.libvhf_rys, name, timed)


mol = build_mol(a)
tag = f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}/{a.patch or "stock"}'
print(f'# {tag}  natm={mol.natm} nao={mol.nao}', flush=True)

mf = dft.RKS(mol, xc=a.xc)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
e = mf.kernel()
print(f'# SCF {e:.9f} converged={mf.converged}', flush=True)

h = mf.Hessian()
h.auxbasis_response = 0
# the sorted molecule the kernels index, which is not mol
opt = JK._VHFOpt(mol, mf.direct_scf_tol).build()
MOL_BAS = opt.sorted_mol._bas
_wrap('RYS_per_atom_jk_ip2_type12')
_wrap('RYS_per_atom_jk_ip2_type3')

dm0 = cp.asarray(mf.make_rdm1())
_sync()
t = time.perf_counter()
ejk = hrhf._partial_ejk_ip2(mol, dm0, opt, 1.0, 1.0)
_sync()
total = time.perf_counter() - t
print(f'# _partial_ejk_ip2 {total:.3f} s, trace of its diagonal = '
      f'{float(cp.asarray(ejk).sum()):.9e}', flush=True)

acc = sum(T.values())
groups = {'type12/unrolled': 0.0, 'type12/general': 0.0,
          'type3/unrolled': 0.0, 'type3/general': 0.0}
for (kind, lll), t_ in T.items():
    groups[f'{kind}/{"unrolled" if lll in UNROLLED else "general"}'] += t_

print(f'\n{"group":24s} {"time/s":>9s} {"share of kernels":>17s} '
      f'{"share of stage":>15s}')
print('-' * 70)
for g, t_ in groups.items():
    print(f'{g:24s} {t_:9.3f} {100*t_/acc:16.1f}% {100*t_/total:14.1f}%')
print('-' * 70)
print(f'{"all kernels":24s} {acc:9.3f} {100.0:16.1f}% {100*acc/total:14.1f}%')

rows = sorted(T.items(), key=lambda kv: -kv[1])
print(f'\nthe twelve most expensive classes\n{"type":8s} {"class":6s} '
      f'{"unrolled":9s} {"time/s":>8s} {"calls":>6s} {"share":>7s}')
print('-' * 52)
for (kind, lll), t_ in rows[:12]:
    print(f'{kind:8s} {lll:6s} {"yes" if lll in UNROLLED else "no":9s} '
          f'{t_:8.3f} {N[(kind, lll)]:6d} {100*t_/acc:6.1f}%')

print('\n' + json.dumps({
    'tag': tag, 'natm': mol.natm, 'nao': int(mol.nao), 'total': total,
    'groups': groups,
    'per_class': {f'{k}/{c}': v for (k, c), v in T.items()},
    'calls': {f'{k}/{c}': v for (k, c), v in N.items()}}))
