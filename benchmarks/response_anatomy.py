"""Where does a response-property wall clock go, on GPU4PySCF 1.8.1?

The catalogue's polarizability and TDA profiles (`instructions/06` 2.4, 2.5)
are 1.7.0 profiles taken with a sampling profiler, and they leave 18.4 s of a
44.31 s polarizability "not instrumented; mostly `nr_rks_fxc`".  This script
attributes it, with a `cudaDeviceSynchronize()` on both sides of every stage,
so the same two numbers can be read off 1.8.1 before anything is written.

    python benchmarks/response_anatomy.py --prop pol \
        --xyz benchmarks/molecules/penicillinV.xyz --basis def2-svp [--patch all]
    python benchmarks/response_anatomy.py --prop tda --nstates 5 ...

Nested rows (prefixed `..`) are sub-intervals of their parent and overlap it,
so a column does not sum to its parent.  `nset` is the set of density counts
one stage was asked for, which is the whole point on a response path: every
row of it is a multi-density call.
"""
import json
import time
from collections import OrderedDict

import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--prop', default='pol', choices=('pol', 'tda', 'tddft'))
ap.add_argument('--nstates', type=int, default=5)
ap.add_argument('--cphf-tol', type=float, default=1e-7)
a = ap.parse_args()
apply_patch(a.patch)

import numpy as np                                          # noqa: E402
from gpu4pyscf import dft                                   # noqa: E402
from gpu4pyscf.dft import numint as gnumint                 # noqa: E402
from gpu4pyscf.hessian import rks as hrks                   # noqa: E402
from gpu4pyscf.properties import polarizability as gpol     # noqa: E402
from gpu4pyscf.tdscf import rhf as gtdrhf                   # noqa: E402
from gpu4pyscf.tdscf import rks as gtdrks                   # noqa: E402
from gpu4pyscf.scf import cphf as gcphf                     # noqa: E402

T = OrderedDict()
N = OrderedDict()
NSET = OrderedDict()
_sync = cp.cuda.Stream.null.synchronize


def _nset(x):
    """Number of density matrices / trial vectors in a call argument."""
    n = getattr(x, 'ndim', None)
    if n is None:
        return None
    return 1 if n == 2 else (x.shape[0] if n == 3 else None)


def _wrap(obj, name, label, nset_from=None):
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
        if nset_from is not None and len(args) > nset_from:
            n = _nset(args[nset_from])
            if n is not None:
                NSET.setdefault(label, set()).add(n)
        return out
    timed.__name__ = getattr(orig, '__name__', name)
    setattr(obj, name, timed)


NI = gnumint.NumInt
# `nr_rks_fxc`/`nr_rks_fxc_st` are bound in the class body, so the class
# attribute is the one the call site reaches (catalogue 8's warning).
# `eval_ao` is looked up as a module global inside `_block_loop`.
PLAN = [
    ('solve', 'the linear solve / Davidson', 0),
    ('..vind', '..the matrix-vector product', 1),
    ('....fxc', '....XC response (nr_rks_fxc*)', 2),
    ('......eval_ao', '......eval_ao (AOs on the grid)', 3),
    ('....jk', '....Coulomb + exchange response', 2),
    ('....fnlc', '....VV10 response', 2),
]
_wrap(NI, 'nr_rks_fxc', '....fxc', nset_from=5)
_wrap(NI, 'nr_rks_fxc_st', '....fxc', nset_from=5)
_wrap(gnumint, 'eval_ao', '......eval_ao')
_wrap(hrks, 'nr_rks_fnlc_mo', '....fnlc')
# the matrix-vector product itself: `gen_vind` returns a closure, so the
# closure is what has to be wrapped, not the generator.
for _cls, _gen in ((gtdrhf.TDA, 'gen_vind'), (gtdrhf.TDHF, 'gen_vind')):
    _orig_gen = getattr(_cls, _gen)

    def _wrapped_gen(self, mf=None, _orig_gen=_orig_gen):
        vind, hdiag = _orig_gen(self, mf)
        T.setdefault('..vind', 0.0)
        N.setdefault('..vind', 0)

        def timed(zs):
            _sync()
            t = time.perf_counter()
            out = vind(zs)
            _sync()
            T['..vind'] += time.perf_counter() - t
            N['..vind'] += 1
            NSET.setdefault('..vind', set()).add(len(zs))
            return out
        return timed, hdiag
    setattr(_cls, _gen, _wrapped_gen)
# `tdscf.rhf` and `tdscf.rks` do `from ._lr_eig import eigh as lr_eigh` at
# import time, so the name to wrap is theirs, not `_lr_eig`'s.
for _mod in (gtdrhf, gtdrks):
    if hasattr(_mod, 'lr_eigh'):
        _wrap(_mod, 'lr_eigh', 'solve')
_wrap(gcphf, 'solve', 'solve')

mol = build_mol(a)
tag = (f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}/L{a.grids}/'
       f'{a.prop}/{a.patch or "stock"}')
print(f'# {tag}  natm={mol.natm} nao={mol.nao}', flush=True)

mf = dft.RKS(mol, xc=a.xc)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
_sync()
t = time.perf_counter()
e = mf.kernel()
_sync()
t_scf = time.perf_counter() - t
print(f'# SCF {e:.9f} in {t_scf:.2f} s, converged={mf.converged}', flush=True)

# `mf.get_jk` etc. are wrapped on the instance so the closure in
# `_gen_rhf_response` reaches them; the SCF above must not be counted.
_wrap(mf, 'get_jk', '....jk', nset_from=1)
_wrap(mf, 'get_j', '....jk', nset_from=1)
_wrap(mf, 'get_k', '....jk', nset_from=1)
for k in list(T):
    T[k] = 0.0
    N[k] = 0
NSET.clear()

_sync()
t = time.perf_counter()
if a.prop == 'pol':
    _orig_fx = gpol.gen_vind

    def _timed_gen_vind(mf, mo_coeff, mo_occ, with_nlc=True):
        fx = _orig_fx(mf, mo_coeff, mo_occ, with_nlc=with_nlc)
        T.setdefault('..vind', 0.0)
        N.setdefault('..vind', 0)

        def timed(mo1):
            _sync()
            t = time.perf_counter()
            out = fx(mo1)
            _sync()
            T['..vind'] += time.perf_counter() - t
            N['..vind'] += 1
            return out
        return timed
    gpol.gen_vind = _timed_gen_vind
    out = gpol.eval_polarizability(mf, tol=a.cphf_tol)
    _sync()
    total = time.perf_counter() - t
    print(f'# polarizability trace = {np.trace(out):.12e}  '
          f'frobenius = {np.linalg.norm(out):.12e}', flush=True)
else:
    td = mf.TDA() if a.prop == 'tda' else mf.TDDFT()
    td.nstates = a.nstates
    td.kernel()
    _sync()
    total = time.perf_counter() - t
    ev = np.asarray(td.e) * 27.211386245988
    print('# excitation energies / eV = '
          + ' '.join(f'{x:.9f}' for x in ev), flush=True)

print(f'\n{"stage":34s} {"time/s":>9s} {"share":>7s} {"calls":>6s}  nset')
print('-' * 70)
print(f'{"TOTAL":34s} {total:9.3f} {100.0:6.1f}%')
for key, label, ind in PLAN:
    if N.get(key, 0) == 0:
        continue
    ns = NSET.get(key)
    ns = '' if not ns else ','.join(str(x) for x in sorted(ns))
    print(f'{label:34s} {T[key]:9.3f} {100*T[key]/total:6.1f}% '
          f'{N[key]:6d}  {ns}')
acc = sum(T[k] for k, _, ind in PLAN if ind == 0)
print('-' * 70)
print(f'{"accounted (indent 0)":34s} {acc:9.3f} {100*acc/total:6.1f}%')
print(f'{"remainder":34s} {total-acc:9.3f} {100*(total-acc)/total:6.1f}%')

out = {'tag': tag, 'prop': a.prop, 'natm': mol.natm, 'nao': int(mol.nao),
       'total': total, 'scf': t_scf, 'times': dict(T), 'calls': dict(N),
       'nset': {k: sorted(v) for k, v in NSET.items()}}
print('\n' + json.dumps(out))
