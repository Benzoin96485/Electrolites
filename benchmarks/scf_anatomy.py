"""Where does an SCF (and, unless --no-grad, a gradient) wall clock go?

`fixed_density.py` says `get_veff` is 3.3x faster patched while the SCF wall
clock is only 2.1x, so about a third of that wall clock is somewhere else.
This script says where.  Every method the SCF driver calls per cycle is
wrapped with a `cudaDeviceSynchronize()` on both sides, so the columns are
attributable; `total` is the wall clock of `mf.kernel()` including the
synchronisation points this adds.

    python benchmarks/scf_anatomy.py --xyz benchmarks/molecules/PfPMT.xyz \
        --charge -1 [--patch all]

Nested rows (prefixed `..`) are sub-intervals of their parent and overlap it,
so a column does not sum to its parent.
"""
import json
import time
from collections import OrderedDict

import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--max-cycle', type=int, default=50)
a = ap.parse_args()
apply_patch(a.patch)

import gpu4pyscf                                            # noqa: E402
from gpu4pyscf import dft                                   # noqa: E402
from gpu4pyscf.scf import hf as ghf                         # noqa: E402
from gpu4pyscf.dft import numint as gnumint               # noqa: E402
from gpu4pyscf.grad import rhf as ggrhf                     # noqa: E402
from gpu4pyscf.grad import rks as ggrks, uks as gguks        # noqa: E402
from gpu4pyscf.lib import diis as gdiis                      # noqa: E402
from gpu4pyscf.lib import cupy_helper as gch                # noqa: E402

T = OrderedDict()
N = OrderedDict()
_sync = cp.cuda.Stream.null.synchronize


def _wrap(obj, name, label, is_class=True):
    """Wrap obj.name with a synchronised timer recorded under `label`."""
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
    timed.__name__ = getattr(orig, '__name__', name)
    setattr(obj, name, timed)


mol = build_mol(a)
tag = f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}/{a.patch or "stock"}'
print(f'[{tag}] natm={mol.natm} nbas={mol.nbas} nao={mol.nao} '
      f'spin={mol.spin} charge={mol.charge} lmax={int(mol._bas[:, 1].max())} '
      f'gpu4pyscf={gpu4pyscf.__version__}', flush=True)

mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc)
mf.conv_tol = a.conv_tol
mf.direct_scf_tol = 1e-13
mf.grids.level = a.grids
mf.verbose = 0
mf.max_cycle = a.max_cycle

# --- the per-cycle driver calls, in the order scf.hf._kernel makes them ----
_wrap(mf, 'get_fock', 'get_fock(+DIIS)')
_wrap(mf, 'eig', 'eig')
_wrap(mf, 'get_occ', 'get_occ')
_wrap(mf, 'make_rdm1', 'make_rdm1')
_wrap(mf, 'get_veff', 'get_veff')
_wrap(mf, 'energy_tot', 'energy_tot')
_wrap(mf, 'get_grad', 'get_grad')
_wrap(mf, 'get_hcore', 'get_hcore')
_wrap(mf, 'get_ovlp', 'get_ovlp')
_wrap(mf, 'get_init_guess', 'get_init_guess')
_wrap(mf, 'check_linear_dependency', 'check_lindep')

# --- inside get_veff: J, K, XC, and the grid build -------------------------
_wrap(mf, 'get_j', '..get_j')
_wrap(mf, 'get_k', '..get_k')
_wrap(mf, 'get_jk', '..get_jk')
_wrap(mf._numint, 'nr_rks' if not mol.spin else 'nr_uks',
      '..nr_rks' if not mol.spin else '..nr_uks')
_wrap(mf.grids, 'build', '..grids.build')
if hasattr(mf, 'nlcgrids'):
    _wrap(mf.nlcgrids, 'build', '..nlcgrids.build')
_wrap(mf._numint, 'nr_nlc_vxc', '..nr_nlc_vxc')
_wrap(gnumint, '_vv10nlc', '..vv10nlc')
# `fastxcnlc` runs the XC and NLC grid passes *before* GPU4PySCF's own
# `get_veff`, and then answers `nr_rks`/`nr_uks` from the result -- so with it
# loaded the `..nr_rks`/`..nr_uks` and `..nr_nlc_vxc` rows go to nothing and
# the work lands here instead.  Both columns are needed to read the fusion:
# stock puts it in the first two, patched in this one.
try:
    from electrolites import fastxcnlc as _fxnlc
    _wrap(_fxnlc, 'nr_uks_nlc' if mol.spin else 'nr_rks_nlc',
          '..fused xc+nlc')
except ImportError:
    pass

# --- inside get_fock: what DIIS actually costs -----------------------------
_wrap(gdiis.DIIS, 'update', '..diis.update')
_wrap(gdiis.DIIS, 'extrapolate', '..diis.extrapolate')
# --- inside eig: the fp64 generalised eigendecomposition -------------------
if hasattr(gch, 'eigh'):
    _wrap(gch, 'eigh', '..cupy_helper.eigh')

# --- the gradient's stages, in the order grad_elec makes them.  Their labels
#     are collected separately: `_wrap` registers a label as soon as it wraps,
#     so a row cannot be assigned to the SCF or the gradient by when it first
#     appears in `T`.
_GRAD_KEYS = []
if not a.no_grad:
    for _obj, _name, _label in (
            (ggrhf, '_hcore_energy', 'grad hcore+ovlp'),
            (ggrhf, '_jk_energy_per_atom', 'grad ejk'),
            (gguks if mol.spin else ggrks, 'get_exc', 'grad xc'),
            (ggrks, 'get_nlc_exc', 'grad nlc')):
        _wrap(_obj, _name, _label)
        _GRAD_KEYS.append(_label)

t = time.perf_counter()
e = mf.kernel()
_sync()
total = time.perf_counter() - t

grad_total = 0.0
if not a.no_grad:
    g = mf.nuc_grad_method()
    g.verbose = 0
    _sync()
    t = time.perf_counter()
    de = g.kernel()
    _sync()
    grad_total = time.perf_counter() - t

print(f'[{tag}] SCF {total:9.3f} s  cycles={mf.cycles}  '
      f'E={e:.10f} conv={mf.converged}', flush=True)
print(f'{"":30s} {"time":>9s} {"share":>7s} {"calls":>6s}')
top = sum(v for k, v in T.items()
          if not k.startswith('..') and k not in _GRAD_KEYS)
for k, v in T.items():
    if N[k] == 0 or k in _GRAD_KEYS:
        continue
    print(f'{k:30s} {v:9.3f} {100*v/total:6.1f}% {N[k]:6d}')
print(f'{"[sum of top-level rows]":30s} {top:9.3f} {100*top/total:6.1f}%')
print(f'{"[unattributed]":30s} {total-top:9.3f} {100*(total-top)/total:6.1f}%')

if grad_total:
    print(f'\n[{tag}] GRADIENT {grad_total:9.3f} s  '
          f'|de|={float(abs(de).max()):.8f}')
    gsum = 0.0
    for k in _GRAD_KEYS:
        if N[k] == 0:
            continue
        gsum += T[k]
        print(f'{k:30s} {T[k]:9.3f} {100*T[k]/grad_total:6.1f}% {N[k]:6d}')
    print(f'{"[sum of grad rows]":30s} {gsum:9.3f} '
          f'{100*gsum/grad_total:6.1f}%')
    print(f'{"[grad unattributed]":30s} {grad_total-gsum:9.3f} '
          f'{100*(grad_total-gsum)/grad_total:6.1f}%')

print('RESULT ' + json.dumps(dict(
    tag=tag, gpu4pyscf=gpu4pyscf.__version__, xyz=a.xyz, basis=a.basis,
    xc=a.xc, patch=a.patch, natm=mol.natm, nao=int(mol.nao),
    spin=int(mol.spin), charge=int(mol.charge), e=float(e),
    conv=bool(mf.converged), cycles=int(mf.cycles), total=total,
    grad_total=grad_total,
    t={k: v for k, v in T.items() if N[k]},
    n={k: v for k, v in N.items() if v})), flush=True)
