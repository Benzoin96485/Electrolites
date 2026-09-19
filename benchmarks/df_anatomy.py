"""Where does a *density-fitted* SCF (and its gradient) wall clock go?

`scf_anatomy.py` answers this for the direct-SCF path, which is the only path
every module in this repository is on.  A `density_fit()` run takes a
different one: `df/df_jk.py get_jk` replaces `scf/jk.py`'s J and K builds
entirely, so `fastj`, `fastk` and `fastjk` are not on it at all, and the whole
two-electron half is stock GPU4PySCF with `patch_all()` loaded.  This script
says how much of the run that is, and splits the DF build into the three parts
that can be attacked separately: the one-time cderi build, the per-block
unpacking inside `DF.loop`, and the contractions that consume the blocks.

    python benchmarks/df_anatomy.py --xyz benchmarks/molecules/taxol.xyz \
        [--auxbasis def2-universal-jkfit] [--patch all]

Nested rows (prefixed `..`) are sub-intervals of their parent and overlap it,
so a column does not sum to its parent.  `...cderi loop` is the time spent
inside `DF.loop` producing blocks -- the unpack, and on a host-resident cderi
the PCIe copy -- and is a sub-interval of `..with_df.get_jk`.
"""
import json
import time
from collections import OrderedDict

import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--max-cycle', type=int, default=50)
ap.add_argument('--auxbasis', default=None,
                help='auxiliary basis; default is the basis set\'s own '
                     'jkfit/ri set, which is what density_fit() picks')
ap.add_argument('--only-dfj', action='store_true',
                help='density-fit J only and take K from the direct path '
                     '(RIJONX).  **Broken in stock GPU4PySCF 1.8.1**: '
                     '`_DFHF.get_jk` passes `mol` where `DF.get_jk` expects '
                     'the density (`df/df_jk.py:213`), so the run dies with '
                     '"\'Mole\' object has no attribute \'shape\'".  '
                     '`fastdfj` repairs it, so this needs `--patch all` (or '
                     '`--patch fastdfj`) to run at all -- and `FASTDFJ_OFF=1` '
                     'puts the failure back, so the check is one command '
                     'rather than a claim.  `benchmarks/dfj_probe.py` is '
                     'what measures the configuration; this flag only says '
                     'where a RIJONX run spends its time.')
ap.add_argument('--cpu-cderi', action='store_true',
                help='force the cderi tensor onto host memory, which is the '
                     'regime the GPU4PySCF README calls "bounded by CPU '
                     'memory" -- it is the default only once it does not fit')
a = ap.parse_args()
apply_patch(a.patch)

import gpu4pyscf                                            # noqa: E402
from gpu4pyscf import dft                                   # noqa: E402
from gpu4pyscf.df import df as gdf, df_jk as gdf_jk         # noqa: E402
from gpu4pyscf.df import int3c2e_bdiv as gbdiv              # noqa: E402
from gpu4pyscf.dft import numint as gnumint                 # noqa: E402
from gpu4pyscf.grad import rhf as ggrhf                     # noqa: E402
from gpu4pyscf.grad import rks as ggrks, uks as gguks       # noqa: E402
from gpu4pyscf.lib import diis as gdiis                     # noqa: E402

T = OrderedDict()
N = OrderedDict()
_sync = cp.cuda.Stream.null.synchronize


def _wrap(obj, name, label):
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


def _wrap_gen(obj, name, label):
    """Time the *production* of each item of a generator, not its consumption.

    `DF.loop` yields one unpacked cderi block at a time; everything the caller
    does with a block happens between two `next()` calls and is therefore
    outside these intervals.  So this row is the unpack (and, when the cderi
    lives on the host, the PCIe copy) and nothing else.
    """
    orig = getattr(obj, name)
    T.setdefault(label, 0.0)
    N.setdefault(label, 0)

    def timed(*args, **kwargs):
        it = orig(*args, **kwargs)
        while True:
            _sync()
            t = time.perf_counter()
            try:
                item = next(it)
            except StopIteration:
                T[label] += time.perf_counter() - t
                return
            _sync()
            T[label] += time.perf_counter() - t
            N[label] += 1
            yield item
    timed.__name__ = getattr(orig, '__name__', name)
    setattr(obj, name, timed)


mol = build_mol(a)
aux = a.auxbasis
tag = (f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{aux or "auto"}/{a.xc}/'
       f'{a.patch or "stock"}{"/dfj" if a.only_dfj else ""}')

mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc)
mf = mf.density_fit(auxbasis=aux, only_dfj=a.only_dfj)
if a.cpu_cderi:
    mf.with_df.use_gpu_memory = False
mf.conv_tol = a.conv_tol
mf.grids.level = a.grids
mf.verbose = 0
mf.max_cycle = a.max_cycle

print(f'[{tag}] natm={mol.natm} nbas={mol.nbas} nao={mol.nao} '
      f'naux={mf.with_df.auxmol.nao if mf.with_df.auxmol else "?"} '
      f'spin={mol.spin} charge={mol.charge} '
      f'lmax={int(mol._bas[:, 1].max())} gpu4pyscf={gpu4pyscf.__version__}',
      flush=True)

# --- the per-cycle driver calls -------------------------------------------
_wrap(mf, 'get_fock', 'get_fock(+DIIS)')
_wrap(mf, 'eig', 'eig')
_wrap(mf, 'make_rdm1', 'make_rdm1')
_wrap(mf, 'get_veff', 'get_veff')
_wrap(mf, 'get_init_guess', 'get_init_guess')
_wrap(mf, 'energy_tot', 'energy_tot')
_wrap(mf, 'get_grad', 'get_grad')
_wrap(mf, 'get_hcore', 'get_hcore')
_wrap(mf, 'get_ovlp', 'get_ovlp')
_wrap(mf, 'get_occ', 'get_occ')

# --- inside get_veff -------------------------------------------------------
_wrap(mf, 'get_j', '..get_j')
_wrap(mf, 'get_k', '..get_k')
_wrap(mf, 'get_jk', '..get_jk')
_wrap(mf._numint, 'nr_rks' if not mol.spin else 'nr_uks',
      '..nr_rks' if not mol.spin else '..nr_uks')
_wrap(mf.grids, 'build', '..grids.build')
_wrap(mf._numint, 'nr_nlc_vxc', '..nr_nlc_vxc')

# --- the DF machinery itself ----------------------------------------------
# On the *class*, not on `mf.with_df`.  A range-separated functional makes
# `DF.range_coulomb` hand the long-range build a `self.copy()` of the
# density-fitting object, and a timer installed on the instance is copied with
# it -- still closed over the *original* object's bound method, so the copy's
# `build()` would fill the original's `_cderi` and the copy's would stay None.
# That is not a hypothetical: it is what made every wB97M-V run here die in
# `get_jk` with `'NoneType' object is not subscriptable`, patched and
# unpatched alike.
_wrap(gdf.DF, 'build', '..with_df.build')
_wrap(gdf.DF, 'get_jk', '..with_df.get_jk')
_wrap(gdf_jk, 'get_j', '..df_jk.get_j (J only)')
_wrap(gdf, '_cholesky_eri', '...cholesky_eri')
_wrap(gbdiv, 'int2c2e', '...int2c2e (j2c)')
_wrap(gdf, '_decompose_j2c', '...decompose j2c')
_wrap_gen(gdf.DF, 'loop', '...cderi loop (unpack)')
_wrap(gdiis.DIIS, 'update', '..diis.update')

_GRAD_KEYS = []
if not a.no_grad:
    from gpu4pyscf.df.grad import rhf as gdfgrad                # noqa: E402
    for _obj, _name, _label in (
            (ggrhf, '_hcore_energy', 'grad hcore+ovlp'),
            (gdfgrad, '_jk_energy_per_atom', 'grad df ejk'),
            (gguks if mol.spin else ggrks, 'get_exc', 'grad xc'),
            (ggrks, 'get_nlc_exc', 'grad nlc')):
        if not hasattr(_obj, _name):
            continue
        _wrap(_obj, _name, _label)
        _GRAD_KEYS.append(_label)

t = time.perf_counter()
e = mf.kernel()
_sync()
total = time.perf_counter() - t

grad_total = 0.0
de = None
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
    auxbasis=aux, xc=a.xc, patch=a.patch, only_dfj=a.only_dfj,
    cpu_cderi=a.cpu_cderi, natm=mol.natm, nao=int(mol.nao),
    naux=int(mf.with_df.auxmol.nao), spin=int(mol.spin),
    charge=int(mol.charge), e=float(e), conv=bool(mf.converged),
    cycles=int(mf.cycles), total=total, grad_total=grad_total,
    t={k: v for k, v in T.items() if N[k]},
    n={k: v for k, v in N.items() if v})), flush=True)
