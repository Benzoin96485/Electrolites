"""Where does a k-point hybrid DFT wall clock go, on GPU4PySCF 1.8.1?

`docs/KERNEL_TARGETS_1.8.1.md` J1-J8 are source readings of the periodic code
with no profile under them.  This times one k-point PBE0 SCF with each of the
two density-fitting back ends -- GDF (`density_fit()`, J1/J2) and FFTDF (the
`KRKS` default, J3) -- and every stage those entries name, synchronised on
both sides, the same way `path_anatomy.py` does for the molecular features.

    python benchmarks/kpt_anatomy.py --cell diamond --kpts 2,2,2 --df gdf \
        [--patch all]

`--cell diamond` is the 2-atom primitive cell, `--cell diamond8` the 8-atom
cubic one, so that a pair of sizes is one flag apart.  Electrolites patches
no periodic entry point; `--patch all` is there to show that it is inert.
"""
import argparse
import json
import time
from collections import OrderedDict
import importlib

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--cell', default='diamond', choices=('diamond', 'diamond8'))
ap.add_argument('--kpts', default='2,2,2')
ap.add_argument('--df', default='gdf', choices=('gdf', 'fftdf'))
ap.add_argument('--xc', default='pbe0')
ap.add_argument('--basis', default='gth-dzvp')
ap.add_argument('--conv-tol', type=float, default=1e-8)
ap.add_argument('--patch', default='none')
ap.add_argument('--grad', action='store_true',
                help='nuclear gradient after the SCF (J7: the per-atom hcore)')
ap.add_argument('--stress', action='store_true',
                help='stress tensor after the SCF (J4 with --multigrid, J5)')
ap.add_argument('--multigrid', action='store_true',
                help='multigrid_v2 numerical integration (pure functionals)')
ap.add_argument('--cprofile', default='')
a = ap.parse_args()

import electrolites                                         # noqa: E402
if a.patch == 'all':
    electrolites.patch_all()
import cupy as cp                                           # noqa: E402
import numpy as np                                          # noqa: E402
import gpu4pyscf                                            # noqa: E402
from pyscf.pbc import gto                                   # noqa: E402

_sync = cp.cuda.Stream.null.synchronize
T = OrderedDict()
N = OrderedDict()


def wrap(path, label):
    mod, attr = path.split(':')
    parts = attr.split('.')
    name = parts[-1]
    try:
        holder = importlib.import_module(mod)
        for p in parts[:-1]:
            holder = getattr(holder, p)
        orig = getattr(holder, name)
    except (AttributeError, ImportError) as err:
        print(f'# cannot wrap {path}: {err}')
        return
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
    setattr(holder, name, timed)


PLAN = [
    ('gpu4pyscf.pbc.scf.khf:KSCF.get_hcore', '..get_hcore (J5/J6 pp)'),
    ('gpu4pyscf.pbc.df.df:GDF.build', '..GDF.build (cderi)'),
    ('gpu4pyscf.pbc.df.rsdf_builder:_precontract_j2c_aux_coeff',
     '....j2c metric per k (J2)'),
    ('gpu4pyscf.pbc.df.rsdf_builder:decompose_j2c', '......decompose_j2c (J2)'),
    ('gpu4pyscf.pbc.df.df_jk:get_j_kpts', '..GDF get_j_kpts (J1)'),
    ('gpu4pyscf.pbc.df.df_jk:get_k_kpts', '..GDF get_k_kpts (J1)'),
    ('gpu4pyscf.pbc.df.fft_jk:get_j_kpts', '..FFTDF get_j_kpts'),
    ('gpu4pyscf.pbc.df.fft_jk:get_k_kpts', '..FFTDF get_k_kpts (J3)'),
    ('gpu4pyscf.pbc.dft.numint:KNumInt.nr_rks', '..KNumInt.nr_rks (XC)'),
    ('gpu4pyscf.pbc.scf.khf:KSCF.eig', '..eig (J8)'),
]
PLAN += [
    ('gpu4pyscf.pbc.grad.krhf:grad_elec', 'gradient grad_elec'),
    ('gpu4pyscf.pbc.grad.krks:Gradients.get_veff', '..grad get_veff'),
    ('gpu4pyscf.pbc.grad.krhf:get_hcore', '..grad get_hcore'),
    ('gpu4pyscf.pbc.grad.krks_stress:kernel', 'stress kernel'),
    ('gpu4pyscf.pbc.grad.krks_stress:get_vxc', '..stress get_vxc'),
    ('gpu4pyscf.pbc.grad.krks_stress:_rks_exc_strain_deriv',
     '....multigrid strain FD (J4)'),
    ('gpu4pyscf.pbc.dft.multigrid_v2:evaluate_density_on_g_mesh',
     '......evaluate_density_on_g_mesh'),
    ('gpu4pyscf.pbc.grad.krks_stress:_get_pp_nonloc_strain_derivatives',
     '..pp non-local strain (J5)'),
    ('gpu4pyscf.pbc.grad.krks_stress:ewald', '..ewald strain'),
]
for path, label in PLAN:
    wrap(path, label)

# J7: `grad_elec` asks `hcore_generator` for a closure and calls it once per
# atom, so the closure is what has to be timed
import gpu4pyscf.pbc.grad.krhf as _pkrhf                    # noqa: E402
_gen = _pkrhf.Gradients.hcore_generator
T.setdefault('..hcore_deriv per atom (J7)', 0.0)
N.setdefault('..hcore_deriv per atom (J7)', 0)


def _timed_gen(self, *args, **kwargs):
    fn = _gen(self, *args, **kwargs)

    def per_atom(*a2, **k2):
        _sync()
        t = time.perf_counter()
        out = fn(*a2, **k2)
        _sync()
        T['..hcore_deriv per atom (J7)'] += time.perf_counter() - t
        N['..hcore_deriv per atom (J7)'] += 1
        return out
    return per_atom


_pkrhf.Gradients.hcore_generator = _timed_gen

if a.cell == 'diamond':
    atom = 'C 0 0 0; C 0.8917 0.8917 0.8917'
    lat = [[0, 1.7834, 1.7834], [1.7834, 0, 1.7834], [1.7834, 1.7834, 0]]
else:
    x = 3.5668
    frac = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0),
            (.25, .25, .25), (.25, .75, .75), (.75, .25, .75), (.75, .75, .25)]
    atom = '; '.join(f'C {x*f[0]} {x*f[1]} {x*f[2]}' for f in frac)
    lat = np.eye(3) * x
cell = gto.M(atom=atom, a=lat, basis=a.basis, pseudo='gth-pade', verbose=0,
             max_memory=4000)
kmesh = [int(k) for k in a.kpts.split(',')]
kpts = cell.make_kpts(kmesh)

from gpu4pyscf.pbc import dft as pdft                       # noqa: E402
mf = pdft.KRKS(cell, kpts=kpts, xc=a.xc)
if a.df == 'gdf':
    mf = mf.density_fit()
if a.multigrid:
    mf = mf.multigrid_numint()
mf.conv_tol = a.conv_tol
mf.verbose = 0

tag = (f'kpt/{a.cell}/{a.kpts}/{a.df}{"+mg" if a.multigrid else ""}/'
       f'{a.basis}/{a.xc}/patch={a.patch}')
print(f'[{tag}] natm={cell.natm} nao={cell.nao} nk={len(kpts)} '
      f'gpu={cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}',
      flush=True)
if a.cprofile:
    import cProfile
    import pstats
    prof = cProfile.Profile()
    prof.enable()
_sync()
t0 = time.perf_counter()
e = mf.kernel()
_sync()
wall = time.perf_counter() - t0
if a.grad:
    g = mf.nuc_grad_method()
    g.verbose = 0
    _sync()
    t = time.perf_counter()
    g.kernel()
    _sync()
    T['gradient total'] = time.perf_counter() - t
    N['gradient total'] = 1
if a.stress:
    g = mf.nuc_grad_method()
    g.verbose = 0
    _sync()
    t = time.perf_counter()
    g.get_stress()
    _sync()
    T['stress total'] = time.perf_counter() - t
    N['stress total'] = 1
if a.cprofile:
    prof.disable()
    prof.dump_stats(a.cprofile)
    print('\n# cProfile, by self time')
    pstats.Stats(prof).sort_stats('tottime').print_stats(30)
print(f'\n[{tag}] SCF E={e:.10f} conv={mf.converged} wall {wall:.2f} s')
print(f'{"stage":40s} {"s":>9s} {"%wall":>7s} {"calls":>6s}')
for k, v in T.items():
    if N[k]:
        print(f'{k:40s} {v:9.2f} {100*v/wall:6.1f}% {N[k]:6d}')
print('RESULT ' + json.dumps(dict(
    tag=tag, cell=a.cell, kpts=a.kpts, df=a.df, natm=cell.natm,
    nao=int(cell.nao), nk=len(kpts), e=float(e), wall=wall,
    stages={k: [T[k], N[k]] for k in T if N[k]})), flush=True)
