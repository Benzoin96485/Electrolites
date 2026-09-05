"""How much slack is left in GPU4PySCF's per-block AO screen?

The XC build's cost is set by ``nao_sub``, the number of AOs a grid block
keeps: both the rho contraction and the potential contraction are
``nao_sub^2 * ngrids`` per block.  GPU4PySCF decides membership with
``GDFTscreen_index_legacy``, which keeps a shell when the **radial** sum
``sum_p c_p exp(-a_p r^2)`` exceeds ``AO_THRESHOLD = 1e-10`` at any point of
the block -- no ``r^l`` factor (the estimate that carries one is in that file,
commented out) and no reference to the density.

JoltQC screens on a log-magnitude estimate instead.  Before writing one, this
measures what such a screen could remove, from the AO values GPU4PySCF has
already evaluated:

  ``exact``    keep AO i where ``max_g |ao_i(g)| > 1e-10`` -- what the current
               screen is trying to approximate;
  ``density``  keep AO i where ``max_g|ao_i| * (|dm| @ max_g|ao|)_i > tol`` --
               the density-aware bound, i.e. drop an AO that cannot move rho
               by ``tol`` anywhere in the block.

The columns are the fraction of AOs kept and the implied ``nao_sub^2`` work.

    python benchmarks/aoscreen_probe.py --xyz molecules/PfPMT.xyz --charge -1
"""
import argparse
import json

import cupy as cp
import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--xyz', required=True)
ap.add_argument('--charge', type=int, default=0)
ap.add_argument('--spin', type=int, default=0)
ap.add_argument('--basis', default='6-31G*')
ap.add_argument('--xc', default='b3lyp')
ap.add_argument('--grids', type=int, default=3)
ap.add_argument('--tol', type=float, nargs='*', default=[1e-13, 1e-11, 1e-9])
ap.add_argument('--cycles', type=int, default=0,
                help='SCF cycles to run first; a converged density is less '
                     'sparse than the minao guess, so the screen removes less')
a = ap.parse_args()

from pyscf import gto                                       # noqa: E402
import gpu4pyscf                                            # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.dft import numint as NI                      # noqa: E402

mol = gto.M(atom=a.xyz, basis=a.basis, charge=a.charge, spin=a.spin,
            verbose=0, max_memory=80000)
mf = dft.RKS(mol, xc=a.xc)
mf.grids.level = a.grids
mf.verbose = 0
if a.cycles:
    mf.max_cycle = a.cycles
    mf.conv_tol = 1e-12
    mf.kernel()
    dm0 = cp.asarray(mf.make_rdm1())
else:
    dm0 = cp.asarray(mf.get_init_guess(mol, 'minao'))
mf.grids.build()
ni = mf._numint
ni.build(mol, mf.grids.coords)
opt = ni.gdftopt
smol = opt._sorted_mol
nao = smol.nao
dm = opt.sort_orbitals(cp.asarray(dm0), axis=[0, 1])
adm = cp.abs(dm)

print(f'[{a.xyz.rsplit("/", 1)[-1]}/{a.basis}] density='
      f'{"scf-" + str(a.cycles) if a.cycles else "minao"} nao={nao} '
      f'grids={mf.grids.coords.shape[0]} gpu4pyscf={gpu4pyscf.__version__}',
      flush=True)

tot = {k: 0.0 for k in ['now', 'exact'] + [f'dens{t:g}' for t in a.tol]}
sq = dict(tot)
nblk = 0
for ao_mask, idx, weight, _ in ni.block_loop(smol, mf.grids, nao, 1,
                                             max_memory=None):
    nblk += 1
    n0 = len(idx)
    amax = cp.abs(ao_mask[0]).max(axis=1)
    tot['now'] += n0
    sq['now'] += n0 * n0
    keep = int((amax > 1e-10).sum())
    tot['exact'] += keep
    sq['exact'] += keep * keep
    sub = adm[cp.ix_(idx, idx)]
    bound = amax * (sub @ amax)
    for t in a.tol:
        k = int((bound > t).sum())
        tot[f'dens{t:g}'] += k
        sq[f'dens{t:g}'] += k * k

print(f'{nblk} blocks')
base_n, base_sq = tot['now'], sq['now']
print(f'{"screen":>12} {"mean nao_sub":>13} {"kept":>7} {"sum nao_sub^2":>15} '
      f'{"work":>7}')
for k in tot:
    print(f'{k:>12} {tot[k]/nblk:13.1f} {tot[k]/base_n:7.3f} '
          f'{sq[k]:15.4g} {sq[k]/base_sq:7.3f}')
print('RESULT ' + json.dumps(dict(
    xyz=a.xyz, basis=a.basis, nao=int(nao), nblocks=nblk,
    mean_nao_sub={k: tot[k]/nblk for k in tot},
    work_ratio={k: sq[k]/base_sq for k in sq})), flush=True)
