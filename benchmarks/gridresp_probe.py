"""Where does the grid-response Hessian's int32 assert fire? (B9, Q4)

`hessian/rks.py:2009-2010` materialises `d2weight_dAdB` as
`(natm, natm, 3, 3, ngrids_block)` and asserts its size is below 2**31.  The
catalogue computed that this fails above ~240 atoms for a 4096-point block.
This builds the molecule's grids, reads the block sizes `ni.block_loop`
actually hands the grid-response code, and calls `get_d2weight_dAdB` on the
largest one, so the claim is a run rather than arithmetic.

    python benchmarks/gridresp_probe.py --xyz benchmarks/molecules/PfPMT.xyz --charge -1
"""
import argparse
import time

import cupy as cp
import numpy as np
from pyscf import gto

ap = argparse.ArgumentParser()
ap.add_argument('--xyz', required=True)
ap.add_argument('--charge', type=int, default=0)
ap.add_argument('--basis', default='def2-svp')
a = ap.parse_args()

from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.hessian import rks as hrks                    # noqa: E402

mol = gto.M(atom=a.xyz, basis=a.basis, charge=a.charge, verbose=0)
mf = dft.RKS(mol, xc='b3lyp')
mf.grids.level = 3
mf.grids.build()
ni = mf._numint
ni.build(mol, mf.grids.coords)
smol = ni.gdftopt._sorted_mol
sizes = []
for ao, idx, weight, _ in ni.block_loop(smol, mf.grids, mol.nao, deriv=0,
                                        strict_grid_order=True):
    sizes.append(weight.shape[0])
sizes = np.array(sizes)
lim = np.iinfo(np.int32).max
need = mol.natm ** 2 * 9 * sizes
print(f'natm={mol.natm} ngrids={mf.grids.coords.shape[0]} blocks={len(sizes)} '
      f'block size min/median/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}')
print(f'natm^2*9*block: max {need.max():.3e} against int32 max {lim:.3e}; '
      f'{(need >= lim).sum()} of {len(sizes)} blocks exceed it; '
      f'largest d2w block = {need.max()*8/1e9:.1f} GB')
g1 = int(sizes.max())
cp.cuda.Device().synchronize()
t = time.perf_counter()
try:
    d2w = hrks.get_d2weight_dAdB(smol, mf.grids, (0, g1))
    cp.cuda.Device().synchronize()
    print(f'get_d2weight_dAdB on a {g1}-point block: {time.perf_counter()-t:.2f} s, '
          f'{d2w.nbytes/1e9:.2f} GB')
except AssertionError:
    print(f'get_d2weight_dAdB on a {g1}-point block: AssertionError (the int32 guard)')
except Exception as err:                                      # noqa: BLE001
    print(f'get_d2weight_dAdB on a {g1}-point block: {type(err).__name__}: '
          f'{str(err)[:100]}')
