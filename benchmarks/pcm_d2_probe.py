"""Do the PCM Hessian's dense second-derivative tensors fit on the card?

`docs/KERNEL_TARGETS_1.8.1.md` F10 / Q3 says, from shapes alone, that
`get_d2F_d2A` allocates two `(ngrids, natm, natm, 3, 3)` doubles and
`get_d2D_d2S` two `(3, 3, ngrids, ngrids)`-class arrays, and that this is
10 GB each at taxol and an OOM at 168 atoms on an 80 GB card.  This builds the
cavity of the given molecule and calls the two functions alone, timing them
and reading the memory pool's high-water mark, so the claim is a measurement
rather than arithmetic.

    python benchmarks/pcm_d2_probe.py --xyz benchmarks/molecules/taxol.xyz
"""
import argparse
import time

import cupy as cp
from pyscf import gto

ap = argparse.ArgumentParser()
ap.add_argument('--xyz', required=True)
ap.add_argument('--charge', type=int, default=0)
ap.add_argument('--method', default='IEF-PCM')
a = ap.parse_args()

from gpu4pyscf.solvent import pcm                            # noqa: E402
from gpu4pyscf.solvent.hessian import pcm as hpcm            # noqa: E402

mol = gto.M(atom=a.xyz, basis='def2-svp', charge=a.charge, verbose=0)
obj = pcm.PCM(mol)
obj.method = a.method
obj.lebedev_order = 29
obj.eps = 78.3553
obj.build()
s = obj.surface
ng = s['grid_coords'].shape[0]
pool = cp.get_default_memory_pool()
free, total = cp.cuda.runtime.memGetInfo()
print(f'natm={mol.natm} ngrids={ng} card={total/1e9:.1f} GB free={free/1e9:.1f} GB')
print(f'  by shape: d2F + d2A = {2*ng*mol.natm**2*9*8/1e9:.2f} GB, '
      f'one (3,3,ng,ng) = {9*ng*ng*8/1e9:.2f} GB')
for name, fn in (('get_d2F_d2A', lambda: hpcm.get_d2F_d2A(s)),
                 ('get_d2D_d2S', lambda: hpcm.get_d2D_d2S(s, with_S=True, with_D=True))):
    pool.free_all_blocks()
    cp.cuda.Device().synchronize()
    # GPU4PySCF installs its own allocator (`lib/cupy_helper.py` malloc), so
    # the CuPy pool's counters do not see these arrays: read the device
    base = cp.cuda.runtime.memGetInfo()[0]
    t = time.perf_counter()
    try:
        out = fn()
        cp.cuda.Device().synchronize()
        dt = time.perf_counter() - t
        held = base - cp.cuda.runtime.memGetInfo()[0]
        print(f'{name}: {dt:.2f} s, device memory held by the result '
              f'{held/1e9:.2f} GB')
        del out
    except Exception as err:                    # noqa: BLE001
        print(f'{name}: FAILED after {time.perf_counter()-t:.1f} s: '
              f'{type(err).__name__}: {str(err)[:120]}')
