"""How far apart the int3c1e tasks are, and what screening them would be worth.

    python benchmarks/int3c1e_farfield_probe.py --xyz m.xyz --basis def2-svp

Two questions about the (primitive pair, cavity point) task space that decide
what `fast1e` should do, both answered from the pair data GPU4PySCF has already
built rather than from a model.

1. **Where is the Boys argument?**  ``x = a0*r2`` with ``a0 = p*q/(p+q)``.  Above
   ``x = 40`` ``erf(sqrt(x))`` and ``exp(-x)`` are constant to double precision
   and the charge-smearing algebra cancels -- that is the branch `fast1e` adds.
   The share of tasks above the threshold is the size of the prize.

2. **What would screening be worth instead?**  In the far field a task's value
   is exactly ``2*pi*sqrt(pi/4)*|eij|*p^-1.5 / r`` for the s-s component, which
   is a cheap rigorous bound on it.  Dropping a set of tasks moves the potential
   by at most the sum of their bounds, so "how many tasks fit inside an absolute
   error budget" is a measurement, not a guess.  It comes out at a fifth to a
   quarter -- and they come off the branch that is now nearly free, which is why
   `fast1e` does not screen.  See "What screening is worth" in
   docs/README_fast1e.md.

This reads the primitive-pair arrays out of GPU4PySCF's BasisProdCache through
the same ctypes mirror `fast1e` uses, so it reports the numbers the kernels
actually see.
"""
import argparse
import ctypes
import sys

import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import dft

sys.path.insert(0, __file__.rsplit('/', 2)[0])
from electrolites.fast1e import _BasisProdCache            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--xyz', required=True)
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--lebedev', type=int, default=29)
ap.add_argument('--chunk', type=int, default=200,
                help='cavity points per pass; the task space is ngrids x nprim')
a = ap.parse_args()

mol = pyscf.M(atom=a.xyz, basis=a.basis, verbose=0, output='/dev/null')
mf = dft.RKS(mol, xc='b3lyp').PCM()
mf.verbose = 0
mf.with_solvent.method = 'IEF-PCM'
mf.with_solvent.lebedev_order = a.lebedev
pcm = mf.with_solvent
pcm.build()
grids = pcm.surface['grid_coords']
q = pcm.surface['charge_exp'] ** 2
intopt = pcm.intopt

c = ctypes.cast(intopt.bpcache, ctypes.POINTER(_BasisProdCache)).contents
ncp = c.ncptype
ppl = np.ctypeslib.as_array(
    ctypes.cast(c.primitive_pairs_locs, ctypes.POINTER(ctypes.c_int)),
    (ncp + 1,)).copy()
nprim = int(ppl[-1])


def dev(ptr, n):
    mem = cp.cuda.UnownedMemory(ptr, n * 8, None)
    return cp.ndarray((n,), dtype=cp.float64,
                      memptr=cp.cuda.MemoryPointer(mem, 0))


a12, e12 = dev(c.a12, nprim), dev(c.e12, nprim)
x12, y12, z12 = dev(c.x12, nprim), dev(c.y12, nprim), dev(c.z12, nprim)

# which primitive pairs belong to a one-root (i_l + j_l <= 1) angular type
lsum = np.zeros(nprim, dtype=np.int32)
for k in range(ncp):
    li = intopt.angular[intopt.cp_idx[k]]
    lj = intopt.angular[intopt.cp_jdx[k]]
    lsum[ppl[k]:ppl[k + 1]] = li + lj
one_root = cp.asarray(lsum <= 1)

# the far-field amplitude, C/r, with C independent of the point
C = (2.0 * np.pi * 0.8862269254527580136) * cp.abs(e12) * a12 ** -1.5

X_TH = [20.0, 40.0, 60.0]
V_TH = [1e-16, 1e-14, 1e-12]
ntot = nprim * grids.shape[0]
n1 = int(cp.count_nonzero(one_root)) * grids.shape[0]
cx = np.zeros(len(X_TH))
cx1 = np.zeros(len(X_TH))
cv = np.zeros(len(V_TH))
sv = np.zeros(len(V_TH))
total = 0.0

for p0 in range(0, grids.shape[0], a.chunk):
    g = grids[p0:p0 + a.chunk]
    qq = q[p0:p0 + a.chunk]
    dx = x12[:, None] - g[None, :, 0]
    dy = y12[:, None] - g[None, :, 1]
    dz = z12[:, None] - g[None, :, 2]
    r2 = dx * dx + dy * dy + dz * dz
    a0 = a12[:, None] * qq[None, :] / (a12[:, None] + qq[None, :])
    x = a0 * r2
    for k, t in enumerate(X_TH):
        m = x > t
        cx[k] += int(cp.count_nonzero(m))
        cx1[k] += int(cp.count_nonzero(m & one_root[:, None]))
    v = C[:, None] / cp.sqrt(r2)
    total += float(v.sum())
    for k, t in enumerate(V_TH):
        m = v < t
        n = int(cp.count_nonzero(m))
        cv[k] += n
        if n:
            sv[k] += float(v[m].sum())

print(f'# {a.xyz.rsplit("/", 1)[-1]} natm={mol.natm} nao={mol.nao} '
      f'basis={a.basis} lebedev={a.lebedev} cavity={grids.shape[0]} '
      f'primitive_pairs={nprim}')
print(f'# tasks = {ntot:.3e}; {100 * n1 / ntot:.1f} % of them are one-root '
      f'(the (ss| and (ps| kernels)')
print('# 1. Boys argument x = a0*r2')
for k, t in enumerate(X_TH):
    print(f'    x > {t:5.1f} : {100 * cx[k] / ntot:6.2f} % of all tasks, '
          f'{100 * cx1[k] / max(n1, 1):6.2f} % of the one-root tasks')
print(f'# 2. far-field amplitude, sum over all tasks = {total:.6e}')
for k, t in enumerate(V_TH):
    print(f'    drop |v| < {t:.0e} : {100 * cv[k] / ntot:6.2f} % of tasks, '
          f'total dropped magnitude {sv[k]:.3e}')
