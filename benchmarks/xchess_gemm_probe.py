"""`fastxchess._deriv1_gemms` stage by stage, and the GEMM rate (W28).

Replays `_deriv1_gemms` with a synchronisation after every step on one
converged SCF, and prints the fp64 rate of the two GEMMs and the write rate
of `aow` -- the numbers behind docs/README_fastxchessmgga.md §3.

    python benchmarks/xchess_gemm_probe.py --xyz benchmarks/molecules/penicillinV.xyz \
        --basis def2-svp --xc tpss
"""
import sys, time
from _common import build_mol, parser
ap = parser('x'); a = ap.parse_args()
import cupy as cp, numpy as np
import electrolites
electrolites.patch('fastxchess')
from electrolites import fastxchess as fx
from gpu4pyscf import dft
from gpu4pyscf.hessian import rks as hrks
from gpu4pyscf.lib.cupy_helper import contract
mol = build_mol(a)
mf = dft.RKS(mol, xc=a.xc); mf.grids.level = a.grids; mf.verbose = 0; mf.kernel()
h = mf.Hessian()
lay, grids, ni, nc = fx._setup(h, mf.mo_occ)
blocks = lay.block_atoms(grids)
nocc = int((mf.mo_occ > 0).sum())
ng = 4096
fl = 0.
for atoms, rows in blocks:
    fl += 3*len(atoms)*len(rows)*nocc*ng*2*(1 + (4 if nc == 5 else 1))
print(f'# GEMM flops in _deriv1_gemms {fl/1e12:.2f} Tflop  (blocks {len(blocks)}, nc {nc})')
T = {}
S = cp.cuda.Stream.null.synchronize
orig = fx._deriv1_gemms
def sub(ao, mo, w, nsub, nocc, nb, ng, bufs, nc):
    def st(k, t0):
        S(); T[k] = T.get(k, 0) + time.perf_counter() - t0; return time.perf_counter()
    S(); t = time.perf_counter()
    aow = bufs('aow', (nb, nsub, ng))
    fx._scale_sum(ao, w, aow); t = st('aow build', t)
    t1 = bufs('t1', (nb, nsub, nocc))
    cp.dot(aow.reshape(-1, ng), mo[0].T, out=t1.reshape(-1, nocc)); t = st('gemm1', t)
    t2 = bufs('t2', (nb, nocc, nsub))
    mow = bufs('mow', (nb, nocc, 4, ng))
    fx._scale_sum(mo, w, mow, tau_fac=2.); t = st('mow build (+tau)', t)
    aoT = bufs('aoT', (nsub, 4, ng)); aoT[:] = ao[:4].transpose(1, 0, 2); t = st('aoT', t)
    cp.dot(mow.reshape(-1, 4*ng), aoT.reshape(nsub, 4*ng).T, out=t2.reshape(-1, nsub)); t = st('gemm2 (K=4ng)', t)
    t1 += t2.transpose(0, 2, 1); t = st('transpose add', t)
    return t1
fx._deriv1_gemms = sub
S(); t0 = time.perf_counter(); hrks._get_vxc_deriv1(h, mf.mo_coeff, mf.mo_occ, 4000); S()
tot = time.perf_counter() - t0
print(f'# deriv1 total {tot:.3f} s')
for k, v in T.items(): print(f'  {k:18s} {v:7.3f} s')
g = T['gemm1'] + T['gemm2 (K=4ng)']
print(f'# GEMMs {g:.3f} s = {fl/g/1e12:.1f} TF/s')
el = fl / (2*nocc*(1 + (4 if nc == 5 else 1)))   # elements of aow
print(f'# aow: {el*8/1e9:.1f} GB written, {el*8/T["aow build"]/1e9:.0f} GB/s')
