"""W36: can a hand-written fp64 tensor-core GEMM carry a fused epilogue?

`docs/README_fastxcfxclocal.md` §3.  Fusing W33's density kernel into its
``Y = M^T ao`` GEMM (or generating ``W`` inside the potential GEMM's operand
load) removes ``fxc_rho_occ_*`` and ``fxc_w_occ_*`` -- 10 % of a 113-atom
call -- but only if the GEMM that carries the fusion runs within that 10 %
of cuBLAS.  This times ``fxc_dmma_probe.cu`` (no epilogue: the ceiling of
any fused version) against ``cupy.dot`` on the same operands, with ``lda``
padded even as a fused kernel would allocate it.

    python benchmarks/fxc_dmma_probe.py --shape 76614,4096,401 \\
        --cfgs 64,128,32,64,16,3 128,64,64,32,16,2

A config is ``BM,BN,WM,WN,BK,NST``.  The taxol shape is ``nset*nocc =
339*226`` rows, a 4096-point grid block and ``nsub`` = 401.
"""
import argparse
import os
import time

import cupy as cp
import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--shape', default='76614,4096,401', help='M,N,K')
ap.add_argument('--cfgs', nargs='+',
                default=['64,128,32,64,16,3', '64,128,32,64,16,2',
                         '128,64,64,32,16,2', '128,128,32,64,16,2'])
ap.add_argument('--reps', type=int, default=5)
a = ap.parse_args()
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'fxc_dmma_probe.cu')).read()
M, N, K = [int(x) for x in a.shape.split(',')]
lda = K + (K & 1)
Ap = cp.random.standard_normal((M, lda))
A = Ap[:, :K]
Ac = cp.ascontiguousarray(A)
B = cp.random.standard_normal((K, N))
ref = Ac @ B


def bench(f):
    f()
    cp.cuda.Stream.null.synchronize()
    t = time.perf_counter()
    for _ in range(a.reps):
        f()
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t) / a.reps


fl = 2 * M * N * K
print(f'# {cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}  '
      f'M,N,K = {M},{N},{K}')
t0 = bench(lambda: cp.dot(Ac, B, out=ref))
print(f'cuBLAS           {t0 * 1e3:8.2f} ms  {fl / t0 / 1e12:5.2f} TF/s')
for cfg in a.cfgs:
    bm, bn, wm, wn, bk, nst = [int(x) for x in cfg.split(',')]
    mod = cp.RawModule(code=src, options=(
        '-std=c++17', f'-DBM={bm}', f'-DBN={bn}', f'-DWM={wm}', f'-DWN={wn}',
        f'-DBK={bk}', f'-DNST={nst}'))
    k = mod.get_function('dmma_nn')
    nt = (bm // wm) * (bn // wn) * 32
    shm = 8 * nst * (bm * (bk + 4) + bk * (bn + 4))
    try:
        k.max_dynamic_shared_size_bytes = shm
    except Exception as e:                                   # noqa: BLE001
        print(f'{cfg:16s} does not fit ({shm} B shared): {e}')
        continue
    C = cp.empty((M, N))
    grid = ((N + bn - 1) // bn, (M + bm - 1) // bm)
    t = bench(lambda: k(grid, (nt,), (
        C, Ap, B, np.int32(M), np.int32(N), np.int32(K), np.int32(lda),
        np.int32(N), np.int32(N), np.float64(0)), shared_mem=shm))
    err = float(abs(C - ref).max() / abs(ref).max())
    print(f'{cfg:16s} {t * 1e3:8.2f} ms  {fl / t / 1e12:5.2f} TF/s  '
          f'{t0 / t:5.1%} of cuBLAS  err {err:.1e}  regs {k.num_regs}')
