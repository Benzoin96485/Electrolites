"""Catalogue Q4: at large `n_dm`, is the exchange contraction bound by the
*layout* of the density?

`benchmarks/ndm_k.py` fits the multi-density exchange build as
``T = R + n_dm*C`` and finds ``C`` identical between GPU4PySCF's kernels and
this repository's (1.01x): the whole of `fastk`'s advantage is in ``R``, the
per-quartet integral work, and at the ``n_dm = 126`` an analytic Hessian's CPHF
asks for, ``R`` is a sixth of the build.  So anything further has to come out of
``C``.

Catalogue **Q4** guesses where ``C`` goes -- "at large n_dm the bottleneck moves
to uncoalesced density reads and atomicAdd contention" -- and marks it `[?]`,
inferred from code shape and never confirmed.  The reads are indeed strided:
both codes address the densities as ``dm[i_dm*nao*nao + a*nao + b]``, so the
``n_dm`` values a quartet needs of one density-matrix element are ``nao*nao``
doubles apart, one cache line each.  In the transposed layout
``dm[(a*nao+b)*n_dm + i_dm]`` they are contiguous, and one 128-byte line serves
sixteen densities.

This measures the difference on the access pattern alone, with no integrals in
the way: one kernel, one thread per shell quartet, four ``nf x nf`` blocks read
per density and four ``atomicAdd`` per density, exactly the shape of
``contract_compact``'s four exchange terms.  What it reports is a ceiling for
transposing the density -- it charges nothing for the transpose itself, which
is an ``n_dm * nao^2`` copy (0.19 GB at penicillin V's size).

**The answer is no, and by a wide margin**: on an A100-PCIE-40GB at nao = 430,
the transposed layout is **0.30x-0.38x** at ``n_dm`` >= 32 for both a p and a
d shell, i.e. two and a half to three times *slower*.  The reasoning behind
Q4's hypothesis leaves out the reuse that matters.  A quartet does read the
``n_dm`` copies of one matrix element from ``n_dm`` different cache lines --
that part is true -- but in exchange each density's ``nao^2`` block stays
compact (1.5 MB at this size, so an L2 resident) and is reused by every one of
the million quartets, and each thread's ``nf`` consecutive elements of a row
stay contiguous.  Transposing trades a stride *between* densities for a stride
*between quartets*, and there are four orders of magnitude more quartets than
densities.  So whatever ``C`` is bound by, it is not this, and the remaining
candidate from Q4's pair is the ``atomicAdd`` traffic -- 4*nf^2 atomics per
quartet per density, 144 of them for a d shell.

    python benchmarks/ndm_layout_probe.py --nao 430 --ndm 1,8,32,126 --nf 3,6
"""
import argparse
import json
import time

import cupy as cp
import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--nao', type=int, default=430,
                help='AO count; 430 is penicillin V/def2-SVP, 1123 taxol')
ap.add_argument('--ndm', default='1,8,32,126',
                help='comma separated density counts')
ap.add_argument('--nf', default='3,6',
                help='cartesian functions per shell (3 = p, 6 = d, 10 = f)')
ap.add_argument('--tasks', type=int, default=1 << 20,
                help='shell quartets per launch')
ap.add_argument('--reps', type=int, default=5)
a = ap.parse_args()

SRC = r'''
// One thread per shell quartet, four nf x nf blocks of the density read and
// four atomicAdds issued per density -- the shape of contract_compact's four
// exchange terms, with the integrals replaced by a formula.  They live in
// registers in the real kernel and cost no memory traffic; holding them in a
// local array here would spill NF^4 doubles and measure the spill instead.
__device__ __forceinline__ double gout_of(int n) { return 1.0 + n * 1e-9; }

extern "C" __global__ void __launch_bounds__(256, 4)
contract(double *vk_all, const double *dm_all, const int *offs,
         int ntasks, int n_dm, int nao, int dmajor)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ntasks) return;
    int i0 = offs[4*t+0], j0 = offs[4*t+1];
    int k0 = offs[4*t+2], l0 = offs[4*t+3];

    size_t nao2 = (size_t)nao * nao;
    for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
        const double *dm;
        double *vk;
        long st;
        if (dmajor) {          // dm[(a*nao+b)*n_dm + i_dm]
            dm = dm_all + i_dm;
            vk = vk_all + i_dm;
            st = n_dm;
        } else {               // dm[i_dm*nao*nao + a*nao + b]
            dm = dm_all + i_dm * nao2;
            vk = vk_all + i_dm * nao2;
            st = 1;
        }
#define G(ci,dj,dk,dl) gout_of((((dl)*NF+(dk))*NF+(dj))*NF+(ci))
        // vk[i,l] += sum_jk gout * dm[j,k]
        for (int ci = 0; ci < NF; ++ci)
            for (int dl = 0; dl < NF; ++dl) {
                double v = 0.;
                for (int dj = 0; dj < NF; ++dj)
                    for (int dk = 0; dk < NF; ++dk)
                        v += G(ci,dj,dk,dl)
                           * dm[((size_t)(j0+dj)*nao + (k0+dk))*st];
                atomicAdd(vk + ((size_t)(i0+ci)*nao + (l0+dl))*st, v);
            }
        // vk[i,k] += sum_jl gout * dm[j,l]
        for (int ci = 0; ci < NF; ++ci)
            for (int dk = 0; dk < NF; ++dk) {
                double v = 0.;
                for (int dj = 0; dj < NF; ++dj)
                    for (int dl = 0; dl < NF; ++dl)
                        v += G(ci,dj,dk,dl)
                           * dm[((size_t)(j0+dj)*nao + (l0+dl))*st];
                atomicAdd(vk + ((size_t)(i0+ci)*nao + (k0+dk))*st, v);
            }
        // vk[j,l] += sum_ik gout * dm[i,k]
        for (int dj = 0; dj < NF; ++dj)
            for (int dl = 0; dl < NF; ++dl) {
                double v = 0.;
                for (int ci = 0; ci < NF; ++ci)
                    for (int dk = 0; dk < NF; ++dk)
                        v += G(ci,dj,dk,dl)
                           * dm[((size_t)(i0+ci)*nao + (k0+dk))*st];
                atomicAdd(vk + ((size_t)(j0+dj)*nao + (l0+dl))*st, v);
            }
        // vk[j,k] += sum_il gout * dm[i,l]
        for (int dj = 0; dj < NF; ++dj)
            for (int dk = 0; dk < NF; ++dk) {
                double v = 0.;
                for (int ci = 0; ci < NF; ++ci)
                    for (int dl = 0; dl < NF; ++dl)
                        v += G(ci,dj,dk,dl)
                           * dm[((size_t)(i0+ci)*nao + (l0+dl))*st];
                atomicAdd(vk + ((size_t)(j0+dj)*nao + (k0+dk))*st, v);
            }
#undef G
    }
}
'''

print(f'# {cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}  '
      f'nao={a.nao} tasks={a.tasks}')
rng = np.random.default_rng(0)
rows = []
for nf in [int(x) for x in a.nf.split(',')]:
    mod = cp.RawModule(code=f'#define NF {nf}\n' + SRC,
                       options=('-std=c++17',), backend='nvrtc')
    kern = mod.get_function('contract')
    print(f'\nnf = {nf}  ({nf**4} integrals per quartet, '
          f'{4*nf*nf} atomicAdds per density)')
    print(f'{"n_dm":>5s} {"dm-minor/s":>11s} {"dm-major/s":>11s} {"ratio":>7s} '
          f'{"GB read (minor)":>16s}')
    print('-' * 58)
    for n_dm in [int(x) for x in a.ndm.split(',')]:
        need = n_dm * a.nao * a.nao * 8 * 2 / 2**30
        if need > 18:
            print(f'{n_dm:5d}  skipped: {need:.1f} GiB of density and vk')
            continue
        offs = cp.asarray(rng.integers(0, a.nao - nf, size=(a.tasks, 4)),
                          dtype=cp.int32).ravel()
        dm = cp.asarray(rng.standard_normal(n_dm * a.nao * a.nao))
        vk = cp.zeros_like(dm)
        ts = {}
        for dmajor in (0, 1):
            args = (vk, dm, offs, np.int32(a.tasks), np.int32(n_dm),
                    np.int32(a.nao), np.int32(dmajor))
            blocks = (a.tasks + 255) // 256
            kern((blocks,), (256,), args)
            cp.cuda.Stream.null.synchronize()
            t = []
            for _ in range(a.reps):
                cp.cuda.Stream.null.synchronize()
                t0 = time.perf_counter()
                kern((blocks,), (256,), args)
                cp.cuda.Stream.null.synchronize()
                t.append(time.perf_counter() - t0)
            ts[dmajor] = min(t)
        gb = a.tasks * n_dm * 4 * nf * nf * 8 / 2**30
        print(f'{n_dm:5d} {ts[0]:11.4f} {ts[1]:11.4f} '
              f'{ts[0]/ts[1]:6.2f}x {gb:16.1f}')
        rows.append(dict(nf=nf, n_dm=n_dm, minor=ts[0], major=ts[1],
                         ratio=ts[0] / ts[1]))
        del dm, vk, offs
        cp.get_default_memory_pool().free_all_blocks()

print('\n' + json.dumps({'nao': a.nao, 'tasks': a.tasks, 'rows': rows}))
