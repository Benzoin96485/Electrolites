"""W5's acceptance test: a tiled first contraction against the dense one.

`DF_WORK_LIST.md` W5 asks for the exchange build's first contraction --
``rhok[i,k,L] = sum_j A[i,j,L] f[j,k]``, which `df_probe.py` measures at
**55.9 ms** of a 115 ms block at 284 atoms -- to run only on the AO tiles the
screening kept.  Its acceptance criterion is "step 1 faster than the dense
contraction at 113 atoms *and* at 284, with the crossover reported", and its
falsification criterion is that the tile GEMMs lose to one dense contraction
at taxol's fill.  This script is that test.

It times four things per (molecule, AO order, tile size), round robin so a
competing job's drift does not land on one variant (`DF_WORK_LIST.md` §5.2):

    unpack  dense   GPU4PySCF's `fill_symmetric` into `(nao, nao, nL)`
    step1   dense   `contract('Lij,njk->nikL')`, which is what `fastdf` runs
    unpack  tiled   W14's one-pass CUDA scatter into `_dfblock.RowCompressed`'s
                    buffer, and (`--unpacks kernel,cupy`) the transpose into a
                    staging buffer plus two fancy-index scatters it replaced
    step1   tiled   one `contract('ijL,jk->ikL')` per tile row

`--check` compares both unpacks against `fill_symmetric`'s own output pair by
pair, not just the two contractions against each other: the row-compressed
buffer and the dense one have different shapes, and the only statement that
means anything about a scatter is that every stored pair landed on the same
number in both.

No SCF and no cderi: the contraction's cost does not depend on the values, so
the block is random and `nocc` is the molecule's own.  That is what lets the
284-atom case run in seconds instead of the twenty minutes a real build costs
-- and the pattern, which is the only thing that matters here, is the real
one.

    python benchmarks/df_block_bench.py --cases taxol:0:0,PfPMT:0:-1
    python benchmarks/df_block_bench.py --cases penicillinV:0:0 --check
"""
import argparse
import json
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--cases', default='taxol:0:0,PfPMT:0:-1',
                help='comma separated name:spin:charge from benchmarks/molecules')
ap.add_argument('--tiles', default='32,64')
ap.add_argument('--orders', default='native,rcm')
ap.add_argument('--nl', type=int, default=128,
                help='auxiliary functions per block; 128 is what `df_probe.py` '
                     'measures the 284-atom build choosing')
ap.add_argument('--nocc', type=int, default=0, help='override nelec/2')
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--check', action='store_true',
                help='also verify the tiled contraction against the dense one')
ap.add_argument('--unpacks', default='kernel',
                help='which tiled unpacks to time: `kernel` (W14), `cupy` '
                     '(what it replaced), or both')
a = ap.parse_args()

import cupy as cp                                            # noqa: E402
from cupy.cuda import cublas                                 # noqa: E402
from pyscf import gto                                        # noqa: E402
from gpu4pyscf.lib.cupy_helper import (                      # noqa: E402
    contract, fill_symmetric, get_avail_mem)
from electrolites import _dfblock                          # noqa: E402
from electrolites._dfblock import (pair_indices,             # noqa: E402
    atom_order, ao_perm, RowCompressed)

UNPACKS = a.unpacks.split(',')


def timed(fn, reps):
    """Best of `reps`, in ms.  Best rather than mean because the failure mode
    on a shared node is a slow outlier, never a fast one."""
    out = []
    for _ in range(reps):
        cp.cuda.Device().synchronize()
        t = time.perf_counter()
        fn()
        cp.cuda.Device().synchronize()
        out.append((time.perf_counter() - t) * 1e3)
    return min(out)


_ONE, _ZERO = np.array(1.0), np.array(0.0)


def syrk(rhok, vk, nao, K):
    """`fastdf`'s step 2, for scale: it is the same call in both variants, so
    the ratio on a whole block is (unpack + step1 + step2) either way."""
    cublas.dsyrk(cp.cuda.device.get_cublas_handle(),
                 cublas.CUBLAS_FILL_MODE_LOWER, cublas.CUBLAS_OP_T,
                 nao, K, _ONE.ctypes.data, rhok.data.ptr, K,
                 _ONE.ctypes.data, vk.data.ptr, nao)


def fastdf_blksize(nao, nocc, naux):
    """What `fastdf._blksize` would choose here with a host-resident cderi --
    reported because the dense `(nao, nao, blk)` buffer is what caps it, and
    the tiled buffer is `fill` times as large."""
    words = int(get_avail_mem() * .2 / 8)
    return max(1, min(256, naux, words // (nao * nao + nao * nocc)))


print(f'{"molecule":12s} {"order":7s} {"tile":>4s} {"fill":>6s} '
      f'{"unpack_d":>9s} {"step1_d":>8s} {"TF/s_d":>7s} '
      f'{"unpack_t":>9s} {"step1_t":>8s} {"TF/s_t":>7s} '
      f'{"step2":>7s} {"step1":>6s} {"total":>6s} {"block":>6s} '
      f'{"buf_d":>7s} {"buf_t":>7s}')
results = []
for case in a.cases.split(','):
    name, spin, charge = (case.split(':') + ['0', '0'])[:3]
    mol = gto.M(atom=f'benchmarks/molecules/{name}.xyz', basis=a.basis,
                spin=int(spin), charge=int(charge), verbose=0)
    rows0, cols0, nao, naux = pair_indices(mol, a.auxbasis)
    nocc = a.nocc or mol.nelectron // 2
    nL = min(a.nl, naux)
    npair = len(rows0)
    flop_dense = 2 * nao**2 * nocc * nL
    print(f'# {name}: nao {nao} naux {naux} nocc {nocc} npair {len(rows0)}  '
          f'block {nL} (fastdf would choose '
          f'{fastdf_blksize(nao, nocc, naux)} here)', flush=True)

    # One auxiliary block.  Random, because neither contraction's cost
    # depends on the values -- but random *consistently*: the pair list holds
    # a few off-diagonal AO pairs in both orders (`int3c2e_bdiv.py:487` says
    # why), and a real cderi gives those two entries the same number.  Drawing
    # them independently would make the two unpacks disagree by whichever
    # write landed last and turn `--check` into noise.
    key = (np.minimum(rows0, cols0) * nao + np.maximum(rows0, cols0))
    _, inv = np.unique(key, return_inverse=True)
    blk = cp.random.random((nL, int(inv.max()) + 1))[:, cp.asarray(inv)]
    blk = cp.ascontiguousarray(blk)
    for order in a.orders.split(','):
        perm = ao_perm(mol, atom_order(mol, rows0, cols0, order))
        rows, cols = perm[rows0], perm[cols0]
        pair_idx = cp.asarray(rows * nao + cols, dtype=np.int32)
        f = cp.random.random((nao, nocc))

        work = cp.zeros((nao, nao, nL))
        rl_d = cp.empty((1, nao, nocc, nL))
        vk = cp.zeros((nao, nao))

        def step2():
            syrk(rl_d[0], vk, nao, nocc * nL)

        def dense_unpack():
            fill_symmetric(blk.T, pair_idx, nao, 0, nL, out=work)

        def dense_step1():
            contract('Lij,njk->nikL', work.transpose(2, 0, 1), f[None],
                     out=rl_d)

        for T in [int(x) for x in a.tiles.split(',')]:
            rc = RowCompressed(rows, cols, nao, T, nL).to_device()
            fg = rc.factor(f)
            rl_t = cp.empty((nao, nocc, nL))
            flop_tiled = 2 * rc.nrow * nocc * nL

            def tiled_unpack(mode=UNPACKS[0]):
                _dfblock.UNPACK = mode
                rc.unpack(blk, nL)

            def tiled_step1():
                rc.step1(fg, rl_t, nL)

            timing = ([('unpack_d', dense_unpack), ('step1_d', dense_step1)]
                      + [(f'unpack_t:{m}', (lambda m=m: tiled_unpack(m)))
                         for m in UNPACKS]
                      + [('step1_t', tiled_step1), ('step2', step2)])
            for _, fn in timing:
                fn()                                   # warm the plans
            cp.cuda.Device().synchronize()
            t = {}
            for _ in range(a.reps):                    # round robin
                for key, fn in timing:
                    t[key] = min(t.get(key, 1e9), timed(fn, 1))
            t['unpack_t'] = t[f'unpack_t:{UNPACKS[0]}']

            if a.check:
                # every stored pair, against `fill_symmetric`'s own output --
                # the two buffers have different shapes, so the contraction
                # agreeing is a weaker statement than the scatter agreeing
                dense_unpack()
                d1 = cp.asarray(rc._d1)
                flat = rc.buf.reshape(rc.nrow, rc.ldL)
                rows_d, cols_d = cp.asarray(rows), cp.asarray(cols)
                for mode in UNPACKS:
                    tiled_unpack(mode)
                    worst = 0.
                    for c0 in range(0, npair, 1 << 17):
                        sl = slice(c0, min(c0 + (1 << 17), npair))
                        ref = work[rows_d[sl], cols_d[sl], :nL]
                        worst = max(worst, float(
                            cp.abs(flat[d1[sl], :nL] - ref).max()))
                    print(f'#   check {name}/{order}/T={T} unpack={mode}: '
                          f'max |tiled - fill_symmetric| over stored pairs '
                          f'= {worst:.2e}')
                del d1, rows_d, cols_d
                tiled_unpack(); dense_step1(); tiled_step1()
                err = float(cp.abs(rl_d[0] - rl_t).max()
                            / cp.abs(rl_d).max())
                print(f'#   check {name}/{order}/T={T}: '
                      f'step 1 max relative deviation {err:.2e}')

            row = dict(name=name, order=order, tile=T, nao=nao, naux=naux,
                       nocc=nocc, nL=nL, fill=rc.fill,
                       buf_dense_gb=work.nbytes / 1e9,
                       buf_tiled_gb=rc.nbytes(UNPACKS[0]) / 1e9,
                       tf_dense=flop_dense / t['step1_d'] / 1e9,
                       tf_tiled=flop_tiled / t['step1_t'] / 1e9,
                       unpacks={m: t[f'unpack_t:{m}'] for m in UNPACKS},
                       step1_ratio=t['step1_d'] / t['step1_t'],
                       total_ratio=((t['unpack_d'] + t['step1_d'])
                                    / (t['unpack_t'] + t['step1_t'])),
                       block_ratio=((t['unpack_d'] + t['step1_d'] + t['step2'])
                                    / (t['unpack_t'] + t['step1_t']
                                       + t['step2'])), **t)
            print(f'{name:12s} {order:7s} {T:4d} {100*rc.fill:5.1f}% '
                  f'{t["unpack_d"]:9.2f} {t["step1_d"]:8.2f} '
                  f'{row["tf_dense"]:7.1f} '
                  f'{t["unpack_t"]:9.2f} {t["step1_t"]:8.2f} '
                  f'{row["tf_tiled"]:7.1f} '
                  f'{t["step2"]:7.2f} '
                  f'{row["step1_ratio"]:5.2f}x {row["total_ratio"]:5.2f}x '
                  f'{row["block_ratio"]:5.2f}x '
                  f'{row["buf_dense_gb"]:6.2f}G {row["buf_tiled_gb"]:6.2f}G'
                  + (''.join(f'  {m}={t[f"unpack_t:{m}"]:.2f}'
                             for m in UNPACKS) if len(UNPACKS) > 1 else ''),
                  flush=True)
            results.append(row)
            del rc, fg, rl_t
            cp.get_default_memory_pool().free_all_blocks()
        del work, rl_d, vk
        cp.get_default_memory_pool().free_all_blocks()
    del blk
    cp.get_default_memory_pool().free_all_blocks()
print('RESULT ' + json.dumps(dict(basis=a.basis, rows=results)))
