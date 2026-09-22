"""Where ``contract('Lpq,snqi->snpiL', ...)`` stops being planable.

``docs/DF_WORK_LIST.md`` **W8**.  GPU4PySCF's DF Hessian CPHF build sizes its
density batch from free memory alone (``df/hessian/rhf.py:1291``) and then
hands the whole batch to one cuTENSOR contraction.  On an 80 GB card that
batch is the *whole* right-hand side, and past a certain shape cuTENSOR
refuses the plan outright:

    CuTensorError: CUTENSOR_STATUS_NOT_SUPPORTED

raised from ``estimateWorkspaceSize`` -- so it is the operation descriptor
that is rejected, not the workspace that is missing, and no amount of free
memory fixes it.  Measured in ``benchmarks/df_hess_bench.py``, that is a
**42-atom def2-TZVPP** DF Hessian and a **113-atom def2-SVP** one, both of
which fit comfortably in memory and neither of which GPU4PySCF can run.

This sweep says what the boundary is a function of, so the cap
``electrolites/fastdfhess.py`` applies is a measured number rather than a
guessed one.  It allocates nothing but the operands, so it can walk a grid of
``(nao, nocc, blksize, dm_batch)`` in one process.

    python benchmarks/df_hess_cutensor_limit.py
    python benchmarks/df_hess_cutensor_limit.py --nao 1123 --nocc 226
"""
import argparse
import json

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--cases', default='',
                help='semicolon separated nao,nocc,blksize,nb quadruples; '
                     'the default walks the four systems the round measures')
ap.add_argument('--bisect', action='store_true',
                help='for each (nao, nocc, blksize), bisect on dm_batch for '
                     'the largest one cuTENSOR will plan')
ap.add_argument('--contiguous', action='store_true',
                help='hand the contraction a C-contiguous (L,p,q) block '
                     'instead of the strided view `DF.loop` yields')
a = ap.parse_args()

import numpy as np                                           # noqa: E402
import cupy as cp                                            # noqa: E402
from gpu4pyscf.lib.cupy_helper import contract               # noqa: E402

GB = 1024.0 ** 3


def try_shape(nao, nocc, nL, nb, contiguous=False):
    """Can cuTENSOR plan the contraction at this shape?  Returns (ok, why)."""
    need = (nL * nao * nao + nb * nao * nocc * nL + nb * nao * nocc) * 8
    free = cp.cuda.runtime.memGetInfo()[0]
    if need > 0.8 * free:
        return None, f'skipped, needs {need / GB:.1f} GB of {free / GB:.1f}'
    try:
        work = cp.zeros((nao, nao, nL))
        cderi = work.transpose(2, 0, 1)
        if contiguous:
            cderi = cp.ascontiguousarray(cderi)
        x = cp.zeros((nb, nao, nocc))
        out = cp.empty((nb, nao, nocc, nL))
        contract('Lpq,nqi->npiL', cderi, x, out=out)
        cp.cuda.Stream.null.synchronize()
        ok, why = True, ''
    except Exception as exc:                                  # noqa: BLE001
        ok, why = False, f'{type(exc).__name__}'
    finally:
        work = cderi = x = out = None
        cp.get_default_memory_pool().free_all_blocks()
    return ok, why


DEFAULT = [
    # nao, nocc, blksize, dm_batch, label
    (430, 92, 256, 126, 'penicillin V / def2-SVP, 42 atoms'),
    (478, 39, 256, 66, 'alanine dipeptide / def2-TZVPP, 22 atoms'),
    (1007, 92, 256, 126, 'penicillin V / def2-TZVPP, 42 atoms'),
    (1123, 226, 256, 24, 'taxol / def2-SVP, 8 atoms of RHS'),
    (1123, 226, 256, 339, 'taxol / def2-SVP, all 113 atoms'),
]

cases = []
if a.cases:
    for chunk in a.cases.split(';'):
        nao, nocc, nL, nb = (int(x) for x in chunk.split(','))
        cases.append((nao, nocc, nL, nb, f'{nao},{nocc},{nL},{nb}'))
else:
    cases = DEFAULT

print(f'{"nao":>6s} {"nocc":>5s} {"blk":>5s} {"nb":>5s} '
      f'{"out elements":>14s} {"extent product":>16s}  result')
results = []
for nao, nocc, nL, nb, label in cases:
    ok, why = try_shape(nao, nocc, nL, nb, a.contiguous)
    elems = nb * nao * nocc * nL
    extent = nL * nao * nao * nb * nocc
    print(f'{nao:6d} {nocc:5d} {nL:5d} {nb:5d} {elems:14.3e} {extent:16.3e}  '
          f'{"ok" if ok else ("SKIP " + why if ok is None else "FAIL " + why)}'
          f'   {label}')
    results.append({'nao': nao, 'nocc': nocc, 'blksize': nL, 'nb': nb,
                    'elements': elems, 'extent': extent, 'ok': ok,
                    'why': why, 'label': label})

if a.bisect:
    print('\n## the largest dm_batch cuTENSOR will plan, by bisection')
    for nao, nocc, nL, _, label in cases:
        lo, hi = 0, 1
        while True:
            ok, why = try_shape(nao, nocc, nL, hi)
            if ok is None:
                break
            if not ok:
                break
            lo = hi
            hi *= 2
            if hi > 4096:
                break
        while hi - lo > 1:
            mid = (lo + hi) // 2
            ok, _ = try_shape(nao, nocc, nL, mid)
            if ok:
                lo = mid
            else:
                hi = mid
        if lo:
            print(f'  nao {nao:5d} nocc {nocc:4d} blk {nL:4d}  '
                  f'dm_batch <= {lo:5d}   out {lo * nao * nocc * nL:.3e} '
                  f'elements, extent {nL * nao * nao * lo * nocc:.3e}   '
                  f'{label}')
            results.append({'bisect': True, 'nao': nao, 'nocc': nocc,
                            'blksize': nL, 'max_nb': lo, 'label': label,
                            'elements': lo * nao * nocc * nL,
                            'extent': nL * nao * nao * lo * nocc})
        else:
            print(f'  nao {nao:5d} nocc {nocc:4d} blk {nL:4d}  '
                  f'no dm_batch works   {label}')

print('\n@@JSON ' + json.dumps(results))
