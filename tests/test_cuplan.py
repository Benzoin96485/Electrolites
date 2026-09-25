"""Does asking cuTENSOR about a descriptor predict what a real call does?

`docs/DF_WORK_LIST.md` **W21**.  ``electrolites/_cuplan.py`` decides block and
batch sizes by asking cuTENSOR to plan a contraction on descriptors that point
into a one-kilobyte buffer.  That is only worth anything if the answer is the
one a real ``contract`` on real arrays gets, so this allocates the arrays --
outputs of 2.7 and 4.3 GB -- and checks both directions:

* an extent the query refuses raises ``CUTENSOR_STATUS_NOT_SUPPORTED`` for
  real, and an extent it accepts runs and is right (against ``cp.matmul``);
* the extent the query accepts may be *larger* than one it refuses -- the
  refusal is not a size -- and both of those are checked for real too;
* ``_cuplan.prange`` hands back only extents that plan, including the
  remainder, on the shape the gradient's ip1 loop met at 113 atoms;
* the three modules that consult it choose what W8, W10 and W17 chose or
  something that plans: ``fastdfhess`` the density batch W8 bisected by hand
  (22 at nao 1007), ``fastdfgrad`` no block of 914.

    python tests/test_cuplan.py

Needs one GPU with about 10 GB free.
"""
import sys

import cupy as cp

from gpu4pyscf.lib.cupy_helper import contract

from electrolites import _cuplan
from electrolites._cuplan import planable, CuTensorError

fails = 0


def check(ok, what):
    global fails
    fails += not ok
    print(f'{"ok  " if ok else "FAIL"} {what}')


def real(nocc, nao, dk):
    """``contract('iqr,pi->pqr')`` for real: (ran, relative error)."""
    rng = cp.random.default_rng(7)
    a = rng.standard_normal((nocc, nao, dk))
    b = rng.standard_normal((nao, nocc))
    out = cp.empty((nao, nao, dk))
    try:
        contract('iqr,pi->pqr', a, b, out=out)
    except CuTensorError:
        return False, None
    finally:
        cp.cuda.Stream.null.synchronize()
    # a slab of the answer against cuBLAS
    ref = (b @ a[:, :7].reshape(nocc, -1)).reshape(nao, 7, dk)
    err = float(abs(out[:, :7] - ref).max() / abs(ref).max())
    a = b = out = None
    cp.get_default_memory_pool().free_all_blocks()
    return True, err


def main():
    print(f'# cuTENSOR {_cuplan._cb.get_version()}  '
          f'{cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}')
    nocc, nao = 8, 800
    # 800**2 * 839 = 5.37e8 > 2**29 and 839 is prime; 838 = 2 x 419 and a
    # piece of 419 is under the limit; 1024 = 2**10 is larger than both
    for dk, want in ((839, False), (838, True), (1024, True), (1678, False)):
        q = planable('iqr,pi->pqr', (nocc, nao, dk), (nao, nocc),
                     (nao, nao, dk))
        ran, err = real(nocc, nao, dk) if dk * nao * nao < 1.1e9 else (q, 0.)
        check(q == want and ran == q and (err is None or err < 1e-13),
              f'iqr,pi->pqr at (nocc {nocc}, nao {nao}, r {dk}): query '
              f'{"accepts" if q else "refuses"}, a real call '
              f'{"runs" if ran else "is refused"}'
              + (f', error {err:.1e}' if err is not None else ''))

    # the gradient's ip1 loop on taxol/def2-SVP, Cartesian sorted nao 1185
    nao, nocc = 1185, 226

    def ip1(dk):
        return (planable('rji,qj->iqr', (dk, nocc, nocc), (nao, nocc),
                         (nocc, nao, dk))
                and planable('iqr,pi->pqr', (nocc, nao, dk), (nao, nocc),
                             (nao, nao, dk)))
    check(not ip1(914) and ip1(1096),
          'ip1 at taxol: 914 refused and 1096 accepted, the pair W10 met')
    blocks = list(_cuplan.prange(0, 2010, 1096, ip1))
    ext = [k1 - k0 for k0, k1 in blocks]
    check(sum(ext) == 2010 and all(ip1(e) for e in ext),
          f'prange(0, 2010, 1096) -> {ext}, every extent plans')
    for n in (1000, 2000, 5598):
        for step in (382, 1096, 1456, 3072):
            ext = [k1 - k0 for k0, k1 in _cuplan.prange(0, n, step, ip1)]
            if not (sum(ext) == n and all(ip1(e) for e in ext)
                    and max(ext) <= step):
                check(False, f'prange(0, {n}, {step}) -> {ext}')
                break
        else:
            continue
        break
    else:
        check(True, 'prange covers 1000, 2000 and 5598 at four steps with '
                    'extents that all plan')

    # fastdfhess: the batch W8 bisected by hand at (nao 1007, nocc 92, 256)
    from electrolites import fastdfhess
    plans = fastdfhess._batch_plans(1, 1007, 92, 256)
    nb = _cuplan.largest(126, plans)
    check(nb == 22, f'fastdfhess batch at nao 1007, nocc 92: {nb} (W8 '
                    'bisected 22)')
    nb = _cuplan.largest(600, fastdfhess._batch_plans(1, 430, 92, 256))
    check(nb == 530, f'fastdfhess batch at nao 430, nocc 92: {nb} (W8 '
                     'bisected 530)')

    print(f'\n_cuplan {_cuplan.stats()}')
    print('FAILURES' if fails else 'all ok')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
