"""Does ``planguard`` turn cuTENSOR's refusals into the right answer?

`docs/DF_WORK_LIST.md` **W27**.  ``electrolites/planguard.py`` wraps
``gpu4pyscf/lib/cutensor.py:contraction`` so that a contraction cuTENSOR
refuses is cut along a free output mode into pieces it plans.  Checked, on
real arrays (outputs of 4.5-10 GB):

* the contraction W21 found refused in the DF gradient at 113 atoms, called
  the way upstream calls it -- through the ``contract`` a GPU4PySCF module
  bound at import -- raises without the guard and is right with it (against
  one ``cp.matmul``), with ``out``, ``alpha`` and ``beta``;
* the CPHF's own ``gen_vind`` product (``hessian/rhf.py``, ``npi,qi->npq``)
  at a right-hand-side count a 150-atom Hessian reaches, which is refused
  for ``nao = 1123``;
* a pattern whose largest free mode is on the *second* operand, and one whose
  output is a strided view;
* a contraction under the limit is not touched (no query, no cut).

    python tests/test_planguard.py

Needs one GPU with about 30 GB free.
"""
import sys

import numpy as np
import cupy as cp

from gpu4pyscf.hessian import rhf as hess_rhf       # binds `contract` now
from gpu4pyscf.lib.cupy_helper import contract

from electrolites import _cuplan
from electrolites._cuplan import CuTensorError

fails = 0


def check(ok, what):
    global fails
    fails += not ok
    print(f'{"ok  " if ok else "FAIL"} {what}', flush=True)


def rel(got, ref):
    return float(cp.abs(got - ref).max() / cp.abs(ref).max())


def main():
    rng = cp.random.default_rng(7)
    nocc, nao, dk = 226, 1185, 914
    tmp = rng.standard_normal((nocc, nao, dk))
    c = rng.standard_normal((nao, nocc))
    ref = (c @ tmp.reshape(nocc, -1)).reshape(nao, nao, dk)

    # without the guard: upstream's refusal, as W21 measured it
    try:
        contract('iqr,pi->pqr', tmp, c)
        check(False, 'unguarded W21 contraction was refused')
    except CuTensorError:
        check(True, 'unguarded W21 contraction was refused')
    cp.get_default_memory_pool().free_all_blocks()

    import electrolites
    electrolites.patch('planguard')
    q0 = _cuplan.stats()['queries']
    got = contract('iqr,pi->pqr', tmp, c)
    check(rel(got, ref) < 1e-14,
          f'guarded, out=None: {rel(got, ref):.1e}  {_cuplan.stats()}')
    got[:] = 1.
    contract('iqr,pi->pqr', tmp, c, 2., .5, out=got)
    check(rel(got, 2 * ref + .5) < 1e-14, 'guarded, with out, alpha, beta')
    # the same pattern with the operands swapped: the free mode is then on
    # the second one
    got = contract('pi,iqr->pqr', c, tmp)
    check(rel(got, ref) < 1e-14, 'guarded, free mode on the second operand')
    # a strided output: every other slice of a larger array
    big = cp.zeros((nao, nao, 2 * dk))
    view = big[:, :, ::2]
    contract('iqr,pi->pqr', tmp, c, out=view)
    check(rel(view, ref) < 1e-14 and float(abs(big[:, :, 1::2]).max()) == 0,
          'guarded, into a strided view (and nothing else written)')
    big = view = got = ref = tmp = c = None
    cp.get_default_memory_pool().free_all_blocks()

    # hessian/rhf.py:gen_vind's density product at 450 right-hand sides
    n, nao, nocc = 450, 1123, 226
    mo1_mo = rng.standard_normal((n, nao, nocc))
    orbo = rng.standard_normal((nao, nocc))
    planned = _cuplan.planable('npi,qi->npq', mo1_mo, orbo, (n, nao, nao))
    got = hess_rhf.contract('npi,qi->npq', mo1_mo, orbo)
    ref = (mo1_mo.reshape(-1, nocc) @ orbo.T).reshape(n, nao, nao)
    check(not planned and rel(got, ref) < 1e-14,
          f'gen_vind product at n={n}, nao={nao} (refused: {not planned}): '
          f'{rel(got, ref):.1e}')
    got = ref = mo1_mo = None
    cp.get_default_memory_pool().free_all_blocks()

    # under the limit: the plain call
    q0, cut0 = _cuplan.stats()['queries'], _cuplan.stats().get('cut', 0)
    a = rng.standard_normal((64, 80))
    got = contract('ij,kj->ik', a, a)
    check(rel(got, a @ a.T) < 1e-13 and _cuplan.stats()['queries'] == q0
          and _cuplan.stats().get('cut', 0) == cut0,
          'a contraction under 2**29 is the plain call')

    print(f'\n_cuplan {_cuplan.stats()}')
    print('FAILURES' if fails else 'all ok')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
