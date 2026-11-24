"""The four-centre block engine on GPU4PySCF >= 1.9 (`electrolites/_i4blocks.py`)
and `fastmp2`'s (ia|pq) built on it, against PySCF's CPU integrals.

Every bra group pair's block ``(ij|kl)`` is contracted with random orbital
coefficients -- both bra orders of an off-diagonal group pair, once for a
diagonal one -- and the sum compared with ``ao2mo`` of ``int2e``; small groups
force many off-diagonal pairs.  Then ``fastmp2.get_int4c2e_ovpq`` (on device,
and batched one occupied orbital at a time into a host result) against the
same reference.  Skipped on a release that still has the 1.8 driver.

    python tests/test_i4blocks.py
"""
import os
import sys

import numpy as np
import cupy as cp
from pyscf import gto, ao2mo

from electrolites import compat

if not compat.INT4C2E_FULL:
    print('skip: GPU4PySCF < 1.9 (the 1.8 int4c2e driver is used there)')
    sys.exit(0)

from electrolites._i4blocks import BlockOpt                     # noqa: E402

fails = []


def check(tag, got, ref, tol=1e-10):
    err = float(np.abs(np.asarray(got) - np.asarray(ref)).max())
    ok = err <= tol * max(1., float(np.abs(ref).max()))
    if not ok:
        fails.append(f'{tag}: {err:.3e}')
    print(f'  {"ok  " if ok else "FAIL"} {tag:52s} {err:.3e}', flush=True)


MOLS = {
    'water def2-svp': gto.M(atom='O 0 0 0.117; H 0 0.757 -0.467; '
                            'H 0 -0.757 -0.467', basis='def2-svp', verbose=0),
    'CH2O def2-tzvp (f)': gto.M(atom='C 0 0 0; O 0 0 1.21; H 0 0.94 -0.59; '
                                'H 0 -0.94 -0.59', basis='def2-tzvp',
                                verbose=0),
}
rng = np.random.default_rng(3)
for name, mol in MOLS.items():
    nao0 = mol.nao
    n = 5
    M = rng.standard_normal((nao0, n))
    ref = ao2mo.general(mol, (M, M, M, M), compact=False).reshape(n, n, n, n)
    for g in (2, 5, None):
        opt = BlockOpt(mol, g)
        Ms = opt.coeff @ cp.asarray(M)
        out = cp.zeros((n, n, n, n))
        for gid in range(len(opt.log_qs)):
            ish0, ish1, jsh0, jsh1, a0, a1, b0, b1 = opt.ao_ranges(gid)
            eri = opt.block(gid)                         # [i, k, j, l]
            # one index at a time (a single 5-operand einsum can plan a
            # dense intermediate far larger than the block)
            t = cp.einsum('ikjl,ip->pkjl', eri, Ms[a0:a1])
            t = cp.einsum('pkjl,jq->pqkl', t, Ms[b0:b1])
            t = cp.einsum('pqkl,kr->pqrl', t, Ms)
            t = cp.einsum('pqrl,ls->pqrs', t, Ms)
            out += t
            if ish0 != jsh0:
                out += t.transpose(1, 0, 2, 3)
        check(f'{name} blocks, group_size={g}', out.get(), ref)

    from electrolites import fastmp2
    nocc = 4
    mo = rng.standard_normal((nao0, 9))
    o, v = mo[:, :nocc], mo[:, nocc:]
    ref_ovov = ao2mo.general(mol, (o, v, o, v), compact=False).reshape(
        nocc, 5, nocc, 5)
    got = fastmp2.get_int4c2e_ovov(mol, o, v)
    check(f'{name} fastmp2 (ia|jb), device', cp.asnumpy(got), ref_ovov)
    os.environ['FASTMP2_BATCH'] = '1'
    os.environ['FASTMP2_HOST'] = '1'
    got = fastmp2.get_int4c2e_ovov(mol, o, v)
    os.environ.pop('FASTMP2_BATCH')
    os.environ.pop('FASTMP2_HOST')
    check(f'{name} fastmp2 (ia|jb), host, batch 1',
          np.asarray(got).reshape(nocc, 5, nocc, 5), ref_ovov)
    p = mo[:, :3]
    ref_ovpq = ao2mo.general(mol, (o, v, p, v), compact=False).reshape(
        nocc, 5, 3, 5)
    got = fastmp2.get_int4c2e_ovpq(mol, o, v, p, v)
    check(f'{name} fastmp2 (ia|pq)', cp.asnumpy(got), ref_ovpq)

if fails:
    print('\nFAILED:\n  ' + '\n  '.join(fails))
    sys.exit(1)
print('\nall ok')
