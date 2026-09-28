"""W35: the restricted h1 term's stage D2 (W30's occupied form) taken apart.

``fastdfhessip._get_veff``'s stage D2, with item 12 taken, is for every
auxiliary block of ``dk`` functions, every direction ``x`` and every atom
``A`` (``u`` its AOs):

    vhf_atm[A,x][j,q] += sum_{u in A, r} j3c[u,q,r] s[u,r,j]
                       + sum_{u in A, r} j3c_pri[u,r,j] t[u,r,q]

two contractions per atom and direction, ``2 natm`` launches per block and
direction, and ``2 2 nao dk nao nocc`` flops per block and direction however
the atoms are split.  This probe builds the molecule's real shapes (Cartesian
``nao``, the AOs grouped by atom) with random operands and times one block's
D2 at each ``dk`` as

* ``cutensor``  the per-atom ``contract`` calls the module makes;
* ``cublas``    the same per-atom products as cuBLAS GEMMs, with ``j3c``
                held transposed (``jT[q,u,r]``) so that ``(u, r)`` is one
                contiguous index of both operands;
* ``batched``   those GEMMs as one ``cublasDgemmBatched`` per distinct atom
                size and product (a few launches instead of ``2 natm``);

and prints each one's rate and its host-side enqueue time (the time to issue
the launches with nothing synchronised), which is what a launch-bound loop
costs.

    python benchmarks/df_h1d2_probe.py --xyz benchmarks/molecules/taxol.xyz \\
        --basis def2-svp [--dk 32,64,128,256,512] [--nspin 2]
"""
import time

import numpy as np
import cupy as cp

from _common import build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--dk', default='16,32,64,128,256,512')
ap.add_argument('--nspin', type=int, default=1)
ap.add_argument('--nocc', type=int, default=0)
ap.add_argument('--variants', default='cutensor,cublas,batched')
a = ap.parse_args()

from gpu4pyscf.lib.cupy_helper import contract  # noqa: E402
from electrolites import _h1d2  # noqa: E402

mol = build_mol(a)
ncart = (mol._bas[:, 1] + 1) * (mol._bas[:, 1] + 2) // 2 * mol._bas[:, 3]
counts = np.bincount(mol._bas[:, 0], weights=ncart,
                     minlength=mol.natm).astype(int)
nao = int(counts.sum())
aoslices = np.append(0, np.cumsum(counts))
natm = mol.natm
nocc = a.nocc or (mol.nelectron // 2 if a.nspin == 1 else
                  (mol.nelectron + 1) // 2)
ns = a.nspin
print(f'# natm {natm}  nao {nao}  nocc {nocc}  nspin {ns}  atom sizes '
      f'{sorted(set(counts.tolist()))}', flush=True)
plan = _h1d2.AtomGroups(aoslices)
print(f'# {plan.ngroups} distinct atom sizes: '
      + ', '.join(f'{n} AOs x {len(g)}' for n, g in plan.groups), flush=True)

rng = cp.random.default_rng(1)
vhf = cp.zeros((natm, ns, nocc, nao))


def timed(fn, reps=3):
    fn()
    cp.cuda.Stream.null.synchronize()
    best = hbest = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        h = time.perf_counter() - t
        cp.cuda.Stream.null.synchronize()
        best = min(best, time.perf_counter() - t)
        hbest = min(hbest, h)
    return best, hbest


for dk in [int(x) for x in a.dk.split(',')]:
    try:
        j3c = rng.standard_normal((nao, nao, dk)) * 1e-2
        jT = cp.ascontiguousarray(j3c.transpose(1, 0, 2))
        s = rng.standard_normal((ns, nao, dk, nocc)) * 1e-2
        t_ = rng.standard_normal((ns, nao, dk, nao)) * 1e-2
        pri = rng.standard_normal((ns, nao, dk, nocc)) * 1e-2
        priT = cp.ascontiguousarray(pri.transpose(0, 3, 1, 2))  # (n,i,p,r)
    except cp.cuda.memory.OutOfMemoryError:
        print(f'dk {dk}: out of memory')
        break
    fl = 2 * 2 * nao * dk * nao * nocc * ns

    def cutensor():
        for i, (p0, p1) in enumerate(zip(aoslices[:-1], aoslices[1:])):
            if ns == 1:
                contract('uqr,urj->jq', j3c[p0:p1], s[0, p0:p1], -.5, beta=1,
                         out=vhf[i, 0])
                contract('urj,urq->jq', pri[0, p0:p1], t_[0, p0:p1], -.5,
                         beta=1, out=vhf[i, 0])
            else:
                contract('uqr,nurj->njq', j3c[p0:p1], s[:, p0:p1], -.5,
                         beta=1, out=vhf[i])
                contract('nurj,nurq->njq', pri[:, p0:p1], t_[:, p0:p1], -.5,
                         beta=1, out=vhf[i])

    def cublas():
        _h1d2.d2_per_atom(jT, s, priT, t_, vhf, -.5, aoslices)

    def batched():
        plan.d2(jT, s, priT, t_, vhf, -.5)

    fns = {'cutensor': cutensor, 'cublas': cublas, 'batched': batched}
    ref = None
    line = f'dk {dk:4d}'
    for v in a.variants.split(','):
        vhf[:] = 0
        fns[v]()
        out = vhf.get()
        if ref is None:
            ref = out
        err = np.abs(out - ref).max() / np.abs(ref).max()
        tt, th = timed(fns[v])
        line += (f'   {v} {tt*1e3:7.2f} ms {fl/tt*1e-12:5.1f} TF/s '
                 f'(host {th*1e3:6.2f}, err {err:.0e})')
    print(line, flush=True)
    j3c = jT = s = t_ = pri = priT = None
    cp.get_default_memory_pool().free_all_blocks()
