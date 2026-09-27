"""W32: the restricted h1 term's stage C taken apart, on synthetic operands.

``fastdfhessip._get_veff``'s stage C is, per auxiliary atom ``A``,

    j3c_1[x,r,i,q] = sum_s j2c_10[x,r,s] dm_3c[s,i,q]          r in A
    X_A[x,p,q]    += sum_{r in A, i} j3c_1[x,r,i,p] dm_3c[r,i,q]   (exchange)
    vhf_atm[A,x]  += sum_{r in A} j3c_1[x,r] auxvec[r] + ...       (Coulomb)

This probe builds the molecule's real shapes (``nao`` and ``naux`` Cartesian,
the auxiliary functions grouped by atom) with random operands and times the
pieces per atom and grouped, so the GEMM rate can be read at each ``M``
without an SCF.

    python benchmarks/df_h1c_probe.py --xyz benchmarks/molecules/taxol.xyz \\
        --basis def2-svp [--rows 150,600,1500,3000] [--nspin 2]
"""
import time

import numpy as np
import cupy as cp

from _common import build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default='def2-universal-jkfit')
ap.add_argument('--rows', default='0,300,600,1200,2400,4800',
                help='GEMM rows per grouped launch (3 x auxiliary functions); '
                     '0 = one atom per launch')
ap.add_argument('--nspin', type=int, default=1)
ap.add_argument('--nocc', type=int, default=0)
a = ap.parse_args()

from pyscf import df  # noqa: E402
from gpu4pyscf.lib.cupy_helper import contract  # noqa: E402

mol = build_mol(a)
auxmol = df.addons.make_auxmol(mol, a.auxbasis)
nao = mol.nao_cart()
nocc = a.nocc or mol.nelectron // 2
nsp = a.nspin
ncart = (auxmol._bas[:, 1] + 1) * (auxmol._bas[:, 1] + 2) // 2
counts = np.bincount(auxmol._bas[:, 0], weights=ncart).astype(int)
naux = int(counts.sum())
off = np.append(0, np.cumsum(counts))
print(f'# natm {mol.natm}  nao {nao}  nocc {nocc}  naux {naux}  nspin {nsp}  '
      f'naux per atom {counts.min()}-{counts.max()} (mean {counts.mean():.0f})')

rng = cp.random.default_rng(1)
N = nsp * nocc * nao
dm_3c = rng.standard_normal((naux, nsp, nocc, nao)) * 1e-2
j2c_10 = rng.standard_normal((3, naux, naux)) * 1e-2
auxvec = rng.standard_normal(naux)
vhf_atm_ao = cp.zeros((mol.natm, 3, nsp, nao, nao))
vhf_atm = cp.zeros((mol.natm, 3, nsp, nocc, nao))
nA = int(counts.max())
fl_gemm = 2 * 3 * naux * naux * N
fl_k = 2 * 3 * naux * nsp * nocc * nao * nao


def timed(fn, reps=2):
    fn()
    cp.cuda.Stream.null.synchronize()
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        best = min(best, time.perf_counter() - t)
    return best


buf = cp.empty((3, nA, nsp, nocc, nao))
buf1 = cp.empty((nA, nsp, nocc, nao))


def per_atom(parts):
    for i in range(mol.natm):
        p0, p1 = off[i], off[i+1]
        j3c_1 = buf[:, :p1-p0]
        if 'take' in parts:
            dm_atm = cp.take(dm_3c, cp.arange(p0, p1), axis=0,
                             out=buf1[:p1-p0])
        else:
            dm_atm = dm_3c[p0:p1]
        if 'gemm' in parts:
            contract('xrs,sniq->xrniq', j2c_10[:, p0:p1], dm_3c, out=j3c_1)
        if 'k' in parts:
            contract('xrnip,rniq->xnpq', j3c_1, dm_atm, -.5, beta=1,
                     out=vhf_atm_ao[i])
        if 'j' in parts:
            contract('xrniq,r->xniq', j3c_1, auxvec[p0:p1], 1., beta=1,
                     out=vhf_atm[i])
            tmp = cp.einsum('xrs,s->xr', j2c_10[:, p0:p1], auxvec)
            contract('rniq,xr->xniq', dm_atm, tmp, 1., beta=1, out=vhf_atm[i])


rows = []
for parts in (('take', 'gemm', 'k', 'j'), ('gemm',), ('k',), ('j',),
              ('take',)):
    t = timed(lambda: per_atom(parts))
    extra = ''
    if parts == ('gemm',):
        extra = f'  {fl_gemm / t * 1e-12:.1f} TF/s'
    elif parts == ('k',):
        extra = f'  {fl_k / t * 1e-12:.1f} TF/s'
    print(f'per atom  {"+".join(parts):16s} {t * 1e3:9.1f} ms{extra}',
          flush=True)

# one 2-D GEMM per atom, M = 3 nA: the rows of j2c_10 gathered as (x, r)
a_buf = cp.empty((3 * nA, naux))


def per_atom_2d():
    d2 = dm_3c.reshape(naux, N)
    for i in range(mol.natm):
        p0, p1 = off[i], off[i+1]
        m = 3 * (p1 - p0)
        am = a_buf[:m].reshape(3, p1-p0, naux)
        am[:] = j2c_10[:, p0:p1]
        out = buf.reshape(-1)[:m*N].reshape(m, N)
        cp.cublas.gemm('N', 'N', a_buf[:m], d2, out=out)


t = timed(per_atom_2d)
print(f'per atom  cublas 2-D GEMM    {t * 1e3:9.1f} ms  '
      f'{fl_gemm / t * 1e-12:.1f} TF/s', flush=True)

# grouped: consecutive atoms up to `r` GEMM rows per launch
for r in [int(x) for x in a.rows.split(',') if int(x) > 0]:
    groups = []
    g0 = 0
    for i in range(mol.natm):
        if 3 * (off[i+1] - off[g0]) > r and i > g0:
            groups.append((g0, i))
            g0 = i
    groups.append((g0, mol.natm))
    gmax = max(off[g1] - off[g0] for g0, g1 in groups)
    try:
        gbuf = cp.empty((3, gmax, nsp, nocc, nao))
        abuf = cp.empty((3 * gmax, naux))
    except cp.cuda.memory.OutOfMemoryError:
        print(f'rows {r}: out of memory')
        continue
    d2 = dm_3c.reshape(naux, N)

    def grouped(k=True):
        for g0, g1 in groups:
            q0, q1 = off[g0], off[g1]
            m = 3 * (q1 - q0)
            am = abuf[:m].reshape(3, q1-q0, naux)
            am[:] = j2c_10[:, q0:q1]
            out = gbuf.reshape(-1)[:m*N].reshape(m, N)
            cp.cublas.gemm('N', 'N', abuf[:m], d2, out=out)
            if not k:
                continue
            j3 = out.reshape(3, q1-q0, nsp, nocc, nao)
            for i in range(g0, g1):
                p0, p1 = off[i] - q0, off[i+1] - q0
                contract('xrnip,rniq->xnpq', j3[:, p0:p1],
                         dm_3c[off[i]:off[i+1]], -.5, beta=1,
                         out=vhf_atm_ao[i])
    tg = timed(lambda: grouped(False))
    tk = timed(lambda: grouped(True))
    print(f'grouped rows<={r:5d} ({len(groups):3d} launches, '
          f'{gbuf.nbytes / 1e9:.2f} GB)  gemm {tg * 1e3:8.1f} ms '
          f'{fl_gemm / tg * 1e-12:5.1f} TF/s   gemm+k {tk * 1e3:8.1f} ms',
          flush=True)
    gbuf = abuf = None
    cp.get_default_memory_pool().free_all_blocks()

# the exchange contraction per atom as one cuBLAS GEMM per direction and spin


def k_cublas():
    for i in range(mol.natm):
        p0, p1 = off[i], off[i+1]
        j3c_1 = buf[:, :p1-p0]
        dm_atm = dm_3c[p0:p1]
        for x in range(3):
            for n in range(nsp):
                a_ = j3c_1[x, :, n].reshape((p1-p0) * nocc, nao) if nsp == 1 \
                    else cp.ascontiguousarray(j3c_1[x, :, n]).reshape(-1, nao)
                b_ = dm_atm[:, n].reshape((p1-p0) * nocc, nao) if nsp == 1 \
                    else cp.ascontiguousarray(dm_atm[:, n]).reshape(-1, nao)
                cp.cublas.gemm('T', 'N', a_, b_, out=vhf_atm_ao[i, x, n],
                               alpha=-.5, beta=1.)


t = timed(k_cublas)
print(f'per atom  k as cuBLAS        {t * 1e3:9.1f} ms  '
      f'{fl_k / t * 1e-12:.1f} TF/s', flush=True)
