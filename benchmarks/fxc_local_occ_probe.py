"""W36's precondition: how many localised occupied orbitals does a grid block see?

`docs/DF_WORK_LIST.md` **W36**.  W33's occupied-side route contracts every
grid block over all ``nocc`` occupied orbitals, so its margin over the
AO-matrix route is ``nsub/nocc`` and falls with the molecule.  With the
occupied orbitals rotated to localised ones (``C_o -> C_o U``, ``M -> M U``,
which leaves ``D = M (2C_o)^T + h.c.`` and ``V C_o C_o^T`` unchanged), an
orbital whose values ``X_k[n][g] = sum_u C_uk ao[n][u][g]`` are negligible on
a block contributes nothing to that block's density ``sum_k Y_k X_k`` nor to
its potential ``(V C_o)[:, k] = sum_n ao[n] W_k[n]`` (``W_k`` is linear in
``X_k``).  The entry says to count these on the block's AO subset *before*
writing any code, because W15's orbitals were local and its item failed
anyway on the index it screened on.  This is that count.

For every block of the grid the CPHF uses (level 3, the same
``block_loop`` and AO screen as ``fastxcfxc``) it prints the number of
orbitals whose ``max_{n,g} |X_k|`` is above ``tau`` (``grid`` columns: the
screen an implementation can apply, since ``X`` is computed per block
anyway) and, as the cruder alternative, whose coefficients on the block's AOs
reach ``tau`` (``coef``), for the canonical orbitals and for localised ones.

    python benchmarks/fxc_local_occ_probe.py --xyz benchmarks/molecules/taxol.xyz \\
        --basis def2-svp --loc chol,canon --cache $SCRATCH/w36/mo
"""
import os
import time

import numpy as np
import cupy as cp

from _common import build_mol, parser                          # noqa: E402

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--loc', default='chol,canon',
                help='localisations: canon, chol (pivoted Cholesky of '
                     'C_o C_o^T), pm, boys (pyscf.lo, on the host)')
ap.add_argument('--taus', default='1e-6,1e-8,1e-10,1e-12')
ap.add_argument('--cache', default='',
                help='prefix for cached converged orbitals')
ap.add_argument('--hist', action='store_true')
ap.add_argument('--dm', default='',
                help='a cached converged restricted density matrix (.npy, the '
                     'molecule\'s AO order) in place of an SCF: the occupied '
                     'orbitals are then its pivoted Cholesky factor, which is '
                     'an orthonormal occupied set already, and `canon` is not '
                     'available.  For the 284-atom case, whose SCF is the '
                     'expensive part of the run')
ap.add_argument('--blk', type=int, default=0,
                help='grid points per block (GPU4PySCF: 4096); the count '
                     'is only meaningful for the block size the route runs')
ap.add_argument('--scf-grids', action='store_true',
                help='count on the SCF grid (`--grids` level) instead of the '
                     'CPHF grid a Hessian actually runs the route on '
                     '(`mf.cphf_grids`, SG-1 pruned (50, 194)): 3x more '
                     'points, so each 4096-point block is spatially smaller '
                     'and the count more favourable')
ap.add_argument('--err', action='store_true',
                help='also price each tau in the currency that matters: the '
                     'error the truncation makes in a CPHF density '
                     'sum_k Y_k X_k and in the potential columns (V C)[:, k] '
                     'it leaves out, with three random right-hand sides')
a = ap.parse_args()

from gpu4pyscf import dft                                      # noqa: E402

mol = build_mol(a)
uks = mol.spin != 0
tag = (f'{os.path.basename(a.xyz)[:-4]}.{a.basis}.{a.xc}.q{a.charge}'
       f's{a.spin}')
fn = f'{a.cache}.{tag}.npz' if a.cache else ''
mf = (dft.UKS if uks else dft.RKS)(mol, xc=a.xc).density_fit()
mf.grids.level = a.grids
mf.verbose = 0
if a.dm:
    from scipy.linalg import lapack
    dm = np.load(a.dm)
    L, piv, rank, info = lapack.dpstrf(.5 * dm, lower=1, tol=1e-10)
    nocc = mol.nelectron // 2
    f = np.zeros((mol.nao, nocc))
    f[piv - 1] = np.tril(L)[:, :nocc]
    Sm = mol.intor_symmetric('int1e_ovlp')
    print(f'# --dm: rank {rank}, nocc {nocc}, |f^T S f - 1| '
          f'{abs(f.T @ Sm @ f - np.eye(nocc)).max():.1e}', flush=True)
    mo_coeff, mo_occ = f, np.full(nocc, 2.)
    mf.grids.build()
elif fn and os.path.exists(fn):
    z = np.load(fn)
    mo_coeff, mo_occ = z['c'], z['o']
    mf.grids.build()
else:
    t = time.perf_counter()
    mf.conv_tol = 1e-8
    mf.kernel()
    print(f'# SCF {time.perf_counter() - t:.1f} s, E = {mf.e_tot:.8f}, '
          f'converged {mf.converged}', flush=True)
    mo_coeff, mo_occ = cp.asnumpy(mf.mo_coeff), cp.asnumpy(mf.mo_occ)
    if fn:
        np.savez(fn, c=mo_coeff, o=mo_occ)
if not uks:
    mo_coeff, mo_occ = mo_coeff[None], mo_occ[None]
occs = [mo_coeff[s][:, mo_occ[s] > 0] for s in range(len(mo_occ))]
S = mol.intor_symmetric('int1e_ovlp')


def localise(c, how):
    if how == 'canon':
        return c
    if how == 'chol':
        from scipy.linalg import lapack
        P = c @ c.T
        L, piv, rank, info = lapack.dpstrf(P, lower=1, tol=-1)
        n = c.shape[1]
        L = np.tril(L)[:, :n]
        f = np.zeros_like(L)
        f[piv - 1] = L
        # f f^T = P, and f = C U with U = C^T S f orthogonal
        U = c.T @ S @ f
        err = abs(U @ U.T - np.eye(n)).max()
        assert err < 1e-8, err
        return c @ U
    from pyscf import lo
    t = time.perf_counter()
    if how == 'pm':
        r = lo.PM(mol, c).kernel()
    else:
        r = lo.Boys(mol, c).kernel()
    print(f'# {how}: {time.perf_counter() - t:.1f} s', flush=True)
    return r


from gpu4pyscf.dft import numint as NI                         # noqa: E402
if a.blk:
    NI.MIN_BLK_SIZE = a.blk
grids = mf.grids
if not a.scf_grids:
    grids = mf.cphf_grids
    grids.build()
ni = mf._numint
ni.build(mol, grids.coords)
opt = ni.gdftopt
taus = [float(x) for x in a.taus.split(',')]
print(f'# {tag}  natm {mol.natm}  nao {mol.nao}  nocc '
      f'{"/".join(str(c.shape[1]) for c in occs)}  ngrids '
      f'{grids.weights.size} ({"SCF" if a.scf_grids else "CPHF"} grid)',
      flush=True)
for how in a.loc.split(','):
    for sp, c in enumerate(occs):
        nocc = c.shape[1]
        cl = localise(c, how)
        cs = opt.sort_orbitals(cp.asarray(cl), axis=[0])
        cnt_g = {t: [] for t in taus}
        cnt_c = {t: [] for t in taus}
        nsubs, ngs = [], []
        rng = np.random.default_rng(1)
        if a.err:
            # M = C mo1 for 3 random mo1, as `fxc_occ_probe.py` makes them
            nmo = mo_coeff.shape[-1]
            mo1 = rng.standard_normal((3, nmo, nocc)) * 1e-2
            if a.dm:        # no virtual orbitals: a random AO-basis M
                mo_coeff = np.eye(mol.nao)[None]
                nmo = mol.nao
                mo1 = rng.standard_normal((3, nmo, nocc)) * 1e-2
            if how != 'canon' and not a.dm:
                U = c.T @ S @ cl
                mo1 = mo1 @ U
            Ms = opt.sort_orbitals(cp.asarray(
                np.einsum('up,npk->nuk', mo_coeff[sp], mo1)), axis=[1])
            e_rho = {t: 0. for t in taus}
            e_pot = {t: 0. for t in taus}
            m_rho = m_pot = 0.
        for ao, idx, w, _ in ni.block_loop(opt._sorted_mol, grids,
                                           mol.nao, 1, max_memory=None,
                                           strict_grid_order=True):
            if len(idx) == 0:
                continue
            csub = cs[idx]
            X = cp.einsum('uk,nug->nkg', csub, ao)
            xm = abs(X).max(axis=(0, 2))
            cm = abs(csub).max(axis=0)
            for t in taus:
                cnt_g[t].append(int((xm > t).sum()))
                cnt_c[t].append(int((cm > t).sum()))
            nsubs.append(len(idx))
            if a.err:
                Y0 = cp.einsum('nuk,ug->nkg', Ms[:, idx], ao[0])
                prod = Y0 * X[0][None]                      # (3, nocc, ng)
                rho = prod.sum(axis=1)
                m_rho = max(m_rho, float(abs(rho).max()))
                # a potential column (V C)[u, k] = sum_g ao_u wv X_k with a
                # weight of the grid's own size
                pot = cp.einsum('ug,kg->uk', ao[0] * w, X[0])
                m_pot = max(m_pot, float(abs(pot).max()))
                for t in taus:
                    drop = xm <= t
                    if bool(drop.any()):
                        e_rho[t] = max(e_rho[t], float(abs(
                            prod[:, drop].sum(axis=1)).max()))
                        e_pot[t] = max(e_pot[t], float(abs(
                            pot[:, drop]).max()))
            ngs.append(w.size)
        nsub = np.array(nsubs, float)
        ng = np.array(ngs, float)
        W = ng.sum()
        wm = lambda v: (np.asarray(v, float) * ng).sum() / W     # noqa: E731
        head = (f'{how:5s} spin {sp}  nocc {nocc}  blocks {len(nsub)}  '
                f'nsub mean {wm(nsub):.0f} max {nsub.max():.0f}  '
                f'nsub/nocc {wm(nsub) / nocc:.2f}')
        print('# ' + head)
        print('| tau | local nocc (grid-weighted mean / max) | coef screen '
              '| nsub/nloc | occ-route flops, local / global |')
        for t in taus:
            nl = np.array(cnt_g[t], float)
            ncf = np.array(cnt_c[t], float)
            # per set and point: 8 nloc against 8 nocc (X shared, ignored)
            fl = (nl * ng).sum() / (nocc * W)
            print(f'| {t:.0e} | {wm(nl):.1f} / {nl.max():.0f} | '
                  f'{wm(ncf):.1f} / {ncf.max():.0f} | '
                  f'{wm(nsub) / max(wm(nl), 1):.2f} | {fl:.3f} |', flush=True)
        if a.err:
            print('| tau | max density error / max density | max dropped '
                  'potential column / max column |')
            for t in taus:
                print(f'| {t:.0e} | {e_rho[t] / m_rho:.1e} | '
                      f'{e_pot[t] / m_pot:.1e} |')
        if a.hist:
            nl = np.array(cnt_g[taus[len(taus) // 2]])
            h, e = np.histogram(nl / nocc, bins=10, range=(0, 1),
                                weights=ng / W)
            print('# hist of nloc/nocc (grid weight):',
                  ' '.join(f'{x:.2f}' for x in h))
