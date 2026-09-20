"""What a robust pair-atomic exchange build would cost.  W6's next question.

``docs/DF_WORK_LIST.md`` **W6** leaves one thing above every other: *"robust
pair-atomic exchange does not factorise the way ``vk = rhok rhok^T`` does --
the coefficients depend on the pair, so the density cannot be contracted
first, and that identity is what the whole of ``fastdf`` rests on.  PARI-K is
a different contraction order and estimating its cost is the next piece of
work."*  This is that estimate, and it is three measurements rather than a
flop count, because two of the three are not predictable from the pair list.

**1 -- the width.**  ``electrolites/dflocal.py`` writes the robust exchange as
one contraction, ``vk = M X + (M X)^T`` with ``X[n,Q,l] = sum_s c[ls,Q]
D[n,s]``.  ``X``'s second AO index runs only over the AOs that overlap the
auxiliary function's own atom -- call it ``nlambda`` -- where global DF's
equivalent intermediate runs over the *occupied* orbitals, twice.  So the
whole comparison is

    global DF / PARI  =  2 nocc / nlambda

and ``nlambda`` is a property of the screened pair list alone.  This is the
default output, and it needs no integrals and no GPU.

**2 -- how much of that width is real** (``--lam``).  A ``lambda`` at the edge
of the screening carries a fitting coefficient of 1e-8, and dropping it is an
error in the exchange matrix, not in the fit.  ``--lam`` sweeps
``DFLOCAL_LAMBDA_TOL``, reports what each threshold leaves of ``nlambda``, and
**measures the exchange matrix it produces** against the unscreened one and
against global DF's own distance from exact.

**3 -- what it would have to store** (``--residual``, ``--residual-k``).  The robust form needs
the exact three-centre integrals ``(mn|Q)`` for *every* pair against *every*
auxiliary function, not only the domain ones, so the 110x storage reduction
``docs/README_dflocal.md`` §1 measures for the coefficients does not by itself
carry to the build.  What can carry is the split ``B = R + T``: ``T = c J`` is
rebuilt from the coefficients in a GEMM and ``R = B - T`` is the *residual's*
potential, which is exactly zero on the domain and decays faster than either
half.  ``--residual`` builds ``R`` and ``B`` and measures how much of each survives
a threshold **on ``B``'s scale**, which is the scale the exchange build's
accuracy is set on; ``--residual-k`` then builds the exchange matrix from the
thresholded tensor, which is what says whether the storage the count promises
is usable.

    python benchmarks/df_local_k_probe.py                      # 1, no GPU
    python benchmarks/df_local_k_probe.py --lam --cases penicillinV:0:0
    python benchmarks/df_local_k_probe.py --lam-fit --cases PfPMT:0:-1
    python benchmarks/df_local_k_probe.py --residual --residual-k --cases taxol:0:0
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--cases', default='glycine:0:0,alanine_dipeptide:0:0,'
                                   'penicillinV:0:0,taxol:0:0,PfPMT:0:-1,'
                                   'HcgC:0:0')
ap.add_argument('--lam', action='store_true',
                help='sweep the lambda threshold and measure the exchange '
                     'matrix each one gives')
ap.add_argument('--lam-tols', default='0,1e-10,1e-8,1e-6,1e-4')
ap.add_argument('--lam-fit', action='store_true',
                help='the same sweep with the fit only -- no tensor, no '
                     'exchange matrix, and therefore no error column, but it '
                     'runs at 284 and 545 atoms where the tensor is 86 and '
                     '414 GB and a contraction cannot')
ap.add_argument('--residual', action='store_true',
                help='build R = B - cJ and measure what survives a threshold')
ap.add_argument('--res-tols', default='1e-4,1e-6,1e-8,1e-10')
ap.add_argument('--residual-k', action='store_true',
                help='with --residual, also build the exchange matrix from '
                     'the thresholded residual, which is what says whether '
                     'the storage the previous column counts is usable')
ap.add_argument('--domain', default='pair')
a = ap.parse_args()

from pyscf import gto                                        # noqa: E402
from pyscf.df import addons                                  # noqa: E402
from electrolites._dfblock import pair_indices, ao2atom      # noqa: E402


def load(case):
    name, spin, charge = (case.split(':') + ['0', '0'])[:3]
    mol = gto.M(atom=f'benchmarks/molecules/{name}.xyz', basis=a.basis,
                spin=int(spin), charge=int(charge), verbose=0)
    return name, mol


def widths(mol, rows, cols, aux):
    """``nlambda`` per atom: how many AOs pair with an AO of that atom."""
    a2a = ao2atom(mol)
    xsl = aux.aoslice_by_atom()
    ar, ac = a2a[rows], a2a[cols]
    nlam = np.empty(mol.natm)
    naux_A = np.empty(mol.natm)
    for A in range(mol.natm):
        m = (ar == A) | (ac == A)
        nlam[A] = len(np.union1d(rows[m], cols[m]))
        naux_A[A] = xsl[A, 3] - xsl[A, 2]
    return nlam, naux_A


# --------------------------------------------------------------------------
# 1 -- the width, and what it makes the contraction cost
# --------------------------------------------------------------------------
print(f'{"molecule":20s} {"natm":>5s} {"nao":>5s} {"naux":>6s} {"nocc":>5s} '
      f'{"nlambda":>8s} {"nl/nao":>7s} {"globalK/TF":>10s} {"pariK/TF":>9s} '
      f'{"ratio":>6s}')
rows_out = []
for case in a.cases.split(','):
    name, mol = load(case)
    r, c, nao, naux = pair_indices(mol, a.auxbasis)
    aux = addons.make_auxmol(mol, a.auxbasis)
    nlam, naux_A = widths(mol, r, c, aux)
    w = float((naux_A * nlam).sum() / naux_A.sum())      # aux-weighted
    nocc = mol.nelectron // 2
    # global DF: `rhok = cderi . mo` then `vk = rhok rhok^T`, both
    # `2 nao^2 nocc naux`; `fastdf`'s dsyrk halves the second above 800 AOs
    g = 4 * nao**2 * nocc * naux
    # PARI: one pass, `2 nao^2 naux nlambda`, plus the X intermediate
    p = 2 * nao**2 * naux * w
    print(f'{name:20s} {mol.natm:5d} {nao:5d} {naux:6d} {nocc:5d} '
          f'{w:8.0f} {w/nao:7.2f} {g/1e12:10.1f} {p/1e12:9.1f} '
          f'{g/p:5.2f}x')
    rows_out.append(dict(name=name, natm=int(mol.natm), nao=int(nao),
                         naux=int(naux), nocc=int(nocc), npair=int(len(r)),
                         nlambda=w, flops_global=float(g), flops_pari=float(p)))

n = np.log([x['natm'] for x in rows_out])
for key, lab in (('nlambda', 'nlambda'), ('nocc', 'nocc')):
    e = np.polyfit(n, np.log([x[key] for x in rows_out]), 1)[0]
    print(f'{lab:>12s} ~ N^{e:.2f}')
e = np.polyfit(n, np.log([x['flops_global'] / x['flops_pari']
                          for x in rows_out]), 1)[0]
print(f'{"ratio":>12s} ~ N^{e:.2f}   '
      f'(PARI-K passes global DF-K where 2 nocc > nlambda)')

out = dict(basis=a.basis, domain=a.domain, rows=rows_out)

# --------------------------------------------------------------------------
# 2 -- what a threshold leaves of it, and what that costs the exchange matrix
# --------------------------------------------------------------------------
if a.lam_fit:
    import cupy as cp
    import electrolites.dflocal as dl

    print()
    print(f'{"molecule":14s} {"tol":>8s} {"nlambda":>8s} {"of full":>8s} '
          f'{"2nocc/nl":>9s} {"vs globalK":>11s}')
    fit_rows = []
    for case in a.cases.split(','):
        name, mol = load(case)
        ref = [x for x in rows_out if x['name'] == name][0]
        # the fit does not depend on the threshold -- only the lambda lists
        # do -- so it is done once and the sweep re-lays-out the blocks
        o = dl.PARIDF(mol, a.auxbasis, domain=a.domain)
        o.build(fit_only=True)
        for tol in [float(t) for t in a.lam_tols.split(',')]:
            o.lambda_tol = tol
            o.cA = o.cAt = None
            cp.get_default_memory_pool().free_all_blocks()
            o._build_domain_blocks()
            nl = dl.stats()['nlambda']
            nocc, nao = ref['nocc'], ref['nao']
            # the AO order, then the MO order: `2 nocc / nlambda` against
            # `2 nao nocc / (nao nocc + nocc nlambda)` -- whichever is larger
            r_ao = 2 * nocc / nl
            r_mo = 2 * nao / (nao + nl)
            print(f'{name:14s} {tol:8.0e} {nl:8.0f} {nl/ref["nlambda"]:8.2f} '
                  f'{r_ao:9.2f} {max(r_ao, r_mo):10.2f}x')
            fit_rows.append(dict(name=name, tol=tol, nlambda=nl,
                                 ratio_ao=r_ao, ratio_mo=r_mo))
        del o
        cp.get_default_memory_pool().free_all_blocks()
    out['lam_fit'] = fit_rows

if a.lam:
    import cupy as cp
    from gpu4pyscf import scf as gscf
    import electrolites.dflocal as dl

    print()
    print(f'{"molecule":14s} {"tol":>8s} {"nlambda":>8s} {"of full":>8s} '
          f'{"relK vs PARI":>13s} {"relK vs globalDF":>17s}')
    lam_rows = []
    for case in a.cases.split(','):
        name, mol = load(case)
        mf = gscf.RHF(mol).density_fit()
        mf.conv_tol = 1e-10
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        vk_df = mf.with_df.get_jk(dm, with_j=False)[1]
        scale = float(cp.abs(vk_df).max())
        ref = None
        for tol in [float(t) for t in a.lam_tols.split(',')]:
            o = dl.PARIDF(mol, a.auxbasis, domain=a.domain, lambda_tol=tol)
            o.build()
            vk = o.get_jk(dm, with_j=False)[1]
            if ref is None:
                ref = vk
            d_self = float(cp.abs(vk - ref).max()) / scale
            d_df = float(cp.abs(vk - vk_df).max()) / scale
            nl = dl.stats()['nlambda']
            full = [x['nlambda'] for x in rows_out if x['name'] == name][0]
            print(f'{name:14s} {tol:8.0e} {nl:8.0f} {nl/full:8.2f} '
                  f'{d_self:13.2e} {d_df:17.2e}')
            lam_rows.append(dict(name=name, tol=tol, nlambda=nl,
                                 rel_self=d_self, rel_df=d_df))
            del o, vk
            cp.get_default_memory_pool().free_all_blocks()
    out['lam'] = lam_rows

# --------------------------------------------------------------------------
# 3 -- the residual, and whether it can be stored instead of the integrals
# --------------------------------------------------------------------------
if a.residual:
    import cupy as cp
    import electrolites.dflocal as dl

    def _tensor(x):
        """The folded tensor as `PARIDF` wants it: C-contiguous
        ``(naux, npair)``, on the device if it fits."""
        try:
            return cp.asarray(x)
        except cp.cuda.memory.OutOfMemoryError:
            return x

    print()
    print(f'{"molecule":14s} {"tol":>8s} {"R kept":>8s} {"B kept":>8s} '
          f'{"R+c/GB":>8s} {"global/GB":>10s} {"ratio":>7s} {"relK":>9s}')
    res_rows = []
    for case in a.cases.split(','):
        name, mol = load(case)
        o = dl.PARIDF(mol, a.auxbasis, domain=a.domain)
        o.build(fold=1.0)                     # M holds R = B - cJ
        R = o.M if isinstance(o.M, np.ndarray) else cp.asnumpy(o.M)
        c_gb = o.c.size * 8 / 1e9
        gb = R.size * 8 / 1e9
        del o
        cp.get_default_memory_pool().free_all_blocks()
        # the control: the same threshold on the raw integrals.  Without it
        # "34 % of R survives 1e-6" says nothing -- the question is whether
        # the residual decays faster than what it is the residual of
        o = dl.PARIDF(mol, a.auxbasis, domain=a.domain)
        o.build(fold=0.0)                     # M holds B
        B = o.M if isinstance(o.M, np.ndarray) else cp.asnumpy(o.M)
        sR, sB = np.abs(R).max(), np.abs(B).max()
        print(f'#   {name}: max|R| / max|B| = {sR/sB:.2e}')
        kref = dm = None
        if a.residual_k:
            # the exchange matrix the unthresholded tensor gives, to compare
            # against.  `M = B - T/2 = (B + R)/2`, since `T = B - R`
            from gpu4pyscf import scf as gscf
            mf = gscf.RHF(mol).density_fit()
            mf.conv_tol = 1e-10
            mf.kernel()
            dm = cp.asarray(mf.make_rdm1())
            kdf = mf.with_df.get_jk(dm, with_j=False)[1]
            sK = float(cp.abs(kdf).max())
            del mf
            cp.get_default_memory_pool().free_all_blocks()
            o.M = _tensor(np.ascontiguousarray((B + R) * .5))
            kref = o.get_jk(dm, with_j=False)[1]
            print(f'#   {name}: PARI relK vs global DF '
                  f'{float(cp.abs(kref - kdf).max()) / sK:.2e}')
        for tol in [float(t) for t in a.res_tols.split(',')]:
            # the threshold that matters is on the *integrals'* scale: what
            # the exchange build can drop is what is small against `B`, not
            # what is small against `R`
            keep = float((np.abs(R) > tol * sB).mean())
            keepB = float((np.abs(B) > tol * sB).mean())
            relk = float('nan')
            if a.residual_k:
                Rt = np.where(np.abs(R) > tol * sB, R, 0.)
                o.M = _tensor(np.ascontiguousarray((B + Rt) * .5))
                # M = (B + R)/2 with R thresholded: the part of the tensor
                # that is rebuilt from the coefficients is untouched and only
                # the stored residual is approximated
                k = o.get_jk(dm, with_j=False)[1]
                relk = float(cp.abs(k - kref).max()) / \
                    float(cp.abs(kref).max())
                del k, Rt
            print(f'{name:14s} {tol:8.0e} {keep:8.4f} {keepB:8.4f} '
                  f'{gb*keep + c_gb:8.3f} {gb:10.2f} '
                  f'{gb/max(gb*keep + c_gb, 1e-9):6.1f}x {relk:9.2e}')
            res_rows.append(dict(name=name, tol=tol, kept=keep, keptB=keepB,
                                 scale_ratio=float(sR / sB), relk=relk,
                                 global_gb=gb, local_gb=gb * keep + c_gb))
        del o, R, B
        cp.get_default_memory_pool().free_all_blocks()
    out['residual'] = res_rows

print('RESULT ' + json.dumps(out))
