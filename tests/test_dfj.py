"""Correctness of the repaired ``only_dfj`` (RIJONX) branch.

`fastdfj` repairs the branch of ``df/df_jk.py:_DFHF.get_jk`` that GPU4PySCF
1.8.1 cannot run at all -- it passes ``mol`` where ``DF.get_jk`` expects the
density, so the call dies in ``df_jk.get_j`` with ``'Mole' object has no
attribute 'shape'``.  There is therefore no upstream answer to compare
against, and "it no longer crashes" is not a correctness bar.  What is checked
here instead is that the branch computes **its definition**:

    vj = the density-fitted Coulomb matrix, i.e. what ``DF.get_jk`` returns
         for a J-only call -- the three-centre ``j_engine_3c2e`` contraction,
         no cderi;
    vk = the exact four-centre exchange matrix, i.e. what a *direct* SCF
         object's ``get_k`` returns for the same density.

Both references are assembled independently, from objects that never saw
``only_dfj``, and compared element-wise at a fixed density.  Then the four
``(with_j, with_k)`` combinations, ``hermi`` 0 and 1, one and two densities,
restricted and unrestricted.

Two further things, which a single tolerance would hide:

* A **pure** functional's ``only_dfj`` energy must equal its full-DF energy to
  round-off -- there is no exchange, so the flag can change nothing.  It is
  the cheapest test that the branch is not quietly altering the Coulomb half.
* For a hybrid it is a **different approximation**, not a different route to
  the same number.  The last section measures the gap against both limits
  rather than asserting one: RIJONX against full DF, and each against the
  direct SCF that has no fitting error at all.

And the driver change, which is the larger half of what the module is worth:
an ``only_dfj`` object is handed to its own class's ``get_veff``, so the
exchange is built from ``D - D_last``.  That must converge to the same energy
as building it from ``D``, which is what ``FASTDFJ_NO_INCREMENTAL=1`` does,
and to within one iteration of the same cycle count -- an incremental build
screens on the difference, so the Fock matrix differs by of order
``direct_scf_tol`` and a DIIS trajectory can cross the convergence threshold
one iteration early or late.

    python tests/test_dfj.py
    python tests/test_dfj.py --basis def2-tzvpp

Needs one GPU.
"""
import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--tol', type=float, default=1e-12,
                help='maximum difference relative to the largest element of '
                     'the reference matrix')
ap.add_argument('--no-scf', action='store_true', help='skip the end-to-end part')
ap.add_argument('--verbose', action='store_true')
a = ap.parse_args()

import electrolites                                           # noqa: E402
# the arrangement `patch_all()` produces, not `fastdfj` alone: `fastxcnlc`
# patches the same `_DFHF.get_veff` this module wraps, and a VV10 functional
# is the only thing that reaches its fused grid pass -- so the driver change
# of §1.2 has to be checked with that wrapper underneath it, which is what a
# user will have.
electrolites.patch_all(only=['fastxc', 'fastnlc', 'fastxcnlc', 'fastdfj'])

import cupy as cp                                             # noqa: E402
from pyscf import gto                                         # noqa: E402
from gpu4pyscf import dft, scf                                # noqa: E402
from gpu4pyscf.df import df_jk as DFJK                        # noqa: E402
from gpu4pyscf.scf import hf as GHF_MOD                        # noqa: E402
from electrolites import fastdfj                              # noqa: E402

ETHANOL = '''
C  -0.748  -0.015   0.024
C   0.558   0.420  -0.679
O   1.598  -0.513  -0.421
H  -1.293  -0.849  -0.435
H  -1.393   0.878   0.086
H  -0.581  -0.323   1.061
H   0.384   0.442  -1.766
H   0.874   1.415  -0.352
H   2.437  -0.166  -0.775
'''
CH3 = 'C 0 0 0; H 0 1.079 0; H 0.934 -0.539 0; H -0.934 -0.539 0'


def _err(got, ref):
    got = cp.asnumpy(got)
    ref = cp.asnumpy(ref)
    scale = max(float(np.abs(ref).max()), 1e-300)
    return float(np.abs(got - ref).max()) / scale


def _dm(mol, spin, ndm, seed=0):
    """A physical density, and ``ndm`` of them when asked.

    The extra ones are perturbed copies rather than random matrices: the
    Coulomb half reads the two triangles differently at ``hermi`` 0 and a
    density that is not close to idempotent would not exercise the same
    screening the SCF does.
    """
    rng = np.random.default_rng(seed)
    mf = (scf.UHF if spin else scf.RHF)(mol)
    mf.max_cycle = 1
    mf.conv_tol = 1e-4
    mf.verbose = 0
    mf.kernel()
    dm0 = cp.asarray(mf.make_rdm1())
    if ndm == 1:
        return dm0
    out = [dm0]
    for i in range(ndm - 1):
        d = cp.asarray(rng.normal(scale=1e-3, size=dm0.shape))
        out.append(dm0 + (d + d.swapaxes(-1, -2)) * .5)
    return cp.stack(out)


def matrices():
    """The two halves of the branch, against independent references."""
    fails = 0
    for name, geom, spin in (('ethanol', ETHANOL, 0), ('CH3', CH3, 1)):
        mol = gto.M(atom=geom, basis=a.basis, spin=spin, verbose=0)
        # the object under test: only_dfj, as the SCF builds it
        mf = (scf.UHF if spin else scf.RHF)(mol).density_fit(
            auxbasis=a.auxbasis, only_dfj=True)
        mf.verbose = 0
        # the two references, from objects that never saw only_dfj
        ref_df = (scf.UHF if spin else scf.RHF)(mol).density_fit(
            auxbasis=a.auxbasis)
        ref_df.verbose = 0
        ref_direct = (scf.UHF if spin else scf.RHF)(mol)
        ref_direct.verbose = 0

        for ndm in (1, 2):
            for hermi in (1, 0):
                dm = _dm(mol, spin, ndm)
                vj, vk = mf.get_jk(mol, dm, hermi)
                # DF J: a J-only call on a plain density-fitting object, which
                # `df/df.py:111` sends to `j_engine_3c2e` without a cderi.
                # The leading axes are flattened here for the same reason
                # `fastdfj` flattens them -- `df_jk.get_j` asserts
                # `1 < ndim <= 3`, and an unrestricted two-density case is
                # `(2, 2, nao, nao)`.  A full-DF run never meets that assert
                # because its SCF has built a cderi by then.
                d = cp.asarray(dm)
                vj_ref = ref_df.with_df.get_jk(
                    d.reshape(-1, d.shape[-1], d.shape[-1]), hermi,
                    with_j=True, with_k=False)[0].reshape(d.shape)
                # exact K: the direct path's own build
                vk_ref = ref_direct.get_k(mol, dm, hermi)
                ej, ek = _err(vj, vj_ref), _err(vk, vk_ref)
                ok = ej < a.tol and ek < a.tol
                fails += not ok
                print(f'{"ok  " if ok else "FAIL"} {name}/{a.basis} '
                      f'spin={spin} ndm={ndm} hermi={hermi}: '
                      f'J {ej:.1e}  K {ek:.1e}')

        # the four (with_j, with_k) combinations
        dm = _dm(mol, spin, 1)
        full_j, full_k = mf.get_jk(mol, dm, 1)
        for wj, wk in ((True, True), (True, False), (False, True),
                       (False, False)):
            vj, vk = mf.get_jk(mol, dm, 1, with_j=wj, with_k=wk)
            bad = []
            if wj and _err(vj, full_j) >= a.tol:
                bad.append(f'J {_err(vj, full_j):.1e}')
            if not wj and vj is not None:
                bad.append('J returned when not asked')
            if wk and _err(vk, full_k) >= a.tol:
                bad.append(f'K {_err(vk, full_k):.1e}')
            if not wk and vk is not None:
                bad.append('K returned when not asked')
            fails += bool(bad)
            print(f'{"ok  " if not bad else "FAIL"} {name} '
                  f'with_j={int(wj)} with_k={int(wk)}: '
                  f'{"; ".join(bad) or "as asked"}')

        # a GHF object keeps GPU4PySCF's own branch, which is already correct
        before = fastdfj.stats()['passthrough']
        gmf = scf.GHF(mol).density_fit(auxbasis=a.auxbasis, only_dfj=True)
        gmf.verbose = 0
        gdm = cp.asarray(gmf.get_init_guess(mol))
        gmf.get_jk(mol, gdm, 1)
        passed = fastdfj.stats()['passthrough'] > before
        fails += not passed
        print(f'{"ok  " if passed else "FAIL"} {name} GHF only_dfj: '
              f'{"handed to GPU4PySCF" if passed else "intercepted"}')
    return fails


def energies():
    """End to end: the flag's effect on an SCF energy, both limits measured."""
    fails = 0
    rows = []
    for name, geom, spin, xc in (('ethanol', ETHANOL, 0, 'pbe'),
                                 ('ethanol', ETHANOL, 0, 'b3lyp'),
                                 ('ethanol', ETHANOL, 0, 'wb97x'),
                                 ('CH3', CH3, 1, 'b3lyp'),
                                 ('CH3', CH3, 1, 'wb97x')):
        mol = gto.M(atom=geom, basis=a.basis, spin=spin, verbose=0)
        e = {}
        for tag, kw in (('rijonx', dict(only_dfj=True)),
                        ('fulldf', dict(only_dfj=False)), ('direct', None)):
            mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc)
            if kw is not None:
                mf = mf.density_fit(auxbasis=a.auxbasis, **kw)
            mf.conv_tol = 1e-11
            mf.grids.level = 3
            mf.verbose = 0
            e[tag] = mf.kernel()
        d_rij = e['rijonx'] - e['direct']
        d_df = e['fulldf'] - e['direct']
        rows.append((f'{name}/{xc}', e['rijonx'], d_rij, d_df))
        # a pure functional has no exchange, so the flag must change nothing
        if xc == 'pbe':
            same = abs(e['rijonx'] - e['fulldf']) < 1e-9
            fails += not same
            print(f'{"ok  " if same else "FAIL"} {name}/pbe: only_dfj changes '
                  f'a pure functional by {abs(e["rijonx"]-e["fulldf"]):.1e} Eh '
                  f'(must be 0: there is no exchange)')
        else:
            # the fitting error must stay in the milli-Hartree range; a sign
            # that the exchange is not the exact one would be orders larger
            sane = abs(d_rij) < 1e-2
            fails += not sane
            print(f'{"ok  " if sane else "FAIL"} {name}/{xc}: '
                  f'RIJONX - direct = {d_rij:+.2e} Eh')

    print(f'\n{"case":22s} {"E(RIJONX)":>18s} {"RIJONX-direct":>15s} '
          f'{"fullDF-direct":>15s}')
    for tag, e_rij, d_rij, d_df in rows:
        print(f'{tag:22s} {e_rij:18.10f} {d_rij:+15.2e} {d_df:+15.2e}')
    print('\nThe two error columns are the point of this table: RIJONX and '
          'full DF are\ndifferent approximations, and on a hybrid the fitted '
          'J and the fitted K of full\nDF cancel in part -- which RIJONX, '
          'having no fitted K, cannot do.')
    return fails


def incremental():
    """The driver change: an incremental Fock must not move the answer.

    `fastdfj` hands an ``only_dfj`` object to its own class's ``get_veff``
    rather than to ``_DFHF``'s, so the exchange is built from ``D - D_last``
    instead of from ``D`` every cycle.  That is the difference between RIJONX
    losing to the direct path over a whole SCF and beating it
    (`docs/README_fastdfj.md` §3.3) -- and it is a *screening* change, so it
    can only be checked by converging both and comparing.

    The bar is the SCF's own: the two must agree far inside ``conv_tol``, and
    they must take the same number of cycles.  A silent change in either would
    mean the incremental build is dropping something real.
    """
    fails = 0
    for name, geom, spin, xc, cls in (
            ('ethanol', ETHANOL, 0, 'b3lyp', None),
            ('ethanol', ETHANOL, 0, 'wb97x', None),
            ('CH3', CH3, 1, 'b3lyp', None),
            # a VV10 functional, which is the one thing that reaches
            # `fastxcnlc`'s fused grid pass -- and reaches it only because an
            # `only_dfj` object now drives `RKS.get_veff` rather than
            # `_DFHF.get_veff`
            ('ethanol', ETHANOL, 0, 'wb97m-v', None),
            ('ethanol', ETHANOL, 0, None, scf.RHF),
            ('CH3', CH3, 1, None, scf.UHF)):
        mol = gto.M(atom=geom, basis=a.basis, spin=spin, verbose=0)
        out = {}
        for no_incr in (False, True):
            fastdfj._NO_INCREMENTAL = no_incr
            if cls is None:
                mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc)
                mf.grids.level = 3
            else:
                mf = cls(mol)
            mf = mf.density_fit(auxbasis=a.auxbasis, only_dfj=True)
            mf.conv_tol = 1e-11
            mf.verbose = 0
            out[no_incr] = (mf.kernel(), int(mf.cycles))
        fastdfj._NO_INCREMENTAL = False
        de = abs(out[False][0] - out[True][0])
        dcyc = out[False][1] - out[True][1]
        # The energies have to agree; the *cycle counts* are allowed to differ
        # by one and sometimes do.  An incremental build screens on
        # `D - D_last`, so the Fock matrix differs from the whole-density one
        # by of order `direct_scf_tol`, and near convergence that is enough to
        # move a DIIS trajectory across the `conv_tol` line one iteration
        # earlier or later.  Measured: def2-TZVPP wB97X 10 against 11 cycles
        # and wB97M-V 12 against 11, with the two energies 1.7e-11 and 3.1e-11
        # Eh apart on a `conv_tol` of 1e-11.  A larger drift than that, or a
        # cycle count that moves by more than one, is not this effect.
        ok = de < 1e-9 and abs(dcyc) <= 1
        fails += not ok
        print(f'{"ok  " if ok else "FAIL"} {name}/{xc or cls.__name__}: '
              f'incremental - whole-density = {de:.1e} Eh, '
              f'cycles {out[False][1]} vs {out[True][1]}')
    return fails


def main():
    print(f'# gpu4pyscf {electrolites.gpu4pyscf_version()}  '
          f'patched: {electrolites.patched()}')
    fails = matrices()
    if not a.no_scf:
        fails += energies()
        fails += incremental()
    from electrolites import fastxcnlc as _fxnlc
    print(f'\nfastdfj {fastdfj.stats()}')
    st = _fxnlc.stats()
    # A RIJONX wB97M-V run has to *reach* the fused grid pass, not merely be
    # allowed to.  `fastxcnlc`'s restricted wrapper declines any object that
    # carries a `with_df`, because on a full-DF run `_DFHF.get_veff` is the
    # real entry point and fusing twice would be wrong -- and `only_dfj` is
    # named as the exception there precisely because `fastdfj` routes such an
    # object to `RKS.get_veff` on purpose.  Without that exception the VV10
    # sweep silently goes back to walking the grid twice, which is exactly the
    # class of bug `docs/README_fastdf.md` §5 is about, so it is asserted.
    ok = st['fused'] > 0
    fails += not ok
    print(f'{"ok  " if ok else "FAIL"} fastxcnlc {st}  '
          '(`fused` > 0 = the VV10 sweep a RIJONX run reaches)')
    print('FAILURES' if fails else 'all ok')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
