"""Correctness of the density-fitted J and K build.

Catalogue item **A13**.  `fastdf` replaces ``df/df_jk.py:get_jk``, the single
function every RI/DF calculation's two-electron half goes through -- the SCF,
the response solve, and a Hessian's CPHF.  It is checked here against
GPU4PySCF's own build, **element-wise on the AO-space matrices at a fixed
density**, both implementations in one process on one cderi tensor.

Three things a converged energy would not separate are checked separately.

* Both **density factorisations**.  A ``mo_coeff``-tagged density factorises
  as ``C C^T`` and the exchange matrix is then a symmetric rank-k update of
  one tensor with itself, which is the case `fastdf` rearranges; an untagged
  density goes through an eigendecomposition or an SVD and gives *two*
  factors, where the same rearrangement does not apply and a general GEMM has
  to stay.  Both reach this function and they are different code.
* ``hermi`` 0 as well as 1, because the Coulomb half reads the density's two
  triangles differently in the two cases (``df_jk.py:490``).
* Several **density counts**, because the ground state asks for one or two and
  a CPHF or Davidson solve for many, and the block that fits in memory --
  and therefore the number of passes over the cderi -- changes with it.

Then, end to end: a density-fitted SCF energy and its gradient against the
unpatched pair, which is the only check that the whole path is still the same
calculation.

    python tests/test_df.py                    # five molecule/basis cases
    python tests/test_df.py --basis def2-tzvpp  # up to f functions
    python tests/test_df.py --ablate            # each item of the module alone
"""
import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--ndm', default='1,2,4', help='comma separated density counts')
ap.add_argument('--tol', type=float, default=1e-11,
                help='maximum difference relative to the largest element of '
                     'the reference matrix')
ap.add_argument('--no-scf', action='store_true', help='skip the end-to-end pair')
ap.add_argument('--ablate', action='store_true',
                help='also check the two paths the default configuration does '
                     'not take on these molecules: the cuBLAS first '
                     'contraction, and GPU4PySCF\'s Coulomb pass')
ap.add_argument('--verbose', action='store_true')
a = ap.parse_args()

import cupy as cp                                             # noqa: E402
from pyscf import gto                                         # noqa: E402
from gpu4pyscf.df import df_jk as DFJK                        # noqa: E402
from gpu4pyscf.df.df import DF                                # noqa: E402
from gpu4pyscf.lib.cupy_helper import tag_array               # noqa: E402

_REF = DFJK.get_jk                                            # before patching

H2O = 'O 0.0 0.0 0.117; H 0.0 0.757 -0.467; H 0.0 -0.757 -0.467'
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
SI = ('Si 0.0 0.0 0.0; H 0.0 0.0 1.48; H 1.40 0.0 -0.48; '
      'H -0.70 1.21 -0.48; H -0.70 -1.21 -0.48')

CH3 = ('C 0.0 0.0 0.0; H 0.0 1.079 0.0; H 0.934 -0.539 0.0; '
       'H -0.934 -0.539 0.0')

CASES = [('H2O', H2O, 0, 0, '6-31G*'),
         ('H2O', H2O, 0, 0, 'cc-pVDZ'),
         ('ethanol', ETHANOL, 0, 0, 'def2-SVP'),
         ('SiH4', SI, 0, 0, '6-31G*'),
         # an open-shell reference reaches this function with two densities in
         # one call and one `mo_occ` row per spin, which is a different
         # factorisation branch from the closed-shell one
         ('CH3', CH3, 0, 1, 'def2-SVP'),
         ('ethanol', ETHANOL, 0, 0, 'def2-TZVP')]
if a.basis:
    seen, uniq = set(), []
    for n, g, c, s, _ in CASES:
        if n not in seen:
            seen.add(n)
            uniq.append((n, g, c, s, a.basis))
    CASES = uniq

NDMS = [int(x) for x in a.ndm.split(',')]


def _mo_density(mol, n_dm, rng):
    """``n_dm`` densities carrying ``mo_coeff``/``mo_occ``: the mode `fastdf`
    rearranges.  One of them is an actual SCF-shaped density and the rest are
    perturbations of it, so a wrong per-density stride cannot hide."""
    nao = mol.nao
    nocc = max(1, mol.nelectron // 2)
    mo = rng.standard_normal((n_dm, nao, nocc + 2))
    occ = np.zeros((n_dm, nocc + 2))
    occ[:, :nocc] = 2.
    dm = np.einsum('nui,ni,nvi->nuv', mo, occ, mo)
    if n_dm == 1:
        dm, mo, occ = dm[0], mo[0], occ[0]
    return tag_array(cp.asarray(dm), mo_coeff=cp.asarray(mo),
                     mo_occ=cp.asarray(occ))


def _plain_density(mol, n_dm, hermi, rng):
    """Untagged densities: two factors from an eigendecomposition or an SVD."""
    nao = mol.nao
    nocc = max(1, mol.nelectron // 2)
    c = rng.standard_normal((n_dm, nao, nocc))
    dm = np.einsum('nui,nvi->nuv', c, c) * (2. / nao)
    if hermi == 0:
        dm = dm + rng.standard_normal((n_dm, nao, nao)) * 1e-3
    if n_dm == 1:
        dm = dm[0]
    return cp.asarray(dm)


def _cmp(got, ref):
    scale = float(cp.abs(ref).max()) or 1.
    return float(cp.abs(cp.asarray(got) - cp.asarray(ref)).max()) / scale


#: Which internal configurations every case is checked in.  The molecules here
#: are all below `SYRK_MIN_NAO`, so the default configuration never takes the
#: rank-k branch on them and forcing it is the only way this suite covers it.
CONFIGS = [{'_STEP2': 'gemm'}, {'_STEP2': 'syrk'}]
DEFAULTS = {}


def main():
    import electrolites
    electrolites.patch('fastdf')
    from electrolites import fastdf
    for cfg in CONFIGS:
        DEFAULTS.update({k: getattr(fastdf, k) for k in cfg})
    if a.ablate:
        CONFIGS.extend([{'_STEP1': 'bgemm'}, {'_JPASS': 'tensor'}])
        for cfg in CONFIGS:
            DEFAULTS.update({k: getattr(fastdf, k) for k in cfg})
    print(f'# gpu4pyscf {electrolites.gpu4pyscf_version()}  '
          f'configurations: {CONFIGS}', flush=True)

    rng = np.random.default_rng(20260917)
    worst = {'J': 0., 'K': 0.}
    fails = 0
    for name, geom, charge, spin, basis in CASES:
        mol = gto.M(atom=geom, basis=basis, charge=charge, spin=spin,
                    verbose=0)
        dfobj = DF(mol, auxbasis=a.auxbasis).build()
        lmax = int(mol._bas[:, 1].max())
        case_fails = 0
        for hermi in (1, 0):
            for kind in ('mo', 'plain'):
                for n_dm in NDMS:
                    if kind == 'mo':
                        if hermi == 0:
                            continue          # an mo_coeff density is hermitian
                        dms = _mo_density(mol, n_dm, rng)
                    else:
                        dms = _plain_density(mol, n_dm, hermi, rng)
                    rj, rk = _REF(dfobj, dms, hermi, True, True)
                    # A Coulomb-only call on an existing cderi is its own
                    # branch: nothing is unpacked and no exchange tensor is
                    # formed, so only the Coulomb pass and the residency
                    # apply to it.
                    relj_only = _cmp(
                        fastdf.get_jk(dfobj, dms, hermi, True, False)[0], rj)
                    worst['J'] = max(worst['J'], relj_only)
                    if relj_only > a.tol:
                        print(f'FAIL {name:8s} {basis:10s} {kind:5s} '
                              f'hermi={hermi} n_dm={n_dm:2d} J-only '
                              f'{relj_only:.2e}', flush=True)
                        fails += 1
                        case_fails += 1
                    for cfg in CONFIGS:
                        for attr, val in cfg.items():
                            setattr(fastdf, attr, val)
                        gj, gk = fastdf.get_jk(dfobj, dms, hermi, True, True)
                        for attr in cfg:
                            setattr(fastdf, attr, DEFAULTS[attr])
                        relj, relk = _cmp(gj, rj), _cmp(gk, rk)
                        ok = relj <= a.tol and relk <= a.tol
                        worst['J'] = max(worst['J'], relj)
                        worst['K'] = max(worst['K'], relk)
                        fails += not ok
                        case_fails += not ok
                        if a.verbose or not ok:
                            tag = ','.join(f'{k[1:].lower()}={v}'
                                           for k, v in cfg.items()) or 'default'
                            print(f'{"ok  " if ok else "FAIL"} {name:8s} '
                                  f'{basis:10s} lmax={lmax} {kind:5s} '
                                  f'hermi={hermi} n_dm={n_dm:2d} {tag:22s} '
                                  f'J {relj:.2e}  K {relk:.2e}', flush=True)
        print(f'{name:8s} {basis:10s} lmax={lmax}: '
              f'{"ok" if not case_fails else "FAILURES"}', flush=True)

    print(f'\nworst relative difference: J {worst["J"]:.3e}, '
          f'K {worst["K"]:.3e}  (tolerance {a.tol:.0e})')
    print(f'fastdf stats: {fastdf.stats()}')

    fails += _host_resident(fastdf)
    fails += _block_partition(fastdf)
    if not a.no_scf:
        fails += _scf_pair()
    if fails:
        print(f'{fails} case(s) FAILED')
    return 1 if fails else 0


def _host_resident(fastdf):
    """The same build with the cderi on the host, which is a different branch.

    `_cholesky_eri` stores the tensor transposed when it stays on the device
    and untransposed when it does not, so the packed block arrives with the
    *auxiliary* index contiguous in one case and the *pair* index in the
    other.  The Coulomb pass and the block iterator both branch on that, and
    nothing in the cases above reaches the second branch: on these molecules
    the tensor always fits.
    """
    mol = gto.M(atom=ETHANOL, basis='def2-TZVP', verbose=0)
    rng = np.random.default_rng(11)
    fails = 0
    for n_dm in (1, 2, 3):
        dms = _mo_density(mol, n_dm, rng)
        for on_gpu in (True, False):
            dfobj = DF(mol)
            dfobj.use_gpu_memory = on_gpu
            dfobj.build()
            rj, rk = _REF(dfobj, dms, 1, True, True)
            gj, gk = fastdf.get_jk(dfobj, dms, 1, True, True)
            relj, relk = _cmp(gj, rj), _cmp(gk, rk)
            ok = relj <= a.tol and relk <= a.tol
            fails += not ok
            if a.verbose or not ok:
                print(f'{"ok  " if ok else "FAIL"} cderi on '
                      f'{"device" if on_gpu else "host  "} n_dm={n_dm}  '
                      f'J {relj:.2e}  K {relk:.2e}', flush=True)
    print(f'{"ok" if not fails else "FAILURES"}  cderi on the host and on the '
          f'device, both packed layouts')
    return fails


def _block_partition(fastdf):
    """Item 3's mixed resident/streamed iterator is a partition of the tensor.

    Nothing numerical would catch a block boundary that drops or repeats a row
    -- the exchange matrix would simply be wrong by an amount no tolerance
    distinguishes from a bad kernel -- so the iterator is checked directly,
    at every split of resident against streamed and every block size.
    """
    import cupy as cp
    rng = np.random.default_rng(12)
    host = np.ascontiguousarray(rng.standard_normal((37, 11)))
    fails = 0
    for rows in (0, 1, 8, 13, 37):
        resident = cp.asarray(host[:rows]) if rows else None
        for blk in (1, 4, 8, 16, 64):
            got = np.vstack([cp.asnumpy(b) for b, nL in
                             fastdf._blocks(host, blk, False, resident)])
            if got.shape != host.shape or np.abs(got - host).max() != 0:
                print(f'FAIL block partition rows={rows} blk={blk}')
                fails += 1
    print(f'{"ok" if not fails else "FAILURES"}  the resident/streamed block '
          f'iterator is a partition, at every split and block size')
    return fails


def _scf_pair():
    """A whole DF-B3LYP energy and gradient, patched against unpatched.

    Both run in this process on separate ``DF`` objects; the patch is already
    global by the time this runs, so the reference is taken by handing the
    build back with ``FASTDF_OFF`` set on the module object.
    """
    from gpu4pyscf import dft
    from electrolites import fastdf
    # the density-fitted `get_veff` binding this round added, which a VV10
    # functional is the only thing here that reaches
    import electrolites
    electrolites.patch('fastxcnlc', 'fastnlc', 'fastxc')
    from electrolites import fastxcnlc as _fxnlc
    fails = 0
    for name, geom, spin, xc in (('ethanol', ETHANOL, 0, 'b3lyp'),
                                 ('CH3', CH3, 1, 'b3lyp'),
                                 # a VV10 functional reaches `_DFHF.get_veff`,
                                 # which `fastxcnlc` only patches since this
                                 # round -- and which is 77-80 % of such a run
                                 ('ethanol', ETHANOL, 0, 'wb97m-v'),
                                 ('CH3', CH3, 1, 'wb97m-v')):
        mol = gto.M(atom=geom, basis='def2-svp', spin=spin, verbose=0)
        out = {}
        for label in ('stock', 'fastdf'):
            fastdf._OFF = (label == 'stock')
            _fxnlc._OFF = (label == 'stock')
            mf = (dft.UKS if spin else dft.RKS)(mol, xc=xc).density_fit()
            mf.conv_tol = 1e-11
            mf.grids.level = 3
            mf.verbose = 0
            e = mf.kernel()
            de = mf.nuc_grad_method().kernel()
            out[label] = (e, np.asarray(de))
        fastdf._OFF = False
        _fxnlc._OFF = False
        de_err = float(np.abs(out['fastdf'][1] - out['stock'][1]).max())
        e_err = abs(out['fastdf'][0] - out['stock'][0])
        ok = e_err < 1e-9 and de_err < 1e-8
        print(f'{"ok  " if ok else "FAIL"} '
              f'DF-{xc}/def2-svp {name} (spin={spin}): '
              f'dE = {e_err:.2e} Eh, max |d(grad)| = {de_err:.2e}'
              f'   fastxcnlc {_fxnlc.stats()}')
        fails += not ok
    return fails


if __name__ == '__main__':
    sys.exit(main())
