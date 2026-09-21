"""Correctness of the single-precision cderi, which is a different claim.

``docs/DF_WORK_LIST.md`` **W7**.  Every other suite in this repository checks
that a patched build computes what an unpatched one does, to 1e-11 or better.
This one cannot: storing the three-index tensor as ``float32`` *changes the
answer*, and the whole content of the item is how much.  So the suite is in
two halves.

**Exact, and not negotiable.**  Nothing about the mode may change while it is
off -- the tensor is ``float64``, in upstream's layout, and element for
element what the double build produced.  With it on, the pieces that are
bookkeeping rather than arithmetic stay exact: the widened block iterator is
a partition of the tensor, the widened block is the stored value widened and
nothing else, and the patched ``DF.loop`` hands a double tensor to upstream's
own generator unchanged.

**Approximate, and measured.**  J, K, the total energy, the gradient and the
Hessian against the *same calculation* in double precision, on both storage
locations, both density factorisations, ``hermi`` 0 and 1 and several density
counts.  The bars below are not tolerances in the usual sense -- they are the
claim ``docs/README_fastdffp32.md`` makes, and a failure means the error is
bigger than the write-up says, not that something is broken.

One thing here is a correctness check in the ordinary sense and is the reason
the mode is not silent: a narrow tensor has **no fall-through**.
``fill_symmetric`` takes a ``void*`` and is compiled for ``double``, so
GPU4PySCF's own ``get_jk`` handed one would read past the end of it and
return a plausible wrong answer.  Every decline in `fastdf` raises instead,
and that is asserted.

    python tests/test_dffp32.py
    python tests/test_dffp32.py --basis def2-tzvpp
    python tests/test_dffp32.py --no-scf        # the matrices only
"""
import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='')
ap.add_argument('--ndm', default='1,2,4', help='comma separated density counts')
ap.add_argument('--tol-matrix', type=float, default=3e-7,
                help='bar on |dJ| and |dK| relative to the largest element '
                     'of the double-precision matrix')
ap.add_argument('--tol-exact', type=float, default=0.,
                help='bar on the pieces that must not change at all')
ap.add_argument('--tol-energy', type=float, default=1e-6,
                help='bar on the total energy, in Eh')
ap.add_argument('--tol-grad', type=float, default=1e-6,
                help='bar on the largest gradient component, in Eh/bohr')
ap.add_argument('--no-scf', action='store_true', help='skip the end-to-end pair')
ap.add_argument('--verbose', action='store_true')
a = ap.parse_args()

import cupy as cp                                             # noqa: E402
from pyscf import gto                                         # noqa: E402
from gpu4pyscf.df.df import DF                                # noqa: E402
from gpu4pyscf.lib.cupy_helper import tag_array               # noqa: E402

import electrolites                                           # noqa: E402
electrolites.patch('fastdf', 'fastdfcderi', 'fastdffp32')
from electrolites import fastdf, fastdffp32                   # noqa: E402
from electrolites import _dffp32 as FP32                      # noqa: E402

H2O = 'O 0.0 0.0 0.117; H 0.0 0.757 -0.467; H 0.0 -0.757 -0.467'
ETHANOL = '''
C  -0.748  -0.015   0.024
C   0.558   0.420  -0.679
O   1.598  -0.513  -0.421
H  -1.293  -0.849  -0.435
H  -1.393   0.878   0.086
H   0.384   0.442  -1.766
H  -0.581  -0.323   1.061
H   0.874   1.415  -0.352
H   2.437  -0.166  -0.775
'''
CH3 = ('C 0.0 0.0 0.0; H 0.0 1.079 0.0; H 0.934 -0.539 0.0; '
       'H -0.934 -0.539 0.0')

CASES = [('H2O', H2O, 0, 0, '6-31G*'),
         ('ethanol', ETHANOL, 0, 0, 'def2-SVP'),
         # an open-shell reference reaches `get_jk` with two densities in one
         # call, which is the other factorisation branch
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


def _build(mol, fp32, on_gpu=True):
    """One cderi, in the precision asked for.  The switch is a module
    attribute rather than the environment so that both precisions can be
    built in one process and compared on the same integrals."""
    FP32._FP32 = fp32
    try:
        dfobj = DF(mol)
        dfobj.use_gpu_memory = on_gpu
        return dfobj.build()
    finally:
        FP32._FP32 = False


def _host(x):
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else np.asarray(x)


def _cmp(got, ref):
    scale = float(cp.abs(cp.asarray(ref)).max()) or 1.
    return float(cp.abs(cp.asarray(got) - cp.asarray(ref)).max()) / scale


def _mo_density(mol, n_dm, rng):
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
    nao = mol.nao
    nocc = max(1, mol.nelectron // 2)
    c = rng.standard_normal((n_dm, nao, nocc))
    dm = np.einsum('nui,nvi->nuv', c, c) * (2. / nao)
    if hermi == 0:
        dm = dm + rng.standard_normal((n_dm, nao, nao)) * 1e-3
    if n_dm == 1:
        dm = dm[0]
    return cp.asarray(dm)


# --------------------------------------------------------------------------
def test_off_changes_nothing():
    """With the switch off the build is upstream's, bit for bit.

    Not "to a tolerance": the same code path with the same inputs, so the
    tensor, its dtype and its layout must be identical.  A change here is a
    default that moved, which is the one thing an opt-in mode may not do.
    """
    mol = gto.M(atom=ETHANOL, basis='def2-svp', verbose=0)
    fails = 0
    for on_gpu in (True, False):
        d1 = _build(mol, False, on_gpu)
        d2 = _build(mol, False, on_gpu)
        c1, c2 = d1._cderi[0], d2._cderi[0]
        same = (c1.dtype == np.float64 == c2.dtype
                and c1.strides == c2.strides
                and np.array_equal(_host(c1), _host(c2)))
        fails += not same
        if a.verbose or not same:
            print(f'{"ok  " if same else "FAIL"} switch off, cderi on '
                  f'{"device" if on_gpu else "host  "}: dtype {c1.dtype}, '
                  f'strides {c1.strides}')
    ok = not fails and FP32.enabled() is False
    print(f'{"ok" if ok else "FAILURES"}  the mode is off by default and '
          f'changes nothing while it is')
    return fails + (not FP32.enabled() is False)


def test_tensor():
    """The stored tensor is the double one rounded, in the stated layout."""
    fails = 0
    for name, geom, charge, spin, basis in CASES:
        mol = gto.M(atom=geom, basis=basis, charge=charge, spin=spin,
                    verbose=0)
        for on_gpu in (True, False):
            c64 = _build(mol, False, on_gpu)._cderi[0]
            c32 = _build(mol, True, on_gpu)._cderi[0]
            a64, a32 = _host(c64), _host(c32)
            # the layout claim: a narrow tensor keeps upstream's layout,
            # which is `(npairs, naux)` seen transposed on the device and
            # `(naux, npairs)` on the host.  `electrolites/_dffp32.py` has
            # the measurement that says it has to -- the widened block
            # reaches `fill_symmetric`, and the two layouts take different
            # kernels there
            layout = (a32.dtype == np.float32 and a32.shape == a64.shape
                      and [x // 4 for x in c32.strides]
                      == [x // 8 for x in c64.strides])
            err = float(np.abs(a64 - a32.astype(np.float64)).max()
                        / max(np.abs(a64).max(), 1e-300))
            # the elements are `float64` values rounded to `float32`, so the
            # error is bounded by half an ulp of the largest of them
            exact = np.array_equal(a32, a64.astype(np.float32))
            ok = layout and exact and err <= 6e-8
            fails += not ok
            if a.verbose or not ok:
                print(f'{"ok  " if ok else "FAIL"} {name:8s} {basis:10s} '
                      f'{"device" if on_gpu else "host  "}  '
                      f'rel {err:.2e}  layout={layout} rounded={exact}')
    print(f'{"ok" if not fails else "FAILURES"}  the stored tensor is the '
          f'double one rounded, in upstream\'s layout in both places')
    return fails


def test_blocks():
    """The widened block iterator is a partition, and widens and nothing else.

    `test_df.py` makes the same check on the double path, and for the same
    reason: nothing numerical would catch a block boundary that drops or
    repeats an auxiliary row.
    """
    rng = np.random.default_rng(7)
    host = np.ascontiguousarray(
        rng.standard_normal((37, 11)).astype(np.float32))
    fails = 0
    for rows in (0, 1, 8, 13, 37):
        resident = cp.asarray(host[:rows]) if rows else None
        for blk in (1, 4, 8, 16, 64):
            got = np.vstack([cp.asnumpy(b) for b, nL in
                             fastdf._blocks(host, blk, False, resident)])
            if (got.shape != host.shape or got.dtype != np.float64
                    or np.abs(got - host.astype(np.float64)).max() != 0):
                print(f'FAIL fp32 block partition rows={rows} blk={blk}')
                fails += 1
    dev = cp.asarray(host)
    for blk in (1, 5, 37, 64):
        got = np.vstack([cp.asnumpy(b) for b, nL in
                         fastdf._blocks(dev, blk, True)])
        if np.abs(got - host.astype(np.float64)).max() != 0:
            print(f'FAIL fp32 device block partition blk={blk}')
            fails += 1
    print(f'{"ok" if not fails else "FAILURES"}  the widened block iterator '
          f'is a partition and widens exactly, at every split and block size')
    return fails


def test_loop():
    """`DF.loop`: upstream's generator on a double tensor, widened on a narrow
    one, and the same blocks either way."""
    mol = gto.M(atom=ETHANOL, basis='def2-svp', verbose=0)
    fails = 0
    for on_gpu in (True, False):
        d64 = _build(mol, False, on_gpu)
        d32 = _build(mol, True, on_gpu)
        for unpack in (True, False):
            b64 = [(None if u is None else cp.asnumpy(u), cp.asnumpy(t))
                   for u, t in d64.loop(blksize=37, unpack=unpack)]
            b32 = [(None if u is None else cp.asnumpy(u), cp.asnumpy(t))
                   for u, t in d32.loop(blksize=37, unpack=unpack)]
            ok = len(b64) == len(b32)
            if ok:
                packed = max(np.abs(x[1] - y[1]).max()
                             for x, y in zip(b64, b32))
                scale = max(np.abs(x[1]).max() for x in b64)
                ok = packed / scale <= 6e-8
                if unpack:
                    unp = max(np.abs(x[0] - y[0]).max()
                              for x, y in zip(b64, b32))
                    ok = ok and unp / scale <= 6e-8
            fails += not ok
            if a.verbose or not ok:
                print(f'{"ok  " if ok else "FAIL"} loop '
                      f'{"device" if on_gpu else "host  "} unpack={unpack}: '
                      f'{len(b64)} blocks both')
    print(f'{"ok" if not fails else "FAILURES"}  `DF.loop` yields the same '
          f'blocks in both precisions, unpacked and packed')
    return fails


def test_no_fallthrough():
    """A declined call on a narrow tensor raises rather than falling through."""
    mol = gto.M(atom=H2O, basis='6-31G*', verbose=0)
    d32 = _build(mol, True)
    rng = np.random.default_rng(3)
    dms = _mo_density(mol, 1, rng)
    fastdf._OFF = True
    try:
        fastdf.get_jk(d32, dms, 1, True, True)
        raised = False
    except RuntimeError as exc:
        raised = 'FASTDF_FP32' in str(exc)
    finally:
        fastdf._OFF = False
    # and the same call on a double tensor still falls through as it should
    d64 = _build(mol, False)
    fastdf._OFF = True
    try:
        fastdf.get_jk(d64, dms, 1, True, True)
        fell = True
    except RuntimeError:
        fell = False
    finally:
        fastdf._OFF = False
    ok = raised and fell
    print(f'{"ok" if ok else "FAILURES"}  a declined call raises on a narrow '
          f'tensor ({raised}) and falls through on a double one ({fell})')
    return not ok


def test_matrices():
    """J and K against the same calculation in double precision."""
    rng = np.random.default_rng(20260920)
    worst = {'J': 0., 'K': 0.}
    fails = 0
    for name, geom, charge, spin, basis in CASES:
        mol = gto.M(atom=geom, basis=basis, charge=charge, spin=spin,
                    verbose=0)
        for on_gpu in (True, False):
            d64 = _build(mol, False, on_gpu)
            d32 = _build(mol, True, on_gpu)
            for hermi in (1, 0):
                for kind in ('mo', 'plain'):
                    for n_dm in NDMS:
                        if kind == 'mo':
                            if hermi == 0:
                                continue
                            dms = _mo_density(mol, n_dm, rng)
                        else:
                            dms = _plain_density(mol, n_dm, hermi, rng)
                        rj, rk = fastdf.get_jk(d64, dms, hermi, True, True)
                        gj, gk = fastdf.get_jk(d32, dms, hermi, True, True)
                        relj, relk = _cmp(gj, rj), _cmp(gk, rk)
                        jo = _cmp(fastdf.get_jk(d32, dms, hermi, True,
                                                False)[0], rj)
                        worst['J'] = max(worst['J'], relj, jo)
                        worst['K'] = max(worst['K'], relk)
                        ok = max(relj, relk, jo) <= a.tol_matrix
                        fails += not ok
                        if a.verbose or not ok:
                            print(f'{"ok  " if ok else "FAIL"} {name:8s} '
                                  f'{basis:10s} '
                                  f'{"device" if on_gpu else "host  "} '
                                  f'{kind:5s} hermi={hermi} n_dm={n_dm:2d}  '
                                  f'J {relj:.2e}  K {relk:.2e}  '
                                  f'J-only {jo:.2e}')
    print(f'{"ok" if not fails else "FAILURES"}  worst relative difference '
          f'against double precision: J {worst["J"]:.3e}, K {worst["K"]:.3e} '
          f'(bar {a.tol_matrix:.0e})')
    return fails


def test_scf():
    """A whole DF-B3LYP energy, gradient and Hessian in both precisions.

    Both in this process, on separate ``DF`` objects, so the only difference
    between the two is the precision the tensor was stored in.
    """
    from gpu4pyscf import dft
    electrolites.patch_all()
    fails = 0
    for name, geom, spin, basis in (('ethanol', ETHANOL, 0, 'def2-svp'),
                                    ('CH3', CH3, 1, 'def2-svp')):
        mol = gto.M(atom=geom, basis=basis, spin=spin, verbose=0)
        out = {}
        for fp32 in (False, True):
            FP32._FP32 = fp32
            try:
                mf = (dft.UKS if spin else dft.RKS)(mol, xc='b3lyp')
                mf = mf.density_fit()
                mf.conv_tol = 1e-10
                mf.grids.level = 3
                mf.verbose = 0
                e = mf.kernel()
                de = np.asarray(mf.nuc_grad_method().kernel())
                hs = np.asarray(mf.Hessian().kernel())
            finally:
                FP32._FP32 = False
            out[fp32] = (e, de, hs, int(mf.cycles), bool(mf.converged))
            del mf
            cp.get_default_memory_pool().free_all_blocks()
        de_ = abs(out[True][0] - out[False][0])
        dg_ = float(np.abs(out[True][1] - out[False][1]).max())
        dh_ = float(np.abs(out[True][2] - out[False][2]).max())
        ok = (de_ <= a.tol_energy and dg_ <= a.tol_grad and out[True][4]
              and out[True][3] <= out[False][3] + 1)
        fails += not ok
        print(f'{"ok  " if ok else "FAIL"} DF-B3LYP/{basis} {name} '
              f'(spin={spin}): dE = {de_:.2e} Eh, max |d(grad)| = {dg_:.2e}, '
              f'max |d(hess)| = {dh_:.2e}, cycles '
              f'{out[False][3]} -> {out[True][3]}')
    return fails


def main():
    print(f'# gpu4pyscf {electrolites.gpu4pyscf_version()}  '
          f'bars: matrix {a.tol_matrix:.0e}, energy {a.tol_energy:.0e} Eh',
          flush=True)
    fails = 0
    fails += test_off_changes_nothing()
    fails += test_tensor()
    fails += test_blocks()
    fails += test_loop()
    fails += test_no_fallthrough()
    fails += test_matrices()
    if not a.no_scf:
        fails += test_scf()
    print(f'\nfastdffp32 stats: {fastdffp32.stats()}')
    if fails:
        print(f'{fails} case(s) FAILED')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
