"""Correctness of the multi-density Coulomb build (catalogue item B4a).

The oracle is GPU4PySCF's own ``MD_build_j`` on the same molecule, the same
screening data and the same Hermite density -- an **element-wise, fixed-density
comparison of the AO-space J**, which is what `docs/CATALOGUE_06_STATUS.md` and
the catalogue's 6 require for anything that changes an integral kernel.  A
converged energy is not used and would not be a useful signal here: these
kernels are only reached with more than one density, i.e. inside a CPHF or
Davidson solve, never on the SCF's own path.

Three families of density are checked, because they exercise different parts of
the kernel:

* **identical copies.**  Every density is the same matrix, so every J must come
  back the same -- to round-off, not bit for bit, since each density is
  accumulated by ``atomicAdd`` at its own address and the summation order
  differs between them.  This catches a wrong ``m*dm_size`` stride, the failure
  mode where the kernel reads or writes the wrong density, which a single
  random draw can hide if the error happens to be small.
* **random symmetric perturbations of a real density.**  Different magnitudes
  per density, so the union screening (``dm_cond`` is an absmax over all of
  them) is exercised, and a density whose own contribution is tiny still has to
  come out right.
* **atom-local densities**, the shape a CPHF right-hand side actually has: each
  density has support only on the shells of one atom.  These are the sparsest
  case and the one where a mis-set ``vj_kl_cache`` offset shows up.

``n_dm`` is chosen to straddle the block sizes the kernels are generated for,
so that every path runs: n_dm = 1 (the single-density kernels, which this
round did not change), whole blocks, and a tail of fewer than four densities
that GPU4PySCF handles.

    python tests/test_j_ndm.py                    # six molecule/basis combinations
    python tests/test_j_ndm.py --basis def2-tzvp  # up to f functions
"""
import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='',
                help='restrict to one basis instead of the built-in sweep')
ap.add_argument('--ndm', default='1,2,3,4,5,8,13,16,21',
                help='comma separated density counts to check')
ap.add_argument('--tol', type=float, default=1e-12,
                help='maximum allowed difference on the AO-space J, relative '
                     'to its largest element.  Relative, because the test '
                     'densities are synthetic and not normalised the way a '
                     'converged density is, so an absolute criterion would '
                     'measure how big they happen to be')
ap.add_argument('--verbose', action='store_true')
a = ap.parse_args()

import cupy as cp                                             # noqa: E402
from pyscf import gto                                         # noqa: E402
from gpu4pyscf.scf import j_engine as JE                      # noqa: E402

H2O = '''O 0.0 0.0 0.117; H 0.0 0.757 -0.467; H 0.0 -0.757 -0.467'''
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
SI = '''Si 0.0 0.0 0.0; H 0.0 0.0 1.48; H 1.40 0.0 -0.48; H -0.70 1.21 -0.48;
        H -0.70 -1.21 -0.48'''

CASES = [('H2O', H2O, 0, '6-31G*'),
         ('H2O', H2O, 0, 'cc-pVDZ'),
         ('ethanol', ETHANOL, 0, '6-31G*'),
         ('ethanol', ETHANOL, 0, 'def2-TZVP'),
         ('SiH4', SI, 0, '6-31G*'),
         ('ethanol', ETHANOL, 0, 'cc-pVTZ')]
if a.basis:
    CASES = [(n, g, c, a.basis) for n, g, c, _ in CASES]
    seen, uniq = set(), []
    for c in CASES:
        if c[0] not in seen:
            seen.add(c[0])
            uniq.append(c)
    CASES = uniq

NDMS = [int(x) for x in a.ndm.split(',')]


def densities(mol, kind, n_dm, rng):
    """``n_dm`` symmetric AO-space densities of one shape."""
    nao = mol.nao
    base = np.zeros((nao, nao))
    # a crude but non-trivial density: a normalised random low-rank matrix,
    # so it is symmetric, dense and O(1) without needing an SCF here
    nocc = max(1, mol.nelectron // 2)
    c = rng.standard_normal((nao, nocc))
    base = 2. * c @ c.T / nao
    if kind == 'same':
        return np.repeat(base[None], n_dm, axis=0)
    if kind == 'random':
        out = np.empty((n_dm, nao, nao))
        for i in range(n_dm):
            p = rng.standard_normal((nao, nao)) * 10. ** -(i % 5)
            out[i] = base + (p + p.T) * 1e-2
        return out
    if kind == 'atomic':
        # a CPHF right-hand side's shape: support on one atom's shells only
        ao_slices = mol.aoslice_by_atom()
        out = np.zeros((n_dm, nao, nao))
        for i in range(n_dm):
            p0, p1 = ao_slices[i % mol.natm][2:]
            blk = rng.standard_normal((p1 - p0, p1 - p0))
            out[i, p0:p1, p0:p1] = blk + blk.T
        return out
    raise ValueError(kind)


def build(vhfopt, dms, patched):
    """The AO-space J for every density, through one code path."""
    import electrolites.fastj as fj
    dm = cp.asarray(dms)
    sorted_dm = vhfopt.apply_coeff_C_mat_CT(dm)
    s = fj._setup_181(vhfopt, sorted_dm)
    if not patched:
        s.use_ours = False
    vj_xyz = cp.zeros_like(s.dm_xyz)
    for task in s.tasks:
        fj._launch_task(s, task, vj_xyz, force=None if patched else 'ref')
    vj = JE._Rt_to_dm(s.mol, vj_xyz.get(), s.pair_lst, s.pair_loc,
                      vhfopt.rys_envs)
    vj *= 2.
    return vhfopt.apply_coeff_CT_mat_C(vj)


def main():
    import electrolites
    electrolites.patch('fastj')
    import electrolites.fastj as fj
    print(f'# gpu4pyscf {electrolites.gpu4pyscf_version()}  '
          f'multi-density blocks per class: '
          f'{sorted({tuple(v) for v in fj._MDM_BLOCKS.values()})}')
    rng = np.random.default_rng(20240904)
    worst = 0.
    fails = 0
    for name, geom, charge, basis in CASES:
        mol = gto.M(atom=geom, basis=basis, charge=charge, verbose=0)
        vhfopt = JE._VHFOpt(mol, 1e-14).build()
        lmax = int(mol._bas[:, 1].max())
        case_fails = 0
        for kind in ('same', 'random', 'atomic'):
            for n_dm in NDMS:
                dms = densities(mol, kind, n_dm, rng)
                ref = build(vhfopt, dms, False)
                got = build(vhfopt, dms, True)
                d = float(cp.abs(got - ref).max())
                scale = float(cp.abs(ref).max()) or 1.
                rel = d / scale
                worst = max(worst, rel)
                ok = rel <= a.tol
                extra = ''
                if kind == 'same':
                    # Identical densities must give identical J -- to within
                    # round-off, not bit for bit: each density is accumulated
                    # by atomicAdd at its own address, and the hardware
                    # serialises per address, so the summation order differs
                    # between the m slots.  A stride bug shows up orders of
                    # magnitude above this, not at 1e-16 relative.
                    spread = float(cp.abs(got - got[0]).max()) / scale
                    ok = ok and spread <= a.tol
                    extra = f'  spread={spread:.1e}'
                    if spread > a.tol:
                        extra += '  <-- densities are not independent'
                fails += not ok
                case_fails += not ok
                if a.verbose or not ok:
                    print(f'{"ok " if ok else "FAIL"} {name:9s} {basis:10s} '
                          f'lmax={lmax} {kind:7s} n_dm={n_dm:3d}  '
                          f'max|dJ|={d:.2e} rel={rel:.2e}{extra}',
                          flush=True)
        print(f'{name:9s} {basis:10s} lmax={lmax}: '
              f'{"ok" if not case_fails else "FAILURES"}', flush=True)
    print(f'\nworst relative difference on J: {worst:.3e}  '
          f'(tolerance {a.tol:.0e})')
    if fails:
        print(f'{fails} case(s) FAILED')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
