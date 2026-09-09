"""Correctness of the multi-density exchange build and the fused J+K build.

Catalogue item **B4b**.  Two entry points changed in that round and both are
checked here against GPU4PySCF's own kernels, on the same molecule and the same
screening data, **element-wise on the AO-space matrices at a fixed density**:

* ``jk._VHFOpt.get_k`` with ``n_dm > 1``, which is what an analytic Hessian's
  CPHF solve calls (``hessian/rhf.py:857``, ``n_dm = 3*natm``).  `fastk` used
  to hand every such call back (`fastk.py`'s ``if n_dm != 1``), so all of it was
  GPU4PySCF's.
* ``jk._VHFOpt.get_jk``, the fused build every response property calls
  (``scf/_response_functions.py``) and which no module here was on at all --
  neither `fastj` nor `fastk` patches it.  `fastjk` is that kernel.

A converged energy is not a useful oracle for either: both are only reached
with more than one density, i.e. inside a CPHF or Davidson solve, never on the
SCF's own path.  The comparison is therefore the fixed-density, matrix-by-matrix
one that the catalogue's 6 asks for.

``hermi = 0`` is checked as well as ``hermi = 1``, because every TDDFT call is
``hermi = 0`` and it is a different code path: ``jk.py:443`` stacks the
transposed densities so a tril-only quartet enumeration reaches both triangles,
which doubles ``n_dm`` inside the kernel and changes which index of the density
each contraction reads.

Three density shapes are used, for the reasons ``tests/test_j_ndm.py`` gives:
identical copies (which must come back identical, so a wrong per-density stride
cannot hide), random perturbations of different magnitudes (which exercise the
union screening), and atom-local densities (the shape of a CPHF right-hand
side).

    python tests/test_jk.py                     # six molecule/basis combinations
    python tests/test_jk.py --basis def2-tzvp   # up to f functions
"""
import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='',
                help='restrict to one basis instead of the built-in sweep')
ap.add_argument('--ndm', default='1,2,3,5,8',
                help='comma separated density counts to check')
ap.add_argument('--tol', type=float, default=1e-12,
                help='maximum allowed difference, relative to the largest '
                     'element of the reference matrix')
ap.add_argument('--verbose', action='store_true')
a = ap.parse_args()

import cupy as cp                                             # noqa: E402
from pyscf import gto                                         # noqa: E402
from gpu4pyscf.scf import jk as JK                            # noqa: E402

# the unpatched kernels, kept before either module patches anything
_REF_K = JK._VHFOpt.get_k
_REF_JK = JK._VHFOpt.get_jk

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
    seen, uniq = set(), []
    for n, g, c, _ in CASES:
        if n not in seen:
            seen.add(n)
            uniq.append((n, g, c, a.basis))
    CASES = uniq

NDMS = [int(x) for x in a.ndm.split(',')]


def densities(mol, kind, n_dm, hermi, rng):
    """``n_dm`` AO-space densities of one shape, symmetric iff ``hermi``."""
    nao = mol.nao
    nocc = max(1, mol.nelectron // 2)
    c = rng.standard_normal((nao, nocc))
    base = 2. * c @ c.T / nao
    if kind == 'same':
        out = np.repeat(base[None], n_dm, axis=0)
    elif kind == 'random':
        out = np.empty((n_dm, nao, nao))
        for i in range(n_dm):
            p = rng.standard_normal((nao, nao)) * 10. ** -(i % 5)
            out[i] = base + p * 1e-2
    elif kind == 'atomic':
        # a CPHF right-hand side's shape: support on one atom's shells only
        ao_slices = mol.aoslice_by_atom()
        out = np.zeros((n_dm, nao, nao))
        for i in range(n_dm):
            p0, p1 = ao_slices[i % mol.natm][2:]
            out[i, p0:p1, p0:p1] = rng.standard_normal((p1 - p0, p1 - p0))
    else:
        raise ValueError(kind)
    if hermi == 1:
        out = out + out.transpose(0, 2, 1)
    return out


def _report(fails, worst, ok, label, name, basis, lmax, kind, n_dm, hermi,
            rel, extra=''):
    if a.verbose or not ok:
        print(f'{"ok " if ok else "FAIL"} {label:3s} {name:9s} {basis:10s} '
              f'lmax={lmax} {kind:7s} hermi={hermi} n_dm={n_dm:3d}  '
              f'rel={rel:.2e}{extra}', flush=True)


def main():
    import electrolites
    electrolites.patch('fastk', 'fastjk')
    import electrolites.fastk as fastk
    import electrolites.fastjk as fastjk
    print(f'# gpu4pyscf {electrolites.gpu4pyscf_version()}  '
          f'fastjk classes: {len(fastjk.IMPLEMENTED)} one-thread + '
          f'{len(fastjk.IMPLEMENTED2)} threadIdx.y')
    rng = np.random.default_rng(20240908)
    worst = {'K': 0., 'J': 0., 'JK': 0.}
    fails = 0
    for name, geom, charge, basis in CASES:
        mol = gto.M(atom=geom, basis=basis, charge=charge, verbose=0)
        opt = JK._VHFOpt(mol, 1e-14).build()
        lmax = int(mol._bas[:, 1].max())
        case_fails = 0
        for hermi in (1, 0):
            for kind in ('same', 'random', 'atomic'):
                for n_dm in NDMS:
                    # the kernels work in the sorted cartesian AO basis, so
                    # the densities are built in mol's basis (where "one
                    # atom's shells" means something) and transformed the way
                    # every caller transforms them
                    dms = opt.apply_coeff_C_mat_CT(
                        cp.asarray(densities(mol, kind, n_dm, hermi, rng)))
                    dms = cp.asarray(dms, order='C')
                    # ---- the exchange build on its own (the Hessian's path)
                    ref = _REF_K(opt, dms.copy(), hermi, 0, 0.0, 1.0, 1.0)
                    got = fastk.get_k(opt, dms.copy(), hermi, 0, 0.0, 1.0, 1.0)
                    scale = float(cp.abs(ref).max()) or 1.
                    rel = float(cp.abs(got - ref).max()) / scale
                    ok = rel <= a.tol
                    extra = ''
                    if kind == 'same':
                        # identical densities must give identical K, to
                        # round-off: each density is accumulated by atomicAdd
                        # at its own address, so the summation order differs
                        # between the slots.  A wrong stride shows up orders
                        # of magnitude above this
                        spread = float(cp.abs(got - got[0]).max()) / scale
                        ok = ok and spread <= a.tol
                        extra = f'  spread={spread:.1e}'
                    worst['K'] = max(worst['K'], rel)
                    fails += not ok
                    case_fails += not ok
                    _report(fails, worst, ok, 'K', name, basis, lmax, kind,
                            n_dm, hermi, rel, extra)
                    # ---- the fused build (every response property's path)
                    rj, rk = _REF_JK(opt, dms.copy(), hermi, 0)
                    gj, gk = fastjk.get_jk(opt, dms.copy(), hermi, 0)
                    sj = float(cp.abs(rj).max()) or 1.
                    sk = float(cp.abs(rk).max()) or 1.
                    relj = float(cp.abs(gj - rj).max()) / sj
                    relk = float(cp.abs(gk - rk).max()) / sk
                    okj = relj <= a.tol and relk <= a.tol
                    worst['J'] = max(worst['J'], relj)
                    worst['JK'] = max(worst['JK'], relk)
                    fails += not okj
                    case_fails += not okj
                    _report(fails, worst, okj, 'JK', name, basis, lmax, kind,
                            n_dm, hermi, max(relj, relk),
                            f'  (J {relj:.1e}, K {relk:.1e})')
        print(f'{name:9s} {basis:10s} lmax={lmax}: '
              f'{"ok" if not case_fails else "FAILURES"}', flush=True)
    print(f'\nworst relative difference: K {worst["K"]:.3e}, '
          f'fused J {worst["J"]:.3e}, fused K {worst["JK"]:.3e}  '
          f'(tolerance {a.tol:.0e})')
    if fails:
        print(f'{fails} case(s) FAILED')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
