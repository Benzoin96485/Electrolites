"""`fastxcfxc`'s unrestricted path against GPU4PySCF, in one process.

`nr_uks_fxc` is called both ways on the same converged UKS SCF and the same
`fxc` cache, so the comparison is exact up to the reassociation of the grid
sums.  Both `hermi` values are checked: 1 is CPKS/NMR/the polarizability and
0 is TDA/TDDFT, and `hermi = 0` is the one place where the two
implementations do genuinely different arithmetic (GPU4PySCF forms a second
`dm^T . ao` product per grid block; this module symmetrises the density once
per call instead).

Two things this suite checks that `test_fxc.py` cannot.

* **The channel coupling.**  `fxc` is a 2x2 block in the spin index, so the
  potential of spin `a` takes both spins' perturbed densities.  The two
  returned blocks are compared separately, and the right-hand sides are built
  so that the two channels are *different* -- a doublet with `nocca != noccb`,
  and per-spin perturbations rather than one perturbation copied.  A coupling
  written the wrong way round would still be symmetric and still look like a
  potential.
* **That the patch is reached.**  `fastxcfxc._usable_u` is asserted directly,
  because an unrestricted run that fell through to GPU4PySCF would agree with
  GPU4PySCF perfectly and say nothing.

Each case is checked in both modes.  `blocked` is the shipped path, with the
custom kernels; `nokernel` keeps the same layout and the same batched GEMMs
and puts GPU4PySCF's own per-set contraction helpers back.  Running both
separates the two things that could be wrong here: the *layout* -- the
interleaving that lets one set of kernels serve both spin cases -- and the
one CUDA kernel that had to be written for the spin coupling.  A layout error
fails both columns; a kernel error fails only `blocked`.

The end-to-end column is a **UKS TDA** spectrum with the module patched and
unpatched on the same SCF.  It is TDA rather than a polarizability because
`gpu4pyscf.properties.polarizability` is restricted-only on 1.8.1.

The second table is the **`mo1` branch**, which nothing above reaches: an
unrestricted Hessian's CPHF is the only caller that tags the perturbed
density with `mo1` (`hessian/uhf.py:gen_vind`), and on that input GPU4PySCF
evaluates the density from the MO-basis perturbation instead of from a
density matrix.  This module keeps that evaluation and batches everything
after it, so the check is a whole UKS Hessian with `nr_uks_fxc` patched and
unpatched -- with `hself`, the same Hessian run twice on GPU4PySCF's own
code, beside it, because the CPHF's own run-to-run spread is the floor.
`fastxcfxc.stats()` is asserted at the end: the branch has to have been
taken, since a fall-through would agree perfectly and say nothing.

    python tests/test_ufxc.py [--quick]
"""
import argparse

import cupy as cp
import numpy as np

import electrolites
electrolites.patch('fastxcfxc')

from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.dft import numint as NI                       # noqa: E402
from electrolites import fastxcfxc                           # noqa: E402

O2 = 'O 0. 0. 0.; O 0. 0. 1.208'
CH3 = '''C 0. 0. 0.; H 1.079 0. 0.
H -0.539 0.934 0.; H -0.539 -0.934 0.'''
NO = 'N 0. 0. 0.; O 0. 0. 1.154'
M = 'benchmarks/molecules/'

# (label, atom, basis, xc, grid level, charge, spin, nset)
CASES = [
    ('O2', O2, 'def2-svp', 'b3lyp', 3, 0, 2, 3),
    ('O2', O2, 'def2-svp', 'svwn', 3, 0, 2, 3),               # LDA
    ('O2', O2, 'def2-tzvpp', 'pbe', 3, 0, 2, 2),              # reaches g
    ('CH3', CH3, '6-31g*', 'pbe0', 3, 0, 1, 5),               # nocca != noccb
    ('CH3', CH3, 'def2-svp', 'blyp', 3, 0, 1, 8),
    ('NO', NO, 'def2-svp', 'wb97x-d3bj', 3, 0, 1, 4),         # range separated
    ('CuCl2', M + 'tmc_cucl2.xyz', 'def2-svp', 'b3lyp', 3, 0, 1, 3),
    ('CuCl2', M + 'tmc_cucl2.xyz', 'def2-tzvp', 'b3lyp', 3, 0, 1, 6),
    ('FeCl4-', M + 'tmc_fecl4.xyz', 'def2-svp', 'b3lyp', 3, -1, 5, 3),
    ('Pt amine', M + 'tmc_pt_amine.xyz', 'def2-svp', 'b3lyp', 3, 0, 2, 3),
    ('Ni dithiolene', M + 'tmc_ni_dithiolene.xyz', 'def2-svp', 'b3lyp',
     3, -1, 1, 5),
    # meta-GGA, which this module covers on the response path even though
    # `fastxc` declines it in the SCF
    ('O2', O2, 'def2-svp', 'm06-2x', 3, 0, 2, 3),
    ('CH3', CH3, 'def2-svp', 'tpss', 3, 0, 1, 5),
    ('CuCl2', M + 'tmc_cucl2.xyz', 'def2-svp', 'm06-2x', 3, 0, 1, 3),
]

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
ap.add_argument('--tda-max-nao', type=int, default=130,
                help='run the end-to-end TDA column only up to this many AOs')
ap.add_argument('--section', default='all', choices=('all', 'terms', 'hess'),
                help='`terms` is the table of the plain-density path, `hess` '
                     'the `mo1` branch through a whole UKS Hessian.  They are '
                     'separable because each is minutes long and a batch '
                     'queue has a wall clock.')
a = ap.parse_args()
cases = CASES[:4] + CASES[-3:] if a.quick else CASES
if a.section == 'hess':
    cases = []

FAST = fastxcfxc._nr_uks_fxc_dispatch
NOKERN = fastxcfxc._nr_uks_fxc_nokernel
ORIG = NI._nr_uks_fxc_orig


def build(atom, basis, charge, spin):
    kw = dict(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0,
              max_memory=40000)
    if basis.lower().startswith('def2') and any(
            gto.charge(sym) > 36 for sym, _ in gto.format_atom(atom)):
        kw['ecp'] = basis
    return gto.M(**kw)


def perturbed_dms(mf, nset, hermi):
    """Right-hand sides of the shape a solver produces, per spin channel.

    The two channels are perturbed differently on purpose: the dipole
    components are taken in a different order for beta than for alpha, so a
    coupling term picked up from the wrong channel does not cancel.
    """
    mo = [cp.asarray(mf.mo_coeff[s]) for s in range(2)]
    occ = [cp.asarray(mf.mo_occ[s]) for s in range(2)]
    e = [cp.asarray(mf.mo_energy[s]) for s in range(2)]
    with mf.mol.with_common_orig((0, 0, 0)):
        h1 = cp.asarray(mf.mol.intor('int1e_r'))
    out = []
    for s in range(2):
        mocc, mvir = mo[s][:, occ[s] > 0], mo[s][:, occ[s] == 0]
        de = e[s][occ[s] == 0][:, None] - e[s][occ[s] > 0][None, :]
        xs = [-(mvir.T @ h1[(i + s) % 3] @ mocc) / de for i in range(3)]
        order = cp.argsort(de.ravel())
        for k in range(max(0, nset - 3)):
            x = cp.zeros(de.shape)
            x.ravel()[order[(k + s) % de.size]] = 1.0
            xs.append(x)
        dm1 = cp.asarray([mvir @ x @ mocc.T for x in xs[:nset]])
        if hermi == 1:
            dm1 = dm1 + dm1.transpose(0, 2, 1)
        out.append(cp.asarray(dm1, order='C'))
    return cp.asarray(out, order='C')


print(f'{"case":30s} {"xc":11s} {"nao":>5s} {"nset":>4s} {"hermi1":>10s} '
      f'{"hermi0":>10s} {"h1 nokern":>10s} {"h0 nokern":>10s} {"tda/eV":>9s} '
      f'{"used":>5s}')
print('-' * 128)
fails = skipped = 0
for label, atom, basis, xc, level, charge, spin, nset in cases:
    mol = build(atom, basis, charge, spin)
    mf = dft.UKS(mol, xc=xc)
    mf.grids.level = level
    mf.conv_tol = 1e-10
    mf.direct_scf_tol = 1e-13
    mf.max_cycle = 200
    mf.kernel()
    tag = f'{label}/{basis}'
    if not mf.converged:
        print(f'{tag:30s} {xc:11s} SCF did not converge -- skipped',
              flush=True)
        skipped += 1
        mol = mf = None
        cp.get_default_memory_pool().free_all_blocks()
        continue

    ni = mf._numint
    ni.build(mol, mf.grids.coords)
    mo, occ = cp.asarray(mf.mo_coeff), cp.asarray(mf.mo_occ)
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, xc, mo, occ, 1)

    errs = {}
    used = None
    for hermi in (1, 0):
        dms = perturbed_dms(mf, nset, hermi)
        used = fastxcfxc._usable_u(ni, xc, dms, hermi, fxc)
        args = (ni, mol, mf.grids, xc, None, dms, 0, hermi, rho0, vxc, fxc)
        v_ref = cp.asarray(ORIG(*args, 2000, None))
        # both spin blocks, separately: a dropped spin index shows in one
        scale = max(float(cp.abs(v_ref[s]).max()) for s in range(2)) or 1.0
        for which, fn in (('', FAST), ('n', NOKERN)):
            v = cp.asarray(fn(*args) if which == ''
                           else fn(*args, 2000, None))
            errs[which + str(hermi)] = max(
                float(cp.abs(v[s] - v_ref[s]).max())
                for s in range(2)) / scale
            v = None

    d_td = float('nan')
    if mol.nao <= a.tda_max_nao:
        # on the *same* SCF: nothing here is called during an SCF, so a second
        # one would only add its own run-to-run spread (test_fxc.py records a
        # case where that alone moved a spectrum by 2.7e-6 eV)
        def _tda():
            td = mf.TDA()
            td.nstates = 3
            td.kernel()
            return np.asarray(td.e)

        e_fast = _tda()
        NI.nr_uks_fxc = NI.NumInt.nr_uks_fxc = ORIG
        e_ref = _tda()
        NI.nr_uks_fxc = NI.NumInt.nr_uks_fxc = FAST
        d_td = float(np.abs(e_fast - e_ref).max()) * 27.211386245988

    ok = (bool(used) and max(errs.values()) <= 1e-9
          and not (d_td == d_td and d_td > 1e-6))
    fails += not ok
    print(f'{tag:30s} {xc:11s} {mol.nao:5d} {nset:4d} {errs["1"]:10.2e} '
          f'{errs["0"]:10.2e} {errs["n1"]:10.2e} {errs["n0"]:10.2e} '
          f'{d_td:9.2e} {str(bool(used)):>5s}'
          f'{"" if ok else "   FAIL"}', flush=True)
    mol = mf = None
    cp.get_default_memory_pool().free_all_blocks()

if cases:
    print('-' * 128)
    print(f'{len(cases) - fails - skipped}/{len(cases)} cases pass; '
          f'{skipped} skipped')

# ---- the `mo1` branch: an unrestricted Hessian's CPHF --------------------
HESS = [] if a.section == 'terms' else [
        ('O2', O2, 'def2-svp', 'b3lyp', 3, 0, 2),
        ('CH3', CH3, '6-31g*', 'pbe0', 3, 0, 1),
        ('CuCl2', M + 'tmc_cucl2.xyz', 'def2-svp', 'b3lyp', 3, 0, 1),
        ('O2', O2, 'def2-svp', 'm06-2x', 3, 0, 2)]
if a.quick:
    HESS = HESS[:2]
ncases = len(cases) + len(HESS)

before = fastxcfxc.stats()
if HESS:
    print(f'\n{"case":30s} {"xc":11s} {"nat":>4s} {"nao":>5s} '
          f'{"hessian":>10s} {"hself":>10s} {"|H|":>9s} {"mo1 calls":>9s}')
    print('-' * 96)
for label, atom, basis, xc, level, charge, spin in HESS:
    mol = build(atom, basis, charge, spin)
    mf = dft.UKS(mol, xc=xc)
    mf.grids.level = level
    mf.conv_tol = 1e-10
    mf.conv_tol_cpscf = 1e-6
    mf.direct_scf_tol = 1e-13
    mf.max_cycle = 200
    mf.kernel()
    tag = f'{label}/{basis}'
    if not mf.converged:
        print(f'{tag:30s} {xc:11s} SCF did not converge -- skipped')
        skipped += 1
        mol = mf = None
        cp.get_default_memory_pool().free_all_blocks()
        continue
    n0 = fastxcfxc.stats()['ours_u_mo']
    h = mf.Hessian()
    h.auxbasis_response = 0
    NI.nr_uks_fxc = NI.NumInt.nr_uks_fxc = FAST
    hf = cp.asarray(h.kernel())
    n_mo = fastxcfxc.stats()['ours_u_mo'] - n0
    NI.nr_uks_fxc = NI.NumInt.nr_uks_fxc = ORIG
    h.de = np.zeros((0, 0, 3, 3))
    ho = cp.asarray(h.kernel())
    h.de = np.zeros((0, 0, 3, 3))
    ho2 = cp.asarray(h.kernel())
    NI.nr_uks_fxc = NI.NumInt.nr_uks_fxc = FAST
    eh = float(cp.abs(hf - ho).max())
    ehs = float(cp.abs(ho2 - ho).max())
    nrm = float(cp.abs(ho).max())
    ok = n_mo > 0 and eh <= max(1e-8 * max(nrm, 1.), 10 * ehs)
    fails += not ok
    print(f'{tag:30s} {xc:11s} {mol.natm:4d} {mol.nao:5d} {eh:10.2e} '
          f'{ehs:10.2e} {nrm:9.2e} {n_mo:9d}{"" if ok else "   FAIL"}',
          flush=True)
    mol = mf = h = hf = ho = ho2 = None
    cp.get_default_memory_pool().free_all_blocks()
print('-' * 96)
print(f'{ncases - fails - skipped}/{ncases} cases pass; {skipped} skipped')
print(f'fastxcfxc.stats() = {fastxcfxc.stats()}  (was {before})')

raise SystemExit(1 if fails else 0)
