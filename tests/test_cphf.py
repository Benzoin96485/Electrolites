"""Does W44's CPHF solve give GPU4PySCF's first-order orbitals?

``docs/DF_WORK_LIST.md`` **W44** (``electrolites/fastcphf.py``:
``hessian/{rhf,uhf}.py:solve_mo1`` and the Krylov solver under it).  Against
GPU4PySCF's own ``solve_mo1`` in one process, on one converged SCF and one
``h1mo``: ``mo1`` and ``mo_e1``, relative to the largest element.  The two
solve the same equations to the same tolerance, and ``krylov``'s blocks
change the rounding of the projections (``FASTCPHF_KRYLOV=upstream`` inside
this module's ``solve_mo1`` agrees to 1e-14), so the bar is ``--tol``.

Checked, restricted and unrestricted, density-fitted and direct, an organic
molecule and an open-shell transition-metal complex:

* the default (one atom block, the response in one group);
* the response in groups of right-hand sides (``FASTCPHF_FXSETS``);
* the subspace in page-locked host memory (``FASTCPHF_STORE=host``), also
  with groups;
* one Gram-Schmidt pass (``FASTCPHF_REORTH=0``) and upstream's Krylov
  solver under this module's blocks (``FASTCPHF_KRYLOV=upstream``);
* a whole Hessian, patched against ``FASTCPHF_OFF=1``;
* a whole direct Hessian with item 4's screening (catalogue B12) against
  ``FASTCPHF_SCREEN=off``, at ``conv_tol_cpscf = 1e-6``: the screen must be
  1e-9 and move the Hessian by less than ``--tol-screen``, a thirtieth of
  what the solve's own tolerance moves it (1e-6 against 1e-9: 4.7e-5 on
  penicillin V, 1.3e-4 on alanine dipeptide, 3.2e-4 on a Ni complex).

    python tests/test_cphf.py [--no-hessian]
"""
import argparse
import os
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--basis', default='def2-svp')
ap.add_argument('--tol', type=float, default=1e-9)
ap.add_argument('--tol-direct-groups', type=float, default=1e-8,
                help="the direct path's exchange screens on the largest "
                     "density of the call (`scf/jk.py`'s absmax dm_cond), so "
                     "a call on fewer sets drops different integrals: 2e-9 "
                     "on ethanol+, which upstream's own atom blocks move by "
                     "their Krylov tolerance")
ap.add_argument('--tol-hessian', type=float, default=1e-8,
                help='bar on the largest Hessian element, in Eh/bohr^2')
ap.add_argument('--tol-screen', type=float, default=1e-6,
                help="bar on the screened Hessian's largest change")
ap.add_argument('--no-hessian', action='store_true')
a = ap.parse_args()

import cupy as cp                                            # noqa: E402
from pyscf import gto                                        # noqa: E402
from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.lib import logger                             # noqa: E402

import electrolites                                          # noqa: E402
electrolites.patch_all()
from electrolites import fastcphf                            # noqa: E402

ETHANOL = '''
C  -0.748  -0.015   0.024
C   0.558   0.420  -0.679
O   1.598  -0.513  -0.421
H  -1.593   0.365  -0.554
H  -0.834   0.355   1.048
H  -0.780  -1.106   0.023
H   0.640   1.502  -0.580
H   0.567   0.174  -1.741
H   2.421  -0.190  -0.792
'''
HERE = os.path.dirname(os.path.abspath(__file__))
CUCL2 = os.path.join(HERE, '..', 'benchmarks', 'molecules', 'tmc_cucl2.xyz')

fails = []


def check(label, got, ref, tol=None):
    tol = a.tol if tol is None else tol
    err = 0.
    for g, r in zip(got, ref):
        g, r = np.asarray(g), np.asarray(r)
        if g.shape != r.shape:
            fails.append(f'{label}: shape {g.shape} != {r.shape}')
            print(f'  FAIL {label:58s} shape {g.shape} != {r.shape}')
            return
        err = max(err, abs(g - r).max() / (abs(r).max() or 1.))
    ok = err <= tol
    if not ok:
        fails.append(f'{label}: {err:.3e} > {tol:.1e}')
    print(f'  {"ok  " if ok else "FAIL"} {label:58s} {err:.3e}', flush=True)


class env:
    def __init__(self, **kw):
        self.kw = {f'FASTCPHF_{k}': str(v) for k, v in kw.items()}

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.kw}
        os.environ.update(self.kw)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def scf(atom, charge, spin, df):
    mol = gto.M(atom=atom, basis=a.basis, charge=charge, spin=spin, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol, xc='b3lyp')
    if df:
        mf = mf.density_fit()
    mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.conv_tol_cpscf = 1e-6
    mf.kernel()
    return mf


VARIANTS = [('', {}), (' [FXSETS=4]', {'FXSETS': 4}),
            (' [STORE=host]', {'STORE': 'host'}),
            (' [STORE=host FXSETS=5]', {'STORE': 'host', 'FXSETS': 5}),
            (' [REORTH=0]', {'REORTH': 0}),
            (' [KRYLOV=upstream]', {'KRYLOV': 'upstream'})]


def run(tag, mf, df):
    from gpu4pyscf.hessian import rhf as HR, uhf as HU
    uks = mf.mol.spin != 0
    stock = (HU if uks else HR)._solve_mo1_orig
    ours = fastcphf.solve_mo1_u if uks else fastcphf.solve_mo1_r
    mo_energy = cp.asarray(mf.mo_energy)
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    hess = mf.Hessian()
    log = logger.new_logger(hess, 0)
    h1mo = hess.make_h1(mo_coeff, mo_occ, None, None, log)
    fx = hess.gen_vind(mo_coeff, mo_occ)

    def flat(r):
        mo1, e1 = r
        return [cp.asnumpy(x) for x in (list(mo1) + list(e1) if uks
                                         else [mo1, e1])]
    ref = flat(stock(mf, mo_energy, mo_coeff, mo_occ, h1mo, fx, None, 4000,
                     log))
    for lbl, sw in VARIANTS:
        s0 = fastcphf.stats()
        with env(**sw):
            got = flat(ours(mf, mo_energy, mo_coeff, mo_occ, h1mo, fx, None,
                            4000, log))
        s1 = fastcphf.stats()
        extra = ''
        if s1['fx_groups'] - s0['fx_groups'] > s1['fx_calls'] - s0['fx_calls']:
            extra += ' grouped'
        if s1['store_host'] > s0['store_host']:
            extra += ' host'
        tol = (a.tol_direct_groups if not df and 'grouped' in extra
               else None)
        check(f'{tag} mo1, mo_e1{lbl}{extra}', got, ref, tol)
        if sw.get('FXSETS') and 'grouped' not in extra:
            fails.append(f'{tag}{lbl}: the response was not grouped')
        if sw.get('STORE') == 'host' and 'host' not in extra:
            fails.append(f'{tag}{lbl}: nothing went to the host')


CASES = [('ethanol RKS DF', ETHANOL, 0, 0, True),
         ('ethanol+ UKS DF', ETHANOL, 1, 1, True),
         ('ethanol RKS direct', ETHANOL, 0, 0, False),
         ('ethanol+ UKS direct', ETHANOL, 1, 1, False),
         ('CuCl2 UKS DF (open-shell TM)', CUCL2, 0, 1, True)]

print(f'# patched {electrolites.patched()}  basis {a.basis}')
for tag, atom, charge, spin, df in CASES:
    if atom.endswith('.xyz') and not os.path.exists(atom):
        print(f'  skip {tag}: no {atom}')
        continue
    print(f'\n## {tag}')
    run(tag, scf(atom, charge, spin, df), df)

if not a.no_hessian:
    print('\n## whole Hessians, patched against FASTCPHF_OFF=1')
    for tag, atom, charge, spin, df in (CASES[0], CASES[1]):
        mf = scf(atom, charge, spin, df)
        hs = []
        for off in ('', '1'):
            with env(OFF=off):
                hs.append(np.asarray(mf.Hessian().kernel()))
        err = abs(hs[0] - hs[1]).max()
        ok = err <= a.tol_hessian
        if not ok:
            fails.append(f'{tag} Hessian: {err:.3e}')
        print(f'  {"ok  " if ok else "FAIL"} {tag + " whole Hessian":58s} '
              f'{err:.3e} Eh/bohr^2', flush=True)

    print('\n## whole direct Hessians, screened against FASTCPHF_SCREEN=off')
    for tag, atom, charge, spin, df in (CASES[2], CASES[3]):
        mf = scf(atom, charge, spin, df)
        hs = []
        for scr in ('on', 'off'):
            fastcphf._S['screen'] = 0.
            with env(SCREEN=scr):
                hs.append(np.asarray(mf.Hessian().kernel()))
            if scr == 'on' and fastcphf.stats()['screen'] != 1e-9:
                fails.append(f'{tag}: screen {fastcphf.stats()["screen"]}')
        err = abs(hs[0] - hs[1]).max()
        ok = err <= a.tol_screen
        if not ok:
            fails.append(f'{tag} screened Hessian: {err:.3e}')
        print(f'  {"ok  " if ok else "FAIL"} {tag + " screened Hessian":58s} '
              f'{err:.3e} Eh/bohr^2', flush=True)

print('\n# fastcphf', fastcphf.stats())
if fails:
    print('\nFAILED:\n  ' + '\n  '.join(fails))
    sys.exit(1)
print('\nall ok')
