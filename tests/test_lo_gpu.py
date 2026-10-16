"""Orbital localisation on the GPU (``electrolites.lo_gpu``) against
PySCF's ``lo`` from the same starting orbitals.

Boys and Pipek-Mezey (meta-Lowdin, Mulliken, IAO populations) run PySCF's
CIAH on the same objective, gradient and Hessian, so they must land on
PySCF's orbitals.  Edmiston-Ruedenberg is density-fitted here: against
PySCF's ER with its J/K replaced by the same density fitting, and against
the exact objective at the same orbitals to the DF error.  IBO's parallel Jacobi rounds visit the pairs in
another order than PySCF's sequential sweep, so the check is the
converged objective and the orbital set (each orbital matched up to sign).

    python tests/test_lo_gpu.py
"""
import sys

import numpy as np
from pyscf import gto, scf, lo, df
import importlib
from pyscf.lo import edmiston
pibo = importlib.import_module('pyscf.lo.ibo')

from electrolites import lo_gpu

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


def same_set(c1, c2, s):
    """max over orbitals of 1 - |best overlap|."""
    m = abs(c1.T @ s @ c2)
    return float(max(1 - m.max(axis=1).min(), 1 - m.max(axis=0).min()))


for name in ('glycine', 'alanine_dipeptide'):
    mol = gto.M(atom=f'benchmarks/molecules/{name}.xyz', basis='def2-svp',
                verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    s = mol.intor_symmetric('int1e_ovlp')
    nocc = mol.nelectron // 2
    from pyscf.data import elements
    ncore = elements.chemcore(mol)
    mo = mf.mo_coeff[:, ncore:nocc]
    for label, ref, new in (
            ('Boys', lambda: lo.Boys(mol, mo), lambda: lo_gpu.Boys(mol, mo)),
            ('PM meta-Lowdin', lambda: lo.PM(mol, mo, mf),
             lambda: lo_gpu.PM(mol, mo, mf)),
            ('PM Mulliken', lambda: lo.PM(mol, mo, mf, pop_method='mulliken'),
             lambda: lo_gpu.PM(mol, mo, mf, pop_method='mulliken')),
            ('PM IAO', lambda: lo.PM(mol, mo, mf, pop_method='iao'),
             lambda: lo_gpu.PM(mol, mo, mf, pop_method='iao'))):
        r, n = ref(), new()
        r.conv_tol = n.conv_tol = 1e-10
        c1, c2 = r.kernel(), n.kernel()
        d = same_set(c1, c2, s)
        e = abs(lo_gpu.Boys(mol, c1).cost_function()
                - lo_gpu.Boys(mol, c2).cost_function())
        report(f'{name} {label} == PySCF', d < 1e-6, f'orbitals {d:.1e}, '
               f'Boys spread {e:.1e}')

    # Edmiston-Ruedenberg: PySCF with density-fitted J/K, and exact
    aux = df.addons.make_auxbasis(mol)
    dfobj = df.DF(mol, auxbasis=aux).build()
    real_get_jk = edmiston.hf.get_jk
    edmiston.hf.get_jk = lambda m, dms, hermi=1: df.df_jk.get_jk(
        dfobj, np.asarray(dms), hermi)
    try:
        r = lo.ER(mol, mo)
        r.conv_tol = 1e-10
        c1 = r.kernel()
    finally:
        edmiston.hf.get_jk = real_get_jk
    n = lo_gpu.ER(mol, mo)
    n.conv_tol = 1e-10
    c2 = n.kernel()
    d = same_set(c1, c2, s)
    report(f'{name} ER == PySCF ER on DF J/K', d < 1e-6, f'orbitals {d:.1e}')
    if name == 'glycine':
        # the ER landscape has several maxima (exact ER and DF-ER from the
        # same start land on different ones, Boys orbitals score higher
        # than both), so the check is the DF objective against the exact
        # one at the same orbitals
        fx = lo.ER(mol, c2).cost_function(np.eye(mo.shape[1]))
        fd = n.cost_function()
        report(f'{name} DF-ER objective against exact', abs(fx - fd) / fx
               < 1e-5, f'sum (ii|ii) exact {fx:.6f}, DF {fd:.6f}')

    # IBO
    occ = mf.mo_coeff[:, :nocc]
    c1 = pibo.ibo(mol, occ, locmethod='IBO', grad_tol=1e-10, verbose=0)
    c2 = lo_gpu.ibo(mol, occ, grad_tol=1e-10)
    d = same_set(c1, c2, s)
    pm = lo.PM(mol, occ, mf, pop_method='iao')
    pm.exponent = 4
    cf = [pm.cost_function(np.linalg.lstsq(occ, c, rcond=None)[0])
          for c in (c1, c2)]
    report(f'{name} IBO == PySCF', d < 1e-6 and abs(cf[0] - cf[1]) < 1e-8,
           f'orbitals {d:.1e}, IAO^4 objective {cf[0]:.8f} / {cf[1]:.8f}')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
