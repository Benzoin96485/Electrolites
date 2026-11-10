"""Measure the added local-integral approximation against the legacy energy.

All calculations use the same stationary occupied orbitals and DLPNO
thresholds.  Report full energy and force changes; this is an accuracy
scan, not a universal error bound for the coefficient threshold.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pyscf import gto, scf

from benchmarks.dlpno_adjoint_scaling import alkane
from electrolites.dlpno.local import LocalDLPNO
from electrolites.dlpno.reference_cpu import CPUDLPNO, orbitals
from electrolites.dlpno.setup import count_frozen_core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--carbons', type=int, default=4)
    parser.add_argument('--cuts', nargs='+', type=float, default=[0., 1e-4, 1e-3, 3e-3])
    parser.add_argument('--triples', choices=['none', 't0', 't1'], default='t0')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parents[1]/'electrolites'/'dlpno'
    report = dict(settings=vars(args), complete=False, rows=[],
                  code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')})

    def save():
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2)+'\n')
        temp.replace(path)

    save()
    mol = gto.M(atom=alkane(args.carbons), basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('SCF did not converge')
    orb = orbitals(mol, mf.get_ovlp(), mf.get_fock(), mf.mo_coeff, mol.nelectron//2,
                   count_frozen_core(mol), ensure_minimum=True)
    options = dict(pno='loose', triples=None if args.triples == 'none' else args.triples,
                   c_lmo=orb.C_lmo, verbose=0, E_CONVERGENCE=1e-11,
                   R_CONVERGENCE=1e-10, MAXITER=150, T_CUT_ITER=0.)
    original = CPUDLPNO(mf, **options)
    original.kernel()
    reference_energy = dict(original.e)
    reference_force = original.nuc_grad_method(method='adjoint').kernel()
    report.update(reference_energy=reference_energy, reference_force=reference_force.tolist(),
                  options=original.opt, geometry_bohr=mol.atom_coords().tolist())
    del original
    gc.collect()
    for screen, cut in [('component', 0.), *[('atom', c) for c in args.cuts]]:
        start = time.perf_counter()
        state = LocalDLPNO(mf, pao_screen=screen, lmo_cut=cut, pao_cut=cut, **options)
        state.kernel()
        gradient = state.nuc_grad_method()
        force = gradient.kernel()
        row = dict(screen=screen, coefficient_cut=cut, energy=state.e,
                   energy_error={k: float(v-reference_energy[k]) for k, v in state.e.items()},
                   force_max_error=float(np.max(abs(force-reference_force))),
                   force_rms_error=float(np.sqrt(np.mean((force-reference_force)**2))),
                   torque=float(np.max(abs(np.cross(mol.atom_coords(), force).sum(axis=0)))),
                   support=gradient.diagnostics['integrals']['coefficient_support'],
                   seconds=time.perf_counter()-start)
        report['rows'].append(row)
        save()
        print(json.dumps(row), flush=True)
        del state, gradient
        gc.collect()
    report['complete'] = True
    exact = report['rows'][0]
    report['exact_limit_passed'] = abs(exact['energy_error']['corr']) < 1e-9 and exact['force_max_error'] < 1e-8
    save()
    if not report['exact_limit_passed']:
        raise RuntimeError('Full-support local provider failed its legacy-energy limit')


if __name__ == '__main__':
    main()
