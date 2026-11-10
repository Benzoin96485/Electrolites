"""Audit local Boys response using a physical MP2/CCSD/(T) adjoint source.

The full response is the oracle. Failed omitted-equation residuals are
retained, and only accepted responses are contracted to nuclear forces.
Dense input production, CPHF and force contractions are explicitly timed;
this benchmark does not certify an entire linear-scaling gradient.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from types import SimpleNamespace

import numpy as np
from pyscf import gto, lo, scf

from benchmarks.dlpno_adjoint_scaling import alkane
from electrolites.dlpno.local import LocalDLPNO
from electrolites.dlpno.local_boys import LocalBoysResidualError
from electrolites.dlpno.orbital_adjoint import orbital_densities, contract_orbital_densities


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_reference(path, base, gradient, report):
    """Portable numeric reference; no executable pickle or node-local DF file."""
    path = Path(path)
    arrays = dict(mol=np.array(base.mol.dumps()), c_lmo=base.orb.C_lmo,
                  ncore=np.array(base.orb.ncore), mo_coeff=base.mf.mo_coeff,
                  mo_occ=base.mf.mo_occ, mo_energy=base.mf.mo_energy,
                  e_tot=np.array(base.mf.e_tot), gradient=gradient.de,
                  orbital_force=gradient.components['orbital'])
    arrays.update({'source_'+k: v for k, v in gradient.orbital_sources.items()})
    arrays.update({'relaxed_'+k: v for k, v in gradient.relaxed_densities.items()})
    np.savez_compressed(path, **arrays)
    manifest = dict(data=str(path), sha256=digest(path), reference_report=report,
                    source_sha256=report['code_sha256'], purpose='Fixed physical source for Boys response comparisons')
    target = Path(str(path)+'.json')
    target.write_text(json.dumps(manifest, indent=2)+'\n')
    return str(target)


def load_reference(path):
    manifest = json.loads(Path(path).read_text())
    data = Path(manifest['data'])
    if digest(data) != manifest['sha256']:
        raise ValueError('Response reference data hash mismatch')
    with np.load(data, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if not all(np.isfinite(v).all() for k, v in arrays.items() if k != 'mol'):
        raise ValueError('Nonfinite response reference')
    mol = gto.loads(arrays['mol'].item())
    mf = scf.RHF(mol).density_fit()
    for key in ('mo_coeff', 'mo_occ', 'mo_energy', 'e_tot'):
        setattr(mf, key, arrays[key])
    mf.converged = True
    mf.with_df.build()
    orbital_residual = float(np.max(abs(mf.get_grad(mf.mo_coeff, mf.mo_occ))))
    if orbital_residual > 1e-8:
        raise ValueError(f'Loaded SCF is not stationary: {orbital_residual:.3e}')
    base = SimpleNamespace(mf=mf, mol=mol, orb=SimpleNamespace(
        C_lmo=arrays['c_lmo'], ncore=int(arrays['ncore'])))
    sources = {k[7:]: v for k, v in arrays.items() if k.startswith('source_')}
    relaxed = {k[8:]: v for k, v in arrays.items() if k.startswith('relaxed_')}
    return base, sources, relaxed, arrays['gradient'], arrays['orbital_force'], manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--carbons', type=int, default=8)
    parser.add_argument('--level', choices=['auto', 'mp2', 'ccsd'], default='auto')
    parser.add_argument('--triples', choices=['none', 't0', 't1'], default='t0')
    parser.add_argument('--radii', nargs='+', type=float, default=[4., 8., 12., 16., 24., 32.])
    parser.add_argument('--reference', help='Hash-verified NPZ manifest from an earlier run of this benchmark')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parents[1]/'electrolites'/'dlpno'
    report = dict(complete=False, goal_complete=False, settings=vars(args), rows=[],
                  code_sha256={p.name: digest(p) for p in code.glob('*.py')},
                  validator_sha256=digest(__file__), job_id=os.environ.get('SLURM_JOB_ID'),
                  limitations=['Dense dipole/source adapter and full residual oracle remain.',
                               'AO backtransforms, CPHF and orbital nuclear contractions remain nonlinear.',
                               'Source retention and solver iterations must remain bounded for a linear solve.'])
    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(path)
    save()
    if args.reference:
        start = time.perf_counter()
        base, sources, reference_relaxed, force, orbital_force, manifest = load_reference(args.reference)
        report.update(reference_manifest=str(args.reference), reference_data_sha256=manifest['sha256'],
                      reference_code_sha256=manifest['source_sha256'],
                      reference_settings=manifest['reference_report']['settings'],
                      input_rebuild_seconds=time.perf_counter()-start)
    else:
        mol = gto.M(atom=alkane(args.carbons), basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-9, 150
        report['phase'] = 'scf'; save()
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('SCF did not converge')
        base = LocalDLPNO(mf, pno='loose', triples=None if args.triples == 'none' else args.triples,
                          cache_bytes=1 << 30, dipole_tail_options={}, local_screening_options={},
                          verbose=0, E_CONVERGENCE=1e-11, R_CONVERGENCE=1e-10, MAXITER=150, T_CUT_ITER=0.)
        report['phase'] = 'energy'; save()
        start = time.perf_counter(); base.kernel()
        report['energy_seconds'] = time.perf_counter()-start
        report['energy'] = dict(base.e)
        report['phase'] = 'full_gradient'; save()
        def progress(event):
            report['last_gradient_event'] = event; save()
        gradient = base.nuc_grad_method(level=args.level, progress=progress)
        start = time.perf_counter(); force = gradient.kernel()
        report.update(full_gradient_seconds=time.perf_counter()-start,
                      full_gradient_timings=gradient.timings, full_response=gradient.diagnostics['orbitals'])
        sources, reference_relaxed = gradient.orbital_sources, gradient.relaxed_densities
        orbital_force = gradient.components['orbital']
        report['reference_manifest'] = save_reference(path.with_suffix('.response.npz'), base, gradient, dict(report))
    n = base.orb.C_lmo.shape[1]
    report.update(natm=base.mol.natm, occupied=n, full_variables=n*(n-1)//2,
                  parent_translation_residual=float(np.max(abs(force.sum(axis=0)))),
                  parent_rotation_residual=float(np.max(abs(np.cross(base.mol.atom_coords(), force).sum(axis=0)))))
    dipoles = lo.boys.dipole_integral(base.mol, base.orb.C_lmo)
    centers = np.diagonal(dipoles, axis1=1, axis2=2).T
    # A bounding-box diameter safely includes every pair for the oracle
    # endpoint, without constructing a dense distance matrix.
    full_radius = float(np.linalg.norm(np.ptp(centers, axis=0))+1.)
    for radius in sorted(set([*args.radii, full_radius])):
        report['phase'] = f'radius_{radius:.6f}'; save()
        row = dict(radius=radius, covers_full_box=radius >= full_radius)
        start = time.perf_counter()
        try:
            relaxed, info = orbital_densities(base, sources, boys_pair_radius=radius)
        except LocalBoysResidualError as error:
            row.update(accepted=False, boys=error.diagnostics, response_seconds=time.perf_counter()-start)
        else:
            row.update(accepted=True, response=info, boys=info['boys'], response_seconds=time.perf_counter()-start,
                       raw_density_max_errors={k: float(np.max(abs(relaxed[k]-reference_relaxed[k]))) for k in relaxed},
                       symmetric_density_max_errors={k: float(np.max(abs(
                           (relaxed[k]-reference_relaxed[k])+
                           (relaxed[k]-reference_relaxed[k]).swapaxes(-1, -2))))*.5 for k in relaxed})
            start = time.perf_counter()
            changed_force = contract_orbital_densities(base, relaxed)-orbital_force
            row.update(contraction_seconds=time.perf_counter()-start,
                       force_max_error=float(np.max(abs(changed_force))),
                       force_rms_error=float(np.sqrt(np.mean(changed_force**2))),
                       force_passed=float(np.max(abs(changed_force))) < 1e-7)
        report['rows'].append(row); save()
        print(json.dumps(row), flush=True)
    full = [row for row in report['rows'] if row['covers_full_box']]
    report.update(complete=True, phase='complete',
                  full_limit_passed=all(row.get('accepted') and row.get('force_passed') for row in full),
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save()
    if not report['full_limit_passed']:
        raise RuntimeError('Full-radius response did not recover the reference gradient')


if __name__ == '__main__':
    main()
