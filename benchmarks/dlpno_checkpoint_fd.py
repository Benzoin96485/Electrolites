"""Independent energy differences against a saved, completed local gradient.

The trusted-local energy checkpoint, gradient report and current source must
agree by hash. Each displaced SCF, Boys solution, PNO/TNO space and amplitude
solution is recomputed. Only the localization branch is followed; no orbital
or PAO projection is frozen. All coordinate units are bohr.
"""
import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
from pyscf import gto, lo, scf
from scipy.optimize import linear_sum_assignment

from benchmarks.dlpno_adjoint_scaling import file_sha256, load_primal
from benchmarks.dlpno_gradient_check import tracked_driver, topology
from benchmarks.dlpno_reference_energy import reference_energy
from electrolites.dlpno.local import LocalDLPNO
from electrolites.dlpno.reference_cpu import stationary_boys, minimum_boys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--directions', nargs='+', required=True, help='Zero-based atom,Cartesian pairs')
    parser.add_argument('--steps', nargs='+', type=float, default=[2e-4, 1e-4])
    parser.add_argument('--scf-backend', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--hf-energy-backend', choices=['scf', 'cpu', 'legacy-gpu'], default='scf',
                        help='Independent HF energy evaluator; original SCF energies and FD errors remain in the report')
    parser.add_argument('--scf-energy-tolerance', type=float, default=1e-13)
    parser.add_argument('--scf-gradient-tolerance', type=float, default=1e-10)
    parser.add_argument('--boys-gradient-tolerance', type=float, default=1e-10)
    parser.add_argument('--save-orbitals', action='store_true',
                        help='Save each converged displaced orbital solution for independent audits')
    parser.add_argument('--save-points', action='store_true',
                        help='Save each completed energy point before evaluating the next displacement')
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if any(not np.isfinite(value) or value <= 0 for value in (
            args.scf_energy_tolerance, args.scf_gradient_tolerance, args.boys_gradient_tolerance)):
        parser.error('Orbital convergence tolerances must be finite and positive')
    previous = json.loads(Path(args.reference).read_text())
    if not previous.get('complete') or not previous.get('passed'):
        raise ValueError('The reference gradient must be complete and pass its invariance gates')
    code = Path(inspect.getfile(LocalDLPNO)).parent
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')}
    if hashes != previous['code_sha256']:
        raise ValueError('The current correlation source differs from the reference gradient')
    base, _ = load_primal(args.checkpoint, dict(code_sha256=hashes, settings=previous['settings']))
    mol, mf0 = base.mol, base.mf
    settings = previous['settings']
    if settings['energy_backend'] != 'gpu':
        mf0.with_df.build()  # Recreate CPU node-local fitting files only when used.
    cl0 = base._orbital_reference.C_lmo.copy()
    ncore, nocc = base._orbital_reference.ncore, int(np.count_nonzero(mf0.mo_occ > 0))
    options = dict(base.opt)
    options.pop('PNO', None)
    options.update(E_CONVERGENCE=1e-12, R_CONVERGENCE=1e-11, MAXITER=200, T_CUT_ITER=0.)
    driver = LocalDLPNO
    if settings['energy_backend'] == 'gpu':
        from electrolites.dlpno.local_gpu import LocalGPUDLPNO
        from electrolites.dlpno.mgpu import devices
        if len(devices()) != 1:
            raise ValueError('GPU topology validation requires DLPNO_NGPU=1 to trace every PAO selection')
        driver = LocalGPUDLPNO
    driver = tracked_driver(driver)
    keywords = dict(pno=settings['pno'], triples=base.triples, frozen_core=base.frozen_core,
                    auxbasis=base.auxbasis, lmo_cut=settings['coefficient_cut'],
                    pao_cut=settings['coefficient_cut'], ri_backend=settings['ri_backend'],
                    cache_bytes=settings['cache_mib'] << 20,
                    dipole_tail_options=settings.get('dipole_tail_options'),
                    local_screening_options=settings.get('local_screening_options'),
                    screen_storage=settings.get('screen_storage', 'auto'), verbose=0, **options)
    directions = [tuple(map(int, x.split(','))) for x in args.directions]
    if any(len(d) != 2 or not 0 <= d[0] < mol.natm or not 0 <= d[1] < 3 for d in directions):
        raise ValueError('Invalid atom/Cartesian direction')
    if any(h <= 0 for h in args.steps):
        raise ValueError('Finite-difference steps must be positive')
    report = dict(complete=False, phase='reference_energy', settings=vars(args),
                  source_sha256=hashes, reference_sha256=file_sha256(args.reference),
                  checkpoint_sha256=file_sha256(args.checkpoint),
                  validator_sha256=file_sha256(__file__), units='Eh and Eh/bohr',
                  reference_energy_validator_sha256=file_sha256(inspect.getfile(reference_energy)),
                  natm=mol.natm, nao=mol.nao, rows=[])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    def save():
        report['seconds'] = time.perf_counter()-start
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(output)

    def release():
        gc.collect()
        if settings['energy_backend'] == 'gpu' or args.scf_backend == 'gpu':
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()

    del base
    release()
    save()

    def evaluate(m, reference=False):
        local_start = time.perf_counter()
        residuals = None
        if reference:
            mf, cl, localization = mf0, cl0, None
        else:
            mf = scf.RHF(m).density_fit(auxbasis=mf0.with_df.auxbasis)
            if args.scf_backend == 'gpu':
                mf = mf.to_gpu()
            mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = (
                args.scf_energy_tolerance, args.scf_gradient_tolerance, 200)
            occupied = scf.addons.project_mo_nr2nr(mol, mf0.mo_coeff[:, :nocc], m)
            density = 2*occupied@occupied.T
            if args.scf_backend == 'gpu':
                import cupy
                density = cupy.asarray(density)
            mf.kernel(dm0=density)
            actual_grad = mf.get_grad(mf.mo_coeff, mf.mo_occ,
                                      mf.get_fock(dm=mf.make_rdm1()))
            if hasattr(actual_grad, 'get'):
                actual_grad = actual_grad.get()
            residuals = dict(maximum=float(np.max(abs(actual_grad), initial=0.)),
                             norm=float(np.linalg.norm(actual_grad)))
            if not mf.converged:
                report['failed_scf'] = dict(phase=report['phase'], energy=float(mf.e_tot),
                                           residuals=residuals)
                save()
                raise RuntimeError('Displaced SCF did not converge')
            if args.scf_backend == 'gpu':
                mf = mf.to_cpu()
            active = mf.mo_coeff[:, ncore:nocc]
            cross = gto.intor_cross('int1e_ovlp', mol, m)
            u, _, vt = np.linalg.svd(active.T@cross.T@cl0)
            guess = active@u@vt
            localizer = lo.Boys(m, guess)
            localizer.init_guess = None
            localizer.conv_tol, localizer.conv_tol_grad, localizer.max_cycle = (
                1e-13, args.boys_gradient_tolerance, 200)
            localizer.verbose = 0
            cl = stationary_boys(m, localizer.kernel(), tol=args.boys_gradient_tolerance)
            cl, localization = minimum_boys(m, cl, tol=args.boys_gradient_tolerance)
            overlap = cl0.T@cross@cl
            rows, columns = linear_sum_assignment(-abs(overlap))
            columns = columns[np.argsort(rows)]
            matched = overlap[np.arange(len(columns)), columns]
            if np.min(abs(matched)) < .9:
                raise RuntimeError('Displaced Boys solution left the reference branch')
            cl = cl[:, columns]*np.sign(matched)
        point_path = output.with_name(output.stem+'-'+report['phase'].replace(':', '_'))
        orbital_record = None
        if args.save_orbitals:
            orbital_path = Path(str(point_path)+'-orbitals.npz')
            np.savez(orbital_path, mo_coeff=mf.mo_coeff, mo_energy=mf.mo_energy,
                     mo_occ=mf.mo_occ, local=cl, coordinates=m.atom_coords())
            orbital_record = dict(path=str(orbital_path), sha256=file_sha256(orbital_path))
            orbital_path.with_suffix('.json').write_text(json.dumps(dict(
                mol_dump=m.dumps(), scf_auxbasis=mf.with_df.auxbasis,
                scf_energy=float(mf.e_tot), converged=bool(mf.converged),
                scf_conv_tol=mf.conv_tol, scf_conv_tol_grad=mf.conv_tol_grad,
                residuals=residuals, orbital_input=orbital_record), indent=2)+'\n')
        state = driver(mf, c_lmo=cl, **keywords)
        state.kernel()
        hf_check = reference_energy(mf, args.hf_energy_backend)
        if abs(hf_check['correction']) > 1e-8:
            raise ValueError('Independent HF energy differs by more than numerical precision')
        record = dict(energy=dict(state.e), topology=topology(state),
                      effective_topology=topology(state, True), localization=localization,
                      scf_residuals=residuals, orbital_input=orbital_record,
                      hf_energy_check=hf_check,
                      validation_energy=state.e['tot']+hf_check['correction'],
                      seconds=time.perf_counter()-local_start)
        if args.save_points:
            point_file = Path(str(point_path)+'-energy.json')
            temporary = Path(str(point_file)+'.tmp')
            temporary.write_text(json.dumps(dict(
                reference_sha256=report['reference_sha256'], source_sha256=hashes,
                validator_sha256=report['validator_sha256'], settings=vars(args),
                phase=report['phase'], record=record), indent=2)+'\n')
            temporary.replace(point_file)
            record['saved_point'] = dict(path=str(point_file), sha256=file_sha256(point_file))
        del state
        release()
        return record

    central = evaluate(mol, reference=True)
    report['central'] = central
    report['central_energy_error'] = central['energy']['tot']-previous['energy']['tot']
    if abs(report['central_energy_error']) > 1e-8:
        save()
        raise ValueError('The tighter reference energy differs from the saved energy')
    gradient = np.asarray(previous['gradient'])
    for atom, xyz in directions:
        row = dict(atom=atom, xyz=xyz, analytic=float(gradient[atom, xyz]), differences=[])
        report['rows'].append(row)
        for step in args.steps:
            displaced = []
            for sign in (1, -1):
                report['phase'] = f'energy:{atom},{xyz}:{sign*step:g}'
                save()
                coordinates = mol.atom_coords()
                coordinates[atom, xyz] += sign*step
                displaced.append(evaluate(mol.copy().set_geom_(coordinates, unit='Bohr')))
            plus, minus = displaced
            native = (plus['energy']['tot']-minus['energy']['tot'])/(2*step)
            numerical = (plus['validation_energy']-minus['validation_energy'])/(2*step)
            stable = plus['effective_topology'] == central['effective_topology'] == minus['effective_topology']
            item = dict(step=step, numerical=numerical, error=row['analytic']-numerical,
                        original_scf_numerical=native, original_scf_error=row['analytic']-native,
                        topology_stable=stable, plus=plus, minus=minus)
            row['differences'].append(item)
            save()
            print(f'{atom},{xyz} h={step:g} error={item["error"]:.3e} topology={stable}', flush=True)
    report['complete'] = True
    report['phase'] = 'complete'
    report['passed'] = all(d['topology_stable'] and abs(d['error']) < args.tolerance
                           for row in report['rows'] for d in row['differences'])
    save()
    if not report['passed']:
        raise RuntimeError('Independent finite-difference validation failed; inspect the saved evidence')


if __name__ == '__main__':
    main()
