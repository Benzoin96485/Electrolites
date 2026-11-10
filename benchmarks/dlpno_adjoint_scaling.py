"""Connected alkane scaling with phase, domain, graph and solver evidence.

Run each size in a separate process for meaningful peak-RSS measurements.
Output belongs in scratch; fixed thresholds are serialized in each report.
No exponent is asserted by this runner.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
from pyscf import gto, scf

from electrolites.dlpno.local import LocalDLPNO


def alkane(carbons):
    """An extended, slightly distorted, connected C_n H_(2n+2) chain."""
    if carbons < 2:
        raise ValueError('Use at least two carbon atoms')
    a, b = np.sqrt(2/3), np.sqrt(1/3)
    carbon = np.array([[1.54*a*i, 1.54*b*(i % 2), 0.] for i in range(carbons)])
    atoms = [('C', r) for r in carbon]
    for i, center in enumerate(carbon):
        neighbors = [carbon[j]-center for j in (i-1, i+1) if 0 <= j < carbons]
        neighbors = [v/np.linalg.norm(v) for v in neighbors]
        if len(neighbors) == 2:
            outward = -sum(neighbors)
            outward /= np.linalg.norm(outward)
            normal = np.cross(*neighbors)
            normal /= np.linalg.norm(normal)
            hydrogens = [b*outward+sign*a*normal for sign in (-1, 1)]
        else:
            bond = neighbors[0]
            u = np.cross(bond, [0., 0., 1.])
            u /= np.linalg.norm(u)
            v = np.cross(bond, u)
            hydrogens = [-bond/3+np.sqrt(8/9)*(np.cos(t)*u+np.sin(t)*v)
                         for t in (0., 2*np.pi/3, 4*np.pi/3)]
        atoms.extend(('H', center+1.09*direction) for direction in hydrogens)
    return [(symbol, xyz+.002*np.sin(np.arange(3)+index*.73))
            for index, (symbol, xyz) in enumerate(atoms)]


def file_sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_primal(path, base, report):
    """Durable trusted-local checkpoint, with the matching source manifest."""
    import cloudpickle
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    fitting = getattr(base.mf, 'with_df', None)
    optimizer = getattr(fitting, 'intopt', None)
    transient = (fitting is not None and type(fitting).__module__.startswith('pyscf.')
                 and type(optimizer).__module__.startswith('gpu4pyscf.'))
    # GPU4PySCF.to_cpu may retain its GPU integral optimizer on a CPU DF
    # object. Its ctypes pointers are runtime state, not reference data;
    # CPU DF rebuilds its own caches after loading. Restore the live cache.
    if transient:
        del fitting.intopt
    try:
        with temporary.open('wb') as handle:
            cloudpickle.dump(base, handle, protocol=5)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if transient:
            fitting.intopt = optimizer
    temporary.replace(path)
    manifest = dict(sha256=file_sha256(path), bytes=path.stat().st_size,
                    primal_report=report, code_sha256=report['code_sha256'])
    target = Path(str(path)+'.json')
    temporary = Path(str(target)+'.tmp')
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    temporary.replace(target)
    return dict(path=str(path), sha256=manifest['sha256'], bytes=manifest['bytes'])


def load_primal(path, report):
    import cloudpickle
    path = Path(path)
    manifest = json.loads(Path(str(path)+'.json').read_text())
    if manifest['code_sha256'] != report['code_sha256'] or manifest['sha256'] != file_sha256(path):
        raise ValueError('Energy checkpoint source or data hash does not match')
    previous = manifest['primal_report']
    for name in ('carbons', 'basis', 'pno', 'triples', 'coefficient_cut', 'option', 'cache_mib',
                 'scf_backend', 'energy_backend', 'ri_backend',
                 'dipole_tail_options', 'local_screening_options', 'screen_storage'):
        if previous['settings'].get(name) != report['settings'].get(name):
            raise ValueError(f'Energy checkpoint setting differs: {name}')
    with path.open('rb') as handle:
        base = cloudpickle.load(handle)
    if not base.mf.converged or not np.isfinite(base.mf.mo_coeff).all() or not all(np.isfinite(v) for v in base.e.values()):
        raise ValueError('Invalid converged reference in energy checkpoint')
    return base, previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--carbons', type=int, required=True)
    parser.add_argument('--basis', default='sto-3g')
    parser.add_argument('--pno', default='loose')
    parser.add_argument('--triples', choices=['none', 't0', 't1'], default='t0')
    parser.add_argument('--backend', choices=['numpy', 'cupy'], default='numpy')
    parser.add_argument('--scf-backend', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--energy-backend', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--ri-backend', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--response-backend', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--triples-backend', choices=['numpy', 'cupy'], default=None)
    parser.add_argument('--ccsd-primal-backend', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--ccsd-adjoint-backend', choices=['cupy'], default=None)
    parser.add_argument('--response-checkpoint', default=None)
    parser.add_argument('--triples-primal-backend', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--krylov-memory-mib', type=float, default=8192)
    parser.add_argument('--checkpoint-triples', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--checkpoint-integrals', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--checkpoint-ccsd', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--checkpoint-pnos', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--coefficient-cut', type=float, default=1e-4)
    parser.add_argument('--cache-mib', type=int, default=64)
    parser.add_argument('--dipole-tail-options', type=json.loads,
                        help='JSON object enabling the complete hierarchical tail')
    parser.add_argument('--local-screening-options', type=json.loads,
                        help='JSON object enabling local DOI and centered local transition dipoles')
    parser.add_argument('--screen-storage', choices=['auto', 'sparse', 'dense'], default='auto')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--energy-checkpoint', help='Save the converged energy state before the gradient')
    parser.add_argument('--resume-energy', help='Resume a hash-verified checkpoint generated by this runner')
    parser.add_argument('--energy-only', action='store_true')
    parser.add_argument('--option', action='append', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    for name in ('dipole_tail_options', 'local_screening_options'):
        value = getattr(args, name)
        if value is not None and not isinstance(value, dict):
            parser.error(name+' must be a JSON object')
    if args.local_screening_options is not None and args.dipole_tail_options is None:
        parser.error('--local-screening-options requires --dipole-tail-options')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(settings=vars(args), complete=False, phase='scf', started=time.time())
    report['runtime'] = dict(python=sys.version, node=platform.node(),
                             slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                             omp_threads=os.environ.get('OMP_NUM_THREADS'),
                             openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'))
    report['validator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    code = Path(__file__).resolve().parents[1]/'electrolites'/'dlpno'
    report['code_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')}
    gpu_enabled = args.backend == 'cupy' or 'gpu' in (
        args.scf_backend, args.energy_backend, args.ri_backend, args.response_backend)
    if gpu_enabled:
        import cupy as cp
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        report['runtime']['gpu'] = dict(name=properties['name'].decode(),
                                         memory_bytes=properties['totalGlobalMem'])
        report['gpu_memory_observations'] = []

    def memory_observation():
        if gpu_enabled:
            pool = cp.get_default_memory_pool()
            report['gpu_memory_observations'].append(dict(
                phase=report['phase'], used_bytes=pool.used_bytes(), reserved_bytes=pool.total_bytes()))

    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(path)

    save()
    mol = gto.M(atom=alkane(args.carbons), basis=args.basis, verbose=0)
    if args.resume_energy:
        base, previous = load_primal(args.resume_energy, report)
        if not np.array_equal(base.mol.atom_charges(), mol.atom_charges()) or np.max(abs(base.mol.atom_coords()-mol.atom_coords())) > 1e-12:
            raise ValueError('Checkpoint molecule does not match the requested geometry')
        mf = base.mf
        start = time.perf_counter()
        mf.with_df.build()  # Regenerate node-local DF files outside the gradient timer.
        report['resume_df_warmup_seconds'] = time.perf_counter()-start
        report['resumed_from'] = args.resume_energy
        report['energy_original_runtime'] = previous['runtime']
        report['scf_seconds'] = previous['scf_seconds']
        energy_seconds = previous['energy_seconds']
    else:
        start = time.perf_counter()
        mf = scf.RHF(mol).density_fit()
        if args.scf_backend == 'gpu':
            mf = mf.to_gpu()
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-9, 150
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('SCF did not converge')
        report.update(scf_seconds=time.perf_counter()-start, phase='energy')
        if args.scf_backend == 'gpu':
            mf = mf.to_cpu()
        memory_observation()
        save()
        options = dict(E_CONVERGENCE=1e-11, R_CONVERGENCE=1e-10, MAXITER=150, T_CUT_ITER=0.)
        for item in args.option:
            key, value = item.split('=', 1)
            options[key] = json.loads(value)
        driver = LocalDLPNO
        if args.energy_backend == 'gpu':
            from electrolites.dlpno.local_gpu import LocalGPUDLPNO
            driver = LocalGPUDLPNO
        base = driver(mf, pno=args.pno, triples=None if args.triples == 'none' else args.triples,
                          ri_backend=args.ri_backend,
                          lmo_cut=args.coefficient_cut, pao_cut=args.coefficient_cut,
                          dipole_tail_options=args.dipole_tail_options,
                          local_screening_options=args.local_screening_options,
                          screen_storage=args.screen_storage,
                          cache_bytes=args.cache_mib << 20, verbose=args.verbose, **options)
        start = time.perf_counter()
        original_tick = base._tick

        def energy_progress(stage, previous):
            result = original_tick(stage, previous)
            report.update(phase='energy:'+stage, energy_timings=dict(base.timings))
            memory_observation()
            save()
            return result

        base._tick = energy_progress
        try:
            base.kernel()
        except Exception as error:
            report.update(failed_phase=report['phase'], phase='failed',
                          error=type(error).__name__+': '+str(error))
            save()
            raise
        finally:
            del base._tick
        energy_seconds = time.perf_counter()-start
    report.update(energy_seconds=energy_seconds, energy=base.e, energy_timings=base.timings,
                  natm=mol.natm, nao=mol.nao, geometry_bohr=mol.atom_coords().tolist(),
                  energy_integrals=base.local_integrals.statistics(),
                  localization=base._orbital_reference.localization_info,
                  options=base.opt, occupied=base.n, pairs=len(base.pl), triplets=len(getattr(base, 'trip', [])),
                  pair_domain_max=max(map(len, base.pao)), pair_domain_total=sum(map(len, base.pao)),
                  pno_max=max(x.shape[1] for x in base.X),
                  pair_fit_max=max(map(len, base.ribfs)),
                  triplet_pao_max=max(map(len, getattr(base, 't_paos', [])), default=0),
                  triplet_fit_max=max(map(len, getattr(base, 't_ribfs', [])), default=0),
                  tno_max=(max((x.shape[1] for x in base.X_tno), default=0)
                           if hasattr(base, 'X_tno') else None),
                  neighbors_max=max(len(base.pl.neighbors(i)) for i in range(base.n)), phase='gradient')
    report['preparation'] = {key: getattr(base, key) for key in (
        'screening_diagnostics', 'dipole_preparation_diagnostics',
        'dipole_prescreen_diagnostics', 'dipole_tail_diagnostics') if hasattr(base, key)}
    memory_observation()
    save()
    if args.energy_checkpoint:
        start = time.perf_counter()
        report['energy_checkpoint'] = save_primal(args.energy_checkpoint, base, dict(report))
        report['energy_checkpoint']['seconds'] = time.perf_counter()-start
        save()
    if args.energy_only:
        report['phase'] = 'energy_complete'
        save()
        return
    start = time.perf_counter()
    def progress(event):
        report['phase'] = event['event']+':'+event['stage']
        report.setdefault('gradient_progress', []).append(dict(event, elapsed=time.perf_counter()-start))
        memory_observation()
        save()

    gradient = base.nuc_grad_method(backend=args.backend, response_backend=args.response_backend,
                                    checkpoint_triples=args.checkpoint_triples,
                                    checkpoint_integrals=args.checkpoint_integrals, progress=progress,
                                    triples_backend=args.triples_backend,
                                    krylov_memory_mib=args.krylov_memory_mib,
                                    ccsd_primal_backend=args.ccsd_primal_backend,
                                    ccsd_adjoint_backend=args.ccsd_adjoint_backend,
                                    response_checkpoint=args.response_checkpoint,
                                    checkpoint_ccsd=args.checkpoint_ccsd,
                                    triples_primal_backend=args.triples_primal_backend,
                                    checkpoint_pnos=args.checkpoint_pnos)
    try:
        force = gradient.kernel()
    except Exception as error:
        report.update(failed_phase=report['phase'], phase='failed',
                      error=type(error).__name__+': '+str(error))
        save()
        raise
    report.update(gradient_seconds=time.perf_counter()-start, gradient=force.tolist(),
                  gradient_timings=gradient.timings, diagnostics=gradient.diagnostics,
                  component_torques={k: np.cross(mol.atom_coords(), v).sum(axis=0).tolist()
                                     for k, v in gradient.components.items()},
                  translation_residual=np.max(abs(force.sum(axis=0))).item(),
                  rotation_residual=np.max(abs(np.cross(mol.atom_coords(), force).sum(axis=0))).item(),
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  complete=True, phase='complete')
    report['passed'] = (report['translation_residual'] < 1e-7 and report['rotation_residual'] < 1e-7)
    memory_observation()
    save()
    print(json.dumps({k: report[k] for k in ('natm', 'nao', 'pairs', 'triplets', 'energy_seconds',
                                            'gradient_seconds', 'peak_rss_kib')}, indent=2), flush=True)
    if not report['passed']:
        np.savez_compressed(path.with_suffix('.densities.npz'),
                            **gradient.relaxed_densities, geometry=mol.atom_coords(),
                            mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ, mo_energy=mf.mo_energy,
                            local_ri=gradient.components['local_ri'], hf=gradient.components['hf'])
        raise RuntimeError('Rigid-motion invariance failed; inspect the saved gradient report')


if __name__ == '__main__':
    main()
