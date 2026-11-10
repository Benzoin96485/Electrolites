"""Connected-chain orbital/domain locality audit, without CC amplitudes.

This is a structural bound audit, not a substitute for complete-gradient
timings or a correlation-energy accuracy test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pyscf import gto, lib, scf
from pyscf.scf import chkfile
from threadpoolctl import threadpool_limits

from benchmarks.dlpno_adjoint_scaling import alkane
from electrolites.dlpno.local import LocalDLPNO
from electrolites.dlpno.local_integrals import CoefficientMaps
from electrolites.dlpno.reference_cpu import boys_curvature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[8, 16, 32, 64])
    parser.add_argument('--scf-threads', type=int, default=1)
    parser.add_argument('--scf-checkpoint', help='Resume one size from a verified RHF checkpoint')
    parser.add_argument('--newton', action='store_true', help='Use the RHF second-order optimizer')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.scf_threads < 1 or (args.scf_checkpoint and len(args.sizes) != 1):
        parser.error('Use positive SCF threads and a single size when resuming a checkpoint')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parents[1]/'electrolites'/'dlpno'
    report = dict(complete=False, rows=[], description=__doc__, settings=vars(args),
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')})

    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(path)

    save()
    for size in args.sizes:
        start = time.perf_counter()
        mol = gto.M(atom=alkane(size), basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-9, 150
        solver = mf.newton() if args.newton else mf
        if args.newton:
            # Total HF energies of a large chain have a several-ulp
            # noise floor near 1e-12.  The orbital residual remains the
            # tight 1e-9 reference criterion.
            solver.conv_tol, solver.conv_tol_grad = 1e-11, 1e-9
            # The default AH linear-dependence cutoff can drop search
            # vectors before a 1e-9 orbital-gradient target is reached.
            solver.ah_lindep, solver.ah_conv_tol = 1e-28, 1e-22
            solver.max_cycle = 50
        initial = {}
        if args.scf_checkpoint:
            old = chkfile.load_mol(args.scf_checkpoint)
            if old.nao != mol.nao or np.max(abs(old.atom_coords()-mol.atom_coords())) > 1e-10:
                raise ValueError('Checkpoint geometry or basis size does not match this chain')
            state = chkfile.load(args.scf_checkpoint, 'scf')
            if args.newton:
                initial = dict(mo_coeff=state['mo_coeff'], mo_occ=state['mo_occ'])
            else:
                initial = dict(dm0=(state['mo_coeff']*state['mo_occ'])@state['mo_coeff'].T)
            report['checkpoint_sha256'] = hashlib.sha256(Path(args.scf_checkpoint).read_bytes()).hexdigest()
        solver.chkfile = str(path.with_suffix(f'.c{size}.chk'))
        report.update(phase='scf', current_carbons=size)
        save()
        def scf_progress(state):
            report.setdefault('scf_progress', []).append(
                dict(carbons=size, iteration=int(state.get('imacro', state.get('cycle', -1))),
                     energy=float(state['e_tot']), gradient_norm=float(state.get('norm_gorb', -1.))))
            save()
        solver.callback = scf_progress
        previous_threads = lib.num_threads()
        lib.num_threads(args.scf_threads)
        try:
            with threadpool_limits(limits=args.scf_threads):
                solver.kernel(**initial)
                for name in ('mo_coeff', 'mo_occ', 'mo_energy', 'e_tot', 'converged'):
                    setattr(mf, name, getattr(solver, name))
                residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, mf.get_fock())))
        finally:
            lib.num_threads(previous_threads)
        report.update(scf_converged=bool(mf.converged), scf_orbital_residual=residual,
                      scf_seconds=time.perf_counter()-start, phase='domains')
        save()
        if not mf.converged or residual > 1e-8:
            raise RuntimeError(f'RHF reference is not stationary: {residual:.3e}')
        base = LocalDLPNO(mf, pno='loose', triples='t0', verbose=0)
        base.setup()
        orb, opt = base._orbital_reference, base.opt
        curvature, _ = boys_curvature(mol, orb.C_lmo)
        dom = base.domains(opt['T_CUT_MKN'], opt['T_CUT_DO'])
        mask = ((base.doi_ij > opt['T_CUT_DO_IJ']) | (abs(base.dip_b) > opt['T_CUT_PRE'])
                | np.eye(base.n, dtype=bool))
        with mol.with_common_orig((0., 0., 0.)):
            r, r2 = mol.intor('int1e_r', comp=3), mol.intor('int1e_r2')
        centers = np.einsum('mi,xmn,ni->ix', orb.C_lmo, r, orb.C_lmo, optimize=True)
        spread = np.einsum('mi,mn,ni->i', orb.C_lmo, r2, orb.C_lmo, optimize=True)-(centers**2).sum(axis=1)
        supports = {}
        for cut in (1e-4, 3e-4, 1e-3, 3e-3):
            maps = CoefficientMaps(base, cut, cut)
            supports[str(cut)] = {}
            for kind, domains in maps.atoms.items():
                counts = [sum(len(base.atom_bfs[int(a)]) for a in atoms) for atoms in domains]
                supports[str(cut)][kind] = dict(maximum=max(counts), total=sum(counts),
                                                mean=float(np.mean(counts)))
        occupied = int(np.count_nonzero(mf.mo_occ))
        row = dict(carbons=size, natm=mol.natm, nao=mol.nao, occupied=base.n,
                   scf_seconds=report['scf_seconds'], scf_orbital_residual=residual,
                   scf_gap=float(mf.mo_energy[occupied]-mf.mo_energy[occupied-1]),
                   boys_curvature=curvature, localization=orb.localization_info,
                   lmo_spread_max_bohr2=float(max(spread)),
                   prescreen_pairs=int(mask.sum()), prescreen_neighbors_max=int(mask.sum(axis=1).max()),
                   lmo_pao_domain_max=max(map(len, dom.lmo_paos)),
                   lmo_fitting_domain_max=max(map(len, dom.lmo_ribfs)),
                   supports=supports, setup_timings=base.timings, seconds=time.perf_counter()-start)
        report['rows'].append(row)
        save()
        print(json.dumps(row), flush=True)
    report['complete'] = True
    report['phase'] = 'complete'
    save()


if __name__ == '__main__':
    main()
