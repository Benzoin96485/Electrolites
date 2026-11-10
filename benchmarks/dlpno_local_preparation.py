"""Compare spatial domain/dipole preparation with its dense oracle.

The dense diagnostic comparisons below are outside the reported local
preparation timers. This audit does not certify total-gradient accuracy
or the scaling of global orbital preparation/relaxation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pyscf import gto, scf
from pyscf.scf import chkfile

from benchmarks.dlpno_adjoint_scaling import alkane
from electrolites.dlpno.local import LocalDLPNO
from electrolites.dlpno import setup as su
from electrolites.dlpno.dipole_tail import orbital_inputs
from electrolites.dlpno.dipole_laplace import exact_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--carbons', required=True, type=int)
    parser.add_argument('--reference-npz', help='Hash-manifested numeric reference exported from a verified parent')
    parser.add_argument('--scf-checkpoint', help='PySCF checkpoint with matching geometry and basis')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.reference_npz and args.scf_checkpoint:
        parser.error('Use one reference source')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parents[1]/'electrolites'/'dlpno'
    report = dict(complete=False, full_gradient_validated=False, settings=vars(args), rows=[],
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')})
    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n'); temporary.replace(path)
    save()
    mol = gto.M(atom=alkane(args.carbons), basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    c_lmo = None
    if args.reference_npz:
        source = Path(args.reference_npz)
        manifest = json.loads(source.with_suffix('.json').read_text())
        if not manifest['complete'] or hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
            raise ValueError('Reference data hash differs from its manifest')
        data = np.load(source, allow_pickle=False)
        if not np.array_equal(data['charges'], mol.atom_charges()) or np.max(abs(data['geometry']-mol.atom_coords())) > 1e-12:
            raise ValueError('Reference molecule differs')
        for key in ('mo_coeff', 'mo_occ', 'mo_energy'):
            setattr(mf, key, data[key])
        mf.e_tot, mf.converged, c_lmo = float(data['e_tot']), True, data['c_lmo']
        report['reference_manifest'] = manifest
    elif args.scf_checkpoint:
        source = Path(args.scf_checkpoint)
        old = chkfile.load_mol(source)
        if old.nao != mol.nao or old._basis != mol._basis or np.max(abs(old.atom_coords()-mol.atom_coords())) > 1e-12:
            raise ValueError('SCF checkpoint molecule differs')
        data = chkfile.load(source, 'scf')
        for key in ('mo_coeff', 'mo_occ', 'mo_energy', 'e_tot'):
            setattr(mf, key, data[key])
        mf.converged = True
        report['reference_checkpoint_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    else:
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        mf.kernel()
    residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, mf.get_fock())))
    if not mf.converged or residual > 1e-8:
        raise ValueError(f'Nonstationary reference: {residual}')
    report['scf_residual'] = residual
    report['phase'] = 'dense_oracle'; save()
    oracle = LocalDLPNO(mf, c_lmo=c_lmo, triples='t0', pno='loose', verbose=0)
    start = time.perf_counter(); oracle.setup()
    report['dense_setup_seconds'] = time.perf_counter()-start
    report['dense_timings'] = oracle.timings
    orb = oracle._orbital_reference
    report['localization'] = orb.localization_info
    with mol.with_common_orig((0., 0., 0.)):
        position = mol.intor('int1e_r', comp=3)
    def inputs(state):
        o = state._orbital_reference
        return orbital_inputs(state, dict(s=o.S, cl=o.C_lmo, p=o.C_pao,
            sp=o.S_pao, fp=o.F_pao, fl=o.F_lmo), position)
    original_inputs = inputs(oracle)
    original_mask = ((oracle.doi_ij > oracle.opt['T_CUT_DO_IJ'])
                     | (abs(oracle.dip_b) > oracle.opt['T_CUT_PRE']) | np.eye(oracle.n, dtype=bool))
    original_exclusions = tuple((i, j) for i, j in su.mask_pairs(original_mask) if i < j)
    original_tail = float(np.sum(oracle.dip_e[~original_mask]))
    for settings in ({}, dict(coefficient_cut=1e-10, grid_radius=25., partition_radius=60.)):
        report['phase'] = 'local_setup'; save()
        local = LocalDLPNO(mf, c_lmo=orb.C_lmo, triples='t0', pno='loose', verbose=0,
                           dipole_tail_options={}, local_screening_options=settings)
        start = time.perf_counter(); local.setup(); elapsed = time.perf_counter()-start
        row = dict(settings=settings, setup_seconds=elapsed, timings=local.timings,
                   screening=local.screening_diagnostics, dipole_preparation=local.dipole_preparation_diagnostics,
                   prescreen=local.dipole_prescreen_diagnostics, tree=local.dipole_tail_diagnostics)
        # All comparisons below are diagnostic work, not local timings.
        mask = local._initial_pair_mask.to_dense()
        row['initial_pair_mask_changes'] = int(np.count_nonzero(mask != original_mask))
        row['pao_pair_mask_changes'] = int(np.count_nonzero(local.pao_pair_mask.to_dense() != oracle.pao_pair_mask))
        row['doi_threshold_changes'] = {}
        for name, thresholds in (
            ('doi_ij', ['T_CUT_DO_IJ']), ('doi_uv', ['T_CUT_DO_UV']),
            ('doi_iu', ['T_CUT_DO_PRE', 'T_CUT_DO', 'T_CUT_DO_TRIPLES_PRE', 'T_CUT_DO_TRIPLES']),
            ('mkn', ['T_CUT_MKN', 'T_CUT_MKN_TRIPLES_PRE', 'T_CUT_MKN_TRIPLES'])):
            sparse, dense = getattr(local, name), getattr(oracle, name)
            for key in thresholds:
                row['doi_threshold_changes'][name+':'+key] = int(np.count_nonzero(
                    (abs(sparse) > local.opt[key]).to_dense() != (abs(dense) > oracle.opt[key])))
        current = inputs(local)
        row['center_max_error'] = float(np.max(abs(current[0]-original_inputs[0])))
        row['dipole_max_error'] = max((float(np.max(abs(a-b))) if a.shape == b.shape else float('inf'))
                                    for a, b in zip(current[1], original_inputs[1]))
        row['gap_max_error'] = max((float(np.max(abs(a-b))) if a.shape == b.shape else float('inf'))
                                 for a, b in zip(current[2], original_inputs[2]))
        exclusions = frozenset(original_exclusions)
        exact_local = sum(exact_pair(current[1][i], current[2][i], current[1][j], current[2][j],
                                    current[0][i]-current[0][j], derivatives=False)[0]
                          for i in range(local.n) for j in range(i) if (j, i) not in exclusions)
        row.update(oracle_tail=original_tail, local_dipole_tail=exact_local,
                   local_dipole_energy_error=exact_local-original_tail,
                   combined_tail_energy_error=local._distant_tail_energy-original_tail,
                   all_selections_identical=not (row['initial_pair_mask_changes'] or row['pao_pair_mask_changes']
                                                or any(row['doi_threshold_changes'].values())))
        report['rows'].append(row); save()
        print(json.dumps({key: row[key] for key in ('settings', 'setup_seconds', 'all_selections_identical',
            'center_max_error', 'dipole_max_error', 'local_dipole_energy_error', 'combined_tail_energy_error')}), flush=True)
    report.update(complete=True, phase='complete'); save()


if __name__ == '__main__':
    main()
