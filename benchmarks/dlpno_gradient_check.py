"""Independent nuclear finite differences for the analytic DLPNO response.

Run with --output pointing outside the checkout.  Geometry displacements,
gradients and finite-difference steps all use atomic units.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pyscf import gto, scf
from scipy.optimize import linear_sum_assignment

from electrolites.dlpno.reference_cpu import CPUDLPNO, orbitals
from electrolites.dlpno.orbital_response import RHFResponse
from electrolites.dlpno.response import ResponseDLPNO
from electrolites.dlpno.setup import count_frozen_core


CASES = {
    'water': 'O 0 0 0; H 0 .76 .59; H .07 -.75 .57',
    'ammonia': 'N .01 -.03 .1; H .1 .94 -.3; H .81 -.44 -.35; H -.84 -.49 -.31',
    'hf': 'H 0 0 0; F .05 .03 .92',
    'water_dimer': ('O 0 0 0; H 0 .76 .59; H .07 -.75 .57; '
                    'O 4.3 .21 .12; H 4.8 .86 .56; H 4.58 -.64 .49'),
}
STRESS = dict(T_CUT_PNO=1e-4, T_CUT_PNO_MP2=1e-5,
              T_CUT_TRACE=0., T_CUT_ENERGY=0., T_CUT_TRACE_MP2=0., T_CUT_ENERGY_MP2=0.,
              MIN_PNOS=1, MIN_TNOS=1, T_CUT_TNO=1e-5, T_CUT_TNO_PRE=1e-4)


def topology(d, effective=False):
    events = getattr(d, '_selection_events', [])
    orthogonalizations = getattr(d, '_orthogonalization_events', [])
    if effective:
        # Triplet priorities enter the energy only through the resulting
        # retained TNO ranks.  Swapping a priority without changing any
        # retained rank leaves the continuous energy branch unchanged.
        events = [e[:2] if e[0] == 'tnos' else e for e in events]
        # Every PAO lies in the global virtual space. An independent set
        # of nvirt columns therefore spans that entire space, even when
        # the domain contains fewer than all AO columns. Changing these
        # pivots is a basis change, not a changed truncation. Use the
        # actual returned orthogonal-basis rank (recorded after GPU eigh).
        nvirt = d.mol.nao-int(np.count_nonzero(d.mf.mo_occ))
        orthogonalizations = [
            [entry[0], 'complete_virtual_span', nvirt]
            if len(entry) == 3 and len(entry[1]) == entry[2] == nvirt else entry
            for entry in orthogonalizations]
    return dict(pairs=[list(map(int, p)) for p in d.pl.pairs], strong=np.asarray(d.strong).tolist(),
                paos=[p.tolist() for p in d.pao],
                pno=[x.shape[1] for x in d.X], triplets=[list(map(int, t)) for t in getattr(d, 'trip', [])],
                tno=[x.shape[1] for x in getattr(d, 'X_tno', [])],
                dipole_tree=getattr(getattr(d, '_dipole_tail_tree', None), 'signature', None),
                orthogonalizations=orthogonalizations,
                events=events)


def tracked_driver(cls):
    """Observe the energy driver's selections, without changing its algebra."""
    lazy_tno = hasattr(cls, '_tno_chunk')
    class Tracked(cls):
        def kernel(self):
            from electrolites.dlpno.setup import orthogonalization_trace
            self._orthogonalization_events = []
            token = orthogonalization_trace.set(self._orthogonalization_events)
            try:
                return super().kernel()
            finally:
                orthogonalization_trace.reset(token)

        def setup(self):
            super().setup()
            self._selection_events = []
            orbital_reference = getattr(self, '_orbital_reference', self.orb)
            for name, mask in [
                ('doi_uv', getattr(self, 'pao_pair_mask', self.doi_uv > self.opt['T_CUT_DO_UV'])),
                ('f_mp2', abs(orbital_reference.F_lmo) > self.opt['F_CUT']),
                ('f_triples', abs(orbital_reference.F_lmo) >= self.opt['F_CUT_T']),
                ('dipole_domains', abs(self.doi_iu) > self.opt['T_CUT_DO_PRE']),
            ]:
                self._selection_events.append([name, hashlib.sha256(mask.tobytes()).hexdigest()])
            if hasattr(self, 'local_integrals'):
                for name, supports in self.local_integrals.maps.atoms.items():
                    self._selection_events.append(['coefficient_support_'+name, [s.tolist() for s in supports]])
            if hasattr(self, '_local_dipole_maps'):
                for name, supports in self._local_dipole_maps.atoms.items():
                    self._selection_events.append(['dipole_coefficient_support_'+name,
                                                   [s.tolist() for s in supports]])

        def set_pairs(self, mask, dom):
            super().set_pairs(mask, dom)
            self._selection_events.append(['pairs', mask.tolist(),
                                           [x.tolist() for x in self.pao],
                                           [x.tolist() for x in self.ribfs]])

        def lmp2_pnos(self):
            super().lmp2_pnos()
            self._selection_events.append(['mp2_pnos', [x.shape[1] for x in self.X]])

        def recompute_pnos(self):
            super().recompute_pnos()
            self._selection_events.append(['cc_pnos', [x.shape[1] for x in self.X],
                                           self.strong.tolist()])

        def triples_sparsity(self, prescreen):
            super().triples_sparsity(prescreen)
            self._selection_events.append(['triples', prescreen,
                                           [list(map(int, x)) for x in self.trip],
                                           [x.tolist() for x in self.t_paos],
                                           [x.tolist() for x in self.t_ribfs]])

        def tno_transform(self, cut):
            super().tno_transform(cut)
            if lazy_tno:
                self._observed_tno_ranks = np.zeros(len(self.trip), dtype=int)
            else:
                self._selection_events.append(['tnos', [x.shape[1] for x in self.X_tno],
                                               self.tno_scale.tolist()])

        def _tno_chunk(self, indices, padded_size, cut):
            result = super()._tno_chunk(indices, padded_size, cut)
            self._observed_tno_ranks[indices] = result[2]
            return result

        def lccsd_t0(self, save=False):
            energy = super().lccsd_t0(save=save)
            if lazy_tno:
                self._selection_events.append(['tnos', self._observed_tno_ranks.tolist(),
                                               self.tno_scale.tolist()])
            return energy
    return Tracked


def frozen_orthogonalization_driver(pivots):
    """CPU local-energy oracle on a fixed PAO-pivot selection branch.

    Only the discrete column choices are reused.  SCF, Boys, integrals,
    PAO orthogonalization, PNO/TNO vectors and amplitudes are recomputed.
    Bind dependencies on this new class without modifying module globals.
    """
    from types import SimpleNamespace
    from electrolites.dlpno import setup as su
    from electrolites.dlpno.local import LocalDLPNO
    from electrolites.dlpno.ref_rhf import RefDLPNO
    from electrolites.dlpno.reference_cpu import rebind, dipole_pair_energies
    selections = iter(pivots)
    consumed = []

    def select(matrix, cut):
        try:
            n, columns = next(selections)[:2]
        except StopIteration as error:
            raise RuntimeError('Displacement changed the PAO orthogonalization call sequence') from error
        if matrix.shape != (n, n):
            raise RuntimeError('Displacement changed a PAO domain dimension')
        consumed.append(None)
        trace = su.orthogonalization_trace.get()
        if trace is not None:
            trace.append([n, list(columns), len(columns)])
        return list(columns)

    namespace = SimpleNamespace(**vars(su))
    namespace.orthocanonicalize = rebind(su.orthocanonicalize, pivoted_cholesky=select)
    dipoles = rebind(dipole_pair_energies, su=namespace)

    class Fixed(LocalDLPNO):
        setup = rebind(LocalDLPNO.setup, dipole_pair_energies=dipoles)
        _K_pao = rebind(RefDLPNO._K_pao, su=namespace)
        tno_transform = rebind(RefDLPNO.tno_transform, su=namespace)

        def kernel(self):
            result = super().kernel()
            if len(consumed) != len(pivots):
                raise RuntimeError('Displacement removed PAO orthogonalization calls')
            return result

    return tracked_driver(Fixed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=[*CASES, 'alkane'], default='water')
    parser.add_argument('--carbons', type=int, default=8, help='Chain length for --case alkane')
    parser.add_argument('--basis', default='6-31g')
    parser.add_argument('--pno', default='normal')
    parser.add_argument('--triples', choices=['none', 't0', 't1'], default='t1')
    parser.add_argument('--backend', choices=['cpu', 'gpu', 'local'], default='cpu')
    parser.add_argument('--method', choices=['direct', 'adjoint'], default='direct')
    parser.add_argument('--coefficient-cut', type=float, default=1e-4)
    parser.add_argument('--freeze-pao-pivots', action='store_true',
                        help='Local CPU validation of the continuous energy branch with fixed PAO column choices')
    parser.add_argument('--response-backend', choices=['numpy', 'cupy'])
    parser.add_argument('--stress', action='store_true')
    parser.add_argument('--screening-stress', action='store_true')
    parser.add_argument('--df-scf', action='store_true')
    parser.add_argument('--all-electron', action='store_true')
    parser.add_argument('--direction', default='all', help='all, or zero-based atom,Cartesian')
    parser.add_argument('--steps', nargs='+', type=float, default=[2e-4, 1e-4])
    parser.add_argument('--option', action='append', default=[], metavar='NAME=JSON',
                        help='Override an energy-driver option, e.g. T_CUT_PAIRS=0.01')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.freeze_pao_pivots and args.backend != 'local':
        parser.error('--freeze-pao-pivots requires --backend local')
    if args.response_backend is None:
        args.response_backend = 'cupy' if args.backend == 'gpu' else 'numpy'
    kwargs = dict(pno=args.pno, triples=None if args.triples == 'none' else args.triples,
                  frozen_core=not args.all_electron, verbose=0, E_CONVERGENCE=1e-11,
                  R_CONVERGENCE=1e-10, MAXITER=150, T_CUT_ITER=0.)
    if args.stress:
        kwargs.update(STRESS)
    if args.screening_stress:
        kwargs.update(T_CUT_DO_IJ=1e-4, T_CUT_PRE=1e-5, T_CUT_PAIRS_MP2=2e-5,
                      T_CUT_PAIRS=1e-3, T_CUT_TRIPLES_WEAK=1e-5)
    for item in args.option:
        name, value = item.split('=', 1)
        kwargs[name] = json.loads(value)
    cls = CPUDLPNO
    if args.backend == 'gpu':
        from electrolites.dlpno.rhf import DLPNO
        cls = DLPNO
    if args.backend == 'local':
        from electrolites.dlpno.local import LocalDLPNO
        cls = LocalDLPNO
        kwargs.update(lmo_cut=args.coefficient_cut, pao_cut=args.coefficient_cut)
        if args.method != 'adjoint':
            parser.error('--backend local requires --method adjoint')
    cls = tracked_driver(cls)
    if args.case == 'alkane':
        from benchmarks.dlpno_adjoint_scaling import alkane
        atoms = alkane(args.carbons)
    else:
        atoms = CASES[args.case]
    mol = gto.M(atom=atoms, basis=args.basis, verbose=0)
    reference_lmo = None
    reference_pivots = None

    def evaluate(m):
        mf = scf.RHF(m)
        if args.df_scf:
            mf = mf.density_fit()
        mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
        mf.max_cycle = 150
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('Finite-difference SCF did not converge')
        nc = 0 if args.all_electron else count_frozen_core(m)
        orb = orbitals(m, mf.get_ovlp(), mf.get_fock(), mf.mo_coeff,
                       m.nelectron // 2, nc, conv_tol=1e-12,
                       ensure_minimum=args.backend == 'local')
        if reference_lmo is not None:
            overlap = reference_lmo.T @ gto.intor_cross('int1e_ovlp', mol, m) @ orb.C_lmo
            rows, columns = linear_sum_assignment(-abs(overlap))
            columns = columns[np.argsort(rows)]
            matched = overlap[np.arange(len(columns)), columns]
            if np.min(abs(matched)) < .9:
                raise RuntimeError('Finite difference left the reference localization branch')
            # Only the permutation and signs are aligned.  The displaced
            # orbitals were independently reoptimized and remain so.
            orb.C_lmo = orb.C_lmo[:, columns] * np.sign(matched)
        evaluation_cls = (frozen_orthogonalization_driver(reference_pivots)
                          if args.freeze_pao_pivots and reference_pivots is not None else cls)
        d = evaluation_cls(mf, c_lmo=orb.C_lmo, **kwargs)
        d.kernel()
        return d

    start = time.perf_counter()
    base = evaluate(mol)
    reference_lmo = base.orb.C_lmo
    reference_pivots = base._orthogonalization_events
    orbital = RHFResponse(base) if args.method == 'direct' else None
    adjoint = None
    if args.method == 'adjoint':
        from electrolites.dlpno.local import LocalGradients
        adjoint = LocalGradients(base, backend=args.response_backend)
        adjoint.kernel()
    level = 'ccsd' if args.triples == 'none' else 'ccsd_t'
    selected = None if args.direction == 'all' else tuple(map(int, args.direction.split(',')))
    code = Path(__file__).resolve().parents[1] / 'electrolites' / 'dlpno'
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code.glob('*.py'))}
    report = dict(settings=vars(args), options=base.opt, code_sha256=hashes,
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  geometry_bohr=mol.atom_coords().tolist(), symbols=[mol.atom_symbol(i) for i in range(mol.natm)],
                  auxbasis_sha256=hashlib.sha256(repr(base.auxmol._basis).encode()).hexdigest(),
                  energy={k: float(v) for k, v in base.e.items()},
                  topology=topology(base), units='Eh and Eh/bohr', rows=[], complete=False)
    if adjoint is not None:
        report.update(adjoint=adjoint.diagnostics, adjoint_timings=adjoint.timings,
                      energy_timings=base.timings)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report['seconds'] = time.perf_counter() - start
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(path)

    save()
    directions = (orbital.directions() if orbital is not None else
                  [(a, x) for a in range(mol.natm) for x in range(3)])
    for direction in directions:
        atom, xyz = (direction.atom, direction.xyz) if orbital is not None else direction
        if selected is not None and (atom, xyz) != selected:
            continue
        if adjoint is None:
            response = ResponseDLPNO(base, orbital, direction, level, backend=args.response_backend)
            ecorr = response.kernel()
            analytic = {k: float(getattr(v, 'd', 0.)) for k, v in response.e.items()}
            analytic['tot'] = float(ecorr.d + orbital.hf_gradient[atom, xyz])
            row = dict(atom=atom, xyz=xyz, analytic=analytic, response=response.diagnostics,
                       orbital=orbital.diagnostics[-1],
                       energy_errors={k: float(getattr(v, 'v', v) - base.e[k])
                                      for k, v in response.e.items() if k in base.e}, finite_differences=[])
        else:
            analytic = dict(tot=float(adjoint.de[atom, xyz]),
                            corr=float(adjoint.de[atom, xyz]-adjoint.components['hf'][atom, xyz]))
            row = dict(atom=atom, xyz=xyz, analytic=analytic,
                       energy_errors=dict(corr=adjoint.diagnostics['energy_error']), finite_differences=[])
        for step in args.steps:
            displacements = []
            for sign in [1, -1]:
                coords = mol.atom_coords().copy()
                coords[atom, xyz] += sign * step
                displaced = mol.copy().set_geom_(coords, unit='Bohr')
                displacements.append(evaluate(displaced))
            plus, minus = displacements
            raw_stable = topology(plus) == topology(base) == topology(minus)
            stable = topology(plus, True) == topology(base, True) == topology(minus, True)
            fd = {k: float((plus.e[k] - minus.e[k]) / (2 * step)) for k in analytic if k in plus.e}
            errors = {k: analytic[k] - value for k, value in fd.items()}
            row['finite_differences'].append(dict(step=step, topology_stable=stable,
                                                  selection_parameters_stable=raw_stable,
                                                  gradient=fd, errors=errors,
                                                  plus_energy={k: float(v) for k, v in plus.e.items()},
                                                  minus_energy={k: float(v) for k, v in minus.e.items()}))
            if not raw_stable:
                row['finite_differences'][-1].update(plus_topology=topology(plus),
                                                    minus_topology=topology(minus))
            print(f'{args.case}/{args.basis} {atom},{xyz} h={step:g} '
                  f'analytic={analytic["tot"]:.10f} FD={fd["tot"]:.10f} '
                  f'error={errors["tot"]:.3e} topology={stable}', flush=True)
        report['rows'].append(row)
        save()
    report['complete'] = True
    report['passed'] = bool(report['rows']) and all(
        item['topology_stable'] and max(abs(x) for x in item['errors'].values()) < 1e-6
        for row in report['rows'] for item in row['finite_differences'])
    report['passed'] &= all(max(abs(x) for x in row['energy_errors'].values()) < 1e-8
                            for row in report['rows'])
    if selected is None:
        gradient = np.array([row['analytic']['tot'] for row in report['rows']]).reshape(mol.natm, 3)
        report['translation_residual'] = gradient.sum(axis=0).tolist()
        report['rotation_residual'] = np.cross(mol.atom_coords(), gradient).sum(axis=0).tolist()
        report['passed'] &= (max(abs(x) for x in report['translation_residual']) < 1e-7
                             and max(abs(x) for x in report['rotation_residual']) < 1e-7)
    save()
    if not report['passed']:
        raise SystemExit('Gradient check failed; inspect errors and topology in the JSON report')


if __name__ == '__main__':
    main()
