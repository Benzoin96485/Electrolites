"""Bounded spatial quadrature for local DOI/domain preparation.

This is an explicit numerical screening definition. It uses the same
atomic radial/angular quadrature and Becke partition as the dense oracle,
but restricts radial points, partition neighbors and small orbital
coefficients. The Gaussian AO omission radius follows a radial envelope.
Selection agreement and refinement must be checked against the dense
oracle; the Gaussian envelope alone does not certify total DOI error.

With fixed numerical settings, bounded coefficient supports and bounded
spatial density, each atom contributes bounded grid/Gram work and sparse
storage. Converting the dense reference orbitals to coefficient supports
is measured separately and still scans their dense input matrices.
"""
from itertools import product
import math
import os
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from pyscf import gto
from pyscf.dft import gen_grid, numint, radi

from .local_integrals import CoefficientMaps, compact_molecule
from .screening import SparseValues


def gaussian_ao_radii(mol, tolerance=1e-14):
    """Atom radii outside which every individual AO is below tolerance.

    For each contracted shell the envelope is
    A_l r**l sum_p max_c |c_pc| exp(-alpha_p*r*r). A_l bounds the
    Cartesian monomials and the libcint Cartesian-to-spherical map.
    Beyond sqrt(l/(2*min(alpha))), every radial term is decreasing.
    """
    if not 0 < tolerance < 1:
        raise ValueError('AO envelope tolerance must lie between zero and one')
    radii = np.zeros(mol.natm)
    for shell in range(mol.nbas):
        atom, angular = mol.bas_atom(shell), mol.bas_angular(shell)
        exponents = mol.bas_exp(shell)
        nprim, nctr = mol.bas_nprim(shell), mol.bas_nctr(shell)
        pointer = mol._bas[shell, gto.PTR_COEFF]
        coefficients = mol._env[pointer:pointer+nprim*nctr].reshape(nctr, nprim)
        maximum = np.max(abs(coefficients), axis=0)
        common = (1/math.sqrt(4*math.pi) if angular == 0 else
                  math.sqrt(3/(4*math.pi)) if angular == 1 else 1.)
        angular_bound = common if mol.cart else common*np.max(np.sum(abs(gto.cart2sph(angular, normalized='sp')), axis=0))
        keep = maximum > 0
        if not np.any(keep):
            continue
        log_coefficient, alpha = np.log(maximum[keep]), exponents[keep]
        def log_bound(radius):
            power = angular*math.log(radius) if angular else 0.
            return math.log(angular_bound)+power+float(logsumexp(log_coefficient-alpha*radius*radius))
        peak = math.sqrt(angular/(2*min(alpha))) if angular else 0.
        target = math.log(tolerance)
        if log_bound(peak) <= target:
            radius = peak
        else:
            upper = max(peak+1., 2.)
            while log_bound(upper) > target:
                upper *= 2
            radius = brentq(lambda r: log_bound(r)-target, peak, upper)
        radii[atom] = max(radii[atom], radius*(1+128*np.finfo(float).eps))
    return radii


def atom_neighbors(coords, radius):
    """Fixed-radius spatial hash, without a global atom-distance square."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Neighbor radius must be positive and finite')
    coords = np.asarray(coords)
    keys = np.floor(coords/radius).astype(np.int64)
    cells = {}
    for i, key in enumerate(keys):
        cells.setdefault(tuple(key), []).append(i)
    offsets = tuple(product((-1, 0, 1), repeat=3))
    neighbors, tests = [], 0
    for center, key in zip(coords, keys):
        selected = []
        for delta in offsets:
            group = cells.get(tuple(int(k+d) for k, d in zip(key, delta)), ())
            tests += len(group)
            selected.extend(j for j in group if np.linalg.norm(coords[j]-center) <= radius)
        neighbors.append(np.array(sorted(selected), dtype=int))
    return neighbors, tests


def _accumulate(rows, left, right, block):
    added = 0
    for i, values in zip(left, block):
        row = rows[int(i)]
        for k in np.flatnonzero(values):
            j, value = int(right[k]), float(values[k])
            if j in row:
                row[j] += value
            else:
                row[j] = value
                added += 1
    return added


def _sqrt_table(shape, rows, floor):
    return SparseValues(shape, [{j: math.sqrt(max(value, 0.)) for j, value in row.items()
                                  if value > floor*floor} for row in rows], floor)


def local_mulliken(base, overlap, maps, floor):
    """Same population algebra on explicit local coefficient supports."""
    result = [dict() for _ in range(base.mol.natm)]
    largest, operations = 0, 0
    for i, atoms in enumerate(maps.atoms['lmo']):
        rows, c = maps.matrix('lmo', [i], atoms)
        c = c[:, 0]
        p = c[:, None]*overlap[np.ix_(rows, rows)]*c[None, :]
        diagonal = np.diag(p)
        denominator = diagonal[:, None]+diagonal[None, :]
        with np.errstate(invalid='ignore', divide='ignore'):
            weight = np.where(denominator != 0., diagonal[:, None]/denominator, 0.)
        per_bf = 2*np.sum(p*weight, axis=1)
        row_atoms = base.bf_atom[rows]
        for atom in atoms:
            value = float(np.sum(per_bf[row_atoms == atom]))
            if abs(value) > floor:
                result[int(atom)][i] = value
        largest = max(largest, len(rows)); operations += len(rows)**2
    return SparseValues((base.mol.natm, base.n), result, floor), dict(largest_ao=largest, ao_square_work=operations)


def compute_local_screening(base, *, coefficient_cut=1e-8, ao_tolerance=1e-14,
                            grid_radius=20., partition_radius=40.):
    """Return sparse DOI tables, populations and measured preparation costs.

    Finite radii are in bohr. All settings are explicit parts of the
    domain-selection protocol. No distance cutoff is applied to the
    distant correlation energy. Dense input support discovery is reported
    separately; global SCF/Boys orbital operations are not hidden here.
    """
    if coefficient_cut <= 0 or not math.isfinite(coefficient_cut):
        raise ValueError('Local screening requires a positive finite coefficient cutoff')
    if grid_radius <= 0 or not math.isfinite(grid_radius) or partition_radius <= 0 or not math.isfinite(partition_radius):
        raise ValueError('Finite positive radial and partition radii are required')
    if os.environ.get('DLPNO_DOI_GRID'):
        raise ValueError('An external global DOI grid requires the dense screening oracle')
    mol, opt, orb = base.mol, base.opt, base._orbital_reference
    start = time.perf_counter()
    maps = CoefficientMaps(base, coefficient_cut, coefficient_cut)
    # Reuse these exact discrete supports in the parent dipole energy
    # and in its orbital adjoint. Selection is not repeated in reverse.
    base._local_dipole_maps = maps
    inverse = {kind: [set() for _ in range(mol.natm)] for kind in ('lmo', 'pao')}
    for kind in inverse:
        for column, atoms in enumerate(maps.atoms[kind]):
            for atom in atoms:
                inverse[kind][int(atom)].add(column)
    support_time = time.perf_counter()-start
    start = time.perf_counter()
    ao_radii = gaussian_ao_radii(mol, ao_tolerance)
    neighborhood_radius = max(partition_radius, grid_radius+max(ao_radii, default=0.))
    neighbors, distance_tests = atom_neighbors(mol.atom_coords(), neighborhood_radius)
    settings = gen_grid.Grids(mol)
    atomic = gen_grid.gen_atomic_grids(mol, atom_grid=opt['DOI_GRID'], radi_method=radi.treutler, prune=None)
    n, nao = base.n, orb.nbf
    ij, iu, uv = [dict() for _ in range(n)], [dict() for _ in range(n)], [dict() for _ in range(nao)]
    counts = dict(atoms=mol.natm, grid_points=0, omitted_radial_points=0, maximum_neighbor_atoms=0,
                  maximum_ao=0, maximum_active_lmo=0, maximum_active_pao=0,
                  ao_point_work=0, gram_work=0, partition_pair_point_work=0,
                  distance_tests=distance_tests, accumulator_entries=0)
    for atom, domain in enumerate(neighbors):
        local = compact_molecule(mol, domain)
        local_atom = int(np.flatnonzero(domain == atom)[0])
        coordinates, volumes = atomic[mol.atom_symbol(atom)]
        keep = np.linalg.norm(coordinates, axis=1) <= grid_radius
        counts['omitted_radial_points'] += int(np.count_nonzero(~keep))
        # Ask the same PySCF partition implementation for only this
        # atom's grid, using empty grids on the other local atoms.
        table = {local.atom_symbol(i): (np.zeros((0, 3)), np.zeros(0)) for i in range(local.natm)}
        table[local.atom_symbol(local_atom)] = coordinates[keep], volumes[keep]
        point_blocks, weight_blocks = gen_grid.get_partition(
            local, table, radii_adjust=settings.radii_adjust, atomic_radii=settings.atomic_radii,
            becke_scheme=settings.becke_scheme, concat=False)
        points, weights = point_blocks[local_atom], weight_blocks[local_atom]
        active = {kind: np.array(sorted(set().union(*(inverse[kind][int(a)] for a in domain))), dtype=int)
                  for kind in inverse}
        if len(active['lmo']) == 0 and len(active['pao']) == 0:
            continue
        ao = numint.eval_ao(local, points)
        _, cl = maps.matrix('lmo', active['lmo'], domain)
        _, cp = maps.matrix('pao', active['pao'], domain)
        li, lu = (ao@cl)**2, (ao@cp)**2
        weighted_l, weighted_u = li*weights[:, None], lu*weights[:, None]
        counts['accumulator_entries'] += _accumulate(ij, active['lmo'], active['lmo'], weighted_l.T@li)
        counts['accumulator_entries'] += _accumulate(iu, active['lmo'], active['pao'], weighted_l.T@lu)
        counts['accumulator_entries'] += _accumulate(uv, active['pao'], active['pao'], weighted_u.T@lu)
        ng, nl, nv = len(points), len(active['lmo']), len(active['pao'])
        counts['grid_points'] += ng
        counts['maximum_neighbor_atoms'] = max(counts['maximum_neighbor_atoms'], len(domain))
        counts['maximum_ao'] = max(counts['maximum_ao'], local.nao)
        counts['maximum_active_lmo'] = max(counts['maximum_active_lmo'], nl)
        counts['maximum_active_pao'] = max(counts['maximum_active_pao'], nv)
        counts['ao_point_work'] += ng*local.nao
        counts['gram_work'] += ng*(nl*nl+nl*nv+nv*nv)
        counts['partition_pair_point_work'] += ng*len(domain)**2
    doi_floor = min(opt[k] for k in ('T_CUT_DO_PRE', 'T_CUT_DO', 'T_CUT_DO_TRIPLES_PRE', 'T_CUT_DO_TRIPLES'))
    tables = (_sqrt_table((n, n), ij, opt['T_CUT_DO_IJ']), _sqrt_table((n, nao), iu, doi_floor),
              _sqrt_table((nao, nao), uv, opt['T_CUT_DO_UV']))
    mkn_floor = min(opt[k] for k in ('T_CUT_MKN', 'T_CUT_MKN_TRIPLES_PRE', 'T_CUT_MKN_TRIPLES'))
    mkn, mkn_info = local_mulliken(base, orb.S, maps, mkn_floor)
    counts.update(storage='sparse', producer='local_spatial_grid',
                  numerical_approximation=True, total_doi_error_certified=False,
                  linear_work_conditional_on_bounded_domains_and_density=True,
                  reference_support_setup_seconds=support_time, spatial_screening_seconds=time.perf_counter()-start,
                  coefficient_cut=coefficient_cut, ao_tolerance=ao_tolerance,
                  maximum_ao_envelope_radius=float(np.max(ao_radii, initial=0)),
                  grid_radius=grid_radius, requested_partition_radius=partition_radius,
                  actual_neighborhood_radius=neighborhood_radius, mulliken=mkn_info,
                  entries=dict(zip(('doi_ij', 'doi_iu', 'doi_uv', 'mkn'), [a.nnz for a in (*tables, mkn)])))
    return *tables, mkn, counts
