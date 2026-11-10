"""Local integral supports, primitive blocks and their nuclear derivatives."""
import numpy as np
import pytest
from pyscf import gto, scf

from electrolites.dlpno.reference_cpu import CPUDLPNO
from electrolites.dlpno.local_integrals import LocalRIBlocks


def test_stationary_boys_saddle_is_repaired():
    from pyscf import lo
    from electrolites.dlpno.reference_cpu import minimum_boys, boys_curvature
    # Canonical occupied orbitals of distant He2 have identical centers:
    # the Boys gradient vanishes at a maximum of the spread functional.
    mol = gto.M(atom='He 0 0 -2; He 0 0 2', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-13)
    original = mf.mo_coeff[:, mf.mo_occ > 0]
    before, _ = boys_curvature(mol, original)
    assert before['stationary_residual'] < 1e-8
    assert before['scaled_curvature'] < -.1
    repaired, info = minimum_boys(mol, original)
    assert info['verified_minimum']
    assert info['saddle_repairs'] == 1
    assert info['scaled_curvature'] > 0
    assert info['stationary_residual'] < 1e-10
    assert lo.Boys(mol, repaired).cost_function() < lo.Boys(mol, original).cost_function()-1
    assert np.max(abs(repaired@repaired.T-original@original.T)) < 1e-12


def test_normalized_cholesky_roundoff_ties_are_deterministic():
    from electrolites.dlpno.setup import pivoted_cholesky
    matrix = np.array([[1., .8, .2], [.8, 1., .4], [.2, .4, 1.]])
    expected = pivoted_cholesky(matrix, 1e-8)
    for atom in range(3):
        changed = matrix.copy()
        changed[atom, atom] += 2*np.finfo(float).eps
        assert pivoted_cholesky(changed, 1e-8) == expected


def test_sparse_pair_topology_matches_dense_lookup():
    from electrolites.dlpno.setup import PairList
    rng = np.random.default_rng(56)
    mask = rng.random((31, 31)) < .08
    mask |= mask.T | np.eye(len(mask), dtype=bool)
    dense, sparse = PairList(len(mask), mask), PairList(len(mask), mask, sparse=True)
    rows, cols = np.array([0, 8, 30])[:, None], np.array([2, 4, 14])[None, :]
    assert np.array_equal(dense.ij[rows, cols], sparse.ij[rows, cols])
    assert np.array_equal(dense.ij[:, [1, 4]], sparse.ij[:, [1, 4]])
    assert np.array_equal(dense.ij[[1, 4], :], sparse.ij[[1, 4], :])
    assert np.array_equal(dense.ij[:, :], sparse.ij[:, :])
    assert np.array_equal(dense.ji, sparse.ji)
    assert len(sparse.ij.entries) == len(sparse)
    for p in range(len(dense)):
        assert np.array_equal(dense.lmos(p), sparse.lmos(p))


@pytest.fixture(scope='module')
def water_pair():
    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57; '
                     'O 0 0 12; H .02 .76 12.59; H .09 -.75 12.57',
                basis='6-31g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 150
    mf.kernel()
    base = CPUDLPNO(mf, triples=None, verbose=0)
    base.setup()
    return base


def test_local_block_exact_limit_and_bounded_cache(water_pair):
    base = water_pair
    blocks = LocalRIBlocks(base, lmo_cut=0., pao_cut=0., cache_bytes=1 << 16)
    aux = np.array([0, 2, base.auxmol.nao-1])
    occupied = np.array([0, base.n-1])
    virtual = np.array([0, 3, base.orb.nbf-1])
    for kind, key in [('qia', np.ix_(aux, occupied, virtual)),
                       ('qij', np.ix_(aux, occupied, occupied)),
                       ('qab', np.ix_(aux, virtual, virtual))]:
        assert np.max(abs(blocks.read(kind, key)-getattr(base, kind)[key])) < 1e-11
    key = np.ix_(aux, aux)
    assert np.max(abs(blocks.read('metric', key)-base.J2c[key])) < 1e-11
    assert blocks.counts['cache_peak_bytes'] <= blocks.budget
    assert not hasattr(blocks, 'qia')
    assert blocks.statistics()['primitive_integral_global_storage_bytes'] == 0
    blocks.clear_cache()
    assert np.max(abs(blocks.read('metric', key)-base.J2c[key])) < 1e-11
    key = np.ix_(aux, occupied, virtual)
    assert np.max(abs(blocks.read('qia', key)-base.qia[key])) < 1e-11


def test_remote_ao_support_is_not_evaluated(water_pair):
    base = water_pair
    blocks = LocalRIBlocks(base, lmo_cut=1e-4, pao_cut=1e-4)
    first = np.concatenate(base.atom_bfs[:3])
    occupied = int(np.argmax(np.sum(base.orb.C_lmo[first]**2, axis=0)))
    key = np.ix_(base.atom_ribfs[0], [occupied], base.atom_bfs[1])
    value = blocks.read('qia', key)
    assert np.max(abs(value-base.qia[key])) < 1e-7
    assert blocks.counts['largest_ao'] <= len(first)
    assert blocks.counts['largest_aux'] == len(base.atom_ribfs[0])
    for columns in base.atom_bfs:
        for column in columns:
            assert np.array_equal(blocks.maps.atoms['pao'][column], blocks.maps.atoms['pao'][columns[0]])


def test_gradient_rejects_changed_parent_coefficient_supports():
    from electrolites.dlpno.local import LocalDLPNO
    mol = gto.M(atom='He 0 0 -4; He 0 0 4', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-13)
    base = LocalDLPNO(mf, triples=None, lmo_cut=0., pao_cut=0., verbose=0)
    base.kernel()
    # Distant coefficients are tiny. Even an arbitrarily loose energy
    # replay tolerance cannot authorize differentiating a different mask.
    with pytest.raises(ValueError, match='supports differ from the parent energy'):
        base.nuc_grad_method(lmo_cut=1e-4, energy_tolerance=float('inf')).kernel()
    same = LocalRIBlocks(base, lmo_cut=0., pao_cut=0.)
    same.validate_energy_support()
    # Distinct thresholds are allowed when they select identical supports.
    local = LocalDLPNO(mf, triples=None, lmo_cut=1e-4, pao_cut=1e-4, verbose=0)
    local.setup()
    LocalRIBlocks(local, lmo_cut=1e-5, pao_cut=1e-5).validate_energy_support()


def test_auxiliary_envelope_reuses_overlapping_requested_blocks(water_pair):
    base = water_pair
    blocks = LocalRIBlocks(base, lmo_cut=1e-4, pao_cut=1e-4, cache_bytes=1 << 20)
    reference = LocalRIBlocks(base, lmo_cut=1e-4, pao_cut=1e-4,
                              reuse_envelopes=False, reuse_transforms=False)
    first = np.concatenate(base.atom_bfs[:3])
    occupied = int(np.argmax(np.sum(base.orb.C_lmo[first]**2, axis=0)))
    key = np.ix_([0], [occupied], base.atom_bfs[1])
    expected = reference.read('qia', key)
    assert np.max(abs(blocks.read('qia', key)-expected)) < 1e-12
    # Force this auxiliary atom to serve the other molecule as well.
    wide = np.ix_([0], np.arange(base.n), base.atom_bfs[-1])
    assert np.max(abs(blocks.read('qia', wide)-reference.read('qia', wide))) < 1e-12
    calls = blocks.counts['integral_calls']
    assert np.max(abs(blocks.read('qia', key)-expected)) < 1e-12
    assert blocks.counts['integral_calls'] == calls
    assert blocks.used <= blocks.budget


def test_compact_density_indices_match_expanded_gathers(water_pair, monkeypatch):
    from electrolites.dlpno import local_integrals
    from electrolites.dlpno.local_blocks import _key_token
    base = water_pair
    rng = np.random.default_rng(809)
    aux = np.concatenate([base.atom_ribfs[1][:3], base.atom_ribfs[0][:4],
                           base.atom_ribfs[-1][:2]])
    occupied = np.array([2, 0, 2])  # Repeated and unsorted indices.
    virtual = np.array([4, 1, 4, base.orb.nbf-1])
    selections = [('qij', (aux, 1, 2)), ('qij', np.ix_(aux, occupied, occupied)),
                  ('qia', (aux[:, None], 1, virtual[None, :])),
                  ('qia', np.ix_(aux, occupied, virtual)),
                  ('qab', np.ix_(aux, virtual, virtual)), ('metric', np.ix_(aux, aux))]
    blocks = LocalRIBlocks(base, lmo_cut=1e-4, pao_cut=1e-4)
    bars = {}
    values = []
    for kind, key in selections:
        token = kind, _key_token(key)
        blocks.selections[token] = key
        values.append(blocks.read(kind, key))
        bars[token] = rng.normal(size=values[-1].shape)
    actual_sources, actual_force = blocks.pullback(bars)
    assert blocks.counts['compact_index_elements'] < blocks.counts['compact_density_elements']
    monkeypatch.setattr(local_integrals, 'cartesian_axes', lambda key, shape: None)
    for (kind, key), value in zip(selections, values):
        assert np.max(abs(blocks.read(kind, key)-value)) < 1e-12
    expected_sources, expected_force = blocks.pullback(bars)
    assert np.max(abs(actual_force-expected_force)) < 1e-10
    for name in actual_sources:
        assert np.max(abs(actual_sources[name]-expected_sources[name])) < 1e-10


@pytest.mark.parametrize('kind', ['qab', 'qia', 'qij'])
def test_shared_auxiliary_leaves_match_independent_blocks(water_pair, monkeypatch, kind):
    from types import MethodType
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.local_blocks import ReferenceBlocks
    base = water_pair
    parent = LocalRIBlocks(base, lmo_cut=0., pao_cut=0.)
    monkeypatch.setattr(base, 'local_integrals', parent, raising=False)
    aux = np.concatenate([base.atom_ribfs[1][:3], base.atom_ribfs[0][:4]])
    left, right = (np.arange(size) for size in parent.shapes[kind][1:])
    parent.read(kind, np.ix_(aux, left, right))
    selections = [np.ix_(aux, [2, 0, 2, 5], [4, 1, 4]),
                  np.ix_(aux[::-1], [0, 5, 1], [2, 4, 1])]
    rng = np.random.default_rng(907)
    weights = [rng.normal(size=parent.read(kind, key).shape) for key in selections]

    def evaluate(shared):
        blocks = LocalRIBlocks(base, lmo_cut=0., pao_cut=0.)
        blocks.xp = np
        if not shared:
            blocks.tracked_read = MethodType(ReferenceBlocks.tracked_read, blocks)
        energy = sum(np.sum(weight*blocks.tracked_read(kind, key))
                     for key, weight in zip(selections, weights))
        graph = ad.Linearization([energy], list(blocks.leaves.values()))
        bars = dict(zip(blocks.leaves, graph([np.array(1.)])))
        sources, force = blocks.pullback(bars)
        return float(energy.v), sources, force, blocks.counts

    actual, expected = evaluate(True), evaluate(False)
    assert actual[3]['shared_'+kind+'_reads'] == 2
    assert abs(actual[0]-expected[0]) < 1e-12
    assert np.max(abs(actual[2]-expected[2])) < 1e-10
    for name in actual[1]:
        assert np.max(abs(actual[1][name]-expected[1][name])) < 1e-10


@pytest.mark.parametrize('kind', ['s_pao', 'f_pao'])
@pytest.mark.parametrize('checkpoint', [False, True])
def test_tiled_orbital_leaves_match_independent_pullback(water_pair, kind, checkpoint):
    from electrolites.dlpno import reverse as ad
    base = water_pair
    selections = [np.ix_([9, 0, 9, 17, 2], [21, 4, 8, 21]),
                  np.ix_([2, 17, 0], [8, 21, 4]),
                  (9, np.array([21, 4, 21])),
                  (np.array([9, 0, 9]), 4), (9, 21)]
    rng = np.random.default_rng(1359)
    matrix = getattr(base.orb, 'S_pao' if kind == 's_pao' else 'F_pao')
    weights = [rng.normal(size=matrix[key].shape) for key in selections]

    def evaluate(tile_size):
        blocks = LocalRIBlocks(base, lmo_cut=0., pao_cut=0., orbital_tile_size=tile_size)
        blocks.xp = np
        terms = []
        for key, weight in zip(selections, weights):
            def forward(key=key, weight=weight):
                value = blocks.tracked_read(kind, key)
                assert np.array_equal(value.v, matrix[key])
                return (np.sum(weight*value*value),)
            terms.append(ad.checkpoint(forward)[0][0] if checkpoint else forward()[0])
        energy = sum(terms)
        statistics = blocks.statistics()
        statistics['integral_blocks'] = dict(statistics['integral_blocks'])
        graph = ad.Linearization([energy], list(blocks.leaves.values()))
        value = float(energy.v)
        bars = dict(zip(blocks.leaves, graph([np.array(1.)],
                                           release_graph=True, release_inputs=True)))
        sources, force = blocks.pullback(bars)
        return value, sources, force, statistics

    actual, expected = evaluate(4), evaluate(0)
    assert actual[3]['blocks'][kind]['largest_block'] <= 16
    assert actual[3]['blocks'][kind]['elements'] <= matrix.size
    assert actual[3]['integral_blocks']['shared_'+kind+'_reads'] == len(selections)
    assert abs(actual[0]-expected[0]) < 1e-11
    assert np.max(abs(actual[2]-expected[2])) < 1e-10
    for name in actual[1]:
        assert np.max(abs(actual[1][name]-expected[1][name])) < 1e-10


@pytest.mark.parametrize('kind', ['s_pao', 'f_pao'])
def test_orbital_tile_fallback_preserves_general_indexing(water_pair, kind):
    from electrolites.dlpno import reverse as ad
    blocks = LocalRIBlocks(water_pair, orbital_tile_size=4)
    blocks.xp = np
    matrix = blocks.orbital_sources[kind]
    selections = [(np.array([2, 1, 2]), np.array([4, 9, 4])),
                  np.ix_([-1, 2, -1], [4, -2]),
                  (slice(1, 4), slice(2, 5)),
                  np.ix_([], [4])]
    expected = np.zeros_like(matrix)
    terms = []
    for key in selections:
        value = blocks.tracked_read(kind, key)
        assert np.array_equal(value.v, matrix[key])
        terms.append(value.sum())
        np.add.at(expected, key, np.ones_like(matrix[key]))
    bars = ad.gradients(sum(terms), list(blocks.leaves.values()))
    actual = np.zeros_like(matrix)
    for token, bar in zip(blocks.leaves, bars):
        np.add.at(actual, blocks.selections[token], bar)
    assert np.array_equal(actual, expected)


def test_screened_pao_supports_preserve_rigid_rotation():
    from electrolites.dlpno.local import LocalDLPNO
    from benchmarks.dlpno_adjoint_scaling import alkane
    mol = gto.M(atom=alkane(2), basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
    mf.kernel()
    base = LocalDLPNO(mf, pno='loose', triples=None, verbose=0,
                      E_CONVERGENCE=1e-11, R_CONVERGENCE=1e-10, MAXITER=150)
    base.kernel()
    force = base.nuc_grad_method().kernel()
    assert np.max(abs(force.sum(axis=0))) < 1e-8
    assert np.max(abs(np.cross(mol.atom_coords(), force).sum(axis=0))) < 1e-8


def test_local_energy_adjoint_and_independent_nuclear_difference():
    from electrolites.dlpno.local import LocalDLPNO
    from benchmarks.dlpno_gradient_check import STRESS
    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57', basis='6-31g', verbose=0)

    def solve(m):
        mf = scf.RHF(m)
        mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
        mf.kernel()
        state = LocalDLPNO(mf, triples='t1', verbose=0, **STRESS,
                           E_CONVERGENCE=1e-11, R_CONVERGENCE=1e-10, T_CUT_ITER=0.)
        state.kernel()
        return state

    base = solve(mol)
    gradient = base.nuc_grad_method()
    force = gradient.kernel()
    energies = []
    step = 2e-4
    for sign in [1, -1]:
        xyz = mol.atom_coords()
        xyz[1, 2] += sign*step
        energies.append(solve(mol.copy().set_geom_(xyz, unit='Bohr')).e['tot'])
    assert abs(force[1, 2]-(energies[0]-energies[1])/(2*step)) < 1e-7
    assert np.max(abs(force.sum(axis=0))) < 1e-8
    assert np.max(abs(np.cross(mol.atom_coords(), force).sum(axis=0))) < 1e-8
    assert gradient.diagnostics['integrals']['primitive_integral_global_storage_bytes'] == 0
    for stage in ('mp2', 'ccsd', 'triples'):
        assert gradient.diagnostics['amplitudes'][stage]['lambda_solves'] == 1


@pytest.mark.parametrize('kind', ['s_pao', 'f_pao'])
def test_tiled_orbital_gather_bounds_retained_arrays(water_pair, kind):
    from electrolites.dlpno import reverse as ad
    blocks = LocalRIBlocks(water_pair, orbital_tile_size=4)
    blocks.xp = np
    matrix = blocks.orbital_sources[kind]
    # Multiple tiles, repeated entries, and nonmonotone order expose
    # copies retained by decomposing the gather into concatenations.
    rows = np.tile([17, 0, 9, 2, 17, 21], 7)
    columns = np.tile([21, 4, 8, 0, 21], 5)
    key = np.ix_(rows, columns)
    value = blocks.tracked_read(kind, key)
    assert np.array_equal(value.v, matrix[key])
    leaves = list(blocks.leaves.values())
    graph = ad.Linearization([value], leaves)
    retained_bound = value.v.nbytes + sum(x.v.nbytes for x in leaves)
    assert graph.statistics['value_bytes'] <= retained_bound
    seed = np.arange(value.size, dtype=float).reshape(value.shape)/value.size
    expected = np.zeros_like(matrix)
    np.add.at(expected, key, seed)
    bars = graph([seed])
    for token, bar in zip(blocks.leaves, bars):
        assert np.max(abs(bar-expected[blocks.selections[token]])) < 1e-12
    # Selected VJPs must omit unrelated tile cotangents while preserving
    # the same repeated-index accumulation on the requested tile.
    target = len(leaves)//2
    selected = ad.Linearization([value], [leaves[target]])([seed])[0]
    assert np.array_equal(selected, bars[target])


@pytest.mark.parametrize('kind', ['qab', 'qia', 'qij'])
@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
def test_identity_ri_sources_preserve_independent_pullback(
        water_pair, monkeypatch, kind, checkpoint, backend):
    """Full leaves, reordered atoms and repeated subsets share exact VJPs."""
    import json
    from types import MethodType
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.adjoint import host
    from electrolites.dlpno.local_blocks import ReferenceBlocks
    xp = np if backend == 'numpy' else pytest.importorskip('cupy')
    base = water_pair
    parent = LocalRIBlocks(base, lmo_cut=0., pao_cut=0.)
    monkeypatch.setattr(base, 'local_integrals', parent, raising=False)
    q0, q1 = base.atom_ribfs[:2]
    left, right = (np.arange(size) for size in parent.shapes[kind][1:])
    parent.read(kind, np.ix_(np.concatenate([q0, q1]), left, right))
    full = np.ix_(q0, left, right)
    selections = [full, np.ix_(np.concatenate([q1, q0]), left, right),
                  np.ix_(q0[::-1], [len(left)-1, 0, len(left)-1], [2, 1, 2]),
                  full, np.ix_(np.r_[q0[2::-1], q0[0]], left, right),
                  (q0, 0, 0)]
    rng = np.random.default_rng(17007)
    weights = [rng.normal(size=parent.read(kind, key).shape)
               /np.sqrt(max(1, parent.read(kind, key).size)) for key in selections]

    def evaluate(shared):
        blocks = LocalRIBlocks(base, lmo_cut=0., pao_cut=0.)
        blocks.xp = xp
        if not shared:
            blocks.tracked_read = MethodType(ReferenceBlocks.tracked_read, blocks)
        terms = []
        value_errors = []
        for key, weight in zip(selections, weights):
            def forward(key=key, weight=weight):
                value = blocks.tracked_read(kind, key)
                value_errors.append(float(np.max(abs(host(value)-parent.read(kind, key)))))
                return (np.sum(xp.asarray(weight)*value*value),)
            terms.append(ad.checkpoint(forward)[0][0] if checkpoint else forward()[0])
        energy = sum(terms)
        scalar = float(energy.v)
        leaves = list(blocks.leaves.values())
        graph = ad.Linearization([energy], leaves)
        bars = graph([xp.asarray(1.)], release_graph=True, release_inputs=True)
        densities = {token: host(bar) for token, bar in zip(blocks.leaves, bars)}
        sources, force = blocks.pullback(densities)
        assert all(leaf.v is None for leaf in leaves)
        return scalar, sources, force, dict(blocks.counts), max(value_errors)

    actual, expected = evaluate(True), evaluate(False)
    energy_error = abs(actual[0]-expected[0])
    force_error = float(np.max(abs(actual[2]-expected[2])))
    source_error = max(float(np.max(abs(actual[1][key]-expected[1][key])))
                       for key in actual[1])
    identity_parts = actual[3]['shared_'+kind+'_identity_parts']
    assert identity_parts >= 4
    assert energy_error < 1e-10 and force_error < 2e-10 and source_error < 2e-10
    assert max(actual[4], expected[4]) < 1e-12
    print('RI_IDENTITY_REPORT '+json.dumps(dict(
        kind=kind, backend=backend, checkpoint=checkpoint,
        energy_error=energy_error, force_error=force_error,
        orbital_source_error=source_error, maximum_value_error=max(actual[4], expected[4]),
        identity_parts=identity_parts,
        identity_elements=actual[3]['shared_'+kind+'_identity_elements'],
        independent_block_reference=True, repeated_indices=True,
        reordered_auxiliary_atoms=True, scalar_indices=True,
        consuming_inputs_released=True)))
