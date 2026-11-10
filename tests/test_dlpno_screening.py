import numpy as np
from electrolites.dlpno.screening import SparsePositions


def test_sparse_positions_preserve_batched_lookup_without_global_rows():
    columns = [np.array([2, 17, 900_000]), np.array([], dtype=int), np.array([1, 9])]
    positions = SparsePositions(columns, 1_000_000)
    assert positions.nnz == 5
    assert positions[0, 900_000] == 2
    assert positions[1, 9] == -1
    assert np.array_equal(positions[0][np.array([[17, 1], [2, 900_000]])], [[1, -1], [0, 2]])
    assert np.array_equal(positions[np.array([0, 2])[:, None], np.array([1, 9, 17])[None]],
                          [[-1, -1, 1], [0, 1, -1]])
    assert positions[1][np.zeros((0, 2), dtype=int)].shape == (0, 2)
import pytest

from electrolites.dlpno.screening import (
    SparseValues, PairMask, atom_complete_mask, row_above, column_above, empty_mask_like,
)
from electrolites.dlpno.setup import PairList, Domains


def test_threshold_complete_sparse_queries_and_local_masks():
    rng = np.random.default_rng(73)
    values = rng.normal(size=(20, 15)); values[abs(values) < 1.] = 0.
    sparse = SparseValues.from_dense(values, .5)
    assert sparse.nnz == np.count_nonzero(values)
    for cut in (.5, 1., 2.):
        for i in range(20):
            assert np.array_equal(row_above(sparse, i, cut), np.flatnonzero(abs(values[i]) > cut))
        for j in range(15):
            assert np.array_equal(column_above(sparse, j, cut), np.flatnonzero(abs(values[:, j]) > cut))
        assert np.array_equal((abs(sparse) > cut).to_dense(), abs(values) > cut)
    with pytest.raises(ValueError, match='storage floor'):
        sparse.row_above(0, .1)
    i, j = np.array([0, 3, 9])[:, None], np.array([1, 5])[None, :]
    assert np.array_equal(sparse[i, j], values[i, j])


def test_sparse_pair_refinement_keeps_order_and_adjacency():
    rng = np.random.default_rng(73)
    dense = rng.random((37, 37)) < .1
    dense |= dense.T | np.eye(37, dtype=bool)
    mask = PairMask(dense.shape, zip(*np.nonzero(dense)))
    original, sparse = PairList(37, dense, sparse=True), PairList(37, mask, sparse=True)
    assert original.pairs == sparse.pairs
    assert np.array_equal(original.ji, sparse.ji)
    refined, reference = empty_mask_like(mask), np.zeros_like(dense)
    for i, j in sparse.pairs:
        if (i+j)%3 == 0 or i == j:
            refined[i, j] = reference[i, j] = True
    assert PairList(37, refined, sparse=True).pairs == PairList(37, reference, sparse=True).pairs
    assert mask.nnz == np.count_nonzero(dense)
    assert mask.tobytes() == dense.tobytes()  # explicit diagnostic conversion


def test_sparse_domains_and_atom_complete_screen_match_dense():
    rng = np.random.default_rng(74)
    bf_atom = np.repeat(np.arange(5), [2, 3, 2, 4, 3])
    atom_bfs = [np.flatnonzero(bf_atom == i) for i in range(5)]
    atom_ribfs = [np.arange(3*i, 3*i+3) for i in range(5)]
    doi, mkn = rng.random((8, len(bf_atom))), rng.normal(size=(5, 8))
    dense = Domains(mkn, doi, .5, .6, bf_atom, atom_bfs, atom_ribfs)
    sparse = Domains(SparseValues.from_dense(mkn, .4), SparseValues.from_dense(doi, .4),
                      .5, .6, bf_atom, atom_bfs, atom_ribfs)
    for name in ('lmo_riatoms', 'lmo_ribfs', 'lmo_paos'):
        for a, b in zip(getattr(dense, name), getattr(sparse, name)):
            assert np.array_equal(a, b)
    uv = rng.random((len(bf_atom), len(bf_atom)))
    actual = atom_complete_mask(SparseValues.from_dense(uv, .9), .95, bf_atom, 5)
    expected = np.zeros_like(uv, dtype=bool)
    for a in atom_bfs:
        for b in atom_bfs:
            expected[np.ix_(a, b)] = np.max(uv[np.ix_(a, b)]) > .95
    assert np.array_equal(actual.to_dense(), expected)
