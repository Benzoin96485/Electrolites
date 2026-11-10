"""Sparse screening tables, masks and local row/column queries.

Only values above the smallest threshold used by the caller need storage.
Queries below that declared floor raise instead of silently changing a
domain. Converting an already dense table is an exact storage adapter;
it does not accelerate the original dense DOI quadrature producer.
"""
import numpy as np


class SparsePositions:
    """Local positions of global indices, without a dense row lookup.

    ``positions[atom][indices]`` and ``positions[atom, indices]`` both
    preserve NumPy's scalar/advanced-index shapes. Missing entries are -1.
    """
    def __init__(self, columns, width):
        self.shape, self.ndim, self.dtype = (len(columns), int(width)), 2, np.dtype(np.int64)
        self.rows = [{int(column): position for position, column in enumerate(row)} for row in columns]

    @property
    def nnz(self):
        return sum(map(len, self.rows))

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            if not np.isscalar(key):
                raise TypeError('Select one sparse position row, or use a pair of index arrays')
            parent, atom = self, int(key)
            class Row:
                def __getitem__(self, columns):
                    return parent[atom, columns]
            return Row()
        atoms, columns = key
        if np.isscalar(atoms) and np.isscalar(columns):
            return self.rows[int(atoms)].get(int(columns), -1)
        atoms, columns = np.broadcast_arrays(atoms, columns)
        return np.fromiter((self.rows[int(a)].get(int(c), -1) for a, c in zip(atoms.ravel(), columns.ravel())),
                           dtype=np.int64, count=atoms.size).reshape(atoms.shape)


class PairMask:
    """Mutable sparse Boolean matrix with deterministic row ordering."""
    def __init__(self, shape, pairs=()):
        self.shape = tuple(map(int, shape))
        self.ndim, self.dtype = 2, np.dtype(bool)
        self.rows = [set() for _ in range(self.shape[0])]
        for i, j in pairs:
            self[int(i), int(j)] = True

    def __setitem__(self, key, value):
        i, j = map(int, key)
        if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
            raise IndexError('Sparse pair index out of range')
        if value:
            self.rows[i].add(j)
        else:
            self.rows[i].discard(j)

    def __getitem__(self, key):
        i, j = key
        if np.isscalar(i) and np.isscalar(j):
            return int(j) in self.rows[int(i)]
        rows, columns = np.broadcast_arrays(i, j)
        return np.fromiter((int(b) in self.rows[int(a)] for a, b in zip(rows.ravel(), columns.ravel())),
                           dtype=bool, count=rows.size).reshape(rows.shape)

    def pairs(self, upper=False):
        return [(i, j) for i, row in enumerate(self.rows) for j in sorted(row)
                if not upper or i < j]

    def nonzero(self):
        pairs = self.pairs()
        if not pairs:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        return tuple(np.asarray(pairs, dtype=int).T)

    def add_diagonal(self):
        for i in range(min(self.shape)):
            self.rows[i].add(i)
        return self

    def copy(self):
        result = PairMask(self.shape)
        result.rows = [set(row) for row in self.rows]
        return result

    def __or__(self, other):
        if not isinstance(other, PairMask) or other.shape != self.shape:
            raise TypeError('Sparse mask unions require matching PairMask objects')
        result = PairMask(self.shape)
        result.rows = [a | b for a, b in zip(self.rows, other.rows)]
        return result

    @property
    def nnz(self):
        return sum(map(len, self.rows))

    def to_dense(self):
        """Explicit diagnostic/export conversion; not used in solver loops."""
        result = np.zeros(self.shape, dtype=bool)
        for i, row in enumerate(self.rows):
            result[i, list(row)] = True
        return result

    def tolist(self):
        return self.to_dense().tolist()

    def tobytes(self):
        return self.to_dense().tobytes()


class SparseValues:
    """Threshold-complete values with linear row and column adjacency."""
    def __init__(self, shape, rows, minimum=0.):
        self.shape = tuple(map(int, shape))
        self.ndim = 2
        self.minimum = float(minimum)
        if self.minimum < 0 or not np.isfinite(self.minimum):
            raise ValueError('A nonnegative finite stored threshold floor is required')
        if len(rows) != self.shape[0]:
            raise ValueError('Sparse row count differs from the declared shape')
        self.rows = []
        self.columns = [dict() for _ in range(self.shape[1])]
        for i, row in enumerate(rows):
            kept = {}
            for j, value in row.items():
                j, value = int(j), float(value)
                if not 0 <= j < self.shape[1] or not np.isfinite(value):
                    raise ValueError('Invalid sparse table entry')
                if abs(value) > self.minimum:
                    kept[j] = value
                    self.columns[j][i] = value
            self.rows.append(kept)

    @classmethod
    def from_dense(cls, matrix, minimum=0.):
        matrix = np.asarray(matrix)
        rows = [{int(j): float(row[j]) for j in np.flatnonzero(abs(row) > minimum)}
                for row in matrix]
        return cls(matrix.shape, rows, minimum)

    def _check(self, threshold):
        if threshold < self.minimum:
            raise ValueError('Domain threshold is below the sparse table storage floor')

    def row_above(self, row, threshold):
        self._check(threshold)
        return np.array(sorted(j for j, value in self.rows[int(row)].items() if abs(value) > threshold), dtype=int)

    def column_above(self, column, threshold):
        self._check(threshold)
        return np.array(sorted(i for i, value in self.columns[int(column)].items() if abs(value) > threshold), dtype=int)

    def __gt__(self, threshold):
        self._check(threshold)
        return PairMask(self.shape, ((i, j) for i, row in enumerate(self.rows)
                                    for j, value in row.items() if value > threshold))

    def __abs__(self):
        return SparseValues(self.shape, [{j: abs(v) for j, v in row.items()} for row in self.rows], self.minimum)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key, slice(None))
        i, j = key
        if np.isscalar(i) and np.isscalar(j):
            return self.rows[int(i)].get(int(j), 0.)
        if isinstance(i, slice) or isinstance(j, slice):
            # Diagnostic compatibility. Domain loops use row_above or
            # column_above, never dense row/column scans.
            ri = np.arange(self.shape[0])[i] if isinstance(i, slice) else np.asarray(i)
            cj = np.arange(self.shape[1])[j] if isinstance(j, slice) else np.asarray(j)
            if ri.ndim and cj.ndim:
                ri, cj = np.ix_(ri, cj)
            i, j = ri, cj
        rows, columns = np.broadcast_arrays(i, j)
        return np.fromiter((self.rows[int(a)].get(int(b), 0.) for a, b in zip(rows.ravel(), columns.ravel())),
                           dtype=float, count=rows.size).reshape(rows.shape)

    @property
    def nnz(self):
        return sum(map(len, self.rows))


class AtomPairMask:
    """AO pair screen backed by atom-pair adjacency, without an AO square."""
    def __init__(self, atoms, bf_atom):
        self.atoms, self.bf_atom = atoms, np.asarray(bf_atom, dtype=int)
        self.shape = (len(bf_atom), len(bf_atom))
        self.ndim, self.dtype = 2, np.dtype(bool)

    def __getitem__(self, key):
        i, j = key
        return self.atoms[self.bf_atom[i], self.bf_atom[j]]

    def to_dense(self):
        return self[np.arange(len(self.bf_atom))[:, None], np.arange(len(self.bf_atom))[None, :]]

    def tobytes(self):
        return self.to_dense().tobytes()


def row_above(matrix, row, threshold):
    if hasattr(matrix, 'row_above'):
        return matrix.row_above(row, threshold)
    return np.flatnonzero(abs(matrix[row]) > threshold)


def column_above(matrix, column, threshold):
    if hasattr(matrix, 'column_above'):
        return matrix.column_above(column, threshold)
    return np.flatnonzero(abs(matrix[:, column]) > threshold)


def mask_pairs(mask):
    return mask.pairs() if isinstance(mask, PairMask) else list(zip(*np.nonzero(mask)))


def empty_mask_like(mask):
    return PairMask(mask.shape) if isinstance(mask, PairMask) else np.zeros(mask.shape, dtype=bool)


def atom_complete_mask(doi_uv, threshold, bf_atom, natm):
    if not isinstance(doi_uv, SparseValues):
        raise TypeError('Sparse atom-pair screening requires SparseValues')
    doi_uv._check(threshold)
    pairs = PairMask((natm, natm))
    for i, row in enumerate(doi_uv.rows):
        for j, value in row.items():
            if value > threshold:
                pairs[int(bf_atom[i]), int(bf_atom[j])] = True
    return AtomPairMask(pairs, bf_atom)
