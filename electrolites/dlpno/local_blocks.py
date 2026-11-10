"""Local primitive leaves and sparse occupied intermediates for adjoints.

The adapters here expose only requested blocks to the correlation graph.
``ReferenceBlocks`` initially reads an existing energy object's integral
store; this is an exact storage/adjoint oracle, not yet the final screened
AO block producer.  Its remaining global source storage is explicitly
reported and must not be confused with the size of the local graph.
"""
from types import SimpleNamespace

import numpy as np

from . import reverse as ad
from .adjoint import AdjointDLPNO, host


def _key_token(key):
    if not isinstance(key, tuple):
        key = (key,)
    tokens = []
    for index in key:
        if isinstance(index, slice):
            tokens.append(('slice', index.start, index.stop, index.step))
        else:
            array = np.asarray(index)
            tokens.append((array.dtype.str, array.shape, array.tobytes()))
    return tuple(tokens)


class SelectionTensor:
    def __init__(self, shape, getter, primal_getter=None):
        self.shape = shape
        self.ndim = len(shape)
        self.getter = getter
        self.primal_getter = primal_getter or getter

    def __getitem__(self, key):
        return self.getter(key)

    def primal_copy(self):
        return SelectionTensor(self.shape, self.primal_getter)


class SparseMatrix:
    """Copy/update of occupied intermediates without copying an n*n node."""

    def __init__(self, shape, getter, primal_getter=None, entries=None):
        self.shape = shape
        self.ndim = 2
        self.getter = getter
        self.primal_getter = primal_getter or getter
        self.entries = {} if entries is None else entries

    def __getitem__(self, key):
        i, j = key
        if np.isscalar(i) and np.isscalar(j):
            pair = (int(i), int(j))
            return self.entries[pair] if pair in self.entries else self.getter(pair)
        if isinstance(i, slice):
            i = np.arange(*i.indices(self.shape[0]))
        if isinstance(j, slice):
            j = np.arange(*j.indices(self.shape[1]))
        rows, cols = np.broadcast_arrays(i, j)
        elements = [self[int(a), int(b)] for a, b in zip(rows.ravel(), cols.ravel())]
        if any(isinstance(x, ad.Variable) for x in elements):
            return ad.Numpy(ad._xp(elements)).array(elements).reshape(rows.shape)
        return np.asarray(elements).reshape(rows.shape)

    def __setitem__(self, key, value):
        i, j = key
        if not np.isscalar(i) or not np.isscalar(j):
            raise NotImplementedError('Occupied sparse updates must address individual elements')
        self.entries[int(i), int(j)] = value

    def copy(self):
        return SparseMatrix(self.shape, self.getter, self.primal_getter, dict(self.entries))

    def primal_copy(self):
        return SparseMatrix(self.shape, self.primal_getter,
                            entries={k: host(v) for k, v in self.entries.items()})


class ReferenceBlocks:
    """Exact primitive-block oracle backed by existing host energy arrays."""

    def __init__(self, base):
        self.base = base
        self.orb = getattr(base, '_orbital_reference', base.orb)
        self.leaves, self.selections = {}, {}
        # A separate streamed producer will replace these dense source
        # arrays; preserving them here permits independent block tests.
        if not hasattr(base, 'qia'):
            raise NotImplementedError('ReferenceBlocks currently requires the host reference integral store')
        self.sources = dict(qia=base.qia, qij=base.qij, qab=base.qab, metric=base.J2c,
                            f_lmo=base.orb.F_lmo, f_pao=base.orb.F_pao,
                            s_pao=base.orb.S_pao, dipole=base.dip_e)
        self.source_bytes = sum(value.nbytes for value in self.sources.values())
        self.shapes = {name: value.shape for name, value in self.sources.items()}

    def read(self, name, key):
        return self.sources[name][key]

    def tracked_read(self, name, key):
        token = name, _key_token(key)
        if token not in self.leaves:
            value = self.read(name, key)
            self.leaves[token] = ad.Variable(self.xp.asarray(value))
            self.selections[token] = key
        return self.leaves[token]

    def attach(self, state, tracked=True):
        self.xp = getattr(state, 'xp', np)
        if tracked:
            state.leaves = self.leaves

        def tensor(name, sparse=False):
            def get(key):
                return self.tracked_read(name, key)

            def plain(key):
                return self.read(name, key)

            cls = SparseMatrix if sparse else SelectionTensor
            return cls(self.shapes[name], get if tracked else plain, plain)

        state.orb = SimpleNamespace(**vars(self.orb))
        state.orb.S_pao, state.orb.F_pao = tensor('s_pao'), tensor('f_pao')
        state.orb.F_lmo = tensor('f_lmo', sparse=True)
        state.qia, state.qij, state.qab = [tensor(name) for name in ('qia', 'qij', 'qab')]
        state.J2c = tensor('metric')
        if 'dipole_tail' in self.shapes:
            state.dip_e = None
            state._distant_tail_energy = tensor('dipole_tail')[()]
        else:
            state.dip_e = tensor('dipole')
        state._S = {}

    def directional_derivative(self, bars, direction_state):
        """Contract with a direct-response oracle; validation only."""
        from .dual import tangent
        tangent_sources = dict(qia=direction_state.qia, qij=direction_state.qij,
                               qab=direction_state.qab, metric=direction_state.J2c,
                               f_lmo=direction_state.orb.F_lmo, f_pao=direction_state.orb.F_pao,
                               s_pao=direction_state.orb.S_pao, dipole=direction_state.dip_e)
        return sum(float(np.sum(bar*tangent(tangent_sources[key[0]][self.selections[key]])))
                   for key, bar in bars.items())

    def statistics(self):
        by_kind = {} if self.leaves else getattr(self, '_released_leaf_statistics', {})
        for (kind, _), leaf in self.leaves.items():
            info = by_kind.setdefault(kind, dict(blocks=0, elements=0, largest_block=0))
            info['blocks'] += 1
            info['elements'] += leaf.size
            info['largest_block'] = max(info['largest_block'], leaf.size)
        return dict(blocks=by_kind, dense_source_bytes=self.source_bytes,
                    linear_integral_producer=False)

    def release_leaves(self, statistics=None):
        """Keep scalar evidence while releasing primal integral graph roots.

        Cotangents and selection maps are independent arrays and remain
        available for the subsequent streamed nuclear contraction. Pass
        pre-sweep statistics if the sweep already consumed leaf values.
        """
        self._released_leaf_statistics = (ReferenceBlocks.statistics(self)['blocks']
                                         if statistics is None else statistics)
        self.leaves.clear()


class BlockAdjointDLPNO(AdjointDLPNO):
    def __init__(self, base, inputs=None, **kwargs):
        super().__init__(base, inputs=ReferenceBlocks(base) if inputs is None else inputs, **kwargs)

    def setup(self):
        self.inputs.attach(self)
