"""Screened AO support maps and bounded local three-center integral blocks.

The same provider can serve a parent energy and its primitive adjoints.
All selection masks belong to the energy definition and remain fixed in
an analytic derivative.  Coefficient cutoffs of zero give an exact dense-
support oracle, while positive cutoffs allow bounded AO supports in a
localized, gapped system.  Actual support growth still has to be measured.
"""
from collections import OrderedDict
import time

import numpy as np
from pyscf import df, gto

from . import reverse as ad
from .adjoint import orthocanonicalize
from .local_blocks import ReferenceBlocks
from . import setup as su


def compact_molecule(mol, atoms):
    """A genuinely compact libcint environment, preserving shell order.

    Merely slicing ``_bas`` leaves the global atom/environment tables in
    every block and gives an avoidable size-dependent cost per integral
    call.  Rebuild the selected atoms with their explicit original bases.
    """
    atoms = tuple(map(int, atoms))
    specification, basis = [], {}
    for local, atom in enumerate(atoms):
        label = mol.atom_pure_symbol(atom)+str(local)
        key = mol.atom_symbol(atom)
        if key not in mol._basis:
            key = mol.atom_pure_symbol(atom)
        basis[label] = mol._basis[key]
        specification.append((label, mol.atom_coord(atom)))
    nuclear_charge = int(sum(mol.atom_charge(atom) for atom in atoms))
    return gto.M(atom=specification, basis=basis, unit='Bohr', cart=mol.cart,
                 spin=nuclear_charge % 2, charge=0, verbose=0)


class CoefficientMaps:
    """Atom-complete supports of individual LMO and PAO columns."""

    def __init__(self, base, lmo_cut, pao_cut):
        self.base = base
        orb = getattr(base, '_orbital_reference', base.orb)
        self.coefficients = dict(lmo=orb.C_lmo, pao=orb.C_pao)
        self.cutoffs = dict(lmo=float(lmo_cut), pao=float(pao_cut))
        self.atoms = {}
        for kind, matrix in self.coefficients.items():
            cut = self.cutoffs[kind]
            supports = []
            for column in range(matrix.shape[1]):
                if cut == 0:
                    selected = np.arange(base.mol.natm, dtype=int)
                else:
                    rows = np.flatnonzero(abs(matrix[:, column]) >= cut)
                    selected = np.unique(base.bf_atom[rows])
                if len(selected) == 0:
                    raise ValueError(f'{kind} coefficient cutoff removes an entire orbital')
                supports.append(selected)
            if kind == 'pao':
                # PAOs originating on one atom mix under a rigid rotation
                # of its AO basis.  A common support for that entire
                # column block makes coefficient truncation covariant.
                # Independent p_x/p_y/p_z masks produce a spurious torque.
                for columns in base.atom_bfs:
                    selected = np.unique(np.concatenate([supports[int(c)] for c in columns]))
                    for column in columns:
                        supports[int(column)] = selected
            self.atoms[kind] = supports

    def union(self, kind, columns):
        if len(columns) == 0:
            return np.zeros(0, dtype=int)
        return np.unique(np.concatenate([self.atoms[kind][int(c)] for c in columns]))

    def matrix(self, kind, columns, ao_atoms):
        rows = np.concatenate([self.base.atom_bfs[int(a)] for a in ao_atoms])
        matrix = self.coefficients[kind][np.ix_(rows, columns)].copy()
        # Each column has its own mask, even inside a union support.
        row_atoms = self.base.bf_atom[rows]
        for j, column in enumerate(columns):
            matrix[~np.isin(row_atoms, self.atoms[kind][int(column)]), j] = 0.
        return rows, matrix

    def mask(self, kind, columns, rows):
        return np.column_stack([np.isin(self.base.bf_atom[rows], self.atoms[kind][int(c)])
                                for c in columns])


def index_arrays(key, shape):
    """Broadcast advanced-index arrays, retaining scalar-index behavior."""
    if not isinstance(key, tuple):
        key = (key,)
    if len(key) == 1 and np.asarray(key[0]).dtype == bool:
        return np.nonzero(key[0])
    arrays = []
    for axis, index in enumerate(key):
        arrays.append(np.arange(*index.indices(shape[axis])) if isinstance(index, slice) else np.asarray(index))
    return np.broadcast_arrays(*arrays)


def cartesian_axes(key, shape):
    """Compact axis lists for outer selections; None for paired gathers.

    Integral domains use np.ix_ or a single vector with scalar indices.
    Keep those index lists instead of expanding three integer tensors
    with the same size as every primitive density.
    """
    if not isinstance(key, tuple) or len(key) != len(shape):
        return None
    arrays = [np.arange(*x.indices(n)) if isinstance(x, slice) else np.asarray(x)
              for x, n in zip(key, shape)]
    if any(x.dtype == bool for x in arrays):
        return None
    ndim = max(x.ndim for x in arrays)
    positions = []
    for x in arrays:
        varying = [ndim-x.ndim+j for j, size in enumerate(x.shape) if size != 1]
        if len(varying) > 1:
            return None
        positions.extend(varying)
    # Scalar occupied indices remove an output axis, e.g. Q[:,None], i,
    # u[None,:]. Also retain empty axes. Repeated output dimensions are
    # paired gathers, and permuted dimensions need the general fallback.
    if positions != sorted(set(positions)):
        return None
    return tuple(x.reshape(-1) for x in arrays)


def _atom_row_slices(indices, atom_map):
    """Return views for sorted domains, preserving arbitrary gathers too."""
    atom_ids = atom_map[indices]
    for atom in np.unique(atom_ids):
        rows = np.flatnonzero(atom_ids == atom)
        selection = slice(int(rows[0]), int(rows[-1])+1) if rows[-1]-rows[0]+1 == len(rows) else rows
        yield int(atom), selection


class LocalRIBlocks(ReferenceBlocks):
    """On-demand transformed integrals without global qia/qij/qab arrays."""

    def __init__(self, base, lmo_cut=None, pao_cut=None, cache_bytes=64 << 20,
                 reuse_envelopes=True, reuse_transforms=True, orbital_tile_size=128):
        self.base = base
        self.orb = getattr(base, '_orbital_reference', base.orb)
        self.leaves, self.selections = {}, {}
        lmo_cut = base.opt['T_CUT_CLMO'] if lmo_cut is None else lmo_cut
        pao_cut = base.opt['T_CUT_CPAO'] if pao_cut is None else pao_cut
        self.maps = CoefficientMaps(base, lmo_cut, pao_cut)
        self.pao_pair_mask = getattr(base, 'pao_pair_mask', base.doi_uv > base.opt['T_CUT_DO_UV'])
        naux, n, nbf = base.auxmol.nao, base.n, self.orb.nbf
        self.shapes = dict(qia=(naux, n, nbf), qij=(naux, n, n), qab=(naux, nbf, nbf),
                           metric=(naux, naux), f_lmo=(n, n), f_pao=(nbf, nbf),
                           s_pao=(nbf, nbf), dipole=(n, n))
        self.orbital_sources = dict(f_lmo=self.orb.F_lmo, f_pao=self.orb.F_pao,
                                    s_pao=self.orb.S_pao)
        if hasattr(base, '_distant_tail_energy'):
            self.shapes.pop('dipole')
            self.shapes['dipole_tail'] = ()
            self.orbital_sources['dipole_tail'] = np.asarray(base._distant_tail_energy)
        else:
            self.orbital_sources['dipole'] = getattr(base, '_dipole_reference', base.dip_e)
        self.source_bytes = sum(v.nbytes for v in self.orbital_sources.values())
        if not isinstance(orbital_tile_size, (int, np.integer)) or orbital_tile_size < 0:
            raise ValueError('orbital_tile_size must be a nonnegative integer')
        self.orbital_tile_size = int(orbital_tile_size)
        self.budget = cache_bytes
        self.reuse_envelopes = reuse_envelopes
        self.reuse_transforms = reuse_transforms
        self.ao_envelopes = {}
        self.column_envelopes = {}
        self.cache, self.used = OrderedDict(), 0
        self.counts = dict(integral_calls=0, largest_ao=0, largest_aux=0,
                            largest_integral_bytes=0, cache_peak_bytes=0)
        parent = getattr(base, 'local_integrals', None)
        if parent is not None and parent.orb is self.orb:
            # The gradient knows the complete supports visited by its
            # parent energy. Avoid rebuilding the same GPU shell-pair
            # optimizer whenever the next pair enlarges an envelope.
            if self.reuse_envelopes:
                self.ao_envelopes = dict(parent.ao_envelopes)
            if self.reuse_transforms:
                self.column_envelopes = dict(parent.column_envelopes)
            self.counts['inherited_auxiliary_envelopes'] = len(self.ao_envelopes)

    def clear_cache(self):
        """Release regenerable integrals while retaining all support plans."""
        released = self.used
        self.cache.clear()
        self.used = 0
        return released

    def plan_domains(self, lmos, paos):
        """Plan bounded auxiliary-atom envelopes before the GPU pair loop.

        Existing extended domains supply conservative local column sets.
        Unanticipated requests still grow an envelope by the usual union;
        coefficients outside their individual supports remain masked.
        No integral is evaluated or retained by this planning step.
        """
        for atom, (occupied, virtual) in enumerate(zip(lmos, paos)):
            if not len(occupied) and not len(virtual):
                continue
            if self.reuse_transforms:
                for name, left, right in [('qij', occupied, occupied),
                                          ('qia', occupied, virtual), ('qab', virtual, virtual)]:
                    if len(left) and len(right):
                        self.column_envelopes[name, atom] = left, right
            if self.reuse_envelopes:
                self.ao_envelopes[atom,] = np.union1d(
                    self.maps.union('lmo', occupied), self.maps.union('pao', virtual))
        self.counts['planned_auxiliary_envelopes'] = len(self.ao_envelopes)

    def validate_energy_support(self):
        """A close replay energy cannot certify identical derivative masks."""
        parent = getattr(self.base, 'local_integrals', None)
        for kind, supports in self.maps.atoms.items():
            expected = (parent.maps.atoms[kind] if parent is not None else
                        [np.arange(self.base.mol.natm)]*len(supports))
            if len(supports) != len(expected) or any(
                    not np.array_equal(actual, original)
                    for actual, original in zip(supports, expected)):
                raise ValueError(f'{kind} coefficient supports differ from the parent energy; '
                                 'recompute the energy with the requested coefficient cutoffs')

    def _put(self, key, value):
        category = lambda token: ('raw' if str(token[0]).startswith('int3c2e')
                                  else 'metric' if token[0] == 'metric' else 'transformed')
        kind = category(key)
        limit = self.budget//4 if kind == 'raw' else self.budget//8 if kind == 'metric' else self.budget
        # An oversized streamed block must not evict useful small blocks.
        if value.nbytes > limit:
            return value
        old = self.cache.pop(key, None)
        if old is not None:
            self.used -= old.nbytes
        if kind != 'transformed':
            matching = [k for k in self.cache if category(k) == kind]
            used = sum(self.cache[k].nbytes for k in matching)
            for old_key in matching:
                if used+value.nbytes <= limit:
                    break
                old = self.cache.pop(old_key)
                self.used -= old.nbytes
                used -= old.nbytes
        while self.cache and self.used+value.nbytes > self.budget:
            _, old = self.cache.popitem(last=False)
            self.used -= old.nbytes
        if value.nbytes <= self.budget:
            self.cache[key] = value
            self.used += value.nbytes
            self.counts['cache_peak_bytes'] = max(self.counts['cache_peak_bytes'], self.used)
        return value

    def _evaluate_raw(self, mol, aux, intor, comp):
        return df.incore.aux_e2(mol, aux, intor=intor, aosym='s1', comp=comp)

    def _transform_raw(self, raw, left, right):
        return np.einsum('mnQ,mi,nj->Qij', raw, left, right, optimize=True)

    def raw(self, ao_atoms, aux_atoms, intor='int3c2e', comp=1):
        requested = np.asarray(ao_atoms, dtype=int)
        auxiliary = tuple(map(int, aux_atoms))
        if self.reuse_envelopes:
            previous = self.ao_envelopes.get(auxiliary, np.empty(0, dtype=int))
            envelope = np.union1d(previous, requested)
            if len(envelope) != len(previous):
                # One expanding support per auxiliary atom avoids a new
                # cache entry for every nearly identical pair AO set.
                # An evicted envelope is regenerated on its established
                # support.  No coefficient outside the requested support
                # enters a transformed integral.
                for integral in ('int3c2e', 'int3c2e_ip1', 'int3c2e_ip2'):
                    old_key = integral, tuple(map(int, previous)), auxiliary
                    old = self.cache.pop(old_key, None)
                    if old is not None:
                        self.used -= old.nbytes
                self.ao_envelopes[auxiliary] = envelope
                self.counts['envelope_grows'] = self.counts.get('envelope_grows', 0)+1
        else:
            envelope = requested
        key = intor, tuple(map(int, envelope)), auxiliary
        if key in self.cache:
            self.cache.move_to_end(key)
            value = self.cache[key]
        else:
            mol = compact_molecule(self.base.mol, envelope)
            aux = compact_molecule(self.base.auxmol, aux_atoms)
            value = self._evaluate_raw(mol, aux, intor, comp)
            self.counts['integral_calls'] += 1
            self.counts['largest_ao'] = max(self.counts['largest_ao'], mol.nao)
            self.counts['largest_aux'] = max(self.counts['largest_aux'], aux.nao)
            self.counts['largest_integral_bytes'] = max(self.counts['largest_integral_bytes'], value.nbytes)
            self._put(key, value)
        if np.array_equal(envelope, requested):
            return value
        rows = np.concatenate([self.base.atom_bfs[int(a)] for a in requested])
        all_rows = np.concatenate([self.base.atom_bfs[int(a)] for a in envelope])
        positions = np.searchsorted(all_rows, rows)
        if comp == 1:
            return value[positions[:, None], positions[None, :], :]
        return value[:, positions[:, None], positions[None, :], :]

    def metric(self, indices1, indices2):
        atoms = np.unique(self.base.ribf_atom[np.union1d(indices1, indices2)])
        key = 'metric', tuple(map(int, atoms))
        if key in self.cache:
            matrix = self.cache[key]
            self.cache.move_to_end(key)
        else:
            aux = compact_molecule(self.base.auxmol, atoms)
            matrix = self._put(key, aux.intor('int2c2e'))
        global_indices = np.concatenate([self.base.atom_ribfs[int(a)] for a in atoms])
        return matrix[np.searchsorted(global_indices, indices1), np.searchsorted(global_indices, indices2)]

    def _transformed_columns(self, name, atom, columns1, columns2, kind1, kind2):
        if self.reuse_transforms:
            old1, old2 = self.column_envelopes.get((name, int(atom)),
                                                   (np.empty(0, int), np.empty(0, int)))
            columns1, columns2 = np.union1d(old1, columns1), np.union1d(old2, columns2)
            if len(columns1) != len(old1) or len(columns2) != len(old2):
                old_key = name, int(atom), tuple(map(int, old1)), tuple(map(int, old2))
                old = self.cache.pop(old_key, None)
                if old is not None:
                    self.used -= old.nbytes
                self.column_envelopes[name, int(atom)] = columns1, columns2
            key = name, int(atom), tuple(map(int, columns1)), tuple(map(int, columns2))
        else:
            key = None
        if key is not None and key in self.cache:
            self.cache.move_to_end(key)
            transformed = self.cache[key]
            self.counts['transformed_cache_hits'] = self.counts.get('transformed_cache_hits', 0)+1
        else:
            ao_atoms = np.union1d(self.maps.union(kind1, columns1), self.maps.union(kind2, columns2))
            _, c1 = self.maps.matrix(kind1, columns1, ao_atoms)
            _, c2 = self.maps.matrix(kind2, columns2, ao_atoms)
            raw = self.raw(ao_atoms, [atom])
            transformed = self._transform_raw(raw, c1, c2)
            self.counts['transform_calls'] = self.counts.get('transform_calls', 0)+1
            if key is not None:
                self._put(key, transformed)
        return columns1, columns2, transformed

    def read(self, name, key):
        if name in self.orbital_sources:
            return self.orbital_sources[name][key]
        axes = cartesian_axes(key, self.shapes[name])
        if axes is not None:
            shape = index_arrays(key, self.shapes[name])[0].shape
            if name == 'metric':
                return self.metric(*np.ix_(*axes)).reshape(shape)
            q, i, j = axes
            output = np.empty((len(q), len(i), len(j)))
            if not output.size:
                return output.reshape(shape)
            kind1, kind2 = dict(qij=('lmo', 'lmo'), qia=('lmo', 'pao'), qab=('pao', 'pao'))[name]
            left, right = np.unique(i), np.unique(j)
            mask = self.pao_pair_mask[i[:, None], j] if name == 'qab' else None
            for atom, rows in _atom_row_slices(q, self.base.ribf_atom):
                columns1, columns2, transformed = self._transformed_columns(
                    name, atom, left, right, kind1, kind2)
                indices = (np.searchsorted(self.base.atom_ribfs[atom], q[rows]),
                           np.searchsorted(columns1, i), np.searchsorted(columns2, j))
                value = transformed[np.ix_(*indices)]
                if mask is not None:
                    value *= mask[None]
                output[rows] = value
            self.counts['compact_reads'] = self.counts.get('compact_reads', 0)+1
            return output.reshape(shape)
        indices = index_arrays(key, self.shapes[name])
        if name == 'metric':
            return self.metric(*indices)
        q, i, j = indices
        output = np.zeros(q.shape)
        if output.size == 0:
            return output
        kind1, kind2 = dict(qij=('lmo', 'lmo'), qia=('lmo', 'pao'), qab=('pao', 'pao'))[name]
        qf, left, right, out = q.ravel(), i.ravel(), j.ravel(), output.ravel()
        for atom in np.unique(self.base.ribf_atom[qf]):
            selected = np.flatnonzero(self.base.ribf_atom[qf] == atom)
            columns1, columns2 = np.unique(left[selected]), np.unique(right[selected])
            columns1, columns2, transformed = self._transformed_columns(
                name, atom, columns1, columns2, kind1, kind2)
            qlocal = np.searchsorted(self.base.atom_ribfs[int(atom)], qf[selected])
            values = transformed[qlocal, np.searchsorted(columns1, left[selected]),
                                 np.searchsorted(columns2, right[selected])]
            if name == 'qab':
                values *= self.pao_pair_mask[left[selected], right[selected]]
            out[selected] = values
        return output

    def statistics(self):
        stats = super().statistics()
        support = {}
        for kind, domains in self.maps.atoms.items():
            sizes = [sum(len(self.base.atom_bfs[int(a)]) for a in domain) for domain in domains]
            support[kind] = dict(columns=len(sizes), total_ao=sum(sizes), largest_ao=max(sizes, default=0))
        stats.update(producer='local_ao', coefficient_cutoffs=self.maps.cutoffs,
                      pao_pair_screen=getattr(self.base, 'pao_screen', 'component'),
                      primitive_integral_global_storage_bytes=0, integral_blocks=self.counts,
                      coefficient_support=support,
                      locality_scaling_verified=False)
        envelope_sizes = [sum(len(self.base.atom_bfs[int(a)]) for a in atoms)
                          for atoms in self.ao_envelopes.values()]
        stats['ao_envelopes'] = dict(enabled=self.reuse_envelopes, count=len(envelope_sizes),
                                    maximum=max(envelope_sizes, default=0), total=sum(envelope_sizes))
        stats['transformed_envelopes'] = dict(enabled=self.reuse_transforms,
                                              count=len(self.column_envelopes),
                                              largest_columns=max((max(len(a), len(b)) for a, b in self.column_envelopes.values()), default=0))
        stats['orbital_tiles'] = dict(size=self.orbital_tile_size,
                                     enabled=self.orbital_tile_size > 0,
                                     kinds=['s_pao', 'f_pao'])
        # Keep the final proof flag false until connected-system scaling
        # and the complete force backtransformation are verified.
        return stats

    def _tracked_orbital_read(self, name, key):
        from .local_read_plans import OPTIONS, tracked_orbital_read
        if OPTIONS.get('orbital_plan'):
            return tracked_orbital_read(self, name, key)
        return self._reference_tracked_orbital_read(name, key)

    def _reference_tracked_orbital_read(self, name, key):
        """Share exact two-index sources through bounded tile leaves.

        Overlap checkpoints can revisit hundreds of thousands of local
        selections. Retaining each selection as a primitive duplicates
        both its primal array and its final density. Slice shared tiles
        instead, allowing checkpoint replay to discard those selections.
        A fixed tile size also bounds each checkpoint's input cotangent;
        using one global matrix leaf would allocate an N*N zero buffer
        for every local replay. No orbital or overlap entries are cut.
        The existing dense reference matrices remain separately counted.
        """
        shape = self.shapes[name]
        axes = cartesian_axes(key, shape)
        if (axes is None or not all(len(x) for x in axes)
                or any(not np.issubdtype(x.dtype, np.integer)
                       or np.any(x < 0) or np.any(x >= n)
                       for x, n in zip(axes, shape))):
            return super().tracked_read(name, key)
        size = self.orbital_tile_size
        groups = [[(int(tile), np.flatnonzero(axis//size == tile))
                   for tile in np.unique(axis//size)] for axis in axes]
        # A single primitive gathers all tiles.  Recording each gather and
        # concatenation separately retains several copies of every local
        # selection outside checkpoints, especially across tile boundaries.
        cartesian_shape = tuple(map(len, axes))
        value = self.xp.empty(cartesian_shape, dtype=self.orbital_sources[name].dtype)
        sources, plans = [], []
        for row_tile, row_indices in groups[0]:
            for col_tile, col_indices in groups[1]:
                canonical = (slice(row_tile*size, min((row_tile+1)*size, shape[0])),
                             slice(col_tile*size, min((col_tile+1)*size, shape[1])))
                source = super().tracked_read(name, canonical)
                local = np.ix_(axes[0][row_indices]-row_tile*size,
                               axes[1][col_indices]-col_tile*size)
                output = np.ix_(row_indices, col_indices)
                value[output] = source.v[local]
                sources.append(source)
                plans.append((source.shape, local, output))

        def selected_pullback(bar, needed):
            bar = bar.reshape(cartesian_shape)
            return tuple(ad.IndexedCotangent(tile_shape, local, bar[output])
                         if need else None
                         for (tile_shape, local, output), need in zip(plans, needed))

        def pullback(bar):
            return selected_pullback(bar, [True]*len(plans))

        self.counts['shared_'+name+'_reads'] = self.counts.get('shared_'+name+'_reads', 0)+1
        self.counts['orbital_gather_records'] = self.counts.get('orbital_gather_records', 0)+1
        value = value.reshape(index_arrays(key, shape)[0].shape)
        return ad.operation(value, sources, pullback, selected_pullback)

    def tracked_read(self, name, key):
        from .local_read_plans import OPTIONS, tracked_ri_read
        if OPTIONS.get('ri_fused') and name in ('qab', 'qia', 'qij'):
            return tracked_ri_read(self, name, key)
        return self._reference_tracked_read(name, key)

    def _reference_tracked_read(self, name, key):
        if self.orbital_tile_size and name in ('s_pao', 'f_pao'):
            return self._tracked_orbital_read(name, key)
        # Triplets request overlapping (Q|uv), (Q|iu), and (Q|ij) blocks. Give
        # them slices of common auxiliary-atom leaves so checkpointing
        # can discard/rebuild those slices instead of retaining a new
        # large primitive array for every triplet. The parent energy's
        # already observed envelopes bound these local source blocks.
        axes = cartesian_axes(key, self.shapes[name]) if name in ('qab', 'qia', 'qij') else None
        parent = getattr(self.base, 'local_integrals', None)
        envelopes = getattr(parent, 'column_envelopes', {})
        if axes is None or not all(len(x) for x in axes):
            return super().tracked_read(name, key)
        q, left, right = axes
        parts, order = [], []
        groups = list(_atom_row_slices(q, self.base.ribf_atom))
        for atom, _ in groups:
            columns = envelopes.get((name, atom))
            if columns is None or not (np.isin(left, columns[0]).all()
                                       and np.isin(right, columns[1]).all()):
                self.counts['unshared_'+name+'_reads'] = self.counts.get('unshared_'+name+'_reads', 0)+1
                return super().tracked_read(name, key)
        for atom, rows in groups:
            full_q = self.base.atom_ribfs[atom]
            columns1, columns2 = envelopes[name, atom]
            canonical = np.ix_(full_q, columns1, columns2)
            source = super().tracked_read(name, canonical)
            selected_q = q[rows]
            if (np.array_equal(selected_q, full_q)
                    and np.array_equal(left, columns1)
                    and np.array_equal(right, columns2)):
                # The selection is exactly this shared leaf. Avoid an
                # advanced-index copy and its elementwise reverse scatter.
                # Concatenation/reshape below still returns a fresh handle.
                parts.append(source)
                counter = 'shared_'+name+'_identity_parts'
                self.counts[counter] = self.counts.get(counter, 0)+1
                counter = 'shared_'+name+'_identity_elements'
                self.counts[counter] = self.counts.get(counter, 0)+source.size
            else:
                indices = np.ix_(np.searchsorted(full_q, selected_q),
                                 np.searchsorted(columns1, left), np.searchsorted(columns2, right))
                parts.append(source[indices])
            order.append(np.arange(len(q))[rows])
        value = ad.concatenate(parts, axis=0)
        permutation = np.concatenate(order)
        if not np.array_equal(permutation, np.arange(len(q))):
            value = value[np.argsort(permutation)]
        self.counts['shared_'+name+'_reads'] = self.counts.get('shared_'+name+'_reads', 0)+1
        return value.reshape(index_arrays(key, self.shapes[name])[0].shape)

    def pullback(self, bars):
        """Local RI/metric forces and sources for the orbital adjoints.

        No AO three-center density is formed over the full molecule.
        The remaining global two-index transformations are the reference
        orbital/PAO transformations, and are reported separately.
        """
        base = self.base
        start = time.perf_counter()
        force = np.zeros((base.mol.natm, 3))
        coefficient_bars = {kind: np.zeros_like(matrix) for kind, matrix in self.maps.coefficients.items()}
        orbital_bars = {name: np.zeros_like(value) for name, value in self.orbital_sources.items()}
        ri_terms, metric_terms = {}, {}
        for token, bar in bars.items():
            if not np.any(bar):
                continue
            name, _ = token
            key = self.selections[token]
            if name in orbital_bars:
                np.add.at(orbital_bars[name], key, bar)
                continue
            axes = cartesian_axes(key, self.shapes[name])
            if axes is not None:
                density = bar.reshape(tuple(len(x) for x in axes))
                self.counts['compact_density_elements'] = self.counts.get('compact_density_elements', 0)+bar.size
                self.counts['compact_index_elements'] = self.counts.get('compact_index_elements', 0)+sum(len(x) for x in axes)
                if name == 'metric':
                    p, q = axes
                    for left, right, weight in ((p, q, density), (q, p, density.T)):
                        for atom, rows in _atom_row_slices(left, base.ribf_atom):
                            metric_terms.setdefault(atom, []).append((left[rows], right, weight[rows]))
                else:
                    q, i, j = axes
                    for atom, rows in _atom_row_slices(q, base.ribf_atom):
                        ri_terms.setdefault(atom, []).append((name, q[rows], i, j, density[rows]))
                continue
            indices = index_arrays(key, self.shapes[name])
            if name == 'metric':
                p, q = indices
                for left, right in ((p.ravel(), q.ravel()), (q.ravel(), p.ravel())):
                    for atom in np.unique(base.ribf_atom[left]):
                        selected = base.ribf_atom[left] == atom
                        metric_terms.setdefault(int(atom), []).append(
                            (left[selected], right[selected], bar.ravel()[selected]))
                continue
            q, i, j = [x.ravel() for x in indices]
            weight = bar.ravel()
            for atom in np.unique(base.ribf_atom[q]):
                selected = np.flatnonzero(base.ribf_atom[q] == atom)
                w = weight[selected].copy()
                if name == 'qab':
                    w *= self.pao_pair_mask[i[selected], j[selected]]
                ri_terms.setdefault(int(atom), []).append((name, q[selected], i[selected], j[selected], w))
        # Accumulate all primitive bars before evaluating derivative
        # integrals.  This is the auxiliary-atom inverse map: each atom
        # has one extended AO block, independent of the pair-loop order.
        self.counts['ri_backtransform_atoms'] = len(ri_terms)
        self.counts['metric_backtransform_atoms'] = len(metric_terms)
        for atom, terms in ri_terms.items():
            self._pullback_aux_atom(atom, terms, coefficient_bars, force)
        for atom, terms in metric_terms.items():
            columns = np.unique(np.concatenate([q for _, q, _ in terms]))
            atoms = np.union1d([atom], np.unique(base.ribf_atom[columns]))
            aux = compact_molecule(base.auxmol, atoms)
            global_indices = np.concatenate([base.atom_ribfs[int(a)] for a in atoms])
            density = np.zeros((len(base.atom_ribfs[atom]), len(global_indices)))
            for p, q, weight in terms:
                indices = (np.searchsorted(base.atom_ribfs[atom], p), np.searchsorted(global_indices, q))
                np.add.at(density, np.ix_(*indices) if weight.ndim == 2 else indices, weight)
            local = int(np.searchsorted(atoms, atom))
            first, last = aux.aoslice_by_atom()[local, :2]
            derivative = aux.intor('int2c2e_ip1', comp=3,
                                    shls_slice=(int(first), int(last), 0, aux.nbas))
            force[atom] -= np.einsum('PQ,xPQ->x', density, derivative)
        self.counts['local_ri_backtransform_seconds'] = time.perf_counter()-start
        start = time.perf_counter()
        sources = orbital_pullback(base, coefficient_bars, orbital_bars)
        self.counts['orbital_source_seconds'] = time.perf_counter()-start
        return sources, force

    def _pullback_aux_atom(self, atom, terms, coefficient_bars, force):
        base = self.base
        kinds = dict(qij=('lmo', 'lmo'), qia=('lmo', 'pao'), qab=('pao', 'pao'))
        columns = dict(lmo=set(), pao=set())
        for name, _, i, j, _ in terms:
            left, right = kinds[name]
            columns[left].update(i)
            columns[right].update(j)
        columns = {k: np.asarray(sorted(v), dtype=int) for k, v in columns.items() if v}
        ao_atoms = np.unique(np.concatenate([self.maps.union(k, v) for k, v in columns.items()]))
        rows = np.concatenate([base.atom_bfs[int(a)] for a in ao_atoms])
        coefficients = {k: self.maps.matrix(k, v, ao_atoms)[1] for k, v in columns.items()}
        nq = len(base.atom_ribfs[atom])
        # H is the one-index-backtransformed density.  Symmetry of B_mnQ
        # combines its left and right coefficient sources in one tensor.
        h = {k: np.zeros((nq, len(rows), len(v))) for k, v in columns.items()}
        for name, q, i, j, weight in terms:
            kind1, kind2 = kinds[name]
            left, right = np.unique(i), np.unique(j)
            p1, p2 = np.searchsorted(columns[kind1], left), np.searchsorted(columns[kind2], right)
            density = np.zeros((nq, len(left), len(right)))
            indices = (np.searchsorted(base.atom_ribfs[atom], q),
                       np.searchsorted(left, i), np.searchsorted(right, j))
            if weight.ndim == 3:
                if name == 'qab':
                    weight = weight*self.pao_pair_mask[i[:, None], j][None]
                indices = np.ix_(*indices)
            np.add.at(density, indices, weight)
            h[kind1][:, :, p1] += np.einsum('nj,Qij->Qni', coefficients[kind2][:, p2], density)
            h[kind2][:, :, p2] += np.einsum('mi,Qij->Qmj', coefficients[kind1][:, p1], density)
        raw = self.raw(ao_atoms, [atom])
        gamma = np.zeros_like(raw)
        self.counts['largest_backtransform_ao'] = max(self.counts.get('largest_backtransform_ao', 0), len(rows))
        self.counts['peak_local_density_bytes'] = max(self.counts.get('peak_local_density_bytes', 0),
                                                     gamma.nbytes+sum(v.nbytes for v in h.values()))
        for kind, selected in columns.items():
            gradient = np.einsum('mnQ,Qni->mi', raw, h[kind], optimize=True)
            gradient *= self.maps.mask(kind, selected, rows)
            coefficient_bars[kind][np.ix_(rows, selected)] += gradient
            gamma += .5*np.einsum('mi,Qni->mnQ', coefficients[kind], h[kind], optimize=True)
        derivative = self.raw(ao_atoms, [atom], 'int3c2e_ip1', 3)
        centers = -np.einsum('mnQ,xmnQ->xm', gamma+gamma.transpose(1, 0, 2), derivative)
        for xyz in range(3):
            np.add.at(force[:, xyz], base.bf_atom[rows], centers[xyz])
        derivative = self.raw(ao_atoms, [atom], 'int3c2e_ip2', 3)
        force[atom] -= np.einsum('mnQ,xmnQ->x', gamma, derivative)


def dipole_objective(base, orb, r, weights):
    """Scalar distant-pair Lagrangian; no n*n array-version graph.

    This retains the energy's exact distant-pair sum.  Its quadratic pair
    count is still an explicit obstacle for the strict scaling objective.
    """
    if not np.any(weights):
        return 0.
    cl, p, opt = orb['cl'], orb['p'], base.opt
    centers = ad.einsum('mi,xmn,ni->ix', cl, r, cl, optimize=True)
    lp = ad.einsum('mi,xmn,nu->ixu', cl, r, p, optimize=True)
    needed = np.flatnonzero(np.any(weights != 0, axis=0) | np.any(weights != 0, axis=1))
    ds, es = {}, {}
    for i in needed:
        dom = su.contract_to_atoms(np.flatnonzero(abs(base.doi_iu[i]) > opt['T_CUT_DO_PRE']),
                                    base.bf_atom, base.atom_bfs)
        x, e = orthocanonicalize(orb['sp'][np.ix_(dom, dom)], orb['fp'][np.ix_(dom, dom)], opt['S_CUT'])
        ds[i], es[i] = lp[i][:, dom]@x, e
    result = 0.
    for position, i in enumerate(needed):
        for j in needed[position+1:]:
            weight = weights[i, j]+weights[j, i]
            if weight == 0:
                continue
            rij = centers[i]-centers[j]
            distance = np.sqrt(np.sum(rij*rij))
            unit = rij/distance
            dip = ds[i].T@ds[j]-3*np.outer(unit@ds[i], unit@ds[j])
            den = es[i][:, None]+es[j][None, :]-orb['fl'][i, i]-orb['fl'][j, j]
            result += -4*weight/distance**6*np.sum(dip**2/den)
    return result


def orbital_pullback(base, coefficient_bars, primitive_bars):
    """Backtransform two-index orbital sources, separately from local RI."""
    mf = base.mf.to_cpu() if hasattr(base.mf, 'to_cpu') else base.mf
    with base.mol.with_common_orig((0., 0., 0.)):
        r0 = base.mol.intor('int1e_r', comp=3)
    orb = getattr(base, '_orbital_reference', base.orb)
    inputs = dict(s=ad.Variable(orb.S), f=ad.Variable(orb.F),
                   co=ad.Variable(np.asarray(mf.mo_coeff)[:, np.asarray(mf.mo_occ)>0]),
                   cl=ad.Variable(orb.C_lmo), r=ad.Variable(r0))
    s, f, co, cl, r = [inputs[k] for k in ('s', 'f', 'co', 'cl', 'r')]
    p = np.eye(len(s))-co@co.T@s
    p = p/np.sqrt(np.diag(p.T@s@p))[None, :]
    sp, fp, fl = p.T@s@p, p.T@f@p, cl.T@f@cl
    objective = (np.sum(p*coefficient_bars['pao'])+np.sum(cl*coefficient_bars['lmo'])
                 + np.sum(sp*primitive_bars['s_pao'])+np.sum(fp*primitive_bars['f_pao'])
                 + np.sum(fl*primitive_bars['f_lmo']))
    if 'dipole' in primitive_bars:
        objective += dipole_objective(base, dict(cl=cl, p=p, sp=sp, fp=fp, fl=fl), r,
                                      primitive_bars['dipole'])
    if 'dipole_tail' in primitive_bars and np.any(primitive_bars['dipole_tail']):
        from .dipole_tail import orbital_inputs
        centers, ds, gs = orbital_inputs(base, dict(s=s, cl=cl, p=p, sp=sp, fp=fp, fl=fl), r,
                                         einsum=ad.einsum, canonicalize=orthocanonicalize)
        objective += primitive_bars['dipole_tail']*base.dipole_tail_driver.energy_ad(
            centers, ds, gs, exclusions=base._dipole_tail_exclusions, tree=base._dipole_tail_tree)
    bars = ad.gradients(objective, list(inputs.values()))
    return dict(zip(inputs, bars))
