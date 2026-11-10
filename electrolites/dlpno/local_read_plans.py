"""Exact coarse gather primitives and bounded metadata-only read plans.

Plans never retain AD records or primitive arrays. Source leaves are looked
up again on each call and replay. Cached RI plans validate their envelope
contents, so replacing or mutating a domain invalidates the old plan.
"""
from collections import OrderedDict
import json
import os
import sys
import numpy as np
from . import reverse as ad
from .deferred_reads import DeferredRead,active as deferred_active
from .local_blocks import ReferenceBlocks, _key_token

OPTIONS = json.loads(os.environ.get('DLPNO_READ_SCREEN_OPTIONS', '{}'))
if set(OPTIONS)-{'ri_fused', 'ri_plan', 'orbital_plan', 'overlap_replay'}:
    raise ValueError('Unknown local-read screening option')


def _bytes(value):
    if isinstance(value, np.ndarray):
        return sys.getsizeof(value)+value.nbytes
    if isinstance(value, (tuple, list)):
        return sys.getsizeof(value)+sum(_bytes(x) for x in value)
    if isinstance(value, dict):
        return sys.getsizeof(value)+sum(_bytes(k)+_bytes(v) for k, v in value.items())
    return sys.getsizeof(value)


def _get_plan(blocks, key, build, valid, enabled):
    if not enabled:
        return build()
    cache = getattr(blocks, '_gather_plans', None)
    if cache is None:
        cache = blocks._gather_plans = OrderedDict()
        blocks._gather_plan_bytes = 0
    if key in cache:
        plan, size = cache.pop(key)
        if valid(plan):
            cache[key] = (plan, size)
            blocks.counts['gather_plan_hits'] = blocks.counts.get('gather_plan_hits', 0)+1
            return plan
        blocks._gather_plan_bytes -= size
        blocks.counts['gather_plan_invalidations'] = blocks.counts.get('gather_plan_invalidations', 0)+1
    blocks.counts['gather_plan_misses'] = blocks.counts.get('gather_plan_misses', 0)+1
    plan = build()
    if plan is None:
        return None
    size = _bytes((key, plan))+256
    budget = getattr(blocks, '_gather_plan_budget', 64 << 20)
    limit = getattr(blocks, '_gather_plan_limit', 2048)
    if size > budget or limit < 1:
        return plan
    while cache and (blocks._gather_plan_bytes+size > budget or len(cache) >= limit):
        _, (_, old_size) = cache.popitem(last=False)
        blocks._gather_plan_bytes -= old_size
        blocks.counts['gather_plan_evictions'] = blocks.counts.get('gather_plan_evictions', 0)+1
    cache[key] = (plan, size)
    blocks._gather_plan_bytes += size
    blocks.counts['gather_plan_peak_bytes'] = max(blocks.counts.get('gather_plan_peak_bytes', 0), blocks._gather_plan_bytes)
    blocks.counts['gather_plan_peak_entries'] = max(blocks.counts.get('gather_plan_peak_entries', 0), len(cache))
    return plan


def tracked_ri_read(blocks, name, key):
    from .local_integrals import cartesian_axes, index_arrays, _atom_row_slices
    shape = blocks.shapes[name]
    parent = getattr(blocks.base, 'local_integrals', None)
    envelopes = getattr(parent, 'column_envelopes', {})

    def build():
        axes = cartesian_axes(key, shape)
        if axes is None or not all(len(x) for x in axes):
            return None
        q, left, right = axes
        groups = list(_atom_row_slices(q, blocks.base.ribf_atom))
        validations, parts = [], []
        for atom, rows in groups:
            columns = envelopes.get((name, atom))
            if columns is None or not (np.isin(left, columns[0]).all() and np.isin(right, columns[1]).all()):
                blocks.counts['unshared_'+name+'_reads'] = blocks.counts.get('unshared_'+name+'_reads', 0)+1
                return None
            full_q = blocks.base.atom_ribfs[atom]
            c1, c2 = columns
            selected_q = q[rows]
            identity = np.array_equal(selected_q, full_q) and np.array_equal(left, c1) and np.array_equal(right, c2)
            canonical = np.ix_(full_q, c1, c2)
            local = None if identity else np.ix_(np.searchsorted(full_q, selected_q), np.searchsorted(c1, left), np.searchsorted(c2, right))
            parts.append((canonical, local, rows, identity))
            validations.append((atom, c1.copy(), c2.copy(), full_q.copy()))
        return dict(cartesian_shape=tuple(map(len, axes)),
                    output_shape=index_arrays(key, shape)[0].shape,
                    parts=tuple(parts), validations=tuple(validations))

    def valid(plan):
        for atom, c1, c2, full_q in plan['validations']:
            now = envelopes.get((name, atom))
            if now is None or not np.array_equal(now[0], c1) or not np.array_equal(now[1], c2) or not np.array_equal(blocks.base.atom_ribfs[atom], full_q):
                return False
        return True

    token = ('ri', name, shape, _key_token(key)) if OPTIONS.get('ri_plan') else None
    plan = _get_plan(blocks, token, build, valid, OPTIONS.get('ri_plan', False))
    if plan is None:
        return ReferenceBlocks.tracked_read(blocks, name, key)
    if deferred_active():
        sources=[];parts=[]
        for canonical,local,rows,identity in plan['parts']:
            sources.append(ReferenceBlocks.tracked_read(blocks,name,canonical))
            parts.append((None if identity else local,rows))
        blocks.counts['deferred_ri_reads']=blocks.counts.get('deferred_ri_reads',0)+1
        return DeferredRead(sources,parts,plan['cartesian_shape'],plan['output_shape'])
    sources, scatter = [], []
    value = None
    for canonical, local, rows, identity in plan['parts']:
        source = ReferenceBlocks.tracked_read(blocks, name, canonical)
        if value is None:
            value = blocks.xp.empty(plan['cartesian_shape'], dtype=source.v.dtype)
        value[rows] = source.v if identity else source.v[local]
        sources.append(source)
        scatter.append((source.shape, local, rows, identity))
        if identity:
            counter = 'shared_'+name+'_identity_parts'
            blocks.counts[counter] = blocks.counts.get(counter, 0)+1
            counter = 'shared_'+name+'_identity_elements'
            blocks.counts[counter] = blocks.counts.get(counter, 0)+source.size

    def selected_pullback(bar, needed):
        bar = bar.reshape(plan['cartesian_shape'])
        return tuple((bar[rows] if identity else ad.IndexedCotangent(source_shape, local, bar[rows])) if need else None
                     for (source_shape, local, rows, identity), need in zip(scatter, needed))
    blocks.counts['shared_'+name+'_reads'] = blocks.counts.get('shared_'+name+'_reads', 0)+1
    blocks.counts['fused_ri_gather_records'] = blocks.counts.get('fused_ri_gather_records', 0)+1
    return ad.operation(value.reshape(plan['output_shape']), sources,
                        lambda bar: selected_pullback(bar, [True]*len(sources)), selected_pullback)


def tracked_orbital_read(blocks, name, key):
    from .local_integrals import cartesian_axes, index_arrays
    shape, size = blocks.shapes[name], blocks.orbital_tile_size

    def build():
        axes = cartesian_axes(key, shape)
        if (axes is None or not all(len(x) for x in axes)
                or any(not np.issubdtype(x.dtype, np.integer) or np.any(x < 0) or np.any(x >= n) for x, n in zip(axes, shape))):
            return None
        groups = [[(int(tile), np.flatnonzero(axis//size == tile)) for tile in np.unique(axis//size)] for axis in axes]
        parts = []
        for row_tile, rows in groups[0]:
            for col_tile, cols in groups[1]:
                canonical = (slice(row_tile*size, min((row_tile+1)*size, shape[0])),
                             slice(col_tile*size, min((col_tile+1)*size, shape[1])))
                local = np.ix_(axes[0][rows]-row_tile*size, axes[1][cols]-col_tile*size)
                parts.append((canonical, local, np.ix_(rows, cols)))
        return dict(parts=tuple(parts), cartesian_shape=tuple(map(len, axes)), output_shape=index_arrays(key, shape)[0].shape)

    plan = _get_plan(blocks, ('orbital', name, shape, size, _key_token(key)), build, lambda plan: True, True)
    if plan is None:
        return ReferenceBlocks.tracked_read(blocks, name, key)
    if deferred_active():
        sources=[];parts=[]
        for canonical,local,output in plan['parts']:
            sources.append(ReferenceBlocks.tracked_read(blocks,name,canonical))
            parts.append((local,output))
        blocks.counts['deferred_orbital_reads']=blocks.counts.get('deferred_orbital_reads',0)+1
        return DeferredRead(sources,parts,plan['cartesian_shape'],plan['output_shape'])
    sources, scatter = [], []
    value = blocks.xp.empty(plan['cartesian_shape'], dtype=blocks.orbital_sources[name].dtype)
    for canonical, local, output in plan['parts']:
        source = ReferenceBlocks.tracked_read(blocks, name, canonical)
        value[output] = source.v[local]
        sources.append(source)
        scatter.append((source.shape, local, output))
    def selected_pullback(bar, needed):
        bar = bar.reshape(plan['cartesian_shape'])
        return tuple(ad.IndexedCotangent(source_shape, local, bar[output]) if need else None
                     for (source_shape, local, output), need in zip(scatter, needed))
    blocks.counts['shared_'+name+'_reads'] = blocks.counts.get('shared_'+name+'_reads', 0)+1
    blocks.counts['orbital_gather_records'] = blocks.counts.get('orbital_gather_records', 0)+1
    return ad.operation(value.reshape(plan['output_shape']), sources,
                        lambda bar: selected_pullback(bar, [True]*len(sources)), selected_pullback)
