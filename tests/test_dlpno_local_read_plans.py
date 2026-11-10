"""Dense-source oracle, repeated indices, selected VJPs and plan lifetime."""
from types import SimpleNamespace, MethodType
import json
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import host
from electrolites.dlpno.local_integrals import LocalRIBlocks
from electrolites.dlpno import local_read_plans as plans


def provider(xp=np):
    rng = np.random.default_rng(269)
    blocks = LocalRIBlocks.__new__(LocalRIBlocks)
    atoms = [np.arange(0, 3), np.arange(3, 6), np.arange(6, 9)]
    columns = {(name, atom): (np.arange(6), np.arange(7))
               for name in ('qia', 'qij', 'qab') for atom in range(3)}
    blocks.base = SimpleNamespace(atom_ribfs=atoms, ribf_atom=np.repeat(np.arange(3), 3),
                                 local_integrals=SimpleNamespace(column_envelopes=columns))
    blocks.sources = {name: rng.normal(size=(9, 6, 7)) for name in ('qia', 'qij', 'qab')}
    blocks.sources.update({name: rng.normal(size=(11, 13)) for name in ('s_pao', 'f_pao')})
    blocks.shapes = {k: v.shape for k, v in blocks.sources.items()}
    blocks.orbital_sources = {k: blocks.sources[k] for k in ('s_pao', 'f_pao')}
    blocks.xp, blocks.leaves, blocks.selections, blocks.counts = xp, {}, {}, {}
    blocks.orbital_tile_size = 4
    blocks.read = MethodType(lambda self, name, key: self.sources[name][key], blocks)
    return blocks, rng


@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('kind', ['qia', 'qij', 'qab'])
def test_fused_ri_gather_matches_dense_source_and_selected_vjps(monkeypatch, backend, cached, kind):
    xp = np if backend == 'numpy' else pytest.importorskip('cupy')
    monkeypatch.setitem(plans.OPTIONS, 'ri_fused', True)
    monkeypatch.setitem(plans.OPTIONS, 'ri_plan', cached)
    blocks, rng = provider(xp)
    keys = [np.ix_([7, 1, 7, 0, 4], [4, 0, 4], [6, 1, 1]),
            np.ix_([3, 4, 5, 0, 1, 2], np.arange(6), np.arange(7)),
            (np.array([8, 0, 8])[:, None], 2, np.array([5, 1])[None, :])]
    dense = ad.Variable(xp.asarray(blocks.sources[kind]))
    actual, _ = ad.checkpoint(lambda: tuple(blocks.tracked_read(kind, key) for key in keys))
    expected = [dense[key] for key in keys]
    seeds = [xp.asarray(rng.normal(size=x.shape)) for x in expected]
    gold, = ad.Linearization(expected, [dense])(seeds)
    leaves = list(blocks.leaves.values())
    graph = ad.Linearization(actual, leaves)
    errors = []
    for scale in (1., .37):
        bars = graph([scale*x for x in seeds])
        whole = xp.zeros_like(dense.v)
        for token, bar in zip(blocks.leaves, bars):
            xp.add.at(whole, blocks.selections[token], bar)
        errors.append(float(np.max(abs(host(whole)-scale*host(gold)))))
        assert errors[-1] < 1e-12
        for index, leaf in enumerate(leaves):
            selected, = ad.Linearization(actual, [leaf], boundary=leaves)([scale*x for x in seeds])
            assert np.max(abs(host(selected)-host(bars[index]))) < 1e-12
    consumed = graph(seeds, release_graph=True, release_inputs=True)
    assert all(leaf.v is None for leaf in leaves)
    if cached:
        assert blocks.counts['gather_plan_hits'] >= len(keys)
        assert blocks._gather_plan_bytes <= 64 << 20
    print('FUSED_RI_GATHER_REPORT '+json.dumps(dict(backend=backend, cached=cached, kind=kind,
        maximum_source_error=max(errors), consuming_inputs_released=True)))


@pytest.mark.parametrize('kind', ['s_pao', 'f_pao'])
def test_orbital_plans_bound_cache_and_use_current_leaves(monkeypatch, kind):
    monkeypatch.setitem(plans.OPTIONS, 'orbital_plan', True)
    blocks, rng = provider()
    blocks._gather_plan_limit = 2
    keys = [np.ix_([8, 0, 8], [12, 1, 12]), np.ix_([2, 7], [3, 5]), np.ix_([10, 3], [12, 0])]
    for key in keys:
        assert np.array_equal(blocks.tracked_read(kind, key).v, blocks.sources[kind][key])
    assert len(blocks._gather_plans) == 2 and blocks.counts['gather_plan_evictions'] >= 1
    # Clearing roots is allowed between independent graphs; cached metadata
    # must never resurrect a prior AD record or its primitive values.
    blocks.leaves.clear();blocks.selections.clear()
    blocks.sources[kind] += 1.25
    value = blocks.tracked_read(kind, keys[-1])
    assert np.array_equal(value.v, blocks.sources[kind][keys[-1]])
    assert blocks.counts['gather_plan_hits'] >= 1
    def check(x):
        assert not isinstance(x, (ad.Variable, ad.Record))
        if isinstance(x, (list, tuple)):
            for item in x:check(item)
        elif isinstance(x, dict):
            for key, item in x.items():check(key);check(item)
    check(blocks._gather_plans)


def test_ri_plan_invalidates_changed_envelope_and_respects_byte_budget(monkeypatch):
    monkeypatch.setitem(plans.OPTIONS, 'ri_fused', True)
    monkeypatch.setitem(plans.OPTIONS, 'ri_plan', True)
    blocks, _ = provider()
    env = blocks.base.local_integrals.column_envelopes
    env['qia', 0] = (np.array([0, 1, 2, 4]), np.arange(7))
    key = np.ix_([0, 2], [0, 1], [2, 1])
    expected = blocks.sources['qia'][key]
    assert np.array_equal(blocks.tracked_read('qia', key).v, expected)
    env['qia', 0][0][-1] = 5  # An in-place change must invalidate the plan.
    assert np.array_equal(blocks.tracked_read('qia', key).v, expected)
    assert blocks.counts['gather_plan_invalidations'] == 1
    blocks2, _ = provider();blocks2._gather_plan_budget = 1
    assert np.array_equal(blocks2.tracked_read('qia', key).v, blocks2.sources['qia'][key])
    assert not blocks2._gather_plans and blocks2._gather_plan_bytes == 0


def test_shape_bucket_planner_preserves_previous_sorted_order():
    from electrolites.dlpno.triples_source_batch_gpu import source_batches, _layout, workspace_bytes
    from test_dlpno_triples_source_batch_gpu import multi_state
    from test_dlpno_triples_gpu_sources import raw_sources
    state = multi_state(raw_sources()[0])
    layouts = [_layout(state, t) for t in range(len(state.trip))]
    expected = tuple(x['t'] for x in sorted(layouts, key=lambda d:(d['n'],d['q'],d['p'],d['l'])))
    assert tuple(t for group in source_batches(state, max_count=1) for t in group) == expected
