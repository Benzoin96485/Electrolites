"""PNO overlap batches must keep local indexing and the original algebra."""
import numpy as np
import pytest

from electrolites.dlpno.overlap_plan import overlap_chunks, scatter_rows, subset_positions


def test_overlap_plan_has_no_global_pair_or_ao_rows():
    # Large, sparse AO identifiers must not induce a universe-size mask.
    for count in (64, 128, 256):
        pao = [1_000_000_000 + 10*i + np.arange(4) for i in range(count)]
        left, right = np.arange(count-1), np.arange(1, count)
        nv = np.full(count, 3)
        parts = overlap_chunks(pao, nv, 2_000_000_000, left, right, max_bytes=32768)
        assert np.array_equal(np.sort(np.concatenate([p[0] for p in parts])), np.arange(count-1))
        total_rows = 0
        for selected, union, involved in parts:
            offsets, rows = scatter_rows(pao, union, involved)
            assert 16 * len(involved) * len(union) * 3 <= 32768
            assert len(union) <= 52
            for k, pair in enumerate(involved):
                assert np.array_equal(union[rows[offsets[k]:offsets[k]+4]], pao[pair])
            assert np.array_equal(involved[subset_positions(involved, left[selected])], left[selected])
            total_rows += len(rows)
        assert total_rows <= 8*count
    with pytest.raises(ValueError, match='omits'):
        subset_positions(np.array([3, 8]), np.array([4]))
    with pytest.raises(ValueError, match='omits'):
        scatter_rows([np.array([1, 8])], np.array([1, 3]), np.array([0]))
    assert overlap_chunks([], [], 0, [], []) == []


@pytest.mark.parametrize('extra', [0, 2])
def test_gpu_overlap_compact_scatter_against_dense_products(extra):
    cp = pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('No CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('No CUDA device')
    from electrolites.dlpno.ovgen import OverlapGen
    from electrolites.dlpno.ragged import Layout, Ragged
    rng = np.random.default_rng(321)
    nbf = 101
    a = rng.normal(size=(nbf, nbf))
    metric = a@a.T/nbf
    pao = [rng.choice(nbf, size=n, replace=False) for n in (4, 7, 3, 6, 5)]
    coeff = [rng.normal(size=(len(p)+extra, n)) for p, n in zip(pao, (2, 3, 1, 4, 2))]
    left = np.array([4, 2, 0, 3, 1, 4, 0, 3])
    right = np.array([0, 2, 1, 4, 3, 1, 0, 2])
    gen = OverlapGen(cp.asarray(metric), pao, coeff, ne=extra, max_bytes=5000)
    layout = Layout(gen.nv[left], gen.nv[right])
    out = Ragged(layout)
    plans = gen.prepare(left, right, out_layout=layout, out_idx=np.arange(len(left)))
    assert len(plans) > 1
    for plan in plans:
        assert plan['rows'].size == sum(len(pao[p]) for p in plan['inv'])
        gen.run(plan, out)
    for index, (p, q) in enumerate(zip(left, right)):
        exact = coeff[p][:len(pao[p])].T@metric[np.ix_(pao[p], pao[q])]@coeff[q][:len(pao[q])]
        if extra:
            exact += coeff[p][-extra:].T@coeff[q][-extra:]
        actual = cp.asnumpy(out.data[layout.group[index]][layout.slot[index]])
        assert np.max(abs(actual[:len(exact), :exact.shape[1]]-exact)) < 2e-12
    # The same compact row map must remain valid after workspace reuse.
    for block in out.data:
        block.fill(0.)
    for plan in reversed(plans):
        gen.run(plan, out)
    for index, (p, q) in enumerate(zip(left, right)):
        exact = coeff[p][:len(pao[p])].T@metric[np.ix_(pao[p], pao[q])]@coeff[q][:len(pao[q])]
        if extra:
            exact += coeff[p][-extra:].T@coeff[q][-extra:]
        actual = cp.asnumpy(out.data[layout.group[index]][layout.slot[index]])
        assert np.max(abs(actual[:len(exact), :exact.shape[1]]-exact)) < 2e-12


def test_local_triplet_projection_matches_full_ao_oracle():
    cp = pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('No CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('No CUDA device')
    from electrolites.dlpno.local_gpu import project_local_triplet_pairs
    rng = np.random.default_rng(527)
    metric = cp.asarray(rng.normal(size=(87,87)))
    metric = metric+metric.T
    indices = cp.asarray(rng.integers(0,87,size=(4,6)))
    mask = cp.asarray(rng.integers(0,2,size=(4,6)),dtype=float)
    coefficients = cp.asarray(rng.normal(size=(4,6,3)))
    tables = dict(PP=5,VP=4,dom=cp.asarray(rng.integers(0,87,size=(7,5))),
                  dmk=cp.asarray(rng.integers(0,2,size=(7,5)),dtype=float),
                  X=cp.asarray(rng.normal(size=(7,5,4))))
    b, p = cp.asarray([3,0,1,1,2,0,3]), cp.asarray([2,6,0,4,5,3,1])
    full = cp.matmul(metric[:,indices].transpose(1,0,2)*mask[:,None,:],coefficients)
    oracle = cp.matmul(tables['X'][p].transpose(0,2,1),
                       full[b[:,None],tables['dom'][p]]*tables['dmk'][p][:,:,None])
    source = metric,indices,mask,coefficients
    for budget in (100,1 << 28):
        actual = project_local_triplet_pairs(source,tables,b,p,max_bytes=budget)
        assert float(cp.max(abs(actual-oracle))) < 2e-12
