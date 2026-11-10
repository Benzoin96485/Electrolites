"""Batched GPU chain VJPs against independent per-term NumPy derivatives."""
import numpy as np
import pytest


@pytest.fixture(scope='module', autouse=True)
def cuda():
    cp = pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('A CUDA device is required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('A CUDA device is required')


def make_ragged(layout, rng):
    import cupy as cp
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.ragged import Ragged
    from electrolites.dlpno.ragged_adjoint import AdjointRagged
    blocks = [rng.normal(size=(r, c)) for r, c in zip(layout.rows, layout.cols)]
    numeric = Ragged.from_list(layout, blocks)
    variable = ad.Variable(numeric.flat())
    return AdjointRagged(layout).set_flat(variable), variable, blocks


def fetched(operand, blocks, term):
    value = blocks[operand.idx[term]]
    if operand.row is not None:
        value = value[operand.row[term]:operand.row[term]+1]
    return value.T if operand.trans else value


def oracle(operands, blocks, out_idx, out_row, output, seeds, weights, alpha):
    result = [x.copy() for x in output]
    bars = [[np.zeros_like(x) for x in values] for values in blocks]
    weight_bar = np.zeros(len(out_idx))
    tangent_dot = 0.
    rng = np.random.default_rng(190)
    tangent = [[rng.normal(size=x.shape) for x in values] for values in blocks]
    for term, target in enumerate(out_idx):
        factors = [fetched(o, v, term) for o, v in zip(operands, blocks)]
        product = np.linalg.multi_dot(factors) if len(factors)>2 else factors[0]@factors[1]
        seed = seeds[target] if out_row is None else seeds[target][out_row[term]].reshape(product.shape)
        if out_row is None:
            result[target] += alpha*weights[term]*product
        else:
            result[target][out_row[term]] += alpha*weights[term]*product.ravel()
        weight_bar[term] = alpha*np.sum(seed*product)
        for k, operand in enumerate(operands):
            prefix = np.eye(factors[0].shape[0])
            for factor in factors[:k]:prefix = prefix@factor
            suffix = np.eye(factors[-1].shape[1])
            for factor in factors[:k:-1]:suffix = factor@suffix
            delta = prefix@fetched(operand, tangent[k], term)@suffix
            tangent_dot += alpha*weights[term]*np.sum(seed*delta)
            bar = alpha*weights[term]*(prefix.T@seed@suffix.T)
            if operand.trans:bar = bar.T
            if operand.row is None:
                bars[k][operand.idx[term]] += bar
            else:
                bars[k][operand.idx[term]][operand.row[term]] += bar.ravel()
    return result, bars, weight_bar, tangent, tangent_dot


@pytest.mark.parametrize('flags', [(False, False), (True, False, True), (False, True, True, False, True, False)])
def test_repeated_rectangular_chain_and_weight_vjp(flags):
    import cupy as cp
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.ragged import Layout, Ragged, op
    from electrolites.dlpno.ragged_adjoint import AdjointChainPlan
    rng = np.random.default_rng(1901)
    layout = Layout([3, 5, 3, 7, 5, 7], [3, 5, 3, 7, 5, 7])
    indices = np.array([0, 1, 2, 3, 4, 5, 0, 4, 2, 5])
    partner = np.array([2, 4, 0, 5, 1, 3])
    operands = [op(layout, indices if k%2==0 else partner[indices], flag) for k, flag in enumerate(flags)]
    triples = [make_ragged(layout, rng) for _ in flags]
    rags, variables, blocks = map(list, zip(*triples))
    out, initial, output = make_ragged(layout, rng)
    _, _, seeds = make_ragged(layout, rng)
    weights = rng.normal(size=len(indices)); weights[2] = 0.
    w = ad.Variable(cp.asarray(weights));alpha = -.37
    plan = AdjointChainPlan(layout, indices, operands)
    actual = plan.run(out, rags, w=w, alpha=alpha)
    seed = Ragged.from_list(layout, seeds).flat()
    linearization = ad.Linearization([actual.flat()], [initial, *variables, w])
    result, expected_bars, expected_w, tangent, tangent_dot = oracle(
        operands, blocks, indices, None, output, seeds, weights, alpha)
    assert np.max(abs(cp.asnumpy(actual.flat().v-Ragged.from_list(layout, result).flat()))) < 1e-10
    for repeat in range(2):
        gradients = linearization([seed])
        assert np.max(abs(cp.asnumpy(gradients[0]-seed))) < 1e-12
        for g, b in zip(gradients[1:-1], expected_bars):
            assert np.max(abs(cp.asnumpy(g-Ragged.from_list(layout, b).flat()))) < 1e-10
        assert np.max(abs(cp.asnumpy(gradients[-1])-expected_w)) < 1e-10
        dot = sum(float(cp.dot(g, Ragged.from_list(layout, t).flat()))
                  for g, t in zip(gradients[1:-1], tangent))
        assert abs(dot-tangent_dot) < 1e-9


def test_selected_input_and_output_rows_with_transpose():
    import cupy as cp
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.ragged import Layout, Ragged, op
    from electrolites.dlpno.ragged_adjoint import AdjointChainPlan
    rng = np.random.default_rng(1902)
    a, b, c = Layout([3, 5, 3], [2, 4, 2]), Layout([3]*3, [2, 4, 2]), Layout([4]*3, [3, 5, 3])
    indices = np.array([0, 1, 0, 2, 1]);rows = np.array([0, 1, 0, 2, 1]);out_rows = np.array([0, 1, 0, 3, 1])
    operands = [op(a, indices), op(b, indices, True, row=rows)]
    av, ax, ab = make_ragged(a, rng);bv, bx, bb = make_ragged(b, rng)
    out, initial, output = make_ragged(c, rng);_, _, seeds = make_ragged(c, rng)
    weights = np.array([1., -.3, .2, 0., -2.]);w = ad.Variable(cp.asarray(weights))
    plan = AdjointChainPlan(c, indices, operands, out_row=out_rows)
    plan.run(out, [av, bv], w=w, alpha=.73)
    result, bars, weight_bar, _, _ = oracle(operands, [ab, bb], indices, out_rows, output, seeds, weights, .73)
    gradients = ad.Linearization([out.flat()], [ax, bx, w])([Ragged.from_list(c, seeds).flat()])
    assert np.max(abs(cp.asnumpy(out.flat().v-Ragged.from_list(c, result).flat()))) < 1e-11
    for actual, expected, layout in zip(gradients[:2], bars, (a, b)):
        assert np.max(abs(cp.asnumpy(actual-Ragged.from_list(layout, expected).flat()))) < 1e-11
    assert np.max(abs(cp.asnumpy(gradients[2])-weight_bar)) < 1e-11


def test_dot_plan_differentiates_both_sides():
    import cupy as cp
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.ragged import Layout, Ragged, op
    from electrolites.dlpno.ragged_adjoint import AdjointDotPlan
    rng = np.random.default_rng(1903)
    layout = Layout([3, 5, 3], [3, 5, 3]);indices = np.array([0, 1, 0, 2, 1])
    right = [op(layout, indices, True), op(layout, indices), op(layout, indices, True)]
    triples = [make_ragged(layout, rng) for _ in range(4)]
    rags, variables, blocks = map(list, zip(*triples))
    plan = AdjointDotPlan([op(layout, indices)], right)
    value = plan.run(rags[:1], rags[1:]);seeds = rng.normal(size=len(indices))
    gradients = ad.Linearization([value], variables)([cp.asarray(seeds)])
    zero = [np.zeros_like(x) for x in blocks[0]]
    result, bars, _, _, _ = oracle(right, blocks[1:], indices, None, zero, blocks[0], seeds, 1.)
    assert np.max(abs(cp.asnumpy(gradients[0]-Ragged.from_list(layout, result).flat()))) < 1e-11
    for actual, expected in zip(gradients[1:], bars):
        assert np.max(abs(cp.asnumpy(actual-Ragged.from_list(layout, expected).flat()))) < 1e-11
    expected = [np.sum(fetched(op(layout, indices), blocks[0], t)*
        np.linalg.multi_dot([fetched(o, b, t) for o, b in zip(right, blocks[1:])])) for t in range(len(indices))]
    assert np.max(abs(cp.asnumpy(value.v)-expected)) < 1e-11


def test_constant_overlap_factory_is_replayed_for_vjp():
    import cupy as cp
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.ragged import Layout, Ragged, op
    from electrolites.dlpno.ragged_adjoint import AdjointChainPlan, AdjointRagged
    rng = np.random.default_rng(1904)
    layout = Layout([3, 5], [3, 5]);indices = np.array([0, 1, 0])
    _, _, constants = make_ragged(layout, rng)
    b, variable, blocks = make_ragged(layout, rng)
    calls = []
    def regenerate():
        calls.append(1)
        return Ragged.from_list(layout, constants)
    out = AdjointRagged(layout)
    plan = AdjointChainPlan(layout, indices, [op(layout, indices), op(layout, indices)])
    plan.run(out, [None, b], constant_factories={0:regenerate})
    graph = ad.Linearization([out.flat()], [variable]);seed = cp.ones(out.flat().size)
    actual = graph([seed])[0]
    expected = [np.zeros_like(x) for x in blocks]
    for i in indices:expected[i] += constants[i].T@np.ones_like(blocks[i])
    assert np.max(abs(cp.asnumpy(actual-Ragged.from_list(layout, expected).flat()))) < 1e-11
    assert len(calls) == 2
