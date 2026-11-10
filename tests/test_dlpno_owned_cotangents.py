"""Explicit ownership never changes the Jacobian or ordinary alias behavior."""
from concurrent.futures import Future
import gc
import weakref
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.reverse_async import Pending


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('owned', [False, True])
def test_owned_repeated_operands_broadcast_selection_and_consumption(monkeypatch, asynchronous, owned):
    monkeypatch.setenv('DLPNO_ASYNC_SOURCE_REVERSE', str(int(asynchronous)))
    rng = np.random.default_rng(921)
    a = rng.normal(size=(4, 3)); b = rng.normal(size=(1, 3))
    x = ad.Variable(a.copy()); y = ad.Variable(b.copy()); unused = ad.Variable(np.ones(2))
    def pullback(bar, needed):
        # Deliberately repeated x operand; every returned array is disjoint.
        return tuple(value if flag else None for value, flag in
                     zip((bar*b, bar*(2*a), bar*b), needed))
    def submit(bar, needed):
        future = Future(); future.set_result(pullback(bar, needed))
        return Pending((future,), future.result)
    output = ad.operation(2*a*b, [x, y, x], lambda z: pullback(z, [True]*3), pullback,
                          async_pullback=submit, owns_cotangents=owned)
    seed = rng.normal(size=a.shape); seed_before = seed.copy()
    graph = ad.Linearization([output], [x, y, unused])
    first = graph([seed])
    expected = (2*seed*b, np.sum(2*seed*a, axis=0, keepdims=True), np.zeros(2))
    for u, v in zip(first, expected): np.testing.assert_allclose(u, v, rtol=0, atol=1e-14)
    assert (graph.last_statistics['adopted_elements'] > 0) == owned
    for u, v in zip(graph([.37*seed]), expected):np.testing.assert_allclose(u, .37*v, rtol=0, atol=1e-14)
    for u, v in zip(first, expected):np.testing.assert_allclose(u, v, rtol=0, atol=1e-14)
    chosen = ad.Linearization([output], [y], boundary=[x, y, unused])([seed])
    np.testing.assert_allclose(chosen[0], expected[1], rtol=0, atol=1e-14)
    da = rng.normal(size=a.shape); db = rng.normal(size=b.shape); h = 2e-5
    fd = (np.sum(2*(a+h*da)*(b+h*db)*seed)-np.sum(2*(a-h*da)*(b-h*db)*seed))/(2*h)
    assert abs(fd-np.sum(first[0]*da)-np.sum(first[1]*db)) < 1e-9
    np.testing.assert_array_equal(seed, seed_before)
    np.testing.assert_array_equal(x.v, a);np.testing.assert_array_equal(y.v, b)
    graph([seed], release_graph=True, release_inputs=True)
    assert x.v is y.v is unused.v is None
    with pytest.raises(RuntimeError, match='consumed'):graph([seed])


@pytest.mark.parametrize('asynchronous', [False, True])
def test_ordinary_aliased_views_still_copy(monkeypatch, asynchronous):
    monkeypatch.setenv('DLPNO_ASYNC_SOURCE_REVERSE', str(int(asynchronous)))
    x, y = ad.Variable(np.ones(5)), ad.Variable(np.ones(5))
    def pullback(z, needed):return z, z[::-1]
    def submit(z, needed):
        future = Future();future.set_result(pullback(z, needed));return Pending((future,), future.result)
    output = ad.operation(x.v+y.v, [x, y], lambda z: pullback(z, None), pullback, async_pullback=submit)
    seed = np.arange(5.)
    graph = ad.Linearization([output], [x, y]); a, b = graph([seed])
    a[:] = 100
    np.testing.assert_array_equal(b, seed[::-1]);np.testing.assert_array_equal(seed, np.arange(5.))
    assert graph.last_statistics['adopted_elements'] == 0


def test_async_drops_completed_sync_temporary_before_next_source(monkeypatch):
    monkeypatch.setenv('DLPNO_ASYNC_SOURCE_REVERSE', '1')
    seen = []; inspected = []
    x, y = ad.Variable(np.ones(8)), ad.Variable(np.ones(8))
    def early_pullback(z):
        temporary = 3*z
        seen.append(weakref.ref(temporary))
        return temporary,
    early = ad.operation(x.v.copy(), [x], early_pullback)
    def submit(z, needed):
        assert seen, 'Expected completed synchronous branch before this source'
        gc.collect()
        assert seen[-1]() is None, 'Finished synchronous contribution is retained across an async task'
        inspected.append(True)
        future = Future();future.set_result((2*z,));return Pending((future,), future.result)
    late = ad.operation(y.v.copy(), [y], lambda z: (2*z,), lambda z,n: (2*z,), async_pullback=submit)
    graph = ad.Linearization([early, late], [x, y])
    bars = graph([np.ones(8), np.ones(8)])
    assert inspected
    np.testing.assert_array_equal(bars[0], np.full(8, 3.))
    np.testing.assert_array_equal(bars[1], np.full(8, 2.))
