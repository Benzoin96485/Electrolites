"""Reverse tensor contractions for local-correlation Lagrangians.

Records are immutable versions, including indexed assignments.  A VJP
linearization retains only paths from the requested inputs to its outputs;
amplitude lambda iterations therefore never revisit the integral/PNO
construction or differentiate through a nonlinear iteration history.
The same records support NumPy and CuPy.  This module does not introduce
a nuclear-coordinate axis or materialize an amplitude Jacobian.
"""
from itertools import count
from types import SimpleNamespace

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from .dual import array_module, eigenvalue_tolerance


_ids = count()


class IndexedCotangent:
    """A local scatter into an existing adjoint buffer.

    Slicing a global amplitude vector must not allocate a global zero
    vector for every local amplitude block: that would be quadratic work
    in an otherwise local VJP.  The linearization allocates the parent
    buffer once and scatters each contribution directly into it.
    """

    def __init__(self, shape, key, values):
        self.shape, self.key, self.values = shape, key, values

    def add_to(self, target):
        _xp([target]).add.at(target, self.key, self.values)


class Record:
    __slots__ = ('value', 'parents', 'pullback', 'selected_pullback', 'tracked', 'id')

    def __init__(self, value, parents=(), pullback=None, tracked=True, selected_pullback=None):
        self.value = value
        self.parents = tuple(parents)
        self.pullback = pullback
        self.selected_pullback = selected_pullback
        self.tracked = tracked
        self.id = next(_ids)


def primal(x):
    return x.v if isinstance(x, Variable) else x


def _xp(xs):
    for x in xs:
        mod = array_module(primal(x))
        if mod is not np:
            return mod
    return np


def _unbroadcast(value, shape):
    if isinstance(value, IndexedCotangent):
        if value.shape != shape:
            raise ValueError('An indexed cotangent cannot be broadcast')
        return value
    while value.ndim > len(shape):
        value = value.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and value.shape[axis] != 1:
            value = value.sum(axis=axis, keepdims=True)
    return value.reshape(shape)


def operation(value, operands, pullback, selected_pullback=None):
    """Create a record; optionally compute only requested operand bars."""
    active = [(i, x._record) for i, x in enumerate(operands)
              if isinstance(x, Variable) and x._record.tracked]
    if not active:
        return Variable(value, tracked=False)

    if selected_pullback is None:
        def reverse(bar):
            all_bars = pullback(bar)
            return tuple(_unbroadcast(all_bars[i], record.value.shape) for i, record in active)
        return Variable._from_record(Record(value, [r for _, r in active], reverse))

    def reverse_selected(bar, needed):
        requested = [False]*len(operands)
        for need, (i, _) in zip(needed, active):
            requested[i] = need
        all_bars = selected_pullback(bar, requested)
        return tuple(_unbroadcast(all_bars[i], record.value.shape) if need else None
                     for need, (i, record) in zip(needed, active))

    def reverse(bar):
        return reverse_selected(bar, [True]*len(active))

    return Variable._from_record(Record(value, [r for _, r in active], reverse,
                                        selected_pullback=reverse_selected))


class Variable(NDArrayOperatorsMixin):
    __array_priority__ = 1000

    def __init__(self, value, tracked=True):
        xp = _xp([value])
        self._record = Record(xp.asarray(primal(value)), tracked=tracked)

    @classmethod
    def _from_record(cls, record):
        out = object.__new__(cls)
        out._record = record
        return out

    @property
    def v(self):
        return self._record.value

    @property
    def shape(self):
        return self.v.shape

    @property
    def ndim(self):
        return self.v.ndim

    @property
    def size(self):
        return self.v.size

    @property
    def dtype(self):
        return self.v.dtype

    @property
    def T(self):
        return self.transpose()

    def __len__(self):
        return len(self.v)

    def __array__(self, dtype=None, copy=None):
        raise TypeError('Implicit conversion of a reverse variable would lose its adjoint')

    def __bool__(self):
        return bool(self.v)

    def __format__(self, spec):
        return format(float(self.v), spec)

    def __getitem__(self, key):
        value, shape = self.v[key], self.shape

        def reverse(bar):
            return (IndexedCotangent(shape, key, bar),)

        result = operation(value, [self], reverse)
        spectrum = getattr(self, '_spectrum', None)
        if spectrum is not None and isinstance(key, tuple) and len(key) == 2 and result.ndim == 2:
            xp = _xp([self])
            rows, columns = key
            if isinstance(rows, slice) and rows == slice(None):
                selected = xp.arange(self.shape[1])[columns]
                discarded = xp.ones(self.shape[1], dtype=bool)
                discarded[selected] = False
                tolerance = getattr(self, '_spectrum_tolerance', None)
                if tolerance is None:
                    tolerance = eigenvalue_tolerance(spectrum)
                if selected.size and bool(xp.any(discarded)):
                    gap = abs(spectrum[selected, None] - spectrum[None, discarded])
                    if float(gap.min()) <= tolerance:
                        raise ValueError('PNO/TNO selection splits a degenerate eigenvalue block')
                result._spectrum = spectrum[columns]
                result._spectrum_tolerance = tolerance
        return result

    def __setitem__(self, key, other):
        # Capture immutable records before replacing this mutable handle.
        value = self.v.copy()
        value[key] = primal(other)

        def reverse(bar):
            old = bar.copy()
            old[key] = 0.
            return old, bar[key]

        self._record = operation(value, [self, other], reverse)._record

    def copy(self):
        return operation(self.v.copy(), [self], lambda b: (b,))

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __itruediv__(self, other):
        return self / other

    def reshape(self, *shape, **kw):
        original = self.shape
        return operation(self.v.reshape(*shape, **kw), [self], lambda b: (b.reshape(original),))

    def ravel(self, *args, **kw):
        original = self.shape
        return operation(self.v.ravel(*args, **kw), [self], lambda b: (b.reshape(original),))

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        inverse = tuple(np.argsort(axes))
        return operation(self.v.transpose(*axes), [self], lambda b: (b.transpose(*inverse),))

    def sum(self, axis=None, keepdims=False, **kw):
        shape, xp = self.shape, _xp([self])
        axes = tuple(range(self.ndim)) if axis is None else ((axis,) if np.isscalar(axis) else tuple(axis))
        axes = tuple(a % self.ndim for a in axes)

        def reverse(bar):
            if not keepdims:
                expanded = list(shape)
                for a in axes:
                    expanded[a] = 1
                bar = bar.reshape(expanded)
            return (xp.broadcast_to(bar, shape),)

        return operation(self.v.sum(axis=axis, keepdims=keepdims, **kw), [self], reverse)

    def max(self, *args, **kw):
        return self.v.max(*args, **kw)

    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        if method != '__call__' or kw:
            return NotImplemented
        xp = _xp(inputs)
        values = [xp.asarray(primal(x)) for x in inputs]
        a = values[0]
        if len(values) == 2:
            b = values[1]
        name = ufunc.__name__
        if name == 'add':
            return operation(a+b, inputs, lambda g: (g, g))
        if name == 'subtract':
            return operation(a-b, inputs, lambda g: (g, -g))
        if name == 'multiply':
            return operation(a*b, inputs, lambda g: (g*b, g*a),
                             lambda g, need: (g*b if need[0] else None, g*a if need[1] else None))
        if name in ('divide', 'true_divide'):
            return operation(a/b, inputs, lambda g: (g/b, -g*a/b**2),
                             lambda g, need: (g/b if need[0] else None,
                                              -g*a/b**2 if need[1] else None))
        if name == 'matmul':
            av = a
            if a.ndim == 1:
                av = a[None, :]
            bv = b[:, None] if b.ndim == 1 else b

            def reverse(g, need=(True, True)):
                if a.ndim == b.ndim == 1:
                    g = g.reshape(1, 1)
                elif a.ndim == 1:
                    g = xp.expand_dims(g, -2)
                elif b.ndim == 1:
                    g = xp.expand_dims(g, -1)
                ga = g @ xp.swapaxes(bv, -1, -2) if need[0] else None
                gb = xp.swapaxes(av, -1, -2) @ g if need[1] else None
                if a.ndim == 1 and need[0]:
                    ga = xp.squeeze(ga, -2)
                if b.ndim == 1 and need[1]:
                    gb = xp.squeeze(gb, -1)
                return ga, gb

            return operation(a@b, inputs, reverse, reverse)
        if name == 'power':
            if isinstance(inputs[1], Variable) and inputs[1]._record.tracked:
                return operation(a**b, inputs, lambda g: (g*b*a**(b-1), g*a**b*xp.log(a)))
            return operation(a**b, inputs, lambda g: (g*b*a**(b-1), xp.zeros_like(b)))
        if name == 'negative':
            return operation(-a, inputs, lambda g: (-g,))
        if name == 'positive':
            return operation(a, inputs, lambda g: (g,))
        if name in ('absolute', 'fabs'):
            return operation(abs(a), inputs, lambda g: (g*xp.sign(a),))
        if name == 'sqrt':
            value = xp.sqrt(a)
            return operation(value, inputs, lambda g: (g/(2*value),))
        if name == 'square':
            return operation(a*a, inputs, lambda g: (2*g*a,))
        if name in ('less', 'less_equal', 'greater', 'greater_equal', 'equal', 'not_equal', 'isfinite'):
            return getattr(xp, name)(*values)
        return NotImplemented

    def __array_function__(self, func, types, args, kw):
        if func.__name__ in _FUNCTIONS:
            return _FUNCTIONS[func.__name__](*args, **kw)
        return NotImplemented


class Linearization:
    """Reusable VJP graph, restricted to paths from ``inputs``."""

    def __init__(self, outputs, inputs=None, boundary=None):
        self._consumed = False
        self.outputs = [x._record for x in outputs]
        self.inputs = None if inputs is None else [x._record for x in inputs]
        stops = set() if self.inputs is None else set(self.inputs)
        if boundary is not None:
            stops.update(x._record for x in boundary)
        seen, ordered = set(), []
        todo = [(record, False) for record in self.outputs]
        while todo:
            record, expanded = todo.pop()
            if expanded:
                ordered.append(record)
                continue
            if record in seen or not record.tracked:
                continue
            if record.value is None:
                raise RuntimeError('A consumed reverse graph cannot be reused')
            seen.add(record)
            todo.append((record, True))
            if record not in stops:
                todo.extend((parent, False) for parent in record.parents)
        if self.inputs is None:
            self.active = seen
        else:
            self.active = set(self.inputs)
            for record in ordered:
                if record not in stops and any(p in self.active for p in record.parents):
                    self.active.add(record)
        self.records = [r for r in ordered if r in self.active]
        self.stops = stops
        self.statistics = dict(records=len(self.records),
                               value_elements=sum(r.value.size for r in self.records),
                               value_bytes=sum(r.value.nbytes for r in self.records))

    def __call__(self, seeds, *, release_graph=False, release_inputs=False):
        """Evaluate a VJP, optionally consuming an exclusively owned graph.

        Krylov actions keep the default reusable graph. A final density
        sweep may release each intermediate after all its children have
        been visited. Requested inputs and other explicit boundary
        records survive by default; the consumed graph must not be used
        again. ``release_inputs`` also discards requested primal inputs
        after their final use, retaining only their independent bars.
        """
        if self._consumed:
            raise RuntimeError('A consumed reverse graph cannot be reused')
        if release_graph and self.inputs is None:
            raise ValueError('Graph release requires explicit retained inputs')
        if release_inputs and not release_graph:
            raise ValueError('Input release requires a consuming graph sweep')
        self._consumed = release_graph
        input_records = set() if self.inputs is None else set(self.inputs)
        bars = {}
        stats = dict(allocated_elements=0, contribution_elements=0, indexed_contributions=0,
                     indexed_elements=0, skipped_parent_pullbacks=0,
                     released_records=0, released_value_bytes=0,
                     released_input_records=0, released_input_bytes=0)

        def release(record, is_input=False):
            if record.value is None:
                return
            if is_input and record not in bars:
                # Unused inputs still have a well-defined zero bar;
                # allocate it while shape/backend information survives.
                bars[record] = _xp([record.value]).zeros_like(record.value)
                stats['allocated_elements'] += bars[record].size
            stats['released_records'] += 1
            stats['released_value_bytes'] += record.value.nbytes
            if is_input:
                stats['released_input_records'] += 1
                stats['released_input_bytes'] += record.value.nbytes
            record.value = None
            record.parents = ()
            record.pullback = record.selected_pullback = None

        def accumulate(record, contribution):
            indexed = isinstance(contribution, IndexedCotangent)
            size = contribution.values.size if indexed else contribution.size
            stats['contribution_elements'] += size
            if record not in bars:
                if indexed:
                    bars[record] = _xp([record.value]).zeros_like(record.value)
                else:
                    # Pullbacks frequently return views of other bars.
                    # Own the accumulator before mutating it in place.
                    bars[record] = contribution.copy()
                stats['allocated_elements'] += bars[record].size
                if not indexed:
                    return
            if indexed:
                stats['indexed_contributions'] += 1
                stats['indexed_elements'] += size
                contribution.add_to(bars[record])
            else:
                bars[record] += contribution

        for record, seed in zip(self.outputs, seeds):
            if record in self.active:
                accumulate(record, seed)
        for record in reversed(self.records):
            bar = bars.get(record)
            if bar is not None and record.pullback is not None and record not in self.stops:
                if record.selected_pullback is None:
                    contributions = record.pullback(bar)
                else:
                    needed = [parent in self.active for parent in record.parents]
                    stats['skipped_parent_pullbacks'] += needed.count(False)
                    contributions = record.selected_pullback(bar, needed)
                for parent, contribution in zip(record.parents, contributions):
                    if parent in self.active:
                        accumulate(parent, contribution)
                if self.inputs is not None:
                    bars.pop(record, None)
            if release_graph and record not in self.stops:
                # Pullback closures retain primal arrays as well as
                # parent records. Both must go after the final use.
                release(record)
            elif release_inputs and record in input_records:
                release(record, is_input=True)
        if release_inputs:
            # Inputs disconnected from every output were never visited.
            for record in input_records:
                release(record, is_input=True)
        self.last_statistics = stats
        if self.inputs is None:
            return bars
        return [bars[r] if r in bars else _xp([r.value]).zeros_like(r.value) for r in self.inputs]


def gradients(output, inputs):
    return Linearization([output], inputs)([_xp([output]).ones_like(output.v)])


def checkpoint(function):
    """Replace a pure local tensor graph by a recomputed VJP.

    ``function`` returns a sequence of Variables and must retain exactly
    the same inputs when called during the reverse sweep. Records already
    present on entry, plus newly created primitive leaves, form its
    boundary. Inner records are discarded after their output values have
    been packed. No numerical differentiation or approximate adjoint is
    involved.
    """
    first_id = next(_ids)
    outputs = tuple(function())
    seen, boundary = set(), set()
    stack = [x._record for x in outputs]
    while stack:
        record = stack.pop()
        if record in seen or not record.tracked:
            continue
        seen.add(record)
        if record.id < first_id or not record.parents:
            boundary.add(record)
        else:
            stack.extend(record.parents)
    inputs = [Variable._from_record(r) for r in sorted(boundary, key=lambda r: r.id)]
    shapes = [x.shape for x in outputs]
    offsets = np.cumsum([0]+[x.size for x in outputs])
    xp = _xp(outputs)
    packed = xp.concatenate([x.v.ravel() for x in outputs])
    statistics = dict(records=len(seen-boundary), boundary_records=len(boundary),
                       inner_value_bytes=sum(r.value.nbytes for r in seen-boundary),
                       output_bytes=packed.nbytes)

    def pullback(seed, needed=None):
        replay = tuple(function())
        seeds = [seed[a:b].reshape(shape) for a, b, shape in zip(offsets[:-1], offsets[1:], shapes)]
        if needed is None:
            return Linearization(replay, inputs)(seeds)
        selected = [x for x, need in zip(inputs, needed) if need]
        bars = iter(Linearization(replay, selected, boundary=inputs)(seeds))
        return tuple(next(bars) if need else None for need in needed)

    result = operation(packed, inputs, pullback, pullback)
    return tuple(result[a:b].reshape(shape) for a, b, shape in zip(offsets[:-1], offsets[1:], shapes)), statistics


def einsum(spec, *xs, **kw):
    xp = _xp(xs)
    values = [xp.asarray(primal(x)) for x in xs]
    # RefDLPNO uses pyscf.lib.einsum, which plans multi-operand
    # contractions.  NumPy's default unplanned contraction can add two
    # powers of a local rank (or a global orbital dimension).
    kw.setdefault('optimize', True)
    left, right = spec.replace(' ', '').split('->')
    terms = left.split(',')
    if any(len(set(t)) != len(t) for t in terms) or '.' in spec:
        raise NotImplementedError('Reverse einsum requires explicit, nonrepeated labels')
    value = xp.einsum(spec, *values, **kw)

    def reverse(bar, needed=None):
        result = []
        for i, target in enumerate(terms):
            if needed is not None and not needed[i]:
                result.append(None)
                continue
            labels = [right] + terms[:i] + terms[i+1:]
            arrays = [bar] + values[:i] + values[i+1:]
            present = set(''.join(labels))
            missing = ''.join(k for k in target if k not in present)
            if missing:
                dims = [values[i].shape[target.index(k)] for k in missing]
                labels.append(missing)
                arrays.append(xp.ones(dims))
            result.append(xp.einsum(','.join(labels)+'->'+target, *arrays, optimize=True))
        return result

    return operation(value, xs, reverse, reverse)


def solve(a, b, **kw):
    xp = _xp([a, b])
    av, bv = xp.asarray(primal(a)), xp.asarray(primal(b))
    value = xp.linalg.solve(av, bv)

    def reverse(bar, needed=(True, True)):
        y = xp.linalg.solve(av.T, bar)
        ga = (-xp.outer(y, value) if value.ndim == 1 else -y @ value.T) if needed[0] else None
        return ga, y if needed[1] else None

    return operation(value, [a, b], reverse, reverse)


def eigh(a):
    xp = _xp([a])
    w, u = xp.linalg.eigh(primal(a))
    gap = w[None, :] - w[:, None]
    tolerance = eigenvalue_tolerance(w)
    mask = abs(gap) > tolerance
    inverse = xp.where(mask, 1 / xp.where(mask, gap, 1.), 0.)
    values = operation(w, [a], lambda b: ((u*b) @ u.T,))

    def reverse(bar):
        b = (u.T @ bar) * inverse
        return (u @ ((b+b.T)*.5) @ u.T,)

    vectors = operation(u, [a], reverse)
    vectors._spectrum = w
    vectors._spectrum_tolerance = tolerance
    return values, vectors


def sym_power(a, power):
    xp = _xp([a])
    w, u = xp.linalg.eigh(primal(a))
    if bool(xp.any(w <= 0)):
        raise ValueError('Response requires a positive definite metric')
    f = w**power
    wi, wj = w[:, None], w[None, :]
    gap = wi-wj
    close = abs(gap) <= 1e-10*xp.maximum(wi, wj)
    dd = (f[:, None]-f[None, :]) / xp.where(close, 1., gap)
    dd = xp.where(close, power*((wi+wj)*.5)**(power-1), dd)
    value = (u*f) @ u.T
    return operation(value, [a], lambda b: (u @ (dd*(u.T@((b+b.T)*.5)@u)) @ u.T,))


def diag(x, k=0):
    xp = _xp([x])
    v = xp.diag(primal(x), k=k)
    if x.ndim == 1:
        return operation(v, [x], lambda b: (xp.diag(b, k=k),))

    def reverse(bar):
        result = xp.zeros(x.shape)
        rows = xp.arange(len(bar)) + max(0, -k)
        cols = xp.arange(len(bar)) + max(0, k)
        result[rows, cols] = bar
        return (result,)

    return operation(v, [x], reverse)


def concatenate(xs, axis=0):
    xp = _xp(xs)
    values = [xp.asarray(primal(x)) for x in xs]
    bounds = np.cumsum([0]+[v.shape[axis] for v in values])

    def reverse(bar):
        result = []
        for first, last in zip(bounds[:-1], bounds[1:]):
            key = [slice(None)]*bar.ndim
            key[axis] = slice(first, last)
            result.append(bar[tuple(key)])
        return result

    return operation(xp.concatenate(values, axis=axis), xs, reverse)


def argsort(x, **kw):
    value = primal(x)
    return np.argsort(value.get() if hasattr(value, 'get') else value, **kw)


_FUNCTIONS = dict(sum=lambda x, **kw: x.sum(**kw), diag=diag,
                  outer=lambda a, b: a[:, None]*b[None, :],
                  zeros_like=lambda x, **kw: Variable(_xp([x]).zeros_like(primal(x), **kw), tracked=False),
                  concatenate=concatenate, einsum=einsum, argsort=argsort)


class Numpy:
    def __init__(self, xp=np):
        self.xp = xp
        self.linalg = SimpleNamespace(eigh=eigh, solve=solve, LinAlgError=np.linalg.LinAlgError)

    def zeros(self, shape, dtype=float):
        if np.dtype(dtype).kind in 'biu':
            return np.zeros(shape, dtype=dtype)
        return Variable(self.xp.zeros(shape, dtype=dtype), tracked=False)

    def array(self, xs, dtype=None):
        if isinstance(xs, Variable):
            return xs.copy()
        if isinstance(xs, (list, tuple)) and any(isinstance(x, Variable) for x in xs):
            return concatenate([x.reshape(1, *x.shape) if isinstance(x, Variable)
                                else self.xp.asarray(x).reshape(1, *np.shape(x)) for x in xs])
        return np.array(xs, dtype=dtype)

    def __getattr__(self, name):
        return _FUNCTIONS.get(name, getattr(np, name))


scipy_linalg = SimpleNamespace(solve=solve, cho_factor=lambda a: (a, True),
                               cho_solve=lambda fac, b: solve(fac[0], b))
