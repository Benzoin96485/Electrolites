"""First-order tensor algebra for the DLPNO response reference.

``Dual(value, tangent)`` represents ``value + epsilon*tangent``, with
epsilon**2 = 0.  There is no displacement or numerical differentiation.
The deliberately small NumPy interface below fails on unsupported operations
instead of silently discarding a derivative.  It is also usable with CuPy
arrays; integer topology tables remain host arrays.
"""
from types import SimpleNamespace

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


def primal(x):
    return x.v if isinstance(x, Dual) else x


def tangent(x):
    return x.d if isinstance(x, Dual) else 0.


def array_module(x):
    x = primal(x)
    if type(x).__module__.split('.')[0] == 'cupy':
        import cupy
        return cupy
    return np


def common_module(xs):
    for x in xs:
        xp = array_module(x)
        if xp is not np:
            return xp
    return np


def eigenvalue_tolerance(spectrum):
    """Roundoff scale of a symmetric eigensystem, including weak PNO pairs.

    An absolute gap cutoff misclassifies distinct eigenvalues when a pair
    density is small. This bound scales with its spectral norm and size;
    multiplying the matrix by a positive scalar preserves its subspaces.
    """
    xp = array_module(spectrum)
    radius = float(xp.max(xp.abs(spectrum))) if spectrum.size else 0.
    return 32*max(1, spectrum.size)*np.finfo(spectrum.dtype).eps*radius


class Dual(NDArrayOperatorsMixin):
    __array_priority__ = 1000

    def __init__(self, value, derivative=0.):
        xp = array_module(value)
        self.v = xp.asarray(value)
        self.d = xp.broadcast_to(xp.asarray(derivative), self.v.shape).copy()

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

    def __getitem__(self, key):
        result = Dual(self.v[key], self.d[key])
        spectrum = getattr(self, '_spectrum', None)
        if spectrum is not None and isinstance(key, tuple) and len(key) == 2 and result.ndim == 2:
            rows, cols = key
            if isinstance(rows, slice) and rows == slice(None):
                xp = array_module(self)
                selected = xp.arange(self.shape[1])[cols]
                discarded = xp.ones(self.shape[1], dtype=bool)
                discarded[selected] = False
                tolerance = getattr(self, '_spectrum_tolerance', None)
                if tolerance is None:
                    tolerance = eigenvalue_tolerance(spectrum)
                if selected.size and bool(xp.any(discarded)):
                    gap = xp.abs(spectrum[selected, None] - spectrum[None, discarded])
                    if float(gap.min()) <= tolerance:
                        raise ValueError('PNO/TNO selection splits a degenerate eigenvalue block; '
                                         'the retained-subspace derivative is undefined')
                result._spectrum = spectrum[cols]
                result._spectrum_tolerance = tolerance
        return result

    def __setitem__(self, key, x):
        self.v[key] = primal(x)
        self.d[key] = tangent(x)

    def __array__(self, dtype=None, copy=None):
        raise TypeError('Implicit conversion of Dual would discard its derivative')

    def __bool__(self):
        return bool(self.v)

    def __format__(self, spec):
        return format(float(self.v), spec)

    def copy(self):
        return Dual(self.v.copy(), self.d.copy())

    def __iadd__(self, x):
        return self + x

    def __isub__(self, x):
        return self - x

    def __imul__(self, x):
        return self * x

    def __itruediv__(self, x):
        return self / x

    def reshape(self, *shape, **kw):
        return Dual(self.v.reshape(*shape, **kw), self.d.reshape(*shape, **kw))

    def ravel(self, *args, **kw):
        return Dual(self.v.ravel(*args, **kw), self.d.ravel(*args, **kw))

    def transpose(self, *axes):
        return Dual(self.v.transpose(*axes), self.d.transpose(*axes))

    def sum(self, axis=None, **kw):
        return Dual(self.v.sum(axis=axis, **kw), self.d.sum(axis=axis, **kw))

    def max(self, *args, **kw):
        # Used only for convergence and topology decisions.
        return self.v.max(*args, **kw)

    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        if method != '__call__' or kw:
            return NotImplemented
        xp = common_module(inputs)
        vals = [xp.asarray(primal(x)) for x in inputs]
        dots = [xp.broadcast_to(xp.asarray(tangent(x)), v.shape)
                for x, v in zip(inputs, vals)]
        a, da = vals[0], dots[0]
        name = ufunc.__name__
        if len(vals) == 2:
            b, db = vals[1], dots[1]
        if name == 'add':
            return Dual(a + b, da + db)
        if name == 'subtract':
            return Dual(a - b, da - db)
        if name == 'multiply':
            return Dual(a * b, da * b + a * db)
        if name in ('divide', 'true_divide'):
            return Dual(a / b, (da - (a / b) * db) / b)
        if name == 'matmul':
            return Dual(a @ b, da @ b + a @ db)
        if name == 'power':
            if bool(xp.any(db)):
                return Dual(a**b, a**b * (db * xp.log(a) + b * da / a))
            return Dual(a**b, b * a**(b - 1) * da)
        if name == 'negative':
            return Dual(-a, -da)
        if name == 'positive':
            return Dual(a, da)
        if name in ('absolute', 'fabs'):
            return Dual(xp.abs(a), xp.sign(a) * da)
        if name == 'sqrt':
            v = xp.sqrt(a)
            return Dual(v, da / (2 * v))
        if name == 'square':
            return Dual(a * a, 2 * a * da)
        if name in ('less', 'less_equal', 'greater', 'greater_equal',
                    'equal', 'not_equal', 'isfinite'):
            return getattr(xp, name)(*vals)
        return NotImplemented

    def __array_function__(self, func, types, args, kw):
        if func.__name__ in _FUNCTIONS:
            return _FUNCTIONS[func.__name__](*args, **kw)
        return NotImplemented


def einsum(spec, *xs, **kw):
    xp = common_module(xs)
    vs = [xp.asarray(primal(x)) for x in xs]
    v = xp.einsum(spec, *vs, **kw)
    if not any(isinstance(x, Dual) for x in xs):
        return v
    d = xp.zeros_like(v)
    for k, x in enumerate(xs):
        if isinstance(x, Dual):
            terms = vs.copy()
            terms[k] = x.d
            d += xp.einsum(spec, *terms, **kw)
    return Dual(v, d)


def solve(a, b, **kw):
    xp = array_module(a)
    av, bv = primal(a), primal(b)
    x = xp.linalg.solve(av, bv)
    if not isinstance(a, Dual) and not isinstance(b, Dual):
        return x
    ad = tangent(a)
    rhs = tangent(b) - (ad @ x if isinstance(a, Dual) else 0.)
    return Dual(x, xp.linalg.solve(av, rhs))


def eigh(a):
    xp = array_module(a)
    w, u = xp.linalg.eigh(primal(a))
    if not isinstance(a, Dual):
        return w, u
    b = u.T @ a.d @ u
    # Choose the eigenbasis of the perturbation inside a degenerate block.
    # This defines a directionally differentiable canonical frame; a later
    # retained/discarded cut through the block is rejected by __getitem__.
    wh = w.get() if hasattr(w, 'get') else w
    tolerance = eigenvalue_tolerance(w)
    boundaries = np.r_[0, np.flatnonzero(np.diff(wh) > tolerance) + 1, len(w)]
    for first, last in zip(boundaries[:-1], boundaries[1:]):
        if last - first > 1:
            _, rotation = xp.linalg.eigh(b[first:last, first:last])
            u[:, first:last] = u[:, first:last] @ rotation
    b = u.T @ a.d @ u
    gap = w[None, :] - w[:, None]
    # Parallel transport after the above choice inside each block.
    mask = xp.abs(gap) > tolerance
    den = xp.where(mask, gap, 1.)
    rot = xp.where(mask, b / den, 0.)
    vectors = Dual(u, u @ rot)
    vectors._spectrum = w
    vectors._spectrum_tolerance = tolerance
    return Dual(w, xp.diag(b)), vectors


def sym_power(a, power):
    """Spectral matrix power with its continuous Frechet derivative.

    Unlike eigenvector derivatives this remains well defined at repeated
    eigenvalues, including the fitting metric inverse square root.
    """
    xp = array_module(a)
    w, u = xp.linalg.eigh(primal(a))
    if bool(xp.any(w <= 0)):
        raise ValueError('Response requires a positive definite metric')
    f = w**power
    v = (u * f) @ u.T
    if not isinstance(a, Dual):
        return v
    wi, wj = w[:, None], w[None, :]
    gap = wi - wj
    close = xp.abs(gap) <= 1e-10 * xp.maximum(wi, wj)
    frac = (f[:, None] - f[None, :]) / xp.where(close, 1., gap)
    frac = xp.where(close, power * ((wi + wj) * .5)**(power - 1), frac)
    return Dual(v, u @ (frac * (u.T @ a.d @ u)) @ u.T)


def _unary(name, x, *args, **kw):
    xp = array_module(x)
    f = getattr(xp, name)
    return Dual(f(x.v, *args, **kw), f(x.d, *args, **kw)) if isinstance(x, Dual) else f(x, *args, **kw)


def concatenate(xs, axis=0):
    xp = common_module(xs)
    if not any(isinstance(x, Dual) for x in xs):
        return xp.concatenate(xs, axis=axis)
    vs = [xp.asarray(primal(x)) for x in xs]
    ds = [xp.broadcast_to(xp.asarray(tangent(x)), v.shape) for x, v in zip(xs, vs)]
    return Dual(xp.concatenate(vs, axis=axis), xp.concatenate(ds, axis=axis))


def argsort(x, **kw):
    # Sorting determines discrete selections, whose tables stay on the host.
    v = primal(x)
    return np.argsort(v.get() if hasattr(v, 'get') else v, **kw)


_FUNCTIONS = dict(
    sum=lambda x, **kw: x.sum(**kw),
    diag=lambda x, **kw: _unary('diag', x, **kw),
    outer=lambda a, b: a[:, None] * b[None, :],
    zeros_like=lambda x, **kw: Dual(array_module(x).zeros_like(primal(x), **kw)),
    concatenate=concatenate,
    einsum=einsum,
    argsort=argsort,
)


class Numpy:
    """The reference equations' numeric namespace (not a global patch)."""

    def __init__(self, xp=np):
        self.xp = xp
        self.linalg = SimpleNamespace(eigh=eigh, solve=solve,
                                      LinAlgError=np.linalg.LinAlgError)

    def zeros(self, shape, dtype=float):
        if np.dtype(dtype).kind in 'biu':
            return np.zeros(shape, dtype=dtype)
        return Dual(self.xp.zeros(shape, dtype=dtype))

    def array(self, xs, dtype=None):
        if isinstance(xs, Dual):
            return xs.copy()
        if isinstance(xs, (list, tuple)) and any(isinstance(x, Dual) for x in xs):
            vs = self.xp.stack([self.xp.asarray(primal(x), dtype=dtype) for x in xs])
            ds = self.xp.stack([self.xp.asarray(tangent(x), dtype=dtype) for x in xs])
            return Dual(vs, ds)
        return np.array(xs, dtype=dtype)

    def __getattr__(self, name):
        if name in _FUNCTIONS:
            return _FUNCTIONS[name]
        return getattr(np, name)


scipy_linalg = SimpleNamespace(
    solve=solve,
    cho_factor=lambda a: (a, True),
    cho_solve=lambda fac, b: solve(fac[0], b),
)
