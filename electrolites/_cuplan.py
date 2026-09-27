"""Ask cuTENSOR whether it will plan a contraction, before there is one to run.

`docs/DF_WORK_LIST.md` **W21**.  Three modules here hit
``CUTENSOR_STATUS_NOT_SUPPORTED`` with tens of GB free -- `fastdfhess` on the
DF Hessian's CPHF build, `fastdfgrad` on the gradient's ip1 loop,
`fastdfhessip` on the Gram form of the per-atom occupied block -- and each of
them guarded it with a measured constant, a retry, or both.
`docs/README_fastdfplan.md` is what cuTENSOR 2.6.0 actually refuses, measured
without allocating anything, and the answer is one sentence:

    **An output tensor of more than 2**29 elements is refused unless one
    particular mode of it can be cut, by factors no larger than 13, into
    pieces of at most 2**29 elements each.**

The limit counts *elements*, not bytes -- it is the same 536 870 912 for
float32, float64 and complex128 -- and it applies to the output only: an input
four times that size plans.  Which mode may be cut is a property of the
pattern (for every pattern this repository runs, the last output mode of the
first operand), so an extent of 914 = 2 x 457 is refused where 1096 = 8 x 137
is accepted, which is the "not monotonic" boundary W10 measured.

So a refusal is predictable, and predicting it is cheap: a contraction's plan
depends on shapes and strides and not on data or free memory, so it can be
asked for on a *descriptor* whose data pointer is a one-kilobyte buffer.  That
is what this module does.  A query is ~0.3 ms and cached, which is small
enough to put in front of every block loop that sizes its blocks from free
memory, and it is exact on whatever cuTENSOR and card the process has -- the
constant in the sentence above is 2.6.0's on an A100 and nothing here depends
on it.

This module patches nothing.
"""
import functools
import os
import sys

import numpy as np
import cupy as cp

try:                                         # cuTENSOR is optional upstream
    from cupyx import cutensor as _ct
    from cupy_backends.cuda.libs import cutensor as _cb
    from cupy_backends.cuda.libs.cutensor import CuTensorError
    from gpu4pyscf.lib import cutensor as _G
    HAVE_CUTENSOR = _G.cutensor is not None
except ImportError:                                           # pragma: no cover
    HAVE_CUTENSOR = False
    CuTensorError = RuntimeError

#: what cuTENSOR 2.6.0 plans without cutting a mode, on sm_80.  Documented
#: rather than used: every decision here is a query, so a different cuTENSOR
#: moves the answer without moving this number.
OUTPUT_ELEMENT_LIMIT = 1 << 29

_BASE = {}
STATS = {'queries': 0, 'refused': 0, 'cache_hits': 0}


def _base():
    """A small device buffer every probe descriptor points into, per device."""
    dev = cp.cuda.Device().id
    buf = _BASE.get(dev)
    if buf is None:
        buf = _BASE[dev] = cp.empty(128)
    return buf


def _strides(shape, itemsize):
    out, s = [], itemsize
    for n in reversed(shape):
        out.insert(0, s)
        s *= n
    return tuple(out)


def _view(shape, strides, dtype):
    """A CuPy array with this shape and these strides over ``_base()``.

    Never read or written: cuTENSOR takes the pointer only to derive an
    alignment, and the plan does not depend on it (measured: offsets of 0,
    8, 16, 64 and 128 bytes give the same accepted set).
    """
    return cp.ndarray(shape, dtype, _base().data, strides)


@functools.lru_cache(maxsize=65536)
def _query(pattern, shapes, strides, dtype, dev):
    STATS['queries'] += 1
    pattern = pattern.replace(' ', '')
    sa, rest = pattern.split(',')
    sb, sc = rest.split('->')
    arrs = [_view(sh, st, dtype) for sh, st in zip(shapes, strides)]
    descs = [_G._create_tensor_descriptor(x) for x in arrs]
    modes = [_ct.create_mode(*s) for s in (sa, sb, sc)]
    op = _ct.create_contraction(descs[0], modes[0], _G.OP_IDENTITY,
                                descs[1], modes[1], _G.OP_IDENTITY,
                                descs[2], modes[2], _G.OP_IDENTITY, 0)
    pref = _ct.create_plan_preference(algo=_G.ALGO_DEFAULT,
                                      jit_mode=_G.JIT_MODE_NONE)
    # exactly the two calls `gpu4pyscf/lib/cutensor.py:contraction` makes,
    # with its workspace preference.  `estimateWorkspaceSize` reports a
    # refusal through an exception Cython prints and swallows, returning 0;
    # `create_plan` is where it surfaces
    # -- and writes the traceback straight to file descriptor 2 for every
    # refusal, which for a query that expects refusals is noise; the
    # descriptor is pointed at /dev/null for this call only
    sys.stderr.flush()
    saved = os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 2)
    try:
        ws = _cb.estimateWorkspaceSize(_ct._get_handle().ptr, op.ptr,
                                       pref.ptr, _G.WORKSPACE_RECOMMENDED)
        _ct.create_plan(op, pref, ws_limit=ws)
    except CuTensorError:
        STATS['refused'] += 1
        return False
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(null)
    return True


def planable(pattern, *operands, dtype=np.float64):
    """Will ``contract(pattern, a, b, out=c)`` get a plan?

    Each operand is a CuPy array, or a shape (C order is assumed), or a
    ``(shape, strides)`` pair with strides in bytes -- so a block loop can ask
    about a block before it allocates one.  Three operands, the output last.
    Without cuTENSOR this is always True: the contraction then goes through
    ``cupy.einsum``, which has no such limit.
    """
    if not HAVE_CUTENSOR:
        return True
    dtype = np.dtype(dtype)
    shapes, strides = [], []
    for x in operands:
        if isinstance(x, cp.ndarray):
            sh, st = x.shape, x.strides
            dtype = x.dtype
        elif (len(x) == 2 and isinstance(x[0], (tuple, list))):
            sh, st = tuple(x[0]), tuple(x[1])
        else:
            sh = tuple(int(n) for n in x)
            st = _strides(sh, dtype.itemsize)
        if any(n == 0 for n in sh):
            return True                  # upstream sends these to einsum
        shapes.append(tuple(int(n) for n in sh))
        strides.append(tuple(int(s) for s in st))
    info = _query.cache_info()
    ok = _query(pattern, tuple(shapes), tuple(strides), dtype.str,
                cp.cuda.Device().id)
    if _query.cache_info().hits > info.hits:
        STATS['cache_hits'] += 1
    return ok


def largest(n, accepts, lo=1, walk=64):
    """The largest ``m`` in ``[lo, n]`` with ``accepts(m)``.

    Refused extents are sparse near an accepted one when the refusal comes
    from the cuttable mode's factors (the gradient's ip1 loop: 914 is refused,
    913 is not), so this walks down first.  When it comes from a mode that
    cannot be cut at all the refusal is a threshold (the Hessian's density
    batch: every batch past 22 at nao 1007), and ``walk`` steps in the walk
    gives way to a bisection that assumes it is one.  Either way the answer
    is an accepted extent -- the bisection only ever returns one it asked
    about -- and ``lo`` is assumed to be accepted.
    """
    m = int(n)
    for _ in range(walk):
        if m <= lo or accepts(m):
            return max(m, lo)
        m -= 1
    good, bad = lo, m + 1
    while bad - good > 1:
        mid = (good + bad) // 2
        if accepts(mid):
            good = mid
        else:
            bad = mid
    return good


def prange(start, stop, step, accepts):
    """``pyscf.lib.prange`` with every block's extent one cuTENSOR plans.

    ``accepts(extent)`` says whether a block of that extent plans.  Each block
    is the largest accepted extent no larger than ``step``; the remainder is
    the block a fixed ``step`` gets wrong -- W10's refused block of "1096"
    was the 914 left over at the end of an auxiliary batch -- and when it is
    refused it is cut in two near the middle rather than into an accepted
    block and a sliver (914 -> 458 + 456, not 913 + 1).
    """
    k0 = start
    while k0 < stop:
        rem = stop - k0
        if rem <= step and not accepts(rem) and rem > 1:
            for d in range(rem // 2):
                a = rem // 2 - d
                if accepts(a) and accepts(rem - a):
                    yield k0, k0 + rem - a
                    yield k0 + rem - a, stop
                    return
        k1 = k0 + largest(min(step, rem), accepts)
        yield k0, k1
        k0 = k1


def _free_modes(sa, sb, sc):
    """Output modes that appear in exactly one operand, with that operand.

    Cutting one of them cuts the contraction into independent pieces: each
    piece reads a slice of one operand, all of the other, and writes a slice
    of the output.
    """
    out = []
    for m in sc:
        if (m in sa) != (m in sb):
            out.append((m, 0 if m in sa else 1))
    return out


def _cut(x, axis, k0, k1):
    return x[(slice(None),) * axis + (slice(k0, k1),)]


@functools.lru_cache(maxsize=4096)
def _parse(pattern):
    p = pattern.replace(' ', '')
    sa, rest = p.split(',')
    sb, sc = rest.split('->')
    return p, sa, sb, sc


def contract_with(call, pattern, a, b, alpha=1.0, beta=0.0, out=None,
                  **kw):
    """``call(pattern, a, b, alpha, beta, out=out)``, cut if cuTENSOR refuses.

    `docs/DF_WORK_LIST.md` **W27**.  ``call`` is
    ``gpu4pyscf.lib.cutensor.contraction`` or anything with its signature.
    An output of at most 2**29 elements is never refused, so below that this
    is the plain call and costs one product of the output's extents.  Above
    it cuTENSOR is asked first (`planable`, cached); a refused contraction is
    cut along an output mode that belongs to one operand only -- the largest
    such mode, into the largest pieces that plan (`prange`) -- and each piece
    is the same call on views.  The arithmetic of every element is unchanged:
    a piece is the same contraction over the same summed indices, so the
    result is the same to the last bit whenever cuTENSOR picks the same
    kernel for the pieces.  If no mode can be cut into pieces that plan, the
    call goes to ``cupy.einsum``, which has no such limit.
    """
    p, sa, sb, sc = _parse(pattern)
    if out is not None:
        n = out.size
    else:
        ext = dict(zip(sa + sb, a.shape + b.shape))
        n = 1
        for m in sc:
            n *= ext[m]
    if n <= OUTPUT_ELEMENT_LIMIT or a.size == 0 or b.size == 0:
        return call(pattern, a, b, alpha, beta, out=out, **kw)
    ext = dict(zip(sa + sb, a.shape + b.shape))
    shape = tuple(out.shape) if out is not None else tuple(ext[m] for m in sc)
    if out is None:
        out = cp.empty(shape, dtype=np.result_type(a.dtype, b.dtype))
        beta = 0.0
    if planable(p, a, b, out):
        return call(pattern, a, b, alpha, beta, out=out, **kw)
    STATS['cut'] = STATS.get('cut', 0) + 1
    for m, side in sorted(_free_modes(sa, sb, sc), key=lambda t: -ext[t[0]]):
        ax_in = (sa if side == 0 else sb).index(m)
        ax_out = sc.index(m)
        src = a if side == 0 else b

        def pieces_plan(k, ax_in=ax_in, ax_out=ax_out, src=src, side=side):
            x = _cut(src, ax_in, 0, k)
            ops = (x, b) if side == 0 else (a, x)
            return planable(p, *ops, _cut(out, ax_out, 0, k))
        if not pieces_plan(1):
            continue
        for k0, k1 in prange(0, ext[m], ext[m], pieces_plan):
            x = _cut(src, ax_in, k0, k1)
            ops = (x, b) if side == 0 else (a, x)
            call(pattern, *ops, alpha, beta, out=_cut(out, ax_out, k0, k1),
                 **kw)
        return out
    STATS['einsum'] = STATS.get('einsum', 0) + 1
    from gpu4pyscf.lib import cutensor as _g
    return _g._contract_einsum(p, a, b, alpha, beta, out)


def contract(pattern, a, b, alpha=1.0, beta=0.0, out=None):
    """``cupy_helper.contract`` through `contract_with`."""
    if not HAVE_CUTENSOR:
        from gpu4pyscf.lib.cupy_helper import contract as _c
        return _c(pattern, a, b, alpha, beta, out=out)
    from gpu4pyscf.lib import cutensor as _g
    # the unguarded call when `planguard` has installed itself over it
    call = getattr(_g, '_contraction_unguarded', _g.contraction)
    return contract_with(call, pattern, a, b, alpha, beta, out=out)


def stats():
    return dict(STATS)


__all__ = ['planable', 'largest', 'prange', 'contract', 'contract_with',
           'stats',
           'CuTensorError', 'OUTPUT_ELEMENT_LIMIT', 'HAVE_CUTENSOR']
