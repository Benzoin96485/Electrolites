"""
The VV10 response pair sum on the fp64 tensor cores (catalogue B13).

Importing this module replaces the ``libgdft`` symbol
``VXC_vv10nlc_hess_eval_f_t`` -- GPU4PySCF's ``O(ngrids^2)`` pair sum for the
response of the VV10 potential to a set of perturbed densities -- with the
kernel in ``kernels/fastvv10ft.cu``.  All four callers in 1.8.1 look the
symbol up on ``gpu4pyscf.dft.numint.libgdft`` at call time
(``hessian/rks.py:930`` ``_get_enlc_deriv2``, ``:1188`` its grid-response
branch, ``:2243`` ``_get_vnlc_deriv1`` and ``:3806`` ``nr_rks_fnlc_mo``), and
so does ``fastfnlc``'s replacement of the last, so one attribute reaches a
VV10 polarizability, NMR shielding, C6, and every term of a VV10 Hessian.

What it changes: GPU4PySCF evaluates every pair's four kernel values once per
**six** trial densities (``ceil(ntrial/6)`` times, 21 at a 42-atom Hessian's
``3 natm``) and contracts them on the fp64 SIMT pipe.  Here a pair is
evaluated once per trial group of up to ``8 NTW`` and the contraction is a
``mma.m8n8k4.f64``, with the lane that evaluates a pair holding exactly that
pair's A-fragment elements.  ``docs/README_fastvv10ft.md``.

The arithmetic is the same function to fp64 round-off: the grid sum is
re-associated (the MMA's order) and the one division is a MUFU-seeded
reciprocal with two Newton steps.  ``tests/test_vv10ft.py`` compares the two
element-wise.

``FASTVV10FT_OFF=1`` leaves the symbol alone; ``FASTVV10FT_MT``,
``_NTW``, ``_NW`` choose the tile (m-tiles per warp, trial tiles per warp,
warps per block); by default the trial tile is chosen from ``ntrial``.
"""
import ctypes
import functools
import os

import cupy
import numpy

from gpu4pyscf.dft import numint

from ._paths import KERNEL_DIR as HERE

libgdft = numint.libgdft

_OFF = bool(os.environ.get('FASTVV10FT_OFF'))
_MT = int(os.environ.get('FASTVV10FT_MT', 0))
_NTW = int(os.environ.get('FASTVV10FT_NTW', 0))
_NW = int(os.environ.get('FASTVV10FT_NW', 4))
_JT = int(os.environ.get('FASTVV10FT_JT', 32))
#: box separation (Bohr) beyond which a pair is evaluated in fp32; inf = all
#: fp64
RCUT = float(os.environ.get('FASTVV10FT_RCUT', 10.0))
#: trial densities per launch: the weighted, reordered copy of the trial
#: densities the kernel reads is (2, CHUNK, ngrids)
CHUNK = int(os.environ.get('FASTVV10FT_CHUNK', 128))

_VARIANTS = {(1, 1, 4), (2, 1, 4), (1, 2, 4), (2, 2, 4), (1, 4, 4), (2, 4, 4),
             (1, 8, 4), (2, 8, 4), (1, 4, 8), (2, 4, 8), (1, 8, 8),
             (1, 2, 8)}

STATS = {'ours': 0, 'stock': 0, 'euw': 0}


def stats():
    return dict(STATS)


@functools.lru_cache(maxsize=1)
def _module():
    src = open(os.path.join(HERE, 'fastvv10ft.cu')).read()
    return cupy.RawModule(code=src, backend='nvrtc',
                          options=('-std=c++17', f'-DJT={_JT}'))


@functools.lru_cache(maxsize=32)
def _kernel(name):
    return _module().get_function(name)


def _tile(ntrial):
    """(MT, NTW, NW) for a call with `ntrial` trial densities."""
    if _MT and _NTW:
        return _MT, _NTW, _NW
    # one trial group when it fits in 64; otherwise groups of 32 or 64,
    # whichever wastes fewer padded trials
    if ntrial <= 8:
        ntw = 1
    elif ntrial <= 16:
        ntw = 2
    elif ntrial <= 32:
        ntw = 4
    else:
        ntw = 8
    if ntw == 1:
        return 2, 1, 4
    return 1, ntw, 8


def _arr(p, shape):
    n = int(numpy.prod(shape))
    mem = cupy.cuda.UnownedMemory(p, n * 8, None)
    return cupy.ndarray(shape, dtype=numpy.float64,
                        memptr=cupy.cuda.MemoryPointer(mem, 0))


def _ptr(x):
    return x.value if isinstance(x, ctypes.c_void_p) else int(x)


def _int(x):
    return x.value if hasattr(x, 'value') else int(x)


def _spread(v):
    """Every third bit of a 21-bit integer, for a Morton code (as fastnlc)."""
    i = numpy.int64
    v = v & i(0x1fffff)
    v = (v | (v << i(32))) & i(0x1f00000000ffff)
    v = (v | (v << i(16))) & i(0x1f0000ff0000ff)
    v = (v | (v << i(8))) & i(0x100f00f00f00f00f)
    v = (v | (v << i(4))) & i(0x10c30c30c30c30c3)
    v = (v | (v << i(2))) & i(0x1249249249249249)
    return v


def _morton(c):
    """Z-order of the (3, n) points ``c``, so that consecutive ones are close.

    The near/far split is made from tile bounding boxes, and the order the
    callers hand the points in (atom by atom, a whole Lebedev sphere in
    consecutive slots) gives wide boxes.  The sum is over the same terms.
    """
    lo = c.min(axis=1, keepdims=True)
    h = float((c.max(axis=1, keepdims=True) - lo).max()) / (1 << 20) + 1e-300
    idx = ((c - lo) * (1.0 / h)).astype(numpy.int64)
    code = _spread(idx[0]) | (_spread(idx[1]) << numpy.int64(1)) \
        | (_spread(idx[2]) << numpy.int64(2))
    return cupy.argsort(code)


def _boxes(c, n):
    """(len/n, 6) float32 bounding boxes of consecutive `n`-point groups."""
    x = c.reshape(3, -1, n)
    b = cupy.empty((x.shape[1], 6), dtype=numpy.float32)
    b[:, 0:3] = x.min(axis=2).T
    b[:, 3:6] = x.max(axis=2).T
    return b


def f_t(f_rho_t, f_gamma_t, coords, weight, rho, omega, kappa,
        U, W, A, B, C, domega_drho, domega_dgamma, dkappa_drho,
        d2omega_drho2, d2omega_dgamma2, d2omega_drho_dgamma, d2kappa_drho2,
        rho_t, gamma_t, ntrial, mt=None, ntw=None, nw=None, rcut=None):
    """The response pair sum on cupy arrays; writes f_rho_t and f_gamma_t.

    ``rcut`` (Bohr) is the box separation beyond which a pair is evaluated in
    single precision; ``inf`` keeps every pair in fp64 (and skips the sort).
    """
    ng = rho.shape[0]
    if mt is None:
        mt, ntw, nw = _tile(ntrial)
    rcut = RCUT if rcut is None else rcut
    mix = numpy.isfinite(rcut)
    iext = 8 * mt * nw
    step = iext * _JT // numpy.gcd(iext, _JT)
    npad = -(-ng // step) * step
    c = coords.T
    perm = cupy.full(npad, -1, dtype=numpy.int32)
    if mix:
        order = _morton(c)
        perm[:ng] = order
    else:
        order = slice(None)
        perm[:ng] = cupy.arange(ng, dtype=numpy.int32)
    P = cupy.zeros((9, npad))
    P[3:5, ng:] = 1.
    # translation invariant; centred so that the fp32 copy keeps its digits
    P[0:3, :ng] = (c - c.mean(axis=1, keepdims=True))[:, order]
    P[0:3, ng:] = P[0:3, ng - 1:ng]       # padding inside the last box
    P[3, :ng] = omega[order]
    P[4, :ng] = kappa[order]
    P[5, :ng] = (rho * dkappa_drho)[order]
    P[6, :ng] = (rho * domega_drho)[order]
    P[7, :ng] = (rho * domega_dgamma)[order]
    P[8, :ng] = weight[order]
    if mix:
        Pf = P.astype(numpy.float32)
        wbox = _boxes(Pf[0:3], 8 * mt)
        tbox = _boxes(Pf[0:3], _JT)
        name = 'ft_mixed'
    else:
        Pf = wbox = tbox = P             # unread
        name = 'ft_dmma'
    rc2 = numpy.float32(min(rcut, 1e18) ** 2)
    # the trial densities weighted and in the kernel's order, a chunk of
    # trial groups at a time so that the copy stays a bounded fraction of
    # what the caller already holds
    tg = 8 * ntw
    nchunk_t = max(tg, min(-(-ntrial // tg) * tg, CHUNK // tg * tg))
    X = cupy.empty((2, nchunk_t, npad))
    w = P[8]
    kern = _kernel(f'{name}_{mt}_{ntw}_{nw}')
    for t0 in range(0, ntrial, nchunk_t):
        nt = min(nchunk_t, ntrial - t0)
        ntp = -(-nt // tg) * tg
        Xc = cupy.ndarray((2, ntp, npad), memptr=X.data)
        Xc[:, nt:] = 0.
        Xc[:, :nt, ng:] = 0.
        cupy.multiply(rho_t[t0:t0 + nt][:, order], w[:ng], out=Xc[0, :nt, :ng])
        cupy.multiply(gamma_t[t0:t0 + nt][:, order], w[:ng],
                      out=Xc[1, :nt, :ng])
        kern((npad // iext, ntp // tg), (32 * nw,),
             (f_rho_t, f_gamma_t, P, Pf, perm, Xc, numpy.uint64(ntp * npad),
              wbox, tbox, rc2, numpy.int32(npad), numpy.int32(ng),
              numpy.int32(t0), numpy.int32(ntrial)))
    del X
    _kernel('ft_diagonal')(
        (-(-ng // 128),), (128,),
        (f_rho_t, f_gamma_t, rho, U, W, A, B, C, domega_drho, domega_dgamma,
         dkappa_drho, d2omega_drho2, d2omega_dgamma2, d2omega_drho_dgamma,
         d2kappa_drho2, rho_t, gamma_t, numpy.int32(ng), numpy.int32(ntrial)))


def _entry(stream, f_rho_t, f_gamma_t, grid_coord, grid_weight, rho, omega,
           kappa, U, W, A, B, C, domega_drho, domega_dgamma, dkappa_drho,
           d2omega_drho2, d2omega_dgamma2, d2omega_drho_dgamma, d2kappa_drho2,
           rho_t, gamma_t, ngrids, ntrial):
    """Drop-in for the ``libgdft`` entry point: same arguments, same result.

    The callers pass raw device pointers (``ctypes.c_void_p``) and
    ``ctypes.c_int`` sizes; they are wrapped as unowned cupy arrays.  The
    callers launch on cupy's current stream and pass its handle, so the
    kernels go on the stream named.
    """
    ng, nt = _int(ngrids), _int(ntrial)
    if ng == 0 or nt == 0:
        STATS['ours'] += 1
        return 0
    v = lambda p: _arr(_ptr(p), (ng,))
    tv = lambda p: _arr(_ptr(p), (nt, ng))
    sp = _ptr(stream) or 0
    cur = cupy.cuda.get_current_stream()
    ctx = cur if sp == cur.ptr else cupy.cuda.ExternalStream(sp)
    with ctx:
        f_t(tv(f_rho_t), tv(f_gamma_t), _arr(_ptr(grid_coord), (ng, 3)),
            v(grid_weight), v(rho), v(omega), v(kappa), v(U), v(W), v(A),
            v(B), v(C), v(domega_drho), v(domega_dgamma), v(dkappa_drho),
            v(d2omega_drho2), v(d2omega_dgamma2), v(d2omega_drho_dgamma),
            v(d2kappa_drho2), tv(rho_t), tv(gamma_t), nt)
    STATS['ours'] += 1
    return 0


def euw_w1(Ew, Uw, Ww, coords, weight1, rho, omega, kappa, nder,
           mt=None, ntw=None, nw=None, rcut=None):
    """`VXC_vv10nlc_hess_eval_EUW_with_weight1` on cupy arrays.

    weight1 is (nder, ng); writes Ew, Uw, Ww, each (nder, ng).
    """
    ng = rho.shape[0]
    if mt is None:
        mt, ntw, nw = _tile(nder)
    rcut = RCUT if rcut is None else rcut
    mix = numpy.isfinite(rcut)
    iext = 8 * mt * nw
    step = iext * _JT // numpy.gcd(iext, _JT)
    npad = -(-ng // step) * step
    c = coords.T
    perm = cupy.full(npad, -1, dtype=numpy.int32)
    if mix:
        order = _morton(c)
        perm[:ng] = order
    else:
        order = slice(None)
        perm[:ng] = cupy.arange(ng, dtype=numpy.int32)
    Q = cupy.zeros((6, npad))
    Q[3:5, ng:] = 1.
    Q[0:3, :ng] = (c - c.mean(axis=1, keepdims=True))[:, order]
    Q[0:3, ng:] = Q[0:3, ng - 1:ng]
    Q[3, :ng] = omega[order]
    Q[4, :ng] = kappa[order]
    Q[5, :ng] = rho[order]
    if mix:
        Qf = Q.astype(numpy.float32)
        wbox = _boxes(Qf[0:3], 8 * mt)
        tbox = _boxes(Qf[0:3], _JT)
        name = 'euw_mixed'
    else:
        Qf = wbox = tbox = Q
        name = 'euw_dmma'
    rc2 = numpy.float32(min(rcut, 1e18) ** 2)
    tg = 8 * ntw
    nchunk_t = max(tg, min(-(-nder // tg) * tg, 2 * CHUNK // tg * tg))
    X = cupy.empty((nchunk_t, npad))
    kern = _kernel(f'{name}_{mt}_{ntw}_{nw}')
    for t0 in range(0, nder, nchunk_t):
        nt = min(nchunk_t, nder - t0)
        ntp = -(-nt // tg) * tg
        Xc = cupy.ndarray((ntp, npad), memptr=X.data)
        Xc[nt:] = 0.
        Xc[:nt, ng:] = 0.
        Xc[:nt, :ng] = weight1[t0:t0 + nt][:, order]
        kern((npad // iext, ntp // tg), (32 * nw,),
             (Ew, Uw, Ww, Q, Qf, perm, Xc, wbox, tbox, rc2,
              numpy.int32(npad), numpy.int32(ng), numpy.int32(t0),
              numpy.int32(nder)))
    del X


def _entry_euw(stream, Ew, Uw, Ww, grid_coord, grid_weight1, rho, omega,
               kappa, ngrids, nderivative):
    """Drop-in for ``VXC_vv10nlc_hess_eval_EUW_with_weight1``."""
    ng, nd = _int(ngrids), _int(nderivative)
    if ng == 0 or nd == 0:
        STATS['euw'] += 1
        return 0
    v = lambda p: _arr(_ptr(p), (ng,))
    tv = lambda p: _arr(_ptr(p), (nd, ng))
    sp = _ptr(stream) or 0
    cur = cupy.cuda.get_current_stream()
    ctx = cur if sp == cur.ptr else cupy.cuda.ExternalStream(sp)
    with ctx:
        euw_w1(tv(Ew), tv(Uw), tv(Ww), _arr(_ptr(grid_coord), (ng, 3)),
               tv(grid_weight1), v(rho), v(omega), v(kappa), nd)
    STATS['euw'] += 1
    return 0


_PATCHED = False
_stock = None
_stock_euw = None


def stock():
    """GPU4PySCF's own entry point, for the A/B and the tests."""
    return _stock


def stock_euw():
    """GPU4PySCF's own ``EUW_with_weight1``, for the A/B and the tests."""
    return _stock_euw


def apply_patch():
    global _PATCHED, _stock, _stock_euw
    if _PATCHED or _OFF:
        return
    _stock = libgdft.VXC_vv10nlc_hess_eval_f_t
    _stock_euw = libgdft.VXC_vv10nlc_hess_eval_EUW_with_weight1
    # a ctypes CDLL caches looked-up symbols as instance attributes, so
    # setting the attribute is what every `libgdft.VXC_...` call site reaches
    libgdft.VXC_vv10nlc_hess_eval_f_t = _entry
    libgdft.VXC_vv10nlc_hess_eval_EUW_with_weight1 = _entry_euw
    _PATCHED = True


apply_patch()
