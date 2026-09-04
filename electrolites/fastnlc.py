"""
Faster VV10 non-local-correlation build for GPU4PySCF.

Importing this module replaces ``gpu4pyscf.dft.numint._vv10nlc`` -- the
O(n_grid^2) double sum that wB97M-V, wB97X-V and VV10 need -- with the kernel
in ``fastvv10.cu``.  Everything around it (the grid, the density on it, the
1e-10 zero-density threshold, the VV10 parameters, the potential contraction)
is GPU4PySCF's own code, and any case the kernel does not cover falls back to
it.

The double sum is the whole cost of an NLC build: 61.4 s of the 65.1 s
``nr_nlc_vxc`` takes for wB97M-V/def2-TZVPD on the 284-atom PfPMT cluster.
See README_fastnlc.md for what the kernel changes.
"""
import os, functools
import numpy as np
import cupy
from gpu4pyscf.dft import numint as NI
from .compat import NEW_VV10

from ._paths import KERNEL_DIR as HERE

NT = 128                                  # threads per block, = inner tile
NGO = int(os.environ.get('FASTNLC_NGO', 1))   # outer points per thread
# how the pair reciprocal is formed: 0 IEEE fp64 division, 1 MUFU.RCP64H seed
# plus two Newton steps, 2 MUFU.RCP (f32) seed plus two Newton steps
RCP = int(os.environ.get('FASTNLC_RCP', 1))
# one Newton step on the fp32 far-field reciprocal (0 = the bare MUFU.RCP)
NRF = int(os.environ.get('FASTNLC_NRF', 0))
# 'mixed' spatially sorts the points and evaluates pairs whose bounding boxes
# are more than RCUT Bohr apart in single precision; 'plain' is the unsorted
# all-fp64 kernel.  RCUT = 0 makes every pair single precision and RCUT = inf
# every pair double, so the two ends of the choice are measurable.
MODE = os.environ.get('FASTNLC_MODE', 'mixed')
RCUT = float(os.environ.get('FASTNLC_RCUT', 10.0))
# Budget, in the same units as rho*weight, for the grid points left out of the
# pair sum: the points with the smallest |rho*w| are dropped until their total
# reaches BUDGET.  Every pair term is |rho_j w_j|/(g gp gt) with
# g, gp >= K > 0.34 and gt > 0.68, so dropping a set of them changes F by at
# most 1.5*BUDGET/0.08 and E_nlc by at most half of that times the electron
# count -- about 1e-8 Eh at the default, against GPU4PySCF's own 1e-10 Eh of
# run-to-run scatter on these systems.  BUDGET = 0 keeps every point.
BUDGET = float(os.environ.get('FASTNLC_BUDGET', 1e-12))
_DISABLE = bool(os.environ.get('FASTNLC_OFF'))
_STATS = bool(os.environ.get('FASTNLC_STATS'))


@functools.lru_cache(maxsize=16)
def _module(rcp, nrf):
    src = open(os.path.join(HERE, 'fastvv10.cu')).read()
    return cupy.RawModule(code=src, backend='nvrtc',
                          options=('-std=c++17', f'-DRCP={rcp}', f'-DNRF={nrf}'))


@functools.lru_cache(maxsize=64)
def _kernel(ngo, rcp, nrf):
    return _module(rcp, nrf).get_function(f'vv10_f64_{ngo}')


@functools.lru_cache(maxsize=64)
def _kernel_mixed(ngo, rcp, nrf):
    return _module(rcp, nrf).get_function(f'vv10_mixed_{ngo}')



def _spread(v):
    """Every third bit of a 21-bit integer, for a Morton code."""
    i = np.int64
    v = v & i(0x1fffff)
    v = (v | (v << i(32))) & i(0x1f00000000ffff)
    v = (v | (v << i(16))) & i(0x1f0000ff0000ff)
    v = (v | (v << i(8)))  & i(0x100f00f00f00f00f)
    v = (v | (v << i(4)))  & i(0x10c30c30c30c30c3)
    v = (v | (v << i(2)))  & i(0x1249249249249249)
    return v


def _morton_order(coords, origin, h):
    """Z-order the points, so that a tile of consecutive ones is compact.

    The kernel's near/far decision is made from tile bounding boxes, so what
    it can screen is set by how tight those boxes are.  The grid GPU4PySCF
    hands us is ordered by nearest atom and then by radial shell, which puts a
    whole Lebedev sphere in consecutive slots; Morton order over 2^16 boxes
    per axis makes a tile local instead.  The sum is over the same terms.
    """
    idx = [((coords[d] - origin[d]) * (1.0 / h)).astype(np.int64) for d in range(3)]
    code = _spread(idx[0]) | (_spread(idx[1]) << np.int64(1)) \
                           | (_spread(idx[2]) << np.int64(2))
    return cupy.argsort(code)


def _boxes(c, tile):
    """(ntile, 6) float32 bounding boxes of consecutive `tile`-point groups."""
    n = c.shape[1]
    nt = n // tile
    x = c.reshape(3, nt, tile)
    b = cupy.empty((nt, 6), dtype=np.float32)
    b[:, 0:3] = x.min(axis=2).T.astype(np.float32)
    b[:, 3:6] = x.max(axis=2).T.astype(np.float32)
    return b


def _pad(arr, n, fill):
    """arr padded up to length n along its last axis with `fill`."""
    if arr.shape[-1] == n:
        return cupy.ascontiguousarray(arr)
    shape = arr.shape[:-1] + (n,)
    out = cupy.full(shape, fill, dtype=np.float64)
    out[..., :arr.shape[-1]] = arr
    return out


def vv10_FUW(coords, W0, K, vvcoords, W0p, Kp, RpW, ngo=None, rcp=None):
    """The three lattice sums of VV10, for the points passed in.

    coords/vvcoords are (3, n) contiguous.  Returns F (already carrying the
    -1.5 of VV10's kernel), U and W for each outer point.
    """
    ngo = NGO if ngo is None else ngo
    rcp = RCP if rcp is None else rcp
    nout = coords.shape[1]
    nin = vvcoords.shape[1]
    npo = -(-nout // (NT * ngo)) * (NT * ngo)
    npi = -(-nin // NT) * NT

    co = _pad(coords, npo, 0.)
    W0o = _pad(W0, npo, 1.)
    Ko = _pad(K, npo, 1.)
    ci = _pad(vvcoords, npi, 0.)
    W0i = _pad(W0p, npi, 1.)
    Ki = _pad(Kp, npi, 1.)
    RpWi = _pad(RpW, npi, 0.)

    F = cupy.empty(npo)
    U = cupy.empty(npo)
    W = cupy.empty(npo)
    _kernel(ngo, rcp, NRF)((npo // (NT * ngo),), (NT,),
                 (F, U, W, ci, co, W0i, W0o, Ko, Ki, RpWi,
                  np.int32(npi), np.int32(npo)))
    F = F[:nout]; U = U[:nout]; W = W[:nout]
    F *= -1.5
    return F, U, W


def vv10_FUW_mixed(coords, W0, K, vvcoords, W0p, Kp, RpW,
                   ngo=None, rcp=None, rcut=None):
    """The same three sums, spatially sorted, with the distant pairs in fp32."""
    ngo = NGO if ngo is None else ngo
    rcp = RCP if rcp is None else rcp
    rcut = RCUT if rcut is None else rcut
    nout = coords.shape[1]
    nin = vvcoords.shape[1]

    lo = cupy.minimum(coords.min(axis=1), vvcoords.min(axis=1))
    hi = cupy.maximum(coords.max(axis=1), vvcoords.max(axis=1))
    lo = cupy.asnumpy(lo); hi = cupy.asnumpy(hi)
    h = float((hi - lo).max()) / 65535.0 + 1e-30

    one_set = vvcoords is coords and W0p is W0 and Kp is K
    oo = _morton_order(coords, lo, h)
    co = cupy.ascontiguousarray(coords[:, oo])
    W0o = W0[oo]; Ko = K[oo]
    if one_set:
        oi, ci, W0i, Ki = oo, co, W0o, Ko
    else:
        oi = _morton_order(vvcoords, lo, h)
        ci = cupy.ascontiguousarray(vvcoords[:, oi])
        W0i = W0p[oi]; Ki = Kp[oi]
    RpWi = RpW[oi]

    ot = NT * ngo
    npo = -(-nout // ot) * ot
    npi = -(-nin // NT) * NT
    # the padding repeats the last real point (so the bounding boxes stay
    # tight) with RpW = 0 on the inner side (so it contributes nothing)
    shared = one_set and npi == npo
    co = _pad_edge(co, npo); W0o = _pad_edge(W0o, npo); Ko = _pad_edge(Ko, npo)
    if shared:                       # one point set: do not pad it twice
        ci, W0i, Ki = co, W0o, Ko
    else:
        ci = _pad_edge(ci, npi); W0i = _pad_edge(W0i, npi); Ki = _pad_edge(Ki, npi)
    RpWi = _pad(RpWi, npi, 0.)

    tbox = _boxes(ci, NT)
    obox = _boxes(co, ot)

    F = cupy.empty(npo); U = cupy.empty(npo); W = cupy.empty(npo)
    _kernel_mixed(ngo, rcp, NRF)((npo // ot,), (NT,),
                            (F, U, W, ci, co, W0i, W0o, Ko, Ki, RpWi,
                             tbox, obox, np.int32(npi), np.int32(npo),
                             np.float32(rcut * rcut)))
    out = cupy.empty((3, nout))
    out[0, oo] = F[:nout]; out[1, oo] = U[:nout]; out[2, oo] = W[:nout]
    out[0] *= -1.5
    return out[0], out[1], out[2]


def _pad_edge(arr, n):
    """arr padded to length n by repeating its last element."""
    if arr.shape[-1] == n:
        return cupy.ascontiguousarray(arr)
    shape = arr.shape[:-1] + (n,)
    out = cupy.empty(shape, dtype=np.float64)
    out[..., :arr.shape[-1]] = arr
    out[..., arr.shape[-1]:] = arr[..., -1:]
    return out


def _trim(keep, rho0, weight):
    """Drop the grid points that carry the least density-weight.

    A pair term is  rho_j*w_j / (g*gp*gt), and the caller's own 1e-10 density
    threshold already puts  g, gp >= Kvv*(1e-10)^(1/6) = 0.34  and  gt >= 0.68,
    so the total that a dropped set can contribute to any F_i is bounded by
    sum|rho_j w_j| / 0.079.  This picks the largest set whose total is under
    BUDGET, which is the same criterion GPU4PySCF's 1e-10 density cut is
    reaching for -- 14 % of a level-3 grid carries 1e-14 of the weight -- but
    stated as the error it makes rather than as a density.  The same set is
    dropped from the outer sum, where a point's share of E_nlc is
    rho_i*w_i*(Beta + F_i/2) and of the potential matrix is
    w_i*vxc_i*<ao_i|ao_j>, both of which it also bounds.
    """
    rw = cupy.abs(rho0 * weight)
    srt = cupy.sort(rw[keep])
    k = min(int(cupy.searchsorted(cupy.cumsum(srt), BUDGET)), srt.size - 1)
    if k <= 0:
        return keep
    cut = float(srt[k-1])
    out = keep & (rw > cut)
    # the threshold form of the cut can take extra points along if |rho*w| ties
    # at it (in practice only the exact zeros), so the total actually left out
    # is checked rather than assumed
    drop = float(cupy.where(out, 0., rw * keep).sum())
    if _STATS:
        print(f'[fastnlc] kept {int(out.sum())} of {int(keep.sum())} points, '
              f'dropped sum|rho*w| = {drop:.3e} (budget {BUDGET:.0e})')
    if drop > 4 * BUDGET:
        return keep
    return out


def _vv10nlc_two_sets(rho, coords, vvrho, vvweight, vvcoords, nlc_pars):
    """The VV10 double sum over an outer and an inner point set.

    GPU4PySCF 1.7.x's ``numint._vv10nlc`` had exactly this signature, and its
    gradient called it with the two sets genuinely different (a block of the
    outer grid against the whole inner grid).  1.8.0 collapsed the API to one
    grid; the two-set form is kept here because the kernels support it and it
    is what lets one point set be sorted and padded once when the sets
    coincide.
    """
    thr = NI.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD
    exc = cupy.zeros(rho[0, :].size)
    vxc = cupy.zeros([2, rho[0, :].size])

    same = vvrho is rho and vvcoords is coords
    ii = vvrho[0, :] >= thr
    if BUDGET > 0:
        ii = _trim(ii, vvrho[0, :], vvweight)
    ti = ii if same else (rho[0, :] >= thr)

    R = rho[0, :][ti]
    Gx = rho[1, :][ti]; Gy = rho[2, :][ti]; Gz = rho[3, :][ti]
    G = Gx**2 + Gy**2 + Gz**2

    Pi = np.pi
    Pi43 = 4. * Pi / 3.
    Bvv, Cvv = nlc_pars
    Kvv = Bvv * 1.5 * Pi * ((9. * Pi) ** (-1. / 6.))
    Beta = ((3. / (Bvv * Bvv)) ** 0.75) / 32.

    W0tmp = G / (R**2)
    W0tmp = Cvv * W0tmp * W0tmp
    W0 = (W0tmp + Pi43 * R) ** 0.5
    dW0dR = (0.5 * Pi43 * R - 2. * W0tmp) / W0
    dW0dG = W0tmp * R / (G * W0)
    K = Kvv * (R ** (1. / 6.))
    dKdR = (1. / 6.) * K

    if same:
        # the inner set is the outer set, so W0p and Kp are W0 and K; sharing
        # the arrays lets the kernel launcher sort and pad one point set
        RpW = R * vvweight[ii]
        W0p, Kp = W0, K
    else:
        Rp = vvrho[0, :][ii]
        RpW = Rp * vvweight[ii]
        Gxp = vvrho[1, :][ii]; Gyp = vvrho[2, :][ii]; Gzp = vvrho[3, :][ii]
        Gp = Gxp**2 + Gyp**2 + Gzp**2
        W0p = Gp / (Rp * Rp)
        W0p = Cvv * W0p * W0p
        W0p = (W0p + Pi43 * Rp) ** 0.5
        Kp = Kvv * (Rp ** (1. / 6.))

    if not bool(ti.any()):
        return exc, vxc              # no point carries any density
    co = cupy.ascontiguousarray(coords[ti].T)
    if same:
        ci = co
    else:
        ci = cupy.ascontiguousarray(vvcoords[ii].T)
    if MODE == 'mixed':
        F, U, W = vv10_FUW_mixed(co, W0, K, ci, W0p, Kp, RpW)
    else:
        F, U, W = vv10_FUW(co, W0, K, ci, W0p, Kp, RpW)

    exc[ti] = Beta + 0.5 * F
    vxc[0, ti] = Beta + F + 1.5 * (U * dKdR + W * dW0dR)
    vxc[1, ti] = 1.5 * W * dW0dG
    return exc, vxc


def _vv10nlc_one_set(rho_drho, coords, weights, nlc_pars):
    """GPU4PySCF 1.8.x's signature: one grid, summed against itself."""
    return _vv10nlc_two_sets(rho_drho, coords, rho_drho, weights, coords,
                             nlc_pars)


# the entry point that goes on numint, in the shape this release expects
_vv10nlc = _vv10nlc_one_set if NEW_VV10 else _vv10nlc_two_sets

_PATCHED = False


def apply_patch():
    global _PATCHED
    if not _PATCHED and not _DISABLE:
        NI._vv10nlc_orig = NI._vv10nlc
        NI._vv10nlc = _vv10nlc
        _PATCHED = True


apply_patch()
