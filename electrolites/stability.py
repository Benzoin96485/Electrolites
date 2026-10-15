"""SCF stability analysis on the GPU (`docs/FEATURE_TARGETS_PYSCF_2.13.md` S2).

GPU4PySCF 1.8.1 has ``stability = NotImplemented`` on RHF, UHF and ROHF
(``scf/hf.py:865``, ``uhf.py:324``, ``rohf.py:100``), and PySCF's driver
cannot take a GPU object, so a user's only route was ``mf.to_cpu()
.stability()`` -- the whole analysis, every orbital-Hessian product, on the
host: 53-126 times the GPU SCF it checks (FEATURE_TARGETS §11.4).

This is PySCF 2.13's ``pyscf/scf/stability.py`` term by term -- the same
orbital-Hessian products, the same response functions (``gen_response`` with
the same ``singlet`` / ``hermi`` / ``with_j`` / ``with_nlc``), the same
scaling, initial guess, preconditioner, eigenvalue threshold (-1e-5) and the
same rotated orbitals returned -- with two differences in *how*, none in
*what*:

* the products are **batched**: a block of trial vectors becomes one stack of
  transition densities and one ``gen_response`` call, so the J/K and XC
  response kernels (Electrolites' ``fastjk``/``fastk``/``fastxcfxc`` under
  ``patch_all()``) see many densities at once instead of one per call;
* the Davidson is a block Davidson on device vectors (``_davidson``),
  started from the PySCF guess plus the lowest diagonal elements, converged
  to a residual of sqrt(tol)/10 (PySCF's ``lib.davidson`` stops at sqrt(tol)),
  and for the UKS UHF -> GHF operator -- which is not symmetric -- solved as
  a non-symmetric eigenproblem (see ``_davidson``).

Covered: RHF/RKS internal and external (real -> complex, RHF -> UHF),
UHF/UKS internal and external (real -> complex, UHF -> GHF), ROHF/ROKS
internal (PySCF has no ROHF external).  Point-group symmetry restrictions
(``with_symmetry`` on a symmetric molecule) are not applied, which only
widens the search; GPU4PySCF drops symmetry anyway (FEATURE_TARGETS D11).

    import electrolites
    electrolites.enable('stability')
    mo_i, mo_e, stable_i, stable_e = mf.stability(external=True,
                                                  return_status=True)
"""
import numpy
import cupy as cp

from pyscf.scf import stability as stab_cpu

from gpu4pyscf.lib import logger
from gpu4pyscf.scf import hf, uhf, rohf

from electrolites.features import attach, register_gpu_module

STAB_NROOTS = stab_cpu.STAB_NROOTS
STAB_TOL = stab_cpu.STAB_TOL
# PySCF: `stable = not (e < -1e-5)`
_THRESHOLD = -1e-5
#: the lowest eigenvalues of the last analysis of each kind (diagnostics;
#: what the tests compare with PySCF's)
LAST = {}


def _davidson(hop, hdiag, x0_main, nroots, tol, log, max_cycle=200,
              hermitian=True, max_space=64, nguess=None):
    """Lowest ``nroots`` eigenpairs of ``hop`` (a batched product: rows in,
    rows out) by a block Davidson on the device.

    ``x0_main`` is PySCF's guess; the other starting vectors are unit vectors
    on the smallest ``hdiag`` elements.  The subspace matrix is diagonalised
    on the host; with ``hermitian=False`` by a general eigensolver, taking
    the eigenvalues with the lowest real part -- needed for the UKS
    UHF -> GHF operator, which PySCF defines with a collinear XC kernel and
    which is **not symmetric** (|H - H^T| = 0.023 on H2O+ / B3LYP): a
    symmetric solver's answer there depends on its tolerance (PySCF moves
    from -0.0714 at tol 1e-4 to -0.0727 at 1e-8; the exact lowest real
    eigenvalue is -0.07269).  Converged to a residual of ``sqrt(tol)/10``.
    """
    size = hdiag.size
    nroots = min(nroots, size)
    if nguess is None:
        nguess = nroots + 1
    guesses = [x0_main / max(float(cp.linalg.norm(x0_main)), 1e-300)]
    for i in cp.argsort(hdiag)[:nguess].get():
        if len(guesses) >= nguess:
            break
        u = cp.zeros(size)
        u[i] = 1.
        guesses.append(u)
    V = _orth(cp.stack(guesses), None)
    AV = hop(V)
    tol_r = tol ** .5 * .1
    w = None
    for cycle in range(max_cycle):
        hs = cp.asnumpy(V @ AV.T)
        if hermitian:
            w_all, u_all = numpy.linalg.eigh((hs + hs.T) * .5)
        else:
            w_all, u_all = numpy.linalg.eig(hs)
            order = numpy.argsort(w_all.real)
            w_all, u_all = w_all[order].real, u_all[:, order].real
        w = w_all[:nroots]
        U = cp.asarray(u_all[:, :nroots])
        X = U.T @ V
        AX = U.T @ AV
        R = AX - cp.asarray(w)[:, None] * X
        rnorm = cp.linalg.norm(R, axis=1).get()
        log.debug('stability Davidson %d  dim %d  e %s  |r| %s',
                  cycle, len(V), w, rnorm)
        if numpy.all(rnorm < tol_r):
            break
        hd = hdiag[None, :] - cp.asarray(w)[:, None]
        hd[abs(hd) < 1e-8] = 1e-8
        new = (R / hd)[rnorm >= tol_r]
        if len(V) + len(new) > max_space:         # collapse to the Ritz vectors
            V, AV = _orth_pair(X, AX)
        new = _orth(new, V)
        if new is None:
            log.warn('stability Davidson: subspace exhausted, |r| %s', rnorm)
            break
        V = cp.vstack([V, new])
        AV = cp.vstack([AV, hop(new)])
    else:
        log.warn('stability Davidson not converged, |r| %s', rnorm)
    return numpy.asarray(w), X


def _orth(vecs, basis, lindep=1e-10):
    """Orthonormalise rows of ``vecs`` against ``basis`` (rows) and each
    other; None if nothing is left."""
    out = []
    for v in vecs:
        for _ in range(2):
            if basis is not None and len(basis):
                v = v - (basis @ v) @ basis
            for u in out:
                v = v - (u @ v) * u
        nv = float(cp.linalg.norm(v))
        if nv > lindep:
            out.append(v / nv)
    if not out:
        return None
    return cp.stack(out)


def _orth_pair(X, AX):
    """Orthonormalise the Ritz vectors X and carry A X with them."""
    q, r = cp.linalg.qr(X.T)
    rinv = cp.linalg.inv(r)
    return q.T, (AX.T @ rinv).T


def _occ_vir(mo_occ):
    mo_occ = cp.asarray(mo_occ)
    return cp.nonzero(mo_occ > 0)[0], cp.nonzero(mo_occ == 0)[0]


def _fock_mo(mf):
    dm0 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    return mf.get_hcore() + mf.get_veff(mf.mol, dm0)


def _rotate(mo_coeff, mo_occ, v):
    return cp.asarray(stab_cpu._rotate_mo(cp.asnumpy(mo_coeff),
                                          cp.asnumpy(mo_occ), cp.asnumpy(v)))


# ------------------------------------------------------------------ RHF


def _rhf_blocks(mf):
    mo, occ = cp.asarray(mf.mo_coeff), cp.asarray(mf.mo_occ)
    oi = cp.nonzero(occ == 2)[0]
    vi = cp.nonzero(occ == 0)[0]
    orbo, orbv = mo[:, oi], mo[:, vi]
    fock = mo.T.dot(_fock_mo(mf)).dot(mo)
    foo = fock[oi[:, None], oi]
    fvv = fock[vi[:, None], vi]
    return mo, occ, orbo, orbv, foo, fvv, fock[vi[:, None], oi]


def _rhf_product(vresp, orbo, orbv, foo, fvv, sign, scale):
    """x -> scale * [fvv x - x foo + orbv^T v1(dm1) orbo], batched, with
    dm1 = d + sign d^T, d = orbv (2x) orbo^T (PySCF's RHF products)."""
    nocc, nvir = orbo.shape[1], orbv.shape[1]

    def hop(xs):
        xs = cp.asarray(xs).reshape(-1, nvir, nocc)
        x2 = cp.matmul(fvv, xs) - cp.matmul(xs, foo)
        d = cp.matmul(cp.matmul(orbv, xs * 2), orbo.T)
        dm1 = d + sign * d.transpose(0, 2, 1)
        v1 = vresp(dm1)
        x2 += cp.matmul(cp.matmul(orbv.T, v1), orbo)
        return (x2 * scale).reshape(len(xs), -1)
    return hop


def rhf_internal(mf, verbose=None, return_status=False,
                 nroots=STAB_NROOTS, tol=STAB_TOL):
    log = logger.new_logger(mf, verbose)
    mo, occ, orbo, orbv, foo, fvv, fvo = _rhf_blocks(mf)
    vresp = mf.gen_response(mo, occ, singlet=None, hermi=1, with_nlc=False)
    # newton_ah's h_op is 2 x the product, stability doubles it again
    hop = _rhf_product(vresp, orbo, orbv, foo, fvv, +1, 4.)
    g = (fvo * 2).ravel()
    hdiag = ((fvv.diagonal()[:, None] - foo.diagonal()) * 4).ravel()
    x0 = cp.zeros_like(g)
    x0[g != 0] = 1. / hdiag[g != 0]
    e, v = _davidson(hop, hdiag, x0, nroots, tol, log)
    log.info('rhf_internal: lowest eigs of H = %s', e)
    LAST['rhf_internal'] = e
    stable = not (e[0] < _THRESHOLD)
    stab_cpu.dump_status(log, stable, f'{mf.__class__}', 'internal')
    out = mf.mo_coeff if stable else _rotate(mo, occ, v[0])
    return (out, stable) if return_status else out


def rhf_external(mf, verbose=None, return_status=False,
                 nroots=STAB_NROOTS, tol=STAB_TOL):
    log = logger.new_logger(mf, verbose)
    mo, occ, orbo, orbv, foo, fvv, _ = _rhf_blocks(mf)
    hdiag = (fvv.diagonal()[:, None] - foo.diagonal()).ravel()
    x0 = cp.zeros_like(hdiag)
    x0[hdiag > 1e-5] = 1. / hdiag[hdiag > 1e-5]

    # real -> complex: antisymmetric density, no J and no fxc
    vz = mf.gen_response(singlet=None, hermi=2)
    hop1 = _rhf_product(vz, orbo, orbv, foo, fvv, -1, 1.)
    e1, _ = _davidson(hop1, hdiag, x0, nroots, tol, log)
    log.info('rhf_real2complex: lowest eigs of H = %s', e1)
    LAST['rhf_real2complex'] = e1
    stab_cpu.dump_status(log, not (e1[0] < _THRESHOLD), f'{mf.__class__}',
                         'real -> complex')

    # RHF -> UHF: the triplet response
    v3 = mf.gen_response(singlet=False, hermi=1)
    hop3 = _rhf_product(v3, orbo, orbv, foo, fvv, +1, 1.)
    e3, v = _davidson(hop3, hdiag, x0, nroots, tol, log)
    log.info('rhf_external: lowest eigs of H = %s', e3)
    LAST['rhf_external'] = e3
    stable = not (e3[0] < _THRESHOLD)
    stab_cpu.dump_status(log, stable, f'{mf.__class__}', 'RHF/RKS -> UHF/UKS')
    if stable:
        out = (mf.mo_coeff, mf.mo_coeff)
    else:
        out = (_rotate(mo, occ, v[0]), mf.mo_coeff)
    return (out, stable) if return_status else out


# ------------------------------------------------------------------ UHF


def _uhf_blocks(mf):
    mo = [cp.asarray(c) for c in mf.mo_coeff]
    occ = [cp.asarray(o) for o in mf.mo_occ]
    fock_ao = _fock_mo(mf)
    out = []
    for s in (0, 1):
        oi, vi = _occ_vir(occ[s])
        f = mo[s].T.dot(fock_ao[s]).dot(mo[s])
        out.append(dict(orbo=mo[s][:, oi], orbv=mo[s][:, vi],
                        foo=f[oi[:, None], oi], fvv=f[vi[:, None], vi],
                        fvo=f[vi[:, None], oi]))
    return mo, occ, out


def uhf_internal(mf, verbose=None, return_status=False,
                 nroots=STAB_NROOTS, tol=STAB_TOL):
    log = logger.new_logger(mf, verbose)
    mo, occ, (a, b) = _uhf_blocks(mf)
    vresp = mf.gen_response(mf.mo_coeff, mf.mo_occ, hermi=1, with_nlc=False)
    na, va = a['orbo'].shape[1], a['orbv'].shape[1]
    nb, vb = b['orbo'].shape[1], b['orbv'].shape[1]
    sa = na * va

    def hop(xs):
        xs = cp.asarray(xs)
        n = len(xs)
        xa = xs[:, :sa].reshape(n, va, na)
        xb = xs[:, sa:].reshape(n, vb, nb)
        x2a = cp.matmul(a['fvv'], xa) - cp.matmul(xa, a['foo'])
        x2b = cp.matmul(b['fvv'], xb) - cp.matmul(xb, b['foo'])
        da = cp.matmul(cp.matmul(a['orbv'], xa), a['orbo'].T)
        db = cp.matmul(cp.matmul(b['orbv'], xb), b['orbo'].T)
        dm1 = cp.stack([da + da.transpose(0, 2, 1), db + db.transpose(0, 2, 1)])
        v1 = vresp(dm1)
        x2a += cp.matmul(cp.matmul(a['orbv'].T, v1[0]), a['orbo'])
        x2b += cp.matmul(cp.matmul(b['orbv'].T, v1[1]), b['orbo'])
        # stability doubles newton_ah's product
        return cp.hstack([x2a.reshape(n, -1), x2b.reshape(n, -1)]) * 2

    g = cp.hstack([a['fvo'].ravel(), b['fvo'].ravel()])
    hdiag = cp.hstack([(a['fvv'].diagonal()[:, None] - a['foo'].diagonal()).ravel(),
                       (b['fvv'].diagonal()[:, None] - b['foo'].diagonal()).ravel()]) * 2
    x0 = cp.zeros_like(g)
    x0[g != 0] = 1. / hdiag[g != 0]
    e, v = _davidson(hop, hdiag, x0, nroots, tol, log)
    log.info('uhf_internal: lowest eigs of H = %s', e)
    LAST['uhf_internal'] = e
    stable = not (e[0] < _THRESHOLD)
    stab_cpu.dump_status(log, stable, f'{mf.__class__}', 'internal')
    if stable:
        out = mf.mo_coeff
    else:
        out = (_rotate(mo[0], occ[0], v[0][:sa]),
               _rotate(mo[1], occ[1], v[0][sa:]))
    return (out, stable) if return_status else out


def uhf_external(mf, verbose=None, return_status=False,
                 nroots=STAB_NROOTS, tol=STAB_TOL):
    log = logger.new_logger(mf, verbose)
    mo, occ, (a, b) = _uhf_blocks(mf)
    na, va = a['orbo'].shape[1], a['orbv'].shape[1]
    nb, vb = b['orbo'].shape[1], b['orbv'].shape[1]

    # real -> complex
    sa = na * va
    vz = mf.gen_response(with_j=False, hermi=2)

    def hop1(xs):
        xs = cp.asarray(xs)
        n = len(xs)
        xa = xs[:, :sa].reshape(n, va, na)
        xb = xs[:, sa:].reshape(n, vb, nb)
        x2a = cp.matmul(a['fvv'], xa) - cp.matmul(xa, a['foo'])
        x2b = cp.matmul(b['fvv'], xb) - cp.matmul(xb, b['foo'])
        da = cp.matmul(cp.matmul(a['orbv'], xa), a['orbo'].T)
        db = cp.matmul(cp.matmul(b['orbv'], xb), b['orbo'].T)
        dm1 = cp.stack([da - da.transpose(0, 2, 1), db - db.transpose(0, 2, 1)])
        v1 = vz(dm1)
        x2a += cp.matmul(cp.matmul(a['orbv'].T, v1[0]), a['orbo'])
        x2b += cp.matmul(cp.matmul(b['orbv'].T, v1[1]), b['orbo'])
        return cp.hstack([x2a.reshape(n, -1), x2b.reshape(n, -1)])

    hdiag1 = cp.hstack([(a['fvv'].diagonal()[:, None] - a['foo'].diagonal()).ravel(),
                        (b['fvv'].diagonal()[:, None] - b['foo'].diagonal()).ravel()])
    x0 = cp.zeros_like(hdiag1)
    x0[hdiag1 > 1e-5] = 1. / hdiag1[hdiag1 > 1e-5]
    e1, _ = _davidson(hop1, hdiag1, x0, nroots, tol, log)
    log.info('uhf_real2complex: lowest eigs of H = %s', e1)
    LAST['uhf_real2complex'] = e1
    stab_cpu.dump_status(log, not (e1[0] < _THRESHOLD), f'{mf.__class__}',
                         'real -> complex')

    # UHF -> GHF (no spin flip of the total): alpha-vir/beta-occ and
    # beta-vir/alpha-occ rotations
    sab = va * nb
    v3 = mf.gen_response(with_j=False, hermi=0)

    def hop3(xs):
        xs = cp.asarray(xs)
        n = len(xs)
        xab = xs[:, :sab].reshape(n, va, nb)
        xba = xs[:, sab:].reshape(n, vb, na)
        x2ab = cp.matmul(a['fvv'], xab) - cp.matmul(xab, b['foo'])
        x2ba = cp.matmul(b['fvv'], xba) - cp.matmul(xba, a['foo'])
        dab = cp.matmul(cp.matmul(a['orbv'], xab), b['orbo'].T)
        dba = cp.matmul(cp.matmul(b['orbv'], xba), a['orbo'].T)
        dm1 = cp.stack([dab + dba.transpose(0, 2, 1),
                        dba + dab.transpose(0, 2, 1)])
        v1 = v3(dm1)
        x2ab += cp.matmul(cp.matmul(a['orbv'].T, v1[0]), b['orbo'])
        x2ba += cp.matmul(cp.matmul(b['orbv'].T, v1[1]), a['orbo'])
        return cp.hstack([x2ab.reshape(n, -1), x2ba.reshape(n, -1)])

    hdiag2 = cp.hstack([(a['fvv'].diagonal()[:, None] - b['foo'].diagonal()).ravel(),
                        (b['fvv'].diagonal()[:, None] - a['foo'].diagonal()).ravel()])
    x0 = cp.zeros_like(hdiag2)
    x0[hdiag2 > 1e-5] = 1. / hdiag2[hdiag2 > 1e-5]
    e3, v = _davidson(hop3, hdiag2, x0, nroots, tol, log, hermitian=False)
    log.info('uhf_external: lowest eigs of H = %s', e3)
    LAST['uhf_external'] = e3
    stable = not (e3[0] < _THRESHOLD)
    stab_cpu.dump_status(log, stable, f'{mf.__class__}', 'UHF/UKS -> GHF/GKS')
    # the GHF-shaped orbitals PySCF returns, built on the host as PySCF does
    import scipy.linalg
    moh = [cp.asnumpy(c) for c in mo]
    occh = [cp.asnumpy(o) for o in occ]
    out = scipy.linalg.block_diag(*moh)
    if not stable:
        from pyscf.soscf import newton_ah
        oa = numpy.where(occh[0] > 0)[0]
        vai = numpy.where(occh[0] == 0)[0]
        ob = numpy.where(occh[1] > 0)[0]
        vbi = numpy.where(occh[1] == 0)[0]
        nmo = len(occh[0])
        vv = cp.asnumpy(v[0])
        dx = numpy.zeros((nmo * 2, nmo * 2))
        dx[vai[:, None], nmo + ob] = vv[:sab].reshape(va, nb)
        dx[nmo + vbi[:, None], oa] = vv[sab:].reshape(vb, na)
        u = newton_ah.expmat(dx - dx.conj().T)
        out = out.dot(u)
        out = numpy.hstack([out[:, :na], out[:, nmo:nmo + nb],
                            out[:, na:nmo], out[:, nmo + nb:]])
    out = cp.asarray(out)
    return (out, stable) if return_status else out


# ----------------------------------------------------------------- ROHF


def rohf_internal(mf, verbose=None, return_status=False,
                  nroots=STAB_NROOTS, tol=STAB_TOL):
    """ROHF/ROKS internal: GPU4PySCF's single-vector ``gen_g_hop_rohf``
    (PySCF's), applied vector by vector."""
    from gpu4pyscf.scf.soscf import gen_g_hop_rohf
    log = logger.new_logger(mf, verbose)
    g, h_op, hdiag = gen_g_hop_rohf(mf, cp.asarray(mf.mo_coeff),
                                    cp.asarray(mf.mo_occ))
    hdiag = hdiag * 2

    def hop(xs):
        return cp.stack([h_op(x).real * 2 for x in cp.asarray(xs)])

    x0 = cp.zeros_like(g)
    x0[g != 0] = 1. / hdiag[g != 0]
    e, v = _davidson(hop, hdiag, x0, nroots, tol, log)
    log.info('rohf_internal: lowest eigs of H = %s', e)
    LAST['rohf_internal'] = e
    stable = not (e[0] < _THRESHOLD)
    stab_cpu.dump_status(log, stable, f'{mf.__class__}', 'internal')
    out = mf.mo_coeff if stable else _rotate(mf.mo_coeff, mf.mo_occ, v[0])
    return (out, stable) if return_status else out


# --------------------------------------------------------------- drivers


def _driver(internal_fn, external_fn):
    def stability(mf, internal=True, external=False, verbose=None,
                  return_status=False, nroots=STAB_NROOTS, tol=STAB_TOL):
        mo_i = mo_e = stable_i = stable_e = None
        if internal:
            mo_i, stable_i = internal_fn(mf, verbose, True, nroots, tol)
        if external:
            if external_fn is None:
                raise NotImplementedError('ROHF/ROKS external stability '
                                          '(PySCF has none either)')
            mo_e, stable_e = external_fn(mf, verbose, True, nroots, tol)
        if return_status:
            return mo_i, mo_e, stable_i, stable_e
        return mo_i, mo_e
    return stability


rhf_stability = _driver(rhf_internal, rhf_external)
uhf_stability = _driver(uhf_internal, uhf_external)
rohf_stability = _driver(rohf_internal, None)

rhf_stability.__doc__ = stab_cpu.rhf_stability.__doc__
uhf_stability.__doc__ = stab_cpu.uhf_stability.__doc__
rohf_stability.__doc__ = stab_cpu.rohf_stability.__doc__

# ROHF is a subclass of RHF in GPU4PySCF, so it gets its own binding
attach(rohf.ROHF, 'stability', rohf_stability)
attach(uhf.UHF, 'stability', uhf_stability)
attach(hf.RHF, 'stability', rhf_stability)

import sys as _sys                                         # noqa: E402
register_gpu_module('gpu4pyscf.scf.stability', _sys.modules[__name__])
