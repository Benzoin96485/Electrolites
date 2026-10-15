"""Periodic TDA excitation energies for RKS/RHF on the GPU
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` X1: Gamma point, ``TDA``; X2: k-points
at kshift 0, ``KTDA``, at the end of the module).

GPU4PySCF 1.8.1 has no periodic TDDFT: ``pyscf.pbc.tdscf.TDA(gpu_mf)`` is an
AttributeError, the molecular GPU TDA on a Gamma-point object fails (the
PBC ``gen_response`` takes one density), and that response has a latent
argument-order bug in its singlet branch (D10: ``singlet`` lands in
``nr_rks_fxc``'s ``fxc`` slot).  This is PySCF's ``pbc.tdscf.rhf.TDA`` --
the molecular TDA operator with the exchange evaluated with ``exxdiv=None``
and orbital energies from that Fock -- built on the GPU periodic SCF's own
pieces, per trial vector:

    A z = (e_a - e_i) z + C_o^T v1[D] C_v,  D = 2 C_o z C_v^T,
    v1 = fxc_{singlet|triplet}[D/2] + (J[D] if singlet) - hyb/2 K[D],

J and K from the cell's GPU J/K (``exxdiv=None``), fxc from GPU4PySCF's
periodic ``nr_rks_fxc_st`` on the SCF grid (the kernel cached once), and
the lowest roots by the block Davidson of ``electrolites.stability``.

    td = electrolites.pbc_tdscf.TDA(gpu_cell_mf).run(nstates=5)
    td.e        # excitation energies (Hartree)
"""
import numpy as np
import cupy as cp

from pyscf import lib
from pyscf.lib import logger


def _fxc_batch(ni, cell, grids, xctype, fxc, dms, kpts):
    """fxc[D] for a stack of real symmetric Gamma-point densities
    (nz, nao, nao), LDA/GGA: per grid block one GEMM for all the
    densities' rho1 and one batched GEMM for the potentials (GPU4PySCF's
    ``nr_rks_fxc`` loops densities and einsums per block)."""
    nz, nao = dms.shape[:2]
    ao_deriv = 0 if xctype == 'LDA' else 1
    out = cp.zeros_like(dms)
    dcat = cp.ascontiguousarray(dms.transpose(1, 0, 2)).reshape(nao, -1)
    p0 = p1 = 0
    for ao_ks, weight, _ in ni.block_loop(cell, grids, ao_deriv, kpts,
                                          sort_grids=True):
        p0, p1 = p1, p1 + weight.size
        ao = ao_ks[0].real if cp.iscomplexobj(ao_ks) else ao_ks[0]
        wf = fxc[:, :, p0:p1] * weight
        if xctype == 'LDA':
            ao = ao.reshape(1, -1, nao)
        ng = ao.shape[1]
        c0 = (ao[0] @ dcat).reshape(ng, nz, nao)
        rho1 = cp.einsum('xgi,gzi->zxg', ao, c0)
        if xctype != 'LDA':
            rho1[:, 1:] *= 2
        wv = cp.einsum('zxg,xyg->zyg', rho1, wf)
        wv[:, 0] *= .5
        aow = cp.einsum('zyg,ygi->zgi', wv, ao)
        out += cp.matmul(ao[0].T[None], aow)
    return out + out.transpose(0, 2, 1)


def gen_response(mf, singlet=True, hermi=0):
    """The singlet or triplet linear response of a Gamma-point GPU
    RKS/RHF for any stack of densities (..., nao, nao):
    fxc_{singlet|triplet}[D/2] + (J[D] if singlet) - hyb/2 K[D], with
    the SCF's ``exxdiv`` (the TDA sets None).  Also D10's replacement for
    GPU4PySCF's own (``electrolites.fixes``)."""
    cell = mf.cell
    ni = mf._numint
    kpts = mf.kpt.reshape(1, 3)
    is_ks = hasattr(mf, 'xc')
    if is_ks:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=0)
        if omega != 0:
            raise NotImplementedError('PBC response: range-separated hybrids')
        dm0 = mf.make_rdm1()
        fxc = ni.cache_xc_kernel1(cell, mf.grids, mf.xc, dm0[None], 1,
                                  kpts, is_rhf=True)[2]
        xctype = ni._xc_type(mf.xc)
        fast_fxc = xctype in ('LDA', 'GGA')
        fxc_st = fxc[0, :, 0] + (fxc[0, :, 1] if singlet else -fxc[0, :, 1])
        if xctype == 'LDA':
            fxc_st = fxc_st.reshape(1, 1, -1)
    else:
        hyb = 1.
    ngrids = int(np.prod(cell.mesh))
    free = cp.cuda.runtime.memGetInfo()[0] + \
        cp.get_default_memory_pool().free_bytes()
    kblk = max(1, int(.2 * free // (cell.nao * ngrids * 16 * 3)))
    # fxc: per density 5 (GGA) block-sized AO-shaped buffers
    zblk = max(1, int(.2 * free // (cell.nao * 20000 * 8 * 6)))
    nao = cell.nao

    def vresp(dms):
        shape = dms.shape
        dms = cp.asarray(dms).reshape(-1, nao, nao)
        # the density (fxc, J) sees only the symmetric part of D
        ds = .5 * (dms + dms.transpose(0, 2, 1))
        v1 = cp.zeros_like(dms)
        if is_ks and hermi != 2 and fast_fxc:
            for z0 in range(0, len(dms), zblk):
                v1[z0:z0 + zblk] += _fxc_batch(
                    ni, cell, mf.grids, xctype, fxc_st,
                    .5 * ds[z0:z0 + zblk], kpts)
        elif is_ks and hermi != 2:
            v1 += cp.asarray(ni.nr_rks_fxc_st(
                cell, mf.grids, mf.xc, dm0, (.5 * ds)[:, None], 1,
                singlet, fxc, kpts)).reshape(dms.shape)
        if singlet and hermi != 2:
            v1 += cp.asarray(mf.get_j(cell, ds, 1)).reshape(dms.shape)
        if hyb != 0:
            # FFT exchange holds nao x ngrids complex per density
            for z0 in range(0, len(dms), kblk):
                z1 = z0 + kblk
                v1[z0:z1] -= .5 * hyb * cp.asarray(mf.get_k(
                    cell, dms[z0:z1], hermi)).reshape(dms[z0:z1].shape)
        return v1.reshape(shape)
    return vresp


class TDA(lib.StreamObject):
    """Gamma-point TDA on a GPU4PySCF periodic RKS/RHF."""

    def __init__(self, mf, singlet=True):
        self._scf = mf
        self.cell = mf.cell
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.singlet = singlet
        self.nstates = 3
        self.conv_tol = 1e-5
        self.e = self.xy = None

    def _orbital_energies(self):
        """mo_energy from the Fock matrix with exxdiv=None (PySCF's
        ``scf.addons.mo_energy_with_exxdiv_none``)."""
        mf = self._scf
        C = cp.asarray(mf.mo_coeff)
        with lib.temporary_env(mf, exxdiv=None):
            dm = mf.make_rdm1()
            f = mf.get_fock(dm=dm)
        return cp.diag(C.T @ cp.asarray(f) @ C)

    def gen_vind(self):
        mf = self._scf
        C = cp.asarray(mf.mo_coeff)
        occ = cp.asnumpy(mf.mo_occ)
        co = C[:, occ == 2]
        cv = C[:, occ == 0]
        no, nv = co.shape[1], cv.shape[1]
        e = self._orbital_energies()
        e_ia = (e[occ == 0][None, :] - e[occ == 2][:, None]).ravel()
        vresp = gen_response(mf, singlet=self.singlet)

        def vind(zs):
            zs = cp.asarray(zs).reshape(-1, no, nv)
            dms = 2 * co @ zs @ cv.T
            with lib.temporary_env(mf, exxdiv=None):
                v1 = vresp(dms)
            out = co.T @ v1 @ cv
            out += zs * e_ia.reshape(no, nv)[None]
            return out.reshape(len(zs), -1)
        return vind, e_ia

    def kernel(self, nstates=None):
        from electrolites.stability import _davidson
        if nstates is not None:
            self.nstates = nstates
        log = logger.new_logger(self)
        vind, hdiag = self.gen_vind()
        x0 = cp.zeros_like(hdiag)
        x0[int(cp.argmin(hdiag))] = 1.
        # residual norm sqrt(conv_tol), PySCF's lr_eigh criterion; start
        # from every (near-)degenerate partner of the nstates-th lowest
        # e_ia (a loose residual otherwise converges onto a set missing a
        # degenerate root)
        hs = cp.sort(hdiag)
        cut = float(hs[min(self.nstates, hs.size) - 1]) + 1e-4
        nguess = max(self.nstates + 1, int((hdiag <= cut).sum()) + 2)
        w, x = _davidson(vind, hdiag, x0, self.nstates, 100 * self.conv_tol,
                         log, nguess=nguess)
        self.e = np.asarray(w)
        self.xy = x
        log.note('GPU PBC TDA (%s) excitation energies (eV): %s',
                 'singlet' if self.singlet else 'triplet',
                 self.e * 27.211386245988)
        return self.e

    def run(self, nstates=None):
        self.kernel(nstates)
        return self


# ------------------------------------------------- X2: k-point TDA (kshift 0)
def gen_response_k(mf, singlet=True):
    """The singlet/triplet response of a GPU KRKS/KRHF for transition
    densities at kshift = 0, (nset, nkpts, nao, nao) complex:
    fxc_{s|t}[D/2] + (J[D] if singlet) - hyb/2 K[D], the SCF's exxdiv
    (the TDA sets None).  PySCF ``pbc.scf.khf`` response, hermi=0."""
    cell = mf.cell
    ni = mf._numint
    kpts = mf.kpts
    if hasattr(kpts, 'kpts'):
        kpts = kpts.kpts
    is_ks = hasattr(mf, 'xc')
    gamma_only = bool(np.allclose(np.asarray(kpts), 0))
    if is_ks:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=0)
        if omega != 0:
            raise NotImplementedError('PBC response: range-separated hybrids')
        dm0 = mf.make_rdm1()
        fxc = ni.cache_xc_kernel1(cell, mf.grids, mf.xc, dm0, 1, kpts,
                                  is_rhf=True)[2]
    else:
        hyb = 1.

    def vresp(dms):
        v1 = cp.zeros_like(dms)
        if is_ks:
            def f(d):
                return cp.asarray(ni.nr_rks_fxc_st(
                    cell, mf.grids, mf.xc, dm0, .5 * d, 0, singlet, fxc,
                    kpts)).reshape(dms.shape)
            if gamma_only:
                # real AOs: GPU4PySCF's fxc matrix cannot mix them with a
                # complex density; fxc is linear, so Re and Im separately
                v1 += f(dms.real) + 1j * f(dms.imag)
            else:
                v1 += f(dms)
        for z, d in enumerate(dms):
            if singlet:
                # J of a non-Hermitian D through its Hermitian parts (the
                # multigrid J takes hermi=1 only): D = H1 + i H2
                dh = d.conj().transpose(0, 2, 1)
                h1 = .5 * (d + dh)
                h2 = -.5j * (d - dh)
                v1[z] += cp.asarray(mf.get_j(cell, h1, 1, kpts)
                                    ).reshape(d.shape)
                v1[z] += 1j * cp.asarray(mf.get_j(cell, h2, 1, kpts)
                                         ).reshape(d.shape)
            if hyb != 0:
                v1[z] -= .5 * hyb * cp.asarray(mf.get_k(cell, d, 0, kpts)
                                               ).reshape(d.shape)
        return v1
    return vresp


class KTDA(lib.StreamObject):
    """k-point TDA at kshift = 0 on a GPU4PySCF KRKS/KRHF (PySCF's
    ``pbc.tdscf.krhf.TDA`` operator)."""

    def __init__(self, mf, singlet=True):
        self._scf = mf
        self.cell = mf.cell
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.singlet = singlet
        self.nstates = 3
        self.conv_tol = 1e-5
        self.e = self.xy = None

    def _orbital_energies(self):
        mf = self._scf
        with lib.temporary_env(mf, exxdiv=None):
            dm = mf.make_rdm1()
            f = cp.asarray(mf.get_fock(dm=dm))
        C = cp.asarray(mf.mo_coeff)
        return cp.einsum('kpi,kpq,kqi->ki', C.conj(), f, C).real

    def gen_vind(self):
        mf = self._scf
        C = cp.asarray(mf.mo_coeff)                      # (nk, nao, nmo)
        occ = cp.asnumpy(mf.mo_occ)
        nk = C.shape[0]
        no = int((occ[0] == 2).sum())
        if any(int((o == 2).sum()) != no for o in occ):
            raise NotImplementedError('k-point TDA: metallic occupations')
        co = C[:, :, :no]
        cv = C[:, :, no:]
        nv = cv.shape[2]
        e = self._orbital_energies()
        e_ia = e[:, no:][:, None, :] - e[:, :no][:, :, None]   # (k, o, v)
        vresp = gen_response_k(mf, singlet=self.singlet)

        def vind(zs):
            zs = cp.asarray(zs).reshape(-1, nk, no, nv)
            # dms[z,k] = 2 cv z^T co^H  (PySCF: einsum('ov,pv,qo->pq'))
            dms = 2 * cp.einsum('kpv,zkov,kqo->zkpq', cv, zs, co.conj())
            with lib.temporary_env(mf, exxdiv=None):
                v1 = vresp(dms)
            out = cp.einsum('zkpq,kqo,kpv->zkov', v1, co, cv.conj())
            out += zs * e_ia[None]
            return out.reshape(len(zs), -1)
        return vind, e_ia.ravel()

    def kernel(self, nstates=None):
        if nstates is not None:
            self.nstates = nstates
        from pyscf.lib import logger as plog
        log = plog.new_logger(self)
        vind, hdiag = self.gen_vind()
        w, x = _davidson_herm(vind, hdiag, self.nstates, self.conv_tol, log)
        self.e = np.asarray(w)
        self.xy = x
        log.note('GPU k-point TDA (%s) excitation energies (eV): %s',
                 'singlet' if self.singlet else 'triplet',
                 self.e * 27.211386245988)
        return self.e

    def run(self, nstates=None):
        self.kernel(nstates)
        return self


def _davidson_herm(hop, hdiag, nroots, conv_tol, log, max_cycle=100,
                   max_space=80):
    """Lowest roots of a complex Hermitian operator (rows in, rows out):
    block Davidson from the unit vectors on the lowest e_ia, residual
    sqrt(conv_tol) (PySCF's lr_eigh)."""
    size = hdiag.size
    hs = cp.sort(hdiag)
    cut = float(hs[min(nroots, size) - 1]) + 1e-4
    # every near-degenerate partner of the nroots-th lowest e_ia, and at
    # least 3 nroots vectors (fewer converged PBE0 on a 2-k diamond onto a
    # set missing two roots, 0.35 eV off)
    ndeg = int((hdiag <= cut).sum())
    idx = cp.argsort(hdiag)[:min(size, max(3 * nroots, ndeg + nroots))]
    V = cp.zeros((len(idx), size), dtype=np.complex128)
    V[cp.arange(len(idx)), idx] = 1
    AV = hop(V).astype(np.complex128)
    tol_r = conv_tol ** .5
    for cycle in range(max_cycle):
        h = cp.asnumpy(V.conj() @ AV.T)
        h = (h + h.conj().T) * .5
        w_all, u_all = np.linalg.eigh(h)
        w = w_all[:nroots]
        U = cp.asarray(u_all[:, :nroots])
        X = U.T @ V
        AX = U.T @ AV
        R = AX - cp.asarray(w)[:, None] * X
        rn = cp.asnumpy(cp.linalg.norm(R, axis=1))
        log.debug('k-TDA Davidson %d dim %d e %s |r| %s', cycle, len(V),
                  w, rn)
        if np.all(rn < tol_r):
            break
        d = hdiag[None, :] - cp.asarray(w)[:, None]
        d[abs(d) < 1e-8] = 1e-8
        new = (R / d)[rn >= tol_r]
        if len(V) + len(new) > max_space:
            V, AV = X, AX
        for v in new:                      # Gram-Schmidt twice
            for _ in range(2):
                v = v - V.T @ (V.conj() @ v)
            nv = float(cp.linalg.norm(v))
            if nv > 1e-6:
                v = v / nv
                V = cp.vstack([V, v[None]])
                AV = cp.vstack([AV, hop(v[None]).astype(np.complex128)])
    else:
        log.warn('k-TDA Davidson not converged, |r| %s', rn)
    return w, X
