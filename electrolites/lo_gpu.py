"""Orbital localisation on the GPU: Foster-Boys, Pipek-Mezey (any
population), Edmiston-Ruedenberg and intrinsic bond orbitals.

PySCF's ``lo`` runs on the host and raises on CuPy arrays.  Its CIAH
localisers rebuild AO quantities on every call of the cost function,
gradient and Hessian: Boys re-evaluates ``int1e_r`` (and ``int1e_r2``) and
transforms it, Pipek-Mezey recomputes ``orth_ao('meta_lowdin')`` and all
``natm`` projectors, and Edmiston-Ruedenberg makes ``nmo`` rank-one
densities and a direct J/K build.  IBO is a pure-Python loop over orbital
pairs and atoms per sweep.

Every one of these quantities is a congruence of a fixed tensor by the
current rotation u (mo = mo0 u): the dipole matrices dip(u) = u^T dip0 u,
the atomic projectors P_A(u) = u^T P_A(mo0) u, the DF three-index tensor
B^P(u) = u^T B^P(mo0) u.  So the AO work is done once, on the device, and
each CIAH step is batched device GEMMs of orbital size; PySCF's CIAH driver
(``soscf.ciah``) runs unchanged on top.  The objective functions are
PySCF's: the projectors come from PySCF's own ``atomic_pops`` on mo0, so
every population method it has (meta-Lowdin, Lowdin, Mulliken, IAO,
Becke) is the same here.  Edmiston-Ruedenberg uses density fitting (PySCF
uses exact J/K).  IBO's 2x2 sweeps run as parallel Jacobi rounds: n/2
disjoint pairs rotated at once (round-robin ordering), every pair's
atom-resolved populations from one segmented reduction.

    from electrolites import lo_gpu
    loc = lo_gpu.Boys(mol, mo).kernel()          # also PM(mol, mo, mf), ER
    ibos = lo_gpu.ibo(mol, mo_occ)
"""
import numpy as np
import cupy as cp

from pyscf import lib
from pyscf.lo import boys, pipek, edmiston, orth
from pyscf.lo import iao as _iao


def _host(a):
    return cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)


class _DeviceMixin:
    """CIAH localiser whose gradient, Hessian-vector product and cost run
    on device tensors congruent in u."""

    def _tensors(self):
        """(re)build the u-independent device tensors for self.mo_coeff."""
        mo = self.mo_coeff
        if getattr(self, '_mo_src', None) is not mo:
            self._mo_src = mo
            self._build(cp.asarray(mo))
            n = mo.shape[1]
            self._tril = cp.tril_indices(n, -1)

    def pack_uniq_var(self, mat):
        if isinstance(mat, cp.ndarray):
            n = mat.shape[0]
            i, j = cp.tril_indices(n, -1)
            return mat[i, j]
        return super().pack_uniq_var(mat)

    def _unpack(self, v):
        n = self.mo_coeff.shape[1]
        i, j = self._tril
        m = cp.zeros((n, n))
        m[i, j] = cp.asarray(v)
        return m - m.T

    def __init__(self, mol, mo_coeff=None, *args, **kwargs):
        if mo_coeff is not None:
            mo_coeff = _host(mo_coeff)
        super().__init__(mol, mo_coeff, *args, **kwargs)

    def kernel(self, mo_coeff=None, callback=None, verbose=None):
        if mo_coeff is not None:
            mo_coeff = _host(mo_coeff)
        elif self.mo_coeff is not None:
            self.mo_coeff = _host(self.mo_coeff)
        return boys.kernel(self, mo_coeff, callback, verbose)


def _congruence(u, t):
    """u^T t u over the leading batch axis of t (b, n, n)."""
    return cp.matmul(cp.matmul(u.T[None], t), u[None])


class Boys(_DeviceMixin, boys.Boys):
    """Foster-Boys: maximise sum_i |<i|r|i>|^2."""

    def _build(self, mo):
        mol = self.mol
        z = mol.atom_charges()
        centre = np.einsum('z,zx->x', z, mol.atom_coords()) / z.sum()
        with mol.with_common_origin(centre):
            r = cp.asarray(mol.intor_symmetric('int1e_r', comp=3))
            r2 = cp.asarray(mol.intor_symmetric('int1e_r2'))
        self._dip0 = cp.einsum('pi,xpq,qj->xij', mo, r, mo)
        self._r2 = float(cp.einsum('pi,pq,qi->', mo, r2, mo))

    def _dip(self, u):
        self._tensors()
        return _congruence(cp.asarray(u), self._dip0)

    def gen_g_hop(self, u):
        dip = self._dip(u)
        g0 = cp.einsum('xii,xip->pi', dip, dip)
        g = -self.pack_uniq_var(g0 - g0.T) * 2
        dd = cp.einsum('xii->xi', dip)
        h_diag = cp.einsum('xi,xp->pi', dd, dd) * 2
        h_diag -= g0.diagonal() + g0.diagonal()[:, None]
        h_diag += cp.einsum('xip,xip->pi', dip, dip) * 4
        h_diag = -self.pack_uniq_var(h_diag) * 2
        g0 = g0 + g0.T

        def h_op(x):
            x = self._unpack(x)
            hx = (x.T @ g0.T)
            hx += cp.einsum('xip,xi->pi', dip,
                            cp.einsum('qi,xiq->xi', x, dip)) * 2
            hx -= cp.einsum('xp,xip->pi', dd, cp.matmul(dip, x)) * 2
            hx -= cp.einsum('xip,xp->pi', dip,
                            cp.einsum('qp,xpq->xp', x, dip)) * 2
            return _host(-self.pack_uniq_var(hx - hx.T))

        return _host(g), h_op, _host(h_diag)

    def get_grad(self, u=None):
        if u is None:
            u = np.eye(self.mo_coeff.shape[1])
        dip = self._dip(u)
        g0 = cp.einsum('xii,xip->pi', dip, dip)
        return _host(-self.pack_uniq_var(g0 - g0.T) * 2)

    def cost_function(self, u=None):
        if u is None:
            u = np.eye(self.mo_coeff.shape[1])
        dip = self._dip(u)
        return self._r2 - float(cp.einsum('xii,xii->', dip, dip))


FB = BF = Boys


class PipekMezey(_DeviceMixin, pipek.PipekMezey):
    """Pipek-Mezey with PySCF's population methods (``pop_method``)."""

    def __init__(self, mol, mo_coeff=None, mf=None, pop_method=None):
        if mf is not None and isinstance(getattr(mf, 'mo_coeff', None),
                                         cp.ndarray):
            mf = mf.to_cpu()            # the IAO population reads mo_coeff
        if mo_coeff is not None:
            mo_coeff = _host(mo_coeff)
        pipek.PipekMezey.__init__(self, mol, mo_coeff, mf, pop_method)

    def _build(self, mo):
        proj = pipek.PipekMezey.atomic_pops(self, self.mol, _host(mo),
                                            self.pop_method)
        self._proj0 = cp.asarray(np.asarray(proj).real)

    def _proj(self, u):
        self._tensors()
        return _congruence(cp.asarray(u), self._proj0)

    def gen_g_hop(self, u):
        exponent = self.exponent
        projR = self._proj(u)
        pop = cp.einsum('xii->xi', projR)
        popexp1 = pop ** (exponent - 1)
        popexp2 = pop ** (exponent - 2)
        g = cp.einsum('xj,xij->ij', popexp1, projR)
        g = -2 * exponent * self.pack_uniq_var(g - g.T)
        g1 = cp.einsum('xi,xi->i', popexp1, pop)
        g2 = cp.einsum('xi,xj->ij', popexp1, pop)
        h_diag = -2 * exponent * (g1[:, None] - g2)
        g1 = cp.einsum('xi,xij->ij', popexp2, projR ** 2)
        h_diag += 4 * exponent * (exponent - 1) * g1
        h_diag = -self.pack_uniq_var(h_diag + h_diag.T)
        G = cp.einsum('xi,xij->ij', popexp1, projR)
        G += G.T

        def h_op(x):
            xR = self._unpack(x)
            j0 = popexp2 * cp.einsum('xik,ki->xi', projR, xR)
            hx = -4 * exponent * (exponent - 1) * cp.einsum(
                'xi,xij->ij', j0, projR)
            hx += -exponent * (G @ xR)
            hx += 2 * exponent * cp.einsum('xj,xij->ij', popexp1,
                                           cp.matmul(projR, xR))
            hx -= hx.T
            return _host(-self.pack_uniq_var(hx))

        return _host(g), h_op, _host(h_diag)

    def get_grad(self, u=None):
        if u is None:
            u = np.eye(self.mo_coeff.shape[1])
        projR = self._proj(u)
        popexp1 = cp.einsum('xii->xi', projR) ** (self.exponent - 1)
        g = cp.einsum('xj,xij->ij', popexp1, projR)
        return _host(-2 * self.exponent * self.pack_uniq_var(g - g.T))

    def cost_function(self, u=None):
        if u is None:
            u = np.eye(self.mo_coeff.shape[1])
        projR = self._proj(u)
        return float((cp.einsum('xii->xi', projR) ** self.exponent).sum())


PM = Pipek = PipekMezey


class EdmistonRuedenberg(_DeviceMixin, edmiston.EdmistonRuedenberg):
    """Edmiston-Ruedenberg, maximising sum_i (ii|ii), with density-fitted
    integrals (``auxbasis``; PySCF's default fitting basis if None)."""

    auxbasis = None

    def _build(self, mo):
        from pyscf import df
        from electrolites.dfmp2_grad import _RI
        auxmol = df.addons.make_auxmol(
            self.mol, self.auxbasis or df.addons.make_auxbasis(self.mol))
        ri = _RI(self.mol, auxmol)
        b = ri.fit(ri.raw(mo, mo))[0]                         # (p, q, P)
        self._b0 = cp.ascontiguousarray(b.transpose(2, 0, 1))  # (P, p, q)

    def get_jk(self, u):
        self._tensors()
        b = _congruence(cp.asarray(u), self._b0)               # (P, p, q)
        n = b.shape[1]
        bd = cp.einsum('Pii->Pi', b)
        vj = (bd.T @ b.reshape(len(b), -1)).reshape(n, n, n)
        bt = cp.ascontiguousarray(b.transpose(1, 0, 2))        # (i, P, q)
        vk = cp.matmul(bt.transpose(0, 2, 1), bt)              # (i, p, q)
        return vj, vk

    def gen_g_hop(self, u):
        vj, vk = self.get_jk(u)
        g0 = cp.einsum('iip->pi', vj)
        g = -self.pack_uniq_var(g0 - g0.T) * 2
        h_diag = cp.einsum('ipp->pi', vj) * 2
        gd = g0.diagonal()
        h_diag -= gd + gd[:, None]
        h_diag += cp.einsum('ipp->pi', vk) * 4
        h_diag = -self.pack_uniq_var(h_diag) * 2
        g0 = g0 + g0.T

        def h_op(x):
            x = self._unpack(x)
            hx = cp.einsum('iq,qp->pi', g0, x)
            hx += cp.einsum('qi,iqp->pi', x, vk) * 2
            hx -= cp.einsum('qp,piq->pi', x, vj) * 2
            hx -= cp.einsum('qp,piq->pi', x, vk) * 2
            return _host(-self.pack_uniq_var(hx - hx.T))

        return _host(g), h_op, _host(h_diag)

    def get_grad(self, u=None):
        if u is None:
            u = np.eye(self.mo_coeff.shape[1])
        vj = self.get_jk(u)[0]
        g0 = cp.einsum('iip->pi', vj)
        return _host(-self.pack_uniq_var(g0 - g0.T) * 2)

    def cost_function(self, u=None):
        if u is None:
            u = np.eye(self.mo_coeff.shape[1])
        self._tensors()
        b = _congruence(cp.asarray(u), self._b0)
        bd = cp.einsum('Pii->Pi', b)
        return float((bd * bd).sum())


ER = Edmiston = EdmistonRuedenberg


def _round_robin(n):
    """n-1 (n even) rounds of n/2 disjoint pairs covering every pair once."""
    m = n + (n & 1)
    ring = list(range(m))
    rounds = []
    for _ in range(m - 1):
        pairs = [(ring[k], ring[m - 1 - k]) for k in range(m // 2)]
        pairs = [(max(p), min(p)) for p in pairs if max(p) < n]
        rounds.append(np.array(pairs, dtype=np.int64).reshape(-1, 2))
        ring = [ring[0]] + [ring[-1]] + ring[1:-1]
    return rounds


def ibo(mol, orbocc, iaos=None, s=None, exponent=4, grad_tol=1e-8,
        max_iter=200, minao='minao', verbose=None):
    """Intrinsic bond orbitals, PySCF's ``lo.ibo.ibo(locmethod='IBO')``
    objective (IAO populations to the power ``exponent``), 2x2 rotations
    in parallel Jacobi rounds on the device."""
    from pyscf.lib import logger
    assert exponent in (2, 4)
    orbocc = _host(orbocc)
    if s is None:
        s = mol.intor_symmetric('int1e_ovlp')
    if iaos is None:
        iaos = _iao.iao(mol, orbocc, minao=minao)
    iaos = orth.vec_lowdin(_host(iaos), s)
    iao_mol = _iao.reference_mol(mol, minao=minao)
    off = iao_mol.offset_nr_by_atom()[:, 2:]
    niao = iaos.shape[1]
    seg = np.zeros((iao_mol.natm, niao))
    for a, (p0, p1) in enumerate(off):
        seg[a, p0:p1] = 1
    seg = cp.asarray(seg)
    C = cp.asarray(iaos.T @ s @ orbocc)                     # (niao, nocc)
    n = C.shape[1]
    rounds = [cp.asarray(r) for r in _round_robin(n)]
    log = logger.new_logger(mol, verbose)
    conv = False
    for it in range(max_iter):
        fgrad = 0.
        for r in rounds:
            if len(r) == 0:
                continue
            i, j = r[:, 0], r[:, 1]
            Ci, Cj = C[:, i], C[:, j]
            Qii = seg @ (Ci * Ci)                            # (natm, npair)
            Qij = seg @ (Ci * Cj)
            Qjj = seg @ (Cj * Cj)
            if exponent == 2:
                A = (4 * Qij ** 2 - (Qii - Qjj) ** 2).sum(0)
                B = (4 * Qij * (Qii - Qjj)).sum(0)
            else:
                B = (4 * Qij * (Qii ** 3 - Qjj ** 3)).sum(0)
                A = (-Qii ** 4 - Qjj ** 4
                     + 6 * (Qii ** 2 + Qjj ** 2) * Qij ** 2
                     + Qii ** 3 * Qjj + Qii * Qjj ** 3).sum(0)
            phi = .25 * cp.arctan2(B, -A)
            fgrad += float((B * B).sum())
            cs, ss = cp.cos(phi), cp.sin(phi)
            C[:, i] = cs * Ci + ss * Cj
            C[:, j] = -ss * Ci + cs * Cj
        fgrad = fgrad ** .5
        log.debug('IBO sweep %d  |g| %.2e', it + 1, fgrad)
        if fgrad < grad_tol:
            conv = True
            break
    if not conv:
        log.warn('IBO localisation not converged: |g| %.2e after %d sweeps',
                 fgrad, max_iter)
    C = _host(C)
    return iaos @ orth.vec_lowdin(C)
