"""Analytic tangent equations for the closed-shell DLPNO energy pipeline.

The tensor expressions are shared with ``ref_rhf`` by rebinding only their
numeric namespace.  The converged MP2 and CC amplitudes are differentiated
implicitly, J dt = -dR, using analytic Jacobian-vector products.  Neither
the nonlinear solver history nor nuclear finite differences are used.
"""
from types import SimpleNamespace, MethodType

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from . import setup as su
from .dual import Dual, Numpy, primal, tangent, einsum, eigh, sym_power, scipy_linalg, array_module
from .ref_rhf import RefDLPNO, _permute
from .reference_cpu import rebind


def _host(x):
    return x.get() if hasattr(x, 'get') else np.asarray(x)


def _values(x):
    if isinstance(x, Dual):
        return _host(x.v).copy()
    if isinstance(x, list):
        return [_values(y) for y in x]
    if isinstance(x, tuple):
        return tuple(_values(y) for y in x)
    if isinstance(x, dict):
        return {k: _values(v) for k, v in x.items()}
    if isinstance(x, SimpleNamespace):
        return SimpleNamespace(**_values(vars(x)))
    return x


def orthocanonicalize(s, f, cut):
    nrm = 1 / np.sqrt(np.diag(s))
    sn = s * nrm[:, None] * nrm[None, :]
    piv = sorted(su.pivoted_cholesky(_host(primal(sn)), cut))
    e, u = eigh(sn[np.ix_(piv, piv)])
    x = Dual(array_module(s).zeros((len(s), len(piv))))
    x[piv] = u / np.sqrt(e)
    x *= nrm[:, None]
    eps, v = eigh(x.T @ f @ x)
    return x @ v, eps


def canonicalize(x, f):
    eps, v = eigh(x.T @ f @ x)
    return v, eps


def _inverse_sqrt_solver(a):
    inverse = sym_power(a, -.5)
    return lambda x: (inverse @ x.reshape(len(a), -1)).reshape(x.shape)


class PairVariables:
    """Independent doubles (ij/ji transpose and ii symmetry), plus singles."""

    def __init__(self, state, cc=False):
        self.state, self.cc = state, cc
        self.pairs = [p for p, (i, j) in enumerate(state.pl.pairs)
                      if i <= j and (not cc or state.strong[p])]
        self.diag = {p: np.tril_indices(state.T[p].shape[0]) for p in self.pairs
                     if state.pl.pairs[p][0] == state.pl.pairs[p][1]}
        self.sizes = ([x.size for x in state.T1] if cc else []) + [
            len(self.diag[p][0]) if p in self.diag else state.T[p].size for p in self.pairs]

    def pack(self, doubles, singles=None, derivative=False):
        part = (list(singles) if singles is not None else []) + [
            doubles[p][self.diag[p]] if p in self.diag else doubles[p] for p in self.pairs]
        f = tangent if derivative else primal
        return np.concatenate([_host(f(x)).ravel() for x in part])

    def assign(self, vector):
        s, offset = self.state, 0
        if self.cc:
            for x in s.T1:
                x.d = s.xp.asarray(vector[offset:offset + x.size].reshape(x.shape)).copy()
                offset += x.size
        for p in self.pairs:
            x = s.T[p]
            if p in self.diag:
                d = np.zeros(x.shape)
                ix = self.diag[p]
                length = len(ix[0])
                d[ix] = vector[offset:offset + length]
                d = d + d.T - np.diag(np.diag(d))
            else:
                length = x.size
                d = vector[offset:offset + length].reshape(x.shape).copy()
            x.d = s.xp.asarray(d)
            q = s.pl.ji[p]
            if p != q:
                s.T[q] = x.T
            offset += length

    def denominator(self):
        s, f = self.state, primal(self.state.orb.F_lmo)
        doubles = []
        for p, (i, j) in enumerate(s.pl.pairs):
            e = primal(s.e_pno[p])
            doubles.append(e[:, None] + e[None, :] - f[i, i] - f[j, j])
        singles = ([primal(s.e_pno[s.pl.ij[i, i]]) - f[i, i] for i in range(s.n)]
                   if self.cc else None)
        return self.pack(doubles, singles)


class ResponseDLPNO(RefDLPNO):
    """One nuclear direction of a fixed-topology, fully relaxed energy."""

    def __init__(self, base, orbital, direction, level='ccsd', backend='numpy'):
        if hasattr(base, '_distant_tail_energy'):
            raise ValueError('The hierarchical dipole energy requires its matching local adjoint gradient')
        self.__dict__ = dict(base.__dict__)
        self.opt = dict(base.opt)
        self.verbose = 0
        self.e, self.timings, self.diagnostics = {}, {}, {}
        self.level = level
        self._orbital, self._direction = orbital, direction
        self.triples = None if level in ('mp2', 'ccsd') else base.triples
        self._base = base
        if backend not in ('numpy', 'cupy'):
            raise ValueError("backend must be 'numpy' or 'cupy'")
        self.xp = np
        if backend == 'cupy':
            import cupy
            self.xp = cupy
        self.numeric = Numpy(self.xp)
        if backend == 'cupy':
            names = dict(_names, np=self.numeric)
            for name in _SHARED:
                setattr(self, name, MethodType(rebind(getattr(RefDLPNO, name), **names), self))

    def _dual(self, value, derivative=0.):
        return Dual(self.xp.asarray(value), self.xp.asarray(derivative))

    def setup(self):
        r, d, base = self._orbital, self._direction, self._base
        s, f = self._dual(r.s, d.ds), self._dual(r.f, d.df)
        co, cl = self._dual(r.co, d.dc), self._dual(r.l, d.dl)
        p = np.eye(len(s)) - co @ co.T @ s
        p = p / np.sqrt(np.diag(p.T @ s @ p))[None, :]
        self.orb = SimpleNamespace(S=s, F=f, C_lmo=cl, C_pao=p, S_pao=p.T @ s @ p,
                                   F_pao=p.T @ f @ p, F_lmo=cl.T @ f @ cl,
                                   nbf=len(s), ncore=base.orb.ncore, naocc=self.n, mol=self.mol)
        b = self._dual(r.int3c, d.db)
        self.J2c = self._dual(r.metric, d.dm)
        self.qij = einsum('mnQ,mi,nj->Qij', b, cl, cl, optimize=True)
        self.qia = einsum('mnQ,mi,nu->Qiu', b, cl, p, optimize=True)
        self.qab = einsum('mnQ,mu,nv->Quv', b, p, p, optimize=True)
        self.qab *= (base.doi_uv > self.opt['T_CUT_DO_UV'])[None]
        self._S = {}
        self.dip_e = self._dipole_energy(self._dual(r.r, d.dr))

    def _dipole_energy(self, r):
        """Differentiate the actual distant-pair energy, with its fixed domains."""
        orb = self.orb
        cl, p, opt = orb.C_lmo, orb.C_pao, self.opt
        centers = einsum('mi,xmn,ni->ix', cl, r, cl, optimize=True)
        lp = einsum('mi,xmn,nu->ixu', cl, r, p, optimize=True)
        energy = self._dual(np.zeros((self.n, self.n)))
        mask = ((self.doi_ij > opt['T_CUT_DO_IJ']) | (np.abs(self.dip_b) > opt['T_CUT_PRE'])
                | np.eye(self.n, dtype=bool))
        needed = np.where(np.any(~mask, axis=0))[0]
        ds, es = {}, {}
        for i in needed:
            dom = su.contract_to_atoms(np.where(abs(self.doi_iu[i]) > opt['T_CUT_DO_PRE'])[0],
                                       self.bf_atom, self.atom_bfs)
            x, e = orthocanonicalize(orb.S_pao[np.ix_(dom, dom)],
                                     orb.F_pao[np.ix_(dom, dom)], opt['S_CUT'])
            ds[i], es[i] = lp[i][:, dom] @ x, e
        for i in needed:
            for j in needed:
                if j <= i or mask[i, j]:
                    continue
                rij = centers[i] - centers[j]
                distance = np.sqrt(np.sum(rij * rij))
                direction = rij / distance
                dip = ds[i].T @ ds[j] - 3 * np.outer(direction @ ds[i], direction @ ds[j])
                den = es[i][:, None] + es[j][None, :] - orb.F_lmo[i, i] - orb.F_lmo[j, j]
                energy[i, j] = -4 / distance**6 * np.sum(dip**2 / den)
                energy[j, i] = energy[i, j]
        return energy

    def _primal_state(self):
        state = object.__new__(RefDLPNO)
        state.__dict__ = {k: _values(v) for k, v in self.__dict__.items()
                          if k not in ('_base', '_orbital', '_direction', 'xp', 'numeric')
                          and k not in _SHARED}
        state._S = {}
        return state

    def _mp2_residual(self):
        f, pl, out = self.orb.F_lmo, self.pl, [None] * len(self.pl)
        for p, (i, j) in enumerate(pl.pairs):
            if i > j:
                continue
            e = self.e_pno[p]
            r = self.K[p] + (e[:, None] + e[None, :] - f[i, i] - f[j, j]) * self.T[p]
            for k in np.union1d(pl.neighbors(i), pl.neighbors(j)):
                kj, ik = pl.ij[k, j], pl.ij[i, k]
                if kj >= 0 and i != k and abs(f[i, k]) > self.opt['F_CUT']:
                    s = self.S(p, kj)
                    r -= f[i, k] * (s @ self.T[kj] @ s.T)
                if ik >= 0 and j != k and abs(f[k, j]) > self.opt['F_CUT']:
                    s = self.S(p, ik)
                    r -= f[k, j] * (s @ self.T[ik] @ s.T)
            out[p], out[pl.ji[p]] = r, r.T
        return out, None

    def _cc_residual(self):
        self._T_n()
        self._t1_ints()
        self._t1_fock()
        return self._R2(), self._R1()

    def _respond(self, cc):
        variables = PairVariables(self, cc)
        residual = self._cc_residual if cc else self._mp2_residual
        size = sum(variables.sizes)
        variables.assign(np.zeros(size))
        r2, r1 = residual()
        b = variables.pack(r2, r1, derivative=True)
        primal_error = float(np.max(abs(variables.pack(r2, r1))))
        if primal_error > 1e-8:
            raise RuntimeError(f'Amplitudes are insufficiently converged for response '
                               f'(residual={primal_error:.3e}); tighten E_CONVERGENCE and R_CONVERGENCE')

        def action(vector):
            variables.assign(vector)
            r2, r1 = residual()
            return variables.pack(r2, r1, derivative=True) - b

        operator = LinearOperator((size, size), matvec=action, dtype=float)
        den = variables.denominator()
        pre = LinearOperator(operator.shape, matvec=lambda x: x / den, dtype=float)
        dt, status = gmres(operator, -b, M=pre, atol=1e-11, rtol=1e-10,
                           restart=min(size, 40), maxiter=100)
        error = float(np.max(abs(action(dt) + b)))
        if status or error > 1e-8:
            raise RuntimeError(f'{"CCSD" if cc else "MP2"} response failed: status={status}, residual={error}')
        variables.assign(dt)
        self.diagnostics['ccsd' if cc else 'mp2'] = dict(
            response_residual=error, primal_residual=primal_error, variables=size)

    def lmp2_iterations(self):
        state = self._primal_state()
        RefDLPNO.lmp2_iterations(state)
        self.T = [self._dual(x) for x in state.T]
        self._respond(cc=False)
        eij = self.numeric.array([np.sum(k * (2 * t - t.T)) for k, t in zip(self.K, self.T)])
        self.e_lmp2 = np.sum(eij)
        return eij

    def lccsd_iterations(self):
        state = self._primal_state()
        RefDLPNO.lccsd_iterations(state)
        # Weak amplitudes are the projected MP2 amplitudes, not independent
        # CC variables.  Preserve their upstream response.
        self.T = [self._dual(x) if self.strong[p] else self._dual(x, self.T[p].d)
                  for p, x in enumerate(state.T)]
        self.T1 = [self._dual(x) for x in state.T1]
        self._respond(cc=True)
        energy, weak = 0., 0.
        for p, (i, j) in enumerate(self.pl.pairs):
            ti = self.S(p, self.pl.ij[i, i]) @ self.T1[i]
            tj = self.S(p, self.pl.ij[j, j]) @ self.T1[j]
            eij = np.sum((self.T[p] + np.outer(ti, tj)) * self.L_iajb[p])
            energy += eij
            if not self.strong[p]:
                weak += eij
        self.e_lccsd, self.de_weak = energy - weak, weak

    def _triples_residual(self):
        connections = self._triples_couplings()
        return [self._triplet_residual(t, connections[t]) for t in range(len(self.trip))]

    def lccsd_t_iterations(self):
        if not self.trip:
            self.diagnostics['triples'] = dict(variables=0, response_residual=0., primal_residual=0.)
            return self._dual(0.)
        state = self._primal_state()
        state.opt = dict(state.opt, T_CUT_ITER=0.)
        RefDLPNO.lccsd_t_iterations(state)
        self.T3 = [self._dual(t) for t in state.T3]
        sizes = [t.size for t in self.T3]
        offsets = np.cumsum([0] + sizes)

        def assign(vector):
            for t, a, b in zip(self.T3, offsets[:-1], offsets[1:]):
                t.d = self.xp.asarray(vector[a:b].reshape(t.shape)).copy()

        r = self._triples_residual()
        b = np.concatenate([_host(x.d).ravel() for x in r])
        primal_error = float(max(float(self.xp.max(abs(x.v))) for x in r))
        if primal_error > 1e-8:
            raise RuntimeError(f'Triples amplitudes are insufficiently converged for response '
                               f'(residual={primal_error:.3e})')

        def action(vector):
            assign(vector)
            return np.concatenate([_host(x.d).ravel() for x in self._triples_residual()]) - b

        f = primal(self.orb.F_lmo)
        denominators = []
        for t, (i, j, k) in enumerate(self.trip):
            e = primal(self.e_tno[t])
            denominators.append((e[:, None, None] + e[None, :, None] + e[None, None, :]
                                 - f[i, i] - f[j, j] - f[k, k]).ravel())
        den = np.concatenate([_host(d) for d in denominators])
        operator = LinearOperator((len(b), len(b)), matvec=action, dtype=float)
        pre = LinearOperator(operator.shape, matvec=lambda x: x / den, dtype=float)
        dt, status = gmres(operator, -b, M=pre, rtol=1e-10, atol=1e-11,
                           restart=min(len(b), 40), maxiter=100)
        error = float(np.max(abs(action(dt) + b)))
        if status or error > 1e-8:
            raise RuntimeError(f'Triples response failed: status={status}, residual={error}')
        assign(dt)
        self.diagnostics['triples'] = dict(variables=len(b), response_residual=error,
                                           primal_residual=primal_error)
        self.e_ijk = self.numeric.array([self._e_ijk(i, j, k, self.V3[t], self.T3[t])
                                   for t, (i, j, k) in enumerate(self.trip)])
        return np.sum(self.e_ijk)

    def kernel(self):
        opt = self.opt
        self.setup()
        mask = (self._initial_pair_mask if hasattr(self, '_initial_pair_mask') else
                ((self.doi_ij > opt['T_CUT_DO_IJ']) | (np.abs(self.dip_b) > opt['T_CUT_PRE'])
                 | np.eye(self.n, dtype=bool)))
        self.de_dipole = (self._distant_tail_energy if hasattr(self, '_distant_tail_energy')
                          else np.sum(self.dip_e[~mask]))
        self.set_pairs(mask, self.domains(opt['T_CUT_MKN'] * 100, opt['T_CUT_DO'] * 2))
        crude = self.sc_mp2_pair_energies()
        keep = su.empty_mask_like(mask)
        self.de_lmp2_eliminated = 0.
        for p, (i, j) in enumerate(self.pl.pairs):
            if abs(crude[p]) >= opt['T_CUT_PAIRS_MP2'] or i == j:
                keep[i, j] = True
            else:
                self.de_lmp2_eliminated += crude[p]
        self.set_pairs(keep, self.domains(opt['T_CUT_MKN'], opt['T_CUT_DO']))
        self.lmp2_pnos()
        eij = self.lmp2_iterations()
        self.strong = np.array([bool(abs(eij[p]) >= opt['T_CUT_PAIRS']) or i == j
                                for p, (i, j) in enumerate(self.pl.pairs)])
        self.e['lmp2'] = self.e_lmp2 + self.de_lmp2_eliminated + self.de_dipole + self.de_pno_total
        if self.level == 'mp2':
            return self.e['lmp2']
        self.recompute_pnos()
        self.pno_integrals()
        self.lccsd_iterations()
        corr = (self.e_lccsd + self.de_weak + self.de_lmp2_eliminated
                + self.de_dipole + self.de_pno_total)
        self.e.update(ccsd=corr, lccsd=self.e_lccsd, weak=self.de_weak,
                      sc_lmp2=self.de_lmp2_eliminated, dipole=self.de_dipole, pno_trunc=self.de_pno_total)
        if self.triples:
            self.triples_sparsity(True)
            self.tno_transform(opt['T_CUT_TNO_PRE'])
            self.lccsd_t0()
            self.triples_sparsity(False)
            self.tno_transform(opt['T_CUT_TNO'])
            et0 = self.lccsd_t0()
            et = et0 + self.de_t_screened
            self.e.update(t0=et, t_screened=self.de_t_screened)
            if self.triples in ('t', 't1'):
                self.sort_triplets(et0)
                self.tno_transform(opt['T_CUT_TNO'])
                crude = self.lccsd_t0(save=True)
                iterative = self.lccsd_t_iterations()
                et += iterative - crude
                self.e.update(t0_crude=crude, t_iter=iterative)
            self.e['t'] = et
            corr += et
        self.e['corr'] = corr
        return corr


# Keep the energy and response algebra identical, without a second copy of
# the several hundred lines of T1-dressed CC and triples tensor equations.
_setup = SimpleNamespace(**{n: getattr(su, n) for n in dir(su) if not n.startswith('__')})
_setup.orthocanonicalize = orthocanonicalize
_setup.canonicalize = canonicalize
_names = dict(np=Numpy(), lib=SimpleNamespace(einsum=einsum), su=_setup,
              scipy=SimpleNamespace(linalg=scipy_linalg), _sym_inv_sqrt_solver=_inverse_sqrt_solver)
_SHARED = ('_K_pao', '_sc_mp2_pair_energy', '_lmp2_pno_pair', 'sc_mp2_pair_energies',
              'lmp2_pnos', '_store_pair', 'S', 'recompute_pnos',
              'pno_integrals', '_pno_pair_integrals', '_T_n', '_t1_ints', '_t1_fock', '_beta', '_gamma', '_delta',
              '_Fkj_dd', '_R1', '_R2', '_R2_pair', 'triples_sparsity', 'tno_transform', '_tno_triplet', '_S_tp',
              '_triplet_t0', 'lccsd_t0', 'sort_triplets', '_matmul_3d', '_triplet_residual')
for _name in _SHARED:
    setattr(ResponseDLPNO, _name, rebind(getattr(RefDLPNO, _name), **_names))
ResponseDLPNO._e_ijk = staticmethod(rebind(RefDLPNO._e_ijk, **_names))
