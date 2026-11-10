"""Implicit MP2/CCSD/triples adjoints sharing the energy tensor equations.

This is the algebraic engine for the local-density gradient implementation.
Its initial dense input adapter is also an oracle for testing sparse block
adapters.  The dense adapter itself is NOT a linear-scaling implementation.
"""
from types import MethodType, SimpleNamespace
import gc
from functools import partial
import time

import numpy as np
from pyscf import df
from scipy.sparse.linalg import LinearOperator, gmres

from . import reverse as ad
from . import setup as su
from .ref_rhf import RefDLPNO
from .reference_cpu import rebind
from .response import ResponseDLPNO, _SHARED


def host(x):
    x = ad.primal(x)
    return x.get() if hasattr(x, 'get') else np.asarray(x)


def values(x):
    if hasattr(x, 'primal_copy'):
        return x.primal_copy()
    if isinstance(x, ad.Variable):
        return host(x).copy()
    if isinstance(x, list):
        return [values(y) for y in x]
    if isinstance(x, tuple):
        return tuple(values(y) for y in x)
    if isinstance(x, dict):
        return {k: values(v) for k, v in x.items()}
    if isinstance(x, SimpleNamespace):
        return SimpleNamespace(**values(vars(x)))
    return x


def orthocanonicalize(s, f, cut):
    nrm = 1 / np.sqrt(np.diag(s))
    sn = s*nrm[:, None]*nrm[None, :]
    piv = sorted(su.pivoted_cholesky(host(sn), cut))
    xp = ad._xp([s])
    metric = sn[np.ix_(piv, piv)]
    # The metric's eigenvectors are only an intermediate orthonormal
    # gauge. Their individual derivatives are undefined at degeneracy.
    # Differentiate the symmetric matrix function instead; a fixed right
    # frame recovers the original primal convention. The subsequent
    # Fock canonicalization cancels any change of this right frame.
    _, frame = xp.linalg.eigh(ad.primal(metric))
    x = ad.Variable(xp.zeros((len(s), len(piv))), tracked=False)
    x[piv] = ad.sym_power(metric, -.5)@frame
    x *= nrm[:, None]
    eps, v = ad.eigh(x.T@f@x)
    return x@v, eps


def canonicalize(x, f):
    eps, v = ad.eigh(x.T@f@x)
    return v, eps


def inverse_sqrt_solver(a):
    inverse = ad.sym_power(a, -.5)
    return lambda x: (inverse@x.reshape(len(a), -1)).reshape(x.shape)


class TriplesTranspose:
    """Exact amplitude transpose for the linear coupled triples equations.

    R_t = W_t + D_t T_t - sum_u f_tu S_tu**(3) P_tu(T_u).
    Only the denominator and local overlap matrices are stored. The
    geometry/CC sources still come from the complete residual pullback.
    """

    def __init__(self, state, diagonal, backend=None, gpu_batch_bytes=128 << 20):
        if backend not in (None, 'numpy', 'cupy'):
            raise ValueError("triples_backend must be 'numpy', 'cupy', or None")
        self.xp = state.xp if backend is None else np
        if backend == 'cupy':
            import cupy
            self.xp = cupy
        xp = self.xp
        self.released = False
        self.shapes = [t.shape for t in state.T3]
        self.offsets = np.cumsum([0]+[t.size for t in state.T3])
        self.diagonal = xp.asarray(diagonal)
        self.edges, overlaps = [], {}
        for t, rows in enumerate(state._triples_couplings()):
            for key, u, occ, old in rows:
                weight = float(host(state.orb.F_lmo[occ, old]))
                if abs(weight) < state.opt['F_CUT_T']:
                    continue
                if (t, u) not in overlaps:
                    metric = ad.primal(state.orb.S_pao[np.ix_(state.t_paos[t], state.t_paos[u])])
                    overlap = (ad.primal(state.X_tno[t]).T@metric
                               @ad.primal(state.X_tno[u]))
                    overlaps[t, u] = host(overlap) if xp is np else overlap
                # RefDLPNO._permute maps the sorted occupied labels to
                # the requested order. Its inverse sorts the labels,
                # with the same stable tie convention for repeated LMOs.
                inverse = tuple(np.argsort(key, kind='stable'))
                self.edges.append((t, u, weight, overlaps[t, u], inverse))
        elements = sum(s.size for s in overlaps.values())
        self.statistics = dict(operator_kind='explicit_triples_transpose', records=0,
                               backend='numpy' if xp is np else 'cupy',
                               couplings=len(self.edges), overlap_elements=elements,
                               value_elements=elements+self.diagonal.size,
                               value_bytes=8*(elements+self.diagonal.size))
        self.last_statistics = {}
        if xp is not np:
            # Keep long-lived plan coefficients out of cached temporary
            # energy buffers. Small live pieces in a split default-pool
            # allocation can prevent gigabytes of idle space being freed.
            self._coefficient_pool = xp.cuda.MemoryPool()
            with xp.cuda.using_allocator(self._coefficient_pool.malloc):
                self._prepare_gpu_plan(state.xp, gpu_batch_bytes)
            self._coefficient_pool.free_all_blocks()
            self.statistics.update(
                coefficient_allocator='dedicated_cupy_pool',
                coefficient_pool_used_bytes=self._coefficient_pool.used_bytes(),
                coefficient_pool_reserved_bytes=self._coefficient_pool.total_bytes())

    def _prepare_gpu_plan(self, coefficient_xp, budget):
        from .ragged import padded
        xp = self.xp
        ranks = np.array([shape[0] for shape in self.shapes])
        padding = padded(ranks)
        groups = {}
        for edge in self.edges:
            t, u, _, _, inverse = edge
            groups.setdefault((int(padding[t]), int(padding[u]), inverse), []).append(edge)
        self.plan = []
        largest = 0
        for (vt, vu, inverse), edges in groups.items():
            workspace = 8*4*max(vt, vu)**3
            step = min(65535, max(1, budget//workspace))
            for start in range(0, len(edges), step):
                batch = edges[start:start+step]
                tt, uu = np.array([(e[0], e[1]) for e in batch]).T
                # Construct overlaps on the tensor backend before moving
                # them. A hybrid solve retains exactly the CPU coefficients.
                overlaps = coefficient_xp.zeros((len(batch), vt, vu))
                for row, (t, u, _, overlap, _) in enumerate(batch):
                    overlaps[row, :ranks[t], :ranks[u]] = overlap
                nt = ranks[tt]
                self.plan.append(dict(vt=vt, vu=vu, inverse=inverse,
                    overlap=xp.asarray(overlaps),
                    input_offsets=xp.asarray(8*self.offsets[tt], dtype=xp.int64),
                    input_strides=xp.asarray(np.stack([nt**2, nt, np.ones_like(nt)], axis=1), dtype=xp.int64),
                    input_sizes=xp.asarray(np.repeat(nt[:, None], 3, axis=1), dtype=xp.int32),
                    output_offsets=xp.asarray(self.offsets[uu], dtype=xp.int64),
                    output_sizes=xp.asarray(ranks[uu], dtype=xp.int32),
                    weights=xp.asarray([-e[2] for e in batch])))
                largest = max(largest, workspace*len(batch))
        arrays = [v for batch in self.plan for v in batch.values() if hasattr(v, 'nbytes')]
        self.statistics.update(batches=len(self.plan), workspace_bound_bytes=largest,
                               value_bytes=self.diagonal.nbytes+sum(v.nbytes for v in arrays))
        self.edges = ()  # The batched plan now owns the necessary overlaps.

    def release(self):
        # These are coefficient copies owned by the transpose, not the
        # outer residual's differentiable orbital/integral operands.
        self.diagonal, self.edges = None, ()
        self.__dict__.pop('plan', None)
        pool = self.__dict__.pop('_coefficient_pool', None)
        if pool is not None:
            pool.free_all_blocks()
        self.released = True
        if self.xp is not np:
            self.xp.get_default_memory_pool().free_all_blocks()

    def __call__(self, seeds):
        if self.released:
            raise RuntimeError('A released triples transpose cannot be reused')
        vector = self.xp.asarray(host(seeds[0]) if self.xp is np else seeds[0])
        result = self.diagonal*vector
        if self.xp is not np:
            from .gather import gather3p, scatter3p
            for batch in self.plan:
                vt, vu, overlap = batch['vt'], batch['vu'], batch['overlap']
                pointers = batch['input_offsets']+vector.data.ptr
                term = gather3p(pointers, batch['input_strides'], batch['input_sizes'], vt, vt, vt)
                count = len(pointers)
                left = overlap.transpose(0, 2, 1)
                term = (left@term.reshape(count, vt, vt*vt)).reshape(count, vu, vt, vt)
                term = (term.reshape(count, vu*vt, vt)@overlap).reshape(count, vu, vt, vu)
                term = (left[:, None]@term).reshape(count, vu, vu, vu)
                term = term.transpose((0,)+tuple(a+1 for a in batch['inverse']))
                scatter3p(term, batch['output_offsets'], batch['output_sizes'], batch['weights'], result)
            self.last_statistics = dict(operator_kind='explicit_triples_transpose', backend='cupy',
                                        couplings=self.statistics['couplings'], batches=len(self.plan),
                                        output_elements=result.size)
            return [result]
        blocks = [vector[a:b].reshape(shape) for a, b, shape in
                  zip(self.offsets[:-1], self.offsets[1:], self.shapes)]
        output = [result[a:b].reshape(shape) for a, b, shape in
                  zip(self.offsets[:-1], self.offsets[1:], self.shapes)]
        for t, u, weight, overlap, inverse in self.edges:
            term = self.xp.einsum('aA,bB,cC,abc->ABC', overlap, overlap, overlap,
                                 blocks[t], optimize=True)
            output[u] -= weight*term.transpose(inverse)
        self.last_statistics = dict(operator_kind='explicit_triples_transpose',
                                    couplings=len(self.edges), output_elements=result.size)
        return [result]


class Amplitudes:
    """One flat independent amplitude vector, with local unpacking maps."""

    def __init__(self, state, cc=False, triples=False):
        self.state, self.cc, self.triples = state, cc, triples
        self.pairs = [] if triples else [p for p, (i, j) in enumerate(state.pl.pairs)
                                         if i <= j and (not cc or state.strong[p])]
        self.diag = {p: np.tril_indices(state.T[p].shape[0]) for p in self.pairs
                     if state.pl.pairs[p][0] == state.pl.pairs[p][1]}

    def pack(self, doubles, singles=None):
        if self.triples:
            part = doubles
        else:
            part = (list(singles) if singles is not None else []) + [
                doubles[p][self.diag[p]] if p in self.diag else doubles[p] for p in self.pairs]
        return ad.concatenate([x.ravel() for x in part])

    def install(self, vector):
        s, offset = self.state, 0
        if self.triples:
            old = s.T3
            s.T3 = []
            for t in old:
                s.T3.append(vector[offset:offset+t.size].reshape(t.shape))
                offset += t.size
            return
        # Local residual checkpoints retain the previous amplitude lists.
        # Replace each list once per installation, rather than copying the
        # full pair list separately for every local residual block.
        s.T = list(s.T)
        if self.cc:
            s.T1 = list(s.T1)
            for i, t in enumerate(s.T1):
                s.T1[i] = vector[offset:offset+t.size].reshape(t.shape)
                offset += t.size
        for p in self.pairs:
            shape = s.T[p].shape
            if p in self.diag:
                ix = self.diag[p]
                length = len(ix[0])
                t = s.numeric.zeros(shape)
                t[ix] = vector[offset:offset+length]
                t = t+t.T-np.diag(np.diag(t))
            else:
                length = int(np.prod(shape))
                t = vector[offset:offset+length].reshape(shape)
            s.T[p] = t
            if p != s.pl.ji[p]:
                s.T[s.pl.ji[p]] = t.T
            offset += length

    def denominator(self):
        s = self.state
        diagonal_f = [float(host(s.orb.F_lmo[i, i])) for i in range(s.n)]
        if self.triples:
            parts = []
            for t, (i, j, k) in enumerate(s.trip):
                e = host(s.e_tno[t])
                parts.append(e[:, None, None]+e[None, :, None]+e[None, None, :]
                             -diagonal_f[i]-diagonal_f[j]-diagonal_f[k])
            return host(self.pack(parts))
        doubles = []
        for p, (i, j) in enumerate(s.pl.pairs):
            e = host(s.e_pno[p])
            doubles.append(e[:, None]+e[None, :]-diagonal_f[i]-diagonal_f[j])
        singles = ([host(s.e_pno[s.pl.ij[i, i]])-diagonal_f[i] for i in range(s.n)] if self.cc else None)
        return host(self.pack(doubles, singles))


class DenseInputs:
    """Unsparsified oracle inputs; no coordinate response is constructed."""

    def __init__(self, base):
        if hasattr(base, '_distant_tail_energy'):
            raise ValueError('The hierarchical dipole energy requires local scalar input adjoints')
        mf = base.mf.to_cpu() if hasattr(base.mf, 'to_cpu') else base.mf
        mol = mf.mol
        with mol.with_common_orig((0., 0., 0.)):
            r = mol.intor('int1e_r', comp=3)
        self.data = dict(s=np.asarray(mf.get_ovlp()), f=np.asarray(mf.get_fock()),
                         co=np.asarray(mf.mo_coeff)[:, np.asarray(mf.mo_occ)>0],
                         cl=base.orb.C_lmo, r=r,
                         b=df.incore.aux_e2(mol, base.auxmol, 'int3c2e'),
                         m=base.auxmol.intor('int2c2e'))


class AdjointDLPNO(RefDLPNO):
    """A scalar energy graph with implicit, coordinate-independent solves."""

    def __init__(self, base, level='auto', backend='numpy', inputs=None, progress=None,
                 checkpoint_triples=False, checkpoint_integrals=False, triples_backend=None,
                 krylov_memory_mib=8192, ccsd_primal_backend='cpu', checkpoint_ccsd=False,
                 triples_primal_backend='cpu', checkpoint_pnos=False, ccsd_adjoint_backend=None):
        self.__dict__ = dict(base.__dict__)
        self.opt = dict(base.opt)
        self.e, self.timings, self.diagnostics = {}, {}, {}
        self.verbose = 0
        self.level = ('ccsd_t' if base.triples else 'ccsd') if level == 'auto' else level
        self.triples = None if self.level in ('mp2', 'ccsd') else base.triples
        self._base = base
        self.progress = progress
        if triples_backend not in (None, 'numpy', 'cupy'):
            raise ValueError("triples_backend must be 'numpy', 'cupy', or None")
        self.triples_backend = triples_backend
        if ccsd_primal_backend not in ('cpu', 'gpu'):
            raise ValueError("ccsd_primal_backend must be 'cpu' or 'gpu'")
        self.ccsd_primal_backend = ccsd_primal_backend
        if ccsd_adjoint_backend not in (None, 'cupy'):
            raise ValueError("ccsd_adjoint_backend must be None or 'cupy'")
        self.ccsd_adjoint_backend = ccsd_adjoint_backend
        if triples_primal_backend not in ('cpu', 'gpu'):
            raise ValueError("triples_primal_backend must be 'cpu' or 'gpu'")
        self.triples_primal_backend = triples_primal_backend
        if not np.isfinite(krylov_memory_mib) or krylov_memory_mib <= 0:
            raise ValueError('krylov_memory_mib must be finite and positive')
        self.krylov_memory_bytes = int(krylov_memory_mib*(1 << 20))
        if backend not in ('numpy', 'cupy'):
            raise ValueError("backend must be 'numpy' or 'cupy'")
        self.xp = np
        if backend == 'cupy':
            import cupy
            self.xp = cupy
        self.numeric = ad.Numpy(self.xp)
        self.inputs = inputs if inputs is not None else DenseInputs(base)
        self.leaves = {}
        names = dict(_names, np=self.numeric)
        for name in _SHARED:
            setattr(self, name, MethodType(rebind(getattr(RefDLPNO, name), **names), self))
        if checkpoint_triples:
            self.lccsd_t0 = MethodType(_checkpointed_triples, self)
            self._triplet_residual = MethodType(_checkpointed_triplet_residual, self)
        if checkpoint_integrals:
            self._pno_pair_integrals = MethodType(_checkpointed_pno_pair, self)
        if checkpoint_ccsd:
            self._R2_pair = MethodType(_checkpointed_ccsd_pair, self)
        if checkpoint_pnos:
            self._sc_mp2_pair_energy = MethodType(_checkpointed_sc_mp2_pair, self)
            self._lmp2_pno_pair = MethodType(_checkpointed_mp2_pno_pair, self)
            self._tno_triplet = MethodType(_checkpointed_tno_triplet, self)

    def _dual(self, value):
        # Name retained for the shared continuous dipole expression.
        if isinstance(value, ad.Variable):
            return value.copy()
        return ad.Variable(self.xp.asarray(value), tracked=False)

    def setup(self):
        self.leaves = {name: ad.Variable(self.xp.asarray(value)) for name, value in self.inputs.data.items()}
        s, f, co, cl, b, m, r = [self.leaves[k] for k in ('s', 'f', 'co', 'cl', 'b', 'm', 'r')]
        p = np.eye(len(s))-co@co.T@s
        p = p/np.sqrt(np.diag(p.T@s@p))[None, :]
        self.orb = SimpleNamespace(S=s, F=f, C_lmo=cl, C_pao=p, S_pao=p.T@s@p,
                                   F_pao=p.T@f@p, F_lmo=cl.T@f@cl,
                                   nbf=len(s), ncore=self._base.orb.ncore, naocc=self.n, mol=self.mol)
        self.J2c = m
        self.qij = ad.einsum('mnQ,mi,nj->Qij', b, cl, cl, optimize=True)
        self.qia = ad.einsum('mnQ,mi,nu->Qiu', b, cl, p, optimize=True)
        self.qab = ad.einsum('mnQ,mu,nv->Quv', b, p, p, optimize=True)
        self.qab *= (self._base.doi_uv > self.opt['T_CUT_DO_UV'])[None]
        self._S = {}
        self.dip_e = self._dipole_energy(r)

    def _primal_state(self):
        state = object.__new__(RefDLPNO)
        exclude = {'_base', 'xp', 'numeric', 'inputs', 'leaves', 'diagnostics', '_densities'} | set(_SHARED)
        state.__dict__ = {k: values(v) for k, v in self.__dict__.items() if k not in exclude}
        state._S = {}
        return state

    def _implicit(self, name, independent, residual, variables):
        diagonal = variables.denominator()
        if name == 'triples':
            jacobian_transpose = TriplesTranspose(self, diagonal, backend=self.triples_backend)
        elif name == 'ccsd' and self.ccsd_adjoint_backend == 'cupy':
            from .ccsd_adjoint_gpu import CCSDTranspose
            jacobian_transpose = CCSDTranspose(self, variables, independent, residual)
        else:
            jacobian_transpose = ad.Linearization([residual], [independent])
        gpu_krylov = name == 'triples' and jacobian_transpose.xp is not np
        if gpu_krylov:
            # The transpose already owns the device denominator. Keeping
            # a second flat host copy costs gigabytes for large T1 spaces.
            diagonal = None
        primal_error = float(np.max(abs(host(residual)), initial=0.))
        if primal_error > 1e-8:
            raise RuntimeError(f'{name} primal residual {primal_error:.3e}; tighten amplitude convergence')
        info = dict(primal_residual=primal_error, variables=independent.size,
                    lambda_solves=0, **jacobian_transpose.statistics)
        vector_bytes = 8*independent.size
        restart = min(independent.size, 40,
                      max(2, self.krylov_memory_bytes//max(1, vector_bytes)-1))
        info.update(krylov_restart=restart, krylov_basis_bytes=(restart+1)*vector_bytes,
                    krylov_memory_target_bytes=self.krylov_memory_bytes,
                    krylov_solver_backend='cupy' if gpu_krylov else 'scipy',
                    krylov_basis_backend='cupy' if gpu_krylov else 'numpy')
        self.diagnostics[name] = info

        def pullback(seed):
            if self.progress is not None:
                self.progress(dict(event='adjoint_start', stage=name, **info))
            if gpu_krylov:
                from .krylov_gpu import gmres as gpu_gmres
                xp = jacobian_transpose.xp
                # GPU4PySCF may allocate large vectors directly with CUDA,
                # outside CuPy's default pool. Return unused cached blocks
                # before allocating the RHS and Arnoldi workspace; live
                # operator and derivative arrays remain owned and intact.
                xp.get_default_memory_pool().free_all_blocks()
                # Copy once to the device before negating; never mutate
                # the outer graph's seed or form a full host RHS copy.
                rhs = xp.array(seed, dtype=xp.float64, copy=True)
                xp.negative(rhs, out=rhs)
                lam, solve_info = gpu_gmres(
                    lambda vector: jacobian_transpose([vector])[0], rhs,
                    jacobian_transpose.diagonal, restart=restart,
                    maxiter=100, rtol=1e-10, atol=1e-11, max_residual=1e-8)
                del rhs
                status = solve_info['status']
                error = solve_info['residual_maximum']
                iteration_count = solve_info['iterations']
                info['krylov_solver'] = solve_info
            else:
                def action(vector):
                    xp = getattr(jacobian_transpose, 'xp', self.xp)
                    return host(jacobian_transpose([xp.asarray(vector)])[0])

                size = seed.size
                operator = LinearOperator((size, size), matvec=action, dtype=float)
                preconditioner = LinearOperator(operator.shape, matvec=lambda x: x/diagonal, dtype=float)
                rhs = -host(seed)
                iterations = []
                lam, status = gmres(operator, rhs, M=preconditioner, rtol=1e-10, atol=1e-11,
                                    restart=restart, maxiter=100, callback=iterations.append,
                                    callback_type='pr_norm')
                error = float(np.max(abs(action(lam)-rhs), initial=0.))
                iteration_count = len(iterations)
            if status or error > 1e-8:
                raise RuntimeError(f'{name} adjoint failed: status={status}, residual={error:.3e}')
            if gpu_krylov and self.xp is np:
                # Only the accepted solution returns to the host density
                # graph; Arnoldi work vectors never cross this boundary.
                lam = host(lam)
            info.update(jacobian_transpose.statistics)
            info.update(lambda_solves=info['lambda_solves']+1, adjoint_residual=error,
                        iterations=iteration_count, vjp_accumulation=jacobian_transpose.last_statistics)
            # A consuming density sweep visits this implicit node exactly
            # once. Its outer coefficient pullback uses the independent
            # residual graph, so the amplitude-only GPU plan can go now.
            # Reusable VJPs retain their operator and remain callable.
            release = getattr(jacobian_transpose, 'release', None)
            if getattr(self, '_release_transposes', False) and release is not None:
                start = time.perf_counter()
                release()
                info.update(operator_released=True,
                            operator_release_seconds=time.perf_counter()-start)
            if self.progress is not None:
                self.progress(dict(event='adjoint_done', stage=name, **info))
            return (self.xp.asarray(lam),)

        return ad.operation(independent.v.copy(), [residual], pullback)

    def lmp2_iterations(self):
        state = self._primal_state()
        RefDLPNO.lmp2_iterations(state)
        variables = Amplitudes(self)
        independent = ad.Variable(self.xp.asarray(host(variables.pack(state.T))))
        variables.install(independent)
        residual, _ = self._mp2_residual()
        converged = self._implicit('mp2', independent, variables.pack(residual), variables)
        variables.install(converged)
        eij = self.numeric.array([np.sum(k*(2*t-t.T)) for k, t in zip(self.K, self.T)])
        self.e_lmp2 = np.sum(eij)
        return eij

    def lccsd_iterations(self):
        state = self._primal_state()
        start = time.perf_counter()
        if self.progress is not None:
            self.progress(dict(event='primal_start', stage='ccsd', backend=self.ccsd_primal_backend))
        if self.ccsd_primal_backend == 'gpu':
            from .ccsd_gpu import CCSDGPU
            # This numerical state owns independent copies of the graph's
            # integral lists. CCSDGPU may release them after uploading them.
            # Rebuild in the current PNO basis; no transport of amplitudes
            # from the original energy calculation is assumed.
            # Block input adapters expose indexed reads instead of an
            # array conversion. The GPU plans use these two reference
            # matrices densely, just as the GPU energy driver does.
            for name in ('S_pao', 'F_lmo'):
                matrix = getattr(state.orb, name)
                indices = np.ix_(*(np.arange(n) for n in matrix.shape))
                setattr(state.orb, name, host(matrix[indices]))
            solver = CCSDGPU(state, None)
            state.e_lccsd, state.de_weak = solver.kernel(verbose=False)
            state.T = solver.T.to_list()
            state.T1 = [t.ravel() for t in solver.T1.to_list()]
            del solver
            gc.collect()
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
        else:
            RefDLPNO.lccsd_iterations(state)
        info = dict(backend=self.ccsd_primal_backend, seconds=time.perf_counter()-start)
        self.diagnostics['ccsd_primal'] = info
        if self.progress is not None:
            self.progress(dict(event='primal_done', stage='ccsd', **info))
        self.T1 = [self._dual(x) for x in state.T1]
        variables = Amplitudes(self, cc=True)
        independent = ad.Variable(self.xp.asarray(host(variables.pack(state.T, state.T1))))
        variables.install(independent)
        r2, r1 = self._cc_residual()
        converged = self._implicit('ccsd', independent, variables.pack(r2, r1), variables)
        variables.install(converged)
        energy, weak = 0., 0.
        for p, (i, j) in enumerate(self.pl.pairs):
            ti = self.S(p, self.pl.ij[i, i])@self.T1[i]
            tj = self.S(p, self.pl.ij[j, j])@self.T1[j]
            eij = np.sum((self.T[p]+np.outer(ti, tj))*self.L_iajb[p])
            energy += eij
            if not self.strong[p]:
                weak += eij
        self.e_lccsd, self.de_weak = energy-weak, weak

    def lccsd_t_iterations(self):
        if not self.trip:
            self.diagnostics['triples'] = dict(variables=0, lambda_solves=0, adjoint_residual=0.)
            return self._dual(0.)
        state = self._primal_state()
        state.opt = dict(state.opt, T_CUT_ITER=0.)
        start = time.perf_counter()
        if self.progress is not None:
            self.progress(dict(event='primal_start', stage='triples', backend=self.triples_primal_backend))
        if self.triples_primal_backend == 'gpu':
            import cupy
            from .triples_gpu import TriplesGPU
            metric = state.orb.S_pao
            indices = np.ix_(*(np.arange(n) for n in metric.shape))
            state.S_pao_d = cupy.asarray(host(metric[indices]))
            fock = state.orb.F_lmo
            indices = np.ix_(*(np.arange(n) for n in fock.shape))
            state.orb.F_lmo = host(fock[indices])
            state._trip_arr = np.asarray(state.trip, dtype=int)
            TriplesGPU.lccsd_t_iterations(state, return_amplitudes=True)
            del state.S_pao_d
            cupy.get_default_memory_pool().free_all_blocks()
        else:
            RefDLPNO.lccsd_t_iterations(state)
        info = dict(backend=self.triples_primal_backend, seconds=time.perf_counter()-start)
        self.diagnostics['triples_primal'] = info
        if self.progress is not None:
            self.progress(dict(event='primal_done', stage='triples', **info))
        variables = Amplitudes(self, triples=True)
        independent = ad.Variable(self.xp.asarray(host(variables.pack(state.T3))))
        variables.install(independent)
        residual = variables.pack(self._triples_residual())
        converged = self._implicit('triples', independent, residual, variables)
        variables.install(converged)
        self.e_ijk = self.numeric.array([self._e_ijk(i, j, k, self.V3[t], self.T3[t])
                                        for t, (i, j, k) in enumerate(self.trip)])
        return np.sum(self.e_ijk)

    def densities(self, release_graph=False, release_inputs=False):
        """One reverse sweep, at most one solve of each amplitude adjoint."""
        if hasattr(self, '_densities'):
            return self._densities
        target = 'lmp2' if self.level == 'mp2' else 'corr'
        graph = ad.Linearization([self.e[target]], list(self.leaves.values()))
        previous = getattr(self, '_release_transposes', False)
        self._release_transposes = bool(release_graph)
        try:
            bars = graph([self.xp.ones_like(self.e[target].v)], release_graph=release_graph,
                         release_inputs=release_inputs)
        finally:
            self._release_transposes = previous
        self.diagnostics['density_graph'] = dict(graph.statistics, sweep=graph.last_statistics)
        self._densities = {name: host(bar) for name, bar in zip(self.leaves, bars)}
        return self._densities


def _checkpointed_pair_build(self, p, name):
    # Pair lists change between coarse screening and the LMP2 stage.
    # Each local replay retains its own immutable stage view.
    names = ('pl', 'orb', 'pao', 'ribfs', 'opt', 'qia', 'J2c', 'ncore_scale')
    view = SimpleNamespace(**{key: getattr(self, key) for key in names})
    view._K_pao = MethodType(self._K_pao.__func__, view)
    scalar = name == '_sc_mp2_pair_energy'
    evaluate = rebind(getattr(RefDLPNO, name), **dict(_names, np=self.numeric))

    def forward():
        result = evaluate(view, p)
        return (result,) if scalar else result

    result, info = ad.checkpoint(forward)
    label = 'sc_mp2_checkpoints' if scalar else 'mp2_pno_checkpoints'
    diagnostics = self.diagnostics.setdefault(label, dict(
        count=0, discarded_records=0, discarded_value_bytes=0, stored_output_bytes=0,
        largest_local_value_bytes=0))
    diagnostics['count'] += 1
    diagnostics['discarded_records'] += info['records']
    diagnostics['discarded_value_bytes'] += info['inner_value_bytes']
    diagnostics['stored_output_bytes'] += info['output_bytes']
    diagnostics['largest_local_value_bytes'] = max(diagnostics['largest_local_value_bytes'], info['inner_value_bytes'])
    return result[0] if scalar else result


def _checkpointed_sc_mp2_pair(self, p):
    return _checkpointed_pair_build(self, p, '_sc_mp2_pair_energy')


def _checkpointed_mp2_pno_pair(self, p):
    return _checkpointed_pair_build(self, p, '_lmp2_pno_pair')


def _checkpointed_tno_triplet(self, t, cut):
    # Triplet screening and importance scaling create new lists/arrays
    # between T0 prescreening, the final T0 sum, and iterative triples.
    # Each replay retains the inputs of its own stage. The pair-amplitude
    # list is immutable after its implicit solve and is shared by views.
    names = ('opt', 'pl', 'orb', 'trip', 't_paos', 'T', 'X', 'pao', 'tno_scale')
    view = SimpleNamespace(**{name: getattr(self, name) for name in names})
    evaluate = rebind(RefDLPNO._tno_triplet, **dict(_names, np=self.numeric))

    def forward():
        return evaluate(view, t, cut)

    result, info = ad.checkpoint(forward)
    diagnostics = self.diagnostics.setdefault('tno_subspace_checkpoints', dict(
        count=0, discarded_records=0, discarded_value_bytes=0, stored_output_bytes=0,
        largest_local_value_bytes=0))
    diagnostics['count'] += 1
    diagnostics['discarded_records'] += info['records']
    diagnostics['discarded_value_bytes'] += info['inner_value_bytes']
    diagnostics['stored_output_bytes'] += info['output_bytes']
    diagnostics['largest_local_value_bytes'] = max(
        diagnostics['largest_local_value_bytes'], info['inner_value_bytes'])
    return result


def _checkpointed_ccsd_overlap(self, p, q):
    """Share the exact overlap and its derivative across local pair replays."""
    info = self.diagnostics.setdefault('ccsd_shared_overlaps', dict(
        requests=0, cache_hits=0, checkpoints=0, discarded_records=0,
        discarded_value_bytes=0, stored_output_bytes=0))
    info['requests'] += 1
    key = (p, q)
    if key in self._S:
        info['cache_hits'] += 1
        return self._S[key]
    left, right, metric = self.X[p], self.X[q], self.orb.S_pao
    index = np.ix_(self.pao[p], self.pao[q])

    def forward():
        return (left.T @ metric[index] @ right,)

    (overlap,), statistics = ad.checkpoint(forward)
    self._S[key] = overlap
    info['checkpoints'] += 1
    info['discarded_records'] += statistics['records']
    info['discarded_value_bytes'] += statistics['inner_value_bytes']
    info['stored_output_bytes'] += statistics['output_bytes']
    return overlap


def _checkpointed_ccsd_overlaps_gpu(self, keys, *, prefetch=False):
    """Retain shared PAO-union metric and coefficient sources in each batch."""
    from .overlap_union_adjoint_gpu import union_batches, embedded_batch
    info=self.diagnostics.setdefault('ccsd_shared_overlaps',dict(
        requests=0,cache_hits=0,checkpoints=0,discarded_records=0,
        discarded_value_bytes=0,stored_output_bytes=0))
    missing=tuple(key for key in keys if key not in self._S)
    if prefetch:
        info['prefetch_keys']=info.get('prefetch_keys',0)+len(keys)
    else:
        info['requests']+=len(keys);info['cache_hits']+=len(keys)-len(missing)
    info['backend']='cupy_embedded_overlap_batches'
    info.setdefault('gpu_batches',0);info.setdefault('gpu_overlaps',0)
    info.setdefault('largest_gpu_array_workspace_bound',0)
    metric=self.orb.S_pao
    for group,union,involved,bound in union_batches(missing,self.pao,self.X):
        local={int(p):i for i,p in enumerate(involved)}
        coefficients=tuple(self.X[p] for p in involved)
        positions=tuple(np.searchsorted(union,self.pao[p]) for p in involved)
        couples=np.asarray([(local[p],local[q]) for p,q in group],dtype=np.int64)
        index=np.ix_(union,union)
        def forward(coefficients=coefficients,positions=positions,couples=couples,index=index,metric=metric):
            return embedded_batch(metric[index],coefficients,positions,couples)
        outputs,statistics=ad.checkpoint(forward)
        self._S.update(zip(group,outputs))
        info['checkpoints']+=len(group)
        info['discarded_records']+=statistics['records']
        info['discarded_value_bytes']+=statistics['inner_value_bytes']
        info['stored_output_bytes']+=statistics['output_bytes']
        info['gpu_batches']+=1;info['gpu_overlaps']+=len(group)
        info['largest_gpu_array_workspace_bound']=max(info['largest_gpu_array_workspace_bound'],bound)
    return {key:self._S[key] for key in keys}


def _ccsd_pair_overlap_keys(self, p):
    """The local overlap requests in RefDLPNO._R2_pair, in first-use order."""
    pl = self.pl
    i, j = pl.pairs[p]
    lm = self.lmos[p]
    keys = []
    if i <= j:
        for k in lm:
            for l in lm:
                kl = pl.ij[k, l]
                if kl >= 0:
                    keys.append((kl, p))
    for k in lm:
        ki, kj, ik, jk = pl.ij[k, i], pl.ij[k, j], pl.ij[i, k], pl.ij[j, k]
        keys.extend(((p, ki), (ki, kj), (kj, p),
                     (p, ik), (ik, jk), (jk, p)))
    for k in pl.neighbors(i):
        ik = pl.ij[i, k]
        keys.extend(((p, ik), (ik, p)))
    return tuple(dict.fromkeys(keys))


def _checkpointed_ccsd_pair(self, p, beta, gamma, delta, fdd):
    names = ('pl', 'orb', 'pao', 'X', 'lmos', 'iQa_t1', 'Qab', 'T_n',
             'Qma', 'Fab', 'K', 'J_ikac', 'K_iakc')
    view = SimpleNamespace(**{name: getattr(self, name) for name in names})
    # Amplitudes.install replaces its list; all pair checkpoints share
    # this immutable stage view without O(number_of_pairs**2) references.
    view.T = self.T
    # Create every shared overlap before entering this pair checkpoint.
    # Its boundary then explicitly retains each dependency, so a consuming
    # reverse sweep cannot release an overlap before its final local use.
    keys=_ccsd_pair_overlap_keys(self,p)
    if self.ccsd_adjoint_backend=='cupy':
        # Create the complete shared overlap boundary before the first pair
        # checkpoint. Grouping across pairs amortizes metric transfers.
        if getattr(self,'_ccsd_overlap_prefetch_id',None)!=id(self._S):
            requested=tuple(dict.fromkeys(key for pair in range(len(self.pl))
                if self.strong[pair] for key in _ccsd_pair_overlap_keys(self,pair)))
            _checkpointed_ccsd_overlaps_gpu(self,requested,prefetch=True)
            self._ccsd_overlap_prefetch_id=id(self._S)
            self.diagnostics['ccsd_shared_overlaps']['cache_hit_semantics']='includes_prefetched_values'
        overlaps={key:_checkpointed_ccsd_overlap(self,*key) for key in keys}
    else:
        overlaps={key:_checkpointed_ccsd_overlap(self,*key) for key in keys}
    view.S = lambda left, right: overlaps[left, right]
    evaluate = rebind(RefDLPNO._R2_pair, **dict(_names, np=self.numeric))

    def forward():
        # A missing request raises instead of creating an undeclared
        # dependency during replay. All overlap and metric bars are kept.
        return evaluate(view, p, beta, gamma, delta, fdd)

    result, info = ad.checkpoint(forward)
    diagnostics = self.diagnostics.setdefault('ccsd_pair_checkpoints', dict(
        count=0, discarded_records=0, discarded_value_bytes=0, stored_output_bytes=0,
        largest_local_value_bytes=0))
    diagnostics['count'] += 1
    diagnostics['discarded_records'] += info['records']
    diagnostics['discarded_value_bytes'] += info['inner_value_bytes']
    diagnostics['stored_output_bytes'] += info['output_bytes']
    diagnostics['largest_local_value_bytes'] = max(diagnostics['largest_local_value_bytes'], info['inner_value_bytes'])
    return result


def _checkpointed_triplet_residual(self, t, connections):
    names = ('trip', 'orb', 'e_tno', 'W3', 'T3', 'X_tno', 't_paos', 'opt')
    view = SimpleNamespace(**{name: getattr(self, name) for name in names})
    view._matmul_3d = MethodType(self._matmul_3d.__func__, view)
    evaluate = rebind(RefDLPNO._triplet_residual, **dict(_names, np=self.numeric))

    def forward():
        return (evaluate(view, t, connections),)

    result, info = ad.checkpoint(forward)
    diagnostics = self.diagnostics.setdefault('triples_residual_checkpoints', dict(
        count=0, discarded_records=0, discarded_value_bytes=0, stored_output_bytes=0,
        largest_local_value_bytes=0))
    diagnostics['count'] += 1
    diagnostics['discarded_records'] += info['records']
    diagnostics['discarded_value_bytes'] += info['inner_value_bytes']
    diagnostics['stored_output_bytes'] += info['output_bytes']
    diagnostics['largest_local_value_bytes'] = max(diagnostics['largest_local_value_bytes'], info['inner_value_bytes'])
    return result[0]


def _checkpointed_pno_pair(self, p):
    names = ('pl', 'orb', 'strong', 'ribfs', 'pao', 'lmos', 'X', 'qij', 'qia',
             'qab', 'J2c', 'dom', 'K')
    view = SimpleNamespace(**{name: getattr(self, name) for name in names})
    evaluate = rebind(RefDLPNO._pno_pair_integrals, **dict(_names, np=self.numeric))
    layout = []

    def forward():
        outputs = evaluate(view, p)
        flat, positions, current = [], {}, []
        for key, value in outputs.items():
            sequence = value if isinstance(value, list) else [value]
            indices = []
            for item in sequence:
                identity = id(item)
                if identity not in positions:
                    positions[identity] = len(flat)
                    flat.append(item)
                indices.append(positions[identity])
            current.append((key, isinstance(value, list), indices))
        if not layout:
            layout.extend(current)
        elif current != layout:
            raise RuntimeError('PNO integral checkpoint output structure changed')
        return tuple(flat)

    result, info = ad.checkpoint(forward)
    diagnostics = self.diagnostics.setdefault('integral_checkpoints', dict(
        count=0, discarded_records=0, discarded_value_bytes=0, stored_output_bytes=0,
        largest_local_value_bytes=0))
    diagnostics['count'] += 1
    diagnostics['discarded_records'] += info['records']
    diagnostics['discarded_value_bytes'] += info['inner_value_bytes']
    diagnostics['stored_output_bytes'] += info['output_bytes']
    diagnostics['largest_local_value_bytes'] = max(diagnostics['largest_local_value_bytes'], info['inner_value_bytes'])
    return {key: [result[i] for i in indices] if sequence else result[indices[0]]
            for key, sequence, indices in layout}


def _checkpointed_triples(self, save=False):
    # Capture just the immutable inputs to one triples stage. In
    # particular, do not capture self.e_ijk, which changes during the
    # loop, or the full state/implicit-solve closure cycles.
    names = ('pl', 'orb', 'trip', 't_ribfs', 't_paos', 't_lmos', 'X_tno', 'e_tno',
             'qij', 'qia', 'qab', 'J2c', 'pao', 'X', 'T', 'T1')
    view = SimpleNamespace(**{name: getattr(self, name) for name in names})
    view._S_tp = MethodType(self._S_tp.__func__, view)
    view._e_ijk = self._e_ijk
    from .triples_adjoint import triplet_source
    use_gpu = self.triples_backend == 'cupy'
    evaluate = partial(triplet_source, gpu=use_gpu)
    if save:
        self.W3, self.V3, self.T3 = [], [], []
    energies = []
    diagnostics = self.diagnostics.setdefault('triplet_checkpoints', dict(
        count=0, discarded_records=0, discarded_value_bytes=0, stored_output_bytes=0,
        largest_local_value_bytes=0))
    diagnostics['source_backend'] = 'cupy' if use_gpu else self.xp.__name__
    diagnostics['output_backend'] = self.xp.__name__
    for t in range(len(self.trip)):
        def forward(t=t, view=view, evaluate=evaluate, save=save):
            values = evaluate(view, t)
            return values if save else values[:1]
        result, info = ad.checkpoint(forward)
        energies.append(result[0])
        if save:
            self.W3.append(result[1])
            self.V3.append(result[2])
            self.T3.append(result[3])
        diagnostics['count'] += 1
        diagnostics['discarded_records'] += info['records']
        diagnostics['discarded_value_bytes'] += info['inner_value_bytes']
        diagnostics['stored_output_bytes'] += info['output_bytes']
        diagnostics['largest_local_value_bytes'] = max(diagnostics['largest_local_value_bytes'], info['inner_value_bytes'])
    self.e_ijk = self.numeric.array(energies)
    return np.sum(self.e_ijk)


_setup = SimpleNamespace(**{name: getattr(su, name) for name in dir(su) if not name.startswith('__')})
_setup.orthocanonicalize = orthocanonicalize
_setup.canonicalize = canonicalize
_names = dict(np=ad.Numpy(), lib=SimpleNamespace(einsum=ad.einsum), su=_setup,
              scipy=SimpleNamespace(linalg=ad.scipy_linalg), _sym_inv_sqrt_solver=inverse_sqrt_solver,
              orthocanonicalize=orthocanonicalize, einsum=ad.einsum)
for _name in ('kernel', '_dipole_energy', '_mp2_residual', '_cc_residual', '_triples_residual'):
    setattr(AdjointDLPNO, _name, rebind(getattr(ResponseDLPNO, _name), **_names))
AdjointDLPNO._e_ijk = staticmethod(rebind(RefDLPNO._e_ijk, **_names))
