"""GPU CCSD amplitude transpose using the energy's batched contraction plans.

The fixed-topology amplitude Jacobian includes both operands of products,
all T1-dressed intermediates and independent-pair symmetry maps. Its
coefficients are frozen only for this amplitude solve. Geometry, PNO and
integral derivatives still use the complete outer residual pullback.
"""
import gc
import time
from types import MethodType, SimpleNamespace

import numpy as np
import cupy as cp
import cupyx

from . import reverse as ad
from .ccsd_gpu import CCSDGPU, _ChunkedPlan, _mv, _dot2, _wsum
from .devmem import pooled
from .ragged import Ragged
from .ragged_adjoint import AdjointRagged, AdjointChainPlan, AdjointDotPlan, _numeric
from .reference_cpu import rebind


class _ArrayModule:
    """Only the array operations used by the batched residual need tracing."""
    def __getattr__(self, name):
        return getattr(cp, name)

    @staticmethod
    def zeros(shape, dtype=float):
        return ad.Variable(cp.zeros(shape, dtype=dtype), tracked=False)

    @staticmethod
    def asarray(value, *args, **kwargs):
        return value if isinstance(value, ad.Variable) else cp.asarray(value, *args, **kwargs)

    @staticmethod
    def matmul(a, b):
        return np.matmul(a, b) if isinstance(a, ad.Variable) or isinstance(b, ad.Variable) else cp.matmul(a, b)

    einsum = staticmethod(ad.einsum)


_ARRAY = _ArrayModule()


def _scatter_flat(value, indices, updates):
    array = ad.primal(value).copy()
    cupyx.scatter_add(array.ravel(), indices, ad.primal(updates))
    return ad.operation(array, [value, updates], lambda bar: (bar, bar.ravel()[indices]))


class _AdjointChunkedPlan(_ChunkedPlan):
    _chunk = rebind(_ChunkedPlan._chunk, ChainPlan=AdjointChainPlan, DotPlan=AdjointDotPlan)

    def run(self, out, rags, w=None, alpha=1.):
        dot = isinstance(out, list)
        traced = isinstance(out, AdjointRagged) or (dot and any(
            isinstance(r, AdjointRagged) for r in out+rags))
        if not traced:
            return super().run(out, rags, w=w, alpha=alpha)
        if self.chunks and self.devs != [self.home]:
            raise NotImplementedError('The initial batched CCSD transpose uses one GPU')
        result = ad.Variable(cp.zeros(self.n), tracked=False) if dot else out
        for ch in self.chunks:
            factors = [None if i in self.s_ops else r for i, r in enumerate(rags)]
            offset = len(out) if dot else 0
            factories = {i+offset: (lambda ch=ch: self._S(ch)) for i in self.s_ops}
            if dot:
                result[ch['sel']] = ch['plan'].run(out, factors, constant_factories=factories)
            else:
                ch['plan'].run(out, factors, w=None if w is None else w[ch['sel']],
                               alpha=alpha, constant_factories=factories)
        return result


class _PlannedCCSD(CCSDGPU):
    __init__ = rebind(CCSDGPU.__init__, ChainPlan=AdjointChainPlan,
                      DotPlan=AdjointDotPlan, _ChunkedPlan=_AdjointChunkedPlan)


def _df_checkpoints(self, T, Tn, Fkc_bar, Fab, beta, Rsym):
    """Replay each bounded DF bucket on the GPU, never one pair at a time."""
    destinations = (Fkc_bar, Fab, beta, Rsym)
    diagonal = []
    info = self.adjoint_df_statistics
    for group in self.df:
        if group['dev'] != cp.cuda.Device().id:
            raise NotImplementedError('The initial batched CCSD transpose uses one GPU')

        def forward(group=group, T=T, Tn=Tn):
            outputs = [AdjointRagged(x.layout) for x in destinations]
            items = self._df_groups([group], T, Tn, *outputs)
            flat = [x for output in outputs for x in output.data]
            for _, _, fai, fab, fkc, tn in items:
                flat.extend((fai, fab, fkc, tn))
            return tuple(flat)

        outputs, statistics = ad.checkpoint(forward)
        info['groups'] += 1
        info['discarded_records'] += statistics['records']
        info['largest_inner_value_bytes'] = max(info['largest_inner_value_bytes'], statistics['inner_value_bytes'])
        info['stored_output_bytes'] += statistics['output_bytes']
        offset = 0
        for target in destinations:
            for g in range(target.layout.ngroups):
                target.data[g] = target.data[g]+outputs[offset]
                offset += 1
        if len(group['di']):
            diagonal.append((group, group['dsel'], *outputs[offset:offset+4]))
            offset += 4
        if offset != len(outputs):
            raise RuntimeError('Unexpected DF checkpoint output layout')
    return diagonal


def _freeze(value):
    if isinstance(value, ad.Variable):
        value = value.v
    if isinstance(value, list):
        return [_freeze(x) for x in value]
    if isinstance(value, tuple):
        return tuple(_freeze(x) for x in value)
    if isinstance(value, dict):
        return {k: _freeze(v) for k, v in value.items()}
    # Host arrays can be shared: the solver only replaces integral-list
    # entries when packing buckets. The list containers above are owned.
    return cp.asnumpy(value) if isinstance(value, cp.ndarray) else value


def _fixed_state(state):
    names = ('pl', 'n', 'strong', 'X', 'lmos', 'lmo_pos', 'pao', 'K', 'K_mibj',
             'J_ijmb', 'K_ivvv', 'L_iajb', 'J_ikac', 'K_iakc', 'iQk', 'iQa',
             'Qma', 'Qab', 'e_pno', 'T', 'T1', 'opt')
    result = SimpleNamespace(**{name: _freeze(getattr(state, name)) for name in names})
    matrices = {}
    for name in ('S_pao', 'F_lmo'):
        value = getattr(state.orb, name)
        indices = np.ix_(*(np.arange(n) for n in value.shape))
        matrices[name] = _freeze(value[indices])
    result.orb = SimpleNamespace(**matrices)
    # Bounded DF buckets are transferred when needed instead of keeping
    # every pair's padded four-index intermediates on the GPU.
    result.df_offload, result.verbose = True, 0
    return result


def _flat_positions(layout, item, rows, cols):
    offsets = np.cumsum([0]+[len(m)*r*c for m, (r, c) in zip(layout.members, layout.shapes)])
    group = layout.group[item]
    pr, pc = layout.shapes[group]
    return offsets[group]+layout.slot[item]*pr*pc+rows*pc+cols


def _amplitude_maps(solver, pairs, diagonal):
    positions1, positions2, indices2, selected2 = [], [], [], []
    offset = 0
    for i in range(solver.n):
        n = solver.LT1.rows[i]
        positions1.extend(_flat_positions(solver.LT1, i, np.arange(n), 0))
        offset += n
    for p in pairs:
        n = solver.nv[p]
        if p in diagonal:
            row, col = diagonal[p]
        else:
            row, col = np.indices((n, n)).reshape(2, -1)
        source = np.arange(offset, offset+len(row))
        positions = _flat_positions(solver.LP, p, row, col)
        positions2.extend(positions);indices2.extend(source);selected2.extend(positions)
        q = solver.ji[p]
        if q != p:
            positions2.extend(_flat_positions(solver.LP, q, col, row));indices2.extend(source)
        elif p in diagonal:
            off = row != col
            positions2.extend(_flat_positions(solver.LP, p, col[off], row[off]));indices2.extend(source[off])
        offset += len(row)
    return [cp.asarray(x, dtype=cp.int64) for x in (positions1, positions2, indices2, selected2)], offset


def _expanded(vector, fixed, positions, indices):
    result = fixed.copy()
    result[positions] = vector.v[indices]

    def reverse(bar):
        target = cp.zeros_like(vector.v)
        cupyx.scatter_add(target, indices, bar[positions])
        return (target,)
    return ad.operation(result, [vector], reverse)


class CCSDTranspose:
    """Lazy, single-GPU analytic J.T action for independent CCSD amplitudes."""
    xp = cp

    def __init__(self, state, variables, independent, residual):
        self.fixed = _fixed_state(state)
        self.pairs = tuple(variables.pairs)
        self.diagonal = dict(variables.diag)
        self.amplitudes = _freeze(independent).copy()
        self.expected_residual = _freeze(residual).copy()
        self.linearization = None
        self.released = False
        self.calls = 0
        self.statistics = dict(operator_kind='batched_ccsd_transpose', backend='cupy',
                               records=0, value_bytes=0, lazy_initialization=True,
                               amplitude_only=True, coefficients_approximated=False)
        self.last_statistics = {}

    def _build(self):
        from . import mgpu
        if len(mgpu.devices()) != 1:
            raise NotImplementedError('Set DLPNO_NGPU=1 for the initial batched CCSD transpose')
        start = time.perf_counter()
        solver = _PlannedCCSD(self.fixed, None)
        solver.init_state()
        view = SimpleNamespace(**vars(solver))
        self.fixed = None
        namespace = dict(cp=_ARRAY, Ragged=AdjointRagged,
                         _mv=rebind(_mv, cp=_ARRAY), _dot2=rebind(_dot2, cp=_ARRAY),
                         _wsum=rebind(_wsum, cp=_ARRAY))
        for name in ('swap', 'swapT', 'masked', '_beta_w', '_v3', '_df_groups'):
            setattr(view, name, MethodType(rebind(getattr(CCSDGPU, name), **namespace), view))
        for name in ('_gat', '_put', '_scal'):
            setattr(view, name, rebind(getattr(CCSDGPU, name), **namespace))
        view._df_all = MethodType(_df_checkpoints, view)
        view.adjoint_df_statistics = dict(groups=0, discarded_records=0,
                                          largest_inner_value_bytes=0, stored_output_bytes=0)
        maps, size = _amplitude_maps(solver, self.pairs, self.diagonal)
        if size != self.amplitudes.size:
            raise ValueError('The GPU CCSD amplitude map differs from the independent vector')
        single, double, indices, select = maps
        independent = ad.Variable(cp.asarray(self.amplitudes))
        view.T1 = AdjointRagged(solver.LT1).set_flat(_expanded(
            independent, solver.T1.flat(), single, cp.arange(len(single), dtype=cp.int64)))
        view.T = AdjointRagged(solver.LP).set_flat(_expanded(independent, solver.T.flat(), double, indices))
        evaluate = rebind(_batched_residuals, **namespace)
        r1, r2 = evaluate(view)
        residual = ad.concatenate([r1.flat()[single], r2.flat()[select]])
        error = float(cp.max(cp.abs(residual.v-cp.asarray(self.expected_residual))))
        if error > 2e-8:
            raise RuntimeError(f'The GPU CCSD residual differs from the outer residual by {error:.3e}')
        self.linearization = ad.Linearization([residual], [independent])
        self.view = view
        cp.cuda.get_current_stream().synchronize()
        self.statistics.update(self.linearization.statistics, lazy_initialization=False,
            build_seconds=time.perf_counter()-start, residual_replay_error=error,
            df_checkpoints=dict(view.adjoint_df_statistics), amplitude_map_bytes=sum(m.nbytes for m in maps),
            fixed_df_storage='host_pinned_buckets', derivative_plans='variable_size_batched_gemm')
        self.amplitudes = self.expected_residual = None

    def __call__(self, seeds):
        if self.released:
            raise RuntimeError('A released CCSD transpose cannot be reused')
        with pooled():
            if self.linearization is None:
                self._build()
            start = time.perf_counter()
            result = self.linearization([cp.asarray(seeds[0])])
            cp.cuda.get_current_stream().synchronize()
            self.calls += 1
            self.last_statistics = dict(self.linearization.last_statistics,
                operator_kind='batched_ccsd_transpose', backend='cupy', calls=self.calls,
                seconds=time.perf_counter()-start)
        return result

    def release(self):
        self.linearization = self.fixed = None
        self.__dict__.pop('view', None)
        self.released = True
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()


def _batched_residuals(self):
    from electrolites.dlpno.devmem import Prof
    prof = self.__dict__.setdefault('_prof', Prof('LCCSD residual'))
    prof('tn')
    T, T1, S = self.T, self.T1, self.S
    LP, LN, LV, LL, LT1, L11 = self.LP, self.LN, self.LV, self.LL, self.LT1, self.L11
    n = self.n
    Tt = T * 2. - T.T()
    Tn = self.p_tn.run(Ragged(LN), [S, T1])
    TiP = self.p_ti.run(Ragged(LV), [S, T1])
    TjP = self.p_tj.run(Ragged(LV), [S, T1])
    # ---- T1-dressed Fock matrices and the DF terms -------------------------
    sc = 2 * Tn.vdot_items(self.Jijmb) - Tn.vdot_items(self.swap(self.Kmibj))
    pij = cp.asarray(self.pi * n + self.pj)
    Fij_bar = self.F.copy()
    Fij_bar = _scatter_flat(Fij_bar, pij, sc)
    Fkc_bar = Ragged(LN)
    Fab = Ragged(LP)
    beta = Ragged(LL)
    Rsym = Ragged(LP)
    Fai = Ragged(LT1)
    prof('df')
    diag_items = self._df_all(T, Tn, Fkc_bar, Fab, beta, Rsym)
    prof('fock')
    v = self.p_fkj.run(Ragged(L11), [Fkc_bar, T1])
    Fkj = Fij_bar.copy()
    Fkj = _scatter_flat(Fkj, pij, self._scal(v))
    Fkc = self.p_fkc.run(Ragged(LV), [S, self.Lij, S, T1])
    for d, ds, fai, fab, fkc, Tnd in diag_items:
        di = d['di']
        g1, sl1 = LT1.group[di], LT1.slot[di]
        t1 = cp.zeros((len(di), fab.shape[-1]))
        for k, (g, s) in enumerate(zip(g1, sl1)):
            t1[k, :LT1.shapes[g][0]] = T1.data[g][s][:, 0]
        col = Fij_bar[d['lm_idx'][ds], cp.asarray(di)[:, None]] * d['lm_msk'][ds]
        fai = (fai + cp.einsum('gab,gb->ga', fab, t1)
               - cp.einsum('gka,gk->ga', Tnd, cp.einsum('gkb,gb->gk', fkc, t1))
               - cp.einsum('gka,gk->ga', Tnd, col))
        for k, (g, s) in enumerate(zip(g1, sl1)):
            Fai.data[g][s, :, 0] = fai[k, :LT1.shapes[g][0]]
    # ---- gamma, delta, F'' ------------------------------------------------
    prof('gamma_delta')
    K3 = self.K3
    ps = self.pslot[id(LP)]
    G = Ragged(LP)
    self.p_g1.run(G, [Tn, self.Jijmb], alpha=-1.)
    for g in range(LP.ngroups):
        if G.data[g].shape[0]:
            G.data[g] += _wsum(TjP.data[g][:, :, 0], K3[g])
    self.p_g3.run(G, [S, T1, T1, S, self.K, S], alpha=-1.)
    self.p_g4.run(G, [S, T, S, self.K, S], alpha=-.5)
    Dl = Ragged(LP)
    self.p_d1.run(Dl, [Tn, self.Kmibj], alpha=-2.)
    self.p_d1.run(Dl, [Tn, self.Jijmb], alpha=1.)
    for g in range(LP.ngroups):
        if Dl.data[g].shape[0]:
            Kq = K3[g][ps[g]]
            tiv = TiP.data[g][:, :, 0]
            Dl.data[g] += 2 * _mv(Kq, tiv).transpose(0, 2, 1)
            Dl.data[g] -= _wsum(tiv, Kq)
    self.p_d3.run(Dl, [S, T1, T1, S, self.Lij, S], alpha=-1.)
    self.p_d4.run(Dl, [S, Tt, S, self.Lij, S], alpha=.5)
    Fdd = Fkj.copy()
    Fdd = _scatter_flat(Fdd, self.fdd_flat, self.p_fdd.run([self.K], [S, Tt, S]))
    # ---- R1 ---------------------------------------------------------------
    prof('r1')
    R1 = Fai
    Fai = Fai.copy()
    self.p_r1a.run(R1, [S, self._v3(Tt)])
    w = self.p_r1b_w.run([self.K], [S, Tt, S])
    self.p_r1b.run(R1, [S, T1], w=w, alpha=-1.)
    self.p_r1c.run(R1, [S, Tt, Fkc])
    B = Ragged(LN)
    self.p_B1.run(B, [self.Kmibj, Tt])
    self.p_B2.run(B, [Tn, self.K, Tt])
    self.p_r1d.run(R1, [S, B], alpha=-1.)
    # ---- R2 ---------------------------------------------------------------
    prof('r2_b_f')
    self.p_rb.run(Rsym, [S, T, S], w=self._beta_w(beta))
    Fbc = Fab
    self.p_rf.run(Fbc, [S, Tt, self.K, S], alpha=-1.)
    for g in range(LP.ngroups):
        if Rsym.data[g].shape[0]:
            Tg, Fg = T.data[g], Fbc.data[g]
            Rsym.data[g] += cp.matmul(Tg, Fg.transpose(0, 2, 1)) + cp.matmul(Fg, Tg)
    R = Rsym + self.masked(self.swapT(Rsym), self.offdiag)
    prof('r2_cd')
    Cm = Ragged(LP)
    self.p_rc1.run(Cm, [self.Jikac, T, S], alpha=-1.)
    self.p_rc2.run(Cm, [S, G, S, T, S], alpha=-1.)
    Dm = Ragged(LP)
    self.p_rd1.run(Dm, [self.Kiakc, Tt, S], alpha=2.)
    self.p_rd1.run(Dm, [self.Jikac, Tt, S], alpha=-1.)
    self.p_rd3.run(Dm, [S, Dl, S, Tt, S])
    Rn = Cm * .5 + Cm.T() + Dm * .5
    self.p_rg.run(Rn, [S, T, S], w=Fdd.ravel()[self.rg_f], alpha=-1.)
    R = R + Rn + self.swapT(Rn)
    R = self.masked(R, self.strong)
    prof(None)
    self.last = dict(Fkj=Fkj, Fdd=Fdd, G=G, Dl=Dl, beta=beta, Fai=Fai, Tn=Tn)
    return R1, R
