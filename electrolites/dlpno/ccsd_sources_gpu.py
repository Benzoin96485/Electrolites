"""Complete batched CCSD coefficient VJPs, with resident GPU intermediates.

This uses the energy's ragged contraction plans, tracing all coefficient
operands as well as both amplitudes. Host packing/scattering is one coarse
AD operation per bucket, so no coefficient or overlap derivative is lost.
The initial path requires a single primary GPU and bounded cached overlaps;
its purpose is to make useful batches before distributing them.
"""
import os
import time
import weakref
from types import MethodType, SimpleNamespace
import numpy as np
import cupy as cp
import cupyx
from . import reverse as ad
from .adjoint import _checkpointed_ccsd_overlaps_gpu, host
from .ccsd_adjoint_gpu import _PlannedCCSD, _fixed_state, _ARRAY, _df_checkpoints, _batched_residuals, _amplitude_maps
from .ccsd_gpu import CCSDGPU, _mv, _dot2, _wsum, _VCouples, _Code
from .ragged_adjoint import AdjointRagged, AdjointChainPlan
from .ragged import op
from .reference_cpu import rebind


class _CachedEnvironment:
    def get(self, name, default=None):
        return '0' if name == 'DLPNO_ONFLY_S' else os.environ.get(name, default)


class _FullCouples(_VCouples):
    """Retain reverse overlaps explicitly for unrestricted metric VJPs."""
    def finalize(self, terms):
        columns = [(t,k) for t in terms for k,v in t.cols.items() if isinstance(v,_Code)]
        super().finalize(terms)
        old = np.asarray(self.p,dtype=np.int64)*self.npair+np.asarray(self.q,dtype=np.int64)
        reverse = np.asarray(self.q,dtype=np.int64)*self.npair+np.asarray(self.p,dtype=np.int64)
        codes = np.unique(np.concatenate((old,reverse)))
        mapping = np.searchsorted(codes,old)
        for term,key in columns:
            term.cols[key] = mapping[term.cols[key]]
        self.p,self.q = (codes//self.npair).tolist(),(codes%self.npair).tolist()


class SourcePlan(_PlannedCCSD):
    """Reusable topology with fully differentiable, bucket-packed sources."""
    _initialize = rebind(_PlannedCCSD.__init__, os=SimpleNamespace(environ=_CachedEnvironment()), _VCouples=_FullCouples)

    def __init__(self, state, *, budget_bytes=2 << 30, prepare_only=False):
        from . import mgpu
        if len(mgpu.devices()) != 1:
            raise ValueError('Batched CCSD sources currently require DLPNO_NGPU=1')
        if state.xp is not np:
            raise ValueError('Batched CCSD sources currently require a host outer graph')
        self.source = state
        self.host_inputs, self.device_inputs = [], []
        self.budget_bytes, self.input_bytes = int(budget_bytes), 0
        self.released = False
        self.statistics = dict(kind='batched_ccsd_complete_sources', backend='cupy',
            coefficients_approximated=False, amplitude_only=False, forward_seconds=0., reverse_seconds=0., reverse_calls=0)
        self.owner = weakref.ref(state)
        self.device = cp.cuda.Device().id
        self.pool = cp.cuda.MemoryPool()
        self.finished = False
        start = time.perf_counter()
        try:
            with cp.cuda.using_allocator(self.pool.malloc):
                self._initialize(_fixed_state(state, before_ccsd=prepare_only), None)
                # The energy kernel replaces S(ik,p) with S(p,ik).T in
                # G. For full source VJPs keep S(ik,p) as a separate edge.
                old = self.p_rg
                couples = {(p,q):i for i,(p,q) in enumerate(zip(self.C.p,self.C.q))}
                indices = old.operands[0].idx
                reverse = np.asarray([couples[self.C.q[i],self.C.p[i]] for i in indices],dtype=np.int64)
                self.p_rg = AdjointChainPlan(self.LP,old.output_indices,
                    [old.operands[0],old.operands[1],op(self.LS,reverse)])
                namespace = dict(cp=_ARRAY, Ragged=AdjointRagged,
                                 _mv=rebind(_mv, cp=_ARRAY), _dot2=rebind(_dot2, cp=_ARRAY),
                                 _wsum=rebind(_wsum, cp=_ARRAY))
                for name in ('swap', 'swapT', 'masked', '_beta_w', '_v3', '_df_groups'):
                    setattr(self, name, MethodType(rebind(getattr(CCSDGPU, name), **namespace), self))
                for name in ('_gat', '_put', '_scal'):
                    setattr(self, name, rebind(getattr(CCSDGPU, name), **namespace))
                cp.cuda.get_current_stream().synchronize()
            self.statistics.update(prepare_seconds=time.perf_counter()-start,
                prepare_only=bool(prepare_only), amplitudes_deferred=bool(prepare_only))
        except BaseException:
            self.release()
            raise
        if not prepare_only:
            self.finish()

    def finish(self):
        """Capture the current amplitudes only after the numerical solve joins."""
        if self.released or self.finished:
            raise RuntimeError('CCSD source preparation must be finished exactly once')
        start = time.perf_counter()
        try:
            with cp.cuda.Device(self.device), cp.cuda.using_allocator(self.pool.malloc):
                # The initial snapshot can contain MP2 amplitudes. No amplitude
                # leaf is captured by preparation; all final leaves come here.
                from .ccsd_adjoint_gpu import _freeze
                self.ref.T = _freeze(self.source.T)
                self.ref.T1 = _freeze(self.source.T1)
                self.init_state()
                namespace = dict(cp=_ARRAY, Ragged=AdjointRagged,
                                 _mv=rebind(_mv, cp=_ARRAY), _dot2=rebind(_dot2, cp=_ARRAY),
                                 _wsum=rebind(_wsum, cp=_ARRAY))
                keep_df = os.environ.get('DLPNO_CCSD_SOURCE_KEEP_DF','0') == '1'
                self._df_all = MethodType(CCSDGPU._df_all if keep_df else _df_checkpoints, self)
                if not keep_df and os.environ.get('DLPNO_CCSD_COMPACT_DF') == '1':
                    from .ccsd_df_compact_gpu import compact_df_checkpoints
                    self._df_all = MethodType(compact_df_checkpoints, self)
                if os.environ.get('DLPNO_CCSD_DF_DEVICES'):
                    from .ccsd_df_parallel_gpu import distributed_df
                    self._df_all = MethodType(distributed_df, self)
                    keep_df = True
                self.statistics['resident_df_graph'] = keep_df
                self.adjoint_df_statistics = dict(groups=0, discarded_records=0,
                    largest_inner_value_bytes=0, stored_output_bytes=0)
                r1, r2 = rebind(_batched_residuals, **namespace)(self)
                if keep_df:
                    self.adjoint_df_statistics.update(groups=len(self.df),retained=True)
                outputs = [r2.get(p) for p in range(self.npair)]+[r1.get(i).reshape(-1) for i in range(self.n)]
                self.shapes = [x.shape for x in outputs]
                self.offsets = np.cumsum([0]+[x.size for x in outputs])
                self.output = ad.concatenate([x.ravel() for x in outputs])
                self.graph = ad.Linearization([self.output], self.device_inputs)
                self.statistics['ragged_flat_cache'] = {name:dict(value.flat_cache_statistics)
                    for name,value in vars(self).items() if isinstance(value,AdjointRagged)
                    and hasattr(value,'flat_cache_statistics')}
                self.statistics.update(self.graph.statistics, input_bytes=self.input_bytes,
                                       input_buckets=len(self.host_inputs), df_checkpoints=dict(self.adjoint_df_statistics))
                # Logical node values count aliased views separately. Record
                # actual pool storage alongside this explicit logical budget.
                self.statistics.update(pool_used_bytes_at_graph_finish=self.pool.used_bytes(),
                    pool_reserved_bytes_at_graph_finish=self.pool.total_bytes(),
                    logical_graph_value_bytes_including_views=self.graph.statistics['value_bytes'])
                if self.graph.statistics['value_bytes'] > self.budget_bytes:
                    raise MemoryError('Batched CCSD source graph exceeds its explicit resident-value budget')
                self.packed = cp.asnumpy(self.output.v)
                cp.cuda.get_current_stream().synchronize()
            self.statistics.update(forward_seconds=time.perf_counter()-start,
                pool_used_bytes=self.pool.used_bytes(), pool_reserved_bytes=self.pool.total_bytes())
            self.finished = True
            self.statistics['finish_seconds'] = self.statistics['forward_seconds']
            self.statistics['forward_seconds'] += self.statistics['prepare_seconds']
        except BaseException:
            self.release()
            raise
        finally:
            self.__dict__.pop('source', None)
            self.__dict__.pop('ref', None)
        return self

    def _capture(self, value):
        raw = host(value)
        if raw.dtype != np.float64:
            raise ValueError('CCSD source bucket must have real float64 precision')
        self.input_bytes += raw.nbytes
        if self.input_bytes > self.budget_bytes:
            raise MemoryError('Batched CCSD source inputs exceed their explicit budget')
        parent = ad.Variable._from_record(value._record) if isinstance(value, ad.Variable) else value
        leaf = ad.Variable(cp.asarray(raw), tracked=isinstance(parent, ad.Variable) and parent._record.tracked)
        self.host_inputs.append(parent)
        self.device_inputs.append(leaf)
        return leaf

    def _pack(self, shape, entries):
        """One host packing node, one upload, and exact unpacked cotangents."""
        if 8*int(np.prod(shape)) > self.budget_bytes-self.input_bytes:
            raise MemoryError('CCSD source bucket exceeds remaining input budget')
        entries = tuple(entries)
        values = tuple(ad.Variable._from_record(value._record) if isinstance(value,ad.Variable) else value for _,value in entries)
        shapes = tuple(np.shape(ad.primal(value)) for value in values)
        keys = tuple(key for key, _ in entries)
        data = np.zeros(shape)
        for key, value in entries:
            data[key] = host(value).reshape(data[key].shape)
        def pullback(bar, needed):
            return tuple(bar[key].reshape(shape) if need else None
                         for key, shape, need in zip(keys, shapes, needed))
        packed = ad.operation(data, values, lambda bar: pullback(bar, [True]*len(values)), pullback)
        return self._capture(packed)

    def _ragged(self, layout, values):
        data = []
        for members, (pr, pc) in zip(layout.members, layout.shapes):
            entries = [((s, slice(0, int(layout.rows[p])), slice(0, int(layout.cols[p]))), values[p])
                       for s, p in enumerate(members)]
            data.append(self._pack((len(members), pr, pc), entries))
        return AdjointRagged(layout, data)

    def _overlaps(self, left, right):
        keys = tuple((int(p), int(q)) for p, q in zip(left, right))
        if self.LS.nbytes() > self.budget_bytes:
            raise MemoryError('Cached CCSD source overlaps exceed the explicit budget')
        if os.environ.get('DLPNO_CCSD_SHARED_OVERLAPS','0')=='1':
            from .ccsd_shared_overlap_gpu import build
            return build(self,keys)
        overlaps = _checkpointed_ccsd_overlaps_gpu(self.source, keys, prefetch=True)
        return self._ragged(self.LS, [overlaps[key] for key in keys])

    def _load_ints(self, ints, rc, rd):
        # The inherited loader owns a numerical copy used only to build
        # layouts/plans. Replace every equation coefficient with AD data.
        super()._load_ints(ints, rc, rd)
        source = self.source
        for target, origin, layout in (('K','K',self.LP), ('Lij','L_iajb',self.LP),
                                       ('Kmibj','K_mibj',self.LN), ('Jijmb','J_ijmb',self.LN)):
            setattr(self, target, self._ragged(layout, getattr(source, origin)))
        self.K3 = []
        for members, (pv, _) in zip(self.LP.members, self.LP.shapes):
            entries = [((s, slice(0,int(self.nv[p])), slice(0,int(self.nv[p])), slice(0,int(self.nv[p]))),
                        source.K_ivvv[p]) for s,p in enumerate(members)]
            self.K3.append(self._pack((len(members),pv,pv,pv),entries))
        items = [(int(p),k) for p in np.where(self.strong)[0] for k in range(len(source.lmos[p]))]
        for target, origin in (('Jikac','J_ikac'), ('Kiakc','K_iakc')):
            setattr(self, target, self._ragged(self.LJ,[getattr(source,origin)[p][k] for p,k in items]))

    def _load_df_group(self, mem, A, Lp, Vp, g):
        super()._load_df_group(mem,A,Lp,Vp,g)
        d, source = self.df[-1], self.source
        specifications = dict(Qma=((g,A,Lp,Vp),[]), Qab=((g,A,Vp,Vp),[]),
                              Qk=((g,2,A,Lp),[]), Qa=((g,2,A,Vp),[]),ev=((g,Vp),[]))
        for s,p in enumerate(mem):
            q = self.ji[p]
            na,nl,nv = source.Qma[p].shape
            specifications['Qma'][1].append(((s,slice(0,na),slice(0,nl),slice(0,nv)),source.Qma[p]))
            specifications['Qab'][1].append(((s,slice(0,na),slice(0,nv),slice(0,nv)),source.Qab[p]))
            for side,pair in enumerate((p,q)):
                specifications['Qk'][1].append(((s,side,slice(0,na),slice(0,nl)),source.iQk[pair]))
                specifications['Qa'][1].append(((s,side,slice(0,na),slice(0,nv)),source.iQa[pair]))
            specifications['ev'][1].append(((s,slice(0,nv)),source.e_pno[p]))
        for key,(shape,entries) in specifications.items():
            d[key] = self._pack(shape,entries)

    def init_state(self):
        super().init_state()
        f = self.source.orb.F_lmo
        self.F = self._capture(f[np.ix_(*(np.arange(n) for n in f.shape))])
        self.T = self._ragged(self.LP,self.source.T)
        self.T1 = self._ragged(self.LT1,self.source.T1)

    def pullback(self, seed, needed, *, consume=False):
        if self.released:
            raise RuntimeError('A released CCSD source graph cannot be reused')
        start = time.perf_counter()
        with cp.cuda.Device(self.device), cp.cuda.using_allocator(self.pool.malloc):
            selected = [v for v,flag in zip(self.device_inputs,needed) if flag]
            graph = ad.Linearization([self.output],selected,boundary=self.device_inputs)
            bars = iter(graph([cp.asarray(seed)],release_graph=consume,release_inputs=consume))
            result = tuple(cp.asnumpy(next(bars)) if flag else None for flag in needed)
            cp.cuda.get_current_stream().synchronize()
        self.statistics['reverse_seconds'] += time.perf_counter()-start
        self.statistics['reverse_calls'] += 1
        self.statistics['last_vjp'] = dict(graph.last_statistics)
        del graph, bars, selected
        if consume:
            self.release()
        return result

    def release(self):
        if self.released:
            return
        self.released = True
        distributed = getattr(self, '_distributed_df_plan', None)
        # Bound equation methods and plan closures can hold the object.
        # Remove all graph/plan references deterministically after use.
        keep = {'statistics','released','pool','device','shapes','offsets','npair','n'}
        for name in list(self.__dict__):
            if name not in keep:
                del self.__dict__[name]
        if distributed is not None:
            distributed.release()
        import gc
        gc.collect()
        with cp.cuda.Device(self.device):
            cp.cuda.get_current_stream().synchronize()
            self.pool.free_all_blocks()
            self.statistics.update(released=True, released_pool_live_bytes=self.pool.used_bytes())

    def bridge(self):
        inputs, data = tuple(self.host_inputs), self.packed
        def reverse(seed, needed):
            owner = self.owner()
            consume = owner is not None and getattr(owner,'_release_transposes',False)
            return self.pullback(seed,needed,consume=consume)
        output = ad.operation(data,inputs,lambda seed: reverse(seed,[True]*len(inputs)),reverse)
        outputs = [output[a:b].reshape(shape) for a,b,shape in zip(self.offsets[:-1],self.offsets[1:],self.shapes)]
        return outputs[:self.npair], outputs[self.npair:]


def residual(state, *, plan=None):
    budget = int(os.environ.get('DLPNO_CCSD_SOURCE_MEMORY_MIB','2048')) << 20
    if not (32 << 20) <= budget <= (64 << 30):
        raise ValueError('CCSD source memory budget must be between 32 and 65536 MiB')
    if plan is None:
        plan = SourcePlan(state,budget_bytes=budget)
    else:
        if plan.owner() is not state:
            raise ValueError('Prepared CCSD source graph belongs to another state')
        plan.finish()
    state.diagnostics['batched_ccsd_sources'] = plan.statistics
    if os.environ.get('DLPNO_CCSD_SOURCE_REUSE_TRANSPOSE','0') == '1':
        state._batched_ccsd_source_plan = plan
    return plan.bridge()


class SourceTranspose:
    """Exact independent-amplitude transpose of the resident source graph.

    Select amplitude buckets, then sum pair-transpose and diagonal maps
    on the GPU. Coefficient leaves are stops for this lambda action only;
    the final density sweep still differentiates every coefficient.
    """
    xp = cp

    def __init__(self, plan, variables, independent, residual):
        self.plan, self.released, self.calls = plan, False, 0
        start = time.perf_counter()
        with cp.cuda.Device(plan.device), cp.cuda.using_allocator(plan.pool.malloc):
            maps, size = _amplitude_maps(plan, variables.pairs, variables.diag)
            if size != independent.size:
                raise ValueError('Source transpose independent-amplitude map differs')
            self.single, self.double, self.indices, _ = maps
            positions = []
            for i in range(plan.n):
                positions.extend(range(int(plan.offsets[plan.npair+i]),int(plan.offsets[plan.npair+i+1])))
            for p in variables.pairs:
                n = plan.shapes[p][0]
                if p in variables.diag:
                    row,col = variables.diag[p]
                    positions.extend((plan.offsets[p]+row*n+col).tolist())
                else:
                    positions.extend(range(int(plan.offsets[p]),int(plan.offsets[p+1])))
            output = plan.output[cp.asarray(positions,dtype=cp.int64)]
            error = float(cp.max(cp.abs(output.v-cp.asarray(host(residual)))))
            if error > 2e-8:
                raise RuntimeError(f'Source transpose residual differs by {error:.3e}')
            self.singles_count = len(plan.T1.data)
            self.inputs = plan.T1.data+plan.T.data
            self.graph = ad.Linearization([output],self.inputs,boundary=plan.device_inputs)
            self.size = size
        self.statistics = dict(self.graph.statistics,operator_kind='resident_ccsd_source_transpose',backend='cupy',
            amplitude_only=True,coefficients_approximated=False,shared_source_graph=True,
            build_seconds=time.perf_counter()-start,residual_replay_error=error,amplitude_map_bytes=sum(x.nbytes for x in maps))
        self.last_statistics = {}
        self.replay = None
        self.replay_stream = cp.cuda.Stream(non_blocking=True)

    def _action(self, seeds):
        bars = self.graph(seeds)
        singles = cp.concatenate([x.ravel() for x in bars[:self.singles_count]])
        doubles = cp.concatenate([x.ravel() for x in bars[self.singles_count:]])
        result = cp.zeros(self.size)
        result[:len(self.single)] = singles[self.single]
        cupyx.scatter_add(result,self.indices,doubles[self.double])
        return [result]

    def __call__(self,seeds):
        if self.released or self.plan.released:
            raise RuntimeError('A released source transpose cannot be reused')
        start = time.perf_counter()
        plan = self.plan
        with cp.cuda.Device(plan.device), cp.cuda.using_allocator(plan.pool.malloc):
            from .cuda_graph_replay import Replay, enabled
            seed = cp.asarray(seeds[0])
            if enabled() and not hasattr(plan, '_distributed_df_plan'):
                cp.cuda.get_current_stream().synchronize()
                with self.replay_stream:
                    if self.replay is None:
                        self.replay = Replay(self._action, [seed])
                        self.statistics['cuda_graph'] = self.replay.statistics
                    result = self.replay([seed])[0]
                    self.replay_stream.synchronize()
            else:
                result = self._action([seed])[0]
                cp.cuda.get_current_stream().synchronize()
        self.calls += 1
        self.last_statistics = dict(self.graph.last_statistics,operator_kind='resident_ccsd_source_transpose',
            backend='cupy',calls=self.calls,seconds=time.perf_counter()-start)
        return [result]

    def release(self):
        # The source graph must survive until the coefficient pullback.
        if self.replay is not None:
            self.replay.release()
            self.replay = None
        self.graph = self.inputs = self.plan = None
        self.single = self.double = self.indices = None
        self.released = True
