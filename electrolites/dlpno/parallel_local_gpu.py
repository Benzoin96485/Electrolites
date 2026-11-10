"""Independent PNO/TNO and CCSD pair graphs with exact shared-source VJPs.

The parent alone reads orbital/RI providers and creates their AD leaves.
Workers receive selected host arrays and metadata, never live AD records.
Tasks retain all differentiable coefficients, including shared overlaps.
The host graph owns deterministic cotangent reduction. Amplitude solves
and global reference response still run on the primary device.
"""
from dataclasses import dataclass
import os
import time
from types import MethodType, SimpleNamespace
import numpy as np
from . import reverse as ad
from .deferred_reads import DeferredRead,EncodedRead
from .local_blocks import _key_token


@dataclass(frozen=True)
class Operand:
    index: int


@dataclass
class Selection:
    shape: tuple
    reads: dict

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, key):
        return self.reads[_key_token(key)]


class PairView(SimpleNamespace):
    def neighbors(self, i):
        return self.neighbor_rows[i]


class Capture:
    def __init__(self):
        self.inputs, self.identity = [], {}

    def value(self, value):
        if isinstance(value,DeferredRead):
            return value.encode(self.value)
        raw = ad.primal(value)
        if ad._xp([value]) is not np or np.asarray(raw).dtype != np.float64:
            raise ValueError('Parallel local graphs require host float64 operands')
        token = ('record', value._record) if isinstance(value, ad.Variable) else ('array', id(value))
        if token not in self.identity:
            self.identity[token] = len(self.inputs)
            self.inputs.append(ad.Variable._from_record(value._record) if isinstance(value, ad.Variable) else value)
        return Operand(self.identity[token])

    def tree(self, value):
        if isinstance(value, (list, tuple)):
            return type(value)(self.tree(x) for x in value)
        return self.value(value)

    def selection(self, source, keys):
        result = {}
        for key in keys:
            token = _key_token(key)
            if token not in result:
                result[token] = self.value(source[key])
        return Selection(source.shape, result)


def _decode(value, operands):
    if isinstance(value,EncodedRead):
        if hasattr(operands,"read"):return operands.read(value)
        return value.evaluate(tuple(_decode(x,operands) for x in value.sources))
    if isinstance(value, Operand):
        return operands[value.index]
    if isinstance(value, Selection):
        return Selection(value.shape, _decode(value.reads, operands))
    if isinstance(value, SimpleNamespace):
        return type(value)(**_decode(vars(value), operands))
    if isinstance(value, dict):
        return {key: _decode(item, operands) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_decode(item, operands) for item in value)
    return value


def _prepare_subspace(state, kind, item, capture, cut):
    pl, orb = state.pl, state.orb
    if kind == 'tno':
        labels = state.trip[item]
        pairs = tuple((a, b) for a, b in ((labels[0], labels[1]), (labels[1], labels[2]), (labels[0], labels[2])))
        ids = tuple(dict.fromkeys(int(pl.ij[key]) for key in pairs))
        paos = np.asarray(state.t_paos[item]).copy()
        ppaos = {p: np.asarray(state.pao[p]).copy() for p in ids}
        view = SimpleNamespace(opt=dict(state.opt), pl=PairView(ij={key: int(pl.ij[key]) for key in pairs}),
            trip={item: tuple(labels)}, t_paos={item: paos}, pao=ppaos,
            tno_scale={item: float(state.tno_scale[item])},
            X={p: capture.value(state.X[p]) for p in ids}, T={p: capture.value(state.T[p]) for p in ids})
        overlap_keys = [np.ix_(paos, paos)]+[np.ix_(ppaos[p], paos) for p in ids]
        view.orb = SimpleNamespace(S_pao=capture.selection(orb.S_pao, overlap_keys),
            F_pao=capture.selection(orb.F_pao, [np.ix_(paos, paos)]))
        return view
    i, j = pl.pairs[item]
    aux, paos = np.asarray(state.ribfs[item]).copy(), np.asarray(state.pao[item]).copy()
    return SimpleNamespace(opt=dict(state.opt), ncore_scale=state.ncore_scale,
        pl=PairView(pairs={item: (int(i), int(j))}), pao={item: paos}, ribfs={item: aux},
        qia=capture.selection(state.qia, [np.ix_(aux, [i], paos), np.ix_(aux, [j], paos)]),
        J2c=capture.selection(state.J2c, [np.ix_(aux, aux)]),
        orb=SimpleNamespace(S_pao=capture.selection(orb.S_pao, [np.ix_(paos, paos)]),
            F_pao=capture.selection(orb.F_pao, [np.ix_(paos, paos)]),
            F_lmo=capture.selection(orb.F_lmo, [(i, i), (j, j)])))


def _prepare_ccsd(state, p, capture, arguments):
    from .adjoint import _ccsd_pair_overlap_keys, _checkpointed_ccsd_overlap, _checkpointed_ccsd_overlaps_gpu
    pl = state.pl
    i, j = pl.pairs[p]
    lm, neighbors = np.asarray(state.lmos[p]).copy(), tuple(pl.neighbors(i))
    keys = _ccsd_pair_overlap_keys(state, p)
    if state.ccsd_adjoint_backend == 'cupy' and getattr(state, '_ccsd_overlap_prefetch_id', None) != id(state._S):
        requested = tuple(dict.fromkeys(key for pair in range(len(pl)) if state.strong[pair]
                                       for key in _ccsd_pair_overlap_keys(state, pair)))
        _checkpointed_ccsd_overlaps_gpu(state, requested, prefetch=True)
        state._ccsd_overlap_prefetch_id = id(state._S)
    overlap = {key: capture.value(_checkpointed_ccsd_overlap(state, *key)) for key in keys}
    ijkeys = [(k, l) for k in lm for l in lm] if i <= j else []
    ijkeys += [key for k in lm for key in ((k, i), (k, j), (i, k), (j, k))]
    ijkeys += [(i, k) for k in neighbors]
    ij = {key: int(pl.ij[key]) for key in ijkeys}
    tids = {p} | {ij[key] for key in ij if ij[key] >= 0}
    kids = {ij[k, l] for k in lm for l in lm if ij[k, l] >= 0} if i <= j else set()
    view = SimpleNamespace(pl=PairView(pairs={p: (i, j)}, ji={p: int(pl.ji[p])}, ij=ij,
        neighbor_rows={i: neighbors}), lmos={p: lm}, overlap=overlap,
        T={k: capture.value(state.T[k]) for k in sorted(tids)}, K={k: capture.value(state.K[k]) for k in sorted(kids)})
    for name in ('J_ikac', 'K_iakc'):
        setattr(view, name, {p: capture.tree(getattr(state, name)[p])})
    if i <= j:
        for name in ('Qab', 'T_n', 'Qma', 'Fab'):
            setattr(view, name, {p: capture.tree(getattr(state, name)[p])})
        view.iQa_t1 = {q: capture.value(state.iQa_t1[q]) for q in dict.fromkeys((p, int(pl.ji[p])))}
    beta, gamma, delta, fdd = arguments
    args = ({p: capture.value(beta[p])} if i <= j else {},
            {ij[k, i]: capture.value(gamma[ij[k, i]]) for k in lm},
            {ij[i, k]: capture.value(delta[ij[i, k]]) for k in lm},
            capture.selection(fdd, [(k, j) for k in neighbors]))
    return view, args


def prepare_local_task(state, kind, items, *, cut=None, arguments=None, backend='cupy'):
    if kind not in ('sc_mp2', 'pno', 'tno', 'ccsd_pair') or backend not in ('numpy', 'cupy'):
        raise ValueError('Unknown local graph kind or backend')
    capture, views = Capture(), []
    for item in items:
        item = int(item)
        view = (_prepare_ccsd(state, item, capture, arguments) if kind == 'ccsd_pair'
                else _prepare_subspace(state, kind, item, capture, cut))
        views.append((item, view))
    spec = dict(task_kind='local_graph', local_kind=kind, backend=backend, views=views, cut=cut)
    return spec, tuple(capture.inputs)


def evaluate_local(spec, values, seed, needed):
    import cupy as cp
    from .adjoint import _names
    from .ref_rhf import RefDLPNO
    from .reference_cpu import rebind
    xp = cp if spec['backend'] == 'cupy' else np
    variables = [ad.Variable(xp.asarray(value), tracked=seed is not None and flag)
                 for value, flag in zip(values, needed)]
    names = dict(_names, np=ad.Numpy(xp))
    kind = spec['local_kind']
    method = {'sc_mp2': '_sc_mp2_pair_energy', 'pno': '_lmp2_pno_pair',
              'tno': '_tno_triplet', 'ccsd_pair': '_R2_pair'}[kind]
    evaluate = rebind(getattr(RefDLPNO, method), **names)
    outputs = []
    for item, encoded in spec['views']:
        view = _decode(encoded, variables)
        if kind == 'ccsd_pair':
            view, args = view
            view.S = lambda p, q, table=view.overlap: table[p, q]
            result = evaluate(view, item, *args)
        elif kind == 'tno':
            result = evaluate(view, item, spec['cut'])
        else:
            view._K_pao = MethodType(rebind(RefDLPNO._K_pao, **names), view)
            try:
                result = evaluate(view, item)
            finally:
                del view._K_pao  # Break the temporary bound-method reference cycle.
            if kind == 'sc_mp2':
                result = (result,)
        outputs.extend(result)
    if seed is None:
        packed = xp.concatenate([ad.primal(x).ravel() for x in outputs])
        return (cp.asnumpy(packed) if xp is cp else packed), tuple(x.shape for x in outputs)
    offsets = np.cumsum([0]+[x.size for x in outputs])
    seeds = [xp.asarray(seed[a:b]).reshape(x.shape) for x, a, b in zip(outputs, offsets[:-1], offsets[1:])]
    selected = [x for x, flag in zip(variables, needed) if flag]
    bars = iter(ad.Linearization(outputs, selected, boundary=variables)(seeds, release_graph=True, release_inputs=True))
    result = []
    for flag in needed:
        bar = next(bars) if flag else None
        result.append(cp.asnumpy(bar) if bar is not None and xp is cp else bar)
    return tuple(result)


def local_window(state, kind, groups, *, devices, cut=None, arguments=None, backend='cupy', pool=None):
    from .parallel_sources_gpu import task_window
    start = time.perf_counter()
    prepared = [prepare_local_task(state, kind, group, cut=cut, arguments=arguments, backend=backend) for group in groups]
    return task_window(prepared, devices=devices, kind='multi_gpu_'+kind, pool=pool,
                       prepare_start=start, extra_info=dict(backend=backend, local_blocks=sum(map(len, groups))))


def _windows(state, kind, items, *, cut=None, arguments=None):
    from .parallel_sources_gpu import configured_devices
    devices = configured_devices()
    batch = int(os.environ.get('DLPNO_GRADIENT_LOCAL_BATCH', '8'))
    window = int(os.environ.get('DLPNO_GRADIENT_WINDOW', '2'))
    backend = os.environ.get('DLPNO_GRADIENT_LOCAL_BACKEND', 'cupy')
    if not 1 <= batch <= 32 or not 1 <= window <= 8:
        raise ValueError('Local graph batch/window outside bounded limits')
    groups = [tuple(items[i:i+batch]) for i in range(0, len(items), batch)]
    stride = dict(sc_mp2=1, pno=5, tno=2, ccsd_pair=2)[kind]
    for start in range(0, len(groups), window):
        current = groups[start:start+window]
        outputs, info = local_window(state, kind, current, devices=devices, cut=cut, arguments=arguments, backend=backend)
        state.diagnostics.setdefault('parallel_local_graphs', []).append(info)
        for pos, item in enumerate(item for group in current for item in group):
            yield item, outputs[stride*pos:stride*(pos+1)]


def sc_mp2_pair_energies(self):
    result = self.numeric.zeros(len(self.pl))
    items = [p for p, (i, j) in enumerate(self.pl.pairs) if i <= j]
    for p, (energy,) in _windows(self, 'sc_mp2', items):
        result[p] = energy
        result[self.pl.ji[p]] = energy
    return result


def lmp2_pnos(self):
    count = len(self.pl)
    self.X, self.e_pno, self.K, self.T = ([None]*count for _ in range(4))
    self.de_pno = self.numeric.zeros(count)
    items = [p for p, (i, j) in enumerate(self.pl.pairs) if i <= j]
    for p, (x, e, k, t, correction) in _windows(self, 'pno', items):
        self.de_pno[p] = correction
        self._store_pair(p, x, e, k, t)
    self.de_pno_total = np.sum(self.de_pno)


def tno_transform(self, cut):
    self.X_tno, self.e_tno = [], []
    for _, (x, e) in _windows(self, 'tno', list(range(len(self.trip))), cut=cut):
        self.X_tno.append(x)
        self.e_tno.append(e)


def r2(self):
    beta, gamma, delta, fdd = self._beta(), self._gamma(), self._delta(), self._Fkj_dd()
    result = [self.numeric.zeros(t.shape) for t in self.T]
    nonsymmetric = [self.numeric.zeros(t.shape) for t in self.T]
    items = [p for p in range(len(self.pl)) if self.strong[p]]
    for p, (tot, ns) in _windows(self, 'ccsd_pair', items, arguments=(beta, gamma, delta, fdd)):
        i, j = self.pl.pairs[p]
        q = self.pl.ji[p]
        if i <= j:
            result[p] += tot
            if i != j:
                result[q] += tot.T
        nonsymmetric[p] += ns
    for p in range(len(self.pl)):
        if self.strong[p]:
            result[p] += nonsymmetric[p]+nonsymmetric[self.pl.ji[p]].T
    return result


def configure(state, *, checkpoint_pnos, checkpoint_ccsd):
    from .parallel_sources_gpu import configured_devices
    regions = set(filter(None, os.environ.get('DLPNO_GRADIENT_REGIONS', 'triples').split(',')))
    if regions-{'triples', 'pno', 'tno', 'ccsd'}:
        raise ValueError('Unknown gradient parallel region')
    if not configured_devices() or not regions-{'triples'}:
        return
    if state.xp is not np:
        raise ValueError('Parallel local graphs require a host outer graph')
    if regions & {'pno', 'tno'} and not checkpoint_pnos:
        raise ValueError('Parallel subspaces require checkpoint_pnos')
    if 'ccsd' in regions and not checkpoint_ccsd:
        raise ValueError('Parallel CCSD sources require checkpoint_ccsd')
    bindings = {}
    if 'pno' in regions:
        bindings.update(sc_mp2_pair_energies=sc_mp2_pair_energies, lmp2_pnos=lmp2_pnos)
    if 'tno' in regions:
        bindings['tno_transform'] = tno_transform
    if 'ccsd' in regions:
        bindings['_R2'] = r2
    for name, function in bindings.items():
        setattr(state, name, MethodType(function, state))
