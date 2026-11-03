"""Batched, device-resident closed-shell DLPNO-CCSD iterations.

The equations are those of ``ref_rhf.RefDLPNO`` (Jiang et al. 2024, the
T1-dressed formulation Psi4 implements), term for term.  What changes is the
execution: every per-pair or per-(pair, LMO) loop of the reference becomes a
static list of terms, grouped once by the padded shapes of its operands
(``ragged.ChainPlan``) and replayed as strided-batched GEMMs every
iteration.  The PNO overlap matrices every term needs are formed once, as a
batched GEMM over pair couples, and kept on the device.
"""
import time

import numpy as np
import cupy as cp
import cupyx

import os
from electrolites.dlpno.ragged import Layout, Ragged, ChainPlan as _ChainPlan, DotPlan as _DotPlan, op
from electrolites.dlpno.vbatch import VChainPlan, VDotPlan

#: DLPNO_VBATCH=0 runs the shape-grouped cupy batches instead of the
#: variable-size kernels (kept for A/B and as a cross-check)
if os.environ.get("DLPNO_VBATCH", "1") == "1":
    ChainPlan, DotPlan = VChainPlan, VDotPlan
else:
    ChainPlan, DotPlan = _ChainPlan, _DotPlan



# ---- batched contractions as GEMMs (CuPy's einsum takes a slower path) ----
def _mv(M, v):
    """'gA...ab,gb->gA...a': M (g, ..., a, b) times per-g vector v (g, b)."""
    sh = (v.shape[0],) + (1,) * (M.ndim - 3) + (v.shape[1], 1)
    return cp.matmul(M, v.reshape(sh))[..., 0]


def _dot2(M, T):
    """'gAme,gme->gA'."""
    g, A = M.shape[:2]
    return cp.matmul(M.reshape(g, A, -1), T.reshape(g, -1, 1))[..., 0]


def _wsum(w, M):
    """'gA,gA...->g...'."""
    g, A = M.shape[:2]
    return cp.matmul(w[:, None, :], M.reshape(g, A, -1)).reshape((g,) + M.shape[2:])


def _rnorm(*Rs):
    """residual measures of the residual blocks ``Rs``: (largest per-pair
    RMS -- Psi4's measure, largest absolute element -- ORCA's "max.
    residuum")."""
    rms = max(float(R.rms_per_item().max()) for R in Rs)
    mx = [cp.abs(d).max() for R in Rs for d in R.data if d.size]
    return rms, (float(cp.stack(mx).max()) if mx else 0.)


def _converged(de, rn, opt):
    """CC convergence: |dE| < E_CONVERGENCE and the residual criterion of
    R_NORM -- 'rms' (Psi4: per-pair RMS < R_CONVERGENCE), 'max' (ORCA:
    largest element < R_CONVERGENCE_MAX) or 'both' (default, the tighter
    of the two: both must hold)."""
    mode = opt.get('R_NORM', 'both')
    ok_rms = rn[0] < opt['R_CONVERGENCE']
    ok_max = rn[1] < opt.get('R_CONVERGENCE_MAX', 1e-5)
    ok = {'rms': ok_rms, 'max': ok_max, 'both': ok_rms and ok_max}[mode]
    return abs(de) < opt['E_CONVERGENCE'] and ok


_PAD = cp.RawKernel(r'''
extern "C" __global__ void pad_members(const double* src, const long long* off, const int* dims,
                                       double* out, int X, int Y, int Z) {
    // member m = blockIdx.y: its (x, y, z) block at src + off[m] into out[m, :x, :y, :z]
    int m = blockIdx.y;
    int x = dims[3 * m], y = dims[3 * m + 1], z = dims[3 * m + 2];
    long long n = (long long)x * y * z;
    const double* s = src + off[m];
    double* o = out + (long long)m * X * Y * Z;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
         t += (long long)gridDim.x * blockDim.x) {
        int c = t % z, b = (t / z) % y, a = t / ((long long)z * y);
        o[((long long)a * Y + b) * Z + c] = s[t];
    }
}
''', 'pad_members')


class _Packed:
    """host-resident group tensor without padding: the members' blocks back to
    back in pinned memory; ``device()`` uploads them and pads to the group's
    shape (members, [sub,] A, R, C) on the current device."""

    def __init__(self, blocks, shape, sub=1, store=None):
        from electrolites.dlpno.devmem import pinned_empty, as_host
        dims = np.array([(b.shape + (1,) * (3 - b.ndim)) for b in blocks], np.int32).reshape(-1, 3)
        sz = dims.astype(np.int64).prod(axis=1)
        self.off = np.concatenate([[0], np.cumsum(sz)[:-1]]).astype(np.int64)
        # store: the packed buffer in a file on the scratch file system, read
        # back at every use (host memory too small for the CC's DF tensors)
        buf = np.empty(int(sz.sum())) if store is not None else pinned_empty(int(sz.sum()))
        for b, o, n in zip(blocks, self.off, sz):
            buf[o:o + n] = as_host(b).ravel()
        self.ref = store.put(buf) if store is not None else None
        self.buf = buf if store is None else None
        nbytes = buf.nbytes
        del buf
        self.dims = dims
        self.shape = (len(blocks) // sub,) + ((sub,) if sub > 1 else ()) + tuple(shape)
        self.inner = (tuple(shape) + (1,))[:3]
        self.nbytes = nbytes

    def device(self):
        X, Y, Z = self.inner
        m = len(self.dims)
        out = cp.zeros(self.shape)
        nmax = int(self.dims.astype(np.int64).prod(axis=1).max()) if m else 0
        if nmax:
            if self.ref is None:
                src = cp.asarray(self.buf)
            else:
                pf = getattr(self, 'prefetch', None)
                src = cp.asarray(pf.get(self.ref) if pf is not None else self.ref.load())
            _PAD((min(1024, -(-nmax // 256)), m), (256,),
                 (src, cp.asarray(self.off), cp.asarray(self.dims.ravel()), out,
                  np.int32(X), np.int32(Y), np.int32(Z)))
        return out


def _malloc_trim():
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except (OSError, AttributeError):
        pass


_SWS = {}      # device -> overlap workspace of the chunked plans (shared)
_KEPT = {}     # device -> [(chunk, budget, bytes)] of kept regenerated overlaps
_NOKEEP = set()  # devices that stopped keeping after an eviction


def _evict(dev):
    """drop the kept overlaps of a device (regenerated from now on)."""
    for ch, bud, nb in _KEPT.pop(dev, []):
        ch.pop('S', None)
        if bud is not None:
            with bud.lock:
                bud.left[dev] = bud.left.get(dev, 0) + nb
    _NOKEEP.add(dev)
    _SWS.pop(dev, None)
    cp.get_default_memory_pool().free_all_blocks()


class _ChunkedPlan:
    """A chain (or dot) plan whose PNO-overlap operands (layout ``LS``) are
    generated on the device per chunk of terms (``ovgen``) instead of being
    stored for every couple.

    With several GPUs the chunks are spread over the devices: each device
    keeps the overlaps of its chunks (as long as its share of ``budget``
    lasts; beyond that they are regenerated every call), runs its chunks on
    replicas of the dynamic operands and its partial output is added into
    the home device's."""

    MAX_BYTES = 1 << 29

    def __init__(self, gens, LS, cp_, cq_, nv, out_layout, out_idx, operands, out_row=None, left=None,
                 devs=None, budget=None, store=None):
        self.store = store
        if store is None and 'MAX_BYTES' not in self.__dict__ and type(self).MAX_BYTES == (1 << 29):
            # large systems: fewer, larger chunks (each builds a plan and a
            # generator chunk); 6% of the free device memory, 0.5 .. 4 GB
            fr = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
            self.MAX_BYTES = int(min(4 << 30, max(1 << 29, 0.06 * fr)))
        n = len(out_idx) if left is None else len(left[0].idx)
        self.n = self.nterms = n
        self.gens = gens if callable(gens) else (lambda d, g=gens: g)
        self.home = cp.cuda.Device().id
        devs = devs or [self.home]
        self.budget = budget
        self.chunks = []
        if n == 0:
            self.devs = [self.home]
            return
        s_ops = [i for i, o in enumerate(operands) if o.layout is LS]
        self.s_ops = s_ops
        cid = np.stack([np.asarray(operands[i].idx, np.int64) for i in s_ops], 1)
        byt = 8 * nv[cp_] * nv[cq_]
        order = (np.argsort(np.asarray(out_idx), kind='stable') if out_idx is not None else np.arange(n))
        # greedy chunks of terms whose distinct overlaps stay within MAX_BYTES
        # (vectorised: first uses counted within a window from the chunk start)
        bounds, start = [], 0
        k = cid.shape[1]
        W0 = 4096
        while start < n:
            W = min(n - start, W0)
            ids = cid[order[start:start + W]].ravel()
            _, first = np.unique(ids, return_index=True)
            add = np.zeros(W * k)
            add[first] = byt[ids[first]]
            cum = np.cumsum(add.reshape(W, k).sum(axis=1))
            over = np.flatnonzero(cum > self.MAX_BYTES)
            if not len(over) and W < n - start:
                W0 *= 2
                continue
            L = max(1, int(over[0])) if len(over) else W
            bounds.append((start, start + L, float(cum[L - 1])))
            start += L
        load = np.zeros(len(devs))
        for a, b, nb in bounds:
            r = int(np.argmin(load))
            load[r] += nb + 8 * (b - a)
            with cp.cuda.Device(devs[r]):
                self._chunk(order[a:b], cid, nv, cp_, cq_, operands, out_layout, out_idx, out_row, left)
                self.chunks[-1]['dev'] = devs[r]
        self.devs = sorted({ch['dev'] for ch in self.chunks})

    def _chunk(self, sel, cid, nv, cp_, cq_, operands, out_layout, out_idx, out_row, left):
        s_ops = self.s_ops
        uni = np.unique(cid[sel])
        LSc = Layout(nv[cp_[uni]], nv[cq_[uni]])
        if self.store is not None:
            # overlaps computed once into a store spread over the GPUs: this
            # chunk gathers its blocks by pointer (peer reads over NVLink)
            gen, preps = None, self.store.tables(LSc, uni)
        else:
            gen = self.gens(cp.cuda.Device().id)
            preps = gen.prepare(cp_[uni], cq_[uni], out_layout=LSc, out_idx=np.arange(len(uni)))
        ops = []
        for i, o in enumerate(operands):
            row = None if o.row is None else o.row[sel]
            if i in s_ops:
                ops.append(op(LSc, np.searchsorted(uni, o.idx[sel]), o.trans, row))
            else:
                ops.append(op(o.layout, o.idx[sel], o.trans, row))
        if left is None:
            plan = ChainPlan(out_layout, np.asarray(out_idx)[sel], ops,
                             out_row=None if out_row is None else np.asarray(out_row)[sel])
        else:
            lo = left[0]
            plan = DotPlan([op(lo.layout, lo.idx[sel], lo.trans)], ops)
        self.chunks.append(dict(sel=cp.asarray(sel), sel_h=sel, LS=LSc, preps=preps, plan=plan, gen=gen))

    def _S(self, ch):
        S = ch.get('S')
        if S is not None:
            return S
        if self.store is not None:
            return self.store.gather(ch['LS'], ch['preps'])
        nb = ch['LS'].nbytes()
        keep = False
        if self.budget is not None:
            # kept only while a quarter of the device stays free: the budget was
            # set at the build, before the replicas and the DF groups' temporaries
            # (memory not yet taken by the pool: cached pool blocks are
            # fragmented and may not serve the temporaries)
            fr, tot = cp.cuda.runtime.memGetInfo()
            keep = ch['dev'] not in _NOKEEP and \
                fr - nb > float(os.environ.get('DLPNO_S_KEEP_FREE', '0.4')) * tot \
                and self.budget.take(ch['dev'], nb)
        if keep:
            S = Ragged(ch['LS'])
        else:
            # regenerated every use: one zeroed workspace per device, shared by
            # all the plans (they run one after another on a device), no
            # allocation per chunk and iteration
            ws_ = _SWS
            n = ch['LS'].nbytes() // 8
            w = ws_.get(ch['dev'])
            if w is None or w.size < n:
                ws_.pop(ch['dev'], None)
                w = None
                cp.get_default_memory_pool().free_all_blocks()
                w = ws_[ch['dev']] = cp.empty(max(n, 1))
            w[:n].fill(0.)
            data, off = [], 0
            for m, (r, c) in zip(ch['LS'].members, ch['LS'].shapes):
                k = len(m) * r * c
                data.append(w[off:off + k].reshape(len(m), r, c))
                off += k
            S = Ragged(ch['LS'], data)
        for P in ch['preps']:
            ch['gen'].run(P, S)
        if keep:
            ch['S'] = S                   # kept: this chunk's overlaps are not regenerated
            _KEPT.setdefault(ch['dev'], []).append((ch, self.budget, nb))
        return S

    def _run_chunks(self, chunks, out, rags, w, alpha, dot):
        res = []
        for ch in chunks:
            # the kept overlaps must not crowd out the temporaries (they grew
            # after the build): when the plan's largest temporary set would
            # not fit, this device drops them and regenerates from now on
            need = 8. * max((sum(c[3]) for c in getattr(ch['plan'], 'chunks', [])), default=0)
            fr = cp.cuda.runtime.memGetInfo()[0]
            if _KEPT.get(ch['dev']) and fr < 1.5 * need + 2e9:
                _evict(ch['dev'])
            try:
                S = self._S(ch)
            except cp.cuda.memory.OutOfMemoryError:
                if not _KEPT.get(ch['dev']):
                    raise
                _evict(ch['dev'])          # nothing written yet: retry
                S = self._S(ch)
            r = [S if i in self.s_ops else x for i, x in enumerate(rags)]
            if dot:
                res.append((ch, ch['plan'].run(out, r)))
            else:
                ch['plan'].run(out, r, w=None if w is None else w[ch['sel']], alpha=alpha)
        return res

    def run(self, out, rags, w=None, alpha=1.):
        dot = isinstance(out, list)            # dot plan: run(lrags, rrags)
        if self.devs == [self.home] or not self.chunks:
            if dot:
                outv = cp.zeros(self.n)
                for ch, v in self._run_chunks(self.chunks, out, rags, None, alpha, True):
                    outv[ch['sel']] = v
                return outv
            self._run_chunks(self.chunks, out, rags, w, alpha, False)
            return out
        from electrolites.dlpno import mgpu
        parts = {}

        def rep(x):
            if isinstance(x, Ragged):
                if getattr(x, '_static', False):
                    # fixed integrals: replicated once per device
                    c = x.__dict__.setdefault('_replicas', {})
                    d = cp.cuda.Device().id
                    if d not in c:
                        c[d] = Ragged(x.layout, mgpu.replicate(x.data))
                    return c[d]
                return Ragged(x.layout, mgpu.replicate(x.data))
            return x

        def work(dv):
            chs = [ch for ch in self.chunks if ch['dev'] == dv]
            if dv == self.home:
                parts[dv] = (self._run_chunks(chs, out, rags, w, alpha, dot), None)
                return
            rr = [None if i in self.s_ops else rep(x) for i, x in enumerate(rags)]
            if dot:
                parts[dv] = (self._run_chunks(chs, [rep(x) for x in out], rr, None, alpha, True), None)
                return
            o = Ragged(out.layout)
            parts[dv] = (self._run_chunks(chs, o, rr, None if w is None else mgpu.replicate(w), alpha, False), o)
        mgpu.run_per_device(work, self.devs)
        if dot:
            outv = cp.zeros(self.n)
            for res, _ in parts.values():
                for ch, v in res:
                    outv[cp.asarray(ch['sel_h'])] = cp.asarray(v)
            return outv
        for dv, (_, o) in parts.items():
            if o is not None:
                for g in range(len(out.data)):
                    if out.data[g].size:
                        out.data[g] += cp.asarray(o.data[g])
        return out



class _SStore:
    """PNO overlaps of all registered couples, computed once and spread over
    the GPUs (contiguous couple ranges); ``tables`` / ``gather`` assemble a
    chunk's overlaps from (possibly remote) blocks by pointer."""

    def __init__(self, S_pao, pao, X, cp_, cq_, nv, devs, ne=0):
        from electrolites.dlpno.ovgen import OverlapGen
        n = len(cp_)
        byt = 8. * nv[cp_] * nv[cq_]
        cum = np.cumsum(byt)
        cuts = np.searchsorted(cum, cum[-1] * np.arange(1, len(devs)) / len(devs)) if n else []
        bounds = np.concatenate([[0], cuts, [n]]).astype(np.int64)
        self.ptr = np.zeros(n, np.int64)
        self.ld = np.zeros(n, np.int32)
        self.nr = nv[cp_].astype(np.int32)
        self.nc = nv[cq_].astype(np.int32)
        self.rags = []
        rng = {d: (a, b) for d, a, b in zip(devs, bounds[:-1], bounds[1:]) if b > a}

        def build(d):                      # every device generates its range concurrently
            a, b = rng[d]
            gen = OverlapGen(cp.asarray(S_pao), pao, X, ne=ne, max_bytes=1 << 31)
            idx = np.arange(a, b)
            lay = Layout(nv[cp_[idx]], nv[cq_[idx]])
            R = Ragged(lay)
            for P_ in gen.prepare(cp_[idx], cq_[idx], out_layout=lay, out_idx=np.arange(len(idx))):
                gen.run(P_, R)
            del gen
            shp = np.array(lay.shapes, np.int64).reshape(-1, 2)
            base = np.array([x.data.ptr for x in R.data], np.int64)
            self.ptr[idx] = base[lay.group] + 8 * lay.slot * shp[lay.group, 0] * shp[lay.group, 1]
            self.ld[idx] = shp[lay.group, 1]
            self.rags.append(R)
        from electrolites.dlpno import mgpu
        mgpu.run_per_device(build, sorted(rng))
        self.nbytes = float(np.sum(byt))

    def tables(self, LSc, uni):
        """per layout group of the chunk: device pointer tables of its items."""
        out = []
        for g, m in enumerate(LSc.members):
            if not len(m):
                out.append(None)
                continue
            c = uni[m]
            out.append((cp.asarray(self.ptr[c]), cp.asarray(self.ld[c]), cp.asarray(self.nr[c]),
                        cp.asarray(self.nc[c])) + tuple(LSc.shapes[g]))
        return out

    @staticmethod
    def gather(LSc, tabs):
        from electrolites.dlpno.gather import gather2d
        S = Ragged(LSc)
        for g, t in enumerate(tabs):
            if t is not None:
                ptr, ld, nr, nc, pr, pc = t
                gather2d(ptr, ld, nr, nc, pr, pc, out=S.data[g])
        return S

class _SBudget:
    """bytes of generated overlaps each device may keep (thread safe)."""

    def __init__(self, per_dev):
        import threading
        self.left = dict(per_dev)
        self.lock = threading.Lock()

    def take(self, dev, nb):
        with self.lock:
            if self.left.get(dev, 0) >= nb:
                self.left[dev] -= nb
                return True
            return False


class _Couples:
    """Registry of the PNO overlap blocks S(p, q) the equations need."""

    def __init__(self):
        self.keys = {}
        self.p = []
        self.q = []

    def __call__(self, p, q):
        k = (p, q)
        c = self.keys.get(k)
        if c is None:
            c = self.keys[k] = len(self.p)
            self.p.append(p)
            self.q.append(q)
        return c

    def many(self, p, q):
        return np.array([self(int(a), int(b)) for a, b in zip(p, q)], dtype=np.int64)


class _Terms:
    """Accumulates index columns of one term list."""

    def __init__(self, *names):
        self.cols = {n: [] for n in names}

    def add(self, **kw):
        for k, v in kw.items():
            self.cols[k].append(v)

    def __getitem__(self, k):
        return np.asarray(self.cols[k], dtype=np.int64)

    def __len__(self):
        return len(next(iter(self.cols.values())))


def _partner_slots(layout, partner):
    """per group: slots of each member's partner item (must share the group)."""
    out = []
    for g, m in enumerate(layout.members):
        pp = partner[m]
        assert np.all(layout.group[pp] == g)
        out.append(cp.asarray(layout.slot[pp]))
    return out



class _VCouples:
    """Registry of PNO-overlap couples (p, q), vectorised: ``C(p_array,
    q_array)`` returns provisional codes; ``finalize`` renumbers every
    registered code list to dense couple indices (couples in first-use
    order is not needed: sorted by code)."""

    def __init__(self, npair):
        self.npair = npair
        self._codes = []

    def __call__(self, p, q):
        c = np.asarray(p, np.int64) * self.npair + np.asarray(q, np.int64)
        return _Code(c)

    def finalize(self, terms):
        cols = [(t, k) for t in terms for k, v in t.cols.items() if isinstance(v, _Code)]
        allc = np.concatenate([t.cols[k].c for t, k in cols]) if cols else np.zeros(0, np.int64)
        uni = np.unique(allc)
        for t, k in cols:
            t.cols[k] = np.searchsorted(uni, t.cols[k].c).astype(np.int64)
        self.p = (uni // self.npair).tolist()
        self.q = (uni % self.npair).tolist()


class _Code:
    __slots__ = ('c',)

    def __init__(self, c):
        self.c = c


class _VTerms:
    """columns of one term list (numpy arrays)."""

    def __init__(self, **cols):
        self.cols = dict(cols)

    def __getitem__(self, k):
        v = self.cols[k]
        return v if isinstance(v, np.ndarray) else np.asarray(v, dtype=np.int64)

    def __len__(self):
        v = next(iter(self.cols.values()))
        return len(v.c) if isinstance(v, _Code) else len(v)


class CCSDGPU:
    """Device data and plans for one set of CC PNOs (built from a RefDLPNO
    object after ``recompute_pnos`` and ``pno_integrals``)."""

    def __init__(self, ref, ints):
        self.ref = ref
        self.t_build = {}
        t0 = time.perf_counter()
        pl = ref.pl
        n = ref.n
        npair = len(pl)
        self.n, self.npair = n, npair
        nv = np.array([x.shape[1] for x in ref.X], dtype=np.int64)
        nl = np.array([len(l) for l in ref.lmos], dtype=np.int64)
        self.nv, self.nl = nv, nl
        ij = pl.ij
        self.ij = ij
        ji = pl.ji
        self.ji = ji
        dii = ij[np.arange(n), np.arange(n)]
        self.dii = dii
        pi = np.array([a for a, b in pl.pairs], dtype=np.int64)
        pj = np.array([b for a, b in pl.pairs], dtype=np.int64)
        self.pi, self.pj = pi, pj
        strong = np.asarray(ref.strong, dtype=bool)
        self.strong = strong
        canon = pi <= pj
        self.canon = canon

        LP = self.LP = Layout(nv, nv)
        LV = self.LV = Layout(nv, 1)
        LN = self.LN = Layout(nl, nv)
        LL = self.LL = Layout(nl, nl)
        LT1 = self.LT1 = Layout(nv[dii], 1)
        L11 = self.L11 = Layout(np.ones(npair, np.int64), 1)
        self.pslot = {id(L): _partner_slots(L, ji) for L in (LP, LV, LN, LL)}

        pos = ref.lmo_pos
        lmos = ref.lmos

        # ---- term lists (vectorised over pairs x domain LMOs) -------------------
        C = self.C = _VCouples(npair)
        cnt = nl
        Pl = np.repeat(np.arange(npair), cnt)                     # pair of each (p, l)
        Ll = np.concatenate([np.asarray(l, np.int64) for l in lmos]) if npair else np.zeros(0, np.int64)
        Rl = np.concatenate([np.arange(c) for c in cnt]) if npair else np.zeros(0, np.int64)
        Il, Jl = pi[Pl], pj[Pl]
        T = _VTerms
        # T_n and S(p,ii) T1[i]
        tn = T(p=Pl, row=Rl, s=C(Pl, dii[Ll]), n=Ll)
        allp = np.arange(npair)
        ti = T(p=allp, s=C(allp, dii[pi]), i=pi)
        tj = T(p=allp, s=C(allp, dii[pj]), j=pj)
        # Fkc (all pairs)
        ik_ = ij[Il, Ll]
        fkc = T(p=Pl, s1=C(Pl, ik_), ik=ik_, s2=C(ik_, dii[Ll]), k=Ll)
        # gamma (all pairs p = (k, i)): k = pi, i = pj
        kl_, ll_, li_ = ij[Il, Ll], dii[Ll], ij[Ll, Jl]
        g3 = T(p=Pl, s1=C(Pl, ll_), l=Ll, i=Jl, s2=C(kl_, dii[Jl]), kl=kl_, s3=C(Pl, kl_))
        g4 = T(p=Pl, s1=C(Pl, li_), li=li_, s2=C(li_, kl_), kl=kl_, s3=C(kl_, Pl))
        # delta (all pairs p = (i, k))
        lk_, il_ = ij[Ll, Jl], ij[Il, Ll]
        d3 = T(p=Pl, s1=C(Pl, ll_), l=Ll, i=Il, s2=C(lk_, dii[Il]), lk=lk_, s3=C(lk_, Pl))
        d4 = T(p=Pl, s1=C(Pl, il_), il=il_, s2=C(il_, lk_), lk=lk_, s3=C(lk_, Pl))
        # Fkj double tilde
        li2, lj2 = ij[Ll, Il], ij[Ll, Jl]
        m_ = (li2 >= 0) & (lj2 >= 0)
        fdd = T(p=Pl[m_], li=li2[m_], s1=C(li2[m_], lj2[m_]), lj=lj2[m_], s2=C(lj2[m_], li2[m_]),
                flat=Il[m_] * n + Jl[m_])
        # R1
        qq = ji[allp]
        r1a = T(i=pi, s=C(qq, dii[pi]), q=qq)
        klb = ij[Jl, Ll]
        r1b = T(i=Il, s=C(dii[Il], dii[Ll]), l=Ll, kl=klb, q=ji[Pl], sa=C(klb, ji[Pl]), sb=C(ji[Pl], klb))
        r1c = T(i=pi, s=C(allp, dii[pi]), p=allp, q=qq)
        r1d = T(i=Ll, s=C(Pl, dii[Ll]), p=Pl, row=Rl)
        # R2: B and Fbc over (k, l) in the pair's LMO domain (strong canonical pairs)
        sc_ = np.where(strong & canon)[0]
        pa, ka, la, fa = [], [], [], []
        for p in sc_:
            lm = np.asarray(lmos[p], np.int64)
            L = len(lm)
            kk, ll2 = np.meshgrid(lm, lm, indexing='ij')
            pa.append(np.full(L * L, p))
            ka.append(kk.ravel())
            la.append(ll2.ravel())
            fa.append(np.arange(L * L))
        if pa:
            pa, ka, la, fa = (np.concatenate(x) for x in (pa, ka, la, fa))
        else:
            pa = ka = la = fa = np.zeros(0, np.int64)
        klr = ij[ka, la]
        m_ = klr >= 0
        rb = T(p=pa[m_], s=C(klr[m_], pa[m_]), kl=klr[m_], bflat=fa[m_])
        ms = strong[Pl]
        Ps, Ks, Is, Js = Pl[ms], Ll[ms], Il[ms], Jl[ms]
        ki_, kj_, ik2, jk_ = ij[Ks, Is], ij[Ks, Js], ij[Is, Ks], ij[Js, Ks]
        neg = np.full(len(Ps), -1)
        rc = T(p=Ps, jk_cpl=neg, kj=kj_, s1=C(Ps, ki_), ki=ki_, s2=C(ki_, kj_), s3=C(kj_, Ps))
        rd = T(p=Ps, jk_cpl=neg, jk=jk_, s1=C(Ps, ik2), ik=ik2, s2=C(ik2, jk_), s3=C(jk_, Ps))
        sp_ = np.where(strong)[0]
        Pg = np.repeat(sp_, n)
        Kg = np.tile(np.arange(n), len(sp_))
        ikg = ij[pi[Pg], Kg]
        m_ = ikg >= 0
        rg = T(p=Pg[m_], s=C(Pg[m_], ikg[m_]), ik=ikg[m_], fflat=Kg[m_] * n + pj[Pg[m_]])
        C.finalize([tn, ti, tj, fkc, g3, g4, d3, d4, fdd, r1a, r1b, r1c, r1d, rb, rc, rd, rg])
        # energy
        self.t_build['terms'] = time.perf_counter() - t0
        t0 = time.perf_counter()

        # ---- PNO overlaps --------------------------------------------------
        cp_, cq_ = np.array(C.p, np.int64), np.array(C.q, np.int64)
        LS = self.LS = Layout(nv[cp_], nv[cq_])
        mode = os.environ.get('DLPNO_ONFLY_S', 'auto')
        from electrolites.dlpno import mgpu
        devs = mgpu.devices()
        home = cp.cuda.Device().id
        free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        if mode == 'auto':
            # store the overlaps when they fit comfortably, else regenerate them;
            # several GPUs: always chunked, the chunks (and their overlaps) spread
            need = float(np.sum(8. * nv[cp_] * nv[cq_]))
            # pooled free blocks are fragmented after the PNO integrals: only
            # half of them is counted (the store needs large contiguous groups)
            free_c = cp.cuda.runtime.memGetInfo()[0] + 0.5 * cp.get_default_memory_pool().free_bytes()
            onfly = need > 0.5 * free_c or len(devs) > 1
        else:
            onfly = mode == '1'
        self.onfly_s = onfly
        if onfly:
            # chunked plans; each device keeps the overlaps of its chunks up to a
            # budget, the rest is regenerated per use
            from electrolites.dlpno.ovgen import OverlapGen
            gens = {}

            def gen(d):
                if d not in gens:
                    with cp.cuda.Device(d):
                        fr = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
                        gens[d] = OverlapGen(cp.asarray(ref.orb.S_pao), ref.pao, ref.X,
                                             max_bytes=int(min(4 << 30, max(1 << 29, 0.1 * fr))))
                return gens[d]
            frac = float(os.environ.get('DLPNO_S_KEEP', '0.45' if len(devs) > 1 else '0.3'))
            per = {}
            for d in devs:
                with cp.cuda.Device(d):
                    per[d] = frac * (cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes())
            per[home] = min(per[home], frac * free)
            sbud = _SBudget(per) if mode != '1' else None
            self.S = None
            store = None
            cdevs = devs
            need = float(np.sum(8. * nv[cp_] * nv[cq_]))
            if len(devs) > 1 and mode != '1' and need < sum(per.values()):
                # the home device keeps the PNO integrals and the plans: the
                # store goes to the other devices when they can hold it
                others = [d for d in devs if d != home]
                sdevs = others if need < sum(per[d] for d in others) else devs
                store = _SStore(ref.orb.S_pao, ref.pao, ref.X, cp_, cq_, nv, sdevs)
                sbud = None
                if sdevs == others and os.environ.get('DLPNO_CC_HOME_CHUNKS', 'auto') != '1':
                    # the home device is the fullest (integrals, amplitudes):
                    # the chunks (gathered overlaps, plan tables) run on the others
                    cdevs = others
                cp.cuda.Device(home).use()
            hc = os.environ.get('DLPNO_CC_HOME_CHUNKS', 'auto')
            if store is None and len(devs) > 2 and hc != '1':
                # overlaps regenerated per chunk: the chunks leave the home
                # device too
                # (decided before the CC's own home-device data arrive: always,
                # crambin's home device went from half to full in the iterations)
                cdevs = [d for d in devs if d != home]
            if getattr(ref, 'verbose', 0) >= 2:
                print(f'[mem] PNO overlaps {need / 1e9:.1f} GB:',
                      'stored over the GPUs' if store is not None else 'regenerated per chunk', flush=True)
        else:
            self.S = self._overlaps(cp_, cq_)
        self.t_build['overlaps'] = time.perf_counter() - t0
        t0 = time.perf_counter()

        # ---- plans ---------------------------------------------------------
        def P(out_layout, out_idx, operands, out_row=None):
            if onfly and any(o.layout is LS for o in operands):
                return _ChunkedPlan(gen, LS, cp_, cq_, nv, out_layout, out_idx, operands, out_row=out_row,
                                    devs=cdevs, budget=sbud, store=store)
            return ChainPlan(out_layout, out_idx, operands, out_row=out_row)

        def DotPlan_(left, right):
            if onfly and any(o.layout is LS for o in right):
                return _ChunkedPlan(gen, LS, cp_, cq_, nv, None, None, right, left=left, devs=cdevs, budget=sbud,
                                    store=store)
            return DotPlan(left, right)
        self._P = P
        self.p_tn = P(LN, tn['p'], [op(LS, tn['s']), op(LT1, tn['n'])], out_row=tn['row'])
        self.p_ti = P(LV, ti['p'], [op(LS, ti['s']), op(LT1, ti['i'])])
        self.p_tj = P(LV, tj['p'], [op(LS, tj['s']), op(LT1, tj['j'])])
        allp = np.arange(npair)
        self.p_e1 = P(L11, allp, [op(LV, allp, True), op(LP, allp), op(LV, allp)])
        self.p_fkc = P(LV, fkc['p'], [op(LS, fkc['s1']), op(LP, fkc['ik']),
                                      op(LS, fkc['s2']), op(LT1, fkc['k'])])
        jj = dii[pj]
        rowj = np.array([pos[jj[p]][pi[p]] for p in range(npair)], dtype=np.int64)
        self.p_fkj = P(L11, allp, [op(LN, jj, row=rowj), op(LT1, pj)])
        # gamma
        self.p_g1 = P(LP, allp, [op(LN, allp, True), op(LN, allp)])
        self.p_g3 = P(LP, g3['p'], [op(LS, g3['s1']), op(LT1, g3['l']), op(LT1, g3['i'], True),
                                    op(LS, g3['s2'], True), op(LP, g3['kl']), op(LS, g3['s3'], True)])
        self.p_g4 = P(LP, g4['p'], [op(LS, g4['s1']), op(LP, g4['li']), op(LS, g4['s2']),
                                    op(LP, g4['kl']), op(LS, g4['s3'])])
        # delta
        self.p_d1 = P(LP, allp, [op(LN, allp, True), op(LN, allp)])
        self.p_d3 = P(LP, d3['p'], [op(LS, d3['s1']), op(LT1, d3['l']), op(LT1, d3['i'], True),
                                    op(LS, d3['s2'], True), op(LP, d3['lk']), op(LS, d3['s3'])])
        self.p_d4 = P(LP, d4['p'], [op(LS, d4['s1']), op(LP, d4['il']), op(LS, d4['s2']),
                                    op(LP, d4['lk']), op(LS, d4['s3'])])
        # Fkj''
        self.p_fdd = DotPlan_([op(LP, fdd['li'])],
                             [op(LS, fdd['s1']), op(LP, fdd['lj']), op(LS, fdd['s2'])])
        self.fdd_flat = cp.asarray(fdd['flat'])
        # R1
        self.p_r1a = P(LT1, r1a['i'], [op(LS, r1a['s'], True), op(LV, r1a['q'])])
        self.p_r1b_w = DotPlan_([op(LP, r1b['kl'])],
                               [op(LS, r1b['sa']), op(LP, r1b['q']), op(LS, r1b['sb'])])
        self.p_r1b = P(LT1, r1b['i'], [op(LS, r1b['s']), op(LT1, r1b['l'])])
        self.p_r1c = P(LT1, r1c['i'], [op(LS, r1c['s'], True), op(LP, r1c['p']), op(LV, r1c['q'])])
        self.p_B1 = P(LN, allp, [op(LN, allp), op(LP, allp, True)])
        self.p_B2 = P(LN, allp, [op(LN, allp), op(LP, allp), op(LP, allp, True)])
        self.p_r1d = P(LT1, r1d['i'], [op(LS, r1d['s'], True), op(LN, r1d['p'], True, row=r1d['row'])])
        # R2
        self.p_rb = P(LP, rb['p'], [op(LS, rb['s'], True), op(LP, rb['kl']), op(LS, rb['s'])])
        self.p_rf = P(LP, rb['p'], [op(LS, rb['s'], True), op(LP, rb['kl']), op(LP, rb['kl'], True),
                                    op(LS, rb['s'])])
        bflat_off = np.zeros(npair, np.int64)
        self.rb_beta_idx = (rb['p'], rb['bflat'])
        self.p_rc2 = P(LP, rc['p'], [op(LS, rc['s1']), op(LP, rc['ki']), op(LS, rc['s2']),
                                     op(LP, rc['kj'], True), op(LS, rc['s3'])])
        self.p_rd3 = P(LP, rd['p'], [op(LS, rd['s1']), op(LP, rd['ik']), op(LS, rd['s2']),
                                     op(LP, rd['jk'], True), op(LS, rd['s3'])])
        self.p_rg = P(LP, rg['p'], [op(LS, rg['s']), op(LP, rg['ik']), op(LS, rg['s'], True)])
        self.rg_f = cp.asarray(rg['fflat'])
        self.t_build['plans'] = time.perf_counter() - t0
        t0 = time.perf_counter()

        # ---- integrals -----------------------------------------------------
        self._load_ints(ints, rc, rd)
        self.t_build['ints'] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    def _overlaps(self, cp_, cq_):
        """S(p,q) = X_p^T S_pao[dom_p, dom_q] X_q for all registered couples,
        as one batched GEMM through PAO-embedded PNO coefficients."""
        ref = self.ref
        from electrolites.dlpno.lmp2_gpu import overlaps_chunked
        return overlaps_chunked(cp.asarray(ref.orb.S_pao), ref.pao, ref.X, cp_, cq_, self.LS)
        nbf = ref.orb.nbf
        npair = self.npair
        Spao = cp.asarray(ref.orb.S_pao)
        LX = Layout(np.full(npair, nbf), self.nv)
        Xs = Ragged(LX)
        for g, m in enumerate(LX.members):
            buf = np.zeros((len(m),) + LX.shapes[g])
            for s, p in enumerate(m):
                buf[s, ref.pao[p], :self.nv[p]] = ref.X[p]
            Xs.data[g] = cp.asarray(buf)
        pr = LX.shapes[0][0]
        Sp = cp.zeros((pr, pr))
        Sp[:nbf, :nbf] = Spao
        SX = Ragged(LX, [cp.matmul(Sp, d) for d in Xs.data])
        S = Ragged(self.LS)
        plan = ChainPlan(self.LS, np.arange(len(cp_)), [op(LX, cp_, True), op(LX, cq_)])
        plan.run(S, [Xs, SX])
        return S

    def _load_ints(self, ints, rc, rd):
        """Upload the PNO-basis integrals of the reference (pno_integrals)."""
        ref = self.ref
        LP, LN = self.LP, self.LN
        self.K = Ragged.from_list(LP, ref.K)
        self.Lij = Ragged.from_list(LP, ref.L_iajb)
        self.Kmibj = Ragged.from_list(LN, ref.K_mibj)
        self.Jijmb = Ragged.from_list(LN, ref.J_ijmb)
        # K_ivvv as (nv, nv, nv) per LP group
        self.K3 = []
        for g, m in enumerate(LP.members):
            pv = LP.shapes[g][0]
            buf = cp.zeros((len(m), pv, pv, pv))
            for s, p in enumerate(m):
                v = self.nv[p]
                buf[s, :v, :v, :v] = cp.asarray(ref.K_ivvv[p]).reshape(v, v, v)
                ref.K_ivvv[p] = None          # grouped now: the per-pair copy goes
            self.K3.append(buf)
            cp.get_default_memory_pool().free_all_blocks()
        # non-projected integrals, per (strong p, k in lmos[p])
        pk_p, pk_k, pk_r = [], [], []
        for p in np.where(self.strong)[0]:
            for kk, k in enumerate(ref.lmos[p]):
                pk_p.append(p)
                pk_k.append(k)
                pk_r.append(kk)
        pk_p = np.array(pk_p, np.int64)
        pk_k = np.array(pk_k, np.int64)
        kj = self.ij[pk_k, self.pj[pk_p]]
        LJ = self.LJ = Layout(self.nv[pk_p], self.nv[kj])
        self.Jikac = Ragged.from_list(LJ, [ref.J_ikac[p][r] for p, r in zip(pk_p, pk_r)])
        self.Kiakc = Ragged.from_list(LJ, [ref.K_iakc[p][r] for p, r in zip(pk_p, pk_r)])
        cid = np.arange(len(pk_p))
        P = self._P
        LS = self.LS
        # C: -J(p,k) T[kj]^T S(kj,p) ; D: (2K - J)(p,k) Tt[jk]^T S(jk,p)
        self.p_rc1 = P(LP, pk_p, [op(LJ, cid), op(LP, rc['kj'], True), op(LS, rc['s3'])])
        self.p_rd1 = P(LP, pk_p, [op(LJ, cid), op(LP, rd['jk'], True), op(LS, rd['s3'])])
        # DF groups over strong canonical pairs; the per-pair copies are then dropped
        for x in (self.K, self.Lij, self.Kmibj, self.Jijmb, self.Jikac, self.Kiakc):
            x._static = True
        # the per-pair lists are no longer needed: free them before the DF groups
        for name in ('K_mibj', 'J_ijmb', 'K_ivvv', 'J_ikac', 'K_iakc'):
            setattr(ref, name, None)
        self._load_df()
        for name in ('Qma', 'Qab', 'iQk', 'iQa'):
            setattr(ref, name, [None] * len(getattr(ref, name)))
        if getattr(ref, '_df_spill', None) is not None:
            ref._df_spill.close()          # grouped: the scratch files go
            ref._df_spill = None
        cp.get_default_memory_pool().free_all_blocks()

    def _load_df(self):
        """DF tensors of the strong canonical pairs in coarse size buckets
        (aux to 128, LMOs to 16, PNOs to 32: few groups, few launches per
        iteration); a member's items are gathered from / scattered to their
        own layout groups."""
        ref = self.ref
        sel = np.where(self.strong & self.canon)[0]
        naux = np.array([ref.Qma[p].shape[0] for p in sel], np.int64)
        # spilled pair DF tensors: the packed groups stay on the scratch file
        # system too and are read at every iteration (DLPNO_CC_DF_DISK=auto|0|1)
        import os
        env = os.environ.get('DLPNO_CC_DF_DISK', 'auto')
        self._df_store = None
        if getattr(ref, 'df_offload', False) and (env == '1' or (
                env == 'auto' and getattr(ref, '_df_spill', None) is not None)):
            from electrolites.dlpno.devmem import SpillStore
            self._df_store = SpillStore('ccdf')
        bk = lambda n, q: np.maximum(q, -(-np.asarray(n) // q) * q)
        Ab, Lb, Vb = bk(naux, 128), bk(self.nl[sel], 16), bk(self.nv[sel], 32)
        key = np.stack([Ab, Lb, Vb], 1)
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        self.df = []
        chunks = []
        for u in range(len(uk)):
            A, Lp, Vp = (int(x) for x in uk[u])
            per = 8 * A * max(Vp * Vp, Lp * Lp, Lp * Vp) * 4
            step = max(1, (1 << 28) // per)
            allm = sel[inv == u]
            for c0 in range(0, len(allm), step):
                chunks.append((A, Lp, Vp, allm[c0:c0 + step]))
        from electrolites.dlpno import mgpu
        self.devs = mgpu.devices()
        load = np.zeros(len(self.devs))
        home = cp.cuda.Device().id
        off = getattr(ref, 'df_offload', False)
        dbg = os.environ.get('DLPNO_DEBUG_HOST') == '1'

        def rss():
            for l_ in open('/proc/self/status'):
                if l_.startswith('VmRSS'):
                    return int(l_.split()[1]) / 1e6
        if dbg:
            nb0 = sum(x.nbytes for nm in ('Qma', 'Qab', 'iQk', 'iQa') for x in getattr(ref, nm) if x is not None)
            print(f'[debug] before DF grouping: RSS {rss():.2f} GB, per-pair lists {nb0 / 1e9:.2f} GB (with dups)', flush=True)
        for A, Lp, Vp, mem in chunks:
            g = len(mem)
            # several GPUs: the groups (and their DF tensors) are spread
            r = int(np.argmin(load))
            load[r] += g * A * (Vp * Vp + Lp * Vp)
            with cp.cuda.Device(self.devs[r]):
                self._load_df_group(mem, A, Lp, Vp, g)
            if off:
                # the released per-pair host copies back to the OS (the C heap
                # keeps freed blocks), so that they make room for the groups
                _malloc_trim()
        if dbg:
            left = sum(1 for nm in ('Qma', 'Qab') for x in getattr(ref, nm) if x is not None)
            print(f'[debug] after DF grouping: RSS {rss():.2f} GB, per-pair arrays left {left}', flush=True)
        if self._df_store is not None:
            self._df_store.flush()
        if off and getattr(ref, 'verbose', 0) >= 2:
            nb = sum(sum(d[k].nbytes for k in ('Qma', 'Qab', 'Qk', 'Qa')) for d in self.df)
            where = f'on scratch ({self._df_store.dir})' if self._df_store is not None else 'in pinned host memory'
            print(f'[mem] CC DF groups {where}: {nb / 1e9:.1f} GB', flush=True)
        self.df_dev = sorted({d['dev'] for d in self.df})
        cp.cuda.Device(home).use()

    @staticmethod
    def _items(layout, its):
        """[(layout group, member rows, slots, rows, cols)] of items ``its``."""
        out = []
        gg = layout.group[its]
        for g_ in np.unique(gg):
            m = np.where(gg == g_)[0]
            pr, pc = layout.shapes[g_]
            out.append((int(g_), cp.asarray(m), cp.asarray(layout.slot[its[m]]), pr, pc))
        return out

    @staticmethod
    def _gat(rag, lst, g, R, C):
        out = cp.zeros((g, R, C))
        for gs, mi, sl, pr, pc in lst:
            out[mi, :pr, :pc] = rag.data[gs][sl]
        return out

    @staticmethod
    def _put(rag, lst, X, add=False):
        for gs, mi, sl, pr, pc in lst:
            if add:
                rag.data[gs][sl] += X[mi, :pr, :pc]
            else:
                rag.data[gs][sl] = X[mi, :pr, :pc]

    def _load_df_group(self, mem, A, Lp, Vp, g):
        ref = self.ref
        off = getattr(ref, 'df_offload', False)
        q = self.ji[mem]
        if off:
            # host memory: the members' blocks back to back (no padding), padded
            # on the device when the group is used
            st = getattr(self, '_df_store', None)
            Qma = _Packed([ref.Qma[p] for p in mem], (A, Lp, Vp), store=st)
            Qab = _Packed([ref.Qab[p] for p in mem], (A, Vp, Vp), store=st)
            Qk = _Packed([x for p, r in zip(mem, q) for x in (ref.iQk[p], ref.iQk[r])], (A, Lp), 2, store=st)
            Qa = _Packed([x for p, r in zip(mem, q) for x in (ref.iQa[p], ref.iQa[r])], (A, Vp), 2, store=st)
            for p, r in zip(mem, q):
                ref.Qma[p] = ref.Qab[p] = ref.iQk[p] = ref.iQa[p] = None
                ref.Qma[r] = ref.Qab[r] = ref.iQk[r] = ref.iQa[r] = None
        else:
            Qma = cp.zeros((g, A, Lp, Vp))
            Qab = cp.zeros((g, A, Vp, Vp))
            Qk = cp.zeros((g, 2, A, Lp))
            Qa = cp.zeros((g, 2, A, Vp))
            for s, p in enumerate(mem):
                r = q[s]
                na, nl, nv = ref.Qma[p].shape
                Qma[s, :na, :nl, :nv] = cp.asarray(ref.Qma[p])
                Qab[s, :na, :nv, :nv] = cp.asarray(ref.Qab[p])
                Qk[s, 0, :na, :nl] = cp.asarray(ref.iQk[p])
                Qk[s, 1, :na, :nl] = cp.asarray(ref.iQk[r])
                Qa[s, 0, :na, :nv] = cp.asarray(ref.iQa[p])
                Qa[s, 1, :na, :nv] = cp.asarray(ref.iQa[r])
                # release the per-pair copies as soon as they are grouped
                ref.Qma[p] = ref.Qab[p] = ref.iQk[p] = ref.iQa[p] = None
                ref.Qma[r] = ref.Qab[r] = ref.iQk[r] = ref.iQa[r] = None
        pos = ref.lmo_pos
        dmask = self.pi[mem] == self.pj[mem]
        ev = np.zeros((g, Vp))
        lm_idx = np.zeros((g, Lp), np.int64)
        lm_msk = np.zeros((g, Lp))
        for s, p in enumerate(mem):
            ev[s, :self.nv[p]] = ref.e_pno[p]
            lm_idx[s, :self.nl[p]] = ref.lmos[p]
            lm_msk[s, :self.nl[p]] = 1.
        it = self._items
        self.df.append(dict(
            p=mem, q=q, A=A, Lb=Lp, Vb=Vp,
            N=it(self.LN, mem), Nq=it(self.LN, q), P=it(self.LP, mem), Pq=it(self.LP, q),
            L=it(self.LL, mem), Lq=it(self.LL, q),
            ri=cp.asarray([pos[p][self.pi[p]] for p in mem]),
            rj=cp.asarray([pos[p][self.pj[p]] for p in mem]),
            dsel=cp.asarray(np.where(dmask)[0]), di=self.pi[mem][dmask],
            ev=cp.asarray(ev), lm_idx=cp.asarray(lm_idx), lm_msk=cp.asarray(lm_msk),
            Qma=Qma, Qab=Qab, Qk=Qk, Qa=Qa, dev=cp.cuda.Device().id))

    # ------------------------------------------------------------------
    # elementwise helpers
    def swapT(self, R):
        """R'[p] = R[ji(p)]^T for a square pair layout."""
        ps = self.pslot[id(R.layout)]
        return Ragged(R.layout, [d[s].transpose(0, 2, 1) for d, s in zip(R.data, ps)])

    def swap(self, R):
        ps = self.pslot[id(R.layout)]
        return Ragged(R.layout, [d[s] for d, s in zip(R.data, ps)])

    def masked(self, R, sel):
        """zero all items not in boolean mask sel (over pairs); float masks
        are cached (boolean indexing would synchronise the stream)."""
        cache = self.__dict__.setdefault('_mcache', {})
        key = (id(R.layout), id(sel))
        mk = cache.get(key)
        if mk is None:
            mk = cache[key] = [cp.asarray(sel[m].astype(np.float64))[:, None, None]
                               for m in R.layout.members]
        return Ragged(R.layout, [d * k for d, k in zip(R.data, mk)])

    # ------------------------------------------------------------------
    # iterations
    # ------------------------------------------------------------------
    def init_state(self):
        ref = self.ref
        f = ref.orb.F_lmo
        self.F = cp.asarray(f)
        self.T = Ragged.from_list(self.LP, ref.T)
        self.T1 = Ragged(self.LT1)
        fd = np.diag(f)
        D2 = []
        for g, m in enumerate(self.LP.members):
            pv = self.LP.shapes[g][0]
            e = np.zeros((len(m), pv))
            for s, p in enumerate(m):
                e[s, :self.nv[p]] = ref.e_pno[p]
            d = (e[:, :, None] + e[:, None, :]
                 - (fd[self.pi[m]] + fd[self.pj[m]])[:, None, None])
            d = np.where(self.LP.mask()[g].get() > 0, d, 1.)
            D2.append(cp.asarray(d))
        self.D2 = Ragged(self.LP, D2)
        D1 = []
        for g, m in enumerate(self.LT1.members):
            pv = self.LT1.shapes[g][0]
            e = np.ones((len(m), pv, 1))
            for s, i in enumerate(m):
                ii = self.dii[i]
                e[s, :self.nv[ii], 0] = ref.e_pno[ii] - fd[i]
            D1.append(cp.asarray(e))
        self.D1 = Ragged(self.LT1, D1)
        self.offdiag = self.pi != self.pj

    def _beta_w(self, beta):
        p, bflat = self.rb_beta_idx
        LL = self.LL
        if not hasattr(self, '_bw_idx'):
            self._bw_idx = []
            for g in range(LL.ngroups):
                sel = np.where(LL.group[p] == g)[0]
                if len(sel) == 0:
                    continue
                Lp = LL.shapes[g][0]
                nlp = self.nl[p[sel]]
                a, b = bflat[sel] // nlp, bflat[sel] % nlp
                self._bw_idx.append((g, cp.asarray(sel), cp.asarray(LL.slot[p[sel]]),
                                     cp.asarray(a * Lp + b)))
        w = cp.zeros(len(p))
        for g, sel, sl, fl in self._bw_idx:
            w[sel] = beta.data[g].reshape(beta.data[g].shape[0], -1)[sl, fl]
        return w

    @staticmethod
    def _scal(r):
        """values of a (1x1)-block Ragged in item order."""
        lay = r.layout
        out = cp.zeros(lay.n)
        for g, m in enumerate(lay.members):
            if len(m):
                out[cp.asarray(m)] = r.data[g][:, 0, 0]
        return out

    def _v3(self, Tt):
        """v[x]_a = sum_cd K_ivvv[x]_{c,d,a} Tt[x]_{cd}, on LV."""
        V3 = Ragged(self.LV)
        LP, LV = self.LP, self.LV
        for g, m in enumerate(LP.members):
            if not len(m):
                continue
            v = _wsum(Tt.data[g].reshape(Tt.data[g].shape[0], -1), self.K3[g].reshape(self.K3[g].shape[0], -1, self.K3[g].shape[3]))
            gv = LV.group[m]
            assert np.all(gv == gv[0])
            V3.data[gv[0]][cp.asarray(LV.slot[m]), :, 0] = v
        return V3

    def residuals(self):
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
        cupyx.scatter_add(Fij_bar.ravel(), pij, sc)
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
        cupyx.scatter_add(Fkj.ravel(), pij, self._scal(v))
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
                # K3[ji(p)] contracted with t_p: the partner map is an
                # involution, so permute the vectors and results instead of
                # gathering a copy of the (large) 3-external group
                pg = ps[g]
                tivp = TiP.data[g][:, :, 0][pg]
                Dl.data[g] += 2 * _mv(K3[g], tivp)[pg].transpose(0, 2, 1)
                Dl.data[g] -= _wsum(tivp, K3[g])[pg]
        self.p_d3.run(Dl, [S, T1, T1, S, self.Lij, S], alpha=-1.)
        self.p_d4.run(Dl, [S, Tt, S, self.Lij, S], alpha=.5)
        Fdd = Fkj.copy()
        cupyx.scatter_add(Fdd.ravel(), self.fdd_flat, self.p_fdd.run([self.K], [S, Tt, S]))
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

    def _df_groups(self, groups, T, Tn, Fkc_bar, Fab, beta, Rsym):
        """T1-dressed DF terms of the strong pairs of ``groups`` (all on the
        current device) into the given outputs; returns the diagonal items."""
        diag_items = []
        gat, put = self._gat, self._put
        for d in groups:
            g, Lb, Vb = len(d['p']), d['Lb'], d['Vb']
            Tnp = gat(Tn, d['N'], g, Lb, Vb)
            Tnq = gat(Tn, d['Nq'], g, Lb, Vb)
            Tp = gat(T, d['P'], g, Vb, Vb)
            Tq = gat(T, d['Pq'], g, Vb, Vb)
            Qma, Qab, Qk, Qa = (d[k_].device() if isinstance(d[k_], _Packed) else d[k_]
                                for k_ in ('Qma', 'Qab', 'Qk', 'Qa'))
            ar = cp.arange(len(d['p']))
            ti = Tnp[ar, d['ri']]
            tj = Tnq[ar, d['rj']]
            qk_p = Qk[:, 0] + _mv(Qma, ti)
            qa_p = Qa[:, 0] - cp.matmul(qk_p, Tnp) + _mv(Qab, ti)
            qk_q = Qk[:, 1] + _mv(Qma, tj)
            qa_q = Qa[:, 1] - cp.matmul(qk_q, Tnq) + _mv(Qab, tj)
            gam = _dot2(Qma, Tnp)
            X = cp.matmul(Qma, Tnp.transpose(0, 2, 1)[:, None])          # gAkm
            fkc = 2 * _wsum(gam, Qma) - cp.matmul(X, Qma).sum(axis=1)
            Y = cp.matmul(Qab, Tnp.transpose(0, 2, 1)[:, None])          # gAam
            fab = 2 * _wsum(gam, Qab) - cp.matmul(Y, Qma).sum(axis=1)
            ev = d['ev']
            fab = fab + ev[:, :, None] * cp.eye(ev.shape[1])[None]
            put(Fkc_bar, d['N'], fkc)
            put(Fab, d['P'], fab - cp.matmul(Tnp.transpose(0, 2, 1), fkc))
            QT = cp.matmul(Qma, Tp[:, None])
            put(beta, d['L'], cp.matmul(qk_p.transpose(0, 2, 1), qk_q)
                + cp.matmul(QT, Qma.transpose(0, 1, 3, 2)).sum(axis=1))
            QT = cp.matmul(Qma, Tq[:, None])
            put(beta, d['Lq'], cp.matmul(qk_q.transpose(0, 2, 1), qk_p)
                + cp.matmul(QT, Qma.transpose(0, 1, 3, 2)).sum(axis=1))
            Kij = cp.matmul(qa_p.transpose(0, 2, 1), qa_q)
            Qt = Qab - cp.matmul(Tnp.transpose(0, 2, 1)[:, None], Qma)
            A = cp.matmul(cp.matmul(Qt, Tp[:, None]), Qt.transpose(0, 1, 3, 2)).sum(axis=1)
            put(Rsym, d['P'], Kij + A, add=True)
            if len(d['di']):
                ds = d['dsel']
                fai = 2 * cp.einsum('gA,gAa->ga', gam[ds], Qa[ds, 0])
                fai -= cp.einsum('gAam,gAm->ga', Y[ds], Qk[ds, 0])
                diag_items.append((d, ds, fai, fab[ds], fkc[ds], Tnp[ds]))
        return diag_items

    def _df_all(self, T, Tn, Fkc_bar, Fab, beta, Rsym):
        """DF terms of all groups; with several GPUs every device runs its own
        groups on replicas of T, Tn and adds its outputs into the home ones."""
        devs = getattr(self, 'df_dev', [cp.cuda.Device().id])
        home = cp.cuda.Device().id
        if devs == [home]:
            return self._df_groups(self.df, T, Tn, Fkc_bar, Fab, beta, Rsym)
        from electrolites.dlpno import mgpu
        res = {}

        def work(ctx, d_):
            grp = [d for d in self.df if d['dev'] == d_]
            if d_ == home:
                res[d_] = (self._df_groups(grp, T, Tn, Fkc_bar, Fab, beta, Rsym), None)
                return
            T_, Tn_ = Ragged(T.layout, mgpu.replicate(T.data)), Ragged(Tn.layout, mgpu.replicate(Tn.data))
            outs = [Ragged(x.layout) for x in (Fkc_bar, Fab, beta, Rsym)]
            di = self._df_groups(grp, T_, Tn_, *outs)
            res[d_] = (di, outs)
        mgpu.run_per_device(lambda d_: work(None, d_), devs)
        diag_items = []
        for d_ in devs:
            di, outs = res[d_]
            if outs is not None:
                for dst, src in zip((Fkc_bar, Fab, beta, Rsym), outs):
                    for g in range(len(dst.data)):
                        if dst.data[g].size:
                            dst.data[g] += cp.asarray(src.data[g])
                di = [(d, cp.asarray(ds), cp.asarray(a), cp.asarray(b), cp.asarray(c), cp.asarray(e))
                      for d, ds, a, b, c, e in di]
            diag_items += di
        return diag_items

    def energy(self):
        TiP = self.p_ti.run(Ragged(self.LV), [self.S, self.T1])
        TjP = self.p_tj.run(Ragged(self.LV), [self.S, self.T1])
        return self.T.vdot_items(self.Lij) + self._scal(
            self.p_e1.run(Ragged(self.L11), [TiP, self.Lij, TjP]))

    def kernel(self, verbose=True):
        opt = self.ref.opt
        self.init_state()
        # DLPNO_CC_AMP=<file on scratch>: the amplitudes are saved after every
        # iteration and, when the file exists, read as the starting point
        # (CC restarts across allocations; the DIIS history is saved next to
        # it, ``<file>.diis``, and continued; DLPNO_CC_DIIS_RESUME=0 restarts it)
        amp = os.environ.get('DLPNO_CC_AMP')
        if amp and os.path.exists(amp):
            z = np.load(amp)
            self.T1.set_flat(cp.asarray(z['t1']))
            self.T.set_flat(cp.asarray(z['t2']))
            print(f'  CC amplitudes read from {amp} (iteration {int(z["it"])})', flush=True)
        diis = _DeviceDIIS(opt['DIIS_MAX_VECS'])
        if amp and os.path.exists(amp) and 'tok' in z and os.environ.get('DLPNO_CC_DIIS_RESUME', '1') == '1':
            if diis.load(amp + '.diis', float(z['tok']), int(self.T1.flat().size + self.T.flat().size)):
                print(f'  DIIS history read ({len(diis.x)} vectors)', flush=True)
        e_prev = 0.
        weak = cp.asarray(~self.strong)
        if getattr(self.ref, 'verbose', 0) >= 2:
            from electrolites.dlpno import mgpu
            used = {}
            for d in mgpu.devices():
                with cp.cuda.Device(d):
                    fr, tot = cp.cuda.runtime.memGetInfo()
                    used[d] = (round((tot - fr) / 1e9, 1),
                               round(cp.get_default_memory_pool().free_bytes() / 1e9, 1))
            print('[mem] device GB (used, pool free) before the CC iterations:', used, flush=True)
        for it in range(1, opt['MAXITER'] + 1):
            t0 = time.perf_counter()
            R1, R2 = self.residuals()
            rn = _rnorm(R1, R2)
            s1, s2 = R1 / self.D1, R2 / self.D2
            self.T1 = self.T1 - s1
            self.T = self.T - s2
            n1 = self.T1.flat().size
            # DIIS error: the residual, or the (preconditioned) amplitude step,
            # which weighs small-denominator components fairly
            err = (cp.concatenate([s1.flat(), s2.flat()]) if _diis_step(opt)
                   else cp.concatenate([R1.flat(), R2.flat()]))
            del s1, s2
            x = diis.update(cp.concatenate([self.T1.flat(), self.T.flat()]), err)
            del err
            self.T1.set_flat(x[:n1])
            self.T.set_flat(x[n1:])
            e = self.energy()
            if amp:
                tmp = amp + '.tmp.npz'
                tok = time.time()
                diis.save(amp + '.diis', tok)
                np.savez(tmp, t1=cp.asnumpy(self.T1.flat()), t2=cp.asnumpy(self.T.flat()), it=it, tok=tok)
                os.replace(tmp, amp)
            if it % 10 == 0 or (it <= 2 and os.environ.get('DLPNO_PROF') == '1'):
                self._prof.report()
            e_cur = float(e.sum())
            e_weak = float(e[weak].sum())
            if verbose:
                print(f'  GPU LCCSD iter {it:3d} {e_cur:.12f} {e_cur - e_prev:.3e} '
                      f'{rn[0]:.3e} {rn[1]:.3e}  {time.perf_counter() - t0:.2f}s', flush=True)
            if _converged(e_cur - e_prev, rn, opt):
                break
            e_prev = e_cur
        else:
            raise RuntimeError('LCCSD did not converge')
        return e_cur - e_weak, e_weak


def _diis_step(opt):
    """DIIS error vector of the CC iterations: 'step' (amplitude update R/D)
    or 'residual' (DIIS_ERR option, env DLPNO_DIIS_ERR)."""
    return os.environ.get('DLPNO_DIIS_ERR', opt.get('DIIS_ERR', 'residual')) == 'step'


class _DeviceDIIS:
    """DIIS on device vectors.  The Gram matrix of the error vectors is kept
    on the host and extended by one row per update; the extrapolation is
    accumulated vector by vector (no stacked copies of the history).  When
    the history would take more than 10% of the free device memory (5% of the device) it
    is kept in pinned host memory, and when it would take more than 25% of
    the host's available memory, in files on scratch (memory-mapped, read
    slice by slice; DLPNO_DIIS_DISK=auto|0|1)."""

    def __init__(self, nvec=6):
        self.nvec = nvec
        self.x, self.e = [], []
        self.G = np.zeros((0, 0))
        self.host = None
        self.disk = None

    def _keep(self, v):
        if self.host is None:
            free, tot = cp.cuda.runtime.memGetInfo()
            free += cp.get_default_memory_pool().free_bytes()
            # the history grows over the first iterations, after the plans
            # have taken their share: on the device only when it is small
            # (also above 1 GB: the history grows by two vectors an iteration
            # after the other stores have taken the device; the 309-atom
            # mimochrome quintet ran out of device memory in iteration 5)
            self.host = (2 * self.nvec * v.nbytes > min(0.1 * free, 0.05 * tot, 1 << 30)
                         or os.environ.get('DLPNO_DIIS_HOST') == '1')       # testing
            from electrolites.dlpno.dfints import _host_available, scratch_dir
            env = os.environ.get('DLPNO_DIIS_DISK', 'auto')
            if self.host and (env == '1' or (env == 'auto' and 2 * self.nvec * v.nbytes
                                             > 0.25 * _host_available())):
                self.disk = scratch_dir('diis')
                self._nfile = 0
        if self.disk is not None:
            fn = os.path.join(self.disk, f'v{self._nfile}.npy')
            self._nfile += 1
            m = np.lib.format.open_memmap(fn, mode='w+', dtype=v.dtype, shape=v.shape)
            S = self.SEG
            for i in range(0, v.size, S):
                m[i:i + S] = cp.asnumpy(v[i:i + S])
            m.flush()
            del m
            return np.load(fn, mmap_mode='r')
        if self.host:
            from electrolites.dlpno.devmem import to_pinned
            return to_pinned(v)
        return v.copy()

    def _drop(self, v):
        if self.disk is not None and isinstance(v, np.memmap):
            fn = v.filename
            del v
            try:
                os.remove(fn)
            except OSError:
                pass

    def close(self):
        if self.disk is not None:
            import shutil
            self.x, self.e = [], []
            shutil.rmtree(self.disk, ignore_errors=True)
            self.disk = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _dev(v):
        return v if isinstance(v, cp.ndarray) else cp.asarray(v)

    # --- the history in files next to the saved amplitudes (DLPNO_CC_AMP),
    # so that a restart in another allocation continues the extrapolation
    # instead of starting it afresh (each new pair is written once)

    def save(self, d, tok):
        os.makedirs(d, exist_ok=True)
        ids = self.__dict__.setdefault('_ids', [])
        done = self.__dict__.setdefault('_written', set())
        S = self.SEG
        for i, x, e in zip(ids, self.x, self.e):
            if i in done:
                continue
            for nm, v in (('x', x), ('e', e)):
                m = np.lib.format.open_memmap(os.path.join(d, f'{nm}{i}.npy'), mode='w+',
                                              dtype=np.float64, shape=(v.size,))
                for a in range(0, v.size, S):
                    m[a:a + S] = cp.asnumpy(v[a:a + S]) if isinstance(v, cp.ndarray) else v[a:a + S]
                m.flush()
                del m
            done.add(i)
        tmp = os.path.join(d, 'index.tmp.npz')
        np.savez(tmp, ids=np.array(ids, np.int64), G=self.G, tok=tok, nxt=self.__dict__.get('_next', 0))
        os.replace(tmp, os.path.join(d, 'index.npz'))
        for i in sorted(done - set(ids)):
            for nm in ('x', 'e'):
                try:
                    os.remove(os.path.join(d, f'{nm}{i}.npy'))
                except OSError:
                    pass
            done.discard(i)

    def load(self, d, tok, size):
        """the history saved with amplitudes ``tok`` (vectors of ``size``
        elements); False when there is none that matches."""
        fn = os.path.join(d, 'index.npz')
        if not os.path.exists(fn):
            return False
        z = np.load(fn)
        if float(z['tok']) != float(tok):
            return False
        ids = [int(i) for i in z['ids']]
        vs = []
        for i in ids:
            pair = []
            for nm in ('x', 'e'):
                p = os.path.join(d, f'{nm}{i}.npy')
                if not os.path.exists(p):
                    return False
                v = np.load(p, mmap_mode='r')
                if v.size != size:
                    return False
                pair.append(v)
            vs.append(pair)
        for x, e in vs:
            for v, lst in ((x, self.x), (e, self.e)):
                lst.append(self._restore(v))
        self.G = np.array(z['G'])
        self._ids = ids
        self._next = int(z['nxt'])
        self._written = set(ids)
        return True

    def _restore(self, v):
        """a saved history vector (memory-mapped) into this DIIS's storage"""
        if self.host is None:
            free, tot = cp.cuda.runtime.memGetInfo()
            free += cp.get_default_memory_pool().free_bytes()
            self.host = (2 * self.nvec * v.nbytes > min(0.1 * free, 0.05 * tot, 1 << 30)
                         or os.environ.get('DLPNO_DIIS_HOST') == '1')
        if not self.host:
            return cp.asarray(np.ascontiguousarray(v))
        return self._keep(v)

    SEG = 1 << 26        # elements per host -> device slice (512 MB)

    def _dot(self, e, v):
        if isinstance(v, cp.ndarray):
            return float(cp.dot(e, v))
        S = self.SEG
        return sum(float(cp.dot(e[i:i + S], cp.asarray(v[i:i + S]))) for i in range(0, v.size, S))

    def update(self, x, e):
        g = np.array([self._dot(e, v) for v in self.e] + [float(cp.dot(e, e))])
        n0 = len(self.e)
        G = np.zeros((n0 + 1, n0 + 1))
        G[:n0, :n0] = self.G
        G[n0, :] = G[:, n0] = g
        self.G = G
        self.x.append(self._keep(x))
        self.e.append(self._keep(e))
        ids = self.__dict__.setdefault('_ids', [])
        ids.append(self.__dict__.get('_next', 0))
        self._next = ids[-1] + 1
        if len(self.x) > self.nvec:
            k = int(np.argmax(np.diag(self.G)))
            ids.pop(k)
            xk, ek = self.x.pop(k), self.e.pop(k)
            self._drop(xk)
            self._drop(ek)
            del xk, ek
            self.G = np.delete(np.delete(self.G, k, 0), k, 1)
        n = len(self.x)
        if n < 2:
            return x
        B = -np.ones((n + 1, n + 1))
        B[n, n] = 0
        B[:n, :n] = self.G
        rhs = np.zeros(n + 1)
        rhs[n] = -1
        try:
            c = np.linalg.solve(B, rhs)[:n]
        except np.linalg.LinAlgError:
            return x
        if not self.host:
            out = self.x[0] * c[0]
            for ci, v in zip(c[1:], self.x[1:]):
                out += ci * v
            return out
        # host history: the extrapolation slice by slice (no full-size device
        # temporaries per vector), written over x
        S = self.SEG
        for i in range(0, x.size, S):
            seg = x[i:i + S]
            seg[...] = cp.asarray(self.x[0][i:i + S]) * c[0]
            for ci, v in zip(c[1:], self.x[1:]):
                seg += ci * cp.asarray(v[i:i + S])
        return x
