"""Ragged batches of small dense blocks on the device.

A DLPNO calculation is a few million small dense matrices whose dimensions
differ from pair to pair (PNOs per pair, LMOs per pair domain, TNOs per
triplet).  They are stored here padded to a handful of sizes and grouped by
padded shape, so that every term of the amplitude equations becomes a few
strided-batched GEMMs per shape combination instead of one GEMM per pair.

Padding is always with zeros, and every operation below maps zero padding to
zero padding, so the padded entries never leak into a result.  Padded sizes
depend only on the true size (``padded``), so a chain of blocks whose true
inner dimensions agree also agrees in its padded inner dimensions.

``ChainPlan`` is the work horse: a static list of terms

    out[o_t] += w_t * F1[i1_t]^{op} @ F2[i2_t]^{op} @ ...

grouped once by the shapes of their operands and replayed every iteration.
"""
import numpy as np
import cupy as cp
import cupyx


def padded(n):
    n = np.asarray(n, dtype=np.int64)
    small = np.maximum(8, -(-n // 8) * 8)
    big = -(-n // 16) * 16
    out = np.where(n <= 64, small, big)
    return np.where(n == 1, 1, np.where(n == 0, 1, out))


_GATHER = r'''
// dst[s][i, j] = src_s[i * rs_s + j * cs_s]  for i < nr[s], j < nc[s]
extern "C" __global__ void gather_items(const long long* __restrict__ srcp, const long long* __restrict__ rs,
                                        const long long* __restrict__ cs, const int* __restrict__ nr,
                                        const int* __restrict__ nc, double* __restrict__ dst,
                                        const int pr, const int pc, const long long* __restrict__ slots)
{
    const int s = blockIdx.x;
    const long long ds = slots ? slots[s] : s;
    const double* src = (const double*)srcp[s];
    const int r = nr[s], c = nc[s];
    const long long a = rs[s], b = cs[s];
    double* d = dst + ds * pr * pc;
    for (int e = threadIdx.x; e < r * c; e += blockDim.x) {
        const int i = e / c, j = e % c;
        d[(long long)i * pc + j] = src[i * a + j * b];
    }
}
'''
_gk = None


def _gather_items(out, items, rows, cols, slots=None):
    """out[s, :rows[s], :cols[s]] = items[s] (device arrays, any strides; an
    item may be stored flat with rows[s] * cols[s] elements)."""
    global _gk
    if _gk is None:
        _gk = cp.RawKernel(_GATHER, 'gather_items')
    n = len(items)
    ptr = np.empty(n, np.int64)
    rs = np.empty(n, np.int64)
    cs = np.empty(n, np.int64)
    for s, a in enumerate(items):
        if a.ndim != 2 or a.shape != (rows[s], cols[s]):
            a = cp.ascontiguousarray(a).reshape(int(rows[s]), int(cols[s]))
            items[s] = a           # keep alive until the kernel ran
        ptr[s] = a.data.ptr
        rs[s] = a.strides[0] // 8
        cs[s] = a.strides[1] // 8
    pr, pc = out.shape[1], out.shape[2]
    _gk((n,), (128,), (cp.asarray(ptr), cp.asarray(rs), cp.asarray(cs),
                       cp.asarray(np.asarray(rows, np.int32)), cp.asarray(np.asarray(cols, np.int32)),
                       out, np.int32(pr), np.int32(pc),
                       cp.asarray(np.asarray(slots, np.int64)) if slots is not None else np.uint64(0)))
    cp.cuda.get_current_stream().synchronize()


class Layout:
    """Shapes and storage slots of a list of (rows[i], cols[i]) blocks."""

    def __init__(self, rows, cols):
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        cols = np.asarray(cols, dtype=np.int64).reshape(-1)
        if cols.size == 1 and rows.size != 1:
            cols = np.full_like(rows, cols[0])
        self.n = len(rows)
        self.rows, self.cols = rows, cols
        self.pr, self.pc = padded(rows), padded(cols)
        keys = self.pr * (1 << 20) + self.pc
        uk, inv = np.unique(keys, return_inverse=True)
        self.group = inv.astype(np.int64)
        self.shapes = [(int(k >> 20), int(k & ((1 << 20) - 1))) for k in uk]
        order = np.argsort(inv, kind='stable')
        counts = np.bincount(inv, minlength=len(uk))
        self.members = np.split(order, np.cumsum(counts)[:-1])
        self.slot = np.empty(self.n, dtype=np.int64)
        for m in self.members:
            self.slot[m] = np.arange(len(m))
        self._mask = None

    @property
    def ngroups(self):
        return len(self.shapes)

    def nbytes(self):
        return 8 * sum(len(m) * r * c for m, (r, c) in zip(self.members, self.shapes))

    def mask(self):
        """1 inside each block, 0 in its padding (cached)."""
        if self._mask is None:
            out = []
            for m, (pr, pc) in zip(self.members, self.shapes):
                r = cp.asarray(self.rows[m])[:, None, None]
                c = cp.asarray(self.cols[m])[:, None, None]
                out.append(((cp.arange(pr)[None, :, None] < r)
                            & (cp.arange(pc)[None, None, :] < c)).astype(cp.float64))
            self._mask = out
        return self._mask


class Ragged:
    """Device storage for one Layout."""

    def __init__(self, layout, data=None):
        self.layout = layout
        if data is None:
            data = [cp.zeros((len(m), r, c)) for m, (r, c) in
                    zip(layout.members, layout.shapes)]
        self.data = data

    # ---- construction / extraction --------------------------------------
    @classmethod
    def from_list(cls, layout, mats):
        r = cls(layout)
        dev = cp.cuda.Device().id
        for g, m in enumerate(layout.members):
            if len(m) == 0:
                continue
            pr, pc = layout.shapes[g]
            if all(isinstance(mats[it], cp.ndarray) and mats[it].device.id == dev for it in m):
                # device items: one copy kernel for the whole group (no per-item sync)
                _gather_items(r.data[g], [mats[it] for it in m],
                              layout.rows[m], layout.cols[m])
                continue
            buf = np.zeros((len(m), pr, pc))
            for s, it in enumerate(m):
                a = mats[it]
                if hasattr(a, 'get'):
                    a = a.get()
                a = np.asarray(a).reshape(layout.rows[it], layout.cols[it])
                buf[s, :a.shape[0], :a.shape[1]] = a
            r.data[g] = cp.asarray(buf)
        return r

    def to_list(self):
        lay = self.layout
        out = [None] * lay.n
        for g, m in enumerate(lay.members):
            h = self.data[g].get()
            for s, it in enumerate(m):
                out[it] = h[s, :lay.rows[it], :lay.cols[it]]
        return out

    def get(self, it):
        lay = self.layout
        return self.data[lay.group[it]][lay.slot[it], :lay.rows[it], :lay.cols[it]]

    # ---- arithmetic -------------------------------------------------------
    def copy(self):
        return Ragged(self.layout, [d.copy() for d in self.data])

    def zeros_like(self):
        return Ragged(self.layout)

    def _bin(self, other, op):
        if isinstance(other, Ragged):
            return Ragged(self.layout, [op(a, b) for a, b in zip(self.data, other.data)])
        return Ragged(self.layout, [op(a, other) for a in self.data])

    def __add__(self, o):
        return self._bin(o, lambda a, b: a + b)

    def __sub__(self, o):
        return self._bin(o, lambda a, b: a - b)

    def __mul__(self, o):
        return self._bin(o, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self._bin(o, lambda a, b: a / b)

    def __neg__(self):
        return Ragged(self.layout, [-a for a in self.data])

    def iadd(self, o, alpha=1.):
        for a, b in zip(self.data, o.data):
            if alpha == 1.:
                a += b
            else:
                a += alpha * b
        return self

    def T(self):
        """Blockwise transpose (needs a layout with rows == cols)."""
        return Ragged(self.layout, [a.transpose(0, 2, 1).copy() for a in self.data])

    def flat(self):
        return cp.concatenate([a.ravel() for a in self.data]) if self.data else cp.zeros(0)

    def set_flat(self, v):
        off = 0
        for g, a in enumerate(self.data):
            self.data[g] = v[off:off + a.size].reshape(a.shape)
            off += a.size

    def rms_per_item(self):
        lay = self.layout
        out = cp.zeros(lay.n)
        for g, m in enumerate(lay.members):
            if len(m):
                ss = cp.sum(self.data[g] ** 2, axis=(1, 2))
                out[cp.asarray(m)] = ss
        size = cp.asarray(np.maximum(lay.rows * lay.cols, 1))
        return cp.sqrt(out / size)

    def vdot_items(self, other):
        """sum(A_i * B_i) per item."""
        lay = self.layout
        out = cp.zeros(lay.n)
        for g, m in enumerate(lay.members):
            if len(m):
                out[cp.asarray(m)] = cp.sum(self.data[g] * other.data[g], axis=(1, 2))
        return out


class _Operand:
    """Block ``idx`` of a Ragged with ``layout``; with ``row`` given, only
    that row of each block, as a (1, cols) matrix."""
    __slots__ = ('layout', 'idx', 'trans', 'row')

    def __init__(self, layout, idx, trans=False, row=None):
        self.layout = layout
        self.idx = np.asarray(idx, dtype=np.int64)
        self.trans = trans
        self.row = None if row is None else np.asarray(row, dtype=np.int64)

    def shape(self, g):
        r, c = self.layout.shapes[g]
        if self.row is not None:
            r = 1
        return (c, r) if self.trans else (r, c)

    def fetch(self, rag, g, slots, rows):
        if rows is None:
            m = rag.data[g][slots]
        else:
            m = rag.data[g][slots, rows][:, None, :]
        return m.transpose(0, 2, 1) if self.trans else m


def op(layout, idx, trans=False, row=None):
    return _Operand(layout, idx, trans, row)


class ChainPlan:
    """out[o_t] (+)= w_t * prod_f op_f(F_f[i_{f,t}]) for a static term list.

    ``out`` may be None, in which case ``run`` returns the products grouped
    as (batch descriptors, arrays) for a custom reduction (see DotPlan)."""

    def __init__(self, out_layout, out_idx, operands, max_batch_bytes=1 << 28,
                 out_row=None):
        self.out_layout = out_layout
        out_row = None if out_row is None else np.asarray(out_row, dtype=np.int64)
        out_idx = np.asarray(out_idx, dtype=np.int64)
        self.nterms = len(out_idx)
        self.operands = operands
        if self.nterms == 0:
            self.batches = []
            return
        cols = [out_layout.group[out_idx]] if out_layout is not None else []
        for o in operands:
            assert len(o.idx) == self.nterms
            cols.append(o.layout.group[o.idx])
        keys = np.stack(cols, axis=1)
        uk, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        order = np.argsort(inv, kind='stable')
        counts = np.bincount(inv, minlength=len(uk))
        self.batches = []
        for u, terms in zip(uk, np.split(order, np.cumsum(counts)[:-1])):
            gs = list(u[1:]) if out_layout is not None else list(u)
            # shape of the largest intermediate, for chunking
            shapes = [o.shape(g) for o, g in zip(operands, gs)]
            for k in range(len(shapes) - 1):
                assert shapes[k][1] == shapes[k + 1][0], (shapes, 'inner dims')
            per = 8 * max([r * c for r, c in shapes] + [shapes[0][0] * shapes[-1][1]]) * 2
            step = max(1, int(max_batch_bytes // max(per, 1)))
            for t0 in range(0, len(terms), step):
                tt = terms[t0:t0 + step]
                b = dict(terms=cp.asarray(tt),
                         gs=gs,
                         slots=[cp.asarray(o.layout.slot[o.idx[tt]]) for o in operands],
                         rows=[None if o.row is None else cp.asarray(o.row[tt])
                               for o in operands],
                         og=int(u[0]) if out_layout is not None else None,
                         oslot=(cp.asarray(out_layout.slot[out_idx[tt]])
                                if out_layout is not None else None),
                         orow=None if out_row is None else cp.asarray(out_row[tt]))
                self.batches.append(b)

    def _product(self, b, rags):
        prod = None
        for o, g, sl, rw, r in zip(self.operands, b['gs'], b['slots'], b['rows'], rags):
            m = o.fetch(r, g, sl, rw)
            prod = m if prod is None else cp.matmul(prod, m)
        return prod

    def run(self, out, rags, w=None, alpha=1.):
        """out += alpha * w * chain.  w: device array over terms (or None)."""
        for b in self.batches:
            prod = self._product(b, rags)
            if w is not None:
                prod = prod * w[b['terms']][:, None, None]
            if alpha != 1.:
                prod *= alpha
            if b['orow'] is None:
                cupyx.scatter_add(out.data[b['og']], b['oslot'], prod)
            else:
                cupyx.scatter_add(out.data[b['og']], (b['oslot'], b['orow']),
                                  prod.reshape(prod.shape[0], -1))
        return out


class DotPlan:
    """s_t = sum( op(A[a_t]) * op(B[b_t]) ) for a static term list, where both
    operands are chains (ChainPlans with out=None) of equal shape."""

    def __init__(self, left, right):
        self.lops, self.rops = left, right
        n = len(left[0].idx)
        cols = [o.layout.group[o.idx] for o in left + right]
        keys = np.stack(cols, axis=1)
        uk, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        self.n = n
        self.batches = []
        for u in range(len(uk)):
            tt = np.where(inv == u)[0]
            gs = list(uk[u])
            self.batches.append(dict(
                terms=cp.asarray(tt), gs=gs,
                slots=[cp.asarray(o.layout.slot[o.idx[tt]]) for o in left + right],
                rows=[None if o.row is None else cp.asarray(o.row[tt])
                      for o in left + right]))

    def run(self, lrags, rrags):
        out = cp.zeros(self.n)
        nl = len(self.lops)
        for b in self.batches:
            prods = []
            for side, ops, rags in ((0, self.lops, lrags), (1, self.rops, rrags)):
                prod = None
                for k, (o, r) in enumerate(zip(ops, rags)):
                    g = b['gs'][k + side * nl]
                    m = o.fetch(r, g, b['slots'][k + side * nl], b['rows'][k + side * nl])
                    prod = m if prod is None else cp.matmul(prod, m)
                prods.append(prod)
            out[b['terms']] = cp.sum(prods[0] * prods[1], axis=(1, 2))
        return out
