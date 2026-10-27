"""Variable-size batched GEMM chains on the device.

``VChainPlan`` has the interface of ``ragged.ChainPlan`` but runs every term
of a chain step in ONE kernel launch whatever the terms' shapes: each thread
block finds its term, the term's true (m, n, k), and reads its operands
in place from the Ragged storage through per-term pointers.  There is no
grouping by shape, no gather copy of the operands and no arithmetic on
padding.  The last step of a chain scales by the per-term weight and adds
into the output block with atomics (several terms usually share an output).

The association order of a chain (left-to-right or right-to-left) is chosen
once per plan from the mean term shapes, which matters for chains that end
in a vector (S T S^T t).
"""
import os

import numpy as np
import cupy as cp
import cupyx

from electrolites.dlpno.ragged import padded

_SRC = r'''
#define TS 32
extern "C" __global__ void vgemm(const long long* __restrict__ Ap,
                                 const long long* __restrict__ Bp,
                                 const long long* __restrict__ Cp,
                                 const int* __restrict__ dims,     // m n k lda ldb ldc per term
                                 const double* __restrict__ alpha,
                                 const int ntiles_n_max,
                                 const int ta, const int tb, const int accumulate)
{
    const int t = blockIdx.x;
    const int m = dims[6 * t], n = dims[6 * t + 1], k = dims[6 * t + 2];
    const int lda = dims[6 * t + 3], ldb = dims[6 * t + 4], ldc = dims[6 * t + 5];
    const int tiles_n = (n + TS - 1) / TS;
    const int tile = blockIdx.y;
    const int tm = tile / ntiles_n_max, tn = tile % ntiles_n_max;
    if (tn >= tiles_n || tm * TS >= m) return;
    const double* __restrict__ A = (const double*)Ap[t];
    const double* __restrict__ B = (const double*)Bp[t];
    double* C = (double*)Cp[t];
    __shared__ double As[TS][TS + 1];
    __shared__ double Bs[TS][TS + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;      // 16 x 16
    const int lid = ty * 16 + tx;
    double acc00 = 0., acc01 = 0., acc10 = 0., acc11 = 0.;
    const int r0 = tm * TS, c0 = tn * TS;
    for (int k0 = 0; k0 < k; k0 += TS) {
        for (int idx = lid; idx < TS * TS; idx += 256) {
            int r = idx / TS, c = idx % TS;
            int gr = r0 + r, gc = k0 + c;
            double v = 0.;
            if (gr < m && gc < k) v = ta ? A[(long long)gc * lda + gr] : A[(long long)gr * lda + gc];
            As[r][c] = v;
            int br = k0 + r, bc = c0 + c;
            double u = 0.;
            if (br < k && bc < n) u = tb ? B[(long long)bc * ldb + br] : B[(long long)br * ldb + bc];
            Bs[r][c] = u;
        }
        __syncthreads();
        const int kmax = min(TS, k - k0);
        #pragma unroll 8
        for (int kk = 0; kk < kmax; ++kk) {
            double a0 = As[ty * 2][kk], a1 = As[ty * 2 + 1][kk];
            double b0 = Bs[kk][tx * 2], b1 = Bs[kk][tx * 2 + 1];
            acc00 = fma(a0, b0, acc00); acc01 = fma(a0, b1, acc01);
            acc10 = fma(a1, b0, acc10); acc11 = fma(a1, b1, acc11);
        }
        __syncthreads();
    }
    const double al = alpha[t];
    const int rr = r0 + ty * 2, cc = c0 + tx * 2;
    double vals[4] = {acc00, acc01, acc10, acc11};
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        int r = rr + (q >> 1), c = cc + (q & 1);
        if (r < m && c < n) {
            double v = al * vals[q];
            if (accumulate) atomicAdd(&C[(long long)r * ldc + c], v);
            else C[(long long)r * ldc + c] = v;
        }
    }
}

// s[t] = sum_{r<m, c<n} A[r, c] * B[r, c]   (A, B with their own ld)
extern "C" __global__ void vdot(const long long* __restrict__ Ap,
                                const long long* __restrict__ Bp,
                                const int* __restrict__ dims,      // m n lda ldb
                                double* __restrict__ out)
{
    const int t = blockIdx.x;
    const int m = dims[4 * t], n = dims[4 * t + 1], lda = dims[4 * t + 2], ldb = dims[4 * t + 3];
    const double* A = (const double*)Ap[t];
    const double* B = (const double*)Bp[t];
    double s = 0.;
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        int r = idx / n, c = idx % n;
        s += A[(long long)r * lda + c] * B[(long long)r * ldb + c];
    }
    __shared__ double red[256];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) red[threadIdx.x] += red[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[t] = red[0];
}
'''

_mod = None


def _kern(name):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_SRC, options=('--std=c++14',))
    return _mod.get_function(name)


def _shapes(lay):
    sh = getattr(lay, '_shp_arr', None)
    if sh is None:
        sh = lay._shp_arr = np.array(lay.shapes, np.int64).reshape(-1, 2)
    return sh


#: a plan's index tables stay on the host (uploaded chunk by chunk when it
#: runs) once building it on the device would leave less than this fraction
#: of the device free (compact proteins: 10^8 open-shell CC terms)
HOST_HEADROOM = 0.35


def _host_plan(nbytes):
    env = os.environ.get('DLPNO_PLAN_HOST', 'auto')
    if env in ('0', '1'):                               # testing
        return env == '1'
    # whatever its size: the open-shell CC builds 10^5 small chunk plans
    free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
    return free - nbytes < HOST_HEADROOM * cp.cuda.Device().mem_info[1]


def _pinned(a):
    out = cupyx.empty_pinned(a.shape, a.dtype)
    out[...] = a
    return out


class _Op:
    """Static description of one chain operand over the term list."""

    def __init__(self, o, host=False):
        lay = o.layout
        idx = o.idx
        self.trans = bool(o.trans)
        self.group = lay.group[idx]
        shp = _shapes(lay)
        pr, pc = shp[self.group, 0], shp[self.group, 1]
        rows, cols = lay.rows[idx], lay.cols[idx]
        off = lay.slot[idx] * pr * pc
        if o.row is not None:
            off = off + o.row * pc
            rows = np.ones_like(rows)
        self.off = off                     # element offset within the group buffer
        self.ld = pc
        self.m = cols if self.trans else rows   # effective shape
        self.n = rows if self.trans else cols
        self.ngroups = lay.ngroups
        if host:
            self.off, self.group = _pinned(off), _pinned(self.group)
            self.goff = self.gg = None
        else:
            self.goff = cp.asarray(off)
            self.gg = cp.asarray(self.group)

    def ptrs(self, rag, sl):
        """element pointers of terms sl of a host-resident operand"""
        return _base(rag)[cp.asarray(self.group[sl])] + cp.asarray(self.off[sl]) * 8


_BASE = {}


def _base(rag):
    """device array of the group buffer addresses of a Ragged (cached by
    address tuple: a host->device copy synchronises the stream)."""
    key = tuple(d.data.ptr for d in rag.data)
    b = _BASE.get(key)
    if b is None:
        if len(_BASE) > 4096:
            _BASE.clear()
        b = _BASE[key] = cp.asarray(np.array(key, np.int64))
    return b


def _ptrs(rag, op):
    """device array of element pointers for an operand (cached per buffer
    addresses: persistent buffers cost nothing after the first call)."""
    key = tuple(d.data.ptr for d in rag.data)
    c = op.__dict__.setdefault('_pc', {})
    p = c.get(key)
    if p is None:
        if len(c) >= 2:
            c.clear()
        p = c[key] = _base(rag)[op.gg] + op.goff * 8
    return p


class VChainPlan:
    #: largest total of chain intermediates per launch (doubles)
    MAX_TMP = 1 << 27

    def __init__(self, out_layout, out_idx, operands, out_row=None, order=None, host=None):
        self.out_layout = out_layout
        out_idx = np.asarray(out_idx, np.int64)
        self.nterms = nt = len(out_idx)
        k = len(operands)
        if host is None:
            host = _host_plan(nt * (16. * k + 16 + 32 * max(k - 1, 0)))
        self.host = bool(host) and nt > 0
        self.ops = [_Op(o, self.host) for o in operands]
        if nt == 0:
            return
        for a, b in zip(self.ops[:-1], self.ops[1:]):
            assert np.all(a.n == b.m), 'inner dimensions'
        # output
        lay = out_layout
        g = lay.group[out_idx]
        shp = _shapes(lay)
        pr, pc = shp[g, 0], shp[g, 1]
        off = lay.slot[out_idx] * pr * pc
        if out_row is not None:
            off = off + np.asarray(out_row, np.int64) * pc
        self.out = (_pinned(g), _pinned(off), pc) if self.host else (cp.asarray(g), cp.asarray(off), pc)
        self.out_row = out_row is not None
        m_out, n_out = self.ops[0].m, self.ops[-1].n
        if self.out_row:
            # a (P x 1) product written into a row
            assert np.all(n_out == 1) or np.all(m_out == 1)
        # association order
        k = len(self.ops)
        if order is None:
            mm = [o.m.mean() for o in self.ops] + [self.ops[-1].n.mean()]
            l2r = sum(mm[0] * mm[s + 1] * mm[s + 2] for s in range(k - 1))
            r2l = sum(mm[s] * mm[s + 1] * mm[k] for s in range(k - 1))
            order = 'l2r' if l2r <= r2l else 'r2l'
        self.order = order
        # temporaries: per step the product shape
        self.steps = []
        if k == 1:
            self.steps.append(('copy',))
            return
        if order == 'l2r':
            cur_m, cur_n = self.ops[0].m, self.ops[0].n
            for s in range(1, k):
                nm, nn, kk = cur_m, self.ops[s].n, cur_n
                self.steps.append((s, nm, nn, kk))
                cur_n = nn
        else:
            cur_m, cur_n = self.ops[k - 1].m, self.ops[k - 1].n
            for s in range(k - 2, -1, -1):
                nm, nn, kk = self.ops[s].m, cur_n, cur_m
                self.steps.append((s, nm, nn, kk))
                cur_m = nm
        sizes = [st[1] * st[2] for st in self.steps[:-1]]
        # terms run in chunks whose intermediates fit MAX_TMP doubles
        per = np.sum(sizes, axis=0) if sizes else np.zeros(nt, np.int64)
        bounds = [0]
        acc = 0
        cap = (1 << 22) if self.host else nt + 1     # host tables: bounded uploads
        for t_, v in enumerate(per):
            if (acc + v > self.MAX_TMP and t_ > bounds[-1]) or t_ - bounds[-1] >= cap:
                bounds.append(t_)
                acc = 0
            acc += int(v)
        bounds.append(nt)
        self.chunks = []
        for t0, t1 in zip(bounds[:-1], bounds[1:]):
            offs = [np.concatenate([[0], np.cumsum(sz[t0:t1])[:-1]]).astype(np.int64) * 8
                    for sz in sizes]
            if not self.host:
                offs = [cp.asarray(o_) for o_ in offs]
            self.chunks.append((t0, t1, offs, [int(sz[t0:t1].sum()) for sz in sizes]))
        self.dims = []
        self.tiles = []
        last = len(self.steps) - 1
        for q, (s, nm, nn, kk) in enumerate(self.steps):
            if self.order == 'l2r':
                lda = self.ops[0].ld if q == 0 else self.steps[q - 1][2]
                ldb = self.ops[s].ld
            else:
                lda = self.ops[s].ld
                ldb = self.ops[k - 1].ld if q == 0 else self.steps[q - 1][2]
            ldc = pc if q == last else nn
            if self.out_row and q == last:
                ldc = np.ones_like(nn) if np.all(nn == 1) else pc
            d = np.stack([nm, nn, kk, np.broadcast_to(lda, nm.shape),
                          np.broadcast_to(ldb, nm.shape), np.broadcast_to(ldc, nm.shape)], 1)
            d = d.astype(np.int32)
            self.dims.append(_pinned(d) if self.host else cp.asarray(d))
            tn = int(((nn + 31) // 32).max())
            tm = int(((nm + 31) // 32).max())
            self.tiles.append((tm, tn))
        if self.out_row and np.all(self.steps[-1][2] == 1):
            # product is a column (m x 1); write it as a row: stride 1 along m
            if self.host:
                self.dims[-1][:, 5] = 1
            else:
                d = self.dims[-1].get()
                d[:, 5] = 1
                self.dims[-1] = cp.asarray(d)

    def run(self, out, rags, w=None, alpha=1.):
        if self.nterms == 0:
            return out
        if self.host:
            return self._run_host(out, rags, w, alpha)
        kern = _kern('vgemm')
        nt = self.nterms
        ptrs = [_ptrs(r, o) for r, o in zip(rags, self.ops)]
        og, ooff, _ = self.out
        optr = _base(out)[og] + ooff * 8
        if w is None:
            if getattr(self, '_al', None) is None or self._al[0] != alpha:
                self._al = (alpha, cp.full(nt, float(alpha)))
            al = self._al[1]
        else:
            al = (w * alpha).astype(cp.float64)
        if getattr(self, '_ones', None) is None:
            self._ones = cp.ones(nt)
        ones = self._ones
        k = len(self.ops)
        last = len(self.steps) - 1
        for t0, t1, offs, tots in self.chunks:
            tmps = [cp.empty(max(t, 1)) for t in tots]
            tptr = [tmps[q].data.ptr + offs[q] for q in range(len(tmps))]
            sl = slice(t0, t1)
            for q, (s, nm, nn, kk) in enumerate(self.steps):
                if self.order == 'l2r':
                    A = ptrs[0][sl] if q == 0 else tptr[q - 1]
                    ta = self.ops[0].trans if q == 0 else False
                    B, tb = ptrs[s][sl], self.ops[s].trans
                else:
                    A, ta = ptrs[s][sl], self.ops[s].trans
                    B = ptrs[k - 1][sl] if q == 0 else tptr[q - 1]
                    tb = self.ops[k - 1].trans if q == 0 else False
                C = optr[sl] if q == last else tptr[q]
                tm, tn = self.tiles[q]
                kern((t1 - t0, tm * tn), (16, 16),
                     (A, B, C, self.dims[q][sl], (al if q == last else ones)[sl], np.int32(tn),
                      np.int32(ta), np.int32(tb), np.int32(1 if q == last else 0)))
        return out


    def _run_host(self, out, rags, w, alpha):
        """run() with the index tables on the host: each chunk's slice
        uploaded (and its pointers formed) right before its launches."""
        kern = _kern('vgemm')
        og, ooff, _ = self.out
        bo = _base(out)
        k = len(self.ops)
        last = len(self.steps) - 1
        for t0, t1, offs, tots in self.chunks:
            sl = slice(t0, t1)
            n_ = t1 - t0
            ptrs = [o.ptrs(r, sl) for r, o in zip(rags, self.ops)]
            optr = bo[cp.asarray(og[sl])] + cp.asarray(ooff[sl]) * 8
            al = cp.full(n_, float(alpha)) if w is None else (w[sl] * alpha).astype(cp.float64)
            ones = cp.ones(n_)
            tmps = [cp.empty(max(t, 1)) for t in tots]
            tptr = [tmps[q].data.ptr + cp.asarray(offs[q]) for q in range(len(tmps))]
            for q, (s, nm, nn, kk) in enumerate(self.steps):
                if self.order == 'l2r':
                    A = ptrs[0] if q == 0 else tptr[q - 1]
                    ta = self.ops[0].trans if q == 0 else False
                    B, tb = ptrs[s], self.ops[s].trans
                else:
                    A, ta = ptrs[s], self.ops[s].trans
                    B = ptrs[k - 1] if q == 0 else tptr[q - 1]
                    tb = self.ops[k - 1].trans if q == 0 else False
                C = optr if q == last else tptr[q]
                tm, tn = self.tiles[q]
                kern((n_, tm * tn), (16, 16),
                     (A, B, C, cp.asarray(self.dims[q][sl]), (al if q == last else ones), np.int32(tn),
                      np.int32(ta), np.int32(tb), np.int32(1 if q == last else 0)))
        return out


class VDotPlan:
    """s_t = sum( A[a_t] * chain_t ) with A a single operand and the chain a
    VChainPlan evaluated into a temporary."""

    def __init__(self, left, right):
        lo = left[0]
        n = len(lo.idx)
        self.n = n
        self.lop = _Op(lo)
        # temporary output layout for the right chain: dense (m x n) per term
        from electrolites.dlpno.ragged import Layout
        r0, rl = _Op(right[0]), _Op(right[-1])
        self.tlay = Layout(r0.m, rl.n)
        self.chain = VChainPlan(self.tlay, np.arange(n), right)
        g = self.tlay.group
        pc = _shapes(self.tlay)[g, 1]
        d = np.stack([self.lop.m, self.lop.n, self.lop.ld, pc], 1).astype(np.int32)
        assert self.lop.trans is False
        self.ddims = cp.asarray(d)
        self.tptr_op = _Op(type('o', (), dict(layout=self.tlay, idx=np.arange(n),
                                               trans=False, row=None))())

    def run(self, lrags, rrags):
        from electrolites.dlpno.ragged import Ragged
        tmp = Ragged(self.tlay)
        self.chain.run(tmp, rrags)
        out = cp.zeros(self.n)
        _kern('vdot')((self.n,), (256,), (_ptrs(lrags[0], self.lop),
                                         _ptrs(tmp, self.tptr_op), self.ddims, out))
        return out
