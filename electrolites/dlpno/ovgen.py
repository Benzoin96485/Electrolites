"""PNO overlaps generated on the device, chunk by chunk.

S(p, q) = X_p^T S_pao[d_p, d_q] X_q (+ Y_p^T Y_q for extra identity-metric
rows Y, e.g. the SOMO part of beta virtuals) is needed for millions of
couples (p, q) in the LMP2 and CC equations; storing them all scales as
npair x (domain size) x nv^2 and does not fit a GPU beyond a few dozen atoms.
``OverlapGen`` keeps only each pair's compact coefficients (its PAO rows)
on the device and, for one chunk of couples at a time, embeds the involved
pairs into the union of their PAO domains (a scatter kernel), forms
W = S_pao[U, U] X_U and evaluates every couple of the chunk with one
variable-size batched GEMM.  Callers run their chunk's terms right after.
"""
import os
import time

import numpy as np
import cupy as cp

from electrolites.dlpno.ragged import Layout, Ragged, op, padded
from electrolites.dlpno.vbatch import VChainPlan

_SCATTER = r'''
// dst[r][posU[pao[off[r] + i]], c] = src[r][i, c]   (i < np[r]);  extra rows
// src[r][np[r] + e, c] -> dst[r][nU + e, c]          (e < ne)
extern "C" __global__ void scatter_rows(const long long* __restrict__ srcp, const long long* __restrict__ dstp,
                                        const int* __restrict__ lds, const int* __restrict__ ldd,
                                        const int* __restrict__ npao, const int* __restrict__ ncol,
                                        const long long* __restrict__ off, const int* __restrict__ pao,
                                        const int* __restrict__ posU, const int nU, const int ne)
{
    const int r = blockIdx.x;
    const double* src = (const double*)srcp[r];
    double* dst = (double*)dstp[r];
    const int n = npao[r], nc = ncol[r], ls = lds[r], ld = ldd[r];
    const long long o = off[r];
    for (int e = threadIdx.x; e < (n + ne) * nc; e += blockDim.x) {
        const int i = e / nc, c = e % nc;
        const int drow = (i < n) ? posU[pao[o + i]] : nU + (i - n);
        dst[(long long)drow * ld + c] = src[(long long)i * ls + c];
    }
}
'''
_sk = None

_GATHER_W = r'''
// dst[r][i, c] = W_r[U[i], c] (i < nU);  dst[r][nU + e, c] = W_r[nbf + e, c] (e < ne)
extern "C" __global__ void gather_w(const long long* __restrict__ srcp, const long long* __restrict__ dstp,
                                    const int* __restrict__ lds, const int* __restrict__ ldd,
                                    const int* __restrict__ ncol, const int* __restrict__ U,
                                    const int nU, const int nbf, const int ne)
{
    const int r = blockIdx.x;
    const double* src = (const double*)srcp[r];
    double* dst = (double*)dstp[r];
    const int nc = ncol[r], ls = lds[r], ld = ldd[r];
    const long long tot = (long long)(nU + ne) * nc;
    for (long long e = (long long)blockIdx.y * blockDim.x + threadIdx.x; e < tot;
         e += (long long)gridDim.y * blockDim.x) {
        const int i = (int)(e / nc), c = (int)(e % nc);
        const int srow = (i < nU) ? U[i] : nbf + (i - nU);
        dst[(long long)i * ld + c] = src[(long long)srow * ls + c];
    }
}
'''
_gwk = None

# S(p, q) = X_p^T W_q[rows of p]: X_p compact (its PAO rows, then the extra
# rows), W_q the per-pair table over all PAO rows (+ extra rows).  One block
# per (couple, 32 x 32 output tile), 16 x 16 threads with 2 x 2 outputs each.
_COUPLE = r'''
#define TS 32
extern "C" __global__ void couple_ov(const long long* __restrict__ Xp, const int* __restrict__ ldx,
                                     const int* __restrict__ kp, const int* __restrict__ mp,
                                     const long long* __restrict__ Wq, const int* __restrict__ ldw,
                                     const int* __restrict__ nq, const long long* __restrict__ poff,
                                     const int* __restrict__ npao, const int* __restrict__ pao,
                                     const int nbf, const long long* __restrict__ Op,
                                     const int* __restrict__ ldo, const int ntn)
{
    const int t = blockIdx.x;
    const int m = mp[t], n = nq[t], k = kp[t], np_ = npao[t];
    const int tm = blockIdx.y / ntn, tn = blockIdx.y % ntn;
    const int r0 = tm * TS, c0 = tn * TS;
    if (r0 >= m || c0 >= n) return;
    const double* __restrict__ X = (const double*)Xp[t];
    const double* __restrict__ W = (const double*)Wq[t];
    const int* __restrict__ rows = pao + poff[t];
    const int lx = ldx[t], lw = ldw[t];
    __shared__ double As[TS][TS + 1];      // As[kk][a] = X[k0 + kk, r0 + a]
    __shared__ double Bs[TS][TS + 1];      // Bs[kk][b] = W[row(k0 + kk), c0 + b]
    const int tx = threadIdx.x, ty = threadIdx.y, lid = ty * 16 + tx;
    double acc00 = 0., acc01 = 0., acc10 = 0., acc11 = 0.;
    for (int k0 = 0; k0 < k; k0 += TS) {
        for (int idx = lid; idx < TS * TS; idx += 256) {
            const int kk = idx / TS, c = idx % TS;
            const int kr = k0 + kk;
            double a = 0., b = 0.;
            if (kr < k) {
                if (r0 + c < m) a = X[(long long)kr * lx + r0 + c];
                if (c0 + c < n) {
                    const int wr = (kr < np_) ? rows[kr] : nbf + (kr - np_);
                    b = W[(long long)wr * lw + c0 + c];
                }
            }
            As[kk][c] = a;
            Bs[kk][c] = b;
        }
        __syncthreads();
        const int kmax = min(TS, k - k0);
        #pragma unroll 8
        for (int kk = 0; kk < kmax; ++kk) {
            const double a0 = As[kk][ty * 2], a1 = As[kk][ty * 2 + 1];
            const double b0 = Bs[kk][tx * 2], b1 = Bs[kk][tx * 2 + 1];
            acc00 = fma(a0, b0, acc00); acc01 = fma(a0, b1, acc01);
            acc10 = fma(a1, b0, acc10); acc11 = fma(a1, b1, acc11);
        }
        __syncthreads();
    }
    double* O = (double*)Op[t];
    const int lo = ldo[t];
    const int rr = r0 + ty * 2, cc = c0 + tx * 2;
    double v[4] = {acc00, acc01, acc10, acc11};
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int r = rr + (q >> 1), c = cc + (q & 1);
        if (r < m && c < n) O[(long long)r * lo + c] += v[q];
    }
}
'''
_cok = None


def _couple_kernel():
    global _cok
    if _cok is None:
        _cok = cp.RawKernel(_COUPLE, 'couple_ov')
    return _cok

class _WTables:
    __slots__ = ('w', 'ptr', 'ld', '__weakref__')


#: W tables shared by the live generators of the same pairs and coefficients
#: (one per spin: the per-device generators of the kl terms reuse them);
#: freed with the last generator holding them
import weakref
_WCACHE = weakref.WeakValueDictionary()


def _gather_w_kernel():
    global _gwk
    if _gwk is None:
        _gwk = cp.RawKernel(_GATHER_W, 'gather_w')
    return _gwk

#: chunk statistics with DLPNO_PROF=1 (synchronises): chunks, embedded
#: pairs, union rows, W GEMM flops and seconds, couples, total seconds
STATS = {}
_PROF = os.environ.get('DLPNO_PROF') == '1'


def stats_report(reset=True):
    if STATS:
        s = STATS
        print(f"[ovgen] {s['runs']} chunks, {s['pairs']} pair embeddings, mean union {s.get('rows', 0) / s['runs']:.0f} "
              f"rows, W GEMM {s.get('flops', 0) / 1e12:.1f} TF in {s.get('t_w', 0):.1f} s, {s['couples']} couples, "
              f"{s['t']:.1f} s total", flush=True)
        if reset:
            STATS.clear()


def _scatter_kernel():
    global _sk
    if _sk is None:
        _sk = cp.RawKernel(_SCATTER, 'scatter_rows')
    return _sk


class OverlapGen:
    """pao: PAO index arrays per pair; X: per pair (npao_p + ne, nv_p)
    coefficients (PAO rows, then ne extra identity-metric rows)."""

    def __init__(self, S_pao_d, pao, X, ne=0, max_bytes=None):
        if max_bytes is None:
            # per-chunk workspace (DLPNO_OVGEN_MB): fewer, larger chunks keep the
            # per-chunk setup (plans, small uploads) off the critical path
            import os
            max_bytes = int(float(os.environ.get('DLPNO_OVGEN_MB', '1024')) * 2 ** 20)
        self.S = S_pao_d
        self.nbf = S_pao_d.shape[0]
        self.pao = pao
        self.ne = ne
        self.max_bytes = max_bytes
        npair = len(X)
        self.npao = np.array([len(p) for p in pao], np.int64)
        self.nv = np.array([x.shape[1] for x in X], np.int64)
        self.LXc = Layout(self.npao + ne, self.nv)
        self.Xc = Ragged(self.LXc)
        for g, m in enumerate(self.LXc.members):
            if not len(m):
                continue
            pr, pc = self.LXc.shapes[g]
            buf = np.zeros((len(m), pr, pc))
            for s, p in enumerate(m):
                x = X[p].get() if hasattr(X[p], 'get') else X[p]
                buf[s, :x.shape[0], :x.shape[1]] = x
            self.Xc.data[g] = cp.asarray(buf)
        self.off = np.concatenate([[0], np.cumsum(self.npao)[:-1]]).astype(np.int64)
        self.pao_flat = cp.asarray(np.concatenate([np.asarray(p, np.int64) for p in pao]).astype(np.int32))
        shp = np.array(self.LXc.shapes, np.int64).reshape(-1, 2)
        self.c_ld = shp[self.LXc.group, 1]
        self.c_off = self.LXc.slot * shp[self.LXc.group, 0] * shp[self.LXc.group, 1]

    def _w_tables(self):
        """W_q = S_pao[:, d_q] X_q over all PAO rows (+ the extra rows of
        X_q) for every pair, once: a chunk then gathers its union rows of W
        instead of forming S_UU X_U (nU^2 flops per coefficient; the overlap
        generation was ~70% of a compact protein's CC iteration).  Placed on
        the devices with room (read by pointer, peer access); None when they
        do not fit (DLPNO_OVGEN_W=auto|0|1)."""
        st = getattr(self, '_w', 'unset')
        if st != 'unset':
            return st
        if not hasattr(self, '_wkey'):
            self._wkey = (len(self.nv), self.ne, self.nbf, int(self.npao.sum()), int(self.nv.sum()),
                          tuple(float(d.sum()) for d in self.Xc.data if d is not None))
        hit = _WCACHE.get(self._wkey)
        if hit is not None:
            self._wt = hit
            self._w, self._w_ptr, self._w_ld = hit.w, hit.ptr, hit.ld
            return self._w
        self._w = None
        env = os.environ.get('DLPNO_OVGEN_W', 'auto')
        if env == '0':
            return None
        from electrolites.dlpno import mgpu
        nr = self.nbf + self.ne
        LW = Layout(np.full(len(self.nv), nr), self.nv)
        home = cp.cuda.Device().id
        devs = mgpu.devices() if home in mgpu.devices() else [home]
        free = {}
        for d in devs:
            with cp.cuda.Device(d):
                free[d] = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        frac = 1.0 if env == '1' else 0.4
        # the home device runs the chunks (workspaces, plans, DF / 3-external
        # groups): it takes a smaller share
        left = {d: frac * free[d] * (0.5 if d == home and len(devs) > 1 else 1.) for d in devs}
        place = []
        for g, (m, (r, c)) in enumerate(zip(LW.members, LW.shapes)):
            nb = 8. * len(m) * r * c
            d = max(devs, key=lambda x: left[x])
            if left[d] < nb:
                return None
            left[d] -= nb
            place.append(d)
        if len(set(place)) > 1 or place[0] != home:
            mgpu.enable_peer(devs)
        data = []
        for g, (m, (r, c)) in enumerate(zip(LW.members, LW.shapes)):
            with cp.cuda.Device(place[g]):
                data.append(cp.zeros((len(m), r, c)))
        S = self.S
        for g, m in enumerate(LW.members):
            for s, q in enumerate(m):
                q = int(q)
                npo, nv = int(self.npao[q]), int(self.nv[q])
                xq = self.Xc.data[self.LXc.group[q]][self.LXc.slot[q]]
                w = cp.matmul(S[:, cp.asarray(self.pao[q])], xq[:npo, :nv])
                if self.ne:
                    w = cp.concatenate([w, xq[npo:npo + self.ne, :nv]])
                if place[g] == home:
                    data[g][s, :nr, :nv] = w
                else:
                    with cp.cuda.Device(place[g]):
                        data[g][s, :nr, :nv] = cp.asarray(w)
        cp.cuda.Device().synchronize()
        self._w = (LW, Ragged(LW, data))
        shp = np.array(LW.shapes, np.int64).reshape(-1, 2)
        base = np.array([d.data.ptr for d in data], np.int64)
        self._w_ptr = base[LW.group] + LW.slot * shp[LW.group, 0] * shp[LW.group, 1] * 8
        self._w_ld = shp[LW.group, 1].astype(np.int32)
        self._wt = h = _WTables()
        h.w, h.ptr, h.ld = self._w, self._w_ptr, self._w_ld
        _WCACHE[self._wkey] = h
        if os.environ.get('V', '0') >= '2':
            print(f'[mem] overlap W tables {LW.nbytes() / 1e9:.1f} GB on GPUs {sorted(set(place))}', flush=True)
        return self._w

    def _pao_rows(self, pairs):
        """the PAO indices of all the given pairs, concatenated (with repeats)"""
        if not hasattr(self, '_pao_np'):
            self._pao_np = np.concatenate([np.asarray(p, np.int64) for p in self.pao])
        n = self.npao[pairs]
        st = np.repeat(self.off[pairs] - np.concatenate([[0], np.cumsum(n)[:-1]]), n)
        return self._pao_np[st + np.arange(int(n.sum()))]

    def _src_ptrs(self, pairs):
        base = np.array([d.data.ptr for d in self.Xc.data], np.int64)
        return base[self.LXc.group[pairs]] + self.c_off[pairs] * 8

    def chunks(self, pp, qq):
        """split couple indices into chunks (sorted by q); returns
        [(couple ids, U, involved pairs)].  Vectorised greedy split: a chunk
        grows while its embedded coefficients and W = S_UU X_U, 16 x (union
        PAO bound + ne) x (sum of the involved pairs' nv), and its overlaps,
        8 x sum nv_p nv_q, stay within max_bytes."""
        pp, qq = np.asarray(pp, np.int64), np.asarray(qq, np.int64)
        N = len(pp)
        if N == 0:
            return []
        order = np.argsort(qq, kind='stable')
        P_, Q_ = pp[order], qq[order]
        ne, Vm = self.ne, int(self.nv.max()) if len(self.nv) else 1
        nbf = self.nbf
        out = []
        start = 0
        guess = max(64, int(self.max_bytes // (16 * (nbf + ne) * Vm)) * 8)
        while start < N:
            W = min(N - start, guess)
            ids = np.stack([P_[start:start + W], Q_[start:start + W]], 1).ravel()
            _, first = np.unique(ids, return_index=True)
            isnew = np.zeros(2 * W, np.int64)
            isnew[first] = 1
            nvc = np.cumsum((self.nv[ids] * isnew).reshape(W, 2).sum(axis=1))
            nuc = np.minimum(np.cumsum((self.npao[ids] * isnew).reshape(W, 2).sum(axis=1)), nbf)
            nout = np.cumsum(self.nv[P_[start:start + W]] * self.nv[Q_[start:start + W]])
            est = np.maximum(16. * nvc * (nuc + ne), 8. * nout)
            over = np.flatnonzero(est > self.max_bytes)
            L = int(over[0]) if len(over) else W
            L = max(1, L)
            if not len(over) and W < N - start:
                guess *= 2
            sel = order[start:start + L]
            inv = np.unique(np.concatenate([P_[start:start + L], Q_[start:start + L]]))
            mask = np.zeros(nbf, bool)
            mask[self._pao_rows(inv)] = True
            out.append((sel, np.flatnonzero(mask), inv))
            start += L
        return out

    def prepare(self, pp, qq, out_layout=None, out_idx=None, parts=None):
        """static chunk plans for couples (pp, qq), written to items out_idx
        of a Ragged with out_layout, or (default) to a per-chunk Ragged
        P['LS'] whose item i is couple P['sel'][i]."""
        pp, qq = np.asarray(pp, np.int64), np.asarray(qq, np.int64)
        if out_layout is not None and parts is None and len(pp) \
                and os.environ.get('DLPNO_OVGEN_DIRECT', '1') == '1' and self._w_tables() is not None:
            # direct couples from the W tables: no union workspace, so one
            # launch for all the couples (10^5 chunks per CC iteration before)
            idx_o = np.asarray(out_idx, np.int64)
            return [dict(direct=True, sel=np.arange(len(pp)), LS=None, inv=np.unique(np.concatenate([pp, qq])),
                         nterms=len(pp), cpl=(pp, qq, out_layout, idx_o))]
        plans = []
        for sel, U, inv in (parts if parts is not None else self.chunks(pp, qq)):
            if out_layout is None:
                LSc = Layout(self.nv[pp[sel]], self.nv[qq[sel]])
                lay_o, idx_o = LSc, np.arange(len(sel))
            else:
                LSc, lay_o, idx_o = None, out_layout, np.asarray(out_idx, np.int64)[sel]
            nr = len(U) + self.ne
            loc = -np.ones(len(self.nv), np.int64)
            loc[inv] = np.arange(len(inv))
            LXu = Layout(np.full(len(inv), nr), self.nv[inv])
            plan = VChainPlan(lay_o, idx_o, [op(LXu, loc[pp[sel]], True), op(LXu, loc[qq[sel]])])
            shp = np.array(LXu.shapes, np.int64).reshape(-1, 2)
            d_ld = shp[LXu.group, 1]
            d_off = LXu.slot * shp[LXu.group, 0] * shp[LXu.group, 1]
            plans.append(dict(sel=sel, LS=LSc, U=cp.asarray(U), nU=len(U), inv=inv, LXu=LXu, plan=plan,
                              cpl=(pp[sel], qq[sel], lay_o, idx_o),
                              src=cp.asarray(self._src_ptrs(inv)), lds=cp.asarray(self.c_ld[inv].astype(np.int32)),
                              ldd=cp.asarray(d_ld.astype(np.int32)), d_off=d_off,
                              npao=cp.asarray(self.npao[inv].astype(np.int32)),
                              ncol=cp.asarray(self.nv[inv].astype(np.int32)),
                              off=cp.asarray(self.off[inv])))
        return plans

    def _views(self, LXu, ws, off=0):
        data = []
        for m, (r, c) in zip(LXu.members, LXu.shapes):
            n = len(m) * r * c
            data.append(ws[off:off + n].reshape(len(m), r, c))
            off += n
        return Ragged(LXu, data), off

    def run_tmp(self, P):
        """like run(P), into a persistent zeroed workspace of this generator:
        valid only until its next run_tmp (the caller consumes it at once);
        no allocation per call."""
        LS = P['LS']
        nout = LS.nbytes() // 8
        ows = getattr(self, '_ows', None)
        if ows is None or ows.size < nout:
            self._ows = ows = cp.empty(max(nout, 1))
        ows[:nout].fill(0.)
        out, _ = self._views(LS, ows)
        return self.run(P, out)

    def _run_direct(self, P, out):
        """S(p, q) of the chunk straight from the compact coefficients and
        the W tables (only p's own rows; no union workspace)."""
        if '_cd' not in P:
            from electrolites.dlpno.vbatch import _shapes
            pp, qq, lay, idx = P['cpl']
            shp = _shapes(lay)
            g = lay.group[idx]
            k = self.npao[pp] + self.ne
            P['_cd'] = dict(
                x=cp.asarray(self._src_ptrs(pp)), ldx=cp.asarray(self.c_ld[pp].astype(np.int32)),
                k=cp.asarray(k.astype(np.int32)), m=cp.asarray(self.nv[pp].astype(np.int32)),
                w=cp.asarray(self._w_ptr[qq]), ldw=cp.asarray(self._w_ld[qq]),
                n=cp.asarray(self.nv[qq].astype(np.int32)), poff=cp.asarray(self.off[pp]),
                npao=cp.asarray(self.npao[pp].astype(np.int32)),
                og=cp.asarray(g), ooff=cp.asarray(lay.slot[idx] * shp[g, 0] * shp[g, 1]),
                ldo=cp.asarray(shp[g, 1].astype(np.int32)),
                ntm=int(-(-self.nv[pp].max() // 32)), ntn=int(-(-self.nv[qq].max() // 32)), nt=len(pp))
        d = P['_cd']
        from electrolites.dlpno.vbatch import _base
        optr = _base(out)[d['og']] + d['ooff'] * 8
        _couple_kernel()((d['nt'], d['ntm'] * d['ntn']), (16, 16),
                         (d['x'], d['ldx'], d['k'], d['m'], d['w'], d['ldw'], d['n'], d['poff'], d['npao'],
                          self.pao_flat, np.int32(self.nbf), optr, d['ldo'], np.int32(d['ntn'])))
        return out

    def run(self, P, out=None):
        """evaluate one prepared chunk into ``out`` (accumulates: zero it first),
        by default a new Ragged(P['LS']), which is returned."""
        if _PROF:
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
        if out is None:
            out = Ragged(P['LS'])
        if P.get('direct') or ('cpl' in P and os.environ.get('DLPNO_OVGEN_DIRECT', '1') == '1'
                               and self._w_tables() is not None):
            self._run_direct(P, out)
            if _PROF:
                cp.cuda.Device().synchronize()
                nt = P['nterms'] if P.get('direct') else P['plan'].nterms
                for k_, v in (('runs', 1), ('pairs', len(P['inv'])), ('couples', nt),
                              ('t', time.perf_counter() - t0)):
                    STATS[k_] = STATS.get(k_, 0) + v
            return out
        LXu = P['LXu']
        need = LXu.nbytes() // 8
        ws = getattr(self, '_ws', None)
        if ws is None or ws.size < 2 * need:
            # persistent workspace (stable addresses -> cached pointer tables)
            self._ws = ws = cp.empty(2 * need)
        if P.get('_ws') != ws.data.ptr:
            XU, o = self._views(LXu, ws)
            base = np.array([d.data.ptr for d in XU.data], np.int64)
            P['_dst'] = cp.asarray(base[LXu.group] + P['d_off'] * 8)
            WU, _ = self._views(LXu, ws, o)
            base = np.array([d.data.ptr for d in WU.data], np.int64)
            P['_wdst'] = cp.asarray(base[LXu.group] + P['d_off'] * 8)
            P['_ws'] = ws.data.ptr
        XU, o = self._views(LXu, ws)
        WU, _ = self._views(LXu, ws, o)
        wt = self._w_tables()
        ws[:need].fill(0.)
        dst = P['_dst']
        posU = cp.full(self.nbf, -1, dtype=cp.int32)
        posU[P['U']] = cp.arange(P['nU'], dtype=cp.int32)
        _scatter_kernel()((len(P['inv']),), (256,),
                          (P['src'], dst, P['lds'], P['ldd'], P['npao'], P['ncol'], P['off'],
                           self.pao_flat, posU, np.int32(P['nU']), np.int32(self.ne)))
        pr = LXu.shapes[0][0]
        U = P['U']
        nU = P['nU']
        if _PROF:
            cp.cuda.Device().synchronize()
            t1 = time.perf_counter()
        if wt is not None:
            # W rows of the union gathered from the tables
            if '_wsrc' not in P:
                P['_wsrc'] = cp.asarray(self._w_ptr[P['inv']])
                P['_wld'] = cp.asarray(self._w_ld[P['inv']])
                P['_U32'] = U.astype(cp.int32)
            ny = int(min(64, max(1, (nU + self.ne) * int(self.nv[P['inv']].max()) // 4096)))
            _gather_w_kernel()((len(P['inv']), ny), (256,),
                               (P['_wsrc'], P['_wdst'], P['_wld'], P['ldd'], P['ncol'], P['_U32'],
                                np.int32(nU), np.int32(self.nbf), np.int32(self.ne)))
        else:
            M = cp.zeros((pr, pr))
            M[:nU, :nU] = self.S[U[:, None], U[None, :]]
            if self.ne:
                M[nU:nU + self.ne, nU:nU + self.ne] = cp.eye(self.ne)
            for d, wd in zip(XU.data, WU.data):
                cp.matmul(M, d, out=wd)
        if _PROF:
            cp.cuda.Device().synchronize()
            t2 = time.perf_counter()
        P['plan'].run(out, [XU, WU])
        if _PROF:
            cp.cuda.Device().synchronize()
            nv = sum(len(m) * c for m, (r, c) in zip(LXu.members, LXu.shapes))
            for k, v in (('runs', 1), ('pairs', len(P['inv'])), ('rows', pr), ('flops', 2. * pr * pr * nv),
                         ('t_w', t2 - t1), ('couples', P['plan'].nterms), ('t', time.perf_counter() - t0)):
                STATS[k] = STATS.get(k, 0) + v
        return out
