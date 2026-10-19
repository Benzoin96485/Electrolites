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


def _scatter_kernel():
    global _sk
    if _sk is None:
        _sk = cp.RawKernel(_SCATTER, 'scatter_rows')
    return _sk


class OverlapGen:
    """pao: PAO index arrays per pair; X: per pair (npao_p + ne, nv_p)
    coefficients (PAO rows, then ne extra identity-metric rows)."""

    def __init__(self, S_pao_d, pao, X, ne=0, max_bytes=1 << 29):
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

    def _src_ptrs(self, pairs):
        base = np.array([d.data.ptr for d in self.Xc.data], np.int64)
        return base[self.LXc.group[pairs]] + self.c_off[pairs] * 8

    def chunks(self, pp, qq):
        """split couple indices into chunks (sorted by q); returns
        [(couple ids, U, involved pairs)].  Vectorised greedy split: a chunk
        grows while 16 x (pairs) x (union PAO bound + ne) x max(nv) stays
        within max_bytes."""
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
            npc = np.cumsum(isnew.reshape(W, 2).sum(axis=1))
            nuc = np.minimum(np.cumsum((self.npao[ids] * isnew).reshape(W, 2).sum(axis=1)), nbf)
            est = 16. * npc * (nuc + ne) * Vm
            over = np.flatnonzero(est > self.max_bytes)
            L = int(over[0]) if len(over) else W
            L = max(1, L)
            if not len(over) and W < N - start:
                guess *= 2
            sel = order[start:start + L]
            inv = np.unique(np.concatenate([P_[start:start + L], Q_[start:start + L]]))
            mask = np.zeros(nbf, bool)
            for r in inv:
                mask[self.pao[r]] = True
            out.append((sel, np.flatnonzero(mask), inv))
            start += L
        return out

    def prepare(self, pp, qq, out_layout=None, out_idx=None, parts=None):
        """static chunk plans for couples (pp, qq), written to items out_idx
        of a Ragged with out_layout, or (default) to a per-chunk Ragged
        P['LS'] whose item i is couple P['sel'][i]."""
        pp, qq = np.asarray(pp, np.int64), np.asarray(qq, np.int64)
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

    def run(self, P, out=None):
        """evaluate one prepared chunk into ``out`` (accumulates: zero it first),
        by default a new Ragged(P['LS']), which is returned."""
        if out is None:
            # persistent output workspace, zeroed per chunk (consumed before
            # the next chunk is generated on this stream): no allocation per call
            LS = P['LS']
            nout = LS.nbytes() // 8
            ows = getattr(self, '_ows', None)
            if ows is None or ows.size < nout:
                self._ows = ows = cp.empty(max(nout, 1))
            ows[:nout].fill(0.)
            out, _ = self._views(LS, ows)
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
            P['_ws'] = ws.data.ptr
        XU, o = self._views(LXu, ws)
        WU, _ = self._views(LXu, ws, o)
        ws[:need].fill(0.)
        dst = P['_dst']
        posU = cp.full(self.nbf, -1, dtype=cp.int32)
        posU[P['U']] = cp.arange(P['nU'], dtype=cp.int32)
        _scatter_kernel()((len(P['inv']),), (256,),
                          (P['src'], dst, P['lds'], P['ldd'], P['npao'], P['ncol'], P['off'],
                           self.pao_flat, posU, np.int32(P['nU']), np.int32(self.ne)))
        pr = LXu.shapes[0][0]
        M = cp.zeros((pr, pr))
        U = P['U']
        nU = P['nU']
        M[:nU, :nU] = self.S[U[:, None], U[None, :]]
        if self.ne:
            M[nU:nU + self.ne, nU:nU + self.ne] = cp.eye(self.ne)
        for d, wd in zip(XU.data, WU.data):
            cp.matmul(M, d, out=wd)
        P['plan'].run(out, [XU, WU])
        return out
