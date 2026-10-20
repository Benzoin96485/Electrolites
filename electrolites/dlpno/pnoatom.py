"""Atom-major PNO integrals: the parts that need (Q|uv) blocks.

The pair-major path (``pnoints._pno_chunk``) evaluates, for each chunk of
pairs, the (Q_a|uv) block of every aux atom a of their fitting domains.  For
compact molecules a chunk's fitting domains span most atoms and a block over
the extended PAO domains is large, so the regions holding the blocks a chunk
needs contain only a few chunks, and each atom's AO integrals are evaluated
(or read back) ~100 times (crambin: 1743 regions, 81k atom blocks for 642
atoms).

Here the pairs are split over the devices (contiguous in Morton order) and
each device loops over the aux atoms its pairs use: one (Q_a|uv) block per
atom over the union of those pairs' PAOs (rows) and PAOs / extended domains
(columns), from which every pair gets its contributions, all of them linear in
the aux functions of a:

    q_vv(Q_a, b, c)        = X_p^T (Q_a|uv) X_p              (strong: Qab rows)
    K_ivvv(b, cd)         += (Q_a|ib)^fit q_vv(Q_a, c, d)
    (ik|b c_kj)           += sum_Q (Q_a|ik)^fit [X_p^T (Q_a|u e) X_kj](b, c)

with the fitted factors (Q|ib)^fit = J^-1 (Q|ib) of the pair's fitting domain
from the pair-major pass.  Qab is half-fitted at the end (L^-1 of the pair
metric).  Same sums as the pair-major path, in another order.
"""
import numpy as np
import cupy as cp

from electrolites.dlpno.gather import blocks as gather_blocks, gather2d
from electrolites.dlpno.linalg import chol_inv_batched
from electrolites.dlpno.ragged import padded

_SCAT = None


def _scatter_add():
    """out[off[r] + v * nc[r] + c] += src[r, v, c] for v < nr[r], c < nc[r]."""
    global _SCAT
    if _SCAT is None:
        _SCAT = cp.RawKernel(r'''
extern "C" __global__ void scat_add(const double* __restrict__ src, const long long* __restrict__ off,
                                    const int* __restrict__ nr, const int* __restrict__ nc,
                                    double* __restrict__ out, const int R, const int C)
{
    const int r = blockIdx.y;
    const long long n = (long long)nr[r] * nc[r];
    const double* s = src + (long long)r * R * C;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
         t += (long long)gridDim.x * blockDim.x) {
        const int v = (int)(t / nc[r]), c = (int)(t % nc[r]);
        out[off[r] + t] += s[(long long)v * C + c];
    }
}
''', 'scat_add')
    return _SCAT


def _tr(Y):
    return Y.swapaxes(-1, -2)


def _owners(obj, canon, devs):
    """canonical pairs split over the devices: contiguous in Morton order of
    their fitting domains, balanced by an estimate of their work."""
    order = obj.dfi.spatial_order([obj.riatoms[p] for p in canon])
    w = []
    for k in order:
        p = canon[k]
        na = sum(len(obj.dfi.atom_ribfs[a]) for a in obj.riatoms[p])
        P, V = len(obj.pao[p]), obj.X[p].shape[1]
        e = len(obj._ext[p]) if obj.strong[p] else 0
        w.append(na * P * (P + e) * V)
    w = np.cumsum(w)
    cut = np.searchsorted(w, w[-1] * np.arange(1, len(devs)) / len(devs))
    parts = np.split(np.asarray(order), cut)
    return {d: [canon[k] for k in part] for d, part in zip(devs, parts)}


def _device_pass(w, pairs, verbose=0):
    """this device's pairs: K_ivvv / K_jvvv, (ik|ac) / (jk|ac) items and the
    strong pairs' raw (Q|ab) rows, from one (Q_a|uv) block per aux atom."""
    import time
    dfi, pl = w.dfi, w.pl
    t0 = time.perf_counter()
    nbf = dfi.nbf
    # --- per pair data
    meta = {}
    for p in pairs:
        offs, o = {}, 0
        for a in w.riatoms[p]:
            offs[int(a)] = o
            o += len(dfi.atom_ribfs[a])
        meta[p] = dict(offs=offs, A=o, V=w.X[p].shape[1], P=len(w.pao[p]), q=int(pl.ji[p]),
                       strong=bool(w.strong[p]))
    # --- accumulators: K_ivvv per pair side, J items, raw Qab (host)
    Kv = {}
    for p in pairs:
        V = meta[p]['V']
        Kv[p] = cp.zeros((V, V * V))
        if meta[p]['q'] != p:
            Kv[meta[p]['q']] = cp.zeros((V, V * V))
    jitems = []                       # (pair, side, kk, target pair)
    jbase = {}
    for p in pairs:
        if not meta[p]['strong']:
            continue
        i, j = pl.pairs[p]
        lm = np.asarray(w.lmos[p], np.int64)
        for side, tg in ((0, pl.ij[lm, j]), (1, pl.ij[i, lm])):
            if side == 1 and meta[p]['q'] == p:
                continue
            jbase[(p, side)] = len(jitems)
            for kk, t in enumerate(tg):
                jitems.append((p, side, kk, int(t)))
    jv = np.array([meta[it[0]]['V'] for it in jitems], np.int64)
    jt = np.array([w.X[it[3]].shape[1] for it in jitems], np.int64)
    joff = np.concatenate([[0], np.cumsum(jv * jt)]).astype(np.int64)
    Jacc = cp.zeros(max(int(joff[-1]), 1))
    Qab_raw = {p: np.zeros((meta[p]['A'], meta[p]['V'], meta[p]['V'])) for p in pairs if meta[p]['strong']}
    # --- atoms, in Morton order
    users = {}
    for p in pairs:
        for a in w.riatoms[p]:
            users.setdefault(int(a), []).append(p)
    atoms = sorted(users)
    atoms = [atoms[k] for k in dfi.spatial_order([[a] for a in atoms])]
    Xs, xptr, xld = w._xcompact()
    st = dict(blocks=0, t_blk=0., t_pair=0., bytes=0.)
    for a in atoms:
        na = len(dfi.atom_ribfs[a])
        if na == 0:
            continue
        for sub in _split_users(w, users[a], na):
            _atom_block(w, a, na, sub, meta, Kv, jitems, jbase, joff, Jacc, Qab_raw, xptr, xld, st)
    cp.cuda.Device().synchronize()
    if verbose >= 2:
        print(f'[mem] atom-major PNO integrals on GPU {cp.cuda.Device().id}: {len(pairs)} pairs, '
              f'{len(atoms)} atoms, {st["blocks"]} blocks ({st["bytes"] / 1e9:.0f} GB) in {st["t_blk"]:.0f} s, '
              f'pair work {st["t_pair"]:.0f} s, total {time.perf_counter() - t0:.0f} s', flush=True)
    return Kv, jitems, joff, Jacc, Qab_raw


def _split_users(w, us, na):
    """the pairs of an aux atom in groups whose (rows x columns) block fits a
    fifth of the free device memory."""
    free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
    lim = 0.2 * free
    nbf = w.dfi.nbf
    out, cur = [], []
    rm, cm = np.zeros(nbf, bool), np.zeros(nbf, bool)
    for p in us:
        r2, c2 = rm.copy(), cm.copy()
        r2[w.pao[p]] = True
        c2[w.pao[p]] = True
        if w.strong[p]:
            c2[w._ext[p]] = True
        if cur and 8. * na * r2.sum() * c2.sum() > lim:
            out.append(cur)
            cur = []
            r2, c2 = np.zeros(nbf, bool), np.zeros(nbf, bool)
            r2[w.pao[p]] = True
            c2[w.pao[p]] = True
            if w.strong[p]:
                c2[w._ext[p]] = True
        cur.append(p)
        rm, cm = r2, c2
    if cur:
        out.append(cur)
    return out


def _atom_block(w, a, na, us, meta, Kv, jitems, jbase, joff, Jacc, Qab_raw, xptr, xld, st):
    import time
    dfi, pl = w.dfi, w.pl
    nbf = dfi.nbf
    t1 = time.perf_counter()
    rows = np.unique(np.concatenate([w.pao[p] for p in us]))
    cols = np.unique(np.concatenate([w.pao[p] if not w.strong[p] else np.union1d(w.pao[p], w._ext[p])
                                     for p in us]))
    aosub = None
    if dfi._pao_sup is not None:
        g = dfi._region_groups([a], {a: (rows, cols)})
        aosub = g[0][1]
    blk, rmap, cmap, _ = dfi._direct_rect(a, rows, cols, cache=False, aosub=aosub)
    st['blocks'] += 1
    st['bytes'] += blk.nbytes
    st['t_blk'] += time.perf_counter() - t1
    t1 = time.perf_counter()
    # pairs in size groups (padded PAO domain, PNOs)
    key = {}
    for p in us:
        key.setdefault((int(padded(meta[p]['P'])), int(padded(meta[p]['V']))), []).append(p)
    for (P, V), grp in key.items():
        step = max(1, int((1 << 28) // (8 * na * P * max(P, V))))
        for c0 in range(0, len(grp), step):
            _pair_batch(w, a, na, grp[c0:c0 + step], P, V, blk, rmap, cmap, meta, Kv,
                        jitems, jbase, joff, Jacc, Qab_raw, xptr, xld)
    del blk
    st['t_pair'] += time.perf_counter() - t1


def _pair_batch(w, a, na, bp, P, V, blk, rmap, cmap, meta, Kv, jitems, jbase, joff, Jacc, Qab_raw,
                xptr, xld):
    pl = w.pl
    b = len(bp)
    pidx, pm = np.zeros((b, P), np.int64), np.zeros((b, P))
    Xh = np.zeros((b, P, V))
    qiv, qjv = np.zeros((b, na, V)), np.zeros((b, na, V))
    for s, p in enumerate(bp):
        n_, v_ = meta[p]['P'], meta[p]['V']
        pidx[s, :n_], pm[s, :n_] = w.pao[p], 1.
        Xh[s, :n_, :v_] = w.X[p]
        o = meta[p]['offs'][a]
        qiv[s, :, :v_] = w.aQiv[p][o:o + na]
        if meta[p]['q'] != p:
            qjv[s, :, :v_] = w.aQjv[p][o:o + na]
    gpm = cp.asarray(pm)
    X = cp.asarray(Xh)
    rpos = cp.asarray(np.where(pm > 0, rmap[pidx], 0))
    cpos = cp.asarray(np.where(pm > 0, cmap[pidx], 0))
    blkp = gather_blocks(blk, rpos, cpos, gpm, gpm, ru=True)                     # (b, P, na, P)
    Y = cp.matmul(_tr(X), blkp.reshape(b, P, na * P))                            # (b, V, na*P)
    del blkp
    Y = cp.matmul(Y.reshape(b, V * na, P), X).reshape(b, V, na, V)
    qvv = cp.ascontiguousarray(Y.transpose(0, 2, 1, 3))                           # (b, na, V, V)
    del Y
    g_iv = cp.asarray(qiv)
    kv = cp.matmul(_tr(g_iv), qvv.reshape(b, na, V * V))                         # (b, V, V*V)
    g_jv = cp.asarray(qjv)
    kq = cp.matmul(_tr(g_jv), qvv.reshape(b, na, V * V))
    for s, p in enumerate(bp):
        v_ = meta[p]['V']
        Kv[p] += kv[s, :v_].reshape(v_, V, V)[:, :v_, :v_].reshape(v_, v_ * v_)
        q = meta[p]['q']
        if q != p:
            Kv[q] += kq[s, :v_].reshape(v_, V, V)[:, :v_, :v_].reshape(v_, v_ * v_)
        if meta[p]['strong']:
            o = meta[p]['offs'][a]
            Qab_raw[p][o:o + na] = cp.asnumpy(qvv[s, :, :v_, :v_])
    del kv, kq
    sp = [s for s, p in enumerate(bp) if meta[p]['strong']]
    if not sp:
        return
    # ---- (ik|b c_kj): extended-domain columns, projected onto the kj PNOs
    sps = [bp[s] for s in sp]
    ns = len(sps)
    L = int(padded(max(len(w.lmos[p]) for p in sps)))
    E = int(padded(max(len(w._ext[p]) for p in sps)))
    eidx, emk = np.zeros((ns, E), np.int64), np.zeros((ns, E))
    qio, qjo = np.zeros((ns, na, L)), np.zeros((ns, na, L))
    for r, p in enumerate(sps):
        e_ = w._ext[p]
        eidx[r, :len(e_)], emk[r, :len(e_)] = e_, 1.
        o = meta[p]['offs'][a]
        nl = len(w.lmos[p])
        qio[r, :, :nl] = w.aQio[p][o:o + na]
        if meta[p]['q'] != p:
            qjo[r, :, :nl] = w.aQjo[p][o:o + na]
    gsp = cp.asarray(np.asarray(sp, np.int64))
    gem = cp.asarray(emk)
    epos = cp.asarray(np.where(emk > 0, cmap[eidx], 0))
    blke = gather_blocks(blk, rpos[gsp], epos, gpm[gsp], gem, ru=True)            # (ns, P, na, E)
    ve = cp.matmul(_tr(X[gsp]), blke.reshape(ns, P, na * E))                      # (ns, V, na*E)
    del blke
    ve = cp.ascontiguousarray(ve.reshape(ns, V, na, E).transpose(0, 2, 1, 3)).reshape(ns, na, V * E)
    Kio = cp.matmul(_tr(cp.asarray(qio)), ve).reshape(ns, L, V, E)
    Kjo = cp.matmul(_tr(cp.asarray(qjo)), ve).reshape(ns, L, V, E)
    del ve
    # items of these pairs: gather the kj PAO columns, times the kj PNOs
    its = []
    for r, p in enumerate(sps):
        pos = -np.ones(w.dfi.nbf, np.int64)
        pos[w._ext[p]] = np.arange(len(w._ext[p]))
        for side in (0, 1):
            if (p, side) not in jbase:
                continue
            b0 = jbase[(p, side)]
            for kk in range(len(w.lmos[p])):
                it = jitems[b0 + kk]
                its.append((r, side, kk, it[3], pos, b0 + kk, meta[p]['V']))
    if not its:
        return
    n_t = np.array([len(w.pao[it[3]]) for it in its])
    v_t = np.array([w.X[it[3]].shape[1] for it in its])
    Pq, Vq = int(n_t.max()), int(v_t.max())
    step = max(1, int((1 << 29) // (8 * (V * E + 2 * Pq * (V + Vq)))))
    for c0 in range(0, len(its), step):
        ch = its[c0:c0 + step]
        m = len(ch)
        colsq = np.zeros((m, Pq), np.int64)
        msk = np.zeros((m, Pq))
        for r_, it in enumerate(ch):
            c_ = it[4][w.pao[it[3]]]
            colsq[r_, :len(c_)] = c_
            msk[r_, :len(c_)] = 1.
        gc, gm = cp.asarray(colsq), cp.asarray(msk)
        tt = np.array([it[3] for it in ch], np.int64)
        Xt = gather2d(cp.asarray(xptr[tt]), cp.asarray(xld[tt]), cp.asarray(n_t[c0:c0 + m].astype(np.int32)),
                      cp.asarray(v_t[c0:c0 + m].astype(np.int32)), Pq, Vq)        # (m, Pq, Vq)
        rr = cp.asarray(np.array([it[0] for it in ch], np.int64))
        kk = cp.asarray(np.array([it[2] for it in ch], np.int64))
        s1 = cp.asarray(np.array([it[1] for it in ch]) == 1)
        Ka = cp.where(s1[:, None, None], Kjo[rr, kk], Kio[rr, kk])                   # (m, V, E)
        Am = cp.take_along_axis(Ka, gc[:, None, :], axis=2) * gm[:, None, :]       # (m, V, Pq)
        Jm = cp.ascontiguousarray(cp.matmul(Am, Xt))                                # (m, V, Vq)
        idx = np.array([it[5] for it in ch], np.int64)
        nr = np.array([it[6] for it in ch], np.int32)
        ncs = v_t[c0:c0 + m].astype(np.int32)
        _scatter_add()((max(1, -(-int((nr * ncs).max()) // 256)), m), (256,),
                       (Jm, cp.asarray(joff[idx]), cp.asarray(nr), cp.asarray(ncs), Jacc,
                        np.int32(V), np.int32(Vq)))


def pno_atom_pass(obj, devs, verbose=0):
    """the (Q|uv)-dependent PNO integrals of every pair, atom-major, over the
    devices; fills obj.K_ivvv, obj.J_ikac, obj.Qab."""
    from electrolites.dlpno import mgpu
    pl = obj.pl
    canon = [p for p, (i, j) in enumerate(pl.pairs) if i <= j]
    owners = _owners(obj, canon, devs)
    res = {}

    def dwork(d):
        wv = mgpu.device_view(obj, ('S_pao_d',),
                              sub={'dfi': ('qij', 'J2c', 'C_pao_d', 'C_lmo_d', 'mask_uv_d')})
        res[d] = _device_pass(wv, owners[d], verbose)
    mgpu.run_per_device(dwork, list(owners))
    home = cp.cuda.Device().id
    off = obj.df_offload
    # the CC driver reads them on the home device: one copy per device's
    # accumulators, then views
    for d, (Kv, jitems, joff, Jacc, Qab_raw) in res.items():
        Jh = mgpu.replicate(Jacc)
        for p, k in Kv.items():
            obj.K_ivvv[p] = mgpu.replicate(k)
        groups = {}
        for n_, it in enumerate(jitems):
            groups.setdefault((it[0], it[1]), []).append(n_)
        for (p, side), ns_ in groups.items():
            lst = []
            for n_ in ns_:
                v_ = obj.X[p].shape[1]
                vt = obj.X[jitems[n_][3]].shape[1]
                lst.append(Jh[int(joff[n_]):int(joff[n_]) + v_ * vt].reshape(v_, vt))
            obj.J_ikac[p if side == 0 else pl.ji[p]] = lst
        del Kv, Jacc
        _half_fit(obj, Qab_raw, off)
    res.clear()
    for d in devs:
        with cp.cuda.Device(d):
            cp.get_default_memory_pool().free_all_blocks()


def _half_fit(obj, Qab_raw, off):
    """strong pairs' Qab = L^-1 (Q|ab) over their fitting domains (L: the
    Cholesky factor of the domain metric), as the pair-major path stores it."""
    dfi, pl = obj.dfi, obj.pl
    ps = list(Qab_raw)
    if not ps:
        return
    na_ = np.array([Qab_raw[p].shape[0] for p in ps])
    order = np.argsort(na_)
    i0 = 0
    while i0 < len(ps):
        A = int(-(-na_[order[i0]] // 32) * 32)
        grp = []
        while i0 < len(ps) and int(-(-na_[order[i0]] // 32) * 32) == A and len(grp) < 64:
            grp.append(ps[order[i0]])
            i0 += 1
        b = len(grp)
        Vm = max(obj.X[p].shape[1] for p in grp)
        aidx, am = np.zeros((b, A), np.int64), np.zeros((b, A))
        Q = np.zeros((b, A, Vm * Vm))
        for s, p in enumerate(grp):
            ax = dfi.aux_index(obj.riatoms[p])
            aidx[s, :len(ax)], am[s, :len(ax)] = ax, 1.
            v = obj.X[p].shape[1]
            q = np.zeros((len(ax), Vm, Vm))
            q[:, :v, :v] = Qab_raw[p]
            Q[s, :len(ax)] = q.reshape(len(ax), -1)
        ga, gam = cp.asarray(aidx), cp.asarray(am)
        Jm = dfi.J2c[ga[:, :, None], ga[:, None, :]] * (gam[:, :, None] * gam[:, None, :])
        Jm += cp.eye(A)[None] * (1 - gam)[:, :, None]
        Linv = chol_inv_batched(Jm)
        H = cp.matmul(Linv, cp.asarray(Q)).reshape(b, A, Vm, Vm)
        for s, p in enumerate(grp):
            nax, v = Qab_raw[p].shape[0], obj.X[p].shape[1]
            x = cp.ascontiguousarray(H[s, :nax, :v, :v])
            val = cp.asnumpy(x) if off else x
            obj.Qab[p] = val
            q = pl.ji[p]
            if q != p:
                obj.Qab[q] = val
            Qab_raw[p] = None
