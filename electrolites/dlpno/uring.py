"""Atom-major (Q|uv) part of the open-shell non-projected ring integrals.

``uhf._ring_np_pair`` builds, per pair p and registry key n = (p, K, Q, bp,
c, q),

    res_n = Z_n X_q,   Z_n = c1 (k u|B q) - c2 sum_Q Boo[Q, k, qo] Vx[Q, B, u]

with Vx = L^-1 (Q|B_p u) over the extended PAOs u of all targets q (L: the
Cholesky factor of p's fitting metric).  The (Q|uv) part of Vx needs a block
per aux atom of p over (P_p x ext); with direct (Q|uv) the regions read those
blocks many times.  Here they come atom-major, one block per aux atom and
device, and the (Q|uv) part of every key is contracted at once:

    res_n^(uv) = - sum_a F_a[:, k, qo]^T (X_p^T (Q_a|P_p, P_q) X_q)

with F = J^-1 (Q|k qo) over p's fitting domain (so that Boo^T Vx = F^T v).
The rest of res_n (the (Q|iu) / (Q|ij) parts) is unchanged.
"""
import numpy as np
import cupy as cp
import cupyx.scipy.linalg as cpla

from electrolites.dlpno.uatom import _owners


def _pair_setup(u, p, keys, idx):
    """the keys of p with a (Q|uv) part, their (k, qo) rows / targets and F."""
    so, dfi = u.so, u.dfi
    lm = list(u.lmos[p])
    lpos = {m: r for r, m in enumerate(lm)}
    act = []
    for n in idx:
        (_, K, Q, bp, c_, q) = keys[n]
        if so[K][1] == so[Q][1] and bp == c_:
            act.append(n)
    if not act:
        return None
    tg = sorted({(keys[n][5], keys[n][4]) for n in act})
    tpos = {t: r for r, t in enumerate(tg)}
    aux = u.ribfs[p]
    # qo (the LMO of Q) is one of the pair's two LMOs: F over those columns only
    qcols = sorted({lpos[so[keys[n][2]][0]] for n in act})
    qmap = {c: r for r, c in enumerate(qcols)}
    ga, gl = cp.asarray(aux), cp.asarray(np.asarray(lm, np.int64))
    gq = cp.asarray(np.asarray([lm[c] for c in qcols], np.int64))
    qoo = dfi.qij[ga[:, None, None], gl[None, :, None], gq[None, None, :]]
    Lc = cp.linalg.cholesky(dfi.J2c[ga[:, None], ga[None, :]])
    A, nl, nq = len(aux), len(lm), len(qcols)
    F = cpla.solve_triangular(Lc, qoo.reshape(A, nl * nq), lower=True)
    F = cpla.solve_triangular(Lc.T, F, lower=False).reshape(A, nl, nq)
    del qoo, Lc
    offs, o = {}, 0
    for a in np.unique(u.ribf_atom[aux]):
        offs[int(a)] = o
        o += len(dfi.atom_ribfs[a])
    # target coefficients (PAO rows), padded
    Pq = max(len(u.pao[q]) for q, _ in tg)
    Vq = max((u.nva(q) if c_ == 0 else u.nvb(q)) for q, c_ in tg)
    Xt = np.zeros((len(tg), Pq, Vq))
    cols = np.unique(np.concatenate([u.pao[q] for q, _ in tg]))
    cpos = -np.ones(u.dfi.nbf, np.int64)
    cpos[cols] = np.arange(len(cols))
    tcol, tmsk = np.zeros((len(tg), Pq), np.int64), np.zeros((len(tg), Pq))
    for r, (q, c_) in enumerate(tg):
        n_ = len(u.pao[q])
        X = u.Xa[q] if c_ == 0 else u.Xb[q][:n_]
        Xt[r, :n_, :X.shape[1]] = X
        tcol[r, :n_], tmsk[r, :n_] = cpos[u.pao[q]], 1.
    per_bp = {}
    for bp in (0, 1):
        ns_ = [n for n in act if keys[n][3] == bp]
        if not ns_:
            continue
        # distinct (target, k, qo) combinations: the keys of one combination
        # (spin cases of K, Q) share the contraction
        trip = np.array([(tpos[(keys[n][5], keys[n][4])], lpos[so[keys[n][1]][0]], qmap[lpos[so[keys[n][2]][0]]])
                         for n in ns_], np.int64).reshape(-1, 3)
        cmb, kmap = np.unique(trip, axis=0, return_inverse=True)
        V = u.nva(p) if bp == 0 else u.nvb(p)
        per_bp[bp] = dict(n=ns_, comb=kmap.reshape(-1), t=cp.asarray(cmb[:, 0]), k=cp.asarray(cmb[:, 1]),
                          qo=cp.asarray(cmb[:, 2]), acc=cp.zeros((len(cmb), V, Vq)))
    return dict(F=F, offs=offs, cols=cols, Xt=cp.asarray(Xt), tcol=cp.asarray(tcol), tmsk=cp.asarray(tmsk),
                per_bp=per_bp, Vq=Vq)


def _pair_bytes(u, p, keys, idx):
    """device bytes a live pair holds in the pass: F (A x nl x 2), the target
    coefficients and one accumulator per distinct (target, k, qo) and spin."""
    so = u.so
    nl = len(u.lmos[p])
    cmb, tg = set(), set()
    for n in idx:
        (_, K, Q, bp, c_, q) = keys[n]
        if so[K][1] == so[Q][1] and bp == c_:
            cmb.add((q, c_, so[K][0], so[Q][0], bp))
            tg.add((q, c_))
    if not cmb:
        return 0.
    Vq = max((u.nva(q) if c_ == 0 else u.nvb(q)) for q, c_ in tg)
    Pq = max(len(u.pao[q]) for q, _ in tg)
    V = max(u.nva(p), u.nvb(p))
    return 8. * (len(u.ribfs[p]) * nl * 2 + len(cmb) * V * Vq + len(tg) * Pq * (Vq + 2))


def _ring_device_pass(u, items, keys, verbose=0, sink=None):
    import time
    dfi = u.dfi
    t0 = time.perf_counter()
    users, left = {}, {}
    for p, _ in items:
        nat = 0
        for a in np.unique(u.ribf_atom[u.ribfs[p]]):
            if len(dfi.atom_ribfs[a]):
                users.setdefault(int(a), []).append(p)
                nat += 1
        left[p] = nat
    idx_of = dict(items)
    live, out = {}, {}
    atoms = sorted(users)
    atoms = [atoms[k] for k in dfi.spatial_order([[a] for a in atoms])]
    st = dict(blocks=0, bytes=0., t_blk=0., t_pair=0.)
    for a in atoms:
        na = len(dfi.atom_ribfs[a])
        for p in users[a]:
            if p not in live:
                live[p] = _pair_setup(u, p, keys, idx_of[p])
        us = [p for p in users[a] if live[p] is not None]
        cp.get_default_memory_pool().free_all_blocks()
        free = cp.cuda.runtime.memGetInfo()[0]
        # the block, next to the AO integral chunk of _direct_rect (up to 15% of
        # the free memory) and its transforms
        lim = 0.12 * free
        groups, cur = [], []
        rm, cm = np.zeros(dfi.nbf, bool), np.zeros(dfi.nbf, bool)
        for p in us:
            r2, c2 = rm.copy(), cm.copy()
            r2[u.pao[p]] = True
            c2[live[p]['cols']] = True
            if cur and 8. * na * float(r2.sum()) * float(c2.sum()) > lim:
                groups.append(cur)
                cur, r2, c2 = [], np.zeros(dfi.nbf, bool), np.zeros(dfi.nbf, bool)
                r2[u.pao[p]] = True
                c2[live[p]['cols']] = True
            cur.append(p)
            rm, cm = r2, c2
        if cur:
            groups.append(cur)
        for sub in groups:
            t1 = time.perf_counter()
            rows = np.unique(np.concatenate([u.pao[p] for p in sub]))
            cols = np.unique(np.concatenate([live[p]['cols'] for p in sub]))
            aosub = None
            if dfi._pao_sup is not None:
                aosub = dfi._region_groups([a], {a: (rows, cols)})[0][1]
            blk, rmap, cmap, _ = dfi._direct_rect(a, rows, cols, cache=False, aosub=aosub)
            st['blocks'] += 1
            st['bytes'] += blk.nbytes
            st['t_blk'] += time.perf_counter() - t1
            t1 = time.perf_counter()
            for p in sub:
                L_ = live[p]
                o = L_['offs'][a]
                rp = cp.asarray(rmap[u.pao[p]])
                cq = cp.asarray(cmap[L_['cols']])
                b_ = blk[:, rp[:, None], cq[None, :]]                       # (na, P, E)
                Xa, Xbp, _ = u._vir_coef_d(p)
                nt, Pq = L_['tcol'].shape
                Vq = L_['Vq']
                for bp, D in L_['per_bp'].items():
                    Xp = Xa if bp == 0 else Xbp
                    V = Xp.shape[1]
                    E = b_.shape[2]
                    Y = cp.matmul(Xp.T[None], b_).reshape(na, V * E)         # (na, V*E)
                    G = L_['F'][o:o + na][:, D['k'], D['qo']]               # (na, nc)
                    nc = G.shape[1]
                    cstep = max(1, int((1 << 28) // (8 * V * E)))
                    for c0 in range(0, nc, cstep):
                        sl = slice(c0, min(nc, c0 + cstep))
                        # sum_Q F[Q, k, qo] X_p^T (Q|P_p, cols), then the
                        # combination's target columns times its coefficients
                        H = cp.matmul(G[:, sl].T, Y).reshape(-1, V, E)        # (ncs, V, E)
                        tt = D['t'][sl]
                        Hs = cp.take_along_axis(H, L_['tcol'][tt][:, None, :], axis=2) \
                            * L_['tmsk'][tt][:, None, :]                         # (ncs, V, Pq)
                        D['acc'][sl] += cp.matmul(Hs, L_['Xt'][tt])
                        del H, Hs
                    del Y, G
                del b_
            del blk
            st['t_pair'] += time.perf_counter() - t1
        for p in users[a]:
            left[p] -= 1
            if left[p] == 0:
                L_ = live.pop(p)
                if L_ is None:
                    continue
                for bp, D in L_['per_bp'].items():
                    acc = D['acc'].get()            # finished: to the host
                    for r, n in enumerate(D['n']):
                        q, c_ = keys[n][5], keys[n][4]
                        w = u.nva(q) if c_ == 0 else u.nvb(q)
                        if sink is not None:
                            sink(n, -acc[D['comb'][r], :, :w])
                        else:
                            out[n] = -acc[D['comb'][r], :, :w]
                    del acc
    cp.cuda.Device().synchronize()
    if verbose >= 2:
        print(f'[mem] atom-major ring (Q|uv) part on GPU {cp.cuda.Device().id}: {len(items)} pairs, '
              f'{len(atoms)} atoms, {st["blocks"]} blocks ({st["bytes"] / 1e9:.0f} GB) in {st["t_blk"]:.0f} s, '
              f'pair work {st["t_pair"]:.0f} s, total {time.perf_counter() - t0:.0f} s', flush=True)
    return out


def ring_qab_pass(u, items, keys, devs, verbose=0, sink=None):
    """{key index: (Q|uv) part of the non-projected ring integral} for
    items [(p, key indices)] (host arrays), or each part handed to
    sink(key index, part) as it is finished."""
    from electrolites.dlpno import mgpu
    ps = [p for p, _ in items]
    owners = _owners(u, ps, devs)
    idx = dict(items)
    res = {}

    def dwork(d):
        wv = mgpu.device_view(u, ('S_pao_d',), sub={'dfi': ('qij', 'J2c', 'C_pao_d', 'mask_uv_d')})
        # the pairs in batches whose live data (F, accumulators) fit a quarter
        # of the free device memory; each batch loops over its own atoms
        free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        lim = 0.25 * free
        batches, cur, mem = [], [], 0.
        for p in owners[d]:
            m = _pair_bytes(u, p, keys, idx[p])
            if cur and mem + m > lim:
                batches.append(cur)
                cur, mem = [], 0.
            cur.append(p)
            mem += m
        if cur:
            batches.append(cur)
        res[d] = {}
        for b in batches:
            res[d].update(_ring_device_pass(wv, [(p, idx[p]) for p in b], keys, verbose, sink))
            cp.get_default_memory_pool().free_all_blocks()
    mgpu.run_per_device(dwork, list(owners))
    out = {}
    for r in res.values():
        out.update(r)
    return out
