"""Atom-major (Q|AB) blocks of the open-shell CC pair integrals.

The pair-major path (``uhf._pair_core_chunk`` under ``DFInts.run_regions``)
needs, for each chunk of pairs, the (Q_a|uv) block of every aux atom of their
fitting domains.  For a compact protein the chunks' domains span most atoms,
so every atom's block is evaluated or read back many times (mimochrome
quintet: 502 regions, 36556 atom blocks from a 573 GB scratch store).

Here the canonical pairs are split over the devices (contiguous in Morton
order) and each device loops once over the aux atoms its pairs use: one
(Q_a|uv) block over the union of those pairs' PAOs, from which every pair gets
its rows of

    q_vv^s(Q_a, A, B) = X_s^T (Q_a|uv) X_s      (s = alpha, beta PAO part)

(not fitted).  A pair's blocks go to the scratch store as soon as its last
aux atom is done.  ``_pair_core_chunk`` then reads them instead of the
regions; the same sums in another order.
"""
import numpy as np
import cupy as cp

from electrolites.dlpno.gather import blocks as gather_blocks, xtbx
from electrolites.dlpno.ragged import padded


def _owners(u, clist, devs):
    """the pairs split over the devices: contiguous in Morton order of their
    fitting domains, balanced by an estimate of their work."""
    dfi = u.dfi
    order = dfi.spatial_order([np.unique(u.ribf_atom[u.ribfs[c]]) for c in clist])
    w = []
    for k in order:
        c = clist[k]
        P = len(u.pao[c])
        w.append(len(u.ribfs[c]) * P * (P + u.nva(c) + u.nvb(c)))
    w = np.cumsum(w)
    cut = np.searchsorted(w, w[-1] * np.arange(1, len(devs)) / len(devs))
    return {d: [clist[k] for k in part] for d, part in zip(devs, np.split(np.asarray(order), cut))}


def _split_users(u, us, na):
    """an atom's pairs in groups whose square block over the union of their
    PAOs fits a fifth of the free device memory."""
    cp.get_default_memory_pool().free_all_blocks()
    free = cp.cuda.runtime.memGetInfo()[0]
    # the block, next to the AO integral chunk of _direct_rect (up to 15% of
    # the free memory) and its transforms
    lim = 0.12 * free
    out, cur = [], []
    m = np.zeros(u.dfi.nbf, bool)
    for c in us:
        m2 = m.copy()
        m2[u.pao[c]] = True
        if cur and 8. * na * float(m2.sum()) ** 2 > lim:
            out.append(cur)
            cur = []
            m2 = np.zeros(u.dfi.nbf, bool)
            m2[u.pao[c]] = True
        cur.append(c)
        m = m2
    if cur:
        out.append(cur)
    return out


def _device_pass(u, pairs, store, verbose=0):
    import time
    dfi = u.dfi
    t0 = time.perf_counter()
    offs, left = {}, {}
    users = {}
    for c in pairs:
        o, at = 0, {}
        for a in np.unique(u.ribf_atom[u.ribfs[c]]):
            at[int(a)] = o
            o += len(dfi.atom_ribfs[a])
            users.setdefault(int(a), []).append(c)
        offs[c] = (at, o)
        left[c] = len(at)
    acc = {}
    out = {}
    atoms = sorted(users)
    atoms = [atoms[k] for k in dfi.spatial_order([[a] for a in atoms])]
    st = dict(blocks=0, bytes=0., t_blk=0., t_pair=0.)
    for a in atoms:
        na = len(dfi.atom_ribfs[a])
        for sub in _split_users(u, users[a], na):
            t1 = time.perf_counter()
            rows = np.unique(np.concatenate([u.pao[c] for c in sub]))
            aosub = None
            if dfi._pao_sup is not None:
                aosub = dfi._region_groups([a], {a: rows})[0][1]
            blk, rmap, _, _ = dfi._direct_rect(a, rows, None, cache=False, aosub=aosub)
            st['blocks'] += 1
            st['bytes'] += blk.nbytes
            st['t_blk'] += time.perf_counter() - t1
            t1 = time.perf_counter()
            # pairs in groups of equal padded PAO domain / PNO counts
            key = {}
            for c in sub:
                key.setdefault((int(padded(len(u.pao[c]))), int(padded(max(u.nva(c), u.nvb(c))))), []).append(c)
            for (P, V), grp in key.items():
                step = max(1, int((1 << 28) // (8 * na * P * max(P, V))))
                for c0 in range(0, len(grp), step):
                    bp = grp[c0:c0 + step]
                    b = len(bp)
                    pidx, pm = np.zeros((b, P), np.int64), np.zeros((b, P))
                    Xa, Xb = np.zeros((b, P, V)), np.zeros((b, P, V))
                    for s_, c in enumerate(bp):
                        n_ = len(u.pao[c])
                        pidx[s_, :n_], pm[s_, :n_] = u.pao[c], 1.
                        Xa[s_, :n_, :u.nva(c)] = u.Xa[c]
                        Xb[s_, :n_, :u.nvb(c)] = u.Xb[c][:n_]
                    gpm = cp.asarray(pm)
                    pos = cp.asarray(np.where(pm > 0, rmap[pidx], 0))
                    bk = gather_blocks(blk, pos, pos, gpm, gpm, ru=True)          # (b, P, na, P)
                    qa = xtbx(bk, cp.asarray(Xa), cp.asarray(Xa)).get()
                    qb = xtbx(bk, cp.asarray(Xb), cp.asarray(Xb)).get()
                    del bk
                    for s_, c in enumerate(bp):
                        at, A = offs[c]
                        if c not in acc:
                            acc[c] = (np.zeros((A, u.nva(c), u.nva(c))), np.zeros((A, u.nvb(c), u.nvb(c))))
                        o = at[a]
                        acc[c][0][o:o + na] = qa[s_, :, :u.nva(c), :u.nva(c)]
                        acc[c][1][o:o + na] = qb[s_, :, :u.nvb(c), :u.nvb(c)]
            del blk
            st['t_pair'] += time.perf_counter() - t1
        for c in users[a]:
            left[c] -= 1
            if left[c] == 0:                # all rows of this pair: to the store
                qa_, qb_ = acc.pop(c)
                out[c] = (store.put(qa_), store.put(qb_)) if store is not None else (qa_, qb_)
    cp.cuda.Device().synchronize()
    if verbose >= 2:
        print(f'[mem] atom-major open-shell (Q|AB) on GPU {cp.cuda.Device().id}: {len(pairs)} pairs, '
              f'{len(atoms)} atoms, {st["blocks"]} blocks ({st["bytes"] / 1e9:.0f} GB) in {st["t_blk"]:.0f} s, '
              f'pair work {st["t_pair"]:.0f} s, total {time.perf_counter() - t0:.0f} s', flush=True)
    return out


def qvv_atom_pass(u, clist, devs, store=None, verbose=0):
    """{c: (raw alpha, raw beta) (Q|AB) blocks} of the canonical pairs
    ``clist`` (host arrays or SpillRefs into ``store``)."""
    from electrolites.dlpno import mgpu
    owners = _owners(u, list(clist), devs)
    res = {}

    def dwork(d):
        wv = mgpu.device_view(u, ('S_pao_d',), sub={'dfi': ('J2c', 'C_pao_d', 'mask_uv_d')})
        res[d] = _device_pass(wv, owners[d], store, verbose)
    mgpu.run_per_device(dwork, list(owners))
    out = {}
    for r in res.values():
        out.update(r)
    if store is not None:
        store.flush()
    for d in devs:
        with cp.cuda.Device(d):
            cp.get_default_memory_pool().free_all_blocks()
    return out
