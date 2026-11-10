"""Compact host plans for batches of local PNO overlaps.

All lookup arrays refer to the pairs and PAOs in one batch. The reference
PAO metric remains global; these helpers do not change its representation.
"""
import numpy as np


def subset_positions(ordered, values):
    """Indices into a sorted unique local set, rejecting missing members."""
    ordered, values = np.asarray(ordered), np.asarray(values)
    position = np.searchsorted(ordered, values)
    if np.any(position >= len(ordered)):
        raise ValueError('Overlap batch omits a requested pair or PAO')
    if np.any(ordered[position] != values):
        raise ValueError('Overlap batch omits a requested pair or PAO')
    return position


def overlap_chunks(pao, nv, nbf, pp, qq, ne=0, max_bytes=1 << 29,
                   pao_sizes=None, maximum_nv=None):
    """Greedy, vectorized batches bounded by the existing workspace estimate.

    Return (couple indices, sorted PAO union, sorted involved pair indices).
    A single oversized couple is retained so the caller can process it.
    No array has length ``nbf``. The initial stable couple ordering and
    local unique operations are retained from the energy implementation.
    """
    pp, qq = np.asarray(pp, np.int64), np.asarray(qq, np.int64)
    if pp.shape != qq.shape or pp.ndim != 1:
        raise ValueError('Overlap couple indices must be matching vectors')
    if not len(pp):
        return []
    nv = np.asarray(nv, np.int64)
    # A persistent GPU producer supplies these once. Recomputing them for
    # each small requested chunk would scan all pairs repeatedly.
    npao = (np.asarray([len(p) for p in pao], np.int64) if pao_sizes is None
            else np.asarray(pao_sizes, np.int64))
    if (len(nv) != len(pao) or min(pp.min(), qq.min()) < 0
            or max(pp.max(), qq.max()) >= len(pao)):
        raise ValueError('Invalid overlap pair index')
    order = np.argsort(qq, kind='stable')
    left, right = pp[order], qq[order]
    width = max(1, int(nv.max()) if maximum_nv is None else int(maximum_nv))
    guess = max(64, int(max_bytes // (16 * max(1, nbf + ne) * width)) * 8)
    out, start = [], 0
    while start < len(pp):
        window = min(len(pp) - start, guess)
        ids = np.stack([left[start:start + window], right[start:start + window]], 1).ravel()
        _, first = np.unique(ids, return_index=True)
        isnew = np.zeros(2 * window, np.int64)
        isnew[first] = 1
        pairs = np.cumsum(isnew.reshape(window, 2).sum(axis=1))
        union_bound = np.minimum(np.cumsum((npao[ids] * isnew).reshape(window, 2).sum(axis=1)), nbf)
        estimate = 16. * pairs * (union_bound + ne) * width
        over = np.flatnonzero(estimate > max_bytes)
        length = max(1, int(over[0])) if len(over) else window
        if not len(over) and window < len(pp) - start:
            guess *= 2
        selected = order[start:start + length]
        involved = np.unique(np.concatenate([left[start:start + length], right[start:start + length]]))
        union = np.unique(np.concatenate([np.asarray(pao[p], np.int64) for p in involved]))
        out.append((selected, union, involved))
        start += length
    return out


def scatter_rows(pao, union, involved):
    """Flatten local PAO-to-batch rows, with offsets for each involved pair."""
    sizes = np.array([len(pao[p]) for p in involved], np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]) if len(sizes) else np.zeros(0, np.int64)
    columns = (np.concatenate([np.asarray(pao[p], np.int64) for p in involved])
               if len(involved) else np.zeros(0, np.int64))
    return offsets, subset_positions(union, columns).astype(np.int32)
