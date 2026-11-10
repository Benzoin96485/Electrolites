"""Common-left triples overlaps with complete, bounded GPU pullbacks.

For each local pair p, S[p] = A.T M[:, positions[p]] B[p]. Additively
embedding the B[p] into a common column domain gives S = (A.T M) B.
The shared projection and its reverse are evaluated once per bounded batch.
The metric need not be symmetric, and coefficient rows may be repeated.
"""
import numpy as np
import cupy as cp

from . import reverse as ad


def workspace_bytes(left_rows, union_rows, left_rank, total_rank, local_elements):
    """Conservative live device-array envelope, excluding inputs and allocator."""
    n, u, v, r, b = map(int, (left_rows, union_rows, left_rank,
                             total_rank, local_elements))
    return 8*(4*n*u + 3*n*v + 4*u*r + 4*v*r + 3*v*u + 3*b)


def overlap_batches(keys, left, pao, coefficients, *, max_bytes=256 << 20,
                    max_count=128):
    """Preserve first-use order and cap the shared projection workspace."""
    keys = tuple(dict.fromkeys(keys))
    groups, current, union = [], [], set()
    total_rank = local_elements = 0

    def envelope(rows, rank, elements):
        return workspace_bytes(left.shape[0], len(rows), left.shape[1],
                               rank, elements)

    def finish():
        groups.append((tuple(current), np.asarray(sorted(union), dtype=np.int64),
                       envelope(union, total_rank, local_elements)))

    for p in keys:
        rows = set(map(int, pao[p]))
        rank, elements = coefficients[p].shape[1], int(np.prod(coefficients[p].shape))
        trial_union = union | rows
        bound = envelope(trial_union, total_rank+rank, local_elements+elements)
        if current and (len(current) >= max_count or bound > max_bytes):
            finish()
            current, union = [], set()
            total_rank = local_elements = 0
            trial_union = rows
            bound = envelope(rows, rank, elements)
        if bound > max_bytes:
            raise ValueError('One common-left overlap exceeds the GPU workspace bound')
        current.append(p)
        union = trial_union
        total_rank += rank
        local_elements += elements
    if current:
        finish()
    return groups


def common_left_batch(metric, left, coefficients, positions, *, reuse_workspace=None):
    """Return GPU overlaps and pull back to each operand's original backend.

    If H=A.T M, Y=H B and Z is the output seed, then
    bar_H=Z B.T, bar_A=M bar_H.T, bar_M=A bar_H and bar_B=H.T Z.
    Unembedding bar_B gathers rows, including repeated indices. Coefficients
    can alias other operands; the reverse engine sums their contributions.
    Recompute temporary GPU arrays in the VJP so checkpoints retain no GPU
    workspaces when their outputs are on the host.
    """
    coefficients = tuple(coefficients)
    positions = tuple(np.asarray(pos, dtype=np.int64) for pos in positions)
    if len(coefficients) != len(positions):
        raise ValueError('Coefficient and position counts differ')
    if not coefficients:
        return ()
    operands = (metric, left)+coefficients
    values = tuple(ad.primal(x) for x in operands)
    backends = tuple(ad._xp([x]) for x in operands)
    matrix, a, *xs = values
    if any(x.ndim != 2 or x.dtype != np.float64 for x in values):
        raise ValueError('Common-left overlaps require real float64 matrices')
    n, u = matrix.shape
    if a.shape[0] != n:
        raise ValueError('Left coefficient and metric row counts differ')
    for x, pos in zip(xs, positions):
        if len(pos) != x.shape[0] or np.any(pos < 0) or np.any(pos >= u):
            raise ValueError('Invalid common-left coefficient rows')
    ranks = tuple(x.shape[1] for x in xs)
    offsets = np.cumsum([0]+list(ranks))
    unique = tuple(len(np.unique(pos)) == len(pos) for pos in positions)
    host_right = all(xp is np for xp in backends[2:])

    def embedded():
        xp = np if host_right else cp
        b = xp.zeros((u, int(offsets[-1])))
        for x, pos, first, last, is_unique in zip(xs, positions, offsets[:-1],
                                                 offsets[1:], unique):
            if first == last:
                continue
            part = b[:, first:last]
            if is_unique:
                part[pos] = xp.asarray(x)
            else:
                xp.add.at(part, xp.asarray(pos), xp.asarray(x))
        return cp.asarray(b)

    from .local_read_plans import OPTIONS
    reuse = OPTIONS.get('overlap_replay', False) if reuse_workspace is None else bool(reuse_workspace)
    workspace_key = object()

    def rebuild():
        matrix_device, left_device = cp.asarray(matrix), cp.asarray(a)
        right_device = embedded()
        h = left_device.T @ matrix_device
        return matrix_device, left_device, right_device, h

    def restore(value, xp):
        return cp.asnumpy(value) if xp is np else value

    def pullback(seed, needed):
        z = cp.asarray(seed)
        result = [None]*len(operands)
        workspace = ad.checkpoint_replay_workspace()
        primals = None if not reuse or workspace is None else workspace.pop(workspace_key, None)
        if primals is not None:
            matrix_device, left_device, right_device, h = primals
        if needed[0] or needed[1]:
            gh = z @ (embedded() if primals is None else right_device).T
            if needed[0]:
                result[0] = restore((cp.asarray(a) if primals is None else left_device) @ gh, backends[0])
            if needed[1]:
                result[1] = restore((cp.asarray(matrix) if primals is None else matrix_device) @ gh.T, backends[1])
        if any(needed[2:]):
            if primals is None:
                h = cp.asarray(a).T @ cp.asarray(matrix)
            gb = h.T @ z
            if host_right:
                gb = cp.asnumpy(gb)
            for i, (pos, first, last, need, xp) in enumerate(zip(
                    positions, offsets[:-1], offsets[1:], needed[2:], backends[2:])):
                if need:
                    block = gb[pos, first:last].copy()
                    result[i+2] = block if host_right else restore(block, xp)
        return tuple(result)

    if reuse:
        primals = rebuild()
        y = primals[3] @ primals[2]
        workspace = ad.checkpoint_replay_workspace()
        if workspace is not None:
            workspace[workspace_key] = primals
    else:
        y = (cp.asarray(a).T @ cp.asarray(matrix)) @ embedded()
    packed = ad.operation(y, operands,
                          lambda seed: pullback(seed, [True]*len(operands)), pullback)
    outputs = []
    for first, last in zip(offsets[:-1], offsets[1:]):
        key = (slice(None), slice(int(first), int(last)))

        def output_bar(bar, key=key):
            return (ad.IndexedCotangent(packed.shape, key, bar),)

        outputs.append(ad.operation(packed.v[key], [packed], output_bar))
    return tuple(outputs)


def triplet_overlaps(state, t, keys):
    """Build exactly the requested triplet/pair overlaps on bounded PAO unions."""
    left = state.X_tno[t]
    groups = overlap_batches(keys, left, state.pao, state.X)
    result = {}
    for group, union, _ in groups:
        metric = state.orb.S_pao[np.ix_(state.t_paos[t], union)]
        coefficients = tuple(state.X[p] for p in group)
        positions = tuple(np.searchsorted(union, state.pao[p]) for p in group)
        result.update(zip(group, common_left_batch(metric, left, coefficients, positions)))
    return result
