"""Exact GPU sources for one coupled (T1) residual.

The three-index transformation uses three GEMMs. Its VJP differentiates
all three occurrences of the overlap. A fused residual accumulates one
coupling at a time and recomputes its intermediates on the reverse sweep;
the production host checkpoint retains no cubic device workspaces.
"""
import numpy as np
import cupy as cp

from . import reverse as ad
from .ref_rhf import _permute
from .triples_overlap_adjoint_gpu import common_left_batch, overlap_batches


def workspace_bytes(n, m):
    """Conservative per-edge device envelope, excluding inputs and allocator.

    Returned GPU cotangents and shared overlap outputs are not workspace.
    Production inputs and tensor cotangents live on the host, so cubic
    temporary storage does not grow with the number of coupled triplets.
    """
    n, m = int(n), int(m)
    return 8*(8*n**3 + 10*n*m*m + 10*n*n*m + 5*m**3 + 16*n*m + 8*(n+m))


def _project(a, s):
    n, m = s.shape
    left = (s @ a.reshape(m, m*m)).reshape(n, m, m)
    middle = s[None] @ left
    return (middle.reshape(n*n, m) @ s.T).reshape(n, n, n)


def _project_vjp(a, s, z, need_tensor, need_overlap):
    """Pull back Y_abc = S_aA S_bB S_cC A_ABC without a Jacobian."""
    n, m = s.shape
    gz = z.reshape(n*n, n)
    middle_bar = (gz @ s).reshape(n, n, m)
    left_bar = s.T[None] @ middle_bar
    tensor_bar = ((s.T @ left_bar.reshape(n, m*m)).reshape(m, m, m)
                  if need_tensor else None)
    overlap_bar = None
    if need_overlap:
        left = (s @ a.reshape(m, m*m)).reshape(n, m, m)
        middle = s[None] @ left
        overlap_bar = gz.T @ middle.reshape(n*n, m)
        overlap_bar += (middle_bar.transpose(1, 0, 2).reshape(n, n*m)
                        @ left.transpose(0, 2, 1).reshape(n*m, m))
        overlap_bar += left_bar.reshape(n, m*m) @ a.reshape(m, m*m).T
    return tensor_bar, overlap_bar


def _restore(value, xp):
    return cp.asnumpy(value) if xp is np else value


def coupled_residual(w, tensor, energies, occupied_sum, edges):
    """Fused residual and complete selected pullback.

    Each edge is (tensor_u, overlap_tu, F_occ_old, permutation_key).
    Aliases remain separate graph edges and the reverse engine adds them.
    Only small overlap matrices may be retained on the GPU; all other
    converted inputs are rebuilt as needed, one coupling at a time.
    """
    edges = tuple(edges)
    operands = (w, tensor, energies, occupied_sum)+tuple(
        x for a, s, f, key in edges for x in (a, s, f))
    values = tuple(ad.primal(x) for x in operands)
    backends = tuple(ad._xp([x]) for x in operands)
    if any(np.dtype(x.dtype) != np.float64 for x in values):
        raise ValueError('Triples residual requires real float64 operands')
    n = values[2].size
    if values[0].shape != (n, n, n) or values[1].shape != (n, n, n):
        raise ValueError('Inconsistent triples residual tensor shape')
    if values[2].shape != (n,) or values[3].shape != ():
        raise ValueError('Invalid triples residual orbital energies')
    inverses = tuple(tuple(np.argsort(key, kind='stable')) for *_, key in edges)
    for edge, (a, s, f) in enumerate(zip(values[4::3], values[5::3], values[6::3])):
        if s.ndim != 2 or s.shape[0] != n or a.shape != (s.shape[1],)*3 or f.shape != ():
            raise ValueError(f'Invalid triples residual coupling {edge}')

    def denominator():
        e = cp.asarray(values[2])
        return e[:, None, None]+e[None, :, None]+e[None, None, :]-cp.asarray(values[3])

    def forward():
        result = cp.array(values[0], copy=True)
        result += cp.asarray(values[1])*denominator()
        for pos, (_, _, _, key) in enumerate(edges):
            a, s, f = values[4+3*pos:7+3*pos]
            projected = _project(cp.asarray(_permute(a, *key)), cp.asarray(s))
            projected *= cp.asarray(f)
            result -= projected
        return _restore(result, backends[0])

    def pullback(seed, needed):
        z = cp.asarray(seed)
        bars = [None]*len(operands)
        if needed[0]:
            bars[0] = _restore(z, backends[0])
        if needed[1]:
            bars[1] = _restore(z*denominator(), backends[1])
        if needed[2] or needed[3]:
            weighted = z*cp.asarray(values[1])
            if needed[2]:
                bars[2] = _restore(weighted.sum((1, 2))+weighted.sum((0, 2))
                                   + weighted.sum((0, 1)), backends[2])
            if needed[3]:
                bars[3] = _restore(-weighted.sum(), backends[3])
            del weighted
        for pos, (_, _, _, key) in enumerate(edges):
            start = 4+3*pos
            need_a, need_s, need_f = needed[start:start+3]
            if not (need_a or need_s or need_f):
                continue
            a, s, f = values[start:start+3]
            a = cp.asarray(_permute(a, *key))
            s = cp.asarray(s)
            if need_f:
                bars[start+2] = _restore(-cp.sum(z*_project(a, s)), backends[start+2])
            if need_a or need_s:
                ga, gs = _project_vjp(a, s, -cp.asarray(f)*z, need_a, need_s)
                if need_a:
                    bars[start] = _restore(ga.transpose(inverses[pos]), backends[start])
                if need_s:
                    bars[start+1] = _restore(gs, backends[start+1])
        return tuple(bars)

    return ad.operation(forward(), operands,
                        lambda seed: pullback(seed, [True]*len(operands)), pullback)


def triplet_residual(state, t, connections, *, max_workspace_bytes=8 << 30):
    """Build the residual, including PAO metric and both TNO responses.

    Returning None requests the unchanged CPU oracle for an oversized local
    workspace. Screening is the original strict F_CUT_T test, evaluated on
    primal values; every included Fock element remains differentiable.
    """
    i, j, k = state.trip[t]
    f = state.orb.F_lmo
    active = []
    n = state.e_tno[t].size
    if workspace_bytes(n, 0) > max_workspace_bytes:
        return None
    for key, u, occ, old in connections:
        fl = f[occ, old]
        fl_value = ad.primal(fl)
        fl_host = fl_value if ad._xp([fl]) is np else cp.asnumpy(fl_value)
        if abs(float(fl_host)) < state.opt['F_CUT_T']:
            continue
        # Use a genuine residual operand for metadata. Neighbor orbital
        # energies are not dependencies here and may already be released
        # when a consuming reverse sweep replays this checkpoint.
        if workspace_bytes(n, state.T3[u].shape[0]) > max_workspace_bytes:
            return None
        active.append((key, u, fl))
    overlaps = {}
    groups = overlap_batches((u for _, u, _ in active), state.X_tno[t],
                             state.t_paos, state.X_tno)
    for group, union, _ in groups:
        metric = state.orb.S_pao[np.ix_(state.t_paos[t], union)]
        coefficients = tuple(state.X_tno[u] for u in group)
        positions = tuple(np.searchsorted(union, state.t_paos[u]) for u in group)
        overlaps.update(zip(group, common_left_batch(metric, state.X_tno[t],
                                                     coefficients, positions)))
    edges = [(state.T3[u], overlaps[u], fl, key) for key, u, fl in active]
    return coupled_residual(state.W3[t], state.T3[t], state.e_tno[t],
                            f[i, i]+f[j, j]+f[k, k], edges)
