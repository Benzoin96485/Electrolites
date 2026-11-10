"""Sparse, conservative discovery of the existing dipole preselection.

This does not change the preselection formula and does not remove any
distant-pair energy. Spatial hashing enumerates all pairs that *could*
exceed its threshold; only those require an explicit virtual-index sum.
The optional hierarchical-tail driver uses this adapter. DOI/domain
construction still uses dense tables and is separate remaining work.
"""
from itertools import product
import math

import numpy as np


def excitation_strength(dipoles, gaps):
    """a_i = sum_a ||d_ia||**2/sqrt(g_ia), with strictly positive gaps."""
    dipoles, gaps = np.asarray(dipoles), np.asarray(gaps)
    if dipoles.shape != (3, len(gaps)) or not np.isfinite(dipoles).all():
        raise ValueError('Finite local dipoles of shape (3,nvirtual) are required')
    if not np.isfinite(gaps).all() or np.any(gaps <= 0):
        raise ValueError('Positive finite excitation gaps are required for the bound')
    return float(np.sum(dipoles*dipoles/np.sqrt(gaps)[None, :]))


def bound_pairs(centers, strengths, threshold):
    """Discover candidate unordered pairs without an all-pairs table.

    For the original B_ij=-16/R**6 sum_ab (d_ia.d_jb)**2/(g_ia+g_jb),
    Cauchy-Schwarz and x+y >= 2 sqrt(x*y) give
    |B_ij| <= 8*a_i*a_j/R**6. The largest such radius sets hash-cell
    width. Under bounded density, strengths, inverse threshold and gap,
    enumerating the 27 neighboring cells costs O(N). The original
    coincident-center rule (distance < 1e-12) is preserved as well.
    """
    centers, strengths = np.asarray(centers, dtype=float), np.asarray(strengths, dtype=float)
    if centers.shape != (len(strengths), 3) or not np.isfinite(centers).all():
        raise ValueError('One finite Cartesian center per strength is required')
    if not np.isfinite(strengths).all() or np.any(strengths < 0):
        raise ValueError('Nonnegative finite excitation strengths are required')
    if not 0 < threshold < math.inf:
        raise ValueError('Sparse preselection requires a positive finite threshold')
    maximum = float(np.max(strengths, initial=0.))
    # Guard the analytic radius by roundoff; false positives are harmless.
    # A false negative could silently change a retained energy branch.
    guard = 1+128*np.finfo(float).eps
    radius = (8*maximum*maximum/threshold)**(1/6) if maximum else 0.
    width = max(radius*guard, 1e-12*guard)
    scaled = np.floor(centers/width)
    if np.max(abs(scaled), initial=0) >= np.iinfo(np.int64).max:
        raise ValueError('Spatial hash coordinates overflow; translate or rescale the input')
    keys = scaled.astype(np.int64)
    cells = {}
    for i, key in enumerate(keys):
        cells.setdefault(tuple(key), []).append(i)
    offsets = tuple(product((-1, 0, 1), repeat=3))
    candidates, distance_tests = [], 0
    for key, left in cells.items():
        for offset in offsets:
            other = tuple(k+d for k, d in zip(key, offset))
            if other < key or other not in cells:
                continue
            right = cells[other]
            for position, i in enumerate(left):
                for j in (right[position+1:] if other == key else right):
                    distance_tests += 1
                    distance = np.linalg.norm(centers[i]-centers[j])
                    limit = (8*strengths[i]*strengths[j]/threshold)**(1/6)
                    if distance < 1e-12 or distance <= limit*guard:
                        candidates.append((min(i, j), max(i, j)))
    # Hash traversal and orbital indexing are deterministic without an
    # O(N log N) global sort. Callers may build per-row sorted adjacency.
    return candidates, dict(sources=len(centers), cells=len(cells),
                            largest_cell=max(map(len, cells.values()), default=0),
                            distance_tests=distance_tests, virtual_pair_candidates=len(candidates),
                            maximum_candidate_radius=radius, threshold=threshold)


def select_dipole_pairs(centers, dipoles, gaps, threshold):
    """Return retained unordered pairs and exact legacy B_ij values.

    This selector is independent of DOI. Union its result with the DOI
    pairs and the diagonal to obtain the original prescreen mask.
    Unselected pairs still contribute their complete dipole energy.
    """
    if len(centers) != len(dipoles) or len(dipoles) != len(gaps):
        raise ValueError('One local excitation space per orbital is required')
    strengths = np.array([excitation_strength(d, g) for d, g in zip(dipoles, gaps)])
    candidates, info = bound_pairs(centers, strengths, threshold)
    retained = {}
    for i, j in candidates:
        distance = np.linalg.norm(centers[i]-centers[j])
        if distance < 1e-12:
            retained[i, j] = -np.inf
            continue
        dot = np.asarray(dipoles[i]).T@np.asarray(dipoles[j])
        denominator = np.asarray(gaps[i])[:, None]+np.asarray(gaps[j])[None, :]
        value = float(-16/distance**6*np.sum(dot**2/denominator))
        if abs(value) > threshold:
            retained[i, j] = value
    info.update(retained_pairs=len(retained), dense_pair_storage=False,
                unchanged_legacy_selection=True, all_energy_tails_retained=True)
    return retained, info
