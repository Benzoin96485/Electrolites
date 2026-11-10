"""Mutual Cartesian tree for the complete separated dipole interaction.

Every unordered source pair is either evaluated directly or belongs to
exactly one cell-cell interaction. No distance cutoff removes a tail.
The reverse sweep differentiates the *same scalar energy*, including
translations of moments and the response of all cell centroids.

Fixed expansion order, quadrature size, grid width, bounded source density
and bounded exclusion range give linear storage/work for the hierarchy.
These conditions do not make the other DLPNO preparation stages linear.
The energy remainder bound excludes floating point error and is not a
certificate for force error. Discrete tree/interaction choices are fixed
in the derivative, like other DLPNO discrete selections.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import time

import numpy as np
from scipy import sparse, special

from .dipole_laplace import SYMMETRIC_BASIS, quadrature_pair


@lru_cache(maxsize=24)
def _layout(order):
    indices = tuple((a, b, degree-a-b) for degree in range(order+1)
                    for a in range(degree+1) for b in range(degree-a+1))
    lookup = {alpha: i for i, alpha in enumerate(indices)}
    powers = np.array(indices, dtype=int)
    factorial = np.array([math.prod(math.factorial(x) for x in alpha)
                          for alpha in indices], dtype=float)
    lower = np.full((3, len(indices)), -1, dtype=int)
    for i, alpha in enumerate(indices):
        for axis in range(3):
            beta = list(alpha)
            beta[axis] -= 1
            lower[axis, i] = lookup.get(tuple(beta), -1)
    left, right, difference = [], [], []
    for i, alpha in enumerate(indices):
        for j, beta in enumerate(indices):
            delta = tuple(a-b for a, b in zip(alpha, beta))
            if min(delta) >= 0:
                left.append(i); right.append(j); difference.append(lookup[delta])
    return dict(indices=indices, lookup=lookup, powers=powers, factorial=factorial,
                lower=lower, target=np.array(left), source=np.array(right),
                difference=np.array(difference))


def monomials(displacement, order):
    """Cartesian monomials u**alpha/alpha! and their spatial derivatives."""
    layout = _layout(order)
    displacement = np.asarray(displacement)
    values = np.prod(displacement[..., None, :]**layout['powers'], axis=-1)/layout['factorial']
    response = np.zeros(values.shape+(3,))
    for axis, lower in enumerate(layout['lower']):
        use = lower >= 0
        response[..., use, axis] = values[..., lower[use]]
    return values, response


def _translate(displacement, order):
    layout = _layout(order)
    value, derivative = monomials(displacement, order)
    size = len(value)
    matrix, response = np.zeros((size, size)), np.zeros((3, size, size))
    a, b, delta = layout['target'], layout['source'], layout['difference']
    matrix[a, b] = value[delta]
    response[:, a, b] = derivative[delta].T
    return matrix, response


def kernel_taylor(displacement, order):
    """D**alpha K(R)/alpha! through total degree ``order``.

    K_ab = delta_ab/r**6 - 6*(B_a r).(B_b r)/r**8
           + 9*(r.B_a.r)*(r.B_b.r)/r**10.
    Polynomial inverse powers use the finite binomial expansion of
    (1 + (2 R.u + u.u)/R**2)**(-s). No numerical differentiation.
    """
    r = np.asarray(displacement, dtype=float)
    radius2 = float(r@r)
    if not radius2 > 0 or not np.isfinite(r).all():
        raise ValueError('Distinct finite cell centers are required')
    layout = _layout(order)
    lookup, indices = layout['lookup'], layout['indices']
    size = len(indices)
    shifts = []
    for axis in range(3):
        for degree, coefficient in ((1, 2*r[axis]/radius2), (2, 1/radius2)):
            src, dst = [], []
            for i, alpha in enumerate(indices):
                beta = list(alpha); beta[axis] += degree
                j = lookup.get(tuple(beta))
                if j is not None:
                    src.append(i); dst.append(j)
            shifts.append((np.array(src, dtype=int), np.array(dst, dtype=int), coefficient))
    power = np.zeros(size); power[0] = 1.
    inverse = np.zeros((3, size))
    factors = np.ones(3)
    for degree in range(order+1):
        inverse += factors[:, None]*power
        next_power = np.zeros(size)
        for src, dst, coefficient in shifts:
            next_power[dst] += coefficient*power[src]
        power = next_power
        factors *= -(np.arange(3, 6)+degree)/(degree+1)
    inverse *= radius2**(-np.arange(3., 6.))[:, None]

    # Numerator polynomials are degree two/four, independent of order.
    small = _layout(2)['indices']
    v = SYMMETRIC_BASIS@r
    s_poly, c_poly = {}, {}
    zero = (0, 0, 0)
    s_poly[zero] = v@v.T
    c_poly[zero] = np.einsum('i,aij,j->a', r, SYMMETRIC_BASIS, r)
    for axis in range(3):
        alpha = tuple(int(k == axis) for k in range(3))
        unit = SYMMETRIC_BASIS[:, :, axis]
        s_poly[alpha] = unit@v.T+v@unit.T
        c_poly[alpha] = 2*v[:, axis]
        for other in range(axis, 3):
            alpha = tuple(int(k == axis)+int(k == other) for k in range(3))
            left, right = SYMMETRIC_BASIS[:, :, axis], SYMMETRIC_BASIS[:, :, other]
            s_poly[alpha] = left@right.T if axis == other else left@right.T+right@left.T
            c_poly[alpha] = SYMMETRIC_BASIS[:, axis, other]*(1 if axis == other else 2)
    cc_poly = {}
    for alpha in small:
        for beta in small:
            gamma = tuple(a+b for a, b in zip(alpha, beta))
            cc_poly[gamma] = cc_poly.get(gamma, 0.)+np.outer(c_poly[alpha], c_poly[beta])
    result = inverse[0, :, None, None]*np.eye(6)
    for polynomial, radial, factor in ((s_poly, inverse[1], -6.), (cc_poly, inverse[2], 9.)):
        for alpha, matrix in polynomial.items():
            src, dst = [], []
            for i, beta in enumerate(indices):
                j = lookup.get(tuple(a+b for a, b in zip(alpha, beta)))
                if j is not None:
                    src.append(i); dst.append(j)
            result[dst] += factor*radial[src, None, None]*matrix
    return result


def kernel_remainder_bound(order, radius, extent):
    """Spectral-norm bound on the total-degree Cartesian kernel remainder.

    For eta=h/R<1, Gegenbauer coefficients obey
    |C_n**s(cos(theta))| <= binomial(n+2*s-1,n). Multiplying the
    degree 0/2/4 numerator polynomials yields the bound below.
    No assumption about the orientation or sign of source tensors.
    """
    eta = float(extent/radius)
    if not 0 <= eta < 1:
        return math.inf
    if eta == 0:
        return 0.
    def tail(s, first):
        return (1-eta)**(-2*s)*(special.betainc(first, 2*s, eta) if first > 0 else 1.)
    bound = tail(3, order+1)
    bound += 6*sum(math.comb(2, k)*eta**k*tail(4, order-k+1) for k in range(3))
    bound += 9*sum(math.comb(4, k)*eta**k*tail(5, order-k+1) for k in range(5))
    return float(bound/radius**6)


@lru_cache(maxsize=24)
def _interaction_layout(order):
    layout, full = _layout(order), _layout(order+1)
    left, right, gamma, response, sign = [], [], [], [], []
    for i, alpha in enumerate(layout['indices']):
        for j, beta in enumerate(layout['indices']):
            delta = tuple(a+b for a, b in zip(alpha, beta))
            if sum(delta) > order:
                continue
            left.append(i); right.append(j); gamma.append(full['lookup'][delta])
            response.append([full['lookup'][tuple(d+int(k == axis) for k, d in enumerate(delta))]
                             for axis in range(3)])
            sign.append((-1.)**sum(beta))
    left, right = np.array(left), np.array(right)
    pointers = np.r_[0, np.cumsum(np.bincount(left, minlength=len(layout['indices'])))]
    return right, pointers, np.array(gamma), np.array(response), np.array(sign)


def _interaction_matrices(displacement, order, derivatives=True, coefficients=None):
    if coefficients is None:
        coefficients = kernel_taylor(displacement, order+1)*_layout(order+1)['factorial'][:, None, None]
    right, pointers, gamma, response, sign = _interaction_layout(order)
    size = 6*len(_layout(order)['indices'])
    def matrix(which):
        return sparse.bsr_matrix((sign[:, None, None]*coefficients[which], right, pointers), shape=(size, size))
    return matrix(gamma), tuple(matrix(response[:, axis]) for axis in range(3)) if derivatives else ()


@dataclass(frozen=True)
class _Cell:
    children: tuple
    sources: tuple
    count: int


class MutualTree:
    """Fixed-width hashed leaves, bottom-up cells and mutual interactions.

    ``exclusions`` contains unordered pair indices retained explicitly by
    the correlated solver. A cell pair is accepted only when its minimum
    separation exceeds the longest excluded separation. Thus exclusions
    are always resolved in direct leaves, never subtracted inconsistently
    from an approximate far interaction. This is a switch of evaluation
    method, not an interaction cutoff.
    """
    def __init__(self, centers, *, order=6, theta=.3, leaf_width=4., exclusions=()):
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all() or not len(centers):
            raise ValueError('Finite nonempty (nsource,3) centers are required')
        if not isinstance(order, int) or not 0 <= order <= 18:
            raise ValueError('Taylor order must be an integer between zero and 18')
        if not 0 < theta < 1 or not math.isfinite(leaf_width) or leaf_width <= 0:
            raise ValueError('Require 0 < theta < 1 and a positive finite leaf width')
        self.order, self.theta, self.leaf_width = order, float(theta), float(leaf_width)
        self.centers = centers.copy()
        self.exclusions = frozenset((min(int(i), int(j)), max(int(i), int(j)))
                                    for i, j in exclusions if i != j)
        if any(i < 0 or j >= len(centers) for i, j in self.exclusions):
            raise ValueError('Excluded pair indices out of range')
        coordinates = np.floor(centers/leaf_width).astype(np.int64)
        coordinates -= coordinates.min(axis=0)
        groups = {}
        for i, coordinate in enumerate(coordinates):
            groups.setdefault(tuple(coordinate), []).append(i)
        cells, level = [], {}
        for key, indices in groups.items():
            level[key] = len(cells)
            cells.append(_Cell((), tuple(indices), len(indices)))
        depth = 1
        while len(level) > 1:
            groups = {}
            for key, index in level.items():
                groups.setdefault(tuple(k//2 for k in key), []).append(index)
            level = {}
            for key, children in groups.items():
                level[key] = len(cells)
                cells.append(_Cell(tuple(children), (), sum(cells[i].count for i in children)))
            depth += 1
        self.cells, self.root, self.depth = tuple(cells), next(iter(level.values())), depth
        centroids, radii = self._geometry(centers)
        exclusion_range = self._exclusion_range(centers)
        far, near, pending = [], [], [(self.root, self.root)]
        visits = 0
        while pending:
            a, b = pending.pop(); visits += 1
            left, right = cells[a], cells[b]
            if a == b:
                if left.children:
                    pending.extend((x, y) for k, x in enumerate(left.children) for y in left.children[k:])
                else:
                    near.append((a, b))
                continue
            distance = np.linalg.norm(centroids[a]-centroids[b])
            extent = radii[a]+radii[b]
            if extent < theta*distance and distance-extent > exclusion_range:
                far.append((a, b))
            elif not left.children and not right.children:
                near.append((a, b))
            elif left.children and (not right.children or radii[a] >= radii[b]):
                pending.extend((child, b) for child in left.children)
            else:
                pending.extend((a, child) for child in right.children)
        self.far, self.near, self.visits = tuple(far), tuple(near), visits
        represented = sum(cells[a].count*cells[b].count for a, b in far)
        represented += sum(cells[a].count*cells[b].count if a != b else cells[a].count*(cells[a].count-1)//2
                           for a, b in near)
        if represented != len(centers)*(len(centers)-1)//2:
            raise RuntimeError('Mutual tree pair partition is incomplete or duplicated')
        self.represented_pairs = represented-len(self.exclusions)
        signature = repr((self.cells, self.far, self.near, sorted(self.exclusions))).encode()
        self.signature = hashlib.sha256(signature).hexdigest()

    def _exclusion_range(self, centers):
        return max((float(np.linalg.norm(centers[i]-centers[j])) for i, j in self.exclusions), default=0.)

    def _geometry(self, centers):
        centroids, radii = np.zeros((len(self.cells), 3)), np.zeros(len(self.cells))
        for k, cell in enumerate(self.cells):
            if cell.children:
                centroids[k] = sum(self.cells[c].count*centroids[c] for c in cell.children)/cell.count
                radii[k] = max(radii[c]+np.linalg.norm(centroids[c]-centroids[k]) for c in cell.children)
            else:
                points = centers[list(cell.sources)]
                centroids[k] = np.mean(points, axis=0)
                radii[k] = np.max(np.linalg.norm(points-centroids[k], axis=1))
        return centroids, radii

    def direct_pairs(self):
        """Unordered, nonexcluded pairs assigned to direct leaf interactions."""
        for a, b in self.near:
            left, right = self.cells[a].sources, self.cells[b].sources
            for row, i in enumerate(left):
                for j in (right[row+1:] if a == b else right):
                    if (min(i, j), max(i, j)) not in self.exclusions:
                        yield i, j

    def evaluate(self, sources, *, centers=None, channel_block=32, include_near=True, derivatives=True,
                 geometry_cache_bytes=64 << 20):
        """Return scalar energy, source bars, center bars and diagnostics.

        Passing displaced centers explicitly retains this tree branch.
        Its separation/exclusion bounds are checked anew. All continuous
        moments, cell centers, kernels and their responses are recomputed.
        """
        centers = self.centers if centers is None else np.asarray(centers, dtype=float)
        sources = np.asarray(sources, dtype=float)
        if centers.shape != self.centers.shape or not np.isfinite(centers).all():
            raise ValueError('Centers must match the tree source count and be finite')
        if sources.ndim != 3 or sources.shape[0] != len(centers) or sources.shape[2] != 6 or not np.isfinite(sources).all():
            raise ValueError('Sources must be finite (nsource,nquadrature,6) tensors')
        if channel_block < 1:
            raise ValueError('channel_block must be positive')
        if not isinstance(geometry_cache_bytes, (int, np.integer)) or geometry_cache_bytes < 0:
            raise ValueError('Geometry cache budget must be a nonnegative integer')
        centroids, radii = self._geometry(centers)
        exclusion_range = self._exclusion_range(centers)
        for a, b in self.far:
            distance = np.linalg.norm(centroids[a]-centroids[b])
            if distance <= radii[a]+radii[b] or distance-radii[a]-radii[b] <= exclusion_range:
                raise ValueError('Displacement invalidates the fixed tree separation/exclusion branch')
        energy, energy_bound = 0., 0.
        source_bar = np.zeros_like(sources) if derivatives else None
        center_bar = np.zeros_like(centers) if derivatives else None
        size = len(_layout(self.order)['indices'])
        peak_moments = 0
        # These coefficients depend on the cell geometry, not the
        # quadrature channel. Cache their compact Taylor arrays rather
        # than four much larger sparse interaction matrices. The cache
        # is fresh for every evaluation, including displaced geometries.
        geometry_cache = {}
        geometry_info = dict(numeric_payload_bytes=0, budget_bytes=int(geometry_cache_bytes),
                             kernel_builds=0, kernel_cache_hits=0, kernel_build_seconds=0.)
        def interaction_geometry(a, b):
            key = a, b
            if key in geometry_cache:
                geometry_info['kernel_cache_hits'] += 1
                return geometry_cache[key]
            start = time.perf_counter()
            displacement = centroids[a]-centroids[b]
            coefficients = kernel_taylor(displacement, self.order+1)*_layout(self.order+1)['factorial'][:, None, None]
            bound = kernel_remainder_bound(self.order, np.linalg.norm(displacement), radii[a]+radii[b])
            geometry_info['kernel_builds'] += 1
            geometry_info['kernel_build_seconds'] += time.perf_counter()-start
            payload = coefficients.nbytes+8
            if geometry_info['numeric_payload_bytes']+payload <= geometry_cache_bytes:
                geometry_cache[key] = coefficients, bound
                geometry_info['numeric_payload_bytes'] += payload
            return coefficients, bound
        for start in range(0, sources.shape[1], channel_block):
            stop = min(start+channel_block, sources.shape[1])
            block = sources[:, start:stop]
            width = stop-start
            moments = np.zeros((len(self.cells), size, width, 6))
            moment_bar = np.zeros_like(moments) if derivatives else None
            weights = np.zeros((len(self.cells), width))
            centroid_bar = np.zeros_like(centroids)
            peak_moments = max(peak_moments, moments.nbytes*(2 if derivatives else 1)+weights.nbytes)
            for k, cell in enumerate(self.cells):
                if cell.children:
                    for child in cell.children:
                        translation, _ = _translate(centroids[child]-centroids[k], self.order)
                        moments[k] += (translation@moments[child].reshape(size, -1)).reshape(size, width, 6)
                        weights[k] += weights[child]
                else:
                    ids = list(cell.sources)
                    values, _ = monomials(centers[ids]-centroids[k], self.order)
                    moments[k] = np.einsum('im,iqa->mqa', values, block[ids], optimize=True)
                    weights[k] = np.linalg.norm(block[ids], axis=2).sum(axis=0)
            for a, b in self.far:
                displacement = centroids[a]-centroids[b]
                coefficients, kernel_bound = interaction_geometry(a, b)
                matrix, derivative = _interaction_matrices(displacement, self.order, derivatives, coefficients)
                left = moments[a].transpose(0, 2, 1).reshape(6*size, width)
                right = moments[b].transpose(0, 2, 1).reshape(6*size, width)
                left_bar = -8*(matrix@right)
                energy += float(np.sum(left*left_bar))
                if derivatives:
                    right_bar = -8*(matrix.T@left)
                    moment_bar[a] += left_bar.reshape(size, 6, width).transpose(0, 2, 1)
                    moment_bar[b] += right_bar.reshape(size, 6, width).transpose(0, 2, 1)
                    gradient = np.array([-8*np.sum(left*(entry@right)) for entry in derivative])
                    centroid_bar[a] += gradient; centroid_bar[b] -= gradient
                energy_bound += 8*kernel_bound*float(weights[a]@weights[b])
            # Adjoint of the upward pass is the downward field pass.
            for k in (range(len(self.cells)-1, -1, -1) if derivatives else ()):
                cell = self.cells[k]
                if cell.children:
                    bar = moment_bar[k].reshape(size, -1)
                    for child in cell.children:
                        translation, derivative = _translate(centroids[child]-centroids[k], self.order)
                        moment_bar[child] += (translation.T@bar).reshape(size, width, 6)
                        local = moments[child].reshape(size, -1)
                        gradient = np.array([np.sum(bar*(entry@local)) for entry in derivative])
                        centroid_bar[child] += gradient; centroid_bar[k] -= gradient
                    for child in cell.children:
                        centroid_bar[child] += centroid_bar[k]*self.cells[child].count/cell.count
                else:
                    ids = list(cell.sources)
                    values, derivative = monomials(centers[ids]-centroids[k], self.order)
                    source_bar[ids, start:stop] += np.einsum('im,mqa->iqa', values, moment_bar[k], optimize=True)
                    gradient = np.einsum('imx,iqa,mqa->ix', derivative, block[ids], moment_bar[k], optimize=True)
                    center_bar[ids] += gradient+(centroid_bar[k]-gradient.sum(axis=0))/cell.count
        direct_pairs = 0
        if include_near:
            for i, j in self.direct_pairs():
                e, bi, bj, g = quadrature_pair(sources[i], sources[j], centers[i]-centers[j])
                energy += e
                if derivatives:
                    source_bar[i] += bi; source_bar[j] += bj
                    center_bar[i] += g; center_bar[j] -= g
                direct_pairs += 1
        info = dict(order=self.order, theta=self.theta, leaf_width=self.leaf_width,
                    sources=len(centers), channels=sources.shape[1], cells=len(self.cells), depth=self.depth,
                    largest_leaf=max(len(c.sources) for c in self.cells),
                    visits=self.visits, far_cell_pairs=len(self.far), near_leaf_pairs=len(self.near),
                    direct_pairs=direct_pairs, represented_pairs=self.represented_pairs,
                    exclusions=len(self.exclusions), exclusion_range=exclusion_range,
                    peak_moment_bytes=peak_moments, topology_sha256=self.signature,
                    geometry_cache=geometry_info,
                    analytic_energy_absolute_bound=energy_bound,
                    bound_excludes_roundoff=True, force_error_certified=False)
        return energy, source_bar, center_bar, info


def tree_energy_ad(centers, sources, **options):
    """One scalar primitive with the reverse of the same hierarchy."""
    from . import reverse as ad
    x, p = ad.primal(centers), ad.primal(sources)
    tree = MutualTree(x, **options)
    energy, pb, xb, _ = tree.evaluate(p)
    return ad.operation(np.asarray(energy), [centers, sources], lambda bar: (bar*xb, bar*pb))
