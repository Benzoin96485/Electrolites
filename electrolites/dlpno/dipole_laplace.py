"""Separable representation of the *complete* distant-pair dipole kernel.

Positive Laplace quadrature separates the local excitation denominators.
Its nodes and weights depend on fixed, explicit gap bounds, not on the
current geometry. Both energy and gap response have analytic quadrature
error bounds (floating-point roundoff is separate).

This module is the algebraic prerequisite for a hierarchical far-field
sum. ``quadrature_pair`` is a two-center reference, not a linear-scaling
many-center algorithm. The existing production energy is unchanged.
"""
from dataclasses import dataclass
import math

import numpy as np


def _log_power_max(power, lower, upper, start):
    x = upper if start == 0. else np.clip(power/start, lower, upper)
    return power*math.log(x)-start*x


def _panel_bounds(start, stop, order, lower, upper):
    """Gauss-Legendre remainder bounds for x*I0 and x**2*I1.

    I0=int exp(-x*t) dt and I1=int t*exp(-x*t) dt. The derivative
    maxima are bounded over both the panel and the denominator interval.
    """
    degree = 2*order
    common = (4*math.lgamma(order+1)-3*math.lgamma(degree+1)
              -math.log(degree+1)+(degree+1)*math.log(stop-start))
    energy = math.exp(common+_log_power_max(degree+1, lower, upper, start))
    response = (math.exp(common+math.log(stop)
                         +_log_power_max(degree+2, lower, upper, start))
                +math.exp(common+math.log(degree)
                          +_log_power_max(degree+1, lower, upper, start)))
    return energy, response


@dataclass(frozen=True)
class LaplaceQuadrature:
    """Fixed quadrature for denominators ``gap_i + gap_j``.

    The gap interval is part of the numerical energy definition. A
    displacement may not silently retune its nodes. A violation raises
    instead; the caller can select a wider fixed interval and rerun.
    """
    gap_bounds: tuple
    tolerance: float
    nodes: np.ndarray
    weights: np.ndarray
    order: int
    panels: tuple
    energy_relative_bound: float
    gap_response_relative_bound: float

    @classmethod
    def build(cls, gap_bounds=(1e-3, 1e3), tolerance=1e-12):
        minimum, maximum = map(float, gap_bounds)
        if not (0. < minimum <= maximum and math.isfinite(maximum)):
            raise ValueError('Positive finite fixed gap bounds are required')
        if not (0. < tolerance < .1):
            raise ValueError('Quadrature tolerance must lie between zero and 0.1')
        lower, upper = 2*minimum, 2*maximum
        # The I1 tail is larger than the I0 tail. Bound both by tol/4.
        target = tolerance/4
        end = max(1., -math.log(target))
        while (1+end)*math.exp(-end) > target:
            end += 1.
        terminal = end/lower
        panels = [(0., min(1/upper, terminal))]
        while panels[-1][1] < terminal:
            start = panels[-1][1]
            panels.append((start, min(2*start, terminal)))
        tails = math.exp(-end), (1+end)*math.exp(-end)
        for order in range(6, 41, 2):
            bounds = np.sum([_panel_bounds(a, b, order, lower, upper)
                             for a, b in panels], axis=0)+tails
            if max(bounds) <= tolerance:
                break
        else:
            raise ValueError('Requested tolerance exceeds the quadrature construction limit')
        grid, weight = np.polynomial.legendre.leggauss(order)
        nodes = np.concatenate([a+(grid+1)*(b-a)/2 for a, b in panels])
        weights = np.concatenate([weight*(b-a)/2 for a, b in panels])
        nodes.flags.writeable = weights.flags.writeable = False
        return cls((minimum, maximum), tolerance, nodes, weights, order,
                   tuple(panels), float(bounds[0]), float(bounds[1]))

    def check_gaps(self, gaps):
        gaps = np.asarray(gaps)
        if not np.isfinite(gaps).all() or np.any(gaps < self.gap_bounds[0]) or np.any(gaps > self.gap_bounds[1]):
            raise ValueError('Excitation gap outside the fixed Laplace interval; '
                             'widen the interval and recompute the parent energy')

    def statistics(self):
        return dict(gap_bounds=self.gap_bounds, tolerance=self.tolerance,
                    nodes=len(self.nodes), panels=len(self.panels), order=self.order,
                    analytic_energy_relative_bound=self.energy_relative_bound,
                    analytic_gap_response_relative_bound=self.gap_response_relative_bound,
                    bound_excludes_floating_point_roundoff=True)


# An orthonormal basis for real symmetric Cartesian 3x3 tensors.
SYMMETRIC_BASIS = np.zeros((6, 3, 3))
SYMMETRIC_BASIS[np.arange(3), np.arange(3), np.arange(3)] = 1.
for index, (a, b) in enumerate(((0, 1), (0, 2), (1, 2)), start=3):
    SYMMETRIC_BASIS[index, a, b] = SYMMETRIC_BASIS[index, b, a] = 1/math.sqrt(2)
SYMMETRIC_BASIS.flags.writeable = False


def polarizability(dipoles, gaps, quadrature):
    """sqrt(w_q) sum_a d_a d_a.T exp(-t_q gap_a), in six components."""
    dipoles, gaps = np.asarray(dipoles), np.asarray(gaps)
    if dipoles.shape != (3, len(gaps)) or not np.isfinite(dipoles).all():
        raise ValueError('Dipoles must be a finite (3, nvirtual) matrix')
    quadrature.check_gaps(gaps)
    attenuation = np.exp(-quadrature.nodes[:, None]*gaps[None, :])
    tensor = np.einsum('ia,qa,ja->qij', dipoles, attenuation, dipoles, optimize=True)
    return np.einsum('bij,qij->qb', SYMMETRIC_BASIS, tensor)*np.sqrt(quadrature.weights)[:, None]


def polarizability_pullback(dipoles, gaps, quadrature, incoming):
    """Analytic local sources for transition dipoles and excitation gaps."""
    quadrature.check_gaps(gaps)
    attenuation = np.exp(-quadrature.nodes[:, None]*np.asarray(gaps)[None, :])
    tensor_bar = np.einsum('qb,bij->qij', np.asarray(incoming), SYMMETRIC_BASIS)
    tensor_bar *= np.sqrt(quadrature.weights)[:, None, None]
    dipole_bar = 2*np.einsum('qij,ja,qa->ia', tensor_bar, dipoles, attenuation, optimize=True)
    gap_bar = -np.einsum('q,qij,ia,ja,qa->a', quadrature.nodes, tensor_bar,
                         dipoles, dipoles, attenuation, optimize=True)
    return dipole_bar, gap_bar


def polarizability_ad(dipoles, gaps, quadrature):
    """Local custom VJP for the existing orbital-source reverse engine."""
    from . import reverse as ad
    d, g = ad.primal(dipoles), ad.primal(gaps)
    value = polarizability(d, g, quadrature)
    return ad.operation(value, [dipoles, gaps],
                        lambda bar: polarizability_pullback(d, g, quadrature, bar))


def dipole_tensor(displacement):
    """T=(I-3*u*u.T)/r**3 and its three analytic spatial derivatives."""
    r = np.asarray(displacement, dtype=float)
    radius2 = float(r@r)
    if not radius2 > 0. or not np.isfinite(r).all():
        raise ValueError('Distinct finite orbital centers are required')
    inverse = 1/math.sqrt(radius2)
    outer = np.outer(r, r)
    tensor = np.eye(3)*inverse**3-3*outer*inverse**5
    derivative = np.empty((3, 3, 3))
    for axis, unit in enumerate(np.eye(3)):
        derivative[axis] = (-3*r[axis]*np.eye(3)*inverse**5
                            -3*(np.outer(unit, r)+np.outer(r, unit))*inverse**5
                            +15*r[axis]*outer*inverse**7)
    return tensor, derivative


def dipole_kernel(displacement):
    """K_ab=Tr[B_a T B_b T], with analytic derivatives of the same K."""
    tensor, derivative = dipole_tensor(displacement)
    transformed = SYMMETRIC_BASIS@tensor
    response = np.einsum('aij,xjk->xaik', SYMMETRIC_BASIS, derivative)
    kernel = np.einsum('aij,bji->ab', transformed, transformed)
    gradient = (np.einsum('xaij,bji->xab', response, transformed)
                +np.einsum('aij,xbji->xab', transformed, response))
    return kernel, gradient


def quadrature_pair(left, right, displacement, multiplicity=2.):
    """One pair energy and sources; multiplicity=2 matches ordered ij/ji."""
    kernel, derivative = dipole_kernel(displacement)
    factor = -4*multiplicity
    energy = factor*np.einsum('qa,ab,qb->', left, kernel, right, optimize=True)
    bar_left, bar_right = factor*right@kernel.T, factor*left@kernel
    gradient = factor*np.einsum('qa,xab,qb->x', left, derivative, right, optimize=True)
    return float(energy), bar_left, bar_right, gradient


def exact_pair(dipoles_i, gaps_i, dipoles_j, gaps_j, displacement, multiplicity=2., derivatives=True):
    """Independent virtual-index reference for the parent dipole formula."""
    tensor, derivative = dipole_tensor(displacement)
    coupled = dipoles_i.T@tensor@dipoles_j
    denominator = np.asarray(gaps_i)[:, None]+np.asarray(gaps_j)[None, :]
    factor = -4*multiplicity
    energy = factor*np.sum(coupled**2/denominator)
    if not derivatives:
        return float(energy), None, None, None, None, None
    bar_coupled = 2*factor*coupled/denominator
    dipoles_i_bar = tensor@dipoles_j@bar_coupled.T
    dipoles_j_bar = tensor@dipoles_i@bar_coupled
    denominator_bar = -factor*coupled**2/denominator**2
    tensor_bar = dipoles_i@bar_coupled@dipoles_j.T
    gradient = np.einsum('ij,xij->x', tensor_bar, derivative)
    return (float(energy), dipoles_i_bar, denominator_bar.sum(axis=1),
            dipoles_j_bar, denominator_bar.sum(axis=0), gradient)
