"""Complete distant-pair energy/adjoint with exact near and hierarchical far.

Inputs are the existing DLPNO local canonical transition dipoles and
positive excitation gaps. This module accelerates their complete sum;
orbital preparation, domain selection and nuclear relaxation are callers'
responsibilities. The default production DLPNO driver is unchanged.
"""
import numpy as np

from .dipole_laplace import (LaplaceQuadrature, polarizability,
                              polarizability_pullback, exact_pair)
from .dipole_tree import MutualTree


class DipoleTail:
    """Explicit, geometry-independent quadrature and hierarchy settings."""
    def __init__(self, *, gap_bounds=(1e-3, 1e3), quadrature_tolerance=1e-12,
                 order=6, theta=.25, leaf_width=4., channel_block=32,
                 geometry_cache_bytes=64 << 20):
        self.quadrature = LaplaceQuadrature.build(gap_bounds, quadrature_tolerance)
        self.tree_options = dict(order=order, theta=theta, leaf_width=leaf_width)
        self.channel_block = channel_block
        self.geometry_cache_bytes = geometry_cache_bytes

    def evaluate(self, centers, dipoles, gaps, *, exclusions=(), tree=None, derivatives=True):
        """Return energy, local dipole/gap/center sources and diagnostics.

        A supplied ``tree`` explicitly freezes its discrete branch for
        validation. The quadrature never adapts to displaced gaps.
        Near pairs use the original virtual-index denominator exactly.
        The reported bound covers the scalar energy, not force error.
        """
        centers = np.asarray(centers, dtype=float)
        if len(centers) != len(dipoles) or len(dipoles) != len(gaps):
            raise ValueError('One dipole matrix and gap vector per center are required')
        if tree is None:
            tree = MutualTree(centers, exclusions=exclusions, **self.tree_options)
        else:
            excluded = frozenset(tuple(sorted((int(i), int(j)))) for i, j in exclusions if i != j)
            if tree.exclusions != excluded:
                raise ValueError('Frozen tree exclusions differ from the parent energy')
        sources = np.stack([polarizability(d, g, self.quadrature) for d, g in zip(dipoles, gaps)])
        far_energy, source_bar, center_bar, info = tree.evaluate(
            sources, centers=centers, channel_block=self.channel_block, include_near=False,
            derivatives=derivatives, geometry_cache_bytes=self.geometry_cache_bytes)
        dipole_bar, gap_bar = [], []
        if derivatives:
            for d, g, bar in zip(dipoles, gaps, source_bar):
                db, gb = polarizability_pullback(d, g, self.quadrature, bar)
                dipole_bar.append(db); gap_bar.append(gb)
        near_energy, count = 0., 0
        for i, j in tree.direct_pairs():
            e, di, gi, dj, gj, x = exact_pair(dipoles[i], gaps[i], dipoles[j], gaps[j],
                                             centers[i]-centers[j], derivatives=derivatives)
            near_energy += e
            if derivatives:
                dipole_bar[i] += di; gap_bar[i] += gi
                dipole_bar[j] += dj; gap_bar[j] += gj
                center_bar[i] += x; center_bar[j] -= x
            count += 1
        # Positive quadrature integrates a sum of squares. Its error is
        # relative to the magnitude of the unexpanded far energy. Bound
        # that quantity by |E_tree| plus the Taylor remainder, then solve
        # the relative-error inequality for the exact denominator energy.
        taylor_bound = info['analytic_energy_absolute_bound']
        relative = self.quadrature.energy_relative_bound
        laplace_bound = relative*(abs(far_energy)+taylor_bound)/(1-relative)
        info.update(direct_pairs=count, exact_near_energy=near_energy,
                    hierarchical_far_energy=far_energy,
                    taylor_energy_absolute_bound=taylor_bound,
                    laplace_energy_absolute_bound=laplace_bound,
                    analytic_energy_absolute_bound=taylor_bound+laplace_bound,
                    quadrature=self.quadrature.statistics())
        return far_energy+near_energy, dipole_bar, gap_bar, center_bar, info

    def energy_ad(self, centers, dipoles, gaps, *, exclusions=(), tree=None):
        """Single scalar custom VJP compatible with local orbital response."""
        from . import reverse as ad
        x = ad.primal(centers)
        ds, gs = [ad.primal(d) for d in dipoles], [ad.primal(g) for g in gaps]
        energy, db, gb, xb, _ = self.evaluate(x, ds, gs, exclusions=exclusions, tree=tree)
        parents = [centers, *dipoles, *gaps]
        bars = [xb, *db, *gb]
        return ad.operation(np.asarray(energy), parents, lambda incoming: tuple(incoming*b for b in bars))


def orbital_inputs(base, orb, position_integrals, *, einsum=np.einsum, canonicalize=None):
    """Prepare transition dipoles, with the parent's fixed orbital supports.

    Without local spatial screening this is the original dense oracle.
    With screening, each occupied orbital uses only its support and the
    supports of its local PAO domain. Truncated orbitals are not exactly
    orthogonal: use c=<l|r|l>/<l|S|l> and <l|r-c*S|p> so translation
    covariance survives coefficient screening. This is an explicit
    local approximation, with the original expression as its full-support
    limit. Reverse algebra includes center normalization, overlap, PAO,
    coefficient and gap response on the same discrete branch.
    """
    from . import setup as su
    canonicalize = su.orthocanonicalize if canonicalize is None else canonicalize
    cl, p, r, opt = orb['cl'], orb['p'], position_integrals, base.opt
    maps = getattr(base, '_local_dipole_maps', None)
    if maps is None:
        centers = einsum('mi,xmn,ni->ix', cl, r, cl, optimize=True)
        lp = einsum('mi,xmn,nu->ixu', cl, r, p, optimize=True)
    else:
        centers = []
        counts = dict(preparation='local_centered_transition_dipoles',
                      numerical_approximation=True, coefficient_cut=maps.cutoffs,
                      maximum_ao=0, maximum_pao=0, ao_square_pao_work=0)
    ds, gs = [], []
    for i in range(base.n):
        domain = su.contract_to_atoms(su.row_above(base.doi_iu, i, opt['T_CUT_DO_PRE']),
                                      base.bf_atom, base.atom_bfs)
        x, e = canonicalize(orb['sp'][np.ix_(domain, domain)],
                             orb['fp'][np.ix_(domain, domain)], opt['S_CUT'])
        if maps is None:
            transition = lp[i][:, domain]
        else:
            atoms = np.union1d(maps.atoms['lmo'][i], maps.union('pao', domain))
            rows = np.concatenate([base.atom_bfs[int(a)] for a in atoms])
            left = cl[rows, i]*maps.mask('lmo', [i], rows)[:, 0]
            right = p[np.ix_(rows, domain)]*maps.mask('pao', domain, rows)
            overlap = orb['s'][np.ix_(rows, rows)]
            position = r[:, rows[:, None], rows[None, :]]
            norm = left@overlap@left
            center = einsum('m,xmn,n->x', left, position, left, optimize=True)/norm
            centered = position-center[:, None, None]*overlap[None, :, :]
            transition = einsum('m,xmn,nu->xu', left, centered, right, optimize=True)
            centers.append(center)
            counts['maximum_ao'] = max(counts['maximum_ao'], len(rows))
            counts['maximum_pao'] = max(counts['maximum_pao'], len(domain))
            counts['ao_square_pao_work'] += len(rows)**2*len(domain)
        ds.append(transition@x)
        gs.append(e-orb['fl'][i, i])
    if maps is not None:
        from . import reverse as ad
        centers = ad.Numpy().array(centers)
        base.dipole_preparation_diagnostics = counts
    return centers, ds, gs
