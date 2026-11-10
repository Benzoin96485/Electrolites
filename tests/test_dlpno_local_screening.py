import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.dft import numint

from electrolites.dlpno.local_screening import gaussian_ao_radii, compute_local_screening, atom_neighbors
from electrolites.dlpno.local import LocalDLPNO
from electrolites.dlpno.screening import row_above, column_above, atom_complete_mask


@pytest.mark.parametrize('cart', [False, True])
def test_gaussian_radial_envelope_for_s_p_d_shells(cart):
    mol = gto.M(atom='C 0 0 0', basis='def2-svp', cart=cart, verbose=0)
    radius = gaussian_ao_radii(mol, 1e-12)[0]
    rng = np.random.default_rng(123)
    directions = rng.normal(size=(500, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    points = directions*radius*rng.uniform(1., 1.8, (500, 1))
    ao = numint.eval_ao(mol, points, cutoff=1e-30)
    assert np.max(abs(ao)) < 1e-12
    with pytest.raises(ValueError, match='tolerance'):
        gaussian_ao_radii(mol, 0.)


def test_local_grid_preserves_dense_domain_selections_and_has_exact_limit(record_property):
    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57; '
                     'O 0 0 12; H .02 .76 12.59; H .09 -.75 12.57', basis='6-31g', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-13)
    base = LocalDLPNO(mf, triples=None, verbose=0); base.setup()
    oracle = compute_local_screening(base, coefficient_cut=1e-14, grid_radius=100., partition_radius=150.)
    actual = compute_local_screening(base)
    for candidate, label in ((oracle, 'expanded'), (actual, 'production')):
        for table, expected, threshold in zip(candidate[:3], (base.doi_ij, base.doi_iu, base.doi_uv),
                                             (base.opt['T_CUT_DO_IJ'], base.opt['T_CUT_DO_PRE'], base.opt['T_CUT_DO_UV'])):
            assert np.array_equal((table > threshold).to_dense(), expected > threshold)
        for threshold in (base.opt['T_CUT_DO'], base.opt['T_CUT_DO_TRIPLES_PRE'], base.opt['T_CUT_DO_TRIPLES']):
            for i in range(base.n):
                assert np.array_equal(row_above(candidate[1], i, threshold), row_above(base.doi_iu, i, threshold))
        for threshold in (base.opt['T_CUT_MKN'], base.opt['T_CUT_MKN_TRIPLES_PRE'], base.opt['T_CUT_MKN_TRIPLES']):
            for i in range(base.n):
                assert np.array_equal(column_above(candidate[3], i, threshold), column_above(base.mkn, i, threshold))
        mask = atom_complete_mask(candidate[2], base.opt['T_CUT_DO_UV'], base.bf_atom, mol.natm)
        assert np.array_equal(mask.to_dense(), base.pao_pair_mask)
        error = max(abs(table[i, j]-expected[i, j])
                    for table, expected in zip(candidate[:3], (base.doi_ij, base.doi_iu, base.doi_uv))
                    for i, row in enumerate(table.rows) for j in row)
        record_property(label+'_maximum_retained_doi_error', error)
        if label == 'expanded':
            assert error < 1e-11
    assert actual[-1]['producer'] == 'local_spatial_grid'
    assert not actual[-1]['total_doi_error_certified']


def test_fixed_radius_neighbor_work_is_bounded_for_a_chain():
    records = []
    for n in (128, 256, 512):
        positions = np.zeros((n, 3)); positions[:, 0] = 2*np.arange(n)
        neighbors, count = atom_neighbors(positions, 10.)
        assert max(map(len, neighbors)) == 11
        records.append(count)
    assert records[2] < 2.1*records[1] < 2.1**2*records[0]
