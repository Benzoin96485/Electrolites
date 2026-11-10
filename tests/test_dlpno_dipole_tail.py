import numpy as np
import pytest

from electrolites.dlpno.dipole_laplace import exact_pair
from electrolites.dlpno.dipole_tail import DipoleTail
from electrolites.dlpno.dipole_tree import MutualTree


def test_screened_transition_dipoles_translation_and_all_input_derivatives():
    from types import SimpleNamespace
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.adjoint import orthocanonicalize
    from electrolites.dlpno.dipole_tail import orbital_inputs
    from electrolites.dlpno.local_integrals import CoefficientMaps
    rng = np.random.default_rng(281)
    cl, p = rng.normal(size=(6, 2)), rng.normal(size=(6, 6))
    s = np.eye(6)+.03*rng.normal(size=(6, 6)); s = s@s.T
    sp = np.diag(np.arange(1., 7.))
    fp = np.diag(np.arange(2., 8.))
    fl = np.diag([-1., -2.])
    r = rng.normal(size=(3, 6, 6)); r += r.transpose(0, 2, 1)
    base = SimpleNamespace(n=2, mol=SimpleNamespace(natm=3),
        orb=SimpleNamespace(C_lmo=cl, C_pao=p), bf_atom=np.repeat(np.arange(3), 2),
        atom_bfs=[np.arange(2*i, 2*i+2) for i in range(3)],
        doi_iu=np.ones((2, 6)), opt=dict(T_CUT_DO_PRE=.1, S_CUT=1e-10))
    maps = CoefficientMaps(base, .01, .01)
    maps.atoms = dict(lmo=[np.array([0, 1]), np.array([1, 2])],
                      pao=[np.array([i//2]) for i in range(6)])
    base._local_dipole_maps = maps
    data = dict(s=s, cl=cl, p=p, sp=sp, fp=fp, fl=fl)
    centers, dipoles, gaps = orbital_inputs(base, data, r)
    shift = np.array([3., -4., 2.])
    moved = orbital_inputs(base, data, r+shift[:, None, None]*s)
    assert np.max(abs(moved[0]-centers-shift)) < 1e-13
    for actual, expected in zip(moved[1], dipoles):
        assert np.max(abs(actual-expected)) < 1e-13
    variables = {name: ad.Variable(value) for name, value in dict(data, r=r).items()}
    def objective(values, reverse=False):
        x, ds, gs = orbital_inputs(base, values, values['r'],
            einsum=ad.einsum if reverse else np.einsum,
            canonicalize=orthocanonicalize if reverse else None)
        return np.sum(x*x)+sum(np.sum(d*d) for d in ds)+sum(np.sum(g*g) for g in gs)
    bars = ad.gradients(objective(variables, True), list(variables.values()))
    values = dict(data, r=r)
    direction = {key: rng.normal(size=value.shape) for key, value in values.items()}
    for key in ('s', 'sp', 'fp'):
        direction[key] = .5*(direction[key]+direction[key].T)
    direction['r'] = .5*(direction['r']+direction['r'].transpose(0, 2, 1))
    step = 1e-6
    fd = (objective({key: value+step*direction[key] for key, value in values.items()})
          -objective({key: value-step*direction[key] for key, value in values.items()}))/(2*step)
    analytic = sum(np.sum(bar*direction[key]) for key, bar in zip(variables, bars))
    assert abs(analytic-fd) < 2e-6


def test_full_tail_exact_near_ragged_gaps_and_derivative():
    rng = np.random.default_rng(941)
    centers = np.array([[.2, .3, .4], [1.5, .5, .7], [.3, 20.2, .5], [1.4, 20.4, .9],
                        [20.3, 20.4, .8], [21.3, 20.7, 1.1]])
    dipoles = [.2*rng.normal(size=(3, n)) for n in (3, 4, 2, 5, 3, 4)]
    gaps = [rng.uniform(.1, 5., d.shape[1]) for d in dipoles]
    exclusions = [(0, 1)]
    driver = DipoleTail(gap_bounds=(.05, 10.), order=6, theta=.25, leaf_width=4.)
    tree = MutualTree(centers, exclusions=exclusions, **driver.tree_options)
    energy, db, gb, xb, info = driver.evaluate(centers, dipoles, gaps, exclusions=exclusions, tree=tree)
    forward = driver.evaluate(centers, dipoles, gaps, exclusions=exclusions, tree=tree, derivatives=False)
    assert forward[0] == energy
    assert forward[3] is None
    assert info['direct_pairs'] > 0 and info['far_cell_pairs'] > 0
    assert info['represented_pairs'] == 14
    expected, expected_x = 0., np.zeros_like(centers)
    expected_d, expected_g = [np.zeros_like(d) for d in dipoles], [np.zeros_like(g) for g in gaps]
    for i in range(len(centers)):
        for j in range(i):
            if (j, i) in exclusions:
                continue
            e, di, gi, dj, gj, x = exact_pair(dipoles[i], gaps[i], dipoles[j], gaps[j], centers[i]-centers[j])
            expected += e; expected_d[i] += di; expected_g[i] += gi
            expected_d[j] += dj; expected_g[j] += gj
            expected_x[i] += x; expected_x[j] -= x
    assert abs(energy-expected) <= info['analytic_energy_absolute_bound']+1e-15
    assert abs(energy-expected) < 1e-11
    assert np.max(abs(xb-expected_x)) < 1e-10
    for d, g, ed, eg in zip(db, gb, expected_d, expected_g):
        assert np.max(abs(d-ed)) < 1e-10
        assert np.max(abs(g-eg)) < 1e-11
    dx = rng.normal(size=centers.shape)
    dd, dg = [rng.normal(size=d.shape) for d in dipoles], [rng.normal(size=g.shape) for g in gaps]
    step = 1e-6
    values = []
    for sign in (1, -1):
        values.append(driver.evaluate(centers+sign*step*dx,
            [d+sign*step*h for d, h in zip(dipoles, dd)],
            [g+sign*step*h for g, h in zip(gaps, dg)], exclusions=exclusions, tree=tree)[0])
    analytic = np.sum(xb*dx)+sum(np.sum(a*b) for a, b in zip(db, dd))+sum(np.sum(a*b) for a, b in zip(gb, dg))
    assert abs((values[0]-values[1])/(2*step)-analytic) < 1e-9
    from electrolites.dlpno import reverse as ad
    vx = ad.Variable(centers)
    vd, vg = [ad.Variable(d) for d in dipoles], [ad.Variable(g) for g in gaps]
    objective = driver.energy_ad(vx, vd, vg, exclusions=exclusions)
    bars = ad.gradients(objective, [vx, *vd, *vg])
    for got, want in zip(bars, [xb, *db, *gb]):
        assert np.max(abs(got-want)) < 1e-14


@pytest.mark.parametrize('local_screening_options', [None, {}])
def test_hierarchical_tail_in_full_t1_nuclear_adjoint(record_property, local_screening_options):
    from pyscf import gto, scf
    from scipy.optimize import linear_sum_assignment
    from electrolites.dlpno.local import LocalDLPNO
    from electrolites.dlpno.reference_cpu import orbitals
    from electrolites.dlpno.setup import count_frozen_core
    from benchmarks.dlpno_gradient_check import tracked_driver, topology
    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57; '
                     'O 0 0 12; H .02 .76 12.59; H .09 -.75 12.57',
                basis='6-31g', verbose=0)
    options = dict(triples='t1', pno='loose', verbose=0,
                   E_CONVERGENCE=1e-12, R_CONVERGENCE=1e-11, MAXITER=150, T_CUT_ITER=0.)
    tail_options = dict(order=7, theta=.3, gap_bounds=(.01, 100.))
    def mean_field(m):
        mf = scf.RHF(m).density_fit()
        mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
        mf.kernel()
        assert mf.converged
        return mf
    mf = mean_field(mol)
    parent = LocalDLPNO(mf, **options)
    parent.kernel()
    exact = parent.nuc_grad_method().kernel()
    cls = tracked_driver(LocalDLPNO)
    fast = cls(mf, c_lmo=parent._orbital_reference.C_lmo,
               dipole_tail_options=tail_options, local_screening_options=local_screening_options, **options)
    fast.kernel()
    assert fast.dip_e is None and fast.dip_b is None
    expected_mask = ((parent.doi_ij > parent.opt['T_CUT_DO_IJ'])
                     | (abs(parent.dip_b) > parent.opt['T_CUT_PRE']) | np.eye(parent.n, dtype=bool))
    assert np.array_equal(fast._initial_pair_mask.to_dense(), expected_mask)
    info = fast.dipole_tail_diagnostics
    assert info['far_cell_pairs'] > 0
    assert abs(parent.e['dipole']) > 1e-10
    assert abs(fast.e['dipole']-parent.e['dipole']) <= info['analytic_energy_absolute_bound']+1e-15
    gradient = fast.nuc_grad_method().kernel()
    assert np.max(abs(gradient-exact)) < 2e-9
    record_property('exact_dipole_energy', parent.e['dipole'])
    record_property('tree_dipole_energy_error', fast.e['dipole']-parent.e['dipole'])
    record_property('tree_energy_absolute_bound', info['analytic_energy_absolute_bound'])
    record_property('tree_force_max_error', float(np.max(abs(gradient-exact))))
    record_property('far_cell_pairs', info['far_cell_pairs'])
    step, values = 2e-4, []
    for sign in (1, -1):
        xyz = mol.atom_coords().copy(); xyz[0, 2] += sign*step
        moved = mol.copy().set_geom_(xyz, unit='Bohr')
        shifted_mf = mean_field(moved)
        orb = orbitals(moved, shifted_mf.get_ovlp(), shifted_mf.get_fock(), shifted_mf.mo_coeff,
                        moved.nelectron//2, count_frozen_core(moved), ensure_minimum=True)
        overlap = fast._orbital_reference.C_lmo.T@gto.intor_cross('int1e_ovlp', mol, moved)@orb.C_lmo
        rows, columns = linear_sum_assignment(-abs(overlap))
        columns = columns[np.argsort(rows)]
        c_lmo = orb.C_lmo[:, columns]*np.sign(overlap[np.arange(len(columns)), columns])
        displaced = cls(shifted_mf, c_lmo=c_lmo, dipole_tail_options=tail_options,
                        local_screening_options=local_screening_options, **options)
        displaced.kernel()
        assert topology(displaced, True) == topology(fast, True)
        assert displaced._dipole_tail_tree.signature == fast._dipole_tail_tree.signature
        values.append(displaced.e['tot'])
    error = (values[0]-values[1])/(2*step)-gradient[0, 2]
    record_property('nuclear_fd_step', step)
    record_property('nuclear_fd_error', float(error))
    assert abs(error) < 1e-7


def test_hierarchical_tail_gpu_parent_and_cupy_adjoint(record_property):
    import pytest
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('A CUDA device is required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('A CUDA device is required')
    from pyscf import gto, scf
    from electrolites.dlpno.local import LocalDLPNO
    from electrolites.dlpno.local_gpu import LocalGPUDLPNO
    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57; '
                     'O 0 0 12; H .02 .76 12.59; H .09 -.75 12.57',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-13)
    options = dict(triples='t0', pno='loose', verbose=0,
                   E_CONVERGENCE=1e-12, R_CONVERGENCE=1e-11, MAXITER=150, T_CUT_ITER=0.,
                   dipole_tail_options=dict(order=7, theta=.3, gap_bounds=(.01, 100.)))
    options['local_screening_options'] = {}
    cpu = LocalDLPNO(mf, **options); cpu.kernel()
    gpu = LocalGPUDLPNO(mf, c_lmo=cpu._orbital_reference.C_lmo, **options); gpu.kernel()
    assert cpu.dipole_tail_diagnostics['far_cell_pairs'] > 0
    assert abs(gpu.e['corr']-cpu.e['corr']) < 1e-9
    cpu_gradient = cpu.nuc_grad_method().kernel()
    gpu_gradient = gpu.nuc_grad_method(backend='cupy').kernel()
    record_property('cpu_gpu_energy_error', gpu.e['corr']-cpu.e['corr'])
    record_property('cpu_gpu_force_max_error', float(np.max(abs(cpu_gradient-gpu_gradient))))
    assert np.max(abs(cpu_gradient-gpu_gradient)) < 1e-8
