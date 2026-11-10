"""Adjoint identities and complete energy derivatives (no Jacobian assembly)."""
import numpy as np
import pytest

from electrolites.dlpno import reverse as rev
from electrolites.dlpno import dual


def require_cuda():
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('A CUDA device is required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('A CUDA device is required')


@pytest.mark.parametrize('shapes', [((3, 4), (4, 5)), ((4,), (4, 5)),
                                   ((3, 4), (4,)), ((4,), (4,)),
                                   ((2, 3, 4), (4,)), ((4,), (2, 4, 3))])
def test_reverse_matmul_duality(shapes):
    rng = np.random.default_rng(45)
    a, b = [rng.normal(size=s) for s in shapes]
    da, db = [rng.normal(size=s) for s in shapes]
    x, y = rev.Variable(a), rev.Variable(b)
    value = x@y
    seed = rng.normal(size=value.shape)
    ga, gb = rev.Linearization([value], [x, y])([seed])
    tangent = (dual.Dual(a, da)@dual.Dual(b, db)).d
    assert abs(np.sum(ga*da)+np.sum(gb*db)-np.sum(seed*tangent)) < 1e-11


def test_checkpoint_multiple_outputs_and_late_primitive():
    from electrolites.dlpno import reverse as ad
    rng = np.random.default_rng(602)
    a = ad.Variable(rng.normal(size=(5, 4)))
    b = ad.Variable(rng.normal(size=(4, 3)))
    cache = {}

    def forward():
        if 'primitive' not in cache:
            cache['primitive'] = ad.Variable(rng.normal(size=(5, 3)))
        x = a@b+cache['primitive']
        y = x*x+.3*x
        return y, np.sum(y*y)

    (y, energy), info = ad.checkpoint(forward)
    target = np.sum(y)+.23*energy
    actual = ad.gradients(target, [a, b, cache['primitive']])
    x = a.v@b.v+cache['primitive'].v
    y0 = x*x+.3*x
    derivative = (1+.46*y0)*(2*x+.3)
    for got, expected in zip(actual, [derivative@b.v.T, a.v.T@derivative, derivative]):
        assert np.max(abs(got-expected)) < 1e-12
    restricted = ad.Linearization([target], [a])([np.array(1.)])[0]
    assert np.max(abs(restricted-actual[0])) < 1e-12
    assert info['boundary_records'] == 3
    assert info['inner_value_bytes'] > info['output_bytes']


def test_partial_linearization_stops_at_checkpoint_boundary():
    x = rev.Variable(np.array([.2, -.3]))
    z = x*x
    target = np.sum(x*z)
    partial = rev.Linearization([target], [x], boundary=[x, z])([np.array(1.)])[0]
    assert np.max(abs(partial-z.v)) < 1e-14
    complete = rev.gradients(target, [x])[0]
    assert np.max(abs(complete-3*x.v*x.v)) < 1e-14


@pytest.mark.parametrize('release_inputs', [False, True])
def test_consuming_density_sweep_preserves_shared_checkpoint_inputs(release_inputs):
    rng = np.random.default_rng(620)
    x = rev.Variable(rng.normal(size=(4, 3)))
    unused = rev.Variable(np.zeros((2, 7)))
    original = x.v.copy()
    shared = x@x.T
    (left,), _ = rev.checkpoint(lambda: (shared@shared+.2*shared,))
    (right,), _ = rev.checkpoint(lambda: (shared@x,))
    target = np.sum(left*left)+np.sum(right*right)
    graph = rev.Linearization([target], [x, unused])
    seed = [np.array(1.)]
    expected = graph(seed)[0]
    bars = graph(seed, release_graph=True, release_inputs=release_inputs)
    actual = bars[0]
    assert np.max(abs(actual-expected)) < 1e-12
    assert np.array_equal(bars[1], np.zeros((2, 7)))
    if release_inputs:
        assert x.v is unused.v is None
        assert graph.last_statistics['released_input_records'] == 2
        assert graph.last_statistics['released_input_bytes'] == original.nbytes+2*7*8
    else:
        assert np.array_equal(x.v, original)
    assert graph.last_statistics['released_records'] > 0
    assert graph.last_statistics['released_value_bytes'] > 0
    with pytest.raises(RuntimeError, match='consumed'):
        graph(seed)
    with pytest.raises(RuntimeError, match='consumed'):
        rev.Linearization([target], [x])


@pytest.mark.parametrize('backend', ['numpy', 'cupy', 'hybrid'])
def test_explicit_triples_transpose_and_local_replay(backend):
    from itertools import permutations
    from types import MethodType, SimpleNamespace
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.adjoint import (
        Amplitudes, TriplesTranspose, _checkpointed_triplet_residual, host,
    )
    from electrolites.dlpno.ref_rhf import RefDLPNO
    from electrolites.dlpno.reference_cpu import rebind
    xp = np
    if backend == 'hybrid':
        require_cuda()
    if backend == 'cupy':
        require_cuda()
        import cupy as xp
    rng = np.random.default_rng(921)
    trip = [(0, 0, 1), (0, 1, 1), (0, 1, 2), (0, 2, 2)]
    ranks = [2, 3, 4, 2]
    matrix = rng.normal(size=(8, 8))
    metric = xp.asarray(matrix@matrix.T/8)
    fock = xp.asarray([[-1., .03, -.04], [.03, -.8, .02], [-.04, .02, -.6]])
    state = SimpleNamespace(xp=xp, n=3, numeric=ad.Numpy(xp), trip=trip,
                            trip_idx={p: t for t, key in enumerate(trip) for p in permutations(key)},
                            opt={'F_CUT_T': 1e-9}, diagnostics={},
                            orb=SimpleNamespace(F_lmo=fock, S_pao=metric),
                            t_paos=[rng.choice(8, size=n, replace=False) for n in [5, 6, 7, 4]],
                            e_tno=[xp.asarray(np.linspace(.2, 1., n)) for n in ranks],
                            W3=[xp.asarray(rng.normal(size=(n, n, n))) for n in ranks])
    state.X_tno = [xp.asarray(.2*rng.normal(size=(len(p), n))) for p, n in zip(state.t_paos, ranks)]
    independent = ad.Variable(xp.asarray(rng.normal(size=sum(n**3 for n in ranks))))
    offsets = np.cumsum([0]+[n**3 for n in ranks])
    state.T3 = [independent[a:b].reshape((n, n, n)) for a, b, n in zip(offsets[:-1], offsets[1:], ranks)]
    state._triples_couplings = MethodType(RefDLPNO._triples_couplings, state)
    state._matmul_3d = MethodType(rebind(RefDLPNO._matmul_3d,
                                       lib=SimpleNamespace(einsum=ad.einsum)), state)
    evaluate = rebind(RefDLPNO._triplet_residual, np=state.numeric)
    connections = state._triples_couplings()
    residual = ad.concatenate([evaluate(state, t, rows).ravel() for t, rows in enumerate(connections)])
    seeds = [xp.asarray(rng.normal(size=independent.size))]
    expected = ad.Linearization([residual], [independent])(seeds)[0]
    diagonal = Amplitudes(state, triples=True).denominator()
    direct = TriplesTranspose(state, diagonal, backend='cupy' if backend == 'hybrid' else None)
    assert np.max(abs(host(direct(seeds)[0])-host(expected))) < 1e-11
    assert direct.statistics['records'] == 0
    if backend != 'numpy':
        split = TriplesTranspose(state, diagonal, backend='cupy', gpu_batch_bytes=1)
        assert np.max(abs(host(split(seeds)[0])-host(expected))) < 1e-11
        assert split.statistics['batches'] == split.statistics['couplings']
    replayed = ad.concatenate([_checkpointed_triplet_residual(state, t, rows).ravel()
                               for t, rows in enumerate(connections)])
    assert np.max(abs(host(replayed)-host(residual))) < 1e-12
    state.T3 = []  # Reinstalling converged amplitudes must not change the captured residual.
    actual = ad.Linearization([replayed], [independent])(seeds)[0]
    assert np.max(abs(host(actual)-host(expected))) < 1e-11
    assert state.diagnostics['triples_residual_checkpoints']['count'] == len(trip)


def test_reverse_composite_and_mutation_versions():
    rng = np.random.default_rng(12)
    a = rng.normal(size=(4, 4))
    a = a@a.T + np.eye(4)
    da = rng.normal(size=(4, 4))
    da = da+da.T
    b, db = rng.normal(size=(2, 4, 3)), rng.normal(size=(2, 4, 3))

    def function(ns, matrix, tensor):
        w, u = ns.eigh(matrix)
        s = ns.sym_power(matrix, -.5)
        q = ns.einsum('ab,qbc->qac', u, tensor)
        z = q.copy()
        z[0, :, 1] += w
        z[1] = z[0]*2 + z[1]
        solved = ns.solve(matrix, z[0])
        return ns.einsum('ab,qbc,qac->', s, z, z) + np.sum(solved**2)

    x, y = rev.Variable(a), rev.Variable(b)
    energy = function(rev, x, y)
    ga, gb = rev.gradients(energy, [x, y])
    tangent = function(dual, dual.Dual(a, da), dual.Dual(b, db)).d
    assert abs(np.sum(ga*da)+np.sum(gb*db)-tangent) < 2e-10
    # Only amplitude-dependent paths are visited by a lambda matvec.
    restricted = rev.Linearization([energy], [y])
    assert restricted.statistics['records'] < rev.Linearization([energy]).statistics['records']
    assert np.max(abs(restricted([np.array(1.)])[0]-gb)) < 1e-11
    assert restricted.last_statistics['skipped_parent_pullbacks'] > 0


def test_restricted_einsum_computes_only_requested_operand(monkeypatch):
    a = rev.Variable(np.arange(12, dtype=float).reshape(3, 4))
    b = rev.Variable(np.arange(8, dtype=float).reshape(4, 2))
    output = rev.einsum('ab,bc->ac', a, b)
    action = rev.Linearization([output], [b])
    original = np.einsum
    calls = []

    def observe(spec, *args, **kwargs):
        calls.append(spec)
        return original(spec, *args, **kwargs)

    monkeypatch.setattr(np, 'einsum', observe)
    seed = np.arange(6, dtype=float).reshape(3, 2)
    actual = action([seed])[0]
    assert np.array_equal(actual, a.v.T@seed)
    assert calls == ['ac,ab->bc']
    assert action.last_statistics['skipped_parent_pullbacks'] == 1


def test_degenerate_reverse_metric_and_projector():
    a = np.diag([1., 1., 3., 5.])
    da = np.array([[.2, .4, -.3, .1], [.4, -.1, .7, .2],
                   [-.3, .7, .1, -.2], [.1, .2, -.2, .3]])
    rng = np.random.default_rng(4)
    weight = rng.normal(size=a.shape)
    x = rev.Variable(a)
    _, u = rev.eigh(x)
    energy = np.sum(weight*(u[:, :2]@u[:, :2].T + rev.sym_power(x, -.5)))
    (bar,) = rev.gradients(energy, [x])
    _, v = dual.eigh(dual.Dual(a, da))
    direct = np.sum(weight*(v[:, :2]@v[:, :2].T + dual.sym_power(dual.Dual(a, da), -.5)))
    assert abs(np.sum(bar*da)-direct.d) < 1e-10
    with pytest.raises(ValueError, match='degenerate'):
        _ = u[:, :1]


@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
@pytest.mark.parametrize('scale', [1., 1e-8, 1e-14])
def test_spectral_projector_response_is_scale_invariant(backend, scale):
    if backend == 'cupy':
        require_cuda()
        import cupy as xp
    else:
        xp = np
    rng = np.random.default_rng(617)
    frame, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    a = (frame*np.array([1., 2., 3., 5.]))@frame.T
    direction = rng.normal(size=a.shape)
    direction = (direction+direction.T)*.5
    weight = rng.normal(size=a.shape)

    def energy(matrix):
        _, vectors = np.linalg.eigh(matrix)
        return np.sum(weight*(vectors[:, :2]@vectors[:, :2].T))

    step = 1e-5
    expected = (energy(a+step*direction)-energy(a-step*direction))/(2*step)
    x = rev.Variable(xp.asarray(scale*a))
    _, u = rev.eigh(x)
    scalar = np.sum(xp.asarray(weight)*(u[:, :2]@u[:, :2].T))
    (bar,) = rev.gradients(scalar, [x])
    actual = float(xp.sum(bar*xp.asarray(scale*direction)))
    _, v = dual.eigh(dual.Dual(xp.asarray(scale*a), xp.asarray(scale*direction)))
    tangent = np.sum(xp.asarray(weight)*(v[:, :2]@v[:, :2].T)).d
    assert abs(actual-expected) < 2e-9
    assert abs(float(tangent)-expected) < 2e-9

    # Exact repeated eigenvalues remain a genuine subspace ambiguity at
    # every scale, including the identically zero pair-density matrix.
    for values in ([1., 1., 3., 5.], [0., 0., 0., 0.]):
        degenerate = xp.diag(xp.asarray(values)*scale)
        _, u = rev.eigh(rev.Variable(degenerate))
        _, v = dual.eigh(dual.Dual(degenerate, xp.asarray(scale*direction)))
        for vectors in (u, v):
            with pytest.raises(ValueError, match='degenerate'):
                _ = vectors[:, :1]


def test_local_slices_have_linear_reverse_accumulation_cost():
    costs = []
    for size in (64, 128, 256):
        values = np.linspace(-1., 1., size)
        x = rev.Variable(values)
        result = 0.
        for start in range(0, size, 2):
            result += np.sum(x[start:start+2]**2)
        # Repeated indices require scatter-add, not indexed assignment.
        result += np.sum(x[[0, 0, size-1]]**2)
        linear = rev.Linearization([result], [x])
        (gradient,) = linear([np.asarray(1.)])
        expected = 2*values
        np.add.at(expected, [0, 0, size-1], 2*values[[0, 0, size-1]])
        assert np.max(abs(gradient-expected)) < 1e-12
        assert linear.last_statistics['indexed_elements'] == size+3
        costs.append(linear.last_statistics['contribution_elements'])
    assert costs[1] < 2.05*costs[0]
    assert costs[2] < 2.05*costs[1]


def test_df_response_force_retains_small_density_components():
    from pyscf import gto, scf
    from electrolites.dlpno.orbital_adjoint import _jk_quadratic_force
    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).density_fit().run(conv_tol=1e-12)
    rng = np.random.default_rng(56)
    vectors, _ = np.linalg.qr(rng.normal(size=(mol.nao, mol.nao)))
    eigenvalues = np.linspace(-.7, 1., mol.nao)
    eigenvalues[2] = 1e-4
    density = (vectors*eigenvalues)@vectors.T
    grad = mf.nuc_grad_method()
    scale = 1e-6
    large = _jk_quadratic_force(grad, density)
    small = _jk_quadratic_force(grad, density*scale)/scale**2
    assert np.max(abs(large-small)) < 1e-10


@pytest.mark.parametrize('triples', ['t0', 't1'])
def test_local_gpu_energy_and_adjoint_match_local_cpu(triples):
    require_cuda()
    from pyscf import gto, scf
    from electrolites.dlpno.local import LocalDLPNO
    from electrolites.dlpno.local_gpu import LocalGPUDLPNO
    from benchmarks.dlpno_gradient_check import STRESS
    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57', basis='6-31g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
    mf.kernel()
    options = dict(triples=triples, verbose=0, **STRESS,
                   E_CONVERGENCE=1e-11, R_CONVERGENCE=1e-10, MAXITER=150, T_CUT_ITER=0.)
    cpu = LocalDLPNO(mf, **options)
    cpu.kernel()
    gpu = LocalGPUDLPNO(mf, c_lmo=cpu.orb.C_lmo, **options)
    gpu.kernel()
    assert abs(gpu.e['corr']-cpu.e['corr']) < 1e-8
    force_cpu = cpu.nuc_grad_method().kernel()
    gradient = gpu.nuc_grad_method(backend='cupy')
    assert np.max(abs(gradient.kernel()-force_cpu)) < 1e-8
    assert gpu.dfi.nbytes['ia'] == gpu.dfi.nbytes['ij'] == 0
    assert gradient.diagnostics['orbitals']['cphf']['solves'] == 1


@pytest.mark.parametrize('level,triples,df_scf', [('mp2', None, False), ('ccsd', None, False),
                                                ('ccsd_t', 't0', False), ('ccsd_t', 't1', False),
                                                ('ccsd', None, True), ('ccsd_t', 't1', True)])
@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
def test_implicit_adjoint_against_nuclear_tangent(level, triples, df_scf, backend):
    from pyscf import gto, scf
    from electrolites.dlpno.reference_cpu import CPUDLPNO
    from electrolites.dlpno.adjoint import AdjointDLPNO, host
    from electrolites.dlpno.orbital_response import RHFResponse
    from electrolites.dlpno.response import ResponseDLPNO
    from electrolites.dlpno.orbital_adjoint import (
        orbital_densities, contract_orbital_densities, contract_dense_df_densities,
    )
    if backend == 'cupy':
        require_cuda()

    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57', basis='6-31g', verbose=0)
    mf = scf.RHF(mol)
    if df_scf:
        mf = mf.density_fit()
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
    mf.kernel()
    options = dict(T_CUT_PNO=1e-4, T_CUT_PNO_MP2=1e-5, T_CUT_TRACE=0., T_CUT_ENERGY=0.,
                   T_CUT_TRACE_MP2=0., T_CUT_ENERGY_MP2=0., MIN_PNOS=1, MIN_TNOS=1,
                   T_CUT_TNO=1e-5, T_CUT_TNO_PRE=1e-4, E_CONVERGENCE=1e-11,
                   R_CONVERGENCE=1e-10, MAXITER=150, T_CUT_ITER=0.)
    base = CPUDLPNO(mf, triples=triples, verbose=0, **options)
    base.kernel()
    short_restart = backend == 'numpy' and triples == 't1' and df_scf
    adjoint = AdjointDLPNO(base, level=level, backend=backend,
                          krylov_memory_mib=1e-6 if short_restart else 8192,
                          checkpoint_pnos=(backend == 'cupy'))
    energy = adjoint.kernel()
    target = 'lmp2' if level == 'mp2' else 'corr'
    assert abs(float(host(energy))-base.e[target]) < 1e-9
    bars = adjoint.densities(release_graph=(backend == 'cupy'), release_inputs=(backend == 'cupy'))
    if short_restart:
        assert all(adjoint.diagnostics[name]['krylov_restart'] == 2
                   for name in ('mp2', 'ccsd', 'triples'))
    orbital = RHFResponse(base)
    direction = next(d for d in orbital.directions() if (d.atom, d.xyz) == (1, 2))
    source = dict(s=direction.ds, f=direction.df, co=direction.dc, cl=direction.dl,
                  b=direction.db, m=direction.dm, r=direction.dr)
    gradient = sum(np.sum(bars[k]*source[k]) for k in bars)
    direct = ResponseDLPNO(base, orbital, direction, level).kernel()
    assert abs(gradient-float(direct.d)) < 1e-8
    relaxed, orbital_info = orbital_densities(base, bars)
    assert orbital_info['cphf']['solves'] == 1
    assert orbital_info['boys']['solves'] == 1
    explicit = (np.sum(relaxed['s']*direction.ds)
                + np.sum(relaxed['f']*orbital.f1[1][2])
                + np.sum(relaxed['r']*direction.dr)
                + np.sum(bars['b']*direction.db)+np.sum(bars['m']*direction.dm))
    assert abs(explicit-gradient) < 1e-8
    integral_gradient, stats = contract_dense_df_densities(base, bars)
    nuclear = (integral_gradient+contract_orbital_densities(base, relaxed)
               + orbital.hf_gradient)
    assert abs(nuclear[1, 2]-gradient-orbital.hf_gradient[1, 2]) < 1e-8
    assert np.max(abs(nuclear.sum(axis=0))) < 1e-8
    assert np.max(abs(np.cross(mol.atom_coords(), nuclear).sum(axis=0))) < 1e-8
    assert stats['derivative_aux_blocks'] == mol.natm
    for name in ('mp2', 'ccsd', 'triples'):
        if name in adjoint.diagnostics:
            assert adjoint.diagnostics[name]['lambda_solves'] == 1
            assert adjoint.diagnostics[name]['adjoint_residual'] < 1e-8
