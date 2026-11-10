"""Local triples sources against independent equations and unfrozen forces."""
import json
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import _names, host
from electrolites.dlpno.ref_rhf import RefDLPNO
from electrolites.dlpno.reference_cpu import rebind
from electrolites.dlpno.triples_adjoint import triplet_source


def array_module(backend):
    if backend == 'numpy':
        return np
    cp = pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('A CUDA device is required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('A CUDA device is required')
    return cp


def state_from(values, triplet, empty):
    ranks = [2, 3, 4, 3, 2, 4, 4, 3, 2]
    state = SimpleNamespace(pl=SimpleNamespace(ij=np.arange(9).reshape(3, 3)),
        trip=[triplet], t_ribfs=[np.array([6, 1, 4, 0, 3])],
        t_paos=[np.array([5, 1, 3, 0])],
        t_lmos=[np.array([], dtype=int) if empty else np.array([2, 0, 1])],
        pao=[np.roll(np.arange(6), p)[:5] for p in range(9)],
        X=[values['x'+str(p)] for p in range(9)],
        T=[values['t'+str(p)] for p in range(9)],
        T1=[values['s'+str(i)] for i in range(3)],
        X_tno=[values['xt']], e_tno=[values['eps']],
        qij=values['qo'], qia=values['qv'], qab=values['qvv'],
        J2c=(values['metric']+values['metric'].T)*.5,
        orb=SimpleNamespace(S_pao=(values['overlap']+values['overlap'].T)*.5,
                            F_lmo=(values['fock']+values['fock'].T)*.5))
    state._S_tp = MethodType(rebind(RefDLPNO._S_tp, **_names), state)
    state._e_ijk = rebind(RefDLPNO._e_ijk, **_names)
    return state


@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
@pytest.mark.parametrize('triplet,empty', [((0, 1, 2), False), ((0, 0, 2), False),
                                         ((1, 1, 1), False), ((0, 1, 2), True)])
def test_local_source_values_pullbacks_and_direction(backend, triplet, empty):
    xp = array_module(backend)
    rng = np.random.default_rng(731)
    ranks = [2, 3, 4, 3, 2, 4, 4, 3, 2]
    raw = dict(qo=.2*rng.normal(size=(7, 3, 3)), qv=.2*rng.normal(size=(7, 3, 6)),
               qvv=.2*rng.normal(size=(7, 6, 6)), xt=.2*rng.normal(size=(4, 3)),
               eps=np.array([.2, .5, .9]), fock=-np.diag([1., .8, .6]))
    # Repeated metric eigenvalues exercise the differentiable matrix function.
    frame, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    raw['metric'] = (frame*np.array([.4, .4, .4, 1., 1., 2., 3.]))@frame.T
    raw['overlap'] = np.eye(6)+.02*rng.normal(size=(6, 6))
    for p, n in enumerate(ranks):
        raw['x'+str(p)] = .3*rng.normal(size=(5, n))
        raw['t'+str(p)] = .2*rng.normal(size=(n, n))
    for i in range(3):
        raw['s'+str(i)] = .1*rng.normal(size=ranks[3*i+i])
    variables = {name: ad.Variable(xp.asarray(value)) for name, value in raw.items()}
    state = state_from(variables, triplet, empty)
    oracle = rebind(RefDLPNO._triplet_t0, **_names)
    expected = oracle(state, 0)
    actual, stats = ad.checkpoint(lambda: triplet_source(state, 0))
    seeds = [xp.asarray(rng.normal(size=value.shape)) for value in expected]
    original_graph = ad.Linearization(expected, list(variables.values()))
    previous = original_graph(seeds)
    current = ad.Linearization(actual, list(variables.values()))(seeds)
    value_error = max(float(np.max(abs(host(a)-host(b)))) for a, b in zip(actual, expected))
    derivative_error = max(float(np.max(abs(host(a)-host(b)))) for a, b in zip(current, previous))
    assert value_error < 1e-11
    assert derivative_error < 2e-10
    # Central differences use the original unoptimized equations, varying
    # the metric, integrals, subspaces, orbital energies and both amplitudes.
    directions = {name: .1*rng.normal(size=value.shape) for name, value in raw.items()}
    h = 2e-5
    targets = []
    for sign in (1, -1):
        displaced = {name: xp.asarray(value+sign*h*directions[name]) for name, value in raw.items()}
        outputs = oracle(state_from(displaced, triplet, empty), 0)
        targets.append(sum(float(np.sum(host(a)*host(b))) for a, b in zip(outputs, seeds)))
    numerical = (targets[0]-targets[1])/(2*h)
    analytic = sum(float(np.sum(host(bar)*directions[name])) for name, bar in zip(variables, current))
    assert abs(analytic-numerical) < 2e-8
    print('TRIPLES_SOURCE_REPORT '+json.dumps(dict(backend=backend,triplet=triplet,empty=empty,
          max_value_error=value_error,max_pullback_error=derivative_error,
          directional_error=abs(analytic-numerical),checkpoint=stats,
          original_graph=original_graph.statistics),sort_keys=True))


@pytest.mark.parametrize('triples', ['t0', 't1'])
def test_complete_gradient_and_unfrozen_nuclear_difference(triples):
    from pyscf import gto, scf
    from benchmarks.dlpno_gradient_check import STRESS, topology, tracked_driver
    from electrolites.dlpno.local import LocalDLPNO
    array_module('cupy')
    mol = gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57',basis='6-31g',verbose=0)
    reference = scf.RHF(mol).density_fit().run(conv_tol=1e-13,conv_tol_grad=1e-10)
    assert reference.converged
    options = dict(triples=triples,verbose=0,**STRESS,E_CONVERGENCE=1e-11,
                   R_CONVERGENCE=1e-10,MAXITER=150,T_CUT_ITER=0.)
    driver = tracked_driver(LocalDLPNO)
    base = driver(reference,**options);base.kernel()
    # Disabling triples checkpoints selects the unchanged RefDLPNO equations.
    previous = base.nuc_grad_method(checkpoint_triples=False,checkpoint_ccsd=True).kernel()
    gradient = base.nuc_grad_method(checkpoint_triples=True,checkpoint_ccsd=True,
                                   ccsd_adjoint_backend='cupy',triples_backend='cupy')
    actual = gradient.kernel()
    error = float(np.max(abs(actual-previous)))
    assert error < 2e-8
    center = topology(base,effective=True)
    differences = []
    for step in (1e-4,5e-5):
        energies = []
        for sign in (1,-1):
            coordinates = mol.atom_coords();coordinates[1,2] += sign*step
            displaced_mol = mol.copy().set_geom_(coordinates,unit='Bohr')
            mf = scf.RHF(displaced_mol).density_fit(auxbasis=reference.with_df.auxbasis)
            mf.conv_tol,mf.conv_tol_grad = 1e-13,1e-10
            mf.kernel(dm0=reference.make_rdm1());assert mf.converged
            displaced = driver(mf,**options);displaced.kernel()
            assert topology(displaced,effective=True)==center
            energies.append(mf.e_tot+displaced.e['corr'])
        numerical = (energies[0]-energies[1])/(2*step)
        differences.append(dict(step_bohr=step,error=actual[1,2]-numerical,
                               topology_stable=True,scf_boys_pno_recomputed=True))
    assert max(abs(row['error']) for row in differences)<2e-7
    print('TRIPLES_GRADIENT_REPORT '+json.dumps(dict(triples=triples,
          max_gradient_error=error,independent_nuclear_fd=differences),sort_keys=True))
