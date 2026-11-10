"""Shared exact PNO overlaps: source derivatives and full physical gradients."""
import json
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
def test_shared_overlap_retains_metric_and_coefficients_through_consuming_replay(backend):
    from electrolites.dlpno import reverse as ad
    from electrolites.dlpno.adjoint import _checkpointed_ccsd_overlap, host
    xp = np
    if backend == 'cupy':
        xp = pytest.importorskip('cupy')
        try:
            if not xp.cuda.runtime.getDeviceCount():
                pytest.skip('CUDA required')
        except xp.cuda.runtime.CUDARuntimeError:
            pytest.skip('CUDA required')
    rng = np.random.default_rng(114)
    pao = [np.array([0, 2, 4, 5]), np.array([1, 2, 3]), np.array([0, 1, 5])]
    raw = [rng.normal(size=(6, 6))] + [rng.normal(size=(len(rows), n)) for rows, n in zip(pao, (2, 3, 2))]
    # A nonsymmetric metric prevents silently imposing S(p,q)=S(q,p).T.
    leaves = [ad.Variable(xp.asarray(x)) for x in raw]
    state = SimpleNamespace(orb=SimpleNamespace(S_pao=leaves[0]), X=leaves[1:], pao=pao,
                            _S={}, diagnostics={})
    a = _checkpointed_ccsd_overlap(state, 0, 1)
    assert _checkpointed_ccsd_overlap(state, 0, 1) is a
    b = _checkpointed_ccsd_overlap(state, 1, 0)
    (first,), _ = ad.checkpoint(lambda: (a @ b,))
    c = _checkpointed_ccsd_overlap(state, 1, 2)
    (second,), _ = ad.checkpoint(lambda: (a @ c,))
    target = np.sum(first*first)+.31*np.sum(second*second)
    graph = ad.Linearization([target], leaves)
    expected = graph([xp.asarray(1.)])
    repeat = graph([xp.asarray(.37)])
    assert max(np.max(abs(host(x)-.37*host(y)), initial=0.) for x,y in zip(repeat,expected)) < 2e-10
    actual = graph([xp.asarray(1.)], release_graph=True, release_inputs=True)
    error = max(float(np.max(abs(host(x)-host(y)), initial=0.)) for x,y in zip(actual,expected))
    assert error < 2e-10
    direction = [rng.normal(size=x.shape) for x in raw]
    norm = np.sqrt(sum(np.sum(x*x) for x in direction));direction = [x/norm for x in direction]
    def oracle(values):
        metric,x,y,z = values
        aa=x.T@metric[np.ix_(pao[0],pao[1])]@y
        bb=y.T@metric[np.ix_(pao[1],pao[0])]@x
        cc=y.T@metric[np.ix_(pao[1],pao[2])]@z
        return np.sum((aa@bb)**2)+.31*np.sum((aa@cc)**2)
    h=1e-5
    finite=(oracle([x+h*d for x,d in zip(raw,direction)])-oracle([x-h*d for x,d in zip(raw,direction)]))/(2*h)
    analytic=sum(np.sum(host(x)*d) for x,d in zip(actual,direction))
    fd_error=float(abs(finite-analytic));assert fd_error < 2e-6
    assert all(x.v is None for x in leaves)
    info=state.diagnostics['ccsd_shared_overlaps']
    assert info['checkpoints']==3 and info['cache_hits']==1
    print('SHARED_OVERLAP_REPORT '+json.dumps(dict(backend=backend,reusable_and_consuming_error=error,source_directional_fd_error=fd_error,statistics=info)))


@pytest.mark.parametrize('triples', [None, 't0', 't1'])
def test_shared_overlap_complete_gradient_matches_uncheckpointed_equations(triples, monkeypatch):
    cp=pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount():pytest.skip('CUDA required')
    except cp.cuda.runtime.CUDARuntimeError:pytest.skip('CUDA required')
    from pyscf import gto,scf
    from benchmarks.dlpno_gradient_check import STRESS
    from electrolites.dlpno.local import LocalDLPNO
    monkeypatch.setenv('DLPNO_NGPU','1')
    mol=gto.M(atom='O 0 0 0; H 0 .76 .59; H .07 -.75 .57',basis='6-31g',verbose=0)
    mf=scf.RHF(mol).density_fit().run(conv_tol=1e-13,conv_tol_grad=1e-10)
    assert mf.converged
    base=LocalDLPNO(mf,triples=triples,verbose=0,**STRESS,E_CONVERGENCE=1e-11,R_CONVERGENCE=1e-10,MAXITER=150,T_CUT_ITER=0.)
    base.kernel()
    expected=base.nuc_grad_method(checkpoint_ccsd=False).kernel()
    gradient=base.nuc_grad_method(checkpoint_ccsd=True,ccsd_adjoint_backend='cupy')
    actual=gradient.kernel()
    error=float(np.max(abs(actual-expected)));assert error < 2e-8
    info=gradient.diagnostics['amplitudes']['ccsd_shared_overlaps']
    assert info['requests']>info['checkpoints'] and info['cache_hits']>0
    print('SHARED_OVERLAP_GRADIENT_REPORT '+json.dumps(dict(triples=triples,max_gradient_error=error,statistics=info)))
