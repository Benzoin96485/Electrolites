"""Numerical T1 solver parity and parent lifetime for selective primal snapshots."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import AdjointDLPNO,host,_compact_triples_primal_state
from test_dlpno_triples_residual_gpu import raw_sources,state_from,gpu

@pytest.mark.parametrize('backend',['numpy','cupy'])
@pytest.mark.parametrize('cut',[0.,.035,2.])
def test_compact_snapshot_full_gpu_iteration_matches_legacy(backend,cut):
    cp=gpu()
    from electrolites.dlpno.triples_gpu import TriplesGPU
    raw,rng=raw_sources();xp=cp if backend=='cupy' else np
    variables={k:ad.Variable(xp.asarray(v.copy())) for k,v in raw.items()}
    view=state_from(variables,cut);source=object.__new__(AdjointDLPNO);source.__dict__.update(vars(view))
    source.n=3;source.verbose=0;source.xp=np
    source.opt.update(R_CONVERGENCE=1e-12,E_CONVERGENCE=1e-12,MAXITER=100,T_CUT_ITER=0.)
    source.e_ijk=ad.Variable(xp.zeros(4));source.V3=[ad.Variable(xp.asarray(rng.normal(size=x.shape))) for x in source.T3]
    # Unrelated CCSD state and a cache discarded by the old snapshot builder.
    source.unused_ccsd_integrals=ad.Variable(xp.ones((1024,1024)))
    source._S={(0,1):ad.Variable(xp.ones((512,512)))}
    before={k:host(v).copy() for k,v in variables.items()}
    legacy=AdjointDLPNO._primal_state(source)
    class MustNotCopy:
        def primal_copy(self):raise AssertionError('The T1 snapshot touched an unrelated CCSD object')
    source.unused_marker=MustNotCopy()
    compact=_compact_triples_primal_state(source)
    assert not hasattr(compact,'unused_ccsd_integrals') and not hasattr(compact,'unused_marker')
    assert compact._S=={} and legacy._S=={}
    energies=[];amplitudes=[]
    for state in (legacy,compact):
        state.S_pao_d=cp.asarray(state.orb.S_pao);state._trip_arr=np.asarray(state.trip,dtype=int)
        energy=TriplesGPU.lccsd_t_iterations(state,return_amplitudes=True)
        energies.append(energy);amplitudes.append(state.T3)
        del state.S_pao_d
    error=max(float(np.max(abs(a-b))) for a,b in zip(*amplitudes))
    assert error<2e-12 and abs(energies[0]-energies[1])<2e-12
    for k,v in variables.items():np.testing.assert_array_equal(host(v),before[k])
    assert source.W3 is not None and source.V3 is not None and source.T3 is not None
    assert np.all(host(source.e_ijk)==0.)
    compact.T3[0].fill(99.)
    np.testing.assert_array_equal(host(source.T3[0]),before['t0'])
    print('COMPACT_T1_PRIMAL_REPORT '+json.dumps(dict(backend=backend,cut=cut,amplitude_error=error,energy_error=abs(energies[0]-energies[1]),irrelevant_copy_avoided_bytes=source.unused_ccsd_integrals.size*8+source._S[(0,1)].size*8,parent_unchanged=True)))
