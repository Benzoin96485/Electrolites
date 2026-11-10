"""Packed Fock sources against scalar indexing and independent differences."""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from electrolites.dlpno import reverse as ad
from electrolites.dlpno.triples_resident_gpu import (
    batched_fock_sources, batched_overlap_indices, _flat)
from test_dlpno_triples_residual_gpu import gpu


def sparse_view(value):
    from electrolites.dlpno.local_blocks import SparseMatrix
    cache={}
    def get(key):
        if key not in cache:cache[key]=value[key]
        return cache[key]
    return SparseMatrix(value.shape,get,get)


def legacy(fock, triplets, connections, cut):
    active, weights = [], []
    for rows in connections:
        here = []
        for key, u, occ, old in rows:
            value = fock[occ, old]
            if abs(float(ad.primal(value))) >= cut:
                here.append((key, u, None))
                weights.append(value)
        active.append(here)
    occupied = [fock[i, i] + fock[j, j] + fock[k, k] for i, j, k in triplets]
    return active, _flat([x.reshape(1) for x in occupied]), _flat([x.reshape(1) for x in weights])


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
@pytest.mark.parametrize('cut', [0., .03, .035, 2.])
def test_fock_sources(backend, cut, sparse):
    cp = gpu(); xp = cp if backend == 'cupy' else np
    f = np.array([[-1., .03, -.04], [.07, -.8, .02], [-.05, .02, -.6]])
    triplets = [(0, 0, 1), (0, 1, 2), (2, 2, 2)]
    connections = [
        [((0, 1, 2), 1, 0, 1), ((1, 0, 2), 1, 0, 1), ((2, 0, 1), 2, 1, 0)],
        [], [((0, 1, 2), 0, 2, 0), ((2, 2, 2), 2, 2, 2)]]
    leaf, reference = ad.Variable(xp.asarray(f)), ad.Variable(xp.asarray(f))
    source = sparse_view(leaf) if sparse else leaf
    oracle = sparse_view(reference) if sparse else reference
    active, *actual = batched_fock_sources(source, triplets, connections, cut)
    old_active, *expected = legacy(oracle, triplets, connections, cut)
    assert active == old_active
    rng = np.random.default_rng(643)
    seeds = [xp.asarray(rng.normal(size=x.shape)) for x in actual]
    graph = ad.Linearization(actual, [leaf])
    bar = graph(seeds)[0]
    gold = ad.Linearization(expected, [reference])(seeds)[0]
    def maximum(delta): return float(xp.max(xp.abs(delta))) if delta.size else 0.
    value_error = max(maximum(xp.asarray(x.v)-xp.asarray(y.v)) for x,y in zip(actual,expected))
    vjp_error = maximum(bar-gold)
    assert value_error == 0. and vjp_error < 3e-14
    assert float(xp.max(xp.abs(graph([.37*z for z in seeds])[0]-.37*bar))) < 3e-14
    direction = .1*rng.normal(size=f.shape)
    fd_error = None
    if cut != .03:
        def numeric(raw):
            values = [raw[i, i]+raw[j, j]+raw[k, k] for i,j,k in triplets]
            weights = [raw[i,j] for rows in connections for _,_,i,j in rows if abs(raw[i,j]) >= cut]
            z = [cp.asnumpy(x) if backend=='cupy' else x for x in seeds]
            return np.dot(values,z[0])+np.dot(weights,z[1])
        h=2e-5
        finite=(numeric(f+h*direction)-numeric(f-h*direction))/(2*h)
        analytic=float(xp.sum(bar*xp.asarray(direction)))
        fd_error=abs(finite-analytic)
        assert fd_error < 5e-10
    consumed=graph(seeds,release_graph=True,release_inputs=True)[0]
    assert float(xp.max(xp.abs(consumed-bar))) < 3e-14 and leaf.v is None
    print('TRIPLES_METADATA_REPORT '+json.dumps(dict(backend=backend,cut=cut,sparse=sparse,
        value_error=value_error,vjp_error=vjp_error,directional_fd_error=fd_error,
        records=graph.statistics['records'])))


def test_empty_connections():
    f = ad.Variable(np.eye(2))
    active,occupied,edges=batched_fock_sources(f,[],[],0.)
    assert active==[] and occupied.shape==edges.shape==(0,)
    bars=ad.Linearization([occupied,edges],[f])([np.empty(0),np.empty(0)])
    assert np.array_equal(bars[0],np.zeros((2,2)))


def test_overlap_index_maps():
    rng=np.random.default_rng(644);ranks=np.array([0,1,7,9,3])
    edges=[(int(t),int(u),s,None,None) for s,(t,u) in enumerate(rng.integers(0,5,size=(17,2)))]
    offsets=np.cumsum([0]+[int(ranks[t]*ranks[u]) for t,u,*_ in edges])
    selected=[4,2,4,0,16,5]
    actual=batched_overlap_indices(ranks,offsets,edges,selected,16,16)
    expected=np.full(actual.shape,offsets[-1],dtype=np.int64)
    for row,e in enumerate(selected):
        t,u,s,_,_=edges[e]
        expected[row,:ranks[t],:ranks[u]]=offsets[s]+np.arange(ranks[t])[:,None]*ranks[u]+np.arange(ranks[u])
    assert np.array_equal(actual,expected)
    assert batched_overlap_indices(ranks,offsets,edges,[],16,16).shape==(0,16,16)


def test_fixed_checkpoint_metadata_micro():
    """Only saved COF2 coupling metadata, no new energy or gradient solve."""
    gpu()
    import cloudpickle
    from electrolites.dlpno.ref_rhf import RefDLPNO
    p=Path('/pscratch/sd/w/weiliang/dlpno-gpu-gradient-20261018/cof384-primal.pkl')
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='fce75b5efd87b69440c622a13a3e08cbe9fcbbf817fd2b6f135b72cf51d11741'
    with p.open('rb') as stream:base=cloudpickle.load(stream)
    assert base.mol.natm==4 and base.mol.nao==56
    raw=np.asarray(base._orbital_reference.F_lmo)
    connections=RefDLPNO._triples_couplings(base)
    triplets=base.trip;cut=base.opt['F_CUT_T']
    rng=np.random.default_rng(645);runs=[];expected=None
    for mode in ['legacy','packed','packed','legacy','packed','legacy']:
        leaf=ad.Variable(raw.copy());begin=time.perf_counter()
        active,*outputs=(legacy if mode=='legacy' else batched_fock_sources)(sparse_view(leaf),triplets,connections,cut)
        graph=ad.Linearization(outputs,[leaf]);forward=time.perf_counter()-begin
        # Fixed seeds ensure all duplicate occurrences contribute identically.
        seeds=[np.cos(np.arange(x.size)).reshape(x.shape) for x in outputs]
        begin=time.perf_counter();bar=graph(seeds,release_graph=True)[0];reverse=time.perf_counter()-begin
        if expected is None:expected=bar.copy()
        error=float(np.max(abs(bar-expected),initial=0.));assert error<1e-11
        runs.append(dict(mode=mode,forward_seconds=forward,reverse_seconds=reverse,
            complete_seconds=forward+reverse,records=graph.statistics['records'],vjp_error=error))
    old=np.median([x['complete_seconds'] for x in runs if x['mode']=='legacy'])
    new=np.median([x['complete_seconds'] for x in runs if x['mode']=='packed'])
    report=dict(complete=True,source_representation="cached_sparse_block_view",atoms=4,ao=56,triplets=len(triplets),couplings=sum(map(len,active)),runs=runs,
        median_old_seconds=float(old),median_new_seconds=float(new),median_saving_seconds=float(old-new),
        molecular_gradients=0,new_scf_calculations=0,new_energy_calculations=0,
        limitation='Local source assembly and pullback only; not a complete-gradient speedup.')
    print('TRIPLES_METADATA_MICRO '+json.dumps(report),flush=True)


def test_complete_resident_t1_with_sparse_fock(monkeypatch):
    import test_dlpno_triples_resident_gpu as checks
    original=checks.state_from
    def state(values,cut):
        result=original(values,cut)
        result.orb.F_lmo=sparse_view(result.orb.F_lmo)
        return result
    monkeypatch.setattr(checks,'state_from',state)
    checks.test_resident_complete_sources((0,),.035)
