"""Production gather plans against ordinary CPU reads and independent FD."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad
from electrolites.dlpno.adjoint import host
from electrolites.dlpno.local_blocks import ReferenceBlocks
from electrolites.dlpno.local_read_plans import tracked_ri_read,tracked_orbital_read
from electrolites.dlpno.deferred_reads import DeferredRead,OperandBank,capture_reads
from electrolites.dlpno.parallel_local_gpu import Capture,_decode


class Blocks(ReferenceBlocks):
 def __init__(self,arrays):
  self.sources=arrays;self.shapes={k:v.shape for k,v in arrays.items()}
  self.leaves={};self.selections={};self.xp=np;self.counts={}
  self.orbital_sources=arrays;self.orbital_tile_size=2
  atom=np.array([0,0,0,1,1,2,2,2])
  self.base=SimpleNamespace(ribf_atom=atom,atom_ribfs=[np.flatnonzero(atom==a) for a in range(3)],
    local_integrals=SimpleNamespace(column_envelopes={('qia',a):(np.arange(4),np.arange(6)) for a in range(3)}))


@pytest.mark.parametrize('case',['ri_identity','ri_reordered_repeat','orbital_reordered_repeat'])
def test_production_read_plan(case):
 import cupy as cp
 rng=np.random.default_rng(24970)
 arrays=dict(qia=rng.normal(size=(8,4,6)),s_pao=rng.normal(size=(7,7)))
 blocks=Blocks(arrays)
 if case.startswith('ri'):
  name='qia';read=tracked_ri_read
  key=np.ix_(np.arange(8),np.arange(4),np.arange(6)) if case=='ri_identity' else np.ix_([7,2,5,2,0],[3,1,3],[5,1,0])
  other=np.ix_([0,4,6],[2,0],[1,4])
 else:
  name='s_pao';read=tracked_orbital_read
  key=np.ix_([6,1,4,1],[0,5,2,5]);other=np.ix_([2,6],[4,0])
 expected=(read(blocks,name,key),read(blocks,name,other))
 with capture_reads() as stats:
  deferred=(read(blocks,name,key),read(blocks,name,other))
 assert stats['reads']==2 and all(isinstance(x,DeferredRead) for x in deferred)
 cap=Capture();encoded=tuple(cap.value(x) for x in deferred)
 assert len(cap.inputs)==len(blocks.leaves)
 seeds=[rng.normal(size=x.shape) for x in expected]
 gold=ad.Linearization(expected,cap.inputs)(seeds)
 values=tuple(ad.Variable(cp.asarray(ad.primal(x))) for x in cap.inputs)
 bank=OperandBank(values);outputs=tuple(_decode(x,bank) for x in encoded)
 assert _decode(encoded[0],bank) is outputs[0]
 assert bank.statistics['materialized']==2 and bank.statistics['cache_hits']==1
 assert outputs[0] is not outputs[1]
 assert max(np.max(abs(host(a)-host(b)),initial=0.) for a,b in zip(outputs,expected))==0.
 graph=ad.Linearization(outputs,values)
 bars=graph([cp.asarray(x) for x in seeds])
 error=max(float(np.max(abs(host(a)-b),initial=0.)) for a,b in zip(bars,gold))
 assert error<1e-12,error
 indices=list(range(0,len(values),2));selected=ad.Linearization(outputs,[values[i] for i in indices],boundary=values)
 selected_bars=selected([cp.asarray(x) for x in seeds])
 assert max(np.max(abs(host(a)-gold[i]),initial=0.) for a,i in zip(selected_bars,indices))<1e-12
 # Independent numerical perturbations of the original source arrays, not the encoded reader.
 directions={k:.1*rng.normal(size=v.shape) for k,v in arrays.items()};h=1e-5
 evaluate=lambda sign:sum(np.sum((arrays[name]+sign*h*directions[name])[k]*z) for k,z in zip((key,other),seeds))
 fd=(evaluate(1)-evaluate(-1))/(2*h)
 tokens={leaf._record:token for token,leaf in blocks.leaves.items()}
 analytic=sum(float(np.sum(host(bar)*directions[tokens[x._record][0]][blocks.selections[tokens[x._record]]])) for x,bar in zip(cap.inputs,bars))
 assert abs(fd-analytic)<2e-8,(fd,analytic)
 final=graph([cp.asarray(x) for x in seeds],release_graph=True,release_inputs=True)
 assert all(v.v is None for v in values)
 assert max(np.max(abs(host(a)-b),initial=0.) for a,b in zip(final,gold))<1e-12
 print('DEFERRED_READ_REPORT '+json.dumps(dict(case=case,vjp_error=error,directional_fd_error=abs(fd-analytic),capture=stats,bank=bank.statistics)))
