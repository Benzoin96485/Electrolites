"""Exact compact ragged permutation with independently padded block oracle."""
import json
import numpy as np
import pytest
from electrolites.dlpno import reverse as ad


@pytest.mark.parametrize('count',[17,513])
def test_compact_packing_independent_blocks_padding_and_chunk_limits(monkeypatch,count):
    from electrolites.dlpno.ccsd_shared_overlap_gpu import RaggedPacking
    from electrolites.dlpno.ragged import Layout
    rng=np.random.default_rng(775);shapes=[(0,7),(17,31),(1,23),(17,31),(9,5)]
    shapes=[shapes[i%len(shapes)] for i in range(count)]
    keys=[(i,i+1) for i in range(count)];layout=Layout([x[0] for x in shapes],[x[1] for x in shapes])
    order=rng.permutation(count);source_keys=[keys[i] for i in order];source_shapes=[shapes[i] for i in order]
    offsets=np.cumsum([0]+[a*b for a,b in source_shapes]);raw=rng.normal(size=offsets[-1]);records=[]
    outputs=[];oracle_map=[]
    position={key:i for i,key in enumerate(source_keys)}
    for members,(pr,pc) in zip(layout.members,layout.shapes):
        out=np.zeros((len(members),pr,pc));mapping=np.full(out.shape,-1,dtype=np.int64)
        for slot,index in enumerate(members):
            source=position[keys[index]];a,b=offsets[source:source+2];nr,nc=shapes[index]
            out[slot,:nr,:nc]=raw[a:b].reshape(nr,nc)
            mapping[slot,:nr,:nc]=np.arange(a,b).reshape(nr,nc)
        outputs.append(out.ravel());oracle_map.append(mapping.ravel())
    expected=np.concatenate(outputs);mapping=np.concatenate(oracle_map);seed=rng.normal(size=expected.shape)
    wanted=np.zeros_like(raw);valid=mapping>=0;wanted[mapping[valid]]=seed[valid]
    for compact in [False,True]:
        monkeypatch.setenv('DLPNO_CCSD_COMPACT_PACKING',str(int(compact)))
        pack=RaggedPacking(layout,keys,source_keys,source_shapes,offsets,workspace_bytes=64<<10)
        leaf=ad.Variable(raw.copy());output=pack.pack(leaf);graph=ad.Linearization([output],[leaf])
        assert np.array_equal(output.v,expected)
        bar=graph([seed])[0];assert np.array_equal(bar,wanted)
        assert np.array_equal(graph([.37*seed])[0],.37*wanted)
        padded=np.ones_like(seed);padded[valid]=0.;assert np.max(abs(graph([padded])[0]),initial=0.)==0.
        direction=.01*rng.normal(size=raw.shape);h=1e-5
        plus=pack.pack(ad.Variable(raw+h*direction,tracked=False)).v
        minus=pack.pack(ad.Variable(raw-h*direction,tracked=False)).v
        error=abs(np.dot(plus-minus,seed)/(2*h)-np.dot(bar,direction));assert error<1e-7,error
        assert np.array_equal(graph([seed],release_graph=True,release_inputs=True)[0],wanted)
        if compact:
            assert pack.statistics['maximum_packing_workspace_bytes']<=64<<10
            assert pack.statistics['map_bytes']<raw.nbytes
            bounded=RaggedPacking(layout,keys,source_keys,source_shapes,offsets,map_bytes=pack.statistics['map_bytes'])
            assert np.array_equal(bounded.pack(ad.Variable(raw,tracked=False)).v,expected)
            monkeypatch.setenv('DLPNO_CCSD_COMPACT_PACKING','0')
            with pytest.raises(MemoryError,match='packing map'):RaggedPacking(layout,keys,source_keys,source_shapes,offsets,map_bytes=pack.statistics['map_bytes'])
        records.append(dict(compact=compact,fd_error=error,statistics=pack.statistics))
    print('COMPACT_CCSD_PACK_REPORT '+json.dumps(dict(blocks=count,records=records)))
