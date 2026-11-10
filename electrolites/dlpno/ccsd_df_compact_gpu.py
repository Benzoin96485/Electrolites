"""Exact DF group replay with local output blocks and one global scatter.

Each replay uses the unchanged upstream DF contractions and assignment order.
Only the addressed output slots survive a group checkpoint. The final bucket
operation adds the group contributions in their original order, retaining all
coefficient and amplitude derivatives and avoiding repeated global copies.
"""
from types import SimpleNamespace
import numpy as np
import cupy as cp
from . import reverse as ad


def scatter_groups(base, updates):
    """Add local blocks to one bucket; slots within each update are unique."""
    if not updates:
        return base
    operands = (base, *(value for slots, value in updates))
    slots = tuple(indices for indices, value in updates)
    value = ad.primal(base).copy()
    for index, block in updates:
        value[index] += ad.primal(block)
    def reverse(seed, needed):
        return (seed if needed[0] else None, *(seed[index] if need else None
                    for index, need in zip(slots, needed[1:])))
    return ad.operation(value, operands, lambda seed: reverse(seed, [True]*len(operands)), reverse)


def _template(group, names):
    """Retain full padded blocks, with global slots replaced by local slots."""
    host = {}
    for name in names:
        for gs, mi, sl, pr, pc in group[name]:
            host.setdefault(gs, []).append(cp.asnumpy(sl))
    slots = {gs: np.unique(np.concatenate(parts)) for gs, parts in host.items()}
    maps, shapes = {}, {}
    for name in names:
        local = []
        for gs, mi, sl, pr, pc in group[name]:
            index = cp.asarray(np.searchsorted(slots[gs], cp.asnumpy(sl)))
            local.append((gs, mi, index, pr, pc))
            shape = (len(slots[gs]), pr, pc)
            if gs in shapes and shapes[gs] != shape:
                raise ValueError('Inconsistent padded DF output block shape')
            shapes[gs] = shape
        maps[id(group[name])] = local
    return tuple(sorted(shapes)), shapes, maps, {gs:cp.asarray(indices) for gs,indices in slots.items()}


def compact_df_checkpoints(self, T, Tn, Fkc_bar, Fab, beta, Rsym):
    destinations = (Fkc_bar, Fab, beta, Rsym)
    updates = [{g:[] for g in range(x.layout.ngroups)} for x in destinations]
    diagonal, info = [], self.adjoint_df_statistics
    info.update(compact_local_outputs=True, global_output_bytes_avoided=0, compact_map_bytes=0)
    function = self._df_groups.__func__
    gather, put = self._gat, self._put
    global_bytes = sum(x.layout.nbytes() for x in destinations)
    for group in self.df:
        if group['dev'] != cp.cuda.Device().id:
            raise ValueError('DF checkpoint group is on a different GPU')
        templates = tuple(_template(group,names) for names in [('N',),('P',),('L','Lq'),('P',)])
        info['compact_map_bytes'] += sum(index.nbytes for _,_,maps,slots in templates
            for entries in maps.values() for gs,mi,index,pr,pc in entries)
        info['compact_map_bytes'] += sum(index.nbytes for _,_,_,slots in templates for index in slots.values())
        def forward(group=group, templates=templates, T=T, Tn=Tn):
            outputs = [SimpleNamespace(data={g:ad.Variable(cp.zeros(shapes[g]),tracked=False) for g in order},maps=maps)
                       for order,shapes,maps,slots in templates]
            def local_put(rag, items, value, add=False):
                put(rag, rag.maps[id(items)], value, add=add)
            view = SimpleNamespace(_gat=gather,_put=local_put)
            items = function(view,[group],T,Tn,*outputs)
            values = [out.data[g] for out,(order,_,_,_) in zip(outputs,templates) for g in order]
            for _,_,fai,fab,fkc,tn in items:
                values.extend((fai,fab,fkc,tn))
            return tuple(values)
        outputs, statistics = ad.checkpoint(forward)
        info['groups'] += 1
        info['discarded_records'] += statistics['records']
        info['largest_inner_value_bytes'] = max(info['largest_inner_value_bytes'],statistics['inner_value_bytes'])
        info['stored_output_bytes'] += statistics['output_bytes']
        local_bytes = sum(int(np.prod(shape))*8 for _,shapes,_,_ in templates for shape in shapes.values())
        info['global_output_bytes_avoided'] += global_bytes-local_bytes
        offset = 0
        for target,(order,shapes,maps,slots) in zip(updates,templates):
            for g in order:
                target[g].append((slots[g],outputs[offset]));offset+=1
        if len(group['di']):
            diagonal.append((group,group['dsel'],*outputs[offset:offset+4]));offset+=4
        if offset != len(outputs):
            raise RuntimeError('Unexpected compact DF output layout')
    for destination, target in zip(destinations,updates):
        for g in range(destination.layout.ngroups):
            destination.data[g] = scatter_groups(destination.data[g],target[g])
    return diagonal
