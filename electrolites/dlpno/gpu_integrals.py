"""GPU local three-center integrals and auxiliary-atom density contractions.

The host provider retains its bounded cache and discrete support maps.
Only one compact AO/auxiliary block is sent through the GPU at a time;
there is no global AO three-center integral or derivative allocation.
Two-center metrics and the small index maps remain on the CPU.
"""
import time

import numpy as np
import cupy as cp
from gpu4pyscf.df.int3c2e_bdiv import (
    aux_e2, Int3c2eOpt, _int3c2e_ip1_evaluator,
    int3c2e_scheme_ip1, int3c2e_scheme_ipaux)

from .local_integrals import LocalRIBlocks


class GPULocalRIBlocks(LocalRIBlocks):
    def _evaluate_raw(self, mol, aux, intor, comp):
        kinds = {'int3c2e': ('', 1), 'int3c2e_ip1': ('ip1', 3),
                 'int3c2e_ip2': ('ip2', 3)}
        if intor not in kinds or kinds[intor][1] != comp:
            raise NotImplementedError(f'Unsupported GPU local integral {intor}, comp={comp}')
        start = time.perf_counter()
        # The Rys implementation supports independent AO/auxiliary atom
        # tables. Legacy VHFOpt incorrectly aliases their atom numbers
        # for independently compacted local domains.
        if comp == 1:
            value = aux_e2(mol, aux)
        else:
            value = self._derivative(mol, aux, intor)
        result = value.get()
        self.counts['gpu_integral_calls'] = self.counts.get('gpu_integral_calls', 0)+1
        self.counts['gpu_integral_seconds'] = self.counts.get('gpu_integral_seconds', 0.)+time.perf_counter()-start
        self.counts['gpu_largest_integral_bytes'] = max(self.counts.get('gpu_largest_integral_bytes', 0), value.nbytes)
        return result

    @staticmethod
    def _derivative(mol, aux, intor):
        # ip1 is not symmetric in its AO indices. Request both shell
        # orientations, then restore contractions and spherical order.
        opt = Int3c2eOpt(mol, aux).build(cutoff=1e-14, tril=False)
        # fill_int3c2e_ip1/ipaux have GOUT_IP1_WIDTH=27. The helper's
        # default omits this bound and leaves high-l output unwritten.
        scheme, kernel = ((int3c2e_scheme_ip1(gout_width=27), 'fill_int3c2e_ip1')
                          if intor == 'int3c2e_ip1' else
                          (int3c2e_scheme_ipaux(gout_width=27), 'fill_int3c2e_ipaux'))
        evaluate, sorting, offsets = _int3c2e_ip1_evaluator(
            opt, scheme, max(1, opt.auxmol.nao), kern=kernel)
        aux_coeff = cp.empty_like(opt.aux_coeff)
        aux_coeff[sorting] = opt.aux_coeff
        address = opt.pair_and_diag_indices(cart=True, original_ao_order=False)[0]
        rows, cols = divmod(address, opt.mol.nao)
        coeff = opt.coeff
        result = cp.zeros((3, mol.nao, mol.nao, aux.nao))
        for batch in range(len(offsets)-1):
            buffer = cp.zeros((3, address.size, offsets[batch+1]-offsets[batch]))
            block = evaluate(batch, out=buffer)
            matrix = cp.zeros((3, opt.mol.nao, opt.mol.nao, block.shape[-1]))
            matrix[:, rows, cols] = block
            result += cp.einsum('xmnp,mi,nj,pQ->xijQ', matrix, coeff, coeff,
                                aux_coeff[offsets[batch]:offsets[batch+1]], optimize=True)
        return result

    def _transform_raw(self, raw, left, right):
        start = time.perf_counter()
        result = cp.einsum('mnQ,mi,nj->Qij', cp.asarray(raw), cp.asarray(left),
                           cp.asarray(right), optimize=True).get()
        self.counts['gpu_transform_calls'] = self.counts.get('gpu_transform_calls', 0)+1
        self.counts['gpu_transform_seconds'] = self.counts.get('gpu_transform_seconds', 0.)+time.perf_counter()-start
        return result

    def _pullback_aux_atom(self, atom, terms, coefficient_bars, force):
        start = time.perf_counter()
        base = self.base
        kinds = dict(qij=('lmo', 'lmo'), qia=('lmo', 'pao'), qab=('pao', 'pao'))
        columns = dict(lmo=set(), pao=set())
        for name, _, i, j, _ in terms:
            left, right = kinds[name]
            columns[left].update(i)
            columns[right].update(j)
        columns = {k: np.asarray(sorted(v), dtype=int) for k, v in columns.items() if v}
        ao_atoms = np.unique(np.concatenate([self.maps.union(k, v) for k, v in columns.items()]))
        rows = np.concatenate([base.atom_bfs[int(a)] for a in ao_atoms])
        coeff = {k: cp.asarray(self.maps.matrix(k, v, ao_atoms)[1]) for k, v in columns.items()}
        nq = len(base.atom_ribfs[atom])
        h = {k: cp.zeros((nq, len(rows), len(v))) for k, v in columns.items()}
        for name, q, i, j, weight in terms:
            kind1, kind2 = kinds[name]
            left, right = np.unique(i), np.unique(j)
            p1, p2 = np.searchsorted(columns[kind1], left), np.searchsorted(columns[kind2], right)
            density = cp.zeros((nq, len(left), len(right)))
            indices = (np.searchsorted(base.atom_ribfs[atom], q),
                       np.searchsorted(left, i), np.searchsorted(right, j))
            indices = tuple(cp.asarray(index) for index in indices)
            weight = cp.asarray(weight)
            if weight.ndim == 3:
                if name == 'qab':
                    weight = weight*cp.asarray(self.pao_pair_mask[i[:, None], j])[None]
                indices = cp.ix_(*indices)
            cp.add.at(density, indices, weight)
            h[kind1][:, :, p1] += cp.einsum('nj,Qij->Qni', coeff[kind2][:, p2], density)
            h[kind2][:, :, p2] += cp.einsum('mi,Qij->Qmj', coeff[kind1][:, p1], density)
        raw = cp.asarray(self.raw(ao_atoms, [atom]))
        gamma = cp.zeros_like(raw)
        self.counts['largest_backtransform_ao'] = max(self.counts.get('largest_backtransform_ao', 0), len(rows))
        self.counts['peak_local_density_bytes'] = max(self.counts.get('peak_local_density_bytes', 0),
                                                     gamma.nbytes+sum(v.nbytes for v in h.values()))
        for kind, selected in columns.items():
            gradient = cp.einsum('mnQ,Qni->mi', raw, h[kind], optimize=True)
            gradient *= cp.asarray(self.maps.mask(kind, selected, rows))
            coefficient_bars[kind][np.ix_(rows, selected)] += gradient.get()
            gamma += .5*cp.einsum('mi,Qni->mnQ', coeff[kind], h[kind], optimize=True)
        derivative = cp.asarray(self.raw(ao_atoms, [atom], 'int3c2e_ip1', 3))
        centers = -cp.einsum('mnQ,xmnQ->xm', gamma+gamma.transpose(1, 0, 2), derivative).get()
        for xyz in range(3):
            np.add.at(force[:, xyz], base.bf_atom[rows], centers[xyz])
        del derivative
        derivative = cp.asarray(self.raw(ao_atoms, [atom], 'int3c2e_ip2', 3))
        force[atom] -= cp.einsum('mnQ,xmnQ->x', gamma, derivative).get()
        self.counts['gpu_backtransform_atoms'] = self.counts.get('gpu_backtransform_atoms', 0)+1
        self.counts['gpu_backtransform_seconds'] = self.counts.get('gpu_backtransform_seconds', 0.)+time.perf_counter()-start

    def statistics(self):
        result = super().statistics()
        result['producer'] = 'local_gpu_3center_host_metric'
        return result
