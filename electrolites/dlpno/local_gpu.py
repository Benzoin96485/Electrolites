"""Local RI provider for the existing batched GPU energy implementation.

The integral producer uses compact AO domains, with CPU or GPU three-center
evaluation. Local blocks are transferred to the requesting device; MP2,
CCSD and triples use the established batched GPU equations. No global
qia/qij array is built.
"""
from collections import OrderedDict
from threading import RLock

import numpy as np
from . import setup as su
import cupy as cp

from .dfints import DFInts, ExtDomains
from .local import LocalDLPNO
from .local_blocks import SelectionTensor
from .local_integrals import cartesian_axes, index_arrays
from .rhf import DLPNO


def project_local_triplet_pairs(source, pair_tables, bidx, pidx, max_bytes=1 << 28):
    """X_p.T S[d_p,d_t] X_t without a triplet-by-global-AO intermediate.

    The source holds the global reference metric and compact triplet
    indices/masks/coefficients. Only the requested PAO submatrices are
    gathered. Work per requested pair/triplet edge is bounded by its
    local spaces, independently of the total AO count.
    """
    metric, indices, mask, coefficients = source
    pp, vp, v, pt = pair_tables['PP'], pair_tables['VP'], coefficients.shape[2], indices.shape[1]
    count = len(pidx)
    step = max(1, int(max_bytes // max(1, 8*(pp*pt+pp*v+pp*vp+pt*v+vp*v))))
    out = cp.empty((count, vp, v))
    for start in range(0, count, step):
        b, p = bidx[start:start+step], pidx[start:start+step]
        left, right = pair_tables['dom'][p], indices[b]
        block = metric[left[:, :, None], right[:, None, :]]
        block *= pair_tables['dmk'][p][:, :, None]*mask[b][:, None, :]
        projected = cp.matmul(block, coefficients[b])
        out[start:start+step] = cp.matmul(pair_tables['X'][p].transpose(0, 2, 1), projected)
    return out


class LocalDFInts(DFInts):
    def __init__(self, base, ext):
        self.provider, self.ext = base.local_integrals, ext
        self.mol, self.auxmol = base.mol, base.auxmol
        self.naux, self.n, self.nbf = base.auxmol.nao, base.n, base.orb.nbf
        self.ribf_atom, self.atom_ribfs = base.ribf_atom, base.atom_ribfs
        self._lock = RLock()
        self._local_cache, self._local_bytes = {}, {}
        self._chol = {}
        self.qab_host = True
        self.cache_stats = dict(hit=0, miss=0, bytes=0.)
        self.nbytes = dict(ia=0, ij=0, ab=0)
        for attribute, name in [('qia', 'qia'), ('qij', 'qij'), ('J2c', 'metric')]:
            setattr(self, attribute, SelectionTensor(self.provider.shapes[name],
                                                     lambda key, name=name: self.get(name, key)))

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop('_lock')
        state.update(_local_cache={}, _local_bytes={}, _chol={})
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = RLock()

    def get(self, name, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = tuple(cp.asnumpy(i) if isinstance(i, cp.ndarray) else i for i in key)
        with self._lock:
            if name == 'metric':
                # Each batched sample has its own local fitting domain.
                # Do not build a metric over the union of a whole batch.
                p, q = index_arrays(key, self.provider.shapes[name])
                if p.ndim > 2:
                    shape = p.shape
                    p, q = p.reshape(-1, *shape[-2:]), q.reshape(-1, *shape[-2:])
                    value = np.stack([self.provider.metric(a, b) for a, b in zip(p, q)]).reshape(shape)
                else:
                    value = self.provider.metric(p, q)
            else:
                # GPU pair/triplet batches prepend a sample axis to
                # Cartesian domain selectors. Stream samples without
                # expanding three integer arrays over B*Q*i*u entries.
                ndim = max(np.ndim(x) for x in key)
                batched = (ndim >= 2 and all(np.ndim(x) == ndim for x in key)
                           and key[0].shape[0] > 1
                           and all(x.shape[0] in (1, key[0].shape[0]) for x in key))
                if batched:
                    sample = tuple(x[0] for x in key)
                    batched = cartesian_axes(sample, self.provider.shapes[name]) is not None
                if batched:
                    shape = index_arrays(key, self.provider.shapes[name])[0].shape
                    value = np.empty(shape)
                    for batch in range(shape[0]):
                        sample = tuple(x[batch if x.shape[0] > 1 else 0] for x in key)
                        value[batch] = self.provider.read(name, sample)
                    self.provider.counts['compact_batched_reads'] = self.provider.counts.get('compact_batched_reads', 0)+1
                else:
                    value = self.provider.read(name, key)
        return cp.asarray(value)

    def qab_get(self, atom):
        paos = self.ext.paos[atom]
        if not len(paos):
            return None
        dev = cp.cuda.Device().id
        with self._lock:
            cache = self._local_cache.setdefault(dev, OrderedDict())
            if atom in cache:
                self.cache_stats['hit'] += 1
                cache.move_to_end(atom)
                return cache[atom]
            self.cache_stats['miss'] += 1
            key = np.ix_(self.atom_ribfs[atom], paos, paos)
            block = cp.asarray(self.provider.read('qab', key))
            used = self._local_bytes.get(dev, 0)
            while cache and used+block.nbytes > self.provider.budget:
                _, old = cache.popitem(last=False)
                used -= old.nbytes
            if block.nbytes <= self.provider.budget:
                cache[atom] = block
                used += block.nbytes
            self._local_bytes[dev] = used
            self.cache_stats['bytes'] += block.nbytes
        return block

    def metric_chol(self, atoms):
        key = cp.cuda.Device().id, tuple(map(int, atoms))
        with self._lock:
            if key not in self._chol:
                indices = self.aux_index(atoms)
                self._chol[key] = cp.linalg.cholesky(self.get('metric', np.ix_(indices, indices)))
            return self._chol[key]

    def cache_clear(self):
        with self._lock:
            self._local_cache.clear()
            self._local_bytes.clear()
            self._chol.clear()


class LocalGPUDLPNO(DLPNO, LocalDLPNO):
    """Batched GPU energy and the same local-integral adjoint gradient."""

    def prepare_gradient_storage(self):
        from .primal_storage import prepare_gradient_storage
        return prepare_gradient_storage(self)

    def _projection_source(self, metric, indices, mask, coefficients):
        return metric, indices, mask, coefficients

    def _project(self, source, bidx, pidx):
        return project_local_triplet_pairs(source, self._pt, bidx, pidx)

    def setup(self):
        LocalDLPNO.setup(self)
        # Batched energy kernels consume ordinary orbital matrices.  The
        # adjoint installs sparse primitive views on its own state.
        self.orb, self.dip_e = self._orbital_reference, self._dipole_reference
        opt = self.opt
        mask = (self._initial_pair_mask if hasattr(self, '_initial_pair_mask') else
                ((self.doi_ij > opt['T_CUT_DO_IJ']) | (abs(self.dip_b) > opt['T_CUT_PRE'])
                 | np.eye(self.n, dtype=bool)))
        dom = self.domains(min(opt['T_CUT_MKN'], opt['T_CUT_MKN_TRIPLES']),
                           min(opt['T_CUT_DO'], opt['T_CUT_DO_TRIPLES']))
        self.ext = ExtDomains(self.mol.natm, dom.lmo_riatoms, dom.lmo_paos,
                              su.mask_pairs(mask), self.orb.nbf, sparse_positions=True)
        self.local_integrals.plan_domains(self.ext.lmos, self.ext.paos)
        self.dfi = LocalDFInts(self, self.ext)
        self.J2c = self.dfi.J2c
        self.S_pao_d = cp.asarray(self.orb.S_pao)
