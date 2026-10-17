"""Three-index integrals in the LMO/PAO basis, on the device.

(Q|i u), (Q|i j) are kept as dense device arrays over all auxiliary
functions (naux * nlmo * nbf, the size of the occupied half of an MP2 DF
tensor); (Q|u v) is kept per auxiliary atom A as a dense block over the
PAOs of A's *extended* domain only (Psi4's ``riatom_to_paos_ext``: the PAO
domains of every LMO whose fitting domain, or that of an LMO it pairs with,
contains A), masked by DOI(u, v) > T_CUT_DO_UV as in Psi4.

AO three-centre integrals come from GPU4PySCF's ``int3c2e_bdiv`` one
group of auxiliary atoms at a time, so the (nao, nao, naux) tensor is never
held whole.
"""
import os
from collections import OrderedDict

import numpy as np
import cupy as cp

from electrolites.dlpno.devmem import to_pinned

from pyscf import gto
from gpu4pyscf.df import int3c2e_bdiv


def sub_auxmol(auxmol, atoms):
    """auxmol restricted to the shells on ``atoms`` (AO order preserved)."""
    sub = auxmol.copy(deep=False)
    sel = np.isin(auxmol._bas[:, gto.ATOM_OF], atoms)
    sub._bas = auxmol._bas[sel]
    return sub


def aux_e2_atoms(mol, auxmol, atoms):
    """(mu nu|Q) for the aux functions on ``atoms``, as (nao, nao, naux_sub)."""
    sub = sub_auxmol(auxmol, atoms)
    return int3c2e_bdiv.aux_e2(mol, sub)


def _decide_qab_host(nbytes):
    """keep the (Q|uv) blocks in pinned host memory when they would take
    more than 35% of the free device memory (DLPNO_QAB_HOST=auto|0|1)."""
    env = os.environ.get('DLPNO_QAB_HOST', 'auto')
    if env != 'auto':
        return env == '1'
    free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
    return nbytes > 0.35 * free


class ExtDomains:
    """Psi4's extended LMO / PAO domains of each auxiliary atom."""

    def __init__(self, natm, lmo_riatoms, lmo_paos, pairs, nbf):
        n = len(lmo_riatoms)
        ext = [set() for _ in range(n)]
        for (i, j) in pairs:
            ext[i].update(lmo_riatoms[j].tolist())
        self.lmos = [[] for _ in range(natm)]
        for i in range(n):
            for a in ext[i]:
                self.lmos[a].append(i)
        self.lmos = [np.array(sorted(l), dtype=np.int64) for l in self.lmos]
        self.paos = []
        for a in range(natm):
            if len(self.lmos[a]):
                self.paos.append(np.unique(np.concatenate([lmo_paos[i] for i in self.lmos[a]])))
            else:
                self.paos.append(np.zeros(0, np.int64))
        # dense position lookups, -1 when absent
        self.pao_pos = -np.ones((natm, nbf), dtype=np.int64)
        for a in range(natm):
            self.pao_pos[a, self.paos[a]] = np.arange(len(self.paos[a]))


class DFInts:
    """Device three-index integrals for one set of LMOs/PAOs."""

    def __init__(self, mol, auxmol, C_lmo, C_pao, ext, doi_uv, t_cut_do_uv,
                 ribf_atom, atom_ribfs, group_mem=2 << 30, need_ab=True):
        self.mol, self.auxmol = mol, auxmol
        nao = mol.nao
        naux = auxmol.nao
        n = C_lmo.shape[1]
        nbf = C_pao.shape[1]
        self.naux, self.n, self.nbf = naux, n, nbf
        L = cp.asarray(C_lmo)
        P = cp.asarray(C_pao)
        self.qia = cp.empty((naux, n, nbf))
        self.qij = cp.empty((naux, n, n))
        self.qab = [None] * mol.natm
        self.ext = ext
        mask_uv = cp.asarray(doi_uv > t_cut_do_uv) if need_ab else None
        natm = mol.natm
        # (Q|uv) blocks: on the device when they fit, else in pinned host
        # memory behind a device cache (``qab_get``)
        nab = sum(8. * len(atom_ribfs[a]) * len(ext.paos[a]) ** 2 for a in range(natm)) if need_ab else 0.
        self.qab_host = _decide_qab_host(nab)
        # several GPUs: rather than pinned host memory, shard the blocks over
        # the other devices (served by peer copies over NVLink)
        self._shard = None
        if ((self.qab_host and os.environ.get('DLPNO_QAB_HOST', 'auto') == 'auto')
                or os.environ.get('DLPNO_QAB_SHARD') == '1'):
            from electrolites.dlpno import mgpu
            devs = mgpu.devices()
            home = cp.cuda.Device().id
            others = [d for d in devs if d != home]
            if others:
                cap = 0.
                for d in others:
                    with cp.cuda.Device(d):
                        cap += cp.cuda.runtime.memGetInfo()[0]
                if nab < 0.35 * cap:
                    mgpu.enable_peer(devs)
                    self.qab_host = False
                    self._shard = dict(devs=others, load=np.zeros(len(others)))
        self._cache = {}               # device -> OrderedDict(atom -> block)
        self._cache_bytes = {}
        self.cache_stats = dict(hit=0, miss=0, bytes=0.)
        # group aux atoms so that one (nao, nao, naux_group) block fits
        per_aux = nao * nao * 8 * 2
        groups, cur, cur_n = [], [], 0
        for a in range(natm):
            na = len(atom_ribfs[a])
            if cur and (cur_n + na) * per_aux > group_mem:
                groups.append(cur)
                cur, cur_n = [], 0
            cur.append(a)
            cur_n += na
        if cur:
            groups.append(cur)
        for atoms in groups:
            idx = np.concatenate([atom_ribfs[a] for a in atoms])
            if len(idx) == 0:
                continue
            j3c = aux_e2_atoms(mol, auxmol, atoms)              # (nao, nao, nq)
            nq = j3c.shape[2]
            assert nq == len(idx)
            # (Q|i nu) then (Q|i u), (Q|i j)
            t = cp.matmul(L.T, j3c.reshape(nao, nao * nq)).reshape(n, nao, nq)
            t = t.transpose(2, 0, 1)                               # (nq, n, nao)
            gi = cp.asarray(idx)
            self.qia[gi] = cp.matmul(t, P)
            self.qij[gi] = cp.matmul(t, L)
            del t
            if need_ab:
                off = 0
                for a in atoms:
                    na = len(atom_ribfs[a])
                    pa = ext.paos[a]
                    if na == 0 or len(pa) == 0:
                        off += na
                        continue
                    Pa = P[:, cp.asarray(pa)]
                    blk = j3c[:, :, off:off + na]                       # (nao, nao, na)
                    u = cp.matmul(Pa.T, blk.reshape(nao, nao * na)).reshape(len(pa), nao, na)
                    u = cp.matmul(u.transpose(2, 0, 1), Pa)            # (na, npa, npa)
                    m = mask_uv[cp.asarray(pa)[:, None], cp.asarray(pa)[None, :]]
                    u *= m[None]
                    del m
                    if self.qab_host:
                        self.qab[a] = to_pinned(u)
                    elif self._shard is not None:
                        sh = self._shard
                        r = int(np.argmin(sh['load']))
                        sh['load'][r] += u.nbytes
                        with cp.cuda.Device(sh['devs'][r]):
                            self.qab[a] = cp.asarray(u)
                    else:
                        self.qab[a] = u
                    del u
                    off += na
            del j3c
        self.J2c = cp.asarray(auxmol.intor('int2c2e'))
        self.nbytes = dict(ia=self.qia.nbytes, ij=self.qij.nbytes,
                           ab=sum(q.nbytes for q in self.qab if q is not None))
        self.ribf_atom = ribf_atom
        self.atom_ribfs = atom_ribfs
        self._chol = {}

    # ------------------------------------------------------------------
    def aux_index(self, atoms):
        return np.concatenate([self.atom_ribfs[a] for a in atoms]).astype(np.int64)

    def metric_chol(self, atoms):
        """Cholesky factor of the fitting metric restricted to ``atoms``."""
        key = tuple(int(a) for a in atoms)
        L = self._chol.get(key)
        if L is None:
            idx = cp.asarray(self.aux_index(atoms))
            A = self.J2c[idx[:, None], idx[None, :]]
            L = cp.linalg.cholesky(A)
            if len(self._chol) > 4096:
                self._chol.clear()
            self._chol[key] = L
        return L

    # ------------------------------------------------------------------
    def qab_get(self, a):
        """device (Q_a|uv) block of aux atom ``a`` (None if empty).  Host
        mode: served from an LRU device cache of ``cache_budget()`` bytes."""
        q = self.qab[a]
        dev = cp.cuda.Device().id
        if q is None or (not self.qab_host and q.device.id == dev):
            return q
        # host blocks, or device blocks of another GPU (peer copy)
        from electrolites.dlpno.mgpu import lock
        with lock():
            cache = self._cache.setdefault(dev, OrderedDict())
            d = cache.get(a)
            if d is not None:
                cache.move_to_end(a)
                self.cache_stats['hit'] += 1
                return d
            budget = self.cache_budget()
            nb = self._cache_bytes.get(dev, 0)
            while cache and nb + q.nbytes > budget:
                _, old = cache.popitem(last=False)
                nb -= old.nbytes
            self.cache_stats['miss'] += 1
            self.cache_stats['bytes'] += q.nbytes
        d = cp.asarray(q)
        with lock():
            cache[a] = d
            self._cache_bytes[dev] = nb + d.nbytes
        return d

    def cache_budget(self):
        import os
        env = os.environ.get('DLPNO_QAB_CACHE_GB')
        if env is not None:
            return float(env) * 1e9
        # grow while a quarter of the device stays free (whole tensor resident
        # when it fits), never below a quarter of the device
        free, tot = cp.cuda.runtime.memGetInfo()
        free += cp.get_default_memory_pool().free_bytes()
        mine = self._cache_bytes.get(cp.cuda.Device().id, 0)
        return max(0.25 * tot, min(0.7 * tot, free + mine - 0.25 * tot))

    def spatial_order(self, atom_sets):
        """order of items (given by their aux-atom sets) along a Morton curve
        of their centres: consecutive chunks then share most aux atoms (and
        their cached (Q|uv) blocks)."""
        xyz = np.asarray(self.mol.atom_coords())
        c = np.array([xyz[np.asarray(a, np.int64)].mean(axis=0) if len(a) else np.zeros(3)
                      for a in atom_sets])
        lo, hi = c.min(axis=0), c.max(axis=0)
        q = ((c - lo) / np.maximum(hi - lo, 1e-9) * 1023).astype(np.int64)
        code = np.zeros(len(c), np.int64)
        for bit in range(10):
            for d in range(3):
                code |= ((q[:, d] >> bit) & 1) << (3 * bit + d)
        return np.argsort(code, kind='stable')

    def cache_clear(self):
        self._cache.clear()
        self._cache_bytes.clear()

    def q_lmo_pao(self, aux, i, paos):
        """(Q|i u) for Q in aux (global index array), u in paos."""
        return self.qia[cp.asarray(aux)[:, None], int(i), cp.asarray(paos)[None, :]]

    def q_lmo_lmo(self, aux, i, lmos):
        return self.qij[cp.asarray(aux)[:, None], int(i), cp.asarray(lmos)[None, :]]

    def q_vv(self, atoms, paos1, X1, paos2=None, X2=None):
        """sum_{uv} X1[u,a] (Q|uv) X2[v,b] for Q on ``atoms`` (in order),
        u in paos1, v in paos2 (defaults: the same).  -> (naux, n1, n2)."""
        if paos2 is None:
            paos2, X2 = paos1, X1
        out = []
        ext = self.ext
        for a in atoms:
            q = self.qab_get(a)
            if q is None:
                continue
            p1 = ext.pao_pos[a, paos1]
            p2 = ext.pao_pos[a, paos2]
            assert (p1 >= 0).all() and (p2 >= 0).all(), 'PAO outside the extended domain'
            blk = q[:, cp.asarray(p1)[:, None], cp.asarray(p2)[None, :]]
            out.append(cp.matmul(cp.matmul(X1.T[None], blk), X2[None]))
        return cp.concatenate(out, axis=0)
