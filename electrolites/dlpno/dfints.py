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


def sub_auxmol(auxmol, atoms, shells=None):
    """auxmol restricted to the shells on ``atoms`` (AO order preserved), or
    to the given shell indices."""
    sub = auxmol.copy(deep=False)
    if shells is not None:
        sub._bas = auxmol._bas[np.asarray(shells, np.int64)]
        return sub
    sel = np.isin(auxmol._bas[:, gto.ATOM_OF], atoms)
    sub._bas = auxmol._bas[sel]
    return sub


def sub_mol_shells(mol, shells):
    """mol restricted to the AO shells ``shells`` (order preserved) and the
    AO indices they span."""
    sub = mol.copy(deep=False)
    shells = np.asarray(shells, np.int64)
    sub._bas = mol._bas[shells]
    loc = mol.ao_loc_nr()
    ao = np.concatenate([np.arange(loc[i], loc[i + 1]) for i in shells]) if len(shells) else np.zeros(0, np.int64)
    return sub, ao


_QCOND = {}


def _memo_qcond():
    """GPU4PySCF's Int3c2eOpt.build() recomputes the AO shell-pair Schwarz
    screening of the whole molecule for every call, although it depends only
    on the AO basis: memoised per (molecule, device), so that evaluating the
    integrals of one aux atom at a time costs only the integrals."""
    m = int3c2e_bdiv
    if getattr(m, '_dlpno_qcond_memo', False):
        return
    orig = m._cache_q_cond_and_non0pairs

    def memo(mol, rys_envs, precision=1e-14, tile=4, tril=True):
        key = (mol.natm, mol.nbas, round(float(np.asarray(mol.atom_coords()).sum()), 8),
               hash(np.asarray(mol._bas).tobytes()), precision, tile, tril, cp.cuda.Device().id)
        r = _QCOND.get(key)
        if r is None:
            if len(_QCOND) > 64:              # sub-molecules of the AO-screened regions
                _QCOND.pop(next(iter(_QCOND)))
            r = _QCOND[key] = orig(mol, rys_envs, precision, tile, tril)
        return r
    m._cache_q_cond_and_non0pairs = memo
    m._dlpno_qcond_memo = True


def aux_e2_atoms(mol, auxmol, atoms):
    """(mu nu|Q) for the aux functions on ``atoms``, as (nao, nao, naux_sub)."""
    _memo_qcond()
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


_BQ_GATHER = None


def _bq_kernel():
    global _BQ_GATHER
    if _BQ_GATHER is None:
        _BQ_GATHER = cp.ElementwiseKernel(
            'int64 qq, int64 ii, int64 uu, uint64 bufp, raw int32 atom_of, raw int64 aoff, '
            'raw int32 a0, raw int32 nl, raw int32 npo, raw int32 lpos, raw int32 ppos, int32 n, int32 nbf',
            'float64 y',
            '''
            const int a = atom_of[qq];
            const int li = lpos[(long long) a * n + ii];
            const int pu = ppos[(long long) a * nbf + uu];
            y = (li < 0 || pu < 0) ? 0.0
                : ((const double*) bufp)[aoff[a] + ((long long) (qq - a0[a]) * nl[a] + li) * npo[a] + pu];
            ''', 'dlpno_bq_gather')
    return _BQ_GATHER


class BlockQ:
    """(Q|i u) stored per auxiliary atom A over A's extended LMO and PAO
    domains (the domains every pair / triplet using A lies in) instead of
    densely over all (Q, i, u): linear in the system size.  ``bq[Q, i, u]``
    with broadcastable integer index arrays gathers in one kernel (entries
    outside the domains are 0)."""

    def __init__(self, naux, n, nbf, ribf_atom, atom_ribfs, ext_lmos, ext_paos, buf=None, host=False):
        natm = len(atom_ribfs)
        self.shape = (naux, n, nbf)
        self.ndim = 3
        self.n, self.nbf = n, nbf
        self.lmos, self.paos = ext_lmos, ext_paos
        self.nl = np.array([len(ext_lmos[a]) for a in range(natm)], np.int32)
        self.npo = np.array([len(ext_paos[a]) for a in range(natm)], np.int32)
        self.na = np.array([len(atom_ribfs[a]) for a in range(natm)], np.int64)
        self.a0 = np.array([atom_ribfs[a][0] if len(atom_ribfs[a]) else 0 for a in range(natm)], np.int32)
        sz = self.na * self.nl * self.npo
        self.aoff = np.concatenate([[0], np.cumsum(sz)[:-1]]).astype(np.int64)
        self.size = int(sz.sum())
        self.nbytes = 8 * self.size
        lpos = -np.ones((natm, n), np.int32)
        ppos = -np.ones((natm, nbf), np.int32)
        for a in range(natm):
            lpos[a, ext_lmos[a]] = np.arange(len(ext_lmos[a]))
            ppos[a, ext_paos[a]] = np.arange(len(ext_paos[a]))
        self._host = dict(atom_of=np.asarray(ribf_atom, np.int32), aoff=self.aoff, a0=self.a0,
                          nl=self.nl, npo=self.npo, lpos=lpos.ravel(), ppos=ppos.ravel())
        if buf is None:
            if host:
                # pinned host memory, read by the gather kernel directly (zero copy)
                from electrolites.dlpno.devmem import pinned_empty
                buf = pinned_empty((max(self.size, 1),))
                buf[...] = 0.
            else:
                buf = cp.zeros(max(self.size, 1))
        self.buf = buf
        self._tabs = None

    def block(self, a):
        """the stored (na, nl, np) block of aux atom ``a`` (a view)."""
        o, s = int(self.aoff[a]), int(self.na[a] * self.nl[a] * self.npo[a])
        return self.buf[o:o + s].reshape(int(self.na[a]), int(self.nl[a]), int(self.npo[a]))

    @property
    def host(self):
        return isinstance(self.buf, np.ndarray)

    def set_block(self, a, x):
        """store the (na, nl, np) device block of aux atom ``a``."""
        if self.host:
            self.block(a)[...] = cp.asnumpy(x)
        else:
            self.block(a)[...] = x

    def _bufp(self):
        if not self.host:
            return np.uint64(self.buf.data.ptr)
        p = getattr(self, '_hostp', None)
        if p is None or p[0] != self.buf.ctypes.data:
            at = cp.cuda.runtime.pointerGetAttributes(self.buf.ctypes.data)
            p = self._hostp = (self.buf.ctypes.data, np.uint64(at.devicePointer))
        return p[1]

    def _dev_tabs(self):
        dev = cp.cuda.Device().id
        if self._tabs is None or self._tabs[0] != dev:
            self._tabs = (dev, {k: cp.asarray(v) for k, v in self._host.items()})
        return self._tabs[1]

    def __getitem__(self, key):
        Q, i, u = (cp.asarray(k, dtype=cp.int64) for k in key)
        t = self._dev_tabs()
        return _bq_kernel()(Q, i, u, self._bufp(), t['atom_of'], t['aoff'], t['a0'], t['nl'], t['npo'],
                            t['lpos'], t['ppos'], np.int32(self.n), np.int32(self.nbf))

    def _with(self, buf):
        c = object.__new__(BlockQ)
        c.__dict__.update(self.__dict__)
        c.buf, c._tabs = buf, None
        return c

    def replicate(self):
        """a view whose storage is on the current device (host storage: shared)."""
        if self.host or self.buf.device.id == cp.cuda.Device().id:
            return self
        return self._with(cp.asarray(self.buf))

    def to_host(self):
        return self._with(cp.asnumpy(self.buf) if isinstance(self.buf, cp.ndarray) else self.buf)

    def to_device(self):
        return self if self.host else self._with(cp.asarray(self.buf))


def _host_available():
    """MemAvailable of the node (bytes)."""
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return float(line.split()[1]) * 1024.
    except OSError:
        pass
    return 0.


def _decide_qab_direct(nbytes, host):
    """evaluate (Q|uv) per use (no stored blocks) when the blocks go to the
    host and would take more than 40% of its available memory
    (DLPNO_QAB_DIRECT=auto|0|1)."""
    env = os.environ.get('DLPNO_QAB_DIRECT', 'auto')
    if env != 'auto':
        return env == '1'
    return bool(host) and nbytes > 0.4 * _host_available()


class _CheckedQ:
    """debug: a BlockQ whose gathers are compared with the dense tensor."""

    def __init__(self, bq, dense):
        self.bq, self.dense = bq, dense
        self.shape, self.ndim, self.nbytes = bq.shape, 3, bq.nbytes
        self.host = False

    def __getitem__(self, key):
        a = self.bq[key]
        k = tuple(cp.asarray(x, dtype=cp.int64) for x in key)
        b = self.dense[k]
        # padded slots use index 0 (masked by the callers): ignored here
        valid = (k[0] != 0) & (k[1] != 0) & (k[2] != 0)
        d = float(cp.abs((a - b) * valid).max()) if a.size else 0.
        if a.size and os.environ.get('DLPNO_DEBUG_QIJ') == '2':
            e = cp.abs(a - b)
            parts = []
            for nm, cond in (('Q=0', (k[0] == 0) & (k[1] != 0) & (k[2] != 0)),
                             ('i=0', (k[1] == 0) & (k[0] != 0) & (k[2] != 0)),
                             ('j=0', (k[2] == 0) & (k[0] != 0) & (k[1] != 0))):
                c = cp.broadcast_to(cond, e.shape)
                parts.append((nm, float((e * c).max())))
            if any(v > 1e-12 for _, v in parts):
                print('[debug qij idx0]', parts, flush=True)
        if d > 1e-12:
            import traceback
            print(f'[debug qij] max |sparse - dense| = {d:.2e}; shape {a.shape}', flush=True)
            traceback.print_stack(limit=6)
        return b

    def replicate(self):
        return self


def _decide_qia_host(nbytes):
    """block-sparse (Q|i u) in pinned host memory (read zero-copy by the
    gathers) when it would take more than a quarter of the free device
    memory (DLPNO_QIA_HOST=auto|0|1)."""
    env = os.environ.get('DLPNO_QIA_HOST', 'auto')
    if env != 'auto':
        return env == '1'
    free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
    return nbytes > 0.25 * free


def _decide_qia_sparse(nbytes):
    """block-sparse (Q|i u) when the dense tensor would take more than a
    quarter of the free device memory (DLPNO_QIA_SPARSE=auto|0|1)."""
    env = os.environ.get('DLPNO_QIA_SPARSE', 'auto')
    if env != 'auto':
        return env == '1'
    free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
    return nbytes > 0.25 * free


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
                 ribf_atom, atom_ribfs, group_mem=2 << 30, need_ab=True, sparse_ij=True):
        self.mol, self.auxmol = mol, auxmol
        nao = mol.nao
        naux = auxmol.nao
        n = C_lmo.shape[1]
        nbf = C_pao.shape[1]
        self.naux, self.n, self.nbf = naux, n, nbf
        L = cp.asarray(C_lmo)
        P = cp.asarray(C_pao)
        self.qia_sparse = _decide_qia_sparse(8. * naux * n * nbf)
        if self.qia_sparse:
            nl_ = np.array([len(ext.lmos[a]) for a in range(mol.natm)])
            np_ = np.array([len(ext.paos[a]) for a in range(mol.natm)])
            na_ = np.array([len(atom_ribfs[a]) for a in range(mol.natm)])
            self.qia = BlockQ(naux, n, nbf, ribf_atom, atom_ribfs, ext.lmos, ext.paos,
                              host=_decide_qia_host(8. * float(np.sum(na_ * nl_ * np_))))
        else:
            self.qia = cp.empty((naux, n, nbf))
        # (Q|i j) block-sparse over the aux atoms' extended LMO domains when the
        # dense tensor would take more than 10% of the free device memory
        # (closed shell; exact: every gather is checked against the dense
        # tensor with DLPNO_DEBUG_QIJ=1; the open-shell code also reads SOMO
        # columns for every aux atom and uses the complete host tensor)
        env = os.environ.get('DLPNO_QIJ_SPARSE', 'auto')
        if env == 'auto':
            free_ = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
            self.qij_sparse = sparse_ij and 8. * naux * n * n > 0.1 * free_
        else:
            self.qij_sparse = env == '1'
        # exact alternative: the complete (Q|i j) in pinned host memory, read
        # zero-copy by the gathers (DLPNO_QIJ_HOST=auto: when the dense tensor
        # would take more than 10% of the free device memory)
        envh = os.environ.get('DLPNO_QIJ_HOST', 'auto')
        if envh == 'auto':
            free_ = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
            qij_host = 8. * naux * n * n > 0.1 * free_
        else:
            qij_host = envh == '1'
        self._qij_lmos = None
        if self.qij_sparse:
            self.qij = BlockQ(naux, n, n, ribf_atom, atom_ribfs, ext.lmos, ext.lmos)
        elif qij_host:
            allm = np.arange(n, dtype=np.int64)
            self._qij_lmos = [allm] * mol.natm
            self.qij = BlockQ(naux, n, n, ribf_atom, atom_ribfs, self._qij_lmos, self._qij_lmos, host=True)
            self.qij_sparse = True          # filled block by block below
        else:
            self.qij = cp.empty((naux, n, n))
        self.qab = [None] * mol.natm
        self.ext = ext
        mask_uv = cp.asarray(doi_uv > t_cut_do_uv) if need_ab else None
        # the PAO coefficients and the DOI mask: kept for the direct (Q|uv)
        self.C_pao_d, self.mask_uv_d = P, mask_uv
        # AO screening of the direct (Q|uv): AOs carrying |C_pao| above
        # DLPNO_AO_CUT x (largest |C| of that PAO); the region integrals are
        # evaluated only over the shells of the union of its PAOs' supports
        self.ao_cut = float(os.environ.get('DLPNO_AO_CUT', '1e-8'))
        self._pao_sup = None
        if self.ao_cut > 0:
            Cp = np.abs(np.asarray(C_pao))
            self._pao_sup = Cp > self.ao_cut * Cp.max(axis=0)[None, :]      # (nao, nbf)
            loc = mol.ao_loc_nr()
            self._ao_shell = np.repeat(np.arange(mol.nbas), np.diff(loc))
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
        # blocks that would not fit the host either: evaluated per use from the
        # AO three-index integrals of the aux atom (``qab_block``)
        self.qab_direct = need_ab and _decide_qab_direct(nab, self.qab_host)
        if self.qab_direct:
            self.qab_host = False
            self._shard = None
        self._ao_cache, self._ao_bytes = {}, {}
        self._regions = {}             # device -> region blocks (direct mode)
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
        # AO screening of the DF build: per aux atom, (mu nu|Q_a) only over the
        # shells carrying its extended LMOs and PAOs (relative cut ao_cut);
        # linear for extended molecules.  Used when those supports are local
        # (mean below 60% of the AOs; DLPNO_DF_SCREEN=auto|0|1); compact
        # molecules keep the grouped full-AO evaluation (fewer, larger calls).
        env_s = os.environ.get('DLPNO_DF_SCREEN', 'auto')
        sups = None
        if self._pao_sup is not None and env_s != '0':
            Lh = np.abs(np.asarray(C_lmo))
            lmo_sup = Lh > self.ao_cut * Lh.max(axis=0)[None, :]
            lm_sets = self._qij_lmos if self._qij_lmos is not None else ext.lmos
            sups = []
            for a in range(natm):
                la, pa = lm_sets[a], ext.paos[a]
                sup = lmo_sup[:, la].any(axis=1) if len(la) else np.zeros(nao, bool)
                if len(ext.lmos[a]):
                    sup |= lmo_sup[:, ext.lmos[a]].any(axis=1)
                if len(pa):
                    sup |= self._pao_sup[:, pa].any(axis=1)
                sups.append(sup)
            # consecutive aux atoms grouped while the union of their supports
            # grows by less than 30% over the largest member and the group's
            # (nd, nd, nq) block stays within group_mem: one integral call each
            sgroups, cur, cu, cq = [], [], None, 0
            for a in range(natm):
                na_ = len(atom_ribfs[a])
                if cur:
                    un = cu | sups[a]
                    big = max(int(sups[b_].sum()) for b_ in cur + [a])
                    if (un.sum() > 1.3 * big or 8. * un.sum() ** 2 * (cq + na_) > group_mem):
                        sgroups.append((cur, cu))
                        cur, cu, cq = [], None, 0
                if not cur:
                    cur, cu, cq = [a], sups[a].copy(), na_
                else:
                    cur.append(a)
                    cu = cu | sups[a]
                    cq += na_
            if cur:
                sgroups.append((cur, cu))
            cost_s = sum(8. * float(u.sum()) ** 2 for _, u in sgroups)
            cost_f = 8. * nao ** 2 * len(groups)
            frac = float(np.mean([u.sum() for _, u in sgroups])) / nao
            self.df_screen_frac = frac
            if env_s == 'auto' and cost_s > 0.7 * cost_f:
                sups = None
        if sups is not None:
            if not self.qia_sparse:
                self.qia = None
                self.qia = cp.zeros((naux, n, nbf))
            if not self.qij_sparse:
                self.qij = None
                self.qij = cp.zeros((naux, n, n))
            for g_atoms, g_sup in sgroups:
              g_atoms = [a for a in g_atoms if len(atom_ribfs[a])]
              if not g_atoms:
                  continue
              shells = np.unique(self._ao_shell[g_sup])
              smol, ao = sub_mol_shells(mol, shells)
              nd = len(ao)
              Jg = cp.ascontiguousarray(aux_e2_atoms(smol, auxmol, g_atoms))       # (nd, nd, nq)
              gao = cp.asarray(ao)
              goff = 0
              for a in g_atoms:
                na = len(atom_ribfs[a])
                J = cp.ascontiguousarray(Jg[:, :, goff:goff + na])
                goff += na
                la, pa = ext.lmos[a], ext.paos[a]
                lj = (self._qij_lmos[a] if self._qij_lmos is not None else la)
                if na == 0 or (len(la) == 0 and len(lj) == 0):
                    continue
                ga_ = cp.asarray(atom_ribfs[a])
                gla = cp.asarray(la)
                Lr = L[gao]
                if len(la):
                    Ls = Lr[:, gla]
                    t = cp.matmul(Ls.T, J.reshape(nd, nd * na)).reshape(len(la), nd, na).transpose(2, 0, 1)
                    if len(pa):
                        q = cp.matmul(t, P[gao][:, cp.asarray(pa)])            # (na, nl, np)
                        if self.qia_sparse:
                            self.qia.set_block(a, q)
                        else:
                            self.qia[ga_[:, None, None], gla[None, :, None], cp.asarray(pa)[None, None, :]] = q
                        del q
                    if lj is la:
                        q = cp.matmul(t, Ls)
                    else:
                        q = None
                    del t
                else:
                    q = None
                if q is None and len(lj):
                    glj = cp.asarray(lj)
                    Lj = Lr[:, glj]
                    tj = cp.matmul(Lj.T, J.reshape(nd, nd * na)).reshape(len(lj), nd, na).transpose(2, 0, 1)
                    q = cp.matmul(tj, Lj)
                    del tj
                if q is not None:
                    glj = cp.asarray(lj)
                    if self.qij_sparse:
                        self.qij.set_block(a, q)
                    else:
                        self.qij[ga_[:, None, None], glj[None, :, None], glj[None, None, :]] = q
                    del q
                if need_ab and not self.qab_direct and len(pa):
                    gpa = cp.asarray(pa)
                    Pa = P[gao][:, gpa]
                    u = cp.matmul(Pa.T, J.reshape(nd, nd * na)).reshape(len(pa), nd, na)
                    u = cp.matmul(u.transpose(2, 0, 1), Pa)                    # (na, npa, npa)
                    u *= mask_uv[gpa[:, None], gpa[None, :]][None]
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
                del J
              del Jg
            groups = []
            # one sub-molecule per aux atom: their screening tables are not reused
            _QCOND.clear()
            cp.get_default_memory_pool().free_all_blocks()
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
            if self.qia_sparse:
                off = 0
                for a in atoms:
                    na = len(atom_ribfs[a])
                    la, pa = ext.lmos[a], ext.paos[a]
                    if na and len(la) and len(pa):
                        tq = t[off:off + na][:, cp.asarray(la)]                  # (na, nl, nao)
                        self.qia.set_block(a, cp.matmul(tq, P[:, cp.asarray(pa)]))
                    off += na
            else:
                self.qia[gi] = cp.matmul(t, P)
            if self.qij_sparse:
                off = 0
                lm_sets = self._qij_lmos if self._qij_lmos is not None else ext.lmos
                for a in atoms:
                    na = len(atom_ribfs[a])
                    la = lm_sets[a]
                    if na and len(la):
                        gla = cp.asarray(la)
                        self.qij.set_block(a, cp.matmul(t[off:off + na][:, gla], L[:, gla]))
                    off += na
            else:
                self.qij[gi] = cp.matmul(t, L)
            del t
            if need_ab and not self.qab_direct:
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
        if os.environ.get('DLPNO_DEBUG_QIJ') and isinstance(self.qij, BlockQ) and not self.qij.host:
            # debug: every gather of the sparse (Q|i j) checked against the dense tensor
            dense = cp.empty((naux, n, n))
            for atoms in groups:
                idx = np.concatenate([atom_ribfs[a] for a in atoms])
                if len(idx) == 0:
                    continue
                j3c = aux_e2_atoms(mol, auxmol, atoms)
                nq = j3c.shape[2]
                t = cp.matmul(L.T, j3c.reshape(nao, nao * nq)).reshape(n, nao, nq).transpose(2, 0, 1)
                dense[cp.asarray(idx)] = cp.matmul(t, L)
                del j3c, t
            self.qij = _CheckedQ(self.qij, dense)
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
    def qab_block(self, a, need):
        """(Q_a|uv) block of aux atom ``a`` and the map PAO -> position in it
        (-1: absent).  Stored blocks: the extended-domain block.  Direct mode:
        evaluated over the PAOs ``need`` only (an iterable of index arrays,
        their union), from the AO integrals of ``a``.  (None, None) if empty."""
        if not self.qab_direct:
            return self.qab_get(a), self.ext.pao_pos[a]
        reg = getattr(self, '_regions', None)
        reg = reg.get(cp.cuda.Device().id) if reg else None
        if reg is not None and a in reg:
            blk, pmap = reg[a]
            return self._region_dev(a, blk), pmap
        return self._direct_block(a, need)

    def _direct_block(self, a, need, cache=True, aosub=None):
        na = len(self.atom_ribfs[a])
        need = [np.asarray(x, np.int64).ravel() for x in need]
        need = np.unique(np.concatenate(need)) if need else np.zeros(0, np.int64)
        if na == 0 or len(need) == 0:
            return None, None
        import time
        t0 = time.perf_counter()
        gn = cp.asarray(need)
        if aosub is not None:
            # AO-screened: (mu nu|Q_a) over the region's AO support only
            smol, ao = aosub
            Pn = self.C_pao_d[cp.asarray(ao)][:, gn]
            nd, m = len(ao), len(need)
            free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
            per_q = 8. * (nd * nd + m * nd + m * m)            # J column + both transforms
            lim = min(1.5e9, 0.1 * free)
            if os.environ.get('DLPNO_AUXCHUNK_GB'):           # testing
                lim = float(os.environ['DLPNO_AUXCHUNK_GB']) * 1e9
            if na * per_q > lim:
                # large AO domains (triple zeta): the atom's aux shells in chunks
                _memo_qcond()
                shl = np.flatnonzero(self.auxmol._bas[:, gto.ATOM_OF] == a)
                aloc = self.auxmol.ao_loc_nr()
                out = cp.empty((na, m, m))
                q0, cur = 0, []
                chunks = []
                cap = max(1, int(lim // per_q))
                for sh in shl:
                    w = aloc[sh + 1] - aloc[sh]
                    if cur and sum(aloc[c + 1] - aloc[c] for c in cur) + w > cap:
                        chunks.append(cur)
                        cur = []
                    cur.append(sh)
                if cur:
                    chunks.append(cur)
                for ch in chunks:
                    Jc = cp.ascontiguousarray(int3c2e_bdiv.aux_e2(smol, sub_auxmol(self.auxmol, None, ch)))
                    nc = Jc.shape[2]
                    t1 = time.perf_counter()
                    uc = cp.matmul(Pn.T, Jc.reshape(nd, nd * nc)).reshape(m, nd, nc)
                    del Jc
                    out[q0:q0 + nc] = cp.matmul(uc.transpose(2, 0, 1), Pn)
                    del uc
                    q0 += nc
                assert q0 == na
                out *= self.mask_uv_d[gn[:, None], gn[None, :]][None]
                pmap = -np.ones(self.nbf, np.int64)
                pmap[need] = np.arange(len(need))
                st = self.cache_stats
                st['direct'] = st.get('direct', 0) + 1
                st['t_ao'] = st.get('t_ao', 0.) + (time.perf_counter() - t0)
                return out, pmap
            J = cp.ascontiguousarray(aux_e2_atoms(smol, self.auxmol, [a]))
        else:
            J = self._ao_j3c(a) if cache else cp.ascontiguousarray(aux_e2_atoms(self.mol, self.auxmol, [a]))
            Pn = self.C_pao_d[:, gn]
        nao = J.shape[0]
        t1 = time.perf_counter()
        u = cp.matmul(Pn.T, J.reshape(nao, nao * na)).reshape(len(need), nao, na)
        u = cp.matmul(u.transpose(2, 0, 1), Pn)                     # (na, m, m)
        u *= self.mask_uv_d[gn[:, None], gn[None, :]][None]
        pmap = -np.ones(self.nbf, np.int64)
        pmap[need] = np.arange(len(need))
        st = self.cache_stats
        st['direct'] = st.get('direct', 0) + 1
        st['direct_m'] = st.get('direct_m', 0) + len(need)
        if os.environ.get('DLPNO_PROF') == '1':
            cp.cuda.Device().synchronize()
        st['t_ao'] = st.get('t_ao', 0.) + (t1 - t0)
        st['t_tr'] = st.get('t_tr', 0.) + (time.perf_counter() - t1)
        return u, pmap

    # ---- regions: direct (Q|uv) evaluated once per atom for a group of batches
    def plan_regions(self, item_needs, budget):
        """split items (in order) into regions: consecutive items whose
        (Q_a|uv) blocks over the union of their PAOs per atom stay within
        ``budget`` bytes.  item_needs[k]: {atom: PAO index array}.  Returns
        [(item indices, {atom: PAO indices})]."""
        natm, nbf = len(self.atom_ribfs), self.nbf
        na = np.array([len(self.atom_ribfs[a]) for a in range(natm)], np.float64)
        regions = []

        def fresh():
            return [], np.zeros((natm, nbf), bool), np.zeros(natm, np.int64), 0.

        def grow(mask, nd):
            add = {}
            for a, u in nd.items():
                m = add[a] if a in add else mask[a].copy()
                m[np.asarray(u, np.int64)] = True
                add[a] = m
            return add
        cur, mask, cnt, tot = fresh()
        for k, nd in enumerate(item_needs):
            add = grow(mask, nd)
            new = tot + sum(8. * na[a] * (int(m.sum()) ** 2 - int(cnt[a]) ** 2) for a, m in add.items())
            if cur and new > budget:
                regions.append((cur, {a: np.flatnonzero(mask[a]) for a in range(natm) if cnt[a]}))
                cur, mask, cnt, tot = fresh()
                add = grow(mask, nd)
                new = sum(8. * na[a] * int(m.sum()) ** 2 for a, m in add.items())
            for a, m in add.items():
                mask[a] = m
                cnt[a] = int(m.sum())
            tot = new
            cur.append(k)
        if cur:
            regions.append((cur, {a: np.flatnonzero(mask[a]) for a in range(natm) if cnt[a]}))
        return regions

    def region_begin(self, needs, verbose=0):
        """(Q_a|uv) over needs[a] for every atom a, on the current device, each
        from one evaluation of its AO integrals; kept on the device when they
        fit (30% of its free memory), else in pinned host memory behind the
        device LRU cache."""
        import time
        t0 = time.perf_counter()
        dev = cp.cuda.Device().id
        self._dev_clear(dev)
        na = {a: len(self.atom_ribfs[a]) for a in needs}
        tot = sum(8. * na[a] * len(u) ** 2 for a, u in needs.items())
        free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        on_dev = tot < 0.3 * free
        aosub = None
        if self._pao_sup is not None and needs:
            allp = np.unique(np.concatenate([np.asarray(u, np.int64) for u in needs.values()]))
            ao_used = self._pao_sup[:, allp].any(axis=1)
            shells = np.unique(self._ao_shell[ao_used])
            if len(shells) < self.mol.nbas:
                aosub = sub_mol_shells(self.mol, shells)
        reg = {}
        for a in sorted(needs):
            if na[a] == 0 or len(needs[a]) == 0:
                continue
            blk, pmap = self._direct_block(a, [needs[a]], cache=False, aosub=aosub)
            reg[a] = (blk if on_dev else to_pinned(blk), pmap)
            del blk
        self._regions[dev] = reg
        if verbose >= 2:
            nd = len(aosub[1]) if aosub is not None else self.mol.nao
            print(f'[mem] (Q|uv) region on GPU {dev}: {len(reg)} atoms, {tot / 1e9:.1f} GB on the '
                  f'{"device" if on_dev else "host"}, {nd} of {self.mol.nao} AOs, '
                  f'built in {time.perf_counter() - t0:.1f} s', flush=True)

    def region_end(self):
        dev = cp.cuda.Device().id
        self._regions.pop(dev, None)
        self._dev_clear(dev)
        import gc
        gc.collect()
        # pinned blocks back to the OS (the pinned pool would keep them, and the
        # next region's blocks have other sizes); this device's pool too
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.get_default_memory_pool().free_all_blocks()

    def _dev_clear(self, dev):
        from electrolites.dlpno.mgpu import lock
        with lock():
            for d in (self._cache, self._cache_bytes, self._ao_cache, self._ao_bytes):
                d.pop(dev, None)

    def _region_dev(self, a, blk):
        """a region block on the current device (LRU cache for host / peer blocks)."""
        dev = cp.cuda.Device().id
        if isinstance(blk, cp.ndarray) and blk.device.id == dev:
            return blk
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
            while cache and nb + blk.nbytes > budget:
                _, old = cache.popitem(last=False)
                nb -= old.nbytes
            self.cache_stats['miss'] += 1
            self.cache_stats['bytes'] += blk.nbytes
        d = cp.asarray(blk)
        with lock():
            cache[a] = d
            self._cache_bytes[dev] = nb + d.nbytes
        return d

    def run_regions(self, items, item_needs, work, setup, devs, verbose=0):
        """work(ctx, item) for every item, the items grouped into regions
        (direct (Q|uv)); the regions go to the devices through a work queue
        (``setup(dev)`` -> ctx), each device building its region's blocks and
        running its items.  Region budget: a quarter of the smallest free
        device memory (DLPNO_REGION_GB overrides)."""
        from electrolites.dlpno import mgpu
        free = []
        for d in devs:
            with cp.cuda.Device(d):
                cp.get_default_memory_pool().free_all_blocks()
                free.append(cp.cuda.runtime.memGetInfo()[0])
        env = os.environ.get('DLPNO_REGION_GB')
        budget = float(env) * 1e9 if env else 0.25 * min(free)
        regions = self.plan_regions(item_needs, budget)
        if verbose >= 2:
            print(f'[mem] (Q|uv) regions: {len(regions)} for {len(items)} batches '
                  f'(budget {budget / 1e9:.1f} GB per region)', flush=True)
        self._regions = {}

        def rwork(ctx, region):
            idx, needs = region
            dv = getattr(ctx, 'dfi', None) or self        # this device's view
            dv.region_begin(needs, verbose)
            try:
                for k in idx:
                    work(ctx, items[k])
            finally:
                dv.region_end()
        try:
            return mgpu.run_queue(regions, rwork, devs, setup=setup)
        finally:
            self._regions = {}

    def _ao_j3c(self, a):
        """AO (mu nu|Q_a) of aux atom ``a`` (nao, nao, na), from a per-device
        LRU cache of ``cache_budget()`` bytes."""
        from electrolites.dlpno.mgpu import lock
        dev = cp.cuda.Device().id
        with lock():
            cache = self._ao_cache.setdefault(dev, OrderedDict())
            J = cache.get(a)
            if J is not None:
                cache.move_to_end(a)
                self.cache_stats['hit'] += 1
                return J
        J = cp.ascontiguousarray(aux_e2_atoms(self.mol, self.auxmol, [a]))
        with lock():
            self.cache_stats['miss'] += 1
            budget = self.cache_budget()
            nb = self._ao_bytes.get(dev, 0)
            while cache and nb + J.nbytes > budget:
                _, old = cache.popitem(last=False)
                nb -= old.nbytes
            if J.nbytes <= budget:
                cache[a] = J
                nb += J.nbytes
            self._ao_bytes[dev] = nb
        return J

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
        # the floor never exceeds half of what is actually available (a busy
        # device: other stores, region temporaries)
        floor = min(0.2 * tot, 0.5 * (free + mine))
        return max(floor, min(0.55 * tot, free + mine - 0.35 * tot))

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
        self._ao_cache.clear()
        self._ao_bytes.clear()

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
            q, pmap = self.qab_block(a, [paos1, paos2])
            if q is None:
                continue
            p1 = pmap[np.asarray(paos1)]
            p2 = pmap[np.asarray(paos2)]
            assert (p1 >= 0).all() and (p2 >= 0).all(), 'PAO outside the extended domain'
            blk = q[:, cp.asarray(p1)[:, None], cp.asarray(p2)[None, :]]
            out.append(cp.matmul(cp.matmul(X1.T[None], blk), X2[None]))
        return cp.concatenate(out, axis=0)
