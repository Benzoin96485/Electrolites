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
            if len(_QCOND) > 16:              # sub-molecules of the AO-screened regions (used once)
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
            'int64 qq, int64 ii, int64 uu, raw uint64 abase, raw int32 atom_of, '
            'raw int32 a0, raw int32 nl, raw int32 npo, raw int32 lpos, raw int32 ppos, int32 n, int32 nbf',
            'float64 y',
            '''
            const int a = atom_of[qq];
            const int li = lpos[(long long) a * n + ii];
            const int pu = ppos[(long long) a * nbf + uu];
            y = (li < 0 || pu < 0) ? 0.0
                : ((const double*) abase[a])[((long long) (qq - a0[a]) * nl[a] + li) * npo[a] + pu];
            ''', 'dlpno_bq_gather')
    return _BQ_GATHER


class BlockQ:
    """(Q|i u) stored per auxiliary atom A over A's extended LMO and PAO
    domains (the domains every pair / triplet using A lies in) instead of
    densely over all (Q, i, u): linear in the system size.  ``bq[Q, i, u]``
    with broadcastable integer index arrays gathers in one kernel (entries
    outside the domains are 0).

    Storage: one device or pinned-host buffer, or (``placement`` =
    [(device id or None for host, bytes), ...]) the atoms' blocks spread over
    several devices' memory and the host, read by the gathers through
    per-atom pointers (peer access / zero copy)."""

    def __init__(self, naux, n, nbf, ribf_atom, atom_ribfs, ext_lmos, ext_paos, buf=None, host=False,
                 placement=None):
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
        self._host = dict(atom_of=np.asarray(ribf_atom, np.int32), a0=self.a0,
                          nl=self.nl, npo=self.npo, lpos=lpos.ravel(), ppos=ppos.ravel())
        self.bufs = None
        if placement is not None and buf is None:
            # the atoms in order into the listed stores, each up to its capacity
            from electrolites.dlpno.devmem import pinned_empty
            self.abuf = np.zeros(natm, np.int64)
            self.aoff = np.zeros(natm, np.int64)
            sizes, k, used = [], 0, 0
            for a in range(natm):
                while k < len(placement) - 1 and 8 * (used + sz[a]) > placement[k][1]:
                    sizes.append(used)
                    k, used = k + 1, 0
                self.abuf[a], self.aoff[a] = k, used
                used += int(sz[a])
            sizes.append(used)
            sizes += [0] * (len(placement) - len(sizes))
            self.bufs = []
            for (d, _), m in zip(placement, sizes):
                if d is None:
                    b = pinned_empty((max(m, 1),))
                    b[...] = 0.
                else:
                    with cp.cuda.Device(d):
                        b = cp.zeros(max(m, 1))
                self.bufs.append(b)
            self.buf = self.bufs[0]
            self.placed = [(d, 8. * m) for (d, _), m in zip(placement, sizes)]
            self._tabs = None
            return
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

    def _buf_of(self, a):
        return self.bufs[int(self.abuf[a])] if self.bufs is not None else self.buf

    def block(self, a):
        """the stored (na, nl, np) block of aux atom ``a`` (a view)."""
        o, s = int(self.aoff[a]), int(self.na[a] * self.nl[a] * self.npo[a])
        return self._buf_of(a)[o:o + s].reshape(int(self.na[a]), int(self.nl[a]), int(self.npo[a]))

    @property
    def host(self):
        """all storage in host memory"""
        if self.bufs is not None:
            return all(isinstance(b, np.ndarray) for b in self.bufs)
        return isinstance(self.buf, np.ndarray)

    @property
    def split(self):
        return self.bufs is not None

    def set_block(self, a, x):
        """store the (na, nl, np) device block of aux atom ``a``."""
        dst = self.block(a)
        if isinstance(dst, np.ndarray):
            cp.ascontiguousarray(x).get(out=dst)          # no pageable temporary
        elif dst.device.id == cp.cuda.Device().id:
            dst[...] = x
        else:
            x = cp.ascontiguousarray(x)
            cp.cuda.Device().synchronize()
            cp.cuda.runtime.memcpy(dst.data.ptr, x.data.ptr, x.nbytes, 4)     # UVA, peer

    @staticmethod
    def _devptr(b):
        if isinstance(b, cp.ndarray):
            return int(b.data.ptr)
        return int(cp.cuda.runtime.pointerGetAttributes(b.ctypes.data).devicePointer)

    def _dev_tabs(self):
        dev = cp.cuda.Device().id
        if self._tabs is None or self._tabs[0] != dev:
            t = {k: cp.asarray(v) for k, v in self._host.items()}
            if self.bufs is not None:
                base = np.array([self._devptr(b) for b in self.bufs], np.uint64)[self.abuf]
            else:
                base = np.full(len(self.aoff), self._devptr(self.buf), np.uint64)
            t['abase'] = cp.asarray(base + self.aoff.astype(np.uint64) * np.uint64(8))
            self._tabs = (dev, t)
        return self._tabs[1]

    def __getitem__(self, key):
        Q, i, u = (cp.asarray(k, dtype=cp.int64) for k in key)
        t = self._dev_tabs()
        return _bq_kernel()(Q, i, u, t['abase'], t['atom_of'], t['a0'], t['nl'], t['npo'],
                            t['lpos'], t['ppos'], np.int32(self.n), np.int32(self.nbf))

    def _with(self, buf):
        c = object.__new__(BlockQ)
        c.__dict__.update(self.__dict__)
        c.buf, c._tabs = buf, None
        return c

    def replicate(self):
        """a view whose storage is on the current device (host storage: shared;
        a large device buffer is shared too and read by peer access: a copy
        per device would not fit next to the rest, e.g. crambin's 21 GB (Q|ij))."""
        if self.split or self.host or self.buf.device.id == cp.cuda.Device().id:
            return self
        if self.buf.nbytes > float(os.environ.get('DLPNO_BQ_REPLICATE_GB', '4')) * 1e9:
            from electrolites.dlpno import mgpu
            mgpu.enable_peer(mgpu.devices())
            return self._with(self.buf)
        return self._with(cp.asarray(self.buf))

    def to_host(self):
        if self.split:
            return self
        return self._with(cp.asnumpy(self.buf) if isinstance(self.buf, cp.ndarray) else self.buf)

    def to_device(self):
        return self if (self.host or self.split) else self._with(cp.asarray(self.buf))


def _host_available():
    """MemAvailable of the node (bytes)."""
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return float(line.split()[1]) * 1024.
    except OSError:
        pass
    return 0.


def _decide_qia_direct(nbytes, host):
    """(Q|i u) evaluated with the direct (Q|uv) regions from the same AO
    integrals (no stored tensor) when the block-sparse tensor goes to the host
    and would take more than 40% of its available memory
    (DLPNO_QIA_DIRECT=auto|0|1)."""
    env = os.environ.get('DLPNO_QIA_DIRECT', 'auto')
    if env != 'auto':
        return env == '1'
    return bool(host) and nbytes > 0.4 * _host_available()


def _qia_placement(nbytes):
    """[(device or None = host, bytes)] holding ``nbytes`` of (Q|iu): the
    host up to 40% of its available memory, then the other devices and the
    home device (half share), each up to DLPNO_QIA_DEV_FRAC (0.45) of its
    free memory (DLPNO_QIA_DEV_GB: a fixed size, testing); None if that is
    not enough."""
    from electrolites.dlpno import mgpu
    home = cp.cuda.Device().id
    devs = mgpu.devices()
    frac = float(os.environ.get('DLPNO_QIA_DEV_FRAC', '0.45'))
    fixed = os.environ.get('DLPNO_QIA_DEV_GB')

    def cap(d, f):
        with cp.cuda.Device(d):
            fr = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        return float(fixed) * 1e9 if fixed else f * fr
    # the host first (device memory sets the region sizes and the stored
    # overlaps of the LMP2 / CC); the home device keeps the (Q|ij), the DF
    # build and the pair data: it comes last, with a smaller share
    place = [(None, float(os.environ.get('DLPNO_QIA_HOST_FRAC', '0.4')) * _host_available())]
    place += [(d, cap(d, frac)) for d in devs if d != home]
    place.append((home, cap(home, 0.5 * frac)))
    if sum(c for _, c in place) < 1.05 * nbytes:
        return None
    if len(devs) > 1:
        mgpu.enable_peer(devs)
    return place


def _decide_qab_direct(nbytes, host):
    """evaluate (Q|uv) per use (no stored blocks) when the blocks go to the
    host and would take more than 40% of its available memory
    (DLPNO_QAB_DIRECT=auto|0|1)."""
    env = os.environ.get('DLPNO_QAB_DIRECT', 'auto')
    if env != 'auto':
        return env == '1'
    return bool(host) and nbytes > 0.4 * _host_available()


def scratch_dir(tag):
    """a fresh directory for bulk run data on the scratch file system
    (DLPNO_SCRATCH, else $PSCRATCH/dlpno_scratch, else $SCRATCH); never in
    the home directory."""
    import tempfile
    base = os.environ.get('DLPNO_SCRATCH')
    if not base:
        for v in ('PSCRATCH', 'SCRATCH'):
            if os.environ.get(v):
                base = os.path.join(os.environ[v], 'dlpno_scratch')
                break
    if not base:
        raise RuntimeError('no scratch file system: set DLPNO_SCRATCH (bulk integral data '
                           'is not written to the home directory)')
    home = os.path.realpath(os.path.expanduser('~'))
    if os.path.realpath(base).startswith(home):
        raise RuntimeError(f'DLPNO_SCRATCH={base} is inside the home directory')
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=f'{tag}_{os.getpid()}_', dir=base)


class QabDiskStore:
    """(Q_a|uv) blocks of every aux atom over the union of what a phase's
    batches need (rows x columns), evaluated once and kept on the scratch file
    system; the regions read their sub-blocks back instead of recomputing the
    AO integrals.  File per atom, layout (rows, aux, columns): a row subset
    is a set of contiguous reads.  ``close()`` deletes the files."""

    def __init__(self, dfi, item_needs, devs, setup, verbose=0):
        import time
        from electrolites.dlpno import mgpu
        t0 = time.perf_counter()
        natm, nbf = len(dfi.atom_ribfs), dfi.nbf
        rm, cm = np.zeros((natm, nbf), bool), np.zeros((natm, nbf), bool)
        square = np.ones(natm, bool)
        for nd in item_needs:
            for a, u in nd.items():
                if isinstance(u, tuple):
                    rm[a, np.asarray(u[0], np.int64)] = True
                    cm[a, np.asarray(u[1], np.int64)] = True
                    square[a] = False
                else:
                    rm[a, np.asarray(u, np.int64)] = True
                    cm[a, np.asarray(u, np.int64)] = True
        atoms = [a for a in range(natm) if rm[a].any() and len(dfi.atom_ribfs[a])]
        self.rows = {a: np.flatnonzero(rm[a]) for a in atoms}
        self.cols = {a: (self.rows[a] if square[a] else np.flatnonzero(cm[a])) for a in atoms}
        self.rpos, self.cpos, self.shape, self.path = {}, {}, {}, {}
        for a in atoms:
            r = -np.ones(nbf, np.int64)
            r[self.rows[a]] = np.arange(len(self.rows[a]))
            self.rpos[a] = r
            if self.cols[a] is self.rows[a]:
                self.cpos[a] = r
            else:
                c = -np.ones(nbf, np.int64)
                c[self.cols[a]] = np.arange(len(self.cols[a]))
                self.cpos[a] = c
        self.dir = scratch_dir('qab')
        needs = {a: (self.rows[a] if self.cols[a] is self.rows[a] else (self.rows[a], self.cols[a]))
                 for a in atoms}
        groups = [(atoms, None)]
        if dfi._pao_sup is not None and atoms:
            groups = dfi._region_groups(atoms, needs)
        # one integral evaluation per atom, the groups spread over the devices
        jobs = [(g_atoms[k:k + 8], aosub) for g_atoms, aosub in groups for k in range(0, len(g_atoms), 8)]
        nbytes = [0.]

        def bwork(ctx, job):
            dv = getattr(ctx, 'dfi', None) or dfi
            g_atoms, aosub = job
            for a in g_atoms:
                rows, cols = self.rows[a], self.cols[a]
                na_ = len(dfi.atom_ribfs[a])
                # rows in pieces when the block (and its transposed copy) would
                # not fit next to what the device already holds; each piece
                # re-evaluates the atom's AO integrals
                fr = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
                full = 8. * na_ * len(rows) * len(cols)
                npc = max(1, int(np.ceil(2.5 * full / (0.5 * fr))), int(os.environ.get('DLPNO_DISK_ROWPIECES', 1)))
                p = os.path.join(self.dir, f'a{a}.bin')
                with open(p, 'wb') as fh:
                    for rr in np.array_split(rows, npc):
                        sq = cols is rows and npc == 1
                        blk, _, _, _ = dv._direct_rect(a, rr, None if sq else cols, cache=False, aosub=aosub)
                        h = cp.asnumpy(cp.ascontiguousarray(blk.transpose(1, 0, 2)))     # (rows, aux, cols)
                        del blk
                        cp.get_default_memory_pool().free_all_blocks()
                        h.tofile(fh)
                        nbytes[0] += h.nbytes
                        del h
                self.shape[a] = (len(rows), na_, len(cols))
                self.path[a] = p
        try:
            mgpu.run_queue(jobs, bwork, devs, setup=setup)
        except BaseException:
            self.close()
            raise
        self.nbytes = nbytes[0]
        if verbose >= 2:
            print(f'[mem] (Q|uv) disk store: {len(atoms)} atoms, {self.nbytes / 1e9:.1f} GB in {self.dir}, '
                  f'built in {time.perf_counter() - t0:.1f} s', flush=True)

    def __contains__(self, a):
        return a in self.path

    def read(self, a, u):
        """device block of atom ``a`` over ``u`` (PAOs, or (rows, columns)),
        with its row and column maps (the same object for a square need)."""
        rows, cols = u if isinstance(u, tuple) else (u, None)
        rows = np.asarray(rows, np.int64)
        ir = self.rpos[a][rows]
        ic = ir if cols is None else self.cpos[a][np.asarray(cols, np.int64)]
        assert (ir >= 0).all() and (ic >= 0).all(), 'block outside the stored domain'
        mm = np.memmap(self.path[a], dtype=np.float64, mode='r', shape=self.shape[a])
        sub = np.asarray(mm[ir])                                    # contiguous row reads
        if cols is None and len(ir) == self.shape[a][2] and np.array_equal(ic, np.arange(len(ic))):
            pass
        else:
            sub = sub[:, :, ic]
        del mm
        blk = cp.ascontiguousarray(cp.asarray(sub).transpose(1, 0, 2))      # (aux, rows, cols)
        nbf = len(self.rpos[a])
        rmap = -np.ones(nbf, np.int64)
        rmap[rows] = np.arange(len(rows))
        if cols is None:
            cmap = rmap
        else:
            cmap = -np.ones(nbf, np.int64)
            cmap[np.asarray(cols, np.int64)] = np.arange(len(cols))
        return blk, rmap, cmap

    def close(self):
        import shutil
        if getattr(self, 'dir', None) and os.path.isdir(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)
        self.dir = None


def _decide_qab_disk(dfi, regions, item_lneeds):
    """the on-disk store when the regions would evaluate each atom's AO
    integrals many times (DLPNO_QAB_DISK=auto|0|1; auto: more than 4 times on
    average, and no region (Q|iu), which needs the AO integrals anyway)."""
    env = os.environ.get('DLPNO_QAB_DISK', 'auto')
    if env != 'auto':
        return env == '1' and item_lneeds is None
    if item_lneeds is not None:
        return False
    evals = sum(len(nd) for _, nd, _ in regions)
    atoms = len({a for _, nd, _ in regions for a in nd})
    return atoms > 0 and evals > 4 * atoms


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


def _qia_stats(natm, lmo_riatoms, lmo_paos, pairs, nbf, ext_lmos):
    """diagnostic: (Q|iu) storage per aux atom (current), per (atom, LMO) over
    the LMO's extended PAOs, and per (atom, LMO) over exactly what its pairs
    through that atom need (in units of aux functions per atom = 1)."""
    n = len(lmo_riatoms)
    P = cp.zeros((n, nbf), cp.float32)
    A = cp.zeros((n, natm), cp.float32)
    for i in range(n):
        P[i, cp.asarray(lmo_paos[i])] = 1
        A[i, cp.asarray(lmo_riatoms[i])] = 1
    G = cp.zeros((n, n), cp.float32)
    pr = np.asarray(pairs)
    G[cp.asarray(pr[:, 0]), cp.asarray(pr[:, 1])] = 1
    G[cp.asarray(pr[:, 1]), cp.asarray(pr[:, 0])] = 1
    G[cp.arange(n), cp.arange(n)] = 1
    EP = (G @ P) > 0                                   # extended PAOs of each LMO
    s0 = s1 = s2 = 0.
    for a in range(natm):
        L = cp.asarray(ext_lmos[a])
        if len(L) == 0:
            continue
        s0 += float(len(L)) * float(cp.count_nonzero(P[L].sum(0)))
        s1 += float(cp.count_nonzero(EP[L]))
    for i in range(n):
        # pairs (i, j): aux(ij) = aux(i) | aux(j), pao(ij) = P(i) | P(j)
        Ai = G[i][:, None] * cp.maximum(A, A[i][None, :])      # (j, atom): j pairs with i and a in aux(ij)
        U = (Ai.T @ cp.maximum(P, P[i][None, :])) > 0           # (atom, pao)
        s2 += float(cp.count_nonzero(U))
    print(f'[debug] (Q|iu) entries per aux function: per atom {s0:.3e}, per (atom, LMO) ext-PAO {s1:.3e}, '
          f'per (atom, LMO) exact {s2:.3e}', flush=True)


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
        if os.environ.get('DLPNO_DEBUG_QIA_STATS') == '1':
            _qia_stats(natm, lmo_riatoms, lmo_paos, pairs, nbf, self.lmos)
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
        self.qia_direct = self.qia_spread = False
        if self.qia_sparse:
            nl_ = np.array([len(ext.lmos[a]) for a in range(mol.natm)])
            np_ = np.array([len(ext.paos[a]) for a in range(mol.natm)])
            na_ = np.array([len(atom_ribfs[a]) for a in range(mol.natm)])
            nb_ia = 8. * float(np.sum(na_ * nl_ * np_))
            if os.environ.get('V', '0') >= '2' or os.environ.get('DLPNO_VERBOSE_MEM') == '1':
                print(f'[mem] (Q|iu) block-sparse {nb_ia / 1e9:.1f} GB (dense {8e-9 * naux * n * nbf:.1f} GB), '
                      f'(Q|ij) block-sparse {8e-9 * float(np.sum(na_ * nl_ * nl_)):.1f} GB '
                      f'(dense {8e-9 * naux * n * n:.1f} GB), (Q|uv) extended {8e-9 * float(np.sum(na_ * np_.astype(np.float64) ** 2)):.1f} GB, naux {naux}, LMOs {n}, '
                      f'host available {_host_available() / 1e9:.1f} GB', flush=True)
            host_ia = _decide_qia_host(nb_ia)
            # closed shell: too large for the host as well -> spread over the
            # devices' memory and the host while that fits (kept for the LMP2
            # and the PNO integrals, then dropped: the triples evaluate theirs
            # per region), else per region throughout (direct)
            self.qia_direct = sparse_ij and need_ab and _decide_qia_direct(nb_ia, host_ia)
            place = _qia_placement(nb_ia) if (self.qia_direct or os.environ.get('DLPNO_QIA_SPREAD') == '1') \
                and os.environ.get('DLPNO_QIA_SPREAD') != '0' and os.environ.get('DLPNO_QIA_DIRECT') != '1' else None
            self.qia_spread = place is not None
            if self.qia_spread:
                self.qia_direct = False
                self.qia = BlockQ(naux, n, nbf, ribf_atom, atom_ribfs, ext.lmos, ext.paos, placement=place)
                if os.environ.get('V', '0') >= '2':
                    print('[mem] (Q|iu) spread over', [(('host' if d is None else f'GPU {d}'), round(b / 1e9, 1))
                                                      for d, b in self.qia.placed], 'GB', flush=True)
            elif self.qia_direct:
                self.qia = None
            else:
                self.qia = BlockQ(naux, n, nbf, ribf_atom, atom_ribfs, ext.lmos, ext.paos, host=host_ia)
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
        self.C_lmo_d = L if (self.qia_direct or self.qia_spread) else None
        # AO screening of the direct (Q|uv): AOs carrying |C_pao| above
        # DLPNO_AO_CUT x (largest |C| of that PAO); the region integrals are
        # evaluated only over the shells of the union of its PAOs' supports
        self.ao_cut = float(os.environ.get('DLPNO_AO_CUT', '1e-8'))
        self._pao_sup = None
        if self.ao_cut > 0:
            Cp = np.abs(np.asarray(C_pao))
            self._pao_sup = Cp > self.ao_cut * Cp.max(axis=0)[None, :]      # (nao, nbf)
            if self.qia_direct or self.qia_spread:
                Lh_ = np.abs(np.asarray(C_lmo))
                self._lmo_sup = Lh_ > self.ao_cut * Lh_.max(axis=0)[None, :]  # (nao, n)
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
        self.qab_direct = need_ab and (self.qia_direct or self.qia_spread
                                       or _decide_qab_direct(nab, self.qab_host))
        if self.qab_direct:
            self.qab_host = False
            self._shard = None
        self._ao_cache, self._ao_bytes = {}, {}
        self._regions = {}             # device -> region blocks (direct mode)
        self._region_qia = {}          # device -> region (Q|iu) (direct (Q|iu))
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
                if len(pa) and not (self.qia_direct and self.qab_direct):
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
            do_ia = self.qia is not None
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
                    if len(pa) and do_ia:
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
            if self.qia is None:
                pass
            elif self.qia_sparse:
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
        self.nbytes = dict(ia=self.qia.nbytes if self.qia is not None else 0., ij=self.qij.nbytes,
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
            blk, rmap, cmap = reg[a]
            assert rmap is cmap, 'rectangular region block read as square'
            return self._region_dev(a, blk), rmap
        return self._direct_block(a, need)

    def qab_block_rect(self, a, rows, cols):
        """(Q_a|uv) for u in the PAOs ``rows``, v in ``cols`` (index arrays
        or iterables of them): (block, row map, column map); stored blocks:
        the extended-domain block with its map for both.  Direct mode: only
        the rectangle is evaluated (columns may extend far beyond the rows,
        e.g. the extended domains of the PNO integrals)."""
        if not self.qab_direct:
            m = self.ext.pao_pos[a]
            return self.qab_get(a), m, m
        reg = getattr(self, '_regions', None)
        reg = reg.get(cp.cuda.Device().id) if reg else None
        if reg is not None and a in reg:
            blk, rmap, cmap = reg[a]
            return self._region_dev(a, blk), rmap, cmap
        out, rmap, cmap, _ = self._direct_rect(a, rows, cols)
        return out, rmap, cmap

    def _direct_block(self, a, need, cache=True, aosub=None, iu=None):
        """(Q_a|uv) over the PAOs ``need`` and its PAO -> position map, from the
        AO integrals of aux atom ``a`` (over the AO domain ``aosub`` when
        given).  With ``iu`` = (LMOs, PAOs) also (Q_a|i u) over them, from the
        same AO integrals: returns (uv block, map, iu block) then."""
        out, rmap, _, oiu = self._direct_rect(a, need, None, cache, aosub, iu)
        return (out, rmap) if iu is None else (out, rmap, oiu)

    @staticmethod
    def _union(x):
        if isinstance(x, np.ndarray):
            x = [x]
        x = [np.asarray(v, np.int64).ravel() for v in x]
        return np.unique(np.concatenate(x)) if x else np.zeros(0, np.int64)

    def _direct_rect(self, a, need, cols=None, cache=True, aosub=None, iu=None):
        """(Q_a|uv), u in ``need``, v in ``cols`` (None: square over ``need``):
        (block, row map, column map, (Q_a|iu) block or None)."""
        na = len(self.atom_ribfs[a])
        need = self._union(need)
        square = cols is None
        colv = need if square else self._union(cols)
        m, mc = len(need), len(colv)
        if na == 0 or (m == 0 and iu is None):
            return None, None, None, None
        import time
        t0 = time.perf_counter()
        gn = cp.asarray(need)
        gc = gn if square else cp.asarray(colv)
        if aosub is not None:
            smol, ao = aosub
            gao = cp.asarray(ao)
            Cp = self.C_pao_d[gao]
            nd = len(ao)
        else:
            smol, gao, Cp, nd = self.mol, None, self.C_pao_d, self.mol.nao
        Pn = Cp[:, gn] if m else None
        Pc = Pn if square else (Cp[:, gc] if mc else None)
        nl = mi = 0
        if iu is not None:
            li, pi = (np.asarray(x, np.int64) for x in iu)
            Cl = self.C_lmo_d if gao is None else self.C_lmo_d[gao]
            Ls, Pi = Cl[:, cp.asarray(li)], Cp[:, cp.asarray(pi)]
            nl, mi = len(li), len(pi)
        free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        # J column + the transforms of one aux function
        per_q = 8. * (nd * nd + m * nd + m * mc + nl * nd + nl * mi)
        # few, large integral calls: each call over a big AO domain has a fixed
        # screening cost (crambin: 300 MB per aux function over all 6177 AOs)
        lim = min(6e9, 0.15 * free)
        if os.environ.get('DLPNO_AUXCHUNK_GB'):           # testing
            lim = float(os.environ['DLPNO_AUXCHUNK_GB']) * 1e9
        if na * per_q > lim and not (aosub is None and cache):
            # large AO domains (triple zeta, compact molecules): the atom's aux
            # shells in chunks
            _memo_qcond()
            shl = np.flatnonzero(self.auxmol._bas[:, gto.ATOM_OF] == a)
            aloc = self.auxmol.ao_loc_nr()
            chunks, cur = [], []
            cap = max(1, int(lim // per_q))
            for sh in shl:
                w = aloc[sh + 1] - aloc[sh]
                if cur and sum(aloc[c + 1] - aloc[c] for c in cur) + w > cap:
                    chunks.append(cur)
                    cur = []
                cur.append(sh)
            if cur:
                chunks.append(cur)
        else:
            chunks = [None]
        out = cp.empty((na, m, mc)) if m else None
        oiu = cp.empty((na, nl, mi)) if iu is not None else None
        t_tr, q0 = 0., 0
        for ch in chunks:
            if ch is not None:
                Jc = int3c2e_bdiv.aux_e2(smol, sub_auxmol(self.auxmol, None, ch))
            elif aosub is None and cache:
                Jc = self._ao_j3c(a)
            else:
                Jc = aux_e2_atoms(smol, self.auxmol, [a])
            Jc = cp.ascontiguousarray(Jc)
            nc = Jc.shape[2]
            t1 = time.perf_counter()
            if m:
                uc = cp.matmul(Pn.T, Jc.reshape(nd, nd * nc)).reshape(m, nd, nc)
                out[q0:q0 + nc] = cp.matmul(uc.transpose(2, 0, 1), Pc)
                del uc
            if iu is not None and nl and mi:
                tc = cp.matmul(Ls.T, Jc.reshape(nd, nd * nc)).reshape(nl, nd, nc)
                oiu[q0:q0 + nc] = cp.matmul(tc.transpose(2, 0, 1), Pi)
                del tc
            elif iu is not None:
                oiu[q0:q0 + nc] = 0.
            del Jc
            q0 += nc
            t_tr += time.perf_counter() - t1
        assert q0 == na
        pmap = cmap = None
        if m:
            out *= self.mask_uv_d[gn[:, None], gc[None, :]][None]
            pmap = -np.ones(self.nbf, np.int64)
            pmap[need] = np.arange(m)
            if square:
                cmap = pmap
            else:
                cmap = -np.ones(self.nbf, np.int64)
                cmap[colv] = np.arange(mc)
        st = self.cache_stats
        st['direct'] = st.get('direct', 0) + 1
        st['direct_m'] = st.get('direct_m', 0) + m
        if os.environ.get('DLPNO_PROF') == '1':
            cp.cuda.Device().synchronize()
        tt = time.perf_counter() - t0
        st['t_ao'] = st.get('t_ao', 0.) + (tt - t_tr)
        st['t_tr'] = st.get('t_tr', 0.) + t_tr
        return out, pmap, cmap, oiu

    # ---- regions: direct (Q|uv) evaluated once per atom for a group of batches
    def plan_regions(self, item_needs, budget, item_lneeds=None):
        """split items (in order) into regions: consecutive items whose
        (Q_a|uv) blocks over the union of their PAOs per atom (and, direct
        (Q|iu), (Q_a|i u) blocks over the union of their LMOs x PAOs) stay
        within ``budget`` bytes.  item_needs[k]: {atom: PAO index array, or
        (row PAOs, column PAOs) for a rectangular block}; item_lneeds[k]:
        {atom: (LMO index array, PAO index array)}.  Returns
        [(item indices, {atom: PAOs or (rows, columns)}, {atom: (LMOs, PAOs)})]."""
        natm, nbf, n = len(self.atom_ribfs), self.nbf, self.n
        na = np.array([len(self.atom_ribfs[a]) for a in range(natm)], np.float64)
        if item_lneeds is None:
            item_lneeds = [{}] * len(item_needs)
        regions = []

        class R:
            pass

        def fresh():
            r = R()
            r.items, r.tot = [], 0.
            r.pm, r.pc = np.zeros((natm, nbf), bool), np.zeros(natm, np.int64)
            r.cm, r.cc = np.zeros((natm, nbf), bool), np.zeros(natm, np.int64)
            r.lm, r.qm = np.zeros((natm, n), bool), np.zeros((natm, nbf), bool)
            r.lc, r.qc = np.zeros(natm, np.int64), np.zeros(natm, np.int64)
            return r

        def grow(r, nd, ld):
            """rows to add and the new total."""
            addp, addc, addl, addq = {}, {}, {}, {}
            for a, u in nd.items():
                ur, uc = u if isinstance(u, tuple) else (u, u)
                m = addp[a] if a in addp else r.pm[a].copy()
                m[np.asarray(ur, np.int64)] = True
                addp[a] = m
                m = addc[a] if a in addc else r.cm[a].copy()
                m[np.asarray(uc, np.int64)] = True
                addc[a] = m
            for a, (l_, u) in ld.items():
                m = addl[a] if a in addl else r.lm[a].copy()
                m[np.asarray(l_, np.int64)] = True
                addl[a] = m
                m = addq[a] if a in addq else r.qm[a].copy()
                m[np.asarray(u, np.int64)] = True
                addq[a] = m
            new = r.tot + sum(8. * na[a] * (int(m.sum()) * int(addc[a].sum()) - int(r.pc[a]) * int(r.cc[a]))
                              for a, m in addp.items())
            for a in addl:
                lc, qc = int(addl[a].sum()), int(addq[a].sum())
                new += 8. * na[a] * (lc * qc - int(r.lc[a]) * int(r.qc[a]))
            return addp, addc, addl, addq, new

        def close(r):
            nd = {}
            for a in range(natm):
                if r.pc[a]:
                    rows = np.flatnonzero(r.pm[a])
                    nd[a] = rows if np.array_equal(r.pm[a], r.cm[a]) else (rows, np.flatnonzero(r.cm[a]))
            ld = {a: (np.flatnonzero(r.lm[a]), np.flatnonzero(r.qm[a])) for a in range(natm) if r.lc[a]}
            regions.append((r.items, nd, ld))
        r = fresh()
        for k, (nd, ld) in enumerate(zip(item_needs, item_lneeds)):
            addp, addc, addl, addq, new = grow(r, nd, ld)
            if r.items and new > budget:
                close(r)
                r = fresh()
                addp, addc, addl, addq, new = grow(r, nd, ld)
            for a, m in addp.items():
                r.pm[a], r.pc[a] = m, int(m.sum())
                r.cm[a], r.cc[a] = addc[a], int(addc[a].sum())
            for a in addl:
                r.lm[a], r.lc[a] = addl[a], int(addl[a].sum())
                r.qm[a], r.qc[a] = addq[a], int(addq[a].sum())
            r.tot = new
            r.items.append(k)
        if r.items:
            close(r)
        return regions

    @staticmethod
    def _need_bytes(na, u):
        """bytes of a region block over ``u`` (PAOs, or (rows, columns))."""
        if isinstance(u, tuple):
            return 8. * na * len(u[0]) * len(u[1])
        return 8. * na * len(u) ** 2

    def region_begin(self, needs, verbose=0, lneeds=None):
        """(Q_a|uv) over needs[a] for every atom a, on the current device, each
        from one evaluation of its AO integrals; kept on the device when they
        fit (30% of its free memory), else in pinned host memory behind the
        device LRU cache.  Direct (Q|iu): with ``lneeds`` {a: (LMOs, PAOs)}
        also the region's (Q|iu) (a device BlockQ over these domains) from the
        same AO integrals."""
        import time
        t0 = time.perf_counter()
        dev = cp.cuda.Device().id
        self._dev_clear(dev)
        lneeds = lneeds or {}
        allat = set(needs) | set(lneeds)
        na = {a: len(self.atom_ribfs[a]) for a in allat}
        tot = sum(self._need_bytes(na[a], u) for a, u in needs.items())
        tot_iu = sum(8. * na[a] * len(l_) * len(u) for a, (l_, u) in lneeds.items())
        free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        on_dev = tot + tot_iu < 0.3 * free and os.environ.get('DLPNO_REGION_HOST') != '1'   # testing
        atoms = [a for a in sorted(allat) if na[a] and (a in needs or a in lneeds)]
        disk = getattr(self, '_disk', None)
        if disk is not None and not lneeds:
            # blocks read back from the scratch store (no AO integrals)
            reg = {}
            for a in atoms:
                blk, rmap, cmap = disk.read(a, needs[a])
                reg[a] = (blk if on_dev else to_pinned(blk), rmap, cmap)
                del blk
            self._regions[dev] = reg
            self._region_qia[dev] = None
            if verbose >= 3:
                print(f'[mem] (Q|uv) region on GPU {dev}: {len(atoms)} atoms, {tot / 1e9:.1f} GB read '
                      f'from disk in {time.perf_counter() - t0:.1f} s', flush=True)
            return
        groups = [(atoms, None)]
        if self._pao_sup is not None and atoms:
            groups = self._region_groups(atoms, needs, lneeds)
        bq = None
        if lneeds:
            natm = len(self.atom_ribfs)
            e0 = np.zeros(0, np.int64)
            bq = BlockQ(self.naux, self.n, self.nbf, self.ribf_atom, self.atom_ribfs,
                        [lneeds[a][0] if a in lneeds else e0 for a in range(natm)],
                        [lneeds[a][1] if a in lneeds else e0 for a in range(natm)],
                        host=not on_dev)
        reg = {}
        for g_atoms, aosub in groups:
            for a in g_atoms:
                u = needs.get(a)
                rows, cols = (u if isinstance(u, tuple) else (u, None)) if u is not None else ([], None)
                blk, rmap, cmap, qiu = self._direct_rect(a, rows, cols, cache=False, aosub=aosub,
                                                         iu=lneeds.get(a))
                if qiu is not None:
                    bq.set_block(a, qiu)
                    del qiu
                if blk is not None:
                    reg[a] = (blk if on_dev else to_pinned(blk), rmap, cmap)
                del blk
        self._regions[dev] = reg
        self._region_qia[dev] = bq
        if verbose >= 2:
            nds = [len(s[1]) if s is not None else self.mol.nao for _, s in groups]
            iu_s = f' + (Q|iu) {tot_iu / 1e9:.1f} GB' if lneeds else ''
            print(f'[mem] (Q|uv) region on GPU {dev}: {len(atoms)} atoms, {tot / 1e9:.1f} GB{iu_s} on the '
                  f'{"device" if on_dev else "host"}, {len(groups)} AO domains of {min(nds)}..{max(nds)} '
                  f'of {self.mol.nao} AOs, built in {time.perf_counter() - t0:.1f} s', flush=True)

    def _region_groups(self, atoms, needs, lneeds=None):
        """the region's aux atoms in groups of neighbours (Morton order), each
        with the AO shells supporting its atoms' PAOs (and LMOs, direct
        (Q|iu)): the AO integrals and the transformation of an atom run over
        its group's AO domain instead of the region's (DLPNO_REGION_SPLIT=0:
        one domain for the region).
        -> [(atoms, (sub-molecule, AO indices) or None)]"""
        nbas = self.mol.nbas
        loc = self.mol.ao_loc_nr()
        if getattr(self, '_pao_shsup', None) is None:
            self._pao_shsup = np.ascontiguousarray(
                np.logical_or.reduceat(self._pao_sup, loc[:-1], axis=0).T)      # (nbf, nbas)
        lneeds = lneeds or {}
        if lneeds and getattr(self, '_lmo_shsup', None) is None:
            self._lmo_shsup = np.ascontiguousarray(
                np.logical_or.reduceat(self._lmo_sup, loc[:-1], axis=0).T)      # (n, nbas)
        sup = {}
        for a in atoms:
            m = np.zeros(nbas, bool)
            if a in needs:
                u = needs[a]
                u = np.concatenate(u) if isinstance(u, tuple) else u
                if len(u):
                    m |= self._pao_shsup[np.asarray(u, np.int64)].any(axis=0)
            if a in lneeds:
                l_, u = lneeds[a]
                if len(u):
                    m |= self._pao_shsup[np.asarray(u, np.int64)].any(axis=0)
                if len(l_):
                    m |= self._lmo_shsup[np.asarray(l_, np.int64)].any(axis=0)
            sup[a] = m
        if os.environ.get('DLPNO_REGION_SPLIT', '1') == '0':
            parts = [atoms]
        else:
            order = [atoms[k] for k in self.spatial_order([[a] for a in atoms])]
            parts, cur, u = [], [], None
            for a in order:
                # join the group while its AO domain grows by at most 10%
                if cur:
                    v = u | sup[a]
                    if v.sum() <= 1.1 * max(u.sum(), sup[a].sum()):
                        cur.append(a)
                        u = v
                        continue
                    parts.append(cur)
                cur, u = [a], sup[a].copy()
            parts.append(cur)
        out = []
        for g in parts:
            m = np.zeros(nbas, bool)
            for a in g:
                m |= sup[a]
            shells = np.flatnonzero(m)
            out.append((sorted(g), sub_mol_shells(self.mol, shells) if len(shells) < nbas else None))
        return out

    def region_end(self):
        dev = cp.cuda.Device().id
        self._regions.pop(dev, None)
        self._region_qia.pop(dev, None)
        self._dev_clear(dev)
        import gc
        gc.collect()
        # pinned blocks back to the OS (the pinned pool would keep them, and the
        # next region's blocks have other sizes); this device's pool too
        from electrolites.dlpno.devmem import free_pinned
        free_pinned()
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

    def run_regions(self, items, item_needs, work, setup, devs, verbose=0, item_lneeds=None):
        """work(ctx, item) for every item, the items grouped into regions
        (direct (Q|uv); direct (Q|iu) with ``item_lneeds``: ctx.dfi.qia is the
        region's); the regions go to the devices through a work queue
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
        if item_lneeds is not None and not self.qia_direct:
            item_lneeds = None
        regions = self.plan_regions(item_needs, budget, item_lneeds)
        if verbose >= 2:
            evals = sum(len(nd) for _, nd, _ in regions)
            print(f'[mem] (Q|uv) regions: {len(regions)} for {len(items)} batches '
                  f'(budget {budget / 1e9:.1f} GB per region, {evals} atom blocks)', flush=True)
        self._regions = {}
        self._region_qia = {}
        self._disk = None
        if _decide_qab_disk(self, regions, item_lneeds):
            self._disk = QabDiskStore(self, item_needs, devs, setup, verbose)

        def rwork(ctx, region):
            idx, needs, lneeds = region
            dv = getattr(ctx, 'dfi', None) or self        # this device's view
            dev = cp.cuda.Device().id
            dv._dev_clear(dev)
            cp.get_default_memory_pool().free_all_blocks()
            fr = cp.cuda.runtime.memGetInfo()[0]
            tot = sum(self._need_bytes(len(self.atom_ribfs[a]), u) for a, u in needs.items())
            tot += sum(8. * len(self.atom_ribfs[a]) * len(l_) * len(u) for a, (l_, u) in lneeds.items())
            if tot >= 0.3 * fr and len(idx) > 1 and not env:
                # this device has less room than planned: smaller regions kept
                # on the device rather than one in pinned host memory
                subs = [([idx[j] for j in si], nd, ld) for si, nd, ld in
                        self.plan_regions([item_needs[k] for k in idx], 0.25 * fr,
                                          None if item_lneeds is None else [item_lneeds[k] for k in idx])]
            else:
                subs = [region]
            for sidx, sneeds, slneeds in subs:
                dv.region_begin(sneeds, verbose, slneeds)
                if slneeds:
                    dv.qia = dv._region_qia.get(dev)
                try:
                    for k in sidx:
                        work(ctx, items[k])
                finally:
                    if slneeds:
                        dv.qia = None
                    dv.region_end()
        try:
            return mgpu.run_queue(regions, rwork, devs, setup=setup)
        finally:
            self._regions = {}
            self._region_qia = {}
            if self._disk is not None:
                self._disk.close()
            self._disk = None

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
        try:
            d = cp.asarray(q)
        except cp.cuda.memory.OutOfMemoryError:
            # the budget was taken while the other device threads still had
            # their batch temporaries to allocate: drop this device's cache
            with lock():
                cache.clear()
                self._cache_bytes[dev] = nb = 0
            cp.get_default_memory_pool().free_all_blocks()
            self.cache_stats['oom_evict'] = self.cache_stats.get('oom_evict', 0) + 1
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
        # one scale for all axes: a dimension with a tiny extent (a planar
        # zig-zag chain) would otherwise get the full bit range and its two
        # levels would split the order into far-apart halves
        q = ((c - lo) / max(float((hi - lo).max()), 1e-9) * 1023).astype(np.int64)
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
