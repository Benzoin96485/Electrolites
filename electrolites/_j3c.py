"""The device-side constants of a three-centre Coulomb build, uploaded once.

`fastdfj3c` and anything else that drives ``df/j_engine_3c2e.py`` needs the
same handful of integer tables on the card -- the shell-pair list, the block
offsets, the Hermite offsets of each pair, the auxiliary ``ao_loc`` -- and
GPU4PySCF re-uploads every one of them on **every** call.  They are properties
of the basis and the screening, both fixed at ``Int3c2eOpt.build()``, so they
are uploaded here once and cached on the object that owns them.

Both caches are keyed on the **device**.  GPU4PySCF distributes the cderi
build across devices with ``multi_gpu.run``, which hands the *same*
``SortedMole`` to a thread per device, and a CuPy array carries the device it
was allocated on -- so a cache without that key would give one device another
device's pointers.  `docs/DF_WORK_LIST.md` W2 records the same trap on
``fastdf``'s ``_resident_prefix``, where it was found rather than avoided.

This module patches nothing.  It exists so that a benchmark can import the
caches without importing the patches, which is `docs/DF_WORK_LIST.md` §5.12.
"""
import numpy as np
import cupy as cp

#: attribute the per-device caches hang off; the ``int3c2e_envs`` each was
#: built for is recorded with it, so a second ``build()`` on the same
#: ``Int3c2eOpt`` cannot serve a stale pair list (``DF.reset()`` and a geometry
#: optimisation both do that)
_ATTR = '_electrolites_j3c'


class Int3c2eConstants:
    """Everything ``contract_dm`` and ``contract_auxvec`` re-derive per call.

    Upstream builds each of these inside the call: five ``cp.asarray`` of host
    arrays, a ``np.cumsum`` over the auxiliary shells and a Python loop over
    the angular-momentum groups.  None of it depends on the density.
    """

    def __init__(self, opt):
        from gpu4pyscf.df.int3c2e_bdiv import L_AUX_MAX          # noqa: F401
        mol = opt.mol
        auxmol = opt.auxmol
        self.envs = opt.int3c2e_envs

        self.nao = int(mol.ao_loc[-1])
        self.naux_cart = int(auxmol.nao_nr(cart=True))
        self.omega = mol.omega
        self.nbas = mol.nbas
        self.auxnbas = auxmol.nbas

        # the angular momentum of each shell-pair group and of each auxiliary
        # group, which is what sizes the shared memory a block of that class
        # needs.  Read from the first pair of each group, which is what the
        # kernels themselves do.
        nbas = mol.nbas
        bas_l = np.asarray(mol._bas[:, 1], dtype=np.int32)      # ANG_OF
        off0 = np.asarray(opt.shl_pair_offsets, dtype=np.int64)
        idx0 = np.asarray(opt.shl_pair_idx)
        self.group_lij = np.array(
            [bas_l[idx0[lo] // nbas] + bas_l[idx0[lo] % nbas]
             if hi > lo else 0
             for lo, hi in zip(off0[:-1], off0[1:])], dtype=np.int32)

        self.shl_pair_idx = cp.asarray(opt.shl_pair_idx, dtype=np.int32)
        self.npair = len(opt.shl_pair_idx)
        self._sp_offsets_host = np.asarray(opt.shl_pair_offsets,
                                           dtype=np.int32)
        self.pair_ij_offsets = cp.asarray(opt.shl_pair_offsets, dtype=np.int32)
        self.sp_blocks = len(opt.shl_pair_offsets) - 1
        #: chunk -> (device offsets, block count), memoised
        self._sp_split = {0: (self.pair_ij_offsets, self.sp_blocks)}
        self.pair_loc = cp.asarray(opt.pair_loc, dtype=np.int32)
        self.dm_xyz_size = int(opt.pair_loc[-1])
        # `_dm_to_Rt`/`_Rt_to_dm` upload this one per call as well
        self.ao_loc = cp.asarray(mol.ao_loc, dtype=np.int32)

        aux_loc = auxmol.ao_loc
        self.naux_sph = int(aux_loc[-1])
        self.aux_loc = cp.asarray(aux_loc, dtype=np.int32)
        l = auxmol._bas[:, 1]                                   # ANG_OF
        nf3 = (l + 1) * (l + 2) * (l + 3) // 6
        aux_xyz_loc = np.asarray(np.append(0, np.cumsum(nf3)), dtype=np.int32)
        self.aux_xyz_loc = cp.asarray(aux_xyz_loc)
        self.aux_xyz_size = int(aux_xyz_loc[-1])

        # the load-balancing split of the auxiliary shells within each
        # angular-momentum group -- a Python loop upstream runs on every
        # Coulomb build, 16 shells at a time
        self._aux_counts = list(auxmol.l_ctr_counts)
        self._aux_l = [int(x) for x in auxmol.uniq_l_ctr[:, 0]]
        self._ksh_split = {}
        self._ksh_group_split = {}
        self.ksh_offsets, self.ksh_blocks = self.ksh_batches(16)

        # Scratch that the two contractions allocate and free on every call.
        # `docs/DF_WORK_LIST.md` §5.8: a temporary allocated inside a loop
        # does not cost its allocation, it costs the `cudaFree` that ends it,
        # which is a synchronisation point.  Only buffers the caller never
        # sees are kept here -- `vj_aux` and `vj` are returned and so cannot
        # be recycled.
        self._scratch = {}

    def ksh_batches(self, chunk=16):
        """Auxiliary-shell block boundaries, ``chunk`` shells at a time.

        16 is GPU4PySCF's.  It trades the auxiliary-vector kernel's
        parallelism against how often it repeats a shell pair's Gaussian
        product centre, which it computes outside the auxiliary loop.
        """
        got = self._ksh_split.get(chunk)
        if got is not None:
            return got
        parts = []
        k0 = k1 = self.nbas
        for n in self._aux_counts:
            k0, k1 = k1, k1 + n
            parts.append(np.arange(k0, k1, chunk, dtype=np.int32))
        parts.append(np.int32(k1))
        off = np.hstack(parts, dtype=np.int32)
        got = (cp.asarray(off), len(off) - 1)
        self._ksh_split[chunk] = got
        return got

    def ksh_batches_by_group(self, chunk=16):
        """``ksh_batches``, kept per auxiliary angular-momentum group.

        Returned as ``[(lk, device offsets, block count), ...]``.  A launch
        restricted to one group can ask for the shared memory *that* class
        needs instead of the widest class in the whole basis.
        """
        got = self._ksh_group_split.get(chunk)
        if got is not None:
            return got
        out = []
        k0 = k1 = self.nbas
        for lk, n in zip(self._aux_l, self._aux_counts):
            k0, k1 = k1, k1 + n
            if n == 0:
                continue
            off = np.append(np.arange(k0, k1, chunk, dtype=np.int32),
                            np.int32(k1))
            out.append((lk, cp.asarray(off), len(off) - 1))
        self._ksh_group_split[chunk] = out
        return out

    def pair_offsets_by_group(self, chunk=0):
        """``pair_offsets``, kept per shell-pair angular-momentum group.

        Returned as ``[(lij, device offsets, block count), ...]``.
        """
        key = ('by_group', chunk)
        got = self._sp_split.get(key)
        if got is not None:
            return got
        off = self._sp_offsets_host
        out = []
        for lij, lo, hi in zip(self.group_lij, off[:-1], off[1:]):
            if hi <= lo:
                continue
            part = (np.append(np.arange(lo, hi, chunk, dtype=np.int32),
                              np.int32(hi)) if chunk else
                    np.array([lo, hi], dtype=np.int32))
            out.append((int(lij), cp.asarray(part), len(part) - 1))
        self._sp_split[key] = out
        return out

    def auto_chunk(self, ksh_blocks, target, lo, hi):
        """The shell-pair block size this molecule and basis want.

        A fixed size is wrong, and `docs/README_fastdfj3c.md` §5 is the sweep
        that says so: the auxiliary-vector kernel's optimum moves from 256
        shell pairs per block on a 10-atom molecule to 8192 on a 284-atom one,
        because what the split buys is thread blocks and how many the kernel
        already has is ``groups x auxiliary-shell batches``.  So the size is
        derived from the count it is trying to reach rather than assumed:
        enough blocks for the card, and then as few as possible, because every
        extra block repeats the block prologue.

        Rounded **down** to a power of two, which is what was swept, and
        clamped at both ends.
        """
        want_sp = -(-target // max(int(ksh_blocks), 1))
        chunk = self.npair / max(want_sp, 1)
        chunk = 1 << max(int(chunk).bit_length() - 1, 0) if chunk >= 1 else lo
        return int(min(max(chunk, lo), hi))

    def pair_offsets(self, chunk=0):
        """Shell-pair block boundaries, each group split into ``chunk`` pairs.

        ``Int3c2eOpt.build`` makes **one** block per angular-momentum group
        and nothing splits them further, so the auxiliary-vector kernel --
        whose grid is (groups x auxiliary-shell batches) -- launches a few
        hundred thread blocks on a 108-SM card, and the groups differ in size
        by two orders of magnitude.  Both kernels read the group's ``(li,lj)``
        from the *first* pair of their range and stride the rest, so a range
        may be any sub-interval of a group: splitting one is a launch
        configuration change and not a kernel change.  ``chunk = 0`` is
        GPU4PySCF's own, whole groups.
        """
        got = self._sp_split.get(chunk)
        if got is not None:
            return got
        off = self._sp_offsets_host
        parts = [np.arange(lo, hi, chunk, dtype=np.int32)
                 for lo, hi in zip(off[:-1], off[1:]) if hi > lo]
        parts.append(np.int32(off[-1]))
        split = np.hstack(parts, dtype=np.int32)
        got = (cp.asarray(split), len(split) - 1)
        self._sp_split[chunk] = got
        return got

    def scratch(self, key, size, zero=False):
        """A device buffer of ``size`` doubles, reused across calls."""
        buf = self._scratch.get(key)
        if buf is None or buf.size < size:
            buf = cp.empty(size)
            self._scratch[key] = buf
        buf = buf[:size]
        if zero:
            buf.fill(0.)
        return buf


def constants(opt):
    """The cached constants of one ``Int3c2eOpt`` on this device."""
    if opt.int3c2e_envs is None:
        opt.build()
    by_device = getattr(opt, _ATTR, None)
    if by_device is None:
        by_device = {}
        setattr(opt, _ATTR, by_device)
    key = cp.cuda.Device().id
    cache = by_device.get(key)
    if cache is None or cache.envs is not opt.int3c2e_envs:
        cache = Int3c2eConstants(opt)
        by_device[key] = cache
    return cache


#: the five device arrays every ``SortedGTO`` recontraction kernel uploads on
#: every call.  They hang off the object, one set per device.
_SORTED_ATTR = '_electrolites_ctr'


class SortedGTOConstants:
    """``c_ao_loc``, ``p_ao_loc`` and the three recontraction tables.

    ``SortedGTO.c_ao_loc`` is a *property*: it recomputes a NumPy cumsum and
    copies it to the device every time it is read, and each of the four
    recontraction kernels reads it plus four more host arrays.  All five are
    written once in ``SortedGTO.from_mol`` and never again.
    """

    def __init__(self, mol):
        self.c_ao_loc = cp.asarray(mol.c_ao_loc, dtype=np.int32)
        self.p_ao_loc = cp.asarray(mol.p_ao_loc, dtype=np.int32)
        self.recontract_coef = cp.asarray(mol.recontract_coef)
        self.recontract_bas = cp.asarray(mol.recontract_bas)
        self.recontraction_idx = cp.asarray(mol.recontraction_idx)
        self.nrecontract = len(mol.recontract_bas)


def sorted_constants(mol):
    """The cached recontraction tables of one ``SortedGTO`` on this device."""
    by_device = getattr(mol, _SORTED_ATTR, None)
    if by_device is None:
        by_device = {}
        setattr(mol, _SORTED_ATTR, by_device)
    key = cp.cuda.Device().id
    cache = by_device.get(key)
    if cache is None:
        cache = SortedGTOConstants(mol)
        by_device[key] = cache
    return cache


__all__ = ['constants', 'sorted_constants', 'Int3c2eConstants',
           'SortedGTOConstants']
