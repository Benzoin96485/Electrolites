"""Dense four-centre integral blocks on GPU4PySCF >= 1.9 (``RYS_fill_int4c2e``).

GPU4PySCF 1.9.0 (#936) removed the 1.8 ``scf/int4c2e.py`` driver this
repository's post-HF modules were built on -- ``int4c2e._VHFOpt(mol,
'int2e').build(group_size, diag_block_with_triu=True)``, its ``coeff``,
``l_ctr_offsets``, ``log_qs`` and ``uniq_l_ctr``, and
``ccsd_incore._fill_eri_block`` -- and moved the in-core CCSD onto
``libvhf_rys.RYS_fill_int4c2e`` over ``jk._VHFOpt``'s per-group pair lists.

:class:`BlockOpt` is that 1.8 interface rebuilt on the 1.9 kernel, so
`fastccsd`, `eom_gpu` and everything on their ladder keep one code path:

* the shells are sorted into ``(l, nprim)`` groups of at most ``group_size``
  shells (``SortedMole.from_mol(..., group_size=)``), which is what bounds a
  block, as ``build(group_size=...)`` did;
* ``coeff`` is the sorted-AO x original-AO matrix (``C @ mo`` gives the MO
  coefficients in the sorted basis, ``C.T @ X @ C`` takes a sorted-AO matrix
  back);
* ``l_ctr_offsets``/``uniq_l_ctr``/``log_qs`` enumerate the lower-triangle
  group pairs in ``np.tril_indices`` order, ``log_qs[g]`` empty where no
  shell pair of the pair survives the screen;
* :meth:`block` returns ``eri[i, k, j, l] = (ij|kl)`` for bra group pair
  ``g``, all ``k, l``, both triangles filled -- the block 1.8's
  ``_fill_eri_block`` filled at offsets ``[0, 0, i0, j0]`` with
  ``diag_block_with_triu=True`` and its callers then symmetrised over
  ``(k, l)``.

The kernel call and the scatter are GPU4PySCF 1.9.2's own
(``cc/ccsd_incore.py:_direct_ovvv_vvvv``, ``fint``), reached by group-pair
id -- except one line of the off-diagonal scatter, which does not survive
split groups (see :meth:`BlockOpt.block`).
"""
import ctypes
import math

import numpy as np
import cupy as cp


class BlockOpt:
    """The 1.8 ``int4c2e._VHFOpt`` surface over 1.9's ``jk._VHFOpt``."""

    def __init__(self, mol, group_size=None, direct_scf_tol=1e-13):
        from gpu4pyscf.scf import jk
        from gpu4pyscf.gto.mole import SortedMole
        smol = SortedMole.from_mol(mol, decontract=False,
                                   group_size=group_size)
        self.vhfopt = jk._VHFOpt(smol, direct_scf_tol).build()
        # `build` re-sorts through `SortedGTO.from_mol`, which hands a
        # SortedGTO back unchanged -- so this is the grouping asked for
        self.mol = smol = self.vhfopt.sorted_mol
        self.direct_scf_tol = direct_scf_tol
        self.uniq_l_ctr = smol.uniq_l_ctr
        self.l_ctr_offsets = np.append(0, np.cumsum(smol.l_ctr_counts))
        self.nao = int(smol.ao_loc[-1])
        nao0 = mol.nao
        self.coeff = smol.apply_C_dot(cp.eye(nao0))
        ngroups = len(self.uniq_l_ctr)
        cp_idx, cp_jdx = np.tril_indices(ngroups)
        cache = self.vhfopt.bas_pair_cache
        self._keys = list(zip(cp_idx.tolist(), cp_jdx.tolist()))
        self.log_qs = [cp.asarray(cache[k][1]) if k in cache
                       else cp.empty(0, dtype=np.float32)
                       for k in self._keys]
        self._prepared = None

    # -- GPU4PySCF 1.9.2's pair tables, built once -----------------------
    def _prepare(self):
        if self._prepared is not None:
            return self._prepared
        mol = self.mol
        nao = self.nao
        uniq_l = mol.uniq_l_ctr[:, 0]
        nf = (uniq_l + 1) * (uniq_l + 2) // 2
        carts = [cp.arange(n) for n in nf]
        ao_loc_gpu = cp.asarray(mol.ao_loc)
        cache = {k: [cp.asarray(x) for x in v]
                 for k, v in self.vhfopt.bas_pair_cache.items()}
        pair_loc = []
        ao_pair_addresses = []
        ao_pair_offsets = {}
        p0 = p1 = 0
        for i, j in cache:
            bas_ij = cache[i, j][0]
            pair_loc.append(cp.arange(len(bas_ij), dtype=np.int32)
                            * int(nf[i] * nf[j]))
            ish, jsh = divmod(bas_ij, mol.nbas)
            iaddr = ao_loc_gpu[ish, None] + carts[i]
            jaddr = ao_loc_gpu[jsh, None] + carts[j]
            ao_pair_addresses.append(
                (iaddr[:, None, :] * nao + jaddr[:, :, None]).ravel())
            p0, p1 = p1, p1 + len(ao_pair_addresses[-1])
            ao_pair_offsets[i, j] = (p0, p1)
        ao_pair_addresses = cp.hstack(ao_pair_addresses).astype(np.int32)
        iaddr, jaddr = divmod(ao_pair_addresses, nao)
        from gpu4pyscf.__config__ import props as gpu_specs
        from gpu4pyscf.scf import jk
        pool = cp.empty(gpu_specs['multiProcessorCount'] * jk.QUEUE_DEPTH + 1,
                        dtype=np.int32)
        self._prepared = dict(
            nf=nf, cache=cache, keys=list(cache.keys()), pair_loc=pair_loc,
            ao_pair_addresses=ao_pair_addresses, ao_pair_offsets=ao_pair_offsets,
            iaddr=iaddr, jaddr=jaddr, pool=pool,
            l_ctr_bas_loc=self.l_ctr_offsets)
        return self._prepared

    def block(self, group_id):
        """``eri[i, k, j, l] = (ij|kl)`` for tril group pair ``group_id``
        (``np.tril_indices`` order), shape ``(di, nao, dj, nao)``."""
        from gpu4pyscf.scf import jk
        from gpu4pyscf.__config__ import shm_size
        from gpu4pyscf.lib.cupy_helper import fill_symmetric
        P = self._prepare()
        mol = self.mol
        nao = self.nao
        ao_loc = mol.ao_loc
        k, l = self._keys[group_id]
        ish0, ish1 = self.l_ctr_offsets[k], self.l_ctr_offsets[k + 1]
        jsh0, jsh1 = self.l_ctr_offsets[l], self.l_ctr_offsets[l + 1]
        i0, i1 = int(ao_loc[ish0]), int(ao_loc[ish1])
        j0, j1 = int(ao_loc[jsh0]), int(ao_loc[jsh1])
        di, dj = i1 - i0, j1 - j0
        eri = cp.zeros([di, nao, dj, nao])
        cache = P['cache']
        if (k, l) not in cache:
            return eri
        pair_kl, q_cond_kl, _ = cache[k, l]
        npairs_kl = pair_kl.size
        if npairs_kl == 0:
            return eri
        nf = P['nf']
        kl_id = P['keys'].index((k, l))
        nao_kl_pairs = int(npairs_kl * nf[k] * nf[l])
        nao_pairs = len(P['ao_pair_addresses'])
        # screened quartets are not written by RYS_fill_int4c2e
        eri_tmp = cp.zeros((nao_pairs, nao_kl_pairs))
        log_cutoff = math.log(self.direct_scf_tol)
        lb = P['l_ctr_bas_loc']
        kern = jk.libvhf_rys.RYS_fill_int4c2e
        ij0 = ij1 = 0
        for ij_id, (i, j) in enumerate(cache):
            shls_slice = lb[[i, i + 1, j, j + 1, k, k + 1, l, l + 1]]
            pair_ij, q_cond_ij, _ = cache[i, j]
            npairs_ij = pair_ij.size
            if npairs_ij == 0:
                continue
            ij0, ij1 = ij1, ij1 + int(npairs_ij * nf[i] * nf[j])
            err = kern(
                ctypes.cast(eri_tmp[ij0:ij1].data.ptr, ctypes.c_void_p),
                ctypes.c_int(nao_kl_pairs),
                ctypes.c_double(0.), ctypes.c_double(1.), ctypes.c_double(0.),
                ctypes.byref(self.vhfopt.rys_envs),
                (ctypes.c_int * 8)(*shls_slice),
                ctypes.c_int(shm_size),
                ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                ctypes.cast(pair_ij.data.ptr, ctypes.c_void_p),
                ctypes.cast(pair_kl.data.ptr, ctypes.c_void_p),
                ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                ctypes.cast(P['pair_loc'][ij_id].data.ptr, ctypes.c_void_p),
                ctypes.cast(P['pair_loc'][kl_id].data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff),
                ctypes.cast(P['pool'].data.ptr, ctypes.c_void_p),
                mol._bas.ctypes)
            if err != 0:
                raise RuntimeError('RYS_fill_int4c2e failed')
        eri_tmp = fill_symmetric(eri_tmp, P['ao_pair_addresses'], nao)
        p0, p1 = P['ao_pair_offsets'][k, l]
        i = P['iaddr'][p0:p1] - i0
        j = P['jaddr'][p0:p1] - j0
        if ish0 == jsh0:
            eri_tmp = fill_symmetric(eri_tmp.reshape(nao**2, -1).T,
                                     i * dj + j, di)
            eri[:] = eri_tmp.reshape(di, di, nao, nao).transpose(0, 2, 1, 3)
        else:
            # (i in the first group, j in the second) only.  Upstream's own
            # `fint` (1.9.2 cc/ccsd_incore.py) first also writes
            # `eri[j, :, i]`, indexing the first group's axis with the second
            # group's offsets.  In upstream's use -- one group per (l, nprim),
            # never split -- its CCSD still matches PySCF (glycine/def2-SVP,
            # 8e-9 Eh, measured); with the groups split by `group_size` as
            # here, that write left entries the second one does not
            # overwrite, and glycine's (ia|jb) came out 1e-3 off
            eri[i, :, j] = eri_tmp.transpose(2, 0, 1)
        return eri

    def ao_ranges(self, group_id):
        """``(ish0, ish1, jsh0, jsh1, i0, i1, j0, j1)`` of a group pair."""
        k, l = self._keys[group_id]
        lo = self.l_ctr_offsets
        ao_loc = self.mol.ao_loc
        ish0, ish1, jsh0, jsh1 = lo[k], lo[k + 1], lo[l], lo[l + 1]
        return (ish0, ish1, jsh0, jsh1, int(ao_loc[ish0]), int(ao_loc[ish1]),
                int(ao_loc[jsh0]), int(ao_loc[jsh1]))


def fill_into(eri, strides, vhfopt, group_id):
    """Write :meth:`BlockOpt.block` into ``eri`` in whichever of the two
    layouts 1.8's ``_fill_eri_block`` was asked for: strides
    ``[1, dj*nao, dj*nao**2, nao]`` (``(i, k, j, l)``, the ladder's) or
    ``[1, di*dj*nao, dj*nao, nao]`` (``(k, i, j, l)``, ``_make_eris``')."""
    blk = vhfopt.block(group_id)
    if strides[2] > strides[1]:
        eri[...] = blk
    else:
        eri[...] = blk.transpose(1, 0, 2, 3)
    return eri
