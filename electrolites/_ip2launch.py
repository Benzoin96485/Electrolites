"""The ``ejk_int3c2e_ip2`` kernel launched as tiles (``docs/DF_WORK_LIST.md`` W29).

The density-fitted Hessian's ip2 term contracts its second-derivative
three-centre integrals with a pseudo-density inside one GPU4PySCF kernel,
``gvhf-rys/ejk_int3c2e_ip2.cu``.  Its thread block walks a list of shell
pairs against a range of auxiliary shells, ``nst = 256 / gout_stride``
``(pair, auxiliary shell)`` triples side by side -- ``min(nst, 16)``
auxiliary shells times ``nst / min(nst, 16)`` pairs -- and loops over the
rest.  Upstream (and ``fastdfhessip`` item 1) gives each block
``nsp_per_block[0]`` pairs, the *lk = 0* scheme's count (256 for an ``(ss|``
pair), and the whole auxiliary batch -- every shell of one angular momentum
and contraction pattern, up to a few hundred -- along a grid of height one.
On a 42-atom molecule that is a launch of a few hundred blocks, each of
which runs its pairs past a few hundred auxiliary shells in series, on a
card that holds one such block per SM (the kernel takes 255 registers at 256
threads): the launch is a handful of waves with a tail as long as the
heaviest block, and the heaviest blocks are the high-``l`` classes'.

This launches the same kernel, unchanged, as its own tile:

* **pairs**: per auxiliary angular momentum ``lk``, ``nst / min(nst, 16)``
  pairs per block for each ``(li, lj)`` -- what the block runs at once for
  that ``lk``, so it does not loop over pairs at all
  (``FASTDFHESSIP_IP2_PCHUNK=<n>`` fixes the count instead; ``upstream`` is
  item 1's);
* **auxiliary shells**: the batch cut into ranges of 16 shells along the
  grid's second dimension (``FASTDFHESSIP_IP2_KCHUNK``; 0 is one range, as
  upstream).

The kernel reads the pseudo-density of an auxiliary range as
``dm[pair, ao_loc[ksh0] + k * nksh + (ksh - ksh0)]`` -- component-major
*within the range it was given* -- so cutting a batch into ranges changes
the layout it expects.  :meth:`Tiles.permutation` is the column order that
puts a batch's component-major columns into range-by-range component-major
order; the caller gathers the pseudo-density's columns through it once per
batch (``nao_pair x naux`` doubles, a fraction of a millisecond).  The
Coulomb-only path (``density_auxvec`` given) indexes the auxiliary vector by
``ao_loc[ksh]`` and needs no permutation.

Every triple is computed by the same code with the same arguments; only the
assignment of triples to blocks and the order of the atomic additions into
``ejk`` change, so the result agrees with upstream to the kernel's own
run-to-run spread (its atomics are unordered either way).

``docs/DF_WORK_LIST.md`` W31 then replaced the kernel itself: by default the
entry point is ``libmyip2``'s (``electrolites/kernels/cuda/myip2``, built by
``_nvcc.ip2_library``) -- the same arguments and launch, with the light
classes' triples in registers, the contraction down to 21 sums and the
atomics reduced per block (``docs/README_fastdfhessip4.md``).
``FASTDFHESSIP_IP2_KERNEL=upstream`` puts GPU4PySCF's back; so does a
machine without ``nvcc``.
"""
import ctypes

import numpy as np
import cupy as cp

from pyscf import lib
from gpu4pyscf.df.int3c2e_bdiv import libvhf_rys
from gpu4pyscf.df.hessian.rhf import int3c2e_scheme_ip2

import os


def _env(name, default):
    return os.environ.get(name, default)


_KERNEL = {}


def kernel():
    """The ``ejk_int3c2e_ip2`` entry point to launch: W31's, unless
    ``FASTDFHESSIP_IP2_KERNEL=upstream`` or it cannot be built."""
    mode = _env('FASTDFHESSIP_IP2_KERNEL', 'w31')
    if mode not in _KERNEL:
        fn = None
        if mode != 'upstream':
            from ._nvcc import ip2_library
            path = ip2_library()
            if path is not None:
                fn = ctypes.CDLL(path).ejk_int3c2e_ip2
                fn.restype = ctypes.c_int
        if fn is None:
            fn = libvhf_rys.ejk_int3c2e_ip2
        _KERNEL[mode] = fn
    return _KERNEL[mode]


#: auxiliary shells per block, the kernel's BLOCK_SIZE
_AUX_PER_BLOCK = 16
_THREADS = 256


def pair_offsets(mol, bas_ij_cache, nsp_per_block):
    """``mol.aggregate_shl_pairs(bas_ij_cache, nsp_per_block)[1]`` built on
    the host in one transfer, and cached on the cache dict's owner by
    block size: upstream's builds one device ``arange`` per pair class and
    concatenates them, which at ten atoms costs more than the kernels it
    launches."""
    size = np.asarray(nsp_per_block)
    key = (size.tobytes(), size.shape)
    store = _OFFSETS.setdefault(id(bas_ij_cache), (bas_ij_cache, {}))[1]
    if key not in store:
        l = mol.uniq_l_ctr[:, 0]
        off = []
        sp0 = 0
        for (i, j), bas_ij in bas_ij_cache.items():
            n = len(bas_ij)
            bs = int(size) if size.ndim == 0 else int(size[l[i], l[j]])
            off.append(np.arange(sp0, sp0 + n, bs))
            sp0 += n
        off.append([sp0])
        store[key] = cp.asarray(np.concatenate(off), dtype=np.int32)
    return store[key]


def _bas_ij_idx(bas_ij_cache):
    """The pair list ``aggregate_shl_pairs`` concatenates, cached."""
    store = _OFFSETS.setdefault(id(bas_ij_cache), (bas_ij_cache, {}))[1]
    if 'idx' not in store:
        store['idx'] = cp.asarray(cp.hstack(
            [cp.asarray(b) for b in bas_ij_cache.values()]), dtype=np.int32)
    return store['idx']


#: per ``bas_ij_cache`` (held, so its id is not reused): block size -> offsets
_OFFSETS = {}


class Tiles:
    """Launch geometry of ``ejk_int3c2e_ip2`` for one ``Int3c2eOpt``.

    ``spmult`` is ``fastdfhessip`` item 1's multiplier; it is only used by
    ``FASTDFHESSIP_IP2_PCHUNK=upstream``.
    """

    def __init__(self, int3c2e_opt, omega=0., spmult=1, gout=None):
        mol = int3c2e_opt.mol
        auxmol = int3c2e_opt.auxmol
        self.mol, self.auxmol = mol, auxmol
        self.cache = int3c2e_opt.bas_ij_cache
        nsp_per_block, gout_stride, shm_size = int3c2e_scheme_ip2(omega, gout)
        gs = np.asarray(gout_stride.get() if hasattr(gout_stride, 'get')
                        else gout_stride)
        self.gout_stride = cp.asarray(gs, dtype=np.int32)
        lmax = mol.uniq_l_ctr[:, 0].max()
        laux = auxmol.uniq_l_ctr[:, 0].max()
        self.shm_size_max = int(shm_size[:laux+1, :lmax+1, :lmax+1].max())
        self.pmode = _env('FASTDFHESSIP_IP2_PCHUNK', 'tile')
        self.kchunk = int(_env('FASTDFHESSIP_IP2_KCHUNK', str(_AUX_PER_BLOCK)))
        if self.pmode == 'upstream':
            self.kchunk = 0
        nst = _THREADS // gs
        self._tile = nst // np.minimum(nst, _AUX_PER_BLOCK)
        self._agg = nsp_per_block[0] * spmult
        self._offsets = {}
        self._ksh = {}
        self.bas_ij_idx = None
        self.launches = 0

    def pair_offsets(self, lk):
        """``shl_pair_offsets`` for a launch on auxiliary shells of ``lk``."""
        if self.pmode == 'tile':
            key = int(lk)
            size = self._tile[lk]
        elif self.pmode == 'upstream':
            key = -1
            size = self._agg
        else:
            key = -2
            size = int(self.pmode)
        if self.bas_ij_idx is None:
            self.bas_ij_idx = _bas_ij_idx(self.cache)
        return pair_offsets(self.mol, self.cache, size)

    def ksh_ranges(self, ks0, ks1):
        """Auxiliary shell ranges ``[ks0, ks1)`` is cut into (sorted-aux
        shell indices, not offset by ``nbas``)."""
        if self.kchunk <= 0 or ks1 - ks0 <= self.kchunk:
            return [ks0, ks1]
        return list(range(ks0, ks1, self.kchunk)) + [ks1]

    def _ksh_gpu(self, ks0, ks1):
        key = (ks0, ks1)
        if key not in self._ksh:
            r = np.asarray(self.ksh_ranges(ks0, ks1)) + self.mol.nbas
            self._ksh[key] = cp.asarray(r, dtype=np.int32)
        return self._ksh[key]

    def permutation(self, ks0, ks1):
        """Column order for the pseudo-density of the batch ``[ks0, ks1)``
        (one angular momentum and contraction pattern): ``new[c] = old[p[c]]``
        with ``old`` component-major over the whole batch and ``new``
        component-major within each range of :meth:`ksh_ranges`.  ``None``
        when the batch is one range."""
        r = self.ksh_ranges(ks0, ks1)
        if len(r) == 2:
            return None
        aux_loc = self.auxmol.ao_loc
        nksh = ks1 - ks0
        nfk = int(aux_loc[ks0+1] - aux_loc[ks0])
        perm = []
        for s0, s1 in zip(r[:-1], r[1:]):
            s = np.arange(s0, s1) - ks0
            k = np.arange(nfk)[:, None]
            perm.append((k * nksh + s).ravel())
        return cp.asarray(np.concatenate(perm))

    def launch(self, ejk, dm, density_auxvec, envs, omega, lr_factor,
               sr_factor, ks0, ks1, aux_offset, naux):
        """One ``ejk_int3c2e_ip2`` over every shell pair and the auxiliary
        shells ``[ks0, ks1)``, which share one ``lk`` and one contraction
        pattern (the pseudo-density path's batch)."""
        self._run(ejk, dm, density_auxvec, envs, omega, lr_factor, sr_factor,
                  self._ksh_gpu(ks0, ks1), int(self.auxmol._bas[ks0, 1]),
                  aux_offset, naux)

    def launch_auxvec(self, ejk, dm, density_auxvec, envs, omega, lr_factor,
                      sr_factor, l_ctr_offsets):
        """The Coulomb-only path: every auxiliary shell, one launch per
        ``lk``, each ``l_ctr`` pattern cut into ranges."""
        off = [int(x) for x in l_ctr_offsets]
        lks = [int(self.auxmol._bas[k, 1]) for k in off[:-1]]
        for lk in sorted(set(lks)):
            idx = [i for i, l in enumerate(lks) if l == lk]
            # the patterns of one lk are adjacent in the sorted auxiliary
            # basis, so their ranges make one contiguous boundary list
            assert idx == list(range(idx[0], idx[-1] + 1))
            r = []
            for i in idx:
                r.extend(self.ksh_ranges(off[i], off[i+1])[:-1])
            r.append(off[idx[-1] + 1])
            ksh = cp.asarray(np.asarray(r) + self.mol.nbas, dtype=np.int32)
            self._run(ejk, dm, density_auxvec, envs, omega, lr_factor,
                      sr_factor, ksh, lk, 0, int(self.auxmol.nao))

    def _run(self, ejk, dm, density_auxvec, envs, omega, lr_factor, sr_factor,
             ksh, lk, aux_offset, naux):
        spo = self.pair_offsets(lk)
        self.launches += 1
        err = kernel()(
            ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            (lib.c_null_ptr() if density_auxvec is None else
             ctypes.cast(density_auxvec.data.ptr, ctypes.c_void_p)),
            ctypes.byref(envs), ctypes.c_double(omega),
            ctypes.c_double(lr_factor), ctypes.c_double(sr_factor),
            ctypes.c_int(self.shm_size_max),
            ctypes.c_int(len(spo) - 1),
            ctypes.c_int(len(ksh) - 1),
            ctypes.cast(spo.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(self._ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(aux_offset),
            ctypes.c_int(naux))
        if err != 0:
            raise RuntimeError('ejk_int3c2e_ip2 failed')

    def set_ao_pair_loc(self, ao_pair_loc):
        self._ao_pair_loc = ao_pair_loc
