"""The density-fitted Hessian's other two two-electron terms.

``docs/DF_WORK_LIST.md`` **W17** and **W18**, found by W8's anatomy of a
density-fitted Hessian (``docs/README_fastdfhess.md`` §5):

* ``df/hessian/rhf.py:_jk_energy_per_atom`` -- the ``ip2`` two-electron term
  of ``partial_hess_elec``, the **sixth** two-electron code path.  It is not
  ``df/grad/rhf.py:_jk_energy_per_atom`` (the gradient's, which
  ``fastdfgrad`` replaces) and not ``hessian/rhf.py:_partial_ejk_ip2`` (the
  direct path's, which ``fastejk2`` replaces).
* ``df/hessian/rhf.py:_get_veff`` -- the ``h1`` term of ``make_h1``, the
  **seventh**.

Both build a density-fitted quantity per atom out of the same
``Int3c2eOpt``, and ``docs/README_fastdfhessip.md`` is the round.  Every
integral, every screening decision, the metric and every symmetry factor are
GPU4PySCF's; what changes is how the work is launched and in what order it
is evaluated, so the answers agree to round-off.  The single-device path is a
copy of upstream's with the changes below, each behind a switch so that the
tables in the write-up can be re-measured:

1. **Every Rys kernel is launched one shell-pair batch deep.**  Upstream
   aggregates shell pairs at ``nsp_per_block*4`` for all four kernels these
   functions launch -- the ``ejk_int3c2e_ip2`` contraction kernel, the two
   ``fill_int3c2e_ip1``/``fill_int3c2e_ipaux`` derivative fills and the
   plain ``int3c2e`` fill -- so each thread block loops four times and the
   card runs a quarter of the blocks it could.  This is ``fastdfgrad``'s
   finding (``docs/README_fastdfgrad.md`` §4.2) on four more kernels.
   ``FASTDFHESSIP_SPMULT``.

2. **The per-atom occupied block is two Gram matrices.**
   ``_jk_energy_per_atom`` forms
   ``X[A,x,r,i,j] = sum_{u in A} j3c_100[x,r,u,i] C[u,j]``, symmetrises it
   with ``transpose_sum`` -- ``S = X + X^T`` over ``(i,j)``, on an array of
   ``3 natm naux nocc^2`` doubles, 18.7 GB at 42 atoms and 775 GB at 113,
   whose transpose is uncoalesced on one side -- and contracts
   ``sum_{r,i,j} S_A S_B / w_r``.  That sum is ``2 (T1 + T2)`` with both
   halves folded out of the atom sums through two ``nao x nao`` matrices
   (``_oo_gram``), and no array has ``natm`` in its size.
   ``FASTDFHESSIP_OOSUM=gram`` (default); ``gemm`` keeps ``X`` and builds
   ``S`` as two accumulating contractions, ``transpose`` is upstream's.

3. **The auxiliary derivative integrals are walked one time fewer.**
   ``_jk_energy_per_atom`` walks every auxiliary batch through
   ``fill_int3c2e_ipaux`` to build ``(1|00)``, and then walks them all again
   -- ``fill_int3c2e_ipaux`` *and* ``fill_int3c2e_ip1`` -- once per chunk of
   the metric dimension to build ``(10|0)``.  Nothing between the two
   passes needs anything the second pass produces, so the first is folded
   into the second pass's first chunk, unless the memory that costs would
   add a chunk.  ``FASTDFHESSIP_FUSE``.

``_get_veff`` has two more, and upstream's ``_j_energy_per_atom`` -- the
Coulomb-only ``ip2`` term a pure functional reaches -- a third:

4. **The exchange half's per-atom intermediate is hoisted.**
   ``sum_i C[u,i] dm_3c[r,i,q]`` is formed once per block for every AO
   instead of once per atom and per Cartesian direction, in the tail of the
   block loop's own buffer.  ``FASTDFHESSIP_H1ATOM``.
5. **The Coulomb half's per-atom loop is one GEMM.**  ``sum_{p in A, q}
   j3c[p,q,r] D[p,q]`` for every atom is a row-wise contraction and a GEMM
   against the atom-of-AO indicator.  ``FASTDFHESSIP_H1J``.
6. *(Measured and off.)*  The metric applied as one GEMM with ``V W^-1 V^T``
   formed once, instead of two: 1.4e-8 relative on the long-range operator,
   whose metric is ill-conditioned.  ``FASTDFHESSIP_H1METRIC=inverse``.
7. **The Coulomb-only derivative densities without the dense scatter**:
   two GEMMs against the stored pair list (``_j_energy_per_atom``).
   ``FASTDFHESSIP_JPAIR``.

``docs/DF_WORK_LIST.md`` **W24** (``docs/README_fastdfhessip2.md``) added four
more to ``_jk_energy_per_atom``:

8. **The metric as its Cholesky factor on the spherical auxiliary index.**
   Upstream's ``metric_v`` is ``C v`` with ``C`` block diagonal by shell, so
   the eigenbasis carries a Cartesian index it does not need.  With
   ``J_sph = L L^T`` the ``(10|0)`` block is held spherical and whole (the
   memory the one-chunk eigen pass already holds), D3 is ``C^T`` applied per
   integral batch, the eigenbasis in and out of D5 is two ``dtrmm`` against
   ``L^-1``, and D5's contraction is spherical in both auxiliary indices:
   upstream's three ``naux_cart x naux_sph`` contractions become two
   ``naux_sph^2`` ones (``_SphAux``, ``_chol_d4_d5``).  The factor is used in
   *every* line of the call, as ``V = C L^-T`` with ``W = 1`` wherever
   upstream's arithmetic takes ``V W^-1 V^T``: this function's terms cancel,
   and a Cholesky ``J^-1`` in some of them against an eigen ``J^-1`` in the
   others leaves the metric's round-off in the answer (2e-9 on a metric of
   condition 1.6e9).  ``FASTDFHESSIP_METRIC=eigen`` is upstream's basis;
   ``=mixed`` is that mistake, kept so that it can be measured.
9. **The ``h_aux`` block's two Gram matrices as ``dsyrk``** (``_gram``).
   ``FASTDFHESSIP_SYRK=0``.
10. **D4's ``R`` as ``dsyrk``, and ``Q`` from one GEMM** and a shared-memory
    transpose (``_oo_gram_rq_syrk``, ``_HADAMARD_T``).  ``FASTDFHESSIP_D4=gemm``
    is W17's.
11. **The ``h_aux`` block's temporaries in the integral buffers** the fused
    pass has finished with, so item 3's margin is ``10 naux^2`` instead of
    ``24``.  ``FASTDFHESSIP_CARVE=0``.

``docs/DF_WORK_LIST.md`` **W30** (``docs/README_fastdfh1occ.md``) added two to
``_get_veff``, in both spin cases:

12. **The per-atom exchange term on the occupied side** where the atoms are
    big: ``(X + X^T) C`` as two contractions per atom against intermediates
    formed once per block, ``8 nao^2 nocc`` flops per auxiliary function
    and direction instead of item 4's ``4 nao^3`` -- W25's item 12 in the
    restricted function, with the same shape score and threshold.
    ``FASTDFHESSIP_H1ATOM=auto|occ|batched``, ``FASTDFHESSIP_H1OCC_SCORE``.
14. **No exchange at all when ``k_factor = 0``** -- a pure functional's h1
    call, which upstream computes in full and scales by zero.
    ``FASTDFHESSIP_K0=compute`` puts it back.

``docs/DF_WORK_LIST.md`` **W29** (``docs/README_fastdfhessip3.md``) changed
how two of the Rys kernels are launched, and nothing inside them:

15. **``ejk_int3c2e_ip2`` launched as its own tile**
    (``electrolites/_ip2launch.py``).  Its blocks got item 1's pair count and
    a whole auxiliary batch -- a few hundred shells in series, on a card that
    holds one 255-register block per SM -- so a launch was a few waves with
    a tail as long as its heaviest block.  Now a block is as many pairs as
    it runs side by side at the batch's ``lk`` and 16 auxiliary shells, with
    the pseudo-density's columns permuted into the layout that cut implies.
    Used by ``_ejk_ip2`` (both spins) and ``_j_energy_per_atom``.
    ``FASTDFHESSIP_IP2_TILE=0``; ``FASTDFHESSIP_IP2_KCHUNK``,
    ``FASTDFHESSIP_IP2_PCHUNK`` move the tile.
16. **The two derivative fills sized the same way.**
    ``fill_int3c2e_ip1``/``fill_int3c2e_ipaux`` walk a block's (pair,
    auxiliary shell) triples as one flat list, so only the pair count needs
    changing (the auxiliary range stays the batch, and so does the output's
    layout): ``ceil(FILL_TILE nst / nksh)`` pairs per block, per ``(li, lj)``
    at the batch's ``lk``.  Every ``_ip1_evaluator`` caller -- ip2's second
    pass and first, h1's, the Coulomb-only term's -- gets it.
    ``FASTDFHESSIP_FILL_TILE=0`` is item 1's; the default is 2 passes.

``docs/DF_WORK_LIST.md`` **W31** (``docs/README_fastdfhessip4.md``) replaced
the kernel item 15 launches:

17. **``ejk_int3c2e_ip2`` rewritten** (``electrolites/kernels/cuda/myip2``,
    built by ``_nvcc.ip2_library``, chosen in ``_ip2launch.kernel``): the
    same entry point and arguments; the light classes' triples in registers
    with every index a literal, the contraction down to the 21 sums that are
    not linear combinations of the others, the rest on twice upstream's
    threads, and the atomics summed per block first.  Only under item 15's
    launch.  ``FASTDFHESSIP_IP2_KERNEL=upstream``.

What falls through to GPU4PySCF: more than one device, and ``FASTDFHESSIP_OFF``.
The unrestricted twins in ``df/hessian/uhf.py`` are ``electrolites/_dfhessip_u.py``
(``docs/DF_WORK_LIST.md`` **W25**), which this module loads and patches, under
the same switches: items 1-5, 7 and 9-11 on the spin index; not item 8.

    FASTDFHESSIP_OFF=1       hand every call back to GPU4PySCF
    FASTDFHESSIP_SPMULT=<n>  item 1: shell pairs per thread block, as a
                             multiple of the scheme's; 4 is upstream's
    FASTDFHESSIP_OOSUM=gram|gemm|transpose
                             item 2: the Gram form (default), ``X`` with the
                             sum as two contractions, or upstream's
    FASTDFHESSIP_FUSE=0      item 3 off: two ``ipaux`` passes, as upstream
    FASTDFHESSIP_H1ATOM=upstream, FASTDFHESSIP_H1J=upstream,
    FASTDFHESSIP_JPAIR=0     items 4, 5 and 7 off
    FASTDFHESSIP_H1ATOM=occ|batched, FASTDFHESSIP_H1OCC_SCORE=<s>
                             item 12 forced on or off, or its threshold
                             (default auto, 4e6)
    FASTDFHESSIP_K0=compute  item 14 off
    FASTDFHESSIP_H1METRIC=inverse
                             item 6 on
    FASTDFHESSIP_IP2_GOUT=<n>
                             the ``ip2`` kernel's ``gout_width`` (upstream's
                             ``None``; swept, +-3 %)
    FASTDFHESSIP_METRIC=chol|eigen|mixed, FASTDFHESSIP_TRI=mult|solve,
    FASTDFHESSIP_SYRK=0, FASTDFHESSIP_D4=gemm, FASTDFHESSIP_T2SPH=0,
    FASTDFHESSIP_CARVE=0     items 8-11 off, or measured variants of them
    FASTDFHESSIP_IP2_TILE=0, FASTDFHESSIP_FILL_TILE=0
                             items 15 and 16 off (item 1's launch);
    FASTDFHESSIP_IP2_KCHUNK=<n>, FASTDFHESSIP_IP2_PCHUNK=tile|upstream|<n>,
    FASTDFHESSIP_FILL_TILE=<passes>
                             their tile shapes (16, tile, 2)
    FASTDFHESSIP_IP2_KERNEL=upstream
                             item 17 off: GPU4PySCF's kernel under item 15
    FASTDFHESSIP_TIME=1      print a synchronised stage table per call; =2
                             also times the ip2 kernel per auxiliary l
    FASTDFHESSIP_RCHUNK=<n>  W17's 1.2e8-element bound on the Gram form's
                             auxiliary chunk; the default asks cuTENSOR
                             (`docs/DF_WORK_LIST.md` W21)
    FASTDFHESSIP_MEM=<bytes>, FASTDFHESSIP_VCHUNK=<n>,
    FASTDFHESSIP_OCC_BATCH=<n>
                             test-only: reach the paths upstream takes when
                             memory is short (``tests/test_dfhessip.py``)
"""
import ctypes
import os
import time as _time

import numpy as np
import cupy as cp

from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, ndarray, transpose_sum, get_avail_mem, empty_mapped,
    fill_symmetric)
from gpu4pyscf.df.int3c2e_bdiv import (
    _split_l_ctr_pattern, argsort_aux, get_ao_pair_loc, libvhf_rys,
    int2c2e, int2c2e_ip1, _check_rsh_factors, int3c2e_scheme_ip1,
    int3c2e_scheme_ipaux)
from gpu4pyscf.df.hessian import rhf as DFHESS
from gpu4pyscf.df.hessian.rhf import (
    _factorize_dm, _factorize_j2c, _bas_atom_labels, _argsort_aux_by_atom,
    _int2c2e_ip2_per_atom, int3c2e_scheme_ip2, _allocate,
    _balance_batch_size)
from gpu4pyscf.hessian.rhf import _aggregate_to_atoms
from gpu4pyscf.lib import multi_gpu

from ._dfaux import _spmult, trmm_left_lower
from . import _cuplan
from . import _ip2launch


def _env(name, default):
    return os.environ.get(name, default)


#: W17's chunk bound on the Gram form's auxiliary index, in elements of one
#: ``(3, r, nao, nao)`` slab; 0 (the default) asks cuTENSOR instead.
_RCHUNK = float(_env('FASTDFHESSIP_RCHUNK', 0))


_S = {'ip2': 0, 'h1': 0, 'declined': 0, 'fused': 0, 'vchunks': 0,
      'occ_batches': 0}


def stats():
    return dict(_S)


_orig_jk_energy_per_atom = DFHESS._jk_energy_per_atom
_orig_get_veff = DFHESS._get_veff


class _Stages:
    """Synchronised stage timers, printed when FASTDFHESSIP_TIME is set."""

    def __init__(self, tag):
        self.on = int(_env('FASTDFHESSIP_TIME', '0')) > 0
        self.tag = tag
        self.t = {}
        self.order = []
        if self.on:
            cp.cuda.Stream.null.synchronize()
        self.t0 = self.last = _time.perf_counter()

    def __call__(self, label):
        """Charge the time since the previous mark to ``label``."""
        if not self.on:
            return
        cp.cuda.Stream.null.synchronize()
        now = _time.perf_counter()
        if label not in self.t:
            self.t[label] = 0.
            self.order.append(label)
        self.t[label] += now - self.last
        self.last = now

    def note(self, **kw):
        """Sizes to print with the table, e.g. the second pass's plan."""
        if self.on:
            self.notes = {**getattr(self, 'notes', {}), **kw}

    def report(self):
        if not self.on:
            return
        if getattr(self, 'notes', None):
            print(f'[fastdfhessip {self.tag}] ' + '  '.join(
                f'{k}={v}' for k, v in self.notes.items()), flush=True)
        cp.cuda.Stream.null.synchronize()
        tot = _time.perf_counter() - self.t0
        print(f'[fastdfhessip {self.tag}] total {tot * 1e3:9.1f} ms  ' +
              '  '.join(f'{k} {self.t[k] * 1e3:.1f}' for k in self.order),
              flush=True)


def _ip1_evaluator(int3c2e_opt, scheme, batch_size, kern, omega, spmult):
    """``int3c2e_bdiv._int3c2e_ip1_evaluator`` with the aggregation exposed.

    Upstream's body with one change: ``nsp_per_block[0]*4`` becomes
    ``nsp_per_block[0]*spmult`` (item 1 of the module docstring).
    """
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    omega, lr_factor, sr_factor = _check_rsh_factors(mol, omega, None, None)
    nsp_per_block, gout_stride, shm_size = scheme
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    lmax = mol.uniq_l_ctr[:, 0].max()
    laux = auxmol.uniq_l_ctr[:, 0].max()
    shm_size_max = shm_size[:laux+1, :lmax+1, :lmax+1].max()

    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0] * spmult)
    ao_pair_loc = get_ao_pair_loc(mol.uniq_l_ctr[:, 0],
                                  int3c2e_opt.bas_ij_cache)
    nao_pair = int(ao_pair_loc[-1].get())

    l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, auxmol.uniq_l_ctr, batch_size)
    aux_sorting = argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux)
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu + mol.nbas, dtype=np.int32)
    aux_loc = auxmol.ao_loc
    aux_offsets = aux_loc[ksh_offsets_cpu]
    kernel = getattr(libvhf_rys, kern)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    npair_batches = len(shl_pair_offsets) - 1
    # item 16 (W29): pairs per block from the batch's own shape
    fill_tile = float(_env('FASTDFHESSIP_FILL_TILE', '2'))
    if fill_tile > 0:
        nst = 256 // np.asarray(gout_stride.get())
        tiled = {}

    def evaluate_j3c(batch_id, out=None):
        ksh0 = ksh_offsets_cpu[batch_id]
        ksh1 = ksh_offsets_cpu[batch_id + 1]
        aux_ao_offset = aux_loc[ksh0]
        naux = aux_loc[ksh1] - aux_ao_offset
        out = ndarray((3, nao_pair, naux), buffer=out)
        if out.size == 0:
            return out
        spo, nb = shl_pair_offsets, npair_batches
        if fill_tile > 0:
            # The fill kernels walk a block's (pair, auxiliary shell)
            # triples as one flat list, nst_per_block at a time, and the
            # auxiliary range is the whole batch (its layout is
            # component-major over the batch, so the range cannot be cut
            # without permuting the output).  So the pairs are what is
            # sized: enough per block for `fill_tile` passes of the
            # block's threads, per (li, lj) at this batch's lk.
            lk = int(auxmol._bas[ksh0, 1])
            nksh = int(ksh1 - ksh0)
            key = (lk, nksh)
            if key not in tiled:
                size = np.maximum(1, np.ceil(
                    nst[lk] * fill_tile / nksh)).astype(int)
                tiled[key] = _ip2launch.pair_offsets(
                    mol, int3c2e_opt.bas_ij_cache, size)
            spo = tiled[key]
            nb = len(spo) - 1
        err = kernel(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.byref(int3c2e_envs),
            ctypes.c_double(omega),
            ctypes.c_double(lr_factor), ctypes.c_double(sr_factor),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(nb),
            ctypes.c_int(1),
            ctypes.cast(spo.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets_gpu[batch_id:].data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(0),
            ctypes.c_int(aux_ao_offset),
            ctypes.c_int(nao_pair),
            ctypes.c_int(naux))
        if err != 0:
            raise RuntimeError(f'{kern} failed')
        return out
    return evaluate_j3c, aux_sorting, aux_offsets


def _avail():
    """Free device memory, as upstream reads it -- or less, for the tests.

    ``FASTDFHESSIP_MEM=<bytes>`` caps it.  Nothing in a patched run sets it:
    it exists so that ``tests/test_dfhessip.py`` can drive the occupied
    batching, the host-resident ``dm_oo`` and the multi-chunk second pass
    on a molecule small enough to run in seconds, all of which upstream only
    takes when memory is short.
    """
    mem = get_avail_mem(exclude_memory_pool=True)
    cap = _env('FASTDFHESSIP_MEM', '')
    return min(mem, int(float(cap))) if cap else mem


def _atom_indicator(aoslices):
    """``(natm, nao)``, 1 where the AO is on the atom.

    The three-centre integrals are in the original AO order, in which each
    atom's AOs are contiguous (``aoslices``); ``_bas_atom_labels`` is in the
    sorted molecule's shell order and only its counts carry over.
    """
    natm = len(aoslices) - 1
    ao_atm = np.repeat(np.arange(natm), np.diff(aoslices))
    return cp.asarray(
        (np.arange(natm)[:, None] == ao_atm[None, :]).astype(np.float64))


def _forced_occ_batch(n):
    """``FASTDFHESSIP_OCC_BATCH=<n>``: occupied orbitals per batch, for the
    tests -- the batched path is otherwise only reached when memory is short
    on a molecule with many more occupied orbitals than a test can afford."""
    forced = int(_env('FASTDFHESSIP_OCC_BATCH', '0'))
    return min(n, forced) if forced > 0 else n


def _usable():
    if _env('FASTDFHESSIP_OFF', ''):
        return False
    return multi_gpu.num_devices == 1


# --------------------------------------------------------------------------
# W17: the ip2 term of partial_hess_elec
# --------------------------------------------------------------------------
def _jk_energy_per_atom(int3c2e_opt, dm, j_factor=1, k_factor=1, omega=None,
                        verbose=None):
    if not _usable():
        _S['declined'] += 1
        # through the module attribute, so `benchmarks/coverage_probe.py`
        # can count a decline
        return DFHESS._jk_energy_per_atom_orig(int3c2e_opt, dm, j_factor,
                                               k_factor, omega, verbose)
    if k_factor == 0:
        # upstream's own branch for this: a pure functional's ip2 term
        return _j_energy_per_atom(int3c2e_opt, dm, verbose) * j_factor
    _S['ip2'] += 1
    spmult = int(_env('FASTDFHESSIP_SPMULT', '1'))
    oosum = _env('FASTDFHESSIP_OOSUM', 'gram')
    fuse = _env('FASTDFHESSIP_FUSE', '1') != '0'
    metric = _env('FASTDFHESSIP_METRIC', 'chol')
    st = _Stages('ip2')

    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    omega, lr_factor, sr_factor = _check_rsh_factors(mol, omega, None, None)

    dm_factor_l = _factorize_dm(mol, dm)
    nao, nocc = dm_factor_l.shape

    natm = mol.natm
    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=True)[0]
    pair_addresses = cp.asarray(pair_addresses, dtype=np.int32)
    nao_pair = len(pair_addresses)
    aux_loc = auxmol.ao_loc
    naux = int(aux_loc[-1])
    largest_shell_nao = int((aux_loc[1:] - aux_loc[:-1]).max())

    cp.get_default_memory_pool().free_all_blocks()
    mem_avail = _avail()
    word_avail = mem_avail // 8
    num_occ_batches = int(nocc*nocc*naux*4*8 / (mem_avail*0.4) + 1)
    nocc_per_batch = (nocc + num_occ_batches - 1) // num_occ_batches
    nocc_per_batch = min(nocc, (nocc_per_batch + 7) & 0xffffff8)
    nocc_per_batch = _forced_occ_batch(nocc_per_batch)
    num_occ_batches = (nocc + nocc_per_batch - 1) // nocc_per_batch
    _S['occ_batches'] = num_occ_batches

    word_avail -= 4 * naux * nocc * nocc_per_batch  # j3c_oo1, dm_oo
    word_avail -= 6 * naux**2  # j2c_10v and metric_v
    batch_size = int(word_avail * 0.02) // nao_pair
    batch_size = max(largest_shell_nao, min(batch_size, naux))
    with _spmult(spmult, short_range=omega < 0):
        eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
            aux_batch_size=batch_size, reorder_aux=True, cart=True,
            omega=omega)
    batch_size = min(batch_size, int((aux_offsets[1:]-aux_offsets[:-1]).max()))
    num_aux_batches = len(aux_offsets) - 1

    blksize = min(naux, int(word_avail * 0.5) // (nao**2*8) * 8)
    assert blksize > 0, 'Insufficient GPU memory'
    blksize = min(blksize, batch_size)
    st('setup')

    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((batch_size, nao_pair))
    buf1 = cp.empty((blksize, nocc, nao))
    j3c_oo = cp.empty((naux, nocc, nocc))
    for kbatch in range(num_aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:, :, :dk]
            fill_symmetric(compressed, pair_addresses, nao, k0, k1, out=j3c)
            tmp = ndarray((nocc, nao, dk), buffer=buf1)
            contract('pqr,pi->iqr', j3c, dm_factor_l, out=tmp)
            contract('iqr,qj->rij', tmp, dm_factor_l, out=j3c_oo[aux0:aux1])
    j3c_full = buf = buf1 = eval_j3c = j3c = tmp = compressed = None
    st('A j3c_oo')

    # item 8: the metric as its Cholesky factor, everywhere in this call.
    # Upstream only ever uses `V W^-1 V^T`, so `V = C L^-T` with `W = 1`
    # serves every line of it unchanged -- and it has to serve *every* line:
    # the terms of this function cancel to a part in 1e9 on an ill-conditioned
    # metric, and a Cholesky J^-1 in one term against an eigen J^-1 in
    # another leaves 2e-9 of the metric's round-off in the answer (W24).
    # (`FASTDFHESSIP_METRIC=mixed` is that mistake, kept so it can be
    # measured: the factor in D3-D5 and upstream's eigenbasis everywhere
    # else -- `benchmarks/df_hess_metric_sens.py`.)
    sa = None
    if (metric in ('chol', 'mixed') and omega == 0 and num_occ_batches == 1
            and oosum == 'gram' and not auxmol.mol.cart):
        try:
            sa = _SphAux(auxmol, aux_sorting, omega,
                         _bas_atom_labels(auxmol, aux_sorting))
        except RuntimeError:
            sa = None
    if sa is not None and metric == 'chol':
        metric_v = sa.factor()
        metric_w = cp.ones(sa.nsph)
        _S['chol'] = _S.get('chol', 0) + 1
    else:
        metric_w, metric_v = _factorize_j2c(auxmol, aux_sorting, omega)
        if sa is not None and metric_v.shape[1] != sa.nsph:
            sa = None
    metric_size = metric_v.shape[1]
    dm_oo = contract('uv,uij->vij', metric_v, j3c_oo)
    dm_oo /= metric_w[:, None, None]
    dm_oo = contract('uv,vij->uij', metric_v, dm_oo, out=j3c_oo)
    j3c_oo = None
    if j_factor != 0:
        auxvec = dm_oo.trace(axis1=1, axis2=2)

    dm_oo_full = dm_oo
    if num_occ_batches > 1:
        dm_oo_full = empty_mapped(dm_oo.shape)
        dm_oo.get(out=dm_oo_full)
    dm_oo = None
    st('B metric')

    # (00|0)(2|0)(0|00)
    ejk = _ejk_ip2(int3c2e_opt, dm_factor_l, dm_oo_full, auxvec if j_factor
                   else None, j_factor, k_factor, omega, lr_factor,
                   sr_factor, pair_addresses, nao_pair, batch_size, blksize,
                   spmult, st)
    ejk = ejk + ejk.transpose(1, 0, 3, 2)
    ejk *= 2 * 2 * .5
    st('C ip2 kernel')

    atm_ao_counts = np.bincount(_bas_atom_labels(mol))
    aoslices = np.append(0, atm_ao_counts.cumsum())
    nao_on_atom = atm_ao_counts.max()
    aux_atm_labels = _bas_atom_labels(auxmol, aux_sorting)
    atm_ind = _atom_indicator(aoslices)

    i_addr, j_addr = divmod(pair_addresses, nao)
    _aux_sorting = cp.asarray(aux_sorting)
    _metric_v = metric_v
    _metric_w = metric_w
    _dm_factor_l = dm_factor_l
    if j_factor != 0:
        dm = _dm_factor_l.dot(_dm_factor_l.T)
        _auxvec = auxvec

    eval_ipaux = _ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ipaux(omega, 27), batch_size,
        'fill_int3c2e_ipaux', omega, spmult)[0]
    eval_ip1 = _ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ip1(omega, 27), batch_size,
        'fill_int3c2e_ip1', omega, spmult)[0]

    if j_factor != 0:
        auxvec_ipauxp = cp.empty((3, naux))

    h_ao_aux = cp.zeros((natm, naux, 3, 3))
    ejk_ao = cp.zeros((natm, natm, 3, 3))
    h_sph = None

    for batch_id in range(num_occ_batches):
        nocc0 = batch_id * nocc_per_batch
        nocc1 = min(nocc, nocc0 + nocc_per_batch)
        nocc_in_batch = nocc1 - nocc0

        dm_oo = cp.asarray(dm_oo_full[:, nocc0:nocc1])
        j3c_oo1 = cp.empty((3, naux, nocc_in_batch, nocc))  # = (1|00)

        # The sizes the second pass is going to use, worked out before the
        # first so that item 3 can tell whether there will be one chunk.
        # Upstream derives them from the memory free *after* the first pass
        # and the h_aux block; the arithmetic below is upstream's, applied to
        # the same `word_avail` those steps leave (they free what they
        # allocate), so the chunking is the same as upstream's.
        mem_avail = _avail()
        word_avail = mem_avail // 8
        blksize1 = min(naux, word_avail // (nao**2*8) * 8)
        assert blksize1 > 0, 'Insufficient GPU memory'
        blksize1 = min(blksize1, batch_size)

        plan = _plan_v(word_avail, naux, nao, nocc, nocc_in_batch, nao_pair,
                       batch_size, nao_on_atom, natm, metric_size, batch_id)
        do_fuse = False
        plan_f = (None,)
        if fuse:
            # Fused, the first pass rides on the second pass's *first*
            # chunk and the h_aux block runs after that chunk's integrals,
            # while its buffers are held rather than before they are
            # allocated -- so the plan is made against what that block's
            # temporaries leave, about 24 naux^2 words at their peak, and
            # taken only if that does not cost the second pass a chunk: a
            # chunk is a whole walk over the integrals, and the fusion saves
            # exactly one.
            # Item 11: the block's naux^2 temporaries can live in the
            # second pass's integral buffers, which the walk is done with by
            # then and which are at least 6 batch_size nao_pair words however
            # the pass is planned -- so where that covers them, the margin is
            # what the block still allocates (j2c_10, its sorted copy and
            # j2c_10v), not the 24 naux^2 of all of it.
            carve = (_env('FASTDFHESSIP_CARVE', '1') != '0' and
                     6 * batch_size * nao_pair >=
                     _h_aux_words(naux, metric_size))
            margin = (10 if carve else 24) * naux**2
            try:
                plan_f = _plan_v(word_avail - margin, naux, nao, nocc,
                                 nocc_in_batch, nao_pair, batch_size,
                                 nao_on_atom, natm, metric_size, batch_id)
            except AssertionError:
                plan_f = (0,)
            if plan_f[0] > 0 and (-(-metric_size // plan_f[0]) <=
                                  -(-metric_size // plan[0])):
                plan, do_fuse = plan_f, True
        aux_batch_size, aux_blksize, blksize2 = plan
        # item 8's structured second pass needs the whole auxiliary index in
        # its one chunk; otherwise the factor goes through upstream's chunked
        # arithmetic like an eigenbasis would
        chol = sa if (sa is not None and aux_batch_size >= metric_size) \
            else None
        if sa is not None and chol is None:
            _S['chol_chunked'] = _S.get('chol_chunked', 0) + 1
        st.note(free_gb=round(word_avail * 8e-9, 1), plan=plan[0],
                plan_fused=plan_f[0])
        st.note(chol=chol is not None, nao=nao, nocc=nocc, naux=naux, metric_size=metric_size,
                nao_pair=nao_pair, batch_size=batch_size,
                num_aux_batches=num_aux_batches, vchunk=aux_batch_size,
                blksize2=blksize2, fused=do_fuse, occ_batches=num_occ_batches)
        _S['vchunks'] = (metric_size + aux_batch_size - 1) // aux_batch_size
        if do_fuse:
            _S['fused'] += 1

        if not do_fuse:
            _ipaux_pass(eval_ipaux, num_aux_batches, blksize1, nao, nao_pair,
                        batch_size, nocc_in_batch, pair_addresses,
                        _dm_factor_l, nocc0, nocc1, j3c_oo1,
                        auxvec_ipauxp if (j_factor != 0 and batch_id == 0)
                        else None, dm if j_factor != 0 else None)
            st('D1 ipaux pass')
            j3c_oo1p, h_aux_parts = _h_aux_block(
                auxmol, omega, _aux_sorting, _metric_v, _metric_w, dm_oo,
                j3c_oo1, naux, metric_size, k_factor, j_factor, batch_id,
                auxvec_ipauxp if j_factor != 0 else None,
                _auxvec if j_factor != 0 else None, natm, aux_atm_labels)
            ejk_ao += h_aux_parts[0]
            j2c_10v = h_aux_parts[1]
            j2c_10c = _j2c_10c(chol, h_aux_parts[2])
            h_aux_parts = None
            if chol is not None:
                j2c_10v = None
            st('D2 h_aux')
        else:
            j3c_oo1p = j3c_oo1
            j2c_10v = j2c_10c = None

        # the second pass: (10|0), and with item 3 the (1|00) of the first
        if batch_id == 0:
            j3c_buf = cp.empty((3, aux_batch_size, nao, nocc))
        else:
            j3c_buf = cp.empty((3, aux_batch_size, nao, nocc_in_batch))
        aux_unit2 = natm*3*nocc*nocc*2 if batch_id == 0 else 0
        work = cp.empty(
            max(aux_unit2*aux_blksize,
                3*nao_pair*batch_size * 2 + (nocc+nao)*nao * blksize2,
                3*naux*nao_on_atom*nocc_in_batch * 2,
                3*naux*nao_on_atom*nocc_in_batch + 3*aux_batch_size*naux))
        for v0, v1 in lib.prange(0, metric_size, aux_batch_size):
            dv = v1 - v0
            if batch_id == 0:
                j3c_100 = ndarray((3, dv, nao, nocc), buffer=j3c_buf)
            else:
                j3c_100 = ndarray((3, dv, nao, nocc_in_batch), buffer=j3c_buf)
            j3c_100[:] = 0.
            j3c_full, work1 = _allocate((nao, nao, blksize2), work)
            j3c_full[:] = 0.
            buf0, work1 = _allocate((3, nao_pair, batch_size), work1)
            buf1, work1 = _allocate((3, nao_pair, batch_size), work1)
            # the second pass's buffers, charged to a stage of their own
            # (W29: they were half of what 'D3 integrals' reported)
            st('D3 buffers')
            aux0 = aux1 = 0
            for kbatch in range(num_aux_batches):
                compressed_di = eval_ip1(kbatch, out=buf0)
                compressed_dk = eval_ipaux(kbatch, out=buf1)
                naux_in_batch = compressed_di.shape[-1]
                st('D3 integrals')
                if do_fuse and v0 == 0:
                    # item 3: upstream's first pass, on this batch's
                    # ipaux integrals before they are folded into dj
                    _ipaux_batch(compressed_dk, aux1, naux_in_batch,
                                 blksize2, j3c_full, work1, nao,
                                 nocc_in_batch, pair_addresses, _dm_factor_l,
                                 nocc0, nocc1, j3c_oo1,
                                 auxvec_ipauxp if (j_factor != 0 and
                                                   batch_id == 0) else None,
                                 dm if j_factor != 0 else None)
                    st('D1 ipaux (fused)')
                compressed_dk += compressed_di
                compressed_dj = compressed_dk
                compressed_di *= -1
                for k0, k1 in lib.prange(0, naux_in_batch, blksize2):
                    dk = k1 - k0
                    aux0, aux1 = aux1, aux1 + dk
                    j3c = j3c_full[:, :, :dk]
                    if batch_id == 0:
                        tmp = ndarray((nao, nocc, dk), buffer=work1)
                    else:
                        tmp = ndarray((nao, nocc_in_batch, dk), buffer=work1)
                    for x in range(3):
                        j3c[j_addr, i_addr] = compressed_dj[x, :, k0:k1]
                        j3c[i_addr, j_addr] = compressed_di[x, :, k0:k1]
                        if batch_id == 0:
                            contract('pqr,qi->pir', j3c, _dm_factor_l, out=tmp)
                        else:
                            contract('pqr,qi->pir', j3c,
                                     _dm_factor_l[:, nocc0:nocc1], out=tmp)
                        if chol is None:
                            contract('rs,pir->spi',
                                     _metric_v[aux0:aux1, v0:v1], tmp,
                                     beta=1, out=j3c_100[x])
                        else:
                            # item 8: the spherical (10|0), one shell block
                            # at a time -- a pass over memory, not a GEMM
                            c0, c1 = chol.cols(aux0, aux1)
                            contract('rs,pir->spi', chol.ms[aux0:aux1, c0:c1],
                                     tmp, beta=1, out=j3c_100[x, c0:c1])
                st('D3 j3c_100')
            j3c_full = buf0 = buf1 = None
            compressed_di = compressed_dj = compressed_dk = None

            if do_fuse and v0 == 0:
                # the h_aux block that upstream runs between the passes, in
                # the integral buffers the walk has finished with
                j3c_oo1p, h_aux_parts = _h_aux_block(
                    auxmol, omega, _aux_sorting, _metric_v, _metric_w, dm_oo,
                    j3c_oo1, naux, metric_size, k_factor, j_factor, batch_id,
                    auxvec_ipauxp if j_factor != 0 else None,
                    _auxvec if j_factor != 0 else None, natm, aux_atm_labels,
                    work=work if carve else None)
                ejk_ao += h_aux_parts[0]
                j2c_10v = h_aux_parts[1]
                j2c_10c = _j2c_10c(chol, h_aux_parts[2])
                h_aux_parts = None
                if chol is not None:
                    j2c_10v = None
                st('D2 h_aux')

            if chol is not None:
                h_sph = _chol_d4_d5(
                    chol, j3c_100, work, natm, nocc, atm_ind, aoslices,
                    _dm_factor_l, dm_oo, j3c_oo1p, j2c_10c, k_factor,
                    j_factor, ejk_ao, h_ao_aux,
                    _auxvec if j_factor != 0 else None,
                    auxvec_ipauxp if j_factor != 0 else None, st)
                continue
            if batch_id == 0 and oosum == 'gram':
                _oo_gram(j3c_100, v0, dv, work, natm, nocc, atm_ind,
                         _dm_factor_l, _metric_w, _metric_v, k_factor,
                         j_factor, ejk_ao, h_ao_aux, j2c_10v,
                         _auxvec if j_factor != 0 else None,
                         auxvec_ipauxp if j_factor != 0 else None, naux)
                st('D4 oo_atm')
            elif batch_id == 0:
                for k0, k1 in lib.prange(0, dv, aux_blksize):
                    _oo_atm_block(j3c_100, k0, k1, v0, work, natm, nocc,
                                  aoslices, _dm_factor_l, _metric_w,
                                  _metric_v, k_factor, j_factor, ejk_ao,
                                  h_ao_aux, j2c_10v,
                                  _auxvec if j_factor != 0 else None,
                                  auxvec_ipauxp if j_factor != 0 else None,
                                  naux, oosum)
                st('D4 oo_atm')

            if batch_id == 0:
                j3c_100 = j3c_100[:, :, :, nocc0:nocc1]
            for i, (p0, p1) in enumerate(zip(aoslices[:-1], aoslices[1:])):
                tmp0, work1 = _allocate((3, naux, p1-p0, nocc_in_batch), work)
                tmp1, work1 = _allocate((3, naux, p1-p0, nocc_in_batch), work1)
                contract('xruj,sr->xsuj', j3c_100[:, :, p0:p1],
                         _metric_v[:, v0:v1]/_metric_w[v0:v1], out=tmp0)
                contract('ysji,ui->ysuj', j3c_oo1p, _dm_factor_l[p0:p1],
                         out=tmp1)
                contract('xsuj,ysuj->sxy', tmp0, tmp1, -.5*k_factor*2, 1,
                         h_ao_aux[i])
                tmp0, work1 = _allocate((naux, p1-p0, nocc_in_batch), work)
                tmp1, work1 = _allocate((3, naux, v1-v0), work1)
                contract('sji,ui->suj', dm_oo, _dm_factor_l[p0:p1], out=tmp0)
                contract('suj,xruj->xsr', tmp0, j3c_100[:, :, p0:p1],
                         out=tmp1)
                tmp1 /= _metric_w[v0:v1]
                contract('xsr,ysr->sxy', tmp1, j2c_10v[:, :, v0:v1],
                         .5*k_factor*2, 1, h_ao_aux[i])
            st('D5 per-atom')
        j3c_buf = work = work1 = tmp0 = tmp1 = None
        j3c = j3c_100 = None
        j2c_10v = j2c_10c = j3c_oo1p = j3c_oo1 = None

    ejk_ao_aux = _aggregate_to_atoms(h_ao_aux, natm, aux_atm_labels, axis=1)
    if h_sph is not None:
        ejk_ao_aux += h_sph
    ejk += ejk_ao_aux
    ejk += ejk_ao_aux.transpose(1, 0, 3, 2)
    ejk += ejk_ao
    st('E aggregate')
    st.report()
    log.timer_debug1('_jk_energy_per_atom (fastdfhessip)', *t0)
    return ejk


def _j_energy_per_atom(int3c2e_opt, dm, verbose=None):
    """``df/hessian/rhf.py:_j_energy_per_atom`` -- the Coulomb-only ip2 term.

    What a *pure* functional's ``partial_hess_elec`` reaches (upstream's
    ``_jk_energy_per_atom`` hands ``k_factor == 0`` straight to it).  Two
    changes, both behind switches:

    * item 1's launch configuration on its three Rys kernels;
    * **item 7, the per-atom derivative densities without the dense
      scatter.**  Upstream scatters each block of ``(10|0)`` integrals from
      the stored pair list into a dense ``(3, nao, nao, blk)`` array --
      ``j3c[j,i] = dj``, then ``j3c[i,j] = di`` -- and contracts it with the
      density one atom at a time, ``2 natm`` launches per block.  The same
      numbers are two GEMMs against the pair list itself:

          sum_{p in A, q} j3c[x,p,q,r] D[p,q]
              = sum_e S_i[A,e] di[x,e,r] + sum_e S_j[A,e] dj[x,e,r]

      with ``S_i[A,e] = D[i_e,j_e] [atom(i_e) = A]`` and
      ``S_j[A,e] = D[j_e,i_e] [atom(j_e) = A]`` over the stored pairs ``e``,
      except that ``S_j`` is zero on the pairs inside a diagonal shell block:
      there both orderings of an AO pair are stored, and the second scatter
      (``di``) overwrites what the first wrote, so ``dj`` never reaches the
      dense array.  ``FASTDFHESSIP_JPAIR``.
    """
    spmult = int(_env('FASTDFHESSIP_SPMULT', '1'))
    jpair = _env('FASTDFHESSIP_JPAIR', '1') != '0'
    st = _Stages('ip2 J')
    _S['ip2_j'] = _S.get('ip2_j', 0) + 1
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    dm = mol.apply_C_mat_CT(dm)
    auxvec = int3c2e_opt.contract_dm(dm, hermi=1)
    naux = len(auxvec)

    metric_w, metric_v = _factorize_j2c(auxmol)
    auxvec = (metric_v).dot(metric_v.T.dot(auxvec) / metric_w)

    dm_aux = auxvec[:, None] * auxvec
    ej_aux = _int2c2e_ip2_per_atom(auxmol, dm_aux)
    st('A auxvec')

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme_ip2()
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    lmax = mol.uniq_l_ctr[:, 0].max()
    laux = auxmol.uniq_l_ctr[:, 0].max()
    shm_size_max = shm_size[:laux+1, :lmax+1, :lmax+1].max()
    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0] * spmult)
    ao_pair_loc = get_ao_pair_loc(mol.uniq_l_ctr[:, 0],
                                  int3c2e_opt.bas_ij_cache)
    ksh_offsets_cpu = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu + mol.nbas, dtype=np.int32)
    int3c2e_envs = int3c2e_opt.int3c2e_envs
    ej = cp.zeros_like(ej_aux)
    tiles = _ip2_tiles(int3c2e_opt, 0., spmult, None, ao_pair_loc)
    if tiles is not None:
        # item 15 (W29): one launch per auxiliary l, as tiles
        tiles.launch_auxvec(ej, dm, auxvec, int3c2e_envs, 0., 1., 1.,
                            ksh_offsets_cpu)
        err = 0
    else:
        err = libvhf_rys.ejk_int3c2e_ip2(
            ctypes.cast(ej.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            ctypes.cast(auxvec.data.ptr, ctypes.c_void_p),
            ctypes.byref(int3c2e_envs), ctypes.c_double(0.),
            ctypes.c_double(1.), ctypes.c_double(1.),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.c_int(len(ksh_offsets_cpu) - 1),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets_gpu.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(0),
            ctypes.c_int(naux))
    if err != 0:
        raise RuntimeError('ejk_int3c2e_ip2 failed')
    ej = ej + ej.transpose(1, 0, 3, 2)
    ej *= 2 * 2 * .5
    ej -= ej_aux
    ej_aux = None
    st('B ip2 kernel')

    natm = mol.natm
    nao = dm.shape[-1]
    pair_addresses, diag_idx = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=True)
    pair_addresses = cp.asarray(pair_addresses, dtype=np.int32)
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)

    ao_idx = mol.get_ao_idx(cart=True)
    dm = dm[ao_idx[:, None], ao_idx]
    dm_compressed = dm[i_addr, j_addr]
    dm_compressed[diag_idx] *= .5
    dm_compressed *= 2

    mem_avail = _avail()
    word_avail = mem_avail // 8
    batch_size = min(naux, int(word_avail*0.75) // (3*nao_pair*2))
    blksize = min(naux, int(word_avail*0.15) // (nao**2*8) * 8)
    assert batch_size > 0 and blksize > 0

    omega = 0
    eval_ipaux, aux_sorting, aux_offsets = _ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ipaux(omega, 27), batch_size,
        'fill_int3c2e_ipaux', omega, spmult)
    eval_ip1 = _ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ip1(omega, 27), batch_size,
        'fill_int3c2e_ip1', omega, spmult)[0]
    num_aux_batches = len(aux_offsets) - 1

    atm_ao_counts = np.bincount(_bas_atom_labels(mol))
    aoslices = np.append(0, atm_ao_counts.cumsum())

    auxvec_ipaux = cp.empty((3, naux))
    auxvec_100_atm = cp.empty((3, natm, naux))
    buf0 = cp.empty((3, nao_pair, batch_size))
    buf1 = cp.empty((3, nao_pair, batch_size))
    if jpair:
        ao_atm = cp.asarray(np.repeat(np.arange(natm), np.diff(aoslices)))
        dm_ij = dm[i_addr, j_addr]
        s_i = cp.zeros((natm, nao_pair))
        s_i[ao_atm[i_addr], cp.arange(nao_pair)] = dm_ij
        not_diag = cp.ones(nao_pair, dtype=bool)
        not_diag[cp.asarray(diag_idx)] = False
        s_j = cp.zeros((natm, nao_pair))
        s_j[ao_atm[j_addr], cp.arange(nao_pair)] = dm[j_addr, i_addr] * not_diag
        dm_ij = not_diag = None
    else:
        j3c_full = cp.zeros((3, nao, nao, blksize))
    aux0 = aux1 = 0
    for kbatch in range(num_aux_batches):
        compressed_di = eval_ip1(kbatch, out=buf0)
        compressed_dk = eval_ipaux(kbatch, out=buf1)
        aux0, aux1 = aux1, aux1 + compressed_dk.shape[-1]
        st('C integrals')
        auxvec_ipaux[:, aux0:aux1] = contract('xpr,p->xr', compressed_dk,
                                              dm_compressed)
        compressed_dk += compressed_di
        compressed_dj = compressed_dk
        compressed_di *= -1
        naux_in_batch = compressed_di.shape[-1]
        if jpair:
            for x in range(3):
                t = s_i.dot(compressed_di[x])
                t += s_j.dot(compressed_dj[x])
                auxvec_100_atm[x, :, aux0:aux1] = t
            t = None
        else:
            _aux1 = aux0
            for k0, k1 in lib.prange(0, naux_in_batch, blksize):
                dk = k1 - k0
                _aux0, _aux1 = _aux1, _aux1 + dk
                j3c = j3c_full[:, :, :, :dk]
                j3c[:, j_addr, i_addr] = compressed_dj[:, :, k0:k1]
                j3c[:, i_addr, j_addr] = compressed_di[:, :, k0:k1]
                for i, (p0, p1) in enumerate(zip(aoslices[:-1],
                                                 aoslices[1:])):
                    contract('xpqr,pq->xr', j3c[:, p0:p1], dm[p0:p1],
                             out=auxvec_100_atm[:, i, _aux0:_aux1])
        st('C per-atom')
    auxvec_ipaux = auxvec_ipaux[:, aux_sorting]
    auxvec_100_atm = auxvec_100_atm[:, :, aux_sorting]
    auxvec_100_atm *= 2
    buf0 = buf1 = compressed_di = compressed_dj = compressed_dk = None
    j3c_full = s_i = s_j = None

    j2c_10 = int2c2e_ip1(auxmol, sort_output=False)
    j2c_10 *= -1
    j2c_10v = contract('xrs,st->xrt', j2c_10, metric_v)
    j2c_10v_w = j2c_10v / metric_w
    j2c_ip2 = contract('xrs,yts->rtxy', j2c_10v_w, j2c_10v)
    j2c_ip2 *= dm_aux[:, :, None, None]
    h_aux = j2c_ip2
    tmp = dm_aux = None

    auxvec_ipaux *= -1
    auxvec_ipauxp = contract('xuv,v->xu', j2c_10, auxvec, alpha=-1, beta=1,
                             out=auxvec_ipaux)
    dm_aux1 = cp.einsum('xr,t->xrt', auxvec_ipauxp, auxvec)
    tmp = contract('xrt,st->xrs', j2c_10v_w, metric_v)
    w10_100 = contract('xrt,ytr->rtxy', tmp, dm_aux1)
    h_aux -= w10_100
    h_aux -= w10_100.transpose(1, 0, 3, 2)
    dm_aux1 = j2c_10v = j2c_10v_w = w10_100 = tmp = None

    dm_aux11 = cp.einsum('xr,ys->rsxy', auxvec_ipauxp, auxvec_ipauxp)
    j2c_inv = (metric_v / metric_w).dot(metric_v.T)
    dm_aux11 *= j2c_inv[:, :, None, None]
    h_aux += dm_aux11
    h_aux = h_aux + h_aux.transpose(1, 0, 3, 2)

    aux_atm_labels = _bas_atom_labels(auxmol)
    ej_aux = _aggregate_to_atoms(h_aux, natm, aux_atm_labels, axis=(0, 1))
    ej += ej_aux * .5
    dm_aux11 = h_aux = j2c_inv = None

    tmp = contract('xpr,rs->xps', auxvec_100_atm, metric_v)
    tmp /= metric_w
    auxvec_100v_atm = contract('xps,rs->xpr', tmp, metric_v)
    ej_ao = contract('xpr,yqr->pqxy', auxvec_100v_atm, auxvec_100_atm)
    ej += ej_ao

    h_ao_aux = contract('xpr,yr->xypr', auxvec_100v_atm, auxvec_ipauxp)
    j2c_10 *= auxvec[:, None]
    contract('xpt,yrt->xypr', auxvec_100v_atm, j2c_10, alpha=-1, beta=1,
             out=h_ao_aux)
    ej_ao_aux = _aggregate_to_atoms(h_ao_aux.transpose(2, 3, 0, 1), natm,
                                    aux_atm_labels, axis=1)
    ej += ej_ao_aux
    ej += ej_ao_aux.transpose(1, 0, 3, 2)
    st('D metric')
    st.report()
    log.timer_debug1('_j_energy_per_atom (fastdfhessip)', *t0)
    return ej


class _SphAux:
    """The auxiliary index in spherical functions, for item 8 (W24).

    The derivative integrals are Cartesian in the auxiliary index and so is
    everything upstream builds from them; the metric is spherical, and
    ``metric_v`` is ``C v`` with ``C`` the Cartesian-to-spherical
    coefficients (``_factorize_j2c``: ``auxmol.apply_C_dot``).  ``C`` has a
    block per shell, so applying it is a pass over memory, not a GEMM.

    ``ms`` is ``C`` with its rows in the integrals' order (``aux_sorting``)
    and its columns permuted so that the rows of consecutive shells map to
    consecutive columns; ``lo``/``hi`` are each row's column range, and
    ``blocks`` a partition of the rows into runs whose column ranges do not
    overlap, for the block-diagonal apply.  ``linv`` is the inverse of the
    Cholesky factor of the spherical metric in the same column order, or
    the constructor raises ``RuntimeError`` if the metric is not positive
    definite.
    """

    def __init__(self, auxmol, aux_sorting, omega, aux_atm_labels,
                 block=256):
        nsph = auxmol.mol.nao
        mc0 = auxmol.apply_C_dot(cp.eye(nsph))
        ncart = mc0.shape[0]
        mc = cp.empty_like(mc0)
        mc[cp.asarray(aux_sorting)] = mc0
        mc0 = None
        nz = cp.asnumpy(mc != 0)
        # columns in the order of the first row that reaches them
        perm = np.lexsort((np.arange(nsph), nz.argmax(axis=0)))
        self.ms = cp.ascontiguousarray(mc[:, cp.asarray(perm)])
        mc = None
        nzp = nz[:, perm]
        self.lo = nzp.argmax(axis=1)
        self.hi = nsph - nzp[:, ::-1].argmax(axis=1)
        # a row may start a block when every earlier row maps below every
        # later one
        pre = np.maximum.accumulate(self.hi)
        suf = np.minimum.accumulate(self.lo[::-1])[::-1]
        cuts = [0] + [r for r in range(1, ncart) if pre[r-1] <= suf[r]]
        cuts.append(ncart)
        self.blocks = []
        a0 = 0
        for i, r in enumerate(cuts[1:], 1):
            if r - a0 >= block or r == ncart:
                self.blocks.append((a0, r, int(self.lo[a0:r].min()),
                                    int(self.hi[a0:r].max())))
                a0 = r
        self.nsph, self.ncart = nsph, ncart
        j2c = int2c2e(auxmol, omega=omega)
        pj = cp.asarray(perm)
        j2c = j2c[pj[:, None], pj]
        low = cp.linalg.cholesky(j2c)
        if not bool(cp.isfinite(low).all()):
            raise RuntimeError('metric not positive definite')
        from cupyx.scipy.linalg import solve_triangular
        self.linv = cp.ascontiguousarray(solve_triangular(
            low, cp.eye(nsph), lower=True))
        self.atm = np.asarray(aux_atm_labels)[nzp.argmax(axis=0)]

    def cols(self, a0, a1):
        return int(self.lo[a0:a1].min()), int(self.hi[a0:a1].max())

    def to_sph(self, x, out):
        """``out[:nsph] = ms^T x`` along axis 0; ``out`` may be ``x``."""
        n = x.shape[0]
        assert n == self.ncart
        x2 = x.reshape(n, -1)
        o2 = out.reshape(out.shape[0], -1)
        for a0, a1, c0, c1 in self.blocks:
            # c1 <= a1 and the blocks run in order, so a block writes only
            # rows the earlier blocks have already read
            o2[c0:c1] = self.ms[a0:a1, c0:c1].T.dot(x2[a0:a1])
        return out[:self.nsph]

    def to_sph_pinv(self, x, out):
        """``out[:nsph] = mp^T x``, ``mp = C (C^T C)^-1`` shell by shell:
        the spherical ``y`` of a Cartesian ``x = C y``, exactly."""
        if not hasattr(self, 'mp'):
            self.mp = cp.zeros_like(self.ms)
            for a0, a1, c0, c1 in self.blocks:
                b = self.ms[a0:a1, c0:c1]
                self.mp[a0:a1, c0:c1] = b.dot(cp.linalg.inv(b.T.dot(b)))
        n = x.shape[0]
        x2 = x.reshape(n, -1)
        o2 = out.reshape(out.shape[0], -1)
        for a0, a1, c0, c1 in self.blocks:
            o2[c0:c1] = self.mp[a0:a1, c0:c1].T.dot(x2[a0:a1])
        return out[:self.nsph]

    def factor(self):
        """``C L^-T``, Cartesian rows in the integrals' order: with unit
        weights, what ``_factorize_j2c``'s ``V`` is to its ``w``."""
        out = cp.empty((self.ncart, self.nsph))
        lt = self.linv.T
        for a0, a1, c0, c1 in self.blocks:
            cp.dot(self.ms[a0:a1, c0:c1], lt[c0:c1], out=out[a0:a1])
        return out


def _plan_v(word_avail, naux, nao, nocc, nocc_in_batch, nao_pair, batch_size,
            nao_on_atom, natm, metric_size, batch_id):
    """Upstream's sizing of the second pass, unchanged."""
    word_avail -= 4 * naux**2  # j2c_10v
    word_avail -= 3 * batch_size * nao_pair * 2  # eval_ip1, eval_ipaux
    if batch_id == 0:
        aux_unit1 = 3*nao*nocc + 3*naux
    else:
        aux_unit1 = 3*nao*nocc_in_batch + 3*naux
    aux_batch_size = min(int(word_avail*.85),
                         word_avail-6*naux*nao_on_atom*nocc) // aux_unit1
    assert aux_batch_size > 0, 'Insufficient GPU memory'
    # FASTDFHESSIP_VCHUNK=<n> caps the chunk, for the tests: a multi-chunk
    # second pass is otherwise only reached when memory is short
    forced = int(_env('FASTDFHESSIP_VCHUNK', '0'))
    if forced > 0:
        aux_batch_size = min(aux_batch_size, forced)
    aux_batch_size = _balance_batch_size(aux_batch_size, metric_size, 16)
    word_avail -= aux_unit1 * aux_batch_size
    if batch_id == 0:
        aux_unit2 = natm*3*nocc*nocc * 2
        aux_blksize = word_avail // aux_unit2
        assert aux_blksize > 0, 'Insufficient GPU memory'
        if aux_blksize > 8:
            aux_blksize = aux_blksize & 0xfffff8
        aux_blksize = min(aux_blksize, aux_batch_size)
    else:
        aux_blksize = 0
    blksize = word_avail // ((nocc+nao)*nao * 16) * 16
    assert blksize > 0, 'Insufficient GPU memory'
    blksize = min(blksize, batch_size)
    return aux_batch_size, aux_blksize, blksize


def _ipaux_pass(eval_ipaux, num_aux_batches, blksize, nao, nao_pair,
                batch_size, nocc_in_batch, pair_addresses, dm_factor_l,
                nocc0, nocc1, j3c_oo1, auxvec_ipauxp, dm):
    """Upstream's first pass: ``(1|00)`` and, on batch 0, its trace."""
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((3, nao_pair, batch_size))
    buf1 = cp.empty((blksize * nocc_in_batch * nao))
    aux0 = 0
    for kbatch in range(num_aux_batches):
        compressed = eval_ipaux(kbatch, out=buf)
        naux_in_batch = compressed.shape[-1]
        _ipaux_batch(compressed, aux0, naux_in_batch, blksize, j3c_full, buf1,
                     nao, nocc_in_batch, pair_addresses, dm_factor_l, nocc0,
                     nocc1, j3c_oo1, auxvec_ipauxp, dm)
        aux0 += naux_in_batch


def _ipaux_batch(compressed, aux0, naux_in_batch, blksize, j3c_full, work,
                 nao, nocc_in_batch, pair_addresses, dm_factor_l, nocc0,
                 nocc1, j3c_oo1, auxvec_ipauxp, dm):
    for k0, k1 in lib.prange(0, naux_in_batch, blksize):
        dk = k1 - k0
        a0, a1 = aux0 + k0, aux0 + k1
        j3c = j3c_full[:, :, :dk]
        tmp = ndarray((nocc_in_batch, nao, dk), buffer=work)
        for i in range(3):
            fill_symmetric(compressed[i], pair_addresses, nao, k0, k1,
                           out=j3c)
            contract('pqr,pi->iqr', j3c, dm_factor_l[:, nocc0:nocc1], out=tmp)
            # d/dX = -d/dr
            contract('iqr,qj->rij', tmp, dm_factor_l, alpha=-1,
                     out=j3c_oo1[i, a0:a1])
            if auxvec_ipauxp is not None:
                contract('pqr,pq->r', j3c, dm, -1, out=auxvec_ipauxp[i, a0:a1])
    # the symmetric fill writes the whole of j3c; the (10|0) pass that may
    # share this buffer writes only the stored pairs and relies on the rest
    # being zero, which is what upstream's `j3c_full[:] = 0.` sets up
    j3c_full[:] = 0.


_MIRROR = cp.RawKernel(r'''
extern "C" __global__
void mirror_upper(double* __restrict__ c, const long long n)
{
    // c[i][j] = c[j][i] below the diagonal, from the upper triangle
    long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long i = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < i) c[i*n + j] = c[j*n + i];
}
''', 'mirror_upper')
_ONE_H = np.ones(1)


def _gram(a, alpha, out=None):
    """``alpha a a^T`` over the leading index, the rest flattened: item 9.

    Upstream writes these as ``contract('xrij,ysij->xrys', a, a)``, a general
    contraction that computes both triangles of a symmetric result.  cuBLAS
    ``dsyrk`` computes one -- half the arithmetic -- and the other is a copy.
    ``docs/README_fastdf.md`` 3.2 found ``dsyrk`` slower than the GEMM below
    about 800 rows; these are ``naux`` and ``3 naux`` rows, thousands.
    ``FASTDFHESSIP_SYRK=0`` keeps the contraction.
    """
    lead = a.shape[0] if a.ndim == 3 else a.shape[0] * a.shape[1]
    if _env('FASTDFHESSIP_SYRK', '1') == '0' or not a.flags.c_contiguous:
        a2 = a.reshape(lead, -1)
        return contract('rk,sk->rs', a2, a2, alpha, out=out)
    from cupy.cuda import cublas
    a2 = a.reshape(lead, -1)
    n, k = a2.shape
    if out is None:
        out = cp.empty((n, n))
    al = np.array([alpha], dtype=np.float64)
    ze = np.zeros(1)
    # C-ordered (n, k) is the column-major (k, n) a^T: op = T gives a a^T,
    # and the column-major lower triangle is the C-ordered upper one
    cublas.dsyrk(cp.cuda.device.get_cublas_handle(),
                 cublas.CUBLAS_FILL_MODE_LOWER, cublas.CUBLAS_OP_T,
                 n, k, al.ctypes.data, a2.data.ptr, k, ze.ctypes.data,
                 out.data.ptr, n)
    bs = 32
    _MIRROR(((n + bs - 1) // bs, (n + bs - 1) // bs), (bs, bs),
            (out, np.int64(n)))
    return out


def _h_aux_words(naux, metric_size):
    """What ``_h_aux_block``'s temporaries take from ``work``, in words."""
    return 15 * naux**2 + naux * metric_size


def _h_aux_block(auxmol, omega, _aux_sorting, _metric_v, _metric_w, dm_oo,
                 j3c_oo1, naux, metric_size, k_factor, j_factor, batch_id,
                 auxvec_ipauxp, _auxvec, natm, aux_atm_labels, work=None):
    """Upstream's code between the two passes, in upstream's order.

    With ``work`` -- the second pass's integral buffers, which the fused
    path has finished with when it gets here -- the ``naux^2``-sized
    temporaries are carved from it instead of allocated (item 11, W24):
    that is 15-16 of the 24 ``naux^2`` words item 3 had to hold back, and
    what decided that a 113-atom call could not be fused.
    """
    n = naux
    carve = work is not None and work.size >= _h_aux_words(n, metric_size)
    if carve:
        h_buf = work[:9*n*n]
        rest = work[9*n*n:]

    def take(shape, off):
        if not carve:
            return cp.empty(shape)
        size = int(np.prod(shape))
        return ndarray(shape, buffer=rest[off:off+size])

    j2c_10 = int2c2e_ip1(auxmol, sort_output=False, omega=omega)
    j2c_10 *= -1
    j2c_10, tmp = cp.empty_like(j2c_10), j2c_10
    j2c_10[:, _aux_sorting[:, None], _aux_sorting] = tmp
    j2c_10v = contract('xrs,st->xrt', j2c_10, _metric_v)
    tmp = None

    if _cuplan.planable('xuv,vij->xuij', j2c_10, dm_oo, j3c_oo1):
        j3c_oo1p = contract('xuv,vij->xuij', j2c_10, dm_oo, alpha=-1, beta=1,
                            out=j3c_oo1)
    else:
        # an output past cuTENSOR's 2**29 elements (W21), which the
        # unrestricted twin's 2 nocc reaches at 113 atoms: the same GEMM
        # through cuBLAS, one direction at a time
        _S['plan_gemm'] = _S.get('plan_gemm', 0) + 1
        d2 = dm_oo.reshape(dm_oo.shape[0], -1)
        for x in range(3):
            cp.cublas.gemm('N', 'N', j2c_10[x], d2,
                           out=j3c_oo1[x].reshape(j3c_oo1.shape[1], -1),
                           alpha=-1., beta=1.)
        j3c_oo1p = j3c_oo1
    h_aux = _gram(j3c_oo1p, -.5*k_factor,
                  out=ndarray((3*n, 3*n), buffer=h_buf) if carve else None
                  ).reshape(3, n, 3, n)
    if j_factor != 0 and batch_id == 0:
        contract('xuv,v->xu', j2c_10, _auxvec, -1, 1, out=auxvec_ipauxp)
        contract('xr,ys->xrys', auxvec_ipauxp, auxvec_ipauxp, j_factor, 1,
                 h_aux)
    vw = take((n, metric_size), n*n)
    cp.divide(_metric_v, _metric_w, out=vw)
    j2c_inv = cp.dot(vw, _metric_v.T, out=take((n, n), 0))
    h_aux *= j2c_inv[None, :, None, :]
    j2c_inv = vw = None

    dm_aux = _gram(dm_oo, -.5*k_factor, out=take((n, n), 0))
    if j_factor != 0 and batch_id == 0:
        contract('r,s->rs', _auxvec, _auxvec, j_factor, 1, dm_aux)
    ejk_aux = _int2c2e_ip2_per_atom(
        auxmol, dm_aux[_aux_sorting[:, None], _aux_sorting], omega)
    ejk = -ejk_aux

    j2c_10v_w = take((n, metric_size), n*n)
    j2c_ip2 = take((n, n), n*n + n*metric_size)
    for x in range(3):
        cp.divide(j2c_10v[x], _metric_w, out=j2c_10v_w)
        for y in range(3):
            j2c_ip2 = contract('rs,ts->rt', j2c_10v_w, j2c_10v[y], out=j2c_ip2)
            j2c_ip2 *= dm_aux
            h_aux[x, :, y] += j2c_ip2
    dm_aux = j2c_ip2 = j2c_10v_w = None

    dm_aux1 = contract('xrij,sij->xrs', j3c_oo1p, dm_oo, -.5*k_factor,
                       out=take((3, n, n), 0))
    if j_factor != 0 and batch_id == 0:
        contract('xr,t->xrt', auxvec_ipauxp, _auxvec, j_factor, 1, out=dm_aux1)
    vw = take((n, metric_size), 3*n*n)
    cp.divide(_metric_v, _metric_w, out=vw)
    tmp = contract('xrt,st->xrs', j2c_10v, vw,
                   out=take((3, n, n), 3*n*n + n*metric_size))
    h_aux = contract('xrt,ytr->xryt', tmp, dm_aux1, alpha=-2, beta=1,
                     out=h_aux)
    dm_aux1 = tmp = vw = None

    h_aux = transpose_sum(h_aux.reshape(3*n, 3*n), inplace=True)
    h_aux = h_aux.reshape(3, n, 3, n).transpose(1, 3, 0, 2)
    ejk_aux = _aggregate_to_atoms(h_aux, natm, aux_atm_labels, axis=(0, 1))
    ejk += ejk_aux * .5
    return j3c_oo1p, (ejk, j2c_10v, j2c_10)


def _j2c_10c(sa, j2c_10):
    """``j2c_10 C``: the metric derivative with its second index spherical.

    Only item 8 needs it; the eigen form drops ``j2c_10`` here, as upstream
    does.
    """
    if sa is None:
        return None
    out = cp.empty((3, sa.ncart, sa.nsph))
    for x in range(3):
        cp.dot(j2c_10[x], sa.ms, out=out[x])
    return out


def _chol_d4_d5(sa, ksph, work, natm, nocc, atm_ind, aoslices, dm_factor_l,
                dm_oo, j3c_oo1p, j2c_10c, k_factor, j_factor, ejk, h_ao_aux,
                auxvec, auxvec_ipauxp, st):
    """Item 8: D4 and D5 with the metric as its Cholesky factor.

    Upstream carries ``(10|0)`` into the metric's eigenbasis
    (``j3c_100 = V^T K``, D3), back out of it for the cross terms
    (``(V/w) j3c_100``, the first half of D5) and against ``dm_oo`` there
    (the second half): three ``2 3 naux^2 nao nocc`` contractions, each with
    one Cartesian and one spherical auxiliary index.  With ``J = L L^T`` on
    the spherical index and ``K`` the spherical ``(10|0) C``, which D3 has
    left in ``ksph`` instead:

    * every sum ``sum_r' X_r' Y_r' / w_r'`` D4 forms is the same sum over
      ``Y = L^{-1} K`` -- a rotation of the whitened eigenbasis ``js`` --
      so D4 runs unchanged on it;
    * ``(V/w) j3c_100 = C J^{-1} K = C Z`` with ``Z = L^{-T} Y``, and ``C``
      is block diagonal by shell, so the first half of D5 contracts ``Z``
      with ``C^T`` applied to its partner, per auxiliary shell, rather than
      forming ``C Z``;
    * the second half is ``sum_t j2c_10C[y,s,t] sum_{u in A,j} P[s,u,j]
      Z[x,t,u,j]``, one contraction as before.

    The two triangular multiplies against ``L^{-1}`` are one
    ``naux_sph^2``-sized contraction between them, so the three become two,
    and both are spherical in the auxiliary index that was Cartesian.
    Returns the spherical-indexed cross terms, summed over auxiliary atoms.
    """
    nsph = sa.nsph
    nao = dm_factor_l.shape[0]
    solve = _env('FASTDFHESSIP_TRI', 'mult') == 'solve'
    if solve:
        from cupyx.scipy.linalg import solve_triangular
        low = cp.linalg.inv(sa.linv)
    for x in range(3):
        if solve:
            ksph[x] = solve_triangular(low, ksph[x].reshape(nsph, -1),
                                       lower=True).reshape(ksph[x].shape)
        else:
            trmm_left_lower(sa.linv, ksph[x].reshape(nsph, -1))
    st('D8 L^-1 K')
    _oo_gram(ksph, 0, nsph, work, natm, nocc, atm_ind, dm_factor_l,
             cp.ones(nsph), None, k_factor, j_factor, ejk, None, None,
             auxvec, None, None, h_j=False)
    st('D4 oo_atm')
    for x in range(3):
        if solve:
            ksph[x] = solve_triangular(low, ksph[x].reshape(nsph, -1),
                                       lower=True, trans='T'
                                       ).reshape(ksph[x].shape)
        else:
            trmm_left_lower(sa.linv, ksph[x].reshape(nsph, -1), trans=True)
    z = ksph
    st('D8 L^-T Y')
    if j_factor != 0:
        # upstream's (10|0)(0|1)(0|00) term of the Gram block, which needs
        # J^{-1} K and not L^{-1} K: sum_r' a[r']/w V[s,r'] = (C J^-1 a)[s]
        d = contract('xtui,ui->xtu', z, dm_factor_l)
        at = contract('au,xtu->axt', atm_ind, d, alpha=2.)
        tmp, _ = _allocate((3, sa.ncart, nsph), work)
        # upstream's `contract(..., beta=-1)`: ipaux' V - j2c_10v auxvec
        cp.multiply(j2c_10c, -auxvec[None, :, None], out=tmp)
        tmp += auxvec_ipauxp[:, :, None] * sa.ms
        contract('pxt,yst->psxy', at, tmp, j_factor, 1, h_ao_aux)
        d = at = tmp = None
    # C^T on the partner of the first half, in place
    for x in range(3):
        sa.to_sph(j3c_oo1p[x], j3c_oo1p[x])
    t1s = j3c_oo1p[:, :nsph]
    # and on the output index of the second half: its s is summed over an
    # auxiliary atom, and C does not mix atoms, so sum_{s in B} of a product
    # linear in P_cart = C P_sph is sum_{s' in B} of C^T on the other factor
    t2sph = _env('FASTDFHESSIP_T2SPH', '1') != '0'
    if t2sph:
        for y in range(3):
            sa.to_sph(j2c_10c[y], j2c_10c[y])
        j10s = j2c_10c[:, :nsph]
    h_sph = cp.zeros((natm, nsph, 3, 3))
    for i, (p0, p1) in enumerate(zip(aoslices[:-1], aoslices[1:])):
        t1, work1 = _allocate((3, nsph, p1-p0, nocc), work)
        contract('ysji,ui->ysuj', t1s, dm_factor_l[p0:p1], out=t1)
        contract('xsuj,ysuj->sxy', z[:, :, p0:p1], t1, -k_factor, 1,
                 h_sph[i])
        pa, work1 = _allocate((sa.ncart, p1-p0, nocc), work)
        contract('sji,ui->suj', dm_oo, dm_factor_l[p0:p1], out=pa)
        if t2sph:
            ps = sa.to_sph_pinv(pa, pa)
            m, work1 = _allocate((3, nsph, nsph), work1)
            contract('suj,xtuj->xst', ps, z[:, :, p0:p1], out=m)
            contract('xst,yst->sxy', m, j10s, k_factor, 1, h_sph[i])
        else:
            m, work1 = _allocate((3, sa.ncart, nsph), work1)
            contract('suj,xtuj->xst', pa, z[:, :, p0:p1], out=m)
            contract('xst,yst->sxy', m, j2c_10c, k_factor, 1, h_ao_aux[i])
    st('D5 per-atom')
    return _aggregate_to_atoms(h_sph, natm, sa.atm, axis=1)


def _oo_gram_rq(j3c_100, v0, dv, work, nocc, _dm_factor_l, _metric_w):
    """W17's ``R`` and ``Q``: ``R`` as one contraction, ``Q`` from ``M`` and
    its transpose as two GEMMs.  ``FASTDFHESSIP_D4=gemm``."""
    nao = _dm_factor_l.shape[0]
    per_r = 3*nao*nocc + 6*nao*nao
    # the chunk is bounded by the buffer upstream sized for X.  W17 also
    # bounded it at 1.2e8 // (3 nao^2) because R's contraction was refused
    # once at 42 atoms with the whole auxiliary set in one chunk; W21 found
    # that cuTENSOR's refusal is a property of the *output* (past 2**29
    # elements), R's output is (3, 3, nao, nao) whatever the chunk, and the
    # whole-set chunk at 42 atoms plans.  So the bound is now a query
    # (`electrolites/_cuplan.py`); `FASTDFHESSIP_RCHUNK=<elements>` puts
    # W17's back, and `_accumulate_r` still halves along r if a plan is
    # refused anyway
    kb = max(1, min(dv, work.size // per_r))
    if _RCHUNK:
        kb = max(1, min(kb, int(_RCHUNK // (3*nao*nao))))
    else:
        kb = _cuplan.largest(kb, lambda m: _cuplan.planable(
            'xrui,yrvi->xyuv', (3, m, nao, nocc), (3, m, nao, nocc),
            (3, 3, nao, nao)))
    R = cp.zeros((3, 3, nao, nao))
    Q = cp.zeros((3, 3, nao, nao))
    for k0, k1 in lib.prange(0, dv, kb):
        dk = k1 - k0
        js, work1 = _allocate((3, dk, nao, nocc), work)
        m, work1 = _allocate((3, dk, nao, nao), work1)
        n, work1 = _allocate((3, dk, nao, nao), work1)
        sw = 1. / cp.sqrt(_metric_w[v0+k0:v0+k1])
        cp.multiply(j3c_100[:, k0:k1], sw[:, None, None], out=js)
        _accumulate_r('xrui,yrvi->xyuv', js, js, R)
        # M[x,r,u,v] = sum_i js[x,r,u,i] C[v,i] is one GEMM on the flattened
        # (x,r,u) row index, so cuBLAS rather than cuTENSOR; its transpose
        # N[x,r,u,v] = M[x,r,v,u] = sum_i C[u,i] js[x,r,v,i] is a strided
        # batch of the same GEMM, so that the r-sum of M_x o M_y^T below
        # reads both operands in the same layout
        cp.dot(js.reshape(-1, nocc), _dm_factor_l.T,
               out=m.reshape(-1, nao))
        cp.matmul(_dm_factor_l, js.reshape(3*dk, nao, nocc).transpose(0, 2, 1),
                  out=n.reshape(3*dk, nao, nao))
        _hadamard_r(m, n, Q)
    return R, Q


def _oo_gram_rq_syrk(j3c_100, v0, dv, work, nocc, _dm_factor_l, _metric_w):
    """Item 10: ``R`` as a ``dsyrk`` and ``Q`` from one GEMM.

    ``R[(x,u),(y,v)] = sum_{r,i} js[x,r,u,i] js[y,r,v,i]`` is ``A A^T`` for
    ``A[(x,u),(r,i)]``, symmetric, and ``3 nao`` rows is thousands -- so
    the chunk is copied into that layout once and half of the contraction
    is not computed.  In the same layout ``M'[x,u,r,v] = sum_i A C[v,i]`` is
    one GEMM, and ``Q = sum_r M'[x,u,r,v] M'[y,v,r,u]`` reads its transpose
    through shared memory (``_HADAMARD_T``) instead of forming it with a
    second GEMM.
    """
    from cupy.cuda import cublas
    nao = _dm_factor_l.shape[0]
    per_r = 3*nao*nocc + 3*nao*nao
    kb = max(1, min(dv, work.size // per_r))
    n3 = 3 * nao
    R2 = cp.zeros((n3, n3))
    Q = cp.zeros((3, 3, nao, nao))
    one = np.ones(1)
    handle = cp.cuda.device.get_cublas_handle()
    for k0, k1 in lib.prange(0, dv, kb):
        dk = k1 - k0
        jt, work1 = _allocate((3, nao, dk, nocc), work)
        m, work1 = _allocate((3, nao, dk, nao), work1)
        sw = 1. / cp.sqrt(_metric_w[v0+k0:v0+k1])
        cp.multiply(j3c_100[:, k0:k1].transpose(0, 2, 1, 3),
                    sw[None, None, :, None], out=jt)
        kk = dk * nocc
        cublas.dsyrk(handle, cublas.CUBLAS_FILL_MODE_LOWER,
                     cublas.CUBLAS_OP_T, n3, kk, one.ctypes.data,
                     jt.data.ptr, kk, one.ctypes.data, R2.data.ptr, n3)
        cp.dot(jt.reshape(-1, nocc), _dm_factor_l.T, out=m.reshape(-1, nao))
        _hadamard_t(m, Q)
    jt = m = None
    bs = 32
    _MIRROR(((n3 + bs - 1) // bs, (n3 + bs - 1) // bs), (bs, bs),
            (R2, np.int64(n3)))
    R = cp.ascontiguousarray(R2.reshape(3, nao, 3, nao).transpose(0, 2, 1, 3))
    return R, Q


_HADAMARD_T = cp.RawKernel(r'''
#define T 16
extern "C" __global__
void hadamard_t(const double* __restrict__ m, double* __restrict__ q,
                const int nao, const int nr)
{
    // q[x][y][u][v] += sum_r m[x][u][r][v] * m[y][v][r][u]
    __shared__ double sa[3][T][T+1];
    __shared__ double sb[3][T][T+1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int u0 = blockIdx.y * T, v0 = blockIdx.x * T;
    const long long sx = (long long)nao * nr * nao;
    const long long su = (long long)nr * nao;
    double acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    const int ua = u0 + ty, va = v0 + tx;   // sa: m[x][ua][r][va]
    const int vb = v0 + ty, ub = u0 + tx;   // sb: m[y][vb][r][ub]
    for (int r = 0; r < nr; r++) {
        for (int x = 0; x < 3; x++) {
            sa[x][ty][tx] = (ua < nao && va < nao) ?
                m[x*sx + ua*su + (long long)r*nao + va] : 0.;
            sb[x][ty][tx] = (vb < nao && ub < nao) ?
                m[x*sx + vb*su + (long long)r*nao + ub] : 0.;
        }
        __syncthreads();
        // this thread's (u, v) = (u0+ty, v0+tx); sb holds [v-v0][u-u0]
        for (int x = 0; x < 3; x++) {
            const double a = sa[x][ty][tx];
            for (int y = 0; y < 3; y++) acc[x*3+y] += a * sb[y][tx][ty];
        }
        __syncthreads();
    }
    const int u = u0 + ty, v = v0 + tx;
    if (u < nao && v < nao) {
        const long long np = (long long)nao * nao;
        for (int k = 0; k < 9; k++) q[k*np + (long long)u*nao + v] += acc[k];
    }
}
''', 'hadamard_t')


def _hadamard_t(m, q):
    """``q[x,y,u,v] += sum_r m[x,u,r,v] m[y,v,r,u]``."""
    nao = m.shape[1]
    nr = m.shape[2]
    t = 16
    g = (nao + t - 1) // t
    _HADAMARD_T((g, g), (t, t), (m, q, np.int32(nao), np.int32(nr)))


def _oo_gram(j3c_100, v0, dv, work, natm, nocc, atm_ind, _dm_factor_l,
             _metric_w, _metric_v, k_factor, j_factor, ejk, h_ao_aux,
             j2c_10v, _auxvec, auxvec_ipauxp, naux, h_j=True):
    """Item 2 as a Gram matrix: the (10|0)(0|0)(0|01) block without ``X``.

    Upstream builds ``X[A,x,r,i,j] = sum_{u in A} J[x,r,u,i] C[u,j]`` for
    every atom, symmetrises it, and contracts
    ``sum_{r,i,j} S[A,x,r,i,j] S[B,y,r,i,j] / w_r``: ``3 natm nocc^2``
    doubles per auxiliary function (138 MB at 113 atoms) and
    ``9 natm^2 naux nocc^2`` flops.  With ``S = X + X^T`` the sum is
    ``2 (T1 + T2)``, and both halves fold the atom sums out of the
    contraction:

        T1[A,B] = sum_{u in A, v in B} P[u,v] R[u,v],
                  R[u,v] = sum_{r,i} J[r,u,i] J[r,v,i] / w_r,   P = C C^T
        T2[A,B] = sum_{u in A, v in B} Q[u,v],
                  Q[u,v] = sum_r M[r,u,v] M[r,v,u] / w_r,    M_r = J_r C^T

    which is ``9 nao^2 naux nocc`` flops for ``R`` and ``3 nao^2 naux nocc``
    for ``M``, against upstream's ``9 natm^2 naux nocc^2`` -- fewer by
    ``natm^2 nocc / (4/3 nao^2)``, a ratio that grows linearly with the
    system -- and no array whose size has ``natm`` in it.  The ``1/w_r``
    is split as ``1/sqrt(w_r)`` on each factor.
    """
    nao = _dm_factor_l.shape[0]
    if _env('FASTDFHESSIP_D4', 'syrk') == 'syrk':
        R, Q = _oo_gram_rq_syrk(j3c_100, v0, dv, work, nocc, _dm_factor_l,
                                _metric_w)
    else:
        R, Q = _oo_gram_rq(j3c_100, v0, dv, work, nocc, _dm_factor_l,
                           _metric_w)
    if j_factor != 0:
        kb = max(1, min(dv, int(2e8) // (3*(naux or nao))))
        for k0, k1 in lib.prange(0, dv, kb):
            dk = k1 - k0
            # the trace of S: 2 sum_{u in A, i} J[x,r,u,i] C[u,i]
            d = contract('xrui,ui->xru', j3c_100[:, k0:k1], _dm_factor_l)
            auxvec_100_atm = contract('au,xru->axr', atm_ind, d, alpha=2.)
            w_part = _metric_w[v0+k0:v0+k1]
            auxvec_100_atm_w = auxvec_100_atm / w_part
            contract('pxr,qyr->pqxy', auxvec_100_atm_w, auxvec_100_atm,
                     j_factor, beta=1, out=ejk)
            if not h_j:
                continue
            tmp = cp.empty((3, naux, dk))
            cp.multiply(j2c_10v[:, :, v0+k0:v0+k1], _auxvec[:, None], out=tmp)
            contract('ys,sr->ysr', auxvec_ipauxp, _metric_v[:, v0+k0:v0+k1],
                     beta=-1, out=tmp)
            contract('pxr,ysr->psxy', auxvec_100_atm_w, tmp, j_factor, 1,
                     h_ao_aux)
            d = tmp = auxvec_100_atm = auxvec_100_atm_w = None
    P = _dm_factor_l.dot(_dm_factor_l.T)
    R *= P
    R += Q
    Q = None
    t = contract('au,xyuv->axyv', atm_ind, R)
    # -.5 k from the Coulomb operator, 2 from S = X + X^T
    contract('axyv,bv->abxy', t, atm_ind, -.5*k_factor*2, beta=1, out=ejk)


_HADAMARD_R = cp.RawKernel(r'''
extern "C" __global__
void hadamard_r(const double* __restrict__ m, const double* __restrict__ n,
                double* __restrict__ q, const int nr, const long long np)
{
    // q[x][y][p] += sum_r m[x][r][p] * n[y][r][p]
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= np) return;
    double acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    const long long sx = (long long)nr * np;
    for (int r = 0; r < nr; r++) {
        const long long o = (long long)r * np + p;
        double m0 = m[o], m1 = m[sx + o], m2 = m[2*sx + o];
        double n0 = n[o], n1 = n[sx + o], n2 = n[2*sx + o];
        acc[0] += m0*n0; acc[1] += m0*n1; acc[2] += m0*n2;
        acc[3] += m1*n0; acc[4] += m1*n1; acc[5] += m1*n2;
        acc[6] += m2*n0; acc[7] += m2*n1; acc[8] += m2*n2;
    }
    for (int k = 0; k < 9; k++) q[k*np + p] += acc[k];
}
''', 'hadamard_r')


def _hadamard_r(m, n, q):
    """``q[x,y,u,v] += sum_r m[x,r,u,v] n[y,r,u,v]``, one pass over both."""
    nr = m.shape[1]
    npq = m.shape[2] * m.shape[3]
    threads = 256
    _HADAMARD_R(((npq + threads - 1) // threads,), (threads,),
                (m, n, q, np.int32(nr), np.int64(npq)))


def _accumulate_r(pattern, a, b, out):
    """``out += contract(pattern, a, b)``, split along axis 1 if refused.

    Axis 1 is the auxiliary index ``r``, which both patterns sum over, so a
    split is exact.  A refused plan raises before any operand is read
    (`docs/README_fastdfhess.md` 4), so nothing has been accumulated when
    the halves are tried.
    """
    from cupy_backends.cuda.libs.cutensor import CuTensorError
    try:
        contract(pattern, a, b, beta=1, out=out)
    except CuTensorError:
        n = a.shape[1]
        if n < 2:
            raise
        _S['plan_splits'] = _S.get('plan_splits', 0) + 1
        h = n // 2
        _accumulate_r(pattern, a[:, :h], b[:, :h], out)
        _accumulate_r(pattern, a[:, h:], b[:, h:], out)


def _oo_atm_block(j3c_100, k0, k1, v0, work, natm, nocc, aoslices,
                  _dm_factor_l, _metric_w, _metric_v, k_factor, j_factor,
                  ejk, h_ao_aux, j2c_10v, _auxvec, auxvec_ipauxp, naux,
                  oosum):
    """(10|0)(0|0)(0|01) and its three partners, for one auxiliary block."""
    j3c_oo_atm, work1 = _allocate((natm, 3, k1-k0, nocc, nocc), work)
    if oosum == 'transpose':
        for i, (p0, p1) in enumerate(zip(aoslices[:-1], aoslices[1:])):
            contract('xrui,uj->xrij', j3c_100[:, k0:k1, p0:p1],
                     _dm_factor_l[p0:p1], out=j3c_oo_atm[i])
        transpose_sum(j3c_oo_atm.reshape(-1, nocc, nocc), inplace=True)
    else:
        # item 2: X + X^T as two contractions into the same buffer, the
        # second writing its output indices the other way round
        for i, (p0, p1) in enumerate(zip(aoslices[:-1], aoslices[1:])):
            contract('xrui,uj->xrij', j3c_100[:, k0:k1, p0:p1],
                     _dm_factor_l[p0:p1], out=j3c_oo_atm[i])
            contract('xruj,ui->xrij', j3c_100[:, k0:k1, p0:p1],
                     _dm_factor_l[p0:p1], beta=1, out=j3c_oo_atm[i])
    w_part = _metric_w[v0+k0:v0+k1]
    j3c_oo_atm_w = ndarray(j3c_oo_atm.shape, buffer=work1)
    cp.divide(j3c_oo_atm, w_part[:, None, None], out=j3c_oo_atm_w)
    contract('pxrij,qyrij->pqxy', j3c_oo_atm_w, j3c_oo_atm, -.5*k_factor,
             beta=1, out=ejk)
    if j_factor != 0:
        auxvec_100_atm = cp.einsum('pxrii->pxr', j3c_oo_atm)
        auxvec_100_atm_w = auxvec_100_atm / w_part
        contract('pxr,qyr->pqxy', auxvec_100_atm_w, auxvec_100_atm,
                 j_factor, beta=1, out=ejk)
        tmp = ndarray((3, naux, k1-k0), buffer=work)
        cp.multiply(j2c_10v[:, :, v0+k0:v0+k1], _auxvec[:, None], out=tmp)
        contract('ys,sr->ysr', auxvec_ipauxp, _metric_v[:, v0+k0:v0+k1],
                 beta=-1, out=tmp)
        contract('pxr,ysr->psxy', auxvec_100_atm_w, tmp, j_factor, 1,
                 h_ao_aux)



def _ip2_tiles(int3c2e_opt, omega, spmult, gout, ao_pair_loc):
    """Item 15 (W29): ``ejk_int3c2e_ip2`` launched as its own tile
    (``electrolites/_ip2launch.py``); ``None`` is item 1's launch.
    ``FASTDFHESSIP_IP2_TILE=0``."""
    if _env('FASTDFHESSIP_IP2_TILE', '1') == '0':
        return None
    from ._ip2launch import Tiles
    t = Tiles(int3c2e_opt, omega, spmult, gout)
    t.set_ao_pair_loc(ao_pair_loc)
    _S['ip2_tiles'] = _S.get('ip2_tiles', 0) + 1
    return t

def _ejk_ip2(int3c2e_opt, dm_factor_l, dm_oo_full, auxvec, j_factor,
             k_factor, omega, lr_factor, sr_factor, pair_addresses, nao_pair,
             batch_size, blksize, spmult, st):
    """(20|0)(0|0)(0|00) + (10|1)(0|0)(0|00): upstream's kernel loop."""
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    natm = mol.natm
    nao, nocc = dm_factor_l.shape
    aux_loc = auxmol.ao_loc
    if j_factor != 0:
        dm = dm_factor_l.dot(dm_factor_l.T)

    gout = int(_env('FASTDFHESSIP_IP2_GOUT', '0')) or None
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme_ip2(omega, gout)
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    lmax = mol.uniq_l_ctr[:, 0].max()
    laux = auxmol.uniq_l_ctr[:, 0].max()
    shm_size_max = shm_size[:laux+1, :lmax+1, :lmax+1].max()

    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0] * spmult)
    ao_pair_loc = get_ao_pair_loc(mol.uniq_l_ctr[:, 0],
                                  int3c2e_opt.bas_ij_cache)

    l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, auxmol.uniq_l_ctr, batch_size)
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu + mol.nbas, dtype=np.int32)
    aux_offsets = aux_loc[ksh_offsets_cpu]
    num_aux_batches = len(aux_offsets) - 1

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern_ip2 = libvhf_rys.ejk_int3c2e_ip2
    ejk = cp.zeros((natm, natm, 3, 3))
    buf = cp.empty((nao_pair*batch_size))
    buf2 = cp.empty((blksize, nao, nao))
    buf1 = cp.empty((blksize, nao, nocc))
    tiles = _ip2_tiles(int3c2e_opt, omega, spmult, gout, ao_pair_loc)
    if tiles is not None:
        bufp = cp.empty((nao_pair*batch_size))
    per_class = _env('FASTDFHESSIP_TIME', '0') == '2'
    if per_class:
        # `FASTDFHESSIP_TIME=2`: the ip2 kernel timed per auxiliary batch,
        # which is one angular momentum and contraction pattern (W24)
        lk = np.repeat(uniq_l_ctr_aux[:, 0], np.diff(l_ctr_aux_offsets))
        ev = []
    for kbatch in range(num_aux_batches):
        naux_in_batch = aux_offsets[kbatch+1] - aux_offsets[kbatch]
        aux0 = aux1 = aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        compressed = ndarray((nao_pair, naux_in_batch), buffer=buf)
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            dm_tensor = ndarray((nao, nao, dk), buffer=buf2)
            tmp = ndarray((nocc, nao, dk), buffer=buf1)
            beta = 0
            if j_factor != 0:
                cp.multiply(dm[:, :, None], auxvec[aux0:aux1], out=dm_tensor)
                beta = j_factor
            dm_oo = cp.asarray(dm_oo_full[aux0:aux1])
            contract('rji,qj->iqr', dm_oo, dm_factor_l, out=tmp)
            contract('iqr,pi->pqr', tmp, dm_factor_l, -.5*k_factor, beta,
                     out=dm_tensor)
            cp.take(dm_tensor.reshape(-1, dk), pair_addresses, axis=0,
                    out=compressed[:, k0:k1])
        st('C pseudo-dm')
        if per_class:
            e0 = cp.cuda.Event()
            e0.record()
        if tiles is not None:
            # item 15 (W29): the same kernel launched as its own tile
            ks0, ks1 = ksh_offsets_cpu[kbatch], ksh_offsets_cpu[kbatch+1]
            perm = tiles.permutation(ks0, ks1)
            if perm is not None:
                compressed = cp.take(compressed, perm, axis=1, out=ndarray(
                    (nao_pair, naux_in_batch), buffer=bufp))
            tiles.launch(ejk, compressed, None, int3c2e_envs, omega,
                         lr_factor, sr_factor, ks0, ks1, aux_ao_offset,
                         naux_in_batch)
        else:
            err = kern_ip2(
                ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
                lib.c_null_ptr(),
                ctypes.byref(int3c2e_envs), ctypes.c_double(omega),
                ctypes.c_double(lr_factor), ctypes.c_double(sr_factor),
                ctypes.c_int(shm_size_max),
                ctypes.c_int(len(shl_pair_offsets) - 1),
                ctypes.c_int(1),
                ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(ksh_offsets_gpu[kbatch:].data.ptr,
                            ctypes.c_void_p),
                ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(aux_ao_offset),
                ctypes.c_int(naux_in_batch))
            if err != 0:
                raise RuntimeError('ejk_int3c2e_ip2 failed')
        if per_class:
            e1 = cp.cuda.Event()
            e1.record()
            ev.append((kbatch, e0, e1))
        st('C ip2 kernel')
    if per_class:
        cp.cuda.Stream.null.synchronize()
        by_l = {}
        for kbatch, e0, e1 in ev:
            ks0, ks1 = ksh_offsets_cpu[kbatch], ksh_offsets_cpu[kbatch+1]
            l = int(lk[ks0])
            t = cp.cuda.get_elapsed_time(e0, e1)
            n = int(aux_offsets[kbatch+1] - aux_offsets[kbatch])
            a = by_l.setdefault(l, [0., 0, 0])
            a[0] += t
            a[1] += ks1 - ks0
            a[2] += n
        tot = sum(v[0] for v in by_l.values())
        print('[fastdfhessip ip2 kernel] by auxiliary l: ' + '  '.join(
            f'l={l}: {v[0]:.1f} ms ({100 * v[0] / tot:.0f} %, {v[1]} shells,'
            f' {v[2]} functions)' for l, v in sorted(by_l.items())),
            flush=True)
    return ejk


# --------------------------------------------------------------------------
# W18: the h1 term of make_h1
# --------------------------------------------------------------------------
def _get_veff(int3c2e_opt, mo_coeff, mo_occ, j_factor=1, k_factor=1,
              omega=None, verbose=None):
    if not _usable():
        _S['declined'] += 1
        return DFHESS._get_veff_orig(int3c2e_opt, mo_coeff, mo_occ, j_factor,
                                     k_factor, omega, verbose)
    _S['h1'] += 1
    spmult = int(_env('FASTDFHESSIP_SPMULT', '1'))
    jbatch = _env('FASTDFHESSIP_H1J', 'batched') != 'upstream'
    metric_inv = _env('FASTDFHESSIP_H1METRIC', 'upstream') == 'inverse'
    # item 14: a pure functional's h1 call (and the Coulomb half of any
    # other) has k_factor = 0, and upstream forms every exchange
    # contraction anyway and scales it by zero
    do_k = k_factor != 0 or _env('FASTDFHESSIP_K0', 'skip') != 'skip'
    st = _Stages('h1')

    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    if omega is None:
        omega = mol.omega

    ao_idx = mol.get_ao_idx()
    mo_coeff = mol.apply_C_dot(mo_coeff, axis=0)
    mo_coeff = mo_coeff[ao_idx]
    orbo = mo_coeff[:, mo_occ > 0]
    nao, nocc = orbo.shape

    natm = mol.natm
    aux_loc = auxmol.ao_loc
    naux = int(aux_loc[-1])

    atm_ao_counts = np.bincount(_bas_atom_labels(mol))
    aoslices = np.append(0, atm_ao_counts.cumsum())
    nao_on_atom = atm_ao_counts.max()

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=True)[0]
    nao_pair = len(pair_addresses)

    cp.get_default_memory_pool().free_all_blocks()
    mem_avail = _avail()
    word_avail = mem_avail // 8
    word_avail -= natm*3*nocc*nao  # vhf_atm
    word_avail -= naux**2 * 3  # metric_v

    num_occ_batches = int(naux*nocc*(nao+nocc) / (word_avail*.7) + 1)
    nocc_per_batch = (nocc + num_occ_batches - 1) // num_occ_batches
    nocc_per_batch = min(nocc, (nocc_per_batch + 7) & 0xffffff8)
    nocc_per_batch = _forced_occ_batch(nocc_per_batch)
    num_occ_batches = (nocc + nocc_per_batch - 1) // nocc_per_batch
    _S['occ_batches'] = num_occ_batches

    word_avail -= naux*nocc_per_batch*nao  # dm_3c ~ (aux,i,a)
    word_avail -= naux*nocc*nocc_per_batch  # dm_oo ~ (aux,i,j)
    assert word_avail > 0, 'Insufficient GPU memory'

    mem_sufficient = natm*3*nao*nao < word_avail * .6
    # the per-atom AO-pair matrices hold exchange terms only
    k_ao = mem_sufficient and do_k
    if k_ao:
        word_avail -= natm * 3 * nao * nao

    pair_size_max = max(nao_pair, nocc_per_batch*nao)
    _unit = 6*pair_size_max
    largest_shell_nao = int((aux_loc[1:] - aux_loc[:-1]).max())
    batch_size = int(word_avail*.8 / _unit)
    if batch_size < largest_shell_nao:
        mem_req = (largest_shell_nao-batch_size) * _unit * 1.5 * 1e-6
        raise MemoryError(f'Insufficient GPU memory. Need {mem_req} MB more.')
    batch_size = min(batch_size, max(largest_shell_nao, naux))

    with _spmult(spmult, short_range=omega < 0):
        eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
            aux_batch_size=batch_size, reorder_aux=True, cart=True,
            omega=omega)
    batch_size = min(batch_size, int((aux_offsets[1:]-aux_offsets[:-1]).max()))
    num_aux_batches = len(aux_offsets) - 1

    # item 4 holds one more (nao, blksize, nao) array per block -- the
    # occupied-contracted density, hoisted out of the loop over x -- so the
    # block is sized with it in the denominator
    h1atom = _env('FASTDFHESSIP_H1ATOM', 'auto')
    hoist = k_ao and h1atom != 'upstream'
    # item 12 (W30): the per-atom exchange term with the AO pair projected on
    # the occupied orbitals before the atom's contraction instead of after
    # it, as `_dfhessip_u` does for the unrestricted twin: 8 nao^2 nocc
    # against 4 nao^3 flops per auxiliary function and direction, for one
    # more launch per atom.  Taken where the saving per atom is large (the
    # score and its threshold are the twin's; `docs/README_fastdfh1occ.md`
    # measures where the restricted crossover sits).  `=occ` and `=batched`
    # force either form
    score = nao / natm * nao * max(nao - 2*nocc, 0)
    occ_k = hoist and (h1atom == 'occ' or
                       (h1atom == 'auto' and score > float(
                           _env('FASTDFHESSIP_H1OCC_SCORE', '4e6'))))
    _S['h1_occ'] = bool(occ_k)
    _S['h1_score'] = float(score)
    # with the occupied index batched, D1's j3c . C_o covers the batch only,
    # and the occupied form needs it over every occupied orbital
    pri_full = occ_k and num_occ_batches > 1
    blk_unit = (nao**2 * (2 if hoist else 1) +
                (nao+nocc)*max(nao_on_atom, nocc) +
                (nao*nocc * (2 if pri_full else 1) if occ_k else 0))
    blksize = int((word_avail*.95 - batch_size*_unit) / blk_unit) // 8 * 8
    assert blksize > 0, 'Insufficient GPU memory'
    blksize = min(blksize, batch_size)

    aux_idx, aux_slices = _argsort_aux_by_atom(auxmol, aux_sorting)
    # the atom-of-AO indicator: sum over the AOs of each atom as one GEMM
    atm_ind = _atom_indicator(aoslices)
    small_atoms = [(i, p0, p1) for i, (p0, p1) in
                   enumerate(zip(aoslices[:-1], aoslices[1:]))]
    counts = aux_slices[:, 1] - aux_slices[:, 0]
    naux_in_atm = int(counts.max())
    atm_id_for_aux = np.empty(naux, dtype=int)
    atm_id_for_aux[aux_idx] = np.repeat(np.arange(auxmol.natm), counts)
    aux_filling_order = np.empty(naux, dtype=int)
    aux_filling_order[aux_idx] = np.arange(naux)

    _pair_addresses = cp.asarray(pair_addresses, dtype=np.int32)
    i_addr, j_addr = divmod(_pair_addresses, nao)
    _orbo = cp.asarray(orbo)
    _aux_sorting = aux_sorting

    metric_w, metric_v = _factorize_j2c(auxmol, _aux_sorting, omega)

    eval_ip1 = _ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ip1(omega, 27), batch_size,
        'fill_int3c2e_ip1', omega, spmult)[0]
    eval_ipaux = _ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ipaux(omega, 27), batch_size,
        'fill_int3c2e_ipaux', omega, spmult)[0]

    vhf_atm = cp.zeros((natm, 3, nocc, nao))
    vhf1 = cp.zeros((3, nao, nao))
    if k_ao:
        vhf_atm_ao = cp.zeros((natm, 3, nao, nao))

    if j_factor != 0:
        auxvec = cp.empty(naux)
        dm = _orbo.dot(_orbo.T)
    st('setup')

    for it, batch_id in enumerate(range(num_occ_batches)):
        nocc0 = batch_id * nocc_per_batch
        nocc1 = min(nocc, nocc0 + nocc_per_batch)
        nocc_in_batch = nocc1 - nocc0

        aux0 = aux1 = 0
        j3c_full = cp.zeros((nao, nao, blksize))
        buf = cp.empty((batch_size, nao_pair))
        j3c_00 = cp.empty((naux, nocc_in_batch, nao))
        for kbatch in range(num_aux_batches):
            compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
            naux_in_batch = compressed.shape[1]
            for k0, k1 in lib.prange(0, naux_in_batch, blksize):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                j3c = j3c_full[:, :, :dk]
                fill_symmetric(compressed, _pair_addresses, nao, k0, k1,
                               out=j3c)
                contract('pqr,pi->riq', j3c, _orbo[:, nocc0:nocc1],
                         out=j3c_00[aux0:aux1])
                if j_factor != 0 and it == 0:
                    contract('pqr,pq->r', j3c, dm, out=auxvec[aux0:aux1])
        j3c_full = buf = j3c = compressed = None
        st('A j3c_00')

        if j_factor != 0 and it == 0:
            auxvec = metric_v.dot(auxvec.dot(metric_v) / metric_w)
            auxvec_by_atm = auxvec[aux_idx]

        nw = metric_v.shape[1]
        if metric_inv:
            # item 6: V W^-1 V^T once, then one GEMM per slice instead of two
            if it == 0:
                j2c_inv = (metric_v / metric_w).dot(metric_v.T)
            work = cp.empty((naux, 16, nao))
            for i0, i1 in lib.prange(0, nocc_in_batch, 16):
                tmp = ndarray((naux, i1-i0, nao), buffer=work)
                contract('uv,vij->uij', j2c_inv, j3c_00[:, i0:i1], out=tmp)
                j3c_00[:, i0:i1] = tmp
        else:
            work = cp.empty((nw, 16, nao))
            for i0, i1 in lib.prange(0, nocc_in_batch, 16):
                tmp = ndarray((nw, i1-i0, nao), buffer=work)
                contract('uv,uij->vij', metric_v, j3c_00[:, i0:i1], out=tmp)
                tmp /= metric_w[:, None, None]
                contract('uv,vij->uij', metric_v, tmp, out=j3c_00[:, i0:i1])
        dm_3c, j3c_00 = j3c_00, None
        tmp = work = None

        dm_oo = contract('riq,qj->rij', dm_3c, _orbo) if do_k else None
        st('B metric')

        # (00|0)(1|0)(0|00)
        j2c_10 = int2c2e_ip1(auxmol, sort_output=False, omega=omega)
        inv_aux_sorting = cp.empty(naux, dtype=int)
        inv_aux_sorting[_aux_sorting] = cp.arange(naux)
        j2c_10 = j2c_10[:, inv_aux_sorting[aux_idx, None], inv_aux_sorting]
        buf = cp.empty((3, naux_in_atm, nocc_in_batch, nao))
        buf1 = cp.empty((naux_in_atm, nocc_in_batch, nao))
        for i, (p0, p1) in enumerate(aux_slices):
            if not do_k:
                dm_3c_atm = cp.take(dm_3c, aux_idx[p0:p1], axis=0,
                                    out=buf1[:p1-p0])
                tmp = ndarray((3, p1-p0, nocc_in_batch, nao), buffer=buf)
                j3c_1 = contract('xrs,siq->xriq', j2c_10[:, p0:p1], dm_3c,
                                 out=tmp)
            elif mem_sufficient:
                dm_3c_atm = cp.take(dm_3c, aux_idx[p0:p1], axis=0,
                                    out=buf1[:p1-p0])
                tmp = ndarray((3, p1-p0, nocc_in_batch, nao), buffer=buf)
                j3c_1 = contract('xrs,siq->xriq', j2c_10[:, p0:p1], dm_3c,
                                 out=tmp)
                contract('xrip,riq->xpq', j3c_1, dm_3c_atm, -.5*k_factor,
                         beta=1, out=vhf_atm_ao[i])
            else:
                dm_3c_atm = cp.take(dm_3c, aux_idx[p0:p1], axis=0,
                                    out=buf1[:p1-p0])
                dm_oo_atm = dm_oo[aux_idx[p0:p1]]
                tmp = ndarray((3, p1-p0, nocc_in_batch, nocc), buffer=buf)
                tmp = contract('xrs,sij->xrij', j2c_10[:, p0:p1], dm_oo,
                               out=tmp)
                contract('riq,xrij->xjq', dm_3c_atm, tmp, -.5*k_factor,
                         beta=1, out=vhf_atm[i])
                tmp = ndarray((3, p1-p0, nocc_in_batch, nao), buffer=buf)
                j3c_1 = contract('xrs,siq->xriq', j2c_10[:, p0:p1], dm_3c,
                                 out=tmp)
                contract('xriq,rij->xjq', j3c_1, dm_oo_atm, -.5*k_factor,
                         beta=1, out=vhf_atm[i])
            if j_factor != 0:
                contract('xriq,r->xiq', j3c_1, auxvec_by_atm[p0:p1],
                         j_factor, beta=1, out=vhf_atm[i, :, nocc0:nocc1])
                tmp = cp.einsum('xrs,s->xr', j2c_10[:, p0:p1], auxvec)
                contract('riq,xr->xiq', dm_3c_atm, tmp, j_factor, beta=1,
                         out=vhf_atm[i, :, nocc0:nocc1])
        tmp = j3c_1 = dm_3c_atm = dm_oo_atm = None
        buf = buf1 = j2c_10 = None
        st('C j2c_10')

        # (10|0)(0|0)(0|00)
        work_size = (6*pair_size_max*batch_size +
                     (nao*nao + (nao+nocc) *
                      max(nao_on_atom, nocc_per_batch)) * blksize)
        t_size = nao*nao*blksize if hoist else 0
        s_size = nao*nocc*blksize if occ_k else 0
        # item 4's array (and item 12's two) live at the tail of the same
        # buffer: they are dead before the (00|1) half of each batch reuses
        # that space
        work = cp.empty(work_size + t_size + s_size * (2 if pri_full else 1))
        t_buf = work[work_size:work_size+t_size] if hoist else None
        s_buf = work[work_size+t_size:work_size+t_size+s_size] if occ_k else None
        p_buf = work[work_size+t_size+s_size:] if pri_full else None
        aux0 = aux1 = 0
        for kbatch in range(num_aux_batches):
            naux_in_batch = aux_offsets[kbatch+1] - aux_offsets[kbatch]
            aux0, aux1 = aux1, aux1 + naux_in_batch

            j3c_full, work1 = _allocate((nao, nao, blksize), work)
            j3c_full[:] = 0.

            buf0, work1 = _allocate((3, nao_pair, batch_size), work1)
            buf1, work1 = _allocate((3, nao_pair, batch_size), work1)
            compressed_dk = eval_ipaux(kbatch, out=buf0)
            compressed_di = eval_ip1(kbatch, out=buf1)
            st('D integrals')

            compressed_dk += compressed_di
            compressed_dj = compressed_dk
            compressed_di *= -1
            _aux1 = aux0
            for k0, k1 in lib.prange(0, naux_in_batch, blksize):
                dk = k1 - k0
                _aux0, _aux1 = _aux1, _aux1 + dk
                j3c = j3c_full[:, :, :dk]
                if hoist:
                    # item 4: C_o^T-contracted density for every AO at once,
                    # which upstream recomputes per atom and per x
                    t_all = ndarray((nao, dk, nao), buffer=t_buf)
                    contract('ui,riq->urq', _orbo[:, nocc0:nocc1],
                             dm_3c[_aux0:_aux1], out=t_all)
                if occ_k:
                    s_all = ndarray((nao, dk, nocc), buffer=s_buf)
                    contract('ui,rij->urj', _orbo[:, nocc0:nocc1],
                             dm_oo[_aux0:_aux1], out=s_all)
                for x in range(3):
                    j3c[j_addr, i_addr] = compressed_dj[x, :, k0:k1]
                    j3c[i_addr, j_addr] = compressed_di[x, :, k0:k1]
                    if do_k:
                        tmp = ndarray((nao, dk, nocc_in_batch), buffer=work1)
                        contract('pqr,qi->pri', j3c, _orbo[:, nocc0:nocc1],
                                 out=tmp)
                        contract('pri,riq->pq', tmp, dm_3c[_aux0:_aux1],
                                 -.5*k_factor, beta=1, out=vhf1[x])
                    st('D1 vhf1')
                    if not do_k:
                        pass
                    elif occ_k:
                        # sum_{u in A, r} of the two halves of (X + X^T) C,
                        # each already on the occupied side: upstream's
                        # memory-short branch with both of its intermediates
                        # formed once per block (j3c_pri is D1's when the
                        # occupied index is not batched)
                        if pri_full:
                            j3c_pri = ndarray((nao, dk, nocc), buffer=p_buf)
                            contract('pqr,qi->pri', j3c, _orbo, out=j3c_pri)
                        else:
                            j3c_pri = tmp
                        for i, p0, p1 in small_atoms:
                            contract('uqr,urj->jq', j3c[p0:p1], s_all[p0:p1],
                                     -.5*k_factor, beta=1, out=vhf_atm[i, x])
                            contract('urj,urq->jq', j3c_pri[p0:p1],
                                     t_all[p0:p1], -.5*k_factor, beta=1,
                                     out=vhf_atm[i, x])
                        j3c_pri = None
                    elif hoist:
                        for i, p0, p1 in small_atoms:
                            contract('upr,urq->pq', j3c[p0:p1], t_all[p0:p1],
                                     -.5*k_factor, beta=1,
                                     out=vhf_atm_ao[i, x])
                    elif k_ao:
                        for i, (p0, p1) in enumerate(zip(aoslices[:-1],
                                                         aoslices[1:])):
                            if p1 - p0 < nocc_in_batch:
                                tmp = ndarray((p1-p0, dk, nao), buffer=work1)
                                contract('ui,riq->urq',
                                         _orbo[p0:p1, nocc0:nocc1],
                                         dm_3c[_aux0:_aux1], out=tmp)
                                contract('upr,urq->pq', j3c[p0:p1], tmp,
                                         -.5*k_factor, beta=1,
                                         out=vhf_atm_ao[i, x])
                            else:
                                tmp = ndarray((nao, dk, nocc_in_batch),
                                              buffer=work1)
                                contract('pqr,pi->qri', j3c[p0:p1],
                                         _orbo[p0:p1, nocc0:nocc1], out=tmp)
                                contract('pri,riq->pq', tmp,
                                         dm_3c[_aux0:_aux1], -.5*k_factor,
                                         beta=1, out=vhf_atm_ao[i, x])
                    else:
                        for i, (p0, p1) in enumerate(zip(aoslices[:-1],
                                                         aoslices[1:])):
                            tmp, tmp1 = _allocate((p1-p0, dk, nocc), work1)
                            tmp1 = ndarray((p1-p0, dk, nao), buffer=tmp1)
                            contract('ui,rij->urj', _orbo[p0:p1, nocc0:nocc1],
                                     dm_oo[_aux0:_aux1], out=tmp)
                            contract('uqr,urj->jq', j3c[p0:p1], tmp,
                                     -.5*k_factor, beta=1, out=vhf_atm[i, x])
                            contract('uqr,qj->urj', j3c[p0:p1], _orbo,
                                     out=tmp)
                            contract('ui,riq->urq', _orbo[p0:p1, nocc0:nocc1],
                                     dm_3c[_aux0:_aux1], out=tmp1)
                            contract('urj,urq->jq', tmp, tmp1, -.5*k_factor,
                                     beta=1, out=vhf_atm[i, x])
                    st('D2 K per-atom')
                    if j_factor != 0:
                        if batch_id == 0:
                            contract('pqr,r->pq', j3c, auxvec[_aux0:_aux1],
                                     j_factor, beta=1, out=vhf1[x])
                        if jbatch:
                            # item 5: sum_{p in A, q} j3c[p,q,r] dm[p,q] for
                            # every atom A is a row-wise contraction and a
                            # GEMM against the atom indicator, and the
                            # per-atom update is then one contraction
                            w_pr = contract('pqr,pq->pr', j3c, dm)
                            av_atm = atm_ind.dot(w_pr)
                            contract('riq,ar->aiq', dm_3c[_aux0:_aux1],
                                     av_atm, 2*j_factor, beta=1,
                                     out=vhf_atm[:, x, nocc0:nocc1])
                        else:
                            for i, (p0, p1) in enumerate(zip(aoslices[:-1],
                                                             aoslices[1:])):
                                auxvec1 = cp.einsum('pqr,pq->r', j3c[p0:p1],
                                                    dm[p0:p1])
                                contract('riq,r->iq', dm_3c[_aux0:_aux1],
                                         auxvec1, 2*j_factor, beta=1,
                                         out=vhf_atm[i, x, nocc0:nocc1])
                    st('D3 J per-atom')

            # (00|1)(0|0)(0|00)
            compressed_dk += compressed_di
            j3c_aux, work1 = _allocate((3, naux_in_batch, nocc_in_batch, nao),
                                       work)
            off = max(j3c_full.size + compressed_dk.size, j3c_aux.size)
            j3c_aux_tmp = ndarray((3, naux_in_batch, nocc_in_batch, nao),
                                  buffer=work[off:])
            if j_factor != 0:
                auxvec_ipaux = cp.empty((3, naux_in_batch))
            for k0, k1 in lib.prange(0, naux_in_batch, blksize):
                dk = k1 - k0
                j3c = j3c_full[:, :, :dk]
                for i in range(3):
                    fill_symmetric(compressed_dk[i], _pair_addresses, nao, k0,
                                   k1, out=j3c)
                    contract('pqr,pi->riq', j3c, _orbo[:, nocc0:nocc1],
                             alpha=-1, out=j3c_aux_tmp[i, k0:k1])
                    if j_factor != 0:
                        contract('pqr,pq->r', j3c, dm, alpha=-1,
                                 out=auxvec_ipaux[i, k0:k1])
            idx = np.argsort(aux_filling_order[aux0:aux1])
            j3c_aux = cp.take(j3c_aux_tmp, idx, axis=1, out=j3c_aux)

            dm_3c_batch, work1 = _allocate(
                (naux_in_batch, nocc_in_batch, nao), work1)
            dm_oo_batch, work1 = _allocate(
                (naux_in_batch, nocc_in_batch, nocc), work1)
            dm_3c_batch = cp.take(dm_3c[aux0:aux1], idx, axis=0,
                                  out=dm_3c_batch)
            if do_k:
                dm_oo_batch = cp.take(dm_oo[aux0:aux1], idx, axis=0,
                                      out=dm_oo_batch)
            if j_factor != 0:
                auxvec_batch = auxvec[aux0:aux1][idx]
                auxvec_ipaux = auxvec_ipaux[:, idx]

            counts = np.bincount(atm_id_for_aux[aux0:aux1])
            oo_buf = None
            if work1.size > 3*counts.max()*nocc*nocc_in_batch:
                oo_buf = work1
            st('E ipaux fill')

            p0 = p1 = 0
            for i, count in enumerate(counts):
                if count == 0:
                    continue
                p0, p1 = p1, p1 + count
                if not do_k:
                    pass
                elif mem_sufficient:
                    contract('xrip,riq->xpq', j3c_aux[:, p0:p1],
                             dm_3c_batch[p0:p1], -.5*k_factor, beta=1,
                             out=vhf_atm_ao[i])
                else:
                    tmp = ndarray((3, p1-p0, nocc_in_batch, nocc),
                                  buffer=oo_buf)
                    j3c_1 = contract('xriq,qj->xrij', j3c_aux[:, p0:p1], _orbo,
                                     out=tmp)
                    contract('riq,xrij->xjq', dm_3c_batch[p0:p1], j3c_1,
                             -.5*k_factor, beta=1, out=vhf_atm[i])
                    contract('xriq,rij->xjq', j3c_aux[:, p0:p1],
                             dm_oo_batch[p0:p1], -.5*k_factor, beta=1,
                             out=vhf_atm[i])
                if j_factor != 0:
                    contract('riq,xr->xiq', dm_3c_batch[p0:p1],
                             auxvec_ipaux[:, p0:p1], j_factor, beta=1,
                             out=vhf_atm[i, :, nocc0:nocc1])
                    contract('xriq,r->xiq', j3c_aux[:, p0:p1],
                             auxvec_batch[p0:p1], j_factor, beta=1,
                             out=vhf_atm[i, :, nocc0:nocc1])
            st('E per-aux-atom')
        work = work1 = buf0 = buf1 = oo_buf = dm_3c_batch = dm_oo_batch = None
        j3c_full = j3c = j3c_1 = j3c_aux = j3c_aux_tmp = tmp = tmp1 = None
        compressed_di = compressed_dj = compressed_dk = None
        dm_oo = dm_3c = None

    if k_ao:
        transpose_sum(vhf_atm_ao.reshape(-1, nao, nao), inplace=True)
        contract('nxpq,qj->nxjp', vhf_atm_ao, _orbo, beta=1, out=vhf_atm)

    for i, (p0, p1) in enumerate(zip(aoslices[:-1], aoslices[1:])):
        contract('xpq,pi->xiq', vhf1[:, p0:p1], orbo[p0:p1], beta=1,
                 out=vhf_atm[i])
        contract('xpq,qi->xip', vhf1[:, p0:p1], orbo, beta=1,
                 out=vhf_atm[i, :, :, p0:p1])
    vhf_atm *= 2
    vhf_atm = contract('nxjp,pi->nxij', vhf_atm, mo_coeff)
    st('F finish')
    st.report()
    log.timer_debug1('_get_veff (fastdfhessip)', *t0)
    return vhf_atm


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED:
        return
    DFHESS._jk_energy_per_atom_orig = _orig_jk_energy_per_atom
    DFHESS._get_veff_orig = _orig_get_veff
    DFHESS._jk_energy_per_atom = _jk_energy_per_atom
    DFHESS._get_veff = _get_veff
    # W25: the unrestricted twins, df/hessian/uhf.py
    from . import _dfhessip_u
    _dfhessip_u.apply_patch()
    _PATCHED = True


apply_patch()
