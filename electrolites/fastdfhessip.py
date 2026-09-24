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

What falls through to GPU4PySCF: more than one device, and ``FASTDFHESSIP_OFF``.
The unrestricted twins live in ``df/hessian/uhf.py`` and are not covered.

    FASTDFHESSIP_OFF=1       hand every call back to GPU4PySCF
    FASTDFHESSIP_SPMULT=<n>  item 1: shell pairs per thread block, as a
                             multiple of the scheme's; 4 is upstream's
    FASTDFHESSIP_OOSUM=gram|gemm|transpose
                             item 2: the Gram form (default), ``X`` with the
                             sum as two contractions, or upstream's
    FASTDFHESSIP_FUSE=0      item 3 off: two ``ipaux`` passes, as upstream
    FASTDFHESSIP_H1ATOM=upstream, FASTDFHESSIP_H1J=upstream,
    FASTDFHESSIP_JPAIR=0     items 4, 5 and 7 off
    FASTDFHESSIP_H1METRIC=inverse
                             item 6 on
    FASTDFHESSIP_IP2_GOUT=<n>
                             the ``ip2`` kernel's ``gout_width`` (upstream's
                             ``None``; swept, +-3 %)
    FASTDFHESSIP_TIME=1      print a synchronised stage table per call
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
    int2c2e_ip1, _check_rsh_factors, int3c2e_scheme_ip1,
    int3c2e_scheme_ipaux)
from gpu4pyscf.df.hessian import rhf as DFHESS
from gpu4pyscf.df.hessian.rhf import (
    _factorize_dm, _factorize_j2c, _bas_atom_labels, _argsort_aux_by_atom,
    _int2c2e_ip2_per_atom, int3c2e_scheme_ip2, _allocate,
    _balance_batch_size)
from gpu4pyscf.hessian.rhf import _aggregate_to_atoms
from gpu4pyscf.lib import multi_gpu

from ._dfaux import _spmult


def _env(name, default):
    return os.environ.get(name, default)


_S = {'ip2': 0, 'h1': 0, 'declined': 0, 'fused': 0, 'vchunks': 0,
      'occ_batches': 0}


def stats():
    return dict(_S)


_orig_jk_energy_per_atom = DFHESS._jk_energy_per_atom
_orig_get_veff = DFHESS._get_veff


class _Stages:
    """Synchronised stage timers, printed when FASTDFHESSIP_TIME is set."""

    def __init__(self, tag):
        self.on = bool(int(_env('FASTDFHESSIP_TIME', '0')))
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

    def report(self):
        if not self.on:
            return
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

    def evaluate_j3c(batch_id, out=None):
        ksh0 = ksh_offsets_cpu[batch_id]
        ksh1 = ksh_offsets_cpu[batch_id + 1]
        aux_ao_offset = aux_loc[ksh0]
        naux = aux_loc[ksh1] - aux_ao_offset
        out = ndarray((3, nao_pair, naux), buffer=out)
        if out.size == 0:
            return out
        err = kernel(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.byref(int3c2e_envs),
            ctypes.c_double(omega),
            ctypes.c_double(lr_factor), ctypes.c_double(sr_factor),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(npair_batches),
            ctypes.c_int(1),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
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

    metric_w, metric_v = _factorize_j2c(auxmol, aux_sorting, omega)
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
        if fuse:
            # Fused, the first pass rides on the second pass's *first*
            # chunk and the h_aux block runs after that chunk's integrals,
            # while its buffers are held rather than before they are
            # allocated -- so the plan is made against what that block's
            # temporaries leave, about 24 naux^2 words at their peak, and
            # taken only if that does not cost the second pass a chunk: a
            # chunk is a whole walk over the integrals, and the fusion saves
            # exactly one.
            try:
                plan_f = _plan_v(word_avail - 24 * naux**2, naux, nao, nocc,
                                 nocc_in_batch, nao_pair, batch_size,
                                 nao_on_atom, natm, metric_size, batch_id)
            except AssertionError:
                plan_f = (0,)
            if plan_f[0] > 0 and (-(-metric_size // plan_f[0]) <=
                                  -(-metric_size // plan[0])):
                plan, do_fuse = plan_f, True
        aux_batch_size, aux_blksize, blksize2 = plan
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
            st('D2 h_aux')
        else:
            j3c_oo1p = j3c_oo1
            j2c_10v = None

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
                        contract('rs,pir->spi', _metric_v[aux0:aux1, v0:v1],
                                 tmp, beta=1, out=j3c_100[x])
                st('D3 j3c_100')
            j3c_full = buf0 = buf1 = None
            compressed_di = compressed_dj = compressed_dk = None

            if do_fuse and v0 == 0:
                # the h_aux block that upstream runs between the passes
                j3c_oo1p, h_aux_parts = _h_aux_block(
                    auxmol, omega, _aux_sorting, _metric_v, _metric_w, dm_oo,
                    j3c_oo1, naux, metric_size, k_factor, j_factor, batch_id,
                    auxvec_ipauxp if j_factor != 0 else None,
                    _auxvec if j_factor != 0 else None, natm, aux_atm_labels)
                ejk_ao += h_aux_parts[0]
                j2c_10v = h_aux_parts[1]
                st('D2 h_aux')

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
        j2c_10v = j3c_oo1p = j3c_oo1 = None

    ejk_ao_aux = _aggregate_to_atoms(h_ao_aux, natm, aux_atm_labels, axis=1)
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


def _h_aux_block(auxmol, omega, _aux_sorting, _metric_v, _metric_w, dm_oo,
                 j3c_oo1, naux, metric_size, k_factor, j_factor, batch_id,
                 auxvec_ipauxp, _auxvec, natm, aux_atm_labels):
    """Upstream's code between the two passes, unchanged."""
    j2c_10 = int2c2e_ip1(auxmol, sort_output=False, omega=omega)
    j2c_10 *= -1
    j2c_10, tmp = cp.empty_like(j2c_10), j2c_10
    j2c_10[:, _aux_sorting[:, None], _aux_sorting] = tmp
    j2c_10v = contract('xrs,st->xrt', j2c_10, _metric_v)
    tmp = None

    j3c_oo1p = contract('xuv,vij->xuij', j2c_10, dm_oo, alpha=-1, beta=1,
                        out=j3c_oo1)
    h_aux = contract('xrij,ysij->xrys', j3c_oo1p, j3c_oo1p, -.5*k_factor)
    if j_factor != 0 and batch_id == 0:
        contract('xuv,v->xu', j2c_10, _auxvec, -1, 1, out=auxvec_ipauxp)
        contract('xr,ys->xrys', auxvec_ipauxp, auxvec_ipauxp, j_factor, 1,
                 h_aux)
    j2c_inv = (_metric_v / _metric_w).dot(_metric_v.T)
    h_aux *= j2c_inv[None, :, None, :]

    dm_aux = contract('rij,sij->rs', dm_oo, dm_oo, alpha=-.5*k_factor)
    if j_factor != 0 and batch_id == 0:
        contract('r,s->rs', _auxvec, _auxvec, j_factor, 1, dm_aux)
    ejk_aux = _int2c2e_ip2_per_atom(
        auxmol, dm_aux[_aux_sorting[:, None], _aux_sorting], omega)
    ejk = -ejk_aux

    j2c_ip2 = None
    for x in range(3):
        j2c_10v_w = j2c_10v[x] / _metric_w
        for y in range(3):
            j2c_ip2 = contract('rs,ts->rt', j2c_10v_w, j2c_10v[y], out=j2c_ip2)
            j2c_ip2 *= dm_aux
            h_aux[x, :, y] += j2c_ip2
    dm_aux = j2c_ip2 = j2c_10v_w = None

    dm_aux1 = contract('xrij,sij->xrs', j3c_oo1p, dm_oo, -.5*k_factor)
    if j_factor != 0 and batch_id == 0:
        contract('xr,t->xrt', auxvec_ipauxp, _auxvec, j_factor, 1, out=dm_aux1)
    tmp = contract('xrt,st->xrs', j2c_10v, _metric_v/_metric_w)
    h_aux = contract('xrt,ytr->xryt', tmp, dm_aux1, alpha=-2, beta=1,
                     out=h_aux)
    dm_aux1 = j2c_10 = tmp = None

    h_aux = transpose_sum(h_aux.reshape(3*naux, 3*naux), inplace=True)
    h_aux = h_aux.reshape(3, naux, 3, naux).transpose(1, 3, 0, 2)
    ejk_aux = _aggregate_to_atoms(h_aux, natm, aux_atm_labels, axis=(0, 1))
    ejk += ejk_aux * .5
    return j3c_oo1p, (ejk, j2c_10v)


def _oo_gram(j3c_100, v0, dv, work, natm, nocc, atm_ind, _dm_factor_l,
             _metric_w, _metric_v, k_factor, j_factor, ejk, h_ao_aux,
             j2c_10v, _auxvec, auxvec_ipauxp, naux):
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
    per_r = 3*nao*nocc + 6*nao*nao
    # the chunk is bounded by the buffer upstream sized for X, and by a
    # size at which the two cuTENSOR contractions below plan (W21: past a
    # few 1e8 elements `createPlan` can refuse a contraction outright);
    # `_accumulate_r` halves along r if one is refused anyway
    kb = max(1, min(dv, work.size // per_r, int(1.2e8 // (3*nao*nao))))
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
        if j_factor != 0:
            # the trace of S: 2 sum_{u in A, i} J[x,r,u,i] C[u,i]
            d = contract('xrui,ui->xru', j3c_100[:, k0:k1], _dm_factor_l)
            auxvec_100_atm = contract('au,xru->axr', atm_ind, d, alpha=2.)
            w_part = _metric_w[v0+k0:v0+k1]
            auxvec_100_atm_w = auxvec_100_atm / w_part
            contract('pxr,qyr->pqxy', auxvec_100_atm_w, auxvec_100_atm,
                     j_factor, beta=1, out=ejk)
            tmp = cp.empty((3, naux, dk))
            cp.multiply(j2c_10v[:, :, v0+k0:v0+k1], _auxvec[:, None], out=tmp)
            contract('ys,sr->ysr', auxvec_ipauxp, _metric_v[:, v0+k0:v0+k1],
                     beta=-1, out=tmp)
            contract('pxr,ysr->psxy', auxvec_100_atm_w, tmp, j_factor, 1,
                     h_ao_aux)
            d = tmp = auxvec_100_atm = auxvec_100_atm_w = None
    js = m = None
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
            ctypes.cast(ksh_offsets_gpu[kbatch:].data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(aux_ao_offset),
            ctypes.c_int(naux_in_batch))
        if err != 0:
            raise RuntimeError('ejk_int3c2e_ip2 failed')
        st('C ip2 kernel')
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
    if mem_sufficient:
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
    hoist = mem_sufficient and _env('FASTDFHESSIP_H1ATOM', 'batched') != 'upstream'
    blk_unit = nao**2 * (2 if hoist else 1) + (nao+nocc)*max(nao_on_atom, nocc)
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
    if mem_sufficient:
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

        dm_oo = contract('riq,qj->rij', dm_3c, _orbo)
        st('B metric')

        # (00|0)(1|0)(0|00)
        j2c_10 = int2c2e_ip1(auxmol, sort_output=False, omega=omega)
        inv_aux_sorting = cp.empty(naux, dtype=int)
        inv_aux_sorting[_aux_sorting] = cp.arange(naux)
        j2c_10 = j2c_10[:, inv_aux_sorting[aux_idx, None], inv_aux_sorting]
        buf = cp.empty((3, naux_in_atm, nocc_in_batch, nao))
        buf1 = cp.empty((naux_in_atm, nocc_in_batch, nao))
        for i, (p0, p1) in enumerate(aux_slices):
            if mem_sufficient:
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
        # item 4's array lives at the tail of the same buffer: it is dead
        # before the (00|1) half of each batch reuses that space
        work = cp.empty(work_size + t_size)
        t_buf = work[work_size:] if hoist else None
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
                for x in range(3):
                    j3c[j_addr, i_addr] = compressed_dj[x, :, k0:k1]
                    j3c[i_addr, j_addr] = compressed_di[x, :, k0:k1]
                    tmp = ndarray((nao, dk, nocc_in_batch), buffer=work1)
                    contract('pqr,qi->pri', j3c, _orbo[:, nocc0:nocc1],
                             out=tmp)
                    contract('pri,riq->pq', tmp, dm_3c[_aux0:_aux1],
                             -.5*k_factor, beta=1, out=vhf1[x])
                    st('D1 vhf1')
                    if hoist:
                        for i, p0, p1 in small_atoms:
                            contract('upr,urq->pq', j3c[p0:p1], t_all[p0:p1],
                                     -.5*k_factor, beta=1,
                                     out=vhf_atm_ao[i, x])
                    elif mem_sufficient:
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
                if mem_sufficient:
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

    if mem_sufficient:
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
    _PATCHED = True


apply_patch()
