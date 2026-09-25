"""The three-centre Coulomb build a *pure* density-fitted functional runs on.

`docs/DF_WORK_LIST.md` **W9**.  A DF-PBE, DF-BLYP or DF-TPSS run never builds a
cderi: ``df/df.py:111`` returns ``df_jk.get_j`` before the tensor is touched,
so the whole two-electron half of such a calculation is one function --

    dm --> mol.apply_C_mat_CT          decontract the density
       --> Int3c2eOpt.contract_dm      rho_P = sum_ij (ij|P) D_ij
       --> auxmol.apply_CT_dot
       --> two triangular solves with the Coulomb metric
       --> auxmol.apply_C_dot
       --> Int3c2eOpt.contract_auxvec  vj_ij = sum_P (ij|P) rho_P
       --> mol.apply_CT_mat_C

-- and `fastdf`, which rearranges the *cderi* contraction, is never called at
all.  The same function is also the Coulomb half of every RIJONX run
(`fastdfj`), which makes it the Coulomb build of the largest hybrid this
repository can run.

**What this module is not.**  `docs/README_fastdfj.md` §3.1 measured the
McMurchie--Davidson integral engine underneath all of that against the direct
four-centre build and found it **1.67x-2.43x faster** at every size on the
ladder.  The kernels are good and this module does not touch them.  What it
touches is everything around them, because `benchmarks/df_j3c_probe.py` says
that on a 10-atom molecule the two integral kernels are **28 %** of the
Coulomb build and the other 72 % is host-side setup, density transformations
and five host-to-device copies of tables that never change.

**What it changes.**  Three things, none of them arithmetic.

1. *The per-call constants are uploaded once.*  ``contract_dm`` and
   ``contract_auxvec`` between them copy the shell-pair list, the block
   offsets, the Hermite pair offsets and the auxiliary ``ao_loc`` to the
   device on **every** call, re-run a ``np.cumsum`` over the auxiliary shells
   and rebuild the 16-shell load-balancing split in a Python loop.  Every one
   of those is fixed at ``Int3c2eOpt.build()``.  They move there
   (`electrolites/_j3c.py`).

2. *The recontraction tables are uploaded once.*  ``SortedGTO.c_ao_loc`` is a
   **property** -- it recomputes a NumPy cumsum and copies it to the device
   every time it is read -- and each of the four recontraction kernels reads
   it plus four more host arrays.  A single ``get_j`` reads six of those
   methods, so thirty host-to-device copies and six cumsums per Coulomb
   matrix.  They are written once in ``SortedGTO.from_mol`` and never again,
   so they are cached on the object.  This one is not only W9's: the cderi
   build calls ``mat_dot_C`` once per auxiliary block
   (`docs/README_fastdfcderi.md`), and so does every ``get_hcore``.

3. *The scratch of the two contractions is allocated once.*  Both build a
   Hermite work buffer per call and free it again -- and
   `docs/DF_WORK_LIST.md` §5.8 is this repository's own lesson that such a
   temporary does not cost its allocation, it costs the ``cudaFree`` that ends
   it, which is a synchronisation point.  The three that the caller never sees
   are kept on the option object; the two that are returned are not.

4. *The Hermite transformations stop re-uploading ``ao_loc``.*
   ``scf/j_engine.py:_dm_to_Rt`` and ``_Rt_to_dm`` take the pair list and the
   pair offsets as device arrays and then copy ``mol.ao_loc`` from the host
   themselves.  This module calls the same two CUDA kernels with the cached
   copy instead; the direct path's own ``_dm_to_Rt`` is left alone.

5. *The auxiliary-vector kernel is launched with enough thread blocks.*
   ``Int3c2eOpt.build`` makes exactly **one** block range per
   angular-momentum group, and that kernel's grid is
   ``(groups x auxiliary-shell batches)`` -- six by sixty-two on penicillin V,
   i.e. **372 thread blocks on a 108-SM A100**, with the groups differing in
   size by seventy times (11 306 shell pairs against 159).  Both kernels read
   their group's ``(li,lj)`` from the first pair of their range and stride the
   rest, so a range may be any sub-interval of a group: this is a launch
   configuration change and **not** a kernel change, the same shape as W3's
   ``nsp_per_block*4`` finding.  The density contraction is left alone,
   because its grid already carries the auxiliary shells and splitting only
   costs it an extra warp reduction per block -- ``df_j3c_sweep.py`` measures
   both and `docs/README_fastdfj3c.md` §5 is the table.

6. *More than one density goes to the McMurchie--Davidson engine too.*
   ``df_jk.get_j`` uses ``j_engine_3c2e`` only when it is handed a **single**
   density and sends every other call to the *Rys* ``int3c2e_bdiv`` engine
   instead (``df/df_jk.py:594``).  That looks like the right choice, because
   the Rys engine carries the density index **inside** one kernel launch where
   the MD engine loops over densities in Python and rebuilds the integrals for
   each -- but measured, the Rys kernels are an order of magnitude the slower
   of the two, and looping the faster engine wins even though it repeats the
   integral work.  `benchmarks/df_j3c_probe.py` has the two side by side and
   `docs/README_fastdfj3c.md` §4 the ratio on the ladder.  What reaches this
   is every **response** property of a pure density-fitted functional --
   ``scf/_response_functions.py`` asks for one density per right-hand side --
   and the RIJONX response solve with it.

7. *The two integral kernels are this repository's* (`docs/DF_WORK_LIST.md`
   W19).  After items 1-6 the call is 85-97 % two kernels, and
   `electrolites/codegen/gen_j3c.py` writes both again, one function per
   ``(lij, lk)`` class with one thread carrying one shell pair through a
   batch of auxiliary shells in registers: the pair's product centre and
   ``1/aij`` read from a table instead of recomputed per auxiliary shell (no
   division is left), every Hermite index a literal, ``omega == 0`` compiled
   in, and the auxiliary Hermite components of the wrong parity -- zero by
   construction -- never formed.  Classes are grouped into tiers of similar
   register need, one launch per tier (`electrolites/_j3ck.py`).  Shell-pair
   groups that meet a class past ``FASTDFJ3C_KMAX_ORDER`` stay on
   GPU4PySCF's kernels, launched with those groups' offsets alone.

8. *A single-density call can be built from the difference to the previous
   one, screened on it* (W20) -- **off**, because measured against item 7 it
   is a loss on every whole run (0.89x-0.97x, `docs/README_fastdfj3cinc.md`):
   once the kernels are this cheap the pairs a Schwarz screen drops are worth
   less than the screen's host work and the extra SCF cycle it can cost.

Items 1-6 are a copy, not an approximation: the same kernels are launched
with the same arguments in the same order.  Item 7 evaluates the same
integrals in a different order and with ``rsqrt`` for two divisions, so it
agrees to round-off rather than bit for bit; ``tests/test_dfj3c.py`` checks
the Coulomb matrix element-wise against an unpatched binding in the same
process either way.

Switches, which are what every ablation column of
``docs/README_fastdfj3c.md`` is measured with:

    FASTDFJ3C_OFF=1            hand every call back to GPU4PySCF
    FASTDFJ3C_NO_CTR=1         keep GPU4PySCF's recontraction kernels
                               (item 2), i.e. measure the rest alone
    FASTDFJ3C_NO_OPT=1         keep GPU4PySCF's two contractions (items 1,
                               3, 4 and 5), i.e. measure item 2 alone
    FASTDFJ3C_NO_MDM=1         keep GPU4PySCF's engine choice (item 6)
    FASTDFJ3C_SP_CHUNK_AUX=0   keep GPU4PySCF's whole-group thread blocks
                               (item 5)
    FASTDFJ3C_SP_CHUNK_DM=<n>  the same knob on the other kernel, where the
                               measured answer is to leave it at 0
    FASTDFJ3C_KSH_CHUNK=<n>    auxiliary shells per block; 16 is GPU4PySCF's
                               and is the measured optimum
    FASTDFJ3C_MDM_MAX_NDM=<n>  send more than `n` densities back to the Rys
                               engine.  No crossing was found up to 32, so
                               the default does not fire
    FASTDFJ3C_SHM_FRAC=<x>     scale the dynamic shared memory the kernels
                               ask for.  A rejected idea kept runnable: the
                               launch asks for 91 KB per block, which is one
                               block per A100 SM, and cutting it is
                               monotonically *slower* -- these kernels want
                               shell pairs in flight, not blocks per SM
    FASTDFJ3C_KERNELS=upstream keep GPU4PySCF's two integral kernels
                               (item 7)
    FASTDFJ3C_KMAX_ORDER=<n>   the highest lij + lk item 7's kernels serve;
                               a shell-pair group that meets a higher class
                               is left to GPU4PySCF's kernels whole
    FASTDFJ3C_INCREMENTAL=1    item 8, the build from the difference;
                               FASTDFJ3C_INC_TOL (1e-11), _INC_REBUILD (8)
                               and _INC_MAXFRAC (0.1) are its threshold, the
                               builds between full ones and the difference
                               past which a call is built whole
    FASTDFJ3C_PER_CLASS=1      the other rejected idea: one launch of the
                               auxiliary-vector kernel per (shell-pair class,
                               auxiliary class), each asking only for the
                               shared memory that class lays out.  The
                               launches serialise on the default stream and
                               it is 1.4x-5x slower
"""
import ctypes
import functools
import os

import numpy as np
import cupy as cp

from gpu4pyscf.df import df_jk as DFJK
from gpu4pyscf.df import j_engine_3c2e as J3C
from gpu4pyscf.df.int3c2e_bdiv import int2c2e, LMAX, L_AUX_MAX, THREADS
from gpu4pyscf.scf.jk import SHM_SIZE, _nearest_power2
from gpu4pyscf.gto import mole as GMOL
from gpu4pyscf.gto.mole import SortedGTO, libvhf_rys
from gpu4pyscf.lib.cupy_helper import ndarray, transpose_sum

from . import _j3c
from . import _j3ck
from . import _j3cinc

_OFF = bool(os.environ.get('FASTDFJ3C_OFF'))
_NO_CTR = bool(os.environ.get('FASTDFJ3C_NO_CTR'))
_NO_OPT = bool(os.environ.get('FASTDFJ3C_NO_OPT'))
_NO_MDM = bool(os.environ.get('FASTDFJ3C_NO_MDM'))
#: Shell pairs per thread block of each of the two integral kernels.  0 is
#: GPU4PySCF's own -- one block per angular-momentum group, however many pairs
#: that is.  The two want opposite things and both defaults are measured
#: (`docs/README_fastdfj3c.md` §5, `benchmarks/df_j3c_sweep.py`): splitting
#: costs the density contraction a warp reduction and an `atomicAdd` per
#: block and buys it nothing, because its grid already carries the auxiliary
#: shells; the auxiliary-vector kernel's grid does not, and whole groups leave
#: it with a few hundred blocks on a 108-SM card.
SP_CHUNK_DM = int(os.environ.get('FASTDFJ3C_SP_CHUNK_DM', 0))
#: ``-1`` derives the auxiliary-vector kernel's block size from the molecule
#: and the basis (``Int3c2eConstants.auto_chunk``), because the swept optimum
#: moves by a factor of thirty-two across the size ladder -- 256 shell pairs
#: per block at ten atoms, 8192 at 284.  ``0`` is GPU4PySCF's whole groups and
#: a positive value fixes it.
SP_CHUNK_AUX = int(os.environ.get('FASTDFJ3C_SP_CHUNK_AUX', -1))
#: the three constants of that rule, all measured
#: (`benchmarks/df_j3c_sweep.py`, seven systems in three bases): the thread
#: blocks the kernel is aimed at, and the floor and ceiling of the block size.
AUTO_TARGET_BLOCKS = int(os.environ.get('FASTDFJ3C_TARGET_BLOCKS', 4000))
AUTO_CHUNK_MIN = int(os.environ.get('FASTDFJ3C_CHUNK_MIN', 256))
AUTO_CHUNK_MAX = int(os.environ.get('FASTDFJ3C_CHUNK_MAX', 8192))
#: auxiliary shells per thread block of the auxiliary-vector kernel.  16 is
#: GPU4PySCF's, and it is the other half of the same launch geometry: a larger
#: batch repeats each shell pair's Gaussian product centre fewer times, a
#: smaller one gives the kernel more blocks.  Swept alongside `SP_CHUNK_AUX`.
KSH_CHUNK = int(os.environ.get('FASTDFJ3C_KSH_CHUNK', 16))
#: the largest number of densities the McMurchie--Davidson engine is used for.
#: It rebuilds the integrals once per density where the Rys engine carries the
#: density index inside one launch, so the two must cross somewhere; where, is
#: measured rather than reasoned (`docs/README_fastdfj3c.md` §6) and this is
#: where the measurement is kept.
MDM_MAX_NDM = int(os.environ.get('FASTDFJ3C_MDM_MAX_NDM', 1 << 30))
#: multiplier on the dynamic shared memory the two kernels are launched with.
#: Both schemes size the per-block shell-pair count from `scf/jk.py:SHM_SIZE`
#: and then request the **largest** class's requirement for the whole launch,
#: so one wide angular-momentum class sets the occupancy of every block in the
#: grid.  1.0 is GPU4PySCF's.  Swept in `docs/README_fastdfj3c.md` §5.2.
SHM_FRAC = float(os.environ.get('FASTDFJ3C_SHM_FRAC', 1.0))
#: 0 restores one launch of the auxiliary-vector kernel for the whole basis,
#: which is GPU4PySCF's -- and with it the widest angular-momentum class's
#: shared-memory request for every block in the grid.
PER_CLASS = int(os.environ.get('FASTDFJ3C_PER_CLASS', 0))
#: 'ours' launches `electrolites/_j3ck.py`'s kernels for every class they
#: serve (item 7); 'upstream' keeps GPU4PySCF's for all of them.
KERNELS = os.environ.get('FASTDFJ3C_KERNELS', 'ours')
#: item 8, `docs/DF_WORK_LIST.md` W20: build a single-density Coulomb matrix
#: from the difference to the previous one, screened on it.  Off: a measured
#: loss once item 7's kernels are in (`docs/README_fastdfj3cinc.md`).  The threshold on
#: the Schwarz bound of what a dropped pair contributes, the full builds
#: between which the error may accumulate, and the size of a difference past
#: which the call is not an SCF step (a response density, a new geometry) and
#: is built whole.
INCREMENTAL = os.environ.get('FASTDFJ3C_INCREMENTAL', '0') != '0'
INC_TOL = float(os.environ.get('FASTDFJ3C_INC_TOL', 1e-11))
INC_REBUILD = int(os.environ.get('FASTDFJ3C_INC_REBUILD', 8))
INC_MAXFRAC = float(os.environ.get('FASTDFJ3C_INC_MAXFRAC', 0.1))

_libmd = J3C.libvhf_md

#: counters, so a coverage claim is read off a run rather than assumed
STATS = {'inc_full': 0, 'inc_step': 0, 'inc_pairs_dm': 0, 'inc_pairs_aux': 0,
         'inc_pairs_all': 0,
         'contract_dm': 0, 'contract_auxvec': 0, 'ctr': 0, 'passthrough': 0,
         'get_j': 0, 'get_j_md': 0, 'get_j_rys': 0}


def stats():
    """What this module did in this process."""
    return dict(STATS)


# --------------------------------------------------------------------------
# 1, 3, 4 and 5.  The two contractions: constants and scratch hoisted to build
# time, and the auxiliary-vector kernel's thread blocks split.
# --------------------------------------------------------------------------

_orig_contract_dm = J3C.Int3c2eOpt.contract_dm
_orig_contract_auxvec = J3C.Int3c2eOpt.contract_auxvec


def _dm_to_Rt(c, dm, out):
    """``scf/j_engine.py:_dm_to_Rt``'s device branch, without the uploads."""
    err = _libmd.dm_to_Rt(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(dm.data.ptr, ctypes.c_void_p),
        ctypes.c_int(1), ctypes.byref(c.envs),
        ctypes.cast(c.shl_pair_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(c.pair_loc.data.ptr, ctypes.c_void_p),
        ctypes.c_int(c.npair),
        ctypes.cast(c.ao_loc.data.ptr, ctypes.c_void_p))
    if err != 0:
        raise RuntimeError('dm_to_Rt kernel failed')
    return out


def _Rt_to_dm(c, dm_xyz, out):
    """``scf/j_engine.py:_Rt_to_dm``'s device branch, without the uploads."""
    out[:] = 0.
    err = _libmd.Rt_to_dm(
        ctypes.cast(out.data.ptr, ctypes.c_void_p),
        ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
        ctypes.c_int(1), ctypes.byref(c.envs),
        ctypes.cast(c.shl_pair_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(c.pair_loc.data.ptr, ctypes.c_void_p),
        ctypes.c_int(c.npair),
        ctypes.cast(c.ao_loc.data.ptr, ctypes.c_void_p))
    if err != 0:
        raise RuntimeError('Rt_to_dm kernel failed')
    return transpose_sum(out)


@functools.lru_cache(maxsize=None)
def _scheme(kind, frac):
    """``_int3c2e_{dm,auxvec}_scheme`` with a scaled shared-memory budget.

    GPU4PySCF's two schemes are copied exactly at ``frac = 1.0`` -- they are
    reproduced here rather than called, because the only thing that changes is
    the ``SHM_SIZE`` they read, and that is a module constant.
    """
    orig = (J3C._int3c2e_dm_scheme if kind == 'dm'
            else J3C._int3c2e_auxvec_scheme)
    if frac == 1.0:
        return orig()
    budget = int(SHM_SIZE * frac)
    li = np.arange(LMAX * 2 + 1)
    lk = np.arange(L_AUX_MAX + 1)
    order = li[:, None] + lk
    nf3ijkl = (order + 1) * (order + 2) * (order + 3) // 6
    if kind == 'dm':
        Rt_swap_size = np.array([42, 42, 42, 42, 42, 42, 30])
        Rt_stride = (nf3ijkl + Rt_swap_size - 1) // Rt_swap_size
        unit = order + 1 + nf3ijkl
        shm_budget = budget
        extra = 0
    else:
        nf3k = (L_AUX_MAX + 1) * (L_AUX_MAX + 2) * (L_AUX_MAX + 3) // 6
        Rt_swap_size = np.array([35, 35, 35, 35, 35, 21, 21])
        Rt_stride = (nf3ijkl + Rt_swap_size - 1) // Rt_swap_size
        nfij = (li + 1) * (li + 2) * (li + 3) // 6
        IJ_SIZE = np.array([35, 21, 15, 11, 8, 8, 8])
        Rt_stride_min = (nfij[:, None] + IJ_SIZE - 1) // IJ_SIZE
        Rt_stride = np.where(Rt_stride > Rt_stride_min, Rt_stride,
                             Rt_stride_min)
        unit = order + 1 + nf3ijkl
        shm_budget = budget - nf3k * 8
        extra = nf3k

    nsp_max = THREADS // Rt_stride
    nsp = shm_budget // (unit * 8)
    nsp = np.where(nsp < nsp_max, nsp, nsp_max)
    nsp = _nearest_power2(nsp)
    nsp[nsp > THREADS] = THREADS
    # a class that cannot fit even one shell pair in the reduced budget keeps
    # one: the kernel divides by `nsp_per_block`
    nsp[nsp < 1] = 1
    shm_size = int(((nsp * unit).max() + extra) * 8)
    return cp.asarray(nsp, dtype=np.int32), shm_size


def _auxvec_shm(lij, lk, nsp_lookup_host):
    """Exactly the shared memory one ``(lij, lk)`` block of the
    auxiliary-vector kernel lays out.

    The kernel places ``gamma_inc`` (``(order+1) * nsp``), then the
    ``nf3k`` auxiliary coefficients, then ``Rt`` (``nf3ijkl * nsp``).  Upstream
    asks for the largest of those over the **whole** ``LMAX x L_AUX_MAX``
    table, with ``nf3k`` reserved for ``L_AUX_MAX``, and asks for it on every
    block in the grid.
    """
    order = lij + lk
    nf3ijkl = (order + 1) * (order + 2) * (order + 3) // 6
    nf3k = (lk + 1) * (lk + 2) * (lk + 3) // 6
    nsp = int(nsp_lookup_host[lij, lk])
    return int((nsp * (order + 1 + nf3ijkl) + nf3k) * 8)


def _kernel_plan(opt, c):
    """Item 7's launch plan for this option object, or None to keep
    GPU4PySCF's kernels for every class."""
    if KERNELS != 'ours' or opt.mol.omega != 0:
        return None
    try:
        return _j3ck.plan(opt, c)
    except RuntimeError:
        # a basis the kernels were not written for (they assert a
        # decontracted one, which `Int3c2eOpt.build` always makes)
        return None


def contract_dm(self, dm, hermi=0):
    """``rho_P = sum_ij (ij|P) D_ij``, with the pair tables already resident."""
    if _OFF or _NO_OPT:
        STATS['passthrough'] += 1
        return _orig_contract_dm(self, dm, hermi)
    STATS['contract_dm'] += 1
    c = _j3c.constants(self)
    nao = c.nao
    assert dm.shape[-1] == nao, \
        'Requires transforming dm: mol.apply_C_mat_CT(dm)'

    dm_ndim = dm.ndim
    if dm_ndim == 2:
        dm = dm[None]
    n_dm = len(dm)
    dm = cp.asarray(dm.reshape(n_dm, nao, nao), order='C')
    if hermi != 1:
        dm = transpose_sum(dm)

    nsp_lookup, shm_size = _scheme('dm', SHM_FRAC)
    kp = _kernel_plan(self, c)
    runs = (kp.upstream_runs(SP_CHUNK_DM) if kp is not None
            else [c.pair_offsets(SP_CHUNK_DM)])
    dm_xyz = c.scratch('dm_xyz', c.dm_xyz_size)
    vj_aux = cp.zeros((n_dm, c.naux_cart))
    for i_dm in range(n_dm):
        _dm_to_Rt(c, dm[i_dm], dm_xyz)
        if kp is not None:
            kp.contract_dm(dm_xyz, vj_aux[i_dm])
        for pair_ij_offsets, sp_blocks in runs:
            err = _libmd.contract_int3c2e_dm(
                ctypes.cast(vj_aux[i_dm].data.ptr, ctypes.c_void_p),
                ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
                ctypes.c_int(1), ctypes.c_int(c.naux_cart),
                ctypes.byref(c.envs), ctypes.c_int(shm_size),
                ctypes.c_int(sp_blocks),
                ctypes.c_int(c.auxnbas),
                ctypes.cast(pair_ij_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(c.shl_pair_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(c.pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(nsp_lookup.data.ptr, ctypes.c_void_p),
                ctypes.c_double(self.mol.omega))
            if err != 0:
                raise RuntimeError('contract_int3c2e_dm kernel failed')
    if hermi == 1:
        vj_aux *= 2
    if dm_ndim == 2:
        vj_aux = vj_aux[0]
    return vj_aux


def contract_auxvec(self, auxvec):
    """``vj_ij = sum_P (ij|P) rho_P``, with the auxiliary split already made."""
    if _OFF or _NO_OPT:
        STATS['passthrough'] += 1
        return _orig_contract_auxvec(self, auxvec)
    STATS['contract_auxvec'] += 1
    c = _j3c.constants(self)
    assert auxvec.shape[-1] == c.naux_sph

    auxvec_ndim = auxvec.ndim
    auxvec = cp.asarray(auxvec.reshape(-1, c.naux_sph), order='C')
    n_dm = len(auxvec)

    nsp_lookup, shm_size = _scheme('auxvec', SHM_FRAC)
    ksh_offsets, ksh_blocks = c.ksh_batches(KSH_CHUNK)
    chunk = (c.auto_chunk(ksh_blocks, AUTO_TARGET_BLOCKS,
                          AUTO_CHUNK_MIN, AUTO_CHUNK_MAX)
             if SP_CHUNK_AUX < 0 else SP_CHUNK_AUX)
    pair_ij_offsets, sp_blocks = c.pair_offsets(chunk)
    aux_xyz = c.scratch('aux_xyz', c.aux_xyz_size)
    vj_xyz = c.scratch('vj_xyz', n_dm * c.dm_xyz_size,
                       zero=True).reshape(n_dm, c.dm_xyz_size)
    vj = cp.empty((n_dm, c.nao, c.nao))
    if PER_CLASS:
        # one launch per (shell-pair class, auxiliary class), each asking for
        # the shared memory that class actually lays out.  The grid is the
        # same blocks, partitioned.  Measured slower and off by default
        # (`docs/README_fastdfj3c.md` §8); the host copy of `nsp_lookup` is
        # taken only here, because it is a device-to-host synchronisation.
        nsp_host = cp.asnumpy(nsp_lookup)
        launches = [(g_off, g_blk, k_off, k_blk,
                     _auxvec_shm(lij, lk, nsp_host))
                    for lij, g_off, g_blk in c.pair_offsets_by_group(chunk)
                    for lk, k_off, k_blk in c.ksh_batches_by_group(KSH_CHUNK)]
    else:
        launches = [(pair_ij_offsets, sp_blocks, ksh_offsets, ksh_blocks,
                     shm_size)]
    kp = _kernel_plan(self, c)
    if kp is not None:
        launches = [(off, n, ksh_offsets, ksh_blocks, shm_size)
                    for off, n in kp.upstream_runs(chunk)]

    for i_dm in range(n_dm):
        _libmd.aux_to_Rt(
            ctypes.cast(aux_xyz.data.ptr, ctypes.c_void_p),
            ctypes.cast(auxvec[i_dm].data.ptr, ctypes.c_void_p),
            ctypes.byref(c.envs),
            ctypes.cast(c.aux_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(c.aux_xyz_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(c.auxnbas))
        if kp is not None:
            kp.contract_auxvec(aux_xyz, vj_xyz[i_dm])
        for sp_off, nsp_blk, k_off, n_k_blk, shm in launches:
            err = _libmd.contract_int3c2e_auxvec(
                ctypes.cast(vj_xyz[i_dm].data.ptr, ctypes.c_void_p),
                ctypes.cast(aux_xyz.data.ptr, ctypes.c_void_p),
                ctypes.c_int(1), ctypes.c_int(c.naux_sph),
                ctypes.byref(c.envs), ctypes.c_int(shm),
                ctypes.c_int(nsp_blk),
                ctypes.c_int(n_k_blk),
                ctypes.cast(sp_off.data.ptr, ctypes.c_void_p),
                ctypes.cast(k_off.data.ptr, ctypes.c_void_p),
                ctypes.cast(c.shl_pair_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(c.pair_loc.data.ptr, ctypes.c_void_p),
                # the kernel's `aux_loc` indexes the *Hermite* auxiliary
                # vector it was just handed, not the spherical one
                ctypes.cast(c.aux_xyz_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(nsp_lookup.data.ptr, ctypes.c_void_p),
                ctypes.c_double(self.mol.omega))
            if err != 0:
                raise RuntimeError('contract_int3c2e_auxvec kernel failed')
        _Rt_to_dm(c, vj_xyz[i_dm], vj[i_dm:i_dm + 1])

    if auxvec_ndim == 1:
        vj = vj[0]
    return vj


# --------------------------------------------------------------------------
# 2.  The recontraction kernels, with their five tables already resident.
#
# The bodies are GPU4PySCF's own (`gto/mole.py`) with the five `cp.asarray`
# calls replaced by one lookup.  Nothing else differs, which is why the test
# compares element-wise rather than to a tolerance.
# --------------------------------------------------------------------------

_orig_CT_dot_mat = SortedGTO.CT_dot_mat
_orig_C_dot_mat = SortedGTO.C_dot_mat
_orig_mat_dot_C = SortedGTO.mat_dot_C
_orig_mat_dot_CT = SortedGTO.mat_dot_CT


def _tables(self):
    STATS['ctr'] += 1
    return _j3c.sorted_constants(self)


def CT_dot_mat(self, mat, out=None):
    """``ctr_coeff.T.dot(mat)``"""
    if _OFF or _NO_CTR:
        return _orig_CT_dot_mat(self, mat, out)
    mat = cp.asarray(mat, order='C')
    mat_ndim = mat.ndim
    if mat_ndim == 1:
        return self.mat_dot_C(mat)
    elif mat_ndim == 2:
        mat = mat[None]

    kern = (libvhf_rys.bra_sorted2cart if self.mol.cart
            else libvhf_rys.bra_sorted2sph)
    nao = self.mol.nao
    counts, nao_sorted, ncol = mat.shape
    t = _tables(self)
    assert nao_sorted == self.p_ao_loc[-1]
    if mat.dtype == np.complex128:
        ncol *= 2
    out = ndarray((counts, nao, ncol), buffer=out)
    out[:] = 0.
    if out.size > 0:
        err = kern(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(mat.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.recontract_coef.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.recontract_bas.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.recontraction_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.c_ao_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.p_ao_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(t.nrecontract), ctypes.c_int(self.nbas),
            ctypes.c_int(ncol), ctypes.c_int(counts))
        assert err == 0

    if mat.dtype == np.complex128:
        out = out.view(np.complex128)
    if mat_ndim == 2:
        out = out[0]
    return out


def C_dot_mat(self, mat, out=None):
    """``ctr_coeff.dot(mat)``"""
    if _OFF or _NO_CTR:
        return _orig_C_dot_mat(self, mat, out)
    mat = cp.asarray(mat, order='C')
    mat_ndim = mat.ndim
    if mat_ndim == 1:
        return self.mat_dot_CT(mat)
    elif mat_ndim == 2:
        mat = mat[None]

    kern = (libvhf_rys.bra_cart2sorted if self.mol.cart
            else libvhf_rys.bra_sph2sorted)
    t = _tables(self)
    nao_sorted = self.p_ao_loc[-1]
    counts, nao, ncol = mat.shape
    assert nao == self.mol.nao
    if mat.dtype == np.complex128:
        ncol *= 2
    out = ndarray((counts, nao_sorted, ncol), buffer=out)
    out[:] = 0.
    if out.size > 0:
        err = kern(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(mat.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.recontract_coef.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.recontract_bas.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.recontraction_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.c_ao_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.p_ao_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(t.nrecontract), ctypes.c_int(self.nbas),
            ctypes.c_int(ncol), ctypes.c_int(counts))
        assert err == 0

    if mat.dtype == np.complex128:
        out = out.view(np.complex128)
    if mat_ndim == 2:
        out = out[0]
    return out


def mat_dot_C(self, mat, out=None):
    """``mat.dot(ctr_coeff)``"""
    if _OFF or _NO_CTR:
        return _orig_mat_dot_C(self, mat, out)
    mat_ndim = mat.ndim
    mat_dtype = mat.dtype
    if mat_ndim == 1:
        mat = mat[None, None]
    elif mat_ndim == 2:
        mat = mat[None]

    kern = (libvhf_rys.ket_sorted2cart if self.mol.cart
            else libvhf_rys.ket_sorted2sph)
    t = _tables(self)
    nao = self.mol.nao
    counts, nrow, nao_sorted = mat.shape
    assert nao_sorted == self.p_ao_loc[-1]
    if mat_dtype == np.complex128:
        mat = cp.asarray(mat.view(np.float64).transpose(0, 1, 3, 2), order='C')
    else:
        mat = cp.asarray(mat, dtype=np.float64, order='C')
    out = ndarray((counts, nrow, nao), buffer=out)
    out[:] = 0.
    if out.size > 0:
        err = kern(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(mat.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.recontract_coef.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.recontract_bas.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.recontraction_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.c_ao_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.p_ao_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(t.nrecontract), ctypes.c_int(self.nbas),
            ctypes.c_int(nrow * counts))
        assert err == 0

    if mat_dtype == np.complex128:
        mat = None
        out, tmp = cp.empty((counts, nrow, nao), dtype=np.complex128), out
        out.real = tmp[:, :nrow]
        out.imag = tmp[:, nrow:]
    if mat_ndim == 1:
        out = out[0, 0]
    elif mat_ndim == 2:
        out = out[0]
    return out


def mat_dot_CT(self, mat, out=None):
    """``mat.dot(ctr_coeff.T)``"""
    if _OFF or _NO_CTR:
        return _orig_mat_dot_CT(self, mat, out)
    mat_ndim = mat.ndim
    mat_dtype = mat.dtype
    if mat_ndim == 1:
        mat = mat[None, None]
    elif mat_ndim == 2:
        mat = mat[None]

    kern = (libvhf_rys.ket_cart2sorted if self.mol.cart
            else libvhf_rys.ket_sph2sorted)
    t = _tables(self)
    nao_sorted = self.p_ao_loc[-1]
    counts, nrow, nao = mat.shape
    assert nao == self.mol.nao
    if mat_dtype == np.complex128:
        mat = cp.asarray(mat.view(np.float64).transpose(0, 1, 3, 2), order='C')
    else:
        mat = cp.asarray(mat, dtype=np.float64, order='C')
    out = ndarray((counts, nrow, nao_sorted), buffer=out)
    out[:] = 0.
    if out.size > 0:
        err = kern(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(mat.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.recontract_coef.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.recontract_bas.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.recontraction_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.c_ao_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(t.p_ao_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(t.nrecontract), ctypes.c_int(self.nbas),
            ctypes.c_int(nrow * counts))
        assert err == 0

    if mat_dtype == np.complex128:
        mat = None
        out, tmp = cp.empty((counts, nrow, nao_sorted),
                            dtype=np.complex128), out
        out.real = tmp[:, :nrow]
        out.imag = tmp[:, nrow:]
    if mat_ndim == 1:
        out = out[0, 0]
    elif mat_ndim == 2:
        out = out[0]
    return out


# --------------------------------------------------------------------------
# 6.  The engine choice.
#
# `df_jk.get_j`'s body is GPU4PySCF's own apart from two lines: the branch
# that picks the engine, and the guard that hands a call back when the MD
# engine cannot serve it.
# --------------------------------------------------------------------------

_orig_get_j = DFJK.get_j


def get_j(dfobj, dm, hermi=1):
    """``df/df_jk.py:get_j`` with the McMurchie--Davidson engine for every
    density count, not only for one."""
    STATS['get_j'] += 1
    if _OFF or _NO_MDM:
        STATS['get_j_rys'] += 1
        return _orig_get_j(dfobj, dm, hermi)
    if dfobj.intopt is None:
        dfobj.build(build_cderi=False)
    if dfobj._cd_j2c is None:
        j2c = int2c2e(dfobj.auxmol)
        try:
            dfobj._cd_j2c = DFJK.cholesky(j2c), 'cd'
        except RuntimeError:
            dfobj._cd_j2c = j2c, None

    n_dm = 1 if dm.ndim == 2 else int(np.prod(dm.shape[:-2]))
    if n_dm > MDM_MAX_NDM:
        STATS['get_j_rys'] += 1
        return _orig_get_j(dfobj, dm, hermi)

    if dfobj.j_engine is None:
        try:
            dfobj.j_engine = J3C.Int3c2eOpt(dfobj.mol, dfobj.auxmol).build()
        except AssertionError:
            # `Int3c2eOpt.build` asserts its angular-momentum ceiling; a basis
            # past it is GPU4PySCF's to serve, and it serves it on the Rys
            # engine for one density as well as for many
            dfobj.j_engine = False
    if dfobj.j_engine is False:
        STATS['get_j_rys'] += 1
        return _orig_get_j(dfobj, dm, hermi)
    STATS['get_j_md'] += 1

    dm_shape = dm.shape
    intopt = dfobj.j_engine
    if INCREMENTAL and dm.ndim == 2 and hermi == 1 and not _OFF:
        vj = _get_j_incremental(dfobj, intopt, dm)
        if vj is not None:
            return vj.reshape(dm_shape)
    mol = intopt.mol
    auxmol = intopt.auxmol
    dm = mol.apply_C_mat_CT(dm)
    rhoj = intopt.contract_dm(dm, hermi)
    rhoj = auxmol.apply_CT_dot(rhoj, axis=-1)

    j2c, tag = dfobj._cd_j2c
    if tag == 'cd':
        # looked up on the module rather than imported, so a probe that wraps
        # `df_jk.solve_triangular` still sees these two
        rhoj = DFJK.solve_triangular(j2c, rhoj.T, lower=True)
        rhoj = DFJK.solve_triangular(j2c.T, rhoj, lower=False).T
    else:
        rhoj = cp.linalg.solve(j2c, rhoj.T).T

    rhoj = auxmol.apply_C_dot(rhoj, axis=-1)
    vj = intopt.contract_auxvec(rhoj)
    vj = mol.apply_CT_mat_C(vj)
    return vj.reshape(dm_shape)


def _solve_metric(dfobj, rhoj):
    j2c, tag = dfobj._cd_j2c
    if tag == 'cd':
        rhoj = DFJK.solve_triangular(j2c, rhoj.T, lower=True)
        return DFJK.solve_triangular(j2c.T, rhoj, lower=False).T
    return cp.linalg.solve(j2c, rhoj.T).T


# --------------------------------------------------------------------------
# 8.  The Coulomb matrix from the difference, screened on it (W20).
# --------------------------------------------------------------------------
_INC_ATTR = '_electrolites_jinc'


def _get_j_incremental(dfobj, intopt, dm):
    """``J(D) = J(D_last) + J(D - D_last)`` with the pairs the difference
    cannot reach dropped from both contractions.

    The state is the previous call's sorted-basis density, its fitted
    auxiliary density before the metric (``rho``), its coefficients after it
    (``cs``) and its sorted-basis Coulomb matrix, all on the ``DF`` object.
    ``rho`` and the Coulomb matrix are accumulated; the metric solve, which
    is a small share of the call and couples every auxiliary function to
    every other, is done on the whole ``rho`` every time.  Returns None when
    the kernels are not this repository's, so that the caller builds whole.
    """
    c = _j3c.constants(intopt)
    kp = _kernel_plan(intopt, c)
    if kp is None:
        return None
    mol, auxmol = intopt.mol, intopt.auxmol
    dms = mol.apply_C_mat_CT(dm)
    st = getattr(dfobj, _INC_ATTR, None)
    full = (st is None or st['envs'] is not intopt.int3c2e_envs
            or st['n'] >= INC_REBUILD or st['dms'].shape != dms.shape)
    if not full:
        ddm = dms - st['dms']
        full = (float(abs(ddm).max())
                > INC_MAXFRAC * float(abs(dms).max()))
    if full:
        STATS['inc_full'] += 1
        rho = intopt.contract_dm(dms, 1)
    else:
        STATS['inc_step'] += 1
        scr = _j3cinc.screen(intopt, c)
        keep = scr.keep_dm(ddm, INC_TOL)
        rho = st['rho'] + _contract_dm_subset(c, kp, ddm, keep)
    cs = auxmol.apply_C_dot(
        _solve_metric(dfobj, auxmol.apply_CT_dot(rho, axis=-1)), axis=-1)
    if full:
        vjs = intopt.contract_auxvec(cs)
    else:
        dcs = cs - st['cs']
        keep2 = scr.keep_aux(dcs, INC_TOL)
        vjs = st['vjs'] + _contract_auxvec_subset(c, kp, dcs, keep2)
        STATS['inc_pairs_dm'] += int(keep.sum())
        STATS['inc_pairs_aux'] += int(keep2.sum())
        STATS['inc_pairs_all'] += c.npair
    setattr(dfobj, _INC_ATTR, {'envs': intopt.int3c2e_envs, 'dms': dms,
                               'rho': rho, 'cs': cs, 'vjs': vjs,
                               'n': 0 if full else st['n'] + 1})
    return mol.apply_CT_mat_C(vjs)


def _contract_dm_subset(c, kp, ddm, keep):
    """``contract_dm(ddm, hermi=1)`` over the pairs ``keep`` selects."""
    dm_xyz = c.scratch('dm_xyz', c.dm_xyz_size)
    _dm_to_Rt(c, ddm, dm_xyz)
    sub, spi = kp.subset(keep, c.shl_pair_idx)
    out = cp.zeros(c.naux_cart)
    sub.contract_dm(dm_xyz, out)
    nsp_lookup, shm_size = _scheme('dm', SHM_FRAC)
    for off, nblk in sub.upstream_runs(0):
        err = _libmd.contract_int3c2e_dm(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
            ctypes.c_int(1), ctypes.c_int(c.naux_cart),
            ctypes.byref(c.envs), ctypes.c_int(shm_size),
            ctypes.c_int(nblk), ctypes.c_int(c.auxnbas),
            ctypes.cast(off.data.ptr, ctypes.c_void_p),
            ctypes.cast(spi.data.ptr, ctypes.c_void_p),
            ctypes.cast(sub.pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(nsp_lookup.data.ptr, ctypes.c_void_p),
            ctypes.c_double(0.))
        if err != 0:
            raise RuntimeError('contract_int3c2e_dm kernel failed')
    out *= 2
    return out


def _contract_auxvec_subset(c, kp, dcs, keep):
    """``contract_auxvec(dcs)`` over the pairs ``keep`` selects."""
    aux_xyz = c.scratch('aux_xyz', c.aux_xyz_size)
    _libmd.aux_to_Rt(
        ctypes.cast(aux_xyz.data.ptr, ctypes.c_void_p),
        ctypes.cast(dcs.data.ptr, ctypes.c_void_p),
        ctypes.byref(c.envs),
        ctypes.cast(c.aux_loc.data.ptr, ctypes.c_void_p),
        ctypes.cast(c.aux_xyz_loc.data.ptr, ctypes.c_void_p),
        ctypes.c_int(c.auxnbas))
    vj_xyz = c.scratch('vj_xyz', c.dm_xyz_size, zero=True)
    sub, spi = kp.subset(keep, c.shl_pair_idx)
    sub.contract_auxvec(aux_xyz, vj_xyz)
    runs = sub.upstream_runs(1024)
    if runs:
        nsp_lookup, shm_size = _scheme('auxvec', SHM_FRAC)
        ksh_offsets, ksh_blocks = c.ksh_batches(KSH_CHUNK)
        for off, nblk in runs:
            err = _libmd.contract_int3c2e_auxvec(
                ctypes.cast(vj_xyz.data.ptr, ctypes.c_void_p),
                ctypes.cast(aux_xyz.data.ptr, ctypes.c_void_p),
                ctypes.c_int(1), ctypes.c_int(c.naux_sph),
                ctypes.byref(c.envs), ctypes.c_int(shm_size),
                ctypes.c_int(nblk), ctypes.c_int(ksh_blocks),
                ctypes.cast(off.data.ptr, ctypes.c_void_p),
                ctypes.cast(ksh_offsets.data.ptr, ctypes.c_void_p),
                ctypes.cast(spi.data.ptr, ctypes.c_void_p),
                ctypes.cast(sub.pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(c.aux_xyz_loc.data.ptr, ctypes.c_void_p),
                ctypes.cast(nsp_lookup.data.ptr, ctypes.c_void_p),
                ctypes.c_double(0.))
            if err != 0:
                raise RuntimeError('contract_int3c2e_auxvec kernel failed')
    out = cp.empty((1, c.nao, c.nao))
    _Rt_to_dm(c, vj_xyz, out)
    return out[0]


DFJK.get_j = get_j
DFJK._get_j_orig = _orig_get_j

J3C.Int3c2eOpt.contract_dm = contract_dm
J3C.Int3c2eOpt.contract_auxvec = contract_auxvec
J3C.Int3c2eOpt._contract_dm_orig = _orig_contract_dm
J3C.Int3c2eOpt._contract_auxvec_orig = _orig_contract_auxvec

SortedGTO.CT_dot_mat = CT_dot_mat
SortedGTO.C_dot_mat = C_dot_mat
SortedGTO.mat_dot_C = mat_dot_C
SortedGTO.mat_dot_CT = mat_dot_CT
# `SortedMole` and `SortedCell` are views of `SortedGTO`, so both inherit the
# four above; the originals are stashed where `tests/test_patch_aliases.py`
# can walk for a stale reference to them.
GMOL._CT_dot_mat_orig = _orig_CT_dot_mat
GMOL._C_dot_mat_orig = _orig_C_dot_mat
GMOL._mat_dot_C_orig = _orig_mat_dot_C
GMOL._mat_dot_CT_orig = _orig_mat_dot_CT

__all__ = ['stats', 'get_j', 'contract_dm', 'contract_auxvec']
