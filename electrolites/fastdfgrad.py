"""The density-fitted gradient's two-electron derivative, without the
three-centre integrals it does not have to evaluate.

``docs/DF_WORK_LIST.md`` **W3**.  ``df/grad/rhf.py:_jk_energy_per_atom`` is the
one uncovered stage of the density-fitting path that *grows* with basis size:
29 % of a def2-SVP DF gradient and **83 %** of a def2-TZVPP one, and nothing
in this repository was on it.  W3's first instruction is to profile it stage
by stage before assuming anything, with §2's lesson attached -- *do not assume
the contraction is the cost* -- and that profile
(``benchmarks/df_grad_probe.py``) is what this module is built on.  It says
something the work list did not expect:

    ip1 kernel (3c2e derivatives)    52-59 %
    eval_j3c (3c2e integrals)        24-26 %
    everything else                  15-24 %

The **contraction is not the cost here either**: the two half-transforms, the
two fancy-index scatters W3 singles out and the ``cp.take`` together are
4.4 % of the function at 42 atoms and 14.5 % at 113, and they already run at
9-16 TF/s of an A100's 19.5 and at 640-1820 GB/s.  ``docs/README_fastdfgrad.md``
§2 has the three tables.  So W3's own suggestion -- that the ``(L,i,j)``
layout would turn both contraction pairs into clean batched GEMMs -- is
answered by measuring it rather than by trying it: a *perfect* version of
every contraction in the function is worth under 5 % at 42 atoms.

What is worth having is the rest, and it comes in two pieces.

**One.  A quarter of the function is integrals the SCF already computed.**
``eval_j3c`` evaluates the three-centre integrals ``(pq|P)`` -- and the SCF
that just finished has them, in the **cderi** ``df/df.py:_cholesky_eri``
built and that ``jk_energy_per_atom`` throws away with a ``with_df.reset()``
on its first line.  The identity that makes the reuse exact is one line of
algebra.  ``_decompose_j2c`` returns the matrix ``aux_coef`` the cderi is
defined by,

    cderi[L,pq] = sum_v (pq|v) aux_coef[v,L]

and it satisfies ``aux_coef @ aux_coef.T == metric``, the very metric the
gradient rebuilds for itself -- a Cholesky solve in the ``'cd'`` branch, a
filtered pseudo-inverse in the ``'ed'`` one, and the identity holds in both.
So with

    rhok[L,i,j] = sum_pq cderi[L,pq] C[p,i] C[q,j]

the pseudo-density the ip1 kernel wants is ``dm_oo = aux_coef @ rhok``,
exactly.  The half-transform is the same arithmetic the function was doing
anyway -- and on a *smaller* basis, because the cderi is contracted and
spherical where the gradient's own pair list is decontracted and Cartesian
(1007 functions against 1146 at def2-TZVPP) -- so what the reuse removes is
the integral evaluation and one of the two metric solves, and nothing is
added.  Worth **1.22x-1.46x** on the function across six systems.

**Two.  The ip1 kernel is launched four shell pairs deep where one is
faster.**  ``df/grad/rhf.py:141`` aggregates shell pairs at
``nsp_per_block*4``, so each thread block loops four times instead of once
and the device runs a quarter of the blocks it could.  Swept over
``{1,2,4,8,16,32}`` and four ``gout_width``s
(``benchmarks/df_grad_ip1_sweep.py``), ``1`` is the fastest of the six at
every size and basis measured, and by a margin that *grows*: **1.257x** on
the kernel at 42 atoms/def2-SVP, **1.769x** at def2-TZVPP and **3.153x** at
113 atoms.  Nothing about the arithmetic changes -- the kernel's answer is
identical across the whole grid, which is what says this is a launch
configuration and not an approximation.  The same constant is in
``int3c2e_evaluator`` and in ``_j_energy_per_atom``, so the finding is wider
than this module; ``docs/DF_WORK_LIST.md`` W4 is where that is recorded.

Together, on one ``jk_energy_per_atom`` call at fixed density: **1.75x** at
42 atoms/def2-SVP, **2.15x** at def2-TZVPP and **2.59x** on 113-atom taxol,
and **1.45x / 2.01x / 2.27x** on the whole gradient those sit in.  On two
open-shell transition-metal complexes, 1.43x at 21 atoms and 1.92x at 60.
``docs/README_fastdfgrad.md`` §4 has the ablation table.

Two smaller things, one of which is an item and one of which is a rejection:

* **The ``Int3c2eOpt`` is the SCF's**, for the full-range call.  ``DF.build``
  constructs exactly the object ``jk_energy_per_atom`` constructs --
  ``Int3c2eOpt(mol, auxmol)`` with ``mol`` replaced by the same decontracted
  ``SortedMole``, the same default cutoff -- and upstream's ``reset()``
  throws it away one line before rebuilding it.  Worth **1.14x** of the
  1.75x at 42 atoms/def2-SVP and nothing to speak of past it, because it is
  a fixed setup cost against a term that grows.
* **Building the Coulomb term of the pseudo-density on the pair list does
  not pay, and is off.**  Upstream writes ``dm[p,q]*auxvec[r]`` into the
  dense ``(nao, nao, blk)`` buffer and makes the exchange contraction
  accumulate onto it with ``beta = j_factor``; only ``nao_pair`` of those
  ``nao*nao`` entries survive the ``cp.take``, so the same term is one
  ``axpy`` on the compressed list and strictly less work.  Measured, it is
  **0-1.4 % slower** on all five systems it was tried on, and never faster.
  ``FASTDFGRAD_JPAIR=1`` still runs it, because a rejected alternative that
  can be re-run is worth more than one described in prose.
* **The auxiliary-basis response term** reuses ``dm_oo`` as upstream does;
  nothing changes there.

Coverage.  The reuse needs the tensor it reuses, so this module answers a call
only when the SCF's cderi is still live and the call is the ground-state one:
``hermi == 1``, a symmetric density factorisation (a ``mo_coeff``-tagged
density -- every SCF gradient), ``k_factor != 0`` (a pure functional's
gradient never reaches this function: ``df/grad/rhf.py:68`` sends it to
``_j_energy_per_atom``, which builds no cderi), and one device.  A
range-separated functional's *second* call asks for the long-range operator,
and the SCF built a long-range cderi for it under ``with_df._rsh_df``; that
one is used when it is there, which is what ``_release`` is for -- upstream's
``reset()`` would have deleted it before the second call could reach it.
Everything else -- a TDDFT or NAC gradient's asymmetric factorisation,
``hermi != 1``, an ``only_dfj`` object (which has no cderi at all,
``docs/README_fastdfj.md`` §6), a second device -- is handed back to
GPU4PySCF's own function, so a patched run computes the same gradient as an
unpatched one.  ``stats()`` says which, because nothing numerical can.

One thing the reuse changes, and it is worth stating plainly.  GPU4PySCF's
gradient decomposes the fitting metric *again*, and not always the way the
cderi was decomposed: ``_jk_energy_per_atom`` picks the Cholesky branch when
``mol.omega <= 0 and not auxmol.mol.cart`` and a filtered eigen-decomposition
otherwise, where ``_decompose_j2c`` takes the Cholesky whenever it succeeds.
Where the two choices differ, the gradient upstream returns is not quite the
derivative of the energy it was computed from; this module's is.  Measured,
they agree to **1.5e-9 relative** at 42 atoms and **7.0e-8** at 113, and §5
of the write-up shows that merely *associating* the metric contraction the other way
round on upstream's own ``j3c_oo`` moves the answer by as much -- the fitting
metric is conditioned at 1e8 in the full-range case and rank deficient in the
long-range one, and that, not the reuse, is what separates the two answers.

Ablations, for the tables in ``docs/README_fastdfgrad.md``:

    FASTDFGRAD_OFF=1        hand every call back to GPU4PySCF
    FASTDFGRAD_CDERI=0      keep the second half's items but rebuild the
                            three-centre integrals as upstream does, so the
                            reuse can be priced on its own
    FASTDFGRAD_JPAIR=1      build the Coulomb term of the pseudo-density on
                            the compressed pair list instead of in the dense
                            buffer; measured, and slower -- section 4.3
    FASTDFGRAD_INTOPT=0     rebuild the `Int3c2eOpt` rather than reusing the
                            one the SCF's `DF.build` made
    FASTDFGRAD_SPMULT=<n>   shell pairs per thread block of the ip1 kernel, as
                            a multiple of the scheme's `nsp_per_block`; 4 is
                            GPU4PySCF's, 1 is the default here
    FASTDFGRAD_GOUT=<n>     the ip1 scheme's `gout_width`; 54 is GPU4PySCF's
                            and is kept, and above 54 the kernel is wrong
"""
import ctypes
import os

import numpy as np
import cupy as cp
from pyscf import lib

from gpu4pyscf.df import df as DFMOD
from gpu4pyscf.df import df_jk as DFJK
from gpu4pyscf.df.grad import rhf as df_rhf_grad
from gpu4pyscf.df.grad import rks as df_rks_grad
from gpu4pyscf.df.grad import uhf as df_uhf_grad
from gpu4pyscf.df.grad import uks as df_uks_grad
from gpu4pyscf.df.int3c2e_bdiv import (
    _split_l_ctr_pattern, argsort_aux, get_ao_pair_loc, int3c2e_scheme,
    libvhf_rys, Int3c2eOpt)
from gpu4pyscf.gto.mole import SortedMole
from gpu4pyscf.lib import logger, multi_gpu
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem, ndarray

_OFF = bool(os.environ.get('FASTDFGRAD_OFF'))
#: reuse the SCF's cderi for the occupied-occupied fitting coefficients rather
#: than evaluating the three-centre integrals a second time.  ``0`` restores
#: GPU4PySCF's first loop and is how the item is priced on its own.
_CDERI = os.environ.get('FASTDFGRAD_CDERI', '1') != '0'
#: build the Coulomb term of the pseudo-density on the compressed pair list
#: rather than in the dense buffer.  **Off by default, as a measurement**: it
#: does strictly less work and is 0.4-1.4 % *slower* on all three systems it
#: was tried on -- `docs/README_fastdfgrad.md` section 4.3.  Kept because a
#: rejected alternative that can still be run is worth more than one
#: described in prose.
_JPAIR = os.environ.get('FASTDFGRAD_JPAIR', '0') != '0'
#: how many shell pairs one thread block of the ip1 kernel is given, as a
#: multiple of the scheme's `nsp_per_block`.  GPU4PySCF fixes it at 4
#: (`df/grad/rhf.py:141`); 1 is a measurement, and the only one of the six
#: swept that is not a loss -- `benchmarks/df_grad_ip1_sweep.py`, and
#: `docs/README_fastdfgrad.md` section 4.2 for the grid.
_SPMULT = int(os.environ.get('FASTDFGRAD_SPMULT', 1))
#: threads-per-shell-pair knob of the same scheme, through
#: `gout_stride = nearest_power2(ceil(nfi*nfj*nfk / gout_width))`.  Left at
#: GPU4PySCF's 54 **on purpose**: 27 is 1.48x better again on an f-shell
#: basis (def2-TZVPP) and 3.7 % *worse* on a d-shell one, so the better value
#: depends on the basis, and two systems is not enough to write that rule
#: down.  Anything above 54 changes the answer -- the kernel's gout buffer
#: overflows -- so the knob has a ceiling, not just a preference; see
#: `docs/README_fastdfgrad.md` section 4.2.
_GOUT = int(os.environ.get('FASTDFGRAD_GOUT', 54))
#: reuse the `Int3c2eOpt` the SCF's `DF.build` already constructed, for the
#: full-range call.  It is the same object with the same cutoff; upstream
#: rebuilds it because `reset()` has just thrown it away.
_INTOPT = os.environ.get('FASTDFGRAD_INTOPT', '1') != '0'

#: counters, so a coverage claim can be read off a run rather than assumed.
STATS = {'calls': 0, 'declined': 0, 'cderi_reuse': 0, 'intopt_reuse': 0,
         'rebuilt': 0, 'blocks': 0, 'naux': 0}
DECLINED = {}


def stats():
    """What this module did in this process.  Nothing numerical can say."""
    out = dict(STATS)
    out['declined_reasons'] = dict(DECLINED)
    return out


def _decline(reason):
    STATS['declined'] += 1
    DECLINED[reason] = DECLINED.get(reason, 0) + 1
    return None


# --------------------------------------------------------------------------
# the cderi the SCF already built
# --------------------------------------------------------------------------
def _pick_df(mf, omega):
    """The DF object whose cderi carries the operator this call asks for.

    ``omega == 0`` is the full-range build the SCF itself ran.  A
    range-separated functional's second call wants the long-range operator,
    and ``DF.range_coulomb`` keeps a separate object per omega under
    ``_rsh_df`` with a cderi of its own -- the SCF built it, because the same
    split is in ``get_veff``.  Returning ``None`` is a decline, not an error.
    """
    with_df = getattr(mf, 'with_df', None)
    if with_df is None:
        return None
    if not omega:
        return with_df
    rsh = getattr(with_df, '_rsh_df', None) or {}
    return rsh.get('%.6f' % omega)


def _release(with_df, used):
    """``DF.reset()``'s memory release, minus the tensor the next call needs.

    Upstream's first line is ``mf.with_df.reset()``, and ``reset`` empties
    ``_rsh_df`` along with everything else.  That is invisible for a plain
    hybrid and expensive for a **range-separated** one, whose gradient makes
    two calls here: the second asks for the long-range operator, and the
    long-range cderi the SCF built for it would have been thrown away by the
    first call before it could be reused.  So what is released is what has
    just been consumed, and a long-range object that still carries a tensor is
    left standing until its own call consumes it.  Everything the SCF was
    already holding through both builds stays held for no longer than the SCF
    held it.
    """
    rsh = dict(getattr(with_df, '_rsh_df', None) or {})
    for obj in (with_df, *rsh.values()):
        if obj is used or getattr(obj, '_cderi', None) is None:
            obj._cderi = None
            obj._cderi_idx = None
            obj._cd_j2c = None
            obj.intopt = None
            obj.j_engine = None
    with_df._rsh_df = {k: v for k, v in rsh.items() if v._cderi is not None}


def _rhok_from_cderi(dfobj, dm, hermi):
    """``rhok[L,i,j] = sum_pq cderi[L,pq] C[p,i] C[q,j]``, or ``None``.

    The two contractions are GPU4PySCF's own, in GPU4PySCF's own layout: the
    block ``DF.loop`` yields is an ``(i,j,L)`` buffer presented as ``(L,i,j)``,
    and taking the ``(p,q,L)`` view back makes this the same pair of calls the
    function's first loop makes on its own dense block.  That is deliberate --
    ``docs/README_fastdf.md`` §3.4 measured three rearrangements of exactly
    this contraction and kept none of them.

    Returns ``(rhok, nocc)`` with ``rhok`` of shape ``(naux, nocc, nocc)``, or
    ``(2, naux, nocc, nocc)`` for an unrestricted density.
    """
    if dfobj is None or getattr(dfobj, '_cderi', None) is None:
        return _decline('no cderi (pure functional, only_dfj, or reset)')
    if multi_gpu.num_devices != 1:
        return _decline('more than one device')
    if hermi != 1:
        return _decline(f'hermi={hermi}')
    if getattr(dm, 'symmetrize', 0):
        return _decline('symmetrized density factorisation (TDDFT/NAC)')

    factor_l, factor_r = DFJK.factorize_dm(dm, hermi)
    if factor_r is not None:
        # the asymmetric factorisations only the TDDFT gradient and the NAC
        # drivers produce; the identity below is stated for C on both sides
        return _decline('asymmetric density factorisation')
    factor_l = cp.asarray(factor_l)
    if factor_l.ndim == 2:
        factor_l = factor_l[None]
    elif factor_l.ndim != 3:
        return _decline(f'density factor of rank {factor_l.ndim}')
    nset, nao, nocc = factor_l.shape
    if nao != dfobj.nao:
        return _decline('density factor is not in the cderi AO basis')

    naux = dfobj._cderi[cp.cuda.Device().id].shape[0]
    rhok = cp.empty((nset, naux, nocc, nocc))
    blksize = dfobj.get_blksize(extra=nocc * nao)
    p1 = 0
    for cderi_blk, _ in dfobj.loop(blksize=blksize, unpack=True):
        nL = cderi_blk.shape[0]
        p0, p1 = p1, p1 + nL
        # `loop` hands back a (L,i,j) view of an (i,j,L) buffer; the view back
        # is what makes both contractions the ones GPU4PySCF already issues
        work = cderi_blk.transpose(1, 2, 0)
        for s in range(nset):
            tmp = contract('pqr,pi->iqr', work, factor_l[s])
            contract('iqr,qj->rij', tmp, factor_l[s], out=rhok[s, p0:p1])
        STATS['blocks'] += 1
    assert p1 == naux
    STATS['naux'] = naux
    return rhok if nset > 1 else rhok[0]


def _dm_oo_from_rhok(auxmol, rhok, aux_sorting):
    """``dm_oo = aux_coef @ rhok`` -- the fitting coefficients, sorted.

    ``aux_coef`` is what ``_decompose_j2c`` hands ``_cholesky_eri``, and the
    identity ``aux_coef @ aux_coef.T == metric`` is what makes this the same
    ``dm_oo`` the function's own first half produces.  Recomputing it costs
    one Cholesky and one triangular solve, where the metric the function
    builds instead costs the same Cholesky, *two* solves and a dense
    ``(naux x naux)`` GEMM on top.

    ``aux_sorting`` has to be the caller's own, not one derived from
    ``auxmol`` alone: the permutation depends on how ``_split_l_ctr_pattern``
    cut the auxiliary groups for *this* call's batch size, and the
    ``dm_aux`` reordering at the end of the function uses the same one.  Only
    ``rhok`` is sorting-independent -- it is a sum over the whole auxiliary
    index -- which is what lets a tensor the SCF built under one split feed a
    gradient running under another.
    """
    aux_coef = DFMOD._decompose_j2c(auxmol, aux_sorting)[0]
    if rhok.ndim == 3:
        naux, nocc = rhok.shape[0], rhok.shape[-1]
        return aux_coef.dot(rhok.reshape(naux, -1)).reshape(-1, nocc, nocc)
    nset, naux, nocc = rhok.shape[0], rhok.shape[1], rhok.shape[-1]
    dm_oo = cp.empty((nset, aux_coef.shape[0], nocc, nocc))
    for s in range(nset):
        dm_oo[s] = aux_coef.dot(rhok[s].reshape(naux, -1)).reshape(
            -1, nocc, nocc)
    return dm_oo


# --------------------------------------------------------------------------
# the half of the function that has to run either way
# --------------------------------------------------------------------------
def _ejk_from_rhok(int3c2e_opt, dm_factor, rhok,
                   j_factor, k_factor, auxbasis_response, log):
    """The pseudo-density, the ip1 kernel and the auxiliary response term.

    This is ``df/grad/rhf.py:132-234`` (and its ``df/grad/uhf.py`` twin, whose
    only difference is a spin index and a factor of two) with one change that
    is on and one that is off.  **On**: the ip1 kernel is launched with
    ``nsp_per_block * _SPMULT`` shell pairs per thread block rather than the
    fixed ``* 4``.  **Off** (``FASTDFGRAD_JPAIR=1``): the Coulomb term of the
    pseudo-density added on the compressed pair list rather than written into
    the dense ``(nao, nao, blk)`` buffer that ``cp.take`` immediately discards
    most of -- strictly less work, and measured slower.  The integrals, the
    batching, the screening and every symmetry factor are GPU4PySCF's.
    """
    from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1_per_atom
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    t0 = log.init_timer()

    unrestricted = rhok.ndim == 4
    nset, nao, nocc = dm_factor.shape
    naux = auxmol.nao
    # the restricted build carries -0.5*k_factor and the unrestricted one
    # -k_factor, because its two spin channels each hold half the density
    # (`df/grad/rhf.py:179` against `df/grad/uhf.py:152`)
    k_half = -k_factor if unrestricted else -.5 * k_factor

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    nao_pair = len(pair_addresses)

    mem_free = get_avail_mem(exclude_memory_pool=True)
    mem_avail = mem_free - nset * naux * nocc**2 * 8 - nao**2 * 8
    batch_size = max(1, min(naux, int(mem_avail * .5 / (nao_pair * 8))))
    blksize = max(1, min(naux, int(mem_avail * .4 / (nao * nao * 2 * 8)) // 8 * 8))
    log.debug1('%.3f GB free. nao_pair=%d naux=%d batch_size=%d blksize=%d',
               mem_free * 1e-9, nao_pair, naux, batch_size, blksize)

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
        short_range=mol.omega < 0, gout_width=_GOUT, deriv=(1, 0, 0))
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    lmax = mol.uniq_l_ctr[:, 0].max()
    laux = auxmol.uniq_l_ctr[:, 0].max()
    shm_size_max = shm_size[:laux + 1, :lmax + 1, :lmax + 1].max()

    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0] * _SPMULT)
    ao_pair_loc = get_ao_pair_loc(mol.uniq_l_ctr[:, 0], int3c2e_opt.bas_ij_cache)
    aux_loc = auxmol.ao_loc

    l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, auxmol.uniq_l_ctr, batch_size)
    # the permutation the auxiliary index is in for *this* batch size, which
    # is the one `dm_oo` has to be built in and the one `dm_aux` is undone
    # with at the end.  Upstream reads it off its own `int3c2e_evaluator`;
    # this reads it off the same two arrays, which is what the assertion
    # commented out at `df/grad/rhf.py:148` says is the same thing.
    aux_sorting = argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux)
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu + mol.nbas, dtype=np.int32)
    aux_offsets = aux_loc[ksh_offsets_cpu]
    aux_batches = len(aux_offsets) - 1

    dm_oo = _dm_oo_from_rhok(auxmol, rhok, aux_sorting)
    rhok = None
    t0 = log.timer_debug1('fastdfgrad: dm_oo from the cderi', *t0)

    if j_factor != 0:
        auxvec = dm_oo.trace(axis1=-2, axis2=-1)
        if unrestricted:
            auxvec = auxvec.sum(axis=0)
        dm = cp.zeros((nao, nao))
        for s in range(nset):
            dm += dm_factor[s].dot(dm_factor[s].T)
        dm_pair = cp.take(dm.reshape(-1), pair_addresses) if _JPAIR else None
    else:
        auxvec = dm_pair = None

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libvhf_rys.sum_ejk_int3c2e_ip1
    aux0 = aux1 = 0
    buf = cp.empty((nao_pair * batch_size))
    buf1 = cp.empty((blksize, nao, nao))
    buf2 = cp.empty((nset, blksize, nocc, nao))
    ejk = cp.zeros((mol.natm, 3))
    ejk_aux = cp.zeros((mol.natm, 3))
    for kbatch in range(aux_batches):
        naux_in_batch = aux_offsets[kbatch + 1] - aux_offsets[kbatch]
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        compressed = ndarray((nao_pair, naux_in_batch), buffer=buf)
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            dm_tensor = ndarray((nao, nao, dk), buffer=buf1)
            tmp = ndarray((nset, nocc, nao, dk), buffer=buf2)
            beta = 0.
            if j_factor != 0 and not _JPAIR:
                cp.multiply(dm[:, :, None], auxvec[aux0:aux1], out=dm_tensor)
                beta = j_factor
            if unrestricted:
                contract('nrji,nqj->niqr', dm_oo[:, aux0:aux1], dm_factor,
                         out=tmp)
                contract('niqr,npi->pqr', tmp, dm_factor, k_half, beta,
                         out=dm_tensor)
            else:
                contract('rji,qj->iqr', dm_oo[aux0:aux1], dm_factor[0],
                         out=tmp[0])
                contract('iqr,pi->pqr', tmp[0], dm_factor[0], k_half, beta,
                         out=dm_tensor)
            cp.take(dm_tensor.reshape(-1, dk), pair_addresses, axis=0,
                    out=compressed[:, k0:k1])
            if j_factor != 0 and _JPAIR:
                # the same term upstream writes into `dm_tensor`, on the only
                # `nao_pair` of its `nao*nao` entries that survive `cp.take`
                compressed[:, k0:k1] += (j_factor * dm_pair[:, None]
                                         * auxvec[aux0:aux1])
        err = kern(
            ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
            ctypes.cast(ejk_aux.data.ptr, ctypes.c_void_p),
            ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(),
            ctypes.c_int(1),
            ctypes.byref(int3c2e_envs),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.c_int(1),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets_gpu[kbatch:].data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(aux_ao_offset),
            ctypes.c_int(nao), ctypes.c_int(nao_pair),
            ctypes.c_int(naux_in_batch), ctypes.c_int(mol.natm))
        if err != 0:
            raise RuntimeError('int3c2e_ejk_ip1 failed')
    buf = buf1 = buf2 = compressed = dm_tensor = tmp = None
    # hermi == 1 throughout: the pair list carries one triangle
    ejk *= 2
    ejk_aux *= 2
    t0 = log.timer_debug1('fastdfgrad: pseudo-density and int3c2e_ip1', *t0)

    if auxbasis_response:
        dm_aux = auxvec[:, None] * auxvec if j_factor != 0 else None
        if unrestricted:
            dm_aux = contract('nrij,nsij->rs', dm_oo, dm_oo,
                              alpha=k_half, beta=j_factor, out=dm_aux)
        else:
            dm_aux = contract('rij,sij->rs', dm_oo, dm_oo,
                              alpha=k_half, beta=j_factor, out=dm_aux)
        dm_aux = dm_aux[aux_sorting[:, None], aux_sorting]
        ejk_aux -= cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux))
        ejk += ejk_aux
        t0 = log.timer_debug1('fastdfgrad: int2c2e_ip1', *t0)
    return ejk.get()


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------
def jk_energy_per_atom(self, dm=None, j_factor=1, k_factor=1, omega=0,
                       hermi=0, verbose=None):
    """Drop-in replacement for ``df/grad/rhf.py:Gradients.jk_energy_per_atom``.

    The cderi is harvested *before* ``with_df.reset()``, which is the whole
    reason this replaces the method rather than the module-level function the
    method calls: by the time ``_jk_energy_per_atom`` runs, the tensor it
    could have reused has already been freed.
    """
    mf = self.base
    if dm is None:
        dm = mf.make_rdm1()
    mol = mf.with_df.mol
    auxmol = mf.with_df.auxmol
    log = logger.new_logger(mol, verbose)

    rhok = None
    dfobj = None
    if not _OFF and _CDERI and k_factor != 0:
        dfobj = _pick_df(mf, omega)
        rhok = _rhok_from_cderi(dfobj, dm, hermi)

    # `DF.build` builds exactly this object -- `Int3c2eOpt(mol, auxmol)` with
    # `mol` replaced by the same decontracted `SortedMole` and the same
    # default cutoff -- and upstream's first line throws it away with
    # everything else.  Only for the full-range call: the screening behind it
    # is the full-range operator's, and a long-range call needs its own.
    reuse_opt = (None if omega or _OFF or not _INTOPT else
                 getattr(mf.with_df, 'intopt', None))

    if rhok is None:
        # upstream's own line, so that `FASTDFGRAD_CDERI=0` is a faithful
        # baseline rather than a nearly-faithful one.  `reuse_opt` survives
        # it: `reset` clears the attribute, not the object a caller is
        # already holding
        mf.with_df.reset()
    else:
        _release(mf.with_df, dfobj)
    with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
        if reuse_opt is not None and reuse_opt._int3c2e_envs is not None:
            int3c2e_opt = reuse_opt
            STATS['intopt_reuse'] += 1
        else:
            sorted_mol = SortedMole.from_mol(mol, decontract=True)
            int3c2e_opt = Int3c2eOpt(sorted_mol, auxmol).build()

        if _OFF or rhok is None:
            if not _OFF and k_factor != 0:
                STATS['rebuilt'] += 1
            return self._jk_energy_per_atom_orig(
                int3c2e_opt, dm, j_factor, k_factor, hermi=hermi,
                auxbasis_response=self.auxbasis_response, verbose=verbose)

        STATS['calls'] += 1
        STATS['cderi_reuse'] += 1
        dm_factor = df_rhf_grad._factorize_dm(int3c2e_opt.mol, dm, hermi)[0]
        if dm_factor.ndim == 2:
            dm_factor = dm_factor[None]
        return _ejk_from_rhok(
            int3c2e_opt, dm_factor, rhok, j_factor, k_factor,
            self.auxbasis_response, log)


_PATCHED = False
#: the replaced methods, for `tests/test_patch_aliases.py`
_ORIG_METHOD_R = _ORIG_METHOD_U = None

#: every class that binds ``jk_energy_per_atom`` as a class attribute.  They
#: bind the *function object*, at class-creation time, so patching
#: ``df/grad/rhf.py`` alone reaches an RHF gradient and none of the other
#: three -- the failure mode ``docs/DF_WORK_LIST.md`` §5 calls "a patch target
#: is not a code path", and the only thing that finds it is
#: ``benchmarks/coverage_probe.py``.
_CLASSES = (df_rhf_grad.Gradients, df_rks_grad.Gradients,
            df_uhf_grad.Gradients, df_uks_grad.Gradients)


def apply_patch():
    global _PATCHED
    if _PATCHED:
        return
    global _ORIG_METHOD_R, _ORIG_METHOD_U
    # the two *methods* being replaced, kept where `tests/test_patch_aliases.py`
    # can walk for a class that still holds one of them -- which would be a
    # DF gradient running GPU4PySCF's own code with this module loaded, and
    # would have no symptom but a wall clock
    _ORIG_METHOD_R = df_rhf_grad.Gradients.jk_energy_per_atom
    _ORIG_METHOD_U = df_uhf_grad.Gradients.jk_energy_per_atom
    for cls in _CLASSES:
        # the module-level function each class's own module would have called,
        # so a decline lands in the right one: `df/grad/uhf.py` has its own
        # `_jk_energy_per_atom`, with the spin index the restricted one has not
        orig = (df_uhf_grad._jk_energy_per_atom
                if cls in (df_uhf_grad.Gradients, df_uks_grad.Gradients)
                else df_rhf_grad._jk_energy_per_atom)
        cls._jk_energy_per_atom_orig = staticmethod(orig)
        cls.jk_energy_per_atom = jk_energy_per_atom
    _PATCHED = True


apply_patch()
