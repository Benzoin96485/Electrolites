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

**Three, and this is ``docs/DF_WORK_LIST.md`` W10.  The occupied-occupied
array is never allocated whole.**  Both routes -- upstream's and the reuse
above -- go through a dense ``(naux, nocc, nocc)`` array, and upstream
allocates it *twice*: ``df/grad/rhf.py:101`` for ``j3c_oo`` and
``df/grad/rhf.py:124`` for ``dm_oo = metric @ j3c_oo``.  At PfPMT/def2-SVP
(284 atoms, naux 13365, nocc 547) that is **32.0 GB each** on a 40 GB card,
and the function dies before it reaches an integral -- so the density-fitted
gradient did not run at a size where the SCF does.  Two things in
``_OOStore`` fix it and neither costs an integral:

* the second copy is never made.  The metric pass runs **in place**, in
  slices of the first occupied index, because ``coef @ oo`` mixes only the
  auxiliary index and a slice of the occupied one is therefore independent
  of every other slice.  The peak falls from ``2 naux nocc^2`` to
  ``(1 + 2/S) naux nocc^2``;
* and the array itself goes to **page-locked host memory** when it still
  does not fit, which is what ``_cholesky_eri`` already does one level down
  with the cderi.  It is filled one auxiliary block at a time, read back one
  slice of the occupied index at a time for the metric pass, and streamed
  back a row slab at a time under the ip1 kernel, with the next slab's read
  overlapping the current slab's kernel.

What the work list expected this to cost -- "extra passes over the cderi" --
it does not: the array is *built* auxiliary-block by auxiliary-block in both
routes, so one pass over the cderi (or over ``eval_j3c``) is enough, and only
the metric pass has to see the whole auxiliary index at once.  What it does
cost is four traversals of an ``naux nocc^2`` array over PCIe plus one
page-locking, which is 1.087x at 42 atoms and 1.294x at 113 and is predicted
to 4 % by those two rates alone.  The threshold between the two stores was
measured rather than guessed, and the measurement says there is none to find
(``benchmarks/df_grad_oo_sweep.py --ballast``): the device store is ahead at
every rung of a ladder that withholds memory until it will no longer
allocate, so it is preferred whenever it *can* be allocated and spills to the
host when it cannot.  ``FASTDFGRAD_OOSTORE`` forces either.

The same batching is available to the route that has **no** cderi to reuse --
a scanner, a gradient on a reconstructed density, or ``FASTDFGRAD_CDERI=0``.
There ``_ejk_from_j3c`` is a clone of upstream's own first loop that writes
its ``j3c_oo`` into the same store, so that route runs at 284 atoms too and
the reuse can be priced against something at that size.  Below the threshold
nothing changes: the call is handed to GPU4PySCF's own function, byte for
byte, as it always was.

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
    FASTDFGRAD_OOSTORE=auto|device|host
                            where the `(naux, nocc, nocc)` array lives.
                            `auto` puts it on the card while it is under
                            `FASTDFGRAD_OOFRAC` of free device memory and on
                            pinned host memory when it is not
    FASTDFGRAD_OOFRAC=<f>   that fraction; 0.8 by default, and the ladder
                            behind that number found no crossover below it
    FASTDFGRAD_OOSLICE=<n>  occupied columns per slice of the metric pass;
                            0 (the default) takes as many as fit
    FASTDFGRAD_OOPREFETCH=0 do not overlap the host read of the next row slab
                            with the current slab's ip1 kernel
    FASTDFGRAD_OOPIN=register|alloc|none
                            how the host array is page-locked.  W12 measured
                            `cudaHostRegister` at 8.6 GB/s against
                            `cudaHostAlloc`'s 2.09, so `register` is the
                            default; `none` leaves it pageable, which costs
                            the overlap
    FASTDFGRAD_J3CBATCH=auto|always|never
                            whether the route that has no cderi to reuse is
                            batched too.  `auto` takes it only where
                            GPU4PySCF cannot allocate at all, so that
                            `FASTDFGRAD_CDERI=0` stays a faithful baseline
                            everywhere else
    FASTDFGRAD_OOSIZING=new|upstream
                            the ip1 loop's block sizing.  An **ablation**:
                            `upstream` puts back the line that reserves for
                            an array this module no longer allocates
    FASTDFGRAD_FILLBUF=0    the other ablation: put back the per-block
                            cuTENSOR temporary in the cderi half-transform
"""
import ctypes
import os

import numpy as np
import cupy as cp
from cupy.cuda import runtime
from pyscf import lib

from gpu4pyscf.df import df as DFMOD
from gpu4pyscf.df import df_jk as DFJK
from gpu4pyscf.df.grad import rhf as df_rhf_grad
from gpu4pyscf.df.grad import rks as df_rks_grad
from gpu4pyscf.df.grad import uhf as df_uhf_grad
from gpu4pyscf.df.grad import uks as df_uks_grad
from gpu4pyscf.df.int3c2e_bdiv import (
    _split_l_ctr_pattern, argsort_aux, get_ao_pair_loc, int2c2e,
    int3c2e_scheme, libvhf_rys, Int3c2eOpt)
from gpu4pyscf.gto.mole import SortedMole
from gpu4pyscf.lib import logger, multi_gpu
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem, ndarray

try:                                         # cuTENSOR is optional upstream
    from cupy_backends.cuda.libs.cutensor import CuTensorError
    _PLAN_ERRORS = (CuTensorError, cp.cuda.memory.OutOfMemoryError)
except ImportError:                                           # pragma: no cover
    _PLAN_ERRORS = (cp.cuda.memory.OutOfMemoryError,)

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

# -- W10, the (naux, nocc, nocc) array ------------------------------------
#: where that array lives: `device`, `host`, or `auto` -- on the card while it
#: is under `_OOFRAC` of free device memory, on pinned host memory when it is
#: not.  `docs/README_fastdfgradoo.md`.
_OOSTORE = os.environ.get('FASTDFGRAD_OOSTORE', 'auto').lower()
#: the fraction of free device memory the array may take before `auto` sends
#: it to the host.  It is a *threshold*, so it was measured rather than
#: guessed -- and the measurement says there is no crossover to find.  Walked
#: up a ballast ladder on taxol until the device store would no longer
#: allocate, it is ahead at **every** rung: 1.34x with 22 GB withheld, 1.18x
#: with 24, and still 1.04x with 24.5, where the host store's time is flat and
#: the device store's is the one degrading.  So the crossover coincides with
#: the allocation limit rather than preceding it, and this is set high and
#: backed by a fall-back on `OutOfMemoryError` rather than set where a
#: crossover was found.  `benchmarks/df_grad_oo_sweep.py --ballast`.
_OOFRAC = float(os.environ.get('FASTDFGRAD_OOFRAC', 0.8))
#: occupied columns per slice of the metric pass; 0 takes as many as fit.
_OOSLICE = int(os.environ.get('FASTDFGRAD_OOSLICE', 0))
#: overlap the host read of the next auxiliary row slab with the current
#: slab's ip1 kernel, the way `DF.loop` overlaps the cderi's.
_OOPREFETCH = os.environ.get('FASTDFGRAD_OOPREFETCH', '1') != '0'
#: how the host array is page-locked -- `docs/DF_WORK_LIST.md` W12 measured
#: `cudaHostRegister` at 8.6 GB/s against `cudaHostAlloc`'s 2.09.
_OOPIN = os.environ.get('FASTDFGRAD_OOPIN', 'register').lower()
#: force the ip1 loop's auxiliary block rather than deriving it from free
#: memory.  0 is off.  It exists to separate two variables that move together
#: otherwise: `cutensor.createPlan` refuses some blocks, and both the block
#: and the memory the plan has to work in are functions of `get_avail_mem`.
#: `docs/README_fastdfgradoo.md` 4.1.
_BLK = int(os.environ.get('FASTDFGRAD_BLK', 0))
#: give the first half-transform of the cderi a buffer allocated once instead
#: of the temporary cuTENSOR allocates per block.  An **ablation**, like
#: `_OOSIZING`: this round is faster than W3's table at sizes that already
#: worked, and between the two of them they say where that came from.
#: `docs/DF_WORK_LIST.md` §5 item 8 is why a per-block allocation is suspected
#: in the first place.
_FILLBUF = os.environ.get('FASTDFGRAD_FILLBUF', '1') != '0'
#: how the ip1 loop's batch size and block size are derived.  `upstream` is
#: GPU4PySCF's own expression, which subtracts a whole `(naux, nocc, nocc)`
#: array from free memory because it is about to allocate one; `new` does not,
#: because there is no second array any more.  An **ablation**: this round is
#: faster at sizes that already worked and this is how much of that is the
#: block sizing the freed array bought.
_OOSIZING = os.environ.get('FASTDFGRAD_OOSIZING', 'new').lower()
#: batch the route that has no cderi to reuse as well, rather than handing it
#: to a GPU4PySCF function that cannot run it at this size.
_J3CBATCH = os.environ.get('FASTDFGRAD_J3CBATCH', 'auto').lower()

#: counters, so a coverage claim can be read off a run rather than assumed.
STATS = {'calls': 0, 'declined': 0, 'cderi_reuse': 0, 'intopt_reuse': 0,
         'rebuilt': 0, 'blocks': 0, 'naux': 0,
         'oo_host': 0, 'oo_device': 0, 'oo_spilled': 0, 'oo_slices': 0,
         'oo_gb': 0., 'plan_retries': 0,
         'j3c_batched': 0}
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
# W10: the (naux, nocc, nocc) array, and the two sides of the link
# --------------------------------------------------------------------------
class _OOStore:
    """The ``(nset, naux, nocc, nocc)`` occupied-occupied array.

    ``docs/DF_WORK_LIST.md`` **W10**.  Upstream holds this array twice --
    ``j3c_oo`` and then ``dm_oo = metric @ j3c_oo`` -- as dense device
    arrays, and at PfPMT/def2-SVP each of them is 32.0 GB on a 40 GB card, so
    the density-fitted gradient does not run at a size where the SCF does.
    This class is the two changes that remove that, and it is the only thing
    in the module that knows which side of the PCIe link the array is on.

    The array is **written** one auxiliary row slab at a time (that is how
    both routes build it, so one pass over the cderi or over ``eval_j3c`` is
    enough), **read back one slice of the first occupied index at a time**
    for the metric pass -- which is the only step that has to see the whole
    auxiliary index at once, and which therefore runs in place -- and read
    again one row slab at a time under the ip1 kernel.  Row slabs are
    contiguous in this layout and occupied slices are a plain 2-D stride, so
    every transfer is one ``cudaMemcpy`` or ``cudaMemcpy2D`` per spin.

    On the device the same calls are views and cost nothing, which is what
    lets one loop body serve both stores.
    """

    def __init__(self, nset, nrows, nocc, host, log=None):
        self.nset = int(nset)
        self.nrows = int(nrows)
        self.nocc = int(nocc)
        self.host = host
        self.rowbytes = self.nocc * self.nocc * 8
        self.nbytes = self.nset * self.nrows * self.rowbytes
        self._registered = None
        if not host:
            try:
                self.arr = cp.empty((nset, nrows, nocc, nocc))
            except cp.cuda.memory.OutOfMemoryError:
                # `_OOFRAC` is a threshold and a threshold can be wrong; the
                # allocation cannot.  Only under `auto` -- a forced store is
                # a measurement and has to fail rather than quietly become
                # the other one.
                if _OOSTORE != 'auto':
                    raise
                STATS['oo_spilled'] += 1
                host = self.host = True
                if log is not None:
                    log.debug1('fastdfgrad: %.2f GB would not allocate on the '
                               'device; the occupied-occupied array spills to '
                               'the host', self.nbytes * 1e-9)
        if host:
            self.arr = _pinned_empty((nset, nrows, nocc, nocc), self, log)
            self.base = self.arr.ctypes.data
            STATS['oo_host'] += 1
        else:
            self.base = None
            STATS['oo_device'] += 1
        STATS['oo_gb'] = max(STATS['oo_gb'], self.nbytes * 1e-9)

    # -- lifetime ---------------------------------------------------------
    def close(self):
        if self._registered is not None:
            # unregistering host memory a copy is still landing in is
            # undefined; this is once per gradient call
            cp.cuda.Device().synchronize()
            try:
                runtime.hostUnregister(self._registered)
            except Exception:                                  # noqa: BLE001
                pass
            self._registered = None
        self.arr = None
        self.base = None

    def __del__(self):
        try:
            self.close()
        except Exception:                                      # noqa: BLE001
            pass

    # -- auxiliary row slabs, the layout the ip1 loop asks for ------------
    def out_rows(self, a0, a1, buf=None):
        """A device array to build rows ``[a0:a1]`` into."""
        if not self.host:
            return self.arr[:, a0:a1]
        return ndarray((self.nset, a1 - a0, self.nocc, self.nocc), buffer=buf)

    def commit_rows(self, a0, a1, src, stream=None):
        if not self.host:
            if src.data.ptr != self.arr[:, a0:a1].data.ptr:
                self.arr[:, a0:a1] = src
            return
        self._rows_copy(a0, a1, src, runtime.memcpyDeviceToHost, stream)

    def read_rows(self, a0, a1, buf=None, stream=None):
        if not self.host:
            return self.arr[:, a0:a1]
        out = ndarray((self.nset, a1 - a0, self.nocc, self.nocc), buffer=buf)
        self._rows_copy(a0, a1, out, runtime.memcpyHostToDevice, stream)
        return out

    def _rows_copy(self, a0, a1, dev, kind, stream):
        # every offset is a Python int on purpose: `aux_offsets` is a NumPy
        # int32 array, and a device pointer plus a NumPy int32 overflows
        a0, a1 = int(a0), int(a1)
        nb = (a1 - a0) * self.rowbytes
        sp = (stream or cp.cuda.get_current_stream()).ptr
        for s in range(self.nset):
            hptr = int(self.base) + (s * self.nrows + a0) * self.rowbytes
            dptr = int(dev.data.ptr) + s * nb
            if kind == runtime.memcpyHostToDevice:
                runtime.memcpyAsync(dptr, hptr, nb, kind, sp)
            else:
                runtime.memcpyAsync(hptr, dptr, nb, kind, sp)

    # -- slices of the first occupied index, for the metric pass ----------
    def read_cols(self, c0, c1, nrows, buf=None, stream=None):
        """``[: , :nrows, c0:c1, :]`` as a contiguous device array."""
        dc = c1 - c0
        if not self.host:
            view = self.arr[:, :nrows, c0:c1]
            if dc == self.nocc and (nrows == self.nrows or self.nset == 1):
                # the whole occupied index, and a leading axis of length one
                # does not break contiguity: no copy
                return view
            out = ndarray((self.nset, nrows, dc, self.nocc), buffer=buf)
            out[:] = view
            return out
        out = ndarray((self.nset, nrows, dc, self.nocc), buffer=buf)
        self._cols_copy(c0, c1, nrows, out, runtime.memcpyHostToDevice, stream)
        return out

    def write_cols(self, c0, c1, nrows, src, stream=None):
        if not self.host:
            if (c1 - c0 == self.nocc and nrows == self.nrows
                    and src.data.ptr == self.arr.data.ptr):
                return
            self.arr[:, :nrows, c0:c1] = src
            return
        self._cols_copy(c0, c1, nrows, src, runtime.memcpyDeviceToHost, stream)

    def _cols_copy(self, c0, c1, nrows, dev, kind, stream):
        c0, c1, nrows = int(c0), int(c1), int(nrows)
        width = (c1 - c0) * self.nocc * 8
        sp = (stream or cp.cuda.get_current_stream()).ptr
        for s in range(self.nset):
            hptr = (int(self.base) + s * self.nrows * self.rowbytes
                    + c0 * self.nocc * 8)
            dptr = int(dev.data.ptr) + s * nrows * width
            if kind == runtime.memcpyHostToDevice:
                runtime.memcpy2DAsync(dptr, width, hptr, self.rowbytes,
                                      width, nrows, kind, sp)
            else:
                runtime.memcpy2DAsync(hptr, self.rowbytes, dptr, width,
                                      width, nrows, kind, sp)

    def rebind(self, arr, nrows):
        """Take ``arr`` as the whole store, without copying it into place.

        Only the device store, and only when the metric pass ran in one
        slice: the output of ``coef @ oo`` is then already the array this
        object is supposed to hold, and copying it back over the input would
        be a second traversal of several GB for nothing.
        """
        assert not self.host
        self.arr = arr.reshape(self.nset, nrows, self.nocc, self.nocc)
        self.nrows = nrows
        self.nbytes = self.nset * nrows * self.rowbytes


def _pinned_empty(shape, owner, log=None):
    """A page-locked host array, by the cheaper of the two routes.

    ``docs/DF_WORK_LIST.md`` **W12** measured them: ``cudaHostRegister`` on
    an ordinary NumPy buffer runs at **8.6 GB/s** and ``cudaHostAlloc`` --
    which is what ``cupy_helper.empty_mapped`` calls -- at 2.09.  At 32 GB
    that is 3.7 s against 15, paid once per gradient, so this takes the
    first.  Both fall back to pageable memory rather than failing the
    calculation; what that costs is the overlap, not the answer.
    """
    if _OOPIN == 'alloc':
        from gpu4pyscf.lib.cupy_helper import empty_mapped
        return empty_mapped(shape)
    arr = np.empty(shape)
    if _OOPIN == 'none':
        return arr
    try:
        runtime.hostRegister(arr.ctypes.data, arr.nbytes, 0)
        owner._registered = arr.ctypes.data
    except Exception as exc:                                   # noqa: BLE001
        if log is not None:
            log.debug1('fastdfgrad: cudaHostRegister of %.2f GB failed (%s); '
                       'the occupied-occupied array stays pageable',
                       arr.nbytes * 1e-9, exc)
    return arr


def _want_host(nbytes, mem_free=None):
    """Whether an array of ``nbytes`` belongs on the host rather than the card.

    A coarse pre-filter rather than the decision: the ladder behind
    ``_OOFRAC`` found no crossover below it, so the device store is preferred
    wherever it can be had and what settles the hard cases is that
    ``_OOStore`` *attempts* the allocation and spills to the host on
    ``OutOfMemoryError``.  A threshold can be wrong about a machine it was
    not measured on; an allocation cannot.
    """
    if _OOSTORE == 'host':
        return True
    if _OOSTORE == 'device':
        return False
    if mem_free is None:
        mem_free = get_avail_mem(exclude_memory_pool=True)
    return nbytes > _OOFRAC * mem_free


def _col_slice(nset, nrows, nocc, extra=0):
    """How many columns of the first occupied index one metric slice takes.

    The pass holds two of them -- the input rows and the output rows, which
    have different auxiliary dimensions when the fit dropped directions --
    so this is sized against both.  ``nocc`` (one slice, the whole array) is
    what comes back whenever it fits, which is every system below the
    threshold and is why nothing changes there.
    """
    if _OOSLICE > 0:
        return max(1, min(nocc, _OOSLICE))
    # 0.4 rather than `_OOFRAC`: that one is a *placement* threshold and is
    # deliberately near the allocation limit, where this is a working set the
    # ip1 loop has to be left room around.  Slicing costs nothing measurable
    # (`docs/README_fastdfgradoo.md` 2.3), so being generous here buys nothing.
    budget = .4 * get_avail_mem(exclude_memory_pool=True) - extra
    per = 2. * nset * nrows * nocc * 8
    return max(1, min(nocc, int(budget // per) if per > 0 else nocc))


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


def _reuse_factor(dfobj, dm, hermi):
    """The density factor this call can be answered with, or ``None``.

    Every decline lives here, and it runs before anything is built or
    allocated -- a call this module cannot answer must reach GPU4PySCF's own
    function having cost nothing and having freed the same memory
    ``reset()`` would have.
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
    if factor_l.shape[1] != dfobj.nao:
        return _decline('density factor is not in the cderi AO basis')
    return factor_l


def _fill_from_cderi(dfobj, factor_l, naux_out, log):
    """``rhok[L,i,j] = sum_pq cderi[L,pq] C[p,i] C[q,j]``, into a store.

    The two contractions are GPU4PySCF's own, in GPU4PySCF's own layout: the
    block ``DF.loop`` yields is an ``(i,j,L)`` buffer presented as ``(L,i,j)``,
    and taking the ``(p,q,L)`` view back makes this the same pair of calls the
    function's first loop makes on its own dense block.  That is deliberate --
    ``docs/README_fastdf.md`` §3.4 measured three rearrangements of exactly
    this contraction and kept none of them.

    Two things here are W10's rather than W3's.  The result goes into an
    ``_OOStore`` **one auxiliary block at a time**, which is why the host
    store costs no extra pass over the cderi -- the array was always built
    this way, it was only ever *held* whole.  And the intermediate of the
    first contraction is now a buffer allocated once rather than a temporary
    allocated per block: ``docs/DF_WORK_LIST.md`` §5 item 8 is what a
    ``cudaFree`` inside a loop that relies on a prefetch costs, and at 284
    atoms this one is 1.6 GB and there are a hundred of them.
    """
    nset, nao, nocc = factor_l.shape
    naux_in = dfobj._cderi[cp.cuda.Device().id].shape[0]
    nrows = max(naux_in, naux_out)
    store = _OOStore(nset, nrows, nocc, _want_host(nset * nrows * nocc**2 * 8),
                     log)
    blksize = dfobj.get_blksize(extra=nocc * nao)
    tmp_buf = cp.empty(nocc * nao * blksize) if _FILLBUF else None
    slab_buf = cp.empty(nset * blksize * nocc * nocc) if store.host else None
    p1 = 0
    for cderi_blk, _ in dfobj.loop(blksize=blksize, unpack=True):
        nL = cderi_blk.shape[0]
        p0, p1 = p1, p1 + nL
        # `loop` hands back a (L,i,j) view of an (i,j,L) buffer; the view back
        # is what makes both contractions the ones GPU4PySCF already issues
        work = cderi_blk.transpose(1, 2, 0)
        slab = store.out_rows(p0, p1, slab_buf)
        for s in range(nset):
            if _FILLBUF:
                tmp = ndarray((nocc, nao, nL), buffer=tmp_buf)
                contract('pqr,pi->iqr', work, factor_l[s], out=tmp)
            else:
                # what W3 did: cuTENSOR allocates the intermediate, and the
                # `cudaFree` that ends it lands inside the block loop
                tmp = contract('pqr,pi->iqr', work, factor_l[s])
            contract('iqr,qj->rij', tmp, factor_l[s], out=slab[s])
        store.commit_rows(p0, p1, slab)
        STATS['blocks'] += 1
    assert p1 == naux_in, (p1, naux_in)
    STATS['naux'] = naux_in
    return store, naux_in


def _fill_from_j3c(int3c2e_opt, dm_factor, batch_size, blksize, naux,
                   host, log):
    """``j3c_oo`` for the route that has no cderi -- upstream's loop, batched.

    ``df/grad/rhf.py:95-110`` verbatim, with two changes: the result goes into
    an ``_OOStore`` rather than a ``(naux, nocc, nocc)`` device array, and the
    per-block intermediate is a buffer rather than a temporary.  Nothing about
    the integrals, the batching or the screening moves, so below the store's
    threshold this computes what upstream computes, block for block.

    It exists because a size ceiling that only the *reuse* clears is not a
    cleared ceiling: a scanner, a gradient on a reconstructed density and
    ``FASTDFGRAD_CDERI=0`` all arrive here, and the last of them is the only
    baseline the reuse can be priced against at 284 atoms.
    """
    nset, nao, nocc = dm_factor.shape
    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    nao_pair = len(pair_addresses)
    i_addr, j_addr = divmod(pair_addresses, nao)
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, reorder_aux=True, cart=True)
    aux_batches = len(aux_offsets) - 1

    store = _OOStore(nset, naux, nocc, host, log)
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((batch_size, nao_pair))
    tmp_buf = cp.empty(nocc * nao * blksize)
    slab_buf = cp.empty(nset * blksize * nocc * nocc) if store.host else None
    aux0 = aux1 = 0
    for kbatch in range(aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:, :, :dk]
            j3c[j_addr, i_addr] = j3c[i_addr, j_addr] = compressed[:, k0:k1]
            slab = store.out_rows(aux0, aux1, slab_buf)
            for s in range(nset):
                tmp = ndarray((nocc, nao, dk), buffer=tmp_buf)
                contract('pqr,pi->iqr', j3c, dm_factor[s], out=tmp)
                contract('iqr,qj->rij', tmp, dm_factor[s], out=slab[s])
            store.commit_rows(aux0, aux1, slab)
            STATS['blocks'] += 1
    assert aux1 == naux, (aux1, naux)
    STATS['j3c_batched'] += 1
    return store, aux_sorting


def _metric_pass(store, coef, naux_in, need_auxvec, need_koo, log):
    """``oo <- coef @ oo``, in slices of the first occupied index, in place.

    This is the whole of why the array never has to exist twice.  ``coef``
    mixes the *auxiliary* index and nothing else -- it is ``aux_coef`` on the
    reuse route and the fitting ``metric`` on the other -- so a slice of the
    occupied index is a self-contained problem: read it, multiply it, put it
    back where it came from.  The two quantities that need the whole array at
    once are accumulated here for the same reason, because they are both sums
    over the occupied indices:

        auxvec[r] = sum_i dm_oo[r,i,i]
        koo[r,s]  = sum_ij dm_oo[r,i,j] dm_oo[s,i,j]

    and ``koo`` is what the auxiliary-response term contracts with the
    two-centre derivative.  Upstream forms both after its ip1 loop, from an
    array it is still holding; this forms them before, from an array it is
    about to stop holding.
    """
    nset, nocc = store.nset, store.nocc
    naux_out = coef.shape[0]
    assert coef.shape[1] == naux_in, (coef.shape, naux_in)
    assert store.nrows >= naux_out, (store.nrows, naux_out)
    extra = naux_out * naux_out * 8 if need_koo else 0
    dc = _col_slice(nset, max(naux_in, naux_out), nocc, extra=extra)
    single = dc >= nocc
    STATS['oo_slices'] = max(STATS['oo_slices'], -(-nocc // dc))
    log.debug1('fastdfgrad: metric pass, %d x %d columns, store on %s',
               -(-nocc // dc), dc, 'host' if store.host else 'device')

    auxvec = cp.zeros(naux_out) if need_auxvec else None
    koo = cp.zeros((naux_out, naux_out)) if need_koo else None
    in_buf = (None if single and not store.host
              else cp.empty(nset * naux_in * dc * nocc))
    out_buf = cp.empty(nset * naux_out * dc * nocc)
    for c0 in range(0, nocc, dc):
        c1 = min(nocc, c0 + dc)
        d = c1 - c0
        src = store.read_cols(c0, c1, naux_in, in_buf)
        dst = ndarray((nset, naux_out, d, nocc), buffer=out_buf)
        for s in range(nset):
            cp.dot(coef, src[s].reshape(naux_in, -1),
                   out=dst[s].reshape(naux_out, -1))
        src = None
        if auxvec is not None:
            x = cp.arange(d)
            auxvec += dst[:, :, x, x + c0].sum(axis=(0, 2))
        if koo is not None:
            flat = dst.reshape(nset, naux_out, d * nocc)
            contract('nrk,nsk->rs', flat, flat, alpha=1., beta=1., out=koo)
        if single and not store.host:
            store.rebind(dst, naux_out)
        else:
            store.write_cols(c0, c1, naux_out, dst)
    return auxvec, koo


def _iter_slabs(store, plan, blksize, prefetch):
    """Yield each auxiliary row slab the ip1 loop asks for, in order.

    On the device it is a view and this is a loop over slices.  On the host it
    is a copy, and the next one is issued on a second stream before the caller
    is handed this one -- the structure ``DF.loop`` uses for the cderi, for
    the same reason: the read is a third of the size of the kernel it hides
    behind, and nothing else in the loop is waiting for it.
    """
    if not store.host:
        for it in plan:
            yield it, store.read_rows(it[3], it[4])
        return
    n = store.nset * blksize * store.nocc * store.nocc
    if not prefetch or len(plan) < 2:
        buf = cp.empty(n)
        for it in plan:
            yield it, store.read_rows(it[3], it[4], buf)
        return
    bufs = [cp.empty(n), cp.empty(n)]
    comput = cp.cuda.get_current_stream()
    io = cp.cuda.Stream(non_blocking=True)
    comp_ev, io_ev = cp.cuda.Event(), cp.cuda.Event()
    # the metric pass wrote this array back to the host on the compute
    # stream, and a *non-blocking* stream does not synchronise with the
    # legacy default one -- without this the first slab is read before the
    # last slice of it has landed, and the answer is wrong by 1e-2 with
    # nothing else to show for it
    comp_ev.record(comput)
    io.wait_event(comp_ev)
    nxt = store.read_rows(plan[0][3], plan[0][4], bufs[1], io)
    io_ev.record(io)
    for i, it in enumerate(plan):
        comp_ev.record(comput)
        bufs[0], bufs[1] = bufs[1], bufs[0]
        cur, nxt = nxt, None
        comput.wait_event(io_ev)
        if i + 1 < len(plan):
            io.wait_event(comp_ev)
            nxt = store.read_rows(plan[i + 1][3], plan[i + 1][4], bufs[1], io)
            io_ev.record(io)
        yield it, cur


# --------------------------------------------------------------------------
# the half of the function that has to run either way
# --------------------------------------------------------------------------
def _ejk(int3c2e_opt, dm_factor, store, naux_in,
         j_factor, k_factor, auxbasis_response, log):
    """The metric pass, the pseudo-density, the ip1 kernel and the response.

    This is ``df/grad/rhf.py:112-234`` (and its ``df/grad/uhf.py`` twin, whose
    only difference is a spin index and a factor of two) with three changes.
    **On**: the ip1 kernel is launched with ``nsp_per_block * _SPMULT`` shell
    pairs per thread block rather than the fixed ``* 4``; and the
    occupied-occupied array is an ``_OOStore`` -- multiplied by the fitting
    coefficients in place, a slice of the occupied index at a time, and
    streamed back a row slab at a time if it is on the host (**W10**).
    **Off** (``FASTDFGRAD_JPAIR=1``): the Coulomb term of the pseudo-density
    added on the compressed pair list rather than written into the dense
    ``(nao, nao, blk)`` buffer that ``cp.take`` immediately discards most of
    -- strictly less work, and measured slower.  The integrals, the batching,
    the screening and every symmetry factor are GPU4PySCF's.

    ``store`` is ``None`` for the route that has no cderi to reuse: this
    builds it from ``eval_j3c`` and multiplies by the fitting ``metric``
    instead of by ``aux_coef``, which is the only other thing that differs
    between the two routes.
    """
    from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1_per_atom
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    t0 = log.init_timer()

    nset, nao, nocc = dm_factor.shape
    unrestricted = nset > 1
    naux = auxmol.nao
    # the restricted build carries -0.5*k_factor and the unrestricted one
    # -k_factor, because its two spin channels each hold half the density
    # (`df/grad/rhf.py:179` against `df/grad/uhf.py:156`)
    k_half = -k_factor if unrestricted else -.5 * k_factor

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    nao_pair = len(pair_addresses)

    mem_free = get_avail_mem(exclude_memory_pool=True)
    # what the occupied-occupied array will take out of the card, which is
    # nothing when it is already allocated (the reuse route) or on the host
    pending = 0
    j3c_host = None
    if store is None:
        oo_bytes = nset * naux * nocc**2 * 8
        j3c_host = _want_host(oo_bytes, mem_free)
        pending = 0 if j3c_host else oo_bytes
    if _OOSIZING == 'upstream':
        # the ablation: GPU4PySCF's own line, which reserves for an array
        # this module no longer allocates
        mem_avail = max(mem_free - nset * naux * nocc**2 * 8 - nao**2 * 8,
                        nao_pair * 8 * 8)
    else:
        mem_avail = mem_free - pending - nao**2 * 8
        if auxbasis_response:
            # `koo` is formed before the ip1 loop rather than after it,
            # because the array it is formed from is gone by then
            mem_avail -= naux * naux * 8
        # upstream's two fractions add to 0.9 of what it thinks is free, and
        # what it thinks is free was already a whole occupied-occupied array
        # smaller than this, because it was about to allocate one.  With that
        # array gone -- or on the host -- there is nothing left subtracting
        # it, so the same two fractions would take the whole card; this is
        # the headroom that puts back.
        mem_avail = max(mem_avail * .9, nao_pair * 8 * 8)
    batch_size = max(1, min(naux, int(mem_avail * .5 / (nao_pair * 8))))
    blksize = max(1, min(naux, int(mem_avail * .4 / (nao * nao * 2 * 8)) // 8 * 8))
    if _BLK:
        blksize = min(_BLK, naux)
    if _OOSIZING != 'upstream' and not _BLK and (store.host if store is not None
                                                 else j3c_host):
        # and the two row-slab buffers the host store streams through
        cap = int(mem_avail * .1 / (2 * nset * nocc * nocc * 8)) // 8 * 8
        blksize = max(8, min(blksize, cap)) if cap >= 8 else blksize
    log.debug1('%.3f GB free. sizing=%s nao_pair=%d naux=%d batch_size=%d '
               'blksize=%d', mem_free * 1e-9, _OOSIZING, nao_pair, naux,
               batch_size, blksize)

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
    # is the one the store has to be multiplied in and the one `dm_aux` is
    # undone with at the end.  Upstream reads it off its own
    # `int3c2e_evaluator`; this reads it off the same two arrays, which is
    # what the assertion commented out at `df/grad/rhf.py:148` says is the
    # same thing.
    aux_sorting = argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux)
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu + mol.nbas, dtype=np.int32)
    aux_offsets = aux_loc[ksh_offsets_cpu]
    aux_batches = len(aux_offsets) - 1

    # -- the occupied-occupied array, and the coefficients it is multiplied by
    if store is None:
        store, sorting_eval = _fill_from_j3c(
            int3c2e_opt, dm_factor, batch_size, blksize, naux, j3c_host, log)
        naux_in = naux
        assert bool(cp.array_equal(sorting_eval, aux_sorting)), (
            'the evaluator and _split_l_ctr_pattern disagree on the '
            'auxiliary permutation')
        t0 = log.timer_debug1('fastdfgrad: j3c_oo (no cderi to reuse)', *t0)
        j2c = int2c2e(auxmol)
        coef = cp.asarray(auxmol.ctr_coeff)
        coef, tmp = cp.empty_like(coef), coef
        coef[aux_sorting] = tmp
        tmp = None
        solver = df_rhf_grad._gen_metric_solver(
            j2c, 'CD' if (mol.omega <= 0 and not auxmol.mol.cart) else 'ED')
        coef = coef.dot(solver(coef.T))
        j2c = solver = None
    else:
        # `aux_coef @ aux_coef.T == metric` is the identity the reuse rests
        # on; associating it this way is one `(naux x naux_fit)` product
        # where upstream's is a `(naux x naux)` one
        coef = DFMOD._decompose_j2c(auxmol, aux_sorting)[0]

    auxvec, koo = _metric_pass(store, coef, naux_in,
                               j_factor != 0, auxbasis_response, log)
    coef = None
    t0 = log.timer_debug1('fastdfgrad: dm_oo, in place', *t0)

    if j_factor != 0:
        dm = cp.zeros((nao, nao))
        for s in range(nset):
            dm += dm_factor[s].dot(dm_factor[s].T)
        dm_pair = cp.take(dm.reshape(-1), pair_addresses) if _JPAIR else None
    else:
        dm = dm_pair = None

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libvhf_rys.sum_ejk_int3c2e_ip1
    ejk = cp.zeros((mol.natm, 3))
    ejk_aux = cp.zeros((mol.natm, 3))

    def launch(kbatch, compressed):
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
            ctypes.c_int(aux_loc[ksh_offsets_cpu[kbatch]]),
            ctypes.c_int(nao), ctypes.c_int(nao_pair),
            ctypes.c_int(int(aux_offsets[kbatch + 1] - aux_offsets[kbatch])),
            ctypes.c_int(mol.natm))
        if err != 0:
            raise RuntimeError('int3c2e_ejk_ip1 failed')

    def ip1_loop(blksize):
        """The pseudo-density, the `cp.take` and the ip1 launches.

        A function of the block size because the block size may have to
        change *during* the call: for some extents ``cutensor.createPlan``
        answers ``CUTENSOR_STATUS_NOT_SUPPORTED`` and the contraction below
        fails with tens of GB free.  That is the same wall
        ``docs/README_fastdfhess.md`` hit on the DF Hessian's CPHF build, and
        W10 made it reachable here: with no second ``(naux, nocc, nocc)``
        array to reserve for, `mem_avail` is larger and this expression picks
        a bigger block than GPU4PySCF would have.

        A **retry rather than a cap**, and that is measured rather than
        preferred.  Pinning the block with ``FASTDFGRAD_BLK`` and moving the
        free memory independently (``docs/README_fastdfgradoo.md`` §4.1) says
        the refusal does not depend on the memory at all -- 1096 is refused
        at 70 GB free and at 46 -- and is **not monotonic in the block**:
        1096 is refused where 1240, 1456, 2048 and 3072 are accepted, the
        last of those with 4.3e9 elements in the output, twice 2**31.  So no
        element cap describes this boundary; what clears a refusal is
        changing the extent, which is what halving does.

        A rejected *plan* raises before any kernel is launched and before any
        operand is read, and nothing in this loop writes to the store, so
        halving the block and starting the loop again is safe.
        """
        buf = cp.empty((nao_pair * batch_size))
        buf1 = cp.empty((blksize, nao, nao))
        buf2 = cp.empty((nset, blksize, nocc, nao))
        plan = []
        a1 = 0
        for kbatch in range(aux_batches):
            nb = aux_offsets[kbatch + 1] - aux_offsets[kbatch]
            for k0, k1 in lib.prange(0, nb, blksize):
                a0, a1 = a1, a1 + (k1 - k0)
                plan.append((kbatch, k0, k1, a0, a1))
        cur = -1
        compressed = None
        for (kbatch, k0, k1, a0, a1), oo in _iter_slabs(
                store, plan, blksize, _OOPREFETCH):
            if kbatch != cur:
                if compressed is not None:
                    launch(cur, compressed)
                cur = kbatch
                compressed = ndarray(
                    (nao_pair,
                     int(aux_offsets[kbatch + 1] - aux_offsets[kbatch])),
                    buffer=buf)
            dk = k1 - k0
            dm_tensor = ndarray((nao, nao, dk), buffer=buf1)
            tmp = ndarray((nset, nocc, nao, dk), buffer=buf2)
            beta = 0.
            if j_factor != 0 and not _JPAIR:
                cp.multiply(dm[:, :, None], auxvec[a0:a1], out=dm_tensor)
                beta = j_factor
            if unrestricted:
                contract('nrji,nqj->niqr', oo, dm_factor, out=tmp)
                contract('niqr,npi->pqr', tmp, dm_factor, k_half, beta,
                         out=dm_tensor)
            else:
                contract('rji,qj->iqr', oo[0], dm_factor[0], out=tmp[0])
                contract('iqr,pi->pqr', tmp[0], dm_factor[0], k_half, beta,
                         out=dm_tensor)
            cp.take(dm_tensor.reshape(-1, dk), pair_addresses, axis=0,
                    out=compressed[:, k0:k1])
            if j_factor != 0 and _JPAIR:
                # the same term upstream writes into `dm_tensor`, on the only
                # `nao_pair` of its `nao*nao` entries that survive `cp.take`
                compressed[:, k0:k1] += (j_factor * dm_pair[:, None]
                                         * auxvec[a0:a1])
        if compressed is not None:
            launch(cur, compressed)

    while True:
        try:
            ip1_loop(blksize)
            break
        except _PLAN_ERRORS:
            if blksize <= 8:
                raise
            blksize = max(8, blksize // 2)
            STATS['plan_retries'] += 1
            ejk[:] = 0.
            ejk_aux[:] = 0.
            # the abandoned slab generator may have a prefetch in flight on
            # its own stream, and the buffer it lands in is about to be
            # handed back to the allocator
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            log.debug1('fastdfgrad: the ip1 contraction was refused; '
                       'block size halved to %d', blksize)
    store.close()
    # hermi == 1 throughout: the pair list carries one triangle
    ejk *= 2
    ejk_aux *= 2
    t0 = log.timer_debug1('fastdfgrad: pseudo-density and int3c2e_ip1', *t0)

    if auxbasis_response:
        dm_aux = koo
        dm_aux *= k_half
        if j_factor != 0:
            dm_aux += j_factor * (auxvec[:, None] * auxvec)
        dm_aux = dm_aux[aux_sorting[:, None], aux_sorting]
        ejk_aux -= cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux))
        ejk += ejk_aux
        t0 = log.timer_debug1('fastdfgrad: int2c2e_ip1', *t0)
    return ejk.get()


def _nocc_hint(dm):
    """How many occupied orbitals this density factorises into, cheaply.

    The size test below has to run on the path that is about to be handed to
    GPU4PySCF, so it must not cost a factorisation: an SCF's density carries
    ``mo_occ`` and this reads the answer off it.  ``None`` means it could not,
    and a call whose size cannot be estimated is left to GPU4PySCF.
    """
    occ = getattr(dm, 'mo_occ', None)
    if occ is None:
        return None
    occ = cp.asarray(occ)
    if occ.ndim == 1:
        return int((occ > 0).sum())
    return int((occ > 0).any(axis=tuple(range(occ.ndim - 1))).sum())


def _batch_the_j3c_route(int3c2e_opt, dm, hermi, k_factor):
    """Whether to run our batched clone of upstream's own function.

    ``auto`` -- the default -- takes it only where GPU4PySCF cannot run at
    all, that is where its **two** dense ``(naux, nocc, nocc)`` arrays and
    the metric would not fit.  Below that the call goes to GPU4PySCF, byte
    for byte, so that ``FASTDFGRAD_CDERI=0`` stays the faithful baseline
    every table in ``docs/README_fastdfgrad.md`` is measured against.
    """
    if _J3CBATCH == 'never' or k_factor == 0 or hermi != 1:
        return False
    if multi_gpu.num_devices != 1 or getattr(dm, 'symmetrize', 0):
        return False
    if _J3CBATCH == 'always':
        return True
    nocc = _nocc_hint(dm)
    if nocc is None:
        return False
    naux = int3c2e_opt.auxmol.nao
    nset = 2 if getattr(dm, 'ndim', 2) == 3 else 1
    need = 2. * nset * naux * nocc**2 * 8 + 2. * naux * naux * 8
    return (_OOSTORE == 'host'
            or need > .85 * get_avail_mem(exclude_memory_pool=True))


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

    factor_l = None
    dfobj = None
    if not _OFF and _CDERI and k_factor != 0:
        dfobj = _pick_df(mf, omega)
        # every decline is settled here, before anything is built: a call
        # this module cannot answer has to reach GPU4PySCF's own function
        # having cost nothing
        factor_l = _reuse_factor(dfobj, dm, hermi)

    # `DF.build` builds exactly this object -- `Int3c2eOpt(mol, auxmol)` with
    # `mol` replaced by the same decontracted `SortedMole` and the same
    # default cutoff -- and upstream's first line throws it away with
    # everything else.  Only for the full-range call: the screening behind it
    # is the full-range operator's, and a long-range call needs its own.
    reuse_opt = (None if omega or _OFF or not _INTOPT else
                 getattr(mf.with_df, 'intopt', None))

    if factor_l is None:
        # upstream's own line, so that `FASTDFGRAD_CDERI=0` is a faithful
        # baseline rather than a nearly-faithful one.  `reuse_opt` survives
        # it: `reset` clears the attribute, not the object a caller is
        # already holding
        mf.with_df.reset()

    with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
        if reuse_opt is not None and reuse_opt._int3c2e_envs is not None:
            int3c2e_opt = reuse_opt
            STATS['intopt_reuse'] += 1
        else:
            sorted_mol = SortedMole.from_mol(mol, decontract=True)
            int3c2e_opt = Int3c2eOpt(sorted_mol, auxmol).build()

        if factor_l is not None:
            # the store is sized against *both* auxiliary dimensions -- the
            # cderi's fitting one and the gradient's Cartesian one -- so that
            # the metric pass can be done in place
            t0 = log.init_timer()
            store, naux_in = _fill_from_cderi(
                dfobj, factor_l, int3c2e_opt.auxmol.nao, log)
            log.timer_debug1('fastdfgrad: rhok from the cderi', *t0)
            factor_l = None
            _release(mf.with_df, dfobj)
            STATS['calls'] += 1
            STATS['cderi_reuse'] += 1
            dm_factor = df_rhf_grad._factorize_dm(int3c2e_opt.mol, dm, hermi)[0]
            if dm_factor.ndim == 2:
                dm_factor = dm_factor[None]
            return _ejk(int3c2e_opt, dm_factor, store, naux_in,
                        j_factor, k_factor, self.auxbasis_response, log)

        if _OFF:
            return self._jk_energy_per_atom_orig(
                int3c2e_opt, dm, j_factor, k_factor, hermi=hermi,
                auxbasis_response=self.auxbasis_response, verbose=verbose)

        if k_factor != 0:
            STATS['rebuilt'] += 1
        if _batch_the_j3c_route(int3c2e_opt, dm, hermi, k_factor):
            fl, fr = df_rhf_grad._factorize_dm(int3c2e_opt.mol, dm, hermi)
            if fr is fl:
                STATS['calls'] += 1
                return _ejk(int3c2e_opt, fl[None] if fl.ndim == 2 else fl,
                            None, None, j_factor, k_factor,
                            self.auxbasis_response, log)
        return self._jk_energy_per_atom_orig(
            int3c2e_opt, dm, j_factor, k_factor, hermi=hermi,
            auxbasis_response=self.auxbasis_response, verbose=verbose)


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
