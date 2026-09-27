"""
Faster XC response kernel (fxc) for GPU4PySCF -- the grid half of every
response property.

Importing this module monkey-patches ``gpu4pyscf.dft.numint.nr_rks_fxc`` and
``NumInt.nr_rks_fxc``, and their unrestricted twins ``nr_uks_fxc`` and
``NumInt.nr_uks_fxc``.  Both halves of each pair are needed: ``nr_rks_fxc_st``,
which is what a closed-shell TDA/TDDFT reaches, calls the **module-level**
``nr_rks_fxc`` rather than the bound method, so patching only the class
attribute leaves the excited-state path on GPU4PySCF's code.

``nr_rks_fxc`` is the single entry point for the XC part of the CPKS solve,
the TDA/TDDFT Davidson matrix-vector product, NMR shielding, IR/Raman and the
C6 coefficients -- ``scf/_response_functions.py`` routes all of them through
it.  Measured on GPU4PySCF 1.8.1 it is 23.2 % of a polarizability and 25.2 %
of a five-root TDA run (``benchmarks/response_anatomy.py``).

What changes, and what does not.  The grid, the AO screening, ``fxc`` itself
and every symmetry factor are GPU4PySCF's; only the arrangement of the work
changes, so the potential matrices agree to round-off.

1. **The two GEMMs are one GEMM each, not one per right-hand side.**
   GPU4PySCF issues, per grid block, a Python loop ``for i in range(nset)``
   for the density and a second one for the potential (``dft/numint.py:1301``,
   ``:1313``).  The ``nset`` masked density matrices are contiguous, so
   ``D . ao_0`` for all of them is one ``(nset*nao_sub, nao_sub) x
   (nao_sub, ngrids)`` GEMM; the ``nset`` weighted AO products likewise give
   the potential in one ``(nset*nao_sub, ngrids) x (ngrids, nao_sub)``.

2. **Three custom kernels replace ten per set.**  ``kernels/fastxcfxc.cu``
   does the density contraction, the ``fxc`` contraction and the AO weighting
   with the set dimension inside the kernel, reading ``ao`` once per tile of
   four sets instead of once per set, and folding in the ``rho1[1:4] *= 2`` of
   the Hermitian density and the ``wv[0] *= .5`` of the GGA potential.  The
   ``fxc`` contraction is the one worth the most: GPU4PySCF's
   ``_contract_rho1_fxc`` broadcasts and sums, which materialises an
   ``(nset, 4, 4, ngrids)`` temporary -- four times the traffic of its own
   result.

3. **A non-Hermitian right-hand side is symmetrised once instead of per
   block.**  For ``hermi = 0`` -- the TDDFT case -- ``eval_rho`` computes a
   second product ``dm^T . ao[0]`` and three more contractions per block.  It
   need not: with ``rho1[0] = sum_u (D.ao_0)_u ao_{0u}`` and
   ``rho1[n] = sum_u ((D + D^T).ao_0)_u ao_{nu}``, both are what the
   ``hermi = 1`` formulae give for ``D -> (D + D^T)/2``, because ``rho1[0]``
   already sees only the symmetric part.  So the density is symmetrised once,
   for ``nset*nao^2`` work per call, and the block loop runs the Hermitian
   path.  ``tests/test_fxc.py`` checks this against GPU4PySCF at
   ``hermi = 0`` rather than resting on the algebra.

**What the catalogue asks for here, and why it is not what shipped.**
Catalogue entries G1 and B4d prescribe P8 -- ``_grouped_block_loop`` plus
``grouped_gemm``, the change that carried ``fastxcgrad``.  It is implemented
here as ``MODE=grouped`` and it is **slower than keeping GPU4PySCF's
one-block-at-a-time issue order**: 0.72 s against the 0.22 s that ships, at
``nset = 20`` on penicillin V.  The reason is that this path already has ``nset`` GEMMs in
flight per block with no synchronisation between them, so there is much less
launch latency to hide than in the gradient, while ``grouped_gemm`` pays a
host-side pointer table built in Python for every ``(block, set)`` pair.  What
does pay is batching along ``nset``, which is a stride in memory rather than a
list of pointers.  ``docs/README_fastxcfxc.md`` has both columns.

**The unrestricted twin.**  ``nr_uks_fxc`` is a separate function in
``dft/numint.py`` with its own per-set Python loop, and on an open-shell
reference it is the grid half of every polarizability, TDA/TDDFT, NMR and C6
run -- ``docs/OPENSHELL_TMC.md`` 4 item 1.  It is the same three contractions
with one structural difference: ``fxc`` is a **2x2 block in the spin index**,
so the potential of one spin channel takes both channels' perturbed
densities.  The layout makes that the *only* difference the kernels see: the
two spins' density matrices are interleaved a, b, a, b, ..., which puts
``rho1`` for ``nset`` unrestricted sets in exactly the memory of ``rho1`` for
``2*nset`` restricted sets, so ``fxc_rho1_*``, ``fxc_scale_*`` and both GEMMs
run unchanged with ``nset`` doubled and only the contraction gets a twin
(``fxc_wv_u_*``, which runs over the pair (spin, component)).

LDA, GGA and meta-GGA are covered.  Meta-GGA is here even though ``fastxc``
declines it in the SCF, because on the response path it is not a tail: an
M06-2X polarizability spends **48.4 %** of its wall clock in this function and
a five-root M06-2X TDA run **47.0 %**, against 25-28 % for B3LYP, since the
``tau`` term needs a density product per gradient component and four potential
GEMMs instead of one.

**Limits, all of which fall through to GPU4PySCF rather than fail.**
The ``mo1``/``occ_coeff`` path and more than one GPU.  ``MODE=grouped``
additionally declines meta-GGA and the unrestricted case: it exists only to
A/B the issue order.

The ``mo1`` path is worth being precise about, because it is *not* the same
function on the two spin cases.  On a restricted reference the Hessian's CPHF
solve does not come here at all -- 1.8.1 serves it from
``hessian/rks.py:nr_rks_fxc_mo``, which this module now also replaces
(``_nr_rks_fxc_mo``, `docs/DF_WORK_LIST.md` W16): the density it is handed is
a density matrix as well as a pair of orbital factors, so it is this
module's blocked path on the matrix and upstream's ``_ao2mo`` after it, with
the density taken from the factors in the grid blocks where that is cheaper
(``docs/README_fastdfhessip.md`` 2).  Upstream's own function contracts
``fxc`` in place and is wrong for meta-GGAs; this one is not.  On an **unrestricted** reference there is no
``nr_uks_fxc_mo``: ``hessian/uks.py:get_veff_resp_mo`` calls ``nr_uks_fxc``
itself, and ``hessian/uhf.py:gen_vind`` tags the perturbed density with
``mo1``, so the open-shell Hessian's CPHF lands on the ``mo1`` branch *of
this function* and is declined here.  Everything that reaches it without that
tag -- TDDFT, CPKS/polarizability, NMR, C6, all of which build a plain
density matrix -- is covered.  ``docs/OPENSHELL_FXC.md`` measures both.

**The occupied side of a Hessian's CPHF (W33).**  Both CPHF callers want
only ``V C_o`` of the potential (``_ao2mo`` multiplies by it next), and hand
a density with factors, ``D = M (2 C_o)^T + h.c.``.  So on a grid block whose
AO count ``nsub`` is above ``nocc`` the block runs on the occupied side:
``X[n] = C_o^T ao[n]`` once, ``Y[n] = M^T ao[n]`` for every set, the density
from ``Y . X`` (``fxc_rho_occ_*``), and ``V C_o`` as four GEMMs
``ao[n] . W[n]^T`` of inner dimension ``ngrids`` (``fxc_w_occ_*``) -- ``8
nocc`` units of work per set and grid point for a meta-GGA where the
AO-matrix route spends ``8 nsub``.  W16's matrix route ran its GEMMs at
15-18 TF/s, the card's rate, so the gap to a GGA was arithmetic, and this
is what changes it; ``docs/README_fastxcfxcmgga.md`` has the tables.  A
restricted GGA takes only the potential half on that side (two GEMMs
against one); an unrestricted Hessian takes both halves for every
functional, through ``hessian/uks.py:get_veff_resp_mo`` -- the method that
calls ``nr_uks_fxc`` and then ``_ao2mo`` -- which this module now replaces.
``FASTXCFXC_OCC=0`` is the ablation.
"""
import functools
import os
import time as _time

import cupy
import numpy as np

from gpu4pyscf.dft import numint as NI
from gpu4pyscf.dft.numint import (
    _contract_rho, _contract_rho1, _contract_rho1_fxc, _scale_ao)
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    add_sparse, contract, grouped_dot, grouped_gemm, ndarray, take_last2d,
    transpose_sum)
from gpu4pyscf.__config__ import num_devices

from ._paths import KERNEL_DIR as HERE

#: 'blocked' is the shipped path.  'nokernel' keeps its batched GEMMs but puts
#: GPU4PySCF's own per-set contraction kernels back, which is the ablation for
#: ``kernels/fastxcfxc.cu``.  'grouped' is the catalogue's P8 prescription --
#: ``grouped_gemm`` over (block, set) -- kept because it is what entry B4d
#: asks for and it measures slower; see the module docstring.
MODE = os.environ.get('FASTXCFXC_MODE', 'blocked')

#: FASTXCFXC_MO=0 hands ``hessian/rks.py:nr_rks_fxc_mo`` back to GPU4PySCF
#: (W16's binding off); it is read at call time, so a benchmark can A/B it
#: in one process.
#:
#: FASTXCFXC_TIME=1 prints the per-stage split of each call.
TIME = int(os.environ.get('FASTXCFXC_TIME', 0))

_WX = int(os.environ.get('FASTXCFXC_WX', 32))
_TSY = int(os.environ.get('FASTXCFXC_TSY', 4))
#: AO indices one block of ``fxc_scale`` writes; the AO index is tiled over
#: blockIdx.z there so that a three-right-hand-side call still fills the card.
_UPB = int(os.environ.get('FASTXCFXC_UPB', 32))

#: One chunk of the grouped ablation holds ``nset * nao_sub * blksize``
#: doubles twice over.  The budget is on bytes rather than on a block count
#: because ``nset`` is a property of the solver -- three dipole directions, or
#: a Davidson subspace -- and not of the molecule.
CHUNK_BYTES = float(os.environ.get('FASTXCFXC_CHUNK_BYTES', 512 << 20))

_T = {'prep': 0., 'gemm1': 0., 'rho1': 0., 'wv': 0., 'scale': 0.,
      'gemm2': 0., 'accum': 0., 'n': 0}

#: `nr_uks_fxc` is answered on two different kinds of input -- a plain
#: perturbed density (TDDFT, CPKS, NMR, C6) and an MO-basis one (an
#: unrestricted Hessian's CPHF) -- and they take different branches here.
#: A decline returns the right answer too, so nothing numerical distinguishes
#: the three; this counter does.  `benchmarks/coverage_probe.py` prints it.
_S = {'ours_u': 0, 'ours_u_mo': 0, 'stock_u': 0, 'ours_mo': 0, 'stock_mo': 0}


def stats():
    """How many unrestricted calls went which way, since import."""
    return dict(_S)


@functools.lru_cache(maxsize=8)
def _module(wx, tsy):
    src = open(os.path.join(HERE, 'fastxcfxc.cu')).read()
    return cupy.RawModule(code=src, backend='nvrtc',
                          options=('-std=c++17', f'-DWX={wx}', f'-DTSY={tsy}'))


@functools.lru_cache(maxsize=16)
def _kernel(name):
    return _module(_WX, _TSY).get_function(name)


def _usable(ni, xc_code, dms, hermi, fxc):
    """Conditions the block loop below assumes.

    None of them is the size of the molecule: they are the functional type,
    what the caller asked for, and how many devices there are.
    """
    if num_devices != 1:
        return False
    if hasattr(dms, 'mo1'):                 # the Hessian's MO-basis CPHF path
        return False
    xctype = ni._xc_type(xc_code)
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        return False
    if getattr(dms, 'ndim', 0) not in (2, 3):
        return False
    # the kernels index fxc as [y][x][g] with a stride of the whole grid
    ncomp = {'LDA': 1, 'GGA': 4, 'MGGA': 5}[xctype]
    if (fxc is None or fxc.ndim != 3 or fxc.shape[:2] != (ncomp, ncomp)
            or not fxc.flags.c_contiguous):
        return False
    return True


def _usable_u(ni, xc_code, dms, hermi, fxc):
    """``_usable``'s unrestricted twin.

    Two differences: the density argument is a pair, and ``fxc`` is a 2x2
    block in the spin index, so the shape the kernels assume is
    ``(2, ncomp, 2, ncomp, ngrids)`` rather than ``(ncomp, ncomp, ngrids)``.
    """
    if num_devices != 1:
        return False
    xctype = ni._xc_type(xc_code)
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        return False
    if hasattr(dms, 'mo1'):
        # Unlike the restricted side, this branch *is* reached on 1.8.1: it
        # is an unrestricted Hessian's CPHF solve.  Only the density
        # evaluation differs and it stays GPU4PySCF's `eval_rho4`; what this
        # module batches is everything after it.
        if getattr(dms, 'ndim', 0) != 4 or len(dms.mo1) != 2:
            return False
        return _fxc_ok(xctype, fxc)
    ndim = getattr(dms, 'ndim', None)
    if ndim is None:                        # a (dma, dmb) pair rather than an
        if dms is None or len(dms) != 2:    # array stacked on the spin axis
            return False
        ndim = getattr(dms[0], 'ndim', 0) + 1
    if ndim not in (3, 4):
        return False
    return _fxc_ok(xctype, fxc)


def _fxc_ok(xctype, fxc):
    # the kernels index the unrestricted fxc as [(t,y)][(s,x)][g] with a
    # stride of the whole grid, which is the (2, ncomp, 2, ncomp, ngrids)
    # array reshaped -- so it has to be contiguous and that exact shape
    ncomp = {'LDA': 1, 'GGA': 4, 'MGGA': 5}[xctype]
    return (fxc is not None and fxc.ndim == 5
            and fxc.shape[:4] == (2, ncomp, 2, ncomp)
            and fxc.flags.c_contiguous)


def _prep(ni, mol, grids, dms, hermi, nspin=1):
    """Sort the density matrices and, for ``nspin = 2``, interleave them.

    Returns ``(opt, dms, nset, dm_shape)`` with ``dms`` of shape
    ``(nspin*nset, nao, nao)``.  The interleaving -- a, b, a, b, ... rather
    than all of a then all of b -- is what lets the unrestricted case reuse
    every kernel but the ``fxc`` contraction: with it, ``rho1`` for ``nset``
    unrestricted sets is bit-for-bit the same memory as ``rho1`` for
    ``2*nset`` restricted sets of ``ncomp`` components, so ``fxc_rho1_*`` and
    ``fxc_scale_*`` run unchanged with ``nset`` doubled.
    """
    opt = getattr(ni, 'gdftopt', None)
    if opt is None or mol not in [opt.mol, opt._sorted_mol]:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    if nspin == 2 and hasattr(dms, 'mo1'):
        # An unrestricted Hessian's CPHF hands the perturbation in the MO
        # basis (`hessian/uhf.py:gen_vind` tags it), and GPU4PySCF then
        # evaluates the density from it rather than from a density matrix.
        # That evaluation is left to GPU4PySCF -- see `_fxc_blocked` -- so
        # all this has to do is sort and report the shapes.
        mo1 = [opt.sort_orbitals(cupy.asarray(m), axis=[1])
               for m in dms.mo1]
        occ_coeff = [opt.sort_orbitals(cupy.asarray(c), axis=[0])
                     for c in dms.occ_coeff]
        return opt, None, mo1[0].shape[0], cupy.asarray(dms[0]).shape, \
            (mo1, occ_coeff)
    if nspin == 1:
        dms = cupy.asarray(dms)
        dm_shape = dms.shape
        nao = dm_shape[-1]
        dms = opt.sort_orbitals(dms.reshape(-1, nao, nao), axis=[1, 2])
        dms = cupy.asarray(dms, order='C')
        nset = dms.shape[0]
    else:
        dma, dmb = cupy.asarray(dms[0]), cupy.asarray(dms[1])
        dm_shape = dma.shape
        nao = dm_shape[-1]
        dma = opt.sort_orbitals(dma.reshape(-1, nao, nao), axis=[1, 2])
        dmb = opt.sort_orbitals(dmb.reshape(-1, nao, nao), axis=[1, 2])
        nset, nao_s = dma.shape[0], dma.shape[-1]
        dms = cupy.empty((nset, 2, nao_s, nao_s))
        dms[:, 0] = dma
        dms[:, 1] = dmb
        dms = dms.reshape(2 * nset, nao_s, nao_s)
    if hermi != 1:
        # item 3 of the module docstring: (D + D^T)/2 gives the same rho1.
        # It is per density matrix and says nothing about spin, so the
        # unrestricted case symmetrises the stacked array and is done.
        dms = transpose_sum(dms, inplace=False)
        dms *= .5
    return opt, dms, nset, dm_shape, None


def _fxc_blocked(ni, mol, grids, xc_code, dm0, dms, relativity, hermi,
                 rho0, vxc, fxc, max_memory, verbose, kernels=True, nspin=1,
                 factors=None, occ=None):
    """The shipped path, for one spin channel or two.

    The unrestricted case is the restricted one with the density matrices
    interleaved (``_prep``) and ``nset`` doubled everywhere the kernels are
    launched -- ``nsk`` below -- because on this path a spin channel is just
    another right-hand side.  The single place where that is not true is the
    ``fxc`` contraction, which couples the channels: `fxc` is a 2x2 block in
    the spin index, so the potential of spin ``t`` takes both spins' perturbed
    densities.  That contraction runs over the pair (spin, component) and has
    its own kernel (``fxc_wv_u_*``); everything on either side of it is shared.

    With ``occ`` (the occupied orbitals ``C`` of a CPHF right-hand side
    ``D = M (2C)^T + h.c.``, ``factors`` then being ``(M, 2C)``) it returns
    ``V C`` of shape ``(nset, nao, nocc)`` instead of ``V``, and takes the
    blocks where that is cheaper on the occupied side -- W33, `_occ_setup`.
    """
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    if TIME:
        cupy.cuda.Stream.null.synchronize()
        _t = _time.perf_counter()
    xctype = ni._xc_type(xc_code)
    opt, dms, nset, dm_shape, mo = _prep(ni, mol, grids, dms, hermi, nspin)
    mo1, occ_coeff = mo if mo is not None else (None, None)
    #: how many right-hand sides the per-set kernels see -- a spin channel is
    #: one of them everywhere except in the `fxc` contraction
    nsk = nspin * nset
    _sorted_mol = opt._sorted_mol

    lda = xctype == 'LDA'
    mgga = xctype == 'MGGA'
    ncomp = 1 if lda else (5 if mgga else 4)
    # how many density products D . ao[n] the density needs: meta-GGA's tau
    # term wants one per gradient component, everything else only ao[0]
    nc0 = 4 if mgga else 1
    ao_deriv = 0 if lda else 1
    scale0 = 1.0 if lda else 0.5
    # `_tau_dot` halves wv[4] once more on top of the .5 its caller applies
    scale4 = 0.25 if mgga else 1.0
    suffix = 'lda' if lda else ('mgga' if mgga else 'gga')
    nao = _sorted_mol.nao
    ngrids_tot = fxc.shape[-1]
    # (2, ncomp, 2, ncomp, ngrids) is contiguous, so the pair index the
    # unrestricted contraction runs over is a reshape rather than a copy
    fxc_k = fxc.reshape(nspin * ncomp, nspin * ncomp, ngrids_tot)
    vmat = cupy.zeros((nsk, nao, nao))

    blk = NI.MIN_BLK_SIZE
    nsubs = [len(x[1]) for x in grids.get_non0ao_idx(opt)]
    nsub_max = max(max(nsubs, default=1), 1)
    # every buffer here is linear in `nsk`, so an unrestricted call holds
    # twice what a restricted one of the same `nset` does.  There is
    # deliberately no threshold test: an allocation that cannot be met
    # raises, where a guard would add a way to decline silently on a case
    # that would have worked -- `docs/ROUND2_1.8.1.md` 5 is the round that
    # cost.
    # The AO-matrix route's big buffers are allocated at the first block
    # that takes that route (W33): when every block goes to the occupied
    # side they are never needed, and at 113 atoms c0's alone is 31 GB.
    # Each is sized, when first needed, for the largest block that will take
    # its route (`_hint`, set below once the route is decided), and grown if
    # a block ever asks for more.
    _lazy = {}
    _hint = {}

    def _buf(name, n):
        # on a block the occupied side does not take, its Y/W buffer is
        # idle, and c0 (dead once the density kernel has run) and a
        # meta-GGA's aow (written after that) can live in it in turn
        if (oc is not None and (name == 'c0' or (name == 'aow' and mgga))
                and oc['y_buf'].size >= n):
            return oc['y_buf']
        b = _lazy.get(name)
        if b is None or b.size < n:
            b = _lazy[name] = cupy.empty(max(n, _hint.get(name, 0)))
        return b
    # the MO path evaluates the two channels into separate arrays and then
    # interleaves them, which is one copy of this size per block
    morho_buf = (cupy.empty(nsk * ncomp * blk) if mo1 is not None else None)
    rho_buf = cupy.empty(nsk * ncomp * blk)
    wv_buf = cupy.empty(nsk * ncomp * blk)
    # the masked densities go in a buffer too: `docs/README_fastxchess.md`
    # 3b measured per-block allocation of an array this shape at 48 % of that
    # function, and this one is the same shape times `nsk`
    # W16's second route to c0, for a density matrix that arrives with its
    # factors: D = M O^T + O M^T, so D . ao_0 = M (O^T ao_0) + O (M^T ao_0),
    # 3 nocc nsub flops per grid point per set against the matrix route's
    # nsub^2.  Two of its three GEMMs have an inner dimension of nocc and run
    # well below the matrix route's rate, so it is taken block by block where
    # nsub > FASTXCFXC_FACTOR_RATIO * nocc (4, measured -- at 3, the flop
    # break-even, it loses on def2-SVP), and only for LDA and GGA, whose
    # density needs D . ao_0 alone.
    fac = None
    if (factors is not None and nspin == 1 and mo1 is None and nc0 == 1
            and os.environ.get('FASTXCFXC_FACTORS', '1') != '0'):
        m_all = opt.sort_orbitals(cupy.asarray(factors[0]), axis=[1])
        o_all = opt.sort_orbitals(cupy.asarray(factors[1]), axis=[0])
        nocc_f = o_all.shape[1]
        ratio = float(os.environ.get('FASTXCFXC_FACTOR_RATIO', '4'))
        if m_all.shape[0] == nset and ratio * nocc_f < nsub_max:
            fac = (cupy.ascontiguousarray(m_all), cupy.ascontiguousarray(o_all),
                   nocc_f, cupy.empty(nset * nsub_max * nocc_f),
                   cupy.empty(nsub_max * nocc_f),
                   cupy.empty(nset * nocc_f * blk), cupy.empty(nocc_f * blk),
                   ratio)
        m_all = o_all = None
    oc = None
    if occ is not None and nspin == 1 and factors is not None:
        # D = M (2C)^T + h.c.: the kernels' factor 4 is for exactly this
        oc = _occ_setup(xctype, nset, nsub_max, blk, [
            opt.sort_orbitals(cupy.asarray(factors[0]), axis=[1])],
            [opt.sort_orbitals(cupy.asarray(occ), axis=[0])])
    elif occ is not None and nspin == 2 and mo1 is not None:
        # D_s = M_s C_s^T + h.c. (`hessian/uhf.py:gen_vind`), half of the
        # restricted form's, which halving M puts back
        oc = _occ_setup(xctype, nset, nsub_max, blk,
                        [.5 * m for m in mo1], occ_coeff)
    # the largest block each AO-matrix-route buffer serves
    n_rho = n_pot = nsub_max
    if oc is not None:
        lim_rho = oc['ratio'] * oc['nocc'] if oc['yrho'] else np.inf
        lim_pot = oc['ratio_pot'] * oc['nocc']
        n_rho = max([n for n in nsubs if n <= lim_rho], default=1)
        n_pot = max([n for n in nsubs if n <= lim_pot], default=1)
    _hint['c0'] = nc0 * nsk * n_rho * blk
    _hint['dm'] = nsk * n_rho * n_rho
    _hint['v'] = nsk * n_pot * n_pot
    _hint['aow'] = nsk * (n_pot if mgga else nsub_max) * blk
    if TIME:
        cupy.cuda.Stream.null.synchronize()
        _T['prep'] += _time.perf_counter() - _t
        _t = _time.perf_counter()

    # `strict_grid_order` keeps the empty blocks in the stream, so p0/p1 stay
    # the block's real grid offset -- which is what `fxc` is sliced by.
    p0 = p1 = 0
    for ao, idx, w, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv,
                                       max_memory=None,
                                       strict_grid_order=True):
        ng = w.size
        p0, p1 = p1, p1 + ng
        nsub = len(idx)
        if nsub == 0:
            continue
        ao0 = ao if lda else ao[0]

        # W33: this block on the occupied side?  Both halves, or for a
        # restricted GGA by default the potential only (its density is W16's
        # factor route, at nocc cost already)
        ob_rho = ob = False
        if oc is not None and ao.flags.c_contiguous:
            ob_rho = oc['yrho'] and oc['ratio'] * oc['nocc'] < nsub
            ob = oc['ratio_pot'] * oc['nocc'] < nsub
        if ob or ob_rho:
            _S['occ_blocks'] = _S.get('occ_blocks', 0) + 1
            for sp in range(nspin):
                _occ_x(oc, sp, idx, ao, nsub, ng)
        # ---- rho1 = (D . ao_0) . ao, one GEMM for every set --------------
        if ob_rho:
            pass                    # below, one spin at a time
        elif (fac is not None and fac[7] * fac[2] < nsub
                and ao0.flags.c_contiguous):
            c0 = ndarray((nc0, nsk * nsub, ng), buffer=_buf('c0', nc0 * nsk * nsub * ng))
            _c0_from_factors(fac, idx, ao0, c0[0], nset, nsub, ng)
            _S['factor_blocks'] = _S.get('factor_blocks', 0) + 1
        elif mo1 is None:
            dm_mask = take_last2d(dms, idx, out=_buf('dm', nsk * nsub * nsub))
            dm2 = dm_mask.reshape(nsk * nsub, nsub)
            c0 = ndarray((nc0, nsk * nsub, ng), buffer=_buf('c0', nc0 * nsk * nsub * ng))
            for n in range(nc0):
                cupy.dot(dm2, ao0 if lda else ao[n], out=c0[n])
            dm_mask = dm2 = None
        if TIME:
            cupy.cuda.Stream.null.synchronize()
            _T['gemm1'] += _time.perf_counter() - _t
            _t = _time.perf_counter()

        rho1 = ndarray((nsk, ncomp, ng), buffer=rho_buf)
        grid = ((ng + _WX - 1) // _WX, (nsk + _TSY - 1) // _TSY)
        grid_wv = ((ng + _WX - 1) // _WX, (nset + _TSY - 1) // _TSY)
        if ob_rho:
            for sp in range(nspin):
                Y = _occ_y(oc, sp, idx, ao, nsub, ng)
                if TIME:            # the Y GEMMs are the density's GEMMs
                    cupy.cuda.Stream.null.synchronize()
                    _T['gemm1'] += _time.perf_counter() - _t
                    _t = _time.perf_counter()
                _kernel(f'fxc_rho_occ_{suffix}')(
                    grid_wv, (_WX, _TSY),
                    (rho1.reshape(nset, nspin, ncomp, ng)[:, sp], Y,
                     oc['x'][sp], np.int32(nset), np.int32(oc['nocc']),
                     np.int32(ng), np.int32(nspin * ncomp)))
            Y = None
        elif mo1 is not None:
            # GPU4PySCF's own MO-basis density, one spin at a time, then one
            # copy into the interleaved layout the rest of this function
            # assumes.  Batching `eval_rho4` itself would be a third density
            # kernel; it is not done, and 4b measures what is left on the
            # table by not doing it.
            tmp = ndarray((nspin, nset, ncomp, ng), buffer=morho_buf)
            for sp in range(nspin):
                NI.eval_rho4(_sorted_mol, ao, occ_coeff[sp][idx],
                             mo1[sp][:, idx], xctype=xctype, hermi=hermi,
                             out=tmp[sp])
            rho1.reshape(nset, nspin, ncomp, ng)[:] = \
                tmp.transpose(1, 0, 2, 3)
            tmp = None
        elif kernels:
            _kernel(f'fxc_rho1_{suffix}')(
                grid, (_WX, _TSY),
                (rho1, c0, ao, np.int32(nsk), np.int32(nsub), np.int32(ng)))
        else:
            c0v = c0.reshape(nc0, nsk, nsub, ng)
            for i in range(nsk):
                if lda:
                    _contract_rho(c0v[0, i], ao0, rho=rho1[i, 0])
                else:
                    _contract_rho1(ao, c0v[0, i], rho=rho1[i][:4])
                if mgga:
                    tau = rho1[i, 4]
                    tau[:] = 0
                    for x in (1, 2, 3):
                        tau += _contract_rho(c0v[x, i], ao[x])
                    tau *= .5
            if not lda:
                rho1[:, 1:4] *= 2
        if TIME:
            cupy.cuda.Stream.null.synchronize()
            _T['rho1'] += _time.perf_counter() - _t
            _t = _time.perf_counter()

        # ---- the fxc contraction ----------------------------------------
        # a meta-GGA block on the occupied side never weights the AOs
        aow = (None if ob and mgga else
               ndarray((nsk * nsub, ng), buffer=_buf('aow', nsk * nsub * ng)))
        nz = (nsub + _UPB - 1) // _UPB
        if kernels:
            wv = ndarray((nsk, ncomp, ng), buffer=wv_buf)
            _kernel(f'fxc_wv_{"u_" if nspin == 2 else ""}{suffix}')(
                grid_wv, (_WX, _TSY),
                (wv, rho1, fxc_k[:, :, p0:p1], w, np.int32(nset),
                 np.int32(ng), np.int32(ngrids_tot), np.float64(scale0),
                 np.float64(scale4)))
        elif nspin == 1:
            wv = _contract_rho1_fxc(rho1, fxc_k[:, :, p0:p1] * w)
            if not lda:
                wv[:, 0] *= scale0
            if mgga:
                wv[:, 4] *= scale4
        else:
            # GPU4PySCF's own helper takes the (2, ncomp) pair index, so the
            # ablation gets the interleaved rho1 back as (nset, 2, ncomp, g)
            wv = _contract_rho1_fxc(rho1.reshape(nset, 2, ncomp, ng),
                                    fxc[:, :, :, :, p0:p1] * w)
            if not lda:
                wv[:, :, 0] *= scale0
            if mgga:
                wv[:, :, 4] *= scale4
            wv = wv.reshape(nsk, ncomp, ng)
        if TIME:
            cupy.cuda.Stream.null.synchronize()
            _T['wv'] += _time.perf_counter() - _t
            _t = _time.perf_counter()

        # ---- the weighted AOs and the potential matrix -------------------
        # aow_i . ao_0^T is the transpose of GPU4PySCF's ao_0 . aow_i^T.  For
        # GGA and meta-GGA the final transpose_sum absorbs it; for LDA the
        # block matrix is symmetric, since aow = ao_0 * wv there, and the tau
        # term of meta-GGA is symmetric on its own.
        if ob:
            for sp in range(nspin):
                _occ_potential(oc, sp, nspin, idx, ao, wv, aow, mgga, nset,
                               nsub, ng, grid_wv, nz)
            if TIME:
                cupy.cuda.Stream.null.synchronize()
                _T['gemm2'] += _time.perf_counter() - _t
                _t = _time.perf_counter()
            continue
        vblk = ndarray((nsk * nsub, nsub), buffer=_buf('v', nsk * nsub * nsub))

        def _weight(nw, w0, a0):
            # aow[i][v][g] = sum_n wv[i][w0+n][g] * ao[a0+n][v][g]
            if kernels:
                _kernel('fxc_scale_lda' if nw == 1 else 'fxc_scale_gga')(
                    (grid[0], grid[1], nz), (_WX, _TSY),
                    (aow, wv, ao, np.int32(nsk), np.int32(nsub),
                     np.int32(ng), np.int32(_UPB), np.int32(ncomp),
                     np.int32(w0), np.int32(a0)))
                return
            aowv = aow.reshape(nsk, nsub, ng)
            src = (ao0 if a0 == 0 else ao[a0]) if nw == 1 else ao[:nw]
            for i in range(nsk):
                _scale_ao(src, wv[i, w0] if nw == 1 else wv[i, w0:w0 + nw],
                          out=aowv[i])

        _weight(1 if lda else 4, 0, 0)
        if TIME:
            cupy.cuda.Stream.null.synchronize()
            _T['scale'] += _time.perf_counter() - _t
            _t = _time.perf_counter()
        cupy.dot(aow, ao0.T, out=vblk)
        if mgga:
            # + sum_{x=1..3} (ao[x] * wv[4]) . ao[x]^T, accumulated in place
            for x in (1, 2, 3):
                _weight(1, 4, x)
                contract('ig,jg->ij', aow, ao[x], beta=1., out=vblk)
        if TIME:
            cupy.cuda.Stream.null.synchronize()
            _T['gemm2'] += _time.perf_counter() - _t
            _t = _time.perf_counter()
        add_sparse(vmat, vblk.reshape(nsk, nsub, nsub), idx)
        if TIME:
            cupy.cuda.Stream.null.synchronize()
            _T['accum'] += _time.perf_counter() - _t
            _t = _time.perf_counter()

    if not lda:
        transpose_sum(vmat)
    if oc is not None:
        # the blocks taken on the matrix route, projected once at the end
        vmat = vmat.reshape(nset, nspin, nao, nao)
        out = []
        for sp in range(nspin):
            u = oc['u'][sp].reshape(nao, nset, oc['nocc']).transpose(1, 0, 2)
            u = contract('nij,jo->nio', vmat[:, sp], oc['c'][sp], beta=1.,
                         out=u.copy())
            out.append(opt.unsort_orbitals(u, axis=[1]))
        vmat = out[0] if nspin == 1 else out
        if TIME:
            cupy.cuda.Stream.null.synchronize()
            _T['n'] += 1
            print('[fastxcfxc] call %2d  ' % _T['n']
                  + '  '.join(f'{k} {_T[k]:6.3f}' for k in
                              ('prep', 'gemm1', 'rho1', 'wv', 'scale',
                               'gemm2', 'accum')), flush=True)
        return vmat
    vmat = opt.unsort_orbitals(vmat, axis=[1, 2])
    if nspin == 2:
        # un-interleave; GPU4PySCF returns the two channels stacked
        vmat = vmat.reshape(nset, 2, nao, nao)
        va = cupy.ascontiguousarray(vmat[:, 0])
        vb = cupy.ascontiguousarray(vmat[:, 1])
        if len(dm_shape) == 2:
            va, vb = va[0], vb[0]
        vmat = cupy.asarray([va, vb])
    elif len(dm_shape) == 2:
        vmat = vmat[0]
    if TIME:
        cupy.cuda.Stream.null.synchronize()
        _T['n'] += 1
        print('[fastxcfxc] call %2d  ' % _T['n']
              + '  '.join(f'{k} {_T[k]:6.3f}' for k in
                          ('prep', 'gemm1', 'rho1', 'wv', 'scale', 'gemm2',
                           'accum')), flush=True)
    log.timer_debug1(
        f'nr_{"u" if nspin == 2 else "r"}ks_fxc (fastxcfxc, blocked)', *t0)
    return cupy.asarray(vmat)


#: FASTXCFXC_FACTORS=0 keeps W16's call on the density-matrix route in every
#: block (the factor route is otherwise taken where FASTXCFXC_FACTOR_RATIO *
#: nocc < nsub); both are read at call time
_ONE = np.array(1.0)
_ZERO = np.array(0.0)


def _c0_from_factors(fac, idx, ao0, c0, nset, nsub, ng):
    """``c0[i] = D_i . ao_0`` for ``D_i = M_i O^T + O M_i^T``, on one block.

    Three cuBLAS calls, all row-major and expressed column-major below:
    ``Y_i = M_i^T ao_0`` and ``c0_i += O Y_i`` are strided batches over the
    sets with the shared operand at stride 0, and ``c0 = M (O^T ao_0)`` is a
    single GEMM on the flattened ``(i, u)`` row index.
    """
    from cupy_backends.cuda.libs import cublas
    m_all, o_all, nocc, m_buf, o_buf, y_buf, x_buf, _ = fac
    handle = cupy.cuda.device.get_cublas_handle()
    m = ndarray((nset, nsub, nocc), buffer=m_buf)
    cupy.take(m_all, idx, axis=1, out=m)
    o = ndarray((nsub, nocc), buffer=o_buf)
    cupy.take(o_all, idx, axis=0, out=o)
    x = ndarray((nocc, ng), buffer=x_buf)
    cupy.dot(o.T, ao0, out=x)
    y = ndarray((nset, nocc, ng), buffer=y_buf)
    n_ = cublas.CUBLAS_OP_N
    t_ = cublas.CUBLAS_OP_T
    # Y_i (nocc x ng) = M_i^T ao_0  <=>  Y_i^T = ao_0^T M_i  (column-major)
    cublas.dgemmStridedBatched(
        handle, n_, t_, ng, nocc, nsub, _ONE.ctypes.data,
        ao0.data.ptr, ng, 0, m.data.ptr, nocc, nsub * nocc,
        _ZERO.ctypes.data, y.data.ptr, ng, nocc * ng, nset)
    # c0 (nset nsub x ng) = M X
    cupy.dot(m.reshape(nset * nsub, nocc), x, out=c0)
    # c0_i += O Y_i  <=>  c0_i^T += Y_i^T O^T  (column-major)
    cublas.dgemmStridedBatched(
        handle, n_, n_, ng, nsub, nocc, _ONE.ctypes.data,
        y.data.ptr, ng, nocc * ng, o.data.ptr, nocc, 0,
        _ONE.ctypes.data, c0.data.ptr, ng, nsub * ng, nset)


#: W33.  FASTXCFXC_OCC=0 keeps every block of a Hessian's CPHF call on the
#: AO-matrix route (the ablation); otherwise a block goes to the occupied side
#: where FASTXCFXC_OCC_RATIO * nocc < nsub (meta-GGA, and every functional on
#: an unrestricted reference) or FASTXCFXC_OCC_RATIO_GGA * nocc < nsub (a
#: restricted GGA).  FASTXCFXC_OCC_RHO_GGA=1 takes a restricted GGA's density
#: on the occupied side too rather than by W16's factor route.  All read at
#: call time.
def _occ_setup(xctype, nset, nsub_max, blk, ms, cs):
    """Everything the occupied-side route needs, per spin; ``ms`` and ``cs``
    are the sorted factors, ``D_s = M_s (2 C_s)^T + h.c.``."""
    if os.environ.get('FASTXCFXC_OCC', '1') == '0':
        return None
    nspin = len(cs)
    if xctype not in ('GGA', 'MGGA'):
        return None
    # the density half: meta-GGA and the unrestricted case (whose matrix
    # route evaluates it with upstream's `eval_rho4`) at `ratio`; the
    # potential half at `ratio` for a meta-GGA, where it is four GEMMs a
    # side, and at the GGA ratio for a GGA, where the occupied side's two
    # nocc-wide GEMMs meet one nsub-wide one
    ratio = float(os.environ.get('FASTXCFXC_OCC_RATIO', '1'))
    ratio_pot = (ratio if xctype == 'MGGA' else
                 float(os.environ.get('FASTXCFXC_OCC_RATIO_GGA', '3')))
    yrho = (xctype == 'MGGA' or nspin == 2
            or os.environ.get('FASTXCFXC_OCC_RHO_GGA', '0') == '1')
    if xctype == 'GGA' and nspin == 1 and yrho:
        ratio = ratio_pot
    nao, nocc = cs[0].shape
    if not min(ratio if yrho else ratio_pot, ratio_pot) * nocc < nsub_max:
        return None
    if any(m.shape != (nset, nao, nocc) for m in ms):
        return None
    ny = 4 if yrho else 1
    return {
        'ratio': ratio, 'ratio_pot': ratio_pot, 'nocc': nocc, 'nset': nset,
        'yrho': yrho,
        'c': [cupy.ascontiguousarray(c) for c in cs],
        'ct': [cupy.ascontiguousarray(c.T) for c in cs],
        # M^T, so that Y = M^T ao is one GEMM over (set, k) rows
        'mt': ([cupy.ascontiguousarray(m.transpose(0, 2, 1)) for m in ms]
               if yrho else None),
        'mt_buf': cupy.empty(nset * nocc * nsub_max) if yrho else None,
        'ct_buf': cupy.empty(nocc * nsub_max),
        'x_buf': [cupy.empty(4 * nocc * blk) for _ in range(nspin)],
        'x': [None] * nspin,
        'y_buf': cupy.empty(ny * nset * nocc * blk),
        'u_buf': cupy.empty(nsub_max * nset * nocc),
        # V_s C_s accumulated as (nao, nset*nocc): the output of
        # ao[n] . W[n]^T, rows scattered to the block's AOs
        'u': [cupy.zeros((nao, nset * nocc)) for _ in range(nspin)]}


def _occ_x(oc, sp, idx, ao, nsub, ng):
    """X[n] = C^T ao[n] on one block, (4, nocc, ng), for spin ``sp``."""
    nocc = oc['nocc']
    ct = ndarray((nocc, nsub), buffer=oc['ct_buf'])
    cupy.take(oc['ct'][sp], idx, axis=1, out=ct)
    X = ndarray((4, nocc, ng), buffer=oc['x_buf'][sp])
    for n in range(4):
        cupy.dot(ct, ao[n], out=X[n])
    oc['x'][sp] = X
    return X


def _occ_y(oc, sp, idx, ao, nsub, ng):
    """Y[n] = M^T ao[n] for every set, (4, nset, nocc, ng)."""
    nocc, nset = oc['nocc'], oc['nset']
    mt = ndarray((nset, nocc, nsub), buffer=oc['mt_buf'])
    cupy.take(oc['mt'][sp], idx, axis=2, out=mt)
    mt2 = mt.reshape(nset * nocc, nsub)
    Y = ndarray((4, nset * nocc, ng), buffer=oc['y_buf'])
    for n in range(4):
        cupy.dot(mt2, ao[n], out=Y[n])
    return Y


def _gemm_nt(m, n, k, a, lda, b, ldb, c, ldc, beta,
             batch=1, sa=0, sb=0, sc=0):
    """Row-major ``c[m, n] (+)= a[m, k] . b[n, k]^T``, optionally batched."""
    from cupy_backends.cuda.libs import cublas
    handle = cupy.cuda.device.get_cublas_handle()
    bt = _ONE if beta else _ZERO
    # column-major: c^T (n x m) = b^T-as-stored (op T) . a-as-stored (op N)
    if batch == 1:
        cublas.dgemm(handle, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N,
                     n, m, k, _ONE.ctypes.data, b, ldb, a, lda,
                     bt.ctypes.data, c, ldc)
    else:
        cublas.dgemmStridedBatched(
            handle, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N,
            n, m, k, _ONE.ctypes.data, b, ldb, sb, a, lda, sa,
            bt.ctypes.data, c, ldc, sc, batch)


def _occ_potential(oc, sp, nspin, idx, ao, wv, aow, mgga, nset, nsub, ng,
                   grid, nz):
    """(V_s C_s)[idx] += this block's contribution; see ``fxc_w_occ_*``.

    ``wv`` is ``(nset, nspin, ncomp, ng)`` flattened; spin ``sp``'s rows
    start ``sp * ncomp`` rows in and recur every ``nspin * ncomp``."""
    nocc = oc['nocc']
    nc = 5 if mgga else 4
    X = oc['x'][sp]
    W = ndarray((4 if mgga else 1, nset * nocc, ng), buffer=oc['y_buf'])
    wv_s = wv.reshape(nset, nspin, nc, ng)[:, sp]
    _kernel('fxc_w_occ_mgga' if mgga else 'fxc_w_occ_gga')(
        grid, (_WX, _TSY),
        (W, wv_s, X, np.int32(nset), np.int32(nocc), np.int32(ng),
         np.int32(nspin * nc)))
    ub = ndarray((nsub, nset * nocc), buffer=oc['u_buf'])
    nw = nset * nocc
    for n in range(4 if mgga else 1):
        _gemm_nt(nsub, nw, ng, ao[n].data.ptr, ng, W[n].data.ptr, ng,
                 ub.data.ptr, nw, n > 0)
    if not mgga:
        # + aow_i . X0^T, written into set i's nocc columns of ub
        aow_s = aow[:nset * nsub]
        _kernel('fxc_scale_gga')(
            (grid[0], grid[1], nz), (_WX, _TSY),
            (aow_s, wv, ao, np.int32(nset), np.int32(nsub), np.int32(ng),
             np.int32(_UPB), np.int32(nspin * 4), np.int32(sp * 4),
             np.int32(0)))
        _gemm_nt(nsub, nocc, ng, aow_s.data.ptr, ng, X.data.ptr, ng,
                 ub.data.ptr, nw, True, batch=nset, sa=nsub * ng, sb=0,
                 sc=nocc)
    oc['u'][sp][idx] += ub


def _nr_rks_fxc_blocked(*a, **kw):
    return _fxc_blocked(*a, nspin=1, **kw)


def _nr_uks_fxc_blocked(*a, **kw):
    return _fxc_blocked(*a, nspin=2, **kw)


def _chunks(nblocks, nset, nao_sub_max, blksize):
    per_block = max(1, nset * nao_sub_max * blksize * 8)
    n = int(CHUNK_BYTES // per_block)
    return max(1, min(nblocks, n))


def _nr_rks_fxc_grouped(ni, mol, grids, xc_code, dm0, dms, relativity, hermi,
                        rho0, vxc, fxc, max_memory, verbose):
    """Catalogue B4d/G1 as written: ``grouped_gemm`` over (block, set).

    Kept because it is the prescription and it is slower than
    ``_nr_rks_fxc_blocked`` -- see the module docstring.  LDA and GGA only:
    it exists to A/B the issue order, and meta-GGA was written straight into
    the blocked path.
    """
    if ni._xc_type(xc_code) == 'MGGA':
        return NI._nr_rks_fxc_orig(ni, mol, grids, xc_code, dm0, dms,
                                   relativity, hermi, rho0, vxc, fxc,
                                   max_memory, verbose)
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    xctype = ni._xc_type(xc_code)
    opt, dms, nset, dm_shape, _ = _prep(ni, mol, grids, dms, hermi)
    _sorted_mol = opt._sorted_mol

    lda = xctype == 'LDA'
    ncomp = 1 if lda else 4
    ao_deriv = 0 if lda else 1
    nao = _sorted_mol.nao
    vmat = cupy.zeros((nset, nao, nao))

    p1 = 0
    for ao_group, idx_group, w_group, _ in NI._grouped_block_loop(
            ni, _sorted_mol, grids, nao, ao_deriv):
        nb = len(ao_group)
        # `_grouped_block_loop` walks every block in grid order and skips
        # none, so the running offset is the block's own grid offset.
        p0s = []
        for w in w_group:
            p0s.append(p1)
            p1 += w.size
        ao0 = [a if lda else a[0] for a in ao_group]
        active = [b for b in range(nb) if len(idx_group[b]) > 0]
        if not active:
            continue
        nsub_max = max(len(idx_group[b]) for b in active)
        blksize = max(w_group[b].size for b in active)
        step = _chunks(len(active), nset, nsub_max, blksize)

        for c0b in range(0, len(active), step):
            sel = active[c0b:c0b + step]
            masks = [take_last2d(dms, idx_group[b]) for b in sel]
            As = [m[i] for m in masks for i in range(nset)]
            Bs = [ao0[b] for b in sel for _ in range(nset)]
            c0_group = grouped_gemm(As, Bs)
            masks = As = Bs = None

            rho1_group = []
            for k, b in enumerate(sel):
                ng = w_group[b].size
                rho1 = cupy.empty((nset, ncomp, ng))
                for i in range(nset):
                    c0 = c0_group[k * nset + i]
                    if lda:
                        _contract_rho(c0, ao0[b], rho=rho1[i, 0])
                    else:
                        _contract_rho1(ao_group[b], c0, rho=rho1[i])
                if not lda:
                    rho1[:, 1:4] *= 2
                rho1_group.append(rho1)
            c0_group = None

            aow_group = []
            for k, b in enumerate(sel):
                ng = w_group[b].size
                q0 = p0s[b]
                wvs = _contract_rho1_fxc(rho1_group[k],
                                         fxc[:, :, q0:q0 + ng] * w_group[b])
                if not lda:
                    wvs[:, 0] *= .5
                nsub = len(idx_group[b])
                aow = cupy.empty((nset, nsub, ng))
                for i in range(nset):
                    _scale_ao(ao_group[b], wvs[i, 0] if lda else wvs[i],
                              out=aow[i])
                aow_group.append(aow)
            rho1_group = None

            vblk = [cupy.empty((nset, len(idx_group[b]), len(idx_group[b])))
                    for b in sel]
            Cs = [v[i] for v in vblk for i in range(nset)]
            As = [ao0[b] for b in sel for _ in range(nset)]
            Bs = [a[i] for a in aow_group for i in range(nset)]
            grouped_dot(As, Bs, Cs)
            aow_group = As = Bs = Cs = None
            for k, b in enumerate(sel):
                add_sparse(vmat, vblk[k], idx_group[b])
            vblk = None

    if not lda:
        transpose_sum(vmat)
    vmat = opt.unsort_orbitals(vmat, axis=[1, 2])
    if len(dm_shape) == 2:
        vmat = vmat[0]
    log.timer_debug1('nr_rks_fxc (fastxcfxc, grouped)', *t0)
    return cupy.asarray(vmat)


def _nr_rks_fxc_nokernel(*args, **kwargs):
    return _nr_rks_fxc_blocked(*args, kernels=False, **kwargs)


def _nr_uks_fxc_nokernel(*args, **kwargs):
    return _nr_uks_fxc_blocked(*args, kernels=False, **kwargs)


_MODES = {'blocked': _nr_rks_fxc_blocked,
          'nokernel': _nr_rks_fxc_nokernel,
          'grouped': _nr_rks_fxc_grouped}

#: no 'grouped' entry: that mode exists to A/B the catalogue's issue order
#: against the shipped one, it lost, and an unrestricted copy of a losing
#: ablation would be two more paths to keep correct for no measurement.
#: ``FASTXCFXC_MODE=grouped`` therefore hands the unrestricted call back.
_MODES_U = {'blocked': _nr_uks_fxc_blocked,
            'nokernel': _nr_uks_fxc_nokernel}

_PATCHED = False


def _nr_uks_fxc_dispatch(ni, mol, grids, xc_code, dm0=None, dms=None,
                         relativity=0, hermi=0, rho0=None, vxc=None, fxc=None,
                         max_memory=2000, verbose=None):
    if fxc is None:
        raise RuntimeError('fxc was not initialized')
    fn = _MODES_U.get(MODE)
    if fn is not None and _usable_u(ni, xc_code, dms, hermi, fxc):
        _S['ours_u_mo' if hasattr(dms, 'mo1') else 'ours_u'] += 1
        return fn(ni, mol, grids, xc_code, dm0, dms, relativity, hermi,
                  rho0, vxc, fxc, max_memory, verbose)
    _S['stock_u'] += 1
    return NI._nr_uks_fxc_orig(ni, mol, grids, xc_code, dm0, dms, relativity,
                               hermi, rho0, vxc, fxc, max_memory, verbose)


def _nr_rks_fxc_dispatch(ni, mol, grids, xc_code, dm0=None, dms=None,
                         relativity=0, hermi=0, rho0=None, vxc=None, fxc=None,
                         max_memory=2000, verbose=None):
    if fxc is None:
        raise RuntimeError('fxc was not initialized')
    fn = _MODES.get(MODE)
    if fn is not None and _usable(ni, xc_code, dms, hermi, fxc):
        return fn(ni, mol, grids, xc_code, dm0, dms, relativity, hermi,
                  rho0, vxc, fxc, max_memory, verbose)
    return NI._nr_rks_fxc_orig(ni, mol, grids, xc_code, dm0, dms, relativity,
                               hermi, rho0, vxc, fxc, max_memory, verbose)


def _nr_rks_fxc_mo(ni, mol, grids, xc_code, dm0=None, dms=None,
                   mo_coeff=None, relativity=0, hermi=0, rho0=None, vxc=None,
                   fxc=None, max_memory=2000, verbose=None):
    """``hessian/rks.py:nr_rks_fxc_mo`` -- a restricted Hessian's CPHF grid half.

    `docs/DF_WORK_LIST.md` **W16**.  Upstream's function is ``nr_rks_fxc``'s
    ``mo1`` branch with the MO transformation folded onto the end: per grid
    block it evaluates the perturbed density from the orbitals
    (``eval_rho4``, a ``contract`` per right-hand side), contracts ``fxc``,
    and builds the potential with one ``ao . aow^T`` per right-hand side --
    ``2 nset`` small cuBLAS calls and ``2 nset`` elementwise kernels per
    block, which is what makes it launch-bound (`docs/README_fastxcfxc.md`
    has the per-set loop; `docs/README_fastdfhessip.md` 2 the profile).

    The perturbed density a CPHF cycle hands it is *also* a density matrix:
    ``hessian/rhf.py:gen_vind`` builds ``dm1 = mo1 (2 C_o)^T + h.c.`` and
    tags it with ``mo1`` and ``occ_coeff``.  So the whole function is the
    blocked ``nr_rks_fxc`` of this module on the plain matrix, with
    ``hermi = 1`` because ``dm1`` is symmetric by construction, followed by
    upstream's own ``_ao2mo`` -- a binding and no new kernel.
    """
    if fxc is None:
        raise RuntimeError('fxc was not initialized')
    fn = _MODES.get(MODE) if MODE != 'grouped' else None
    if os.environ.get('FASTXCFXC_MO', '1') == '0':
        fn = None
    plain = None
    if fn is not None and hasattr(dms, 'occ_coeff'):
        plain = cupy.asarray(dms).view(cupy.ndarray)
        if not _usable(ni, xc_code, plain, 1, fxc):
            plain = None
    if plain is None:
        _S['stock_mo'] += 1
        from gpu4pyscf.hessian import rks as rks_hess
        return rks_hess._nr_rks_fxc_mo_orig(ni, mol, grids, xc_code, dm0, dms,
                                            mo_coeff, relativity, hermi, rho0,
                                            vxc, fxc, max_memory, verbose)
    _S['ours_mo'] += 1
    from gpu4pyscf.hessian.rhf import _ao2mo
    nao = plain.shape[-1]
    kw = {}
    if fn is _nr_rks_fxc_blocked and hasattr(dms, 'mo1'):
        # the factors of `dm1 = mo1 (2 C_o)^T + h.c.` (`gen_vind`)
        kw['factors'] = (dms.mo1, 2 * cupy.asarray(dms.occ_coeff))
        # W33: and since `_ao2mo` wants only V C_o, ask for that
        if os.environ.get('FASTXCFXC_OCC', '1') != '0':
            kw['occ'] = dms.occ_coeff
    v = fn(ni, mol, grids, xc_code, None, plain.reshape(-1, nao, nao), 0, 1,
           rho0, vxc, fxc, max_memory, verbose, **kw)
    if v.shape[-1] == nao:
        v = _ao2mo(v, cupy.asarray(dms.occ_coeff), cupy.asarray(mo_coeff))
    else:
        v = contract('nio,ip->npo', v, cupy.asarray(mo_coeff))
    if plain.ndim == 2:
        v = v[0]
    return v


_nr_rks_fxc_mo_orig = None
_get_veff_resp_mo_u_orig = None


def _get_veff_resp_mo_u(hessobj, mol, dms, mo_coeff, mo_occ, hermi=1):
    """``hessian/uks.py:get_veff_resp_mo`` -- an unrestricted Hessian's CPHF.

    W33's unrestricted half.  Upstream calls ``nr_uks_fxc`` for the two AO
    potentials and takes each to the MO basis with ``_ao2mo`` straight
    after, which only ever wants ``V_s C_s``; so this asks ``_fxc_blocked``
    for that product, and the blocks where it is cheaper run on the occupied
    side (``_occ_setup``).  Everything else -- the NLC term, the Coulomb
    and exchange response -- is upstream's lines unchanged.  Anything the
    route does not cover is upstream's whole function.
    """
    from gpu4pyscf.hessian.rhf import _ao2mo
    from gpu4pyscf.hessian.uks import nr_uks_fnlc_mo
    mf = hessobj.base
    ni = mf._numint
    # by name, not by identity: a profiler that wraps `_nr_uks_fxc_blocked`
    # (benchmarks/df_hess_anatomy.py does) must not turn the route off
    if (MODE != 'blocked'
            or os.environ.get('FASTXCFXC_OCC', '1') == '0'
            or os.environ.get('FASTXCFXC_MO', '1') == '0'
            or not hasattr(dms, 'mo1')
            or ni._xc_type(mf.xc) not in ('GGA', 'MGGA')):
        return _get_veff_resp_mo_u_orig(hessobj, mol, dms, mo_coeff, mo_occ,
                                        hermi)
    mol = hessobj.mol
    grids = getattr(mf, 'cphf_grids', None)
    if grids is not None:
        logger.info(mf, 'Secondary grids defined for CPHF in Hessian')
    else:
        grids = getattr(mf, 'grids', None)
        logger.info(mf, 'Primary grids is used for CPHF in Hessian')
    if grids and grids.coords is None:
        grids.build(mol=mol, with_non0tab=False, sort_grids=True)

    nao, nmoa = mo_coeff[0].shape
    nao, nmob = mo_coeff[1].shape
    mocca = mo_coeff[0][:, mo_occ[0] > 0]
    moccb = mo_coeff[1][:, mo_occ[1] > 0]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    hermi = 1

    rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, mf.xc,
                                        mo_coeff, mo_occ, 1)
    nset = dms.shape[1]
    v1vo = cupy.empty([nset, nmoa * nocca + nmob * noccb])
    if not _usable_u(ni, mf.xc, dms, hermi, fxc):
        v1 = ni.nr_uks_fxc(mol, grids, mf.xc, None, dms, 0, hermi,
                           rho0, vxc, fxc, max_memory=None)
        vc = None
    else:
        _S['ours_u_mo'] += 1
        v1 = _nr_uks_fxc_blocked(ni, mol, grids, mf.xc, None, dms, 0, hermi,
                                 rho0, vxc, fxc, None, None, occ=True)
        # `occ_coeff` is padded to nocca columns for both spins
        vc = v1 if isinstance(v1, list) else None
    if vc is not None:
        _S['occ_u_calls'] = _S.get('occ_u_calls', 0) + 1
        v1vo[:, :nmoa * nocca] = contract(
            'nio,ip->npo', vc[0][:, :, :nocca], mo_coeff[0]
        ).reshape(-1, nmoa * nocca)
        v1vo[:, nmoa * nocca:] = contract(
            'nio,ip->npo', cupy.ascontiguousarray(vc[1][:, :, :noccb]),
            mo_coeff[1]).reshape(-1, nmob * noccb)
    else:
        v1vo[:, :nmoa * nocca] = _ao2mo(v1[0], mocca, mo_coeff[0]).reshape(
            -1, nmoa * nocca)
        v1vo[:, nmoa * nocca:] = _ao2mo(v1[1], moccb, mo_coeff[1]).reshape(
            -1, nmob * noccb)
    v1 = vc = None

    if mf.do_nlc():
        vnlca, vnlcb = nr_uks_fnlc_mo(mf, mol, mo_coeff, mo_occ, dms)
        v1vo[:, :nmoa * nocca] += vnlca.reshape(-1, nmoa * nocca)
        v1vo[:, nmoa * nocca:] += vnlcb.reshape(-1, nmob * noccb)

    if hybrid:
        vj, vk = hessobj.get_jk_mo(mol, dms, mo_coeff, mo_occ, hermi=1)
        vk *= hyb
        if omega > 1e-10:
            _, vk_lr = hessobj.get_jk_mo(mol, dms, mo_coeff, mo_occ,
                                         hermi, with_j=False, omega=omega)
            vk_lr *= (alpha - hyb)
            vk += vk_lr
        v1vo += vj - vk
    else:
        v1vo += hessobj.get_jk_mo(mol, dms, mo_coeff, mo_occ,
                                  hermi=1, with_k=False)[0]
    return v1vo


def apply_patch():
    global _PATCHED, _nr_rks_fxc_mo_orig, _get_veff_resp_mo_u_orig
    if not _PATCHED:
        NI._nr_rks_fxc_orig = NI.nr_rks_fxc
        # the module attribute is the one `nr_rks_fxc_st` (TDA/TDDFT) calls;
        # the class attribute is the one CPKS calls.  Both are needed.
        NI.nr_rks_fxc = _nr_rks_fxc_dispatch
        NI.NumInt.nr_rks_fxc = _nr_rks_fxc_dispatch
        # Every caller of the unrestricted one reaches it as `ni.nr_uks_fxc`
        # -- `scf/_response_functions.py:171` for CPKS, TDDFT, NMR and C6, and
        # `hessian/uks.py:1892` for an unrestricted Hessian's CPHF -- so the
        # class attribute is the one that matters here and the module
        # attribute is patched for symmetry with the restricted pair.  There
        # is no `nr_uks_fxc_st`: an open-shell excited state has no
        # singlet/triplet split to take.
        NI._nr_uks_fxc_orig = NI.nr_uks_fxc
        NI.nr_uks_fxc = _nr_uks_fxc_dispatch
        NI.NumInt.nr_uks_fxc = _nr_uks_fxc_dispatch
        # W16: a restricted Hessian's CPHF does not come through
        # `nr_rks_fxc` at all.  `hessian/rks.py:get_veff_resp_mo` looks this
        # one up on its own module, and nothing else calls it.
        from gpu4pyscf.hessian import rks as rks_hess
        _nr_rks_fxc_mo_orig = rks_hess.nr_rks_fxc_mo
        rks_hess._nr_rks_fxc_mo_orig = _nr_rks_fxc_mo_orig
        rks_hess.nr_rks_fxc_mo = _nr_rks_fxc_mo
        # W33: an unrestricted Hessian's CPHF, which `hessobj` reaches as a
        # method; the DF Hessian inherits it from this class
        from gpu4pyscf.hessian import uks as uks_hess
        _get_veff_resp_mo_u_orig = uks_hess.get_veff_resp_mo
        uks_hess._get_veff_resp_mo_orig = _get_veff_resp_mo_u_orig
        uks_hess.get_veff_resp_mo = _get_veff_resp_mo_u
        uks_hess.Hessian.get_veff_resp_mo = _get_veff_resp_mo_u
        _PATCHED = True


apply_patch()
