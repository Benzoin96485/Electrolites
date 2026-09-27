"""The XC part of an analytic Hessian (``hessian.{rks,uks}._get_vxc_deriv1/2``).

Catalogue entry **B5**: both functions run a Python ``for ia in range(natm)``
loop *inside* ``ni.block_loop``, and both expand the block's masked AO values
back to the full ``nao`` before they start (``ao1 = contract('nip,ij->njp',
ao_mask, coeff[mask])``), which voids the AO screening the mask just did.
Measured on B3LYP/def2-SVP (``benchmarks/vxc_hess_probe.py``), the atom loop is
**90 %** of ``_get_vxc_deriv1`` on penicillin V and **93 %** on taxol, while
the per-block ``eval_xc_eff`` that the catalogue's G1 would batch is **1.5 %**
-- so unlike the gradient (``fastxcgrad``), grouping the grid blocks is not
where the time is here.  The atom loop is, for two reasons at once:

1. **Every atom pays for every block.**  ``d rho / d R_A`` is a sum over atom
   A's own basis functions, so on a block where atom A has no basis function
   in the mask the whole iteration produces zeros -- exactly zeros, not small
   ones: an AO outside the mask *is* zero as far as GPU4PySCF is concerned.
   The iteration runs anyway, and in ``_get_vxc_deriv1`` it costs two
   ``(nao x ngrids) . (ngrids x nocc)`` GEMMs.  Skipping those atoms is a
   screen with no error term of its own.
2. **The GEMMs are at the full ``nao``.**  In the masked AO space they are
   ``nao_sub``, and the mask is what the AO evaluation already produced.

The two fractions, weighted by grid points (``benchmarks/xchess_locality.py``):

    | | penicillin V (42 atoms) | taxol (113 atoms) |
    |---|---|---|
    | ``nao_sub / nao``     | 0.498 | 0.328 |
    | ``natm_local / natm`` | 0.745 | 0.626 |
    | product               | 0.379 | 0.214 |

so the two screens together are worth 2.6x on penicillin V and 4.7x on taxol
before anything is batched -- and the ratio improves with system size, because
``natm_local`` is a property of how much space one grid block covers, not of
how many atoms the molecule has.

What is *not* worth doing, measured rather than assumed: sub-dividing the
4096-point block to sharpen the atom screen.  At 64-point chunks
``natm_local / natm`` only falls from 0.745 to 0.683 (taxol: 0.626 to 0.537),
because GPU4PySCF's AO mask cutoff of 1e-10 keeps a Gaussian tail alive long
past the point where it matters.  A per-chunk atom list and its gather would
buy 9 %; it is not implemented.

On top of the two screens the atom dimension is **batched**:

* ``_get_vxc_deriv2`` loses its atom loop entirely.  Both of its terms are
  ``sum_g W[..,j,g] wv2[A,s,y,g]`` with ``W`` built from the AO values and the
  density product ``D . ao``, neither of which depends on the atom.  Building
  ``W`` once per block and contracting the atom dimension in one cuTENSOR call
  replaces ~10 kernels per atom per block with two per block, and
  materialises nothing of size ``natm x nao_sub x ngrids``.
* ``_get_vxc_deriv1`` cannot avoid ``aow[A,s] = sum_y ao[y] wv2[A,s,y]``,
  because its output carries a free AO index *and* a free occupied index, so
  the two GEMMs per (atom, direction) are inherent.  They are stacked over a
  block of atoms and issued as one GEMM each instead, which is worth having
  for its own sake -- the batched GEMMs run at 12.8 TFLOP/s of fp64, close to
  what a DGEMM of that shape should get -- but is not where this function's
  time was.  See below.

And then the thing the batching creates: ``aow`` is
``natm_blk x 3 x nao_sub x ngrids``, 2.6 GB a block on taxol, and ``nao_sub``
changes from block to block, so **allocating it per block is 48 % of the
function** -- more than five times what the GEMM that consumes it costs.  All
the scratch here therefore comes from one buffer per name, sized once for the
largest ``nao_sub`` any block has (``_Bufs``), which is GPU4PySCF's own idiom
in these two functions and the same hazard ``fastxcgrad`` records for
``_grouped_block_loop``.  That change alone is 3.24 s -> 1.55 s on penicillin
V.

Everything here is the same arithmetic as GPU4PySCF's, reassociated: the sums
over grid points are reordered, so results agree to fp64 rounding rather than
bit for bit.

**Both spin cases.**  ``hessian/uks.py`` carries its own copies of these two
functions -- it does not import the restricted ones -- so patching
``hessian.rks`` alone left every open-shell Hessian on GPU4PySCF's code, which
is ``docs/OPENSHELL_TMC.md`` 4 item 2 and the largest single entry on that
list.  The unrestricted pair is the same four mechanisms with a spin index,
and they are *shared* rather than run twice: one AO evaluation, one mask, one
atom list and one set of AO-resolved ``d rho / d R`` products serve both
channels.  The one line that is genuinely not the restricted expression twice
is ``_wv2_u`` -- ``fxc`` couples the channels, so the perturbed potential of
spin ``b`` takes both spins' perturbed densities -- and it is also the reason
the two spins have to share a block rather than the module being called once
per spin.

**Meta-GGAs** (``docs/DF_WORK_LIST.md`` **W28**,
``docs/README_fastxchessmgga.md``).  Upstream's meta-GGA branches are the GGA
ones with a fifth density component (``tau``) and three more groups of terms,
on top of the same atom loop inside the same block loop, and measured on a
42-atom TPSS Hessian that loop is 92-95 % of both functions while
``eval_xc_eff`` -- whose third derivative is dearer for a meta-GGA -- is
1.2 %.  So the same four mechanisms carry over, and each tau term is folded
into an existing contraction rather than added as a loop of its own:

* ``d rho / d R`` gets its fifth row from the same second-derivative gather
  (``_dR_rho1``, ``nc = 5``);
* in ``_get_vxc_deriv1`` upstream's ``for j in range(1, 4)`` block is the
  tau term *and* its transpose, which are the same matrix; it is one GEMM of
  inner dimension ``4 ngrids`` together with the GGA ``mow . ao0^T``
  (``_deriv1_gemms``);
* in ``_get_vxc_deriv2`` the nine per-atom tau contractions are a fifth row
  of the atom-independent ``W[x]`` (``_deriv2_t``), and the tau part of
  ``ipip`` is one contraction per block (``_ipip_block``).

``FASTXCHESS_MGGA=0`` hands meta-GGAs back to GPU4PySCF.

**Scope.**  GGA and meta-GGA, so hybrids, B3LYP, TPSS, r2SCAN, M06-L,
M06-2X and wB97M-V's semilocal part; restricted and unrestricted.  LDA, the
grid-response path (``hessobj.grid_response``), the VV10 Hessian
(``_get_enlc_deriv2``), ``_get_vxc_diag`` and multi-GPU all fall through to
GPU4PySCF unchanged.  LDA is a separate branch upstream with a different
factor convention -- its ``d rho / d R`` carries no factor 2 where the GGA
branch's ``_make_dR_rho1`` does, and the GGA branch's compensating
``wv2[:,0] *= .5`` is what makes the two agree -- so covering it means
writing that branch too, which at 0.5 % of a Hessian is not where the next
hour goes.  ``docs/README_fastxchess.md`` §5 has the algebra, and
``tests/test_xchess_fd.py`` carries an LDA row so that the claim they agree
is checked rather than asserted.

**The meta-GGA fall-through was repaired before it was replaced.**
``hessian/rks.py`` contracts ``fxc`` in place in three places --
``wv2 = contract('xyg,sxg->syg', wf, dR_rho1, out=dR_rho1)`` in both the GGA
and the meta-GGA branch of ``_get_vxc_deriv1_task``, and ``contract('axg,
xyg->ayg', rho1, fxc_w, out=rho1)`` in ``_nr_rks_fxc_mo_task`` -- and cuTENSOR
does not promise an in-place contraction is correct.  At four density
components it is; at five, on a block of more than a few hundred grid points,
it is off by a third of the largest element, and which elements depends on the
run.  That is why a restricted meta-GGA Hessian was not reproducible
(``docs/DF_WORK_LIST.md`` W23, ``docs/README_mgga_repro.md``).  The module
replaces the ``contract`` that ``hessian/rks.py`` looked up at import with
``_alias_safe_contract``, which is the same call unless ``out`` overlaps an
operand, and then contracts into a temporary and copies it into ``out``.  It
is on whenever the module is loaded, including under ``FASTXCHESS_OFF=1``;
``FASTXCHESS_INPLACE=1`` puts the unsafe call back.

Environment switches (for the ablations):
``FASTXCHESS_OFF=1``         hand both functions back to GPU4PySCF
``FASTXCHESS_INPLACE=1``     keep ``hessian/rks.py``'s in-place contractions
``FASTXCHESS_MGGA=0``        hand meta-GGAs back to GPU4PySCF (W28's baseline)
``FASTXCHESS_SCALESUM=0``    form ``aow``/``mow`` with cuTENSOR, as before W28
``FASTXCHESS_ATOMSCREEN=0``  keep every atom on every block, i.e. mechanism 2
                             alone, so the two screens can be separated
``FASTXCHESS_MEM``           fraction of free memory one atom block may take
                             (default 0.2)
"""
import os

import numpy as np
import cupy
import cupyx
from pyscf.gto.mole import ATOM_OF

from gpu4pyscf.dft import numint
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.hessian import rks as hess_rks
from gpu4pyscf.hessian import uks as hess_uks
from gpu4pyscf.lib.cupy_helper import (
    contract, take_last2d, add_sparse, get_avail_mem)
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import num_devices

_OFF = bool(os.environ.get('FASTXCHESS_OFF'))
_INPLACE = bool(os.environ.get('FASTXCHESS_INPLACE'))
_ATOMSCREEN = os.environ.get('FASTXCHESS_ATOMSCREEN', '1') != '0'
_MGGA = os.environ.get('FASTXCHESS_MGGA', '1') != '0'
_SCALESUM = os.environ.get('FASTXCHESS_SCALESUM', '1') != '0'
MEM_FRACTION = float(os.environ.get('FASTXCHESS_MEM', '0.2'))
GB = 1024 ** 3

#: ``ao[_H[s,t]]`` is d^2/ds dt of the AO.  The packed second-derivative order
#: is XX XY XZ YY YZ ZZ = 4 5 6 7 8 9.
_H = np.array([[4, 5, 6], [5, 7, 8], [6, 8, 9]])
#: ``ao[_SEL[x,y]]`` are the components ``_make_dR_dao_w`` combines: the first
#: derivative for y = 0 and the second derivatives for y = 1,2,3.
_SEL = np.concatenate([np.array([[1], [2], [3]]), _H], axis=1)
#: the same two on the device, keyed by device id -- a cupy index array
#: belongs to the device that made it, and `fast1e` hit this the hard way
_DEV_IDX = {}


class _Layout:
    """The sorted AO basis of one ``gdftopt``, with the atom of every AO.

    A grid block yields AO values in GPU4PySCF's *sorted* AO order, and
    ``opt.coeff`` maps them to the original order
    (``ao_original = coeff.T . ao_sorted``).  Every contraction here stays in
    the sorted order instead: a density matrix sandwiched between two AO
    factors becomes ``coeff . D . coeff.T``, which is what
    ``opt.sort_orbitals`` produces, and an atom-restricted sum over AOs stays
    an atom-restricted sum because ``coeff`` is block diagonal in the shell.
    ``supported`` is False when the sorted basis is not a permutation of the
    original one, in which case the second statement still holds but
    ``sort_orbitals`` is not the transform and this module declines.
    """

    def __init__(self, opt, mol):
        self.opt = opt
        self._sorted_mol = opt._sorted_mol
        self.nao_s = opt.coeff.shape[0]
        self.supported = (self.nao_s == mol.nao)
        dims = self._sorted_mol.ao_loc[1:] - self._sorted_mol.ao_loc[:-1]
        self.atom_of_ao = np.repeat(self._sorted_mol._bas[:, ATOM_OF], dims)
        self.natm = mol.natm
        self.ao_of_atom = [cupy.asarray(np.where(self.atom_of_ao == ia)[0])
                           for ia in range(mol.natm)]

    def block_atoms(self, grids):
        """``(atoms, rows)`` for every block ``block_loop`` will yield.

        ``atoms`` are the atoms with at least one AO in that block's mask --
        the others contribute identically zero there -- and ``rows[p]`` is the
        position of masked AO ``p``'s atom in ``atoms``.

        This runs before the grid loop rather than inside it: the AO mask is a
        property of the grid block alone, so reading it costs one small copy
        per block with nothing in flight, where the same copy inside the loop
        would drain the launch queue 127 times.  The correspondence to
        ``block_loop`` is by order, with the same "skip an empty mask" rule,
        and the caller asserts the lengths agree.
        """
        out = []
        for entry in grids.get_non0ao_idx(self.opt):
            idx = entry[1]
            if len(idx) == 0:
                continue
            at = self.atom_of_ao[cupy.asnumpy(idx)]
            atoms = (np.unique(at) if _ATOMSCREEN
                     else np.arange(self.natm))      # the ablation
            out.append((atoms, np.searchsorted(atoms, at)))
        return out

    @staticmethod
    def sizes(blocks):
        """``(nao_sub_max, natm_local_max)`` over the blocks."""
        return (max(len(rows) for _, rows in blocks),
                max(len(atoms) for atoms, _ in blocks))

    @staticmethod
    def atom_matrix(atoms, rows, nsub):
        """``P[a,p] = 1`` when masked AO ``p`` belongs to ``atoms[a]``.

        The atom dimension of ``d rho / d R`` is a segmented sum over AOs, and
        contracting with ``P`` turns it into one GEMM in place of the 13 slice
        contractions per atom that ``_make_dR_rho1`` issues.
        """
        P = cupy.zeros((len(atoms), nsub))
        P[cupy.asarray(rows), cupy.arange(nsub)] = 1.0
        return P


def _index_arrays():
    """The two index tables, on the current device, one row per direction."""
    dev = cupy.cuda.Device().id
    if dev not in _DEV_IDX:
        _DEV_IDX[dev] = ([cupy.asarray(_H[s]) for s in range(3)],
                         [cupy.asarray(_SEL[s]) for s in range(3)])
    return _DEV_IDX[dev]


class _Bufs:
    """Fixed-size scratch, allocated once per call.

    Every array in the grid loop is sized by ``nao_sub``, which changes from
    block to block, so allocating them per block means the memory pool cannot
    reuse the last block's -- and at 113 atoms ``aow`` alone is 2.6 GB a block.
    Measured, forming ``aow`` and ``mow`` that way is **63 %** of
    ``_get_vxc_deriv1``: 19.0 s and 5.7 s of a 39.3 s call on taxol, against
    7.6 s for the two GEMMs they feed, which run at 12.8 TFLOP/s.  GPU4PySCF's
    own ``_scale_ao`` kernel takes the same 21.7 s for the same array -- two
    unrelated kernels agreeing to 14 % is what says the cost is not in either
    of them.  This is the same hazard ``fastxcgrad`` records for
    ``_grouped_block_loop``: one reused buffer at 0.13 s where per-block
    allocation costs 1.60 s.

    So: one buffer per name, at the largest ``nao_sub`` any block has, with
    ``cupy.ndarray(shape, memptr=...)`` views on top -- GPU4PySCF's own idiom
    in these functions.
    """

    def __init__(self, **sizes):
        self._buf = {k: cupy.empty(n) for k, n in sizes.items()}
        self.nbytes = sum(b.nbytes for b in self._buf.values())

    def __call__(self, name, shape):
        n = int(np.prod(shape))
        buf = self._buf[name]
        assert n <= buf.size, (name, shape, buf.size)
        return cupy.ndarray(shape, memptr=buf.data)


def _rho_gga(ao, c0):
    """rho and its gradient from ``c0 = D . ao[0]``; same as ``eval_rho2``."""
    rho = numint._contract_rho1(ao[:4], c0)
    rho[1:4] *= 2
    return rho


def _rho(ao, c, nc):
    """``_rho_gga``, and for a meta-GGA (``nc = 5``) the kinetic density too.

    ``tau = 1/2 sum_j (d_j ao) . (D . d_j ao)``, GPU4PySCF's convention (the
    one ``eval_rho2`` uses), from the ``c = D . ao[:4]`` this module forms for
    ``d rho / d R`` anyway.
    """
    rho = _rho_gga(ao, c[0])
    if nc == 4:
        return rho
    out = cupy.empty((5, rho.shape[1]))
    out[:4] = rho
    out[4] = numint._contract_rho(ao[1], c[1])
    for j in (2, 3):
        out[4] += numint._contract_rho(ao[j], c[j])
    out[4] *= .5
    return out


def _dR_rho1(ao, c, P, bufs, out=None, nc=4):
    """``d rho / d R_A`` for every present atom at once, GGA or meta-GGA.

    ``hessian.rks._make_dR_rho1`` for all atoms: the AO-resolved products are
    formed for the whole block, and the atom dimension is one contraction.
    ``out`` is where the answer goes; the unrestricted path passes one row of
    a two-spin array, because ``fxc`` couples the channels and both spins'
    ``d rho / d R`` have to be resident at once.

    ``nc = 5`` adds the meta-GGA row, ``_make_dR_rho1``'s
    ``1/2 sum_j (d_s d_j ao) (D . d_j ao)`` restricted to atom A's AOs: the
    same second-derivative gather the gradient rows use, against ``c[1:4]``
    instead of ``c[0]``.
    """
    hg, _ = _index_arrays()
    nsub, ng = ao.shape[1:]
    prod = bufs('prod', (3, nc, nsub, ng))
    hess = bufs('gather', (3, nsub, ng))
    tmp = bufs('tmp', (3, nsub, ng))
    # one direction at a time: the whole (3,4,nao_sub,ngrids) product would
    # otherwise need two temporaries of its own size
    for s in range(3):
        # y = 0:      (d_s ao) c0
        cupy.multiply(ao[s + 1], c[0], out=prod[s, 0])
        # y = 1,2,3:  (d_s d_y ao) c0  +  (d_s ao) c_y
        cupy.take(ao, hg[s], axis=0, out=hess)
        cupy.multiply(hess, c[0], out=prod[s, 1:4])
        cupy.multiply(ao[s + 1], c[1:4], out=tmp)
        prod[s, 1:4] += tmp
        if nc == 5:
            # y = 4:  1/2 sum_j (d_s d_j ao) c_j
            cupy.multiply(hess, c[1:4], out=tmp)
            cupy.sum(tmp, axis=0, out=prod[s, 4])
            prod[s, 4] *= .5
    if out is None:
        out = bufs('dR', (len(P), 3, nc, ng))
    contract('ap,sypg->asyg', P, prod, out=out)
    out *= 2            # for |mu> D <d_R nu|, as _make_dR_rho1 does
    return out


def _wv2(dR_rho1, wf, out, tau=None):
    """``fxc`` contracted with ``d rho / d R``, with GPU4PySCF's factors.

    ``tau`` is the factor on the meta-GGA row, and it is not the same in the
    two functions: 1/4 in ``_get_vxc_deriv1``, which adds the tau term and
    its transpose, and 1/2 in ``_get_vxc_deriv2``, which adds it once.
    """
    wv = contract('xyg,asxg->asyg', wf, dR_rho1, out=out)
    wv[:, :, 0] *= .5
    if tau is not None:
        wv[:, :, 4] *= tau
    return wv


_SCALE_SUM_SRC = r'''
#define AG 4
#define PT 8
extern "C" __global__
void scale_sum4(const double* __restrict__ src, const double* __restrict__ w,
                double* __restrict__ out, int nb, int np, int ng,
                long long w_a, long long o_a, long long o_p, int ntau,
                double tau_fac)
{
    // out[a, p, 0, g] = sum_{y<4} src[y, p, g] w[a, y, g]
    // out[a, p, 1+j, g] = tau_fac src[1+j, p, g] w[a, 4, g]   (j < 3)
    // one thread: one g, AG atoms-directions held in registers, PT AOs
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ng) return;
    int p0 = blockIdx.y * PT;
    int a0 = blockIdx.z * AG;
    double wr[AG][5];
    int na = min(AG, nb - a0);
    for (int k = 0; k < AG; k++) {
        const double* wa = w + (long long)(a0 + (k < na ? k : 0)) * w_a + g;
        for (int y = 0; y < 4; y++) wr[k][y] = wa[y * ng];
        wr[k][4] = ntau ? tau_fac * wa[4 * ng] : 0.;
    }
    long long png = (long long)np * ng;
    int p1 = min(p0 + PT, np);
    for (int p = p0; p < p1; p++) {
        long long pg = (long long)p * ng + g;
        double s0 = src[pg], s1 = src[png + pg], s2 = src[2 * png + pg],
               s3 = src[3 * png + pg];
        for (int k = 0; k < na; k++) {
            double* oa = out + (a0 + k) * o_a + p * o_p + g;
            oa[0] = s0 * wr[k][0] + s1 * wr[k][1] + s2 * wr[k][2]
                  + s3 * wr[k][3];
            if (ntau) {
                oa[ng] = s1 * wr[k][4];
                oa[2 * ng] = s2 * wr[k][4];
                oa[3 * ng] = s3 * wr[k][4];
            }
        }
    }
}
'''
_scale_sum_mod = None


def _scale_sum(src, w, out, tau_fac=None):
    """``out[a,p,g] = sum_{y<4} src[y,p,g] w[a,y,g]``, as one kernel.

    ``aow`` and ``mow`` of ``_get_vxc_deriv1``.  As the cuTENSOR contraction
    ``'ypg,ayg->apg'`` (``g`` is a mode of all three operands, a batch index
    that library does not treat as one) this wrote ``aow`` at 139 GB/s, and on
    a 42-atom meta-GGA it was 0.64 s of a 2.5 s function -- as much as the
    GEMMs it feeds, which run at 16 TF/s (``docs/README_fastxchessmgga.md``
    §3).  Here a thread holds one grid point's weights for four rows of ``a``
    in registers and walks eight AOs, so each ``src`` value is read once per
    four outputs: 766 GB/s, 0.115 s.

    With ``tau_fac`` it also writes the meta-GGA slabs of ``_deriv1_gemms``'
    stacked operand, ``out[a,p,1+j,g] = tau_fac src[1+j,p,g] w[a,4,g]``;
    ``out`` is then ``(nb, np, 4, ng)`` and ``w`` carries five components.
    """
    global _scale_sum_mod
    if _scale_sum_mod is None:
        _scale_sum_mod = cupy.RawModule(code=_SCALE_SUM_SRC)
    k = _scale_sum_mod.get_function('scale_sum4')
    ny, np_, ng = src.shape[0], src.shape[1], src.shape[2]
    nb = w.shape[0]
    assert ny >= 4 and src[:4].flags.c_contiguous and src.dtype == np.float64
    assert w.flags.c_contiguous and w.shape[2] == ng
    assert out.flags.c_contiguous
    ntau = 0 if tau_fac is None else 1
    o_p = ng * (4 if ntau else 1)
    threads = 128
    grid = ((ng + threads - 1) // threads, (np_ + 7) // 8, (nb + 3) // 4)
    k(grid, (threads,), (src, w, out, np.int32(nb), np.int32(np_),
                         np.int32(ng), np.int64(w.shape[1] * ng),
                         np.int64(np_ * o_p), np.int64(o_p), np.int32(ntau),
                         np.float64(tau_fac or 0.)))
    return out


def _deriv1_gemms(ao, mo, w, nsub, nocc, nb, ng, bufs, nc):
    """``t[a,p,q]``, one atom block's contribution to ``vmat[a, x, mask, :]``.

    GGA: ``aow . mo0^T + (mow . ao0^T)^T`` with ``aow = sum_y ao_y w_y`` and
    ``mow = sum_y mo_y w_y``.  The meta-GGA adds upstream's ``for j in
    range(1, 4)`` block, ``(ao_j w_4) . mo_j^T`` and its transposed twin
    ``((mo_j w_4) . ao_j^T)^T`` -- which are the same matrix, so the tau term
    is ``2 (mo_j w_4) . ao_j^T`` summed over ``j``.  That sum and the GGA
    ``mow . ao0^T`` share an operand shape, so they are one GEMM with the
    inner dimension ``4 ngrids``: ``[mow | 2 mo_j w_4] . [ao0 | ao_j]^T``.
    Upstream issues eight GEMMs per atom and direction for this; here it is
    two per atom block.
    """
    aow = bufs('aow', (nb, nsub, ng))
    if _SCALESUM:
        _scale_sum(ao, w, aow)
    else:
        contract('ypg,ayg->apg', ao[:4], w[:, :4], out=aow)
    t = bufs('t1', (nb, nsub, nocc))
    cupy.dot(aow.reshape(-1, ng), mo[0].T, out=t.reshape(-1, nocc))
    t2 = bufs('t2', (nb, nocc, nsub))
    if nc == 4:
        mow = bufs('mow', (nb, nocc, ng))
        if _SCALESUM:
            _scale_sum(mo, w, mow)
        else:
            contract('yqg,ayg->aqg', mo, w, out=mow)
        cupy.dot(mow.reshape(-1, ng), ao[0].T, out=t2.reshape(-1, nsub))
    else:
        mow = bufs('mow', (nb, nocc, 4, ng))
        if _SCALESUM:
            _scale_sum(mo, w, mow, tau_fac=2.)
        else:
            contract('yqg,ayg->aqg', mo, w[:, :4], out=mow[:, :, 0])
            w4 = bufs('w4', (nb, ng))
            cupy.multiply(w[:, 4], 2., out=w4)
            cupy.multiply(mo[1:4].transpose(1, 0, 2)[None],
                          w4[:, None, None], out=mow[:, :, 1:4])
        aoT = bufs('aoT', (nsub, 4, ng))
        aoT[:] = ao[:4].transpose(1, 0, 2)
        cupy.dot(mow.reshape(-1, 4 * ng), aoT.reshape(nsub, 4 * ng).T,
                 out=t2.reshape(-1, nsub))
    t += t2.transpose(0, 2, 1)
    return t


def _deriv1_block(ao, mask, atoms, P, weight, dm_s, mocc_s, ni, xc,
                  v_ip, vmat2d, nao_s, bufs, chunk, nc=4):
    nsub, ng = ao.shape[1:]
    nocc = mocc_s.shape[1]
    xctype = 'GGA' if nc == 4 else 'MGGA'
    dm_mask = take_last2d(dm_s, mask, out=bufs('dm', (nsub, nsub)))
    c = bufs('c', (4, nsub, ng))
    for m in range(4):
        cupy.dot(dm_mask, ao[m], out=c[m])
    rho = _rho(ao, c, nc)
    vxc, fxc = ni.eval_xc_eff(xc, rho, 2, xctype=xctype, spin=0)[1:3]
    rho = None
    wv = cupy.multiply(weight, vxc, out=vxc)
    wv[0] *= .5
    wf = cupy.multiply(weight, fxc, out=fxc)
    vip = rks_grad._gga_grad_sum_(ao, wv, buf=bufs('gga', (3, nsub, ng)),
                                  out=bufs('vip', (3, nsub, nsub)))
    if nc == 5:
        wv[4] *= .5  # for the factor 1/2 in tau
        rks_grad._tau_grad_dot_(ao, wv[4], accumulate=True,
                                buf=bufs('gga', (nsub, ng)), out=vip)
    add_sparse(v_ip, vip, mask)
    vxc = fxc = wv = vip = None

    mo = bufs('mo', (4, nocc, ng))
    contract('xpg,pq->xqg', ao[:4], mocc_s[mask], out=mo)
    wv2 = _wv2(_dR_rho1(ao, c, P, bufs, nc=nc), wf,
               bufs('wv2', (len(atoms), 3, nc, ng)),
               tau=.25 if nc == 5 else None)
    wf = None
    na_all = len(atoms)
    at3 = cupy.asarray((atoms[:, None] * 3 + np.arange(3)).ravel(),
                       dtype=cupy.int64)
    for a0 in range(0, na_all, chunk):
        a1 = min(a0 + chunk, na_all)
        nb = (a1 - a0) * 3
        w = wv2[a0:a1].reshape(nb, nc, ng)
        # the GEMMs, through cupy.dot rather than `contract`: with the
        # atom and AO indices flattened together these are plain DGEMMs of
        # (natm_blk*3*nao_sub x ngrids) . (ngrids x nocc), which is where
        # 12.8 TFLOP/s comes from.  Forming `aow` and `mow` for them is the
        # expensive part, and it is expensive only if it allocates -- see
        # _Bufs.
        t = _deriv1_gemms(ao, mo, w, nsub, nocc, nb, ng, bufs, nc)
        # vmat[atom, x, mask, :] += t, as a flat row scatter
        rows = (at3[a0 * 3:a1 * 3, None] * nao_s
                + mask[None, :].astype(cupy.int64)).ravel()
        cupyx.scatter_add(vmat2d, rows, t.reshape(-1, nocc))
        rows = None


def _deriv1_bufs(ns_max, na_max, nocc, chunk, nc, blk, nspin=1):
    """The scratch of ``_deriv1``/``_deriv1_u``, one buffer per name."""
    mg = nc == 5
    return _Bufs(
        dm=ns_max * ns_max, c=nspin * 4 * ns_max * blk, gga=3 * ns_max * blk,
        vip=3 * ns_max * ns_max, mo=4 * nocc * blk,
        prod=3 * nc * ns_max * blk, gather=3 * ns_max * blk,
        tmp=3 * ns_max * blk, dR=nspin * na_max * 3 * nc * blk,
        wv2=nspin * na_max * 3 * nc * blk,
        aow=chunk * 3 * ns_max * blk,
        mow=chunk * 3 * nocc * blk * (4 if mg else 1),
        w4=chunk * 3 * blk if mg else 1, aoT=4 * ns_max * blk if mg else 1,
        t1=chunk * 3 * ns_max * nocc, t2=chunk * 3 * ns_max * nocc)


def _chunk(ns_max, nocc, na_max, blk, nc):
    """How many atoms one pass may carry.

    ``aow`` and ``mow`` are the only scratch whose size grows with it, and
    they are allocated once for that many; a meta-GGA's ``mow`` carries the
    three tau slabs as well.
    """
    per_atom = 3 * (ns_max + nocc * (4 if nc == 5 else 1) + 1) * blk * 8
    return max(1, min(na_max, int(MEM_FRACTION * get_avail_mem() / per_atom)))


def _deriv1(hessobj, mo_coeff, mo_occ, max_memory, lay, grids, ni, nc=4):
    mol = hessobj.mol
    mf = hessobj.base
    log = logger.new_logger(mol, mol.verbose)
    t0 = log.init_timer()
    nao, nmo = mo_coeff.shape
    mo_coeff = cupy.asarray(mo_coeff)
    mo_occ = cupy.asarray(mo_occ)
    mocc = mo_coeff[:, mo_occ > 0]
    nocc = mocc.shape[1]
    natm = mol.natm
    nao_s = lay.nao_s
    opt = lay.opt
    dm_s = opt.sort_orbitals(mf.make_rdm1(mo_coeff, mo_occ), axis=[0, 1])
    mocc_s = opt.sort_orbitals(mocc, axis=[0])
    mo_coeff_s = opt.sort_orbitals(mo_coeff, axis=[0])

    v_ip = cupy.zeros((3, nao_s, nao_s))
    vmat = cupy.zeros((natm, 3, nao_s, nocc))
    vmat2d = vmat.reshape(natm * 3 * nao_s, nocc)
    blocks = lay.block_atoms(grids)
    ns_max, na_max = lay.sizes(blocks)
    blk = numint.MIN_BLK_SIZE
    chunk = _chunk(ns_max, nocc, na_max, blk, nc)
    bufs = _deriv1_bufs(ns_max, na_max, nocc, chunk, nc, blk)
    log.debug1(f'fastxchess deriv1 (nc={nc}): nao_sub_max {ns_max}, natm_local_max '
               f'{na_max}, {chunk} atoms a pass, {bufs.nbytes/GB:.2f} GB '
               f'scratch')
    it = iter(blocks)
    for ao, mask, weight, _ in ni.block_loop(lay._sorted_mol, grids, nao, 2,
                                             None):
        atoms, rows = next(it)
        assert len(rows) == len(mask)
        P = lay.atom_matrix(atoms, rows, len(mask))
        _deriv1_block(ao, mask, atoms, P, weight, dm_s, mocc_s, ni, mf.xc,
                      v_ip, vmat2d, nao_s, bufs, chunk, nc)
        P = None
    bufs = None
    t0 = log.timer_debug1('vxc_deriv1 (fastxchess)', *t0)

    # v_ip carries no atom index: the atom comes from restricting its rows
    # (and, transposed, its columns) to that atom's AOs -- GPU4PySCF's own
    # closing loop, with index sets in place of the p0:p1 slices.
    v_mo = cupy.empty((natm, 3, nmo, nocc))
    tmp_buf = cupy.empty((3, nao_s, nao_s))
    for ia in range(natm):
        p = lay.ao_of_atom[ia]
        tmp_buf[:] = 0.
        tmp_buf[:, p] = v_ip[:, p]
        tmp_buf[:, :, p] += v_ip[:, p].transpose(0, 2, 1)
        tmp = contract('xij,jq->xiq', tmp_buf, mocc_s)
        tmp += vmat[ia]
        contract('xiq,ip->xpq', tmp, mo_coeff_s, alpha=-1., out=v_mo[ia])
        tmp = None
    return v_mo


def _h_flat():
    """``_H`` flattened, on the current device: ``ao[_h_flat()]`` is the
    ``(3, 3)`` block of second derivatives, row ``k``, column ``d``."""
    dev = cupy.cuda.Device().id
    key = ('hflat', dev)
    if key not in _DEV_IDX:
        _DEV_IDX[key] = cupy.asarray(_H.ravel())
    return _DEV_IDX[key]


def _ipip_block(ao, wv1, nsub, ng, bufs, nc):
    """The atom-independent ``ipip[x, y]`` term of one block, masked.

    GGA: ``_make_dR_dao_w(ao, wv1)[y] . (d_x ao)^T``.  A meta-GGA adds
    upstream's three ``_d1d2_dot_`` calls on the second derivatives,
    ``sum_k (d_k d_y ao w_tau) . (d_k d_x ao)^T`` -- nine GEMMs of inner
    dimension ``ngrids`` there, one contraction of inner dimension
    ``3 ngrids`` per ``(x, y)`` here, all in one call.
    """
    aow = rks_grad._make_dR_dao_w(ao, wv1, out=bufs('gga', (3, nsub, ng)))
    ip = bufs('vip', (3, 3, nsub, nsub))
    for d1 in range(3):
        for d2 in range(3):
            contract('ik,lk->il', aow[d2], ao[1 + d1], out=ip[d1, d2])
    if nc == 5:
        G = bufs('G', (9, nsub, ng))
        cupy.take(ao, _h_flat(), axis=0, out=G)
        Gw = bufs('Gw', (9, nsub, ng))
        cupy.multiply(G, wv1[4], out=Gw)
        contract('kypg,kxqg->xypq', Gw.reshape(3, 3, nsub, ng),
                 G.reshape(3, 3, nsub, ng), beta=1., out=ip)
    return ip


def _deriv2_t(ao, c, wv2, na, nsub, ng, bufs, nc):
    """``t[x, a, s, j]``, one block's ``vmat_dm[atom, s, x, mask]``.

    W[x][y,j,g]: the components _make_dR_dao_w would combine, times the
    density product -- built once per block rather than once per atom, and
    one direction at a time so that nothing of size (3,4,nao_sub,ngrids) is
    ever resident.  A meta-GGA's fifth row of ``W[x]`` is upstream's nine
    ``contract('jg,xjg->xj', ao_mask[d_k d_x], ...)`` lines per atom,
    ``sum_k (d_k d_x ao) c_k``, which is again atom-independent, so its whole
    tau term is the fifth component of the same contraction.
    """
    hg, selg = _index_arrays()
    t = bufs('t1', (3, na, 3, nsub))          # [x, atom, direction, AO]
    W = bufs('W', (nc, nsub, ng))
    for x in range(3):
        cupy.take(ao, selg[x], axis=0, out=W[:4])
        W[:4] *= c[0]
        if nc == 5:
            hx = bufs('gather', (3, nsub, ng))
            cupy.take(ao, hg[x], axis=0, out=hx)
            hx *= c[1:4]
            cupy.sum(hx, axis=0, out=W[4])
        contract('yjg,asyg->asj', W, wv2, out=t[x])
    # U[y][z,j,g] = (d_y ao)[j,g] c[z,j,g]
    U = W[:4]
    for y in range(3):
        cupy.multiply(ao[y + 1], c[:4], out=U)
        contract('zjg,aszg->asj', U, wv2[:, :, :4], beta=1., out=t[y])
    tT = bufs('t2', (na, 3, 3, nsub))
    tT[:] = t.transpose(1, 2, 0, 3)
    return tT


def _deriv2_block(ao, mask, atoms, P, weight, dm_s, ni, xc, ipip, vmat_dm1d,
                  nao_s, bufs, nc=4):
    nsub, ng = ao.shape[1:]
    xctype = 'GGA' if nc == 4 else 'MGGA'
    dm_mask = take_last2d(dm_s, mask, out=bufs('dm', (nsub, nsub)))
    c = bufs('c', (4, nsub, ng))
    for m in range(4):
        cupy.dot(dm_mask, ao[m], out=c[m])
    rho = _rho(ao, c, nc)
    vxc, fxc = ni.eval_xc_eff(xc, rho, 2, xctype=xctype, spin=0)[1:3]
    rho = None
    wv1 = cupy.multiply(weight, vxc, out=vxc)
    wv1[0] *= .5
    if nc == 5:
        wv1[4] *= .25
    wf = cupy.multiply(weight, fxc, out=fxc)
    add_sparse(ipip, _ipip_block(ao, wv1, nsub, ng, bufs, nc), mask)
    vxc = fxc = wv1 = None

    na = len(atoms)
    wv2 = _wv2(_dR_rho1(ao, c, P, bufs, nc=nc), wf,
               bufs('wv2', (na, 3, nc, ng)), tau=.5 if nc == 5 else None)
    wf = None
    t = _deriv2_t(ao, c, wv2, na, nsub, ng, bufs, nc)
    rows = cupy.asarray(
        ((atoms[:, None, None] * 3 + np.arange(3)[None, :, None]) * 3
         + np.arange(3)[None, None, :]).ravel(), dtype=cupy.int64)
    idx = (rows[:, None] * nao_s + mask[None, :].astype(cupy.int64)).ravel()
    cupyx.scatter_add(vmat_dm1d, idx, t.reshape(-1))
    t = rows = idx = None


def _deriv2_bufs(ns_max, na_max, nc, blk, nspin=1):
    mg = nc == 5
    return _Bufs(
        dm=ns_max * ns_max, c=nspin * 4 * ns_max * blk, gga=3 * ns_max * blk,
        vip=9 * ns_max * ns_max, prod=3 * nc * ns_max * blk,
        gather=3 * ns_max * blk, tmp=3 * ns_max * blk,
        dR=nspin * na_max * 3 * nc * blk, wv2=nspin * na_max * 3 * nc * blk,
        W=nc * ns_max * blk, G=9 * ns_max * blk if mg else 1,
        Gw=9 * ns_max * blk if mg else 1,
        t1=9 * na_max * ns_max, t2=9 * na_max * ns_max)


def _deriv2(hessobj, mo_coeff, mo_occ, max_memory, lay, grids, ni, nc=4):
    mol = hessobj.mol
    mf = hessobj.base
    log = logger.new_logger(mol, mol.verbose)
    t0 = log.init_timer()
    nao = mol.nao
    nao_s = lay.nao_s
    natm = mol.natm
    opt = lay.opt
    mo_coeff = cupy.asarray(mo_coeff)
    mo_occ = cupy.asarray(mo_occ)
    dm_s = opt.sort_orbitals(mf.make_rdm1(mo_coeff, mo_occ), axis=[0, 1])

    vmat_dm = cupy.zeros((natm, 3, 3, nao_s))
    vmat_dm1d = vmat_dm.reshape(-1)
    ipip = cupy.zeros((3, 3, nao_s, nao_s))
    blocks = lay.block_atoms(grids)
    ns_max, na_max = lay.sizes(blocks)
    blk = numint.MIN_BLK_SIZE
    bufs = _deriv2_bufs(ns_max, na_max, nc, blk)
    log.debug1(f'fastxchess deriv2 (nc={nc}): nao_sub_max {ns_max}, natm_local_max '
               f'{na_max}, {bufs.nbytes/GB:.2f} GB scratch')
    it = iter(blocks)
    for ao, mask, weight, _ in ni.block_loop(lay._sorted_mol, grids, nao, 2,
                                             None):
        atoms, rows = next(it)
        assert len(rows) == len(mask)
        P = lay.atom_matrix(atoms, rows, len(mask))
        _deriv2_block(ao, mask, atoms, P, weight, dm_s, ni, mf.xc, ipip,
                      vmat_dm1d, nao_s, bufs, nc)
        P = None
    bufs = None
    t0 = log.timer_debug1('vxc_deriv2 (fastxchess)', *t0)

    for ia in range(natm):
        p = lay.ao_of_atom[ia]
        contract('xypq,pq->xyp', ipip[:, :, :, p], dm_s[:, p], beta=1,
                 out=vmat_dm[ia])
        contract('yxqp,pq->xyp', ipip[:, :, p], dm_s[:, p], beta=1,
                 out=vmat_dm[ia])
    return opt.unsort_orbitals(vmat_dm, axis=[3])


# ------------------------------------------------ unrestricted --------------

def _c_and_rho_u(ao, mask, dm_s, bufs, nc=4):
    """``D_s . ao`` for both spins, and the two spin densities.

    The block's masked density matrix is formed one spin at a time into the
    same scratch, because only the product ``c`` has to survive.
    """
    nsub, ng = ao.shape[1:]
    c = bufs('c', (2, 4, nsub, ng))
    rho = cupy.empty((2, nc, ng))
    for s in range(2):
        dm_mask = take_last2d(dm_s[s], mask, out=bufs('dm', (nsub, nsub)))
        for m in range(4):
            cupy.dot(dm_mask, ao[m], out=c[s, m])
        rho[s] = _rho(ao, c[s], nc)
    return c, rho


def _wv2_u(dR, wf, out, tau=None):
    """``fxc`` contracted with ``d rho / d R``, both spins at once.

    This is the one place where an unrestricted XC Hessian is not the
    restricted one run twice: ``fxc`` couples the channels, so the perturbed
    potential of spin ``b`` takes *both* spins' perturbed densities,

        wv2[b] = sum_s fxc[s, :, b] . dR_rho1[s]

    -- which is why the two spins share a grid block here rather than the
    module being called once per spin.  Everything downstream of this line is
    the restricted arithmetic with a spin index, and everything upstream of it
    (the AO evaluation, the mask, the atom list, ``d rho / d R``'s AO-resolved
    products) is shared by both spins anyway.
    """
    contract('xbyg,asxg->basyg', wf[0], dR[0], out=out)
    contract('xbyg,asxg->basyg', wf[1], dR[1], beta=1., out=out)
    out[:, :, :, 0] *= .5
    if tau is not None:
        out[:, :, :, 4] *= tau
    return out


def _dR_and_wv2_u(ao, c, P, wf, bufs, na, nc=4, tau=None):
    """``d rho / d R`` for both spins, and ``fxc`` contracted with it."""
    ng = ao.shape[2]
    dR = bufs('dR', (2, na, 3, nc, ng))
    for s in range(2):
        _dR_rho1(ao, c[s], P, bufs, out=dR[s], nc=nc)
    return _wv2_u(dR, wf, bufs('wv2', (2, na, 3, nc, ng)), tau)


def _deriv1_block_u(ao, mask, atoms, P, weight, dm_s, mocc_s, ni, xc,
                    v_ip, vmat2d, nao_s, bufs, chunk, nc=4):
    nsub, ng = ao.shape[1:]
    xctype = 'GGA' if nc == 4 else 'MGGA'
    c, rho = _c_and_rho_u(ao, mask, dm_s, bufs, nc)
    vxc, fxc = ni.eval_xc_eff(xc, rho, 2, xctype=xctype, spin=1)[1:3]
    rho = None
    wv = cupy.multiply(weight, vxc, out=vxc)
    wv[:, 0] *= .5
    if nc == 5:
        wv[:, 4] *= .5  # for the factor 1/2 in tau
    wf = cupy.multiply(weight, fxc, out=fxc)
    for s in range(2):
        vip = rks_grad._gga_grad_sum_(ao, wv[s],
                                      buf=bufs('gga', (3, nsub, ng)),
                                      out=bufs('vip', (3, nsub, nsub)))
        if nc == 5:
            rks_grad._tau_grad_dot_(ao, wv[s, 4], accumulate=True,
                                    buf=bufs('gga', (nsub, ng)), out=vip)
        add_sparse(v_ip[s], vip, mask)
    vxc = fxc = wv = vip = None

    na_all = len(atoms)
    wv2 = _dR_and_wv2_u(ao, c, P, wf, bufs, na_all, nc,
                        tau=.25 if nc == 5 else None)
    wf = None
    at3 = cupy.asarray((atoms[:, None] * 3 + np.arange(3)).ravel(),
                       dtype=cupy.int64)
    for s in range(2):
        nocc = mocc_s[s].shape[1]
        mo = bufs('mo', (4, nocc, ng))
        contract('xpg,pq->xqg', ao[:4], mocc_s[s][mask], out=mo)
        for a0 in range(0, na_all, chunk):
            a1 = min(a0 + chunk, na_all)
            nb = (a1 - a0) * 3
            w = wv2[s, a0:a1].reshape(nb, nc, ng)
            t = _deriv1_gemms(ao, mo, w, nsub, nocc, nb, ng, bufs, nc)
            rows = (at3[a0 * 3:a1 * 3, None] * nao_s
                    + mask[None, :].astype(cupy.int64)).ravel()
            cupyx.scatter_add(vmat2d[s], rows, t.reshape(-1, nocc))
            rows = None


def _deriv1_u(hessobj, mo_coeff, mo_occ, max_memory, lay, grids, ni, nc=4):
    mol = hessobj.mol
    mf = hessobj.base
    log = logger.new_logger(mol, mol.verbose)
    t0 = log.init_timer()
    mo_coeff = cupy.asarray(mo_coeff)
    mo_occ = cupy.asarray(mo_occ)
    nao, nmo = mo_coeff[0].shape
    natm = mol.natm
    nao_s = lay.nao_s
    opt = lay.opt
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dm_s = [opt.sort_orbitals(cupy.asarray(dm0[s]), axis=[0, 1])
            for s in range(2)]
    mocc = [mo_coeff[s][:, mo_occ[s] > 0] for s in range(2)]
    nocc = [int(m.shape[1]) for m in mocc]
    nocc_max = max(nocc)
    mocc_s = [opt.sort_orbitals(m, axis=[0]) for m in mocc]
    mo_coeff_s = [opt.sort_orbitals(mo_coeff[s], axis=[0]) for s in range(2)]

    v_ip = cupy.zeros((2, 3, nao_s, nao_s))
    vmat = [cupy.zeros((natm, 3, nao_s, nocc[s])) for s in range(2)]
    vmat2d = [vmat[s].reshape(natm * 3 * nao_s, nocc[s]) for s in range(2)]
    blocks = lay.block_atoms(grids)
    ns_max, na_max = lay.sizes(blocks)
    blk = numint.MIN_BLK_SIZE
    # `dR` and `wv2` carry a spin index the restricted path does not
    chunk = _chunk(ns_max, nocc_max, na_max, blk, nc)
    bufs = _deriv1_bufs(ns_max, na_max, nocc_max, chunk, nc, blk, nspin=2)
    log.debug1(f'fastxchess deriv1 (uks, nc={nc}): nao_sub_max {ns_max}, '
               f'natm_local_max {na_max}, {chunk} atoms a pass, '
               f'{bufs.nbytes/GB:.2f} GB scratch')
    it = iter(blocks)
    for ao, mask, weight, _ in ni.block_loop(lay._sorted_mol, grids, nao, 2,
                                             None):
        atoms, rows = next(it)
        assert len(rows) == len(mask)
        P = lay.atom_matrix(atoms, rows, len(mask))
        _deriv1_block_u(ao, mask, atoms, P, weight, dm_s, mocc_s, ni, mf.xc,
                        v_ip, vmat2d, nao_s, bufs, chunk, nc)
        P = None
    bufs = None
    t0 = log.timer_debug1('vxc_deriv1 uks (fastxchess)', *t0)

    out = []
    tmp_buf = cupy.empty((3, nao_s, nao_s))
    for s in range(2):
        v_mo = cupy.empty((natm, 3, nmo, nocc[s]))
        for ia in range(natm):
            p = lay.ao_of_atom[ia]
            tmp_buf[:] = 0.
            tmp_buf[:, p] = v_ip[s][:, p]
            tmp_buf[:, :, p] += v_ip[s][:, p].transpose(0, 2, 1)
            tmp = contract('xij,jq->xiq', tmp_buf, mocc_s[s])
            tmp += vmat[s][ia]
            contract('xiq,ip->xpq', tmp, mo_coeff_s[s], alpha=-1.,
                     out=v_mo[ia])
            tmp = None
        out.append(v_mo)
    return out[0], out[1]


def _deriv2_block_u(ao, mask, atoms, P, weight, dm_s, ni, xc, ipip,
                    vmat_dm1d, nao_s, bufs, nc=4):
    nsub, ng = ao.shape[1:]
    xctype = 'GGA' if nc == 4 else 'MGGA'
    c, rho = _c_and_rho_u(ao, mask, dm_s, bufs, nc)
    vxc, fxc = ni.eval_xc_eff(xc, rho, 2, xctype=xctype, spin=1)[1:3]
    rho = None
    wv1 = cupy.multiply(weight, vxc, out=vxc)
    wv1[:, 0] *= .5
    if nc == 5:
        wv1[:, 4] *= .25
    wf = cupy.multiply(weight, fxc, out=fxc)
    for s in range(2):
        add_sparse(ipip[s], _ipip_block(ao, wv1[s], nsub, ng, bufs, nc),
                   mask)
    vxc = fxc = wv1 = None

    na = len(atoms)
    wv2 = _dR_and_wv2_u(ao, c, P, wf, bufs, na, nc,
                        tau=.5 if nc == 5 else None)
    wf = None
    rows = cupy.asarray(
        ((atoms[:, None, None] * 3 + np.arange(3)[None, :, None]) * 3
         + np.arange(3)[None, None, :]).ravel(), dtype=cupy.int64)
    idx = (rows[:, None] * nao_s + mask[None, :].astype(cupy.int64)).ravel()
    for s in range(2):
        t = _deriv2_t(ao, c[s], wv2[s], na, nsub, ng, bufs, nc)
        cupyx.scatter_add(vmat_dm1d[s], idx, t.reshape(-1))
        t = None
    rows = idx = None


def _deriv2_u(hessobj, mo_coeff, mo_occ, max_memory, lay, grids, ni, nc=4):
    mol = hessobj.mol
    mf = hessobj.base
    log = logger.new_logger(mol, mol.verbose)
    t0 = log.init_timer()
    nao = mol.nao
    nao_s = lay.nao_s
    natm = mol.natm
    opt = lay.opt
    mo_coeff = cupy.asarray(mo_coeff)
    mo_occ = cupy.asarray(mo_occ)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dm_s = [opt.sort_orbitals(cupy.asarray(dm0[s]), axis=[0, 1])
            for s in range(2)]

    vmat_dm = cupy.zeros((2, natm, 3, 3, nao_s))
    vmat_dm1d = [vmat_dm[s].reshape(-1) for s in range(2)]
    ipip = cupy.zeros((2, 3, 3, nao_s, nao_s))
    blocks = lay.block_atoms(grids)
    ns_max, na_max = lay.sizes(blocks)
    blk = numint.MIN_BLK_SIZE
    bufs = _deriv2_bufs(ns_max, na_max, nc, blk, nspin=2)
    log.debug1(f'fastxchess deriv2 (uks, nc={nc}): nao_sub_max {ns_max}, '
               f'natm_local_max {na_max}, {bufs.nbytes/GB:.2f} GB scratch')
    it = iter(blocks)
    for ao, mask, weight, _ in ni.block_loop(lay._sorted_mol, grids, nao, 2,
                                             None):
        atoms, rows = next(it)
        assert len(rows) == len(mask)
        P = lay.atom_matrix(atoms, rows, len(mask))
        _deriv2_block_u(ao, mask, atoms, P, weight, dm_s, ni, mf.xc, ipip,
                        vmat_dm1d, nao_s, bufs, nc)
        P = None
    bufs = None
    t0 = log.timer_debug1('vxc_deriv2 uks (fastxchess)', *t0)

    for s in range(2):
        for ia in range(natm):
            p = lay.ao_of_atom[ia]
            contract('xypq,pq->xyp', ipip[s][:, :, :, p], dm_s[s][:, p],
                     beta=1, out=vmat_dm[s][ia])
            contract('yxqp,pq->xyp', ipip[s][:, :, p], dm_s[s][:, p],
                     beta=1, out=vmat_dm[s][ia])
    out = opt.unsort_orbitals(vmat_dm, axis=[4])
    return out[0], out[1]


# -------------------------------------------------------------- the patch ----

def _setup(hessobj, mo_occ, unrestricted=False):
    """The common preamble, or None when this module declines the call.

    ``mo_occ`` is one vector on a restricted reference and two on an
    unrestricted one, and ``hessian/rks.py`` and ``hessian/uks.py`` have
    separate copies of both functions -- so the check is that the caller and
    the occupancy agree, not merely that the occupancy has some shape.
    """
    mol = hessobj.mol
    mf = hessobj.base
    if _OFF or num_devices > 1:
        return None
    xctype = mf._numint._xc_type(mf.xc)
    if xctype == 'GGA':
        nc = 4
    elif xctype == 'MGGA' and _MGGA:
        nc = 5
    else:
        return None
    if cupy.asarray(mo_occ).ndim != (2 if unrestricted else 1):
        return None
    grids = hessobj.grids if hessobj.grids is not None else mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)   # as GPU4PySCF's own wrapper does
    ni = mf._numint
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    lay = _Layout(opt, mol)
    if not lay.supported:
        return None
    return lay, grids, ni, nc


def get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory):
    ctx = (None if getattr(hessobj, 'grid_response', False)
           else _setup(hessobj, mo_occ))
    if ctx is None:
        return hess_rks._get_vxc_deriv1_orig(hessobj, mo_coeff, mo_occ,
                                             max_memory)
    return _deriv1(hessobj, mo_coeff, mo_occ, max_memory, *ctx)


def get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory):
    ctx = _setup(hessobj, mo_occ)
    if ctx is None:
        return hess_rks._get_vxc_deriv2_orig(hessobj, mo_coeff, mo_occ,
                                             max_memory)
    return _deriv2(hessobj, mo_coeff, mo_occ, max_memory, *ctx)


def get_vxc_deriv1_u(hessobj, mo_coeff, mo_occ, max_memory):
    ctx = (None if getattr(hessobj, 'grid_response', False)
           else _setup(hessobj, mo_occ, unrestricted=True))
    if ctx is None:
        return hess_uks._get_vxc_deriv1_orig(hessobj, mo_coeff, mo_occ,
                                             max_memory)
    return _deriv1_u(hessobj, mo_coeff, mo_occ, max_memory, *ctx)


def get_vxc_deriv2_u(hessobj, mo_coeff, mo_occ, max_memory):
    ctx = _setup(hessobj, mo_occ, unrestricted=True)
    if ctx is None:
        return hess_uks._get_vxc_deriv2_orig(hessobj, mo_coeff, mo_occ,
                                             max_memory)
    return _deriv2_u(hessobj, mo_coeff, mo_occ, max_memory, *ctx)


_PATCHED = False
if not _PATCHED:
    hess_rks._get_vxc_deriv1_orig = hess_rks._get_vxc_deriv1
    hess_rks._get_vxc_deriv2_orig = hess_rks._get_vxc_deriv2
    hess_rks._get_vxc_deriv1 = get_vxc_deriv1
    hess_rks._get_vxc_deriv2 = get_vxc_deriv2
    # `hessian/uks.py` defines its own copies of both -- it does not import
    # the restricted ones -- so patching `hessian.rks` leaves an open-shell
    # Hessian entirely on GPU4PySCF's code with no symptom but the wall
    # clock.  `make_h1` and `_get_exc_deriv2` in that file look both names up
    # as module globals at call time, and `df/hessian/uks.py` reaches them as
    # `uks_hess._get_vxc_deriv1`, so these two assignments cover every caller
    # on 1.8.1; `tests/test_patch_aliases.py` is what keeps that true.
    hess_uks._get_vxc_deriv1_orig = hess_uks._get_vxc_deriv1
    hess_uks._get_vxc_deriv2_orig = hess_uks._get_vxc_deriv2
    hess_uks._get_vxc_deriv1 = get_vxc_deriv1_u
    hess_uks._get_vxc_deriv2 = get_vxc_deriv2_u
    _PATCHED = True


# ------------------------------------------- the in-place contraction (W23) ----

_contract_orig = hess_rks.contract
_ALIASED = [0]


def _alias_safe_contract(pattern, a, b, alpha=1.0, beta=0.0, out=None):
    """``cupy_helper.contract``, correct when ``out`` overlaps an operand.

    ``may_share_memory`` compares address ranges, so it costs nothing and can
    only err towards the copy.  With ``beta`` the old ``out`` is read after
    the temporary is complete, which is what the in-place call meant.
    """
    if out is None or not (cupy.may_share_memory(out, a)
                           or cupy.may_share_memory(out, b)):
        return _contract_orig(pattern, a, b, alpha, beta, out=out)
    _ALIASED[0] += 1
    tmp = _contract_orig(pattern, a, b, alpha, 0.0)
    if beta == 0:
        out[...] = tmp
    else:
        out *= beta
        out += tmp
    return out


if not _INPLACE and getattr(hess_rks, '_contract_orig', None) is None:
    hess_rks._contract_orig = _contract_orig
    hess_rks.contract = _alias_safe_contract


def stats():
    return {'aliased_contractions': _ALIASED[0],
            'inplace_repaired': hess_rks.contract is _alias_safe_contract}
