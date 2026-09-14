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

**Scope.**  GGA, so hybrids and B3LYP; restricted and unrestricted.  LDA,
meta-GGA, the grid-response path (``hessobj.grid_response``), the VV10 Hessian
(``_get_enlc_deriv2``) and multi-GPU all fall through to GPU4PySCF
unchanged.  LDA is a separate branch upstream with a different factor
convention -- its ``d rho / d R`` carries no factor 2 where the GGA branch's
``_make_dR_rho1`` does, and the GGA branch's compensating ``wv2[:,0] *= .5``
is what makes the two agree -- so covering it means writing that branch too,
which at 0.5 % of a Hessian is not where the next hour goes.
``docs/README_fastxchess.md`` §5 has the algebra, and
``tests/test_xchess_fd.py`` carries an LDA row so that the claim they agree is
checked rather than asserted.

Environment switches (for the ablations):
``FASTXCHESS_OFF=1``         hand both functions back to GPU4PySCF
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
_ATOMSCREEN = os.environ.get('FASTXCHESS_ATOMSCREEN', '1') != '0'
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


def _dR_rho1(ao, c, P, bufs, out=None):
    """``d rho / d R_A`` for every present atom at once, GGA.

    ``hessian.rks._make_dR_rho1`` for all atoms: the AO-resolved products are
    formed for the whole block, and the atom dimension is one contraction.
    ``out`` is where the answer goes; the unrestricted path passes one row of
    a two-spin array, because ``fxc`` couples the channels and both spins'
    ``d rho / d R`` have to be resident at once.
    """
    hg, _ = _index_arrays()
    nsub, ng = ao.shape[1:]
    prod = bufs('prod', (3, 4, nsub, ng))
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
    if out is None:
        out = bufs('dR', (len(P), 3, 4, ng))
    contract('ap,sypg->asyg', P, prod, out=out)
    out *= 2            # for |mu> D <d_R nu|, as _make_dR_rho1 does
    return out


def _wv2(dR_rho1, wf, out):
    """``fxc`` contracted with ``d rho / d R``, with GPU4PySCF's factors."""
    wv = contract('xyg,asxg->asyg', wf, dR_rho1, out=out)
    wv[:, :, 0] *= .5
    return wv


def _deriv1_block(ao, mask, atoms, P, weight, dm_s, mocc_s, ni, xc,
                  v_ip, vmat2d, nao_s, bufs, chunk):
    nsub, ng = ao.shape[1:]
    nocc = mocc_s.shape[1]
    dm_mask = take_last2d(dm_s, mask, out=bufs('dm', (nsub, nsub)))
    c = bufs('c', (4, nsub, ng))
    for m in range(4):
        cupy.dot(dm_mask, ao[m], out=c[m])
    rho = _rho_gga(ao, c[0])
    vxc, fxc = ni.eval_xc_eff(xc, rho, 2, xctype='GGA', spin=0)[1:3]
    rho = None
    wv = cupy.multiply(weight, vxc, out=vxc)
    wv[0] *= .5
    wf = cupy.multiply(weight, fxc, out=fxc)
    add_sparse(v_ip, rks_grad._gga_grad_sum_(
        ao, wv, buf=bufs('gga', (3, nsub, ng)),
        out=bufs('vip', (3, nsub, nsub))), mask)
    vxc = fxc = wv = None

    mo = bufs('mo', (4, nocc, ng))
    contract('xpg,pq->xqg', ao[:4], mocc_s[mask], out=mo)
    wv2 = _wv2(_dR_rho1(ao, c, P, bufs), wf,
               bufs('wv2', (len(atoms), 3, 4, ng)))
    wf = None
    na_all = len(atoms)
    at3 = cupy.asarray((atoms[:, None] * 3 + np.arange(3)).ravel(),
                       dtype=cupy.int64)
    for a0 in range(0, na_all, chunk):
        a1 = min(a0 + chunk, na_all)
        nb = (a1 - a0) * 3
        w = wv2[a0:a1].reshape(nb, 4, ng)
        # the two GEMMs, through cupy.dot rather than `contract`: with the
        # atom and AO indices flattened together these are plain DGEMMs of
        # (natm_blk*3*nao_sub x ngrids) . (ngrids x nocc), which is where
        # 12.8 TFLOP/s comes from.  Forming `aow` and `mow` for them is the
        # expensive part, and it is expensive only if it allocates -- see
        # _Bufs.
        aow = bufs('aow', (nb, nsub, ng))
        contract('ypg,ayg->apg', ao[:4], w, out=aow)
        t = bufs('t1', (nb, nsub, nocc))
        cupy.dot(aow.reshape(-1, ng), mo[0].T, out=t.reshape(-1, nocc))
        mow = bufs('mow', (nb, nocc, ng))
        contract('yqg,ayg->aqg', mo, w, out=mow)
        t2 = bufs('t2', (nb, nocc, nsub))
        cupy.dot(mow.reshape(-1, ng), ao[0].T, out=t2.reshape(-1, nsub))
        t += t2.transpose(0, 2, 1)
        # vmat[atom, x, mask, :] += t, as a flat row scatter
        rows = (at3[a0 * 3:a1 * 3, None] * nao_s
                + mask[None, :].astype(cupy.int64)).ravel()
        cupyx.scatter_add(vmat2d, rows, t.reshape(-1, nocc))
        rows = None


def _deriv1(hessobj, mo_coeff, mo_occ, max_memory, lay, grids, ni):
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
    # how many atoms one pass may carry: `aow` and `mow` are the only scratch
    # whose size grows with it, and they are allocated once for that many
    per_atom = 3 * (ns_max + nocc) * blk * 8
    chunk = max(1, min(na_max,
                       int(MEM_FRACTION * get_avail_mem() / per_atom)))
    bufs = _Bufs(
        dm=ns_max * ns_max, c=4 * ns_max * blk, gga=3 * ns_max * blk,
        vip=3 * ns_max * ns_max, mo=4 * nocc * blk,
        prod=12 * ns_max * blk, gather=3 * ns_max * blk,
        tmp=3 * ns_max * blk, dR=na_max * 12 * blk, wv2=na_max * 12 * blk,
        aow=chunk * 3 * ns_max * blk, mow=chunk * 3 * nocc * blk,
        t1=chunk * 3 * ns_max * nocc, t2=chunk * 3 * ns_max * nocc)
    log.debug1(f'fastxchess deriv1: nao_sub_max {ns_max}, natm_local_max '
               f'{na_max}, {chunk} atoms a pass, {bufs.nbytes/GB:.2f} GB '
               f'scratch')
    it = iter(blocks)
    for ao, mask, weight, _ in ni.block_loop(lay._sorted_mol, grids, nao, 2,
                                             None):
        atoms, rows = next(it)
        assert len(rows) == len(mask)
        P = lay.atom_matrix(atoms, rows, len(mask))
        _deriv1_block(ao, mask, atoms, P, weight, dm_s, mocc_s, ni, mf.xc,
                      v_ip, vmat2d, nao_s, bufs, chunk)
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


def _deriv2_block(ao, mask, atoms, P, weight, dm_s, ni, xc, ipip, vmat_dm1d,
                  nao_s, bufs):
    nsub, ng = ao.shape[1:]
    dm_mask = take_last2d(dm_s, mask, out=bufs('dm', (nsub, nsub)))
    c = bufs('c', (4, nsub, ng))
    for m in range(4):
        cupy.dot(dm_mask, ao[m], out=c[m])
    rho = _rho_gga(ao, c[0])
    vxc, fxc = ni.eval_xc_eff(xc, rho, 2, xctype='GGA', spin=0)[1:3]
    rho = None
    wv1 = cupy.multiply(weight, vxc, out=vxc)
    wv1[0] *= .5
    wf = cupy.multiply(weight, fxc, out=fxc)
    aow = rks_grad._make_dR_dao_w(ao, wv1, out=bufs('gga', (3, nsub, ng)))
    ip = bufs('vip', (3, 3, nsub, nsub))
    for d1 in range(3):
        for d2 in range(3):
            contract('ik,lk->il', aow[d2], ao[1 + d1], out=ip[d1, d2])
    add_sparse(ipip, ip, mask)
    vxc = fxc = wv1 = aow = ip = None

    na = len(atoms)
    wv2 = _wv2(_dR_rho1(ao, c, P, bufs), wf, bufs('wv2', (na, 3, 4, ng)))
    wf = None
    # W[x][y,j,g]: the components _make_dR_dao_w would combine, times the
    # density product -- built once per block rather than once per atom, and
    # one direction at a time so that nothing of size (3,4,nao_sub,ngrids)
    # is ever resident
    _, selg = _index_arrays()
    t = bufs('t1', (3, na, 3, nsub))          # [x, atom, direction, AO]
    W = bufs('W', (4, nsub, ng))
    for x in range(3):
        cupy.take(ao, selg[x], axis=0, out=W)
        W *= c[0]
        contract('yjg,asyg->asj', W, wv2, out=t[x])
    # U[y][z,j,g] = (d_y ao)[j,g] c[z,j,g]
    U = W
    for y in range(3):
        cupy.multiply(ao[y + 1], c[:4], out=U)
        contract('zjg,aszg->asj', U, wv2, beta=1., out=t[y])
    W = U = None
    tT = bufs('t2', (na, 3, 3, nsub))
    tT[:] = t.transpose(1, 2, 0, 3)
    t = tT
    rows = cupy.asarray(
        ((atoms[:, None, None] * 3 + np.arange(3)[None, :, None]) * 3
         + np.arange(3)[None, None, :]).ravel(), dtype=cupy.int64)
    idx = (rows[:, None] * nao_s + mask[None, :].astype(cupy.int64)).ravel()
    cupyx.scatter_add(vmat_dm1d, idx, t.reshape(-1))
    t = rows = idx = None


def _deriv2(hessobj, mo_coeff, mo_occ, max_memory, lay, grids, ni):
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
    bufs = _Bufs(
        dm=ns_max * ns_max, c=4 * ns_max * blk, gga=3 * ns_max * blk,
        vip=9 * ns_max * ns_max, prod=12 * ns_max * blk,
        gather=3 * ns_max * blk, tmp=3 * ns_max * blk,
        dR=na_max * 12 * blk, wv2=na_max * 12 * blk, W=4 * ns_max * blk,
        t1=9 * na_max * ns_max, t2=9 * na_max * ns_max)
    log.debug1(f'fastxchess deriv2: nao_sub_max {ns_max}, natm_local_max '
               f'{na_max}, {bufs.nbytes/GB:.2f} GB scratch')
    it = iter(blocks)
    for ao, mask, weight, _ in ni.block_loop(lay._sorted_mol, grids, nao, 2,
                                             None):
        atoms, rows = next(it)
        assert len(rows) == len(mask)
        P = lay.atom_matrix(atoms, rows, len(mask))
        _deriv2_block(ao, mask, atoms, P, weight, dm_s, ni, mf.xc, ipip,
                      vmat_dm1d, nao_s, bufs)
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

def _rho_gga_into(ao, c0, out):
    """``_rho_gga`` into a caller-owned row of a ``(2, 4, ngrids)`` array."""
    out[:] = _rho_gga(ao, c0)
    return out


def _c_and_rho_u(ao, mask, dm_s, bufs):
    """``D_s . ao`` for both spins, and the two spin densities.

    The block's masked density matrix is formed one spin at a time into the
    same scratch, because only the product ``c`` has to survive.
    """
    nsub, ng = ao.shape[1:]
    c = bufs('c', (2, 4, nsub, ng))
    rho = cupy.empty((2, 4, ng))
    for s in range(2):
        dm_mask = take_last2d(dm_s[s], mask, out=bufs('dm', (nsub, nsub)))
        for m in range(4):
            cupy.dot(dm_mask, ao[m], out=c[s, m])
        _rho_gga_into(ao, c[s, 0], rho[s])
    return c, rho


def _wv2_u(dR, wf, out):
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
    return out


def _dR_and_wv2_u(ao, c, P, wf, bufs, na):
    """``d rho / d R`` for both spins, and ``fxc`` contracted with it."""
    ng = ao.shape[2]
    dR = bufs('dR', (2, na, 3, 4, ng))
    for s in range(2):
        _dR_rho1(ao, c[s], P, bufs, out=dR[s])
    return _wv2_u(dR, wf, bufs('wv2', (2, na, 3, 4, ng)))


def _deriv1_block_u(ao, mask, atoms, P, weight, dm_s, mocc_s, ni, xc,
                    v_ip, vmat2d, nao_s, bufs, chunk):
    nsub, ng = ao.shape[1:]
    c, rho = _c_and_rho_u(ao, mask, dm_s, bufs)
    vxc, fxc = ni.eval_xc_eff(xc, rho, 2, xctype='GGA', spin=1)[1:3]
    rho = None
    wv = cupy.multiply(weight, vxc, out=vxc)
    wv[:, 0] *= .5
    wf = cupy.multiply(weight, fxc, out=fxc)
    for s in range(2):
        add_sparse(v_ip[s], rks_grad._gga_grad_sum_(
            ao, wv[s], buf=bufs('gga', (3, nsub, ng)),
            out=bufs('vip', (3, nsub, nsub))), mask)
    vxc = fxc = wv = None

    na_all = len(atoms)
    wv2 = _dR_and_wv2_u(ao, c, P, wf, bufs, na_all)
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
            w = wv2[s, a0:a1].reshape(nb, 4, ng)
            aow = bufs('aow', (nb, nsub, ng))
            contract('ypg,ayg->apg', ao[:4], w, out=aow)
            t = bufs('t1', (nb, nsub, nocc))
            cupy.dot(aow.reshape(-1, ng), mo[0].T, out=t.reshape(-1, nocc))
            mow = bufs('mow', (nb, nocc, ng))
            contract('yqg,ayg->aqg', mo, w, out=mow)
            t2 = bufs('t2', (nb, nocc, nsub))
            cupy.dot(mow.reshape(-1, ng), ao[0].T, out=t2.reshape(-1, nsub))
            t += t2.transpose(0, 2, 1)
            rows = (at3[a0 * 3:a1 * 3, None] * nao_s
                    + mask[None, :].astype(cupy.int64)).ravel()
            cupyx.scatter_add(vmat2d[s], rows, t.reshape(-1, nocc))
            rows = None


def _deriv1_u(hessobj, mo_coeff, mo_occ, max_memory, lay, grids, ni):
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
    # `aow` and `mow` are the only scratch whose size grows with the atom
    # block; `dR` and `wv2` carry a spin index the restricted path does not.
    per_atom = 3 * (ns_max + nocc_max) * blk * 8
    chunk = max(1, min(na_max,
                       int(MEM_FRACTION * get_avail_mem() / per_atom)))
    bufs = _Bufs(
        dm=ns_max * ns_max, c=2 * 4 * ns_max * blk, gga=3 * ns_max * blk,
        vip=3 * ns_max * ns_max, mo=4 * nocc_max * blk,
        prod=12 * ns_max * blk, gather=3 * ns_max * blk,
        tmp=3 * ns_max * blk, dR=2 * na_max * 12 * blk,
        wv2=2 * na_max * 12 * blk,
        aow=chunk * 3 * ns_max * blk, mow=chunk * 3 * nocc_max * blk,
        t1=chunk * 3 * ns_max * nocc_max, t2=chunk * 3 * ns_max * nocc_max)
    log.debug1(f'fastxchess deriv1 (uks): nao_sub_max {ns_max}, '
               f'natm_local_max {na_max}, {chunk} atoms a pass, '
               f'{bufs.nbytes/GB:.2f} GB scratch')
    it = iter(blocks)
    for ao, mask, weight, _ in ni.block_loop(lay._sorted_mol, grids, nao, 2,
                                             None):
        atoms, rows = next(it)
        assert len(rows) == len(mask)
        P = lay.atom_matrix(atoms, rows, len(mask))
        _deriv1_block_u(ao, mask, atoms, P, weight, dm_s, mocc_s, ni, mf.xc,
                        v_ip, vmat2d, nao_s, bufs, chunk)
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
                    vmat_dm1d, nao_s, bufs):
    nsub, ng = ao.shape[1:]
    c, rho = _c_and_rho_u(ao, mask, dm_s, bufs)
    vxc, fxc = ni.eval_xc_eff(xc, rho, 2, xctype='GGA', spin=1)[1:3]
    rho = None
    wv1 = cupy.multiply(weight, vxc, out=vxc)
    wv1[:, 0] *= .5
    wf = cupy.multiply(weight, fxc, out=fxc)
    for s in range(2):
        aow = rks_grad._make_dR_dao_w(ao, wv1[s],
                                      out=bufs('gga', (3, nsub, ng)))
        ip = bufs('vip', (3, 3, nsub, nsub))
        for d1 in range(3):
            for d2 in range(3):
                contract('ik,lk->il', aow[d2], ao[1 + d1], out=ip[d1, d2])
        add_sparse(ipip[s], ip, mask)
    vxc = fxc = wv1 = aow = ip = None

    na = len(atoms)
    wv2 = _dR_and_wv2_u(ao, c, P, wf, bufs, na)
    wf = None
    _, selg = _index_arrays()
    rows = cupy.asarray(
        ((atoms[:, None, None] * 3 + np.arange(3)[None, :, None]) * 3
         + np.arange(3)[None, None, :]).ravel(), dtype=cupy.int64)
    idx = (rows[:, None] * nao_s + mask[None, :].astype(cupy.int64)).ravel()
    for s in range(2):
        t = bufs('t1', (3, na, 3, nsub))      # [x, atom, direction, AO]
        W = bufs('W', (4, nsub, ng))
        for x in range(3):
            cupy.take(ao, selg[x], axis=0, out=W)
            W *= c[s, 0]
            contract('yjg,asyg->asj', W, wv2[s], out=t[x])
        U = W
        for y in range(3):
            cupy.multiply(ao[y + 1], c[s, :4], out=U)
            contract('zjg,aszg->asj', U, wv2[s], beta=1., out=t[y])
        W = U = None
        tT = bufs('t2', (na, 3, 3, nsub))
        tT[:] = t.transpose(1, 2, 0, 3)
        cupyx.scatter_add(vmat_dm1d[s], idx, tT.reshape(-1))
        t = tT = None
    rows = idx = None


def _deriv2_u(hessobj, mo_coeff, mo_occ, max_memory, lay, grids, ni):
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
    bufs = _Bufs(
        dm=ns_max * ns_max, c=2 * 4 * ns_max * blk, gga=3 * ns_max * blk,
        vip=9 * ns_max * ns_max, prod=12 * ns_max * blk,
        gather=3 * ns_max * blk, tmp=3 * ns_max * blk,
        dR=2 * na_max * 12 * blk, wv2=2 * na_max * 12 * blk,
        W=4 * ns_max * blk, t1=9 * na_max * ns_max, t2=9 * na_max * ns_max)
    log.debug1(f'fastxchess deriv2 (uks): nao_sub_max {ns_max}, '
               f'natm_local_max {na_max}, {bufs.nbytes/GB:.2f} GB scratch')
    it = iter(blocks)
    for ao, mask, weight, _ in ni.block_loop(lay._sorted_mol, grids, nao, 2,
                                             None):
        atoms, rows = next(it)
        assert len(rows) == len(mask)
        P = lay.atom_matrix(atoms, rows, len(mask))
        _deriv2_block_u(ao, mask, atoms, P, weight, dm_s, ni, mf.xc, ipip,
                        vmat_dm1d, nao_s, bufs)
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
    if mf._numint._xc_type(mf.xc) != 'GGA':
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
    return lay, grids, ni


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
