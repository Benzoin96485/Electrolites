"""The unrestricted twins of W17 and W18.

``docs/DF_WORK_LIST.md`` **W25**.  ``df/hessian/uhf.py`` has its own
``_jk_energy_per_atom`` and ``_get_veff`` -- the ``ip2`` term of
``partial_hess_elec`` and the ``h1`` term of ``make_h1`` for an open-shell
density-fitted Hessian, reached by ``df/hessian/{uhf,uks}.py`` through a
module-global lookup on ``df/hessian/uhf.py`` -- and they are the restricted
functions with a spin index on every occupied quantity: the occupied factors
are ``(2, nao, nocc)`` with ``nocc = max(nocca, noccb)`` and the beta columns
past ``noccb`` zero, every ``(i, j)`` pair becomes ``(n, i, j)``, and the
exchange factor is ``-k`` on each spin density instead of ``-k/2`` on the
total.  Nothing else differs, except that upstream's unrestricted functions
have no occupied batching.

So ``fastdfhessip``'s items transplant, as the entry said.  This module is
the unrestricted half of that module, loaded and patched by it; the switches
are the same ``FASTDFHESSIP_*`` ones and mean the same thing:

* ``ip2``: **item 1** (the launch constant, on every Rys kernel), **item 2**
  (the per-atom occupied block as Gram matrices -- per spin, since
  ``sum_{n,i,j} S_A S_B`` is a sum over the two spins of the restricted
  sum), **item 3** (the first ``ipaux`` pass folded into the second), **items
  9-11** (``dsyrk`` for the ``h_aux`` block's Gram matrices and D4's ``R``,
  and the block's temporaries in the second pass's integral buffers), and,
  for a pure functional (``k = 0``), upstream's own hand-off to the
  restricted ``_j_energy_per_atom`` on the total density -- which is
  ``fastdfhessip``'s (items 1 and 7) here.
* ``h1``: **item 1**, **item 4** (the exchange half's per-atom intermediate
  hoisted, one per spin) and **item 5** (the Coulomb half's per-atom loop as
  one GEMM against the atom indicator).  The Coulomb ``vhf1`` term, which
  upstream forms once per spin from the same ``auxvec``, is formed once.
  **Item 12** (the per-atom exchange term on the occupied side, chosen by a
  shape score) started here and was carried back to the restricted function
  by W30, which also added **item 14** to both (no exchange contraction at
  all when ``k_factor = 0``).

Item 8 (the metric as its Cholesky factor, W24) is not carried over: it
lives in ``_chol_d4_d5``, whose five contractions would all need the spin
index, and W24 measured it at 1.05x-1.06x of the call below 42 atoms.
"""
import ctypes

import numpy as np
import cupy as cp

from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, ndarray, transpose_sum, fill_symmetric)
from gpu4pyscf.df.int3c2e_bdiv import (
    _split_l_ctr_pattern, get_ao_pair_loc, libvhf_rys, int2c2e_ip1,
    _check_rsh_factors, int3c2e_scheme_ip1, int3c2e_scheme_ipaux)
from gpu4pyscf.df.hessian import uhf as DFUHESS
from gpu4pyscf.df.hessian.rhf import (
    _factorize_dm, _factorize_j2c, _bas_atom_labels, _argsort_aux_by_atom,
    int3c2e_scheme_ip2, _allocate, _balance_batch_size)
from gpu4pyscf.hessian.rhf import _aggregate_to_atoms

from ._dfaux import _spmult
from . import _cuplan
from . import fastdfhessip as H
from .fastdfhessip import (_env, _S, _Stages, _ip1_evaluator, _avail,
                           _atom_indicator, _usable)

_orig_jk_energy_per_atom = DFUHESS._jk_energy_per_atom
_orig_get_veff = DFUHESS._get_veff


# --------------------------------------------------------------------------
# the ip2 term of partial_hess_elec
# --------------------------------------------------------------------------
def _jk_energy_per_atom(int3c2e_opt, dm, j_factor=1, k_factor=1, omega=None,
                        verbose=None):
    if not _usable():
        _S['declined'] += 1
        return DFUHESS._jk_energy_per_atom_orig(int3c2e_opt, dm, j_factor,
                                                k_factor, omega, verbose)
    assert dm.ndim == 3
    if k_factor == 0:
        # upstream's own branch: a pure functional's ip2 term is the
        # restricted Coulomb-only one on the total density
        _S['ip2_u'] = _S.get('ip2_u', 0) + 1
        return H._j_energy_per_atom(int3c2e_opt, dm[0] + dm[1],
                                    verbose) * j_factor
    _S['ip2_u'] = _S.get('ip2_u', 0) + 1
    spmult = int(_env('FASTDFHESSIP_SPMULT', '1'))
    oosum = _env('FASTDFHESSIP_OOSUM', 'gram')
    fuse = _env('FASTDFHESSIP_FUSE', '1') != '0'
    st = _Stages('ip2 U')

    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    omega, lr_factor, sr_factor = _check_rsh_factors(mol, omega, None, None)

    dm_factor_l = _factorize_dm(mol, dm)
    nao, nocc = dm_factor_l.shape[1:]

    natm = mol.natm
    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=True)[0]
    pair_addresses = cp.asarray(pair_addresses, dtype=np.int32)
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)
    aux_loc = auxmol.ao_loc
    naux = int(aux_loc[-1])
    largest_shell_nao = int((aux_loc[1:] - aux_loc[:-1]).max())

    cp.get_default_memory_pool().free_all_blocks()
    mem_avail = _avail()
    word_avail = mem_avail // 8
    word_avail -= 2 * 3 * naux * nocc**2  # j3c_oo1
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
    buf1 = cp.empty((2, blksize, nocc, nao))
    j3c_oo = cp.empty((naux, 2, nocc, nocc))
    for kbatch in range(num_aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:, :, :dk]
            fill_symmetric(compressed, pair_addresses, nao, k0, k1, out=j3c)
            # one spin at a time, in upstream's order.  This is the one
            # stage whose rounding the answer does not forgive: with both
            # spins in one contraction the long-range operator's ip2 term
            # moves by 1.1e-10 relative on a triplet (the metric that
            # dm_oo goes through next is the ill-conditioned one), and with
            # this stage as upstream's and every other one reassociated it
            # agrees to 4.6e-14
            tmp = ndarray((nocc, nao, dk), buffer=buf1)
            for n in range(2):
                contract('pqr,pi->iqr', j3c, dm_factor_l[n], out=tmp)
                contract('iqr,qj->rij', tmp, dm_factor_l[n],
                         out=j3c_oo[aux0:aux1, n])
    j3c_full = buf = buf1 = eval_j3c = j3c = tmp = compressed = None
    st('A j3c_oo')

    metric_w, metric_v = _factorize_j2c(auxmol, aux_sorting, omega)
    metric_size = metric_v.shape[1]
    # item 13: the contractions whose output can pass cuTENSOR's 2**29
    # elements go to cuBLAS when cuTENSOR would refuse them
    # (`docs/DF_WORK_LIST.md` W21, W27) -- upstream's own function stops
    # here, or at the pseudo-density, on a 113-atom open shell
    dm_oo = cp.empty((metric_size, 2, nocc, nocc))
    _mm('uv,unij->vnij', metric_v, j3c_oo, dm_oo, 'T', 'N')
    dm_oo /= metric_w[:, None, None, None]
    _mm('uv,vnij->unij', metric_v, dm_oo, j3c_oo, 'N', 'N')
    dm_oo = j3c_oo
    j3c_oo = None
    auxvec = None
    if j_factor != 0:
        auxvec = cp.einsum('rnii->r', dm_oo)
    st('B metric')

    # (00|0)(2|0)(0|00)
    ejk = _ejk_ip2(int3c2e_opt, dm_factor_l, dm_oo, auxvec, j_factor,
                   k_factor, omega, lr_factor, sr_factor, pair_addresses,
                   nao_pair, batch_size, blksize, spmult)
    ejk = ejk + ejk.transpose(1, 0, 3, 2)
    ejk *= 2 * 2 * .5
    st('C ip2 kernel')

    atm_ao_counts = np.bincount(_bas_atom_labels(mol))
    aoslices = np.append(0, atm_ao_counts.cumsum())
    nao_on_atom = atm_ao_counts.max()
    aux_atm_labels = _bas_atom_labels(auxmol, aux_sorting)
    atm_ind = _atom_indicator(aoslices)
    _aux_sorting = cp.asarray(aux_sorting)

    eval_ipaux = _ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ipaux(omega, 27), batch_size,
        'fill_int3c2e_ipaux', omega, spmult)[0]
    eval_ip1 = _ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ip1(omega, 27), batch_size,
        'fill_int3c2e_ip1', omega, spmult)[0]

    j3c_oo1 = cp.empty((3, naux, 2, nocc, nocc))  # = (1|00)

    # The second pass's sizes, worked out before the first pass so that
    # item 3 can tell whether there will be one chunk.  Upstream derives
    # them from the memory free after the first pass and the h_aux block,
    # which free what they allocate except j2c_10v; the arithmetic is
    # upstream's, on the same words.
    cp.get_default_memory_pool().free_all_blocks()
    word_avail = _avail() // 8
    blksize1 = min(naux, word_avail // (nao**2*8) * 8)
    assert blksize1 > 0, 'Insufficient GPU memory'
    blksize1 = min(blksize1, batch_size)
    gram = oosum != 'transpose'
    plan = _plan_v(word_avail, naux, nao, nocc, nao_pair, batch_size,
                   nao_on_atom, natm, metric_size, gram)
    do_fuse = False
    carve = False
    plan_f = (None,)
    if fuse:
        # as in fastdfhessip: fused, the h_aux block runs while the second
        # pass's buffers are held, so the plan is made against what its
        # temporaries leave and taken only if it costs no chunk; item 11
        # puts most of them in the integral buffers the walk is done with
        carve = (_env('FASTDFHESSIP_CARVE', '1') != '0' and
                 6 * batch_size * nao_pair >=
                 H._h_aux_words(naux, metric_size))
        margin = (10 if carve else 24) * naux**2
        try:
            plan_f = _plan_v(word_avail - margin, naux, nao, nocc, nao_pair,
                             batch_size, nao_on_atom, natm, metric_size, gram)
        except AssertionError:
            plan_f = (0,)
        if plan_f[0] > 0 and (-(-metric_size // plan_f[0]) <=
                              -(-metric_size // plan[0])):
            plan, do_fuse = plan_f, True
    aux_batch_size, aux_blksize, blksize2 = plan
    st.note(free_gb=round(word_avail * 8e-9, 1), plan=plan[0],
            plan_fused=plan_f[0], nao=nao, nocc=nocc, naux=naux,
            metric_size=metric_size, nao_pair=nao_pair,
            batch_size=batch_size, num_aux_batches=num_aux_batches,
            vchunk=aux_batch_size, blksize2=blksize2, fused=do_fuse)
    _S['vchunks'] = (metric_size + aux_batch_size - 1) // aux_batch_size
    if do_fuse:
        _S['fused'] += 1

    h_ao_aux = cp.zeros((natm, naux, 3, 3))
    ejk_ao = cp.zeros((natm, natm, 3, 3))
    auxvec_ipauxp = None

    def h_aux(work=None):
        # upstream's code between the passes; fastdfhessip's copy of it
        # runs on the spin index folded into the first occupied one, with
        # -.5 (2k) = -k
        nonlocal auxvec_ipauxp
        if j_factor != 0:
            # the (1|00) half of auxvec_ipauxp; the block adds the rest
            auxvec_ipauxp = cp.einsum('xrnii->xr', j3c_oo1)
        o1p, parts = H._h_aux_block(
            auxmol, omega, _aux_sorting, metric_v, metric_w,
            dm_oo.reshape(naux, 2*nocc, nocc),
            j3c_oo1.reshape(3, naux, 2*nocc, nocc), naux, metric_size,
            2*k_factor, j_factor, 0, auxvec_ipauxp, auxvec, natm,
            aux_atm_labels, work=work)
        return o1p.reshape(3, naux, 2, nocc, nocc), parts[0], parts[1]

    if not do_fuse:
        _ipaux_pass(eval_ipaux, num_aux_batches, blksize1, nao, nao_pair,
                    batch_size, nocc, pair_addresses, dm_factor_l, j3c_oo1)
        st('D1 ipaux pass')
        j3c_oo1p, e_aux, j2c_10v = h_aux()
        ejk_ao += e_aux
        st('D2 h_aux')

    j3c_buf = cp.empty((3, aux_batch_size, 2, nao, nocc))
    work = cp.empty(
        max(0 if gram else natm*3*2*nocc*nocc*2*aux_blksize,
            3*nao_pair*batch_size * 2 + (nocc*2+nao)*nao * blksize2,
            3*naux*2*nao_on_atom*nocc * 2,
            3*naux*2*nao_on_atom*nocc + 3*aux_batch_size*naux))
    for v0, v1 in lib.prange(0, metric_size, aux_batch_size):
        dv = v1 - v0
        j3c_100 = ndarray((3, dv, 2, nao, nocc), buffer=j3c_buf)
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
                # item 3: the first pass, on this batch's ipaux integrals
                # before they are folded into dj
                _ipaux_batch(compressed_dk, aux1, naux_in_batch, blksize2,
                             j3c_full, work1, nao, nocc, pair_addresses,
                             dm_factor_l, j3c_oo1)
                st('D1 ipaux (fused)')
            # (di/dr j|k) + (i dj/dr|k) + (i j|dk/dr) = 0
            compressed_dk += compressed_di
            compressed_dj = compressed_dk
            compressed_di *= -1
            for k0, k1 in lib.prange(0, naux_in_batch, blksize2):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                j3c = j3c_full[:, :, :dk]
                tmp = _allocate((2, nao, nocc, dk), work1)[0]
                for x in range(3):
                    j3c[j_addr, i_addr] = compressed_dj[x, :, k0:k1]
                    j3c[i_addr, j_addr] = compressed_di[x, :, k0:k1]
                    tmp = contract('pqr,nqi->npir', j3c, dm_factor_l, out=tmp)
                    _mm('rs,npir->snpi', metric_v[aux0:aux1, v0:v1], tmp,
                        j3c_100[x], 'T', 'T', beta=1.)
            st('D3 j3c_100')
        j3c_full = buf0 = buf1 = tmp = None
        compressed_di = compressed_dj = compressed_dk = None

        if do_fuse and v0 == 0:
            j3c_oo1p, e_aux, j2c_10v = h_aux(work if carve else None)
            ejk_ao += e_aux
            st('D2 h_aux')

        if gram:
            _oo_gram(j3c_100, v0, dv, work, natm, nocc, atm_ind, dm_factor_l,
                     metric_w, metric_v, k_factor, j_factor, ejk_ao,
                     h_ao_aux, j2c_10v, auxvec, auxvec_ipauxp, naux)
        else:
            for k0, k1 in lib.prange(0, dv, aux_blksize):
                _oo_atm_block(j3c_100, k0, k1, v0, work, natm, nocc, aoslices,
                              dm_factor_l, metric_w, metric_v, k_factor,
                              j_factor, ejk_ao, h_ao_aux, j2c_10v, auxvec,
                              auxvec_ipauxp, naux)
        st('D4 oo_atm')

        for i, (p0, p1) in enumerate(zip(aoslices[:-1], aoslices[1:])):
            # (10|0)(1|0)(0|00) + (10|0)(0|0)(1|00)
            tmp0, work1 = _allocate((3, naux, 2, p1-p0, nocc), work)
            tmp1, work1 = _allocate((3, naux, 2, p1-p0, nocc), work1)
            contract('xrnuj,sr->xsnuj', j3c_100[:, :, :, p0:p1],
                     metric_v[:, v0:v1]/metric_w[v0:v1], out=tmp0)
            contract('ysnij,nui->ysnuj', j3c_oo1p, dm_factor_l[:, p0:p1],
                     out=tmp1)
            contract('xsnuj,ysnuj->sxy', tmp0, tmp1, -k_factor*2, 1,
                     h_ao_aux[i])
            # (10|0)(0|1)(0|00)
            tmp0, work1 = _allocate((naux, 2, p1-p0, nocc), work)
            tmp1, work1 = _allocate((3, naux, v1-v0), work1)
            contract('snij,nui->snuj', dm_oo, dm_factor_l[:, p0:p1], out=tmp0)
            contract('xrnuj,snuj->xsr', j3c_100[:, :, :, p0:p1], tmp0,
                     out=tmp1)
            tmp1 /= metric_w[v0:v1]
            contract('xsr,ysr->sxy', tmp1, j2c_10v[:, :, v0:v1], k_factor*2,
                     1, h_ao_aux[i])
        st('D5 per-atom')
    j3c_buf = work = work1 = tmp0 = tmp1 = j3c = j3c_100 = None
    j2c_10v = j3c_oo1p = j3c_oo1 = None

    ejk_ao_aux = _aggregate_to_atoms(h_ao_aux, natm, aux_atm_labels, axis=1)
    ejk += ejk_ao_aux
    ejk += ejk_ao_aux.transpose(1, 0, 3, 2)
    ejk += ejk_ao
    st('E aggregate')
    st.report()
    log.timer_debug1('_jk_energy_per_atom UHF (fastdfhessip)', *t0)
    return ejk


def _mm(pattern, a, b, out, ta, tb, alpha=1., beta=0.):
    """``contract(pattern, a, b, alpha, beta, out=out)`` for a pattern that
    is one GEMM on the flattened operands -- ``op_a(a2) op_b(b2)`` with
    ``a2``/``b2`` the operands as 2-d matrices in the order the pattern's
    output wants them -- through cuBLAS when cuTENSOR will not plan it."""
    if _cuplan.planable(pattern, a, b, out):
        return contract(pattern, a, b, alpha, beta, out=out)
    _S['plan_gemm'] = _S.get('plan_gemm', 0) + 1
    a2 = cp.ascontiguousarray(a).reshape(a.shape[0], -1)
    b2 = b.reshape(b.shape[0], -1) if tb == 'N' else \
        b.reshape(-1, b.shape[-1])
    out2 = out.reshape(out.shape[0], -1)
    cp.cublas.gemm(ta, tb, a2, b2, out=out2, alpha=alpha, beta=beta)
    return out


def _plan_v(word_avail, naux, nao, nocc, nao_pair, batch_size, nao_on_atom,
            natm, metric_size, gram):
    """Upstream's sizing of the unrestricted second pass.

    One change: upstream asserts that one auxiliary function's per-atom
    occupied block fits (``natm 12 nocc^2`` words), which item 2's Gram form
    never builds -- so with it on, that block's size is not a limit.
    """
    word_avail -= 4 * naux**2  # j2c_10v, which the h_aux block leaves
    word_avail -= 3 * batch_size * nao_pair * 2  # eval_ip1, eval_ipaux
    aux_unit1 = 3*2*nao*nocc + 3*naux  # j3c_100
    aux_batch_size = (min(int(word_avail*.85),
                          word_avail-12*naux*nao_on_atom*nocc)
                      // (aux_unit1 * 8) * 8)
    assert aux_batch_size > 0, 'Insufficient GPU memory'
    forced = int(_env('FASTDFHESSIP_VCHUNK', '0'))
    if forced > 0:
        aux_batch_size = min(aux_batch_size, forced)
    aux_batch_size = _balance_batch_size(aux_batch_size, metric_size, 16)
    word_avail -= aux_unit1 * aux_batch_size
    aux_unit2 = natm*3*2*nocc*nocc * 2  # j3c_oo_atm
    aux_blksize = word_avail // (aux_unit2 * 8) * 8
    if not gram:
        assert aux_blksize > 0, 'Insufficient GPU memory'
    aux_blksize = max(8, min(aux_blksize, aux_batch_size)) if gram else \
        min(aux_blksize, aux_batch_size)
    blksize = word_avail // ((nocc*2+nao)*nao * 16) * 16
    assert blksize > 0, 'Insufficient GPU memory'
    blksize = min(blksize, batch_size)
    return aux_batch_size, aux_blksize, blksize


def _ipaux_pass(eval_ipaux, num_aux_batches, blksize, nao, nao_pair,
                batch_size, nocc, pair_addresses, dm_factor_l, j3c_oo1):
    """Upstream's first pass: ``(1|00)``, both spins."""
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((3, nao_pair, batch_size))
    buf1 = cp.empty(2 * blksize * nocc * nao)
    aux0 = 0
    for kbatch in range(num_aux_batches):
        compressed = eval_ipaux(kbatch, out=buf)
        naux_in_batch = compressed.shape[-1]
        _ipaux_batch(compressed, aux0, naux_in_batch, blksize, j3c_full, buf1,
                     nao, nocc, pair_addresses, dm_factor_l, j3c_oo1)
        aux0 += naux_in_batch


def _ipaux_batch(compressed, aux0, naux_in_batch, blksize, j3c_full, work,
                 nao, nocc, pair_addresses, dm_factor_l, j3c_oo1):
    for k0, k1 in lib.prange(0, naux_in_batch, blksize):
        dk = k1 - k0
        a0, a1 = aux0 + k0, aux0 + k1
        j3c = j3c_full[:, :, :dk]
        tmp = ndarray((2, nocc, nao, dk), buffer=work)
        for i in range(3):
            fill_symmetric(compressed[i], pair_addresses, nao, k0, k1,
                           out=j3c)
            contract('pqr,npi->niqr', j3c, dm_factor_l, out=tmp)
            # d/dX = -d/dr
            contract('niqr,nqj->rnij', tmp, dm_factor_l, alpha=-1,
                     out=j3c_oo1[i, a0:a1])
    # the (10|0) scatter that may share this buffer writes only the stored
    # pairs and relies on the rest being zero
    j3c_full[:] = 0.


def _oo_gram(j3c_100, v0, dv, work, natm, nocc, atm_ind, dm_factor_l,
             metric_w, metric_v, k_factor, j_factor, ejk, h_ao_aux, j2c_10v,
             auxvec, auxvec_ipauxp, naux):
    """Item 2 on both spins: ``fastdfhessip._oo_gram`` with a spin sum.

    Upstream's ``sum_{r,n,i,j} S[A,x,r,n,i,j] S[B,y,r,n,i,j] / w_r`` is the
    restricted sum once per spin, so each spin contributes its own
    ``P_n o R_n + Q_n`` and the atom sums are taken once on the total.  The
    Coulomb half's trace is summed over spins before it is squared, as
    upstream's ``einsum('pxrnii->pxr')`` does.
    """
    nao = dm_factor_l.shape[1]
    need = 3*nao*nocc + 6*nao*nao
    if work.size < need:
        work = cp.empty(need)
    rq = (H._oo_gram_rq_syrk if _env('FASTDFHESSIP_D4', 'syrk') == 'syrk'
          else H._oo_gram_rq)
    acc = None
    for n in range(2):
        R, Q = rq(j3c_100[:, :, n], v0, dv, work, nocc, dm_factor_l[n],
                  metric_w)
        R *= dm_factor_l[n].dot(dm_factor_l[n].T)
        R += Q
        Q = None
        if acc is None:
            acc = R
        else:
            acc += R
        R = None
    if j_factor != 0:
        kb = max(1, min(dv, int(2e8) // (3*naux)))
        for k0, k1 in lib.prange(0, dv, kb):
            dk = k1 - k0
            # the trace of S, over both spins
            d = contract('xrnui,nui->xru', j3c_100[:, k0:k1], dm_factor_l)
            a = contract('au,xru->axr', atm_ind, d, alpha=2.)
            aw = a / metric_w[v0+k0:v0+k1]
            contract('pxr,qyr->pqxy', aw, a, j_factor, beta=1, out=ejk)
            tmp = cp.empty((3, naux, dk))
            cp.multiply(j2c_10v[:, :, v0+k0:v0+k1], auxvec[:, None], out=tmp)
            contract('ys,sr->ysr', auxvec_ipauxp, metric_v[:, v0+k0:v0+k1],
                     beta=-1, out=tmp)
            contract('pxr,ysr->psxy', aw, tmp, j_factor, 1, h_ao_aux)
            d = a = aw = tmp = None
    t = contract('au,xyuv->axyv', atm_ind, acc)
    # -k from the Coulomb operator on each spin, 2 from S = X + X^T
    contract('axyv,bv->abxy', t, atm_ind, -k_factor*2, beta=1, out=ejk)


def _oo_atm_block(j3c_100, k0, k1, v0, work, natm, nocc, aoslices,
                  dm_factor_l, metric_w, metric_v, k_factor, j_factor, ejk,
                  h_ao_aux, j2c_10v, auxvec, auxvec_ipauxp, naux):
    """Upstream's per-atom occupied block, ``FASTDFHESSIP_OOSUM=transpose``."""
    j3c_oo_atm, work1 = _allocate((natm, 3, k1-k0, 2, nocc, nocc), work)
    for i, (p0, p1) in enumerate(zip(aoslices[:-1], aoslices[1:])):
        contract('xrnuj,nui->xrnij', j3c_100[:, k0:k1, :, p0:p1],
                 dm_factor_l[:, p0:p1], out=j3c_oo_atm[i])
    w_part = metric_w[v0+k0:v0+k1]
    transpose_sum(j3c_oo_atm.reshape(-1, nocc, nocc), inplace=True)
    j3c_oo_atm_w = ndarray(j3c_oo_atm.shape, buffer=work1)
    cp.divide(j3c_oo_atm, w_part[:, None, None, None], out=j3c_oo_atm_w)
    contract('pxrnij,qyrnij->pqxy', j3c_oo_atm_w, j3c_oo_atm, -k_factor,
             beta=1, out=ejk)
    if j_factor != 0:
        auxvec_100_atm = cp.einsum('pxrnii->pxr', j3c_oo_atm)
        auxvec_100_atm_w = auxvec_100_atm / w_part
        contract('pxr,qyr->pqxy', auxvec_100_atm_w, auxvec_100_atm,
                 j_factor, beta=1, out=ejk)
        tmp = ndarray((3, naux, k1-k0), buffer=work)
        cp.multiply(j2c_10v[:, :, v0+k0:v0+k1], auxvec[:, None], out=tmp)
        contract('ys,sr->ysr', auxvec_ipauxp, metric_v[:, v0+k0:v0+k1],
                 beta=-1, out=tmp)
        contract('pxr,ysr->psxy', auxvec_100_atm_w, tmp, j_factor, 1,
                 h_ao_aux)


def _ejk_ip2(int3c2e_opt, dm_factor_l, dm_oo, auxvec, j_factor, k_factor,
             omega, lr_factor, sr_factor, pair_addresses, nao_pair,
             batch_size, blksize, spmult):
    """(20|0)(0|0)(0|00) + (10|1)(0|0)(0|00): upstream's kernel loop, with
    item 1's launch and both spins' pseudo-density in one contraction."""
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    natm = mol.natm
    nao, nocc = dm_factor_l.shape[1:]
    aux_loc = auxmol.ao_loc
    if j_factor != 0:
        dm = contract('npi,nqi->pq', dm_factor_l, dm_factor_l)

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme_ip2(omega)
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
    buf1 = cp.empty((2, blksize, nao, nocc))
    aux0 = aux1 = 0

    def plans(m):
        # the pseudo-density's (nao, nao, m) output, item 13
        return _cuplan.planable('niqr,npi->pqr', (2, nocc, nao, m),
                                (2, nao, nocc), (nao, nao, m))
    for kbatch in range(num_aux_batches):
        naux_in_batch = aux_offsets[kbatch+1] - aux_offsets[kbatch]
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        compressed = ndarray((nao_pair, naux_in_batch), buffer=buf)
        for k0, k1 in _cuplan.prange(0, naux_in_batch, blksize, plans):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            dm_tensor = ndarray((nao, nao, dk), buffer=buf2)
            tmp = ndarray((2, nocc, nao, dk), buffer=buf1)
            beta = 0
            if j_factor != 0:
                cp.multiply(dm[:, :, None], auxvec[aux0:aux1], out=dm_tensor)
                beta = j_factor
            contract('rnji,nqj->niqr', dm_oo[aux0:aux1], dm_factor_l,
                     out=tmp)
            contract('niqr,npi->pqr', tmp, dm_factor_l, -k_factor, beta,
                     out=dm_tensor)
            cp.take(dm_tensor.reshape(-1, dk), pair_addresses, axis=0,
                    out=compressed[:, k0:k1])
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
    return ejk


# --------------------------------------------------------------------------
# the h1 term of make_h1
# --------------------------------------------------------------------------
def _get_veff(int3c2e_opt, mo_coeff, mo_occ, j_factor=1, k_factor=1,
              omega=None, verbose=None):
    if not _usable():
        _S['declined'] += 1
        return DFUHESS._get_veff_orig(int3c2e_opt, mo_coeff, mo_occ,
                                      j_factor, k_factor, omega, verbose)
    _S['h1_u'] = _S.get('h1_u', 0) + 1
    spmult = int(_env('FASTDFHESSIP_SPMULT', '1'))
    jbatch = _env('FASTDFHESSIP_H1J', 'batched') != 'upstream'
    # item 14 (W30): no exchange contraction at all when k_factor = 0
    do_k = k_factor != 0 or _env('FASTDFHESSIP_K0', 'skip') != 'skip'
    st = _Stages('h1 U')

    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    if omega is None:
        omega = mol.omega

    ao_idx = mol.get_ao_idx()
    mo_coeff = mol.apply_C_dot(mo_coeff, axis=1)
    mo_coeff = mo_coeff[:, ao_idx]
    mask = mo_occ > 0
    nocca, noccb = mask.sum(axis=1).get()
    nocc = max(nocca, noccb)
    nao = mo_coeff.shape[1]
    orbo = cp.zeros((2, nao, nocc))
    orbo[0, :, :nocca] = mo_coeff[0][:, mask[0]]
    orbo[1, :, :noccb] = mo_coeff[1][:, mask[1]]

    natm = mol.natm
    aux_loc = auxmol.ao_loc
    naux = int(aux_loc[-1])
    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=True)[0]
    pair_addresses = cp.asarray(pair_addresses, dtype=np.int32)
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)

    vhf_atm = cp.zeros((natm, 3, 2, nocc, nao))
    vhf1 = cp.zeros((3, 2, nao, nao))

    cp.get_default_memory_pool().free_all_blocks()
    mem_avail = _avail()
    word_avail = mem_avail // 8
    word_avail -= 2 * naux*nocc*nao  # dm_3c ~ (aux,i,a)
    word_avail -= naux**2 * 3  # metric_v
    word_avail -= 2 * naux*nocc**2  # dm_oo ~ (aux,i,j)
    assert word_avail > 0, 'Insufficient GPU memory'

    mem_sufficient = natm*3*2*nao**2 < word_avail * .6
    # the per-atom AO-pair matrices hold exchange terms only
    k_ao = mem_sufficient and do_k
    if k_ao:
        vhf_atm_ao = cp.zeros((natm, 3, 2, nao, nao))
        word_avail -= vhf_atm_ao.size

    pair_size_max = max(nao_pair, 2*nocc*nao)
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

    atm_ao_counts = np.bincount(_bas_atom_labels(mol))
    aoslices = np.append(0, atm_ao_counts.cumsum())
    nao_on_atom = atm_ao_counts.max()
    atm_ind = _atom_indicator(aoslices)

    # item 4 holds one more (2, nao, blksize, nao) array per block, the
    # occupied-contracted density of both spins, so the block is sized with
    # it in the denominator
    h1atom = _env('FASTDFHESSIP_H1ATOM', 'auto')
    hoist = k_ao and h1atom != 'upstream'
    # item 12: the same per-atom exchange term with the AO pair projected
    # on the occupied orbitals before the atom's contraction, not after.
    # Per atom and block it saves 4 nao_A dk nao (nao - 2 nocc) flops and
    # costs one more contraction launch (and two that run below the AO
    # form's GEMM rate), so it is taken where the saving per atom is large:
    # measured, it loses at every def2-SVP case up to 60 atoms (score
    # 3e4-1.7e6) and wins at 113 atoms (7.7e6-9.1e6) and in def2-TZVPP at
    # 42 (3.0e7).  `=occ` and `=batched` force either form
    score = nao / natm * nao * max(nao - 2*nocc, 0)
    occ_k = hoist and (h1atom == 'occ' or
                       (h1atom == 'auto' and score > float(
                           _env('FASTDFHESSIP_H1OCC_SCORE', '4e6'))))
    _S['h1_occ'] = bool(occ_k)
    blk_unit = (nao**2 + (nao+nocc*2) * max(nao_on_atom, nocc) +
                (2*nao*(nao + (nocc if occ_k else 0)) if hoist else 0))
    blksize = int((word_avail*.95 - batch_size*_unit) / blk_unit) // 8 * 8
    assert blksize > 0, 'Insufficient GPU memory'
    blksize = min(blksize, batch_size)
    st('setup')

    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((batch_size, nao_pair))
    j3c_00 = cp.empty((naux, 2, nocc, nao))
    for kbatch in range(num_aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:, :, :dk]
            fill_symmetric(compressed, pair_addresses, nao, k0, k1, out=j3c)
            contract('pqr,npi->rniq', j3c, orbo, out=j3c_00[aux0:aux1])
    j3c_full = j3c = buf = eval_j3c = compressed = None
    st('A j3c_00')

    aux_idx, aux_slices = _argsort_aux_by_atom(auxmol, aux_sorting)
    counts = aux_slices[:, 1] - aux_slices[:, 0]
    atm_id_for_aux = np.empty(naux, dtype=int)
    atm_id_for_aux[aux_idx] = np.repeat(np.arange(auxmol.natm), counts)
    aux_filling_order = np.empty(naux, dtype=int)
    aux_filling_order[aux_idx] = np.arange(naux)

    metric_w, metric_v = _factorize_j2c(auxmol, aux_sorting, omega)
    nw = metric_v.shape[1]
    work = cp.empty((nw, 2, 8, nao))
    for i0, i1 in lib.prange(0, nocc, 8):
        tmp = ndarray((nw, 2, i1-i0, nao), buffer=work)
        contract('rs,rniq->sniq', metric_v, j3c_00[:, :, i0:i1], out=tmp)
        tmp /= metric_w[:, None, None, None]
        contract('rs,sniq->rniq', metric_v, tmp, out=j3c_00[:, :, i0:i1])
    dm_3c = j3c_00
    j3c_00 = tmp = work = metric_v = metric_w = None

    dm_oo = contract('rniq,nqj->rnij', dm_3c, orbo)
    if j_factor != 0:
        auxvec = cp.einsum('rnii->r', dm_oo)
        auxvec_by_atm = auxvec[aux_idx]
        dm = contract('npi,nqi->pq', orbo, orbo)
    st('B metric')

    # (00|0)(1|0)(0|00)
    j2c_10 = int2c2e_ip1(auxmol, sort_output=False, omega=omega)
    inv_aux_sorting = cp.empty(naux, dtype=int)
    inv_aux_sorting[aux_sorting] = cp.arange(naux)
    j2c_10 = j2c_10[:, inv_aux_sorting[aux_idx, None], inv_aux_sorting]
    naux_in_atm = (aux_slices[:, 1] - aux_slices[:, 0]).max()
    buf = cp.empty((3, naux_in_atm, 2, nocc, nao))
    buf1 = cp.empty((naux_in_atm, 2, nocc, nao))
    for i, (p0, p1) in enumerate(aux_slices):
        if not do_k:
            dm_3c_atm = cp.take(dm_3c, aux_idx[p0:p1], axis=0,
                                out=buf1[:p1-p0])
            j3c_1 = ndarray((3, p1-p0, 2, nocc, nao), buffer=buf)
            contract('xrs,sniq->xrniq', j2c_10[:, p0:p1], dm_3c, out=j3c_1)
        elif mem_sufficient:
            dm_3c_atm = cp.take(dm_3c, aux_idx[p0:p1], axis=0,
                                out=buf1[:p1-p0])
            j3c_1 = ndarray((3, p1-p0, 2, nocc, nao), buffer=buf)
            contract('xrs,sniq->xrniq', j2c_10[:, p0:p1], dm_3c, out=j3c_1)
            contract('xrnip,rniq->xnpq', j3c_1, dm_3c_atm, -k_factor, beta=1,
                     out=vhf_atm_ao[i])
        else:
            dm_3c_atm = cp.take(dm_3c, aux_idx[p0:p1], axis=0,
                                out=buf1[:p1-p0])
            dm_oo_atm = dm_oo[aux_idx[p0:p1]]
            tmp = ndarray((3, p1-p0, 2, nocc, nocc), buffer=buf)
            tmp = contract('xrs,snij->xrnij', j2c_10[:, p0:p1], dm_oo, out=tmp)
            contract('rniq,xrnij->xnjq', dm_3c_atm, tmp, -k_factor, beta=1,
                     out=vhf_atm[i])
            j3c_1 = ndarray((3, p1-p0, 2, nocc, nao), buffer=buf)
            contract('xrs,sniq->xrniq', j2c_10[:, p0:p1], dm_3c, out=j3c_1)
            contract('xrniq,rnij->xnjq', j3c_1, dm_oo_atm, -k_factor, beta=1,
                     out=vhf_atm[i])
        if j_factor != 0:
            contract('xrniq,r->xniq', j3c_1, auxvec_by_atm[p0:p1], j_factor,
                     beta=1, out=vhf_atm[i])
            tmp = cp.einsum('xrs,s->xr', j2c_10[:, p0:p1], auxvec)
            contract('rniq,xr->xniq', dm_3c_atm, tmp, j_factor, beta=1,
                     out=vhf_atm[i])
    tmp = j3c_1 = dm_3c_atm = dm_oo_atm = None
    buf = buf1 = j2c_10 = None
    st('C j2c_10')

    # (10|0)(0|0)(0|00)
    eval_ip1 = _ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ip1(omega, 27), batch_size,
        'fill_int3c2e_ip1', omega, spmult)[0]
    eval_ipaux = _ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ipaux(omega, 27), batch_size,
        'fill_int3c2e_ipaux', omega, spmult)[0]

    work_size = max(
        nao**2*blksize + (3*pair_size_max + 6*nocc*nao) * batch_size,
        6*nocc*nao*batch_size * 2,
        6*pair_size_max*batch_size +
        (nao*nao + (nao+nocc)*2 * max(nao_on_atom, nocc)) * blksize)
    # item 4's array lives at the tail of the same buffer: it is dead before
    # the (00|1) half of each batch reuses that space
    t_size = 2*nao*nao*blksize if hoist else 0
    s_size = 2*nao*nocc*blksize if occ_k else 0
    work = cp.empty(work_size + t_size + s_size)
    t_buf = work[work_size:work_size+t_size] if hoist else None
    s_buf = work[work_size+t_size:] if occ_k else None
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

        # (di/dr j|k) + (i dj/dr|k) + (i j|dk/dr) = 0
        compressed_dk += compressed_di
        compressed_dj = compressed_dk
        compressed_di *= -1
        _aux1 = aux0
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            _aux0, _aux1 = _aux1, _aux1 + dk
            j3c = j3c_full[:, :, :dk]
            if hoist:
                # item 4: the occupied-contracted density of both spins for
                # every AO at once, which upstream forms per atom and per x
                t_all = ndarray((2, nao, dk, nao), buffer=t_buf)
                contract('nui,rniq->nurq', orbo, dm_3c[_aux0:_aux1],
                         out=t_all)
            if occ_k:
                s_all = ndarray((2, nao, dk, nocc), buffer=s_buf)
                contract('nui,rnij->nurj', orbo, dm_oo[_aux0:_aux1],
                         out=s_all)
            for x in range(3):
                j3c[j_addr, i_addr] = compressed_dj[x, :, k0:k1]
                j3c[i_addr, j_addr] = compressed_di[x, :, k0:k1]
                if do_k:
                    tmp = ndarray((2, nao, dk, nocc), buffer=work1)
                    j3c_pri = contract('pqr,nqi->npri', j3c, orbo, out=tmp)
                    contract('npri,rniq->npq', j3c_pri, dm_3c[_aux0:_aux1],
                             -k_factor, beta=1, out=vhf1[x])
                st('D1 vhf1')
                if not do_k:
                    pass
                elif occ_k:
                    # sum_{u in A, r} of the two halves of X + X^T, each
                    # already on the occupied side: upstream's
                    # memory-short branch with both of its intermediates
                    # formed once per block (j3c_pri is D1's), which is
                    # 8 nao^2 nocc flops per auxiliary function and
                    # direction against 4 nao^3 for the AO-pair form
                    for i, (p0, p1) in enumerate(zip(aoslices[:-1],
                                                     aoslices[1:])):
                        contract('uqr,nurj->njq', j3c[p0:p1],
                                 s_all[:, p0:p1], -k_factor, beta=1,
                                 out=vhf_atm[i, x])
                        contract('nurj,nurq->njq', j3c_pri[:, p0:p1],
                                 t_all[:, p0:p1], -k_factor, beta=1,
                                 out=vhf_atm[i, x])
                elif hoist:
                    for i, (p0, p1) in enumerate(zip(aoslices[:-1],
                                                     aoslices[1:])):
                        contract('upr,nurq->npq', j3c[p0:p1],
                                 t_all[:, p0:p1], -k_factor, beta=1,
                                 out=vhf_atm_ao[i, x])
                elif k_ao:
                    for i, (p0, p1) in enumerate(zip(aoslices[:-1],
                                                     aoslices[1:])):
                        if p1 - p0 < nocc:
                            tmp = ndarray((2, p1-p0, dk, nao), buffer=work1)
                            contract('nui,rniq->nurq', orbo[:, p0:p1],
                                     dm_3c[_aux0:_aux1], out=tmp)
                            contract('upr,nurq->npq', j3c[p0:p1], tmp,
                                     -k_factor, beta=1, out=vhf_atm_ao[i, x])
                        else:
                            tmp = ndarray((nao, dk, 2, nocc), buffer=work1)
                            contract('pqr,npi->qrni', j3c[p0:p1],
                                     orbo[:, p0:p1], out=tmp)
                            contract('prni,rniq->npq', tmp,
                                     dm_3c[_aux0:_aux1], -k_factor, beta=1,
                                     out=vhf_atm_ao[i, x])
                else:
                    for i, (p0, p1) in enumerate(zip(aoslices[:-1],
                                                     aoslices[1:])):
                        tmp, buf2 = _allocate((2, p1-p0, dk, nocc), work1)
                        tmp1 = ndarray((2, p1-p0, dk, nao), buffer=buf2)
                        contract('nui,rnij->nurj', orbo[:, p0:p1],
                                 dm_oo[_aux0:_aux1], out=tmp)
                        contract('uqr,nurj->njq', j3c[p0:p1], tmp,
                                 -k_factor, beta=1, out=vhf_atm[i, x])
                        contract('uqr,nqj->nurj', j3c[p0:p1], orbo, out=tmp)
                        contract('nui,rniq->nurq', orbo[:, p0:p1],
                                 dm_3c[_aux0:_aux1], out=tmp1)
                        contract('nurj,nurq->njq', tmp, tmp1, -k_factor,
                                 beta=1, out=vhf_atm[i, x])
                st('D2 K per-atom')
                if j_factor != 0:
                    # the same Coulomb matrix for both spins: once
                    vj1 = contract('pqr,r->pq', j3c, auxvec[_aux0:_aux1])
                    vhf1[x, 0] += j_factor * vj1
                    vhf1[x, 1] += j_factor * vj1
                    vj1 = None
                    if jbatch:
                        # item 5: sum_{p in A, q} j3c[p,q,r] dm[p,q] for
                        # every atom A is a row-wise contraction and a GEMM
                        # against the atom indicator
                        w_pr = contract('pqr,pq->pr', j3c, dm)
                        av_atm = atm_ind.dot(w_pr)
                        contract('rniq,ar->aniq', dm_3c[_aux0:_aux1], av_atm,
                                 2*j_factor, beta=1, out=vhf_atm[:, x])
                        w_pr = av_atm = None
                    else:
                        for i, (p0, p1) in enumerate(zip(aoslices[:-1],
                                                         aoslices[1:])):
                            auxvec1 = cp.einsum('pqr,pq->r', j3c[p0:p1],
                                                dm[p0:p1])
                            contract('rniq,r->niq', dm_3c[_aux0:_aux1],
                                     auxvec1, 2*j_factor, beta=1,
                                     out=vhf_atm[i, x])
                st('D3 J per-atom')

        # (00|1)(0|0)(0|00)
        compressed_dk += compressed_di
        j3c_aux, work1 = _allocate((3, naux_in_batch, 2, nocc, nao), work)
        off = max(j3c_full.size + compressed_dk.size, j3c_aux.size)
        j3c_aux_tmp = ndarray(j3c_aux.shape, buffer=work[off:])
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            j3c = j3c_full[:, :, :dk]
            for i in range(3):
                fill_symmetric(compressed_dk[i], pair_addresses, nao, k0, k1,
                               out=j3c)
                # d/dX = -d/dr
                contract('pqr,npi->rniq', j3c, orbo, alpha=-1,
                         out=j3c_aux_tmp[i, k0:k1])

        idx = np.argsort(aux_filling_order[aux0:aux1])
        j3c_aux = cp.take(j3c_aux_tmp, idx, axis=1, out=j3c_aux)

        dm_3c_batch, work1 = _allocate((naux_in_batch, 2, nocc, nao), work1)
        dm_oo_batch, work1 = _allocate((naux_in_batch, 2, nocc, nocc), work1)
        # dm_3c_batch overlaps j3c_aux_tmp, which is copied out above
        dm_3c_batch = cp.take(dm_3c[aux0:aux1], idx, axis=0, out=dm_3c_batch)
        dm_oo_batch = cp.take(dm_oo[aux0:aux1], idx, axis=0, out=dm_oo_batch)
        if j_factor != 0:
            auxvec_batch = auxvec[aux0:aux1][idx]

        counts = np.bincount(atm_id_for_aux[aux0:aux1])
        oo_buf = None
        if work1.size > 6*counts.max()*nocc**2:
            oo_buf = work1
        st('E ipaux fill')

        p0 = p1 = 0
        for i, count in enumerate(counts):
            if count == 0:
                continue
            p0, p1 = p1, p1 + count
            if not do_k:
                auxvec_ipaux = contract('xrnip,npi->xr', j3c_aux[:, p0:p1],
                                        orbo)
            elif mem_sufficient:
                auxvec_ipaux = contract('xrnip,npi->xr', j3c_aux[:, p0:p1],
                                        orbo)
                contract('xrnip,rniq->xnpq', j3c_aux[:, p0:p1],
                         dm_3c_batch[p0:p1], -k_factor, beta=1,
                         out=vhf_atm_ao[i])
            else:
                tmp = ndarray((3, p1-p0, 2, nocc, nocc), buffer=oo_buf)
                j3c_1 = contract('xrniq,nqj->xrnij', j3c_aux[:, p0:p1], orbo,
                                 out=tmp)
                auxvec_ipaux = cp.einsum('xrnii->xr', j3c_1)
                contract('rniq,xrnij->xnjq', dm_3c_batch[p0:p1], j3c_1,
                         -k_factor, beta=1, out=vhf_atm[i])
                contract('xrniq,rnij->xnjq', j3c_aux[:, p0:p1],
                         dm_oo_batch[p0:p1], -k_factor, beta=1,
                         out=vhf_atm[i])
            if j_factor != 0:
                contract('rniq,xr->xniq', dm_3c_batch[p0:p1], auxvec_ipaux,
                         j_factor, beta=1, out=vhf_atm[i])
                contract('xrniq,r->xniq', j3c_aux[:, p0:p1],
                         auxvec_batch[p0:p1], j_factor, beta=1,
                         out=vhf_atm[i])
        st('E per-aux-atom')
    work = work1 = buf0 = buf1 = oo_buf = dm_3c_batch = dm_oo_batch = None
    j3c_full = j3c = j3c_1 = j3c_aux = j3c_aux_tmp = tmp = tmp1 = None
    compressed_di = compressed_dj = compressed_dk = t_all = s_all = None
    dm_oo = dm_3c = None

    if k_ao:
        transpose_sum(vhf_atm_ao.reshape(-1, nao, nao), inplace=True)
        contract('nxspq,sqj->nxsjp', vhf_atm_ao, orbo, beta=1, out=vhf_atm)
        vhf_atm_ao = None

    # Distribute <d/dR i|Veff|j> to derivatives on atoms
    for i, (p0, p1) in enumerate(zip(aoslices[:-1], aoslices[1:])):
        contract('xnpq,npi->xniq', vhf1[:, :, p0:p1], orbo[:, p0:p1], beta=1,
                 out=vhf_atm[i])
        contract('xnpq,nqi->xnip', vhf1[:, :, p0:p1], orbo, beta=1,
                 out=vhf_atm[i, :, :, :, p0:p1])

    vhf_atm = contract('nxsjp,spi->snxij', vhf_atm, mo_coeff)
    vhfa = vhf_atm[0, :, :, :, :nocca]
    vhfb = vhf_atm[1, :, :, :, :noccb]
    st('F finish')
    st.report()
    log.timer_debug1('_get_veff UHF (fastdfhessip)', *t0)
    return [vhfa, vhfb]


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED:
        return
    DFUHESS._jk_energy_per_atom_orig = _orig_jk_energy_per_atom
    DFUHESS._get_veff_orig = _orig_get_veff
    DFUHESS._jk_energy_per_atom = _jk_energy_per_atom
    DFUHESS._get_veff = _get_veff
    _PATCHED = True
