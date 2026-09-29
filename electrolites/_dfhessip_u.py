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
have no occupied batching (``_get_veff`` has it here: item 23).

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
  all when ``k_factor = 0``).  **Item 17** (W32: stage C's per-auxiliary-
  atom term grouped into cuBLAS GEMMs, and its Coulomb half without
  ``j3c_1``) is ``fastdfhessip._h1_stage_c`` on the spin index, and
  **item 20** (W35: item 12's per-atom products as batched GEMMs,
  ``_h1d2``) batches over the atoms and both spins at once.  **Item 23**
  (W38) is the restricted function's occupied batching, which upstream's
  twin does not have: without it the call needs ``2 naux nocc (nao +
  nocc)`` words of density factors at once, 29 GB at 113 atoms.

Item 8 (the metric as its Cholesky factor, W24) was not carried over by
W25 -- W24 had measured it at 1.05x-1.06x of the restricted call below 42
atoms -- and at 113 atoms its absence was most of the call: the eigenbasis
GEMMs of D3 and D5 were 26 of the Co(II) sextet's 42 s.  **W37** carried it
over as **item 25** (``_chol_d4_d5_u``, a pass per spin where both do not
fit), with **items 22, 24, 26 and 28** (``fastdfhessip``'s docstring).

W29's two launch items apply here unchanged: **item 15** (the
``ejk_int3c2e_ip2`` kernel as its own tile, both spins' pseudo-density
permuted into the tile's layout) in ``_ejk_ip2``, and **item 16** (the
derivative fills' pair count) through ``fastdfhessip._ip1_evaluator``.
"""
import ctypes

import numpy as np
import cupy as cp
from cupy_backends.cuda.api import runtime

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
from . import _h1d2
from . import _dfpin
from . import fastdfhessip as H
from .fastdfhessip import (_env, _S, _Stages, _ip1_evaluator, _avail,
                           _atom_indicator, _usable, _forced_occ_batch)

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
    # item 30 (W39): fastdfhessip's lower-triangle stage A, per spin; not
    # on the long-range operator (below)
    low = (H._LowTri(pair_addresses, nao)
           if _env('FASTDFHESSIP_ALOW', '1') != '0' and omega == 0 else None)
    if low is not None:
        _S['a_low'] = _S.get('a_low', 0) + 1
        g = cp.empty((blksize, nocc, nocc))
        cs = [cp.ascontiguousarray(dm_factor_l[n]) for n in range(2)]
    for kbatch in range(num_aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:, :, :dk]
            if low is not None:
                low.scatter(compressed, k0, k1, j3c)
                tmpf = ndarray((nocc, nao, blksize), buffer=buf1)
                for n in range(2):
                    low.lt_c(cs[n], j3c_full, blksize, tmpf)
                    gk = contract('iqr,qj->rij', tmpf[:, :, :dk], cs[n],
                                  out=g[:dk])
                    cp.add(gk, gk.transpose(0, 2, 1),
                           out=j3c_oo[aux0:aux1, n])
                continue
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
    tmpf = g = gk = cs = low = None
    st('A j3c_oo')

    # item 8 on the spin index (W37): the metric as its Cholesky factor on
    # the spherical auxiliary index, in every line of the call, as the
    # restricted function has it -- the terms cancel, so one J^-1 must
    # serve all of them (W24)
    sa = None
    if (_env('FASTDFHESSIP_METRIC', 'chol') == 'chol' and
            _env('FASTDFHESSIP_UCHOL', '1') != '0' and omega == 0 and
            oosum == 'gram' and not auxmol.mol.cart):
        try:
            sa = H._SphAux(auxmol, aux_sorting, omega,
                           _bas_atom_labels(auxmol, aux_sorting))
        except RuntimeError:
            sa = None
    if sa is not None:
        metric_v = sa.factor()
        metric_w = cp.ones(sa.nsph)
        _S['chol_u'] = _S.get('chol_u', 0) + 1
    else:
        metric_w, metric_v = _factorize_j2c(auxmol, aux_sorting, omega)
    metric_size = metric_v.shape[1]
    # item 13: the contractions whose output can pass cuTENSOR's 2**29
    # elements go to cuBLAS when cuTENSOR would refuse them
    # (`docs/DF_WORK_LIST.md` W21, W27) -- upstream's own function stops
    # here, or at the pseudo-density, on a 113-atom open shell
    if _env('FASTDFHESSIP_PACK', '1') != '0' and omega == 0:
        # item 22: stage B on packed pairs (both spins)
        H._metric_packed(metric_v, metric_w, j3c_oo, 2)
    else:
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

    # (1|00), allocated once item 34 has said in which form
    j1_shape = (3, naux, 2, nocc, nocc)

    # The second pass's sizes, worked out before the first pass so that
    # item 3 can tell whether there will be one chunk.  Upstream derives
    # them from the memory free after the first pass and the h_aux block,
    # which free what they allocate except j2c_10v; the arithmetic is
    # upstream's, on the same words.
    cp.get_default_memory_pool().free_all_blocks()
    word_avail = H._avail_less(8 * int(np.prod(j1_shape))) // 8
    blksize1 = min(naux, word_avail // (nao**2*8) * 8)
    assert blksize1 > 0, 'Insufficient GPU memory'
    blksize1 = min(blksize1, batch_size)
    gram = oosum != 'transpose'
    # item 8's second pass needs the whole spherical index in one chunk; if
    # both spins' (10|0) do not fit, it takes them one spin at a time, which
    # walks the integrals twice (the eigen form would walk them once per
    # chunk anyway).  j10s, its spherical j2c_10, is held beside it.
    nsp = 2
    if sa is not None:
        chol_w = word_avail - 3 * metric_size**2
        for nsp in (2, 1):
            try:
                pl = _plan_v(chol_w, naux, nao, nocc, nao_pair, batch_size,
                             nao_on_atom, natm, metric_size, gram, nsp)
            except AssertionError:
                continue
            if pl[0] >= metric_size:
                word_avail = chol_w
                break
        else:
            nsp = 2
            if _env('FASTDFHESSIP_CROWS', '1') != '0':
                # item 33 (W40): the factor in chunks of rows, with room
                # for its spherical j2c_10v
                word_avail = chol_w
            else:
                sa = None
        if _env('FASTDFHESSIP_USPIN', '') == '1' and sa is not None:
            nsp = 1
    def make_plan(word_avail, rows):
      plan = _plan_v(word_avail, naux, nao, nocc, nao_pair, batch_size,
                     nao_on_atom, natm, metric_size, gram, nsp, rows)
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
                             batch_size, nao_on_atom, natm, metric_size, gram,
                             nsp, rows)
        except AssertionError:
            plan_f = (0,)
        if plan_f[0] > 0 and (-(-metric_size // plan_f[0]) <=
                              -(-metric_size // plan[0])):
            plan, do_fuse = plan_f, True
      return plan, plan_f, do_fuse, carve
    plan, plan_f, do_fuse, carve = make_plan(word_avail, False)
    aux_batch_size, aux_blksize, blksize2 = plan
    chol = sa if (sa is not None and aux_batch_size >= metric_size) else None
    if chol is None:
        nsp = 2
    if sa is not None and chol is None:
        _S['chol_chunked'] = _S.get('chol_chunked', 0) + 1
    # item 33 (W40): fastdfhessip's chunks of rows of Y, both spins in each
    crow = None
    pack34 = rows34 = False
    if (sa is not None and chol is None and
            _env('FASTDFHESSIP_CROWS', '1') != '0'):
        crow = H._CholRows(sa)
        _S['chol_rows'] = _S.get('chol_rows', 0) + 1
        # item 34, as in fastdfhessip (FASTDFHESSIP_PACKROWS=0: item 33
        # alone); either holds its spherical j2c_10v back
        if _env('FASTDFHESSIP_PACKROWS', '1') != '0':
            try:
                plan, plan_f, do_fuse, carve = make_plan(
                    word_avail - 3*naux*metric_size, True)
            except AssertionError:
                # not a row before the packing: unfused, and planned after
                # it
                plan, plan_f, do_fuse, carve = ((metric_size, 0, 0), (0,),
                                                False, False)
            aux_batch_size, aux_blksize, blksize2 = plan
            pack34 = not do_fuse
            rows34 = True
        else:
            plan, plan_f, do_fuse, carve = make_plan(
                word_avail - 3*naux*metric_size, False)
            aux_batch_size, aux_blksize, blksize2 = plan
        if pack34:
            # item 34: dm_oo and (1|00) as their weighted pairs from the
            # first pass on (fastdfhessip's note)
            j3c_oo1 = H._SymRows(3*naux, 2, nocc, xstride=naux)
            dmp = H._SymRows(naux, 2, nocc)
            dmp.put(0, dm_oo)
            dm_oo = dmp
    if not pack34:
        j3c_oo1 = cp.empty(j1_shape)  # = (1|00)
    st.note(chol=chol is not None, spin_passes=2 // nsp)
    _S['u_spin_passes'] = 2 // nsp if chol is not None else 0
    st.note(free_gb=round(word_avail * 8e-9, 1), plan=plan[0],
            plan_fused=plan_f[0], nao=nao, nocc=nocc, naux=naux,
            metric_size=metric_size, nao_pair=nao_pair,
            batch_size=batch_size, num_aux_batches=num_aux_batches,
            vchunk=aux_batch_size, blksize2=blksize2, fused=do_fuse)
    _S['vchunks'] = int(-(-metric_size // aux_batch_size))
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
        pin = isinstance(j3c_oo1, H._SymRows)
        if j_factor != 0:
            # the (1|00) half of auxvec_ipauxp; the block adds the rest
            auxvec_ipauxp = (j3c_oo1.diag_sum().reshape(3, naux) if pin else
                             cp.einsum('xrnii->xr', j3c_oo1))
        o1p, parts = H._h_aux_block(
            auxmol, omega, _aux_sorting, metric_v, metric_w,
            dm_oo if pin else dm_oo.reshape(naux, 2*nocc, nocc),
            j3c_oo1 if pin else j3c_oo1.reshape(3, naux, 2*nocc, nocc),
            naux, metric_size, 2*k_factor, j_factor, 0, auxvec_ipauxp,
            auxvec, natm, aux_atm_labels, work=work, nspin=2)
        if not pin:
            o1p = o1p.reshape(3, naux, 2, nocc, nocc)
        if crow is not None:
            # item 33: D5's partners, spherical once
            crow.set_j2c_10v(parts[1])
            for x in range(3):
                rx = o1p.p[x*naux:(x+1)*naux] if pin else o1p[x]
                sa.to_sph(rx, rx)
        return (o1p, parts[0], parts[1], parts[2])

    st('D0 plan')
    if not do_fuse:
        _ipaux_pass(eval_ipaux, num_aux_batches, blksize1, nao, nao_pair,
                    batch_size, nocc, pair_addresses, dm_factor_l, j3c_oo1)
        st('D1 ipaux pass')
        j3c_oo1p, e_aux, j2c_10v, j2c_10 = h_aux()
        ejk_ao += e_aux
        j2c_10c = H._j2c_10c(chol, j2c_10)
        j2c_10 = None
        if pack34:
            # item 34: the second pass planned on what the packing left
            t1p = j3c_oo1p
            j3c_oo1p = j3c_oo1 = dm_oo = None
            word_avail = _avail() // 8
            plan = _plan_v(word_avail, naux, nao, nocc, nao_pair, batch_size,
                           nao_on_atom, natm, metric_size, gram, nsp, True,
                           True)
            aux_batch_size, aux_blksize, blksize2 = plan
            _S['vchunks'] = int(-(-metric_size // aux_batch_size))
            _S['packrows'] = _S.get('packrows', 0) + 1
            st.note(free_gb_packed=round(word_avail * 8e-9, 1),
                    vchunk_packed=aux_batch_size)
            if _env('FASTDFHESSIP_MEMLOG', '') == '1':
                print(f'[memlog ip2 U] packed: free {word_avail*8e-9:.2f} GB,'
                      f' plan {plan}', flush=True)
        st('D2 h_aux')

    j3c_buf = cp.empty((3, aux_batch_size, nsp, nao, nocc))
    wuse = [0 if gram else natm*3*2*nocc*nocc*2*aux_blksize,
            3*nao_pair*batch_size * 2 + (nocc*2+nao)*nao * blksize2,
            3*naux*2*nao_on_atom*nocc * 2,
            3*naux*2*nao_on_atom*nocc + 3*aux_batch_size*naux]
    if rows34:
        # item 34: item 33's D5 in place of upstream's per-atom one
        wuse[2:] = [H._D5_ROWS_WORDS]
    if chol is not None:
        ns_, nc_ = chol.nsph, chol.ncart
        wuse += [3*nsp*nao*nocc + 3*nao*nao, 3*nc_*ns_,
                 3*ns_*nsp*nao_on_atom*nocc + nc_*nsp*nao_on_atom*nocc +
                 3*ns_*ns_]
    work = cp.empty(max(wuse))
    if _env('FASTDFHESSIP_MEMLOG', '') == '1':
        print(f'[memlog ip2 U] after D3 alloc: free {_avail()*1e-9:.2f} GB, '
              f'j3c_buf {j3c_buf.nbytes*1e-9:.2f}, work {work.nbytes*1e-9:.2f}'
              f', plan {plan}, wuse {[round(w*8e-9, 2) for w in wuse]}',
              flush=True)
    st('D3 alloc')
    # the spin groups of the second pass: both at once, or with item 8 on a
    # short card one at a time, the integrals walked once per group
    groups = [(0, 2)] if nsp == 2 else [(0, 1), (1, 2)]
    # item 24: the fused first pass from the second pass's M C, which needs
    # both spins in the pass that runs it
    d1d3 = (do_fuse and nsp == 2 and
            _env('FASTDFHESSIP_D1D3', '1') != '0')
    ud = {}
    vchunks = list(lib.prange(0, metric_size, aux_batch_size))
    if crow is not None:
        # item 33: the last rows first, whose walk is the whole one
        vchunks = vchunks[::-1]
        h_raw = cp.zeros((natm, sa.nsph, 3, 3))
    for ci, ((n0, n1), (v0, v1)) in enumerate(
            (g, v) for g in groups for v in vchunks):
        dv = v1 - v0
        first = n0 == 0
        vfirst = v0 == vchunks[0][0]
        ncp = crow.ncart(v1) if crow is not None else naux
        j3c_100 = ndarray((3, dv, n1-n0, nao, nocc), buffer=j3c_buf)
        j3c_100[:] = 0.
        j3c_full, work1 = _allocate((nao, nao, blksize2), work)
        j3c_full[:] = 0.
        buf0, work1 = _allocate((3, nao_pair, batch_size), work1)
        buf1, work1 = _allocate((3, nao_pair, batch_size), work1)
        st('D3 buffers')
        aux0 = aux1 = 0
        for kbatch in range(num_aux_batches):
            if aux1 >= ncp:
                # item 33: no row past here reaches this chunk
                break
            compressed_di = eval_ip1(kbatch, out=buf0)
            compressed_dk = eval_ipaux(kbatch, out=buf1)
            naux_in_batch = compressed_di.shape[-1]
            st('D3 integrals')
            if do_fuse and vfirst and first and not d1d3:
                # item 3: the first pass, on this batch's ipaux integrals
                # before they are folded into dj
                _ipaux_batch(compressed_dk, aux1, naux_in_batch, blksize2,
                             j3c_full, work1, nao, nocc, pair_addresses,
                             dm_factor_l, j3c_oo1)
                st('D1 ipaux (fused)')
            # (di/dr j|k) + (i dj/dr|k) + (i j|dk/dr) = 0
            d3sc = _env('FASTDFHESSIP_D3SCATTER', '1') != '0'
            if not d3sc:
                compressed_dk += compressed_di
                compressed_dj = compressed_dk
                compressed_di *= -1
            for k0, k1 in lib.prange(0, naux_in_batch, blksize2):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                j3c = j3c_full[:, :, :dk]
                tmp = _allocate((n1-n0, nao, nocc, dk), work1)[0]
                for x in range(3):
                    if d3sc:
                        # item 28
                        H._scatter_d3(compressed_di[x], compressed_dk[x], k0,
                                      k1, pair_addresses, nao, j3c)
                    else:
                        j3c[j_addr, i_addr] = compressed_dj[x, :, k0:k1]
                        j3c[i_addr, j_addr] = compressed_di[x, :, k0:k1]
                    tmp = contract('pqr,nqi->npir', j3c, dm_factor_l[n0:n1],
                                   out=tmp)
                    if d1d3 and vfirst:
                        H._d1_from_d3(tmp, dm_factor_l, j3c_oo1[x, aux0:aux1],
                                      None, nspin=2)
                    if chol is None:
                        _mm('rs,npir->snpi', metric_v[aux0:aux1, v0:v1],
                            tmp, j3c_100[x], 'T', 'T', beta=1.)
                    else:
                        # item 8: the spherical (10|0), shell block by
                        # shell block -- a pass over memory, not a GEMM
                        c0, c1 = chol.cols(aux0, aux1)
                        contract('rs,npir->snpi', chol.ms[aux0:aux1, c0:c1],
                                 tmp, beta=1, out=j3c_100[x, c0:c1])
            st('D3 j3c_100')
        j3c_full = buf0 = buf1 = tmp = None
        compressed_di = compressed_dj = compressed_dk = None

        if do_fuse and vfirst and first:
            j3c_oo1p, e_aux, j2c_10v, j2c_10 = h_aux(work if carve else None)
            ejk_ao += e_aux
            j2c_10c = H._j2c_10c(chol, j2c_10)
            j2c_10 = None
            st('D2 h_aux')

        if chol is not None:
            _chol_d4_d5_u(chol, j3c_100, n0, n1, work, natm, nocc, atm_ind,
                          aoslices, dm_factor_l, dm_oo, j3c_oo1p, j2c_10c,
                          k_factor, j_factor, ejk_ao, h_ao_aux, auxvec,
                          auxvec_ipauxp, ud, n1 == 2, st)
            continue
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

        if crow is not None:
            crow.d5(j3c_100, v0, v1,
                    t1p if pack34 else j3c_oo1p[:, :sa.nsph],
                    dmp if pack34 else dm_oo,
                    dm_factor_l, aoslices, -k_factor*2, k_factor*2, h_raw,
                    work)
            st('D5 per-atom')
            continue
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
    if 'h_sph' in ud:
        ejk_ao_aux += _aggregate_to_atoms(ud['h_sph'], natm, chol.atm, axis=1)
    if crow is not None:
        ejk_ao_aux += _aggregate_to_atoms(h_raw, natm, sa.atm, axis=1)
        h_raw = crow = t1p = dmp = None
    ud = None
    ejk += ejk_ao_aux
    ejk += ejk_ao_aux.transpose(1, 0, 3, 2)
    ejk += ejk_ao
    st('E aggregate')
    st.report()
    log.timer_debug1('_jk_energy_per_atom UHF (fastdfhessip)', *t0)
    return ejk


def _chol_d4_d5_u(sa, ksph, n0, n1, work, natm, nocc, atm_ind, aoslices,
                  dm_factor_l, dm_oo, j3c_oo1p, j2c_10c, k_factor, j_factor,
                  ejk, h_ao_aux, auxvec, auxvec_ipauxp, ud, last, st):
    """Item 8 on the spin index (W37): ``fastdfhessip._chol_d4_d5`` for the
    spins ``n0:n1`` of ``ksph`` (``(3, nsph, n1-n0, nao, nocc)``).

    Every term is a sum over the two spins of the restricted term, with
    ``-k`` on each spin density where the restricted function has ``-k/2``
    on the total -- except the Coulomb half of D4, which squares the trace
    summed over both spins; with the spins in two groups its trace is
    accumulated in ``ud`` and squared after the ``last`` one.  The partners
    that go spherical in place (``j3c_oo1p``) are converted once, in the
    first group; ``j2c_10c`` stays Cartesian (the Coulomb half of D5 needs
    it so) and its spherical copy ``j10s`` is kept in ``ud``.
    """
    nsph = sa.nsph
    nao = dm_factor_l.shape[1]
    ns = n1 - n0
    C = dm_factor_l[n0:n1]
    for x in range(3):
        H._trmm(sa, ksph[x].reshape(nsph, -1))
    st('D8 L^-1 K')
    # D4 on Y = L^-1 K, unit weights: each spin's P o R + Q (fastdfhessip's
    # _oo_gram per spin, as this module's _oo_gram does)
    rq = (H._oo_gram_rq_syrk if _env('FASTDFHESSIP_D4', 'syrk') == 'syrk'
          else H._oo_gram_rq)
    need = 3*nao*nocc + 3*nao*nao
    wk = work if work.size >= need else cp.empty(need)
    ones = cp.ones(nsph)
    acc = None
    for g in range(ns):
        R, Q = rq(ksph[:, :, g], 0, nsph, wk, nocc, C[g], ones)
        R *= C[g].dot(C[g].T)
        R += Q
        Q = None
        if acc is None:
            acc = R
        else:
            acc += R
        R = None
    t = contract('au,xyuv->axyv', atm_ind, acc)
    acc = None
    contract('axyv,bv->abxy', t, atm_ind, -k_factor*2, beta=1, out=ejk)
    t = None
    if j_factor != 0:
        d = contract('xrnui,nui->xru', ksph, C)
        if 'dY' in ud:
            ud['dY'] += d
        else:
            ud['dY'] = d
        d = None
        if last:
            a = contract('au,xru->axr', atm_ind, ud.pop('dY'), alpha=2.)
            contract('pxr,qyr->pqxy', a, a, j_factor, beta=1, out=ejk)
            a = None
    st('D4 oo_atm')
    for x in range(3):
        H._trmm(sa, ksph[x].reshape(nsph, -1), trans=True)
    z = ksph
    st('D8 L^-T Y')
    if j_factor != 0:
        # (10|0)(0|1)(0|00): linear in Z, so each group adds its spins
        d = contract('xtnui,nui->xtu', z, C)
        at = contract('au,xtu->axt', atm_ind, d, alpha=2.)
        tmp, _ = _allocate((3, sa.ncart, nsph), work)
        cp.multiply(j2c_10c, -auxvec[None, :, None], out=tmp)
        tmp += auxvec_ipauxp[:, :, None] * sa.ms
        contract('pxt,yst->psxy', at, tmp, j_factor, 1, h_ao_aux)
        d = at = tmp = None
    if 'j10s' not in ud:
        for x in range(3):
            sa.to_sph(j3c_oo1p[x], j3c_oo1p[x])
        j10s = cp.empty((3, nsph, nsph))
        for y in range(3):
            sa.to_sph(j2c_10c[y], j10s[y])
        ud['j10s'] = j10s
        ud['h_sph'] = cp.zeros((natm, nsph, 3, 3))
    j10s = ud['j10s']
    h_sph = ud['h_sph']
    t1s = j3c_oo1p[:, :nsph, n0:n1]
    if _env('FASTDFHESSIP_D5', 'group') != 'atom':
        # item 26
        H._d5_chol(sa, z, t1s, dm_oo[:, n0:n1], C, aoslices, -k_factor*2,
                   k_factor*2, j10s, h_sph, work)
        st('D5 per-atom')
        return
    for i, (p0, p1) in enumerate(zip(aoslices[:-1], aoslices[1:])):
        t1, work1 = _allocate((3, nsph, ns, p1-p0, nocc), work)
        contract('ysnji,nui->ysnuj', t1s, C[:, p0:p1], out=t1)
        contract('xsnuj,ysnuj->sxy', z[:, :, :, p0:p1], t1, -k_factor*2, 1,
                 h_sph[i])
        pa, work1 = _allocate((sa.ncart, ns, p1-p0, nocc), work)
        contract('snji,nui->snuj', dm_oo[:, n0:n1], C[:, p0:p1], out=pa)
        ps = sa.to_sph_pinv(pa, pa)
        m, work1 = _allocate((3, nsph, nsph), work1)
        contract('snuj,xtnuj->xst', ps, z[:, :, :, p0:p1], out=m)
        contract('xst,yst->sxy', m, j10s, k_factor*2, 1, h_sph[i])
    st('D5 per-atom')


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
            natm, metric_size, gram, nsp=2, rows=False, after_h_aux=False):
    """Upstream's sizing of the unrestricted second pass.

    One change: upstream asserts that one auxiliary function's per-atom
    occupied block fits (``natm 12 nocc^2`` words), which item 2's Gram form
    never builds -- so with it on, that block's size is not a limit.
    """
    if not after_h_aux:
        word_avail -= 4 * naux**2  # j2c_10v, which the h_aux block leaves
    word_avail -= 3 * batch_size * nao_pair * 2  # eval_ip1, eval_ipaux
    # j3c_100, nsp spins of it, and _oo_gram's Coulomb temporary; with
    # `rows` (item 34, W40) item 33's D5, which has none of upstream's
    # naux-sized per-atom temporaries
    aux_unit1 = 3*nsp*nao*nocc + 3*naux
    # (rows: and D4's few (3, 3, nao, nao) results and temporaries per
    # spin, which the Gram form allocates beside the two buffers)
    held = (H._D5_ROWS_WORDS + 64*nao*nao if rows else
            12*naux*nao_on_atom*nocc)
    aux_batch_size = (min(int(word_avail*.85), word_avail-held)
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
    """Upstream's first pass: ``(1|00)``, both spins (``j3c_oo1`` a
    ``_SymRows``: item 34, written packed)."""
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((3, nao_pair, batch_size))
    buf1 = cp.empty(2 * blksize * nocc * nao)
    if isinstance(j3c_oo1, H._SymRows):
        j3c_oo1.tmp = cp.empty((blksize, 2, nocc, nocc))
    aux0 = 0
    for kbatch in range(num_aux_batches):
        compressed = eval_ipaux(kbatch, out=buf)
        naux_in_batch = compressed.shape[-1]
        _ipaux_batch(compressed, aux0, naux_in_batch, blksize, j3c_full, buf1,
                     nao, nocc, pair_addresses, dm_factor_l, j3c_oo1)
        aux0 += naux_in_batch
    if isinstance(j3c_oo1, H._SymRows):
        j3c_oo1.tmp = None


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
            if isinstance(j3c_oo1, H._SymRows):
                t = contract('niqr,nqj->rnij', tmp, dm_factor_l, alpha=-1,
                             out=j3c_oo1.tmp[:dk])
                j3c_oo1.put(i*j3c_oo1.xstride + a0, t)
            else:
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
    tiles = H._ip2_tiles(int3c2e_opt, omega, spmult, None, ao_pair_loc)
    if tiles is not None:
        bufp = cp.empty((nao_pair*batch_size))
    aux0 = aux1 = 0
    # item 29 (W39): the pseudo-density on the lower triangle's row tiles,
    # both spins in one GEMM of depth 2 nocc
    low = (H._LowTri(pair_addresses, nao)
           if _env('FASTDFHESSIP_PDLOW', '1') != '0' and omega == 0
           else None)
    if low is not None:
        _S['pd_low'] = _S.get('pd_low', 0) + 1
        cn = cp.ascontiguousarray(dm_factor_l.transpose(1, 0, 2)).reshape(
            nao, 2*nocc)
        if j_factor != 0:
            dm_pairs = dm.ravel()[low.addr]

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
            if low is not None:
                contract('rnji,nqj->niqr', dm_oo[aux0:aux1], dm_factor_l,
                         out=tmp)
                low.c_t(cn, tmp, dk, -k_factor, dm_tensor)
                cp.take(dm_tensor.reshape(-1, dk), low.addr, axis=0,
                        out=compressed[:, k0:k1])
                if j_factor != 0:
                    H._ADD_OUTER(dm_pairs, auxvec[aux0:aux1],
                                 np.float64(j_factor), np.int32(dk),
                                 compressed[:, k0:k1])
                continue
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
        if tiles is not None:
            # fastdfhessip item 15 (W29): the kernel launched as its tile
            ks0, ks1 = ksh_offsets_cpu[kbatch], ksh_offsets_cpu[kbatch+1]
            perm = tiles.permutation(ks0, ks1)
            if perm is not None:
                compressed = cp.take(compressed, perm, axis=1, out=ndarray(
                    (nao_pair, naux_in_batch), buffer=bufp))
            tiles.launch(ejk, compressed, None, int3c2e_envs, omega,
                         lr_factor, sr_factor, ks0, ks1, aux_ao_offset,
                         naux_in_batch)
            continue
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
    # item 23 (W38): the occupied index in batches when both spins' density
    # factors do not fit -- upstream's twin holds dm_3c and dm_oo for every
    # occupied orbital, 2 naux nocc (nao + nocc) words, 29 GB at 113 atoms,
    # and asserts on a 40 GB card.  When they would take more than 70 % of
    # what the metric leaves (the restricted function's rule, with the spin
    # index), stages A-C run per batch of occupied orbitals, each batch
    # holding dm_3c[r,n,i,q] for its own i and dm_oo[r,n,i,j] for every j.
    # Stages D and E then need dm_3c one auxiliary batch at a time and every
    # orbital of it: by default (`host`) each batch's dm_3c is copied to
    # page-locked host memory and D and E run once, reading their rows back
    # per auxiliary batch, with dm_oo whole on the device.  `occ` runs D and
    # E per occupied batch as well, as the restricted function does -- which
    # repeats their integrals and D2's products once per batch, because D2
    # contracts the batch's i away (W38 measures both).
    # FASTDFHESSIP_H1UBATCH=0 is upstream's single batch
    ubatch = _env('FASTDFHESSIP_H1UBATCH', 'host')
    num_occ_batches = 1
    nocc_per_batch = nocc
    if ubatch != '0':
        wa = word_avail - naux**2 * 3
        num_occ_batches = (int(2*naux*nocc*(nao+nocc) / (wa*.7) + 1)
                           if wa > 0 else nocc)
        if _env('FASTDFHESSIP_OCC_BATCH', ''):
            # test-only: forced, the batching is taken whatever fits
            num_occ_batches = max(num_occ_batches, 2)
    host = num_occ_batches > 1 and ubatch == 'host'
    # the host copies on a stream of their own, each behind the stage that
    # does not need it: a batch's copy out behind its stage C, and the next
    # auxiliary batch's rows in behind the D and E work of this one, in two
    # device buffers.  FASTDFHESSIP_H1UHOST_ASYNC=0 copies synchronously,
    # into one
    hasync = host and _env('FASTDFHESSIP_H1UHOST_ASYNC', '1') != '0'
    if num_occ_batches > 1 and not host:
        nocc_per_batch = (nocc + num_occ_batches - 1) // num_occ_batches
        nocc_per_batch = min(nocc, (nocc_per_batch + 7) & 0xffffff8)
        nocc_per_batch = _forced_occ_batch(nocc_per_batch)
        num_occ_batches = (nocc + nocc_per_batch - 1) // nocc_per_batch
    nb = nocc_per_batch
    if host:
        # stages D and E hold dm_oo whole and one auxiliary batch of dm_3c;
        # the occupied batch of stages A-C is sized after the D loop is
        word_avail -= naux**2 * 3  # metric_v
        word_avail -= 2 * naux*nocc*nocc  # dm_oo, every batch's
    else:
        word_avail -= 2 * naux*nb*nao  # dm_3c ~ (aux,i,a)
        word_avail -= naux**2 * 3  # metric_v
        word_avail -= 2 * naux*nocc*nb  # dm_oo ~ (aux,i,j)
    assert word_avail > 0, 'Insufficient GPU memory'

    mem_sufficient = (natm*3*2*nao**2 < word_avail * .6 and
                      _env('FASTDFHESSIP_H1KAO', '1') != '0')
    # the per-atom AO-pair matrices hold exchange terms only
    k_ao = mem_sufficient and do_k
    _S['h1_kocc'] = do_k and not mem_sufficient
    if k_ao:
        vhf_atm_ao = cp.zeros((natm, 3, 2, nao, nao))
        word_avail -= vhf_atm_ao.size

    pair_size_max = max(nao_pair, 2*nb*nao)
    # with dm_3c on the host, each auxiliary function of a batch is also a
    # row of the device copy of its dm_3c
    _unit = 6*pair_size_max + (2*nocc*nao * (2 if hasync else 1) if host
                               else 0)
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
    # W30's threshold (4e6) priced the occupied form with 2 natm launches
    # per block and direction; with item 20's batched D2 it wins or ties at
    # every size measured (`docs/README_fastdfh1d2.md` §3), so it is taken
    # wherever it saves arithmetic
    d2_batch_on = jbatch and _env('FASTDFHESSIP_H1D2', 'batched') != 'atom'
    occ_k = hoist and (h1atom == 'occ' or
                       (h1atom == 'auto' and score > float(
                           _env('FASTDFHESSIP_H1OCC_SCORE',
                                '0' if d2_batch_on else '4e6'))))
    _S['h1_occ'] = bool(occ_k)
    # item 20 (W35): D2's per-atom products as a few batched GEMMs; the
    # memory-short branch (no per-atom AO matrices) forms the same two
    # products per atom and takes it too, with item 12's hoisted arrays
    kshort = do_k and not mem_sufficient and h1atom != 'upstream'
    d2_batched = (occ_k or kshort) and d2_batch_on
    _S['h1_d2_batched'] = bool(d2_batched)
    t_on = hoist or d2_batched
    s_on = occ_k or d2_batched
    # with the occupied index batched (item 23), D1's j3c . C_o covers the
    # batch only, and the occupied form needs it over every occupied orbital
    pri_full = s_on and num_occ_batches > 1 and not host
    blk_unit = (nao**2 + (nao+nocc*2) * max(nao_on_atom, nocc) +
                (2*nao*(nao + (nocc if s_on else 0)) if t_on else 0) +
                (2*nao*nocc if pri_full else 0))
    blksize = int((word_avail*.95 - batch_size*_unit) / blk_unit) // 8 * 8
    assert blksize > 0, 'Insufficient GPU memory'
    blksize = min(blksize, batch_size)
    # item 21 (W35): the integral batch takes 80 % of the budget and the D
    # loop's block what is left, so on a short card the block falls to a
    # few functions while the batch still holds dozens.  Below
    # FASTDFHESSIP_H1BLK_MIN functions the batch gives the block that many
    # (the integral evaluator is rebuilt for the smaller batch)
    blk_min = int(_env('FASTDFHESSIP_H1BLK_MIN', '32'))
    _S['h1_rebalanced'] = False
    if blksize < min(blk_min, batch_size):
        b2 = int((word_avail*.95 - blk_min*blk_unit) / _unit)
        if b2 >= max(largest_shell_nao, blk_min):
            batch_size = min(b2, max(largest_shell_nao, naux))
            with _spmult(spmult, short_range=omega < 0):
                eval_j3c, aux_sorting, _, aux_offsets = \
                    int3c2e_opt.int3c2e_evaluator(
                        aux_batch_size=batch_size, reorder_aux=True,
                        cart=True, omega=omega)
            batch_size = min(batch_size,
                             int((aux_offsets[1:]-aux_offsets[:-1]).max()))
            num_aux_batches = len(aux_offsets) - 1
            blksize = min(blk_min, batch_size)
            _S['h1_rebalanced'] = True
    # test-only: a smaller block, as a smaller card would leave (W35)
    blksize = min(blksize, int(_env('FASTDFHESSIP_H1BLK', blksize)))
    _S['h1_blksize'] = blksize
    # item 35 (W41): in host mode, stage A once over every orbital, its
    # j3c_00 written into the host array dm_3c will take column by column,
    # instead of the integrals once per occupied batch.
    # FASTDFHESSIP_H1UA1=0 is W38's stage A per batch
    a_once = host and _env('FASTDFHESSIP_H1UA1', '1') != '0'
    # stage B's orbitals per chunk (test-only: FASTDFHESSIP_H1UB_CHUNK, so
    # that a small molecule's batch is fetched in several chunks)
    bchunk = int(_env('FASTDFHESSIP_H1UB_CHUNK', '8'))
    if host:
        # stages A-C: one occupied batch of dm_3c (and of dm_oo, before it
        # joins the whole one) beside stage A's two buffers (item 35: B's
        # chunk buffer instead), in what the D loop's budget leaves before
        # its own buffers are allocated
        held = (naux*2*bchunk*nao if a_once else
                nao*nao*blksize + batch_size*nao_pair)
        nb = int((word_avail*.9 - held) / (2*naux*(nao+nocc)))
        nb = min(nocc, nb // 8 * 8 if nb >= 8 else nb)
        assert nb > 0, 'Insufficient GPU memory'
        nb = _forced_occ_batch(nb)
        num_occ_batches = (nocc + nb - 1) // nb
        nb = (nocc + num_occ_batches - 1) // num_occ_batches
    batched = num_occ_batches > 1
    _S['occ_batches_u'] = int(num_occ_batches)
    _S['h1_host'] = bool(host)
    _S['h1_a_once'] = bool(a_once)
    st.note(naux=naux, batch_size=batch_size, blksize=blksize,
            aux_batches=num_aux_batches, occ_batches=num_occ_batches,
            occ_k=int(bool(occ_k)), k_ao=int(bool(k_ao)),
            d2=('batched' if d2_batched else 'atom' if occ_k else
                'short' if do_k and not k_ao else 'ao' if do_k else '-'))
    d2_plan = _h1d2.AtomGroups(aoslices) if d2_batched else None

    aux_idx, aux_slices = _argsort_aux_by_atom(auxmol, aux_sorting)
    counts = aux_slices[:, 1] - aux_slices[:, 0]
    atm_id_for_aux = np.empty(naux, dtype=int)
    atm_id_for_aux[aux_idx] = np.repeat(np.arange(auxmol.natm), counts)
    aux_filling_order = np.empty(naux, dtype=int)
    aux_filling_order[aux_idx] = np.arange(naux)

    metric_w, metric_v = _factorize_j2c(auxmol, aux_sorting, omega)
    nw = metric_v.shape[1]
    if j_factor != 0 or batched:
        dm = contract('npi,nqi->pq', orbo, orbo)
    if j_factor != 0 and batched:
        # item 23: dm_oo's trace needs every orbital; from the total density
        # instead, V W^-1 V^T (j3c . dm), as the restricted function forms it
        auxvec = cp.empty(naux)
    eval_ip1 = _ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ip1(omega, 27), batch_size,
        'fill_int3c2e_ip1', omega, spmult)[0]
    eval_ipaux = _ip1_evaluator(
        int3c2e_opt, int3c2e_scheme_ipaux(omega, 27), batch_size,
        'fill_int3c2e_ipaux', omega, spmult)[0]
    st('setup')

    def _stage_de(nocc0, nocc1, dm_3c, dm_oo, first):
        # stages D and E for the occupied orbitals nocc0:nocc1: the batch's
        # dm_3c on the device, or (item 23's host mode, dm_3c None) every
        # orbital's, read back from dm3_host one auxiliary batch at a time;
        # dm_oo is the batch's rows i and every j
        nob = nocc1 - nocc0
        orbo_b = orbo[:, :, nocc0:nocc1]
        vhf_i = vhf_atm[:, :, :, nocc0:nocc1]
        # (10|0)(0|0)(0|00)
        work_size = max(
            nao**2*blksize + (3*pair_size_max + 6*nob*nao) * batch_size,
            6*nob*nao*batch_size * 2,
            6*pair_size_max*batch_size +
            (nao*nao + (nao+nocc)*2 * max(nao_on_atom, nocc)) * blksize)
        # item 4's array lives at the tail of the same buffer: it is dead
        # before the (00|1) half of each batch reuses that space
        t_size = 2*nao*nao*blksize if t_on else 0
        s_size = 2*nao*nocc*blksize if s_on else 0
        p_size = 2*nao*nocc*blksize if pri_full else 0
        work = cp.empty(work_size + t_size + s_size + p_size)
        t_buf = work[work_size:work_size+t_size] if t_on else None
        s_buf = (work[work_size+t_size:work_size+t_size+s_size] if s_on
                 else None)
        p_buf = work[work_size+t_size+s_size:] if pri_full else None
        aux0 = aux1 = 0
        for kbatch in range(num_aux_batches):
            naux_in_batch = aux_offsets[kbatch+1] - aux_offsets[kbatch]
            aux0, aux1 = aux1, aux1 + naux_in_batch
            if dm_3c is None and hasync:
                # item 23: this auxiliary batch's rows of dm_3c, every
                # occupied orbital, fetched from the host while the previous
                # batch ran, and the next batch's fetch issued behind it
                bi = kbatch % 2
                if kbatch == 0:
                    _fetch(0, 0)
                cp.cuda.Stream.null.wait_event(ready[bi])
                if kbatch + 1 < num_aux_batches:
                    if used[1 - bi] is not None:
                        xs.wait_event(used[1 - bi])
                    _fetch(kbatch + 1, 1 - bi)
                dm3k = stage_bufs[bi][:naux_in_batch]
                st('D dm_3c in')
            elif dm_3c is None:
                # item 23: this auxiliary batch's rows of dm_3c, every
                # occupied orbital, back from the host
                dm3k = stage_bufs[0][:naux_in_batch]
                runtime.memcpy(int(dm3k.data.ptr),
                               dm3_host.ctypes.data +
                               int(aux0) * int(dm3k[0].nbytes),
                               int(dm3k.nbytes), runtime.memcpyHostToDevice)
                st('D dm_3c in')
            else:
                dm3k = dm_3c[aux0:aux1]

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
                if t_on:
                    # item 4: the occupied-contracted density of both spins
                    # for every AO at once, which upstream forms per atom and
                    # per x
                    t_all = ndarray((2, nao, dk, nao), buffer=t_buf)
                    contract('nui,rniq->nurq', orbo_b, dm3k[_aux0-aux0:_aux1-aux0],
                             out=t_all)
                if s_on:
                    s_all = ndarray((2, nao, dk, nocc), buffer=s_buf)
                    contract('nui,rnij->nurj', orbo_b, dm_oo[_aux0:_aux1],
                             out=s_all)
                for x in range(3):
                    if d2_batched:
                        # item 20: j3c held as jT[q,p,r] and D1's product
                        # with C_o as (n, i, p, r); D1 and D3 read it with
                        # the two AO indices swapped (fastdfhessip's
                        # restricted loop)
                        jT = _h1d2.contiguous_block(j3c_full, dk, x == 0)
                        jT[i_addr, j_addr] = compressed_dj[x, :, k0:k1]
                        jT[j_addr, i_addr] = compressed_di[x, :, k0:k1]
                        priT = ndarray((2, nob, nao, dk), buffer=work1)
                        contract('qpr,nqi->nipr', jT, orbo_b, out=priT)
                        contract('nipr,rniq->npq', priT, dm3k[_aux0-aux0:_aux1-aux0],
                                 -k_factor, beta=1, out=vhf1[x])
                        st('D1 vhf1')
                        if pri_full:
                            priT = ndarray((2, nocc, nao, dk), buffer=p_buf)
                            contract('qpr,nqi->nipr', jT, orbo, out=priT)
                        d2_plan.d2(jT, s_all, priT, t_all, vhf_atm[:, x],
                                   -k_factor)
                        priT = None
                        st('D2 K per-atom')
                        if j_factor != 0:
                            if first:
                                vj1 = contract('qpr,r->pq', jT,
                                               auxvec[_aux0:_aux1])
                                vhf1[x, 0] += j_factor * vj1
                                vhf1[x, 1] += j_factor * vj1
                            w_pr = contract('qpr,pq->pr', jT, dm)
                            av_atm = atm_ind.dot(w_pr)
                            contract('rniq,ar->aniq', dm3k[_aux0-aux0:_aux1-aux0],
                                     av_atm, 2*j_factor, beta=1,
                                     out=vhf_i[:, x])
                            vj1 = w_pr = av_atm = None
                        st('D3 J per-atom')
                        continue
                    j3c[j_addr, i_addr] = compressed_dj[x, :, k0:k1]
                    j3c[i_addr, j_addr] = compressed_di[x, :, k0:k1]
                    if do_k:
                        tmp = ndarray((2, nao, dk, nob), buffer=work1)
                        j3c_pri = contract('pqr,nqi->npri', j3c, orbo_b,
                                           out=tmp)
                        contract('npri,rniq->npq', j3c_pri,
                                 dm3k[_aux0-aux0:_aux1-aux0], -k_factor, beta=1,
                                 out=vhf1[x])
                    st('D1 vhf1')
                    if not do_k:
                        pass
                    elif occ_k:
                        # sum_{u in A, r} of the two halves of X + X^T, each
                        # already on the occupied side: upstream's
                        # memory-short branch with both of its intermediates
                        # formed once per block (j3c_pri is D1's when the
                        # occupied index is not batched), which is
                        # 8 nao^2 nocc flops per auxiliary function and
                        # direction against 4 nao^3 for the AO-pair form
                        if pri_full:
                            j3c_pri = ndarray((2, nao, dk, nocc),
                                              buffer=p_buf)
                            contract('pqr,nqi->npri', j3c, orbo, out=j3c_pri)
                        for i, (p0, p1) in enumerate(zip(aoslices[:-1],
                                                         aoslices[1:])):
                            contract('uqr,nurj->njq', j3c[p0:p1],
                                     s_all[:, p0:p1], -k_factor, beta=1,
                                     out=vhf_atm[i, x])
                            contract('nurj,nurq->njq', j3c_pri[:, p0:p1],
                                     t_all[:, p0:p1], -k_factor, beta=1,
                                     out=vhf_atm[i, x])
                        j3c_pri = None
                    elif hoist:
                        for i, (p0, p1) in enumerate(zip(aoslices[:-1],
                                                         aoslices[1:])):
                            contract('upr,nurq->npq', j3c[p0:p1],
                                     t_all[:, p0:p1], -k_factor, beta=1,
                                     out=vhf_atm_ao[i, x])
                    elif k_ao:
                        for i, (p0, p1) in enumerate(zip(aoslices[:-1],
                                                         aoslices[1:])):
                            if p1 - p0 < nob:
                                tmp = ndarray((2, p1-p0, dk, nao),
                                              buffer=work1)
                                contract('nui,rniq->nurq', orbo_b[:, p0:p1],
                                         dm3k[_aux0-aux0:_aux1-aux0], out=tmp)
                                contract('upr,nurq->npq', j3c[p0:p1], tmp,
                                         -k_factor, beta=1,
                                         out=vhf_atm_ao[i, x])
                            else:
                                tmp = ndarray((nao, dk, 2, nob),
                                              buffer=work1)
                                contract('pqr,npi->qrni', j3c[p0:p1],
                                         orbo_b[:, p0:p1], out=tmp)
                                contract('prni,rniq->npq', tmp,
                                         dm3k[_aux0-aux0:_aux1-aux0], -k_factor,
                                         beta=1, out=vhf_atm_ao[i, x])
                    else:
                        for i, (p0, p1) in enumerate(zip(aoslices[:-1],
                                                         aoslices[1:])):
                            tmp, buf2 = _allocate((2, p1-p0, dk, nocc), work1)
                            tmp1 = ndarray((2, p1-p0, dk, nao), buffer=buf2)
                            contract('nui,rnij->nurj', orbo_b[:, p0:p1],
                                     dm_oo[_aux0:_aux1], out=tmp)
                            contract('uqr,nurj->njq', j3c[p0:p1], tmp,
                                     -k_factor, beta=1, out=vhf_atm[i, x])
                            contract('uqr,nqj->nurj', j3c[p0:p1], orbo,
                                     out=tmp)
                            contract('nui,rniq->nurq', orbo_b[:, p0:p1],
                                     dm3k[_aux0-aux0:_aux1-aux0], out=tmp1)
                            contract('nurj,nurq->njq', tmp, tmp1, -k_factor,
                                     beta=1, out=vhf_atm[i, x])
                    st('D2 K per-atom')
                    if j_factor != 0:
                        if first:
                            # the same Coulomb matrix for both spins: once
                            vj1 = contract('pqr,r->pq', j3c,
                                           auxvec[_aux0:_aux1])
                            vhf1[x, 0] += j_factor * vj1
                            vhf1[x, 1] += j_factor * vj1
                            vj1 = None
                        if jbatch:
                            # item 5: sum_{p in A, q} j3c[p,q,r] dm[p,q] for
                            # every atom A is a row-wise contraction and a
                            # GEMM against the atom indicator
                            w_pr = contract('pqr,pq->pr', j3c, dm)
                            av_atm = atm_ind.dot(w_pr)
                            contract('rniq,ar->aniq', dm3k[_aux0-aux0:_aux1-aux0],
                                     av_atm, 2*j_factor, beta=1,
                                     out=vhf_i[:, x])
                            w_pr = av_atm = None
                        else:
                            for i, (p0, p1) in enumerate(zip(aoslices[:-1],
                                                             aoslices[1:])):
                                auxvec1 = cp.einsum('pqr,pq->r', j3c[p0:p1],
                                                    dm[p0:p1])
                                contract('rniq,r->niq', dm3k[_aux0-aux0:_aux1-aux0],
                                         auxvec1, 2*j_factor, beta=1,
                                         out=vhf_i[i, x])
                    st('D3 J per-atom')

            if d2_batched:
                _h1d2.restore(j3c_full, naux_in_batch)
            # (00|1)(0|0)(0|00)
            compressed_dk += compressed_di
            j3c_aux, work1 = _allocate((3, naux_in_batch, 2, nob, nao), work)
            off = max(j3c_full.size + compressed_dk.size, j3c_aux.size)
            j3c_aux_tmp = ndarray(j3c_aux.shape, buffer=work[off:])
            if batched:
                # item 23: the trace over every orbital, from the total
                # density, as the restricted function forms it
                auxvec_ipaux_b = cp.empty((3, naux_in_batch))
            for k0, k1 in lib.prange(0, naux_in_batch, blksize):
                dk = k1 - k0
                j3c = j3c_full[:, :, :dk]
                for i in range(3):
                    fill_symmetric(compressed_dk[i], pair_addresses, nao, k0,
                                   k1, out=j3c)
                    # d/dX = -d/dr
                    contract('pqr,npi->rniq', j3c, orbo_b, alpha=-1,
                             out=j3c_aux_tmp[i, k0:k1])
                    if batched:
                        contract('pqr,pq->r', j3c, dm, alpha=-1,
                                 out=auxvec_ipaux_b[i, k0:k1])

            idx = np.argsort(aux_filling_order[aux0:aux1])
            j3c_aux = cp.take(j3c_aux_tmp, idx, axis=1, out=j3c_aux)

            dm_3c_batch, work1 = _allocate((naux_in_batch, 2, nob, nao),
                                           work1)
            dm_oo_batch, work1 = _allocate((naux_in_batch, 2, nob, nocc),
                                           work1)
            # dm_3c_batch overlaps j3c_aux_tmp, which is copied out above
            dm_3c_batch = cp.take(dm3k, idx, axis=0,
                                  out=dm_3c_batch)
            dm_oo_batch = cp.take(dm_oo[aux0:aux1], idx, axis=0,
                                  out=dm_oo_batch)
            if j_factor != 0:
                auxvec_batch = auxvec[aux0:aux1][idx]
            if batched:
                auxvec_ipaux_b = auxvec_ipaux_b[:, idx]

            counts = np.bincount(atm_id_for_aux[aux0:aux1])
            oo_buf = None
            if work1.size > 6*counts.max()*nob*nocc:
                oo_buf = work1
            st('E ipaux fill')

            p0 = p1 = 0
            for i, count in enumerate(counts):
                if count == 0:
                    continue
                p0, p1 = p1, p1 + count
                if not do_k:
                    if not batched:
                        auxvec_ipaux = contract('xrnip,npi->xr',
                                                j3c_aux[:, p0:p1], orbo)
                elif mem_sufficient:
                    if not batched:
                        auxvec_ipaux = contract('xrnip,npi->xr',
                                                j3c_aux[:, p0:p1], orbo)
                    contract('xrnip,rniq->xnpq', j3c_aux[:, p0:p1],
                             dm_3c_batch[p0:p1], -k_factor, beta=1,
                             out=vhf_atm_ao[i])
                else:
                    tmp = ndarray((3, p1-p0, 2, nob, nocc), buffer=oo_buf)
                    j3c_1 = contract('xrniq,nqj->xrnij', j3c_aux[:, p0:p1],
                                     orbo, out=tmp)
                    if not batched:
                        auxvec_ipaux = cp.einsum('xrnii->xr', j3c_1)
                    contract('rniq,xrnij->xnjq', dm_3c_batch[p0:p1], j3c_1,
                             -k_factor, beta=1, out=vhf_atm[i])
                    contract('xrniq,rnij->xnjq', j3c_aux[:, p0:p1],
                             dm_oo_batch[p0:p1], -k_factor, beta=1,
                             out=vhf_atm[i])
                if j_factor != 0:
                    if batched:
                        auxvec_ipaux = auxvec_ipaux_b[:, p0:p1]
                    contract('rniq,xr->xniq', dm_3c_batch[p0:p1],
                             auxvec_ipaux, j_factor, beta=1, out=vhf_i[i])
                    contract('xrniq,r->xniq', j3c_aux[:, p0:p1],
                             auxvec_batch[p0:p1], j_factor, beta=1,
                             out=vhf_i[i])
            if dm_3c is None and hasync:
                # the buffer is free for the fetch two batches on
                used[bi] = cp.cuda.Event(disable_timing=True)
                used[bi].record(cp.cuda.Stream.null)
            st('E per-aux-atom')
        work = work1 = buf0 = buf1 = oo_buf = dm_3c_batch = dm_oo_batch = None
        j3c_full = j3c = j3c_1 = j3c_aux = j3c_aux_tmp = tmp = tmp1 = None
        compressed_di = compressed_dj = compressed_dk = t_all = s_all = None
        auxvec_ipaux = auxvec_ipaux_b = None
        # every view of `work` too, or the next batch allocates beside it
        t_buf = s_buf = p_buf = jT = priT = j3c_pri = None
        dm_oo = dm_3c = dm3k = None

    if host:
        # item 23: every batch's dm_3c, page-locked (the pool keeps it for
        # the next call of the same size), and dm_oo whole on the device
        dm3_host = _dfpin.empty_mapped_pooled((naux, 2, nocc, nao))
        xs = cp.cuda.Stream(non_blocking=True) if hasync else None
        assert int(aux_offsets[0]) == 0
        # page-locking it is paid once per size: the pool keeps it (W12)
        st('A host buffer')
        dm_oo_all = cp.empty((naux, 2, nocc, nocc))
    if a_once:
        # item 35: stage A for every orbital, one integral pass, each block
        # of j3c_00 copied into its rows of dm3_host behind the next block's
        # integrals (two device buffers on the copy stream when asynchronous)
        j3c_full = cp.zeros((nao, nao, blksize))
        buf = cp.empty((batch_size, nao_pair))
        abufs = [cp.empty((blksize, 2, nocc, nao))
                 for _ in range(2 if hasync else 1)]
        adone = [None, None]
        if hasync:
            ev = cp.cuda.Event(disable_timing=True)
            ev.record(cp.cuda.Stream.null)
            xs.wait_event(ev)
        row = 2 * nocc * nao * 8
        aux0 = aux1 = 0
        nblk = 0
        for kbatch in range(num_aux_batches):
            compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
            naux_in_batch = compressed.shape[1]
            for k0, k1 in lib.prange(0, naux_in_batch, blksize):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                j3c = j3c_full[:, :, :dk]
                fill_symmetric(compressed, pair_addresses, nao, k0, k1,
                               out=j3c)
                b = nblk % len(abufs)
                nblk += 1
                if adone[b] is not None:
                    cp.cuda.Stream.null.wait_event(adone[b])
                out = abufs[b][:dk]
                contract('pqr,npi->rniq', j3c, orbo, out=out)
                if j_factor != 0:
                    contract('pqr,pq->r', j3c, dm, out=auxvec[aux0:aux1])
                if hasync:
                    ev = cp.cuda.Event(disable_timing=True)
                    ev.record(cp.cuda.Stream.null)
                    xs.wait_event(ev)
                    runtime.memcpyAsync(int(dm3_host.ctypes.data + aux0*row),
                                        int(out.data.ptr), int(dk * row),
                                        runtime.memcpyDeviceToHost, xs.ptr)
                    adone[b] = cp.cuda.Event(disable_timing=True)
                    adone[b].record(xs)
                else:
                    runtime.memcpy(int(dm3_host.ctypes.data + aux0*row),
                                   int(out.data.ptr), int(dk * row),
                                   runtime.memcpyDeviceToHost)
        if hasync:
            xs.synchronize()
        j3c_full = j3c = buf = compressed = out = abufs = adone = None
        st('A j3c_00 (once)')

    def _j2c_10():
        j2c_10 = int2c2e_ip1(auxmol, sort_output=False, omega=omega)
        inv_aux_sorting = cp.empty(naux, dtype=int)
        inv_aux_sorting[aux_sorting] = cp.arange(naux)
        return j2c_10[:, inv_aux_sorting[aux_idx, None], inv_aux_sorting]

    if host:
        # item 35: every batch's device arrays in the same buffers.  Each is
        # over GPU4PySCF's 100 MB pool threshold, so allocated per batch it
        # is a cudaFree and a cudaMalloc of the same 7 GB (taxol+), and the
        # malloc waits for the driver to take the free back: 0.25 s a batch
        bbuf = cp.empty(naux*2*nb*nao)
        obuf = cp.empty(naux*2*nb*nocc)
        wbuf = cp.empty(nw*2*bchunk*nao)
    for it in range(num_occ_batches):
        nocc0 = it * nb
        nocc1 = min(nocc, nocc0 + nb)
        nob = nocc1 - nocc0
        orbo_b = orbo[:, :, nocc0:nocc1]
        # the views every term whose output index is the batch's i writes
        vhf_i = vhf_atm[:, :, :, nocc0:nocc1]

        aux0 = aux1 = 0
        if a_once:
            j3c_00 = ndarray((naux, 2, nob, nao), buffer=bbuf)
            # item 35: the batch's columns of j3c_00 back from the host,
            # eight orbitals at a time, each chunk's fetch behind stage B's
            # work on the chunk before it
            chunks = list(lib.prange(0, nob, bchunk))
            fdone = [None] * len(chunks)

            def _fetch_a(c):
                i0, i1 = chunks[c]
                args = (int(j3c_00.data.ptr) + int(i0*nao*8), int(nob*nao*8),
                        int(dm3_host.ctypes.data + (nocc0 + i0)*nao*8),
                        int(nocc*nao*8), int((i1 - i0)*nao*8), int(naux*2),
                        runtime.memcpyHostToDevice)
                if hasync:
                    runtime.memcpy2DAsync(*args, xs.ptr)
                    fdone[c] = cp.cuda.Event(disable_timing=True)
                    fdone[c].record(xs)
                else:
                    runtime.memcpy2D(*args)
            if hasync:
                ev = cp.cuda.Event(disable_timing=True)
                ev.record(cp.cuda.Stream.null)
                xs.wait_event(ev)
            _fetch_a(0)
            st('A j3c_00 in')
        else:
            j3c_full = cp.zeros((nao, nao, blksize))
            buf = cp.empty((batch_size, nao_pair))
            j3c_00 = (ndarray((naux, 2, nob, nao), buffer=bbuf) if host else
                      cp.empty((naux, 2, nob, nao)))
        for kbatch in range(0 if a_once else num_aux_batches):
            compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
            naux_in_batch = compressed.shape[1]
            for k0, k1 in lib.prange(0, naux_in_batch, blksize):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                j3c = j3c_full[:, :, :dk]
                fill_symmetric(compressed, pair_addresses, nao, k0, k1,
                               out=j3c)
                contract('pqr,npi->rniq', j3c, orbo_b, out=j3c_00[aux0:aux1])
                if j_factor != 0 and batched and it == 0:
                    contract('pqr,pq->r', j3c, dm, out=auxvec[aux0:aux1])
        j3c_full = j3c = buf = compressed = None
        st('A j3c_00')

        work = (ndarray((nw, 2, bchunk, nao), buffer=wbuf) if host else
                cp.empty((nw, 2, bchunk, nao)))
        for c, (i0, i1) in enumerate(lib.prange(0, nob, bchunk)):
            if a_once:
                if hasync:
                    cp.cuda.Stream.null.wait_event(fdone[c])
                if c + 1 < len(chunks):
                    _fetch_a(c + 1)
            tmp = ndarray((nw, 2, i1-i0, nao), buffer=work)
            contract('rs,rniq->sniq', metric_v, j3c_00[:, :, i0:i1], out=tmp)
            tmp /= metric_w[:, None, None, None]
            contract('rs,sniq->rniq', metric_v, tmp, out=j3c_00[:, :, i0:i1])
        dm_3c = j3c_00
        j3c_00 = tmp = work = None

        fdone = None
        dm_oo = contract('rniq,nqj->rnij', dm_3c, orbo,
                         out=ndarray((naux, 2, nob, nocc), buffer=obuf)
                         if host else None)
        if j_factor != 0 and it == 0:
            if batched:
                auxvec = metric_v.dot(auxvec.dot(metric_v) / metric_w)
            else:
                auxvec = cp.einsum('rnii->r', dm_oo)
            auxvec_by_atm = auxvec[aux_idx]
        if not batched:
            metric_v = metric_w = None
        st('B metric')

        # (00|0)(1|0)(0|00)
        j2c_10 = _j2c_10()
        st('C int2c2e_ip1')
        if hasync:
            # item 23: the batch's dm_3c to the host behind stage C, which
            # only reads it (issued after j2c_10 is formed, whose sorting
            # frees a copy of it: a cudaFree waits for every stream): each (r, n) row of it a contiguous run of
            # nob*nao into a row of nocc*nao
            ev = cp.cuda.Event(disable_timing=True)
            ev.record(cp.cuda.Stream.null)
            xs.wait_event(ev)
            runtime.memcpy2DAsync(
                dm3_host.ctypes.data + int(nocc0*nao*8), int(nocc*nao*8),
                int(dm_3c.data.ptr), int(nob*nao*8), int(nob*nao*8),
                int(naux*2), runtime.memcpyDeviceToHost, xs.ptr)
        if _env('FASTDFHESSIP_H1C', 'grouped') != 'atom':
            # item 18 (W32): fastdfhessip's grouped stage C, on the spin index
            H._h1_stage_c(
                j2c_10, dm_3c, dm_oo if do_k else None, aux_idx, aux_slices,
                auxvec if j_factor != 0 else None,
                auxvec_by_atm if j_factor != 0 else None, vhf_i,
                vhf_atm_ao if k_ao else None,
                vhf_atm if do_k and not mem_sufficient else None,
                -k_factor, j_factor, st)
        else:
            naux_in_atm = (aux_slices[:, 1] - aux_slices[:, 0]).max()
            buf = cp.empty((3, naux_in_atm, 2, nob, nao))
            buf1 = cp.empty((naux_in_atm, 2, nob, nao))
            for i, (p0, p1) in enumerate(aux_slices):
                if not do_k:
                    dm_3c_atm = cp.take(dm_3c, aux_idx[p0:p1], axis=0,
                                        out=buf1[:p1-p0])
                    j3c_1 = ndarray((3, p1-p0, 2, nob, nao), buffer=buf)
                    contract('xrs,sniq->xrniq', j2c_10[:, p0:p1], dm_3c,
                             out=j3c_1)
                elif mem_sufficient:
                    dm_3c_atm = cp.take(dm_3c, aux_idx[p0:p1], axis=0,
                                        out=buf1[:p1-p0])
                    j3c_1 = ndarray((3, p1-p0, 2, nob, nao), buffer=buf)
                    contract('xrs,sniq->xrniq', j2c_10[:, p0:p1], dm_3c,
                             out=j3c_1)
                    contract('xrnip,rniq->xnpq', j3c_1, dm_3c_atm, -k_factor,
                             beta=1, out=vhf_atm_ao[i])
                else:
                    dm_3c_atm = cp.take(dm_3c, aux_idx[p0:p1], axis=0,
                                        out=buf1[:p1-p0])
                    dm_oo_atm = dm_oo[aux_idx[p0:p1]]
                    tmp = ndarray((3, p1-p0, 2, nob, nocc), buffer=buf)
                    tmp = contract('xrs,snij->xrnij', j2c_10[:, p0:p1], dm_oo,
                                   out=tmp)
                    contract('rniq,xrnij->xnjq', dm_3c_atm, tmp, -k_factor,
                             beta=1, out=vhf_atm[i])
                    j3c_1 = ndarray((3, p1-p0, 2, nob, nao), buffer=buf)
                    contract('xrs,sniq->xrniq', j2c_10[:, p0:p1], dm_3c,
                             out=j3c_1)
                    contract('xrniq,rnij->xnjq', j3c_1, dm_oo_atm, -k_factor,
                             beta=1, out=vhf_atm[i])
                if j_factor != 0:
                    contract('xrniq,r->xniq', j3c_1, auxvec_by_atm[p0:p1],
                             j_factor, beta=1, out=vhf_i[i])
                    tmp = cp.einsum('xrs,s->xr', j2c_10[:, p0:p1], auxvec)
                    contract('rniq,xr->xniq', dm_3c_atm, tmp, j_factor,
                             beta=1, out=vhf_i[i])
            tmp = j3c_1 = dm_3c_atm = dm_oo_atm = None
            buf = buf1 = j2c_10 = None
        j2c_10 = None
        st('C j2c_10')

        if hasync:
            dm_oo_all[:, :, nocc0:nocc1] = dm_oo
            xs.synchronize()
            st('C dm_3c out')
        elif host:
            # item 23: this batch's dm_3c to the host, each (r, n) row of it
            # a contiguous run of nob*nao into a row of nocc*nao, and its
            # dm_oo into the whole one
            runtime.memcpy2D(dm3_host.ctypes.data + int(nocc0*nao*8),
                             int(nocc*nao*8), int(dm_3c.data.ptr),
                             int(nob*nao*8), int(nob*nao*8), int(naux*2),
                             runtime.memcpyDeviceToHost)
            dm_oo_all[:, :, nocc0:nocc1] = dm_oo
            st('C dm_3c out')
        else:
            _stage_de(nocc0, nocc1, dm_3c, dm_oo, it == 0)
        dm_oo = dm_3c = None
    if host:
        bbuf = obuf = wbuf = None
        stage_bufs = [cp.empty((batch_size, 2, nocc, nao))
                      for _ in range(2 if hasync else 1)]
        ready = [None, None]
        used = [None, None]
        if hasync:
            # the two buffers come from the pool, whose blocks are ordered
            # on the null stream only: the copies on `xs` must not start
            # before the kernels of the block's previous owner are done
            ev = cp.cuda.Event(disable_timing=True)
            ev.record(cp.cuda.Stream.null)
            xs.wait_event(ev)

        def _fetch(k, b):
            r0, r1 = int(aux_offsets[k]), int(aux_offsets[k+1])
            row = int(2 * nocc * nao * 8)
            runtime.memcpyAsync(int(stage_bufs[b].data.ptr),
                                dm3_host.ctypes.data + r0 * row,
                                (r1 - r0) * row, runtime.memcpyHostToDevice,
                                xs.ptr)
            ready[b] = cp.cuda.Event(disable_timing=True)
            ready[b].record(xs)
        _stage_de(0, nocc, None, dm_oo_all, True)
        if hasync:
            xs.synchronize()
        stage_bufs = ready = used = dm_oo_all = dm3_host = None
    eval_j3c = eval_ip1 = eval_ipaux = metric_v = metric_w = None

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
