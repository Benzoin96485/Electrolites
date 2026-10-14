"""DF-MP2 relaxed density and analytic nuclear gradient on the GPU
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` P3).

GPU4PySCF 1.8.1's DF-MP2 (``gpu4pyscf.mp.dfmp2.DFMP2``) has an energy and
nothing else: ``Gradients()`` raises ``NotImplementedError`` -- and PySCF
2.13 has no DF-MP2 gradient at all (every DF class raises), only the relaxed
density in ``mp/dfmp2_native.py``.  This is the closed-shell RI-MP2 gradient
on an RHF reference (exact or density-fitted SCF), with frozen core:

* amplitudes from the fitted (ia|P), blocked over the first occupied index,
  giving the unrelaxed density blocks P_ij, P_ab and the three-index
  two-particle density Gamma^P_ia = sum_jb 2(2t - t~)_iajb c^P_jb (raw
  auxiliary index) -- the amplitudes are never stored whole;
* the orbital Lagrangian from Gamma and raw MO three-index integrals;
* the Z-vector by preconditioned conjugate gradients through the SCF's own
  response (``mf.gen_response``), to |r| < 1e-7 |b| (the DF response
  builds stagnate near 3e-8: their round-off, not the solver's);
* the gradient: GPU4PySCF's hcore/overlap and J/K derivative contractions of
  the SCF gradient object with the relaxed and energy-weighted densities,
  and the non-separable part through GPU4PySCF's three-index derivative
  kernel (``sum_ejk_int3c2e_ip1``, the one its DF J/K gradient uses) fed
  Gamma, plus the metric term.

The derivation is in ``docs/FEATURE_TARGETS_PYSCF_2.13.md`` §14.  Checked
against finite differences of the GPU DF-MP2 energy and against PySCF's
relaxed density (``tests/test_dfmp2_grad.py``).

    import electrolites
    electrolites.enable('dfmp2_grad')
    pt = gpu4pyscf.mp.dfmp2.DFMP2(mf).run()
    g = pt.Gradients().kernel()
    dm = pt.make_rdm1_relaxed()
"""
import ctypes
import time

import numpy as np
import cupy as cp

from pyscf import lib

STATS = {}          # stage -> seconds of the last grad_elec (synchronised)


class _AO3c:
    """GPU4PySCF's three-index machinery on (mol, auxmol) in the layout its DF
    gradient kernel uses: sorted, decontracted, cartesian AOs; the auxiliary
    index sorted and cartesian.  ``to_kernel`` maps a raw auxiliary index in
    ``auxmol``'s own (spherical, PySCF) order onto it."""

    def __init__(self, mol, auxmol):
        from gpu4pyscf.lib.cupy_helper import get_avail_mem
        from gpu4pyscf.df.int3c2e_bdiv import Int3c2eOpt
        from gpu4pyscf.gto.mole import SortedMole
        self.mol = mol
        self.opt = Int3c2eOpt(SortedMole.from_mol(mol, decontract=True),
                              auxmol).build()
        self.smol, self.saux = self.opt.mol, self.opt.auxmol
        nao = self.smol.nao
        self.nao = nao
        self.pair_addresses = self.opt.pair_and_diag_indices(
            cart=True, original_ao_order=False)[0]
        npair = len(self.pair_addresses)
        mem = get_avail_mem(exclude_memory_pool=True) - 2 * nao**2 * 8
        self.mem = mem
        self.batch_size = max(1, min(self.saux.nao, int(mem * .5 / (npair * 8))))
        _, self.aux_sorting, _, _ = self.opt.int3c2e_evaluator(
            aux_batch_size=self.batch_size, reorder_aux=True, cart=True)
        coeff = cp.asarray(self.saux.ctr_coeff)
        self.aux_coeff = cp.empty_like(coeff)
        self.aux_coeff[self.aux_sorting] = coeff

    def to_kernel(self, x):
        """Raw auxiliary index (first axis), spherical -> kernel order."""
        shp = x.shape
        return (self.aux_coeff @ x.reshape(shp[0], -1)).reshape(-1, *shp[1:])

    def ao(self, c):
        return self.smol.apply_C_dot(cp.asarray(c, dtype=np.float64), axis=0)

    def yz(self, C, Ca, Cv, gamk):
        """Y[r, i] = sum_aP (ra|P) Gam[P,i,a] and Z[r, a] = sum_iP (ri|P)
        Gam[P,i,a] from one pass over the AO integrals (gamk in kernel
        auxiliary order)."""
        from gpu4pyscf.lib.cupy_helper import contract
        C, Ca, Cv = self.ao(C), self.ao(Ca), self.ao(Cv)
        nao = self.nao
        ni, nv = Ca.shape[1], Cv.shape[1]
        npair = len(self.pair_addresses)
        i_addr, j_addr = divmod(self.pair_addresses, nao)
        eval_j3c, _, _, aux_offsets = self.opt.int3c2e_evaluator(
            aux_batch_size=self.batch_size, reorder_aux=True, cart=True)
        per = nao * (nao + nv + ni) * 8 * 2
        blk = max(1, min(self.saux.nao, int(self.mem * .3 / per)))
        yao = cp.zeros((nao, ni))
        zao = cp.zeros((nao, nv))
        buf = cp.empty((self.batch_size, npair))
        j3c_full = cp.zeros((nao, nao, blk))
        aux1 = 0
        for kb in range(len(aux_offsets) - 1):
            comp = eval_j3c(aux_batch_id=kb, out=buf)
            nk = comp.shape[1]
            for k0, k1 in lib.prange(0, nk, blk):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                j3c = j3c_full[:, :, :dk]
                j3c[j_addr, i_addr] = j3c[i_addr, j_addr] = comp[:, k0:k1]
                g = gamk[aux0:aux1]                              # (P, i, a)
                x = contract('mnP,na->maP', j3c, Cv)
                yao += contract('maP,Pia->mi', x, g)
                x = contract('mnP,ni->miP', j3c, Ca)
                zao += contract('miP,Pia->ma', x, g)
                x = None
        return C.T @ yao, C.T @ zao

    def grad(self, cl, cr, gamk, gtk):
        """Per-atom derivative of
            E = 2 sum_{ijP} (ij|P) Gam[P, i, j] - sum_PQ (P|Q) Gt[P, Q]
        with (ij|P) = sum_mn cl[m, i] cr[n, j] (mn|P), the coefficient sets
        held fixed (gamk, gtk in kernel auxiliary order).  The 3c part is
        GPU4PySCF's ``sum_ejk_int3c2e_ip1`` -- the kernel of its DF J/K
        gradient -- fed a general pair density, the 2c part its
        ``int2c2e_ip1_per_atom``.  (natm, 3) on the host."""
        from gpu4pyscf.lib.cupy_helper import contract, ndarray
        from gpu4pyscf.df.int3c2e_bdiv import (
            _split_l_ctr_pattern, get_ao_pair_loc, int3c2e_scheme,
            libvhf_rys)
        from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1_per_atom
        mol, opt, smol, saux = self.mol, self.opt, self.smol, self.saux
        cl, cr = self.ao(cl), self.ao(cr)
        nao = self.nao
        ni, nj = gamk.shape[1:]
        # contract the larger of the two MO indices first:
        # nao ni nj + nao^2 min(ni, nj) instead of + nao^2 max(ni, nj)
        j_first = nj >= ni
        pair_addresses = self.pair_addresses
        nao_pair = len(pair_addresses)
        batch_size = self.batch_size
        blksize = max(1, min(gamk.shape[0],
                             int(self.mem * .4 / (nao * nao * 2 * 8))))
        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
            short_range=mol.omega < 0, gout_width=54, deriv=(1, 0, 0))
        gout_stride = cp.asarray(gout_stride, dtype=np.int32)
        lmax = smol.uniq_l_ctr[:, 0].max()
        laux = saux.uniq_l_ctr[:, 0].max()
        shm_size_max = shm_size[:laux + 1, :lmax + 1, :lmax + 1].max()
        bas_ij_idx, shl_pair_offsets = smol.aggregate_shl_pairs(
            opt.bas_ij_cache, nsp_per_block[0] * 4)
        ao_pair_loc = get_ao_pair_loc(smol.uniq_l_ctr[:, 0], opt.bas_ij_cache)
        aux_loc = saux.ao_loc
        l_ctr_aux_offsets = np.append(0, np.cumsum(saux.l_ctr_counts))
        l_ctr_aux_offsets, _ = _split_l_ctr_pattern(
            l_ctr_aux_offsets, saux.uniq_l_ctr, batch_size)
        ksh_offsets_cpu = l_ctr_aux_offsets
        ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu + smol.nbas,
                                     dtype=np.int32)
        aux_offsets = aux_loc[ksh_offsets_cpu]

        kern = libvhf_rys.sum_ejk_int3c2e_ip1
        envs = opt.int3c2e_envs
        buf = cp.empty(nao_pair * batch_size)
        buf1 = cp.empty((blksize, nao, nao))
        buf2 = cp.empty((blksize, nao, nao))
        ejk = cp.zeros((mol.natm, 3))
        ejk_aux = cp.zeros((mol.natm, 3))
        aux1 = 0
        for kbatch in range(len(aux_offsets) - 1):
            nk = int(aux_offsets[kbatch + 1] - aux_offsets[kbatch])
            aux_ao_offset = int(aux_loc[ksh_offsets_cpu[kbatch]])
            compressed = ndarray((nao_pair, nk), buffer=buf)
            for k0, k1 in lib.prange(0, nk, blksize):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                d = ndarray((nao, nao, dk), buffer=buf1)
                if j_first:
                    t = ndarray((ni, nao, dk), buffer=buf2)
                    contract('rij,qj->iqr', gamk[aux0:aux1], cr, out=t)
                    contract('iqr,pi->pqr', t, cl, out=d)
                else:
                    t = ndarray((nao, nj, dk), buffer=buf2)
                    contract('rij,pi->pjr', gamk[aux0:aux1], cl, out=t)
                    contract('pjr,qj->pqr', t, cr, out=d)
                d1 = ndarray((nao, nao, dk), buffer=buf2)
                d1[:] = d.transpose(1, 0, 2)
                d1 += d
                cp.take(d1.reshape(-1, dk), pair_addresses, axis=0,
                        out=compressed[:, k0:k1])
            err = kern(
                ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
                ctypes.cast(ejk_aux.data.ptr, ctypes.c_void_p),
                ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
                lib.c_null_ptr(),
                ctypes.c_int(1),
                ctypes.byref(envs),
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
                ctypes.c_int(nao), ctypes.c_int(nao_pair),
                ctypes.c_int(nk), ctypes.c_int(mol.natm))
            if err != 0:
                raise RuntimeError('sum_ejk_int3c2e_ip1 failed')
        buf = buf1 = buf2 = compressed = None
        gtk = gtk[self.aux_sorting[:, None], self.aux_sorting]
        e2c = cp.asarray(int2c2e_ip1_per_atom(saux, gtk + gtk.T))
        # the kernel's normalisation, measured element-wise against PySCF's
        # int3c2e_ip1/ip2 and int2c2e_ip1 (tests/test_dfmp2_grad.py): it
        # returns half of the 3c term as written and the 2c term with the
        # opposite sign
        return (2 * (ejk + ejk_aux) - e2c).get()


def nonsep_grad(mol, auxmol, cl, cr, gam, gt):
    """``_AO3c.grad`` with the raw auxiliary index of ``gam``/``gt`` in
    ``auxmol``'s own (spherical, PySCF) order."""
    k = _AO3c(mol, auxmol)
    gam = k.to_kernel(cp.asarray(gam, dtype=np.float64))
    gt = k.to_kernel(k.to_kernel(cp.asarray(gt, dtype=np.float64)).T).T
    return k.grad(cl, cr, gam, gt)


# ------------------------------------------------------------ the driver
def _free():
    return cp.cuda.runtime.memGetInfo()[0] + \
        cp.get_default_memory_pool().free_bytes()


class _RI:
    """The RI integrals of one DF-MP2: the bdiv integral optimiser, the
    Cholesky factor of the metric, raw MO three-index blocks."""

    def __init__(self, mol, auxmol):
        from gpu4pyscf.df.int3c2e_bdiv import Int3c2eOpt
        from gpu4pyscf.mp import dfmp2_addons
        self.mol, self.auxmol = mol, auxmol
        self.opt = Int3c2eOpt(mol, auxmol).build()
        self.j2c = cp.asarray(dfmp2_addons.get_j2c_bdiv(self.opt))
        self.low = cp.linalg.cholesky(self.j2c)

    def raw(self, c1, c2):
        """(i j|P), raw auxiliary index, (n1, n2, naux)."""
        from gpu4pyscf.mp import dfmp2_addons
        c1 = cp.asarray(c1, dtype=np.float64)
        c2 = cp.asarray(c2, dtype=np.float64)
        buf = cp.empty((c1.shape[1], c2.shape[1], self.auxmol.nao_cart()))
        return dfmp2_addons.get_j3c_ovl_gpu_bdiv(self, self.opt, [c1], [c2],
                                                 [buf])[0]

    verbose = 0
    stdout = None

    def fit(self, x):
        """B = L^-1 x and c = V^-1 x for x[..., P]."""
        from cupyx.scipy.linalg import solve_triangular
        shp = x.shape
        b = solve_triangular(self.low, x.reshape(-1, shp[-1]).T, lower=True)
        c = solve_triangular(self.low.T, b, lower=False)
        return (cp.ascontiguousarray(b.T).reshape(shp),
                cp.ascontiguousarray(c.T).reshape(shp))


_DIV_DEN = cp.ElementwiseKernel(
    'raw float64 k, raw float64 eo_blk, raw float64 eo, raw float64 ev, '
    'int64 nv, int64 no', 'float64 t',
    """
    long long b = i % nv;
    long long j = (i / nv) % no;
    long long a = (i / (nv * no)) % nv;
    long long ii = i / (nv * no * nv);
    t = k[i] / (eo_blk[ii] + eo[j] - ev[a] - ev[b]);
    """, 'electrolites_dfmp2_div_den')

# tt[i,a,j,b] = c1 t[i,a,j,b] - c2 t[i,b,j,a]: a batched (i, j) nv x nv
# transpose through 32x32 shared-memory tiles (both reads coalesced)
_TT = cp.RawKernel(r"""
extern "C" __global__
void tt_kernel(const double* __restrict__ t, double* __restrict__ tt,
               int nv, int no, double c1, double c2)
{
    __shared__ double tile[32][33];
    int ij = blockIdx.z;                 // i * no + j
    int i = ij / no, j = ij % no;
    long long base = ((long long)i * nv * no + j) * nv;   // t[i, 0, j, 0]
    long long rs = (long long)no * nv;                     // stride of a
    int a0 = blockIdx.y * 32, b0 = blockIdx.x * 32;
    int tx = threadIdx.x, ty = threadIdx.y;                // 32 x 8
    for (int k = ty; k < 32; k += 8) {                     // t[i, b0+k, j, a0+tx]
        int b = b0 + k, a = a0 + tx;
        if (b < nv && a < nv) tile[k][tx] = t[base + b * rs + a];
    }
    __syncthreads();
    for (int k = ty; k < 32; k += 8) {                     // out[i, a0+k, j, b0+tx]
        int a = a0 + k, b = b0 + tx;
        if (a < nv && b < nv) {
            long long o = base + a * rs + b;
            tt[o] = c1 * t[o] - c2 * tile[tx][k];
        }
    }
}
""", 'tt_kernel')


def _make_tt(t, c1, c2):
    ni, nv, no, _ = t.shape
    tt = cp.empty_like(t)
    g = ((nv + 31) // 32, (nv + 31) // 32, ni * no)
    _TT(g, (32, 8, 1), (t, tt, np.int32(nv), np.int32(no), np.float64(c1),
                        np.float64(c2)))
    return tt


def _amplitude_terms(B, c, eo, ev, cos=1., css=1., low=None):
    """E2, P_oo, P_vv and Gamma[i, a, P] = sum_jb D2[i,a,j,b] c[j,b,P],
    blocked over the first occupied index.  With spin-component scaling
    E2 = cos E_os + css E_ss = sum t~ K, t~ = (cos + css) t - css t^T
    (PySCF dfmp2_native's ps, pt), P_ab = 2 sum t t~, P_ij = -2 sum t t~ and
    D2 = 2 t~; cos = css = 1 is MP2.  The density blocks are batched GEMMs
    on the amplitude block as it lies (no transposed copies)."""
    no, nv, naux = B.shape
    per_i = no * nv * nv * 8 * 3 + nv * no * no * 8
    blk = max(1, min(no, int(0.5 * _free() // per_i)))
    B2 = B.reshape(no * nv, naux)
    c2 = c.reshape(no * nv, naux)
    poo = cp.zeros((no, no))
    pvv = cp.zeros((nv, nv))
    gam = cp.empty((no, nv, naux))
    for i0 in range(0, no, blk):
        i1 = min(no, i0 + blk)
        ni = i1 - i0
        k = (B[i0:i1].reshape(ni * nv, naux) @ B2.T).reshape(ni, nv, no, nv)
        t = cp.empty_like(k)
        _DIV_DEN(k, cp.ascontiguousarray(eo[i0:i1]), eo, ev, nv, no, t)
        k = None
        tt = _make_tt(t, cos + css, css)
        # P_ab += 2 sum_{ijc} t[i,a,j,c] tt[i,b,j,c]
        # (one 2-D GEMM per i: CuPy's batched matmul with a transposed
        # operand copies it, ccsd_t.py)
        t3 = t.reshape(ni, nv, no * nv)
        tt3 = tt.reshape(ni, nv, no * nv)
        for ii in range(ni):
            pvv += t3[ii] @ tt3[ii].T

        # P_ij += -2 sum_{kba} t[k,b,i,a] tt[k,b,j,a]  (k in the block)
        t3 = t.reshape(ni * nv, no, nv)
        tt3 = tt.reshape(ni * nv, no, nv)
        poo -= 2 * cp.matmul(t3, tt3.transpose(0, 2, 1)).sum(axis=0)
        t = t3 = tt3 = None
        g = tt.reshape(ni * nv, no * nv) @ c2
        g *= 2
        gam[i0:i1] = g.reshape(ni, nv, naux)
        tt = g = None
    pvv *= 2
    # E2 = sum t~ K = 1/2 sum_{iaP} Gamma^P_ia (ia|P), (ia|P) = L B
    raw = B2 @ low.T
    e2 = .5 * float(cp.vdot(gam.reshape(-1, naux), raw))
    return e2, poo, pvv, gam


def _pcg(aop, b, prec, tol=1e-7, max_cycle=60, log=None):
    """Preconditioned conjugate gradients for the symmetric positive definite
    Z-vector equation, to |r| < tol |b|."""
    x = b * prec
    r = b - aop(x)
    z = r * prec
    p = z
    rz = float(cp.vdot(r, z))
    nb = float(cp.linalg.norm(b))
    for it in range(max_cycle):
        nr = float(cp.linalg.norm(r))
        if log is not None:
            log.debug('Z-vector PCG %d  |r|/|b| = %.3e', it, nr / nb)
        if nr < tol * nb:
            return x, it
        ap = aop(p)
        alpha = rz / float(cp.vdot(p, ap))
        x = x + alpha * p
        r = r - alpha * ap
        z = r * prec
        rz, rz0 = float(cp.vdot(r, z)), rz
        p = z + (rz / rz0) * p
    if nr < 1e-6 * nb:
        # stagnation at the response build's round-off level
        if log is not None:
            log.warn('Z-vector PCG stopped at |r|/|b| = %.1e after %d '
                     'iterations', nr / nb, max_cycle)
        return x, max_cycle
    raise RuntimeError(f'Z-vector PCG not converged in {max_cycle} '
                       f'iterations (|r|/|b| = {nr / nb:.2e})')


_T = [0.]


def _stage(log, label, t0):
    cp.cuda.Stream.null.synchronize()
    now = time.perf_counter()
    STATS[label] = STATS.get(label, 0.) + now - _T[0]
    _T[0] = now
    return log.timer('DF-MP2 grad: ' + label, *t0)


def _frozen_occ(mp, nocc, nmo):
    """Boolean mask of the frozen occupied orbitals (frozen virtuals are not
    supported)."""
    mask = ~cp.asnumpy(mp.get_frozen_mask())
    if mask[nocc:].any():
        raise NotImplementedError('DF-MP2 gradient: frozen virtual orbitals')
    return mask[:nocc]


def xc_cross_grad(mf, dm0, p):
    """Per-atom d/dX of sum_mn p_mn Vxc[dm0]_mn at fixed dm0 and p (the XC
    part of p . F^(x)): the AO derivatives of Vxc's basis functions
    contracted with p, plus those of rho_dm0 under fxc . rho_p contracted
    with dm0 -- GPU4PySCF's ``grad.rks`` XC-gradient kernels on both
    potentials.  No grid response (as GPU4PySCF's default KS gradient).
    LDA and GGA."""
    from gpu4pyscf.dft import numint
    from gpu4pyscf.grad.rks import _gga_grad_sum_, _d1_dot_, _reduce_to_atom
    from gpu4pyscf.lib.cupy_helper import take_last2d, contract
    ni = mf._numint
    mol = mf.mol
    grids = mf.grids
    if grids.coords is None:
        grids.build(sort_grids=True)
    xctype = ni._xc_type(mf.xc)
    if xctype not in ('LDA', 'GGA'):
        raise NotImplementedError(f'DF-MP2 gradient: {xctype} functionals')
    if mf.do_nlc():
        raise NotImplementedError('DF-MP2 gradient: NLC functionals')
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    smol = opt._sorted_mol
    nao = mol.nao
    dm0 = opt.sort_orbitals(cp.asarray(dm0), axis=[0, 1])
    p = opt.sort_orbitals(cp.asarray(p), axis=[0, 1])
    exc1 = cp.zeros((nao, 3))
    ao_deriv = 1 if xctype == 'LDA' else 2
    for ao, idx, weight, _ in ni.block_loop(smol, grids, nao, ao_deriv):
        d0 = take_last2d(dm0, idx)
        pm = take_last2d(p, idx)
        if xctype == 'LDA':
            rho = numint.eval_rho(smol, ao[0], d0, xctype=xctype, hermi=1)
            rhop = numint.eval_rho(smol, ao[0], pm, xctype=xctype, hermi=1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype,
                                      spin=0)[1:3]
            wv = weight * vxc[0]
            v = _d1_dot_(ao[1:4], (ao[0] * wv).T)
            exc1[idx] += contract('nij,ij->in', v, pm)
            wv = weight * fxc[0, 0] * rhop
            v = _d1_dot_(ao[1:4], (ao[0] * wv).T)
            exc1[idx] += contract('nij,ij->in', v, d0)
        else:
            rho = numint.eval_rho(smol, ao[:4], d0, xctype=xctype, hermi=1)
            rhop = numint.eval_rho(smol, ao[:4], pm, xctype=xctype, hermi=1)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype,
                                      spin=0)[1:3]
            wv = vxc * weight
            wv[0] *= .5
            v = _gga_grad_sum_(ao, cp.ascontiguousarray(wv))
            exc1[idx] += contract('nij,ij->in', v, pm)
            wv = contract('yg,xyg->xg', rhop, fxc) * weight
            wv[0] *= .5
            v = _gga_grad_sum_(ao, cp.ascontiguousarray(wv))
            exc1[idx] += contract('nij,ij->in', v, d0)
    return -2 * _reduce_to_atom(smol, exc1)


def _hybrid(mf):
    """(is_ks, hyb); range-separated hybrids are not supported."""
    if not mf.istype('KohnShamDFT'):
        return False, 1.
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=0)
    if omega != 0:
        raise NotImplementedError('DF-MP2 gradient: range-separated hybrids')
    return True, hyb


def _jk_cross(mf, g, D, P, hyb, log, P_mo=None):
    """Per-atom skeleton derivative of P . (J[D] - hyb/2 K[D])."""
    from gpu4pyscf.df.df_jk import _DFHF
    if not isinstance(mf, _DFHF):
        # exact integrals: polarisation, E_G(D+P) - E_G(D) - E_G(P)
        f = dict(j_factor=1, k_factor=hyb, hermi=1)
        return (cp.asnumpy(g.jk_energy_per_atom(D + P, **f))
                - cp.asnumpy(g.jk_energy_per_atom(D, **f))
                - cp.asnumpy(g.jk_energy_per_atom(P, **f)))
    # DF: J by polarisation (J costs the same for any density); K through
    # the general three-index kernel with D's occupied factor, so that the
    # full-rank P is never factorised
    def sub(label, t):
        cp.cuda.Stream.null.synchronize()
        now = time.perf_counter()
        STATS[label] = STATS.get(label, 0.) + now - t
        return now
    t = time.perf_counter()
    f = dict(j_factor=1, k_factor=0, hermi=1)
    ej = .5 * (cp.asnumpy(g.jk_energy_per_atom(D + P, **f))
               - cp.asnumpy(g.jk_energy_per_atom(D - P, **f)))
    t = sub('..J cross', t)
    if hyb == 0:
        return ej
    mol = mf.mol
    C = cp.asarray(mf.mo_coeff)
    occ = cp.asarray(mf.mo_occ) > 0
    Co = C[:, occ]
    ri = _RI(mol, mf.with_df.auxmol)
    _, c = ri.fit(ri.raw(C, Co))                          # (p, i, P), MO
    t = sub('..K cross: (pi|P) and fit', t)
    nmo, no, naux = c.shape
    # E_Kc = P . K[D] = 2 sum_i sum_pq P_pq (pi|qi) in the MO basis
    if P_mo is None:
        sc = cp.asarray(mf.get_ovlp()) @ C
        P_mo = sc.T @ P @ sc
    pmo = P_mo
    gam = 2 * (pmo @ c.reshape(nmo, -1)).reshape(nmo, no, naux)
    gt = c.reshape(-1, naux).T @ gam.reshape(-1, naux)
    c = None
    k = _AO3c(mol, ri.auxmol)
    gamk = k.to_kernel(cp.ascontiguousarray(gam.transpose(2, 0, 1)))
    gam = None
    gtk = k.to_kernel(k.to_kernel(gt).T).T
    t = sub('..K cross: Gamma', t)
    ek = k.grad(C, Co, gamk, gtk)
    sub('..K cross: derivative kernel', t)
    return ej - .5 * hyb * ek


def grad_elec(mp, z_tol=1e-7, verbose=None, with_grad=True):
    """(relaxed MP2 difference density in the MO basis, electronic gradient
    without the nuclear repulsion, E2) for a GPU4PySCF DF-MP2.

    The reference is an RHF or a (global-hybrid or pure) RKS -- exact or
    density-fitted; on an RKS the MP2 is the double-hybrid correlation, and
    the response is the KS one (``mf.gen_response`` with fxc).  Spin-
    component scaling through ``mp.ps`` / ``mp.pt`` (os / ss, PySCF
    dfmp2_native's names; default 1).

    Frozen occupied orbitals f: the frozen-active rotations are fixed by
    keeping the orbitals canonical, U_fi = -(dF_fi)/(e_f - e_i), which adds
    z_fi = 2 Y_fi / (e_f - e_i) to the relaxed density and its Fock response
    to the Z-vector right-hand side (PySCF's ``dfmp2_native``)."""
    from gpu4pyscf.lib import logger
    mf = mp._scf
    mol = mp.mol
    log = logger.new_logger(mp, verbose)
    STATS.clear()
    cp.cuda.Stream.null.synchronize()
    _T[0] = time.perf_counter()
    t0 = log.init_timer()
    if not mf.istype('RHF'):
        raise NotImplementedError(
            f'DF-MP2 gradient: restricted references only (got '
            f'{type(mf).__name__})')
    is_ks, hyb = _hybrid(mf)
    cos = getattr(mp, 'ps', 1.)
    css = getattr(mp, 'pt', 1.)
    C = cp.asarray(mf.mo_coeff)
    e = cp.asarray(mf.mo_energy)
    occ = cp.asnumpy(mf.mo_occ)
    nocc = int(np.count_nonzero(occ > 0))
    nmo = C.shape[1]
    frz = _frozen_occ(mp, nocc, nmo)
    act = np.where(~frz)[0]
    fro = np.where(frz)[0]
    Co, Cv = C[:, :nocc], C[:, nocc:]
    Ca = C[:, act]
    eo, ev = e[:nocc], e[nocc:]
    ea = e[act]

    ri = _RI(mol, mp.auxmol)
    B, c = ri.fit(ri.raw(Ca, Cv))
    e2, paa, pvv, gam = _amplitude_terms(B, c, ea, ev, cos, css, ri.low)
    B = None
    naux = gam.shape[2]
    gt = c.reshape(-1, naux).T @ gam.reshape(-1, naux)
    c = None
    t0 = _stage(log, 'amplitudes, P, Gamma', t0)

    k3 = _AO3c(mol, ri.auxmol)
    gamk = k3.to_kernel(cp.ascontiguousarray(gam.transpose(2, 0, 1)))
    gam = None
    gtk = k3.to_kernel(k3.to_kernel(gt).T).T
    gt = None
    Y, Z = k3.yz(C, Ca, Cv, gamk)          # Y[r, i active], Z[r, a]
    t0 = _stage(log, 'Y, Z', t0)

    vresp = mf.gen_response(C, mf.mo_occ, hermi=1)

    def gop(dm):
        return C.T @ vresp(dm) @ C

    P = cp.zeros((nmo, nmo))
    P[act[:, None], act] = paa
    P[nocc:, nocc:] = pvv
    Zf = cp.zeros((nmo, nmo))
    if len(fro):
        Zf[fro[:, None], act] = 2 * Y[fro] / (e[fro][:, None] - ea[None, :])
    Zfs = .5 * (Zf + Zf.T)
    Gp = gop(C @ (P - Zfs) @ C.T)
    L = 4 * Gp[nocc:, :nocc] - 2 * Z[:nocc].T
    L[:, act] += 2 * Y[nocc:]
    dia = ev[:, None] - eo[None, :]

    def aop(x):
        dm = Cv @ x @ Co.T
        return dia * x + 2 * (Cv.T @ vresp(dm + dm.T) @ Co)
    z, nit = _pcg(aop, L, 1 / dia, tol=z_tol, log=log)
    log.debug('Z-vector converged in %d iterations', nit)
    STATS['z iterations'] = nit
    t0 = _stage(log, 'Z-vector', t0)

    Zd = cp.zeros((nmo, nmo))
    Zd[nocc:, :nocc] = z
    Zs = .5 * (Zd + Zd.T)
    prel_mo = P - Zs - Zfs
    if not with_grad:
        return prel_mo, None, e2
    X = cp.zeros((nmo, nmo))
    X[act[:, None], act] = -paa * ea[:, None]
    X[:nocc, :nocc] -= 2 * gop(C @ P @ C.T)[:nocc, :nocc]
    X[act[:, None], act] -= Y[act]
    X[nocc:, nocc:] = -pvv * ev[:, None] - Z[nocc:]
    X[:nocc, nocc:] -= 2 * Z[:nocc]
    X[nocc:, :nocc] += z * eo[None, :]
    if len(fro):
        X[fro[:, None], act] += Zf[fro[:, None], act] * ea[None, :]
    X[:nocc, :nocc] += 2 * gop(C @ (Zs + Zfs) @ C.T)[:nocc, :nocc]
    W = C @ X @ C.T
    W = .5 * (W + W.T)
    t0 = _stage(log, 'Lagrangian', t0)

    g = mf.nuc_grad_method()
    D = cp.asarray(mf.make_rdm1())
    prel = C @ prel_mo @ C.T
    de = cp.asnumpy(g.grad_elec())                 # the SCF's own, at D
    de += cp.asnumpy(g._hcore_energy(prel, -W))
    t0 = _stage(log, 'SCF gradient + h1/S terms', t0)
    de += _jk_cross(mf, g, D, prel, hyb, log, prel_mo)
    t0 = _stage(log, 'J/K cross terms', t0)
    if is_ks:
        de += xc_cross_grad(mf, D, prel)
        t0 = _stage(log, 'XC cross terms', t0)
    de += k3.grad(Ca, Cv, gamk, gtk)
    _stage(log, 'three-index derivatives', t0)
    return prel_mo, de, e2


class Gradients(lib.StreamObject):
    """``DFMP2.Gradients()``: the DF-MP2 nuclear gradient (Hartree/Bohr)."""

    def __init__(self, mp):
        self.base = mp
        self.mol = mp.mol
        self.verbose = mp.verbose
        self.stdout = mp.stdout
        self.de = None
        self.z_tol = 1e-7

    def grad_elec(self):
        mp = self.base
        if mp.e_corr is None:
            mp.kernel()
        return grad_elec(mp, self.z_tol, self.verbose)[1]

    def grad_nuc(self):
        g = self.base._scf.nuc_grad_method()
        de = g.grad_nuc()
        if self.base._scf.do_disp():
            de = de + g.get_dispersion()
        return de

    def kernel(self):
        self.de = self.grad_elec() + self.grad_nuc()
        return self.de

    grad = kernel

    @property
    def e_tot(self):
        return self.base.e_tot


def make_rdm1_relaxed(mp, ao_repr=False):
    """The relaxed DF-MP2 one-particle density (spin-traced, the HF part
    included), PySCF ``dfmp2_native``'s ``make_rdm1(relaxed=True)``; on the
    device."""
    nocc = int(np.count_nonzero(cp.asnumpy(mp._scf.mo_occ) > 0))
    dm = grad_elec(mp, with_grad=False)[0]
    dm[cp.arange(nocc), cp.arange(nocc)] += 2
    if ao_repr:
        C = cp.asarray(mp._scf.mo_coeff)
        dm = C @ dm @ C.T
    return dm


def _gradients(mp):
    return Gradients(mp)


from gpu4pyscf.mp import dfmp2 as _dfmp2                        # noqa: E402
from electrolites.features import attach                       # noqa: E402
from gpu4pyscf.mp.mp2 import MP2Base as _MP2Base                # noqa: E402
# DFMP2.Gradients is MP2Base's stub, which raises NotImplementedError; if
# upstream ever gives DFMP2 its own, do not shadow it
if (_dfmp2.DFMP2.Gradients is not _MP2Base.Gradients
        and not str(getattr(_dfmp2.DFMP2.Gradients, '__module__', '')
                    ).startswith('electrolites')):
    raise RuntimeError('gpu4pyscf DFMP2.Gradients now exists upstream; '
                       'compare it with electrolites.dfmp2_grad first')
attach(_dfmp2.DFMP2, 'Gradients', _gradients, replace_broken=True)
attach(_dfmp2.DFMP2, 'make_rdm1_relaxed', make_rdm1_relaxed)
