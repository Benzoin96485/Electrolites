"""CASCI / CASSCF with the integrals, J/K and CI on the GPU
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` M1-M5 and M2).

GPU4PySCF 1.8.1 has no multireference methods: ``mol.CASSCF`` raises, a GPU
mean field cannot be handed to ``pyscf.mcscf`` (CuPy implicit conversion),
and the only working route is ``mf.to_cpu()`` -- a host DF-CASSCF that spends
85-93 % of its time, at CAS <= (10,9), in DF J/K builds and in the per-macro-
iteration ``_ERIS`` rebuild, which writes naux*nmo^2 to an HDF5 file and
reads it back, and that gets *slower* with more CPU threads (§11.3).

``CASSCF(mf, ncas, nelecas)`` for a DF mean field (or ``density_fit=True``)
returns PySCF's own DF-CASSCF object --
its macro/micro iteration, augmented-Hessian solver, convergence logic and
results unchanged -- with four things moved to the device:

* ``with_df`` is a GPU proxy (``GPUDF``): three-centre integrals in the MO
  basis from GPU4PySCF's block-divergent int3c2e kernel
  (``get_j3c_ovl_gpu_bdiv``), fitted with the Cholesky factor of the
  two-centre metric, both in fp64; its ``get_jk`` is GPU4PySCF's DF J/K on
  the same auxiliary basis (Electrolites' ``fastdf`` under ``patch_all()``);
* ``ao2mo`` builds the CASSCF ``_ERIS`` (ppaa, papa, j_pc, k_pc, vhf_c) on the
  device from L[p, q, P] in blocks of p, with no disk and no nmo^2*naux
  tensor held whole;
* ``get_jk`` -- every J/K the orbital step asks for (the core density in
  ``h1e_for_cas``, the two densities of ``update_jk_in_ah`` per AH step) --
  goes to the GPU DF J/K;
* the FCI solver is ``fci_gpu.FCISolver`` (sigma build and RDMs on the
  device), with the caller's settings (``nroots``, ``spin``, tolerances)
  carried over; state averaging wraps it as PySCF does.

Without density fitting (a non-DF mean field, or ``density_fit=False``) it is
PySCF's exact-integral CASSCF (M2): ``_ERIS``' ppaa and papa from one batch
of direct J and K builds over the active-pair densities, and every other J/K
from GPU4PySCF's direct J/K.

The auxiliary basis is the same on both paths and is the one PySCF's
DF-CASSCF would use (``df.make_auxbasis`` unless given), so results compare
with PySCF's DF-CASSCF to its own convergence.

    import electrolites
    electrolites.patch_all()
    from electrolites.casscf_gpu import CASSCF
    mc = CASSCF(gpu_mf, 10, 9, density_fit=True).run()      # DF-CASSCF
    mc = CASSCF(gpu_mf, 10, 9).run()      # DF iff gpu_mf is DF
    mc = CASSCF(gpu_mf, 10, 9, density_fit=True).state_average_([.2] * 5).run()
"""
import time

import numpy as np
import cupy as cp

from pyscf import lib, mcscf, df as df_cpu
from pyscf.lib import logger

from electrolites import fci_gpu

STATS = {'eris': 0, 'eris_s': 0., 'jk': 0, 'jk_s': 0., 'l3c_s': 0.}


def _stack(dm):
    """A density or a list/tuple of them (host or device) as one array of
    the array module of its first element."""
    if isinstance(dm, (list, tuple)):
        if any(isinstance(d, cp.ndarray) for d in dm):
            return cp.stack([cp.asarray(d) for d in dm])
        return np.asarray(dm)
    return dm


class GPUDF:
    """The parts of ``pyscf.df.DF`` a DF-CASSCF uses, on the GPU."""

    def __init__(self, mol, auxbasis=None, verbose=None):
        self.mol = mol
        if auxbasis is None:
            auxbasis = df_cpu.make_auxbasis(mol)
        self.auxbasis = auxbasis
        self.auxmol = df_cpu.addons.make_auxmol(mol, auxbasis)
        self.verbose = mol.verbose if verbose is None else verbose
        self.stdout = mol.stdout
        self.blockdim = 240
        self._intopt = self._j2c = self._jkmf = None
        self._lao = None
        self._lt = None
        self._lao_refused = False

    # ---------------------------------------------------------- integrals
    def _build(self):
        if self._intopt is None:
            from gpu4pyscf.df.int3c2e_bdiv import Int3c2eOpt
            from gpu4pyscf.mp import dfmp2_addons
            self._intopt = Int3c2eOpt(self.mol, self.auxmol).build()
            j2c = dfmp2_addons.get_j2c_bdiv(self._intopt)
            self._j2c = dfmp2_addons.get_j2c_decomp_gpu(self.auxmol, j2c=j2c,
                                                         alg='cd')
        return self._intopt, self._j2c

    def get_naoaux(self):
        return self.auxmol.nao

    def _ao_cderi(self):
        """The fitted AO three-index tensor L[mu, nu, P], computed once and
        kept on the device when it fits in 30 % of the free memory (the
        geometry does not change during a CASSCF; the MO transformation is
        then two GEMMs per macro-iteration instead of new integrals)."""
        if self._lao is None and not self._lao_refused:
            nao = self.mol.nao
            need = nao * nao * self.auxmol.nao_cart() * 8 * 2.5
            free = cp.cuda.runtime.memGetInfo()[0] + \
                cp.get_default_memory_pool().free_bytes()
            if need < 0.3 * free:
                eye = cp.eye(nao)
                self._lao = self._lmo_int(eye, eye)
            else:
                self._lao_refused = True
        return self._lao

    def lmo(self, c1, c2):
        """L[i, j, P] = sum_Q (ij|Q) (V^-1/2)[Q, P] for MO sets c1, c2."""
        lao = self._ao_cderi()
        if lao is None:
            return self._lmo_int(c1, c2)
        t0 = time.perf_counter()
        c1 = cp.asarray(c1, dtype=np.float64)
        c2 = cp.asarray(c2, dtype=np.float64)
        nao, naux = lao.shape[0], lao.shape[2]
        t = (c1.T @ lao.reshape(nao, nao * naux)).reshape(-1, nao, naux)
        out = cp.matmul(t.transpose(0, 2, 1), c2)            # (n1, P, n2)
        out = cp.ascontiguousarray(out.transpose(0, 2, 1))
        STATS['l3c_s'] += time.perf_counter() - t0
        return out

    def _lmo_int(self, c1, c2):
        """The same from new integrals (GPU4PySCF's bdiv int3c2e)."""
        from gpu4pyscf.mp import dfmp2_addons
        t0 = time.perf_counter()
        intopt, j2c = self._build()
        c1 = cp.asarray(c1, dtype=np.float64)
        c2 = cp.asarray(c2, dtype=np.float64)
        buf = cp.empty((c1.shape[1], c2.shape[1], self.auxmol.nao_cart()))
        out = dfmp2_addons.get_j3c_ovl_gpu_bdiv(self, intopt, [c1], [c2],
                                                [buf])[0]
        out = dfmp2_addons.decompose_j3c_gpu(self, j2c, [out])[0]
        STATS['l3c_s'] += time.perf_counter() - t0
        return out

    def ao2mo(self, mo_coeffs, compact=True):
        """(ij|kl) in the MO basis, as ``pyscf.df.DF.ao2mo``."""
        if isinstance(mo_coeffs, np.ndarray) and mo_coeffs.ndim == 2:
            mo_coeffs = (mo_coeffs,) * 4
        c1, c2, c3, c4 = mo_coeffs
        l12 = self.lmo(c1, c2)
        l34 = l12 if (c3 is c1 and c4 is c2) else self.lmo(c3, c4)
        n1, n2 = l12.shape[:2]
        n3, n4 = l34.shape[:2]
        eri = l12.reshape(n1 * n2, -1) @ l34.reshape(n3 * n4, -1).T
        eri = cp.asnumpy(eri)
        if compact and n1 == n2 == n3 == n4 and c1 is c2 is c3 is c4:
            from pyscf import ao2mo
            eri = ao2mo.restore(4, eri.reshape(n1, n1, n1, n1), n1)
        return eri

    # ---------------------------------------------------------------- J/K
    def _gpu_mf(self):
        if self._jkmf is None:
            from gpu4pyscf import scf
            self._jkmf = scf.RHF(self.mol).density_fit(auxbasis=self.auxbasis)
            self._jkmf.verbose = 0
        return self._jkmf

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True, omega=None,
               direct_scf_tol=None, device=False):
        """DF J/K; ``device=True`` takes and returns device arrays (the
        orbital step of ``casscf_orb``)."""
        t0 = time.perf_counter()
        STATS['jk'] += 1
        lao = self._ao_cderi() if omega is None else None
        if lao is not None:
            out = self._jk_cached(lao, dm, hermi, with_j, with_k, device)
            STATS['jk_s'] += time.perf_counter() - t0
            return out
        if device:
            dm = cp.asnumpy(_stack(dm))
        mf = self._gpu_mf()
        dm = np.asarray(dm)
        vj, vk = mf.get_jk(self.mol, cp.asarray(dm), hermi=hermi,
                           with_j=with_j, with_k=with_k, omega=omega)
        conv = cp.asarray if device else cp.asnumpy
        out = (None if vj is None else conv(vj),
               None if vk is None else conv(vk))
        STATS['jk_s'] += time.perf_counter() - t0
        return out

    def _jk_cached(self, lao, dm, hermi, with_j, with_k, device=False):
        """J and K from the cached fitted tensor L[mu, nu, P].

        J = L (L . D) as two GEMVs.  K = sum_P L_P D L_P: for a symmetric D
        through its eigen-decomposition D = U s U^T (the CASSCF densities --
        core, and the AH step's core/active rotations -- are low rank), as
        M_P = L_P U and K = sum_P M_P s M_P^T, two GEMMs over rank r instead
        of nao; a non-symmetric D directly."""
        dms = cp.asarray(_stack(dm), dtype=np.float64)
        single = dms.ndim == 2
        dms = dms.reshape(-1, dms.shape[-2], dms.shape[-1])
        nao = dms.shape[-1]
        naux = lao.shape[2]
        L2 = lao.reshape(nao * nao, naux)
        vj = vk = None
        if with_j:
            rho = dms.reshape(len(dms), -1) @ L2                  # (n, P)
            vj = (rho @ L2.T).reshape(dms.shape)
        if with_k:
            vk = cp.empty_like(dms)
            # L as (mu, lambda, P) -> (mu, P, lambda) for M_P = L_P U
            lt = self._lt
            if lt is None:
                lt = self._lt = cp.ascontiguousarray(lao.transpose(0, 2, 1))
            for i, d in enumerate(dms):
                if hermi == 1:
                    s, u = cp.linalg.eigh((d + d.T) * .5)
                    keep = abs(s) > 1e-14 * max(float(abs(s).max()), 1e-300)
                    s, u = s[keep], u[:, keep]
                    m = (lt.reshape(nao * naux, nao) @ u).reshape(nao, naux * len(s))
                    ms = (m.reshape(nao, naux, -1) * s).reshape(nao, -1)
                    vk[i] = ms @ m.T
                else:
                    m = (lt.reshape(nao * naux, nao) @ d).reshape(nao, naux, nao)
                    # K[mu,nu] = sum_P sum_ls L[mu,l,P] D[l,s] L[s,nu,P]
                    vk[i] = m.reshape(nao, naux * nao) @ \
                        lt.reshape(nao, naux * nao).T
        if single:
            vj = None if vj is None else vj[0]
            vk = None if vk is None else vk[0]
        if device:
            return vj, vk
        return (None if vj is None else cp.asnumpy(vj),
                None if vk is None else cp.asnumpy(vk))

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
            self.auxmol = df_cpu.addons.make_auxmol(mol, self.auxbasis)
        self._intopt = self._j2c = self._jkmf = None
        self._lao = None
        self._lt = None
        self._lao_refused = False
        return self

    def build(self):
        return self


class _ERIS:
    """PySCF ``mcscf.df._ERIS`` (ppaa, papa, j_pc, k_pc, vhf_c) built on the
    device; the arrays handed back are host arrays, as PySCF's code reads
    them."""

    def __init__(self, casscf, mo, with_df):
        t0 = time.perf_counter()
        STATS['eris'] += 1
        ncore, ncas = casscf.ncore, casscf.ncas
        nocc = ncore + ncas
        nao, nmo = mo.shape
        mo = cp.asarray(mo)
        naux = with_df.get_naoaux()
        free = cp.cuda.runtime.memGetInfo()[0] + \
            cp.get_default_memory_pool().free_bytes()
        # one block of L[p0:p1, :, P] plus its share of ppaa
        per_p = (nmo * with_df.auxmol.nao_cart() + nmo * ncas * ncas) * 8 * 2
        blk = max(1, min(nmo, int(0.4 * free // per_p)))
        lpa = cp.empty((nmo, ncas, naux))
        ldiag = cp.empty((nmo, naux))
        k_cp = cp.zeros((ncore, nmo))
        blocks = []
        for p0 in range(0, nmo, blk):
            p1 = min(nmo, p0 + blk)
            L = with_df.lmo(mo[:, p0:p1], mo)               # (b, nmo, naux)
            lpa[p0:p1] = L[:, ncore:nocc]
            ldiag[p0:p1] = L[cp.arange(p1 - p0), cp.arange(p0, p1)]
            c0, c1 = p0, min(p1, ncore)
            if c0 < c1:
                Lc = L[:c1 - c0]
                k_cp[c0:c1] = cp.einsum('cpk,cpk->cp', Lc, Lc)
            blocks.append((p0, p1, L if len(range(0, nmo, blk)) == 1 else None))
            del L
        laa = cp.ascontiguousarray(lpa[ncore:nocc]).reshape(ncas * ncas, naux)
        ppaa = np.empty((nmo, nmo, ncas, ncas))
        for p0, p1, L in blocks:
            if L is None:
                L = with_df.lmo(mo[:, p0:p1], mo)
            ppaa[p0:p1] = cp.asnumpy(
                (L.reshape(-1, naux) @ laa.T).reshape(p1 - p0, nmo, ncas, ncas))
            del L
        lpa2 = lpa.reshape(nmo * ncas, naux)
        self.papa = cp.asnumpy(lpa2 @ lpa2.T).reshape(nmo, ncas, nmo, ncas)
        self.ppaa = ppaa
        self.j_pc = cp.asnumpy(ldiag @ ldiag[:ncore].T)
        self.k_pc = cp.asnumpy(k_cp.T)
        mo_h = cp.asnumpy(mo)
        dm_core = mo_h[:, :ncore] @ mo_h[:, :ncore].T
        vj, vk = casscf.get_jk(casscf.mol, dm_core)
        self.vhf_c = mo_h.T @ (vj * 2 - vk) @ mo_h
        STATS['eris_s'] += time.perf_counter() - t0
        logger.timer(casscf, 'GPU DF _ERIS', logger.process_clock(), t0)



def _get_fock(mc, mo_coeff=None, ci=None, eris=None, casdm1=None,
              verbose=None):
    """PySCF ``casci.get_fock``; without ``eris`` PySCF builds the J/K from
    ``mc._scf`` -- here the host, exact-integral SCF -- so that branch uses
    the CASSCF's own GPU J/K (DF or direct, as the CASSCF itself)."""
    from pyscf.mcscf import casci
    if eris is not None:
        return casci.get_fock(mc, mo_coeff, ci, eris, casdm1, verbose)
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if casdm1 is None:
        if ci is None:
            ci = mc.ci
        casdm1 = mc.fcisolver.make_rdm1(ci, mc.ncas, mc.nelecas)
    mo_coeff = cp.asnumpy(mo_coeff)
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    dm_core = np.dot(mo_coeff[:, :ncore] * 2, mo_coeff[:, :ncore].T)
    mocas = mo_coeff[:, ncore:nocc]
    dm = dm_core + mocas @ casdm1 @ mocas.T
    vj, vk = mc.get_jk(mc.mol, dm)
    return mc.get_hcore() + np.asarray(vj) - .5 * np.asarray(vk)


from electrolites.casscf_orb import OrbStepMixin  # noqa: E402


class _GPUCAS(OrbStepMixin):
    """Mixin ahead of PySCF's DF-CASCI/CASSCF in the MRO."""
    __name_mixin__ = 'GPU'

    def get_jk(self, mol, dm, hermi=1, with_j=True, with_k=True, omega=None):
        return self.with_df.get_jk(dm, hermi, with_j=with_j, with_k=with_k,
                                   omega=omega)

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        return _ERIS(self, mo_coeff, self.with_df)

    get_fock = _get_fock

    def nuc_grad_method(self):
        """The GPU DF-CASSCF gradient (``electrolites.casscf_grad``)."""
        from electrolites.casscf_grad import Gradients
        return Gradients(self)

    Gradients = nuc_grad_method

    def _state_average_nuc_grad_method(self, state=None):
        """PySCF's SA hook: the GPU SA-CASSCF gradient."""
        from electrolites.sacasscf_grad import Gradients
        return Gradients(self, state=state)

    def _state_average_nac_method(self):
        """PySCF's SA hook: GPU DF SA-CASSCF non-adiabatic couplings."""
        from electrolites.sacasscf_grad import NonAdiabaticCouplings
        return NonAdiabaticCouplings(self)


class _DirectERIS(_ERIS):
    """PySCF ``mc_ao2mo._ERIS`` for the exact-integral route.  ppaa and papa
    come from one batch of direct J and K builds over the active-pair
    densities D_uv = c_u c_v^T (u <= v):

        J[D_uv + D_vu]_mn = 2 (mn|uv)  ->  ppaa[p,q,u,v] = C^T J C / 2
        K[D_uv]_mn        = (mu|nv)    ->  papa[p,u,q,v] = C^T K C

    with K[D_vu] = K[D_uv]^T for the other triangle.  That is GPU4PySCF's J/K
    kernel, 2.8x faster here than transforming the four-centre integrals
    (Electrolites' ``get_int4c2e_ovpq``: 8.4 against 3.0 s per macro
    iteration on Ni dithiolene-/def2-SVP, 33.6 against 12.3 s on the Cu
    macrocycle).  vhf_c is the exact core J/K.  j_pc and k_pc, which PySCF
    reads only for the diagonal preconditioner of the orbital step
    (``mc1step.gen_g_hop``), are taken from the fitted integrals
    (``casscf.with_df``, used for nothing else here): the converged result
    does not depend on them."""

    def __init__(self, casscf, mo, with_df):
        super().__init__(casscf, mo, with_df)
        t0 = time.perf_counter()
        ncore, ncas = casscf.ncore, casscf.ncas
        mo = cp.asarray(mo)
        nao, nmo = mo.shape
        act = mo[:, ncore:ncore + ncas]
        iu, iv = np.triu_indices(ncas)
        dms = act.T[iu][:, :, None] * act.T[iv][:, None, :]     # D_uv
        mf = casscf._jk_mf()
        mol = casscf.mol
        vj = mf.get_j(mol, dms + dms.transpose(0, 2, 1), hermi=1)
        vk = mf.get_k(mol, dms, hermi=0)
        del dms
        jf = cp.empty((ncas, ncas, nao, nao))
        jf[iu, iv] = vj
        jf[iv, iu] = vj
        kf = cp.empty((ncas, ncas, nao, nao))
        kf[iu, iv] = vk
        kf[iv, iu] = vk.transpose(0, 2, 1)
        del vj, vk
        self.ppaa = cp.asnumpy(
            (mo.T @ jf @ mo).transpose(2, 3, 0, 1) * .5)
        del jf
        self.papa = cp.asnumpy(
            (mo.T @ kf @ mo).transpose(2, 0, 3, 1))
        STATS['eris_s'] += time.perf_counter() - t0


class _GPUDirect(OrbStepMixin):
    """Mixin ahead of PySCF's (exact-integral) CASCI/CASSCF in the MRO."""
    __name_mixin__ = 'GPU'

    get_fock = _get_fock

    def nuc_grad_method(self):
        raise NotImplementedError(
            'GPU CASSCF gradient: density-fitted CASSCF only '
            '(CASSCF(..., density_fit=True))')

    Gradients = nuc_grad_method

    def _jk_mf(self):
        mf = self.__dict__.get('_jkmf')
        if mf is None:
            from gpu4pyscf import scf
            mf = self._jkmf = scf.RHF(self.mol)
            mf.verbose = 0
        return mf

    def get_jk(self, mol, dm, hermi=1, with_j=True, with_k=True, omega=None):
        t0 = time.perf_counter()
        STATS['jk'] += 1
        mf = self._jk_mf()
        vj, vk = mf.get_jk(self.mol, cp.asarray(np.asarray(dm)), hermi=hermi,
                           with_j=with_j, with_k=with_k, omega=omega)
        out = (None if vj is None else cp.asnumpy(vj),
               None if vk is None else cp.asnumpy(vk))
        STATS['jk_s'] += time.perf_counter() - t0
        return out

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        return _DirectERIS(self, mo_coeff, self.with_df)


def _to_cpu_mf(mf):
    if getattr(mf, 'device', 'cpu') == 'gpu':
        mf = mf.to_cpu()
    return mf


def _make(kind, mf, ncas, nelecas, auxbasis=None, ncore=None, frozen=None,
          density_fit=None):
    if density_fit is None:
        density_fit = getattr(mf, 'with_df', None) is not None
    mf_cpu = _to_cpu_mf(mf)
    if auxbasis is None:
        auxbasis = getattr(getattr(mf, 'with_df', None), 'auxbasis', None)
    mol = mf_cpu.mol
    gdf = GPUDF(mol, auxbasis)
    base = (mcscf.CASSCF if kind == 'casscf' else mcscf.CASCI)(
        mf_cpu.undo_df() if hasattr(mf_cpu, 'undo_df') else mf_cpu,
        ncas, nelecas, ncore=ncore)
    if kind == 'casscf' and frozen is not None:
        base.frozen = frozen
    if density_fit:
        mc = mcscf.df.density_fit(base, auxbasis=auxbasis, with_df=gdf)
        mixin = _GPUCAS
    else:
        mc = base
        mc.with_df = gdf                  # j_pc / k_pc only (_DirectERIS)
        mixin = _GPUDirect
    old = mc.fcisolver
    new = fci_gpu.FCISolver(mol)
    for k, v in old.__dict__.items():
        if k not in ('mol', 'stdout'):
            setattr(new, k, v)
    mc.fcisolver = new
    return lib.set_class(mc, (mixin, mc.__class__))


def CASSCF(mf, ncas, nelecas, auxbasis=None, ncore=None, frozen=None,
           density_fit=None):
    """PySCF's CASSCF with integrals, J/K and CI on the GPU.  Density-fitted
    when ``mf`` is (or ``density_fit=True``), else exact integrals, as
    ``mcscf.CASSCF`` on the CPU."""
    return _make('casscf', mf, ncas, nelecas, auxbasis, ncore, frozen,
                 density_fit)


def CASCI(mf, ncas, nelecas, auxbasis=None, ncore=None, density_fit=None):
    """PySCF's CASCI with integrals, J/K and CI on the GPU (DF as CASSCF)."""
    return _make('casci', mf, ncas, nelecas, auxbasis, ncore,
                 density_fit=density_fit)
