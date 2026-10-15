"""State-averaged DF-CASSCF nuclear gradients on the GPU
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` M6).

PySCF's ``grad/sacasscf.py`` Lagrangian, unchanged as an algorithm, on the
object ``electrolites.casscf_gpu`` builds:

* the Lagrange equations (``lagrange.Gradients``: the state's orbital/CI
  gradient, the SA Hessian-vector products of ``newton_casscf.gen_g_hop``,
  the CG solve) are PySCF's code, run on CASSCF objects that carry the GPU
  mixin -- so their integrals, J/K and CI sigma are the GPU ones;
* the three derivative contractions are GPU ports of PySCF's DF ones
  (``df/grad/sacasscf.py``): the state's Hellmann-Feynman gradient
  (``casscf_grad.grad_elec`` with the generalised Fock over all rows),
  ``Lorb_dot_dgorb_dx`` and ``Lci_dot_dgci_dx`` -- h1/S through GPU4PySCF's
  hcore gradient, J/K through its DF J/K gradient by polarisation, and the
  active-active 2-RDM through ``dfmp2_grad``'s three-index kernel.

    mc = CASSCF(gpu_mf, 8, 8, density_fit=True).state_average_([.5, .5]).run()
    g1 = mc.nuc_grad_method().kernel(state=1)
"""
import numpy as np
import cupy as cp

from pyscf import lib, mcscf
from pyscf.lib import logger
from pyscf.grad import sacasscf as sacasscf_grad

from electrolites import casscf_grad


def _gpu_fcasscf(base, fcasscf):
    """Give a PySCF-built CASSCF (from ``make_fcasscf``) the DF class and the
    GPU mixin of ``base`` (it already has its __dict__: with_df, fcisolver)."""
    from electrolites.casscf_gpu import _GPUCAS
    with_df = base.with_df
    fc = mcscf.df.density_fit(fcasscf, auxbasis=with_df.auxbasis,
                              with_df=with_df)
    return lib.set_class(fc, (_GPUCAS, fc.__class__))


def Lorb_dot_dgorb_dx(Lorb, mc, mo_coeff, ci, eris, kit):
    """GPU port of PySCF ``df/grad/sacasscf.Lorb_dot_dgorb_dx``."""
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nocc = ncore + ncas
    C = cp.asarray(mo_coeff)
    nmo = C.shape[1]
    L = cp.asarray(Lorb)
    CL = C @ L
    s0_inv = C @ C.T
    Cc, Ca = C[:, :ncore], C[:, ncore:nocc]
    CLc, CLa = CL[:, :ncore], CL[:, ncore:nocc]
    casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)
    dm1a = cp.asarray(casdm1)
    dm2 = cp.asarray(casdm2)
    Dc = 2 * Cc @ Cc.T
    Da = Ca @ dm1a @ Ca.T
    DcL = 2 * CLc @ Cc.T
    DaL = CLa @ dm1a @ Ca.T
    DcL += DcL.T
    DaL += DaL.T
    dm1 = Dc + Da
    dm1L = DcL + DaL

    ppaa = cp.asarray(eris.ppaa)                 # (p, q, u, v) = (pq|uv)
    papa = cp.asarray(eris.papa)                 # (p, u, q, v) = (pu|qv)
    La = L[:, ncore:nocc]
    # aapa[u,v,i,w] = (uv|iw);  aapaL[u,v,i,w] = sum_p (ip|uv) L[p,w]
    #   + K1[i,w,u,v] + K1[i,w,v,u],  K1[i,u,v,w] = sum_p (iu|pv) L[p,w]
    aapa = ppaa[:, ncore:nocc].transpose(2, 3, 0, 1)
    aapaL = cp.einsum('ipuv,pw->uviw', ppaa, La)
    k1 = cp.einsum('iupv,pw->iuvw', papa, La)
    aapaL += k1.transpose(2, 3, 0, 1) + k1.transpose(3, 2, 0, 1)
    k1 = ppaa = papa = None

    gdf = mc.with_df
    vj, vk = gdf.get_jk(cp.asnumpy(cp.stack([Dc, Da, DcL, DaL])), hermi=1)
    vj, vk = cp.asarray(vj), cp.asarray(vk)
    vhf = vj - .5 * vk
    vhf_c, vhf_a, vhfL_c, vhfL_a = vhf
    h1 = cp.asarray(mc.get_hcore())
    gfock = h1 @ dm1L
    gfock += (vhf_c + vhf_a) @ DcL
    gfock += (vhfL_c + vhfL_a) @ Dc
    gfock += vhfL_c @ Da
    gfock += vhf_c @ DaL
    gfock = s0_inv @ gfock
    gfock += C @ cp.einsum('uviw,uvtw->it', aapaL, dm2) @ Ca.T
    gfock += C @ cp.einsum('uviw,vuwt->it', aapa, dm2) @ CLa.T
    dme0 = .5 * (gfock + gfock.T)

    de = kit.hcore(dm1L, dme0)
    # vhf1c.dm1L + vhf1cL.dm1 + vhf1a.dmL_core + vhf1aL.dm_core (x2)
    #   = Dc.G[DcL + DaL] + Da.G[DcL]   (skeleton derivatives)
    de += kit.bilinear(Dc, DcL + DaL) + kit.bilinear(Da, DcL)
    de += kit.active_rotated(Ca, CLa, dm2)
    return de


def Lci_dot_dgci_dx(Lci, weights, mc, mo_coeff, ci, eris, kit):
    """GPU port of PySCF ``df/grad/sacasscf.Lci_dot_dgci_dx``."""
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nocc = ncore + ncas
    C = cp.asarray(mo_coeff)
    Cc, Ca, Co = C[:, :ncore], C[:, ncore:nocc], C[:, :nocc]
    casdm1, casdm2 = mc.fcisolver.trans_rdm12(Lci, ci, ncas, nelecas)
    dm1a = cp.asarray(casdm1)
    dm2 = cp.asarray(casdm2)
    dm1a = dm1a + dm1a.T
    dm2 = dm2 + dm2.transpose(1, 0, 3, 2)
    Dc = 2 * Cc @ Cc.T
    Da = Ca @ dm1a @ Ca.T
    gdf = mc.with_df
    vj, vk = gdf.get_jk(cp.asnumpy(cp.stack([Dc, Da])), hermi=1)
    vj, vk = cp.asarray(vj), cp.asarray(vk)
    vhf_c = vj[0] - .5 * vk[0]
    vhf_a = vj[1] - .5 * vk[1]
    h1 = cp.asarray(mc.get_hcore())
    nmo = C.shape[1]
    gfock = cp.zeros((nmo, nmo))
    gfock[:, :nocc] = C.T @ vhf_a @ Co * 2
    gfock[:, ncore:nocc] = C.T @ (h1 + vhf_c) @ Ca @ dm1a
    gfock[:, ncore:nocc] += casscf_grad._two_e_fock(gdf, C, Ca, dm2)
    dme0 = C @ (.5 * (gfock + gfock.T)) @ C.T
    de = kit.hcore(Da, dme0)
    de += kit.bilinear(Dc, Da)
    de += kit.active(Ca, dm2)
    return de


class Gradients(sacasscf_grad.Gradients):
    """``mc.nuc_grad_method()`` of a GPU state-averaged DF-CASSCF:
    ``kernel(state=i)`` is state i's gradient."""

    def make_fcasscf(self, state=None, casscf_attr={}, fcisolver_attr={}):
        fc = super().make_fcasscf(state, casscf_attr, fcisolver_attr)
        return _gpu_fcasscf(self.base, fc)

    def _kit(self):
        kit = self.__dict__.get('_gpu_kit')
        if kit is None:
            kit = self._gpu_kit = casscf_grad._Kit(self.base)
        return kit

    def get_ham_response(self, state=None, atmlst=None, verbose=None, mo=None,
                         ci=None, eris=None, mf_grad=None, **kwargs):
        if state is None:
            state = self.state
        if mo is None:
            mo = self.base.mo_coeff
        if ci is None:
            ci = self.base.ci
        fc = self.make_fcasscf(state)
        de = casscf_grad.grad_elec(fc, mo, ci[state], verbose, kit=self._kit())
        return de + casscf_grad.Gradients(self.base).grad_nuc()

    def get_LdotJnuc(self, Lvec, state=None, atmlst=None, verbose=None,
                     mo=None, ci=None, eris=None, mf_grad=None, **kwargs):
        if state is None:
            state = self.state
        if mo is None:
            mo = self.base.mo_coeff
        if ci is None:
            ci = self.base.ci[state]
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo(mo)
        elif eris is None:
            eris = self.eris
        Lorb, Lci = self.unpack_uniq_var(Lvec)
        kit = self._kit()
        de = Lci_dot_dgci_dx(Lci, self.weights, self.base, mo, ci, eris, kit)
        de += Lorb_dot_dgorb_dx(Lorb, self.base, mo, ci, eris, kit)
        return de

    def kernel(self, state=None, **kwargs):
        if state is None:
            state = self.state
        if state is None:
            raise ValueError('SA-CASSCF gradient: give state=<root>')
        # PySCF builds a CPU SCF gradient object it does not need here
        kwargs.setdefault('mf_grad', object())
        return super().kernel(state=state, **kwargs)


class NonAdiabaticCouplings(Gradients):
    """SA-CASSCF non-adiabatic couplings <bra|d ket/dR> on the GPU, DF
    (PySCF has them only without density fitting, ``nac/sacasscf.py``;
    this is that algorithm with the DF/GPU pieces above).
    ``kernel(state=(ket, bra))``; ``mult_ediff``, ``use_etfs`` as PySCF."""

    def __init__(self, mc, state=None, mult_ediff=False, use_etfs=False):
        self.mult_ediff = mult_ediff
        self.use_etfs = use_etfs
        Gradients.__init__(self, mc, state=state)

    def make_fcasscf_nacs(self, state=None, casscf_attr=None,
                          fcisolver_attr=None):
        if state is None:
            state = self.state
        casscf_attr = {} if casscf_attr is None else casscf_attr
        fcisolver_attr = {} if fcisolver_attr is None else fcisolver_attr
        ket, bra = state
        base = self.base
        solver = base.fcisolver
        tdm = getattr(solver, '_base_class', type(solver))
        t1, t2 = tdm.trans_rdm12(solver, base.ci[bra], base.ci[ket],
                                 base.ncas, base.nelecas)
        t1 = .5 * (t1 + t1.T)
        t2 = .5 * (t2 + t2.transpose(1, 0, 3, 2))
        fcisolver_attr['make_rdm12'] = lambda *a, **k: (t1, t2)
        fcisolver_attr['make_rdm1'] = lambda *a, **k: t1
        fcisolver_attr['make_rdm2'] = lambda *a, **k: t2
        return self.make_fcasscf(state=ket, casscf_attr=casscf_attr,
                                 fcisolver_attr=fcisolver_attr)

    def get_wfn_response(self, **kwargs):
        from pyscf.nac import sacasscf as nac_sacasscf
        with lib.temporary_env(nac_sacasscf.NonAdiabaticCouplings,
                               make_fcasscf_nacs=NonAdiabaticCouplings.make_fcasscf_nacs):
            return nac_sacasscf.NonAdiabaticCouplings.get_wfn_response(
                self, **kwargs)

    def _core_grad(self, mo):
        """The core-electron (RHF-like) part of the Hellmann-Feynman gradient
        (PySCF ``nac.sacasscf.grad_elec_core``)."""
        mc = self.base
        ncore = mc.ncore
        C = cp.asarray(mo)
        eris = self.eris if self.eris is not None else mc.ao2mo(mo)
        f0 = C.T @ cp.asarray(mc.get_hcore()) @ C + cp.asarray(eris.vhf_c)
        occ = cp.zeros(C.shape[1])
        occ[:ncore] = 2.
        f0 = f0 * occ[None, :]
        dme0 = C @ (.5 * (f0 + f0.T)) @ C.T
        Dc = 2 * C[:, :ncore] @ C[:, :ncore].T
        kit = self._kit()
        return kit.hcore(Dc, dme0) + kit.eg(Dc)

    def nac_csf(self, mo_coeff=None, ci=None, state=None, mf_grad=None,
                atmlst=None):
        """The 'CSF contribution' (PySCF ``nac.sacasscf.nac_csf``), times
        the energy gap as PySCF returns it."""
        from gpu4pyscf.grad import rhf as rhf_grad_gpu
        mc = self.base
        if state is None:
            state = self.state
        if mo_coeff is None:
            mo_coeff = mc.mo_coeff
        if ci is None:
            ci = mc.ci
        ket, bra = state
        solver = mc.fcisolver
        tdm = getattr(solver, '_base_class', type(solver))
        t1 = np.asarray(tdm.trans_rdm1(solver, ci[bra], ci[ket], mc.ncas,
                                       mc.nelecas))
        t1 = t1.T - t1
        C = cp.asnumpy(mo_coeff)
        ca = C[:, mc.ncore:mc.ncore + mc.ncas]
        tm1 = ca @ t1 @ ca.T
        mol = mc.mol
        from gpu4pyscf import scf as scf_gpu
        s1 = cp.asnumpy(rhf_grad_gpu.Gradients(
            scf_gpu.RHF(casscf_grad._closed(mol))).get_ovlp(mol))
        nac = np.zeros((mol.natm, 3))
        for k, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
            nac[k] += .5 * np.einsum('xij,ij->x', s1[:, p0:p1], tm1[p0:p1])
        e = mc.e_states
        return nac * (e[bra] - e[ket])

    def get_ham_response(self, state=None, atmlst=None, verbose=None, mo=None,
                         ci=None, eris=None, mf_grad=None, **kwargs):
        if state is None:
            state = self.state
        if mo is None:
            mo = self.base.mo_coeff
        if ci is None:
            ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo(mo)
        use_etfs = kwargs.get('use_etfs', self.use_etfs)
        ket, bra = state
        fc = self.make_fcasscf_nacs(state)
        nac = casscf_grad.grad_elec(fc, mo, ci[ket], verbose, kit=self._kit())
        nac -= self._core_grad(mo)
        if not use_etfs:
            nac += self.nac_csf(mo_coeff=mo, ci=ci, state=state)
        return nac

    def kernel(self, *args, **kwargs):
        mult_ediff = kwargs.pop('mult_ediff', self.mult_ediff)
        state = kwargs.get('state', self.state)
        assert len(state) == 2, 'NACs are defined between 2 states'
        if state[0] == state[1]:
            return np.zeros((self.mol.natm, 3))
        kwargs['state'] = state
        nac = Gradients.kernel(self, *args, **kwargs)
        if not mult_ediff:
            ket, bra = state
            e = self.base.e_states
            nac /= e[bra] - e[ket]
        return nac
