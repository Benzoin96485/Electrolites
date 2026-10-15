"""MC-PDFT on the GPU CASSCF (`docs/FEATURE_TARGETS_PYSCF_2.13.md` M8).

PySCF 2.13's ``mcpdft`` (MC-PDFT, translated on-top functionals) wraps a
PySCF CASSCF/CASCI; ``mcpdft.CASSCF(gpu_mf, ...)`` cannot take a GPU mean
field, and its two expensive pieces run on the host: the classical energy
(a J/K build) and the on-top energy -- a grid loop that builds the on-top
pair density Pi(r) = rho_a rho_b + 1/2 sum phi_u phi_v phi_w phi_x
lambda_uvwx (the 2-cumulant) and evaluates a translated functional.

``MCPDFT(gpu_mc, 'tPBE')`` puts PySCF's MC-PDFT class on top of a GPU
CASSCF/CASCI from ``electrolites.casscf_gpu`` (its CASSCF on the GPU, as in
Phase 2) and replaces those two pieces:

* ``energy_mcwfn``: PySCF's expression, J and K from the CASSCF's own GPU
  J/K (``mc.get_jk``: DF on a DF CASSCF, direct otherwise);
* ``energy_dft``: the on-top energy on the device -- the AO values on the
  ``ot.grids`` points (the very grid PySCF would use), rho_a/b and their
  gradients, Pi from one GEMM per block over the active pairs (ncas^2), the
  translation rho_{a,b} = rho/2 (1 +- zeta), zeta = (1 - 4 Pi/rho^2)^1/2,
  and GPU4PySCF's libxc evaluation (spin-polarised).  Translated ('t')
  LDA/GGA functionals and their global hybrids; fully translated ('ft')
  and meta-GGA functionals fall back to PySCF's host loop.

    from electrolites import casscf_gpu, mcpdft_gpu
    mc = mcpdft_gpu.MCPDFT(casscf_gpu.CASSCF(gpu_mf, 6, 6, density_fit=True),
                           'tPBE').run()
    mc.e_tot, mc.e_mcscf
"""
import numpy as np
import cupy as cp

from pyscf import lib
from pyscf.lib import logger

BLOCK = 32768


def _ao_values(mol, coords, deriv):
    """(nd, nao, ng) AO values (and gradients) on the device."""
    from gpu4pyscf.dft import numint
    ao = numint.eval_ao(mol, coords, deriv=deriv)
    nao = mol.nao
    if ao.ndim == 2:
        ao = ao[None]
    if ao.shape[1] != nao:
        ao = ao.transpose(0, 2, 1)
    return cp.ascontiguousarray(ao)


def _rho(ao, dm, nd):
    """rho (and its gradient) of a symmetric AO density, (nd, ng)."""
    x = dm @ ao[0]                                       # (nao, ng)
    out = cp.empty((nd, ao.shape[2]))
    out[0] = (ao[0] * x).sum(axis=0)
    for k in range(1, nd):
        out[k] = 2 * (ao[k] * x).sum(axis=0)
    return out


_GRIDS = {}


def _grid(grids, mol):
    """The device coordinates/weights of ``grids``.  PySCF's ``energy_tot``
    resets the on-top functional (and so its grid) on every call, and the
    host build was two thirds of a GPU MC-PDFT energy (Ni dithiolene,
    0.46 of 0.69 s); it is cached per molecule and grid setting."""
    key = (id(mol), mol.atom_coords().tobytes(), grids.level,
           str(grids.atom_grid), str(grids.prune), str(grids.radi_method),
           str(grids.becke_scheme), grids.radii_adjust is None)
    hit = _GRIDS.get(key)
    if hit is not None and hit[0] is mol:
        return hit[1], hit[2]
    if grids.coords is None:
        grids.build()
    out = (mol, cp.asarray(grids.coords), cp.asarray(grids.weights))
    _GRIDS.clear()
    _GRIDS[key] = out
    return out[1], out[2]


def energy_ot(ot, casdm1s, casdm2, mo_coeff, ncore, max_memory=2000,
              hermi=1):
    """GPU version of PySCF ``otfnal.energy_ot`` for translated functionals
    (falls back to PySCF's for anything else)."""
    from pyscf.mcpdft import _dms
    from pyscf.mcpdft import otfnal as otf
    from gpu4pyscf.dft import numint as gnumint
    prefix = getattr(ot, 'transl_prefix', '')
    if (prefix != 't' or isinstance(ot, otf.ftransfnal)
            or ot.xctype not in ('LDA', 'GGA')):
        return otf.energy_ot(ot, casdm1s, casdm2, mo_coeff, ncore,
                             max_memory=max_memory, hermi=hermi)
    xc = ot.otxc[len(prefix):]
    mol = ot.mol
    ncas = casdm2.shape[0]
    cascm2 = _dms.dm2_cumulant(casdm2, casdm1s)
    dm1s = _dms.casdm1s_to_dm1s(ot, casdm1s, mo_coeff=mo_coeff, ncore=ncore,
                                ncas=ncas)
    dm1s = cp.asarray(dm1s)
    ca = cp.asarray(mo_coeff[:, ncore:ncore + ncas])
    cm2 = cp.asarray(cascm2).reshape(ncas * ncas, ncas * ncas)
    grids = ot.grids
    coords, weights = _grid(grids, mol)
    nd = 1 if ot.xctype == 'LDA' else 4
    ni = gnumint.NumInt()
    e_ot = 0.
    for g0 in range(0, coords.shape[0], BLOCK):
        g1 = min(coords.shape[0], g0 + BLOCK)
        ao = _ao_values(mol, coords[g0:g1], 0 if nd == 1 else 1)
        rho = cp.stack([_rho(ao, dm1s[0], nd), _rho(ao, dm1s[1], nd)])
        phi = ca.T @ ao[0]                               # (ncas, ng)
        pp = (phi[:, None, :] * phi[None, :, :]).reshape(ncas * ncas, -1)
        pi = rho[0, 0] * rho[1, 0] + .5 * (pp * (cm2 @ pp)).sum(axis=0)
        # translation (PySCF transfnal.get_ratio / get_zeta, 0/0 = 1)
        rho_avg = .5 * (rho[0] + rho[1])
        r = cp.where(pi == 0, 0., 1.)
        ok = rho_avg[0] >= .5e-15
        r = cp.where(ok, pi / cp.where(ok, rho_avg[0], 1.) ** 2, r)
        zeta = cp.where(r < 1, cp.sqrt(cp.clip(1 - r, 0, None)), 0.)
        w = rho_avg * zeta[None]
        rho_t = cp.stack([rho_avg + w, rho_avg - w])
        if nd == 1:
            rho_t = rho_t[:, 0]
        exc = ni.eval_xc_eff(xc, rho_t, deriv=0, xctype=ot.xctype,
                             spin=1)[0]
        dens = rho_t[0] + rho_t[1] if nd == 1 else rho_t[0, 0] + rho_t[1, 0]
        e_ot += float((exc.ravel() * dens * weights[g0:g1]).sum())
    return e_ot


class _GPUPDFT:
    """Mixin ahead of PySCF's MC-PDFT class (itself over a GPU CASSCF)."""
    __name_mixin__ = 'GPU'

    def energy_mcwfn(self, mo_coeff=None, ci=None, ot=None, state=0,
                     casdm1s=None, casdm2=None, verbose=None):
        from pyscf import ao2mo
        from pyscf.mcpdft import _dms
        if ot is None:
            ot = self.otfnal
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ci is None:
            ci = self.ci
        if casdm1s is None:
            casdm1s = self.make_one_casdm1s(ci=ci, state=state)
        if casdm2 is None:
            casdm2 = self.make_one_casdm2(ci=ci, state=state)
        mo_coeff = cp.asnumpy(mo_coeff)
        dm1s = _dms.casdm1s_to_dm1s(self, casdm1s, mo_coeff=mo_coeff)
        cascm2 = _dms.dm2_cumulant(casdm2, casdm1s)
        hyb_x, hyb_c = ot._numint.rsh_and_hybrid_coeff(ot.otxc,
                                                       self.mol.spin)[2]
        vnn = self._scf.energy_nuc()
        h = self._scf.get_hcore()
        dm1 = dm1s[0] + dm1s[1]
        vj, vk = self.get_jk(self.mol, np.asarray(dm1s), hermi=1)
        vj = np.asarray(vj)
        vk = np.asarray(vk)
        e = vnn + np.tensordot(h, dm1) + np.tensordot(vj[0] + vj[1], dm1) / 2
        if abs(hyb_x) > 1e-10:
            e_x = -(np.tensordot(vk[0], dm1s[0])
                    + np.tensordot(vk[1], dm1s[1])) / 2
            e += hyb_x * e_x
        if abs(hyb_c) > 1e-10:
            aeri = ao2mo.restore(1, self.get_h2eff(mo_coeff), self.ncas)
            e += hyb_c * np.tensordot(aeri, cascm2, axes=4) / 2
        return e

    def energy_dft(self, mo_coeff=None, ci=None, ot=None, state=0,
                   casdm1s=None, casdm2=None, max_memory=None, hermi=1):
        if ot is None:
            ot = self.otfnal
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ci is None:
            ci = self.ci
        if casdm1s is None:
            casdm1s = self.make_one_casdm1s(ci, state=state)
        if casdm2 is None:
            casdm2 = self.make_one_casdm2(ci, state=state)
        return energy_ot(ot, casdm1s, casdm2, cp.asnumpy(mo_coeff),
                         self.ncore)


def MCPDFT(mc, ot='tPBE', **kwargs):
    """PySCF's MC-PDFT class over a GPU CASSCF/CASCI (``casscf_gpu``),
    with its classical and on-top energies on the GPU."""
    from pyscf.mcpdft.mcpdft import get_mcpdft_child_class
    if getattr(mc, 'with_df', None) is not None and not hasattr(mc._scf,
                                                                'with_df'):
        # PySCF's _PDFT re-initialises a DF class from mc._scf.with_df; the
        # GPU CASSCF keeps its (GPU) with_df on itself, not on the host SCF
        with lib.temporary_env(mc._scf, with_df=mc.with_df):
            obj = get_mcpdft_child_class(mc, ot, **kwargs)
    else:
        obj = get_mcpdft_child_class(mc, ot, **kwargs)
    return lib.set_class(obj, (_GPUPDFT, obj.__class__))


def CASSCF(mf, ot, ncas, nelecas, **kwargs):
    from electrolites import casscf_gpu
    return MCPDFT(casscf_gpu.CASSCF(mf, ncas, nelecas, **kwargs), ot)


def CASCI(mf, ot, ncas, nelecas, **kwargs):
    from electrolites import casscf_gpu
    return MCPDFT(casscf_gpu.CASCI(mf, ncas, nelecas, **kwargs), ot)
