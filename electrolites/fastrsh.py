"""
Fused exchange build for range-separated hybrids (omega-B97X and friends).

A range-separated hybrid with a non-zero short-range Hartree-Fock fraction
needs

    K = hyb * K(1/r)  +  (alpha - hyb) * K(erf(omega r)/r)

GPU4PySCF's ``dft.rks.get_veff`` builds those two matrices with two
independent calls to ``get_k``.  The two passes screen the same shell pairs,
walk the same quartets, read the same density and write the same matrix; only
the Rys roots differ.  Importing this module replaces ``RKS.get_veff`` with a
copy of GPU4PySCF's that asks ``fastk.get_k_rsh`` for the combination in one
pass, so the geometry, the primitive-pair data and the atomicAdds are paid
once.  Everything else in ``get_veff`` -- grids, NLC, the Coulomb build, the
energy bookkeeping -- is GPU4PySCF's own code, and every case the fused path
does not cover falls back to it unchanged.

    import fastk, fastrsh          # fastrsh needs fastk's kernels

When ``fastnlc`` is imported as well, ``get_veff`` also asks ``fastxcnlc`` to
build the XC and the non-local-correlation potentials in one pass over the
grid instead of two -- see that module.

Spin-restricted closed-shell only; UKS passes two density matrices and is
handed straight back to GPU4PySCF.
"""
import sys
import numpy as np
import cupy

from gpu4pyscf.dft import rks as _rks
from gpu4pyscf.scf import jk as _jk
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array, asarray
from gpu4pyscf.__config__ import num_devices

from . import fastk
from .compat import NEW_JK_ABI


def _opt(mf, mol, omega):
    """GPU4PySCF's per-omega _VHFOpt cache (see scf.hf.RHF.get_k)."""
    vhfopt = mf._opt_gpu.get(omega)
    if vhfopt is None:
        with mol.with_range_coulomb(omega):
            vhfopt = mf._opt_gpu[omega] = _jk._VHFOpt(
                mol, mf.direct_scf_tol, tile=1).build()
    return vhfopt


def get_k_rsh(mf, mol, dm, hermi, omega, coef_f, coef_l):
    """coef_f*K(full range) + coef_l*K(long range), or None if unsupported."""
    if num_devices != 1 or hermi != 1 or getattr(dm, 'ndim', 0) != 2:
        return None
    if not isinstance(dm, cupy.ndarray):
        return None                       # GPU4PySCF's get_k moves it back
    if getattr(mf, 'with_df', None) is not None:
        return None                       # density fitting has its own get_veff
    if not hasattr(mf, '_opt_gpu'):
        return None                       # not a direct-SCF object
    opt_f = _opt(mf, mol, mol.omega)
    opt_l = _opt(mf, mol, omega)
    if opt_f.h_shls or opt_l.h_shls:      # l > LMAX goes through the CPU path
        return None
    nao_orig = opt_f.mol.nao
    dms = cupy.asarray(dm, order='C').reshape(-1, nao_orig, nao_orig)
    dms = cupy.asarray(opt_f.apply_coeff_C_mat_CT(dms), order='C')
    vk = fastk.get_k_rsh(opt_f, opt_l, dms, hermi, coef_f, coef_l)
    vk = opt_f.apply_coeff_CT_mat_C(vk)
    return vk.reshape(dm.shape)


def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    """gpu4pyscf.dft.rks.get_veff with one line changed: when the functional
    needs both a short- and a long-range Hartree-Fock fraction, the two
    exchange builds are fused into one."""
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    t0 = logger.init_timer(ks)
    _rks.initialize_grids(ks, mol, dm)

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        nlc_code = None
        if ks.do_nlc():
            if ni.libxc.is_nlc(ks.xc):
                nlc_code = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                nlc_code = ks.nlc
        fused = None
        if nlc_code is not None and 'fastnlc' in sys.modules:
            # RKS.grids and RKS.nlcgrids are the same points, so the XC and
            # NLC builds can share one walk over them (fastxcnlc)
            from . import fastxcnlc
            if fastxcnlc.same_grid(ks.grids, ks.nlcgrids):
                fused = fastxcnlc.nr_rks_nlc(ni, mol, ks.grids, ks.xc,
                                             nlc_code, dm, hermi, ks.verbose)
        if fused is not None:
            n, exc, vxc = fused
        else:
            n, exc, vxc = ni.nr_rks(mol, ks.grids, ks.xc, dm)
            if nlc_code is not None:
                n, enlc, vnlc = ni.nr_nlc_vxc(mol, ks.nlcgrids, nlc_code, dm)
                exc += enlc
                vxc += vnlc
        logger.debug(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)

    dm_orig = dm
    vj_last = getattr(vhf_last, 'vj', None)
    if vj_last is not None:
        dm = asarray(dm) - asarray(dm_last)
    vj = ks.get_j(mol, dm, hermi)
    if vj_last is not None:
        vj += asarray(vj_last)
    vxc += vj
    ecoul = float(cupy.einsum('ij,ij', dm_orig, vj).real) * .5

    vk = None
    if ni.libxc.is_hybrid_xc(ks.xc):
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
        vk = None
        if omega != 0 and alpha != 0 and hyb != 0:
            # the fused build; None means the case is not covered
            vk = get_k_rsh(ks, mol, dm, hermi, omega, hyb, alpha - hyb)
        if vk is not None:
            pass
        elif omega == 0:
            vk = ks.get_k(mol, dm, hermi)
            vk *= hyb
        elif alpha == 0: # LR=0, only SR exchange
            vk = ks.get_k(mol, dm, hermi, omega=-omega)
            vk *= hyb
        elif hyb == 0: # SR=0, only LR exchange
            vk = ks.get_k(mol, dm, hermi, omega=omega)
            vk *= alpha
        else: # SR and LR exchange with different ratios
            vk = ks.get_k(mol, dm, hermi)
            vk *= hyb
            vklr = ks.get_k(mol, dm, hermi, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        vk *= .5
        if vj_last is not None:
            vk += asarray(vhf_last.vk)
        vxc -= vk
        exc -= float(cupy.einsum('ij,ij', dm_orig, vk).real) * .5
    t0 = logger.timer(ks, 'veff', *t0)
    vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc


_PATCHED = False
# GPU4PySCF 1.8.0 fused the two range-separated exchange builds itself:
# dft.rks.get_veff now makes a single ks.get_k(mol, dm, hermi, omega, alpha,
# hyb) call for lr*erf + sr*erfc, where 1.7.x made two get_k calls and scaled
# them.  Replacing get_veff therefore buys nothing on 1.8.x -- and would put a
# 1.7.x-shaped reimplementation of it in the way of upstream's own incremental
# ecoul handling.  fastk serves the fused request at its own patch point
# instead (fastk.get_k_rsh through _get_k_dispatch), so this module stands
# down.
_RETIRED = NEW_JK_ABI


def apply_patch():
    global _PATCHED
    if _RETIRED or _PATCHED:
        return
    _rks.RKS._get_veff_orig = _rks.RKS.get_veff
    _rks.RKS.get_veff = get_veff
    _PATCHED = True


apply_patch()
