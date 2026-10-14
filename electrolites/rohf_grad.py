"""ROHF/ROKS analytic gradients on the GPU (`docs/FEATURE_TARGETS_PYSCF_2.13.md`
S3).

GPU4PySCF 1.8.1 sets ``nuc_grad_method = NotImplemented`` on ROHF/ROKS while
both inherit the RHF ``Gradients``, which asserts (``grad/rhf.py:424``);
``electrolites.guards`` sends them to PySCF on the CPU.  PySCF's ROHF/ROKS
gradient (``pyscf/grad/rohf.py``, ``roks.py``) is the UHF/UKS gradient with
two changes, made here on GPU4PySCF's UHF/UKS gradient classes (direct and
density-fitted):

* the energy-weighted density uses the Roothaan pieces: for each spin
  P_s C_s^T F_s C_s P_s with P_s the occupied projector and F_s the
  alpha/beta Fock matrix (PySCF's ``rohf.make_rdm1e``);
* the density is tagged with stacked orbitals (C, C) and occupations
  (occ > 0, occ == 2), so the J/K and XC derivative paths see a UHF-shaped
  density (PySCF's ``_tag_rdm1``).

Everything else -- the 1e derivatives, the two-electron and XC derivative
kernels (and Electrolites' patches of them) -- is the UHF/UKS code.

    import electrolites
    electrolites.enable('rohf_grad')
    g = gpu4pyscf.dft.ROKS(mol, xc='b3lyp').run().Gradients().kernel()
"""
import cupy as cp
import numpy as np

from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.scf import rohf
from gpu4pyscf.dft import roks
from gpu4pyscf.grad import uhf as uhf_grad, uks as uks_grad
from gpu4pyscf.df import df_jk
from gpu4pyscf.df.grad import uhf as df_uhf_grad, uks as df_uks_grad

from electrolites.features import attach


def make_rdm1e(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None):
    mf = mf_grad.base
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    vhf = mf.get_veff(mf.mol, dm)
    h1e = mf.get_hcore()
    out = []
    for s, occ in enumerate((mo_occ > 0, mo_occ == 2)):
        f = h1e + vhf[s]
        c = mo_coeff[:, occ]
        p = c @ c.T
        out.append(p @ f @ p)
    return cp.asarray(out)


def _stacked(mo_coeff, mo_occ):
    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)
    return (cp.stack([mo_coeff, mo_coeff]),
            cp.stack([(mo_occ > 0).astype(np.float64),
                      (mo_occ == 2).astype(np.float64)]))


def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None,
              atmlst=None):
    """GPU4PySCF's UHF ``grad_elec`` with ROHF densities."""
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_occ is None:
        mo_occ = mf.mo_occ
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)
    t0 = log.init_timer()
    dm0 = mf.make_rdm1(cp.asarray(mo_coeff), cp.asarray(mo_occ))
    dme0 = make_rdm1e(mf_grad, mo_energy, mo_coeff, mo_occ)
    c2, occ2 = _stacked(mo_coeff, mo_occ)
    dm0 = tag_array(cp.asarray(dm0), mo_coeff=c2, mo_occ=occ2)
    e1 = mf_grad._hcore_energy(dm0[0] + dm0[1], dme0[0] + dme0[1])
    e2 = mf_grad.energy_ee(mol, dm0)
    de = e1 + e2
    de += cp.asnumpy(mf_grad.extra_force())
    log.timer_debug1('ROHF/ROKS gradients of electronic part', *t0)
    return de


def _gradient_class(base):
    return type('RO' + base.__name__, (base,),
                {'grad_elec': grad_elec,
                 'make_rdm1e': make_rdm1e,
                 '__module__': __name__})


ROHFGradients = _gradient_class(uhf_grad.Gradients)
ROKSGradients = _gradient_class(uks_grad.Gradients)
DFROHFGradients = _gradient_class(df_uhf_grad.Gradients)
DFROKSGradients = _gradient_class(df_uks_grad.Gradients)


def gradients(mf):
    """The ROHF/ROKS gradient object for ``mf`` (direct or DF)."""
    ks = isinstance(mf, roks.ROKS)
    if isinstance(mf, df_jk._DFHF):
        return (DFROKSGradients if ks else DFROHFGradients)(mf)
    return (ROKSGradients if ks else ROHFGradients)(mf)


def _method(self):
    return gradients(self)


# ROHF/ROKS inherit RHF's (asserting) Gradients upstream; electrolites.guards
# may have put its CPU fallback there.  Both are replaced explicitly.
for cls in (rohf.ROHF, roks.ROKS):
    attach(cls, 'nuc_grad_method', _method, replace_broken=True)
    attach(cls, 'Gradients', _method, replace_broken=True)

# density fitting puts `_DFHF` (whose Gradients sends ROHF to the RHF class)
# ahead in the MRO
_df_gradients_orig = df_jk._DFHF.Gradients


def _df_gradients(self):
    if self.istype('ROHF'):
        return gradients(self)
    return _df_gradients_orig(self)


_df_gradients.__wrapped__ = _df_gradients_orig
df_jk._DFHF.Gradients = _df_gradients
