"""Guards for the places GPU4PySCF 1.8.1 runs and returns a wrong answer.

`docs/FEATURE_TARGETS_PYSCF_2.13.md` §3.2, measured by
`benchmarks/feature_gap_probe.py` against finite differences:

* **ROHF / ROKS Hessian** -- ``class ROHF(hf.RHF)`` (``scf/rohf.py:73``)
  inherits the ``Hessian`` that ``hessian/rhf.py:937`` attaches to
  ``scf.hf.RHF``, so the closed-shell formula runs on an open-shell
  wavefunction: 0.509 Eh/bohr^2 off on H2O+ / STO-3G, with no error (the UHF
  Hessian of the same doublet is right to 4e-7).  PySCF has no ROHF Hessian
  either, so this raises.
* **QM/MM Hessian** -- ``QMMMSCF`` defines a gradient but not a Hessian, so
  ``mf.Hessian()`` is the vacuum Hessian with the MM charges' second
  derivatives left out: 0.065 Eh/bohr^2 off with +-0.8 e at 3 bohr.  PySCF
  has none either, so this raises.

and, around them, the ROHF / ROKS entry points that only crashed:

* ``Gradients`` / ``nuc_grad_method`` -- the inherited RHF gradient asserts
  (``grad/rhf.py:424``), the density-fitted one likewise.  PySCF *has* an
  ROHF / ROKS gradient, so these go to it on the CPU, with a warning
  (``features.cpu_fallback``), until the GPU one exists (S3).
* ``TDA`` / ``TDHF`` / ``TDDFT`` -- a TypeError deep in the response
  function; PySCF has no ROHF TDDFT, so this raises with the reason.

Patched by ``electrolites.patch('guards')`` and by ``patch_all()``.  Each
guard is a plain method on the class, so ``mf.undo_qmmm().Hessian()`` -- the
vacuum Hessian, asked for explicitly -- still works.
"""
from gpu4pyscf.scf import rohf
from gpu4pyscf.dft import roks
from gpu4pyscf.df import df_jk
from gpu4pyscf.qmmm import itrf

from electrolites.features import cpu_fallback, not_available

_ROHF_HESS = ('GPU4PySCF 1.8.1 applies the closed-shell (RHF) Hessian to an '
              'ROHF/ROKS wavefunction and returns a wrong result, and PySCF '
              'has no ROHF/ROKS Hessian.  Use UHF/UKS for a Hessian '
              '(docs/FEATURE_TARGETS_PYSCF_2.13.md 3.2)')
_ROHF_TD = ('neither GPU4PySCF 1.8.1 nor PySCF implements TDA/TDDFT on an '
            'ROHF/ROKS reference; use UHF/UKS')
_ROHF_GRAD = 'GPU4PySCF 1.8.1 has no ROHF/ROKS gradient'
_QMMM_HESS = ('GPU4PySCF 1.8.1 has no QM/MM Hessian: mf.Hessian() would '
              'return the vacuum Hessian without the MM charges\' terms '
              '(docs/FEATURE_TARGETS_PYSCF_2.13.md 3.2).  Call '
              'mf.undo_qmmm().Hessian() if that is what is wanted')


def _rohf_gradients(self):
    import sys
    gpu = sys.modules.get('electrolites.rohf_grad')
    if gpu is not None:                 # the GPU gradient feature is enabled
        return gpu.gradients(self)
    return cpu_fallback(self, f'{type(self).__name__}.Gradients',
                        _ROHF_GRAD).Gradients()


for cls in (rohf.ROHF, roks.ROKS):
    cls.Hessian = not_available(f'{cls.__name__}.Hessian', _ROHF_HESS)
    cls.Gradients = _rohf_gradients
    cls.nuc_grad_method = _rohf_gradients
    for name in ('TDA', 'TDHF', 'TDDFT', 'CasidaTDDFT'):
        setattr(cls, name, not_available(f'{cls.__name__}.{name}', _ROHF_TD))

# density fitting puts `_DFHF` ahead of ROHF in the MRO, with its own
# Gradients / Hessian dispatch that sends an ROHF object to the RHF classes
_df_gradients_orig = df_jk._DFHF.Gradients
_df_hessian_orig = df_jk._DFHF.Hessian


def _df_gradients(self):
    if self.istype('ROHF'):
        return _rohf_gradients(self)
    return _df_gradients_orig(self)


def _df_hessian(self):
    if self.istype('ROHF'):
        raise NotImplementedError(f'{type(self).__name__}.Hessian: '
                                  f'{_ROHF_HESS}')
    return _df_hessian_orig(self)


_df_gradients.__wrapped__ = _df_gradients_orig
_df_hessian.__wrapped__ = _df_hessian_orig
df_jk._DFHF.Gradients = _df_gradients
df_jk._DFHF.Hessian = _df_hessian

itrf.QMMMSCF.Hessian = not_available('QMMMSCF.Hessian', _QMMM_HESS)
