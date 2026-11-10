"""DLPNO-CCSD(T) on the GPU for GPU4PySCF mean fields (GPU4PySCF has none).

    import electrolites
    electrolites.enable('dlpno')
    mf = gpu4pyscf.scf.RHF(mol).density_fit().run()        # or ROHF / UHF
    e = mf.DLPNO_CCSD_T(pno='normal', triples='t1').kernel()

* closed shell (RHF): ``electrolites.dlpno.rhf.DLPNO`` -- the algorithm of
  Psi4 1.11 (Jiang et al., JCP 161, 082502 (2024)): T1-dressed DLPNO-CCSD,
  semicanonical (T0) and iterative (T) (``triples='t0'`` / ``'t1'``);
* high-spin open shell (ROHF, or UHF through its quasi-restricted orbitals):
  ``electrolites.dlpno.uhf.UDLPNO`` -- the same thresholds with one set of
  PNOs per spatial pair (beta virtuals = PNOs + SOMOs), spin-orbital
  T1-dressed CCSD and spin-blocked (T0) / iterative (T).

``pno`` is one of 'loose', 'normal', 'tight', 'very_tight' (Psi4's
presets, ``electrolites.dlpno.options``); any threshold can be overridden
by keyword (``T_CUT_PNO=1e-8``).  ``triples=None`` stops after the CCSD.
Results: ``.e`` (component dictionary), ``.e_tot``, ``.e_corr``.
"""
import numpy as np

from gpu4pyscf.scf import hf

from electrolites.features import attach


class DLPNO_CCSD_T:
    def __init__(self, mf, pno='normal', triples='t0', auxbasis=None,
                 frozen_core=True, integral_backend='dense', **opts):
        self.mf = mf
        self.pno, self.triples, self.auxbasis = pno, triples, auxbasis
        self.frozen_core = frozen_core
        if integral_backend not in ('dense', 'local'):
            raise ValueError("integral_backend must be 'dense' or 'local'")
        self.integral_backend = integral_backend
        self.opts = opts
        self.solver = None
        self.e = {}

    def _cpu_mf(self):
        mf = self.mf
        return mf.to_cpu() if hasattr(mf, 'to_cpu') else mf

    def build_solver(self):
        mf = self._cpu_mf()
        occ = np.asarray(mf.mo_occ)
        kw = dict(pno=self.pno, triples=self.triples, auxbasis=self.auxbasis,
                  frozen_core=self.frozen_core, **self.opts)
        if self.integral_backend == 'local':
            if occ.ndim != 1 or np.any((occ > .5) & (occ < 1.5)):
                raise NotImplementedError('Local integral adjoints currently require closed-shell RHF')
            from electrolites.dlpno.local_gpu import LocalGPUDLPNO
            return LocalGPUDLPNO(mf, **kw)
        if occ.ndim == 2:
            from electrolites.dlpno.uhf import UDLPNO
            return UDLPNO(mf, reference='qro', **kw)
        if np.any((occ > .5) & (occ < 1.5)):
            from electrolites.dlpno.uhf import UDLPNO
            return UDLPNO(mf, **kw)
        from electrolites.dlpno.rhf import DLPNO
        return DLPNO(mf, **kw)

    def kernel(self):
        self.solver = s = self.build_solver()
        s.kernel()
        self.e = dict(s.e)
        return self.e_tot

    @property
    def e_corr(self):
        return self.e.get('corr')

    @property
    def e_tot(self):
        return self.e.get('tot')

    run = kernel

    def nuc_grad_method(self, method='auto', **kwargs):
        if method == 'auto':
            method = 'adjoint' if self.integral_backend == 'local' else 'direct'
        if method == 'adjoint':
            from electrolites.dlpno.local import LocalGradients
            return LocalGradients(self, **kwargs)
        if method != 'direct':
            raise ValueError("method must be 'auto', 'direct' or 'adjoint'")
        from electrolites.dlpno.gradient import Gradients
        return Gradients(self, **kwargs)

    Gradients = nuc_grad_method


def _dlpno(self, pno='normal', triples='t0', **kw):
    return DLPNO_CCSD_T(self, pno=pno, triples=triples, **kw)


attach(hf.SCF, 'DLPNO_CCSD_T', _dlpno)
