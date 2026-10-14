"""``mf.CASSCF`` / ``mf.CASCI`` on GPU4PySCF mean fields
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` M1-M5): the GPU CASSCF/CASCI of
``electrolites.casscf_gpu`` -- density-fitted for a DF mean field, exact
integrals otherwise, as PySCF -- bound where PySCF binds its own (GPU4PySCF's
SCF classes have neither).

    import electrolites
    electrolites.enable('casscf')
    mc = gpu_mf.CASSCF(10, 9).run()
"""
from gpu4pyscf.scf import hf

from electrolites.features import attach
from electrolites import casscf_gpu


def _casscf(self, ncas, nelecas, auxbasis=None, ncore=None, frozen=None,
            density_fit=None):
    return casscf_gpu.CASSCF(self, ncas, nelecas, auxbasis, ncore, frozen,
                             density_fit)


def _casci(self, ncas, nelecas, auxbasis=None, ncore=None, density_fit=None):
    return casscf_gpu.CASCI(self, ncas, nelecas, auxbasis, ncore, density_fit)


attach(hf.SCF, 'CASSCF', _casscf)
attach(hf.SCF, 'CASCI', _casci)
