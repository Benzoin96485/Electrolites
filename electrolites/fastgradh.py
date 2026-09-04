"""The nuclear-attraction half of the core-Hamiltonian gradient, on the GPU.

``gpu4pyscf.grad.rhf.get_hcore`` builds ``(nabla i | h | j)`` with two PySCF
CPU integrals, ``int1e_ipkin`` and ``int1e_ipnuc``.  The kinetic one is
O(nbas^2); the nuclear-attraction one carries a sum over every nucleus, so it
is O(natm * nbas^2) and it is the whole of the cost: 13 % of a B3LYP/6-31G*
gradient on the 545-atom HcgC cluster, single-threaded on the host while the
GPU idles.

GPU4PySCF already has the integral -- ``gto.int3c1e_ip.int1e_grids_ip1`` with
the nuclei as the charge points and ``-Z`` as the charges is exactly
``int1e_ipnuc`` -- it is just not what the gradient calls.  This module points
``get_hcore`` at it.  Nothing else changes: the kinetic part, the ECP part and
the contraction with the density are GPU4PySCF's.
"""
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf.grad import rhf as grhf
from gpu4pyscf.gto.ecp import get_ecp_ip
from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip1
from gpu4pyscf.gto import int3c1e

# One entry, holding the Mole itself rather than its id: a gradient scanner
# walking a geometry creates and drops a Mole per step, and an id can be reused
# by the next one, which would silently hand back the previous geometry's
# screening data.
_OPT = [None, None]


def _intopt(mol, tol=1e-13):
    if _OPT[0] is not mol:
        opt = int3c1e.VHFOpt(mol)
        opt.build(tol, aosym=False)
        _OPT[0], _OPT[1] = mol, opt
    return _OPT[1]


def get_hcore(mf_grad, mol=None, exclude_ecp=False):
    """Drop-in replacement for gpu4pyscf.grad.rhf.get_hcore."""
    if mol is None:
        mol = mf_grad.mol
    if mol._pseudo:
        raise NotImplementedError('Nuclear gradients for GTH PP')
    h = cupy.asarray(mol.intor('int1e_ipkin', comp=3))
    charges = cupy.asarray(-mol.atom_charges(), dtype=np.float64)
    h += int1e_grids_ip1(mol, mol.atom_coords(), charges=charges,
                         intopt=_intopt(mol))
    if not exclude_ecp and len(mol._ecpbas) > 0:
        h += get_ecp_ip(mol).sum(axis=0)
    return -h


_PATCHED = False


def apply_patch():
    global _PATCHED
    if not _PATCHED:
        grhf._get_hcore_orig = grhf.GradientsBase.get_hcore
        grhf.GradientsBase.get_hcore = get_hcore
        _PATCHED = True


apply_patch()
