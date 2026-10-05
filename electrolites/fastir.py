"""
IR intensities without the second CPHF solve (catalogue C10).

Importing this module replaces ``gpu4pyscf.properties.ir.
eval_ir_freq_intensity``.  GPU4PySCF's driver computes the Hessian
(``hessian_obj.kernel()``, which builds ``make_h1`` and solves the
``3 natm``-right-hand-side CPHF inside ``hess_elec``) and then, for the
dipole derivative, builds ``make_h1`` and solves **the same CPHF again**,
under its own ``# TODO: compact with hessian method, which can save one time
cphf solve`` (1.8.1 ``properties/ir.py:91-97``).  Measured at 41.8 % of a
42-atom IR run after ``patch_all()`` (``docs/KERNEL_TARGETS_1.8.1.md``
§10.2).

Here the Hessian's own ``solve_mo1`` result is kept: the driver wraps the
Hessian object's ``solve_mo1`` for the duration of ``kernel()`` (an instance
attribute, removed afterwards), and the dipole derivative uses that ``mo1``.
The two solves have identical inputs -- the same ``mo_energy``,
``mo_coeff``, ``mo_occ``, ``make_h1`` for every atom, ``gen_vind`` -- so this
is the same quantity, computed once.  If the Hessian path did not go through
``hessian_obj.solve_mo1`` (nothing in 1.8.1 does that, but a subclass
could), the driver falls back to the second solve, as upstream.

Everything after the solve is GPU4PySCF's code, copied from 1.8.1.
``FASTIR_OFF=1`` leaves the upstream driver in place.
``docs/README_fastir.md``.
"""
import os
from functools import reduce

import cupy
import numpy as np
from pyscf.data import elements, nist
from pyscf.hessian import thermo
from scipy.constants import physical_constants

from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.properties import ir as IR
from gpu4pyscf.scf.hf import RHF

LINDEP_THRESHOLD = IR.LINDEP_THRESHOLD
_OFF = bool(os.environ.get('FASTIR_OFF'))

#: 'reused' counts drivers that took the Hessian's mo1, 'resolved' those that
#: had to solve again
STATS = {'reused': 0, 'resolved': 0}


def stats():
    return dict(STATS)


def _hessian_keeping_mo1(hessian_obj):
    """``hessian_obj.kernel()``, plus the ``(mo1, mo_e1)`` its CPHF returned."""
    kept = []
    bound = hessian_obj.solve_mo1
    had_attr = 'solve_mo1' in hessian_obj.__dict__

    def solve_mo1(*args, **kwargs):
        out = bound(*args, **kwargs)
        kept.append(out)
        return out
    hessian_obj.solve_mo1 = solve_mo1
    try:
        hessian = hessian_obj.kernel()
    finally:
        if had_attr:
            hessian_obj.solve_mo1 = bound
        else:
            del hessian_obj.solve_mo1
    return hessian, (kept[-1] if len(kept) == 1 else None)


def eval_ir_freq_intensity(mf, hessian_obj):
    '''Main driver of infrared spectra intensity (GPU4PySCF 1.8.1's, with the
    Hessian's CPHF solution reused for the dipole derivative).

    Returns:
        frequency mode: in cm^-1
        infrared spectra intensity: in km/mol
    '''
    log = logger.new_logger(hessian_obj, mf.mol.verbose)
    assert isinstance(mf, RHF)
    hessian, kept = _hessian_keeping_mo1(hessian_obj)
    hartree_kj = nist.HARTREE2J*1e3
    unit2cm = ((hartree_kj * nist.AVOGADRO)**.5 / (nist.BOHR*1e-10)
               / (2*np.pi*nist.LIGHT_SPEED_SI) * 1e-2)
    natm = mf.mol.natm
    nao = mf.mol.nao
    dm0 = mf.make_rdm1()

    atom_charges = mf.mol.atom_charges()
    mass = cupy.array([elements.MASSES[atom_charges[i]] for i in range(natm)])
    hessian_mass = contract('ijkl,i->ijkl', cupy.array(hessian), 1/cupy.sqrt(mass))
    hessian_mass = contract('ijkl,j->ijkl', hessian_mass, 1/cupy.sqrt(mass))

    TR = thermo._get_TR(mass.get(), mf.mol.atom_coords())
    TRspace = []
    TRspace.append(TR[:3])

    rot_const = thermo.rotation_const(mass.get(), mf.mol.atom_coords())
    rotor_type = thermo._get_rotor_type(rot_const)
    if rotor_type == 'ATOM':
        pass
    elif rotor_type == 'LINEAR':  # linear molecule
        TRspace.append(TR[3:5])
    else:
        TRspace.append(TR[3:])

    if TRspace:
        TRspace = cupy.vstack(TRspace)
        q, r = cupy.linalg.qr(TRspace.T)
        P = cupy.eye(natm * 3) - q.dot(q.T)
        w, v = cupy.linalg.eigh(P)
        bvec = v[:,w > LINDEP_THRESHOLD]
        h = reduce(cupy.dot, (bvec.T, hessian_mass.transpose(0,2,1,3).reshape(3*natm,3*natm), bvec))
        e, mode = cupy.linalg.eigh(h)
        mode = bvec.dot(mode)

    c = contract('ixn,i->ixn', mode.reshape(natm, 3, -1),
                  1/np.sqrt(mass)).reshape(3*natm, -1)
    freq = cupy.sign(e)*cupy.sqrt(cupy.abs(e))*unit2cm

    mo_coeff = cupy.array(mf.mo_coeff)
    mo_occ = cupy.array(mf.mo_occ)
    mo_energy = cupy.array(mf.mo_energy)
    mocc = mo_coeff[:, mo_occ > 0]

    atmlst = range(natm)
    if kept is not None:
        # the Hessian's own solve: same mo_energy/mo_coeff/mo_occ, make_h1
        # for every atom and gen_vind as the second solve upstream does
        mo1, mo_e1 = kept
        STATS['reused'] += 1
    else:
        h1ao = hessian_obj.make_h1(mo_coeff, mo_occ, None, atmlst)
        fx = hessian_obj.gen_vind(mo_coeff, mo_occ)
        mo1, mo_e1 = hessian_obj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao,
                                           fx, atmlst, hessian_obj.max_memory, log)
        STATS['resolved'] += 1
    mo1 = cupy.asarray(mo1)
    assert mo1.shape[:2] == (natm, 3)

    tmp = cupy.empty((3, 3, natm))  # dipole moment, x,y,z
    aoslices = mf.mol.aoslice_by_atom()
    with mf.mol.with_common_orig((0, 0, 0)):
        hmuao = mf.mol.intor('int1e_r')  # mu
        hmuao11 = -mf.mol.intor('int1e_irp').reshape(3, 3, nao, nao)
    hmuao = cupy.array(hmuao)
    hmuao11 = cupy.array(hmuao11)
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h11ao = cupy.zeros((3, 3, nao, nao))

        h11ao[:, :, :, p0:p1] += hmuao11[:, :, :, p0:p1]
        h11ao[:, :, p0:p1] += hmuao11[:, :, :, p0:p1].transpose(0, 1, 3, 2)

        tmp0 = contract('ypi,vi->ypv', mo1[ia], mocc)  # nabla
        dm1 = contract('ypv,up->yuv', tmp0, mo_coeff)
        tmp[:, :, ia] = -contract('xuv,yuv->xy', hmuao, dm1) * 4
        tmp[:, :, ia] -= contract('xyuv,vu->xy', h11ao, dm0)
        tmp[:, :, ia] += mf.mol.atom_charge(ia)*cupy.eye(3)

    alpha = physical_constants["fine-structure constant"][0]
    amu = physical_constants["atomic mass constant"][0]
    m_e = physical_constants["electron mass"][0]
    N_A = physical_constants["Avogadro constant"][0]
    a_0 = physical_constants["Bohr radius"][0]
    unit_kmmol = alpha**2 * (1e-3 / amu) * m_e * N_A * np.pi * a_0 / 3

    intensity = contract('xym,myn->xn', tmp, c.reshape(natm, 3, -1))
    intensity = contract('xn,xn->n', intensity, intensity)
    intensity *= unit_kmmol

    return freq, intensity


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    IR._eval_ir_freq_intensity_orig = IR.eval_ir_freq_intensity
    IR.eval_ir_freq_intensity = eval_ir_freq_intensity
    _PATCHED = True


apply_patch()
