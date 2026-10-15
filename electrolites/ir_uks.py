"""Infrared intensities for UHF/UKS on the GPU
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` O3, the IR gate).

GPU4PySCF's ``properties.ir.eval_ir_freq_intensity`` asserts an RHF/RKS
reference.  This is the same driver for both: the Hessian object's own
first-order Fock matrices and CPHF response (GPU4PySCF's UHF/UKS Hessian
code, per spin), the dipole derivatives (atomic polar tensors)

    P[t, x, A] = Z_A delta_tx - sum_s occ_s (mu_t . D1_s[A, x]) * 2
                 - (d mu_t / d R_Ax) . D,

and the harmonic analysis of GPU4PySCF's (translations and rotations
projected out).  Equal to GPU4PySCF's for RHF/RKS and to pyscf-properties'
``infrared.uhf/uks`` for UHF/UKS (``tests/test_ir_uks.py``).

    freq, inten = electrolites.ir_uks.eval_ir_freq_intensity(mf, mf.Hessian())
"""
from functools import reduce

import numpy as np
import cupy as cp

from pyscf.data import elements, nist
from pyscf.hessian import thermo
from scipy.constants import physical_constants

LINDEP_THRESHOLD = 1e-7


def _modes(mol, hessian):
    hartree_kj = nist.HARTREE2J * 1e3
    unit2cm = ((hartree_kj * nist.AVOGADRO) ** .5 / (nist.BOHR * 1e-10)
               / (2 * np.pi * nist.LIGHT_SPEED_SI) * 1e-2)
    natm = mol.natm
    charges = mol.atom_charges()
    mass = np.array([elements.MASSES[charges[i]] for i in range(natm)])
    hm = np.asarray(hessian) / np.sqrt(mass)[:, None, None, None] \
        / np.sqrt(mass)[None, :, None, None]
    TR = thermo._get_TR(mass, mol.atom_coords())
    space = [TR[:3]]
    rotor = thermo._get_rotor_type(thermo.rotation_const(mass,
                                                         mol.atom_coords()))
    if rotor == 'LINEAR':
        space.append(TR[3:5])
    elif rotor != 'ATOM':
        space.append(TR[3:])
    space = np.vstack(space)
    q, _ = np.linalg.qr(space.T)
    P = np.eye(natm * 3) - q @ q.T
    w, v = np.linalg.eigh(P)
    bvec = v[:, w > LINDEP_THRESHOLD]
    h = reduce(np.dot, (bvec.T, hm.transpose(0, 2, 1, 3).reshape(3 * natm,
                                                                3 * natm),
                        bvec))
    e, mode = np.linalg.eigh(h)
    mode = bvec @ mode
    c = (mode.reshape(natm, 3, -1) / np.sqrt(mass)[:, None, None]
         ).reshape(3 * natm, -1)
    freq = np.sign(e) * np.sqrt(abs(e)) * unit2cm
    return freq, c


def dipole_derivatives(mf, hessobj):
    """Atomic polar tensors P[t, x, A] (dipole component t, displacement x
    of atom A), RHF/RKS or UHF/UKS."""
    from gpu4pyscf.lib import logger
    mol = mf.mol
    natm, nao = mol.natm, mol.nao
    log = logger.new_logger(hessobj, mol.verbose)
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    mo_energy = cp.asarray(mf.mo_energy)
    atmlst = range(natm)
    h1 = hessobj.make_h1(mo_coeff, mo_occ, None, atmlst)
    fx = hessobj.gen_vind(mo_coeff, mo_occ)
    mo1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1, fx, atmlst,
                            hessobj.max_memory, log)[0]
    restricted = mo_coeff.ndim == 2
    if restricted:
        spins = [(mo_coeff, mo_occ, cp.asarray(mo1), 4.)]
    else:
        spins = [(mo_coeff[s], mo_occ[s], cp.asarray(mo1[s]), 2.)
                 for s in (0, 1)]
    dm0 = cp.asarray(mf.make_rdm1())
    if dm0.ndim == 3:
        dm0 = dm0[0] + dm0[1]
    with mol.with_common_orig((0, 0, 0)):
        hmu = cp.asarray(mol.intor('int1e_r'))
        hmu11 = -cp.asarray(mol.intor('int1e_irp')).reshape(3, 3, nao, nao)
    out = cp.empty((3, 3, natm))
    for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:, 2:]):
        h11 = cp.zeros((3, 3, nao, nao))
        h11[:, :, :, p0:p1] += hmu11[:, :, :, p0:p1]
        h11[:, :, p0:p1] += hmu11[:, :, :, p0:p1].transpose(0, 1, 3, 2)
        t = -cp.einsum('xyuv,vu->xy', h11, dm0)
        for C, occ, m1, fac in spins:
            mocc = C[:, occ > 0]
            dm1 = C @ m1[ia] @ mocc.T                       # (3, nao, nao)
            t -= cp.einsum('xuv,yuv->xy', hmu, dm1) * fac
        t += mol.atom_charge(ia) * cp.eye(3)
        out[:, :, ia] = t
    return cp.asnumpy(out)


def eval_ir_freq_intensity(mf, hessobj):
    """(frequencies cm^-1, intensities km/mol), RHF/RKS or UHF/UKS."""
    hessian = hessobj.kernel()
    freq, c = _modes(mf.mol, hessian)
    apt = dipole_derivatives(mf, hessobj)
    alpha = physical_constants['fine-structure constant'][0]
    amu = physical_constants['atomic mass constant'][0]
    m_e = physical_constants['electron mass'][0]
    N_A = physical_constants['Avogadro constant'][0]
    a_0 = physical_constants['Bohr radius'][0]
    unit = alpha ** 2 * (1e-3 / amu) * m_e * N_A * np.pi * a_0 / 3
    inten = np.einsum('xym,myn->xn', apt, c.reshape(mf.mol.natm, 3, -1))
    return freq, (inten ** 2).sum(axis=0) * unit
