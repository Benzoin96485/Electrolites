"""QM/MM's QM-nucleus--MM-charge terms on the device, computed once (F6 / F13).

``gpu4pyscf.qmmm.itrf`` evaluates the point-charge Coulomb interaction
between the QM nuclei and the MM charges with host NumPy on an
``n_mm x natm`` array, and does it

* in ``QMMMSCF.energy_nuc`` -- which ``scf/hf.py:energy_tot`` calls **every
  SCF cycle** although neither set of positions changes (17 % of a patched
  QM/MM SCF + gradient at 10^5 charges, ``docs/KERNEL_TARGETS_1.8.1.md``
  §12.3);
* in ``QMMMGrad.grad_nuc`` and ``grad_nuc_mm`` -- the same pair forces, once
  summed over the MM charges and once over the QM atoms.

Here the three are one CuPy evaluation (in blocks of MM charges, so the
working array stays bounded), cached on the object against the coordinates
and charges it was computed from: the energy for the SCF's cycles, and the
pair forces for both gradient methods.  The expressions are GPU4PySCF's; the
summation order is the device's, so the energy agrees with the host's to
round-off, not bitwise.

``FASTQMMM_OFF=1`` keeps GPU4PySCF's methods.
"""
import os

import numpy as np
import cupy as cp

from gpu4pyscf.qmmm import itrf as I

_OFF = bool(os.environ.get('FASTQMMM_OFF'))
#: MM charges per block of the device evaluation
BLOCK = 1 << 16
STATS = {'energy': 0, 'energy_cached': 0, 'forces': 0, 'forces_cached': 0}


def stats():
    return dict(STATS)


def _key(mol, mm_mol):
    return (mol.atom_coords(), mol.atom_charges(), mm_mol.atom_coords(),
            mm_mol.atom_charges())


def _same(a, b):
    return b is not None and all(
        x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b))


def _pair_terms(key, forces):
    """``sum Z_A q_C / r`` and, if ``forces``, the per-QM-atom and per-charge
    sums of ``Z_A q_C (R_A - C) / r^3``."""
    rq, zq, rc, qc = (cp.asarray(x, dtype=np.float64) for x in key)
    e = cp.zeros(())
    g_qm = cp.zeros((len(zq), 3)) if forces else None
    g_mm = cp.empty((len(qc), 3)) if forces else None
    for c0 in range(0, len(qc), BLOCK):
        c1 = min(len(qc), c0 + BLOCK)
        d = rq[None, :, :] - rc[c0:c1, None, :]          # (C, A, 3)
        inv_r = 1. / cp.sqrt(cp.einsum('cad,cad->ca', d, d))
        zz = zq[None, :] * qc[c0:c1, None] * inv_r       # Z_A q_C / r
        e += zz.sum()
        if forces:
            f = (zz * inv_r * inv_r)[:, :, None] * d     # Z q (R_A - C) / r^3
            g_qm += f.sum(axis=0)
            g_mm[c0:c1] = f.sum(axis=1)
    return (float(e), None if g_qm is None else g_qm.get(),
            None if g_mm is None else g_mm.get())


def energy_nuc(self):
    nuc = super(I.QMMMSCF, self).energy_nuc()
    assert self.mm_mol.charge_model == 'point'
    key = _key(self.mol, self.mm_mol)
    cache = self.__dict__.get('_fastqmmm_enuc')
    if cache is not None and _same(key, cache[0]):
        STATS['energy_cached'] += 1
        return nuc + cache[1]
    STATS['energy'] += 1
    e = _pair_terms(key, False)[0]
    # not in _keys: a cache, not an attribute anyone sets
    self.__dict__['_fastqmmm_enuc'] = (key, e)
    return nuc + e


def _forces(self, mol):
    assert self.base.mm_mol.charge_model == 'point'
    key = _key(mol, self.base.mm_mol)
    cache = self.__dict__.get('_fastqmmm_force')
    if cache is not None and _same(key, cache[0]):
        STATS['forces_cached'] += 1
        return cache[1], cache[2]
    STATS['forces'] += 1
    _, g_qm, g_mm = _pair_terms(key, True)
    self.__dict__['_fastqmmm_force'] = (key, g_qm, g_mm)
    return g_qm, g_mm


def grad_nuc(self, mol=None, atmlst=None):
    if mol is None:
        mol = self.mol
    g_qm = super(I.QMMMGrad, self).grad_nuc(mol, atmlst)
    return g_qm - _forces(self, mol)[0]


def grad_nuc_mm(self, mol=None):
    if mol is None:
        mol = self.mol
    return _forces(self, mol)[1].copy()


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    I.QMMMSCF._energy_nuc_orig = I.QMMMSCF.energy_nuc
    I.QMMMGrad._grad_nuc_orig = I.QMMMGrad.grad_nuc
    I.QMMMGrad._grad_nuc_mm_orig = I.QMMMGrad.grad_nuc_mm
    I.QMMMSCF.energy_nuc = energy_nuc
    I.QMMMGrad.grad_nuc = grad_nuc
    I.QMMMGrad.grad_nuc_mm = grad_nuc_mm
    _PATCHED = True


apply_patch()
