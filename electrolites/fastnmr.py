"""
The NMR shielding's XC term with the GIAO AOs on the GPU (catalogue C7).

Importing this module replaces ``gpu4pyscf.properties.shielding.nr_rks``.
GPU4PySCF 1.8.1's function walks the grid blocks on the GPU and, in each
block, copies the grid coordinates to the host and evaluates the GIAO AO
derivatives ``GTOval_ig`` (3 components) and, for a GGA, ``GTOval_ipig``
(9 components) with libcint on the CPU **over all ``nao`` AOs**, copies them
back and keeps the block's ``nsub`` columns (``shielding.py:91, 101-102``).
That was 72-78 % of a patched NMR run (``KERNEL_TARGETS_1.8.1.md`` §10.4).

The GIAO factors need no new integrals.  Verified against libcint
(``docs/README_fastnmr.md`` §1):

    GTOval_ig[d]      = 1/2 (R_mu x r)_d  phi_mu
    GTOval_ipig[x, d] = 1/2 [ (R_mu x r)_d d_x phi_mu + eps_{dax} R_mu,a phi_mu ]

with ``R_mu`` the AO's centre and ``r`` the grid point, so both are built on
the GPU from the ``ao`` the block loop already holds, for the block's AOs
only.  The contraction is regrouped by the AO factor it ends on, so the
three field directions and all four AO components of a block are one GEMM:

    V_d = [L0_d, L1_d, L2_d, L3_d] . [phi, d_x phi, d_y phi, d_z phi]^T
    L0_d = 1/2 [ C_d (w0 phi + sum_x w_x d_x phi) + (R x w)_d phi ]
    Lx_d = 1/2 C_d w_x phi,          C_d = (R_mu x r)_d

which is upstream's two contractions per direction, reassociated.  The
result agrees with GPU4PySCF's to round-off (``tests/test_nmr.py``).

``FASTNMR_OFF=1`` leaves upstream's function.
"""
import os

import cupy
import numpy

from gpu4pyscf.dft import numint
from gpu4pyscf.lib.cupy_helper import add_sparse, sandwich_dot
from gpu4pyscf.properties import shielding as SH

_OFF = bool(os.environ.get('FASTNMR_OFF'))
STATS = {'calls': 0, 'blocks': 0}


def stats():
    return dict(STATS)


def _ao_centres(mol):
    from pyscf.gto.mole import ATOM_OF
    ao_loc = mol.ao_loc_nr()
    atm = numpy.repeat(mol._bas[:, ATOM_OF], ao_loc[1:] - ao_loc[:-1])
    return cupy.asarray(mol.atom_coords()[atm])               # (nao, 3)


def _cross(a, b):
    """(a x b) for a (n, 3) and b (3, g): returns (3, n, g)."""
    return cupy.stack([
        a[:, 1:2] * b[2] - a[:, 2:3] * b[1],
        a[:, 2:3] * b[0] - a[:, 0:1] * b[2],
        a[:, 0:1] * b[1] - a[:, 1:2] * b[0]])


def nr_rks(ni, mol, grids, xc_code, dms):
    xctype = ni._xc_type(xc_code)
    if xctype not in ('LDA', 'GGA', 'HF'):
        return SH._nr_rks_orig(ni, mol, grids, xc_code, dms)
    STATS['calls'] += 1
    mo_coeff = getattr(dms, 'mo_coeff', None)
    mo_occ = getattr(dms, 'mo_occ', None)

    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol

    coeff = cupy.asarray(opt.coeff)
    nao, nao0 = coeff.shape
    mo_coeff = coeff.dot(mo_coeff)
    vmat = cupy.zeros((3, nao, nao))
    if xctype == 'HF':
        return vmat[:, :nao0, :nao0] * 0
    ao_deriv = 0 if xctype == 'LDA' else 1
    centres = _ao_centres(_sorted_mol)

    for ao, index, weight, coords in ni.block_loop(_sorted_mol, grids, nao,
                                                    ao_deriv):
        STATS['blocks'] += 1
        mo_coeff_mask = mo_coeff[index, :]
        rho = numint.eval_rho2(_sorted_mol, ao, mo_coeff_mask, mo_occ, None,
                               xctype)
        vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[1]
        R = centres[index]                                    # (nsub, 3)
        r = coords.T                                          # (3, ng)
        C = _cross(R, r)                                      # (3, nsub, ng)
        nsub, ng = R.shape[0], r.shape[1]
        if xctype == 'LDA':
            wv = weight * vxc[0]
            L = (.5 * C) * (ao * wv)                          # (3, nsub, ng)
            v = L.reshape(3 * nsub, ng) @ ao.T
        else:
            wv = vxc * weight                                 # (4, ng)
            ao0 = ao[0]
            fw = ao0 * wv[0] + (ao[1:4] * wv[1:4, None, :]).sum(axis=0)
            L = cupy.empty((3, nsub, 4, ng))
            L[:, :, 0] = C * fw
            L[:, :, 0] += _cross(R, wv[1:4]) * ao0
            for x in range(3):
                L[:, :, 1 + x] = C * (ao0 * wv[1 + x])
            L *= .5
            B = cupy.ascontiguousarray(ao[0:4].transpose(0, 2, 1)
                                       ).reshape(4 * ng, nsub)
            v = L.reshape(3 * nsub, 4 * ng) @ B
        v = v.reshape(3, nsub, nsub)
        for d in range(3):
            add_sparse(vmat[d], cupy.ascontiguousarray(v[d]), index)
        ao = L = v = C = None

    vmat = sandwich_dot(vmat, coeff)
    if numint.FREE_CUPY_CACHE:
        cupy.get_default_memory_pool().free_all_blocks()
    return (vmat - vmat.transpose(0, 2, 1))


def get_vxc(mf, dm0):
    """``shielding.get_vxc`` with the GIAO two-electron term on the GPU
    (``electrolites.giao_jk``: shifted-basis J/K instead of the host
    ``int2e_ig1``; K only for hybrids).  ``FASTNMR_HOST2E=1`` keeps the
    host term (J-only for pure functionals)."""
    ni = mf._numint
    mol = mf.mol
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    vxc = SH.nr_rks(ni, mol, mf.grids, mf.xc, mf.make_rdm1())
    if os.environ.get('FASTNMR_HOST2E'):
        if hybrid:
            return SH._get_vxc_orig(mf, dm0)
        from pyscf.scf import _vhf
        vj = _vhf.direct_mapdm(mol._add_suffix('int2e_ig1'), 'a4ij',
                               'lk->s1ij', cupy.asnumpy(dm0), 3, mol._atm,
                               mol._bas, mol._env)
        return vxc - cupy.asarray(vj)
    from electrolites import giao_jk
    vj, vk = giao_jk.get_jk(mol, dm0, with_k=hybrid)
    if not hybrid:
        return vxc + vj
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    assert omega == 0
    return vxc + vj - vk * hyb * .5


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    SH._nr_rks_orig = SH.nr_rks
    # `get_vxc` calls the module-level name
    SH.nr_rks = nr_rks
    SH._get_vxc_orig = SH.get_vxc
    SH.get_vxc = get_vxc
    _PATCHED = True


apply_patch()
