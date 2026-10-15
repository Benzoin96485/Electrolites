"""Cube files -- density, orbital, molecular electrostatic potential -- on
the GPU (`docs/FEATURE_TARGETS_PYSCF_2.13.md` T1).

``pyscf.tools.cubegen`` raises on CuPy arrays and runs on the host: AO
values by ``eval_gto`` per 8 000 points, and the MEP by a host three-centre
integral per 600 points (the slow one: minutes for a protein-sized box).
Same functions, same box and file (PySCF's ``Cube``), with the AO values
from GPU4PySCF's ``eval_ao`` and the electronic potential from its
density-contracted ``int1e_grids`` (one pass over the box).  Inputs may be
host or device arrays.

    from electrolites import cubegen_gpu
    cubegen_gpu.density(mol, 'rho.cube', gpu_mf.make_rdm1())
    cubegen_gpu.mep(mol, 'mep.cube', gpu_mf.make_rdm1())
"""
import numpy as np
import cupy as cp

from pyscf.tools.cubegen import Cube, RESOLUTION, BOX_MARGIN

BLK = 32768


def _ao(mol, coords):
    """AO values (ng, nao) in mol's own AO order (GPU4PySCF ``eval_ao``)."""
    from gpu4pyscf.dft import numint
    return numint.eval_ao(mol, cp.asarray(coords), deriv=0)


def _check(mol):
    from pyscf.pbc.gto import Cell
    if isinstance(mol, Cell):
        raise NotImplementedError('GPU cube files: molecules only '
                                  '(use pyscf.tools.cubegen for cells)')


def density(mol, outfile, dm, nx=80, ny=80, nz=80, resolution=RESOLUTION,
            margin=BOX_MARGIN):
    """Electron density on the cube grid (e/Bohr^3), written and returned."""
    _check(mol)
    cc = Cube(mol, nx, ny, nz, resolution, margin)
    coords = cc.get_coords()
    dm = cp.asarray(dm)
    if dm.ndim == 3:
        dm = dm[0] + dm[1]
    rho = cp.empty(len(coords))
    for p0 in range(0, len(coords), BLK):
        ao = _ao(mol, coords[p0:p0 + BLK])
        rho[p0:p0 + BLK] = ((ao @ dm) * ao).sum(axis=1)
    rho = cp.asnumpy(rho).reshape(cc.nx, cc.ny, cc.nz)
    cc.write(rho, outfile, comment='Electron density in real space (e/Bohr^3)')
    return rho


def orbital(mol, outfile, coeff, nx=80, ny=80, nz=80, resolution=RESOLUTION,
            margin=BOX_MARGIN):
    """One orbital's values on the cube grid, written and returned."""
    _check(mol)
    cc = Cube(mol, nx, ny, nz, resolution, margin)
    coords = cc.get_coords()
    c = cp.asarray(coeff)
    out = cp.empty(len(coords))
    for p0 in range(0, len(coords), BLK):
        out[p0:p0 + BLK] = _ao(mol, coords[p0:p0 + BLK]) @ c
    out = cp.asnumpy(out).reshape(cc.nx, cc.ny, cc.nz)
    cc.write(out, outfile, comment='Orbital value in real space (1/Bohr^3)')
    return out


def mep(mol, outfile, dm, nx=80, ny=80, nz=80, resolution=RESOLUTION,
        margin=BOX_MARGIN):
    """Molecular electrostatic potential on the cube grid, written and
    returned: sum_A Z_A / |r - R_A| - int rho(r') / |r - r'|."""
    from gpu4pyscf.gto.int3c1e import int1e_grids
    _check(mol)
    cc = Cube(mol, nx, ny, nz, resolution, margin)
    coords = cc.get_coords()
    dm = cp.asarray(dm)
    if dm.ndim == 3:
        dm = dm[0] + dm[1]
    g = cp.asarray(coords)
    vnuc = cp.zeros(len(coords))
    for z, r in zip(mol.atom_charges(), mol.atom_coords()):
        vnuc += z / cp.linalg.norm(g - cp.asarray(r), axis=1)
    vele = cp.asarray(int1e_grids(mol, coords, dm=dm)).ravel()
    out = cp.asnumpy(vnuc - vele).reshape(cc.nx, cc.ny, cc.nz)
    cc.write(out, outfile, 'Molecular electrostatic potential in real space')
    return out
