"""Check density-weighted bounds and tiled Coulomb builds on one GPU."""
from unittest.mock import patch

import cupy as cp
import numpy as np
from pyscf import gto
from gpu4pyscf.scf import j_engine


def test_block_bounds():
    from electrolites.fastj import _PairScreen

    rng = np.random.default_rng(519)
    weighted = rng.uniform(-85., 1., 1193).astype(np.float32)
    # Construct the native layout independently, including the padded tail.
    padded = np.pad(weighted, (0, -len(weighted) % 32), constant_values=-700.)
    levels = [padded.reshape(-1, width).max(axis=1) for width in (1, 2, 4, 8, 16, 32)]
    screen = _PairScreen(cp.asarray(weighted), cp.asarray(np.concatenate(levels)))
    for width in (16, 17, 32, 176, 480):
        for start in (0, 16, 32, 64, 480, 1008, 1192):
            expected = [weighted[i:i + width].max()
                        for i in range(start, len(weighted), width)]
            np.testing.assert_array_equal(screen.block_max(width, start).get(), expected)


def test_tiled_j():
    import electrolites
    electrolites.patch('fastj')
    from electrolites import fastj

    mol = gto.M(atom=[('O', (3*i, 0., 0.)) for i in range(5)],
                basis='def2-tzvpd', verbose=0)
    opt = j_engine._VHFOpt(mol, 1e-13).build()
    rng = np.random.default_rng(518)
    # Reuse one optimizer while its density support changes, including disjoint
    # support across the densities of a single multi-density launch.
    for count, local in ((1, True), (1, False), (5, True), (5, False)):
        dm = cp.asarray(rng.standard_normal((count, mol.nao, mol.nao)))
        dm += dm.transpose(0, 2, 1).copy()
        if local:
            slices = mol.aoslice_by_atom()[:, 2:]
            blocks = cp.zeros_like(dm)
            for index in range(count):
                lo, hi = slices[index]
                blocks[index, lo:hi, lo:hi] = dm[index, lo:hi, lo:hi]
            dm = blocks
        if count == 1:
            dm = dm[0]
        with patch.object(j_engine._VHFOpt, 'get_j', j_engine._VHFOpt._get_j_orig):
            ref = j_engine.get_j(mol, dm, vhfopt=opt)
        with patch.object(fastj, '_MAX_GRID_Y', 2):
            got = j_engine.get_j(mol, dm, vhfopt=opt)
        cp.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    test_block_bounds()
    test_tiled_j()
    print('forced small grid: single and multiple density J passed')
