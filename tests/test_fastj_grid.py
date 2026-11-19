"""Exercise the physical grid overflow on small real J matrices on one GPU."""
from unittest.mock import patch

import cupy as cp
import numpy as np
from pyscf import gto
from gpu4pyscf.scf import j_engine


def test_tiled_j():
    import electrolites
    electrolites.patch('fastj')
    from electrolites import fastj

    mol = gto.M(atom=[('O', (3*i, 0., 0.)) for i in range(5)],
                basis='def2-tzvpd', verbose=0)
    opt = j_engine._VHFOpt(mol, 1e-13).build()
    rng = np.random.default_rng(518)
    for count in (1, 5):
        dm = cp.asarray(rng.standard_normal((count, mol.nao, mol.nao)))
        dm += dm.transpose(0, 2, 1).copy()
        if count == 1:
            dm = dm[0]
        with patch.object(j_engine._VHFOpt, 'get_j', j_engine._VHFOpt._get_j_orig):
            ref = j_engine.get_j(mol, dm, vhfopt=opt)
        with patch.object(fastj, '_MAX_GRID_Y', 2):
            got = j_engine.get_j(mol, dm, vhfopt=opt)
        cp.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    test_tiled_j()
    print('forced small grid: single and multiple density J passed')
