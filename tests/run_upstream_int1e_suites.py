"""GPU4PySCF's own int1e_grids and PCM test suites, with fast1e patched or not.

    python tests/run_upstream_int1e_suites.py --patch fast1e --src /path/to/gpu4pyscf

These are upstream's tests, not this repository's: they compare the GPU
integrals against PySCF's CPU intor() on def2-QZVPP and run PCM energies and
gradients end to end.  Running them patched is the cheapest statement that
`fast1e` did not change what GPU4PySCF computes anywhere it is reached from.
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--patch', default='')
ap.add_argument('--src', required=True,
                help='a GPU4PySCF source checkout matching the installed version')
a, rest = ap.parse_known_args()

import electrolites
if a.patch:
    electrolites.patch(a.patch.split(','))
print('patched:', electrolites.patched() or '(none)',
      '| gpu4pyscf', electrolites.gpu4pyscf_version(), file=sys.stderr)

import pytest
T = os.path.join(a.src, 'gpu4pyscf')
sys.exit(pytest.main(['-q', '--no-header',
                      f'{T}/gto/tests/test_int1e_grids.py',
                      f'{T}/gto/tests/test_int1e_grids_ip.py',
                      f'{T}/gto/tests/test_int1e_grids_ipip.py',
                      f'{T}/solvent/tests/test_pcm.py',
                      f'{T}/solvent/tests/test_pcm_grad.py'] + rest))
