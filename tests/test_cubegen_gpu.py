"""Cube files on the GPU (``electrolites.cubegen_gpu``, FEATURE_TARGETS T1)
against ``pyscf.tools.cubegen`` on the same density / orbital and box.

    python tests/test_cubegen_gpu.py
"""
import os
import sys
import tempfile
import time

import numpy as np
from pyscf import gto
from pyscf.tools import cubegen
from gpu4pyscf import dft

from electrolites import cubegen_gpu

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


tmp = tempfile.mkdtemp()
for label, mol in (
        ('water/def2-SVP', gto.M(atom='O 0 0 0; H 0 -0.757 0.587; '
                                 'H 0 0.757 0.587', basis='def2-svp',
                                 verbose=0)),
        ('glycine/def2-TZVP', gto.M(
            atom='/global/homes/w/weiliang/vibeqc/Electrolites/benchmarks/'
            'molecules/glycine.xyz', basis='def2-tzvp', verbose=0))):
    mf = dft.RKS(mol, xc='pbe').run()
    dm = mf.make_rdm1()
    dmh = dm.get()
    mo = mf.mo_coeff.get()[:, mol.nelectron // 2 - 1]
    for what, g_fn, c_fn, arg_g, arg_c in (
            ('density', cubegen_gpu.density, cubegen.density, dm, dmh),
            ('HOMO', cubegen_gpu.orbital, cubegen.orbital, mo, mo),
            ('MEP', cubegen_gpu.mep, cubegen.mep, dm, dmh)):
        t = time.perf_counter()
        a = g_fn(mol, os.path.join(tmp, 'g.cube'), arg_g, nx=40, ny=40, nz=40)
        tg = time.perf_counter() - t
        t = time.perf_counter()
        b = c_fn(mol, os.path.join(tmp, 'c.cube'), arg_c, nx=40, ny=40, nz=40)
        tc = time.perf_counter() - t
        err = abs(a - b).max() / abs(b).max()
        same_header = (open(os.path.join(tmp, 'g.cube')).readlines()[2:6]
                       == open(os.path.join(tmp, 'c.cube')).readlines()[2:6])
        report(f'{label} {what}', err < 1e-10 and same_header,
               f'rel {err:.1e}  GPU {tg:.2f} s, CPU {tc:.2f} s')
print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
