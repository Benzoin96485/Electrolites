"""Time the GPU DF-MP2 gradient (``electrolites.dfmp2_grad``) against the GPU
DF-MP2 energy and the GPU SCF gradient on the same DF-RHF reference.

    python benchmarks/dfmp2_grad_bench.py --xyz benchmarks/molecules/x.xyz \\
        --basis def2-svp --frozen 0
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parser, build_mol                        # noqa: E402

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--frozen', type=int, default=-1,
                help='frozen occupied orbitals; -1 = the chemical core')
ap.add_argument('--dh', default='',
                help='a double hybrid (electrolites.dhdft) instead of RHF')
a = ap.parse_args()

import numpy as np                                             # noqa: E402
import cupy as cp                                              # noqa: E402
import electrolites                                            # noqa: E402
electrolites.patch_all()
electrolites.enable('dfmp2_grad')
from electrolites import dfmp2_grad                            # noqa: E402
from gpu4pyscf import scf                                      # noqa: E402
from gpu4pyscf.mp import dfmp2                                 # noqa: E402


def sync_time(fn):
    cp.cuda.Stream.null.synchronize()
    t = time.perf_counter()
    out = fn()
    cp.cuda.Stream.null.synchronize()
    return out, time.perf_counter() - t


mol = build_mol(a)
frozen = a.frozen
if frozen < 0:
    from pyscf.data import elements
    frozen = elements.chemcore(mol)
if a.dh:
    from electrolites.dhdft import DoubleHybrid
    dh = DoubleHybrid(mol, a.dh, frozen=frozen or None)
    mf = dh._scf
else:
    mf = scf.RHF(mol).density_fit()
mf.conv_tol = a.conv_tol
mf.conv_tol_grad = 1e-7
_, t_scf = sync_time(mf.kernel)
_, t_scfg = sync_time(lambda: mf.nuc_grad_method().kernel())
if a.dh:
    _, t_e = sync_time(dh.kernel)
    mp = dh.mp
    g, t_g = sync_time(lambda: dh.Gradients().kernel())
    e_tot = dh.e_tot
else:
    mp = dfmp2.DFMP2(mf, frozen=frozen or None)
    _, t_e = sync_time(mp.kernel)
    g, t_g = sync_time(lambda: mp.Gradients().kernel())
    e_tot = mp.e_tot
rec = dict(xyz=a.xyz, basis=a.basis, natm=mol.natm, nao=mol.nao,
           nocc=mol.nelectron // 2, frozen=frozen, naux=mp.auxmol.nao,
           scf_s=t_scf, scf_grad_s=t_scfg, mp2_energy_s=t_e,
           mp2_grad_s=t_g, stages=dict(dfmp2_grad.STATS),
           e_tot=float(e_tot), dh=a.dh, grad_norm=float(np.linalg.norm(g)),
           gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
print(f"[{os.path.basename(a.xyz)}/{a.basis}{'/' + a.dh if a.dh else ''}] nao={mol.nao} naux={rec['naux']} "
      f"frozen={frozen}  SCF {t_scf:.1f} s  SCF grad {t_scfg:.1f} s  "
      f"DF-MP2 E {t_e:.1f} s  DF-MP2 grad {t_g:.1f} s  "
      + '  '.join(f'{k} {v:.1f}' for k, v in dfmp2_grad.STATS.items()))
print('RESULT ' + json.dumps(rec))
