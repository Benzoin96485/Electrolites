"""S1's gate (`docs/FEATURE_TARGETS_PYSCF_2.13.md` §11.1, §14): what one
seminumerical (COSX) exchange build would cost on the GPU, built from the
kernel GPU4PySCF already has, against the exact exchange build it would
replace (``fastk`` under ``patch_all()``).

COSX:  K_mn = sum_g w_g phi_mg sum_l A_nl(g) F_lg,  F = D phi,
A_nl(g) = (n | 1/|r - g| | l).  Every build needs A over all grid points;
the GPU kernel for it is ``GINTfill_int3c1e`` (``gpu4pyscf/gto/int3c1e.py``,
the PCM/QM-MM one).  This times, per build, on PySCF's SGX grids:

* the A evaluation (the kernel only, device-resident, in grid slices);
* the contraction G_n(g) = sum_l A_nl(g) F_lg (a batched GEMV, memory-bound)
  and K += X G^T;

with no screening -- the lower bound of an unscreened sn-K -- and reports
how far that K is from the exact one (the COSX error, a sanity check of the
contraction).  Then the exact K build.

    python benchmarks/snk_gate.py --xyz benchmarks/molecules/penicillinV.xyz \\
        --basis def2-tzvpd --level 2
"""
import json
import os
import sys
import time
import ctypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parser, build_mol                        # noqa: E402

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--level', type=int, default=2)
ap.add_argument('--opt-grids', action='store_true',
                help="PySCF's SGX pruned grids (use_opt_grids)")
a = ap.parse_args()

import numpy as np                                             # noqa: E402
import cupy as cp                                              # noqa: E402
import electrolites                                            # noqa: E402
electrolites.patch_all()
from pyscf import lib                                          # noqa: E402
from pyscf.sgx import sgx_jk                                   # noqa: E402
from gpu4pyscf import scf                                      # noqa: E402
from gpu4pyscf.dft import numint                               # noqa: E402
from gpu4pyscf.gto.int3c1e import VHFOpt, libgint              # noqa: E402
from gpu4pyscf.lib.cupy_helper import get_avail_mem            # noqa: E402


def sync():
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter()


mol = build_mol(a)
mf = (scf.UHF if mol.spin else scf.RHF)(mol).density_fit()
mf.conv_tol = 1e-8
mf.kernel()
dm = cp.asarray(mf.make_rdm1())
if dm.ndim == 3:
    dm = dm[0]                       # one spin's K: the same cost

grids = sgx_jk.get_gridss(mol, level=a.level, use_opt_grids=a.opt_grids)
coords = cp.asarray(grids.coords)
weights = cp.asarray(grids.weights)
ng = coords.shape[0]
nao = mol.nao

intopt = VHFOpt(mol)
intopt.build(1e-13, aosym=True)
omega = 0.


def a_slice(p0, p1):
    """A[g, n, l] for grid points p0:p1, device-resident (the body of
    gpu4pyscf.gto.int3c1e.get_int3c1e without the host copy)."""
    out = cp.zeros([p1 - p0, nao, nao])
    gs = coords[p0:p1]
    for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
        if len(log_q_ij) == 0:
            continue
        cpi = intopt.cp_idx[cp_ij_id]
        cpj = intopt.cp_jdx[cp_ij_id]
        li = intopt.angular[cpi]
        lj = intopt.angular[cpj]
        i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi + 1]
        j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj + 1]
        ni, nj = i1 - i0, j1 - j0
        ao_offsets = np.array([i0, j0], dtype=np.int32)
        strides = np.array([ni, ni * nj], dtype=np.int32)
        sl = cp.zeros([p1 - p0, nj, ni])
        bins = np.array([0, len(log_q_ij)], dtype=np.int32)
        err = libgint.GINTfill_int3c1e(
            ctypes.cast(cp.cuda.get_current_stream().ptr, ctypes.c_void_p),
            intopt.bpcache, ctypes.cast(gs.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(), ctypes.c_int(p1 - p0),
            ctypes.cast(sl.data.ptr, ctypes.c_void_p),
            strides.ctypes.data_as(ctypes.c_void_p),
            ao_offsets.ctypes.data_as(ctypes.c_void_p),
            bins.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1),
            ctypes.c_int(cp_ij_id), ctypes.c_double(omega))
        if err != 0:
            raise RuntimeError('GINTfill_int3c1e failed')
        i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi + 1]
        j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj + 1]
        if not mol.cart:
            from gpu4pyscf.lib.cupy_helper import cart2sph
            sl = cart2sph(sl, axis=1, ang=lj)
            sl = cart2sph(sl, axis=2, ang=li)
        out[:, j0:j1, i0:i1] = sl
    row, col = np.tril_indices(nao)
    out[:, row, col] = out[:, col, row]
    return intopt.unsort_orbitals(out, axis=[1, 2])


free = get_avail_mem()
step = max(1, int(0.25 * free // (nao * nao * 8 * 2)))
t_a = t_c = 0.
K = cp.zeros((nao, nao))
t0 = sync()
for p0 in range(0, ng, step):
    p1 = min(ng, p0 + step)
    ta = sync()
    A = a_slice(p0, p1)
    tb = sync()
    ao = numint.eval_ao(mol, coords[p0:p1], deriv=0)
    if ao.shape[0] != nao:
        ao = ao.T                                           # (nao, g)
    F = dm @ ao                                             # (l, g)
    G = cp.matmul(A, F.T[:, :, None])[:, :, 0]              # (g, n)
    K += (ao * weights[p0:p1]) @ G
    tc = sync()
    t_a += tb - ta
    t_c += tc - tb
    A = None
t_snk = sync() - t0
K = .5 * (K + K.T)

# the exact K it would replace (fastk under patch_all)
mfx = scf.RHF(mol)
vk = mfx.get_k(mol, dm)
t0 = sync()
for _ in range(2):
    vk = mfx.get_k(mol, dm)
t_k = (sync() - t0) / 2
err = float(abs(K - vk).max())
rec = dict(xyz=a.xyz, basis=a.basis, nao=nao, ngrids=ng, level=a.level,
           opt_grids=a.opt_grids, snk_s=t_snk, snk_A_s=t_a,
           snk_contract_s=t_c, exact_k_s=t_k, max_dK_vs_exact=err,
           integrals=float(ng) * nao * (nao + 1) / 2,
           gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
print(f"[{os.path.basename(a.xyz)}/{a.basis} sgx level {a.level}"
      f"{' opt' if a.opt_grids else ''}] nao={nao} ngrids={ng}  "
      f"sn-K build {t_snk:.2f} s (A {t_a:.2f} s, contraction {t_c:.2f} s)  "
      f"exact K {t_k:.2f} s  |K_sn - K_exact| {err:.1e}")
print('RESULT ' + json.dumps(rec))
