"""GPU (T) (`electrolites/ccsd_t.py`) on fastccsd's amplitudes, timed.

Runs the GPU RHF and ``fastccsd``'s CCSD at ``--cc-conv-tol`` (1e-9, as
`cpu_baseline.py`), then the (T) with its two stages timed (MO integrals,
triples), and prints e_corr and e_t in full for the gate against the CPU
record of the same system (`docs/CPU_BASELINE.md` §5: CCSD 1e-7, and the
(T) difference that follows from it).

    python benchmarks/ccsd_t_bench.py --xyz benchmarks/molecules/glycine.xyz --basis def2-tzvp
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parser, build_mol                        # noqa: E402

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--cc-conv-tol', type=float, default=1e-9)
ap.add_argument('--frozen', type=int, default=0)
a = ap.parse_args()

import cupy as cp                                            # noqa: E402
import electrolites                                          # noqa: E402
electrolites.patch_all()
electrolites.enable('ccsd_t')
from electrolites import ccsd_t as ct                        # noqa: E402
from gpu4pyscf import scf                                    # noqa: E402
from gpu4pyscf.cc import ccsd_incore                         # noqa: E402

mol = build_mol(a)
t0 = time.perf_counter()
mf = scf.RHF(mol)
mf.conv_tol = a.conv_tol
mf.kernel()
t1 = time.perf_counter()
cc = ccsd_incore.CCSD(mf, frozen=a.frozen or None)
cc.conv_tol = a.cc_conv_tol
cc.kernel()
cp.cuda.Stream.null.synchronize()
t2 = time.perf_counter()
et = cc.ccsd_t()
cp.cuda.Stream.null.synchronize()
t3 = time.perf_counter()
rec = dict(xyz=a.xyz, basis=a.basis, natm=mol.natm, nao=mol.nao,
           nocc=int(cc.nocc), nvir=int(cc.nmo - cc.nocc),
           scf_s=t1 - t0, ccsd_s=t2 - t1, t_s=t3 - t2,
           t_integrals_s=ct.STATS['integrals_s'],
           t_triples_s=ct.STATS['triples_s'], prof=dict(ct.PROF),
           e_scf=float(mf.e_tot), e_corr=float(cc.e_corr), e_t=float(et),
           gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
print(f"[{os.path.basename(a.xyz)}/{a.basis}] nao={mol.nao} o={rec['nocc']} "
      f"v={rec['nvir']}  SCF {rec['scf_s']:.1f} s  CCSD {rec['ccsd_s']:.1f} s  "
      f"(T) {rec['t_s']:.1f} s (integrals {rec['t_integrals_s']:.1f}, "
      f"triples {rec['t_triples_s']:.1f})")
print(f"  e_corr = {rec['e_corr']!r}  e_t = {rec['e_t']!r}  prof {ct.PROF}")
print('RESULT ' + json.dumps(rec))
