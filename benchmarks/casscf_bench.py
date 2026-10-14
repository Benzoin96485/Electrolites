"""GPU DF-CASSCF (`electrolites/casscf_gpu.py`) timed, with its macro-
iteration count and stage split, and optionally PySCF's DF-CASSCF from the
same orbitals in the same process for the energy gate.

The reference orbitals are the GPU (RO)HF's; the CPU-node anatomies of
FEATURE_TARGETS §11.3 started from PySCF's own (RO)HF, so the per-macro-
iteration times compare, and the energies compare where both reached the
same solution.

    python benchmarks/casscf_bench.py --xyz benchmarks/molecules/tmc_ni_dithiolene.xyz \
        --charge -1 --spin 1 --basis def2-svp --cas 10,9 [--nroots 5] [--cpu-ref]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parser, build_mol                          # noqa: E402

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--cas', default='10,10')
ap.add_argument('--nroots', type=int, default=1)
ap.add_argument('--max-cycle', type=int, default=150)
ap.add_argument('--cpu-ref', action='store_true')
ap.add_argument('--cpu-ref-macro', type=int, default=0,
                help='cap the CPU reference at N macro iterations (timing)')
ap.add_argument('--direct', action='store_true', help='exact integrals (no DF)')
ap.add_argument('--casci-check', action='store_true',
                help='CPU CASCI at the GPU final orbitals: same energy?')
ap.add_argument('--polish-macro', type=int, default=50)
ap.add_argument('--polish', action='store_true',
                help='CPU DF-CASSCF from the GPU solution: same stationary point?')
ap.add_argument('--max-macro', type=int, default=0)
ap.add_argument('--mc-conv-tol', type=float, default=1e-9)
ap.add_argument('--cprofile', default='')
a = ap.parse_args()

import cupy as cp                                               # noqa: E402
import electrolites                                             # noqa: E402
electrolites.patch_all()
from electrolites import casscf_gpu, fci_gpu                    # noqa: E402
from gpu4pyscf import scf                                       # noqa: E402

mol = build_mol(a)
norb, nelec = (int(x) for x in a.cas.split(','))
mf = (scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol))
mf.conv_tol = a.conv_tol
mf.max_cycle = a.max_cycle
t0 = time.perf_counter()
mf.kernel()
t1 = time.perf_counter()
mc = casscf_gpu.CASSCF(mf, norb, nelec, density_fit=not a.direct)
if a.nroots > 1:
    mc = mc.state_average_([1. / a.nroots] * a.nroots)
mc.conv_tol = a.mc_conv_tol
if a.max_macro:
    mc.max_cycle_macro = a.max_macro
macro = []
t2 = time.perf_counter()
if a.cprofile:
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()
mc.kernel()
if a.cprofile:
    pr.disable()
    pr.dump_stats(a.cprofile)
    pstats.Stats(a.cprofile).sort_stats('tottime').print_stats(30)
cp.cuda.Stream.null.synchronize()
t3 = time.perf_counter()
rec = dict(xyz=a.xyz, basis=a.basis, natm=mol.natm, nao=mol.nao,
           cas=[norb, nelec], nroots=a.nroots, scf_s=t1 - t0,
           scf_converged=bool(mf.converged), casscf_s=t3 - t2,
           macro=casscf_gpu.STATS['eris'],
           per_macro_s=(t3 - t2) / max(1, casscf_gpu.STATS['eris']),
           converged=bool(mc.converged), e_casscf=float(mc.e_tot),
           e_states=[float(x) for x in getattr(mc, 'e_states', [mc.e_tot])],
           stats=dict(casscf_gpu.STATS), fci=dict(fci_gpu.STATS),
           auxbasis=str(mc.with_df.auxbasis),
           gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
if a.cpu_ref:
    from pyscf import mcscf
    mfc = mf.to_cpu()
    cm = mcscf.CASSCF(mfc if a.direct else
                     mfc.density_fit(auxbasis=mc.with_df.auxbasis), norb, nelec)
    if a.nroots > 1:
        cm = cm.state_average_([1. / a.nroots] * a.nroots)
    cm.conv_tol = 1e-9
    if a.cpu_ref_macro:
        cm.max_cycle_macro = a.cpu_ref_macro
    nmac = set()
    cm.callback = lambda env: nmac.add(env['imacro'])
    t4 = time.perf_counter()
    cm.kernel(mfc.mo_coeff)
    rec['cpu_ref'] = dict(s=time.perf_counter() - t4, e=float(cm.e_tot),
                          converged=bool(cm.converged), macro=len(nmac),
                          per_macro_s=(time.perf_counter() - t4) / max(1, len(nmac)),
                          threads=os.environ.get('OMP_NUM_THREADS'))
    rec['dE'] = abs(rec['e_casscf'] - cm.e_tot)
if a.casci_check:
    from pyscf import mcscf
    mfc = mf.to_cpu()
    ci = mcscf.CASCI(mfc if a.direct else
                     mfc.density_fit(auxbasis=mc.with_df.auxbasis), norb, nelec)
    if a.nroots > 1:
        ci = ci.state_average_([1. / a.nroots] * a.nroots)
    ci.fcisolver.conv_tol = 1e-12
    ci.kernel(cp.asnumpy(mc.mo_coeff))
    # the GPU's CASCI at the same orbitals and CI tolerance: the CASSCF's
    # own e_tot carries its last CI solve's looser tolerance (1e-8)
    gi = casscf_gpu.CASCI(mf, norb, nelec, density_fit=not a.direct)
    if a.nroots > 1:
        gi = gi.state_average_([1. / a.nroots] * a.nroots)
    gi.fcisolver.conv_tol = 1e-12
    gi.kernel(mc.mo_coeff)
    rec['casci_check'] = dict(e=float(ci.e_tot), e_gpu=float(gi.e_tot),
                              dE=abs(gi.e_tot - ci.e_tot),
                              dE_casscf=abs(rec['e_casscf'] - ci.e_tot))
if a.polish:
    from pyscf import mcscf
    mfc = mf.to_cpu()
    cm = mcscf.CASSCF(mfc if a.direct else
                     mfc.density_fit(auxbasis=mc.with_df.auxbasis), norb, nelec)
    if a.nroots > 1:
        cm = cm.state_average_([1. / a.nroots] * a.nroots)
    cm.conv_tol = 1e-10
    cm.max_cycle_macro = a.polish_macro
    nmac = set()
    cm.callback = lambda env: nmac.add(env['imacro'])
    cm.kernel(cp.asnumpy(mc.mo_coeff), ci0=mc.ci)
    rec['polish'] = dict(e=float(cm.e_tot), converged=bool(cm.converged),
                         macro=len(nmac), dE=abs(rec['e_casscf'] - cm.e_tot),
                         e_states=[float(x) for x in
                                   getattr(cm, 'e_states', [cm.e_tot])])
print(f"[{os.path.basename(a.xyz)}/{a.basis} CAS{tuple(rec['cas'])} x{a.nroots}] "
      f"nao={mol.nao}  SCF {rec['scf_s']:.1f} s  CASSCF {rec['casscf_s']:.1f} s "
      f"({rec['macro']} macro, {rec['per_macro_s']:.2f} s each)  E={rec['e_casscf']!r}"
      + (f"  CPU-ref {rec['cpu_ref']['s']:.1f} s ({rec['cpu_ref']['macro']} macro, "
         f"{rec['cpu_ref']['per_macro_s']:.2f} s each) dE={rec['dE']:.1e}" if a.cpu_ref else '')
      + (f"  CASCI-check dE={rec['casci_check']['dE']:.1e}" if a.casci_check else '')
      + (f"  polish dE={rec['polish']['dE']:.1e} in {rec['polish']['macro']} macro"
         if a.polish else ''))
print('RESULT ' + json.dumps(rec))
