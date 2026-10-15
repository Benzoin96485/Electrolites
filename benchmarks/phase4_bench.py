"""Phase 4 timings (FEATURE_TARGETS §15): one step of each new method, on
the GPU (default) or with PySCF on the CPU (``--cpu``), timed in isolation
after the reference it needs.

    python benchmarks/phase4_bench.py --what nevpt2 --cas 12,12 \\
        --xyz benchmarks/molecules/pbenzyne.xyz --basis def2-svp [--cpu]

what: nevpt2 | mcpdft | rpa | gw | sagrad | nac | casgrad
(casgrad: single-state DF-CASSCF gradient; ``--mo`` file: the GPU run
writes its converged orbitals there, the CPU run starts from them, so the
CPU CASSCF takes ~1 macro-iteration and only the gradient is compared)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parser, build_mol                        # noqa: E402

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--what', required=True,
                choices=('nevpt2', 'mcpdft', 'rpa', 'gw', 'sagrad', 'nac',
                         'casgrad'))
ap.add_argument('--mo', default=None)
ap.add_argument('--cpu', action='store_true')
ap.add_argument('--cas', default='10,10')
ap.add_argument('--nroots', type=int, default=2)
ap.add_argument('--ot', default='tPBE')
a = ap.parse_args()

import numpy as np                                             # noqa: E402

mol = build_mol(a)
norb, nelec = (int(x) for x in a.cas.split(','))
rec = dict(what=a.what, xyz=a.xyz, basis=a.basis, nao=mol.nao,
           natm=mol.natm, cas=[norb, nelec], cpu=a.cpu,
           threads=os.environ.get('OMP_NUM_THREADS'))


def timed(label, fn, warm=False):
    """Time fn; with ``warm`` on the GPU, run it once first (kernel
    compilation and plan caches), the cold time kept as label_cold."""
    if warm and not a.cpu:
        timed(label + '_cold', fn)
    if not a.cpu:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    t = time.perf_counter()
    out = fn()
    if not a.cpu:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    rec[label] = time.perf_counter() - t
    return out



if a.cpu:
    from pyscf import scf, dft, mcscf, mrpt
    hf = (scf.RHF if mol.spin == 0 else scf.ROHF)
    ks = dft.RKS
else:
    import cupy as cp                                          # noqa: F401
    import electrolites
    electrolites.patch_all()
    from gpu4pyscf import scf, dft
    from electrolites import casscf_gpu
    hf = (scf.RHF if mol.spin == 0 else scf.ROHF)
    ks = dft.RKS

if a.what in ('nevpt2', 'mcpdft', 'sagrad', 'nac', 'casgrad'):
    mf = hf(mol).density_fit()
    mf.conv_tol = 1e-9
    timed('scf_s', mf.kernel)
    if a.cpu:
        CASCI, CASSCF = mcscf.CASCI, mcscf.CASSCF
    else:
        def CASCI(m, n, e):
            return casscf_gpu.CASCI(m, n, e, density_fit=True)

        def CASSCF(m, n, e):
            return casscf_gpu.CASSCF(m, n, e, density_fit=True)
    if a.what in ('nevpt2', 'mcpdft'):
        mc = CASCI(mf, norb, nelec)
        timed('casci_s', mc.kernel)
        if a.what == 'nevpt2':
            if a.cpu:
                obj = mrpt.NEVPT(mc)
            else:
                from electrolites.nevpt2_gpu import NEVPT
                obj = NEVPT(mc)
            rec['e'] = float(timed('nevpt2_s', obj.kernel, True))
        else:
            from pyscf.mcpdft.mcpdft import energy_tot
            if a.cpu:
                from pyscf import mcpdft
                pd = mcpdft.CASCI(mf, a.ot, norb, nelec)
            else:
                from electrolites.mcpdft_gpu import MCPDFT
                pd = MCPDFT(mc, a.ot)
            mo = mc.mo_coeff
            mo = mo.get() if hasattr(mo, 'get') else np.asarray(mo)
            rec['e'] = float(timed('mcpdft_s', lambda: energy_tot(
                pd, mo_coeff=mo, ci=mc.ci)[0], True))
    elif a.what == 'casgrad':
        mc = CASSCF(mf, norb, nelec)
        mc.conv_tol = 1e-9
        mc.max_cycle_macro = 300
        mo0 = None
        if a.cpu and a.mo:
            mo0 = np.load(a.mo)
        timed('casscf_s', lambda: mc.kernel(mo0))
        if not a.cpu and a.mo:
            mo = mc.mo_coeff
            np.save(a.mo, mo.get() if hasattr(mo, 'get') else np.asarray(mo))
        rec['e'] = float(mc.e_tot)
        g = timed('grad_s', lambda: mc.nuc_grad_method().kernel(), True)
        rec['norm'] = float(np.linalg.norm(g))
    else:
        mc = CASSCF(mf, norb, nelec)
        if a.what == 'nac':
            mc.fix_spin_(ss=0)
        mc = mc.state_average_([1. / a.nroots] * a.nroots)
        mc.conv_tol = 1e-9
        timed('casscf_s', mc.kernel)
        if a.what == 'sagrad':
            g = timed('grad_s', lambda: mc.nuc_grad_method().kernel(state=0),
                      True)
        else:
            g = timed('nac_s', lambda: mc.nac_method().kernel(state=(0, 1)),
                      True)
        rec['norm'] = float(np.linalg.norm(g))
elif a.what == 'rpa':
    mf = ks(mol, xc='pbe').density_fit()
    mf.conv_tol = 1e-9
    timed('scf_s', mf.kernel)
    if a.cpu:
        from pyscf.gw import rpa
        r = rpa.RPA(mf)
    else:
        from electrolites.rpa_gpu import RPA
        r = RPA(mf)
    timed('rpa_s', lambda: (setattr(r, 'e_hf', None), r.kernel()), True)
    rec['e'] = float(r.e_corr)
elif a.what == 'gw':
    mf = ks(mol, xc='pbe').density_fit()
    mf.conv_tol = 1e-9
    timed('scf_s', mf.kernel)
    nocc = mol.nelectron // 2
    if a.cpu:
        from pyscf.gw import gw_ac
        g = gw_ac.GWAC(mf)
    else:
        from electrolites.gw_gpu import GWAC
        g = GWAC(mf)
    g.orbs = list(range(nocc - 3, nocc + 3))
    timed('gw_s', g.kernel, True)
    rec['homo'] = float(np.asarray(g.mo_energy)[nocc - 1])
side = 'CPU' if a.cpu else 'GPU'
print(f"[{a.what} {side} {os.path.basename(a.xyz)}/{a.basis} CAS{a.cas}] "
      + '  '.join(f'{k} {v:.2f}' for k, v in rec.items()
                  if k.endswith('_s')))
print('RESULT ' + json.dumps(rec, default=str))
