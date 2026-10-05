"""Where does the wall clock of a feature no profile here has reached go?

`docs/KERNEL_TARGETS_1.8.1.md` 2.5 lists the GPU4PySCF 1.8.1 features that
had never been profiled -- TDDFT gradients and NACs, IR, NMR, the PCM Hessian,
the restricted meta-GGA SCF, HSE06, the UKS range-separated gradient, DF-MP2,
CCSD, k-point hybrids -- and its 8 the measurements owed.  This is the one
harness for all of them, so that every row of every table is taken the same
way:

* every stage named in the feature's plan is wrapped with a
  `cudaDeviceSynchronize()` on both sides and timed, with its call count, and
  rows prefixed `..` are sub-intervals of their parent and overlap it;
* every Electrolites entry point the run *actually reached* is counted (the
  `coverage` block), so that no share is ever claimed for a module that is not
  on the path -- an entry point is counted only if the object at its target is
  Electrolites' own (`__module__` starts with `electrolites`), i.e. only where
  the patch replaced something;
* `--patch ''` is GPU4PySCF 1.8.1 as shipped and `--patch all` is
  `patch_all()`; the two are separate processes, run back to back.

    python benchmarks/path_anatomy.py --feature tdgrad \
        --xyz benchmarks/molecules/penicillinV.xyz --basis def2-svp --xc b3lyp \
        [--patch all]

Features: tdgrad, ir, scf, nmr, pcmhess, hess, grad, dfmp2, ccsd (§10 of
the document), and forcenac, sftda, smdhess, ecp, gks, eda, c6, ris, pm6
(§11).  A stage reached from several timed callers is also attributed to each
caller (`by caller`).  The periodic path is `kpt_anatomy.py`.
"""
import json
import os
import time
from collections import OrderedDict
import importlib

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--feature', required=True,
                choices=('tdgrad', 'ir', 'scf', 'nmr', 'pcmhess', 'hess',
                         'grad', 'dfmp2', 'ccsd', 'forcenac', 'sftda',
                         'smdhess', 'ecp', 'gks', 'eda', 'c6', 'ris', 'pm6',
                         'raman', 'dftdgrad', 'qmmm', 'x2c', 'fci', 'mp2',
                         'disphess', 'analysis'))
ap.add_argument('--nstates', type=int, default=5)
ap.add_argument('--tda', action='store_true')
ap.add_argument('--state', type=int, default=1,
                help='the excited state whose gradient is taken')
ap.add_argument('--nac', default='1,2',
                help="the state pair of the NAC, '' for none")
ap.add_argument('--solvent', default='IEF-PCM')
ap.add_argument('--density-fit', action='store_true')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--max-cycle', type=int, default=50)
ap.add_argument('--cphf-max-cycle', type=int, default=0,
                help="the TDDFT gradient's Z-vector iteration cap (0: "
                     "GPU4PySCF's default)")
ap.add_argument('--conv-tol-cpscf', type=float, default=1e-6)
ap.add_argument('--first-call', action='store_true',
                help="time `get_hcore` (and the gradient's `get_hcore`) on a "
                     'fresh process before anything else is built, then '
                     'again: the first-call/JIT question of R2')
ap.add_argument('--grid-response', action='store_true',
                help="the Hessian's grid-response terms (B9), off by default")
ap.add_argument('--multistate', action='store_true',
                help='forcenac: also run the batched (experimental, singlet-'
                     'only) NAC_multistates on the same pairs')
ap.add_argument('--frag2', default='',
                help='eda: the second fragment (the first is --xyz)')
ap.add_argument('--collinear', default='mcol')
ap.add_argument('--nmm', type=int, default=10000,
                help='qmmm: the number of MM point charges')
ap.add_argument('--cas', default='12,12', help="fci: 'norb,nelec'")
ap.add_argument('--memsample', action='store_true',
                help='sample the device free memory every 20 ms and report '
                     'the peak in use (GPU4PySCF allocates outside the CuPy '
                     "pool, so the pool's counters do not see it)")
ap.add_argument('--cprofile', default='',
                help='write a cProfile of the feature to this file and print '
                     'the 40 functions with the most self time')
ap.add_argument('--wrap', action='append', default=[],
                help="extra 'module:Attr.path:label' rows")
a = ap.parse_args()
if a.patch == 'none':
    a.patch = ''
active = apply_patch(a.patch)

import cupy as cp                                           # noqa: E402
import numpy as np                                          # noqa: E402
import gpu4pyscf                                            # noqa: E402
import electrolites                                         # noqa: E402

_sync = cp.cuda.Stream.null.synchronize
T = OrderedDict()
N = OrderedDict()
NDM = OrderedDict()
COV = OrderedDict()
# (label, enclosing timed label) -> seconds: which parent a shared stage's
# time was spent under, e.g. a kernel called from three different drivers
TP = OrderedDict()
STACK = []


def _resolve(path):
    mod, attr = path.split(':')
    obj = importlib.import_module(mod)
    parts = attr.split('.')
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def _ndm(x):
    shape = getattr(x, 'shape', None)
    if shape is None:
        return len(x) if isinstance(x, (list, tuple)) else 1
    return 1 if len(shape) == 2 else int(np.prod(shape[:-2]))


def wrap(path, label, ndm_arg=None):
    """Time `path` ('module:Class.attr') under `label`, synchronised."""
    if path.startswith('electrolites.'):
        # importing an Electrolites module patches GPU4PySCF, so a row on
        # one of its own names is only wrapped when that module is active
        if path.split(':')[0].split('.')[1] not in active:
            return
    try:
        holder, name = _resolve(path)
        orig = getattr(holder, name)
    except (AttributeError, ImportError) as err:
        print(f'# cannot wrap {path}: {err}', flush=True)
        return
    T.setdefault(label, 0.0)
    N.setdefault(label, 0)

    def timed(*args, **kwargs):
        _sync()
        parent = STACK[-1] if STACK else None
        STACK.append(label)
        t = time.perf_counter()
        try:
            out = orig(*args, **kwargs)
            _sync()
        finally:
            STACK.pop()
        dt = time.perf_counter() - t
        T[label] += dt
        N[label] += 1
        TP[(label, parent)] = TP.get((label, parent), 0.0) + dt
        if ndm_arg is not None and len(args) > ndm_arg:
            NDM.setdefault(label, set()).add(_ndm(args[ndm_arg]))
        return out
    timed.__name__ = getattr(orig, '__name__', name)
    timed.__wrapped__ = orig
    setattr(holder, name, timed)


def count_coverage():
    """Count calls into every Electrolites replacement that is in place."""
    for mod in active:
        tg = electrolites._TARGETS.get(mod)
        if tg is None:
            continue
        if isinstance(tg[0], str):
            tg = (tg,)
        for modpath, attr in tg:
            try:
                holder, name = _resolve(f'{modpath}:{attr}')
                fn = getattr(holder, name)
            except (AttributeError, ImportError):
                continue
            if not str(getattr(fn, '__module__', '')).startswith('electrolites'):
                continue                    # reached into, not replaced
            key = f'{mod} {attr}'
            COV.setdefault(key, 0)

            def counted(*args, __fn=fn, __key=key, **kwargs):
                COV[__key] += 1
                return __fn(*args, **kwargs)
            counted.__name__ = getattr(fn, '__name__', name)
            try:
                setattr(holder, name, counted)
            except (AttributeError, TypeError):
                pass


# the originals each module saved, which it calls when it declines a case:
# a call here is a call the module was on the path for and handed to GPU4PySCF
# (the same pairs `coverage_probe.py` counts)
_FALLBACKS = [
    ('fastk', 'gpu4pyscf.scf.jk:_VHFOpt._get_k_orig'),
    ('fastjk', 'gpu4pyscf.scf.jk:_VHFOpt._get_jk_orig'),
    ('fastj', 'gpu4pyscf.scf.j_engine:_VHFOpt._get_j_orig'),
    ('fastxc', 'gpu4pyscf.dft.numint:_nr_rks_orig'),
    ('fastxc', 'gpu4pyscf.dft.numint:_nr_uks_orig'),
    ('fastnlc', 'gpu4pyscf.dft.numint:_vv10nlc_orig'),
    ('fastejk', 'gpu4pyscf.grad.rhf:_jk_energy_per_atom_orig'),
    ('fastgradh', 'gpu4pyscf.grad.rhf:_get_hcore_orig'),
    ('fastgrad', 'gpu4pyscf.grad.rks:Gradients._energy_ee_orig'),
    ('fastxcgrad', 'gpu4pyscf.grad.rks:_get_exc_orig'),
    ('fastxcgrad', 'gpu4pyscf.grad.uks:_get_exc_orig'),
    ('fastxcfxc', 'gpu4pyscf.dft.numint:_nr_rks_fxc_orig'),
    ('fastxcfxc', 'gpu4pyscf.dft.numint:_nr_uks_fxc_orig'),
    ('fastxchess', 'gpu4pyscf.hessian.rks:_get_vxc_deriv1_orig'),
    ('fastxchess', 'gpu4pyscf.hessian.rks:_get_vxc_deriv2_orig'),
    ('fastfnlc', 'gpu4pyscf.hessian.rks:_nr_rks_fnlc_mo_orig'),
]
# inside the two-electron builds: class blocks taken by our kernels and by
# GPU4PySCF's (a basis-set or class restriction shows up here)
_INNER = [
    ('fastk', ('_run_k1', '_run_k2', '_run_general'), ('_run_ref', '_run_ref_new')),
    ('fastjk', ('_run_jk1', '_run_jk2'), ('_run_ref',)),
    ('fastj', ('_run_mdm',), ('_run_ref',)),
    ('fastejk', ('_run',), ('_run_ref', '_run_ref_new')),
]


def count_fallbacks():
    def _counter(holder, name, key):
        fn = getattr(holder, name, None)
        if fn is None:
            return
        COV.setdefault(key, 0)

        def counted(*args, __fn=fn, __key=key, **kwargs):
            COV[__key] += 1
            return __fn(*args, **kwargs)
        try:
            setattr(holder, name, counted)
        except (AttributeError, TypeError):
            pass
    for mod, path in _FALLBACKS:
        if mod not in active:
            continue
        try:
            holder, name = _resolve(path)
        except (AttributeError, ImportError):
            continue
        _counter(holder, name, f'{mod} -> stock {path.split(":")[1]}')
    for mod, ours, stock in _INNER:
        if mod not in active:
            continue
        m = importlib.import_module(f'electrolites.{mod}')
        for f in ours:
            _counter(m, f, f'{mod} classes ours {f}')
        for f in stock:
            _counter(m, f, f'{mod} classes stock {f}')


def stage(label):
    """Context manager timing a top-level stage of the driver."""
    class _S:
        def __enter__(self):
            _sync()
            self.t = time.perf_counter()

        def __exit__(self, *exc):
            _sync()
            T[label] = T.get(label, 0.0) + time.perf_counter() - self.t
            N[label] = N.get(label, 0) + 1
    return _S()


# --- the plans: (path, label, index of the density argument or None) ------
JK = [
    ('gpu4pyscf.scf.j_engine:_VHFOpt.get_j', '....get_j', 1),
    ('gpu4pyscf.scf.jk:_VHFOpt.get_k', '....get_k', 1),
    ('gpu4pyscf.scf.jk:_VHFOpt.get_jk', '....get_jk', 1),
]
PLANS = {
    'tdgrad': [
        ('gpu4pyscf.tdscf.rhf:TDHF.kernel', 'td.kernel (TDDFT)', None),
        ('gpu4pyscf.tdscf.rhf:TDA.kernel', 'td.kernel (TDA)', None),
        ('gpu4pyscf.scf.cphf:solve', '..cphf.solve (Z-vector)', None),
        ('gpu4pyscf.grad.tdrks:_contract_xc_kernel', '..contract_xc_kernel (D9)', None),
        ('gpu4pyscf.grad.tdrhf:Gradients.jk_energies_per_atom',
         '..jk_energies_per_atom grad (D10)', None),
        ('gpu4pyscf.nac.tdrhf:NAC.jk_energies_per_atom',
         '..jk_energies_per_atom nac (D10)', None),
        ('gpu4pyscf.dft.numint:NumInt.nr_rks_fxc', '....nr_rks_fxc', None),
        ('gpu4pyscf.dft.numint:NumInt.nr_rks_fxc_st', '....nr_rks_fxc_st', None),
        ('gpu4pyscf.grad.rhf:GradientsBase.get_hcore', '..grad get_hcore', None),
        ('gpu4pyscf.df.int3c2e:get_dh1e', '..get_dh1e', None),
    ] + JK,
    'ir': [
        ('gpu4pyscf.hessian.rhf:HessianBase.kernel', 'hessian.kernel', None),
        ('gpu4pyscf.hessian.rks:Hessian.partial_hess_elec', '..partial_hess_elec', None),
        ('gpu4pyscf.hessian.rks:Hessian.make_h1', '..make_h1', None),
        ('gpu4pyscf.hessian.rhf:HessianBase.solve_mo1', '..solve_mo1', None),
        ('gpu4pyscf.hessian.rhf:hess_elec', '..hess_elec', None),
    ] + JK,
    'scf': [
        ('gpu4pyscf.dft.rks:RKS.get_veff', '..get_veff (RKS)', None),
        ('gpu4pyscf.dft.uks:UKS.get_veff', '..get_veff (UKS)', None),
        ('gpu4pyscf.dft.numint:NumInt.nr_rks', '....nr_rks', None),
        ('gpu4pyscf.dft.numint:NumInt.nr_uks', '....nr_uks', None),
        ('gpu4pyscf.dft.numint:NumInt.nr_nlc_vxc', '....nr_nlc_vxc', None),
        ('gpu4pyscf.scf.hf:SCF.get_hcore', '..get_hcore', None),
        ('gpu4pyscf.grad.rhf:GradientsBase.get_hcore', '..grad get_hcore', None),
    ] + JK,
    'nmr': [
        ('gpu4pyscf.properties.shielding:get_h1ao', '..get_h1ao', None),
        ('gpu4pyscf.properties.shielding:get_vxc', '....get_vxc', None),
        ('gpu4pyscf.properties.shielding:nr_rks', '......nr_rks GIAO (C7)', None),
        ('gpu4pyscf.properties.shielding:get_jk', '......get_jk int2e_ig1 (C6)', None),
        ('gpu4pyscf.scf.jk:get_jk', '..jk.get_jk (C8)', None),
        ('gpu4pyscf.scf.cphf:solve', '..cphf.solve', None),
        ('gpu4pyscf.dft.numint:NumInt.nr_rks_fxc', '....nr_rks_fxc', None),
        ('gpu4pyscf.dft.numint:NumInt.nr_rks_fxc_st', '....nr_rks_fxc_st', None),
    ] + JK,
    'pcmhess': [
        ('gpu4pyscf.hessian.rhf:HessianBase.kernel', 'hessian.kernel (solute)', None),
        ('gpu4pyscf.hessian.rks:Hessian.partial_hess_elec', '..partial_hess_elec', None),
        ('gpu4pyscf.solvent.hessian.pcm:WithSolventHess.make_h1', '..make_h1 (+solvent)', None),
        ('gpu4pyscf.hessian.rks:Hessian.make_h1', '....make_h1 (vacuum)', None),
        ('gpu4pyscf.solvent.hessian.pcm:analytical_grad_vmat', '....pcm grad_vmat (F9)', None),
        ('gpu4pyscf.solvent.hessian.pcm:get_dqsym_dx', '......get_dqsym_dx (F11)', None),
        ('gpu4pyscf.solvent.hessian.pcm:int1e_grids', '......int1e_grids x3natm (F9)', None),
        ('gpu4pyscf.hessian.rhf:HessianBase.solve_mo1', '..solve_mo1', None),
        ('gpu4pyscf.solvent.hessian.pcm:WithSolventHess.get_veff_resp_mo', '....veff_resp_mo (+solvent)', None),
        ('gpu4pyscf.solvent.pcm:PCM._B_dot_x', '......PCM._B_dot_x', None),
        ('gpu4pyscf.solvent.pcm:PCM.hess', 'pcm solvent hess (d2)', None),
        ('gpu4pyscf.solvent.hessian.pcm:analytical_hess_nuc', '..pcm hess_nuc', None),
        ('gpu4pyscf.solvent.hessian.pcm:analytical_hess_qv', '..pcm hess_qv', None),
        ('gpu4pyscf.solvent.hessian.pcm:analytical_hess_solver', '..pcm hess_solver', None),
        ('gpu4pyscf.solvent.hessian.pcm:get_d2D_d2S', '....get_d2D_d2S (F10)', None),
        ('gpu4pyscf.solvent.hessian.pcm:get_d2F_d2A', '....get_d2F_d2A (F10)', None),
    ] + JK,
    'hess': [
        ('gpu4pyscf.hessian.rhf:HessianBase.kernel', 'hessian.kernel', None),
        ('gpu4pyscf.hessian.rks:Hessian.partial_hess_elec', '..partial_hess_elec', None),
        ('gpu4pyscf.hessian.rks:Hessian.make_h1', '..make_h1', None),
        ('gpu4pyscf.grad.rhf:get_grad_hcore', '....get_grad_hcore', None),
        ('gpu4pyscf.grad.rhf:GradientsBase.get_hcore', '......grad get_hcore', None),
        ('gpu4pyscf.hessian.rhf:_get_jk_ip1', '....jk_ip1', None),
        ('gpu4pyscf.hessian.rhf:HessianBase.solve_mo1', '..solve_mo1', None),
        ('gpu4pyscf.hessian.rks:_get_exc_deriv2_grid_response', '....exc_deriv2 grid response (B9)', None),
        ('gpu4pyscf.hessian.rks:_get_vxc_deriv1_grid_response', '....vxc_deriv1 grid response (B9)', None),
        ('gpu4pyscf.hessian.rks:get_d2weight_dAdB', '......get_d2weight_dAdB (B9)', None),
        ('gpu4pyscf.hessian.rks:get_dweight_dA', '......get_dweight_dA (B9)', None),
        ('gpu4pyscf.hessian.rks:_get_vnlc_deriv1', '....vnlc_deriv1 (B14+B13)', None),
        ('gpu4pyscf.hessian.rks:_get_enlc_deriv2', '....enlc_deriv2 (B14+B13)', None),
        ('gpu4pyscf.hessian.rks:nr_rks_fnlc_mo', '....nr_rks_fnlc_mo (CPHF, B13)', None),
        ('gpu4pyscf.hessian.rks:libgdft.VXC_vv10nlc_hess_eval_f_t', '......VV10 f_t kernel (B13)', None),
        ('gpu4pyscf.hessian.rks:libgdft.VXC_vv10nlc_hess_eval_UWABCE', '......VV10 UWABCE kernel', None),
        ('gpu4pyscf.hessian.rks:libgdft.VXC_vv10nlc_hess_eval_omega_derivative', '......VV10 omega-derivative kernel', None),
    ] + JK,
    'grad': [
        ('gpu4pyscf.grad.rhf:GradientsBase.get_hcore', '..grad get_hcore', None),
        ('gpu4pyscf.grad.rks:Gradients.get_veff', '..get_veff (grad RKS)', None),
        ('gpu4pyscf.grad.uks:Gradients.get_veff', '..get_veff (grad UKS)', None),
        ('gpu4pyscf.grad.rhf:_jk_energy_per_atom', '....jk_energy_per_atom', None),
        ('gpu4pyscf.grad.uks:get_exc', '....uks get_exc', None),
        ('gpu4pyscf.grad.rks:get_exc', '....rks get_exc', None),
        ('gpu4pyscf.grad.uks:get_nlc_exc', '....uks get_nlc_exc', None),
        ('gpu4pyscf.grad.rhf:_grad_nuc_without_ecp', '....grad_nuc_without_ecp', None),
        ('gpu4pyscf.grad.rhf:GradientsBase.get_ovlp', '..grad get_ovlp', None),
        ('gpu4pyscf.grad.rhf:GradientsBase.grad_nuc', '..grad_nuc', None),
        ('gpu4pyscf.df.int3c2e:get_dh1e', '..get_dh1e', None),
        ('gpu4pyscf.dft.numint:NumInt.nr_rks', 'scf nr_rks', None),
        ('gpu4pyscf.dft.numint:NumInt.nr_uks', 'scf nr_uks', None),
    ] + JK,
    'forcenac': [
        ('gpu4pyscf.tdscf.rhf:TDHF.kernel', 'td.kernel (TDDFT)', None),
        ('gpu4pyscf.scf.cphf:solve', '..cphf.solve (Z-vector)', None),
        ('gpu4pyscf.grad.tdrks:_contract_xc_kernel', '..contract_xc_kernel (D9)', None),
        ('gpu4pyscf.nac.tdrks_grad_nacv:_contract_xc_kernel_batched',
         '..contract_xc_kernel_batched (multistate)', None),
        ('gpu4pyscf.grad.tdrhf:Gradients.jk_energies_per_atom',
         '..jk_energies_per_atom grad (D10)', None),
        ('gpu4pyscf.nac.tdrhf:NAC.jk_energies_per_atom',
         '..jk_energies_per_atom nac (D10)', None),
        ('gpu4pyscf.nac.tdrhf_grad_nacv:NAC_multistates.jk_energies_per_atom',
         '..jk_energies_per_atom multistate', None),
        ('gpu4pyscf.nac.tdrks:get_nacv_ee', '..get_nacv_ee (one pair)', None),
        ('gpu4pyscf.nac.tdrks:get_nacv_ge', '..get_nacv_ge (one pair)', None),
    ] + JK,
    'sftda': [
        ('gpu4pyscf.tdscf.uhf:SpinFlipTDA.kernel', 'sf-tda.kernel', None),
        ('gpu4pyscf.tdscf._uhf_resp_sf:cache_xc_kernel_sf',
         '..cache_xc_kernel_sf (D13, mcol fxc)', None),
        ('gpu4pyscf.tdscf._uhf_resp_sf:eval_xc_eff_sf',
         '....eval_xc_eff_sf (sampled libxc)', None),
        ('gpu4pyscf.dft.numint:NumInt.nr_rks_fxc', '..nr_rks_fxc (SF response)', None),
        ('gpu4pyscf.grad.tduks_sf:_contract_xc_kernel',
         '..sf grad _contract_xc_kernel (D13)', None),
        ('gpu4pyscf.scf.ucphf:solve', '..ucphf.solve (Z-vector)', None),
        ('gpu4pyscf.grad.tduhf:Gradients.jk_energies_per_atom',
         '..jk_energies_per_atom (D10, UHF)', None),
        ('gpu4pyscf.grad.tduks_sf:Gradients.jk_energies_per_atom',
         '..jk_energies_per_atom (D10, SF)', None),
    ] + JK,
    'smdhess': [
        ('gpu4pyscf.hessian.rhf:HessianBase.kernel', 'hessian.kernel (solute)', None),
        ('gpu4pyscf.solvent.hessian.smd:get_cds', 'SMD CDS Hessian by FD (F12)', None),
        ('gpu4pyscf.solvent.smd:SMD.get_cds', '..SMD.get_cds', None),
        ('gpu4pyscf.solvent.smd:SMD.hess', 'SMD electrostatic hess (d2)', None),
        ('gpu4pyscf.solvent.hessian.pcm:analytical_grad_vmat', '..pcm grad_vmat', None),
        ('gpu4pyscf.solvent.pcm:PCM._B_dot_x', '..PCM._B_dot_x', None),
        ('gpu4pyscf.hessian.rhf:HessianBase.solve_mo1', '..solve_mo1', None),
    ] + JK,
    'ecp': [
        ('gpu4pyscf.scf.hf:get_ecp', '..get_ecp (SCF hcore)', None),
        ('gpu4pyscf.grad.rhf:get_ecp_ip', '..get_ecp_ip (grad.rhf)', None),
        ('gpu4pyscf.hessian.rhf:get_ecp_ip', '..get_ecp_ip (hessian.rhf)', None),
        ('gpu4pyscf.hessian.rhf:get_ecp_ipip', '..get_ecp_ipip (hessian.rhf)', None),
        ('electrolites.fastgradh:get_ecp_ip', '..get_ecp_ip (fastgradh)', None),
        ('gpu4pyscf.grad.uhf:get_ecp_ip', '..get_ecp_ip (grad.uhf)', None),
        ('gpu4pyscf.hessian.uhf:get_ecp_ip', '..get_ecp_ip (hessian.uhf)', None),
        ('gpu4pyscf.hessian.rhf:HessianBase.kernel', 'hessian.kernel', None),
        ('gpu4pyscf.hessian.uks:Hessian.partial_hess_elec', '..partial_hess_elec', None),
        ('gpu4pyscf.hessian.uks:Hessian.make_h1', '..make_h1', None),
        ('gpu4pyscf.hessian.rhf:HessianBase.solve_mo1', '..solve_mo1', None),
    ] + JK,
    'gks': [
        ('gpu4pyscf.dft.numint2c:NumInt2C.nr_vxc', '..NumInt2C.nr_vxc (A21)', None),
        ('gpu4pyscf.dft.mcfun_gpu:eval_xc_eff', '....mcfun eval_xc_eff', None),
        ('gpu4pyscf.lib.cupy_helper:eigh', '..eigh', None),
    ] + JK,
    'eda': [
        ('gpu4pyscf.properties.eda:get_eda_classical_electrostatic_energy', '..classical electrostatic', None),
        ('gpu4pyscf.properties.eda:get_eda_electrostatic_energy', '..frozen / electrostatic', None),
        ('gpu4pyscf.properties.eda:get_eda_dispersion_energy', '..dispersion', None),
        ('gpu4pyscf.properties.eda:get_eda_polarization_energy', '..polarization', None),
        ('gpu4pyscf.properties.eda:get_eda_charge_transfer_energy', '..charge transfer', None),
        ('gpu4pyscf.properties.eda:_get_eda_polarization_FERF_subspace_projector', '..FERF projector', None),
    ] + JK,
    'c6': [
        ('gpu4pyscf.properties.c6:_solve_full_spectrum', '..full spectrum (get_ab + host eig)', None),
        ('gpu4pyscf.tdscf.rhf:TDBase.get_ab', '....get_ab (D4)', None),
    ] + JK,
    'ris': [
        ('gpu4pyscf.tdscf.ris:get_Tpq', '..get_Tpq', None),
        ('gpu4pyscf.tdscf.ris:get_eri2c_inv_lower', '..get_eri2c_inv_lower', None),
    ],
    'pm6': [
        ('gpu4pyscf.sem.integral.fock:get_hcore', '..get_hcore', None),
        ('gpu4pyscf.sem.integral.fock:get_jk', '..get_jk (K1)', None),
        ('gpu4pyscf.sem.scf.hf:get_jk', '..hf.get_jk', None),
        ('gpu4pyscf.sem.grad.rhf:Gradients._grad_elec_nuc_batched',
         '..FD pair gradient (K2)', None),
    ],
    'raman': [
        ('gpu4pyscf.hessian.rhf:HessianBase.kernel', 'hessian.kernel', None),
        ('gpu4pyscf.properties.raman:polarizability_derivative_numerical_dEdE',
         'polarizability derivative by field FD (C11)', None),
        ('gpu4pyscf.scf.hf:SCF.scf', '..SCF (each)', None),
        ('gpu4pyscf.grad.rhf:GradientsBase.kernel', '..gradient (each)', None),
        ('gpu4pyscf.properties.polarizability:eval_polarizability', '..polarizability', None),
        ('gpu4pyscf.hessian.rhf:HessianBase.solve_mo1', '..solve_mo1', None),
    ] + JK,
    'dftdgrad': [
        ('gpu4pyscf.tdscf.rhf:TDHF.kernel', 'td.kernel (TDDFT)', None),
        ('gpu4pyscf.tdscf.rhf:TDA.kernel', 'td.kernel (TDA)', None),
        ('gpu4pyscf.scf.cphf:solve', '..cphf.solve (Z-vector)', None),
        ('gpu4pyscf.grad.tdrks:_contract_xc_kernel', '..contract_xc_kernel (D9)', None),
        ('gpu4pyscf.df.grad.tdrhf:_jk_energies_per_atom', '..DF jk_energies_per_atom (D11)', None),
        ('gpu4pyscf.df.grad.tdrhf:_jk_energies_by_dm_factors', '....by_dm_factors', None),
        ('gpu4pyscf.df.grad.tdrhf:_j_energies_per_atom', '..DF j_energies_per_atom', None),
    ],
    'qmmm': [
        ('gpu4pyscf.qmmm.itrf:QMMMSCF.get_hcore', '..SCF get_hcore (+MM)', None),
        ('gpu4pyscf.qmmm.itrf:QMMMSCF.energy_nuc', '..energy_nuc (+MM)', None),
        ('gpu4pyscf.qmmm.itrf:_mm_charge_integrals', '....MM charge integrals', None),
        ('gpu4pyscf.qmmm.itrf:QMMMGrad.get_hcore', '..grad get_hcore (+MM)', None),
        ('gpu4pyscf.qmmm.itrf:QMMMGrad.grad_hcore_mm', '..grad_hcore_mm (MM-atom forces, ip2)', None),
        ('gpu4pyscf.qmmm.itrf:QMMMGrad.grad_nuc', '..grad_nuc (+MM, host numpy)', None),
        ('gpu4pyscf.qmmm.itrf:QMMMGrad.grad_nuc_mm', '..grad_nuc_mm (host numpy)', None),
        ('gpu4pyscf.qmmm.itrf:QMMMGrad.extra_force', '..extra_force', None),
        ('gpu4pyscf.grad.rhf:GradientsBase.kernel', 'gradient', None),
    ] + JK,
    'x2c': [
        ('gpu4pyscf.x2c.sfx2c1e:SFX2C1E_SCF.get_hcore', '..get_hcore (sfX2C)', None),
        ('gpu4pyscf.x2c.sfx2c1e:SpinFreeX2CHelper.get_hcore', '....X2C helper get_hcore (K3)', None),
        ('gpu4pyscf.x2c.sfx2c1e:_atomic_1e_x', '......atomic 1e X', None),
    ] + JK,
    'fci': [],
    'mp2': [
        ('gpu4pyscf.mp.mp2:_make_eris', '..conventional MP2 _make_eris (I3)', None),
        ('gpu4pyscf.scf.int4c2e:get_int4c2e_ovov', '....get_int4c2e_ovov', None),
        ('gpu4pyscf.mp.mp2:kernel', '..conventional MP2 kernel', None),
        ('gpu4pyscf.mp.dfmp2_drivers:dfump2_kernel_one_gpu', '..DF-UMP2 one-GPU driver', None),
        ('gpu4pyscf.mp.dfmp2_addons:get_dfump2_energy_pair_intra', '....UMP2 pair-energy loop', None),
        ('gpu4pyscf.mp.dfmp2_addons:get_dfmp2_energy_pair_intra', '....RMP2 pair-energy loop', None),
    ],
    'disphess': [
        ('gpu4pyscf.hessian.rhf:HessianBase.kernel', 'hessian.kernel', None),
        ('gpu4pyscf.hessian.rks:Hessian.get_dispersion', '..Hessian.get_dispersion (F1)', None),
        ('gpu4pyscf.hessian.dispersion:get_dispersion', '..dispersion.get_dispersion (F1)', None),
        ('pyscf.dispersion.dftd3:DFTD3Dispersion.get_dispersion', '....one D3 evaluation (CPU)', None),
        ('pyscf.dispersion.dftd3:DFTD3Dispersion.__init__', '....D3 model rebuild', None),
        ('gpu4pyscf.hessian.rhf:HessianBase.solve_mo1', '..solve_mo1', None),
    ],
    'analysis': [],
    'dfmp2': [
        ('gpu4pyscf.mp.dfmp2_drivers:dfmp2_kernel_one_gpu', 'dfmp2_kernel_one_gpu', None),
        ('gpu4pyscf.df.int3c2e_bdiv:Int3c2eOpt.build', '..Int3c2eOpt.build', None),
        ('gpu4pyscf.mp.dfmp2_addons:get_j2c_bdiv', '..get_j2c_bdiv', None),
        ('gpu4pyscf.mp.dfmp2_addons:get_j2c_decomp_gpu', '..get_j2c_decomp_gpu', None),
        ('gpu4pyscf.mp.dfmp2_addons:get_j3c_ovl_gpu_bdiv', '..get_j3c_ovl (3c2e + transform)', None),
        ('gpu4pyscf.mp.dfmp2_addons:decompose_j3c_gpu', '..decompose_j3c_gpu', None),
        ('gpu4pyscf.mp.dfmp2_addons:get_dfmp2_energy_pair_intra',
         '..pair energy loop (I1)', None),
    ],
    'ccsd': [
        ('gpu4pyscf.cc.ccsd_incore:CCSD.ao2mo', '..ao2mo (_make_eris_incore)', None),
        ('gpu4pyscf.cc.ccsd_incore:CCSD.update_amps', '..update_amps', None),
        ('gpu4pyscf.cc.ccsd_incore:_direct_ovvv_vvvv', '....AO-direct ovvv/vvvv (I5)', None),
        ('gpu4pyscf.cc.ccsd_incore:CCSD.run_diis', '..run_diis', None),
        ('gpu4pyscf.cc.ccsd_incore:CCSD.energy', '..energy', None),
    ],
}
# coverage first, so that the timers wrap the counters and the counters see
# Electrolites' own objects rather than a timer
count_coverage()
count_fallbacks()
for path, label, nd in PLANS[a.feature]:
    wrap(path, label, nd)
for spec in a.wrap:
    path, label = spec.rsplit(':', 1)
    wrap(path, label)


def build_mf(mol, xc=None):
    from gpu4pyscf import dft, scf
    xc = a.xc if xc is None else xc
    if xc.lower() in ('hf', 'rhf', 'uhf'):
        mf = (scf.UHF if mol.spin else scf.RHF)(mol)
    else:
        mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=xc)
        mf.grids.level = a.grids
    if a.density_fit:
        mf = mf.density_fit(auxbasis=a.auxbasis)
    mf.conv_tol = a.conv_tol
    mf.direct_scf_tol = 1e-13
    mf.max_cycle = a.max_cycle
    mf.verbose = 0
    return mf


INFO = {}


def run_scf(mf):
    with stage('scf'):
        e = mf.kernel()
    INFO.update(e=float(e), converged=bool(mf.converged),
                cycles=int(getattr(mf, 'cycles', -1)))
    return e


def feature_tdgrad(mol):
    mf = build_mf(mol)
    run_scf(mf)
    td = mf.TDA() if a.tda else mf.TDDFT()
    td.nstates = a.nstates
    td.verbose = 0
    # `_lr_eig` sizes its trial space -- and a dense (2*max_space)^2 device
    # matrix -- from `max_memory`, which is a *host* budget.  At the 80 GB
    # `build_mol` sets, penicillin V asks for a 31 GB subspace matrix and
    # OOMs a 40 GB card in stock and patched alike; PySCF's own default is
    # 4000 MB, which is what an unconfigured user runs with
    td.max_memory = 4000
    with stage('td total'):
        td.kernel()
    INFO['e_td'] = [float(x) for x in cp.asnumpy(cp.asarray(td.e))]
    g = td.nuc_grad_method()
    g.verbose = 0
    if a.cphf_max_cycle:
        g.cphf_max_cycle = a.cphf_max_cycle
    with stage('td gradient'):
        de = g.kernel(state=a.state)
    INFO['grad_norm'] = float(np.linalg.norm(cp.asnumpy(cp.asarray(de))))
    if a.nac:
        i, j = (int(x) for x in a.nac.split(','))
        nac = td.nac_method()
        nac.states = (i, j)
        nac.verbose = 0
        with stage('nac'):
            nac.kernel()
        INFO['nac_norm'] = float(np.linalg.norm(cp.asnumpy(cp.asarray(nac.de))))


def feature_ir(mol):
    from gpu4pyscf.properties import ir
    mf = build_mf(mol)
    run_scf(mf)
    h = mf.Hessian()
    h.conv_tol_cpscf = a.conv_tol_cpscf
    h.verbose = 0
    with stage('ir total'):
        freq, inten = ir.eval_ir_freq_intensity(mf, h)
    INFO['freq_max'] = float(np.max(cp.asnumpy(cp.asarray(freq))))
    INFO['inten_sum'] = float(np.sum(cp.asnumpy(cp.asarray(inten))))


def feature_scf(mol):
    mf = build_mf(mol)
    if a.first_call:
        with stage('first get_hcore (fresh process)'):
            mf.get_hcore(mol)
        with stage('second get_hcore'):
            mf.get_hcore(mol)
    run_scf(mf)
    if not a.no_grad:
        g = mf.nuc_grad_method()
        g.verbose = 0
        with stage('gradient'):
            de = g.kernel()
        INFO['grad_norm'] = float(np.linalg.norm(cp.asnumpy(cp.asarray(de))))


def feature_grad(mol):
    feature_scf(mol)


def feature_nmr(mol):
    from gpu4pyscf.properties import shielding
    mf = build_mf(mol)
    run_scf(mf)
    with stage('nmr total'):
        d, p = shielding.eval_shielding(mf)
    tot = cp.asnumpy(cp.asarray(d)) + cp.asnumpy(cp.asarray(p))
    INFO['iso_first3'] = [float(np.trace(t)) / 3 for t in tot[:3]]


def feature_pcmhess(mol):
    mf = build_mf(mol).PCM()
    mf.with_solvent.method = a.solvent
    mf.with_solvent.eps = 78.3553
    mf.with_solvent.lebedev_order = 29
    run_scf(mf)
    h = mf.Hessian()
    h.conv_tol_cpscf = a.conv_tol_cpscf
    h.verbose = 0
    with stage('pcm hessian total'):
        hess = h.kernel()
    hess = cp.asnumpy(cp.asarray(hess))
    INFO['hess_trace'] = float(sum(np.trace(hess[i, i]) for i in range(mol.natm)))
    INFO['ngrids_cav'] = int(mf.with_solvent.surface['grid_coords'].shape[0])


def feature_hess(mol):
    mf = build_mf(mol)
    run_scf(mf)
    h = mf.Hessian()
    h.conv_tol_cpscf = a.conv_tol_cpscf
    h.verbose = 0
    if a.grid_response:
        h.grid_response = True
    with stage('hessian total'):
        hess = h.kernel()
    hess = cp.asnumpy(cp.asarray(hess))
    INFO['hess_trace'] = float(sum(np.trace(hess[i, i]) for i in range(mol.natm)))


def feature_forcenac(mol):
    """D12: one force plus the NACs of every state pair, as FSSH asks."""
    mf = build_mf(mol)
    run_scf(mf)
    td = mf.TDDFT()
    td.nstates = a.nstates
    td.verbose = 0
    td.max_memory = 4000
    with stage('td total'):
        td.kernel()
    g = td.nuc_grad_method()
    g.verbose = 0
    if a.cphf_max_cycle:
        g.cphf_max_cycle = a.cphf_max_cycle
    nac = td.nac_method()
    nac.verbose = 0
    pairs = [(i, j) for i in range(a.nstates + 1) for j in range(i + 1, a.nstates + 1)]
    with stage('force_and_nacv (serial pairs)'):
        force, nacvs = td.force_and_nacv(a.state, nac_pairs=pairs,
                                         td_grad=g, td_nac=nac)
    INFO['npairs'] = len(pairs)
    INFO['force_norm'] = float(np.linalg.norm(cp.asnumpy(cp.asarray(force))))
    ref = {k: cp.asnumpy(cp.asarray(v[0])) for k, v in nacvs.items()}
    INFO['nac_norms'] = {f'{i},{j}': float(np.linalg.norm(v)) for (i, j), v in ref.items()}
    if a.multistate:
        from gpu4pyscf.nac.tdrks_grad_nacv import NAC_multistates
        ms = NAC_multistates(td)
        ms.verbose = 0
        with stage('NAC_multistates (batched, all pairs + force)'):
            ms.kernel(states=list(range(a.nstates + 1)), grad_state=a.state)
        res = ms.results
        INFO['multistate_keys'] = sorted(str(k) for k in res)[:40]
        # agreement with the serial pairs, wherever the result names a pair
        diffs = {}
        if getattr(ms, 'grad_result', None) is not None:
            gm = np.asarray(cp.asnumpy(cp.asarray(ms.grad_result)), dtype=float)
            INFO['multistate_force_diff'] = float(np.abs(
                -gm - cp.asnumpy(cp.asarray(force))).max())
        for k, v in res.items():
            try:
                kk = tuple(int(x) for x in k)
                vv = v['de_scaled'] if isinstance(v, dict) else v[0]
                vv = np.asarray(cp.asnumpy(cp.asarray(vv)), dtype=float)
            except Exception:                          # noqa: BLE001
                continue
            if kk in ref and vv.shape == ref[kk].shape:
                d = min(np.abs(vv - ref[kk]).max(), np.abs(vv + ref[kk]).max())
                diffs[f'{kk[0]},{kk[1]}'] = float(d)
        INFO['multistate_maxdiff'] = diffs


def feature_sftda(mol):
    """D13: spin-flip TDA from a high-spin reference, and its gradient."""
    mf = build_mf(mol)
    run_scf(mf)
    td = mf.SFTDA()
    td.nstates = a.nstates
    td.collinear = a.collinear
    td.verbose = 0
    with stage('sf-tda total'):
        td.kernel()
    INFO['e_td'] = [float(x) for x in cp.asnumpy(cp.asarray(td.e))]
    g = td.nuc_grad_method()
    g.verbose = 0
    with stage('sf-tda gradient'):
        de = g.kernel(state=a.state)
    INFO['grad_norm'] = float(np.linalg.norm(cp.asnumpy(cp.asarray(de))))


def feature_smdhess(mol):
    mf = build_mf(mol).SMD()
    mf.with_solvent.solvent = 'water'
    mf.with_solvent.lebedev_order = 29
    run_scf(mf)
    h = mf.Hessian()
    h.conv_tol_cpscf = a.conv_tol_cpscf
    h.verbose = 0
    with stage('smd hessian total'):
        hess = h.kernel()
    hess = cp.asnumpy(cp.asarray(hess))
    INFO['hess_trace'] = float(sum(np.trace(hess[i, i]) for i in range(mol.natm)))


def feature_ecp(mol):
    """F8: an ECP-bearing system through SCF, gradient and Hessian."""
    mf = build_mf(mol)
    run_scf(mf)
    g = mf.nuc_grad_method()
    g.verbose = 0
    with stage('gradient'):
        g.kernel()
    if not a.no_grad:
        h = mf.Hessian()
        h.conv_tol_cpscf = a.conv_tol_cpscf
        h.verbose = 0
        with stage('hessian total'):
            h.kernel()


def feature_gks(mol):
    """A21: a GKS SCF with the multi-collinear functional."""
    from gpu4pyscf import dft
    mf = dft.GKS(mol, xc=a.xc)
    mf.collinear = a.collinear
    mf.grids.level = a.grids
    mf.conv_tol = a.conv_tol
    mf.max_cycle = a.max_cycle
    mf.verbose = 0
    run_scf(mf)


def feature_eda(mol):
    """C4: ALMO-EDA of two fragments."""
    from pyscf import gto
    from gpu4pyscf.properties import eda
    m2 = gto.M(atom=a.frag2, basis=a.basis, verbose=0)
    m1 = gto.M(atom=a.xyz, basis=a.basis, charge=a.charge, verbose=0)
    with stage('eda total'):
        res, dft_res = eda.eval_ALMO_EDA_2_energies(
            [m1, m2], xc=a.xc, verbose=0, conv_tol=a.conv_tol)
    INFO['eda'] = {k: float(v) for k, v in res.items()
                   if isinstance(v, (int, float))}


def feature_c6(mol):
    """C5: C6 coefficient of the molecule with itself."""
    from gpu4pyscf.properties import c6
    mf = build_mf(mol)
    run_scf(mf)
    td = mf.TDDFT()
    td.verbose = 0
    with stage('c6 total'):
        val = c6.calc_c6(td, td)
    INFO['c6'] = float(np.asarray(cp.asnumpy(cp.asarray(val))).ravel()[0])


def feature_ris(mol):
    """D6: TDDFT-ris against the full TDDFT on the same SCF."""
    from gpu4pyscf.tdscf import ris
    mf = build_mf(mol)
    run_scf(mf)
    td = ris.TDDFT(mf, nstates=a.nstates)
    td.verbose = 0
    with stage('ris total'):
        td.kernel()
    full = mf.TDDFT()
    full.nstates = a.nstates
    full.verbose = 0
    full.max_memory = 4000
    with stage('full TDDFT, for scale'):
        full.kernel()


def feature_pm6(mol):
    """K1/K2: PM6 SCF and its finite-difference gradient."""
    from gpu4pyscf.sem.gto import params as semparams
    from gpu4pyscf.sem.gto.mole import Mole
    from gpu4pyscf.sem.scf import hf as semhf
    # the 1.8.1 wheel ships without `sem/data/pm6_*.npy`, so PM6 cannot start
    # from an installed GPU4PySCF at all; read them from the 1.8.1 source tree
    import os
    _src = '/pscratch/sd/w/weiliang/g4psrc/g4p-1.8.1/gpu4pyscf/sem/data'
    _exists = os.path.exists

    def _load(self, _src=_src):
        prefix = self.method.lower()
        for key in ('alpha_bond', 'x_factor'):
            self._data[key] = np.load(os.path.join(_src, f'{prefix}_{key}.npy'))
    semparams.SEMParams._load_binary_matrices = _load
    m = Mole(atom=a.xyz, charge=a.charge, max_memory=32000, verbose=0)
    m.build()
    mf = semhf.RHF(m)
    mf.conv_tol = a.conv_tol
    mf.verbose = 0
    with stage('pm6 scf'):
        e = mf.kernel()
    INFO['e'] = float(e)
    g = mf.nuc_grad_method()
    g.verbose = 0
    with stage('pm6 gradient (FD pairs)'):
        g.kernel()


def feature_raman(mol):
    """C11: Raman intensities (a Hessian plus field finite differences)."""
    from gpu4pyscf.properties import raman
    mf = build_mf(mol)
    run_scf(mf)
    with stage('raman total'):
        out = raman.eval_raman_intensity(mf)
    INFO['n_out'] = len(out) if isinstance(out, (tuple, list)) else 1


def feature_dftdgrad(mol):
    """D11: a density-fitted TDDFT gradient, with its memory."""
    mf = build_mf(mol)          # --density-fit makes it DF
    run_scf(mf)
    td = mf.TDA() if a.tda else mf.TDDFT()
    td.nstates = a.nstates
    td.verbose = 0
    td.max_memory = 4000
    with stage('td total'):
        td.kernel()
    g = td.nuc_grad_method()
    g.verbose = 0
    with stage('td gradient'):
        de = g.kernel(state=a.state)
    INFO['grad_norm'] = float(np.linalg.norm(cp.asnumpy(cp.asarray(de))))


def feature_qmmm(mol):
    """F13: QM/MM SCF and gradient with many point charges."""
    from gpu4pyscf.qmmm import itrf
    # neutral point dipoles (+0.4 / -0.4 e, 1 A apart) scattered no closer
    # than 4 A to any QM atom, in a box 25 A beyond the molecule: an
    # electrostatic environment the SCF converges in, like a solvent shell
    rng = np.random.default_rng(20261004)
    c = mol.atom_coords() * 0.52917721      # Angstrom
    lo, hi = c.min(0) - 25.0, c.max(0) + 25.0
    npair = a.nmm // 2
    pts = []
    while sum(len(p) for p in pts) < npair:
        q = rng.uniform(lo, hi, size=(npair, 3))
        d = np.linalg.norm(q[:, None, :] - c[None, :, :], axis=2).min(1)
        pts.append(q[d > 4.0])
    centres = np.concatenate(pts)[:npair]
    u = rng.normal(size=(npair, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    coords = np.concatenate([centres + 0.5 * u, centres - 0.5 * u])
    charges = np.concatenate([np.full(npair, 0.4), np.full(npair, -0.4)])
    mf = itrf.add_mm_charges(build_mf(mol), coords, charges, unit='Angstrom')
    run_scf(mf)
    g = mf.nuc_grad_method()
    g.verbose = 0
    with stage('qmmm gradient'):
        de = g.kernel()
    INFO['nmm'] = a.nmm
    INFO['grad_norm'] = float(np.linalg.norm(cp.asnumpy(cp.asarray(de))))
    if hasattr(g, 'grad_nuc_mm'):
        with stage('MM-atom forces: grad_hcore_mm + grad_nuc_mm'):
            dm = mf.make_rdm1()
            g.grad_hcore_mm(dm)
            g.grad_nuc_mm()


def feature_x2c(mol):
    """K3: spin-free X2C SCF and gradient."""
    mf = build_mf(mol).x2c()
    run_scf(mf)


def feature_fci(mol):
    """I6: GPU4PySCF's FCI sigma build against PySCF's, on a CAS of --cas.

    1.8.1's `FCI` class swaps in a GPU `contract_2e` that wants a 2-D CI
    vector, while PySCF's Davidson hands it a 1-D one, so the solver is tried
    end to end first and the sigma build is then timed on its own."""
    from pyscf import scf as cscf, mcscf, ao2mo
    from pyscf.fci import direct_spin1 as cfci, cistring
    from gpu4pyscf.fci import direct_spin1 as gfci
    norb, nelec = (int(x) for x in a.cas.split(','))
    m = mol.copy()
    m.verbose = 0
    mfc = cscf.RHF(m).run()
    mc = mcscf.CASCI(mfc, norb, nelec)
    mc.fcisolver = gfci.FCI(m)
    mc.verbose = 0
    try:
        with stage(f'CASCI({nelec},{norb}) with the GPU FCI solver'):
            INFO['e_gpu'] = float(mc.kernel()[0])
    except Exception as err:                             # noqa: BLE001
        INFO['gpu_solver_error'] = f'{type(err).__name__}: {str(err)[:120]}'
    mc = mcscf.CASCI(mfc, norb, nelec)
    mc.verbose = 0
    with stage(f'CASCI({nelec},{norb}) with the PySCF CPU solver'):
        INFO['e_cpu'] = float(mc.kernel()[0])
    h1, ecore = mc.get_h1eff()
    h2 = ao2mo.restore(1, mc.get_h2eff(), norb)
    eri = cfci.absorb_h1e(h1, h2, norb, nelec, .5)
    link = cistring.gen_linkstr_index(range(norb), nelec // 2, tril=True)
    na = link.shape[0]
    rng = np.random.default_rng(1)
    ci = rng.random((na, na))
    ci /= np.linalg.norm(ci)
    eri_g = ao2mo.restore(4, eri, norb)
    # the GPU sigma build returns a host array (`direct_spin1.py:187`), so
    # both of its rows include the device-to-host copy of the result
    for name, fn in (('GPU contract_2e from a host CI vector',
                      lambda: gfci.contract_2e(eri_g, cp.asarray(ci), norb, nelec,
                                               (link, link))),
                     ('GPU contract_2e from a device CI vector',
                      lambda: gfci.contract_2e(eri_g, ci_d, norb, nelec, (link, link))),
                     ('CPU contract_2e (PySCF)',
                      lambda: cfci.contract_2e(eri, ci, norb, nelec, (link, link)))):
        ci_d = cp.asarray(ci)
        fn()
        _sync()
        with stage(f'sigma: {name}'):
            for _ in range(3):
                fn()
        N[f'sigma: {name}'] = 3
    INFO['na'] = int(na)


def feature_mp2(mol):
    """I2/I3: conventional RMP2, and DF-UMP2 on an open shell."""
    from gpu4pyscf import scf
    if mol.spin:
        from gpu4pyscf.mp import dfump2
        mf = scf.UHF(mol).density_fit()
        mf.conv_tol = a.conv_tol
        mf.max_cycle = a.max_cycle
        mf.verbose = 0
        run_scf(mf)
        pt = dfump2.DFUMP2(mf)
        pt.verbose = 0
        with stage('DF-UMP2 total'):
            pt.kernel()
    else:
        from gpu4pyscf.mp import mp2
        mf = scf.RHF(mol)
        mf.conv_tol = a.conv_tol
        mf.verbose = 0
        run_scf(mf)
        pt = mp2.MP2(mf)
        pt.verbose = 0
        with stage('conventional RMP2 total'):
            pt.kernel()
    INFO['e_corr'] = float(pt.e_corr)


def feature_disphess(mol):
    """F1: a -D3(BJ) Hessian; the dispersion part by CPU finite differences.

    With --no-grad only the dispersion Hessian is evaluated (it needs no SCF),
    which is how it is sized at 284 and 545 atoms."""
    if not a.no_grad:
        feature_hess(mol)
        return
    mf = build_mf(mol)
    h = mf.Hessian()
    with stage('dispersion Hessian alone'):
        h.get_dispersion()


def feature_analysis(mol):
    """E1, E2, E4, E5: population analysis, multipoles, ESP/RESP, CHELPG."""
    from gpu4pyscf.pop import esp
    from gpu4pyscf.qmmm import chelpg
    mf = build_mf(mol)
    run_scf(mf)
    dm = mf.make_rdm1()
    with stage('E2 dip_moment'):
        mf.dip_moment(verbose=0)
    with stage('E1 analyze (to_cpu + meta-Lowdin)'):
        mf.analyze(verbose=0)
    with stage('E4 esp_solve'):
        esp.esp_solve(mol, dm)
    with stage('E4 resp_solve'):
        esp.resp_solve(mol, dm)
    with stage('E5 CHELPG'):
        chelpg.eval_chelpg_layer_gpu(mf)


def feature_dfmp2(mol):
    from gpu4pyscf import scf
    from gpu4pyscf.mp import dfmp2
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = a.conv_tol
    mf.verbose = 0
    run_scf(mf)
    pt = dfmp2.DFMP2(mf)
    pt.verbose = 0
    with stage('dfmp2 total'):
        e = pt.kernel()
    INFO['e_corr'] = float(pt.e_corr)


def feature_ccsd(mol):
    from gpu4pyscf import scf
    from gpu4pyscf.cc import ccsd_incore
    mf = scf.RHF(mol)
    mf.conv_tol = a.conv_tol
    mf.verbose = 0
    run_scf(mf)
    cc = ccsd_incore.CCSD(mf)
    cc.verbose = 0
    with stage('ccsd total'):
        cc.kernel()
    INFO['e_corr'] = float(cc.e_corr)
    INFO['ccsd_cycles'] = int(getattr(cc, 'cycles', -1))


mol = build_mol(a)
if a.feature in ('dfmp2', 'ccsd'):
    a.xc = 'hf'                 # both run on an RHF reference
tag = (f'{a.feature}/{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}/'
       f'{"UKS" if mol.spin else "RKS"}/patch={a.patch or "none"}')
print(f'[{tag}] natm={mol.natm} nao={mol.nao} nelec={mol.nelectron} '
      f'OMP={os.environ.get("OMP_NUM_THREADS")} '
      f'gpu={cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}',
      flush=True)
if a.memsample:
    import threading
    _free0, _total = cp.cuda.runtime.memGetInfo()
    _mem = {'min_free': _free0, 'stop': False}
    _dev = cp.cuda.Device().id

    def _sampler():
        with cp.cuda.Device(_dev):
            while not _mem['stop']:
                f = cp.cuda.runtime.memGetInfo()[0]
                if f < _mem['min_free']:
                    _mem['min_free'] = f
                time.sleep(0.02)
    _th = threading.Thread(target=_sampler, daemon=True)
    _th.start()
t0 = time.perf_counter()
if a.cprofile:
    import cProfile
    import pstats
    prof = cProfile.Profile()
    prof.enable()
globals()[f'feature_{a.feature}'](mol)
_sync()
wall = time.perf_counter() - t0
if a.cprofile:
    prof.disable()
    prof.dump_stats(a.cprofile)
    print('\n# cProfile, by self time')
    pstats.Stats(prof).sort_stats('tottime').print_stats(40)
peak = cp.get_default_memory_pool().total_bytes() / 1e9
if a.memsample:
    _mem['stop'] = True
    INFO['device_peak_in_use_GB'] = round((_total - _mem['min_free']) / 1e9, 2)
    INFO['device_total_GB'] = round(_total / 1e9, 2)
    print(f"# device memory: peak in use {INFO['device_peak_in_use_GB']} GB "
          f"of {INFO['device_total_GB']} GB")

print(f'\n[{tag}] wall {wall:.2f} s   pool now {peak:.1f} GB')
print(f'{"stage":44s} {"s":>9s} {"%wall":>7s} {"calls":>6s}  ndm')
for k, v in T.items():
    if N[k] == 0:
        continue
    nd = sorted(NDM.get(k, ()))
    print(f'{k:44s} {v:9.2f} {100*v/wall:6.1f}% {N[k]:6d}  '
          f'{nd if nd else ""}')
_parents = {}
for (lab, par), v in TP.items():
    _parents.setdefault(lab, []).append((par, v))
for lab, lst in _parents.items():
    if len(lst) > 1:
        print(f'  {lab} by caller: ' + ', '.join(
            f'{par or "(top)"} {v:.2f} s' for par, v in lst))
print('\ncoverage (Electrolites entry points reached):')
for k, v in COV.items():
    if v:
        print(f'  {k:50s} {v}')
print('  not reached: ' + ', '.join(k for k, v in COV.items() if not v))
stats = {}
for m in active:
    mod = __import__(f'electrolites.{m}', fromlist=['x'])
    if hasattr(mod, 'stats'):
        try:
            stats[m] = mod.stats()
        except Exception as err:            # noqa: BLE001
            stats[m] = repr(err)
print('stats', stats)
print('RESULT ' + json.dumps(dict(
    tag=tag, feature=a.feature, xyz=a.xyz, basis=a.basis, xc=a.xc,
    patch=a.patch, natm=mol.natm, nao=int(mol.nao), wall=wall,
    stages={k: [T[k], N[k]] for k in T if N[k]}, coverage=dict(COV),
    by_caller={f'{k[0]} <- {k[1]}': v for k, v in TP.items()},
    info=INFO), default=str), flush=True)
