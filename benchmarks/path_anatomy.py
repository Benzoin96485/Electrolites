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

Features: tdgrad, ir, scf, nmr, pcmhess, hess, grad, dfmp2, ccsd.  The
periodic path is `kpt_anatomy.py`.
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
                         'grad', 'dfmp2', 'ccsd'))
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
        t = time.perf_counter()
        out = orig(*args, **kwargs)
        _sync()
        T[label] += time.perf_counter() - t
        N[label] += 1
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
    with stage('hessian total'):
        hess = h.kernel()
    hess = cp.asnumpy(cp.asarray(hess))
    INFO['hess_trace'] = float(sum(np.trace(hess[i, i]) for i in range(mol.natm)))


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

print(f'\n[{tag}] wall {wall:.2f} s   pool now {peak:.1f} GB')
print(f'{"stage":44s} {"s":>9s} {"%wall":>7s} {"calls":>6s}  ndm')
for k, v in T.items():
    if N[k] == 0:
        continue
    nd = sorted(NDM.get(k, ()))
    print(f'{k:44s} {v:9.2f} {100*v/wall:6.1f}% {N[k]:6d}  '
          f'{nd if nd else ""}')
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
    info=INFO), default=str), flush=True)
