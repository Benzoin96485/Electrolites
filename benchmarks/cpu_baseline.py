"""The CPU side of every GPU-against-CPU number: PySCF 2.13 on a CPU node.

`docs/FEATURE_TARGETS_PYSCF_2.13.md` §0 rule 1 and `docs/CPU_BASELINE.md`:
a feature GPU4PySCF does not have is measured against **PySCF on the CPU**,
and this harness is the only way that number is taken, so that every CPU row
in `docs/` is taken the same way and carries its machine with it:

* the stages of the feature are wrapped and timed (with call counts), as
  `path_anatomy.py` does on the GPU; ``..`` rows are sub-intervals of their
  parent and overlap it;
* the run records the node, CPU model and core count, the thread counts
  (OpenMP, PySCF's, NumPy's BLAS) and **both** BLAS libraries -- NumPy's and
  the one PySCF's C extensions bundle, which is a different and much older
  OpenBLAS in the 2.13 wheel -- plus the peak resident memory;
* the energy (and whatever the feature computes) is printed in full, so the
  GPU run's correctness gate can be checked against the same line.

    OMP_NUM_THREADS=128 python benchmarks/cpu_baseline.py --feature ccsd_t \\
        --xyz benchmarks/molecules/glycine.xyz --basis def2-tzvp

Features: ``scf`` (exact exchange, the reference COSX is checked against),
``sgx`` (COSX), ``ccsd_t`` (CCSD then (T), split), ``casscf`` (CASSCF or
SA-CASSCF, conventional or ``--density-fit``), ``nevpt2`` (after a CASSCF),
``stability`` (internal and external), ``mp2grad`` (conventional MP2 and its
analytic gradient -- PySCF has no DF-MP2 gradient), ``dfmp2`` (DF-MP2 energy
and ``dfmp2_native``'s relaxed density: the pieces of a DF-MP2 gradient PySCF
has; with ``--density-fit`` for the SCF).
"""
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
import importlib
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import parser, needs_ecp                       # noqa: E402

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--feature', required=True,
                choices=('scf', 'sgx', 'ccsd_t', 'casscf', 'nevpt2',
                         'stability', 'mp2grad', 'dfmp2'))
ap.add_argument('--hf', action='store_true',
                help='Hartree-Fock reference instead of --xc')
ap.add_argument('--density-fit', action='store_true')
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--sgx-level', type=int, default=2)
ap.add_argument('--pjs', action='store_true',
                help='sgx: DF-J with P-junction-screened K (ORCA RIJCOSX-like)')
ap.add_argument('--frozen', type=int, default=0,
                help='ccsd_t/mp2grad/dfmp2: frozen core orbitals')
ap.add_argument('--ri-auxbasis', default=None,
                help='dfmp2: the RI (mp2fit) basis; default make_auxbasis')
ap.add_argument('--cas', default='10,10', help="casscf/nevpt2: 'norb,nelec'")
ap.add_argument('--nroots', type=int, default=1,
                help='casscf: state-average over this many roots')
ap.add_argument('--external', action='store_true',
                help='stability: also the external (RHF->UHF / UHF->GHF) check')
ap.add_argument('--max-cycle', type=int, default=100)
ap.add_argument('--json', default='', help='also write the record here')
a = ap.parse_args()

import numpy as np                                          # noqa: E402
import pyscf                                                # noqa: E402
from pyscf import gto, lib                                  # noqa: E402

T = OrderedDict()
N = OrderedDict()


def _resolve(path):
    mod, attr = path.split(':')
    obj = importlib.import_module(mod)
    parts = attr.split('.')
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def wrap(path, label):
    """Time ``module:Attr.path`` under ``label`` (missing targets are noted)."""
    try:
        holder, name = _resolve(path)
        orig = getattr(holder, name)
    except (AttributeError, ImportError) as err:
        print(f'# cannot wrap {path}: {err}', flush=True)
        return
    T.setdefault(label, 0.0)
    N.setdefault(label, 0)

    def timed(*args, **kwargs):
        t = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            T[label] += time.perf_counter() - t
            N[label] += 1
    timed.__wrapped__ = orig
    setattr(holder, name, timed)


def wrap_obj(obj, attr, label):
    """Time a method on one instance (e.g. ``mc.fcisolver``)."""
    orig = getattr(obj, attr)
    T.setdefault(label, 0.0)
    N.setdefault(label, 0)

    def timed(*args, **kwargs):
        t = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            T[label] += time.perf_counter() - t
            N[label] += 1
    setattr(obj, attr, timed)


def _blas_of_pyscf():
    """The BLAS PySCF's own C extensions load (not NumPy's)."""
    try:
        so = os.path.join(os.path.dirname(lib.__file__), 'libnp_helper.so')
        out = subprocess.run(['ldd', so], capture_output=True, text=True).stdout
        return [ln.split('=>')[0].strip() for ln in out.splitlines()
                if any(k in ln for k in ('blas', 'mkl', 'blis', 'flame'))]
    except Exception as exc:                                # noqa: BLE001
        return [f'unknown ({exc})']


def _blas_of_numpy():
    try:
        from threadpoolctl import threadpool_info
        return [{k: d.get(k) for k in ('internal_api', 'version',
                                       'num_threads', 'filepath')}
                for d in threadpool_info()]
    except ImportError:
        return ['threadpoolctl not installed']


def _cpu_model():
    try:
        with open('/proc/cpuinfo') as f:
            for ln in f:
                if ln.startswith('model name'):
                    return ln.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


ENV = dict(
    host=socket.gethostname(),
    slurm_job=os.environ.get('SLURM_JOB_ID'),
    cpu_model=_cpu_model(),
    cpus_visible=len(os.sched_getaffinity(0)),
    omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
    omp_proc_bind=os.environ.get('OMP_PROC_BIND'),
    pyscf_num_threads=lib.num_threads(),
    pyscf=pyscf.__version__,
    numpy=np.__version__,
    blas_numpy=_blas_of_numpy(),
    blas_pyscf_c=_blas_of_pyscf(),
    python=platform.python_version(),
)

# 400 GB of the CPU node's 512: PySCF's own default (4 GB) would push the
# larger runs out of core, which no one running PySCF on this node would do
kw = dict(atom=a.xyz, basis=a.basis, charge=a.charge, spin=a.spin,
          verbose=0, max_memory=int(os.environ.get('PYSCF_MAX_MEMORY',
                                                   400000)))
if needs_ecp(a.xyz, a.basis):
    kw['ecp'] = a.basis
mol = gto.M(**kw)
tag = f'{os.path.basename(a.xyz)}/{a.basis}/{a.feature}'
print(f'[{tag}] natm={mol.natm} nao={mol.nao} nelec={mol.nelectron} '
      f'spin={mol.spin}', flush=True)
print('# env ' + json.dumps(ENV), flush=True)

# ------------------------------------------------------------------ stages
# RHF/UHF.get_jk go in-core (8-fold ERIs) whenever max_memory allows, else
# to SCF.get_jk (direct); DF and SGX objects override both
wrap('pyscf.scf.hf:RHF.get_jk', 'get_jk (RHF)')
wrap('pyscf.scf.uhf:UHF.get_jk', 'get_jk (UHF)')
wrap('pyscf.scf.hf:SCF.get_jk', '..direct get_jk')
wrap('pyscf.df.df_jk:get_jk', 'DF get_jk')
wrap('pyscf.dft.numint:NumInt.nr_rks', 'nr_rks')
wrap('pyscf.dft.numint:NumInt.nr_uks', 'nr_uks')
wrap('pyscf.dft.gen_grid:Grids.build', 'grids.build')
if a.feature == 'sgx':
    wrap('pyscf.sgx.sgx:SGX.build', 'sgx grids + setup')
    wrap('pyscf.sgx.sgx_jk:get_k_only', '..sgx get_k_only')
    wrap('pyscf.sgx.sgx_jk:get_jk_favork', '..sgx get_jk_favork')
    wrap('pyscf.sgx.sgx_jk:get_jk_favorj', '..sgx get_jk_favorj')
if a.feature == 'ccsd_t':
    wrap('pyscf.cc.ccsd:_make_eris_incore', 'ccsd ao2mo (incore)')
    wrap('pyscf.cc.ccsd:_make_eris_outcore', 'ccsd ao2mo (outcore)')
    wrap('pyscf.cc.ccsd:CCSD.update_amps', 'ccsd update_amps')
    wrap('pyscf.cc.ccsd_t:_sort_eri', '..(T) sort_eri')
if a.feature in ('casscf', 'nevpt2'):
    wrap('pyscf.mcscf.mc_ao2mo:_ERIS.__init__', 'casscf _ERIS (conventional)')
    wrap('pyscf.mcscf.df:_ERIS.__init__', 'casscf _ERIS (DF)')
    wrap('pyscf.mcscf.mc1step:CASSCF.update_jk_in_ah', '..update_jk_in_ah')
    wrap('pyscf.mcscf.mc1step:gen_g_hop', 'gen_g_hop')
    wrap('pyscf.mcscf.casci:CASBase.get_h2eff', 'casci get_h2eff')
if a.feature == 'nevpt2':
    for fn in ('Sr', 'Si', 'Sijrs', 'Sijr', 'Srsi', 'Srs', 'Sij', 'Sir',
               '_contract4pdm', '_trans'):
        wrap(f'pyscf.mrpt.nevpt2:{fn}', f'nevpt2 {fn}')
if a.feature == 'stability':
    for fn in ('rhf_internal', 'rhf_external', 'uhf_internal',
               'uhf_external', 'rohf_internal'):
        wrap(f'pyscf.scf.stability:{fn}', f'stability {fn}')

RESULT = OrderedDict()


def timed_stage(label, fn):
    t = time.perf_counter()
    out = fn()
    T[label] = T.get(label, 0.0) + time.perf_counter() - t
    N[label] = N.get(label, 0) + 1
    return out


def mean_field():
    from pyscf import scf, dft
    # CASSCF wants restricted orbitals, so an open shell gets ROHF/ROKS
    ro = a.feature in ('casscf', 'nevpt2')
    if a.hf:
        mf = (scf.RHF(mol) if mol.spin == 0 else
              scf.ROHF(mol) if ro else scf.UHF(mol))
    else:
        mf = (dft.RKS(mol, xc=a.xc) if mol.spin == 0 else
              dft.ROKS(mol, xc=a.xc) if ro else dft.UKS(mol, xc=a.xc))
        mf.grids.level = a.grids
    if a.feature == 'sgx':
        from pyscf import sgx
        mf = sgx.sgx_fit(mf, auxbasis=a.auxbasis, pjs=a.pjs)
        mf.with_df.grids_level_i = mf.with_df.grids_level_f = a.sgx_level
    elif a.density_fit:
        mf = mf.density_fit(auxbasis=a.auxbasis)
    mf.conv_tol = a.conv_tol
    mf.max_cycle = a.max_cycle
    return mf


wall0 = time.perf_counter()
mf = mean_field()
timed_stage('SCF', mf.kernel)
RESULT['e_scf'] = mf.e_tot
RESULT['scf_converged'] = bool(mf.converged)
RESULT['scf_cycles'] = mf.cycles if hasattr(mf, 'cycles') else None
RESULT['eri_incore'] = getattr(mf, '_eri', None) is not None

if a.feature == 'ccsd_t':
    from pyscf import cc
    mycc = cc.CCSD(mf, frozen=a.frozen or None)
    mycc.conv_tol = 1e-9
    timed_stage('CCSD', mycc.kernel)
    et = timed_stage('(T)', mycc.ccsd_t)
    RESULT.update(e_corr_ccsd=mycc.e_corr, e_t=et,
                  e_ccsd_t=mycc.e_tot + et, ccsd_converged=mycc.converged)
elif a.feature in ('casscf', 'nevpt2'):
    from pyscf import mcscf
    norb, nelec = (int(x) for x in a.cas.split(','))
    mc = mcscf.CASSCF(mf, norb, nelec)
    if a.density_fit and not isinstance(mc, mcscf.df._DFCASSCF):
        mc = mc.density_fit(auxbasis=a.auxbasis)
    if a.nroots > 1:
        mc = mc.state_average_([1. / a.nroots] * a.nroots)
    mc.conv_tol = 1e-9
    fs = mc.fcisolver
    for attr, lab in (('kernel', 'fcisolver.kernel'),
                      ('make_rdm12', '..fci make_rdm12'),
                      ('contract_2e', '..fci contract_2e')):
        if hasattr(fs, attr):
            wrap_obj(fs, attr, lab)
    timed_stage('CASSCF', mc.kernel)
    RESULT.update(e_casscf=mc.e_tot, casscf_converged=bool(mc.converged),
                  e_states=list(np.atleast_1d(getattr(mc, 'e_states',
                                                      mc.e_tot))))
    if a.feature == 'nevpt2':
        from pyscf import mrpt
        if a.nroots > 1:
            raise SystemExit('nevpt2: run it on a single-root CASSCF')
        e2 = timed_stage('NEVPT2', lambda: mrpt.NEVPT(mc).kernel())
        RESULT['e_nevpt2_corr'] = e2
elif a.feature == 'mp2grad':
    from pyscf import mp
    pt = mp.MP2(mf, frozen=a.frozen or None)
    timed_stage('MP2', lambda: pt.kernel(with_t2=True))
    g = timed_stage('MP2 gradient', lambda: pt.nuc_grad_method().kernel())
    RESULT.update(e_mp2=pt.e_tot, grad=np.asarray(g).tolist())
elif a.feature == 'dfmp2':
    from pyscf.mp import dfmp2, dfmp2_native
    pt = dfmp2.DFMP2(mf, frozen=a.frozen or None)
    if a.ri_auxbasis:
        pt.with_df.auxbasis = a.ri_auxbasis
    timed_stage('DF-MP2 energy', pt.kernel)
    nat = dfmp2_native.DFRMP2(mf, frozen=a.frozen or None,
                              auxbasis=a.ri_auxbasis)
    timed_stage('DF-MP2 relaxed density (dfmp2_native)',
                lambda: nat.make_rdm1(relaxed=True))
    RESULT.update(e_dfmp2=pt.e_tot)
elif a.feature == 'stability':
    st = mf.stability
    out = timed_stage('stability', lambda: st(external=a.external,
                                              return_status=True))
    RESULT['stability_status'] = [bool(x) for x in out[-2:]]
wall = time.perf_counter() - wall0

peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
print(f'\n[{tag}] wall {wall:.2f} s   peak RSS {peak_gb:.1f} GB')
print(f'{"stage":44s} {"s":>9s} {"%wall":>7s} {"calls":>6s}')
for k, v in T.items():
    if N.get(k):
        print(f'{k:44s} {v:9.2f} {100*v/wall:6.1f}% {N[k]:6d}')
for k, v in RESULT.items():
    print(f'  {k} = {v!r}')
rec = dict(tag=tag, feature=a.feature, xyz=a.xyz, basis=a.basis,
           xc=None if a.hf else a.xc, charge=a.charge, spin=a.spin,
           natm=mol.natm, nao=mol.nao, wall=wall, peak_rss_gb=peak_gb,
           stages={k: [T[k], N[k]] for k in T if N.get(k)},
           result=RESULT, env=ENV, args=vars(a))
print('RESULT ' + json.dumps(rec, default=str))
if a.json:
    with open(a.json, 'w') as f:
        json.dump(rec, f, indent=1, default=str)
