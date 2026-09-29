"""A whole density-fitted Hessian, stage by stage.

`benchmarks/df_hess_probe.py --whole` is where this table was first taken
(`docs/README_fastdfhess.md` §5); that script also runs W8's traced clone of
``_get_jk`` in front of it, which costs minutes at 113 atoms and has nothing
to do with the Hessian around it.  This one only builds the SCF and times the
Hessian, with the same wraps plus the ones W16-W18 need, and runs it
``--rounds`` times in one process so the last round is warm
(`docs/DF_WORK_LIST.md` §5.3).

    python benchmarks/df_hess_anatomy.py --xyz benchmarks/molecules/penicillinV.xyz \\
        --basis def2-svp --xc b3lyp --patch all --rounds 2

Nested rows (``..``) are sub-intervals of their parent.  The trace of the
Hessian is printed so two runs can be compared without a file.

An open-shell molecule (``--spin`` not 0) runs a UKS SCF and Hessian, and
the rows are the unrestricted entry points (W25).
"""
import importlib
import json
import sys
import time
from collections import OrderedDict

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--auxbasis', default=None)
ap.add_argument('--rounds', type=int, default=2)
ap.add_argument('--conv-tol-cpscf', type=float, default=1e-6)
ap.add_argument('--save', default='', help='write the last Hessian here (.npy)')
ap.add_argument('--ballast', type=float, default=0,
                help='GB held on the device for the whole run, before the '
                     'SCF: 40 on an 80 GB card is the memory of a 40 GB one '
                     '(W40)')
a = ap.parse_args()
if a.ballast:
    _ballast = cp.empty(int(a.ballast * 1e9) // 8)
    print(f'# ballast {a.ballast} GB, '
          f'{cp.cuda.runtime.memGetInfo()[0] / 1e9:.1f} GB free', flush=True)
patched = apply_patch(a.patch)

from gpu4pyscf import dft                                    # noqa: E402

_sync = cp.cuda.Stream.null.synchronize

mol = build_mol(a)
natm = mol.natm
uks = mol.spin != 0
mf = (dft.UKS if uks else dft.RKS)(mol, xc=a.xc).density_fit(
    auxbasis=a.auxbasis)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.conv_tol_cpscf = a.conv_tol_cpscf
mf.verbose = 0
t0 = time.perf_counter()
e = mf.kernel()
print(f'# {natm} atoms  nao {mol.nao}  basis {a.basis}  xc {a.xc}  '
      f'E = {e:.10f}  SCF {time.perf_counter() - t0:.2f} s', flush=True)

T, N = OrderedDict(), OrderedDict()


def wrap(mod_path, attr, label):
    try:
        mod = importlib.import_module(mod_path)
    except ImportError:
        return
    owner, name = mod, attr
    if '.' in attr:
        cls, name = attr.split('.')
        owner = getattr(mod, cls, None)
        if owner is None:
            return
    orig = getattr(owner, name, None)
    if orig is None:
        return
    T.setdefault(label, 0.0)
    N.setdefault(label, 0)

    def timed(*args, __orig=orig, __label=label, **kw):
        _sync()
        t = time.perf_counter()
        out = __orig(*args, **kw)
        _sync()
        T[__label] += time.perf_counter() - t
        N[__label] += 1
        return out
    setattr(owner, name, timed)


# class attributes, bound at class creation (tests/test_patch_aliases.py)
if uks:
    wrap('gpu4pyscf.df.hessian.uks', 'Hessian.partial_hess_elec',
         'partial_hess_elec')
    wrap('gpu4pyscf.df.hessian.uks', 'Hessian.make_h1', 'make_h1')
    wrap('gpu4pyscf.hessian.uhf', 'solve_mo1', 'solve_mo1 (CPHF)')
    wrap('gpu4pyscf.df.hessian.uhf', '_jk_energy_per_atom', '.. ip2 (W25)')
    wrap('gpu4pyscf.hessian.uks', '_get_exc_deriv2', '.. _get_exc_deriv2')
    wrap('gpu4pyscf.df.hessian.uhf', '_get_veff', '.. h1 _get_veff (W25)')
    wrap('gpu4pyscf.hessian.uks', '_get_vxc_deriv1', '.. _get_vxc_deriv1')
    wrap('gpu4pyscf.grad.rhf', 'get_grad_hcore', '.. get_grad_hcore')
    wrap('gpu4pyscf.dft.numint', 'NumInt.nr_uks_fxc', '.. nr_uks_fxc')
    # W33's route does not call `nr_uks_fxc`: the replaced CPHF method calls
    # the blocked function itself.  Same row.  Only if the patch is already
    # loaded -- importing the module here would apply it (§5.13).
    if 'electrolites.fastxcfxc' in sys.modules:
        wrap('electrolites.fastxcfxc', '_nr_uks_fxc_blocked', '.. nr_uks_fxc')
else:
    wrap('gpu4pyscf.df.hessian.rks', 'Hessian.partial_hess_elec',
         'partial_hess_elec')
    wrap('gpu4pyscf.df.hessian.rks', 'Hessian.make_h1', 'make_h1')
    wrap('gpu4pyscf.hessian.rhf', 'solve_mo1', 'solve_mo1 (CPHF)')
    # module globals, looked up at call time -- after every patch has run,
    # so these wrap whatever is installed
    wrap('gpu4pyscf.df.hessian.rhf', '_jk_energy_per_atom', '.. ip2 (W17)')
    wrap('gpu4pyscf.hessian.rks', '_get_exc_deriv2', '.. _get_exc_deriv2')
    wrap('gpu4pyscf.df.hessian.rhf', '_get_veff', '.. h1 _get_veff (W18)')
    wrap('gpu4pyscf.hessian.rks', '_get_vxc_deriv1', '.. _get_vxc_deriv1')
    wrap('gpu4pyscf.grad.rhf', 'get_grad_hcore', '.. get_grad_hcore')
    wrap('gpu4pyscf.hessian.rks', 'nr_rks_fxc_mo', '.. nr_rks_fxc_mo (W16)')
wrap('gpu4pyscf.df.hessian.rhf', '_get_jk', '.. _get_jk (W8)')
wrap('gpu4pyscf.dft.numint', 'NumInt.cache_xc_kernel', '.. cache_xc_kernel')
wrap('gpu4pyscf.df.df', '_cholesky_eri', 'cderi build')

res = {'natm': natm, 'nao': int(mol.nao), 'basis': a.basis, 'xc': a.xc,
       'patched': list(patched), 'rounds': []}
for r in range(a.rounds):
    for k in T:
        T[k] = 0.0
        N[k] = 0
    hess = mf.Hessian()
    hess.auxbasis_response = 2
    mf.with_df.reset()
    cp.get_default_memory_pool().free_all_blocks()
    _sync()
    t = time.perf_counter()
    h = hess.kernel()
    _sync()
    t_h = time.perf_counter() - t
    tr = float(np.trace(np.asarray(h).transpose(0, 2, 1, 3)
                        .reshape(3 * natm, 3 * natm)))
    print(f'\n## round {r}   whole Hessian {t_h:8.2f} s   trace {tr:.10f}')
    for k in T:
        print(f'  {k:28s} {T[k]:8.2f} s {100 * T[k] / t_h:6.1f} %'
              f'   {N[k]} calls')
    res['rounds'].append({'t': t_h, 'trace': tr, 'stages': dict(T),
                          'calls': dict(N)})
    if a.save and r == a.rounds - 1:
        np.save(a.save, np.asarray(h))
    h = hess = None

try:
    from electrolites import fastdfhessip, fastxcfxc
    res['stats'] = {'fastdfhessip': fastdfhessip.stats(),
                    'fastxcfxc': fastxcfxc.stats()}
    print('# fastdfhessip', fastdfhessip.stats())
    print('# fastxcfxc', fastxcfxc.stats())
except ImportError:
    pass
print('\n@@JSON ' + json.dumps(res))
