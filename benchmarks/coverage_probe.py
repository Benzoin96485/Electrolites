"""Which modules are on the path of *this* calculation, and how much of it?

Every module in `electrolites` falls through to GPU4PySCF for what it does not
cover, which is the right default and the reason a patched run is correct.  It
also means a whole module can be inert on a code path without anything saying
so: the run is simply as fast as an unpatched one.  Two entry points here were
inert on every open-shell ground state until this round, and the only way that
became visible was to count.

This script patches everything, then wraps each patch and the original it saved
with a counter, runs an SCF (and a gradient), and reports per entry point:

    calls   how many times the patched entry point was called at all
    ours    how many of those our code answered
    stock   how many it handed to GPU4PySCF -- a module that is on the code
            path but declines this case
    (-)     the entry point was never called: the module is not on this path

For three of the two-electron modules it also counts *inside* the call: how many
angular-momentum class blocks our kernels took and how many went to
GPU4PySCF's, which is where a basis-set restriction (`MAX_PRIM_PAIR`, a class
past the written set) shows up rather than as a fall-through of the whole call.

    python benchmarks/coverage_probe.py --xyz benchmarks/molecules/tmc_fecl4.xyz \
        --charge -1 --spin 5 --basis def2-svp --xc b3lyp

`--spin` picks UKS.  Compare the same molecule at `--spin 0` to see which rows
only a closed shell reaches.
"""
import json
from collections import OrderedDict

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--max-cycle', type=int, default=30)
a = ap.parse_args()
active = apply_patch(a.patch or 'all')

import cupy as cp                                           # noqa: E402
import gpu4pyscf                                            # noqa: E402
from gpu4pyscf import dft                                   # noqa: E402

C = OrderedDict()


def _count(holder, attr, label, kind):
    """Wrap holder.attr with a counter recorded under (label, kind)."""
    fn = getattr(holder, attr, None)
    if fn is None:
        return False
    key = (label, kind)
    C.setdefault(key, 0)

    def counted(*args, **kwargs):
        C[key] += 1
        return fn(*args, **kwargs)

    counted.__name__ = getattr(fn, '__name__', attr)
    try:
        setattr(holder, attr, counted)
    except (AttributeError, TypeError):
        return False
    return True


def _entry(label, holder, attr, orig_holder, orig_attr):
    """One entry point: the dispatch (ours+stock) and the original (stock)."""
    _count(orig_holder, orig_attr, label, 'stock')
    _count(holder, attr, label, 'calls')


# --- the entry points, in the order a run reaches them ---------------------
import electrolites                                          # noqa: E402
from gpu4pyscf.scf import jk as _jk, j_engine as _je         # noqa: E402
from gpu4pyscf.dft import numint as _ni                      # noqa: E402
from gpu4pyscf.dft import rks as _rks                        # noqa: E402
from gpu4pyscf.grad import rhf as _grhf, rks as _grks        # noqa: E402
from gpu4pyscf.grad import uks as _guks                      # noqa: E402
from gpu4pyscf.hessian import rks as _hrks                   # noqa: E402

_ENTRIES = [
    ('fastk   get_k',        _jk._VHFOpt, 'get_k',   _jk._VHFOpt, '_get_k_orig'),
    ('fastjk  get_jk',       _jk._VHFOpt, 'get_jk',  _jk._VHFOpt, '_get_jk_orig'),
    ('fastj   get_j',        _je._VHFOpt, 'get_j',   _je._VHFOpt, '_get_j_orig'),
    ('fastxc  nr_rks',       _ni.NumInt, 'nr_rks',   _ni, '_nr_rks_orig'),
    ('fastxc  nr_uks',       _ni.NumInt, 'nr_uks',   _ni, '_nr_uks_orig'),
    ('fastnlc _vv10nlc',     _ni, '_vv10nlc',        _ni, '_vv10nlc_orig'),
    ('fastxcnlc get_veff',   _rks.RKS, 'get_veff',   _rks.RKS, '_get_veff_orig'),
    ('fastejk  ejk_per_atom', _grhf, '_jk_energy_per_atom',
     _grhf, '_jk_energy_per_atom_orig'),
    ('fastgradh get_hcore',  _grhf.GradientsBase, 'get_hcore',
     _grhf, '_get_hcore_orig'),
    ('fastgrad  energy_ee',  _grks.Gradients, 'energy_ee',
     _grks.Gradients, '_energy_ee_orig'),
    ('fastxcgrad get_exc',   _grks, 'get_exc',       _grks, '_get_exc_orig'),
    ('fastxcgrad get_nlc',   _grks, 'get_nlc_exc',   _grks, '_get_nlc_exc_orig'),
    ('fastxcgrad uks_exc',   _guks, 'get_exc',       _guks, '_get_exc_orig'),
    ('fastxcfxc nr_rks_fxc', _ni, 'nr_rks_fxc',      _ni, '_nr_rks_fxc_orig'),
    ('fastxchess vxc_deriv1', _hrks, '_get_vxc_deriv1',
     _hrks, '_get_vxc_deriv1_orig'),
    ('fastfnlc fnlc_mo',     _hrks, 'nr_rks_fnlc_mo',
     _hrks, '_nr_rks_fnlc_mo_orig'),
]
for e in _ENTRIES:
    _entry(*e)

# --- inside the two-electron builds: class blocks ours vs GPU4PySCF's -----
_INNER = []
for mod, ours, stock in (
        ('fastk',  ('_run_k1', '_run_k2', '_run_general'),
                   ('_run_ref', '_run_ref_new')),
        ('fastjk', ('_run_jk1', '_run_jk2'), ('_run_ref',)),
        # fastj's single-density path launches its kernels inline rather than
        # through a named function, so only its multi-density family
        # (`_run_mdm`, the CPHF/TDDFT build) and its fall-through are counted
        # here.  An ordinary SCF therefore shows `(-)` on this row even when
        # `fastj get_j` above says every call was answered.
        ('fastj',  ('_run_mdm',), ('_run_ref',)),
        ('fastejk', ('_run',), ('_run_ref', '_run_ref_new'))):
    if mod not in active:
        continue
    m = __import__(f'electrolites.{mod}', fromlist=['x'])
    for f in ours:
        _count(m, f, f'  {mod} classes', 'ours')
    for f in stock:
        _count(m, f, f'  {mod} classes', 'stock')

mol = build_mol(a)
tag = (f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}/'
       f'{"UKS" if mol.spin else "RKS"}')
print(f'[{tag}] natm={mol.natm} nao={mol.nao} spin={mol.spin} '
      f'charge={mol.charge} lmax={int(mol._bas[:, 1].max())} '
      f'nprim_max={int(mol._bas[:, 2].max())} '
      f'ecp={"y" if len(mol._ecpbas) else "n"} '
      f'gpu4pyscf={gpu4pyscf.__version__}', flush=True)

mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc)
mf.conv_tol = a.conv_tol
mf.direct_scf_tol = 1e-13
mf.grids.level = a.grids
mf.verbose = 0
mf.max_cycle = a.max_cycle
e = mf.kernel()
cp.cuda.Stream.null.synchronize()
print(f'[{tag}] SCF E={e:.10f} conv={mf.converged} cycles={mf.cycles}',
      flush=True)
if not a.no_grad:
    mf.nuc_grad_method().kernel()
    cp.cuda.Stream.null.synchronize()
    print(f'[{tag}] gradient done', flush=True)

labels = []
for lab, _kind in C:
    if lab not in labels:
        labels.append(lab)
print(f'\n{"entry point":26s} {"calls":>7s} {"ours":>7s} {"stock":>7s}')
rows = {}
for lab in labels:
    calls = C.get((lab, 'calls'))
    stock = C.get((lab, 'stock'), 0)
    if calls is None:                       # an inner counter: ours vs stock
        ours = C.get((lab, 'ours'), 0)
        calls = ours + stock
        shown_ours = ours
    else:
        shown_ours = calls - stock
    rows[lab] = dict(calls=calls, ours=shown_ours, stock=stock)
    if calls == 0:
        print(f'{lab:26s} {"(-)":>7s} {"":>7s} {"":>7s}')
    else:
        print(f'{lab:26s} {calls:7d} {shown_ours:7d} {stock:7d}')

print('RESULT ' + json.dumps(dict(
    tag=tag, gpu4pyscf=gpu4pyscf.__version__, xyz=a.xyz, basis=a.basis,
    xc=a.xc, natm=mol.natm, nao=int(mol.nao), spin=int(mol.spin),
    charge=int(mol.charge), lmax=int(mol._bas[:, 1].max()),
    nprim_max=int(mol._bas[:, 2].max()), e=float(e),
    cycles=int(mf.cycles), rows=rows)), flush=True)
