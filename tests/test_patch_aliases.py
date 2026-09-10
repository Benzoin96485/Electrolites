"""Does every reference to a patched function actually see the patch?

Each module here replaces a GPU4PySCF entry point by assigning over it.  That
works for callers that look the name up at call time -- a method on a class, or
a module-global inside the same file.  It does **not** work for a caller that
bound the function to a name of its own at import time, and GPU4PySCF does that
in several places:

    gpu4pyscf/grad/uks.py:   get_nlc_exc = rks_grad.get_nlc_exc
    gpu4pyscf/hessian/uks.py: from gpu4pyscf.hessian.rks import nr_rks_fnlc_mo

A stale alias like that has no symptom.  The patched module is loaded, the
answer is right, and the run is exactly as fast as an unpatched one -- which is
how `grad.uks.get_exc` and `dft.numint.NumInt.nr_uks` went unnoticed through
four rounds of benchmarking on closed-shell systems.

This test patches everything, then walks every loaded ``gpu4pyscf`` module and
class for an attribute that still holds one of the *original* functions the
patches saved.  Each hit is either a real miss or a deliberate one; the
deliberate ones are listed in ``EXPECTED`` with the reason, so a new one shows
up as a failure rather than as a slow benchmark.

    python tests/test_patch_aliases.py

Needs no GPU: nothing is evaluated, only imported.
"""
import importlib
import inspect
import sys

import electrolites

# import the whole surface first, so an alias bound at import time exists
# before anything is patched -- which is the situation a user is in
for _m in ('gpu4pyscf.properties.shielding', 'gpu4pyscf.grad.rkspu',
           'gpu4pyscf.dft.rks', 'gpu4pyscf.dft.uks', 'gpu4pyscf.dft.numint',
           'gpu4pyscf.scf.jk', 'gpu4pyscf.scf.j_engine',
           'gpu4pyscf.grad.rhf', 'gpu4pyscf.grad.rks', 'gpu4pyscf.grad.uhf',
           'gpu4pyscf.grad.uks', 'gpu4pyscf.hessian.rhf',
           'gpu4pyscf.hessian.rks', 'gpu4pyscf.hessian.uks',
           'gpu4pyscf.gto.int3c1e', 'gpu4pyscf.tdscf.rks',
           'gpu4pyscf.tdscf.uks'):
    try:
        importlib.import_module(_m)
    except ImportError:
        pass

electrolites.patch_all()

#: where each module stashed the function it replaced
ORIGINALS = [
    ('fastk', 'gpu4pyscf.scf.jk', '_VHFOpt._get_k_orig'),
    ('fastjk', 'gpu4pyscf.scf.jk', '_VHFOpt._get_jk_orig'),
    ('fastj', 'gpu4pyscf.scf.j_engine', '_VHFOpt._get_j_orig'),
    ('fastxc', 'gpu4pyscf.dft.numint', '_nr_rks_orig'),
    ('fastxc', 'gpu4pyscf.dft.numint', '_nr_uks_orig'),
    ('fastnlc', 'gpu4pyscf.dft.numint', '_vv10nlc_orig'),
    ('fastxcfxc', 'gpu4pyscf.dft.numint', '_nr_rks_fxc_orig'),
    ('fastxcnlc', 'electrolites.fastxcnlc', '_ORIG_GET_VEFF'),
    ('fastejk', 'gpu4pyscf.grad.rhf', '_jk_energy_per_atom_orig'),
    ('fastgradh', 'gpu4pyscf.grad.rhf', '_get_hcore_orig'),
    ('fastgrad', 'gpu4pyscf.grad.rks', 'Gradients._energy_ee_orig'),
    ('fastxcgrad', 'gpu4pyscf.grad.rks', '_get_exc_orig'),
    ('fastxcgrad', 'gpu4pyscf.grad.rks', '_get_nlc_exc_orig'),
    ('fastxcgrad', 'gpu4pyscf.grad.uks', '_get_exc_orig'),
    ('fastxchess', 'gpu4pyscf.hessian.rks', '_get_vxc_deriv1_orig'),
    ('fastxchess', 'gpu4pyscf.hessian.rks', '_get_vxc_deriv2_orig'),
    ('fastfnlc', 'gpu4pyscf.hessian.rks', '_nr_rks_fnlc_mo_orig'),
]

#: aliases that are known and deliberate: (holder, attribute) -> why
EXPECTED = {
    # the saved originals themselves
    **{(mod, attr): 'the saved original' for _, mod, attr in ORIGINALS},
    ('gpu4pyscf.grad.uks', 'get_nlc_exc'):
        'grad/uks.py aliases grad.rks.get_nlc_exc at import.  Not a miss: the '
        'unrestricted VV10 gradient is declined anyway (two densities), and '
        'the term is reached through grad.rks._get_denlc, whose lookup of '
        'get_nlc_exc is a module-global in grad/rks.py and does see the patch.',
    ('gpu4pyscf.grad.uks', 'get_nlc_exc_full_response'):
        'the grid-response path, which no module here replaces.',
    ('gpu4pyscf.hessian.uks', 'nr_rks_fnlc_mo'):
        'hessian/uks.py imports the name at import time, so nr_uks_fnlc_mo '
        'calls the unpatched one.  fastfnlc declines UKS regardless, so this '
        'costs nothing today -- but it is where the patch has to go if that '
        'ever changes.',
}


def _originals():
    out = {}
    for name, mod_path, attr in ORIGINALS:
        obj = sys.modules.get(mod_path)
        if obj is None:
            continue
        for part in attr.split('.'):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            out[id(inspect.unwrap(obj))] = (name, f'{mod_path}.{attr}')
    return out


ORIG_BY_ID = _originals()
missing = [f'{m}.{a}' for _, m, a in ORIGINALS
           if not any(v[1] == f'{m}.{a}' for v in ORIG_BY_ID.values())]

hits = []
for mod_name, mod in list(sys.modules.items()):
    if not mod_name.startswith('gpu4pyscf') or mod is None:
        continue
    for attr, val in list(vars(mod).items()):
        if callable(val) and id(inspect.unwrap(val)) in ORIG_BY_ID:
            hits.append((mod_name, attr, ORIG_BY_ID[id(inspect.unwrap(val))]))
        elif isinstance(val, type) and getattr(val, '__module__', '') == mod_name:
            for m_attr, m_val in list(vars(val).items()):
                f = getattr(m_val, '__func__', m_val)
                if callable(f) and id(inspect.unwrap(f)) in ORIG_BY_ID:
                    hits.append((mod_name, f'{attr}.{m_attr}',
                                 ORIG_BY_ID[id(inspect.unwrap(f))]))

# an attribute whose name ends in _orig is a save slot, not a live caller
unexpected = [h for h in hits
              if (h[0], h[1]) not in EXPECTED and not h[1].endswith('_orig')]

print(f'{len(ORIG_BY_ID)} patched entry points, {len(hits)} references to the '
      f'functions they replaced\n')
for mod_name, attr, (owner, orig) in sorted(hits):
    why = EXPECTED.get((mod_name, attr))
    mark = '   ' if why else '>> '
    print(f'{mark}{mod_name}.{attr:28s} <- {owner} ({orig.rsplit(".", 1)[-1]})')
    if why and why != 'the saved original':
        print(f'      {why}')

if missing:
    print('\nentry points whose saved original was not found (the patch may '
          'have moved):')
    for m in missing:
        print(f'  {m}')
if unexpected:
    print('\nUNEXPECTED stale aliases -- these see the unpatched function:')
    for mod_name, attr, (owner, _) in unexpected:
        print(f'  {mod_name}.{attr}  ({owner})')

raise SystemExit(1 if (unexpected or missing) else 0)
