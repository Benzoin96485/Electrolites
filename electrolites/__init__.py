"""Electrolites -- drop-in GPU kernel replacements for GPU4PySCF.

Importing one of the modules below monkey-patches the corresponding GPU4PySCF
entry point in place.  Whatever a module does not cover -- an angular-momentum
class, an open-shell density, a functional it declines -- falls through to
GPU4PySCF's own code, so a patched run computes the same quantity as an
unpatched one.

    import electrolites
    electrolites.patch_all()                     # every applicable module

    from pyscf import gto
    from gpu4pyscf import dft
    mol = gto.M(atom='...', basis='def2-tzvp')
    mf = dft.RKS(mol, xc='wb97m-v')
    e = mf.kernel()
    de = mf.nuc_grad_method().kernel()

``patch_all(only=[...])`` or ``patch('fastk', 'fastxc')`` patch a subset, which
is how the ablations in ``docs/`` were measured.  There is no unpatch: the
patches are global and the benchmarks compare separate processes, which is the
only honest way to A/B them.

Which module replaces what, and what it declines, is in README.md.
"""
import importlib
import inspect

__version__ = '0.1.0'

#: GPU4PySCF releases this has been run against end to end.
TESTED_GPU4PYSCF = ('1.7.0', '1.8.1')

#: The SCF modules, then the gradient modules, in the order they should load.
MODULES = (
    'fastk', 'fastj', 'fastxc', 'fastrsh', 'fastnlc',
    'fastejk', 'fastgrad', 'fastxcgrad', 'fastgradh',
)

# (module, dotted path of what it replaces) -- checked before anything patches,
# so a GPU4PySCF whose API moved fails loudly here instead of silently
# mis-patching or computing the wrong operator.
_TARGETS = {
    'fastk': ('gpu4pyscf.scf.jk', '_VHFOpt.get_k'),
    'fastj': ('gpu4pyscf.scf.j_engine', '_VHFOpt.get_j'),
    'fastxc': ('gpu4pyscf.dft.numint', 'NumInt.nr_rks'),
    'fastrsh': ('gpu4pyscf.dft.rks', 'RKS.get_veff'),
    'fastnlc': ('gpu4pyscf.dft.numint', '_vv10nlc'),
    'fastejk': ('gpu4pyscf.grad.rhf', '_jk_energy_per_atom'),
    'fastgrad': ('gpu4pyscf.grad.rks', 'Gradients.energy_ee'),
    'fastxcgrad': ('gpu4pyscf.grad.rks', 'get_exc'),
    'fastgradh': ('gpu4pyscf.grad.rhf', 'GradientsBase.get_hcore'),
}

_patched = []


def gpu4pyscf_version():
    import gpu4pyscf
    return gpu4pyscf.__version__


def check_targets(names=None):
    """Report which patch targets exist in the installed GPU4PySCF.

    Returns ``{module: None}`` when the target is present, or
    ``{module: reason}`` when it is missing.
    """
    out = {}
    for name in (names or MODULES):
        mod_path, attr_path = _TARGETS[name]
        try:
            obj = importlib.import_module(mod_path)
            for part in attr_path.split('.'):
                obj = getattr(obj, part)
            out[name] = None
        except (ImportError, AttributeError) as exc:
            out[name] = f'{mod_path}.{attr_path} is missing ({exc})'
    return out


def patch(*names, strict=True):
    """Patch the named modules.  ``strict`` raises if a target is missing."""
    names = [n for group in names for n in
             ((group,) if isinstance(group, str) else group)]
    unknown = [n for n in names if n not in _TARGETS]
    if unknown:
        raise ValueError(f'unknown module(s) {unknown}; known: {list(MODULES)}')
    missing = {k: v for k, v in check_targets(names).items() if v}
    if missing:
        msg = (f'GPU4PySCF {gpu4pyscf_version()} does not expose the entry '
               f'point(s) these modules replace: {missing}. '
               f'Tested against {", ".join(TESTED_GPU4PYSCF)}.')
        if strict:
            raise RuntimeError(msg)
        names = [n for n in names if n not in missing]
    for name in names:
        importlib.import_module(f'.{name}', __name__)
        if name not in _patched:
            _patched.append(name)
    return tuple(_patched)


def patch_all(only=None, strict=True):
    """Patch every module (or ``only`` a subset), in dependency order."""
    names = list(only) if only else list(MODULES)
    return patch([n for n in MODULES if n in names], strict=strict)


def patched():
    """The modules patched in this process, in the order they were applied."""
    return tuple(_patched)


__all__ = ['patch', 'patch_all', 'patched', 'check_targets',
           'gpu4pyscf_version', 'MODULES', 'TESTED_GPU4PYSCF', '__version__']
