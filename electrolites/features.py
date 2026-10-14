"""The interface for features Electrolites *adds* to GPU4PySCF.

Every module listed in ``electrolites.MODULES`` *replaces* a GPU4PySCF entry
point, and ``_TARGETS`` names the attribute it overwrites, so a GPU4PySCF
whose API moved fails loudly before anything is patched.  A feature that
GPU4PySCF does not have at all -- stability analysis, CCSD(T), CASSCF, COSX,
… (`docs/FEATURE_TARGETS_PYSCF_2.13.md`) -- needs the inverse check: the
attribute it provides must be **absent** upstream (missing, ``None`` or
``NotImplemented``), so that the day GPU4PySCF ships its own version this
repository notices instead of silently shadowing it.

A feature is declared with :func:`register` and turned on with
:func:`enable`::

    import electrolites
    electrolites.enable('stability')        # or electrolites.enable_all()

and it does three things, all through the helpers below:

* :func:`attach` binds a method onto a GPU4PySCF class (``RHF.stability``,
  ``RHF.CASSCF``, …), after checking that the class does not already have a
  working one;
* :func:`register_gpu_module` puts a module where PySCF's ``to_gpu()`` looks
  for it: ``pyscf/lib/misc.py`` maps ``pyscf.X.Y`` to ``gpu4pyscf.X.Y``, so
  ``pyscf_obj.to_gpu()`` reaches the GPU version once that name resolves;
* :func:`cpu_fallback` is the one way a case the feature declines goes back
  to PySCF: with a warning in the object's log, never silently and never by
  computing something else (`docs/FEATURE_TARGETS_PYSCF_2.13.md` §0 rule 4).

``python -c "import electrolites; print(electrolites.check_absent())"``
reports, for every registered feature, whether its upstream slot is still
empty.
"""
import importlib
import sys
import warnings

#: name -> spec dict (see :func:`register`)
FEATURES = {}
_enabled = []


def _resolve(mod_path, attr_path):
    obj = importlib.import_module(mod_path)
    for part in attr_path.split('.'):
        obj = getattr(obj, part)
    return obj


def is_absent(obj):
    """True when a GPU4PySCF slot is empty: missing, None or NotImplemented."""
    return obj is None or obj is NotImplemented


def register(name, module, absent=(), gpu_modules=(), doc=''):
    """Declare a feature.

    ``module``      -- the ``electrolites`` submodule whose import provides it;
    ``absent``      -- ``(module path, dotted attribute)`` pairs that must be
                       empty in GPU4PySCF for the feature to be enabled;
    ``gpu_modules`` -- ``gpu4pyscf.<path>`` names the feature registers for
                       PySCF's ``to_gpu()``, which must not already exist.
    """
    FEATURES[name] = dict(module=module, absent=tuple(absent),
                          gpu_modules=tuple(gpu_modules), doc=doc)


def check_absent(names=None):
    """``{feature: None}`` when every upstream slot it fills is still empty,
    else ``{feature: reason}``."""
    out = {}
    for name in (names or FEATURES):
        spec = FEATURES[name]
        out[name] = None
        for mod_path, attr_path in spec['absent']:
            try:
                obj = _resolve(mod_path, attr_path)
            except AttributeError:
                continue                         # missing: absent
            except ImportError as exc:
                out[name] = f'{mod_path} cannot be imported ({exc})'
                break
            if str(getattr(obj, '__module__', '')).startswith('electrolites'):
                continue                         # ours, from an earlier enable
            if not is_absent(obj):
                out[name] = (f'{mod_path}.{attr_path} now exists upstream '
                             f'({obj!r}); compare it with this feature '
                             'before shadowing it')
                break
        if out[name] is None:
            for path in spec['gpu_modules']:
                try:
                    importlib.import_module(path)
                except ImportError:
                    continue
                if not getattr(sys.modules[path], '_electrolites_feature',
                               False):
                    out[name] = f'{path} already exists upstream'
                    break
    return out


def enable(*names, strict=True):
    """Enable the named features; ``strict`` raises if an upstream slot is
    no longer empty (otherwise that feature is skipped with a warning)."""
    names = [n for group in names for n in
             ((group,) if isinstance(group, str) else group)]
    unknown = [n for n in names if n not in FEATURES]
    if unknown:
        raise ValueError(f'unknown feature(s) {unknown}; '
                         f'known: {list(FEATURES)}')
    clash = {k: v for k, v in check_absent(
        [n for n in names if n not in _enabled]).items() if v}
    if clash:
        if strict:
            raise RuntimeError(f'GPU4PySCF now provides what these features '
                               f'add: {clash}')
        warnings.warn(f'skipping features GPU4PySCF now provides: {clash}')
        names = [n for n in names if n not in clash]
    for name in names:
        importlib.import_module(f'electrolites.{FEATURES[name]["module"]}')
        if name not in _enabled:
            _enabled.append(name)
    return tuple(_enabled)


def enable_all(strict=True):
    return enable(list(FEATURES), strict=strict)


def enabled():
    return tuple(_enabled)


def attach(cls, name, fn, replace_broken=False):
    """Bind ``fn`` as ``cls.name``.

    The slot must be empty (missing, None, NotImplemented) unless
    ``replace_broken`` -- for a slot that exists but is known wrong or
    crashing, which the caller has to say explicitly.
    """
    cur = getattr(cls, name, None)
    if not (is_absent(cur) or replace_broken):
        raise RuntimeError(f'{cls.__module__}.{cls.__name__}.{name} already '
                           'exists; refusing to shadow it')
    setattr(cls, name, fn)
    return fn


def register_gpu_module(path, module):
    """Make ``import gpu4pyscf.<path>`` (and so ``pyscf_obj.to_gpu()``)
    resolve to ``module``."""
    assert path.startswith('gpu4pyscf.')
    module._electrolites_feature = True
    sys.modules[path] = module
    parent_path, _, leaf = path.rpartition('.')
    parent = importlib.import_module(parent_path)
    setattr(parent, leaf, module)
    return module


def cpu_fallback(obj, what, reason):
    """Hand ``obj`` to PySCF on the CPU, loudly.

    Logs a warning on ``obj``'s logger (and a Python warning, which survives
    ``verbose=0``) and returns ``obj.to_cpu()``.
    """
    from pyscf.lib import logger
    msg = (f'{what}: {reason}; running it on the CPU through PySCF '
           '(electrolites cpu_fallback)')
    logger.warn(obj, msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    return obj.to_cpu()


def not_available(what, reason):
    """A method that raises NotImplementedError with the reason, for a slot
    upstream fills with something wrong."""
    def method(self, *args, **kwargs):
        raise NotImplementedError(f'{what}: {reason}')
    method.__name__ = what.rsplit('.', 1)[-1]
    method._electrolites_guard = True
    return method


# ------------------------------------------------------------ the catalogue
# FEATURE_TARGETS_PYSCF_2.13.md IDs in brackets.

register('stability', 'stability',
         absent=[('gpu4pyscf.scf.hf', 'RHF.stability'),
                 ('gpu4pyscf.scf.uhf', 'UHF.stability'),
                 ('gpu4pyscf.scf.rohf', 'ROHF.stability')],
         gpu_modules=['gpu4pyscf.scf.stability'],
         doc='[S2] RHF/RKS, UHF/UKS internal + external, ROHF/ROKS internal')
register('rohf_grad', 'rohf_grad',
         absent=[('gpu4pyscf.scf.rohf', 'ROHF.nuc_grad_method'),
                 ('gpu4pyscf.dft.roks', 'ROKS.nuc_grad_method')],
         doc='[S3] ROHF/ROKS gradients, direct and DF')
register('sap', 'sap',
         absent=[('gpu4pyscf.scf.hf', 'SCF.init_guess_by_sap')],
         doc='[S8] the SAP initial guess (Gaussian fit)')
register('mp2_rdm', 'mp2_rdm',
         absent=[('gpu4pyscf.mp.mp2', 'MP2Base.make_rdm1'),
                 ('gpu4pyscf.mp.mp2', 'MP2Base.make_fno')],
         doc='[P5] MP2 unrelaxed 1-RDM and frozen natural orbitals')
register('casscf', 'casscf',
         absent=[('gpu4pyscf.scf.hf', 'SCF.CASSCF'),
                 ('gpu4pyscf.scf.hf', 'SCF.CASCI')],
         doc='[M1-M5, M2] CASSCF/CASCI (DF or exact) with integrals, J/K and FCI on the GPU')
register('ccsd_t', 'ccsd_t',
         absent=[('gpu4pyscf.cc.ccsd_incore', 'CCSDBase.ccsd_t')],
         doc='[P1] closed-shell (T) on the device')

__all__ = ['FEATURES', 'register', 'check_absent', 'enable', 'enable_all',
           'enabled', 'attach', 'register_gpu_module', 'cpu_fallback',
           'not_available', 'is_absent']
