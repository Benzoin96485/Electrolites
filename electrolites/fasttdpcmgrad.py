"""One build of the PCM surface derivatives per TDDFT-PCM gradient
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` V5).

GPU4PySCF's TDDFT-PCM gradient (``solvent/tdscf/pcm.py``
``WithSolventTDSCFGradient.grad_elec``) calls ``grad_solver`` three times --
once through ``pcmobj.grad`` and twice with the transition and relaxed
potentials -- and each call rebuilds the dense surface derivatives
dF/dA, dSii and [3, n, n] dD/dS (``solvent/grad/pcm.py``
``get_dF_dA``, ``get_dSii``, ``get_dD_dS``) for IEF-PCM, SMD and SS(V)PE:
the same matrices, since they depend only on the cavity.  Here the three
builders are memoised for the duration of one ``grad_elec`` (keyed on the
surface object and arguments, then dropped), so the matrices are built
once (36 builds instead of 108 over six glycine gradients).  The gradient
agrees with GPU4PySCF's to 2e-13 (``tests/test_fasttdpcmgrad.py``).

``FASTTDPCMGRAD_OFF=1`` keeps the three builds.
"""
import os

from gpu4pyscf.solvent.grad import pcm as gpcm
from gpu4pyscf.solvent.tdscf import pcm as tdpcm

_OFF = bool(os.environ.get('FASTTDPCMGRAD_OFF'))
_ACTIVE = []
STATS = {'built': 0, 'reused': 0}


def _memo(name):
    orig = getattr(gpcm, name)

    def f(surface, *args, **kwargs):
        if not _ACTIVE:
            return orig(surface, *args, **kwargs)
        cache = _ACTIVE[-1]
        # array arguments (get_dSii's dF) by identity: they come from the
        # memoised get_dF_dA, so the same object on a repeat
        key = (name, id(surface),
               tuple(a if isinstance(a, (str, int, float, bool, type(None)))
                     else ('id', id(a)) for a in args),
               tuple(sorted((k, v if isinstance(v, (str, int, float, bool,
                                                    type(None)))
                             else ('id', id(v))) for k, v in kwargs.items())))
        hit = cache.get(key)
        if hit is not None and hit[0] is surface:
            STATS['reused'] += 1
            return hit[1]
        out = orig(surface, *args, **kwargs)
        cache[key] = (surface, out, args, kwargs)   # keep id-keyed args alive
        STATS['built'] += 1
        return out
    f.__wrapped__ = orig
    return f


def _wrap_grad_elec(orig):
    def grad_elec(self, *args, **kwargs):
        _ACTIVE.append({})
        try:
            return orig(self, *args, **kwargs)
        finally:
            _ACTIVE.pop()
    grad_elec.__wrapped__ = orig
    return grad_elec


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    for name in ('get_dF_dA', 'get_dSii', 'get_dD_dS'):
        setattr(gpcm, name, _memo(name))
    cls = tdpcm.WithSolventTDSCFGradient
    cls.grad_elec = _wrap_grad_elec(cls.grad_elec)
    _PATCHED = True


apply_patch()
