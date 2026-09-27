"""Every cuTENSOR contraction, cut instead of refused.

`docs/DF_WORK_LIST.md` **W27**.  W21 (`docs/README_fastdfplan.md`) found that
cuTENSOR refuses a contraction whose *output* passes 2**29 elements unless one
particular mode of it factors, and that the refusal can be predicted from the
shapes alone.  Three modules then asked before their own block loops, and
W25 added a fourth.  W27 asked about everything else: `benchmarks/
cutensor_audit.py` records every contraction a density-fitted SCF, gradient,
Hessian and response calculation makes, by call site, and bounds its output
at the 545-atom HcgC case.  `docs/README_fastdfaudit.md` is the list.  At
that size most of a Hessian's contractions and a good part of a gradient's
are past the limit, in upstream's code as well as this repository's, and a
call site's extents move with the molecule, so a per-site guard would be a
list to keep in step with every future change.

This module is one guard for all of them.  It replaces
``gpu4pyscf/lib/cutensor.py:contraction`` -- which every ``contract`` in
GPU4PySCF and in this repository goes through, looked up at call time -- with
``_cuplan.contract_with`` over the original: the plain call for any output of
at most 2**29 elements (one product of extents, measured below the noise), a
cached plan query above it, and for a refused call the same contraction cut
along a free output mode into pieces that plan.  A contraction cuTENSOR
accepts is exactly the call it was; a refused one computes every element
with the same arithmetic, where before it raised
``CUTENSOR_STATUS_NOT_SUPPORTED``.

The block loops that already ask (`fastdfgrad`, `fastdfhess`, `fastdfhessip`,
`_dfhessip_u`) keep asking: they choose a block that plans whole, which is
one launch where a cut is several.

    PLANGUARD_OFF=1     leave cuTENSOR's refusal in place
"""
import os

from gpu4pyscf.lib import cutensor as _G

from . import _cuplan

_OFF = bool(os.environ.get('PLANGUARD_OFF'))

_orig_contraction = _G.contraction


def contraction(pattern, a, b, alpha, beta, out=None, **kw):
    return _cuplan.contract_with(_orig_contraction, pattern, a, b, alpha,
                                 beta, out=out, **kw)


def stats():
    return {k: v for k, v in _cuplan.stats().items() if k in ('cut', 'einsum')}


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF or not _cuplan.HAVE_CUTENSOR:
        return
    _G._contraction_unguarded = _orig_contraction
    _G.contraction = contraction
    _PATCHED = True


apply_patch()
