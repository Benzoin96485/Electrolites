"""
``force_and_nacv`` through the batched multi-state NAC solver (catalogue D12).

Importing this module replaces ``gpu4pyscf.tdscf.rhf.TDBase.force_and_nacv``.
GPU4PySCF 1.8.1's method computes the force on one state and then loops
``td_nac.kernel(states=(i, j))`` over the requested state pairs
(``tdscf/rhf.py:478-489``): one Z-vector CPHF, one set of derivative
integrals per pair.  GPU4PySCF also ships a batched solver,
``td.NACGradients()`` (``nac/tdr{h,k}s_grad_nacv.py``, ``NAC_multistates``,
labelled experimental and singlet-only), which solves the Z-vector equations
of every pair and of the force in one CPHF and contracts the derivative
integrals for all of them in one multi-density call.  It is the same
quantity: the same response equations to the same tolerance, the same
integrals, summed in a different order.  ``docs/KERNEL_TARGETS_1.8.1.md``
§11.1 measured it 2.1x / 2.0x / 1.86x faster on the patched force + all-NAC
run and agreeing with the serial path to the Z-vector tolerance.

Here the method takes the batched path when it is valid:

* a closed-shell (restricted) reference and singlet excitations -- the
  batched solver raises otherwise;
* no solvent model (``get_nacv_multi`` raises);
* the requested pairs are *all* pairs among the states they name, which is
  what the method's own default (``nac_pairs=None``) and an FSSH step ask for.
  A sparser request is left to the serial loop, because the batched solver
  would compute every pair among those states.

Anything else falls through to GPU4PySCF's loop.  The Z-vector settings
(``cphf_conv_tol``, ``cphf_max_cycle``) are taken from the ``td_nac`` object
when one is passed, as the serial loop would use them.  The return value has
the serial method's form: ``force`` and ``{(i, j): (de_scaled,
de_etf_scaled)}``.

``FASTNACMS_OFF=1`` keeps the serial loop.  ``docs/README_fastnacms.md``.
"""
import os

import cupy
import numpy as np

from gpu4pyscf.tdscf import rhf as TDR

_OFF = bool(os.environ.get('FASTNACMS_OFF'))

STATS = {'batched': 0, 'serial': 0}
DECLINED = {}


def stats():
    return dict(STATS, declined=dict(DECLINED))


def _decline(why):
    STATS['serial'] += 1
    DECLINED[why] = DECLINED.get(why, 0) + 1
    return None


def _batched(td, grad_state, nac_pairs, td_grad, td_nac):
    from gpu4pyscf.scf.hf import RHF
    mf = td._scf
    if not isinstance(mf, RHF):
        return _decline('not a restricted reference')
    if not getattr(td, 'singlet', True):
        return _decline('triplet')
    if getattr(mf, 'with_solvent', None) is not None:
        return _decline('solvent')
    nstates = td.nstates + 1
    if nac_pairs is None:
        nac_pairs = [(i, j) for i in range(nstates - 1)
                     for j in range(i + 1, nstates)]
    pairs = {(min(i, j), max(i, j)) for i, j in nac_pairs}
    if any(i == j for i, j in pairs) or not pairs:
        return _decline('pair request')
    states = sorted({s for p in pairs for s in p})
    full = {(states[a], states[b]) for a in range(len(states))
            for b in range(a + 1, len(states))}
    if pairs != full:
        return _decline('not all pairs of the named states')
    try:
        ms = td.NACGradients()
    except NotImplementedError:
        return _decline('no batched solver for this class')
    src = td_nac if td_nac is not None else td_grad
    for k in ('cphf_conv_tol', 'cphf_max_cycle'):
        if src is not None and hasattr(src, k):
            setattr(ms, k, getattr(src, k))
    ms.verbose = getattr(td_nac, 'verbose', td.verbose)
    res = ms.kernel(states=states, grad_state=grad_state)
    force = -cupy.asnumpy(cupy.asarray(ms.grad_result))
    nacvs = {}
    for i, j in nac_pairs:
        r = res[(min(i, j), max(i, j))]
        nacvs[i, j] = (r['de_scaled'], r['de_etf_scaled'])
    STATS['batched'] += 1
    return force, nacvs


def force_and_nacv(self, grad_state, nac_pairs=None, td_grad=None,
                   td_nac=None):
    out = _batched(self, grad_state, nac_pairs, td_grad, td_nac)
    if out is not None:
        return out
    return TDR.TDBase._force_and_nacv_orig(self, grad_state, nac_pairs,
                                           td_grad, td_nac)


force_and_nacv.__doc__ = TDR.TDBase.force_and_nacv.__doc__

_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    TDR.TDBase._force_and_nacv_orig = TDR.TDBase.force_and_nacv
    TDR.TDBase.force_and_nacv = force_and_nacv
    _PATCHED = True


apply_patch()
