"""Direct SCF: a looser integral screen while the density is still far off
(catalogue A20, the cycle-dependent-threshold half).

GPU4PySCF's direct SCF already builds J and K incrementally from
``dm - dm_last``, but at the converged precision, ``direct_scf_tol`` (1e-13),
from the first cycle on -- when the density is still wrong in the third
digit.  Here, while the largest element of ``dm - dm_last`` is above
``SWITCH`` (1e-3), the J and K builds screen at ``LOOSE`` (1e-10); at the
first cycle below it the Fock matrix is **rebuilt from scratch** at
``direct_scf_tol`` (``dm_last``/``vhf_last`` dropped for that call) and the
SCF continues incrementally and tight.  Everything the converged answer is
made of -- the last full build and every increment after it -- is at the
original threshold; the loose cycles only shape the path there.

Measured (B3LYP/def2-SVP, ``conv_tol`` 1e-9; ``docs/KERNEL_TARGETS_1.8.1.md``
§13.24): taxol 18.6 -> 15.5 s, PfPMT 65.1 -> 50.5 s, the Ni dithiolene anion
(UKS, 35 cycles) unchanged; converged energies within 1e-10 Eh of the tight
run, and within PfPMT's own SCF noise there.  1e-9 as the loose threshold
moved taxol's energy by 1e-8 and was rejected.

The wrapper sits on the SCF classes' ``get_veff`` (``RKS``, ``UKS``,
``RHF``, ``UHF``), around whatever is there (``fastrsh``'s, for RKS); it
temporarily raises ``direct_scf_tol`` on the SCF object and on the J/K
option objects it holds (``_opt_gpu``, ``_opt_jengine``; the builds read it
at call time), and puts it back on all of them, new ones included, after the
call.  The state is per SCF object and reset by the
first, ``dm_last``-less, call of a run.  ``FASTSCFSCREEN_OFF=1`` keeps the
tight threshold throughout.
"""
import functools
import os

import cupy as cp

_OFF = bool(os.environ.get('FASTSCFSCREEN_OFF'))
LOOSE = float(os.environ.get('FASTSCFSCREEN_LOOSE', 1e-10))
SWITCH = float(os.environ.get('FASTSCFSCREEN_SWITCH', 1e-3))
STATS = {'loose_builds': 0, 'rebuilds': 0, 'tight_builds': 0}


def stats():
    return dict(STATS)


def _opts(mf):
    """The J/K option objects the SCF object holds (one per ``omega``)."""
    out = []
    for name in ('_opt_gpu', '_opt_jengine'):
        d = getattr(mf, name, None)
        if isinstance(d, dict):
            out += [o for o in d.values() if hasattr(o, 'direct_scf_tol')]
    # fastk's full-range sibling of a long-range option object
    out += [o._fastk_opt_full for o in list(out)
            if getattr(o, '_fastk_opt_full', None) is not None]
    return out


def _wrap_get_veff(get_veff):
    @functools.wraps(get_veff)
    def run(mf, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
        if (not getattr(mf, 'direct_scf', True)
                or getattr(mf, 'with_df', None) is not None):
            return get_veff(mf, mol, dm, dm_last, vhf_last, hermi)
        incremental = (dm_last is not None and vhf_last is not None
                       and not isinstance(dm_last, (int, float)))
        if not incremental:
            # the first build of a run (or a caller's own full build)
            mf.__dict__['_fastscfscreen_phase'] = 'loose'
            STATS['tight_builds'] += 1
            return get_veff(mf, mol, dm, dm_last, vhf_last, hermi)
        phase = mf.__dict__.get('_fastscfscreen_phase', 'tight')
        if phase == 'loose':
            ddm = float(abs(cp.asarray(dm) - cp.asarray(dm_last)).max())
            if ddm < SWITCH:
                mf.__dict__['_fastscfscreen_phase'] = 'tight'
                STATS['rebuilds'] += 1
                return get_veff(mf, mol, dm, None, None, hermi)
            STATS['loose_builds'] += 1
            tight = mf.direct_scf_tol
            mf.direct_scf_tol = max(tight, LOOSE)
            for opt in _opts(mf):
                opt.direct_scf_tol = max(tight, LOOSE)
            try:
                return get_veff(mf, mol, dm, dm_last, vhf_last, hermi)
            finally:
                # including any option object built during the call
                mf.direct_scf_tol = tight
                for opt in _opts(mf):
                    opt.direct_scf_tol = tight
        STATS['tight_builds'] += 1
        return get_veff(mf, mol, dm, dm_last, vhf_last, hermi)
    return run


_PATCHED = False


def apply_patch():
    global _PATCHED
    if _PATCHED or _OFF:
        return
    from gpu4pyscf.dft import rks, uks
    from gpu4pyscf.scf import hf, uhf
    for cls in (rks.RKS, uks.UKS, hf.SCF, hf.RHF, uhf.UHF):
        if 'get_veff' in cls.__dict__:
            cls._get_veff_fastscfscreen_orig = cls.__dict__['get_veff']
            cls.get_veff = _wrap_get_veff(cls.__dict__['get_veff'])
    _PATCHED = True


apply_patch()
