"""RIJONX -- density-fitted Coulomb with **exact** exchange -- made runnable.

`fastdf` is the other half of the density-fitting story: it rearranges the
work GPU4PySCF does *inside* the cderi contraction.  This module does not
touch a contraction at all.  It repairs the one branch of
``df/df_jk.py:_DFHF.get_jk`` that decides a density-fitted run does **not**
build a cderi, and so hands the exchange back to the direct path -- which is
where every other module in this repository already is.

**The bug.**  ``density_fit(only_dfj=True)`` is ORCA's RIJONX: fit J in the
auxiliary basis, take K from the four-centre integrals.  On GPU4PySCF 1.8.1
its Coulomb call is

    vj = self.with_df.get_jk(mol, dm, hermi, with_k=False)[0]   # df_jk.py:213

and ``DF.get_jk``'s signature is ``(dm, hermi, with_j, with_k, ...)`` -- there
is no ``mol`` argument.  So the molecule arrives as the density, ``hermi`` as
``with_j``, and the run dies in ``df_jk.get_j`` with ``'Mole' object has no
attribute 'shape'`` before it has computed anything.  Every ``only_dfj=True``
calculation on 1.8.1 fails this way, restricted or unrestricted, hybrid or
pure.  The GHF spin-free branch twenty lines above (``df_jk.py:170``) makes
the same call *correctly*, which is what the fixed line is copied from.

This module intercepts exactly that branch and leaves the rest of
``_DFHF.get_jk`` -- the whole GHF/DHF complex-density build, and the full-DF
path everything else takes -- to GPU4PySCF.  ``FASTDFJ_OFF=1`` restores the
broken call, so the failure above is one environment variable away rather
than a claim in a document.

**Two more entry points, for the same reason.**  Repairing the argument order
makes RIJONX run; it does not make it fast, and measuring it showed why.

* ``_DFHF.get_veff`` rebuilds from the **whole** density every cycle, and
  ``_DFHF.__init__`` sets ``direct_scf = False`` to say so.  That is right for
  a cderi contraction, which is dense and does not screen on the density.  It
  is badly wrong for a four-centre exchange build, which does: measured on
  taxol, RIJONX was *cheaper per call* than the direct path and *slower over a
  whole SCF*, and the whole of the difference was this.  An ``only_dfj``
  object is handed to its own class's ``get_veff`` instead, so the exchange is
  built from ``D - D_last``.  ``FASTDFJ_NO_INCREMENTAL=1`` puts it back.
* ``_DFHF.get_k`` serves a range-separated request as *two* exchange builds
  that it scales and adds.  The direct path has served ``lr*erf + sr*erfc`` in
  one pass since 1.8.0 and `fastk` does too, so under ``only_dfj`` the factors
  are handed straight through and a range-separated run builds K once.

**Why the branch is worth repairing.**  A J-only DF build never constructs a
cderi at all: ``df/df.py:111`` returns ``df_jk.get_j`` before the tensor is
touched, and that contracts the three-centre integrals with the density
directly in ``df/j_engine_3c2e.py``.  So RIJONX has none of the storage that
makes a 284-atom DF run expensive -- measured here, ``docs/DF_WORK_LIST.md``
§3.2, the cderi is 86.4 GB at 284 atoms and lives on the host, and §3.3
measures a full-DF ``get_jk`` call there at 12.52 s of which 12.38 s is the
exchange.  The exchange RIJONX needs instead is `fastk`, which this
repository has run at 284 and 545 atoms on the direct path.

What that is worth, and what it costs, is `docs/README_fastdfj.md`.

**Two things a caller has to know.**

*It is a different energy.*  RIJONX is not a faster route to the full-DF
number: J is fitted and K is not, so the fitting error no longer partially
cancels between the two.  ``tests/test_dfj.py`` measures the gap against both
limits on the size ladder rather than asserting one tolerance.

*The gradient does not follow.*  ``_DFHF.Gradients()`` returns the DF gradient
whatever ``only_dfj`` is -- nothing in ``df/grad/`` reads the flag -- so a
gradient taken from an ``only_dfj=True`` object is not the derivative of the
energy that object computed.  That is upstream's behaviour and this module
does not change it; it warns once instead, and the benchmark takes
``--no-grad``.
"""
import os
import warnings

import cupy as cp

from gpu4pyscf.df import df_jk as DFJK
from gpu4pyscf.dft import rks, uks
from gpu4pyscf.scf import ghf, hf, rohf, uhf

_OFF = bool(os.environ.get('FASTDFJ_OFF'))
#: `1` keeps GPU4PySCF's `_DFHF.get_veff`, which rebuilds the exchange
#: from the *whole* density every cycle.  The ablation column of
#: `docs/README_fastdfj.md` 4 is measured with this.
_NO_INCREMENTAL = bool(os.environ.get('FASTDFJ_NO_INCREMENTAL'))

_orig_get_jk = DFJK._DFHF.get_jk
#: where the other modules stash the function they replaced, so
#: `tests/test_patch_aliases.py` can walk for a stale reference to it
DFJK._DFHF._get_jk_orig = _orig_get_jk

#: counters, so a coverage claim can be read off a run rather than assumed.
#: `only_dfj` is the number of calls that took the repaired branch and
#: `passthrough` the number that went to GPU4PySCF unchanged.
STATS = {'calls': 0, 'only_dfj': 0, 'passthrough': 0, 'j': 0, 'k': 0,
         'veff': 0, 'veff_passthrough': 0, 'fused_k': 0}


def stats():
    """What this module did in this process.  Nothing numerical can say."""
    return dict(STATS)


def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
           omega=None):
    """``_DFHF.get_jk`` with the ``only_dfj`` Coulomb call's arguments fixed.

    Only the restricted/unrestricted ``only_dfj=True`` branch is served here.
    A GHF/DHF object keeps GPU4PySCF's own build even under ``only_dfj``,
    because that branch is already correct, and so does every ``only_dfj=False``
    call -- which is the whole of `fastdf`'s territory and none of this
    module's.
    """
    STATS['calls'] += 1
    if _OFF or not getattr(self, 'only_dfj', False) or isinstance(self, ghf.GHF):
        STATS['passthrough'] += 1
        return _orig_get_jk(self, mol, dm, hermi, with_j, with_k, omega)

    if mol is None:
        mol = self.mol
    if dm is None:
        dm = self.make_rdm1()
    STATS['only_dfj'] += 1

    vj = vk = None
    if with_j:
        # the changed line.  `omega` is forwarded because the full-DF branch
        # forwards it; `DF.get_jk` ignores it on the no-cderi return, which is
        # upstream's behaviour for a J-only call on either branch and is
        # harmless here -- no driver in GPU4PySCF asks a density-fitting
        # object for a range-separated *Coulomb* matrix.
        #
        # The leading axes are flattened because this branch reaches
        # `df_jk.get_j` for *every* Coulomb matrix it ever builds -- there is
        # no cderi to fall back on -- and that function asserts
        # `1 < ndim <= 3` (`gto/mole.py:907`).  An unrestricted response solve
        # asks for `(2, nset, nao, nao)`, which a full-DF run survives only
        # because its SCF has already built a cderi and taken the other
        # branch.  The flattening is a reshape and a reshape back; `get_j`
        # reads no tag off the density.
        # a 2- or 3-dimensional density is handed on untouched, so that the
        # common path is byte for byte what the repaired upstream line would
        # do -- including its handling of an array that is not on the device
        dm_j, shape = dm, getattr(dm, 'shape', None)
        flat = getattr(dm, 'ndim', 2) > 3
        if flat:
            dm_j = cp.asarray(dm).reshape(-1, shape[-2], shape[-1])
        vj = self.with_df.get_jk(dm_j, hermi, with_j=True, with_k=False,
                                 omega=omega)[0]
        if flat:
            vj = vj.reshape(shape)
        STATS['j'] += 1
    if with_k:
        # the exact four-centre build, i.e. `fastk` under `patch_all()`
        vk = hf.SCF.get_k(self, mol, dm, hermi, omega=omega)
        STATS['k'] += 1
    return vj, vk


def get_k(self, mol=None, dm=None, hermi=1, omega=None, lr_factor=None,
          sr_factor=None):
    """``_DFHF.get_k``, with the range-separated request kept in one call.

    Upstream's ``_DFHF.get_k`` serves ``lr*erf(wr)/r + sr*erfc(wr)/r`` as
    *two* ``get_jk`` calls that it then scales and adds.  That is the right
    shape when the exchange comes out of a cderi, where the two operators need
    two different tensors.  Under ``only_dfj`` the exchange is the direct
    path's, and ``scf/jk.py`` has served the fused request in a single pass
    over the Rys integrals since 1.8.0 (`fastk` does too) -- so splitting it
    doubles the work for nothing.  Handing the factors straight through is the
    whole of this function.
    """
    if _OFF or not getattr(self, 'only_dfj', False) or isinstance(self, ghf.GHF):
        return _orig_get_k(self, mol, dm, hermi, omega, lr_factor, sr_factor)
    if mol is None:
        mol = self.mol
    if dm is None:
        dm = self.make_rdm1()
    STATS['fused_k'] += 1
    return hf.SCF.get_k(self, mol, dm, hermi, omega, lr_factor, sr_factor)


def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
    """``_DFHF.get_veff``, but an ``only_dfj`` object drives its own class's.

    This is the larger half of what `fastdfj` is worth, and it is a *driver*
    change rather than a kernel one.

    ``_DFHF.get_veff`` exists because a density-fitted build gains nothing
    from an incremental Fock matrix: the cderi contraction is dense, it does
    not screen on the density, and building it from ``D - D_last`` costs
    exactly what building it from ``D`` costs while accumulating round-off.
    So the function ignores ``dm_last``/``vhf_last`` entirely and rebuilds
    from the whole density every cycle -- and ``_DFHF.__init__`` sets
    ``direct_scf = False`` to say so.

    Under ``only_dfj`` that reasoning no longer holds for the half that
    matters.  The exchange is the **four-centre** build, whose cost is set by
    how much of the shell-pair space survives screening, and a converged SCF's
    ``D - D_last`` screens away most of it.  Measured on taxol
    (`docs/README_fastdfj.md` 3.3), RIJONX is *cheaper per call* than the
    direct path and *slower over a whole SCF*, and the entire difference is
    this: the direct path builds from the difference and RIJONX was building
    from the whole density fifteen times.

    So an ``only_dfj`` object is handed to the ``get_veff`` of the class it
    was built from -- ``dft/rks.py``, ``dft/uks.py``, ``scf/hf.py`` or
    ``scf/uhf.py`` -- which is the incremental one.  Its Coulomb half still
    comes from ``_DFHF.get_j``, i.e. the three-centre fit; nothing about the
    *approximation* changes, only how many times the exchange is built from
    scratch.  ``FASTDFJ_NO_INCREMENTAL=1`` restores upstream's.

    One line of `fastxcnlc` had to move with it.  That module's restricted
    wrapper declines any object carrying a ``with_df``, because on a full-DF
    run ``_DFHF.get_veff`` is the real entry point and fusing the XC and VV10
    grid passes twice would be wrong.  Routing an ``only_dfj`` object to
    ``RKS.get_veff`` walked into that guard and a RIJONX wB97M-V run lost the
    fused sweep, so ``only_dfj`` is named there as the exception --
    ``tests/test_dfj.py`` asserts the fusion happens rather than trusting it.

    ROHF and GHF keep ``_DFHF.get_veff``: the first rewrites the density into
    a two-row array before it calls ``nr_uks`` and the second is complex, and
    neither has been measured here.
    """
    if (_OFF or _NO_INCREMENTAL or not getattr(self, 'only_dfj', False)
            or isinstance(self, (ghf.GHF, rohf.ROHF))):
        STATS['veff_passthrough'] += 1
        return _orig_get_veff(self, mol, dm, dm_last, vhf_last, hermi)
    if mol is None:
        mol = self.mol
    if dm is None:
        dm = self.make_rdm1()
    # looked up on the class at call time, not bound at import, so that
    # `fastxcnlc`'s patch of the same three names is the one that runs
    if isinstance(self, rks.KohnShamDFT):
        driver = (uks.UKS if isinstance(self, uhf.UHF) else rks.RKS).get_veff
        STATS['veff'] += 1
        return driver(self, mol, dm, dm_last, vhf_last, hermi)
    # plain HF: `scf/hf.py:get_veff` gates the incremental build on
    # `mf.direct_scf`, which `_DFHF.__init__` set to False for the reason the
    # docstring gives.  It is true of this object for the length of one call.
    driver = (uhf.UHF if isinstance(self, uhf.UHF) else hf.RHF).get_veff
    saved = self.direct_scf
    self.direct_scf = True
    STATS['veff'] += 1
    try:
        return driver(self, mol, dm, dm_last, vhf_last, hermi)
    finally:
        self.direct_scf = saved


_WARNED = False


def Gradients(self):
    """`_DFHF.Gradients`, with the one warning `only_dfj` deserves.

    ``df/grad/`` never reads ``only_dfj``: the gradient it returns is the
    derivative of the *full-DF* energy, not of the RIJONX energy the SCF just
    converged.  Repairing that is a different piece of work (it needs the
    direct path's ``_jk_energy_per_atom`` for the exchange term and the DF one
    for the Coulomb term); until it exists the mismatch should be loud.
    """
    global _WARNED
    if getattr(self, 'only_dfj', False) and not _WARNED:
        _WARNED = True
        warnings.warn(
            'only_dfj=True: GPU4PySCF returns the full-DF gradient, which is '
            'not the derivative of the RIJONX energy this object converged. '
            'See electrolites/fastdfj.py.', RuntimeWarning, stacklevel=2)
    return _orig_Gradients(self)


_orig_get_k = DFJK._DFHF.get_k
_orig_get_veff = DFJK._DFHF.get_veff
_orig_Gradients = DFJK._DFHF.Gradients
DFJK._DFHF._get_k_orig = _orig_get_k
DFJK._DFHF._get_veff_orig = _orig_get_veff
DFJK._DFHF._Gradients_orig = _orig_Gradients

DFJK._DFHF.get_jk = get_jk
DFJK._DFHF.get_k = get_k
DFJK._DFHF.get_veff = get_veff
DFJK._DFHF.Gradients = Gradients
