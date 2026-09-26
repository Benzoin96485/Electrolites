"""A range-separated density-fitted SCF with one cderi tensor instead of two.

``docs/DF_WORK_LIST.md`` **W22**; ``docs/README_fastdfrsh.md`` is the round.

A density-fitted range-separated hybrid asks ``_DFHF.get_veff`` for

    vj, vk = get_jk(dm)                 # the full-range tensor: J and K(1/r)
    vk    *= hyb
    vk    += (alpha - hyb) * get_k(dm, omega)   # a second, long-range tensor

and ``DF.range_coulomb`` keeps the long-range tensor next to the full-range one
for the rest of the SCF.  W22 found the pair on the host and asked whether the
second tensor has to be a second tensor.  Measured, three things:

1. **The pair is not two tensors of one size.**  The long-range metric
   ``(P|erf|Q)`` is numerically singular in any auxiliary basis built for
   ``1/r``, GPU4PySCF's Cholesky fails on it on every system measured, and its
   eigen route keeps only the modes above ``LINEAR_DEP_THR``: 26-46 % of the
   rows.  At 284 atoms the pair is 86.4 + 26.5 GB, not 2 x 86.4.

2. **A long-range-only functional pays for a tensor it multiplies by zero.**
   ωB97, LC-ωPBE, LRC-ωPBE and LC-VV10 have ``hyb = 0``, and ``get_veff``
   still builds the full-range tensor, streams it through a full exchange
   pass every cycle, and scales the result by ``0``.  The Coulomb matrix is
   the only thing it is used for, and the three-centre engine
   (``df_jk.get_j``, `fastdfj3c`) computes the same fitted J without any
   tensor at all.  So for those functionals this module builds **only** the
   long-range tensor -- the same one, through the same call -- and takes J
   from the engine.  That is the same calculation: nothing in it is a new
   approximation, and it is on by default.

3. **One tensor fits the functional's own operator as well as two fit its
   parts.**  For ``hyb > 0`` the exchange operator is
   ``g = hyb/r + (alpha - hyb) erf(omega r)/r``, which is positive definite
   whenever ``hyb >= 0`` and ``alpha >= 0``.  Fitting ``g`` itself -- its own
   three-centre integrals, its own metric -- is a robust fit of exactly the
   exchange the functional asks for, in *one* full-rank tensor and *one*
   exchange pass, where upstream fits the two parts separately.  It is not
   the same number: the two fits differ by 2e-7 Eh on glycine against a
   fitting error of 7e-5 that both share (``docs/README_fastdfrsh.md`` §4).
   So it is **off** unless asked for, the way ``FASTDF_FP32`` is.

Switches:

    FASTDFRSH=auto     (default) item 2 only: exact, and only where hyb == 0
    FASTDFRSH=fuse     item 2, and item 3 for every hybrid it is defined for
    FASTDFRSH=off      hand every call to GPU4PySCF

What it does not cover, and hands back: GHF and DHF, ``only_dfj`` (RIJONX,
which is `fastdfj`'s), a ``DF`` object that is not GPU4PySCF's own class (the
local fitting of `dflocal`, ``DF4C``), more than one device, and item 3
without `fastdfcderi` loaded -- the fused build lives there and this module
does not import it, for the reason ``docs/DF_WORK_LIST.md`` §5.12 gives.

*The gradient and the response do not follow item 3.*  Both reset or bypass
this path and use upstream's two tensors, so under ``FASTDFRSH=fuse`` they
are the derivative of a slightly different energy -- by the same 1e-7 Eh
margin.  Item 2 has no such caveat: its energy *is* upstream's.
"""
import os
import sys

from gpu4pyscf.df import df as DFMOD
from gpu4pyscf.df import df_jk as DFJK
from gpu4pyscf.dft import rks as _rks
from gpu4pyscf.lib import logger, multi_gpu
from gpu4pyscf.scf import ghf

_MODE = os.environ.get('FASTDFRSH', 'auto').lower()
if _MODE not in ('auto', 'fuse', 'off'):
    raise ValueError(f'FASTDFRSH={_MODE!r}: expected auto, fuse or off')

STATS = {'lr_only': 0, 'fused': 0, 'declined': 0, 'fused_builds': 0}
DECLINED = {}
_WARNED = set()


def stats():
    """What this module did in this process.  Nothing numerical can say."""
    out = dict(STATS)
    out['mode'] = _MODE
    out['declined_reasons'] = dict(DECLINED)
    return out


def _decline(reason):
    STATS['declined'] += 1
    DECLINED[reason] = DECLINED.get(reason, 0) + 1
    return None


def _plan(ks):
    """``(kind, omega, alpha, hyb)`` for a call this module takes, or None."""
    if _MODE == 'off':
        return None
    if not isinstance(ks, _rks.KohnShamDFT):
        return None
    ni = ks._numint
    if not ni.libxc.is_hybrid_xc(ks.xc):
        return None
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=ks.mol.spin)
    if omega == 0 or alpha == hyb:
        return None                          # not range-separated
    if isinstance(ks, ghf.GHF):
        return _decline('GHF')
    if getattr(ks, 'only_dfj', False):
        return _decline('only_dfj')
    if type(ks.with_df) is not DFMOD.DF:
        return _decline('not GPU4PySCF\'s own DF object')
    if multi_gpu.num_devices > 1:
        return _decline('more than one device')
    if omega < 0:
        return _decline('short-range omega')
    if hyb == 0:
        return 'lr_only', omega, alpha, hyb
    if _MODE != 'fuse':
        return None
    if hyb < 0 or alpha < 0:
        # `hyb/r + (alpha-hyb) erf/r` is 4pi/k^2 (hyb + (alpha-hyb) e^{..})
        # in reciprocal space: positive for every k only when both ends are
        return _decline('operator not positive definite')
    fc = sys.modules.get('electrolites.fastdfcderi')
    if fc is None or fc._OFF:
        return _decline('fused build needs fastdfcderi')
    return 'fused', omega, alpha, hyb


class _FusedDF(DFMOD.DF):
    """A ``DF`` object whose tensor fits ``a_full/r + a_lr erf(w r)/r``."""

    fuse = None

    def build(self, direct_scf_tol=None, omega=None, build_cderi=True):
        from electrolites import fastdfcderi as FC
        from pyscf.df import addons
        from gpu4pyscf.df import int3c2e_bdiv
        from gpu4pyscf.gto.mole import SortedMole
        assert omega is None
        mol = self.mol
        log = logger.new_logger(mol, mol.verbose)
        t0 = log.init_timer()
        if self.auxmol is None:
            self.auxmol = addons.make_auxmol(mol, self.auxbasis)
        self.nao = mol.nao
        self.naux = self.auxmol.nao
        self.intopt = intopt = int3c2e_bdiv.Int3c2eOpt(mol, self.auxmol)
        intopt.mol = SortedMole.from_mol(mol, decontract=True)
        intopt.build()
        if build_cderi:
            self._cderi, self._cderi_idx = FC._cholesky_eri(
                intopt, use_gpu_memory=self.use_gpu_memory, fuse=self.fuse)
            STATS['fused_builds'] += 1
            log.timer_debug1('cholesky_eri (fused)', *t0)
        return self


def _fused_df(with_df, omega, alpha, hyb):
    """The fused object, kept where ``DF.reset`` will drop it."""
    key = f'fused/{omega:.6f}/{hyb:.8f}/{alpha:.8f}'
    fdf = with_df._rsh_df.get(key)
    if fdf is None:
        fdf = _FusedDF(with_df.mol, auxbasis=with_df.auxbasis)
        fdf.auxmol = with_df.auxmol
        fdf.use_gpu_memory = with_df.use_gpu_memory
        fdf.verbose = with_df.verbose
        fdf.stdout = with_df.stdout
        fdf.max_memory = with_df.max_memory
        fdf.fuse = (hyb, alpha - hyb, omega)
        with_df._rsh_df[key] = fdf
    return fdf


_orig_get_veff = None


def get_veff(ks, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
    """``_DFHF.get_veff`` with the exchange of a range-separated hybrid built
    from one tensor.

    The body stays upstream's -- the grid pass, the ``ecoul``/``exc``
    bookkeeping, the returned ``tag_array`` -- and two things are answered for
    the length of the call: ``get_jk`` returns J from the three-centre engine
    and the functional's *whole* exchange from one tensor, and
    ``rsh_and_hybrid_coeff`` tells the body that exchange is already scaled
    (``omega = 0``, ``hyb = 1``), so it neither rescales it nor asks for the
    long-range part a second time.
    """
    plan = _plan(ks)
    if plan is None:
        return _orig_get_veff(ks, mol, dm, dm_last, vhf_last, hermi)
    kind, omega, alpha, hyb = plan
    STATS[kind] += 1
    with_df = ks.with_df
    if kind == 'fused' and 'fused' not in _WARNED:
        _WARNED.add('fused')
        logger.note(ks, 'FASTDFRSH=fuse: the exchange of %s is fitted as one '
                    'operator; gradients and response use the two-tensor '
                    'fit and differ from this energy by the fitting '
                    'difference (docs/README_fastdfrsh.md)', ks.xc)

    omega_lr = omega

    def get_jk(mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        # only the body's own hybrid call reaches this, and it names no omega
        assert omega is None
        vj = vk = None
        if with_j:
            # `DF.get_jk` hands a J-only request with no tensor built to
            # `df_jk.get_j`, the three-centre engine
            vj = with_df.get_jk(dm, hermi, with_j=True, with_k=False)[0]
        if with_k:
            if kind == 'lr_only':
                # upstream's own long-range call, the one `_DFHF.get_k` makes
                vk = with_df.get_jk(dm, hermi, with_j=False, with_k=True,
                                    omega=omega_lr)[1]
                vk *= alpha
            else:
                fdf = _fused_df(with_df, omega_lr, alpha, hyb)
                vk = fdf.get_jk(dm, hermi, with_j=False, with_k=True)[1]
        return vj, vk

    ni = ks._numint
    had_jk = 'get_jk' in ks.__dict__
    old_jk = ks.__dict__.get('get_jk')
    had_rsh = 'rsh_and_hybrid_coeff' in ni.__dict__
    old_rsh = ni.__dict__.get('rsh_and_hybrid_coeff')
    ks.get_jk = get_jk
    ni.rsh_and_hybrid_coeff = lambda *args, **kw: (0.0, 1.0, 1.0)
    try:
        return _orig_get_veff(ks, mol, dm, dm_last, vhf_last, hermi)
    finally:
        if had_jk:
            ks.__dict__['get_jk'] = old_jk
        else:
            ks.__dict__.pop('get_jk', None)
        if had_rsh:
            ni.__dict__['rsh_and_hybrid_coeff'] = old_rsh
        else:
            ni.__dict__.pop('rsh_and_hybrid_coeff', None)


def apply_patch():
    global _orig_get_veff
    if _orig_get_veff is not None:
        return
    # whatever is there -- `fastxcnlc`'s fused grid pass under `patch_all()`,
    # which then runs inside this one
    _orig_get_veff = DFJK._DFHF.get_veff
    DFJK._DFHF.get_veff = get_veff


apply_patch()
