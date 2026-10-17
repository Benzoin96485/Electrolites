"""CCSD one- and two-particle density matrices on the GPU (closed shell).

GPU4PySCF's ``CCSD.make_rdm1/make_rdm2`` are ``NotImplemented``.  PySCF's
``cc.ccsd_rdm`` -- the γ intermediates and their assembly -- runs here with
CuPy as its array module (``_gamma1_intermediates``, ``_gamma2_outcore``
with its HDF5 scratch replaced by device arrays, ``_make_rdm2``), from the
amplitudes of ``fastccsd`` and the λ of ``ccsd_lambda_gpu``.  Same
conventions as PySCF (spin-traced; dm2 in chemists' order,
E = einsum('pqrs,pqrs', eri, dm2); the vvvv block symmetrised).  Device
arrays are returned.  The 2-RDM is a dense nmo^4 array (with its v^4
intermediate), so it is limited by device memory.

    dm1 = mycc.make_rdm1()          # solves Λ first if mycc.l1 is None
    dm2 = mycc.make_rdm2()
"""
import numpy as np
import cupy as cp

_RDM = None


def _mod():
    global _RDM
    if _RDM is None:
        from pyscf.cc import ccsd_rdm
        from electrolites.adc_gpu import _transplant, _NP
        _RDM = _transplant(ccsd_rdm, {'numpy': _NP})
    return _RDM


def _amps(mycc, t1, t2, l1, l2):
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    if l1 is None or l2 is None:
        if getattr(mycc, 'l1', None) is None:
            from electrolites import ccsd_lambda_gpu
            ccsd_lambda_gpu.kernel(mycc, tol=mycc.conv_tol_normt)
        l1, l2 = mycc.l1, mycc.l2
    return [cp.asarray(x) for x in (t1, t2, l1, l2)]


def _embed(mycc, dm, nfull_occ, ndim, with_mf):
    """Put the active-space dm into the full MO space (frozen orbitals)."""
    nmo = mycc.mo_occ.size
    idx = cp.asarray(np.where(mycc.get_frozen_mask())[0])
    if ndim == 2:
        out = cp.zeros((nmo, nmo))
        if with_mf:
            d = cp.arange(nfull_occ)
            out[d, d] = 2
        out[idx[:, None], idx] = dm
        return out
    out = cp.zeros((nmo,) * 4)
    out[idx[:, None, None, None], idx[None, :, None, None],
        idx[None, None, :, None], idx] = dm
    return out


def make_rdm1(*args, **kwargs):
    from electrolites.eom_gpu import _on_pool
    return _on_pool(_make_rdm1)(*args, **kwargs)


def make_rdm2(*args, **kwargs):
    from electrolites.eom_gpu import _on_pool
    return _on_pool(_make_rdm2)(*args, **kwargs)


def _make_rdm1(mycc, t1=None, t2=None, l1=None, l2=None, ao_repr=False,
              with_frozen=True, with_mf=True):
    """Spin-traced CCSD 1-RDM (PySCF ``ccsd_rdm.make_rdm1``), device."""
    m = _mod()
    t1, t2, l1, l2 = _amps(mycc, t1, t2, l1, l2)
    d1 = m._gamma1_intermediates(mycc, t1, t2, l1, l2)
    dm1 = m._make_rdm1(mycc, d1, with_frozen=False, with_mf=with_mf)
    if with_frozen and mycc.frozen is not None:
        dm1 = _embed(mycc, dm1, int((mycc.mo_occ > 0).sum()), 2, with_mf)
    if ao_repr:
        mo = cp.asarray(mycc.mo_coeff)
        dm1 = mo @ dm1 @ mo.T
    return dm1


def _make_rdm2(mycc, t1=None, t2=None, l1=None, l2=None, ao_repr=False,
              with_frozen=True, with_dm1=True):
    """Spin-traced CCSD 2-RDM (PySCF ``ccsd_rdm.make_rdm2``), device."""
    m = _mod()
    t1, t2, l1, l2 = _amps(mycc, t1, t2, l1, l2)
    d1 = m._gamma1_intermediates(mycc, t1, t2, l1, l2)
    from electrolites.adc_gpu import _DictH5
    d2 = m._gamma2_outcore(mycc, t1, t2, l1, l2, _DictH5(), False)
    if with_frozen and mycc.frozen is not None:
        # PySCF adds the reference/1-RDM pieces after embedding; so do we
        dm2 = m._make_rdm2(mycc, d1, d2, with_dm1=False, with_frozen=False)
        dm2 = dm2.transpose(1, 0, 3, 2)          # back to <p+ r+ s q> order
        nocc = int((mycc.mo_occ > 0).sum())
        dm2 = _embed(mycc, dm2, nocc, 4, False)
        if with_dm1:
            dm1 = make_rdm1(mycc, t1, t2, l1, l2, with_frozen=True)
            d = cp.arange(nocc)
            dm1[d, d] -= 2
            for i in range(nocc):
                dm2[i, i, :, :] += dm1 * 2
                dm2[:, :, i, i] += dm1 * 2
                dm2[:, i, i, :] -= dm1
                dm2[i, :, :, i] -= dm1.T
            dm2[d[:, None], d[:, None], d, d] += 4
            dm2[d[:, None], d, d, d[:, None]] -= 2
        dm2 = dm2.transpose(1, 0, 3, 2)
    else:
        dm2 = m._make_rdm2(mycc, d1, d2, with_dm1=with_dm1,
                           with_frozen=False)
    if ao_repr:
        mo = cp.asarray(mycc.mo_coeff)
        dm2 = cp.einsum('pi,ijkl,qj,rk,sl->pqrs', mo, dm2, mo, mo, mo,
                        optimize=True)
    return dm2


def _make_rdm1_method(self, t1=None, t2=None, l1=None, l2=None,
                      ao_repr=False, with_frozen=True, with_mf=True):
    return make_rdm1(self, t1, t2, l1, l2, ao_repr, with_frozen, with_mf)


def _make_rdm2_method(self, t1=None, t2=None, l1=None, l2=None,
                      ao_repr=False, with_frozen=True, with_dm1=True):
    return make_rdm2(self, t1, t2, l1, l2, ao_repr, with_frozen, with_dm1)


from gpu4pyscf.cc import ccsd_incore as _ci                   # noqa: E402
from electrolites.features import attach as _attach           # noqa: E402

for _name, _fn in (('make_rdm1', _make_rdm1_method),
                   ('make_rdm2', _make_rdm2_method)):
    if getattr(_ci.CCSDBase, _name, None) is NotImplemented:
        _attach(_ci.CCSDBase, _name, _fn)
