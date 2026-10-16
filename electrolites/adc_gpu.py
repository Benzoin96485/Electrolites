"""Closed-shell ADC(2) excitation, ionisation and attachment energies on the
GPU.

GPU4PySCF has no ADC.  PySCF's restricted ADC (``pyscf.adc.radc*``) is a
few thousand lines of ``lib.einsum`` over NumPy arrays, on the host, after
an AO->MO transformation on the host.  Here its own source runs with CuPy
as the array module: ``radc_amplitudes``, ``radc_ee``, ``radc_ip`` and
``radc_ea`` are compiled again into private namespaces whose ``np`` is
CuPy, whose ``lib.einsum`` is ``cupy.einsum`` and whose ``unpack_eri_1`` is
the identity (``ovvv`` is kept unpacked).  The algebra is then PySCF's
term for term -- the same intermediates, the same M_ab, the same sigma
vector -- with every array on the device; PySCF's Davidson
(``lib.linalg_helper.davidson_nosym1``) runs unchanged on the host with
the sigma vectors converted at the boundary.  The MO integrals come from
``fastmp2``'s half transformation (exact, as PySCF's incore ones).

ADC(2) for EE (singlets), IP and EA.  ADC(2)-x and ADC(3) need the vvvv
ladder and are not enabled here; nor are the transition properties
(``compute_properties`` is off).

    from electrolites import adc_gpu
    e, v, p, x = adc_gpu.RADC(gpu_mf, method_type='ee').kernel(nroots=5)
"""
import inspect
import types

import numpy
import cupy as cp

from pyscf import lib as _pylib
from pyscf.adc import radc as _radc
from pyscf.adc import radc_amplitudes as _amp
from pyscf.adc import radc_ee as _ee
from pyscf.adc import radc_ip as _ip
from pyscf.adc import radc_ea as _ea
from pyscf.lib import logger


class _CuPyAsNumPy(types.ModuleType):
    """``np`` for the transplanted modules: CuPy, with NumPy's dtypes and
    integer scalar types for ``isinstance`` checks."""

    def __init__(self):
        super().__init__('numpy_on_cupy')

    def __getattr__(self, name):
        if name in ('integer', 'floating', 'complexfloating', 'double',
                    'float64', 'int64', 'int32', 'bool_'):
            return getattr(numpy, name)
        return getattr(cp, name)


def _einsum(subscripts, *ops, **kw):
    kw.pop('optimize', None)
    ops = [cp.asarray(o) for o in ops]
    return cp.einsum(subscripts, *ops)


class _Lib(types.ModuleType):
    def __init__(self):
        super().__init__('pyscf_lib_on_cupy')
        self.einsum = _einsum

    def __getattr__(self, name):
        return getattr(_pylib, name)


class _AO2MO(types.ModuleType):
    def __init__(self):
        super().__init__('radc_ao2mo_on_cupy')

    @staticmethod
    def unpack_eri_1(eri, nvir):
        return eri

    def __getattr__(self, name):
        from pyscf.adc import radc_ao2mo
        return getattr(radc_ao2mo, name)


_NP, _LIB, _A2M = _CuPyAsNumPy(), _Lib(), _AO2MO()


def _transplant(mod, extra=None):
    src = inspect.getsource(mod)
    ns = {'__name__': 'electrolites.adc_gpu.' + mod.__name__.rsplit('.')[-1],
          '__file__': mod.__file__}
    exec(compile(src, mod.__file__, 'exec'), ns)
    ns.update(np=_NP, lib=_LIB, radc_ao2mo=_A2M)
    if extra:
        ns.update(extra)
    return types.SimpleNamespace(**ns)


AMP = _transplant(_amp)
_extra = {'radc_amplitudes': AMP}
EE = _transplant(_ee, _extra)
IP = _transplant(_ip, _extra)
EA = _transplant(_ea, _extra)


class _ERIs:
    """oooo, ovoo, oovv, ovvo, ovvv (unpacked) on the device."""

    def __init__(self, adc):
        from electrolites import fastmp2
        mo = cp.asarray(adc.mo_coeff)
        no = adc._nocc
        orbo, orbv = mo[:, :no], mo[:, no:]
        mol = adc.mol
        self.ovvv = cp.asarray(fastmp2.get_int4c2e_ovpq(mol, orbo, orbv,
                                                        orbv, orbv))
        ovon = cp.asarray(fastmp2.get_int4c2e_ovpq(mol, orbo, orbv, orbo, mo))
        self.ovoo = cp.ascontiguousarray(ovon[:, :, :, :no])
        self.ovvo = cp.ascontiguousarray(
            ovon[:, :, :, no:].transpose(0, 1, 3, 2))
        ovon = None
        oonn = cp.asarray(fastmp2.get_int4c2e_ovpq(mol, orbo, orbo, mo, mo))
        self.oooo = cp.ascontiguousarray(oonn[:, :, :no, :no])
        self.oovv = cp.ascontiguousarray(oonn[:, :, no:, no:])
        oonn = None
        self.vvvv = None
        self.Lov = self.Lvv = None


def _es_class(kind):
    mod = {'ee': EE, 'ip': IP, 'ea': EA}[kind]
    base = getattr(mod, 'RADC' + kind.upper())

    class GPU(base):
        compute_properties = False

        def gen_matvec(self, imds=None, eris=None):
            if imds is None:
                imds = self.get_imds(eris)
            diag = self.get_diag(imds, eris)
            sigma = self.matvec(imds, eris)

            def matvec(x):
                return cp.asnumpy(sigma(cp.asarray(x)))

            return matvec, cp.asnumpy(diag)

        def get_init_guess(self, nroots=1, diag=None, ascending=True,
                           type=None, eris=None, ini=None):
            if type not in (None, 'read'):
                raise NotImplementedError(f'ADC guess {type!r} on the GPU')
            return _radc_init_guess(self, nroots, diag, ascending, type, ini)

    GPU.__name__ = base.__name__
    return GPU


def _radc_init_guess(es, nroots, diag, ascending, type, ini):
    """PySCF's default guess (lowest diagonal elements) on host arrays."""
    if type == 'read':
        return list(numpy.asarray(ini))
    diag = numpy.asarray(diag)
    idx = numpy.argsort(diag) if ascending else numpy.argsort(diag)[::-1]
    guess = numpy.zeros((len(diag), nroots))
    guess[idx[:nroots], numpy.arange(nroots)] = 1.
    return list(guess.T)


class RADC(_radc.RADC):
    """PySCF's ``RADC`` (ADC(2): EE, IP, EA) with integrals, amplitudes and
    sigma vectors on the device."""

    compute_amplitudes = AMP.compute_amplitudes
    compute_energy = AMP.compute_energy

    def __init__(self, mf, frozen=None, method_type='ee', method='adc(2)'):
        if getattr(mf, 'device', 'cpu') == 'gpu':
            mf = mf.to_cpu()
        _radc.RADC.__init__(self, mf, frozen=frozen)
        self.method_type = method_type
        self.method = method
        self.compute_properties = False
        self.approx_trans_moments = True

    def kernel(self, nroots=1, guess=None, eris=None):
        self.method = self.method.lower()
        if self.method != 'adc(2)':
            raise NotImplementedError('GPU ADC: adc(2) only')
        log = logger.new_logger(self)
        if eris is None:
            eris = _ERIs(self)
        host_e = self.mo_energy
        self.mo_energy = cp.asarray(host_e)
        try:
            self.e_corr, self.t1, self.t2 = AMP.compute_amplitudes_energy(
                self, eris=eris, verbose=self.verbose)
            self.e_corr = float(self.e_corr)
            log.note('MP2 correlation energy of reference state (a.u.) = '
                     '%.8f', self.e_corr)
            es = _es_class(self.method_type.lower())(self)
            es.compute_properties = False
            e, v, p, x = es.kernel(nroots, guess, eris)
        finally:
            self.mo_energy = host_e
        self._adc_es = es
        return e, v, p, x
