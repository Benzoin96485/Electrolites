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
vector -- with every array on the device.  The eigensolver is a device
block Davidson with PySCF's convergence test and preconditioner: PySCF's
host ``davidson_nosym1`` on o^2 v^2-long vectors was 95 % of an ADC(2)-EE
run (alanine dipeptide: 400 of 425 s in its subspace projections).  The MO integrals come from
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


PROF = {}


def _einsum(subscripts, *ops, **kw):
    kw.pop('optimize', None)
    ops = [cp.asarray(o) for o in ops]
    if PROF.get('on'):
        import time
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        out = cp.einsum(subscripts, *ops)
        cp.cuda.Stream.null.synchronize()
        key = (subscripts, tuple(o.shape for o in ops))
        PROF[key] = PROF.get(key, 0.) + time.perf_counter() - t
        return out
    return cp.einsum(subscripts, *ops)


def _direct_sum(subscripts, *ops):
    """``lib.direct_sum`` on device arrays ('-i+jab->ijab' & co.)."""
    import re
    lhs, rhs = subscripts.replace(' ', '').split('->')
    terms = re.findall(r'([+-]?)([A-Za-z]+)', lhs)
    out = 0
    for (sign, sub), op in zip(terms, ops):
        op = cp.asarray(op)
        order = ''.join(c for c in rhs if c in sub)
        arr = cp.einsum(sub + '->' + order, op)
        shape = [arr.shape[order.index(c)] if c in sub else 1 for c in rhs]
        arr = arr.reshape(shape)
        out = out - arr if sign == '-' else out + arr
    return out


class _DictH5(dict):
    """``lib.H5TmpFile`` stand-in holding device arrays."""

    def create_dataset(self, name, shape=None, dtype='d', data=None, **kw):
        arr = cp.zeros(shape, dtype) if data is None else cp.asarray(data)
        self[name] = arr
        return arr

    def flush(self):
        pass


class _Lib(types.ModuleType):
    def __init__(self):
        super().__init__('pyscf_lib_on_cupy')
        self.einsum = _einsum
        self.direct_sum = _direct_sum
        self.H5TmpFile = _DictH5

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

        def kernel(self, nroots=1, guess=None, eris=None):
            return _es_kernel(self, nroots, guess, eris)

    GPU.__name__ = base.__name__
    return GPU


def _davidson(matvec, x0, diag, nroots, tol, tol_residual, max_cycle,
              max_space, log):
    """Block Davidson for the lowest roots of the ADC matrix on the device.
    PySCF's ADC matrix is not symmetric in its spin-adapted doubles basis,
    so, as ``davidson_nosym1``: the projected matrix is diagonalised as a
    general one (on the host: it is at most max_space wide), complex pairs
    are dropped, the right Ritz vectors are kept, PySCF's convergence test
    (energy change < tol and residual < tol_residual) and diagonal
    preconditioner."""
    def orth(vs, basis):
        out = []
        for v in vs:
            for _ in range(2):
                if basis is not None and len(basis):
                    v = v - basis.T @ (basis @ v)
                for u in out:
                    v = v - u * (u @ v)
            nrm = cp.linalg.norm(v)
            if nrm > 1e-7:
                out.append(v / nrm)
        return out

    nwant = len(x0)              # roots tracked (nroots + spare guesses)
    V = cp.stack(orth(list(x0), None))
    # PySCF's sigma functions may work in place on their argument: copies
    AV = cp.stack([matvec(v.copy()) for v in V])
    e_last = None
    conv = numpy.zeros(nroots, dtype=bool)
    X = V[:nroots]
    for it in range(max_cycle):
        H = cp.asnumpy(V @ AV.T)
        w, c = numpy.linalg.eig(H)
        real = numpy.where(abs(w.imag) < 1e-3)[0]
        w, c = w[real].real, c[:, real].real
        order = numpy.argsort(w)[:nwant]
        w, c = w[order], cp.asarray(c[:, order])
        X = c.T @ V
        AX = c.T @ AV
        nrm = cp.linalg.norm(X, axis=1)
        X /= nrm[:, None]
        AX /= nrm[:, None]
        wd = cp.asarray(w)
        R = AX - wd[:, None] * X
        rnorm = cp.asnumpy(cp.linalg.norm(R, axis=1))
        if e_last is not None and len(e_last) == len(w):
            de = abs(w - e_last)
        else:
            de = numpy.ones(len(w))
        conv = (de < tol) & (rnorm < tol_residual)
        log.debug('ADC Davidson %d  space %d  |r| %s  dE %s', it, len(V),
                  rnorm, de)
        e_last = w
        if conv[:nroots].all():
            break
        new = []
        # corrections for the wanted roots only; the spare Ritz vectors stay
        # in the space (and survive restarts) so that a lower root missed
        # by the guess can still enter
        for k in numpy.where(~conv[:nroots])[0]:
            d = wd[k] - diag
            d = cp.where(abs(d) < 1e-8, 1e-8, d)
            new.append(R[k] / d)
        if len(V) + len(new) > max_space:
            # restart on the Ritz vectors, orthonormalised
            q, r_ = cp.linalg.qr(X.T)
            V = q.T
            AV = cp.linalg.solve(r_.T, AX)
        new = orth(new, V)
        if not new:
            break
        Vn = cp.stack(new)
        AVn = cp.stack([matvec(v.copy()) for v in Vn])
        V = cp.vstack([V, Vn])
        AV = cp.vstack([AV, AVn])
    return conv[:nroots], e_last[:nroots], cp.asnumpy(X[:nroots])


def _es_kernel(es, nroots, guess, eris):
    from electrolites.eom_gpu import _on_pool
    return _on_pool(_es_kernel_impl)(es, nroots, guess, eris)


def _es_kernel_impl(es, nroots, guess, eris):
    """``radc.kernel`` with the device Davidson (PySCF's host Davidson on
    o^2 v^2 vectors was 95 % of an ADC(2)-EE run)."""
    from pyscf.data.nist import HARTREE2EV
    log = logger.new_logger(es)
    imds = es.get_imds(eris)
    diag = cp.asarray(es.get_diag(imds, eris))
    sigma = es.matvec(imds, eris)
    # spare guesses: a nonsymmetric Davidson started on exactly nroots unit
    # vectors can converge onto a higher root (water EA, 4th root)
    nwant = min(len(diag), nroots + max(3, nroots // 2))
    if guess is None:
        idx = cp.argsort(diag)[:nwant]
        x0 = cp.zeros((nwant, len(diag)))
        x0[cp.arange(nwant), idx] = 1.
    else:
        x0 = cp.asarray(numpy.asarray(guess)).reshape(nroots, -1)
    # trial vectors and their products live on the device: cap the space
    # by the free memory (two arrays of max_space vectors)
    free = cp.cuda.runtime.memGetInfo()[0] + \
        cp.get_default_memory_pool().free_bytes()
    room = int(.5 * free // (2 * 8 * len(diag)))
    max_space = max(es.max_space, 4 * nroots) + nwant
    if room < max_space:
        if room < 2 * nwant:
            raise MemoryError(f'ADC Davidson: {room} trial vectors of '
                              f'{len(diag)} fit on the device')
        log.warn('ADC Davidson: subspace capped at %d vectors (memory)', room)
        max_space = room
    conv, e, U = _davidson(sigma, x0, diag, nroots, es.conv_tol,
                           es.tol_residual, es.max_cycle, max_space, log)
    es.E, es.U = e, U.T.copy()
    es.P = es.X = None
    for n in range(nroots):
        log.info('%s root %d  |  Energy (Eh) = %14.10f  |  Energy (eV) = '
                 '%12.8f  |  conv = %s', es.method, n, e[n],
                 e[n] * HARTREE2EV, conv[n])
    if not conv.all():
        log.warn('ADC Davidson: %d root(s) did not converge',
                 int((~conv).sum()))
    return es.E, es.U, es.P, es.X


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
