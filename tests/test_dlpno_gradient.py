"""Analytic response versus independent changes of the underlying problem."""
import numpy as np
import pytest

from pyscf import gto, scf, df

from electrolites.dlpno.dual import Dual, sym_power, eigh
from electrolites.dlpno.reference_cpu import CPUDLPNO, orbitals
from electrolites.dlpno.setup import count_frozen_core
from electrolites.dlpno.orbital_response import (
    RHFResponse, one_center_derivatives, three_center_derivatives,
)
from electrolites.dlpno.response import ResponseDLPNO


WATER = 'O 0 0 0; H 0 .76 .59; H .07 -.75 .57'
STRESS = dict(T_CUT_PNO=1e-4, T_CUT_PNO_MP2=1e-5, T_CUT_TRACE=0.,
              T_CUT_ENERGY=0., T_CUT_TRACE_MP2=0., T_CUT_ENERGY_MP2=0.,
              MIN_PNOS=1, MIN_TNOS=1, T_CUT_TNO=1e-5, T_CUT_TNO_PRE=1e-4)
TIGHT = dict(E_CONVERGENCE=1e-11, R_CONVERGENCE=1e-10, MAXITER=150, T_CUT_ITER=0.)


def _displace(mol, step):
    xyz = mol.atom_coords().copy()
    xyz[1, 2] += step
    return mol.copy().set_geom_(xyz, unit='Bohr')


def _energy(mol, triples='t1', **kwargs):
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
    mf.kernel()
    assert mf.converged
    d = CPUDLPNO(mf, triples=triples, verbose=0, **TIGHT, **kwargs)
    d.kernel()
    return d


def _require_cuda():
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('A CUDA device is required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('A CUDA device is required')


def test_ao_and_auxiliary_center_derivatives():
    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    aux = df.addons.make_auxmol(mol, df.make_auxbasis(mol, mp2fit=True))
    one = list(one_center_derivatives(mol))[5]
    three = list(three_center_derivatives(mol, aux))[5]
    analytic = [one[2], one[3], three[2], three[3]]
    h = 5e-5
    values = []
    for sign in (1, -1):
        m = _displace(mol, sign * h)
        a = df.addons.make_auxmol(m, aux.basis)
        values.append([m.intor('int1e_ovlp'), m.intor('int1e_r', comp=3),
                       df.incore.aux_e2(m, a, 'int3c2e'), a.intor('int2c2e')])
    for g, plus, minus in zip(analytic, *values):
        assert np.max(abs(g - (plus - minus) / (2 * h))) < 2e-8
    # Translating every center must leave overlap and Coulomb integrals fixed.
    assert np.max(abs(sum(x[2] for x in one_center_derivatives(mol) if x[1] == 0))) < 1e-12
    assert np.max(abs(sum(x[2] for x in three_center_derivatives(mol, aux) if x[1] == 0))) < 1e-11
    assert np.max(abs(sum(x[3] for x in three_center_derivatives(mol, aux) if x[1] == 0))) < 1e-11


def test_metric_response_at_degenerate_eigenvalues():
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    a = (q * np.array([1., 1., 2., 3., 3.])) @ q.T
    da = rng.normal(size=(5, 5))
    da = (da + da.T) * .5
    response = sym_power(Dual(a, da), -.5)
    h = 1e-5
    fd = (sym_power(a + h * da, -.5) - sym_power(a - h * da, -.5)) / (2 * h)
    assert np.max(abs(response.d - fd)) < 1e-8


def test_degenerate_subspace_projector_and_undefined_cut():
    a = np.diag([1., 1., 3., 5.])
    da = np.array([[.2, .4, -.3, .1], [.4, -.1, .7, .2],
                   [-.3, .7, .1, -.2], [.1, .2, -.2, .3]])
    _, u = eigh(Dual(a, da))
    projector = u[:, :2] @ u[:, :2].T
    h = 1e-5
    projected = []
    for sign in (1, -1):
        _, v = np.linalg.eigh(a + sign * h * da)
        projected.append(v[:, :2] @ v[:, :2].T)
    assert np.max(abs(projector.d - (projected[0] - projected[1]) / (2 * h))) < 1e-8
    with pytest.raises(ValueError, match='degenerate'):
        _ = u[:, :1]


@pytest.mark.parametrize('triples', ['t0', 't1'])
@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
def test_truncated_pipeline_against_independent_nuclear_difference(triples, backend):
    if backend == 'cupy':
        _require_cuda()
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    base = _energy(mol, triples=triples, **STRESS)
    nv = mol.nao - mol.nelectron // 2
    assert min(x.shape[1] for x in base.X) < nv
    assert min(x.shape[1] for x in base.X_tno) < nv
    orbital = RHFResponse(base)
    direction = next(d for d in orbital.directions() if (d.atom, d.xyz) == (1, 2))
    response = ResponseDLPNO(base, orbital, direction, 'ccsd_t', backend=backend)
    energy = response.kernel()
    assert abs(float(energy.v) - base.e['corr']) < 1e-9
    h = 1e-4
    plus = _energy(_displace(mol, h), triples=triples, **STRESS)
    minus = _energy(_displace(mol, -h), triples=triples, **STRESS)
    assert [x.shape[1] for x in plus.X] == [x.shape[1] for x in base.X] == [x.shape[1] for x in minus.X]
    assert plus.trip == base.trip == minus.trip
    for key, value in response.e.items():
        fd = (plus.e[key] - minus.e[key]) / (2 * h)
        assert abs(float(getattr(value, 'd', 0.)) - fd) < 2e-7, key
    fd = (plus.e['tot'] - minus.e['tot']) / (2 * h)
    assert abs(float(energy.d) + orbital.hf_gradient[1, 2] - fd) < 2e-7


@pytest.mark.parametrize('method', ['direct', 'adjoint'])
def test_public_gradient_interface_and_mp2_level(method):
    mol = gto.M(atom='H 0 0 0; H .12 .07 .73', basis='sto-3g', verbose=0)
    base = _energy(mol, triples=None)
    g = base.nuc_grad_method(method=method, level='mp2').kernel()
    assert g.shape == (2, 3)
    assert np.max(abs(g.sum(axis=0))) < 1e-9
    assert np.max(abs(np.cross(mol.atom_coords(), g).sum(axis=0))) < 1e-9
    with pytest.raises(ValueError, match='atmlst'):
        base.nuc_grad_method().kernel(atmlst=[0, 0])


def test_gpu_feature_gradient_with_df_scf():
    """Exercise the public adapter, GPU SCF conversion, and atom selection."""
    _require_cuda()
    import electrolites
    from gpu4pyscf import scf as gscf
    electrolites.enable('dlpno')

    def calculate(mol):
        mf = gscf.RHF(mol).density_fit()
        mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
        mf.max_cycle = 150
        mf.kernel()
        assert mf.converged
        cpu = mf.to_cpu()
        orb = orbitals(mol, cpu.get_ovlp(), cpu.get_fock(), cpu.mo_coeff,
                       mol.nelectron // 2, count_frozen_core(mol))
        return mf.DLPNO_CCSD_T(triples='t1', c_lmo=orb.C_lmo, verbose=0, **TIGHT)

    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    base = calculate(mol)
    grad = base.nuc_grad_method(backend='cupy')
    value = grad.kernel(atmlst=[1])  # Also builds an unrun energy object.
    assert value.shape == (1, 3)
    assert len(grad.diagnostics) == 3
    assert max(abs(x['energy_error']) for x in grad.diagnostics) < 1e-8
    h = 5e-4
    plus, minus = calculate(_displace(mol, h)), calculate(_displace(mol, -h))
    finite_difference = (plus.kernel() - minus.kernel()) / (2 * h)
    assert abs(value[0, 2] - finite_difference) < 5e-7


@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
def test_empty_triples_gradient(backend):
    cls = CPUDLPNO
    if backend == 'cupy':
        _require_cuda()
        from electrolites.dlpno.rhf import DLPNO
        cls = DLPNO
    mol = gto.M(atom='H 0 0 0; H .12 .07 .73', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
    mf.kernel()
    base = cls(mf, triples='t1', verbose=0, **TIGHT)
    base.kernel()
    assert base.trip == []
    assert base.e['t'] == 0.
    gradient = base.nuc_grad_method(backend=backend).kernel()
    assert np.max(abs(gradient.sum(axis=0))) < 1e-9
    h = 1e-4
    plus, minus = _energy(_displace(mol, h), triples=None), _energy(_displace(mol, -h), triples=None)
    fd = (plus.e['tot'] - minus.e['tot']) / (2 * h)
    assert abs(gradient[1, 2] - fd) < 2e-7
