"""DLPNO-CCSD(T) on the GPU (electrolites.dlpno).

* canonical limit (every threshold off): closed shell = PySCF RCCSD(T),
  open shell = PySCF UCCSD(T) on the semicanonical ROHF determinant, for
  CCSD, (T0)-free iterative (T) to 1e-8 Eh;
* GPU drivers = host references (identical algorithm) at Normal PNO;
* closed-shell molecule through the open-shell code = closed-shell code;
* the feature entry point (mf.DLPNO_CCSD_T).
"""
import numpy as np
import pytest

pytest.importorskip('cupy')
from pyscf import gto, scf, cc
from pyscf.data import elements

FULL = dict(T_CUT_PNO=0., T_CUT_PNO_MP2=0., T_CUT_DO=0., T_CUT_MKN=0., T_CUT_PAIRS=0.,
            T_CUT_DO_IJ=-1., T_CUT_PRE=0., T_CUT_PAIRS_MP2=0., MIN_PNOS=10000,
            E_CONVERGENCE=1e-10, R_CONVERGENCE=1e-9, T_CUT_DO_UV=-1.,
            T_CUT_TNO=0., T_CUT_TNO_PRE=0., T_CUT_DO_TRIPLES=0., T_CUT_DO_TRIPLES_PRE=0.,
            T_CUT_MKN_TRIPLES=0., T_CUT_MKN_TRIPLES_PRE=0., T_CUT_TRIPLES_WEAK=0.,
            F_CUT_T=0., T_CUT_ITER=0., MIN_TNOS=10000, TRIPLES_MAX_WEAK_PAIRS=3)

WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
CH3 = 'C 0 0 0; H 0 1.08 0; H 0.935 -0.54 0; H -0.935 -0.54 0'


def test_closed_canonical_limit():
    from electrolites.dlpno.rhf import DLPNO
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    d = DLPNO(mf, pno='normal', triples='t1', auxbasis='cc-pvdz-ri', **FULL)
    d.kernel()
    mycc = cc.RCCSD(mf).density_fit(auxbasis='cc-pvdz-ri')
    mycc.frozen = elements.chemcore(mol)
    mycc.conv_tol = 1e-11
    mycc.kernel()
    et = mycc.ccsd_t()
    assert abs(d.e['ccsd'] - mycc.e_corr) < 1e-8
    assert abs(d.e['t'] - et) < 1e-8


def _semicanonical_uccsd_t(mf, nc, aux):
    mu = scf.addons.convert_to_uhf(mf)
    F = mu.get_fock(dm=mu.make_rdm1())
    C, occ = np.array(mu.mo_coeff), np.array(mu.mo_occ)
    Cs = []
    for s in range(2):
        o = occ[s] > 0
        core = np.zeros_like(o)
        core[:nc] = True
        blocks = []
        for sel in (core, o & ~core, ~o):
            c = C[s][:, sel]
            w, U = np.linalg.eigh(c.T @ F[s] @ c)
            blocks.append(c @ U)
        Cs.append(np.hstack(blocks))
    mu.mo_coeff = np.array(Cs)
    mu.mo_occ = np.array([np.sort(occ[0])[::-1], np.sort(occ[1])[::-1]])
    mycc = cc.UCCSD(mu, frozen=nc).density_fit(auxbasis=aux)
    mycc.conv_tol = 1e-11
    mycc.kernel()
    return mycc.e_corr, mycc.ccsd_t()


def test_open_canonical_limit():
    from electrolites.dlpno.uhf import UDLPNO
    mol = gto.M(atom=CH3, basis='cc-pvdz', spin=1, verbose=0)
    mf = scf.ROHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').run(conv_tol=1e-11)
    d = UDLPNO(mf, pno='normal', triples='t1', auxbasis='cc-pvdz-ri', **FULL)
    d.kernel()
    e_cc, e_t = _semicanonical_uccsd_t(mf, d.ncore, 'cc-pvdz-ri')
    assert abs(d.e['ccsd'] - e_cc) < 1e-8
    assert abs(d.e['t'] - e_t) < 1e-8


def test_open_gpu_equals_host_reference():
    from electrolites.dlpno.uhf import UDLPNO
    from electrolites.dlpno.ref_uhf2 import RefUDLPNO2
    mol = gto.M(atom='C 0 0 0; C 1.49 0 0; H -.4 1 0; H -.4 -.5 .87; H -.4 -.5 -.87; '
                     'H 2.05 .93 0; H 2.05 -.93 0', basis='cc-pvdz', spin=1, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-10)
    out = []
    for C in (UDLPNO, RefUDLPNO2):
        d = C(mf, pno='loose', triples='t1', auxbasis='cc-pvdz-ri', verbose=0,
              E_CONVERGENCE=1e-9, R_CONVERGENCE=1e-8)
        d.kernel()
        out.append(d.e)
    g, r = out
    assert abs(g['lmp2'] - r['lmp2']) < 1e-9
    assert abs(g['ccsd'] - r['ccsd']) < 1e-8
    assert abs(g['t'] - r['t']) < 1e-8


def test_closed_molecule_through_open_code():
    from electrolites.dlpno.uhf import UDLPNO
    from electrolites.dlpno.rhf import DLPNO
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    tight = dict(E_CONVERGENCE=1e-10, R_CONVERGENCE=1e-9)
    u = UDLPNO(scf.ROHF(mol).run(conv_tol=1e-11), pno='normal', triples='t1',
               auxbasis='cc-pvdz-ri', verbose=0, **tight)
    u.kernel()
    r = DLPNO(scf.RHF(mol).run(conv_tol=1e-11), pno='normal', triples='t1',
              auxbasis='cc-pvdz-ri', c_lmo=u.orb.C_lmo, verbose=0, **tight)
    r.kernel()
    assert abs(u.e['lmp2'] - r.e['lmp2']) < 1e-8
    assert abs(u.e['ccsd'] - r.e['ccsd']) < 1e-7
    assert abs(u.e['t'] - r.e['t']) < 1e-8


def test_feature_entry_point():
    import electrolites
    electrolites.enable('dlpno')
    from gpu4pyscf import scf as gscf
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = gscf.RHF(mol).run(conv_tol=1e-10)
    solver = mf.DLPNO_CCSD_T(pno='normal', triples='t0')
    e = solver.kernel()
    assert solver.e['ccsd'] < 0 and solver.e['t'] < 0
    assert abs(e - (mf.e_tot + solver.e_corr)) < 1e-8
