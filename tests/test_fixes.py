"""The small upstream repairs (`electrolites/fixes.py`, FEATURE_TARGETS §3.3).

D1 the device vSAP matrix against PySCF's on the same grid, and the vSAP
guess converging RKS/UKS/ROKS to the minao energies; D2 ROHF's Newton solver
against its plain SCF; D3 ``CCSD.to_cpu()`` carrying the amplitudes;
D4 a clear error for an open-shell CCSD; D7 frozen-core DF-(U)MP2 with
``with_t2`` against the unfrozen-free reference (PySCF DF-MP2 on the CPU).

    python tests/test_fixes.py
"""
import sys

import numpy as np
import cupy

import electrolites
electrolites.patch('fixes')

from pyscf import gto, dft as cdft, scf as cscf, mp as cmp  # noqa: E402
from gpu4pyscf import scf, dft                              # noqa: E402

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


water = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.467; H 0 -0.757 -0.467',
              basis='def2-svp', verbose=0)
cation = gto.M(atom='O 0 0 0.1; H 0 0.76 -0.45; H 0 -0.76 -0.45',
               basis='def2-svp', charge=1, spin=1, verbose=0)

print('D1 vSAP')
ks = dft.RKS(water, xc='pbe')
ks.grids.build()
v = ks.get_vsap().get()
cks = cdft.RKS(water, xc='pbe')
err = abs(v - cks.get_vsap()).max()
report('get_vsap == PySCF nr_sap (same grid)', err < 1e-10, f'{err:.1e}')
for name, mk in (('RKS', lambda: dft.RKS(water, xc='pbe')),
                 ('UKS', lambda: dft.UKS(cation, xc='pbe')),
                 ('ROKS', lambda: dft.ROKS(cation, xc='pbe'))):
    e_v = mk().set(init_guess='vsap', conv_tol=1e-10).run().e_tot
    e_m = mk().set(conv_tol=1e-10).run().e_tot
    report(f'{name} vsap guess converges to the minao energy',
           abs(e_v - e_m) < 1e-8, f'{abs(e_v - e_m):.1e}')

print('D2 ROHF Newton')
e_n = scf.ROHF(cation).newton().run(conv_tol=1e-10).e_tot
e_p = scf.ROHF(cation).run(conv_tol=1e-10).e_tot
report('ROHF.newton() == ROHF', abs(e_n - e_p) < 1e-8, f'{abs(e_n - e_p):.1e}')

print('D3 / D4 CCSD')
from gpu4pyscf.cc import ccsd_incore                        # noqa: E402
# no d shells: the legacy (dd|dd) fill needs GBs of launch-time local
# memory (KT 13.27), which a shared login-node card may not have
water_s = gto.M(atom=water.atom, basis='6-31g', verbose=0)
cc = ccsd_incore.CCSD(scf.RHF(water_s).run(conv_tol=1e-11)).run()
cpu = cc.to_cpu()
ok = (type(cpu).__module__ == 'pyscf.cc.ccsd'
      and isinstance(cpu.t2, np.ndarray)
      and abs(cpu.e_corr - cc.e_corr) == 0)
report('CCSD.to_cpu() is a PySCF CCSD with the amplitudes', ok)
et = cpu.ccsd_t()
ref = cscf.RHF(water_s).run(conv_tol=1e-11).CCSD().run()
report('... and its (T) matches PySCF', abs(et - ref.ccsd_t()) < 1e-8,
       f'{abs(et - ref.ccsd_t()):.1e}')
try:
    ccsd_incore.CCSD(scf.UHF(cation).run())
    report('CCSD(UHF) raises NotImplementedError', False, 'did not raise')
except NotImplementedError:
    report('CCSD(UHF) raises NotImplementedError', True)

print('D7 frozen DF-MP2 with t2')
mf = scf.RHF(water).run(conv_tol=1e-11)
from gpu4pyscf.mp import dfmp2, dfump2                      # noqa: E402
for frozen in (1, [0, 23]):
    pt = dfmp2.DFMP2(mf, frozen=frozen)
    pt.with_t2 = True
    pt.kernel()
    c = cmp.dfmp2.DFMP2(cscf.RHF(water).run(conv_tol=1e-11), frozen=frozen)
    c.kernel()
    t2 = pt.t2.get() if isinstance(pt.t2, cupy.ndarray) else pt.t2
    report(f'DF-RMP2 frozen={frozen} with_t2',
           abs(pt.e_corr - c.e_corr) < 1e-8
           and t2.shape[:2] == (c.nocc, c.nocc)
           and t2.shape[2] == c.nmo - c.nocc,
           f'{abs(pt.e_corr - c.e_corr):.1e} t2 {t2.shape}')
um = scf.UHF(cation).run(conv_tol=1e-11)
pt = dfump2.DFUMP2(um, frozen=1)
pt.with_t2 = True
pt.kernel()
c = cmp.dfump2.DFUMP2(cscf.UHF(cation).run(conv_tol=1e-11), frozen=1)
c.kernel()
report('DF-UMP2 frozen=1 with_t2', abs(pt.e_corr - c.e_corr) < 1e-8,
       f'{abs(pt.e_corr - c.e_corr):.1e}')

print('V3 Gaussian MM charges (QM/MM) against PySCF, B3LYP water + 3 charges')
from pyscf import qmmm as cqmmm                                 # noqa: E402
from gpu4pyscf import dft as gdft                               # noqa: E402
from gpu4pyscf.qmmm import itrf as gitrf                        # noqa: E402
wat = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
            basis='def2-svp', verbose=0)
mm_xyz = np.array([[1.5, 0.3, 2.0], [-1.8, 0.4, -1.2], [0.2, 2.5, 0.7]])
mm_q = np.array([-0.8, 0.4, 0.4])
for rad in (None, np.array([0.8, 0.5, 0.5])):
    mf = gitrf.mm_charge(gdft.RKS(wat, xc='b3lyp'), mm_xyz, mm_q, radii=rad,
                         unit='Angstrom').run(conv_tol=1e-11)
    gg = mf.nuc_grad_method()
    gq = cupy.asnumpy(gg.kernel())
    gm = gg.grad_nuc_mm() + cupy.asnumpy(gg.grad_hcore_mm(mf.make_rdm1()))
    mc = cqmmm.mm_charge(wat.RKS(xc='b3lyp'), mm_xyz, mm_q, radii=rad,
                         unit='Angstrom')
    mc.grids.level = mf.grids.level
    mc.run(conv_tol=1e-11)
    gc = mc.nuc_grad_method()
    gcq = gc.kernel()
    gcm = gc.grad_nuc_mm() + gc.grad_hcore_mm(mc.make_rdm1())
    de, dq, dm_ = (abs(mf.e_tot - mc.e_tot), abs(gq - gcq).max(),
                   abs(gm - gcm).max())
    # 8e-8 on the QM gradient is the KS grid response, as for point charges
    report(f'{"Gaussian" if rad is not None else "point"} charges: E, '
           'QM and MM gradients', de < 1e-9 and dq < 5e-7 and dm_ < 1e-7,
           f'{de:.1e} {dq:.1e} {dm_:.1e}')

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
