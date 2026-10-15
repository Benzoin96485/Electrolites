"""MC-PDFT on the GPU CASSCF (``electrolites.mcpdft_gpu``, FEATURE_TARGETS M8)
against PySCF's ``mcpdft`` on a CPU DF-CASSCF with the same orbitals, CI
vectors, auxiliary basis and grid.

    python tests/test_mcpdft_gpu.py
"""
import sys

import cupy as cp

from pyscf import gto, mcpdft
from pyscf.mcpdft.mcpdft import energy_tot
from gpu4pyscf import scf

from electrolites.casscf_gpu import CASSCF
from electrolites.mcpdft_gpu import MCPDFT
import electrolites.mcpdft_gpu as mg

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


def check(label, mol, rohf, cas, ot, nroots=1):
    mf = (scf.ROHF if rohf else scf.RHF)(mol).run(conv_tol=1e-11)
    pd = MCPDFT(CASSCF(mf, *cas, density_fit=True), ot)
    if nroots > 1:                      # as PySCF: SA on the MC-PDFT object
        pd = pd.state_average_([1. / nroots] * nroots)
    pd.conv_tol = 1e-10
    pd.kernel()
    mc = pd
    mfc = mf.to_cpu().density_fit(auxbasis=mc.with_df.auxbasis)
    cc = mcpdft.CASSCF(mfc, ot, *cas)
    if nroots > 1:
        cc = cc.state_average_([1. / nroots] * nroots)
    mo = cp.asnumpy(mc.mo_coeff)
    err = 0.
    for st in range(nroots):
        eg = energy_tot(pd, mo_coeff=mo, ci=mc.ci, state=st)[0]
        ec = energy_tot(cc, mo_coeff=mo, ci=mc.ci, state=st)[0]
        err = max(err, abs(eg - ec))
    report(label, err < 1e-9, f'{err:.1e}')


print('MC-PDFT energies against PySCF (same orbitals, CI and grid)')
n2 = gto.M(atom='N 0 0 0; N 0 0.1 1.15', basis='6-31g*', verbose=0)
for ot in ('tLDA', 'tPBE', 'tBLYP', 'tPBE0'):
    check(f'N2 CAS(6,6) {ot}', n2, False, (6, 6), ot)
o2 = gto.M(atom='O 0 0 0; O 0 0.1 1.22', basis='6-31g*', spin=2, verbose=0)
check('O2 triplet CAS(6,8) tPBE (ROHF orbitals)', o2, True, (6, 8), 'tPBE')
check('N2 SA-3 CAS(6,6) tPBE, every state', n2, False, (6, 6), 'tPBE', 3)

print('the on-top energy is the GPU one')
from pyscf.mcpdft import otfnal                                # noqa: E402
calls = []
orig = otfnal.energy_ot
otfnal.energy_ot = lambda *a, **k: calls.append(1) or orig(*a, **k)
mf = scf.RHF(n2).run()
pd = mg.CASSCF(mf, 'tPBE', 6, 6, density_fit=True).run()
report('mcpdft_gpu.CASSCF(..., "tPBE").run(): no host E_ot', not calls,
       f'E = {pd.e_tot:.10f}')
otfnal.energy_ot = orig

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
