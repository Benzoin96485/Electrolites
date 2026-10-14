"""GPU FCI solver and DF-CASSCF (`electrolites/fci_gpu.py`,
`electrolites/casscf_gpu.py`) against PySCF 2.13.

* the sigma build, 1-/2-RDMs (reordered and not) and transition RDMs on
  random vectors and integrals, closed and open shell, against PySCF's
  ``direct_spin1`` (relative 1e-12);
* DF-CASSCF on N2 (closed shell) and O2 (triplet), DF-CASCI on the same
  orbitals, and state-averaged CASSCF from the same starting orbitals,
  against PySCF's DF-CASSCF with the same auxiliary basis (1e-9 Eh).

    python tests/test_casscf_gpu.py
"""
import sys

import numpy as np

from pyscf import gto, fci, mcscf                              # noqa: E402
from gpu4pyscf import scf                                      # noqa: E402
from electrolites import fci_gpu                               # noqa: E402
from electrolites.casscf_gpu import CASSCF, CASCI              # noqa: E402

fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


def rel(a, b):
    return abs(np.asarray(a) - np.asarray(b)).max() / max(abs(np.asarray(b)).max(), 1e-300)


print('FCI solver')
rng = np.random.default_rng(0)
for norb, nelec in ((6, (3, 3)), (7, (4, 2)), (8, (4, 4)), (9, (5, 4))):
    na = fci.cistring.num_strings(norb, nelec[0])
    nb = fci.cistring.num_strings(norb, nelec[1])
    c = rng.normal(size=(na, nb))
    k = rng.normal(size=(na, nb))
    h1 = rng.normal(size=(norb, norb))
    h1 = h1 + h1.T
    eri = rng.normal(size=(norb,) * 4)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(2, 3, 0, 1)
    s, g = fci.direct_spin1.FCISolver(), fci_gpu.FCISolver()
    h2 = s.absorb_h1e(h1, eri, norb, nelec, .5)
    errs = [rel(g.contract_2e(h2, c.ravel(), norb, nelec),
                s.contract_2e(h2, c.ravel(), norb, nelec).ravel())]
    for fn in ('make_rdm12',):
        a1, a2 = getattr(g, fn)(c, norb, nelec)
        b1, b2 = getattr(s, fn)(c, norb, nelec)
        errs += [rel(a1, b1), rel(a2, b2)]
    a1, a2 = g.make_rdm12(c, norb, nelec, reorder=False)
    b1, b2 = s.make_rdm12(c, norb, nelec, reorder=False)
    errs.append(rel(a2, b2))
    a1, a2 = g.trans_rdm12(c, k, norb, nelec)
    b1, b2 = s.trans_rdm12(c, k, norb, nelec)
    errs += [rel(a1, b1), rel(a2, b2)]
    report(f'CAS({norb}, {nelec}) sigma, rdm12, trans_rdm12', max(errs) < 1e-12,
           f'{max(errs):.1e}')

for norb, nelec, nroots in ((10, (5, 5), 1), (10, (5, 4), 3), (11, (6, 5), 2)):
    h1 = rng.normal(size=(norb, norb)) * .1
    h1 = h1 + h1.T + np.diag(np.arange(norb, dtype=float))
    eri = rng.normal(size=(norb,) * 4) * .02
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(2, 3, 0, 1)
    s, g = fci.direct_spin1.FCISolver(), fci_gpu.FCISolver()
    s.nroots = g.nroots = nroots
    s.conv_tol = g.conv_tol = 1e-11
    es = s.kernel(h1, eri, norb, nelec)[0]
    eg = g.kernel(h1, eri, norb, nelec)[0]
    err = abs(np.asarray(eg) - np.asarray(es)).max()
    report(f'CAS({norb}, {nelec}) device Davidson, {nroots} root(s)',
           err < 1e-9 and np.all(g.converged), f'{err:.1e}')

print('DF-CASSCF / CASCI')
n2 = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='6-31g', verbose=0)
mf = scf.RHF(n2).run(conv_tol=1e-11)
mfc = mf.to_cpu()
mc = CASSCF(mf, 8, 8, density_fit=True)
mc.conv_tol = 1e-10
e = mc.kernel()[0]
aux = mc.with_df.auxbasis
cm = mcscf.CASSCF(mfc.density_fit(auxbasis=aux), 8, 8)
cm.conv_tol = 1e-10
ec = cm.kernel(mfc.mo_coeff)[0]
report('N2 CAS(8,8) CASSCF', abs(e - ec) < 1e-9, f'{abs(e - ec):.1e}')
e1 = CASCI(mf, 6, 6, density_fit=True).kernel()[0]
e2 = mcscf.CASCI(mfc.density_fit(auxbasis=aux), 6, 6).kernel()[0]
report('N2 CAS(6,6) CASCI, same orbitals', abs(e1 - e2) < 1e-9, f'{abs(e1 - e2):.1e}')
mc = CASSCF(mf, 8, 8, density_fit=True).state_average_([.25] * 4)
mc.conv_tol = 1e-10
e = mc.kernel()[0]
cm = mcscf.CASSCF(mfc.density_fit(auxbasis=aux), 8, 8).state_average_([.25] * 4)
cm.conv_tol = 1e-10
ec = cm.kernel(mfc.mo_coeff)[0]
# the averaged energy is what is converged (1e-10); the individual states of
# the degenerate Pi pair only to ~1e-8
dst = abs(np.sort(mc.e_states) - np.sort(cm.e_states)).max()
report('N2 SA-4 CASSCF, same start', abs(e - ec) < 1e-9 and dst < 1e-7,
       f'{abs(e - ec):.1e}; states {dst:.1e}; {e!r} {ec!r}')
o2 = gto.M(atom='O 0 0 0; O 0 0 1.21', basis='6-31g', spin=2, verbose=0)
mf = scf.ROHF(o2).run(conv_tol=1e-11)
mfc = mf.to_cpu()
mc = CASSCF(mf, 6, 8, density_fit=True)
mc.conv_tol = 1e-10
e = mc.kernel()[0]
cm = mcscf.CASSCF(mfc.density_fit(auxbasis=mc.with_df.auxbasis), 6, 8)
cm.conv_tol = 1e-10
ec = cm.kernel(mfc.mo_coeff)[0]
report('O2 triplet CAS(6,8) CASSCF', abs(e - ec) < 1e-9, f'{abs(e - ec):.1e}')

print('exact-integral CASSCF / CASCI')
mf = scf.RHF(n2).run(conv_tol=1e-11)
mfc = mf.to_cpu()
mc = CASSCF(mf, 8, 8)
mc.conv_tol = 1e-10
e = mc.kernel()[0]
cm = mcscf.CASSCF(mfc, 8, 8)
cm.conv_tol = 1e-10
ec = cm.kernel(mfc.mo_coeff)[0]
report('N2 CAS(8,8) CASSCF, exact integrals', abs(e - ec) < 1e-9, f'{abs(e - ec):.1e}')
e1 = CASCI(mf, 6, 6).kernel()[0]
e2 = mcscf.CASCI(mfc, 6, 6).kernel()[0]
report('N2 CAS(6,6) CASCI, exact integrals', abs(e1 - e2) < 1e-9, f'{abs(e1 - e2):.1e}')
mf = scf.ROHF(o2).run(conv_tol=1e-11)
mfc = mf.to_cpu()
mc = CASSCF(mf, 6, 8)
mc.conv_tol = 1e-10
e = mc.kernel()[0]
cm = mcscf.CASSCF(mfc, 6, 8)
cm.conv_tol = 1e-10
ec = cm.kernel(mfc.mo_coeff)[0]
report('O2 triplet CAS(6,8) CASSCF, exact integrals', abs(e - ec) < 1e-9,
       f'{abs(e - ec):.1e}')
mf = scf.RHF(n2).density_fit().run(conv_tol=1e-11)
from electrolites import casscf_gpu                            # noqa: E402
report('DF mean field -> DF route, plain -> exact route',
       isinstance(CASSCF(mf, 8, 8), casscf_gpu._GPUCAS) and
       isinstance(CASSCF(mf.undo_df(), 8, 8), casscf_gpu._GPUDirect))

print('the feature binding')
import electrolites                                             # noqa: E402
electrolites.enable('casscf')
mf = scf.RHF(n2).run(conv_tol=1e-11)
mc = mf.CASSCF(8, 8)
mc.conv_tol = 1e-10
report('gpu_mf.CASSCF(8, 8) == CASSCF(gpu_mf, 8, 8)',
       abs(mc.kernel()[0] - CASSCF(mf, 8, 8).run(conv_tol=1e-10).e_tot) < 1e-9)
mc = mf.density_fit().CASSCF(8, 8)
mc.conv_tol = 1e-10
report('gpu_mf.density_fit().CASSCF(8, 8) == CASSCF(.., density_fit=True)',
       abs(mc.kernel()[0] - CASSCF(mf, 8, 8, density_fit=True).run(
           conv_tol=1e-10).e_tot) < 1e-9)

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
