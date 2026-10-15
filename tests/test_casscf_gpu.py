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
import cupy as cp

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

print('DF-CASSCF nuclear gradient (electrolites.casscf_grad)')
for atom, spin, cas in (('N 0 0 0; N 0 0.1 1.15', 0, (6, 6)),
                        ('O 0 0 0; O 0 0.1 1.25', 2, (6, 8))):
    m = gto.M(atom=atom, basis='6-31g*', spin=spin, verbose=0)
    mf = (scf.RHF if spin == 0 else scf.ROHF)(m).run(conv_tol=1e-11)
    mc = CASSCF(mf, *cas, density_fit=True)
    mc.conv_tol = 1e-11
    mc.kernel()
    g = mc.nuc_grad_method().kernel()
    cm = mcscf.CASSCF(mf.to_cpu().density_fit(auxbasis=mc.with_df.auxbasis),
                      *cas)
    cm.mo_coeff, cm.ci = cp.asnumpy(mc.mo_coeff), mc.ci
    gc = cm.nuc_grad_method().kernel(mo_coeff=cm.mo_coeff, ci=cm.ci)
    report(f'{atom[0]}2 CAS{cas} gradient == PySCF DF-CASSCF, same orbitals',
           abs(g - gc).max() < 1e-9, f'{abs(g - gc).max():.1e}')


def e_at(z):
    m = gto.M(atom=f'N 0 0 0; N 0 0.1 {z}', basis='6-31g*', verbose=0)
    mf = scf.RHF(m).run(conv_tol=1e-12)
    mc = CASSCF(mf, 6, 6, density_fit=True)
    mc.conv_tol = 1e-12
    mc.conv_tol_grad = 1e-7
    return mc.kernel()[0]


m = gto.M(atom='N 0 0 0; N 0 0.1 1.15', basis='6-31g*', verbose=0)
mc = CASSCF(scf.RHF(m).run(conv_tol=1e-12), 6, 6, density_fit=True)
mc.conv_tol = 1e-12
mc.conv_tol_grad = 1e-7
mc.kernel()
gz = mc.nuc_grad_method().kernel()[1, 2]
m = gto.M(atom='N 0 0 0; N 0 0.1 1.15', basis='6-31g*', verbose=0)
mf = scf.RHF(m).run(conv_tol=1e-11)
mc = CASSCF(mf, 6, 6, density_fit=True).state_average_([1 / 3] * 3)
mc.conv_tol = 1e-11
mc.kernel()
cm = mcscf.CASSCF(mf.to_cpu().density_fit(auxbasis=mc.with_df.auxbasis),
                  6, 6).state_average_([1 / 3] * 3)
cm.mo_coeff, cm.ci = cp.asnumpy(mc.mo_coeff), mc.ci
for st in (0, 1):
    g = mc.nuc_grad_method().kernel(state=st)
    gc = cm.nuc_grad_method().kernel(state=st, mo=cm.mo_coeff, ci=cm.ci)
    report(f'N2 SA-3 CAS(6,6) state {st} gradient == PySCF, same orbitals',
           abs(g - gc).max() < 1e-9, f'{abs(g - gc).max():.1e}')

print('DF SA-CASSCF non-adiabatic couplings (PySCF has them only without DF;')
print('the reference is its NAC class run on its DF gradient pieces)')
from pyscf import lib                                          # noqa: E402
from pyscf.nac import sacasscf as nac_sa                       # noqa: E402
from pyscf.grad import sacasscf as sa_grad                     # noqa: E402
from pyscf.df.grad import sacasscf as dfsa_grad                # noqa: E402
from pyscf.df.grad import casscf as dfcas_grad                 # noqa: E402
from pyscf.df.grad import rhf as dfrhf_grad                    # noqa: E402
m = gto.M(atom='Li 0 0 0; H 0 0.05 2.8', basis='6-31g*', verbose=0)
mfl = scf.RHF(m).run(conv_tol=1e-11)
mc = CASSCF(mfl, 4, 2, density_fit=True)
mc.fix_spin_(ss=0)                      # two singlets (root 1 is a triplet)
mc = mc.state_average_([.5, .5])
mc.conv_tol = 1e-11
mc.kernel()
mfc = mfl.to_cpu().density_fit(auxbasis=mc.with_df.auxbasis)
cm = mcscf.CASSCF(mfc, 4, 2)
cm.fix_spin_(ss=0)
cm = cm.state_average_([.5, .5])
cm.conv_tol = 1e-11
cm.kernel(cp.asnumpy(mc.mo_coeff), mc.ci)
mo = cp.asnumpy(mc.mo_coeff)
for etfs in (False, True):
    nac = mc.nac_method()
    nac.use_etfs = etfs
    g = nac.kernel(state=(0, 1))
    with lib.temporary_env(nac_sa, casscf_grad=dfcas_grad), \
            lib.temporary_env(sa_grad,
                              Lci_dot_dgci_dx=dfsa_grad.Lci_dot_dgci_dx,
                              Lorb_dot_dgorb_dx=dfsa_grad.Lorb_dot_dgorb_dx):
        gc = nac_sa.NonAdiabaticCouplings(cm, use_etfs=etfs).kernel(
            state=(0, 1), mo=mo, ci=mc.ci, mf_grad=dfrhf_grad.Gradients(mfc))
    err = abs(g - gc).max() / abs(gc).max()
    report(f'LiH SA-2 NAC <1|d0/dR>, use_etfs={etfs}', err < 1e-6,
           f'{err:.1e} relative (|NAC| {abs(gc).max():.2f})')

f = [(e_at(1.15 + d) - e_at(1.15 - d)) / (2 * d / 0.52917721092)
     for d in (4e-3, 2e-3)]                       # Angstrom steps, Bohr units
fd = (4 * f[1] - f[0]) / 3
# the analytic gradient assumes a stationary CASSCF; at an orbital gradient
# of 1e-7 that is good to a few 1e-7 (the same-orbital PySCF check is tight)
report('N2 CAS(6,6) dE/dz against a finite difference (Richardson)',
       abs(gz - fd) < 5e-7, f'{abs(gz - fd):.1e}')
for label, obj in (('CASCI gradient raises', CASCI(mf, 6, 6, density_fit=True)),
                   ('exact-integral CASSCF gradient raises', CASSCF(mf, 6, 6))):
    try:
        obj.kernel()
        obj.nuc_grad_method().kernel()
        report(label, False)
    except NotImplementedError:
        report(label, True)

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
