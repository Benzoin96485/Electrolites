"""Left singlet EOM-EE-CCSD and transition properties
(``eom_gpu.eomee_singlet_properties``) against a brute-force reference in
the full determinant space: H̄ = e^-T H e^T and mū = e^-T mu e^T built as
dense matrices from the converged T (no spin adaptation, no Λ code), its
left and right eigenvectors in the reference + singles + doubles
determinant space, <0|(1+Λ) mū R_k|0> and <0|L_k mū|0>.

    python tests/test_eomee_props_gpu.py
"""
import sys

import numpy as np
import scipy.linalg
from pyscf import gto, ao2mo, fci
from pyscf.fci import cistring
from gpu4pyscf import scf
from gpu4pyscf.cc import ccsd_incore

import electrolites
from electrolites import eom_gpu

electrolites.patch_all()
fails = []


def report(label, ok, detail=''):
    if not ok:
        fails.append(label)
    print(f'  {"ok  " if ok else "FAIL"} {label:52s} {detail}', flush=True)


def string_ops(norb, nel):
    """E_pq for one spin on the string space: dense (norb, norb, ns, ns)."""
    link = cistring.gen_linkstr_index(range(norb), nel)
    ns = link.shape[0]
    E = np.zeros((norb, norb, ns, ns))
    for s0 in range(ns):
        for a, i, s1, sgn in link[s0]:
            E[a, i, s1, s0] += sgn
    return E


def reference(mol, mf, mycc, nroots):
    """Exact-in-the-SD-space EOM-CCSD from the converged amplitudes."""
    C = np.asarray(mf.mo_coeff.get())
    norb = C.shape[1]
    na = mol.nelectron // 2
    Es = string_ops(norb, na)
    ns = Es.shape[2]
    I = np.eye(ns)

    def spin_sum(A):
        """kron(A, 1) + kron(1, A): a spin-free one-body operator."""
        return np.kron(A, I) + np.kron(I, A)
    # FCI Hamiltonian as a dense matrix
    h1 = C.T @ np.asarray(mf.get_hcore().get()) @ C
    eri = ao2mo.restore(1, ao2mo.kernel(mol, C), norb)
    h2 = fci.direct_spin1.absorb_h1e(h1, eri, norb, (na, na), .5)
    dim = ns * ns
    H = np.empty((dim, dim))
    for k in range(dim):
        v = np.zeros(dim)
        v[k] = 1
        H[:, k] = fci.direct_spin1.contract_2e(h2, v, norb, (na, na)).ravel()
    # T from the active amplitudes
    act = np.where(mycc.get_frozen_mask())[0]
    t1, t2 = np.asarray(mycc.t1.get() if hasattr(mycc.t1, 'get') else
                        mycc.t1), np.asarray(mycc.t2.get() if hasattr(
                            mycc.t2, 'get') else mycc.t2)
    no, nv = t1.shape
    occ, vir = act[:no], act[no:]
    Eai = np.array([[spin_sum(Es[vir[a], occ[i]]) for a in range(nv)]
                    for i in range(no)])                 # (o, v, dim, dim)
    T1 = np.einsum('ia,iaxy->xy', t1, Eai)
    X = np.einsum('ijab,jbxy->iaxy', t2, Eai)
    T2 = .5 * np.einsum('iaxy,iayz->xz', Eai, X)
    T = T1 + T2
    eT = scipy.linalg.expm(T)
    emT = scipy.linalg.expm(-T)
    Hb = emT @ H @ eT
    # reference determinant and the SD space (core always occupied)
    strs = cistring.make_strings(range(norb), na)
    ref = strs[0]
    core = sum(1 << int(p) for p in range(occ[0]))

    def level(s):
        return bin(int(s) & ~int(ref)).count('1')
    lv = np.array([level(s) for s in strs])
    cok = np.array([(int(s) & core) == core for s in strs])
    P = [ia * ns + ib for ia in range(ns) for ib in range(ns)
         if lv[ia] + lv[ib] <= 2 and cok[ia] and cok[ib]]
    P = np.array(P)
    i0 = int(np.where(P == 0)[0][0])
    HP = Hb[np.ix_(P, P)]
    resid = abs(np.delete(HP[:, i0], i0)).max()
    w, vl, vr = scipy.linalg.eig(HP, left=True, right=True)
    w = w.real
    vl, vr = vl.real, vr.real
    ecc = HP[i0, i0]
    g = int(np.argmin(abs(w - ecc)))
    L0 = vl[:, g] / vl[i0, g]                 # <0|(1+Λ)
    # dipole
    charges = mol.atom_charges()
    orig = charges @ mol.atom_coords() / charges.sum()
    with mol.with_common_orig(orig):
        d = np.einsum('xuv,up,vq->xpq', mol.intor('int1e_r', comp=3), C, C)
    mub = [emT @ spin_sum(np.einsum('pq,pqxy->xy', d[x], Es)) @ eT
           for x in range(3)]
    out = []
    for k in range(len(w)):
        if k == g:
            continue
        r, lk = vr[:, k], vl[:, k]
        lk = lk / (lk @ r)
        m0k = np.array([L0 @ (mub[x][np.ix_(P, P)] @ r) for x in range(3)])
        mk0 = np.array([lk @ mub[x][np.ix_(P, P)][:, i0] for x in range(3)])
        out.append((w[k] - ecc, m0k, mk0, r[i0]))
    return resid, sorted(out, key=lambda x: x[0])


cases = (
    ('H2O distorted STO-3G', 'O 0 0 0.11; H 0 0.78 -0.45; H 0.05 -0.74 -0.5',
     'sto-3g', None),
    ('H2O distorted STO-3G fc', 'O 0 0 0.11; H 0 0.78 -0.45; H 0.05 -0.74 -0.5',
     'sto-3g', 1),
    ('NH3 distorted STO-3G', 'N 0 0 0; H 0 0.94 0.35; H 0.8 -0.5 0.33; '
     'H -0.85 -0.45 0.4', 'sto-3g', None),
    ('LiH 6-31G', 'Li 0 0 0; H 0 0 1.6', '6-31g', None),
)
for name, atom, basis, frozen in cases:
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mycc = ccsd_incore.CCSD(mf, frozen=frozen)
    mycc.conv_tol = 1e-12
    mycc.conv_tol_normt = 1e-9
    mycc.kernel()
    nroots = 4
    eom, props = eom_gpu.eomee_singlet_properties(mycc, nroots, conv_tol=1e-10)
    resid, ref = reference(mol, mf, mycc, nroots)
    report(f'{name}: CCSD equations hold in the SD space', resid < 1e-8,
           f'max |<mu|H̄|0>| {resid:.1e}')
    e = props['e']
    report(f'{name}: left roots == right roots',
           abs(props['e_left'] - e).max() < 1e-7,
           f'{abs(props["e_left"] - e).max():.1e} Eh')
    # match each GPU root to the reference root of the same energy (the
    # reference has triplets too); compare the products M0k.Mk0 summed over
    # degenerate partners, which are invariant
    wr = np.array([x[0] for x in ref])
    for k in range(nroots):
        j = int(np.argmin(abs(wr - e[k])))
        de = abs(wr[j] - e[k])
        grp = np.where(abs(wr - wr[j]) < 1e-6)[0]
        f_ref = sum(2 / 3 * ref[i][0] * (ref[i][1] @ ref[i][2]) for i in grp)
        gk = np.where(abs(e - e[k]) < 1e-6)[0]
        f_new = props['f'][gk].sum()
        ok = de < 1e-7 and abs(f_new - f_ref) < 1e-7 + 1e-6 * abs(f_ref)
        report(f'{name}: root {k} energy and f == brute force', ok,
               f'dE {de:.1e}  f {f_new:.8f} ref {f_ref:.8f}')

print('FAILED: ' + ', '.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
