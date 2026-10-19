"""Orbitals, differential overlaps, dipole prescreening and sparse maps.

Everything here is O(N^2) bookkeeping on matrices of AO or LMO size, shared
by the closed-shell reference (``ref_rhf``), the GPU implementation and the
open-shell code (which calls it once per spin).  The quantities follow
Pinski et al., JCP 143, 034108 (2015) and their Psi4 implementation
(``dlpno.cc``): DOI integrals (EQ 4), dipole pair energies (EQ 17), Mulliken
fitting domains, PAO domains completed by atom.
"""
import numpy as np
import scipy.linalg

from pyscf import lib
from pyscf.dft import gen_grid, numint, radi
from pyscf.data import elements


def asnumpy(a):
    try:
        import cupy
        if isinstance(a, cupy.ndarray):
            return a.get()
    except ImportError:
        pass
    return np.asarray(a)


# --------------------------------------------------------------------------
# small dense helpers
# --------------------------------------------------------------------------

def pivoted_cholesky(S, tol):
    """Indices chosen by a pivoted Cholesky of S, stopping at diag < tol
    (Psi4 ``Matrix::pivoted_cholesky``)."""
    n = S.shape[0]
    L = np.zeros((n, n))
    d = np.array(np.diag(S), dtype=float)
    piv = []
    for m in range(n):
        p = int(np.argmax(d))
        if d[p] < tol:
            break
        piv.append(p)
        lp = S[:, p] - L[:, :m] @ L[p, :m]
        lp = lp / np.sqrt(d[p])
        L[:, m] = lp
        d = d - lp * lp
        d[piv] = -1.
    return piv


def orthocanonicalize(S, F, s_cut):
    """Orthonormal, canonical combination of a (redundant) set of functions.

    Psi4 ``DLPNO::orthocanonicalizer``: normalise, partial (pivoted)
    Cholesky with tolerance S_CUT, canonical orthogonalisation of the kept
    functions, then diagonalise F.  Returns X (n, m) and orbital energies."""
    nrm = 1. / np.sqrt(np.diag(S))
    Sn = S * nrm[:, None] * nrm[None, :]
    piv = pivoted_cholesky(Sn, s_cut)
    piv = sorted(piv)
    e, U = np.linalg.eigh(Sn[np.ix_(piv, piv)])
    Xs = U / np.sqrt(e)
    X = np.zeros((S.shape[0], len(piv)))
    X[piv] = Xs
    X *= nrm[:, None]
    eps, V = np.linalg.eigh(X.T @ F @ X)
    return X @ V, eps


def canonicalize(X, F):
    """Rotate orthonormal X (in a basis with Fock F) to canonical form."""
    eps, V = np.linalg.eigh(X.T @ F @ X)
    return V, eps


def contract_to_atoms(idx, bf_atom, atom_bfs):
    """All functions of every atom that owns one of ``idx`` (sorted)."""
    atoms = np.unique(bf_atom[np.asarray(idx, dtype=int)]) if len(idx) else []
    out = [atom_bfs[a] for a in atoms]
    return np.concatenate(out).astype(int) if out else np.zeros(0, int)


# --------------------------------------------------------------------------
# orbitals
# --------------------------------------------------------------------------

def count_frozen_core(mol):
    return int(elements.chemcore(mol))


def localize(mol, c_act, method='boys', conv_tol=1e-12, mf=None):
    """Localise the active occupied orbitals (host).  Starting guess is the
    canonical orbitals themselves, as in Psi4."""
    from pyscf import lo
    if c_act.shape[1] == 0:
        return c_act
    if method.lower() in ('boys', 'fb'):
        # PySCF's CIAH Boys with the dipole tensors on the device
        # (electrolites.lo_gpu): same objective and driver, no per-step AO work
        try:
            from electrolites.lo_gpu import Boys
        except ImportError:
            Boys = lo.Boys
        loc = Boys(mol, c_act)
    elif method.lower() in ('pm', 'pipek', 'pipek_mezey'):
        loc = lo.PM(mol, c_act, mf)
        loc.pop_method = 'mulliken'
    else:
        raise ValueError(method)
    loc.init_guess = None
    loc.conv_tol = conv_tol
    loc.verbose = 0
    return loc.kernel()


class Orbitals:
    """LMOs, PAOs and their Fock/overlap matrices for one spin."""

    def __init__(self, mol, S, F, C, nocc, ncore, localizer='boys',
                 c_lmo=None, mf=None, conv_tol=1e-12):
        self.mol = mol
        self.S = S
        self.F = F
        self.nbf = S.shape[0]
        self.ncore = ncore
        c_occ = C[:, :nocc]
        c_act = C[:, ncore:nocc]
        if c_lmo is None:
            c_lmo = localize(mol, c_act, localizer, conv_tol, mf)
        self.C_lmo = np.asarray(c_lmo)
        self.F_lmo = self.C_lmo.T @ F @ self.C_lmo
        self.naocc = self.C_lmo.shape[1]
        nbf = self.nbf
        c_pao = np.eye(nbf) - c_occ @ (c_occ.T @ S)
        s_pao = c_pao.T @ S @ c_pao
        c_pao = c_pao / np.sqrt(np.diag(s_pao))[None, :]
        self.C_pao = c_pao
        self.S_pao = c_pao.T @ S @ c_pao
        self.F_pao = c_pao.T @ F @ c_pao


# --------------------------------------------------------------------------
# DOI and dipole integrals
# --------------------------------------------------------------------------

def doi_grid(mol, grid=(25, 50)):
    """the DOI quadrature (Treutler radial x Lebedev, Becke partition).  On the
    GPU when GPU4PySCF is present: the same grid (weights equal to 1e-13; the
    CPU Becke partition costs O(atoms x points), 90 s for 450 atoms); points
    with zero weight (padding, < 1e-22) dropped."""
    try:
        from gpu4pyscf.dft import gen_grid as ggrid
        h = ggrid.Grids(mol)
        h.atom_grid = grid
        h.radi_method = radi.treutler
        h.prune = None
        h.verbose = 0
        h.build(with_non0tab=False)
        c, w = h.coords.get(), h.weights.get()
        nz = w != 0
        return np.ascontiguousarray(c[nz]), np.ascontiguousarray(w[nz])
    except ImportError:
        pass
    g = gen_grid.Grids(mol)
    g.atom_grid = grid
    g.radi_method = radi.treutler
    g.prune = None
    g.verbose = 0
    g.build(with_non0tab=False)
    return g.coords, g.weights


def compute_doi(mol, C_lmo, C_pao, grid=(25, 50), blksize=4096, need_uv=True):
    """sqrt(int |i|^2 |u|^2): DOI_ij, DOI_iu and (optionally) DOI_uv."""
    import os
    if os.environ.get('DLPNO_DOI_GRID'):            # externally supplied grid (validation)
        g = np.load(os.environ['DLPNO_DOI_GRID'])
        coords, weights = g[:, :3], g[:, 3]
    else:
        coords, weights = doi_grid(mol, grid)
    import cupy as cp
    try:
        from gpu4pyscf.dft import numint as gnum
        ev = lambda c: gnum.eval_ao(mol, cp.asarray(c))
    except ImportError:
        ev = lambda c: cp.asarray(numint.eval_ao(mol, c))
    nl, nu = C_lmo.shape[1], C_pao.shape[1]
    L, P = cp.asarray(C_lmo), cp.asarray(C_pao)
    ij = cp.zeros((nl, nl))
    iu = cp.zeros((nl, nu))
    uv = cp.zeros((nu, nu)) if need_uv else None
    wd = cp.asarray(weights)
    # screening (DLPNO_DOI_SCREEN, relative coefficient cut; 0: off): per grid
    # block only the AOs that are non-negligible on it and the LMOs / PAOs whose
    # AO support touches those AOs (local blocks: linear cost for large systems)
    cut = float(os.environ.get('DLPNO_DOI_SCREEN', '1e-8'))
    screen = cut > 0 and nl + nu > 2000
    if screen:
        La, Pa = cp.abs(L), cp.abs(P)
        lsup = La > cut * La.max(axis=0)[None, :]
        psup = Pa > cut * Pa.max(axis=0)[None, :]
        del La, Pa
    for p0 in range(0, len(weights), blksize):
        p1 = min(p0 + blksize, len(weights))
        ao = ev(coords[p0:p1])
        if ao.shape[0] != p1 - p0:
            ao = ao.T
        w = wd[p0:p1, None]
        if screen:
            sig = cp.abs(ao).max(axis=0) > 1e-10
            sa = cp.flatnonzero(sig)
            ls = cp.flatnonzero(lsup[sa].any(axis=0))
            us = cp.flatnonzero(psup[sa].any(axis=0))
            aos = ao[:, sa]
            li = (aos @ L[sa][:, ls]) ** 2
            lu = (aos @ P[sa][:, us]) ** 2
            ij[ls[:, None], ls[None, :]] += (li * w).T @ li
            iu[ls[:, None], us[None, :]] += (li * w).T @ lu
            if need_uv:
                uv[us[:, None], us[None, :]] += (lu * w).T @ lu
            continue
        li = (ao @ L) ** 2
        lu = (ao @ P) ** 2
        ij += (li * w).T @ li
        iu += (li * w).T @ lu
        if need_uv:
            uv += (lu * w).T @ lu
    ij, iu = ij.get(), iu.get()
    uv = uv.get() if need_uv else None
    return (np.sqrt(np.maximum(ij, 0)), np.sqrt(np.maximum(iu, 0)),
            np.sqrt(np.maximum(uv, 0)) if need_uv else None)


def dipole_pair_energies(orb, doi_iu, t_cut_do_pre, s_cut, bf_atom, atom_bfs):
    """Approximate SC-MP2 pair energies of distant pairs from dipole
    integrals (Pinski EQ 17).  Returns (estimate, bound) matrices.

    On the device: the dipole PAO domains of all LMOs are orthocanonicalised
    in one batch (pivoted Cholesky selection at S_CUT, as ``orthocanonicalize``)
    and the pair sums run as batched contractions over j for each i."""
    import cupy as cp
    from electrolites.dlpno.linalg import eigh_batched, orthonormal_domain
    from electrolites.dlpno.ragged import padded
    mol = orb.mol
    with mol.with_common_orig((0, 0, 0)):
        r = mol.intor_symmetric('int1e_r')
    L, P = orb.C_lmo, orb.C_pao
    n = orb.naocc
    rd, Ld, Pd = cp.asarray(r), cp.asarray(L), cp.asarray(P)
    rL = cp.matmul(rd, Ld[None])                                          # (3, nao, n)
    R = cp.einsum('xmi,mi->ix', rL, Ld).get()
    lp = cp.matmul(rL.transpose(0, 2, 1), Pd[None])                       # (3, n, npao)
    doms = [contract_to_atoms(np.where(np.abs(doi_iu[i]) > t_cut_do_pre)[0], bf_atom, atom_bfs)
            for i in range(n)]
    nn = np.array([len(d) for d in doms])
    Pm = int(padded(max(nn.max(), 1)))
    idx = np.zeros((n, Pm), np.int64)
    msk = np.zeros((n, Pm))
    for i, d in enumerate(doms):
        idx[i, :len(d)] = d
        msk[i, :len(d)] = 1.
    gi, gm = cp.asarray(idx), cp.asarray(msk)
    mm = gm[:, :, None] * gm[:, None, :]
    Sd = cp.asarray(orb.S_pao)[gi[:, :, None], gi[:, None, :]] * mm
    Fd = cp.asarray(orb.F_pao)[gi[:, :, None], gi[:, None, :]] * mm
    X0, keep = orthonormal_domain(Sd, nn, s_cut, 'pivchol')
    Fo = cp.matmul(X0.transpose(0, 2, 1), cp.matmul(Fd, X0)) + 1e6 * cp.eye(Pm)[None] * (~keep)[:, :, None]
    e, V = eigh_batched(Fo)
    X = cp.matmul(X0, V)
    valid = e < 1e5
    ar = cp.arange(n)
    D = cp.einsum('xbu,buv->bxv', lp[:, ar[:, None], gi] * gm[None], X)      # (n, 3, Pm)
    D = D * valid[:, None, :]
    f = cp.asarray(np.diag(orb.F_lmo))
    ev = cp.where(valid, e, 1e6)
    E = np.zeros((n, n))
    B = np.zeros((n, n))
    Rd = cp.asarray(R)
    for i in range(n - 1):
        js = cp.arange(i + 1, n)
        Rij = Rd[i][None] - Rd[js]                                         # (m, 3)
        d = cp.linalg.norm(Rij, axis=1)
        rh = Rij / d[:, None]
        dot = cp.einsum('xu,jxv->juv', D[i], D[js])                         # (m, P, P)
        ri = cp.einsum('jx,xu->ju', rh, D[i])
        rj = cp.einsum('jx,jxv->jv', rh, D[js])
        den = ev[i][None, :, None] + ev[js][:, None, :] - f[i] - f[js][:, None, None]
        num = (dot - 3 * ri[:, :, None] * rj[:, None, :]) ** 2
        Ei = -4 * d ** -6 * cp.sum(num / den, axis=(1, 2))
        Bi = -4 * d ** -6 * cp.sum((2 * dot) ** 2 / den, axis=(1, 2))
        E[i, i + 1:] = E[i + 1:, i] = Ei.get()
        B[i, i + 1:] = B[i + 1:, i] = Bi.get()
    return E, B


def mulliken_atom_pops(mol, S, C_lmo):
    """Psi4's Mulliken population of each LMO on each atom (natm, nlmo)."""
    bf_atom = np.array([s[0] for s in mol.ao_labels(fmt=False)])
    natm = mol.natm
    out = np.zeros((natm, C_lmo.shape[1]))
    for i in range(C_lmo.shape[1]):
        c = C_lmo[:, i]
        p = c[:, None] * S * c[None, :]
        dd = np.diag(p)
        den = dd[:, None] + dd[None, :]
        with np.errstate(invalid='ignore', divide='ignore'):
            w = np.where(den != 0, dd[:, None] / den, 0.)
        per_bf = 2 * np.sum(p * w, axis=1)
        out[:, i] = np.bincount(bf_atom, per_bf, minlength=natm)
    return out


def basis_maps(mol, auxmol):
    bf_atom = np.array([s[0] for s in mol.ao_labels(fmt=False)])
    ribf_atom = np.array([s[0] for s in auxmol.ao_labels(fmt=False)])
    atom_bfs = [np.where(bf_atom == a)[0] for a in range(mol.natm)]
    atom_ribfs = [np.where(ribf_atom == a)[0] for a in range(mol.natm)]
    return bf_atom, ribf_atom, atom_bfs, atom_ribfs


class Domains:
    """LMO aux / PAO domains for one setting of (T_CUT_MKN, T_CUT_DO)."""

    def __init__(self, mkn_pops, doi_iu, t_cut_mkn, t_cut_do, bf_atom,
                 atom_bfs, atom_ribfs):
        n = doi_iu.shape[0]
        self.lmo_riatoms = [np.where(np.abs(mkn_pops[:, i]) > t_cut_mkn)[0]
                            for i in range(n)]
        self.lmo_ribfs = [np.concatenate([atom_ribfs[a] for a in at]).astype(int)
                          if len(at) else np.zeros(0, int)
                          for at in self.lmo_riatoms]
        self.lmo_paos = [contract_to_atoms(np.where(np.abs(doi_iu[i]) > t_cut_do)[0],
                                           bf_atom, atom_bfs) for i in range(n)]


def union(*lists):
    if not lists:
        return np.zeros(0, int)
    return np.unique(np.concatenate([np.asarray(l, dtype=int) for l in lists]))


class PairList:
    """Ordered LMO pairs (both ij and ji) with dense lookup."""

    def __init__(self, n, mask):
        self.n = n
        self.ij = -np.ones((n, n), dtype=int)
        pairs = [(i, j) for i in range(n) for j in range(n) if mask[i, j]]
        self.pairs = pairs
        for p, (i, j) in enumerate(pairs):
            self.ij[i, j] = p
        self.ji = np.array([self.ij[j, i] for (i, j) in pairs], dtype=int)

    def __len__(self):
        return len(self.pairs)

    def lmos(self, p):
        """LMOs m with (i,m) and (j,m) both valid pairs."""
        i, j = self.pairs[p]
        return np.where((self.ij[i] >= 0) & (self.ij[j] >= 0))[0]
