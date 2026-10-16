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
        loc = lo.Boys(mol, c_act)
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
    g = gen_grid.Grids(mol)
    g.atom_grid = grid
    g.radi_method = radi.treutler
    g.prune = None
    g.verbose = 0
    g.build(with_non0tab=False)
    return g.coords, g.weights


def compute_doi(mol, C_lmo, C_pao, grid=(25, 50), blksize=4096, need_uv=True):
    """sqrt(int |i|^2 |u|^2): DOI_ij, DOI_iu and (optionally) DOI_uv."""
    coords, weights = doi_grid(mol, grid)
    nl, nu = C_lmo.shape[1], C_pao.shape[1]
    ij = np.zeros((nl, nl))
    iu = np.zeros((nl, nu))
    uv = np.zeros((nu, nu)) if need_uv else None
    for p0 in range(0, len(weights), blksize):
        p1 = min(p0 + blksize, len(weights))
        ao = numint.eval_ao(mol, coords[p0:p1])
        li = (ao @ C_lmo) ** 2
        lu = (ao @ C_pao) ** 2
        w = weights[p0:p1, None]
        ij += (li * w).T @ li
        iu += (li * w).T @ lu
        if need_uv:
            uv += (lu * w).T @ lu
    return (np.sqrt(np.maximum(ij, 0)), np.sqrt(np.maximum(iu, 0)),
            np.sqrt(np.maximum(uv, 0)) if need_uv else None)


def dipole_pair_energies(orb, doi_iu, t_cut_do_pre, s_cut, bf_atom, atom_bfs):
    """Approximate SC-MP2 pair energies of distant pairs from dipole
    integrals (Pinski EQ 17).  Returns (estimate, bound) matrices."""
    mol = orb.mol
    with mol.with_common_orig((0, 0, 0)):
        r = mol.intor_symmetric('int1e_r')
    L, P = orb.C_lmo, orb.C_pao
    n = orb.naocc
    R = np.einsum('xmi,mi->ix', np.einsum('xmn,ni->xmi', r, L), L)
    lp = np.einsum('mi,xmn,nu->xiu', L, r, P)
    dr, ee = [], []
    for i in range(n):
        idx = contract_to_atoms(np.where(np.abs(doi_iu[i]) > t_cut_do_pre)[0],
                                bf_atom, atom_bfs)
        X, e = orthocanonicalize(orb.S_pao[np.ix_(idx, idx)],
                                 orb.F_pao[np.ix_(idx, idx)], s_cut)
        dr.append(lp[:, i, idx] @ X)          # (3, nu_i)
        ee.append(e)
    E = np.zeros((n, n))
    B = np.zeros((n, n))
    f = np.diag(orb.F_lmo)
    for i in range(n):
        for j in range(i + 1, n):
            Rij = R[i] - R[j]
            d = np.linalg.norm(Rij)
            rh = Rij / d
            iu, jv = dr[i], dr[j]
            dot = iu.T @ jv
            num = (dot - 3 * np.outer(rh @ iu, rh @ jv)) ** 2
            numb = (-2 * dot) ** 2
            den = ee[i][:, None] + ee[j][None, :] - f[i] - f[j]
            E[i, j] = E[j, i] = -4 * d ** -6 * np.sum(num / den)
            B[i, j] = B[j, i] = -4 * d ** -6 * np.sum(numb / den)
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
