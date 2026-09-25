"""The density screen of the three-centre Coulomb build.

`docs/DF_WORK_LIST.md` **W20**.  ``J`` is linear in ``D``, so ``J(D) =
J(D_last) + J(D - D_last)`` exactly, and near convergence ``D - D_last`` is
small almost everywhere -- but GPU4PySCF's three-centre build walks every
stored shell pair against every auxiliary shell on every cycle.  This module
holds what a screen on the difference needs, all of it fixed at
``Int3c2eOpt.build()``:

* ``q_ij = sqrt(max (ij|ij))`` for every stored pair, which the build already
  computed (as a logarithm, in ``bas_pair_cache``) and kept;
* ``q_P = sqrt(max (P|P))`` for every sorted auxiliary shell, in closed form
  (`self_coulomb`) once per distinct ``(l, exponent, coefficient)`` -- a
  hundred or so for a 113-atom molecule, whatever its size.  Not through
  libcint: a one-shell ``int2c2e`` costs 0.2 s on a loaded login node, where
  its OpenMP threads spin, and a 2454-shell basis would pay that a hundred
  times;
* the shell structure of the sorted basis, so that the largest ``|dD|`` in a
  shell-pair block is a handful of reductions on the card.

A pair can leave the density contraction when ``q_ij max|dD_ij| max_P q_P`` is
below the threshold, and the auxiliary-vector contraction when
``q_ij sum_P q_P max|dc_P|`` is -- both Schwarz bounds on everything the pair
contributes.  `benchmarks/df_j3c_screen_probe.py` is what those two keep
through an SCF.

This module patches nothing.
"""
import numpy as np
import cupy as cp

_ATTR = '_electrolites_j3cinc'


class Screen:
    """The Schwarz factors and shell structure of one ``Int3c2eOpt``."""

    def __init__(self, opt, c):
        self.envs = opt.int3c2e_envs
        mol, aux = opt.mol, opt.auxmol
        nbas = mol.nbas
        self.q_pair = cp.exp(cp.hstack(
            [v[1] for v in opt.bas_pair_cache.values()]).astype(np.float64))
        assert len(self.q_pair) == c.npair
        idx = cp.asarray(opt.shl_pair_idx, dtype=np.int64)
        self.ish = idx // nbas
        self.jsh = idx % nbas
        self.nbas = nbas

        ls = np.asarray(mol._bas[:, 1])
        ao_loc = np.asarray(mol.ao_loc)
        self.runs = _runs(ls, ao_loc)
        aux_ls = np.asarray(aux._bas[:, 1])
        self.aux_runs = _runs(aux_ls, np.asarray(aux.ao_loc))
        self.naux_sh = aux.nbas

        cache = {}
        qP = np.empty(aux.nbas)
        for k in range(aux.nbas):
            key = (int(aux_ls[k]), float(aux._env[aux._bas[k, 5]]),
                   float(aux._env[aux._bas[k, 6]]))
            v = cache.get(key)
            if v is None:
                v = cache[key] = self_coulomb(*key) ** .5
            qP[k] = v
        self.q_aux = cp.asarray(qP)
        self.q_aux_max = float(qP.max())
        self.q_pair_max = float(cp.asnumpy(self.q_pair.max()))

    def pair_absmax(self, m):
        """``max |m|`` over each stored pair's shell block of a sorted-basis
        matrix, in ``shl_pair_idx`` order."""
        m = cp.abs(m)
        nbas = self.nbas
        rows = cp.empty((nbas, m.shape[1]))
        for s0, s1, a0, nf in self.runs:
            rows[s0:s1] = m[a0:a0 + (s1 - s0) * nf].reshape(
                s1 - s0, nf, -1).max(1)
        blk = cp.empty((nbas, nbas))
        for s0, s1, a0, nf in self.runs:
            blk[:, s0:s1] = rows[:, a0:a0 + (s1 - s0) * nf].reshape(
                nbas, s1 - s0, nf).max(2)
        return blk[self.ish, self.jsh]

    def aux_absmax(self, v):
        """``max |v|`` over each sorted auxiliary shell of a vector in the
        sorted (spherical-index) space ``contract_auxvec`` is handed."""
        v = cp.abs(v)
        out = cp.empty(self.naux_sh)
        for s0, s1, a0, nf in self.aux_runs:
            out[s0:s1] = v[a0:a0 + (s1 - s0) * nf].reshape(s1 - s0, nf).max(1)
        return out

    def keep_dm(self, ddm, tol):
        """Pairs the density contraction of ``ddm`` cannot drop."""
        return self.q_pair * self.pair_absmax(ddm) * self.q_aux_max >= tol

    def keep_aux(self, dc, tol):
        """Pairs the auxiliary-vector contraction of ``dc`` cannot drop."""
        s = float(cp.dot(self.q_aux, self.aux_absmax(dc)))
        return self.q_pair * s >= tol


def _dfact(n):
    """``n!!`` for odd ``n``, with ``(-1)!! = 1``."""
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


def self_coulomb(l, a, c):
    """``max (P|P)`` over the Cartesian components of one primitive shell.

    McMurchie--Davidson at zero separation: with ``x^i = sum_t E^i_t
    Lambda_t`` (`gen_j3c.e_coeffs`) and the reduced exponent ``a/2``, only
    even ``t+t'`` survive and ``R_{2i,2j,2k} = (2i-1)!!(2j-1)!!(2k-1)!!
    (-a)^N / (2N+1)``.  The s and p factors are libcint's ``common_fac_sp``,
    which PySCF's normalised coefficients carry; with them this is
    ``SortedMole.intor('int2c2e')``'s diagonal to the last digit.
    """
    import math
    from .codegen.gen_j3c import e_coeffs, cart_list
    best = 0.
    for comp in cart_list(l):
        ex = [e_coeffs(i) for i in comp]
        tot = 0.
        for t, (c1, p1) in ex[0].items():
            for t2, (c2, p2) in ex[0].items():
                if (t + t2) % 2:
                    continue
                for u, (c3, p3) in ex[1].items():
                    for u2, (c4, p4) in ex[1].items():
                        if (u + u2) % 2:
                            continue
                        for v, (c5, p5) in ex[2].items():
                            for v2, (c6, p6) in ex[2].items():
                                if (v + v2) % 2:
                                    continue
                                i, j, k = (t + t2) // 2, (u + u2) // 2, \
                                    (v + v2) // 2
                                n = i + j + k
                                r = (_dfact(2*i - 1) * _dfact(2*j - 1)
                                     * _dfact(2*k - 1) * (-a) ** n / (2*n + 1))
                                e = (c1 * c2 * c3 * c4 * c5 * c6
                                     * (.5 / a) ** (p1 + p2 + p3 + p4 + p5
                                                    + p6))
                                tot += e * (-1) ** (t2 + u2 + v2) * r
        best = max(best, abs(tot))
    fac = {0: 1. / (4 * math.pi), 1: 3. / (4 * math.pi)}.get(l, 1.)
    return best * fac * c * c * 2 * math.pi ** 2.5 / (a * a * math.sqrt(2 * a))


def _runs(ls, ao_loc):
    """``[(first shell, end shell, first function, functions per shell)]``
    for each run of equal angular momentum in a sorted basis."""
    runs = []
    n = len(ls)
    s = 0
    while s < n:
        e = s
        while e < n and ls[e] == ls[s]:
            e += 1
        runs.append((s, e, int(ao_loc[s]), int(ao_loc[s + 1] - ao_loc[s])))
        s = e
    return runs


def screen(opt, c):
    """The cached `Screen` of one ``Int3c2eOpt`` on this device."""
    by_device = getattr(opt, _ATTR, None)
    if by_device is None:
        by_device = {}
        setattr(opt, _ATTR, by_device)
    key = cp.cuda.Device().id
    s = by_device.get(key)
    if s is None or s.envs is not opt.int3c2e_envs:
        s = Screen(opt, c)
        by_device[key] = s
    return s


__all__ = ['screen', 'Screen']
