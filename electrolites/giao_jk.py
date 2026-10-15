"""The GIAO two-electron term of NMR shielding on the GPU, through ordinary
J/K builds (`docs/FEATURE_TARGETS_PYSCF_2.13.md` O1, KT:C6).

GPU4PySCF (and pyscf-properties) form the first-order GIAO Coulomb and
exchange matrices from libcint's ``int2e_ig1`` on the host
(``_vhf.direct_mapdm``): 98 % of a GPU NMR run at 42 atoms, 914 of 930 s at
113.  No new integral kernel is needed.  libcint's operator is

    (g_x i j|k l) = -1/2 [(R_i - R_j) x (i r j|k l)]_x

(checked on ``int1e_igovlp`` and on the result below), and r chi_j =
(r - R_j) chi_j + R_j chi_j, where (r - R_j)_k chi_j is a Cartesian Gaussian
of one higher angular momentum on the same centre with the same
exponents.  So with a *shifted* basis -- every shell's l raised by one, the
contraction coefficients rescaled by libcint's s/p factors -- the GIAO
matrices are blocks of plain J/K builds on the concatenated (Cartesian)
basis, with the density on the original block only:

    J^g = -1/2 [ (R_i - R_j) x M  +  (R_i x R_j) J ],
        M_k[i, j] = (i, x_k j | D)          (original x shifted block of J)
    K^g = -1/2 [ R_i x K'[D]  -  (K'_z[D_y] - K'_y[D_z], ...)
                 - (R_i,z K[D_y] - R_i,y K[D_z], ...) ],
        K'_k[i, l] = sum (x_k i, j|k l) D_jk,  D_a = diag(R_a) D

(``-vj``, ``-(vk - vk^T)`` as GPU4PySCF's ``shielding.get_jk`` returns
them).  One J build and one four-density K build on a basis about 3.5x
larger (GPU4PySCF's direct J/K, LMAX 4: up to f in the original basis).
Agrees with ``int2e_ig1`` to round-off (``tests/test_giao_jk.py``).
"""
import numpy as np
import cupy as cp

from pyscf import gto
from pyscf.gto.mole import ATOM_OF, ANG_OF, PTR_COEFF, NPRIM_OF, NCTR_OF

_SP_FAC = {0: 1 / np.sqrt(3), 1: 0.488602511902919921}


def _cart_list(l):
    return [(lx, ly, l - lx - ly) for lx in range(l, -1, -1)
            for ly in range(l - lx, -1, -1)]


def _shifted(molc):
    """Cartesian copy of ``molc`` with l + 1 on every shell and coefficients
    making its functions (r - R)_k times ``molc``'s Cartesian functions."""
    m2 = molc.copy()
    env = m2._env.copy()
    bas = m2._bas.copy()
    done = set()
    for ib in range(len(bas)):
        l = bas[ib, ANG_OF]
        p = bas[ib, PTR_COEFF]
        n = bas[ib, NPRIM_OF] * bas[ib, NCTR_OF]
        if p not in done:                 # shells of one element share it
            env[p:p + n] *= _SP_FAC.get(l, 1.)
            done.add(p)
        bas[ib, ANG_OF] = l + 1
    m2._env = env
    m2._bas = bas
    return m2


class GIAOJK:
    """The concatenated basis and index maps of one molecule (cached)."""

    def __init__(self, mol):
        molc = mol.copy()
        molc.cart = True
        molc.build(False, False)
        m2 = _shifted(molc)
        self.cmol = gto.conc_mol(molc, m2)
        self.cmol.cart = True
        self.n1 = n1 = molc.nao_cart()
        loc1 = molc.ao_loc_nr(cart=True)
        loc2 = m2.ao_loc_nr(cart=True)
        idx = np.zeros((3, n1), dtype=np.int64)
        for ib in range(molc.nbas):
            l = molc._bas[ib, ANG_OF]
            cl, cl2 = _cart_list(l), _cart_list(l + 1)
            for ic in range(molc._bas[ib, NCTR_OF]):
                for ci, c in enumerate(cl):
                    for k in range(3):
                        cc = list(c)
                        cc[k] += 1
                        idx[k, loc1[ib] + ic * len(cl) + ci] = \
                            n1 + loc2[ib] + ic * len(cl2) + cl2.index(tuple(cc))
        self.idx = cp.asarray(idx)
        self.c2s = cp.asarray(mol.cart2sph_coeff()) if not mol.cart \
            else cp.eye(n1)
        aoloc = mol.ao_loc_nr()
        self.R = cp.asarray(mol.atom_coords()[np.repeat(
            mol._bas[:, ATOM_OF], aoloc[1:] - aoloc[:-1])].T)   # (3, nao)
        self.ntot = self.cmol.nao_cart()
        self._opt = None

    def _vhfopt(self):
        # GPU4PySCF's get_j/get_k build their screening/sorting object on
        # every call otherwise (0.6 s at penicillin V's concatenated basis)
        from gpu4pyscf.scf import jk
        if self._opt is None:
            self._opt = jk._VHFOpt(self.cmol).build()
        return self._opt

    def _pad(self, dms):
        big = cp.zeros((len(dms), self.ntot, self.ntot))
        big[:, :self.n1, :self.n1] = self.c2s @ dms @ self.c2s.T
        return big

    def _sph(self, m):
        return self.c2s.T @ m @ self.c2s

    def get_jk(self, dm, with_k=True, with_j=True):
        """GIAO (J, K) for dm (nao, nao) or (nset, nao, nao), each (3, nao,
        nao) per set, in ``shielding.get_jk``'s convention."""
        from gpu4pyscf.scf import jk
        dm = cp.asarray(dm)
        dms = dm.reshape(-1, *dm.shape[-2:])
        nset, nao = dms.shape[0], dms.shape[-1]
        R = self.R
        Ri = R[:, :, None]
        Rj = R[:, None, :]
        dRx = cp.stack([Ri[1] * Rj[2] - Ri[2] * Rj[1],
                        Ri[2] * Rj[0] - Ri[0] * Rj[2],
                        Ri[0] * Rj[1] - Ri[1] * Rj[0]])         # R_i x R_j
        dR = Ri - Rj
        n1 = self.n1
        vjs, vks = [], []
        for d in dms:
            if with_j:
                big = self._pad(d[None])
                vj = jk.get_j(self.cmol, big, hermi=1,
                              vhfopt=self._vhfopt())[0]
                J = self._sph(vj[:n1, :n1])
                Mp = cp.stack([self._sph(vj[:n1, self.idx[k]])
                               for k in range(3)])
                raw = -.5 * (_cross(dR, Mp) + dRx * J)
                vjs.append(-raw)
            if not with_k:
                continue
            dd = cp.stack([d] + [R[a][:, None] * d for a in range(3)])
            vk = jk.get_k(self.cmol, self._pad(dd), hermi=0,
                          vhfopt=self._vhfopt())
            Kd = [self._sph(vk[i][:n1, :n1]) for i in range(4)]
            Kp = [cp.stack([self._sph(vk[i][self.idx[k], :n1])
                            for k in range(3)]) for i in range(4)]
            Rb = cp.broadcast_to(Ri, (3, nao, nao))
            t1 = _cross(Rb, Kp[0])
            t2 = -cp.stack([Kp[2][2] - Kp[3][1], Kp[3][0] - Kp[1][2],
                            Kp[1][1] - Kp[2][0]])
            t3 = -cp.stack([Rb[2] * Kd[2] - Rb[1] * Kd[3],
                            Rb[0] * Kd[3] - Rb[2] * Kd[1],
                            Rb[1] * Kd[1] - Rb[0] * Kd[2]])
            raw = -.5 * (t1 + t2 + t3)
            vks.append(-(raw - raw.transpose(0, 2, 1)))
        shape = dm.shape[:-2] + (3, nao, nao)
        vj = cp.stack(vjs).reshape(shape) if with_j else None
        vk = cp.stack(vks).reshape(shape) if with_k else None
        return vj, vk


def _cross(a, b):
    return cp.stack([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


_CACHE = {}

# Below this many AOs the host int2e_ig1 (16 threads, the GPU node's share)
# is faster than the concatenated-basis J/K, whose cost there is GPU4PySCF's
# per-angular-momentum-class overhead (glycine/def2-SVP, 95 AOs: 1.7 s on the
# device, 1.3 s on the host; alanine dipeptide, 200 AOs: 2.8 s against 5.7 s)
NAO_MIN = 150


def use_gpu(mol):
    """The GIAO two-electron term on the GPU for this molecule?
    ``FASTNMR_HOST2E=1`` / ``FASTNMR_GPU2E=1`` force either side."""
    import os
    if os.environ.get('FASTNMR_HOST2E'):
        return False
    if os.environ.get('FASTNMR_GPU2E'):
        return True
    return mol.nao >= NAO_MIN


def get_jk(mol, dm, with_k=True, with_j=True):
    """GIAO J/K on the GPU for ``mol`` (the shifted basis cached per mol)."""
    key = id(mol)
    obj = _CACHE.get(key)
    if obj is None or obj[0] is not mol:
        obj = (mol, GIAOJK(mol))
        _CACHE.clear()
        _CACHE[key] = obj
    return obj[1].get_jk(dm, with_k, with_j)
