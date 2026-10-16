"""GHF TDA and TDHF on the GPU.

GPU4PySCF has a GHF SCF (complex two-component densities, J/K through its
spin-free J/K on the alpha/beta blocks) but no GHF response and no GHF
TDDFT; PySCF's ``tdscf.ghf`` runs on the host.  Here PySCF's GHF TDA/TDHF
classes keep their guesses and eigensolvers, and the products with A (and
B) run on the device: the trial vectors' transition densities
C_v X C_o^H (+ C_o Y^T C_v^T for TDHF), GPU4PySCF's GHF J - K on the
whole batch, and the transformation back.  Hartree-Fock only (GKS needs a
non-collinear XC kernel).

    td = electrolites.ghf_td_gpu.TDA(gpu_ghf).run(nstates=6)
"""
import numpy as np
import cupy as cp

from pyscf.tdscf import ghf as _ghf


def _setup(td):
    mf = td._gpu_scf
    mask = td.get_frozen_mask()
    C = cp.asarray(mf.mo_coeff)[:, cp.asarray(mask)]
    e = cp.asarray(mf.mo_energy)[cp.asarray(mask)]
    occ = cp.asnumpy(mf.mo_occ)[mask]
    o = np.where(occ == 1)[0]
    v = np.where(occ == 0)[0]
    orbo, orbv = C[:, o], C[:, v]
    e_ia = e[v][None, :] - e[o][:, None]
    return mf, orbo, orbv, e_ia


def _vresp(mf, dms):
    """J - K for a batch of GHF densities (n, 2nao, 2nao).

    Not ``mf.get_jk``: for a batch of real densities GPU4PySCF 1.8.1's GHF
    J/K adapter stacks the four spin blocks of all densities and then reads
    back only the first four K matrices (``scf/ghf.py``, real branch), so
    every density but the first gets the first one's exchange."""
    from gpu4pyscf.scf import hf
    if mf.istype('KohnShamDFT'):
        raise NotImplementedError('GHF TDDFT on the GPU: Hartree-Fock only')
    mol = mf.mol
    n, nso = dms.shape[0], dms.shape[-1]
    nao = nso // 2
    blocks = [dms[:, :nao, :nao], dms[:, nao:, nao:],
              dms[:, :nao, nao:], dms[:, nao:, :nao]]
    parts = [blocks]
    if cp.iscomplexobj(dms):
        parts = [[b.real for b in blocks], [b.imag for b in blocks]]
    out = []
    for bl in parts:
        stack = cp.ascontiguousarray(cp.concatenate(bl))       # (4n, nao, nao)
        vj = hf.SCF.get_j(mf, mol, cp.ascontiguousarray(bl[0] + bl[1]),
                          hermi=0)
        vk = hf.SCF.get_k(mf, mol, stack, hermi=0).reshape(4, n, nao, nao)
        v = cp.zeros((n, nso, nso))
        vj = vj.reshape(n, nao, nao)
        v[:, :nao, :nao] = vj - vk[0]
        v[:, nao:, nao:] = vj - vk[1]
        v[:, :nao, nao:] = -vk[2]
        v[:, nao:, :nao] = -vk[3]
        out.append(v)
    if len(out) == 2:
        return out[0] + 1j * out[1]
    return out[0]


#: trial vectors per J/K batch
_BLK = 32


class _GPUTD:
    def __init__(self, mf):
        if getattr(mf, 'device', 'cpu') != 'gpu':
            raise ValueError('a GPU4PySCF GHF object is expected')
        self._gpu_scf = mf
        super().__init__(mf.to_cpu())


class TDA(_GPUTD, _ghf.TDA):
    """PySCF's GHF TDA with A x on the device."""

    def gen_vind(self, mf=None):
        mf, orbo, orbv, e_ia = _setup(self)
        no, nv = e_ia.shape
        hdiag = cp.asnumpy(e_ia).ravel().real

        def vind(zs):
            zs = np.asarray(zs).reshape(-1, no, nv)
            out = np.empty((len(zs), no * nv), dtype=np.result_type(
                zs.dtype, orbo.dtype))
            for p0 in range(0, len(zs), _BLK):
                z = cp.asarray(zs[p0:p0 + _BLK])
                dms = orbv @ z.transpose(0, 2, 1) @ orbo.conj().T
                v1 = _vresp(mf, dms)
                v1mo = (orbv.conj().T @ v1 @ orbo).transpose(0, 2, 1)
                v1mo = v1mo + z * e_ia
                out[p0:p0 + _BLK] = cp.asnumpy(v1mo.reshape(len(z), -1))
            return out

        return vind, hdiag


class TDHF(_GPUTD, _ghf.TDHF):
    """PySCF's GHF TDHF (RPA) with the [A B; -B* -A*] product on the
    device."""

    def gen_vind(self, mf=None):
        mf, orbo, orbv, e_ia = _setup(self)
        no, nv = e_ia.shape
        d = cp.asnumpy(e_ia).ravel().real
        hdiag = np.hstack((d, -d))

        def vind(xys):
            xys = np.asarray(xys).reshape(-1, 2, no, nv)
            out = np.empty((len(xys), 2 * no * nv), dtype=np.result_type(
                xys.dtype, orbo.dtype))
            for p0 in range(0, len(xys), _BLK):
                xy = cp.asarray(xys[p0:p0 + _BLK])
                xs, ys = xy[:, 0], xy[:, 1]
                dms = orbv @ xs.transpose(0, 2, 1) @ orbo.conj().T
                dms += orbo @ ys @ orbv.conj().T
                v1 = _vresp(mf, dms)
                top = (orbv.conj().T @ v1 @ orbo).transpose(0, 2, 1)
                bot = orbo.conj().T @ v1 @ orbv
                top = top + xs * e_ia
                bot = bot + ys * e_ia
                nz = len(xy)
                out[p0:p0 + _BLK] = cp.asnumpy(cp.hstack(
                    (top.reshape(nz, -1), -bot.reshape(nz, -1))))
            return out

        return vind, hdiag


RPA = TDHF
