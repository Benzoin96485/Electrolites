"""GIAO NMR shielding for UHF/UKS on the GPU
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` O1, the open-shell half).

GPU4PySCF's ``properties.shielding`` is RHF/RKS only (a ValueError on a UKS
object).  This is pyscf-properties' ``prop.nmr.uhf``/``uks`` (GIAO gauge)
on a GPU4PySCF UHF/UKS:

* the spin-resolved GIAO XC matrix on the device, by the regrouped GEMM of
  ``electrolites.fastnmr`` with the spin-polarised kernel (one pass over
  the grid for both spins, the GIAO AO factors built from the block's AOs);
* the GIAO two-electron term on the device through shifted-basis J/K
  (``electrolites.giao_jk``; ``host_2e=True``: libcint ``int2e_ig1`` on
  the host as GPU4PySCF's RKS path), K only for hybrids;
* the coupled equations with the first-order overlap (pyscf ``ucphf``
  ``solve_withs1``: occupied-occupied block -S1/2, Krylov in the
  virtual-occupied block) on the device, the response
  ``mf.gen_response(hermi=2)`` (only -hyb K for imaginary perturbations;
  pure functionals are uncoupled, as pyscf-properties);
* diamagnetic and paramagnetic terms from the per-nucleus one-electron
  operators (host libcint, one set per nucleus, as GPU4PySCF's RKS path).

    sh = electrolites.nmr_uks.NMR(gpu_uks).kernel()     # (natm, 3, 3) ppm
"""
import numpy as np
import cupy as cp

from pyscf import lib
from pyscf.lib import logger
from pyscf.data import nist
from pyscf.scf import _vhf


def krylov_refined(aop, b, tol=1e-9, max_cycle=20, max_refine=3, rtol=None):
    """(1 + A) x = b for rows of b by GPU4PySCF's Krylov solver plus
    iterative refinement.  That solver stops once a new subspace vector's
    squared norm drops below its absolute ``lindep`` (1e-13), i.e. near a
    1e-6 relative residual whatever ``tol`` (UHF water NMR: CPHF residual
    2.6e-5 against pyscf's 2.3e-7); each refinement re-solves for the
    residual scaled to unit norm, multiplying the accuracy by the same
    factor."""
    from gpu4pyscf.lib.cupy_helper import krylov
    b = cp.asarray(b)
    x = krylov(aop, b, tol=tol, max_cycle=max_cycle)
    bnorm = cp.linalg.norm(b, axis=1)
    for _ in range(max_refine):
        r = b - x - aop(x)
        rn = cp.linalg.norm(r, axis=1)
        if float((rn / cp.maximum(bnorm, 1e-300)).max()) < (rtol or tol):
            break
        keep = rn > 0
        sc = cp.where(keep, rn, 1.)
        x = x + krylov(aop, r / sc[:, None], tol=tol,
                       max_cycle=max_cycle) * sc[:, None]
    return x


def get_vxc_giao(ni, mol, grids, xc_code, mo_coeff, mo_occ):
    """Spin-resolved GIAO XC matrices (2, 3, nao, nao), GPU4PySCF's sign
    convention (``shielding.nr_rks``)."""
    from gpu4pyscf.dft import numint
    from gpu4pyscf.lib.cupy_helper import add_sparse, sandwich_dot
    from electrolites.fastnmr import _ao_centres, _cross
    xctype = ni._xc_type(xc_code)
    if xctype not in ('LDA', 'GGA', 'HF'):
        raise NotImplementedError(f'GPU UKS NMR: {xctype} functionals')
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    smol = opt._sorted_mol
    coeff = cp.asarray(opt.coeff)
    nao, nao0 = coeff.shape
    vmat = cp.zeros((2, 3, nao, nao))
    if xctype == 'HF':
        return vmat[:, :, :nao0, :nao0]
    mos = [coeff.dot(cp.asarray(mo_coeff[s])) for s in (0, 1)]
    occs = [cp.asarray(mo_occ[s]) for s in (0, 1)]
    ao_deriv = 0 if xctype == 'LDA' else 1
    centres = _ao_centres(smol)
    for ao, index, weight, coords in ni.block_loop(smol, grids, nao,
                                                    ao_deriv):
        rho = cp.stack([numint.eval_rho2(smol, ao, mos[s][index, :],
                                         occs[s], None, xctype)
                        for s in (0, 1)])
        vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype,
                             spin=1)[1]
        R = centres[index]
        r = coords.T
        C = _cross(R, r)
        nsub, ng = R.shape[0], r.shape[1]
        if xctype == 'GGA':
            B = cp.ascontiguousarray(ao[0:4].transpose(0, 2, 1)
                                     ).reshape(4 * ng, nsub)
        for s in (0, 1):
            if xctype == 'LDA':
                wv = weight * vxc[s, 0]
                L = (.5 * C) * (ao * wv)
                v = L.reshape(3 * nsub, ng) @ ao.T
            else:
                wv = vxc[s] * weight
                ao0 = ao[0]
                fw = ao0 * wv[0] + (ao[1:4] * wv[1:4, None, :]).sum(axis=0)
                L = cp.empty((3, nsub, 4, ng))
                L[:, :, 0] = C * fw
                L[:, :, 0] += _cross(R, wv[1:4]) * ao0
                for x in range(3):
                    L[:, :, 1 + x] = C * (ao0 * wv[1 + x])
                L *= .5
                v = L.reshape(3 * nsub, 4 * ng) @ B
            v = v.reshape(3, nsub, nsub)
            for d in range(3):
                add_sparse(vmat[s, d], cp.ascontiguousarray(v[d]), index)
    vmat = cp.stack([sandwich_dot(vmat[s], coeff) for s in (0, 1)])
    return vmat - vmat.transpose(0, 1, 3, 2)


def get_jk_giao(mol, dm, with_k=True, host=False):
    """GIAO J (and K) for spin densities dm (2, nao, nao): on the GPU by
    ``electrolites.giao_jk``, or (``host``) from libcint ``int2e_ig1``
    (pyscf-properties' ``rhf.get_jk``)."""
    if not host:
        from electrolites import giao_jk
        return giao_jk.get_jk(mol, dm, with_k=with_k)
    intor = mol._add_suffix('int2e_ig1')
    dm = cp.asnumpy(dm)
    if with_k:
        vj, vk = _vhf.direct_mapdm(intor, 'a4ij', ('lk->s1ij', 'jk->s1il'),
                                   dm, 3, mol._atm, mol._bas, mol._env)
        vk = vk - np.swapaxes(vk, -1, -2)
        return -cp.asarray(vj), -cp.asarray(vk)
    vj = _vhf.direct_mapdm(intor, 'a4ij', 'lk->s1ij', dm, 3,
                           mol._atm, mol._bas, mol._env)
    return -cp.asarray(vj), None


class NMR(lib.StreamObject):
    """GIAO shielding (ppm) of a GPU4PySCF UHF/UKS; ``shielding_nuc``,
    ``cphf``, ``conv_tol``, ``max_cycle_cphf`` as pyscf-properties."""

    def __init__(self, mf):
        self._scf = mf
        self.mol = mf.mol
        self.verbose = mf.mol.verbose
        self.stdout = mf.mol.stdout
        self.shielding_nuc = range(mf.mol.natm)
        self.cphf = True
        self.max_cycle_cphf = 20
        self.conv_tol = 1e-9
        self.mo10 = None
        self.host_2e = False
        self.timings = {}

    def _hyb(self):
        mf = self._scf
        if not hasattr(mf, 'xc'):
            return 0., 1.
        omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(
            mf.xc, spin=mf.mol.spin)
        if omega != 0:
            raise NotImplementedError('GPU UKS NMR: range-separated hybrids')
        return omega, hyb

    def get_fock(self):
        """Imaginary first-order Fock matrices (2, 3, nao, nao)."""
        mf = self._scf
        mol = self.mol
        dm0 = cp.asarray(mf.make_rdm1())
        hyb = self._hyb()[1]
        if hasattr(mf, 'xc'):
            h1 = cp.asarray(get_vxc_giao(mf._numint, mol, mf.grids, mf.xc,
                                         mf.mo_coeff, mf.mo_occ))
        else:
            h1 = cp.zeros((2, 3, mol.nao, mol.nao))
        vj, vk = get_jk_giao(mol, dm0, with_k=abs(hyb) > 1e-10,
                             host=self.host_2e)
        h1 += (vj[0] + vj[1])[None]
        if vk is not None:
            h1 -= hyb * vk
        h1e = .5 * mol.intor('int1e_giao_irjxp', 3)
        h1e += mol.intor_asymmetric('int1e_ignuc', 3)
        if mol.has_ecp():
            h1e += mol.intor_asymmetric('ECPscalar_ignuc', 3)
        h1e += mol.intor('int1e_igkin', 3)
        return h1 - cp.asarray(h1e)[None]

    def solve_mo1(self, h1ao=None):
        """mo10[s] (3, nmo, nocc_s): pyscf ``ucphf.solve`` with S1."""
        mf = self._scf
        if h1ao is None:
            h1ao = self.get_fock()
        s1ao = cp.asarray(-self.mol.intor_asymmetric('int1e_igovlp', 3))
        C = [cp.asarray(mf.mo_coeff[s]) for s in (0, 1)]
        occ = [cp.asnumpy(mf.mo_occ[s]) > 0 for s in (0, 1)]
        e = [cp.asarray(mf.mo_energy[s]) for s in (0, 1)]
        co = [C[s][:, occ[s]] for s in (0, 1)]
        nocc = [int(o.sum()) for o in occ]
        h1 = [C[s].T @ h1ao[s] @ co[s]
              for s in (0, 1)]
        s1 = [C[s].T @ s1ao @ co[s]
              for s in (0, 1)]
        hs = [h1[s] - s1[s] * e[s][occ[s]] for s in (0, 1)]
        eai = [e[s][~occ[s], None] - e[s][occ[s]][None] for s in (0, 1)]
        mo1 = []
        for s in (0, 1):
            m = cp.empty_like(hs[s])
            m[:, ~occ[s]] = -hs[s][:, ~occ[s]] / eai[s]
            m[:, occ[s]] = -.5 * s1[s][:, occ[s]]
            mo1.append(m)
        _, hyb = self._hyb()
        if not self.cphf or abs(hyb) < 1e-10:
            return mo1
        vresp = mf.gen_response(hermi=2)
        nv = [int((~occ[s]).sum()) for s in (0, 1)]

        def vind(m):
            """the virtual-occupied response to full mo1 (3 x 2 spins)."""
            dm = cp.stack([C[s] @ m[s] @ co[s].T
                           for s in (0, 1)])
            dm = dm - dm.transpose(0, 1, 3, 2)
            v = cp.asarray(vresp(dm))
            return [(C[s][:, ~occ[s]].T @ v[s] @
                              co[s]) for s in (0, 1)]

        n0 = nv[0] * nocc[0]

        def pack(a, b):
            return cp.hstack([a.reshape(len(a), -1), b.reshape(len(b), -1)])

        def unpack(x):
            return (x[:, :n0].reshape(-1, nv[0], nocc[0]),
                    x[:, n0:].reshape(-1, nv[1], nocc[1]))

        def full(xa, xb, with_oo):
            out = []
            for s, x in ((0, xa), (1, xb)):
                m = cp.zeros((len(x),) + mo1[s].shape[1:])
                m[:, ~occ[s]] = x
                if with_oo:
                    m[:, occ[s]] = mo1[s][:, occ[s]]
                out.append(m)
            return out

        # (e_a - e_i) x + v[x] = -(hs_vo + v[mo1_oo])
        v_oo = vind(full(cp.zeros((3, nv[0], nocc[0])),
                         cp.zeros((3, nv[1], nocc[1])), True))
        b = pack(*[(-hs[s][:, ~occ[s]] - v_oo[s]) / eai[s] for s in (0, 1)])

        def aop(x):
            xa, xb = unpack(x.reshape(-1, b.shape[1]))
            va, vb = vind(full(xa, xb, False))
            return pack(va / eai[0], vb / eai[1])
        x = krylov_refined(aop, b, self.conv_tol, self.max_cycle_cphf)
        xa, xb = unpack(x)
        return full(xa, xb, True)

    def dia(self, dm0=None):
        mol = self.mol
        if dm0 is None:
            dm0 = self._scf.make_rdm1()
        dm0 = cp.asnumpy(dm0)
        dm0 = dm0[0] + dm0[1]
        out = []
        for atm_id in self.shielding_nuc:
            with mol.with_rinv_origin(mol.atom_coord(atm_id)):
                h11 = mol.intor('int1e_giao_a11part', comp=9)
                e11 = np.einsum('xij,ij->x', h11, dm0).reshape(3, 3)
                e11 = e11 - np.eye(3) * e11.trace()
                h11 = mol.intor('int1e_a01gp', comp=9)
                e11 += np.einsum('xij,ij->x', h11, dm0).reshape(3, 3)
            out.append(e11)
        return np.array(out).reshape(-1, 3, 3)

    def para(self, mo10):
        mf = self._scf
        mol = self.mol
        dm10 = 0
        for s in (0, 1):
            C = cp.asarray(mf.mo_coeff[s])
            co = C[:, cp.asnumpy(mf.mo_occ[s]) > 0]
            dm10 = dm10 + C @ mo10[s] @ co.T
        out = []
        for atm_id in self.shielding_nuc:
            with mol.with_rinv_origin(mol.atom_coord(atm_id)):
                h01 = cp.asarray(mol.intor_asymmetric('int1e_prinvxp', 3))
            out.append(cp.asnumpy(cp.einsum('xji,yij->xy', dm10, h01) * 2))
        return np.asarray(out)

    def kernel(self):
        mf = self._scf
        if not (mf.istype('UHF') or mf.istype('UKS')):
            raise NotImplementedError('GPU UKS NMR: UHF/UKS references')
        log = logger.new_logger(self)

        def stage(label, fn):
            cp.cuda.Stream.null.synchronize()
            t0 = (logger.process_clock(), logger.perf_counter())
            out = fn()
            cp.cuda.Stream.null.synchronize()
            self.timings[label] = logger.perf_counter() - t0[1]
            log.timer('UKS NMR ' + label, *t0)
            return out
        unit = nist.ALPHA ** 2 * 1e6
        d = stage('dia', self.dia)
        h1 = stage('h1 (XC + GIAO 2e)', self.get_fock)
        self.mo10 = stage('CPHF', lambda: self.solve_mo1(h1))
        p = stage('para', lambda: self.para(self.mo10))
        return (d + p) * unit

    shielding = kernel
