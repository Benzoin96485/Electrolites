"""NMR spin-spin coupling (RHF/RKS) on the GPU
(`docs/FEATURE_TARGETS_PYSCF_2.13.md` O2).

pyscf-properties' ``prop.ssc.rhf.SpinSpinCoupling`` (non-relativistic,
Chem. Rev. 99, 293) for a closed-shell GPU4PySCF reference; GPU4PySCF has no
spin-spin coupling.  The four terms per nucleus pair (A, B):

* DSO: -tr D <r_A r_B / r_A^3 r_B^3> (minus its trace): all pairs from
  one device GEMM over the Becke grid (``dso='grid'``), or the 30-point
  radial quadrature of pyscf-properties (``dso='exact'``, host libcint,
  one ``int3c1e_iprinv`` per pair -- which dominates at 40+ atoms);
* PSO: the singlet, imaginary (``hermi=2``) CPHF for the 3 N_B right-hand
  sides h1 = -<p (1/r_B) x p>, solved together on the device with
  GPU4PySCF's response (only -hyb/2 K survives);
* FC: the triplet CPHF for the N_B right-hand sides
  8 pi/3 * 1/2 phi_a(R_B) phi_i(R_B) (AO values at the nuclei);
* FC+SD (``with_fcsd``): the triplet CPHF for the 9 N_B components of
  ``int1e_sa01sp``.

All right-hand sides of one kind go through one multi-RHS Krylov solve
(``gpu4pyscf.scf.cphf.solve``), the response from ``mf.gen_response`` (J/K
and the singlet/triplet fxc on the device).  The one-electron operators are
host libcint integrals, one per nucleus.

    ssc = electrolites.ssc_gpu.SSC(gpu_mf)
    e11 = ssc.kernel()          # (npair, 3, 3) reduced couplings, a.u.
    ssc.k_hz, ssc.j_hz          # isotropic K and J (Hz), natm x natm
"""
import numpy as np
import cupy as cp

from pyscf import gto, lib
from pyscf.lib import logger
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor

NUMINT_GRIDS = 30


def dso_integral(mol, orig1, orig2):
    """<r_1 r_2 / (|r - orig1|^3 |r - orig2|^3)> (3, 3, nao, nao) by the
    radial quadrature of pyscf-properties (JCP 73, 5718)."""
    t, w = np.polynomial.legendre.leggauss(NUMINT_GRIDS)
    a = (1 + t) / (1 - t) * .8
    w *= 2 / (1 - t) ** 2 * .8
    fakemol = gto.Mole()
    fakemol._atm = np.asarray([[0, 0, 0, 0, 0, 0]], dtype=np.int32)
    fakemol._bas = np.asarray([[0, 1, NUMINT_GRIDS, 1, 0, 3,
                                3 + NUMINT_GRIDS, 0]], dtype=np.int32)
    p_cart2sph_factor = 0.488602511902919921
    fakemol._env = np.hstack((orig2, a ** 2, a ** 2 * w * 4 / np.pi ** .5
                              / p_cart2sph_factor))
    fakemol._built = True
    pmol = mol + fakemol
    pmol.set_rinv_origin(orig1)
    mat1 = pmol.intor(mol._add_suffix('int3c1e_iprinv'), comp=3,
                      shls_slice=(0, mol.nbas, 0, mol.nbas,
                                  mol.nbas, pmol.nbas))
    mat = pmol.intor(mol._add_suffix('int3c1e_iprinv'), comp=3,
                     shls_slice=(mol.nbas, pmol.nbas, 0, mol.nbas,
                                 0, mol.nbas))
    mat += mat1.transpose(0, 3, 1, 2) + mat1.transpose(0, 3, 2, 1)
    return mat


def _uniq(lst):
    lst = sorted(set(lst))
    return lst, {a: k for k, a in enumerate(lst)}


def atom_gyro_list(mol):
    gyro = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        mass = mol.nucprop.get(symb, {}).get('mass', None)
        gyro.append(get_nuc_g_factor(symb, mass))
    return np.array(gyro)


class SSC(lib.StreamObject):
    """Spin-spin coupling of a GPU4PySCF RHF/RKS (pyscf-properties' options:
    ``nuc_pair``, ``with_fc``, ``with_fcsd``, ``cphf``, ``conv_tol``,
    ``max_cycle_cphf``)."""

    def __init__(self, mf):
        self._scf = mf
        self.mol = mf.mol
        self.verbose = mf.mol.verbose
        self.stdout = mf.mol.stdout
        mol = mf.mol
        self.nuc_pair = [(i, j) for i in range(mol.natm) for j in range(i)]
        self.with_fc = True
        self.with_fcsd = False
        self.cphf = True
        self.max_cycle_cphf = 20
        self.conv_tol = 1e-9
        self.dso = 'grid'
        self.max_dms = 48
        # relative CPHF residual the refinement stops at: 1e-6 is pyscf's
        # own level (~2e-7) at a third of the cost of 1e-9 (each refinement
        # is another Krylov solve over 3 N_atom right-hand sides)
        self.refine_tol = 1e-6
        self.e11 = self.k_hz = self.j_hz = None

    # ------------------------------------------------------------ helpers
    def _mo(self):
        mf = self._scf
        C = cp.asarray(mf.mo_coeff)
        occ = cp.asnumpy(mf.mo_occ)
        return (C[:, occ > 0], C[:, occ == 0], cp.asarray(mf.mo_energy),
                cp.asarray(occ))

    def _solve(self, h1, singlet):
        """Multi-RHS CPHF (e_a - e_i) x + v[x] = -h1, h1 (nset, nvir, nocc):
        the singlet imaginary (PSO) or the triplet real (FC/SD) response."""
        mf = self._scf
        co, cv, e, occ = self._mo()
        nvir, nocc = cv.shape[1], co.shape[1]
        if not self.cphf:
            e_ai = e[occ == 0][:, None] - e[occ > 0][None, :]
            return -h1 / e_ai
        if singlet:
            vresp = mf.gen_response(singlet=True, hermi=2)
        else:
            vresp = mf.gen_response(singlet=False, hermi=1)

        def fx(x):
            x = x.reshape(-1, nvir, nocc)
            out = cp.empty_like(x)
            # densities in groups (GPU4PySCF's K holds per-density
            # intermediates; 339 at once fail in cuTENSOR at 113 atoms)
            for z0 in range(0, len(x), self.max_dms):
                z1 = min(len(x), z0 + self.max_dms)
                dm = 2 * cv @ x[z0:z1] @ co.T
                dm = dm - dm.transpose(0, 2, 1) if singlet \
                    else dm + dm.transpose(0, 2, 1)
                v = cp.asarray(vresp(dm))
                out[z0:z1] = cv.T @ v @ co
            return out
        # pyscf ``cphf.solve_nos1``: (1 + D^-1 V) x = -D^-1 h1, with the
        # refined Krylov solve (GPU4PySCF's alone stalls near 1e-6)
        from electrolites.nmr_uks import krylov_refined
        e_ai = e[occ == 0][:, None] - e[occ > 0][None, :]
        nset = h1.shape[0]

        def aop(x):
            return (fx(x) / e_ai).reshape(len(x), -1)
        b = (-h1 / e_ai).reshape(nset, -1)
        return krylov_refined(aop, b, self.conv_tol, self.max_cycle_cphf,
                              rtol=self.refine_tol).reshape(h1.shape)

    def _vo(self, h1ao):
        co, cv = self._mo()[:2]
        return cv.T @ cp.asarray(h1ao) @ co

    # ------------------------------------------------------------- terms
    def make_dso(self, mol=None, dm0=None, nuc_pair=None):
        """DSO.  ``dso='grid'`` (default): -int rho w f_A f_B^T with
        f_A = r_A / r_A^3 on the SCF's (or a level-3) Becke grid, all pairs
        from one GEMM on the device; agrees with the 30-point radial
        quadrature of pyscf-properties to that quadrature's own ~1e-5
        relative error.  ``dso='exact'``: that quadrature (host libcint,
        one 3c1e integral per pair)."""
        mol = mol or self.mol
        nuc_pair = nuc_pair or self.nuc_pair
        if dm0 is None:
            dm0 = self._scf.make_rdm1()
        out = []
        if self.dso == 'exact':
            dm0 = cp.asnumpy(dm0)
            for ia, ja in nuc_pair:
                h11 = dso_integral(mol, mol.atom_coord(ia),
                                   mol.atom_coord(ja))
                a11 = -np.einsum('xyij,ji->xy', h11, dm0)
                out.append(a11 - a11.trace() * np.eye(3))
            return np.asarray(out) * nist.ALPHA ** 4
        from gpu4pyscf.dft import gen_grid, numint
        grids = getattr(self._scf, 'grids', None)
        if grids is None or grids.coords is None:
            grids = gen_grid.Grids(mol)
            grids.level = 3
            grids.build()
        ni = numint.NumInt()
        rho = numint.get_rho(ni, mol, cp.asarray(dm0), grids)
        coords = cp.asarray(grids.coords)
        rw = rho * cp.asarray(grids.weights)
        R = cp.asarray(mol.atom_coords())
        m = cp.zeros((mol.natm * 3, mol.natm * 3))
        blk = max(1, int(2e8 // (mol.natm * 3 * 8 * 2)))
        for p0 in range(0, coords.shape[0], blk):
            d = coords[p0:p0 + blk, None, :] - R[None]      # (g, A, 3)
            r = cp.linalg.norm(d, axis=2)
            f = (d / r[..., None] ** 3).reshape(d.shape[0], -1)
            m -= (f * rw[p0:p0 + blk, None]).T @ f
        m = cp.asnumpy(m).reshape(mol.natm, 3, mol.natm, 3)
        for ia, ja in nuc_pair:
            a11 = m[ia, :, ja]
            out.append(a11 - a11.trace() * np.eye(3))
        return np.asarray(out) * nist.ALPHA ** 4

    def h1_pso(self, atmlst):
        mol = self.mol
        h = []
        for ia in atmlst:
            with mol.with_rinv_origin(mol.atom_coord(ia)):
                h.append(-mol.intor_asymmetric('int1e_prinvxp', 3))
        return self._vo(np.concatenate(h))           # (3 natm, nvir, nocc)

    def make_pso(self, nuc_pair=None):
        nuc_pair = nuc_pair or self.nuc_pair
        l1, d1 = _uniq(i for i, j in nuc_pair)
        l2, d2 = _uniq(j for i, j in nuc_pair)
        co, cv = self._mo()[:2]
        nvir, nocc = cv.shape[1], co.shape[1]
        mo1 = self._solve(self.h1_pso(l2), True).reshape(-1, 3, nvir, nocc)
        h1 = self.h1_pso(l1).reshape(-1, 3, nvir, nocc)
        e = cp.einsum('axij,byij->abxy', h1, mo1) * 4
        e = cp.asnumpy(e)
        return np.asarray([e[d1[i], d2[j]] for i, j in nuc_pair]) \
            * nist.ALPHA ** 4

    def h1_fc(self, atmlst):
        mol = self.mol
        co, cv = self._mo()[:2]
        ao = cp.asarray(mol.eval_gto('GTOval', mol.atom_coords()[atmlst]))
        fac = 8 * np.pi / 3 * .5
        return fac * cp.einsum('pa,pi->pai', ao @ cv, ao @ co)

    def make_fc(self, nuc_pair=None):
        nuc_pair = nuc_pair or self.nuc_pair
        l1, d1 = _uniq(i for i, j in nuc_pair)
        l2, d2 = _uniq(j for i, j in nuc_pair)
        mo1 = self._solve(self.h1_fc(l2), False)
        h1 = self.h1_fc(l1)
        e = cp.asnumpy(cp.einsum('aij,bij->ab', h1, mo1) * 4)
        para = np.asarray([e[d1[i], d2[j]] for i, j in nuc_pair])
        return np.einsum(',k,xy->kxy', nist.ALPHA ** 4, para, np.eye(3))

    def h1_fcsd(self, atmlst):
        mol = self.mol
        nao = mol.nao
        h = []
        for ia in atmlst:
            with mol.with_rinv_origin(mol.atom_coord(ia)):
                a01p = mol.intor('int1e_sa01sp', 12).reshape(3, 4, nao, nao)
            h.append(-(a01p[:, :3] + a01p[:, :3].transpose(0, 1, 3, 2))
                     .reshape(9, nao, nao) * .5)
        return self._vo(np.concatenate(h))           # (9 natm, nvir, nocc)

    def make_fcsd(self, nuc_pair=None):
        nuc_pair = nuc_pair or self.nuc_pair
        l1, d1 = _uniq(i for i, j in nuc_pair)
        l2, d2 = _uniq(j for i, j in nuc_pair)
        co, cv = self._mo()[:2]
        nvir, nocc = cv.shape[1], co.shape[1]
        mo1 = self._solve(self.h1_fcsd(l2), False).reshape(
            -1, 3, 3, nvir, nocc)
        h1 = self.h1_fcsd(l1).reshape(-1, 3, 3, nvir, nocc)
        e = cp.asnumpy(cp.einsum('axwij,bywij->abxy', h1, mo1) * 4)
        return np.asarray([e[d1[i], d2[j]] for i, j in nuc_pair]) \
            * nist.ALPHA ** 4

    # ------------------------------------------------------------ driver
    def kernel(self):
        mf = self._scf
        if mf.mol.spin != 0 or not (mf.istype('RHF') or
                                    mf.istype('RKS')):
            raise NotImplementedError('GPU SSC: closed-shell RHF/RKS only')
        if hasattr(mf, 'xc'):
            omega = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=0)[0]
            if omega != 0:
                raise NotImplementedError('GPU SSC: range-separated hybrids')
        log = logger.new_logger(self)
        self.timings = {}

        def stage(label, fn):
            cp.cuda.Stream.null.synchronize()
            t0 = (logger.process_clock(), logger.perf_counter())
            out = fn()
            cp.cuda.Stream.null.synchronize()
            self.timings[label] = logger.perf_counter() - t0[1]
            log.timer('SSC ' + label, *t0)
            return out
        e11 = stage('DSO', self.make_dso)
        e11 = e11 + stage('PSO', self.make_pso)
        if self.with_fcsd:
            e11 = e11 + stage('FC+SD', self.make_fcsd)
        elif self.with_fc:
            e11 = e11 + stage('FC', self.make_fc)
        self.e11 = e11
        nuc_magneton = .5 * (nist.E_MASS / nist.PROTON_MASS)
        unit = nist.HARTREE2J / nist.PLANCK * nuc_magneton ** 2
        natm = self.mol.natm
        k = np.zeros((natm, natm))
        for n, (i, j) in enumerate(self.nuc_pair):
            k[i, j] = k[j, i] = unit * np.trace(e11[n]) / 3
        gyro = atom_gyro_list(self.mol)
        self.k_hz = k
        self.j_hz = np.einsum('ij,i,j->ij', k, gyro, gyro)
        return e11

    run = lambda self: (self.kernel(), self)[1]     # noqa: E731
