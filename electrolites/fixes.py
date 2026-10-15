"""Small GPU4PySCF 1.8.1 defects found by `docs/FEATURE_TARGETS_PYSCF_2.13.md`
§3.3, each a crash on a legal input or a wrong error message.  None is a
performance change; each is checked in `tests/test_fixes.py`.

* **D1 vSAP guess.**  ``init_guess='vsap'`` cannot run: RKS has no
  ``get_vsap``, UKS binds PySCF's under the misspelled name ``get_vasp``
  (``dft/uks.py:107``), and PySCF's ``init_guess_by_vsap`` mixes a host
  kinetic matrix with the device overlap.  Replaced by a device
  ``get_vsap`` -- the same potential, -sum_A Z_eff,A(r)/r_A from PySCF's
  ``sap_Zeff`` table, contracted over the KS grid in one pass -- and a device
  ``init_guess_by_vsap`` for RKS, UKS and ROKS.  Equal to PySCF's matrix to
  1e-14 on the same grid.
* **D2 ROHF Newton.**  ``ROHF.newton().run()`` raises a CuPy implicit
  conversion: the ROHF initial guess comes back from PySCF's CPU function as
  a NumPy array, which the plain SCF casts and the second-order solver does
  not.  ``ROHF.get_init_guess`` now returns a device array.
* **D3 CCSD.to_cpu().**  Called ``.to_cpu()`` on its already-CPU ``_scf``
  and would import a ``pyscf.cc.ccsd_incore`` that does not exist.  Now
  returns a ``pyscf.cc.ccsd.CCSD`` with the settings and amplitudes copied.
* **D4 CCSD on an open-shell reference** died with a bare ``IndexError``
  inside PySCF's MP2 guess; it now says that the GPU CCSD is closed-shell
  only and what to use instead.
* **D7 DF-MP2 with frozen orbitals and ``with_t2``** asserted: t2 was
  allocated with the full ``nocc``/``nmo`` (restricted) or ``nao - nocc``
  virtuals (unrestricted) while the driver fills the active block.  t2 is
  now allocated from the active occupied/virtual split.
* **D10 periodic singlet/triplet response.**  The Gamma-point RKS
  ``gen_response(singlet=True)`` passed ``singlet`` into ``nr_rks_fxc``'s
  ``fxc`` slot (an error), ``singlet=False`` dropped the triplet fxc
  altogether, and both took a single density.  They are now
  ``electrolites.pbc_tdscf.gen_response`` (fxc_{s|t} + J - hyb/2 K, any
  number of densities; PySCF's response to 1e-10).  The k-point KRKS
  version has the same argument-order bug and now raises for
  ``singlet`` not None (X2 is not built); ``singlet=None`` (SOSCF) is
  upstream's in both.
* **V3 Gaussian MM charges in QM/MM.**  The one-electron MM integrals
  already take the Gaussian exponents, but ``energy_nuc`` and the gradient's
  ``grad_nuc`` assert point charges, and ``grad_nuc_mm`` silently used the
  point-charge kernel for Gaussian ones.  All three now use PySCF's
  erf-screened nucleus-MM interaction (``qmmm/itrf.py``); point charges are
  unchanged.
"""
import numpy as np
import cupy

from gpu4pyscf.dft import rks, uks, roks
from gpu4pyscf.scf import rohf
from gpu4pyscf.cc import ccsd_incore
from gpu4pyscf.mp import dfmp2, dfump2, dfmp2_addons

# ----------------------------------------------------------------- D1 vSAP


def get_vsap(ks, mol=None):
    """Superposition of atomic potentials on the KS grid, on the device.

    PySCF's ``NumInt.nr_sap_vxc`` term by term: V(r) = -sum_A Z_eff,A(r_A)/r_A
    with Z_eff linearly interpolated in ``pyscf.dft.sap.sap_Zeff`` (zero past
    its range, as PySCF), then V_mn = sum_g w_g V(r_g) phi_m phi_n.
    """
    from pyscf.dft.sap import sap_Zeff
    from gpu4pyscf.dft import numint
    if mol is None:
        mol = ks.mol
    assert not mol.has_ecp(), 'ECP or PP not supported'
    grids = ks.grids
    if grids.coords is None:
        grids.build()
    coords = cupy.asarray(grids.coords)
    weights = cupy.asarray(grids.weights)
    table = cupy.asarray(sap_Zeff)
    rmax = sap_Zeff.shape[0]               # PySCF's cutoff, kept as is
    atom_coords = cupy.asarray(mol.atom_coords())
    charges = mol.atom_charges()
    nao = mol.nao
    vmat = cupy.zeros((nao, nao))
    blk = max(1024, int(2**27 // max(nao, 1)))
    for p0 in range(0, coords.shape[0], blk):
        c = coords[p0:p0 + blk]
        v = cupy.zeros(c.shape[0])
        for ia, z in enumerate(charges):
            if z < 1:
                continue
            r = cupy.linalg.norm(c - atom_coords[ia], axis=1)
            zeff = cupy.interp(r, table[0], table[z])
            zeff[r >= rmax] = 0.
            v -= zeff / r
        ao = numint.eval_ao(mol, c, deriv=0)          # (ngrid, nao)
        vmat += ao.T @ (ao * (weights[p0:p0 + blk] * v)[:, None])
    return vmat


def _get_vsap_uks(ks, mol=None):
    v = get_vsap(ks, mol)
    return cupy.stack([v, v])


def geneig(h, s):
    """h c = s c e by canonical orthogonalisation and a standard eigh.

    GPU4PySCF's ``eig`` without an orthogonaliser calls cuSOLVER's
    generalised solver, which failed ('failed in eigh kernel') on a compute
    node in the job environment while the same call passed on the login
    node -- the cause was not isolated.  The SCF itself never takes that
    path (it passes ``x``), so neither does this.  ``h`` may be stacked
    (2, n, n)."""
    w, u = cupy.linalg.eigh((s + s.T) * .5)
    x = u / cupy.sqrt(w)
    if h.ndim == 3:
        out = [geneig(hs, s) for hs in h]
        return cupy.stack([o[0] for o in out]), cupy.stack([o[1] for o in out])
    hx = x.T @ ((h + h.T) * .5) @ x
    e, c = cupy.linalg.eigh(hx)
    return e, x @ c


def init_guess_by_vsap(mf, mol=None):
    """The vSAP guess density, on the device (PySCF's recipe)."""
    if mol is None:
        mol = mf.mol
    vsap = mf.get_vsap(mol)
    t = cupy.asarray(mol.intor_symmetric('int1e_kin'))
    s = mf.get_ovlp(mol)
    mo_energy, mo_coeff = geneig(t + vsap, s)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    return mf.make_rdm1(mo_coeff, mo_occ)


rks.RKS.get_vsap = get_vsap
roks.ROKS.get_vsap = get_vsap
uks.UKS.get_vsap = _get_vsap_uks
for _cls in (rks.RKS, uks.UKS, roks.ROKS):
    _cls.init_guess_by_vsap = init_guess_by_vsap

# ------------------------------------------------- D2 ROHF initial guess
_rohf_init_guess_orig = rohf.ROHF.get_init_guess


def _rohf_get_init_guess(self, *args, **kwargs):
    dm = _rohf_init_guess_orig(self, *args, **kwargs)
    if not isinstance(dm, cupy.ndarray):
        dm = cupy.asarray(dm)
    return dm


_rohf_get_init_guess.__wrapped__ = _rohf_init_guess_orig
rohf.ROHF.get_init_guess = _rohf_get_init_guess

# --------------------------------------------------- D3 / D4 GPU CCSD
_CC_COPY = ('max_cycle', 'conv_tol', 'conv_tol_normt', 'iterative_damping',
            'diis', 'diis_space', 'diis_start_cycle', 'diis_start_energy_diff',
            'direct', 'incore_complete', 'level_shift', 'cc2', 'verbose',
            'max_memory', 'stdout', 'converged', 'e_hf', 'e_corr', 'emp2')


def _ccsd_to_cpu(self):
    from pyscf.cc import ccsd
    mf = self._scf
    if getattr(mf, 'device', 'cpu') == 'gpu':
        mf = mf.to_cpu()
    mo = self.mo_coeff
    occ = self.mo_occ
    obj = ccsd.CCSD(mf, frozen=self.frozen,
                    mo_coeff=None if mo is None else cupy.asnumpy(mo),
                    mo_occ=None if occ is None else cupy.asnumpy(occ))
    for k in _CC_COPY:
        if k in self.__dict__:
            setattr(obj, k, self.__dict__[k])
    for k in ('t1', 't2'):
        v = getattr(self, k, None)
        setattr(obj, k, None if v is None else cupy.asnumpy(v))
    return obj


_ccsd_init_orig = ccsd_incore.CCSD.__init__


def _ccsd_init(self, mf, *args, **kwargs):
    if (mf.istype('UHF') or mf.istype('GHF') or mf.istype('ROHF')
            or not mf.istype('RHF')):
        raise NotImplementedError(
            f'GPU4PySCF CCSD is closed-shell (RHF) only; got '
            f'{type(mf).__name__}.  Use pyscf.cc.UCCSD(mf.to_cpu()) or '
            'pyscf.cc.GCCSD(mf.to_cpu()) on the CPU')
    _ccsd_init_orig(self, mf, *args, **kwargs)


_ccsd_init.__wrapped__ = _ccsd_init_orig
ccsd_incore.CCSDBase.to_cpu = _ccsd_to_cpu
ccsd_incore.CCSD.__init__ = _ccsd_init

# ------------------------------------------ D7 DF-MP2 frozen + with_t2


def _dtype(mp, kwargs):
    fp = kwargs.get('fp_type')
    if fp is None:
        fp = mp.fp_type
    return np.float32 if fp in (np.float32, 'FP32') else np.float64


def _wrap_restricted(orig):
    def kernel(mp, *args, **kwargs):
        with_t2 = kwargs.get('with_t2')
        if with_t2 is None:
            with_t2 = mp.with_t2
        if with_t2 is True and mp.t2 is None:
            _, occ, vir, _ = dfmp2_addons.split_mo_coeff_restricted(
                mp, mo_coeff=kwargs.get('mo_coeff'),
                frozen_mask=kwargs.get('frozen_mask'),
                mo_occ=kwargs.get('mo_occ'))
            no, nv = occ.shape[1], vir.shape[1]
            mp.t2 = cupy.empty((no, no, nv, nv), dtype=_dtype(mp, kwargs))
        return orig(mp, *args, **kwargs)
    kernel.__wrapped__ = orig
    return kernel


def _wrap_unrestricted(orig):
    def kernel(mp, *args, **kwargs):
        with_t2 = kwargs.get('with_t2')
        if with_t2 is None:
            with_t2 = mp.with_t2
        if with_t2 is True and mp.t2 is None:
            _, occ, vir, _ = dfmp2_addons.split_mo_coeff_unrestricted(
                mp, mo_coeff=kwargs.get('mo_coeff'),
                frozen_mask=kwargs.get('frozen_mask'),
                mo_occ=kwargs.get('mo_occ'))
            no = [o.shape[1] for o in occ]
            nv = [v.shape[1] for v in vir]
            dt = _dtype(mp, kwargs)
            mp.t2 = [cupy.empty((no[0], no[0], nv[0], nv[0]), dtype=dt),
                     cupy.empty((no[0], no[1], nv[0], nv[1]), dtype=dt),
                     cupy.empty((no[1], no[1], nv[1], nv[1]), dtype=dt)]
        return orig(mp, *args, **kwargs)
    kernel.__wrapped__ = orig
    return kernel


dfmp2.DFMP2._kernel_impl = _wrap_restricted(dfmp2.DFMP2._kernel_impl)
dfump2.DFUMP2._kernel_impl = _wrap_unrestricted(dfump2.DFUMP2._kernel_impl)


# ------------------------------------- D10 periodic singlet/triplet response
from gpu4pyscf.pbc.dft import rks as pbc_rks, krks as pbc_krks   # noqa: E402


def _wrap_pbc_response(orig):
    def gen_response(mf, mo_coeff=None, mo_occ=None, singlet=None, hermi=0,
                     max_memory=None, with_nlc=False):
        if singlet is None or mo_coeff is not None or mo_occ is not None:
            return orig(mf, mo_coeff, mo_occ, singlet, hermi, max_memory,
                        with_nlc)
        from electrolites.pbc_tdscf import gen_response as resp
        return resp(mf, singlet=singlet, hermi=hermi)
    gen_response.__wrapped__ = orig
    return gen_response


def _wrap_kpbc_response(orig):
    def gen_response(mf, mo_coeff=None, mo_occ=None, singlet=None, hermi=0,
                     max_memory=None, with_nlc=False):
        if singlet is not None:
            raise NotImplementedError(
                'GPU KRKS singlet/triplet response: GPU4PySCF 1.8.1 passes '
                'singlet into nr_rks_fxc\'s fxc slot (D10); the k-point '
                'TDDFT (X2) is not built')
        return orig(mf, mo_coeff, mo_occ, singlet, hermi, max_memory,
                    with_nlc)
    gen_response.__wrapped__ = orig
    return gen_response


if not hasattr(pbc_rks.RKS.gen_response, '__wrapped__'):
    pbc_rks.RKS.gen_response = _wrap_pbc_response(pbc_rks.RKS.gen_response)
if not hasattr(pbc_krks.KRKS.gen_response, '__wrapped__'):
    pbc_krks.KRKS.gen_response = _wrap_kpbc_response(
        pbc_krks.KRKS.gen_response)


# ---------------------------------------------- V3 Gaussian MM charges
from gpu4pyscf.qmmm import itrf as _qmmm                     # noqa: E402


def _nuc_mm_kernel(mol, mm_mol):
    """(q_A Z_I, R_A - R_I, f(r), g(r)): E = sum q Z f(r), dE/dR_A =
    sum q Z g(r) (R_A - R_I); f = 1/r or erf(r/a)/r, g = -1/r^3 or
    -(erf(r/a)/r - 2/(sqrt(pi) a) exp(-r^2/a^2)) / r^2 (PySCF qmmm)."""
    from scipy.special import erf
    coords = mm_mol.atom_coords()
    charges = mm_mol.atom_charges()
    z = mol.atom_charges()
    d = mol.atom_coords()[:, None, :] - coords[None, :, :]       # (A, I, 3)
    r = np.linalg.norm(d, axis=2)
    qz = z[:, None] * charges[None, :]
    if mm_mol.charge_model == 'gaussian':
        expnts = mm_mol.get_zetas()
        radii = 1 / np.sqrt(expnts)
        f = erf(r / radii) / r
        g = -(f - 2 / (np.sqrt(np.pi) * radii) * np.exp(-expnts * r ** 2)) \
            / r ** 2
    else:
        f = 1 / r
        g = -1 / r ** 3
    return qz, d, f, g


def _energy_nuc(self):
    nuc = super(_qmmm.QMMMSCF, self).energy_nuc()
    qz, d, f, g = _nuc_mm_kernel(self.mol, self.mm_mol)
    return nuc + float((qz * f).sum())


def _grad_nuc(self, mol=None, atmlst=None):
    if mol is None:
        mol = self.mol
    g_qm = super(_qmmm.QMMMGrad, self).grad_nuc(mol, atmlst)
    qz, d, f, g = _nuc_mm_kernel(mol, self.base.mm_mol)
    g_mm = np.einsum('AI,AIx->Ax', qz * g, d)
    if atmlst is not None:
        g_mm = g_mm[atmlst]
    return g_qm + g_mm


def _grad_nuc_mm(self, mol=None):
    if mol is None:
        mol = self.mol
    qz, d, f, g = _nuc_mm_kernel(mol, self.base.mm_mol)
    return -np.einsum('AI,AIx->Ix', qz * g, d)


_qmmm.QMMMSCF.energy_nuc = _energy_nuc
_qmmm.QMMMGrad.grad_nuc = _grad_nuc
_qmmm.QMMMGrad.grad_nuc_mm = _grad_nuc_mm
