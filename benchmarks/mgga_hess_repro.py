"""Which stage of a meta-GGA Hessian is not reproducible (W23).

One converged SCF, then every stage of a density-fitted (or direct) RKS
Hessian evaluated ``--reps`` times on it, in one process, each repeat compared
element-wise with the first.  A stage that differs between repeats on the same
inputs is a run-to-run noise source; one that agrees is not, and the noise is
upstream of it.  ``--save``/``--load`` share the SCF between processes so the
SCF's own run-to-run spread can be separated from the Hessian's.

    python benchmarks/mgga_hess_repro.py --xyz benchmarks/molecules/glycine.xyz \\
        --basis def2-svp --xc tpss --reps 3 [--patch all] [--direct]
"""
import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--direct', action='store_true', help='no density fitting')
ap.add_argument('--save', default='', help='write mo_coeff/energy/occ (.npz)')
ap.add_argument('--load', default='', help='read them instead of iterating')
ap.add_argument('--stages', default='all')
ap.add_argument('--conv-tol-cpscf', type=float, default=1e-6)
a = ap.parse_args()
apply_patch(a.patch)

from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.hessian import rks as rks_hess                # noqa: E402
from gpu4pyscf.hessian import rhf as rhf_hess                # noqa: E402

mol = build_mol(a)
mf = dft.RKS(mol, xc=a.xc)
if not a.direct:
    mf = mf.density_fit()
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.conv_tol_cpscf = a.conv_tol_cpscf
mf.verbose = 0
if a.load:
    z = np.load(a.load)
    mf.mo_coeff = cp.asarray(z['mo_coeff'])
    mf.mo_energy = cp.asarray(z['mo_energy'])
    mf.mo_occ = cp.asarray(z['mo_occ'])
    mf.e_tot = float(z['e_tot'])
    mf.converged = True
    mf.grids.build()
else:
    mf.kernel()
dm = mf.make_rdm1()
print(f'# E = {mf.e_tot:.12f}  |dm| = {float(cp.linalg.norm(dm)):.15f}  '
      f'sum(mo_energy) = {float(mf.mo_energy.sum()):.12f}', flush=True)
if a.save:
    np.savez(a.save, mo_coeff=cp.asnumpy(mf.mo_coeff),
             mo_energy=cp.asnumpy(mf.mo_energy),
             mo_occ=cp.asnumpy(mf.mo_occ), e_tot=mf.e_tot)

hess = mf.Hessian()
if not a.direct:
    hess.auxbasis_response = 2
mo_e, mo_c, mo_o = mf.mo_energy, mf.mo_coeff, mf.mo_occ
mocc = mo_c[:, mo_o > 0]
dm0 = mocc.dot(mocc.T) * 2
natm = mol.natm


def _ret(x):
    if isinstance(x, (tuple, list)):
        return [cp.asarray(y) if y is not None else None for y in x]
    return cp.asarray(x)


def stage(name, fn):
    if a.stages != 'all' and name not in a.stages.split(','):
        return None
    ref = _ret(fn())
    refs = ref if isinstance(ref, list) else [ref]
    worst = []
    for r in range(1, a.reps):
        out = _ret(fn())
        outs = out if isinstance(out, list) else [out]
        errs = []
        for x, y in zip(refs, outs):
            if x is None:
                continue
            scale = float(abs(x).max()) or 1.0
            errs.append(float(abs(x - y).max()) / scale)
        worst.append(max(errs))
    tag = 'REPRODUCIBLE' if max(worst, default=0) < 1e-12 else 'DIFFERS'
    print(f'@@ {name:28s} rel. max diff over {a.reps - 1} repeats: '
          + ' '.join(f'{w:.2e}' for w in worst) + f'   {tag}', flush=True)
    return ref


def _intopt():
    if mf.with_df.intopt is None:
        mf.with_df.build(build_cderi=False)
    return mf.with_df.intopt


mem = 4000
stage('cache_xc_kernel', lambda: mf._numint.cache_xc_kernel(
    mol, mf.grids, mf.xc, mo_c, mo_o, spin=0)[1:])
stage('_get_vxc_diag', lambda: rks_hess._get_vxc_diag(hess, mo_c, mo_o, mem))
stage('_get_vxc_deriv2', lambda: rks_hess._get_vxc_deriv2(hess, mo_c, mo_o, mem))
stage('_get_exc_deriv2', lambda: rks_hess._get_exc_deriv2(
    hess, mo_c, mo_o, dm0, mem))
stage('_get_vxc_deriv1', lambda: rks_hess._get_vxc_deriv1(hess, mo_c, mo_o, mem))
if not a.direct:
    from gpu4pyscf.df.hessian import rhf as df_rhf_hess
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=0)
    stage('ip2 _jk_energy_per_atom', lambda: df_rhf_hess._jk_energy_per_atom(
        _intopt(), dm0, 1., hyb))
    stage('h1 _get_veff', lambda: df_rhf_hess._get_veff(
        _intopt(), mo_c, mo_o, 1., hyb))
stage('partial_hess_elec', lambda: hess.partial_hess_elec(mo_e, mo_c, mo_o))
h1mo = stage('make_h1', lambda: hess.make_h1(mo_c, mo_o))
if h1mo is None:
    h1mo = hess.make_h1(mo_c, mo_o)
fx = hess.gen_vind(mo_c, mo_o)
nocc = int((mo_o > 0).sum())
rng = np.random.default_rng(7)
mo1_probe = cp.asarray(rng.standard_normal((6, mo_c.shape[1], nocc)) * 1e-2)
stage('fx (gen_vind), 6 RHS', lambda: fx(mo1_probe))
ni = mf._numint
_, _, fxc0 = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mo_c, mo_o, spin=0)
mo1_mo = cp.einsum('pa,nai->npi', mo_c, mo1_probe)
dms = cp.einsum('npi,qi->npq', mo1_mo, mocc * 2)
dms = cp.ascontiguousarray(dms + dms.transpose(0, 2, 1))
import gpu4pyscf.lib.cupy_helper as _ch                      # noqa: E402
dms_t = _ch.tag_array(dms, mo1=mo1_mo, occ_coeff=mocc)
stage('nr_rks_fxc (AO basis)', lambda: ni.nr_rks_fxc(
    mol, mf.grids, mf.xc, dm0, dms, 0, 1, None, None, fxc0))
stage('nr_rks_fxc_mo', lambda: rks_hess.nr_rks_fxc_mo(
    ni, mol, mf.grids, mf.xc, None, dms_t, mo_c, 0, 1, fxc=fxc0))
stage('solve_mo1', lambda: hess.solve_mo1(mo_e, mo_c, mo_o, h1mo, fx))
stage('hess_elec', lambda: hess.hess_elec(mo_e, mo_c, mo_o))
