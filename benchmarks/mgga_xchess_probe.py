"""Where a meta-GGA XC Hessian spends its time (W28), before anything is written.

`fastxchess` covers GGAs only, so a meta-GGA Hessian runs GPU4PySCF's
`hessian/rks.py:_get_vxc_deriv1` and `_get_exc_deriv2` (= `_get_vxc_diag` +
`_get_vxc_deriv2`).  The entry's falsification criterion is that the time is
in the per-block `eval_xc_eff` -- a meta-GGA's third functional derivative
costs more than a GGA's -- rather than in the atom loop, in which case the
answer is catalogue G1's grouping and not B5's screen.  So this measures:

* the three upstream functions whole, on one converged SCF;
* each of `_get_vxc_deriv1`'s and `_get_vxc_deriv2`'s MGGA branches stage by
  stage (a copy of the branch with a synchronisation point per stage):
  the AO evaluation (`block_loop` alone), the expansion to the full `nao`,
  `eval_rho2`, `eval_xc_eff` (deriv=2), the atom-independent `v_ip`/`ipip`
  terms, and the atom loop;
* `eval_xc_eff` on the whole grid in one call against the same points one
  block at a time.

    python benchmarks/mgga_xchess_probe.py --xyz benchmarks/molecules/penicillinV.xyz \
        --basis def2-svp --xc tpss --grids 3
"""
import json
import time
from collections import OrderedDict

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--skip-stages', action='store_true')
ap.add_argument('--fast-stages', action='store_true',
                help="time fastxchess's own stages instead (each helper "
                     'wrapped with a synchronisation point) and exit')
a = ap.parse_args()
apply_patch(a.patch)

from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.dft import numint                             # noqa: E402
from gpu4pyscf.hessian import rks as hrks                    # noqa: E402
from gpu4pyscf.grad import rks as rks_grad                   # noqa: E402
from gpu4pyscf.lib.cupy_helper import contract, take_last2d  # noqa: E402

_sync = cp.cuda.Stream.null.synchronize
T = OrderedDict()


class Stage:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        _sync()
        self.t = time.perf_counter()

    def __exit__(self, *exc):
        _sync()
        T[self.name] = T.get(self.name, 0.0) + time.perf_counter() - self.t


def timed(fn):
    _sync()
    t = time.perf_counter()
    out = fn()
    _sync()
    return time.perf_counter() - t, out


mol = build_mol(a)
mf = dft.RKS(mol, xc=a.xc).density_fit()
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
mf.verbose = 0
e = mf.kernel()
print(f'# {a.xyz.rsplit("/",1)[-1]}/{a.basis}/{a.xc}/L{a.grids}  '
      f'natm={mol.natm} nao={mol.nao} E={e:.9f}', flush=True)
h = mf.Hessian()
ni = mf._numint
xctype = ni._xc_type(mf.xc)
assert xctype == 'MGGA', xctype
fns = [('_get_vxc_deriv1', getattr(hrks, '_get_vxc_deriv1_orig',
                                   hrks._get_vxc_deriv1)),
       ('_get_vxc_diag', hrks._get_vxc_diag),
       ('_get_vxc_deriv2', getattr(hrks, '_get_vxc_deriv2_orig',
                                   hrks._get_vxc_deriv2))]
whole = {}
for name, fn in fns:
    fn(h, mf.mo_coeff, mf.mo_occ, 4000)          # warm up
    whole[name] = min(timed(lambda: fn(h, mf.mo_coeff, mf.mo_occ, 4000))[0]
                      for _ in range(max(1, a.reps - 1)))
    print(f'# upstream {name:18s} {whole[name]:8.3f} s', flush=True)

if a.fast_stages:
    from electrolites import fastxchess as fx
    import cupyx

    def wrap(mod, name, label):
        fn = getattr(mod, name)

        def w(*args, **kw):
            with Stage(label):
                return fn(*args, **kw)
        setattr(mod, name, w)
        return fn
    saved = [(fx, n, wrap(fx, n, n)) for n in
             ('_rho', '_dR_rho1', '_wv2', '_deriv1_gemms', '_ipip_block',
              '_deriv2_t')]
    saved.append((ni, 'eval_xc_eff', wrap(ni, 'eval_xc_eff', 'eval_xc_eff')))
    saved.append((cupyx, 'scatter_add', wrap(cupyx, 'scatter_add',
                                             'scatter_add')))
    saved.append((fx, 'add_sparse', wrap(fx, 'add_sparse', 'add_sparse')))
    for name in ('_get_vxc_deriv1', '_get_vxc_deriv2'):
        fn = getattr(hrks, name)
        T.clear()
        tot = timed(lambda: fn(h, mf.mo_coeff, mf.mo_occ, 4000))[0]
        print(f'\nfastxchess {name}\n{"stage":24s} {"time/s":>9s} '
              f'{"share":>7s}\n' + '-' * 44)
        print(f'{"TOTAL":24s} {tot:9.3f} {100.0:6.1f}%')
        for k, v in T.items():
            print(f'{k:24s} {v:9.3f} {100*v/tot:6.1f}%')
        acc = sum(T.values())
        print(f'{"accounted":24s} {acc:9.3f} {100*acc/tot:6.1f}%',
              flush=True)
    raise SystemExit

opt = ni.gdftopt
_sorted_mol = opt._sorted_mol
grids = mf.grids
nao, nmo = mf.mo_coeff.shape
mo_coeff = cp.asarray(mf.mo_coeff)
mo_occ = cp.asarray(mf.mo_occ)
mocc = mo_coeff[:, mo_occ > 0]
nocc = mocc.shape[1]
natm = mol.natm
coeff = cp.asarray(opt.coeff)
dm0 = mf.make_rdm1(mo_coeff, mo_occ)
dm0_sorted = opt.sort_orbitals(dm0, axis=[0, 1])
aoslices = mol.aoslice_by_atom()
XX, XY, XZ, YY, YZ, ZZ = 4, 5, 6, 7, 8, 9
YX, ZX, ZY = XY, XZ, YZ

# -------------------------------------------------------------- libxc -------
rho_all, blks = [], []
for ao_mask, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, 2,
                                              None):
    ao1 = contract('nip,ij->njp', ao_mask, coeff[mask])
    rho_all.append(numint.eval_rho2(_sorted_mol, ao1[:10], mo_coeff, mo_occ,
                                    mask, xctype))
    blks.append(len(weight))
rho = cp.hstack(rho_all)
rho_all = ao1 = None
t_one = timed(lambda: ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype,
                                     spin=0))[0]


def per_block():
    p0 = 0
    for n in blks:
        ni.eval_xc_eff(mf.xc, rho[:, p0:p0+n], 2, xctype=xctype, spin=0)
        p0 += n


t_blk = timed(per_block)[0]
print(f'# libxc deriv=2 MGGA: one call {t_one*1e3:8.1f} ms   {len(blks)} '
      f'calls {t_blk*1e3:8.1f} ms', flush=True)
rho = None
cp.get_default_memory_pool().free_all_blocks()

t_ao = timed(lambda: [None for _ in ni.block_loop(_sorted_mol, grids, nao, 2,
                                                  None)])[0]
print(f'# block_loop alone (eval_ao deriv=2): {t_ao:.3f} s', flush=True)
if a.skip_stages:
    raise SystemExit


def report(title, total):
    print(f'\n{title}\n{"stage":24s} {"time/s":>9s} {"share":>7s}')
    print('-' * 44)
    print(f'{"TOTAL (re-run here)":24s} {total:9.3f} {100.0:6.1f}%')
    print(f'{"eval_ao (block_loop)":24s} {t_ao:9.3f} {100*t_ao/total:6.1f}%')
    for k, v in T.items():
        print(f'{k:24s} {v:9.3f} {100*v/total:6.1f}%')
    acc = sum(T.values()) + t_ao
    print(f'{"accounted":24s} {acc:9.3f} {100*acc/total:6.1f}%', flush=True)


# ------------------------------------------- deriv1, MGGA branch, staged ----
def deriv1():
    v_ip = cp.zeros((3, nao, nao))
    vmat = cp.zeros((natm, 3, nao, nocc))
    for ao, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, 2,
                                             None):
        ng = len(weight)
        with Stage('expand ao to nao'):
            ao1 = contract('nip,ij->njp', ao, coeff[mask])
        with Stage('eval_rho2'):
            rho = numint.eval_rho2(_sorted_mol, ao1[:10], mo_coeff, mo_occ,
                                   mask, xctype)
        with Stage('eval_xc_eff'):
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype,
                                      spin=0)[1:3]
        with Stage('v_ip (gga+tau)'):
            wv = cp.multiply(weight, vxc, out=vxc)
            wf = cp.multiply(weight, fxc, out=fxc)
            wv[0] *= .5
            wv[4] *= .5
            aow = cp.empty((3, nao, ng))
            v_ip = rks_grad._gga_grad_sum_(ao1, wv, accumulate=True, buf=aow,
                                           out=v_ip)
            v_ip = rks_grad._tau_grad_dot_(ao1, wv[4], accumulate=True,
                                           buf=aow[0], out=v_ip)
        with Stage('mo, ao_dm0'):
            mo = contract('xig,ip->xpg', ao1, mocc)
            ao_dm0 = contract('ik,pil->pkl', dm0, ao1[:4])
        mow = cp.empty((3, nocc, ng))
        for ia in range(natm):
            with Stage('atom: dR_rho1 + fxc'):
                dR_rho1 = hrks._make_dR_rho1(ao1, ao_dm0, ia, aoslices,
                                             xctype)
                wv2 = contract('xyg,sxg->syg', wf, dR_rho1)
                wv2[:, 0] *= .5
                wv2[:, 4] *= .25
            with Stage('atom: GGA GEMMs'):
                for i in range(3):
                    numint._scale_ao(ao1[:4], wv2[i, :4], out=aow[i])
                    numint._scale_ao(mo[:4], wv2[i, :4], out=mow[i])
                rks_grad._d1_dot_(aow, mo[0].T, beta=1.0, out=vmat[ia])
                rks_grad._d1_dot_(mow, ao1[0].T, beta=1.0, transpose=True,
                                  out=vmat[ia])
            with Stage('atom: tau GEMMs'):
                for j in range(1, 4):
                    for i in range(3):
                        numint._scale_ao(ao1[j], wv2[i, 4], out=aow[i])
                        numint._scale_ao(mo[j], wv2[i, 4], out=mow[i])
                    rks_grad._d1_dot_(aow, mo[j].T, beta=1.0, out=vmat[ia])
                    rks_grad._d1_dot_(mow, ao1[j].T, beta=1.0,
                                      transpose=True, out=vmat[ia])


t_tot = timed(deriv1)[0]
report('_get_vxc_deriv1, MGGA branch', t_tot)
d1 = dict(T)
T.clear()


# ------------------------------------------- deriv2, MGGA branch, staged ----
def deriv2():
    vmat_dm = cp.zeros((natm, 3, 3, nao))
    ipip = cp.zeros((3, 3, nao, nao))
    for ao_mask, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, 2,
                                                  None):
        nsub = len(mask)
        ng = len(weight)
        with Stage('expand ao to nao'):
            ao = contract('nip,ij->njp', ao_mask, coeff[mask])
        with Stage('eval_rho2'):
            rho = numint.eval_rho2(_sorted_mol, ao[:10], mo_coeff, mo_occ,
                                   mask, xctype)
        with Stage('eval_xc_eff'):
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype,
                                      spin=0)[1:3]
        with Stage('ipip (gga+tau)'):
            wv1 = cp.multiply(weight, vxc, out=vxc)
            wf = cp.multiply(weight, fxc, out=fxc)
            wv1[0] *= .5
            wv1[4] *= .25
            aow = rks_grad._make_dR_dao_w(ao, wv1)
            hrks._d1d2_dot_(ipip, mol, aow, ao[1:4], mask, None, False)
            aow = cp.empty((6, nao, ng))
            for i in range(4, 10):
                numint._scale_ao(ao[i], wv1[4], out=aow[i-4])
            hrks._d1d2_dot_(ipip, mol, [aow[0], aow[1], aow[2]],
                            [ao[XX], ao[XY], ao[XZ]], mask, None, False)
            hrks._d1d2_dot_(ipip, mol, [aow[1], aow[3], aow[4]],
                            [ao[YX], ao[YY], ao[YZ]], mask, None, False)
            hrks._d1d2_dot_(ipip, mol, [aow[2], aow[4], aow[5]],
                            [ao[ZX], ao[ZY], ao[ZZ]], mask, None, False)
        with Stage('ao_dm0, masks'):
            dm_mask = take_last2d(dm0_sorted, mask)
            ao_dm0 = contract('ki,nkl->nil', dm0, ao[:4])
            ao_dm_mask = contract('nig,ij->njg', ao_mask[:4], dm_mask)
        aow = cp.empty((3, nsub, ng))
        for ia in range(natm):
            with Stage('atom: dR_rho1 + fxc'):
                dR_rho1 = hrks._make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv = contract('xyg,sxg->syg', wf, dR_rho1)
                wv[:, 0] *= .5
                wv[:, 4] *= .5
            with Stage('atom: GGA terms'):
                tmp = cp.empty((3, 3, nsub))
                for i in range(3):
                    aow = rks_grad._make_dR_dao_w(ao_mask, wv[i], out=aow)
                    contract('xjg,jg->xj', aow, ao_dm_mask[0], out=tmp[i])
                for i in range(3):
                    numint._scale_ao(ao_dm_mask[:4], wv[i, :4], out=aow[i])
                contract('yjg,xjg->xyj', ao_mask[1:4], aow, beta=1, out=tmp)
            with Stage('atom: tau terms'):
                for k, trip in ((1, (XX, XY, XZ)), (2, (YX, YY, YZ)),
                                (3, (ZX, ZY, ZZ))):
                    for i in range(3):
                        numint._scale_ao(ao_dm_mask[k], wv[i, 4], out=aow[i])
                    for c, j in enumerate(trip):
                        contract('jg,xjg->xj', ao_mask[j], aow, beta=1,
                                 out=tmp[:, c])
            with Stage('atom: scatter'):
                vmat_dm[ia][:, :, mask] += tmp


t_tot = timed(deriv2)[0]
report('_get_vxc_deriv2, MGGA branch', t_tot)
print('\n' + json.dumps({'tag': f'{a.xyz.rsplit("/",1)[-1]}/{a.basis}/{a.xc}',
                         'natm': natm, 'nao': int(nao), 'nocc': int(nocc),
                         'whole': whole, 'libxc_one': t_one,
                         'libxc_perblock': t_blk, 'eval_ao': t_ao,
                         'deriv1': d1, 'deriv2': dict(T)}))
