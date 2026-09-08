"""What is `hessian.rks._get_vxc_deriv1/2` spending its time on, and how much
of it is work on atoms that have no basis function on the grid block?

Catalogue entry B5 says the XC Hessian runs a Python `for ia in range(natm)`
inside `ni.block_loop`, so the launch count is natm x nblocks x 5-9.  It does,
but the launch count is only half of it: every one of those atoms pays a
`nao x ngrids_block` scale and a `(nao x blk) . (blk x nocc)` GEMM whether or
not atom `ia` has a single basis function on the block.  This script measures
both halves before anything is written:

* the AO mask per block -- `nao_sub / nao`, and how many atoms own at least
  one masked AO (`natm_local / natm`).  The product of those two fractions is
  the ceiling for the two screens together.
* the stage split of the stock inner loop, with a synchronisation point on
  each stage: the AO evaluation, the expansion of the masked AO block back to
  the full `nao`, `eval_rho2`, `eval_xc_eff` (libxc, deriv=2), and the atom
  loop.
* `eval_xc_eff` on the whole grid in one call against the same points one
  block at a time, which is the same measurement `fastxcgrad` made for the
  gradient (there: 3.2 ms against 388 ms).

    python benchmarks/vxc_hess_probe.py --xyz benchmarks/molecules/penicillinV.xyz \
        --basis def2-svp --xc b3lyp --grids 3
"""
import json
import time
from collections import OrderedDict

import numpy as np
import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
a = ap.parse_args()
apply_patch(a.patch)

from gpu4pyscf import dft                                    # noqa: E402
from gpu4pyscf.dft import numint                             # noqa: E402
from gpu4pyscf.hessian import rks as hrks                     # noqa: E402
from gpu4pyscf.grad import rks as rks_grad                    # noqa: E402
from gpu4pyscf.lib.cupy_helper import contract, take_last2d   # noqa: E402
from pyscf.gto.mole import ATOM_OF                            # noqa: E402

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


mol = build_mol(a)
mf = dft.RKS(mol, xc=a.xc)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
e = mf.kernel()
print(f'# {a.xyz.rsplit("/",1)[-1]}/{a.basis}/{a.xc}/L{a.grids}  '
      f'natm={mol.natm} nao={mol.nao} E={e:.9f}', flush=True)

ni = mf._numint
opt = ni.gdftopt
_sorted_mol = opt._sorted_mol
grids = mf.grids
xctype = ni._xc_type(mf.xc)
nao, nmo = mf.mo_coeff.shape
mo_coeff = cp.asarray(mf.mo_coeff)
mo_occ = cp.asarray(mf.mo_occ)
mocc = mo_coeff[:, mo_occ > 0]
nocc = mocc.shape[1]
natm = mol.natm
coeff = cp.asarray(opt.coeff)
dm0 = mf.make_rdm1(mo_coeff, mo_occ)
aoslices = mol.aoslice_by_atom()

# which atom owns each AO of the *sorted* mol, i.e. of the block mask
ao_dims = _sorted_mol.ao_loc[1:] - _sorted_mol.ao_loc[:-1]
i_atom_of_sorted_ao = np.repeat(_sorted_mol._bas[:, ATOM_OF], ao_dims)

ao_deriv = 1 if xctype == 'LDA' else 2
nd = (ao_deriv + 1) * (ao_deriv + 2) * (ao_deriv + 3) // 6

# ---------------------------------------------------------------- screening --
nsub, nloc, blks = [], [], []
rho_all = []
for ao_mask, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao,
                                              ao_deriv, None):
    nsub.append(len(mask))
    blks.append(len(weight))
    nloc.append(len(np.unique(i_atom_of_sorted_ao[cp.asnumpy(mask)])))
    ao1 = contract('nip,ij->njp', ao_mask, coeff[mask])
    rho_all.append(numint.eval_rho2(_sorted_mol, ao1[:1] if xctype == 'LDA'
                                    else ao1[:4], mo_coeff, mo_occ, mask,
                                    xctype))
nsub, nloc, blks = np.array(nsub), np.array(nloc), np.array(blks)
w = blks / blks.sum()          # weight each block by its grid points
print(f'# {len(blks)} blocks, {blks.sum()} grid points')
print(f'# nao_sub / nao      mean {(nsub*w).sum()/nao:6.3f}   '
      f'min {nsub.min()/nao:.3f}  max {nsub.max()/nao:.3f}')
print(f'# natm_local / natm  mean {(nloc*w).sum()/natm:6.3f}   '
      f'min {nloc.min()/natm:.3f}  max {nloc.max()/natm:.3f}')
print(f'# product (the ceiling for both screens) '
      f'{(nsub*w).sum()/nao * (nloc*w).sum()/natm:6.3f}', flush=True)

# ------------------------------------------------------------------- libxc ---
rho = cp.hstack(rho_all)
rho_all = None
_sync()
t = time.perf_counter()
ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype, spin=0)
_sync()
t_one = time.perf_counter() - t
p0 = 0
_sync()
t = time.perf_counter()
for n in blks:
    ni.eval_xc_eff(mf.xc, rho[..., p0:p0+n] if rho.ndim > 1 else rho[p0:p0+n],
                   2, xctype=xctype, spin=0)
    p0 += n
_sync()
t_blk = time.perf_counter() - t
print(f'# libxc deriv=2: one call {t_one*1e3:8.1f} ms   '
      f'{len(blks)} calls {t_blk*1e3:8.1f} ms   '
      f'({t_blk/t_one:.0f}x)', flush=True)
rho = None
cp.get_default_memory_pool().free_all_blocks()

# ------------------------------------------------ the block loop alone -------
# What the AO evaluation costs on its own -- the floor under both
# implementations, since neither the AO screen nor the atom screen nor any
# batching touches it.  It is what stops a speedup on the rest of the function
# from reaching the ratio the screening fractions above predict.
_sync()
t = time.perf_counter()
for _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None):
    pass
_sync()
t_ao = time.perf_counter() - t
print(f'# the block loop alone (eval_ao at deriv={ao_deriv}): {t_ao:.3f} s',
      flush=True)

# ------------------------------------------------- stage split of deriv1 -----
# The GGA branch of _get_vxc_deriv1_task, stage by stage.
v_ip = cp.zeros((3, nao, nao))
vmat = cp.zeros((natm, 3, nao, nocc))
_sync()
t_total = time.perf_counter()
for ao_mask, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao,
                                              ao_deriv, None):
    with Stage('expand ao to nao'):
        ao1 = contract('nip,ij->njp', ao_mask, coeff[mask])
    with Stage('eval_rho2'):
        rho = numint.eval_rho2(_sorted_mol, ao1[:4], mo_coeff, mo_occ, mask,
                               xctype)
    with Stage('eval_xc_eff'):
        vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype, spin=0)[1:3]
    with Stage('weights + v_ip'):
        wv = cp.multiply(weight, vxc, out=vxc)
        wv[0] *= .5
        wf = cp.multiply(weight, fxc, out=fxc)
        v_ip = rks_grad._gga_grad_sum_(ao1, wv, accumulate=True, out=v_ip)
    with Stage('mo, ao_dm0'):
        mo = contract('xig,ip->xpg', ao1, mocc)
        ao_dm0 = contract('ik,pil->pkl', dm0, ao1[:4])
    with Stage('atom loop'):
        for ia in range(natm):
            dR_rho1 = hrks._make_dR_rho1(ao1, ao_dm0, ia, aoslices, xctype)
            wv2 = contract('xyg,sxg->syg', wf, dR_rho1, out=dR_rho1)
            wv2[:, 0] *= .5
            aow = cp.empty((3, nao, len(weight)))
            mow = cp.empty((3, nocc, len(weight)))
            for i in range(3):
                numint._scale_ao(ao1[:4], wv2[i, :4], out=aow[i])
                numint._scale_ao(mo[:4], wv2[i, :4], out=mow[i])
            rks_grad._d1_dot_(aow, mo[0].T, beta=1.0, out=vmat[ia])
            rks_grad._d1_dot_(mow, ao1[0].T, beta=1.0, transpose=True,
                              out=vmat[ia])
    ao1 = mo = ao_dm0 = aow = mow = None
_sync()
t_total = time.perf_counter() - t_total
print(f'\n{"stage":22s} {"time/s":>9s} {"share":>7s}')
print('-' * 42)
print(f'{"TOTAL (re-run here)":22s} {t_total:9.3f} {100.0:6.1f}%')
for k, v in T.items():
    print(f'{k:22s} {v:9.3f} {100*v/t_total:6.1f}%')
print('-' * 42)
print(f'{"accounted":22s} {sum(T.values()):9.3f} '
      f'{100*sum(T.values())/t_total:6.1f}%')
print('\n' + json.dumps({'tag': f'{a.xyz.rsplit("/",1)[-1]}/{a.basis}',
                         'natm': natm, 'nao': int(nao), 'nocc': int(nocc),
                         'nblk': len(blks), 'ngrids': int(blks.sum()),
                         'nao_sub_frac': float((nsub*w).sum()/nao),
                         'natm_local_frac': float((nloc*w).sum()/natm),
                         'libxc_one': t_one, 'libxc_perblock': t_blk,
                         'block_loop_alone': t_ao,
                         'deriv1_total': t_total, 'stages': dict(T)}))
