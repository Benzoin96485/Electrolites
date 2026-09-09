"""The response path asks for J and K from one *fused* kernel.  Should it?

`scf/_response_functions.py` calls `mf.get_jk(...)`, which lands in
`scf/jk.py:401 _VHFOpt.get_jk` -- the fused `RYS_build_jk`.  The SCF does not:
`scf/hf.py:get_veff` calls `mf.get_j` (the Mc-Murchie-Davidson `j_engine`) and
`mf.get_k` (Rys) separately.  So a polarizability or a TDDFT run reaches
*neither* of the two builds this repository has rewritten -- `fastj` patches
`j_engine._VHFOpt.get_j` and `fastk` patches `jk._VHFOpt.get_k`, and the fused
kernel is a third code path.

This script times the two arrangements on one converged density with
realistic perturbed densities, and checks that they agree:

    python benchmarks/response_jk_probe.py --xyz benchmarks/molecules/penicillinV.xyz \
        --basis def2-svp --nset 1,3,5,10 [--patch all]

`hermi 0` is the TDDFT/TDA case, where the fused kernel stacks the transposed
densities and runs at 2*nset (`jk.py:443`); `hermi 1` is CPKS.
"""
import json
import time

import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--nset', default='1,3,5,10',
                help='comma separated density counts to time')
ap.add_argument('--hermi', default='0,1', help='comma separated hermi values')
a = ap.parse_args()
apply_patch(a.patch)

from gpu4pyscf import dft                                   # noqa: E402
from gpu4pyscf.scf import jk as sjk, j_engine as sje        # noqa: E402
from gpu4pyscf.lib.cupy_helper import contract              # noqa: E402

_sync = cp.cuda.Stream.null.synchronize


def timeit(fn, reps):
    fn()
    _sync()
    ts = []
    for _ in range(reps):
        _sync()
        t = time.perf_counter()
        out = fn()
        _sync()
        ts.append(time.perf_counter() - t)
    return min(ts), out


mol = build_mol(a)
tag = f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}/{a.patch or "stock"}'
print(f'# {tag}  natm={mol.natm} nao={mol.nao}', flush=True)

mf = dft.RKS(mol, xc=a.xc)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
e = mf.kernel()
print(f'# SCF {e:.9f}  converged={mf.converged}', flush=True)

mo_coeff = cp.asarray(mf.mo_coeff)
mo_occ = cp.asarray(mf.mo_occ)
mo_energy = cp.asarray(mf.mo_energy)
mocc = mo_coeff[:, mo_occ > 0]
mvir = mo_coeff[:, mo_occ == 0]
nocc, nvir = mocc.shape[1], mvir.shape[1]
de = mo_energy[mo_occ == 0][:, None] - mo_energy[mo_occ > 0][None, :]

# A realistic perturbed density: the first CPKS/Davidson guess for the dipole
# right-hand sides, X_ai = -<a|r|i> / (e_a - e_i), extended past three
# directions by the lowest single excitations, which is what a Davidson run
# starts from.
with mol.with_common_orig((0, 0, 0)):
    h1 = cp.asarray(mol.intor('int1e_r'))
h1ai = -contract('ap,dpj->daj', mvir.T, h1 @ mocc)
xs = [h1ai[i] / de for i in range(3)]
order = cp.argsort(de.ravel())
for k in range(64):
    x = cp.zeros((nvir, nocc))
    x.ravel()[order[k]] = 1.0
    xs.append(x)
xs = cp.asarray(xs)


def make_dms(nset, hermi):
    x = xs[:nset]
    dm1 = 2 * contract('nui,vi->nuv', contract('nai,ua->nui', x, mvir), mocc)
    if hermi == 1:
        dm1 = dm1 + dm1.transpose(0, 2, 1)
    return cp.asarray(dm1, order='C')


vhfopt = mf._opt_gpu.get(0.0)
if vhfopt is None:
    vhfopt = mf._opt_gpu[0.0] = sjk._VHFOpt(mol, mf.direct_scf_tol).build()
jopt = sje._VHFOpt(mol, mf.direct_scf_tol).build()

# With `fastjk` patched, `_VHFOpt.get_jk` is *our* fused kernel and
# `_get_jk_orig` is GPU4PySCF's, so both arrangements and both fused builds are
# timed in one process against one density -- which is what makes the ratio
# meaningful on a shared node (docs/RESPONSE_1.8.1.md 4b).
_stock_fused = getattr(sjk._VHFOpt, '_get_jk_orig', None)

rows = []
print(f'\n{"hermi":>5s} {"nset":>5s} {"fused/s":>9s} {"g4p fused":>9s} '
      f'{"split/s":>9s} {"..j/s":>8s} {"..k/s":>8s} {"ratio":>7s}  '
      f'{"max|dJ|":>9s} {"max|dK|":>9s}')
print('-' * 96)
for hermi in [int(x) for x in a.hermi.split(',')]:
    for nset in [int(x) for x in a.nset.split(',')]:
        if nset > len(xs):
            continue
        dm1 = make_dms(nset, hermi)
        dms = vhfopt.apply_coeff_C_mat_CT(dm1.reshape(-1, mol.nao, mol.nao))
        dms = cp.asarray(dms, order='C')

        t_fused, (vj0, vk0) = timeit(
            lambda: vhfopt.get_jk(dms, hermi, None), a.reps)
        t_stock = float('nan')
        if _stock_fused is not None:
            t_stock, (vj2, vk2) = timeit(
                lambda: _stock_fused(vhfopt, dms, hermi, None), a.reps)
        t_j, vj1 = timeit(lambda: sje.get_j(mol, dm1, hermi, jopt), a.reps)
        t_k, vk1 = timeit(
            lambda: sjk.get_k(mol, dm1, hermi, vhfopt, None, None, None),
            a.reps)
        vj0 = vhfopt.apply_coeff_CT_mat_C(vj0).reshape(dm1.shape)
        vk0 = vhfopt.apply_coeff_CT_mat_C(vk0).reshape(dm1.shape)
        dj = float(cp.abs(vj0 - vj1).max())
        dk = float(cp.abs(vk0 - vk1).max())
        t_split = t_j + t_k
        print(f'{hermi:5d} {nset:5d} {t_fused:9.4f} {t_stock:9.4f} '
              f'{t_split:9.4f} {t_j:8.4f} {t_k:8.4f} '
              f'{t_fused/t_split:6.2f}x  {dj:9.2e} {dk:9.2e}', flush=True)
        rows.append(dict(hermi=hermi, nset=nset, fused=t_fused,
                         stock_fused=t_stock, split=t_split,
                         j=t_j, k=t_k, dj=dj, dk=dk))

print('\n' + json.dumps({'tag': tag, 'natm': mol.natm, 'nao': int(mol.nao),
                         'rows': rows}))
