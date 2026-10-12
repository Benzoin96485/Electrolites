"""C12: a range-separated hybrid's response J + K, two arrangements.

`scf/_response_functions.py` builds, per call,

    vj, vk = mf.get_jk(dm1)          # fastjk: J and full-range K, one pass
    vk *= hyb
    vk += mf.get_k(dm1, omega) * (alpha - hyb)   # fastk: long-range K

The alternative timed here is `mf.get_j` (fastj's McMurchie-Davidson J) plus
`fastk.get_k_rsh` (both exchange ranges in one pass over the quartets, now
also at hermi = 0).  Same converged density, same perturbed densities as
`response_jk_probe.py`; both arrangements checked against each other.

    python benchmarks/rsh_response_probe.py --xyz benchmarks/molecules/penicillinV.xyz \
        --basis def2-svp --xc wb97m-v --nset 1,5,10 --patch all
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



from electrolites import fastk  # noqa: E402

omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
print(f'# omega {omega} alpha {alpha} hyb {hyb}', flush=True)
opt_f = mf._opt_gpu.get(0.0) or sjk._VHFOpt(mol, mf.direct_scf_tol).build()
mf._opt_gpu[0.0] = opt_f
with mol.with_range_coulomb(omega):
    opt_l = mf._opt_gpu.get(omega) or sjk._VHFOpt(mol, mf.direct_scf_tol).build()
mf._opt_gpu[omega] = opt_l


def current(dm1, hermi):
    vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
    vk *= hyb
    vk += mf.get_k(mol, dm1, hermi, omega) * (alpha - hyb)
    return vj, vk


def fused(dm1, hermi):
    vj = mf.get_j(mol, dm1, hermi=hermi)
    dms = opt_f.apply_coeff_C_mat_CT(dm1.reshape(-1, mol.nao, mol.nao))
    vk = fastk.get_k_rsh(opt_f, opt_l, cp.asarray(dms, order='C'), hermi,
                         hyb, alpha - hyb)
    return vj, opt_f.apply_coeff_CT_mat_C(vk).reshape(dm1.shape)


def parts(dm1, hermi):
    out = {}
    out['get_jk'] = timeit(lambda: mf.get_jk(mol, dm1, hermi=hermi), a.reps)[0]
    out['get_k lr'] = timeit(lambda: mf.get_k(mol, dm1, hermi, omega), a.reps)[0]
    out['get_j'] = timeit(lambda: mf.get_j(mol, dm1, hermi=hermi), a.reps)[0]
    dms = cp.asarray(opt_f.apply_coeff_C_mat_CT(
        dm1.reshape(-1, mol.nao, mol.nao)), order='C')
    out['get_k_rsh'] = timeit(lambda: fastk.get_k_rsh(
        opt_f, opt_l, dms, hermi, hyb, alpha - hyb), a.reps)[0]
    return out


rows = []
for hermi in [int(x) for x in a.hermi.split(',')]:
    for nset in [int(x) for x in a.nset.split(',')]:
        dm1 = make_dms(nset, hermi)
        t0, (vj0, vk0) = timeit(lambda: current(dm1, hermi), a.reps)
        t1, (vj1, vk1) = timeit(lambda: fused(dm1, hermi), a.reps)
        p = parts(dm1, hermi)
        dj = float(cp.abs(vj0 - vj1).max())
        dk = float(cp.abs(vk0 - vk1).max())
        print(f'hermi {hermi} nset {nset:3d}: current {t0:.4f} s, '
              f'J + fused K {t1:.4f} s ({t0/t1:.2f}x)  max|dJ| {dj:.1e} '
              f'max|dK| {dk:.1e}  parts ' +
              ' '.join(f'{k} {v:.4f}' for k, v in p.items()), flush=True)
        rows.append(dict(hermi=hermi, nset=nset, current=t0, fused=t1,
                         dj=dj, dk=dk, **p))
print('RESULT ' + json.dumps(dict(tag=tag, rows=rows)))
