"""`nr_rks_fxc` A/B at a fixed density, both implementations in one process.

The XC response kernel is the grid half of every response property, and what
it costs depends on one number the molecule does not fix: `nset`, the number
of right-hand sides the solver hands it (three dipole directions for a
polarizability, a Davidson subspace for TDDFT, 3*natm for a Hessian's CPHF).
So it is swept.

    python benchmarks/fxc_bench.py --xyz benchmarks/molecules/penicillinV.xyz \
        --basis def2-svp --nset 1,2,3,5,10,20,40

`--mode` selects which implementations to time.  `blocked` is the shipped one;
`nokernel` keeps its batched GEMMs and puts GPU4PySCF's per-set contraction
kernels back, so the difference is what `kernels/fastxcfxc.cu` is worth; and
`grouped` is catalogue B4d/G1 as written -- `grouped_gemm` over every
(block, set) pair -- which measures slower than either.  `grouped` is LDA and
GGA only, so on a meta-GGA functional it reports GPU4PySCF's own timing; ask
for `--mode blocked,nokernel` there.

`--spin` times `nr_uks_fxc` instead, which is a different function with its
own per-set loop rather than the same one with two densities.  `grouped` has
no unrestricted arm and is dropped from the mode list there.
"""
import json
import time

import cupy as cp

from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--nset', default='1,2,3,5,10,20,40')
ap.add_argument('--hermi', default='1,0')
ap.add_argument('--mode', default='blocked,nokernel,grouped')
a = ap.parse_args()
apply_patch('fastxcfxc')

from gpu4pyscf import dft                                   # noqa: E402
from gpu4pyscf.dft import numint as NI                      # noqa: E402
from electrolites import fastxcfxc                          # noqa: E402

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
tag = (f'{a.xyz.rsplit("/", 1)[-1]}/{a.basis}/{a.xc}/L{a.grids}/'
       f'{"UKS" if mol.spin else "RKS"}')
print(f'# {tag}  natm={mol.natm} nao={mol.nao}', flush=True)

unrestricted = bool(mol.spin)
mf = (dft.UKS if unrestricted else dft.RKS)(mol, xc=a.xc)
mf.grids.level = a.grids
mf.conv_tol = a.conv_tol
e = mf.kernel()
print(f'# SCF {e:.9f}  converged={mf.converged}  '
      f'{"UKS" if unrestricted else "RKS"}', flush=True)

ni = mf._numint
ni.build(mol, mf.grids.coords)
mo, occ = cp.asarray(mf.mo_coeff), cp.asarray(mf.mo_occ)
en = cp.asarray(mf.mo_energy)
rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, a.xc, mo, occ,
                                    1 if unrestricted else 0)
with mol.with_common_orig((0, 0, 0)):
    h1 = cp.asarray(mol.intor('int1e_r'))


def _rhs(mo_s, occ_s, en_s, shift):
    """128 right-hand sides of the shape a solver produces, one channel."""
    mocc, mvir = mo_s[:, occ_s > 0], mo_s[:, occ_s == 0]
    de = en_s[occ_s == 0][:, None] - en_s[occ_s > 0][None, :]
    xs = [-(mvir.T @ h1[(i + shift) % 3] @ mocc) / de for i in range(3)]
    order = cp.argsort(de.ravel())
    for k in range(128):
        x = cp.zeros(de.shape)
        x.ravel()[order[(k + shift) % de.size]] = 1.0
        xs.append(x)
    return mocc, mvir, xs


if unrestricted:
    chans = [_rhs(mo[s], occ[s], en[s], s) for s in range(2)]
else:
    chans = [_rhs(mo, occ, en, 0)]


def _dms(nset, hermi):
    out = []
    for mocc, mvir, xs in chans:
        dm1 = cp.asarray([mvir @ x @ mocc.T for x in xs[:nset]])
        if not unrestricted:
            dm1 = 2 * dm1
        if hermi == 1:
            dm1 = dm1 + dm1.transpose(0, 2, 1)
        out.append(cp.asarray(dm1, order='C'))
    return cp.asarray(out[0] if not unrestricted else out, order='C')


if unrestricted:
    FNS = {'blocked': fastxcfxc._nr_uks_fxc_blocked,
           'nokernel': fastxcfxc._nr_uks_fxc_nokernel}
    STOCK = NI._nr_uks_fxc_orig
else:
    FNS = {'blocked': fastxcfxc._nr_rks_fxc_blocked,
           'nokernel': fastxcfxc._nr_rks_fxc_nokernel,
           'grouped': fastxcfxc._nr_rks_fxc_grouped}
    STOCK = NI._nr_rks_fxc_orig
modes = [m for m in a.mode.split(',') if m in FNS]

print(f'\n{"hermi":>5s} {"nset":>5s} {"stock/s":>9s} '
      + ' '.join(f'{m + "/s":>10s}' for m in modes) + ' '
      + ' '.join(f'{m:>8s}' for m in modes) + f' {"rel.err":>9s}')
print('-' * (32 + 19 * len(modes)))
rows = []
for hermi in [int(x) for x in a.hermi.split(',')]:
    for nset in [int(x) for x in a.nset.split(',')]:
        dm1 = _dms(nset, hermi)
        args = (ni, mol, mf.grids, a.xc, None, dm1, 0, hermi, rho0, vxc, fxc)
        t_ref, v_ref = timeit(lambda: STOCK(*args, 2000, None), a.reps)
        scale = float(cp.abs(v_ref).max()) or 1.0
        ts, errs = [], []
        for m in modes:
            t, v = timeit(lambda: FNS[m](*args, 2000, None), a.reps)
            ts.append(t)
            errs.append(float(cp.abs(cp.asarray(v) - v_ref).max()) / scale)
        print(f'{hermi:5d} {nset:5d} {t_ref:9.4f} '
              + ' '.join(f'{t:10.4f}' for t in ts) + ' '
              + ' '.join(f'{t_ref/t:7.2f}x' for t in ts)
              + f' {max(errs):9.2e}', flush=True)
        rows.append(dict(hermi=hermi, nset=nset, stock=t_ref,
                         **{m: t for m, t in zip(modes, ts)},
                         err=max(errs)))

print('\n' + json.dumps({'tag': tag, 'natm': mol.natm, 'nao': int(mol.nao),
                         'spin': int(mol.spin), 'rows': rows}))
