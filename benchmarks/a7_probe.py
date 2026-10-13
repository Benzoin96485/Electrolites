"""A7: how much would fusing the two exchange ranges in the threadIdx.y
("k2"/khigh) classes save?  Upper bound, in one process.

    python benchmarks/a7_probe.py --xyz benchmarks/molecules/taxol.xyz \
        --basis def2-svp --xc wb97m-v

After a few SCF cycles, times the range-separated ``get_k`` (``fastk.get_k_rsh``)
with the gen_khigh classes run one pass per operator (``_FUSED_KH_OFF``) and
in one launch, in one process, and compares the two K matrices.  (The upper
bound quoted in §13.31 came from an earlier version of this script that
dropped the long-range pass of those classes instead.)
"""
import time
import cupy as cp
from _common import apply_patch, build_mol, parser

ap = parser(__doc__.splitlines()[0])
a = ap.parse_args()
apply_patch('all')
from electrolites import fastk                              # noqa: E402
from gpu4pyscf import dft                                   # noqa: E402

mol = build_mol(a)
mf = (dft.UKS if mol.spin else dft.RKS)(mol, xc=a.xc)
mf.verbose = 0
mf.max_cycle = a.cycles or 6
mf.kernel()
dm = cp.asarray(mf.make_rdm1())
omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
sync = cp.cuda.Device().synchronize
res = {}
vks = {}
for rep in range(a.reps + 1):
    for off in (True, False):
        fastk._FUSED_KH_OFF = off
        sync(); t = time.perf_counter()
        vk = mf.get_k(mol, dm, 1, omega, alpha, hyb)
        sync(); dt = time.perf_counter() - t
        if rep:
            res.setdefault(off, []).append(dt)
            vks[off] = cp.asnumpy(vk)
t0, t1 = min(res[True]), min(res[False])
d = abs(vks[True] - vks[False]).max()
print(f'{a.xyz.rsplit("/",1)[-1]}/{a.basis}/{a.xc} spin={mol.spin} nao={mol.nao}: '
      f'RSH get_k two-pass high-l {t0:.4f} s, one-pass {t1:.4f} s '
      f'({t0/t1:.3f}x)  max|dK| {d:.1e}', flush=True)
