"""Profile the singlet EOM-EE step: ladder path (MO vvvv vs AO-direct), sigma
batch sizes, time per piece.  python prof_eom.py Rh/SOXYIN01_comp_0"""
import sys
import time
import collections

sys.argv = sys.argv[:2] + ['5']
src = open('run_eom.py').read().split(
    "t0 = time.perf_counter()\nfrom electrolites import eom_gpu")[0]
exec(src)

import cupy as cp
from electrolites import eom_gpu, fastccsd

T = collections.defaultdict(float)
N = collections.defaultdict(int)


def wrap(mod, name):
    fn = getattr(mod, name)

    def w(*a, **k):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        out = fn(*a, **k)
        cp.cuda.Device().synchronize()
        T[name] += time.perf_counter() - t0
        N[name] += 1
        return out
    setattr(mod, name, w)


for nm in ('_ee_sigma', '_ladder_mo', '_ladder', '_vvvv_mo', '_make_ee',
           '_ee_diag', '_unpack_singlet', '_pack_singlet', '_ee_ladder'):
    wrap(eom_gpu, nm)
blks = []
_blk = eom_gpu._ee_blk
eom_gpu._ee_blk = lambda no, nv: blks.append(_blk(no, nv)) or blks[-1]
_Imds = eom_gpu._Imds


class TImds(_Imds):
    def __init__(self, *a, **k):
        t0 = time.perf_counter()
        super().__init__(*a, **k)
        cp.cuda.Device().synchronize()
        T['_Imds'] += time.perf_counter() - t0


eom_gpu._Imds = TImds

print('after CCSD: avail %.1f GB, AO cache %s %.1f GB, nocc %d nvir %d' % (
    fastccsd._avail() / 1e9, type(getattr(mycc, '_fastccsd_eri', None)).__name__,
    sum(v.nbytes for v in (getattr(mycc, '_fastccsd_eri', None) or {}).values()) / 1e9,
    mycc.nocc, mycc.nmo - mycc.nocc), flush=True)
nv = mycc.nmo - mycc.nocc
print('MO vvvv needs 3*v^4*8 = %.1f GB vs avail/2' % (3 * nv**4 * 8 / 1e9))
eom = eom_gpu.EOMEESinglet(mycc)
eom.conv_tol = 1e-7
t0 = time.perf_counter()
e, v = eom.kernel(nroots=5)
cp.cuda.Device().synchronize()
tt = time.perf_counter() - t0
print(f'EOM total {tt:.1f} s; MO vvvv used: {getattr(mycc, "_eom_vvvv", None) is not None}'
      f'  ao_ladder flag: {getattr(mycc, "_eom_ao_ladder", False)}')
print('sigma batch sizes:', collections.Counter(blks))
for k in sorted(T, key=lambda k: -T[k]):
    print(f'  {k:18s} {N[k]:5d} calls {T[k]:8.1f} s')
