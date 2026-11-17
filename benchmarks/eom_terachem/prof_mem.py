import sys
sys.argv = sys.argv[:2] + ['5']
src = open('run_eom.py').read().split("t0 = time.perf_counter()\nfrom electrolites import eom_gpu")[0]
exec(src)
import cupy as cp
from electrolites import eom_gpu as G, fastccsd as F
GB = 1e9
pool = cp.get_default_memory_pool()
def rep(tag): print(f'{tag:28s} avail {F._avail()/GB:6.1f} GB  pool used {pool.used_bytes()/GB:6.1f} GB', flush=True)
rep('after CCSD')
big = sorted(((k, v.nbytes) for k, v in vars(mycc).items() if isinstance(v, cp.ndarray)), key=lambda x: -x[1])[:6]
print('mycc device arrays:', [(k, round(b/GB, 2)) for k, b in big])
imds = G._make_ee(G._Imds(mycc), mycc)
rep('after _make_ee (incl. W)')
big = sorted(((k, v.nbytes) for k, v in vars(imds).items() if isinstance(v, cp.ndarray)), key=lambda x: -x[1])
print('imds total %.1f GB:' % (sum(b for _, b in big)/GB), [(k, round(b/GB, 2)) for k, b in big[:10]])
print('W', getattr(mycc, '_eom_vvvv', (0, None))[1] is not None)
