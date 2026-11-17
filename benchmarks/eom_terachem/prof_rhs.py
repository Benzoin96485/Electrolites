import sys, time, collections
sys.argv = sys.argv[:2] + ['5']
src = open('run_eom.py').read().split("t0 = time.perf_counter()\nfrom electrolites import eom_gpu")[0]
exec(src)
import cupy as cp
from electrolites import eom_gpu as G, ccsd_lambda_gpu as LG
imds = LG.make_intermediates(G._Imds(mycc)); no, nv = imds.nocc, imds.nvir
l1 = cp.random.rand(no, nv) * 1e-2; l2 = cp.random.rand(no, no, nv, nv) * 1e-3; l2 = l2 + l2.transpose(1, 0, 3, 2)
LG.lambda_rhs(imds, mycc, l1, l2)
T = collections.defaultdict(float); real = cp.einsum
def ein(s, *a):
    cp.cuda.Device().synchronize(); t0 = time.perf_counter(); r = real(s, *a); cp.cuda.Device().synchronize(); T[s] += time.perf_counter() - t0; return r
LG._ein = ein
cp.cuda.Device().synchronize(); t0 = time.perf_counter(); LG.lambda_rhs(imds, mycc, l1, l2); cp.cuda.Device().synchronize()
print(f'lambda_rhs 1 vector {time.perf_counter() - t0:.2f} s, einsums {sum(T.values()):.2f} s')
for s in sorted(T, key=lambda s: -T[s])[:12]: print(f'  {s:24s} {T[s]:.3f}')
