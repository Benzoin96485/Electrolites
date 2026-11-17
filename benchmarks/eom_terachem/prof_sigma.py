import sys, time, collections
sys.argv = sys.argv[:2] + ['5']
src = open('run_eom.py').read().split("t0 = time.perf_counter()\nfrom electrolites import eom_gpu")[0]
exec(src)
import cupy as cp
from electrolites import eom_gpu as G
imds = G._make_ee(G._Imds(mycc), mycc)
no, nv = imds.nocc, imds.nvir
h1, h2 = G._ee_diag(imds, mycc); d = G._pack_singlet(h1[None], h2[None])[0]; h1 = h2 = None
blk = G._ee_blk(no, nv); print('blk', blk)
idx = cp.argsort(d)[:blk]; X = cp.zeros((blk, len(d))); X[cp.arange(blk), idx] = 1; X += 1e-3 * cp.random.rand(*X.shape)
T = collections.defaultdict(float); N = collections.defaultdict(int)
real = cp.einsum
def ein(s, *a):
    cp.cuda.Device().synchronize(); t0 = time.perf_counter(); r = real(s, *a); cp.cuda.Device().synchronize()
    T[s] += time.perf_counter() - t0; N[s] += 1; return r
r1, r2 = G._unpack_singlet(X, no, nv)
G._ee_sigma(imds, mycc, r1, r2)   # warm
G._ein = ein
cp.cuda.Device().synchronize(); t0 = time.perf_counter()
G._ee_sigma(imds, mycc, r1, r2)
cp.cuda.Device().synchronize(); tot = time.perf_counter() - t0
G._ein = real
cp.cuda.Device().synchronize(); t0 = time.perf_counter()
G._ee_ladder(imds, mycc, r2); cp.cuda.Device().synchronize(); tl = time.perf_counter() - t0
print(f'sigma({blk} vectors) {tot:.2f} s; einsum total {sum(T.values()):.2f} s; ladder alone ~{tl:.2f} s')
for s in sorted(T, key=lambda s: -T[s])[:14]:
    print(f'  {s:28s} {T[s]:7.3f} s')
