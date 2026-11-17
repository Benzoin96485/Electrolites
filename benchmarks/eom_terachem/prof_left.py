import sys, time, collections
sys.argv = sys.argv[:2] + ['5']
src = open('run_eom.py').read().split("t0 = time.perf_counter()\nfrom electrolites import eom_gpu")[0]
exec(src)
import cupy as cp, numpy as np
from electrolites import eom_gpu as G, ccsd_lambda_gpu as LG
from pyscf.lib import logger
eom = G.EOMEESinglet(mycc); eom.conv_tol = 1e-7
def tm(f, *a, **k):
    cp.cuda.Device().synchronize(); t0 = time.perf_counter(); r = f(*a, **k); cp.cuda.Device().synchronize(); return r, time.perf_counter() - t0
nvec = collections.Counter()
rs = LG.lambda_rhs; TT = collections.defaultdict(float)
def rhs(*a):
    r, t = tm(rs, *a); TT['lambda_rhs'] += t; nvec['lambda_rhs'] += 1; return r
LG.lambda_rhs = rhs
es = G._ee_sigma
def sig(imds, mycc, r1, r2):
    r, t = tm(es, imds, mycc, r1, r2); TT['ee_sigma'] += t; nvec['ee_sigma vectors'] += len(r1); return r
G._ee_sigma = sig
(e, rv), t_r = tm(eom.kernel, nroots=5)
_, t_l0 = tm(LG.kernel, mycc, tol=1e-6)
TT['lambda_rhs'] = 0; nvec['lambda_rhs'] = 0
eom.verbose = 5
(conv, el, lv), t_l = tm(G._left_kernel, eom, 5, guess=rv)
print(f'right {t_r:.1f} s ({nvec["ee_sigma vectors"]} sigma vectors, {TT["ee_sigma"]:.1f} s)  Lambda {t_l0:.1f} s')
print(f'left {t_l:.1f} s ({nvec["lambda_rhs"]} lambda_rhs calls incl. 1 for eta, {TT["lambda_rhs"]:.1f} s, '
      f'{TT["lambda_rhs"]/max(1,nvec["lambda_rhs"]):.2f} s each)')
