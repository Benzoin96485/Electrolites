"""Is a lower-precision contraction faster than fp64 on this GPU?  (B17 / W46)

The DF Hessian's CPHF exchange (``fastdfhess._jk_once``) is cuTENSOR
contractions at 78 % of the A100's fp64 peak.  W46 asks whether the early
Krylov cycles' images could be built in single precision.  This times one
representative contraction (``Lpq,qn->Lpn`` at taxol's ``nao``) in fp64 and in
fp32 operands with cuTENSOR's 32F, 3xTF32 and TF32 compute descriptors, and
the error of each against fp64.

    python benchmarks/tf32_contract_probe.py
"""
import time, cupy as cp, numpy as np
from gpu4pyscf.lib import cutensor as CT
from cupy_backends.cuda.libs import cutensor as B
print([n for n in dir(B) if n.startswith('COMPUTE_DESC')])
L, p, q, ni = 100, 1123, 1123, 1130
a = cp.random.rand(L, p, q); b = cp.random.rand(q, ni)
def T(f):
    f(); cp.cuda.Device().synchronize(); t = time.perf_counter()
    for _ in range(3): r = f()
    cp.cuda.Device().synchronize(); return (time.perf_counter() - t) / 3, r
flop = 2 * L * p * q * ni
t64, r64 = T(lambda: CT.contraction('Lpq,qn->Lpn', a, b, 1., 0.))
print(f'fp64 {t64*1e3:.1f} ms {flop/t64/1e12:.1f} TF')
a32 = a.astype(np.float32); b32 = b.astype(np.float32)
for name in ('COMPUTE_DESC_32F', 'COMPUTE_DESC_3xTF32', 'COMPUTE_DESC_TF32'):
    if not hasattr(B, name): continue
    cd = getattr(B, name)
    t, r = T(lambda: CT.contraction('Lpq,qn->Lpn', a32, b32, 1., 0., compute_desc=cd))
    err = float(abs(r - r64).max() / abs(r64).max())
    print(f'{name} {t*1e3:.1f} ms {flop/t/1e12:.1f} TF  rel err {err:.1e}')
