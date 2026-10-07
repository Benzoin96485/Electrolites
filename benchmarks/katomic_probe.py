"""Where the per-density exchange contraction `C` goes (catalogue B12 (a)).

``ncu`` is unavailable on this machine, so the atomic traffic is measured by
ablation instead: the ``fastk`` kernels are recompiled with every
double-precision ``atomicAdd`` replaced, and ``T = base + n_dm*C`` is fitted
for each variant on the same density stack as ``benchmarks/ndm_k.py``.

    KPROBE=atomic   the shipped kernels
    KPROBE=none     the value is computed (every density read and FMA kept)
                    but not written -- a store guarded by an impossible value
    KPROBE=plain    a plain, racy ``*p += v``: the memory traffic of the
                    read-modify-write without the atomic

    KPROBE=nok      only the writes into a ``k`` column (K_ik, K_jk) dropped:
                    the ceiling of aggregating them over tasks sharing ``k``

``atomic - none`` is the cost of the writes, ``atomic - plain`` the cost of
their being atomic.  The results of the ablated variants are wrong by design.
"""
import os
import time

import cupy as cp
import numpy as np

from _common import build_mol, parser

ap = parser(__doc__.splitlines()[0])
ap.add_argument('--ndm', default='1,8,32,64')
a = ap.parse_args()
MODE = os.environ.get('KPROBE', 'atomic')

_PRE = {
    'atomic': '',
    'none': r'''
template <class T> __device__ __forceinline__ T kp_add(T *p, T v)
{ return atomicAdd(p, v); }
__device__ __forceinline__ double kp_add(double *p, double v)
{ if (v == 1.2345e300) *p = v; return 0.; }
#define atomicAdd kp_add
''',
    'plain': r'''
template <class T> __device__ __forceinline__ T kp_add(T *p, T v)
{ return atomicAdd(p, v); }
__device__ __forceinline__ double kp_add(double *p, double v)
{ double o = *p; *p = o + v; return o; }
#define atomicAdd kp_add
''',
    'nok1': '',
    'headonly': '',
    'kaggstat': '',
    'nol': r'''
__device__ __forceinline__ double kp_drop(double *p, double v)
{ if (v == 1.2345e300) *p = v; return 0.; }
''',
    'nok': r'''
__device__ __forceinline__ double kp_drop(double *p, double v)
{ if (v == 1.2345e300) *p = v; return 0.; }
''',
    'kagg': r'''
__device__ __forceinline__ void kp_agg(double *p, double v)
{
    unsigned mask = __activemask();
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    unsigned g = __match_any_sync(mask, (unsigned long long)p);
    unsigned n = __reduce_max_sync(mask, __popc(g));
    if (n == 1) { atomicAdd(p, v); return; }
    unsigned lead = __ffs(g) - 1;
    unsigned run = g >> lead;
    bool contig = (run & (run + 1)) == 0;
    unsigned mine = g >> lane;
    for (unsigned off = 1; off < n; off <<= 1) {
        double o = __shfl_down_sync(mask, v, off);
        if (contig && ((mine >> off) & 1u)) v += o;
    }
    if (!contig || lane == lead) atomicAdd(p, v);
}
''',
    'kstat': r'''
__device__ unsigned long long kp_n[4];
__device__ __forceinline__ void kp_count(double *p, double v)
{
    unsigned mask = __activemask();
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    unsigned g = __match_any_sync(mask, (unsigned long long)p);
    unsigned lead = __ffs(g) - 1;
    unsigned run = g >> lead;
    if (lane == lead) {
        atomicAdd(kp_n + 0, (unsigned long long)__popc(g));   // writes
        atomicAdd(kp_n + 1, 1ull);                            // addresses
        atomicAdd(kp_n + 2, (unsigned long long)((run & (run + 1)) == 0));
    }
    if (lane == __ffs(mask) - 1)
        atomicAdd(kp_n + 3, (unsigned long long)__popc(mask));  // active
    atomicAdd(p, v);
}
''',
    'stride': r'''
__device__ __forceinline__ int kp_gcd(int a, int b)
{ while (b) { int t = a % b; a = b; b = t; } return a; }
__device__ __forceinline__ int kp_perm(int t, int n)
{
    // tasks in flight together (t .. t+blockDim) land S apart in the queue
    int S = n / blockDim.x;
    if (S < 2) return t;
    S |= 1;
    while (kp_gcd(S, n) != 1) S += 2;
    return (int)(((long long)t * S) % n);
}
''',
    'perm': r'''
__device__ __forceinline__ int kp_perm(int t, int n)
{
    int c = t >> 10;
    if (((c + 1) << 10) > n) return t;
    return (c << 10) | ((t & 31) << 5) | ((t >> 5) & 31);
}
''',
    'scat': r'''
__device__ __forceinline__ void kp_scat(double *base, size_t total,
                                        double *p, double v)
{
    size_t off = p - base;
    off = (off + (size_t)(blockIdx.x * 256 + threadIdx.x) * 8191) % total;
    atomicAdd(base + off, v);
}
''',
}[MODE]

import electrolites._ksplit as KS                            # noqa: E402
_src = KS.source
import re                                                    # noqa: E402
_KCOL = re.compile(r'atomicAdd\((vk ?\+[^;]*\((?:k0|kk)\+)')
_LCOL = re.compile(r'atomicAdd\((vk ?\+[^;]*\((?:l0|llx)\+)')


def _patched(*x, **k):
    s = _src(*x, **k)
    if MODE == 'nok':          # only the writes to a ket column k
        s = _KCOL.sub(r'kp_drop(\1', s)
    if MODE in ('perm', 'stride'):         # a warp's lanes take tasks 32 apart
        s = s.replace('bas_kl_idx[task_id]', 'bas_kl_idx[kp_perm(task_id, ntasks)]')
    if MODE == 'nol':
        s = _LCOL.sub(r'kp_drop(\1', s)
    if MODE == 'scat':         # same count of k-column writes, scattered
        s = _KCOL.sub(r'kp_scat(vk_all, (size_t)n_dm*nao*nao, \1', s)
    if MODE == 'kstat':
        s = (_LCOL if os.environ.get('KSTAT_L') else _KCOL).sub(
            r'kp_count(\1', s)
    if MODE == 'kagg':         # warp-aggregated writes into a k column
        s = _KCOL.sub(r'kp_agg(\1', s)
    return _PRE + s


KS.source = _patched
_MODS = []
_cpRM = cp.RawModule


def _RM(*x, **k):
    m = _cpRM(*x, **k)
    _MODS.append(m)
    return m


cp.RawModule = _RM
from gpu4pyscf import scf as gscf                            # noqa: E402
from gpu4pyscf.scf import jk as JK                           # noqa: E402
import electrolites.fastk as fastk                           # noqa: E402
if MODE == 'kaggstat':
    fastk._KAGG_SRC = fastk._KAGG_SRC.replace(
        'struct KRun', '__device__ unsigned long long kp_n[4];\nstruct KRun'
    ).replace('    if (r.maxlen > 1) {', '''    if (r.lane == __ffs(r.mask) - 1) {
        atomicAdd(kp_n + 0, (unsigned long long)__popc(r.mask));
        atomicAdd(kp_n + 2, (unsigned long long)r.maxlen);
        atomicAdd(kp_n + 3, 1ull);
    }
    if (r.head) atomicAdd(kp_n + 1, 1ull);
    if (r.maxlen > 1) {''')
if MODE == 'headonly':     # the reduced atomic count without the shuffles
    fastk._KAGG_SRC = fastk._KAGG_SRC.replace(
        'double o = __shfl_down_sync(r.mask, v, off);',
        'double o = 0.;')
if MODE == 'nok1':     # the ket-column writes of the 1-thread family only
    fastk._KAGG_SRC = r'''
struct KRun { int x; };
__device__ __forceinline__ KRun krun_init(int key) { return KRun{key}; }
__device__ __forceinline__ void krun_add(const KRun &r, double *p, double v)
{ if (v == 1.2345e300) *p = v; }
'''
from electrolites.compat import NEW_JK_ABI                   # noqa: E402
_RSH = (0, 0.0, 1.0, 1.0) if NEW_JK_ABI else (0,)
_sync = cp.cuda.Stream.null.synchronize

mol = build_mol(a)
mf = gscf.RHF(mol)
mf.direct_scf_tol = 1e-13
dm = mf.get_init_guess(mol, 'minao')
if a.cycles:
    mf.max_cycle = a.cycles
    mf.kernel(dm0=dm)
    dm = mf.make_rdm1()
dm = cp.asarray(dm).reshape(mol.nao, mol.nao)
opt = JK._VHFOpt(mol, mf.direct_scf_tol).build()
sdm = opt.apply_coeff_C_mat_CT(dm.reshape(1, mol.nao, mol.nao))[0]
print(f'# {a.xyz.rsplit("/", 1)[-1]}/{a.basis} nao={mol.nao} KPROBE={MODE}')
ns, ts = [], []
for n in [int(x) for x in a.ndm.split(',')]:
    dms = cp.repeat(sdm[None], n, axis=0)
    fastk.get_k(opt, dms.copy(), 1, *_RSH)
    best = 1e9
    for _ in range(a.reps):
        x = dms.copy()
        _sync()
        t = time.perf_counter()
        fastk.get_k(opt, x, 1, *_RSH)
        _sync()
        best = min(best, time.perf_counter() - t)
    print(f'{n:5d} {best:10.4f}', flush=True)
    ns.append(n)
    ts.append(best)
if MODE == 'kaggstat':
    for mod in _MODS:
        try:
            n = cp.ndarray((4,), cp.uint64, mod.get_global('kp_n')).get()
        except Exception:
            continue
        print('kaggstat lanes %d heads %d -> %.2f per write; mean maxlen %.2f'
              % (n[0], n[1], n[0] / max(n[1], 1), n[2] / max(n[3], 1)))
if MODE == 'kstat':
    for mod in _MODS:
        n = cp.ndarray((4,), cp.uint64, mod.get_global('kp_n')).get()
        print('kstat writes %d addresses %d contiguous %d active %d '
              '-> %.2f writes per address' % (*n, n[0] / max(n[1], 1)))
A = np.stack([np.ones(len(ns)), np.asarray(ns, float)], axis=1)
(b, c), *_ = np.linalg.lstsq(A, np.asarray(ts), rcond=None)
print(f'FIT {MODE} base {b:.4f} s  C {c:.5f} s/density')
