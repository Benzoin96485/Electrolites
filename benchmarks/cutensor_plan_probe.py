"""What cuTENSOR refuses to plan, measured without allocating anything.

``docs/DF_WORK_LIST.md`` **W21**.  A plan depends on shapes and strides and
not on the data, so every question below is asked of a descriptor whose data
pointer is a one-kilobyte buffer (``electrolites/_cuplan.py``): a 37 GB
output costs the same ~0.3 ms to ask about as a 37 kB one, and thousands of
extents can be walked in a second.  ``docs/README_fastdfplan.md`` is what it
found.

    # which extents of the gradient's ip1 contraction are refused
    python benchmarks/cutensor_plan_probe.py --scan 'iqr,pi->pqr' \\
        --shape 'i=226,q=1185,p=1185' --mode r --range 1:6000

    # where the first refusal is, against the output's size, and in which dtype
    python benchmarks/cutensor_plan_probe.py --threshold

    # does the closed-form rule predict every refusal, on random shapes
    python benchmarks/cutensor_plan_probe.py --rule --trials 4000

    # does any algorithm, workspace preference or JIT mode avoid it
    python benchmarks/cutensor_plan_probe.py --options

    # which contraction of a real run was refused, with its strides
    python benchmarks/cutensor_plan_probe.py --run benchmarks/df_grad_bench.py \\
        --xyz benchmarks/molecules/taxol.xyz ...
"""
import argparse
import json
import os
import random
import runpy
import sys
import time

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('--scan', default='', help='a contraction pattern to scan')
ap.add_argument('--shape', default='',
                help="fixed extents for --scan, as 'm=n,...'")
ap.add_argument('--mode', default='', help='the mode --scan walks')
ap.add_argument('--range', default='1:4000', help='lo:hi for --scan')
ap.add_argument('--threshold', action='store_true')
ap.add_argument('--rule', action='store_true')
ap.add_argument('--trials', type=int, default=4000)
ap.add_argument('--options', action='store_true')
ap.add_argument('--run', default='',
                help='run this script (the remaining arguments are its own) '
                     'with every refused contraction printed as @@REFUSED')
a, rest = ap.parse_known_args()

import numpy as np                                            # noqa: E402
import cupy as cp                                             # noqa: E402

if a.run:
    # installed before the script imports anything, so modules that bind
    # `contract` at import time bind this one
    import gpu4pyscf.lib.cutensor as G
    import gpu4pyscf.lib.cupy_helper as H
    _orig = G.contraction

    def contraction(pattern, x, y, alpha, beta, out=None, **kw):
        try:
            return _orig(pattern, x, y, alpha, beta, out=out, **kw)
        except Exception as exc:                              # noqa: BLE001
            rec = {'pattern': pattern,
                   'a': [x.shape, x.strides], 'b': [y.shape, y.strides],
                   'c': None if out is None else [out.shape, out.strides],
                   'err': f'{type(exc).__name__}: {exc}'}
            print('@@REFUSED ' + json.dumps(rec, default=str), flush=True)
            raise

    def contract(pattern, x, y, alpha=1.0, beta=0.0, out=None):
        return contraction(pattern, x, y, alpha, beta, out=out)
    G.contraction = contraction
    G.contract = contract
    H.contract = contract
    sys.argv = [a.run] + rest
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.run)))
    runpy.run_path(a.run, run_name='__main__')
    sys.exit(0)

from electrolites import _cuplan                              # noqa: E402
from electrolites._cuplan import planable                     # noqa: E402

LIMIT = _cuplan.OUTPUT_ELEMENT_LIMIT
print(f'# cuTENSOR {_cuplan._cb.get_version()}  CUDA runtime '
      f'{cp.cuda.runtime.runtimeGetVersion()}  '
      f'{cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}')


def factor(n):
    out, p = {}, 2
    while p * p <= n:
        while n % p == 0:
            out[p] = out.get(p, 0) + 1
            n //= p
        p += 1
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def smooth_part(n, k=13):
    s = 1
    for p, e in factor(n).items():
        if p <= k:
            s *= p ** e
    return s


def divisors(n):
    ds = [1]
    for p, e in factor(n).items():
        ds = [d * p ** i for d in ds for i in range(e + 1)]
    return ds


def primes(lo, hi):
    return [n for n in range(max(lo, 2), hi) if len(factor(n)) == 1
            and list(factor(n).values()) == [1]]


def split(pattern):
    sa, rest = pattern.replace(' ', '').split(',')
    sb, sc = rest.split('->')
    return sa, sb, sc


def shapes_of(pattern, ext):
    return [tuple(ext[m] for m in s) for s in split(pattern)]


def rule(pattern, ext, k=13, limit=LIMIT):
    """The closed form: refused iff the output has more than ``limit``
    elements and the cuttable mode's ``k``-smooth part cannot bring it under.
    The cuttable mode is the last mode of the first operand that is also an
    output mode."""
    sa, _, sc = split(pattern)
    n_out = int(np.prod([ext[m] for m in sc], dtype=np.int64))
    if n_out <= limit:
        return True
    m = [x for x in sa if x in sc][-1]
    n = ext[m]
    return any(n_out // d <= limit for d in divisors(smooth_part(n, k)))


results = {}

if a.scan:
    ext = {kv.split('=')[0]: int(kv.split('=')[1])
           for kv in a.shape.split(',') if kv}
    lo, hi = (int(x) for x in a.range.split(':'))
    t0 = time.perf_counter()
    refused = []
    for n in range(lo, hi):
        ext[a.mode] = n
        if not planable(a.scan, *shapes_of(a.scan, ext)):
            refused.append(n)
    dt = time.perf_counter() - t0
    miss = [n for n in range(lo, hi)
            if rule(a.scan, {**ext, a.mode: n}) != (n not in refused)]
    print(f'## {a.scan}  {a.shape}  {a.mode} in [{lo}, {hi}): '
          f'{len(refused)} refused, {dt:.2f} s for {hi - lo} queries')
    print(f'   first refused {refused[:12]}')
    print('   rule mismatches: %d %s' % (len(miss), miss[:12]))
    for n in refused[:6]:
        print(f'   {n} = {factor(n)}')
    results['scan'] = {'pattern': a.scan, 'shape': a.shape, 'mode': a.mode,
                       'range': [lo, hi], 'refused': refused,
                       'rule_mismatches': miss}

if a.threshold:
    print('## the first refused prime extent, against the output size')
    print(f'{"pattern":>18s} {"dtype":>10s} {"fixed":>12s} {"prime":>7s} '
          f'{"output elements":>16s} {"log2":>8s}')
    rows = []
    ps = primes(2, 20000)
    for dtype in (np.float64, np.float32, np.complex128):
        for nao in (200, 400, 800, 1185, 1600, 2400):
            for p in ps:
                if not planable('iqr,pi->pqr', (226, nao, p), (nao, 226),
                                (nao, nao, p), dtype=dtype):
                    n_out = nao * nao * p
                    print(f'{"iqr,pi->pqr":>18s} {np.dtype(dtype).name:>10s} '
                          f'{"nao=%d" % nao:>12s} {p:7d} {n_out:16d} '
                          f'{np.log2(n_out):8.4f}')
                    rows.append((np.dtype(dtype).name, nao, p, n_out))
                    break
            if dtype is not np.float64:
                break
    # the other operand order: a large *input* is not refused
    for p in (1031, 4099):
        q = 1048583
        ok_in = planable('ik,kj->ij', (p, q), (q, 8), (p, 8))
        ok_out = planable('ik,kj->ij', (p, 8), (8, q), (p, q))
        print(f'   ik,kj->ij with i={p}, j or k={q}: input of {p * q:.3e} '
              f'elements {"accepted" if ok_in else "REFUSED"}, output of '
              f'the same size {"accepted" if ok_out else "REFUSED"}')
    results['threshold'] = rows

if a.rule:
    random.seed(1)
    pats = ['iqr,pi->pqr', 'rji,qj->iqr', 'Lpq,snqi->snpiL', 'ik,kj->ij',
            'iab,ic->abc', 'ia,ibc->abc', 'xrui,yrvi->xyuv', 'pqk,kr->pqr',
            'pqr,pi->iqr', 'qpr,pi->qir']
    n = 0
    miss = []
    per = {p: [0, 0] for p in pats}
    t0 = time.perf_counter()
    for _ in range(a.trials):
        pat = random.choice(pats)
        sa, sb, sc = split(pat)
        ext = {m: random.randint(2, 64) for m in set(sa + sb + sc)}
        if 's' in ext:
            ext['s'] = random.choice([1, 2])
        for m in 'xy':
            if m in ext:
                ext[m] = 3
        free = [m for m in sc if m not in 'sxy']
        fixed = int(np.prod([ext[m] for m in sc if m in 'sxy']))
        width = (2 ** random.uniform(27.5, 32) / fixed) ** (1 / len(free))
        for m in free:
            ext[m] = max(2, int(width * random.uniform(0.5, 2.0)))
        got = planable(pat, *shapes_of(pat, ext))
        n += 1
        per[pat][0] += 1
        per[pat][1] += not got
        if got != rule(pat, ext):
            miss.append((pat, ext, got))
    print(f'## the rule against {n} random shapes '
          f'({time.perf_counter() - t0:.1f} s), outputs of 2^27.5-2^32 '
          f'elements: {len(miss)} mismatches')
    for p, (m, r) in per.items():
        print(f'   {p:>18s}  {m:5d} shapes  {r:5d} refused')
    for m in miss[:20]:
        print('   MISMATCH', m)
    results['rule'] = {'trials': n, 'mismatches': [str(m) for m in miss],
                       'per_pattern': per}

if a.options:
    from cupyx import cutensor as ct
    from cupy_backends.cuda.libs import cutensor as cb
    import gpu4pyscf.lib.cutensor as G
    base = cp.empty(128)

    def view(shape):
        st, s = [], 8
        for x in reversed(shape):
            st.insert(0, s)
            s *= x
        return cp.ndarray(shape, np.float64, base.data, tuple(st))

    def try_plan(shapes, algo, ws_pref, jit):
        descs = [G._create_tensor_descriptor(view(s)) for s in shapes]
        modes = [ct.create_mode(*s) for s in ('iqr', 'pi', 'pqr')]
        op = ct.create_contraction(descs[0], modes[0], G.OP_IDENTITY,
                                   descs[1], modes[1], G.OP_IDENTITY,
                                   descs[2], modes[2], G.OP_IDENTITY, 0)
        pref = ct.create_plan_preference(algo=algo, jit_mode=jit)
        try:
            ws = cb.estimateWorkspaceSize(ct._get_handle().ptr, op.ptr,
                                          pref.ptr, ws_pref)
            ct.create_plan(op, pref, ws_limit=ws if ws else 1 << 34)
            return True
        except Exception:                                     # noqa: BLE001
            return False

    print('## every algorithm x workspace preference x JIT mode, on the ip1 '
          'extents W10 met')
    rows = []
    for dk in (383, 914, 1096):
        shapes = [(226, 1185, dk), (1185, 226), (1185, 1185, dk)]
        got = {}
        for algo in ('ALGO_DEFAULT', 'ALGO_DEFAULT_PATIENT', 'ALGO_GETT',
                     'ALGO_TGETT', 'ALGO_TTGT'):
            for ws in ('WORKSPACE_MIN', 'WORKSPACE_RECOMMENDED',
                       'WORKSPACE_MAX'):
                for jit in ('JIT_MODE_NONE', 'JIT_MODE_DEFAULT'):
                    got[algo, ws, jit] = try_plan(
                        shapes, getattr(cb, algo), getattr(cb, ws),
                        getattr(cb, jit))
        acc = sum(got.values())
        print(f'   r = {dk:5d}: accepted by {acc} of {len(got)} '
              f'combinations')
        rows.append((dk, acc, len(got)))
    results['options'] = rows

print('\n@@JSON ' + json.dumps(results, default=str))
