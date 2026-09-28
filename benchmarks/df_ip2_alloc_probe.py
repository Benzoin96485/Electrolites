"""What a large ``cudaMalloc`` costs right after a large free (W34).

Two measurements, `docs/README_fastdfhessip5.md` §2:

``--malloc GB[,GB...]``: no chemistry.  For each size, allocate it, write it,
free it, wait ``--waits`` seconds and allocate it again, timing both
allocations -- the driver's reclaim of freed memory, which is what the ip2
call's "D3 buffers" row was.

``--xyz ...``: the ip2 term (``_jk_energy_per_atom``) called again and again
on one converged DF-SCF with ``FASTDFHESSIP_TIME=1``, sleeping
``--sleeps`` seconds between the calls, so that the 'D3 alloc' row can be
read against the time since the previous call freed its buffers.  Set
``FASTDFHESSIP_WORK=upstream`` for upstream's buffer.

    python benchmarks/df_ip2_alloc_probe.py --malloc 8,37 --waits 0,0.5,1,2
    FASTDFHESSIP_WORK=upstream python benchmarks/df_ip2_alloc_probe.py \\
        --xyz benchmarks/molecules/penicillinV.xyz --basis def2-svp \\
        --xc b3lyp --sleeps 0,0.5,1,2
"""
import os
import sys
import time

import cupy as cp

_sync = cp.cuda.Stream.null.synchronize


def _ms(f):
    _sync()
    t = time.perf_counter()
    r = f()
    _sync()
    return r, (time.perf_counter() - t) * 1e3


def malloc_probe(sizes, waits, reps=3):
    rt = cp.cuda.runtime
    for gb in sizes:
        nb = int(gb * 1e9)
        for w in waits:
            row = []
            for _ in range(reps):
                p, t1 = _ms(lambda: rt.malloc(nb))
                rt.memset(p, 1, nb)
                _sync()
                rt.free(p)
                time.sleep(w)
                p, t2 = _ms(lambda: rt.malloc(nb))
                rt.free(p)
                time.sleep(w)
                row.append(f'{t1:.0f}/{t2:.0f}')
            print(f'{gb:6.1f} GB  wait {w:4.1f} s   malloc ms (first/second): '
                  + '  '.join(row), flush=True)


def call_probe(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import apply_patch, build_mol
    apply_patch('all')
    from gpu4pyscf import dft
    from gpu4pyscf.df.hessian import rhf as H
    mol = build_mol(a)
    mf = dft.RKS(mol, xc=a.xc).density_fit()
    mf.verbose = 0
    mf.kernel()
    if mf.with_df.intopt is None:
        mf.with_df.build(build_cderi=False)
    dm0 = mf.make_rdm1()
    hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc)[2]
    os.environ['FASTDFHESSIP_TIME'] = '1'
    H._jk_energy_per_atom(mf.with_df.intopt, dm0, 1., hyb)   # warm
    for s in [float(x) for x in a.sleeps.split(',')] * 2:
        _sync()
        time.sleep(s)
        print(f'@@ sleep {s} s', flush=True)
        H._jk_energy_per_atom(mf.with_df.intopt, dm0, 1., hyb)


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _common import parser
    ap = parser(__doc__.splitlines()[0])
    ap.set_defaults(xyz='')
    for act in ap._actions:
        if act.dest == 'xyz':
            act.required = False
    ap.add_argument('--malloc', default='')
    ap.add_argument('--waits', default='0,0.5,1,2')
    ap.add_argument('--sleeps', default='0,0.5,1,2')
    a = ap.parse_args()
    if a.malloc:
        malloc_probe([float(x) for x in a.malloc.split(',')],
                     [float(x) for x in a.waits.split(',')])
    if a.xyz:
        call_probe(a)
